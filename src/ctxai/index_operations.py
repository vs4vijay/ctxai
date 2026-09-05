"""Application services shared by ctxai's CLI and web interfaces."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigManager
from .embeddings import EmbeddingsFactory
from .graph.operations import GraphHealth, graph_health
from .index_manifest import IndexManifest, IndexManifestError, get_repository_revision
from .utils import get_indexes_dir
from .vector_store import VectorStore, VectorStoreError

INDEX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class IndexSummary:
    name: str
    path: Path
    healthy: bool
    stale: bool
    problems: tuple[str, ...]
    manifest: IndexManifest | None
    storage_chunks: int | None
    size_bytes: int
    graph: GraphHealth | None = None


class IndexOperations:
    """Stable index operations used by user-facing adapters."""

    def __init__(self, project_path: Path | None = None) -> None:
        self.project_path = project_path
        self.indexes_dir = get_indexes_dir(project_path)

    @staticmethod
    def validate_name(name: str) -> str:
        if not INDEX_NAME.fullmatch(name):
            raise IndexManifestError(
                "Invalid index name; use letters, numbers, '.', '_' or '-' (maximum 128 characters)."
            )
        return name

    def path_for(self, name: str) -> Path:
        return self.indexes_dir / self.validate_name(name)

    def list(self) -> list[IndexSummary]:
        if not self.indexes_dir.exists():
            return []
        return [self.inspect(path.name) for path in sorted(self.indexes_dir.iterdir()) if path.is_dir()]

    def inspect(self, name: str) -> IndexSummary:
        path = self.path_for(name)
        if not path.is_dir():
            raise IndexManifestError(f"Index '{name}' does not exist at {path}")
        problems: list[str] = []
        manifest: IndexManifest | None = None
        storage_chunks: int | None = None
        try:
            manifest = IndexManifest.load(path)
        except IndexManifestError as exc:
            problems.append(str(exc))
        try:
            storage_chunks = int(VectorStore(path, name).get_stats()["total_chunks"])
        except (VectorStoreError, KeyError, TypeError, ValueError) as exc:
            problems.append(str(exc))
        stale = False
        if manifest is not None:
            if storage_chunks is not None and storage_chunks != manifest.chunk_count:
                problems.append(f"manifest has {manifest.chunk_count} chunks but storage has {storage_chunks}")
            if manifest.file_count != len(manifest.files):
                problems.append("manifest file count does not match its file records")
            repository = Path(manifest.repository_root)
            missing = [file for file in manifest.files if not Path(file).exists()]
            current_revision = get_repository_revision(repository) if repository.exists() else None
            stale = bool(missing) or (
                manifest.repository_revision is not None
                and current_revision is not None
                and current_revision != manifest.repository_revision
            )
        # Graph health is reported separately: a missing graph is diagnostic
        # only, while mismatch/corruption/count problems surface in doctor.
        try:
            graph = graph_health(path, manifest)
        except Exception as exc:  # pragma: no cover - defensive, never block inspect
            graph = GraphHealth(status="corrupt", metadata=None, problems=(f"graph health check failed: {exc}",))
        size_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return IndexSummary(
            name=name,
            path=path,
            healthy=not problems,
            stale=stale,
            problems=tuple(problems),
            manifest=manifest,
            storage_chunks=storage_chunks,
            size_bytes=size_bytes,
            graph=graph,
        )

    def chunks(self, name: str, limit: int = 100) -> dict[str, Any]:
        summary = self.inspect(name)
        if not summary.healthy:
            raise IndexManifestError("Cannot browse an unhealthy index: " + "; ".join(summary.problems))
        return VectorStore(summary.path, name).collection.get(limit=max(1, min(limit, 100)))

    def query(self, name: str, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        summary = self.inspect(name)
        if not summary.healthy:
            raise IndexManifestError("Cannot query an unhealthy index: " + "; ".join(summary.problems))
        text = query.strip()
        if not text:
            raise ValueError("Query must not be empty")
        config = ConfigManager(self.project_path).load()
        embeddings = EmbeddingsFactory.create(config.embedding)
        manifest = summary.manifest
        assert manifest is not None
        configured_model = config.embedding.model or getattr(embeddings, "model", None) or "default"
        if (
            manifest.embedding_provider != config.embedding.provider
            or manifest.embedding_model != str(configured_model)
            or manifest.embedding_dimension != embeddings.get_dimension()
        ):
            raise ValueError("Configured embedding identity does not match this index")
        return VectorStore(summary.path, name).search(
            query_embedding=embeddings.generate_embedding(text),
            n_results=max(1, min(n_results, 20)),
        )

    def delete(self, name: str) -> Path:
        path = self.path_for(name)
        if not path.is_dir():
            raise IndexManifestError(f"Index '{name}' does not exist at {path}")
        shutil.rmtree(path)
        return path
