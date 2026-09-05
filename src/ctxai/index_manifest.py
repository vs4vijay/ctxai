"""Versioned metadata describing a durable ctxai index."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


class IndexManifestError(RuntimeError):
    """Raised when index metadata is missing, invalid, or incompatible."""


@dataclass
class IndexedFile:
    """Persistent state used to make index updates deterministic."""

    sha256: str
    chunks: int


@dataclass
class IndexManifest:
    """Identity, provenance, and health metadata for an index."""

    index_name: str
    repository_root: str
    repository_revision: str | None
    schema_version: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    created_at: str
    updated_at: str
    file_count: int
    chunk_count: int
    files: dict[str, IndexedFile] = field(default_factory=dict)
    # Graph identity fields (IG-01). All optional: manifests written before
    # the symbol graph existed load unchanged with these set to None, which
    # doctor reports as "graph data has not been built" (a diagnostic, not
    # corruption). Populated by the index workflow after the graph stage
    # succeeds, atomically with the rest of the manifest.
    graph_schema_version: int | None = None
    graph_extractor_version: str | None = None
    graph_generation: int | None = None
    graph_node_count: int | None = None
    graph_edge_count: int | None = None

    @classmethod
    def create(
        cls,
        *,
        index_name: str,
        repository_root: Path,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> IndexManifest:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            index_name=index_name,
            repository_root=str(repository_root.resolve()),
            repository_revision=get_repository_revision(repository_root),
            schema_version=SCHEMA_VERSION,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            created_at=now,
            updated_at=now,
            file_count=0,
            chunk_count=0,
        )

    @classmethod
    def load(cls, index_path: Path) -> IndexManifest:
        path = index_path / MANIFEST_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["files"] = {key: IndexedFile(**value) for key, value in data.get("files", {}).items()}
            manifest = cls(**data)
        except (OSError, ValueError, TypeError) as exc:
            raise IndexManifestError(f"Cannot read index manifest at {path}: {exc}") from exc
        if manifest.schema_version != SCHEMA_VERSION:
            raise IndexManifestError(
                f"Unsupported index schema {manifest.schema_version}; expected {SCHEMA_VERSION}. Rebuild the index."
            )
        return manifest

    @classmethod
    def load_optional(cls, index_path: Path) -> IndexManifest | None:
        if not (index_path / MANIFEST_FILENAME).exists():
            return None
        return cls.load(index_path)

    def save(self, index_path: Path) -> None:
        """Atomically publish the manifest only after storage succeeds."""
        index_path.mkdir(parents=True, exist_ok=True)
        path = index_path / MANIFEST_FILENAME
        temporary = path.with_suffix(".json.tmp")
        payload = asdict(self)
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def get_repository_revision(repository_root: Path) -> str | None:
    """Return the current Git revision when the repository is under Git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
