"""Index lifecycle and integrity operations shared by the CLI."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..index_manifest import MANIFEST_FILENAME, IndexManifest, IndexManifestError
from ..utils import get_indexes_dir
from ..vector_store import VectorStore, VectorStoreError


@dataclass(frozen=True)
class IndexHealth:
    name: str
    path: Path
    healthy: bool
    problems: tuple[str, ...]
    manifest: IndexManifest | None


def list_indexes(project_path: Path | None = None) -> list[IndexManifest]:
    root = get_indexes_dir(project_path)
    if not root.exists():
        return []
    manifests = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()):
        if (path / MANIFEST_FILENAME).exists():
            manifests.append(IndexManifest.load(path))
    return manifests


def get_index_info(name: str, project_path: Path | None = None) -> IndexManifest:
    path = get_indexes_dir(project_path) / name
    if not path.is_dir():
        raise IndexManifestError(f"Index '{name}' does not exist at {path}")
    return IndexManifest.load(path)


def doctor_index(name: str, project_path: Path | None = None) -> IndexHealth:
    path = get_indexes_dir(project_path) / name
    problems: list[str] = []
    manifest = None
    try:
        manifest = get_index_info(name, project_path)
        stats = VectorStore(path, name).get_stats()
        if stats["total_chunks"] != manifest.chunk_count:
            problems.append(f"manifest has {manifest.chunk_count} chunks but storage has {stats['total_chunks']}")
        if manifest.file_count != len(manifest.files):
            problems.append("manifest file count does not match its file records")
        missing = [file for file in manifest.files if not Path(file).exists()]
        if missing:
            problems.append(f"repository has {len(missing)} deleted file(s); run ctxai index to update")
    except (IndexManifestError, VectorStoreError) as exc:
        problems.append(str(exc))
    return IndexHealth(name=name, path=path, healthy=not problems, problems=tuple(problems), manifest=manifest)


def delete_index(name: str, project_path: Path | None = None) -> Path:
    path = get_indexes_dir(project_path) / name
    if not path.is_dir():
        raise IndexManifestError(f"Index '{name}' does not exist at {path}")
    shutil.rmtree(path)
    return path
