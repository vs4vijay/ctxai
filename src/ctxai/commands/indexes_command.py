"""Index lifecycle and integrity operations shared by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..index_manifest import MANIFEST_FILENAME, IndexManifest, IndexManifestError
from ..index_operations import IndexOperations
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
    return [summary.manifest for summary in IndexOperations(project_path).list() if summary.manifest]


def get_index_info(name: str, project_path: Path | None = None) -> IndexManifest:
    summary = IndexOperations(project_path).inspect(name)
    if summary.manifest is None:
        raise IndexManifestError("; ".join(summary.problems))
    return summary.manifest


def doctor_index(name: str, project_path: Path | None = None) -> IndexHealth:
    summary = IndexOperations(project_path).inspect(name)
    problems = list(summary.problems)
    if summary.stale:
        problems.append("repository revision or indexed files changed; run ctxai index to update")
    return IndexHealth(
        name=name, path=summary.path, healthy=not problems, problems=tuple(problems), manifest=summary.manifest
    )


def delete_index(name: str, project_path: Path | None = None) -> Path:
    return IndexOperations(project_path).delete(name)
