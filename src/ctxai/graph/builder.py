"""Graph generation pipeline stage: extraction, incremental diffing, publication (IG-01).

The builder owns per-file ownership semantics: every node belongs to exactly
one repository-relative file, so re-indexing unchanged files performs zero
mutations, while changing or deleting a file replaces only its owned rows and
re-extracts the (deterministically discovered) dependent files whose edges
point into them. All publication happens through :class:`GraphStore`
transactions; a failure here leaves the prior graph and manifest untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .model import (
    GRAPH_SCHEMA_VERSION,
    SUPPORTED_LANGUAGES,
    GraphMetadata,
)
from .python_adapter import PythonAdapter, build_symbol_indexes, resolve_edges
from .store import GraphStore, GraphStoreError


class GraphBuildError(RuntimeError):
    """Raised when graph extraction or publication cannot complete."""


@dataclass(frozen=True)
class GraphBuildResult:
    """Outcome of one graph generation stage.

    Attributes:
        generation: Graph generation published by this run.
        metadata: The published metadata (authoritative counts included).
        mode: ``"full"``, ``"incremental"``, or ``"skipped"`` (zero mutations).
        reextracted_files: Number of files parsed in this run.
        deleted_files: Number of files whose nodes were removed.
    """

    generation: int
    metadata: GraphMetadata
    mode: str
    reextracted_files: int
    deleted_files: int


def _safe_relative(relative_path: str) -> str:
    candidate = relative_path.replace("\\", "/")
    if candidate.startswith("/") or candidate.startswith("../") or "/../" in candidate or candidate == "..":
        raise GraphBuildError(f"Refusing non repository-relative graph path: {relative_path!r}")
    return candidate


class GraphBuilder:
    """Builds and incrementally updates one index's symbol graph."""

    def __init__(self, repository_root: Path, adapter: PythonAdapter | None = None) -> None:
        """Create a builder bound to one repository.

        Args:
            repository_root: Canonical repository root (repository identity).
            adapter: Optional language adapter override (Python by default).
        """
        self.repository_root = Path(repository_root).resolve()
        self.adapter = adapter or PythonAdapter()

    def update(
        self,
        index_path: Path,
        files: set[str],
        changed: set[str],
        deleted: set[str],
        force_full: bool = False,
    ) -> GraphBuildResult:
        """Publish the graph state matching the repository's current files.

        Args:
            index_path: Canonical index directory holding ``graph.sqlite3``.
            files: All repository-relative paths currently in the index.
            changed: Repository-relative paths whose content changed.
            deleted: Repository-relative paths removed from the repository.
            force_full: Rebuild every file from scratch (first build, legacy
                manifest, or extractor upgrade).

        Returns:
            A :class:`GraphBuildResult` describing the publication.

        Raises:
            GraphBuildError: On path-safety violations or unreadable sources;
                extraction and storage failures propagate unchanged so the
                index pipeline aborts before publishing the manifest.
        """
        supported = sorted(_safe_relative(path) for path in files if self.adapter.supports_file(path))
        changed_supported = {_safe_relative(path) for path in changed if self.adapter.supports_file(path)}
        deleted_supported = {_safe_relative(path) for path in deleted if self.adapter.supports_file(path)}
        store = GraphStore(index_path)
        existing = self._existing_metadata(store)
        full = force_full or existing is None or existing.extractor_version != self.adapter.extractor_version
        built_at = datetime.now(timezone.utc).isoformat()

        if full:
            return self._full_build(store, supported, deleted_supported, existing, built_at)

        process = changed_supported | deleted_supported
        if not process:
            assert existing is not None
            return GraphBuildResult(
                generation=existing.generation,
                metadata=existing,
                mode="skipped",
                reextracted_files=0,
                deleted_files=0,
            )
        assert existing is not None  # full is False implies an existing readable store
        process |= self._dependents(store, process)
        return self._incremental_build(store, process, deleted_supported, existing, built_at)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _existing_metadata(store: GraphStore) -> GraphMetadata | None:
        if not store.exists():
            return None
        try:
            return store.read_metadata()
        except GraphStoreError:
            return None  # corrupt or unsupported store: rebuild from scratch

    def _dependents(self, store: GraphStore, process: set[str]) -> set[str]:
        """Discover files whose edges point into ``process`` files' nodes.

        Args:
            store: The live graph store.
            process: Files being replaced or deleted.

        Returns:
            Additional repository-relative paths that must be re-extracted so
            their edges are rebuilt against the new node set.
        """
        discovered: set[str] = set()
        frontier = set(process)
        while frontier:
            found = store.files_with_edges_into(frontier) - process - discovered
            discovered |= found
            frontier = found
        return discovered

    def _extract(self, relative_paths: list[str]) -> list:
        extractions = []
        for relative_path in relative_paths:
            source_path = self.repository_root / relative_path
            try:
                source = source_path.read_bytes()
            except OSError as exc:
                raise GraphBuildError(f"Cannot read {relative_path} for graph extraction: {exc}") from exc
            extractions.append(self.adapter.extract_file(relative_path, source, str(self.repository_root)))
        return extractions

    def _metadata(self, generation: int, built_at: str) -> GraphMetadata:
        return GraphMetadata(
            schema_version=GRAPH_SCHEMA_VERSION,
            extractor_version=self.adapter.extractor_version,
            resolver_version=self.adapter.resolver_version,
            supported_languages=list(SUPPORTED_LANGUAGES),
            built_at=built_at,
            generation=generation,
        )

    def _full_build(
        self,
        store: GraphStore,
        files: list[str],
        deleted: set[str],
        existing: GraphMetadata | None,
        built_at: str,
    ) -> GraphBuildResult:
        generation = (existing.generation if existing else 0) + 1
        extractions = self._extract(files)
        nodes = [node for extraction in extractions for node in extraction.nodes]
        indexes = build_symbol_indexes(nodes)
        edges = resolve_edges(extractions, indexes, str(self.repository_root))
        published = store.replace_all(nodes, edges, self._metadata(generation, built_at))
        return GraphBuildResult(
            generation=generation,
            metadata=published,
            mode="full",
            reextracted_files=len(files),
            deleted_files=len(deleted),
        )

    def _incremental_build(
        self,
        store: GraphStore,
        process: set[str],
        deleted: set[str],
        existing: GraphMetadata,
        built_at: str,
    ) -> GraphBuildResult:
        extractions = self._extract(sorted(process - deleted))
        surviving_nodes = [node for node in store.iter_nodes() if node.file_path not in process]
        fresh_nodes = [node for extraction in extractions for node in extraction.nodes]
        indexes = build_symbol_indexes([*surviving_nodes, *fresh_nodes])
        edges = resolve_edges(extractions, indexes, str(self.repository_root))
        changed_map: dict[str, tuple[list, list]] = {
            extraction.file_path: (
                extraction.nodes,
                [edge for edge in edges if edge.evidence_file == extraction.file_path],
            )
            for extraction in extractions
        }
        generation = existing.generation + 1
        published = store.update_files(changed_map, sorted(deleted), self._metadata(generation, built_at))
        return GraphBuildResult(
            generation=generation,
            metadata=published,
            mode="incremental",
            reextracted_files=len(extractions),
            deleted_files=len(deleted),
        )
