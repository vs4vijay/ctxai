"""Graph generation pipeline stage: extraction, incremental diffing, publication (IG-01/IG-02).

The builder owns per-file ownership semantics: every node belongs to exactly
one repository-relative file, so re-indexing unchanged files performs zero
mutations, while changing or deleting a file replaces only its owned rows and
re-extracts the (deterministically discovered) dependent files whose edges
point into them. Files are dispatched to language adapters through
:mod:`ctxai.graph.adapters`; unsupported languages contribute no graph nodes
and stay indexable as ordinary chunks. Each node records the language adapter
version that extracted it, so an adapter upgrade marks only the affected
files stale and triggers a bounded incremental rebuild. All publication
happens through :class:`GraphStore` transactions; a failure here leaves the
prior graph and manifest untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import get_adapter, language_for_file, resolve_extraction_edges
from .model import (
    GRAPH_SCHEMA_VERSION,
    SUPPORTED_LANGUAGES,
    GraphMetadata,
    combined_adapter_version,
)
from .resolution import FileExtraction, build_symbol_indexes
from .store import GraphStore, GraphStoreError, ModuleExport


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

    def __init__(self, repository_root: Path) -> None:
        """Create a builder bound to one repository.

        Args:
            repository_root: Canonical repository root (repository identity).
        """
        self.repository_root = Path(repository_root).resolve()
        self._adapters: dict[str, Any] = {}

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
                manifest, or pre-v2 store migration).

        Returns:
            A :class:`GraphBuildResult` describing the publication.

        Raises:
            GraphBuildError: On path-safety violations or unreadable sources;
                extraction and storage failures propagate unchanged so the
                index pipeline aborts before publishing the manifest.
        """
        supported = sorted(_safe_relative(path) for path in files if language_for_file(path) is not None)
        changed_supported = {_safe_relative(path) for path in changed if language_for_file(path) is not None}
        deleted_supported = {_safe_relative(path) for path in deleted if language_for_file(path) is not None}
        store = GraphStore(index_path)
        existing = self._existing_metadata(store)
        full = force_full or existing is None
        built_at = datetime.now(timezone.utc).isoformat()

        if full:
            return self._full_build(store, supported, deleted_supported, existing, built_at)

        process = changed_supported | deleted_supported
        process |= self._adapter_stale_files(store, supported)
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
        return self._incremental_build(store, process, deleted_supported, existing, built_at, supported)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _existing_metadata(store: GraphStore) -> GraphMetadata | None:
        if not store.exists():
            return None
        try:
            return store.read_metadata()
        except GraphStoreError:
            return None  # corrupt, v1, or unsupported store: rebuild from scratch

    def _adapter_for_language(self, language: str) -> Any | None:
        """Return the (cached) adapter for a language, or ``None`` if absent."""
        adapter = self._adapters.get(language)
        if adapter is None:
            adapter = get_adapter(language)
            if adapter is None:
                return None
            self._adapters[language] = adapter
        return adapter

    def _adapter_for(self, relative_path: str) -> Any | None:
        """Return the (cached) adapter for a file, or ``None`` if unsupported."""
        language = language_for_file(relative_path)
        if language is None:
            return None
        return self._adapter_for_language(language)

    def _index_languages(self, files: list[str]) -> set[str]:
        """Return the set of graph languages present in the index's files.

        Args:
            files: Repository-relative files currently in the index.

        Returns:
            Sorted-safe set of language names with an adapter.
        """
        return {language for language in (language_for_file(path) for path in files) if language is not None}

    def _adapter_stale_files(self, store: GraphStore, files: list[str]) -> set[str]:
        """Find files whose stored adapter version no longer matches (IG-02).

        An adapter upgrade marks only the files of that language stale so the
        rebuild stays bounded.

        Args:
            store: The live graph store.
            files: Repository-relative files currently in the index.

        Returns:
            The repository-relative paths that must be re-extracted.
        """
        identities = store.file_identities()
        stale: set[str] = set()
        for relative_path in files:
            identity = identities.get(relative_path)
            if identity is None:
                continue
            adapter = self._adapter_for(relative_path)
            if adapter is None:
                continue
            if identity.adapter_version != adapter.extractor_version:
                stale.add(relative_path)
        return stale

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

    def _extract(self, relative_paths: list[str]) -> list[FileExtraction]:
        extractions = []
        for relative_path in relative_paths:
            adapter = self._adapter_for(relative_path)
            if adapter is None:  # pragma: no cover - callers pre-filter languages
                continue
            source_path = self.repository_root / relative_path
            try:
                source = source_path.read_bytes()
            except OSError as exc:
                raise GraphBuildError(f"Cannot read {relative_path} for graph extraction: {exc}") from exc
            extractions.append(adapter.extract_file(relative_path, source, str(self.repository_root)))
        return extractions

    def _fresh_export_maps(self, extractions: list[FileExtraction]) -> tuple[dict[str, dict[str, str]], set[str]]:
        """Build export maps from fresh extractions (module to name to target).

        Args:
            extractions: Fresh extractions sorted deterministically.

        Returns:
            The export bindings map and the set of modules with a statically
            complete export surface.
        """
        exports: dict[str, dict[str, str]] = {}
        complete: set[str] = set()
        for extraction in sorted(extractions, key=lambda item: item.file_path):
            bindings = {record.name: record.target for record in extraction.exports}
            if bindings:
                exports[extraction.module_name] = bindings
            if extraction.exports_complete:
                complete.add(extraction.module_name)
        return exports, complete

    def _module_export_rows(self, extractions: list[FileExtraction]) -> list[ModuleExport]:
        """Flatten fresh extractions into persisted module export rows (IG-02).

        Args:
            extractions: Fresh extractions.

        Returns:
            Deterministically ordered :class:`ModuleExport` rows.
        """
        rows: list[ModuleExport] = []
        for extraction in sorted(extractions, key=lambda item: item.file_path):
            adapter = self._adapters.get(extraction.language)
            version = adapter.extractor_version if adapter is not None else ""
            for record in sorted(extraction.exports, key=lambda item: (item.name, item.target)):
                rows.append(
                    ModuleExport(
                        module_name=extraction.module_name,
                        name=record.name,
                        target=record.target,
                        file_path=extraction.file_path,
                        language=extraction.language,
                        adapter_version=version,
                        complete=extraction.exports_complete,
                    )
                )
        return rows

    def _metadata(self, generation: int, built_at: str, languages: set[str]) -> GraphMetadata:
        """Build the versioned metadata for one generation.

        Args:
            generation: The generation to publish.
            built_at: ISO build timestamp.
            languages: Languages present in the index; each contributes its
                adapter version so upgrades mark only affected files stale.

        Returns:
            The :class:`GraphMetadata` to publish.
        """
        versions = {}
        for language in sorted(languages):
            adapter = self._adapter_for_language(language)
            if adapter is None:  # pragma: no cover - languages come from the registry
                continue
            versions[language] = adapter.extractor_version
        combined = combined_adapter_version(versions)
        return GraphMetadata(
            schema_version=GRAPH_SCHEMA_VERSION,
            extractor_version=combined,
            resolver_version=combined,
            supported_languages=list(SUPPORTED_LANGUAGES),
            built_at=built_at,
            generation=generation,
            adapter_versions=dict(versions),
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
        exports, complete = self._fresh_export_maps(extractions)
        indexes = build_symbol_indexes(nodes, exports=exports, exports_complete=complete)
        edges = resolve_extraction_edges(extractions, indexes, str(self.repository_root))
        published = store.replace_all(
            nodes,
            edges,
            self._metadata(generation, built_at, self._index_languages(files)),
            self._module_export_rows(extractions),
        )
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
        supported: list[str],
    ) -> GraphBuildResult:
        extractions = self._extract(sorted(process - deleted))
        surviving_nodes = [node for node in store.iter_nodes() if node.file_path not in process]
        fresh_nodes = [node for extraction in extractions for node in extraction.nodes]
        # Persisted export surfaces keep cross-file import resolution working
        # for modules that were not re-extracted; fresh extractions override
        # their own modules' persisted rows.
        persisted = store.export_maps()
        exports, complete = self._fresh_export_maps(extractions)
        for module_name, (bindings, _files, module_complete) in persisted.items():
            if module_name not in exports:
                exports[module_name] = bindings
            if module_complete:
                complete.add(module_name)
        indexes = build_symbol_indexes([*surviving_nodes, *fresh_nodes], exports=exports, exports_complete=complete)
        edges = resolve_extraction_edges(extractions, indexes, str(self.repository_root))
        changed_map: dict[str, tuple[list, list]] = {
            extraction.file_path: (
                extraction.nodes,
                [edge for edge in edges if edge.evidence_file == extraction.file_path],
            )
            for extraction in extractions
        }
        generation = existing.generation + 1
        published = store.update_files(
            changed_map,
            sorted(deleted),
            self._metadata(generation, built_at, self._index_languages(supported)),
            self._module_export_rows(extractions),
        )
        return GraphBuildResult(
            generation=generation,
            metadata=published,
            mode="incremental",
            reextracted_files=len(extractions),
            deleted_files=len(deleted),
        )
