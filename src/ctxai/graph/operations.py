"""Graph application service shared by the CLI (IG-01).

``GraphOperations`` is the only surface user-facing adapters use to read the
symbol graph; nothing queries SQLite directly. All inputs are validated and
bounded before work begins (query length, traversal depth, result limits,
edge kinds, directions), and every query is parameterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..index_manifest import IndexManifest, IndexManifestError
from .model import (
    EDGE_KINDS,
    GRAPH_FILENAME,
    GRAPH_SCHEMA_VERSION,
    MAX_RESULT_LIMIT,
    MAX_SYMBOL_QUERY_LENGTH,
    MAX_TRAVERSAL_DEPTH,
    NODE_KINDS,
    RELATIONSHIP_EDGE_KINDS,
    SUPPORTED_LANGUAGES,
    GraphMetadata,
    GraphNode,
)
from .store import GraphSchemaError, GraphStore, GraphStoreError, NeighborResult

if TYPE_CHECKING:
    from .dto import CapabilitiesResult

DIRECTIONS = ("in", "out", "both")


class GraphError(RuntimeError):
    """Raised when a graph operation cannot be served honestly."""


class GraphIndexNotFoundError(GraphError):
    """Raised when the requested index does not exist."""


class GraphNotBuiltError(GraphError):
    """Raised when the index exists but has no readable graph generation."""


@dataclass(frozen=True)
class GraphHealth:
    """Graph health verdict for one index.

    Attributes:
        status: ``healthy``, ``missing`` (never built; diagnostic only),
            ``mismatch``, ``unsupported_schema``, ``corrupt``, or
            ``count_mismatch``.
        metadata: The store metadata when readable, otherwise ``None``.
        problems: Human-readable problems; empty for healthy/missing.
        diagnostic: Informational message when the graph simply does not
            exist yet.
    """

    status: str
    metadata: GraphMetadata | None
    problems: tuple[str, ...]
    diagnostic: str | None = None


@dataclass(frozen=True)
class GraphStats:
    """Stats result for one index.

    Attributes:
        index_name: Index the stats describe.
        path: Canonical index directory.
        health: Graph health verdict.
        metadata: Graph metadata when the store is readable.
    """

    index_name: str
    path: Path
    health: GraphHealth
    metadata: GraphMetadata | None


def graph_health(index_path: Path, manifest: IndexManifest | None) -> GraphHealth:
    """Assess graph health against the index manifest.

    Args:
        index_path: Canonical index directory.
        manifest: The index manifest (may be ``None`` or graph-less).

    Returns:
        A :class:`GraphHealth`; a never-built graph is ``missing`` with no
        problems, while referenced-but-broken graphs carry problems.
    """
    store = GraphStore(index_path)
    manifest_generation = manifest.graph_generation if manifest is not None else None
    if not store.exists():
        if manifest_generation is not None:
            return GraphHealth(
                status="mismatch",
                metadata=None,
                problems=(
                    f"manifest references graph generation {manifest_generation} but {GRAPH_FILENAME} is missing",
                ),
            )
        return GraphHealth(
            status="missing",
            metadata=None,
            problems=(),
            diagnostic="graph data has not been built for this index; run ctxai index to generate it",
        )
    try:
        metadata = store.read_metadata()
    except GraphSchemaError as exc:
        return GraphHealth(status="unsupported_schema", metadata=None, problems=(str(exc),))
    except GraphStoreError as exc:
        return GraphHealth(status="corrupt", metadata=None, problems=(str(exc),))

    problems: list[str] = []
    status = "healthy"
    if manifest is not None:
        if manifest.graph_schema_version is not None and manifest.graph_schema_version != GRAPH_SCHEMA_VERSION:
            problems.append(
                f"manifest references unsupported graph schema {manifest.graph_schema_version};"
                f" expected {GRAPH_SCHEMA_VERSION}. Rebuild the index."
            )
            status = "unsupported_schema"
        elif metadata.schema_version != GRAPH_SCHEMA_VERSION:
            problems.append(f"graph schema {metadata.schema_version} is not supported; expected {GRAPH_SCHEMA_VERSION}")
            status = "unsupported_schema"
        elif manifest.graph_schema_version is not None and manifest.graph_schema_version != metadata.schema_version:
            problems.append(
                f"graph schema {metadata.schema_version} does not match manifest schema {manifest.graph_schema_version}"
            )
            status = "mismatch"
        if manifest_generation is not None and metadata.generation != manifest_generation:
            problems.append(
                f"graph generation {metadata.generation} does not match manifest generation {manifest_generation}"
            )
            status = "mismatch" if status == "healthy" else status
        if (
            manifest.graph_extractor_version is not None
            and manifest.graph_extractor_version != metadata.extractor_version
        ):
            problems.append(
                f"graph extractor version {metadata.extractor_version!r} does not match manifest"
                f" extractor {manifest.graph_extractor_version!r}"
            )
            status = "mismatch" if status == "healthy" else status
        if manifest.graph_node_count is not None and metadata.total_nodes != manifest.graph_node_count:
            problems.append(
                f"graph node count {metadata.total_nodes} does not match manifest count {manifest.graph_node_count}"
            )
            status = "count_mismatch" if status == "healthy" else status
        if manifest.graph_edge_count is not None and metadata.total_edges != manifest.graph_edge_count:
            problems.append(
                f"graph edge count {metadata.total_edges} does not match manifest count {manifest.graph_edge_count}"
            )
            status = "count_mismatch" if status == "healthy" else status
    if metadata.total_nodes != store.count_nodes() or metadata.total_edges != store.count_edges():
        problems.append("graph metadata counts do not match its stored rows")
        status = "count_mismatch" if status == "healthy" else status
    return GraphHealth(status=status, metadata=metadata, problems=tuple(problems))


class GraphOperations:
    """Bounded, validated graph reads shared by user-facing adapters."""

    def __init__(self, project_path: Path | None = None) -> None:
        """Create the service for one project scope.

        Args:
            project_path: Project root used to locate indexes and config.
        """
        self.project_path = project_path
        # Imported lazily: index_operations imports graph_health from this
        # module for IndexOperations.inspect, so a module-level import here
        # would be circular.
        from ..index_operations import IndexOperations

        self._index_operations = IndexOperations(project_path)

    @staticmethod
    def resolve_index_name(project_path: Path | None, index_name: str | None) -> str:
        """Resolve the index to operate on.

        Args:
            project_path: Project root for configuration lookup.
            index_name: Explicit index name, or ``None`` to use the config
                default.

        Returns:
            The resolved index name.

        Raises:
            GraphError: If no name is given and none is configured.
        """
        if index_name:
            return index_name
        from ..config import ConfigManager

        configured = ConfigManager(project_path).load().index_name
        if not configured:
            raise GraphError(
                "No index name given and no default configured; pass an index name (e.g. 'ctxai graph stats my-index')"
            )
        return configured

    def _store_for(self, index_name: str) -> GraphStore:
        try:
            path = self._index_operations.path_for(index_name)
        except IndexManifestError as exc:
            raise GraphError(str(exc)) from exc
        if not path.is_dir():
            raise GraphIndexNotFoundError(f"Index '{index_name}' does not exist at {path}")
        return GraphStore(path)

    def _require_graph(self, index_name: str, store: GraphStore) -> None:
        """Refuse reads against missing, stale, or corrupt graph generations.

        Args:
            index_name: Index being read.
            store: The resolved graph store.

        Raises:
            GraphNotBuiltError: If the graph was never built (diagnostic).
            GraphError: If the graph generation is stale, unsupported, or
                corrupt — reads return a deterministic error instead of
                incomplete data.
        """
        try:
            manifest = IndexManifest.load_optional(store.index_path)
        except IndexManifestError as exc:
            raise GraphError(f"Index manifest for '{index_name}' is unreadable: {exc}") from exc
        health = graph_health(store.index_path, manifest)
        if health.status == "healthy":
            return
        if health.status == "missing":
            raise GraphNotBuiltError(
                f"Graph data has not been built for index '{index_name}'; run ctxai index to generate it"
            )
        detail = "; ".join(health.problems) if health.problems else health.status
        raise GraphError(f"Graph data for index '{index_name}' is not readable: {detail}")

    @staticmethod
    def _validate_query(query: str) -> str:
        text = query.strip()
        if not text:
            raise ValueError("Symbol query must not be empty")
        if len(text) > MAX_SYMBOL_QUERY_LENGTH:
            raise ValueError(f"Symbol query must be at most {MAX_SYMBOL_QUERY_LENGTH} characters")
        return text

    def stats(self, index_name: str) -> GraphStats:
        """Return graph stats and health for one index.

        Args:
            index_name: Index to inspect.

        Returns:
            The :class:`GraphStats` for the index.

        Raises:
            GraphIndexNotFoundError: If the index does not exist.
        """
        path = self._index_operations.path_for(index_name)
        if not path.is_dir():
            raise GraphIndexNotFoundError(f"Index '{index_name}' does not exist at {path}")
        manifest = IndexManifest.load_optional(path)
        health = graph_health(path, manifest)
        return GraphStats(index_name=index_name, path=path, health=health, metadata=health.metadata)

    def capabilities(self, index_name: str | None) -> CapabilitiesResult:
        """Return the versioned capability report, optionally for one index.

        Without an index the report carries the static per-language support
        matrix only. With an index it adds what that index actually contains:
        languages present with node counts, unresolved edges by kind, and the
        number of indexed files whose language has no adapter (they stay
        indexable as ordinary chunks — reported, never silently dropped).

        Args:
            index_name: Index to inspect, or ``None`` for the static matrix.

        Returns:
            The versioned :class:`CapabilitiesResult`.

        Raises:
            GraphIndexNotFoundError: If a named index does not exist.
        """
        from .adapters import capabilities_payload
        from .dto import CapabilitiesResult, GraphHealthRecord, IndexCapabilities

        index: IndexCapabilities | None = None
        if index_name:
            path = self._index_operations.path_for(index_name)
            if not path.is_dir():
                raise GraphIndexNotFoundError(f"Index '{index_name}' does not exist at {path}")
            try:
                manifest = IndexManifest.load_optional(path)
            except IndexManifestError as exc:
                raise GraphError(f"Index manifest for '{index_name}' is unreadable: {exc}") from exc
            health = graph_health(path, manifest)
            store = GraphStore(path)
            language_counts = store.language_node_counts() if store.exists() else {}
            unresolved = store.unresolved_counts_by_kind() if store.exists() else {}
            total_files: int | None = None
            unsupported_files = 0
            if manifest is not None:
                total_files = manifest.file_count
                from .adapters import language_for_file

                supported_files = sum(
                    1 for file_path in manifest.files if language_for_file(Path(file_path).as_posix()) is not None
                )
                unsupported_files = max(0, manifest.file_count - supported_files)
            index = IndexCapabilities(
                index=index_name,
                health=GraphHealthRecord.from_health(health),
                languages_present=language_counts,
                unresolved_edges_by_kind=unresolved,
                unsupported_file_count=unsupported_files,
                total_file_count=total_files,
            )
        return CapabilitiesResult.build(capabilities_payload(), index)

    def find_symbols(
        self,
        index_name: str,
        query: str,
        kind: str | None = None,
        language: str | None = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        """Find symbols by qualified/display name substring.

        Args:
            index_name: Index to search.
            query: Substring of the qualified or display name.
            kind: Optional node kind filter.
            language: Optional language filter.
            limit: Maximum results (1..MAX_RESULT_LIMIT).

        Returns:
            Matching nodes with evidence, deterministically ordered.

        Raises:
            ValueError: On unbounded or invalid inputs.
            GraphError: If the index or its graph data is unavailable.
        """
        text = self._validate_query(query)
        if kind is not None and kind not in NODE_KINDS:
            raise ValueError(f"Unknown node kind {kind!r}; expected one of {', '.join(NODE_KINDS)}")
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}; supported: {', '.join(SUPPORTED_LANGUAGES)}")
        if not 1 <= limit <= MAX_RESULT_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_RESULT_LIMIT}")
        store = self._store_for(index_name)
        self._require_graph(index_name, store)
        return store.find_nodes(text, kind=kind, language=language, limit=limit)

    def files_with_cross_file_relationships(
        self,
        index_name: str,
        file_paths: list[str] | set[str] | tuple[str, ...],
        edge_kinds: tuple[str, ...] | None = None,
    ) -> set[str]:
        """Report which supplied files participate in cross-file relationships.

        Used by the retrieval benchmark to derive the pre-registered
        relationship-oriented case cohort (IG-03). A missing graph yields an
        empty classification (no relationships exist to classify); an
        unreadable store raises :class:`GraphStoreError`.

        Args:
            index_name: Index to classify.
            file_paths: Repository-relative file paths (bounded by
                MAX_RESULT_LIMIT).
            edge_kinds: Relationship edge kinds; defaults to
                ``RELATIONSHIP_EDGE_KINDS`` (the IG-03 allowlist).

        Returns:
            The subset of ``file_paths`` with at least one resolved
            cross-file relationship edge.

        Raises:
            ValueError: On out-of-bounds inputs or unknown edge kinds.
            GraphStoreError: If the graph store is unreadable.
        """
        paths = sorted(set(file_paths))
        if len(paths) > MAX_RESULT_LIMIT:
            raise ValueError(f"file_paths must be at most {MAX_RESULT_LIMIT} paths")
        kinds = tuple(edge_kinds) if edge_kinds is not None else RELATIONSHIP_EDGE_KINDS
        unknown = [kind for kind in kinds if kind not in EDGE_KINDS]
        if unknown:
            raise ValueError(f"Unknown edge kinds {unknown}; expected one of {', '.join(EDGE_KINDS)}")
        if not paths:
            return set()
        store = self._store_for(index_name)
        if not store.exists():
            return set()
        return store.cross_file_relationship_files(paths, kinds)

    def neighbors(
        self,
        index_name: str,
        symbol_id: str,
        edge_kind: str | None = None,
        direction: str = "both",
        depth: int = 1,
        limit: int = 50,
    ) -> NeighborResult:
        """Traverse the graph around one symbol.

        Args:
            index_name: Index to traverse.
            symbol_id: Stable node id, or a unique prefix (>= 8 chars).
            edge_kind: Optional edge kind filter.
            direction: ``"in"``, ``"out"``, or ``"both"``.
            depth: Hops to traverse (1..MAX_TRAVERSAL_DEPTH).
            limit: Maximum returned nodes (1..MAX_RESULT_LIMIT).

        Returns:
            The bounded :class:`NeighborResult`.

        Raises:
            ValueError: On unbounded or invalid inputs.
            GraphError: If the index or its graph data is unavailable.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction {direction!r}; expected one of {', '.join(DIRECTIONS)}")
        if edge_kind is not None and edge_kind not in EDGE_KINDS:
            raise ValueError(f"Unknown edge kind {edge_kind!r}; expected one of {', '.join(EDGE_KINDS)}")
        if not 1 <= depth <= MAX_TRAVERSAL_DEPTH:
            raise ValueError(f"depth must be between 1 and {MAX_TRAVERSAL_DEPTH}")
        if not 1 <= limit <= MAX_RESULT_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_RESULT_LIMIT}")
        if not symbol_id.strip():
            raise ValueError("Symbol id must not be empty")
        if len(symbol_id) > MAX_SYMBOL_QUERY_LENGTH:
            raise ValueError(f"Symbol id must be at most {MAX_SYMBOL_QUERY_LENGTH} characters")
        store = self._store_for(index_name)
        self._require_graph(index_name, store)
        resolved = self._resolve_symbol_id(store, symbol_id.strip())
        return store.neighbors(resolved, edge_kind=edge_kind, direction=direction, depth=depth, limit=limit)

    @staticmethod
    def _resolve_symbol_id(store: GraphStore, symbol_id: str) -> str:
        exact = store.get_node(symbol_id)
        if exact is not None:
            return symbol_id
        if len(symbol_id) < 8:
            raise ValueError("Symbol id prefix must be at least 8 characters (or pass the full id)")
        matches = store.find_node_ids_by_prefix(symbol_id)
        if not matches:
            raise ValueError(f"No symbol matches id prefix {symbol_id!r}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous symbol id prefix {symbol_id!r} matches {len(matches)} symbols")
        return matches[0]
