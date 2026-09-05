"""Versioned graph query/result DTOs shared by the CLI, MCP, and dashboard (IG-02).

Every user-facing surface projects graph reads through these records, so the
same index and query produce identical identity, counts, confidence, and
relationship data everywhere. The DTOs are plain dataclasses with
``to_dict``/``from_dict`` round trips; ``schema_version`` lets clients detect
the payload contract they are reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import GRAPH_SCHEMA_VERSION, GraphEdge, GraphNode
from .operations import GraphHealth, GraphStats
from .store import NeighborResult

GRAPH_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SymbolRecord:
    """Versioned projection of one graph node (identity plus evidence).

    Attributes:
        id: Stable node id.
        kind: Node kind (one of NODE_KINDS).
        qualified_name: Fully qualified name.
        display_name: Bare definition name.
        language: Source language.
        file_path: Repository-relative file path.
        start_line: 1-based start line.
        end_line: 1-based end line.
        evidence: ``file:start-end`` citation.
        visibility: ``public`` or ``private``.
        parent_id: Enclosing node id when contained.
        adapter_version: Language adapter version that extracted the node.
    """

    id: str
    kind: str
    qualified_name: str
    display_name: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    evidence: str
    visibility: str
    parent_id: str | None
    adapter_version: str

    @classmethod
    def from_node(cls, node: GraphNode) -> SymbolRecord:
        """Project a stored node.

        Args:
            node: The graph node.

        Returns:
            The versioned symbol record.
        """
        return cls(
            id=node.id,
            kind=node.kind,
            qualified_name=node.qualified_name,
            display_name=node.display_name,
            language=node.language,
            file_path=node.file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            evidence=node.evidence(),
            visibility=node.visibility,
            parent_id=node.parent_id,
            adapter_version=node.adapter_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "qualified_name": self.qualified_name,
            "display_name": self.display_name,
            "language": self.language,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "evidence": self.evidence,
            "visibility": self.visibility,
            "parent_id": self.parent_id,
            "adapter_version": self.adapter_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SymbolRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt SymbolRecord.
        """
        return cls(
            id=payload["id"],
            kind=payload["kind"],
            qualified_name=payload["qualified_name"],
            display_name=payload["display_name"],
            language=payload["language"],
            file_path=payload["file_path"],
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
            evidence=payload["evidence"],
            visibility=payload["visibility"],
            parent_id=payload["parent_id"],
            adapter_version=payload.get("adapter_version", ""),
        )


@dataclass(frozen=True)
class EdgeRecord:
    """Versioned projection of one graph edge (relationship plus evidence).

    Attributes:
        id: Stable edge id.
        kind: Edge kind (one of EDGE_KINDS).
        source_id: Source node id.
        target_id: Target node id when resolved, otherwise ``None``.
        target_text: Unresolved target text when unresolved.
        evidence_file: Repository-relative evidence file.
        evidence_line: 1-based evidence line.
        evidence: ``file:line`` citation.
        confidence: ``exact``, ``probable``, or ``unresolved``.
        resolver_version: Resolver version that produced the edge.
    """

    id: str
    kind: str
    source_id: str
    target_id: str | None
    target_text: str | None
    evidence_file: str
    evidence_line: int
    evidence: str
    confidence: str
    resolver_version: str

    @classmethod
    def from_edge(cls, edge: GraphEdge) -> EdgeRecord:
        """Project a stored edge.

        Args:
            edge: The graph edge.

        Returns:
            The versioned edge record.
        """
        return cls(
            id=edge.id,
            kind=edge.kind,
            source_id=edge.source_id,
            target_id=edge.target_id,
            target_text=edge.target_text,
            evidence_file=edge.evidence_file,
            evidence_line=edge.evidence_line,
            evidence=edge.evidence(),
            confidence=edge.confidence,
            resolver_version=edge.resolver_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "target_text": self.target_text,
            "evidence_file": self.evidence_file,
            "evidence_line": self.evidence_line,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "resolver_version": self.resolver_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt EdgeRecord.
        """
        return cls(
            id=payload["id"],
            kind=payload["kind"],
            source_id=payload["source_id"],
            target_id=payload["target_id"],
            target_text=payload["target_text"],
            evidence_file=payload["evidence_file"],
            evidence_line=int(payload["evidence_line"]),
            evidence=payload["evidence"],
            confidence=payload["confidence"],
            resolver_version=payload["resolver_version"],
        )


@dataclass(frozen=True)
class GraphHealthRecord:
    """Versioned projection of the graph health verdict.

    Attributes:
        status: One of ``healthy``, ``missing``, ``mismatch``,
            ``unsupported_schema``, ``corrupt``, ``count_mismatch``.
        problems: Human-readable problems; empty when healthy/missing.
        diagnostic: Informational message when the graph simply is absent.
    """

    status: str
    problems: tuple[str, ...]
    diagnostic: str | None = None

    @classmethod
    def from_health(cls, health: GraphHealth) -> GraphHealthRecord:
        """Project a health verdict.

        Args:
            health: The domain health result.

        Returns:
            The versioned health record.
        """
        return cls(status=health.status, problems=health.problems, diagnostic=health.diagnostic)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields with tuple values as lists.
        """
        return {"status": self.status, "problems": list(self.problems), "diagnostic": self.diagnostic}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphHealthRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt GraphHealthRecord.
        """
        return cls(
            status=payload["status"],
            problems=tuple(payload["problems"]),
            diagnostic=payload.get("diagnostic"),
        )


@dataclass(frozen=True)
class GraphStatsResult:
    """Versioned ``graph stats`` projection (also the MCP ``graph_stats`` data).

    Attributes:
        schema_version: Result payload schema version.
        index: Index name.
        graph_schema_version: Graph storage schema version the build used.
        health: Health verdict.
        graph: Graph metadata dictionary when readable, otherwise ``None``.
        capabilities: Versioned per-language support matrix.
    """

    schema_version: int
    index: str
    graph_schema_version: int
    health: GraphHealthRecord
    graph: dict[str, Any] | None
    capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, index: str, stats: GraphStats, capabilities: dict[str, Any]) -> GraphStatsResult:
        """Project a stats result.

        Args:
            index: Index name.
            stats: The domain stats (health plus optional metadata).
            capabilities: The versioned capabilities payload.

        Returns:
            The versioned stats result.
        """
        return cls(
            schema_version=GRAPH_RESULT_SCHEMA_VERSION,
            index=index,
            graph_schema_version=GRAPH_SCHEMA_VERSION,
            health=GraphHealthRecord.from_health(stats.health),
            graph=stats.metadata.to_dict() if stats.metadata else None,
            capabilities=capabilities,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields.
        """
        return {
            "schema_version": self.schema_version,
            "index": self.index,
            "graph_schema_version": self.graph_schema_version,
            "health": self.health.to_dict(),
            "graph": self.graph,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphStatsResult:
        """Rebuild a result from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt GraphStatsResult.
        """
        return cls(
            schema_version=int(payload["schema_version"]),
            index=payload["index"],
            graph_schema_version=int(payload["graph_schema_version"]),
            health=GraphHealthRecord.from_dict(payload["health"]),
            graph=payload["graph"],
            capabilities=payload.get("capabilities", {}),
        )


@dataclass(frozen=True)
class SymbolSearchResult:
    """Versioned ``graph symbol`` projection (also the MCP ``graph_symbol`` data).

    Attributes:
        schema_version: Result payload schema version.
        index: Index name.
        query: The submitted query text.
        kind: Node kind filter when given.
        language: Language filter when given.
        count: Number of returned symbols.
        symbols: Matching symbol records in deterministic order.
    """

    schema_version: int
    index: str
    query: str
    kind: str | None
    language: str | None
    count: int
    symbols: tuple[SymbolRecord, ...]

    @classmethod
    def build(
        cls,
        index: str,
        query: str,
        kind: str | None,
        language: str | None,
        nodes: list[GraphNode],
    ) -> SymbolSearchResult:
        """Project a symbol search.

        Args:
            index: Index name.
            query: The submitted query text.
            kind: Node kind filter when given.
            language: Language filter when given.
            nodes: Matching nodes in deterministic order.

        Returns:
            The versioned search result.
        """
        return cls(
            schema_version=GRAPH_RESULT_SCHEMA_VERSION,
            index=index,
            query=query,
            kind=kind,
            language=language,
            count=len(nodes),
            symbols=tuple(SymbolRecord.from_node(node) for node in nodes),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields with tuple values as lists.
        """
        return {
            "schema_version": self.schema_version,
            "index": self.index,
            "query": self.query,
            "kind": self.kind,
            "language": self.language,
            "count": self.count,
            "symbols": [record.to_dict() for record in self.symbols],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SymbolSearchResult:
        """Rebuild a result from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt SymbolSearchResult.
        """
        return cls(
            schema_version=int(payload["schema_version"]),
            index=payload["index"],
            query=payload["query"],
            kind=payload["kind"],
            language=payload["language"],
            count=int(payload["count"]),
            symbols=tuple(SymbolRecord.from_dict(item) for item in payload["symbols"]),
        )


@dataclass(frozen=True)
class NeighborsResult:
    """Versioned ``graph neighbors`` projection (also the MCP ``graph_neighbors`` data).

    Attributes:
        schema_version: Result payload schema version.
        index: Index name.
        symbol_id: The requested id (may be a prefix).
        resolved_symbol_id: The full id the request resolved to.
        direction: ``in``, ``out``, or ``both``.
        depth: Traversal depth used.
        limit: Result limit used.
        truncated: Whether more nodes existed beyond the limit.
        start: The start node record when found, otherwise ``None``.
        nodes: Reached node records (including the start), id-ordered.
        edges: Traversed edge records, id-ordered.
    """

    schema_version: int
    index: str
    symbol_id: str
    resolved_symbol_id: str
    direction: str
    depth: int
    limit: int
    truncated: bool
    start: SymbolRecord | None
    nodes: tuple[SymbolRecord, ...]
    edges: tuple[EdgeRecord, ...]

    @classmethod
    def build(
        cls,
        index: str,
        symbol_id: str,
        direction: str,
        depth: int,
        limit: int,
        result: NeighborResult,
    ) -> NeighborsResult:
        """Project a bounded traversal.

        Args:
            index: Index name.
            symbol_id: The requested id.
            direction: ``in``, ``out``, or ``both``.
            depth: Traversal depth used.
            limit: Result limit used.
            result: The domain traversal result.

        Returns:
            The versioned neighbors result.
        """
        return cls(
            schema_version=GRAPH_RESULT_SCHEMA_VERSION,
            index=index,
            symbol_id=symbol_id,
            resolved_symbol_id=result.start.id if result.start else "",
            direction=direction,
            depth=depth,
            limit=limit,
            truncated=result.truncated,
            start=SymbolRecord.from_node(result.start) if result.start else None,
            nodes=tuple(SymbolRecord.from_node(node) for node in result.nodes),
            edges=tuple(EdgeRecord.from_edge(edge) for edge in result.edges),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields with tuple values as lists.
        """
        return {
            "schema_version": self.schema_version,
            "index": self.index,
            "symbol_id": self.symbol_id,
            "resolved_symbol_id": self.resolved_symbol_id,
            "direction": self.direction,
            "depth": self.depth,
            "limit": self.limit,
            "truncated": self.truncated,
            "start": self.start.to_dict() if self.start else None,
            "nodes": [record.to_dict() for record in self.nodes],
            "edges": [record.to_dict() for record in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeighborsResult:
        """Rebuild a result from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt NeighborsResult.
        """
        return cls(
            schema_version=int(payload["schema_version"]),
            index=payload["index"],
            symbol_id=payload["symbol_id"],
            resolved_symbol_id=payload["resolved_symbol_id"],
            direction=payload["direction"],
            depth=int(payload["depth"]),
            limit=int(payload["limit"]),
            truncated=bool(payload["truncated"]),
            start=SymbolRecord.from_dict(payload["start"]) if payload["start"] else None,
            nodes=tuple(SymbolRecord.from_dict(item) for item in payload["nodes"]),
            edges=tuple(EdgeRecord.from_dict(item) for item in payload["edges"]),
        )


@dataclass(frozen=True)
class IndexCapabilities:
    """Versioned per-index capability observations.

    Attributes:
        index: Index name.
        health: Graph health verdict (a missing graph is diagnostic only).
        languages_present: Languages that contributed graph nodes, with node
            counts.
        unresolved_edges_by_kind: Unresolved edge counts per edge kind.
        unsupported_file_count: Number of indexed files whose language has no
            adapter (they stay indexable as ordinary chunks).
        total_file_count: Total indexed files when the manifest is readable.
    """

    index: str
    health: GraphHealthRecord
    languages_present: dict[str, int]
    unresolved_edges_by_kind: dict[str, int]
    unsupported_file_count: int
    total_file_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields.
        """
        return {
            "index": self.index,
            "health": self.health.to_dict(),
            "languages_present": dict(self.languages_present),
            "unresolved_edges_by_kind": dict(self.unresolved_edges_by_kind),
            "unsupported_file_count": self.unsupported_file_count,
            "total_file_count": self.total_file_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexCapabilities:
        """Rebuild from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt IndexCapabilities.
        """
        return cls(
            index=payload["index"],
            health=GraphHealthRecord.from_dict(payload["health"]),
            languages_present={str(key): int(value) for key, value in payload["languages_present"].items()},
            unresolved_edges_by_kind={
                str(key): int(value) for key, value in payload["unresolved_edges_by_kind"].items()
            },
            unsupported_file_count=int(payload["unsupported_file_count"]),
            total_file_count=payload.get("total_file_count"),
        )


@dataclass(frozen=True)
class CapabilitiesResult:
    """Versioned ``graph capabilities`` projection (CLI, MCP, dashboard).

    Attributes:
        schema_version: Result payload schema version.
        graph_schema_version: Graph storage schema version this build writes.
        languages: Per-language support matrix entries (static for this build).
        node_kinds: The closed node-kind vocabulary.
        edge_kinds: The closed edge-kind vocabulary.
        index: Per-index observations when an index was inspected.
    """

    schema_version: int
    graph_schema_version: int
    languages: tuple[dict[str, Any], ...]
    node_kinds: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    index: IndexCapabilities | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary of all fields with tuple values as lists.
        """
        return {
            "schema_version": self.schema_version,
            "graph_schema_version": self.graph_schema_version,
            "languages": [dict(entry) for entry in self.languages],
            "node_kinds": list(self.node_kinds),
            "edge_kinds": list(self.edge_kinds),
            "index": self.index.to_dict() if self.index else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CapabilitiesResult:
        """Rebuild a result from :meth:`to_dict` output.

        Args:
            payload: Dictionary produced by :meth:`to_dict`.

        Returns:
            The rebuilt CapabilitiesResult.
        """
        return cls(
            schema_version=int(payload["schema_version"]),
            graph_schema_version=int(payload["graph_schema_version"]),
            languages=tuple(dict(entry) for entry in payload["languages"]),
            node_kinds=tuple(payload["node_kinds"]),
            edge_kinds=tuple(payload["edge_kinds"]),
            index=IndexCapabilities.from_dict(payload["index"]) if payload.get("index") else None,
        )

    @classmethod
    def build(
        cls,
        capabilities: dict[str, Any],
        index: IndexCapabilities | None = None,
    ) -> CapabilitiesResult:
        """Project the versioned capabilities report.

        Args:
            capabilities: The static per-language support matrix payload
                (from :func:`ctxai.graph.adapters.capabilities_payload`).
            index: Per-index observations when an index was inspected.

        Returns:
            The versioned capabilities result.
        """
        return cls(
            schema_version=GRAPH_RESULT_SCHEMA_VERSION,
            graph_schema_version=GRAPH_SCHEMA_VERSION,
            languages=tuple(dict(entry) for entry in capabilities["languages"]),
            node_kinds=tuple(capabilities["node_kinds"]),
            edge_kinds=tuple(capabilities["edge_kinds"]),
            index=index,
        )
