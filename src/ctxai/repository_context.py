"""Repository-aware hybrid retrieval and bounded evidence assembly.

The retrieval path (IG-03) is built from independently measurable candidate
generators (semantic, lexical, symbol, repository-map, and optional graph
expansion) combined by one deterministic fusion policy: every component
contributes ``1 / (offset + rank)`` per ranking, ties break by
``(-score, file_path, start_line)``, and identical index + configuration +
query therefore always produce identical ordering and selected evidence.

Graph expansion (IG-03) is disabled by default. When enabled it seeds from
the top base hits, follows an allowlisted edge policy with depth/confidence
decay, deduplicates by chunk identity, respects hard caps, and only boosts or
adds chunks that exist in the vector store. Retrieval reads a graph only when
its generation matches the index manifest; when expansion is required but the
graph is absent/stale/corrupt the error is explicit, otherwise retrieval
falls back to base behavior with a visible diagnostic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    GRAPH_EDGE_WEIGHTS_DEFAULT,
    MAX_GRAPH_DEPTH,
    MAX_GRAPH_EXPANSION_CAP,
    RetrievalConfig,
)
from .graph.model import GraphNode
from .index_manifest import IndexManifest, IndexManifestError
from .utils import get_indexes_dir
from .vector_store import VectorStore

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Deterministic fusion policy: each component ranking contributes
# 1 / (FUSION_RANK_OFFSET + rank) to a candidate's fused score. This constant
# is part of retrieval behavior; changing it changes all rankings.
FUSION_RANK_OFFSET = 60

# Edge-confidence factors applied during graph expansion (IG-03). Unresolved
# edges never produce traversal targets, so their factor only matters for
# defensive scoring.
CONFIDENCE_FACTORS = {"exact": 1.0, "probable": 0.6, "unresolved": 0.0}

# Bound on symbol-name lookups per seed when mapping chunks to graph nodes.
MAX_SEED_SYMBOL_MATCHES = 10


@dataclass(frozen=True)
class GraphExpansionEvidence:
    """Why one candidate entered (or was boosted) through graph expansion.

    Attributes:
        seed_chunk_id: Chunk id of the base hit the expansion was seeded from.
        seed_citation: ``file:start-end`` citation of the seed chunk.
        seed_symbol: Qualified name of the seed symbol.
        expanded_symbol: Qualified name of the reached symbol.
        edge_kind: Allowlisted edge kind followed (e.g. ``calls``).
        confidence: Edge confidence (``exact``/``probable``).
        depth: Traversal depth of this hop (1-based).
        path: Human-readable ``seed -[kind]-> expanded`` path.
        contribution: Fused-score contribution added by this expansion.
    """

    seed_chunk_id: str
    seed_citation: str
    seed_symbol: str
    expanded_symbol: str
    edge_kind: str
    confidence: str
    depth: int
    path: str
    contribution: float

    def reason_text(self) -> str:
        """Render the relationship evidence as one reason line.

        Returns:
            A reason string citing both the source (seed) and the
            relationship (edge kind, confidence, depth).
        """
        return (
            f"graph expansion from {self.seed_citation} ({self.seed_symbol}) via {self.edge_kind}"
            f" [confidence {self.confidence}, depth {self.depth}] to {self.expanded_symbol}"
        )


@dataclass(frozen=True)
class GraphExpansionSettings:
    """Resolved, validated graph-expansion settings for one retrieval run.

    Attributes:
        enabled: Whether expansion runs at all (default OFF).
        required: When True, an unavailable graph raises instead of falling
            back (CLI ``--graph``).
        edge_weights: Allowlisted edge kind -> weight in ``(0, 1]``.
        seed_count: Maximum top base hits used as expansion seeds.
        expansion_cap: Maximum expanded symbols per query.
        max_neighbors_per_seed: Maximum neighbors returned per seed hop.
        depth: Traversal depth (1 default; 2 is the explicit bounded maximum).
    """

    enabled: bool = False
    required: bool = False
    edge_weights: dict[str, float] = field(default_factory=lambda: dict(GRAPH_EDGE_WEIGHTS_DEFAULT))
    seed_count: int = 3
    expansion_cap: int = 24
    max_neighbors_per_seed: int = 8
    depth: int = 1

    def __post_init__(self) -> None:
        """Validate bounds by reusing the persisted configuration validation.

        Raises:
            ValueError: If any bound or edge weight is out of range.
        """
        RetrievalConfig(
            token_budget=1,  # not a setting here; validated by ContextAssembler
            graph_edge_weights=dict(self.edge_weights),
            graph_seed_count=self.seed_count,
            graph_expansion_cap=self.expansion_cap,
            graph_max_neighbors_per_seed=self.max_neighbors_per_seed,
            graph_depth=self.depth,
        )
        if self.depth > MAX_GRAPH_DEPTH or self.expansion_cap > MAX_GRAPH_EXPANSION_CAP:
            raise ValueError("graph expansion settings exceed the hard maximums")

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig | None = None,
        *,
        enabled: bool | None = None,
        required: bool = False,
    ) -> GraphExpansionSettings:
        """Resolve runtime settings from the persisted retrieval configuration.

        Args:
            config: Persisted retrieval configuration (defaults apply when None).
            enabled: Explicit enablement override (CLI ``--graph/--no-graph``);
                ``None`` uses ``config.graph_enabled``.
            required: Whether an unavailable graph must fail explicitly.

        Returns:
            The validated settings.
        """
        source = config or RetrievalConfig()
        return cls(
            enabled=source.graph_enabled if enabled is None else enabled,
            required=required,
            edge_weights=dict(source.graph_edge_weights),
            seed_count=source.graph_seed_count,
            expansion_cap=source.graph_expansion_cap,
            max_neighbors_per_seed=source.graph_max_neighbors_per_seed,
            depth=source.graph_depth,
        )


@dataclass
class ContextItem:
    """A ranked code chunk with inspectable retrieval evidence."""

    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    graph_evidence: GraphExpansionEvidence | None = None

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class AssembledContext:
    index_name: str
    items: list[ContextItem]
    text: str
    estimated_tokens: int
    truncated: tuple[bool, ...] = ()
    # (citation, reason) for examined-but-not-selected items: "duplicate"
    # (identity already selected) or "budget" (would exceed the token budget).
    excluded: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ComponentRanking:
    """One candidate generator's deterministic ranking over chunk records.

    Attributes:
        component: Generator name (``semantic``, ``lexical``, ``symbol``,
            ``repository-map``).
        ranked: Ordered chunk records.
        adds_reasons: Whether non-debug runs record per-rank reasons for this
            component (the weak repository-map signal only explains in debug).
    """

    component: str
    ranked: list[dict]
    adds_reasons: bool = True


@dataclass
class RetrievalExplain:
    """Per-query explanation of the fusion and expansion decisions.

    Attributes:
        query: The query that produced the result.
        component_counts: Candidates produced per generator.
        components: Chunk id -> component -> fused-score contribution.
        base_order: ``(chunk_id, citation, base_score)`` before graph boost,
            in base-fusion order.
        seeds: Expansion seed records (chunk, symbol, base score).
        graph_candidates: Expanded candidates with path and contribution.
        diagnostics: Caps hit, skipped symbols, fallback notes.
        final_rank: Chunk id -> final 1-based rank.
        final_scores: Chunk id -> final fused score.
    """

    query: str
    component_counts: dict[str, int] = field(default_factory=dict)
    components: dict[str, dict[str, float]] = field(default_factory=dict)
    base_order: list[tuple[str, str, float]] = field(default_factory=list)
    seeds: list[dict] = field(default_factory=list)
    graph_candidates: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    final_rank: dict[str, int] = field(default_factory=dict)
    final_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Ranked retrieval output plus its optional explanation.

    Attributes:
        items: Ranked candidates.
        explain: Explanation report when requested, else ``None``.
        semantic_distances: Chunk id -> vector distance from the semantic
            generator (adapters that used to show ``max(0, 1 - distance)``
            keep that value; chunks without a vector hit are absent).
    """

    items: list[ContextItem]
    explain: RetrievalExplain | None = None
    semantic_distances: dict[str, float] = field(default_factory=dict)


@dataclass
class EvidenceResult:
    """One shared-service retrieval: ranked candidates plus assembled context.

    Attributes:
        index_name: The index that served the query (resolved).
        items: Ranked candidates.
        context: Assembled, budget-bounded context.
        explain: Explanation report when requested, else ``None``.
        graph_diagnostic: Visible fallback diagnostic when graph expansion was
            enabled in configuration but unavailable, else ``None``.
        semantic_distances: Chunk id -> vector distance from the semantic
            generator (for adapters that render a similarity percentage).
    """

    index_name: str | None
    items: list[ContextItem]
    context: AssembledContext
    explain: RetrievalExplain | None = None
    graph_diagnostic: str | None = None
    semantic_distances: dict[str, float] = field(default_factory=dict)


def discover_repository_indexes(project_path: Path) -> list[str]:
    """Find healthy indexes whose manifest identifies the current repository."""
    root = project_path.resolve()
    indexes_dir = get_indexes_dir(root)
    if not indexes_dir.exists():
        return []
    matches: list[tuple[str, str]] = []
    for index_path in indexes_dir.iterdir():
        if not index_path.is_dir():
            continue
        try:
            manifest = IndexManifest.load(index_path)
        except IndexManifestError:
            continue
        if Path(manifest.repository_root).resolve() == root:
            matches.append((manifest.updated_at, manifest.index_name))
    return [name for _, name in sorted(matches, reverse=True)]


def _terms(text: str) -> list[str]:
    return [term.lower() for term in TOKEN_RE.findall(text) if len(term) > 1]


def _item(record: dict) -> ContextItem:
    metadata = record.get("metadata") or {}
    return ContextItem(
        id=record["id"],
        content=record.get("content", ""),
        file_path=metadata.get("file_path", "unknown"),
        start_line=int(metadata.get("start_line", 1)),
        end_line=int(metadata.get("end_line", 1)),
        chunk_type=metadata.get("chunk_type", "unknown"),
    )


@dataclass(frozen=True)
class _Seed:
    """An expansion seed: a top base hit mapped to a graph symbol."""

    item: ContextItem
    base_rank: int
    base_score: float
    symbol: GraphNode  # imported lazily at module level to avoid cycles


@dataclass(frozen=True)
class _GraphCandidate:
    """One expanded symbol mapped back to a vector-store chunk."""

    chunk_id: str
    symbol_id: str
    evidence: GraphExpansionEvidence


class HybridRetriever:
    """Fuse semantic, lexical, symbol, and repository-structure signals.

    Candidate generators are independent and measurable; one deterministic
    fusion policy combines them. When graph expansion is enabled (IG-03), a
    fifth generator seeds from the top base hits and follows an allowlisted,
    capped, decayed edge policy; it may only boost or add chunks that exist
    in the vector store.
    """

    def __init__(
        self,
        project_path: Path,
        embedding_provider,
        index_name: str | None = None,
        graph: GraphExpansionSettings | None = None,
    ):
        """Create the retriever for one repository.

        Args:
            project_path: Repository root used to discover indexes.
            embedding_provider: Embedding provider for the semantic generator.
            index_name: Explicit index name (auto-discovered when None).
            graph: Resolved graph-expansion settings; disabled by default.

        Raises:
            LookupError: If no matching index exists, the index belongs to a
                different repository, or expansion was required but no healthy
                graph generation matches the manifest.
        """
        self.project_path = project_path.resolve()
        matches = discover_repository_indexes(self.project_path)
        self.index_name = index_name or (matches[0] if matches else None)
        if not self.index_name:
            raise LookupError("No index matches the current repository. Run 'ctxai index' first.")
        index_path = get_indexes_dir(self.project_path) / self.index_name
        manifest = IndexManifest.load_optional(index_path)
        if manifest is not None and Path(manifest.repository_root).resolve() != self.project_path:
            raise LookupError(f"Index '{self.index_name}' belongs to a different repository")
        self.store = VectorStore(index_path, self.index_name)
        self.embedding_provider = embedding_provider
        self._manifest = manifest
        self.graph_settings = graph or GraphExpansionSettings()
        self.graph_diagnostic: str | None = None
        self._graph_operations = self._prepare_graph_operations(index_path)

    # ------------------------------------------------------------------
    # Graph reader resolution (stale/missing/corrupt gating)
    # ------------------------------------------------------------------

    def _prepare_graph_operations(self, index_path: Path):
        """Resolve the graph read surface, gating on manifest generation match.

        Args:
            index_path: Canonical index directory.

        Returns:
            A :class:`GraphOperations` bound to this project, or ``None`` when
            expansion is disabled or the graph is unavailable (with a visible
            diagnostic recorded when fallback applies).

        Raises:
            LookupError: When expansion is required but the graph is absent,
                stale, unsupported, or corrupt.
        """
        if not self.graph_settings.enabled:
            return None
        from .graph.operations import GraphOperations, graph_health

        health = graph_health(index_path, self._manifest)
        if health.status == "healthy":
            return GraphOperations(self.project_path)
        if health.status == "missing":
            detail = "graph data has not been built for this index; run 'ctxai index' to generate it"
        else:
            detail = "; ".join(health.problems) if health.problems else health.status
        if self.graph_settings.required:
            raise LookupError(
                "Graph expansion was explicitly required (--graph) but no healthy graph generation"
                f" matches index '{self.index_name}': {detail}"
            )
        self.graph_diagnostic = f"graph expansion unavailable, falling back to base retrieval: {detail}"
        return None

    # ------------------------------------------------------------------
    # Candidate generators (independently measurable)
    # ------------------------------------------------------------------

    def _base_rankings(self, query: str, records: list[dict]) -> list[ComponentRanking]:
        """Run the base candidate generators over the complete local corpus.

        Args:
            query: The natural-language or symbol query.
            records: All chunk records from the vector store.

        Returns:
            Component rankings in fixed, deterministic order.
        """
        query_terms = set(_terms(query))
        # Fuse over the complete local corpus (bounded by index size), then
        # apply the caller's result limit. Early truncation can hide an exact
        # symbol match merely because its vector rank is low.
        semantic = self.store.search(self.embedding_provider.generate_embedding(query), n_results=len(records))
        rankings: list[ComponentRanking] = [ComponentRanking("semantic", semantic)]

        lexical = sorted(
            records,
            key=lambda record: sum(_terms(record.get("content", "")).count(term) for term in query_terms),
            reverse=True,
        )
        lexical = [r for r in lexical if query_terms & set(_terms(r.get("content", "")))]
        rankings.append(ComponentRanking("lexical", lexical))

        symbol = [
            record
            for record in records
            if query_terms & set(_terms((record.get("metadata") or {}).get("meta_name", "")))
        ]
        rankings.append(ComponentRanking("symbol", symbol))

        # Repository-map signal: filenames and important definition types provide
        # useful structure even when a query's wording differs from code prose.
        structure = sorted(
            records,
            key=lambda record: (
                bool(query_terms & set(_terms((record.get("metadata") or {}).get("file_path", "")))),
                (record.get("metadata") or {}).get("chunk_type", "")
                in {"class_definition", "class_declaration", "function_definition", "function_declaration"},
            ),
            reverse=True,
        )
        rankings.append(ComponentRanking("repository-map", structure, adds_reasons=False))
        return rankings

    # ------------------------------------------------------------------
    # Graph expansion generator
    # ------------------------------------------------------------------

    def _to_relative_path(self, file_path: str) -> str | None:
        """Normalize a chunk file path to a safe repository-relative path.

        Args:
            file_path: File path as recorded in chunk metadata.

        Returns:
            Repository-relative POSIX path, or ``None`` when the path is
            absolute and outside the repository (graph evidence outside the
            repository is always excluded).
        """
        if not file_path:
            return None
        candidate = Path(file_path)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.project_path).as_posix()
            except ValueError:
                return None
        text = candidate.as_posix()
        parts = candidate.parts
        if candidate.is_reserved() or text.startswith("../") or ".." in parts or text.startswith("/"):
            return None
        return text

    def _chunk_entries_by_file(self, records: list[dict]) -> tuple[dict[str, list[tuple[str, dict]]], int]:
        """Group chunk records by repository-relative file for symbol mapping.

        Args:
            records: All chunk records.

        Returns:
            Mapping of relative file path to ``(chunk_id, metadata)`` entries
            ordered by ``(start_line, chunk_id)``, plus the number of chunks
            excluded for living outside the repository.
        """
        grouped: dict[str, list[tuple[str, dict]]] = {}
        excluded = 0
        for record in records:
            metadata = record.get("metadata") or {}
            relative = self._to_relative_path(str(metadata.get("file_path", "")))
            if relative is None:
                excluded += 1
                continue
            grouped.setdefault(relative, []).append((record["id"], metadata))
        for entries in grouped.values():
            entries.sort(key=lambda entry: (entry[1].get("start_line", ""), entry[0]))
        return grouped, excluded

    def _chunk_for_node(self, node, grouped_chunks: dict[str, list[tuple[str, dict]]]) -> str | None:
        """Map an expanded graph node back to a chunk in the vector store.

        Args:
            node: The reached GraphNode.
            grouped_chunks: Chunk entries grouped by relative file path.

        Returns:
            The matching chunk id (same file plus symbol-name match, falling
            back to line-range containment), or ``None`` when no indexed
            chunk covers the symbol.
        """
        entries = grouped_chunks.get(node.file_path, [])
        for chunk_id, metadata in entries:
            if metadata.get("meta_name") == node.display_name:
                return chunk_id
        for chunk_id, metadata in entries:
            try:
                start = int(metadata.get("start_line", 0))
                end = int(metadata.get("end_line", 0))
            except (TypeError, ValueError):
                continue
            if start <= node.start_line and node.end_line <= end:
                return chunk_id
        return None

    def _seed_symbol_for_chunk(self, name: str, relative: str) -> GraphNode | None:
        """Find the deterministic graph symbol for a seed chunk.

        Args:
            name: Symbol name from chunk metadata.
            relative: Repository-relative file of the chunk.

        Returns:
            The matching GraphNode, or ``None`` when the graph has no symbol
            for this name in this file.
        """
        operations = self._graph_operations
        try:
            nodes = operations.find_symbols(self.index_name, name, limit=MAX_SEED_SYMBOL_MATCHES)
        except (ValueError, RuntimeError):
            return None
        matches = [node for node in nodes if node.file_path == relative]

        def preference(node):
            if node.display_name == name:
                name_rank = 0
            elif node.qualified_name == name or node.qualified_name.endswith("." + name):
                name_rank = 1
            else:
                name_rank = 2
            return (name_rank, node.qualified_name, node.id)

        return min(matches, key=preference) if matches else None

    def _expand_graph(
        self,
        base_order: list[ContextItem],
        grouped_chunks: dict[str, list[tuple[str, dict]]],
        excluded_chunks: int,
    ) -> tuple[list[_GraphCandidate], list[dict], list[str]]:
        """Run the bounded, allowlisted graph-expansion generator.

        Seeds from the top base hits, follows allowlisted edges up to the
        configured depth with ``weight ** depth * confidence`` decay, tracks
        visited symbols (cycle handling), caps neighbors and total expansion,
        and maps every expanded symbol back to a vector-store chunk.

        Args:
            base_order: Base-fusion ordering (pre-graph) of all items.
            grouped_chunks: Chunk entries grouped by relative file path.
            excluded_chunks: Chunks already excluded as outside the repository.

        Returns:
            ``(candidates, seeds, diagnostics)`` where candidates are dedupli-
            cated by chunk identity and deterministically ordered.
        """
        settings = self.graph_settings
        diagnostics: list[str] = []
        if excluded_chunks:
            diagnostics.append(f"{excluded_chunks} chunk(s) outside the repository excluded from graph evidence")
        metadata_by_chunk: dict[str, dict] = {
            chunk_id: metadata for entries in grouped_chunks.values() for chunk_id, metadata in entries
        }

        seeds: list[_Seed] = []
        for base_rank, item in enumerate(base_order, 1):
            if len(seeds) >= settings.seed_count:
                break
            metadata = metadata_by_chunk.get(item.id)
            if metadata is None:
                diagnostics.append(f"seed skipped (chunk outside repository): {item.citation}")
                continue
            name = str(metadata.get("meta_name", "") or "")
            relative = self._to_relative_path(str(metadata.get("file_path", "")))
            if not name or relative is None:
                diagnostics.append(f"seed skipped (no symbol name): {item.citation}")
                continue
            symbol = self._seed_symbol_for_chunk(name, relative)
            if symbol is None:
                diagnostics.append(f"seed skipped (no graph symbol for '{name}' in {relative})")
                continue
            seeds.append(_Seed(item=item, base_rank=base_rank, base_score=item.score, symbol=symbol))

        seed_reports = [
            {
                "chunk_id": seed.item.id,
                "citation": seed.item.citation,
                "symbol": seed.symbol.qualified_name,
                "symbol_id": seed.symbol.id,
                "base_rank": seed.base_rank,
                "base_score": seed.base_score,
            }
            for seed in seeds
        ]
        candidates: list[_GraphCandidate] = []
        if not seeds:
            diagnostics.append("no graph expansion seeds resolved from the top base hits")
            return candidates, seed_reports, diagnostics

        candidate_by_chunk: dict[str, _GraphCandidate] = {}
        accepted_symbols = 0
        visited: set[str] = {str(seed.symbol.id) for seed in seeds}
        # Frontier entries: (symbol, seed, path). Seeds start the walk; each
        # accepted expansion may continue when depth permits.
        frontier: list[tuple[GraphNode, _Seed, tuple[str, ...]]] = [
            (seed.symbol, seed, (str(seed.symbol.qualified_name),)) for seed in seeds
        ]
        cap_hit = False
        for depth in range(1, settings.depth + 1):
            next_frontier: list[tuple[GraphNode, _Seed, tuple[str, ...]]] = []
            for symbol, seed, path in frontier:
                if accepted_symbols >= settings.expansion_cap:
                    cap_hit = True
                    break
                neighbors_result = self._safe_neighbors(str(symbol.id), diagnostics)
                if neighbors_result is None:
                    continue
                neighbor_nodes = [node for node in neighbors_result.nodes if node.id != symbol.id]
                if neighbors_result.truncated:
                    diagnostics.append(
                        f"neighbors truncated at cap {settings.max_neighbors_per_seed} for {symbol.qualified_name}"
                    )
                for node in neighbor_nodes:
                    if accepted_symbols >= settings.expansion_cap:
                        cap_hit = True
                        break
                    if node.id in visited:
                        continue
                    edge = self._best_allowlisted_edge(
                        result=neighbors_result,
                        source_id=str(symbol.id),
                        target_id=str(node.id),
                        allowed_kinds=settings.edge_weights,
                    )
                    if edge is None:
                        # Reached only through non-allowlisted edges (e.g.
                        # contains); structural context, not evidence.
                        continue
                    weight = settings.edge_weights[edge.kind]
                    confidence_factor = CONFIDENCE_FACTORS.get(edge.confidence, 0.0)
                    contribution = seed.base_score * (weight**depth) * confidence_factor
                    visited.add(str(node.id))
                    accepted_symbols += 1
                    evidence = GraphExpansionEvidence(
                        seed_chunk_id=seed.item.id,
                        seed_citation=seed.item.citation,
                        seed_symbol=str(seed.symbol.qualified_name),
                        expanded_symbol=node.qualified_name,
                        edge_kind=edge.kind,
                        confidence=edge.confidence,
                        depth=depth,
                        path=f"{seed.symbol.qualified_name} -[{edge.kind}]-> {node.qualified_name}",
                        contribution=contribution,
                    )
                    chunk_id = self._chunk_for_node(node, grouped_chunks)
                    if chunk_id is None:
                        diagnostics.append(
                            f"expanded symbol without an indexed chunk (no evidence added):"
                            f" {node.qualified_name} ({node.evidence()})"
                        )
                    elif chunk_id in candidate_by_chunk:
                        diagnostics.append(
                            f"duplicate chunk identity deduplicated: {evidence.path}"
                            f" (kept {candidate_by_chunk[chunk_id].evidence.path})"
                        )
                    elif contribution <= 0:
                        diagnostics.append(f"expanded candidate dropped (zero contribution): {evidence.path}")
                    else:
                        candidate = _GraphCandidate(chunk_id=chunk_id, symbol_id=str(node.id), evidence=evidence)
                        candidate_by_chunk[chunk_id] = candidate
                        candidates.append(candidate)
                    if depth < settings.depth:
                        next_frontier.append((node, seed, path + (node.qualified_name,)))
            if cap_hit:
                diagnostics.append(f"expansion cap {settings.expansion_cap} reached; further traversal stopped")
                break
            frontier = next_frontier
        candidates.sort(key=lambda candidate: (-candidate.evidence.contribution, candidate.chunk_id))
        return candidates, seed_reports, diagnostics

    def _safe_neighbors(self, symbol_id: str, diagnostics: list[str]):
        """Read one bounded one-hop neighborhood, recording failures.

        Args:
            symbol_id: Start symbol id.
            diagnostics: Diagnostic list to append to.

        Returns:
            The NeighborResult, or ``None`` when the read failed.
        """
        try:
            return self._graph_operations.neighbors(
                self.index_name,
                symbol_id,
                direction="both",
                depth=1,
                limit=self.graph_settings.max_neighbors_per_seed + 1,
            )
        except (ValueError, RuntimeError) as exc:
            diagnostics.append(f"graph read failed for {symbol_id}: {exc}")
            return None

    @staticmethod
    def _best_allowlisted_edge(result, source_id: str, target_id: str, allowed_kinds: dict[str, float]):
        """Pick the deterministic strongest allowlisted edge between two symbols.

        Args:
            result: The NeighborResult holding the traversed edges.
            source_id: One endpoint symbol id.
            target_id: The other endpoint symbol id.
            allowed_kinds: Allowlisted edge kind -> weight (the configured
                policy); any other kind is never followed.

        Returns:
            The best GraphEdge by ``(weight * confidence, edge id)``, or
            ``None`` when no allowlisted edge connects the endpoints.
        """
        connecting = [
            edge
            for edge in result.edges
            if edge.kind in allowed_kinds
            and (
                (edge.source_id == source_id and edge.target_id == target_id)
                or (edge.source_id == target_id and edge.target_id == source_id)
            )
        ]
        if not connecting:
            return None
        return min(
            connecting,
            key=lambda edge: (
                -(allowed_kinds[edge.kind] * CONFIDENCE_FACTORS.get(edge.confidence, 0.0)),
                edge.id,
            ),
        )

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def retrieve(self, query: str, limit: int = 20, debug: bool = False) -> list[ContextItem]:
        """Retrieve ranked candidates (concise interface).

        Args:
            query: Natural-language or symbol query.
            limit: Maximum number of returned candidates.
            debug: Record per-component rank reasons for every component.

        Returns:
            Ranked candidates, fused deterministically.
        """
        return self.retrieve_detailed(query, limit=limit, explain=debug).items

    def retrieve_detailed(self, query: str, limit: int = 20, explain: bool = False) -> RetrievalResult:
        """Retrieve ranked candidates with an optional explanation report.

        Args:
            query: Natural-language or symbol query.
            limit: Maximum number of returned candidates.
            explain: When True, attach per-component contributions, base
                order, seeds, graph paths, and diagnostics.

        Returns:
            The :class:`RetrievalResult` with ranked items.
        """
        records = self.store.get_chunks()
        if not records:
            empty = RetrievalExplain(query=query, diagnostics=["index has no chunks"]) if explain else None
            return RetrievalResult([], empty)
        rankings = self._base_rankings(query, records)
        semantic_distances = {
            record["id"]: float(record["distance"]) for record in rankings[0].ranked if "distance" in record
        }
        by_id = {record["id"]: _item(record) for record in records}
        contributions: dict[str, dict[str, float]] = {}
        for ranking in rankings:
            for rank, record in enumerate(ranking.ranked, 1):
                item = by_id[record["id"]]
                contribution = 1.0 / (FUSION_RANK_OFFSET + rank)
                item.score += contribution
                components = contributions.setdefault(item.id, {})
                components[ranking.component] = components.get(ranking.component, 0.0) + contribution
                if explain or ranking.adds_reasons:
                    item.reasons.append(f"{ranking.component} rank {rank}")

        base_order = sorted(by_id.values(), key=lambda item: (-item.score, item.file_path, item.start_line))
        base_snapshot = [(item.id, item.citation, item.score) for item in base_order[:limit]]

        seeds: list[dict] = []
        graph_candidates: list[_GraphCandidate] = []
        diagnostics: list[str] = []
        if self.graph_settings.enabled and self._graph_operations is not None:
            grouped_chunks, excluded_chunks = self._chunk_entries_by_file(records)
            graph_candidates, seeds, diagnostics = self._expand_graph(
                base_order=base_order,
                grouped_chunks=grouped_chunks,
                excluded_chunks=excluded_chunks,
            )
            for candidate in graph_candidates:
                item = by_id[candidate.chunk_id]
                item.score += candidate.evidence.contribution
                components = contributions.setdefault(item.id, {})
                components["graph"] = components.get("graph", 0.0) + candidate.evidence.contribution
                item.graph_evidence = candidate.evidence
                item.reasons.append(candidate.evidence.reason_text())

        ordered = sorted(by_id.values(), key=lambda item: (-item.score, item.file_path, item.start_line))[:limit]

        explain_report: RetrievalExplain | None = None
        if explain:
            explain_report = RetrievalExplain(
                query=query,
                component_counts={ranking.component: len(ranking.ranked) for ranking in rankings},
                components=contributions,
                base_order=base_snapshot,
                seeds=seeds,
                graph_candidates=[
                    {
                        "chunk_id": candidate.chunk_id,
                        "citation": by_id[candidate.chunk_id].citation,
                        "path": candidate.evidence.path,
                        "edge_kind": candidate.evidence.edge_kind,
                        "confidence": candidate.evidence.confidence,
                        "depth": candidate.evidence.depth,
                        "contribution": candidate.evidence.contribution,
                    }
                    for candidate in graph_candidates
                ],
                diagnostics=diagnostics,
                final_rank={item.id: rank for rank, item in enumerate(ordered, 1)},
                final_scores={item.id: item.score for item in ordered},
            )
        return RetrievalResult(ordered, explain_report, semantic_distances)


class ContextAssembler:
    """Deduplicate and format ranked evidence within a hard approximate token budget."""

    def __init__(self, token_budget: int = 2000, debug: bool = False):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget
        self.debug = debug

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return math.ceil(len(text) / 4)

    def assemble(self, index_name: str, items: list[ContextItem]) -> AssembledContext:
        """Select evidence within the token budget, deduplicating identities.

        Args:
            index_name: Index the evidence came from.
            items: Ranked candidates in fusion order.

        Returns:
            The assembled context, including examined-but-not-selected items
            with their exclusion reasons (``duplicate``/``budget``).
        """
        selected: list[ContextItem] = []
        blocks: list[str] = []
        truncated_flags: list[bool] = []
        excluded: list[tuple[str, str]] = []
        seen: set[tuple[str, int, int]] = set()
        used = 0
        for item in items:
            identity = (item.file_path, item.start_line, item.end_line)
            if identity in seen:
                excluded.append((item.citation, "duplicate"))
                continue
            reason = f"\nSelected because: {', '.join(item.reasons)}" if self.debug else ""
            header = f"[{item.citation}] ({item.chunk_type}){reason}\n"
            remaining_chars = (self.token_budget - used) * 4 - len(header)
            if remaining_chars <= 0:
                excluded.append((item.citation, "budget"))
                break
            content = item.content[:remaining_chars]
            block = header + content
            cost = self.estimate_tokens(block)
            if used + cost > self.token_budget:
                excluded.append((item.citation, "budget"))
                break
            blocks.append(block)
            selected.append(item)
            truncated_flags.append(len(content) < len(item.content))
            seen.add(identity)
            used += cost
        return AssembledContext(
            index_name, selected, "\n\n".join(blocks), used, tuple(truncated_flags), tuple(excluded)
        )


def retrieve_evidence(
    project_path: Path,
    query: str,
    *,
    embedding_provider,
    index_name: str | None = None,
    limit: int = 5,
    token_budget: int | None = None,
    graph: GraphExpansionSettings | None = None,
    explain: bool = False,
) -> EvidenceResult:
    """Shared retrieval service: one query, ranked candidates, assembled context.

    Every user-facing surface (CLI query, agent semantic search, MCP query,
    dashboard query, evaluation) routes through this function so fusion,
    graph expansion, budgets, and diagnostics behave identically everywhere.

    Args:
        project_path: Repository root.
        query: Natural-language or symbol query.
        embedding_provider: Embedding provider (created by the caller from
            configuration so identity checks stay at the boundary).
        index_name: Explicit index name (auto-discovered when None).
        limit: Maximum ranked candidates returned.
        token_budget: Approximate context token budget (defaults to the
            packaged default configuration).
        graph: Resolved graph-expansion settings (disabled by default).
        explain: When True, attach the explanation report and debug reasons.

    Returns:
        The :class:`EvidenceResult` for the query.

    Raises:
        LookupError: When no matching index exists or a required graph is
            unavailable.
    """
    retriever = HybridRetriever(project_path, embedding_provider, index_name=index_name, graph=graph)
    result = retriever.retrieve_detailed(query, limit=max(1, limit), explain=explain)
    budget = token_budget if token_budget is not None else RetrievalConfig().token_budget
    context = ContextAssembler(token_budget=max(1, budget), debug=explain).assemble(
        retriever.index_name if retriever.index_name is not None else "", result.items
    )
    return EvidenceResult(
        index_name=retriever.index_name,
        items=result.items,
        context=context,
        explain=result.explain,
        graph_diagnostic=retriever.graph_diagnostic,
        semantic_distances=result.semantic_distances,
    )
