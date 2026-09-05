"""Unit tests for the IG-02 versioned graph DTOs and the capability report."""

from __future__ import annotations

import pytest

from ctxai.graph.adapters import (
    CAPABILITIES_SCHEMA_VERSION,
    GraphCapabilities,
    capabilities_payload,
    capability_matrix_markdown,
    get_adapter,
    language_for_file,
)
from ctxai.graph.dto import (
    GRAPH_RESULT_SCHEMA_VERSION,
    CapabilitiesResult,
    EdgeRecord,
    GraphHealthRecord,
    GraphStatsResult,
    IndexCapabilities,
    NeighborsResult,
    SymbolRecord,
    SymbolSearchResult,
)
from ctxai.graph.model import (
    CONFIDENCE_EXACT,
    GRAPH_SCHEMA_VERSION,
    GraphEdge,
    GraphMetadata,
    GraphNode,
    derive_edge_id,
    derive_node_id,
)
from ctxai.graph.operations import GraphHealth, GraphStats

REPO = "/repos/demo"


def node(kind: str = "function", qualified: str = "pkg.run", file_path: str = "pkg/mod.py") -> GraphNode:
    return GraphNode(
        id=derive_node_id(REPO, file_path, kind, qualified),
        kind=kind,
        qualified_name=qualified,
        display_name=qualified.rsplit(".", 1)[-1],
        language="python",
        file_path=file_path,
        start_line=3,
        end_line=7,
        parent_id=None,
        visibility="public",
        source_hash="ab" * 32,
        adapter_version="python/1",
    )


def edge(source: str, target: str | None) -> GraphEdge:
    return GraphEdge(
        id=derive_edge_id(REPO, "calls", source, target, None if target else "run()", "pkg/mod.py", 4),
        kind="calls",
        source_id=source,
        target_id=target,
        target_text=None if target else "run()",
        evidence_file="pkg/mod.py",
        evidence_line=4,
        confidence=CONFIDENCE_EXACT if target else "unresolved",
        resolver_version="python/1",
    )


def health(status: str = "healthy") -> GraphHealth:
    return GraphHealth(status=status, metadata=None, problems=(), diagnostic="d" if status == "missing" else None)


# -- record projections -------------------------------------------------------


class TestRecordRoundTrips:
    def test_symbol_record_round_trip(self):
        record = SymbolRecord.from_node(node())
        assert SymbolRecord.from_dict(record.to_dict()) == record
        assert record.evidence == "pkg/mod.py:3-7"

    def test_edge_record_round_trip(self):
        source_id = node().id
        record = EdgeRecord.from_edge(edge(source_id, None))
        assert EdgeRecord.from_dict(record.to_dict()) == record
        assert record.evidence == "pkg/mod.py:4"
        assert record.target_text == "run()"

    def test_health_record_round_trip(self):
        record = GraphHealthRecord.from_health(health("missing"))
        assert GraphHealthRecord.from_dict(record.to_dict()) == record

    def test_index_capabilities_round_trip(self):
        record = IndexCapabilities(
            index="demo",
            health=GraphHealthRecord.from_health(health()),
            languages_present={"python": 12, "javascript": 3},
            unresolved_edges_by_kind={"calls": 2},
            unsupported_file_count=4,
            total_file_count=20,
        )
        assert IndexCapabilities.from_dict(record.to_dict()) == record

    def test_stats_result_round_trip(self):
        metadata = GraphMetadata(
            schema_version=GRAPH_SCHEMA_VERSION,
            extractor_version="python/1",
            resolver_version="python/1",
            supported_languages=["python"],
            built_at="2026-09-05T00:00:00+00:00",
            generation=2,
        )
        stats = GraphStats(index_name="demo", path=None, health=health(), metadata=metadata)  # type: ignore[arg-type]
        result = GraphStatsResult.build("demo", stats, capabilities_payload())
        assert GraphStatsResult.from_dict(result.to_dict()) == result
        assert result.schema_version == GRAPH_RESULT_SCHEMA_VERSION

    def test_search_and_neighbors_results_round_trip(self):
        start = node()
        target = node(qualified="pkg.helper")
        search = SymbolSearchResult.build("demo", "run", None, "python", [start])
        assert SymbolSearchResult.from_dict(search.to_dict()) == search
        from ctxai.graph.store import NeighborResult

        neighbor = NeighborResult(
            start=start,
            nodes=[start, target],
            edges=[edge(start.id, target.id)],
            truncated=True,
        )
        result = NeighborsResult.build("demo", "abc123", "both", 2, 50, neighbor)
        assert NeighborsResult.from_dict(result.to_dict()) == result
        assert result.truncated is True
        assert result.resolved_symbol_id == start.id

    def test_capabilities_result_round_trip(self):
        result = CapabilitiesResult.build(capabilities_payload())
        assert CapabilitiesResult.from_dict(result.to_dict()) == result


class TestProjectionAgreement:
    def test_same_node_projects_identically_for_every_surface(self):
        """CLI JSON, MCP, and dashboard build from the same projection helpers."""
        graph_node = node()
        via_search = SymbolSearchResult.build("demo", "run", None, None, [graph_node]).symbols[0].to_dict()
        from ctxai.graph.store import NeighborResult

        via_neighbors = NeighborsResult.build(
            "demo", graph_node.id, "both", 1, 50, NeighborResult(graph_node, [graph_node], [], False)
        ).start.to_dict()
        direct = SymbolRecord.from_node(graph_node).to_dict()
        assert via_search == via_neighbors == direct
        assert via_search["evidence"] == "pkg/mod.py:3-7"
        assert via_search["adapter_version"] == "python/1"


# -- capability report --------------------------------------------------------


class TestCapabilitiesPayload:
    def test_supported_languages_carry_adapters_and_extensions(self):
        payload = capabilities_payload()
        assert payload["schema_version"] == CAPABILITIES_SCHEMA_VERSION
        by_language = {entry["language"]: entry for entry in payload["languages"]}
        for language in ("python", "javascript", "typescript"):
            entry = by_language[language]
            assert entry["supported"] is True
            assert entry["adapter_version"] == get_adapter(language).extractor_version
            assert entry["file_extensions"]
            assert entry["node_kinds"]
            assert entry["edge_kinds"]
            assert entry["supported_constructs"]
            assert entry["unsupported_constructs"]

    def test_unsupported_languages_are_listed_explicitly(self):
        payload = capabilities_payload()
        by_language = {entry["language"]: entry for entry in payload["languages"]}
        for language in ("java", "go", "rust"):
            entry = by_language[language]
            assert entry["supported"] is False
            assert entry["adapter_version"] is None
            assert entry["node_kinds"] == []
            assert "no adapter in this build" in entry["unsupported_constructs"][0]

    def test_graph_capabilities_round_trip(self):
        payload = capabilities_payload()
        capabilities = GraphCapabilities.from_dict(payload)
        assert capabilities.to_dict() == payload
        assert capabilities.node_kinds == tuple(payload["node_kinds"])

    def test_matrix_markdown_has_generated_markers(self):
        markdown = capability_matrix_markdown()
        assert markdown.startswith("<!-- CAPABILITY-MATRIX:BEGIN")
        assert markdown.endswith("<!-- CAPABILITY-MATRIX:END -->")
        assert "| python | yes |" in markdown.replace(" python ", " python ")
        assert "javascript" in markdown
        assert "typescript" in markdown

    def test_language_for_file_is_case_sensitive_on_extension_only(self):
        assert language_for_file("src/App.TS") is None or language_for_file("src/App.TS") == "typescript"

    @pytest.mark.parametrize("path,language", [("a.js", "javascript"), ("b.jsx", "javascript"), ("c.ts", "typescript")])
    def test_extension_detection(self, path, language):
        assert language_for_file(path) == language
