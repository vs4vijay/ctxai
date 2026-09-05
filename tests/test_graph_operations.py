"""Unit tests for the IG-01 graph application service: bounds, health, lookup."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ctxai.graph.model import (
    GRAPH_SCHEMA_VERSION,
    GraphMetadata,
    GraphNode,
    derive_node_id,
)
from ctxai.graph.operations import GraphError, GraphHealth, GraphOperations, graph_health
from ctxai.graph.store import GraphStore
from ctxai.index_manifest import IndexManifest

REPO = "/repos/demo"


def metadata(generation: int = 1, nodes: int = 2, edges: int = 1, unresolved: int = 0) -> GraphMetadata:
    return GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        extractor_version="python/1",
        resolver_version="python/1",
        supported_languages=["python"],
        built_at="2026-09-05T00:00:00+00:00",
        generation=generation,
        node_counts={"function": nodes},
        edge_counts={"calls": edges},
        unresolved_edges=unresolved,
    )


def manifest(tmp_path, **graph_fields) -> IndexManifest:
    manifest = IndexManifest.create(
        index_name="demo",
        repository_root=tmp_path,
        embedding_provider="local",
        embedding_model="mock",
        embedding_dimension=384,
    )
    for key, value in graph_fields.items():
        setattr(manifest, key, value)
    return manifest


def seed_store(tmp_path) -> list[GraphNode]:
    index_path = tmp_path / ".ctxai" / "indexes" / "demo"
    index_path.mkdir(parents=True, exist_ok=True)
    store = GraphStore(index_path)
    run = derive_node_id(REPO, "a.py", "function", "a.run")
    helper = derive_node_id(REPO, "a.py", "function", "a.helper")
    nodes = [
        GraphNode(
            id=run,
            kind="function",
            qualified_name="a.run",
            display_name="run",
            language="python",
            file_path="a.py",
            start_line=1,
            end_line=2,
        ),
        GraphNode(
            id=helper,
            kind="function",
            qualified_name="a.helper",
            display_name="helper",
            language="python",
            file_path="a.py",
            start_line=4,
            end_line=5,
        ),
    ]
    store.replace_all(nodes, [], metadata(generation=3, nodes=2))
    return nodes


def seeded_index_path(tmp_path):
    """Return the index directory seed_store built."""
    return tmp_path / ".ctxai" / "indexes" / "demo"


class TestGraphHealth:
    def test_missing_graph_without_manifest_reference_is_diagnostic_only(self, tmp_path):
        health = graph_health(tmp_path, manifest(tmp_path))
        assert health.status == "missing"
        assert health.problems == ()
        assert health.metadata is None

    def test_healthy_graph_matches_manifest(self, tmp_path):
        seed_store(tmp_path)
        health = graph_health(
            seeded_index_path(tmp_path),
            manifest(
                tmp_path,
                graph_schema_version=GRAPH_SCHEMA_VERSION,
                graph_extractor_version="python/1",
                graph_generation=3,
                graph_node_count=2,
                graph_edge_count=0,
            ),
        )
        assert health.status == "healthy"
        assert health.problems == ()
        assert health.metadata is not None
        assert health.metadata.generation == 3

    def test_missing_graph_file_referenced_by_manifest_is_a_problem(self, tmp_path):
        health = graph_health(
            tmp_path,
            manifest(tmp_path, graph_generation=2, graph_node_count=2, graph_edge_count=0),
        )
        assert health.status == "mismatch"
        assert any("missing" in problem for problem in health.problems)

    def test_generation_mismatch_is_detected(self, tmp_path):
        seed_store(tmp_path)
        health = graph_health(
            seeded_index_path(tmp_path),
            manifest(
                tmp_path,
                graph_schema_version=GRAPH_SCHEMA_VERSION,
                graph_extractor_version="python/1",
                graph_generation=2,
                graph_node_count=2,
                graph_edge_count=0,
            ),
        )
        assert health.status == "mismatch"
        assert any("generation" in problem for problem in health.problems)

    def test_count_mismatch_is_detected(self, tmp_path):
        seed_store(tmp_path)
        health = graph_health(
            seeded_index_path(tmp_path),
            manifest(
                tmp_path,
                graph_schema_version=GRAPH_SCHEMA_VERSION,
                graph_extractor_version="python/1",
                graph_generation=3,
                graph_node_count=99,
                graph_edge_count=0,
            ),
        )
        assert health.status == "count_mismatch"
        assert any("count" in problem for problem in health.problems)

    def test_unsupported_schema_is_detected(self, tmp_path):
        seed_store(tmp_path)
        health = graph_health(
            seeded_index_path(tmp_path),
            manifest(
                tmp_path,
                graph_schema_version=GRAPH_SCHEMA_VERSION + 5,
                graph_extractor_version="python/1",
                graph_generation=3,
                graph_node_count=2,
                graph_edge_count=0,
            ),
        )
        assert health.status == "unsupported_schema"
        assert any("schema" in problem for problem in health.problems)

    def test_corrupt_store_is_detected(self, tmp_path):
        (tmp_path / "graph.sqlite3").write_bytes(b"not a database at all")
        health = graph_health(
            tmp_path,
            manifest(tmp_path, graph_generation=1, graph_node_count=0, graph_edge_count=0),
        )
        assert health.status == "corrupt"
        assert health.problems

    def test_extract_version_mismatch_is_reported(self, tmp_path):
        seed_store(tmp_path)
        health = graph_health(
            seeded_index_path(tmp_path),
            manifest(
                tmp_path,
                graph_schema_version=GRAPH_SCHEMA_VERSION,
                graph_extractor_version="python/0",
                graph_generation=3,
                graph_node_count=2,
                graph_edge_count=0,
            ),
        )
        assert health.status == "mismatch"
        assert any("extractor" in problem for problem in health.problems)


class TestBounds:
    def make_operations(self, tmp_path):
        seed_store(tmp_path)
        return GraphOperations(project_path=tmp_path)

    def test_symbol_query_length_is_bounded(self, tmp_path):
        operations = self.make_operations(tmp_path)
        with pytest.raises(ValueError, match="200"):
            operations.find_symbols("demo", "x" * 201)

    def test_empty_symbol_query_is_rejected(self, tmp_path):
        operations = self.make_operations(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            operations.find_symbols("demo", "   ")

    def test_unknown_kind_is_rejected(self, tmp_path):
        operations = self.make_operations(tmp_path)
        with pytest.raises(ValueError, match="kind"):
            operations.find_symbols("demo", "run", kind="galaxy")

    def test_unknown_language_is_rejected(self, tmp_path):
        operations = self.make_operations(tmp_path)
        with pytest.raises(ValueError, match="language"):
            operations.find_symbols("demo", "run", language="cobol")

    def test_result_limit_is_bounded(self, tmp_path):
        operations = self.make_operations(tmp_path)
        with pytest.raises(ValueError, match="limit"):
            operations.find_symbols("demo", "run", limit=501)

    def test_traversal_depth_is_bounded(self, tmp_path):
        operations = self.make_operations(tmp_path)
        run = derive_node_id(REPO, "a.py", "function", "a.run")
        with pytest.raises(ValueError, match="depth"):
            operations.neighbors("demo", run, depth=4)

    def test_neighbor_limit_is_bounded(self, tmp_path):
        operations = self.make_operations(tmp_path)
        run = derive_node_id(REPO, "a.py", "function", "a.run")
        with pytest.raises(ValueError, match="limit"):
            operations.neighbors("demo", run, limit=501)

    def test_direction_and_edge_kind_are_validated(self, tmp_path):
        operations = self.make_operations(tmp_path)
        run = derive_node_id(REPO, "a.py", "function", "a.run")
        with pytest.raises(ValueError, match="direction"):
            operations.neighbors("demo", run, direction="sideways")
        with pytest.raises(ValueError, match="edge kind"):
            operations.neighbors("demo", run, edge_kind="teleports")


class TestLookups:
    def make_operations(self, tmp_path):
        seed_store(tmp_path)
        return GraphOperations(project_path=tmp_path)

    def test_find_symbols_returns_nodes_with_evidence(self, tmp_path):
        operations = self.make_operations(tmp_path)
        results = operations.find_symbols("demo", "run")
        assert len(results) == 1
        assert results[0].evidence() == "a.py:1-2"
        assert results[0].qualified_name == "a.run"

    def test_find_symbols_kind_filter(self, tmp_path):
        operations = self.make_operations(tmp_path)
        assert operations.find_symbols("demo", "helper", kind="function")
        assert operations.find_symbols("demo", "helper", kind="class") == []

    def test_neighbors_returns_start_node_and_edges(self, tmp_path):
        operations = self.make_operations(tmp_path)
        run = derive_node_id(REPO, "a.py", "function", "a.run")
        result = operations.neighbors("demo", run, direction="in")
        assert result.truncated is False
        assert result.nodes[0].id == run

    def test_neighbors_resolves_unique_id_prefix(self, tmp_path):
        operations = self.make_operations(tmp_path)
        run = derive_node_id(REPO, "a.py", "function", "a.run")
        result = operations.neighbors("demo", run[:12])
        assert result.nodes[0].id == run

    def test_ambiguous_or_short_prefix_is_rejected(self, tmp_path):
        operations = self.make_operations(tmp_path)
        with pytest.raises(ValueError):
            operations.neighbors("demo", "ab")

    def test_missing_index_is_rejected(self, tmp_path):
        operations = GraphOperations(project_path=tmp_path)
        with pytest.raises(GraphError):
            operations.find_symbols("nope", "run")

    def test_missing_graph_data_is_reported(self, tmp_path):
        (tmp_path / ".ctxai" / "indexes" / "bare").mkdir(parents=True)
        operations = GraphOperations(project_path=tmp_path)
        with pytest.raises(GraphError, match="not been built"):
            operations.find_symbols("bare", "run")

    def test_stats_reports_health_and_metadata(self, tmp_path):
        seed_store(tmp_path)
        operations = GraphOperations(project_path=tmp_path)
        stats = operations.stats("demo")
        assert stats.health.status == "healthy"
        assert stats.metadata is not None
        assert stats.metadata.generation == 3

    def test_stats_on_missing_graph_returns_missing_health(self, tmp_path):
        (tmp_path / ".ctxai" / "indexes" / "bare").mkdir(parents=True)
        operations = GraphOperations(project_path=tmp_path)
        stats = operations.stats("bare")
        assert stats.health.status == "missing"
        assert stats.metadata is None


class TestIndexNameResolution:
    def test_explicit_name_wins(self, tmp_path):
        assert GraphOperations.resolve_index_name(tmp_path, "given") == "given"

    def test_missing_name_raises_helpful_error(self, tmp_path, monkeypatch):
        from ctxai.config import Config, ConfigManager

        monkeypatch.setattr(ConfigManager, "load", lambda self: Config())
        with pytest.raises(GraphError, match="index name"):
            GraphOperations.resolve_index_name(tmp_path, None)

    def test_configured_default_is_used(self, tmp_path, monkeypatch):
        from ctxai.config import Config, ConfigManager

        def fake_load(self):
            config = Config()
            config.index_name = "configured-index"
            return config

        monkeypatch.setattr(ConfigManager, "load", fake_load)
        assert GraphOperations.resolve_index_name(tmp_path, None) == "configured-index"


class TestGraphHealthModel:
    def test_health_dataclass_carries_status_and_problems(self):
        health = GraphHealth(status="missing", metadata=None, problems=(), diagnostic="d")
        assert health.problems == ()
        assert replace(health, status="healthy").status == "healthy"
