"""Unit tests for the IG-01 graph store: schema, transactions, queries, migration."""

from __future__ import annotations

import sqlite3

import pytest

from ctxai.graph.model import (
    CONFIDENCE_EXACT,
    CONFIDENCE_UNRESOLVED,
    GRAPH_FILENAME,
    GRAPH_SCHEMA_VERSION,
    GraphEdge,
    GraphMetadata,
    GraphNode,
    derive_edge_id,
    derive_node_id,
)
from ctxai.graph.store import GraphStore, GraphStoreError
from ctxai.index_manifest import IndexManifest

REPO = "/repos/demo"


def node(node_id: str, qualified: str, file_path: str = "a.py", kind: str = "function", **overrides):
    values = dict(
        id=node_id,
        kind=kind,
        qualified_name=qualified,
        display_name=qualified.rsplit(".", 1)[-1],
        language="python",
        file_path=file_path,
        start_line=1,
        end_line=2,
        parent_id=None,
        visibility="public",
        source_hash=None,
    )
    values.update(overrides)
    return GraphNode(**values)


def edge(edge_id: str, kind: str, source_id: str, target_id: str | None, file_path: str = "a.py", line: int = 1):
    return GraphEdge(
        id=edge_id,
        kind=kind,
        source_id=source_id,
        target_id=target_id,
        target_text=None if target_id else "unresolved_target",
        evidence_file=file_path,
        evidence_line=line,
        confidence=CONFIDENCE_EXACT if target_id else CONFIDENCE_UNRESOLVED,
        resolver_version="python/1",
    )


def metadata(generation: int = 1) -> GraphMetadata:
    return GraphMetadata(
        schema_version=GRAPH_SCHEMA_VERSION,
        extractor_version="python/1",
        resolver_version="python/1",
        supported_languages=["python"],
        built_at="2026-09-05T00:00:00+00:00",
        generation=generation,
        node_counts={},
        edge_counts={},
        unresolved_edges=0,
    )


def n(qual: str, file_path: str = "a.py", kind: str = "function") -> str:
    """Deterministic node id helper."""
    return derive_node_id(REPO, file_path, kind, qual)


def e(kind: str, source: str, target: str | None, file_path: str = "a.py", line: int = 1) -> str:
    return derive_edge_id(REPO, kind, source, target, None if target else "unresolved_target", file_path, line)


class TestFullPublication:
    def test_replace_all_publishes_metadata_counts_and_rows(self, tmp_path):
        store = GraphStore(tmp_path)
        a, b = n("a.run"), n("b.helper", "b.py")
        published = store.replace_all(
            [node(a, "a.run"), node(b, "b.helper", "b.py")],
            [edge(e("calls", a, b), "calls", a, b)],
            metadata(generation=1),
        )
        assert published.generation == 1
        assert published.node_counts == {"function": 2}
        assert published.edge_counts == {"calls": 1}
        assert published.unresolved_edges == 0
        assert store.count_nodes() == 2
        assert store.count_edges() == 1
        assert (tmp_path / GRAPH_FILENAME).exists()
        assert store.integrity_check() == "ok"

    def test_replace_all_failure_leaves_prior_graph_visible(self, tmp_path):
        store = GraphStore(tmp_path)
        a, b = n("a.run"), n("b.helper", "b.py")
        store.replace_all([node(a, "a.run"), node(b, "b.helper", "b.py")], [], metadata(generation=1))
        with pytest.raises(GraphStoreError):
            # A dangling edge source must abort the whole publication.
            store.replace_all(
                [node(n("c.x", "c.py"), "c.x", "c.py")],
                [edge(e("calls", "f" * 64, a), "calls", "f" * 64, a, "c.py")],
                metadata(generation=2),
            )
        assert store.count_nodes() == 2
        assert store.read_metadata().generation == 1
        assert not list(tmp_path.glob("*.tmp"))

    def test_foreign_keys_are_enforced(self, tmp_path):
        store = GraphStore(tmp_path)
        with pytest.raises(GraphStoreError, match="FOREIGN KEY|integrity"):
            store.replace_all(
                [node(n("a.run"), "a.run")],
                [edge(e("calls", "f" * 64, n("a.run")), "calls", "f" * 64, n("a.run"))],
                metadata(),
            )


class TestIncrementalPublication:
    def build_two_files(self, tmp_path):
        store = GraphStore(tmp_path)
        a, b = n("a.run"), n("b.helper", "b.py")
        store.replace_all(
            [node(a, "a.run"), node(b, "b.helper", "b.py")],
            [edge(e("calls", a, b), "calls", a, b)],
            metadata(generation=1),
        )
        return store, a, b

    def test_changed_file_replaces_only_owned_rows(self, tmp_path):
        store, a, b = self.build_two_files(tmp_path)
        new_a = n("a.run_v2")
        published = store.update_files(
            {"a.py": ([node(new_a, "a.run_v2")], [])},
            [],
            metadata(generation=2),
        )
        assert published.generation == 2
        assert store.count_nodes() == 2
        assert store.get_node(a) is None
        assert store.get_node(new_a) is not None
        assert store.get_node(b) is not None
        # The edge into the replaced file's old node is gone deterministically.
        assert store.count_edges() == 0

    def test_deleted_file_removes_nodes_and_touching_edges(self, tmp_path):
        store, a, b = self.build_two_files(tmp_path)
        store.update_files({"b.py": ([], [])}, [], metadata(generation=2))
        assert store.get_node(b) is None
        assert store.get_node(a) is not None
        assert store.count_edges() == 0

    def test_failed_transaction_leaves_prior_graph_and_generation(self, tmp_path):
        store, a, b = self.build_two_files(tmp_path)
        with pytest.raises(GraphStoreError):
            store.update_files(
                {
                    "a.py": (
                        [node(n("a.v2"), "a.v2")],
                        [edge(e("calls", "f" * 64, n("a.v2")), "calls", "f" * 64, n("a.v2"))],
                    )
                },
                [],
                metadata(generation=2),
            )
        assert store.count_nodes() == 2
        assert store.get_node(a) is not None
        assert store.count_edges() == 1
        assert store.read_metadata().generation == 1

    def test_files_with_edges_into_finds_dependents(self, tmp_path):
        store, a, b = self.build_two_files(tmp_path)
        assert store.files_with_edges_into(["b.py"]) == {"a.py"}
        assert store.files_with_edges_into(["a.py"]) == set()
        assert store.files_with_edges_into(["missing.py"]) == set()


class TestQueries:
    def seed(self, tmp_path):
        store = GraphStore(tmp_path)
        run = n("a.run")
        helper = n("a.helper")
        mod = n("a", kind="module")
        ext = n("ext.fn", "ext.py")
        rows = [
            node(mod, "a", kind="module"),
            node(run, "a.run"),
            node(helper, "a.helper"),
            node(ext, "ext.fn", "ext.py"),
        ]
        edges = [
            edge(e("contains", mod, run), "contains", mod, run),
            edge(e("calls", run, helper), "calls", run, helper),
            edge(e("references", helper, ext, "a.py", 9), "references", helper, ext, "a.py", 9),
        ]
        store.replace_all(rows, edges, metadata())
        return store, mod, run, helper, ext

    def test_find_nodes_substring_and_filters(self, tmp_path):
        store, *_ = self.seed(tmp_path)
        assert [n.qualified_name for n in store.find_nodes("run")] == ["a.run"]
        assert {n.qualified_name for n in store.find_nodes("a")} == {"a", "a.run", "a.helper"}
        assert [n.qualified_name for n in store.find_nodes("a", kind="module")] == ["a"]
        assert [n.qualified_name for n in store.find_nodes("fn", language="python")] == ["ext.fn"]
        assert store.find_nodes("zzz") == []

    def test_find_nodes_escapes_like_wildcards(self, tmp_path):
        store = GraphStore(tmp_path)
        weird = n("a.we%ird_name")
        store.replace_all([node(weird, "a.we%ird_name"), node(n("a.x1"), "a.x1")], [], metadata())
        assert [n.qualified_name for n in store.find_nodes("%")] == ["a.we%ird_name"]
        assert [n.qualified_name for n in store.find_nodes("_")] == ["a.we%ird_name"]

    def test_find_nodes_limit_is_bounded(self, tmp_path):
        store, *_ = self.seed(tmp_path)
        assert len(store.find_nodes("a", limit=2)) == 2

    def test_get_node_missing_returns_none(self, tmp_path):
        store = GraphStore(tmp_path)
        store.replace_all([], [], metadata())
        assert store.get_node("f" * 64) is None

    def test_neighbors_direction_depth_and_kind(self, tmp_path):
        store, mod, run, helper, ext = self.seed(tmp_path)
        out = store.neighbors(run, direction="out")
        assert [n.id for n in out.nodes] == [run, helper]
        assert [ed.kind for ed in out.edges] == ["calls"]
        assert out.truncated is False

        inc = store.neighbors(helper, direction="in")
        assert {n.id for n in inc.nodes} == {helper, run}

        both = store.neighbors(run, direction="both")
        assert {n.id for n in both.nodes} == {mod, run, helper}

        depth2 = store.neighbors(run, direction="both", depth=2)
        assert ext in {n.id for n in depth2.nodes}

        only_contains = store.neighbors(mod, edge_kind="contains", direction="out")
        assert {n.id for n in only_contains.nodes} == {mod, run}

    def test_neighbors_limit_truncates_deterministically(self, tmp_path):
        store, *_ = self.seed(tmp_path)
        result = store.neighbors(n("a", kind="module"), direction="out", depth=3, limit=2)
        assert result.truncated is True
        assert len(result.nodes) <= 2

    def test_neighbors_of_missing_node_returns_empty(self, tmp_path):
        store = GraphStore(tmp_path)
        store.replace_all([], [], metadata())
        result = store.neighbors("f" * 64)
        assert result.nodes == [] and result.edges == [] and result.truncated is False


class TestMetadataAndHealth:
    def test_read_metadata_round_trip(self, tmp_path):
        store = GraphStore(tmp_path)
        store.replace_all([], [], metadata(generation=7))
        loaded = store.read_metadata()
        assert loaded.generation == 7
        assert loaded.schema_version == GRAPH_SCHEMA_VERSION
        assert loaded.extractor_version == "python/1"
        assert loaded.supported_languages == ["python"]

    def test_unsupported_schema_is_rejected(self, tmp_path):
        store = GraphStore(tmp_path)
        store.replace_all([], [], metadata())
        conn = sqlite3.connect(tmp_path / GRAPH_FILENAME)
        conn.execute("UPDATE graph_meta SET value = '999' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()
        with pytest.raises(GraphStoreError, match="Unsupported graph schema"):
            store.read_metadata()

    def test_corrupt_store_raises(self, tmp_path):
        (tmp_path / GRAPH_FILENAME).write_bytes(b"this is not a database")
        store = GraphStore(tmp_path)
        with pytest.raises(GraphStoreError, match="corrupt|not a database"):
            store.read_metadata()
        with pytest.raises(GraphStoreError):
            store.count_nodes()

    def test_missing_store_raises_on_read(self, tmp_path):
        store = GraphStore(tmp_path)
        with pytest.raises(GraphStoreError, match="missing"):
            store.read_metadata()
        assert store.exists() is False

    def test_generation_increments_across_publications(self, tmp_path):
        store = GraphStore(tmp_path)
        store.replace_all([], [], metadata(generation=1))
        store.update_files({}, [], metadata(generation=2))
        assert store.read_metadata().generation == 2


class TestManifestMigration:
    def base_manifest(self, tmp_path):
        return IndexManifest.create(
            index_name="demo",
            repository_root=tmp_path,
            embedding_provider="local",
            embedding_model="mock",
            embedding_dimension=384,
        )

    def test_old_manifest_without_graph_fields_loads_with_nones(self, tmp_path):
        manifest = self.base_manifest(tmp_path)
        manifest.save(tmp_path)
        loaded = IndexManifest.load(tmp_path)
        assert loaded.graph_schema_version is None
        assert loaded.graph_extractor_version is None
        assert loaded.graph_generation is None
        assert loaded.graph_node_count is None
        assert loaded.graph_edge_count is None

    def test_manifest_graph_fields_round_trip(self, tmp_path):
        manifest = self.base_manifest(tmp_path)
        manifest.graph_schema_version = GRAPH_SCHEMA_VERSION
        manifest.graph_extractor_version = "python/1"
        manifest.graph_generation = 4
        manifest.graph_node_count = 12
        manifest.graph_edge_count = 20
        manifest.save(tmp_path)
        loaded = IndexManifest.load(tmp_path)
        assert loaded.graph_generation == 4
        assert loaded.graph_node_count == 12
        assert loaded.graph_edge_count == 20
        assert loaded.graph_schema_version == GRAPH_SCHEMA_VERSION
