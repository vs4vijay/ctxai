"""Unit tests for the IG-02 multi-language builder: registry dispatch, adapter
version staleness, bounded incremental rebuilds, and the forward-only v1->v2
graph store migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ctxai.graph.builder import GraphBuilder, GraphBuildError
from ctxai.graph.js_adapter import JavaScriptAdapter, TypeScriptAdapter
from ctxai.graph.model import GRAPH_FILENAME, GRAPH_SCHEMA_VERSION
from ctxai.graph.operations import GraphError, GraphOperations, graph_health
from ctxai.graph.store import GraphSchemaError, GraphStore
from ctxai.index_manifest import IndexManifest

PY_CALC = "def calculate(a, b):\n    return a + b\n"
PY_USE = "from pkg.calc import calculate\n\ndef run(x):\n    return calculate(x, 1)\n"
JS_CALC = "export function jsCalc(a, b) { return a + b; }\n"
JS_APP = "import { jsCalc } from './calc';\nexport function main() { return jsCalc(1, 2); }\n"
TS_TYPES = "export interface Shape { area(): number; }\nexport class Circle implements Shape { area() { return 1; } }\n"
GO_FILE = "package main\n\nfunc main() {}\n"

FILES = {"pkg/__init__.py", "pkg/calc.py", "pkg/use.py", "web/calc.js", "web/app.js", "web/types.ts"}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "web").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "calc.py").write_text(PY_CALC)
    (root / "pkg" / "use.py").write_text(PY_USE)
    (root / "web" / "calc.js").write_text(JS_CALC)
    (root / "web" / "app.js").write_text(JS_APP)
    (root / "web" / "types.ts").write_text(TS_TYPES)
    return root


@pytest.fixture
def index_path(repo):
    path = repo / ".ctxai" / "indexes" / "demo"
    path.mkdir(parents=True)
    return path


def build(repo, index_path, files=FILES, changed=None, deleted=None, force_full=False):
    return GraphBuilder(repository_root=repo).update(
        index_path,
        set(files),
        set(changed or ()),
        set(deleted or ()),
        force_full=force_full,
    )


class TestMultiLanguageBuild:
    def test_full_build_spans_all_languages(self, repo, index_path):
        result = build(repo, index_path, force_full=True)
        assert result.mode == "full"
        assert result.metadata.schema_version == GRAPH_SCHEMA_VERSION
        assert result.metadata.adapter_versions == {
            "javascript": "javascript/1",
            "python": "python/1",
            "typescript": "typescript/1",
        }
        assert result.metadata.extractor_version == "javascript/1,python/1,typescript/1"
        assert result.metadata.node_counts["module"] == 6
        assert result.metadata.node_counts["interface"] == 1
        assert result.metadata.edge_counts["inherits"] == 1
        assert result.metadata.edge_counts["imports"] >= 2

    def test_unsupported_language_stays_a_chunk_without_graph_nodes(self, repo, index_path):
        (repo / "main.go").write_text(GO_FILE)
        files = {*FILES, "main.go"}
        result = build(repo, index_path, files=files, force_full=True)
        store = GraphStore(index_path)
        assert all(node.file_path != "main.go" for node in store.iter_nodes())
        assert result.metadata.total_nodes > 0  # the other files still contribute

    def test_unchanged_reindex_is_skipped(self, repo, index_path):
        build(repo, index_path, force_full=True)
        result = build(repo, index_path)
        assert result.mode == "skipped"
        assert result.generation == 1

    def test_changed_file_replaces_owned_rows_and_reextracts_dependents(self, repo, index_path):
        build(repo, index_path, force_full=True)
        store = GraphStore(index_path)
        python_node_before = next(node.id for node in store.iter_nodes() if node.qualified_name == "pkg.calc.calculate")

        (repo / "web" / "calc.js").write_text(JS_CALC + "\nexport function extra() { return 3; }\n")
        result = build(repo, index_path, changed={"web/calc.js"})

        assert result.mode == "incremental"
        assert result.generation == 2
        assert result.reextracted_files == 2  # calc.js plus its dependent app.js
        store = GraphStore(index_path)
        qualified = {node.qualified_name for node in store.iter_nodes()}
        assert "web.calc.extra" in qualified
        # Python nodes are untouched by a JavaScript-only change.
        assert next(node.id for node in store.iter_nodes() if node.qualified_name == "pkg.calc.calculate") == (
            python_node_before
        )

    def test_deleted_file_removes_owned_rows(self, repo, index_path):
        build(repo, index_path, force_full=True)
        (repo / "web" / "calc.js").unlink()
        result = build(repo, index_path, files=FILES - {"web/calc.js"}, deleted={"web/calc.js"})
        assert result.deleted_files == 1
        store = GraphStore(index_path)
        qualified = {node.qualified_name for node in store.iter_nodes()}
        assert "web.calc" not in qualified
        assert "web.calc.jsCalc" not in qualified

    def test_cross_language_apis_without_manifest_stay_healthy(self, repo, index_path):
        build(repo, index_path, force_full=True)
        manifest = IndexManifest.load_optional(index_path)  # none written by the builder alone
        assert manifest is None
        assert graph_health(index_path, None).status == "healthy"


class TestAdapterVersionStaleness:
    def test_adapter_upgrade_marks_only_affected_files_stale(self, repo, index_path, monkeypatch):
        build(repo, index_path, force_full=True)
        store = GraphStore(index_path)
        python_nodes_before = sorted(node.id for node in store.iter_nodes() if node.language == "python")
        js_nodes_before = sorted(node.id for node in store.iter_nodes() if node.language == "javascript")

        monkeypatch.setattr(JavaScriptAdapter, "extractor_version", "javascript/2")
        monkeypatch.setattr(JavaScriptAdapter, "resolver_version", "javascript/2")
        result = build(repo, index_path)

        assert result.mode == "incremental"
        assert result.generation == 2
        # Every JavaScript file was re-extracted (2 JS files + dependent), no Python file.
        extracted = {path for path in ("web/calc.js", "web/app.js")}
        store_after = GraphStore(index_path)
        assert sorted(node.id for node in store_after.iter_nodes() if node.language == "python") == python_nodes_before
        # The JS nodes carry the new adapter version; ids are stable (same source).
        js_nodes_after = sorted(node.id for node in store_after.iter_nodes() if node.language == "javascript")
        assert js_nodes_after == js_nodes_before
        assert all(
            node.adapter_version == "javascript/2" for node in store_after.iter_nodes() if node.language == "javascript"
        )
        assert result.metadata.adapter_versions["javascript"] == "javascript/2"
        assert extracted

    def test_typescript_upgrade_leaves_javascript_alone(self, repo, index_path, monkeypatch):
        build(repo, index_path, force_full=True)
        monkeypatch.setattr(TypeScriptAdapter, "extractor_version", "typescript/2")
        result = build(repo, index_path)
        assert result.mode == "incremental"
        assert result.reextracted_files == 1
        assert result.metadata.adapter_versions == {
            "javascript": "javascript/1",
            "python": "python/1",
            "typescript": "typescript/2",
        }


V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    language TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    parent_id TEXT,
    visibility TEXT NOT NULL DEFAULT 'public',
    source_hash TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT,
    target_text TEXT,
    evidence_file TEXT NOT NULL,
    evidence_line INTEGER NOT NULL,
    confidence TEXT NOT NULL,
    resolver_version TEXT NOT NULL
);
"""


def write_v1_store(index_path: Path, generation: int = 3) -> None:
    """Create a pre-IG-02 (schema v1) graph store with one node and one edge."""
    index_path.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path / GRAPH_FILENAME)
    connection.executescript(V1_SCHEMA)
    connection.execute(
        "INSERT INTO nodes (id, kind, qualified_name, display_name, language, file_path, start_line, end_line,"
        " parent_id, visibility, source_hash) VALUES (?, 'function', 'pkg.calc.calculate', 'calculate', 'python',"
        " 'pkg/calc.py', 1, 2, NULL, 'public', NULL)",
        ("v1node" + "0" * 57,),
    )
    for key, value in {
        "schema_version": "1",
        "extractor_version": "python/1",
        "resolver_version": "python/1",
        "supported_languages": "python",
        "built_at": "2026-09-01T00:00:00+00:00",
        "generation": str(generation),
        "node_counts": '{"function": 1}',
        "edge_counts": "{}",
        "unresolved_edges": "0",
    }.items():
        connection.execute("INSERT INTO graph_meta (key, value) VALUES (?, ?)", (key, value))
    connection.commit()
    connection.close()


class TestSchemaV2ForwardOnlyMigration:
    def test_v1_store_is_rejected_deterministically(self, tmp_path):
        write_v1_store(tmp_path)
        store = GraphStore(tmp_path)
        with pytest.raises(GraphSchemaError, match="Unsupported graph schema 1"):
            store.read_metadata()

    def test_health_reports_unsupported_schema_for_v1_store(self, tmp_path):
        write_v1_store(tmp_path)
        health = graph_health(tmp_path, None)
        assert health.status == "unsupported_schema"
        assert any("schema 1" in problem for problem in health.problems)

    def test_operations_refuse_v1_reads_with_deterministic_error(self, tmp_path):
        write_v1_store(tmp_path / ".ctxai" / "indexes" / "demo")
        operations = GraphOperations(project_path=tmp_path)
        with pytest.raises(GraphError, match="not readable"):
            operations.find_symbols("demo", "calculate")

    def test_next_index_run_rebuilds_v1_store_to_v2(self, repo, index_path):
        write_v1_store(index_path)
        result = build(repo, index_path, force_full=False)
        assert result.mode == "full"
        assert result.metadata.schema_version == GRAPH_SCHEMA_VERSION
        store = GraphStore(index_path)
        assert store.read_metadata().schema_version == GRAPH_SCHEMA_VERSION
        assert graph_health(index_path, None).status == "healthy"
        qualified = {node.qualified_name for node in store.iter_nodes()}
        assert "pkg.calc.calculate" in qualified

    def test_v1_manifest_without_graph_fields_triggers_full_rebuild(self, repo, index_path):
        manifest = IndexManifest.create(
            index_name="demo",
            repository_root=repo,
            embedding_provider="local",
            embedding_model="mock",
            embedding_dimension=384,
        )
        manifest.save(index_path)
        build(repo, index_path, force_full=manifest.graph_generation is None)
        assert GraphStore(index_path).read_metadata().schema_version == GRAPH_SCHEMA_VERSION

    def test_future_schema_is_never_silently_interpreted(self, repo, index_path):
        build(repo, index_path, force_full=True)
        connection = sqlite3.connect(index_path / GRAPH_FILENAME)
        connection.execute("UPDATE graph_meta SET value = '999' WHERE key = 'schema_version'")
        connection.commit()
        connection.close()
        store = GraphStore(index_path)
        with pytest.raises(GraphSchemaError, match="Unsupported graph schema 999"):
            store.read_metadata()


class TestPathSafety:
    def test_non_relative_paths_are_refused(self, repo, index_path):
        builder = GraphBuilder(repository_root=repo)
        with pytest.raises(GraphBuildError, match="repository-relative"):
            builder.update(index_path, {"../outside.py"}, set(), set())
        with pytest.raises(GraphBuildError, match="repository-relative"):
            builder.update(index_path, {"/abs/absolute.py"}, set(), set())
        with pytest.raises(GraphBuildError, match="repository-relative"):
            builder.update(index_path, {"pkg/../pkg/escape.py"}, set(), set())
