"""Acceptance coverage for IG-01: inspectable symbol graph for one repository."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.commands.index_command import index_codebase
from ctxai.commands.indexes_command import doctor_index
from ctxai.graph.model import GRAPH_FILENAME, GRAPH_SCHEMA_VERSION
from ctxai.graph.operations import graph_health
from ctxai.graph.store import GraphStore
from ctxai.index_manifest import MANIFEST_FILENAME, IndexManifest

INDEX_NAME = "graphdemo"

PKG_INIT = '"""Demo package."""\n'

PKG_CALC = '''"""Simple calculation helpers."""


def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

PKG_MODELS = '''"""Model classes."""


class Animal:
    def speak(self) -> str:
        return "..."

    def _hidden(self) -> int:
        return 1


class Dog(Animal):
    def speak(self) -> str:
        return "woof"
'''

PKG_SERVICE = '''"""Service entry points."""

from .calc import calculate
from .models import Animal, Dog

import pkg.calc as pc


def run(a: int, b: int) -> int:
    total = calculate(a, b)
    return pc.calculate(total, 1)
'''

MAIN = '''"""Demo entry point."""

import pkg.service
from pkg.calc import calculate


def main() -> int:
    value = pkg.service.run(1, 2)
    return calculate(value, 3)
'''

DUP_A = 'def process() -> str:\n    return "a"\n'
DUP_B = 'def process() -> str:\n    return "b"\n'

AMB = '''"""Ambiguous caller."""


def caller() -> str:
    return process()
'''

DYN = '''"""Dynamic dispatch."""


def dispatch(fn):
    return fn()
'''

NESTED = """def outer():
    def inner():
        return 1

    return inner()


class Shell:
    def run(self) -> int:
        return self._private()

    def _private(self) -> int:
        return 2
"""

TEST_SERVICE = """from pkg.service import run


def test_run():
    assert run(1, 2) is not None
"""

BROKEN = "def broken(:\n    pass\n"


def write_graph_project(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text(PKG_INIT)
    (root / "pkg" / "calc.py").write_text(PKG_CALC)
    (root / "pkg" / "models.py").write_text(PKG_MODELS)
    (root / "pkg" / "service.py").write_text(PKG_SERVICE)
    (root / "main.py").write_text(MAIN)
    (root / "dup_a.py").write_text(DUP_A)
    (root / "dup_b.py").write_text(DUP_B)
    (root / "amb.py").write_text(AMB)
    (root / "dyn.py").write_text(DYN)
    (root / "nested.py").write_text(NESTED)
    (root / "tests" / "test_service.py").write_text(TEST_SERVICE)
    (root / "broken.py").write_text(BROKEN)


@pytest.fixture
def graph_project(temp_dir):
    root = temp_dir / "repo"
    root.mkdir()
    write_graph_project(root)
    return root


@pytest.fixture
def indexes_dir(temp_dir, monkeypatch):
    """Route the in-process pipeline to a temp indexes directory."""
    monkeypatch.delenv("CTXAI_HOME", raising=False)
    directory = temp_dir / ".ctxai" / "indexes"
    monkeypatch.setattr("ctxai.commands.index_command.get_indexes_dir", lambda _path: directory)
    return directory


def index(graph_project, indexes_dir, name=INDEX_NAME):
    return index_codebase(graph_project, name, ["*.py"], follow_gitignore=False)


pytest_plugins = []  # fixtures come from tests/e2e/conftest.py


def run_cli(args: list[str], ctxai_home: Path) -> subprocess.CompletedProcess[str]:
    """Run the real CLI in a fresh process against a temp CTXAI_HOME."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    environment["CTXAI_HOME"] = str(ctxai_home)
    return subprocess.run(
        [sys.executable, "-m", "ctxai", *args],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(ctxai_home),
    )


@pytest.mark.e2e
@pytest.mark.indexing
def test_graph_publishes_matching_generations_atomically(graph_project, indexes_dir, patch_embeddings_factory):
    """Acceptance 1: vector and graph generations publish under one manifest."""
    result = index(graph_project, indexes_dir)
    assert result.files == 12
    index_path = indexes_dir / INDEX_NAME
    manifest = IndexManifest.load(index_path)
    assert manifest.graph_schema_version == GRAPH_SCHEMA_VERSION
    assert manifest.graph_generation == 1
    assert manifest.graph_node_count > 0
    assert manifest.graph_edge_count > manifest.graph_node_count  # contains edges dominate

    health = graph_health(index_path, manifest)
    assert health.status == "healthy"
    assert health.metadata.generation == manifest.graph_generation
    assert health.metadata.total_nodes == manifest.graph_node_count
    assert health.metadata.total_edges == manifest.graph_edge_count

    # The vector store matches the same manifest revision.
    from ctxai.vector_store import VectorStore

    assert VectorStore(index_path, INDEX_NAME).get_stats()["total_chunks"] == manifest.chunk_count
    assert doctor_index(INDEX_NAME, indexes_dir.parent.parent).healthy


@pytest.mark.e2e
@pytest.mark.indexing
def test_fresh_process_locates_definitions_and_relationships(
    graph_project, indexes_dir, patch_embeddings_factory, temp_dir
):
    """Acceptance 2 + 3: a new process finds definitions, relationships, and evidence."""
    index(graph_project, indexes_dir)
    ctxai_home = temp_dir / ".ctxai"

    stats = run_cli(["graph", "stats", INDEX_NAME, "--json"], ctxai_home)
    assert stats.returncode == 0, stats.stderr
    stats_envelope = json.loads(stats.stdout)
    assert stats_envelope["schema_version"] == 1
    assert stats_envelope["health"]["status"] == "healthy"
    assert stats_envelope["graph"]["generation"] == 1
    assert stats_envelope["graph"]["node_counts"]["module"] >= 9
    assert stats_envelope["graph"]["edge_counts"]["contains"] > 0

    symbol = run_cli(
        ["graph", "symbol", "run", "--kind", "function", "--language", "python", "--json", INDEX_NAME], ctxai_home
    )
    assert symbol.returncode == 0, symbol.stderr
    symbol_envelope = json.loads(symbol.stdout)
    matches = {item["qualified_name"]: item for item in symbol_envelope["symbols"]}
    assert "pkg.service.run" in matches
    run_node = matches["pkg.service.run"]
    assert run_node["file_path"] == "pkg/service.py"
    assert run_node["evidence"] == "pkg/service.py:9-11"
    assert run_node["kind"] == "function"

    neighbors = run_cli(["graph", "neighbors", run_node["id"], "--json", "--depth", "3", INDEX_NAME], ctxai_home)
    assert neighbors.returncode == 0, neighbors.stderr
    envelope = json.loads(neighbors.stdout)
    nodes = {node["qualified_name"] for node in envelope["nodes"]}
    edges = envelope["edges"]
    assert envelope["schema_version"] == 1
    node_names = {node["id"]: node["qualified_name"] for node in envelope["nodes"]}
    # Imports: importers of this module surface as exact import edges.
    assert any(edge["kind"] == "imports" and edge["confidence"] == "exact" for edge in edges)
    # Callers and callees across files.
    callers = {
        node_names[edge["source_id"]]
        for edge in edges
        if edge["kind"] == "calls" and edge["target_id"] == run_node["id"]
    }
    assert "main.main" in callers
    assert "tests.test_service.test_run" in callers
    # Every node and edge carries repository-relative evidence.
    for node in envelope["nodes"]:
        assert node["file_path"] and node["start_line"] >= 1 and node["end_line"] >= node["start_line"]
    for edge in edges:
        assert edge["evidence_file"] and edge["evidence_line"] >= 1
        assert edge["confidence"] in ("exact", "probable", "unresolved")
    # Depth-2 traversal reaches the importing test module through containment.
    assert "tests.test_service" in nodes

    tests_only = run_cli(
        ["graph", "neighbors", run_node["id"], "--edge", "tests", "--json", "--direction", "in", INDEX_NAME], ctxai_home
    )
    assert tests_only.returncode == 0, tests_only.stderr
    tests_envelope = json.loads(tests_only.stdout)
    tests_edges = [edge for edge in tests_envelope["edges"]]
    assert tests_edges and all(edge["kind"] == "tests" for edge in tests_edges)
    assert {node["qualified_name"] for node in tests_envelope["nodes"]} >= {"tests.test_service.test_run"}

    doctor = run_cli(["indexes", "doctor", INDEX_NAME], ctxai_home)
    assert doctor.returncode == 0, doctor.stderr
    assert "Graph: healthy" in doctor.stdout


@pytest.mark.e2e
@pytest.mark.indexing
def test_graph_symbol_cli_reports_ambiguity_and_unresolved_edges(
    graph_project, indexes_dir, patch_embeddings_factory, temp_dir
):
    index(graph_project, indexes_dir)
    ctxai_home = temp_dir / ".ctxai"

    process_matches = run_cli(["graph", "symbol", "process", "--json", INDEX_NAME], ctxai_home)
    assert process_matches.returncode == 0, process_matches.stderr
    envelope = json.loads(process_matches.stdout)
    assert {item["qualified_name"] for item in envelope["symbols"]} == {"dup_a.process", "dup_b.process"}

    caller = run_cli(["graph", "symbol", "caller", "--json", INDEX_NAME], ctxai_home)
    caller_id = json.loads(caller.stdout)["symbols"][0]["id"]
    amb_neighbors = run_cli(["graph", "neighbors", caller_id, "--json", INDEX_NAME], ctxai_home)
    amb_envelope = json.loads(amb_neighbors.stdout)
    unresolved_calls = [
        edge for edge in amb_envelope["edges"] if edge["kind"] == "calls" and edge["confidence"] == "unresolved"
    ]
    assert [edge["target_text"] for edge in unresolved_calls] == ["process"]

    dispatch = run_cli(["graph", "symbol", "dispatch", "--json", INDEX_NAME], ctxai_home)
    dispatch_id = json.loads(dispatch.stdout)["symbols"][0]["id"]
    dyn_neighbors = run_cli(["graph", "neighbors", dispatch_id, "--json", INDEX_NAME], ctxai_home)
    dyn_unresolved = [
        edge
        for edge in json.loads(dyn_neighbors.stdout)["edges"]
        if edge["kind"] == "calls" and edge["confidence"] == "unresolved"
    ]
    assert [edge["target_text"] for edge in dyn_unresolved] == ["fn"]


@pytest.mark.e2e
@pytest.mark.indexing
def test_unchanged_reindex_performs_zero_graph_mutations(graph_project, indexes_dir, patch_embeddings_factory):
    """Acceptance 4 (first half): re-indexing unchanged files mutates nothing."""
    index(graph_project, indexes_dir)
    index_path = indexes_dir / INDEX_NAME
    store = GraphStore(index_path)
    nodes_before = sorted(node.id for node in store.iter_nodes())
    generation_before = store.read_metadata().generation

    result = index(graph_project, indexes_dir)
    assert result.embedded_chunks == 0

    after = IndexManifest.load(index_path)
    assert after.graph_generation == generation_before
    assert sorted(node.id for node in store.iter_nodes()) == nodes_before


@pytest.mark.e2e
@pytest.mark.indexing
def test_changed_and_deleted_files_replace_only_owned_nodes(graph_project, indexes_dir, patch_embeddings_factory):
    """Acceptance 4 (second half): per-file ownership with dangling-edge removal."""
    index(graph_project, indexes_dir)
    index_path = indexes_dir / INDEX_NAME
    store = GraphStore(index_path)
    generation_before = store.read_metadata().generation

    service = graph_project / "pkg" / "service.py"
    service.write_text(service.read_text() + "\n\ndef extra() -> int:\n    return 1\n")
    (graph_project / "dup_b.py").unlink()

    result = index(graph_project, indexes_dir)
    assert result.changed_files == 1
    assert result.deleted_files == 1

    manifest = IndexManifest.load(index_path)
    assert manifest.graph_generation == generation_before + 1
    health = graph_health(index_path, manifest)
    assert health.status == "healthy"

    # The deleted file's node is gone; the surviving duplicate remains.
    qualified = {node.qualified_name for node in store.iter_nodes()}
    assert "dup_b.process" not in qualified
    assert "dup_a.process" in qualified
    assert "pkg.service.extra" in qualified

    # Cross-file edges into the changed file were deterministically rebuilt.
    run_node = next(node for node in store.iter_nodes() if node.qualified_name == "pkg.service.run")
    main_node = next(node for node in store.iter_nodes() if node.qualified_name == "main.main")
    neighbor = store.neighbors(run_node.id, edge_kind="calls", direction="in", depth=1)
    assert main_node.id in {node.id for node in neighbor.nodes}

    # The ambiguous call to the deleted duplicate stays honestly unresolved.
    caller_node = next(node for node in store.iter_nodes() if node.qualified_name == "amb.caller")
    amb = store.neighbors(caller_node.id, edge_kind="calls", direction="out", depth=1)
    assert [(edge.target_id, edge.target_text) for edge in amb.edges] == [(None, "process")]


@pytest.mark.e2e
@pytest.mark.indexing
def test_injected_failures_cannot_publish_a_current_manifest(graph_project, indexes_dir, patch_embeddings_factory):
    """Acceptance 5: extraction or storage failure keeps prior graph and manifest."""
    from unittest.mock import patch

    index(graph_project, indexes_dir)
    index_path = indexes_dir / INDEX_NAME
    manifest_before = json.loads((index_path / MANIFEST_FILENAME).read_text())
    generation_before = GraphStore(index_path).read_metadata().generation

    (graph_project / "pkg" / "calc.py").write_text(PKG_CALC + "\n\ndef another():\n    return 2\n")

    from ctxai.graph.python_adapter import PythonAdapter
    from ctxai.graph.store import GraphStoreError

    with patch.object(PythonAdapter, "extract_file", side_effect=RuntimeError("injected extraction failure")):
        with pytest.raises(RuntimeError, match="injected extraction failure"):
            index(graph_project, indexes_dir)
    manifest_after = json.loads((index_path / MANIFEST_FILENAME).read_text())
    assert manifest_after["graph_generation"] == manifest_before["graph_generation"]
    assert GraphStore(index_path).read_metadata().generation == generation_before

    with patch.object(GraphStore, "update_files", side_effect=GraphStoreError("injected storage failure")):
        with pytest.raises(GraphStoreError, match="injected storage failure"):
            index(graph_project, indexes_dir)
    manifest_after = json.loads((index_path / MANIFEST_FILENAME).read_text())
    assert manifest_after["graph_generation"] == manifest_before["graph_generation"]
    assert GraphStore(index_path).read_metadata().generation == generation_before
    assert graph_health(index_path, IndexManifest.load(index_path)).status == "healthy"


@pytest.mark.e2e
@pytest.mark.indexing
def test_doctor_detects_graph_problems_and_exits_nonzero(graph_project, indexes_dir, patch_embeddings_factory):
    """Acceptance 6: revision mismatch, corruption, schema, and counts."""
    index(graph_project, indexes_dir)
    index_path = indexes_dir / INDEX_NAME
    runner = CliRunner()

    def manifest_payload() -> dict:
        return json.loads((index_path / MANIFEST_FILENAME).read_text())

    def save_manifest(payload: dict) -> None:
        (index_path / MANIFEST_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def doctor(name: str = INDEX_NAME):
        return runner.invoke(app, ["indexes", "doctor", name, "--project-path", str(indexes_dir.parent.parent)])

    # Baseline: healthy.
    healthy = doctor()
    assert healthy.exit_code == 0
    assert "Graph: healthy" in healthy.stdout

    # Count inconsistency.
    payload = manifest_payload()
    payload["graph_node_count"] = payload["graph_node_count"] + 5
    save_manifest(payload)
    count_mismatch = doctor()
    assert count_mismatch.exit_code == 1
    assert "count" in count_mismatch.output

    # Revision mismatch.
    payload = manifest_payload()
    payload["graph_generation"] = payload["graph_generation"] + 1
    save_manifest(payload)
    revision = doctor()
    assert revision.exit_code == 1
    assert "generation" in revision.output

    # Unsupported schema.
    payload = manifest_payload()
    payload["graph_schema_version"] = GRAPH_SCHEMA_VERSION + 7
    save_manifest(payload)
    unsupported = doctor()
    assert unsupported.exit_code == 1
    assert "schema" in unsupported.output

    # Corruption.
    (index_path / GRAPH_FILENAME).write_bytes(b"this is not a sqlite database")
    corrupt = doctor()
    assert corrupt.exit_code == 1
    assert "corrupt" in corrupt.output

    # A graph-less legacy manifest is a diagnostic, not a failure.
    payload = manifest_payload()
    for key in (
        "graph_schema_version",
        "graph_extractor_version",
        "graph_generation",
        "graph_node_count",
        "graph_edge_count",
    ):
        payload[key] = None
    save_manifest(payload)
    (index_path / GRAPH_FILENAME).unlink()
    missing = doctor()
    assert missing.exit_code == 0
    assert "not been built" in missing.stdout
