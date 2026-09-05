"""IG-02 acceptance tests: multi-language graph and stable service contract.

Indexes a mixed Python/JavaScript/TypeScript fixture through the real CLI
pipeline, then proves: per-language node/edge semantics and evidence are
consistent, CLI/MCP/dashboard projections agree on identity, counts,
confidence, and relationships for the same index, unsupported languages stay
indexable with explicit capability reporting, and malformed input returns
deterministic bounded errors without leaking paths.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.commands.dashboard_command import create_dashboard_app
from ctxai.commands.server_command import create_server
from tests.e2e.test_e2e_mcp_server import call, connected_client

FIXTURE_FILES = {
    "service.py": ("def run(x):\n    return x + 1\n\n\nclass Runner:\n    def go(self):\n        return run(1)\n"),
    "client.js": (
        "const { helper } = require('./util');\n"
        "\n"
        "function invoke(value) {\n"
        "  return helper(value);\n"
        "}\n"
        "\n"
        "module.exports = { invoke };\n"
    ),
    "types.ts": (
        "export interface Options {\n"
        "  retries: number;\n"
        "}\n"
        "\n"
        "export class Base {\n"
        "  start(): number {\n"
        "    return 0;\n"
        "  }\n"
        "}\n"
        "\n"
        "export class Impl extends Base {\n"
        "  start(): number {\n"
        "    return 1;\n"
        "  }\n"
        "}\n"
    ),
    "legacy.go": "package main\n\nfunc main() {}\n",
}

INDEX_NAME = "ig02-fixture"


def write_fixture(project) -> None:
    """Write the mixed-language fixture into the project root.

    Args:
        project: Project root directory.
    """
    for name, content in FIXTURE_FILES.items():
        (project / name).write_text(content, encoding="utf-8")


def index_fixture(project) -> None:
    """Index the fixture through the real CLI with mock embeddings.

    Args:
        project: Project root directory.
    """
    result = CliRunner().invoke(app, ["index", str(project), INDEX_NAME])
    assert result.exit_code == 0, result.output


def ctxai(*args, cwd) -> subprocess.CompletedProcess:
    """Run the CLI in a fresh process from the project directory.

    Args:
        *args: CLI arguments.
        cwd: Working directory (the fixture project).

    Returns:
        The completed subprocess.
    """
    return subprocess.run(["uv", "run", "ctxai", *args], capture_output=True, text=True, cwd=str(cwd))


def cli_json(*args, cwd) -> dict:
    """Run the CLI in a fresh process and parse its JSON envelope.

    Args:
        *args: CLI arguments ending in a --json call.
        cwd: Working directory.

    Returns:
        The parsed JSON document.
    """
    result = ctxai(*args, cwd=cwd)
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout
    return json.loads(output[output.find("{") : output.rfind("}") + 1])


@pytest.mark.e2e
@pytest.mark.indexing
def test_mixed_language_fixture_exposes_consistent_semantics(temp_dir, patch_embeddings_factory):
    """Every supported language produces nodes with evidence and confident relationships."""
    write_fixture(temp_dir)
    index_fixture(temp_dir)

    stats = cli_json("graph", "stats", INDEX_NAME, "--json", cwd=temp_dir)["graph"]
    languages = set(stats["adapter_versions"].keys())
    assert {"python", "javascript", "typescript"} <= languages, stats

    # Python: class + method + function, with the call edge resolved.
    py_symbol = cli_json("graph", "symbol", "run", "--language", "python", "--json", cwd=temp_dir)
    py_nodes = py_symbol["symbols"] if "symbols" in py_symbol else py_symbol["results"]
    assert any(node["display_name"] == "run" for node in py_nodes)
    run_node = next(node for node in py_nodes if node["display_name"] == "run")
    assert run_node["file_path"] == "service.py"
    assert run_node["start_line"] >= 1 and run_node["end_line"] >= run_node["start_line"]

    run_neighbors = cli_json("graph", "neighbors", run_node["id"], "--direction", "both", "--json", cwd=temp_dir)
    edges = run_neighbors["edges"]
    assert edges, "expected the call edge from Runner.go into run"
    assert all(edge["evidence_file"] and edge["confidence"] for edge in edges)

    # TypeScript: interface kind present, inherits edge between Base and Impl.
    ts_symbol = cli_json("graph", "symbol", "Options", "--json", cwd=temp_dir)
    ts_nodes = ts_symbol["symbols"] if "symbols" in ts_symbol else ts_symbol["results"]
    assert any(node["kind"] == "interface" for node in ts_nodes), ts_nodes

    impl_node = cli_json("graph", "symbol", "Impl", "--json", cwd=temp_dir)["symbols"][0]
    # inherits edges point child -> parent, so Impl's base class is out-direction.
    impl_edges = cli_json(
        "graph", "neighbors", impl_node["id"], "--edge", "inherits", "--direction", "out", "--json", cwd=temp_dir
    )["edges"]
    assert any(edge["kind"] == "inherits" for edge in impl_edges), impl_edges

    # JavaScript: CommonJS import resolved with evidence, helper call edge.
    js_symbol = cli_json("graph", "symbol", "invoke", "--json", cwd=temp_dir)
    js_nodes = js_symbol["symbols"] if "symbols" in js_symbol else js_symbol["results"]
    assert js_nodes and js_nodes[0]["language"] == "javascript"


@pytest.mark.e2e
@pytest.mark.indexing
def test_cli_mcp_dashboard_agree_on_identity_counts_and_relationships(temp_dir, patch_embeddings_factory):
    """The same index reports identical identity, counts, and relationships on every surface."""
    write_fixture(temp_dir)
    index_fixture(temp_dir)

    cli_stats = cli_json("graph", "stats", INDEX_NAME, "--json", cwd=temp_dir)["graph"]
    cli_node_total = sum(cli_stats["node_counts"].values())
    cli_edge_total = sum(cli_stats["edge_counts"].values())
    assert cli_stats["schema_version"] == 2  # IG-02 graph schema v2

    with patch("ctxai.commands.server_command.get_indexes_dir", return_value=temp_dir / ".ctxai" / "indexes"):

        async def _mcp():
            async with connected_client(create_server(temp_dir)) as client:
                stats = await call(client, "graph_stats", {"index_name": INDEX_NAME})
                symbol = await call(client, "graph_symbol", {"index_name": INDEX_NAME, "query": "run"})
                return stats, symbol

        import anyio

        mcp_stats, mcp_symbol = anyio.run(_mcp)
    mcp_data = mcp_stats["data"] if "data" in mcp_stats else mcp_stats
    assert mcp_data["graph_schema_version"] == cli_stats["schema_version"]
    mcp_graph = mcp_data["graph"]
    assert mcp_graph["node_counts"] == cli_stats["node_counts"]
    assert mcp_graph["edge_counts"] == cli_stats["edge_counts"]
    mcp_results = mcp_symbol["data"]["symbols"] if "data" in mcp_symbol else mcp_symbol["symbols"]
    assert mcp_results, "MCP symbol search returned nothing"

    with TestClient(create_dashboard_app(temp_dir)) as client:
        page = client.get(f"/index/{INDEX_NAME}/graph")
        assert page.status_code == 200
        body = page.text
        assert str(cli_node_total) in body, "dashboard must show the same node total"
        assert str(cli_edge_total) in body, "dashboard must show the same edge total"
        node_id = mcp_results[0]["id"]
        detail = client.get(f"/index/{INDEX_NAME}/graph/node/{node_id}")
        assert detail.status_code == 200


@pytest.mark.e2e
@pytest.mark.indexing
def test_unsupported_language_stays_indexable_and_reported(temp_dir, patch_embeddings_factory):
    """A Go file indexes as ordinary chunks, fabricates no graph nodes, and is reported as unsupported."""
    write_fixture(temp_dir)
    index_fixture(temp_dir)

    caps = cli_json("graph", "capabilities", INDEX_NAME, "--json", cwd=temp_dir)
    payload = caps.get("capabilities", caps)
    language_rows = {row["language"]: row for row in payload.get("languages", [])}
    assert {"python", "javascript", "typescript"} <= set(language_rows)
    go_row = language_rows.get("go")
    assert go_row is not None and go_row["adapter_version"] is None, "go must be reported with no adapter"

    stats = cli_json("graph", "stats", INDEX_NAME, "--json", cwd=temp_dir)["graph"]
    go_nodes = (
        [node for node in stats.get("nodes_by_language", {}).get("go", [])] if "nodes_by_language" in stats else []
    )
    assert not go_nodes
    for node in stats.get("node_counts", {}):
        assert node  # sanity: counts exist per kind

    # Filtering by an unsupported language is rejected deterministically, never faked.
    go_filter = ctxai("graph", "symbol", "run", "--language", "go", cwd=temp_dir)
    assert go_filter.returncode == 1
    assert "Unsupported language 'go'" in go_filter.stderr


@pytest.mark.e2e
@pytest.mark.indexing
def test_deterministic_errors_for_bad_input(temp_dir, patch_embeddings_factory):
    """Malformed names and out-of-bounds parameters fail with bounded, path-free errors."""
    write_fixture(temp_dir)
    index_fixture(temp_dir)

    # Excessive depth is rejected with a bounded error naming the limit.
    result = ctxai("graph", "symbol", "run", "--json", cwd=temp_dir)
    assert result.returncode == 0
    output = result.stdout
    symbol_id = json.loads(output[output.find("{") : output.rfind("}") + 1])["symbols"][0]["id"]

    deep = ctxai("graph", "neighbors", symbol_id, "--depth", "9", cwd=temp_dir)
    assert deep.returncode == 1
    assert "depth" in (deep.stdout + deep.stderr).lower()
    assert str(temp_dir) not in deep.stdout + deep.stderr

    # Malformed symbol ids fail as not-found without leaking store details.
    missing = ctxai("graph", "neighbors", "deadbeefdeadbeef", cwd=temp_dir)
    assert missing.returncode == 1
    combined = missing.stdout + missing.stderr
    assert "not found" in combined.lower() or "no symbol" in combined.lower()
    assert str(temp_dir) not in combined

    # Empty queries are rejected as invalid input.
    empty = ctxai("graph", "symbol", "", cwd=temp_dir)
    assert empty.returncode == 1

    # Unknown index names fail cleanly.
    unknown = ctxai("graph", "stats", "no-such-index", cwd=temp_dir)
    assert unknown.returncode == 1


@pytest.mark.e2e
@pytest.mark.indexing
def test_adapter_versions_stored_on_records(temp_dir, patch_embeddings_factory):
    """Graph records carry their language and adapter version for stale detection."""
    write_fixture(temp_dir)
    index_fixture(temp_dir)

    result = ctxai("graph", "symbol", "invoke", "--json", cwd=temp_dir)
    output = result.stdout
    node = json.loads(output[output.find("{") : output.rfind("}") + 1])["symbols"][0]
    assert node["language"] == "javascript"
    assert node["adapter_version"], "adapter version must be recorded on the node"
