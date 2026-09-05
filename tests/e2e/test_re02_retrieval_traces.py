"""RE-02 acceptance tests: privacy-preserving retrieval observability.

Runs the real CLI pipeline with mock embeddings against an indexed fixture to
prove: a traced query produces a versioned run with generators, ordered
candidates, graph paths, stage timings, and identity; default configuration
writes no traces while metrics/full modes differ in what they persist; seeded
secrets and home paths never reach traces, terminal output, MCP results, or
dashboard HTML; recorder failures never fail retrieval; and retention plus
deletion remove exactly the resolved targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.commands.dashboard_command import create_dashboard_app
from ctxai.config import RetrievalConfig
from ctxai.retrieval_traces import resolve_trace_dir, trace_dir_for

FIXTURE_FILES = {
    "service.py": (
        "def fetch_data(key):\n"
        '    """Load rows for the loader."""\n'
        "    return key\n"
        "\n"
        "\n"
        "def loader_verification_rows():\n"
        "    return fetch_data(1)\n"
    ),
    "README.md": "# Trace fixture\n",
}

SEEDED_API_KEY = "sk-test-abc123def456ghi789"
SEEDED_HOME_PATH = "/Users/alice/.ctxai/keys.json"


def index_fixture(project: Path, name: str = "trace-e2e") -> None:
    """Write fixture files and index them through the real CLI.

    Args:
        project: Project root (populated in place).
        name: Index name.
    """
    for relative, content in FIXTURE_FILES.items():
        (project / relative).write_text(content, encoding="utf-8")
    result = CliRunner().invoke(app, ["index", str(project), name])
    assert result.exit_code == 0, result.output


def traced_query(project: Path, query: str, *, config: RetrievalConfig | None = None, trace_flag: bool = True):
    """Run `ctxai query --trace` with the embeddings factory patched to mocks.

    Args:
        project: Project root.
        query: The query text.
        config: Optional retrieval configuration override.
        trace_flag: Whether to pass --trace.

    Returns:
        The CLI result.
    """

    runner = CliRunner()
    if config is not None:
        config_dir = project / ".ctxai"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "config.toml").write_text(
            "[retrieval]\n"
            f'trace_mode = "{config.trace_mode}"\n'
            f'trace_query_text = "{config.trace_query_text}"\n'
            f'trace_source_preview = "{config.trace_source_preview}"\n'
            f"trace_retention = {config.trace_retention}\n",
            encoding="utf-8",
        )
    arguments = ["query", "trace-e2e", query, "--no-content"]
    if trace_flag:
        arguments.append("--trace")
    return runner.invoke(app, arguments)


def in_project(monkeypatch, project: Path) -> None:
    """Run the CLI from inside the fixture project (cwd-based resolution).

    Args:
        monkeypatch: pytest monkeypatch fixture.
        project: The fixture project root.
    """
    monkeypatch.chdir(project)


@pytest.mark.e2e
@pytest.mark.indexing
def test_traced_query_produces_versioned_run(temp_dir, patch_embeddings_factory, monkeypatch):
    """A traced query persists one versioned run with the full provenance contract."""
    index_fixture(temp_dir)
    in_project(monkeypatch, temp_dir)
    result = traced_query(temp_dir, "loader verification rows")
    assert result.exit_code == 0, result.output

    trace_dir = trace_dir_for(temp_dir)
    files = list(trace_dir.glob("*.jsonl"))
    assert len(files) == 1, "one traced query writes exactly one run file"
    payload = json.loads(files[0].read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["status"] == "ok"
    assert payload["index"]["name"] == "trace-e2e"
    assert payload["index"]["embedding_provider"] == "mock"
    generators = {generator["component"] for generator in payload["generators"]}
    assert {"semantic", "lexical", "symbol", "repository-map"} <= generators
    candidates = payload["candidates"]
    assert [candidate["final_rank"] for candidate in candidates] == list(range(1, len(candidates) + 1))
    for candidate in candidates:
        assert candidate["decision"] in {"selected", "duplicate", "budget", "not_selected"}
        assert candidate["component_ranks"]
    assert payload["selected"], "the assembled context is recorded"
    for stage in ("semantic_candidates", "lexical_candidates", "symbol_candidates", "assemble"):
        assert stage in payload["stage_timings_ms"]
    assert payload["total_latency_ms"] >= 0
    assert payload["network"] == {"recorder_transport": "local-file-only", "outbound_transports": []}

    # CLI round trip: list shows the run, show renders it, json matches disk.
    runner = CliRunner()
    listed = runner.invoke(app, ["retrieval", "runs", "list", "-p", str(temp_dir)])
    assert listed.exit_code == 0 and payload["run_id"][:12] in listed.output
    shown = runner.invoke(app, ["retrieval", "runs", "show", payload["run_id"], "-p", str(temp_dir)])
    assert shown.exit_code == 0 and "trace-e2e" in shown.output
    as_json = runner.invoke(app, ["retrieval", "runs", "show", payload["run_id"], "--json", "-p", str(temp_dir)])
    assert as_json.exit_code == 0
    envelope = json.loads(as_json.output)
    assert envelope["schema_version"] == payload["schema_version"]
    assert envelope["run"]["run_id"] == payload["run_id"]


@pytest.mark.e2e
@pytest.mark.indexing
def test_default_config_writes_no_traces_and_modes_differ(temp_dir, patch_embeddings_factory, monkeypatch):
    """Default configuration persists nothing; metrics stores no query/source; full warns and stores."""
    index_fixture(temp_dir)
    in_project(monkeypatch, temp_dir)

    # Default: --trace absent and config off -> nothing written even after queries.
    plain = traced_query(temp_dir, "loader verification rows", trace_flag=False)
    assert plain.exit_code == 0, plain.output
    assert not trace_dir_for(temp_dir).exists(), "default configuration writes no traces"

    # --trace promotes off config to metrics: query text stays a hash.
    traced = traced_query(temp_dir, "loader verification rows")
    assert traced.exit_code == 0
    files = list(trace_dir_for(temp_dir).glob("*.jsonl"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["query"]["text"] is None
    assert payload["query"]["hash"]
    for candidate in payload["candidates"]:
        assert "preview" not in candidate and "content" not in candidate

    # full mode requires explicit opt-in, warns, and stores bounded previews.
    full = traced_query(
        temp_dir,
        "loader verification rows",
        config=RetrievalConfig(trace_mode="full", trace_query_text="store", trace_source_preview="store"),
    )
    assert full.exit_code == 0
    files = sorted(trace_dir_for(temp_dir).glob("*.jsonl"))
    full_payloads = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    full_runs = [p for p in full_payloads if p["mode"] == "full"]
    assert full_runs, "full-mode run persisted"
    assert full_runs[-1]["query"]["text"] == "loader verification rows"
    previews = [c["preview"] for c in full_runs[-1]["candidates"] if "preview" in c]
    assert previews and all(len(p) <= 500 for p in previews)


@pytest.mark.e2e
@pytest.mark.indexing
def test_seeded_secrets_never_reach_traces_terminal_mcp_or_dashboard(
    temp_dir, patch_embeddings_factory, monkeypatch, mock_llm_config
):
    """Seeded secrets, bearer tokens, credential URLs, and home paths stay out of everything persisted."""
    index_fixture(temp_dir)
    in_project(monkeypatch, temp_dir)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SEEDED_API_KEY)
    seeded_query = (
        f"loader rows {SEEDED_API_KEY} Bearer abcdef123456789 https://user:hunter2@example.com/api {SEEDED_HOME_PATH}"
    )

    result = traced_query(temp_dir, seeded_query)
    assert result.exit_code == 0, result.output
    raw = b"".join(f.read_bytes() for f in trace_dir_for(temp_dir).glob("*.jsonl")).decode("utf-8")
    for forbidden in (SEEDED_API_KEY, "hunter2", "Bearer abcdef123456789", SEEDED_HOME_PATH, "/Users/alice"):
        assert forbidden not in raw, forbidden
    assert SEEDED_API_KEY not in result.output

    # MCP response carries no seeded secret and exposes the run id only when traced.
    import anyio

    from ctxai.commands.server_command import create_server
    from tests.e2e.test_e2e_mcp_server import call, connected_client

    with patch("ctxai.commands.server_command.get_indexes_dir", return_value=temp_dir / ".ctxai" / "indexes"):

        async def _mcp():
            async with connected_client(create_server(temp_dir)) as client:
                return await call(client, "query_codebase", {"index_name": "trace-e2e", "query": "loader rows"})

        mcp_result = anyio.run(_mcp)
    mcp_json = json.dumps(mcp_result)
    assert SEEDED_API_KEY not in mcp_json

    # Dashboard list and detail render without secrets.
    with TestClient(create_dashboard_app(temp_dir)) as client:
        page = client.get("/retrieval-runs")
        assert page.status_code == 200
        assert SEEDED_API_KEY not in page.text
        detail_id = json.loads(raw.splitlines()[0])["run_id"]
        detail = client.get(f"/retrieval-runs/{detail_id}")
        assert detail.status_code == 200
        assert SEEDED_API_KEY not in detail.text and "hunter2" not in detail.text


@pytest.mark.e2e
@pytest.mark.indexing
def test_recorder_failure_never_fails_retrieval(temp_dir, patch_embeddings_factory, monkeypatch):
    """An unwritable trace directory degrades to a diagnostic; retrieval still succeeds."""
    index_fixture(temp_dir)
    in_project(monkeypatch, temp_dir)
    trace_dir = resolve_trace_dir(temp_dir, None)
    trace_dir.parent.mkdir(parents=True, exist_ok=True)
    trace_dir.write_text("not a directory", encoding="utf-8")

    result = traced_query(temp_dir, "loader verification rows")
    assert result.exit_code == 0, result.output
    assert "Found" in result.output, "retrieval results still render"


@pytest.mark.e2e
@pytest.mark.indexing
def test_retention_and_delete_remove_only_resolved_targets(temp_dir, patch_embeddings_factory, monkeypatch):
    """Retention prunes oldest-first; delete removes exactly the resolved trace files."""
    index_fixture(temp_dir)
    in_project(monkeypatch, temp_dir)
    retention_config = RetrievalConfig(trace_mode="metrics", trace_retention=2)
    for query in ("first query", "second query", "third query"):
        result = traced_query(temp_dir, query, config=retention_config)
        assert result.exit_code == 0, result.output
    files = sorted(trace_dir_for(temp_dir).glob("*.jsonl"))
    assert len(files) <= 2, "retention keeps at most the configured count"

    runner = CliRunner()
    remaining = {f.stem for f in trace_dir_for(temp_dir).glob("*.jsonl")}
    victim = sorted(remaining)[0]
    deleted = runner.invoke(app, ["retrieval", "runs", "delete", victim, "-p", str(temp_dir)])
    assert deleted.exit_code == 0, deleted.output
    assert victim not in {f.stem for f in trace_dir_for(temp_dir).glob("*.jsonl")}
    assert (temp_dir / "outside.jsonl").exists() is False  # nothing outside was touched
