"""RE-01 acceptance tests: executable, versioned retrieval benchmark (e2e).

Builds a fixture repository from checked-in fixture code (the benchmark's
expected evidence files), indexes it through the real CLI pipeline with
``patch_embeddings_factory`` (deterministic mock embeddings, no network), and
exercises ``ctxai eval retrieval`` end to end:

1. the migrated 20-question benchmark runs at runtime and reports all
   required metrics, per-case evidence, and configuration identity;
2. repeated runs are byte-stable apart from documented volatile fields;
3. baseline comparison gates regressions with checked-in tolerances and fail
   clearly on missing/incompatible baselines;
4. invalid expectations, stale/unhealthy indexes, and embedding mismatches
   are explicit failures, never silent passes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.evals.common import strip_volatile

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "tests" / "fixtures" / "retrieval_benchmark.json"

REQUIRED_METRICS = (
    "successful_query_rate",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    "evidence_precision@5",
    "latency_p50_ms",
    "latency_p95_ms",
    "selected_token_mean",
    "selected_token_p95",
    "duplicate_token_ratio",
    "graph_contribution_rate",
)


@pytest.fixture
def runner(monkeypatch, patch_embeddings_factory) -> CliRunner:
    """CLI runner with a clean CTXAI_HOME and deterministic mock embeddings.

    Every RE-01 e2e run uses the patched ``EmbeddingsFactory`` (no network,
    no model download) exactly like the other e2e slices.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        patch_embeddings_factory: Mock embedding patch fixture.

    Returns:
        Configured CliRunner.
    """
    monkeypatch.delenv("CTXAI_HOME", raising=False)
    return CliRunner()


def build_fixture_project(tmp_path: Path) -> Path:
    """Create a fresh project containing the benchmark's evidence files.

    The benchmark expects repository-relative paths from the real ctxai tree,
    so the fixture copies exactly those files (plus a README) into a clean
    project, keeping their relative layout.

    Args:
        tmp_path: Scratch directory.

    Returns:
        The fixture project root.
    """
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    needed: set[str] = set()
    for case in payload["cases"]:
        needed.update(case["expected"]["files"])
        needed.update(case["expected"].get("line_ranges", {}))
    project = tmp_path / "project"
    for relative in sorted(needed):
        source = REPO_ROOT / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    (project / "README.md").write_text("# Retrieval benchmark fixture project\n", encoding="utf-8")
    return project


def index_fixture(runner: CliRunner, project: Path) -> None:
    """Index the fixture project through the real CLI with mock embeddings.

    Args:
        runner: CLI runner.
        project: Fixture project root.
    """
    result = runner.invoke(app, ["index", str(project), "fixture-index"])
    assert result.exit_code == 0, result.output


def run_eval(runner: CliRunner, project: Path, *extra: str, index: str = "fixture-index"):
    """Invoke ``ctxai eval retrieval`` against the fixture project.

    Args:
        runner: CLI runner.
        project: Fixture project root.
        *extra: Additional CLI arguments.
        index: Index name to evaluate against.

    Returns:
        The CLI invocation result.
    """
    return runner.invoke(
        app,
        ["eval", "retrieval", str(BENCHMARK_PATH), "--index", index, "--project-path", str(project), *extra],
    )


def load_payload(path: Path) -> dict:
    """Read and parse an artifact JSON file.

    Args:
        path: Artifact path.

    Returns:
        The parsed artifact payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_output(output: str) -> dict:
    """Parse a --json CLI payload, tolerating any non-JSON leading noise.

    Args:
        output: Captured CLI stdout.

    Returns:
        The parsed JSON document.
    """
    stripped = output.strip()
    decoder = json.JSONDecoder()
    for start in range(len(stripped)):
        if stripped[start] == "{":
            try:
                return decoder.decode(stripped[start:])
            except ValueError:
                continue
    raise AssertionError(f"No JSON object found in output: {output[:200]!r}")


def flat(text: str) -> str:
    """Collapse all whitespace so substring checks survive Rich wrapping.

    Args:
        text: Raw CLI output.

    Returns:
        Whitespace-normalized text.
    """
    return " ".join(text.split())


@pytest.mark.e2e
@pytest.mark.e2e
def test_clean_install_run_reports_required_metrics_and_identity(runner, tmp_path):
    """Criteria 1+2: one command uses the fixture index and reports all
    required available metrics plus per-case ranks, citations, timing,
    selected tokens, and configuration identity."""
    project = build_fixture_project(tmp_path)
    index_fixture(runner, project)

    result = run_eval(runner, project, "--json")
    assert result.exit_code == 0, result.output
    payload = parse_json_output(result.output)

    assert payload["schema_version"] == 1
    assert payload["kind"] == "retrieval"
    assert payload["status"] == "complete"
    assert payload["benchmark"]["name"] == "ctxai-retrieval-core"
    assert payload["benchmark"]["case_count"] == 20
    assert payload["benchmark"]["fingerprint"]
    assert len(payload["runs"]) == 20

    # Retrieved results are produced at runtime, not embedded in the fixture.
    benchmark_payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    assert "retrieved_locations" not in json.dumps(benchmark_payload)

    overall = payload["aggregates"]["overall"]
    for metric in REQUIRED_METRICS:
        entry = overall["metrics"][metric]
        if metric == "graph_contribution_rate":
            assert entry["available"] is False
            assert "graph expansion not enabled" in entry["reason"]
        else:
            assert entry["available"] is True, f"{metric} unexpectedly unavailable: {entry}"
    assert overall["metrics"]["successful_query_rate"]["value"] == 1.0
    assert set(payload["aggregates"]["by_cohort"]) == {"agent", "interface", "pipeline"}
    assert set(payload["aggregates"]["by_split"]) == {"test", "dev", "train"}
    assert set(overall["confidence_intervals"]) == {"recall@5", "mrr"}

    first = payload["runs"][0]
    assert first["case_id"] == "q01-agent-iteration-loop"
    assert first["status"] == "ok"
    assert first["query"] == "agent iteration loop"
    assert first["candidate_count"] >= 1
    assert first["selected_count"] >= 1
    assert first["estimated_tokens"] > 0
    assert first["latency"]["values_ms"]
    assert first["timings"]["retrieve_ms"] >= 0.0
    assert first["first_relevant_rank"] >= 1
    for name in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "evidence_precision@5"):
        assert first["metrics"][name]["available"] is True
    ranked = [candidate for candidate in first["candidates"]]
    assert [candidate["final_rank"] for candidate in ranked] == list(range(1, len(ranked) + 1))
    assert any(candidate["decision"] == "selected" for candidate in ranked)
    assert all("citation" in candidate and candidate["citation"].startswith("src/ctxai/") for candidate in ranked)
    assert all(candidate["chunk_id"] for candidate in ranked)
    assert all("reasons" in candidate for candidate in ranked)

    assert payload["configuration"]["fingerprint"]
    assert payload["configuration"]["embedding"]["provider"] == "local"
    assert payload["configuration"]["embedding"]["dimension"] == 384
    assert payload["index"]["name"] == "fixture-index"
    assert payload["index"]["healthy"] is True
    assert payload["index"]["stale"] is False
    assert payload["index"]["chunk_count"] > 0
    assert payload["environment"]["network_access"] == "none"
    assert payload["comparison"] is None

    # Default persistence lands under .ctxai/evaluations/retrieval.
    artifacts = list((project / ".ctxai" / "evaluations" / "retrieval").glob("*.json"))
    assert artifacts, "artifact persisted by default"


@pytest.mark.e2e
def test_repeated_runs_are_byte_stable_modulo_volatile_fields(runner, tmp_path):
    """Criterion 3: two runs differ only in documented volatile fields."""
    project = build_fixture_project(tmp_path)
    index_fixture(runner, project)
    first_path = project / "first.json"
    second_path = project / "second.json"

    assert run_eval(runner, project, "--output", str(first_path), "--repeat", "2").exit_code == 0
    assert run_eval(runner, project, "--output", str(second_path), "--repeat", "2").exit_code == 0

    first = load_payload(first_path)
    second = load_payload(second_path)
    assert strip_volatile(first) == strip_volatile(second)
    assert first["created_at"] != second["created_at"]
    assert first["run_id"] != second["run_id"]
    assert first["runs"][0]["latency"]["warmup_excluded"] == 1
    assert len(first["runs"][0]["latency"]["values_ms"]) == 1


@pytest.mark.e2e
def test_baseline_gates_report_regressions_and_fail_clearly(runner, tmp_path):
    """Criterion 4: gates use checked-in tolerances, report each failing
    gate, and missing/incompatible baselines fail clearly."""
    project = build_fixture_project(tmp_path)
    index_fixture(runner, project)
    baseline_path = project / "baseline.json"
    assert run_eval(runner, project, "--output", str(baseline_path)).exit_code == 0

    # Identical deterministic run against its own baseline passes the gates.
    ok = run_eval(runner, project, "--baseline", str(baseline_path), "--fail-on-regression", "--json")
    assert ok.exit_code == 0, ok.output
    comparison = parse_json_output(ok.output)["comparison"]
    assert comparison["status"] == "pass"
    assert comparison["compatible"] is True
    assert all(gate["status"] == "pass" for gate in comparison["gates"])

    # Seeded regression: an impossible baseline recall makes every current
    # value a regression beyond tolerance; the gate is named.
    degraded = load_payload(baseline_path)
    degraded["aggregates"]["overall"]["metrics"]["recall@5"]["value"] = 2.0
    degraded["aggregates"]["by_cohort"]["agent"]["metrics"]["recall@5"]["value"] = 2.0
    degraded_path = project / "degraded-baseline.json"
    degraded_path.write_text(json.dumps(degraded), encoding="utf-8")
    failing = run_eval(runner, project, "--baseline", str(degraded_path), "--fail-on-regression")
    assert failing.exit_code == 1
    assert "recall@5" in flat(failing.output)
    assert "Gate failed" in flat(failing.output)

    # Missing baseline fails clearly instead of silently passing.
    missing = run_eval(runner, project, "--baseline", str(project / "nope.json"), "--fail-on-regression")
    assert missing.exit_code == 1
    assert "not readable" in flat(missing.output)

    # Incompatible baseline (different benchmark document) fails clearly.
    other = load_payload(baseline_path)
    other["benchmark"]["fingerprint"] = "0" * 64
    other_path = project / "other-benchmark.json"
    other_path.write_text(json.dumps(other), encoding="utf-8")
    incompatible = run_eval(runner, project, "--baseline", str(other_path), "--fail-on-regression", "--json")
    assert incompatible.exit_code == 1
    assert parse_json_output(incompatible.output)["comparison"]["status"] == "incompatible"

    # --fail-on-regression without a baseline is a usage error.
    no_baseline = run_eval(runner, project, "--fail-on-regression")
    assert no_baseline.exit_code == 1


@pytest.mark.e2e
def test_validate_subcommand(runner, tmp_path):
    """Schema/ID/path/split validation without retrieval, plus repository-
    level evidence checks when a project root is given."""
    valid = runner.invoke(app, ["eval", "retrieval", "validate", str(BENCHMARK_PATH)])
    assert valid.exit_code == 0, valid.output
    assert "ctxai-retrieval-core" in valid.output

    valid_json = runner.invoke(app, ["eval", "retrieval", "validate", str(BENCHMARK_PATH), "--json"])
    assert valid_json.exit_code == 0
    assert parse_json_output(valid_json.output)["valid"] is True

    project = build_fixture_project(tmp_path)
    with_repo = runner.invoke(
        app, ["eval", "retrieval", "validate", str(BENCHMARK_PATH), "--project-path", str(project)]
    )
    assert with_repo.exit_code == 0, with_repo.output

    broken = {
        "schema_version": 1,
        "name": "broken",
        "cases": [
            {
                "id": "x1",
                "query": "one",
                "tags": [],
                "cohort": "core",
                "split": "test",
                "expected": {"files": ["src/a.py"], "symbols": [], "line_ranges": {}},
                "relevance": {},
            },
            {
                "id": "x1",
                "query": "two",
                "tags": [],
                "cohort": "core",
                "split": "holdout",
                "expected": {"files": ["/absolute/path.py"], "symbols": [], "line_ranges": {}},
                "relevance": {},
            },
        ],
    }
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    invalid = runner.invoke(app, ["eval", "retrieval", "validate", str(broken_path), "--json"])
    assert invalid.exit_code == 1
    envelope = parse_json_output(invalid.output)
    errors = " ".join(envelope["errors"])
    assert "duplicate case id" in errors
    assert "split" in errors
    assert "repository-relative" in errors


@pytest.mark.e2e
def test_invalid_expectations_partial_runs_and_unhealthy_indexes_fail(runner, tmp_path):
    """Criterion 5: invalid expectations, partial runs, stale identity, and
    embedding mismatch are explicit failures with non-zero exits."""
    project = build_fixture_project(tmp_path)
    index_fixture(runner, project)

    # A benchmark case whose line range runs past the file end.
    partial_benchmark = {
        "schema_version": 1,
        "name": "partial-benchmark",
        "cases": [
            {
                "id": "p1",
                "query": "agent iteration loop",
                "tags": [],
                "cohort": "agent",
                "split": "test",
                "expected": {
                    "files": ["src/ctxai/agent/core.py"],
                    "symbols": [],
                    "line_ranges": {"src/ctxai/agent/core.py": [900000, 900100]},
                },
                "relevance": {"src/ctxai/agent/core.py": 3},
            }
        ],
    }
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(json.dumps(partial_benchmark), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "eval",
            "retrieval",
            str(partial_path),
            "--index",
            "fixture-index",
            "--project-path",
            str(project),
            "--output",
            str(project / "partial-artifact.json"),
        ],
    )
    assert result.exit_code == 1
    assert "partial" in flat(result.output)
    payload = load_payload(project / "partial-artifact.json")
    assert payload["status"] == "partial"
    assert payload["runs"][0]["status"] == "error"
    assert "beyond file length" in payload["runs"][0]["error"]

    # Unknown index fails clearly.
    unknown = run_eval(runner, project, index="no-such-index")
    assert unknown.exit_code == 1
    assert "cannot be inspected" in flat(unknown.output)

    # Output paths outside the project boundary are refused.
    outside = run_eval(runner, project, "--output", str(tmp_path / "outside.json"))
    assert outside.exit_code == 1
    assert "outside the project boundary" in flat(outside.output)

    # Embedding identity mismatch (manifest claims a different model).
    manifest_path = project / ".ctxai" / "indexes" / "fixture-index" / "manifest.json"
    manifest = load_payload(manifest_path)
    manifest["embedding_model"] = "not-the-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mismatch = run_eval(runner, project)
    assert mismatch.exit_code == 1
    assert "embedding identity" in flat(mismatch.output)
