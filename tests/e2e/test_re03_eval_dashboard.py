"""RE-03 acceptance tests: retrieval quality dashboard and CI regression gate.

Runs the real benchmark through the CLI, then proves: the dashboard evaluation
views (list, summary, compare) render from the same JSON artifacts the CLI
consumes with matching numbers, the compare command reports incompatible
artifacts with every mismatched identity field and a rebuild action, a seeded
regression trips the gate with a named metric and a non-zero exit, and the
checked-in baseline refresh workflow is a deliberate documented command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.commands.dashboard_command import create_dashboard_app

FIXTURES = Path(__file__).parents[1] / "fixtures"
BENCHMARK_PATH = FIXTURES / "retrieval_benchmark.json"


def flat(text: str) -> str:
    """Normalize Rich output for assertions.

    Args:
        text: Raw output.

    Returns:
        Printable characters plus newlines and tabs.
    """
    return "".join(char for char in text if char.isprintable() or char in "\n\t")


def artifact_path_for(project: Path, run_id: str) -> Path:
    """Locate the on-disk artifact for a run (timestamped file names).

    Args:
        project: Fixture project root.
        run_id: The run identifier from the artifact payload.

    Returns:
        The artifact file path.

    Raises:
        AssertionError: When no matching artifact exists.
    """
    matches = list((project / ".ctxai" / "evaluations" / "retrieval").glob(f"*{run_id[:8]}.json"))
    assert matches, f"no artifact file for run {run_id}"
    return matches[0]


def run_benchmark(runner: CliRunner, project: Path) -> dict:
    """Run the benchmark against the fixture project and return the artifact.

    Args:
        runner: CLI runner.
        project: Fixture project root.

    Returns:
        The parsed evaluation artifact.
    """
    result = runner.invoke(
        app,
        [
            "eval",
            "retrieval",
            str(BENCHMARK_PATH),
            "--index",
            "fixture-index",
            "--project-path",
            str(project),
            "--json",
        ],
    )
    assert result.exit_code == 0, flat(result.output)
    output = result.output
    return json.loads(output[output.find("{") : output.rfind("}") + 1])


@pytest.mark.e2e
@pytest.mark.indexing
def test_dashboard_evaluation_views_match_cli_artifacts(temp_dir, patch_embeddings_factory):
    """Dashboard list, summary, and compare render the same numbers as the CLI JSON."""
    from tests.e2e.test_re01_retrieval_benchmark import build_fixture_project, index_fixture

    project = build_fixture_project(temp_dir)
    index_fixture(CliRunner(), project)

    runner = CliRunner()
    artifact = run_benchmark(runner, project)
    node_total = artifact["aggregates"]["overall"]["metrics"]
    recall5 = node_total["recall@5"]["value"]
    status = artifact["status"]

    with TestClient(create_dashboard_app(project)) as client:
        listing = client.get("/evaluations")
        assert listing.status_code == 200
        assert artifact["run_id"][:16] in listing.text

        summary = client.get(f"/evaluations/{artifact['run_id']}")
        assert summary.status_code == 200
        assert str(recall5) in summary.text or f"{recall5:.4f}" in summary.text
        assert status in summary.text

        compare_page = client.get(f"/evaluations/compare?baseline={artifact['run_id']}&candidate={artifact['run_id']}")
        assert compare_page.status_code == 200
        compare_text = compare_page.text
        # Self-comparison: no regressions, deltas are zero.
        assert "regression" not in compare_text.lower() or "no regressions" in compare_text.lower()

    # CLI compare on the same artifacts (self-comparison is comparable, no regressions).
    artifact_path = artifact_path_for(project, artifact["run_id"])
    compare = runner.invoke(app, ["eval", "retrieval", "compare", str(artifact_path), str(artifact_path), "--json"])
    assert compare.exit_code == 0, flat(compare.output)
    payload = json.loads(compare.output[compare.output.find("{") : compare.output.rfind("}") + 1])
    assert payload["compatible"] is True
    failing = [gate for gate in payload["gates"] if gate["status"] == "fail"]
    assert not failing


@pytest.mark.e2e
@pytest.mark.indexing
def test_incompatible_artifacts_fail_with_named_fields_and_actions(temp_dir, patch_embeddings_factory):
    """Comparing different benchmarks lists every mismatch and a rebuild action."""
    from tests.e2e.test_re01_retrieval_benchmark import build_fixture_project, index_fixture

    project = build_fixture_project(temp_dir)
    index_fixture(CliRunner(), project)
    runner = CliRunner()
    artifact = run_benchmark(runner, project)
    artifact_path = artifact_path_for(project, artifact["run_id"])

    # A benchmark with a different name produces a different fingerprint.
    other_benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    other_benchmark["name"] = "other-benchmark"
    other_path = project / "other-benchmark.json"
    other_path.write_text(json.dumps(other_benchmark), encoding="utf-8")
    other_run = runner.invoke(
        app,
        ["eval", "retrieval", str(other_path), "--index", "fixture-index", "--project-path", str(project), "--json"],
    )
    assert other_run.exit_code == 0, flat(other_run.output)
    other_output = other_run.output
    other_artifact = json.loads(other_output[other_output.find("{") : other_output.rfind("}") + 1])
    other_artifact_path = artifact_path_for(project, other_artifact["run_id"])

    compare = runner.invoke(
        app, ["eval", "retrieval", "compare", str(artifact_path), str(other_artifact_path), "--json"]
    )
    assert compare.exit_code == 2, "incompatible comparisons must not pass silently"
    payload = json.loads(compare.output[compare.output.find("{") : compare.output.rfind("}") + 1])
    assert payload["compatible"] is False
    assert payload["incompatibilities"], "every mismatched identity field is named"
    assert any("benchmark" in item.lower() for item in payload["incompatibilities"])
    combined = flat(compare.output) + json.dumps(payload)
    assert "re-run" in combined.lower() or "rebuild" in combined.lower() or "refresh" in combined.lower()

    # The dashboard comparison surfaces the incompatibility banner.
    with TestClient(create_dashboard_app(project)) as client:
        page = client.get(f"/evaluations/compare?baseline={artifact['run_id']}&candidate={other_artifact['run_id']}")
        assert page.status_code == 200
        assert "incompatible" in page.text.lower()


@pytest.mark.e2e
@pytest.mark.indexing
def test_seeded_regression_trips_gate_with_named_metric(temp_dir, patch_embeddings_factory):
    """An impossible baseline makes recall@5 fail by name with a non-zero exit (CI criterion)."""
    from tests.e2e.test_re01_retrieval_benchmark import build_fixture_project, index_fixture

    project = build_fixture_project(temp_dir)
    index_fixture(CliRunner(), project)
    runner = CliRunner()
    artifact = run_benchmark(runner, project)
    artifact_path = artifact_path_for(project, artifact["run_id"])

    degraded = json.loads(json.dumps(artifact))
    degraded["aggregates"]["overall"]["metrics"]["recall@5"]["value"] = 2.0
    degraded_path = project / "degraded-baseline.json"
    degraded_path.write_text(json.dumps(degraded), encoding="utf-8")

    failing = runner.invoke(app, ["eval", "retrieval", "compare", str(degraded_path), str(artifact_path)])
    assert failing.exit_code == 1
    flat_output = flat(failing.output)
    assert "recall@5" in flat_output
    assert "Gate failed" in flat_output

    # Latency regressions alone never gate (criterion 3: noisy timing).
    latency_only = json.loads(json.dumps(artifact))
    for block in [latency_only["aggregates"]["overall"]["metrics"]] + [
        cohort["metrics"] for cohort in latency_only["aggregates"]["by_cohort"].values()
    ]:
        if "latency_p95_ms" in block and block["latency_p95_ms"].get("value") is not None:
            block["latency_p95_ms"]["value"] = block["latency_p95_ms"]["value"] * 100
        if "latency_p50_ms" in block and block["latency_p50_ms"].get("value") is not None:
            block["latency_p50_ms"]["value"] = block["latency_p50_ms"]["value"] * 100
    latency_path = project / "latency-baseline.json"
    latency_path.write_text(json.dumps(latency_only), encoding="utf-8")
    latency_compare = runner.invoke(
        app, ["eval", "retrieval", "compare", str(latency_path), str(artifact_path), "--json"]
    )
    assert latency_compare.exit_code == 0, "latency-only differences must not fail the gate"


@pytest.mark.e2e
@pytest.mark.indexing
def test_checked_in_baseline_is_current_and_reviewable(temp_dir):
    """The CI gate script passes against the checked-in baseline (deterministic refresh path)."""
    import subprocess

    baseline_path = FIXTURES / "retrieval_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    from ctxai.evals.common import content_fingerprint

    assert baseline["benchmark"]["fingerprint"] == content_fingerprint(
        json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    ), "a benchmark change requires a deliberate baseline refresh"

    # The exact command CI runs: deterministic, credential-free, and
    # non-zero only on regression or incompatibility.
    result = subprocess.run(
        ["uv", "run", "python", "scripts/ci_retrieval_eval.py", "--artifacts-dir", str(temp_dir / "ci-artifacts")],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[2]),
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-400:]

    # The fresh artifact carried the same benchmark fingerprint as the baseline.
    candidate_file = temp_dir / "ci-artifacts" / "candidate.json"
    assert candidate_file.exists(), "the gate script persisted its candidate artifact"
    fresh_payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    assert fresh_payload["benchmark"]["fingerprint"] == baseline["benchmark"]["fingerprint"]
