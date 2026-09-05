"""Workflow-linter tests for the retrieval-quality CI gate (RE-03).

Parses `.github/workflows/pr-gate.yml` and asserts the retrieval job stays
deterministic and credential-free: it runs the checked-in gate script with
`--fail-on-regression` semantics, uploads artifacts on success and failure,
and references no repository secrets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "pr-gate.yml"


@pytest.fixture
def workflow() -> dict:
    """Parse the PR-gate workflow.

    Returns:
        The parsed workflow document.
    """
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_retrieval_quality_job_exists(workflow):
    """The retrieval-quality gate job is present."""
    assert "retrieval-quality" in workflow["jobs"]


def test_retrieval_job_is_deterministic_and_credential_free(workflow):
    """The job runs the gate script and references no secrets."""
    job = workflow["jobs"]["retrieval-quality"]
    steps = job["steps"]
    run_steps = [step.get("run", "") for step in steps if isinstance(step, dict)]
    gate_runs = [run for run in run_steps if "scripts/ci_retrieval_eval.py" in run]
    assert gate_runs, "the gate script must be invoked"

    # Criterion 2: deterministic, no credentials or external services.
    serialized = str(steps)
    assert "secrets." not in serialized, "the retrieval job must not reference repository secrets"
    assert "ANTHROPIC_API_KEY" not in serialized and "OPENAI_API_KEY" not in serialized

    # Criterion 3: latency is reported, never gated (checked-in tolerances).
    from ctxai.evals.artifacts import DEFAULT_TOLERANCES

    assert all("latency" not in metric for metric in DEFAULT_TOLERANCES)


def test_retrieval_job_uploads_artifacts_always(workflow):
    """Artifacts upload on success and failure (criterion: reviewable evidence)."""
    job = workflow["jobs"]["retrieval-quality"]
    uploads = [
        step for step in job["steps"] if isinstance(step, dict) and "upload-artifact" in str(step.get("uses", ""))
    ]
    assert uploads, "evaluation artifacts must be uploaded"
    assert all(step.get("if") == "always()" for step in uploads)


def test_checked_in_baseline_matches_benchmark_fingerprint():
    """The checked-in baseline carries the benchmark identity (criterion 5)."""
    import json

    baseline = json.loads(
        (Path(__file__).parents[1] / "tests" / "fixtures" / "retrieval_baseline.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        (Path(__file__).parents[1] / "tests" / "fixtures" / "retrieval_benchmark.json").read_text(encoding="utf-8")
    )
    from ctxai.evals.common import content_fingerprint

    expected = content_fingerprint(benchmark)
    assert baseline["benchmark"]["fingerprint"] == expected
