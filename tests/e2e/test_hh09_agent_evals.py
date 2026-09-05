"""HH-09 acceptance tests: agent task evaluation harness.

Executes the deterministic mock-provider task benchmark through the real CLI
and the real agent loop (no network, no credentials) to prove: a clean-install
run produces a versioned artifact, all shipped cases pass with
``MockLLMProvider`` while a seeded regression trips a named gate with a
non-zero exit, forbidden-path violations and budget overruns are scored as
failures, the mock provider-conformance suite runs without network, and the
checked-in baseline compares clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.app import app

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agent_benchmark"
BENCHMARK_PATH = FIXTURES / "benchmark.json"
BASELINE_PATH = FIXTURES / "baseline.json"


def flat(text: str) -> str:
    """Normalize CLI output for assertions.

    Args:
        text: Raw CLI output.

    Returns:
        Text with Rich box-drawing characters stripped.
    """
    return "".join(char for char in text if char.isprintable() or char in "\n\t")


def parse_json_output(output: str) -> dict:
    """Parse the JSON document printed by ``--json`` mode.

    Args:
        output: Raw CLI output containing a JSON document.

    Returns:
        The parsed JSON payload.
    """
    start = output.find("{")
    return json.loads(output[start : output.rfind("}") + 1])


def run_agent_eval(runner: CliRunner, project: Path, *extra: str) -> object:
    """Invoke ``ctxai eval agent`` against the shipped benchmark.

    Args:
        runner: CLI runner.
        project: Project root receiving artifacts and workspaces.
        extra: Additional CLI arguments.

    Returns:
        The CliRunner result.
    """
    return runner.invoke(
        app,
        ["eval", "agent", str(BENCHMARK_PATH), "--project-path", str(project), *extra],
    )


def write_case_benchmark(project: Path, payload: dict) -> Path:
    """Write a one-case benchmark document into the project.

    Args:
        project: Project root.
        payload: Benchmark payload with a single case.

    Returns:
        Path to the written benchmark document.
    """
    path = project / "case-benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.e2e
@pytest.mark.agent
def test_clean_install_mock_run_produces_versioned_artifact(temp_dir, mock_llm_config):
    """One command runs the benchmark from a clean install with no network or credentials."""
    runner = CliRunner()
    result = run_agent_eval(runner, temp_dir, "--json")
    assert result.exit_code == 0, flat(result.output)

    artifact = parse_json_output(result.output)
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "complete"
    assert artifact["kind"] == "agent"
    assert artifact["benchmark"]["fingerprint"]
    assert artifact["configuration"]["provider"]["mode"] == "mock"
    assert artifact["configuration"]["runner"]

    on_disk = sorted((temp_dir / ".ctxai" / "evaluations" / "agent").glob("*.json"))
    assert on_disk, "expected an artifact under .ctxai/evaluations/agent/"
    persisted = json.loads(on_disk[-1].read_text(encoding="utf-8"))
    assert persisted["schema_version"] == artifact["schema_version"]
    assert persisted["benchmark"]["fingerprint"] == artifact["benchmark"]["fingerprint"]

    case_ids = {run["case_id"] for run in artifact["runs"]}
    assert case_ids == {"hello-file", "fix-typo", "plan-refactor"}
    pass_rate = artifact["aggregates"]["overall"]["metrics"]["pass_rate"]["value"]
    assert pass_rate == 1.0
    # HH-04 transcripts attach as per-case evidence.
    for run in artifact["runs"]:
        assert run["transcript"]["events"] > 0


@pytest.mark.e2e
@pytest.mark.agent
def test_shipped_cases_pass_and_seeded_regression_trips_named_gate(temp_dir, mock_llm_config):
    """All shipped cases pass; an impossible baseline makes the pass_rate gate fail by name."""
    runner = CliRunner()
    first = run_agent_eval(runner, temp_dir, "--json")
    assert first.exit_code == 0, flat(first.output)
    artifact = parse_json_output(first.output)
    assert artifact["aggregates"]["overall"]["metrics"]["pass_rate"]["value"] == 1.0

    degraded_path = temp_dir / "degraded-baseline.json"
    degraded = json.loads(json.dumps(artifact))
    # An impossible baseline pass_rate makes the fresh run (1.0) a regression
    # far beyond tolerance; the gate must be named in the output.
    degraded["aggregates"]["overall"]["metrics"]["pass_rate"]["value"] = 2.0
    degraded_path.write_text(json.dumps(degraded), encoding="utf-8")

    failing = run_agent_eval(runner, temp_dir, "--baseline", str(degraded_path), "--fail-on-regression")
    assert failing.exit_code == 1
    flat_output = flat(failing.output)
    assert "pass_rate" in flat_output
    assert "Gate failed" in flat_output


@pytest.mark.e2e
@pytest.mark.agent
def test_forbidden_path_violation_is_scored_as_failure(temp_dir, mock_llm_config):
    """A scripted run that overwrites a forbidden setup file fails the case."""
    case = {
        "schema_version": 1,
        "name": "forbidden-probe",
        "cases": [
            {
                "id": "touch-forbidden",
                "instruction": "Rewrite the readme",
                "cohort": "file-ops",
                "split": "test",
                "setup": {"files": {"README.md": "original contents\n"}},
                "expected_checks": [{"command": "true", "description": "always passes"}],
                "forbidden_paths": ["README.md"],
                "plan_required": False,
                "max_iterations": 8,
                "mock_script": [
                    {
                        "content": "Reading the readme first.",
                        "tool_calls": [{"name": "read_file", "parameters": {"path": "README.md"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                    {
                        "content": "Rewriting the readme.",
                        "tool_calls": [
                            {"name": "write_file", "parameters": {"path": "README.md", "content": "clobbered"}}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                    {"content": "Done.", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                ],
            }
        ],
    }
    benchmark_path = write_case_benchmark(temp_dir, case)
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "agent", str(benchmark_path), "--project-path", str(temp_dir), "--json"])
    assert result.exit_code == 0, flat(result.output)  # the run completes; the case fails
    artifact = parse_json_output(result.output)
    judgment = {j["name"]: j for j in artifact["runs"][0]["judgments"]}
    assert judgment["forbidden_paths"]["passed"] is False
    assert "README.md" in (judgment["forbidden_paths"]["reason"] or "")
    assert artifact["aggregates"]["overall"]["metrics"]["pass_rate"]["value"] == 0.0


@pytest.mark.e2e
@pytest.mark.agent
def test_budget_overrun_is_scored_as_failure(temp_dir, mock_llm_config):
    """A case whose script needs more iterations than allowed fails the budget judgment."""
    case = {
        "schema_version": 1,
        "name": "budget-probe",
        "cases": [
            {
                "id": "over-budget",
                "instruction": "Create a file",
                "cohort": "file-ops",
                "split": "test",
                "setup": {"files": {}},
                "expected_checks": [{"command": "test -f done.txt", "description": "file exists"}],
                "forbidden_paths": [],
                "plan_required": False,
                "max_iterations": 1,
                "mock_script": [
                    {
                        "content": "Creating the file.",
                        "tool_calls": [{"name": "write_file", "parameters": {"path": "done.txt", "content": "x"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                    {"content": "Done.", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                ],
            }
        ],
    }
    benchmark_path = write_case_benchmark(temp_dir, case)
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "agent", str(benchmark_path), "--project-path", str(temp_dir), "--json"])
    assert result.exit_code == 0, flat(result.output)
    artifact = parse_json_output(result.output)
    judgment = {j["name"]: j for j in artifact["runs"][0]["judgments"]}
    assert judgment["budget"]["passed"] is False
    assert "iteration" in (judgment["budget"]["reason"] or "").lower()


@pytest.mark.e2e
@pytest.mark.agent
def test_mock_provider_conformance_runs_without_network(temp_dir):
    """The mock conformance suite verifies declared PROVIDER_SPECS capabilities offline."""
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "providers", "--project-path", str(temp_dir), "--json"])
    assert result.exit_code == 0, flat(result.output)
    report = parse_json_output(result.output)
    assert report["network"] is False
    assert report["reports"], "expected one conformance report"
    mock_report = report["reports"][0]
    assert mock_report["provider"] == "mock"
    assert mock_report["checks"], "expected conformance checks"
    for check in mock_report["checks"]:
        assert check["passed"] is True and check["drift"] is False, check


@pytest.mark.e2e
@pytest.mark.agent
def test_baseline_comparison_against_shipped_baseline_passes(temp_dir, mock_llm_config):
    """A fresh deterministic run compares clean against the checked-in baseline."""
    runner = CliRunner()
    result = run_agent_eval(runner, temp_dir, "--baseline", str(BASELINE_PATH), "--fail-on-regression", "--json")
    assert result.exit_code == 0, flat(result.output)
    artifact = parse_json_output(result.output)
    assert artifact["comparison"]["status"] == "pass"
    assert artifact["comparison"]["compatible"] is True
