"""Unit tests for HH-09 agent task benchmark schema, fingerprints, artifact
redaction, and baseline gate comparison.

Mirrors the RE-01 artifact discipline: one fingerprinting, redaction, and
comparison approach shared by both eval frameworks.
"""

import json
from pathlib import Path

import pytest

from ctxai.evals.agent_artifacts import (
    AGENT_GATED_METRICS,
    AGENT_METRIC_DIRECTIONS,
    AGENT_TOLERANCES,
    AgentEvalArtifact,
    agent_default_artifact_path,
    agent_evaluations_dir,
    compare_agent_with_baseline,
    save_agent_artifact,
)
from ctxai.evals.artifacts import CohortMetricsBlock
from ctxai.evals.benchmark import VALID_SPLITS
from ctxai.evals.common import MetricValue, content_fingerprint, strip_volatile
from ctxai.evals.task_benchmark import (
    AGENT_BENCHMARK_SCHEMA_VERSION,
    AgentTaskBenchmark,
    AgentTaskCase,
    BenchmarkValidationError,
    agent_benchmark_from_payload,
    load_agent_benchmark,
    validate_agent_benchmark_payload,
)


def _case_payload(case_id: str = "hello-file", **overrides) -> dict:
    payload = {
        "id": case_id,
        "instruction": "Create a file named hello.txt containing exactly the word hello",
        "setup": {"files": {"README.md": "# fixture\n"}},
        "expected_checks": [{"command": "cat hello.txt", "description": "contains hello", "expect_output": "hello"}],
        "forbidden_paths": ["README.md"],
        "plan_required": False,
        "max_iterations": 8,
        "tags": ["files"],
        "cohort": "file-ops",
        "split": "test",
    }
    payload.update(overrides)
    return payload


def _benchmark_payload(**overrides) -> dict:
    payload = {
        "schema_version": AGENT_BENCHMARK_SCHEMA_VERSION,
        "name": "unit-agent-benchmark",
        "description": "schema test",
        "cases": [_case_payload(), _case_payload("fix-typo", cohort="file-ops", split="dev")],
    }
    payload.update(overrides)
    return payload


class TestBenchmarkValidation:
    def test_valid_payload_round_trips(self):
        benchmark = agent_benchmark_from_payload(_benchmark_payload())
        assert benchmark.name == "unit-agent-benchmark"
        assert [case.id for case in benchmark.cases] == ["hello-file", "fix-typo"]
        restored = AgentTaskBenchmark.from_dict(benchmark.to_dict())
        assert restored == benchmark

    def test_case_round_trip_preserves_script_and_checks(self):
        script = [{"content": "done", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        case = AgentTaskCase.from_dict(_case_payload("scripted", mock_script=script))
        assert case.mock_script == script
        assert AgentTaskCase.from_dict(case.to_dict()) == case

    def test_duplicate_ids_rejected(self):
        payload = _benchmark_payload(cases=[_case_payload("dup"), _case_payload("dup")])
        errors = validate_agent_benchmark_payload(payload)
        assert any("duplicate case id" in error for error in errors)

    def test_invalid_split_rejected(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", split="holdout")])
        errors = validate_agent_benchmark_payload(payload)
        assert any("split must be one of" in error for error in errors)
        assert set(VALID_SPLITS) == {"train", "dev", "test"}

    def test_absolute_setup_path_rejected(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", setup={"files": {"/etc/passwd": "x"}})])
        errors = validate_agent_benchmark_payload(payload)
        assert any("repository-relative" in error for error in errors)

    def test_parent_traversal_setup_path_rejected(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", setup={"files": {"../escape.txt": "x"}})])
        errors = validate_agent_benchmark_payload(payload)
        assert any("'..'" in error for error in errors)

    def test_empty_checks_rejected(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", expected_checks=[])])
        errors = validate_agent_benchmark_payload(payload)
        assert any("expected_checks" in error for error in errors)

    def test_check_without_command_rejected(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", expected_checks=[{"description": "no command"}])])
        errors = validate_agent_benchmark_payload(payload)
        assert any("command" in error for error in errors)

    def test_max_iterations_bounds_enforced(self):
        for bad in (0, -3, 10_000):
            payload = _benchmark_payload(cases=[_case_payload("c1", max_iterations=bad)])
            errors = validate_agent_benchmark_payload(payload)
            assert any("max_iterations" in error for error in errors), f"max_iterations={bad} must be rejected"

    def test_case_id_charset_enforced_for_transcript_safety(self):
        payload = _benchmark_payload(cases=[_case_payload("Bad Id!")])
        errors = validate_agent_benchmark_payload(payload)
        assert any("id" in error.lower() for error in errors)

    def test_mock_script_must_be_a_list_of_objects(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", mock_script=["not an object"])])
        errors = validate_agent_benchmark_payload(payload)
        assert any("mock_script" in error for error in errors)

    def test_setup_must_be_files_mapping(self):
        payload = _benchmark_payload(cases=[_case_payload("c1", setup={"dirs": ["x"]})])
        errors = validate_agent_benchmark_payload(payload)
        assert any("setup" in error for error in errors)

    def test_invalid_payload_raises_with_all_errors(self):
        with pytest.raises(BenchmarkValidationError) as excinfo:
            agent_benchmark_from_payload({"schema_version": 99, "name": "", "cases": []})
        assert len(excinfo.value.errors) >= 3

    def test_load_agent_benchmark_rejects_oversized_files(self, tmp_path: Path):
        path = tmp_path / "big.json"
        path.write_text("x" * 1_100_000, encoding="utf-8")
        with pytest.raises(BenchmarkValidationError):
            load_agent_benchmark(path)

    def test_load_agent_benchmark_reads_checked_in_fixture(self):
        fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "agent_benchmark" / "benchmark.json"
        if not fixture.is_file():  # pragma: no cover - e2e fixture lands with the slice
            pytest.fail("shipped agent benchmark fixture is missing")
        benchmark = load_agent_benchmark(fixture)
        assert benchmark.cases
        assert not validate_agent_benchmark_payload(benchmark.to_dict())


class TestFingerprints:
    def test_fingerprint_is_content_derived_and_stable(self):
        first = agent_benchmark_from_payload(_benchmark_payload())
        second = agent_benchmark_from_payload(_benchmark_payload())
        assert first.fingerprint == second.fingerprint
        assert first.fingerprint == content_fingerprint(first.to_dict())

    def test_fingerprint_changes_when_any_case_content_changes(self):
        base = agent_benchmark_from_payload(_benchmark_payload())
        changed_script = _benchmark_payload(
            cases=[
                _case_payload(),
                _case_payload("fix-typo", cohort="file-ops", split="dev", mock_script=[{"content": "x"}]),
            ]
        )
        changed = agent_benchmark_from_payload(changed_script)
        assert base.fingerprint != changed.fingerprint

    def test_mock_script_is_part_of_the_fingerprint(self):
        scripted = _case_payload("c1", mock_script=[{"content": "hello"}])
        plain = _case_payload("c1")
        assert (
            agent_benchmark_from_payload(_benchmark_payload(cases=[scripted])).fingerprint
            != agent_benchmark_from_payload(_benchmark_payload(cases=[plain])).fingerprint
        )


def _run_record(case_id: str, *, passed: bool, iterations: int = 3) -> dict:
    return {
        "case_id": case_id,
        "run_id": "eval-run",
        "timestamp": "2026-09-04T00:00:00+00:00",
        "cohort": "file-ops",
        "split": "test",
        "status": "passed" if passed else "failed",
        "error": None,
        "iterations": iterations,
        "max_iterations": 8,
        "tokens": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "calls": iterations},
        "cost": {"available": False, "value": None, "reason": "no price entry for mock-model-v1"},
        "judgments": [{"name": "checks", "passed": passed, "reason": None if passed else "check failed"}],
        "checks": [],
        "forbidden_paths": [],
        "plan": {"required": False, "submitted": False, "actions": 0, "actions_completed": 0},
        "approvals": {"mutations": 0, "approved_records": 0, "unapproved": []},
        "changed_files": [],
        "transcript": {"path": ".ctxai/runs/eval-x.jsonl", "run_id": f"eval-{case_id}", "events": 10},
    }


def _artifact_payload(*, pass_rate: float = 1.0, mean_iterations: float = 3.0) -> dict:
    passed = pass_rate >= 1.0
    metrics = {
        "pass_rate": {"available": True, "value": pass_rate},
        "mean_iterations": {"available": True, "value": mean_iterations},
        "p95_iterations": {"available": True, "value": mean_iterations},
        "token_mean": {"available": True, "value": 120.0},
        "cost_total": {"available": False, "value": None, "reason": "no price entry for mock-model-v1"},
    }
    block = {"cases": 1, "successful": int(passed), "errored": 0, "metrics": metrics, "confidence_intervals": {}}
    return {
        "schema_version": 1,
        "kind": "agent",
        "run_id": "run-current",
        "created_at": "2026-09-04T00:00:00+00:00",
        "duration_ms": 1.0,
        "status": "complete",
        "benchmark": {"name": "bench", "fingerprint": "f" * 64, "schema_version": 1, "case_count": 1},
        "configuration": {"fingerprint": "c" * 64, "provider": {"mode": "mock", "model": "mock-model-v1"}},
        "environment": {"python_version": "3.13"},
        "runs": [_run_record("c1", passed=passed)],
        "aggregates": {"overall": block, "by_cohort": {"file-ops": block}, "by_split": {"test": block}},
        "comparison": None,
        "errors": [],
    }


class TestBaselineComparison:
    def test_identical_deterministic_run_passes_gates(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload())
        comparison = compare_agent_with_baseline(current, _artifact_payload())
        assert comparison.compatible is True
        assert comparison.status == "pass"
        assert all(gate.status == "pass" for gate in comparison.gates)

    def test_pass_rate_regression_is_named(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload(pass_rate=0.5))
        comparison = compare_agent_with_baseline(current, _artifact_payload(pass_rate=1.0))
        assert comparison.status == "regression"
        failing = [gate for gate in comparison.gates if gate.status == "regression"]
        assert {(gate.cohort, gate.metric) for gate in failing} == {("overall", "pass_rate"), ("file-ops", "pass_rate")}

    def test_iteration_increase_within_tolerance_passes(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload(mean_iterations=3.5))
        comparison = compare_agent_with_baseline(current, _artifact_payload(mean_iterations=3.0))
        assert comparison.status == "pass"

    def test_kind_mismatch_is_incompatible(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload())
        baseline = _artifact_payload()
        baseline["kind"] = "retrieval"
        comparison = compare_agent_with_baseline(current, baseline)
        assert comparison.status == "incompatible"
        assert any("kind" in item for item in comparison.incompatibilities)

    def test_benchmark_fingerprint_mismatch_is_incompatible(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload())
        baseline = _artifact_payload()
        baseline["benchmark"]["fingerprint"] = "0" * 64
        comparison = compare_agent_with_baseline(current, baseline)
        assert comparison.status == "incompatible"
        assert any("benchmark fingerprint" in item for item in comparison.incompatibilities)

    def test_case_set_difference_is_incompatible(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload())
        baseline = _artifact_payload()
        baseline["runs"] = [_run_record("other-case", passed=True)]
        comparison = compare_agent_with_baseline(current, baseline)
        assert comparison.status == "incompatible"
        assert any("case set" in item for item in comparison.incompatibilities)

    def test_agent_gates_cover_the_declared_metric_set(self):
        current = AgentEvalArtifact.from_dict(_artifact_payload())
        comparison = compare_agent_with_baseline(current, _artifact_payload())
        gated = {(gate.cohort, gate.metric) for gate in comparison.gates if gate.cohort == "overall"}
        assert gated == {("overall", metric) for metric in AGENT_GATED_METRICS}
        for metric in AGENT_GATED_METRICS:
            assert metric in AGENT_METRIC_DIRECTIONS
            assert metric in AGENT_TOLERANCES

    def test_unavailable_gate_is_not_a_regression(self):
        current_payload = _artifact_payload()
        current_payload["aggregates"]["overall"]["metrics"]["pass_rate"] = {
            "available": False,
            "value": None,
            "reason": "no cases",
        }
        current = AgentEvalArtifact.from_dict(current_payload)
        comparison = compare_agent_with_baseline(current, _artifact_payload())
        assert comparison.status == "pass"
        unavailable = [gate for gate in comparison.gates if gate.status == "unavailable"]
        assert unavailable, "the unavailable metric is reported, never treated as zero"


class TestArtifactStorage:
    def test_default_paths_live_under_evaluations_agent(self, tmp_path: Path):
        directory = agent_evaluations_dir(tmp_path)
        assert directory == tmp_path / ".ctxai" / "evaluations" / "agent"
        path = agent_default_artifact_path(tmp_path, "My Benchmark", "abcdef1234", "2026-09-04T10:00:00+00:00")
        assert path.parent == directory
        assert path.name == "My-Benchmark-20260904T100000+0000-abcdef12.json"

    def test_save_redacts_absolute_paths_and_secrets(self, tmp_path: Path):
        artifact = AgentEvalArtifact.from_dict(_artifact_payload())
        # Environment is a free-form evidence dict; embed a workspace path and
        # a seeded secret to prove the redaction pipeline runs on save.
        artifact.environment["workspace"] = str(tmp_path / "workspaces" / "run-1")
        artifact.environment["leak"] = "api_key=sk-test-0123456789abcdef"
        destination = tmp_path / "artifact.json"
        save_agent_artifact(artifact, destination, tmp_path)

        saved = json.loads(destination.read_text(encoding="utf-8"))
        assert str(tmp_path) not in json.dumps(saved)
        assert "sk-test-0123456789abcdef" not in json.dumps(saved)
        assert "<project>" in json.dumps(saved)
        assert "[REDACTED]" in json.dumps(saved)

    def test_strip_volatile_enables_byte_stable_comparison(self):
        first = _artifact_payload()
        second = _artifact_payload()
        second["run_id"] = "other-run"
        second["created_at"] = "2027-01-01T00:00:00+00:00"
        second["duration_ms"] = 99.0
        second["runs"][0]["timestamp"] = "2027-01-01T00:00:00+00:00"
        second["runs"][0]["run_id"] = "other-run"
        second["runs"][0]["transcript"]["run_id"] = "eval-other"
        assert strip_volatile(first) == strip_volatile(second)

    def test_artifact_comparison_block_round_trip(self):
        artifact = AgentEvalArtifact.from_dict(_artifact_payload())
        compared = compare_agent_with_baseline(artifact, _artifact_payload())
        from dataclasses import replace

        artifact_with_comparison = replace(artifact, comparison=compared)
        restored = AgentEvalArtifact.from_dict(artifact_with_comparison.to_dict())
        assert restored.comparison == compared
        assert restored == artifact_with_comparison


class TestSharedDiscipline:
    def test_agent_and_retrieval_share_metric_value_and_gate_models(self):
        from ctxai.evals.artifacts import GateResult, MetricTolerance

        gate = GateResult(
            cohort="overall",
            metric="pass_rate",
            baseline=1.0,
            current=0.5,
            delta=-0.5,
            direction="higher",
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            status="regression",
        )
        assert gate.to_dict()["status"] == "regression"
        assert GateResult.from_dict(gate.to_dict()) == gate
        assert isinstance(AGENT_TOLERANCES["pass_rate"], MetricTolerance)
        assert MetricValue.available(1.0).is_available
        assert CohortMetricsBlock(0, 0, 0, {}).metrics == {}
