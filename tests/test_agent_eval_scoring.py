"""Unit tests for HH-09 agent task scoring: judgments, forbidden-path
detection, budget math, plan/approval workflow judgments, and aggregate
metric computation.

Pure functions over plain values so scoring is directly testable without an
agent loop; the runner applies them to real runs (covered by e2e).
"""

import pytest

from ctxai.evals.agent_artifacts import AgentCaseRunRecord, AgentEvalArtifact
from ctxai.evals.common import MetricValue, content_fingerprint
from ctxai.evals.scoring import (
    CaseJudgment,
    aggregate_case_records,
    case_is_passed,
    judge_approvals,
    judge_budget,
    judge_checks,
    judge_forbidden_paths,
    judge_plan,
)


def _record(case_id: str = "c1", *, status: str = "passed", iterations: int = 3, tokens: int = 360, **overrides):
    defaults = {
        "case_id": case_id,
        "run_id": "eval-run",
        "timestamp": "2026-09-04T00:00:00+00:00",
        "cohort": "file-ops",
        "split": "test",
        "status": status,
        "error": None if status != "error" else "case failed",
        "iterations": iterations,
        "max_iterations": 10,
        "tokens": {"prompt_tokens": tokens - 60, "completion_tokens": 60, "total_tokens": tokens, "calls": iterations},
        "cost": MetricValue.unavailable("no price entry for mock-model-v1"),
        "judgments": [CaseJudgment(name="checks", passed=status == "passed")],
        "checks": [{"command": "cat out.txt", "description": "content", "passed": status == "passed"}],
        "forbidden_paths": [],
        "plan": {"required": False, "submitted": False, "actions": 0, "actions_completed": 0},
        "approvals": {"mutations": 0, "approved_records": 0, "unapproved": []},
        "changed_files": [],
        "transcript": {"path": ".ctxai/runs/eval-c1.jsonl", "run_id": "eval-c1", "events": 12},
    }
    defaults.update(overrides)
    return AgentCaseRunRecord(**defaults)


class TestCheckJudgment:
    def test_all_checks_pass(self):
        results = [
            {"command": "cat hello.txt", "description": "content check", "passed": True, "exit_code": 0},
            {"command": "grep -c hello hello.txt", "description": "count", "passed": True, "exit_code": 0},
        ]
        judgment = judge_checks(results)
        assert judgment.passed
        assert judgment.name == "checks"
        assert judgment.reason is None

    def test_failing_check_fails_with_reason(self):
        results = [{"command": "cat hello.txt", "description": "content check", "passed": False, "exit_code": 1}]
        judgment = judge_checks(results)
        assert not judgment.passed
        assert "cat hello.txt" in judgment.reason

    def test_missing_check_execution_fails(self):
        judgment = judge_checks([])
        assert not judgment.passed
        assert "no check" in judgment.reason.lower()

    def test_expectation_marker_mismatch_is_reported(self):
        results = [
            {
                "command": "cat hello.txt",
                "description": "content check",
                "passed": False,
                "exit_code": 0,
                "expect_output": "hello",
                "output_matched": False,
            }
        ]
        judgment = judge_checks(results)
        assert not judgment.passed
        assert "hello" in judgment.reason


class TestForbiddenPathJudgment:
    def test_untouched_paths_pass(self):
        findings = [
            {"path": "README.md", "untouched": True, "detail": "byte-identical to setup content"},
            {"path": "notes.txt", "untouched": True, "detail": "does not exist"},
        ]
        judgment = judge_forbidden_paths(findings)
        assert judgment.passed

    def test_modified_forbidden_path_fails(self):
        findings = [{"path": "README.md", "untouched": False, "detail": "content differs from setup"}]
        judgment = judge_forbidden_paths(findings)
        assert not judgment.passed
        assert "README.md" in judgment.reason

    def test_created_forbidden_path_fails(self):
        findings = [{"path": "extra.txt", "untouched": False, "detail": "forbidden path exists but was not in setup"}]
        judgment = judge_forbidden_paths(findings)
        assert not judgment.passed


class TestBudgetJudgment:
    def test_success_within_budget_passes(self):
        judgment = judge_budget(succeeded=True, iterations=4, max_iterations=8)
        assert judgment.passed

    def test_failed_run_fails_budget(self):
        judgment = judge_budget(succeeded=False, iterations=8, max_iterations=8)
        assert not judgment.passed
        assert "8" in judgment.reason

    def test_iteration_overrun_is_reported_even_when_state_says_succeeded(self):
        judgment = judge_budget(succeeded=True, iterations=11, max_iterations=10)
        assert not judgment.passed
        assert "iteration" in judgment.reason.lower()


class TestPlanJudgment:
    def test_plan_required_and_submitted_passes(self):
        judgment = judge_plan(
            plan_required=True,
            plan={"required": True, "submitted": True, "actions": 2, "actions_completed": 2},
        )
        assert judgment.passed

    def test_plan_required_but_missing_fails(self):
        judgment = judge_plan(
            plan_required=True,
            plan={"required": True, "submitted": False, "actions": 0, "actions_completed": 0},
        )
        assert not judgment.passed
        assert "submit_plan" in judgment.reason

    def test_plan_not_required_passes_regardless(self):
        assert judge_plan(plan_required=False, plan=None).passed


class TestApprovalJudgment:
    def test_no_mutations_passes(self):
        assert judge_approvals(unapproved_mutations=[]).passed

    def test_unapproved_mutation_fails(self):
        judgment = judge_approvals(unapproved_mutations=["hello.txt"])
        assert not judgment.passed
        assert "hello.txt" in judgment.reason


class TestCaseComposition:
    def test_case_is_passed_requires_every_judgment(self):
        assert case_is_passed([CaseJudgment("checks", True), CaseJudgment("budget", True)])
        assert not case_is_passed([CaseJudgment("checks", True), CaseJudgment("budget", False, "over budget")])

    def test_judgment_round_trip(self):
        judgment = CaseJudgment(name="checks", passed=False, reason="cat hello.txt failed")
        restored = CaseJudgment.from_dict(judgment.to_dict())
        assert restored == judgment


class TestAggregation:
    def test_pass_rate_and_iteration_percentiles(self):
        records = [
            _record("c1", iterations=2),
            _record("c2", iterations=4),
            _record("c3", status="failed", iterations=9),
        ]
        block = aggregate_case_records(records)
        assert block.metrics["pass_rate"].value == pytest.approx(2 / 3)
        assert block.metrics["mean_iterations"].value == pytest.approx(5.0)
        assert block.metrics["p95_iterations"].value == pytest.approx(8.5)
        assert block.cases == 3
        assert block.successful == 2
        assert block.errored == 0

    def test_token_mean_and_cost_aggregation(self):
        records = [
            _record("c1", tokens=360),
            _record("c2", tokens=480),
        ]
        block = aggregate_case_records(records)
        assert block.metrics["token_mean"].value == pytest.approx(420.0)
        cost = block.metrics["cost_total"]
        assert not cost.is_available
        assert "price" in cost.reason

    def test_error_cases_are_counted_and_excluded_from_means(self):
        records = [
            _record("c1", iterations=2),
            _record("c2", status="error", iterations=0, error="timeout"),
        ]
        block = aggregate_case_records(records)
        assert block.metrics["pass_rate"].value == pytest.approx(0.5)
        assert block.errored == 1
        assert block.metrics["mean_iterations"].value == pytest.approx(2.0)

    def test_known_costs_sum_when_every_case_is_priced(self):
        records = [
            _record("c1", cost=MetricValue.available(0.5)),
            _record("c2", cost=MetricValue.available(1.0)),
        ]
        block = aggregate_case_records(records)
        assert block.metrics["cost_total"].value == pytest.approx(1.5)

    def test_empty_group_is_explicitly_unavailable(self):
        block = aggregate_case_records([])
        assert not block.metrics["pass_rate"].is_available
        assert block.metrics["pass_rate"].reason


class TestRecordRoundTrip:
    def test_case_record_round_trip(self):
        record = _record("round-trip")
        restored = AgentCaseRunRecord.from_dict(record.to_dict())
        assert restored == record

    def test_artifact_round_trip(self):
        from ctxai.evals.artifacts import CohortMetricsBlock

        artifact = AgentEvalArtifact(
            schema_version=1,
            kind="agent",
            run_id="run-1",
            created_at="2026-09-04T00:00:00+00:00",
            duration_ms=12.5,
            status="complete",
            benchmark={
                "name": "bench",
                "fingerprint": content_fingerprint({"a": 1}),
                "schema_version": 1,
                "case_count": 1,
            },
            configuration={"fingerprint": content_fingerprint({"b": 2}), "provider": {"mode": "mock"}},
            environment={"python_version": "3.13"},
            runs=[_record("c1")],
            aggregates={"overall": CohortMetricsBlock(1, 1, 0, {}), "by_cohort": {}, "by_split": {}},
            comparison=None,
            errors=[],
        )
        restored = AgentEvalArtifact.from_dict(artifact.to_dict())
        assert restored == artifact
