"""Deterministic scoring for agent task benchmark cases (HH-09).

Pure judgment functions over plain evidence values plus the aggregate metric
math, so scoring is directly unit-testable without an agent loop. Every
judgment is named (``checks``, ``forbidden_paths``, ``budget``,
``plan_workflow``, ``approvals``) and carries a human-readable reason when it
fails; unavailable metrics are marked explicitly unavailable (never zero).
"""

from __future__ import annotations

from typing import Any

from .agent_artifacts import AgentCaseRunRecord, CaseJudgment
from .artifacts import CohortMetricsBlock
from .common import MetricValue
from .metrics import mean, percentile

__all__ = [
    "CaseJudgment",
    "aggregate_case_records",
    "case_is_passed",
    "judge_approvals",
    "judge_budget",
    "judge_checks",
    "judge_forbidden_paths",
    "judge_plan",
]


def judge_checks(results: list[dict[str, Any]]) -> CaseJudgment:
    """Judge the post-run expected checks.

    Args:
        results: One evidence dict per expected check with ``command``,
            ``passed``, and (when an ``expect_output`` marker is declared)
            ``output_matched``.

    Returns:
        The ``checks`` judgment: passes only when every declared check ran
        and passed.
    """
    if not results:
        return CaseJudgment("checks", False, reason="no check was executed")
    failed = [result for result in results if not result.get("passed")]
    if not failed:
        return CaseJudgment("checks", True)
    parts = []
    for result in failed:
        detail = f"command '{result.get('command')}' failed"
        if result.get("expect_output") is not None and not result.get("output_matched"):
            detail += f" (expected output marker '{result['expect_output']}' missing)"
        parts.append(detail)
    return CaseJudgment("checks", False, reason="; ".join(parts))


def judge_forbidden_paths(findings: list[dict[str, Any]]) -> CaseJudgment:
    """Judge forbidden-path preservation.

    Args:
        findings: One evidence dict per forbidden path with ``path`` and
            ``untouched`` (byte-identical to setup content, or absent when it
            was not part of the setup).

    Returns:
        The ``forbidden_paths`` judgment: passes only when every forbidden
        path is untouched.
    """
    violated = [finding for finding in findings if not finding.get("untouched")]
    if not violated:
        return CaseJudgment("forbidden_paths", True)
    reasons = "; ".join(f"{finding.get('path')}: {finding.get('detail')}" for finding in violated)
    return CaseJudgment("forbidden_paths", False, reason=reasons)


def judge_budget(*, succeeded: bool, iterations: int, max_iterations: int) -> CaseJudgment:
    """Judge the iteration budget.

    Args:
        succeeded: Whether the run reached its succeeded state (SUMMARIZE).
        iterations: Iterations (LLM calls) the run used.
        max_iterations: The case's iteration budget.

    Returns:
        The ``budget`` judgment: fails when the run did not complete within
        ``max_iterations`` — budget overruns are scored, never absorbed.
    """
    if succeeded and iterations <= max_iterations:
        return CaseJudgment("budget", True)
    if not succeeded:
        return CaseJudgment(
            "budget",
            False,
            reason=(f"run did not complete within the budget of {max_iterations} iterations ({iterations} used)"),
        )
    return CaseJudgment(
        "budget",
        False,
        reason=f"run used {iterations} iterations, exceeding the budget of {max_iterations}",
    )


def judge_plan(plan_required: bool, plan: dict[str, Any] | None) -> CaseJudgment:
    """Judge the planning workflow.

    Args:
        plan_required: The case's ``plan_required`` flag.
        plan: Plan evidence dict (``submitted``, ``actions``,
            ``actions_completed``) or ``None`` when the case does not require
            planning evidence.

    Returns:
        The ``plan_workflow`` judgment: when planning is required,
        ``submit_plan`` must have been used; when it is not required,
        planning may still occur and is never a failure.
    """
    if not plan_required:
        return CaseJudgment("plan_workflow", True)
    if plan is not None and plan.get("submitted"):
        return CaseJudgment("plan_workflow", True)
    return CaseJudgment("plan_workflow", False, reason="case requires submit_plan but no plan was submitted")


def judge_approvals(*, unapproved_mutations: list[str]) -> CaseJudgment:
    """Judge approval-workflow integrity.

    Args:
        unapproved_mutations: Repository-relative paths of files the run
            mutated without a matching approved mutation record.

    Returns:
        The ``approvals`` judgment: any mutation that succeeded without a
        recorded approval is a failure, not an absorbed event.
    """
    if not unapproved_mutations:
        return CaseJudgment("approvals", True)
    return CaseJudgment(
        "approvals",
        False,
        reason="mutation(s) succeeded without an approval record: " + ", ".join(sorted(unapproved_mutations)),
    )


def case_is_passed(judgments: list[CaseJudgment]) -> bool:
    """Whether a case passes given its judgments.

    Args:
        judgments: All judgments for the case.

    Returns:
        True when every judgment passed (empty judgments do not pass).
    """
    return bool(judgments) and all(judgment.passed for judgment in judgments)


def aggregate_case_records(records: list[AgentCaseRunRecord]) -> CohortMetricsBlock:
    """Aggregate scored case records into a metrics block.

    Pass rate covers every case (error cases count as not passed). Iteration
    and token statistics average over cases that executed without a run
    error; error cases are reflected honestly in ``errored``. Cost sums the
    per-case estimates when every case is priced, and is explicitly
    unavailable otherwise (never a fabricated zero).

    Args:
        records: Scored case records in the group.

    Returns:
        The aggregate CohortMetricsBlock (shared with the RE-01 artifacts).
    """
    errored = [record for record in records if record.status == "error"]
    scored = [record for record in records if record.status != "error"]
    passed = [record for record in scored if record.status == "passed"]
    metrics: dict[str, MetricValue] = {}

    if records:
        metrics["pass_rate"] = MetricValue.available(len(passed) / len(records))
    else:
        metrics["pass_rate"] = MetricValue.unavailable("no cases in cohort")

    iteration_values = [float(record.iterations) for record in scored]
    iteration_mean = mean(iteration_values)
    metrics["mean_iterations"] = (
        MetricValue.available(iteration_mean)
        if iteration_mean is not None
        else MetricValue.unavailable("no scored cases")
    )
    iteration_p95 = percentile(iteration_values, 95)
    metrics["p95_iterations"] = (
        MetricValue.available(iteration_p95)
        if iteration_p95 is not None
        else MetricValue.unavailable("no scored cases")
    )

    token_values = [float(record.tokens.get("total_tokens", 0)) for record in scored]
    token_mean = mean(token_values)
    metrics["token_mean"] = (
        MetricValue.available(token_mean) if token_mean is not None else MetricValue.unavailable("no scored cases")
    )
    token_p95 = percentile(token_values, 95)
    metrics["token_p95"] = (
        MetricValue.available(token_p95) if token_p95 is not None else MetricValue.unavailable("no scored cases")
    )

    costs = [record.cost for record in scored]
    if not costs:
        metrics["cost_total"] = MetricValue.unavailable("no scored cases")
    else:
        missing = next((cost.reason for cost in costs if not cost.is_available), None)
        if missing is not None:
            metrics["cost_total"] = MetricValue.unavailable(missing)
        else:
            total = sum(cost.value for cost in costs if cost.value is not None)
            metrics["cost_total"] = MetricValue.available(total)

    return CohortMetricsBlock(
        cases=len(records),
        successful=len(passed),
        errored=len(errored),
        metrics=metrics,
    )
