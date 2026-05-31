"""Tests for ctxai.agent.planning."""

import pytest

from ctxai.agent.planning import (
    Plan,
    PlanExecutor,
    PlanStatus,
    PlanStep,
    StepStatus,
    create_plan,
)


def test_create_plan_assigns_ids():
    plan = create_plan(
        "Refactor module",
        [
            {"description": "Read code", "agent_type": "explorer"},
            {"description": "Apply changes", "agent_type": "coder"},
        ],
    )
    assert plan.goal == "Refactor module"
    assert len(plan.steps) == 2
    assert all(step.step_id for step in plan.steps)


def test_step_lifecycle():
    step = PlanStep(step_id="1", description="d", agent_type="coder")
    assert step.status == StepStatus.PENDING
    step.start()
    assert step.status == StepStatus.IN_PROGRESS
    step.complete("done")
    assert step.status == StepStatus.DONE
    assert step.result == "done"


def test_step_failure():
    step = PlanStep(step_id="1", description="d", agent_type="coder")
    step.fail("oops")
    assert step.status == StepStatus.FAILED
    assert step.error == "oops"


def test_get_next_steps_respects_dependencies():
    plan = create_plan(
        "Goal",
        [
            {"description": "s1", "agent_type": "coder"},
            {"description": "s2", "agent_type": "coder"},
        ],
    )
    plan.steps[1].dependencies = [plan.steps[0].step_id]
    nxt = plan.get_next_steps()
    assert len(nxt) == 1
    assert nxt[0].step_id == plan.steps[0].step_id


def test_plan_progress_metrics():
    plan = create_plan("g", [{"description": "x", "agent_type": "c"} for _ in range(4)])
    plan.steps[0].complete("ok")
    plan.steps[1].fail("nope")
    p = plan.get_progress()
    assert p["total"] == 4
    assert p["completed"] == 1
    assert p["failed"] == 1
    assert p["pending"] == 2


@pytest.mark.asyncio
async def test_plan_executor_runs_steps():
    plan = create_plan(
        "g",
        [
            {"description": "s1", "agent_type": "c"},
            {"description": "s2", "agent_type": "c"},
        ],
    )

    async def runner(step):
        return {"success": True, "result": f"did {step.description}"}

    result = await PlanExecutor(plan).execute(runner)
    assert result["success"] is True
    assert plan.status == PlanStatus.COMPLETED


@pytest.mark.asyncio
async def test_plan_executor_records_failure():
    plan = create_plan("g", [{"description": "s1", "agent_type": "c"}])

    async def runner(step):
        return {"success": False, "error": "no"}

    result = await PlanExecutor(plan).execute(runner)
    assert result["success"] is False
    assert plan.status == PlanStatus.FAILED


def test_plan_to_dict_serializable():
    plan = create_plan("g", [{"description": "s1", "agent_type": "c"}])
    d = plan.to_dict()
    assert d["goal"] == "g"
    assert "progress" in d
