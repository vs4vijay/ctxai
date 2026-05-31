"""
planning_workflow.py — Use Plan / PlanExecutor to break a goal into steps.
"""

import asyncio

from ctxai_core import Plan, PlanExecutor, create_plan


async def main() -> None:
    plan = create_plan(
        goal="Add a CLI flag to enable verbose output",
        steps=[
            {"description": "Locate the CLI entry point", "agent_type": "explorer"},
            {"description": "Add the --verbose argument", "agent_type": "coder"},
            {"description": "Wire it to logging", "agent_type": "coder"},
            {"description": "Add a test", "agent_type": "tester"},
        ],
    )
    # Chain second step on first, last on rest.
    plan.steps[1].dependencies = [plan.steps[0].step_id]
    plan.steps[2].dependencies = [plan.steps[1].step_id]
    plan.steps[3].dependencies = [plan.steps[2].step_id]

    async def runner(step):
        print(f"[exec] {step.description}")
        await asyncio.sleep(0)
        return {"success": True, "result": f"completed: {step.description}"}

    result = await PlanExecutor(plan).execute(runner)
    print("Plan finished:", result["success"])
    print("Progress:", plan.get_progress())


if __name__ == "__main__":
    asyncio.run(main())
