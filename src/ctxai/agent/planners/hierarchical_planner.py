"""Hierarchical planner — produces a top-level plan with sub-plans per step."""

from __future__ import annotations

from ctxai.agent.planners.base import BasePlanner, heuristic_requires_planning
from ctxai.agent.planners.simple_planner import SimplePlanner
from ctxai.agent.planning import Plan, PlanStep, create_plan


class HierarchicalPlanner(BasePlanner):
    """
    Wraps another planner to add sub-step decomposition.

    The outer plan contains "phases"; each phase's metadata holds a
    nested Plan with concrete steps. This is a simple, deterministic
    decomposition that doesn't require an LLM.
    """

    PHASES = (
        "Understand the request",
        "Design the change",
        "Implement",
        "Verify",
    )

    def __init__(self, inner: BasePlanner | None = None):
        self.inner = inner or SimplePlanner()

    def should_plan(self, message: str) -> bool:
        return heuristic_requires_planning(message)

    async def create_plan(self, message: str, context: dict | None = None) -> Plan:
        outer = create_plan(
            goal=message.strip(),
            steps=[
                {"description": phase, "agent_type": "orchestrator"}
                for phase in self.PHASES
            ],
        )
        # Attach inner plan as metadata on the implementation phase.
        impl_step = next((s for s in outer.steps if s.description == "Implement"), None)
        if impl_step is not None:
            inner_plan = await self.inner.create_plan(message, context)
            outer.metadata["implementation_plan"] = inner_plan.to_dict()
        return outer
