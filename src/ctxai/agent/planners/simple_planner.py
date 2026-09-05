"""Rule-based planner — splits a message into atomic steps using heuristics."""

from __future__ import annotations

import re

from ctxai.agent.planners.base import BasePlanner, heuristic_requires_planning
from ctxai.agent.planning import Plan, create_plan


class SimplePlanner(BasePlanner):
    """Heuristic planner. No LLM calls; suitable for cheap pre-planning."""

    def __init__(self, min_steps: int = 2, max_steps: int = 8):
        self.min_steps = min_steps
        self.max_steps = max_steps

    def should_plan(self, message: str) -> bool:
        return heuristic_requires_planning(message)

    async def create_plan(self, message: str, context: dict | None = None) -> Plan:
        steps = self._extract_steps(message)
        if len(steps) < self.min_steps:
            steps = self._default_steps(message)
        steps = steps[: self.max_steps]
        return create_plan(
            goal=message.strip(),
            steps=[{"description": s, "agent_type": "coder"} for s in steps],
        )

    @staticmethod
    def _extract_steps(message: str) -> list[str]:
        # Look for numbered or bulleted lists in the prompt itself.
        bullets = re.findall(r"(?m)^[\s\-•*]*(?:\d+[\.\)]\s+)?(.+)$", message)
        cleaned = [b.strip() for b in bullets if 5 < len(b.strip()) < 240]
        # Heuristic: if there are multiple short imperative-looking lines, use them.
        candidates = [c for c in cleaned if re.match(r"^[A-Za-z]", c)]
        return candidates if len(candidates) >= 2 else []

    @staticmethod
    def _default_steps(message: str) -> list[str]:
        return [
            f"Investigate the request: {message.strip()[:120]}",
            "Identify files and modules involved",
            "Apply changes incrementally",
            "Verify with tests or a manual check",
        ]
