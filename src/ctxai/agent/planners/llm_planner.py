"""LLM-driven planner — asks the model itself to produce a plan."""

from __future__ import annotations

import json
import re
from typing import Any

from ctxai.agent.llm.base import BaseLLMProvider, Message, MessageRole
from ctxai.agent.planners.base import BasePlanner, heuristic_requires_planning
from ctxai.agent.planning import Plan, create_plan

_PLAN_PROMPT = """You are a planning assistant. Break the user's request into 2-8 concrete steps.

Respond ONLY with a JSON array of objects with keys:
- description (string)
- agent_type (string, one of: explorer, coder, reviewer, orchestrator)
- tools_needed (array of strings, optional)
- dependencies (array of step indices that must complete first, optional)

Do not include any commentary.

User request:
{message}
"""


class LLMPlanner(BasePlanner):
    """Uses a BaseLLMProvider to generate a Plan."""

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    def should_plan(self, message: str) -> bool:
        return heuristic_requires_planning(message)

    async def create_plan(self, message: str, context: dict | None = None) -> Plan:
        prompt = _PLAN_PROMPT.format(message=message.strip())
        response = self.llm.chat([Message(role=MessageRole.USER, content=prompt)])
        steps = self._parse_steps(response.content or "")
        if not steps:
            steps = [
                {
                    "description": "Investigate request",
                    "agent_type": "explorer",
                },
                {
                    "description": message.strip()[:240],
                    "agent_type": "coder",
                },
            ]
        return create_plan(goal=message.strip(), steps=steps)

    @staticmethod
    def _parse_steps(raw: str) -> list[dict[str, Any]]:
        text = raw.strip()
        # Strip optional code fences / labels.
        fence = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        steps: list[dict[str, Any]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or "description" not in item:
                continue
            agent_type = item.get("agent_type", "coder")
            deps_in = item.get("dependencies", [])
            # We don't yet have step_ids, so convert indices to placeholders;
            # the caller will translate them after create_plan() generates ids.
            steps.append(
                {
                    "description": str(item["description"])[:300],
                    "agent_type": agent_type,
                    "tools_needed": [str(t) for t in item.get("tools_needed", [])],
                    "dependencies": [int(d) for d in deps_in if isinstance(d, (int, float))],
                }
            )
        # Convert numeric dependencies to actual step ids after the fact.
        # Because create_plan assigns uuids, we resolve in two passes.
        return steps
