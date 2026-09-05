"""Base class for planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ctxai.agent.planning import Plan

PLANNING_INDICATORS = (
    "refactor",
    "migrate",
    "add feature",
    "implement",
    "create project",
    "setup",
    "multi-step",
    "build",
    "create the",
    "extract",
)


def heuristic_requires_planning(message: str, min_words: int = 12) -> bool:
    """
    Cheap pre-filter shared by all planners: only invoke real planning
    when the user's request is long enough to warrant it OR contains a
    planning trigger word.
    """
    lower = message.lower()
    if any(indicator in lower for indicator in PLANNING_INDICATORS):
        return True
    return len(message.split()) >= min_words


class BasePlanner(ABC):
    """Strategy interface for plan generation."""

    @abstractmethod
    def should_plan(self, message: str) -> bool:
        """Return True if this message warrants creating a Plan."""

    @abstractmethod
    async def create_plan(self, message: str, context: dict | None = None) -> Plan:
        """Produce a Plan for `message`."""
