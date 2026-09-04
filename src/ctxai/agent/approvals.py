"""Approval decisions and session-scoped approval memory (HH-07).

Every approval is resolved in one place by the agent loop: a session-scope
:class:`ApprovalMemory` hit auto-approves the exact ``(tool, target)`` pair
without prompting, and everything else is asked through a decision callback.
Legacy boolean callbacks keep working through :func:`as_decision_callback`
(``True`` -> :attr:`ApprovalDecision.APPROVE_ONCE`, ``False`` -> DENY).

Memory entries live in ``ConversationContext.metadata`` under
:data:`APPROVAL_MEMORY_KEY`, so they ride along with persisted sessions and are
redacted like all session data. They expire with the session (bounded by a
time-to-live so a resumed session cannot auto-approve forever) and never reach
global configuration. Scope never widens capabilities: the pattern is the exact
canonical path for mutations and the executable name for commands, and command
policy plus capability checks keep gating every execution.
"""

from __future__ import annotations

import logging
import os
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .llm.base import ToolCall

LOGGER = logging.getLogger(__name__)

APPROVAL_MEMORY_KEY = "approval_memory"
"""``ConversationContext.metadata`` key holding the serialized :class:`ApprovalMemory`."""

DEFAULT_MAX_AGE_SECONDS = 8 * 3600.0
"""Session-scope approvals expire after this bound, even inside a resumed session."""

MUTATION_TOOL_NAMES: tuple[str, ...] = ("write_file", "edit_file")
"""Tools whose approvals key on the exact canonical target path."""

COMMAND_TOOL_NAMES: tuple[str, ...] = ("bash",)
"""Tools whose approvals key on the executable name of the command."""

_KEY_SEPARATOR = "\x1f"
"""Separator between tool and target in persisted memory keys (unit separator)."""


class ApprovalDecision(str, Enum):
    """The decision a human (or session memory) makes about one exact action.

    Part II contract: ``APPROVE_ONCE`` executes this exact action once,
    ``APPROVE_SESSION`` records a session-scope grant for the (tool, target)
    pair, and ``DENY`` keeps the existing ``APPROVAL_DENIAL`` failure path.
    """

    APPROVE_ONCE = "once"
    APPROVE_SESSION = "session"
    DENY = "deny"


ApprovalCallback = Callable[[ToolCall], bool]
"""Legacy boolean approval protocol (kept for backward compatibility)."""

DecisionCallback = Callable[[ToolCall], ApprovalDecision]
"""HH-07 decision protocol: callbacks return an :class:`ApprovalDecision`."""


def adapt_bool_callback(callback: ApprovalCallback) -> DecisionCallback:
    """Adapt a legacy boolean approval callback to the decision protocol.

    Args:
        callback: Legacy callback returning True to approve once, False to deny.

    Returns:
        A DecisionCallback mapping True to APPROVE_ONCE and False to DENY.
    """

    def adapted(call: ToolCall) -> ApprovalDecision:
        """Return the legacy callback's boolean answer as a decision.

        Args:
            call: The approval-shaped tool call presented to the human.

        Returns:
            APPROVE_ONCE when the callback returns True, DENY otherwise.
        """
        return ApprovalDecision.APPROVE_ONCE if callback(call) else ApprovalDecision.DENY

    return adapted


def as_decision_callback(callback: ApprovalCallback | DecisionCallback | None) -> DecisionCallback | None:
    """Normalize any supported callback shape into a DecisionCallback.

    Accepts callbacks returning :class:`ApprovalDecision`, booleans (legacy,
    adapted), or raw decision-value strings. Unrecognizable results fail closed
    to DENY — an unclear answer never approves an action.

    Args:
        callback: The configured approval callback, or None.

    Returns:
        The normalized DecisionCallback, or None when callback is None.
    """
    if callback is None:
        return None

    def decision(call: ToolCall) -> ApprovalDecision:
        """Normalize one callback answer, failing closed on garbage.

        Args:
            call: The approval-shaped tool call presented to the human.

        Returns:
            The normalized decision (DENY for unrecognized answers).
        """
        result = callback(call)
        if isinstance(result, ApprovalDecision):
            return result
        if isinstance(result, bool):
            return ApprovalDecision.APPROVE_ONCE if result else ApprovalDecision.DENY
        try:
            return ApprovalDecision(str(result))
        except ValueError:
            LOGGER.warning("approval callback returned unrecognized value %r; denying", result)
            return ApprovalDecision.DENY

    return decision


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp, returning None for empty or invalid values.

    Args:
        value: The raw stored timestamp.

    Returns:
        The parsed timezone-aware datetime, or None when unparseable.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class ApprovalMemory:
    """Session-scoped approval decisions keyed by ``(tool, target)``.

    Only session-scope grants are recorded. The target pattern is the canonical
    exact path for mutation tools and the executable name for commands (see
    :func:`approval_memory_target`), so a grant never widens beyond the tool it
    was given to and never becomes a global allow: command policy, capability
    checks, and the stale-approval binding keep applying to every execution.
    Entries carry a ``recorded_at`` timestamp and expire after
    ``max_age_seconds`` (``None`` disables time-based expiry; the natural
    session boundary still ends them).
    """

    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_age_seconds: float | None = DEFAULT_MAX_AGE_SECONDS

    @staticmethod
    def key(tool: str, target: str) -> str:
        """Build the persisted key for one (tool, target) pair.

        Args:
            tool: Tool name (e.g. ``write_file`` or ``bash``).
            target: The exact-path or executable pattern for the tool.

        Returns:
            The deterministic string key used in the decisions mapping.
        """
        return f"{tool}{_KEY_SEPARATOR}{target}"

    def check(self, tool: str, target: str, *, now: datetime | None = None) -> ApprovalDecision | None:
        """Return the session-scope decision for (tool, target), honoring expiry.

        Args:
            tool: Tool name.
            target: The exact-path or executable pattern.
            now: Current time (defaults to the wall clock; injectable for tests).

        Returns:
            The recorded ApprovalDecision, or None when absent, expired (the
            expired entry is dropped), or corrupt.
        """
        key = self.key(tool, target)
        entry = self.decisions.get(key)
        if entry is None:
            return None
        if self.max_age_seconds is not None:
            recorded_at = _parse_timestamp(entry.get("recorded_at"))
            current = now or datetime.now(timezone.utc)
            if recorded_at is None or (current - recorded_at).total_seconds() > self.max_age_seconds:
                del self.decisions[key]
                return None
        try:
            return ApprovalDecision(str(entry.get("decision")))
        except ValueError:
            return None

    def record(
        self,
        tool: str,
        target: str,
        decision: ApprovalDecision,
        *,
        recorded_at: datetime | None = None,
    ) -> None:
        """Record a session-scope decision for (tool, target).

        Args:
            tool: Tool name.
            target: The exact-path or executable pattern.
            decision: The decision to record (session grants are the intended use).
            recorded_at: Timestamp override (defaults to the wall clock; injectable for tests).
        """
        self.decisions[self.key(tool, target)] = {
            "decision": decision.value if isinstance(decision, ApprovalDecision) else str(decision),
            "recorded_at": (recorded_at or datetime.now(timezone.utc)).isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to the persisted dictionary shape (JSON-serializable).

        Returns:
            Dictionary representation stored under :data:`APPROVAL_MEMORY_KEY`.
        """
        return {
            "decisions": {key: dict(entry) for key, entry in self.decisions.items()},
        }

    @classmethod
    def from_dict(cls, data: Any) -> ApprovalMemory:
        """Create from a dictionary, starting fresh on missing or corrupt input.

        Args:
            data: Dictionary produced by ``to_dict`` (or anything else).

        Returns:
            The reconstructed :class:`ApprovalMemory`.
        """
        memory = cls()
        decisions = data.get("decisions") if isinstance(data, dict) else None
        if isinstance(decisions, dict):
            for key, entry in decisions.items():
                if isinstance(entry, dict):
                    memory.decisions[str(key)] = {
                        "decision": str(entry.get("decision", "")),
                        "recorded_at": str(entry.get("recorded_at", "")),
                    }
        return memory


def approval_memory_target(call: ToolCall, project_root: Path | None = None) -> str:
    """Compute the session-memory target pattern for one approval-shaped call.

    Mutation tools key on the canonical exact path (repository-relative when
    the file lives inside the project); commands key on the executable name so
    "always this session" covers the program, not arbitrary flags — every
    execution still passes the unchanged command-policy checks. Other tools key
    on the approval target recorded with the approval call.

    Args:
        call: The approval-shaped tool call.
        project_root: Project root used to canonicalize relative paths.

    Returns:
        The target pattern string used in :class:`ApprovalMemory` keys.
    """
    if call.name in MUTATION_TOOL_NAMES:
        raw = call.parameters.get("path") or call.parameters.get("file_path")
        if raw:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute() and project_root is not None:
                path = project_root / path
            resolved = Path(os.path.realpath(path))
            if project_root is not None:
                try:
                    return resolved.relative_to(Path(os.path.realpath(project_root))).as_posix()
                except ValueError:
                    return str(resolved)
            return str(resolved)
    if call.name in COMMAND_TOOL_NAMES:
        command = str(call.parameters.get("command") or "")
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError:
            argv = command.split()
        if argv:
            return Path(argv[0]).name.lower()
        return command.strip() or call.name
    target = call.parameters.get("approval_target")
    return str(target) if target else call.name
