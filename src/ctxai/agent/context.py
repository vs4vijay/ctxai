"""
Conversation context management for agent.
"""

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from .llm.base import Message, MessageRole, ToolCall

# Prefix shared by every deterministic summary message so there is exactly one
# summary slot in a context (a later summary replaces an earlier one).
SUMMARY_MESSAGE_PREFIX = "Conversation summary (preserve for future turns):\n"

# Exact shape of an elision marker; used to detect already-elided bodies so
# compaction is idempotent and never double-counts.
_ELISION_MARKER_PATTERN = re.compile(r"^\[elided \d+ chars of tool result for .+\]$")


def _elision_marker(chars: int, tool_name: str | None) -> str:
    """Build an honest marker describing elided tool-result content.

    Args:
        chars: Number of characters that were removed.
        tool_name: Name of the tool that produced the content, if known.

    Returns:
        A marker string such as ``[elided 400 chars of tool result for read_file]``.
    """
    return f"[elided {chars} chars of tool result for {tool_name or 'unknown_tool'}]"


def _is_elision_marker(content: str) -> bool:
    """Check whether content is exactly an elision marker.

    Args:
        content: Message content to inspect.

    Returns:
        True when the content is a whole-body elision marker.
    """
    return bool(_ELISION_MARKER_PATTERN.match(content))


def _cap_tool_result(content: str, max_output_chars: int, tool_name: str | None) -> str:
    """Cap a tool-result body at ``max_output_chars`` with an elision marker.

    The operation is idempotent: content that already ends with a whole-body
    elision marker (or was already capped) is returned unchanged.

    Args:
        content: Original tool-result body.
        max_output_chars: Maximum characters retained from the body.
        tool_name: Name of the tool that produced the content, if known.

    Returns:
        The capped body, or the original content when it already fits.
    """
    if max_output_chars <= 0 or len(content) <= max_output_chars:
        return content
    if _is_elision_marker(content) or _is_elision_marker(content.rsplit("\n", 1)[-1]):
        return content
    overflow = len(content) - max_output_chars
    return content[:max_output_chars] + "\n" + _elision_marker(overflow, tool_name)


@dataclass
class ConversationContext:
    """Manages conversation history and context.

    Token accounting is measured-first: whenever the provider reports usage
    (``prompt_tokens``), that measurement is the context-size basis until the
    next compaction invalidates it; otherwise the chars-div-4 heuristic is
    used. Compaction counters (``compaction_count``, ``elided_message_count``,
    ``last_compaction``) are the HH-03 attributes HH-04 will persist as
    ``compaction`` events.
    """

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    current_plan: Any | None = None  # Plan object from planning.py
    last_reported_prompt_tokens: int | None = None  # Measured basis from the latest usage report
    compaction_count: int = 0  # Effective compactions performed on this context
    elided_message_count: int = 0  # Tool-result bodies replaced by markers, cumulative
    last_compaction: dict[str, int] | None = None  # Stats of the most recent compaction

    def add_message(self, role: MessageRole, content: str, tool_calls: list[ToolCall] = None) -> None:
        """Add a message to conversation history."""
        message = Message(role=role, content=content, tool_calls=tool_calls)
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self.add_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str, tool_calls: list[ToolCall] = None) -> None:
        """Add assistant message."""
        self.add_message(MessageRole.ASSISTANT, content, tool_calls)

    def add_system_message(self, content: str) -> None:
        """Add system message."""
        self.add_message(MessageRole.SYSTEM, content)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        """Add tool result message."""
        message = Message(role=MessageRole.USER, content=result, tool_call_id=tool_call_id, name=tool_name)
        self.messages.append(message)

    def get_messages_for_llm(self) -> list[Message]:
        """Get messages formatted for LLM."""
        return self.messages.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize context without provider credentials or runtime objects."""
        return {
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    "tool_calls": [
                        {"id": call.id, "name": call.name, "parameters": call.parameters}
                        for call in (message.tool_calls or [])
                    ],
                    "tool_call_id": message.tool_call_id,
                    "name": message.name,
                }
                for message in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        """Restore a context while validating its portable message structure."""
        context = cls(metadata=dict(data.get("metadata", {})))
        created_at = data.get("created_at")
        if created_at:
            context.created_at = datetime.fromisoformat(created_at)
        for item in data.get("messages", []):
            context.messages.append(
                Message(
                    role=MessageRole(item["role"]),
                    content=str(item.get("content", "")),
                    tool_calls=[ToolCall(**call) for call in item.get("tool_calls", [])] or None,
                    tool_call_id=item.get("tool_call_id"),
                    name=item.get("name"),
                )
            )
        return context

    def set_current_plan(self, plan: Any) -> None:
        """Set the current execution plan."""
        self.current_plan = plan

    def clear_plan(self) -> None:
        """Clear the current plan."""
        self.current_plan = None

    def note_reported_usage(self, usage: dict[str, Any]) -> None:
        """Record provider-reported usage as the measured token basis.

        Args:
            usage: ``LLMResponse.usage`` payload from the most recent call.
                Only a positive ``prompt_tokens`` value becomes the basis.
        """
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0:
            self.last_reported_prompt_tokens = prompt_tokens

    def get_token_count_estimate(self) -> int:
        """
        Estimate total token count.

        Heuristic estimation: ~4 characters per token. Used as the fallback
        whenever the provider has not reported usage.

        Returns:
            Estimated token count.
        """
        total_chars = sum(len(msg.content) for msg in self.messages)
        return total_chars // 4

    def estimate_context_tokens(self) -> int:
        """Estimate the current context size, preferring measured usage.

        The basis is the ``prompt_tokens`` value reported by the provider on
        the most recent call (an honest measurement of the exact payload the
        provider accepted, including tool schemas and formatting overhead).
        When no usage has been reported — or after a compaction invalidated
        the measurement — the chars-div-4 heuristic is used.

        Returns:
            Estimated context size in tokens.
        """
        if self.last_reported_prompt_tokens is not None:
            return self.last_reported_prompt_tokens
        return self.get_token_count_estimate()

    @staticmethod
    def _is_system_group(group: list[Message]) -> bool:
        """Check whether a group is a standalone system message.

        Args:
            group: A group of consecutive messages.

        Returns:
            True when the group is exactly one system message.
        """
        return len(group) == 1 and group[0].role == MessageRole.SYSTEM

    def _group_messages(self) -> list[list[Message]]:
        """Split history into atomic groups, preserving order.

        A group is either a single system message, an assistant message with
        tool calls followed by its paired tool-result messages, or any other
        single message. Grouping is what keeps
        ``assistant(tool_calls) ↔ tool results`` pairs atomic under compaction
        and truncation.

        Returns:
            Groups of consecutive messages covering the full history.
        """
        groups: list[list[Message]] = []
        index = 0
        while index < len(self.messages):
            message = self.messages[index]
            if message.role == MessageRole.SYSTEM or not message.tool_calls:
                groups.append([message])
                index += 1
                continue
            group = [message]
            index += 1
            while index < len(self.messages) and self.messages[index].tool_call_id:
                group.append(self.messages[index])
                index += 1
            groups.append(group)
        return groups

    def compact(self, target_tokens: int, keep_recent: int = 6, max_output_chars: int = 20_000) -> bool:
        """Deterministically shrink history when it approaches the context budget.

        Compaction never removes messages and never breaks
        ``assistant(tool_calls) ↔ tool results`` pairing:

        1. Every tool-result body is capped at ``max_output_chars`` with an
           elision marker.
        2. Tool-result bodies outside the recent window are replaced by honest
           elision markers. The elision decision is per group — an assistant
           message with tool calls and its results move between states
           together, so a group is never split across states. Bodies already
           marked, or smaller than their own marker, are left untouched.
        3. The turns elided by this compaction are summarized (via
           ``_summarize_messages``) into a single user-role message placed
           after the leading system messages; a later compaction replaces that
           summary in place. The user role keeps the summary out of the
           system-prompt slot, so the system prompt survives unchanged.
        4. Effective compactions update ``compaction_count``,
           ``elided_message_count``, and ``last_compaction`` (the attribute
           HH-04 will persist as a ``compaction`` event).

        Identical history plus identical arguments produce identical messages:
        no wall-clock, randomness, or external state participates.

        Args:
            target_tokens: Soft context budget that triggered compaction.
                Recorded for reporting; elision is window-based, not
                target-fitted.
            keep_recent: Number of most recent non-system groups kept verbatim
                (subject only to the ``max_output_chars`` cap).
            max_output_chars: Hard cap applied to every tool-result body.

        Returns:
            True when any message content changed (a real compaction happened).
        """
        tokens_before = self.get_token_count_estimate()
        keep = max(0, keep_recent)
        groups = self._group_messages()
        non_system_positions = [position for position, group in enumerate(groups) if not self._is_system_group(group)]
        recent_positions = set(non_system_positions[len(non_system_positions) - keep :]) if keep else set()

        changed = False
        newly_elided = 0
        elided_turn_messages: list[Message] = []
        rebuilt: list[Message] = []

        for position, group in enumerate(groups):
            if self._is_system_group(group) or position in recent_positions:
                for message in group:
                    if message.tool_call_id:
                        capped = _cap_tool_result(message.content, max_output_chars, message.name)
                        if capped != message.content:
                            changed = True
                            rebuilt.append(replace(message, content=capped))
                            continue
                    rebuilt.append(message)
                continue

            elidable = [
                message
                for message in group
                if message.tool_call_id
                and not _is_elision_marker(message.content)
                and len(message.content) > len(_elision_marker(len(message.content), message.name))
            ]
            if not elidable:
                rebuilt.extend(group)
                continue
            changed = True
            elided_turn_messages.extend(group)
            for message in group:
                if message.tool_call_id and not _is_elision_marker(message.content):
                    newly_elided += 1
                    rebuilt.append(replace(message, content=_elision_marker(len(message.content), message.name)))
                else:
                    rebuilt.append(message)

        if elided_turn_messages:
            summary_text = self._summarize_messages(elided_turn_messages)
            if summary_text:
                summary_content = SUMMARY_MESSAGE_PREFIX + summary_text
                existing_index = next(
                    (
                        index
                        for index, message in enumerate(rebuilt)
                        if message.content.startswith(SUMMARY_MESSAGE_PREFIX)
                    ),
                    None,
                )
                if existing_index is not None:
                    rebuilt[existing_index] = replace(rebuilt[existing_index], content=summary_content)
                else:
                    lead = 0
                    for message in rebuilt:
                        if message.role == MessageRole.SYSTEM:
                            lead += 1
                        else:
                            break
                    rebuilt.insert(lead, Message(role=MessageRole.USER, content=summary_content))

        if not changed:
            return False

        self.messages = rebuilt
        self.compaction_count += 1
        self.elided_message_count += newly_elided
        self.last_compaction = {
            "tokens_before": tokens_before,
            "tokens_after": self.get_token_count_estimate(),
            "elided_messages": newly_elided,
            "target_tokens": int(target_tokens),
        }
        self.last_reported_prompt_tokens = None
        return True

    def truncate_old_messages(self, max_tokens: int = 100000) -> None:
        """
        Truncate old messages if context is too large.
        Keeps system messages, the newest whole message groups that fit the
        token budget, and a deterministic summary of everything omitted.
        Groups (assistant tool calls together with their results) are never
        split, so tool-call pairing survives truncation.
        """
        if self.get_token_count_estimate() <= max_tokens:
            return

        groups = self._group_messages()
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        activity_groups = [group for group in groups if not self._is_system_group(group)]

        # Keep recent groups whole while they fit the budget.
        retained_ids: set[int] = set()
        token_count = 0
        for group in reversed(activity_groups):
            group_tokens = sum(len(msg.content) for msg in group) // 4
            if token_count + group_tokens > max_tokens:
                break
            for msg in group:
                retained_ids.add(id(msg))
            token_count += group_tokens

        omitted = [msg for group in activity_groups for msg in group if id(msg) not in retained_ids]
        system_tokens = sum(len(msg.content) for msg in system_messages) // 4
        # The summary itself must fit the remaining budget (chars//4 heuristic).
        summary_budget_chars = max(0, max_tokens - token_count - system_tokens) * 4
        summary = self._summarize_messages(omitted, max_chars=min(6000, summary_budget_chars))
        summary_message = []
        if summary:
            summary_message = [Message(role=MessageRole.SYSTEM, content=SUMMARY_MESSAGE_PREFIX + summary)]
        retained_messages = [msg for group in activity_groups for msg in group if id(msg) in retained_ids]
        self.messages = system_messages + summary_message + retained_messages

    @staticmethod
    def _summarize_messages(messages: list[Message], max_chars: int = 6000) -> str:
        """Create a deterministic, relevance-oriented summary without another LLM call.

        When the assembled summary exceeds ``max_chars``, the oldest
        low-relevance entries are dropped first, and remaining entries are
        shortened uniformly, so decisions and failures survive tight budgets.
        """
        if max_chars <= 0:
            return ""
        labels = {
            MessageRole.USER: "Request/decision",
            MessageRole.ASSISTANT: "Assistant outcome",
        }
        entries: list[tuple[bool, str]] = []
        for message in messages:
            label = "Tool result" if message.tool_call_id else labels.get(message.role, "Context")
            content = " ".join(message.content.split())
            if not content:
                continue
            important = any(
                term in content.lower()
                for term in ("decid", "chang", "fail", "error", "todo", "open", "risk", "test", "verify")
            )
            limit = 700 if important else 300
            entries.append((important, f"- {label}: {content[:limit]}"))

        def _joined(items: list[tuple[bool, str]]) -> str:
            return "\n".join(line for _, line in items)

        if len(_joined(entries)) <= max_chars:
            return _joined(entries)

        kept = list(entries)
        while len(_joined(kept)) > max_chars:
            drop_index = next((i for i, (important, _) in enumerate(kept) if not important), None)
            if drop_index is None:
                break
            kept.pop(drop_index)
        if not kept:
            # A summary must still exist for omitted history; keep the newest
            # entry and let the sizing below fit it into the budget.
            kept = [entries[-1]]

        summary = _joined(kept)
        if len(summary) > max_chars:
            per_line = max(40, max_chars // max(1, len(kept)) - 1)
            summary = "\n".join(line[:per_line] for _, line in kept)
        return summary[-max_chars:]

    def prune_to_fit(self, max_tokens: int = 100000) -> int:
        """
        Aggressively prune until token budget is satisfied. Returns the
        number of non-system messages dropped.
        """
        before = len(self.messages)
        self.truncate_old_messages(max_tokens=max_tokens)
        return before - len(self.messages)

    def clear(self) -> None:
        """Clear all messages except system messages."""
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        self.messages = system_messages
        self.current_plan = None

    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationContext(messages={len(self.messages)}, tokens≈{self.get_token_count_estimate()})"
