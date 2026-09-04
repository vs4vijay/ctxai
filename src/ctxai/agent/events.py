"""Agent event protocol (HH-05): the loop's in-memory streaming vocabulary.

Two cooperating shapes make up the protocol:

- :class:`AgentEvent` is what the agent loop emits to its UI surfaces. An
  ``AgentEvent`` is a small immutable value object with a ``kind`` from the
  closed :class:`AgentEventKind` set, a human-readable ``text`` line, and a
  structured ``data`` mapping. ``Agent.stream_message`` yields these events;
  ``Agent.process_message`` consumes the same core and discards them.
- :class:`StreamEvent` is what LLM providers emit inside
  ``BaseLLMProvider.stream_chat_events``: a ``(kind, payload)`` tuple where
  ``kind`` is ``"text"`` (payload: ``str`` delta), ``"tool_call_delta"``
  (payload: ``dict`` partial tool-call fragment), or ``"usage"`` (payload:
  ``dict`` token counts). The loop maps ``"text"`` deltas onto ``token``
  events; complete tool calls and usage always arrive on the returned
  :class:`~ctxai.agent.llm.base.LLMResponse`, which is authoritative.

Token deltas are transient UI state: they are never persisted to run
transcripts beyond what HH-04 already records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

StreamEvent = tuple[str, Any]
"""A provider-level streaming event: ``("text", str) | ("tool_call_delta", dict) | ("usage", dict)``."""


class AgentEventKind(str, Enum):
    """The closed set of agent loop event kinds (Part II contract, HH-05)."""

    TOKEN = "token"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DECIDED = "approval_decided"
    STATUS = "status"
    USAGE = "usage"
    FINAL_REPORT = "final_report"


@dataclass(frozen=True)
class AgentEvent:
    """One streaming event emitted by the agent loop.

    Attributes:
        kind: The event kind from the closed ``AgentEventKind`` vocabulary.
        text: Human-readable payload (a token delta, a tool name, a status
            line, or the final report).
        data: Structured payload details (tool parameters, usage counts,
            approval decisions, compaction stats).
    """

    kind: AgentEventKind
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
