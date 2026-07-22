"""
Conversation context management for agent.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .llm.base import Message, MessageRole, ToolCall


@dataclass
class ConversationContext:
    """Manages conversation history and context."""

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    current_plan: Any | None = None  # Plan object from planning.py

    def add_message(self, role: MessageRole, content: str, tool_calls: list[ToolCall] = None) -> None:
        """Add a message to conversation history."""
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls
        )
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
        message = Message(
            role=MessageRole.USER,
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        )
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
            context.messages.append(Message(
                role=MessageRole(item["role"]),
                content=str(item.get("content", "")),
                tool_calls=[ToolCall(**call) for call in item.get("tool_calls", [])] or None,
                tool_call_id=item.get("tool_call_id"),
                name=item.get("name"),
            ))
        return context

    def set_current_plan(self, plan: Any) -> None:
        """Set the current execution plan."""
        self.current_plan = plan

    def clear_plan(self) -> None:
        """Clear the current plan."""
        self.current_plan = None

    def get_token_count_estimate(self) -> int:
        """
        Estimate total token count.
        Simple estimation: ~4 characters per token.
        """
        total_chars = sum(len(msg.content) for msg in self.messages)
        return total_chars // 4

    def truncate_old_messages(self, max_tokens: int = 100000) -> None:
        """
        Truncate old messages if context is too large.
        Keeps system messages and recent messages.
        """
        if self.get_token_count_estimate() <= max_tokens:
            return

        # Keep system messages and compact older activity into a durable summary.
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]

        # Keep recent messages
        recent_messages = []
        token_count = 0

        retained_ids: set[int] = set()
        for msg in reversed(self.messages):
            if msg.role == MessageRole.SYSTEM:
                continue

            msg_tokens = len(msg.content) // 4
            if token_count + msg_tokens > max_tokens:
                break

            recent_messages.insert(0, msg)
            retained_ids.add(id(msg))
            token_count += msg_tokens

        omitted = [
            msg for msg in self.messages
            if msg.role != MessageRole.SYSTEM and id(msg) not in retained_ids
        ]
        summary = self._summarize_messages(omitted)
        summary_message = []
        if summary:
            summary_message = [Message(
                role=MessageRole.SYSTEM,
                content="Conversation summary (preserve for future turns):\n" + summary,
            )]
        self.messages = system_messages + summary_message + recent_messages

    @staticmethod
    def _summarize_messages(messages: list[Message], max_chars: int = 6000) -> str:
        """Create a deterministic, relevance-oriented summary without another LLM call."""
        labels = {
            MessageRole.USER: "Request/decision",
            MessageRole.ASSISTANT: "Assistant outcome",
        }
        lines: list[str] = []
        for message in messages:
            label = "Tool result" if message.tool_call_id else labels.get(message.role, "Context")
            content = " ".join(message.content.split())
            if not content:
                continue
            important = any(term in content.lower() for term in (
                "decid", "chang", "fail", "error", "todo", "open", "risk", "test", "verify"
            ))
            limit = 700 if important else 300
            lines.append(f"- {label}: {content[:limit]}")
        summary = "\n".join(lines)
        return summary[-max_chars:]

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
