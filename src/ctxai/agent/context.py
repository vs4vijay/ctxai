"""
Conversation context management for agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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

        # Keep system messages
        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]

        # Keep recent messages
        recent_messages = []
        token_count = 0

        for msg in reversed(self.messages):
            if msg.role == MessageRole.SYSTEM:
                continue

            msg_tokens = len(msg.content) // 4
            if token_count + msg_tokens > max_tokens:
                break

            recent_messages.insert(0, msg)
            token_count += msg_tokens

        # Combine
        self.messages = system_messages + recent_messages

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
