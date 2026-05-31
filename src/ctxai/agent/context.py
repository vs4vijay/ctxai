"""
Conversation context management for agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
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
        Keeps system messages and recent messages, and tries to keep
        tool_call/tool_result pairs together to avoid orphan tool results
        confusing the LLM.
        """
        if self.get_token_count_estimate() <= max_tokens:
            return

        system_messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        non_system = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]

        recent_messages: list[Message] = []
        token_count = 0

        for msg in reversed(non_system):
            msg_tokens = max(1, len(msg.content) // 4)
            if token_count + msg_tokens > max_tokens:
                break
            recent_messages.insert(0, msg)
            token_count += msg_tokens

        # If the first kept message is a tool result with no preceding assistant
        # tool_call, drop it to avoid an orphan that some providers reject.
        while recent_messages and recent_messages[0].tool_call_id:
            recent_messages.pop(0)

        self.messages = system_messages + recent_messages

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
