"""
Base LLM provider interface for agent.

This module defines the abstract interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM."""
    id: str
    name: str
    parameters: dict[str, Any]


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool result messages
    name: str | None = None  # Tool name for tool result messages

    def to_dict(self, format: str = "openai") -> dict[str, Any]:
        """
        Convert message to dictionary format.

        Args:
            format: Format to use ("openai" or "anthropic")

        Returns:
            Message dictionary
        """
        msg = {
            "role": self.role.value,
            "content": self.content or "",
        }

        if self.tool_calls:
            if format == "openai":
                # OpenAI/OpenRouter format
                import json
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.parameters)
                        }
                    }
                    for tc in self.tool_calls
                ]
            else:
                # Anthropic format
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "parameters": tc.parameters
                    }
                    for tc in self.tool_calls
                ]

        if self.tool_call_id:
            # Tool result message
            msg["role"] = "tool"
            msg["tool_call_id"] = self.tool_call_id
            if self.name:
                msg["name"] = self.name

        return msg


@dataclass
class LLMResponse:
    """Represents a response from the LLM."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop", "tool_calls", "length", "error"
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Anthropic, OpenAI, Ollama) must implement this interface.
    """

    def __init__(self, config: 'AgentLLMConfig'):
        """
        Initialize the provider.

        Args:
            config: Agent LLM configuration
        """
        self.config = config
        self.model = config.model or self.get_default_model()
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send messages and get a response.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse containing the response

        Raises:
            Exception: If the API call fails
        """
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response tokens.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific arguments

        Yields:
            Response text chunks

        Raises:
            Exception: If the API call fails
        """
        pass

    @abstractmethod
    def supports_function_calling(self) -> bool:
        """
        Check if provider supports function/tool calling.

        Returns:
            True if function calling is supported
        """
        pass

    def _format_messages(self, messages: list[Message], format: str = "openai") -> list[dict[str, Any]]:
        """
        Format messages for the provider.

        Args:
            messages: List of Message objects
            format: Format to use ("openai" or "anthropic")

        Returns:
            List of message dictionaries
        """
        return [msg.to_dict(format=format) for msg in messages]

    def _extract_system_message(self, messages: list[Message]) -> tuple[str | None, list[Message]]:
        """
        Extract system message from messages list.

        Some providers (like Anthropic) handle system messages separately.

        Args:
            messages: List of messages

        Returns:
            Tuple of (system_message, remaining_messages)
        """
        system_message = None
        remaining = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                remaining.append(msg)

        return system_message, remaining

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Simple estimation: ~4 characters per token.
        Subclasses can override with provider-specific tokenization.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def validate_config(self) -> bool:
        """
        Validate that the provider is properly configured.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model:
            raise ValueError(f"{self.__class__.__name__}: No model specified")

        if not self.api_key and self.requires_api_key():
            raise ValueError(f"{self.__class__.__name__}: API key required but not provided")

        return True

    @abstractmethod
    def requires_api_key(self) -> bool:
        """
        Check if provider requires an API key.

        Returns:
            True if API key is required
        """
        pass

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(model={self.model})"
