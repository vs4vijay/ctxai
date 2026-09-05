"""
Base LLM provider interface for agent.

This module defines the abstract interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..events import StreamEvent


class ProviderErrorKind(str, Enum):
    """Stable error categories shared by every provider."""

    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNSUPPORTED = "unsupported"
    TRANSPORT = "transport"
    INVALID_RESPONSE = "invalid_response"
    CONTEXT_LENGTH = "context_length"


class ProviderError(RuntimeError):
    """Provider failure carrying a stable error kind and optional metadata.

    Supports both construction styles used across the codebase::

        ProviderError(ProviderErrorKind.TIMEOUT, "timed out", provider="anthropic")
        AuthenticationError("bad key", provider="anthropic", status_code=401)
    """

    default_kind: ProviderErrorKind | None = None

    def __init__(
        self,
        kind_or_message: ProviderErrorKind | str,
        message: str | None = None,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ):
        if isinstance(kind_or_message, ProviderErrorKind):
            text = message if message is not None else kind_or_message.value
            kind: ProviderErrorKind | None = kind_or_message
        else:
            text = kind_or_message
            kind = self.default_kind
        super().__init__(text)
        self.kind = kind
        self.provider = provider
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Provider rejected the request because we exceeded rate limits."""

    default_kind = ProviderErrorKind.RATE_LIMIT

    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ContextLengthError(ProviderError):
    """The context exceeds the provider's maximum context window."""

    default_kind = ProviderErrorKind.CONTEXT_LENGTH


class AuthenticationError(ProviderError):
    """API key invalid, expired, or missing."""

    default_kind = ProviderErrorKind.AUTHENTICATION


class ProviderTimeoutError(ProviderError):
    """The provider did not return a response within the timeout."""

    default_kind = ProviderErrorKind.TIMEOUT


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
                        "function": {"name": tc.name, "arguments": json.dumps(tc.parameters)},
                    }
                    for tc in self.tool_calls
                ]
            else:
                # Anthropic format
                msg["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "parameters": tc.parameters} for tc in self.tool_calls
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


@dataclass(frozen=True)
class ProviderCapabilities:
    """Features a provider/model can safely expose to the agent."""

    tools: bool = True
    streaming: bool = True
    images: bool = False
    structured_output: bool = False
    context_size: int = 100_000


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Anthropic, OpenAI, Ollama) must implement this interface.
    """

    def __init__(self, config: Any):
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
    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs) -> LLMResponse:
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
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs
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

    def stream_chat_events(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        """
        Stream a provider response as events, returning the complete response.

        The default implementation is graceful degradation (HH-05): it performs
        one buffered :meth:`chat` call, emits a single ``("text", full_content)``
        event, and returns the :class:`LLMResponse`. Providers with real
        streaming override this method to emit incremental ``("text", delta)``
        events (and may emit ``("tool_call_delta", fragment)`` /
        ``("usage", counts)`` events for richer consumers) while still
        returning the complete response — accumulated tool calls, finish
        reason, and usage on the returned object are authoritative, so the
        agent loop and its approval workflow behave identically on both paths.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional provider-specific arguments

        Yields:
            StreamEvent tuples (``("text", str)``, ``("tool_call_delta", dict)``,
            or ``("usage", dict)``)

        Returns:
            The complete LLMResponse (tool calls, finish_reason, usage)

        Raises:
            Exception: If the API call fails
        """
        response = self.chat(messages, tools=tools, **kwargs)
        if response.content:
            yield ("text", response.content)
        return response

    def get_capabilities(self) -> ProviderCapabilities:
        """Return normalized capabilities used by chat and agent orchestration.

        ``streaming`` means the provider implements :meth:`stream_chat_events`
        (token-delta event streaming with tool support). Providers that only
        implement the older text-only ``stream_chat`` — or nothing at all —
        report ``streaming=False`` and the agent loop uses the buffered
        ``chat()`` fallback for them.
        """
        return ProviderCapabilities(
            tools=self.supports_function_calling(),
            streaming=self.__class__.stream_chat_events is not BaseLLMProvider.stream_chat_events,
        )

    def validate_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        images: bool = False,
        structured_output: bool = False,
        cancel_event: Any = None,
    ) -> None:
        """Reject unsupported requests before a network or local model call."""
        capabilities = self.get_capabilities()
        requested = {
            "tools": bool(tools),
            "streaming": stream,
            "images": images,
            "structured_output": structured_output,
        }
        for feature, enabled in requested.items():
            if enabled and not getattr(capabilities, feature):
                raise ProviderError(
                    ProviderErrorKind.UNSUPPORTED,
                    f"{self.__class__.__name__} does not support {feature}",
                )
        if cancel_event is not None and cancel_event.is_set():
            raise ProviderError(ProviderErrorKind.CANCELLED, "Provider request cancelled")
        if not messages:
            raise ValueError("At least one message is required")

    def normalize_messages(self, messages: list[Message] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert the public Message contract to an OpenAI-compatible wire form."""
        if messages and isinstance(messages[0], Message):
            return self._format_messages(messages)  # type: ignore[arg-type]
        return messages  # type: ignore[return-value]

    def normalize_error(self, error: Exception) -> ProviderError:
        """Map transport-specific exceptions into stable failure categories."""
        if isinstance(error, ProviderError):
            return error
        name = error.__class__.__name__.lower()
        message = str(error)
        lowered = message.lower()
        if (
            "auth" in name
            or "api key" in lowered
            or "unauthorized" in lowered
            or "authentication" in lowered
            or "forbidden" in lowered
            or "401" in lowered
            or "403" in lowered
        ):
            kind = ProviderErrorKind.AUTHENTICATION
        elif "rate" in name or "429" in lowered:
            kind = ProviderErrorKind.RATE_LIMIT
        elif "timeout" in name or "timed out" in lowered:
            kind = ProviderErrorKind.TIMEOUT
        elif (
            "invalid" in name
            or "json" in name
            or "parse" in name
            or "decode" in name
            or "unexpected" in name
            or "malformed" in lowered
            or "invalid json" in lowered
            or "invalid request" in lowered
            or "parse error" in lowered
            or "expecting value" in lowered
        ):
            kind = ProviderErrorKind.INVALID_RESPONSE
        else:
            kind = ProviderErrorKind.TRANSPORT
        return ProviderError(kind, message, provider=self.__class__.__name__)

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

    def health_check(self) -> bool:
        """
        Optional health check — subclasses may override with a cheap ping.

        Default implementation just checks that config is present.
        """
        try:
            return self.validate_config()
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(model={self.model})"
