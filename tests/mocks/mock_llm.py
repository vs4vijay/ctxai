"""
Mock LLM provider for agent testing.

This mock provider returns predefined responses instead of making real API calls,
enabling fast, deterministic testing without costs.
"""

import re
from collections.abc import Generator
from dataclasses import replace
from typing import Any

from ctxai.agent.config import AgentLLMConfig
from ctxai.agent.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ProviderCapabilities,
    StreamEvent,
    ToolCall,
)


def _token_chunks(content: str) -> list[str]:
    """Split content into whitespace-preserving token deltas.

    Args:
        content: The full response content.

    Returns:
        Ordered chunks whose concatenation is exactly ``content``.
    """
    return [match.group(0) for match in re.finditer(r"\S+\s*|\s+", content)]


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM that returns predefined responses.

    Useful for testing agent workflows without actual API calls.
    Supports configuring exact sequences of responses including tool calls,
    per-response usage payloads, an injected context_size for
    context-budget (HH-03) tests, and a streaming mode (HH-05) that emits
    scripted token deltas via ``stream_chat_events``.
    """

    def __init__(
        self,
        config: AgentLLMConfig = None,
        responses: list[dict[str, Any]] = None,
        context_size: int | None = None,
        supports_streaming: bool = False,
    ):
        """
        Initialize mock LLM provider.

        Args:
            config: Agent LLM configuration (will be created if None)
            responses: List of response configurations. Each response can include:
                - content: Response text
                - tool_calls: List of tool call dicts with 'name', 'parameters', optionally 'id'
                - finish_reason: One of "stop", "tool_calls", "length"
                - usage: Usage dict (prompt_tokens/completion_tokens/total_tokens)
            context_size: Optional context size reported by get_capabilities()
                (defaults to the ProviderCapabilities default when None)
            supports_streaming: When True the provider reports
                ``capabilities.streaming = True`` and ``stream_chat_events``
                streams the scripted content as token deltas; when False (the
                default) it degrades to the buffered base-class fallback.
        """
        # Create dummy config if none provided
        if config is None:
            config = AgentLLMConfig(
                provider="mock", model="mock-model", api_key="mock-key", temperature=0.7, max_tokens=4096
            )

        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []  # Track all calls for assertions
        self.context_size = context_size
        self.supports_streaming = supports_streaming

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "mock-model-v1"

    def get_capabilities(self) -> ProviderCapabilities:
        """
        Report capabilities, honoring an injected context_size and streaming mode.

        Returns:
            ProviderCapabilities with the injected context_size when set, and
            ``streaming`` reflecting ``supports_streaming``.
        """
        capabilities = super().get_capabilities()
        if self.context_size is not None:
            capabilities = replace(capabilities, context_size=self.context_size)
        if not self.supports_streaming:
            capabilities = replace(capabilities, streaming=False)
        return capabilities

    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs) -> LLMResponse:
        """
        Return predefined response based on call count.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional arguments (ignored)

        Returns:
            Predefined LLMResponse for current call count
        """
        # Track call for assertions
        self.call_history.append(
            {
                "messages": [msg.to_dict() for msg in messages],
                "tools": tools,
                "kwargs": kwargs,
            }
        )

        # Return next response or default if we've exhausted responses
        if self.call_count >= len(self.responses):
            # Default final response
            response = LLMResponse(
                content="Task completed successfully.",
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            )
            self.call_count += 1
            return response

        # Get configured response
        response_config = self.responses[self.call_count]
        self.call_count += 1

        # Build tool calls if specified
        tool_calls = []
        if "tool_calls" in response_config:
            for i, tc in enumerate(response_config["tool_calls"]):
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{self.call_count}_{i}"),
                        name=tc["name"],
                        parameters=tc.get("parameters", {}),
                    )
                )

        # Determine finish reason
        finish_reason = response_config.get("finish_reason")
        if finish_reason is None:
            finish_reason = "tool_calls" if tool_calls else "stop"

        # Create response
        return LLMResponse(
            content=response_config.get("content", ""),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=response_config.get("usage", {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}),
        )

    def stream_chat(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Mock streaming - yields content character by character.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional arguments (ignored)

        Yields:
            Response text chunks (individual characters)
        """
        # Get the response using regular chat
        response = self.chat(messages, tools, **kwargs)

        # Yield content character by character to simulate streaming
        yield from response.content

    def stream_chat_events(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        """
        Stream the scripted response as token deltas and return the response.

        With ``supports_streaming`` enabled, the configured content is emitted
        as whitespace-preserving word deltas (scripted token streaming) and the
        complete LLMResponse is returned. With streaming disabled, this
        delegates to the base-class buffered fallback (one ``("text", ...)``
        event). Both paths route through :meth:`chat`, so call counting,
        call history, and scripted subclass overrides behave identically on
        the streaming and buffered paths.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional arguments (ignored)

        Yields:
            StreamEvent tuples (``("text", str)`` deltas)

        Returns:
            The complete LLMResponse for the call
        """
        if not self.supports_streaming:
            response = yield from super().stream_chat_events(messages, tools=tools, **kwargs)
            return response

        response = self.chat(messages, tools=tools, **kwargs)
        for chunk in _token_chunks(response.content):
            yield ("text", chunk)
        return response

    def supports_function_calling(self) -> bool:
        """
        Check if provider supports function/tool calling.

        Returns:
            True (mock provider supports tool calling)
        """
        return True

    def requires_api_key(self) -> bool:
        """
        Check if provider requires an API key.

        Returns:
            False (mock provider doesn't need real API key)
        """
        return False

    def reset(self):
        """
        Reset the mock provider to initial state.

        Useful for running multiple tests with the same provider instance.
        """
        self.call_count = 0
        self.call_history = []


def create_mock_response(
    content: str = "",
    tool_calls: list[dict[str, Any]] = None,
    usage: dict[str, int] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """
    Helper function to create a mock response configuration.

    Args:
        content: Response text content
        tool_calls: Optional list of tool calls to include
        usage: Optional provider-reported usage dict (tokens only), e.g.
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        finish_reason: Optional finish reason ("stop", "tool_calls", "length")

    Returns:
        Response configuration dict

    Example:
        >>> responses = [
        ...     create_mock_response(
        ...         content="I'll read the file",
        ...         tool_calls=[{"name": "read_file", "parameters": {"path": "test.py"}}]
        ...     ),
        ...     create_mock_response(
        ...         content="The file contains: ...",
        ...         usage={"prompt_tokens": 500, "completion_tokens": 8, "total_tokens": 508},
        ...     ),
        ... ]
        >>> provider = MockLLMProvider(responses=responses)
    """
    response: dict[str, Any] = {"content": content}
    if tool_calls:
        response["tool_calls"] = tool_calls
    if usage is not None:
        response["usage"] = usage
    if finish_reason is not None:
        response["finish_reason"] = finish_reason
    return response
