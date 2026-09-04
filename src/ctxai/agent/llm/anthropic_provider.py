"""
Anthropic Claude provider implementation.
"""

from collections.abc import Generator
from typing import Any

try:
    from anthropic import Anthropic, AnthropicError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..config import AgentLLMConfig
from ..events import StreamEvent
from .base import BaseLLMProvider, LLMResponse, Message, MessageRole, ToolCall


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: AgentLLMConfig):
        """Initialize Anthropic provider."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")

        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.get_api_key_for_provider("anthropic")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide api_key in config."
            )

        # Initialize client
        self.client = Anthropic(api_key=self.api_key, timeout=self.timeout)

    def get_default_model(self) -> str:
        """Get default Claude model."""
        return "claude-3-5-sonnet-20241022"

    def requires_api_key(self) -> bool:
        """Anthropic requires API key."""
        return True

    def supports_function_calling(self) -> bool:
        """Anthropic supports tool use."""
        return True

    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs) -> LLMResponse:
        """
        Send messages to Claude and get response.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions in Anthropic format
            **kwargs: Additional arguments

        Returns:
            LLMResponse
        """
        try:
            # Extract system message
            system_message, conversation_messages = self._extract_system_message(messages)

            # Format messages for Anthropic
            formatted_messages = self._format_messages_for_anthropic(conversation_messages)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": formatted_messages,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = tools

            # Make API call
            response = self.client.messages.create(**request_params)

            # Parse response
            return self._parse_response(response)

        except AnthropicError as e:
            return LLMResponse(
                content=f"Anthropic API error: {str(e)}",
                finish_reason="error",
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error",
            )

    def stream_chat(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response from Claude.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions
            **kwargs: Additional arguments

        Yields:
            Response text chunks
        """
        try:
            # Extract system message
            system_message, conversation_messages = self._extract_system_message(messages)

            # Format messages
            formatted_messages = self._format_messages_for_anthropic(conversation_messages)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": formatted_messages,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = tools

            # Stream response
            with self.client.messages.stream(**request_params) as stream:
                yield from stream.text_stream

        except AnthropicError as e:
            yield f"\n[Anthropic API error: {str(e)}]"
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    def stream_chat_events(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs
    ) -> Generator[StreamEvent, None, LLMResponse]:
        """
        Stream response events from Claude, returning the complete response.

        Text deltas are emitted as ``("text", chunk)`` StreamEvents as the
        model generates them (via the Messages streaming API). The final
        accumulated message — including any tool-use blocks, the finish
        reason, and usage — is parsed with the same parser as :meth:`chat`
        and returned, so the agent loop's approval workflow behaves
        identically on the streaming and buffered paths. Failures are mapped
        to an error-shaped ``LLMResponse`` (``finish_reason == "error"``),
        mirroring :meth:`chat`.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool definitions in Anthropic format
            **kwargs: Additional arguments

        Yields:
            StreamEvent tuples (``("text", str)`` deltas)

        Returns:
            The complete LLMResponse (tool calls, finish_reason, usage)
        """
        try:
            # Extract system message
            system_message, conversation_messages = self._extract_system_message(messages)

            # Format messages
            formatted_messages = self._format_messages_for_anthropic(conversation_messages)

            # Prepare request parameters
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": formatted_messages,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = tools

            # Stream events; the final message carries tool calls and usage.
            with self.client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    if text:
                        yield ("text", text)
                final_message = stream.get_final_message()

            return self._parse_response(final_message)
        except AnthropicError as e:
            return LLMResponse(
                content=f"Anthropic API error: {str(e)}",
                finish_reason="error",
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason="error",
            )

    def _format_messages_for_anthropic(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Format messages for Anthropic API.

        Args:
            messages: List of Message objects

        Returns:
            List of message dictionaries in Anthropic format
        """
        formatted = []

        for msg in messages:
            # Basic message
            if msg.role == MessageRole.SYSTEM:
                # System messages should be handled separately
                continue

            formatted_msg = {"role": msg.role.value, "content": []}

            # Add text content
            if msg.content:
                formatted_msg["content"].append({"type": "text", "text": msg.content})

            # Add tool use (if assistant with tool calls)
            if msg.tool_calls and msg.role == MessageRole.ASSISTANT:
                for tc in msg.tool_calls:
                    formatted_msg["content"].append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.parameters}
                    )

            # Tool result message (from user role)
            if msg.tool_call_id and msg.role == MessageRole.USER:
                formatted_msg["content"] = [
                    {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}
                ]

            formatted.append(formatted_msg)

        return formatted

    def _parse_response(self, response) -> LLMResponse:
        """
        Parse Anthropic API response.

        Args:
            response: Anthropic API response object

        Returns:
            LLMResponse
        """
        # Extract text content
        content_text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, parameters=block.input))

        # Determine finish reason
        finish_reason = "stop"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"

        # Extract usage
        usage = {}
        if hasattr(response, "usage"):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )
