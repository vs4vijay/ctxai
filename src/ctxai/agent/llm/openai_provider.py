"""
OpenAI LLM provider implementation.

Supports GPT-4, GPT-4o, and other OpenAI models with tool calling.
"""

import os
from collections.abc import Iterator

from openai import OpenAI

from ..config import AgentLLMConfig
from .base import BaseLLMProvider, LLMResponse, ToolCall


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider.

    Supports:
    - GPT-4 (gpt-4, gpt-4-turbo)
    - GPT-4o (gpt-4o, gpt-4o-mini)
    - Function/tool calling
    - Streaming responses
    """

    def __init__(self, config: AgentLLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide it in the config."
            )

        # Initialize client
        self.client = OpenAI(api_key=api_key)

        # Set model with default
        self.model = config.model or "gpt-4o"

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat request to OpenAI.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool schemas in OpenAI format
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and tool calls
        """
        self.validate_request(messages, tools)
        messages = self.normalize_messages(messages)
        # Prepare request
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        # Call API
        response = self.client.chat.completions.create(**request_params)

        # Parse response
        message = response.choices[0].message

        # Extract content
        content = message.content or ""

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse function arguments
                import json

                parameters = json.loads(tc.function.arguments)

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        parameters=parameters,
                    )
                )

        # Map finish reason onto the shared vocabulary ("stop", "tool_calls", "length")
        raw_finish_reason = response.choices[0].finish_reason
        finish_reason = "stop"
        if raw_finish_reason == "tool_calls":
            finish_reason = "tool_calls"
        elif raw_finish_reason == "length":
            finish_reason = "length"

        # Extract provider-reported usage (tokens only)
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=response.model_dump(),
        )

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream chat response from OpenAI.

        Args:
            messages: List of messages
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        self.validate_request(messages, tools, stream=True)
        messages = self.normalize_messages(messages)
        # Prepare request
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        # Stream response
        stream = self.client.chat.completions.create(**request_params)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters for English
        # More accurate with tiktoken library if available
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to rough estimate
            return len(text) // 4

    def get_default_model(self) -> str:
        return "gpt-4o"

    def supports_function_calling(self) -> bool:
        return True

    def requires_api_key(self) -> bool:
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIProvider(model={self.model})"
