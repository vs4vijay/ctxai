"""
Custom OpenAI-compatible LLM provider implementation.

Supports custom API endpoints like Modal (api.us-west-2.modal.direct).
"""

import os
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from ..config import AgentLLMConfig
from .base import BaseLLMProvider, LLMResponse, ToolCall


class CustomProvider(BaseLLMProvider):
    """
    Custom OpenAI-compatible LLM provider.

    Supports:
    - Custom base URLs (e.g., Modal endpoints)
    - OpenAI-compatible APIs (like api.us-west-2.modal.direct)
    - Function/tool calling
    - Streaming responses

    Usage:
        config = AgentLLMConfig(
            provider="custom",
            model="zai-org/GLM-5-FP8",
            api_key="your-api-key",
            base_url="https://api.us-west-2.modal.direct/v1"
        )
        provider = CustomProvider(config)
    """

    def __init__(self, config: AgentLLMConfig):
        """
        Initialize Custom provider.

        Args:
            config: LLM configuration with base_url for custom endpoint

        Raises:
            ValueError: If base_url is not provided or API key is missing
        """
        super().__init__(config)

        # Get base_url from config (required for custom provider)
        self.base_url = config.base_url
        if not self.base_url:
            raise ValueError(
                "Custom provider requires base_url. "
                "Set it in config: AgentLLMConfig(base_url='https://your-endpoint.com/v1')"
            )

        # Get API key from config or environment
        api_key = config.api_key
        if not api_key:
            # Try common environment variable patterns
            api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("MODAL_API_KEY")

        if not api_key:
            raise ValueError(
                "Custom provider requires an API key. "
                "Set it via config.api_key or CUSTOM_API_KEY/MODAL_API_KEY environment variable."
            )

        # Initialize OpenAI client with custom base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url.rstrip("/") + "/",  # Ensure trailing slash
        )

        # Set model with default
        self.model = config.model or "gpt-4o"

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "gpt-4o"

    def requires_api_key(self) -> bool:
        """Check if provider requires an API key."""
        return True

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat request to custom OpenAI-compatible endpoint.

        Args:
            messages: List of messages (dict format for OpenAI compatibility)
            tools: Optional list of tool schemas in OpenAI format
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and tool calls
        """
        # Convert Message objects to dicts if needed
        from .base import Message

        if messages and isinstance(messages[0], Message):
            messages = self._format_messages(messages)

        # Prepare request
        request_params: dict[str, Any] = {
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

        # Extract content (handle reasoning models where content may be null)
        content = message.content or ""

        # Some models (like GLM-5) put reasoning in reasoning_content field
        if not content and hasattr(message, "reasoning_content") and message.reasoning_content:
            content = message.reasoning_content

        # Extract tool calls
        tool_calls: list[ToolCall] = []
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

        # Extract usage info
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def stream_chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream chat response from custom endpoint.

        Args:
            messages: List of messages
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        # Prepare request
        request_params: dict[str, Any] = {
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

    def supports_function_calling(self) -> bool:
        """
        Check if provider supports function/tool calling.

        Most OpenAI-compatible endpoints support function calling.

        Returns:
            True if function calling is supported
        """
        return True

    def validate_config(self) -> bool:
        """
        Validate that the provider is properly configured.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check base_url
        if not self.base_url:
            raise ValueError("Custom provider: base_url is required")

        # Check API key
        if not self.api_key:
            raise ValueError("Custom provider: API key is required")

        # Check model
        if not self.model:
            raise ValueError("Custom provider: model is required")

        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"CustomProvider(model={self.model}, base_url={self.base_url})"
