"""
OpenRouter LLM provider implementation.

Provides access to 100+ models through a single API:
- Claude (Anthropic)
- GPT-4, GPT-4o (OpenAI)
- Gemini (Google)
- DeepSeek (for reasoning)
- Llama, Mixtral, and more
"""

import os
from typing import Iterator, List, Optional

import requests

from .base import BaseLLMProvider, LLMResponse, MessageRole, ToolCall
from ..config import AgentLLMConfig


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter LLM provider.

    Provides unified access to multiple model providers through OpenRouter API.

    Popular models:
    - anthropic/claude-3.5-sonnet (best for coding)
    - anthropic/claude-3-opus (best quality, slower)
    - openai/gpt-4o (fast, good quality)
    - openai/o1 (reasoning model)
    - deepseek/deepseek-chat (cheap, fast)
    - google/gemini-pro (good balance)
    - meta-llama/llama-3.1-70b-instruct (local-style)
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: AgentLLMConfig):
        """
        Initialize OpenRouter provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Get API key from config, environment, or keystore
        self.api_key = config.get_api_key_for_provider("openrouter")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Run 'ctxai login openrouter' or set OPENROUTER_API_KEY environment variable."
            )

        # Set model with default
        # Using deepseek-r1:free as default - completely free and good quality
        self.model = config.model or "deepseek/deepseek-r1:free"

        # Optional: Site URL and App Name for OpenRouter tracking
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com/ctxai")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "ctxai")

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat request to OpenRouter.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool schemas in OpenAI format
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and tool calls
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json",
        }

        # Prepare request body
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Add tools if provided
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Call API
        response = requests.post(
            self.OPENROUTER_API_URL,
            headers=headers,
            json=body,
            timeout=120,  # 2 minute timeout
        )

        # Handle errors
        if response.status_code != 200:
            error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
            raise Exception(error_msg)

        # Parse response
        data = response.json()
        message = data["choices"][0]["message"]

        # Extract content
        content = message.get("content", "")

        # Extract tool calls
        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                # Parse function arguments
                import json
                parameters = json.loads(tc["function"]["arguments"])

                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        parameters=parameters,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=data,
        )

    def stream_chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream chat response from OpenRouter.

        Args:
            messages: List of messages
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
            "Content-Type": "application/json",
        }

        # Prepare request body
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Stream response
        response = requests.post(
            self.OPENROUTER_API_URL,
            headers=headers,
            json=body,
            stream=True,
            timeout=120,
        )

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        import json
                        data = json.loads(data_str)
                        if data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        # More accurate with tiktoken for OpenAI models
        try:
            import tiktoken

            # Try to get encoding for the model
            if "gpt" in self.model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "claude" in self.model:
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # Fallback to rough estimate
                return len(text) // 4

            return len(encoding.encode(text))
        except (ImportError, Exception):
            # Fallback to rough estimate
            return len(text) // 4

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "deepseek/deepseek-r1:free"

    def requires_api_key(self) -> bool:
        """Check if provider requires an API key."""
        return True

    def supports_function_calling(self) -> bool:
        """Check if provider supports function/tool calling."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenRouterProvider(model={self.model})"


# Popular model configurations for easy reference
OPENROUTER_MODELS = {
    # Best for coding (architect)
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-opus": "anthropic/claude-3-opus",

    # Fast and good (editor)
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",

    # Reasoning models (architect for complex tasks)
    "o1": "openai/o1",
    "o1-mini": "openai/o1-mini",
    "deepseek-r1": "deepseek/deepseek-r1",

    # Budget options (editor)
    "deepseek-chat": "deepseek/deepseek-chat",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",

    # Google models
    "gemini-pro": "google/gemini-pro",
    "gemini-flash": "google/gemini-flash-1.5",
}
