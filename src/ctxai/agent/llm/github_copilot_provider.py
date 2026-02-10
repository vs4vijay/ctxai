"""
GitHub Copilot LLM provider implementation.

Provides access to GitHub Copilot's chat completion API using
authenticated tokens from OAuth device code flow.
"""

import json
import os
from collections.abc import Iterator
from typing import List, Optional

import requests

from ..config import AgentLLMConfig
from .base import BaseLLMProvider, LLMResponse, ToolCall


class GitHubCopilotProvider(BaseLLMProvider):
    """
    GitHub Copilot LLM provider.

    Uses GitHub Copilot's chat completions API with tokens obtained
    through OAuth device code flow.

    Supports multiple models:
    - gpt-4 (recommended for coding)
    - gpt-3.5-turbo (faster, cheaper)
    - claude-3.5-sonnet (via Copilot)
    - o1-preview (reasoning model)
    """

    # GitHub Copilot API endpoints
    COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"

    # Integration headers
    COPILOT_INTEGRATION_ID = "vscode-chat"
    USER_AGENT = "GitHubCopilotChat/0.35.0"
    EDITOR_VERSION = "vscode/1.99.3"
    EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"

    def __init__(self, config: AgentLLMConfig):
        """
        Initialize GitHub Copilot provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Get token from config or keystore
        self.token_data = self._get_token_data(config)

        if not self.token_data:
            raise ValueError(
                "GitHub Copilot token not found. Run 'ctxai login github-copilot' to authenticate."
            )

        # Extract the actual API token
        if isinstance(self.token_data, dict):
            self.api_token = self.token_data.get("token")
            if not self.api_token:
                raise ValueError("Invalid Copilot token data: missing 'token' field")
        else:
            # If it's a string, use it directly
            self.api_token = self.token_data

        # Set model with default
        self.model = config.model or "gpt-4"

    def _get_token_data(self, config: AgentLLMConfig) -> dict | str | None:
        """
        Get Copilot token from config or keystore.

        Args:
            config: LLM configuration

        Returns:
            Token data (dict or string) or None
        """
        # Check if token provided directly in config
        if config.api_key:
            return config.api_key

        # Check environment variable
        env_token = os.getenv("GITHUB_COPILOT_TOKEN")
        if env_token:
            return env_token

        # Check keystore
        try:
            from ...auth.keystore import get_keystore

            keystore = get_keystore()
            return keystore.get_key("github-copilot")
        except (ImportError, Exception):
            return None

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat request to GitHub Copilot.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool schemas in OpenAI format
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and tool calls
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Copilot-Integration-Id": self.COPILOT_INTEGRATION_ID,
            "User-Agent": self.USER_AGENT,
            "Editor-Version": self.EDITOR_VERSION,
            "Editor-Plugin-Version": self.EDITOR_PLUGIN_VERSION,
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
        try:
            response = requests.post(
                self.COPILOT_CHAT_URL,
                headers=headers,
                json=body,
                timeout=self.config.timeout,
            )

            # Handle errors
            if response.status_code != 200:
                error_msg = f"GitHub Copilot API error: {response.status_code} - {response.text}"
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

        except requests.RequestException as e:
            raise Exception(f"GitHub Copilot API request failed: {str(e)}")

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream chat response from GitHub Copilot.

        Args:
            messages: List of messages
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Copilot-Integration-Id": self.COPILOT_INTEGRATION_ID,
            "User-Agent": self.USER_AGENT,
            "Editor-Version": self.EDITOR_VERSION,
            "Editor-Plugin-Version": self.EDITOR_PLUGIN_VERSION,
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
        try:
            response = requests.post(
                self.COPILOT_CHAT_URL,
                headers=headers,
                json=body,
                stream=True,
                timeout=self.config.timeout,
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if data["choices"][0]["delta"].get("content"):
                                yield data["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue

        except requests.RequestException as e:
            raise Exception(f"GitHub Copilot streaming failed: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Use tiktoken for accurate counting
        try:
            import tiktoken

            # GitHub Copilot uses GPT models
            if "gpt-4" in self.model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif "claude" in self.model:
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # Fallback to rough estimate
                return len(text) // 4

            return len(encoding.encode(text))
        except (ImportError, Exception):
            # Fallback to rough estimate: 1 token ≈ 4 characters
            return len(text) // 4

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "gpt-4"

    def requires_api_key(self) -> bool:
        """Check if provider requires an API key."""
        return True

    def supports_function_calling(self) -> bool:
        """Check if provider supports function/tool calling."""
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"GitHubCopilotProvider(model={self.model})"


# Popular model configurations
GITHUB_COPILOT_MODELS = {
    # GPT models (most common)
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",

    # Reasoning models
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini",

    # Claude models (via Copilot)
    "claude-3.5-sonnet": "claude-3.5-sonnet",
    "claude-3-opus": "claude-3-opus",

    # Codex models (legacy)
    "gpt-5-codex": "gpt-5-codex",
}
