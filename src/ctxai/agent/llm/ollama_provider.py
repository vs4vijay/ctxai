"""
Ollama LLM provider implementation.

Provides local model execution through Ollama.
"""

import json
import os
from collections.abc import Iterator
from typing import Optional

import requests

from ..config import AgentLLMConfig
from .base import BaseLLMProvider, LLMResponse, MessageRole, ToolCall


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider for local models.

    Supports any model available through Ollama:
    - codellama:13b, codellama:34b (best for coding)
    - deepseek-coder:6.7b, deepseek-coder:33b (fast, good)
    - llama3.1:8b, llama3.1:70b (general purpose)
    - qwen2.5-coder:7b (fast coding)
    - mistral:7b (small, fast)
    """

    def __init__(self, config: AgentLLMConfig):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        # Get base URL (default: localhost)
        self.base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Set model with default
        self.model = config.model or "codellama:13b"

        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama not responding: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat request to Ollama.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool schemas
            **kwargs: Additional parameters

        Returns:
            LLMResponse with content and tool calls
        """
        # Convert messages to Ollama format if needed
        ollama_messages = self._convert_messages(messages)

        # Prepare request
        body = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        # Add tools if provided
        # Note: Tool calling support varies by model
        if tools:
            body["tools"] = self._convert_tools_to_ollama(tools)

        # Call API
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=body,
            timeout=300,  # 5 minute timeout for local models
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        # Parse response
        data = response.json()
        message = data.get("message", {})

        # Extract content
        content = message.get("content", "")

        # Extract tool calls (if supported by model)
        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        name=tc["function"]["name"],
                        parameters=tc["function"].get("arguments", {}),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=data,
        )

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream chat response from Ollama.

        Args:
            messages: List of messages
            tools: Optional tool schemas
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        # Convert messages
        ollama_messages = self._convert_messages(messages)

        # Prepare request
        body = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if tools:
            body["tools"] = self._convert_tools_to_ollama(tools)

        # Stream response
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=body,
            stream=True,
            timeout=300,
        )

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if data.get("message", {}).get("content"):
                        yield data["message"]["content"]
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
        # Rough estimate for local models: 1 token ≈ 4 characters
        return len(text) // 4

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert OpenAI-format messages to Ollama format.

        Args:
            messages: Messages in OpenAI format

        Returns:
            Messages in Ollama format
        """
        # Ollama uses the same format as OpenAI for most cases
        ollama_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Handle system messages
            if role == "system":
                ollama_messages.append({"role": "system", "content": content})

            # Handle user messages
            elif role == "user":
                ollama_messages.append({"role": "user", "content": content})

            # Handle assistant messages
            elif role == "assistant":
                ollama_messages.append({"role": "assistant", "content": content})

            # Handle tool result messages (if needed)
            elif role == "tool":
                # Convert tool results to user messages for Ollama
                ollama_messages.append({
                    "role": "user",
                    "content": f"Tool result: {content}"
                })

        return ollama_messages

    def _convert_tools_to_ollama(self, tools: list[dict]) -> list[dict]:
        """
        Convert OpenAI-format tools to Ollama format.

        Args:
            tools: Tools in OpenAI format

        Returns:
            Tools in Ollama format
        """
        # Ollama's tool format is similar to OpenAI's
        # This is a pass-through for now, but can be customized
        return tools

    def list_available_models(self) -> list[str]:
        """
        List all models available in Ollama.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            pass
        return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600,  # 10 minute timeout for pulling
            )
            return response.status_code == 200
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"OllamaProvider(model={self.model}, url={self.base_url})"


# Recommended models for coding
OLLAMA_CODING_MODELS = {
    # Best for coding
    "codellama-13b": "codellama:13b",
    "codellama-34b": "codellama:34b",
    "deepseek-coder-7b": "deepseek-coder:6.7b",
    "deepseek-coder-33b": "deepseek-coder:33b",
    "qwen-coder-7b": "qwen2.5-coder:7b",

    # General purpose
    "llama3.1-8b": "llama3.1:8b",
    "llama3.1-70b": "llama3.1:70b",
    "mistral-7b": "mistral:7b",

    # Small and fast
    "phi3-mini": "phi3:mini",
    "gemma-2b": "gemma:2b",
}
