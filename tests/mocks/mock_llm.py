"""
Mock LLM provider for agent testing.

This mock provider returns predefined responses instead of making real API calls,
enabling fast, deterministic testing without costs.
"""

from typing import List, Dict, Any, Optional, Generator
from ctxai.agent.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ToolCall,
)
from ctxai.agent.config import AgentLLMConfig


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM that returns predefined responses.

    Useful for testing agent workflows without actual API calls.
    Supports configuring exact sequences of responses including tool calls.
    """

    def __init__(self, config: AgentLLMConfig = None, responses: List[Dict[str, Any]] = None):
        """
        Initialize mock LLM provider.

        Args:
            config: Agent LLM configuration (will be created if None)
            responses: List of response configurations. Each response can include:
                - content: Response text
                - tool_calls: List of tool call dicts with 'name', 'parameters', optionally 'id'
                - finish_reason: One of "stop", "tool_calls", "length"
        """
        # Create dummy config if none provided
        if config is None:
            config = AgentLLMConfig(
                provider="mock",
                model="mock-model",
                api_key="mock-key",
                temperature=0.7,
                max_tokens=4096
            )

        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []  # Track all calls for assertions

    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        return "mock-model-v1"

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
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
        self.call_history.append({
            "messages": [msg.to_dict() for msg in messages],
            "tools": tools,
            "kwargs": kwargs,
        })

        # Return next response or default if we've exhausted responses
        if self.call_count >= len(self.responses):
            # Default final response
            response = LLMResponse(
                content="Task completed successfully.",
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
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
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{self.call_count}_{i}"),
                    name=tc["name"],
                    parameters=tc.get("parameters", {})
                ))

        # Determine finish reason
        finish_reason = response_config.get("finish_reason")
        if finish_reason is None:
            finish_reason = "tool_calls" if tool_calls else "stop"

        # Create response
        return LLMResponse(
            content=response_config.get("content", ""),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=response_config.get("usage", {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120
            })
        )

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
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
        for char in response.content:
            yield char

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


def create_mock_response(content: str = "", tool_calls: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper function to create a mock response configuration.

    Args:
        content: Response text content
        tool_calls: Optional list of tool calls to include

    Returns:
        Response configuration dict

    Example:
        >>> responses = [
        ...     create_mock_response(
        ...         content="I'll read the file",
        ...         tool_calls=[{"name": "read_file", "parameters": {"path": "test.py"}}]
        ...     ),
        ...     create_mock_response(content="The file contains: ...")
        ... ]
        >>> provider = MockLLMProvider(responses=responses)
    """
    response = {"content": content}
    if tool_calls:
        response["tool_calls"] = tool_calls
    return response
