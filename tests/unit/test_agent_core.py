"""Tests for ctxai.agent.core agent loop."""

from pathlib import Path

import pytest

from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.tools.base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from ctxai.agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


class EchoTool(BaseTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="Echo",
            parameters=[
                ToolParameter(
                    name="message",
                    type=ToolParameterType.STRING,
                    description="Text",
                    required=True,
                )
            ],
        )

    async def execute(self, **kwargs):
        return {"success": True, "result": kwargs.get("message", "")}


def _make_agent(responses, tmp_path: Path, max_iter: int = 5):
    llm = MockLLMProvider(responses=responses)
    registry = ToolRegistry()
    registry.register(EchoTool())
    cfg = AgentLoopConfig(
        llm_provider=llm,
        tool_registry=registry,
        agent_config=AgentConfig(),
        working_directory=tmp_path,
        available_indexes=[],
        verbose=False,
        max_iterations=max_iter,
    )
    return Agent(cfg)


@pytest.mark.asyncio
async def test_simple_text_response(tmp_path: Path):
    agent = _make_agent([create_mock_response(content="Hello there")], tmp_path)
    out = await agent.process_message("hi")
    assert out == "Hello there"


@pytest.mark.asyncio
async def test_tool_call_then_final_response(tmp_path: Path):
    responses = [
        create_mock_response(
            content="I'll echo",
            tool_calls=[{"name": "echo", "parameters": {"message": "abc"}}],
        ),
        create_mock_response(content="Done: abc"),
    ]
    agent = _make_agent(responses, tmp_path)
    out = await agent.process_message("say abc")
    assert "Done" in out


@pytest.mark.asyncio
async def test_max_iterations_terminates(tmp_path: Path):
    # Use responses that always include a tool call to trigger the loop.
    responses = [
        create_mock_response(
            content="loop",
            tool_calls=[{"name": "echo", "parameters": {"message": "x"}}],
        )
    ] * 10
    agent = _make_agent(responses, tmp_path, max_iter=3)
    out = await agent.process_message("trigger loop")
    assert out  # Should return either loop-detected msg or max-iter msg


@pytest.mark.asyncio
async def test_clear_conversation_keeps_system(tmp_path: Path):
    agent = _make_agent([create_mock_response(content="ok")], tmp_path)
    await agent.process_message("hi")
    agent.clear_conversation()
    # System message should remain
    assert agent.context.get_message_count() == 1


@pytest.mark.asyncio
async def test_conversation_summary(tmp_path: Path):
    agent = _make_agent([create_mock_response(content="ok")], tmp_path)
    summary = agent.get_conversation_summary()
    assert "Messages" in summary
