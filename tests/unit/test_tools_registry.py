"""Tests for ctxai.agent.tools.registry."""

import pytest

from ctxai.agent.tools.base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from ctxai.agent.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="Echo input",
            parameters=[
                ToolParameter(
                    name="message",
                    type=ToolParameterType.STRING,
                    description="Text to echo",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs):
        return {"success": True, "result": kwargs.get("message")}


class FailingTool(BaseTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(name="failing", description="Always fails")

    async def execute(self, **kwargs):
        raise RuntimeError("boom")


def test_register_and_lookup():
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert reg.has_tool("echo")
    assert reg.get_tool("echo") is not None
    assert len(reg) == 1


def test_unregister_removes_tool():
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert reg.unregister("echo") is True
    assert reg.unregister("not-real") is False
    assert "echo" not in reg


def test_register_multiple():
    reg = ToolRegistry()
    reg.register_multiple([EchoTool(), FailingTool()])
    assert len(reg) == 2


def test_get_all_schemas_anthropic_format():
    reg = ToolRegistry()
    reg.register(EchoTool())
    schemas = reg.get_all_schemas(format="anthropic")
    assert schemas[0]["name"] == "echo"
    assert "input_schema" in schemas[0]


def test_get_all_schemas_openai_format():
    reg = ToolRegistry()
    reg.register(EchoTool())
    schemas = reg.get_all_schemas(format="openai")
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"


def test_get_all_schemas_rejects_invalid_format():
    reg = ToolRegistry()
    reg.register(EchoTool())
    with pytest.raises(ValueError):
        reg.get_all_schemas(format="bogus")


@pytest.mark.asyncio
async def test_execute_tool_returns_result():
    reg = ToolRegistry()
    reg.register(EchoTool())
    result = await reg.execute_tool("echo", message="hi")
    assert result["success"] is True
    assert result["result"] == "hi"


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_error():
    reg = ToolRegistry()
    result = await reg.execute_tool("nope")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_failing_tool_returns_error_not_raise():
    reg = ToolRegistry()
    reg.register(FailingTool())
    result = await reg.execute_tool("failing")
    assert result["success"] is False
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_execute_multiple_concurrent():
    reg = ToolRegistry()
    reg.register(EchoTool())
    results = await reg.execute_multiple(
        [
            {"name": "echo", "parameters": {"message": "a"}},
            {"name": "echo", "parameters": {"message": "b"}},
        ]
    )
    assert [r["result"] for r in results] == ["a", "b"]


def test_tool_descriptions_includes_params():
    reg = ToolRegistry()
    reg.register(EchoTool())
    desc = reg.get_tool_descriptions()
    assert "echo" in desc
    assert "message" in desc


def test_clear_removes_all():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.clear()
    assert len(reg) == 0


def test_validate_parameters_catches_missing():
    tool = EchoTool()
    valid, err = tool.validate_parameters()
    assert valid is False
    assert "message" in err
