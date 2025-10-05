"""Tests for the MCP server."""

import pytest

from ctxai.server import create_server


def test_create_server():
    """Test that the MCP server can be created."""
    server = create_server()
    assert server is not None
    assert server.name == "ctxai"


@pytest.mark.asyncio
async def test_list_tools():
    """Test that tools are listed correctly."""
    server = create_server()
    
    # Get the list_tools handler
    tools = await server._tool_manager.list_tools()
    
    assert len(tools) > 0
    assert any(tool.name == "search_code" for tool in tools)
    
    # Check the search_code tool
    search_tool = next(tool for tool in tools if tool.name == "search_code")
    assert search_tool.description == "Search for code using semantic search"
    assert "query" in search_tool.inputSchema["properties"]

