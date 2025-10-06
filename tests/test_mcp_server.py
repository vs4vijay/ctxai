"""
Tests for MCP server functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Skip all tests if MCP is not available
pytest.importorskip("mcp", reason="MCP not installed")

from ctxai.commands.server_command import create_server
from ctxai.config import Config, EmbeddingConfig, IndexConfig


def test_create_server():
    """Test creating MCP server instance."""
    server = create_server()
    assert server is not None
    assert server.name == "ctxai"


def test_create_server_without_mcp():
    """Test error when MCP is not available."""
    with patch("ctxai.commands.server_command.MCP_AVAILABLE", False):
        with pytest.raises(ImportError, match="MCP is not installed"):
            create_server()


@pytest.mark.asyncio
async def test_list_indexes_via_server(tmp_path):
    """Test listing indexes through server."""
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        mock_indexes_dir = tmp_path / "indexes"
        mock_indexes_dir.mkdir()
        mock_get_indexes_dir.return_value = mock_indexes_dir
        
        server = create_server()
        
        # The tools should be registered
        assert len(server._tool_manager._tools) == 4
        tool_names = [tool.name for tool in server._tool_manager._tools.values()]
        assert "list_indexes" in tool_names
        assert "index_codebase" in tool_names
        assert "query_codebase" in tool_names
        assert "get_index_stats" in tool_names


@pytest.mark.asyncio
async def test_server_with_project_path(tmp_path):
    """Test server with custom project path."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    
    server = create_server(project_path=project_path)
    assert server is not None
    assert server.name == "ctxai"
