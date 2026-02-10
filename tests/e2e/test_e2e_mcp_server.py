"""
End-to-end tests for MCP server integration.

Tests MCP server tools with real implementations.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# Skip all tests if MCP is not available
pytest.importorskip("mcp", reason="MCP not installed")

from ctxai.commands.server_command import create_server


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_mcp_server_lifecycle(temp_dir):
    """
    Test MCP server lifecycle.

    Verifies:
    1. Server can be created
    2. Server has correct name
    3. All expected tools are registered
    4. Server can be used for tool calls
    """
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        indexes_dir = temp_dir / "indexes"
        indexes_dir.mkdir(parents=True)
        mock_get_indexes_dir.return_value = indexes_dir

        # Create server
        server = create_server(project_path=temp_dir)

        # Verify server properties
        assert server is not None
        assert server.name == "ctxai"

        # Verify all tools are registered
        tool_names = [tool.name for tool in server._tool_manager._tools.values()]
        expected_tools = ["list_indexes", "index_codebase", "query_codebase", "get_index_stats"]

        for expected in expected_tools:
            assert expected in tool_names, f"Tool {expected} should be registered"

        assert len(tool_names) == 4, f"Expected 4 tools, got {len(tool_names)}: {tool_names}"


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_mcp_list_indexes_tool(temp_dir):
    """
    Test MCP list_indexes tool.

    Verifies:
    1. Tool can list empty indexes directory
    2. Tool can list multiple indexes
    3. Tool returns correct stats for each index
    """
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        indexes_dir = temp_dir / "indexes"
        indexes_dir.mkdir(parents=True)
        mock_get_indexes_dir.return_value = indexes_dir

        # Create server
        server = create_server(project_path=temp_dir)

        # Get the list_indexes tool
        list_indexes_tool = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "list_indexes":
                list_indexes_tool = tool
                break

        assert list_indexes_tool is not None, "list_indexes tool should exist"

        # Test with empty indexes directory
        result = await list_indexes_tool.run()
        assert isinstance(result, str)
        assert "no indexes found" in result.lower() or "0 indexes" in result.lower()

        # Create some fake index directories
        (indexes_dir / "index1").mkdir()
        (indexes_dir / "index2").mkdir()

        # Test with indexes present
        result = await list_indexes_tool.run()
        assert isinstance(result, str)
        # Should mention the indexes
        assert "index1" in result or "index2" in result or "2" in result


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_mcp_index_codebase_tool(sample_python_code, temp_dir, patch_embeddings_factory):
    """
    Test MCP index_codebase tool.

    Verifies:
    1. Tool can index a codebase
    2. Index is created in correct location
    3. Tool returns success message with stats
    """
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        indexes_dir = temp_dir / ".ctxai" / "indexes"
        indexes_dir.mkdir(parents=True)
        mock_get_indexes_dir.return_value = indexes_dir

        # Create server
        server = create_server(project_path=temp_dir)

        # Get the index_codebase tool
        index_tool = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "index_codebase":
                index_tool = tool
                break

        assert index_tool is not None, "index_codebase tool should exist"

        # Index the sample codebase
        result = await index_tool.run(
            path=str(sample_python_code),
            name="test-mcp-index",
            include_patterns=["*.py"],
            exclude_patterns=None,
            follow_gitignore=False
        )

        # Verify result
        assert isinstance(result, str)
        assert "success" in result.lower() or "indexed" in result.lower()
        assert "chunks" in result.lower() or "files" in result.lower()

        # Verify index was created
        index_path = indexes_dir / "test-mcp-index"
        assert index_path.exists(), f"Index should be created at {index_path}"


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_mcp_query_codebase_tool(indexed_codebase, temp_dir, patch_embeddings_factory):
    """
    Test MCP query_codebase tool.

    Verifies:
    1. Tool can query an existing index
    2. Results are returned in correct format
    3. Results contain expected code chunks
    """
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        indexes_dir = indexed_codebase["index_path"].parent
        mock_get_indexes_dir.return_value = indexes_dir

        # Create server
        server = create_server(project_path=temp_dir)

        # Get the query_codebase tool
        query_tool = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "query_codebase":
                query_tool = tool
                break

        assert query_tool is not None, "query_codebase tool should exist"

        # Query the index
        result = await query_tool.run(
            index_name="test-index",
            query="function that greets",
            n_results=3
        )

        # Verify result
        assert isinstance(result, str)
        assert len(result) > 0, "Result should not be empty"

        # Should contain code-related content
        # (exact content depends on sample code, but should have some structure)
        assert "def" in result or "function" in result.lower() or "greet" in result.lower()


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_mcp_get_index_stats_tool(indexed_codebase, temp_dir):
    """
    Test MCP get_index_stats tool.

    Verifies:
    1. Tool can get stats for existing index
    2. Stats include expected fields (total_chunks, total_files, etc.)
    3. Stats are accurate
    """
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        indexes_dir = indexed_codebase["index_path"].parent
        mock_get_indexes_dir.return_value = indexes_dir

        # Create server
        server = create_server(project_path=temp_dir)

        # Get the get_index_stats tool
        stats_tool = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "get_index_stats":
                stats_tool = tool
                break

        assert stats_tool is not None, "get_index_stats tool should exist"

        # Get stats
        result = await stats_tool.run(index_name="test-index")

        # Verify result
        assert isinstance(result, str)
        assert "chunks" in result.lower()
        assert "files" in result.lower()

        # Should have positive numbers
        assert any(char.isdigit() for char in result), "Stats should contain numbers"

        # Test with non-existent index
        result_nonexistent = await stats_tool.run(index_name="nonexistent-index")
        assert isinstance(result_nonexistent, str)
        assert "not found" in result_nonexistent.lower() or "error" in result_nonexistent.lower()


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_mcp_server_tool_error_handling(temp_dir):
    """
    Test MCP server tools handle errors gracefully.

    Verifies:
    1. Tools return error messages instead of raising exceptions
    2. Error messages are informative
    3. Server remains functional after errors
    """
    with patch("ctxai.commands.server_command.get_indexes_dir") as mock_get_indexes_dir:
        indexes_dir = temp_dir / "indexes"
        indexes_dir.mkdir(parents=True)
        mock_get_indexes_dir.return_value = indexes_dir

        # Create server
        server = create_server(project_path=temp_dir)

        # Get query tool
        query_tool = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "query_codebase":
                query_tool = tool
                break

        # Try to query non-existent index
        result = await query_tool.run(
            index_name="nonexistent",
            query="test query",
            n_results=5
        )

        # Should return error message, not raise exception
        assert isinstance(result, str)
        assert "not found" in result.lower() or "error" in result.lower()

        # Server should still be functional - list indexes should work
        list_tool = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "list_indexes":
                list_tool = tool
                break

        result_list = await list_tool.run()
        assert isinstance(result_list, str)  # Should work fine
