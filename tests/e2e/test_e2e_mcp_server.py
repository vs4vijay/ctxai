"""VS-06 acceptance tests using a real MCP client/server transport."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from unittest.mock import patch

import anyio
import pytest

pytest.importorskip("mcp", reason="MCP not installed")

from mcp import ClientSession

from ctxai.commands.index_command import IndexingCancelled
from ctxai.commands.server_command import create_server


@asynccontextmanager
async def connected_client(server, *, progress_callback=None):
    """Connect the SDK client and low-level server over in-memory MCP streams."""
    client_send, server_receive = anyio.create_memory_object_stream(20)
    server_send, client_receive = anyio.create_memory_object_stream(20)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(
            server._mcp_server.run,
            server_receive,
            server_send,
            server._mcp_server.create_initialization_options(),
        )
        async with ClientSession(client_receive, client_send) as client:
            await client.initialize()
            yield client
        tasks.cancel_scope.cancel()


async def call(client, name, arguments=None, *, progress_callback=None):
    result = await client.call_tool(name, arguments or {}, progress_callback=progress_callback)
    assert not result.isError
    assert result.structuredContent is not None
    assert result.structuredContent["schema_version"] == "1.0"
    return result.structuredContent


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_real_client_discovers_every_versioned_tool(temp_dir):
    with patch("ctxai.commands.server_command.get_indexes_dir", return_value=temp_dir / "indexes"):
        async with connected_client(create_server(temp_dir)) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == {
                "list_indexes",
                "index_codebase",
                "query_codebase",
                "get_index_stats",
                "graph_stats",
                "graph_symbol",
                "graph_neighbors",
            }
            timeout_schema = next(tool for tool in tools.tools if tool.name == "index_codebase").inputSchema
            assert timeout_schema["properties"]["timeout_seconds"]["default"] == 300
            result = await call(client, "list_indexes")
            assert result == {"schema_version": "1.0", "ok": True, "data": {"indexes": [], "count": 0}}


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_real_client_indexes_queries_and_inspects(sample_python_code, temp_dir, patch_embeddings_factory):
    indexes_dir = temp_dir / ".ctxai" / "indexes"
    indexes_dir.mkdir(parents=True)
    progress = []

    async def record_progress(value, total, message):
        progress.append((value, total, message))

    with patch("ctxai.commands.server_command.get_indexes_dir", return_value=indexes_dir):
        async with connected_client(create_server(temp_dir)) as client:
            indexed = await call(
                client,
                "index_codebase",
                {
                    "path": str(sample_python_code),
                    "name": "test-mcp-index",
                    "include_patterns": ["*.py"],
                    "follow_gitignore": False,
                },
                progress_callback=record_progress,
            )
            assert indexed["ok"]
            assert indexed["data"]["chunks"] > 0
            assert indexed["data"]["files"] > 0
            assert progress
            # The indexing pipeline has six stages (IG-01 added the symbol
            # graph stage); progress reports completion of the final one.
            assert progress[-1][0:2] == (6.0, 6.0)

            listed = await call(client, "list_indexes")
            assert listed["data"]["indexes"][0]["name"] == "test-mcp-index"
            assert listed["data"]["indexes"][0]["index_schema_version"] == 1

            queried = await call(
                client,
                "query_codebase",
                {"index_name": "test-mcp-index", "query": "function that greets", "n_results": 3},
            )
            assert queried["ok"]
            assert queried["data"]["count"] > 0
            first = queried["data"]["results"][0]
            assert {"file_path", "start_line", "end_line", "content", "similarity"} <= first.keys()

            stats = await call(client, "get_index_stats", {"index_name": "test-mcp-index"})
            assert stats["ok"]
            assert stats["data"]["chunks"] == indexed["data"]["chunks"]
            assert stats["data"]["index_schema_version"] == 1


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_invalid_inputs_have_stable_error_codes(temp_dir):
    indexes_dir = temp_dir / "indexes"
    indexes_dir.mkdir()
    with patch("ctxai.commands.server_command.get_indexes_dir", return_value=indexes_dir):
        async with connected_client(create_server(temp_dir)) as client:
            missing = await call(client, "query_codebase", {"index_name": "missing", "query": "anything"})
            assert missing["ok"] is False
            assert missing["error"]["code"] == "not_found"

            traversal = await call(client, "get_index_stats", {"index_name": "../escape"})
            assert traversal["ok"] is False
            assert traversal["error"]["code"] == "invalid_input"

            empty = await call(client, "query_codebase", {"index_name": "missing", "query": " "})
            assert empty["error"]["code"] == "invalid_input"


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_index_timeout_is_cooperative_and_deterministic(temp_dir):
    indexes_dir = temp_dir / "indexes"
    indexes_dir.mkdir()

    def wait_until_cancelled(*args, cancel_event, **kwargs):
        while not cancel_event.wait(0.01):
            time.sleep(0.001)
        raise IndexingCancelled("cancelled")

    with (
        patch("ctxai.commands.server_command.get_indexes_dir", return_value=indexes_dir),
        patch("ctxai.commands.server_command.run_index", side_effect=wait_until_cancelled),
    ):
        async with connected_client(create_server(temp_dir)) as client:
            result = await call(
                client,
                "index_codebase",
                {"path": str(temp_dir), "name": "timeout-index", "timeout_seconds": 1},
            )
            assert result["ok"] is False
            assert result["error"]["code"] == "timeout"
            assert not (indexes_dir / "timeout-index" / "manifest.json").exists()


@pytest.mark.e2e
@pytest.mark.mcp
@pytest.mark.asyncio
async def test_cooperative_cancellation_has_stable_error(temp_dir):
    indexes_dir = temp_dir / "indexes"
    indexes_dir.mkdir()
    with (
        patch("ctxai.commands.server_command.get_indexes_dir", return_value=indexes_dir),
        patch("ctxai.commands.server_command.run_index", side_effect=IndexingCancelled("cancelled")),
    ):
        async with connected_client(create_server(temp_dir)) as client:
            result = await call(
                client,
                "index_codebase",
                {"path": str(temp_dir), "name": "cancelled-index"},
            )
            assert result["ok"] is False
            assert result["error"]["code"] == "cancelled"
