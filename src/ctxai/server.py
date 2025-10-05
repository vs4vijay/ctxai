"""MCP Server for semantic code search."""

import argparse
import asyncio
import sys
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

load_dotenv()

# Global variable to store the index name
INDEX_NAME = None


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("ctxai")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search_code",
                description="Search for code using semantic search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to search for code",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        if name == "search_code":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            
            # TODO: Implement actual search logic using the INDEX_NAME
            result = f"Searching in index '{INDEX_NAME}' for: {query} (limit: {limit})"
            
            return [
                TextContent(
                    type="text",
                    text=result,
                )
            ]
        
        raise ValueError(f"Unknown tool: {name}")
    
    return server


async def run_mcp_server(index_name: str):
    """Run the MCP server."""
    global INDEX_NAME
    INDEX_NAME = index_name
    
    print(f"Starting MCP server with index: {index_name}", file=sys.stderr)
    
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run_server(index_name: str, port: int = 8000):
    """Start the MCP server.
    
    Args:
        index_name: Name of the index to use
        port: Port to run the server on (not used for MCP stdio server)
    """
    asyncio.run(run_mcp_server(index_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the ctxai MCP server")
    parser.add_argument("--index", required=True, help="Index name to use")
    parser.add_argument("--port", type=int, default=8000, help="Port (not used for stdio)")
    
    args = parser.parse_args()
    run_server(args.index, args.port)