"""
MCP Server implementation for ctxai.

Exposes index and query functionality as MCP tools that can be used by LLMs
and AI agents through the Model Context Protocol.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from rich.console import Console

from .config import ConfigManager
from .embeddings import EmbeddingsFactory
from .utils import get_indexes_dir
from .vector_store import VectorStore
from .commands.index_command import index_codebase as run_index

console = Console()


def create_mcp_server(project_path: Optional[Path] = None) -> "Server":
    """
    Create and configure the MCP server.
    
    Args:
        project_path: Optional project path for configuration
        
    Returns:
        Configured MCP Server instance
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP is not installed. Install it with: pip install ctxai[mcp]"
        )
    
    # Create server instance
    server = Server("ctxai")
    
    # Tool: list_indexes
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools."""
        return [
            Tool(
                name="list_indexes",
                description="List all available code indexes with their statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="index_codebase",
                description="Index a codebase for semantic search. Creates embeddings and stores them in a vector database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the codebase directory to index",
                        },
                        "name": {
                            "type": "string",
                            "description": "Name for the index",
                        },
                        "include_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File patterns to include (e.g., '*.py', '*.js')",
                        },
                        "exclude_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional file patterns to exclude beyond .gitignore",
                        },
                        "follow_gitignore": {
                            "type": "boolean",
                            "description": "Follow .gitignore patterns when traversing",
                            "default": True,
                        },
                    },
                    "required": ["path", "name"],
                },
            ),
            Tool(
                name="query_codebase",
                description="Query an indexed codebase using natural language. Returns relevant code chunks with metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "index_name": {
                            "type": "string",
                            "description": "Name of the index to query",
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural language query to search the codebase",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["index_name", "query"],
                },
            ),
            Tool(
                name="get_index_stats",
                description="Get detailed statistics about a specific index",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "index_name": {
                            "type": "string",
                            "description": "Name of the index",
                        },
                    },
                    "required": ["index_name"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[TextContent]:
        """Handle tool calls from the LLM."""
        try:
            if name == "list_indexes":
                return await handle_list_indexes(project_path)
            elif name == "index_codebase":
                return await handle_index_codebase(arguments, project_path)
            elif name == "query_codebase":
                return await handle_query_codebase(arguments, project_path)
            elif name == "get_index_stats":
                return await handle_get_index_stats(arguments, project_path)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            error_msg = f"Error executing {name}: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return [TextContent(type="text", text=error_msg)]
    
    return server


async def handle_list_indexes(project_path: Optional[Path] = None) -> list[TextContent]:
    """Handle list_indexes tool call."""
    try:
        indexes_dir = get_indexes_dir(project_path)
        
        if not indexes_dir.exists():
            return [TextContent(
                type="text",
                text="No indexes found. Create one using the index_codebase tool."
            )]
        
        indexes = []
        for index_path in indexes_dir.iterdir():
            if index_path.is_dir():
                try:
                    vector_store = VectorStore(
                        storage_path=index_path,
                        collection_name=index_path.name
                    )
                    stats = vector_store.get_stats()
                    
                    indexes.append({
                        "name": index_path.name,
                        "chunks": stats["total_chunks"],
                        "path": str(index_path),
                    })
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load index {index_path.name}: {e}[/yellow]")
        
        if not indexes:
            return [TextContent(
                type="text",
                text="No valid indexes found."
            )]
        
        result = "Available indexes:\n\n"
        for idx in indexes:
            result += f"- **{idx['name']}**: {idx['chunks']:,} chunks\n"
            result += f"  Path: {idx['path']}\n\n"
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing indexes: {e}")]


async def handle_index_codebase(
    arguments: dict,
    project_path: Optional[Path] = None
) -> list[TextContent]:
    """Handle index_codebase tool call."""
    try:
        path = Path(arguments["path"])
        name = arguments["name"]
        include_patterns = arguments.get("include_patterns")
        exclude_patterns = arguments.get("exclude_patterns")
        follow_gitignore = arguments.get("follow_gitignore", True)
        
        if not path.exists():
            return [TextContent(
                type="text",
                text=f"Error: Path does not exist: {path}"
            )]
        
        if not path.is_dir():
            return [TextContent(
                type="text",
                text=f"Error: Path is not a directory: {path}"
            )]
        
        # Run indexing in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            run_index,
            path,
            name,
            include_patterns,
            exclude_patterns,
            follow_gitignore,
        )
        
        # Get stats after indexing
        indexes_dir = get_indexes_dir(project_path)
        index_path = indexes_dir / name
        vector_store = VectorStore(storage_path=index_path, collection_name=name)
        stats = vector_store.get_stats()
        
        result = f"✓ Successfully indexed codebase '{name}'\n\n"
        result += f"- Total chunks: {stats['total_chunks']:,}\n"
        result += f"- Location: {index_path}\n"
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error indexing codebase: {e}")]


async def handle_query_codebase(
    arguments: dict,
    project_path: Optional[Path] = None
) -> list[TextContent]:
    """Handle query_codebase tool call."""
    try:
        index_name = arguments["index_name"]
        query = arguments["query"]
        n_results = arguments.get("n_results", 5)
        
        # Load configuration
        config_manager = ConfigManager(project_path)
        config = config_manager.load()
        
        # Initialize embedding provider
        embeddings_generator = EmbeddingsFactory.create(config.embedding)
        
        # Load vector store
        indexes_dir = get_indexes_dir(project_path)
        index_path = indexes_dir / index_name
        
        if not index_path.exists():
            return [TextContent(
                type="text",
                text=f"Error: Index '{index_name}' not found. Use list_indexes to see available indexes."
            )]
        
        vector_store = VectorStore(storage_path=index_path, collection_name=index_name)
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            embeddings_generator.generate_embedding,
            query
        )
        
        # Search
        results = vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
        )
        
        if not results:
            return [TextContent(
                type="text",
                text=f"No results found for query: {query}"
            )]
        
        # Format results
        result_text = f"Found {len(results)} result(s) for: \"{query}\"\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            content = result["content"]
            distance = result["distance"]
            similarity = max(0, 1 - distance)
            
            file_path = Path(metadata["file_path"])
            
            result_text += f"## Result {i} (Similarity: {similarity:.1%})\n\n"
            result_text += f"**File:** {file_path}\n"
            result_text += f"**Lines:** {metadata['start_line']}-{metadata['end_line']}\n"
            result_text += f"**Type:** {metadata['chunk_type']} ({metadata['language']})\n\n"
            result_text += "**Code:**\n```" + metadata.get('language', 'text') + "\n"
            
            # Limit content length
            if len(content) > 500:
                result_text += content[:500] + "\n... (truncated)\n"
            else:
                result_text += content + "\n"
            
            result_text += "```\n\n"
        
        return [TextContent(type="text", text=result_text)]
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return [TextContent(
            type="text",
            text=f"Error querying codebase: {e}\n\nDetails:\n{error_details}"
        )]


async def handle_get_index_stats(
    arguments: dict,
    project_path: Optional[Path] = None
) -> list[TextContent]:
    """Handle get_index_stats tool call."""
    try:
        index_name = arguments["index_name"]
        
        indexes_dir = get_indexes_dir(project_path)
        index_path = indexes_dir / index_name
        
        if not index_path.exists():
            return [TextContent(
                type="text",
                text=f"Error: Index '{index_name}' not found."
            )]
        
        vector_store = VectorStore(storage_path=index_path, collection_name=index_name)
        stats = vector_store.get_stats()
        
        # Get additional info
        size_mb = sum(
            f.stat().st_size for f in index_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        
        result = f"## Index: {index_name}\n\n"
        result += f"- **Total chunks:** {stats['total_chunks']:,}\n"
        result += f"- **Storage size:** {size_mb:.2f} MB\n"
        result += f"- **Location:** {index_path}\n"
        
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting index stats: {e}")]


async def run_mcp_server(project_path: Optional[Path] = None):
    """
    Run the MCP server.
    
    Args:
        project_path: Optional project path for configuration
    """
    if not MCP_AVAILABLE:
        console.print("[red]✗ MCP is not installed[/red]\n")
        console.print(
            "[yellow]Install it with:[/yellow] [cyan]pip install ctxai[mcp][/cyan]\n"
        )
        return
    
    console.print("[bold blue]🚀 Starting MCP server...[/bold blue]\n")
    console.print("[dim]The server will communicate via stdio (standard input/output)[/dim]")
    console.print("[dim]Use this with MCP-compatible clients like Claude Desktop[/dim]\n")
    
    server = create_mcp_server(project_path)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
