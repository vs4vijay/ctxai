"""
MCP Server command implementation.

Exposes ctxai functionality as MCP tools that can be used by LLMs
and AI agents through the Model Context Protocol.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from rich.console import Console

from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..utils import get_indexes_dir
from ..vector_store import VectorStore
from .index_command import index_codebase as run_index

# Setup logging to stderr (not stdout for STDIO servers)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Uses stderr by default
)
logger = logging.getLogger(__name__)

console = Console(stderr=True)  # Use stderr to avoid corrupting STDIO communication


def create_server(project_path: Path | None = None) -> "FastMCP":
    """
    Create and configure the MCP server using FastMCP.

    Args:
        project_path: Optional project path for configuration

    Returns:
        Configured FastMCP Server instance
    """
    if not MCP_AVAILABLE:
        raise ImportError("MCP is not installed. Install it with: pip install ctxai[mcp]")

    # Initialize FastMCP server
    mcp = FastMCP("ctxai")

    @mcp.tool()
    async def list_indexes() -> str:
        """
        List all available code indexes with their statistics.

        Returns:
            String containing list of indexes with chunk counts and paths
        """
        try:
            logger.info("Listing indexes")
            indexes_dir = get_indexes_dir(project_path)

            if not indexes_dir.exists():
                return "No indexes found. Create one using the index_codebase tool."

            indexes = []
            for index_path in indexes_dir.iterdir():
                if index_path.is_dir():
                    try:
                        vector_store = VectorStore(storage_path=index_path, collection_name=index_path.name)
                        stats = vector_store.get_stats()

                        indexes.append(
                            {
                                "name": index_path.name,
                                "chunks": stats["total_chunks"],
                                "path": str(index_path),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Could not load index {index_path.name}: {e}")

            if not indexes:
                return "No valid indexes found."

            result = "Available indexes:\n\n"
            for idx in indexes:
                result += f"- **{idx['name']}**: {idx['chunks']:,} chunks\n"
                result += f"  Path: {idx['path']}\n\n"

            logger.info(f"Found {len(indexes)} indexes")
            return result

        except Exception as e:
            error_msg = f"Error listing indexes: {e}"
            logger.error(error_msg)
            return error_msg

    @mcp.tool()
    async def index_codebase(
        path: str,
        name: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        follow_gitignore: bool = True,
    ) -> str:
        """
        Index a codebase for semantic search. Creates embeddings and stores them in a vector database.

        Args:
            path: Path to the codebase directory to index
            name: Name for the index
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: Additional file patterns to exclude beyond .gitignore
            follow_gitignore: Follow .gitignore patterns when traversing (default: True)

        Returns:
            Success message with index statistics
        """
        try:
            logger.info(f"Indexing codebase: path={path}, name={name}")
            path_obj = Path(path)

            if not path_obj.exists():
                return f"Error: Path does not exist: {path}"

            if not path_obj.is_dir():
                return f"Error: Path is not a directory: {path}"

            # Run indexing in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                run_index,
                path_obj,
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

            logger.info(f"Indexing complete: {stats['total_chunks']} chunks")
            return result

        except Exception as e:
            error_msg = f"Error indexing codebase: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @mcp.tool()
    async def query_codebase(index_name: str, query: str, n_results: int = 5) -> str:
        """
        Query an indexed codebase using natural language. Returns relevant code chunks with metadata.

        Args:
            index_name: Name of the index to query
            query: Natural language query to search the codebase
            n_results: Number of results to return (default: 5, max: 20)

        Returns:
            Formatted results with code chunks, similarity scores, and metadata
        """
        try:
            logger.info(f"Querying codebase: index={index_name}, query={query}")

            # Validate n_results
            n_results = max(1, min(n_results, 20))

            # Load configuration
            config_manager = ConfigManager(project_path)
            config = config_manager.load()

            # Initialize embedding provider
            embeddings_generator = EmbeddingsFactory.create(config.embedding)

            # Load vector store
            indexes_dir = get_indexes_dir(project_path)
            index_path = indexes_dir / index_name

            if not index_path.exists():
                return f"Error: Index '{index_name}' not found. Use list_indexes to see available indexes."

            vector_store = VectorStore(storage_path=index_path, collection_name=index_name)

            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(None, embeddings_generator.generate_embedding, query)

            # Search
            results = vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
            )

            if not results:
                return f"No results found for query: {query}"

            # Format results
            result_text = f'Found {len(results)} result(s) for: "{query}"\n\n'

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
                result_text += "**Code:**\n```" + metadata.get("language", "text") + "\n"

                # Limit content length
                if len(content) > 500:
                    result_text += content[:500] + "\n... (truncated)\n"
                else:
                    result_text += content + "\n"

                result_text += "```\n\n"

            logger.info(f"Query returned {len(results)} results")
            return result_text

        except Exception as e:
            error_msg = f"Error querying codebase: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    @mcp.tool()
    async def get_index_stats(index_name: str) -> str:
        """
        Get detailed statistics about a specific index.

        Args:
            index_name: Name of the index

        Returns:
            Statistics including chunk count, storage size, and location
        """
        try:
            logger.info(f"Getting stats for index: {index_name}")

            indexes_dir = get_indexes_dir(project_path)
            index_path = indexes_dir / index_name

            if not index_path.exists():
                return f"Error: Index '{index_name}' not found."

            vector_store = VectorStore(storage_path=index_path, collection_name=index_name)
            stats = vector_store.get_stats()

            # Get additional info
            size_mb = sum(f.stat().st_size for f in index_path.rglob("*") if f.is_file()) / (1024 * 1024)

            result = f"## Index: {index_name}\n\n"
            result += f"- **Total chunks:** {stats['total_chunks']:,}\n"
            result += f"- **Storage size:** {size_mb:.2f} MB\n"
            result += f"- **Location:** {index_path}\n"

            logger.info(f"Stats retrieved for {index_name}")
            return result

        except Exception as e:
            error_msg = f"Error getting index stats: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    return mcp


def start_mcp_server(project_path: Path | None = None):
    """
    Start the MCP server.

    Args:
        project_path: Optional project path for configuration
    """
    if not MCP_AVAILABLE:
        console.print("[red]✗ MCP is not installed[/red]\n")
        console.print("[yellow]Install it with:[/yellow] [cyan]pip install ctxai[mcp][/cyan]\n")
        return

    console.print("[bold blue]🚀 Starting MCP server...[/bold blue]\n")
    console.print("[dim]The server will communicate via stdio (standard input/output)[/dim]")
    console.print("[dim]Use this with MCP-compatible clients like Claude Desktop[/dim]\n")

    logger.info("Initializing MCP server")

    try:
        mcp = create_server(project_path)
        logger.info("MCP server initialized successfully")

        # Run the server
        mcp.run(transport="stdio")
    except Exception as e:
        error_msg = f"Failed to start MCP server: {e}"
        logger.error(error_msg, exc_info=True)
        console.print(f"[red]✗ {error_msg}[/red]\n")
