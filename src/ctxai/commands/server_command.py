"""
MCP Server command implementation.

Exposes ctxai functionality as MCP tools that can be used by LLMs
and AI agents through the Model Context Protocol.
"""

import asyncio
import contextlib
import logging
import re
import threading
from functools import partial
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import Context, FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from rich.console import Console

from ..agent.llm.base import ProviderError, ProviderErrorKind
from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..graph.adapters import capabilities_payload
from ..graph.dto import GraphStatsResult, NeighborsResult, SymbolSearchResult
from ..graph.operations import (
    GraphIndexNotFoundError,
    GraphNotBuiltError,
    GraphOperations,
)
from ..graph.store import GraphStoreError
from ..index_manifest import IndexManifest
from ..mcp_protocol import MCPErrorCode, failure, success
from ..utils import get_indexes_dir
from ..vector_store import VectorStore
from .index_command import IndexingCancelled
from .index_command import index_codebase as run_index

# Setup logging to stderr (not stdout for STDIO servers)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Uses stderr by default
)
logger = logging.getLogger(__name__)

console = Console(stderr=True)  # Use stderr to avoid corrupting STDIO communication
INDEX_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _valid_index_name(name: str) -> bool:
    """Keep index lookups below the configured indexes directory."""
    return bool(INDEX_NAME_PATTERN.fullmatch(name))


def _provider_error_code(error: Exception) -> str | None:
    """Map a ProviderError onto the stable MCP envelope codes.

    Non-provider errors return ``None`` so callers keep their default codes.

    Args:
        error: The raised exception.

    Returns:
        The MCP error code for provider errors, otherwise ``None``.
    """
    if not isinstance(error, ProviderError):
        return None
    if error.kind is ProviderErrorKind.CANCELLED:
        return MCPErrorCode.CANCELLED
    if error.kind is ProviderErrorKind.TIMEOUT:
        return MCPErrorCode.TIMEOUT
    return MCPErrorCode.INTERNAL_ERROR


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
    async def list_indexes() -> dict[str, Any]:
        """
        List all available code indexes with their statistics.

        Returns:
            Versioned result containing indexes, chunk counts, and paths
        """
        try:
            logger.info("Listing indexes")
            indexes_dir = get_indexes_dir(project_path)

            if not indexes_dir.exists():
                return success({"indexes": [], "count": 0})

            indexes = []
            for index_path in indexes_dir.iterdir():
                if index_path.is_dir():
                    try:
                        vector_store = VectorStore(storage_path=index_path, collection_name=index_path.name)
                        stats = vector_store.get_stats()

                        manifest = IndexManifest.load_optional(index_path)
                        indexes.append(
                            {
                                "name": index_path.name,
                                "chunks": stats["total_chunks"],
                                "files": stats.get("unique_files", 0),
                                "path": str(index_path),
                                "index_schema_version": manifest.schema_version if manifest else None,
                                "updated_at": manifest.updated_at if manifest else None,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Could not load index {index_path.name}: {e}")

            logger.info(f"Found {len(indexes)} indexes")
            return success({"indexes": indexes, "count": len(indexes)})

        except Exception as e:
            error_msg = f"Error listing indexes: {e}"
            logger.error(error_msg)
            return failure(MCPErrorCode.STORAGE_FAILED, error_msg)

    @mcp.tool()
    async def index_codebase(
        path: str,
        name: str,
        ctx: Context,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        follow_gitignore: bool = True,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        """
        Index a codebase for semantic search. Creates embeddings and stores them in a vector database.

        Args:
            path: Path to the codebase directory to index
            name: Name for the index
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: Additional file patterns to exclude beyond .gitignore
            follow_gitignore: Follow .gitignore patterns when traversing (default: True)
            timeout_seconds: Maximum indexing duration in seconds

        Returns:
            Versioned result with index statistics or a stable error
        """
        try:
            logger.info(f"Indexing codebase: path={path}, name={name}")
            if not _valid_index_name(name):
                return failure(
                    MCPErrorCode.INVALID_INPUT,
                    "Index name must contain only letters, numbers, '.', '_' or '-'",
                )
            if timeout_seconds < 1 or timeout_seconds > 3600:
                return failure(MCPErrorCode.INVALID_INPUT, "timeout_seconds must be between 1 and 3600")
            path_obj = Path(path).expanduser().resolve()

            if not path_obj.exists():
                return failure(MCPErrorCode.NOT_FOUND, f"Path does not exist: {path}")

            if not path_obj.is_dir():
                return failure(MCPErrorCode.INVALID_INPUT, f"Path is not a directory: {path}")

            # Run indexing in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            cancel_event = threading.Event()

            def progress(completed: int, total: int, message: str) -> None:
                if ctx is not None:
                    asyncio.run_coroutine_threadsafe(ctx.report_progress(completed, total, message), loop)

            operation = partial(
                run_index,
                path_obj,
                name,
                include_patterns,
                exclude_patterns,
                follow_gitignore,
                progress_callback=progress,
                cancel_event=cancel_event,
            )
            future = loop.run_in_executor(None, operation)
            try:
                indexing_result = await asyncio.wait_for(asyncio.shield(future), timeout=timeout_seconds)
            except TimeoutError:
                cancel_event.set()
                with contextlib.suppress(Exception):
                    await future
                return failure(MCPErrorCode.TIMEOUT, f"Indexing exceeded {timeout_seconds} seconds")
            except asyncio.CancelledError:
                cancel_event.set()
                with contextlib.suppress(Exception):
                    await future
                raise

            # Get stats after indexing
            indexes_dir = get_indexes_dir(project_path)
            index_path = indexes_dir / name
            vector_store = VectorStore(storage_path=index_path, collection_name=name)
            stats = vector_store.get_stats()

            logger.info(f"Indexing complete: {stats['total_chunks']} chunks")
            return success(
                {
                    "index_name": name,
                    "path": str(index_path),
                    "files": indexing_result.files,
                    "chunks": stats["total_chunks"],
                    "embedded_chunks": indexing_result.embedded_chunks,
                    "changed_files": indexing_result.changed_files,
                    "deleted_files": indexing_result.deleted_files,
                },
                message=f"Successfully indexed codebase '{name}'",
            )

        except IndexingCancelled:
            return failure(MCPErrorCode.CANCELLED, "Indexing cancelled by client")
        except Exception as e:
            error_msg = f"Error indexing codebase: {e}"
            logger.error(error_msg, exc_info=True)
            return failure(_provider_error_code(e) or MCPErrorCode.INDEX_FAILED, error_msg)

    @mcp.tool()
    async def query_codebase(index_name: str, query: str, n_results: int = 5) -> dict[str, Any]:
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

            if not _valid_index_name(index_name):
                return failure(MCPErrorCode.INVALID_INPUT, "Invalid index name")
            if not query.strip():
                return failure(MCPErrorCode.INVALID_INPUT, "Query must not be empty")
            n_results = max(1, min(n_results, 20))

            indexes_dir = get_indexes_dir(project_path)
            index_path = indexes_dir / index_name

            if not index_path.exists():
                return failure(
                    MCPErrorCode.NOT_FOUND,
                    f"Index '{index_name}' not found. Use list_indexes to see available indexes.",
                )

            # Avoid provider initialization (which may involve network access) until
            # the local request and target index have been validated.
            config_manager = ConfigManager(project_path)
            config = config_manager.load()
            embeddings_generator = EmbeddingsFactory.create(config.embedding)

            vector_store = VectorStore(storage_path=index_path, collection_name=index_name)

            # Generate query embedding
            loop = asyncio.get_running_loop()
            query_embedding = await loop.run_in_executor(None, embeddings_generator.generate_embedding, query)

            # Search
            results = vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
            )

            if not results:
                return success({"index_name": index_name, "query": query, "results": [], "count": 0})

            formatted_results = []
            for result in results:
                metadata = result["metadata"]
                content = result["content"]
                distance = result["distance"]
                similarity = max(0, 1 - distance)
                formatted_results.append(
                    {
                        "file_path": str(Path(metadata["file_path"])),
                        "start_line": metadata["start_line"],
                        "end_line": metadata["end_line"],
                        "chunk_type": metadata["chunk_type"],
                        "language": metadata["language"],
                        "similarity": similarity,
                        "content": content[:500],
                        "truncated": len(content) > 500,
                    }
                )

            logger.info(f"Query returned {len(results)} results")
            return success(
                {
                    "index_name": index_name,
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results),
                }
            )

        except Exception as e:
            error_msg = f"Error querying codebase: {e}"
            logger.error(error_msg, exc_info=True)
            return failure(_provider_error_code(e) or MCPErrorCode.QUERY_FAILED, error_msg)

    @mcp.tool()
    async def get_index_stats(index_name: str) -> dict[str, Any]:
        """
        Get detailed statistics about a specific index.

        Args:
            index_name: Name of the index

        Returns:
            Statistics including chunk count, storage size, and location
        """
        try:
            logger.info(f"Getting stats for index: {index_name}")

            if not _valid_index_name(index_name):
                return failure(MCPErrorCode.INVALID_INPUT, "Invalid index name")
            indexes_dir = get_indexes_dir(project_path)
            index_path = indexes_dir / index_name

            if not index_path.exists():
                return failure(MCPErrorCode.NOT_FOUND, f"Index '{index_name}' not found.")

            vector_store = VectorStore(storage_path=index_path, collection_name=index_name)
            stats = vector_store.get_stats()

            # Get additional info
            size_mb = sum(f.stat().st_size for f in index_path.rglob("*") if f.is_file()) / (1024 * 1024)

            manifest = IndexManifest.load_optional(index_path)

            logger.info(f"Stats retrieved for {index_name}")
            return success(
                {
                    "index_name": index_name,
                    "chunks": stats["total_chunks"],
                    "files": stats.get("unique_files", 0),
                    "storage_size_mb": round(size_mb, 3),
                    "path": str(index_path),
                    "index_schema_version": manifest.schema_version if manifest else None,
                    "repository_root": manifest.repository_root if manifest else None,
                    "updated_at": manifest.updated_at if manifest else None,
                }
            )

        except Exception as e:
            error_msg = f"Error getting index stats: {e}"
            logger.error(error_msg, exc_info=True)
            return failure(_provider_error_code(e) or MCPErrorCode.STORAGE_FAILED, error_msg)

    @mcp.tool()
    async def graph_stats(index_name: str) -> dict[str, Any]:
        """
        Get symbol graph statistics and health for an index (IG-02, read-only).

        Args:
            index_name: Name of the index

        Returns:
            Versioned graph stats: health verdict, schema/adapter versions,
            node/edge counts by kind, unresolved rate, and the per-language
            capability matrix.
        """
        try:
            logger.info(f"Getting graph stats for index: {index_name}")
            if not _valid_index_name(index_name):
                return failure(MCPErrorCode.INVALID_INPUT, "Invalid index name")
            operations = GraphOperations(project_path)
            stats = operations.stats(index_name)
            result = GraphStatsResult.build(index_name, stats, capabilities_payload())
            return success(result.to_dict())
        except GraphStoreError as e:
            return failure(MCPErrorCode.STORAGE_FAILED, f"Graph store error: {e}")
        except (GraphIndexNotFoundError, GraphNotBuiltError) as e:
            return failure(MCPErrorCode.NOT_FOUND, str(e))
        except Exception as e:
            error_msg = f"Error getting graph stats: {e}"
            logger.error(error_msg, exc_info=True)
            return failure(_provider_error_code(e) or MCPErrorCode.INTERNAL_ERROR, error_msg)

    @mcp.tool()
    async def graph_symbol(
        index_name: str,
        query: str,
        kind: str | None = None,
        language: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Find symbol definitions by qualified/display name substring (IG-02, read-only).

        Args:
            index_name: Name of the index to search
            query: Substring of a qualified or display name
            kind: Optional node kind filter (module, class, function, method, interface, test)
            language: Optional language filter (python, javascript, typescript)
            limit: Maximum results (1-500)

        Returns:
            Versioned symbol records with stable ids and file:start-end evidence
        """
        try:
            logger.info(f"Graph symbol search: index={index_name}, query={query}")
            if not _valid_index_name(index_name):
                return failure(MCPErrorCode.INVALID_INPUT, "Invalid index name")
            operations = GraphOperations(project_path)
            nodes = operations.find_symbols(index_name, query, kind=kind, language=language, limit=limit)
            result = SymbolSearchResult.build(index_name, query, kind, language, nodes)
            return success(result.to_dict())
        except GraphStoreError as e:
            return failure(MCPErrorCode.STORAGE_FAILED, f"Graph store error: {e}")
        except (GraphIndexNotFoundError, GraphNotBuiltError) as e:
            return failure(MCPErrorCode.NOT_FOUND, str(e))
        except ValueError as e:
            return failure(MCPErrorCode.INVALID_INPUT, str(e))
        except Exception as e:
            error_msg = f"Error searching graph symbols: {e}"
            logger.error(error_msg, exc_info=True)
            return failure(_provider_error_code(e) or MCPErrorCode.INTERNAL_ERROR, error_msg)

    @mcp.tool()
    async def graph_neighbors(
        index_name: str,
        symbol_id: str,
        edge_kind: str | None = None,
        direction: str = "both",
        depth: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Traverse the symbol graph around one symbol (IG-02, read-only).

        Args:
            index_name: Name of the index to traverse
            symbol_id: Stable symbol id, or a unique prefix of at least 8 characters
            edge_kind: Optional edge kind filter (contains, imports, calls, inherits, references, tests)
            direction: Traversal direction: in, out, or both
            depth: Traversal depth (1-3)
            limit: Maximum returned nodes (1-500)

        Returns:
            Versioned bounded neighborhood with evidence and confidence per edge
        """
        try:
            logger.info(f"Graph neighbors: index={index_name}, symbol_id={symbol_id[:12]}...")
            if not _valid_index_name(index_name):
                return failure(MCPErrorCode.INVALID_INPUT, "Invalid index name")
            operations = GraphOperations(project_path)
            result = operations.neighbors(
                index_name,
                symbol_id,
                edge_kind=edge_kind,
                direction=direction,
                depth=depth,
                limit=limit,
            )
            envelope = NeighborsResult.build(index_name, symbol_id, direction, depth, limit, result)
            return success(envelope.to_dict())
        except GraphStoreError as e:
            return failure(MCPErrorCode.STORAGE_FAILED, f"Graph store error: {e}")
        except (GraphIndexNotFoundError, GraphNotBuiltError) as e:
            return failure(MCPErrorCode.NOT_FOUND, str(e))
        except ValueError as e:
            return failure(MCPErrorCode.INVALID_INPUT, str(e))
        except Exception as e:
            error_msg = f"Error traversing graph: {e}"
            logger.error(error_msg, exc_info=True)
            return failure(_provider_error_code(e) or MCPErrorCode.INTERNAL_ERROR, error_msg)

    return mcp


def start_mcp_server(project_path: Path | None = None):
    """
    Start the MCP server.

    Args:
        project_path: Optional project path for configuration
    """
    if not MCP_AVAILABLE:
        console.print("[red][X] MCP is not installed[/red]\n")
        console.print("[yellow]Install it with:[/yellow] [cyan]pip install ctxai[mcp][/cyan]\n")
        return

    console.print("[bold blue][*] Starting MCP server...[/bold blue]\n")
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
        console.print(f"[red][X] {error_msg}[/red]\n")
