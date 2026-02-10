"""
Semantic code search tool using ctxai's vector store.
"""

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema


class SemanticSearchTool(BaseTool):
    """Tool for semantic code search using indexed codebases."""

    def __init__(self, project_path: Path = None):
        super().__init__()
        self.project_path = project_path

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Search indexed codebase using natural language queries. Returns relevant code chunks with metadata.",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Natural language search query (e.g., 'authentication functions', 'error handling')",
                    required=True
                ),
                ToolParameter(
                    name="index_name",
                    type=ToolParameterType.STRING,
                    description="Name of the index to search (optional, uses default if not specified)",
                    required=False
                ),
                ToolParameter(
                    name="n_results",
                    type=ToolParameterType.INTEGER,
                    description="Number of results to return (default: 5, max: 20)",
                    required=False,
                    default=5
                ),
            ]
        )

    async def execute(self, query: str, index_name: str = None, n_results: int = 5) -> dict[str, Any]:
        try:
            # Import ctxai components
            from ctxai.config import ConfigManager
            from ctxai.embeddings import EmbeddingsFactory
            from ctxai.utils import get_ctxai_home
            from ctxai.vector_store import VectorStore

            # Load config
            config_manager = ConfigManager(self.project_path)
            config = config_manager.load()

            # Determine index name
            if not index_name:
                ctxai_home = get_ctxai_home(self.project_path)
                indexes_dir = ctxai_home / "indexes"
                if not indexes_dir.exists():
                    return {
                        "success": False,
                        "result": None,
                        "error": "No indexes found. Please index a codebase first using 'ctxai index'."
                    }
                # Use first available index
                indexes = [d.name for d in indexes_dir.iterdir() if d.is_dir()]
                if not indexes:
                    return {
                        "success": False,
                        "result": None,
                        "error": "No indexes found. Please index a codebase first."
                    }
                index_name = indexes[0]

            # Limit n_results
            n_results = min(n_results, 20)

            # Initialize embedding provider
            embedding_provider = EmbeddingsFactory.create(config.embedding)

            # Initialize vector store
            vector_store = VectorStore(index_name, embedding_provider)

            # Generate query embedding
            query_embedding = embedding_provider.embed([query])[0]

            # Search
            results = vector_store.search(query_embedding, n_results=n_results)

            if not results:
                return {
                    "success": True,
                    "result": "No matching code found.",
                    "error": None,
                    "metadata": {"matches": 0}
                }

            # Format results
            formatted = []
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                formatted.append(
                    f"[{i}] {metadata.get('file_path', 'unknown')}:{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}\n"
                    f"    Type: {metadata.get('chunk_type', 'unknown')} | Score: {result.get('distance', 0):.3f}\n"
                    f"    {result.get('document', '')[:200]}..."
                )

            result_text = "\n\n".join(formatted)

            return {
                "success": True,
                "result": result_text,
                "error": None,
                "metadata": {
                    "index_name": index_name,
                    "query": query,
                    "matches": len(results),
                }
            }

        except ImportError as e:
            return {
                "success": False,
                "result": None,
                "error": f"Failed to import ctxai components: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Search failed: {str(e)}"
            }
