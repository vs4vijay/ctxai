"""Grounded hybrid repository search tool."""

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema


class SemanticSearchTool(BaseTool):
    """Search the current repository's matching persistent index."""

    def __init__(self, project_path: Path | None = None, embedding_provider=None):
        super().__init__()
        self.project_path = (project_path or Path.cwd()).resolve()
        self.embedding_provider = embedding_provider

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Hybrid semantic, lexical, symbol, and repository-map search with file/line evidence.",
            parameters=[
                ToolParameter("query", ToolParameterType.STRING, "Natural-language or symbol query", required=True),
                ToolParameter(
                    "index_name", ToolParameterType.STRING, "Matching repository index override", required=False
                ),
                ToolParameter(
                    "n_results",
                    ToolParameterType.INTEGER,
                    "Results to return (default 5, max 20)",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    "token_budget",
                    ToolParameterType.INTEGER,
                    "Maximum approximate context tokens",
                    required=False,
                    default=2000,
                ),
                ToolParameter(
                    "debug",
                    ToolParameterType.BOOLEAN,
                    "Explain why context was selected",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(
        self,
        query: str,
        index_name: str | None = None,
        n_results: int = 5,
        token_budget: int = 2000,
        debug: bool = False,
    ) -> dict[str, Any]:
        try:
            from ctxai.config import ConfigManager
            from ctxai.embeddings import EmbeddingsFactory
            from ctxai.repository_context import ContextAssembler, HybridRetriever

            provider = self.embedding_provider
            if provider is None:
                provider = EmbeddingsFactory.create(ConfigManager(self.project_path).load().embedding)
            retriever = HybridRetriever(self.project_path, provider, index_name=index_name)
            ranked = retriever.retrieve(query, limit=min(max(1, n_results), 20), debug=debug)
            context = ContextAssembler(token_budget=max(1, token_budget), debug=debug).assemble(
                retriever.index_name, ranked
            )
            return {
                "success": True,
                "result": context.text or "No matching code found.",
                "error": None,
                "metadata": {
                    "index_name": context.index_name,
                    "query": query,
                    "matches": len(context.items),
                    "estimated_tokens": context.estimated_tokens,
                    "citations": [item.citation for item in context.items],
                },
            }
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "error": f"Search failed: {exc}",
                "error_type": type(exc).__name__,
            }
