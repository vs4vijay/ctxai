"""Grounded hybrid repository search tool."""

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema


class SemanticSearchTool(BaseTool):
    """Search the current repository's matching persistent index.

    Routes through the shared retrieval service (IG-03) so the agent sees the
    same fusion, graph expansion, budget enforcement, and diagnostics as the
    CLI, MCP, and dashboard surfaces.
    """

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
            from ctxai.repository_context import GraphExpansionSettings, retrieve_evidence

            config = ConfigManager(self.project_path).load()
            provider = self.embedding_provider
            if provider is None:
                provider = EmbeddingsFactory.create(config.embedding)
            # Config-driven graph expansion (disabled by default); when
            # enabled but unavailable the tool falls back with a visible
            # diagnostic instead of failing the agent's turn.
            settings = GraphExpansionSettings.from_config(config.retrieval, required=False)
            evidence = retrieve_evidence(
                self.project_path,
                query,
                embedding_provider=provider,
                index_name=index_name,
                limit=min(max(1, n_results), 20),
                token_budget=max(1, token_budget),
                graph=settings,
                explain=debug,
            )
            context = evidence.context
            metadata: dict[str, Any] = {
                "index_name": context.index_name,
                "query": query,
                "matches": len(context.items),
                "estimated_tokens": context.estimated_tokens,
                "citations": [item.citation for item in context.items],
                "graph_expanded": sum(1 for item in context.items if item.graph_evidence is not None),
            }
            if evidence.graph_diagnostic:
                metadata["graph_diagnostic"] = evidence.graph_diagnostic
            if debug:
                components = evidence.explain.components if evidence.explain is not None else {}
                metadata["explanation"] = [
                    {
                        "citation": item.citation,
                        "reasons": list(item.reasons),
                        "components": {name: value for name, value in sorted(components.get(item.id, {}).items())},
                        "graph_path": item.graph_evidence.path if item.graph_evidence is not None else None,
                    }
                    for item in context.items
                ]
            return {
                "success": True,
                "result": context.text or "No matching code found.",
                "error": None,
                "metadata": metadata,
            }
        except Exception as exc:
            return {
                "success": False,
                "result": None,
                "error": f"Search failed: {exc}",
                "error_type": type(exc).__name__,
            }
