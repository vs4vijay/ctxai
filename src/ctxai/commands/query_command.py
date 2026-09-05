"""
Query command implementation.
Search indexed codebase using natural language through the shared hybrid
retrieval service (IG-03): semantic, lexical, symbol, and repository-map
candidate generators plus optional graph expansion, fused deterministically
and assembled within the configured token budget.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..agent.sessions import redact_secrets
from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..index_manifest import IndexManifest
from ..repository_context import EvidenceResult, GraphExpansionSettings, retrieve_evidence
from ..utils import get_indexes_dir

console = Console(legacy_windows=False)
stderr_console = Console(file=sys.stderr, legacy_windows=False)

SYNTAX_LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "cpp": "cpp",
    "c": "c",
    "go": "go",
    "rust": "rust",
    "ruby": "ruby",
}


def _syntax_language(file_path: str) -> str:
    """Guess a syntax-highlighting language from a file extension.

    Args:
        file_path: The chunk file path.

    Returns:
        A Rich syntax language name (``text`` when unknown).
    """
    suffix = Path(file_path).suffix.lower().lstrip(".")
    return SYNTAX_LANGUAGE_MAP.get(suffix, "text")


def _render_evidence(result: EvidenceResult, query: str, show_content: bool) -> None:
    """Render retrieved evidence panels (normal, concise output).

    Args:
        result: The shared-service retrieval result.
        query: The original query (echoed in the header).
        show_content: Whether to render code content.
    """
    context = result.context
    if not context.items:
        console.print("[yellow]No results found[/yellow]\n")
        return

    console.print(f"[bold green][OK] Found {len(context.items)} result(s)[/bold green]\n")
    for i, item in enumerate(context.items, 1):
        header = f"Result {i}: {Path(item.file_path).name}"
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Key", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("File", str(item.file_path))
        info_table.add_row("Lines", f"{item.start_line}-{item.end_line}")
        info_table.add_row("Type", item.chunk_type)
        info_table.add_row("Language", _syntax_language(item.file_path))
        info_table.add_row("[*] Score", f"{item.score:.4f}")

        console.print(Panel(info_table, title=header, border_style="blue"))

        if item.graph_evidence is not None:
            evidence = item.graph_evidence
            console.print(
                f"  [cyan][G] Graph:[/cyan] {escape(evidence.path)}"
                f" ({evidence.confidence}, depth {evidence.depth}, +{evidence.contribution:.4f})"
            )

        if show_content:
            display_content = item.content
            if len(item.content) > 1000:
                display_content = item.content[:1000] + "\n... (truncated)"
            syntax = Syntax(
                display_content,
                _syntax_language(item.file_path),
                theme="monokai",
                line_numbers=True,
                start_line=item.start_line,
            )
            console.print(syntax)

        console.print()

    console.print("[dim]" + "-" * 80 + "[/dim]")
    console.print(f"[dim]Assembled {len(context.items)} evidence block(s); ~{context.estimated_tokens} tokens[/dim]\n")


def _render_explain(result: EvidenceResult) -> None:
    """Render the --explain report (per-query terminal output only).

    Shows base ranks, per-component fusion contributions, graph paths with
    confidence, exclusions (duplicates/budget), and context-budget decisions.

    Args:
        result: The shared-service retrieval result carrying the explanation.
    """
    explain = result.explain
    if explain is None:
        return

    console.print(f"\n[bold blue]Why this context was selected[/bold blue] [dim]({explain.query!r})[/dim]")

    generator_table = Table(title="Candidate generators", show_header=True)
    generator_table.add_column("Generator")
    generator_table.add_column("Candidates")
    for component, count in sorted(explain.component_counts.items()):
        generator_table.add_row(component, str(count))
    console.print(generator_table)

    context = result.context
    detail_table = Table(title="Selected evidence and fusion contribution", show_header=True)
    detail_table.add_column("Rank")
    detail_table.add_column("Citation")
    detail_table.add_column("Base score")
    detail_table.add_column("Components")
    detail_table.add_column("Graph path")
    for rank, item in enumerate(context.items, 1):
        components = explain.components.get(item.id, {})
        component_text = ", ".join(f"{name} {value:.4f}" for name, value in sorted(components.items()))
        graph_text = ""
        if item.graph_evidence is not None:
            graph = item.graph_evidence
            graph_text = f"{escape(graph.path)} ({graph.confidence}, depth {graph.depth}, +{graph.contribution:.4f})"
        detail_table.add_row(
            str(rank),
            item.citation,
            f"{item.score:.4f}",
            component_text or "-",
            graph_text or "-",
        )
    console.print(detail_table)

    if explain.seeds:
        seed_table = Table(title="Graph expansion seeds (top base hits)", show_header=True)
        seed_table.add_column("Base rank")
        seed_table.add_column("Seed symbol")
        seed_table.add_column("Seed citation")
        seed_table.add_column("Base score")
        for seed in explain.seeds:
            seed_table.add_row(
                str(seed["base_rank"]), str(seed["symbol"]), str(seed["citation"]), f"{seed['base_score']:.4f}"
            )
        console.print(seed_table)

    budget_table = Table(title="Context-budget decisions", show_header=True)
    budget_table.add_column("Citation")
    budget_table.add_column("Decision")
    for citation, reason in context.excluded:
        budget_table.add_row(citation, reason)
    budget_table.add_row("(assembled)", f"{len(context.items)} selected, ~{context.estimated_tokens} tokens")
    console.print(budget_table)

    if explain.diagnostics:
        console.print("[yellow]Diagnostics:[/yellow]")
        for diagnostic in explain.diagnostics:
            console.print(f"  [yellow]- {diagnostic}[/yellow]")
    console.print()


def query_codebase(
    index_name: str | None,
    query: str,
    project_path: Path | None = None,
    n_results: int = 5,
    show_content: bool = True,
    explain: bool = False,
    graph: bool | None = None,
    trace: bool | None = None,
):
    """
    Query an indexed codebase using natural language.

    Routes through the shared retrieval service so CLI, agent, MCP, and
    dashboard behave identically.

    Args:
        index_name: Name of the index to query (uses config default if None)
        query: Natural language query
        project_path: Optional project path (uses cwd if not provided)
        n_results: Number of results to return
        show_content: Whether to show full code content
        explain: Show why each item was selected (ranks, graph paths, budget)
        trace: Persist a retrieval trace per the configured mode (RE-02)
        graph: Graph expansion override; True requires a healthy graph,
            False disables expansion, None uses the configured default
    """
    # Load configuration
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Determine index name: use provided or fall back to config
    if index_name is None:
        index_name = config.index_name
        if index_name is None:
            console.print("[red][X][/red] No index name provided and none configured\n")
            console.print("[yellow]Tip:[/yellow] Either:\n")
            console.print('  1. Provide an index name: [cyan]ctxai query <index_name> "your query"[/cyan]\n')
            console.print("  2. Or set a default: [cyan]ctxai config --set index.name <index_name>[/cyan]\n")
            return
        console.print(f"[dim]Using configured index: {index_name}[/dim]")

    console.print(f"\n[bold blue][?] Searching index '{index_name}'...[/bold blue]\n")
    # Secrets pasted into a query must not survive into the terminal either.
    console.print(f"[dim]Query: {escape(redact_secrets(query))}[/dim]\n")

    try:
        # Create embedding provider
        console.print(f"[dim]Using embedding provider: {config.embedding.provider}[/dim]")

        embeddings_generator = EmbeddingsFactory.create(config.embedding)

        # Keep the storage-path boundary check (and its patch seam) explicit.
        indexes_dir = get_indexes_dir(project_path)
        storage_path = indexes_dir / index_name

        if not storage_path.exists():
            console.print(f"[red][X][/red] Index '{index_name}' not found at {storage_path}\n")
            console.print("[yellow]Tip:[/yellow] Run [cyan]ctxai index[/cyan] first to create an index\n")
            return

        # Embedding identity is still checked at this boundary so a mismatched
        # provider/model never silently queries an index.
        manifest = IndexManifest.load_optional(storage_path)
        if manifest is not None:
            configured_model = config.embedding.model or getattr(embeddings_generator, "model", None) or "default"
            if (
                manifest.embedding_provider != config.embedding.provider
                or manifest.embedding_model != str(configured_model)
                or manifest.embedding_dimension != embeddings_generator.get_dimension()
            ):
                raise ValueError(
                    "The configured embedding provider/model does not match this index. "
                    "Use the recorded manifest settings or rebuild the index."
                )

        console.print("[cyan]Running hybrid retrieval...[/cyan]\n")
        settings = GraphExpansionSettings.from_config(
            config.retrieval,
            enabled=graph,
            required=bool(graph),
        )
        from ..retrieval_traces import privacy_warning, resolve_trace_settings

        trace_settings = resolve_trace_settings(config.retrieval, enabled=trace)
        warning = privacy_warning(trace_settings.mode)
        if warning:
            # Bind at call time: the captured stderr of an import-time Console
            # is closed under CLI test runners.
            Console(file=sys.stderr, legacy_windows=False).print(f"[yellow]{warning}[/yellow]")
        result = retrieve_evidence(
            project_path or Path.cwd(),
            query,
            embedding_provider=embeddings_generator,
            index_name=index_name,
            limit=n_results,
            token_budget=config.retrieval.token_budget,
            graph=settings,
            explain=explain,
            trace=trace_settings,
        )

        if result.graph_diagnostic:
            # Visible fallback diagnostic: config-enabled expansion could not
            # run, so this result is base retrieval only.
            stderr_console.print(f"[yellow]WARNING: {result.graph_diagnostic}[/yellow]")

        _render_evidence(result, query, show_content)
        if explain:
            _render_explain(result)

    except Exception as e:
        console.print(f"[red][X] Error querying index: {e}[/red]\n")
        raise
