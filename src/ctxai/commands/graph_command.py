"""Symbol graph inspection commands shared by the CLI (IG-01/IG-02).

All reads go through :class:`~ctxai.graph.operations.GraphOperations`; this
module never touches graph storage directly. JSON output uses the versioned
DTOs from :mod:`ctxai.graph.dto`, so CLI, MCP, and dashboard projections
agree on identity, counts, confidence, and relationships.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..graph.adapters import capabilities_payload
from ..graph.dto import (
    CapabilitiesResult,
    GraphStatsResult,
    NeighborsResult,
    SymbolSearchResult,
)
from ..graph.operations import (
    GraphError,
    GraphOperations,
    GraphStats,
)
from ..graph.store import GraphStoreError, NeighborResult
from ..index_manifest import IndexManifestError

_JSON_ERRORS = (GraphError, GraphStoreError, IndexManifestError)
_QUERY_ERRORS = (*_JSON_ERRORS, ValueError)


def _operations(project_path: Path | None) -> GraphOperations:
    return GraphOperations(project_path=project_path)


def _resolve_name(project_path: Path | None, index_name: str | None) -> str:
    return GraphOperations.resolve_index_name(project_path, index_name)


def _fail(error: Exception) -> typer.Exit:
    """Report one deterministic one-line error and exit non-zero."""
    typer.echo(f"Error: {error}", err=True)
    return typer.Exit(code=1)


def _emit_json(payload: dict) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def graph_stats(index_name: str | None, project_path: Path | None, as_json: bool) -> None:
    """Print graph stats for an index (human table or versioned JSON).

    Args:
        index_name: Optional index name (config default when omitted).
        project_path: Optional project root.
        as_json: Emit the versioned JSON envelope instead of a table.

    Raises:
        typer.Exit: On invalid input or unavailable graph data.
    """
    from rich.console import Console
    from rich.table import Table

    operations = _operations(project_path)
    try:
        name = _resolve_name(project_path, index_name)
        stats: GraphStats = operations.stats(name)
    except _JSON_ERRORS as error:
        raise _fail(error) from error

    if as_json:
        result = GraphStatsResult.build(name, stats, capabilities_payload())
        _emit_json(result.to_dict())
        return

    console = Console()
    console.print(f"\n[bold blue]Symbol graph for index '{name}'[/bold blue]")
    health = stats.health
    if health.status == "healthy" and stats.metadata is not None:
        metadata = stats.metadata
        total_edges = metadata.total_edges
        unresolved_rate = (metadata.unresolved_edges / total_edges) if total_edges else 0.0
        table = Table(show_header=True, header_style="bold")
        table.add_column("Property")
        table.add_column("Value")
        table.add_row("Status", "healthy")
        table.add_row("Schema version", str(metadata.schema_version))
        table.add_row("Extractor / resolver", f"{metadata.extractor_version} / {metadata.resolver_version}")
        table.add_row("Supported languages", ", ".join(metadata.supported_languages))
        table.add_row("Generation", str(metadata.generation))
        table.add_row("Built at", metadata.built_at)
        for kind in sorted(metadata.node_counts):
            table.add_row(f"Nodes: {kind}", str(metadata.node_counts[kind]))
        table.add_row("Nodes: total", str(metadata.total_nodes))
        for kind in sorted(metadata.edge_counts):
            table.add_row(f"Edges: {kind}", str(metadata.edge_counts[kind]))
        table.add_row("Edges: total", str(total_edges))
        table.add_row("Unresolved edges", f"{metadata.unresolved_edges} ({unresolved_rate:.1%})")
        console.print(table)
    elif health.status == "missing":
        console.print(f"[yellow]Graph: {health.diagnostic}[/yellow]")
    else:
        console.print("[red]Graph problems detected:[/red]")
        for problem in health.problems:
            console.print(f"[red]- {problem}[/red]")
        raise typer.Exit(code=1)


def graph_symbol(
    query: str,
    index_name: str | None,
    kind: str | None,
    language: str | None,
    limit: int,
    project_path: Path | None,
    as_json: bool,
) -> None:
    """Search symbols by qualified/display name substring.

    Args:
        query: Substring of the qualified or display name.
        index_name: Optional index name (config default when omitted).
        kind: Optional node kind filter.
        language: Optional language filter.
        limit: Maximum results (1..500).
        project_path: Optional project root.
        as_json: Emit the versioned JSON envelope instead of a table.

    Raises:
        typer.Exit: On invalid input or unavailable graph data.
    """
    from rich.console import Console
    from rich.table import Table

    operations = _operations(project_path)
    try:
        name = _resolve_name(project_path, index_name)
        nodes = operations.find_symbols(name, query, kind=kind, language=language, limit=limit)
    except _QUERY_ERRORS as error:
        raise _fail(error) from error

    if as_json:
        result = SymbolSearchResult.build(name, query, kind, language, nodes)
        _emit_json(result.to_dict())
        return

    console = Console()
    if not nodes:
        console.print(f"[yellow]No symbols matching {query!r} in index '{name}'[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Qualified name")
    table.add_column("Kind")
    table.add_column("Evidence")
    table.add_column("Symbol ID")
    for node in nodes:
        table.add_row(node.qualified_name, node.kind, node.evidence(), node.id)
    console.print(table)
    console.print("[dim]Use `ctxai graph neighbors SYMBOL_ID` to inspect relationships[/dim]")


def graph_neighbors(
    symbol_id: str,
    index_name: str | None,
    edge_kind: str | None,
    direction: str,
    depth: int,
    limit: int,
    project_path: Path | None,
    as_json: bool,
) -> None:
    """Show the bounded neighborhood of one symbol.

    Args:
        symbol_id: Stable symbol id (or unique 8+ char prefix).
        index_name: Optional index name (config default when omitted).
        edge_kind: Optional edge kind filter.
        direction: ``in``, ``out``, or ``both``.
        depth: Traversal depth (1..3).
        limit: Maximum returned nodes (1..500).
        project_path: Optional project root.
        as_json: Emit the versioned JSON envelope instead of a table.

    Raises:
        typer.Exit: On invalid input or unavailable graph data.
    """
    from rich.console import Console
    from rich.table import Table

    operations = _operations(project_path)
    try:
        name = _resolve_name(project_path, index_name)
        result: NeighborResult = operations.neighbors(
            name,
            symbol_id,
            edge_kind=edge_kind,
            direction=direction,
            depth=depth,
            limit=limit,
        )
    except _QUERY_ERRORS as error:
        raise _fail(error) from error

    if as_json:
        envelope = NeighborsResult.build(name, symbol_id, direction, depth, limit, result)
        _emit_json(envelope.to_dict())
        return

    console = Console()
    if result.start is None:
        console.print(f"[yellow]No symbol matching id {symbol_id!r} in index '{name}'[/yellow]")
        return
    start = result.start
    console.print(f"\n[bold blue]Neighborhood of {start.qualified_name}[/bold blue] [dim]({start.evidence()})[/dim]")
    edge_table = Table(show_header=True, header_style="bold")
    edge_table.add_column("Kind")
    edge_table.add_column("Direction")
    edge_table.add_column("From")
    edge_table.add_column("To")
    edge_table.add_column("Evidence")
    edge_table.add_column("Confidence")
    node_names = {node.id: node.qualified_name for node in result.nodes}
    for edge in result.edges:
        origin = "out" if edge.source_id == start.id else "in"
        if edge.target_id is not None:
            target_display = node_names.get(edge.target_id, edge.target_id)
        else:
            target_display = f"[unresolved] {edge.target_text}"
        edge_table.add_row(
            edge.kind,
            origin,
            node_names.get(edge.source_id, edge.source_id),
            target_display,
            edge.evidence(),
            edge.confidence,
        )
    console.print(edge_table)
    node_table = Table(show_header=True, header_style="bold")
    node_table.add_column("Reached symbol")
    node_table.add_column("Kind")
    node_table.add_column("Evidence")
    for node in result.nodes:
        node_table.add_row(node.qualified_name, node.kind, node.evidence())
    console.print(node_table)
    if result.truncated:
        console.print("[yellow]Result truncated by --limit; raise the limit or narrow the traversal[/yellow]")


def graph_capabilities(index_name: str | None, project_path: Path | None, as_json: bool) -> None:
    """Print the language support matrix, with per-index observations (IG-02).

    Without an index the static per-language matrix is shown. With an index,
    per-index observations are added: languages present with node counts,
    unresolved edges by kind, and the count of indexed files in languages
    without an adapter (they stay searchable as ordinary chunks).

    Args:
        index_name: Optional index name (config default when omitted).
        project_path: Optional project root.
        as_json: Emit the versioned JSON envelope instead of a table.

    Raises:
        typer.Exit: On invalid input or an unknown index.
    """
    from rich.console import Console
    from rich.table import Table

    operations = _operations(project_path)
    try:
        name = _resolve_name(project_path, index_name) if index_name else None
        result: CapabilitiesResult = operations.capabilities(name)
    except _JSON_ERRORS as error:
        raise _fail(error) from error

    if as_json:
        _emit_json(result.to_dict())
        return

    console = Console()
    console.print("\n[bold blue]Graph language capabilities[/bold blue]")
    table = Table(show_header=True, header_style="bold")
    for column in ("Language", "Supported", "Adapter", "Extensions", "Node kinds", "Edge kinds"):
        table.add_column(column)
    for entry in result.languages:
        table.add_row(
            entry["language"],
            "yes" if entry["supported"] else "no",
            entry["adapter_version"] or "-",
            ", ".join(entry["file_extensions"]) or "-",
            ", ".join(entry["node_kinds"]) or "-",
            ", ".join(entry["edge_kinds"]) or "-",
        )
    console.print(table)
    for entry in result.languages:
        if not entry["supported"]:
            console.print(
                f"[yellow]{entry['language']}: no adapter — {'; '.join(entry['unsupported_constructs'])}[/yellow]"
            )
        elif entry["unsupported_constructs"]:
            console.print(
                f"[dim]{entry['language']}: never resolved (no fabricated edges): "
                + "; ".join(entry["unsupported_constructs"])
                + "[/dim]"
            )
    observations = result.index
    if observations is not None:
        console.print(f"\n[bold blue]Index '{observations.index}'[/bold blue]")
        counts = observations.languages_present
        if counts:
            for language in sorted(counts):
                console.print(f"Graph nodes in {language}: {counts[language]}")
        else:
            console.print("[yellow]No graph nodes recorded for this index yet[/yellow]")
        for kind in sorted(observations.unresolved_edges_by_kind):
            console.print(f"Unresolved {kind} edges: {observations.unresolved_edges_by_kind[kind]}")
        if observations.unsupported_file_count:
            console.print(
                f"[yellow]{observations.unsupported_file_count} of {observations.total_file_count} indexed files are"
                " in languages without a graph adapter; they remain searchable as ordinary chunks[/yellow]"
            )
