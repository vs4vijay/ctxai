"""Symbol graph inspection commands shared by the CLI (IG-01)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ..graph.operations import GraphError, GraphOperations, GraphStats
from ..graph.store import GraphStoreError, NeighborResult
from ..index_manifest import IndexManifestError


def _operations(project_path: Path | None) -> GraphOperations:
    return GraphOperations(project_path=project_path)


def _resolve_name(project_path: Path | None, index_name: str | None) -> str:
    return GraphOperations.resolve_index_name(project_path, index_name)


def graph_stats(index_name: str | None, project_path: Path | None, as_json: bool) -> None:
    """Print graph stats for an index (human table or versioned JSON).

    Args:
        index_name: Optional index name (config default when omitted).
        project_path: Optional project root.
        as_json: Emit the versioned JSON envelope instead of a table.

    Raises:
        typer.Exit: On invalid input or unavailable graph data.
    """
    import json

    from rich.console import Console
    from rich.table import Table

    operations = _operations(project_path)
    try:
        name = _resolve_name(project_path, index_name)
        stats: GraphStats = operations.stats(name)
    except (GraphError, GraphStoreError, IndexManifestError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if as_json:
        envelope: dict[str, Any] = {
            "schema_version": 1,
            "index": name,
            "health": {
                "status": stats.health.status,
                "problems": list(stats.health.problems),
                "diagnostic": stats.health.diagnostic,
            },
            "graph": stats.metadata.to_dict() if stats.metadata else None,
        }
        typer.echo(json.dumps(envelope, indent=2, sort_keys=True))
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
    import json

    from rich.console import Console
    from rich.table import Table

    operations = _operations(project_path)
    try:
        name = _resolve_name(project_path, index_name)
        nodes = operations.find_symbols(name, query, kind=kind, language=language, limit=limit)
    except (GraphError, GraphStoreError, IndexManifestError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if as_json:
        envelope = {
            "schema_version": 1,
            "index": name,
            "query": query,
            "count": len(nodes),
            "symbols": [{**node.to_dict(), "evidence": node.evidence()} for node in nodes],
        }
        typer.echo(json.dumps(envelope, indent=2, sort_keys=True))
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
    import json

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
    except (GraphError, GraphStoreError, IndexManifestError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if as_json:
        envelope = {
            "schema_version": 1,
            "index": name,
            "symbol_id": symbol_id,
            "edge_kind": edge_kind,
            "direction": direction,
            "depth": depth,
            "limit": limit,
            "truncated": result.truncated,
            "nodes": [{**node.to_dict(), "evidence": node.evidence()} for node in result.nodes],
            "edges": [{**edge.to_dict(), "evidence": edge.evidence()} for edge in result.edges],
        }
        typer.echo(json.dumps(envelope, indent=2, sort_keys=True))
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
