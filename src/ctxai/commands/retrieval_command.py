"""`ctxai retrieval runs` command: inspect local retrieval traces (RE-02).

All operations are read-only or delete-only against the resolved trace
directory inside the project; nothing is uploaded and no retrieval state is
mutated. Corrupt trace files are skipped with diagnostics instead of failing
the listing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ..retrieval_traces import (
    TRACE_SCHEMA_VERSION,
    TraceCorruptError,
    TraceNotFoundError,
    delete_all_trace_runs,
    delete_trace_run,
    list_trace_runs,
    read_run_payload,
    resolve_trace_dir,
)

console = Console(legacy_windows=False)

_RUN_ID_PREFIX_LEN = 12


def _project_path(project_path: Path | None) -> Path:
    """Resolve the project root for trace operations.

    Args:
        project_path: Explicit project root (defaults to the current directory).

    Returns:
        The resolved project root.
    """
    return project_path or Path.cwd()


def list_traces(
    project_path: Path | None = None,
    limit: int | None = None,
    index_name: str | None = None,
    as_json: bool = False,
) -> int:
    """List stored retrieval traces, newest first.

    Args:
        project_path: Project root (defaults to the current directory).
        limit: Maximum number of summaries to show.
        index_name: Only include runs of this index.
        as_json: Print the versioned JSON envelope instead of a table.

    Returns:
        The CLI exit code.
    """
    root = _project_path(project_path)
    summaries, corrupt = list_trace_runs(resolve_trace_dir(root, None), limit=limit, index_name=index_name)
    if as_json:
        import json

        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "runs": [summary.to_dict() for summary in summaries],
            "corrupt": corrupt,
        }
        console.print_json(json.dumps(payload))
        return 0

    table = Table(title=f"Retrieval traces ({len(summaries)})")
    table.add_column("Run")
    table.add_column("Time")
    table.add_column("Status")
    table.add_column("Mode")
    table.add_column("Index")
    table.add_column("Cands", justify="right")
    table.add_column("Sel", justify="right")
    table.add_column("ms", justify="right")
    for summary in summaries:
        table.add_row(
            summary.run_id[:_RUN_ID_PREFIX_LEN],
            (summary.timestamp or "")[11:19],
            summary.status,
            summary.mode,
            summary.index_name or "-",
            str(summary.candidate_count),
            str(summary.selected_count),
            f"{summary.total_latency_ms:.1f}",
        )
    console.print(table)
    for item in corrupt:
        console.print(f"[yellow]skipped corrupt trace: {item}[/yellow]")
    return 0


def show_trace(run_id: str, project_path: Path | None = None, as_json: bool = False) -> int:
    """Show one retrieval trace.

    Args:
        run_id: The run identifier (full id or unambiguous prefix).
        project_path: Project root (defaults to the current directory).
        as_json: Print the exact on-disk JSON payload.

    Returns:
        The CLI exit code.
    """
    root = _project_path(project_path)
    try:
        payload: dict[str, Any] = read_run_payload(run_id, resolve_trace_dir(root, None))
    except ValueError as error:
        console.print(f"[red]Error: Invalid run id ({error})[/red]")
        return 1
    except TraceNotFoundError:
        console.print(f"[red]Error: trace run '{run_id[:_RUN_ID_PREFIX_LEN]}' not found[/red]")
        return 1
    except TraceCorruptError as error:
        console.print(f"[red]Error: trace run is corrupt: {error}[/red]")
        return 1
    if as_json:
        import json

        console.print_json(json.dumps({"schema_version": TRACE_SCHEMA_VERSION, "run": payload}))
        return 0
    console.print(f"[bold blue]Retrieval run[/bold blue] {payload['run_id']}")
    console.print(
        f"[dim]mode: {payload['mode']} | status: {payload['status']} | timestamp: {payload['timestamp']}[/dim]"
    )
    index = payload.get("index") or {}
    console.print(f"index: {index.get('name')} | provider: {index.get('embedding_provider')}")
    query_block = payload.get("query") or {}
    query_text = query_block.get("text") or f"(hash) {query_block.get('hash')}"
    console.print(f"query: {query_text}")
    console.print(
        f"candidates: {payload.get('candidate_count')} | selected: {payload.get('selected_count')} | "
        f"tokens: {payload.get('estimated_tokens')} | latency: {payload.get('total_latency_ms')} ms"
    )
    stage_table = Table(title="Stage timings (ms)", show_header=True)
    stage_table.add_column("Stage")
    stage_table.add_column("Duration")
    for stage, duration in sorted((payload.get("stage_timings_ms") or {}).items()):
        stage_table.add_row(stage, f"{duration:.2f}")
    console.print(stage_table)

    candidate_table = Table(title="Selected evidence", show_header=True)
    candidate_table.add_column("Rank")
    candidate_table.add_column("Citation")
    candidate_table.add_column("Decision")
    candidate_table.add_column("Graph path")
    for candidate in payload.get("candidates") or []:
        if candidate.get("decision") != "selected":
            continue
        candidate_table.add_row(
            str(candidate.get("final_rank")),
            str(candidate.get("citation")),
            str(candidate.get("decision")),
            str(candidate.get("graph_path") or "-"),
        )
    console.print(candidate_table)
    for item in payload.get("errors") or []:
        console.print(f"[red]error: {item}[/red]")
    for item in payload.get("diagnostics") or []:
        console.print(f"[dim]diagnostic: {item}[/dim]")
    return 0


def delete_trace(
    run_id: str | None,
    project_path: Path | None = None,
    delete_all: bool = False,
) -> int:
    """Delete one trace run, or every trace with explicit confirmation.

    Args:
        run_id: The run identifier to delete (required without ``--all``).
        project_path: Project root (defaults to the current directory).
        delete_all: Delete every stored trace (asks for confirmation).

    Returns:
        The CLI exit code.
    """
    root = _project_path(project_path)
    if delete_all:
        import typer

        if not typer.confirm("Delete ALL stored retrieval traces?"):
            console.print("[yellow]Aborted[/yellow]")
            return 1
        removed = delete_all_trace_runs(resolve_trace_dir(root, None))
        console.print(f"[green][OK][/green] Deleted {removed} trace run(s)")
        return 0
    if not run_id:
        console.print("[red]Error: provide RUN_ID or --all[/red]")
        return 1
    try:
        removed_path = delete_trace_run(run_id, resolve_trace_dir(root, None))
    except ValueError as error:
        console.print(f"[red]Error: Invalid run id ({error})[/red]")
        return 1
    except TraceNotFoundError:
        console.print(f"[red]Error: trace run '{run_id[:_RUN_ID_PREFIX_LEN]}' not found[/red]")
        return 1
    console.print(f"[green][OK][/green] Deleted {removed_path.name}")
    return 0
