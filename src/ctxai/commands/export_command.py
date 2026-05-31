"""`ctxai export` CLI commands — text, html, json, stats."""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer
from rich.console import Console

from ctxai.export.config import ExportConfig
from ctxai.export.html_codemap import HtmlCodemap
from ctxai.export.json_export import JsonExporter
from ctxai.export.repo_to_text import RepoTextExporter

console = Console()
export_app = typer.Typer(name="export", help="Export the repository in various formats")


def _make_config(
    include: list[str] | None,
    exclude: list[str] | None,
    max_files: int,
    max_size: int,
    output_format: str,
) -> ExportConfig:
    return ExportConfig(
        max_files=max_files,
        max_total_size_mb=max_size,
        include_patterns=list(include) if include else [],
        exclude_patterns=list(exclude) if exclude else [],
        output_format=output_format,
    )


@export_app.command("text")
def export_text(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    output: Path = typer.Option(Path("repo.md"), "--output", "-o"),
    include: list[str] | None = typer.Option(None, "--include", "-i"),
    exclude: list[str] | None = typer.Option(None, "--exclude", "-e"),
    max_files: int = typer.Option(500, "--max-files"),
    max_size: int = typer.Option(50, "--max-size"),
    output_format: str = typer.Option("markdown", "--format", help="text|markdown|xml"),
    with_tree: bool = typer.Option(True, "--with-tree/--no-tree"),
):
    """Export repository to a single text/markdown/xml file."""
    cfg = _make_config(include, exclude, max_files, max_size, output_format)
    exporter = RepoTextExporter(path, cfg)
    written = exporter.export(output, with_tree=with_tree)
    console.print(f"[green]Wrote {written}[/green]")


@export_app.command("html")
def export_html(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    output: Path = typer.Option(Path("codemap.html"), "--output", "-o"),
    include: list[str] | None = typer.Option(None, "--include", "-i"),
    exclude: list[str] | None = typer.Option(None, "--exclude", "-e"),
    max_files: int = typer.Option(500, "--max-files"),
    max_size: int = typer.Option(50, "--max-size"),
    theme: str = typer.Option("dark", "--theme", help="dark|light"),
):
    """Generate an interactive HTML code map."""
    cfg = _make_config(include, exclude, max_files, max_size, "html")
    codemap = HtmlCodemap(path, cfg, theme=theme)
    written = codemap.export(output)
    console.print(f"[green]Wrote {written}[/green]")


@export_app.command("json")
def export_json(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    output: Path = typer.Option(Path("repo.json"), "--output", "-o"),
    include: list[str] | None = typer.Option(None, "--include", "-i"),
    exclude: list[str] | None = typer.Option(None, "--exclude", "-e"),
    max_files: int = typer.Option(500, "--max-files"),
    max_size: int = typer.Option(50, "--max-size"),
    indent: int = typer.Option(2, "--indent"),
):
    """Export repository as a structured JSON document."""
    cfg = _make_config(include, exclude, max_files, max_size, "json")
    exporter = JsonExporter(path, cfg)
    written = exporter.export(output, indent=indent)
    console.print(f"[green]Wrote {written}[/green]")


@export_app.command("stats")
def export_stats(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    include: list[str] | None = typer.Option(None, "--include", "-i"),
    exclude: list[str] | None = typer.Option(None, "--exclude", "-e"),
    max_files: int = typer.Option(10_000, "--max-files"),
    max_size: int = typer.Option(1_000, "--max-size"),
):
    """Show statistics for a repository without writing any output file."""
    cfg = _make_config(include, exclude, max_files, max_size, "markdown")
    summary = RepoTextExporter(path, cfg).collect()
    console.print(
        _json.dumps(
            {
                "name": summary.name,
                "files": len(summary.files),
                "total_size_bytes": summary.total_size,
                "total_lines": summary.total_lines,
                "languages": summary.languages,
                "truncated": summary.truncated,
            },
            indent=2,
        )
    )
