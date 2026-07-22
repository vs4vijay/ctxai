"""Local web dashboard for inspecting and querying ctxai indexes."""

from __future__ import annotations

from html import escape
from pathlib import Path

from rich.console import Console

from ..index_operations import IndexOperations

try:
    from fasthtml.common import FastHTML, serve

    FASTHTML_AVAILABLE = True
except ImportError:
    FASTHTML_AVAILABLE = False

console = Console(legacy_windows=False)

STYLE = """
body{font:16px system-ui;background:#0f172a;color:#e2e8f0;margin:0}.wrap{max-width:1100px;margin:auto;padding:2rem}
a{color:#7dd3fc}.nav{display:flex;gap:1rem;margin-bottom:2rem}.card{background:#1e293b;border:1px solid #334155;
border-radius:.7rem;padding:1.25rem;margin:1rem 0}table{width:100%;border-collapse:collapse}th,td{text-align:left;
padding:.7rem;border-bottom:1px solid #334155}.ok{color:#6ee7b7}.warn{color:#fcd34d}.bad{color:#fca5a5}
button,.button{background:#0369a1;color:white;border:0;border-radius:.4rem;padding:.55rem .8rem;text-decoration:none}
button.danger{background:#b91c1c}input,select{padding:.6rem;background:#0f172a;color:white;border:1px solid #475569}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#020617;padding:1rem;border-radius:.4rem}
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>{escape(title)} - ctxai</title><style>{STYLE}</style></head><body><main class='wrap'>"
        "<nav class='nav'><a href='/'>Indexes</a><a href='/query'>Query</a></nav>"
        f"{body}</main></body></html>"
    )


def _status(summary) -> str:
    if not summary.healthy:
        return "<span class='bad'>unhealthy</span>"
    if summary.stale:
        return "<span class='warn'>stale</span>"
    return "<span class='ok'>healthy &amp; current</span>"


def create_dashboard_app(project_path: Path | None = None):
    """Build a testable dashboard application backed by shared operations."""
    if not FASTHTML_AVAILABLE:
        raise RuntimeError("FastHTML is not installed; install ctxai[dashboard]")
    app = FastHTML()
    operations = IndexOperations(project_path)

    @app.get("/")
    def home():
        rows = []
        for item in operations.list():
            manifest = item.manifest
            identity = (
                f"schema {manifest.schema_version}; {manifest.embedding_provider}/{manifest.embedding_model}"
                if manifest
                else "manifest unavailable"
            )
            rows.append(
                "<tr>"
                f"<td><a href='/index/{escape(item.name)}'>{escape(item.name)}</a></td>"
                f"<td>{_status(item)}</td><td>{escape(identity)}</td>"
                f"<td>{item.storage_chunks if item.storage_chunks is not None else 'unknown'}</td>"
                f"<td><a class='button' href='/query?index={escape(item.name)}'>Query</a></td></tr>"
            )
        table = (
            "<table><thead><tr><th>Index</th><th>Health / freshness</th><th>Identity</th><th>Chunks</th>"
            "<th>Action</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            if rows
            else "<p>No indexes found. Run <code>ctxai index</code> first.</p>"
        )
        return _page("Indexes", f"<h1>ctxai code intelligence</h1><section class='card'>{table}</section>")

    @app.get("/index/{name}")
    def inspect(name: str):
        try:
            item = operations.inspect(name)
            manifest = item.manifest
            details = ""
            if manifest:
                details = (
                    f"<p>Repository: <code>{escape(manifest.repository_root)}</code></p>"
                    f"<p>Revision: <code>{escape(manifest.repository_revision or 'not recorded')}</code></p>"
                    f"<p>Updated: {escape(manifest.updated_at)} · Files: {manifest.file_count} · "
                    f"Chunks: {manifest.chunk_count}</p>"
                    f"<p>Embedding: {escape(manifest.embedding_provider)}/{escape(manifest.embedding_model)} "
                    f"({manifest.embedding_dimension} dimensions) · Schema: {manifest.schema_version}</p>"
                )
            problems = "".join(f"<li>{escape(problem)}</li>" for problem in item.problems)
            delete = (
                f"<form action='/index/{escape(name)}/delete' method='post'>"
                "<p>Deletion is permanent and only applies to this local index.</p>"
                "<button class='danger' type='submit'>Delete index</button></form>"
            )
            return _page(
                name,
                f"<h1>{escape(name)}</h1><section class='card'><h2>Status: {_status(item)}</h2>{details}"
                f"<ul>{problems}</ul></section><section class='card'>{delete}</section>",
            )
        except Exception as exc:
            return _page("Index error", f"<h1>Index error</h1><p class='bad'>{escape(str(exc))}</p>")

    @app.get("/query")
    def query_page(index: str | None = None):
        options = "".join(
            f"<option value='{escape(item.name)}'{' selected' if item.name == index else ''}>"
            f"{escape(item.name)}</option>"
            for item in operations.list()
            if item.healthy
        )
        form = (
            "<form action='/query/search' method='post'><label>Index <select name='index' required>"
            f"{options}</select></label> <label>Query <input name='query' required></label> "
            "<label>Results <input type='number' name='n_results' value='5' min='1' max='20'></label> "
            "<button type='submit'>Search</button></form>"
        )
        return _page("Query", f"<h1>Grounded code search</h1><section class='card'>{form}</section>")

    @app.post("/query/search")
    def search(index: str, query: str, n_results: int = 5):
        try:
            results = operations.query(index, query, n_results)
            cards = []
            for result in results:
                metadata = result["metadata"]
                location = (
                    f"{metadata.get('file_path', 'unknown')}:"
                    f"{metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}"
                )
                cards.append(
                    f"<article class='card'><h2>{escape(location)}</h2>"
                    f"<p>Similarity: {max(0, 1 - float(result['distance'])):.1%}</p>"
                    f"<pre>{escape(str(result['content'])[:2000])}</pre></article>"
                )
            return _page("Query results", f"<h1>Evidence for “{escape(query)}”</h1>{''.join(cards)}")
        except Exception as exc:
            return _page("Query error", f"<h1>Query error</h1><p class='bad'>{escape(str(exc))}</p>")

    @app.post("/index/{name}/delete")
    def delete(name: str):
        try:
            operations.delete(name)
            return _page("Index deleted", f"<h1>Index deleted</h1><p>{escape(name)} was removed.</p>")
        except Exception as exc:
            return _page("Delete error", f"<h1>Delete error</h1><p class='bad'>{escape(str(exc))}</p>")

    return app


def start_dashboard(
    port: int = 3000,
    project_path: Path | None = None,
    host: str = "127.0.0.1",
    allow_remote: bool = False,
) -> None:
    """Start a localhost-only dashboard unless the caller explicitly changes host."""
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in local_hosts and not allow_remote:
        raise ValueError(
            "Remote dashboard binding is disabled. Pass --allow-remote only on a trusted network; "
            "the dashboard has no authentication or TLS."
        )
    app = create_dashboard_app(project_path)
    console.print(f"[green][OK] Dashboard: http://{host}:{port}[/green]")
    if host not in local_hosts:
        console.print("[yellow]Warning: remote exposure has no authentication; use a trusted network.[/yellow]")
    serve(app=app, host=host, port=port)
