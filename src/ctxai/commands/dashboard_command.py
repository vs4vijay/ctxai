"""Local web dashboard for inspecting and querying ctxai indexes (VS-09/IG-02).

Graph views are read-only and served through the shared
:class:`ctxai.graph.operations.GraphOperations` service; all rendered values
are HTML-escaped and every result is bounded before work begins.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from rich.console import Console

from ..embeddings import EmbeddingsFactory
from ..graph.model import MAX_SYMBOL_QUERY_LENGTH, MAX_TRAVERSAL_DEPTH
from ..graph.operations import GraphOperations
from ..index_operations import IndexOperations

try:
    from fasthtml.common import FastHTML, serve

    FASTHTML_AVAILABLE = True
except ImportError:
    FASTHTML_AVAILABLE = False

console = Console(legacy_windows=False)

# Dashboard result bounds: stricter than the CLI caps, applied before work.
DASHBOARD_SYMBOL_LIMIT = 100
DASHBOARD_GRAPH_LIMIT = 100

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


def _graph_links(name: str) -> str:
    safe = escape(name)
    return (
        f"<p><a class='button' href='/index/{safe}/graph'>Graph summary</a> "
        f"<a class='button' href='/index/{safe}/graph/symbols'>Browse symbols</a></p>"
    )


def _symbol_search_form(
    name: str,
    query: str,
    kind: str | None,
    language: str | None,
    limit: int,
) -> str:
    """Render the bounded, escaped symbol search form."""
    node_kinds = ("module", "class", "function", "method", "interface", "test")
    languages = ("javascript", "python", "typescript")
    kind_options = "<option value=''>any kind</option>" + "".join(
        f"<option value='{value}'{' selected' if kind == value else ''}>{value}</option>" for value in node_kinds
    )
    language_options = "<option value=''>any language</option>" + "".join(
        f"<option value='{value}'{' selected' if language == value else ''}>{value}</option>" for value in languages
    )
    query_label = (
        f"<label>Query <input name='query' value='{escape(query)}'"
        f" maxlength='{MAX_SYMBOL_QUERY_LENGTH}' required></label>"
    )
    limit_label = (
        f"<label>Limit <input type='number' name='limit' value='{limit}'"
        f" min='1' max='{DASHBOARD_SYMBOL_LIMIT}'></label>"
    )
    return (
        f"<form method='get' action='/index/{escape(name)}/graph/symbols'>"
        f"{query_label} "
        f"<label>Kind <select name='kind'>{kind_options}</select></label> "
        f"<label>Language <select name='language'>{language_options}</select></label> "
        f"{limit_label} "
        "<button type='submit'>Search</button></form>"
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
    graph_operations = GraphOperations(project_path)

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
                f"<ul>{problems}</ul></section><section class='card'>{_graph_links(name)}{delete}</section>",
            )
        except Exception as exc:
            return _page("Index error", f"<h1>Index error</h1><p class='bad'>{escape(str(exc))}</p>")

    @app.get("/index/{name}/graph")
    def graph_summary(name: str):
        try:
            stats = graph_operations.stats(name)
            capabilities = graph_operations.capabilities(name)
            health = stats.health
            observations = capabilities.index
            sections = [f"<h1>Symbol graph: {escape(name)}</h1>"]
            if health.status == "healthy" and stats.metadata is not None:
                metadata = stats.metadata
                rows = "".join(
                    f"<tr><td>{escape(kind)}</td><td>{count}</td></tr>"
                    for kind, count in sorted(metadata.node_counts.items())
                )
                edge_rows = "".join(
                    f"<tr><td>{escape(kind)}</td><td>{count}</td></tr>"
                    for kind, count in sorted(metadata.edge_counts.items())
                )
                languages = ", ".join(sorted(capabilities.index.languages_present)) if observations else ""
                sections.append(
                    "<section class='card'>"
                    f"<p>Status: <span class='ok'>healthy</span> · Generation {metadata.generation} · "
                    f"Schema {metadata.schema_version}</p>"
                    f"<p>Adapters: {escape(metadata.extractor_version)} · Languages: {escape(languages)}</p>"
                    f"<p>{metadata.total_nodes} nodes · {metadata.total_edges} edges · "
                    f"{metadata.unresolved_edges} unresolved</p>"
                    f"<h2>Nodes by kind</h2><table><thead><tr><th>Kind</th><th>Count</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table>"
                    f"<h2>Edges by kind</h2><table><thead><tr><th>Kind</th><th>Count</th></tr></thead>"
                    f"<tbody>{edge_rows}</tbody></table></section>"
                )
            elif health.status == "missing":
                sections.append(
                    "<section class='card'><p class='warn'>Graph data has not been built for this index; "
                    "run <code>ctxai index</code> to generate it.</p></section>"
                )
            else:
                problems = "".join(f"<li class='bad'>{escape(problem)}</li>" for problem in health.problems)
                sections.append(
                    f"<section class='card'><h2>Graph status: {escape(health.status)}</h2><ul>{problems}</ul></section>"
                )
            if observations is not None and observations.unsupported_file_count:
                sections.append(
                    "<section class='card'><p class='warn'>"
                    f"{observations.unsupported_file_count} of {observations.total_file_count} indexed files are in "
                    "languages without a graph adapter; they remain searchable as ordinary chunks.</p></section>"
                )
            sections.append("<section class='card'>" + _graph_links(name) + "</section>")
            return _page(f"Graph: {name}", "".join(sections))
        except Exception as exc:
            return _page("Graph error", f"<h1>Graph error</h1><p class='bad'>{escape(str(exc))}</p>")

    @app.get("/index/{name}/graph/symbols")
    def graph_symbols(name: str, query: str = "", kind: str = "", language: str = "", limit: int = 20):
        try:
            text = (query or "").strip()[:MAX_SYMBOL_QUERY_LENGTH]
            bounded_limit = max(1, min(int(limit or 20), DASHBOARD_SYMBOL_LIMIT))
            kind_filter = kind or None
            language_filter = language or None
            if not text:
                form = _symbol_search_form(name, text, kind_filter, language_filter, bounded_limit)
                return _page(f"Symbols: {name}", f"<h1>Symbols: {escape(name)}</h1>{form}")
            nodes = graph_operations.find_symbols(
                name,
                text,
                kind=kind_filter,
                language=language_filter,
                limit=bounded_limit,
            )
            rows = "".join(
                "<tr>"
                f"<td><a href='/index/{escape(name)}/graph/node/{escape(node.id)}'>"
                f"{escape(node.qualified_name)}</a></td>"
                f"<td>{escape(node.kind)}</td><td>{escape(node.language)}</td><td>{escape(node.evidence())}</td>"
                f"<td><code>{escape(node.id[:12])}</code></td></tr>"
                for node in nodes
            )
            table = (
                "<table><thead><tr><th>Qualified name</th><th>Kind</th><th>Language</th><th>Evidence</th>"
                "<th>Symbol ID</th></tr></thead><tbody>" + rows + "</tbody></table>"
                if rows
                else f"<p class='warn'>No symbols matching {escape(text)}.</p>"
            )
            form = _symbol_search_form(name, text, kind_filter, language_filter, bounded_limit)
            return _page(
                f"Symbols: {name}", f"<h1>Symbols: {escape(name)}</h1>{form}<section class='card'>{table}</section>"
            )
        except Exception as exc:
            return _page("Symbol search error", f"<h1>Symbol search error</h1><p class='bad'>{escape(str(exc))}</p>")

    @app.get("/index/{name}/graph/node/{node_id}")
    def graph_node(name: str, node_id: str, direction: str = "both", depth: int = 1, limit: int = 50):
        try:
            bounded_depth = max(1, min(int(depth or 1), MAX_TRAVERSAL_DEPTH))
            bounded_limit = max(1, min(int(limit or 50), DASHBOARD_GRAPH_LIMIT))
            result = graph_operations.neighbors(
                name,
                node_id,
                direction=direction if direction in ("in", "out", "both") else "both",
                depth=bounded_depth,
                limit=bounded_limit,
            )
            if result.start is None:
                return _page(
                    "Symbol not found",
                    f"<h1>Symbol not found</h1><p class='warn'>No symbol matches id {escape(node_id)}.</p>"
                    f"<p><a href='/index/{escape(name)}/graph/symbols'>Back to symbols</a></p>",
                )
            start = result.start
            node_names = {node.id: node.qualified_name for node in result.nodes}
            edge_rows = []
            for edge in result.edges:
                origin = "outgoing" if edge.source_id == start.id else "incoming"
                if edge.target_id is not None:
                    target = node_names.get(edge.target_id, edge.target_id)
                    other_id = edge.target_id if edge.source_id == start.id else edge.source_id
                else:
                    target = f"{edge.target_text} (unresolved)"
                    other_id = None
                related = (
                    f"<a href='/index/{escape(name)}/graph/node/{escape(other_id)}'>{escape(target)}</a>"
                    if other_id
                    else escape(target)
                )
                edge_rows.append(
                    "<tr>"
                    f"<td>{escape(edge.kind)}</td><td>{origin}</td><td>{related}</td>"
                    f"<td>{escape(edge.evidence())}</td><td>{escape(edge.confidence)}</td></tr>"
                )
            relationship_table = (
                "<table><thead><tr><th>Kind</th><th>Direction</th><th>Related symbol</th><th>Evidence</th>"
                "<th>Confidence</th></tr></thead><tbody>" + "".join(edge_rows) + "</tbody></table>"
                if edge_rows
                else "<p class='warn'>No relationships within the traversal bounds.</p>"
            )
            reached_rows = "".join(
                "<tr>"
                f"<td><a href='/index/{escape(name)}/graph/node/{escape(node.id)}'>"
                f"{escape(node.qualified_name)}</a></td>"
                f"<td>{escape(node.kind)}</td><td>{escape(node.evidence())}</td></tr>"
                for node in result.nodes
                if node.id != start.id
            )
            reached_table = (
                "<table><thead><tr><th>Reached symbol</th><th>Kind</th><th>Evidence</th></tr></thead>"
                f"<tbody>{reached_rows}</tbody></table>"
                if reached_rows
                else ""
            )
            controls = (
                f"<form method='get' action='/index/{escape(name)}/graph/node/{escape(start.id)}'>"
                f"<label>Direction <select name='direction'>"
                + "".join(
                    f"<option value='{value}'{' selected' if value == direction else ''}>{value}</option>"
                    for value in ("in", "out", "both")
                )
                + f"</select></label> <label>Depth <input type='number' name='depth' value='{bounded_depth}' "
                f"min='1' max='{MAX_TRAVERSAL_DEPTH}'></label> "
                f"<label>Limit <input type='number' name='limit' value='{bounded_limit}' min='1' "
                f"max='{DASHBOARD_GRAPH_LIMIT}'></label> <button type='submit'>Traverse</button></form>"
            )
            truncated = (
                "<p class='warn'>Result truncated by the limit; raise it or narrow the traversal.</p>"
                if result.truncated
                else ""
            )
            body = (
                f"<h1>{escape(start.qualified_name)}</h1>"
                "<section class='card'>"
                f"<p><strong>Kind:</strong> {escape(start.kind)} · "
                f"<strong>Language:</strong> {escape(start.language)} · "
                f"<strong>Visibility:</strong> {escape(start.visibility)}</p>"
                f"<p><strong>Evidence:</strong> <code>{escape(start.evidence())}</code> · "
                f"<strong>Symbol ID:</strong> <code>{escape(start.id)}</code></p>"
                f"<p><strong>Adapter:</strong> {escape(start.adapter_version)}</p>"
                f"{controls}</section>"
                f"<section class='card'><h2>Relationships</h2>{relationship_table}</section>"
                f"<section class='card'><h2>Reached symbols</h2>{reached_table}{truncated}</section>"
                "<section class='card'>" + _graph_links(name) + "</section>"
            )
            return _page(f"{start.qualified_name} - graph", body)
        except Exception as exc:
            return _page("Graph node error", f"<h1>Graph node error</h1><p class='bad'>{escape(str(exc))}</p>")

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
            # Index-name boundary check stays here; the retrieval itself goes
            # through the shared service (IG-03) so the dashboard sees the
            # same fusion, budgets, and optional graph expansion as the CLI,
            # agent, and MCP.
            operations.validate_name(index)
            from ..config import ConfigManager
            from ..repository_context import GraphExpansionSettings, retrieve_evidence

            config = ConfigManager(project_path).load()
            provider = EmbeddingsFactory.create(config.embedding)
            settings = GraphExpansionSettings.from_config(config.retrieval, required=False)
            evidence = retrieve_evidence(
                project_path or Path.cwd(),
                query,
                embedding_provider=provider,
                index_name=index,
                limit=max(1, min(n_results, 20)),
                token_budget=config.retrieval.token_budget,
                graph=settings,
            )
            cards = []
            for item in evidence.context.items:
                distance = evidence.semantic_distances.get(item.id)
                relevance = (
                    f"Similarity: {max(0, 1 - float(distance)):.1%}"
                    if distance is not None
                    else f"Score: {item.score:.4f}"
                )
                graph_line = (
                    f"<p class='ok'>Graph: {escape(item.graph_evidence.path)} "
                    f"({escape(item.graph_evidence.confidence)})</p>"
                    if item.graph_evidence is not None
                    else ""
                )
                cards.append(
                    f"<article class='card'><h2>{escape(item.citation)}</h2>"
                    f"<p>{escape(relevance)} · {escape(item.chunk_type)}</p>{graph_line}"
                    f"<pre>{escape(str(item.content)[:2000])}</pre></article>"
                )
            diagnostic = ""
            if evidence.graph_diagnostic:
                diagnostic = f"<p class='warn'>{escape(evidence.graph_diagnostic)}</p>"
            return _page(
                "Query results",
                f"<h1>Evidence for “{escape(query)}”</h1>{diagnostic}{''.join(cards)}",
            )
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
