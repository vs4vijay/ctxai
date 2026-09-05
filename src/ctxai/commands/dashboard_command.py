"""Local web dashboard for inspecting and querying ctxai indexes (VS-09/IG-02).

Graph views are read-only and served through the shared
:class:`ctxai.graph.operations.GraphOperations` service; evaluation views
(RE-03) read immutable artifacts exclusively through
:class:`ctxai.evals.operations.EvaluationOperations`; all rendered values
are HTML-escaped and every result is bounded before work begins.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from rich.console import Console

from ..embeddings import EmbeddingsFactory
from ..evals.operations import EvaluationOperations, RunComparison
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
DASHBOARD_EVALUATION_LIMIT = 100
DASHBOARD_CASE_LIMIT = 100
DASHBOARD_PREVIEW_CITATIONS = 3

# Presentation order for aggregate metric tables (mirrors the CLI report).
EVALUATION_METRIC_ORDER = (
    "successful_query_rate",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    "evidence_precision@5",
    "selected_token_mean",
    "selected_token_p95",
    "duplicate_token_ratio",
    "graph_contribution_rate",
    "latency_p50_ms",
    "latency_p95_ms",
)

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
        "<nav class='nav'><a href='/'>Indexes</a><a href='/query'>Query</a>"
        "<a href='/evaluations'>Evaluations</a></nav>"
        f"{body}</main></body></html>"
    )


def _eval_status_chip(status: str) -> str:
    """Render an escaped, color-coded status chip for run/gate statuses."""
    css = {"complete": "ok", "pass": "ok", "partial": "warn", "regression": "bad", "incompatible": "bad"}.get(
        status, "warn"
    )
    return f"<span class='{css}'>{escape(status)}</span>"


def _metric_cell(entry: object) -> str:
    """Render one metric-value entry honoring explicit availability."""
    if not isinstance(entry, dict):
        return "<span class='warn'>malformed entry</span>"
    if entry.get("available"):
        value = entry.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{float(value):.4f}"
        return escape(str(value))
    reason = escape(str(entry.get("reason") or "unavailable"))
    return f"<span class='warn'>unavailable ({reason})</span>"


def _delta_text(value: object) -> str:
    """Format a delta number with an explicit sign (or a dash)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{float(value):+.4f}"


def _trend_cell(trend: object) -> str:
    """Render a trend chip (improved/regressed/flat)."""
    if not trend:
        return "-"
    css = {"improved": "ok", "regressed": "bad", "flat": ""}.get(str(trend), "")
    label = escape(str(trend))
    return f"<span class='{css}'>{label}</span>" if css else label


def _gate_status_cell(status: object) -> str:
    """Render a gate status chip (pass/regression/unavailable/reported)."""
    return _eval_status_chip(str(status)) if status in ("pass", "regression", "unavailable") else escape(str(status))


def _render_comparison_gates(comparison: RunComparison) -> str:
    """Render the dimension-grouped metric delta tables of a comparison."""
    dimension_titles = (
        ("correctness", "Correctness"),
        ("quality", "Quality"),
        ("efficiency", "Context efficiency"),
        ("timing", "Latency (noisy; reported, never gated)"),
        ("other", "Other metrics"),
    )
    sections = []
    for dimension, title in dimension_titles:
        gates = [gate for gate in comparison.gates if gate.dimension == dimension]
        if not gates:
            continue

        def _metric_cell(value: float | None) -> str:
            """Render a possibly-unavailable metric value.

            Args:
                value: The metric value, or None when unavailable.

            Returns:
                The formatted cell text.
            """
            return "-" if value is None else f"{value:.4f}"

        rows = "".join(
            "<tr>"
            f"<td>{escape(gate.cohort)}</td><td>{escape(gate.metric)}</td>"
            f"<td>{_metric_cell(gate.baseline)}</td>"
            f"<td>{_metric_cell(gate.current)}</td>"
            f"<td>{_delta_text(gate.delta)}</td>"
            f"<td>{gate.absolute_tolerance}/{gate.relative_tolerance}</td>"
            f"<td>{_gate_status_cell(gate.status)}</td>"
            f"<td>{_trend_cell(gate.trend)}</td>"
            "</tr>"
            for gate in gates
        )
        sections.append(
            f"<section class='card'><h2>{title} ({escape(comparison.status)})</h2>"
            "<table><thead><tr><th>Cohort</th><th>Metric</th><th>Baseline</th><th>Candidate</th>"
            "<th>Delta</th><th>Tolerance (abs/rel)</th><th>Status</th><th>Trend</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></section>"
        )
    return "".join(sections)


def _render_comparison_cases(comparison: RunComparison) -> str:
    """Render the changed-cases section of a comparison."""
    changed = [delta for delta in comparison.case_deltas if delta.kind != "unchanged"]
    if not changed:
        return "<section class='card'><h2>Changed cases</h2><p>No case outcomes changed.</p></section>"
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(delta.case_id)}</code></td><td>{escape(delta.cohort)}</td>"
        f"<td>{_case_kind_cell(delta.kind)}</td>"
        f"<td>{delta.baseline_first_relevant_rank or '-'} -> {delta.candidate_first_relevant_rank or '-'}</td>"
        f"<td>{delta.baseline_tokens} -> {delta.candidate_tokens} ({delta.delta_tokens:+d})</td>"
        f"<td>{_latency_pair(delta.baseline_latency_ms, delta.candidate_latency_ms)}</td>"
        "</tr>"
        for delta in changed
    )
    summary_lines = ""
    if comparison.newly_passing:
        summary_lines += f"<p class='ok'>Newly passing: {escape(', '.join(comparison.newly_passing))}</p>"
    if comparison.newly_failing:
        summary_lines += f"<p class='bad'>Newly failing: {escape(', '.join(comparison.newly_failing))}</p>"
    return (
        "<section class='card'><h2>Changed cases</h2>"
        "<table><thead><tr><th>Case</th><th>Cohort</th><th>Change</th><th>First rank (base -> cand)</th>"
        "<th>Tokens (base -> cand)</th><th>Latency ms (base -> cand)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{summary_lines}</section>"
    )


def _case_kind_cell(kind: str) -> str:
    """Render a case-change kind with pass/fail coloring."""
    css = {"newly_passing": "ok", "newly_failing": "bad", "improved": "ok", "worsened": "bad"}.get(kind, "warn")
    return f"<span class='{css}'>{escape(kind)}</span>"


def _latency_pair(baseline: float | None, candidate: float | None) -> str:
    """Format a baseline -> candidate latency pair (reported, never gated)."""
    if baseline is None or candidate is None:
        return "-"
    return f"{baseline:.1f} -> {candidate:.1f}"


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
    eval_operations = EvaluationOperations(project_path)

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

    def _format_components(components: dict | None) -> str:
        """Render a candidate's component contributions for the funnel table.

        Args:
            components: Component name to contribution mapping (may be None).

        Returns:
            A compact single-line rendering, or "-" when empty.
        """
        if not components:
            return "-"
        return ", ".join(f"{name} {value:.4f}" for name, value in sorted(components.items()))

    @app.get("/retrieval-runs")
    def retrieval_runs_page(index: str = "", status: str = ""):
        """List local retrieval traces (RE-02): filters for index and status."""
        from ..retrieval_traces import list_trace_runs, resolve_trace_dir

        index_filter = index or None
        status_filter = status or None
        summaries, corrupt = list_trace_runs(
            resolve_trace_dir(project_path or Path.cwd()),
            limit=100,
            index_name=index_filter,
            status=status_filter,
        )
        rows = "".join(
            "<tr>"
            f"<td><a href='/retrieval-runs/{escape(summary.run_id)}'>{escape(summary.run_id[:12])}</a></td>"
            f"<td>{escape(summary.timestamp)}</td>"
            f"<td>{escape(summary.status)}</td><td>{escape(summary.mode)}</td>"
            f"<td>{escape(summary.index_name or '-')}</td>"
            f"<td>{summary.candidate_count}</td><td>{summary.selected_count}</td>"
            f"<td>{summary.total_latency_ms:.1f}</td></tr>"
            for summary in summaries
        )
        table = (
            "<table><thead><tr><th>Run</th><th>Timestamp</th><th>Status</th><th>Mode</th>"
            "<th>Index</th><th>Candidates</th><th>Selected</th><th>Latency (ms)</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            if rows
            else "<p class='warn'>No traces stored. Tracing is off by default; enable it with "
            "retrieval.trace_mode or query --trace.</p>"
        )
        corrupt_lines = "".join(f"<p class='bad'>{escape(item)}</p>" for item in corrupt)
        body = (
            "<h1>Retrieval traces</h1>"
            "<section class='card'>"
            "<form method='get' action='/retrieval-runs'>"
            "<label>Index <input type='text' name='index' value='"
            f"{escape(index_filter or '')}"
            "'></label> <label>Status <select name='status'>"
            + "".join(
                f"<option value='{value}'{' selected' if value == status_filter else ''}>{value or 'any'}</option>"
                for value in ("", "ok", "error")
            )
            + "</select></label> <button type='submit'>Filter</button></form>"
            f"{table}{corrupt_lines}</section>"
        )
        return _page("Retrieval traces", body)

    @app.get("/retrieval-runs/{run_id}")
    def retrieval_run_detail(run_id: str):
        """Show one retrieval trace: the ranking funnel and stage timings."""
        from ..retrieval_traces import (
            TraceCorruptError,
            TraceNotFoundError,
            read_run_payload,
            resolve_trace_dir,
        )

        try:
            payload = read_run_payload(run_id, resolve_trace_dir(project_path or Path.cwd()))
        except (TraceNotFoundError, TraceCorruptError, ValueError) as exc:
            return _page("Retrieval trace", f"<h1>Trace unavailable</h1><p class='bad'>{escape(str(exc))}</p>")
        stage_rows = "".join(
            f"<tr><td>{escape(str(stage))}</td><td>{value:.2f}</td></tr>"
            for stage, value in sorted((payload.get("stage_timings_ms") or {}).items())
        )
        candidate_rows = "".join(
            "<tr>"
            f"<td>{escape(str(candidate.get('final_rank')))}</td>"
            f"<td><code>{escape(str(candidate.get('citation')))}</code></td>"
            f"<td>{escape(str(candidate.get('decision')))}</td>"
            f"<td>{escape(str(candidate.get('graph_path') or '-'))}</td>"
            f"<td>{escape(_format_components(candidate.get('components')))}</td>"
            "</tr>"
            for candidate in payload.get("candidates") or []
        )
        body = (
            f"<h1>Retrieval run {escape(str(payload.get('run_id'))[:12])}</h1>"
            "<section class='card'>"
            f"<p><strong>Index:</strong> {escape(str((payload.get('index') or {}).get('name')))} · "
            f"<strong>Mode:</strong> {escape(str(payload.get('mode')))} · "
            f"<strong>Status:</strong> {escape(str(payload.get('status')))} · "
            f"<strong>Timestamp:</strong> {escape(str(payload.get('timestamp')))}</p>"
            f"<p><strong>Candidates:</strong> {payload.get('candidate_count')} · "
            f"<strong>Selected:</strong> {payload.get('selected_count')} · "
            f"<strong>Tokens:</strong> {payload.get('estimated_tokens')} · "
            f"<strong>Latency:</strong> {payload.get('total_latency_ms')} ms</p>"
            "</section>"
            "<section class='card'><h2>Stage timings (ms)</h2><table><thead><tr><th>Stage</th>"
            "<th>Duration</th></tr></thead><tbody>"
            + (stage_rows or "<tr><td colspan='2'>none recorded</td></tr>")
            + "</tbody></table></section>"
            "<section class='card'><h2>Ranking funnel</h2><table><thead><tr><th>Rank</th>"
            "<th>Citation</th><th>Decision</th><th>Graph path</th><th>Components</th></tr></thead><tbody>"
            + (candidate_rows or "<tr><td colspan='5'>no candidates</td></tr>")
            + "</tbody></table></section>"
        )
        return _page("Retrieval trace detail", body)

    @app.get("/evaluations")
    def evaluations_page():
        """List stored evaluation runs (RE-03), newest first, bounded."""
        summaries, corrupt = eval_operations.list_runs(limit=DASHBOARD_EVALUATION_LIMIT)
        rows = "".join(
            "<tr>"
            f"<td><a href='/evaluations/{escape(summary.run_id)}'>{escape(summary.run_id[:16])}</a></td>"
            f"<td>{escape(summary.created_at)}</td>"
            f"<td><code>{escape(summary.benchmark_fingerprint[:12])}</code></td>"
            f"<td>{_eval_status_chip(summary.status)}</td>"
            f"<td>{'graph' if summary.graph_enabled else 'no-graph'}</td>"
            f"<td>{escape(summary.index_name or '-')}</td>"
            f"<td>{summary.case_count}</td>"
            f"<td>{_eval_status_chip(summary.comparison_status) if summary.comparison_status else '-'}</td>"
            "</tr>"
            for summary in summaries
        )
        table = (
            "<table><thead><tr><th>Run</th><th>Created</th><th>Benchmark fingerprint</th><th>Status</th>"
            "<th>Mode</th><th>Index</th><th>Cases</th><th>Baseline comparison</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            if rows
            else "<p class='warn'>No evaluation artifacts stored. Run <code>ctxai eval retrieval</code> first.</p>"
        )
        compare_form = (
            "<form method='get' action='/evaluations/compare'>"
            "<label>Baseline <input name='baseline' required></label> "
            "<label>Candidate <input name='candidate' required></label> "
            "<button type='submit'>Compare</button></form>"
        )
        corrupt_lines = "".join(f"<p class='bad'>{escape(item)}</p>" for item in corrupt)
        return _page(
            "Evaluations",
            f"<h1>Evaluation runs</h1><section class='card'>{compare_form}{table}</section>{corrupt_lines}",
        )

    @app.get("/evaluations/compare")
    def evaluations_compare(baseline: str = "", candidate: str = ""):
        """Compare two stored evaluation runs (aggregate, cohort, and case deltas)."""
        if not baseline or not candidate:
            return _page(
                "Evaluation comparison",
                "<h1>Evaluation comparison</h1>"
                "<p class='warn'>Provide both run ids: "
                "<code>/evaluations/compare?baseline=ID&amp;candidate=ID</code></p>",
            )
        try:
            comparison = eval_operations.compare_runs(baseline, candidate)
        except Exception as exc:
            return _page("Comparison error", f"<h1>Comparison error</h1><p class='bad'>{escape(str(exc))}</p>")
        head = (
            "<h1>Evaluation comparison</h1>"
            f"<p><strong>Baseline:</strong> <code>{escape(str(comparison.baseline.get('run_id') or baseline))}</code>"
            " · <strong>Candidate:</strong> "
            f"<code>{escape(str(comparison.candidate.get('run_id') or candidate))}</code> · "
            f"Status: {_eval_status_chip(comparison.status)}</p>"
        )
        if not comparison.compatible:
            banner = (
                "<section class='card'><h2 class='bad'>Incompatible artifacts</h2>"
                "<p>The two artifacts cannot be compared as equivalent:</p><ul>"
                + "".join(f"<li class='bad'>{escape(item)}</li>" for item in comparison.incompatibilities)
                + "</ul>"
                + "".join(
                    f"<p class='warn'>Rebuild/rerun action: {escape(action)}</p>" for action in comparison.actions
                )
                + "</section>"
            )
            return _page("Evaluation comparison", head + banner)
        return _page(
            "Evaluation comparison", head + _render_comparison_gates(comparison) + _render_comparison_cases(comparison)
        )

    @app.get("/evaluations/{run_id}")
    def evaluation_run_page(run_id: str):
        """Show one evaluation run: aggregates, cohorts, regressions, and cases."""
        try:
            payload = eval_operations.read_run(run_id)
        except Exception as exc:
            return _page("Evaluation error", f"<h1>Evaluation error</h1><p class='bad'>{escape(str(exc))}</p>")
        benchmark = payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
        configuration = payload.get("configuration") if isinstance(payload.get("configuration"), dict) else {}
        index = payload.get("index") if isinstance(payload.get("index"), dict) else {}
        aggregates = payload.get("aggregates") if isinstance(payload.get("aggregates"), dict) else {}
        overall = aggregates.get("overall") if isinstance(aggregates.get("overall"), dict) else {}
        overall_metrics = overall.get("metrics") if isinstance(overall.get("metrics"), dict) else {}
        intervals = overall.get("confidence_intervals") if isinstance(overall.get("confidence_intervals"), dict) else {}
        by_cohort = aggregates.get("by_cohort") if isinstance(aggregates.get("by_cohort"), dict) else {}
        graph_block = (
            configuration.get("graph_expansion") if isinstance(configuration.get("graph_expansion"), dict) else {}
        )

        identity = (
            "<section class='card'>"
            f"<p><strong>Run:</strong> <code>{escape(str(payload.get('run_id') or run_id))}</code> · "
            f"<strong>Created:</strong> {escape(str(payload.get('created_at')))} · "
            f"Status: {_eval_status_chip(str(payload.get('status')))}</p>"
            f"<p><strong>Benchmark:</strong> {escape(str(benchmark.get('name') or '-'))} "
            f"(fingerprint <code>{escape(str(benchmark.get('fingerprint') or '-'))[:12]}</code>) · "
            f"Configuration fingerprint: <code>{escape(str(configuration.get('fingerprint') or '-'))[:12]}</code></p>"
            f"<p><strong>Index:</strong> {escape(str(index.get('name') or '-'))} · "
            f"Mode: {'graph expansion' if graph_block.get('enabled') else 'no graph expansion'}</p>"
            "</section>"
        )

        metric_names = [name for name in EVALUATION_METRIC_ORDER if name in overall_metrics]
        metric_names += sorted(name for name in overall_metrics if name not in EVALUATION_METRIC_ORDER)
        cohort_columns = sorted(by_cohort)
        metric_rows = ""
        for name in metric_names:
            row = f"<tr><td>{escape(name)}</td><td>{_metric_cell(overall_metrics.get(name))}</td>"
            for cohort in cohort_columns:
                cohort_metrics = by_cohort[cohort].get("metrics") if isinstance(by_cohort[cohort], dict) else {}
                row += f"<td>{_metric_cell(cohort_metrics.get(name))}</td>"
            row += "</tr>"
            metric_rows += row
        metrics_table = (
            "<section class='card'><h2>Aggregate metrics</h2>"
            "<table><thead><tr><th>Metric</th><th>Overall</th>"
            + "".join(f"<th>{escape(cohort)}</th>" for cohort in cohort_columns)
            + "</tr></thead><tbody>"
            + (metric_rows or "<tr><td colspan='2'>no metrics recorded</td></tr>")
            + "</tbody></table></section>"
        )

        interval_rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{bound[0]:.4f}</td><td>{bound[1]:.4f}</td></tr>"
            for name, bound in sorted(intervals.items())
            if isinstance(bound, list) and len(bound) == 2
        )
        intervals_section = (
            f"<section class='card'><h2>Bootstrap 95% confidence intervals (overall)</h2>"
            "<table><thead><tr><th>Metric</th><th>Low</th><th>High</th></tr></thead>"
            f"<tbody>{interval_rows}</tbody></table></section>"
            if interval_rows
            else ""
        )

        comparison_block = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else None
        if comparison_block is None:
            regressions_section = (
                "<section class='card'><h2>Baseline comparison</h2>"
                "<p class='warn'>This run has no embedded baseline comparison; compare it against another "
                "run from the evaluation list.</p></section>"
            )
        elif not comparison_block.get("compatible"):
            items = "".join(
                f"<li class='bad'>{escape(str(item))}</li>" for item in comparison_block.get("incompatibilities", [])
            )
            regressions_section = (
                "<section class='card'><h2>Baseline comparison</h2>"
                f"<p class='bad'>The embedded baseline was incompatible:</p><ul>{items}</ul></section>"
            )
        else:
            regressions = [
                gate
                for gate in comparison_block.get("gates", [])
                if isinstance(gate, dict) and gate.get("status") == "regression"
            ]
            regressions.sort(key=lambda gate: -abs(gate.get("delta") or 0.0))
            worst_rows = "".join(
                "<tr>"
                f"<td>{escape(str(gate.get('cohort')))}</td><td>{escape(str(gate.get('metric')))}</td>"
                f"<td>{_delta_text(gate.get('baseline'))}</td><td>{_delta_text(gate.get('current'))}</td>"
                f"<td>{_delta_text(gate.get('delta'))}</td>"
                "</tr>"
                for gate in regressions[:10]
            )
            body = (
                "<table><thead><tr><th>Cohort</th><th>Metric</th><th>Baseline</th><th>Current</th>"
                "<th>Delta</th></tr></thead><tbody>"
                + (worst_rows or "<tr><td colspan='5'>no gate regressed</td></tr>")
                + "</tbody></table>"
            )
            regressions_section = f"<section class='card'><h2>Worst regressions vs baseline</h2>{body}</section>"

        runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
        case_rows = ""
        for run in runs[:DASHBOARD_CASE_LIMIT]:
            case = run if isinstance(run, dict) else {}
            candidates = case.get("candidates") if isinstance(case.get("candidates"), list) else []
            selected = [c for c in candidates if isinstance(c, dict) and c.get("decision") == "selected"]
            citations = ", ".join(str(item.get("citation")) for item in selected[:DASHBOARD_PREVIEW_CITATIONS])
            graph_count = sum(1 for item in candidates if isinstance(item, dict) and item.get("graph_path"))
            latency_values = []
            latency_block = case.get("latency") if isinstance(case.get("latency"), dict) else {}
            raw_values = latency_block.get("values_ms")
            if isinstance(raw_values, list):
                latency_values = [
                    float(v) for v in raw_values if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
            latency_mean = f"{sum(latency_values) / len(latency_values):.1f}" if latency_values else "-"
            case_rows += (
                "<tr>"
                f"<td><code>{escape(str(case.get('case_id')))}</code></td>"
                f"<td>{escape(str(case.get('cohort')))}</td>"
                f"<td>{_eval_status_chip(str(case.get('status')))}</td>"
                f"<td>{escape(str(case.get('first_relevant_rank') or '-'))}</td>"
                f"<td>{escape(str(case.get('estimated_tokens')))}</td>"
                f"<td>{latency_mean}</td>"
                f"<td>{graph_count}</td>"
                f"<td>{escape(citations[:200])}</td>"
                "</tr>"
            )
        cases_section = (
            "<section class='card'><h2>Per-case results</h2>"
            "<table><thead><tr><th>Case</th><th>Cohort</th><th>Status</th><th>First rank</th>"
            "<th>Tokens</th><th>Latency ms</th><th>Graph candidates</th><th>Selected citations</th></tr></thead>"
            "<tbody>" + (case_rows or "<tr><td colspan='8'>no cases recorded</td></tr>") + "</tbody></table></section>"
        )
        return _page(
            f"Evaluation {run_id}",
            f"<h1>Evaluation run</h1>{identity}{metrics_table}{intervals_section}{regressions_section}{cases_section}",
        )

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
            from ..retrieval_traces import resolve_trace_settings

            evidence = retrieve_evidence(
                project_path or Path.cwd(),
                query,
                embedding_provider=provider,
                index_name=index,
                limit=max(1, min(n_results, 20)),
                token_budget=config.retrieval.token_budget,
                graph=settings,
                trace=resolve_trace_settings(config.retrieval),
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
