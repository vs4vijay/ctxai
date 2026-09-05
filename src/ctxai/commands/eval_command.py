"""``ctxai eval`` command implementations (RE-01 retrieval, HH-09 agent tasks).

Runs the versioned retrieval benchmark against a local index through the
production retrieval services and the versioned agent task benchmark through
the production agent loop, persists redacted immutable artifacts, and applies
baseline gates. Rendering happens in ``app.py`` conventions: this module holds
the logic and returns process exit codes (0 success, 1 errors, partial runs,
incompatible baselines, or failed gates).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..evals.agent_artifacts import (
    AgentEvalArtifact,
    agent_default_artifact_path,
    compare_agent_with_baseline,
    save_agent_artifact,
)
from ..evals.artifacts import (
    ComparisonBlock,
    EvaluationArtifact,
    compare_graph_gate,
    compare_with_baseline,
    default_artifact_path,
)
from ..evals.benchmark import BenchmarkValidationError, RetrievalBenchmark, load_benchmark, validate_benchmark_payload
from ..evals.common import MetricValue, atomic_write_json, redact_artifact
from ..evals.conformance import run_mock_conformance, run_provider_conformance
from ..evals.retrieval_runner import (
    EvalError,
    RetrievalBenchmarkRunner,
    RetrievalEvalConfig,
    runtime_expectation_errors,
)
from ..evals.runner import (
    AgentBenchmarkRunner,
    AgentEvalConfig,
    configured_provider_factory,
    mock_provider_factory,
    mock_provider_identity,
)
from ..evals.task_benchmark import (
    AgentTaskBenchmark,
    load_agent_benchmark,
    validate_agent_benchmark_payload,
)
from ..evals.task_benchmark import (
    BenchmarkValidationError as AgentBenchmarkValidationError,
)

console = Console(legacy_windows=False)
stderr_console = Console(file=sys.stderr, legacy_windows=False)

METRIC_ORDER = (
    "successful_query_rate",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    "evidence_precision@5",
    "latency_p50_ms",
    "latency_p95_ms",
    "selected_token_mean",
    "selected_token_p95",
    "duplicate_token_ratio",
    "graph_contribution_rate",
)

AGENT_METRIC_ORDER = (
    "pass_rate",
    "mean_iterations",
    "p95_iterations",
    "token_mean",
    "token_p95",
    "cost_total",
)

CONFIGURED_PROVIDER_COST_WARNING = (
    "Configured-provider run: this executes real agent tasks over the network with your configured "
    "LLM provider and consumes tokens (cost). It is a maintainer action and is never run by default "
    "test paths or CI."
)


def _fail(message: str) -> int:
    """Print an error line and return the process exit code.

    Args:
        message: The error message to print.

    Returns:
        Exit code 1.
    """
    console.print(f"[red][X] {message}[/red]")
    return 1


def _resolve_output_path(
    output: Path | None,
    project_root: Path,
    benchmark_name: str,
    run_id: str,
    created_at: str,
) -> Path:
    """Resolve the artifact output path under the project-boundary policy.

    Args:
        output: User-selected output path, or ``None`` for the default.
        project_root: Resolved repository root.
        benchmark_name: Benchmark name (used for default naming).
        run_id: Evaluation run id (used for default naming).
        created_at: Creation timestamp (used for default naming).

    Returns:
        The resolved output path.

    Raises:
        EvalError: If the user-selected path escapes the project boundary.
    """
    if output is None:
        return default_artifact_path(project_root, benchmark_name, run_id, created_at)
    resolved = output.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise EvalError(
            f"Output path {output} is outside the project boundary; artifacts must stay inside {project_root}"
        ) from exc
    return resolved


def _resolve_agent_output_path(
    output: Path | None,
    project_root: Path,
    artifact: AgentEvalArtifact,
) -> Path:
    """Resolve the agent artifact output path under the project boundary.

    Args:
        output: User-selected output path, or ``None`` for the default.
        project_root: Resolved repository root.
        artifact: The artifact (used for default naming).

    Returns:
        The resolved output path.

    Raises:
        EvalError: If the user-selected path escapes the project boundary.
    """
    if output is None:
        return agent_default_artifact_path(
            project_root, str(artifact.benchmark["name"]), artifact.run_id, artifact.created_at
        )
    resolved = output.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise EvalError(
            f"Output path {output} is outside the project boundary; artifacts must stay inside {project_root}"
        ) from exc
    return resolved


def _format_metric(value: MetricValue) -> str:
    """Format a metric value for terminal tables.

    Args:
        value: The metric value.

    Returns:
        Human-readable string; unavailable metrics show their reason.
    """
    if not value.is_available:
        return f"[yellow]unavailable ({value.reason})[/yellow]"
    assert value.value is not None
    return f"{value.value:.4f}"


def run_retrieval_eval(
    benchmark_path: Path,
    index_name: str,
    project_path: Path | None = None,
    output: Path | None = None,
    baseline: Path | None = None,
    fail_on_regression: bool = False,
    repeat: int = 1,
    graph: bool | None = False,
    as_json: bool = False,
) -> int:
    """Run the retrieval benchmark and report/aggregate/persist the artifact.

    Args:
        benchmark_path: Path to the versioned benchmark JSON.
        index_name: Index to evaluate against.
        project_path: Project root (defaults to the current directory).
        output: Optional artifact output path (must stay inside the project).
        baseline: Optional baseline artifact to compare against.
        fail_on_regression: Exit non-zero when any gate regresses.
        repeat: Executions per case (1-10; first repeat warms up when > 1).
        graph: Run with graph expansion enabled (IG-03). Requires a healthy,
            generation-matched graph; the run fails explicitly otherwise.
        as_json: Print the exact on-disk artifact JSON instead of tables.

    Returns:
        Process exit code: 0 success, 1 for errors, partial runs,
        incompatible baselines, or failed gates.
    """
    project_root = (project_path or Path.cwd()).resolve()

    try:
        benchmark = load_benchmark(benchmark_path)
    except BenchmarkValidationError as exc:
        console.print("[red][X] Benchmark validation failed:[/red]")
        for error in exc.errors:
            console.print(f"  [red]- {error}[/red]")
        return 1

    try:
        config = RetrievalEvalConfig(repeats=repeat, graph_enabled=bool(graph))
    except ValueError as exc:
        return _fail(str(exc))

    embedding_config = ConfigManager(project_root).load().embedding
    try:
        embedding_provider = EmbeddingsFactory.create(embedding_config)
    except Exception as exc:
        return _fail(f"Cannot create the configured embedding provider: {exc}")

    try:
        runner = RetrievalBenchmarkRunner(
            project_root=project_root,
            benchmark=benchmark,
            index_name=index_name,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            config=config,
        )
    except EvalError as exc:
        return _fail(str(exc))

    try:
        artifact = runner.run()
    except EvalError as exc:
        return _fail(str(exc))

    baseline_payload: dict[str, Any] | None = None
    if fail_on_regression and baseline is None:
        return _fail("--fail-on-regression requires --baseline")
    if baseline is not None:
        try:
            loaded = json.loads(Path(baseline).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return _fail(f"Baseline artifact is not readable JSON at {baseline}: {exc}")
        if not isinstance(loaded, dict):
            return _fail(f"Baseline artifact at {baseline} is not a JSON object")
        baseline_payload = loaded
        comparison = compare_with_baseline(artifact, baseline_payload)
        artifact = dataclasses.replace(artifact, comparison=comparison)

    try:
        destination = _resolve_output_path(
            output, project_root, str(artifact.benchmark["name"]), artifact.run_id, artifact.created_at
        )
    except EvalError as exc:
        return _fail(str(exc))
    payload = redact_artifact(artifact.to_dict(), project_root)
    atomic_write_json(destination, payload)

    exit_code = 0
    if artifact.status != "complete":
        exit_code = 1
    if baseline_payload is not None and artifact.comparison is not None:
        if artifact.comparison.status == "incompatible":
            exit_code = 1
        elif artifact.comparison.status == "regression" and fail_on_regression:
            exit_code = 1

    if as_json:
        console.print_json(json.dumps(payload, sort_keys=True))
    else:
        _render_run_report(project_root, benchmark, index_name, artifact, destination)
    return exit_code


def compare_retrieval_graph_gate(baseline_path: Path, graph_path: Path, as_json: bool = False) -> int:
    """Compare a graph-enabled retrieval artifact against its no-graph baseline.

    Implements the IG-03 acceptance-5 gate: graph expansion must not regress
    Recall@5/MRR (and the other gated quality metrics) beyond the checked-in
    tolerances, and at least one pre-registered relationship-oriented metric
    must improve on the derived ``graph-relationship`` cohort.

    Args:
        baseline_path: Path to the no-graph baseline artifact JSON.
        graph_path: Path to the graph-enabled candidate artifact JSON.
        as_json: Print the verdict as versioned JSON instead of tables.

    Returns:
        Process exit code: 0 when the gate passes, 1 otherwise.
    """
    payloads: dict[str, Any] = {}
    for label, path in (("baseline", baseline_path), ("graph", graph_path)):
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return _fail(f"{label} artifact is not readable JSON at {path}: {exc}")
        if not isinstance(loaded, dict):
            return _fail(f"{label} artifact at {path} is not a JSON object")
        payloads[label] = loaded

    verdict = compare_graph_gate(payloads["baseline"], payloads["graph"])

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "retrieval-graph-gate",
                    "passed": verdict.passed,
                    "compatible": verdict.compatible,
                    "incompatibilities": verdict.incompatibilities,
                    "improvement": verdict.improvement.to_dict() if verdict.improvement else None,
                    "gates": [gate.to_dict() for gate in verdict.gates],
                },
                sort_keys=True,
            )
        )
        return 0 if verdict.passed else 1

    from rich.table import Table

    if not verdict.compatible:
        console.print("[red]Artifacts are not comparable; the gate was not evaluated:[/red]")
        for item in verdict.incompatibilities:
            console.print(f"  [red]- {item}[/red]")
        return 1

    gate_table = Table(title="Graph-expansion gate (graph run vs no-graph baseline)")
    gate_table.add_column("Cohort")
    gate_table.add_column("Metric")
    gate_table.add_column("Baseline (no graph)")
    gate_table.add_column("Graph run")
    gate_table.add_column("Delta")
    gate_table.add_column("Status")
    for gate in verdict.gates:
        baseline_text = f"{gate.baseline:.4f}" if gate.baseline is not None else "-"
        current_text = f"{gate.current:.4f}" if gate.current is not None else "-"
        delta_text = f"{gate.delta:+.4f}" if gate.delta is not None else "-"
        status_style = {"pass": "green", "regression": "red", "unavailable": "yellow"}.get(gate.status, "white")
        gate_table.add_row(
            gate.cohort,
            gate.metric,
            baseline_text,
            current_text,
            delta_text,
            f"[{status_style}]{gate.status}[/{status_style}]",
        )
    console.print(gate_table)

    improvement = verdict.improvement
    if improvement is not None and improvement.status == "pass":
        console.print(
            f"[green][OK] Improvement: {improvement.cohort}/{improvement.metric} "
            f"{improvement.baseline:.4f} -> {improvement.current:.4f} ({improvement.delta:+.4f})[/green]"
        )
    else:
        detail = improvement.detail if improvement is not None else "no improvement evaluated"
        console.print(f"[red][X] No pre-registered relationship-oriented improvement: {detail}[/red]")

    if verdict.passed:
        console.print(
            "[green][OK] Graph-expansion gate passed: no quality regression and a relationship"
            " improvement is present[/green]"
        )
    else:
        console.print(
            "[red][X] Graph-expansion gate failed; graph-expanded retrieval cannot become the"
            " default (criterion IG-03.5)[/red]"
        )
    return 0 if verdict.passed else 1


def _render_run_report(
    project_root: Path,
    benchmark: RetrievalBenchmark,
    index_name: str,
    artifact: EvaluationArtifact,
    destination: Path,
) -> None:
    """Render the human-readable run report.

    Args:
        project_root: Resolved repository root.
        benchmark: The executed benchmark.
        index_name: Index name.
        artifact: The produced artifact.
        destination: Where the artifact was written.
    """
    from rich.table import Table

    console.print(
        f"\n[bold blue]Retrieval benchmark:[/bold blue] {benchmark.name} "
        f"[dim](fingerprint {artifact.benchmark['fingerprint'][:12]})[/dim]"
    )
    console.print(f"[dim]index: {index_name} | run: {artifact.run_id} | status: {artifact.status}[/dim]")
    graph_block = artifact.configuration.get("graph_expansion") or {}
    if graph_block.get("enabled"):
        console.print("[dim]graph expansion: enabled (IG-03)[/dim]")
        cohort_cases = graph_block.get("relationship_cohort_cases") or []
        if cohort_cases:
            console.print(
                f"[dim]graph-relationship cohort: {len(cohort_cases)} case(s) whose expected files have"
                " cross-file relationship edges[/dim]"
            )
        if graph_block.get("note"):
            console.print(f"[yellow]graph cohort note: {graph_block['note']}[/yellow]")

    metrics_table = Table(title="Aggregate metrics")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Overall")
    for cohort in sorted(artifact.aggregates["by_cohort"]):
        metrics_table.add_column(cohort)
    for metric in METRIC_ORDER:
        row = [metric]
        overall = artifact.aggregates["overall"].metrics.get(metric)
        row.append(_format_metric(overall) if overall else "[yellow]unavailable[/yellow]")
        for cohort in sorted(artifact.aggregates["by_cohort"]):
            value = artifact.aggregates["by_cohort"][cohort].metrics.get(metric)
            row.append(_format_metric(value) if value else "-")
        metrics_table.add_row(*row)
    console.print(metrics_table)

    intervals = artifact.aggregates["overall"].confidence_intervals
    if intervals:
        ci_table = Table(title="Bootstrap 95% confidence intervals (overall)")
        ci_table.add_column("Metric")
        ci_table.add_column("Low")
        ci_table.add_column("High")
        for metric in sorted(intervals):
            low, high = intervals[metric]
            ci_table.add_row(metric, f"{low:.4f}", f"{high:.4f}")
        console.print(ci_table)

    if artifact.errors:
        console.print("[red]Case errors (run is partial):[/red]")
        for error in artifact.errors:
            console.print(f"  [red]- {error}[/red]")

    comparison = artifact.comparison
    if comparison is not None:
        _render_baseline_comparison(comparison, artifact.status)

    console.print(f"\n[green][OK] Artifact written to {destination}[/green]")
    console.print(f"[dim]configuration fingerprint: {artifact.configuration['fingerprint'][:12]}[/dim]")
    console.print(f"[dim]artifact directory: {project_root / '.ctxai' / 'evaluations' / 'retrieval'}[/dim]")


def _render_baseline_comparison(comparison: ComparisonBlock | None, run_status: str) -> None:
    """Render the shared baseline-comparison block (both eval frameworks).

    Args:
        comparison: The comparison block of the current artifact (``None``
            renders nothing).
        run_status: The artifact run status (``complete``/``partial``).
    """
    from rich.table import Table

    if comparison is None:
        return
    if comparison.incompatibilities:
        console.print("[red]Baseline is incompatible; gates were not evaluated:[/red]")
        for item in comparison.incompatibilities:
            console.print(f"  [red]- {item}[/red]")
    gate_table = Table(title=f"Baseline comparison ({comparison.status})")
    gate_table.add_column("Cohort")
    gate_table.add_column("Metric")
    gate_table.add_column("Baseline")
    gate_table.add_column("Current")
    gate_table.add_column("Delta")
    gate_table.add_column("Tolerance (abs/rel)")
    gate_table.add_column("Status")
    for gate in comparison.gates:
        baseline_text = f"{gate.baseline:.4f}" if gate.baseline is not None else "-"
        current_text = f"{gate.current:.4f}" if gate.current is not None else "-"
        delta_text = f"{gate.delta:+.4f}" if gate.delta is not None else "-"
        status_style = {"pass": "green", "regression": "red", "unavailable": "yellow"}.get(gate.status, "white")
        gate_table.add_row(
            gate.cohort,
            gate.metric,
            baseline_text,
            current_text,
            delta_text,
            f"{gate.absolute_tolerance}/{gate.relative_tolerance}",
            f"[{status_style}]{gate.status}[/{status_style}]",
        )
    console.print(gate_table)
    if comparison.status == "regression":
        failing = [gate for gate in comparison.gates if gate.status == "regression"]
        for gate in failing:
            console.print(
                f"[red][X] Gate failed: {gate.cohort}/{gate.metric} "
                f"baseline {gate.baseline:.4f} -> current {gate.current:.4f} "
                f"(tolerance abs={gate.absolute_tolerance}, rel={gate.relative_tolerance})[/red]"
            )
        if run_status == "complete":
            console.print(
                "[yellow]Regressions are informational without --fail-on-regression; pass it to gate on them.[/yellow]"
            )


def validate_retrieval_benchmark(
    benchmark_path: Path,
    project_path: Path | None = None,
    as_json: bool = False,
) -> int:
    """Validate a benchmark document without running retrieval.

    Checks schema, duplicate IDs, path formats, evidence ranges, splits, and
    expectations. When a project path is supplied, expectations are also
    checked against the repository (file existence, line-range bounds).

    Args:
        benchmark_path: Path to the benchmark JSON.
        project_path: Optional project root for repository-level checks.
        as_json: Print a versioned JSON validation envelope instead of text.

    Returns:
        Process exit code: 0 valid, 1 invalid.
    """
    errors: list[str] = []
    payload: Any = None
    try:
        payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot parse benchmark file: {exc}")
    if payload is not None:
        errors.extend(validate_benchmark_payload(payload))

    benchmark: RetrievalBenchmark | None = None
    if not errors and payload is not None:
        try:
            benchmark = load_benchmark(benchmark_path)
        except BenchmarkValidationError as exc:
            errors.extend(exc.errors)

    repository_errors: dict[str, list[str]] = {}
    if not errors and benchmark is not None and project_path is not None:
        project_root = project_path.resolve()
        for case in benchmark.cases:
            case_errors = runtime_expectation_errors(case, project_root)
            if case_errors:
                repository_errors[case.id] = case_errors

    valid = not errors and not repository_errors
    if as_json:
        envelope = {
            "schema_version": 1,
            "benchmark": benchmark.name if benchmark else None,
            "valid": valid,
            "errors": errors,
            "repository_errors": repository_errors,
            "case_count": len(benchmark.cases) if benchmark else 0,
        }
        console.print_json(json.dumps(envelope, sort_keys=True))
        return 0 if valid else 1

    if benchmark is None:
        for error in errors:
            console.print(f"  [red]- {error}[/red]")
        return 1

    cohorts: dict[str, int] = {}
    splits: dict[str, int] = {}
    for case in benchmark.cases:
        cohorts[case.cohort] = cohorts.get(case.cohort, 0) + 1
        splits[case.split] = splits.get(case.split, 0) + 1
    console.print(f"[green][OK] Benchmark '{benchmark.name}' is valid[/green]")
    console.print(
        f"[dim]cases: {len(benchmark.cases)} | cohorts: {cohorts} | splits: {splits} | "
        f"fingerprint: {benchmark.fingerprint[:12]}[/dim]"
    )
    if repository_errors:
        for case_id, case_errors in repository_errors.items():
            for error in case_errors:
                console.print(f"  [red]- {case_id}: {error}[/red]")
        return 1
    return 0


# ----------------------------------------------------------------------
# HH-09: agent task benchmark
# ----------------------------------------------------------------------


def _format_iterations(value: MetricValue) -> str:
    """Format an iteration/token metric for terminal tables.

    Args:
        value: The metric value.

    Returns:
        Human-readable string; unavailable metrics show their reason.
    """
    if not value.is_available:
        return f"[yellow]unavailable ({value.reason})[/yellow]"
    assert value.value is not None
    return f"{value.value:.2f}"


def run_agent_eval(
    benchmark_path: Path,
    provider_mode: str = "mock",
    project_path: Path | None = None,
    cases: list[str] | None = None,
    output: Path | None = None,
    baseline: Path | None = None,
    fail_on_regression: bool = False,
    as_json: bool = False,
) -> int:
    """Run the agent task benchmark and report/aggregate/persist the artifact.

    Mock mode (the default and the CI path) scripts every case with
    ``MockLLMProvider``: deterministic, no network, no credentials.
    Configured mode is an explicit maintainer action and prints a cost
    warning to stderr before any provider call.

    Args:
        benchmark_path: Path to the versioned agent benchmark JSON.
        provider_mode: ``mock`` or ``configured``.
        project_path: Project root (defaults to the current directory).
        cases: Optional case-id subset to run.
        output: Optional artifact output path (must stay inside the project).
        baseline: Optional baseline artifact to compare against.
        fail_on_regression: Exit non-zero when any gate regresses.
        as_json: Print the exact on-disk artifact JSON instead of tables.

    Returns:
        Process exit code: 0 success, 1 for errors, partial runs,
        incompatible baselines, or failed gates.
    """
    project_root = (project_path or Path.cwd()).resolve()

    if provider_mode not in ("mock", "configured"):
        return _fail(f"provider must be 'mock' or 'configured', got '{provider_mode}'")

    try:
        benchmark = load_agent_benchmark(benchmark_path)
        selected = benchmark.select_cases(cases)
    except AgentBenchmarkValidationError as exc:
        console.print("[red][X] Benchmark validation failed:[/red]")
        for error in exc.errors:
            console.print(f"  [red]- {error}[/red]")
        return 1

    if provider_mode == "configured":
        stderr_console.print(f"[yellow]WARNING: {CONFIGURED_PROVIDER_COST_WARNING}[/yellow]")

    try:
        if provider_mode == "mock":
            factory = mock_provider_factory()
            identity = mock_provider_identity()
        else:
            factory, identity = configured_provider_factory(project_root)
    except EvalError as exc:
        return _fail(str(exc))

    try:
        config = AgentEvalConfig()
    except ValueError as exc:
        return _fail(str(exc))

    runner = AgentBenchmarkRunner(
        project_root=project_root,
        benchmark=benchmark,
        provider_factory=factory,
        provider_identity=identity,
        config=config,
        cases=selected,
    )
    artifact = runner.run()

    baseline_payload: dict[str, Any] | None = None
    if fail_on_regression and baseline is None:
        return _fail("--fail-on-regression requires --baseline")
    if baseline is not None:
        try:
            loaded = json.loads(Path(baseline).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return _fail(f"Baseline artifact is not readable JSON at {baseline}: {exc}")
        if not isinstance(loaded, dict):
            return _fail(f"Baseline artifact at {baseline} is not a JSON object")
        baseline_payload = loaded
        comparison = compare_agent_with_baseline(artifact, baseline_payload)
        artifact = dataclasses.replace(artifact, comparison=comparison)

    try:
        destination = _resolve_agent_output_path(output, project_root, artifact)
    except EvalError as exc:
        return _fail(str(exc))
    save_agent_artifact(artifact, destination, project_root)

    exit_code = 0
    if artifact.status != "complete":
        exit_code = 1
    if baseline_payload is not None and artifact.comparison is not None:
        if artifact.comparison.status == "incompatible":
            exit_code = 1
        elif artifact.comparison.status == "regression" and fail_on_regression:
            exit_code = 1

    if as_json:
        payload = redact_artifact(artifact.to_dict(), project_root)
        console.print_json(json.dumps(payload, sort_keys=True))
    else:
        _render_agent_report(project_root, benchmark, artifact, destination, provider_mode)
    return exit_code


def _render_agent_report(
    project_root: Path,
    benchmark: AgentTaskBenchmark,
    artifact: AgentEvalArtifact,
    destination: Path,
    provider_mode: str,
) -> None:
    """Render the human-readable agent benchmark report.

    Args:
        project_root: Resolved repository root.
        benchmark: The executed benchmark.
        artifact: The produced artifact.
        destination: Where the artifact was written.
        provider_mode: The provider mode used for the run.
    """
    from rich.table import Table

    console.print(
        f"\n[bold blue]Agent benchmark:[/bold blue] {benchmark.name} "
        f"[dim](fingerprint {artifact.benchmark['fingerprint'][:12]})[/dim]"
    )
    console.print(f"[dim]provider: {provider_mode} | run: {artifact.run_id} | status: {artifact.status}[/dim]")

    metrics_table = Table(title="Aggregate metrics")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Overall")
    for cohort in sorted(artifact.aggregates["by_cohort"]):
        metrics_table.add_column(cohort)
    for metric in AGENT_METRIC_ORDER:
        row = [metric]
        overall = artifact.aggregates["overall"].metrics.get(metric)
        row.append(_format_metric(overall) if metric == "pass_rate" else _format_iterations(overall))
        for cohort in sorted(artifact.aggregates["by_cohort"]):
            value = artifact.aggregates["by_cohort"][cohort].metrics.get(metric)
            row.append(
                _format_metric(value)
                if metric == "pass_rate"
                else _format_iterations(value or MetricValue.unavailable("n/a"))
            )
        metrics_table.add_row(*row)
    console.print(metrics_table)

    cases_table = Table(title="Case results")
    cases_table.add_column("Case")
    cases_table.add_column("Cohort")
    cases_table.add_column("Split")
    cases_table.add_column("Status")
    cases_table.add_column("Iterations")
    cases_table.add_column("Tokens")
    cases_table.add_column("Cost")
    cases_table.add_column("Failed judgments")
    for run in artifact.runs:
        status_style = {"passed": "green", "failed": "red", "error": "yellow"}.get(run.status, "white")
        failed = [judgment.name for judgment in run.judgments if not judgment.passed]
        cost_text = (
            f"${run.cost.value:.6f}" if run.cost.is_available else f"[yellow]unavailable ({run.cost.reason})[/yellow]"
        )
        cases_table.add_row(
            run.case_id,
            run.cohort,
            run.split,
            f"[{status_style}]{run.status}[/{status_style}]",
            str(run.iterations),
            str(run.tokens.get("total_tokens", 0)),
            cost_text,
            ", ".join(failed) or "-",
        )
    console.print(cases_table)

    if artifact.errors:
        console.print("[red]Case errors (run is partial):[/red]")
        for error in artifact.errors:
            console.print(f"  [red]- {error}[/red]")

    _render_baseline_comparison(artifact.comparison, artifact.status)

    console.print(f"\n[green][OK] Artifact written to {destination}[/green]")
    console.print(f"[dim]configuration fingerprint: {artifact.configuration['fingerprint'][:12]}[/dim]")
    console.print(f"[dim]artifact directory: {project_root / '.ctxai' / 'evaluations' / 'agent'}[/dim]")
    console.print(
        "[dim]per-case transcripts: .ctxai/evaluations/agent/workspaces/<run_id>/<case_id>/project/.ctxai/runs/[/dim]"
    )


def validate_agent_benchmark(
    benchmark_path: Path,
    as_json: bool = False,
) -> int:
    """Validate an agent benchmark document without running the agent.

    Checks schema, duplicate/safe ids, setup and forbidden path formats,
    check shapes, iteration budgets, splits, and mock_script shapes.

    Args:
        benchmark_path: Path to the benchmark JSON.
        as_json: Print a versioned JSON validation envelope instead of text.

    Returns:
        Process exit code: 0 valid, 1 invalid.
    """
    errors: list[str] = []
    payload: Any = None
    try:
        payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot parse benchmark file: {exc}")
    if payload is not None:
        errors.extend(validate_agent_benchmark_payload(payload))

    benchmark: AgentTaskBenchmark | None = None
    if not errors and payload is not None:
        try:
            benchmark = load_agent_benchmark(benchmark_path)
        except AgentBenchmarkValidationError as exc:
            errors.extend(exc.errors)

    valid = not errors
    if as_json:
        cohorts: dict[str, int] = {}
        splits: dict[str, int] = {}
        if benchmark is not None:
            for case in benchmark.cases:
                cohorts[case.cohort] = cohorts.get(case.cohort, 0) + 1
                splits[case.split] = splits.get(case.split, 0) + 1
        envelope = {
            "schema_version": 1,
            "benchmark": benchmark.name if benchmark else None,
            "valid": valid,
            "errors": errors,
            "case_count": len(benchmark.cases) if benchmark else 0,
            "cohorts": cohorts,
            "splits": splits,
        }
        console.print_json(json.dumps(envelope, sort_keys=True))
        return 0 if valid else 1

    if benchmark is None:
        for error in errors:
            console.print(f"  [red]- {error}[/red]")
        return 1

    cohort_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for case in benchmark.cases:
        cohort_counts[case.cohort] = cohort_counts.get(case.cohort, 0) + 1
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
    console.print(f"[green][OK] Agent benchmark '{benchmark.name}' is valid[/green]")
    console.print(
        f"[dim]cases: {len(benchmark.cases)} | cohorts: {cohort_counts} | splits: {split_counts} | "
        f"fingerprint: {benchmark.fingerprint[:12]}[/dim]"
    )
    return 0


# ----------------------------------------------------------------------
# HH-09: provider conformance suite
# ----------------------------------------------------------------------


def run_providers_conformance(
    provider: str | None = None,
    project_path: Path | None = None,
    as_json: bool = False,
) -> int:
    """Run the executable provider conformance suite.

    Without ``--provider`` the suite runs against the scripted mock provider
    (the CI path: no network, no credentials). With ``--provider P`` (P from
    ``PROVIDER_SPECS``) it runs against the configured live provider and
    makes real API calls — an explicit maintainer action introduced by a
    stderr cost warning.

    Args:
        provider: Optional live provider name; ``None`` selects the mock.
        project_path: Project root for configuration and credentials
            (defaults to the current directory).
        as_json: Print a versioned JSON report instead of tables.

    Returns:
        Process exit code: 0 when every check passes, 1 on any failure
        (drift counts as failure), 1 for usage errors.
    """
    from rich.table import Table

    from ..agent.config import AgentLLMConfig
    from ..agent.llm.contract import PROVIDER_SPECS, get_provider_spec
    from ..agent.llm.factory import LLMProviderFactory

    project_root = (project_path or Path.cwd()).resolve()

    if provider is None:
        report = run_mock_conformance()
        reports = [report]
        console.print("[dim]mock conformance (CI path): scripted provider, no network[/dim]")
    else:
        try:
            spec = get_provider_spec(provider)
        except ValueError:
            known = ", ".join(item.name for item in PROVIDER_SPECS)
            return _fail(f"unknown provider '{provider}'; PROVIDER_SPECS knows: {known}")
        stderr_console.print(f"[yellow]WARNING: {CONFIGURED_PROVIDER_COST_WARNING}[/yellow]")
        agent_config = ConfigManager(project_root).load().agent
        llm_config = AgentLLMConfig(
            provider=spec.name,
            model=agent_config.llm.model,
            api_key=agent_config.llm.get_api_key_for_provider(spec.name),
            temperature=agent_config.llm.temperature,
            max_tokens=agent_config.llm.max_tokens,
            timeout=agent_config.llm.timeout,
        )
        try:
            live_provider = LLMProviderFactory.create_provider(llm_config)
        except Exception as exc:
            return _fail(f"cannot create provider '{spec.name}': {exc}")
        reports = [run_provider_conformance(live_provider, spec)]

    exit_code = 0 if all(report.status == "pass" for report in reports) else 1

    if as_json:
        envelope = {
            "schema_version": 1,
            "network": provider is not None,
            "reports": [report.to_dict() for report in reports],
            "status": "pass" if exit_code == 0 else "fail",
        }
        console.print_json(json.dumps(envelope, sort_keys=True))
        return exit_code

    for report in reports:
        observed = report.observed
        console.print(f"\n[bold blue]Provider conformance:[/bold blue] {report.provider} [dim]({report.status})[/dim]")
        console.print(
            f"[dim]declared: tools={report.declared['tools']} streaming={report.declared['streaming']} "
            f"transport={report.declared['transport']}[/dim]"
        )
        console.print(
            f"[dim]observed: tools={observed['tools']} streaming={observed['streaming']} "
            f"context_size={observed['context_size']}[/dim]"
        )
        check_table = Table(title=f"Conformance checks ({report.provider})")
        check_table.add_column("Check")
        check_table.add_column("Status")
        check_table.add_column("Detail")
        for check in report.checks:
            if check.passed:
                status_text = "[green]pass[/green]"
            elif check.drift:
                status_text = "[red]FAIL (drift)[/red]"
            else:
                status_text = "[red]FAIL[/red]"
            check_table.add_row(check.name, status_text, check.detail or "-")
        console.print(check_table)
        if report.status != "pass":
            failing = [check for check in report.checks if not check.passed]
            for check in failing:
                label = f"{check.name} (declared-vs-observed drift)" if check.drift else check.name
                console.print(f"[red][X] Conformance failure: {report.provider}/{label}: {check.detail}[/red]")
    return exit_code
