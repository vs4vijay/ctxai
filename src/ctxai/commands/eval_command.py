"""``ctxai eval`` command implementations (RE-01 retrieval benchmark).

Runs the versioned retrieval benchmark against a local index through the
production retrieval services, persists a redacted immutable artifact, and
applies baseline gates. Rendering happens in ``app.py`` conventions: this
module holds the logic and returns process exit codes (0 success, 1 errors,
partial runs, incompatible baselines, or failed gates).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from ..config import ConfigManager
from ..embeddings import EmbeddingsFactory
from ..evals.artifacts import (
    EvaluationArtifact,
    compare_with_baseline,
    default_artifact_path,
)
from ..evals.benchmark import BenchmarkValidationError, RetrievalBenchmark, load_benchmark, validate_benchmark_payload
from ..evals.common import MetricValue, atomic_write_json, redact_artifact
from ..evals.retrieval_runner import (
    EvalError,
    RetrievalBenchmarkRunner,
    RetrievalEvalConfig,
    runtime_expectation_errors,
)

console = Console(legacy_windows=False)

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


def _fail(message: str) -> int:
    """Print an error line and return the process exit code.

    Args:
        message: The error message to print.

    Returns:
        Exit code 1.
    """
    console.print(f"[red][X] {message}[/red]")
    return 1


def _resolve_output_path(output: Path | None, project_root: Path, artifact: EvaluationArtifact) -> Path:
    """Resolve the artifact output path under the project-boundary policy.

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
        return default_artifact_path(project_root, artifact.benchmark["name"], artifact.run_id, artifact.created_at)
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
        config = RetrievalEvalConfig(repeats=repeat)
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
        destination = _resolve_output_path(output, project_root, artifact)
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
            if artifact.status == "complete":
                console.print(
                    "[yellow]Regressions are informational without --fail-on-regression; "
                    "pass it to gate on them.[/yellow]"
                )

    console.print(f"\n[green][OK] Artifact written to {destination}[/green]")
    console.print(f"[dim]configuration fingerprint: {artifact.configuration['fingerprint'][:12]}[/dim]")
    console.print(f"[dim]artifact directory: {project_root / '.ctxai' / 'evaluations' / 'retrieval'}[/dim]")


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
