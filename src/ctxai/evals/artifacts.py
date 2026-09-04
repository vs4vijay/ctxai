"""Versioned evaluation artifacts, baseline comparison, and gates (RE-01).

An ``EvaluationArtifact`` is an immutable, redacted JSON record of one
benchmark run: benchmark and configuration identity, index identity, per-case
runs with ordered candidates and selected context, aggregate metrics per
cohort/split, and an optional baseline comparison with named gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import MetricValue, atomic_write_json, redact_artifact

EVALUATION_SCHEMA_VERSION = 1
EVALUATION_KIND_RETRIEVAL = "retrieval"

# Metric direction: "higher" metrics regress downward, "lower" metrics regress
# upward. Latency metrics are reported but intentionally NOT gated: noisy
# wall-clock timings must not become a hard cross-platform gate (see RE-03).
METRIC_DIRECTIONS: dict[str, str] = {
    "recall@1": "higher",
    "recall@5": "higher",
    "recall@10": "higher",
    "mrr": "higher",
    "ndcg@10": "higher",
    "evidence_precision@5": "higher",
    "successful_query_rate": "higher",
    "duplicate_token_ratio": "lower",
    "selected_token_mean": "lower",
    "selected_token_p95": "lower",
    "latency_p50_ms": "lower",
    "latency_p95_ms": "lower",
}

# Checked-in tolerances per metric: (absolute, relative-to-baseline). A gate
# regresses when the current value is worse than the baseline by more than
# max(absolute, relative * |baseline|).
DEFAULT_TOLERANCES: dict[str, MetricTolerance] = {}
GATED_METRICS = frozenset(
    {
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@10",
        "evidence_precision@5",
        "successful_query_rate",
        "duplicate_token_ratio",
        "selected_token_mean",
        "selected_token_p95",
    }
)


@dataclass(frozen=True)
class MetricTolerance:
    """Checked-in regression tolerance for one metric.

    Attributes:
        absolute: Absolute delta allowed before a regression is declared.
        relative: Fraction of the baseline value allowed additionally.
    """

    absolute: float
    relative: float


for _name, _abs, _rel in [
    ("recall@1", 0.05, 0.05),
    ("recall@5", 0.02, 0.02),
    ("recall@10", 0.02, 0.02),
    ("mrr", 0.05, 0.05),
    ("ndcg@10", 0.05, 0.05),
    ("evidence_precision@5", 0.05, 0.05),
    ("successful_query_rate", 0.0, 0.0),
    ("duplicate_token_ratio", 0.05, 0.10),
    ("selected_token_mean", 50.0, 0.10),
    ("selected_token_p95", 100.0, 0.10),
]:
    DEFAULT_TOLERANCES[_name] = MetricTolerance(absolute=_abs, relative=_rel)


@dataclass(frozen=True)
class CandidateRecord:
    """One ranked candidate with its retrieval provenance and decision.

    Attributes:
        chunk_id: Content-derived chunk identifier from the index.
        file_path: Repository-relative file path.
        start_line: Chunk start line (1-based, inclusive).
        end_line: Chunk end line (inclusive).
        citation: ``file:start-end`` evidence citation.
        chunk_type: Tree-sitter chunk type.
        score: Final fused retrieval score.
        reasons: Component provenance entries (e.g. ``semantic rank 3``).
        final_rank: 1-based rank in the candidate ordering.
        estimated_tokens: Estimated tokens of the chunk content.
        decision: ``selected``, ``duplicate`` (identity already selected), or
            ``budget`` (excluded by the context token budget).
        truncated: Whether the assembler clipped the content (selected items
            only; ``None`` otherwise).
    """

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    citation: str
    chunk_type: str
    score: float
    reasons: list[str]
    final_rank: int
    estimated_tokens: int
    decision: str
    truncated: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for one candidate.
        """
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "citation": self.citation,
            "chunk_type": self.chunk_type,
            "score": self.score,
            "reasons": list(self.reasons),
            "final_rank": self.final_rank,
            "estimated_tokens": self.estimated_tokens,
            "decision": self.decision,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRecord:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed CandidateRecord.
        """
        return cls(
            chunk_id=data["chunk_id"],
            file_path=data["file_path"],
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            citation=data["citation"],
            chunk_type=data["chunk_type"],
            score=float(data["score"]),
            reasons=list(data.get("reasons", [])),
            final_rank=int(data["final_rank"]),
            estimated_tokens=int(data["estimated_tokens"]),
            decision=data["decision"],
            truncated=data.get("truncated"),
        )


@dataclass(frozen=True)
class CaseRunRecord:
    """One benchmark case executed against the production retrieval service.

    Attributes:
        case_id: Benchmark case identifier.
        run_id: Artifact run id.
        timestamp: ISO-8601 execution timestamp (volatile).
        query: Raw query text when query recording is enabled.
        query_hash: Deterministic sha256 of the query when recording is
            disabled (mutually exclusive with ``query``).
        cohort: Benchmark cohort label.
        split: Benchmark split label.
        status: ``ok`` when the case executed without an error.
        error: First error message when the case failed, else ``None``.
        expected: Expected evidence dict for the case.
        candidate_count: Number of ranked candidates returned.
        selected_count: Number of context items selected by the assembler.
        candidates: Ordered candidate records with decisions.
        estimated_tokens: Estimated tokens of the assembled context.
        first_relevant_rank: 1-based rank of the first relevant file.
        metrics: Per-case metric values (may include unavailable entries).
        latency: Measured per-repeat latencies (volatile).
        timings: Per-stage timings (volatile).
        line_range_findings: Evidence-range overlap judgments per expected
            line range.
    """

    case_id: str
    run_id: str
    timestamp: str
    query: str | None
    query_hash: str | None
    cohort: str
    split: str
    status: str
    error: str | None
    expected: dict[str, Any]
    candidate_count: int
    selected_count: int
    candidates: list[CandidateRecord]
    estimated_tokens: int
    first_relevant_rank: int | None
    metrics: dict[str, MetricValue]
    latency: dict[str, Any]
    timings: dict[str, Any]
    line_range_findings: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for one case run.
        """
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "query_hash": self.query_hash,
            "cohort": self.cohort,
            "split": self.split,
            "status": self.status,
            "error": self.error,
            "expected": self.expected,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "estimated_tokens": self.estimated_tokens,
            "first_relevant_rank": self.first_relevant_rank,
            "metrics": {name: value.to_dict() for name, value in self.metrics.items()},
            "latency": self.latency,
            "timings": self.timings,
            "line_range_findings": self.line_range_findings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseRunRecord:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed CaseRunRecord.

        Raises:
            ValueError: If a metric entry is not a valid MetricValue payload.
        """
        return cls(
            case_id=data["case_id"],
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            query=data.get("query"),
            query_hash=data.get("query_hash"),
            cohort=data["cohort"],
            split=data["split"],
            status=data["status"],
            error=data.get("error"),
            expected=data.get("expected") or {},
            candidate_count=int(data.get("candidate_count", 0)),
            selected_count=int(data.get("selected_count", 0)),
            candidates=[CandidateRecord.from_dict(item) for item in data.get("candidates", [])],
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            first_relevant_rank=data.get("first_relevant_rank"),
            metrics={name: MetricValue.from_dict(value) for name, value in (data.get("metrics") or {}).items()},
            latency=data.get("latency") or {},
            timings=data.get("timings") or {},
            line_range_findings=list(data.get("line_range_findings", [])),
        )


@dataclass(frozen=True)
class CohortMetricsBlock:
    """Aggregate metrics for a group of cases (cohort, split, or overall).

    Attributes:
        cases: Number of cases in the group.
        successful: Cases that produced at least one candidate without error.
        errored: Cases that ended in an error.
        metrics: Aggregate metrics; unavailable entries carry a reason.
        confidence_intervals: Deterministic bootstrap CIs keyed by metric.
    """

    cases: int
    successful: int
    errored: int
    metrics: dict[str, MetricValue]
    confidence_intervals: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for a metrics block.
        """
        return {
            "cases": self.cases,
            "successful": self.successful,
            "errored": self.errored,
            "metrics": {name: value.to_dict() for name, value in self.metrics.items()},
            "confidence_intervals": {name: list(value) for name, value in self.confidence_intervals.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CohortMetricsBlock:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed CohortMetricsBlock.

        Raises:
            ValueError: If a metric entry is not a valid MetricValue payload.
        """
        return cls(
            cases=int(data.get("cases", 0)),
            successful=int(data.get("successful", 0)),
            errored=int(data.get("errored", 0)),
            metrics={name: MetricValue.from_dict(value) for name, value in (data.get("metrics") or {}).items()},
            confidence_intervals={
                name: [float(bound) for bound in value]
                for name, value in (data.get("confidence_intervals") or {}).items()
            },
        )


@dataclass(frozen=True)
class GateResult:
    """One metric/cohort gate compared against a baseline.

    Attributes:
        cohort: Cohort (or ``overall``) the gate belongs to.
        metric: Metric name.
        baseline: Baseline value (``None`` when unavailable).
        current: Current value (``None`` when unavailable).
        delta: ``current - baseline`` when both are available.
        direction: ``higher`` or ``lower`` (which direction is better).
        absolute_tolerance: Checked-in absolute tolerance.
        relative_tolerance: Checked-in relative tolerance.
        status: ``pass``, ``regression``, or ``unavailable``.
        detail: Extra context (e.g. the unavailability reason).
    """

    cohort: str
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    direction: str
    absolute_tolerance: float
    relative_tolerance: float
    status: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for one gate.
        """
        return {
            "cohort": self.cohort,
            "metric": self.metric,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "direction": self.direction,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "status": self.status,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateResult:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed GateResult.
        """
        return cls(
            cohort=data["cohort"],
            metric=data["metric"],
            baseline=data.get("baseline"),
            current=data.get("current"),
            delta=data.get("delta"),
            direction=data["direction"],
            absolute_tolerance=float(data["absolute_tolerance"]),
            relative_tolerance=float(data["relative_tolerance"]),
            status=data["status"],
            detail=data.get("detail"),
        )


@dataclass(frozen=True)
class ComparisonBlock:
    """Comparison of an artifact against an optional baseline artifact.

    Attributes:
        baseline: Identity fields of the baseline artifact.
        compatible: Whether the two artifacts are comparable.
        incompatibilities: Named identity fields that prevented comparison.
        gates: Per metric/cohort gate results.
        status: ``pass``, ``regression``, or ``incompatible``.
    """

    baseline: dict[str, Any]
    compatible: bool
    incompatibilities: list[str]
    gates: list[GateResult]
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for the comparison block.
        """
        return {
            "baseline": self.baseline,
            "compatible": self.compatible,
            "incompatibilities": list(self.incompatibilities),
            "gates": [gate.to_dict() for gate in self.gates],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonBlock:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed ComparisonBlock.
        """
        return cls(
            baseline=data.get("baseline") or {},
            compatible=bool(data.get("compatible", False)),
            incompatibilities=list(data.get("incompatibilities", [])),
            gates=[GateResult.from_dict(gate) for gate in data.get("gates", [])],
            status=data["status"],
        )


@dataclass(frozen=True)
class EvaluationArtifact:
    """Immutable, versioned record of one benchmark evaluation run.

    Attributes:
        schema_version: Artifact schema version (currently 1).
        kind: Evaluation kind (``retrieval``; HH-09 adds ``agent``).
        run_id: Unique id of this evaluation run.
        created_at: ISO-8601 creation timestamp (volatile).
        duration_ms: Total wall-clock duration (volatile).
        status: ``complete`` when every case executed, ``partial`` otherwise.
        benchmark: Benchmark identity (name, fingerprint, case count).
        configuration: Configuration identity (fingerprint, embedding,
            retrieval settings).
        index: Index identity and health at run time.
        environment: Environment metadata (python, platform, version).
        runs: Per-case run records.
        aggregates: Metrics per ``overall``/``by_cohort``/``by_split``.
        comparison: Baseline comparison when a baseline was supplied.
        errors: Run-level errors that made the run partial.
    """

    schema_version: int
    kind: str
    run_id: str
    created_at: str
    duration_ms: float
    status: str
    benchmark: dict[str, Any]
    configuration: dict[str, Any]
    index: dict[str, Any]
    environment: dict[str, Any]
    runs: list[CaseRunRecord]
    aggregates: dict[str, Any]
    comparison: ComparisonBlock | None
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to the canonical JSON representation.

        Returns:
            Dictionary matching the on-disk artifact schema.
        """
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "benchmark": self.benchmark,
            "configuration": self.configuration,
            "index": self.index,
            "environment": self.environment,
            "runs": [run.to_dict() for run in self.runs],
            "aggregates": {
                "overall": self.aggregates["overall"].to_dict(),
                "by_cohort": {name: block.to_dict() for name, block in self.aggregates["by_cohort"].items()},
                "by_split": {name: block.to_dict() for name, block in self.aggregates["by_split"].items()},
            },
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationArtifact:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed EvaluationArtifact.

        Raises:
            ValueError: If nested payloads are invalid.
        """
        aggregates = data.get("aggregates") or {}
        return cls(
            schema_version=int(data["schema_version"]),
            kind=data["kind"],
            run_id=data["run_id"],
            created_at=data["created_at"],
            duration_ms=float(data.get("duration_ms", 0.0)),
            status=data["status"],
            benchmark=data.get("benchmark") or {},
            configuration=data.get("configuration") or {},
            index=data.get("index") or {},
            environment=data.get("environment") or {},
            runs=[CaseRunRecord.from_dict(run) for run in data.get("runs", [])],
            aggregates={
                "overall": CohortMetricsBlock.from_dict(aggregates.get("overall") or {}),
                "by_cohort": {
                    name: CohortMetricsBlock.from_dict(block)
                    for name, block in (aggregates.get("by_cohort") or {}).items()
                },
                "by_split": {
                    name: CohortMetricsBlock.from_dict(block)
                    for name, block in (aggregates.get("by_split") or {}).items()
                },
            },
            comparison=ComparisonBlock.from_dict(data["comparison"]) if data.get("comparison") else None,
            errors=list(data.get("errors", [])),
        )


def evaluations_dir(project_root: Path) -> Path:
    """Default immutable-artifact directory for retrieval evaluations.

    Args:
        project_root: Resolved repository root.

    Returns:
        ``<project>/.ctxai/evaluations/retrieval``.
    """
    return project_root / ".ctxai" / "evaluations" / "retrieval"


def default_artifact_path(project_root: Path, benchmark_name: str, run_id: str, created_at: str) -> Path:
    """Build the default artifact path from benchmark identity and time.

    Args:
        project_root: Resolved repository root.
        benchmark_name: Benchmark name (validated separately).
        run_id: Evaluation run id.
        created_at: ISO-8601 timestamp used for a sortable file prefix.

    Returns:
        Artifact path under :func:`evaluations_dir`.
    """
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in benchmark_name)
    stamp = created_at.replace(":", "").replace("-", "")
    return evaluations_dir(project_root) / f"{safe_name}-{stamp}-{run_id[:8]}.json"


def save_artifact(artifact: EvaluationArtifact, path: Path, project_root: Path) -> Path:
    """Redact and atomically persist an evaluation artifact.

    The payload passes the shared redaction pipeline (secrets and absolute
    home paths) before an atomic write with fsync.

    Args:
        artifact: The artifact to persist.
        path: Destination path.
        project_root: Resolved repository root used for path redaction.

    Returns:
        The path the artifact was written to.
    """
    return atomic_write_json(path, redact_artifact(artifact.to_dict(), project_root))


def _metric_gate(
    cohort: str,
    metric: str,
    current: MetricValue,
    baseline: MetricValue,
    tolerances: dict[str, MetricTolerance],
) -> GateResult:
    """Compare one metric for one cohort against the baseline value.

    Args:
        cohort: Cohort name (``overall`` for the aggregate).
        metric: Metric name.
        current: Current metric value.
        baseline: Baseline metric value.
        tolerances: Tolerance table (defaults to ``DEFAULT_TOLERANCES``).

    Returns:
        The resulting GateResult.
    """
    direction = METRIC_DIRECTIONS.get(metric, "higher")
    tolerance = tolerances.get(metric, MetricTolerance(absolute=0.02, relative=0.02))
    if not current.is_available or not baseline.is_available:
        reason = current.reason if not current.is_available else baseline.reason
        return GateResult(
            cohort=cohort,
            metric=metric,
            baseline=baseline.value,
            current=current.value,
            delta=None,
            direction=direction,
            absolute_tolerance=tolerance.absolute,
            relative_tolerance=tolerance.relative,
            status="unavailable",
            detail=reason,
        )
    assert current.value is not None and baseline.value is not None
    delta = current.value - baseline.value
    allowed = max(tolerance.absolute, tolerance.relative * abs(baseline.value))
    worse = delta < -allowed if direction == "higher" else delta > allowed
    return GateResult(
        cohort=cohort,
        metric=metric,
        baseline=baseline.value,
        current=current.value,
        delta=delta,
        direction=direction,
        absolute_tolerance=tolerance.absolute,
        relative_tolerance=tolerance.relative,
        status="regression" if worse else "pass",
    )


def _identity_block(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract an identity block from a baseline payload defensively.

    Args:
        payload: Baseline artifact payload.
        key: Identity block key.

    Returns:
        The block dict, or an empty dict when absent/malformed.
    """
    block = payload.get(key)
    return block if isinstance(block, dict) else {}


def compare_with_baseline(
    current: EvaluationArtifact,
    baseline_payload: dict[str, Any],
    tolerances: dict[str, MetricTolerance] | None = None,
) -> ComparisonBlock:
    """Compare a fresh artifact against a baseline artifact payload.

    Compatibility is checked first: artifact schema, evaluation kind,
    benchmark fingerprint, configuration fingerprint, case set, and cohort
    set. Incompatible artifacts are never compared as equivalent; their
    status is ``incompatible`` with every mismatch named. Gate comparison
    covers ``overall`` and every shared cohort for the gated metric set
    (latency is reported in deltas but intentionally not gated).

    Args:
        current: The freshly produced artifact.
        baseline_payload: Parsed baseline artifact JSON.
        tolerances: Optional tolerance table override.

    Returns:
        The ComparisonBlock embedded in the current artifact.
    """
    tolerances = tolerances if tolerances is not None else DEFAULT_TOLERANCES
    incompatibilities: list[str] = []

    baseline_schema = baseline_payload.get("schema_version")
    if baseline_schema != EVALUATION_SCHEMA_VERSION:
        incompatibilities.append(
            f"artifact schema_version {baseline_schema} != {EVALUATION_SCHEMA_VERSION}; rebuild the baseline"
        )
    baseline_kind = baseline_payload.get("kind")
    if baseline_kind is not None and baseline_kind != current.kind:
        incompatibilities.append(f"evaluation kind {baseline_kind!r} != {current.kind!r}")

    current_benchmark = dict(current.benchmark)
    baseline_benchmark = _identity_block(baseline_payload, "benchmark")
    if current_benchmark.get("fingerprint") != baseline_benchmark.get("fingerprint"):
        incompatibilities.append(
            "benchmark fingerprint mismatch: the baseline was produced from a different benchmark document"
        )
    current_configuration = dict(current.configuration)
    baseline_configuration = _identity_block(baseline_payload, "configuration")
    if current_configuration.get("fingerprint") != baseline_configuration.get("fingerprint"):
        incompatibilities.append("configuration fingerprint mismatch: embedding identity or retrieval settings changed")

    current_cases = sorted(run.case_id for run in current.runs)
    baseline_cases = sorted(
        str(run["case_id"]) for run in baseline_payload.get("runs", []) if isinstance(run, dict) and "case_id" in run
    )
    if current_cases != baseline_cases:
        removed = sorted(set(baseline_cases) - set(current_cases))
        added = sorted(set(current_cases) - set(baseline_cases))
        detail = []
        if removed:
            detail.append(f"missing from current: {', '.join(removed)}")
        if added:
            detail.append(f"new in current: {', '.join(added)}")
        incompatibilities.append("benchmark case set differs (" + "; ".join(detail) + ")")

    baseline_aggregates = baseline_payload.get("aggregates") or {}
    baseline_cohorts = baseline_aggregates.get("by_cohort") or {}
    current_cohorts = current.aggregates["by_cohort"]
    if sorted(baseline_cohorts) != sorted(current_cohorts):
        incompatibilities.append(
            f"cohort set differs: baseline {sorted(baseline_cohorts)} vs current {sorted(current_cohorts)}"
        )

    gates: list[GateResult] = []
    if not incompatibilities:
        baseline_overall = CohortMetricsBlock.from_dict(baseline_aggregates.get("overall") or {})
        gates.extend(_compare_blocks("overall", current.aggregates["overall"], baseline_overall, tolerances))
        for cohort in sorted(current_cohorts):
            baseline_block = CohortMetricsBlock.from_dict(baseline_cohorts.get(cohort) or {})
            gates.extend(_compare_blocks(cohort, current_cohorts[cohort], baseline_block, tolerances))

    if incompatibilities:
        status = "incompatible"
    elif any(gate.status == "regression" for gate in gates):
        status = "regression"
    else:
        status = "pass"

    baseline_identity = {
        "run_id": baseline_payload.get("run_id"),
        "created_at": baseline_payload.get("created_at"),
        "benchmark_fingerprint": baseline_benchmark.get("fingerprint"),
        "configuration_fingerprint": baseline_configuration.get("fingerprint"),
    }
    return ComparisonBlock(
        baseline=baseline_identity,
        compatible=not incompatibilities,
        incompatibilities=incompatibilities,
        gates=gates,
        status=status,
    )


def _compare_blocks(
    cohort: str,
    current: CohortMetricsBlock,
    baseline: CohortMetricsBlock,
    tolerances: dict[str, MetricTolerance],
) -> list[GateResult]:
    """Compare two metrics blocks for the gated metric set.

    Args:
        cohort: Cohort label for the gates.
        current: Current metrics block.
        baseline: Baseline metrics block.
        tolerances: Tolerance table.

    Returns:
        Gate results for every metric present in either block.
    """
    gates: list[GateResult] = []
    metric_names = sorted(set(current.metrics) | set(baseline.metrics))
    for metric in metric_names:
        if metric not in GATED_METRICS:
            continue
        gates.append(
            _metric_gate(
                cohort,
                metric,
                current.metrics.get(metric, MetricValue.unavailable("metric not present in current artifact")),
                baseline.metrics.get(metric, MetricValue.unavailable("metric not present in baseline artifact")),
                tolerances,
            )
        )
    return gates
