"""Evaluation artifact store and run-comparison service (RE-03).

:class:`EvaluationOperations` is the single access layer for immutable
retrieval evaluation artifacts (the dashboard's only path to them): it
resolves a project-contained artifact root, validates artifact ids against
traversal, lists stored runs with corrupt-file diagnostics, fetches run
payloads, and compares two runs.

The comparison core (:func:`compare_retrieval_payloads`) reuses the RE-01
baseline machinery (``compare_with_baseline``) for gated quality/correctness/
efficiency metrics and adds the maintainer-facing view on top: dimension
classification (quality, efficiency, correctness, noisy timing), reported
buts-never-gated latency deltas, case-level newly passing/failing
identification, and rebuild/rerun actions for every incompatibility.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    DEFAULT_TOLERANCES,
    GATED_METRICS,
    METRIC_DIRECTIONS,
    CaseRunRecord,
    CohortMetricsBlock,
    EvaluationArtifact,
    GateResult,
    MetricTolerance,
    compare_with_baseline,
    evaluations_dir,
)
from .common import EvalError, MetricValue

# Artifact ids are single safe path components: alphanumerics plus '.', '_',
# '-' with no leading dot, so ``..``/``.hidden`` and separators are rejected.
ARTIFACT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")

# Fixed comparison status for reported-but-never-gated metrics (latency).
STATUS_REPORTED = "reported"

# Metric dimension classification for the comparison surfaces. Timing
# metrics are noisy wall-clock measurements: reported, never gated.
METRIC_DIMENSIONS: dict[str, str] = {
    "successful_query_rate": "correctness",
    "recall@1": "quality",
    "recall@5": "quality",
    "recall@10": "quality",
    "mrr": "quality",
    "ndcg@10": "quality",
    "evidence_precision@5": "quality",
    "duplicate_token_ratio": "efficiency",
    "selected_token_mean": "efficiency",
    "selected_token_p95": "efficiency",
    "latency_p50_ms": "timing",
    "latency_p95_ms": "timing",
}

# Rebuild/rerun guidance matched against the shared incompatibility wording
# (see ``artifacts.compare_evaluation_payloads``). Order is presentation order.
_ACTION_RULES: list[tuple[str, str]] = [
    (
        "schema_version",
        "Rebuild the baseline: re-run the benchmark with the current artifact schema and deliberately "
        "refresh the stored baseline (see docs/RETRIEVAL_BENCHMARK.md).",
    ),
    (
        "evaluation kind",
        "Re-run with the same evaluation kind; retrieval and agent artifacts are not comparable.",
    ),
    (
        "benchmark fingerprint",
        "Re-run both evaluations against the same benchmark document; if the benchmark change is "
        "intended, refresh the baseline as a separate reviewed change.",
    ),
    (
        "configuration fingerprint",
        "Re-run the candidate with the baseline's embedding identity and retrieval settings; if the "
        "configuration change is intended, refresh the baseline as a separate reviewed change.",
    ),
    (
        "case set",
        "Re-run the evaluation so both artifacts cover the same benchmark case set.",
    ),
    (
        "cohort set",
        "Re-run the evaluation so both artifacts report the same cohort set.",
    ),
]


class EvaluationStoreError(EvalError):
    """Base class for evaluation-artifact store errors."""


class InvalidArtifactIdError(EvaluationStoreError):
    """Raised when an artifact id is not a safe single path component."""


class ArtifactNotFoundError(EvaluationStoreError):
    """Raised when an evaluation artifact does not exist in the store."""


class ArtifactCorruptError(EvaluationStoreError):
    """Raised when an evaluation artifact cannot be parsed."""


class RootOutsideProjectError(EvaluationStoreError):
    """Raised when a configured artifact root escapes the project boundary."""


def is_valid_artifact_id(value: object) -> bool:
    """Check whether an artifact id is a safe single path component.

    Args:
        value: The candidate artifact id.

    Returns:
        True when the id is a string matching the safe-id pattern.
    """
    return isinstance(value, str) and bool(ARTIFACT_ID_PATTERN.fullmatch(value))


def metric_dimension(metric: str) -> str:
    """Classify a metric into its comparison dimension.

    Args:
        metric: Metric name.

    Returns:
        ``quality``, ``correctness``, ``efficiency``, ``timing``, or
        ``other`` for metrics outside the curated table.
    """
    return METRIC_DIMENSIONS.get(metric, "other")


@dataclass(frozen=True)
class EvaluationRunSummary:
    """Projection of one stored evaluation artifact for list views.

    Attributes:
        run_id: The recorded run id (falls back to the file stem).
        file_id: Artifact file stem inside the artifact root.
        created_at: ISO-8601 creation timestamp (volatile).
        status: ``complete`` or ``partial``.
        benchmark_name: Benchmark document name.
        benchmark_fingerprint: Content-derived benchmark fingerprint.
        configuration_fingerprint: Content-derived configuration fingerprint.
        index_name: The index the run evaluated against.
        case_count: Number of case records in the artifact.
        graph_enabled: Whether the run used graph expansion (IG-03).
        comparison_status: Status of the artifact's embedded baseline
            comparison (``None`` when the run had no baseline).
    """

    run_id: str
    file_id: str
    created_at: str
    status: str
    benchmark_name: str
    benchmark_fingerprint: str
    configuration_fingerprint: str
    index_name: str | None
    case_count: int
    graph_enabled: bool
    comparison_status: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the ``--json`` envelope shape.
        """
        return {
            "run_id": self.run_id,
            "file_id": self.file_id,
            "created_at": self.created_at,
            "status": self.status,
            "benchmark_name": self.benchmark_name,
            "benchmark_fingerprint": self.benchmark_fingerprint,
            "configuration_fingerprint": self.configuration_fingerprint,
            "index_name": self.index_name,
            "case_count": self.case_count,
            "graph_enabled": self.graph_enabled,
            "comparison_status": self.comparison_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRunSummary:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed EvaluationRunSummary.
        """
        index_name = data.get("index_name")
        comparison_status = data.get("comparison_status")
        return cls(
            run_id=str(data["run_id"]),
            file_id=str(data["file_id"]),
            created_at=str(data["created_at"]),
            status=str(data["status"]),
            benchmark_name=str(data["benchmark_name"]),
            benchmark_fingerprint=str(data["benchmark_fingerprint"]),
            configuration_fingerprint=str(data["configuration_fingerprint"]),
            index_name=str(index_name) if index_name is not None else None,
            case_count=int(data["case_count"]),
            graph_enabled=bool(data["graph_enabled"]),
            comparison_status=str(comparison_status) if comparison_status is not None else None,
        )


@dataclass(frozen=True)
class MetricDelta:
    """One metric delta for one cohort between two evaluation runs.

    Attributes:
        cohort: Cohort label (``overall`` for the aggregate).
        metric: Metric name.
        dimension: Comparison dimension (quality/correctness/efficiency/
            timing/other).
        gated: Whether the delta participates in regression gating.
        baseline: Baseline value (``None`` when unavailable).
        current: Candidate value (``None`` when unavailable).
        delta: ``current - baseline`` when both are available.
        direction: ``higher`` or ``lower`` (which direction is better).
        absolute_tolerance: Checked-in absolute tolerance.
        relative_tolerance: Checked-in relative tolerance.
        status: ``pass``, ``regression``, ``unavailable`` (gated metrics) or
            ``reported`` (never-gated timing metrics).
        trend: ``improved``, ``regressed``, ``flat`` (``None`` unavailable).
        detail: Extra context, e.g. the unavailability reason.
    """

    cohort: str
    metric: str
    dimension: str
    gated: bool
    baseline: float | None
    current: float | None
    delta: float | None
    direction: str
    absolute_tolerance: float
    relative_tolerance: float
    status: str
    trend: str | None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the comparison envelope schema.
        """
        return {
            "cohort": self.cohort,
            "metric": self.metric,
            "dimension": self.dimension,
            "gated": self.gated,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "direction": self.direction,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "status": self.status,
            "trend": self.trend,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricDelta:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed MetricDelta.
        """
        return cls(
            cohort=str(data["cohort"]),
            metric=str(data["metric"]),
            dimension=str(data["dimension"]),
            gated=bool(data["gated"]),
            baseline=_optional_float(data.get("baseline")),
            current=_optional_float(data.get("current")),
            delta=_optional_float(data.get("delta")),
            direction=str(data["direction"]),
            absolute_tolerance=float(data["absolute_tolerance"]),
            relative_tolerance=float(data["relative_tolerance"]),
            status=str(data["status"]),
            trend=data.get("trend"),
            detail=data.get("detail"),
        )


@dataclass(frozen=True)
class CaseDelta:
    """Per-case outcome change between two compatible evaluation runs.

    Attributes:
        case_id: Benchmark case identifier.
        cohort: Benchmark cohort label.
        split: Benchmark split label.
        kind: ``newly_passing``, ``newly_failing``, ``improved``
            (first relevant rank decreased), ``worsened`` (increased), or
            ``unchanged``.
        baseline_status: Baseline case status (``ok``/``error``).
        candidate_status: Candidate case status.
        baseline_first_relevant_rank: Baseline rank of the first relevant hit.
        candidate_first_relevant_rank: Candidate rank of the first relevant hit.
        baseline_tokens: Baseline selected-context token estimate.
        candidate_tokens: Candidate selected-context token estimate.
        delta_tokens: ``candidate_tokens - baseline_tokens``.
        baseline_latency_ms: Mean baseline per-case latency (volatile).
        candidate_latency_ms: Mean candidate per-case latency (volatile).
    """

    case_id: str
    cohort: str
    split: str
    kind: str
    baseline_status: str
    candidate_status: str
    baseline_first_relevant_rank: int | None
    candidate_first_relevant_rank: int | None
    baseline_tokens: int
    candidate_tokens: int
    delta_tokens: int
    baseline_latency_ms: float | None
    candidate_latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the comparison envelope schema.
        """
        return {
            "case_id": self.case_id,
            "cohort": self.cohort,
            "split": self.split,
            "kind": self.kind,
            "baseline_status": self.baseline_status,
            "candidate_status": self.candidate_status,
            "baseline_first_relevant_rank": self.baseline_first_relevant_rank,
            "candidate_first_relevant_rank": self.candidate_first_relevant_rank,
            "baseline_tokens": self.baseline_tokens,
            "candidate_tokens": self.candidate_tokens,
            "delta_tokens": self.delta_tokens,
            "baseline_latency_ms": self.baseline_latency_ms,
            "candidate_latency_ms": self.candidate_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseDelta:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed CaseDelta.
        """
        return cls(
            case_id=str(data["case_id"]),
            cohort=str(data["cohort"]),
            split=str(data["split"]),
            kind=str(data["kind"]),
            baseline_status=str(data["baseline_status"]),
            candidate_status=str(data["candidate_status"]),
            baseline_first_relevant_rank=_optional_int(data.get("baseline_first_relevant_rank")),
            candidate_first_relevant_rank=_optional_int(data.get("candidate_first_relevant_rank")),
            baseline_tokens=int(data["baseline_tokens"]),
            candidate_tokens=int(data["candidate_tokens"]),
            delta_tokens=int(data["delta_tokens"]),
            baseline_latency_ms=_optional_float(data.get("baseline_latency_ms")),
            candidate_latency_ms=_optional_float(data.get("candidate_latency_ms")),
        )


@dataclass(frozen=True)
class RunComparison:
    """Full maintainer comparison of two evaluation runs.

    Attributes:
        baseline: Baseline identity (run id, timestamps, fingerprints).
        candidate: Candidate identity.
        compatible: Whether the two artifacts may be compared as equivalent.
        incompatibilities: Every named identity mismatch.
        actions: Rebuild/rerun guidance derived from the mismatches.
        gates: Metric deltas per cohort (gated and reported metrics).
        case_deltas: Per-case outcome changes (empty when incompatible).
        status: ``pass``, ``regression``, or ``incompatible``.
    """

    baseline: dict[str, Any]
    candidate: dict[str, Any]
    compatible: bool
    incompatibilities: list[str]
    actions: list[str]
    gates: list[MetricDelta]
    case_deltas: list[CaseDelta]
    status: str

    @property
    def newly_passing(self) -> list[str]:
        """Case ids that pass in the candidate but failed in the baseline."""
        return sorted(delta.case_id for delta in self.case_deltas if delta.kind == "newly_passing")

    @property
    def newly_failing(self) -> list[str]:
        """Case ids that fail in the candidate but passed in the baseline."""
        return sorted(delta.case_id for delta in self.case_deltas if delta.kind == "newly_failing")

    @property
    def regressions(self) -> list[MetricDelta]:
        """Gates that regressed beyond tolerance."""
        return [gate for gate in self.gates if gate.status == "regression"]

    def worst_regressions(self, limit: int = 10) -> list[MetricDelta]:
        """Regressions ordered by magnitude of the delta.

        Args:
            limit: Maximum number of gates to return.

        Returns:
            The worst regression gates, largest delta magnitude first.
        """
        ordered = sorted(
            self.regressions,
            key=lambda gate: -(abs(gate.delta) if gate.delta is not None else 0.0),
        )
        return ordered[: max(0, limit)]

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the comparison envelope schema.
        """
        return {
            "baseline": dict(self.baseline),
            "candidate": dict(self.candidate),
            "compatible": self.compatible,
            "incompatibilities": list(self.incompatibilities),
            "actions": list(self.actions),
            "gates": [gate.to_dict() for gate in self.gates],
            "case_deltas": [delta.to_dict() for delta in self.case_deltas],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunComparison:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed RunComparison.
        """
        return cls(
            baseline=dict(data.get("baseline") or {}),
            candidate=dict(data.get("candidate") or {}),
            compatible=bool(data["compatible"]),
            incompatibilities=list(data.get("incompatibilities", [])),
            actions=list(data.get("actions", [])),
            gates=[MetricDelta.from_dict(gate) for gate in data.get("gates", [])],
            case_deltas=[CaseDelta.from_dict(delta) for delta in data.get("case_deltas", [])],
            status=str(data["status"]),
        )


def _optional_float(value: Any) -> float | None:
    """Coerce an optional JSON number to float.

    Args:
        value: Raw JSON value (may be None).

    Returns:
        The float value, or None.
    """
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    """Coerce an optional JSON number to int.

    Args:
        value: Raw JSON value (may be None).

    Returns:
        The int value, or None.
    """
    return int(value) if value is not None else None


def _identity_block(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract an identity block defensively.

    Args:
        payload: Artifact payload.
        key: Identity block key.

    Returns:
        The block dict, or an empty dict when absent/malformed.
    """
    block = payload.get(key)
    return block if isinstance(block, dict) else {}


def _payload_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the comparison identity summary for one artifact payload.

    Args:
        payload: Parsed artifact JSON.

    Returns:
        Identity dict with run id, timestamp, and both fingerprints.
    """
    benchmark = _identity_block(payload, "benchmark")
    configuration = _identity_block(payload, "configuration")
    return {
        "run_id": payload.get("run_id"),
        "created_at": payload.get("created_at"),
        "benchmark_fingerprint": benchmark.get("fingerprint"),
        "configuration_fingerprint": configuration.get("fingerprint"),
    }


def _trend(direction: str, delta: float | None) -> str | None:
    """Classify a delta against its metric direction.

    Args:
        direction: ``higher`` or ``lower`` (which direction is better).
        delta: ``current - baseline`` (``None`` when unavailable).

    Returns:
        ``improved``, ``regressed``, ``flat``, or ``None``.
    """
    if delta is None:
        return None
    if abs(delta) <= 1e-12:
        return "flat"
    better = delta > 0 if direction == "higher" else delta < 0
    return "improved" if better else "regressed"


def _case_pass(run: CaseRunRecord) -> bool:
    """Whether a case run found its relevant evidence.

    Args:
        run: The case run record.

    Returns:
        True when the case executed without error and found a relevant hit.
    """
    return run.status == "ok" and run.first_relevant_rank is not None


def _mean_latency_ms(run: CaseRunRecord) -> float | None:
    """Mean measured latency of one case run (volatile, reported only).

    Args:
        run: The case run record.

    Returns:
        The mean of the measured values in milliseconds, or ``None``.
    """
    values = run.latency.get("values_ms") if isinstance(run.latency, dict) else None
    if not values:
        return None
    measured = [float(value) for value in values if isinstance(value, (int, float))]
    if not measured:
        return None
    return sum(measured) / len(measured)


def _case_delta(baseline_run: CaseRunRecord, candidate_run: CaseRunRecord) -> CaseDelta:
    """Build the per-case delta between two compatible runs of one case.

    Args:
        baseline_run: Baseline case record.
        candidate_run: Candidate case record.

    Returns:
        The CaseDelta with kind and rank/token/latency changes.
    """
    baseline_pass = _case_pass(baseline_run)
    candidate_pass = _case_pass(candidate_run)
    if not baseline_pass and candidate_pass:
        kind = "newly_passing"
    elif baseline_pass and not candidate_pass:
        kind = "newly_failing"
    elif baseline_pass and candidate_pass:
        assert baseline_run.first_relevant_rank is not None
        assert candidate_run.first_relevant_rank is not None
        if candidate_run.first_relevant_rank < baseline_run.first_relevant_rank:
            kind = "improved"
        elif candidate_run.first_relevant_rank > baseline_run.first_relevant_rank:
            kind = "worsened"
        else:
            kind = "unchanged"
    else:
        kind = "unchanged"
    return CaseDelta(
        case_id=baseline_run.case_id,
        cohort=baseline_run.cohort,
        split=baseline_run.split,
        kind=kind,
        baseline_status=baseline_run.status,
        candidate_status=candidate_run.status,
        baseline_first_relevant_rank=baseline_run.first_relevant_rank,
        candidate_first_relevant_rank=candidate_run.first_relevant_rank,
        baseline_tokens=baseline_run.estimated_tokens,
        candidate_tokens=candidate_run.estimated_tokens,
        delta_tokens=candidate_run.estimated_tokens - baseline_run.estimated_tokens,
        baseline_latency_ms=_mean_latency_ms(baseline_run),
        candidate_latency_ms=_mean_latency_ms(candidate_run),
    )


def _gate_to_delta(gate: GateResult) -> MetricDelta:
    """Convert a shared GateResult into a dimension-classified MetricDelta.

    Args:
        gate: The gate from the RE-01 comparison machinery.

    Returns:
        The MetricDelta for the comparison envelope.
    """
    return MetricDelta(
        cohort=gate.cohort,
        metric=gate.metric,
        dimension=metric_dimension(gate.metric),
        gated=True,
        baseline=gate.baseline,
        current=gate.current,
        delta=gate.delta,
        direction=gate.direction,
        absolute_tolerance=gate.absolute_tolerance,
        relative_tolerance=gate.relative_tolerance,
        status=gate.status,
        trend=_trend(gate.direction, gate.delta),
        detail=gate.detail,
    )


def _reported_delta(
    cohort: str,
    metric: str,
    current: MetricValue | None,
    baseline: MetricValue | None,
    tolerances: dict[str, MetricTolerance],
) -> MetricDelta:
    """Build the reported (never-gated) delta for one non-gated metric.

    Args:
        cohort: Cohort label.
        metric: Metric name.
        current: Candidate metric entry (``None`` when absent).
        baseline: Baseline metric entry (``None`` when absent).
        tolerances: Tolerance table (used only for display values).

    Returns:
        A MetricDelta with status ``reported`` or ``unavailable``.
    """
    direction = METRIC_DIRECTIONS.get(metric, "higher")
    tolerance = tolerances.get(metric, MetricTolerance(absolute=0.0, relative=0.0))
    if current is None or not current.is_available:
        return MetricDelta(
            cohort=cohort,
            metric=metric,
            dimension=metric_dimension(metric),
            gated=False,
            baseline=baseline.value if baseline else None,
            current=None,
            delta=None,
            direction=direction,
            absolute_tolerance=tolerance.absolute,
            relative_tolerance=tolerance.relative,
            status="unavailable",
            trend=None,
            detail=current.reason if current else "metric not present in current artifact",
        )
    if baseline is None or not baseline.is_available:
        return MetricDelta(
            cohort=cohort,
            metric=metric,
            dimension=metric_dimension(metric),
            gated=False,
            baseline=None,
            current=current.value,
            delta=None,
            direction=direction,
            absolute_tolerance=tolerance.absolute,
            relative_tolerance=tolerance.relative,
            status="unavailable",
            trend=None,
            detail=baseline.reason if baseline else "metric not present in baseline artifact",
        )
    assert current.value is not None and baseline.value is not None
    delta = current.value - baseline.value
    return MetricDelta(
        cohort=cohort,
        metric=metric,
        dimension=metric_dimension(metric),
        gated=False,
        baseline=baseline.value,
        current=current.value,
        delta=delta,
        direction=direction,
        absolute_tolerance=tolerance.absolute,
        relative_tolerance=tolerance.relative,
        status=STATUS_REPORTED,
        trend=_trend(direction, delta),
        detail=None,
    )


def _actions_for(incompatibilities: list[str]) -> list[str]:
    """Derive deduplicated rebuild/rerun actions from incompatibilities.

    Args:
        incompatibilities: Named identity mismatches.

    Returns:
        Guidance actions in presentation order.
    """
    actions: list[str] = []
    for needle, action in _ACTION_RULES:
        if any(needle in item for item in incompatibilities) and action not in actions:
            actions.append(action)
    return actions


def compare_retrieval_payloads(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    tolerances: dict[str, MetricTolerance] | None = None,
) -> RunComparison:
    """Compare two retrieval evaluation artifacts for maintainers (RE-03).

    Compatibility and gated regression detection are delegated to the shared
    RE-01 machinery (``compare_with_baseline``). On top of it this adds
    reported-but-never-gated timing deltas, dimension classification, and
    case-level newly passing/failing identification (compatible runs only).

    Args:
        baseline_payload: Parsed baseline artifact JSON.
        candidate_payload: Parsed candidate artifact JSON.
        tolerances: Optional tolerance override (defaults to checked-in).

    Returns:
        The RunComparison for CLI and dashboard rendering.

    Raises:
        ArtifactCorruptError: When either payload is not a valid artifact.
    """
    tolerances = tolerances if tolerances is not None else DEFAULT_TOLERANCES
    try:
        candidate = EvaluationArtifact.from_dict(candidate_payload)
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactCorruptError(f"candidate artifact payload is corrupt: {exc}") from exc
    try:
        baseline_runs = {
            str(run["case_id"]): CaseRunRecord.from_dict(run)
            for run in baseline_payload.get("runs", [])
            if isinstance(run, dict)
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise ArtifactCorruptError(f"baseline artifact payload is corrupt: {exc}") from exc

    comparison = compare_with_baseline(candidate, baseline_payload, tolerances=tolerances)

    gates: list[MetricDelta] = []
    case_deltas: list[CaseDelta] = []
    if comparison.compatible:
        gate_index = {(gate.cohort, gate.metric): gate for gate in comparison.gates}
        baseline_aggregates = _identity_block(baseline_payload, "aggregates")
        baseline_cohorts = _identity_block(baseline_aggregates, "by_cohort")
        candidate_cohorts = candidate.aggregates.get("by_cohort") or {}
        for cohort in ["overall", *sorted(candidate_cohorts)]:
            if cohort == "overall":
                current_block = candidate.aggregates["overall"]
                baseline_block = CohortMetricsBlock.from_dict(baseline_aggregates.get("overall") or {})
            else:
                current_block = candidate_cohorts[cohort]
                baseline_block = CohortMetricsBlock.from_dict(baseline_cohorts.get(cohort) or {})
            for metric in sorted(set(current_block.metrics) | set(baseline_block.metrics)):
                if metric in GATED_METRICS:
                    gate = gate_index.get((cohort, metric))
                    if gate is not None:  # pragma: no branch - shared gate set matches
                        gates.append(_gate_to_delta(gate))
                else:
                    gates.append(
                        _reported_delta(
                            cohort,
                            metric,
                            current_block.metrics.get(metric),
                            baseline_block.metrics.get(metric),
                            tolerances,
                        )
                    )

        candidate_runs = {run.case_id: run for run in candidate.runs}
        for case_id in sorted(candidate_runs):
            baseline_run = baseline_runs.get(case_id)
            if baseline_run is not None:
                case_deltas.append(_case_delta(baseline_run, candidate_runs[case_id]))

    return RunComparison(
        baseline=_payload_identity(baseline_payload),
        candidate=_payload_identity(candidate_payload),
        compatible=comparison.compatible,
        incompatibilities=list(comparison.incompatibilities),
        actions=_actions_for(comparison.incompatibilities),
        gates=gates,
        case_deltas=case_deltas,
        status=comparison.status,
    )


class EvaluationOperations:
    """Project-scoped access layer over immutable evaluation artifacts.

    The dashboard reads evaluation artifacts exclusively through this
    service: artifact ids are validated against traversal, the artifact root
    is project-contained (default ``<project>/.ctxai/evaluations/retrieval``),
    and corrupt files surface as diagnostics instead of failures.
    """

    def __init__(self, project_root: Path | None = None, artifact_root: Path | None = None) -> None:
        """Resolve and validate the artifact root.

        Args:
            project_root: Project root (defaults to the current directory).
            artifact_root: Optional artifact-directory override; it must stay
                inside the project boundary.

        Raises:
            RootOutsideProjectError: When the override escapes the project.
        """
        root = (project_root or Path.cwd()).resolve()
        self.project_root = root
        if artifact_root is None:
            self.artifact_root = evaluations_dir(root)
            return
        candidate = Path(artifact_root).expanduser()
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise RootOutsideProjectError(
                f"Artifact root {artifact_root} is outside the project boundary; evaluation artifacts"
                f" must stay inside {root}"
            )
        self.artifact_root = resolved

    def _load_payload_file(self, path: Path) -> dict[str, Any]:
        """Parse one artifact file.

        Args:
            path: Artifact JSON path inside the artifact root.

        Returns:
            The parsed payload.

        Raises:
            ArtifactCorruptError: When the file is unreadable or not a JSON
                object.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactCorruptError(f"corrupt evaluation artifact {path.name}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ArtifactCorruptError(f"corrupt evaluation artifact {path.name}: not a JSON object")
        return payload

    def list_runs(self, limit: int = 100) -> tuple[list[EvaluationRunSummary], list[str]]:
        """List stored evaluation runs, newest first.

        Corrupt artifact files are skipped with a diagnostic instead of
        failing the listing.

        Args:
            limit: Maximum number of summaries to return.

        Returns:
            ``(summaries, corrupt_diagnostics)``.
        """
        if not self.artifact_root.is_dir():
            return [], []
        summaries: list[EvaluationRunSummary] = []
        corrupt: list[str] = []
        for path in sorted(self.artifact_root.glob("*.json")):
            if not path.is_file():
                continue
            try:
                payload = self._load_payload_file(path)
            except ArtifactCorruptError as error:
                corrupt.append(str(error))
                continue
            summaries.append(_summary_from_payload(payload, path.stem))
        summaries.sort(key=lambda summary: summary.created_at or "", reverse=True)
        return summaries[: max(0, limit)], corrupt

    def read_run(self, run_id: str) -> dict[str, Any]:
        """Fetch one evaluation artifact payload by id.

        The id may be the artifact file stem or the recorded ``run_id``.

        Args:
            run_id: Artifact id (validated against traversal).

        Returns:
            The parsed artifact payload.

        Raises:
            InvalidArtifactIdError: When the id is not a safe component.
            ArtifactNotFoundError: When no artifact matches.
            ArtifactCorruptError: When a matching file cannot be parsed.
        """
        if not is_valid_artifact_id(run_id):
            raise InvalidArtifactIdError(f"Invalid evaluation artifact id: {run_id!r}")
        direct = self.artifact_root / f"{run_id}.json"
        if direct.is_file():
            return self._load_payload_file(direct)
        if self.artifact_root.is_dir():
            for path in sorted(self.artifact_root.glob("*.json")):
                if not path.is_file():
                    continue
                try:
                    payload = self._load_payload_file(path)
                except ArtifactCorruptError:
                    continue
                if payload.get("run_id") == run_id:
                    return payload
        raise ArtifactNotFoundError(f"No evaluation artifact for '{run_id}' under {self.artifact_root}")

    def compare_runs(self, baseline_id: str, candidate_id: str) -> RunComparison:
        """Compare two stored evaluation runs.

        Args:
            baseline_id: Artifact id of the baseline run.
            candidate_id: Artifact id of the candidate run.

        Returns:
            The RunComparison for the two artifacts.

        Raises:
            InvalidArtifactIdError: When either id is not a safe component.
            ArtifactNotFoundError: When either artifact is missing.
            ArtifactCorruptError: When either payload cannot be parsed.
        """
        baseline = self.read_run(baseline_id)
        candidate = self.read_run(candidate_id)
        return compare_retrieval_payloads(baseline, candidate)


def _summary_from_payload(payload: dict[str, Any], file_id: str) -> EvaluationRunSummary:
    """Project an artifact payload onto a list-view summary.

    Args:
        payload: Parsed artifact JSON.
        file_id: Artifact file stem.

    Returns:
        The EvaluationRunSummary for the payload.
    """
    benchmark = _identity_block(payload, "benchmark")
    configuration = _identity_block(payload, "configuration")
    index = _identity_block(payload, "index")
    graph_block = _identity_block(configuration, "graph_expansion")
    comparison = payload.get("comparison")
    runs = payload.get("runs")
    run_count = len(runs) if isinstance(runs, list) else None
    index_name = index.get("name")
    return EvaluationRunSummary(
        run_id=str(payload.get("run_id") or file_id),
        file_id=file_id,
        created_at=str(payload.get("created_at") or ""),
        status=str(payload.get("status") or ""),
        benchmark_name=str(benchmark.get("name") or ""),
        benchmark_fingerprint=str(benchmark.get("fingerprint") or ""),
        configuration_fingerprint=str(configuration.get("fingerprint") or ""),
        index_name=str(index_name) if index_name is not None else None,
        case_count=run_count if run_count is not None else int(benchmark.get("case_count") or 0),
        graph_enabled=bool(graph_block.get("enabled")),
        comparison_status=str(comparison.get("status")) if isinstance(comparison, dict) else None,
    )
