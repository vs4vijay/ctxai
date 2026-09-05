"""Versioned agent task evaluation artifacts and gates (HH-09).

An ``AgentEvalArtifact`` is an immutable, redacted JSON record of one agent
benchmark run: benchmark and configuration identity, per-case judgments with
HH-04 transcript evidence, aggregate metrics per cohort/split, and an
optional baseline comparison with named gates. It deliberately shares the
RE-01 artifact vocabulary — :class:`MetricValue`, :class:`CohortMetricsBlock`,
:class:`GateResult`, :class:`ComparisonBlock`, the fingerprinting and
redaction pipeline, and the comparison code — so both eval frameworks use one
artifact discipline by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    EVALUATION_SCHEMA_VERSION,
    CohortMetricsBlock,
    ComparisonBlock,
    MetricTolerance,
    compare_evaluation_payloads,
)
from .common import MetricValue, atomic_write_json, content_fingerprint, redact_artifact

EVALUATION_KIND_AGENT = "agent"

# Metric direction: "higher" metrics regress downward, "lower" metrics regress
# upward. Cost is reported but intentionally NOT gated: it depends on checked
# in price-table data and provider billing, not on agent behavior quality.
AGENT_METRIC_DIRECTIONS: dict[str, str] = {
    "pass_rate": "higher",
    "mean_iterations": "lower",
    "p95_iterations": "lower",
    "token_mean": "lower",
    "token_p95": "lower",
    "cost_total": "lower",
}

# Checked-in tolerances per metric: (absolute, relative-to-baseline). A gate
# regresses when the current value is worse than the baseline by more than
# max(absolute, relative * |baseline|). pass_rate uses zero tolerance: a
# deterministic benchmark either keeps every case passing or it regresses.
AGENT_GATED_METRICS = frozenset({"pass_rate", "mean_iterations", "p95_iterations", "token_mean"})

AGENT_TOLERANCES: dict[str, MetricTolerance] = {
    "pass_rate": MetricTolerance(absolute=0.0, relative=0.0),
    "mean_iterations": MetricTolerance(absolute=1.0, relative=0.2),
    "p95_iterations": MetricTolerance(absolute=1.0, relative=0.2),
    "token_mean": MetricTolerance(absolute=200.0, relative=0.2),
}


@dataclass(frozen=True)
class CaseJudgment:
    """One named scoring judgment for a benchmark case.

    Attributes:
        name: Judgment name (``checks``, ``forbidden_paths``, ``budget``,
            ``plan_workflow``, ``approvals``).
        passed: Whether the judgment passed.
        reason: Human-readable failure reason, or ``None`` when passed.
    """

    name: str
    passed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for one judgment.
        """
        return {"name": self.name, "passed": self.passed, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseJudgment:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed CaseJudgment.
        """
        return cls(
            name=str(data["name"]),
            passed=bool(data["passed"]),
            reason=data.get("reason"),
        )


@dataclass(frozen=True)
class AgentCaseRunRecord:
    """One benchmark case executed by the real agent loop.

    Attributes:
        case_id: Benchmark case identifier.
        run_id: Artifact run id.
        timestamp: ISO-8601 execution timestamp (volatile).
        cohort: Benchmark cohort label.
        split: Benchmark split label.
        status: ``passed``, ``failed`` (scored failure), or ``error`` (the
            case could not execute).
        error: First error message when the case errored, else ``None``.
        iterations: Iterations used (successful LLM calls).
        max_iterations: The case's iteration budget.
        tokens: Usage totals (``prompt_tokens``, ``completion_tokens``,
            ``total_tokens``, ``calls``) from the run's usage ledger.
        cost: Estimated cost for the case, explicitly unavailable when the
            model has no price-table entry.
        judgments: Named scoring judgments with reasons.
        checks: Per-check evidence (command, exit code, output marker).
        forbidden_paths: Per-path preservation evidence.
        plan: Planning workflow evidence (required/submitted/actions).
        approvals: Approval workflow evidence (mutation/approval pairing).
        changed_files: Repository-relative files the run mutated.
        transcript: HH-04 transcript evidence (path relative to the case
            project, transcript run id, event count — never full content).
    """

    case_id: str
    run_id: str
    timestamp: str
    cohort: str
    split: str
    status: str
    error: str | None
    iterations: int
    max_iterations: int
    tokens: dict[str, int]
    cost: MetricValue
    judgments: list[CaseJudgment]
    checks: list[dict[str, Any]]
    forbidden_paths: list[dict[str, Any]]
    plan: dict[str, Any]
    approvals: dict[str, Any]
    changed_files: list[str]
    transcript: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            Dictionary matching the artifact schema for one case run.
        """
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "cohort": self.cohort,
            "split": self.split,
            "status": self.status,
            "error": self.error,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "tokens": dict(self.tokens),
            "cost": self.cost.to_dict(),
            "judgments": [judgment.to_dict() for judgment in self.judgments],
            "checks": list(self.checks),
            "forbidden_paths": list(self.forbidden_paths),
            "plan": dict(self.plan),
            "approvals": dict(self.approvals),
            "changed_files": list(self.changed_files),
            "transcript": dict(self.transcript),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCaseRunRecord:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed AgentCaseRunRecord.

        Raises:
            ValueError: If the cost entry is not a valid MetricValue payload.
        """
        return cls(
            case_id=data["case_id"],
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            cohort=data["cohort"],
            split=data["split"],
            status=data["status"],
            error=data.get("error"),
            iterations=int(data.get("iterations", 0)),
            max_iterations=int(data.get("max_iterations", 0)),
            tokens={str(key): int(value) for key, value in (data.get("tokens") or {}).items()},
            cost=MetricValue.from_dict(data["cost"]),
            judgments=[CaseJudgment.from_dict(judgment) for judgment in data.get("judgments", [])],
            checks=list(data.get("checks", [])),
            forbidden_paths=list(data.get("forbidden_paths", [])),
            plan=dict(data.get("plan") or {}),
            approvals=dict(data.get("approvals") or {}),
            changed_files=list(data.get("changed_files", [])),
            transcript=dict(data.get("transcript") or {}),
        )


@dataclass(frozen=True)
class AgentEvalArtifact:
    """Immutable, versioned record of one agent benchmark evaluation run.

    Attributes:
        schema_version: Artifact schema version (currently 1).
        kind: Evaluation kind (``agent``).
        run_id: Unique id of this evaluation run.
        created_at: ISO-8601 creation timestamp (volatile).
        duration_ms: Total wall-clock duration (volatile).
        status: ``complete`` when every case executed, ``partial`` otherwise.
        benchmark: Benchmark identity (name, fingerprint, case count).
        configuration: Configuration identity (fingerprint, provider,
            approval policy).
        environment: Environment metadata (python, platform, version).
        runs: Per-case run records with judgments and transcript evidence.
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
    environment: dict[str, Any]
    runs: list[AgentCaseRunRecord]
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
    def from_dict(cls, data: dict[str, Any]) -> AgentEvalArtifact:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed AgentEvalArtifact.

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
            environment=data.get("environment") or {},
            runs=[AgentCaseRunRecord.from_dict(run) for run in data.get("runs", [])],
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


def agent_evaluations_dir(project_root: Path) -> Path:
    """Default immutable-artifact directory for agent task evaluations.

    Args:
        project_root: Resolved repository root.

    Returns:
        ``<project>/.ctxai/evaluations/agent``.
    """
    return project_root / ".ctxai" / "evaluations" / "agent"


def agent_workspaces_dir(project_root: Path) -> Path:
    """Per-run scratch workspace root holding each case's fixture project.

    Args:
        project_root: Resolved repository root.

    Returns:
        ``<project>/.ctxai/evaluations/agent/workspaces``.
    """
    return agent_evaluations_dir(project_root) / "workspaces"


def agent_default_artifact_path(project_root: Path, benchmark_name: str, run_id: str, created_at: str) -> Path:
    """Build the default artifact path from benchmark identity and time.

    Args:
        project_root: Resolved repository root.
        benchmark_name: Benchmark name (validated separately).
        run_id: Evaluation run id.
        created_at: ISO-8601 timestamp used for a sortable file prefix.

    Returns:
        Artifact path under :func:`agent_evaluations_dir`.
    """
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in benchmark_name)
    stamp = created_at.replace(":", "").replace("-", "")
    return agent_evaluations_dir(project_root) / f"{safe_name}-{stamp}-{run_id[:8]}.json"


def save_agent_artifact(artifact: AgentEvalArtifact, path: Path, project_root: Path) -> Path:
    """Redact and atomically persist an agent evaluation artifact.

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


def configuration_fingerprint(provider_identity: dict[str, Any], runner_settings: dict[str, Any]) -> str:
    """Content-derived fingerprint of the agent evaluation configuration.

    Args:
        provider_identity: Provider identity dict (mode, class name, model).
        runner_settings: Result-affecting runner settings (approval policy).

    Returns:
        Hex digest over the canonical JSON of the configuration identity.
    """
    return content_fingerprint(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": EVALUATION_KIND_AGENT,
            "provider": provider_identity,
            "runner": runner_settings,
        }
    )


def compare_agent_with_baseline(
    current: AgentEvalArtifact,
    baseline_payload: dict[str, Any],
    tolerances: dict[str, MetricTolerance] | None = None,
) -> ComparisonBlock:
    """Compare a fresh agent artifact against a baseline artifact payload.

    Delegates to the shared :func:`compare_evaluation_payloads` with the
    agent gated metric set, directions, and tolerances.

    Args:
        current: The freshly produced artifact.
        baseline_payload: Parsed baseline artifact JSON.
        tolerances: Optional tolerance table override.

    Returns:
        The ComparisonBlock embedded in the current artifact.
    """
    return compare_evaluation_payloads(
        kind=current.kind,
        benchmark_block=current.benchmark,
        configuration_block=current.configuration,
        case_ids=sorted(run.case_id for run in current.runs),
        aggregates=dict(current.aggregates),
        baseline_payload=baseline_payload,
        gated_metrics=AGENT_GATED_METRICS,
        tolerances=tolerances if tolerances is not None else AGENT_TOLERANCES,
        metric_directions=AGENT_METRIC_DIRECTIONS,
    )
