"""Execute a versioned retrieval benchmark against the production services.

The runner is the only retrieval-benchmark execution path: it validates the
index and embedding identity, runs each case through the real
``HybridRetriever`` + ``ContextAssembler`` pipeline, records ordered
candidates/selected context/timings, aggregates deterministic metrics per
cohort and split, and produces an immutable :class:`EvaluationArtifact`.
No LLM, network, or evaluator call happens anywhere in this module.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import EmbeddingConfig
from ..index_operations import IndexOperations
from ..repository_context import AssembledContext, ContextAssembler, ContextItem, HybridRetriever
from .artifacts import (
    EVALUATION_KIND_RETRIEVAL,
    EVALUATION_SCHEMA_VERSION,
    CandidateRecord,
    CaseRunRecord,
    CohortMetricsBlock,
    EvaluationArtifact,
)
from .benchmark import BenchmarkCase, RetrievalBenchmark
from .common import MetricValue, content_fingerprint
from .metrics import (
    bootstrap_ci,
    duplicate_token_ratio,
    evidence_precision_at_k,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

# Bounds on the number of candidate records kept per case; out-of-range
# configuration is rejected before any work begins.
MIN_CANDIDATE_LIMIT = 1
MAX_CANDIDATE_LIMIT = 100
MIN_REPEATS = 1
MAX_REPEATS = 10
MIN_BOOTSTRAP_SAMPLES = 0
MAX_BOOTSTRAP_SAMPLES = 10000
# Evidence files larger than this are refused for line-range validation.
MAX_EVIDENCE_FILE_BYTES = 5_000_000

GRAPH_UNAVAILABLE_REASON = "graph expansion not enabled (IG-03 graph retrieval not implemented)"


class EvalError(RuntimeError):
    """Raised when an evaluation cannot run or complete honestly."""


@dataclass
class RetrievalEvalConfig:
    """Runner configuration; result-affecting fields feed the config fingerprint.

    Attributes:
        token_budget: Approximate token budget for context assembly.
        candidate_limit: Candidate ranking depth per case.
        repeats: Executions per case (1-10). With repeats > 1 the first
            execution is a warm-up excluded from latency statistics.
        per_case_timeout_s: Hard per-execution timeout in seconds.
        record_queries: When False, only a deterministic query hash is stored.
        bootstrap_samples: Bootstrap resamples for confidence intervals
            (0 disables CIs, which are then explicitly unavailable).
        bootstrap_seed: Deterministic bootstrap seed.
    """

    token_budget: int = 2000
    candidate_limit: int = 20
    repeats: int = 1
    per_case_timeout_s: float = 60.0
    record_queries: bool = True
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 20260904

    def __post_init__(self) -> None:
        """Reject out-of-bounds configuration before any work begins.

        Raises:
            ValueError: If any bound is violated.
        """
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if not MIN_CANDIDATE_LIMIT <= self.candidate_limit <= MAX_CANDIDATE_LIMIT:
            raise ValueError(f"candidate_limit must be between {MIN_CANDIDATE_LIMIT} and {MAX_CANDIDATE_LIMIT}")
        if not MIN_REPEATS <= self.repeats <= MAX_REPEATS:
            raise ValueError(f"repeats must be between {MIN_REPEATS} and {MAX_REPEATS}")
        if self.per_case_timeout_s <= 0:
            raise ValueError("per_case_timeout_s must be positive")
        if not MIN_BOOTSTRAP_SAMPLES <= self.bootstrap_samples <= MAX_BOOTSTRAP_SAMPLES:
            raise ValueError(f"bootstrap_samples must be between {MIN_BOOTSTRAP_SAMPLES} and {MAX_BOOTSTRAP_SAMPLES}")

    def result_affecting_settings(self) -> dict[str, Any]:
        """Settings that can change metric values (used for fingerprinting).

        Operational-only settings (repeats, timeout) are excluded so they do
        not invalidate baselines.

        Returns:
            Dictionary of result-affecting retrieval settings.
        """
        return {
            "token_budget": self.token_budget,
            "candidate_limit": self.candidate_limit,
            "record_queries": self.record_queries,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
        }


@dataclass
class _Execution:
    """One successful case execution (a single repeat).

    Attributes:
        repeat: Repeat index (0-based).
        ranked: Ranked candidate items.
        assembled: Assembler output.
        latency_ms: End-to-end latency of the execution.
        timings: Per-stage timings.
    """

    repeat: int
    ranked: list[ContextItem]
    assembled: AssembledContext
    latency_ms: float
    timings: dict[str, float]


@dataclass
class _CaseOutcome:
    """Full result of running one benchmark case (all repeats).

    Attributes:
        case: The benchmark case.
        status: ``ok`` or ``error``.
        error: First error message when the case failed.
        execution: First successful execution, or ``None``.
        latency_values_ms: Latencies of successful non-warm-up executions.
        repeat_errors: Per-repeat error diagnostics.
    """

    case: BenchmarkCase
    status: str = "ok"
    error: str | None = None
    execution: _Execution | None = None
    latency_values_ms: list[float] = field(default_factory=list)
    repeat_errors: list[dict[str, Any]] = field(default_factory=list)


def _utc_now() -> str:
    """Current UTC time as ISO-8601.

    Returns:
        Timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def normalize_relative_path(file_path: str, project_root: Path) -> str:
    """Normalize a chunk file path to a repository-relative POSIX path.

    Args:
        file_path: Path as recorded in chunk metadata.
        project_root: Resolved repository root.

    Returns:
        Repository-relative POSIX path, or the original string when the path
        is outside the repository.
    """
    try:
        return Path(file_path).resolve().relative_to(project_root).as_posix()
    except ValueError:
        return file_path


def runtime_expectation_errors(case: BenchmarkCase, project_root: Path) -> list[str]:
    """Validate a case's expectations against the actual repository.

    Checks that expected files exist and that declared line ranges fall
    within the file length. Path format validation already ran at load time.

    Args:
        case: The benchmark case.
        project_root: Resolved repository root.

    Returns:
        List of expectation problems; empty when the expectations are valid.
    """
    errors: list[str] = []
    paths = list(case.expected.files) + [path for path in case.expected.line_ranges if path not in case.expected.files]
    for relative in paths:
        absolute = project_root / relative
        if not absolute.is_file():
            errors.append(f"expected evidence file missing from repository: {relative}")
            continue
        if absolute.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
            errors.append(
                f"expected evidence file {relative} exceeds {MAX_EVIDENCE_FILE_BYTES} bytes; refusing to read"
            )
            continue
    for path, bounds in case.expected.line_ranges.items():
        absolute = project_root / path
        if not absolute.is_file() or absolute.stat().st_size > MAX_EVIDENCE_FILE_BYTES:
            continue
        line_count = absolute.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        start, end = bounds
        if end > line_count:
            errors.append(f"expected line range {path} [{start}, {end}] is beyond file length ({line_count} lines)")
    return errors


class RetrievalBenchmarkRunner:
    """Run a retrieval benchmark through the production retrieval services.

    Accepts its dependencies explicitly (embedding provider, clock) so tests
    need no network, global configuration, or wall-clock timing.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        benchmark: RetrievalBenchmark,
        index_name: str,
        embedding_provider: Any,
        embedding_config: EmbeddingConfig,
        config: RetrievalEvalConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
        run_id: str | None = None,
        uuid_factory: Callable[[], str] | None = None,
    ) -> None:
        """Prepare the runner and validate the target index up front.

        Args:
            project_root: Resolved repository root holding the index.
            benchmark: The validated benchmark to execute.
            index_name: Index to evaluate against.
            embedding_provider: The embedding provider used for retrieval.
            embedding_config: Embedding configuration the provider was
                created from (identity source).
            config: Runner configuration; defaults apply when omitted.
            clock: Monotonic clock for timings.
            run_id: Optional pinned run id.
            uuid_factory: Optional uuid factory override (defaults to uuid4).

        Raises:
            EvalError: If the index is missing, unhealthy, stale, or the
                embedding identity does not match the index manifest.
        """
        import uuid as uuid_module

        self.project_root = project_root.resolve()
        self.benchmark = benchmark
        self.index_name = index_name
        self.embedding_provider = embedding_provider
        self.embedding_config = embedding_config
        self.config = config or RetrievalEvalConfig()
        self.clock = clock
        self.run_id = run_id or uuid_module.uuid4().hex
        self._validate_index()

    def _validate_index(self) -> None:
        """Check index health, staleness, and embedding identity.

        Raises:
            EvalError: On missing, unhealthy, or stale indexes and on
                embedding identity mismatch.
        """
        operations = IndexOperations(self.project_root)
        try:
            summary = operations.inspect(self.index_name)
        except Exception as exc:
            raise EvalError(f"Index '{self.index_name}' cannot be inspected: {exc}") from exc
        if not summary.healthy:
            raise EvalError(f"Index '{self.index_name}' is unhealthy: " + "; ".join(summary.problems))
        if summary.stale:
            raise EvalError(
                f"Index '{self.index_name}' is stale (repository changed after the last index build); "
                "rebuild it with 'ctxai index' before evaluating"
            )
        manifest = summary.manifest
        assert manifest is not None
        provider_name = self.embedding_config.provider
        model_name = self.embedding_config.model or getattr(self.embedding_provider, "model", None) or "default"
        dimension = self.embedding_provider.get_dimension()
        if (
            manifest.embedding_provider != provider_name
            or manifest.embedding_model != str(model_name)
            or manifest.embedding_dimension != dimension
        ):
            raise EvalError(
                "Configured embedding identity "
                f"({provider_name}/{model_name}/{dimension}) does not match index '{self.index_name}' "
                f"({manifest.embedding_provider}/{manifest.embedding_model}/{manifest.embedding_dimension}); "
                "rebuild the index or use the manifest's embedding settings"
            )
        self.index_summary = summary

    def _embedding_identity(self) -> dict[str, Any]:
        """Embedding identity actually used for retrieval.

        Returns:
            Dictionary with provider, model, and dimension.
        """
        model_name = self.embedding_config.model or getattr(self.embedding_provider, "model", None) or "default"
        return {
            "provider": self.embedding_config.provider,
            "model": str(model_name),
            "dimension": self.embedding_provider.get_dimension(),
        }

    def configuration_fingerprint(self) -> str:
        """Content-derived fingerprint of the evaluation configuration.

        Returns:
            Hex digest over the embedding identity plus result-affecting
            retrieval settings and the artifact schema version.
        """
        return content_fingerprint(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "kind": EVALUATION_KIND_RETRIEVAL,
                "embedding": self._embedding_identity(),
                "retrieval": self.config.result_affecting_settings(),
            }
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_once(self, case: BenchmarkCase) -> _Execution:
        """Run one case execution through the production retrieval path.

        Args:
            case: The benchmark case.

        Returns:
            The execution record with ranked items, assembled context, and
            stage timings.
        """
        retriever = HybridRetriever(self.project_root, self.embedding_provider, index_name=self.index_name)
        start = self.clock()
        ranked = retriever.retrieve(case.query, limit=self.config.candidate_limit, debug=True)
        retrieve_ms = (self.clock() - start) * 1000.0
        start = self.clock()
        index_name = retriever.index_name
        if index_name is None:  # pragma: no cover - HybridRetriever raises at init when unresolved
            raise EvalError("retriever could not resolve the index name")
        assembled = ContextAssembler(token_budget=self.config.token_budget, debug=False).assemble(index_name, ranked)
        assemble_ms = (self.clock() - start) * 1000.0
        return _Execution(
            repeat=0,
            ranked=ranked,
            assembled=assembled,
            latency_ms=retrieve_ms + assemble_ms,
            timings={"retrieve_ms": retrieve_ms, "assemble_ms": assemble_ms},
        )

    def _execute_with_timeout(self, case: BenchmarkCase) -> _Execution:
        """Run one execution under the configured per-case timeout.

        Args:
            case: The benchmark case.

        Returns:
            The execution record.

        Raises:
            EvalError: When the execution exceeds ``per_case_timeout_s`` or
                the retrieval service raises.
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctxai-eval-case")
        try:
            future = executor.submit(self._execute_once, case)
            try:
                return future.result(timeout=self.config.per_case_timeout_s)
            except FutureTimeoutError as exc:
                raise EvalError(f"retrieval timed out after {self.config.per_case_timeout_s}s") from exc
        except EvalError:
            raise
        except Exception as exc:
            raise EvalError(f"retrieval failed: {exc}") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _run_case(self, case: BenchmarkCase) -> _CaseOutcome:
        """Execute one benchmark case with warm-up/repeat handling.

        Quality metrics come from the first successful execution (deterministic
        retrieval makes successful executions identical). Latency statistics
        come from successful non-warm-up executions; when ``repeats > 1`` the
        first execution is the warm-up and is excluded from latency.

        Args:
            case: The benchmark case.

        Returns:
            The case outcome.
        """
        outcome = _CaseOutcome(case=case)
        executions: list[_Execution] = []
        for repeat in range(self.config.repeats):
            try:
                execution = self._execute_with_timeout(case)
                execution.repeat = repeat
                executions.append(execution)
            except EvalError as exc:
                outcome.repeat_errors.append({"repeat": repeat, "error": str(exc)})
                if outcome.error is None:
                    outcome.error = str(exc)
        if not executions:
            outcome.status = "error"
            if outcome.error is None:
                outcome.error = "no execution produced a result"
            return outcome
        outcome.execution = executions[0]
        if self.config.repeats > 1:
            outcome.latency_values_ms = [item.latency_ms for item in executions[1:]]
        else:
            outcome.latency_values_ms = [item.latency_ms for item in executions]
        return outcome

    # ------------------------------------------------------------------
    # Case records and aggregation
    # ------------------------------------------------------------------

    def _case_run_record(self, outcome: _CaseOutcome, timestamp: str) -> CaseRunRecord:
        """Build the artifact record for one case outcome.

        Args:
            outcome: The case outcome.
            timestamp: ISO-8601 timestamp for the record.

        Returns:
            The CaseRunRecord with candidates, decisions, and per-case metrics.
        """
        case = outcome.case
        if outcome.status == "error" or outcome.execution is None:
            return CaseRunRecord(
                case_id=case.id,
                run_id=self.run_id,
                timestamp=timestamp,
                query=case.query if self.config.record_queries else None,
                query_hash=None if self.config.record_queries else content_fingerprint(case.query),
                cohort=case.cohort,
                split=case.split,
                status="error",
                error=outcome.error or "case failed",
                expected=case.expected.to_dict(),
                candidate_count=0,
                selected_count=0,
                candidates=[],
                estimated_tokens=0,
                first_relevant_rank=None,
                metrics={},
                latency={"values_ms": [], "warmup_excluded": 0},
                timings={},
                line_range_findings=[],
            )

        execution = outcome.execution
        ranked = execution.ranked
        assembled = execution.assembled
        truncated_by_id = {item.id: truncated for item, truncated in zip(assembled.items, assembled.truncated or ())}

        candidates: list[CandidateRecord] = []
        claimed_identities: set[tuple[str, int, int]] = set()
        selected_identities = {(item.file_path, item.start_line, item.end_line) for item in assembled.items}
        for rank, item in enumerate(ranked, 1):
            identity = (item.file_path, item.start_line, item.end_line)
            relative = normalize_relative_path(item.file_path, self.project_root)
            if identity in selected_identities and identity not in claimed_identities:
                decision = "selected"
                claimed_identities.add(identity)
            elif identity in selected_identities:
                decision = "duplicate"
            else:
                decision = "budget"
            candidates.append(
                CandidateRecord(
                    chunk_id=item.id,
                    file_path=relative,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    citation=f"{relative}:{item.start_line}-{item.end_line}",
                    chunk_type=item.chunk_type,
                    score=item.score,
                    reasons=list(item.reasons),
                    final_rank=rank,
                    estimated_tokens=ContextAssembler.estimate_tokens(item.content),
                    decision=decision,
                    truncated=truncated_by_id.get(item.id) if decision == "selected" else None,
                )
            )

        ranked_files = [normalize_relative_path(item.file_path, self.project_root) for item in ranked]
        selected_files = [normalize_relative_path(item.file_path, self.project_root) for item in assembled.items]
        relevant = case.relevant_paths()
        expected_files = set(case.expected.files)
        effective_grades = case.effective_relevance()
        grades = [effective_grades.get(path, 0) for path in ranked_files]
        first_relevant_rank = next((rank for rank, path in enumerate(ranked_files, 1) if path in relevant), None)
        dup_ratio = duplicate_token_ratio([item.content for item in assembled.items])

        metrics: dict[str, MetricValue] = {
            "recall@1": MetricValue.available(recall_at_k(expected_files, ranked_files, 1)),
            "recall@5": MetricValue.available(recall_at_k(expected_files, ranked_files, 5)),
            "recall@10": MetricValue.available(recall_at_k(expected_files, ranked_files, 10)),
            "mrr": MetricValue.available(reciprocal_rank(relevant, ranked_files)),
            "ndcg@10": MetricValue.available(ndcg_at_k(grades, 10)),
            "evidence_precision@5": MetricValue.available(evidence_precision_at_k(relevant, selected_files, 5)),
            "duplicate_token_ratio": (
                MetricValue.available(dup_ratio)
                if dup_ratio is not None
                else MetricValue.unavailable("no selected context")
            ),
        }

        line_range_findings = self._line_range_findings(case, assembled.items)

        return CaseRunRecord(
            case_id=case.id,
            run_id=self.run_id,
            timestamp=timestamp,
            query=case.query if self.config.record_queries else None,
            query_hash=None if self.config.record_queries else content_fingerprint(case.query),
            cohort=case.cohort,
            split=case.split,
            status="ok",
            error=None,
            expected=case.expected.to_dict(),
            candidate_count=len(ranked),
            selected_count=len(assembled.items),
            candidates=candidates,
            estimated_tokens=assembled.estimated_tokens,
            first_relevant_rank=first_relevant_rank,
            metrics=metrics,
            latency={
                "values_ms": outcome.latency_values_ms,
                "warmup_excluded": 1 if self.config.repeats > 1 else 0,
                "repeat_errors": outcome.repeat_errors,
            },
            timings=execution.timings,
            line_range_findings=line_range_findings,
        )

    def _line_range_findings(self, case: BenchmarkCase, selected: list[ContextItem]) -> list[dict[str, Any]]:
        """Judge whether selected citations overlap expected line ranges.

        Args:
            case: The benchmark case.
            selected: Selected context items.

        Returns:
            One finding per expected line range with overlap evidence.
        """
        findings: list[dict[str, Any]] = []
        for path, bounds in sorted(case.expected.line_ranges.items()):
            start, end = bounds
            overlapping = [
                f"{normalize_relative_path(item.file_path, self.project_root)}:{item.start_line}-{item.end_line}"
                for item in selected
                if normalize_relative_path(item.file_path, self.project_root) == path
                and item.start_line <= end
                and item.end_line >= start
            ]
            findings.append(
                {
                    "path": path,
                    "expected_range": [start, end],
                    "overlap": bool(overlapping),
                    "citations": overlapping,
                }
            )
        return findings

    def _aggregate(self, records: list[CaseRunRecord], latencies_by_case: dict[str, list[float]]) -> CohortMetricsBlock:
        """Aggregate metrics for one group of case records.

        Quality metrics average over cases that executed without error; error
        cases are excluded from quality denominators and reflected honestly in
        ``successful_query_rate`` and the ``errored`` count. Unavailable
        metrics carry a reason instead of a fabricated zero.

        Args:
            records: Case records in the group.
            latencies_by_case: Non-warm-up latency values per case id.

        Returns:
            The aggregate metrics block.
        """
        errored = [record for record in records if record.status == "error"]
        ok = [record for record in records if record.status == "ok"]
        successful = [record for record in ok if record.candidate_count >= 1]
        metrics: dict[str, MetricValue] = {}

        def averaged(metric: str) -> MetricValue:
            values: list[float] = []
            for record in ok:
                metric_value = record.metrics.get(metric)
                if metric_value is not None and metric_value.value is not None:
                    values.append(metric_value.value)
            if not values:
                return MetricValue.unavailable("no successful cases")
            return MetricValue.available(sum(values) / len(values))

        if records:
            metrics["successful_query_rate"] = MetricValue.available(len(successful) / len(records))
        else:
            metrics["successful_query_rate"] = MetricValue.unavailable("no cases in cohort")
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "evidence_precision@5"):
            metrics[metric] = averaged(metric)

        latency_values = [value for values in latencies_by_case.values() for value in values]
        latency_p50 = percentile(latency_values, 50)
        latency_p95 = percentile(latency_values, 95)
        metrics["latency_p50_ms"] = (
            MetricValue.available(latency_p50)
            if latency_p50 is not None
            else MetricValue.unavailable("no latency measurements")
        )
        metrics["latency_p95_ms"] = (
            MetricValue.available(latency_p95)
            if latency_p95 is not None
            else MetricValue.unavailable("no latency measurements")
        )

        token_values = [float(record.estimated_tokens) for record in ok]
        token_mean = mean(token_values)
        token_p95 = percentile(token_values, 95)
        metrics["selected_token_mean"] = (
            MetricValue.available(token_mean)
            if token_mean is not None
            else MetricValue.unavailable("no successful cases")
        )
        metrics["selected_token_p95"] = (
            MetricValue.available(token_p95)
            if token_p95 is not None
            else MetricValue.unavailable("no successful cases")
        )

        dup_values: list[float] = []
        for record in ok:
            metric_value = record.metrics.get("duplicate_token_ratio")
            if metric_value is not None and metric_value.value is not None:
                dup_values.append(metric_value.value)
        dup_mean = mean(dup_values)
        metrics["duplicate_token_ratio"] = (
            MetricValue.available(dup_mean) if dup_mean is not None else MetricValue.unavailable("no selected context")
        )
        metrics["graph_contribution_rate"] = MetricValue.unavailable(GRAPH_UNAVAILABLE_REASON)

        confidence_intervals: dict[str, list[float]] = {}
        for metric in ("recall@5", "mrr"):
            values: list[float] = []
            for record in ok:
                metric_value = record.metrics.get(metric)
                if metric_value is not None and metric_value.value is not None:
                    values.append(metric_value.value)
            interval = bootstrap_ci(
                values,
                samples=self.config.bootstrap_samples,
                seed=self.config.bootstrap_seed,
            )
            if interval is not None:
                confidence_intervals[metric] = [interval[0], interval[1]]

        return CohortMetricsBlock(
            cases=len(records),
            successful=len(successful),
            errored=len(errored),
            metrics=metrics,
            confidence_intervals=confidence_intervals,
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> EvaluationArtifact:
        """Execute the whole benchmark and build the evaluation artifact.

        Cases with invalid runtime expectations (missing evidence files,
        out-of-range line ranges) are recorded as case errors; the run is
        marked ``partial`` and never looks like a successful benchmark.

        Returns:
            The completed (or partial) EvaluationArtifact.
        """
        started = self.clock()
        records: list[CaseRunRecord] = []
        latencies_by_case: dict[str, list[float]] = {}
        run_errors: list[str] = []
        for case in self.benchmark.cases:
            timestamp = _utc_now()
            expectation_errors = runtime_expectation_errors(case, self.project_root)
            if expectation_errors:
                outcome = _CaseOutcome(case=case, status="error", error="; ".join(expectation_errors))
            else:
                outcome = self._run_case(case)
            if outcome.status == "error":
                run_errors.append(f"{case.id}: {outcome.error}")
            else:
                latencies_by_case[case.id] = outcome.latency_values_ms
            records.append(self._case_run_record(outcome, timestamp))
        duration_ms = (self.clock() - started) * 1000.0

        by_cohort: dict[str, CohortMetricsBlock] = {}
        for cohort in sorted({record.cohort for record in records}):
            cohort_records = [record for record in records if record.cohort == cohort]
            cohort_latencies = {record.case_id: latencies_by_case.get(record.case_id, []) for record in cohort_records}
            by_cohort[cohort] = self._aggregate(cohort_records, cohort_latencies)
        by_split: dict[str, CohortMetricsBlock] = {}
        for split in sorted({record.split for record in records}):
            split_records = [record for record in records if record.split == split]
            split_latencies = {record.case_id: latencies_by_case.get(record.case_id, []) for record in split_records}
            by_split[split] = self._aggregate(split_records, split_latencies)
        overall = self._aggregate(records, latencies_by_case)

        manifest = self.index_summary.manifest
        assert manifest is not None
        artifact = EvaluationArtifact(
            schema_version=EVALUATION_SCHEMA_VERSION,
            kind=EVALUATION_KIND_RETRIEVAL,
            run_id=self.run_id,
            created_at=_utc_now(),
            duration_ms=duration_ms,
            status="complete" if not run_errors else "partial",
            benchmark={
                "name": self.benchmark.name,
                "fingerprint": self.benchmark.fingerprint,
                "schema_version": self.benchmark.schema_version,
                "case_count": len(self.benchmark.cases),
            },
            configuration={
                "fingerprint": self.configuration_fingerprint(),
                "embedding": self._embedding_identity(),
                "retrieval": self.config.result_affecting_settings(),
            },
            index={
                "name": self.index_name,
                "schema_version": manifest.schema_version,
                "embedding_provider": manifest.embedding_provider,
                "embedding_model": manifest.embedding_model,
                "embedding_dimension": manifest.embedding_dimension,
                "repository_root": manifest.repository_root,
                "repository_revision": manifest.repository_revision,
                "chunk_count": manifest.chunk_count,
                "file_count": manifest.file_count,
                "index_updated_at": manifest.updated_at,
                "healthy": self.index_summary.healthy,
                "stale": self.index_summary.stale,
            },
            environment=self._environment(),
            runs=records,
            aggregates={"overall": overall, "by_cohort": by_cohort, "by_split": by_split},
            comparison=None,
            errors=run_errors,
        )
        return artifact

    def _environment(self) -> dict[str, Any]:
        """Environment metadata proving the run was local-only.

        Returns:
            Dictionary with python/platform/version and the explicit no-network
            evaluator statement.
        """
        import platform
        import sys

        try:
            from importlib import metadata

            ctxai_version = metadata.version("ctxai")
        except Exception:
            ctxai_version = "unknown"
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "ctxai_version": ctxai_version,
            "evaluator": "local-retrieval",
            "network_access": "none",
        }
