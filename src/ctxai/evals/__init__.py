"""Evaluation frameworks for ctxai (RE-01 retrieval benchmark; HH-09 agent tasks).

The shared artifact discipline lives in :mod:`ctxai.evals.common` (canonical
JSON, content fingerprints, atomic writes, redaction, volatile-field
stripping, :class:`MetricValue`) and :mod:`ctxai.evals.artifacts` (artifact
models, tolerances, baseline comparison) so both eval frameworks use one
fingerprinting, redaction, and comparison approach by design.
"""

from .artifacts import (
    DEFAULT_TOLERANCES,
    EVALUATION_KIND_RETRIEVAL,
    EVALUATION_SCHEMA_VERSION,
    GATED_METRICS,
    METRIC_DIRECTIONS,
    CandidateRecord,
    CaseRunRecord,
    CohortMetricsBlock,
    ComparisonBlock,
    EvaluationArtifact,
    GateResult,
    MetricTolerance,
    compare_with_baseline,
    default_artifact_path,
    evaluations_dir,
    save_artifact,
)
from .benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkValidationError,
    ExpectedEvidence,
    RetrievalBenchmark,
    benchmark_from_payload,
    load_benchmark,
    validate_benchmark_payload,
)
from .common import (
    MetricValue,
    atomic_write_json,
    canonical_json,
    content_fingerprint,
    redact_artifact,
    redact_home_paths,
    strip_volatile,
)
from .retrieval_runner import (
    EvalError,
    RetrievalBenchmarkRunner,
    RetrievalEvalConfig,
    runtime_expectation_errors,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_TOLERANCES",
    "EVALUATION_KIND_RETRIEVAL",
    "EVALUATION_SCHEMA_VERSION",
    "GATED_METRICS",
    "METRIC_DIRECTIONS",
    "BenchmarkCase",
    "BenchmarkValidationError",
    "CandidateRecord",
    "CaseRunRecord",
    "CohortMetricsBlock",
    "ComparisonBlock",
    "EvalError",
    "ExpectedEvidence",
    "EvaluationArtifact",
    "GateResult",
    "MetricTolerance",
    "MetricValue",
    "RetrievalBenchmark",
    "RetrievalBenchmarkRunner",
    "RetrievalEvalConfig",
    "atomic_write_json",
    "benchmark_from_payload",
    "canonical_json",
    "compare_with_baseline",
    "content_fingerprint",
    "default_artifact_path",
    "evaluations_dir",
    "load_benchmark",
    "redact_artifact",
    "redact_home_paths",
    "runtime_expectation_errors",
    "save_artifact",
    "strip_volatile",
    "validate_benchmark_payload",
]
