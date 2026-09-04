"""Unit tests for RE-01 benchmark schema, fingerprints, artifacts, and gates.

Covers schema validation errors, dataclass round-trips, content-derived
fingerprints, artifact redaction, volatile-field stripping, and baseline
comparison thresholds (regressions, tolerances, incompatibilities).
"""

import json
from pathlib import Path

import pytest

from ctxai.evals.artifacts import (
    DEFAULT_TOLERANCES,
    CandidateRecord,
    CaseRunRecord,
    CohortMetricsBlock,
    ComparisonBlock,
    EvaluationArtifact,
    MetricTolerance,
    compare_with_baseline,
    evaluations_dir,
)
from ctxai.evals.benchmark import (
    BenchmarkValidationError,
    RetrievalBenchmark,
    benchmark_from_payload,
    load_benchmark,
)
from ctxai.evals.common import MetricValue, content_fingerprint, redact_artifact, strip_volatile


def _valid_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "unit-benchmark",
        "description": "schema test",
        "cases": [
            {
                "id": "c1",
                "query": "first query",
                "tags": ["t1"],
                "cohort": "core",
                "split": "test",
                "expected": {"files": ["src/a.py"], "symbols": [], "line_ranges": {}},
                "relevance": {"src/a.py": 3},
            },
            {
                "id": "c2",
                "query": "second query",
                "tags": [],
                "cohort": "core",
                "split": "train",
                "expected": {"files": ["src/b.py"], "symbols": ["run"], "line_ranges": {"src/b.py": [1, 5]}},
                "relevance": {},
            },
        ],
    }


def _minimal_artifact(tmp_path: Path, **overrides) -> EvaluationArtifact:
    """Build a minimal complete artifact for comparison tests."""
    run_record = CaseRunRecord(
        case_id="c1",
        run_id="run-1",
        timestamp="2026-09-04T00:00:00+00:00",
        query="first query",
        query_hash=None,
        cohort="core",
        split="test",
        status="ok",
        error=None,
        expected={"files": ["src/a.py"], "symbols": [], "line_ranges": {}},
        candidate_count=2,
        selected_count=1,
        candidates=[
            CandidateRecord(
                chunk_id="ch1",
                file_path="src/a.py",
                start_line=1,
                end_line=2,
                citation="src/a.py:1-2",
                chunk_type="function_definition",
                score=0.5,
                reasons=["semantic rank 1"],
                final_rank=1,
                estimated_tokens=10,
                decision="selected",
                truncated=False,
            )
        ],
        estimated_tokens=10,
        first_relevant_rank=1,
        metrics={
            "recall@5": MetricValue.available(1.0),
            "mrr": MetricValue.available(1.0),
        },
        latency={"values_ms": [1.0], "warmup_excluded": 0},
        timings={"retrieve_ms": 0.5, "assemble_ms": 0.5},
        line_range_findings=[],
    )
    overall_metrics = {
        "recall@5": MetricValue.available(1.0),
        "mrr": MetricValue.available(1.0),
        "duplicate_token_ratio": MetricValue.available(0.8),
        "selected_token_mean": MetricValue.available(100.0),
        "latency_p50_ms": MetricValue.available(1.0),
    }
    overall = CohortMetricsBlock(
        cases=1,
        successful=1,
        errored=0,
        metrics=dict(overall_metrics),
        confidence_intervals={},
    )
    cohort = CohortMetricsBlock(
        cases=1,
        successful=1,
        errored=0,
        metrics=dict(overall_metrics),
        confidence_intervals={},
    )
    fields = dict(
        schema_version=1,
        kind="retrieval",
        run_id="run-1",
        created_at="2026-09-04T00:00:00+00:00",
        duration_ms=12.0,
        status="complete",
        benchmark={
            "name": "unit-benchmark",
            "fingerprint": "bench-fp",
            "schema_version": 1,
            "case_count": 1,
        },
        configuration={
            "fingerprint": "config-fp",
            "embedding": {"provider": "mock", "model": "mock-model", "dimension": 384},
            "retrieval": {"token_budget": 2000, "candidate_limit": 20},
        },
        index={"name": "unit-index", "healthy": True, "stale": False},
        environment={"python_version": "3.13.0", "platform": "test", "ctxai_version": "0.0.1"},
        runs=[run_record],
        aggregates={"overall": overall, "by_cohort": {"core": cohort}, "by_split": {}},
        comparison=None,
        errors=[],
    )
    fields.update(overrides)
    return EvaluationArtifact(**fields)


class TestBenchmarkValidation:
    def test_valid_payload_round_trips(self):
        benchmark = benchmark_from_payload(_valid_payload())
        assert benchmark.name == "unit-benchmark"
        assert [case.id for case in benchmark.cases] == ["c1", "c2"]
        assert benchmark.cases[1].effective_relevance() == {"src/b.py": 2}

    def test_round_trip_to_dict_from_dict(self):
        benchmark = benchmark_from_payload(_valid_payload())
        restored = RetrievalBenchmark.from_dict(benchmark.to_dict())
        assert restored == benchmark
        assert restored.to_dict() == benchmark.to_dict()

    def test_duplicate_case_ids_rejected(self):
        payload = _valid_payload()
        payload["cases"][1]["id"] = "c1"
        with pytest.raises(BenchmarkValidationError) as excinfo:
            benchmark_from_payload(payload)
        assert "duplicate case id" in " ".join(excinfo.value.errors)

    def test_absolute_path_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["expected"]["files"] = ["/etc/passwd"]
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_parent_escape_path_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["expected"]["files"] = ["../secrets.py"]
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_empty_query_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["query"] = "   "
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_invalid_split_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["split"] = "holdout"
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_invalid_relevance_grade_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["relevance"] = {"src/a.py": 4}
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_expected_file_graded_zero_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["relevance"] = {"src/a.py": 0}
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_inverted_line_range_rejected(self):
        payload = _valid_payload()
        payload["cases"][1]["expected"]["line_ranges"] = {"src/b.py": [9, 3]}
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_zero_start_line_range_rejected(self):
        payload = _valid_payload()
        payload["cases"][1]["expected"]["line_ranges"] = {"src/b.py": [0, 3]}
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_missing_expected_files_rejected(self):
        payload = _valid_payload()
        payload["cases"][0]["expected"]["files"] = []
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_wrong_schema_version_rejected(self):
        payload = _valid_payload()
        payload["schema_version"] = 99
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_case_count_bound_rejected(self):
        payload = _valid_payload()
        template = payload["cases"][0]
        payload["cases"] = [dict(template, id=f"c{index}") for index in range(1001)]
        with pytest.raises(BenchmarkValidationError):
            benchmark_from_payload(payload)

    def test_all_errors_reported_together(self):
        payload = _valid_payload()
        payload["cases"][0]["query"] = ""
        payload["cases"][0]["split"] = "nope"
        payload["cases"][1]["id"] = "c1"
        with pytest.raises(BenchmarkValidationError) as excinfo:
            benchmark_from_payload(payload)
        assert len(excinfo.value.errors) == 3

    def test_load_benchmark_rejects_oversized_file(self, tmp_path):
        path = tmp_path / "big.json"
        path.write_text(json.dumps(_valid_payload()) + " " * (1_000_001), encoding="utf-8")
        with pytest.raises(BenchmarkValidationError) as excinfo:
            load_benchmark(path)
        assert "benchmark file exceeds" in str(excinfo.value)

    def test_load_benchmark_rejects_malformed_json(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BenchmarkValidationError):
            load_benchmark(path)

    def test_benchmark_fingerprint_is_content_derived(self):
        first = benchmark_from_payload(_valid_payload())
        second = benchmark_from_payload(_valid_payload())
        assert first.fingerprint == second.fingerprint
        changed = _valid_payload()
        changed["cases"][0]["query"] = "changed query"
        assert first.fingerprint != benchmark_from_payload(changed).fingerprint

    def test_benchmark_fingerprint_ignores_key_order(self):
        payload = _valid_payload()
        reordered = {
            "cases": [{key: case[key] for key in sorted(case, reverse=True)} for case in reversed(payload["cases"])],
            **{key: payload[key] for key in reversed(list(payload))},
        }
        assert content_fingerprint(benchmark_from_payload(payload).to_dict()) == content_fingerprint(
            benchmark_from_payload(reordered).to_dict()
        )


class TestRedaction:
    def test_secrets_are_redacted(self):
        payload = {"note": "api_key=supersecret123", "nested": {"token": "ghp_abcdefghijklmn"}}
        redacted = redact_artifact(payload, project_root=Path("/tmp/project"))
        assert "supersecret123" not in json.dumps(redacted)
        assert "ghp_abcdefghijklmn" not in json.dumps(redacted)

    def test_absolute_paths_are_replaced(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        payload = {"file": str(project / "src" / "a.py"), "list": [str(project)]}
        redacted = redact_artifact(payload, project_root=project)
        encoded = json.dumps(redacted)
        assert str(project) not in encoded
        assert "<project>/src/a.py" == redacted["file"]
        assert "<project>" == redacted["list"][0]

    def test_user_home_is_replaced(self, tmp_path):
        payload = {"home": str(Path.home() / "elsewhere")}
        redacted = redact_artifact(payload, project_root=tmp_path)
        assert str(Path.home()) not in json.dumps(redacted)


class TestVolatileStripping:
    def test_documented_volatile_fields_are_removed(self):
        artifact = _minimal_artifact(Path("/tmp")).to_dict()
        stripped = strip_volatile(artifact)
        assert "created_at" not in stripped
        assert "duration_ms" not in stripped
        assert "timestamp" not in stripped["runs"][0]
        assert "timings" not in stripped["runs"][0]
        assert "latency" not in stripped["runs"][0]
        assert "latency_p50_ms" not in stripped["aggregates"]["overall"]["metrics"]

    def test_stripping_is_idempotent_and_preserves_metrics(self):
        artifact = _minimal_artifact(Path("/tmp")).to_dict()
        once = strip_volatile(artifact)
        assert strip_volatile(once) == once
        assert once["runs"][0]["candidates"][0]["chunk_id"] == "ch1"
        assert once["aggregates"]["overall"]["metrics"]["recall@5"]["value"] == 1.0


class TestArtifactRoundTrip:
    def test_evaluation_artifact_round_trips(self):
        artifact = _minimal_artifact(Path("/tmp"))
        restored = EvaluationArtifact.from_dict(artifact.to_dict())
        assert restored == artifact
        assert restored.to_dict() == artifact.to_dict()

    def test_metric_value_round_trip_including_unavailable(self):
        available = MetricValue.available(0.75)
        unavailable = MetricValue.unavailable("graph expansion not enabled")
        assert MetricValue.from_dict(available.to_dict()) == available
        assert MetricValue.from_dict(unavailable.to_dict()) == unavailable
        assert unavailable.to_dict() == {
            "available": False,
            "value": None,
            "reason": "graph expansion not enabled",
        }
        assert unavailable.value is None

    def test_evaluations_dir_is_project_scoped(self, tmp_path):
        assert evaluations_dir(tmp_path) == tmp_path / ".ctxai" / "evaluations" / "retrieval"


class TestBaselineComparison:
    def _payload(self, tmp_path: Path, artifact: EvaluationArtifact) -> dict:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_identical_metrics_pass(self, tmp_path):
        current = _minimal_artifact(tmp_path)
        comparison = compare_with_baseline(current, self._payload(tmp_path, current))
        assert comparison.status == "pass"
        assert all(gate.status == "pass" for gate in comparison.gates)

    def test_regression_beyond_tolerance_fails_named_gate(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        degraded = _minimal_artifact(tmp_path)
        degraded.aggregates["overall"].metrics["recall@5"] = MetricValue.available(0.5)
        comparison = compare_with_baseline(degraded, self._payload(tmp_path, baseline))
        assert comparison.status == "regression"
        failing = [gate for gate in comparison.gates if gate.status == "regression"]
        assert [gate.metric for gate in failing] == ["recall@5"]
        assert failing[0].baseline == 1.0
        assert failing[0].current == 0.5

    def test_regression_within_tolerance_passes(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        slightly_worse = _minimal_artifact(tmp_path)
        tolerance = DEFAULT_TOLERANCES["recall@5"]
        slightly_worse.aggregates["overall"].metrics["recall@5"] = MetricValue.available(
            1.0 - max(tolerance.absolute, tolerance.relative * 1.0) + 1e-9
        )
        comparison = compare_with_baseline(slightly_worse, self._payload(tmp_path, baseline))
        assert comparison.status == "pass"

    def test_improvement_is_not_a_regression(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        better = _minimal_artifact(tmp_path)
        better.aggregates["overall"].metrics["duplicate_token_ratio"] = MetricValue.available(0.5)
        comparison = compare_with_baseline(better, self._payload(tmp_path, baseline))
        assert comparison.status == "pass"

    def test_lower_is_better_metrics_gate_upward_drift(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        heavier = _minimal_artifact(tmp_path)
        heavier.aggregates["overall"].metrics["selected_token_mean"] = MetricValue.available(10_000.0)
        comparison = compare_with_baseline(heavier, self._payload(tmp_path, baseline))
        assert comparison.status == "regression"
        assert any(gate.metric == "selected_token_mean" for gate in comparison.gates if gate.status == "regression")

    def test_latency_metrics_are_not_gated(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        slower = _minimal_artifact(tmp_path)
        slower.aggregates["overall"].metrics["latency_p50_ms"] = MetricValue.available(999_999.0)
        comparison = compare_with_baseline(slower, self._payload(tmp_path, baseline))
        latency_gates = [gate for gate in comparison.gates if gate.metric == "latency_p50_ms"]
        assert latency_gates == []

    def test_unavailable_metric_is_reported_not_failed(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        current = _minimal_artifact(tmp_path)
        current.aggregates["overall"].metrics["mrr"] = MetricValue.unavailable("no successful cases")
        comparison = compare_with_baseline(current, self._payload(tmp_path, baseline))
        gate = next(gate for gate in comparison.gates if gate.metric == "mrr")
        assert gate.status == "unavailable"
        assert comparison.status == "pass"

    def test_benchmark_fingerprint_mismatch_is_incompatible(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        other_benchmark = _minimal_artifact(tmp_path)
        other_benchmark.benchmark["fingerprint"] = "different-benchmark"
        comparison = compare_with_baseline(other_benchmark, self._payload(tmp_path, baseline))
        assert comparison.status == "incompatible"
        assert any("benchmark fingerprint" in item for item in comparison.incompatibilities)

    def test_configuration_fingerprint_mismatch_is_incompatible(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        other_config = _minimal_artifact(tmp_path)
        other_config.configuration["fingerprint"] = "different-config"
        comparison = compare_with_baseline(other_config, self._payload(tmp_path, baseline))
        assert comparison.status == "incompatible"
        assert any("configuration fingerprint" in item for item in comparison.incompatibilities)

    def test_case_set_mismatch_is_incompatible(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        current = _minimal_artifact(tmp_path)
        current.runs.append(
            CaseRunRecord(
                case_id="c2",
                run_id="run-1",
                timestamp="2026-09-04T00:00:00+00:00",
                query="second",
                query_hash=None,
                cohort="core",
                split="test",
                status="ok",
                error=None,
                expected={"files": ["src/b.py"], "symbols": [], "line_ranges": {}},
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
        )
        comparison = compare_with_baseline(current, self._payload(tmp_path, baseline))
        assert comparison.status == "incompatible"
        assert any("case set" in item for item in comparison.incompatibilities)

    def test_cohort_set_mismatch_is_incompatible(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        current = _minimal_artifact(tmp_path)
        current.aggregates["by_cohort"]["extra-cohort"] = current.aggregates["overall"]
        comparison = compare_with_baseline(current, self._payload(tmp_path, baseline))
        assert comparison.status == "incompatible"
        assert any("cohort" in item for item in comparison.incompatibilities)

    def test_comparison_block_round_trips(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        comparison = compare_with_baseline(_minimal_artifact(tmp_path), self._payload(tmp_path, baseline))
        restored = ComparisonBlock.from_dict(comparison.to_dict())
        assert restored == comparison

    def test_custom_tolerances_are_honored(self, tmp_path):
        baseline = _minimal_artifact(tmp_path)
        degraded = _minimal_artifact(tmp_path)
        degraded.aggregates["overall"].metrics["recall@5"] = MetricValue.available(0.97)
        loose = {"recall@5": MetricTolerance(absolute=0.05, relative=0.0)}
        comparison = compare_with_baseline(degraded, self._payload(tmp_path, baseline), tolerances=loose)
        assert comparison.status == "pass"
