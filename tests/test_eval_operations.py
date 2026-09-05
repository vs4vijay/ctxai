"""Unit tests for the RE-03 evaluation artifact service and run comparison.

Covers the ``EvaluationOperations`` store (project-contained artifact root,
artifact id validation, bounded listing, payload fetch) and the shared
run-comparison core (metric deltas by dimension, tolerance boundaries,
newly passing/failing cases, incompatibilities with rebuild/rerun actions).
"""

import json
from pathlib import Path

import pytest

from ctxai.evals.artifacts import DEFAULT_TOLERANCES, CaseRunRecord, CohortMetricsBlock, MetricTolerance
from ctxai.evals.common import MetricValue
from ctxai.evals.operations import (
    ArtifactCorruptError,
    ArtifactNotFoundError,
    CaseDelta,
    EvaluationOperations,
    EvaluationRunSummary,
    InvalidArtifactIdError,
    MetricDelta,
    RootOutsideProjectError,
    RunComparison,
    compare_retrieval_payloads,
    is_valid_artifact_id,
    metric_dimension,
)
from tests.test_eval_artifacts import _minimal_artifact


def _run_record(case_id: str, **overrides) -> CaseRunRecord:
    """Build one case run record with overridable fields."""
    fields = dict(
        case_id=case_id,
        run_id="run-1",
        timestamp="2026-09-04T00:00:00+00:00",
        query=f"query {case_id}",
        query_hash=None,
        cohort="core",
        split="test",
        status="ok",
        error=None,
        expected={"files": ["src/a.py"], "symbols": [], "line_ranges": {}},
        candidate_count=2,
        selected_count=1,
        candidates=[],
        estimated_tokens=10,
        first_relevant_rank=1,
        metrics={
            "recall@5": MetricValue.available(1.0),
            "mrr": MetricValue.available(1.0),
        },
        latency={"values_ms": [1.0], "warmup_excluded": 0},
        timings={"retrieve_ms": 0.5},
        line_range_findings=[],
    )
    fields.update(overrides)
    return CaseRunRecord(**fields)


def _metrics_block(metrics: dict) -> CohortMetricsBlock:
    """Build a metrics block from a metric mapping (1 successful case)."""
    return CohortMetricsBlock(
        cases=1,
        successful=1,
        errored=0,
        metrics=dict(metrics),
        confidence_intervals={},
    )


def _base_metrics() -> dict:
    """The default metric set shared by baseline and candidate fixtures."""
    return {
        "recall@5": MetricValue.available(1.0),
        "mrr": MetricValue.available(1.0),
        "successful_query_rate": MetricValue.available(1.0),
        "duplicate_token_ratio": MetricValue.available(0.8),
        "selected_token_mean": MetricValue.available(100.0),
        "latency_p50_ms": MetricValue.available(1.0),
        "latency_p95_ms": MetricValue.available(2.0),
    }


def _artifact(tmp_path: Path, run_id: str, **overrides):
    """Build a minimal compatible artifact with one case and full aggregates."""
    defaults = dict(
        run_id=run_id,
        runs=[_run_record("c1", run_id=run_id)],
        aggregates={
            "overall": _metrics_block(_base_metrics()),
            "by_cohort": {"core": _metrics_block(_base_metrics())},
            "by_split": {},
        },
    )
    defaults.update(overrides)
    return _minimal_artifact(tmp_path, **defaults)


def _store_root(tmp_path: Path) -> Path:
    """The default artifact root of a scratch project."""
    return tmp_path / ".ctxai" / "evaluations" / "retrieval"


def _write_artifact(tmp_path: Path, name: str, artifact) -> Path:
    """Persist an artifact payload into the default store root."""
    root = _store_root(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


@pytest.fixture
def store(tmp_path) -> EvaluationOperations:
    """A store over an empty artifact root in a scratch project."""
    return EvaluationOperations(tmp_path)


class TestArtifactIdValidation:
    @pytest.mark.parametrize(
        "value",
        ["abc", "ctxai-retrieval-core-20260905T000000-abcd1234", "Run_1.2", "a" * 200],
    )
    def test_safe_ids_accepted(self, value):
        assert is_valid_artifact_id(value) is True

    @pytest.mark.parametrize(
        "value",
        ["", "..", "../escape", "a/b", "a\\b", ".hidden", "a b", "a\nb", "%2e%2e", "a" * 201, None],
    )
    def test_unsafe_ids_rejected(self, value):
        assert is_valid_artifact_id(value) is False

    def test_service_rejects_traversal_ids(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        for bad in ("../secrets", "a/b", "..", ".hidden"):
            with pytest.raises(InvalidArtifactIdError):
                operations.read_run(bad)


class TestArtifactRoot:
    def test_default_root_is_project_scoped(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        assert operations.artifact_root == tmp_path / ".ctxai" / "evaluations" / "retrieval"

    def test_override_inside_project_is_accepted(self, tmp_path):
        nested = tmp_path / "custom" / "root"
        operations = EvaluationOperations(tmp_path, artifact_root=nested)
        assert operations.artifact_root == nested.resolve()

    def test_override_outside_project_is_refused(self, tmp_path):
        outside = tmp_path.parent / "outside"
        with pytest.raises(RootOutsideProjectError):
            EvaluationOperations(tmp_path, artifact_root=outside)


class TestListRuns:
    def test_empty_root_lists_nothing(self, store):
        summaries, corrupt = store.list_runs()
        assert summaries == []
        assert corrupt == []

    def test_lists_summaries_newest_first(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        older = _artifact(tmp_path, "run-older", created_at="2026-09-01T00:00:00+00:00")
        newer = _artifact(tmp_path, "run-newer", created_at="2026-09-05T00:00:00+00:00")
        _write_artifact(tmp_path, "older.json", older)
        _write_artifact(tmp_path, "newer.json", newer)

        summaries, corrupt = operations.list_runs()
        assert corrupt == []
        assert [summary.run_id for summary in summaries] == ["run-newer", "run-older"]
        first = summaries[0]
        assert isinstance(first, EvaluationRunSummary)
        assert first.benchmark_name == "unit-benchmark"
        assert first.benchmark_fingerprint == "bench-fp"
        assert first.configuration_fingerprint == "config-fp"
        assert first.status == "complete"
        assert first.index_name == "unit-index"
        assert first.case_count == 1
        assert first.graph_enabled is False
        assert first.comparison_status is None

    def test_limit_bounds_results(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        for index in range(5):
            artifact = _artifact(tmp_path, f"run-{index}")
            _write_artifact(tmp_path, f"artifact-{index}.json", artifact)
        summaries, _ = operations.list_runs(limit=2)
        assert len(summaries) == 2

    def test_corrupt_files_are_diagnostics_not_failures(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        _store_root(tmp_path).mkdir(parents=True, exist_ok=True)
        (_store_root(tmp_path) / "broken.json").write_text("{not json", encoding="utf-8")
        artifact = _artifact(tmp_path, "run-ok")
        _write_artifact(tmp_path, "ok.json", artifact)
        summaries, corrupt = operations.list_runs()
        assert [summary.run_id for summary in summaries] == ["run-ok"]
        assert len(corrupt) == 1
        assert "broken.json" in corrupt[0]


class TestReadRun:
    def test_reads_by_file_stem(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        artifact = _artifact(tmp_path, "run-1")
        _write_artifact(tmp_path, "ctxai-retrieval-core-20260905-run1.json", artifact)
        payload = operations.read_run("ctxai-retrieval-core-20260905-run1")
        assert payload["run_id"] == "run-1"

    def test_reads_by_full_run_id_inside_payload(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        artifact = _artifact(tmp_path, "0123456789abcdef0123456789abcdef")
        _write_artifact(tmp_path, "some-stem.json", artifact)
        payload = operations.read_run("0123456789abcdef0123456789abcdef")
        assert payload["run_id"] == "0123456789abcdef0123456789abcdef"

    def test_missing_run_raises_not_found(self, store):
        with pytest.raises(ArtifactNotFoundError):
            store.read_run("does-not-exist")

    def test_corrupt_run_raises_corrupt(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        _store_root(tmp_path).mkdir(parents=True, exist_ok=True)
        (_store_root(tmp_path) / "broken.json").write_text("[]", encoding="utf-8")
        with pytest.raises(ArtifactCorruptError):
            operations.read_run("broken")


class TestMetricDimensions:
    def test_quality_metrics_classified(self):
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "evidence_precision@5"):
            assert metric_dimension(metric) == "quality"

    def test_correctness_efficiency_and_timing_classified(self):
        assert metric_dimension("successful_query_rate") == "correctness"
        assert metric_dimension("duplicate_token_ratio") == "efficiency"
        assert metric_dimension("selected_token_mean") == "efficiency"
        assert metric_dimension("selected_token_p95") == "efficiency"
        assert metric_dimension("latency_p50_ms") == "timing"
        assert metric_dimension("latency_p95_ms") == "timing"

    def test_unknown_metric_is_other(self):
        assert metric_dimension("graph_contribution_rate") == "other"


class TestRunComparison:
    def test_identical_runs_pass_with_flat_deltas(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        assert isinstance(comparison, RunComparison)
        assert comparison.status == "pass"
        assert comparison.compatible is True
        assert comparison.incompatibilities == []
        assert comparison.actions == []
        assert comparison.case_deltas[0].kind == "unchanged"
        recall = next(gate for gate in comparison.gates if gate.metric == "recall@5" and gate.cohort == "overall")
        assert recall.status == "pass"
        assert recall.trend == "flat"
        assert recall.dimension == "quality"
        assert recall.gated is True
        assert recall.baseline == 1.0 and recall.current == 1.0

    def test_latency_is_reported_but_never_gated(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["latency_p95_ms"] = MetricValue.available(999_999.0)
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        latency = next(gate for gate in comparison.gates if gate.metric == "latency_p95_ms")
        assert latency.gated is False
        assert latency.dimension == "timing"
        assert latency.status == "reported"
        assert latency.trend == "regressed"
        assert comparison.status == "pass", "noisy latency must not gate the comparison"

    def test_regression_beyond_tolerance_fails(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["recall@5"] = MetricValue.available(0.5)
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        assert comparison.status == "regression"
        failing = [gate for gate in comparison.gates if gate.status == "regression"]
        assert any(gate.metric == "recall@5" and gate.cohort == "overall" for gate in failing)
        recall = next(gate for gate in failing if gate.metric == "recall@5")
        assert recall.trend == "regressed"
        assert recall.baseline == 1.0 and recall.current == 0.5

    def test_improvement_is_reported_as_improved(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        baseline_payload = baseline.to_dict()
        baseline_payload["aggregates"]["overall"]["metrics"]["mrr"] = {"available": True, "value": 0.5}
        comparison = compare_retrieval_payloads(baseline_payload, candidate.to_dict())
        mrr = next(gate for gate in comparison.gates if gate.metric == "mrr" and gate.cohort == "overall")
        assert mrr.status == "pass"
        assert mrr.trend == "improved"
        assert comparison.status == "pass"

    def test_tolerance_boundary_passes_and_beyond_fails(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        tolerance = DEFAULT_TOLERANCES["recall@5"]
        allowed = max(tolerance.absolute, tolerance.relative * 1.0)
        at_limit = _artifact(tmp_path, "run-at-limit")
        at_limit.aggregates["overall"].metrics["recall@5"] = MetricValue.available(1.0 - allowed + 1e-9)
        assert compare_retrieval_payloads(baseline.to_dict(), at_limit.to_dict()).status == "pass"

        beyond = _artifact(tmp_path, "run-beyond")
        beyond.aggregates["overall"].metrics["recall@5"] = MetricValue.available(1.0 - allowed - 1e-3)
        assert compare_retrieval_payloads(baseline.to_dict(), beyond.to_dict()).status == "regression"

    def test_custom_tolerances_are_honored(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["recall@5"] = MetricValue.available(0.9)
        loose = {"recall@5": MetricTolerance(absolute=0.2, relative=0.0)}
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict(), tolerances=loose)
        assert comparison.status == "pass"

    def test_missing_metric_is_unavailable_not_failed(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["mrr"] = MetricValue.unavailable("no successful cases")
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        mrr = next(gate for gate in comparison.gates if gate.metric == "mrr" and gate.cohort == "overall")
        assert mrr.status == "unavailable"
        assert mrr.trend is None
        assert comparison.status == "pass"

    def test_incompatible_runs_list_every_mismatch_with_actions(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.benchmark["fingerprint"] = "different-benchmark"
        candidate.configuration["fingerprint"] = "different-config"
        candidate.runs.append(_run_record("c2", run_id="run-cand"))
        candidate.aggregates["by_cohort"]["extra"] = candidate.aggregates["overall"]
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        assert comparison.status == "incompatible"
        assert comparison.compatible is False
        assert comparison.gates == [] and comparison.case_deltas == []
        joined = " | ".join(comparison.incompatibilities)
        assert "benchmark fingerprint" in joined
        assert "configuration fingerprint" in joined
        assert "case set" in joined
        assert "cohort set" in joined
        assert comparison.actions, "incompatible comparisons must give a rebuild/rerun action"
        assert any("Re-run" in action for action in comparison.actions)

    def test_schema_mismatch_action_says_rebuild(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        baseline_payload = baseline.to_dict()
        baseline_payload["schema_version"] = 99
        comparison = compare_retrieval_payloads(baseline_payload, candidate.to_dict())
        assert comparison.status == "incompatible"
        assert any("Rebuild" in action for action in comparison.actions)

    def test_cohort_deltas_are_computed_per_cohort(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["by_cohort"]["core"].metrics["recall@5"] = MetricValue.available(0.5)
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        cohort_gate = next(gate for gate in comparison.gates if gate.cohort == "core" and gate.metric == "recall@5")
        assert cohort_gate.status == "regression"

    def test_newly_passing_and_failing_cases_identified(self, tmp_path):
        baseline = _artifact(
            tmp_path,
            "run-base",
            runs=[
                _run_record("c1", run_id="run-base", first_relevant_rank=None),
                _run_record("c2", run_id="run-base", first_relevant_rank=2),
            ],
        )
        candidate = _artifact(
            tmp_path,
            "run-cand",
            runs=[
                _run_record("c1", run_id="run-cand", first_relevant_rank=1),
                _run_record("c2", run_id="run-cand", first_relevant_rank=None),
            ],
        )
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        kinds = {delta.case_id: delta.kind for delta in comparison.case_deltas}
        assert kinds["c1"] == "newly_passing"
        assert kinds["c2"] == "newly_failing"
        assert comparison.newly_passing == ["c1"]
        assert comparison.newly_failing == ["c2"]

    def test_rank_improvement_and_regression_kinds(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base", runs=[_run_record("c1", run_id="run-base", first_relevant_rank=3)])
        improved = _artifact(tmp_path, "run-cand", runs=[_run_record("c1", run_id="run-cand", first_relevant_rank=1)])
        comparison = compare_retrieval_payloads(baseline.to_dict(), improved.to_dict())
        assert comparison.case_deltas[0].kind == "improved"

        worsened = _artifact(tmp_path, "run-cand", runs=[_run_record("c1", run_id="run-cand", first_relevant_rank=5)])
        comparison = compare_retrieval_payloads(baseline.to_dict(), worsened.to_dict())
        assert comparison.case_deltas[0].kind == "worsened"

    def test_case_token_and_latency_deltas_are_reported(self, tmp_path):
        baseline = _artifact(
            tmp_path,
            "run-base",
            runs=[_run_record("c1", run_id="run-base", estimated_tokens=100, latency={"values_ms": [10.0]})],
        )
        candidate = _artifact(
            tmp_path,
            "run-cand",
            runs=[_run_record("c1", run_id="run-cand", estimated_tokens=150, latency={"values_ms": [40.0]})],
        )
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        delta = comparison.case_deltas[0]
        assert delta.baseline_tokens == 100 and delta.candidate_tokens == 150
        assert delta.delta_tokens == 50
        assert delta.baseline_latency_ms == 10.0 and delta.candidate_latency_ms == 40.0

    def test_case_status_change_is_newly_failing(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base", runs=[_run_record("c1", run_id="run-base", first_relevant_rank=1)])
        candidate = _artifact(
            tmp_path,
            "run-cand",
            runs=[_run_record("c1", run_id="run-cand", status="error", error="boom", first_relevant_rank=None)],
        )
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        delta = comparison.case_deltas[0]
        assert delta.kind == "newly_failing"
        assert delta.candidate_status == "error"

    def test_correctness_gate_uses_zero_tolerance(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["successful_query_rate"] = MetricValue.available(0.9)
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        gate = next(
            gate for gate in comparison.gates if gate.metric == "successful_query_rate" and gate.cohort == "overall"
        )
        assert gate.status == "regression"
        assert gate.dimension == "correctness"
        assert gate.absolute_tolerance == 0.0


class TestDtoRoundTrips:
    def test_metric_delta_round_trips(self):
        delta = MetricDelta(
            cohort="overall",
            metric="recall@5",
            dimension="quality",
            gated=True,
            baseline=1.0,
            current=0.5,
            delta=-0.5,
            direction="higher",
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
            status="regression",
            trend="regressed",
            detail=None,
        )
        assert MetricDelta.from_dict(delta.to_dict()) == delta

    def test_case_delta_round_trips_with_none_fields(self):
        delta = CaseDelta(
            case_id="c1",
            cohort="core",
            split="test",
            kind="newly_failing",
            baseline_status="ok",
            candidate_status="error",
            baseline_first_relevant_rank=1,
            candidate_first_relevant_rank=None,
            baseline_tokens=100,
            candidate_tokens=150,
            delta_tokens=50,
            baseline_latency_ms=10.0,
            candidate_latency_ms=None,
        )
        assert CaseDelta.from_dict(delta.to_dict()) == delta
        assert delta.to_dict()["candidate_first_relevant_rank"] is None

    def test_run_comparison_round_trips(self, tmp_path):
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["recall@5"] = MetricValue.available(0.5)
        comparison = compare_retrieval_payloads(baseline.to_dict(), candidate.to_dict())
        restored = RunComparison.from_dict(comparison.to_dict())
        assert restored == comparison
        assert restored.to_dict() == comparison.to_dict()

    def test_run_summary_round_trips(self):
        summary = EvaluationRunSummary(
            run_id="run-1",
            file_id="artifact.json",
            created_at="2026-09-05T00:00:00+00:00",
            status="complete",
            benchmark_name="unit-benchmark",
            benchmark_fingerprint="bench-fp",
            configuration_fingerprint="config-fp",
            index_name="unit-index",
            case_count=1,
            graph_enabled=False,
            comparison_status="pass",
        )
        assert EvaluationRunSummary.from_dict(summary.to_dict()) == summary


class TestServiceCompare:
    def test_compare_runs_via_store(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        baseline = _artifact(tmp_path, "run-base")
        candidate = _artifact(tmp_path, "run-cand")
        candidate.aggregates["overall"].metrics["recall@5"] = MetricValue.available(0.5)
        _write_artifact(tmp_path, "baseline.json", baseline)
        _write_artifact(tmp_path, "candidate.json", candidate)
        comparison = operations.compare_runs("baseline", "candidate")
        assert comparison.status == "regression"

    def test_compare_runs_missing_artifact_raises(self, tmp_path):
        operations = EvaluationOperations(tmp_path)
        baseline = _artifact(tmp_path, "run-base")
        _write_artifact(tmp_path, "baseline.json", baseline)
        with pytest.raises(ArtifactNotFoundError):
            operations.compare_runs("baseline", "missing")
