"""RE-02 unit tests: privacy-preserving retrieval observability.

Covers the trace recorder modes (off/metrics/full), redaction of seeded
secrets and absolute paths, retention (count + age), corruption recovery,
injected-clock stage timing, failure isolation, and the retrieval-trace CLI.
"""

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxai.app import app
from ctxai.chunking import CodeChunker
from ctxai.config import RetrievalConfig
from ctxai.index_manifest import IndexedFile, IndexManifest
from ctxai.repository_context import ContextAssembler, HybridRetriever, retrieve_evidence
from ctxai.retrieval_traces import (
    TRACE_SCHEMA_VERSION,
    NullRetrievalTraceRecorder,
    RetrievalRunRecord,
    StageClock,
    TraceCorruptError,
    TraceNotFoundError,
    TraceSettings,
    create_recorder,
    delete_all_trace_runs,
    delete_trace_run,
    list_trace_runs,
    load_run_record,
    privacy_warning,
    query_hash,
    resolve_trace_dir,
    resolve_trace_settings,
    trace_dir_for,
)
from ctxai.vector_store import VectorStore
from tests.mocks.mock_embeddings import MockEmbeddingProvider

SERVICE_PY = '''def fetch_data():
    """Load rows for the report."""
    return [1, 2, 3]


def run_pipeline():
    """Call the loader and summarize the rows."""
    return sum(fetch_data())
'''

TEST_SERVICE_PY = '''def test_fetch_data():
    """Verify the loader returns rows."""
    assert fetch_data() == [1, 2, 3]
'''

NOISE_PY = '''def archive_helper(payload):
    """Background module with unrelated archive words."""
    return sorted(filter(None, payload))
'''

# Seeds used by the privacy tests. These are fake but must exercise the
# redaction formats: an sk- style key, a bearer token, an api_key assignment,
# a URL with credentials, and an absolute home path.
SEEDED_API_KEY = "sk-abc123def456ghi789jkl012"
SEEDED_BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc123def456"
SEEDED_ASSIGNMENT = "api_key=super-secret-value-123"
SEEDED_URL = "https://user:hunter2@example.com/api"
SEEDED_HOME_PATH = "/Users/alice/.ctxai/keys.json"


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_indexed_project(project: Path, noise_extra: str = "") -> None:
    """Create a small indexed project (service, its test, one noise file)."""
    files = {"service.py": SERVICE_PY, "test_service.py": TEST_SERVICE_PY, "noise.py": NOISE_PY + noise_extra}
    for name, content in files.items():
        (project / name).write_text(content, encoding="utf-8")
    provider = MockEmbeddingProvider()
    chunker = CodeChunker()
    chunks = []
    for name in sorted(files):
        chunks.extend(chunker.chunk_file(project / name))
    index_path = project / ".ctxai" / "indexes" / "trace-index"
    store = VectorStore(index_path, "trace-index")
    store.add_chunks(chunks, provider.generate_embeddings([chunk.content for chunk in chunks]))
    manifest = IndexManifest.create(
        index_name="trace-index",
        repository_root=project,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_dimension=provider.get_dimension(),
    )
    manifest.files = {
        str(project / name): IndexedFile(sha256=hashlib_sha256(project / name), chunks=1) for name in files
    }
    manifest.file_count = len(files)
    manifest.chunk_count = len(chunks)
    manifest.save(index_path)


@pytest.fixture
def indexed_project(tmp_path):
    build_indexed_project(tmp_path)
    return tmp_path


class FakeClock:
    """Deterministic datetime clock (record timestamps)."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        current = self.now
        self.now += timedelta(seconds=1)
        return current


class FakePerfClock:
    """Deterministic monotonic clock for stage timing (microsecond steps)."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001  # 1 ms per read
        return self.value


def metrics_settings(**overrides) -> TraceSettings:
    return resolve_trace_settings(RetrievalConfig(), mode="metrics", **overrides)


def trace_retrieve(project: Path, query: str, settings: TraceSettings, **kwargs):
    from ctxai.config import EmbeddingConfig

    provider = MockEmbeddingProvider(EmbeddingConfig(provider="mock", model="mock-model"))
    return retrieve_evidence(
        project,
        query,
        embedding_provider=provider,
        index_name="trace-index",
        trace=settings,
        clock=kwargs.pop("clock", None),
        **kwargs,
    )


def read_trace_payload(trace_dir: Path, run_id: str) -> dict:
    lines = [line for line in (trace_dir / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    return json.loads(lines[0])


# ----------------------------------------------------------------------
# Settings resolution and configuration
# ----------------------------------------------------------------------


class TestTraceSettings:
    def test_default_config_traces_nothing(self):
        settings = resolve_trace_settings(RetrievalConfig())
        assert settings.mode == "off"
        assert settings.enabled is False

    def test_trace_flag_promotes_off_config_to_metrics(self):
        settings = resolve_trace_settings(RetrievalConfig(), enabled=True)
        assert settings.mode == "metrics"
        assert settings.query_text == "hash"
        assert settings.source_preview == "omit"

    def test_config_mode_enables_persistence_without_flag(self):
        settings = resolve_trace_settings(RetrievalConfig(trace_mode="metrics"))
        assert settings.mode == "metrics"

    def test_explicit_disable_wins_over_config(self):
        settings = resolve_trace_settings(RetrievalConfig(trace_mode="metrics"), enabled=False)
        assert settings.mode == "off"

    def test_metrics_mode_never_stores_raw_query_or_source(self):
        settings = resolve_trace_settings(
            RetrievalConfig(trace_mode="metrics", trace_query_text="store", trace_source_preview="store")
        )
        assert settings.mode == "metrics"
        assert settings.query_text == "hash"
        assert settings.source_preview == "omit"

    def test_full_mode_honors_store_settings(self):
        settings = resolve_trace_settings(
            RetrievalConfig(trace_mode="full", trace_query_text="store", trace_source_preview="store")
        )
        assert settings.mode == "full"
        assert settings.query_text == "store"
        assert settings.source_preview == "store"

    def test_full_mode_privacy_warning_is_returned_once(self):
        assert privacy_warning("full") is not None
        assert privacy_warning("metrics") is None
        assert privacy_warning("off") is None

    def test_settings_round_trip(self):
        settings = TraceSettings(
            mode="full",
            query_text="store",
            source_preview="store",
            retention=10,
            retention_days=7,
            trace_dir=None,
            preview_chars=200,
        )
        assert TraceSettings.from_dict(settings.to_dict()) == settings

    def test_config_round_trip_includes_trace_fields(self):
        config = RetrievalConfig(
            trace_mode="full",
            trace_query_text="store",
            trace_source_preview="store",
            trace_retention=5,
            trace_retention_days=9,
        )
        assert RetrievalConfig.from_dict(config.to_dict()) == config
        assert RetrievalConfig().trace_mode == "off"

    def test_config_rejects_invalid_trace_values(self):
        with pytest.raises(ValueError):
            RetrievalConfig(trace_mode="verbose")
        with pytest.raises(ValueError):
            RetrievalConfig(trace_query_text="yell")
        with pytest.raises(ValueError):
            RetrievalConfig(trace_source_preview="keep")
        with pytest.raises(ValueError):
            RetrievalConfig(trace_retention=0)
        with pytest.raises(ValueError):
            RetrievalConfig(trace_retention_days=0)


# ----------------------------------------------------------------------
# Recorder modes and record shape
# ----------------------------------------------------------------------


class TestRecorderModes:
    def test_off_mode_writes_nothing(self, indexed_project):
        recorder = create_recorder(indexed_project, resolve_trace_settings(RetrievalConfig()))
        assert isinstance(recorder, NullRetrievalTraceRecorder)
        assert trace_dir_for(indexed_project).exists() is False

    def test_metrics_mode_writes_one_line_per_run(self, indexed_project):
        settings = metrics_settings()
        recorder = create_recorder(indexed_project, settings, run_id="run-abc", timestamp_clock=FakeClock())
        run = RetrievalRunRecord(
            run_id=recorder.run_id,
            query_id=recorder.query_id,
            timestamp="2026-09-05T12:00:00+00:00",
            mode="metrics",
            status="ok",
        )
        outcome = recorder.record(run)
        assert outcome.recorded is True
        assert outcome.diagnostic is None
        trace_dir = trace_dir_for(indexed_project)
        payload = read_trace_payload(trace_dir, "run-abc")
        assert payload["schema_version"] == TRACE_SCHEMA_VERSION
        assert payload["run_id"] == "run-abc"

    def test_metrics_record_carries_no_query_text_or_source(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification rows", metrics_settings())
        assert evidence.trace is not None and evidence.trace.recorded
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        query_block = payload["query"]
        assert query_block["text"] is None
        assert query_block["hash"] == query_hash("loader verification rows")
        for candidate in payload["candidates"]:
            assert "preview" not in candidate
            assert "content" not in candidate
        # The chunk ids and citations are identifiers, never source text.
        assert "fetch_data" not in json.dumps(payload["candidates"])

    def test_omit_query_recording_drops_even_the_hash(self, indexed_project):
        settings = resolve_trace_settings(RetrievalConfig(trace_mode="metrics", trace_query_text="omit"), enabled=True)
        evidence = trace_retrieve(indexed_project, "loader verification rows", settings)
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        assert payload["query"] == {
            "recording": "omit",
            "text": None,
            "hash": None,
            "length": len("loader verification rows"),
        }

    def test_full_mode_stores_query_and_bounded_preview(self, indexed_project):
        settings = resolve_trace_settings(
            RetrievalConfig(
                trace_mode="full", trace_query_text="store", trace_source_preview="store", trace_retention=5
            ),
            enabled=True,
        )
        evidence = trace_retrieve(indexed_project, "loader verification rows", settings)
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        assert payload["query"]["text"] == "loader verification rows"
        previews = [candidate["preview"] for candidate in payload["candidates"] if "preview" in candidate]
        assert previews, "full mode with store previews records selected source"
        assert all(len(preview) <= settings.preview_chars for preview in previews)

    def test_record_round_trip(self):
        record = RetrievalRunRecord(
            run_id="run-1",
            query_id="q-1",
            timestamp="2026-09-05T12:00:00+00:00",
            mode="metrics",
            status="ok",
            index={"name": "trace-index"},
        )
        assert RetrievalRunRecord.from_dict(record.to_dict()) == record


# ----------------------------------------------------------------------
# Traced retrieval content (criterion 1, unit level)
# ----------------------------------------------------------------------


class TestTracedRetrievalContent:
    def test_run_records_generators_candidates_and_identity(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", metrics_settings())
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        assert payload["schema_version"] == TRACE_SCHEMA_VERSION
        generators = {generator["component"] for generator in payload["generators"]}
        assert {"semantic", "lexical", "symbol", "repository-map"} <= generators
        candidates = payload["candidates"]
        assert [candidate["final_rank"] for candidate in candidates] == list(range(1, len(candidates) + 1))
        assert payload["candidate_count"] == len(candidates)
        assert payload["selected_count"] == len(payload["selected"])
        assert payload["index"]["name"] == "trace-index"
        assert payload["index"]["embedding_provider"] == "mock"
        assert payload["configuration"]["token_budget"] > 0
        assert payload["configuration"]["fingerprint"]
        assert payload["network"]["recorder_transport"] == "local-file-only"
        assert payload["network"]["outbound_transports"] == []

    def test_stage_and_total_timings_present(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", metrics_settings())
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        timings = payload["stage_timings_ms"]
        for stage in (
            "load_records",
            "semantic_candidates",
            "lexical_candidates",
            "symbol_candidates",
            "structure_candidates",
            "fusion",
            "final_rank",
            "assemble",
        ):
            assert stage in timings
        assert payload["total_latency_ms"] >= 0

    def test_injected_clock_produces_deterministic_stage_timings(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", metrics_settings(), clock=FakePerfClock())
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        # FakePerfClock advances 1 ms per read; each stage consumed at least one read.
        assert all(value >= 0.5 for value in payload["stage_timings_ms"].values())
        # A second identical run produces identical stage timings.
        evidence2 = trace_retrieve(indexed_project, "loader verification", metrics_settings(), clock=FakePerfClock())
        payload2 = read_trace_payload(trace_dir_for(indexed_project), evidence2.trace.run_id)
        assert payload2["stage_timings_ms"] == payload["stage_timings_ms"]

    def test_tracing_does_not_change_ordering(self, indexed_project):
        untraced = trace_retrieve(indexed_project, "loader verification", resolve_trace_settings(RetrievalConfig()))
        traced = trace_retrieve(indexed_project, "loader verification", metrics_settings())
        assert [item.id for item in untraced.items] == [item.id for item in traced.items]
        assert [item.citation for item in untraced.context.items] == [item.citation for item in traced.context.items]

    def test_exclusions_and_decisions_recorded(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", metrics_settings(), token_budget=1, limit=20)
        payload = read_trace_payload(trace_dir_for(indexed_project), evidence.trace.run_id)
        decisions = {candidate["decision"] for candidate in payload["candidates"]}
        assert "budget" in decisions or "not_selected" in decisions
        assert payload["excluded"], "budget-limited assembly records examined-but-not-selected items"

    def test_off_trace_returns_no_outcome(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", resolve_trace_settings(RetrievalConfig()))
        assert evidence.trace is None

    def test_retrieval_failure_is_observable(self, tmp_path):
        settings = metrics_settings()
        with pytest.raises(LookupError):
            trace_retrieve(tmp_path, "anything", settings)
        summaries, corrupt = list_trace_runs(trace_dir_for(tmp_path))
        assert not corrupt
        assert len(summaries) == 1
        assert summaries[0].status == "error"
        record = load_run_record(summaries[0].run_id, trace_dir_for(tmp_path))
        assert record.status == "error"
        assert record.errors


# ----------------------------------------------------------------------
# Failure isolation (criterion 4)
# ----------------------------------------------------------------------


class TestFailureIsolation:
    def test_recorder_construction_failure_does_not_fail_retrieval(self, indexed_project, monkeypatch):
        settings = metrics_settings()
        # Force the trace directory to be unwritable: put a file where the
        # directory would be created.
        trace_dir = trace_dir_for(indexed_project)
        trace_dir.parent.mkdir(parents=True, exist_ok=True)
        trace_dir.write_text("not a directory", encoding="utf-8")
        evidence = trace_retrieve(indexed_project, "loader verification", settings)
        assert evidence.items, "retrieval still succeeds"
        assert evidence.trace is not None
        assert evidence.trace.recorded is False
        assert evidence.trace.diagnostic

    def test_recorder_write_failure_is_a_diagnostic(self, indexed_project, monkeypatch):
        settings = metrics_settings()
        recorder = create_recorder(indexed_project, settings, run_id="run-broken")

        def broken_dump(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr("ctxai.retrieval_traces.json.dumps", broken_dump)
        run = RetrievalRunRecord(
            run_id=recorder.run_id,
            query_id=recorder.query_id,
            timestamp="2026-09-05T12:00:00+00:00",
            mode="metrics",
            status="ok",
        )
        outcome = recorder.record(run)
        assert outcome.recorded is False
        assert outcome.diagnostic is not None and "disk full" in outcome.diagnostic


# ----------------------------------------------------------------------
# Redaction and privacy (criterion 3, unit level)
# ----------------------------------------------------------------------


class TestPrivacy:
    @pytest.fixture
    def seeded_project(self, tmp_path):
        # Seed a secret into source content before indexing so a source
        # preview would carry it unless redaction removes it.
        build_indexed_project(tmp_path, noise_extra=f'\nTOKEN = "{SEEDED_API_KEY}"\n')
        return tmp_path

    @pytest.mark.parametrize("mode", ["metrics", "full"])
    def test_no_seeded_secret_reaches_persisted_traces(self, seeded_project, mode):
        query = f"search {SEEDED_API_KEY} {SEEDED_BEARER} {SEEDED_ASSIGNMENT} {SEEDED_URL} {SEEDED_HOME_PATH}"
        settings = resolve_trace_settings(
            RetrievalConfig(trace_mode=mode, trace_query_text="store", trace_source_preview="store")
        )
        evidence = trace_retrieve(seeded_project, query, settings)
        assert evidence.trace is not None and evidence.trace.recorded
        raw = (trace_dir_for(seeded_project) / f"{evidence.trace.run_id}.jsonl").read_text(encoding="utf-8")
        assert SEEDED_API_KEY not in raw
        assert "super-secret-value-123" not in raw
        assert "eyJhbGciOiJIUzI1NiJ9" not in raw
        assert "hunter2" not in raw
        assert "/Users/alice" not in raw

    def test_metrics_mode_stores_no_source_content_at_all(self, seeded_project):
        evidence = trace_retrieve(seeded_project, "archive helper", metrics_settings())
        payload = read_trace_payload(trace_dir_for(seeded_project), evidence.trace.run_id)
        blob = json.dumps(payload)
        assert "Background module" not in blob
        assert "def archive_helper" not in blob

    def test_error_message_redaction(self, indexed_project):
        settings = metrics_settings()
        recorder = create_recorder(indexed_project, settings, run_id="run-err", timestamp_clock=FakeClock())
        record = RetrievalRunRecord(
            run_id=recorder.run_id,
            query_id=recorder.query_id,
            timestamp="2026-09-05T12:00:00+00:00",
            mode="metrics",
            status="error",
            errors=[f"failed at {SEEDED_HOME_PATH} with {SEEDED_API_KEY}"],
        )
        outcome = recorder.record(record)
        assert outcome.recorded
        raw = (trace_dir_for(indexed_project) / "run-err.jsonl").read_text(encoding="utf-8")
        assert SEEDED_API_KEY not in raw
        assert "/Users/alice" not in raw


# ----------------------------------------------------------------------
# Retention, deletion, corruption, concurrency (criterion 5)
# ----------------------------------------------------------------------


class TestRetentionAndLifecycle:
    def _record_for(self, recorder: object, run_id: str) -> RetrievalRunRecord:
        return RetrievalRunRecord(
            run_id=run_id,
            query_id=f"q-{run_id}",
            timestamp="2026-09-05T12:00:00+00:00",
            mode="metrics",
            status="ok",
        )

    def test_retention_count_prunes_oldest(self, tmp_path):
        settings = TraceSettings(
            mode="metrics", query_text="hash", source_preview="omit", retention=3, retention_days=365
        )
        recorder = create_recorder(tmp_path, settings, timestamp_clock=FakeClock())
        for index in range(5):
            recorder.record(self._record_for(recorder, f"run-{index}"))
        summaries, _ = list_trace_runs(trace_dir_for(tmp_path))
        assert {summary.run_id for summary in summaries} == {"run-2", "run-3", "run-4"}

    def test_retention_days_prunes_expired(self, tmp_path):
        settings = TraceSettings(
            mode="metrics", query_text="hash", source_preview="omit", retention=100, retention_days=30
        )
        recorder = create_recorder(tmp_path, settings, timestamp_clock=FakeClock())
        recorder.record(self._record_for(recorder, "run-old"))
        recorder.record(self._record_for(recorder, "run-new"))
        old_path = trace_dir_for(tmp_path) / "run-old.jsonl"
        expired = datetime.now(timezone.utc) - timedelta(days=40)
        os.utime(old_path, (expired.timestamp(), expired.timestamp()))
        recorder.record(self._record_for(recorder, "run-trigger"))
        summaries, _ = list_trace_runs(trace_dir_for(tmp_path))
        assert {summary.run_id for summary in summaries} == {"run-new", "run-trigger"}

    def test_delete_is_scoped_to_the_trace_directory(self, tmp_path):
        trace_dir = resolve_trace_dir(tmp_path)
        trace_dir.mkdir(parents=True)
        (trace_dir / "run-1.jsonl").write_text("{}\n", encoding="utf-8")
        (trace_dir / "unrelated.json").write_text("keep me", encoding="utf-8")
        deleted = delete_trace_run("run-1", trace_dir)
        assert deleted.name == "run-1.jsonl"
        assert (trace_dir / "unrelated.json").exists()
        with pytest.raises(TraceNotFoundError):
            delete_trace_run("missing", trace_dir)

    def test_delete_all_removes_only_traces(self, tmp_path):
        trace_dir = resolve_trace_dir(tmp_path)
        trace_dir.mkdir(parents=True)
        (trace_dir / "run-1.jsonl").write_text("{}\n", encoding="utf-8")
        (trace_dir / "run-2.jsonl").write_text("{}\n", encoding="utf-8")
        nested = trace_dir / "nested"
        nested.mkdir()
        (nested / "run-3.jsonl").write_text("{}\n", encoding="utf-8")
        deleted = delete_all_trace_runs(trace_dir)
        assert deleted == 2
        assert (nested / "run-3.jsonl").exists()

    def test_corrupt_trace_is_skipped_with_diagnostic(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", metrics_settings())
        trace_dir = trace_dir_for(indexed_project)
        good = trace_dir / f"{evidence.trace.run_id}.jsonl"
        good.write_text('{"schema_version": 1, "run_id": "broken"', encoding="utf-8")  # truncated
        summaries, corrupt = list_trace_runs(trace_dir)
        assert summaries == []
        assert corrupt and "broken" in corrupt[0]
        with pytest.raises(TraceCorruptError):
            load_run_record("broken", trace_dir)

    def test_missing_trace_raises_not_found(self, tmp_path):
        trace_dir = resolve_trace_dir(tmp_path)
        trace_dir.mkdir(parents=True)
        with pytest.raises(TraceNotFoundError):
            load_run_record("missing-run", trace_dir)

    def test_concurrent_writers_do_not_corrupt_committed_runs(self, tmp_path):
        settings = TraceSettings(
            mode="metrics", query_text="hash", source_preview="omit", retention=500, retention_days=365
        )
        errors: list[str] = []

        def writer(index: int) -> None:
            recorder = create_recorder(tmp_path, settings, timestamp_clock=FakeClock())
            outcome = recorder.record(self._record_for(recorder, f"run-{index}"))
            if not outcome.recorded:
                errors.append(str(outcome.diagnostic))

        threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        summaries, corrupt = list_trace_runs(trace_dir_for(tmp_path))
        assert len(summaries) == 8
        assert not corrupt

    def test_trace_dir_override_must_stay_inside_project(self, tmp_path):
        inside = resolve_trace_dir(tmp_path, tmp_path / "custom-traces")
        assert inside == tmp_path / "custom-traces"
        with pytest.raises(ValueError):
            resolve_trace_dir(tmp_path, "/outside/the/project")


# ----------------------------------------------------------------------
# StageClock
# ----------------------------------------------------------------------


class TestStageClock:
    def test_stage_clock_records_stage_boundaries(self):
        clock = FakePerfClock()
        stage_clock = StageClock(clock)
        stage_clock.mark("one")
        stage_clock.mark("two")
        stage_clock.stop()
        assert stage_clock.stage_timings_ms["one"] == pytest.approx(1.0)
        assert stage_clock.stage_timings_ms["two"] == pytest.approx(1.0)

    def test_stage_clock_without_stop_drops_open_stage(self):
        stage_clock = StageClock(FakePerfClock())
        stage_clock.mark("one")
        assert stage_clock.stage_timings_ms == {}

    def test_repeated_stage_names_accumulate(self):
        stage_clock = StageClock(FakePerfClock())
        stage_clock.mark("loop")
        stage_clock.mark("other")
        stage_clock.mark("loop")
        stage_clock.stop()
        assert stage_clock.stage_timings_ms["loop"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# CLI: ctxai retrieval runs (criterion: inspect locally)
# ----------------------------------------------------------------------


class TestRetrievalRunsCLI:
    def test_list_show_delete_round_trip(self, indexed_project):
        evidence = trace_retrieve(indexed_project, "loader verification", metrics_settings())
        runner = CliRunner()

        listed = runner.invoke(app, ["retrieval", "runs", "list", "--project-path", str(indexed_project)])
        assert listed.exit_code == 0, listed.output
        assert evidence.trace.run_id[:12] in listed.output

        shown = runner.invoke(
            app, ["retrieval", "runs", "show", evidence.trace.run_id, "--project-path", str(indexed_project)]
        )
        assert shown.exit_code == 0, shown.output
        assert "trace-index" in shown.output

        as_json = runner.invoke(
            app,
            ["retrieval", "runs", "show", evidence.trace.run_id, "--json", "--project-path", str(indexed_project)],
        )
        assert as_json.exit_code == 0, as_json.output
        payload = json.loads(as_json.output)
        assert payload["schema_version"] == TRACE_SCHEMA_VERSION
        assert payload["run"]["run_id"] == evidence.trace.run_id

        deleted = runner.invoke(
            app, ["retrieval", "runs", "delete", evidence.trace.run_id, "--project-path", str(indexed_project)]
        )
        assert deleted.exit_code == 0, deleted.output
        listed_after = runner.invoke(app, ["retrieval", "runs", "list", "--project-path", str(indexed_project)])
        assert evidence.trace.run_id[:12] not in listed_after.output

    def test_list_json_envelope(self, indexed_project):
        trace_retrieve(indexed_project, "loader verification", metrics_settings())
        runner = CliRunner()
        result = runner.invoke(app, ["retrieval", "runs", "list", "--json", "--project-path", str(indexed_project)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["schema_version"] == TRACE_SCHEMA_VERSION
        assert len(payload["runs"]) == 1
        assert payload["runs"][0]["mode"] == "metrics"

    def test_list_index_filter(self, indexed_project):
        trace_retrieve(indexed_project, "loader verification", metrics_settings())
        runner = CliRunner()
        hit = runner.invoke(
            app,
            ["retrieval", "runs", "list", "--index", "trace-index", "--project-path", str(indexed_project)],
        )
        assert hit.exit_code == 0 and "trace-index" in hit.output
        miss = runner.invoke(
            app, ["retrieval", "runs", "list", "--index", "other-index", "--project-path", str(indexed_project)]
        )
        assert miss.exit_code == 0 and "trace-index" not in miss.output

    def test_delete_all_requires_confirmation(self, indexed_project):
        trace_retrieve(indexed_project, "loader verification", metrics_settings())
        runner = CliRunner()
        aborted = runner.invoke(
            app, ["retrieval", "runs", "delete", "--all", "--project-path", str(indexed_project)], input="n\n"
        )
        assert aborted.exit_code != 0
        assert len(list(trace_dir_for(indexed_project).glob("*.jsonl"))) == 1
        confirmed = runner.invoke(
            app, ["retrieval", "runs", "delete", "--all", "--project-path", str(indexed_project)], input="y\n"
        )
        assert confirmed.exit_code == 0, confirmed.output
        assert list(trace_dir_for(indexed_project).glob("*.jsonl")) == []

    def test_show_invalid_run_id_fails_cleanly(self, indexed_project):
        runner = CliRunner()
        result = runner.invoke(app, ["retrieval", "runs", "show", "../escape", "--project-path", str(indexed_project)])
        assert result.exit_code == 1
        assert "Invalid" in result.output


# ----------------------------------------------------------------------
# Low-level sanity: tracing through HybridRetriever directly
# ----------------------------------------------------------------------


def test_context_assembler_unaffected_by_trace_module(indexed_project):
    retriever = HybridRetriever(indexed_project, MockEmbeddingProvider(), index_name="trace-index")
    result = retriever.retrieve_detailed("loader verification", limit=5)
    context = ContextAssembler(token_budget=2000).assemble(retriever.index_name or "", result.items)
    assert context.items
    assert result.component_counts
    assert result.component_ranks
