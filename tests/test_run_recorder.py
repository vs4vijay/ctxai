"""Unit tests for the HH-04 run transcript recorder.

Covers seq monotonicity, the on-disk schema, seeded-secret redaction,
repository-relative path normalization, deterministic injected clocks,
dataclass round-trips, retention pruning, disabled/no-op mode, recorder
failure isolation, config fields, and the TaskRun.to_event_payloads helper.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ctxai.agent.config import AgentBehaviorConfig, AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.run_recorder import (
    RUN_SCHEMA_VERSION,
    NullRunRecorder,
    RunEvent,
    RunEventKind,
    RunRecorder,
    create_recorder,
    new_run_id,
    prune_runs,
    runs_dir_for,
)
from ctxai.agent.sessions import SessionStore
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import CheckEvidence, TaskRun, TaskState, UsageRecord
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


def fixed_clock(start: datetime | None = None):
    """Return a clock callable advancing one second per call (deterministic tests).

    Args:
        start: Optional first instant; defaults to 2026-09-04T00:00:00Z.

    Returns:
        A zero-argument callable returning a fresh datetime each call.
    """
    instant = start or datetime(2026, 9, 4, tzinfo=timezone.utc)
    state = {"value": instant}

    def _clock() -> datetime:
        current = state["value"]
        state["value"] = datetime.fromtimestamp(current.timestamp() + 1, tz=timezone.utc)
        return current

    return _clock


def read_events(path: Path) -> list[RunEvent]:
    """Parse a transcript file into RunEvents (test helper).

    Args:
        path: The .jsonl transcript path.

    Returns:
        The list of parsed RunEvent objects.
    """
    return [RunEvent.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestRunEventRoundTrip:
    def test_run_event_to_dict_from_dict_round_trip(self):
        """RunEvent survives a to_dict/from_dict round trip unchanged."""
        event = RunEvent(
            schema_version=RUN_SCHEMA_VERSION,
            run_id="abc123",
            seq=7,
            timestamp="2026-09-04T00:00:00+00:00",
            kind=RunEventKind.TOOL_CALL,
            payload={"tool": "read_file", "parameters": {"path": "note.txt"}},
            usage=None,
        )

        restored = RunEvent.from_dict(event.to_dict())

        assert restored == event
        assert restored.to_dict() == event.to_dict()

    def test_run_event_usage_round_trip(self):
        """An event carrying a usage payload round-trips with it."""
        event = RunEvent(
            schema_version=RUN_SCHEMA_VERSION,
            run_id="abc123",
            seq=1,
            timestamp="2026-09-04T00:00:00+00:00",
            kind=RunEventKind.LLM_CALL,
            payload={"call_index": 1},
            usage={
                "provider": "MockLLMProvider",
                "model": "m",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        )

        restored = RunEvent.from_dict(event.to_dict())

        assert restored.usage == event.usage
        assert restored == event

    def test_run_event_kinds_cover_the_contract(self):
        """Every kind named by the Part II contract exists in the enum."""
        expected = {
            "run_started",
            "user_message",
            "llm_call",
            "tool_call",
            "tool_result",
            "approval",
            "state_transition",
            "check",
            "compaction",
            "cancellation",
            "rollback",
            "run_completed",
        }

        assert {kind.value for kind in RunEventKind} == expected


class TestRunRecorder:
    def test_lines_carry_schema_version_and_monotonic_seq(self, temp_dir):
        """Line 1 is run_started and every line carries schema_version; seq is strictly increasing."""
        recorder = RunRecorder(temp_dir, "run-1", clock=fixed_clock())
        recorder.record(RunEventKind.RUN_STARTED, {"goal": "do things"})
        for index in range(5):
            recorder.record(RunEventKind.TOOL_CALL, {"index": index})
        recorder.close()

        lines = (temp_dir / ".ctxai" / "runs" / "run-1.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 6

        events = [json.loads(line) for line in lines]
        assert all(event["schema_version"] == RUN_SCHEMA_VERSION for event in events)
        assert all(event["run_id"] == "run-1" for event in events)
        assert events[0]["kind"] == "run_started"
        seqs = [event["seq"] for event in events]
        assert seqs == list(range(1, 7)), "seq starts at 1 and increases strictly"

    def test_record_appends_atomically_per_line(self, temp_dir):
        """Each record appends exactly one complete JSON line visible on disk."""
        recorder = RunRecorder(temp_dir, "run-2", clock=fixed_clock())
        recorder.record(RunEventKind.RUN_STARTED, {})
        recorder.record(RunEventKind.USER_MESSAGE, {"content": "hello"})
        recorder.close()

        lines = (temp_dir / ".ctxai" / "runs" / "run-2.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["payload"] == {"content": "hello"}

    def test_payloads_are_redacted_before_write(self, temp_dir):
        """Seeded secret shapes and secret-bearing keys never reach the file."""
        recorder = RunRecorder(temp_dir, "run-3", clock=fixed_clock())
        recorder.record(
            RunEventKind.TOOL_RESULT,
            {
                "result": "connected with api_key=sk-live-abcdef1234567890 and Bearer gh.tokenvalue123",
                "token": "ghp_0123456789abcdef",
                "password": "hunter2",
                "nested": {"authorization": "Bearer abcdefgh1234"},
            },
        )
        recorder.close()

        raw = (temp_dir / ".ctxai" / "runs" / "run-3.jsonl").read_text(encoding="utf-8")
        assert "sk-live-abcdef1234567890" not in raw
        assert "ghp_0123456789abcdef" not in raw
        assert "hunter2" not in raw
        assert "abcdefgh1234" not in raw
        assert "[REDACTED]" in raw

    def test_absolute_home_paths_are_normalized(self, temp_dir):
        """Project paths become repository-relative and home paths become ~/..."""
        home_file = Path.home() / "ctxai-hh04-redaction-probe.txt"
        recorder = RunRecorder(temp_dir, "run-4", clock=fixed_clock())
        recorder.record(
            RunEventKind.TOOL_CALL,
            {
                "path": str(temp_dir / "src" / "note.txt"),
                "home_file": str(home_file),
            },
        )
        recorder.close()

        raw = (temp_dir / ".ctxai" / "runs" / "run-4.jsonl").read_text(encoding="utf-8")
        assert str(temp_dir) not in raw, "absolute project paths must be normalized"
        assert "src/note.txt" in raw
        assert str(Path.home()) not in raw, "absolute home paths must be normalized"

    def test_injected_clock_determines_timestamps(self, temp_dir):
        """Timestamps come from the injected clock, not the wall clock."""
        clock = fixed_clock()
        recorder = RunRecorder(temp_dir, "run-5", clock=clock)
        recorder.record(RunEventKind.RUN_STARTED, {})
        recorder.record(RunEventKind.RUN_COMPLETED, {})
        recorder.close()

        events = read_events(temp_dir / ".ctxai" / "runs" / "run-5.jsonl")
        assert events[0].timestamp == "2026-09-04T00:00:00+00:00"
        assert events[1].timestamp == "2026-09-04T00:00:01+00:00"

    def test_write_failure_is_isolated_and_surfaced(self, temp_dir):
        """A failing write never raises, is surfaced via on_error, and recording continues."""
        failures: list[str] = []
        recorder = RunRecorder(temp_dir, "run-6", clock=fixed_clock(), on_error=failures.append)
        recorder.record(RunEventKind.RUN_STARTED, {})
        # A payload that cannot serialize to JSON must not raise out of record().
        recorder.record(RunEventKind.TOOL_CALL, {"bad": object()})
        recorder.record(RunEventKind.RUN_COMPLETED, {"status": "succeeded"})
        recorder.close()

        assert len(failures) >= 1, "the failure was surfaced as a diagnostic"
        assert recorder.error_count >= 1
        events = read_events(temp_dir / ".ctxai" / "runs" / "run-6.jsonl")
        kinds = [event.kind for event in events]
        assert kinds == ["run_started", "run_completed"], "recording continued after the failed write"

    def test_recorder_rejects_unsafe_run_id(self, temp_dir):
        """Run ids cannot traverse out of the runs directory."""
        with pytest.raises(ValueError):
            RunRecorder(temp_dir, "../escape", clock=fixed_clock())


class TestDisabledAndFailureModes:
    def test_disabled_mode_returns_noop_recorder(self, temp_dir):
        """create_recorder(enabled=False) returns a no-op that writes nothing."""
        recorder = create_recorder(temp_dir, new_run_id(), enabled=False)

        assert isinstance(recorder, NullRunRecorder)
        recorder.record(RunEventKind.RUN_STARTED, {"goal": "x"})
        recorder.close()
        assert not runs_dir_for(temp_dir).exists() or not list(runs_dir_for(temp_dir).glob("*.jsonl"))

    def test_construction_failure_falls_back_to_noop_with_diagnostic(self, temp_dir):
        """If the transcript file cannot be opened, a no-op recorder and a diagnostic are used."""
        failures: list[str] = []
        blocker = temp_dir / ".ctxai"
        blocker.mkdir()
        (blocker / "runs").write_text("not a directory", encoding="utf-8")

        recorder = create_recorder(temp_dir, "run-7", enabled=True, on_error=failures.append)

        assert isinstance(recorder, NullRunRecorder)
        assert failures, "the construction failure was surfaced as a diagnostic"
        recorder.record(RunEventKind.RUN_STARTED, {})
        recorder.close()


class TestRetentionPruning:
    def test_prune_deletes_oldest_beyond_keep(self, temp_dir):
        """prune_runs keeps the newest files and deletes the oldest beyond the limit."""
        runs_dir = runs_dir_for(temp_dir)
        runs_dir.mkdir(parents=True)
        for index in range(5):
            path = runs_dir / f"run-{index}.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            os.utime(path, (1_000_000 + index, 1_000_000 + index))

        deleted = prune_runs(runs_dir, keep=2)

        remaining = sorted(path.name for path in runs_dir.glob("*.jsonl"))
        assert remaining == ["run-3.jsonl", "run-4.jsonl"]
        assert sorted(path.name for path in deleted) == ["run-0.jsonl", "run-1.jsonl", "run-2.jsonl"]

    def test_prune_is_scoped_to_the_runs_directory(self, temp_dir):
        """prune_runs only touches .jsonl files directly inside the resolved directory."""
        runs_dir = runs_dir_for(temp_dir)
        runs_dir.mkdir(parents=True)
        stranger = temp_dir / "keep-me.jsonl"
        stranger.write_text("{}\n", encoding="utf-8")
        nested = runs_dir / "nested"
        nested.mkdir()
        (nested / "deep.jsonl").write_text("{}\n", encoding="utf-8")
        for index in range(4):
            path = runs_dir / f"run-{index}.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            os.utime(path, (2_000_000 + index, 2_000_000 + index))

        prune_runs(runs_dir, keep=1)

        assert stranger.exists(), "files outside the runs directory are never touched"
        assert (nested / "deep.jsonl").exists(), "nested directories are never touched"

    def test_prune_handles_missing_directory(self, temp_dir):
        """A missing runs directory prunes nothing and does not raise."""
        assert prune_runs(runs_dir_for(temp_dir), keep=3) == []


class TestConfigFields:
    def test_behavior_config_round_trip(self):
        """record_runs/run_retention survive to_dict/from_dict."""
        config = AgentBehaviorConfig(record_runs=False, run_retention=7)

        restored = AgentBehaviorConfig.from_dict(config.to_dict())

        assert restored.record_runs is False
        assert restored.run_retention == 7
        assert restored == config

    def test_defaults_preserve_current_behavior(self):
        """Defaults keep recording on with a 50-run retention window."""
        config = AgentBehaviorConfig()

        assert config.record_runs is True
        assert config.run_retention == 50

    def test_invalid_retention_is_rejected(self):
        """Non-positive retention is rejected before any work begins."""
        with pytest.raises(ValueError):
            AgentBehaviorConfig(run_retention=0)


class TestUsageRecordRoundTrip:
    def test_usage_record_to_dict_from_dict_round_trip(self):
        """UsageRecord survives a to_dict/from_dict round trip unchanged."""
        record = UsageRecord(
            provider="MockLLMProvider", model="mock-model", prompt_tokens=11, completion_tokens=5, total_tokens=16
        )

        restored = UsageRecord.from_dict(record.to_dict())

        assert restored == record


class TestTaskRunEventPayloads:
    def test_to_event_payloads_drains_new_events_only(self, temp_dir):
        """to_event_payloads returns state transitions, approvals, and checks; a second call returns only new ones."""
        run = TaskRun("do the thing", project_root=temp_dir)
        assert run.to_event_payloads() == [], "the initial state is part of run_started, not a transition"

        run.transition(TaskState.RETRIEVE)
        run.approvals.append({"tool": "write_file", "parameters": {"path": "a.txt"}, "approved": True})
        run.checks.append(CheckEvidence(command="pytest", success=True, output="ok"))

        events = run.to_event_payloads()
        kinds = [kind.value for kind, _payload in events]
        assert kinds == ["state_transition", "approval", "check"]
        payloads = dict((kind.value, payload) for kind, payload in events)
        assert payloads["state_transition"]["state"] == "retrieve"
        assert payloads["approval"]["approved"] is True
        assert payloads["check"]["command"] == "pytest"

        assert run.to_event_payloads() == [], "already-emitted events are not re-emitted"

        run.transition(TaskState.FAILED)
        events = run.to_event_payloads()
        assert [(kind.value, payload["state"]) for kind, payload in events] == [("state_transition", "failed")]

    def test_transitions_reconstruct_from_run_started_plus_events(self, temp_dir):
        """Run transitions equal the initial state plus drained state_transition payloads."""
        run = TaskRun("inspect and finish", project_root=temp_dir)
        run.transition(TaskState.RETRIEVE)
        run.transition(TaskState.EXECUTE)
        run.transition(TaskState.SUMMARIZE)

        drained = [payload["state"] for kind, payload in run.to_event_payloads() if kind.value == "state_transition"]

        assert [TaskState.UNDERSTAND.value, *drained] == [state.value for state in run.transitions]


class TestLoopLevelFailureIsolation:
    async def test_run_completes_when_every_recording_raises(self, temp_dir, monkeypatch):
        """A recorder whose record() always raises can never fail the run."""

        def boom(self, kind, payload, usage=None):
            raise OSError("disk full")

        monkeypatch.setattr(RunRecorder, "record", boom)

        config = AgentLLMConfig(provider="mock", model="mock-model", api_key="mock-key")
        provider = MockLLMProvider(config=config, responses=[create_mock_response(content="Finished cleanly.")])
        context = ToolExecutionContext.for_project(temp_dir)
        registry = ToolRegistry()
        registry.register(ReadFileTool(context=context, max_output_chars=20_000))
        agent = Agent(
            AgentLoopConfig(
                llm_provider=provider,
                tool_registry=registry,
                agent_config=AgentConfig(),
                working_directory=temp_dir,
                available_indexes=[],
                require_user_approval=True,
                approval_callback=lambda call: True,
            )
        )

        report = await agent.process_message("Just answer")

        assert "Status: succeeded" in report, "recording failures never fail the run"
        assert provider.call_count == 1


class TestSessionStoreStillWorks:
    def test_sessions_store_unchanged_by_recorder_import(self, temp_dir):
        """The shared redaction primitive keeps session persistence working."""
        store = SessionStore(temp_dir)
        assert store.storage_dir == temp_dir.resolve() / ".ctxai" / "sessions"
