"""Unit tests for HH-06 checkpoints: capture, restore, safety, retention.

Covers the Checkpoint/CheckpointFile round trips, the ``AgentBehaviorConfig``
checkpoint fields, the ``CheckpointManager`` capture/restore semantics
(modified/created/deleted), the stale-worktree refusal with ``--force``,
path-escape and symlink refusal, retention pruning, the per-run size cap,
and the ``TaskRun.before_tool`` capture hook.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ctxai.agent.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_CHECKPOINT_MAX_BYTES,
    CaptureKind,
    CaptureOutcome,
    Checkpoint,
    CheckpointFile,
    CheckpointManager,
)
from ctxai.agent.config import AgentBehaviorConfig
from ctxai.agent.llm.base import ToolCall
from ctxai.agent.run_recorder import RunEventKind, RunRecorder
from ctxai.agent.workflow import TaskRun


class FakeClock:
    """Deterministic clock advancing one second per call."""

    def __init__(self) -> None:
        self._now = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


def make_manager(root: Path, **kwargs: object) -> CheckpointManager:
    """Build a manager with an injected deterministic clock.

    Args:
        root: Project root for the manager.
        **kwargs: Forwarded to :meth:`CheckpointManager.for_project`.

    Returns:
        The configured CheckpointManager.
    """
    return CheckpointManager.for_project(root, clock=FakeClock(), **kwargs)


def sha(data: bytes) -> str:
    """Hash bytes the way the manager does.

    Args:
        data: The bytes to hash.

    Returns:
        The hex sha256 digest.
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Dataclass round trips (Part V: every persisted artifact round-trips)
# ---------------------------------------------------------------------------


def test_checkpoint_file_round_trip_preserves_all_fields():
    """CheckpointFile to_dict/from_dict round-trips both capture kinds."""
    file_entry = CheckpointFile(
        path="src/a.py",
        kind=CaptureKind.FILE,
        sha256="a" * 64,
        blob="files/" + "a" * 64 + ".blob",
        size=12,
        post_run_sha256="b" * 64,
        post_run_present=True,
    )
    created_entry = CheckpointFile(path="new.txt", kind=CaptureKind.CREATED, post_run_present=False)
    for entry in (file_entry, created_entry):
        assert CheckpointFile.from_dict(entry.to_dict()) == entry


def test_checkpoint_round_trip_preserves_files_and_contract_fields():
    """Checkpoint round-trips and carries the Part II contract fields."""
    checkpoint = Checkpoint(
        checkpoint_id="run-1",
        run_id="run-1",
        created_at="2026-09-04T12:00:01+00:00",
        retained=True,
        files=[
            CheckpointFile(path="src/a.py", kind=CaptureKind.FILE, sha256="a" * 64, blob="files/x.blob", size=12),
            CheckpointFile(path="new.txt", kind=CaptureKind.CREATED),
        ],
        git_head="deadbeef",
        status="finalized",
        updated_at="2026-09-04T12:00:09+00:00",
        bytes_captured=12,
        cap_exceeded=False,
        project_root="/tmp/project",
    )
    assert checkpoint.paths == ["src/a.py", "new.txt"]
    restored = Checkpoint.from_dict(checkpoint.to_dict())
    assert restored == checkpoint
    assert restored.to_dict()["schema_version"] == CHECKPOINT_SCHEMA_VERSION


def test_behavior_config_checkpoint_defaults_and_round_trip():
    """checkpoint_retention/checkpoint_max_bytes default, round-trip, and validate."""
    config = AgentBehaviorConfig()
    assert config.checkpoint_retention == 20
    assert config.checkpoint_max_bytes == DEFAULT_CHECKPOINT_MAX_BYTES == 52_428_800
    restored = AgentBehaviorConfig.from_dict(config.to_dict())
    assert restored.checkpoint_retention == 20
    assert restored.checkpoint_max_bytes == DEFAULT_CHECKPOINT_MAX_BYTES
    custom = AgentBehaviorConfig.from_dict({"checkpoint_retention": 5, "checkpoint_max_bytes": 1024})
    assert custom.checkpoint_retention == 5
    assert custom.checkpoint_max_bytes == 1024
    with pytest.raises(ValueError):
        AgentBehaviorConfig(checkpoint_retention=0)
    with pytest.raises(ValueError):
        AgentBehaviorConfig(checkpoint_max_bytes=0)
    with pytest.raises(ValueError):
        CheckpointManager.for_project(Path("/tmp/x"), retention=0)


# ---------------------------------------------------------------------------
# Capture semantics
# ---------------------------------------------------------------------------


def test_capture_existing_file_stores_pre_mutation_bytes(temp_dir):
    """Capturing an existing file stores kind=file with a content-addressed blob."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"original bytes")
    manager.start_run("run-cap")
    outcome = manager.capture("run-cap", temp_dir / "a.txt")
    assert outcome.status == "captured"
    assert outcome.entry is not None
    assert outcome.entry.kind is CaptureKind.FILE
    assert outcome.entry.sha256 == sha(b"original bytes")
    assert outcome.entry.size == len(b"original bytes")

    blob = manager.run_dir("run-cap") / "files" / f"{outcome.entry.sha256}.blob"
    assert blob.read_bytes() == b"original bytes"

    checkpoint = manager.load("run-cap")
    assert checkpoint.status == "open"
    assert checkpoint.run_id == "run-cap"
    assert checkpoint.checkpoint_id == "run-cap"
    assert [entry.path for entry in checkpoint.files] == ["a.txt"]
    assert checkpoint.bytes_captured == len(b"original bytes")

    manifest = json.loads(manager.manifest_path("run-cap").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert not list(manager.run_dir("run-cap").glob("*.tmp")), "no temporary files are left behind"


def test_capture_missing_file_records_created_marker(temp_dir):
    """Capturing a path that does not exist records the created marker."""
    manager = make_manager(temp_dir)
    manager.start_run("run-new")
    outcome = manager.capture("run-new", temp_dir / "made-by-run.txt")
    assert outcome.status == "captured"
    assert outcome.entry is not None
    assert outcome.entry.kind is CaptureKind.CREATED
    assert outcome.entry.sha256 is None
    assert not list((manager.run_dir("run-new") / "files").glob("*.blob"))


def test_capture_is_first_touch_only(temp_dir):
    """A second capture of the same path is a no-op."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"v1")
    manager.start_run("run-twice")
    first = manager.capture("run-twice", temp_dir / "a.txt")
    (temp_dir / "a.txt").write_bytes(b"v2")
    second = manager.capture("run-twice", temp_dir / "a.txt")
    assert second.status == "already" and second.entry is first.entry
    assert len(manager.load("run-twice").files) == 1
    blob = manager.run_dir("run-twice") / "files" / f"{first.entry.sha256}.blob"
    assert blob.read_bytes() == b"v1", "the pre-mutation bytes are never overwritten"


def test_capture_refuses_paths_outside_project_and_symlinks(temp_dir):
    """Paths escaping the root, and symlinked paths, are refused for capture."""
    project = temp_dir / "project"
    project.mkdir()
    (temp_dir / "outside.txt").write_bytes(b"outside")
    os.symlink(temp_dir / "outside.txt", project / "link.txt")
    manager = make_manager(project)
    manager.start_run("run-safe")

    outside = manager.capture("run-safe", temp_dir / "outside.txt")
    assert outside.status == "refused"
    assert "escapes the project root" in (outside.reason or "")

    symlinked = manager.capture("run-safe", project / "link.txt")
    assert symlinked.status == "refused"
    assert "symlink" in (symlinked.reason or "")

    assert not manager.manifest_path("run-safe").exists(), "refused paths are never captured"


# ---------------------------------------------------------------------------
# Restore semantics
# ---------------------------------------------------------------------------


def test_restore_modified_file_returns_pre_run_bytes(temp_dir):
    """A captured modified file is restored byte-identically."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-mod")
    manager.capture("run-mod", temp_dir / "a.txt")
    (temp_dir / "a.txt").write_bytes(b"BBB")
    checkpoint = manager.finalize("run-mod", retained=False)
    assert checkpoint is not None and checkpoint.status == "finalized"

    result = manager.restore("run-mod")
    assert result.applied is True and result.ok
    assert [(item.path, item.action) for item in result.results] == [("a.txt", "restored")]
    assert (temp_dir / "a.txt").read_bytes() == b"AAA"


def test_restore_created_file_deletes_it(temp_dir):
    """A file the run created is removed by restore."""
    manager = make_manager(temp_dir)
    manager.start_run("run-create")
    manager.capture("run-create", temp_dir / "new.txt")
    (temp_dir / "new.txt").write_bytes(b"made by the run")
    manager.finalize("run-create", retained=False)

    result = manager.restore("run-create")
    assert result.applied is True
    assert [(item.path, item.action) for item in result.results] == [("new.txt", "deleted")]
    assert not (temp_dir / "new.txt").exists()


def test_restore_recreates_file_deleted_during_run(temp_dir):
    """A file captured earlier and deleted later in the run is recreated."""
    manager = make_manager(temp_dir)
    (temp_dir / "gone.txt").write_bytes(b"precious")
    manager.start_run("run-del")
    manager.capture("run-del", temp_dir / "gone.txt")
    (temp_dir / "gone.txt").unlink()
    manager.finalize("run-del", retained=False)

    checkpoint = manager.load("run-del")
    entry = checkpoint.files[0]
    assert entry.post_run_present is False, "finalization recorded the deletion"

    result = manager.restore("run-del")
    assert result.applied is True
    assert [(item.path, item.action) for item in result.results] == [("gone.txt", "recreated")]
    assert (temp_dir / "gone.txt").read_bytes() == b"precious"


def test_restore_refuses_stale_working_tree_and_force_overrides(temp_dir):
    """A post-run manual edit refuses restore with a per-file reason; --force proceeds."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-stale")
    manager.capture("run-stale", temp_dir / "a.txt")
    (temp_dir / "a.txt").write_bytes(b"BBB")
    manager.finalize("run-stale", retained=False)
    (temp_dir / "a.txt").write_bytes(b"user edited this after the run")

    result = manager.restore("run-stale")
    assert result.applied is False
    assert result.results[0].action == "refused"
    assert "hash mismatch" in (result.results[0].detail or "")
    assert (temp_dir / "a.txt").read_bytes() == b"user edited this after the run", "nothing was modified"

    forced = manager.restore("run-stale", force=True)
    assert forced.applied is True
    assert forced.results[0].action == "restored"
    assert (temp_dir / "a.txt").read_bytes() == b"AAA"


def test_restore_without_finalization_skips_stale_check(temp_dir):
    """An unfinalized checkpoint (crashed run) restores without the stale check."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-open")
    manager.capture("run-open", temp_dir / "a.txt")
    (temp_dir / "a.txt").write_bytes(b"BBB")

    result = manager.restore("run-open")
    assert result.applied is True
    assert (temp_dir / "a.txt").read_bytes() == b"AAA"


def test_finalize_records_post_run_hashes_and_retained_flag(temp_dir):
    """Finalization records per-file post-run presence/hashes and the retained flag."""
    manager = make_manager(temp_dir)
    (temp_dir / "kept.txt").write_bytes(b"before")
    (temp_dir / "later-deleted.txt").write_bytes(b"bye")
    manager.start_run("run-fin")
    manager.capture("run-fin", temp_dir / "kept.txt")
    manager.capture("run-fin", temp_dir / "made.txt")
    manager.capture("run-fin", temp_dir / "later-deleted.txt")
    (temp_dir / "kept.txt").write_bytes(b"after")
    (temp_dir / "made.txt").write_bytes(b"created")
    (temp_dir / "later-deleted.txt").unlink()

    manager.finalize("run-fin", retained=True)
    entries = {entry.path: entry for entry in manager.load("run-fin").files}
    assert entries["kept.txt"].post_run_present is True
    assert entries["kept.txt"].post_run_sha256 == sha(b"after")
    assert entries["made.txt"].post_run_present is True
    assert entries["made.txt"].post_run_sha256 == sha(b"created")
    assert entries["later-deleted.txt"].post_run_present is False
    checkpoint = manager.load("run-fin")
    assert checkpoint.retained is True and checkpoint.status == "finalized"


def test_finalize_without_captures_writes_no_checkpoint(temp_dir):
    """A run with no captured files leaves no manifest on disk."""
    manager = make_manager(temp_dir)
    manager.start_run("run-empty")
    assert manager.finalize("run-empty", retained=True) is None
    assert not manager.manifest_path("run-empty").exists()


def test_restore_refuses_unsafe_manifest_paths(temp_dir):
    """A tampered manifest path cannot escape the project root."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-tamper")
    manager.capture("run-tamper", temp_dir / "a.txt")
    manifest_path = manager.manifest_path("run-tamper")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../evil.txt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = manager.restore("run-tamper")
    assert result.applied is False
    assert result.results[0].action == "refused"
    assert "safe repository-relative path" in (result.results[0].detail or "")


def test_restore_refuses_symlink_target(temp_dir):
    """A symlink appearing at a captured path after the run refuses restore."""
    project = temp_dir / "project"
    project.mkdir()
    (temp_dir / "outside.txt").write_bytes(b"outside")
    manager = make_manager(project)
    (project / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-sym")
    manager.capture("run-sym", project / "a.txt")
    (project / "a.txt").unlink()
    os.symlink(temp_dir / "outside.txt", project / "a.txt")

    result = manager.restore("run-sym")
    assert result.applied is False
    assert result.results[0].action == "refused"
    assert "symlink" in (result.results[0].detail or "")
    assert (temp_dir / "outside.txt").read_bytes() == b"outside"


def test_restore_refuses_directory_target(temp_dir):
    """A directory replacing a captured file after the run refuses restore."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-dir")
    manager.capture("run-dir", temp_dir / "a.txt")
    (temp_dir / "a.txt").unlink()
    (temp_dir / "a.txt").mkdir()

    result = manager.restore("run-dir")
    assert result.applied is False
    assert result.results[0].action == "refused"
    assert "regular file" in (result.results[0].detail or "")


# ---------------------------------------------------------------------------
# Storage, retention, and size cap
# ---------------------------------------------------------------------------


def test_start_run_prunes_beyond_retention_window(temp_dir):
    """start_run keeps the newest retention window, oldest pruned first."""
    manager = make_manager(temp_dir, retention=3)
    for name in ("run-1", "run-2", "run-3", "run-4"):
        (temp_dir / f"{name}.txt").write_bytes(name.encode())
        manager.start_run(name)
        manager.capture(name, temp_dir / f"{name}.txt")
    remaining = {path.name for path in (temp_dir / ".ctxai" / "checkpoints").iterdir()}
    assert remaining == {"run-2", "run-3", "run-4"}, "retention pruned the oldest run at start"


def test_prune_keeps_newest_and_never_touches_unrelated_files(temp_dir):
    """prune(keep) removes only checkpoint directories beyond the window."""
    manager = make_manager(temp_dir)
    for name in ("run-a", "run-b", "run-c"):
        (temp_dir / f"{name}.txt").write_bytes(name.encode())
        manager.start_run(name)
        manager.capture(name, temp_dir / f"{name}.txt")
    (temp_dir / ".ctxai" / "checkpoints" / "notes.txt").write_text("unrelated", encoding="utf-8")

    deleted = manager.prune(keep=1)
    assert len(deleted) == 2
    remaining = {path.name for path in (temp_dir / ".ctxai" / "checkpoints").iterdir()}
    assert remaining == {"run-c", "notes.txt"}, "only checkpoint directories older than the window are pruned"


def test_size_cap_stops_capture_with_diagnostic(temp_dir):
    """Exceeding the per-run cap refuses further captures and persists the marker."""
    manager = make_manager(temp_dir, max_bytes=8)
    (temp_dir / "small.txt").write_bytes(b"12345")
    (temp_dir / "big.txt").write_bytes(b"0123456789abcdef")
    manager.start_run("run-cap")

    first = manager.capture("run-cap", temp_dir / "small.txt")
    assert first.status == "captured"
    second = manager.capture("run-cap", temp_dir / "big.txt")
    assert second.status == "capped"
    assert "size cap" in (second.reason or "")

    checkpoint = manager.load("run-cap")
    assert checkpoint.cap_exceeded is True
    assert [entry.path for entry in checkpoint.files] == ["small.txt"], "the checkpoint is honestly partial"


def test_load_rejects_unknown_schema_version(temp_dir):
    """An unknown manifest schema version is refused, not silently loaded."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"AAA")
    manager.start_run("run-schema")
    manager.capture("run-schema", temp_dir / "a.txt")
    manifest_path = manager.manifest_path("run-schema")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        manager.load("run-schema")


def test_list_delete_and_delete_all(temp_dir):
    """list_checkpoints is newest-first; delete and delete_all remove directories."""
    manager = make_manager(temp_dir)
    for name in ("run-1", "run-2", "run-3"):
        (temp_dir / f"{name}.txt").write_bytes(name.encode())
        manager.start_run(name)
        manager.capture(name, temp_dir / f"{name}.txt")

    listed = manager.list_checkpoints()
    assert [checkpoint.run_id for checkpoint in listed] == ["run-3", "run-2", "run-1"]

    manager.delete("run-2")
    assert not manager.run_dir("run-2").exists()
    with pytest.raises(FileNotFoundError):
        manager.delete("run-2")
    with pytest.raises(ValueError):
        manager.delete("../escape")

    assert manager.delete_all() == 2
    assert not any(path.is_dir() for path in (temp_dir / ".ctxai" / "checkpoints").iterdir())


# ---------------------------------------------------------------------------
# TaskRun hook and rollback transcript events
# ---------------------------------------------------------------------------


def test_taskrun_captures_before_mutation(temp_dir):
    """before_tool captures the pre-mutation bytes before the tool would run."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"original")
    run = TaskRun("add a line", project_root=temp_dir, run_id="run-hook", checkpoint_manager=manager)
    run.inspected_files.add(TaskRun.canonical(temp_dir / "a.txt"))
    call = ToolCall(id="1", name="write_file", parameters={"path": "a.txt", "content": "new content"})

    denial = run.before_tool(call, planning_enabled=True, require_approval=False, approval_callback=None)
    assert denial is None

    checkpoint = manager.load("run-hook")
    entry = checkpoint.files[0]
    assert entry.path == "a.txt" and entry.kind is CaptureKind.FILE
    blob = manager.run_dir("run-hook") / "files" / f"{entry.sha256}.blob"
    assert blob.read_bytes() == b"original"


def test_taskrun_captures_created_marker_for_new_file(temp_dir):
    """before_tool on a not-yet-existing path records the created marker."""
    manager = make_manager(temp_dir)
    run = TaskRun("add a line", project_root=temp_dir, run_id="run-hook-new", checkpoint_manager=manager)
    call = ToolCall(id="1", name="write_file", parameters={"path": "b.txt", "content": "fresh"})

    assert run.before_tool(call, planning_enabled=True, require_approval=False, approval_callback=None) is None
    checkpoint = manager.load("run-hook-new")
    assert checkpoint.files[0].kind is CaptureKind.CREATED


def test_taskrun_denied_approval_captures_nothing(temp_dir):
    """A denied approval ends the mutation before any capture happens."""
    manager = make_manager(temp_dir)
    (temp_dir / "a.txt").write_bytes(b"original")
    run = TaskRun("add a line", project_root=temp_dir, run_id="run-hook-deny", checkpoint_manager=manager)
    run.inspected_files.add(TaskRun.canonical(temp_dir / "a.txt"))
    call = ToolCall(id="1", name="write_file", parameters={"path": "a.txt", "content": "x"})

    denial = run.before_tool(call, planning_enabled=True, require_approval=True, approval_callback=lambda c: False)
    assert denial is not None
    assert not manager.manifest_path("run-hook-deny").exists()


def test_rollback_event_appends_to_transcript_with_continued_seq(temp_dir):
    """The rollback event continues the run transcript's sequence numbering."""
    recorder = RunRecorder(temp_dir, "run-event")
    recorder.record(RunEventKind.RUN_STARTED, {"goal": "demo"})
    recorder.record(RunEventKind.RUN_COMPLETED, {"status": "failed"})
    recorder.close()

    from ctxai.agent.checkpoints import FileRestore, RestoreResult
    from ctxai.commands.checkpoints_command import record_rollback_event

    appended = record_rollback_event(
        temp_dir,
        "run-event",
        RestoreResult(checkpoint_id="run-event", applied=True, results=[FileRestore("a.txt", "restored")]),
        forced=False,
    )
    assert appended is True

    events = [
        json.loads(line)
        for line in (temp_dir / ".ctxai" / "runs" / "run-event.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [event["seq"] for event in events] == [1, 2, 3], "seq continues strictly increasing"
    assert events[-1]["kind"] == "rollback"
    assert events[-1]["payload"]["status"] == "restored"
    assert events[-1]["payload"]["files"] == [{"path": "a.txt", "action": "restored", "detail": None}]


def test_rollback_event_without_transcript_is_a_no_op(temp_dir):
    """Recording a rollback with no transcript (recording disabled) is a no-op."""
    from ctxai.agent.checkpoints import RestoreResult
    from ctxai.commands.checkpoints_command import record_rollback_event

    appended = record_rollback_event(
        temp_dir, "run-missing", RestoreResult(checkpoint_id="run-missing", applied=True, results=[]), forced=False
    )
    assert appended is False


def test_capture_outcome_shape():
    """CaptureOutcome exposes status, entry, and reason."""
    outcome = CaptureOutcome("refused", reason="nope")
    assert outcome.status == "refused" and outcome.entry is None and outcome.reason == "nope"
