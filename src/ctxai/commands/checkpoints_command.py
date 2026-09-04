"""Checkpoint inspection, restore, and lifecycle operations for the CLI (HH-06).

Backs the ``ctxai checkpoints`` sub-app: listing captured run checkpoints,
restoring a project's working tree to a checkpoint's pre-run state (with
per-file stale refusals and an appended ``rollback`` transcript event), and
deleting checkpoints. All operations are scoped to the resolved
``.ctxai/checkpoints`` directory of the given project; nothing here uploads
data or touches files outside that directory (restore targets are validated
against the project root by the manager). Rendering happens in ``app.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..agent.checkpoints import Checkpoint, CheckpointManager, RestoreResult
from ..agent.run_recorder import RunEventKind, RunRecorder, is_valid_run_id, runs_dir_for

LOGGER = logging.getLogger(__name__)


def resolve_manager(project_path: Path | None = None) -> CheckpointManager:
    """Resolve the checkpoint manager for a project.

    Args:
        project_path: Project root; defaults to the current directory.

    Returns:
        A CheckpointManager scoped to the project's ``.ctxai/checkpoints``.
    """
    return CheckpointManager.for_project(project_path or Path.cwd())


def resolve_checkpoints_dir(project_path: Path | None = None) -> Path:
    """Resolve the checkpoints directory for a project.

    Args:
        project_path: Project root; defaults to the current directory.

    Returns:
        ``<project>/.ctxai/checkpoints``.
    """
    return resolve_manager(project_path).storage_dir


def list_checkpoints(project_path: Path | None = None, run_id: str | None = None) -> list[Checkpoint]:
    """List the project's checkpoints, newest first.

    Args:
        project_path: Project root; defaults to the current directory.
        run_id: Optional run id to filter by.

    Returns:
        The matching checkpoints.

    Raises:
        ValueError: When ``run_id`` is not a safe identifier.
    """
    if run_id is not None and not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id}")
    checkpoints = resolve_manager(project_path).list_checkpoints()
    if run_id is not None:
        checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.run_id == run_id]
    return checkpoints


def get_checkpoint(checkpoint_id: str, project_path: Path | None = None) -> Checkpoint:
    """Load one checkpoint.

    Args:
        checkpoint_id: The checkpoint identifier (the run id).
        project_path: Project root; defaults to the current directory.

    Returns:
        The parsed checkpoint.

    Raises:
        ValueError: When the id is unsafe or the manifest is corrupt.
        FileNotFoundError: When the checkpoint does not exist.
    """
    return resolve_manager(project_path).load(checkpoint_id)


def restore_checkpoint(checkpoint_id: str, project_path: Path | None = None, *, force: bool = False) -> RestoreResult:
    """Restore a checkpoint and append a ``rollback`` event to its transcript.

    The restore itself is planned against every captured file first: when any
    file is refused (working tree moved on) nothing is modified unless
    ``force`` is set. When the run has a transcript (``record_runs`` was on
    and the transcript was not deleted), the outcome is appended to it as a
    ``rollback`` event with the sequence numbering continued; transcript
    failures never fail the restore.

    Args:
        checkpoint_id: The checkpoint identifier (the run id).
        project_path: Project root; defaults to the current directory.
        force: Restore even when per-file staleness was detected.

    Returns:
        The restore result with per-file outcomes.

    Raises:
        ValueError: When the id is unsafe or the manifest is corrupt.
        FileNotFoundError: When the checkpoint does not exist.
    """
    manager = resolve_manager(project_path)
    result = manager.restore(checkpoint_id, force=force)
    record_rollback_event(manager.project_root, checkpoint_id, result, forced=force)
    return result


def delete_checkpoint(checkpoint_id: str, project_path: Path | None = None) -> Path:
    """Delete one checkpoint directory.

    Args:
        checkpoint_id: The checkpoint identifier (the run id).
        project_path: Project root; defaults to the current directory.

    Returns:
        The deleted directory path.

    Raises:
        ValueError: When the id is unsafe.
        FileNotFoundError: When the checkpoint does not exist.
    """
    return resolve_manager(project_path).delete(checkpoint_id)


def delete_all_checkpoints(project_path: Path | None = None) -> int:
    """Delete every checkpoint directory in the project.

    Args:
        project_path: Project root; defaults to the current directory.

    Returns:
        The number of checkpoint directories deleted.
    """
    return resolve_manager(project_path).delete_all()


def record_rollback_event(
    project_root: Path,
    checkpoint_id: str,
    result: RestoreResult,
    *,
    forced: bool,
) -> bool:
    """Append one ``rollback`` event to the run's transcript, if it exists.

    The event continues the transcript's strictly increasing sequence
    numbering. When the run has no transcript (recording was disabled or the
    transcript was deleted) this is a no-op; failures are diagnostics and
    never fail the restore.

    Args:
        project_root: The project root the run transcript lives under.
        checkpoint_id: The checkpoint identifier (equal to the run id).
        result: The restore result to record.
        forced: Whether the restore bypassed stale-detection refusals.

    Returns:
        True when an event was appended, False otherwise.
    """
    try:
        transcript = runs_dir_for(project_root) / f"{checkpoint_id}.jsonl"
        if not transcript.is_file():
            return False
        max_seq = 0
        for line in transcript.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                max_seq = max(max_seq, int(json.loads(line).get("seq") or 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        recorder = RunRecorder(project_root, checkpoint_id)
        try:
            recorder.seq = max_seq
            recorder.record(
                RunEventKind.ROLLBACK,
                {
                    "checkpoint_id": checkpoint_id,
                    "forced": forced,
                    "status": "restored" if result.applied else "refused",
                    "files": [
                        {"path": item.path, "action": item.action, "detail": item.detail} for item in result.results
                    ],
                },
            )
        finally:
            recorder.close()
        return True
    except Exception as error:  # noqa: BLE001 - the rollback event is a diagnostic
        LOGGER.warning("checkpoints (%s): could not record rollback event: %s", checkpoint_id, error)
        return False
