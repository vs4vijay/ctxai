"""Local run checkpoints and one-command rollback (HH-06).

Before the first mutation of each file in a run, the agent loop (hooked at the
``TaskRun.before_tool`` mutation boundary) stores the file's pre-mutation
bytes — or a ``created`` marker when the file does not exist yet — under

``<project>/.ctxai/checkpoints/<run_id>/``

Layout::

    .ctxai/checkpoints/<run_id>/
        manifest.json          schema-versioned, atomic write + fsync
        files/<sha256>.blob    content-addressed pre-mutation bytes

There is exactly one checkpoint per run: ``checkpoint_id == run_id``. The
manifest is atomically rewritten every time a file is added or the checkpoint
is finalized; per-file content lives in sibling blobs keyed by the content
hash, so manifest paths can never be used for path traversal (blobs are
looked up by hash only).

``restore`` writes pre-run bytes back, deletes files the run created, and
recreates files the run captured and later lost (e.g. deleted through a bash
command). It refuses — per file, before touching anything — when the working
tree moved on since the run (the current hash differs from the post-run hash
recorded at finalization) unless ``force`` is set.

Scope is honest and documented: only the structured file tools
(``write_file``/``edit_file``) are captured, because only they name a target
path before mutating. Files deleted through bash are restorable only when a
structured tool touched them earlier in the same run. Checkpoint data is
local-only; git is used solely to record HEAD for context — ctxai never
rewrites history or creates commits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - only used for a fixed-argument `git rev-parse` below
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .run_recorder import is_valid_run_id
from .sessions import redact_secrets

LOGGER = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_RETENTION = 20
DEFAULT_CHECKPOINT_MAX_BYTES = 50 * 1024 * 1024

Clock = Callable[[], datetime]


class CaptureKind(str, Enum):
    """What the checkpoint holds for one file.

    ``file`` means the pre-mutation bytes were captured (the file existed at
    first touch); ``created`` means the run created the file, so the pre-run
    state is "absent" and restore deletes it.
    """

    FILE = "file"
    CREATED = "created"


@dataclass
class CheckpointFile:
    """One captured file inside a run checkpoint.

    Attributes:
        path: Repository-relative POSIX path of the captured file.
        kind: ``file`` (pre-mutation bytes stored) or ``created`` (no pre-state).
        sha256: Hash of the pre-mutation bytes (``None`` for ``created``).
        blob: Blob name relative to the run directory (``None`` for ``created``).
        size: Size of the pre-mutation content in bytes.
        post_run_sha256: Hash of the file at run end, recorded at finalization.
        post_run_present: Whether the file existed at run end; ``None`` when the
            checkpoint was never finalized (stale check is then skipped).
    """

    path: str
    kind: CaptureKind = CaptureKind.FILE
    sha256: str | None = None
    blob: str | None = None
    size: int = 0
    post_run_sha256: str | None = None
    post_run_present: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the persisted dictionary shape.

        Returns:
            Dictionary representation matching the manifest JSON.
        """
        return {
            "path": self.path,
            "kind": self.kind.value,
            "sha256": self.sha256,
            "blob": self.blob,
            "size": self.size,
            "post_run_sha256": self.post_run_sha256,
            "post_run_present": self.post_run_present,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointFile:
        """Create from a persisted dictionary.

        Args:
            data: Dictionary produced by ``to_dict``.

        Returns:
            The reconstructed ``CheckpointFile``.

        Raises:
            ValueError: When the capture kind is unknown.
        """
        try:
            kind = CaptureKind(str(data.get("kind") or CaptureKind.FILE.value))
        except ValueError as error:
            raise ValueError(f"Unsupported capture kind: {data.get('kind')!r}") from error
        return cls(
            path=str(data["path"]),
            kind=kind,
            sha256=data.get("sha256"),
            blob=data.get("blob"),
            size=int(data.get("size") or 0),
            post_run_sha256=data.get("post_run_sha256"),
            post_run_present=data.get("post_run_present"),
        )


@dataclass
class Checkpoint:
    """The persisted per-run checkpoint (Part II contract, HH-06).

    Attributes:
        checkpoint_id: Identifier of the checkpoint (equal to ``run_id``).
        run_id: The agent run the checkpoint belongs to.
        created_at: ISO-8601 UTC timestamp of checkpoint creation.
        retained: Whether the run succeeded (kept for audit).
        files: Captured files with repository-relative paths.
        git_head: Repository HEAD at run start, recorded for context only.
        status: ``open`` while the run is in flight, ``finalized`` afterwards.
        updated_at: ISO-8601 UTC timestamp of the last manifest rewrite.
        bytes_captured: Sum of captured pre-mutation sizes.
        cap_exceeded: Whether the per-run size cap stopped further captures.
        project_root: Resolved project root the checkpoint belongs to.
    """

    checkpoint_id: str
    run_id: str
    created_at: str
    retained: bool = False
    files: list[CheckpointFile] = field(default_factory=list)
    git_head: str | None = None
    status: str = "open"
    updated_at: str | None = None
    bytes_captured: int = 0
    cap_exceeded: bool = False
    project_root: str | None = None

    @property
    def paths(self) -> list[str]:
        """List the captured repository-relative paths.

        Returns:
            The captured paths in capture order.
        """
        return [entry.path for entry in self.files]

    def to_dict(self) -> dict[str, Any]:
        """Convert to the persisted manifest shape (schema-versioned).

        Returns:
            Dictionary representation matching ``manifest.json``.
        """
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "retained": self.retained,
            "status": self.status,
            "git_head": self.git_head,
            "project_root": self.project_root,
            "bytes_captured": self.bytes_captured,
            "cap_exceeded": self.cap_exceeded,
            "files": [entry.to_dict() for entry in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create from a persisted manifest dictionary.

        Unknown keys (for example ``schema_version``, validated by the loader)
        are ignored so forward-compatible fields keep loading.

        Args:
            data: Dictionary produced by ``to_dict``.

        Returns:
            The reconstructed ``Checkpoint``.

        Raises:
            ValueError: When a file entry carries an unknown capture kind.
        """
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            run_id=str(data["run_id"]),
            created_at=str(data.get("created_at") or ""),
            retained=bool(data.get("retained", False)),
            files=[CheckpointFile.from_dict(item) for item in data.get("files") or []],
            git_head=data.get("git_head"),
            status=str(data.get("status") or "open"),
            updated_at=data.get("updated_at"),
            bytes_captured=int(data.get("bytes_captured") or 0),
            cap_exceeded=bool(data.get("cap_exceeded", False)),
            project_root=data.get("project_root"),
        )


@dataclass(frozen=True)
class CaptureOutcome:
    """Result of one capture attempt.

    Attributes:
        status: ``captured``, ``already`` (first-touch hit), ``refused``
            (safety), ``capped`` (size cap), or ``failed`` (I/O error).
        entry: The stored ``CheckpointFile`` when something was captured.
        reason: Human-readable diagnostic for every non-captured status.
    """

    status: str
    entry: CheckpointFile | None = None
    reason: str | None = None


@dataclass
class FileRestore:
    """Per-file outcome of a restore plan or application.

    Attributes:
        path: Repository-relative path of the file.
        action: ``restored``, ``recreated``, ``deleted``, ``skipped``, or
            ``refused``.
        detail: Human-readable explanation (required for ``refused``).
    """

    path: str
    action: str
    detail: str | None = None


@dataclass
class RestoreResult:
    """Outcome of restoring one checkpoint.

    Attributes:
        checkpoint_id: The checkpoint that was (or was not) restored.
        applied: Whether any file was touched; ``False`` means the restore was
            refused before anything was modified.
        results: Per-file outcomes in manifest order.
    """

    checkpoint_id: str
    applied: bool
    results: list[FileRestore] = field(default_factory=list)

    @property
    def refused(self) -> list[FileRestore]:
        """List the per-file refusals.

        Returns:
            The results whose action is ``refused``.
        """
        return [item for item in self.results if item.action == "refused"]

    @property
    def ok(self) -> bool:
        """Whether the restore applied cleanly with no refusals.

        Returns:
            True when applied and no per-file refusal remains.
        """
        return self.applied and not self.refused


def _safe_relative_path(relative: str) -> Path | None:
    """Validate a stored repository-relative path.

    Args:
        relative: The candidate relative path (POSIX separators).

    Returns:
        The path as a ``Path``, or ``None`` when it is absolute, escapes via
        ``..``, carries backslashes, or uses a Windows drive prefix.
    """
    if not relative or relative.startswith("/") or "\\" in relative:
        return None
    candidate = PurePosixPath(relative)
    if candidate.is_absolute():
        return None
    if any(part in ("", ".", "..") for part in candidate.parts):
        return None
    if PureWindowsPath(relative).is_absolute() or (len(relative) > 1 and relative[1] == ":"):
        return None
    return Path(*candidate.parts)


def _sha256_bytes(data: bytes) -> str:
    """Hash content the way checkpoint entries reference it.

    Args:
        data: The bytes to hash.

    Returns:
        The hex sha256 digest.
    """
    return hashlib.sha256(data).hexdigest()


class CheckpointManager:
    """Stores, restores, lists, and prunes per-run checkpoints for a project.

    The manager is repository-scoped: every artifact lives under
    ``<project>/.ctxai/checkpoints/``. Capture failures and refusals are
    diagnostics — checkpointing never blocks or fails a run. The clock is
    injected for deterministic timestamps (tests).
    """

    def __init__(
        self,
        project_root: Path,
        *,
        clock: Clock | None = None,
        storage_dir: Path | None = None,
        retention: int = DEFAULT_CHECKPOINT_RETENTION,
        max_bytes: int = DEFAULT_CHECKPOINT_MAX_BYTES,
    ) -> None:
        """Initialize the manager for one project.

        Args:
            project_root: The project root; storage lives inside it.
            clock: Injected clock for deterministic timestamps (defaults to
                UTC ``datetime.now``).
            storage_dir: Optional explicit storage directory override (tests).
            retention: Maximum checkpoint directories kept per project; the
                oldest are pruned at run start.
            max_bytes: Per-run capture cap; beyond it captures stop (the run
                proceeds with a partial checkpoint).

        Raises:
            ValueError: When ``retention`` or ``max_bytes`` is not positive.
        """
        if retention < 1:
            raise ValueError("retention must be at least 1")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.project_root = Path(project_root).resolve()
        self.clock: Clock = clock or (lambda: datetime.now(timezone.utc))
        self.storage_dir = storage_dir or self.project_root / ".ctxai" / "checkpoints"
        self.retention = retention
        self.max_bytes = max_bytes
        self._open: dict[str, Checkpoint] = {}

    @classmethod
    def for_project(
        cls,
        project_root: Path,
        *,
        clock: Clock | None = None,
        storage_dir: Path | None = None,
        retention: int = DEFAULT_CHECKPOINT_RETENTION,
        max_bytes: int = DEFAULT_CHECKPOINT_MAX_BYTES,
    ) -> CheckpointManager:
        """Create a manager rooted at a project directory.

        Args:
            project_root: The project root (resolved internally).
            clock: Injected clock for deterministic timestamps.
            storage_dir: Optional explicit storage directory override (tests).
            retention: Maximum checkpoints kept per project.
            max_bytes: Per-run capture cap in bytes.

        Returns:
            The configured CheckpointManager.
        """
        return cls(
            Path(project_root),
            clock=clock,
            storage_dir=storage_dir,
            retention=retention,
            max_bytes=max_bytes,
        )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        """Resolve the checkpoint directory of one run.

        Args:
            run_id: The run identifier.

        Returns:
            ``<storage_dir>/<run_id>``.
        """
        return self.storage_dir / run_id

    def manifest_path(self, run_id: str) -> Path:
        """Resolve the manifest path of one run.

        Args:
            run_id: The run identifier.

        Returns:
            ``<storage_dir>/<run_id>/manifest.json``.
        """
        return self.run_dir(run_id) / "manifest.json"

    def _now(self) -> str:
        """Return the current injected time as an ISO-8601 string.

        Returns:
            The timestamp string.
        """
        return self.clock().isoformat()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_id: str) -> Checkpoint:
        """Register a fresh checkpoint for a run and apply retention.

        Retention pruning runs first (oldest checkpoints beyond
        ``retention - 1`` are removed so the fresh run fits the window). The
        manifest is persisted lazily on the first capture, so runs that never
        mutate leave no directory behind.

        Args:
            run_id: The run identifier (``checkpoint_id``).

        Returns:
            The open in-memory checkpoint for the run.

        Raises:
            ValueError: When ``run_id`` is not a safe path component.
        """
        if not is_valid_run_id(run_id):
            raise ValueError("Run id must use only letters, numbers, '.', '_' or '-'")
        self._prune_retention()
        self._open[run_id] = Checkpoint(
            checkpoint_id=run_id,
            run_id=run_id,
            created_at=self._now(),
            git_head=self._git_head(),
            project_root=str(self.project_root),
        )
        return self._open[run_id]

    def finalize(self, run_id: str, *, retained: bool) -> Checkpoint | None:
        """Record post-run file state and close the run's checkpoint.

        For every captured file the current presence and content hash are
        stored (``post_run_present``/``post_run_sha256``) — restore compares
        against them to refuse a moved-on working tree. Checkpoints without
        captured files are discarded silently.

        Args:
            run_id: The run identifier.
            retained: Whether the run succeeded (kept for audit).

        Returns:
            The finalized checkpoint, or ``None`` when the run captured
            nothing (or was never started on this manager).
        """
        checkpoint = self._open.pop(run_id, None)
        if checkpoint is None:
            return None
        if not checkpoint.files:
            return None
        for entry in checkpoint.files:
            safe = _safe_relative_path(entry.path)
            if safe is None:
                continue
            target = self.project_root / safe
            if target.is_file() and not target.is_symlink():
                entry.post_run_present = True
                entry.post_run_sha256 = _sha256_bytes(target.read_bytes())
            else:
                entry.post_run_present = False
                entry.post_run_sha256 = None
        checkpoint.retained = retained
        checkpoint.status = "finalized"
        checkpoint.updated_at = self._now()
        self._persist(checkpoint)
        return checkpoint

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, run_id: str, path: str | Path) -> CaptureOutcome:
        """Capture the pre-mutation state of one file (first touch per run).

        Refusals (symlinks, paths escaping the project root, non-regular
        files) and the size cap are returned as outcomes — never raised — so
        checkpointing cannot break the run.

        Args:
            run_id: The run identifier the capture belongs to.
            path: The path the run is about to mutate (absolute or relative
                to the project root; symlinks are refused, not followed).

        Returns:
            The capture outcome.
        """
        if not is_valid_run_id(run_id):
            return CaptureOutcome("refused", reason=f"invalid run id: {run_id}")
        try:
            return self._capture(run_id, Path(path))
        except Exception as error:  # noqa: BLE001 - capture must never fail the run
            LOGGER.warning("checkpoints (%s): capture failed for %s: %s", run_id, path, error)
            return CaptureOutcome("failed", reason=str(error))

    def _capture(self, run_id: str, original: Path) -> CaptureOutcome:
        """Perform the capture after validation.

        Args:
            run_id: The run identifier.
            original: The caller-supplied path.

        Returns:
            The capture outcome.
        """
        checkpoint = self._ensure_open(run_id)
        if original.is_symlink():
            return CaptureOutcome("refused", reason=f"refusing symlinked path: {original}")
        resolved = Path(os.path.realpath(original))
        try:
            relative = resolved.relative_to(self.project_root)
        except ValueError:
            return CaptureOutcome("refused", reason=f"path escapes the project root: {original}")
        relative_text = relative.as_posix()
        for entry in checkpoint.files:
            if entry.path == relative_text:
                return CaptureOutcome("already", entry=entry)
        if resolved.is_symlink():
            return CaptureOutcome("refused", reason=f"refusing symlinked path: {original}")
        if resolved.exists() and not resolved.is_file():
            return CaptureOutcome("refused", reason=f"not a regular file: {original}")
        if resolved.is_file():
            data = resolved.read_bytes()
            size = len(data)
            if checkpoint.bytes_captured + size > self.max_bytes:
                checkpoint.cap_exceeded = True
                checkpoint.updated_at = self._now()
                self._persist(checkpoint)
                message = (
                    f"checkpoint size cap of {self.max_bytes} bytes exceeded; "
                    f"{relative_text} was not captured (the checkpoint is partial)"
                )
                LOGGER.warning("checkpoints (%s): %s", run_id, message)
                return CaptureOutcome("capped", reason=message)
            digest = _sha256_bytes(data)
            self._write_blob(run_id, digest, data)
            entry = CheckpointFile(
                path=relative_text,
                kind=CaptureKind.FILE,
                sha256=digest,
                blob=f"files/{digest}.blob",
                size=size,
            )
            checkpoint.bytes_captured += size
        else:
            entry = CheckpointFile(path=relative_text, kind=CaptureKind.CREATED)
        checkpoint.files.append(entry)
        checkpoint.updated_at = self._now()
        self._persist(checkpoint)
        return CaptureOutcome("captured", entry=entry)

    def _ensure_open(self, run_id: str) -> Checkpoint:
        """Return the open checkpoint for a run, creating or loading as needed.

        Args:
            run_id: The run identifier.

        Returns:
            The open checkpoint (memory first, then disk, then fresh).
        """
        checkpoint = self._open.get(run_id)
        if checkpoint is not None:
            return checkpoint
        if self.manifest_path(run_id).is_file():
            try:
                loaded = self.load(run_id)
            except (OSError, ValueError) as error:
                LOGGER.warning("checkpoints (%s): could not reload manifest: %s", run_id, error)
            else:
                self._open[run_id] = loaded
                return loaded
        return self.start_run(run_id)

    def _write_blob(self, run_id: str, digest: str, data: bytes) -> None:
        """Store pre-mutation bytes under their content hash (atomic, fsync).

        Args:
            run_id: The run identifier.
            digest: The content hash naming the blob.
            data: The pre-mutation bytes.
        """
        blob_dir = self.run_dir(run_id) / "files"
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path = blob_dir / f"{digest}.blob"
        if blob_path.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=".blob-", suffix=".tmp", dir=blob_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, blob_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _persist(self, checkpoint: Checkpoint) -> None:
        """Atomically rewrite the manifest (temp file, fsync, rename).

        Args:
            checkpoint: The checkpoint to persist.
        """
        path = self.manifest_path(checkpoint.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = redact_secrets(checkpoint.to_dict())
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load one checkpoint's manifest.

        Args:
            checkpoint_id: The checkpoint identifier (the run id).

        Returns:
            The parsed checkpoint.

        Raises:
            ValueError: When the id is unsafe, the manifest is corrupt, or the
                schema version is unsupported.
            FileNotFoundError: When no manifest exists for the id.
        """
        if not is_valid_run_id(checkpoint_id):
            raise ValueError(f"Invalid checkpoint id: {checkpoint_id}")
        path = self.manifest_path(checkpoint_id)
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint for '{checkpoint_id}' under {self.storage_dir}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Corrupt checkpoint manifest for '{checkpoint_id}': {error}") from error
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema version")
        return Checkpoint.from_dict(payload)

    def list_checkpoints(self) -> list[Checkpoint]:
        """List the project's checkpoints, newest first.

        Args:
            None.

        Returns:
            Checkpoints ordered by ``created_at`` (newest first). Directories
            without a parsable manifest are skipped.
        """
        if not self.storage_dir.is_dir():
            return []
        found: list[Checkpoint] = []
        for path in sorted(self.storage_dir.iterdir()):
            if not path.is_dir():
                continue
            try:
                found.append(self.load(path.name))
            except (OSError, ValueError):
                continue
        found.sort(key=lambda item: item.created_at, reverse=True)
        return found

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, checkpoint_id: str, *, force: bool = False) -> RestoreResult:
        """Restore the working tree to a checkpoint's pre-run state.

        The restore is planned for every captured file first: when any file
        is refused (working tree moved on, unsafe path, symlink) nothing is
        modified unless ``force`` is set. Applied actions: pre-run bytes are
        written back (``restored``/``recreated``) and files the run created
        are deleted (``deleted``).

        Args:
            checkpoint_id: The checkpoint identifier (the run id).
            force: Apply the restore even when per-file staleness was detected.

        Returns:
            The restore result with per-file outcomes.

        Raises:
            ValueError: When the id is unsafe or the manifest is corrupt.
            FileNotFoundError: When the checkpoint does not exist.
        """
        checkpoint = self.load(checkpoint_id)
        planned: list[tuple[CheckpointFile, FileRestore]] = []
        refused = False
        for entry in checkpoint.files:
            action = self._plan_restore(checkpoint, entry, force=force)
            planned.append((entry, action))
            if action.action == "refused":
                refused = True
        results = [action for _, action in planned]
        if refused:
            return RestoreResult(checkpoint_id=checkpoint_id, applied=False, results=results)
        for entry, action in planned:
            self._apply_restore(checkpoint, entry, action)
        return RestoreResult(checkpoint_id=checkpoint_id, applied=True, results=results)

    def _plan_restore(self, checkpoint: Checkpoint, entry: CheckpointFile, *, force: bool) -> FileRestore:
        """Compute the per-file restore action without touching the tree.

        Hard safety refusals (unsafe paths, symlink targets, missing content)
        apply regardless of ``force``; staleness refusals (the working tree
        moved on since the run) are bypassed when ``force`` is set.

        Args:
            checkpoint: The checkpoint being restored.
            entry: The captured file entry.
            force: Skip the post-run staleness checks.

        Returns:
            The planned action for the file.
        """
        safe = _safe_relative_path(entry.path)
        if safe is None:
            return FileRestore(entry.path, "refused", "path is not a safe repository-relative path")
        target = self.project_root / safe
        if target.is_symlink():
            return FileRestore(entry.path, "refused", "target path is a symlink")
        if target.exists() and not target.is_file():
            return FileRestore(entry.path, "refused", "target is not a regular file")
        current_sha: str | None = None
        if target.is_file():
            current_sha = _sha256_bytes(target.read_bytes())
        if not force and entry.post_run_present is not None:
            if (current_sha is None) != (not entry.post_run_present):
                detail = "deleted after the run" if current_sha is None else "created after the run"
                return FileRestore(entry.path, "refused", f"working tree moved on since the run ({detail})")
            if current_sha is not None and entry.post_run_sha256 is not None and current_sha != entry.post_run_sha256:
                return FileRestore(entry.path, "refused", "file changed since the run (hash mismatch)")
        if entry.kind is CaptureKind.CREATED:
            if current_sha is None:
                return FileRestore(entry.path, "skipped", "already absent")
            return FileRestore(entry.path, "deleted", "remove the file the run created")
        if not entry.sha256:
            return FileRestore(entry.path, "refused", "checkpoint entry has no content hash")
        if not self._blob_path(checkpoint.run_id, entry.sha256).is_file():
            return FileRestore(entry.path, "refused", "checkpoint content blob is missing")
        if current_sha is None:
            return FileRestore(entry.path, "recreated", "recreate the file the run deleted")
        return FileRestore(entry.path, "restored", "write back pre-run bytes")

    def _apply_restore(self, checkpoint: Checkpoint, entry: CheckpointFile, action: FileRestore) -> None:
        """Apply one planned restore action.

        Args:
            checkpoint: The checkpoint being restored.
            entry: The captured file entry.
            action: The planned action (refused/skipped entries are ignored).
        """
        if action.action in ("refused", "skipped"):
            return
        safe = _safe_relative_path(entry.path)
        if safe is None:
            return
        target = self.project_root / safe
        if action.action == "deleted":
            target.unlink()
            return
        if not entry.sha256:
            return
        blob = self._blob_path(checkpoint.run_id, entry.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob.read_bytes())

    def _blob_path(self, run_id: str, digest: str) -> Path:
        """Resolve the blob path for a content hash.

        Args:
            run_id: The run identifier.
            digest: The content hash.

        Returns:
            ``<run_dir>/files/<digest>.blob`` (never derived from manifest
            paths, so a tampered manifest cannot point elsewhere).
        """
        return self.run_dir(run_id) / "files" / f"{digest}.blob"

    # ------------------------------------------------------------------
    # Deletion and retention
    # ------------------------------------------------------------------

    def delete(self, checkpoint_id: str) -> Path:
        """Delete one checkpoint directory.

        Args:
            checkpoint_id: The checkpoint identifier (the run id).

        Returns:
            The deleted directory path.

        Raises:
            ValueError: When the id is not a safe path component.
            FileNotFoundError: When no checkpoint directory exists for the id.
        """
        if not is_valid_run_id(checkpoint_id):
            raise ValueError(f"Invalid checkpoint id: {checkpoint_id}")
        path = self.run_dir(checkpoint_id)
        if not path.is_dir():
            raise FileNotFoundError(f"No checkpoint for '{checkpoint_id}' under {self.storage_dir}")
        shutil.rmtree(path)
        self._open.pop(checkpoint_id, None)
        return path

    def delete_all(self) -> int:
        """Delete every checkpoint directory in the project.

        Only directories containing a ``manifest.json`` are removed; unrelated
        files and unrecognized directories are never touched.

        Returns:
            The number of checkpoint directories deleted.
        """
        if not self.storage_dir.is_dir():
            return 0
        deleted = 0
        for path in sorted(self.storage_dir.iterdir()):
            if path.is_dir() and self.manifest_path(path.name).is_file():
                shutil.rmtree(path)
                self._open.pop(path.name, None)
                deleted += 1
        return deleted

    def prune(self, keep: int) -> list[Path]:
        """Delete the oldest checkpoints beyond a retention limit.

        Scoped to directories with a ``manifest.json`` inside the checkpoints
        directory; unrelated paths are never touched. Ordering uses the
        manifest ``created_at`` (falling back to the directory mtime).

        Args:
            keep: Number of newest checkpoints to retain.

        Returns:
            The deleted directory paths (oldest first).
        """
        if not self.storage_dir.is_dir() or keep < 0:
            return []
        candidates = [
            path for path in self.storage_dir.iterdir() if path.is_dir() and self.manifest_path(path.name).is_file()
        ]
        candidates.sort(key=self._checkpoint_order_key)
        excess = len(candidates) - keep
        deleted: list[Path] = []
        for path in candidates[: max(0, excess)]:
            try:
                shutil.rmtree(path)
                self._open.pop(path.name, None)
                deleted.append(path)
            except OSError as error:
                LOGGER.warning("checkpoint retention: could not delete %s: %s", path, error)
        return deleted

    def _prune_retention(self) -> None:
        """Apply the configured retention window (diagnostics only)."""
        try:
            self.prune(keep=self.retention - 1)
        except Exception as error:  # noqa: BLE001 - retention is a diagnostic, never fatal
            LOGGER.warning("checkpoint retention failed: %s", error)

    def _checkpoint_order_key(self, path: Path) -> tuple[str, str]:
        """Order a checkpoint directory oldest-first.

        Args:
            path: The checkpoint directory.

        Returns:
            A ``(created_at, name)`` sort key; an unparsable manifest falls
            back to the directory modification time.
        """
        created_at = ""
        try:
            created_at = self.load(path.name).created_at
        except (OSError, ValueError):
            try:
                created_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                created_at = ""
        return (created_at, path.name)

    def _git_head(self) -> str | None:
        """Record the repository HEAD for context (never required).

        Returns:
            The HEAD commit hash, or ``None`` outside a git repository or when
            git is unavailable.
        """
        git = shutil.which("git")
        if git is None:
            return None
        try:
            completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
                [git, "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None
