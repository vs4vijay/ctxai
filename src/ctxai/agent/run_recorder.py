"""Redacted local run transcripts (HH-04).

Every agent run is recorded as a JSON Lines transcript under
``.ctxai/runs/<run_id>.jsonl`` inside the project (the same repository-scoped
storage as sessions). Each line is one :class:`RunEvent` carrying
``schema_version``; line 1 is the ``run_started`` event (it also carries the
schema version — there is no separate schema header line). Payloads are
normalized to repository-relative paths and passed through
``sessions.redact_secrets`` before anything is written; nothing is uploaded.

The recorder appends one complete JSON line per event (flush after each
write) and fsyncs on close. All recorder I/O failures are surfaced through
the ``on_error`` diagnostic callback and never propagate to the run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .sessions import redact_secrets

LOGGER = logging.getLogger(__name__)

RUN_SCHEMA_VERSION = 1

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

Clock = Callable[[], datetime]
ErrorCallback = Callable[[str], None]


class RunEventKind(str, Enum):
    """The closed set of transcript event kinds (Part II contract, HH-04).

    ``rollback`` is reserved for HH-06; the recorder accepts every kind now.
    """

    RUN_STARTED = "run_started"
    USER_MESSAGE = "user_message"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL = "approval"
    STATE_TRANSITION = "state_transition"
    CHECK = "check"
    COMPACTION = "compaction"
    CANCELLATION = "cancellation"
    ROLLBACK = "rollback"
    RUN_COMPLETED = "run_completed"


RUN_EVENT_KINDS: frozenset[str] = frozenset(kind.value for kind in RunEventKind)


def new_run_id() -> str:
    """Generate a fresh run identifier.

    Returns:
        A uuid4 hex string.
    """
    return uuid.uuid4().hex


def is_valid_run_id(run_id: str) -> bool:
    """Check whether a run id is a safe single path component.

    Args:
        run_id: The candidate run identifier.

    Returns:
        True when the id is safe to join under the runs directory.
    """
    return bool(_SAFE_RUN_ID.fullmatch(run_id))


def runs_dir_for(project_root: Path) -> Path:
    """Resolve the repository-scoped runs directory for a project.

    Args:
        project_root: The project root directory.

    Returns:
        ``<project_root>/.ctxai/runs``.
    """
    return Path(project_root).resolve() / ".ctxai" / "runs"


def _prefix_variants(path: Path) -> list[str]:
    """Return directory prefixes (with trailing separator) for a path.

    Both the given and the symlink-resolved spelling are included so
    payloads built from unresolved macOS-style paths (``/var/...`` vs
    ``/private/var/...``) still normalize.

    Args:
        path: Directory to build prefixes for.

    Returns:
        Deduplicated prefix strings, longest first.
    """
    candidates = {str(path), str(Path(os.path.realpath(path)))}
    return sorted({candidate.rstrip(os.sep) + os.sep for candidate in candidates}, key=len, reverse=True)


def _normalize_paths(value: Any, project_root: Path) -> Any:
    """Rewrite absolute project and home paths in strings to safe relatives.

    Project paths become repository-relative (``<root>/src/x.py`` ->
    ``src/x.py``) and the user's home directory becomes ``~``. Applied
    recursively over dicts, lists, and strings.

    Args:
        value: The payload value to normalize.
        project_root: The project root (resolved spellings are derived).

    Returns:
        The normalized value.
    """
    root_prefixes = _prefix_variants(project_root)
    home_prefixes = _prefix_variants(Path.home())
    exact_roots = {prefix.rstrip(os.sep) for prefix in root_prefixes}

    def _rewrite(text: str) -> str:
        if text in exact_roots:
            return "."
        normalized = text
        # Project-relative first, so paths inside the project (even when the
        # project itself lives under the home directory) become repo-relative;
        # remaining home paths fall back to ~/.
        for prefix in root_prefixes:
            if prefix in normalized:
                normalized = normalized.replace(prefix, "")
        for prefix in home_prefixes:
            if prefix in normalized:
                normalized = normalized.replace(prefix, "~/")
        # Any other user's home directory is equally sensitive (RE-02 traces
        # may carry content from collaborators' machines): rewrite the generic
        # POSIX home shapes to ~.
        normalized = re.sub(r"/(?:Users|home)/[A-Za-z0-9._-]+", "~", normalized)
        return normalized

    if isinstance(value, dict):
        return {key: _normalize_paths(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_paths(item, project_root) for item in value]
    if isinstance(value, str):
        return _rewrite(value)
    return value


def redact_payload(payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Prepare a payload for persistence: normalize paths, then redact secrets.

    Args:
        payload: The raw event payload.
        project_root: The resolved project root for path normalization.

    Returns:
        The redacted, path-normalized payload safe to persist.
    """
    normalized = _normalize_paths(payload, project_root)
    return redact_secrets(normalized)


@dataclass
class RunEvent:
    """One persisted transcript event (JSON Lines line).

    Attributes:
        schema_version: On-disk schema version (always ``RUN_SCHEMA_VERSION``).
        run_id: Identifier shared by every event of one run.
        seq: Monotonic per-run sequence number starting at 1.
        timestamp: ISO-8601 UTC timestamp from the injected clock.
        kind: One of the :class:`RunEventKind` values.
        payload: Redacted, path-normalized event payload.
        usage: Optional ``UsageRecord``-shaped dict for ``llm_call`` events.
    """

    schema_version: int
    run_id: str
    seq: int
    timestamp: str
    kind: str
    payload: dict[str, Any]
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the on-disk dictionary shape.

        Returns:
            Dictionary representation matching the persisted JSON line, with
            ``kind`` as its plain string value.
        """
        kind = self.kind.value if isinstance(self.kind, RunEventKind) else str(self.kind)
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "kind": kind,
            "payload": self.payload,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunEvent:
        """Create from an on-disk dictionary.

        Unknown kinds (a future schema writing kinds this version does not
        know) are preserved as plain strings instead of failing the load.

        Args:
            data: Dictionary produced by ``to_dict`` or a compatible version.

        Returns:
            The reconstructed :class:`RunEvent`.

        Raises:
            KeyError: When a required field is missing.
        """
        return cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            seq=int(data["seq"]),
            timestamp=str(data["timestamp"]),
            kind=str(data["kind"]),
            payload=dict(data.get("payload") or {}),
            usage=data.get("usage"),
        )


class NullRunRecorder:
    """No-op recorder used when recording is disabled or unavailable."""

    def __init__(self, run_id: str = "") -> None:
        """Initialize the no-op recorder.

        Args:
            run_id: The run id the recorder would have used.
        """
        self.run_id = run_id
        self.seq = 0
        self.error_count = 0
        self.path: Path | None = None

    def record(
        self,
        kind: RunEventKind | str,
        payload: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> RunEvent | None:
        """Drop the event.

        Args:
            kind: Event kind (ignored).
            payload: Event payload (ignored).
            usage: Optional usage payload (ignored).

        Returns:
            Always ``None``.
        """
        return None

    def close(self) -> None:
        """No-op close."""
        return None


class RunRecorder:
    """Appends redacted ``RunEvent`` lines to ``.ctxai/runs/<run_id>.jsonl``.

    Writes are atomic per line (one ``write`` call per complete JSON line,
    flushed immediately) and fsynced on :meth:`close`. Every write failure is
    reported through ``on_error`` (and counted in :attr:`error_count`) and
    never raises; a failed event consumes its sequence number but recording
    continues.
    """

    def __init__(
        self,
        project_root: Path,
        run_id: str,
        *,
        clock: Clock | None = None,
        storage_dir: Path | None = None,
        on_error: ErrorCallback | None = None,
    ):
        """Open (or create) the transcript file for one run.

        Args:
            project_root: Project root; the runs directory lives inside it.
            run_id: Stable run identifier shared by all events.
            clock: Injected clock for deterministic timestamps (defaults to
                UTC ``datetime.now``).
            storage_dir: Optional explicit storage directory override (tests).
            on_error: Optional callback receiving one diagnostic string per
                recorder failure.

        Raises:
            ValueError: When ``run_id`` is not a safe single path component.
            OSError: When the runs directory or transcript file cannot be
                created or opened (callers should fall back to a no-op).
        """
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("Run id must use only letters, numbers, '.', '_' or '-'")
        # Keep the caller's spelling of the root: payload paths may use the
        # unresolved (pre-symlink) form, so path normalization derives
        # prefixes from both spellings.
        self._raw_root = Path(project_root)
        self.project_root = self._raw_root.resolve()
        self.run_id = run_id
        self.clock: Clock = clock or (lambda: datetime.now(timezone.utc))
        self.on_error = on_error
        self.storage_dir = storage_dir or runs_dir_for(self.project_root)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path: Path | None = self.storage_dir / f"{run_id}.jsonl"
        self.seq = 0
        self.error_count = 0
        # Append mode: an interrupted run leaves a partial transcript that a
        # retried run with the same id would extend, never truncate.
        self._handle = open(self.path, "a", encoding="utf-8")

    def record(
        self,
        kind: RunEventKind | str,
        payload: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> RunEvent | None:
        """Append one redacted event to the transcript.

        Failures are isolated: this method never raises. A failed write is
        surfaced through ``on_error`` and the next event still records.

        Args:
            kind: Event kind (enum value or plain string).
            payload: Event payload (redacted and path-normalized on write).
            usage: Optional ``UsageRecord`` dict carried on the event.

        Returns:
            The recorded :class:`RunEvent`, or ``None`` when the write failed.
        """
        kind_value = kind.value if isinstance(kind, RunEventKind) else str(kind)
        self.seq += 1
        try:
            # One write per complete line keeps concurrent readers seeing
            # whole events; flush makes the line durable before the run
            # proceeds (fsync happens once on close). Redaction and
            # serialization happen here too: payloads that cannot be
            # prepared surface as diagnostics, never as run failures.
            event = RunEvent(
                schema_version=RUN_SCHEMA_VERSION,
                run_id=self.run_id,
                seq=self.seq,
                timestamp=self.clock().isoformat(),
                kind=kind_value,
                payload=redact_payload(dict(payload or {}), self._raw_root),
                usage=usage,
            )
            line = json.dumps(event.to_dict(), sort_keys=True)
            self._handle.write(line + "\n")
            self._handle.flush()
        except Exception as error:  # noqa: BLE001 - recorder failures are never fatal
            self.error_count += 1
            self._surface(f"failed to write run event seq={self.seq}: {error}")
            return None
        return event

    def close(self) -> None:
        """Flush, fsync, and close the transcript file. Never raises."""
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except Exception as error:  # noqa: BLE001 - close failures are never fatal
            self.error_count += 1
            self._surface(f"failed to fsync run transcript: {error}")
        finally:
            try:
                self._handle.close()
            except Exception as error:  # noqa: BLE001
                self._surface(f"failed to close run transcript: {error}")

    def _surface(self, message: str) -> None:
        """Report one recorder failure as a diagnostic.

        Args:
            message: The diagnostic text.
        """
        LOGGER.warning("run transcript (%s): %s", self.run_id, message)
        if self.on_error is not None:
            self.on_error(message)


def create_recorder(
    project_root: Path,
    run_id: str,
    *,
    enabled: bool,
    clock: Clock | None = None,
    on_error: ErrorCallback | None = None,
) -> RunRecorder | NullRunRecorder:
    """Create the run recorder, degrading to a no-op when disabled or failing.

    This is the single factory the agent loop uses; a recorder that cannot be
    opened (permissions, missing project directory) becomes a no-op plus a
    diagnostic — recording problems never fail a run.

    Args:
        project_root: Project root for the runs directory.
        run_id: Stable run identifier for this run.
        enabled: Whether recording is enabled (``AgentBehaviorConfig.record_runs``).
        clock: Injected clock for deterministic timestamps.
        on_error: Optional callback receiving one diagnostic string per failure.

    Returns:
        A :class:`RunRecorder` when recording is possible, else a
        :class:`NullRunRecorder`.
    """
    if not enabled:
        return NullRunRecorder(run_id=run_id)
    try:
        return RunRecorder(project_root, run_id, clock=clock, on_error=on_error)
    except Exception as error:  # noqa: BLE001 - construction failures are never fatal
        LOGGER.warning("run transcript (%s): recorder unavailable: %s", run_id, error)
        if on_error is not None:
            on_error(f"recorder unavailable: {error}")
        return NullRunRecorder(run_id=run_id)


def prune_runs(storage_dir: Path, keep: int) -> list[Path]:
    """Delete the oldest transcripts beyond a retention limit, oldest first.

    Scoped to ``*.jsonl`` files directly inside ``storage_dir``; nested
    directories and unrelated files are never touched. A missing directory
    prunes nothing.

    Args:
        storage_dir: The resolved runs directory.
        keep: Number of newest transcripts to retain.

    Returns:
        The list of deleted paths (oldest first).
    """
    directory = Path(storage_dir)
    if not directory.is_dir() or keep < 0:
        return []
    transcripts = [path for path in directory.glob("*.jsonl") if path.is_file()]
    transcripts.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    excess = len(transcripts) - keep
    deleted: list[Path] = []
    for path in transcripts[: max(0, excess)]:
        try:
            path.unlink()
            deleted.append(path)
        except OSError as error:
            LOGGER.warning("run transcript retention: could not delete %s: %s", path, error)
    return deleted
