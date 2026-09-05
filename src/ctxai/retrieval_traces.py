"""Privacy-preserving local retrieval traces (RE-02).

Every user-facing retrieval surface (CLI query, agent semantic search, MCP
query, dashboard query, evaluation) can emit one :class:`RetrievalRunRecord`
per query through :class:`RetrievalTraceRecorder`. Records are stored as a
single JSON Line under ``<project>/.ctxai/traces/<run_id>.jsonl`` — one file
per run, written atomically (temp file + rename + fsync), redacted via
``agent.run_recorder.redact_payload`` (repository-relative paths, then
``sessions.redact_secrets``) before anything touches disk.

Recording modes (resolved by :func:`resolve_trace_settings` from
``RetrievalConfig``):

- ``off`` (default): nothing is persisted; the explain path already gives
  per-query terminal insight.
- ``metrics``: identity, ordered candidates without source content, timings,
  counts, errors, and network-proof fields. The raw query is never stored
  (a deterministic hash by default, nothing at all with ``omit``).
- ``full``: everything in ``metrics`` plus the raw query text (when
  ``trace_query_text="store"``) and bounded source previews (when
  ``trace_source_preview="store"``). Enabling full mode shows a one-line
  privacy warning (see :func:`privacy_warning`).

No automatic upload, telemetry SDK, or remote exporter exists anywhere in
this module: the recorder's only transport is a local file write.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent.run_recorder import is_valid_run_id, redact_payload
from .config import RetrievalConfig

LOGGER = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1

TRACE_MODES = ("off", "metrics", "full")
QUERY_TEXT_MODES = ("omit", "hash", "store")
SOURCE_PREVIEW_MODES = ("omit", "store")

MIN_TRACE_RETENTION = 1
MAX_TRACE_RETENTION = 10_000
MIN_TRACE_RETENTION_DAYS = 1
MAX_TRACE_RETENTION_DAYS = 3_650
MIN_TRACE_PREVIEW_CHARS = 50
MAX_TRACE_PREVIEW_CHARS = 4_000
DEFAULT_TRACE_PREVIEW_CHARS = 500

TimestampClock = Callable[[], datetime]

_PRIVACY_WARNING = (
    "Retrieval tracing mode 'full' records raw query text and/or source previews locally "
    "under .ctxai/traces/; they stay on this machine but anyone with file access can read them."
)


def privacy_warning(mode: str) -> str | None:
    """Return the one-line privacy warning for a recording mode.

    Args:
        mode: The effective trace mode.

    Returns:
        The warning line for ``full`` mode, else ``None``.
    """
    if mode == TRACE_MODES[2]:
        return _PRIVACY_WARNING
    return None


def query_hash(query: str) -> str:
    """Compute the deterministic query hash stored when text is not recorded.

    Args:
        query: The raw query text.

    Returns:
        Sha256 hex digest of the UTF-8 query.
    """
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TraceSettings:
    """Resolved, validated tracing settings for one retrieval run.

    Attributes:
        mode: ``off``, ``metrics``, or ``full``.
        query_text: ``omit`` (nothing), ``hash`` (sha256), or ``store``
            (raw text; honored only in ``full`` mode).
        source_preview: ``omit`` or ``store`` (honored only in ``full`` mode).
        retention: Maximum number of retained trace files.
        retention_days: Maximum age in days of retained trace files.
        trace_dir: Optional trace-directory override inside the project.
        preview_chars: Bound on stored source previews.
    """

    mode: str = "off"
    query_text: str = "hash"
    source_preview: str = "omit"
    retention: int = 100
    retention_days: int = 30
    trace_dir: str | None = None
    preview_chars: int = DEFAULT_TRACE_PREVIEW_CHARS

    @property
    def enabled(self) -> bool:
        """Whether traces are persisted.

        Returns:
            True when the mode is ``metrics`` or ``full``.
        """
        return self.mode in ("metrics", "full")

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON/config representation.

        Returns:
            Dictionary of all fields.
        """
        return {
            "mode": self.mode,
            "query_text": self.query_text,
            "source_preview": self.source_preview,
            "retention": self.retention,
            "retention_days": self.retention_days,
            "trace_dir": self.trace_dir,
            "preview_chars": self.preview_chars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceSettings:
        """Rebuild from the JSON/config representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed settings.
        """
        return cls(
            mode=str(data.get("mode", "off")),
            query_text=str(data.get("query_text", "hash")),
            source_preview=str(data.get("source_preview", "omit")),
            retention=int(data.get("retention", 100)),
            retention_days=int(data.get("retention_days", 30)),
            trace_dir=data.get("trace_dir"),
            preview_chars=int(data.get("preview_chars", DEFAULT_TRACE_PREVIEW_CHARS)),
        )


def resolve_trace_settings(
    config: RetrievalConfig | None = None,
    *,
    enabled: bool | None = None,
    mode: str | None = None,
) -> TraceSettings:
    """Resolve tracing settings from the persisted retrieval configuration.

    ``metrics`` mode never stores raw queries or source content: a configured
    ``store`` query-text setting is coerced to ``hash`` and a ``store`` source
    preview is coerced to ``omit``. ``full`` mode honors both. An off config
    stays off unless ``enabled=True`` (the CLI ``--trace`` flag), which
    promotes it to ``metrics``.

    Args:
        config: Persisted retrieval configuration (defaults apply when None).
        enabled: Explicit enablement override (CLI ``--trace``); ``None`` uses
            the configured mode.
        mode: Explicit mode override taking precedence over the config.

    Returns:
        The validated :class:`TraceSettings`.

    Raises:
        ValueError: When the resolved mode is unknown.
    """
    source = config or RetrievalConfig()
    if mode is not None:
        effective_mode = mode
    elif enabled is False:
        effective_mode = "off"
    elif enabled is True and source.trace_mode == "off":
        effective_mode = "metrics"
    else:
        effective_mode = source.trace_mode
    if effective_mode not in TRACE_MODES:
        raise ValueError(f"trace_mode must be one of {', '.join(TRACE_MODES)}")
    query_text = source.trace_query_text
    source_preview = source.trace_source_preview
    if effective_mode == "off":
        query_text, source_preview = "omit", "omit"
    elif effective_mode == "metrics":
        # metrics persists neither raw query nor source, regardless of config.
        query_text = "hash" if query_text == "store" else query_text
        source_preview = "omit"
    return TraceSettings(
        mode=effective_mode,
        query_text=query_text,
        source_preview=source_preview,
        retention=source.trace_retention,
        retention_days=source.trace_retention_days,
        trace_dir=source.trace_dir,
        preview_chars=source.trace_preview_chars,
    )


def trace_dir_for(project_root: Path) -> Path:
    """Resolve the default trace directory for a project.

    Args:
        project_root: The project root directory.

    Returns:
        ``<project_root>/.ctxai/traces``.
    """
    return Path(project_root).resolve() / ".ctxai" / "traces"


def resolve_trace_dir(project_root: Path, override: str | Path | None = None) -> Path:
    """Resolve the trace directory, validating a project-contained override.

    Args:
        project_root: The resolved project root.
        override: Optional trace-directory override (``TraceSettings.trace_dir``).

    Returns:
        The trace directory (override when valid, default otherwise).

    Raises:
        ValueError: When the override escapes the project (deletion must stay
            scoped to the configured trace directory).
    """
    root = Path(project_root).resolve()
    if override is None:
        return trace_dir_for(root)
    candidate = Path(override).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("trace_dir override must stay inside the project directory")
    return resolved


@dataclass
class StageClock:
    """Stage-boundary stopwatch over an injected monotonic clock.

    ``mark(stage)`` closes the currently open stage and opens the next one;
    ``stop()`` closes the open stage. Durations accumulate per stage name so
    repeated stages add up.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        """Create the stopwatch.

        Args:
            clock: Injected monotonic clock (defaults to ``time.perf_counter``).
        """
        import time

        self._clock: Callable[[], float] = clock or time.perf_counter
        self._timings_ms: dict[str, float] = {}
        self._current: str | None = None
        self._started = self._clock()

    def mark(self, stage: str | None) -> None:
        """Close the open stage and open ``stage`` (or nothing when None).

        Closing a stage records its duration. A stage that was already closed
        once keeps its first measured duration: the clock is read once per
        boundary, so re-opening a stage measures a tail whose length depends
        on boundary placement rather than on the work done — the first closed
        span is the meaningful measurement.

        Args:
            stage: The next stage name, or ``None`` to leave nothing open.
        """
        now = self._clock()
        if self._current is not None:
            elapsed_ms = (now - self._started) * 1000.0
            if self._current not in self._timings_ms:
                self._timings_ms[self._current] = elapsed_ms
        self._current = stage
        self._started = now

    def stop(self) -> None:
        """Close the open stage, if any."""
        self.mark(None)

    @property
    def stage_timings_ms(self) -> dict[str, float]:
        """Closed stage durations in milliseconds.

        Returns:
            Mapping of stage name to accumulated duration.
        """
        return dict(self._timings_ms)


@dataclass
class RetrievalRunRecord:
    """One traced retrieval run (Part II ``RetrievalRun`` contract).

    Attributes:
        schema_version: On-disk schema version.
        run_id: Fresh identifier for this run (uuid4 hex).
        query_id: Identifier of the query within the run.
        timestamp: ISO-8601 UTC timestamp from the injected clock.
        mode: Recording mode the run was produced under.
        status: ``ok`` or ``error``.
        index: Index identity (name, schema version, embedding identity,
            repository revision, chunk count).
        graph: Graph identity (enabled, health, generation) when resolved.
        configuration: Retrieval configuration identity (token budget, limit,
            graph settings, trace settings, fingerprint).
        query: Query recording block (``recording``/``text``/``hash``/``length``).
        generators: One entry per candidate generator with its candidate count.
        candidates: Ordered candidate records (identifiers, component ranks
            and scores, graph path, decision, tokens, optional bounded preview).
        selected: Chunk ids of the final assembled context, in order.
        excluded: Examined-but-not-selected items with exclusion reasons.
        stage_timings_ms: Per-stage durations in milliseconds.
        total_latency_ms: End-to-end retrieval latency in milliseconds.
        candidate_count: Number of ranked candidates.
        selected_count: Number of selected evidence blocks.
        estimated_tokens: Estimated tokens of the assembled context.
        labels: Cohort/case labels (evaluation runs).
        errors: Retrieval error messages (redacted on persistence).
        diagnostics: Non-fatal diagnostics (redacted on persistence).
        network: Fields proving the trace pipeline made no outbound requests.
    """

    schema_version: int = TRACE_SCHEMA_VERSION
    run_id: str = ""
    query_id: str = ""
    timestamp: str = ""
    mode: str = "off"
    status: str = "ok"
    index: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    generators: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    candidate_count: int = 0
    selected_count: int = 0
    estimated_tokens: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    network: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the on-disk dictionary shape.

        Returns:
            Dictionary representation matching the persisted JSON line.
        """
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "status": self.status,
            "index": dict(self.index),
            "graph": dict(self.graph),
            "configuration": dict(self.configuration),
            "query": dict(self.query),
            "generators": [dict(generator) for generator in self.generators],
            "candidates": [dict(candidate) for candidate in self.candidates],
            "selected": list(self.selected),
            "excluded": [dict(item) for item in self.excluded],
            "stage_timings_ms": dict(self.stage_timings_ms),
            "total_latency_ms": self.total_latency_ms,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "estimated_tokens": self.estimated_tokens,
            "labels": dict(self.labels),
            "errors": list(self.errors),
            "diagnostics": list(self.diagnostics),
            "network": dict(self.network),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalRunRecord:
        """Rebuild from an on-disk dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict` or a compatible version.

        Returns:
            The reconstructed record.

        Raises:
            KeyError: When the ``run_id`` field is missing.
        """
        return cls(
            schema_version=int(data.get("schema_version", TRACE_SCHEMA_VERSION)),
            run_id=str(data["run_id"]),
            query_id=str(data.get("query_id", "")),
            timestamp=str(data.get("timestamp", "")),
            mode=str(data.get("mode", "off")),
            status=str(data.get("status", "ok")),
            index=dict(data.get("index") or {}),
            graph=dict(data.get("graph") or {}),
            configuration=dict(data.get("configuration") or {}),
            query=dict(data.get("query") or {}),
            generators=[dict(generator) for generator in data.get("generators") or []],
            candidates=[dict(candidate) for candidate in data.get("candidates") or []],
            selected=[str(item) for item in data.get("selected") or []],
            excluded=[dict(item) for item in data.get("excluded") or []],
            stage_timings_ms={str(key): float(value) for key, value in (data.get("stage_timings_ms") or {}).items()},
            total_latency_ms=float(data.get("total_latency_ms", 0.0)),
            candidate_count=int(data.get("candidate_count", 0)),
            selected_count=int(data.get("selected_count", 0)),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            labels={str(key): str(value) for key, value in (data.get("labels") or {}).items()},
            errors=[str(item) for item in data.get("errors") or []],
            diagnostics=[str(item) for item in data.get("diagnostics") or []],
            network=dict(data.get("network") or {}),
        )


def errored_run_record(
    *,
    run_id: str,
    query_id: str,
    timestamp: str,
    mode: str,
    query: str,
    settings: TraceSettings,
    index_name: str | None,
    error: str,
    total_latency_ms: float = 0.0,
    labels: dict[str, str] | None = None,
) -> RetrievalRunRecord:
    """Build the record for a retrieval that raised.

    Retrieval failures themselves remain observable (criterion 4): the run is
    recorded with ``status="error"`` best-effort, never masking the raise.

    Args:
        run_id: The run identifier.
        query_id: The query identifier.
        timestamp: ISO-8601 UTC timestamp.
        mode: Recording mode.
        query: The raw query (recorded per the settings).
        settings: The resolved trace settings.
        index_name: Requested index name, when known.
        error: The error message.
        total_latency_ms: Latency until the failure.
        labels: Optional cohort/case labels.

    Returns:
        The errored run record.
    """
    record = RetrievalRunRecord(
        run_id=run_id,
        query_id=query_id,
        timestamp=timestamp,
        mode=mode,
        status="error",
        query=_query_block(query, settings),
        configuration={"trace": settings.to_dict()},
        labels=dict(labels or {}),
        errors=[error],
        stage_timings_ms={},
        total_latency_ms=total_latency_ms,
        network=_network_block(None),
    )
    if index_name is not None:
        record.index = {"name": index_name}
    return record


@dataclass
class TraceOutcome:
    """What happened when a run was recorded.

    Attributes:
        run_id: The recorded run id, when known.
        mode: The trace mode the run was produced under.
        recorded: Whether a trace file was written.
        path: The trace file path, when recorded.
        diagnostic: A recording-failure diagnostic (recording failures never
            fail the retrieval; criterion 4).
    """

    run_id: str | None = None
    mode: str = "off"
    recorded: bool = False
    path: str | None = None
    diagnostic: str | None = None


class NullRetrievalTraceRecorder:
    """No-op recorder used when tracing is off or unavailable."""

    def __init__(self, mode: str = "off", diagnostic: str | None = None) -> None:
        """Initialize the no-op recorder.

        Args:
            mode: The trace mode that would have applied.
            diagnostic: Why recording is unavailable (construction failure).
        """
        self.mode = mode
        self.diagnostic = diagnostic
        self.run_id: str | None = None
        self.query_id: str | None = None

    def record(self, run: RetrievalRunRecord) -> TraceOutcome:
        """Drop the run.

        Args:
            run: The run record (ignored).

        Returns:
            A non-recorded outcome carrying the diagnostic, if any.
        """
        return TraceOutcome(run_id=self.run_id, mode=self.mode, recorded=False, diagnostic=self.diagnostic)


class RetrievalTraceRecorder:
    """Writes one redacted JSON Line per run under the trace directory.

    Writes are atomic (temp file + ``os.replace`` + fsync) so a concurrent
    reader never observes a partial line and concurrent writers cannot corrupt
    previously committed runs: one run is exactly one file, written once.
    Every failure is isolated — :meth:`record` never raises; failures surface
    as a diagnostic on the returned :class:`TraceOutcome`. After each
    successful write the retention policy (count + age) is applied.
    """

    def __init__(
        self,
        project_root: Path,
        settings: TraceSettings,
        *,
        run_id: str | None = None,
        query_id: str | None = None,
        timestamp_clock: TimestampClock | None = None,
    ):
        """Prepare the recorder.

        Args:
            project_root: Project root; the default trace directory lives
                inside it.
            settings: Resolved tracing settings.
            run_id: Pinned run id (fresh uuid4 hex by default).
            query_id: Pinned query id (fresh uuid4 hex by default).
            timestamp_clock: Injected clock for timestamps and retention.

        Raises:
            OSError: When the trace directory cannot be created (callers fall
                back to a no-op recorder).
        """
        self.project_root = Path(project_root).resolve()
        self.settings = settings
        self.trace_dir = resolve_trace_dir(self.project_root, settings.trace_dir)
        self.run_id = run_id or uuid.uuid4().hex
        self.query_id = query_id or uuid.uuid4().hex
        self._clock: TimestampClock = timestamp_clock or (lambda: datetime.now(timezone.utc))
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The trace file for this run.

        Returns:
            ``<trace_dir>/<run_id>.jsonl``.
        """
        return self.trace_dir / f"{self.run_id}.jsonl"

    def record(self, run: RetrievalRunRecord) -> TraceOutcome:
        """Redact and atomically persist one run record. Never raises.

        Args:
            run: The run record built by the caller.

        Returns:
            The outcome of the write attempt.
        """
        run_id = run.run_id or self.run_id
        try:
            payload = redact_payload(run.to_dict(), self.project_root)
            line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            _atomic_write_line(self.trace_dir / f"{run_id}.jsonl", line)
        except Exception as error:  # noqa: BLE001 - recorder failures are never fatal
            LOGGER.warning("retrieval trace (%s): write failed: %s", run_id, error)
            return TraceOutcome(
                run_id=run_id,
                mode=self.settings.mode,
                recorded=False,
                diagnostic=f"trace recording failed: {error}",
            )
        try:
            prune_traces(
                self.trace_dir,
                keep=self.settings.retention,
                max_age_days=self.settings.retention_days,
                now=self._clock(),
            )
        except Exception as error:  # noqa: BLE001 - retention failures are never fatal
            LOGGER.warning("retrieval trace (%s): retention failed: %s", run_id, error)
        return TraceOutcome(
            run_id=run_id,
            mode=self.settings.mode,
            recorded=True,
            path=str(self.trace_dir / f"{run_id}.jsonl"),
        )


def create_recorder(
    project_root: Path,
    settings: TraceSettings,
    *,
    run_id: str | None = None,
    query_id: str | None = None,
    timestamp_clock: TimestampClock | None = None,
) -> RetrievalTraceRecorder | NullRetrievalTraceRecorder:
    """Create the trace recorder, degrading to a no-op when off or failing.

    This is the single factory every retrieval surface uses; a recorder that
    cannot be created (permissions, override pointing at a file) becomes a
    no-op plus a diagnostic — recording problems never fail a query.

    Args:
        project_root: Project root for the default trace directory.
        settings: Resolved tracing settings.
        run_id: Pinned run id (fresh uuid4 hex by default).
        query_id: Pinned query id (fresh uuid4 hex by default).
        timestamp_clock: Injected clock for timestamps and retention.

    Returns:
        A :class:`RetrievalTraceRecorder` when tracing is possible, else a
        :class:`NullRetrievalTraceRecorder`.
    """
    if not settings.enabled:
        return NullRetrievalTraceRecorder(mode=settings.mode)
    try:
        return RetrievalTraceRecorder(
            project_root,
            settings,
            run_id=run_id,
            query_id=query_id,
            timestamp_clock=timestamp_clock,
        )
    except Exception as error:  # noqa: BLE001 - construction failures are never fatal
        LOGGER.warning("retrieval trace: recorder unavailable: %s", error)
        return NullRetrievalTraceRecorder(mode=settings.mode, diagnostic=f"trace recorder unavailable: {error}")


def _atomic_write_line(path: Path, line: str) -> None:
    """Write exactly one JSON line atomically with fsync.

    Args:
        path: Destination file path; parent must exist.
        line: The complete JSON line (newline appended).
    """
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prune_traces(
    trace_dir: Path,
    *,
    keep: int,
    max_age_days: int,
    now: datetime | None = None,
) -> list[Path]:
    """Apply the retention policy: drop traces older than ``max_age_days``
    and the oldest traces beyond ``keep``.

    Scoped to ``*.jsonl`` files directly inside ``trace_dir``; nested
    directories and unrelated files are never touched. A missing directory
    prunes nothing.

    Args:
        trace_dir: The resolved trace directory.
        keep: Number of newest traces to retain.
        max_age_days: Maximum trace age in days.
        now: Injected current time (defaults to UTC now).

    Returns:
        The deleted paths.
    """
    directory = Path(trace_dir)
    if not directory.is_dir() or keep < 0:
        return []
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(days=max(0, max_age_days))
    traces = [path for path in directory.glob("*.jsonl") if path.is_file()]
    deleted: list[Path] = []

    def _remove(path: Path) -> None:
        try:
            path.unlink()
            deleted.append(path)
        except OSError as error:
            LOGGER.warning("retrieval trace retention: could not delete %s: %s", path, error)

    aged: list[tuple[datetime, Path]] = []
    for path in traces:
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if modified < cutoff:
            _remove(path)
        else:
            aged.append((modified, path))
    remaining = sorted(aged, key=lambda item: (item[0], item[1].name))
    excess = len(remaining) - keep
    for _, path in remaining[: max(0, excess)]:
        _remove(path)
    return deleted


class TraceError(Exception):
    """Base class for trace store errors."""


class TraceNotFoundError(TraceError):
    """Raised when a trace run does not exist."""


class TraceCorruptError(TraceError):
    """Raised when a trace file cannot be parsed."""


@dataclass
class TraceSummary:
    """Projection of one trace run for list views.

    Attributes:
        run_id: The run identifier (trace file stem).
        timestamp: ISO-8601 timestamp of the run.
        status: ``ok`` or ``error``.
        mode: Recording mode the run was produced under.
        index_name: The index that served the query.
        candidate_count: Number of ranked candidates.
        selected_count: Number of selected evidence blocks.
        total_latency_ms: End-to-end retrieval latency.
        schema_version: On-disk schema version.
        labels: Cohort/case labels.
    """

    run_id: str
    timestamp: str
    status: str
    mode: str
    index_name: str | None
    candidate_count: int
    selected_count: int
    total_latency_ms: float
    schema_version: int
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the ``--json`` envelope shape.

        Returns:
            Dictionary representation of the summary.
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "mode": self.mode,
            "index_name": self.index_name,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "total_latency_ms": self.total_latency_ms,
            "schema_version": self.schema_version,
            "labels": dict(self.labels),
        }


def _load_record_file(path: Path) -> RetrievalRunRecord:
    """Parse a single-run trace file.

    Args:
        path: The ``.jsonl`` trace path.

    Returns:
        The parsed record.

    Raises:
        TraceCorruptError: When the file is empty, truncated, or malformed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise TraceCorruptError(f"trace file {path.name} is unreadable: {error}") from error
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise TraceCorruptError(f"trace file {path.name} is empty")
    if len(lines) > 1:
        raise TraceCorruptError(f"trace file {path.name} has unexpected extra lines")
    try:
        payload = json.loads(lines[0])
        return RetrievalRunRecord.from_dict(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise TraceCorruptError(f"corrupt trace file {path.name}: {error}") from error


def list_trace_runs(
    trace_dir: Path,
    *,
    limit: int | None = None,
    index_name: str | None = None,
    status: str | None = None,
) -> tuple[list[TraceSummary], list[str]]:
    """Summarize the stored traces, newest first.

    Corrupt trace files are skipped with a diagnostic instead of failing the
    listing (corruption recovery).

    Args:
        trace_dir: The resolved trace directory.
        limit: Maximum number of summaries to return (all when None).
        index_name: Only include runs of this index.
        status: Only include runs with this status.

    Returns:
        ``(summaries, corrupt_diagnostics)``.
    """
    directory = Path(trace_dir)
    summaries: list[TraceSummary] = []
    corrupt: list[str] = []
    if not directory.is_dir():
        return summaries, corrupt

    def _content_run_id(raw: str) -> str | None:
        """Extract the recorded run_id from a (possibly truncated) payload.

        Args:
            raw: Raw file content.

        Returns:
            The recorded run id when parseable, else ``None``.
        """
        match = re.search(r'"run_id"\s*:\s*"([^"]+)"', raw)
        return match.group(1) if match else None

    for path in sorted(directory.glob("*.jsonl")):
        if not path.is_file():
            continue
        try:
            record = _load_record_file(path)
        except TraceCorruptError as error:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw = ""
            recorded_id = _content_run_id(raw)
            label = f"run_id {recorded_id}" if recorded_id else path.stem
            corrupt.append(f"corrupt trace file ({label}): {error}")
            continue
        if index_name is not None and record.index.get("name") != index_name:
            continue
        if status is not None and record.status != status:
            continue
        summaries.append(
            TraceSummary(
                run_id=record.run_id or path.stem,
                timestamp=record.timestamp,
                status=record.status,
                mode=record.mode,
                index_name=record.index.get("name"),
                candidate_count=record.candidate_count,
                selected_count=record.selected_count,
                total_latency_ms=record.total_latency_ms,
                schema_version=record.schema_version,
                labels=dict(record.labels),
            )
        )
    summaries.sort(key=lambda summary: summary.timestamp or "", reverse=True)
    if limit is not None:
        summaries = summaries[: max(0, limit)]
    return summaries, corrupt


def load_run_record(run_id: str, trace_dir: Path) -> RetrievalRunRecord:
    """Load one trace run record.

    Args:
        run_id: The run identifier (trace file stem).
        trace_dir: The resolved trace directory.

    Returns:
        The parsed record.

    Raises:
        ValueError: When the run id is not a safe identifier.
        TraceNotFoundError: When the trace file does not exist.
        TraceCorruptError: When the trace file cannot be parsed.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id}")
    path = Path(trace_dir) / f"{run_id}.jsonl"
    if not path.is_file():
        # The recorded run_id inside a file is authoritative (a renamed or
        # re-id'd file is still resolvable); scan bounded, newest last.
        for candidate in sorted(Path(trace_dir).glob("*.jsonl")):
            try:
                raw = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            compact = raw.replace(" ", "")
            if f'"run_id":"{run_id}"' in compact:
                try:
                    record = _load_record_file(candidate)
                except TraceCorruptError as error:
                    raise TraceCorruptError(f"trace run '{run_id}' is corrupt: {error}") from error
                return record
        raise TraceNotFoundError(f"No retrieval trace for '{run_id}' under {trace_dir}")
    return _load_record_file(path)


def read_run_payload(run_id: str, trace_dir: Path) -> dict[str, Any]:
    """Load one trace's raw on-disk payload.

    Args:
        run_id: The run identifier (trace file stem).
        trace_dir: The resolved trace directory.

    Returns:
        The payload dictionary.

    Raises:
        ValueError: When the run id is not a safe identifier.
        TraceNotFoundError: When the trace file does not exist.
        TraceCorruptError: When the trace file cannot be parsed.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id}")
    path = Path(trace_dir) / f"{run_id}.jsonl"
    if not path.is_file():
        raise TraceNotFoundError(f"No retrieval trace for '{run_id}' under {trace_dir}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise TraceCorruptError(f"trace file {run_id}.jsonl is empty")
    try:
        return dict(json.loads(lines[0]))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise TraceCorruptError(f"corrupt trace file {run_id}.jsonl: {error}") from error


def delete_trace_run(run_id: str, trace_dir: Path) -> Path:
    """Delete one trace run file.

    Args:
        run_id: The run identifier (trace file stem).
        trace_dir: The resolved trace directory; deletion never leaves it.

    Returns:
        The deleted path.

    Raises:
        ValueError: When the run id is not a safe identifier.
        TraceNotFoundError: When the trace file does not exist.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id}")
    path = Path(trace_dir) / f"{run_id}.jsonl"
    if not path.is_file():
        raise TraceNotFoundError(f"No retrieval trace for '{run_id}' under {trace_dir}")
    path.unlink()
    return path


def delete_all_trace_runs(trace_dir: Path) -> int:
    """Delete every trace run file directly inside the trace directory.

    Args:
        trace_dir: The resolved trace directory.

    Returns:
        The number of trace files deleted.
    """
    directory = Path(trace_dir)
    if not directory.is_dir():
        return 0
    deleted = 0
    for path in directory.glob("*.jsonl"):
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


# ----------------------------------------------------------------------
# Record-building helpers shared by retrieval surfaces
# ----------------------------------------------------------------------


def configuration_fingerprint(payload: dict[str, Any]) -> str:
    """Content-derived fingerprint of retrieval configuration identity.

    Mirrors the evals artifact discipline (canonical JSON + sha256) without
    importing the evals package (which would create an import cycle through
    ``repository_context``).

    Args:
        payload: The configuration subset to fingerprint.

    Returns:
        Hex digest of the canonical JSON encoding.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _query_block(query: str, settings: TraceSettings) -> dict[str, Any]:
    """Build the query recording block for one run.

    Args:
        query: The raw query text.
        settings: Resolved trace settings.

    Returns:
        ``{"recording", "text", "hash", "length"}`` honoring the settings.
    """
    if settings.mode == "full" and settings.query_text == "store":
        return {"recording": "store", "text": query, "hash": query_hash(query), "length": len(query)}
    if settings.query_text == "hash":
        return {"recording": "hash", "text": None, "hash": query_hash(query), "length": len(query)}
    return {"recording": "omit", "text": None, "hash": None, "length": len(query)}


def _network_block(manifest: Any) -> dict[str, Any]:
    """Build the network-proof block for one run.

    Args:
        manifest: The index manifest (or ``None``), used for the embedding
            identity that is the only stage that may contact a configured
            endpoint.

    Returns:
        The network block.
    """
    embedding_provider = getattr(manifest, "embedding_provider", None)
    embedding_model = getattr(manifest, "embedding_model", None)
    return {
        "recorder_transport": "local-file-only",
        "outbound_transports": [],
        "embedding_provider": str(embedding_provider) if embedding_provider else None,
        "embedding_model": str(embedding_model) if embedding_model else None,
        "note": (
            "the trace recorder performs no network I/O; the configured embedding provider is the only "
            "stage that may contact an endpoint, and it receives only the query text needed to embed it"
        ),
    }
