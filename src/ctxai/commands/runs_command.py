"""Run transcript inspection and lifecycle operations for the CLI (HH-04).

Reads, summarizes, and deletes the redacted JSON Lines transcripts the agent
loop writes under ``<project>/.ctxai/runs/``. All operations are scoped to
the resolved runs directory of the given project; nothing here uploads data
or touches files outside that directory. Rendering happens in ``app.py``;
this module holds the logic and the shared usage/cost line formatting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.costing import estimate_run_cost, format_unknown_cost
from ..agent.run_recorder import (
    RUN_SCHEMA_VERSION,
    RunEvent,
    RunEventKind,
    is_valid_run_id,
    runs_dir_for,
)
from ..agent.workflow import TaskRun, UsageRecord

_USAGE_EVENT_KIND = RunEventKind.LLM_CALL.value


@dataclass
class RunSummary:
    """Projection of one run transcript for ``ctxai runs list``.

    Attributes:
        run_id: The run identifier (transcript file stem).
        started_at: Timestamp of the ``run_started`` event, when present.
        completed_at: Timestamp of the ``run_completed`` event, when present.
        status: ``succeeded``, ``failed``, or ``running`` (no completion event).
        event_count: Number of parsed events in the transcript.
        prompt_tokens: Total prompt tokens reported across ``llm_call`` events.
        completion_tokens: Total completion tokens reported across ``llm_call`` events.
        total_tokens: Total tokens reported across ``llm_call`` events.
        calls: Number of recorded LLM calls.
        cost: Estimated USD cost, or ``None`` when any model lacks a price entry.
        unknown_model: The first model id without a price entry, if any.
        schema_version: On-disk schema version of the transcript.
    """

    run_id: str
    started_at: str | None
    completed_at: str | None
    status: str
    event_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    calls: int
    cost: float | None
    unknown_model: str | None
    schema_version: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to the ``--json`` envelope shape.

        Returns:
            Dictionary representation of the summary.
        """
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "event_count": self.event_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "cost": self.cost,
            "unknown_model": self.unknown_model,
            "schema_version": self.schema_version,
        }

    @property
    def cost_display(self) -> str:
        """Render the cost cell: a dollar figure or the explicit unknown wording.

        Returns:
            ``$0.0123`` for a priced run, ``unknown (no price entry for M)``
            for an unpriced model, or ``-`` when no usage was recorded.
        """
        if self.calls == 0:
            return "-"
        if self.cost is None:
            return format_unknown_cost(self.unknown_model)
        return f"${self.cost:.4f}"


def resolve_runs_dir(project_path: Path | None = None) -> Path:
    """Resolve the runs directory for a project.

    Args:
        project_path: Project root; defaults to the current directory.

    Returns:
        ``<project>/.ctxai/runs``.
    """
    return runs_dir_for(project_path or Path.cwd())


def list_runs(project_path: Path | None = None, limit: int | None = None) -> list[RunSummary]:
    """Summarize the project's run transcripts, newest first.

    Args:
        project_path: Project root; defaults to the current directory.
        limit: Maximum number of runs to return (all when ``None``).

    Returns:
        Run summaries ordered by ``run_started`` timestamp, newest first.
        Unparseable transcripts are skipped rather than failing the listing.
    """
    runs_dir = resolve_runs_dir(project_path)
    summaries: list[RunSummary] = []
    if not runs_dir.is_dir():
        return summaries
    for path in sorted(runs_dir.glob("*.jsonl")):
        try:
            events = _read_events(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not events:
            continue
        summaries.append(_summarize(path.stem, events))
    summaries.sort(key=lambda summary: summary.started_at or "", reverse=True)
    if limit is not None:
        summaries = summaries[: max(0, limit)]
    return summaries


def read_run_events(run_id: str, project_path: Path | None = None) -> list[RunEvent]:
    """Parse one run transcript into events.

    Args:
        run_id: The run identifier (transcript file stem).
        project_path: Project root; defaults to the current directory.

    Returns:
        The parsed events in on-disk order.

    Raises:
        ValueError: When ``run_id`` is not a safe identifier or the
            transcript contains unparsable lines.
        FileNotFoundError: When the transcript does not exist.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id}")
    path = resolve_runs_dir(project_path) / f"{run_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No run transcript for '{run_id}' under {resolve_runs_dir(project_path)}")
    return _read_events(path)


def delete_run(run_id: str, project_path: Path | None = None) -> Path:
    """Delete one run transcript.

    Args:
        run_id: The run identifier (transcript file stem).
        project_path: Project root; defaults to the current directory.

    Returns:
        The deleted transcript path.

    Raises:
        ValueError: When ``run_id`` is not a safe identifier.
        FileNotFoundError: When the transcript does not exist.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run id: {run_id}")
    path = resolve_runs_dir(project_path) / f"{run_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No run transcript for '{run_id}'")
    path.unlink()
    return path


def delete_all_runs(project_path: Path | None = None) -> int:
    """Delete every run transcript in the project's runs directory.

    Args:
        project_path: Project root; defaults to the current directory.

    Returns:
        The number of transcripts deleted.
    """
    runs_dir = resolve_runs_dir(project_path)
    if not runs_dir.is_dir():
        return 0
    deleted = 0
    for path in runs_dir.glob("*.jsonl"):
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def format_usage_cost_line(run: TaskRun | None) -> str | None:
    """Format the usage/cost line appended to final agent reports.

    Args:
        run: The completed run whose usage ledger is rendered (``None`` or an
            empty ledger yields ``None`` — nothing is appended when the
            provider reported no usage).

    Returns:
        A line shaped like ``usage: 1,234 prompt + 567 completion tokens over
        3 call(s); cost: $0.0123``, or with ``cost: unknown (no price entry
        for <model>)`` when the model lacks a price entry. Never fabricates a
        zero cost for an unknown model.
    """
    if run is None or run.usage.call_count == 0:
        return None
    totals = run.usage.totals()
    estimate = estimate_run_cost(run.usage.records)
    if estimate.total_cost is None:
        cost_text = format_unknown_cost(estimate.unknown_model)
    else:
        cost_text = f"${estimate.total_cost:.4f}"
    return (
        f"usage: {totals['prompt_tokens']:,} prompt + {totals['completion_tokens']:,} completion tokens "
        f"over {totals['calls']} call(s); cost: {cost_text}"
    )


def _read_events(path: Path) -> list[RunEvent]:
    """Parse a transcript file, rejecting malformed lines.

    Args:
        path: The ``.jsonl`` transcript path.

    Returns:
        The parsed events in order.

    Raises:
        ValueError: When a line is not a valid event.
    """
    events: list[RunEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(RunEvent.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Corrupt transcript line in {path.name}: {error}") from error
    return events


def _summarize(run_id: str, events: list[RunEvent]) -> RunSummary:
    """Build a run summary from parsed events.

    Args:
        run_id: The run identifier.
        events: Parsed transcript events.

    Returns:
        The aggregate summary (usage from ``llm_call`` events, status from
        ``run_completed``).
    """
    started = next((event for event in events if event.kind == RunEventKind.RUN_STARTED.value), None)
    completed = next((event for event in events if event.kind == RunEventKind.RUN_COMPLETED.value), None)
    usage_events = [event.usage for event in events if event.kind == _USAGE_EVENT_KIND and event.usage]
    prompt_tokens = sum(int(usage.get("prompt_tokens") or 0) for usage in usage_events)
    completion_tokens = sum(int(usage.get("completion_tokens") or 0) for usage in usage_events)
    total_tokens = sum(int(usage.get("total_tokens") or 0) for usage in usage_events)
    estimate = estimate_run_cost(
        [record for record in (_usage_record_shape(usage) for usage in usage_events) if record is not None]
    )
    status = "running"
    if completed is not None:
        status = str(completed.payload.get("status") or "failed")
    return RunSummary(
        run_id=run_id,
        started_at=started.timestamp if started is not None else None,
        completed_at=completed.timestamp if completed is not None else None,
        status=status,
        event_count=len(events),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        calls=len(usage_events),
        cost=estimate.total_cost,
        unknown_model=estimate.unknown_model,
        schema_version=events[0].schema_version if events else RUN_SCHEMA_VERSION,
    )


def _usage_record_shape(usage: dict[str, Any]) -> UsageRecord | None:
    """Coerce a usage payload into the record shape estimate_run_cost expects.

    Args:
        usage: A ``UsageRecord``-shaped dict from an ``llm_call`` event.

    Returns:
        The ``UsageRecord``, or ``None`` when the payload lacks a model id.
    """
    model = usage.get("model")
    if not model:
        return None
    return UsageRecord(
        provider=str(usage.get("provider", "")),
        model=str(model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )
