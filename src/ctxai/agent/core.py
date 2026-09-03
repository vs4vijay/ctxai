"""
Core agent implementation with tool calling and planning.
"""

import asyncio
import hashlib
import logging
import random
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import AgentConfig
from .context import ConversationContext
from .llm.base import BaseLLMProvider, LLMResponse, Message, ProviderError, ProviderErrorKind, ToolCall
from .prompts import get_system_prompt, get_tool_error_recovery_prompt
from .resilience import RetryNotice, RetryPolicy, call_with_retry, format_retry_notice
from .run_recorder import (
    NullRunRecorder,
    RunEventKind,
    RunRecorder,
    create_recorder,
    new_run_id,
    prune_runs,
    runs_dir_for,
)
from .sessions import SessionRecord, SessionStore
from .tools.registry import ToolRegistry
from .workflow import (
    ApprovalCallback,
    FailureKind,
    TaskRun,
    TaskState,
    classify_provider_failure,
    discover_verification_commands,
)

LOGGER = logging.getLogger(__name__)

PLAN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit an evidence-backed execution plan for a complex or risky task.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "reasoning": {"type": "string"},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "string"},
                            "description": {"type": "string"},
                            "tool": {"type": "string"},
                            "parameters": {"type": "object"},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                            "completion_criteria": {"type": "string"},
                        },
                        "required": ["description", "tool", "parameters", "evidence", "completion_criteria"],
                    },
                },
            },
            "required": ["goal", "reasoning", "actions"],
        },
    },
}


@dataclass(frozen=True)
class CompactionNotice:
    """Structured description of one context compaction, rendered on CLI surfaces.

    Attributes:
        tokens_before: Heuristic token estimate before compaction.
        tokens_after: Heuristic token estimate after compaction.
        elided_messages: Tool-result bodies replaced by markers in this compaction.
        target_tokens: Soft context budget (context_size * soft-limit ratio).
    """

    tokens_before: int
    tokens_after: int
    elided_messages: int
    target_tokens: int


def format_compaction_notice(notice: CompactionNotice) -> str:
    """Render a compaction notice in the documented one-line CLI format.

    Args:
        notice: The compaction notice to render.

    Returns:
        A string shaped like ``context compacted: ~9000 -> ~2400 tokens
        (12 tool results elided, soft limit 8000)``.
    """
    return (
        f"context compacted: ~{notice.tokens_before} -> ~{notice.tokens_after} tokens "
        f"({notice.elided_messages} tool results elided, soft limit {notice.target_tokens})"
    )


@dataclass
class AgentLoopConfig:
    """Configuration for agent execution.

    All resilience fields are defaulted so existing constructions remain valid:
    provider calls retry transient failures per ``retry_policy``, a
    ``cancel_event`` enables clean Ctrl+C cancellation, ``on_retry`` receives
    structured retry notices for CLI rendering, ``session_store`` persists
    conversation state when a run is cancelled or fails fast,
    ``on_compaction`` receives structured compaction notices when the loop
    elides old tool results to stay under the model's context window (HH-03),
    and ``run_id`` pins the HH-04 run transcript identity: one-shot callers
    pass the ``ToolExecutionContext.request_id``; when ``None`` (the default)
    every run records under a fresh uuid4 hex so interactive chat runs never
    collide.
    """

    llm_provider: BaseLLMProvider
    tool_registry: ToolRegistry
    agent_config: AgentConfig
    working_directory: Path
    available_indexes: list[str]
    planning_enabled: bool = True
    require_user_approval: bool = True
    max_iterations: int = 10
    verbose: bool = False
    approval_callback: ApprovalCallback | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    cancel_event: asyncio.Event | None = None
    on_retry: Callable[[RetryNotice], None] | None = None
    session_store: SessionStore | None = None
    session_name: str = "default"
    on_compaction: Callable[[CompactionNotice], None] | None = None
    run_id: str | None = None


class Agent:
    """
    Autonomous coding agent with tool use and planning capabilities.
    """

    def __init__(self, config: AgentLoopConfig):
        """
        Initialize agent.

        Args:
            config: Agent loop configuration
        """
        self.config = config
        self.llm = config.llm_provider
        self.tools = config.tool_registry
        self.context = ConversationContext()
        self.console = Console(legacy_windows=False)
        self.last_run: TaskRun | None = None
        self._recorder: RunRecorder | NullRunRecorder | None = None
        self._retries_used_last_call = 0

        # Initialize system message
        tool_descriptions = self.tools.get_tool_descriptions()
        system_prompt = get_system_prompt(
            working_directory=config.working_directory,
            available_indexes=config.available_indexes,
            tool_descriptions=tool_descriptions,
            planning_enabled=config.planning_enabled,
            verification_commands=discover_verification_commands(config.working_directory),
        )
        self.context.add_system_message(system_prompt)

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message through the agent loop.

        Resilience semantics (HH-02): transient provider failures
        (RATE_LIMIT/TIMEOUT/TRANSPORT) are retried with bounded exponential
        backoff; AUTHENTICATION and UNSUPPORTED fail fast without burning
        iterations; INVALID_RESPONSE receives exactly one recovery prompt.
        Cancellation — via the loop's cancel event or ``asyncio``
        cancellation — marks the run failed with ``infrastructure_failure``,
        persists the session, and returns the final report; recovery prompts
        are never injected for cancellation. Recovery prompts for other
        (non-provider) failures keep the historical single-prompt behavior.

        Context management (HH-03): before every LLM call the estimated
        context size is compared against the provider's declared
        ``context_size``; above the soft limit the conversation is compacted
        (old tool-result bodies elided, pairing preserved) so long tool-heavy
        tasks continue instead of overflowing. Provider-reported usage is
        captured into the run's ``UsageLedger`` after every call, and a
        ``finish_reason == "length"`` response is treated as an
        INVALID_RESPONSE-class provider failure.

        Run transcripts (HH-04): every run records a redacted JSON Lines
        transcript under ``.ctxai/runs/<run_id>.jsonl`` through the shared
        ``RunRecorder`` — run start, user message, LLM calls with usage,
        tool calls/results, approvals, state transitions, checks,
        compactions, cancellation, and completion. Recording failures are
        diagnostics and never affect the run; with
        ``AgentBehaviorConfig.record_runs`` disabled nothing is written.

        Args:
            user_message: User's input message

        Returns:
            Agent's response (always a status-bearing final report)
        """
        # Add user message to context
        self.context.add_user_message(user_message)
        run = TaskRun(user_message, project_root=self.config.working_directory.resolve())
        self.last_run = run
        self._start_recording(run, user_message)

        if self.config.verbose:
            self.console.print(f"[dim]Processing: {user_message}[/dim]")

        threshold = max(1, self.config.agent_config.behavior.loop_break_threshold)
        result_hashes: deque[str] = deque(maxlen=threshold)
        invalid_response_recoveries = 0

        # Agent loop with tool calling
        iteration = 0
        try:
            while iteration < self.config.max_iterations:
                if self._cancel_requested():
                    return self._finish_cancelled(run)

                if self.config.verbose:
                    self.console.print(f"[dim]Iteration {iteration + 1}/{self.config.max_iterations}[/dim]")

                # Stay under the model's context window before spending a call (HH-03).
                self._enforce_context_budget()

                # Get messages for LLM
                messages = self.context.get_messages_for_llm()

                # Get tool schemas
                tool_format = self._get_tool_format()
                capabilities = self.llm.get_capabilities()
                tools = self.tools.get_all_schemas(format=tool_format) if capabilities.tools else None
                if tools is not None and self.config.planning_enabled:
                    tools.append(self._plan_tool_schema(tool_format))

                # Call LLM
                try:
                    response = await self._call_llm(messages, tools, run=run)

                    if self.config.verbose:
                        preview = response.content[:200] if response.content else "(empty)"
                        self.console.print(f"[dim]LLM response: {preview}...[/dim]")
                        self.console.print(f"[dim]Tool calls: {len(response.tool_calls)}[/dim]")

                    # Check if LLM wants to use tools
                    if response.has_tool_calls:
                        # Add assistant message with tool calls
                        self.context.add_assistant_message(response.content, tool_calls=response.tool_calls)

                        # Execute tools
                        tool_results = await self._execute_tools(response.tool_calls, run=run)

                        # Add tool results to context
                        current_results: list[str] = []
                        for tool_call, result in zip(response.tool_calls, tool_results):
                            result_text = self._format_tool_result(result)
                            current_results.append(result_text)
                            self.context.add_tool_result(
                                tool_call_id=tool_call.id, tool_name=tool_call.name, result=result_text
                            )

                        # Loop detection: break when the same tool-result tuple hash
                        # fills the window (threshold consecutive repeats).
                        if current_results:
                            result_hashes.append(self._tool_results_digest(current_results))
                            if len(result_hashes) == result_hashes.maxlen and len(set(result_hashes)) == 1:
                                if self.config.verbose:
                                    self.console.print("[yellow]! Detected tool loop, breaking[/yellow]")
                                run.failure_kind = FailureKind.INCOMPLETE_WORKFLOW
                                run.failure_message = (
                                    f"Detected {threshold} identical consecutive tool results; "
                                    "the agent loop made no progress"
                                )
                                return self._finalize(
                                    run,
                                    "I stopped because the same tool calls kept producing identical results "
                                    "without progress. Please rephrase the request or break it into smaller steps.",
                                )

                        iteration += 1
                        continue

                    else:
                        # No tool calls - this is the final response
                        self.context.add_assistant_message(response.content)

                        # Truncate context if needed
                        self.context.truncate_old_messages()

                        return self._finalize(run, response.content)

                except asyncio.CancelledError:
                    return self._finish_cancelled(run)

                except ProviderError as error:
                    if error.kind is ProviderErrorKind.CANCELLED:
                        return self._finish_cancelled(run)
                    if error.kind is ProviderErrorKind.INVALID_RESPONSE and invalid_response_recoveries < 1:
                        invalid_response_recoveries += 1
                        if self.config.verbose:
                            self.console.print(
                                f"[yellow]Invalid provider response, attempting recovery: {error}[/yellow]"
                            )
                        self.context.add_user_message(
                            get_tool_error_recovery_prompt(
                                tool_name="LLM", error=str(error), original_goal=user_message
                            )
                        )
                        iteration += 1
                        continue
                    return self._fail_fast(run, error)

                except Exception as e:
                    error_msg = f"Error during agent loop: {str(e)}"
                    if self.config.verbose:
                        self.console.print(f"[red]{error_msg}[/red]")

                    # Try to recover (reserved for non-provider failures such as bugs)
                    recovery_prompt = get_tool_error_recovery_prompt(
                        tool_name="LLM", error=str(e), original_goal=user_message
                    )
                    self.context.add_user_message(recovery_prompt)
                    iteration += 1
                    continue
        finally:
            # Safety net (HH-04): a run that unwinds without finalizing —
            # an unexpected BaseException such as KeyboardInterrupt — still
            # leaves a closed, parseable transcript. Normal paths close the
            # recorder in _finalize, which clears self._recorder first.
            if self._recorder is not None:
                self._record_event(
                    RunEventKind.RUN_COMPLETED,
                    {"status": "failed", "failure_message": "Run ended without a final report"},
                )
                self._close_recorder()

        # Max iterations reached
        run.failure_kind = run.failure_kind or FailureKind.INCOMPLETE_WORKFLOW
        run.failure_message = run.failure_message or (
            f"Max iterations ({self.config.max_iterations}) reached without completing the task"
        )
        return self._finalize(
            run,
            f"Max iterations ({self.config.max_iterations}) reached. "
            "The task may be too complex or an error occurred. "
            "Please try breaking it down into smaller steps.",
        )

    async def stream_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """Run the verified agent loop and yield its evidence-backed final report.

        Tool-capable turns intentionally share the same workflow as non-streaming turns so
        planning, approval, mutation, and verification policy cannot be bypassed by the UI.
        """
        yield await self.process_message(user_message)

    # ------------------------------------------------------------------
    # Run transcript recording (HH-04)
    # ------------------------------------------------------------------

    def _start_recording(self, run: TaskRun, user_message: str) -> None:
        """Open the run transcript, prune retention, and record the run start.

        Applies the ``run_retention`` cleanup (oldest transcripts first,
        scoped to the resolved runs directory) before opening the new
        transcript so the fresh run fits inside the retention window.

        Args:
            run: The TaskRun starting with this message.
            user_message: The user message that started the run.
        """
        behavior = self.config.agent_config.behavior
        run_id = self.config.run_id or new_run_id()
        if behavior.record_runs:
            try:
                prune_runs(runs_dir_for(self.config.working_directory), keep=behavior.run_retention - 1)
            except Exception as error:  # noqa: BLE001 - retention is a diagnostic, never fatal
                LOGGER.warning("run transcript (%s): retention pruning failed: %s", run_id, error)
        self._recorder = create_recorder(
            self.config.working_directory,
            run_id,
            enabled=behavior.record_runs,
            on_error=lambda message: LOGGER.warning("run transcript (%s): %s", run_id, message),
        )
        self._record_event(
            RunEventKind.RUN_STARTED,
            {
                "goal": user_message,
                "state": run.state.value,
                "provider": self.llm.__class__.__name__,
                "model": str(getattr(self.llm, "model", "") or "unknown"),
                "planning_enabled": self.config.planning_enabled,
                "require_user_approval": self.config.require_user_approval,
                "max_iterations": self.config.max_iterations,
                "working_directory": str(self.config.working_directory.resolve()),
                "project_root": str(run.project_root),
            },
        )
        self._record_event(RunEventKind.USER_MESSAGE, {"content": user_message})

    def _record_event(
        self,
        kind: RunEventKind,
        payload: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Record one transcript event; failures are diagnostics, never errors.

        Args:
            kind: The transcript event kind.
            payload: Event payload (redacted and path-normalized by the recorder).
            usage: Optional ``UsageRecord`` dict carried on the event.
        """
        recorder = self._recorder
        if recorder is None:
            return
        try:
            recorder.record(kind, payload, usage)
        except Exception as error:  # noqa: BLE001 - recording must never fail the run
            LOGGER.warning("run transcript: recording %s event failed: %s", kind.value, error)

    def _drain_run_events(self, run: TaskRun | None) -> None:
        """Emit pending TaskRun evidence (transitions/approvals/checks) as events.

        Args:
            run: The active TaskRun, or ``None`` when no run is active.
        """
        if run is None:
            return
        for kind, payload in run.to_event_payloads():
            self._record_event(kind, payload)

    def _finalize(self, run: TaskRun, model_summary: str) -> str:
        """Complete the run: final report, closing transcript events, close.

        Records the final state transition, the ``run_completed`` event with
        the run's evidence summary and usage ledger totals, then closes the
        transcript (flush + fsync). All run exit paths route through here.

        Args:
            run: The active TaskRun.
            model_summary: The model-facing summary for the final report.

        Returns:
            The status-bearing final report.
        """
        report = run.final_report(model_summary)
        self._drain_run_events(run)
        self._record_event(
            RunEventKind.RUN_COMPLETED,
            {
                "status": "succeeded" if run.state is TaskState.SUMMARIZE else "failed",
                "state": run.state.value,
                "failure_kind": run.failure_kind.value if run.failure_kind else None,
                "failure_message": run.failure_message,
                "changed_files": [self._relative_to_root(path, run.project_root) for path in sorted(run.changed_files)],
                "inspected_files": [
                    self._relative_to_root(path, run.project_root) for path in sorted(run.inspected_files)
                ],
                "checks": [{"command": check.command, "success": check.success} for check in run.checks],
                "plan_progress": run.plan.progress if run.plan is not None else None,
                "diff_reviewed": run.diff_reviewed,
                "usage": run.usage.totals(),
            },
        )
        self._close_recorder()
        return report

    def _close_recorder(self) -> None:
        """Close and detach the run transcript, if one is open."""
        if self._recorder is None:
            return
        try:
            self._recorder.close()
        except Exception as error:  # noqa: BLE001 - close failures are diagnostics
            LOGGER.warning("run transcript: close failed: %s", error)
        finally:
            self._recorder = None

    @staticmethod
    def _relative_to_root(path: Path, root: Path) -> str:
        """Render a path relative to the project root when possible.

        Args:
            path: The path to render.
            root: The project root.

        Returns:
            The repository-relative path, or the original string when the
            path lies outside the root.
        """
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    async def _call_llm(
        self, messages: list[Message], tools: list[dict[str, Any]] | None, *, run: TaskRun | None = None
    ) -> LLMResponse:
        """Invoke the provider chat call with retry, backoff, and cancellation support.

        The sync provider call runs in a worker thread so it never blocks the
        event loop shared with the MCP server and dashboard. Raised exceptions
        and ``finish_reason == "error"`` responses are normalized through
        ``normalize_error`` so the retry mapping sees a stable
        ``ProviderErrorKind``. A ``finish_reason == "length"`` response means
        the model hit its output limit and the payload is truncated; it is
        mapped to ``ProviderErrorKind.INVALID_RESPONSE`` so the loop's
        recovery path handles it instead of returning a cut-off answer.
        Only the LLM call is retried — tools are never re-executed by this
        path. Provider-reported usage is captured into ``run.usage`` for
        every call that returns a response (tokens only, never content).

        Args:
            messages: Conversation messages for the provider.
            tools: Optional tool schemas for the provider.
            run: Optional active TaskRun whose usage ledger records the call.

        Returns:
            The successful LLM response.

        Raises:
            ProviderError: Normalized provider failure (retryable kinds raise
                only after retries are exhausted).
            asyncio.CancelledError: When the cancel event is observed.
        """
        policy = self.config.retry_policy
        cancel_event = self.config.cancel_event
        retries_used = 0

        async def _attempt() -> LLMResponse:
            try:
                self.llm.validate_request(messages, tools, cancel_event=cancel_event)
                response = await asyncio.to_thread(self.llm.chat, messages, tools=tools)
            except Exception as error:
                raise self._normalize_provider_error(error) from error
            if run is not None:
                self._record_usage(run, response)
                # The llm_call event carries the same UsageRecord appended to
                # the ledger, so per-run usage totals always equal the sum of
                # transcript usage records (HH-04 acceptance criterion 4).
                records = run.usage.records
                last_record = records[-1] if records else None
                self._record_event(
                    RunEventKind.LLM_CALL,
                    {
                        "provider": self.llm.__class__.__name__,
                        "model": str(getattr(self.llm, "model", "") or "unknown"),
                        "call_index": len(records),
                        "finish_reason": response.finish_reason,
                        "retries_used": retries_used,
                    },
                    usage=last_record.to_dict() if last_record is not None else None,
                )
            if response.finish_reason == "error":
                raise self._normalize_provider_error(
                    RuntimeError(response.content or "Provider returned an error response without details")
                )
            if response.finish_reason == "length":
                raise ProviderError(
                    ProviderErrorKind.INVALID_RESPONSE,
                    f"{self.llm.__class__.__name__} returned a truncated response (finish_reason=length)",
                    provider=self.llm.__class__.__name__,
                )
            return response

        def _should_retry(error: Exception) -> bool:
            kind = error.kind if isinstance(error, ProviderError) else self._normalize_provider_error(error).kind
            return kind in policy.retry_kinds

        def _on_retry(notice: RetryNotice) -> None:
            nonlocal retries_used
            retries_used += 1
            if self.config.verbose:
                self.console.print(f"[yellow]{format_retry_notice(notice)}[/yellow]")
            if self.config.on_retry is not None:
                self.config.on_retry(notice)

        try:
            return await call_with_retry(
                _attempt,
                policy=policy,
                should_retry=_should_retry,
                sleep=asyncio.sleep,
                rng=random.Random(),  # nosec B311 - backoff jitter only, not security-sensitive
                cancel_event=cancel_event,
                on_retry=_on_retry,
            )
        finally:
            self._retries_used_last_call = retries_used

    def _record_usage(self, run: TaskRun, response: LLMResponse) -> None:
        """Capture provider-reported usage into the run ledger and the estimator.

        Args:
            run: The active TaskRun owning this run's UsageLedger.
            response: The LLM response whose usage payload is recorded.
        """
        usage = response.usage or {}
        if not usage:
            return
        run.usage.record(
            provider=self.llm.__class__.__name__,
            model=str(getattr(self.llm, "model", "") or "unknown"),
            usage=usage,
        )
        self.context.note_reported_usage(usage)

    def _enforce_context_budget(self) -> None:
        """Compact the conversation when its estimated size crosses the soft limit.

        The budget is ``context_size * context_soft_limit_ratio`` from the
        provider capabilities and agent behavior config. Providers without a
        usable ``context_size`` (non-positive or missing) skip the check. A
        real compaction prints a one-line notice in verbose mode and is
        reported through the ``on_compaction`` hook; no-op compactions are
        silent and uncounted.
        """
        context_size = getattr(self.llm.get_capabilities(), "context_size", None)
        if not isinstance(context_size, int) or isinstance(context_size, bool) or context_size <= 0:
            return
        behavior = self.config.agent_config.behavior
        budget = int(context_size * behavior.context_soft_limit_ratio)
        if budget <= 0 or self.context.estimate_context_tokens() <= budget:
            return
        changed = self.context.compact(
            target_tokens=budget,
            max_output_chars=self.config.agent_config.tools.max_output_chars,
        )
        if not changed:
            return
        stats = self.context.last_compaction or {}
        notice = CompactionNotice(
            tokens_before=int(stats.get("tokens_before", 0)),
            tokens_after=int(stats.get("tokens_after", 0)),
            elided_messages=int(stats.get("elided_messages", 0)),
            target_tokens=budget,
        )
        if self.config.verbose:
            self.console.print(f"[yellow]{format_compaction_notice(notice)}[/yellow]")
        if self.config.on_compaction is not None:
            self.config.on_compaction(notice)
        self._record_event(
            RunEventKind.COMPACTION,
            {
                "tokens_before": notice.tokens_before,
                "tokens_after": notice.tokens_after,
                "elided_messages": notice.elided_messages,
                "target_tokens": notice.target_tokens,
                "compaction_count": self.context.compaction_count,
                "elided_message_count": self.context.elided_message_count,
            },
        )

    def _normalize_provider_error(self, error: Exception) -> ProviderError:
        """Normalize an exception raised by the provider call.

        Args:
            error: The raised exception or an error-shaped response payload.

        Returns:
            A ProviderError with a stable kind and provider attribution.
        """
        if isinstance(error, ProviderError):
            return error
        return self.llm.normalize_error(error)

    def _cancel_requested(self) -> bool:
        """Check whether the configured cancel event has been set.

        Returns:
            True when a cancel event exists and is set.
        """
        return self.config.cancel_event is not None and self.config.cancel_event.is_set()

    def _finish_cancelled(self, run: TaskRun) -> str:
        """Complete a cancelled run cleanly: failed TaskRun, saved session, final report.

        No recovery prompt is injected on the cancellation path.

        Args:
            run: The active TaskRun.

        Returns:
            The status-bearing final report for the cancelled run.
        """
        run.failure_kind = classify_provider_failure(ProviderErrorKind.CANCELLED)
        run.failure_message = "Run cancelled before completion"
        self._record_event(RunEventKind.CANCELLATION, {"reason": "cancelled by user"})
        run.transition(TaskState.FAILED)
        self._save_session_snapshot()
        return self._finalize(run, "Cancelled by user")

    def _fail_fast(self, run: TaskRun, error: ProviderError) -> str:
        """End the run on a non-retryable provider failure without a recovery prompt.

        Args:
            run: The active TaskRun.
            error: The normalized provider error that ended the run.

        Returns:
            The status-bearing final report with a provider-qualified reason.
        """
        provider_name = error.provider or self.llm.__class__.__name__
        if error.kind in (ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.TIMEOUT, ProviderErrorKind.TRANSPORT):
            reason = (
                f"Provider {provider_name} {error.kind.value} error persisted after "
                f"{self._retries_used_last_call} retries: {error}"
            )
        elif error.kind is ProviderErrorKind.INVALID_RESPONSE:
            reason = f"Provider {provider_name} returned an invalid response after recovery: {error}"
        else:
            reason = f"Provider {provider_name} error ({error.kind.value}): {error}"
        run.failure_kind = classify_provider_failure(error.kind)
        run.failure_message = reason
        run.transition(TaskState.FAILED)
        self._save_session_snapshot()
        return self._finalize(run, reason)

    def _save_session_snapshot(self) -> None:
        """Persist the current conversation through the configured SessionStore, if any."""
        store = self.config.session_store
        if store is None:
            return
        record = SessionRecord(
            name=self.config.session_name,
            context=self.context,
            provider=self.llm.__class__.__name__,
            model=str(getattr(self.llm, "model", "") or "unknown"),
            project_root=str(self.config.working_directory.resolve()),
        )
        store.save(record)

    @staticmethod
    def _tool_results_digest(results: list[str]) -> str:
        """Hash a tool-result tuple so loop detection compares content, not identity.

        Args:
            results: Formatted tool result strings for one iteration.

        Returns:
            SHA-256 digest of the joined results.
        """
        return hashlib.sha256("\n\x1e".join(results).encode("utf-8")).hexdigest()

    async def _execute_tools(self, tool_calls: list[ToolCall], *, run: TaskRun | None = None) -> list[dict]:
        """
        Execute tool calls.

        Every call and its result are recorded as ``tool_call``/``tool_result``
        transcript events (HH-04); approval and state-transition events are
        drained from the TaskRun at the same boundaries so the transcript
        mirrors the workflow chronology. Recording never affects execution.

        Args:
            tool_calls: List of tool calls from LLM
            run: Optional active TaskRun for workflow policy and evidence.

        Returns:
            List of execution results
        """
        results = []

        for tool_call in tool_calls:
            if self.config.verbose:
                self.console.print(f"[cyan]Executing: {tool_call.name}[/cyan]")
                self.console.print(f"[dim]Parameters: {tool_call.parameters}[/dim]")

            self._record_event(
                RunEventKind.TOOL_CALL,
                {"tool": tool_call.name, "call_id": tool_call.id, "parameters": tool_call.parameters},
            )

            try:
                if tool_call.name == "submit_plan":
                    result = (
                        run.submit_plan(**tool_call.parameters)
                        if run is not None
                        else {"success": False, "error": "Planning is unavailable for this run"}
                    )
                    results.append(result)
                    self._record_tool_result_event(tool_call, result)
                    self._drain_run_events(run)
                    continue
                denial = None
                if run is not None:
                    denial = run.before_tool(
                        tool_call,
                        planning_enabled=self.config.planning_enabled,
                        require_approval=self.config.require_user_approval,
                        approval_callback=self.config.approval_callback,
                    )
                    # Approval prompts and policy denials happen inside
                    # before_tool; drain them before the result event.
                    self._drain_run_events(run)
                result = denial or await self.tools.execute_tool(tool_call.name, **tool_call.parameters)
                results.append(result)
                self._record_tool_result_event(tool_call, result)
                if run is not None:
                    run.observe(tool_call, result)
                    self._drain_run_events(run)

                if self.config.verbose:
                    if result.get("success"):
                        self.console.print(f"[green][OK] {tool_call.name} succeeded[/green]")
                        self._print_result_diagnostics(result)
                    else:
                        self.console.print(f"[yellow]⚠ {tool_call.name} failed: {result.get('error')}[/yellow]")

            except Exception as e:
                error_result = {"success": False, "result": None, "error": f"Tool execution exception: {str(e)}"}
                results.append(error_result)
                self._record_tool_result_event(tool_call, error_result)

                if self.config.verbose:
                    self.console.print(f"[red][X] {tool_call.name} exception: {str(e)}[/red]")

        return results

    def _record_tool_result_event(self, tool_call: ToolCall, result: dict) -> None:
        """Record one ``tool_result`` transcript event for an executed tool.

        Args:
            tool_call: The tool call that produced the result.
            result: The tool execution result dictionary.
        """
        self._record_event(
            RunEventKind.TOOL_RESULT,
            {
                "tool": tool_call.name,
                "call_id": tool_call.id,
                "success": bool(result.get("success")),
                "result": result.get("result"),
                "error": result.get("error"),
                "metadata": result.get("metadata"),
            },
        )

    def _print_result_diagnostics(self, result: dict) -> None:
        """Print output-truncation and replacement-count diagnostics for a tool result.

        Args:
            result: Tool execution result including its metadata.
        """
        metadata = result.get("metadata") or {}
        for stream in ("stdout", "stderr"):
            if metadata.get(f"{stream}_truncated"):
                self.console.print(
                    f"[dim]Truncated {stream}: "
                    f"{metadata.get(f'original_{stream}_chars')} chars, "
                    f"limit {self.config.agent_config.tools.max_output_chars}[/dim]"
                )
        if metadata.get("truncated"):
            self.console.print(
                f"[dim]Truncated content: {metadata.get('original_chars')} chars, "
                f"limit {self.config.agent_config.tools.max_output_chars}[/dim]"
            )
        if "replacements" in metadata:
            self.console.print(
                f"[dim]Edit applied: {metadata['replacements']} replacement(s) "
                f"via {metadata.get('strategy', 'exact')} match[/dim]"
            )

    @staticmethod
    def _plan_tool_schema(tool_format: str) -> dict:
        schema = PLAN_TOOL_SCHEMA["function"]
        if tool_format == "anthropic":
            return {"name": schema["name"], "description": schema["description"], "input_schema": schema["parameters"]}
        return PLAN_TOOL_SCHEMA

    def _format_tool_result(self, result: dict) -> str:
        """
        Format tool result for LLM.

        Args:
            result: Tool execution result

        Returns:
            Formatted result string
        """
        if result.get("success"):
            output = "Tool executed successfully.\n\n"
            if result.get("result"):
                output += str(result["result"])
            if result.get("metadata"):
                output += f"\n\nMetadata: {result['metadata']}"
            return output
        else:
            return f"Tool execution failed.\n\nError: {result.get('error', 'Unknown error')}"

    def _get_tool_format(self) -> str:
        """
        Get tool schema format based on LLM provider.

        Returns:
            Format string: "anthropic", "openai", or "generic"
        """
        provider_name = self.llm.__class__.__name__.lower()

        if "anthropic" in provider_name:
            return "anthropic"
        elif "openai" in provider_name or "openrouter" in provider_name or "custom" in provider_name:
            return "openai"
        elif "ollama" in provider_name:
            return "openai"
        else:
            return "openai"  # Default to OpenAI format (most widely supported)

    def clear_conversation(self) -> None:
        """Clear conversation history (except system message)."""
        self.context.clear()

    def get_conversation_summary(self) -> str:
        """
        Get summary of conversation.

        Returns:
            Summary string
        """
        return f"Messages: {self.context.get_message_count()}, Tokens: ~{self.context.get_token_count_estimate()}"

    def __repr__(self) -> str:
        return f"Agent(llm={self.llm.__class__.__name__}, tools={len(self.tools)})"
