"""Deterministic task state and evidence tracking for verified agent runs."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointManager
from .editing import EditError, edit_diff, simulate_edit
from .llm.base import ProviderErrorKind, ToolCall
from .run_recorder import RunEventKind

LOGGER = logging.getLogger(__name__)


class TaskState(str, Enum):
    UNDERSTAND = "understand"
    RETRIEVE = "retrieve"
    PLAN = "plan"
    APPROVE = "approve"
    EXECUTE = "execute"
    VERIFY = "verify"
    SUMMARIZE = "summarize"
    FAILED = "failed"


class FailureKind(str, Enum):
    RECOVERABLE_TOOL_ERROR = "recoverable_tool_error"
    POLICY_DENIAL = "policy_denial"
    TEST_FAILURE = "test_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    APPROVAL_DENIAL = "approval_denial"
    INCOMPLETE_WORKFLOW = "incomplete_workflow"


_PROVIDER_FAILURE_KINDS: dict[ProviderErrorKind, FailureKind] = {
    kind: FailureKind.INFRASTRUCTURE_FAILURE for kind in ProviderErrorKind
}


def classify_provider_failure(kind: ProviderErrorKind) -> FailureKind:
    """Map a provider error kind onto the shared FailureKind taxonomy.

    Provider faults (authentication, rate limits, timeouts, transport,
    malformed responses, cancellation) are infrastructure failures: they are
    never caused by the model's plan or by tool misuse.

    Args:
        kind: Normalized provider error kind raised by the LLM call.

    Returns:
        The FailureKind recorded on the TaskRun.
    """
    return _PROVIDER_FAILURE_KINDS.get(kind, FailureKind.INFRASTRUCTURE_FAILURE)


ApprovalCallback = Callable[[ToolCall], bool]

PLAN_MODES: tuple[str, ...] = ("auto", "force", "off")
"""Explicit planning override values: keyword classification, always plan, never plan."""

MAX_APPROVAL_REPROMPTS = 3
"""Stale-approval re-prompts allowed before the call fails with APPROVAL_DENIAL."""


def validate_plan_mode(value: str) -> str:
    """Validate a plan-mode override value.

    Args:
        value: One of ``auto`` (keyword classification), ``force`` (always plan),
            or ``off`` (never plan).

    Returns:
        The validated value.

    Raises:
        ValueError: When the value is not a known plan mode.
    """
    if value not in PLAN_MODES:
        raise ValueError(f"plan_mode must be one of {', '.join(PLAN_MODES)}")
    return value


def format_approval_prompt(call: ToolCall) -> str:
    """Render the exact action presented to a human approval callback."""
    target = call.parameters.get("approval_target") or call.name
    proposed_diff = call.parameters.get("proposed_diff")
    prompt = f"Approve {call.name}: {target}?"
    if proposed_diff:
        prompt = f"Proposed diff:\n{proposed_diff}\n{prompt}"
    return prompt


@dataclass
class PlannedAction:
    """One measurable, evidence-backed action proposed by the model."""

    action_id: str
    description: str
    tool: str
    parameters: dict[str, Any]
    evidence: list[str]
    completion_criteria: str
    status: str = "pending"
    result: str | None = None

    def matches(self, call: ToolCall) -> bool:
        return self.tool == call.name and all(
            call.parameters.get(key) == value for key, value in self.parameters.items()
        )


@dataclass
class StructuredPlan:
    goal: str
    reasoning: str
    actions: list[PlannedAction]

    @property
    def progress(self) -> str:
        completed = sum(action.status == "completed" for action in self.actions)
        failed = sum(action.status == "failed" for action in self.actions)
        return f"{completed}/{len(self.actions)} completed, {failed} failed"


def discover_verification_commands(project_root: Path) -> list[str]:
    """Return the smallest conventional checks supported by visible project files."""
    candidates = (
        ("pyproject.toml", "python -m pytest"),
        ("pytest.ini", "python -m pytest"),
        ("package.json", "npm test"),
        ("Cargo.toml", "cargo test"),
        ("go.mod", "go test ./..."),
    )
    commands: list[str] = []
    for marker, command in candidates:
        if (project_root / marker).is_file() and command not in commands:
            commands.append(command)
    return commands


@dataclass
class CheckEvidence:
    command: str
    success: bool
    output: str


@dataclass
class UsageRecord:
    """One provider-reported usage snapshot for a single LLM call.

    Stores token counts only — never message content.

    Attributes:
        provider: Provider class name that reported the usage.
        model: Model identifier the call was made against.
        prompt_tokens: Provider-reported prompt (input) tokens.
        completion_tokens: Provider-reported completion (output) tokens.
        total_tokens: Reported total, or prompt + completion when absent.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to the persisted dictionary shape (HH-04 transcripts).

        Returns:
            Dictionary representation of the usage snapshot.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageRecord:
        """Create from a dictionary.

        Args:
            data: Dictionary produced by ``to_dict``.

        Returns:
            The reconstructed ``UsageRecord``.
        """
        return cls(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
        )


@dataclass
class UsageLedger:
    """Aggregates provider-reported token usage for one agent run.

    Each successful LLM call records its ``LLMResponse.usage`` payload here;
    totals are therefore exactly the sum of the per-call provider reports.
    """

    records: list[UsageRecord] = field(default_factory=list)

    def record(self, provider: str, model: str, usage: dict[str, Any]) -> None:
        """Record one provider-reported usage payload.

        Empty payloads (providers that reported nothing) are ignored.

        Args:
            provider: Provider class name that made the call.
            model: Model identifier used for the call.
            usage: Provider-reported usage dict (``prompt_tokens``,
                ``completion_tokens``, optionally ``total_tokens``).
        """
        if not usage:
            return
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reported_total = usage.get("total_tokens")
        total_tokens = int(reported_total) if reported_total is not None else prompt_tokens + completion_tokens
        self.records.append(
            UsageRecord(
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )

    @property
    def call_count(self) -> int:
        """Number of recorded LLM calls.

        Returns:
            Count of recorded usage snapshots.
        """
        return len(self.records)

    def totals(self) -> dict[str, int]:
        """Aggregate recorded usage across the run.

        Returns:
            Dictionary with ``prompt_tokens``, ``completion_tokens``,
            ``total_tokens`` sums and the number of recorded ``calls``.
        """
        return {
            "prompt_tokens": sum(record.prompt_tokens for record in self.records),
            "completion_tokens": sum(record.completion_tokens for record in self.records),
            "total_tokens": sum(record.total_tokens for record in self.records),
            "calls": len(self.records),
        }


@dataclass
class TaskRun:
    goal: str
    project_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    state: TaskState = TaskState.UNDERSTAND
    inspected_files: set[Path] = field(default_factory=set)
    changed_files: set[Path] = field(default_factory=set)
    checks: list[CheckEvidence] = field(default_factory=list)
    diff_reviewed: bool = False
    failure_kind: FailureKind | None = None
    failure_message: str | None = None
    transitions: list[TaskState] = field(default_factory=lambda: [TaskState.UNDERSTAND])
    plan_required: bool = False
    plan: StructuredPlan | None = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageLedger = field(default_factory=UsageLedger)
    # HH-06: transcript identity and the checkpoint manager the loop wires in.
    # The manager is optional (None = checkpointing not wired for this surface);
    # when present, before_tool captures pre-mutation bytes ahead of every
    # approved write_file/edit_file call.
    run_id: str | None = None
    checkpoint_manager: CheckpointManager | None = None
    # HH-07: explicit planning override channel ("auto" keeps keyword
    # classification; "force" always requires submit_plan; "off" never does).
    plan_mode: str = "auto"

    MUTATION_TOOLS = frozenset({"write_file", "edit_file"})
    INSPECTION_TOOLS = frozenset({"read_file", "semantic_search", "grep", "glob", "list_files"})
    DIFF_TOOLS = frozenset({"git_diff"})
    VERIFY_TOOLS = frozenset({"bash"})

    def __post_init__(self) -> None:
        self.plan_mode = validate_plan_mode(self.plan_mode)
        self.project_root = Path(os.path.realpath(self.project_root))
        self.plan_required = self.resolve_plan_required(self.goal, self.plan_mode)
        # Drain cursors for to_event_payloads(): how many entries of each
        # evidence list have already been emitted as transcript events.
        self._emitted_transitions = 1  # transitions[0] is the initial state carried by run_started
        self._emitted_approvals = 0
        self._emitted_checks = 0

    @staticmethod
    def requires_plan(goal: str) -> bool:
        """Classify requests whose scope, uncertainty, or risk merits a plan."""
        normalized = goal.lower()
        signals = (
            "refactor",
            "migrate",
            "redesign",
            "across",
            "multiple files",
            "end to end",
            "architecture",
            "breaking change",
            "delete",
            "remove all",
            "security",
            "database",
            "dependency upgrade",
        )
        return any(signal in normalized for signal in signals)

    @staticmethod
    def resolve_plan_required(goal: str, plan_mode: str = "auto") -> bool:
        """Classify whether a goal must go through submit_plan under a plan mode.

        The explicit override channel (HH-07) wins over keyword classification:
        ``force`` requires a plan for every task, ``off`` never does (tools stay
        policy-gated), and ``auto`` keeps :meth:`requires_plan`.

        Args:
            goal: The user's task goal.
            plan_mode: ``auto``, ``force``, or ``off``.

        Returns:
            Whether mutations/verification require an approved plan.

        Raises:
            ValueError: When plan_mode is not a known plan mode.
        """
        validate_plan_mode(plan_mode)
        if plan_mode == "force":
            return True
        if plan_mode == "off":
            return False
        return TaskRun.requires_plan(goal)

    def submit_plan(self, *, goal: str, reasoning: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate and store a task-specific plan grounded in inspected evidence."""
        if not goal.strip() or not reasoning.strip() or not actions:
            return self._deny(
                FailureKind.INCOMPLETE_WORKFLOW,
                "Plan requires a goal, reasoning, and at least one action",
            )
        planned: list[PlannedAction] = []
        for index, item in enumerate(actions, start=1):
            evidence = item.get("evidence") or []
            criteria = str(item.get("completion_criteria", "")).strip()
            tool = str(item.get("tool", "")).strip()
            parameters = item.get("parameters") or {}
            if not tool or not criteria or not evidence or not isinstance(parameters, dict):
                return self._deny(
                    FailureKind.INCOMPLETE_WORKFLOW,
                    f"Plan action {index} requires tool, parameters, evidence, and completion criteria",
                )
            for citation in evidence:
                match = re.fullmatch(r"(.+):(\d+)-(\d+)", str(citation))
                if not match:
                    return self._deny(
                        FailureKind.INCOMPLETE_WORKFLOW,
                        f"Invalid plan evidence citation: {citation}",
                    )
                path = Path(match.group(1)).expanduser()
                if not path.is_absolute():
                    path = self.project_root / path
                if self.canonical(path) not in self.inspected_files:
                    return self._deny(
                        FailureKind.INCOMPLETE_WORKFLOW,
                        f"Plan evidence was not inspected: {citation}",
                    )
            planned.append(
                PlannedAction(
                    action_id=str(item.get("action_id") or f"action-{index}"),
                    description=str(item.get("description", "")).strip() or f"Run {tool}",
                    tool=tool,
                    parameters=parameters,
                    evidence=[str(value) for value in evidence],
                    completion_criteria=criteria,
                )
            )
        self.plan = StructuredPlan(goal=goal.strip(), reasoning=reasoning.strip(), actions=planned)
        self.failure_kind = None
        self.failure_message = None
        self.transition(TaskState.PLAN)
        return {"success": True, "result": self.plan.progress}

    def _planned_action(self, call: ToolCall) -> PlannedAction | None:
        if self.plan is None:
            return None
        return next((action for action in self.plan.actions if action.matches(call)), None)

    def _approval_call(self, call: ToolCall) -> ToolCall:
        parameters = dict(call.parameters)
        target_value = parameters.get("path") or parameters.get("file_path")
        if call.name in self.MUTATION_TOOLS and target_value:
            target = Path(str(target_value)).expanduser()
            if not target.is_absolute():
                target = self.project_root / target
            before = target.read_text(encoding="utf-8") if target.is_file() else ""
            try:
                after, _count = simulate_edit(call.name, parameters, before)
            except EditError:
                # An ambiguous edit cannot be previewed; it is denied at
                # execution with a count-bearing error, so nothing is written.
                after = before
            proposed_diff = edit_diff(str(target_value), before, after)
            parameters["proposed_diff"] = proposed_diff
            # HH-07 approval binding: hash the exact diff shown and the exact
            # pre-approval bytes so the loop can detect a stale approval (the
            # file changed between the prompt and execution) and re-prompt
            # instead of executing an outdated diff. ``None`` records that the
            # file did not exist at approval time (create semantics).
            parameters["proposed_diff_sha256"] = hashlib.sha256(proposed_diff.encode("utf-8")).hexdigest()
            parameters["pre_approval_content_sha256"] = self._content_hash(target)
        parameters["approval_target"] = target_value or parameters.get("command") or call.name
        return ToolCall(id=call.id, name=call.name, parameters=parameters)

    @staticmethod
    def _content_hash(target: Path) -> str | None:
        """SHA-256 of a file's bytes, or None when absent or unreadable.

        Args:
            target: The file to hash.

        Returns:
            The hex digest, or None when the file does not exist (or its bytes
            cannot be read), which the binding check treats as "expected absent".
        """
        try:
            if not target.is_file():
                return None
            return hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            return None

    def _canonical_mutation_target(self, call: ToolCall) -> Path | None:
        """Canonical absolute target path of a mutation call, or None.

        Args:
            call: The tool call holding ``path``/``file_path`` parameters.

        Returns:
            The realpath'd target path, or None when the call carries no path.
        """
        target_value = call.parameters.get("path") or call.parameters.get("file_path")
        if not target_value:
            return None
        target = Path(str(target_value)).expanduser()
        if not target.is_absolute():
            target = self.project_root / target
        return self.canonical(target)

    def _stale_approval_reason(self, call: ToolCall, approval_call: ToolCall) -> str | None:
        """Return why an approval is stale (binding mismatch), or None when fresh.

        Mutation approvals bind to the exact pre-approval bytes: the target must
        still hash to the value captured when the diff was shown. A file that
        appeared after an approval for a not-yet-existing file is equally stale.
        Commands carry no content binding.

        Args:
            call: The original tool call.
            approval_call: The approval-shaped call holding the binding hashes.

        Returns:
            A human-readable staleness reason, or None when the binding holds.
        """
        if call.name not in self.MUTATION_TOOLS:
            return None
        target = self._canonical_mutation_target(call)
        if target is None:
            return None
        expected = approval_call.parameters.get("pre_approval_content_sha256")
        if expected is None:
            if target.exists():
                return f"{target} appeared after the diff was shown"
            return None
        if not target.is_file():
            return f"{target} was removed after the diff was shown"
        if self._content_hash(target) != expected:
            return f"{target} changed after the diff was shown"
        return None

    @staticmethod
    def canonical(path: Path) -> Path:
        return Path(os.path.realpath(path))

    @property
    def mutated(self) -> bool:
        return bool(self.changed_files)

    def transition(self, state: TaskState) -> None:
        if self.state != state:
            self.state = state
            self.transitions.append(state)

    def to_event_payloads(self) -> list[tuple[RunEventKind, dict[str, Any]]]:
        """Drain new workflow evidence as redactable transcript event payloads.

        Returns state transitions, approvals, and checks recorded since the
        previous call (the initial ``UNDERSTAND`` state is carried by the
        ``run_started`` event, not emitted as a transition). The agent loop
        calls this at tool-batch boundaries so recording stays outside its
        control flow; each payload passes through the recorder's redaction
        before persistence.

        Returns:
            A list of ``(RunEventKind, payload)`` pairs in emission order:
            new transitions, then new approvals, then new checks.
        """
        events: list[tuple[RunEventKind, dict[str, Any]]] = []
        for state in self.transitions[self._emitted_transitions :]:
            events.append(
                (
                    RunEventKind.STATE_TRANSITION,
                    {
                        "state": state.value,
                        "failure_kind": self.failure_kind.value if self.failure_kind else None,
                        "failure_message": self.failure_message,
                    },
                )
            )
        self._emitted_transitions = len(self.transitions)

        for approval in self.approvals[self._emitted_approvals :]:
            events.append(
                (
                    RunEventKind.APPROVAL,
                    {
                        "tool": approval.get("tool"),
                        "parameters": approval.get("parameters") or {},
                        "approved": bool(approval.get("approved")),
                        # HH-07: the decision scope behind the boolean —
                        # "once", "session", or "deny" (None on surfaces that
                        # resolve decisions without the loop's resolver).
                        "decision": approval.get("decision"),
                    },
                )
            )
        self._emitted_approvals = len(self.approvals)

        for check in self.checks[self._emitted_checks :]:
            events.append(
                (
                    RunEventKind.CHECK,
                    {
                        "command": check.command,
                        "success": check.success,
                        "output": check.output,
                    },
                )
            )
        self._emitted_checks = len(self.checks)
        return events

    def classify_failure(self, result: dict[str, Any], tool_name: str) -> FailureKind:
        error_type = str(result.get("error_type", ""))
        error = str(result.get("error", ""))
        try:
            return FailureKind(error_type)
        except ValueError:
            pass
        if error_type == "PolicyDenied" or "Capability denied" in error:
            return FailureKind.POLICY_DENIAL
        if tool_name == "bash":
            return FailureKind.TEST_FAILURE
        if "exception" in error.lower() or error_type in {"OSError", "TimeoutExpired"}:
            return FailureKind.INFRASTRUCTURE_FAILURE
        return FailureKind.RECOVERABLE_TOOL_ERROR

    def before_tool(
        self,
        call: ToolCall,
        *,
        planning_enabled: bool,
        require_approval: bool,
        approval_callback: ApprovalCallback | None,
    ) -> dict[str, Any] | None:
        """Return a synthetic denial when a tool call violates workflow policy."""
        approval_tools = self.MUTATION_TOOLS | self.VERIFY_TOOLS
        if call.name not in approval_tools:
            if call.name in self.INSPECTION_TOOLS:
                self.transition(TaskState.RETRIEVE)
            elif call.name in self.VERIFY_TOOLS:
                self.transition(TaskState.VERIFY)
            return None

        if planning_enabled and self.plan_required:
            action = self._planned_action(call)
            if self.plan is None:
                return self._deny(
                    FailureKind.INCOMPLETE_WORKFLOW,
                    "Workflow denied: this complex or risky task requires submit_plan before execution",
                )
            if action is None:
                return self._deny(
                    FailureKind.INCOMPLETE_WORKFLOW,
                    f"Workflow denied: {call.name} is not an exact action in the approved plan",
                )
        target_value = call.parameters.get("path") or call.parameters.get("file_path")
        target = self._canonical_mutation_target(call)
        must_inspect = call.name == "edit_file" or bool(target and target.exists())
        if must_inspect and target not in self.inspected_files:
            return self._deny(
                FailureKind.INCOMPLETE_WORKFLOW,
                f"Workflow denied: read {target_value} successfully before editing it",
            )
        if require_approval:
            self.transition(TaskState.APPROVE)
            # HH-07: each round builds a fresh approval call (fresh diff and
            # fresh binding hashes), records the decision, and — for mutations
            # — verifies the file still matches the approved pre-approval
            # bytes. A stale approval re-prompts (the human sees a fresh diff)
            # and never executes; after MAX_APPROVAL_REPROMPTS stale rounds the
            # call fails with APPROVAL_DENIAL instead of looping forever.
            stale_reprompts = 0
            while True:
                approval_call = self._approval_call(call)
                approved = approval_callback is not None and approval_callback(approval_call)
                self.approvals.append(
                    {
                        "tool": call.name,
                        "parameters": approval_call.parameters,
                        "approved": approved,
                    }
                )
                if not approved:
                    return self._deny(
                        FailureKind.APPROVAL_DENIAL,
                        f"Approval denied for {call.name}: {approval_call.parameters['approval_target']}",
                    )
                stale_reason = self._stale_approval_reason(call, approval_call)
                if stale_reason is None:
                    break
                stale_reprompts += 1
                if stale_reprompts > MAX_APPROVAL_REPROMPTS:
                    return self._deny(
                        FailureKind.APPROVAL_DENIAL,
                        f"Approval for {call.name} went stale {stale_reprompts} times "
                        f"({stale_reason}); the approved diff no longer matches the file",
                    )
        # HH-06: capture the pre-mutation state before the approved mutation
        # runs (first touch per run). Failures are diagnostics — checkpointing
        # never blocks the mutation.
        if call.name in self.MUTATION_TOOLS:
            self._capture_mutation(target)
        self.transition(TaskState.EXECUTE)
        action = self._planned_action(call)
        if action is not None:
            action.status = "in_progress"
        return None

    def _capture_mutation(self, target: Path | None) -> None:
        """Capture the pre-mutation bytes of a file about to be mutated (HH-06).

        Called from the ``before_tool`` mutation path after policy and approval
        passed and before the tool executes, so every captured mutation is
        preceded by its checkpoint. Capture refusals (symlinks, paths escaping
        the project root, size cap) and I/O errors are logged diagnostics that
        never fail the tool call.

        Args:
            target: The canonical target path of the mutation, or ``None``.
        """
        manager = self.checkpoint_manager
        if manager is None or target is None or not self.run_id:
            return
        try:
            outcome = manager.capture(self.run_id, target)
        except Exception as error:  # noqa: BLE001 - checkpointing never blocks a mutation
            LOGGER.warning("checkpoints (%s): capture failed for %s: %s", self.run_id, target, error)
            return
        if outcome.status not in {"captured", "already"} and outcome.reason:
            LOGGER.warning("checkpoints (%s): %s", self.run_id, outcome.reason)

    def observe(self, call: ToolCall, result: dict[str, Any]) -> None:
        success = bool(result.get("success"))
        action = self._planned_action(call)
        if action is not None:
            action.status = "completed" if success else "failed"
            action.result = str(result.get("result") if success else result.get("error"))
        if not success:
            self.failure_kind = self.classify_failure(result, call.name)
            self.failure_message = str(result.get("error", "Unknown tool error"))
            if call.name in self.VERIFY_TOOLS and self.mutated:
                self.checks.append(
                    CheckEvidence(
                        command=str(call.parameters.get("command", "")),
                        success=False,
                        output=self.failure_message,
                    )
                )
            if self.failure_kind not in {FailureKind.RECOVERABLE_TOOL_ERROR}:
                self.transition(TaskState.FAILED)
            return

        metadata = result.get("metadata") or {}
        if call.name == "read_file":
            value = metadata.get("file_path") or call.parameters.get("path") or call.parameters.get("file_path")
            if value:
                self.inspected_files.add(self.canonical(Path(str(value)).expanduser()))
        elif call.name in self.MUTATION_TOOLS:
            value = metadata.get("file_path") or call.parameters.get("path") or call.parameters.get("file_path")
            if value:
                self.changed_files.add(self.canonical(Path(str(value)).expanduser()))
            self.diff_reviewed = bool(result.get("diff"))
            self.failure_kind = None
            self.failure_message = None
        elif call.name in self.DIFF_TOOLS and self.mutated:
            self.diff_reviewed = True
        elif call.name in self.VERIFY_TOOLS and self.mutated:
            command = str(call.parameters.get("command", ""))
            output = str(result.get("result", ""))
            self.checks.append(CheckEvidence(command=command, success=True, output=output))
            self.failure_kind = None
            self.failure_message = None

    def _deny(self, kind: FailureKind, message: str) -> dict[str, Any]:
        self.failure_kind = kind
        self.failure_message = message
        self.transition(TaskState.FAILED)
        return {"success": False, "result": None, "error": message, "error_type": kind.value}

    def can_succeed(self) -> bool:
        plan_complete = self.plan is None or all(action.status == "completed" for action in self.plan.actions)
        if not self.mutated:
            return plan_complete and (
                self.failure_kind is None or self.failure_kind == FailureKind.RECOVERABLE_TOOL_ERROR
            )
        return (
            plan_complete and self.diff_reviewed and bool(self.checks) and all(check.success for check in self.checks)
        )

    def final_report(self, model_summary: str) -> str:
        success = self.can_succeed()
        if success:
            self.transition(TaskState.SUMMARIZE)
        else:
            if self.failure_kind is None:
                self.failure_kind = FailureKind.INCOMPLETE_WORKFLOW
                missing = []
                if self.mutated and not self.diff_reviewed:
                    missing.append("diff review")
                if self.mutated and not self.checks:
                    missing.append("required verification")
                self.failure_message = "Missing " + " and ".join(missing or ["completion evidence"])
            self.transition(TaskState.FAILED)

        root = self.project_root
        changed = []
        for path in sorted(self.changed_files):
            try:
                changed.append(str(path.relative_to(root)))
            except ValueError:
                changed.append(str(path))
        changed_text = ", ".join(changed) if changed else "None"
        checks_text = (
            "; ".join(f"{check.command} ({'passed' if check.success else 'failed'})" for check in self.checks) or "None"
        )
        risks = "None identified" if success else (self.failure_message or "Task did not complete")
        plan_text = self.plan.progress if self.plan is not None else "Not required"
        return (
            f"Status: {'succeeded' if success else 'failed'}\n"
            f"Changed files: {changed_text}\n"
            f"Checks run: {checks_text}\n"
            f"Plan progress: {plan_text}\n"
            f"Outcome: {model_summary.strip() or 'No model summary provided.'}\n"
            f"Remaining risks: {risks}"
        )
