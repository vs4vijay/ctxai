"""Deterministic task state and evidence tracking for verified agent runs."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .llm.base import ToolCall


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


ApprovalCallback = Callable[[ToolCall], bool]


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

    MUTATION_TOOLS = frozenset({"write_file", "edit_file"})
    INSPECTION_TOOLS = frozenset({"read_file", "semantic_search", "grep", "glob", "list_files"})
    DIFF_TOOLS = frozenset({"git_diff"})
    VERIFY_TOOLS = frozenset({"bash"})

    def __post_init__(self) -> None:
        self.project_root = Path(os.path.realpath(self.project_root))

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
        if call.name not in self.MUTATION_TOOLS:
            if call.name in self.INSPECTION_TOOLS:
                self.transition(TaskState.RETRIEVE)
            elif call.name in self.VERIFY_TOOLS:
                self.transition(TaskState.VERIFY)
            return None

        if planning_enabled:
            self.transition(TaskState.PLAN)
        target_value = call.parameters.get("path") or call.parameters.get("file_path")
        target = Path(str(target_value)).expanduser() if target_value else None
        if target is not None and not target.is_absolute():
            target = self.project_root / target
        target = self.canonical(target) if target is not None else None
        must_inspect = call.name == "edit_file" or bool(target and target.exists())
        if must_inspect and target not in self.inspected_files:
            return self._deny(
                FailureKind.INCOMPLETE_WORKFLOW,
                f"Workflow denied: read {target_value} successfully before editing it",
            )
        if require_approval:
            self.transition(TaskState.APPROVE)
            if approval_callback is None or not approval_callback(call):
                return self._deny(
                    FailureKind.APPROVAL_DENIAL,
                    f"Approval denied for {call.name}: {target_value}",
                )
        self.transition(TaskState.EXECUTE)
        return None

    def observe(self, call: ToolCall, result: dict[str, Any]) -> None:
        success = bool(result.get("success"))
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
        if not self.mutated:
            return self.failure_kind is None or self.failure_kind == FailureKind.RECOVERABLE_TOOL_ERROR
        return self.diff_reviewed and bool(self.checks) and all(check.success for check in self.checks)

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
        checks_text = "; ".join(
            f"{check.command} ({'passed' if check.success else 'failed'})" for check in self.checks
        ) or "None"
        risks = "None identified" if success else (self.failure_message or "Task did not complete")
        return (
            f"Status: {'succeeded' if success else 'failed'}\n"
            f"Changed files: {changed_text}\n"
            f"Checks run: {checks_text}\n"
            f"Outcome: {model_summary.strip() or 'No model summary provided.'}\n"
            f"Remaining risks: {risks}"
        )
