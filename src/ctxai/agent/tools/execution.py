"""Shared safety policy and audit context for agent tools."""

from __future__ import annotations

import os
import shlex
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class Capability(str, Enum):
    READ = "read"
    WORKSPACE_WRITE = "workspace_write"
    COMMAND = "command"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"
    OUTSIDE_PROJECT = "outside_project"


class PolicyDenied(PermissionError):
    """Raised when a requested operation is outside the execution policy."""


ALLOWED_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SHELL",
    "TERM",
    "USER",
    "LOGNAME",
)
"""The only variables subprocesses inherit from ``os.environ`` (HH-01).

``os.environ`` is never passed wholesale: every other name reaches a
subprocess only through :attr:`ToolExecutionContext.env_passthrough` (opt-in,
values still sourced from ``os.environ``) or :attr:`ToolExecutionContext.environment`
(explicit values set by the caller).
"""


@dataclass(frozen=True)
class AuditRecord:
    request_id: str
    timestamp: str
    tool: str
    action: str
    capability: str
    target: str
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


ApprovalCallback = Callable[[Capability, str, str], bool]


@dataclass
class ToolExecutionContext:
    """Repository-rooted permissions and in-memory mutation audit log."""

    project_root: Path
    capabilities: set[Capability] = field(
        default_factory=lambda: {
            Capability.READ,
            Capability.WORKSPACE_WRITE,
            Capability.COMMAND,
        }
    )
    environment: dict[str, str] = field(default_factory=dict)
    env_passthrough: list[str] = field(default_factory=list)
    timeout: int = 30
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    approval_callback: ApprovalCallback | None = None
    audit_log: list[AuditRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve(strict=True)
        if not self.project_root.is_dir():
            raise ValueError(f"Project root is not a directory: {self.project_root}")

    @classmethod
    def for_project(
        cls,
        project_root: str | Path,
        *,
        allow_outside_project: bool = False,
        timeout: int = 30,
        env_passthrough: list[str] | None = None,
    ) -> ToolExecutionContext:
        capabilities = {Capability.READ, Capability.WORKSPACE_WRITE, Capability.COMMAND}
        if allow_outside_project:
            capabilities.add(Capability.OUTSIDE_PROJECT)
        return cls(
            Path(project_root),
            capabilities=capabilities,
            timeout=timeout,
            env_passthrough=list(env_passthrough or []),
        )

    def require(self, capability: Capability, action: str, target: str) -> None:
        if capability in self.capabilities:
            return
        if self.approval_callback and self.approval_callback(capability, action, target):
            return
        raise PolicyDenied(f"Capability denied: {capability.value} ({action}: {target})")

    def resolve_path(
        self,
        value: str | Path,
        *,
        capability: Capability = Capability.READ,
        must_exist: bool = False,
    ) -> Path:
        self.require(capability, "path_access", str(value))
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            self.require(Capability.OUTSIDE_PROJECT, "outside_project", str(resolved))
        return resolved

    def command_environment(self) -> dict[str, str]:
        """Build the subprocess environment from an explicit allowlist.

        Only :data:`ALLOWED_ENVIRONMENT_KEYS` are copied from ``os.environ``
        (secrets are never inherited wholesale), plus the opt-in names in
        :attr:`env_passthrough` (values still sourced from ``os.environ`` when
        present), plus :attr:`environment` which wins on conflicts.

        Returns:
            The environment mapping for subprocess execution.
        """
        env = {name: value for name in ALLOWED_ENVIRONMENT_KEYS if (value := os.environ.get(name)) is not None}
        for name in self.env_passthrough:
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
        env.update(self.environment)
        return env

    def approve_command(self, command: str) -> list[str]:
        """Parse one simple command and classify risky or unsupported shell syntax."""
        self.require(Capability.COMMAND, "execute", command)
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise PolicyDenied(f"Invalid command syntax: {exc}") from exc
        if not argv:
            raise PolicyDenied("Empty command")
        shell_operators = {";", "&&", "||", "|", ">", ">>", "<", "&"}
        if any(token in shell_operators for token in argv):
            raise PolicyDenied("Shell operators are not permitted; execute one command at a time")
        executable = Path(argv[0]).name.lower()
        destructive = {
            "chmod",
            "chown",
            "cp",
            "dd",
            "install",
            "kill",
            "mkfs",
            "mv",
            "pkill",
            "reboot",
            "rm",
            "rmdir",
            "shutdown",
            "tee",
            "truncate",
            "unlink",
        }
        network = {"curl", "wget", "ssh", "scp", "nc", "ncat", "telnet"}
        if executable in destructive:
            self.require(Capability.DESTRUCTIVE, "execute", command)
        if executable in network:
            self.require(Capability.NETWORK, "execute", command)
        if executable in {"bash", "dash", "sh", "zsh"}:
            self.require(Capability.DESTRUCTIVE, "shell", command)
        if executable in {"python", "python3", "node", "ruby", "perl"} and any(
            flag in argv[1:] for flag in {"-c", "-e"}
        ):
            self.require(Capability.DESTRUCTIVE, "inline_code", command)
        if executable == "git" and len(argv) > 1:
            read_only_git = {"diff", "grep", "log", "rev-parse", "show", "status"}
            if argv[1] not in read_only_git:
                self.require(Capability.DESTRUCTIVE, "git_mutation", command)
        return argv

    def record(
        self,
        *,
        tool: str,
        action: str,
        capability: Capability,
        target: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            request_id=self.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool=tool,
            action=action,
            capability=capability.value,
            target=target,
            success=success,
            details=details or {},
        )
        self.audit_log.append(record)
        return record


def coerce_context(
    context: ToolExecutionContext | None,
    working_directory: str | Path | None,
    *,
    timeout: int = 30,
    allow_outside_project: bool = False,
) -> ToolExecutionContext:
    if context is not None:
        return context
    return ToolExecutionContext.for_project(
        working_directory or Path.cwd(),
        allow_outside_project=allow_outside_project,
        timeout=timeout,
    )
