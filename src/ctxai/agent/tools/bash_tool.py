"""Policy-controlled command execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import AgentToolsConfig
from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from .execution import Capability, ToolExecutionContext, coerce_context
from .output_limits import truncate_text
from .sandbox import SandboxBackend, SandboxUnavailableError, select_backend


class BashTool(BaseTool):
    """Execute a single command without invoking a shell.

    Command policy is enforced in a fixed order: classification
    (``approve_command``), the optional exact-name allowlist, then — when a
    sandbox mode is configured and a backend exists — OS-sandboxed execution
    (HH-08). With ``sandbox: "off"`` (the default) the tool behaves exactly
    as it did before HH-08.
    """

    def __init__(
        self,
        config: AgentToolsConfig,
        working_directory: str | Path | None = None,
        *,
        context: ToolExecutionContext | None = None,
        sandbox_backends: list[SandboxBackend] | None = None,
    ):
        """Create the tool.

        Args:
            config: The agent tools configuration (policy and sandbox mode).
            working_directory: Default project root when no context is given.
            context: Shared execution context; constructed when omitted.
            sandbox_backends: Explicit backend preference list (tests);
                defaults to the platform backends from ``sandbox.py``.
        """
        super().__init__()
        self.config = config
        self.context = coerce_context(
            context,
            working_directory,
            timeout=config.bash_timeout,
            allow_outside_project=config.allow_outside_project,
        )
        # The config's opt-in passthrough names compose with whatever the
        # shared context already allows; values still come from os.environ.
        if config.env_passthrough:
            self.context.env_passthrough = sorted(set(self.context.env_passthrough) | set(config.env_passthrough))
        self.timeout = config.bash_timeout
        self._sandbox_backends = sandbox_backends

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Execute one policy-approved command inside the repository (no shell operators).",
            [
                ToolParameter("command", ToolParameterType.STRING, "Command and arguments"),
                ToolParameter(
                    "working_directory",
                    ToolParameterType.STRING,
                    "Contained working directory",
                    required=False,
                    default=".",
                ),
            ],
        )

    async def execute(self, command: str, working_directory: str = ".") -> dict[str, Any]:
        cwd: Path | None = None
        sandbox_backend: SandboxBackend | None = None
        sandbox_details: dict[str, Any] = {"sandbox": None}
        try:
            cwd = self.context.resolve_path(working_directory, must_exist=True)
            if not cwd.is_dir():
                raise ValueError(f"Not a directory: {working_directory}")
            # Resolve the sandbox backend first: in `required` mode a missing
            # backend fails the command closed before anything is classified
            # or executed (HH-08 acceptance criterion 1).
            sandbox_backend, sandbox_diagnostic = select_backend(self.config.sandbox, backends=self._sandbox_backends)
            sandbox_details["sandbox"] = sandbox_backend.name if sandbox_backend.enforces else None
            if sandbox_diagnostic:
                sandbox_details["sandbox_diagnostic"] = sandbox_diagnostic
            if self.config.sandbox == "required" and not sandbox_backend.enforces:
                raise SandboxUnavailableError(
                    f"sandbox mode is 'required' but {sandbox_diagnostic}; command not executed"
                )
            # An enforcing backend plus an explicit network opt-in satisfies
            # the NETWORK capability check — the sandbox is the enforcement
            # boundary. Without an enforcing backend the in-process policy
            # stays in charge (nothing is granted, default behavior preserved).
            if sandbox_backend.enforces and self.config.sandbox_network:
                self.context.capabilities.add(Capability.NETWORK)
            argv = self.context.approve_command(command)
            if self.config.bash_allowed_commands is not None:
                allowed = {Path(item).name for item in self.config.bash_allowed_commands}
                if Path(argv[0]).name not in allowed:
                    raise PermissionError(f"Command executable not allowlisted: {argv[0]}")
            # Wrap the classified argv (never before policy): a real backend
            # composes the OS sandbox here; NoopBackend returns argv unchanged,
            # which is byte-identical to the pre-HH-08 behavior. Network is
            # allowed inside the sandbox when the config opts in, when the
            # context carries Capability.NETWORK, or when the human approved
            # the network capability for exactly this command.
            exec_argv = list(argv)
            env = self.context.command_environment()
            if sandbox_backend.enforces:
                network = (
                    self.config.sandbox_network
                    or Capability.NETWORK in self.context.capabilities
                    or Capability.NETWORK in self.context.approved_capabilities
                )
                sandbox_details["sandbox_network"] = network
                exec_argv = sandbox_backend.wrap(argv, cwd, network=network)
                env = sandbox_backend.adjust_environment(env)
            try:
                process = await asyncio.create_subprocess_exec(
                    *exec_argv,
                    cwd=cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
                except TimeoutError:
                    process.kill()
                    await process.communicate()
                    raise TimeoutError(f"Command timed out after {self.timeout} seconds")
            finally:
                # Profile/temp cleanup happens after execution in every path
                # (success, timeout, spawn failure) — no artifacts are left.
                sandbox_backend.cleanup()
            raw_stdout = stdout.decode(errors="replace")
            raw_stderr = stderr.decode(errors="replace")
            stdout_text = truncate_text(raw_stdout, self.config.max_output_chars, label="stdout")
            stderr_text = truncate_text(raw_stderr, self.config.max_output_chars, label="stderr")
            stdout_truncated = len(raw_stdout) > self.config.max_output_chars
            stderr_truncated = len(raw_stderr) > self.config.max_output_chars
            success = process.returncode == 0
            record = self.context.record(
                tool=self.name,
                action="command",
                capability=Capability.COMMAND,
                target=command,
                success=success,
                details={
                    "cwd": str(cwd),
                    "exit_code": process.returncode,
                    "stdout_chars": len(raw_stdout),
                    "stderr_chars": len(raw_stderr),
                    "truncated": stdout_truncated or stderr_truncated,
                    **sandbox_details,
                },
            )
            metadata: dict[str, Any] = {
                "command": command,
                "working_directory": str(cwd),
                "exit_code": process.returncode,
                "stderr": stderr_text,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "original_stdout_chars": len(raw_stdout),
                "original_stderr_chars": len(raw_stderr),
                "audit": record.__dict__,
                "sandbox": sandbox_details["sandbox"],
            }
            if "sandbox_network" in sandbox_details:
                metadata["sandbox_network"] = sandbox_details["sandbox_network"]
            if "sandbox_diagnostic" in sandbox_details:
                metadata["sandbox_diagnostic"] = sandbox_details["sandbox_diagnostic"]
            return {
                "success": success,
                "result": stdout_text,
                "error": None if success else stderr_text or f"Exit code {process.returncode}",
                "metadata": metadata,
            }
        except Exception as exc:
            if sandbox_backend is not None:
                sandbox_backend.cleanup()
            self.context.record(
                tool=self.name,
                action="command",
                capability=Capability.COMMAND,
                target=command,
                success=False,
                details={"cwd": str(cwd) if cwd else working_directory, "error": str(exc), **sandbox_details},
            )
            return {"success": False, "result": None, "error": str(exc), "error_type": type(exc).__name__}
