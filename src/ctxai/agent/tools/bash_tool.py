"""Policy-controlled command execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import AgentToolsConfig
from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from .execution import Capability, ToolExecutionContext, coerce_context
from .output_limits import truncate_text


class BashTool(BaseTool):
    """Execute a single command without invoking a shell."""

    def __init__(
        self,
        config: AgentToolsConfig,
        working_directory: str | Path | None = None,
        *,
        context: ToolExecutionContext | None = None,
    ):
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
        try:
            cwd = self.context.resolve_path(working_directory, must_exist=True)
            if not cwd.is_dir():
                raise ValueError(f"Not a directory: {working_directory}")
            argv = self.context.approve_command(command)
            if self.config.bash_allowed_commands is not None:
                allowed = {Path(item).name for item in self.config.bash_allowed_commands}
                if Path(argv[0]).name not in allowed:
                    raise PermissionError(f"Command executable not allowlisted: {argv[0]}")
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=self.context.command_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except TimeoutError:
                process.kill()
                await process.communicate()
                raise TimeoutError(f"Command timed out after {self.timeout} seconds")
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
                },
            )
            return {
                "success": success,
                "result": stdout_text,
                "error": None if success else stderr_text or f"Exit code {process.returncode}",
                "metadata": {
                    "command": command,
                    "working_directory": str(cwd),
                    "exit_code": process.returncode,
                    "stderr": stderr_text,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "original_stdout_chars": len(raw_stdout),
                    "original_stderr_chars": len(raw_stderr),
                    "audit": record.__dict__,
                },
            }
        except Exception as exc:
            self.context.record(
                tool=self.name,
                action="command",
                capability=Capability.COMMAND,
                target=command,
                success=False,
                details={"cwd": str(cwd) if cwd else working_directory, "error": str(exc)},
            )
            return {"success": False, "result": None, "error": str(exc), "error_type": type(exc).__name__}
