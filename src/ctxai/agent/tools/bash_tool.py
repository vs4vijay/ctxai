"""Policy-controlled command execution tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import AgentToolsConfig
from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from .execution import Capability, ToolExecutionContext, coerce_context


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
            output = stdout.decode(errors="replace")
            error_output = stderr.decode(errors="replace")
            success = process.returncode == 0
            record = self.context.record(
                tool=self.name,
                action="command",
                capability=Capability.COMMAND,
                target=command,
                success=success,
                details={"cwd": str(cwd), "exit_code": process.returncode},
            )
            return {
                "success": success,
                "result": output,
                "error": None if success else error_output or f"Exit code {process.returncode}",
                "metadata": {
                    "command": command,
                    "working_directory": str(cwd),
                    "exit_code": process.returncode,
                    "stderr": error_output,
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
