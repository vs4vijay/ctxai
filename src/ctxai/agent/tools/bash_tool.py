"""
Bash command execution tool.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from ..config import AgentToolsConfig
from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    def __init__(self, config: AgentToolsConfig):
        super().__init__()
        self.config = config
        self.timeout = config.bash_timeout

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Execute a bash command and return output. Use for scripts, packages, git operations, etc.",
            parameters=[
                ToolParameter(
                    name="command",
                    type=ToolParameterType.STRING,
                    description="The bash command to execute",
                    required=True
                ),
                ToolParameter(
                    name="working_directory",
                    type=ToolParameterType.STRING,
                    description="Working directory for command execution (default: current directory)",
                    required=False,
                    default="."
                ),
            ]
        )

    async def execute(self, command: str, working_directory: str = ".") -> dict[str, Any]:
        try:
            # Check if command is allowed
            if not self.config.is_bash_command_allowed(command):
                return {
                    "success": False,
                    "result": None,
                    "error": f"Command blocked for safety: {command}",
                }

            # Resolve working directory
            cwd = Path(working_directory).resolve()
            if not cwd.exists():
                return {
                    "success": False,
                    "result": None,
                    "error": f"Working directory not found: {working_directory}",
                }

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd)
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )

                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')

                output = ""
                if stdout_text:
                    output += f"STDOUT:\n{stdout_text}"
                if stderr_text:
                    if output:
                        output += "\n\n"
                    output += f"STDERR:\n{stderr_text}"

                success = process.returncode == 0

                return {
                    "success": success,
                    "result": output or "(no output)",
                    "error": None if success else f"Command failed with exit code {process.returncode}",
                    "metadata": {
                        "command": command,
                        "working_directory": str(cwd),
                        "exit_code": process.returncode,
                    }
                }

            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "result": None,
                    "error": f"Command timed out after {self.timeout} seconds",
                }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}
