"""Repository-rooted git tools, split into read-only and gated mutations."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from .execution import Capability, ToolExecutionContext, coerce_context


class _GitTool(BaseTool):
    capability = Capability.READ

    def __init__(self, working_directory: str | Path | None = None, *, context: ToolExecutionContext | None = None):
        super().__init__()
        self.context = coerce_context(context, working_directory)

    def _repo(self, value: str) -> Path:
        repo = self.context.resolve_path(value, capability=self.capability, must_exist=True)
        if not repo.is_dir():
            raise ValueError(f"Not a directory: {value}")
        return repo

    def _run(self, repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=self.context.timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"git exited {result.returncode}")
        return result

    def _error(self, exc: Exception) -> dict[str, Any]:
        return {"success": False, "result": None, "error": str(exc), "error_type": type(exc).__name__}


class GitStatusTool(_GitTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Show repository status.",
            [ToolParameter("path", ToolParameterType.STRING, "Contained repository path", required=False, default=".")],
        )

    async def execute(self, path: str = ".") -> dict[str, Any]:
        try:
            repo = self._repo(path)
            result = self._run(repo, ["status", "--porcelain=v1", "--branch"])
            return {
                "success": True,
                "result": result.stdout or "Working tree clean",
                "error": None,
                "metadata": {"repo_path": str(repo)},
            }
        except Exception as exc:
            return self._error(exc)


class GitDiffTool(_GitTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Show contained repository diffs.",
            [
                ToolParameter(
                    "path", ToolParameterType.STRING, "Optional contained file path", required=False, default=""
                ),
                ToolParameter(
                    "staged", ToolParameterType.BOOLEAN, "Show staged changes", required=False, default=False
                ),
                ToolParameter(
                    "repo_path", ToolParameterType.STRING, "Contained repository path", required=False, default="."
                ),
            ],
        )

    async def execute(self, path: str = "", staged: bool = False, repo_path: str = ".") -> dict[str, Any]:
        try:
            repo = self._repo(repo_path)
            args = ["diff"] + (["--staged"] if staged else [])
            if path:
                target = self.context.resolve_path(repo / path, must_exist=False)
                args += ["--", str(target.relative_to(repo))]
            result = self._run(repo, args)
            lines = result.stdout.splitlines()
            return {
                "success": True,
                "result": result.stdout or "No differences",
                "error": None,
                "metadata": {
                    "additions": sum(line.startswith("+") and not line.startswith("+++") for line in lines),
                    "deletions": sum(line.startswith("-") and not line.startswith("---") for line in lines),
                    "repo_path": str(repo),
                },
            }
        except Exception as exc:
            return self._error(exc)


class GitLogTool(_GitTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Show recent commit history.",
            [
                ToolParameter("limit", ToolParameterType.INTEGER, "Maximum commits", required=False, default=10),
                ToolParameter(
                    "repo_path", ToolParameterType.STRING, "Contained repository path", required=False, default="."
                ),
            ],
        )

    async def execute(self, limit: int = 10, repo_path: str = ".") -> dict[str, Any]:
        try:
            if not 1 <= limit <= 100:
                raise ValueError("limit must be between 1 and 100")
            repo = self._repo(repo_path)
            result = self._run(repo, ["log", f"-{limit}", "--oneline", "--decorate"])
            return {
                "success": True,
                "result": result.stdout,
                "error": None,
                "metadata": {"limit": limit, "repo_path": str(repo)},
            }
        except Exception as exc:
            return self._error(exc)


class _GatedGitMutation(_GitTool):
    capability = Capability.DESTRUCTIVE

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        action = self.name.removeprefix("git_")
        target = str(kwargs.get("repo_path", "."))
        try:
            self.context.require(Capability.DESTRUCTIVE, action, target)
            return {
                "success": False,
                "result": None,
                "error": f"{self.name} requires a dedicated approved implementation",
                "error_type": "PolicyDenied",
            }
        except Exception as exc:
            return self._error(exc)


class GitCommitTool(_GatedGitMutation):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Gated git commit operation.",
            [ToolParameter("message", ToolParameterType.STRING, "Commit message")],
        )


class GitAddTool(_GatedGitMutation):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Gated git staging operation.",
            [ToolParameter("files", ToolParameterType.ARRAY, "Files to stage")],
        )


class GitBranchTool(_GatedGitMutation):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Gated branch mutation operation.",
            [ToolParameter("branch_name", ToolParameterType.STRING, "Branch name")],
        )
