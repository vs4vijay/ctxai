"""Repository-rooted file operation tools."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema
from .execution import Capability, ToolExecutionContext, coerce_context


def _failure(exc: Exception) -> dict[str, Any]:
    return {"success": False, "result": None, "error": str(exc), "error_type": type(exc).__name__}


def _diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
    )


class ContextualFileTool(BaseTool):
    def __init__(
        self,
        working_directory: str | Path | None = None,
        *,
        context: ToolExecutionContext | None = None,
    ) -> None:
        super().__init__()
        self.context = coerce_context(context, working_directory)


class ReadFileTool(ContextualFileTool):
    def __init__(
        self,
        max_file_size_mb: int = 10,
        working_directory: str | Path | None = None,
        *,
        context: ToolExecutionContext | None = None,
    ):
        super().__init__(working_directory, context=context)
        self.max_file_size_mb = max_file_size_mb

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Read a repository file with line numbers.",
            [
                ToolParameter(
                    "file_path",
                    ToolParameterType.STRING,
                    "Repository-relative or contained absolute path",
                    required=False,
                ),
                ToolParameter("path", ToolParameterType.STRING, "Legacy alias for file_path", required=False),
                ToolParameter("start_line", ToolParameterType.INTEGER, "First line (1-indexed)", required=False),
                ToolParameter("end_line", ToolParameterType.INTEGER, "Last line (inclusive)", required=False),
            ],
        )

    async def execute(
        self,
        file_path: str | None = None,
        path: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        value = file_path or path
        if not value:
            return _failure(ValueError("file_path is required"))
        try:
            resolved = self.context.resolve_path(value, must_exist=True)
            if not resolved.is_file():
                raise ValueError(f"Not a file: {value}")
            size = resolved.stat().st_size
            if size > self.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File too large: {size / 1024 / 1024:.2f}MB (max: {self.max_file_size_mb}MB)")
            lines = resolved.read_text(encoding="utf-8", errors="replace").split("\n")
            if start_line is not None and start_line < 1:
                raise ValueError("start_line must be at least 1")
            if end_line is not None and start_line is not None and end_line < start_line:
                raise ValueError("end_line must not be before start_line")
            start = (start_line or 1) - 1
            selected = lines[start:end_line]
            return {
                "success": True,
                "result": "\n".join(f"{i:4d} | {line}" for i, line in enumerate(selected, start=start + 1)),
                "error": None,
                "metadata": {
                    "file_path": str(resolved),
                    "total_lines": len(lines),
                    "returned_lines": len(selected),
                    "size_bytes": size,
                },
            }
        except Exception as exc:
            return _failure(exc)


class WriteFileTool(ContextualFileTool):
    def __init__(
        self,
        allow_overwrite: bool = True,
        working_directory: str | Path | None = None,
        *,
        context: ToolExecutionContext | None = None,
    ):
        super().__init__(working_directory, context=context)
        self.allow_overwrite = allow_overwrite

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Create or overwrite a repository file and return a unified diff.",
            [
                ToolParameter(
                    "file_path",
                    ToolParameterType.STRING,
                    "Repository-relative or contained absolute path",
                    required=False,
                ),
                ToolParameter("path", ToolParameterType.STRING, "Legacy alias for file_path", required=False),
                ToolParameter("content", ToolParameterType.STRING, "Complete new contents"),
            ],
        )

    async def execute(self, content: str, file_path: str | None = None, path: str | None = None) -> dict[str, Any]:
        value = file_path or path
        if not value:
            return _failure(ValueError("file_path is required"))
        resolved: Path | None = None
        try:
            resolved = self.context.resolve_path(value, capability=Capability.WORKSPACE_WRITE)
            existed = resolved.exists()
            if existed and not resolved.is_file():
                raise ValueError(f"Not a file: {value}")
            if existed and not self.allow_overwrite:
                raise FileExistsError(f"File exists and overwrite not allowed: {value}")
            before = resolved.read_text(encoding="utf-8", errors="replace") if existed else ""
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            difference = _diff(resolved, before, content)
            record = self.context.record(
                tool=self.name,
                action="write",
                capability=Capability.WORKSPACE_WRITE,
                target=str(resolved),
                success=True,
                details={"created": not existed, "diff": difference},
            )
            return {
                "success": True,
                "result": f"Wrote {len(content.encode())} bytes to {resolved}",
                "error": None,
                "diff": difference,
                "metadata": {
                    "file_path": str(resolved),
                    "created": not existed,
                    "size_bytes": len(content.encode()),
                    "audit": record.__dict__,
                },
            }
        except Exception as exc:
            self.context.record(
                tool=self.name,
                action="write",
                capability=Capability.WORKSPACE_WRITE,
                target=str(resolved or value),
                success=False,
                details={"error": str(exc)},
            )
            return _failure(exc)


class EditFileTool(ContextualFileTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Replace text in a repository file and return a unified diff.",
            [
                ToolParameter(
                    "file_path",
                    ToolParameterType.STRING,
                    "Repository-relative or contained absolute path",
                    required=False,
                ),
                ToolParameter("path", ToolParameterType.STRING, "Legacy alias for file_path", required=False),
                ToolParameter("old_text", ToolParameterType.STRING, "Text or regex to replace"),
                ToolParameter("new_text", ToolParameterType.STRING, "Replacement text"),
                ToolParameter(
                    "use_regex",
                    ToolParameterType.BOOLEAN,
                    "Use regular expression matching",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(
        self,
        old_text: str,
        new_text: str,
        file_path: str | None = None,
        path: str | None = None,
        use_regex: bool = False,
    ) -> dict[str, Any]:
        value = file_path or path
        if not value:
            return _failure(ValueError("file_path is required"))
        resolved: Path | None = None
        try:
            resolved = self.context.resolve_path(value, capability=Capability.WORKSPACE_WRITE, must_exist=True)
            if not resolved.is_file():
                raise ValueError(f"Not a file: {value}")
            before = resolved.read_text(encoding="utf-8")
            if use_regex:
                after, count = re.subn(old_text, new_text, before)
            else:
                count = before.count(old_text)
                after = before.replace(old_text, new_text)
            if count == 0:
                raise ValueError("Pattern not found in file")
            resolved.write_text(after, encoding="utf-8")
            difference = _diff(resolved, before, after)
            record = self.context.record(
                tool=self.name,
                action="edit",
                capability=Capability.WORKSPACE_WRITE,
                target=str(resolved),
                success=True,
                details={"replacements": count, "diff": difference},
            )
            return {
                "success": True,
                "result": f"Replaced {count} occurrence(s) in {resolved}",
                "error": None,
                "diff": difference,
                "metadata": {"file_path": str(resolved), "replacements": count, "audit": record.__dict__},
            }
        except Exception as exc:
            self.context.record(
                tool=self.name,
                action="edit",
                capability=Capability.WORKSPACE_WRITE,
                target=str(resolved or value),
                success=False,
                details={"error": str(exc)},
            )
            return _failure(exc)


class ListFilesTool(ContextualFileTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "List a contained repository directory.",
            [
                ToolParameter("directory_path", ToolParameterType.STRING, "Directory path", required=False),
                ToolParameter("directory", ToolParameterType.STRING, "Legacy alias for directory_path", required=False),
                ToolParameter(
                    "show_hidden", ToolParameterType.BOOLEAN, "Include dotfiles", required=False, default=False
                ),
            ],
        )

    async def execute(
        self, directory_path: str | None = None, directory: str | None = None, show_hidden: bool = False
    ) -> dict[str, Any]:
        value = directory_path or directory or "."
        try:
            resolved = self.context.resolve_path(value, must_exist=True)
            if not resolved.is_dir():
                raise ValueError(f"Not a directory: {value}")
            entries = [item for item in sorted(resolved.iterdir()) if show_hidden or not item.name.startswith(".")]
            result = (
                "\n".join(f"{'dir' if item.is_dir() else 'file'} {item.name}" for item in entries)
                or "(empty directory)"
            )
            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {"directory_path": str(resolved), "total_entries": len(entries)},
            }
        except Exception as exc:
            return _failure(exc)


class GlobTool(ContextualFileTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Find repository files matching a glob.",
            [
                ToolParameter("pattern", ToolParameterType.STRING, "Glob pattern"),
                ToolParameter(
                    "base_path", ToolParameterType.STRING, "Contained base directory", required=False, default="."
                ),
                ToolParameter("max_results", ToolParameterType.INTEGER, "Maximum results", required=False, default=100),
            ],
        )

    async def execute(self, pattern: str, base_path: str = ".", max_results: int = 100) -> dict[str, Any]:
        try:
            base = self.context.resolve_path(base_path, must_exist=True)
            matches: list[str] = []
            for match in base.glob(pattern):
                safe = self.context.resolve_path(match, must_exist=True)
                matches.append(str(safe.relative_to(base)))
                if len(matches) >= max_results:
                    break
            return {
                "success": True,
                "result": "\n".join(matches) or "(no matches found)",
                "error": None,
                "metadata": {
                    "pattern": pattern,
                    "base_path": str(base),
                    "matches": len(matches),
                    "truncated": len(matches) >= max_results,
                },
            }
        except Exception as exc:
            return _failure(exc)


class GrepTool(ContextualFileTool):
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            "Regex-search contained repository files.",
            [
                ToolParameter("pattern", ToolParameterType.STRING, "Regex pattern"),
                ToolParameter("file_pattern", ToolParameterType.STRING, "File glob"),
                ToolParameter(
                    "base_path", ToolParameterType.STRING, "Contained base directory", required=False, default="."
                ),
                ToolParameter(
                    "case_insensitive", ToolParameterType.BOOLEAN, "Ignore case", required=False, default=False
                ),
                ToolParameter("max_results", ToolParameterType.INTEGER, "Maximum matches", required=False, default=50),
            ],
        )

    async def execute(
        self,
        pattern: str,
        file_pattern: str,
        base_path: str = ".",
        case_insensitive: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        try:
            base = self.context.resolve_path(base_path, must_exist=True)
            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
            matches: list[str] = []
            files_searched = 0
            for candidate in base.glob(file_pattern):
                path = self.context.resolve_path(candidate, must_exist=True)
                if not path.is_file():
                    continue
                files_searched += 1
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{path.relative_to(base)}:{number}: {line}")
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break
            return {
                "success": True,
                "result": "\n".join(matches) or "(no matches found)",
                "error": None,
                "metadata": {"matches": len(matches), "files_searched": files_searched, "base_path": str(base)},
            }
        except Exception as exc:
            return _failure(exc)
