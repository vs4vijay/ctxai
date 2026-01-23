"""
File operation tools for agent.

Includes: read, write, edit, list, glob, grep
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import re
import glob as glob_module

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from .base import BaseTool, ToolSchema, ToolParameter, ToolParameterType


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""

    def __init__(self, max_file_size_mb: int = 10):
        super().__init__()
        self.max_file_size_mb = max_file_size_mb

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Read the contents of a file. Returns file contents with line numbers.",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type=ToolParameterType.STRING,
                    description="Path to the file to read (absolute or relative)",
                    required=True
                ),
                ToolParameter(
                    name="start_line",
                    type=ToolParameterType.INTEGER,
                    description="Starting line number (1-indexed, optional)",
                    required=False
                ),
                ToolParameter(
                    name="end_line",
                    type=ToolParameterType.INTEGER,
                    description="Ending line number (inclusive, optional)",
                    required=False
                ),
            ]
        )

    async def execute(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict[str, Any]:
        try:
            path = Path(file_path).resolve()

            if not path.exists():
                return {"success": False, "result": None, "error": f"File not found: {file_path}"}

            if not path.is_file():
                return {"success": False, "result": None, "error": f"Not a file: {file_path}"}

            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                return {
                    "success": False,
                    "result": None,
                    "error": f"File too large: {size_mb:.2f}MB (max: {self.max_file_size_mb}MB)"
                }

            # Read file
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = await f.read()
            else:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

            lines = content.split('\n')

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]
                start_num = start_line if start_line else 1
            else:
                start_num = 1

            # Format with line numbers
            numbered_lines = []
            for i, line in enumerate(lines, start=start_num):
                numbered_lines.append(f"{i:4d} | {line}")

            result = "\n".join(numbered_lines)

            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {
                    "file_path": str(path),
                    "total_lines": len(lines),
                    "size_bytes": path.stat().st_size,
                }
            }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}


class WriteFileTool(BaseTool):
    """Tool for writing/creating files."""

    def __init__(self, allow_overwrite: bool = True):
        super().__init__()
        self.allow_overwrite = allow_overwrite

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Write content to a file. Creates the file if it doesn't exist, or overwrites if it does.",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type=ToolParameterType.STRING,
                    description="Path to the file to write",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type=ToolParameterType.STRING,
                    description="Content to write to the file",
                    required=True
                ),
            ]
        )

    async def execute(self, file_path: str, content: str) -> Dict[str, Any]:
        try:
            path = Path(file_path).resolve()

            # Check if file exists and overwrite not allowed
            if path.exists() and not self.allow_overwrite:
                return {
                    "success": False,
                    "result": None,
                    "error": f"File exists and overwrite not allowed: {file_path}"
                }

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)

            lines = content.count('\n') + 1
            size = len(content.encode('utf-8'))

            return {
                "success": True,
                "result": f"Wrote {lines} lines ({size} bytes) to {path}",
                "error": None,
                "metadata": {
                    "file_path": str(path),
                    "lines": lines,
                    "size_bytes": size,
                    "created": not path.exists(),
                }
            }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}


class EditFileTool(BaseTool):
    """Tool for editing files with search/replace."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Edit a file by searching for old text and replacing with new text. Supports exact match or regex.",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type=ToolParameterType.STRING,
                    description="Path to the file to edit",
                    required=True
                ),
                ToolParameter(
                    name="old_text",
                    type=ToolParameterType.STRING,
                    description="Text to search for (exact match or regex pattern)",
                    required=True
                ),
                ToolParameter(
                    name="new_text",
                    type=ToolParameterType.STRING,
                    description="Text to replace with",
                    required=True
                ),
                ToolParameter(
                    name="use_regex",
                    type=ToolParameterType.BOOLEAN,
                    description="Whether to use regex for matching (default: false)",
                    required=False,
                    default=False
                ),
            ]
        )

    async def execute(self, file_path: str, old_text: str, new_text: str, use_regex: bool = False) -> Dict[str, Any]:
        try:
            path = Path(file_path).resolve()

            if not path.exists():
                return {"success": False, "result": None, "error": f"File not found: {file_path}"}

            # Read file
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Perform replacement
            if use_regex:
                new_content = re.sub(old_text, new_text, content)
                count = len(re.findall(old_text, content))
            else:
                count = content.count(old_text)
                new_content = content.replace(old_text, new_text)

            if count == 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Pattern not found in file: {old_text}"
                }

            # Write back
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                    await f.write(new_content)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

            return {
                "success": True,
                "result": f"Replaced {count} occurrence(s) in {path}",
                "error": None,
                "metadata": {
                    "file_path": str(path),
                    "replacements": count,
                    "use_regex": use_regex,
                }
            }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}


class ListFilesTool(BaseTool):
    """Tool for listing directory contents."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="List files and directories in a directory.",
            parameters=[
                ToolParameter(
                    name="directory_path",
                    type=ToolParameterType.STRING,
                    description="Path to the directory to list",
                    required=True
                ),
                ToolParameter(
                    name="show_hidden",
                    type=ToolParameterType.BOOLEAN,
                    description="Whether to show hidden files (starting with .)",
                    required=False,
                    default=False
                ),
            ]
        )

    async def execute(self, directory_path: str, show_hidden: bool = False) -> Dict[str, Any]:
        try:
            path = Path(directory_path).resolve()

            if not path.exists():
                return {"success": False, "result": None, "error": f"Directory not found: {directory_path}"}

            if not path.is_dir():
                return {"success": False, "result": None, "error": f"Not a directory: {directory_path}"}

            # List contents
            entries = []
            for item in sorted(path.iterdir()):
                if not show_hidden and item.name.startswith('.'):
                    continue

                size = item.stat().st_size if item.is_file() else 0
                entry_type = "dir" if item.is_dir() else "file"

                entries.append({
                    "name": item.name,
                    "type": entry_type,
                    "size": size,
                    "path": str(item),
                })

            # Format result
            lines = []
            for entry in entries:
                icon = "📁" if entry["type"] == "dir" else "📄"
                size_str = f"{entry['size']:,} bytes" if entry["type"] == "file" else ""
                lines.append(f"{icon} {entry['name']} {size_str}")

            result = "\n".join(lines) if lines else "(empty directory)"

            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {
                    "directory_path": str(path),
                    "total_entries": len(entries),
                    "files": sum(1 for e in entries if e["type"] == "file"),
                    "directories": sum(1 for e in entries if e["type"] == "dir"),
                }
            }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}


class GlobTool(BaseTool):
    """Tool for finding files matching patterns."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Find files matching a glob pattern (e.g., '**/*.py' for all Python files).",
            parameters=[
                ToolParameter(
                    name="pattern",
                    type=ToolParameterType.STRING,
                    description="Glob pattern to match (e.g., '*.py', '**/*.js', 'src/**/*.ts')",
                    required=True
                ),
                ToolParameter(
                    name="base_path",
                    type=ToolParameterType.STRING,
                    description="Base directory to search from (default: current directory)",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="max_results",
                    type=ToolParameterType.INTEGER,
                    description="Maximum number of results to return (default: 100)",
                    required=False,
                    default=100
                ),
            ]
        )

    async def execute(self, pattern: str, base_path: str = ".", max_results: int = 100) -> Dict[str, Any]:
        try:
            base = Path(base_path).resolve()

            if not base.exists():
                return {"success": False, "result": None, "error": f"Base path not found: {base_path}"}

            # Use glob to find matches
            matches = []
            for match in base.glob(pattern):
                matches.append(str(match.relative_to(base)))
                if len(matches) >= max_results:
                    break

            result = "\n".join(matches) if matches else "(no matches found)"

            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {
                    "pattern": pattern,
                    "base_path": str(base),
                    "matches": len(matches),
                    "truncated": len(matches) >= max_results,
                }
            }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}


class GrepTool(BaseTool):
    """Tool for searching file contents with regex."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Search for text patterns in files using regex. Returns matching lines with context.",
            parameters=[
                ToolParameter(
                    name="pattern",
                    type=ToolParameterType.STRING,
                    description="Regex pattern to search for",
                    required=True
                ),
                ToolParameter(
                    name="file_pattern",
                    type=ToolParameterType.STRING,
                    description="Glob pattern for files to search (e.g., '**/*.py')",
                    required=True
                ),
                ToolParameter(
                    name="base_path",
                    type=ToolParameterType.STRING,
                    description="Base directory to search from (default: current directory)",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="case_insensitive",
                    type=ToolParameterType.BOOLEAN,
                    description="Whether to perform case-insensitive search (default: false)",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="max_results",
                    type=ToolParameterType.INTEGER,
                    description="Maximum number of matches to return (default: 50)",
                    required=False,
                    default=50
                ),
            ]
        )

    async def execute(
        self,
        pattern: str,
        file_pattern: str,
        base_path: str = ".",
        case_insensitive: bool = False,
        max_results: int = 50
    ) -> Dict[str, Any]:
        try:
            base = Path(base_path).resolve()

            if not base.exists():
                return {"success": False, "result": None, "error": f"Base path not found: {base_path}"}

            # Compile regex
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)

            # Find files and search
            matches = []
            files_searched = 0

            for file_path in base.glob(file_pattern):
                if not file_path.is_file():
                    continue

                files_searched += 1

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append({
                                    "file": str(file_path.relative_to(base)),
                                    "line": line_num,
                                    "content": line.rstrip(),
                                })

                                if len(matches) >= max_results:
                                    break

                except Exception:
                    # Skip files that can't be read
                    pass

                if len(matches) >= max_results:
                    break

            # Format results
            lines = []
            for match in matches:
                lines.append(f"{match['file']}:{match['line']}: {match['content']}")

            result = "\n".join(lines) if lines else "(no matches found)"

            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {
                    "pattern": pattern,
                    "file_pattern": file_pattern,
                    "base_path": str(base),
                    "matches": len(matches),
                    "files_searched": files_searched,
                    "truncated": len(matches) >= max_results,
                }
            }

        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}
