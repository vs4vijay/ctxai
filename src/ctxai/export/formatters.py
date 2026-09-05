"""Format-specific renderers for exported repository content."""

from __future__ import annotations

import html as html_module
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".sh": "bash",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sql": "sql",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
}


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_EXT.get(path.suffix.lower(), "text")


@dataclass
class FileEntry:
    path: Path
    relative: str
    content: str
    size: int
    lines: int
    language: str
    last_modified: str


@dataclass
class RepositorySummary:
    name: str
    root: Path
    generated_at: str
    files: list[FileEntry]
    total_size: int
    total_lines: int
    languages: dict[str, dict[str, int]]
    truncated: bool

    @classmethod
    def build(cls, root: Path, files: Iterable[FileEntry], truncated: bool) -> RepositorySummary:
        files = list(files)
        languages: dict[str, dict[str, int]] = {}
        total_size = 0
        total_lines = 0
        for f in files:
            languages.setdefault(f.language, {"files": 0, "lines": 0})
            languages[f.language]["files"] += 1
            languages[f.language]["lines"] += f.lines
            total_size += f.size
            total_lines += f.lines
        return cls(
            name=root.name,
            root=root,
            generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            files=files,
            total_size=total_size,
            total_lines=total_lines,
            languages=languages,
            truncated=truncated,
        )


def render_text(summary: RepositorySummary) -> str:
    out: list[str] = []
    out.append(f"Repository: {summary.name}")
    out.append(f"Generated: {summary.generated_at}")
    out.append(f"Files: {len(summary.files)}")
    out.append(f"Total size: {summary.total_size} bytes")
    out.append("")
    for idx, f in enumerate(summary.files, 1):
        out.append(f"--- File {idx}/{len(summary.files)}: {f.relative} ---")
        out.append(f"Language: {f.language} | Lines: {f.lines} | Size: {f.size}")
        out.append("")
        out.append(f.content)
        out.append("")
    return "\n".join(out)


def render_markdown(summary: RepositorySummary, tree: str | None = None) -> str:
    out: list[str] = []
    out.append(f"# Repository Export: {summary.name}")
    out.append("")
    out.append(f"- Generated: {summary.generated_at}")
    out.append(f"- Files: {len(summary.files)}")
    out.append(f"- Total size: {summary.total_size} bytes")
    out.append(f"- Total lines: {summary.total_lines}")
    if summary.truncated:
        out.append("- ⚠ Output truncated by configured limits")
    out.append("")
    out.append("## Languages")
    for lang, stats in sorted(summary.languages.items()):
        out.append(f"- **{lang}** — {stats['files']} files, {stats['lines']} lines")
    if tree:
        out.append("")
        out.append("## Directory Structure")
        out.append("")
        out.append("```")
        out.append(tree)
        out.append("```")
    out.append("")
    out.append("## Files")
    for idx, f in enumerate(summary.files, 1):
        out.append("")
        out.append(f"### File {idx}/{len(summary.files)}: `{f.relative}`")
        out.append("")
        out.append(f"- **Language:** {f.language}")
        out.append(f"- **Lines:** {f.lines}")
        out.append(f"- **Size:** {f.size} bytes")
        out.append(f"- **Last modified:** {f.last_modified}")
        out.append("")
        out.append(f"```{f.language}")
        out.append(f.content)
        out.append("```")
    return "\n".join(out)


def render_xml(summary: RepositorySummary) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<repository>"]
    out.append(f"  <name>{html_module.escape(summary.name)}</name>")
    out.append(f"  <generated_at>{summary.generated_at}</generated_at>")
    out.append(f"  <total_files>{len(summary.files)}</total_files>")
    out.append("  <files>")
    for f in summary.files:
        out.append("    <file>")
        out.append(f"      <path>{html_module.escape(f.relative)}</path>")
        out.append(f"      <language>{f.language}</language>")
        out.append(f"      <lines>{f.lines}</lines>")
        out.append(f"      <size>{f.size}</size>")
        out.append(f"      <content><![CDATA[{f.content.replace(']]>', ']]]]><![CDATA[>')}]]></content>")
        out.append("    </file>")
    out.append("  </files>")
    out.append("</repository>")
    return "\n".join(out)


def render_tree(root: Path, files: Iterable[FileEntry]) -> str:
    """Render a simple directory tree from the selected files."""
    tree: dict = {}
    for f in files:
        cur = tree
        parts = f.relative.replace("\\", "/").split("/")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur.setdefault("__files__", []).append(parts[-1])

    lines: list[str] = [root.name + "/"]

    def _walk(node: dict, prefix: str) -> None:
        entries = [(k, v) for k, v in node.items() if k != "__files__"]
        entries.sort()
        for i, (name, sub) in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 and not node.get("__files__") else "├── "
            lines.append(f"{prefix}{connector}{name}/")
            extension = "    " if i == len(entries) - 1 and not node.get("__files__") else "│   "
            _walk(sub, prefix + extension)
        for j, fname in enumerate(node.get("__files__", []) or []):
            is_last = j == len(node["__files__"]) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{fname}")

    _walk(tree, "")
    return "\n".join(lines)
