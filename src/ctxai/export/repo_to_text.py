"""
Convert a repository to a single-file text/markdown/xml export.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from ctxai.export.config import ExportConfig
from ctxai.export.formatters import (
    FileEntry,
    RepositorySummary,
    detect_language,
    render_markdown,
    render_text,
    render_tree,
    render_xml,
)

_DEFAULT_EXCLUDES = (
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".ctxai",
)


class RepoTextExporter:
    """Walk a repo and emit a single-file export in the requested format."""

    def __init__(self, root: Path | str, config: ExportConfig | None = None):
        self.root = Path(root).resolve()
        self.config = config or ExportConfig()
        self.config.validate()

    # ----- Walking -----

    def _gitignore_spec(self):
        if not self.config.follow_gitignore:
            return None
        try:
            import pathspec
        except ImportError:
            return None
        gi = self.root / ".gitignore"
        if not gi.exists():
            return None
        try:
            return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text().splitlines())
        except Exception:
            return None

    def _iter_candidates(self) -> Iterator[Path]:
        spec = self._gitignore_spec()
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=self.config.follow_symlinks):
            dir_obj = Path(dirpath)
            # Filter directories in place to skip large/irrelevant trees.
            kept_dirs: list[str] = []
            for d in dirnames:
                if d in _DEFAULT_EXCLUDES:
                    continue
                if not self.config.include_hidden and d.startswith("."):
                    continue
                full = dir_obj / d
                rel = full.relative_to(self.root)
                if spec is not None and spec.match_file(str(rel).replace(os.sep, "/") + "/"):
                    continue
                if self.config.matches_exclude(rel):
                    continue
                kept_dirs.append(d)
            dirnames[:] = kept_dirs

            for fname in filenames:
                if not self.config.include_hidden and fname.startswith("."):
                    continue
                path = dir_obj / fname
                try:
                    rel = path.relative_to(self.root)
                except ValueError:
                    continue
                if spec is not None and spec.match_file(str(rel).replace(os.sep, "/")):
                    continue
                if self.config.matches_exclude(rel):
                    continue
                if not self.config.matches_include(path):
                    continue
                yield path

    def _read_file(self, path: Path) -> str | None:
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size > self.config.max_file_size_mb * 1024 * 1024:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if not self.config.include_binary and b"\x00" in data[:8192]:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("latin-1")
            except Exception:
                return None

    # ----- Collection -----

    def collect(self) -> RepositorySummary:
        files: list[FileEntry] = []
        total_size = 0
        truncated = False
        max_total = self.config.max_total_size_mb * 1024 * 1024

        for path in self._iter_candidates():
            if len(files) >= self.config.max_files:
                truncated = True
                break
            content = self._read_file(path)
            if content is None:
                continue
            size = len(content.encode("utf-8"))
            if total_size + size > max_total:
                truncated = True
                break
            try:
                rel = path.relative_to(self.root)
                stat = path.stat()
            except OSError:
                continue
            files.append(
                FileEntry(
                    path=path,
                    relative=str(rel).replace(os.sep, "/"),
                    content=content,
                    size=size,
                    lines=content.count("\n") + (0 if content.endswith("\n") else 1),
                    language=detect_language(path),
                    last_modified=datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
                )
            )
            total_size += size

        return RepositorySummary.build(self.root, files, truncated)

    # ----- Export -----

    def export(self, output: Path | str, with_tree: bool = True) -> Path:
        summary = self.collect()
        text = self.render(summary, with_tree=with_tree)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out

    def render(self, summary: RepositorySummary, with_tree: bool = True) -> str:
        tree = render_tree(self.root, summary.files) if with_tree else None
        fmt = self.config.output_format
        if fmt == "text":
            return render_text(summary)
        if fmt == "markdown":
            return render_markdown(summary, tree=tree)
        if fmt == "xml":
            return render_xml(summary)
        raise ValueError(f"Unsupported text-style format: {fmt}")
