"""Export configuration shared by all formatters."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportConfig:
    max_files: int = 500
    max_total_size_mb: int = 50
    max_file_size_mb: int = 5
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    include_binary: bool = False
    include_hidden: bool = False
    follow_symlinks: bool = False
    follow_gitignore: bool = True
    output_format: str = "markdown"  # text, markdown, json, html, xml

    _ALLOWED_FORMATS = ("text", "markdown", "json", "html", "xml")

    def validate(self) -> None:
        if not 1 <= self.max_files <= 100_000:
            raise ValueError("max_files must be between 1 and 100_000")
        if not 1 <= self.max_total_size_mb <= 10_000:
            raise ValueError("max_total_size_mb must be between 1 and 10000")
        if not 1 <= self.max_file_size_mb <= 1_000:
            raise ValueError("max_file_size_mb must be between 1 and 1000")
        if self.output_format not in self._ALLOWED_FORMATS:
            raise ValueError(
                f"output_format must be one of {self._ALLOWED_FORMATS}, got {self.output_format!r}"
            )

    def matches_include(self, path: Path) -> bool:
        if not self.include_patterns:
            return True
        return any(fnmatch.fnmatch(path.name, p) or fnmatch.fnmatch(str(path), p)
                   for p in self.include_patterns)

    def matches_exclude(self, path: Path) -> bool:
        if not self.exclude_patterns:
            return False
        return any(
            fnmatch.fnmatch(path.name, p)
            or fnmatch.fnmatch(str(path), p)
            or any(fnmatch.fnmatch(part, p) for part in path.parts)
            for p in self.exclude_patterns
        )
