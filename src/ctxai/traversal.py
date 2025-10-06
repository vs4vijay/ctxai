"""
Code traversal module for recursively walking through codebase.
Respects .gitignore patterns and handles file filtering.
"""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import pathspec


class CodeTraversal:
    """Traverse a codebase recursively with gitignore support."""

    def __init__(
        self,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        follow_gitignore: bool = True,
    ):
        """
        Initialize the code traversal.

        Args:
            root_path: Root directory to traverse
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: Additional patterns to exclude
            follow_gitignore: Whether to respect .gitignore files
        """
        self.root_path = root_path
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.follow_gitignore = follow_gitignore

        # Load gitignore patterns
        self.gitignore_spec = self._load_gitignore() if follow_gitignore else None

        # Default exclude patterns
        self.default_excludes = {
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            ".ctxai",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            "*.pyc",
            "*.pyo",
            "*.egg-info",
        }

    def _load_gitignore(self) -> pathspec.PathSpec | None:
        """Load .gitignore patterns from the root directory."""
        gitignore_path = self.root_path / ".gitignore"
        if not gitignore_path.exists():
            return None

        try:
            with open(gitignore_path, encoding="utf-8") as f:
                patterns = f.read().splitlines()
            return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except Exception as e:
            print(f"Warning: Could not load .gitignore: {e}")
            return None

    def _should_exclude_path(self, path: Path) -> bool:
        """Check if a path should be excluded based on various rules."""
        relative_path = path.relative_to(self.root_path)
        relative_str = str(relative_path).replace("\\", "/")

        # Check default excludes
        for part in relative_path.parts:
            if part in self.default_excludes:
                return True

        # Check gitignore patterns
        if self.gitignore_spec:
            if self.gitignore_spec.match_file(relative_str):
                return True
            if path.is_dir() and self.gitignore_spec.match_file(relative_str + "/"):
                return True

        # Check user-defined exclude patterns
        if self.exclude_patterns:
            for pattern in self.exclude_patterns:
                if path.match(pattern):
                    return True

        return False

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on include patterns."""
        # If no include patterns specified, include all files
        if not self.include_patterns:
            return True

        # Check if file matches any include pattern
        for pattern in self.include_patterns:
            if file_path.match(pattern):
                return True

        return False

    def traverse(self) -> Generator[Path, None, None]:
        """
        Traverse the codebase and yield file paths.

        Yields:
            Path objects for each file that should be processed
        """
        for root, dirs, files in os.walk(self.root_path, topdown=True):
            root_path = Path(root)

            # Filter directories in-place to prevent descending into excluded dirs
            dirs[:] = [d for d in dirs if not self._should_exclude_path(root_path / d)]

            # Process files
            for file_name in files:
                file_path = root_path / file_name

                # Skip if excluded
                if self._should_exclude_path(file_path):
                    continue

                # Skip if doesn't match include patterns
                if not self._should_include_file(file_path):
                    continue

                # Skip binary files (basic check)
                if self._is_likely_binary(file_path):
                    continue

                yield file_path

    def _is_likely_binary(self, file_path: Path) -> bool:
        """
        Quick heuristic to detect binary files.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file is likely binary, False otherwise
        """
        binary_extensions = {
            ".pyc",
            ".pyo",
            ".so",
            ".dll",
            ".dylib",
            ".exe",
            ".bin",
            ".dat",
            ".db",
            ".sqlite",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
        }

        if file_path.suffix.lower() in binary_extensions:
            return True

        # Try reading first few bytes to detect binary content
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                # Check for null bytes which typically indicate binary files
                if b"\x00" in chunk:
                    return True
        except Exception:
            # If we can't read it, treat it as binary
            return True

        return False

    def get_file_count(self) -> int:
        """
        Count total number of files that will be processed.

        Returns:
            Total file count
        """
        return sum(1 for _ in self.traverse())
