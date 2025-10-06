"""
Project size validation module.
Checks project size limits and provides warnings/errors.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import IndexConfig


@dataclass
class ProjectStats:
    """Statistics about a project to be indexed."""

    total_files: int
    total_size_bytes: int
    oversized_files: list[tuple[Path, int]]  # Files exceeding max size
    largest_files: list[tuple[Path, int]]  # Top 5 largest files

    @property
    def total_size_mb(self) -> float:
        """Total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"


class ProjectSizeValidator:
    """Validates project size against configured limits."""

    def __init__(self, config: IndexConfig):
        """
        Initialize validator.

        Args:
            config: Index configuration with size limits
        """
        self.config = config

    def analyze_files(self, file_paths: list[Path]) -> ProjectStats:
        """
        Analyze a list of files and gather statistics.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            ProjectStats object with analysis results
        """
        total_size = 0
        oversized_files = []
        file_sizes = []

        max_file_bytes = self.config.max_file_size_mb * 1024 * 1024

        for file_path in file_paths:
            try:
                size = file_path.stat().st_size
                total_size += size
                file_sizes.append((file_path, size))

                if size > max_file_bytes:
                    oversized_files.append((file_path, size))
            except Exception as e:
                print(f"Warning: Could not get size of {file_path}: {e}")

        # Sort by size and get top 5
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        largest_files = file_sizes[:5]

        return ProjectStats(
            total_files=len(file_paths),
            total_size_bytes=total_size,
            oversized_files=oversized_files,
            largest_files=largest_files,
        )

    def validate(self, stats: ProjectStats) -> tuple[bool, list[str]]:
        """
        Validate project stats against limits.

        Args:
            stats: Project statistics

        Returns:
            Tuple of (is_valid, list_of_messages)
            is_valid is False if hard limits are exceeded
        """
        messages = []
        is_valid = True

        # Check file count
        if stats.total_files > self.config.max_files:
            is_valid = False
            messages.append(f"❌ Too many files: {stats.total_files} files (limit: {self.config.max_files})")
            messages.append(
                "   Consider using --include patterns to filter files, or increase max_files in .ctxai/config.json"
            )
        elif stats.total_files > self.config.max_files * 0.8:
            # Warning at 80% of limit
            messages.append(f"⚠️  Approaching file limit: {stats.total_files} files (limit: {self.config.max_files})")

        # Check total size
        max_size_mb = self.config.max_total_size_mb
        if stats.total_size_mb > max_size_mb:
            is_valid = False
            messages.append(
                f"❌ Project too large: {stats.format_size(stats.total_size_bytes)} (limit: {max_size_mb} MB)"
            )
            messages.append(
                "   Consider using --include patterns to filter files, "
                "or increase max_total_size_mb in .ctxai/config.json"
            )
        elif stats.total_size_mb > max_size_mb * 0.8:
            # Warning at 80% of limit
            messages.append(
                f"⚠️  Approaching size limit: {stats.format_size(stats.total_size_bytes)} (limit: {max_size_mb} MB)"
            )

        # Check oversized files
        if stats.oversized_files:
            messages.append(
                f"⚠️  Found {len(stats.oversized_files)} file(s) exceeding "
                f"{self.config.max_file_size_mb} MB (these will be skipped):"
            )
            for file_path, size in stats.oversized_files[:5]:  # Show first 5
                messages.append(f"   • {file_path.name}: {stats.format_size(size)}")
            if len(stats.oversized_files) > 5:
                messages.append(f"   ... and {len(stats.oversized_files) - 5} more")

        return is_valid, messages

    def get_summary(self, stats: ProjectStats) -> list[str]:
        """
        Get a summary of project statistics.

        Args:
            stats: Project statistics

        Returns:
            List of summary messages
        """
        messages = [
            "📊 Project Statistics:",
            f"   • Total files: {stats.total_files:,}",
            f"   • Total size: {stats.format_size(stats.total_size_bytes)}",
        ]

        if stats.largest_files:
            messages.append("   • Largest files:")
            for file_path, size in stats.largest_files:
                messages.append(f"     - {file_path.name}: {stats.format_size(size)}")

        return messages


class ProjectSizeLimitError(Exception):
    """Raised when project exceeds size limits."""

    def __init__(self, messages: list[str]):
        """
        Initialize with validation messages.

        Args:
            messages: List of error/warning messages
        """
        self.messages = messages
        super().__init__("\n".join(messages))
