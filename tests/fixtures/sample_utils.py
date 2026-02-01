"""Utility functions for sample code."""

import os
from typing import List, Optional


def read_file(path: str) -> str:
    """Read file contents."""
    with open(path, 'r') as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write content to file."""
    with open(path, 'w') as f:
        f.write(content)


def list_files(directory: str, pattern: Optional[str] = None) -> List[str]:
    """List files in directory."""
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if pattern is None or pattern in item:
                files.append(item)
    return files


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
