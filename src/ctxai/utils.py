"""
Utility functions for ctxai.
Handles environment variables, path resolution, and common helpers.
"""

import os
from pathlib import Path
from typing import Optional


def get_ctxai_home(project_path: Path | None = None) -> Path:
    """
    Get the .ctxai directory path.

    Priority:
    1. CTXAI_HOME environment variable (absolute path)
    2. project_path/.ctxai (if project_path provided)
    3. Current directory/.ctxai (default)

    Args:
        project_path: Optional project root path

    Returns:
        Path to .ctxai directory

    Examples:
        # Use environment variable
        export CTXAI_HOME=/path/to/global/.ctxai
        get_ctxai_home() -> Path('/path/to/global/.ctxai')

        # Use project-specific
        get_ctxai_home(Path('/my/project')) -> Path('/my/project/.ctxai')

        # Use current directory
        get_ctxai_home() -> Path('./.ctxai')
    """
    # Check environment variable first
    ctxai_home = os.getenv("CTXAI_HOME")
    if ctxai_home:
        home_path = Path(ctxai_home).expanduser().resolve()
        return home_path

    # Use project path if provided
    if project_path:
        return project_path / ".ctxai"

    # Default to current directory
    return Path.cwd() / ".ctxai"


def get_indexes_dir(project_path: Path | None = None) -> Path:
    """
    Get the indexes directory.

    Args:
        project_path: Optional project root path

    Returns:
        Path to indexes directory
    """
    return get_ctxai_home(project_path) / "indexes"


def get_config_path(project_path: Path | None = None) -> Path:
    """
    Get the config file path.

    Args:
        project_path: Optional project root path

    Returns:
        Path to config.json file
    """
    return get_ctxai_home(project_path) / "config.json"


def ensure_ctxai_home(project_path: Path | None = None) -> Path:
    """
    Ensure .ctxai directory exists and return its path.

    Args:
        project_path: Optional project root path

    Returns:
        Path to .ctxai directory
    """
    ctxai_home = get_ctxai_home(project_path)
    ctxai_home.mkdir(parents=True, exist_ok=True)
    return ctxai_home


def is_using_global_home() -> bool:
    """
    Check if using global CTXAI_HOME.

    Returns:
        True if CTXAI_HOME environment variable is set
    """
    return bool(os.getenv("CTXAI_HOME"))


def get_ctxai_home_info() -> dict:
    """
    Get information about CTXAI_HOME configuration.

    Returns:
        Dictionary with CTXAI_HOME information
    """
    ctxai_home_env = os.getenv("CTXAI_HOME")

    return {
        "has_env_var": bool(ctxai_home_env),
        "env_value": ctxai_home_env,
        "resolved_path": str(get_ctxai_home()),
        "is_global": is_using_global_home(),
    }
