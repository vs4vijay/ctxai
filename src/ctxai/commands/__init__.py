"""Commands package."""

from .config_command import (
    edit_config,
    get_config,
    list_config,
    set_config,
    show_config_file,
    unset_config,
)

__all__ = [
    "list_config",
    "get_config",
    "set_config",
    "unset_config",
    "show_config_file",
    "edit_config",
]


def start_dashboard(port: int = 3000, host: str = "127.0.0.1", allow_remote: bool = False):
    """Lazy import for dashboard to avoid chromadb dependency."""
    from .dashboard_command import start_dashboard as _start_dashboard
    return _start_dashboard(port, host=host, allow_remote=allow_remote)


def index_codebase(**kwargs):
    """Lazy import for index command."""
    from .index_command import index_codebase as _index_codebase
    return _index_codebase(**kwargs)


def query_codebase(**kwargs):
    """Lazy import for query command."""
    from .query_command import query_codebase as _query_codebase
    return _query_codebase(**kwargs)


def start_mcp_server(**kwargs):
    """Lazy import for server command."""
    from .server_command import start_mcp_server as _start_mcp_server
    return _start_mcp_server(**kwargs)
