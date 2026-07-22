"""Stable, versioned result contracts for the ctxai MCP surface."""

from __future__ import annotations

from typing import Any

MCP_RESULT_SCHEMA_VERSION = "1.0"


class MCPErrorCode:
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    INDEX_FAILED = "index_failed"
    QUERY_FAILED = "query_failed"
    STORAGE_FAILED = "storage_failed"
    INTERNAL_ERROR = "internal_error"


def success(data: Any, *, message: str | None = None) -> dict[str, Any]:
    """Build a successful protocol result envelope."""
    result: dict[str, Any] = {
        "schema_version": MCP_RESULT_SCHEMA_VERSION,
        "ok": True,
        "data": data,
    }
    if message is not None:
        result["message"] = message
    return result


def failure(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a deterministic error result envelope."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {
        "schema_version": MCP_RESULT_SCHEMA_VERSION,
        "ok": False,
        "error": error,
    }
