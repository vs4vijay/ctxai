"""
Structured logging for ctxai.

Provides JSON-formatted logging with request ID tracking, rotating file
handlers, and environment-based configuration. Loaded once via
`setup_logging()`; subsequent modules grab loggers with `get_logger()`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_request_id_var: ContextVar[str] = ContextVar("ctxai_request_id", default="-")
_initialized = False


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for ingestion by log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName", "taskName",
            }:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Concise multi-line formatter used in DEV mode for readability."""

    DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [req=%(request_id)s] %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.DEFAULT_FORMAT, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = _request_id_var.get()
        return super().format(record)


def _filter_pii(record: logging.LogRecord) -> bool:
    """Strip likely-secret keys from extras before they reach handlers."""
    secret_markers = ("api_key", "apikey", "token", "secret", "password", "authorization")
    for attr_name in list(record.__dict__):
        if any(marker in attr_name.lower() for marker in secret_markers):
            record.__dict__[attr_name] = "***REDACTED***"
    return True


def setup_logging(
    level: str | int | None = None,
    log_file: Path | str | None = None,
    json_format: bool | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configure root logger. Safe to call multiple times — only first call
    actually installs handlers.

    Args:
        level: Log level (string or int). Falls back to CTXAI_LOG_LEVEL env var, else INFO.
        log_file: Optional path for rotating file output.
        json_format: Force JSON output. If None, infers from CTXAI_ENV (prod=>json).
        max_bytes: Max bytes per log file before rotation.
        backup_count: Number of rotated files to keep.
    """
    global _initialized
    if _initialized:
        return

    level = level or os.getenv("CTXAI_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    env = os.getenv("CTXAI_ENV", "dev").lower()
    if json_format is None:
        json_format = env in ("prod", "production")

    root = logging.getLogger("ctxai")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = JsonFormatter() if json_format else HumanFormatter()

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(_filter_pii)
    root.addHandler(stderr_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(_filter_pii)
        root.addHandler(file_handler)

    root.propagate = False
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger under the ctxai root."""
    if not name.startswith("ctxai"):
        name = f"ctxai.{name}"
    return logging.getLogger(name)


def new_request_id() -> str:
    """Generate and set a new request ID for the current context."""
    rid = uuid.uuid4().hex[:12]
    _request_id_var.set(rid)
    return rid


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()


class RequestContext:
    """Context manager that scopes a request ID to a block of work."""

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self._token = None

    def __enter__(self) -> str:
        self._token = _request_id_var.set(self.request_id)
        return self.request_id

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _request_id_var.reset(self._token)
