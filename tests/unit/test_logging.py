"""Tests for ctxai.logging structured logging."""

import json
import logging

import pytest

from ctxai import logging as ctx_logging


def _reset():
    ctx_logging._initialized = False
    root = logging.getLogger("ctxai")
    for h in list(root.handlers):
        root.removeHandler(h)


def test_setup_logging_is_idempotent():
    _reset()
    ctx_logging.setup_logging(level="INFO")
    root = logging.getLogger("ctxai")
    handler_count = len(root.handlers)
    ctx_logging.setup_logging(level="DEBUG")
    assert len(root.handlers) == handler_count


def test_get_logger_returns_namespaced_logger():
    log = ctx_logging.get_logger("agent.core")
    assert log.name == "ctxai.agent.core"


def test_request_id_context_manager():
    with ctx_logging.RequestContext("abc123") as rid:
        assert rid == "abc123"
        assert ctx_logging.get_request_id() == "abc123"
    assert ctx_logging.get_request_id() == "-"


def test_new_request_id_generates_unique_ids():
    a = ctx_logging.new_request_id()
    b = ctx_logging.new_request_id()
    assert a != b
    assert len(a) == 12


def test_json_formatter_emits_valid_json():
    record = logging.LogRecord(
        name="ctxai.test", level=logging.INFO, pathname="x.py", lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    formatter = ctx_logging.JsonFormatter()
    out = formatter.format(record)
    payload = json.loads(out)
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_pii_filter_redacts_secrets(caplog):
    _reset()
    ctx_logging.setup_logging(level="INFO")
    log = ctx_logging.get_logger("test")
    log.info("login", extra={"api_key": "sk-secret-12345"})
    # Filter modifies the record's __dict__, so the redaction happens
    # before output; just verify the filter callable directly.
    record = logging.LogRecord("ctxai", logging.INFO, "x", 1, "m", (), None)
    record.api_key = "sk-secret"
    ctx_logging._filter_pii(record)
    assert record.api_key == "***REDACTED***"
