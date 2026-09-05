"""
Optional OpenTelemetry tracing helpers.

If `opentelemetry-api` is installed, `get_tracer()` returns a real tracer.
Otherwise it returns a no-op shim so the rest of ctxai can stay free of
hard dependencies.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        return

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


def get_tracer(name: str = "ctxai"):
    """Return an OpenTelemetry tracer if available, else a no-op tracer."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


def setup_tracing(service_name: str = "ctxai") -> None:
    """
    Configure OpenTelemetry to export spans via OTLP if env vars are set.

    Honors `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME`.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        return
    import os

    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except ImportError:
            return
