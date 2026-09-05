"""Shared artifact discipline for ctxai evaluation frameworks (RE-01, HH-09).

Holds the pieces every eval framework reuses: canonical JSON + content
fingerprints, atomic writes with fsync, secret/absolute-path redaction,
volatile-field stripping for byte-stable comparisons, and the ``MetricValue``
model that marks metrics which cannot be computed as explicitly unavailable
(never reported as zero).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.sessions import redact_secrets
from ..utils import get_ctxai_home

# Keys whose values are expected to change between otherwise deterministic
# runs: run identity, wall-clock timestamps, and measured durations. Artifact
# comparison ignores exactly these keys and nothing else.
VOLATILE_KEYS = frozenset(
    {
        "run_id",
        "created_at",
        "duration_ms",
        "timestamp",
        "timings",
        "latency",
        "latency_p50_ms",
        "latency_p95_ms",
    }
)


class EvalError(RuntimeError):
    """Raised when an evaluation cannot run or complete honestly."""


def canonical_json(payload: Any) -> str:
    """Serialize a payload to a deterministic canonical JSON string.

    Args:
        payload: Any JSON-serializable structure.

    Returns:
        Canonical JSON text with sorted keys and compact separators.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_fingerprint(payload: Any) -> str:
    """Compute a content-derived sha256 fingerprint of a JSON-serializable payload.

    Args:
        payload: Any JSON-serializable structure.

    Returns:
        Hex digest of the canonical JSON encoding.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> Path:
    """Write JSON atomically with fsync (the ``sessions.py`` pattern).

    Args:
        path: Destination file path; parent directories are created.
        payload: JSON payload to persist.

    Returns:
        The path the payload was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def strip_volatile(payload: Any) -> Any:
    """Recursively remove documented volatile keys from a payload.

    Comparison of two evaluation artifacts ignores exactly the keys in
    ``VOLATILE_KEYS`` (timestamps and measured durations) and nothing else.

    Args:
        payload: Parsed artifact payload (dicts/lists/scalars).

    Returns:
        A copy of the payload with volatile keys removed at every level.
    """
    if isinstance(payload, dict):
        return {key: strip_volatile(item) for key, item in payload.items() if key not in VOLATILE_KEYS}
    if isinstance(payload, list):
        return [strip_volatile(item) for item in payload]
    return payload


def path_redaction_roots(project_root: Path) -> list[tuple[str, str]]:
    """Build the ordered (absolute-path-prefix, placeholder) redaction pairs.

    Args:
        project_root: Resolved repository root of the evaluated project.

    Returns:
        Pairs ordered longest-prefix-first so nested roots replace fully.
    """
    roots = [
        (str(get_ctxai_home(project_root).resolve()), "<ctxai-home>"),
        (str(project_root.resolve()), "<project>"),
        (str(Path.home().resolve()), "~"),
    ]
    return sorted(roots, key=lambda pair: len(pair[0]), reverse=True)


def redact_home_paths(value: Any, project_root: Path) -> Any:
    """Recursively replace known absolute path prefixes with placeholders.

    Args:
        value: Payload (dicts/lists/strings/scalars).
        project_root: Resolved repository root of the evaluated project.

    Returns:
        A copy of the payload with absolute home/project paths replaced.
    """
    roots = path_redaction_roots(project_root)
    return _redact_paths(value, roots)


def _redact_paths(value: Any, roots: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_paths(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item, roots) for item in value]
    if isinstance(value, str):
        redacted = value
        for prefix, placeholder in roots:
            if prefix and prefix in redacted:
                redacted = redacted.replace(prefix, placeholder)
        return redacted
    return value


def redact_artifact(value: Any, project_root: Path) -> Any:
    """Apply the full artifact redaction pipeline before persistence.

    Secrets are redacted via ``sessions.redact_secrets``; absolute project,
    ctxai-home, and user-home path prefixes are replaced with placeholders.

    Args:
        value: Payload to redact.
        project_root: Resolved repository root of the evaluated project.

    Returns:
        The redacted payload.
    """
    return redact_secrets(redact_home_paths(value, project_root))


@dataclass(frozen=True)
class MetricValue:
    """A metric result that is either available or explicitly unavailable.

    Attributes:
        value: The metric value, or ``None`` when unavailable.
        reason: Why the metric is unavailable, or ``None`` when available.
    """

    value: float | None
    reason: str | None

    @classmethod
    def available(cls, value: float) -> MetricValue:
        """Build an available metric value.

        Args:
            value: The computed metric value.

        Returns:
            MetricValue with no unavailability reason.
        """
        return cls(value=value, reason=None)

    @classmethod
    def unavailable(cls, reason: str) -> MetricValue:
        """Build an explicitly unavailable metric value.

        Args:
            reason: Human-readable reason the metric cannot be computed.

        Returns:
            MetricValue with a ``None`` value and a reason.
        """
        return cls(value=None, reason=reason)

    @property
    def is_available(self) -> bool:
        """Whether the metric could be computed.

        Returns:
            True when a value is present.
        """
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON representation.

        Returns:
            ``{"available": True, "value": x}`` or
            ``{"available": False, "value": None, "reason": r}``.
        """
        if self.is_available:
            return {"available": True, "value": self.value}
        return {"available": False, "value": None, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricValue:
        """Rebuild from the JSON representation.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            The reconstructed MetricValue.

        Raises:
            ValueError: If the payload shape is not a metric value.
        """
        if data.get("available"):
            value = data.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("Available MetricValue requires a numeric value")
            return cls.available(float(value))
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("Unavailable MetricValue requires a reason string")
        return cls.unavailable(reason)
