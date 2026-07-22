"""Durable, repository-scoped interactive chat sessions."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context import ConversationContext

SESSION_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)(\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
)
_SECRET_KEYS = {
    "api_key", "apikey", "access_token", "auth_token", "token", "password",
    "secret", "client_secret", "authorization",
}


def redact_secrets(value: Any) -> Any:
    """Recursively redact common credential shapes before persistence/export."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower().replace("-", "_") in _SECRET_KEYS
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if not isinstance(value, str):
        return value
    redacted = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


@dataclass
class SessionRecord:
    name: str
    context: ConversationContext
    provider: str
    model: str
    project_root: str


class SessionStore:
    """Save and restore atomic JSON session files under .ctxai/sessions."""

    def __init__(self, project_root: Path, storage_dir: Path | None = None):
        self.project_root = project_root.resolve()
        self.storage_dir = storage_dir or self.project_root / ".ctxai" / "sessions"

    def _path(self, name: str) -> Path:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("Session name must use only letters, numbers, '.', '_' or '-'")
        return self.storage_dir / f"{name}.json"

    def save(self, record: SessionRecord) -> Path:
        path = self._path(record.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = redact_secrets({
            "schema_version": SESSION_SCHEMA_VERSION,
            "name": record.name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "provider": record.provider,
            "model": record.model,
            "context": record.context.to_dict(),
        })
        fd, temporary = tempfile.mkstemp(prefix=f".{record.name}-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def load(self, name: str) -> SessionRecord:
        payload = json.loads(self._path(name).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise ValueError("Unsupported session schema version")
        if Path(payload["project_root"]).resolve() != self.project_root:
            raise ValueError("Session belongs to a different repository")
        return SessionRecord(
            name=payload["name"],
            context=ConversationContext.from_dict(payload["context"]),
            provider=payload["provider"],
            model=payload["model"],
            project_root=payload["project_root"],
        )

    def clear(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def export(self, record: SessionRecord, destination: Path) -> Path:
        destination = destination.resolve()
        try:
            destination.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("Session exports must stay inside the repository") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = redact_secrets(record.context.to_dict())
        lines = [f"# ctxai session: {record.name}", ""]
        for message in data["messages"]:
            lines.extend((f"## {message['role'].title()}", "", message["content"], ""))
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination
