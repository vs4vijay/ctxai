"""
Persistent state store for the service layer.

The default backend is SQLite (single-node, no external dependencies).
The StateStore Protocol allows swapping in Redis or other backends later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    last_active    TEXT NOT NULL,
    config_json    TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    state          TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_calls_json TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS plans (
    plan_id       TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    goal          TEXT NOT NULL,
    steps_json    TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    completed_at  TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_plans_session ON plans(session_id);
"""


class StateStore(Protocol):
    """Abstract persistence interface."""

    def save_session(self, session_id: str, data: dict[str, Any]) -> None: ...

    def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    def delete_session(self, session_id: str) -> bool: ...

    def list_sessions(self) -> list[dict[str, Any]]: ...

    def add_message(self, session_id: str, message: dict[str, Any]) -> None: ...

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]: ...

    def clear_messages(self, session_id: str) -> int: ...

    def save_plan(self, session_id: str, plan: dict[str, Any]) -> None: ...

    def get_active_plan(self, session_id: str) -> dict[str, Any] | None: ...


class SQLiteStateStore:
    """Single-file SQLite implementation of StateStore."""

    def __init__(self, db_path: str | Path = "ctxai_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            for stmt in _SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

    # ----- Sessions -----

    def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_active, config_json, metadata_json, state)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_active = excluded.last_active,
                    config_json = excluded.config_json,
                    metadata_json = excluded.metadata_json,
                    state = excluded.state
                """,
                (
                    session_id,
                    data.get("created_at", datetime.utcnow().isoformat()),
                    data.get("last_active", datetime.utcnow().isoformat()),
                    json.dumps(data.get("config", {})),
                    json.dumps(data.get("metadata", {})),
                    data.get("state", "active"),
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "last_active": row["last_active"],
            "config": json.loads(row["config_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "state": row["state"],
        }

    def delete_session(self, session_id: str) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM plans WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount > 0

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY last_active DESC").fetchall()
        return [
            {
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "last_active": r["last_active"],
                "config": json.loads(r["config_json"]),
                "metadata": json.loads(r["metadata_json"]),
                "state": r["state"],
            }
            for r in rows
        ]

    # ----- Messages -----

    def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, session_id, role, content, tool_calls_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message["message_id"],
                    session_id,
                    message["role"],
                    message["content"],
                    json.dumps(message.get("tool_calls")) if message.get("tool_calls") else None,
                    message.get("created_at", datetime.utcnow().isoformat()),
                ),
            )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC"
        params: list[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "message_id": r["message_id"],
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "tool_calls": json.loads(r["tool_calls_json"]) if r["tool_calls_json"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def clear_messages(self, session_id: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            return cur.rowcount

    # ----- Plans -----

    def save_plan(self, session_id: str, plan: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plans (plan_id, session_id, goal, steps_json, status, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    steps_json = excluded.steps_json,
                    status = excluded.status,
                    completed_at = excluded.completed_at
                """,
                (
                    plan["plan_id"],
                    session_id,
                    plan["goal"],
                    json.dumps(plan.get("steps", [])),
                    plan.get("status", "draft"),
                    plan.get("created_at", datetime.utcnow().isoformat()),
                    plan.get("completed_at"),
                ),
            )

    def get_active_plan(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM plans
                WHERE session_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "plan_id": row["plan_id"],
            "session_id": row["session_id"],
            "goal": row["goal"],
            "steps": json.loads(row["steps_json"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }
