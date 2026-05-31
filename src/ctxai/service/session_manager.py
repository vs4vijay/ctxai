"""
Session lifecycle management for the service layer.

A session bundles an Agent instance with persistence, configuration, and
TTL-based expiration. Sessions are keyed by UUID and stored in a
StateStore for crash recovery.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ctxai.agent.config import AgentConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.factory import LLMProviderFactory
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.logging import get_logger
from ctxai.monitoring import ACTIVE_SESSIONS, get_metrics
from ctxai.service.state_store import SQLiteStateStore, StateStore

logger = get_logger("service.session")


class SessionState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass
class Session:
    session_id: str
    created_at: datetime
    last_active: datetime
    config: AgentConfig
    agent: Agent
    metadata: dict[str, Any] = field(default_factory=dict)
    state: SessionState = SessionState.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "state": self.state.value,
        }

    def touch(self) -> None:
        self.last_active = datetime.utcnow()


class SessionManager:
    """Owns active sessions plus their persistent state."""

    def __init__(
        self,
        state_store: StateStore | None = None,
        max_sessions: int = 100,
        session_ttl: timedelta = timedelta(hours=24),
        working_directory: Path | None = None,
        max_messages_per_session: int = 1000,
    ):
        self.state_store = state_store or SQLiteStateStore("ctxai_state.db")
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self.working_directory = working_directory or Path.cwd()
        self.max_messages_per_session = max_messages_per_session
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # ----- Lifecycle -----

    async def create_session(
        self,
        config: AgentConfig | None = None,
        metadata: dict[str, Any] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> Session:
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                self._evict_oldest_locked()

            agent_config = config or AgentConfig()
            session_id = uuid.uuid4().hex
            agent = self._build_agent(agent_config, tool_registry)

            now = datetime.utcnow()
            session = Session(
                session_id=session_id,
                created_at=now,
                last_active=now,
                config=agent_config,
                agent=agent,
                metadata=metadata or {},
                state=SessionState.ACTIVE,
            )
            self._sessions[session_id] = session
            self.state_store.save_session(
                session_id,
                {
                    "created_at": now.isoformat(),
                    "last_active": now.isoformat(),
                    "config": agent_config.to_dict(),
                    "metadata": session.metadata,
                    "state": session.state.value,
                },
            )
            get_metrics().set_gauge(ACTIVE_SESSIONS, float(len(self._sessions)))
            logger.info("Session created", extra={"session_id": session_id})
            return session

    async def get_session(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if self._is_expired(session):
            session.state = SessionState.EXPIRED
            return None
        return session

    async def list_sessions(self) -> list[Session]:
        return [s for s in self._sessions.values() if not self._is_expired(s)]

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            removed = self._sessions.pop(session_id, None)
            self.state_store.delete_session(session_id)
            get_metrics().set_gauge(ACTIVE_SESSIONS, float(len(self._sessions)))
            return removed is not None

    async def clear_history(self, session_id: str) -> int:
        session = await self.get_session(session_id)
        if session is None:
            return 0
        session.agent.clear_conversation()
        return self.state_store.clear_messages(session_id)

    # ----- Messaging -----

    async def send_message(self, session_id: str, message: str) -> str:
        session = await self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session.agent.context.get_message_count() > self.max_messages_per_session:
            raise RuntimeError(
                f"Session has exceeded the max-messages limit "
                f"({self.max_messages_per_session})"
            )

        session.touch()
        self._record_message(session_id, "user", message)
        response = await session.agent.process_message(message)
        self._record_message(session_id, "assistant", response)
        self.state_store.save_session(session_id, session.to_dict())
        return response

    async def stream_message(self, session_id: str, message: str):
        session = await self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        session.touch()
        self._record_message(session_id, "user", message)
        chunks: list[str] = []
        async for chunk in session.agent.stream_message(message):
            chunks.append(chunk)
            yield chunk
        self._record_message(session_id, "assistant", "".join(chunks))
        self.state_store.save_session(session_id, session.to_dict())

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        return self.state_store.get_messages(session_id, limit=limit)

    # ----- Recovery -----

    async def recover_from_store(self, tool_registry: ToolRegistry | None = None) -> int:
        """Rebuild Agent instances for all persisted active sessions."""
        recovered = 0
        for data in self.state_store.list_sessions():
            if data["state"] != SessionState.ACTIVE.value:
                continue
            try:
                config = AgentConfig.from_dict(data["config"])
                agent = self._build_agent(config, tool_registry)
                session = Session(
                    session_id=data["session_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_active=datetime.fromisoformat(data["last_active"]),
                    config=config,
                    agent=agent,
                    metadata=data.get("metadata", {}),
                    state=SessionState(data["state"]),
                )
                if self._is_expired(session):
                    continue
                self._replay_history(session)
                self._sessions[session.session_id] = session
                recovered += 1
            except Exception as exc:
                logger.warning(f"Could not recover session {data.get('session_id')}: {exc}")
        get_metrics().set_gauge(ACTIVE_SESSIONS, float(len(self._sessions)))
        return recovered

    # ----- Helpers -----

    def _build_agent(
        self, config: AgentConfig, tool_registry: ToolRegistry | None = None
    ) -> Agent:
        llm_provider = LLMProviderFactory.create_provider(config.llm)
        registry = tool_registry or ToolRegistry()
        loop_cfg = AgentLoopConfig(
            llm_provider=llm_provider,
            tool_registry=registry,
            agent_config=config,
            working_directory=self.working_directory,
            available_indexes=[],
            max_iterations=config.behavior.max_iterations,
            verbose=config.behavior.verbose,
        )
        return Agent(loop_cfg)

    def _record_message(self, session_id: str, role: str, content: str) -> None:
        self.state_store.add_message(
            session_id,
            {
                "message_id": uuid.uuid4().hex,
                "role": role,
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

    def _replay_history(self, session: Session) -> None:
        for msg in self.state_store.get_messages(session.session_id):
            if msg["role"] == "user":
                session.agent.context.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                session.agent.context.add_assistant_message(msg["content"])

    def _is_expired(self, session: Session) -> bool:
        return datetime.utcnow() - session.last_active > self.session_ttl

    def _evict_oldest_locked(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_active)
        self._sessions.pop(oldest.session_id, None)
        oldest.state = SessionState.EXPIRED
        self.state_store.save_session(oldest.session_id, oldest.to_dict())
        logger.info("Evicted oldest session", extra={"session_id": oldest.session_id})

    @property
    def active_count(self) -> int:
        return len(self._sessions)
