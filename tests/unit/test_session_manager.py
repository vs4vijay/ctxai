"""Tests for ctxai.service.session_manager."""

from datetime import timedelta
from pathlib import Path

import pytest

from ctxai.service.session_manager import SessionManager
from ctxai.service.state_store import SQLiteStateStore
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "s.db")


@pytest.fixture
def manager_factory(tmp_path: Path, store, monkeypatch):
    """Patch the LLMProviderFactory so we don't need real network access."""

    from ctxai.service import session_manager as sm

    def _factory_create(_config):
        return MockLLMProvider(responses=[create_mock_response(content="ok-response")] * 5)

    monkeypatch.setattr(sm.LLMProviderFactory, "create_provider", staticmethod(_factory_create))
    return lambda **kwargs: SessionManager(
        state_store=store,
        working_directory=tmp_path,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_create_session_persists_to_store(manager_factory, store):
    mgr = manager_factory()
    session = await mgr.create_session()
    assert session.session_id
    assert mgr.active_count == 1
    persisted = store.get_session(session.session_id)
    assert persisted is not None


@pytest.mark.asyncio
async def test_delete_session(manager_factory, store):
    mgr = manager_factory()
    session = await mgr.create_session()
    assert await mgr.delete_session(session.session_id) is True
    assert store.get_session(session.session_id) is None


@pytest.mark.asyncio
async def test_send_message_records_history(manager_factory, store):
    mgr = manager_factory()
    session = await mgr.create_session()
    response = await mgr.send_message(session.session_id, "hello")
    assert "ok-response" in response
    msgs = store.get_messages(session.session_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_max_sessions_evicts_oldest(manager_factory):
    mgr = manager_factory(max_sessions=2)
    s1 = await mgr.create_session()
    await mgr.create_session()
    await mgr.create_session()
    assert mgr.active_count == 2
    assert s1.session_id not in [s.session_id for s in await mgr.list_sessions()]


@pytest.mark.asyncio
async def test_recover_from_store(manager_factory, store):
    mgr = manager_factory()
    session = await mgr.create_session()
    await mgr.send_message(session.session_id, "remember me")

    # Build fresh manager and recover
    mgr2 = manager_factory()
    recovered = await mgr2.recover_from_store()
    assert recovered >= 1
    sessions = await mgr2.list_sessions()
    assert any(s.session_id == session.session_id for s in sessions)


@pytest.mark.asyncio
async def test_expired_session_not_returned(manager_factory):
    mgr = manager_factory(session_ttl=timedelta(seconds=0))
    session = await mgr.create_session()
    # Sleep is not strictly required because TTL is zero
    result = await mgr.get_session(session.session_id)
    assert result is None
