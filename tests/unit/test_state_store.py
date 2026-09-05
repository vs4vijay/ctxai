"""Tests for ctxai.service.state_store.SQLiteStateStore."""

from pathlib import Path

import pytest

from ctxai.service.state_store import SQLiteStateStore


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStateStore:
    return SQLiteStateStore(tmp_path / "test.db")


def test_save_and_get_session(store: SQLiteStateStore):
    store.save_session(
        "abc",
        {
            "created_at": "2026-01-01T00:00:00",
            "last_active": "2026-01-01T00:00:00",
            "config": {"provider": "openrouter"},
            "metadata": {"x": 1},
            "state": "active",
        },
    )
    data = store.get_session("abc")
    assert data is not None
    assert data["session_id"] == "abc"
    assert data["config"]["provider"] == "openrouter"


def test_delete_session(store: SQLiteStateStore):
    store.save_session("abc", {"created_at": "x", "last_active": "x", "config": {}})
    assert store.delete_session("abc")
    assert store.get_session("abc") is None
    assert not store.delete_session("abc")


def test_list_sessions(store: SQLiteStateStore):
    for sid in ("a", "b", "c"):
        store.save_session(sid, {"created_at": "x", "last_active": "x", "config": {}})
    sessions = store.list_sessions()
    assert {s["session_id"] for s in sessions} == {"a", "b", "c"}


def test_messages_lifecycle(store: SQLiteStateStore):
    store.save_session("s1", {"created_at": "x", "last_active": "x", "config": {}})
    store.add_message(
        "s1",
        {"message_id": "m1", "role": "user", "content": "hi"},
    )
    store.add_message(
        "s1",
        {"message_id": "m2", "role": "assistant", "content": "hello"},
    )
    msgs = store.get_messages("s1")
    assert len(msgs) == 2
    assert msgs[0]["content"] == "hi"
    cleared = store.clear_messages("s1")
    assert cleared == 2


def test_save_and_get_plan(store: SQLiteStateStore):
    store.save_session("s1", {"created_at": "x", "last_active": "x", "config": {}})
    store.save_plan(
        "s1",
        {
            "plan_id": "p1",
            "goal": "test",
            "steps": [{"description": "do thing"}],
            "status": "active",
            "created_at": "x",
        },
    )
    plan = store.get_active_plan("s1")
    assert plan is not None
    assert plan["goal"] == "test"


def test_active_plan_returns_none_when_all_completed(store: SQLiteStateStore):
    store.save_session("s1", {"created_at": "x", "last_active": "x", "config": {}})
    store.save_plan(
        "s1",
        {"plan_id": "p1", "goal": "g", "steps": [], "status": "completed", "created_at": "x"},
    )
    assert store.get_active_plan("s1") is None
