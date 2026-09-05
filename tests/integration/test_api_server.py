"""Integration tests for the FastAPI server."""

from pathlib import Path

import pytest

# Skip the whole module if FastAPI isn't available.
fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from ctxai.service.api_server import APIConfig, create_app  # noqa: E402
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response  # noqa: E402


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    # Patch the LLM factory so session creation works offline.
    from ctxai.service import session_manager as sm

    def _factory_create(_config):
        return MockLLMProvider(responses=[create_mock_response(content="hi-from-mock")] * 5)

    monkeypatch.setattr(sm.LLMProviderFactory, "create_provider", staticmethod(_factory_create))
    cfg = APIConfig(state_db=str(tmp_path / "state.db"))
    app = create_app(cfg)
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"healthy", "degraded", "unhealthy"}


def test_info_endpoint(client):
    r = client.get("/api/v1/info")
    assert r.status_code == 200
    assert r.json()["name"] == "ctxai"


def test_metrics_endpoint(client):
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    assert "ctxai_uptime_seconds" in r.text


def test_session_lifecycle(client):
    r = client.post("/api/v1/sessions", json={"metadata": {"label": "test"}})
    assert r.status_code == 201
    session_id = r.json()["session_id"]

    r = client.get(f"/api/v1/sessions/{session_id}")
    assert r.status_code == 200
    assert r.json()["metadata"]["label"] == "test"

    r = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert r.json()["response"]

    r = client.get(f"/api/v1/sessions/{session_id}/messages")
    assert r.status_code == 200
    assert len(r.json()["messages"]) >= 2

    r = client.delete(f"/api/v1/sessions/{session_id}")
    assert r.status_code == 200


def test_unknown_session_404(client):
    r = client.get("/api/v1/sessions/does-not-exist")
    assert r.status_code == 404
