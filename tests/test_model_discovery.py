"""Tests for provider model discovery via live APIs."""

import pytest

from ctxai.agent.llm import model_discovery


@pytest.fixture(autouse=True)
def _clear_cache():
    model_discovery.clear_discovery_cache()
    yield
    model_discovery.clear_discovery_cache()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    def _install(payload, status=200):
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers))
            if callable(payload):
                return FakeResponse(payload(url, headers), status)
            return FakeResponse(payload, status)

        monkeypatch.setattr(model_discovery.requests, "get", fake_get)
        return calls

    return _install


def test_openrouter_parses_id_name_description_and_context(fake_get):
    calls = fake_get(
        {
            "data": [
                {
                    "id": "anthropic/claude-3.5-sonnet",
                    "name": "Anthropic: Claude 3.5 Sonnet",
                    "description": "Best coding",
                    "context_length": 200000,
                },
                {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o"},
            ]
        }
    )
    models = model_discovery.discover_models("openrouter")
    assert calls[0][0] == "https://openrouter.ai/api/v1/models"
    assert calls[0][1] is None  # no auth headers for public endpoint
    assert [m.id for m in models] == ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"]
    assert models[0].name == "Anthropic: Claude 3.5 Sonnet"
    assert models[0].context_length == 200000


def test_openai_without_key_skips_request(fake_get):
    calls = fake_get({})
    assert model_discovery.discover_models("openai") == []
    assert calls == []  # no HTTP request was made without a key


def test_openai_parses_with_key(monkeypatch, fake_get):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = fake_get({"data": [{"id": "gpt-4o", "object": "model"}]})
    models = model_discovery.discover_models("openai")
    assert [m.id for m in models] == ["gpt-4o"]
    assert calls[0][0] == "https://api.openai.com/v1/models"
    assert calls[0][1]["Authorization"] == "Bearer sk-test"


def test_anthropic_sends_x_api_key_and_parses_display_name(monkeypatch, fake_get):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    calls = fake_get(
        {
            "data": [
                {
                    "type": "model",
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                }
            ]
        }
    )
    models = model_discovery.discover_models("anthropic")
    assert calls[0][0] == "https://api.anthropic.com/v1/models?limit=1000"
    assert calls[0][1]["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in calls[0][1]
    assert models[0].id == "claude-3-5-sonnet-20241022"
    assert models[0].name == "Claude 3.5 Sonnet"


def test_ollama_parses_tags(fake_get):
    calls = fake_get({"models": [{"name": "codellama:13b"}, {"name": "llama3.1:8b"}]})
    models = model_discovery.discover_models("ollama")
    assert calls[0][0] == "http://localhost:11434/api/tags"
    assert [m.id for m in models] == ["codellama:13b", "llama3.1:8b"]


def test_nvidia_uses_openai_shape(fake_get):
    calls = fake_get({"object": "list", "data": [{"id": "meta/llama-3.1-405b-instruct"}]})
    models = model_discovery.discover_models("nvidia")
    assert calls[0][0] == "https://integrate.api.nvidia.com/models"
    assert models[0].id == "meta/llama-3.1-405b-instruct"


def test_nvidia_respects_configured_base_url(fake_get, tmp_path):
    (tmp_path / ".ctxai").mkdir()
    (tmp_path / ".ctxai" / "config.toml").write_text(
        'version = "2.0"\n[providers.nvidia]\nbase_url = "https://proxy.example.com/v1"\n'
    )
    from ctxai.config import ConfigManager

    calls = fake_get({"data": [{"id": "custom/model"}]})
    models = model_discovery.discover_models("nvidia", ConfigManager(tmp_path))
    assert calls[0][0] == "https://proxy.example.com/v1/models"
    assert models[0].id == "custom/model"


def test_custom_uses_config_base_url_and_key(tmp_path, fake_get):
    (tmp_path / ".ctxai").mkdir()
    (tmp_path / ".ctxai" / "config.toml").write_text(
        'version = "2.0"\n[providers.custom]\nbase_url = "https://api.example.com/v1"\napi_key = "sk-custom"\n'
    )
    from ctxai.config import ConfigManager

    calls = fake_get({"data": [{"id": "llama-3-70b"}]})
    models = model_discovery.discover_models("custom", ConfigManager(tmp_path))
    assert calls[0][0] == "https://api.example.com/v1/models"
    assert calls[0][1]["Authorization"] == "Bearer sk-custom"
    assert models[0].id == "llama-3-70b"


def test_copilot_accepts_data_and_models_shapes(monkeypatch, fake_get):
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "ghu_test")
    calls = fake_get({"models": [{"id": "gpt-4"}, {"name": "claude-3.5-sonnet"}]})
    models = model_discovery.discover_models("github-copilot")
    assert calls[0][0] == "https://api.githubcopilot.com/models"
    assert calls[0][1]["Authorization"] == "Bearer ghu_test"
    assert [m.id for m in models] == ["gpt-4", "claude-3.5-sonnet"]


def test_http_error_returns_empty(fake_get):
    fake_get({}, status=500)
    assert model_discovery.discover_models("openrouter") == []


def test_network_error_returns_empty(monkeypatch):
    def boom(url, headers=None, timeout=None):
        raise ConnectionError("refused")

    monkeypatch.setattr(model_discovery.requests, "get", boom)
    assert model_discovery.discover_models("openrouter") == []


def test_unknown_provider_returns_empty():
    assert model_discovery.discover_models("does-not-exist") == []


def test_results_are_cached_per_provider(fake_get):
    fake_get({"data": [{"id": "anthropic/claude-3.5-sonnet"}]})
    first = model_discovery.discover_models("openrouter")
    assert len(first) == 1
    # second call must not hit the network
    fake_get({"data": [{"id": "other"}]})
    second = model_discovery.discover_models("openrouter")
    assert second == first
    # different providers are cached independently
    fake_get({"data": [{"id": "meta/llama-3.1-405b-instruct"}]})
    other = model_discovery.discover_models("nvidia")
    assert other[0].id == "meta/llama-3.1-405b-instruct"
