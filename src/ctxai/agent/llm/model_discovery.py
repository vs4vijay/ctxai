"""
Model discovery via provider APIs.

Fetches the live model list from each provider's discovery endpoint with a
short timeout, per-process caching, and graceful degradation: any failure
returns an empty list so callers can fall back to the static catalog.

Supported discovery endpoints (researched 2026-08):
- openrouter:  GET https://openrouter.ai/api/v1/models        (public)
- openai:      GET https://api.openai.com/v1/models           (Bearer key)
- anthropic:   GET https://api.anthropic.com/v1/models        (x-api-key, paginated)
- ollama:      GET http://localhost:11434/api/tags            (public)
- nvidia:      GET <base_url>/models                          (OpenAI-compatible)
- github-copilot: GET https://api.githubcopilot.com/models    (Bearer token, undocumented)
- custom:      GET <base_url>/models                          (OpenAI-compatible)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests

DISCOVERY_TIMEOUT = 6.0
CACHE_TTL_SECONDS = 300

# provider -> (env var, keystore key) for API-key resolution
_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "github-copilot": "GITHUB_COPILOT_TOKEN",
    "nvidia": "NVIDIA_API_KEY",
    "custom": "CUSTOM_API_KEY",
}

_CACHE: dict[str, tuple[float, list[DiscoveredModel]]] = {}


@dataclass
class DiscoveredModel:
    """A model returned by a provider discovery API."""

    id: str
    name: str | None = None
    description: str | None = None
    context_length: int | None = None


def _get_api_key(provider: str) -> str | None:
    """Resolve an API key: env var first, then the keystore."""
    env_name = _API_KEY_ENV.get(provider)
    if env_name:
        key = os.getenv(env_name)
        if key:
            return key
    try:
        from ...auth.keystore import get_keystore

        return get_keystore().get_key(provider)
    except Exception:
        return None


def _get_json(url: str, headers: dict | None = None) -> dict:
    """GET a JSON payload; returns {} on any failure."""
    try:
        resp = requests.get(url, headers=headers, timeout=DISCOVERY_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _from_openai_shape(data: dict) -> list[DiscoveredModel]:
    """Parse an OpenAI-compatible ``{"data": [{id, ...}]}`` response."""
    models = []
    for m in data.get("data", []):
        mid = m.get("id")
        if mid:
            models.append(DiscoveredModel(id=mid, name=m.get("name") or mid))
    return models


def _fetch_openrouter(_cm=None) -> list[DiscoveredModel]:
    data = _get_json("https://openrouter.ai/api/v1/models")
    models = []
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        models.append(
            DiscoveredModel(
                id=mid,
                name=m.get("name") or mid,
                description=m.get("description"),
                context_length=m.get("context_length"),
            )
        )
    return models


def _fetch_openai(_cm=None) -> list[DiscoveredModel]:
    key = _get_api_key("openai")
    if not key:
        return []
    data = _get_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    return _from_openai_shape(data)


def _fetch_anthropic(_cm=None) -> list[DiscoveredModel]:
    key = _get_api_key("anthropic")
    if not key:
        return []
    data = _get_json(
        "https://api.anthropic.com/v1/models?limit=1000",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    models = []
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        models.append(DiscoveredModel(id=mid, name=m.get("display_name") or mid, description=m.get("display_name")))
    return models


def _fetch_ollama(_cm=None) -> list[DiscoveredModel]:
    data = _get_json("http://localhost:11434/api/tags")
    models = []
    for m in data.get("models", []):
        mid = m.get("name") or m.get("model")
        if mid:
            models.append(DiscoveredModel(id=mid))
    return models


def _fetch_nvidia(cm=None) -> list[DiscoveredModel]:
    base = "https://integrate.api.nvidia.com"
    if cm is not None:
        try:
            base = cm.load().get_provider_config("nvidia").base_url or base
        except Exception:
            pass
    data = _get_json(base.rstrip("/") + "/models")
    return _from_openai_shape(data)


def _fetch_copilot(_cm=None) -> list[DiscoveredModel]:
    key = _get_api_key("github-copilot")
    if not key:
        return []
    data = _get_json(
        "https://api.githubcopilot.com/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    entries = data.get("data") or data.get("models") or []
    models = []
    for m in entries:
        mid = m.get("id") or m.get("name") if isinstance(m, dict) else None
        if mid:
            models.append(DiscoveredModel(id=mid))
    return models


def _fetch_custom(cm=None) -> list[DiscoveredModel]:
    base = None
    key = _get_api_key("custom")
    if cm is not None:
        try:
            pconfig = cm.load().get_provider_config("custom")
            base = pconfig.base_url
            if pconfig.api_key:
                key = pconfig.api_key
        except Exception:
            pass
    if not base:
        base = os.getenv("CUSTOM_BASE_URL") or os.getenv("MODAL_BASE_URL")
    if not base:
        return []
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = _get_json(base.rstrip("/") + "/models", headers=headers)
    return _from_openai_shape(data)


_FETCHERS = {
    "openrouter": _fetch_openrouter,
    "openai": _fetch_openai,
    "anthropic": _fetch_anthropic,
    "ollama": _fetch_ollama,
    "nvidia": _fetch_nvidia,
    "github-copilot": _fetch_copilot,
    "custom": _fetch_custom,
}


def discover_models(provider: str, config_manager=None) -> list[DiscoveredModel]:
    """
    Fetch the live model list for a provider (cached per process).

    Args:
        provider: Provider name (openrouter, openai, anthropic, ollama,
            nvidia, github-copilot, custom)
        config_manager: Optional ConfigManager used to resolve base URLs
            for nvidia/custom providers

    Returns:
        List of discovered models; empty on any failure or unknown provider
    """
    now = time.monotonic()
    cached = _CACHE.get(provider)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    fetcher = _FETCHERS.get(provider)
    models = fetcher(config_manager) if fetcher else []
    _CACHE[provider] = (now, models)
    return models


def clear_discovery_cache() -> None:
    """Drop the in-process model cache (used by tests)."""
    _CACHE.clear()
