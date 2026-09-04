"""Provider metadata and the generated compatibility contract."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ProviderCapabilities


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    transport: str
    local: bool
    capabilities: ProviderCapabilities
    models: str


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    # `streaming` means event streaming (stream_chat_events) with tool-call
    # support (HH-05). Providers without it run the documented buffered
    # fallback in the agent loop; the anthropic/openai/openrouter transports
    # implement real token-delta streaming.
    ProviderSpec("anthropic", "Anthropic Messages", False, ProviderCapabilities(streaming=True), "API/static"),
    ProviderSpec("openai", "OpenAI Chat Completions", False, ProviderCapabilities(streaming=True), "API/static"),
    ProviderSpec("openrouter", "OpenAI-compatible", False, ProviderCapabilities(streaming=True), "API/cached"),
    ProviderSpec("github-copilot", "Copilot Chat", False, ProviderCapabilities(streaming=False), "API/cached"),
    ProviderSpec("ollama", "Ollama Chat", True, ProviderCapabilities(streaming=False), "dynamic/local"),
    ProviderSpec("custom", "OpenAI-compatible", False, ProviderCapabilities(streaming=False), "endpoint-defined"),
    ProviderSpec("nvidia", "OpenAI-compatible", False, ProviderCapabilities(streaming=False), "endpoint-defined"),
)


def get_provider_spec(name: str) -> ProviderSpec:
    normalized = name.lower()
    for spec in PROVIDER_SPECS:
        if spec.name == normalized:
            return spec
    raise ValueError(f"Unknown provider: {name}")


def render_compatibility_matrix() -> str:
    """Generate the public matrix from executable provider metadata."""
    lines = [
        "# Provider compatibility",
        "",
        "Generated from `ctxai.agent.llm.contract.PROVIDER_SPECS`.",
        "",
        "| Provider | Boundary | Transport | Tools | Streaming | Models |",
        "|---|---|---|---:|---:|---|",
    ]
    for spec in PROVIDER_SPECS:
        lines.append(
            f"| {spec.name} | {'local' if spec.local else 'cloud'} | {spec.transport} | "
            f"{'yes' if spec.capabilities.tools else 'no'} | "
            f"{'yes' if spec.capabilities.streaming else 'no'} | {spec.models} |"
        )
    lines.extend(
        [
            "",
            "Fallback is disabled by default. Crossing the local/cloud boundary requires the explicit "
            "`allow_fallback_boundary_crossing` setting.",
            "",
        ]
    )
    return "\n".join(lines)
