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
    ProviderSpec("anthropic", "Anthropic Messages", False, ProviderCapabilities(), "API/static"),
    ProviderSpec("openai", "OpenAI Chat Completions", False, ProviderCapabilities(), "API/static"),
    ProviderSpec("openrouter", "OpenAI-compatible", False, ProviderCapabilities(), "API/cached"),
    ProviderSpec("github-copilot", "Copilot Chat", False, ProviderCapabilities(), "API/cached"),
    ProviderSpec("ollama", "Ollama Chat", True, ProviderCapabilities(), "dynamic/local"),
    ProviderSpec("custom", "OpenAI-compatible", False, ProviderCapabilities(), "endpoint-defined"),
    ProviderSpec("nvidia", "OpenAI-compatible", False, ProviderCapabilities(), "endpoint-defined"),
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
    lines.extend([
        "",
        "Fallback is disabled by default. Crossing the local/cloud boundary requires the explicit "
        "`allow_fallback_boundary_crossing` setting.",
        "",
    ])
    return "\n".join(lines)
