"""VS-07 acceptance coverage for the provider-independent execution contract."""

from __future__ import annotations

import inspect
from pathlib import Path
from threading import Event

import pytest

from ctxai.agent.config import AgentLLMConfig
from ctxai.agent.llm.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    MessageRole,
    ProviderCapabilities,
    ProviderError,
    ProviderErrorKind,
)
from ctxai.agent.llm.contract import PROVIDER_SPECS, render_compatibility_matrix
from ctxai.agent.llm.fallback import FallbackProvider


class ContractProvider(BaseLLMProvider):
    def __init__(self, name: str, *, fail: bool = False, tools: bool = True):
        self.name = name
        self.fail = fail
        self.tools = tools
        self.calls = 0
        super().__init__(AgentLLMConfig(provider=name, model=f"{name}-model"))

    def get_default_model(self) -> str:
        return f"{self.name}-model"

    def chat(self, messages, tools=None, **kwargs):
        self.validate_request(messages, tools, cancel_event=kwargs.get("cancel_event"))
        self.calls += 1
        if self.fail:
            raise TimeoutError(f"{self.name} timed out")
        return LLMResponse(content=f"answer from {self.name}", usage={"total_tokens": 3})

    def stream_chat(self, messages, tools=None, **kwargs):
        self.validate_request(messages, tools, stream=True)
        yield f"answer from {self.name}"

    def supports_function_calling(self) -> bool:
        return self.tools

    def requires_api_key(self) -> bool:
        return False

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(tools=self.tools, streaming=True)


MESSAGES = [Message(MessageRole.USER, "Explain the repository")]
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]


@pytest.mark.parametrize("provider_name", [spec.name for spec in PROVIDER_SPECS])
def test_every_advertised_provider_passes_non_network_contract(provider_name: str) -> None:
    provider = ContractProvider(provider_name)
    response = provider.chat(MESSAGES, tools=TOOLS)
    assert response.content == f"answer from {provider_name}"
    assert response.usage["total_tokens"] == 3
    assert "".join(provider.stream_chat(MESSAGES, tools=TOOLS)) == response.content
    assert provider.normalize_messages(MESSAGES)[0] == {
        "role": "user",
        "content": "Explain the repository",
    }


def test_all_concrete_provider_classes_satisfy_the_shared_interface() -> None:
    from ctxai.agent.llm.anthropic_provider import AnthropicProvider
    from ctxai.agent.llm.custom_provider import CustomProvider
    from ctxai.agent.llm.github_copilot_provider import GitHubCopilotProvider
    from ctxai.agent.llm.ollama_provider import OllamaProvider
    from ctxai.agent.llm.openai_provider import OpenAIProvider
    from ctxai.agent.llm.openrouter_provider import OpenRouterProvider

    classes = [
        AnthropicProvider,
        CustomProvider,
        GitHubCopilotProvider,
        OllamaProvider,
        OpenAIProvider,
        OpenRouterProvider,
    ]
    assert not [provider.__name__ for provider in classes if inspect.isabstract(provider)]


def test_unsupported_tools_and_cancellation_fail_before_transport() -> None:
    provider = ContractProvider("custom", tools=False)
    with pytest.raises(ProviderError) as unsupported:
        provider.chat(MESSAGES, tools=TOOLS)
    assert unsupported.value.kind is ProviderErrorKind.UNSUPPORTED
    assert provider.calls == 0

    cancelled = Event()
    cancelled.set()
    with pytest.raises(ProviderError) as cancellation:
        provider.chat(MESSAGES, cancel_event=cancelled)
    assert cancellation.value.kind is ProviderErrorKind.CANCELLED
    assert provider.calls == 0


def test_fallback_is_observable_and_never_silently_crosses_boundary() -> None:
    local = ContractProvider("ollama", fail=True)
    cloud = ContractProvider("openai")
    fallback = FallbackProvider([("ollama", local), ("openai", cloud)])
    with pytest.raises(ProviderError) as failure:
        fallback.chat(MESSAGES)
    assert failure.value.kind is ProviderErrorKind.TIMEOUT
    assert cloud.calls == 0
    assert [(attempt.provider, attempt.outcome) for attempt in fallback.attempts] == [
        ("ollama", "failed"),
        ("openai", "blocked_boundary"),
    ]

    allowed = FallbackProvider(
        [("ollama", ContractProvider("ollama", fail=True)), ("openai", cloud)],
        allow_boundary_crossing=True,
    )
    assert allowed.chat(MESSAGES).content == "answer from openai"
    assert [(attempt.provider, attempt.outcome) for attempt in allowed.attempts] == [
        ("ollama", "failed"),
        ("openai", "success"),
    ]


def test_fallback_requires_opt_in_and_configuration_round_trips() -> None:
    config = AgentLLMConfig.from_dict(
        {
            "provider": "ollama",
            "fallback_providers": ["openai"],
            "fallback_enabled": True,
            "allow_fallback_boundary_crossing": True,
        }
    )
    assert config.fallback_enabled is True
    assert config.allow_fallback_boundary_crossing is True
    assert AgentLLMConfig().fallback_enabled is False
    assert config.to_dict()["fallback_providers"] == ["openai"]


def test_published_compatibility_matrix_is_generated_from_contract() -> None:
    documented = Path("docs/PROVIDER_COMPATIBILITY.md").read_text(encoding="utf-8")
    assert documented == render_compatibility_matrix()
    assert {spec.name for spec in PROVIDER_SPECS} == {
        "anthropic",
        "openai",
        "openrouter",
        "github-copilot",
        "ollama",
        "custom",
        "nvidia",
    }
