"""Explicit, observable provider fallback execution."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from .base import BaseLLMProvider, LLMResponse, Message, ProviderError, ProviderErrorKind
from .contract import get_provider_spec


@dataclass(frozen=True)
class FallbackAttempt:
    provider: str
    outcome: str
    error_kind: str | None = None


class FallbackProvider(BaseLLMProvider):
    """Try an explicit provider sequence while enforcing privacy boundaries."""

    def __init__(
        self,
        providers: list[tuple[str, BaseLLMProvider]],
        *,
        allow_boundary_crossing: bool = False,
    ):
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers
        self.allow_boundary_crossing = allow_boundary_crossing
        self.attempts: list[FallbackAttempt] = []
        super().__init__(providers[0][1].config)

    def get_default_model(self) -> str:
        return self.providers[0][1].model

    def supports_function_calling(self) -> bool:
        return all(provider.supports_function_calling() for _, provider in self.providers)

    def requires_api_key(self) -> bool:
        return self.providers[0][1].requires_api_key()

    def _eligible(self):
        primary_local = get_provider_spec(self.providers[0][0]).local
        for name, provider in self.providers:
            if get_provider_spec(name).local != primary_local and not self.allow_boundary_crossing:
                self.attempts.append(FallbackAttempt(name, "blocked_boundary"))
                continue
            yield name, provider

    def chat(self, messages: list[Message], tools=None, **kwargs) -> LLMResponse:
        last_error: ProviderError | None = None
        for name, provider in self._eligible():
            try:
                response = provider.chat(messages, tools=tools, **kwargs)
                self.attempts.append(FallbackAttempt(name, "success"))
                return response
            except Exception as error:
                last_error = provider.normalize_error(error)
                self.attempts.append(FallbackAttempt(name, "failed", last_error.kind.value))
        if last_error:
            raise last_error
        raise ProviderError(
            ProviderErrorKind.UNSUPPORTED,
            "No eligible fallback provider; local/cloud boundary crossing is disabled",
        )

    def stream_chat(self, messages: list[Message], tools=None, **kwargs) -> Generator[str, None, None]:
        last_error: ProviderError | None = None
        for name, provider in self._eligible():
            try:
                yield from provider.stream_chat(messages, tools=tools, **kwargs)
                self.attempts.append(FallbackAttempt(name, "success"))
                return
            except Exception as error:
                last_error = provider.normalize_error(error)
                self.attempts.append(FallbackAttempt(name, "failed", last_error.kind.value))
        if last_error:
            raise last_error
        raise ProviderError(
            ProviderErrorKind.UNSUPPORTED,
            "No eligible fallback provider; local/cloud boundary crossing is disabled",
        )
