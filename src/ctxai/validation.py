"""
Configuration validation utilities.

Validates agent/service configuration before runtime, returning a list of
errors rather than raising on the first failure — callers can decide how
to handle aggregated results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .security import SecurityManager


@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


class ConfigValidator:
    """Validates ctxai configuration objects."""

    KNOWN_PROVIDERS = {"openrouter", "openai", "anthropic", "ollama", "github-copilot", "custom"}

    def __init__(self, security: SecurityManager | None = None) -> None:
        self.security = security or SecurityManager()

    def validate_agent_config(self, config: Any) -> list[ValidationIssue]:
        """
        Validate an AgentConfig-like object. Accepts duck-typed objects
        with `.llm`, `.tools`, `.behavior` attributes.
        """
        issues: list[ValidationIssue] = []
        issues.extend(self.validate_llm(getattr(config, "llm", None)))
        issues.extend(self.validate_tools(getattr(config, "tools", None)))
        issues.extend(self.validate_behavior(getattr(config, "behavior", None)))
        return issues

    def validate_llm(self, llm_cfg: Any) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if llm_cfg is None:
            issues.append(ValidationIssue("llm", "Missing LLM configuration"))
            return issues

        provider = getattr(llm_cfg, "provider", None)
        if not provider:
            issues.append(ValidationIssue("llm.provider", "Provider is required"))
        elif provider not in self.KNOWN_PROVIDERS:
            issues.append(
                ValidationIssue(
                    "llm.provider",
                    f"Unknown provider '{provider}'. Known: {sorted(self.KNOWN_PROVIDERS)}",
                    severity="warning",
                )
            )

        api_key = getattr(llm_cfg, "api_key", None)
        if api_key and not SecurityManager.is_valid_api_key_shape(api_key, provider):
            issues.append(
                ValidationIssue("llm.api_key", "API key does not match expected shape")
            )

        temperature = getattr(llm_cfg, "temperature", None)
        if temperature is not None and not 0.0 <= float(temperature) <= 2.0:
            issues.append(
                ValidationIssue("llm.temperature", f"Out of range [0,2]: {temperature}")
            )

        max_tokens = getattr(llm_cfg, "max_tokens", None)
        if max_tokens is not None and (max_tokens < 1 or max_tokens > 1_000_000):
            issues.append(
                ValidationIssue("llm.max_tokens", f"Out of range [1, 1_000_000]: {max_tokens}")
            )

        timeout = getattr(llm_cfg, "timeout", None)
        if timeout is not None and timeout < 1:
            issues.append(ValidationIssue("llm.timeout", "Must be >= 1 second"))

        return issues

    def validate_tools(self, tools_cfg: Any) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if tools_cfg is None:
            return issues
        bash_timeout = getattr(tools_cfg, "bash_timeout", None)
        if bash_timeout is not None and bash_timeout < 1:
            issues.append(ValidationIssue("tools.bash_timeout", "Must be >= 1"))
        max_file_mb = getattr(tools_cfg, "max_file_size_mb", None)
        if max_file_mb is not None and max_file_mb < 1:
            issues.append(ValidationIssue("tools.max_file_size_mb", "Must be >= 1"))
        return issues

    def validate_behavior(self, behavior_cfg: Any) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if behavior_cfg is None:
            return issues
        max_iter = getattr(behavior_cfg, "max_iterations", None)
        if max_iter is not None and (max_iter < 1 or max_iter > 100):
            issues.append(
                ValidationIssue("behavior.max_iterations", "Must be between 1 and 100")
            )
        return issues

    def validate_paths(
        self, paths: dict[str, str | Path], base_dir: Path | None = None
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for label, raw_path in paths.items():
            try:
                self.security.validate_file_path(raw_path, base_dir=base_dir)
            except Exception as exc:
                issues.append(ValidationIssue(label, str(exc)))
        return issues


def assert_valid(issues: list[ValidationIssue]) -> None:
    """Raise ValueError if any issue is an error (warnings are tolerated)."""
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        joined = "\n".join(str(i) for i in errors)
        raise ValueError(f"Configuration is invalid:\n{joined}")
