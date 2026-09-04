"""
Agent configuration classes.
"""

import os
from dataclasses import dataclass, field

SANDBOX_MODES: tuple[str, ...] = ("off", "auto", "required")
"""Valid values for ``AgentToolsConfig.sandbox`` (HH-08).

``off`` (default) preserves pre-HH-08 behavior; ``auto`` uses an OS sandbox
backend when one is available; ``required`` fails commands closed when no
backend exists.
"""


@dataclass
class AgentLLMConfig:
    """Configuration for LLM provider."""

    provider: str = "openrouter"  # "openrouter", "ollama", "anthropic", "openai"
    model: str | None = None  # Provider-specific default if None
    api_key: str | None = None  # API key, will try env vars if None
    base_url: str | None = None  # Base URL (for Ollama, custom endpoints)
    fallback_providers: list[str] = field(default_factory=lambda: ["openrouter", "ollama"])
    fallback_enabled: bool = False
    allow_fallback_boundary_crossing: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60

    def get_api_key_for_provider(self, provider: str) -> str | dict | None:
        """Get API key for specific provider, checking environment variables and keystore."""
        if self.api_key:
            return self.api_key

        # Check environment variables first
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "github-copilot": "GITHUB_COPILOT_TOKEN",
            "ollama": None,  # Ollama doesn't need API key
        }

        env_var = env_vars.get(provider.lower())
        if env_var:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key

        # If not in environment, check keystore
        try:
            from ctxai.auth.keystore import get_keystore

            keystore = get_keystore()
            return keystore.get_key(provider.lower())
        except (ImportError, Exception):
            pass

        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "fallback_providers": self.fallback_providers,
            "fallback_enabled": self.fallback_enabled,
            "allow_fallback_boundary_crossing": self.allow_fallback_boundary_crossing,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            # Note: Don't serialize API key
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentLLMConfig":
        """Create from dictionary."""
        return cls(
            provider=data.get("provider", "anthropic"),
            model=data.get("model"),
            api_key=None,  # Always load from env
            fallback_providers=data.get("fallback_providers", ["openai", "ollama"]),
            fallback_enabled=data.get("fallback_enabled", False),
            allow_fallback_boundary_crossing=data.get("allow_fallback_boundary_crossing", False),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            timeout=data.get("timeout", 60),
        )


@dataclass
class AgentToolsConfig:
    """Configuration for agent tools.

    Command policy is consolidated: the exact-name allowlist enforced by
    ``BashTool`` (``bash_allowed_commands``) plus
    ``ToolExecutionContext.approve_command`` classification are the single
    policy; there is no substring matcher. Environment exposure to subprocesses
    is allowlisted (see ``ToolExecutionContext.command_environment``).
    """

    enabled_tools: list[str] | None = None  # None = all tools enabled
    bash_allowed_commands: list[str] | None = None  # Executable allowlist (None = classification only)
    bash_timeout: int = 30
    max_file_size_mb: int = 10
    allow_outside_project: bool = False  # Allow file ops outside project dir
    max_output_chars: int = 20_000  # Cap for bash stdout/stderr and read_file content
    env_passthrough: list[str] = field(default_factory=list)  # Opt-in os.environ names for subprocesses
    sandbox: str = "off"  # OS sandbox mode (HH-08): off, auto, or required
    sandbox_network: bool = False  # Allow outbound network inside the sandbox (deny by default)

    def __post_init__(self) -> None:
        if self.sandbox not in SANDBOX_MODES:
            raise ValueError(f"sandbox must be one of: {', '.join(SANDBOX_MODES)}")

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            tool_name: Name of the tool.

        Returns:
            True when the tool is enabled.
        """
        if self.enabled_tools is None:
            return True
        return tool_name in self.enabled_tools

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the tool configuration.
        """
        return {
            "enabled_tools": self.enabled_tools,
            "bash_allowed_commands": self.bash_allowed_commands,
            "bash_timeout": self.bash_timeout,
            "max_file_size_mb": self.max_file_size_mb,
            "allow_outside_project": self.allow_outside_project,
            "max_output_chars": self.max_output_chars,
            "env_passthrough": self.env_passthrough,
            "sandbox": self.sandbox,
            "sandbox_network": self.sandbox_network,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentToolsConfig":
        """Create from dictionary.

        Unknown keys (for example the removed ``bash_blocked_commands``) are
        ignored so older serialized configurations keep loading.

        Args:
            data: Dictionary produced by ``to_dict`` or an older version.

        Returns:
            An ``AgentToolsConfig`` instance.
        """
        return cls(
            enabled_tools=data.get("enabled_tools"),
            bash_allowed_commands=data.get("bash_allowed_commands"),
            bash_timeout=data.get("bash_timeout", 30),
            max_file_size_mb=data.get("max_file_size_mb", 10),
            allow_outside_project=data.get("allow_outside_project", False),
            max_output_chars=data.get("max_output_chars", 20_000),
            env_passthrough=data.get("env_passthrough", []),
            sandbox=data.get("sandbox", "off"),
            sandbox_network=data.get("sandbox_network", False),
        )


@dataclass
class AgentBehaviorConfig:
    """Configuration for agent behavior.

    ``context_soft_limit_ratio`` is the fraction of the provider's reported
    ``context_size`` above which the loop compacts the conversation before the
    next LLM call. It bounds the *context window* budget;
    ``AgentLLMConfig.max_tokens`` remains the separate completion budget and
    the two are never conflated. ``record_runs`` controls local run
    transcripts (HH-04): on by default (local-only, redacted, nothing
    uploaded) with the oldest transcripts pruned beyond ``run_retention``.
    ``checkpoint_retention``/``checkpoint_max_bytes`` bound the local
    pre-mutation checkpoints the loop captures before structured file
    mutations (HH-06): at most ``checkpoint_retention`` checkpoint
    directories per project (oldest pruned at run start) and at most
    ``checkpoint_max_bytes`` captured bytes per run (beyond the cap captures
    stop with a diagnostic and the checkpoint stays partial).
    """

    planning_enabled: bool = True
    require_user_approval: bool = True
    max_iterations: int = 10
    auto_save_context: bool = True
    verbose: bool = False
    stream_responses: bool = True  # Stream LLM responses
    loop_break_threshold: int = 3  # Identical consecutive tool-result tuples before the loop breaks
    context_soft_limit_ratio: float = 0.8  # Compact above this fraction of the provider context_size
    record_runs: bool = True  # Write redacted local run transcripts under .ctxai/runs (HH-04)
    run_retention: int = 50  # Maximum run transcripts kept per project (oldest pruned at run start)
    checkpoint_retention: int = 20  # Maximum run checkpoints kept per project (oldest pruned at run start)
    checkpoint_max_bytes: int = 52_428_800  # Per-run capture cap in bytes (50 MB); beyond it captures stop

    def __post_init__(self) -> None:
        if not 0 < self.context_soft_limit_ratio <= 1:
            raise ValueError("context_soft_limit_ratio must be within (0, 1]")
        if self.run_retention < 1:
            raise ValueError("run_retention must be at least 1")
        if self.checkpoint_retention < 1:
            raise ValueError("checkpoint_retention must be at least 1")
        if self.checkpoint_max_bytes <= 0:
            raise ValueError("checkpoint_max_bytes must be positive")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "planning_enabled": self.planning_enabled,
            "require_user_approval": self.require_user_approval,
            "max_iterations": self.max_iterations,
            "auto_save_context": self.auto_save_context,
            "verbose": self.verbose,
            "stream_responses": self.stream_responses,
            "loop_break_threshold": self.loop_break_threshold,
            "context_soft_limit_ratio": self.context_soft_limit_ratio,
            "record_runs": self.record_runs,
            "run_retention": self.run_retention,
            "checkpoint_retention": self.checkpoint_retention,
            "checkpoint_max_bytes": self.checkpoint_max_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentBehaviorConfig":
        """Create from dictionary."""
        return cls(
            planning_enabled=data.get("planning_enabled", True),
            require_user_approval=data.get("require_user_approval", True),
            max_iterations=data.get("max_iterations", 10),
            auto_save_context=data.get("auto_save_context", True),
            verbose=data.get("verbose", False),
            stream_responses=data.get("stream_responses", True),
            loop_break_threshold=data.get("loop_break_threshold", 3),
            context_soft_limit_ratio=data.get("context_soft_limit_ratio", 0.8),
            record_runs=data.get("record_runs", True),
            run_retention=data.get("run_retention", 50),
            checkpoint_retention=data.get("checkpoint_retention", 20),
            checkpoint_max_bytes=data.get("checkpoint_max_bytes", 52_428_800),
        )


@dataclass
class AgentConfig:
    """Main agent configuration."""

    llm: AgentLLMConfig = field(default_factory=AgentLLMConfig)
    tools: AgentToolsConfig = field(default_factory=AgentToolsConfig)
    behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "llm": self.llm.to_dict(),
            "tools": self.tools.to_dict(),
            "behavior": self.behavior.to_dict(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            llm=AgentLLMConfig.from_dict(data.get("llm", {})),
            tools=AgentToolsConfig.from_dict(data.get("tools", {})),
            behavior=AgentBehaviorConfig.from_dict(data.get("behavior", {})),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def get_default(cls) -> "AgentConfig":
        """Get default configuration."""
        return cls()
