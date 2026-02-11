"""
Agent configuration classes.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentLLMConfig:
    """Configuration for LLM provider."""

    provider: str = "openrouter"  # "openrouter", "ollama", "anthropic", "openai"
    model: str | None = None  # Provider-specific default if None
    api_key: str | None = None  # API key, will try env vars if None
    base_url: str | None = None  # Base URL (for Ollama, custom endpoints)
    fallback_providers: list[str] = field(default_factory=lambda: ["openrouter", "ollama"])
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
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            timeout=data.get("timeout", 60),
        )


@dataclass
class AgentToolsConfig:
    """Configuration for agent tools."""

    enabled_tools: list[str] | None = None  # None = all tools enabled
    bash_allowed_commands: list[str] | None = None  # Whitelist (None = use blacklist)
    bash_blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "dd if=",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "chmod -R 777",
        "> /dev/sda",
        "mv / /dev/null",
    ])
    bash_timeout: int = 30
    max_file_size_mb: int = 10
    allow_outside_project: bool = False  # Allow file ops outside project dir

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        if self.enabled_tools is None:
            return True
        return tool_name in self.enabled_tools

    def is_bash_command_allowed(self, command: str) -> bool:
        """Check if a bash command is allowed."""
        # If whitelist exists, command must be in it
        if self.bash_allowed_commands is not None:
            return any(cmd in command for cmd in self.bash_allowed_commands)

        # Otherwise, check blacklist
        command_lower = command.lower().strip()
        for blocked in self.bash_blocked_commands:
            if blocked.lower() in command_lower:
                return False

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "enabled_tools": self.enabled_tools,
            "bash_allowed_commands": self.bash_allowed_commands,
            "bash_blocked_commands": self.bash_blocked_commands,
            "bash_timeout": self.bash_timeout,
            "max_file_size_mb": self.max_file_size_mb,
            "allow_outside_project": self.allow_outside_project,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentToolsConfig":
        """Create from dictionary."""
        return cls(
            enabled_tools=data.get("enabled_tools"),
            bash_allowed_commands=data.get("bash_allowed_commands"),
            bash_blocked_commands=data.get("bash_blocked_commands", cls().bash_blocked_commands),
            bash_timeout=data.get("bash_timeout", 30),
            max_file_size_mb=data.get("max_file_size_mb", 10),
            allow_outside_project=data.get("allow_outside_project", False),
        )


@dataclass
class AgentBehaviorConfig:
    """Configuration for agent behavior."""

    planning_enabled: bool = True
    require_user_approval: bool = True
    max_iterations: int = 10
    auto_save_context: bool = True
    verbose: bool = False
    stream_responses: bool = True  # Stream LLM responses

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "planning_enabled": self.planning_enabled,
            "require_user_approval": self.require_user_approval,
            "max_iterations": self.max_iterations,
            "auto_save_context": self.auto_save_context,
            "verbose": self.verbose,
            "stream_responses": self.stream_responses,
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
