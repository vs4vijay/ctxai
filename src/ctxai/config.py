"""
Configuration management for ctxai.
Handles .ctxai/config.toml for user preferences and settings.
Respects CTXAI_HOME environment variable for custom .ctxai location.

Supports multiple LLM provider configurations with:
- default_provider: which provider to use by default
- providers: dict of provider-specific settings

TOML format only - more readable and maintainable.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomlkit

from .utils import get_ctxai_home

if TYPE_CHECKING:
    pass

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    tomllib = None


def _load_toml(path: Path) -> dict:
    """Load TOML using the standard library or the declared tomlkit fallback."""
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    return dict(tomlkit.parse(path.read_text(encoding="utf-8")))


def _save_toml(path: Path, data: dict) -> None:
    """Save TOML with the directly declared tomlkit dependency."""

    def without_none(value):
        if isinstance(value, dict):
            return {key: without_none(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [without_none(item) for item in value]
        return value

    path.write_text(tomlkit.dumps(without_none(data)), encoding="utf-8")


# ============================================================================
# Configuration Data Classes
# ============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: str = "local"  # "local", "openai", "huggingface"
    model: str | None = None  # Model name, provider-specific default if None
    api_key: str | None = None  # API key for cloud providers
    batch_size: int = 100
    max_tokens: int | None = None


@dataclass
class IndexConfig:
    """Configuration for indexing behavior."""

    max_files: int = 10000  # Maximum number of files to index
    max_total_size_mb: int = 500  # Maximum total size in MB
    max_file_size_mb: int = 5  # Maximum individual file size in MB
    chunk_size: int = 1000  # Maximum characters per chunk
    chunk_overlap: int = 100  # Overlap between chunks


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    enabled: bool = True  # Whether this provider is available
    model: str | None = None  # Default model for this provider
    api_key: str | None = None  # API key (or use env vars)
    base_url: str | None = None  # Base URL (for custom/Ollama endpoints)
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfig":
        """Create from dictionary, ignoring unknown keys."""
        return cls(
            enabled=data.get("enabled", True),
            model=data.get("model"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            timeout=data.get("timeout", 60),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Config:
    """
    Main configuration for ctxai.

    Supports multiple LLM providers with per-provider settings.
    Use default_provider to specify which provider to use by default.
    """

    # Provider configuration
    default_provider: str = "openrouter"  # Which provider to use by default
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Other configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexConfig = field(default_factory=IndexConfig)
    version: str = "2.0"

    # Index metadata
    index_name: str | None = None
    index_status: str | None = None
    index_files_count: int | None = None
    index_size_mb: float | None = None
    index_chunks_count: int | None = None
    index_last_updated: str | None = None

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration with all providers."""
        return cls(
            version="2.0",
            default_provider="openrouter",
            providers={
                "openrouter": ProviderConfig(
                    model="anthropic/claude-3.5-sonnet",
                    timeout=120,
                ),
                "github-copilot": ProviderConfig(
                    model="gpt-4",
                ),
                "ollama": ProviderConfig(
                    model="codellama:13b",
                    base_url="http://localhost:11434",
                ),
                "anthropic": ProviderConfig(
                    model="claude-3-5-sonnet-20241022",
                ),
                "openai": ProviderConfig(
                    model="gpt-4o",
                ),
                "nvidia": ProviderConfig(
                    enabled=False,
                    base_url="https://integrate.api.nvidia.com/v1",
                ),
                "custom": ProviderConfig(
                    enabled=False,  # Disabled by default, enable when configured
                ),
            },
            embedding=EmbeddingConfig(),
            indexing=IndexConfig(),
        )

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderConfig for the provider (creates default if not exists)
        """
        if provider_name not in self.providers:
            self.providers[provider_name] = ProviderConfig()
        return self.providers[provider_name]

    def set_provider_config(
        self,
        provider_name: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        """
        Set configuration for a specific provider.

        Args:
            provider_name: Name of the provider
            model: Model name
            api_key: API key
            base_url: Base URL
            temperature: Temperature
            max_tokens: Max tokens
            enabled: Whether provider is enabled
        """
        if provider_name not in self.providers:
            self.providers[provider_name] = ProviderConfig()

        config = self.providers[provider_name]
        if model is not None:
            config.model = model
        if api_key is not None:
            config.api_key = api_key
        if base_url is not None:
            config.base_url = base_url
        if temperature is not None:
            config.temperature = temperature
        if max_tokens is not None:
            config.max_tokens = max_tokens
        if enabled is not None:
            config.enabled = enabled

    def list_providers(self) -> list[str]:
        """Get list of all configured providers."""
        return list(self.providers.keys())

    def set_provider_model(self, provider_name: str, model: str) -> None:
        """
        Set the default model for a provider.

        Args:
            provider_name: Name of the provider
            model: Model name to set as default
        """
        self.set_provider_config(provider_name, model=model)

        # If this is the default provider, also update it
        if self.default_provider != provider_name:
            self.default_provider = provider_name

    def list_enabled_providers(self) -> list[str]:
        """Get list of enabled providers."""
        return [name for name, config in self.providers.items() if config.enabled]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "default_provider": self.default_provider,
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "embedding": asdict(self.embedding),
            "indexing": asdict(self.indexing),
            "index_name": self.index_name,
            "index_status": self.index_status,
            "index_files_count": self.index_files_count,
            "index_size_mb": self.index_size_mb,
            "index_chunks_count": self.index_chunks_count,
            "index_last_updated": self.index_last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create from dictionary."""
        # Handle providers dict
        providers = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = ProviderConfig.from_dict(pdata)

        return cls(
            version=data.get("version", "2.0"),
            default_provider=data.get("default_provider", "openrouter"),
            providers=providers,
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            indexing=IndexConfig(**data.get("indexing", {})),
            index_name=data.get("index_name"),
            index_status=data.get("index_status"),
            index_files_count=data.get("index_files_count"),
            index_size_mb=data.get("index_size_mb"),
            index_chunks_count=data.get("index_chunks_count"),
            index_last_updated=data.get("index_last_updated"),
        )


# ============================================================================
# Configuration Manager
# ============================================================================


class ConfigManager:
    """Manages configuration loading and saving. Supports both JSON and TOML formats."""

    def __init__(self, project_path: Path | None = None):
        """
        Initialize config manager.

        Args:
            project_path: Optional project root path. If not provided,
                         uses CTXAI_HOME env var or current directory.
        """
        self.project_path = project_path
        self.ctxai_home = get_ctxai_home(project_path)

        # Determine config path - check for both formats, prefer TOML
        self._detect_config_path()

        self._config: Config | None = None

    def _detect_config_path(self) -> None:
        """Detect existing config file or set default path (TOML only)."""
        toml_path = self.ctxai_home / "config.toml"

        # TOML only
        if toml_path.exists():
            self.config_path = toml_path
            self._format = "toml"
        else:
            # Default to TOML for new configs
            self.config_path = toml_path
            self._format = "toml"

    def get_config_path(self) -> Path:
        """Get the config file path."""
        return self.config_path

    def load(self) -> Config:
        """
        Load configuration from file or create default.

        Returns:
            Config object
        """
        if self._config is not None:
            return self._config

        if self.config_path.exists():
            try:
                # Load TOML format
                data = _load_toml(self.config_path)
                self._config = Config.from_dict(data)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
                self._config = Config.default()
        else:
            self._config = Config.default()
            self.save()  # Save default config for user reference

        return self._config

    def save(self, config: Config | None = None) -> None:
        """
        Save configuration to TOML file.

        Args:
            config: Config to save, or use currently loaded config
        """
        if config is not None:
            self._config = config

        if self._config is None:
            raise ValueError("No configuration to save")

        # Ensure directory exists
        self.ctxai_home.mkdir(parents=True, exist_ok=True)

        data = self._config.to_dict()
        _save_toml(self.config_path, data)

    # =========================================================================
    # Provider Configuration Methods
    # =========================================================================

    def get_default_provider(self) -> str:
        """Get the default provider name."""
        return self.load().default_provider

    def set_default_provider(self, provider: str) -> None:
        """Set the default provider."""
        config = self.load()
        config.default_provider = provider
        self.save(config)

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        return self.load().get_provider_config(provider_name)

    def set_provider_config(
        self,
        provider_name: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        """
        Set configuration for a specific provider.

        Args:
            provider_name: Name of the provider
            model: Model name
            api_key: API key
            base_url: Base URL
            temperature: Temperature
            max_tokens: Max tokens
            enabled: Whether provider is enabled
        """
        config = self.load()
        config.set_provider_config(
            provider_name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            enabled=enabled,
        )
        self.save(config)

    def list_providers(self) -> list[str]:
        """List all configured providers."""
        return self.load().list_providers()

    def list_enabled_providers(self) -> list[str]:
        """List enabled providers."""
        return self.load().list_enabled_providers()

    # =========================================================================
    # Embedding Configuration Methods
    # =========================================================================

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.load().embedding

    def update_embedding_provider(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Update embedding provider configuration."""
        config = self.load()
        config.embedding.provider = provider
        if model is not None:
            config.embedding.model = model
        if api_key is not None:
            config.embedding.api_key = api_key
        self.save(config)

    # =========================================================================
    # Index Configuration Methods
    # =========================================================================

    def get_index_config(self) -> IndexConfig:
        """Get indexing configuration."""
        return self.load().indexing

    def get_current_index_name(self) -> str | None:
        """Get the current/default index name."""
        return self.load().index_name

    def update_index_metadata(
        self,
        index_name: str,
        status: str,
        files_count: int | None = None,
        size_mb: float | None = None,
        chunks_count: int | None = None,
    ) -> None:
        """Update index metadata in configuration."""
        from datetime import datetime

        config = self.load()
        config.index_name = index_name
        config.index_status = status

        if files_count is not None:
            config.index_files_count = files_count
        if size_mb is not None:
            config.index_size_mb = round(size_mb, 2)
        if chunks_count is not None:
            config.index_chunks_count = chunks_count

        config.index_last_updated = datetime.utcnow().isoformat() + "Z"

        self.save(config)

    def clear_index_metadata(self) -> None:
        """Clear all index metadata from configuration."""
        config = self.load()
        config.index_name = None
        config.index_status = None
        config.index_files_count = None
        config.index_size_mb = None
        config.index_chunks_count = None
        config.index_last_updated = None
        self.save(config)

    def get_index_metadata(self) -> dict[str, Any]:
        """Get current index metadata."""
        config = self.load()
        return {
            "index_name": config.index_name,
            "index_status": config.index_status,
            "index_files_count": config.index_files_count,
            "index_size_mb": config.index_size_mb,
            "index_chunks_count": config.index_chunks_count,
            "index_last_updated": config.index_last_updated,
        }


# ============================================================================
# Legacy Compatibility - LLMConfig (deprecated)
# ============================================================================


@dataclass
class LLMConfig:
    """
    DEPRECATED: Use ProviderConfig instead.

    This class exists for backward compatibility.
    """

    provider: str = "openrouter"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096

    def to_provider_config(self) -> ProviderConfig:
        """Convert to ProviderConfig."""
        return ProviderConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
