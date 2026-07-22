# ctxai configuration file - TOML format updated
"""
Configuration management for ctxai.
Handles .ctxai/config.toml for user preferences and settings.
Respects CTXAI_HOME environment variable for custom .ctxai location.
Supports hierarchical config: global (~/.config/ctxai/config.toml) and project (./config.toml).
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    model: str | None = None
    api_key: str | None = None
    batch_size: int = 100
    max_tokens: int | None = None


@dataclass
class IndexConfig:
    max_files: int = 10000
    max_total_size_mb: int = 500
    max_file_size_mb: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 100


@dataclass
class ProviderConfig:
    enabled: bool = True
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    base_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepomapConfig:
    enabled: bool = False
    max_tokens: int = 1000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Config:
    version: str = "2.0"
    default_provider: str = "openrouter"
    repomap: RepomapConfig = field(default_factory=RepomapConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexConfig = field(default_factory=IndexConfig)
    index_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls):
        return cls(
            version="2.0",
            default_provider="openrouter",
            repomap=RepomapConfig(),
            providers={
                "openrouter": ProviderConfig(model="anthropic/claude-3.5-sonnet", timeout=120),
                "github-copilot": ProviderConfig(model="gpt-4"),
                "ollama": ProviderConfig(model="codellama:13b", base_url="http://localhost:11434"),
                "anthropic": ProviderConfig(model="claude-3-5-sonnet-20241022"),
                "openai": ProviderConfig(model="gpt-4o"),
            },
            embedding=EmbeddingConfig(),
            indexing=IndexConfig(),
            index_metadata={},
        )
