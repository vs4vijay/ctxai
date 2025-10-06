"""
Configuration management for ctxai.
Handles .ctxai/config.json for user preferences and settings.
Respects CTXAI_HOME environment variable for custom .ctxai location.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .utils import get_config_path, get_ctxai_home


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    provider: str = "local"  # "local", "openai", "huggingface", "azure"
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
class Config:
    """Main configuration for ctxai."""

    embedding: EmbeddingConfig
    indexing: IndexConfig
    version: str = "1.0"

    # Current index metadata
    index_name: str | None = None  # Current/default index name
    index_status: str | None = None  # "indexing", "completed", "failed"
    index_files_count: int | None = None  # Number of files in index
    index_size_mb: float | None = None  # Total size of indexed files in MB
    index_chunks_count: int | None = None  # Total number of chunks
    index_last_updated: str | None = None  # ISO format timestamp

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            embedding=EmbeddingConfig(),
            indexing=IndexConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
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
        return cls(
            version=data.get("version", "1.0"),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            indexing=IndexConfig(**data.get("indexing", {})),
            index_name=data.get("index_name"),
            index_status=data.get("index_status"),
            index_files_count=data.get("index_files_count"),
            index_size_mb=data.get("index_size_mb"),
            index_chunks_count=data.get("index_chunks_count"),
            index_last_updated=data.get("index_last_updated"),
        )


class ConfigManager:
    """Manages configuration loading and saving."""

    CONFIG_FILENAME = "config.json"

    def __init__(self, project_path: Path | None = None):
        """
        Initialize config manager.

        Args:
            project_path: Optional project root path. If not provided,
                         uses CTXAI_HOME env var or current directory.
        """
        self.project_path = project_path
        self.ctxai_home = get_ctxai_home(project_path)
        self.config_path = get_config_path(project_path)
        self._config: Config | None = None

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
                with open(self.config_path, encoding="utf-8") as f:
                    data = json.load(f)
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
        Save configuration to file.

        Args:
            config: Config to save, or use currently loaded config
        """
        if config is not None:
            self._config = config

        if self._config is None:
            raise ValueError("No configuration to save")

        # Ensure directory exists
        self.ctxai_home.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._config.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config to {self.config_path}: {e}")

    def update_embedding_provider(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Update embedding provider configuration.

        Args:
            provider: Provider name ("local", "openai", etc.)
            model: Optional model name
            api_key: Optional API key
        """
        config = self.load()
        config.embedding.provider = provider
        if model is not None:
            config.embedding.model = model
        if api_key is not None:
            config.embedding.api_key = api_key
        self.save(config)

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        return self.load().embedding

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
        """
        Update index metadata in configuration.

        Args:
            index_name: Name of the index
            status: Status of the index ("indexing", "completed", "failed")
            files_count: Number of files indexed
            size_mb: Total size of indexed files in MB
            chunks_count: Total number of chunks created
        """
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
        """
        Get current index metadata.

        Returns:
            Dictionary with index metadata
        """
        config = self.load()
        return {
            "index_name": config.index_name,
            "index_status": config.index_status,
            "index_files_count": config.index_files_count,
            "index_size_mb": config.index_size_mb,
            "index_chunks_count": config.index_chunks_count,
            "index_last_updated": config.index_last_updated,
        }
