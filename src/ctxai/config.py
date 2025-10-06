"""
Configuration management for ctxai.
Handles .ctxai/config.json for user preferences and settings.
Respects CTXAI_HOME environment variable for custom .ctxai location.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

from .utils import get_ctxai_home, get_config_path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    provider: str = "local"  # "local", "openai", "huggingface", "azure"
    model: Optional[str] = None  # Model name, provider-specific default if None
    api_key: Optional[str] = None  # API key for cloud providers
    batch_size: int = 100
    max_tokens: Optional[int] = None


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
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            embedding=EmbeddingConfig(),
            indexing=IndexConfig(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "embedding": asdict(self.embedding),
            "indexing": asdict(self.indexing),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            indexing=IndexConfig(**data.get("indexing", {})),
        )


class ConfigManager:
    """Manages configuration loading and saving."""
    
    CONFIG_FILENAME = "config.json"
    
    def __init__(self, project_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            project_path: Optional project root path. If not provided,
                         uses CTXAI_HOME env var or current directory.
        """
        self.project_path = project_path
        self.ctxai_home = get_ctxai_home(project_path)
        self.config_path = get_config_path(project_path)
        self._config: Optional[Config] = None
    
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
                with open(self.config_path, "r", encoding="utf-8") as f:
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
    
    def save(self, config: Optional[Config] = None) -> None:
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
        model: Optional[str] = None,
        api_key: Optional[str] = None,
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
