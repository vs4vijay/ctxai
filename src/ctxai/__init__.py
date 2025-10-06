"""
ctxai - A semantic code search engine

Transform your codebase into intelligent embeddings for fast, context-aware code retrieval.
Available as both an MCP Server and CLI tool.

Usage:
    # Index a codebase (uses local embeddings by default)
    ctxai index /path/to/codebase "index-name"

    # Configure embedding provider (edit .ctxai/config.json)
    {
      "embedding": {
        "provider": "local",  # or "openai", "huggingface"
        "model": "all-MiniLM-L6-v2"
      }
    }

    # Start MCP server (coming soon)
    ctxai server --index "index-name"

    # View help
    ctxai --help

For more information, visit: https://github.com/vs4vijay/ctxai
"""

__version__ = "0.0.1"
__author__ = "vs4vijay"

from .chunking import CodeChunk, CodeChunker
from .config import Config, ConfigManager, EmbeddingConfig, IndexConfig
from .embeddings import (
    BaseEmbeddingProvider,
    EmbeddingsFactory,
    HuggingFaceEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from .size_validator import ProjectSizeLimitError, ProjectSizeValidator, ProjectStats
from .traversal import CodeTraversal
from .utils import (
    ensure_ctxai_home,
    get_config_path,
    get_ctxai_home,
    get_ctxai_home_info,
    get_indexes_dir,
    is_using_global_home,
)
from .vector_store import VectorStore

__all__ = [
    "CodeChunker",
    "CodeChunk",
    "Config",
    "ConfigManager",
    "EmbeddingConfig",
    "IndexConfig",
    "BaseEmbeddingProvider",
    "EmbeddingsFactory",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "ProjectSizeValidator",
    "ProjectStats",
    "ProjectSizeLimitError",
    "CodeTraversal",
    "get_ctxai_home",
    "get_indexes_dir",
    "get_config_path",
    "ensure_ctxai_home",
    "is_using_global_home",
    "get_ctxai_home_info",
    "VectorStore",
]
