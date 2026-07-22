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

# Lazy import mapping for performance optimization
_LAZY_IMPORTS = {
    # Chunking
    "CodeChunk": ("ctxai.chunking", "CodeChunk"),
    "CodeChunker": ("ctxai.chunking", "CodeChunker"),
    # Config
    "Config": ("ctxai.config", "Config"),
    "ConfigManager": ("ctxai.config", "ConfigManager"),
    "EmbeddingConfig": ("ctxai.config", "EmbeddingConfig"),
    "IndexConfig": ("ctxai.config", "IndexConfig"),
    # Embeddings
    "BaseEmbeddingProvider": ("ctxai.embeddings", "BaseEmbeddingProvider"),
    "EmbeddingsFactory": ("ctxai.embeddings", "EmbeddingsFactory"),
    "HuggingFaceEmbeddingProvider": ("ctxai.embeddings", "HuggingFaceEmbeddingProvider"),
    "LocalEmbeddingProvider": ("ctxai.embeddings", "LocalEmbeddingProvider"),
    "OpenAIEmbeddingProvider": ("ctxai.embeddings", "OpenAIEmbeddingProvider"),
    # Size validator
    "ProjectSizeLimitError": ("ctxai.size_validator", "ProjectSizeLimitError"),
    "ProjectSizeValidator": ("ctxai.size_validator", "ProjectSizeValidator"),
    "ProjectStats": ("ctxai.size_validator", "ProjectStats"),
    # Traversal
    "CodeTraversal": ("ctxai.traversal", "CodeTraversal"),
    # Utils
    "ensure_ctxai_home": ("ctxai.utils", "ensure_ctxai_home"),
    "get_config_path": ("ctxai.utils", "get_config_path"),
    "get_ctxai_home": ("ctxai.utils", "get_ctxai_home"),
    "get_ctxai_home_info": ("ctxai.utils", "get_ctxai_home_info"),
    "get_indexes_dir": ("ctxai.utils", "get_indexes_dir"),
    "is_using_global_home": ("ctxai.utils", "is_using_global_home"),
    # Vector store
    "VectorStore": ("ctxai.vector_store", "VectorStore"),
}


def __getattr__(name: str):
    """Lazy import attributes to improve startup performance."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
