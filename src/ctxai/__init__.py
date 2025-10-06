"""
ctxai - A semantic code search engine

Transform your codebase into intelligent embeddings for fast, context-aware code retrieval.
Available as both an MCP Server and CLI tool.

Usage:
    # Index a codebase
    ctxai index /path/to/codebase "index-name"
    
    # Start MCP server (coming soon)
    ctxai server --index "index-name"
    
    # View help
    ctxai --help

For more information, visit: https://github.com/vs4vijay/ctxai
"""

__version__ = "0.0.1"
__author__ = "vs4vijay"

from .chunking import CodeChunker, CodeChunk
from .embeddings import EmbeddingsGenerator
from .traversal import CodeTraversal
from .vector_store import VectorStore

__all__ = [
    "CodeChunker",
    "CodeChunk",
    "EmbeddingsGenerator",
    "CodeTraversal",
    "VectorStore",
]
