"""
Tests for query command functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ctxai.commands.query_command import query_codebase
from ctxai.config import Config, EmbeddingConfig, IndexConfig


def test_query_command_basic(tmp_path):
    """Test basic query command functionality."""
    # Create mock vector store
    mock_results = [
        {
            "metadata": {
                "file_path": "/path/to/file.py",
                "start_line": 10,
                "end_line": 20,
                "chunk_type": "function",
                "language": "python",
            },
            "content": "def test_function():\n    pass",
            "distance": 0.2,
        }
    ]

    with (
        patch("ctxai.commands.query_command.ConfigManager") as mock_config_manager,
        patch("ctxai.commands.query_command.EmbeddingsFactory") as mock_embeddings_factory,
        patch("ctxai.commands.query_command.VectorStore") as mock_vector_store,
        patch("ctxai.commands.query_command.get_indexes_dir") as mock_get_indexes_dir,
    ):
        # Setup mocks
        mock_config = Config(embedding=EmbeddingConfig(provider="local"), index=IndexConfig())
        mock_config_manager.return_value.load.return_value = mock_config

        mock_embeddings = MagicMock()
        mock_embeddings.generate_embedding.return_value = [0.1] * 384
        mock_embeddings_factory.create.return_value = mock_embeddings

        mock_store = MagicMock()
        mock_store.search.return_value = mock_results
        mock_vector_store.return_value = mock_store

        mock_indexes_dir = tmp_path / "indexes"
        mock_indexes_dir.mkdir()
        (mock_indexes_dir / "test-index").mkdir()
        mock_get_indexes_dir.return_value = mock_indexes_dir

        # Run query
        query_codebase(
            index_name="test-index",
            query="test query",
            n_results=5,
            show_content=True,
        )

        # Verify calls
        mock_embeddings.generate_embedding.assert_called_once_with("test query")
        mock_store.search.assert_called_once()


def test_query_command_no_results(tmp_path):
    """Test query command with no results."""
    with (
        patch("ctxai.commands.query_command.ConfigManager") as mock_config_manager,
        patch("ctxai.commands.query_command.EmbeddingsFactory") as mock_embeddings_factory,
        patch("ctxai.commands.query_command.VectorStore") as mock_vector_store,
        patch("ctxai.commands.query_command.get_indexes_dir") as mock_get_indexes_dir,
    ):
        # Setup mocks
        mock_config = Config(embedding=EmbeddingConfig(provider="local"), index=IndexConfig())
        mock_config_manager.return_value.load.return_value = mock_config

        mock_embeddings = MagicMock()
        mock_embeddings.generate_embedding.return_value = [0.1] * 384
        mock_embeddings_factory.create.return_value = mock_embeddings

        mock_store = MagicMock()
        mock_store.search.return_value = []  # No results
        mock_vector_store.return_value = mock_store

        mock_indexes_dir = tmp_path / "indexes"
        mock_indexes_dir.mkdir()
        (mock_indexes_dir / "test-index").mkdir()
        mock_get_indexes_dir.return_value = mock_indexes_dir

        # Run query - should not raise error
        query_codebase(
            index_name="test-index",
            query="test query",
            n_results=5,
            show_content=True,
        )


def test_query_command_index_not_found(tmp_path):
    """Test query command with non-existent index."""
    with (
        patch("ctxai.commands.query_command.ConfigManager") as mock_config_manager,
        patch("ctxai.commands.query_command.EmbeddingsFactory") as mock_embeddings_factory,
        patch("ctxai.commands.query_command.get_indexes_dir") as mock_get_indexes_dir,
    ):
        # Setup mocks
        mock_config = Config(embedding=EmbeddingConfig(provider="local"), index=IndexConfig())
        mock_config_manager.return_value.load.return_value = mock_config

        mock_embeddings = MagicMock()
        mock_embeddings_factory.create.return_value = mock_embeddings

        mock_indexes_dir = tmp_path / "indexes"
        mock_indexes_dir.mkdir()
        # Don't create the index directory
        mock_get_indexes_dir.return_value = mock_indexes_dir

        # Run query - should handle gracefully
        query_codebase(
            index_name="nonexistent-index",
            query="test query",
            n_results=5,
            show_content=True,
        )


def test_query_command_no_content(tmp_path):
    """Test query command without showing content."""
    mock_results = [
        {
            "metadata": {
                "file_path": "/path/to/file.py",
                "start_line": 10,
                "end_line": 20,
                "chunk_type": "function",
                "language": "python",
            },
            "content": "def test_function():\n    pass",
            "distance": 0.2,
        }
    ]

    with (
        patch("ctxai.commands.query_command.ConfigManager") as mock_config_manager,
        patch("ctxai.commands.query_command.EmbeddingsFactory") as mock_embeddings_factory,
        patch("ctxai.commands.query_command.VectorStore") as mock_vector_store,
        patch("ctxai.commands.query_command.get_indexes_dir") as mock_get_indexes_dir,
    ):
        # Setup mocks
        mock_config = Config(embedding=EmbeddingConfig(provider="local"), index=IndexConfig())
        mock_config_manager.return_value.load.return_value = mock_config

        mock_embeddings = MagicMock()
        mock_embeddings.generate_embedding.return_value = [0.1] * 384
        mock_embeddings_factory.create.return_value = mock_embeddings

        mock_store = MagicMock()
        mock_store.search.return_value = mock_results
        mock_vector_store.return_value = mock_store

        mock_indexes_dir = tmp_path / "indexes"
        mock_indexes_dir.mkdir()
        (mock_indexes_dir / "test-index").mkdir()
        mock_get_indexes_dir.return_value = mock_indexes_dir

        # Run query without content
        query_codebase(
            index_name="test-index",
            query="test query",
            n_results=5,
            show_content=False,  # Don't show content
        )

        # Verify embedding was generated and search was performed
        mock_embeddings.generate_embedding.assert_called_once()
        mock_store.search.assert_called_once()
