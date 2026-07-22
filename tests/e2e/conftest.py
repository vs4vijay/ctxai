"""
E2E-specific pytest fixtures.

This module provides fixtures for end-to-end testing that require
more complex setup (indexed codebases, configured agents, etc.).
"""

from unittest.mock import patch

import pytest

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.chunking import CodeChunker
from ctxai.traversal import CodeTraversal
from ctxai.vector_store import VectorStore
from tests.mocks.mock_llm import MockLLMProvider


@pytest.fixture
def indexed_codebase(sample_python_code, temp_dir, mock_embeddings):
    """
    Create a fully indexed codebase for testing.

    This fixture performs the complete indexing workflow:
    - Traverses files
    - Chunks code with tree-sitter
    - Generates mock embeddings
    - Stores in vector database

    Args:
        sample_python_code: Sample Python codebase fixture
        temp_dir: Temporary directory fixture
        mock_embeddings: Mock embedding provider fixture

    Returns:
        dict: Dictionary containing:
            - index_path: Path to index storage
            - vector_store: VectorStore instance
            - chunks: List of CodeChunk objects
            - codebase_path: Path to original codebase
    """
    # Traverse and chunk the codebase
    traversal = CodeTraversal(sample_python_code, include_patterns=["*.py"], follow_gitignore=False)
    chunker = CodeChunker()

    all_chunks = []
    for file_path in traversal.traverse():
        try:
            chunks = chunker.chunk_file(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            # Skip files that can't be chunked
            print(f"Warning: Failed to chunk {file_path}: {e}")
            continue

    if not all_chunks:
        raise ValueError("No chunks created from sample codebase")

    # Generate embeddings
    texts = [chunk.content for chunk in all_chunks]
    embeddings = mock_embeddings.generate_embeddings(texts)

    # Create vector store in temp directory
    index_path = temp_dir / ".ctxai" / "indexes" / "test-index"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    vector_store = VectorStore(index_path.parent, "test-index")

    # Add chunks to vector store
    vector_store.add_chunks(all_chunks, embeddings)

    return {
        "index_path": index_path,
        "vector_store": vector_store,
        "chunks": all_chunks,
        "codebase_path": sample_python_code,
        "embeddings": embeddings,
    }


@pytest.fixture
def mock_llm_config():
    """
    Create a mock LLM configuration for agent testing.

    Returns:
        AgentLLMConfig: Configuration for mock LLM provider
    """
    return AgentLLMConfig(
        provider="mock", model="mock-model-v1", api_key="mock-key", temperature=0.7, max_tokens=4096, timeout=30
    )


@pytest.fixture
def mock_agent_config():
    """
    Create a mock agent configuration.

    Returns:
        AgentConfig: Configuration for agent
    """
    return AgentConfig()


@pytest.fixture
def mock_llm_provider(mock_llm_config):
    """
    Create a mock LLM provider with default configuration.

    Args:
        mock_llm_config: Mock LLM configuration fixture

    Returns:
        MockLLMProvider: Initialized mock LLM provider
    """
    return MockLLMProvider(config=mock_llm_config)


@pytest.fixture
def mock_llm_provider_with_responses(mock_llm_config):
    """
    Factory fixture to create mock LLM provider with custom responses.

    Returns:
        Callable: Function that creates MockLLMProvider with given responses

    Example:
        def test_example(mock_llm_provider_with_responses):
            provider = mock_llm_provider_with_responses([
                {"content": "Hello!", "tool_calls": []},
                {"content": "Done."}
            ])
    """

    def _create(responses):
        return MockLLMProvider(config=mock_llm_config, responses=responses)

    return _create


@pytest.fixture
def patch_embeddings_factory(mock_embeddings):
    """
    Patch the EmbeddingsFactory to return mock embeddings.

    This fixture automatically patches the factory for the test duration.

    Args:
        mock_embeddings: Mock embedding provider fixture

    Yields:
        MockEmbeddingProvider: The mock provider that will be returned
    """
    from ctxai import embeddings

    def mock_create(config):
        return mock_embeddings

    with patch.object(embeddings.EmbeddingsFactory, "create", side_effect=mock_create):
        yield mock_embeddings


@pytest.fixture
def clean_indexes_dir(temp_dir):
    """
    Provide a clean indexes directory for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to indexes directory
    """
    indexes_dir = temp_dir / ".ctxai" / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    return indexes_dir


@pytest.fixture
def sample_tool_definitions():
    """
    Provide sample tool definitions for agent testing.

    Returns:
        List[dict]: List of tool definition dictionaries
    """
    return [
        {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file to read"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string", "description": "Directory path to list"}},
                "required": ["directory"],
            },
        },
    ]
