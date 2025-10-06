"""
Example script demonstrating how to use ctxai programmatically.
"""

from pathlib import Path

from ctxai.chunking import CodeChunker
from ctxai.config import ConfigManager, EmbeddingConfig
from ctxai.embeddings import EmbeddingsFactory
from ctxai.traversal import CodeTraversal
from ctxai.utils import get_ctxai_home, get_indexes_dir
from ctxai.vector_store import VectorStore


def example_index_with_local_embeddings():
    """Example: Index a codebase with local embeddings (no API key needed)."""

    # 1. Set up traversal
    codebase_path = Path("./your-project")
    traversal = CodeTraversal(
        root_path=codebase_path,
        include_patterns=["*.py", "*.js"],  # Optional: filter by patterns
        follow_gitignore=True,  # Respect .gitignore
    )

    # 2. Set up chunker
    chunker = CodeChunker(
        max_chunk_size=1000,  # Maximum characters per chunk
        overlap=100,  # Overlap between chunks
    )

    # 3. Traverse and chunk
    all_chunks = []
    for file_path in traversal.traverse():
        chunks = chunker.chunk_file(file_path)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from codebase")

    # 4. Create embedding config for local provider (default)
    embedding_config = EmbeddingConfig(
        provider="local",
        model="all-MiniLM-L6-v2",  # Good balance of speed and quality
    )

    # 5. Generate embeddings
    embeddings_gen = EmbeddingsFactory.create(embedding_config)
    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embeddings_gen.generate_embeddings(chunk_texts)

    print(f"Generated {len(embeddings)} embeddings")

    # 6. Store in vector database (respects CTXAI_HOME)
    indexes_dir = get_indexes_dir(codebase_path)
    storage_path = indexes_dir / "my-index"
    vector_store = VectorStore(
        storage_path=storage_path,
        collection_name="my-index",
    )
    vector_store.add_chunks(all_chunks, embeddings)

    print("✓ Indexing complete!")

    # 7. Get stats
    stats = vector_store.get_stats()
    print("\nIndex stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique files: {stats['unique_files']}")
    print(f"  Languages: {stats['languages']}")


def example_index_with_openai():
    """Example: Index with OpenAI embeddings for better quality."""

    codebase_path = Path("./your-project")

    # Load or create config (respects CTXAI_HOME)
    config_manager = ConfigManager(codebase_path)
    config = config_manager.load()

    # Update to use OpenAI
    config.embedding.provider = "openai"
    config.embedding.model = "text-embedding-3-small"
    config_manager.save(config)

    # Rest is the same as local embeddings...
    print("Config updated to use OpenAI embeddings")
    print("Make sure OPENAI_API_KEY is set in environment")


def example_with_config_file():
    """Example: Use configuration from .ctxai/config.json."""

    codebase_path = Path("./your-project")

    # 1. Load configuration (respects CTXAI_HOME)
    config_manager = ConfigManager(codebase_path)
    config = config_manager.load()

    print(f"Embedding provider: {config.embedding.provider}")
    print(f"Max files: {config.indexing.max_files}")

    # 2. Create embedding provider based on config
    embeddings_gen = EmbeddingsFactory.create(config.embedding)

    # 3. Use it...
    text = "def hello(): return 'world'"
    embedding = embeddings_gen.generate_embedding(text)
    print(f"Generated embedding with {len(embedding)} dimensions")


def example_with_ctxai_home():
    """Example: Using CTXAI_HOME for global configuration."""
    import os

    # Set global CTXAI_HOME (can also be in .env or shell profile)
    os.environ["CTXAI_HOME"] = str(Path.home() / ".ctxai")

    # Now all projects use the same .ctxai directory
    ctxai_home = get_ctxai_home()
    print(f"Using global CTXAI_HOME: {ctxai_home}")

    # Configuration is shared across all projects
    config_manager = ConfigManager()
    config = config_manager.load()
    print(f"Global embedding provider: {config.embedding.provider}")


def example_search_codebase():
    """Example: Search an indexed codebase."""

    # 1. Load config and vector store (respects CTXAI_HOME)
    codebase_path = Path("./your-project")
    indexes_dir = get_indexes_dir(codebase_path)
    storage_path = indexes_dir / "my-index"

    vector_store = VectorStore(
        storage_path=storage_path,
        collection_name="my-index",
    )

    # 2. Load config and create embedding provider
    config_manager = ConfigManager(codebase_path)
    config = config_manager.load()
    embeddings_gen = EmbeddingsFactory.create(config.embedding)

    # 3. Generate query embedding
    query = "function to update user profile image"
    query_embedding = embeddings_gen.generate_embedding(query)

    # 4. Search
    results = vector_store.search(
        query_embedding=query_embedding,
        n_results=5,
    )

    # 5. Display results
    print(f"\nSearch results for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  File: {result['metadata']['file_path']}")
        print(f"  Lines: {result['metadata']['start_line']}-{result['metadata']['end_line']}")
        print(f"  Type: {result['metadata']['chunk_type']}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Preview: {result['content'][:100]}...")
        print()


if __name__ == "__main__":
    print("Example 1: Index with local embeddings (no API key needed)")
    print("-" * 70)
    # example_index_with_local_embeddings()

    print("\nExample 2: Index with OpenAI embeddings")
    print("-" * 70)
    # example_index_with_openai()

    print("\nExample 3: Use config file")
    print("-" * 70)
    # example_with_config_file()

    print("\nExample 4: Using CTXAI_HOME for global config")
    print("-" * 70)
    # example_with_ctxai_home()

    print("\nExample 5: Search indexed codebase")
    print("-" * 70)
    # example_search_codebase()

    print("\nNote: Uncomment the function calls to run the examples")
    print("Default local embeddings work without any API keys!")
    print("\nTip: Set CTXAI_HOME to use a global .ctxai directory:")
    print("  export CTXAI_HOME=~/.ctxai")
