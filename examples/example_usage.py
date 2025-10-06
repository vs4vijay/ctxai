"""
Example script demonstrating how to use ctxai programmatically.
"""

from pathlib import Path
from ctxai.chunking import CodeChunker
from ctxai.traversal import CodeTraversal
from ctxai.embeddings import EmbeddingsGenerator
from ctxai.vector_store import VectorStore


def example_index_codebase():
    """Example: Index a codebase programmatically."""
    
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
    
    # 4. Generate embeddings
    embeddings_gen = EmbeddingsGenerator()
    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embeddings_gen.generate_embeddings(chunk_texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    
    # 5. Store in vector database
    storage_path = codebase_path / ".ctxai" / "indexes" / "my-index"
    vector_store = VectorStore(
        storage_path=storage_path,
        collection_name="my-index",
    )
    vector_store.add_chunks(all_chunks, embeddings)
    
    print("✓ Indexing complete!")
    
    # 6. Get stats
    stats = vector_store.get_stats()
    print(f"\nIndex stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique files: {stats['unique_files']}")
    print(f"  Languages: {stats['languages']}")


def example_search_codebase():
    """Example: Search an indexed codebase."""
    
    # 1. Load vector store
    codebase_path = Path("./your-project")
    storage_path = codebase_path / ".ctxai" / "indexes" / "my-index"
    vector_store = VectorStore(
        storage_path=storage_path,
        collection_name="my-index",
    )
    
    # 2. Generate query embedding
    embeddings_gen = EmbeddingsGenerator()
    query = "function to update user profile image"
    query_embedding = embeddings_gen.generate_embedding(query)
    
    # 3. Search
    results = vector_store.search(
        query_embedding=query_embedding,
        n_results=5,
    )
    
    # 4. Display results
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
    print("Example 1: Index a codebase")
    print("-" * 50)
    # example_index_codebase()
    print("\nExample 2: Search indexed codebase")
    print("-" * 50)
    # example_search_codebase()
    print("\nNote: Uncomment the function calls to run the examples")
    print("Make sure to set OPENAI_API_KEY environment variable first!")
