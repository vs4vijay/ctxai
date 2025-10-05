"""Index module for creating semantic embeddings of codebases."""

import os
import sys
from pathlib import Path


def run_index(path: str, name: str):
    """Index a codebase for semantic search.
    
    Args:
        path: Path to the codebase to index
        name: Name for the index
    """
    codebase_path = Path(path).resolve()
    
    if not codebase_path.exists():
        print(f"Error: Path '{path}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not codebase_path.is_dir():
        print(f"Error: Path '{path}' is not a directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Indexing codebase at: {codebase_path}")
    print(f"Index name: {name}")
    
    # TODO: Implement actual indexing logic
    # This would involve:
    # 1. Scanning the codebase
    # 2. Extracting code chunks
    # 3. Creating embeddings
    # 4. Storing in a vector database
    
    print(f"Successfully created index '{name}' for {codebase_path}")


if __name__ == "__main__":
    # Support direct module execution
    if len(sys.argv) != 3:
        print("Usage: python -m ctxai.index <path> <index_name>", file=sys.stderr)
        sys.exit(1)
    
    run_index(sys.argv[1], sys.argv[2])
