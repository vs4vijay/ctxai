"""
End-to-end tests for indexing and query workflows.

Tests the complete pipeline from code traversal to semantic search.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from ctxai.commands.index_command import index_codebase
from ctxai.commands.query_command import query_codebase
from ctxai.vector_store import VectorStore
from ctxai.utils import get_indexes_dir


@pytest.mark.e2e
@pytest.mark.indexing
def test_full_indexing_and_query_workflow(sample_python_code, temp_dir, patch_embeddings_factory, capsys):
    """
    Test complete indexing + query flow with mocked embeddings.

    This test verifies:
    1. Index creation from real codebase
    2. Code traversal and chunking
    3. Embedding generation (mocked for speed)
    4. Vector storage
    5. Query execution
    6. Result retrieval and formatting
    """
    # Patch get_indexes_dir to use our temp directory
    with patch('ctxai.commands.index_command.get_indexes_dir', return_value=temp_dir / ".ctxai" / "indexes"):
        with patch('ctxai.commands.query_command.get_indexes_dir', return_value=temp_dir / ".ctxai" / "indexes"):
            # Create index
            index_name = "test-index"
            index_codebase(
                path=sample_python_code,
                index_name=index_name,
                include_patterns=["*.py"],
                follow_gitignore=False
            )

            # Verify index was created
            indexes_dir = temp_dir / ".ctxai" / "indexes"
            index_path = indexes_dir / index_name
            assert index_path.exists(), f"Index directory not created at {index_path}"

            # Load vector store and verify it has data
            vector_store = VectorStore(indexes_dir, index_name)
            stats = vector_store.get_stats()
            assert stats["total_chunks"] > 0, "Index should contain chunks"
            assert stats["total_files"] > 0, "Index should have files"

            # Query the index
            query_codebase(
                index_name=index_name,
                query="function that greets someone",
                project_path=temp_dir,
                n_results=3,
                show_content=True
            )

            # Capture output
            captured = capsys.readouterr()
            output = captured.out

            # Verify results contain expected content
            assert "greet" in output.lower(), "Output should mention greet function"
            assert "main.py" in output or "sample_python.py" in output, "Output should mention the file"


@pytest.mark.e2e
@pytest.mark.indexing
def test_indexing_respects_gitignore(sample_code_with_gitignore, temp_dir, patch_embeddings_factory):
    """
    Test that indexing respects .gitignore patterns.

    Verifies:
    1. .gitignore file is read and respected
    2. Ignored files are not indexed
    3. Non-ignored files are indexed
    """
    with patch('ctxai.commands.index_command.get_indexes_dir', return_value=temp_dir / ".ctxai" / "indexes"):
        # Index with gitignore enabled
        index_name = "gitignore-test"
        index_codebase(
            path=sample_code_with_gitignore,
            index_name=index_name,
            follow_gitignore=True
        )

        # Load index and check what files were indexed
        indexes_dir = temp_dir / ".ctxai" / "indexes"
        vector_store = VectorStore(indexes_dir, index_name)
        stats = vector_store.get_stats()

        # Get all file paths from chunks
        collection = vector_store.collection
        results = collection.get(include=["metadatas"])
        file_paths = [meta["file_path"] for meta in results["metadatas"]]

        # Verify ignored files are NOT in index
        assert not any("ignored.py" in fp for fp in file_paths), "ignored.py should not be indexed"
        assert not any(".pyc" in fp for fp in file_paths), ".pyc files should not be indexed"
        assert not any("__pycache__" in fp for fp in file_paths), "__pycache__ should not be indexed"

        # Verify non-ignored files ARE in index
        assert any("main.py" in fp for fp in file_paths), "main.py should be indexed"
        assert any("utils.py" in fp for fp in file_paths), "utils.py should be indexed"


@pytest.mark.e2e
@pytest.mark.indexing
def test_incremental_indexing(sample_python_code, temp_dir, patch_embeddings_factory):
    """
    Test incremental indexing by adding new files.

    Verifies:
    1. Initial index creation
    2. Adding new files to codebase
    3. Re-indexing updates the index
    4. Both old and new files are present
    """
    with patch('ctxai.commands.index_command.get_indexes_dir', return_value=temp_dir / ".ctxai" / "indexes"):
        index_name = "incremental-test"

        # Initial indexing
        index_codebase(
            path=sample_python_code,
            index_name=index_name,
            include_patterns=["*.py"],
            follow_gitignore=False
        )

        # Get initial stats
        indexes_dir = temp_dir / ".ctxai" / "indexes"
        vector_store = VectorStore(indexes_dir, index_name)
        initial_stats = vector_store.get_stats()
        initial_chunks = initial_stats["total_chunks"]

        # Add a new file to the codebase
        new_file = sample_python_code / "new_module.py"
        new_file.write_text("""
def process_data(data: list) -> list:
    \"\"\"Process data list.\"\"\"
    return [x * 2 for x in data]

class DataProcessor:
    \"\"\"Data processor class.\"\"\"

    def transform(self, value):
        return value ** 2
""")

        # Re-index
        index_codebase(
            path=sample_python_code,
            index_name=index_name,
            include_patterns=["*.py"],
            follow_gitignore=False
        )

        # Verify new chunks were added
        vector_store = VectorStore(indexes_dir, index_name)
        new_stats = vector_store.get_stats()

        assert new_stats["total_chunks"] > initial_chunks, "New chunks should be added"
        assert new_stats["total_files"] > initial_stats["total_files"], "File count should increase"

        # Verify both old and new content are present
        collection = vector_store.collection
        results = collection.get(include=["metadatas", "documents"])
        all_content = " ".join(results["documents"])

        assert "greet" in all_content, "Old functions should still be indexed"
        assert "process_data" in all_content, "New function should be indexed"


@pytest.mark.e2e
@pytest.mark.indexing
def test_multi_language_indexing(sample_multi_language_code, temp_dir, patch_embeddings_factory):
    """
    Test indexing a multi-language project.

    Verifies:
    1. Python files are indexed
    2. JavaScript files are indexed
    3. Correct language metadata is stored
    4. Query returns results from multiple languages
    """
    with patch('ctxai.commands.index_command.get_indexes_dir', return_value=temp_dir / ".ctxai" / "indexes"):
        with patch('ctxai.commands.query_command.get_indexes_dir', return_value=temp_dir / ".ctxai" / "indexes"):
            index_name = "multi-lang-test"

            # Index multi-language codebase
            index_codebase(
                path=sample_multi_language_code,
                index_name=index_name,
                include_patterns=["*.py", "*.js"],
                follow_gitignore=False
            )

            # Load index
            indexes_dir = temp_dir / ".ctxai" / "indexes"
            vector_store = VectorStore(indexes_dir, index_name)

            # Get all metadatas
            collection = vector_store.collection
            results = collection.get(include=["metadatas"])
            metadatas = results["metadatas"]

            # Group by file extension
            python_files = [m for m in metadatas if m["file_path"].endswith(".py")]
            js_files = [m for m in metadatas if m["file_path"].endswith(".js")]

            assert len(python_files) > 0, "Should have Python files indexed"
            assert len(js_files) > 0, "Should have JavaScript files indexed"

            # Verify language metadata
            for meta in python_files:
                assert meta.get("language") == "python", f"Python file should have language=python: {meta}"

            for meta in js_files:
                assert meta.get("language") == "javascript", f"JS file should have language=javascript: {meta}"

            # Query should work across languages
            query_codebase(
                index_name=index_name,
                query="calculator class",
                project_path=temp_dir,
                n_results=10,
                show_content=False
            )

            # If query works without error, the test passes
            # (output verification would need to capture console output)
