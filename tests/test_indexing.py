"""
Test the indexing functionality.
"""

import shutil
import tempfile
from pathlib import Path

from ctxai.chunking import CodeChunk, CodeChunker
from ctxai.traversal import CodeTraversal


def test_code_traversal():
    """Test that code traversal finds Python files."""
    # Create a temporary directory with some test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test Python file
        test_file = tmpdir_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        # Create a file to ignore
        ignore_file = tmpdir_path / "ignore.pyc"
        ignore_file.write_text("binary stuff")

        # Test traversal
        traversal = CodeTraversal(tmpdir_path)
        files = list(traversal.traverse())

        assert len(files) == 1
        assert files[0].name == "test.py"


def test_code_chunker():
    """Test that code chunker creates chunks from Python code."""
    # Create a temporary Python file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        test_file = tmpdir_path / "test.py"
        test_code = """
def function_one():
    '''First function'''
    return 1

def function_two():
    '''Second function'''
    return 2

class MyClass:
    def method_one(self):
        return 3
"""
        test_file.write_text(test_code)

        # Test chunking
        chunker = CodeChunker()
        chunks = chunker.chunk_file(test_file)

        assert len(chunks) > 0
        assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
        assert any("function_one" in chunk.content for chunk in chunks)


def test_gitignore_respect():
    """Test that traversal respects .gitignore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .gitignore
        gitignore = tmpdir_path / ".gitignore"
        gitignore.write_text("ignored.py\n")

        # Create files
        included = tmpdir_path / "included.py"
        included.write_text("# Included file")

        ignored = tmpdir_path / "ignored.py"
        ignored.write_text("# Ignored file")

        # Test traversal with gitignore
        traversal = CodeTraversal(tmpdir_path, follow_gitignore=True)
        files = list(traversal.traverse())

        # Should find included.py and .gitignore itself (2 files total)
        # The ignored.py should not be found
        assert len(files) == 2
        file_names = [f.name for f in files]
        assert "included.py" in file_names
        assert ".gitignore" in file_names
        assert "ignored.py" not in file_names


def test_include_patterns():
    """Test that include patterns work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create different file types
        (tmpdir_path / "test.py").write_text("# Python")
        (tmpdir_path / "test.js").write_text("// JavaScript")
        (tmpdir_path / "test.txt").write_text("Text")

        # Test with include pattern
        traversal = CodeTraversal(tmpdir_path, include_patterns=["*.py"])
        files = list(traversal.traverse())

        assert len(files) == 1
        assert files[0].name == "test.py"


if __name__ == "__main__":
    test_code_traversal()
    print("✓ Code traversal test passed")

    test_code_chunker()
    print("✓ Code chunker test passed")

    test_gitignore_respect()
    print("✓ Gitignore test passed")

    test_include_patterns()
    print("✓ Include patterns test passed")

    print("\n✅ All tests passed!")
