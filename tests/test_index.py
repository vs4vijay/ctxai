"""Tests for the index module."""

import tempfile
from pathlib import Path

from ctxai.index import run_index


def test_index_valid_directory(capsys):
    """Test indexing a valid directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_index(tmpdir, "test_index")
        
        captured = capsys.readouterr()
        assert "Indexing codebase at:" in captured.out
        assert "test_index" in captured.out
        assert "Successfully created index" in captured.out


def test_index_nonexistent_path():
    """Test indexing a nonexistent path."""
    import sys
    import pytest
    
    nonexistent_path = "/nonexistent/path/that/does/not/exist"
    
    with pytest.raises(SystemExit) as exc_info:
        run_index(nonexistent_path, "test_index")
    
    assert exc_info.value.code == 1


def test_index_file_instead_of_directory():
    """Test indexing a file instead of a directory."""
    import sys
    import pytest
    
    with tempfile.NamedTemporaryFile() as tmpfile:
        with pytest.raises(SystemExit) as exc_info:
            run_index(tmpfile.name, "test_index")
        
        assert exc_info.value.code == 1
