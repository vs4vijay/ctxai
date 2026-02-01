"""
Shared pytest fixtures for all tests.

This module provides common fixtures used across both unit and E2E tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from ctxai.config import EmbeddingConfig
from tests.mocks.mock_embeddings import MockEmbeddingProvider


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for tests.

    Automatically cleaned up after test completes.

    Yields:
        Path: Path object pointing to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_code(temp_dir):
    """
    Create a sample Python codebase for testing.

    Copies sample Python files from fixtures to a temporary directory.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to temporary directory containing sample code
    """
    # Copy sample files to temp directory
    fixtures_dir = Path(__file__).parent / "fixtures"

    (temp_dir / "main.py").write_text((fixtures_dir / "sample_python.py").read_text())
    (temp_dir / "utils.py").write_text((fixtures_dir / "sample_utils.py").read_text())
    (temp_dir / "README.md").write_text("# Sample Project\n\nThis is a test project for ctxai.")

    return temp_dir


@pytest.fixture
def sample_javascript_code(temp_dir):
    """
    Create a sample JavaScript codebase for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to temporary directory containing sample code
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    (temp_dir / "index.js").write_text((fixtures_dir / "sample_javascript.js").read_text())
    (temp_dir / "package.json").write_text("""{
  "name": "sample-project",
  "version": "1.0.0",
  "description": "Sample JavaScript project",
  "main": "index.js"
}""")

    return temp_dir


@pytest.fixture
def sample_multi_language_code(temp_dir):
    """
    Create a multi-language codebase with Python and JavaScript.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to temporary directory containing sample code
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Python files
    (temp_dir / "main.py").write_text((fixtures_dir / "sample_python.py").read_text())
    (temp_dir / "utils.py").write_text((fixtures_dir / "sample_utils.py").read_text())

    # JavaScript files
    (temp_dir / "index.js").write_text((fixtures_dir / "sample_javascript.js").read_text())

    # README
    (temp_dir / "README.md").write_text("# Multi-Language Project\n\nPython and JavaScript code.")

    return temp_dir


@pytest.fixture
def sample_code_with_gitignore(temp_dir):
    """
    Create a sample codebase with .gitignore file.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to temporary directory with .gitignore
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Add some Python files
    (temp_dir / "main.py").write_text((fixtures_dir / "sample_python.py").read_text())
    (temp_dir / "utils.py").write_text((fixtures_dir / "sample_utils.py").read_text())

    # Add files that should be ignored
    (temp_dir / "ignored.py").write_text("def should_be_ignored(): pass")
    (temp_dir / "test.pyc").write_text("fake bytecode")

    # Create __pycache__ directory
    pycache_dir = temp_dir / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "main.cpython-311.pyc").write_text("fake cache")

    # Add .gitignore
    (temp_dir / ".gitignore").write_text("""# Python
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
ignored.py

# IDE
.vscode/
.idea/
*.swp
*.swo
""")

    return temp_dir


@pytest.fixture
def mock_embedding_config():
    """
    Create a mock embedding configuration.

    Returns:
        EmbeddingConfig: Configuration for mock embeddings
    """
    return EmbeddingConfig(
        provider="mock",
        model="mock-model",
        batch_size=32
    )


@pytest.fixture
def mock_embeddings(mock_embedding_config):
    """
    Create a mock embedding provider instance.

    Args:
        mock_embedding_config: Mock embedding configuration fixture

    Returns:
        MockEmbeddingProvider: Initialized mock provider
    """
    return MockEmbeddingProvider(mock_embedding_config)


@pytest.fixture
def sample_texts():
    """
    Provide sample texts for embedding tests.

    Returns:
        List[str]: List of sample text strings
    """
    return [
        "def greet(name): return f'Hello, {name}!'",
        "class Calculator: def add(self, x, y): return x + y",
        "function calculateSum(a, b) { return a + b; }",
        "// JavaScript utility functions",
        "import os\nimport sys\nfrom pathlib import Path",
    ]


@pytest.fixture(autouse=True)
def reset_environment_variables():
    """
    Reset environment variables after each test.

    This fixture automatically runs for each test to prevent
    environment variable pollution between tests.
    """
    import os
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
