"""Tests for the CLI."""

import subprocess
import sys


def test_cli_help():
    """Test CLI help output."""
    result = subprocess.run(
        [sys.executable, "-m", "ctxai", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "ctxai - Semantic code search engine" in result.stdout
    assert "index" in result.stdout
    assert "server" in result.stdout
    assert "playground" in result.stdout
    assert "shell" in result.stdout


def test_cli_no_command():
    """Test CLI with no command."""
    result = subprocess.run(
        [sys.executable, "-m", "ctxai"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "ctxai - Semantic code search engine" in result.stdout


def test_index_command_help():
    """Test index command help."""
    result = subprocess.run(
        [sys.executable, "-m", "ctxai", "index", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "path" in result.stdout
    assert "name" in result.stdout


def test_server_command_help():
    """Test server command help."""
    result = subprocess.run(
        [sys.executable, "-m", "ctxai", "server", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "--index" in result.stdout


def test_playground_command():
    """Test playground command."""
    result = subprocess.run(
        [sys.executable, "-m", "ctxai", "playground"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "playground" in result.stdout


def test_shell_command():
    """Test shell command."""
    result = subprocess.run(
        [sys.executable, "-m", "ctxai", "shell"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "shell" in result.stdout
