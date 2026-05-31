"""Tests for ctxai.security."""

from pathlib import Path

import pytest

from ctxai.security import (
    DEFAULT_BLOCKED_COMMANDS,
    SecurityError,
    SecurityManager,
    SecurityPolicy,
)


def test_validate_file_path_allows_files_in_base(tmp_path: Path):
    inside = tmp_path / "sub" / "file.txt"
    inside.parent.mkdir()
    inside.write_text("ok")
    sm = SecurityManager()
    resolved = sm.validate_file_path(inside, base_dir=tmp_path)
    assert resolved == inside.resolve()


def test_validate_file_path_blocks_traversal(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere.txt"
    sm = SecurityManager()
    with pytest.raises(SecurityError):
        sm.validate_file_path(outside, base_dir=tmp_path)


def test_validate_file_path_blocks_dotdot(tmp_path: Path):
    sm = SecurityManager()
    target = tmp_path / ".." / "etc" / "passwd"
    with pytest.raises(SecurityError):
        sm.validate_file_path(target, base_dir=tmp_path)


def test_max_path_length_enforced(tmp_path: Path):
    policy = SecurityPolicy(max_path_length=10)
    sm = SecurityManager(policy)
    with pytest.raises(SecurityError):
        sm.validate_file_path(tmp_path / "this_is_definitely_too_long.txt", base_dir=tmp_path)


def test_validate_bash_command_blocks_defaults():
    sm = SecurityManager()
    for blocked in DEFAULT_BLOCKED_COMMANDS[:3]:
        with pytest.raises(SecurityError):
            sm.validate_bash_command(f"echo hi; {blocked}")


def test_validate_bash_command_allows_safe():
    sm = SecurityManager()
    assert sm.validate_bash_command("ls -la") == "ls -la"


def test_whitelist_mode():
    policy = SecurityPolicy(allowed_commands=("ls", "pwd", "echo"))
    sm = SecurityManager(policy)
    assert sm.validate_bash_command("ls /tmp") == "ls /tmp"
    with pytest.raises(SecurityError):
        sm.validate_bash_command("rm /etc/foo")


def test_command_too_long_rejected():
    policy = SecurityPolicy(max_command_length=10)
    sm = SecurityManager(policy)
    with pytest.raises(SecurityError):
        sm.validate_bash_command("echo " + "x" * 100)


def test_api_key_shape_anthropic():
    assert SecurityManager.is_valid_api_key_shape("sk-ant-12345678", "anthropic")
    assert not SecurityManager.is_valid_api_key_shape("invalid", "anthropic")


def test_api_key_shape_rejects_whitespace():
    assert not SecurityManager.is_valid_api_key_shape("sk- with space")
    assert not SecurityManager.is_valid_api_key_shape("")
    assert not SecurityManager.is_valid_api_key_shape(None)


def test_contains_shell_meta():
    assert SecurityManager.contains_shell_meta("a | b")
    assert SecurityManager.contains_shell_meta("a && b")
    assert not SecurityManager.contains_shell_meta("just plain text")
