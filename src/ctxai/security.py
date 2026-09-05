"""
Security primitives shared across tools and the service layer.

- File path validation (path-traversal prevention)
- Bash command screening
- API key shape validation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class SecurityError(Exception):
    """Raised when a security policy violation is detected."""


# Patterns that are dangerous regardless of context.
DEFAULT_BLOCKED_COMMANDS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    ":(){ :|:& };:",
    "mkfs",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "> /dev/sda",
    "mv / /dev/null",
    "chmod -R 777 /",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
)

# Shell metacharacters that signal command chaining.
_SHELL_METAS = re.compile(r"[;&|`$<>]")


@dataclass
class SecurityPolicy:
    """Security configuration for tool execution and service endpoints."""

    allowed_base_dirs: list[Path] | None = None
    block_symlinks: bool = False
    allow_hidden_files: bool = True
    blocked_commands: tuple[str, ...] = DEFAULT_BLOCKED_COMMANDS
    allowed_commands: tuple[str, ...] | None = None  # whitelist mode
    max_command_length: int = 4096
    max_path_length: int = 4096


class SecurityManager:
    """Enforce security policy for file and command operations."""

    def __init__(self, policy: SecurityPolicy | None = None):
        self.policy = policy or SecurityPolicy()

    # ----- Path validation -----

    def validate_file_path(self, path: str | Path, base_dir: Path | None = None) -> Path:
        """
        Resolve `path` and ensure it sits inside `base_dir` (or any of the
        configured allowed base dirs).

        Raises:
            SecurityError: If the path escapes the allowed roots, is too long,
                or violates symlink policy.
        """
        path = Path(path)
        if len(str(path)) > self.policy.max_path_length:
            raise SecurityError(f"Path too long: {len(str(path))} > {self.policy.max_path_length}")

        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise SecurityError(f"Could not resolve path {path}: {exc}") from exc

        if not self.policy.allow_hidden_files:
            for part in resolved.parts:
                if part.startswith(".") and part not in (".", ".."):
                    raise SecurityError(f"Hidden files disallowed: {resolved}")

        if self.policy.block_symlinks and path.is_symlink():
            raise SecurityError(f"Symlink resolution disabled: {path}")

        candidate_bases: list[Path] = []
        if base_dir is not None:
            candidate_bases.append(Path(base_dir).resolve())
        if self.policy.allowed_base_dirs:
            candidate_bases.extend(p.resolve() for p in self.policy.allowed_base_dirs)

        if not candidate_bases:
            return resolved

        for base in candidate_bases:
            if self._is_within(resolved, base):
                return resolved

        raise SecurityError(
            f"Path traversal blocked: {resolved} is not inside any allowed base "
            f"({', '.join(str(b) for b in candidate_bases)})"
        )

    @staticmethod
    def _is_within(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    # ----- Command validation -----

    def validate_bash_command(self, command: str) -> str:
        """
        Reject obviously dangerous shell commands.

        Returns the (unchanged) command if it passes screening.

        Raises:
            SecurityError: If the command matches a blocklist entry or
                fails whitelist enforcement.
        """
        if not command or not command.strip():
            raise SecurityError("Empty command")

        if len(command) > self.policy.max_command_length:
            raise SecurityError(f"Command too long: {len(command)} > {self.policy.max_command_length}")

        normalized = command.strip().lower()

        for blocked in self.policy.blocked_commands:
            if blocked.lower() in normalized:
                raise SecurityError(f"Command blocked by policy: matched '{blocked}'")

        if self.policy.allowed_commands is not None:
            # Whitelist mode: command must START with an allowed prefix.
            head = normalized.split()[0] if normalized.split() else ""
            if not any(
                head == cmd.lower() or normalized.startswith(cmd.lower()) for cmd in self.policy.allowed_commands
            ):
                raise SecurityError(f"Command not in whitelist (head='{head}'): {self.policy.allowed_commands}")

        return command

    @staticmethod
    def contains_shell_meta(command: str) -> bool:
        return bool(_SHELL_METAS.search(command))

    # ----- API key validation -----

    @staticmethod
    def is_valid_api_key_shape(key: str | None, provider: str | None = None) -> bool:
        """Light heuristic shape check — not auth, just sanity."""
        if not key or not isinstance(key, str):
            return False
        key = key.strip()
        if len(key) < 8 or len(key) > 256:
            return False
        if any(ch.isspace() for ch in key):
            return False
        if provider == "anthropic" and not key.startswith("sk-"):
            return False
        if provider == "openai" and not key.startswith("sk-"):
            return False
        return True


_default_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    return _default_manager


def set_security_policy(policy: SecurityPolicy) -> None:
    """Replace the default policy. Useful for tests and service setup."""
    global _default_manager
    _default_manager = SecurityManager(policy)
