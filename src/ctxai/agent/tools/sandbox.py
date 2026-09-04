"""OS-sandbox backends for BashTool command execution (HH-08).

A sandbox backend composes an OS-level, deny-by-default wrapper in front of a
classified argv. The in-process command policy
(``ToolExecutionContext.approve_command`` plus the ``BashTool`` allowlist)
always runs first and remains the first line of defense; the sandbox is the
second, OS-enforced layer.

Backends:

- ``MacOSSeatbeltBackend`` — generates a minimal seatbelt profile in a temp
  file and invokes ``sandbox-exec -f <profile> -- <argv>``. Seatbelt is
  deprecated-but-functional on macOS; its profile language is the scope
  boundary of this backend (network denial is solid, write restriction is a
  best-effort allowlist of the working directory and temp directories).
- ``BubblewrapBackend`` — invokes ``bwrap`` with ``--unshare-net`` (when
  network is denied), a read-only bind of the root filesystem, and a writable
  bind of the working directory plus tmpfs on ``/tmp``. Availability depends
  on the host.
- ``NoopBackend`` — identity backend used when sandboxing is ``off`` or when
  ``auto`` mode finds no backend: commands run exactly as before HH-08.

Semantics: ``wrap`` failures and ``required`` mode without a backend fail
closed — the command never runs unsandboxed. Generated profiles are temp
files removed by ``cleanup()`` after execution.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from ..config import SANDBOX_MODES


class SandboxError(RuntimeError):
    """Raised when a sandbox backend cannot wrap a command (fail closed)."""


class SandboxUnavailableError(SandboxError):
    """Raised in ``required`` mode when no sandbox backend is available."""


class SandboxBackend(Protocol):
    """Protocol for an OS-level, deny-by-default command sandbox.

    Implementations must be honest about enforcement: ``enforces`` is ``True``
    only when ``wrap`` actually subjects the child process to OS restrictions.
    """

    name: str
    enforces: bool

    def is_available(self) -> bool:
        """Report whether this backend can run on the current host."""
        ...

    def wrap(self, argv: list[str], cwd: Path, *, network: bool) -> list[str]:
        """Return the argv that runs ``argv`` under the sandbox.

        Args:
            argv: The classified command argv (executable plus arguments).
            cwd: The resolved working directory for the command.
            network: Whether outbound network access is allowed.

        Returns:
            The argv to execute instead of ``argv``.

        Raises:
            SandboxError: When the sandbox cannot be set up; callers must
                treat this as fail-closed and never run ``argv`` directly.
        """
        ...

    def adjust_environment(self, env: dict[str, str]) -> dict[str, str]:
        """Return the environment mapping adjusted for the sandboxed child.

        Args:
            env: The policy-filtered environment built by
                ``ToolExecutionContext.command_environment``.

        Returns:
            The environment mapping to pass to the wrapped process.
        """
        ...

    def cleanup(self) -> None:
        """Delete temporary artifacts (profiles) created by wrap calls."""
        ...


class NoopBackend:
    """Identity backend: runs the command unsandboxed, unchanged.

    This is the default backend. It is selected for sandbox mode ``off`` and
    for ``auto`` mode when no real backend exists on the host; in the latter
    case the caller surfaces a diagnostic and the audit record states that no
    OS sandbox was applied.
    """

    name = "none"
    enforces = False

    def is_available(self) -> bool:
        """Always available: the backend is a passthrough.

        Returns:
            ``True`` unconditionally.
        """
        return True

    def wrap(self, argv: list[str], cwd: Path, *, network: bool) -> list[str]:
        """Return ``argv`` unchanged.

        Args:
            argv: The classified command argv.
            cwd: The resolved working directory (unused).
            network: Whether network access is allowed (unused).

        Returns:
            The exact ``argv`` passed in.
        """
        return list(argv)

    def adjust_environment(self, env: dict[str, str]) -> dict[str, str]:
        """Return the environment unchanged.

        Args:
            env: The policy-filtered environment mapping.

        Returns:
            A copy of ``env`` with no adjustments.
        """
        return dict(env)

    def cleanup(self) -> None:
        """No temporary artifacts to clean."""
        return None


def _seatbelt_quote(value: str) -> str:
    """Quote a string for the seatbelt profile language.

    Args:
        value: The raw path or string value.

    Returns:
        A double-quoted, escaped seatbelt string literal.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_roots(cwd: Path) -> list[Path]:
    """Build the deduplicated list of writable directory roots for a profile.

    Seatbelt evaluates canonical (symlink-resolved) paths, so every root is
    resolved: ``/tmp`` becomes ``/private/tmp`` and so on.

    Args:
        cwd: The resolved working directory for the command.

    Returns:
        Writable roots: the working directory, ``TMPDIR`` when set, and the
        standard system temp directories that exist.
    """
    roots: list[Path] = [cwd]
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        roots.append(Path(tmpdir).resolve())
    # Pinned system temp roots are the sandbox write policy, not temp files.
    for candidate in ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"):  # nosec B108
        path = Path(candidate)
        if path.exists():
            roots.append(path.resolve())
    deduped: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped


def _seatbelt_profile(*, cwd: Path, network: bool) -> str:
    """Generate a minimal seatbelt profile.

    The profile allows everything by default, then denies network access
    (unless ``network`` is requested) and restricts writes to the working
    directory, temp directories, and ``/dev/null``. Write restriction is
    best-effort: commands that need writes elsewhere fail visibly rather
    than silently (see docs/SANDBOXING.md).

    Args:
        cwd: The resolved working directory for the command.
        network: Whether outbound network access is allowed.

    Returns:
        The seatbelt profile source text.
    """
    lines = ["(version 1)", "(allow default)"]
    if not network:
        lines.append("(deny network*)")
    lines.append("(deny file-write*)")
    allows = [f"(subpath {_seatbelt_quote(str(root))})" for root in _write_roots(cwd)]
    allows.append('(literal "/dev/null")')
    lines.append("(allow file-write* " + " ".join(allows) + ")")
    return "\n".join(lines) + "\n"


class MacOSSeatbeltBackend:
    """macOS seatbelt backend via ``/usr/bin/sandbox-exec``.

    Each ``wrap`` writes a minimal profile to a temp file that is deleted by
    ``cleanup`` after the command finishes. The profile language is the scope
    boundary of this backend: network denial is solid, write restriction is
    best-effort, and seatbelt itself is deprecated-but-functional on macOS.
    """

    name = "seatbelt"
    enforces = True

    def __init__(self, executable: str | None = None):
        """Create the backend.

        Args:
            executable: Explicit ``sandbox-exec`` path; resolved from ``PATH``
                when omitted (useful for tests).
        """
        self._executable = executable
        self._pending: list[Path] = []

    def pending_profiles(self) -> list[Path]:
        """List profiles generated but not yet cleaned up.

        Returns:
            The temp profile paths awaiting ``cleanup``.
        """
        return list(self._pending)

    def _resolve_executable(self) -> str | None:
        """Resolve the sandbox-exec binary path.

        Returns:
            The executable path, or ``None`` when not installed.
        """
        if self._executable is not None:
            return self._executable
        return shutil.which("sandbox-exec")

    def is_available(self) -> bool:
        """Report availability on this host.

        Returns:
            ``True`` on macOS when ``sandbox-exec`` is on ``PATH``.
        """
        return sys.platform == "darwin" and self._resolve_executable() is not None

    def wrap(self, argv: list[str], cwd: Path, *, network: bool) -> list[str]:
        """Compose ``sandbox-exec -f <profile> -- <argv>``.

        Args:
            argv: The classified command argv.
            cwd: The resolved working directory for the command.
            network: Whether outbound network access is allowed.

        Returns:
            The wrapped argv referencing a freshly written temp profile.

        Raises:
            SandboxError: When ``sandbox-exec`` is missing or the profile
                cannot be written.
        """
        executable = self._resolve_executable()
        if executable is None:
            raise SandboxError("sandbox-exec not found on PATH; cannot wrap command")
        profile_path = self._write_profile(cwd=cwd, network=network)
        return [executable, "-f", str(profile_path), "--", *argv]

    def _write_profile(self, *, cwd: Path, network: bool) -> Path:
        """Write the seatbelt profile to a temp file.

        Args:
            cwd: The resolved working directory for the command.
            network: Whether outbound network access is allowed.

        Returns:
            The temp profile path (registered for ``cleanup``).

        Raises:
            SandboxError: When the profile file cannot be created or written.
        """
        try:
            handle, raw_path = tempfile.mkstemp(prefix="ctxai-seatbelt-", suffix=".sb")
            path = Path(raw_path)
            with os.fdopen(handle, "w", encoding="utf-8") as profile_file:
                profile_file.write(_seatbelt_profile(cwd=cwd, network=network))
        except OSError as exc:
            # Best-effort removal of a partially created temp file so a failed
            # wrap leaves no artifacts behind.
            if "path" in locals():
                with suppress(OSError):
                    path.unlink(missing_ok=True)
            raise SandboxError(f"could not write seatbelt profile: {exc}") from exc
        self._pending.append(path)
        return path

    def adjust_environment(self, env: dict[str, str]) -> dict[str, str]:
        """Return the environment unchanged (seatbelt needs no adjustments).

        Args:
            env: The policy-filtered environment mapping.

        Returns:
            A copy of ``env``.
        """
        return dict(env)

    def cleanup(self) -> None:
        """Delete every temp profile generated by this backend so far."""
        for path in self._pending:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._pending.clear()


class BubblewrapBackend:
    """Linux bubblewrap backend via ``bwrap``.

    Composes a standard minimal jail: a read-only bind of the root
    filesystem, a writable bind of the working directory, fresh ``/dev`` and
    ``/proc`` mounts, tmpfs on ``/tmp``, and ``--unshare-net`` when network
    access is denied. Availability depends on the host; the backend is not
    exercised on hosts without the ``bwrap`` binary.
    """

    name = "bwrap"
    enforces = True

    def __init__(self, executable: str | None = None):
        """Create the backend.

        Args:
            executable: Explicit ``bwrap`` path; resolved from ``PATH`` when
                omitted (useful for tests).
        """
        self._executable = executable

    def _resolve_executable(self) -> str | None:
        """Resolve the bwrap binary path.

        Returns:
            The executable path, or ``None`` when not installed.
        """
        if self._executable is not None:
            return self._executable
        return shutil.which("bwrap")

    def is_available(self) -> bool:
        """Report availability on this host.

        Returns:
            ``True`` when the ``bwrap`` binary is on ``PATH``.
        """
        return self._resolve_executable() is not None

    def wrap(self, argv: list[str], cwd: Path, *, network: bool) -> list[str]:
        """Compose the ``bwrap`` argv for the command.

        Args:
            argv: The classified command argv.
            cwd: The resolved working directory for the command.
            network: Whether outbound network access is allowed.

        Returns:
            The wrapped argv.

        Raises:
            SandboxError: When ``bwrap`` is missing.
        """
        executable = self._resolve_executable()
        if executable is None:
            raise SandboxError("bwrap not found on PATH; cannot wrap command")
        resolved = Path(cwd).resolve()
        command: list[str] = [
            executable,
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(resolved),
            str(resolved),
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",  # nosec B108 - bwrap tmpfs mount target, the sandbox write policy
        ]
        if not network:
            command.append("--unshare-net")
        command.append("--")
        command.extend(argv)
        return command

    def adjust_environment(self, env: dict[str, str]) -> dict[str, str]:
        """Point ``TMPDIR`` at the writable tmpfs inside the jail.

        Args:
            env: The policy-filtered environment mapping.

        Returns:
            A copy of ``env`` with ``TMPDIR`` set to ``/tmp``.
        """
        adjusted = dict(env)
        adjusted["TMPDIR"] = "/tmp"  # nosec B108 - points at the bwrap tmpfs mounted above
        return adjusted

    def cleanup(self) -> None:
        """No temporary artifacts to clean (the profile is argv-embedded)."""
        return None


def default_backends() -> list[SandboxBackend]:
    """Build the backend preference list for this host.

    Seatbelt is preferred on macOS; bubblewrap is the fallback where the
    ``bwrap`` binary exists.

    Returns:
        Backend instances in preference order.
    """
    return [MacOSSeatbeltBackend(), BubblewrapBackend()]


def select_backend(
    mode: str,
    *,
    backends: list[SandboxBackend] | None = None,
) -> tuple[SandboxBackend, str | None]:
    """Resolve the effective backend for a sandbox mode.

    Args:
        mode: One of :data:`SANDBOX_MODES` (``off``, ``auto``, ``required``).
        backends: Explicit backend list (tests); defaults to
            :func:`default_backends`.

    Returns:
        A ``(backend, diagnostic)`` tuple. A real backend yields a ``None``
        diagnostic. Mode ``off`` — or ``auto`` on a host without any backend —
        yields the :class:`NoopBackend`; the diagnostic then explains why no
        OS sandbox is enforced. ``required`` callers must treat a
        non-enforcing backend as a fail-closed condition.

    Raises:
        ValueError: When ``mode`` is not one of :data:`SANDBOX_MODES`.
    """
    if mode not in SANDBOX_MODES:
        raise ValueError(f"Unknown sandbox mode: {mode!r}; expected one of: {', '.join(SANDBOX_MODES)}")
    if mode == "off":
        return NoopBackend(), None
    for backend in backends if backends is not None else default_backends():
        if backend.is_available():
            return backend, None
    return NoopBackend(), (
        "no sandbox backend available on this host (macOS seatbelt via sandbox-exec or Linux bubblewrap via bwrap)"
    )


def describe_sandbox(
    mode: str,
    network: bool,
    *,
    backends: list[SandboxBackend] | None = None,
) -> str | None:
    """Build the human-readable sandbox status line for chat.

    Args:
        mode: The configured sandbox mode.
        network: Whether network access is allowed by configuration.
        backends: Explicit backend list (tests); defaults to
            :func:`default_backends`.

    Returns:
        ``None`` for mode ``off`` (today's behavior, nothing to show);
        otherwise a badge like ``sandbox: seatbelt (network denied)`` or an
        explicit ``sandbox unavailable`` diagnostic.
    """
    if mode == "off":
        return None
    backend, diagnostic = select_backend(mode, backends=backends)
    if backend.enforces:
        network_state = "network allowed" if network else "network denied"
        return f"sandbox: {backend.name} ({network_state})"
    if mode == "required":
        return f"sandbox unavailable: {diagnostic} — mode is required, bash commands will fail"
    return f"sandbox unavailable: {diagnostic} — commands run without OS sandboxing"
