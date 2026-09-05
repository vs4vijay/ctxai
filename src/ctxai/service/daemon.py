"""
Daemon process management for `ctxai service`.

This module focuses on the cross-platform parts: PID file management,
graceful start/stop/status. Heavy lifting (forking, signal handlers) is
left to the uvicorn process — most users will run via `systemd` or
`docker`, and this layer just gives them a `ctxai service` CLI.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ctxai.logging import get_logger
from ctxai.utils import ensure_ctxai_home

logger = get_logger("service.daemon")


@dataclass
class DaemonStatus:
    running: bool
    pid: int | None
    pid_file: Path
    message: str


class DaemonManager:
    """Start, stop, restart, and inspect a ctxai service process."""

    def __init__(self, pid_file: Path | None = None):
        self.pid_file = pid_file or (ensure_ctxai_home() / "ctxai.pid")

    # ----- PID file -----

    def _read_pid(self) -> int | None:
        try:
            return int(self.pid_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _write_pid(self, pid: int) -> None:
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(pid))

    def _clear_pid(self) -> None:
        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass

    def _process_alive(self, pid: int) -> bool:
        try:
            if sys.platform == "win32":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                SYNCHRONIZE = 0x00100000
                handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
                if not handle:
                    return False
                kernel32.CloseHandle(handle)
                return True
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    # ----- Lifecycle -----

    def status(self) -> DaemonStatus:
        pid = self._read_pid()
        if pid is None:
            return DaemonStatus(False, None, self.pid_file, "Not running")
        if self._process_alive(pid):
            return DaemonStatus(True, pid, self.pid_file, f"Running (pid={pid})")
        self._clear_pid()
        return DaemonStatus(False, None, self.pid_file, "Stale pid file removed")

    def start(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> DaemonStatus:
        current = self.status()
        if current.running:
            return current
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "ctxai.service.api_server:create_app",
            "--factory",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            str(workers),
        ]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )
        else:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        self._write_pid(proc.pid)
        logger.info(f"Started ctxai service (pid={proc.pid}) on {host}:{port}")
        return DaemonStatus(True, proc.pid, self.pid_file, f"Started pid={proc.pid}")

    def stop(self, graceful_timeout: float = 10.0) -> DaemonStatus:
        pid = self._read_pid()
        if pid is None:
            return DaemonStatus(False, None, self.pid_file, "Not running")

        sig = signal.SIGTERM if sys.platform != "win32" else signal.SIGTERM
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, OSError) as exc:
            self._clear_pid()
            return DaemonStatus(False, None, self.pid_file, f"Already gone: {exc}")

        deadline = time.monotonic() + graceful_timeout
        while time.monotonic() < deadline:
            if not self._process_alive(pid):
                self._clear_pid()
                return DaemonStatus(False, None, self.pid_file, "Stopped")
            time.sleep(0.2)

        # Force kill if still alive
        try:
            os.kill(pid, signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM)
        except Exception:
            pass
        self._clear_pid()
        return DaemonStatus(False, None, self.pid_file, "Force-stopped")

    def restart(self, **kwargs) -> DaemonStatus:
        self.stop()
        return self.start(**kwargs)

    def reload(self) -> DaemonStatus:
        """Send SIGHUP for config reload (no-op on Windows)."""
        pid = self._read_pid()
        if pid is None:
            return DaemonStatus(False, None, self.pid_file, "Not running")
        sighup = getattr(signal, "SIGHUP", None)
        if sighup is None:
            return DaemonStatus(True, pid, self.pid_file, "SIGHUP unsupported on this platform")
        try:
            os.kill(pid, sighup)
        except Exception as exc:
            return DaemonStatus(False, pid, self.pid_file, f"Reload failed: {exc}")
        return DaemonStatus(True, pid, self.pid_file, "Sent SIGHUP")
