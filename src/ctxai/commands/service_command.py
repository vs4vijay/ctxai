"""
`ctxai service` CLI commands — start, stop, restart, status, logs.
"""

from __future__ import annotations

import typer
from rich.console import Console

from ctxai.service.daemon import DaemonManager

console = Console()
service_app = typer.Typer(name="service", help="Manage the long-running ctxai service")


@service_app.command("start")
def start(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    workers: int = typer.Option(1, help="Number of workers"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
):
    """Start the ctxai service."""
    if foreground:
        try:
            from ctxai.service.api_server import run
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        run(host=host, port=port, workers=workers)
        return

    status = DaemonManager().start(host=host, port=port, workers=workers)
    color = "green" if status.running else "red"
    console.print(f"[{color}]{status.message}[/{color}]")


@service_app.command("stop")
def stop(timeout: float = typer.Option(10.0, help="Graceful shutdown timeout in seconds")):
    """Stop the ctxai service."""
    status = DaemonManager().stop(graceful_timeout=timeout)
    console.print(status.message)


@service_app.command("restart")
def restart(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    workers: int = typer.Option(1),
):
    """Restart the ctxai service."""
    status = DaemonManager().restart(host=host, port=port, workers=workers)
    console.print(status.message)


@service_app.command("status")
def status():
    """Show service status."""
    s = DaemonManager().status()
    color = "green" if s.running else "yellow"
    console.print(f"[{color}]{s.message}[/{color}]")
    console.print(f"PID file: {s.pid_file}")


@service_app.command("reload")
def reload_config():
    """Send SIGHUP to reload configuration (Unix only)."""
    s = DaemonManager().reload()
    console.print(s.message)


@service_app.command("logs")
def logs(
    tail: int = typer.Option(50, help="Number of lines to tail"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log file"),
):
    """Stream service logs from the configured log file."""
    from pathlib import Path

    from ctxai.utils import ensure_ctxai_home

    log_path = ensure_ctxai_home() / "logs" / "service.log"
    if not log_path.exists():
        console.print(f"[yellow]No log file at {log_path}[/yellow]")
        raise typer.Exit(1)

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-tail:]:
            console.print(line.rstrip())

        if follow:
            import time as _time

            while True:
                where = f.tell()
                line = f.readline()
                if line:
                    console.print(line.rstrip())
                else:
                    _time.sleep(0.5)
                    f.seek(where)
