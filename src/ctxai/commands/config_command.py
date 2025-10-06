"""
Config command implementation.
Manages configuration settings similar to git config.
"""

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import ConfigManager
from ..utils import get_ctxai_home, is_using_global_home

console = Console()


def list_config(project_path: Path | None = None):
    """
    List all configuration settings.

    Args:
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Show where config is located
    ctxai_home = get_ctxai_home(project_path)
    if is_using_global_home():
        console.print(f"[dim]Global config: {config_manager.config_path}[/dim]\n")
    else:
        console.print(f"[dim]Project config: {config_manager.config_path}[/dim]\n")

    # Create a table for configuration
    table = Table(title="Configuration Settings", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="green", no_wrap=True)
    table.add_column("Value", style="yellow")

    # Add index metadata
    if config.index_name:
        table.add_row("index.name", config.index_name)
        table.add_row("index.status", config.index_status or "[dim]unknown[/dim]")
        if config.index_files_count is not None:
            table.add_row("index.files_count", str(config.index_files_count))
        if config.index_size_mb is not None:
            table.add_row("index.size_mb", f"{config.index_size_mb:.2f}")
        if config.index_chunks_count is not None:
            table.add_row("index.chunks_count", str(config.index_chunks_count))
        if config.index_last_updated:
            table.add_row("index.last_updated", config.index_last_updated)
    else:
        table.add_row("index.name", "[dim]not set[/dim]")

    # Add embedding settings
    table.add_row("embedding.provider", config.embedding.provider)
    table.add_row(
        "embedding.model",
        str(config.embedding.model) if config.embedding.model else "[dim]default[/dim]",
    )
    table.add_row("embedding.api_key", "***" if config.embedding.api_key else "[dim]not set[/dim]")
    table.add_row("embedding.batch_size", str(config.embedding.batch_size))
    table.add_row(
        "embedding.max_tokens",
        str(config.embedding.max_tokens) if config.embedding.max_tokens else "[dim]not set[/dim]",
    )

    # Add indexing settings
    table.add_row("indexing.max_files", str(config.indexing.max_files))
    table.add_row("indexing.max_total_size_mb", str(config.indexing.max_total_size_mb))
    table.add_row("indexing.max_file_size_mb", str(config.indexing.max_file_size_mb))
    table.add_row("indexing.chunk_size", str(config.indexing.chunk_size))
    table.add_row("indexing.chunk_overlap", str(config.indexing.chunk_overlap))

    # Add version
    table.add_row("version", config.version)

    console.print(table)
    console.print()


def get_config(key: str, project_path: Path | None = None):
    """
    Get a specific configuration value.

    Args:
        key: Configuration key in dot notation (e.g., "embedding.provider")
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Parse the key
    parts = key.split(".")
    if len(parts) < 1:
        console.print("[red]✗[/red] Invalid key format. Use dot notation (e.g., 'embedding.provider')\n")
        return

    try:
        # Navigate through the config object
        value = config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                console.print(f"[red]✗[/red] Configuration key '{key}' not found\n")
                return

        # Display the value
        if value is None:
            console.print(f"[dim]{key} is not set[/dim]\n")
        elif key.endswith("api_key") and value:
            console.print(f"[yellow]{key}[/yellow] = [dim]***[/dim]\n")
        else:
            console.print(f"[yellow]{key}[/yellow] = [green]{value}[/green]\n")

    except Exception as e:
        console.print(f"[red]✗[/red] Error retrieving config: {e}\n")


def set_config(key: str, value: str, project_path: Path | None = None):
    """
    Set a configuration value.

    Args:
        key: Configuration key in dot notation (e.g., "embedding.provider")
        value: Value to set
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Parse the key
    parts = key.split(".")
    if len(parts) != 2:
        console.print("[red]✗[/red] Invalid key format. Use dot notation with two parts (e.g., 'embedding.provider')\n")
        return

    section, setting = parts

    try:
        # Navigate to the correct section
        if section == "embedding":
            if not hasattr(config.embedding, setting):
                console.print(f"[red]✗[/red] Unknown embedding setting: '{setting}'\n")
                console.print("[yellow]Available settings:[/yellow] provider, model, api_key, batch_size, max_tokens\n")
                return

            # Convert value to appropriate type
            if setting == "batch_size":
                value = int(value)
            elif setting == "max_tokens":
                value = int(value) if value.lower() != "none" else None
            elif setting == "api_key" and value.lower() == "none":
                value = None
            elif setting == "model" and value.lower() == "none":
                value = None

            setattr(config.embedding, setting, value)

        elif section == "indexing":
            if not hasattr(config.indexing, setting):
                console.print(f"[red]✗[/red] Unknown indexing setting: '{setting}'\n")
                console.print(
                    "[yellow]Available settings:[/yellow] "
                    "max_files, max_total_size_mb, max_file_size_mb, chunk_size, chunk_overlap\n"
                )
                return

            # All indexing settings are integers
            value = int(value)
            setattr(config.indexing, setting, value)

        elif section == "version":
            console.print("[red]✗[/red] Cannot modify version setting\n")
            return
        elif section == "index":
            # Allow setting index.name manually
            if setting == "name":
                config.index_name = value
            else:
                console.print(f"[red]✗[/red] Cannot modify index setting: '{setting}'\n")
                console.print(
                    "[yellow]Only 'index.name' can be set manually."
                    "Other index metadata is set automatically during indexing.[/yellow]\n"
                )
                return
        else:
            console.print(f"[red]✗[/red] Unknown configuration section: '{section}'\n")
            console.print("[yellow]Available sections:[/yellow] embedding, indexing, index\n")
            return

        # Save the updated config
        config_manager.save(config)

        # Display success message
        display_value = "***" if setting == "api_key" and value else value
        console.print(f"[green]✓[/green] Set [yellow]{key}[/yellow] = [green]{display_value}[/green]\n")
        console.print(f"[dim]Config saved to: {config_manager.config_path}[/dim]\n")

    except ValueError as e:
        console.print(f"[red]✗[/red] Invalid value type: {e}\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Error setting config: {e}\n")


def unset_config(key: str, project_path: Path | None = None):
    """
    Unset (remove) a configuration value, reverting to default.

    Args:
        key: Configuration key in dot notation (e.g., "embedding.api_key")
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Parse the key
    parts = key.split(".")
    if len(parts) != 2:
        console.print("[red]✗[/red] Invalid key format. Use dot notation with two parts (e.g., 'embedding.api_key')\n")
        return

    section, setting = parts

    try:
        # Only allow unsetting optional fields
        if section == "embedding":
            if setting in ["api_key", "model", "max_tokens"]:
                setattr(config.embedding, setting, None)
            else:
                console.print(f"[red]✗[/red] Cannot unset required setting: '{setting}'\n")
                return
        elif section == "index":
            # Allow clearing index metadata
            if setting == "name":
                config_manager.clear_index_metadata()
                console.print("[green]✓[/green] Cleared all index metadata\n")
                console.print(f"[dim]Config saved to: {config_manager.config_path}[/dim]\n")
                return
            else:
                console.print(f"[red]✗[/red] Cannot unset index setting: '{setting}'\n")
                console.print("[yellow]Use 'index.name' to clear all index metadata[/yellow]\n")
                return
        else:
            console.print(f"[red]✗[/red] Cannot unset settings in section: '{section}'\n")
            console.print(
                "[yellow]Only embedding.api_key, embedding.model, embedding.max_tokens, "
                "and index.name can be unset[/yellow]\n"
            )
            return

        # Save the updated config
        config_manager.save(config)

        console.print(f"[green]✓[/green] Unset [yellow]{key}[/yellow]\n")
        console.print(f"[dim]Config saved to: {config_manager.config_path}[/dim]\n")

    except Exception as e:
        console.print(f"[red]✗[/red] Error unsetting config: {e}\n")


def show_config_file(project_path: Path | None = None):
    """
    Display the raw configuration file content.

    Args:
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    config_manager = ConfigManager(project_path)

    if not config_manager.config_path.exists():
        console.print(f"[yellow]No configuration file found at: {config_manager.config_path}[/yellow]\n")
        console.print("[dim]A default configuration will be created on first use.[/dim]\n")
        return

    try:
        with open(config_manager.config_path, encoding="utf-8") as f:
            content = f.read()

        console.print(
            Panel(
                content,
                title=f"Configuration File: {config_manager.config_path}",
                border_style="cyan",
                expand=False,
            )
        )
        console.print()

    except Exception as e:
        console.print(f"[red]✗[/red] Error reading config file: {e}\n")


def edit_config(project_path: Path | None = None):
    """
    Print instructions for manually editing the configuration file.

    Args:
        project_path: Optional project path (uses CTXAI_HOME if not provided)
    """
    config_manager = ConfigManager(project_path)

    # Ensure config exists
    config_manager.load()

    console.print("\n[bold cyan]Configuration file location:[/bold cyan]")
    console.print(f"  {config_manager.config_path}\n")

    if is_using_global_home():
        console.print("[dim]Using global CTXAI_HOME configuration[/dim]\n")
    else:
        console.print("[dim]Using project-local configuration[/dim]\n")

    console.print("[yellow]You can manually edit this file with your preferred editor.[/yellow]")
    console.print("[yellow]Or use 'ctxai config --set <key> <value>' to modify settings.[/yellow]\n")
