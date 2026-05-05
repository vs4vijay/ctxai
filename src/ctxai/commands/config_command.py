"""
Config command implementation.
Manages configuration settings similar to git config.

Supports multi-provider configuration:
- ctxai config --list
- ctxai config --set default_provider custom
- ctxai config --set providers.custom.model zai-org/GLM-5-FP8
- ctxai config --set providers.custom.api_key your-key
- ctxai config --set providers.custom.base_url https://api.us-west-2.modal.direct/v1
"""

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import ConfigManager
from ..utils import get_ctxai_home, is_using_global_home

console = Console(legacy_windows=False)


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

    # Create main config table
    table = Table(title="Configuration Settings", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="green", no_wrap=True)
    table.add_column("Value", style="yellow")

    # General settings
    table.add_row("version", config.version)
    table.add_row("default_provider", f"[bold]{config.default_provider}[/bold]")

    # Index metadata
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

    # Embedding settings
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

    # Indexing settings
    table.add_row("indexing.max_files", str(config.indexing.max_files))
    table.add_row("indexing.max_total_size_mb", str(config.indexing.max_total_size_mb))
    table.add_row("indexing.max_file_size_mb", str(config.indexing.max_file_size_mb))
    table.add_row("indexing.chunk_size", str(config.indexing.chunk_size))
    table.add_row("indexing.chunk_overlap", str(config.indexing.chunk_overlap))

    console.print(table)

    # Print providers section
    console.print("\n[bold cyan]Providers:[/bold cyan]\n")

    providers_table = Table(show_header=True, header_style="bold cyan")
    providers_table.add_column("Provider", style="green")
    providers_table.add_column("Enabled", style="yellow")
    providers_table.add_column("Model", style="blue")
    providers_table.add_column("Base URL", style="dim")
    providers_table.add_column("API Key", style="dim")

    for name, pconfig in config.providers.items():
        enabled_str = "[green]✓[/green]" if pconfig.enabled else "[red]✗[/red]"
        model_str = pconfig.model or "[dim]default[/dim]"
        base_url_str = pconfig.base_url or "[dim]-[/dim]"
        api_key_str = "***" if pconfig.api_key else "[dim]-[/dim]"
        
        # Highlight default provider
        if name == config.default_provider:
            name_str = f"[bold]{name}[/bold] [dim](default)[/dim]"
        else:
            name_str = name

        providers_table.add_row(
            name_str,
            enabled_str,
            model_str,
            base_url_str,
            api_key_str,
        )

    console.print(providers_table)
    console.print()


def get_config(key: str, project_path: Path | None = None):
    """
    Get a specific configuration value.

    Args:
        key: Configuration key in dot notation (e.g., "providers.custom.model")
        project_path: Optional project path
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Parse the key
    parts = key.split(".")
    if len(parts) < 1:
        console.print("[red][X][/red] Invalid key format. Use dot notation (e.g., 'providers.custom.model')\n")
        return

    try:
        # Handle special cases
        if parts[0] == "providers" and len(parts) >= 2:
            provider_name = parts[1]
            pconfig = config.get_provider_config(provider_name)

            if len(parts) == 2:
                # Show all provider config
                console.print(f"[yellow]{key}[/yellow]:")
                console.print(f"  enabled: {pconfig.enabled}")
                console.print(f"  model: {pconfig.model or '[dim]default[/dim]'}")
                console.print(f"  base_url: {pconfig.base_url or '[dim]-[/dim]'}")
                console.print(f"  api_key: {'***' if pconfig.api_key else '[dim]-[/dim]'}")
                console.print(f"  temperature: {pconfig.temperature}")
                console.print(f"  max_tokens: {pconfig.max_tokens}")
            else:
                # Get specific provider setting
                setting = parts[2]
                if not hasattr(pconfig, setting):
                    console.print(f"[red][X][/red] Unknown provider setting: '{setting}'\n")
                    return

                value = getattr(pconfig, setting)
                if setting == "api_key" and value:
                    console.print(f"[yellow]{key}[/yellow] = [dim]***[/dim]\n")
                elif value is None:
                    console.print(f"[dim]{key} is not set[/dim]\n")
                else:
                    console.print(f"[yellow]{key}[/yellow] = [green]{value}[/green]\n")
        else:
            # Navigate through the config object
            value = config
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    console.print(f"[red][X][/red] Configuration key '{key}' not found\n")
                    return

            # Display the value
            if value is None:
                console.print(f"[dim]{key} is not set[/dim]\n")
            elif key.endswith("api_key") and value:
                console.print(f"[yellow]{key}[/yellow] = [dim]***[/dim]\n")
            else:
                console.print(f"[yellow]{key}[/yellow] = [green]{value}[/green]\n")

    except Exception as e:
        console.print(f"[red][X][/red] Error retrieving config: {e}\n")


def set_config(key: str, value: str, project_path: Path | None = None):
    """
    Set a configuration value.

    Args:
        key: Configuration key in dot notation
        value: Value to set
        project_path: Optional project path
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    # Parse the key
    parts = key.split(".")

    try:
        # Handle providers.*.* pattern
        if parts[0] == "providers" and len(parts) >= 3:
            provider_name = parts[1]
            setting = parts[2]

            # Convert value to appropriate type
            converted_value = value
            if setting == "enabled":
                converted_value = value.lower() in ("true", "yes", "1", "on")
            elif setting == "temperature":
                converted_value = float(value)
            elif setting == "max_tokens":
                converted_value = int(value)
            elif setting == "timeout":
                converted_value = int(value)
            elif setting in ("model", "api_key", "base_url") and value.lower() == "none":
                converted_value = None

            # Validate setting
            valid_settings = ["enabled", "model", "api_key", "base_url", "temperature", "max_tokens", "timeout"]
            if setting not in valid_settings:
                console.print(f"[red][X][/red] Unknown provider setting: '{setting}'\n")
                console.print(f"[yellow]Available settings:[/yellow] {', '.join(valid_settings)}\n")
                return

            config.set_provider_config(provider_name, **{setting: converted_value})
            config_manager.save(config)

            # Display success
            display_value = "***" if setting == "api_key" and converted_value else converted_value
            console.print(f"[green][OK][/green] Set [yellow]{key}[/yellow] = [green]{display_value}[/green]\n")
            console.print(f"[dim]Config saved to: {config_manager.config_path}[/dim]\n")

        elif len(parts) == 1:
            # Top-level setting
            if parts[0] == "default_provider":
                config.default_provider = value
                config_manager.save(config)
                console.print(f"[green][OK][/green] Set [yellow]default_provider[/yellow] = [green]{value}[/green]\n")
            else:
                console.print(f"[red][X][/red] Cannot set top-level key: '{parts[0]}'\n")
                console.print("[yellow]Available top-level keys:[/yellow] default_provider\n")

        elif len(parts) == 2:
            # Two-part key (section.setting)
            section, setting = parts

            if section == "embedding":
                if not hasattr(config.embedding, setting):
                    console.print(f"[red][X][/red] Unknown embedding setting: '{setting}'\n")
                    console.print("[yellow]Available settings:[/yellow] provider, model, api_key, batch_size, max_tokens\n")
                    return

                # Convert value
                if setting == "batch_size":
                    value = int(value)
                elif setting == "max_tokens":
                    value = int(value) if value.lower() != "none" else None
                elif setting in ("api_key", "model") and value.lower() == "none":
                    value = None

                setattr(config.embedding, setting, value)
                config_manager.save(config)
                display_value = "***" if setting == "api_key" and value else value
                console.print(f"[green][OK][/green] Set [yellow]{key}[/yellow] = [green]{display_value}[/green]\n")

            elif section == "indexing":
                if not hasattr(config.indexing, setting):
                    console.print(f"[red][X][/red] Unknown indexing setting: '{setting}'\n")
                    return

                value = int(value)
                setattr(config.indexing, setting, value)
                config_manager.save(config)
                console.print(f"[green][OK][/green] Set [yellow]{key}[/yellow] = [green]{value}[/green]\n")

            else:
                console.print(f"[red][X][/red] Unknown section: '{section}'\n")
                console.print("[yellow]Available sections:[/yellow] embedding, indexing, providers\n")

        else:
            console.print("[red][X][/red] Invalid key format.\n")
            console.print("[yellow]Examples:[/yellow]\n")
            console.print("  default_provider\n")
            console.print("  embedding.provider\n")
            console.print("  providers.custom.model\n")

    except ValueError as e:
        console.print(f"[red][X][/red] Invalid value type: {e}\n")
    except Exception as e:
        console.print(f"[red][X][/red] Error setting config: {e}\n")


def unset_config(key: str, project_path: Path | None = None):
    """
    Unset (remove) a configuration value, reverting to default.

    Args:
        key: Configuration key in dot notation
        project_path: Optional project path
    """
    config_manager = ConfigManager(project_path)
    config = config_manager.load()

    parts = key.split(".")

    try:
        if parts[0] == "providers" and len(parts) >= 3:
            provider_name = parts[1]
            setting = parts[2]

            # Set to None/default
            defaults = {
                "model": None,
                "api_key": None,
                "base_url": None,
                "enabled": True,
                "temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 60,
            }

            if setting in defaults:
                config.set_provider_config(provider_name, **{setting: defaults[setting]})
                config_manager.save(config)
                console.print(f"[green][OK][/green] Unset [yellow]{key}[/yellow]\n")
            else:
                console.print(f"[red][X][/red] Cannot unset required setting: '{setting}'\n")

        elif len(parts) == 2:
            section, setting = parts

            if section == "embedding":
                if setting in ["api_key", "model", "max_tokens"]:
                    setattr(config.embedding, setting, None)
                    config_manager.save(config)
                    console.print(f"[green][OK][/green] Unset [yellow]{key}[/yellow]\n")
                else:
                    console.print(f"[red][X][/red] Cannot unset required setting: '{setting}'\n")

            elif section == "index":
                if setting == "name":
                    config_manager.clear_index_metadata()
                    console.print("[green][OK][/green] Cleared all index metadata\n")
                else:
                    console.print(f"[red][X][/red] Cannot unset index setting: '{setting}'\n")

            else:
                console.print(f"[red][X][/red] Cannot unset settings in section: '{section}'\n")

        else:
            console.print(f"[red][X][/red] Invalid key format: '{key}'\n")

    except Exception as e:
        console.print(f"[red][X][/red] Error unsetting config: {e}\n")


def show_config_file(project_path: Path | None = None):
    """Display the raw configuration file content."""
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
        console.print(f"[red][X][/red] Error reading config file: {e}\n")


def edit_config(project_path: Path | None = None):
    """Print instructions for manually editing the configuration file."""
    config_manager = ConfigManager(project_path)
    config_manager.load()

    console.print("\n[bold cyan]Configuration file location:[/bold cyan]")
    console.print(f"  {config_manager.config_path}\n")

    if is_using_global_home():
        console.print("[dim]Using global CTXAI_HOME configuration[/dim]\n")
    else:
        console.print("[dim]Using project-local configuration[/dim]\n")

    console.print("[yellow]You can manually edit this file with your preferred editor.[/yellow]")
    console.print("[yellow]Or use 'ctxai config --set <key> <value>' to modify settings.[/yellow]\n")

    console.print("[bold]Multi-provider configuration example:[/bold]\n")
    console.print("""[dim]{
  "version": "2.0",
  "default_provider": "custom",
  "providers": {
    "openrouter": {
      "model": "anthropic/claude-3.5-sonnet",
      "enabled": true
    },
    "custom": {
      "enabled": true,
      "model": "zai-org/GLM-5-FP8",
      "api_key": "your-api-key",
      "base_url": "https://api.us-west-2.modal.direct/v1"
    }
  },
  "embedding": {...},
  "indexing": {...}
}[/dim]
""")
