"""
Secure API key storage for ctxai.

Stores API keys in the user's config directory with appropriate permissions.
"""

import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console


console = Console()


class KeyStore:
    """Manages secure storage of API keys."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize key store.

        Args:
            config_dir: Custom config directory (default: ~/.ctxai)
        """
        if config_dir is None:
            config_dir = Path.home() / ".ctxai"

        self.config_dir = Path(config_dir)
        self.keystore_file = self.config_dir / "keys.json"

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions (owner read/write only)
        if os.name != "nt":  # Unix-like systems
            os.chmod(self.config_dir, 0o700)

    def _load_keys(self) -> dict:
        """Load keys from keystore file."""
        if not self.keystore_file.exists():
            return {}

        try:
            with open(self.keystore_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            console.print(f"[yellow]Warning: Could not load keystore: {e}[/yellow]")
            return {}

    def _save_keys(self, keys: dict):
        """Save keys to keystore file."""
        try:
            with open(self.keystore_file, "w") as f:
                json.dump(keys, f, indent=2)

            # Set restrictive permissions (owner read/write only)
            if os.name != "nt":  # Unix-like systems
                os.chmod(self.keystore_file, 0o600)

        except IOError as e:
            console.print(f"[red]Error saving keystore: {e}[/red]")

    def set_key(self, provider: str, api_key: str | dict):
        """
        Store an API key or token data for a provider.

        Args:
            provider: Provider name (e.g., 'openrouter', 'anthropic', 'github-copilot')
            api_key: API key string or token data dict to store
        """
        keys = self._load_keys()
        keys[provider] = api_key
        self._save_keys(keys)

        console.print(f"[green]Saved API key for {provider}[/green]")

    def get_key(self, provider: str) -> Optional[str]:
        """
        Get an API key for a provider.

        Args:
            provider: Provider name

        Returns:
            API key if found, None otherwise
        """
        keys = self._load_keys()
        return keys.get(provider)

    def delete_key(self, provider: str) -> bool:
        """
        Delete an API key for a provider.

        Args:
            provider: Provider name

        Returns:
            True if deleted, False if not found
        """
        keys = self._load_keys()

        if provider in keys:
            del keys[provider]
            self._save_keys(keys)
            console.print(f"[green]Deleted API key for {provider}[/green]")
            return True

        console.print(f"[yellow]No API key found for {provider}[/yellow]")
        return False

    def list_providers(self) -> list[str]:
        """
        List all providers with stored keys.

        Returns:
            List of provider names
        """
        keys = self._load_keys()
        return list(keys.keys())

    def clear_all(self):
        """Clear all stored keys."""
        if self.keystore_file.exists():
            self.keystore_file.unlink()
            console.print("[green]Cleared all API keys[/green]")
        else:
            console.print("[yellow]No keys to clear[/yellow]")


def get_keystore() -> KeyStore:
    """Get the default keystore instance."""
    return KeyStore()
