"""
GitHub Copilot OAuth Device Code Flow implementation.

Implements the OAuth 2.0 device authorization grant for GitHub Copilot,
allowing CLI authentication without browser interaction.
"""

import json
import time
from typing import Optional, Tuple

import requests
from rich.console import Console
from rich.panel import Panel

console = Console()


class GitHubCopilotAuth:
    """
    GitHub Copilot authentication using OAuth Device Code Flow.

    This implements the device authorization grant type, designed for
    CLI applications that can't easily redirect to a browser.
    """

    # GitHub OAuth endpoints
    GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
    GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

    # GitHub Copilot OAuth client ID (public)
    CLIENT_ID = "Iv1.b507a08c87ecfe98"
    SCOPE = "read:user"

    # User agent headers to identify as VS Code
    USER_AGENT = "GitHubCopilotChat/0.35.0"
    EDITOR_VERSION = "vscode/1.99.3"
    EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"

    def __init__(self):
        """Initialize GitHub Copilot authentication."""
        self.device_code: str | None = None
        self.user_code: str | None = None
        self.verification_uri: str | None = None
        self.interval: int = 5
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    def get_device_code(self) -> tuple[bool, str | None]:
        """
        Step 1: Request a device code from GitHub.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            payload = {
                "client_id": self.CLIENT_ID,
                "scope": self.SCOPE,
            }

            headers = {
                "Accept": "application/json",
                "User-Agent": self.USER_AGENT,
            }

            response = requests.post(
                self.GITHUB_DEVICE_CODE_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.text}"

            data = response.json()

            self.device_code = data.get("device_code")
            self.user_code = data.get("user_code")
            self.verification_uri = data.get("verification_uri")
            self.interval = data.get("interval", 5)

            if not all([self.device_code, self.user_code, self.verification_uri]):
                return False, "Incomplete device code response"

            return True, None

        except requests.RequestException as e:
            return False, f"Request failed: {str(e)}"
        except (json.JSONDecodeError, KeyError) as e:
            return False, f"Invalid response: {str(e)}"

    def poll_for_token(self, timeout: int = 300) -> tuple[bool, str | None]:
        """
        Step 2: Poll GitHub for access token after user authorizes.

        Args:
            timeout: Maximum time to poll in seconds (default: 5 minutes)

        Returns:
            Tuple of (success, error_message)
        """
        if not self.device_code:
            return False, "No device code available"

        start_time = time.time()

        console.print("\n[bold]Waiting for authorization...[/bold]")
        console.print("[dim](This may take a few minutes)[/dim]\n")

        while time.time() - start_time < timeout:
            try:
                payload = {
                    "client_id": self.CLIENT_ID,
                    "device_code": self.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }

                headers = {
                    "Accept": "application/json",
                    "User-Agent": self.USER_AGENT,
                }

                response = requests.post(
                    self.GITHUB_ACCESS_TOKEN_URL,
                    json=payload,
                    headers=headers,
                    timeout=30,
                )

                data = response.json()

                # Check for errors
                if "error" in data:
                    error = data["error"]

                    if error == "authorization_pending":
                        # Still waiting for user to authorize
                        time.sleep(self.interval)
                        continue

                    elif error == "slow_down":
                        # Rate limited, increase interval
                        self.interval += 5
                        time.sleep(self.interval)
                        continue

                    elif error == "expired_token":
                        return False, "Device code expired. Please try again."

                    elif error == "access_denied":
                        return False, "Authorization denied by user."

                    else:
                        return False, f"OAuth error: {error}"

                # Success! Got tokens
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")

                if not self.access_token:
                    return False, "No access token in response"

                return True, None

            except requests.RequestException as e:
                return False, f"Request failed: {str(e)}"
            except (json.JSONDecodeError, KeyError) as e:
                return False, f"Invalid response: {str(e)}"

        return False, "Authorization timed out"

    def get_copilot_token(self) -> tuple[bool, str | None, dict | None]:
        """
        Step 3: Exchange GitHub access token for Copilot API token.

        Returns:
            Tuple of (success, error_message, token_data)
        """
        if not self.access_token:
            return False, "No access token available", None

        try:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "User-Agent": self.USER_AGENT,
                "Editor-Version": self.EDITOR_VERSION,
                "Editor-Plugin-Version": self.EDITOR_PLUGIN_VERSION,
                "Accept": "application/json",
            }

            response = requests.get(
                self.GITHUB_COPILOT_TOKEN_URL,
                headers=headers,
                timeout=30,
            )

            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.text}", None

            data = response.json()

            copilot_token = data.get("token")
            if not copilot_token:
                return False, "No Copilot token in response", None

            # Return the full token data (includes expiry, etc.)
            return True, None, {
                "token": copilot_token,
                "expires_at": data.get("expires_at"),
                "refresh_token": self.refresh_token,
                "access_token": self.access_token,
            }

        except requests.RequestException as e:
            return False, f"Request failed: {str(e)}", None
        except (json.JSONDecodeError, KeyError) as e:
            return False, f"Invalid response: {str(e)}", None

    def run_flow(self) -> tuple[bool, dict | None, str | None]:
        """
        Run the complete OAuth device code flow.

        Returns:
            Tuple of (success, token_data, error_message)
        """
        console.print("\n[cyan]Starting GitHub Copilot authentication...[/cyan]\n")

        # Step 1: Get device code
        console.print("[bold]Step 1:[/bold] Requesting device code...")
        success, error = self.get_device_code()

        if not success:
            return False, None, error

        # Display user code and verification URL
        console.print("[green]Device code received![/green]\n")

        panel = Panel(
            f"[bold cyan]User Code:[/bold cyan] [yellow]{self.user_code}[/yellow]\n\n"
            f"[bold cyan]Verification URL:[/bold cyan] {self.verification_uri}",
            title="GitHub Authorization",
            border_style="cyan",
        )
        console.print(panel)

        console.print(
            f"\n[bold]Step 2:[/bold] Please visit [link={self.verification_uri}]{self.verification_uri}[/link]"
        )
        console.print(f"         and enter code: [yellow]{self.user_code}[/yellow]\n")

        # Step 2: Poll for access token
        success, error = self.poll_for_token()

        if not success:
            return False, None, error

        console.print("[green]Authorization successful![/green]\n")

        # Step 3: Get Copilot token
        console.print("[bold]Step 3:[/bold] Getting Copilot API token...")

        success, error, token_data = self.get_copilot_token()

        if not success:
            return False, None, error

        console.print("[green]Copilot token received![/green]\n")

        return True, token_data, None


def authenticate_with_github_copilot() -> dict | None:
    """
    Authenticate with GitHub Copilot using OAuth device code flow.

    Returns:
        Token data dict if successful, None otherwise
    """
    auth = GitHubCopilotAuth()

    success, token_data, error = auth.run_flow()

    if not success:
        console.print(f"\n[red]Authentication failed: {error}[/red]")
        return None

    return token_data
