"""
OAuth PKCE (Proof Key for Code Exchange) implementation for OpenRouter.

This module implements the OAuth 2.0 PKCE flow for secure authentication with
OpenRouter without requiring a client secret.
"""

import base64
import hashlib
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from rich.console import Console


console = Console()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    authorization_code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self):
        """Handle GET request to callback endpoint."""
        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)

        if "code" in params:
            # Success - got authorization code
            OAuthCallbackHandler.authorization_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>✅ Authentication Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

        elif "error" in params:
            # Error occurred
            OAuthCallbackHandler.error = params["error"][0]
            error_desc = params.get("error_description", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = f"""
            <html>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>❌ Authentication Failed</h1>
                <p>{error_desc}</p>
                <p>You can close this window and try again.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

        else:
            self.send_response(400)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Invalid callback")

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


class PKCEFlow:
    """
    OAuth PKCE Flow implementation.

    Implements the complete OAuth 2.0 PKCE flow for OpenRouter:
    1. Generate code verifier and challenge
    2. Redirect user to authorization URL
    3. Receive callback with authorization code
    4. Exchange code for API key
    """

    OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
    OPENROUTER_TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
    CALLBACK_PORT = 8080  # Default local port for callback

    def __init__(self, callback_port: int = CALLBACK_PORT):
        """
        Initialize PKCE flow.

        Args:
            callback_port: Port to run local callback server on
        """
        self.callback_port = callback_port
        self.code_verifier = self._generate_code_verifier()
        self.code_challenge = self._generate_code_challenge(self.code_verifier)

    def _generate_code_verifier(self, length: int = 128) -> str:
        """
        Generate a cryptographically random code verifier.

        Args:
            length: Length of verifier (43-128 characters)

        Returns:
            Code verifier string
        """
        if not 43 <= length <= 128:
            raise ValueError("Code verifier length must be between 43 and 128")

        # Generate random bytes and encode as base64url
        random_bytes = secrets.token_bytes(96)  # 96 bytes = 128 base64url chars
        verifier = base64.urlsafe_b64encode(random_bytes).decode("utf-8")

        # Remove padding and truncate to desired length
        verifier = verifier.replace("=", "")[:length]

        return verifier

    def _generate_code_challenge(self, verifier: str) -> str:
        """
        Generate code challenge from verifier using S256 method.

        Args:
            verifier: Code verifier

        Returns:
            Base64url-encoded SHA256 hash of verifier
        """
        # SHA256 hash
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()

        # Base64url encode
        challenge = base64.urlsafe_b64encode(digest).decode("utf-8")

        # Remove padding
        challenge = challenge.replace("=", "")

        return challenge

    def get_authorization_url(self) -> str:
        """
        Get the authorization URL to redirect user to.

        Returns:
            Authorization URL with PKCE parameters
        """
        callback_url = f"http://localhost:{self.callback_port}/callback"

        params = {
            "callback_url": callback_url,
            "code_challenge": self.code_challenge,
            "code_challenge_method": "S256",
        }

        return f"{self.OPENROUTER_AUTH_URL}?{urlencode(params)}"

    def start_callback_server(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Start local HTTP server to receive OAuth callback.

        Returns:
            Tuple of (success, authorization_code, error_message)
        """
        # Reset class variables
        OAuthCallbackHandler.authorization_code = None
        OAuthCallbackHandler.error = None

        server = HTTPServer(("localhost", self.callback_port), OAuthCallbackHandler)

        console.print(f"\n[dim]Listening for callback on port {self.callback_port}...[/dim]")

        # Handle a single request
        server.handle_request()

        # Get results
        code = OAuthCallbackHandler.authorization_code
        error = OAuthCallbackHandler.error

        if error:
            return False, None, error

        if code:
            return True, code, None

        return False, None, "No authorization code received"

    def exchange_code_for_key(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Exchange authorization code for API key.

        Args:
            code: Authorization code from callback

        Returns:
            Tuple of (success, api_key, error_message)
        """
        try:
            payload = {
                "code": code,
                "code_verifier": self.code_verifier,
                "code_challenge_method": "S256",
            }

            headers = {
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo/ctxai",
                "X-Title": "ctxai - AI Coding Agent",
            }

            response = requests.post(
                self.OPENROUTER_TOKEN_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                api_key = data.get("key")

                if api_key:
                    return True, api_key, None
                else:
                    return False, None, "No API key in response"

            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return False, None, error_msg

        except requests.RequestException as e:
            return False, None, f"Request failed: {str(e)}"

    def run_flow(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run the complete OAuth PKCE flow.

        This will:
        1. Open the authorization URL in browser
        2. Start local callback server
        3. Wait for authorization
        4. Exchange code for API key

        Returns:
            Tuple of (success, api_key, error_message)
        """
        console.print("\n[cyan]Starting OpenRouter OAuth authentication...[/cyan]\n")

        # Step 1: Get authorization URL
        auth_url = self.get_authorization_url()

        console.print("[bold]Step 1:[/bold] Opening browser for authorization...")
        console.print(f"[dim]If browser doesn't open, visit: {auth_url}[/dim]\n")

        # Open browser
        webbrowser.open(auth_url)

        # Step 2: Start callback server and wait for code
        console.print("[bold]Step 2:[/bold] Waiting for authorization...")

        success, code, error = self.start_callback_server()

        if not success:
            return False, None, error

        # Step 3: Exchange code for API key
        console.print("\n[bold]Step 3:[/bold] Exchanging code for API key...")

        success, api_key, error = self.exchange_code_for_key(code)

        if not success:
            return False, None, error

        console.print("\n[green]Authentication successful![/green]")

        return True, api_key, None


def authenticate_with_openrouter(callback_port: int = 8080) -> Optional[str]:
    """
    Authenticate with OpenRouter using OAuth PKCE flow.

    Args:
        callback_port: Port for local callback server

    Returns:
        API key if successful, None otherwise
    """
    flow = PKCEFlow(callback_port=callback_port)

    success, api_key, error = flow.run_flow()

    if not success:
        console.print(f"\n[red]Authentication failed: {error}[/red]")
        return None

    return api_key
