"""
Models command - List available LLM models with capabilities.
"""

import os
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

from ..auth.keystore import get_keystore

console = Console()


def get_api_key(provider: str) -> str | None:
    """Get API key from environment or keystore."""
    env_mapping = {
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "github-copilot": "GITHUB_COPILOT_TOKEN",
        "ollama": None,  # Ollama doesn't use API keys
        "custom": "CUSTOM_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }

    env_var = env_mapping.get(provider)
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    if provider == "ollama":
        return None  # Ollama doesn't need an API key

    keystore = get_keystore()
    return keystore.get_key(provider)


def list_openrouter_models() -> list[dict]:
    """List models available via OpenRouter API."""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "name": m.get("id", ""),
                    "description": m.get("description", ""),
                    "context_length": m.get("context_length", 0),
                    "pricing": m.get("pricing", {}),
                    "top_provider": m.get("top_provider", {}),
                }
                for m in data.get("data", [])
            ]
    except Exception:
        pass
    return []


def list_ollama_models() -> list[dict]:
    """List models available in Ollama."""
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in data.get("models", [])
            ]
    except Exception:
        pass
    return []


def list_openai_models() -> list[dict]:
    """List models available via OpenAI API."""
    api_key = get_api_key("openai")
    if not api_key:
        return []

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        models = client.models.list()
        return [
            {
                "name": m.id,
                "created": m.created,
            }
            for m in models.data
            if "gpt" in m.id or "o1" in m.id or "o3" in m.id
        ]
    except Exception:
        pass
    return []


def get_ollama_model_info(model_name: str) -> dict | None:
    """Get detailed info for an Ollama model."""
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/show", json={"name": model_name}, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


PROVIDER_CAPABILITIES = {
    "openrouter": {
        "name": "OpenRouter",
        "list_models": True,
        "description": "100+ models (Claude, GPT-4o, DeepSeek, Llama, etc.)",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "free_models": True,
    },
    "ollama": {
        "name": "Ollama",
        "list_models": True,
        "description": "Local models (Llama, CodeLlama, DeepSeek, Qwen, etc.)",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "free_models": True,
    },
    "anthropic": {
        "name": "Anthropic",
        "list_models": False,
        "description": "Claude models (Sonnet, Opus, Haiku)",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "free_models": False,
    },
    "openai": {
        "name": "OpenAI",
        "list_models": True,
        "description": "GPT-4o, o1, o3 models",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "free_models": False,
    },
    "github-copilot": {
        "name": "GitHub Copilot",
        "list_models": False,
        "description": "GPT-4, Claude via Copilot subscription",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "free_models": False,
    },
    "custom": {
        "name": "Custom",
        "list_models": False,
        "description": "OpenAI-compatible endpoints",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": None,
        "free_models": None,
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "list_models": False,
        "description": "NVIDIA NIM endpoints",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": None,
        "free_models": False,
    },
}


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def list_models(
    provider: str | None = None,
    limit: int = 20,
    show_all: bool = False,
):
    """
    List available models for a provider or all providers.

    Args:
        provider: Specific provider to list (or None for all)
        limit: Max models per provider to show
        show_all: Show all models (no limit)
    """
    providers = [provider.lower()] if provider else list(PROVIDER_CAPABILITIES.keys())

    for p in providers:
        if p not in PROVIDER_CAPABILITIES:
            console.print(f"[red]Unknown provider: {p}[/red]")
            continue

        cap = PROVIDER_CAPABILITIES[p]
        api_key = get_api_key(p)

        console.print(f"\n[bold cyan]Provider: {cap['name']}[/bold cyan]")
        console.print(f"  Description: {cap['description']}")

        if api_key:
            console.print("  [green]Status: Configured[/green]")
        elif p == "ollama":
            console.print("  [dim]Status: Local (no API key needed)[/dim]")
        else:
            console.print(f"  [yellow]Status: Not configured (set {p.upper()}_API_KEY)[/yellow]")

        console.print("  Capabilities:")
        console.print(f"    Streaming: {'Yes' if cap['supports_streaming'] else 'No'}")
        console.print(f"    Tool Calling: {'Yes' if cap['supports_tools'] else 'No'}")
        console.print(f"    Vision: {'Yes' if cap['supports_vision'] else 'No'}")
        console.print(f"    Free Models: {'Yes' if cap['free_models'] else 'No' if cap['free_models'] is False else '?'}")

        if not cap["list_models"]:
            console.print("  [dim]Model listing not available via API[/dim]")
            console.print("  Known models: See documentation for available models")
            continue

        console.print("  Models:")

        if p == "openrouter":
            models = list_openrouter_models()
            if not models:
                console.print("    [yellow]Failed to fetch models[/yellow]")
                continue

            if not show_all and len(models) > limit:
                console.print(f"    Showing {limit} of {len(models)} models (use --all to see all)")
                models = models[:limit]

            table = Table(show_header=True, header_style="bold")
            table.add_column("Model", style="cyan")
            table.add_column("Context", justify="right")
            table.add_column("Input/1M", justify="right")
            table.add_column("Output/1M", justify="right")

            for m in models:
                ctx = m.get("context_length", 0) or 0
                pricing = m.get("pricing", {})
                input_price = pricing.get("prompt", "0")
                output_price = pricing.get("completion", "0")

                try:
                    input_price = f"${float(input_price):.2f}" if input_price else "N/A"
                except (ValueError, TypeError):
                    input_price = "N/A"
                try:
                    output_price = f"${float(output_price):.2f}" if output_price else "N/A"
                except (ValueError, TypeError):
                    output_price = "N/A"

                table.add_row(
                    m.get("name", "")[:50],
                    f"{ctx // 1000}k" if ctx else "N/A",
                    input_price,
                    output_price,
                )

            console.print(table)

        elif p == "ollama":
            models = list_ollama_models()
            if not models:
                console.print("    [yellow]Ollama not running or no models installed[/yellow]")
                console.print("    Run 'ollama pull <model>' to download models")
                continue

            if not show_all and len(models) > limit:
                console.print(f"    Showing {limit} of {len(models)} models (use --all to see all)")
                models = models[:limit]

            table = Table(show_header=True, header_style="bold")
            table.add_column("Model", style="cyan")
            table.add_column("Size", justify="right")

            for m in models:
                table.add_row(
                    m.get("name", ""),
                    format_size(m.get("size", 0)),
                )

            console.print(table)

        elif p == "openai":
            models = list_openai_models()
            if not models:
                console.print("    [yellow]Failed to fetch models (check OPENAI_API_KEY)[/yellow]")
                continue

            if not show_all and len(models) > limit:
                console.print(f"    Showing {limit} of {len(models)} models (use --all to see all)")
                models = models[:limit]

            table = Table(show_header=True, header_style="bold")
            table.add_column("Model", style="cyan")
            table.add_column("Created", justify="right")

            for m in models:
                from datetime import datetime

                created = m.get("created", 0)
                if created:
                    created = datetime.fromtimestamp(created).strftime("%Y-%m-%d")
                else:
                    created = "N/A"

                table.add_row(
                    m.get("name", ""),
                    created,
                )

            console.print(table)


def show_model_details(model_id: str, provider: str | None = None):
    """Show detailed information about a specific model."""
    if provider:
        providers = [provider.lower()]
    else:
        providers = list(PROVIDER_CAPABILITIES.keys())

    found = False

    for p in providers:
        if p not in PROVIDER_CAPABILITIES:
            continue

        if p == "openrouter":
            models = list_openrouter_models()
            for m in models:
                if model_id.lower() in m.get("name", "").lower():
                    console.print(f"\n[bold cyan]Model: {m.get('name')}[/bold cyan]")
                    console.print("  Provider: OpenRouter")

                    desc = m.get("description", "")
                    if desc:
                        console.print(f"  Description: {desc[:200]}...")

                    ctx = m.get("context_length", 0)
                    console.print(f"  Context Length: {ctx:,} tokens" if ctx else "  Context Length: N/A")

                    pricing = m.get("pricing", {})
                    if pricing:
                        input_price = pricing.get("prompt", "0")
                        output_price = pricing.get("completion", "0")
                        try:
                            input_price = f"${float(input_price):.4f}"
                        except (ValueError, TypeError):
                            input_price = "N/A"
                        try:
                            output_price = f"${float(output_price):.4f}"
                        except (ValueError, TypeError):
                            output_price = "N/A"
                        console.print("  Pricing (per 1M tokens):")
                        console.print(f"    Input: {input_price}")
                        console.print(f"    Output: {output_price}")

                    top_provider = m.get("top_provider", {})
                    if top_provider:
                        console.print(f"  Top Provider: {top_provider.get('provider', 'N/A')}")

                    found = True
                    break

        elif p == "ollama":
            models = list_ollama_models()
            for m in models:
                if model_id.lower() in m.get("name", "").lower():
                    console.print(f"\n[bold cyan]Model: {m.get('name')}[/bold cyan]")
                    console.print("  Provider: Ollama (Local)")
                    console.print(f"  Size: {format_size(m.get('size', 0))}")

                    details = get_ollama_model_info(m.get("name"))
                    if details:
                        if details.get("parameter_size"):
                            console.print(f"  Parameters: {details.get('parameter_size')}")
                        if details.get("quantization"):
                            console.print(f"  Quantization: {details.get('quantization')}")
                        if details.get("description"):
                            console.print(f"  Description: {details.get('description')}")

                    found = True
                    break

    if not found:
        console.print(f"[yellow]Model '{model_id}' not found[/yellow]")


def search_models(query: str, provider: str | None = None, limit: int = 20):
    """
    Search for models by name or description.

    Args:
        query: Search query
        provider: Specific provider to search in
        limit: Max results to show
    """
    if provider:
        providers = [provider.lower()]
    else:
        providers = [p for p in PROVIDER_CAPABILITIES.keys() if PROVIDER_CAPABILITIES[p].get("list_models")]

    query_lower = query.lower()
    results = []

    for p in providers:
        if p == "openrouter":
            models = list_openrouter_models()
            for m in models:
                name = m.get("name", "").lower()
                desc = m.get("description", "").lower()
                if query_lower in name or query_lower in desc:
                    results.append({
                        "provider": "OpenRouter",
                        "model": m.get("name", ""),
                        "description": m.get("description", ""),
                    })

        elif p == "ollama":
            models = list_ollama_models()
            for m in models:
                name = m.get("name", "").lower()
                if query_lower in name:
                    details = get_ollama_model_info(m.get("name"))
                    desc = details.get("description", "") if details else ""
                    results.append({
                        "provider": "Ollama",
                        "model": m.get("name", ""),
                        "description": desc,
                    })

        elif p == "openai":
            models = list_openai_models()
            for m in models:
                name = m.get("name", "").lower()
                if query_lower in name:
                    results.append({
                        "provider": "OpenAI",
                        "model": m.get("name", ""),
                        "description": "GPT model",
                    })

    if not results:
        console.print(f"[yellow]No models found matching '{query}'[/yellow]")
        return

    console.print(f"\n[bold]Search results for '{query}' ({len(results)} found):[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Description", max_width=60)

    for r in results[:limit]:
        table.add_row(
            r["provider"],
            r["model"],
            r["description"][:60] if r["description"] else "",
        )

    console.print(table)

    if len(results) > limit:
        console.print(f"\n[dim]Showing {limit} of {len(results)} results[/dim]")


def pull_ollama_model(model_name: str, verbose: bool = False):
    """
    Pull a model from Ollama library.

    Args:
        model_name: Name of the model to pull
        verbose: Show detailed progress
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        import json
        import threading

        console.print(f"[cyan]Pulling model: {model_name}[/cyan]")

        response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=None,
        )

        if response.status_code != 200:
            console.print(f"[red]Failed to pull model: {response.status_code}[/red]")
            return False

        status_messages = set()
        total = None

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)

                    status = data.get("status", "")
                    digest = data.get("digest", "")[:20] if data.get("digest") else ""

                    if status == "pulling manifest":
                        if status not in status_messages:
                            console.print(f"  [dim]{status}...[/dim]")
                            status_messages.add(status)

                    elif status == "downloading":
                        if verbose:
                            done = data.get("completed", 0)
                            total = data.get("total", total or 0)
                            if total > 0:
                                pct = int(done / total * 100)
                                console.print(f"  [cyan]Downloading: {pct}%[/cyan]", end="\r")

                    elif status == "verifying":
                        if status not in status_messages:
                            console.print(f"  [dim]{status}...[/dim]")
                            status_messages.add(status)

                    elif status == "writing manifest":
                        if status not in status_messages:
                            console.print(f"  [dim]{status}...[/dim]")
                            status_messages.add(status)

                    elif status == "success":
                        console.print(f"\n[green]Successfully pulled {model_name}[/green]")
                        return True

                except json.JSONDecodeError:
                    continue

        console.print(f"\n[green]Successfully pulled {model_name}[/green]")
        return True

    except requests.exceptions.ConnectionError:
        console.print(f"[red]Cannot connect to Ollama at {base_url}[/red]")
        console.print("[dim]Make sure Ollama is running (ollama serve)[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]Error pulling model: {e}[/red]")
        return False


def list_ollama_library_models(limit: int = 30) -> list[dict]:
    """
    List popular models available in Ollama library.
    These can be pulled with 'ollama pull'.

    Note: Ollama doesn't have a public API for library models,
    so this returns a curated list of popular models.
    """
    popular_models = [
        {"name": "llama3.1", "description": "Meta's latest Llama model (8B, 70B)"},
        {"name": "llama3", "description": "Meta Llama 3 (8B, 70B)"},
        {"name": "llama2", "description": "Meta Llama 2 (7B, 13B, 70B)"},
        {"name": "codellama", "description": "Code-optimized Llama (7B, 13B, 34B)"},
        {"name": "mistral", "description": "Mistral AI's model (7B)"},
        {"name": "mixtral", "description": "Mistral's mixture of experts (8x7B)"},
        {"name": "phi3", "description": "Microsoft Phi-3 (mini, small, medium)"},
        {"name": "gemma", "description": "Google Gemma (2B, 7B)"},
        {"name": "gemma2", "description": "Google Gemma 2 (9B, 27B)"},
        {"name": "qwen2.5", "description": "Alibaba Qwen 2.5 (various sizes)"},
        {"name": "qwen2.5-coder", "description": "Qwen code-optimized models"},
        {"name": "deepseek-coder", "description": "DeepSeek code models (6.7B, 33B)"},
        {"name": "deepseek-llm", "description": "DeepSeek LLM (7B, 67B)"},
        {"name": "azure", "description": "Azure Mistral"},
        {"name": "command-r", "description": "Cohere Command R"},
        {"name": "command-r-plus", "description": "Cohere Command R+"},
        {"name": "aya", "description": "Cohere Aya"},
        {"name": "starling-lm", "description": "Starling LM"},
        {"name": "nexus", "description": "Nexus Raven"},
        {"name": "wizardlm2", "description": "WizardLM 2"},
        {"name": "wizardcoder", "description": "WizardLM for code"},
        {"name": "yi", "description": "Yi models (6B, 9B, 34B)"},
        {"name": "hermes3", "description": "Nous Hermes 3"},
        {"name": "orca-mini", "description": "Orca Mini"},
        {"name": "tinyllama", "description": "TinyLlama (1.1B) - very small"},
        {"name": "llama3.2", "description": "Llama 3.2 with vision (11B, 90B)"},
        {"name": "llava", "description": "Llava vision model"},
        {"name": "llava-llama3", "description": "Llava with Llama 3"},
        {"name": "bakllava", "description": "BakLLaVA vision model"},
        {"name": "minicpm-v", "description": "MiniCPM Vision"},
    ]
    return popular_models[:limit]