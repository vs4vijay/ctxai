"""
Setup script for LLM providers.

Checks configuration and provides setup instructions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

console = Console()


def check_providers():
    """Check all provider configurations."""

    console.print("\n[bold cyan]ctxai AI Coding Agent - Provider Setup[/bold cyan]\n")

    # Import factory
    from ctxai.agent.llm.factory import LLMProviderFactory

    # Check all providers
    LLMProviderFactory.print_provider_status()

    # Show recommended models
    console.print("[bold cyan]Recommended Models:[/bold cyan]\n")

    recommended = LLMProviderFactory.get_recommended_models()

    table = Table(show_header=True)
    table.add_column("Use Case", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Cost", style="magenta")

    for key, config in recommended.items():
        table.add_row(
            key.replace("_", " ").title(),
            config["provider"],
            config["model"],
            config["cost"],
        )

    console.print(table)
    console.print()

    # Show architect/editor presets
    console.print("[bold cyan]Architect/Editor Presets:[/bold cyan]\n")

    presets_info = {
        "default": "o1-mini + Claude Sonnet (best balance) - $$",
        "premium": "o1 + Claude Opus (best quality) - $$$$$",
        "budget": "GPT-4o + GPT-4o-mini (lower cost) - $",
        "cheap": "DeepSeek R1 + DeepSeek Chat (cheapest) - ¢",
        "local": "CodeLlama 34B + 13B (fully local) - Free",
        "mixed": "o1-mini + CodeLlama 13B (cloud + local) - $",
    }

    for preset, description in presets_info.items():
        console.print(f"  • [cyan]{preset:12}[/cyan] {description}")

    console.print()

    # Setup instructions
    console.print(LLMProviderFactory.get_setup_instructions())

    # Example commands
    console.print("[bold cyan]Example Commands:[/bold cyan]\n")

    examples = [
        ("OpenRouter + Claude", "ctxai chat --provider openrouter --model anthropic/claude-3.5-sonnet"),
        ("Local Ollama", "ctxai chat --provider ollama --model codellama:13b"),
        ("Architect/Editor (default)", "ctxai chat --architect-editor"),
        ("Architect/Editor (budget)", "ctxai chat --architect-editor --preset budget"),
        ("Mixed (cloud + local)", "ctxai chat --architect-editor --preset mixed"),
        ("Check provider status", "python scripts/setup_providers.py"),
    ]

    for description, command in examples:
        console.print(f"  [green]{description:25}[/green] {command}")

    console.print()


def test_provider(provider_name: str):
    """Test a specific provider."""

    console.print(f"\n[bold cyan]Testing {provider_name}...[/bold cyan]\n")

    from ctxai.agent.config import AgentLLMConfig
    from ctxai.agent.llm.factory import LLMProviderFactory

    try:
        # Check availability first
        available, message = LLMProviderFactory.check_provider_availability(provider_name)

        if not available:
            console.print(f"[red]✗ {message}[/red]")
            return False

        console.print(f"[green]✓ {message}[/green]")

        # Try to create provider
        if provider_name == "ollama":
            config = AgentLLMConfig(provider="ollama", model="codellama:13b")
        else:
            config = AgentLLMConfig(provider=provider_name)

        provider = LLMProviderFactory.create_provider(config)
        console.print(f"[green]✓ Provider created: {provider}[/green]")

        # Simple test
        console.print("\n[cyan]Running simple test...[/cyan]")
        messages = [{"role": "user", "content": "Say 'Hello from ctxai!'"}]

        response = provider.chat(messages)
        console.print(f"\n[green]Response:[/green] {response.content[:200]}")

        console.print(f"\n[bold green]✓ {provider_name} is working![/bold green]\n")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]\n")
        return False


def main():
    """Main setup script."""

    import argparse

    parser = argparse.ArgumentParser(description="Setup and test LLM providers")
    parser.add_argument(
        "--test",
        type=str,
        help="Test specific provider (openrouter, ollama, anthropic, openai)",
    )

    args = parser.parse_args()

    if args.test:
        test_provider(args.test)
    else:
        check_providers()


if __name__ == "__main__":
    main()
