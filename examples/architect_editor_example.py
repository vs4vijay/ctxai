"""
Example: Using OpenRouter + Ollama with Architect/Editor pattern.

This demonstrates:
1. OpenRouter for cloud models (architect + editor)
2. Ollama for local models
3. Architect/Editor pattern for quality + cost optimization
4. Repository mapping for context
"""

import asyncio
import os
from pathlib import Path

from rich.console import Console

console = Console()


async def example_openrouter_architect_editor():
    """Example using OpenRouter with architect/editor pattern."""

    console.print("\n[bold cyan]Example 1: OpenRouter Architect/Editor Pattern[/bold cyan]\n")

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("❌ [red]OPENROUTER_API_KEY not set[/red]")
        console.print("Get your key at: https://openrouter.ai/keys")
        return

    from ctxai.agent.architect_editor import create_architect_editor_agent

    # Create agent with:
    # - Architect: o1-mini (reasoning model)
    # - Editor: Claude Sonnet (fast, high quality)
    agent = create_architect_editor_agent(
        architect_model="openai/o1-mini",  # Reasoning for planning
        editor_model="anthropic/claude-3.5-sonnet",  # Fast implementation
    )

    # Example task
    task = "Add error handling to all API calls in the project"
    context = {
        "working_directory": Path.cwd(),
        "tools": ["read_file", "edit_file", "grep", "bash_exec"],
    }

    console.print(f"[green]Task:[/green] {task}\n")

    # Process with architect/editor
    result = await agent.process_task(task, context, tools=[])

    console.print("\n[bold green]✓ Task completed![/bold green]")
    console.print(f"Used architect: {result.get('architect_used')}")
    console.print(f"Steps completed: {len(result.get('results', []))}")


async def example_ollama_local():
    """Example using Ollama for local execution."""

    console.print("\n[bold cyan]Example 2: Ollama Local Models[/bold cyan]\n")

    try:
        from ctxai.agent.config import AgentLLMConfig
        from ctxai.agent.llm.ollama_provider import OLLAMA_CODING_MODELS, OllamaProvider

        # Create Ollama provider
        config = AgentLLMConfig(
            provider="ollama",
            model="codellama:13b",  # or any other model you have
            base_url="http://localhost:11434",
        )

        ollama = OllamaProvider(config)

        console.print("[green]✓ Connected to Ollama[/green]")
        console.print(f"Model: {ollama.model}")

        # List available models
        models = ollama.list_available_models()
        if models:
            console.print(f"\nAvailable models: {', '.join(models[:5])}")

        # Simple chat example
        messages = [
            {"role": "user", "content": "Write a Python function to check if a number is prime"}
        ]

        console.print("\n[cyan]Generating code...[/cyan]")
        response = ollama.chat(messages)

        console.print("\n[green]Response:[/green]")
        console.print(response.content[:500])

    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        console.print("\nMake sure Ollama is running: ollama serve")
        console.print("Pull a model: ollama pull codellama:13b")


async def example_repository_mapping():
    """Example using repository mapping."""

    console.print("\n[bold cyan]Example 3: Repository Mapping[/bold cyan]\n")

    from ctxai.agent.repomap import create_repository_map

    # Create repository map
    console.print("[cyan]Creating repository map...[/cyan]")

    repo_path = Path.cwd()
    repo_map = create_repository_map(repo_path, max_tokens=500)

    console.print("\n[green]Repository Map:[/green]")
    console.print(repo_map[:1000])  # Show first 1000 chars

    console.print("\n[dim]This map provides context to the LLM about your codebase[/dim]")


async def example_mixed_providers():
    """Example mixing OpenRouter architect + Ollama editor."""

    console.print("\n[bold cyan]Example 4: Mixed Providers (OpenRouter + Ollama)[/bold cyan]\n")

    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("❌ [red]OPENROUTER_API_KEY not set[/red]")
        return

    try:
        from ctxai.agent.architect_editor import ArchitectEditorAgent, ArchitectEditorConfig
        from ctxai.agent.config import AgentLLMConfig
        from ctxai.agent.llm.ollama_provider import OllamaProvider
        from ctxai.agent.llm.openrouter_provider import OpenRouterProvider

        # Architect: OpenRouter with reasoning model
        architect_config = AgentLLMConfig(
            provider="openrouter",
            model="openai/o1-mini",
            temperature=0.3,
        )
        architect = OpenRouterProvider(architect_config)

        # Editor: Ollama local model (free!)
        editor_config = AgentLLMConfig(
            provider="ollama",
            model="codellama:13b",
            temperature=0.7,
        )
        editor = OllamaProvider(editor_config)

        # Create mixed agent
        config = ArchitectEditorConfig(
            architect_provider=architect,
            editor_provider=editor,
        )
        agent = ArchitectEditorAgent(config)

        console.print("[green]✓ Mixed agent created:[/green]")
        console.print(f"  Architect: {architect}")
        console.print(f"  Editor: {editor}")
        console.print("\n[dim]Uses expensive model for planning, free local model for implementation![/dim]")

    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")


async def main():
    """Run all examples."""

    console.print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     ctxai - Multi-Provider Architecture Examples     ║
    ║                                                       ║
    ║  OpenRouter + Ollama + Architect/Editor Pattern      ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    # Run examples
    await example_openrouter_architect_editor()
    await example_ollama_local()
    await example_repository_mapping()
    await example_mixed_providers()

    console.print("\n[bold green]✓ All examples completed![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("1. Set OPENROUTER_API_KEY environment variable")
    console.print("2. Install and run Ollama: https://ollama.ai")
    console.print("3. Run: ctxai chat --provider openrouter")
    console.print("4. Try: ctxai chat --provider ollama --model codellama:13b")


if __name__ == "__main__":
    asyncio.run(main())
