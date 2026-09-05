"""
basic_agent.py — Minimal end-to-end agent using ctxai_core.

Requires an LLM provider configured via env var (e.g. OPENROUTER_API_KEY).
"""

import asyncio
from pathlib import Path

from ctxai_core import create_agent


async def main() -> None:
    agent = create_agent(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        tools=["file_ops", "git"],
        working_directory=Path.cwd(),
        max_iterations=8,
        verbose=True,
    )

    response = await agent.process_message("List the Python files in this directory.")
    print("\n=== Agent response ===\n")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
