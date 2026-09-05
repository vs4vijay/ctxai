"""
custom_tool.py — Build and register a custom tool with @create_tool.
"""

import asyncio
from pathlib import Path

from ctxai_core import create_agent, create_tool
from ctxai_core.api import ToolParameter, ToolParameterType


@create_tool(
    name="word_count",
    description="Count words in a string.",
    parameters=[
        ToolParameter(
            name="text",
            type=ToolParameterType.STRING,
            description="Input text",
            required=True,
        ),
    ],
)
async def word_count(text: str) -> dict:
    return {"success": True, "result": len(text.split())}


async def main() -> None:
    agent = create_agent(
        provider="openrouter",
        tools=[word_count],
        working_directory=Path.cwd(),
    )
    response = await agent.process_message("Count the words in: 'the quick brown fox jumps'")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
