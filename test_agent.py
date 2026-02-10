#!/usr/bin/env python3
"""Direct test of the agent with OpenRouter."""

import sys
import asyncio
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.factory import LLMProviderFactory
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.tools.file_ops import ReadFileTool, ListFilesTool


async def test_agent():
    """Test the agent with a simple query."""
    print("=" * 60)
    print("Testing ctxai Agent with OpenRouter")
    print("=" * 60)

    # Create LLM provider
    llm_config = AgentLLMConfig(
        provider="openrouter",
        model=None,  # Use default
        temperature=0.7,
        max_tokens=4096,
    )

    print(f"\nCreating LLM provider...")
    llm = LLMProviderFactory.create_provider(llm_config)
    print(f"  Provider: {llm}")
    print(f"  Model: {llm.model}")

    # Create tools
    print(f"\nRegistering tools...")
    tools = ToolRegistry(verbose=False)
    tools.register(ReadFileTool())
    tools.register(ListFilesTool())
    print(f"  Registered {len(tools.list_tools())} tools")

    # Create agent
    print(f"\nCreating agent...")
    agent_config = AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=llm,
        tool_registry=tools,
        agent_config=agent_config,
        working_directory=Path.cwd(),
        available_indexes=[],
        max_iterations=10,
        verbose=True,
    )
    agent = Agent(loop_config)
    print(f"  Agent: {agent}")

    # Test with a simple message
    print("\n" + "=" * 60)
    print("Test 1: Simple greeting")
    print("=" * 60)

    try:
        response = await agent.process_message("Hello! What's up?")
        print(f"\n[OK] Agent Response:")
        print(f"{response}")
        print(f"\n[OK] Test 1 passed - no max iterations error!")
    except Exception as e:
        print(f"\n[ERROR] Test 1 failed: {e}")
        return False

    # Test with a file operation
    print("\n" + "=" * 60)
    print("Test 2: File operation (should use tools)")
    print("=" * 60)

    try:
        response = await agent.process_message("List the files in the current directory")
        print(f"\n[OK] Agent Response:")
        print(f"{response}")
        print(f"\n[OK] Test 2 passed!")
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("[OK] ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    result = asyncio.run(test_agent())
    sys.exit(0 if result else 1)
