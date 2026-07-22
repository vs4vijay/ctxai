"""
Quick start example for ctxai agent.

This demonstrates how to use the agent programmatically.
"""

import asyncio
import os
from pathlib import Path

from ctxai.agent.config import AgentConfig, AgentLLMConfig
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.llm.anthropic_provider import AnthropicProvider
from ctxai.agent.tools.bash_tool import BashTool
from ctxai.agent.tools.code_search import SemanticSearchTool
from ctxai.agent.tools.file_ops import EditFileTool, GlobTool, GrepTool, ListFilesTool, ReadFileTool, WriteFileTool
from ctxai.agent.tools.registry import ToolRegistry


async def main():
    """Run agent example."""

    # 1. Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY=your-key-here")
        return

    print("🤖 Initializing ctxai agent...")

    # 2. Configure LLM
    llm_config = AgentLLMConfig(
        provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=4096
    )

    # 3. Initialize provider
    llm = AnthropicProvider(llm_config)
    print(f"✓ LLM: {llm}")

    # 4. Create agent config
    agent_config = AgentConfig()

    # 5. Register tools
    tools = ToolRegistry(verbose=True)
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    tools.register(EditFileTool())
    tools.register(ListFilesTool())
    tools.register(GlobTool())
    tools.register(GrepTool())
    tools.register(BashTool(agent_config.tools))
    tools.register(SemanticSearchTool())
    print(f"✓ Registered {len(tools)} tools")

    # 6. Create agent
    loop_config = AgentLoopConfig(
        llm_provider=llm,
        tool_registry=tools,
        agent_config=agent_config,
        working_directory=Path("."),
        available_indexes=[],
        max_iterations=10,
        verbose=True,
    )
    agent = Agent(loop_config)
    print(f"✓ Agent initialized: {agent}\n")

    # 7. Example interactions
    examples = [
        "List all Python files in the src directory",
        "Read the README.md file and give me a brief summary",
        "What is the current git status?",
    ]

    print("=" * 60)
    print("AGENT EXAMPLES")
    print("=" * 60)

    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example}")
        print("-" * 60)

        try:
            response = await agent.process_message(example)
            print(f"\n🤖 Agent Response:\n{response}\n")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

        print("=" * 60)

    # 8. Interactive mode (optional)
    print("\n💬 Entering interactive mode. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            response = await agent.process_message(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║         ctxai Agent - Quick Start Example           ║
    ║                                                       ║
    ║  This demonstrates the autonomous coding agent       ║
    ║  with file operations, bash, and code search.        ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
