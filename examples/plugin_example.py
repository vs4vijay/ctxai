"""
plugin_example.py — Write and register a plugin that intercepts agent events.
"""

import asyncio
from pathlib import Path

from ctxai_core import create_agent
from ctxai_core.plugins import PluginInterface, PluginMetadata, get_plugin_manager


class LoggingPlugin(PluginInterface):
    metadata = PluginMetadata(
        name="logging",
        version="1.0.0",
        description="Logs every message and tool call",
        author="ctxai team",
    )

    def on_message_start(self, message, context):
        print(f"[plugin] user said: {message}")
        return message

    def on_message_end(self, response, context):
        print(f"[plugin] agent responded with {len(response)} chars")
        return response

    def on_tool_call(self, tool_name, args, context):
        print(f"[plugin] tool {tool_name}({args})")
        return args


async def main() -> None:
    plugin_manager = get_plugin_manager()
    plugin_manager.register(LoggingPlugin())

    agent = create_agent(
        provider="openrouter",
        tools=["file_ops"],
        working_directory=Path.cwd(),
    )

    user_message = "Read the README file."
    # In a real integration, the agent would wire plugin hooks itself.
    user_message = plugin_manager.pipeline("message_start", user_message)
    response = await agent.process_message(user_message)
    response = plugin_manager.pipeline("message_end", response)
    print("FINAL:", response[:200])


if __name__ == "__main__":
    asyncio.run(main())
