"""
End-to-end tests for agent workflows with tool execution.

Tests the agent loop with real tool execution but mocked LLM responses.
"""

import pytest
import asyncio
from pathlib import Path
from ctxai.agent.core import Agent, AgentLoopConfig
from ctxai.agent.config import AgentConfig
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.tools.file_ops import (
    ReadFileTool,
    WriteFileTool,
    ListFilesTool,
)
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_read_file_tool(sample_python_code, temp_dir, mock_llm_config):
    """
    Test agent using read_file tool.

    Verifies:
    1. Agent receives tool call from LLM
    2. Tool executes correctly
    3. Tool result is added to context
    4. Agent returns final response
    """
    # Create file to read
    test_file = sample_python_code / "test.py"
    test_file.write_text("def hello(): return 'world'")

    # Configure mock LLM responses
    responses = [
        # First response: use read_file tool
        create_mock_response(
            content="I'll read the file for you.",
            tool_calls=[{
                "name": "read_file",
                "parameters": {"path": str(test_file)}
            }]
        ),
        # Second response: final answer after seeing tool result
        create_mock_response(
            content="The file contains a function called hello that returns 'world'."
        )
    ]

    mock_llm = MockLLMProvider(config=mock_llm_config, responses=responses)

    # Create tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(ReadFileTool(working_directory=temp_dir))

    # Create agent
    agent_config = AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=5,
        verbose=False
    )

    agent = Agent(loop_config)

    # Process message
    response = await agent.process_message("What's in test.py?")

    # Verify response
    assert "hello" in response.lower() or "world" in response.lower()
    assert mock_llm.call_count == 2, "Should have made 2 LLM calls"

    # Verify tool was called
    assert len(mock_llm.call_history) == 2
    # Second call should have tool result in messages
    second_call_messages = mock_llm.call_history[1]["messages"]
    assert any("tool" in str(msg).lower() for msg in second_call_messages)


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_multi_tool_workflow(sample_python_code, temp_dir, mock_llm_config):
    """
    Test agent using multiple tools in sequence.

    Verifies:
    1. Agent can use list_files tool
    2. Agent can use read_file tool based on list results
    3. Multiple tool calls are handled correctly
    4. Final response incorporates all tool results
    """
    # Configure mock LLM responses for multi-tool workflow
    responses = [
        # First: list files
        create_mock_response(
            content="Let me list the Python files first.",
            tool_calls=[{
                "name": "list_files",
                "parameters": {"directory": str(sample_python_code)}
            }]
        ),
        # Second: read a specific file
        create_mock_response(
            content="Now let me read main.py",
            tool_calls=[{
                "name": "read_file",
                "parameters": {"path": str(sample_python_code / "main.py")}
            }]
        ),
        # Third: final response
        create_mock_response(
            content="I found main.py which contains greeting functions and a Calculator class."
        )
    ]

    mock_llm = MockLLMProvider(config=mock_llm_config, responses=responses)

    # Create tool registry with multiple tools
    tool_registry = ToolRegistry()
    tool_registry.register(ListFilesTool(working_directory=temp_dir))
    tool_registry.register(ReadFileTool(working_directory=temp_dir))

    # Create agent
    agent_config = AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=10,
        verbose=False
    )

    agent = Agent(loop_config)

    # Process message
    response = await agent.process_message("What Python files are there and what's in main.py?")

    # Verify response
    assert "calculator" in response.lower() or "greeting" in response.lower()
    assert mock_llm.call_count == 3, "Should have made 3 LLM calls"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_error_handling(temp_dir, mock_llm_config):
    """
    Test agent error handling when tool execution fails.

    Verifies:
    1. Agent attempts to read non-existent file
    2. Tool returns error
    3. Agent receives error in context
    4. Agent can recover and continue
    """
    # Configure mock to try reading non-existent file
    responses = [
        create_mock_response(
            content="Let me read that file.",
            tool_calls=[{
                "name": "read_file",
                "parameters": {"path": str(temp_dir / "nonexistent.py")}
            }]
        ),
        create_mock_response(
            content="The file doesn't exist. I'll help you create it instead."
        )
    ]

    mock_llm = MockLLMProvider(config=mock_llm_config, responses=responses)

    # Create tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(ReadFileTool(working_directory=temp_dir))

    # Create agent
    agent_config = AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=5,
        verbose=False
    )

    agent = Agent(loop_config)

    # Process message
    response = await agent.process_message("Read nonexistent.py")

    # Verify agent handled the error
    assert "exist" in response.lower() or "create" in response.lower()
    assert mock_llm.call_count >= 1


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_max_iterations(temp_dir, mock_llm_config):
    """
    Test agent stops after max iterations.

    Verifies:
    1. Agent respects max_iterations limit
    2. Appropriate warning message is returned
    3. Prevents infinite loops
    """
    # Configure mock to always return tool calls (infinite loop scenario)
    responses = [
        create_mock_response(
            content="Let me check that file.",
            tool_calls=[{
                "name": "read_file",
                "parameters": {"path": str(temp_dir / "test.py")}
            }]
        )
    ] * 20  # More responses than max iterations

    mock_llm = MockLLMProvider(config=mock_llm_config, responses=responses)

    # Create tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(ReadFileTool(working_directory=temp_dir))

    # Create agent with low max_iterations
    agent_config = AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=3,  # Low limit for testing
        verbose=False
    )

    agent = Agent(loop_config)

    # Create a dummy file so tool doesn't fail
    (temp_dir / "test.py").write_text("dummy content")

    # Process message
    response = await agent.process_message("Read test.py multiple times")

    # Verify max iterations warning
    assert "max iterations" in response.lower() or "reached" in response.lower()
    assert mock_llm.call_count == 3, f"Should stop at max_iterations (3), but called {mock_llm.call_count} times"


@pytest.mark.e2e
@pytest.mark.agent
@pytest.mark.asyncio
async def test_agent_write_and_read_workflow(temp_dir, mock_llm_config):
    """
    Test agent workflow that writes then reads a file.

    Verifies:
    1. Agent can write file with write_file tool
    2. Agent can read file it just created
    3. Tools integrate correctly with file system
    4. Agent maintains context across tool calls
    """
    responses = [
        # Write file
        create_mock_response(
            content="I'll create the file for you.",
            tool_calls=[{
                "name": "write_file",
                "parameters": {
                    "path": str(temp_dir / "new_file.py"),
                    "content": "def greet(name):\n    return f'Hello, {name}!'\n"
                }
            }]
        ),
        # Read it back
        create_mock_response(
            content="Now let me read it back to verify.",
            tool_calls=[{
                "name": "read_file",
                "parameters": {"path": str(temp_dir / "new_file.py")}
            }]
        ),
        # Final response
        create_mock_response(
            content="I've created the file successfully with a greet function."
        )
    ]

    mock_llm = MockLLMProvider(config=mock_llm_config, responses=responses)

    # Create tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(WriteFileTool(working_directory=temp_dir))
    tool_registry.register(ReadFileTool(working_directory=temp_dir))

    # Create agent
    agent_config = AgentConfig()
    loop_config = AgentLoopConfig(
        llm_provider=mock_llm,
        tool_registry=tool_registry,
        agent_config=agent_config,
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=10,
        verbose=False
    )

    agent = Agent(loop_config)

    # Process message
    response = await agent.process_message("Create a file called new_file.py with a greet function")

    # Verify file was created
    new_file = temp_dir / "new_file.py"
    assert new_file.exists(), "File should be created"
    assert "greet" in new_file.read_text(), "File should contain greet function"

    # Verify response
    assert "created" in response.lower() or "success" in response.lower()
