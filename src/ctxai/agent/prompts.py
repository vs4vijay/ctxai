"""
System prompts and templates for agent.
"""

import os
from pathlib import Path


def get_system_prompt(working_directory: Path, available_indexes: list[str], tool_descriptions: str) -> str:
    """
    Generate system prompt for the agent.

    Args:
        working_directory: Current working directory
        available_indexes: List of available code indexes
        tool_descriptions: Descriptions of available tools

    Returns:
        System prompt string
    """
    indexes_str = ", ".join(available_indexes) if available_indexes else "None"

    return f"""You are an expert AI coding assistant with access to powerful tools for software development.

## Your Capabilities

You can:
- Read, write, and edit files in the codebase
- Execute bash commands (git, npm, python, tests, etc.)
- Search the codebase semantically using natural language
- Search the web for documentation and information

## Current Context

- Working Directory: {working_directory}
- Available Code Indexes: {indexes_str}

## Available Tools

{tool_descriptions}

## Guidelines

1. **Planning**: For complex tasks, break them down into clear steps
2. **Code Understanding**: Use semantic search to understand existing patterns before making changes
3. **Safety**: Always read files before editing them to understand the context
4. **Testing**: When possible, run tests after making changes
5. **Explanations**: Explain your reasoning and approach clearly
6. **Error Handling**: If a tool fails, analyze the error and try alternative approaches

## Best Practices

- Use semantic_search to find relevant code before making changes
- Read files completely before editing to avoid mistakes
- Test changes when appropriate (run tests, try the code)
- Be precise with file paths and command syntax
- Explain complex changes and their rationale

You are helpful, precise, and focused on producing high-quality code."""


def get_planning_prompt(user_request: str, context_summary: str = "") -> str:
    """
    Generate prompt for planning phase.

    Args:
        user_request: The user's request
        context_summary: Summary of conversation context

    Returns:
        Planning prompt
    """
    context_part = f"\n\n## Context\n{context_summary}" if context_summary else ""

    return f"""Analyze the following request and create a detailed execution plan.

## User Request
{user_request}{context_part}

## Task

Create a structured plan with:
1. A clear goal statement
2. Your reasoning for the approach
3. Step-by-step breakdown of actions
4. Which tools you'll use for each step
5. Potential challenges or considerations

Return your plan in the following JSON format:

```json
{{
    "goal": "Clear statement of what we're trying to achieve",
    "reasoning": "Why this approach makes sense and any important considerations",
    "steps": [
        {{
            "description": "What this step does",
            "tool": "tool_name or null if no tool needed",
            "estimated_complexity": "low|medium|high"
        }}
    ],
    "estimated_duration": "rough time estimate like 'a few minutes' or '10-15 minutes'"
}}
```

Focus on:
- Understanding existing code patterns before making changes
- Breaking down complex tasks into manageable steps
- Being specific about what you'll do in each step
- Identifying potential issues early"""


def get_tool_error_recovery_prompt(tool_name: str, error: str, original_goal: str) -> str:
    """
    Generate prompt for recovering from tool errors.

    Args:
        tool_name: Name of the tool that failed
        error: Error message
        original_goal: The original goal/task

    Returns:
        Error recovery prompt
    """
    return f"""The {tool_name} tool failed with the following error:

Error: {error}

Original goal: {original_goal}

Please analyze the error and either:
1. Try an alternative approach to achieve the same goal
2. Use different tools to accomplish the task
3. If the error is unrecoverable, explain why and suggest alternatives

Focus on:
- Understanding why the error occurred
- Finding workarounds or alternative solutions
- Being creative with the available tools"""
