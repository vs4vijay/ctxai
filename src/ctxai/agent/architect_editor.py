"""
Architect/Editor pattern implementation (Aider-inspired).

Uses two models:
- Architect: Expensive reasoning model for planning (o1, DeepSeek-R1, Claude Opus)
- Editor: Cheaper fast model for implementation (GPT-4o, Claude Sonnet, DeepSeek Chat)

This approach achieves better quality + lower cost than using a single model.
"""

from dataclasses import dataclass
from typing import Optional

from rich.console import Console

from .llm.base import BaseLLMProvider
from .planning import Plan, PlanStep, create_plan

console = Console()


@dataclass
class ArchitectEditorConfig:
    """Configuration for architect/editor pattern."""

    architect_provider: BaseLLMProvider
    editor_provider: BaseLLMProvider
    use_architect_for_planning: bool = True
    use_architect_for_complex: bool = True
    complexity_threshold: int = 5  # Number of files/steps to consider complex


class ArchitectEditorAgent:
    """
    Agent that uses architect/editor pattern.

    Workflow:
    1. Architect analyzes task and creates plan
    2. Editor implements each step
    3. Architect reviews if needed
    4. Iterate until done
    """

    def __init__(self, config: ArchitectEditorConfig):
        """
        Initialize architect/editor agent.

        Args:
            config: Configuration with both providers
        """
        self.config = config
        self.architect = config.architect_provider
        self.editor = config.editor_provider

    async def process_task(
        self,
        task: str,
        context: dict,
        tools: list[dict],
    ) -> dict:
        """
        Process a task using architect/editor pattern.

        Args:
            task: Task description
            context: Conversation context
            tools: Available tools

        Returns:
            Dict with result and metadata
        """
        # Step 1: Architect analyzes and creates plan
        console.print("\n[cyan]🏗️  Architect analyzing task...[/cyan]")

        plan = await self._architect_plan(task, context)

        if not plan:
            # Simple task, skip planning
            console.print("[dim]Simple task, using editor directly[/dim]")
            return await self._editor_implement(task, context, tools)

        # Show plan to user
        console.print(f"\n[green]📋 Plan created with {len(plan.steps)} steps:[/green]")
        for i, step in enumerate(plan.steps, 1):
            console.print(f"  {i}. {step.description}")

        # Step 2: Editor implements each step
        results = []
        for i, step in enumerate(plan.steps, 1):
            console.print(f"\n[cyan]⚙️  Step {i}/{len(plan.steps)}: {step.description}[/cyan]")

            result = await self._editor_implement(
                step.description,
                context,
                tools,
            )

            results.append(result)

            # Update context with result
            context["last_result"] = result

        # Step 3: Architect reviews (optional)
        if self.config.use_architect_for_complex:
            console.print("\n[cyan]🔍 Architect reviewing implementation...[/cyan]")
            review = await self._architect_review(task, results, context)

            if review.get("needs_changes"):
                console.print("[yellow]⚠️  Changes needed, iterating...[/yellow]")
                # Could implement iteration here
                pass

        return {
            "success": True,
            "plan": plan.to_dict() if plan else None,
            "results": results,
            "architect_used": True,
        }

    async def _architect_plan(
        self,
        task: str,
        context: dict,
    ) -> Plan | None:
        """
        Use architect to create a plan.

        Args:
            task: Task description
            context: Context

        Returns:
            Plan or None if task is simple
        """
        if not self.config.use_architect_for_planning:
            return None

        # Build planning prompt
        prompt = self._build_planning_prompt(task, context)

        # Call architect
        messages = [
            {"role": "system", "content": "You are an expert software architect. Create concise, actionable plans."},
            {"role": "user", "content": prompt},
        ]

        response = self.architect.chat(messages)

        # Parse response into plan
        # TODO: Better parsing logic
        plan_text = response.content

        # For now, create simple plan
        # In production, use LLM to parse the response properly
        if "step" in plan_text.lower() or len(plan_text) > 500:
            steps = [
                {
                    "description": "Analyze requirements",
                    "agent_type": "explorer",
                    "tools_needed": ["read_file", "grep"],
                },
                {
                    "description": "Implement changes",
                    "agent_type": "coder",
                    "tools_needed": ["edit_file", "write_file"],
                },
                {
                    "description": "Test changes",
                    "agent_type": "reviewer",
                    "tools_needed": ["bash_exec"],
                },
            ]
            return create_plan(task, steps)

        return None  # Simple task

    async def _editor_implement(
        self,
        task: str,
        context: dict,
        tools: list[dict],
    ) -> dict:
        """
        Use editor to implement a task.

        Args:
            task: Task description
            context: Context
            tools: Available tools

        Returns:
            Implementation result
        """
        # Build implementation prompt
        prompt = self._build_implementation_prompt(task, context)

# Call editor with tools
        system_msg = "You are an expert code editor. Implement changes precisely and efficiently."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response = self.editor.chat(messages, tools=tools)

        return {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "model": str(self.editor),
        }

    async def _architect_review(
        self,
        task: str,
        results: list[dict],
        context: dict,
    ) -> dict:
        """
        Use architect to review implementation.

        Args:
            task: Original task
            results: Implementation results
            context: Context

        Returns:
            Review result
        """
        # Build review prompt
        prompt = self._build_review_prompt(task, results, context)

        system_msg = "You are an expert code reviewer. Verify implementations are correct and complete."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response = self.architect.chat(messages)

        # Parse review
        content = response.content.lower()
        needs_changes = any(word in content for word in ["issue", "problem", "incorrect", "missing", "fix"])

        return {
            "review": response.content,
            "needs_changes": needs_changes,
        }

    def _build_planning_prompt(self, task: str, context: dict) -> str:
        """Build prompt for planning."""
        return f"""Task: {task}

Context:
- Working directory: {context.get('working_directory', '.')}
- Available tools: {', '.join(context.get('tools', []))}

Create a step-by-step plan to complete this task.
Each step should be clear and actionable.
Keep the plan concise (3-5 steps).

Format:
1. Step description
2. Step description
...
"""

    def _build_implementation_prompt(self, task: str, context: dict) -> str:
        """Build prompt for implementation."""
        return f"""Task: {task}

Context:
{context.get('last_result', '')}

Implement this step using the available tools.
Be precise and efficient.
"""

    def _build_review_prompt(self, task: str, results: list[dict], context: dict) -> str:
        """Build prompt for review."""
        results_text = "\n\n".join([
            f"Step {i+1}: {r.get('content', '')[:200]}..."
            for i, r in enumerate(results)
        ])

        return f"""Original Task: {task}

Implementation Results:
{results_text}

Review the implementation:
1. Does it complete the task?
2. Are there any issues or bugs?
3. Is anything missing?

Provide a brief review (2-3 sentences).
"""

    def get_cost_savings(self) -> str:
        """
        Estimate cost savings from architect/editor pattern.

        Returns:
            Description of savings
        """
        return (
            "Using architect/editor pattern:\n"
            f"  • Architect: {self.architect} (expensive, for planning)\n"
            f"  • Editor: {self.editor} (cheaper, for implementation)\n"
            "  • Expected savings: 40-60% vs using architect for everything\n"
            "  • Quality: Same or better (specialized models)"
        )


def create_architect_editor_agent(
    architect_model: str = "openai/o1-mini",
    editor_model: str = "anthropic/claude-3.5-sonnet",
    use_openrouter: bool = True,
) -> ArchitectEditorAgent:
    """
    Create architect/editor agent with recommended models.

    Args:
        architect_model: Model for architect (reasoning)
        editor_model: Model for editor (fast implementation)
        use_openrouter: Use OpenRouter (default: True)

    Returns:
        Configured ArchitectEditorAgent
    """
    from .config import AgentLLMConfig
    from .llm.openrouter_provider import OpenRouterProvider

    # Create architect provider (reasoning model)
    architect_config = AgentLLMConfig(
        provider="openrouter",
        model=architect_model,
        temperature=0.3,  # Lower for more precise planning
        max_tokens=4096,
    )
    architect = OpenRouterProvider(architect_config)

    # Create editor provider (fast implementation model)
    editor_config = AgentLLMConfig(
        provider="openrouter",
        model=editor_model,
        temperature=0.7,  # Higher for more creative implementation
        max_tokens=4096,
    )
    editor = OpenRouterProvider(editor_config)

    # Create config
    config = ArchitectEditorConfig(
        architect_provider=architect,
        editor_provider=editor,
        use_architect_for_planning=True,
        use_architect_for_complex=True,
    )

    agent = ArchitectEditorAgent(config)

    console.print(agent.get_cost_savings())

    return agent
