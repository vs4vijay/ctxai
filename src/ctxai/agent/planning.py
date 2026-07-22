"""
Planning system for the agent.

Implements planning patterns for complex tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class PlanStatus(str, Enum):
    """Plan status."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Step status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """
    Single step in an execution plan.

    Represents an atomic action that needs to be performed
    to achieve the overall goal.
    """

    step_id: str
    description: str
    agent_type: str  # explorer, coder, reviewer, orchestrator
    tools_needed: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # step_ids
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def start(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self, result: str) -> None:
        """Mark step as completed."""
        self.status = StepStatus.DONE
        self.result = result
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def skip(self, reason: str) -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.completed_at = datetime.now()

    def is_ready(self, completed_steps: set) -> bool:
        """
        Check if step is ready to execute.

        A step is ready if all its dependencies are completed.

        Args:
            completed_steps: Set of completed step IDs

        Returns:
            True if ready to execute
        """
        return all(dep in completed_steps for dep in self.dependencies)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "agent_type": self.agent_type,
            "tools_needed": self.tools_needed,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Plan:
    """
    Execution plan for a complex task.

    Contains a sequence of steps that need to be executed
    to achieve a goal.
    """

    plan_id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def start(self) -> None:
        """Mark plan as started."""
        self.status = PlanStatus.ACTIVE
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark plan as completed."""
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self) -> None:
        """Mark plan as failed."""
        self.status = PlanStatus.FAILED
        self.completed_at = datetime.now()

    def cancel(self) -> None:
        """Cancel the plan."""
        self.status = PlanStatus.CANCELLED
        self.completed_at = datetime.now()

    def get_next_steps(self) -> list[PlanStep]:
        """
        Get next steps that are ready to execute.

        Returns steps that:
        1. Are in pending status
        2. Have all dependencies completed

        Returns:
            List of ready steps
        """
        completed_steps = {step.step_id for step in self.steps if step.status == StepStatus.DONE}

        return [step for step in self.steps if step.status == StepStatus.PENDING and step.is_ready(completed_steps)]

    def get_step(self, step_id: str) -> PlanStep | None:
        """
        Get step by ID.

        Args:
            step_id: Step ID

        Returns:
            PlanStep or None
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.steps.append(step)

    def get_progress(self) -> dict:
        """
        Get plan progress.

        Returns:
            Dict with progress metrics
        """
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        in_progress = sum(1 for s in self.steps if s.status == StepStatus.IN_PROGRESS)
        pending = sum(1 for s in self.steps if s.status == StepStatus.PENDING)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "percentage": (completed / total * 100) if total > 0 else 0,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.get_progress(),
            "metadata": self.metadata,
        }


class PlanExecutor:
    """
    Executes a plan by running steps in dependency order.

    Handles step execution, error recovery, and progress tracking.
    """

    def __init__(self, plan: Plan):
        """
        Initialize executor.

        Args:
            plan: Plan to execute
        """
        self.plan = plan

    async def execute(self, step_executor: callable) -> dict:
        """
        Execute the plan.

        Args:
            step_executor: Async function to execute a step
                          Signature: async def(step: PlanStep) -> Dict

        Returns:
            Dict with execution results
        """
        self.plan.start()

        try:
            while True:
                # Get next steps to execute
                next_steps = self.plan.get_next_steps()

                if not next_steps:
                    # Check if all steps are done
                    pending = [s for s in self.plan.steps if s.status == StepStatus.PENDING]
                    if not pending:
                        # No more pending steps - we're done
                        break
                    else:
                        # Still have pending steps but none are ready
                        # This means dependencies are not met (possible failure)
                        self.plan.fail()
                        return {
                            "success": False,
                            "error": "Plan execution stalled - dependencies not met",
                            "progress": self.plan.get_progress(),
                        }

                # Execute next steps (could be parallel in future)
                for step in next_steps:
                    step.start()

                    try:
                        # Execute step
                        result = await step_executor(step)

                        if result.get("success"):
                            step.complete(result.get("result", ""))
                        else:
                            step.fail(result.get("error", "Unknown error"))
                            # TODO: Implement retry/recovery logic

                    except Exception as e:
                        step.fail(str(e))

            # Check final status
            failed_steps = [s for s in self.plan.steps if s.status == StepStatus.FAILED]

            if failed_steps:
                self.plan.fail()
                return {
                    "success": False,
                    "error": f"{len(failed_steps)} step(s) failed",
                    "failed_steps": [s.to_dict() for s in failed_steps],
                    "progress": self.plan.get_progress(),
                }

            self.plan.complete()
            return {
                "success": True,
                "result": "Plan completed successfully",
                "progress": self.plan.get_progress(),
            }

        except Exception as e:
            self.plan.fail()
            return {
                "success": False,
                "error": f"Plan execution error: {str(e)}",
                "progress": self.plan.get_progress(),
            }


def create_plan(goal: str, steps: list[dict]) -> Plan:
    """
    Create a plan from goal and step descriptions.

    Args:
        goal: Overall goal description
        steps: List of step dictionaries with:
               - description: str
               - agent_type: str
               - tools_needed: List[str] (optional)
               - dependencies: List[str] (optional, step IDs)

    Returns:
        Plan instance
    """
    plan = Plan(
        plan_id=str(uuid4()),
        goal=goal,
    )

    for step_data in steps:
        step = PlanStep(
            step_id=str(uuid4()),
            description=step_data["description"],
            agent_type=step_data["agent_type"],
            tools_needed=step_data.get("tools_needed", []),
            dependencies=step_data.get("dependencies", []),
        )
        plan.add_step(step)

    return plan
