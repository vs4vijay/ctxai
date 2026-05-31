"""Tests for ctxai.agent.planners."""

import pytest

from ctxai.agent.planners.base import heuristic_requires_planning
from ctxai.agent.planners.hierarchical_planner import HierarchicalPlanner
from ctxai.agent.planners.llm_planner import LLMPlanner
from ctxai.agent.planners.simple_planner import SimplePlanner
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response


def test_heuristic_detects_keywords():
    assert heuristic_requires_planning("Please refactor this module")
    assert heuristic_requires_planning("Implement a new feature")


def test_heuristic_uses_length_fallback():
    short = "fix bug"
    long = "I would like you to slowly walk me through fixing this complex bug with many steps"
    assert not heuristic_requires_planning(short)
    assert heuristic_requires_planning(long)


@pytest.mark.asyncio
async def test_simple_planner_creates_default_steps():
    planner = SimplePlanner()
    plan = await planner.create_plan("Refactor the auth module")
    assert plan.goal == "Refactor the auth module"
    assert len(plan.steps) >= 2


@pytest.mark.asyncio
async def test_simple_planner_extracts_numbered_list():
    planner = SimplePlanner()
    message = (
        "Please do the following:\n"
        "1. Read the config\n"
        "2. Validate it\n"
        "3. Apply migrations\n"
    )
    plan = await planner.create_plan(message)
    assert len(plan.steps) >= 2


@pytest.mark.asyncio
async def test_llm_planner_parses_json():
    llm = MockLLMProvider(
        responses=[
            create_mock_response(
                content='[{"description": "Step A", "agent_type": "coder"},'
                ' {"description": "Step B", "agent_type": "coder"}]'
            )
        ]
    )
    planner = LLMPlanner(llm)
    plan = await planner.create_plan("Implement a feature")
    assert len(plan.steps) == 2
    assert plan.steps[0].description == "Step A"


@pytest.mark.asyncio
async def test_llm_planner_handles_invalid_json():
    llm = MockLLMProvider(responses=[create_mock_response(content="not json")])
    planner = LLMPlanner(llm)
    plan = await planner.create_plan("Implement a feature")
    assert len(plan.steps) >= 1


@pytest.mark.asyncio
async def test_llm_planner_strips_code_fence():
    llm = MockLLMProvider(
        responses=[
            create_mock_response(
                content='```json\n[{"description": "A", "agent_type": "coder"}]\n```'
            )
        ]
    )
    plan = await LLMPlanner(llm).create_plan("Refactor the X")
    assert plan.steps[0].description == "A"


@pytest.mark.asyncio
async def test_hierarchical_planner_creates_phases():
    planner = HierarchicalPlanner()
    plan = await planner.create_plan("Implement feature X with edge cases")
    assert len(plan.steps) == len(HierarchicalPlanner.PHASES)
    assert "implementation_plan" in plan.metadata


def test_simple_planner_should_plan():
    sp = SimplePlanner()
    assert sp.should_plan("Refactor the module")
    assert not sp.should_plan("ping")
