"""Tests for ctxai.validation."""

from dataclasses import dataclass

import pytest

from ctxai.agent.config import AgentBehaviorConfig, AgentConfig, AgentLLMConfig, AgentToolsConfig
from ctxai.validation import ConfigValidator, ValidationIssue, assert_valid


def test_default_agent_config_is_valid():
    cfg = AgentConfig()
    issues = ConfigValidator().validate_agent_config(cfg)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []


def test_unknown_provider_warns():
    cfg = AgentConfig(llm=AgentLLMConfig(provider="hypothetical"))
    issues = ConfigValidator().validate_agent_config(cfg)
    assert any(i.severity == "warning" and "Unknown provider" in i.message for i in issues)


def test_invalid_temperature_errors():
    cfg = AgentConfig(llm=AgentLLMConfig(temperature=5.0))
    issues = ConfigValidator().validate_agent_config(cfg)
    assert any(i.field == "llm.temperature" for i in issues)


def test_invalid_max_iterations():
    cfg = AgentConfig(behavior=AgentBehaviorConfig(max_iterations=0))
    issues = ConfigValidator().validate_agent_config(cfg)
    assert any(i.field == "behavior.max_iterations" for i in issues)


def test_assert_valid_raises_on_errors():
    issues = [ValidationIssue("x", "bad")]
    with pytest.raises(ValueError):
        assert_valid(issues)


def test_assert_valid_passes_on_warnings():
    issues = [ValidationIssue("x", "watch out", severity="warning")]
    assert_valid(issues)
