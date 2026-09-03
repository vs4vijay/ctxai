"""Unit tests for context window management (HH-03).

Proves: pairing-preserving deterministic compaction for both provider message
formatters, honest elision markers, the measured-token estimator with
chars-div-4 fallback, UsageLedger aggregation, the soft-limit trigger math,
the ``length`` finish-reason mapping to INVALID_RESPONSE handling, and the new
``AgentBehaviorConfig.context_soft_limit_ratio`` round-trip.
"""

from __future__ import annotations

from typing import Any

import pytest

from ctxai.agent.config import AgentBehaviorConfig, AgentConfig, AgentLLMConfig
from ctxai.agent.context import ConversationContext
from ctxai.agent.core import Agent, AgentLoopConfig, format_compaction_notice
from ctxai.agent.llm.anthropic_provider import AnthropicProvider
from ctxai.agent.llm.base import Message, MessageRole, ProviderCapabilities, ToolCall
from ctxai.agent.resilience import RetryPolicy
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import ReadFileTool
from ctxai.agent.tools.registry import ToolRegistry
from ctxai.agent.workflow import UsageLedger
from tests.mocks.mock_llm import MockLLMProvider, create_mock_response

FAST_POLICY = RetryPolicy(max_retries=1, base_delay_s=0.001, max_delay_s=0.002)
RECOVERY_PROMPT_MARKER = "failed with the following error"


# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture
def mock_llm_config() -> AgentLLMConfig:
    """Provide an LLM configuration for the scripted mock provider.

    Returns:
        AgentLLMConfig for the mock provider.
    """
    return AgentLLMConfig(provider="mock", model="mock-model", api_key="mock-key")


def seed_tool_groups(context: ConversationContext, count: int, body_chars: int = 200) -> None:
    """Append ``count`` atomic tool groups (assistant tool call + result).

    Args:
        context: Context to append to.
        count: Number of tool groups to add.
        body_chars: Length of each synthetic tool-result body.
    """
    for index in range(count):
        context.add_assistant_message(
            f"step {index}",
            [ToolCall(id=f"call-{index}", name="read_file", parameters={"path": f"f{index}.txt"})],
        )
        context.add_tool_result(f"call-{index}", "read_file", "x" * body_chars)


def assert_no_orphan_tool_results(messages: list[Message]) -> None:
    """Assert every tool result is preceded by an assistant carrying its id.

    Validates pairing on the Message level and through both the OpenAI wire
    format and the Anthropic message formatter, so compaction cannot produce a
    payload any provider would reject.

    Args:
        messages: Compacted conversation messages.
    """
    seen_ids: set[str] = set()
    for message in messages:
        if message.role == MessageRole.ASSISTANT and message.tool_calls:
            seen_ids.update(call.id for call in message.tool_calls)
        if message.tool_call_id:
            assert message.tool_call_id in seen_ids, f"orphan tool result {message.tool_call_id}"

    openai_payload = [message.to_dict(format="openai") for message in messages]
    openai_ids: set[str] = set()
    for entry in openai_payload:
        for call in entry.get("tool_calls") or []:
            openai_ids.add(call["id"])
        if entry.get("role") == "tool":
            assert entry["tool_call_id"] in openai_ids, "openai formatter sees an orphan tool result"

    # The Anthropic formatter never touches provider state; call it unbound.
    anthropic_payload = AnthropicProvider._format_messages_for_anthropic(None, messages)
    anthropic_ids: set[str] = set()
    for entry in anthropic_payload:
        for block in entry.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                anthropic_ids.add(block["id"])
            if isinstance(block, dict) and block.get("type") == "tool_result":
                assert block["tool_use_id"] in anthropic_ids, "anthropic formatter sees an orphan tool result"


def make_agent(
    temp_dir,
    config: AgentLLMConfig,
    provider: MockLLMProvider,
    *,
    behavior_config: AgentConfig | None = None,
    on_compaction=None,
    max_iterations: int = 12,
) -> Agent:
    """Build a real agent (loop + registry + read tool) over the given provider.

    Args:
        temp_dir: Project root for the run.
        config: LLM configuration used by the provider.
        provider: Scripted LLM provider instance.
        behavior_config: Optional full agent config (controls behavior fields).
        on_compaction: Optional compaction-notification hook.
        max_iterations: Loop iteration budget.

    Returns:
        The configured Agent.
    """
    context = ToolExecutionContext.for_project(temp_dir)
    registry = ToolRegistry()
    registry.register(ReadFileTool(context=context, max_output_chars=20_000))
    loop_config = AgentLoopConfig(
        llm_provider=provider,
        tool_registry=registry,
        agent_config=behavior_config or AgentConfig(),
        working_directory=temp_dir,
        available_indexes=[],
        max_iterations=max_iterations,
        retry_policy=FAST_POLICY,
        on_compaction=on_compaction,
    )
    return Agent(loop_config)


def recovery_prompts(agent: Agent) -> list[str]:
    """Return injected recovery prompts from the conversation context.

    Args:
        agent: The agent whose context is inspected.

    Returns:
        Recovery prompt message contents.
    """
    return [
        message.content
        for message in agent.context.messages
        if message.role.value == "user" and RECOVERY_PROMPT_MARKER in message.content
    ]


# ============================================================================
# Compaction (ConversationContext.compact)
# ============================================================================


def test_compact_elides_old_tool_results_and_keeps_recent_window():
    """Bodies outside the keep_recent window become markers; window bodies stay."""
    context = ConversationContext()
    context.add_system_message("system prompt")
    context.add_user_message("Please inspect the modules")
    seed_tool_groups(context, count=8, body_chars=400)
    context.add_assistant_message("All modules inspected.")

    changed = context.compact(target_tokens=100, keep_recent=6, max_output_chars=20_000)

    assert changed is True
    bodies = [message.content for message in context.messages if message.tool_call_id]
    # Tool groups 1-3 aged out of the six-group window (user + 8 tool groups +
    # the final assistant message make 10 non-system groups); groups 4-8 stay verbatim.
    assert bodies[0] == "[elided 400 chars of tool result for read_file]"
    assert bodies[1] == "[elided 400 chars of tool result for read_file]"
    assert bodies[2] == "[elided 400 chars of tool result for read_file]"
    assert bodies[3:] == ["x" * 400] * 5
    assert context.compaction_count == 1
    assert context.elided_message_count == 3
    assert context.last_compaction is not None
    assert context.last_compaction["elided_messages"] == 3
    assert context.last_compaction["target_tokens"] == 100
    # The elided turns are summarized right after the system prompt.
    summary = context.messages[1]
    assert summary.role == MessageRole.USER
    assert summary.content.startswith("Conversation summary")
    assert "step 0" in summary.content


def test_compact_preserves_pairing_for_openai_and_anthropic_formatters():
    """No orphan tool results after compaction, for both wire formats."""
    context = ConversationContext()
    context.add_system_message("system prompt")
    context.add_user_message("Inspect everything")
    seed_tool_groups(context, count=9, body_chars=300)
    context.add_assistant_message("Done inspecting.")

    context.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000)

    assert_no_orphan_tool_results(context.messages)
    # Assistant tool-call payloads survive untouched.
    assistants = [message for message in context.messages if message.tool_calls]
    assert len(assistants) == 9
    assert all(len(message.tool_calls or []) == 1 for message in assistants)


def test_compact_is_deterministic_and_idempotent():
    """Identical history yields identical compacted messages; recompaction is a no-op."""

    def build() -> ConversationContext:
        context = ConversationContext()
        context.add_system_message("system prompt")
        context.add_user_message("Inspect everything")
        seed_tool_groups(context, count=9, body_chars=300)
        context.add_assistant_message("Done inspecting.")
        return context

    first = build()
    second = build()
    first.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000)
    second.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000)
    assert [message.to_dict() for message in first.messages] == [message.to_dict() for message in second.messages]

    # A second compaction has nothing new to elide.
    snapshot = [message.to_dict() for message in first.messages]
    assert first.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000) is False
    assert [message.to_dict() for message in first.messages] == snapshot
    assert first.compaction_count == 1, "no-op compactions are not counted"


def test_compact_never_removes_or_changes_the_system_prompt():
    """The system prompt survives every compaction byte-for-byte."""
    context = ConversationContext()
    context.add_system_message("THE SYSTEM PROMPT")
    seed_tool_groups(context, count=8, body_chars=300)

    context.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000)

    system_messages = [message for message in context.messages if message.role == MessageRole.SYSTEM]
    assert [message.content for message in system_messages] == ["THE SYSTEM PROMPT"]


def test_compact_caps_oversized_tool_results_with_elision_marker():
    """Every tool-result body is capped at max_output_chars with an honest marker."""
    context = ConversationContext()
    context.add_system_message("system prompt")
    context.add_user_message("Inspect everything")
    # 7 tool groups keeps all of them inside the six-group window boundary...
    seed_tool_groups(context, count=7, body_chars=500)
    # ...except the first, which ages out and is elided wholesale.
    context.compact(target_tokens=50, keep_recent=6, max_output_chars=100)

    bodies = [message.content for message in context.messages if message.tool_call_id]
    assert bodies[0] == "[elided 500 chars of tool result for read_file]"
    for body in bodies[1:]:
        assert len(body.splitlines()[0]) == 100
        assert body.splitlines()[1] == "[elided 400 chars of tool result for read_file]"
    assert_no_orphan_tool_results(context.messages)


def test_compact_never_elides_when_the_marker_would_grow_the_body():
    """Tiny tool-result bodies are kept instead of inflated by a marker."""
    context = ConversationContext()
    context.add_system_message("system prompt")
    context.add_user_message("Inspect everything")
    seed_tool_groups(context, count=8, body_chars=2)  # "xx" bodies
    context.add_assistant_message("Done.")

    context.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000)

    assert all(message.content == "xx" for message in context.messages if message.tool_call_id)
    assert context.compaction_count == 0


def test_truncate_old_messages_keeps_tool_groups_atomic():
    """Budget-based truncation never splits an assistant from its tool results."""
    context = ConversationContext()
    context.add_system_message("system")
    context.add_user_message("Inspect everything " + "u" * 400)
    seed_tool_groups(context, count=2, body_chars=400)
    context.add_user_message("What remains open?")

    context.truncate_old_messages(max_tokens=150)

    assert_no_orphan_tool_results(context.messages)
    # The newest tool group survived whole; the oldest was summarized away.
    assert sum(1 for message in context.messages if message.tool_calls) == 1
    assert sum(1 for message in context.messages if message.tool_call_id) == 1
    assert context.messages[0].content == "system"
    assert context.messages[1].content.startswith("Conversation summary")


# ============================================================================
# Measured estimator
# ============================================================================


def test_estimator_prefers_reported_usage_and_falls_back_to_heuristic():
    """The estimator uses provider-reported prompt tokens, else chars//4."""
    context = ConversationContext()
    context.add_user_message("a" * 400)  # heuristic: 100 tokens

    assert context.get_token_count_estimate() == 100
    assert context.estimate_context_tokens() == 100, "no usage reported yet: heuristic"

    context.note_reported_usage({"prompt_tokens": 4321, "completion_tokens": 10})
    assert context.estimate_context_tokens() == 4321, "measured basis wins"

    context.add_user_message("b" * 100)
    assert context.estimate_context_tokens() == 4321, "new messages do not reset the measured basis"

    seed_tool_groups(context, count=8, body_chars=300)
    context.compact(target_tokens=50, keep_recent=6, max_output_chars=20_000)
    assert context.last_reported_prompt_tokens is None
    assert context.estimate_context_tokens() == context.get_token_count_estimate(), (
        "compaction invalidates the measured basis until the next report"
    )


def test_estimator_ignores_usage_without_prompt_tokens():
    """Usage payloads without a positive prompt_tokens never become the basis."""
    context = ConversationContext()
    context.add_user_message("a" * 400)

    context.note_reported_usage({"completion_tokens": 5})
    context.note_reported_usage({})
    context.note_reported_usage({"prompt_tokens": 0})

    assert context.last_reported_prompt_tokens is None
    assert context.estimate_context_tokens() == 100


# ============================================================================
# UsageLedger
# ============================================================================


def test_usage_ledger_aggregates_records():
    """Ledger totals equal the sum of per-call provider-reported usage."""
    ledger = UsageLedger()
    ledger.record("MockLLMProvider", "mock-model", {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
    ledger.record("MockLLMProvider", "mock-model", {"prompt_tokens": 250, "completion_tokens": 5, "total_tokens": 255})

    assert ledger.call_count == 2
    totals = ledger.totals()
    assert totals["prompt_tokens"] == 350
    assert totals["completion_tokens"] == 15
    assert totals["total_tokens"] == 365
    assert totals["calls"] == 2
    assert ledger.records[0].provider == "MockLLMProvider"
    assert ledger.records[0].model == "mock-model"


def test_usage_ledger_ignores_empty_usage_and_fills_missing_totals():
    """Empty payloads are no-ops; a missing total_tokens is prompt+completion."""
    ledger = UsageLedger()
    ledger.record("MockLLMProvider", "mock-model", {})
    ledger.record("MockLLMProvider", "mock-model", {"prompt_tokens": 30})

    assert ledger.call_count == 1
    assert ledger.totals()["total_tokens"] == 30
    assert ledger.totals()["completion_tokens"] == 0


# ============================================================================
# Configuration
# ============================================================================


def test_behavior_config_soft_limit_round_trip():
    """context_soft_limit_ratio serializes and restores; the default is backward compatible."""
    config = AgentBehaviorConfig(context_soft_limit_ratio=0.9)
    restored = AgentBehaviorConfig.from_dict(config.to_dict())
    assert restored.context_soft_limit_ratio == 0.9

    assert AgentBehaviorConfig.from_dict({}).context_soft_limit_ratio == 0.8
    assert AgentBehaviorConfig().context_soft_limit_ratio == 0.8


@pytest.mark.parametrize("ratio", [0, -0.5, 1.5])
def test_behavior_config_rejects_invalid_ratio(ratio: float):
    """Ratios outside (0, 1] are rejected before work begins."""
    with pytest.raises(ValueError):
        AgentBehaviorConfig(context_soft_limit_ratio=ratio)


# ============================================================================
# Pre-call budget check
# ============================================================================


def test_pre_call_check_compacts_only_above_soft_limit(temp_dir, mock_llm_config):
    """The budget is context_size * ratio; crossing it triggers compaction."""
    provider = MockLLMProvider(config=mock_llm_config, context_size=500)  # budget = 400
    agent = make_agent(temp_dir, mock_llm_config, provider)
    seed_tool_groups(agent.context, count=8, body_chars=300)

    agent.context.note_reported_usage({"prompt_tokens": 400})
    agent._enforce_context_budget()
    assert agent.context.compaction_count == 0, "at the limit, not above it"

    agent.context.note_reported_usage({"prompt_tokens": 401})
    agent._enforce_context_budget()
    assert agent.context.compaction_count == 1
    assert agent.context.elided_message_count >= 1


def test_pre_call_check_skipped_without_a_known_context_size(temp_dir, mock_llm_config):
    """A non-positive context_size disables the budget check entirely."""
    provider = MockLLMProvider(config=mock_llm_config, context_size=0)
    agent = make_agent(temp_dir, mock_llm_config, provider)
    seed_tool_groups(agent.context, count=8, body_chars=300)
    agent.context.note_reported_usage({"prompt_tokens": 100_000})

    agent._enforce_context_budget()

    assert agent.context.compaction_count == 0


def test_on_compaction_hook_receives_structured_notice(temp_dir, mock_llm_config):
    """The on_compaction hook mirrors on_retry and carries the compaction stats."""
    notices: list[Any] = []
    provider = MockLLMProvider(config=mock_llm_config, context_size=500)
    agent = make_agent(temp_dir, mock_llm_config, provider, on_compaction=notices.append)
    seed_tool_groups(agent.context, count=8, body_chars=2000)
    agent.context.note_reported_usage({"prompt_tokens": 2000})

    agent._enforce_context_budget()

    assert len(notices) == 1
    notice = notices[0]
    assert notice.target_tokens == 400
    assert notice.elided_messages >= 1
    assert notice.tokens_before > notice.tokens_after
    rendered = format_compaction_notice(notice)
    assert "context compacted" in rendered
    assert "400" in rendered


# ============================================================================
# Loop behavior: usage capture and finish_reason=length
# ============================================================================


@pytest.mark.asyncio
async def test_run_usage_ledger_equals_sum_of_per_call_usage(temp_dir, mock_llm_config):
    """Every call's provider-reported usage lands in the run's ledger."""
    (temp_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (temp_dir / "b.txt").write_text("beta", encoding="utf-8")
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(
                tool_calls=[{"name": "read_file", "parameters": {"path": "a.txt"}}],
                usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            ),
            create_mock_response(
                tool_calls=[{"name": "read_file", "parameters": {"path": "b.txt"}}],
                usage={"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            ),
            create_mock_response(
                content="Both files read.",
                usage={"prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33},
            ),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    await agent.process_message("Read both notes")

    assert provider.call_count == 3
    assert agent.last_run is not None
    totals = agent.last_run.usage.totals()
    assert totals["prompt_tokens"] == 60
    assert totals["completion_tokens"] == 6
    assert totals["total_tokens"] == 66
    assert totals["calls"] == 3


@pytest.mark.asyncio
async def test_length_finish_reason_recovers_once_then_completes(temp_dir, mock_llm_config):
    """A length-truncated response gets the INVALID_RESPONSE recovery path, not a crash."""
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(content="Partial ans", finish_reason="length"),
            create_mock_response(content="The complete answer."),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Answer the question")

    assert provider.call_count == 2, "one recovery attempt after the truncated response"
    assert len(recovery_prompts(agent)) == 1
    assert "Status: succeeded" in report
    assert "The complete answer." in report


@pytest.mark.asyncio
async def test_persistent_length_finish_reason_fails_as_invalid_response(temp_dir, mock_llm_config):
    """A second truncated response fails the run through the invalid_response path."""
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[
            create_mock_response(content="Partial ans", finish_reason="length"),
            create_mock_response(content="Partial ans", finish_reason="length"),
        ],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider)

    report = await agent.process_message("Answer the question")

    assert report.startswith("Status: failed")
    assert "truncated" in report
    assert agent.last_run is not None
    assert agent.last_run.failure_kind is not None
    assert len(recovery_prompts(agent)) == 1, "still exactly one recovery prompt"


# ============================================================================
# Mock provider capabilities
# ============================================================================


def test_mock_provider_injects_context_size(mock_llm_config):
    """MockLLMProvider can force a small context_size; the default is unchanged."""
    injected = MockLLMProvider(config=mock_llm_config, context_size=500)
    assert injected.get_capabilities() == ProviderCapabilities(tools=True, streaming=True, context_size=500)

    default = MockLLMProvider(config=mock_llm_config)
    assert default.get_capabilities() == ProviderCapabilities(tools=True, streaming=True)


def test_create_mock_response_scripts_usage_and_finish_reason():
    """create_mock_response forwards usage/finish_reason into the LLMResponse."""
    provider = MockLLMProvider(
        responses=[
            create_mock_response(
                content="hello",
                usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                finish_reason="length",
            )
        ]
    )

    response = provider.chat([Message(role=MessageRole.USER, content="hi")])

    assert response.usage == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert response.finish_reason == "length"


@pytest.mark.asyncio
async def test_loop_completes_when_compaction_frees_a_small_window(temp_dir, mock_llm_config):
    """A run above a small soft limit compacts mid-run and still completes."""
    for index in range(7):
        (temp_dir / f"note-{index}.txt").write_text("x" * 400, encoding="utf-8")
    provider = SmallWindowProvider(
        config=mock_llm_config,
        context_size=600,
        file_count=7,
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, max_iterations=12)
    system_prompt = agent.context.messages[0].content

    report = await agent.process_message("Read each note file in order")

    assert provider.call_count == 8, "seven tool turns plus the final response"
    assert "Status: succeeded" in report
    assert agent.context.compaction_count >= 1
    assert agent.context.elided_message_count >= 1
    assert_no_orphan_tool_results(agent.context.messages)
    system_messages = [message for message in agent.context.messages if message.role == MessageRole.SYSTEM]
    assert len(system_messages) == 1
    assert system_messages[0].content == system_prompt, "the system prompt survives compaction"


class SmallWindowProvider(MockLLMProvider):
    """Mock provider with a small injected context_size and heavy reported usage."""

    def __init__(self, config: AgentLLMConfig, context_size: int, file_count: int):
        """Initialize the provider.

        Args:
            config: LLM configuration.
            context_size: Context size reported by capabilities.
            file_count: Number of scripted note-reading tool turns.
        """
        responses = [
            create_mock_response(
                tool_calls=[{"name": "read_file", "parameters": {"path": f"note-{index}.txt"}}],
                usage={"prompt_tokens": context_size, "completion_tokens": 8, "total_tokens": context_size + 8},
            )
            for index in range(file_count)
        ]
        responses.append(
            create_mock_response(
                content="All notes read.",
                usage={"prompt_tokens": context_size, "completion_tokens": 4, "total_tokens": context_size + 4},
            )
        )
        super().__init__(config=config, responses=responses, context_size=context_size)


@pytest.mark.asyncio
async def test_compaction_notice_is_not_emitted_without_compaction(temp_dir, mock_llm_config):
    """Runs under the soft limit never emit compaction notices."""
    notices: list[Any] = []
    provider = MockLLMProvider(
        config=mock_llm_config,
        responses=[create_mock_response(content="done")],
    )
    agent = make_agent(temp_dir, mock_llm_config, provider, on_compaction=notices.append)

    await agent.process_message("Just answer")

    assert notices == []
    assert agent.context.compaction_count == 0
