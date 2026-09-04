"""HH-07 unit tests: approval decisions, session memory, binding, and plan modes.

Covers the in-memory ``ApprovalDecision`` contract, boolean-callback adaptation
(backward compatibility), ``ApprovalMemory`` keying/expiry/serialization (in
``ConversationContext.metadata`` and through ``SessionStore``), the
stale-approval binding in ``TaskRun.before_tool`` (TOCTOU re-prompt), and the
``plan_mode`` override matrix.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ctxai.agent.approvals import (
    APPROVAL_MEMORY_KEY,
    ApprovalDecision,
    ApprovalMemory,
    adapt_bool_callback,
    approval_memory_target,
    as_decision_callback,
)
from ctxai.agent.config import AgentConfig
from ctxai.agent.context import ConversationContext
from ctxai.agent.core import AgentLoopConfig
from ctxai.agent.llm.base import ToolCall
from ctxai.agent.sessions import SessionRecord, SessionStore
from ctxai.agent.workflow import FailureKind, TaskRun, validate_plan_mode

RECORDED_AT = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


# ============================================================================
# ApprovalDecision and callback adaptation
# ============================================================================


def test_approval_decision_values_match_the_part_ii_contract():
    """The enum exposes exactly once/session/deny with those values."""
    assert ApprovalDecision.APPROVE_ONCE.value == "once"
    assert ApprovalDecision.APPROVE_SESSION.value == "session"
    assert ApprovalDecision.DENY.value == "deny"
    assert {decision.value for decision in ApprovalDecision} == {"once", "session", "deny"}


def test_adapt_bool_callback_maps_true_to_approve_once():
    """A legacy callback returning True adapts to APPROVE_ONCE and receives the call."""
    seen: list[ToolCall] = []
    adapted = adapt_bool_callback(lambda call: seen.append(call) or True)

    call = ToolCall(id="c1", name="write_file", parameters={"path": "a.txt"})
    assert adapted(call) is ApprovalDecision.APPROVE_ONCE
    assert seen == [call]


def test_adapt_bool_callback_maps_false_to_deny():
    """A legacy callback returning False adapts to DENY."""
    adapted = adapt_bool_callback(lambda call: False)

    call = ToolCall(id="c1", name="write_file", parameters={"path": "a.txt"})
    assert adapted(call) is ApprovalDecision.DENY


def test_as_decision_callback_passes_decisions_through():
    """Callbacks already returning ApprovalDecision are preserved exactly."""
    adapted = as_decision_callback(lambda call: ApprovalDecision.APPROVE_SESSION)

    call = ToolCall(id="c1", name="bash", parameters={"command": "python -m pytest"})
    assert adapted(call) is ApprovalDecision.APPROVE_SESSION


def test_as_decision_callback_adapts_booleans_for_backward_compatibility():
    """Legacy boolean callbacks keep working through the decision protocol."""
    call = ToolCall(id="c1", name="write_file", parameters={"path": "a.txt"})
    assert as_decision_callback(lambda c: True)(call) is ApprovalDecision.APPROVE_ONCE
    assert as_decision_callback(lambda c: False)(call) is ApprovalDecision.DENY


def test_as_decision_callback_coerces_value_strings():
    """Raw decision-value strings (e.g. from a config-driven callback) coerce."""
    call = ToolCall(id="c1", name="write_file", parameters={"path": "a.txt"})
    assert as_decision_callback(lambda c: "session")(call) is ApprovalDecision.APPROVE_SESSION


def test_as_decision_callback_fails_closed_on_unrecognized_results():
    """Unrecognizable callback results deny instead of approving."""
    call = ToolCall(id="c1", name="write_file", parameters={"path": "a.txt"})
    assert as_decision_callback(lambda c: "maybe")(call) is ApprovalDecision.DENY
    assert as_decision_callback(lambda c: None)(call) is ApprovalDecision.DENY


def test_as_decision_callback_none_stays_none():
    """A missing callback stays None so the loop keeps denying with no prompt."""
    assert as_decision_callback(None) is None


# ============================================================================
# ApprovalMemory keying and expiry
# ============================================================================


def test_memory_is_keyed_by_tool_and_target_pair():
    """A session grant suppresses prompts for exactly that key and nothing else."""
    memory = ApprovalMemory()
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)

    assert memory.check("write_file", "note.txt", now=RECORDED_AT) is ApprovalDecision.APPROVE_SESSION
    # Same pattern, different tool: no grant.
    assert memory.check("edit_file", "note.txt", now=RECORDED_AT) is None
    assert memory.check("bash", "note.txt", now=RECORDED_AT) is None
    # Same tool, different pattern: no grant.
    assert memory.check("write_file", "other.txt", now=RECORDED_AT) is None


def test_memory_record_overwrites_previous_decision_for_the_same_key():
    """A later decision for the same key replaces the earlier one."""
    memory = ApprovalMemory()
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)

    assert memory.check("write_file", "note.txt", now=RECORDED_AT) is ApprovalDecision.APPROVE_SESSION
    assert len(memory.decisions) == 1


def test_memory_expiry_after_max_age():
    """Entries expire after max_age_seconds and are dropped on check."""
    memory = ApprovalMemory(max_age_seconds=60)
    memory.record("bash", "python", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)

    assert memory.check("bash", "python", now=RECORDED_AT + timedelta(seconds=59)) is ApprovalDecision.APPROVE_SESSION
    assert memory.check("bash", "python", now=RECORDED_AT + timedelta(seconds=61)) is None
    assert memory.key("bash", "python") not in memory.decisions


def test_memory_without_max_age_never_expires():
    """max_age_seconds=None keeps entries valid indefinitely (session lifetime)."""
    memory = ApprovalMemory(max_age_seconds=None)
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)

    assert (
        memory.check("write_file", "note.txt", now=RECORDED_AT + timedelta(days=3650))
        is ApprovalDecision.APPROVE_SESSION
    )


def test_memory_ignores_entries_with_unparseable_timestamps():
    """Corrupt persisted entries fail closed instead of auto-approving."""
    memory = ApprovalMemory(max_age_seconds=60)
    memory.decisions[memory.key("write_file", "note.txt")] = {"decision": "session", "recorded_at": "not-a-date"}

    assert memory.check("write_file", "note.txt", now=RECORDED_AT) is None


# ============================================================================
# ApprovalMemory serialization round trip
# ============================================================================


def test_memory_round_trips_through_to_dict_and_from_dict():
    """to_dict/from_dict preserve decisions and timestamps exactly."""
    memory = ApprovalMemory()
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)
    memory.record("bash", "python", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)

    restored = ApprovalMemory.from_dict(memory.to_dict())

    assert restored.check("write_file", "note.txt", now=RECORDED_AT) is ApprovalDecision.APPROVE_SESSION
    assert restored.check("bash", "python", now=RECORDED_AT) is ApprovalDecision.APPROVE_SESSION
    assert restored.max_age_seconds == ApprovalMemory().max_age_seconds


def test_memory_to_dict_is_json_serializable():
    """The persisted shape must survive the session store's json.dump."""
    memory = ApprovalMemory()
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)

    payload = json.loads(json.dumps(memory.to_dict()))
    assert (
        ApprovalMemory.from_dict(payload).check("write_file", "note.txt", now=RECORDED_AT)
        is ApprovalDecision.APPROVE_SESSION
    )


def test_memory_from_dict_tolerates_garbage_and_empty_payloads():
    """from_dict starts fresh on missing/garbage input instead of raising."""
    assert ApprovalMemory.from_dict({}).decisions == {}
    assert ApprovalMemory.from_dict({"decisions": None}).decisions == {}
    assert ApprovalMemory.from_dict({"decisions": {"bad": "not-a-dict"}}).decisions == {}
    assert ApprovalMemory.from_dict("garbage").decisions == {}


def test_memory_persists_through_conversation_context_round_trip():
    """Approval memory rides in ConversationContext.metadata across to_dict/from_dict."""
    context = ConversationContext()
    memory = ApprovalMemory()
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)
    context.metadata[APPROVAL_MEMORY_KEY] = memory.to_dict()

    restored = ConversationContext.from_dict(context.to_dict())
    assert APPROVAL_MEMORY_KEY in restored.metadata
    assert (
        ApprovalMemory.from_dict(restored.metadata[APPROVAL_MEMORY_KEY]).check(
            "write_file", "note.txt", now=RECORDED_AT
        )
        is ApprovalDecision.APPROVE_SESSION
    )


def test_memory_persists_through_session_store(tmp_path):
    """A saved/resumed session carries its approval memory (redacted like all session data)."""
    store = SessionStore(tmp_path)
    context = ConversationContext()
    memory = ApprovalMemory()
    memory.record("write_file", "note.txt", ApprovalDecision.APPROVE_SESSION, recorded_at=RECORDED_AT)
    context.metadata[APPROVAL_MEMORY_KEY] = memory.to_dict()
    context.add_user_message("hello")
    store.save(SessionRecord(name="default", context=context, provider="mock", model="m", project_root=str(tmp_path)))

    loaded = store.load("default")
    assert (
        ApprovalMemory.from_dict(loaded.context.metadata[APPROVAL_MEMORY_KEY]).check(
            "write_file", "note.txt", now=RECORDED_AT
        )
        is ApprovalDecision.APPROVE_SESSION
    )


# ============================================================================
# Memory key patterns (exact path for mutations, executable for commands)
# ============================================================================


def test_memory_target_for_mutations_is_the_canonical_relative_path(tmp_path):
    """Equivalent path spellings collapse to one repository-relative key."""
    (tmp_path / "sub").mkdir()
    call = ToolCall(id="c1", name="write_file", parameters={"path": "sub/../note.txt", "content": "x"})
    assert approval_memory_target(call, tmp_path) == "note.txt"

    absolute = ToolCall(id="c2", name="write_file", parameters={"path": str(tmp_path / "note.txt"), "content": "x"})
    assert approval_memory_target(absolute, tmp_path) == "note.txt"


def test_memory_target_for_mutations_outside_the_project_stays_absolute(tmp_path):
    """Paths outside the project root keep their absolute canonical form."""
    outside = tmp_path.parent / "outside-approval-target.txt"
    call = ToolCall(id="c1", name="edit_file", parameters={"path": str(outside)})
    assert approval_memory_target(call, tmp_path) == str(Path(os.path.realpath(outside)))


def test_memory_target_for_commands_is_the_executable():
    """Command grants key on the executable name, not the full command line."""
    call = ToolCall(id="c1", name="bash", parameters={"command": "/usr/bin/env python -m pytest -q"})
    assert approval_memory_target(call, None) == "env"

    plain = ToolCall(id="c2", name="bash", parameters={"command": "python3 script.py"})
    assert approval_memory_target(plain, None) == "python3"


def test_memory_target_for_commands_falls_back_to_the_command_when_unparseable():
    """A command that cannot be tokenized keys on the raw command."""
    call = ToolCall(id="c1", name="bash", parameters={"command": "'unbalanced"})
    assert approval_memory_target(call, None) == "'unbalanced"

    empty = ToolCall(id="c2", name="bash", parameters={})
    assert approval_memory_target(empty, None) == "bash"


def test_memory_target_for_other_tools_uses_the_approval_target():
    """Non-mutation, non-command tools fall back to the approval_target parameter."""
    call = ToolCall(id="c1", name="git_diff", parameters={"approval_target": "git diff"})
    assert approval_memory_target(call, None) == "git diff"

    bare = ToolCall(id="c2", name="semantic_search", parameters={})
    assert approval_memory_target(bare, None) == "semantic_search"


# ============================================================================
# Stale-approval binding (TOCTOU) in TaskRun.before_tool
# ============================================================================


def _edit_run(tmp_path: Path) -> TaskRun:
    """Build a TaskRun with app.py inspected so edit_file passes the inspect gate."""
    run = TaskRun("edit task", project_root=tmp_path)
    run.inspected_files.add(run.canonical(tmp_path / "app.py"))
    return run


def _edit_call() -> ToolCall:
    return ToolCall(
        id="c1", name="edit_file", parameters={"path": "app.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"}
    )


def test_before_tool_reprompts_when_file_changed_between_approval_and_execution(tmp_path):
    """A stale approval re-prompts with a fresh diff; the stale approval never executes."""
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    run = _edit_run(tmp_path)
    invocations: list[ToolCall] = []

    def approval(approval_call: ToolCall) -> bool:
        invocations.append(approval_call)
        if len(invocations) == 1:
            # The file moves on after the human saw the diff (another writer).
            target.write_text("VALUE = 1 touched\n", encoding="utf-8")
        return True

    denial = run.before_tool(_edit_call(), planning_enabled=False, require_approval=True, approval_callback=approval)

    assert denial is None, "the fresh re-approval must execute"
    assert len(invocations) == 2, "the stale approval must re-prompt exactly once here"
    assert "VALUE = 1 touched" in invocations[1].parameters["proposed_diff"], "the re-prompt shows a fresh diff"
    assert invocations[0].parameters["proposed_diff"] != invocations[1].parameters["proposed_diff"]
    assert [entry["approved"] for entry in run.approvals] == [True, True]
    assert run.failure_kind is None


def test_before_tool_attaches_binding_hashes_to_mutation_approvals(tmp_path):
    """Mutation approval calls carry the proposed-diff and pre-approval content hashes."""
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    run = _edit_run(tmp_path)

    denial = run.before_tool(
        _edit_call(), planning_enabled=False, require_approval=True, approval_callback=lambda call: True
    )

    assert denial is None
    parameters = run.approvals[0]["parameters"]
    import hashlib

    assert parameters["proposed_diff_sha256"] == hashlib.sha256(parameters["proposed_diff"].encode("utf-8")).hexdigest()
    assert parameters["pre_approval_content_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_before_tool_reprompts_when_approved_file_appears(tmp_path):
    """An approval for a not-yet-existing file re-prompts when the file now exists."""
    target = tmp_path / "created.txt"
    invocations: list[ToolCall] = []

    def approval(approval_call: ToolCall) -> bool:
        invocations.append(approval_call)
        if len(invocations) == 1:
            target.write_text("surprise\n", encoding="utf-8")
        return True

    run = TaskRun("create task", project_root=tmp_path)
    call = ToolCall(id="c1", name="write_file", parameters={"path": "created.txt", "content": "body"})
    denial = run.before_tool(call, planning_enabled=False, require_approval=True, approval_callback=approval)

    assert denial is None
    assert len(invocations) == 2
    assert invocations[0].parameters["pre_approval_content_sha256"] is None


def test_before_tool_fails_after_the_stale_reprompt_cap(tmp_path):
    """Every round going stale ends in APPROVAL_DENIAL instead of an infinite loop."""
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    run = _edit_run(tmp_path)
    invocations: list[ToolCall] = []

    def approval(approval_call: ToolCall) -> bool:
        invocations.append(approval_call)
        # Mutate on every ask so every approval is stale by execution time.
        target.write_text(f"VALUE = 1 moved-{len(invocations)}\n", encoding="utf-8")
        return True

    denial = run.before_tool(_edit_call(), planning_enabled=False, require_approval=True, approval_callback=approval)

    assert denial is not None
    assert denial["error_type"] == FailureKind.APPROVAL_DENIAL.value
    assert len(invocations) == 4, "initial prompt plus at most three stale re-prompts"
    assert "stale" in (run.failure_message or "")


def test_before_tool_skips_binding_for_commands(tmp_path):
    """bash approvals carry no content binding, so one approval suffices."""
    run = TaskRun("run task", project_root=tmp_path)
    invocations: list[ToolCall] = []
    call = ToolCall(id="c1", name="bash", parameters={"command": "python -m pytest"})
    denial = run.before_tool(
        call, planning_enabled=False, require_approval=True, approval_callback=lambda c: invocations.append(c) or True
    )

    assert denial is None
    assert len(invocations) == 1
    assert "pre_approval_content_sha256" not in run.approvals[0]["parameters"]


def test_before_tool_boolean_false_keeps_the_approval_denial_path(tmp_path):
    """Legacy boolean False still produces the existing APPROVAL_DENIAL denial."""
    run = TaskRun("run task", project_root=tmp_path)
    call = ToolCall(id="c1", name="bash", parameters={"command": "python -m pytest"})
    denial = run.before_tool(call, planning_enabled=False, require_approval=True, approval_callback=lambda c: False)

    assert denial is not None
    assert denial["error_type"] == FailureKind.APPROVAL_DENIAL.value
    assert run.approvals[0]["approved"] is False


# ============================================================================
# Plan-mode override matrix
# ============================================================================


@pytest.mark.parametrize(
    ("goal", "plan_mode", "expected"),
    [
        ("refactor across multiple files", "auto", True),  # keyword classification
        ("refactor across multiple files", "off", False),  # off suppresses flagged tasks
        ("add a tiny comment to note.txt", "force", True),  # force flags simple tasks
        ("add a tiny comment to note.txt", "auto", False),
        ("add a tiny comment to note.txt", "off", False),
    ],
)
def test_plan_mode_override_matrix(goal, plan_mode, expected):
    """TaskRun.resolve_plan_required applies the explicit override channel."""
    assert TaskRun.resolve_plan_required(goal, plan_mode) is expected


def test_task_run_applies_plan_mode_in_post_init():
    """TaskRun computes plan_required from plan_mode at construction."""
    flagged = TaskRun("refactor across multiple files")
    assert flagged.plan_required is True
    assert flagged.plan_mode == "auto"

    suppressed = TaskRun("refactor across multiple files", plan_mode="off")
    assert suppressed.plan_required is False

    forced = TaskRun("add a tiny comment to note.txt", plan_mode="force")
    assert forced.plan_required is True


def test_validate_plan_mode_rejects_unknown_values():
    """Only auto/force/off are accepted."""
    assert validate_plan_mode("auto") == "auto"
    with pytest.raises(ValueError):
        validate_plan_mode("sometimes")


def test_task_run_rejects_invalid_plan_mode():
    with pytest.raises(ValueError):
        TaskRun("goal", plan_mode="sometimes")


def _minimal_loop_config(plan_mode: str = "auto") -> AgentLoopConfig:
    """Build an AgentLoopConfig without providers or registries (validation only)."""
    return AgentLoopConfig(
        llm_provider=None,  # type: ignore[arg-type]
        tool_registry=None,  # type: ignore[arg-type]
        agent_config=AgentConfig(),
        working_directory=Path.cwd(),
        available_indexes=[],
        plan_mode=plan_mode,
    )


def test_agent_loop_config_validates_plan_mode():
    """AgentLoopConfig validates plan_mode at construction and via set_plan_mode."""
    config = _minimal_loop_config()
    assert config.plan_mode == "auto"

    with pytest.raises(ValueError):
        _minimal_loop_config("sometimes")

    config.set_plan_mode("force")
    assert config.plan_mode == "force"

    with pytest.raises(ValueError):
        config.set_plan_mode("sometimes")
    assert config.plan_mode == "force", "an invalid set_plan_mode leaves the mode unchanged"
