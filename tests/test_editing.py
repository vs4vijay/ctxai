"""Unit tests for deterministic, uniqueness-checked edits (HH-01)."""

import pytest

from ctxai.agent.editing import EditError, apply_edit, edit_diff, simulate_edit
from ctxai.agent.llm.base import ToolCall
from ctxai.agent.tools.execution import ToolExecutionContext
from ctxai.agent.tools.file_ops import EditFileTool
from ctxai.agent.workflow import TaskRun


def test_single_literal_match_replaces_once_with_exact_strategy():
    outcome = apply_edit("value = 1\n", "1", "2")
    assert outcome.after == "value = 2\n"
    assert outcome.count == 1
    assert outcome.strategy == "exact"


def test_zero_literal_matches_fail_closed_and_name_the_count():
    with pytest.raises(EditError) as excinfo:
        apply_edit("value = 1\n", "missing", "x")
    assert "0" in str(excinfo.value)


def test_multiple_literal_matches_fail_and_name_the_count():
    before = "x = 1\ny = 1\nz = 1\n"
    with pytest.raises(EditError) as excinfo:
        apply_edit(before, "1", "2")
    assert "3" in str(excinfo.value)


def test_file_is_untouched_on_failure_is_enforced_by_apply_purity():
    # apply_edit is a pure function: failure must never produce partial content.
    before = "x = 1\ny = 1\n"
    with pytest.raises(EditError):
        apply_edit(before, "1", "2")
    assert before == "x = 1\ny = 1\n"


def test_replace_all_replaces_every_occurrence_and_names_strategy():
    before = "a=1\nb=1\nc=1\n"
    outcome = apply_edit(before, "1", "2", replace_all=True)
    assert outcome.after == "a=2\nb=2\nc=2\n"
    assert outcome.count == 3
    assert outcome.strategy == "replace_all"


def test_replace_all_with_single_match_still_reports_replace_all_strategy():
    outcome = apply_edit("only 1 here\n", "1", "one", replace_all=True)
    assert outcome.after == "only one here\n"
    assert outcome.count == 1
    assert outcome.strategy == "replace_all"


def test_single_regex_match_replaces_with_exact_strategy():
    outcome = apply_edit("VALUE = 1\n", r"VALUE = \d+", "VALUE = 2", use_regex=True)
    assert outcome.after == "VALUE = 2\n"
    assert outcome.count == 1
    assert outcome.strategy == "exact"


def test_multiple_regex_matches_fail_and_name_the_count():
    before = "v1\nv2\nv3\n"
    with pytest.raises(EditError) as excinfo:
        apply_edit(before, r"v\d", "v", use_regex=True)
    assert "3" in str(excinfo.value)


def test_multiple_regex_matches_with_replace_all_apply_everywhere():
    outcome = apply_edit("v1\nv2\nv3\n", r"v\d", "v", use_regex=True, replace_all=True)
    assert outcome.after == "v\nv\nv\n"
    assert outcome.count == 3
    assert outcome.strategy == "replace_all"


def test_zero_regex_matches_fail_closed():
    with pytest.raises(EditError) as excinfo:
        apply_edit("abc\n", r"xyz\d+", "q", use_regex=True)
    assert "0" in str(excinfo.value)


def test_invalid_regex_fails_as_edit_error():
    with pytest.raises(EditError):
        apply_edit("abc\n", "a(", "b", use_regex=True)


def test_normalized_fallback_collapses_whitespace_runs_and_preserves_original_bytes():
    before = "def   main():\n    return  1\n"
    # Exact pattern (single spaces) does not occur literally.
    outcome = apply_edit(before, "def main():", "def run():")
    assert outcome.after == "def run():\n    return  1\n"
    assert outcome.count == 1
    assert outcome.strategy == "normalized"


def test_normalized_fallback_maps_replacement_into_original_region():
    before = "x  =  1\nkeep = 2\n"
    outcome = apply_edit(before, "x = 1", "x = one")
    assert outcome.after == "x = one\nkeep = 2\n"
    assert outcome.strategy == "normalized"


def test_normalized_fallback_strips_trailing_whitespace_when_matching():
    before = "return value   \nnext = 1\n"
    outcome = apply_edit(before, "return value\nnext = 1", "return value\nnext = 2")
    assert outcome.after == "return value\nnext = 2\n"
    assert outcome.strategy == "normalized"


def test_normalized_fallback_regex_mode_matches_normalized_content():
    before = "def   foo():\n    pass\n"
    outcome = apply_edit(before, r"def  foo\(\)", "def bar()", use_regex=True)
    assert outcome.after == "def bar():\n    pass\n"
    assert outcome.count == 1
    assert outcome.strategy == "normalized"


def test_normalized_fallback_must_also_match_exactly_once():
    before = "a  = 1\na =  1\n"
    with pytest.raises(EditError) as excinfo:
        apply_edit(before, "a = 1", "a = 2")
    assert "2" in str(excinfo.value)


def test_normalized_fallback_zero_matches_names_zero():
    with pytest.raises(EditError) as excinfo:
        apply_edit("alpha\n", "omega", "x")
    assert "0" in str(excinfo.value)


def test_empty_old_text_is_rejected_without_replace_all():
    with pytest.raises(EditError):
        apply_edit("abc\n", "", "x")


def test_simulate_edit_write_file_returns_full_content():
    after, count = simulate_edit("write_file", {"content": "brand new\n"}, "old contents\n")
    assert after == "brand new\n"
    assert count == 1


def test_simulate_edit_edit_file_matches_apply_edit_for_regex_edits():
    before = "v = 1\nw = v1\nv2 = 3\n"
    parameters = {"old_text": r"v\d", "new_text": "vX", "use_regex": True, "replace_all": True}
    simulated_after, simulated_count = simulate_edit("edit_file", parameters, before)
    outcome = apply_edit(before, parameters["old_text"], parameters["new_text"], use_regex=True, replace_all=True)
    assert simulated_after == outcome.after
    assert simulated_count == outcome.count


def test_simulate_edit_edit_file_honors_replace_all():
    before = "1\n1\n"
    after, count = simulate_edit("edit_file", {"old_text": "1", "new_text": "2", "replace_all": True}, before)
    assert after == "2\n2\n"
    assert count == 2


def test_simulate_edit_rejects_unknown_tools():
    with pytest.raises(EditError):
        simulate_edit("bash", {"command": "ls"}, "")


def test_edit_diff_uses_the_passed_display_path():
    difference = edit_diff("src/app.py", "a\n", "b\n")
    assert "--- a/src/app.py" in difference
    assert "+++ b/src/app.py" in difference


@pytest.mark.asyncio
async def test_approval_preview_diff_is_byte_identical_to_applied_diff(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("v = 1\nw = v1 times\n", encoding="utf-8")
    run = TaskRun("edit task", project_root=tmp_path)
    run.inspected_files.add(run.canonical(target))
    call = ToolCall(
        id="call-1",
        name="edit_file",
        parameters={"path": "app.py", "old_text": r"v\d", "new_text": "vX", "use_regex": True},
    )
    captured: list[ToolCall] = []

    def approval(approval_call: ToolCall) -> bool:
        captured.append(approval_call)
        return True

    denial = run.before_tool(call, planning_enabled=False, require_approval=True, approval_callback=approval)
    assert denial is None

    context = ToolExecutionContext.for_project(tmp_path)
    result = await EditFileTool(context=context).execute(
        file_path="app.py", old_text=r"v\d", new_text="vX", use_regex=True
    )
    assert result["success"]
    assert captured[0].parameters["proposed_diff"] == result["diff"]


@pytest.mark.asyncio
async def test_approval_preview_diff_is_byte_identical_for_absolute_paths(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("v = 1\n", encoding="utf-8")
    run = TaskRun("edit task", project_root=tmp_path)
    run.inspected_files.add(run.canonical(target))
    call = ToolCall(
        id="call-2",
        name="edit_file",
        parameters={"file_path": str(target), "old_text": "v = 1", "new_text": "v = 2"},
    )
    captured: list[ToolCall] = []
    denial = run.before_tool(
        call,
        planning_enabled=False,
        require_approval=True,
        approval_callback=lambda approval_call: captured.append(approval_call) or True,
    )
    assert denial is None

    context = ToolExecutionContext.for_project(tmp_path)
    result = await EditFileTool(context=context).execute(file_path=str(target), old_text="v = 1", new_text="v = 2")
    assert result["success"]
    assert captured[0].parameters["proposed_diff"] == result["diff"]
