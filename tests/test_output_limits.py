"""Unit tests for bounded tool output (HH-01)."""

import pytest

from ctxai.agent.tools.output_limits import DEFAULT_MAX_OUTPUT_CHARS, truncate_text

MARKER_PATTERN = "...[truncated {} of {} chars]"


def test_default_limit_matches_configured_bound():
    assert DEFAULT_MAX_OUTPUT_CHARS == 20_000


def test_text_within_limit_is_returned_unchanged():
    text = "short output\n"
    assert truncate_text(text, DEFAULT_MAX_OUTPUT_CHARS, label="stdout") == text


def test_text_exactly_at_limit_is_returned_unchanged():
    text = "x" * 100
    assert truncate_text(text, 100, label="stdout") == text


def test_text_one_char_over_limit_truncates_one_char():
    text = "x" * 101
    result = truncate_text(text, 100, label="stdout")
    assert result == "x" * 100 + MARKER_PATTERN.format(1, 101)


def test_truncation_marker_states_truncated_and_total_counts():
    text = "y" * 250
    result = truncate_text(text, 200, label="stdout")
    assert result.endswith(MARKER_PATTERN.format(50, 250))
    assert len(result) == 200 + len(MARKER_PATTERN.format(50, 250))


def test_empty_string_is_returned_unchanged():
    assert truncate_text("", 100, label="stdout") == ""


def test_text_without_trailing_newline_still_receives_marker():
    result = truncate_text("no trailing newline but long enough" + "!" * 100, 10, label="read_file")
    assert result.startswith("no trailin")
    assert result.endswith(
        MARKER_PATTERN.format(
            len("no trailing newline but long enough" + "!" * 100) - 10,
            len("no trailing newline but long enough" + "!" * 100),
        )
    )


def test_zero_limit_truncates_everything():
    result = truncate_text("abcdef", 0, label="stdout")
    assert result == MARKER_PATTERN.format(6, 6)


def test_negative_limit_truncates_everything():
    result = truncate_text("abcdef", -5, label="stdout")
    assert result == MARKER_PATTERN.format(6, 6)


def test_truncation_never_throws_on_any_input():
    assert truncate_text(None, 10, label="stdout") == ""  # type: ignore[arg-type]
    assert truncate_text("", None, label="stdout") == ""  # type: ignore[arg-type]
    assert truncate_text("text", None, label="stdout") == "text"  # type: ignore[arg-type]
    assert truncate_text("text", "bogus", label="stdout") == "text"  # type: ignore[arg-type]
    assert truncate_text(None, None, label="stdout") == ""  # type: ignore[arg-type]


def test_label_is_accepted_for_every_stream():
    for label in ("stdout", "stderr", "read_file"):
        assert truncate_text("z" * 30, 10, label=label).endswith(MARKER_PATTERN.format(20, 30))


@pytest.mark.parametrize("limit", [0, 1, 7, 999])
def test_truncated_result_is_deterministic(limit):
    text = "abc" * 50
    assert truncate_text(text, limit, label="stdout") == truncate_text(text, limit, label="stdout")
