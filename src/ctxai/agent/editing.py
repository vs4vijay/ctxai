"""Shared, deterministic edit semantics for tool edits and approval previews (HH-01).

Both the approval-preview path (``workflow.TaskRun._approval_call``) and the
applied path (``tools.file_ops.EditFileTool``) route through :func:`apply_edit`
so the diff a human approves is byte-identical to the change that gets written.
Edits fail closed: a pattern that matches zero or several occurrences is an
error that names the match count, unless ``replace_all`` is explicitly set.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

STRATEGY_EXACT = "exact"
STRATEGY_NORMALIZED = "normalized"
STRATEGY_REPLACE_ALL = "replace_all"

_WHITESPACE_RUN = re.compile(r"[ \t]+")


class EditError(ValueError):
    """Raised when an edit cannot be applied unambiguously."""


@dataclass(frozen=True)
class EditOutcome:
    """Result of applying an edit to in-memory content.

    Attributes:
        after: The full content after the edit was applied.
        count: Number of replacements that were applied.
        strategy: How the match was located: ``"exact"``, ``"normalized"``
            (whitespace-tolerant fallback), or ``"replace_all"``.
    """

    after: str
    count: int
    strategy: str


def apply_edit(
    before: str,
    old_text: str,
    new_text: str,
    *,
    use_regex: bool = False,
    replace_all: bool = False,
) -> EditOutcome:
    """Apply one uniqueness-checked replacement to in-memory content.

    The edit requires exactly one match unless ``replace_all`` is set. When the
    exact match fails, one bounded whitespace-tolerant fallback is attempted:
    runs of spaces/tabs are collapsed to a single space and trailing whitespace
    is stripped per line, on both the content and the pattern; the fallback
    must also match exactly once, and the replacement is spliced into the
    original bytes at the matched region (normalized content is never written).

    Args:
        before: The complete content being edited.
        old_text: Literal text to replace, or a regular expression when
            ``use_regex`` is set.
        new_text: Replacement text (pattern expansion applies in regex mode).
        use_regex: Treat ``old_text`` as a regular expression.
        replace_all: Replace every occurrence instead of requiring a unique
            match.

    Returns:
        The :class:`EditOutcome` describing the new content, the replacement
        count, and the applied strategy.

    Raises:
        EditError: When the pattern matches zero or several occurrences
            (without ``replace_all``), or the pattern is not a valid regular
            expression. The error message always names the match count.
    """
    if use_regex:
        return _apply_regex_edit(before, old_text, new_text, replace_all=replace_all)
    return _apply_literal_edit(before, old_text, new_text, replace_all=replace_all)


def simulate_edit(tool_name: str, parameters: dict, before: str) -> tuple[str, int]:
    """Simulate a mutation tool call against known content.

    This is the single simulation used for approval previews; it delegates to
    :func:`apply_edit` so previews cannot diverge from applied edits.

    Args:
        tool_name: Name of the mutation tool (``"edit_file"`` or
            ``"write_file"``).
        parameters: Tool call parameters as provided by the model.
        before: The current content of the target file (empty when the file
            does not exist yet).

    Returns:
        Tuple of the simulated content after the edit and the replacement
        count (a whole-file ``write_file`` counts as one replacement).

    Raises:
        EditError: When the tool is unsupported or the edit is ambiguous.
    """
    if tool_name == "write_file":
        return str(parameters.get("content", "")), 1
    if tool_name == "edit_file":
        outcome = apply_edit(
            before,
            str(parameters.get("old_text", "")),
            str(parameters.get("new_text", "")),
            use_regex=bool(parameters.get("use_regex", False)),
            replace_all=bool(parameters.get("replace_all", False)),
        )
        return outcome.after, outcome.count
    raise EditError(f"simulate_edit does not support tool: {tool_name}")


def edit_diff(display_path: str, before: str, after: str) -> str:
    """Render a unified diff with stable ``a/`` and ``b/`` labels.

    Args:
        display_path: Path label shown in the diff headers.
        before: Content before the change.
        after: Content after the change.

    Returns:
        The unified diff text.
    """
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
        )
    )


def _uniqueness_message(count: int, *, normalized: bool = False) -> str:
    """Build the fail-closed error message for a non-unique match.

    Args:
        count: Number of occurrences the pattern matched.
        normalized: Whether the whitespace-normalized fallback was attempted.

    Returns:
        Error text naming the match count.
    """
    scope = " with exact and whitespace-normalized matching" if normalized else ""
    return (
        f"Edit failed: pattern matched {count} occurrence(s){scope}; "
        "exactly 1 match is required unless replace_all is set; no changes were written"
    )


def _apply_literal_edit(before: str, old_text: str, new_text: str, *, replace_all: bool) -> EditOutcome:
    """Apply a literal replacement with uniqueness enforcement.

    Args:
        before: The complete content being edited.
        old_text: Literal text to replace.
        new_text: Replacement text.
        replace_all: Replace every occurrence instead of requiring a unique
            match.

    Returns:
        The :class:`EditOutcome` for the applied literal edit.

    Raises:
        EditError: When the match is not unique and ``replace_all`` is unset.
    """
    count = before.count(old_text)
    if count >= 1 and replace_all:
        return EditOutcome(before.replace(old_text, new_text), count, STRATEGY_REPLACE_ALL)
    if count == 1:
        return EditOutcome(before.replace(old_text, new_text, 1), 1, STRATEGY_EXACT)
    if count == 0:
        return _normalized_fallback(before, old_text, new_text, use_regex=False)
    raise EditError(_uniqueness_message(count))


def _apply_regex_edit(before: str, old_text: str, new_text: str, *, replace_all: bool) -> EditOutcome:
    """Apply a regular-expression replacement with uniqueness enforcement.

    Args:
        before: The complete content being edited.
        old_text: Regular expression to replace.
        new_text: Replacement text; backreferences are expanded by ``re``.
        replace_all: Replace every occurrence instead of requiring a unique
            match.

    Returns:
        The :class:`EditOutcome` for the applied regex edit.

    Raises:
        EditError: When the match is not unique and ``replace_all`` is unset,
            or the pattern is invalid.
    """
    try:
        count = len(list(re.finditer(old_text, before)))
    except re.error as exc:
        raise EditError(f"Invalid regular expression: {exc}") from exc
    if count == 1 or (replace_all and count >= 1):
        after, applied = re.subn(old_text, new_text, before, count=0 if replace_all else 1)
        return EditOutcome(after, applied, STRATEGY_REPLACE_ALL if replace_all else STRATEGY_EXACT)
    if count == 0:
        return _normalized_fallback(before, old_text, new_text, use_regex=True)
    raise EditError(_uniqueness_message(count))


def _normalized_fallback(before: str, old_text: str, new_text: str, *, use_regex: bool) -> EditOutcome:
    """Attempt the bounded whitespace-tolerant fallback.

    The content and the pattern are normalized (runs of spaces/tabs collapse to
    one space, trailing whitespace is stripped per line); the normalized
    pattern must match the normalized content exactly once. The replacement is
    then spliced into the ORIGINAL content using an index mapping, so only the
    matched region is touched and normalized content is never written back.

    Args:
        before: The complete content being edited.
        old_text: Literal text or regular expression to replace.
        new_text: Replacement text.
        use_regex: Treat ``old_text`` as a regular expression.

    Returns:
        The :class:`EditOutcome` with the ``"normalized"`` strategy.

    Raises:
        EditError: When the normalized pattern does not match exactly once; the
            error names the match count.
    """
    normalized_content, index_map = _normalize_with_index(before)
    normalized_pattern, _ = _normalize_with_index(old_text)
    if not normalized_pattern:
        raise EditError(_uniqueness_message(0, normalized=True))
    try:
        if use_regex:
            matches = list(re.finditer(normalized_pattern, normalized_content))
            spans = [match.span() for match in matches]
            replacement = matches[0].expand(new_text) if len(matches) == 1 else ""
        else:
            spans = _find_all(normalized_content, normalized_pattern)
            replacement = new_text
    except re.error as exc:
        raise EditError(f"Invalid regular expression: {exc}") from exc
    if len(spans) != 1:
        raise EditError(_uniqueness_message(len(spans), normalized=True))
    start, end = spans[0]
    original_start = index_map[start]
    original_end = index_map[end - 1] + 1
    after = before[:original_start] + replacement + before[original_end:]
    return EditOutcome(after, 1, STRATEGY_NORMALIZED)


def _find_all(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Find all (possibly overlapping) occurrences of a literal substring.

    Args:
        haystack: Text to search.
        needle: Literal substring to locate.

    Returns:
        List of ``(start, end)`` spans in ``haystack``.
    """
    spans: list[tuple[int, int]] = []
    if not needle:
        return spans
    position = haystack.find(needle)
    while position != -1:
        spans.append((position, position + len(needle)))
        position = haystack.find(needle, position + 1)
    return spans


def _normalize_with_index(text: str) -> tuple[str, list[int]]:
    """Normalize whitespace while tracking original character positions.

    Normalization collapses runs of spaces/tabs to a single space and strips
    trailing whitespace per line, exactly as the HH-01 fallback prescribes.

    Args:
        text: Text to normalize.

    Returns:
        Tuple of the normalized text and an index map where entry ``i`` is the
        original index of the ``i``-th normalized character.
    """
    chars: list[str] = []
    index_map: list[int] = []
    offset = 0
    lines = text.split("\n")
    for line_number, line in enumerate(lines):
        core = line.rstrip()
        run_start = -1
        for position, char in enumerate(core):
            if char in " \t":
                if run_start < 0:
                    run_start = position
                continue
            if run_start >= 0:
                chars.append(" ")
                index_map.append(offset + run_start)
                run_start = -1
            chars.append(char)
            index_map.append(offset + position)
        if line_number < len(lines) - 1:
            chars.append("\n")
            index_map.append(offset + len(line))
        offset += len(line) + 1
    return "".join(chars), index_map
