"""Bounded tool output before it enters the LLM context (HH-01).

Every tool result that can grow without bound (command stdout/stderr, file
reads) passes through :func:`truncate_text` so the model context receives a
deterministic, explicitly marked prefix instead of an unbounded payload.
"""

from __future__ import annotations

DEFAULT_MAX_OUTPUT_CHARS = 20_000

_MARKER_TEMPLATE = "...[truncated {truncated} of {total} chars]"


def _marker(truncated: int, total: int) -> str:
    """Build the truncation marker for the given counts.

    Args:
        truncated: Number of characters removed from the tail of the text.
        total: Original character count before truncation.

    Returns:
        Marker string in the exact ``...[truncated N of M chars]`` format.
    """
    return _MARKER_TEMPLATE.format(truncated=truncated, total=total)


def truncate_text(text: str, max_chars: int, *, label: str) -> str:
    """Bound text to ``max_chars`` characters, appending an explicit marker.

    The first ``max_chars`` characters are kept and the tail is removed; the
    appended marker states how many characters were truncated of how many the
    text originally contained, so the direction of truncation (head kept, tail
    removed) is always visible in the context. This function never raises: any
    input it cannot interpret (``None`` text, non-numeric limits) degrades to
    the closest safe behavior.

    Args:
        text: Text to bound. ``None`` is treated as an empty string.
        max_chars: Maximum number of characters to keep. Values that are zero
            or negative truncate everything (only the marker remains);
            non-numeric values leave the text unchanged.
        label: Short identifier of the truncated stream (for example
            ``"stdout"``, ``"stderr"``, or ``"read_file"``); callers use it for
            diagnostics and audit metadata.

    Returns:
        The original text when it already fits, otherwise the kept head plus a
        ``...[truncated N of M chars]`` marker.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not isinstance(max_chars, (int, float)) or isinstance(max_chars, bool):
        return text
    limit = max(int(max_chars), 0)
    total = len(text)
    if total <= limit:
        return text
    return text[:limit] + _marker(total - limit, total)
