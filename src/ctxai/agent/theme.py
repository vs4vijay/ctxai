"""
Neon Blue Theme for ctxai Terminal User Interface.

Provides a sci-fi aesthetic with neon colors, glow effects, and
blinking cursor support for the chat interface.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# ============================================================================
# NEON COLOR PALETTE
# ============================================================================

# Primary colors (hex values for Rich)
NEON_CYAN = "#00D9FF"  # Primary accent - bright neon cyan
NEON_BLUE = "#0096FF"  # Secondary - bright blue
NEON_PURPLE = "#7B68EE"  # Accent - neon purple
NEON_GREEN = "#00FF88"  # Success - neon green
NEON_RED = "#FF3366"  # Error - neon red/pink
NEON_GOLD = "#FFD700"  # Warning - neon gold
NEON_PINK = "#FF69B4"  # Additional accent - hot pink
NEON_WHITE = "#E0E0E0"  # Text - soft white
NEON_DIM = "#808080"  # Dim gray for secondary text

# Background colors
BG_DARK = "#0A0E27"  # Dark navy background
BG_MID = "#141833"  # Slightly lighter navy
BG_LIGHT = "#1E2442"  # Light navy for borders

# Glow effect colors (brighter versions for emphasis)
GLOW_PRIMARY = "#00FFFF"  # Full cyan glow
GLOW_SUCCESS = "#00FFAA"  # Green glow


# ============================================================================
# RICH STYLE DEFINITIONS
# ============================================================================


@dataclass
class NeonStyles:
    """Collection of Rich styles for neon theme."""

    # Core text styles
    primary: str = f"bold {NEON_CYAN}"
    secondary: str = f"bold {NEON_BLUE}"
    accent: str = f"bold {NEON_PURPLE}"
    text: str = NEON_WHITE
    text_dim: str = f"dim {NEON_WHITE}"

    # Semantic styles
    success: str = f"bold {NEON_GREEN}"
    error: str = f"bold {NEON_RED}"
    warning: str = f"bold {NEON_GOLD}"
    info: str = f"bold {NEON_BLUE}"

    # Glow effects
    glow_primary: str = f"bold {GLOW_PRIMARY}"
    glow_success: str = f"bold {GLOW_SUCCESS}"

    # Panel borders
    border_primary: str = NEON_CYAN
    border_success: str = NEON_GREEN
    border_error: str = NEON_RED
    border_warning: str = NEON_GOLD
    border_info: str = NEON_BLUE

    # User/Actor styles
    user: str = f"bold {NEON_CYAN}"
    agent: str = f"bold {NEON_PURPLE}"
    system: str = f"dim {NEON_BLUE}"


# Global styles instance
STYLES = NeonStyles()


# ============================================================================
# CURSOR STYLES FOR PROMPT_TOOLKIT
# ============================================================================


class NeonCursor:
    """
    Cursor styles for prompt_toolkit with blinking neon blue cursor.
    """

    # Beam cursor (vertical line) - classic terminal feel
    BEAM = "class:neon.cursor.beam"

    # Block cursor
    BLOCK = "class:neon.cursor.block"

    # Underscore cursor
    UNDERSCORE = "class:neon.cursor.underscore"

    @staticmethod
    def get_style_string() -> str:
        """Get CSS style string for neon cursor."""
        return f"""
        .neon.cursor.beam {{
            color: {NEON_CYAN};
            background-color: transparent;
        }}
        .neon.cursor.block {{
            color: {BG_DARK};
            background-color: {NEON_CYAN};
        }}
        .neon.cursor.underscore {{
            color: {NEON_CYAN};
            text-decoration: overline;
        }}
        """

    @staticmethod
    def html_prompt(style: str = "beam") -> str:
        """
        Get HTML formatted prompt with neon cursor styling.

        Args:
            style: Cursor style - "beam", "block", or "underscore"

        Returns:
            HTML string with styled prompt
        """
        cursor_char = {"beam": "▌", "block": "█", "underscore": "▁"}.get(style, "▌")

        return f"""
<cyan><b>You</b></cyan>: <cyan>{cursor_char}</cyan>
"""


# ============================================================================
# NEON CONSOLE CLASS
# ============================================================================


class NeonConsole:
    """
    Console wrapper with neon theme styling.

    Provides convenience methods for styled output consistent
    with the sci-fi neon blue aesthetic.
    """

    def __init__(self, console: Console | None = None):
        """
        Initialize NeonConsole.

        Args:
            console: Existing Rich Console instance (creates new if None)
        """
        if console is None:
            # Create console with UTF-8 encoding for cross-platform Unicode support
            import sys

            if sys.platform == "win32":
                # On Windows, force UTF-8 mode
                console = Console(legacy_windows=False, force_terminal=True)
            else:
                console = Console(legacy_windows=False)
        self._console = console

    @property
    def console(self) -> Console:
        """Get underlying Rich Console."""
        return self._console

    def print(self, *args, **kwargs):
        """Print with the underlying console."""
        self._console.print(*args, **kwargs)

    def print_neon(self, text: str, style: str = "primary"):
        """
        Print text with neon styling.

        Args:
            text: Text to print
            style: Style name from NeonStyles
        """
        style_attr = getattr(STYLES, style, STYLES.primary)
        self._console.print(text, style=style_attr)

    def print_banner(self, lines: list[str], centered: bool = True):
        """
        Print a neon banner with glow effect.

        Args:
            lines: Lines of text to print as banner
            centered: Whether to center the text
        """
        for line in lines:
            self._console.print(
                f"[{STYLES.glow_primary}]{line}[/{STYLES.glow_primary}]", justify="center" if centered else "left"
            )

    def print_success(self, text: str):
        """Print success message in neon green."""
        self._console.print(f"[{STYLES.success}]✓ {text}[/{STYLES.success}]")

    def print_error(self, text: str):
        """Print error message in neon red."""
        self._console.print(f"[{STYLES.error}]✗ {text}[/{STYLES.error}]")

    def print_warning(self, text: str):
        """Print warning message in neon gold."""
        self._console.print(f"[{STYLES.warning}]{text}[/{STYLES.warning}]")

    def print_info(self, text: str):
        """Print info message in neon blue."""
        self._console.print(f"[{STYLES.info}]ℹ {text}[/{STYLES.info}]")

    def print_dim(self, text: str):
        """Print dimmed text."""
        self._console.print(f"[{STYLES.text_dim}]{text}[/{STYLES.text_dim}]")

    def print_panel(self, content: str | Markdown, title: str | None = None, border_style: str = "primary", **kwargs):
        """
        Print content in a neon-styled panel.

        Args:
            content: Panel content (can be str or Markdown)
            title: Panel title
            border_style: Border color style name
            **kwargs: Additional Panel arguments
        """
        border = getattr(STYLES, f"border_{border_style}", STYLES.border_primary)
        self._console.print(Panel(content, title=title, border_style=border, **kwargs))

    def print_exception(self, **kwargs):
        """Print exception with traceback."""
        self._console.print_exception(**kwargs)

    def print_table(self, data: list[dict], columns: list[str], title: str | None = None):
        """
        Print data as a neon-styled table.

        Args:
            data: List of row dictionaries
            columns: Column names
            title: Table title
        """
        table = Table(
            title=title,
            border_style=STYLES.border_primary,
            header_style=f"bold {NEON_CYAN}",
            row_styles=[NEON_WHITE, BG_MID],
        )

        for col in columns:
            table.add_column(col, style=NEON_WHITE)

        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])

        self._console.print(table)

    def status(self, text: str, spinner: str = "dots"):
        """
        Create a status spinner context.

        Args:
            text: Status text
            spinner: Spinner style

        Returns:
            Status context manager
        """
        return self._console.status(
            f"[{STYLES.text_dim}]{text}[/{STYLES.text_dim}]", spinner=spinner, spinner_style=STYLES.primary
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def neon_text(text: str, style: str = "primary") -> str:
    """
    Format text with neon styling.

    Args:
        text: Text to format
        style: Style name from NeonStyles

    Returns:
        Formatted string ready for Rich
    """
    style_attr = getattr(STYLES, style, STYLES.primary)
    return f"[{style_attr}]{text}[/{style_attr}]"


def neon_panel(content: str, title: str | None = None, border: str = "primary", **kwargs) -> Panel:
    """
    Create a neon-styled panel.

    Args:
        content: Panel content
        title: Panel title
        border: Border color style
        **kwargs: Additional Panel arguments

    Returns:
        Rich Panel with neon styling
    """
    border_style = getattr(STYLES, f"border_{border}", STYLES.border_primary)
    return Panel(content, title=title, border_style=border_style, **kwargs)


def neon_divider(text: str | None = None) -> str:
    """
    Create a neon divider line.

    Args:
        text: Optional text to center in divider

    Returns:
        Formatted divider string
    """
    if text:
        return f"[{STYLES.border_primary}]{'─' * 3} {text} {'─' * 3}[/{STYLES.border_primary}]"
    return f"[{STYLES.border_primary}]{'─' * 60}[/{STYLES.border_primary}]"


def create_prompt_text() -> str:
    """
    Create text prompt for terminal with neon styling.
    Uses Rich formatting with blink effect.

    Returns:
        Formatted string for terminal prompt
    """
    # Use Rich formatting for colors and blink
    return "[bold #00D9FF]You: [blink]▌[/blink][/bold] "


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "NEON_CYAN",
    "NEON_BLUE",
    "NEON_PURPLE",
    "NEON_GREEN",
    "NEON_RED",
    "NEON_GOLD",
    "NEON_PINK",
    "NEON_WHITE",
    "BG_DARK",
    "BG_MID",
    "BG_LIGHT",
    # Classes
    "NeonStyles",
    "NeonConsole",
    "NeonCursor",
    # Instances
    "STYLES",
    # Functions
    "neon_text",
    "neon_panel",
    "neon_divider",
    "create_prompt_text",
]
