"""
Repository mapping (Aider-inspired).

Creates a concise map of the codebase showing key symbols and their relationships.
Uses graph-ranking algorithm to identify the most important code elements.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_language, get_parser


@dataclass
class Symbol:
    """Represents a code symbol (function, class, method, etc.)."""

    name: str
    type: str  # function, class, method, variable
    file_path: str
    line_number: int
    code_snippet: str
    references: set[str] = None  # Other files that reference this symbol

    def __post_init__(self):
        if self.references is None:
            self.references = set()

    def __hash__(self):
        return hash((self.name, self.file_path, self.line_number))


@dataclass
class FileNode:
    """Represents a file in the dependency graph."""

    path: str
    symbols: list[Symbol]
    dependencies: set[str]  # Files this file depends on
    dependents: set[str]  # Files that depend on this file
    importance: float = 0.0  # PageRank-style importance score

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.dependents is None:
            self.dependents = set()


class RepositoryMap:
    """
    Creates a repository map using graph-ranking.

    Analyzes codebase to identify:
    1. Key symbols (functions, classes, etc.)
    2. File dependencies
    3. Most important/referenced code

    Uses graph-ranking to select the most relevant symbols
    to fit within a token budget.
    """

    def __init__(
        self,
        root_dir: Path,
        max_tokens: int = 1000,
        include_patterns: list[str] | None = None,
    ):
        """
        Initialize repository mapper.

        Args:
            root_dir: Root directory of repository
            max_tokens: Maximum tokens for the map (~1000 is good)
            include_patterns: File patterns to include
        """
        self.root_dir = Path(root_dir)
        self.max_tokens = max_tokens
        self.include_patterns = include_patterns or ["*.py", "*.js", "*.ts", "*.go", "*.java"]

        self.files: dict[str, FileNode] = {}
        self.symbols: list[Symbol] = []
        self.dependency_graph: dict[str, set[str]] = defaultdict(set)

    def build_map(self) -> str:
        """
        Build the repository map.

        Returns:
            Formatted repository map as string
        """
        # Step 1: Parse all files and extract symbols
        self._parse_files()

        # Step 2: Build dependency graph
        self._build_dependency_graph()

        # Step 3: Rank symbols by importance
        self._rank_symbols()

        # Step 4: Format map within token budget
        return self._format_map()

    def _parse_files(self):
        """Parse all files and extract symbols."""
        for file_path in self._get_files():
            try:
                symbols = self._extract_symbols(file_path)
                if symbols:
                    self.symbols.extend(symbols)
                    self.files[str(file_path)] = FileNode(
                        path=str(file_path),
                        symbols=symbols,
                        dependencies=set(),
                        dependents=set(),
                    )
            except Exception:
                # Skip files that can't be parsed
                pass

    def _get_files(self) -> list[Path]:
        """Get all files matching patterns."""
        # Directories to skip
        skip_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            '.pytest_cache', '.mypy_cache', '.tox', 'dist', 'build',
            '.eggs', '*.egg-info', '.uv', 'target', 'bin', 'obj'
        }

        files = []
        max_files = 1000  # Limit to prevent hanging on large repos

        for pattern in self.include_patterns:
            for file_path in self.root_dir.rglob(pattern):
                # Skip if in excluded directory
                if any(part in skip_dirs for part in file_path.parts):
                    continue

                files.append(file_path)

                # Stop if we've collected enough files
                if len(files) >= max_files:
                    return files

        return files

    def _extract_symbols(self, file_path: Path) -> list[Symbol]:
        """
        Extract symbols from a file using tree-sitter.

        Args:
            file_path: Path to file

        Returns:
            List of symbols
        """
        try:
            # Get parser for file type
            language = self._get_language_for_file(file_path)
            if not language:
                return []

            parser = get_parser(language)

            # Read file
            with open(file_path, encoding="utf-8") as f:
                code = f.read()

            # Parse
            tree = parser.parse(bytes(code, "utf8"))

            # Extract symbols
            symbols = []
            self._visit_node(tree.root_node, file_path, code, symbols)

            return symbols

        except Exception:
            return []

    def _get_language_for_file(self, file_path: Path) -> str | None:
        """Get tree-sitter language for file extension."""
        ext = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".rs": "rust",
        }
        return language_map.get(ext)

    def _visit_node(self, node, file_path: Path, code: str, symbols: list[Symbol]):
        """
        Visit tree-sitter node and extract symbols.

        Args:
            node: Tree-sitter node
            file_path: File path
            code: Source code
            symbols: List to append symbols to
        """
        # Check if this node is a symbol we care about
        symbol_types = {
            "function_definition": "function",
            "class_definition": "class",
            "method_definition": "method",
            "function_declaration": "function",
            "class_declaration": "class",
        }

        if node.type in symbol_types:
            # Extract symbol name
            name_node = None
            for child in node.children:
                if child.type in ["identifier", "name"]:
                    name_node = child
                    break

            if name_node:
                name = code[name_node.start_byte:name_node.end_byte]

                # Get code snippet (first few lines)
                start_line = node.start_point[0]
                end_line = min(start_line + 3, node.end_point[0])
                lines = code.split("\n")[start_line:end_line + 1]
                snippet = "\n".join(lines)

                symbol = Symbol(
                    name=name,
                    type=symbol_types[node.type],
                    file_path=str(file_path),
                    line_number=start_line + 1,
                    code_snippet=snippet,
                )
                symbols.append(symbol)

        # Visit children
        for child in node.children:
            self._visit_node(child, file_path, code, symbols)

    def _build_dependency_graph(self):
        """Build dependency graph between files."""
        # For now, simple approach: look for imports
        # In production, use proper import analysis

        for file_path, node in self.files.items():
            # Look for import statements in symbols
            for symbol in node.symbols:
                # This is simplified - in production, parse imports properly
                pass

    def _rank_symbols(self):
        """
        Rank symbols by importance using PageRank-style algorithm.

        More referenced symbols = higher importance
        """
        # Count references to each symbol
        symbol_refs = defaultdict(int)

        for symbol in self.symbols:
            # Count how many times this symbol is referenced
            # This is simplified - in production, do proper reference counting
            symbol_refs[symbol.name] += len(symbol.references)

        # Sort symbols by reference count
        self.symbols.sort(key=lambda s: symbol_refs[s.name], reverse=True)

    def _format_map(self) -> str:
        """
        Format repository map within token budget.

        Returns:
            Formatted map string
        """
        lines = []
        lines.append("# Repository Map\n")

        # Group symbols by file
        file_symbols = defaultdict(list)
        for symbol in self.symbols:
            file_symbols[symbol.file_path].append(symbol)

        # Add symbols file by file
        current_tokens = 0
        token_limit = self.max_tokens

        for file_path in sorted(file_symbols.keys()):
            # Add file header
            rel_path = os.path.relpath(file_path, self.root_dir)
            header = f"\n## {rel_path}\n"

            if current_tokens + len(header.split()) > token_limit:
                break

            lines.append(header)
            current_tokens += len(header.split())

            # Add symbols
            for symbol in file_symbols[file_path][:5]:  # Top 5 per file
                symbol_text = f"- {symbol.type} `{symbol.name}` (line {symbol.line_number})\n"

                if current_tokens + len(symbol_text.split()) > token_limit:
                    break

                lines.append(symbol_text)
                current_tokens += len(symbol_text.split())

        lines.append(f"\n_Map contains {current_tokens} tokens_\n")

        return "".join(lines)


def create_repository_map(
    repo_path: Path,
    max_tokens: int = 1000,
) -> str:
    """
    Create a repository map for a codebase.

    Args:
        repo_path: Path to repository
        max_tokens: Maximum tokens for map

    Returns:
        Repository map as string
    """
    mapper = RepositoryMap(repo_path, max_tokens=max_tokens)
    return mapper.build_map()
