"""
Code chunking module using tree-sitter for intelligent code splitting.
Preserves semantic meaning by respecting code structure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tree_sitter_languages


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""

    content: str
    file_path: Path
    start_line: int
    end_line: int
    chunk_type: str  # e.g., "function", "class", "module", "comment"
    language: str
    metadata: Dict[str, str]  # Additional context like function name, class name, etc.


class CodeChunker:
    """Intelligent code chunker using tree-sitter for semantic parsing."""

    # Language detection by file extension
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "c_sharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".m": "objective_c",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".md": "markdown",
        ".rst": "rst",
    }

    # Node types that represent meaningful code units
    CHUNK_NODE_TYPES = {
        "python": [
            "function_definition",
            "class_definition",
            "decorated_definition",
            "import_statement",
            "import_from_statement",
        ],
        "javascript": [
            "function_declaration",
            "function_expression",
            "arrow_function",
            "class_declaration",
            "method_definition",
            "import_statement",
            "export_statement",
        ],
        "typescript": [
            "function_declaration",
            "function_expression",
            "arrow_function",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
            "import_statement",
            "export_statement",
        ],
    }

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize the code chunker.

        Args:
            max_chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.parsers: Dict[str, any] = {}

    def _get_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file extension."""
        return self.LANGUAGE_MAP.get(file_path.suffix.lower())

    def _get_parser(self, language: str):
        """Get or create a tree-sitter parser for the given language."""
        if language not in self.parsers:
            try:
                self.parsers[language] = tree_sitter_languages.get_parser(language)
            except Exception as e:
                print(f"Warning: Could not get parser for {language}: {e}")
                return None
        return self.parsers.get(language)

    def _extract_node_text(self, node, source_code: bytes) -> str:
        """Extract text from a tree-sitter node."""
        return source_code[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _get_node_metadata(self, node, source_code: bytes, language: str) -> Dict[str, str]:
        """Extract metadata from a node based on its type."""
        metadata = {"node_type": node.type}

        # Extract function/class names
        if language == "python":
            if node.type in ["function_definition", "class_definition"]:
                for child in node.children:
                    if child.type == "identifier":
                        name = self._extract_node_text(child, source_code)
                        metadata["name"] = name
                        break
        elif language in ["javascript", "typescript"]:
            if "declaration" in node.type or "definition" in node.type:
                for child in node.children:
                    if child.type == "identifier":
                        name = self._extract_node_text(child, source_code)
                        metadata["name"] = name
                        break

        return metadata

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Chunk a file into semantically meaningful pieces.

        Args:
            file_path: Path to the file to chunk

        Returns:
            List of CodeChunk objects
        """
        language = self._get_language(file_path)
        if not language:
            # Fall back to simple text chunking for unknown file types
            return self._chunk_text_file(file_path)

        parser = self._get_parser(language)
        if not parser:
            return self._chunk_text_file(file_path)

        try:
            with open(file_path, "rb") as f:
                source_code = f.read()

            tree = parser.parse(source_code)
            chunks = []

            # Get relevant node types for this language
            chunk_node_types = self.CHUNK_NODE_TYPES.get(language, [])

            # Traverse the tree and extract chunks
            chunks.extend(
                self._extract_chunks_from_tree(
                    tree.root_node, source_code, file_path, language, chunk_node_types
                )
            )

            # If no chunks were extracted (e.g., simple script), chunk the whole file
            if not chunks:
                chunks = self._chunk_text_file(file_path, language)

            return chunks

        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}")
            return self._chunk_text_file(file_path)

    def _extract_chunks_from_tree(
        self,
        node,
        source_code: bytes,
        file_path: Path,
        language: str,
        chunk_node_types: List[str],
    ) -> List[CodeChunk]:
        """Recursively extract chunks from a tree-sitter tree."""
        chunks = []

        # Check if this node is a chunk boundary
        if node.type in chunk_node_types:
            content = self._extract_node_text(node, source_code)
            metadata = self._get_node_metadata(node, source_code, language)

            # Split large nodes if needed
            if len(content) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(
                    content, file_path, node.start_point[0], node.type, language, metadata
                )
                chunks.extend(sub_chunks)
            else:
                chunk = CodeChunk(
                    content=content,
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    chunk_type=node.type,
                    language=language,
                    metadata=metadata,
                )
                chunks.append(chunk)
        else:
            # Recurse into children
            for child in node.children:
                chunks.extend(
                    self._extract_chunks_from_tree(
                        child, source_code, file_path, language, chunk_node_types
                    )
                )

        return chunks

    def _split_large_chunk(
        self,
        content: str,
        file_path: Path,
        start_line: int,
        chunk_type: str,
        language: str,
        metadata: Dict[str, str],
    ) -> List[CodeChunk]:
        """Split a large chunk into smaller overlapping chunks."""
        chunks = []
        lines = content.split("\n")
        current_chunk_lines = []
        current_size = 0
        current_start_line = start_line

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            current_size += len(line) + 1  # +1 for newline

            if current_size >= self.max_chunk_size:
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=current_start_line + 1,
                        end_line=current_start_line + len(current_chunk_lines),
                        chunk_type=chunk_type,
                        language=language,
                        metadata=metadata,
                    )
                )

                # Calculate overlap
                overlap_lines = []
                overlap_size = 0
                for line in reversed(current_chunk_lines):
                    if overlap_size + len(line) > self.overlap:
                        break
                    overlap_lines.insert(0, line)
                    overlap_size += len(line) + 1

                current_chunk_lines = overlap_lines
                current_start_line = current_start_line + len(current_chunk_lines) - len(
                    overlap_lines
                )
                current_size = overlap_size

        # Add remaining lines as final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start_line + 1,
                    end_line=current_start_line + len(current_chunk_lines),
                    chunk_type=chunk_type,
                    language=language,
                    metadata=metadata,
                )
            )

        return chunks

    def _chunk_text_file(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Simple text-based chunking for files without tree-sitter support.

        Args:
            file_path: Path to the file
            language: Optional language identifier

        Returns:
            List of CodeChunk objects
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            chunks = []
            lines = content.split("\n")
            current_chunk_lines = []
            current_size = 0
            start_line = 0

            for i, line in enumerate(lines):
                current_chunk_lines.append(line)
                current_size += len(line) + 1

                if current_size >= self.max_chunk_size:
                    chunk_content = "\n".join(current_chunk_lines)
                    chunks.append(
                        CodeChunk(
                            content=chunk_content,
                            file_path=file_path,
                            start_line=start_line + 1,
                            end_line=start_line + len(current_chunk_lines),
                            chunk_type="text",
                            language=language or "unknown",
                            metadata={},
                        )
                    )

                    # Handle overlap
                    overlap_lines = []
                    overlap_size = 0
                    for line in reversed(current_chunk_lines):
                        if overlap_size + len(line) > self.overlap:
                            break
                        overlap_lines.insert(0, line)
                        overlap_size += len(line) + 1

                    current_chunk_lines = overlap_lines
                    start_line = i - len(overlap_lines) + 1
                    current_size = overlap_size

            # Add remaining lines
            if current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line + 1,
                        end_line=start_line + len(current_chunk_lines),
                        chunk_type="text",
                        language=language or "unknown",
                        metadata={},
                    )
                )

            return chunks

        except Exception as e:
            print(f"Warning: Could not chunk text file {file_path}: {e}")
            return []
