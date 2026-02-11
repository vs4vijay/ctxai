"""
Git tools for the agent.

Provides git operations like status, diff, commit, branch, etc.
"""

import subprocess
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolParameter, ToolParameterType, ToolSchema


class GitStatusTool(BaseTool):
    """Get git repository status."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Get the current git repository status, including modified, staged, and untracked files.",
            parameters=[
                ToolParameter(
                    name="path",
                    type=ToolParameterType.STRING,
                    description="Path to the git repository (default: current directory)",
                    required=False,
                ),
            ],
        )

    async def execute(self, path: str = ".") -> dict[str, Any]:
        """
        Execute git status.

        Args:
            path: Repository path

        Returns:
            Dict with success, result (status output), and metadata
        """
        try:
            repo_path = Path(path).resolve()

            # Run git status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Git status failed: {result.stderr}",
                }

            # Parse status output
            status_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

# Count file types
            modified = len([line for line in status_lines if line.startswith(" M")])
            added = len([line for line in status_lines if line.startswith("A ")])
            deleted = len([line for line in status_lines if line.startswith(" D")])
            untracked = len([line for line in status_lines if line.startswith("??")])
            staged = len([line for line in status_lines if line[0] in "MADRC"])

            # Get branch info
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

            return {
                "success": True,
                "result": result.stdout or "Working tree clean",
                "metadata": {
                    "branch": current_branch,
                    "modified": modified,
                    "added": added,
                    "deleted": deleted,
                    "untracked": untracked,
                    "staged": staged,
                    "clean": len(status_lines) == 0,
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": "Git status timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Git status error: {str(e)}",
            }


class GitDiffTool(BaseTool):
    """Show git diff for files."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Show git diff for modified files. Can show staged, unstaged, or both changes.",
            parameters=[
                ToolParameter(
                    name="path",
                    type=ToolParameterType.STRING,
                    description="Path to file or directory (default: all files)",
                    required=False,
                ),
                ToolParameter(
                    name="staged",
                    type=ToolParameterType.BOOLEAN,
                    description="Show only staged changes (default: false)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="repo_path",
                    type=ToolParameterType.STRING,
                    description="Path to repository (default: current directory)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        path: str = "",
        staged: bool = False,
        repo_path: str = ".",
    ) -> dict[str, Any]:
        """
        Execute git diff.

        Args:
            path: File/directory path
            staged: Show staged changes only
            repo_path: Repository path

        Returns:
            Dict with success, result (diff output), and metadata
        """
        try:
            repo_dir = Path(repo_path).resolve()

            # Build command
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--staged")

            if path:
                cmd.append(path)

            # Run git diff
            result = subprocess.run(
                cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Git diff failed: {result.stderr}",
                }

            diff_output = result.stdout

# Count changes
            lines = diff_output.split("\n")
            additions = len([line for line in lines if line.startswith("+")])
            deletions = len([line for line in lines if line.startswith("-")])

            return {
                "success": True,
                "result": diff_output or "No differences",
                "metadata": {
                    "additions": additions,
                    "deletions": deletions,
                    "has_changes": bool(diff_output.strip()),
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": "Git diff timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Git diff error: {str(e)}",
            }


class GitCommitTool(BaseTool):
    """Create a git commit."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Create a git commit with a message. Only commits staged files. "
                "Use git_add tool first to stage files."
            ),
            parameters=[
                ToolParameter(
                    name="message",
                    type=ToolParameterType.STRING,
                    description="Commit message",
                    required=True,
                ),
                ToolParameter(
                    name="repo_path",
                    type=ToolParameterType.STRING,
                    description="Path to repository (default: current directory)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        message: str,
        repo_path: str = ".",
    ) -> dict[str, Any]:
        """
        Execute git commit.

        Args:
            message: Commit message
            repo_path: Repository path

        Returns:
            Dict with success, result, and metadata
        """
        try:
            repo_dir = Path(repo_path).resolve()

            # Run git commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Git commit failed: {result.stderr}",
                }

            return {
                "success": True,
                "result": result.stdout,
                "metadata": {
                    "message": message,
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": "Git commit timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Git commit error: {str(e)}",
            }


class GitAddTool(BaseTool):
    """Stage files for commit."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Stage files for git commit. Can stage specific files or all changes.",
            parameters=[
                ToolParameter(
                    name="files",
                    type=ToolParameterType.ARRAY,
                    description="List of file paths to stage (use ['.'] for all files)",
                    required=True,
                    items=ToolParameter(
                        name="file",
                        type=ToolParameterType.STRING,
                        description="File path",
                        required=True,
                    ),
                ),
                ToolParameter(
                    name="repo_path",
                    type=ToolParameterType.STRING,
                    description="Path to repository (default: current directory)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        files: list,
        repo_path: str = ".",
    ) -> dict[str, Any]:
        """
        Execute git add.

        Args:
            files: List of file paths
            repo_path: Repository path

        Returns:
            Dict with success, result, and metadata
        """
        try:
            repo_dir = Path(repo_path).resolve()

            # Run git add
            cmd = ["git", "add"] + files
            result = subprocess.run(
                cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Git add failed: {result.stderr}",
                }

            return {
                "success": True,
                "result": f"Staged {len(files)} file(s)",
                "metadata": {
                    "files": files,
                    "count": len(files),
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": "Git add timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Git add error: {str(e)}",
            }


class GitBranchTool(BaseTool):
    """List, create, or switch git branches."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="List all branches, create a new branch, or get current branch information.",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ToolParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=["list", "current", "create"],
                ),
                ToolParameter(
                    name="branch_name",
                    type=ToolParameterType.STRING,
                    description="Branch name (required for 'create' action)",
                    required=False,
                ),
                ToolParameter(
                    name="repo_path",
                    type=ToolParameterType.STRING,
                    description="Path to repository (default: current directory)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        action: str,
        branch_name: str = None,
        repo_path: str = ".",
    ) -> dict[str, Any]:
        """
        Execute git branch operation.

        Args:
            action: Action to perform (list, current, create)
            branch_name: Branch name (for create)
            repo_path: Repository path

        Returns:
            Dict with success, result, and metadata
        """
        try:
            repo_dir = Path(repo_path).resolve()

            if action == "list":
                result = subprocess.run(
                    ["git", "branch", "-a"],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            elif action == "current":
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

            elif action == "create":
                if not branch_name:
                    return {
                        "success": False,
                        "result": None,
                        "error": "branch_name is required for 'create' action",
                    }

                result = subprocess.run(
                    ["git", "branch", branch_name],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown action: {action}",
                }

            if result.returncode != 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Git branch failed: {result.stderr}",
                }

            return {
                "success": True,
                "result": result.stdout.strip(),
                "metadata": {
                    "action": action,
                    "branch_name": branch_name,
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": "Git branch timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Git branch error: {str(e)}",
            }


class GitLogTool(BaseTool):
    """View git commit history."""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="View git commit history with optional limit.",
            parameters=[
                ToolParameter(
                    name="limit",
                    type=ToolParameterType.INTEGER,
                    description="Number of commits to show (default: 10)",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="repo_path",
                    type=ToolParameterType.STRING,
                    description="Path to repository (default: current directory)",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        limit: int = 10,
        repo_path: str = ".",
    ) -> dict[str, Any]:
        """
        Execute git log.

        Args:
            limit: Number of commits
            repo_path: Repository path

        Returns:
            Dict with success, result, and metadata
        """
        try:
            repo_dir = Path(repo_path).resolve()

            # Run git log
            result = subprocess.run(
                ["git", "log", f"-{limit}", "--oneline", "--decorate"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Git log failed: {result.stderr}",
                }

            commits = result.stdout.strip().split("\n") if result.stdout.strip() else []

            return {
                "success": True,
                "result": result.stdout,
                "metadata": {
                    "commit_count": len(commits),
                    "limit": limit,
                },
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "result": None,
                "error": "Git log timed out",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Git log error: {str(e)}",
            }
