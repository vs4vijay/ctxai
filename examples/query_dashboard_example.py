"""
Example usage of ctxai query and dashboard commands.

This example demonstrates:
1. How to query an indexed codebase from Python
2. How to start the dashboard programmatically
"""

from pathlib import Path

from ctxai.commands.query_command import query_codebase
from ctxai.commands.dashboard_command import start_dashboard


def example_query():
    """Example: Query an indexed codebase."""
    print("=" * 80)
    print("EXAMPLE: Query Command")
    print("=" * 80)

    # Query the index
    query_codebase(
        index_name="my-project",
        query="Find functions that handle user authentication",
        n_results=3,
        show_content=True,
    )


def example_dashboard():
    """Example: Start the web dashboard."""
    print("=" * 80)
    print("EXAMPLE: Dashboard")
    print("=" * 80)
    print()
    print("Starting dashboard on port 3000...")
    print("Open your browser to: http://localhost:3000")
    print()

    # Start dashboard
    start_dashboard(port=3000)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python query_dashboard_example.py [query|dashboard]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "query":
        example_query()
    elif command == "dashboard":
        example_dashboard()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: query, dashboard")
        sys.exit(1)
