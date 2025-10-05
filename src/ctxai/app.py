import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="ctxai - Semantic code search engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index a codebase for semantic search"
    )
    index_parser.add_argument(
        "path", help="Path to the codebase to index"
    )
    index_parser.add_argument(
        "name", help="Name for the index"
    )
    
    # Server command
    server_parser = subparsers.add_parser(
        "server", help="Start the MCP server"
    )
    server_parser.add_argument(
        "--index", required=True, help="Index name to use"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    
    # Playground command
    playground_parser = subparsers.add_parser(
        "playground", help="Start the interactive playground"
    )
    
    # Shell command
    shell_parser = subparsers.add_parser(
        "shell", help="Start an interactive shell"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "index":
        from .index import run_index
        run_index(args.path, args.name)
    elif args.command == "server":
        from .server import run_server
        run_server(args.index, args.port)
    elif args.command == "playground":
        from .playground import run_playground
        run_playground()
    elif args.command == "shell":
        from .shell import run_shell
        run_shell()


if __name__ == "__main__":
    main()