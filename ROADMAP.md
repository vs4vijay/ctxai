# ctxai Roadmap

## ✅ Completed (v0.0.1)

- [x] CLI with Typer framework
- [x] Code traversal with gitignore support
- [x] Tree-sitter based intelligent chunking
- [x] OpenAI embeddings generation
- [x] ChromaDB vector storage
- [x] Index command with progress tracking
- [x] Support for multiple programming languages
- [x] Rich CLI output with progress bars

## 🚧 In Progress

### MCP Server Integration
- [ ] Implement MCP server protocol
- [ ] Search endpoint for semantic queries
- [ ] Integration with GitHub Copilot
- [ ] Support for multiple indexes
- [ ] Query filtering by language/file type

### Search Functionality
- [ ] CLI search command
- [ ] Fuzzy matching
- [ ] Result ranking improvements
- [ ] Context window expansion
- [ ] Search history

## 📋 Planned Features

### Core Features
- [ ] Incremental indexing (update only changed files)
- [ ] Index versioning and migration
- [ ] Multi-repository support
- [ ] Custom embedding models support
- [ ] Local embedding models (no API required)
- [ ] Hybrid search (semantic + keyword)

### CLI Improvements
- [ ] Interactive shell mode (`ctxai shell`)
- [ ] Index management commands (list, delete, info)
- [ ] Configuration file support
- [ ] Verbose/debug logging modes
- [ ] Export/import indexes

### Dashboard
- [ ] Web-based dashboard (`ctxai dashboard`)
- [ ] Visual index browser
- [ ] Search interface
- [ ] Index statistics and analytics
- [ ] Code snippets preview

### Developer Experience
- [ ] Python API for programmatic usage
- [ ] Hooks/plugins system
- [ ] Custom chunking strategies
- [ ] Pre-commit hooks
- [ ] CI/CD integration examples

### Performance
- [ ] Parallel processing
- [ ] Streaming for large files
- [ ] Cache layer for embeddings
- [ ] Compression for storage
- [ ] Memory optimization

### Advanced Features
- [ ] Code relationship graph
- [ ] Duplicate code detection
- [ ] Code quality metrics
- [ ] Security vulnerability scanning
- [ ] License compliance checking
- [ ] Documentation generation

### Integrations
- [ ] VS Code extension
- [ ] GitHub Actions
- [ ] GitLab CI
- [ ] Docker support
- [ ] IDE plugins (PyCharm, IntelliJ)

### AI Enhancements
- [ ] Code explanation generation
- [ ] Automatic tagging
- [ ] Code summarization
- [ ] Related code suggestions
- [ ] Documentation generation

## 🎯 Future Vision

### Agent Framework Integration
- Integration with Autogen
- Pydantic AI support
- Memory integration (mem0)
- Multi-agent workflows

### Advanced Search
- Natural language code generation
- Code refactoring suggestions
- Pattern detection
- API usage examples

### Enterprise Features
- Team collaboration
- Shared indexes
- Access control
- Audit logging
- Custom deployment options

## Contributing

Want to help build these features? Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## Version History

- **v0.0.1** (Current) - Initial release with basic indexing and storage
