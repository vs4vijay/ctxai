# Contributing to ctxai

Thank you for your interest in contributing to ctxai! We welcome contributions from everyone.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/ctxai.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and linters
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ctxai

# Run specific test
pytest tests/test_indexing.py
```

## Code Quality

Before submitting a PR, please ensure:

```bash
# Linting passes
ruff check src/

# Code is formatted
ruff format src/

# Tests pass
pytest
```

## Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for public functions and classes
- Keep functions focused and small
- Add tests for new features

## Pull Request Guidelines

- Include a clear description of the changes
- Reference any related issues
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Keep commits clean and focused

## Reporting Bugs

When reporting bugs, please include:

- Python version
- ctxai version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages or logs

## Feature Requests

We love feature requests! Please:

- Check if the feature already exists or is planned
- Clearly describe the use case
- Explain why it would be useful
- Provide examples if possible

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Be patient with questions
- Focus on constructive feedback
- Respect different viewpoints

## Questions?

Feel free to open an issue for questions or join discussions in the repository.

Thank you for contributing! 🎉
