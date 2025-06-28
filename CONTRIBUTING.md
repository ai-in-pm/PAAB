# Contributing to PsychoPy AI Agent Builder

Thank you for your interest in contributing to PsychoPy AI Agent Builder (PAAB)! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Development Standards](#development-standards)
8. [Testing](#testing)
9. [Documentation](#documentation)
10. [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [ai@example.com](mailto:ai@example.com).

## Getting Started

### Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes, new features, or improvements
- **Documentation**: Improve or add documentation
- **Examples**: Create examples and tutorials
- **Testing**: Help improve test coverage
- **Community Support**: Help other users in discussions

### Before You Start

1. Check existing [issues](https://github.com/ai-in-pm/psychopy-ai-agent-builder/issues) and [pull requests](https://github.com/ai-in-pm/psychopy-ai-agent-builder/pulls)
2. Read this contributing guide thoroughly
3. Set up your development environment
4. Familiarize yourself with the codebase structure

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Instructions

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/psychopy-ai-agent-builder.git
   cd psychopy-ai-agent-builder
   
   # Add upstream remote
   git remote add upstream https://github.com/ai-in-pm/psychopy-ai-agent-builder.git
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

6. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Development Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing
- **Pre-commit**: Git hooks for quality checks

## Contributing Guidelines

### Branch Strategy

- `main`: Stable release branch
- `develop`: Development branch for next release
- `feature/feature-name`: Feature development
- `bugfix/issue-number`: Bug fixes
- `hotfix/issue-number`: Critical fixes for production

### Workflow

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(agents): add learning capabilities to base agent
fix(runtime): resolve memory leak in executor
docs(readme): update installation instructions
test(crews): add integration tests for crew execution
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**
   - Update docstrings for new/modified functions
   - Update README if needed
   - Add examples for new features

2. **Add Tests**
   - Unit tests for new functions/classes
   - Integration tests for new features
   - Ensure test coverage doesn't decrease

3. **Run Quality Checks**
   ```bash
   # Format code
   black src/ tests/ examples/
   
   # Check linting
   flake8 src/ tests/ examples/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest tests/ -v --cov=src
   ```

### PR Requirements

- [ ] Clear description of changes
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All CI checks pass
- [ ] No merge conflicts
- [ ] Follows coding standards

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, PAAB version
- **Steps to Reproduce**: Clear, minimal steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error messages and stack traces
- **Additional Context**: Screenshots, logs, etc.

### Feature Requests

Use the feature request template and include:

- **Problem Description**: What problem does this solve?
- **Proposed Solution**: Detailed description of the feature
- **Alternatives Considered**: Other solutions you've considered
- **Additional Context**: Use cases, examples, etc.

## Development Standards

### Code Style

- **PEP 8**: Follow Python style guidelines
- **Black**: Use Black for code formatting
- **Line Length**: Maximum 88 characters
- **Imports**: Use isort for import organization

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type hints to all functions
- **Comments**: Explain complex logic and decisions
- **README**: Keep README up to date

### Architecture

- **SOLID Principles**: Follow SOLID design principles
- **Separation of Concerns**: Keep modules focused
- **Dependency Injection**: Use dependency injection where appropriate
- **Error Handling**: Implement proper error handling

### Example Code Style

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExampleAgent:
    """
    Example agent demonstrating code style standards.
    
    Args:
        name: Agent name
        capabilities: List of agent capabilities
        config: Optional configuration dictionary
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.name = name
        self.capabilities = capabilities
        self.config = config or {}
        
        logger.info(f"Initialized agent: {name}")
    
    def execute_task(self, task: "BaseTask") -> Dict[str, Any]:
        """
        Execute a task and return results.
        
        Args:
            task: Task to execute
            
        Returns:
            Dictionary containing execution results
            
        Raises:
            ValueError: If task is invalid
        """
        if not task:
            raise ValueError("Task cannot be None")
        
        try:
            # Task execution logic here
            result = {"success": True, "output": "Task completed"}
            logger.info(f"Task {task.id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            return {"success": False, "error": str(e)}
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test fixtures
└── conftest.py    # Pytest configuration
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from src.agents.base import BaseAgent


class TestBaseAgent:
    """Test suite for BaseAgent class."""
    
    def test_agent_initialization(self):
        """Test agent initialization with valid parameters."""
        agent = BaseAgent(
            role="test_agent",
            goal="Test goal",
            backstory="Test backstory"
        )
        
        assert agent.role == "test_agent"
        assert agent.goal == "Test goal"
        assert agent.backstory == "Test backstory"
    
    def test_agent_initialization_invalid_role(self):
        """Test agent initialization with invalid role."""
        with pytest.raises(ValueError, match="Role cannot be empty"):
            BaseAgent(role="", goal="Test", backstory="Test")
    
    @patch('src.agents.base.some_external_service')
    def test_agent_with_mock(self, mock_service):
        """Test agent behavior with mocked dependencies."""
        mock_service.return_value = "mocked_result"
        
        agent = BaseAgent(role="test", goal="test", backstory="test")
        result = agent.some_method()
        
        assert result == "expected_result"
        mock_service.assert_called_once()
```

### Test Guidelines

- **Test Coverage**: Aim for >90% test coverage
- **Test Isolation**: Tests should be independent
- **Descriptive Names**: Use clear, descriptive test names
- **Arrange-Act-Assert**: Follow AAA pattern
- **Mock External Dependencies**: Mock external services and APIs

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **User Documentation**: Guides and tutorials
3. **API Documentation**: Auto-generated from docstrings
4. **Examples**: Working code examples

### Documentation Standards

- **Clarity**: Write for your audience
- **Completeness**: Cover all features and edge cases
- **Examples**: Include practical examples
- **Updates**: Keep documentation current with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

## Community

### Communication Channels

- **GitHub Discussions**: General discussions and Q&A
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat (link in README)
- **Email**: [ai@example.com](mailto:ai@example.com) for private matters

### Getting Help

1. **Search Existing Issues**: Check if your question has been asked
2. **Read Documentation**: Check the docs first
3. **Ask in Discussions**: Use GitHub Discussions for questions
4. **Join Discord**: Get real-time help from the community

### Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions mentioned
- **GitHub**: Contributor statistics and badges

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release
- [ ] Deploy to PyPI

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing issues and discussions
3. Ask in GitHub Discussions
4. Contact maintainers directly

Thank you for contributing to PsychoPy AI Agent Builder! 🚀
