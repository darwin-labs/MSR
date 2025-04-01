# Contributing to the Multi-Step Reasoning (MSR) Framework

Thank you for your interest in contributing to the MSR Framework! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to be respectful, inclusive, and considerate of others.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/MSR.git
   cd MSR
   ```
3. **Set up the development environment**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   
   # Set up environment variables
   cp .env.template .env
   # Edit .env with your API keys
   ```
4. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes**: Implement your feature or bug fix.
2. **Run tests** to ensure your changes don't break existing functionality:
   ```bash
   python -m pytest tests/
   ```
3. **Update documentation** if necessary.
4. **Commit your changes** with descriptive commit messages:
   ```bash
   git commit -m "Add comprehensive feature description"
   ```
5. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a pull request** against the main repository.

## Coding Standards

We follow these coding standards:

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
- Use type hints for function parameters and return values.
- Write docstrings for all public classes, methods, and functions using the following format:
  ```python
  def function_name(param1: type, param2: type) -> return_type:
      """
      Short description of the function.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ExceptionType: When and why this exception is raised
      """
  ```
- Maximum line length is 100 characters.
- Use descriptive variable names.
- Add comments for complex logic.

### Code Organization

- Keep classes and functions focused on a single responsibility.
- Organize related functionality into modules.
- Avoid circular imports.
- Use absolute imports rather than relative imports.

## Submitting Changes

### Commit Messages

- Write clear, concise, and descriptive commit messages.
- Use the present tense ("Add feature" not "Added feature").
- Reference issue numbers if applicable ("Fix #123: Add feature").

### Pull Request Guidelines

- Create a new branch for each feature or bug fix.
- Make sure your code passes all tests.
- Update documentation if needed.
- Add tests for new features.
- Keep pull requests focused on a single change rather than multiple unrelated changes.
- Update the README.md or documentation if necessary.

## Pull Request Process

1. Ensure your code follows the project's coding standards.
2. Update documentation if necessary.
3. Make sure all tests pass successfully.
4. Your pull request will be reviewed by maintainers who may suggest changes.
5. Once approved, your pull request will be merged.

## Testing Guidelines

- Write tests for all new features and bug fixes.
- Ensure existing tests pass before submitting a pull request.
- Use pytest for writing and running tests.
- Aim for high test coverage of your code.
- Include both unit tests and integration tests when appropriate.

Example test:

```python
def test_step_executor_creation():
    """Test that a StepExecutor can be created with default parameters."""
    executor = create_step_executor()
    assert executor.model_name is not None
    assert executor.temperature == 0.7
    assert executor.max_tokens == 2048
```

## Documentation Guidelines

- Keep documentation up-to-date with code changes.
- Write clear, concise, and comprehensive documentation.
- Use Markdown for documentation files.
- Include code examples when relevant.
- Document complex algorithms or workflows with diagrams or step-by-step explanations.

## Issue Reporting

When reporting issues, please include:

1. **Description**: Clear and concise description of the issue.
2. **Steps to reproduce**: Detailed steps to reproduce the issue.
3. **Expected behavior**: What you expected to happen.
4. **Actual behavior**: What actually happened.
5. **Environment**: Python version, operating system, and other relevant details.
6. **Screenshots**: If applicable.
7. **Additional context**: Any other information that might be relevant.

Use the following template:

```
## Description
[Description of the issue]

## Steps to Reproduce
1. [First Step]
2. [Second Step]
3. [and so on...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- Python version: [e.g., 3.9.7]
- OS: [e.g., Ubuntu 20.04]
- MSR version: [e.g., 1.0.0]

## Additional Context
[Any other information, logs, or screenshots]
```

## Feature Requests

When submitting feature requests, please:

1. **Check existing issues** to avoid duplicates.
2. **Provide a clear description** of the proposed feature.
3. **Explain the use case** and benefits of the feature.
4. **Consider implementation details** if you can.

Use the following template:

```
## Feature Description
[Clear description of the feature]

## Use Case
[Explain how this feature would be used and why it's valuable]

## Proposed Implementation
[If you have ideas about how to implement it, share them here]

## Alternatives Considered
[Any alternative solutions or features you've considered]

## Additional Context
[Any other information, mockups, or examples]
```

Thank you for contributing to the MSR Framework! 