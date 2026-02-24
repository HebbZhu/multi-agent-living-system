# Contributing to MALS

First off, thank you for considering contributing to MALS! It's people like you that make open source such a great community.

## Where to Start

- **Bug Reports**: If you find a bug, please open an issue and provide as much detail as possible, including steps to reproduce, error messages, and your environment.
- **Feature Requests**: Have an idea for a new feature? Open an issue to start a discussion. We'd love to hear your thoughts.
- **Pull Requests**: If you're ready to contribute code, we welcome pull requests.

## Development Setup

1.  **Fork & Clone**: Fork the repository and clone it to your local machine.

    ```bash
    git clone https://github.com/<your-username>/multi-agent-living-space.git
    cd multi-agent-living-space
    ```

2.  **Create a Virtual Environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**: Install the project in editable mode with development dependencies.

    ```bash
    pip install -e ".[dev]"
    ```

4.  **Run Tests**: Make sure all tests pass before you start making changes.

    ```bash
    pytest
    ```

## Coding Style

- We use `ruff` for linting and formatting. Please run `ruff check . --fix` and `ruff format .` before committing.
- We use `mypy` for static type checking. Please run `mypy mals` to check for type errors.
- All code should be well-documented with clear docstrings and comments.

## Pull Request Process

1.  Create a new branch for your feature or bug fix.
2.  Make your changes and add tests for them.
3.  Ensure all tests and style checks pass.
4.  Push your branch and open a pull request.
5.  Provide a clear description of your changes in the pull request.

Thank you for your contribution!
By the way，I’m building a more interesting platform.
Hope you’ll keep following, leave me comments, and add me on WeChat: zhu751938340.
Especially welcome friends working on biomedical & healthcare-related models, agents, and skill development!
