# AGENTS.md - Development Guide

This document specifies how agents (human and AI) should build and maintain this project.

## Core Development Philosophy

- Follow **PEP 8** for all Python code.
- Use **uv** for all dependency management.
- **DOCKER ONLY**: Do not install Python on the local machine. All Python commands (including `flake8`, `pytest`, etc.) **MUST** be run within Docker containers.

## Coding Standards

To pass the CI pipeline (`.woodpecker/01-expected-to-pass.yml`), all code must adhere to the following:

- **Style**: 120 character line limit. Ignore E203, W503.
- **Complexity**: All functions must have a cyclomatic complexity of **B** or lower (as measured by `radon`).
- **Duplicates**: Maintain low code duplication. PMD CPD token limit is 150 for frontend and 250 for the whole project.
- **Dependencies**: No `streamlit` allowed in the core client/frontend logic.
- **Documentation**: Google Style docstrings are required for all public interfaces.

To start the development environment:

```bash
docker compose up --build
```

To run a one-off command (e.g., linting):

```bash
docker compose run --rm frontend python -m flake8 .
```

## Agent Instructions

When building new features or agents, you **MUST** refer to and follow the instructions in the `/devprompts` directory:

- [base_agent.md](file:///Users/markharrison/Desktop/github.nosync/GoogleAgents/devprompts/base_agent.md): General principles for all development.
- [frontend_agent.md](file:///Users/markharrison/Desktop/github.nosync/GoogleAgents/devprompts/frontend_agent.md): Specific rules for building the frontend agent.

## Project Structure

- `docker/`: Dockerfiles and related configuration.
- `src/`: Source code for the application.
- `devprompts/`: Specific instruction sets for development agents.
- `AGENTS.md`: This file.
