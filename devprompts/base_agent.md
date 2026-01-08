# Base Agent Instructions

All agents in this project should follow these core principles:

- **Pythonic Code**: Write clean, PEP 8 compliant Python code.
- **Docker Centric**: Never assume Python is available locally. All Python execution must be planned and executed via `docker compose`.
- **Modularity**: Design components to be reusable and easy to test.
- **Environment Driven**: Use environment variables for all configurations (database credentials, API keys, etc.).
- **Logging**: Implement structured logging for better observability.
- **Error Handling**: Gracefully handle exceptions and provide informative error messages.
- **Documentation**: All public functions and classes must have docstrings following the Google Style Guide.
- **Dependency Management**: Use `uv` for lightning-fast and reliable dependency management, controlled by `uv.toml` and `pyproject.toml` where appropriate.
