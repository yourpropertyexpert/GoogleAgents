# Use a modern Python base image
FROM python:3.14-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy configuration files
COPY uv.toml pyproject.toml ./

# Install google-adk
RUN uv pip install --system google-adk

# Verify installation during build
RUN python -c "import google.adk; print('Google ADK imported successfully')"

# We don't copy the source code here as per requirements.
# It will be mounted via docker-compose for development.

# Command to run the application
CMD ["python", "src/frontend/main.py"]
