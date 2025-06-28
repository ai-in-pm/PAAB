# PsychoPy AI Agent Builder - Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add metadata
LABEL maintainer="AI in PM <ai@example.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="PsychoPy AI Agent Builder" \
      org.label-schema.description="CrewAI-style framework for building AI agent teams" \
      org.label-schema.url="https://github.com/ai-in-pm/psychopy-ai-agent-builder" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/ai-in-pm/psychopy-ai-agent-builder" \
      org.label-schema.vendor="AI in PM" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install build

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY README.md LICENSE ./

# Build the package
RUN python -m build

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PAAB_ENV=production \
    PAAB_LOG_LEVEL=INFO \
    PAAB_DATA_PATH=/app/data \
    PAAB_CACHE_PATH=/app/cache \
    PAAB_LOGS_PATH=/app/logs

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r paab \
    && useradd -r -g paab -d /app -s /bin/bash paab

# Create app directory and required subdirectories
WORKDIR /app
RUN mkdir -p data cache logs config && \
    chown -R paab:paab /app

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Copy configuration files
COPY config/ ./config/
COPY management/ ./management/

# Copy startup scripts
COPY scripts/docker-entrypoint.sh /usr/local/bin/
COPY scripts/healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER paab

# Expose ports
EXPOSE 8080 8501 9090

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["paab", "studio", "--host", "0.0.0.0", "--port", "8501"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy development files
COPY tests/ ./tests/
COPY .env.example ./.env

# Switch back to paab user
USER paab

# Override default command for development
CMD ["python", "-m", "src.cli.main", "studio", "--host", "0.0.0.0", "--port", "8501"]

# Testing stage
FROM development as testing

# Switch to root for test setup
USER root

# Copy test configuration
COPY pytest.ini ./
COPY .coveragerc ./

# Run tests during build (optional)
# RUN python -m pytest tests/ -v --cov=src --cov-report=html

# Switch back to paab user
USER paab

# Command for running tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src"]

# Minimal stage for CI/CD
FROM python:3.11-alpine as minimal

# Install minimal runtime dependencies
RUN apk add --no-cache curl

# Create app user
RUN addgroup -g 1000 paab && \
    adduser -D -s /bin/sh -u 1000 -G paab paab

# Set working directory
WORKDIR /app

# Copy only the built package
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Create required directories
RUN mkdir -p data cache logs && \
    chown -R paab:paab /app

# Switch to non-root user
USER paab

# Minimal health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD python -c "import src; print('OK')" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.cli.main", "--help"]
