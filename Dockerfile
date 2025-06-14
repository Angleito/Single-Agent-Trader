# Simple AI Trading Bot Dockerfile
# Optimized for macOS with OrbStack

# Build stage
FROM python:3.12-slim AS builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG POETRY_VERSION=1.8.2

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VENV_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
RUN curl -sSL https://install.python-poetry.org | python3 - --version=${POETRY_VERSION}
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies and ensure venv is in project
RUN poetry config virtualenvs.in-project true \
    && poetry install --only=main --no-root \
    && poetry run pip list \
    && ls -la /app/ \
    && find /app -name ".venv" -type d \
    && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.12-slim AS production

# Copy build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    APP_VERSION=${VERSION}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 botuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash botuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=botuser:botuser /app/.venv /app/.venv

# Copy application code
COPY --chown=botuser:botuser bot/ ./bot/
COPY --chown=botuser:botuser pyproject.toml ./
# Create config directory (config files can be mounted or environment-based)
RUN mkdir -p /app/config

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data \
    && chown -R botuser:botuser /app

# Create simple health check script
RUN echo '#!/bin/bash\nset -e\ncd /app\npython -c "import bot; print(\"Bot module OK\")" || exit 1\necho "Health check passed"' > /app/healthcheck.sh \
    && chmod +x /app/healthcheck.sh \
    && chown botuser:botuser /app/healthcheck.sh

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Expose port for health checks
EXPOSE 8080

# Default command - starts in safe dry-run mode
CMD ["python", "-m", "bot.main", "live", "--dry-run"]

# Labels
LABEL org.opencontainers.image.title="AI Trading Bot" \
      org.opencontainers.image.description="Simple AI-powered crypto trading bot for macOS" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="AI Trading Bot" \
      maintainer="ai-trading-bot"