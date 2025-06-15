# AI Trading Bot Dockerfile with Multi-Exchange Support
# Optimized for VPS deployment with Bluefin integration

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
    git \
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

# Install main dependencies with Poetry
RUN poetry config virtualenvs.in-project true \
    && poetry install --only=main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.12-slim AS production

# Copy build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG EXCHANGE_TYPE=coinbase

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    APP_VERSION=${VERSION} \
    EXCHANGE__EXCHANGE_TYPE=${EXCHANGE_TYPE}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 botuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash botuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=botuser:botuser /app/.venv /app/.venv

# Install exchange-specific dependencies
# This handles Bluefin's conflicting dependencies by installing them separately
RUN if [ "${EXCHANGE_TYPE}" = "bluefin" ]; then \
        echo "Installing Bluefin dependencies..." && \
        /app/.venv/bin/pip install --no-cache-dir \
            bluefin-v2-client==3.2.4 \
            aiohttp==3.8.6 \
            websocket-client==1.6.4; \
    fi

# Copy application code
COPY --chown=botuser:botuser bot/ ./bot/
COPY --chown=botuser:botuser pyproject.toml ./
COPY --chown=botuser:botuser scripts/ ./scripts/
COPY --chown=botuser:botuser docs/ ./docs/

# Create required directories
RUN mkdir -p /app/config /app/logs /app/data /app/prompts \
    && chown -R botuser:botuser /app

# Copy prompt files if they exist
COPY --chown=botuser:botuser prompts/*.md ./prompts/ || true

# Copy health check script
COPY --chown=botuser:botuser healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Expose port for health checks and monitoring
EXPOSE 8080

# Default command - starts in safe dry-run mode
CMD ["python", "-m", "bot.main", "live", "--dry-run"]

# Labels
LABEL org.opencontainers.image.title="AI Trading Bot" \
      org.opencontainers.image.description="Multi-exchange crypto trading bot with Coinbase and Bluefin support" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="AI Trading Bot" \
      org.opencontainers.image.exchange="${EXCHANGE_TYPE}" \
      maintainer="ai-trading-bot"