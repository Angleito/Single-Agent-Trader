# AI Trading Bot Dockerfile - Ubuntu Optimized
# Multi-stage build optimized for Ubuntu deployment

# Build stage
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG POETRY_VERSION=1.8.2
ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu-optimized package installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    libffi-dev \
    autotools-dev \
    automake \
    libtool \
    pkg-config \
    # Ubuntu-specific optimizations
    software-properties-common \
    apt-transport-https \
    gnupg \
    lsb-release \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

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

# Note: bluefin-v2-client can be installed at runtime if live trading is needed
# For paper trading mode, the bot works without it

# Production stage
FROM --platform=linux/amd64 python:3.11-slim AS production

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

# Ubuntu-optimized runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libffi8 \
    # Ubuntu networking tools
    netcat-openbsd \
    dnsutils \
    iputils-ping \
    # Process monitoring
    procps \
    htop \
    # SSL/TLS support
    openssl \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Create non-root user
RUN groupadd --gid 1000 botuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash botuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=botuser:botuser /app/.venv /app/.venv

# Note: Bluefin dependencies are not installed in the main image due to conflicts
# Use Dockerfile.bluefin for Bluefin-specific builds

# Copy application code
COPY --chown=botuser:botuser bot/ ./bot/
COPY --chown=botuser:botuser pyproject.toml ./

# Create required directories with proper Ubuntu permissions
RUN mkdir -p /app/config /app/logs /app/data /app/prompts /app/tmp \
    && chown -R botuser:botuser /app \
    && chmod 755 /app \
    && chmod 775 /app/logs /app/data /app/tmp \
    && chmod 755 /app/config /app/prompts

# Copy prompt files
COPY --chown=botuser:botuser prompts/*.txt ./prompts/

# Copy health check script
COPY --chown=botuser:botuser healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Copy Docker entrypoint script
COPY --chown=botuser:botuser scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Expose port for health checks and monitoring
EXPOSE 8080

# Set entrypoint to initialization script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command - starts in safe dry-run mode
CMD ["python", "-m", "bot.main", "live", "--dry-run"]

# Ubuntu deployment optimized labels
LABEL org.opencontainers.image.title="AI Trading Bot" \
      org.opencontainers.image.description="Ubuntu optimized crypto trading bot with Coinbase and Bluefin support" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="AI Trading Bot" \
      org.opencontainers.image.exchange="${EXCHANGE_TYPE}" \
      org.opencontainers.image.platform="${TARGETPLATFORM}" \
      ubuntu.optimized="true" \
      ubuntu.compatible="22.04+" \
      maintainer="ai-trading-bot"
