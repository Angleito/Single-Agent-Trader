# AI Trading Bot Dockerfile - Ubuntu Optimized
# Multi-stage build optimized for Ubuntu deployment

# Build stage
FROM --platform=linux/amd64 python:3.12-slim AS builder

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

# Install main dependencies with Poetry (regenerate lock file if needed for VPS deployment)
RUN poetry config virtualenvs.in-project true \
    && (poetry install --only=main --no-root || \
        (echo "Lock file out of sync, regenerating..." && \
         poetry lock --no-update && \
         poetry install --only=main --no-root)) \
    && rm -rf $POETRY_CACHE_DIR

# Note: bluefin-v2-client can be installed at runtime if live trading is needed
# For paper trading mode, the bot works without it

# Production stage
FROM --platform=linux/amd64 python:3.12-slim AS production

# Copy build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG EXCHANGE_TYPE=coinbase
ARG USER_ID=1000
ARG GROUP_ID=1000

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

# Create non-root user with dynamic UID/GID - Handle UID conflicts gracefully
# Skip user creation if USER_ID is 0 (root) to avoid conflicts
RUN if [ "${USER_ID}" != "0" ]; then \
        # Create group if it doesn't already exist
        (groupadd --gid ${GROUP_ID} botuser || true) && \
        # Create user if it doesn't already exist
        (useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash botuser || \
         echo "User with UID ${USER_ID} already exists, skipping creation"); \
    else \
        echo "Using root user (UID=0) - skipping user creation"; \
    fi

# Set work directory
WORKDIR /app

# Copy virtual environment from builder with conditional ownership
COPY --from=builder /app/.venv /app/.venv

# Note: Bluefin dependencies are not installed in the main image due to conflicts
# Use Dockerfile.bluefin for Bluefin-specific builds

# Copy application code
COPY bot/ ./bot/
COPY pyproject.toml ./

# Create required directories with proper Ubuntu permissions
# This comprehensive directory setup reduces the burden on the entrypoint script
# and ensures proper permissions for the botuser (1000:1000)
RUN echo "Creating application directories..." \
    # Main application directories
    && mkdir -p /app/config /app/logs /app/data /app/prompts /app/tmp \
    # MCP and memory system directories
    && mkdir -p /app/data/mcp_memory /app/logs/mcp \
    # Exchange-specific logging directories
    && mkdir -p /app/logs/bluefin /app/logs/trades \
    # Trading data directories
    && mkdir -p /app/data/orders /app/data/paper_trading /app/data/positions /app/data/bluefin /app/data/omnisearch_cache \
    # Set ownership conditionally - only if not running as root
    && echo "Setting directory ownership to (${USER_ID}:${GROUP_ID})..." \
    && if [ "${USER_ID}" != "0" ]; then \
        chown -R ${USER_ID}:${GROUP_ID} /app; \
    else \
        echo "Running as root - skipping ownership change"; \
    fi \
    # Set base directory permissions (755 - read/execute for all, write for owner)
    && echo "Setting directory permissions..." \
    && chmod 755 /app \
    # Set writable directory permissions (775 - read/write/execute for owner and group)
    && chmod 775 /app/logs /app/logs/mcp /app/logs/bluefin /app/logs/trades \
    && chmod 775 /app/data /app/data/mcp_memory /app/data/orders /app/data/paper_trading /app/data/positions /app/data/bluefin /app/data/omnisearch_cache \
    && chmod 775 /app/tmp \
    # Set read-only directory permissions (755 - read allowed, no write access)
    && chmod 755 /app/config /app/prompts \
    # Verify directory creation and permissions
    && echo "Verifying directory structure..." \
    && ls -la /app/ \
    && ls -la /app/data/ \
    && ls -la /app/logs/ \
    && echo "Directory setup complete - all required directories created with proper ownership (${USER_ID}:${GROUP_ID})"

# Copy prompt files
COPY prompts/*.txt ./prompts/

# Copy health check script
COPY healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh && \
    if [ "${USER_ID}" != "0" ]; then \
        chown ${USER_ID}:${GROUP_ID} /app/healthcheck.sh; \
    fi

# Copy Docker entrypoint script
COPY scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && \
    if [ "${USER_ID}" != "0" ]; then \
        chown ${USER_ID}:${GROUP_ID} /app/docker-entrypoint.sh; \
    fi

# Switch to appropriate user - stay as root if USER_ID=0, otherwise use created user
RUN if [ "${USER_ID}" != "0" ]; then \
        echo "Switching to user: botuser (${USER_ID}:${GROUP_ID})"; \
    else \
        echo "Staying as root user for container execution"; \
    fi

# Use conditional USER directive - stay as root if USER_ID=0
USER ${USER_ID}:${GROUP_ID}

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
