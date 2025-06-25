# AI Trading Bot Dockerfile - Ubuntu Optimized with Functional Programming Support
# Multi-stage build optimized for Ubuntu deployment with FP runtime capabilities

# Build stage - Ubuntu optimized with simplified architecture
FROM --platform=linux/amd64 python:3.12-slim AS builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG POETRY_VERSION=1.8.2
ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM
ARG FP_ENABLED=true
ARG FP_RUNTIME_MODE=hybrid

# Ubuntu-optimized environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Ubuntu-specific locale settings
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Functional Programming Runtime Environment
    FP_RUNTIME_ENABLED=true \
    FP_RUNTIME_MODE=hybrid \
    FP_EFFECT_TIMEOUT=30.0 \
    FP_MAX_CONCURRENT_EFFECTS=100 \
    FP_ERROR_RECOVERY=true \
    FP_METRICS_ENABLED=true

# Ubuntu-optimized package installation with essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core build tools
    build-essential \
    gcc \
    g++ \
    make \
    # System utilities
    curl \
    wget \
    ca-certificates \
    git \
    # Python development headers
    python3-dev \
    python3-pip \
    # SSL/TLS support
    libssl-dev \
    # Foreign Function Interface
    libffi-dev \
    # Math libraries for numpy/scipy
    libblas-dev \
    liblapack-dev \
    gfortran \
    # XML parsing (for some dependencies)
    libxml2-dev \
    libxslt1-dev \
    # Compression libraries
    zlib1g-dev \
    # Ubuntu-specific system tools
    software-properties-common \
    apt-transport-https \
    gnupg \
    lsb-release \
    # Process management
    procps \
    # Cleanup in single layer for optimal build
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/* \
    && ldconfig

# Install Poetry with Ubuntu-optimized settings and efficient caching
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VENV_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/opt/poetry-cache \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_PATH=/opt/poetry-venvs \
    POETRY_INSTALLER_PARALLEL=true \
    POETRY_INSTALLER_MAX_WORKERS=10

# Set working directory early for consistency
WORKDIR /app

# Install Poetry with error handling and optimized caching
RUN curl -sSL https://install.python-poetry.org | python3 - --version=${POETRY_VERSION} \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry \
    && poetry --version

ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy dependency files first for better layer caching
COPY pyproject.toml poetry.lock* ./

# Configure Poetry for optimal performance and caching
RUN echo "Configuring Poetry for optimized Ubuntu environment..." \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry config cache-dir $POETRY_CACHE_DIR \
    && poetry config installer.parallel true \
    && poetry config installer.max-workers 10 \
    && poetry config virtualenvs.path $POETRY_VIRTUALENVS_PATH \
    && echo "Poetry configured successfully"

# Install dependencies with enhanced caching and error handling
RUN --mount=type=cache,target=/opt/poetry-cache \
    echo "Installing dependencies with cache optimization..." \
    && (poetry install --only=main --no-root --no-dev || \
        (echo "Lock file out of sync, regenerating for Ubuntu..." && \
         poetry lock --no-update && \
         poetry install --only=main --no-root --no-dev)) \
    && echo "Verifying virtual environment..." \
    && ls -la /app/.venv/lib/python*/site-packages/ \
    && echo "Poetry installation completed successfully"

# Clean up Poetry cache in final build step (keeping Docker layer cache)
RUN echo "Cleaning up build artifacts..." \
    && find /app/.venv -name "*.pyc" -delete \
    && find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + || true \
    && echo "Build artifacts cleaned up"

# Note: bluefin-v2-client can be installed at runtime if live trading is needed
# For paper trading mode, the bot works without it

# Production stage - Ubuntu runtime optimized
FROM --platform=linux/amd64 python:3.12-slim AS production

# Copy build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG EXCHANGE_TYPE=coinbase
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG FP_ENABLED=true
ARG FP_RUNTIME_MODE=hybrid

# Ubuntu-optimized production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Python virtual environment activation
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    # Application metadata
    APP_VERSION=${VERSION} \
    EXCHANGE__EXCHANGE_TYPE=${EXCHANGE_TYPE} \
    # Functional Programming Runtime
    FP_RUNTIME_ENABLED=${FP_ENABLED} \
    FP_RUNTIME_MODE=${FP_RUNTIME_MODE} \
    FP_EFFECT_TIMEOUT=30.0 \
    FP_MAX_CONCURRENT_EFFECTS=100 \
    FP_ERROR_RECOVERY=true \
    FP_METRICS_ENABLED=true \
    FP_SCHEDULER_ENABLED=true \
    FP_ASYNC_RUNTIME=true

# Ubuntu-optimized runtime dependencies with minimal footprint
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core runtime libraries
    ca-certificates \
    openssl \
    curl \
    wget \
    # Python runtime dependencies
    libssl3 \
    libffi8 \
    # Math libraries for numpy/pandas
    libblas3 \
    liblapack3 \
    # System utilities
    procps \
    # Ubuntu networking tools
    netcat-openbsd \
    dnsutils \
    iputils-ping \
    # Git for version info
    git \
    # Text processing utilities
    less \
    nano \
    # File compression (for logs)
    gzip \
    # Single-layer cleanup for optimal size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/* \
    && ldconfig

# Create non-root user with Ubuntu-compatible settings
RUN groupadd --gid ${GROUP_ID} botuser && \
    useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash botuser && \
    echo "botuser:!" | chpasswd -e

# Set consistent working directory
WORKDIR /app

# Copy virtual environment from builder with proper ownership
COPY --from=builder --chown=${USER_ID}:${GROUP_ID} /app/.venv /app/.venv

# Verify virtual environment activation works properly
RUN echo "Testing virtual environment activation..." \
    && /app/.venv/bin/python --version \
    && /app/.venv/bin/pip list \
    && echo "Virtual environment verification completed"

# Copy application code with proper ownership
COPY --chown=${USER_ID}:${GROUP_ID} bot/ ./bot/
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml ./

# Ubuntu-optimized directory creation with streamlined permissions
RUN echo "Creating application directory structure..." \
    # Create all required directories in a single command for efficiency
    && mkdir -p \
        /app/config \
        /app/logs/mcp \
        /app/logs/bluefin \
        /app/logs/trades \
        /app/logs/fp/effects \
        /app/logs/fp/scheduler \
        /app/logs/fp/interpreter \
        /app/data/mcp_memory \
        /app/data/orders \
        /app/data/paper_trading \
        /app/data/positions \
        /app/data/bluefin \
        /app/data/omnisearch_cache \
        /app/data/fp_runtime/effects \
        /app/data/fp_runtime/scheduler \
        /app/data/fp_runtime/metrics \
        /app/data/fp_runtime/config \
        /app/data/fp_runtime/adapters \
        /app/prompts \
        /app/tmp \
    # Set ownership to the application user in single operation
    && echo "Setting ownership to ${USER_ID}:${GROUP_ID}..." \
    && chown -R ${USER_ID}:${GROUP_ID} /app \
    # Set optimized permissions for Ubuntu compatibility
    && echo "Setting Ubuntu-compatible permissions..." \
    && find /app -type d -exec chmod 755 {} + \
    && chmod 775 /app/logs /app/data /app/tmp \
    # Verify critical directory structure exists
    && echo "Verifying directory structure..." \
    && test -d /app/config && test -d /app/logs && test -d /app/data \
    && test -d /app/data/fp_runtime && test -d /app/logs/fp \
    && echo "✅ Directory setup completed successfully for Ubuntu environment"

# Copy prompt files with proper ownership
COPY --chown=${USER_ID}:${GROUP_ID} prompts/*.txt ./prompts/

# Copy and setup health check script with Ubuntu compatibility
COPY --chown=${USER_ID}:${GROUP_ID} healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Copy and setup Docker entrypoint script with Ubuntu compatibility
COPY --chown=${USER_ID}:${GROUP_ID} scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Final verification of virtual environment before user switch
RUN echo "Final verification before switching to non-root user..." \
    && test -x /app/.venv/bin/python \
    && /app/.venv/bin/python -c "import sys; print(f'Python {sys.version} ready')" \
    && echo "✅ Python virtual environment verified"

# Switch to non-root user with Ubuntu compatibility
USER ${USER_ID}:${GROUP_ID}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Expose port for health checks and monitoring
EXPOSE 8080

# Set entrypoint to initialization script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command - starts in safe dry-run mode with explicit Python path
CMD ["/app/.venv/bin/python", "-m", "bot.main", "live", "--dry-run"]

# Ubuntu deployment optimized labels with FP support
LABEL org.opencontainers.image.title="AI Trading Bot with Functional Programming Runtime" \
      org.opencontainers.image.description="Ubuntu optimized crypto trading bot with Coinbase and Bluefin support, featuring functional programming runtime" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="AI Trading Bot" \
      org.opencontainers.image.exchange="${EXCHANGE_TYPE}" \
      org.opencontainers.image.platform="${TARGETPLATFORM}" \
      ubuntu.optimized="true" \
      ubuntu.compatible="22.04+" \
      fp.runtime.enabled="${FP_ENABLED}" \
      fp.runtime.mode="${FP_RUNTIME_MODE}" \
      fp.effect.interpreter="true" \
      fp.scheduler="true" \
      fp.adapters="true" \
      maintainer="ai-trading-bot"
