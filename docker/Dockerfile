# AI Trading Bot Dockerfile - Alpine Optimized with Functional Programming Support
# Multi-stage build optimized for Alpine deployment with FP runtime capabilities

# Build stage - Alpine optimized with minimal dependencies
FROM --platform=linux/amd64 python:3.12-alpine AS builder

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG POETRY_VERSION=1.8.2
ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM
ARG FP_ENABLED=true
ARG FP_RUNTIME_MODE=hybrid

# Alpine-optimized environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Alpine-specific locale settings
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Functional Programming Runtime Environment
    FP_RUNTIME_ENABLED=true \
    FP_RUNTIME_MODE=hybrid \
    FP_EFFECT_TIMEOUT=30.0 \
    FP_MAX_CONCURRENT_EFFECTS=100 \
    FP_ERROR_RECOVERY=true \
    FP_METRICS_ENABLED=true

# Alpine-optimized package installation with minimal dependencies
RUN apk add --no-cache \
    # Core build tools
    build-base \
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
    # SSL/TLS support
    openssl-dev \
    # Foreign Function Interface
    libffi-dev \
    # Math libraries for numpy/scipy
    openblas-dev \
    lapack-dev \
    gfortran \
    # XML parsing (for some dependencies)
    libxml2-dev \
    libxslt-dev \
    # Compression libraries
    zlib-dev \
    # Process management
    procps

# Install Poetry with Alpine-optimized settings and efficient caching
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

# Install uv for faster dependency management
RUN pip install uv

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install dependencies directly to system Python (no virtual environment)
RUN echo "Installing dependencies with uv..." \
    && uv pip install --system \
        aiofiles>=24.1.0 \
        aiohttp>=3.12.15 \
        asyncio>=3.4.3 \
        click>=8.2.1 \
        cryptography>=45.0.5 \
        cython>=3.1.2 \
        langchain>=0.3.27 \
        langchain-openai>=0.3.28 \
        numpy>=1.26.4 \
        openai>=1.97.1 \
        pandas>=2.3.1 \
        pandas-ta>=0.3.14b0 \
        psutil>=7.0.0 \
        pydantic>=2.11.7 \
        pydantic-settings>=2.10.1 \
        python-dotenv>=1.1.1 \
        requests>=2.32.4 \
        rich>=14.1.0 \
        setuptools>=75.9.1 \
        websockets>=15.0.1 \
        wheel>=0.45.1 \
    && echo "Dependencies installed successfully"

# Clean up build artifacts for smaller image size
RUN echo "Cleaning up build artifacts..." \
    && find /usr/local/lib/python*/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python*/site-packages -name "__pycache__" -type d -exec rm -rf {} + || true \
    && echo "Build artifacts cleaned up"

# Note: bluefin-v2-client can be installed at runtime if live trading is needed
# For paper trading mode, the bot works without it

# Production stage - Alpine runtime optimized
FROM --platform=linux/amd64 python:3.12-alpine AS production

# Copy build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0
ARG EXCHANGE_TYPE=coinbase
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG FP_ENABLED=true
ARG FP_RUNTIME_MODE=hybrid

# Alpine-optimized production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
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

# Alpine-optimized runtime dependencies with minimal footprint
RUN apk add --no-cache \
    # Core runtime libraries
    ca-certificates \
    openssl \
    curl \
    wget \
    # Math libraries for numpy/pandas
    openblas \
    lapack \
    # System utilities
    procps \
    # Alpine networking tools
    netcat-openbsd \
    bind-tools \
    iputils \
    # Git for version info
    git \
    # Text processing utilities
    less \
    nano \
    # File compression (for logs)
    gzip

# Create non-root user with Alpine-compatible settings
RUN addgroup -g ${GROUP_ID} botuser && \
    adduser -u ${USER_ID} -G botuser -s /bin/sh -D botuser

# Set consistent working directory
WORKDIR /app

# Copy Python packages from builder (system installation, no virtual environment)
COPY --from=builder --chown=${USER_ID}:${GROUP_ID} /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder --chown=${USER_ID}:${GROUP_ID} /usr/local/bin /usr/local/bin

# Verify Python installation works properly
RUN echo "Testing Python installation..." \
    && python --version \
    && pip list \
    && echo "Python installation verification completed"

# Copy application code with proper ownership
COPY --chown=${USER_ID}:${GROUP_ID} bot/ ./bot/
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml ./

# Alpine-optimized directory creation with streamlined permissions
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
    # Set optimized permissions for Alpine compatibility
    && echo "Setting Alpine-compatible permissions..." \
    && find /app -type d -exec chmod 755 {} + \
    && chmod 775 /app/logs /app/data /app/tmp \
    # Verify critical directory structure exists
    && echo "Verifying directory structure..." \
    && test -d /app/config && test -d /app/logs && test -d /app/data \
    && test -d /app/data/fp_runtime && test -d /app/logs/fp \
    && echo "✅ Directory setup completed successfully for Alpine environment"

# Copy prompt files with proper ownership
COPY --chown=${USER_ID}:${GROUP_ID} prompts/*.txt ./prompts/

# Copy and setup health check script with Alpine compatibility
COPY --chown=${USER_ID}:${GROUP_ID} healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Copy and setup Docker entrypoint script with Alpine compatibility
COPY --chown=${USER_ID}:${GROUP_ID} scripts/docker-entrypoint-simple.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Final verification of Python installation before user switch
RUN echo "Final verification before switching to non-root user..." \
    && test -x /usr/local/bin/python \
    && python -c "import sys; print(f'Python {sys.version} ready')" \
    && echo "✅ Python installation verified"

# Switch to non-root user with Alpine compatibility
USER ${USER_ID}:${GROUP_ID}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Expose port for health checks and monitoring
EXPOSE 8080

# Set entrypoint to initialization script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command - starts in safe dry-run mode with system Python
CMD ["live", "--dry-run"]

# Alpine deployment optimized labels with FP support
LABEL org.opencontainers.image.title="AI Trading Bot with Functional Programming Runtime" \
      org.opencontainers.image.description="Alpine optimized crypto trading bot with Coinbase and Bluefin support, featuring functional programming runtime" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="AI Trading Bot" \
      org.opencontainers.image.exchange="${EXCHANGE_TYPE}" \
      org.opencontainers.image.platform="${TARGETPLATFORM}" \
      alpine.optimized="true" \
      alpine.compatible="3.18+" \
      fp.runtime.enabled="${FP_ENABLED}" \
      fp.runtime.mode="${FP_RUNTIME_MODE}" \
      fp.effect.interpreter="true" \
      fp.scheduler="true" \
      fp.adapters="true" \
      maintainer="ai-trading-bot"
