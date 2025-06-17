# Bluefin SDK Service - Isolated container for Bluefin DEX operations
FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first
COPY requirements-minimal.txt /app/

# Install minimal dependencies for WebSocket testing
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy service code
COPY bluefin_service.py /app/

# Create non-root user
RUN groupadd --gid 1000 bluefin \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash bluefin \
    && chown -R bluefin:bluefin /app

USER bluefin

# Expose service port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the service
CMD ["python", "bluefin_service.py"]