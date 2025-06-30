# Testing Dependencies and Docker Configuration for Bluefin Orderbook

## Overview

This document specifies all testing dependencies, Docker configurations, and environment requirements needed to implement the comprehensive testing strategy for Bluefin orderbook functionality.

## Required Testing Dependencies

### Core Testing Framework Dependencies

```toml
# Add to pyproject.toml [tool.poetry.group.testing.dependencies]

# Property-based testing framework
hypothesis = "^6.125.0"              # Core property-based testing
hypothesis-pytest = "^0.19.0"        # Pytest integration for hypothesis
faker = "^30.8.2"                    # Realistic fake data generation

# Enhanced pytest functionality
pytest = "^8.4.0"                    # Core testing framework
pytest-asyncio = "^1.0.0"            # Async test support
pytest-cov = "^6.2.0"                # Coverage reporting
pytest-mock = "^3.14.0"              # Enhanced mocking capabilities
pytest-xdist = "^3.6.3"              # Parallel test execution
pytest-timeout = "^2.3.1"            # Test timeout handling
pytest-html = "^4.1.1"               # HTML test reports
pytest-benchmark = "^4.0.0"          # Performance benchmarking
pytest-rerunfailures = "^14.0"       # Flaky test handling
pytest-sugar = "^1.0.0"              # Better test output formatting

# Financial data testing
pandas-stubs = "^2.2.0"              # Type stubs for pandas
numpy-stubs = "^1.26.0"              # Type stubs for numpy
types-python-dateutil = "^2.9.0"     # Date/time type stubs

# API and WebSocket mocking
responses = "^0.25.4"                # HTTP request mocking
aioresponses = "^0.7.7"              # Async HTTP mocking
pytest-httpserver = "^1.1.0"         # Local HTTP server for testing
websockets = "^15.0.0"               # WebSocket testing utilities
pytest-websockets = "^0.5.3"         # WebSocket-specific test helpers

# Database testing
pytest-postgresql = "^6.1.1"         # PostgreSQL test fixtures
pytest-redis = "^3.1.2"              # Redis test fixtures
sqlalchemy-utils = "^0.42.2"         # Database utilities for testing
alembic = "^1.14.0"                  # Database migration testing

# Load and performance testing
locust = "^2.32.2"                   # Load testing framework
py-spy = "^0.3.14"                   # Performance profiling
memory-profiler = "^0.61.0"          # Memory usage profiling
pympler = "^0.9"                     # Advanced memory analysis
line-profiler = "^4.2.0"             # Line-by-line profiling

# Security testing
bandit = "^1.8.0"                    # Security linting
safety = "^3.5.2"                    # Dependency vulnerability checking
semgrep = "^1.95.0"                  # SAST security analysis

# Test utilities
freezegun = "^1.5.1"                 # Time/date mocking
factory-boy = "^3.3.1"               # Test data factories
pytest-factoryboy = "^2.7.0"         # Factory Boy pytest integration
mimesis = "^18.0.0"                  # Advanced fake data generation
```

### Financial Domain Testing Dependencies

```toml
# Add to pyproject.toml [tool.poetry.group.financial-testing.dependencies]

# Quantitative finance testing
quantlib = "^1.36.0"                 # Quantitative finance library
pyfolio = "^0.9.2"                   # Portfolio performance analysis
zipline = "^3.0.0"                   # Algorithmic trading library
ta-lib = "^0.4.32"                   # Technical analysis library
ccxt = "^4.4.36"                     # Multi-exchange connectivity
pandas-market-calendars = "^4.4.1"   # Market calendar handling
yfinance = "^0.2.48"                 # Yahoo Finance data for testing

# Financial data validation
pydantic-extra-types = "^2.10.1"     # Additional validation types
decimal-precision = "^1.0.0"         # High-precision decimal handling
money = "^1.3.0"                     # Currency handling
babel = "^2.16.0"                    # Locale-aware formatting
```

### Monitoring and Observability Dependencies

```toml
# Add to pyproject.toml [tool.poetry.group.monitoring.dependencies]

# Metrics and monitoring
prometheus-client = "^0.22.0"        # Metrics collection
structlog = "^24.6.0"                # Structured logging
opentelemetry-api = "^1.30.0"        # Distributed tracing
opentelemetry-sdk = "^1.30.0"        # Tracing SDK
psutil = "^6.1.0"                    # System resource monitoring
py-cpuinfo = "^9.0.0"                # CPU information
```

### Development and Debugging Dependencies

```toml
# Add to pyproject.toml [tool.poetry.group.dev-testing.dependencies]

# Debugging and analysis
pudb = "^2024.1.3"                   # Advanced debugger
ipdb = "^0.13.13"                    # IPython debugger
pytest-pdb = "^0.2.0"                # PDB integration
pytest-clarity = "^1.0.1"            # Better assertion output
pytest-picked = "^0.5.0"             # Run tests based on changes

# Code quality for tests
ruff = "^0.12.0"                     # Fast linting (already included)
mypy = "^1.16.0"                     # Type checking (already included)
vulture = "^2.14.0"                  # Dead code detection (already included)
```

## Docker Test Environment Configuration

### Base Test Dockerfile

```dockerfile
# Dockerfile.test
FROM python:3.12-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    redis-tools \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for testing
ENV PYTHONPATH=/app
ENV TESTING_MODE=true
ENV LOG_LEVEL=DEBUG
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.8.4
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies including test dependencies
RUN poetry install --with testing,financial-testing,monitoring,dev-testing

# Copy application code
COPY . .

# Create directories for test outputs
RUN mkdir -p /app/test-results /app/logs /app/coverage

# Set up test entrypoint
COPY tests/docker/test-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/test-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/test-entrypoint.sh"]
CMD ["pytest", "tests/", "-v"]

# Test image with additional debugging tools
FROM base as test-debug

RUN pip install \
    pytest-pdbpp \
    rich \
    icecream

# Load testing image
FROM base as load-test

RUN pip install \
    locust \
    gevent \
    greenlet

COPY tests/load/ /app/load-tests/
WORKDIR /app/load-tests

CMD ["locust", "--host=http://app:8000"]
```

### Docker Compose Test Configuration

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  # Main application under test
  app-test:
    build:
      context: .
      dockerfile: Dockerfile.test
      target: base
    environment:
      - TESTING_MODE=true
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://test_user:test_pass@test-db:5432/test_db
      - REDIS_URL=redis://test-redis:6379/0
      - BLUEFIN_SERVICE_URL=http://mock-bluefin:8080
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
    depends_on:
      test-db:
        condition: service_healthy
      test-redis:
        condition: service_healthy
      mock-bluefin:
        condition: service_started
    volumes:
      - ./tests:/app/tests
      - ./test-results:/app/test-results
      - ./logs:/app/logs
      - test-prometheus-data:/tmp/prometheus
    networks:
      - test-network
    command: >
      bash -c "
        # Wait for dependencies
        ./tests/docker/wait-for-services.sh &&
        # Run test suite
        pytest tests/ -v
          --cov=bot
          --cov-report=html:/app/test-results/coverage
          --cov-report=xml:/app/test-results/coverage.xml
          --junit-xml=/app/test-results/junit.xml
          --html=/app/test-results/report.html
          --self-contained-html
      "

  # Debug version with additional tools
  app-test-debug:
    build:
      context: .
      dockerfile: Dockerfile.test
      target: test-debug
    environment:
      - TESTING_MODE=true
      - LOG_LEVEL=DEBUG
      - DATABASE_URL=postgresql://test_user:test_pass@test-db:5432/test_db
    depends_on:
      - test-db
      - test-redis
    volumes:
      - ./tests:/app/tests
      - ./test-results:/app/test-results
    networks:
      - test-network
    stdin_open: true
    tty: true
    command: bash

  # Load testing service
  load-test:
    build:
      context: .
      dockerfile: Dockerfile.test
      target: load-test
    environment:
      - TARGET_HOST=app-test
      - TARGET_PORT=8000
      - USERS=100
      - SPAWN_RATE=10
      - RUN_TIME=300
    depends_on:
      - app-test
    volumes:
      - ./tests/load:/app/load-tests
      - ./test-results:/app/test-results
    networks:
      - test-network
    ports:
      - "8089:8089"  # Locust web interface

  # Mock Bluefin service
  mock-bluefin:
    build:
      context: ./tests/mocks
      dockerfile: Dockerfile.bluefin-mock
    environment:
      - MOCK_LATENCY_MS=50
      - MOCK_ERROR_RATE=0.01
      - MOCK_RATE_LIMIT=1000
    ports:
      - "8080:8080"
    volumes:
      - ./tests/data:/app/test-data
    networks:
      - test-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Test database
  test-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=test_db
      - POSTGRES_USER=test_user
      - POSTGRES_PASSWORD=test_pass
      - POSTGRES_HOST_AUTH_METHOD=trust
    volumes:
      - test-db-data:/var/lib/postgresql/data
      - ./tests/sql:/docker-entrypoint-initdb.d
    ports:
      - "5433:5432"
    networks:
      - test-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user -d test_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Test Redis
  test-redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    networks:
      - test-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # Test monitoring
  test-prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./tests/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - test-prometheus-data:/prometheus
    networks:
      - test-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'

  # Test results server
  test-results:
    image: nginx:alpine
    ports:
      - "8088:80"
    volumes:
      - ./test-results:/usr/share/nginx/html:ro
      - ./tests/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - test-network

volumes:
  test-db-data:
  test-prometheus-data:

networks:
  test-network:
    driver: bridge
```

### Mock Services Configuration

#### Mock Bluefin Service Dockerfile

```dockerfile
# tests/mocks/Dockerfile.bluefin-mock
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install \
    fastapi==0.115.0 \
    uvicorn==0.34.0 \
    pydantic==2.9.0 \
    aiofiles==24.1.0

# Copy mock service code
COPY bluefin_mock_service.py .
COPY test_data/ ./test_data/

EXPOSE 8080

CMD ["uvicorn", "bluefin_mock_service:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Mock Bluefin Service Implementation

```python
# tests/mocks/bluefin_mock_service.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import asyncio
import random
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

app = FastAPI(title="Mock Bluefin Service")

# Mock configuration
MOCK_LATENCY_MS = int(os.getenv("MOCK_LATENCY_MS", "50"))
MOCK_ERROR_RATE = float(os.getenv("MOCK_ERROR_RATE", "0.01"))
MOCK_RATE_LIMIT = int(os.getenv("MOCK_RATE_LIMIT", "1000"))

# Request tracking for rate limiting
request_counts = {}

class OrderbookLevel(BaseModel):
    price: str
    size: str

class OrderbookResponse(BaseModel):
    symbol: str
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: str

async def add_latency():
    """Add realistic latency to responses."""
    if MOCK_LATENCY_MS > 0:
        # Add some jitter
        latency = MOCK_LATENCY_MS + random.randint(-10, 10)
        await asyncio.sleep(latency / 1000.0)

def check_rate_limit(client_id: str = "default"):
    """Simple rate limiting check."""
    now = datetime.now()
    minute = now.replace(second=0, microsecond=0)

    key = f"{client_id}:{minute}"
    count = request_counts.get(key, 0)

    if count >= MOCK_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    request_counts[key] = count + 1

def maybe_inject_error():
    """Randomly inject errors based on configuration."""
    if random.random() < MOCK_ERROR_RATE:
        error_type = random.choice([
            "timeout", "server_error", "bad_gateway", "service_unavailable"
        ])

        if error_type == "timeout":
            # Simulate timeout by adding extreme delay
            asyncio.sleep(30)
        elif error_type == "server_error":
            raise HTTPException(status_code=500, detail="Internal server error")
        elif error_type == "bad_gateway":
            raise HTTPException(status_code=502, detail="Bad gateway")
        elif error_type == "service_unavailable":
            raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/orderbook", response_model=OrderbookResponse)
async def get_orderbook(symbol: str = Query(..., description="Trading symbol")):
    """Mock orderbook endpoint with realistic data."""
    await add_latency()
    check_rate_limit()
    maybe_inject_error()

    # Generate realistic orderbook data
    base_price = 50000.0 if symbol.startswith("BTC") else 3000.0

    # Generate bids (decreasing prices)
    bids = []
    for i in range(10):
        price = base_price - (i * 10)
        size = random.uniform(0.1, 5.0)
        bids.append(OrderbookLevel(price=f"{price:.2f}", size=f"{size:.4f}"))

    # Generate asks (increasing prices)
    asks = []
    for i in range(10):
        price = base_price + 10 + (i * 10)
        size = random.uniform(0.1, 5.0)
        asks.append(OrderbookLevel(price=f"{price:.2f}", size=f"{size:.4f}"))

    return OrderbookResponse(
        symbol=symbol,
        bids=bids,
        asks=asks,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/symbols")
async def get_symbols():
    """Return available trading symbols."""
    await add_latency()
    check_rate_limit()
    maybe_inject_error()

    return {
        "symbols": [
            "BTC-PERP", "ETH-PERP", "SOL-PERP", "SUI-PERP",
            "AVAX-PERP", "NEAR-PERP", "APT-PERP"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Test Orchestration Scripts

#### Master Test Runner

```bash
#!/bin/bash
# tests/docker/run_comprehensive_tests.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.test.yml"
TEST_RESULTS_DIR="./test-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${TEST_RESULTS_DIR}/run_${TIMESTAMP}"

echo -e "${BLUE}🚀 Starting comprehensive Bluefin orderbook tests...${NC}"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up test environment...${NC}"
    docker-compose -f $COMPOSE_FILE down -v --remove-orphans
}
trap cleanup EXIT

# Pre-flight checks
echo -e "${BLUE}🔍 Running pre-flight checks...${NC}"

# Check Docker
if ! docker --version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check Docker Compose
if ! docker-compose --version > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Check available ports
for port in 5433 6380 8080 8088 8089 9091; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $port is in use. Tests may fail.${NC}"
    fi
done

# Start test environment
echo -e "${BLUE}📦 Starting test environment...${NC}"
docker-compose -f $COMPOSE_FILE build
docker-compose -f $COMPOSE_FILE up -d test-db test-redis mock-bluefin

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
timeout 60 bash -c 'until docker-compose -f docker-compose.test.yml exec test-db pg_isready -U test_user; do sleep 1; done'
timeout 60 bash -c 'until docker-compose -f docker-compose.test.yml exec test-redis redis-cli ping; do sleep 1; done'
timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 1; done'

# Test execution matrix
declare -A test_suites=(
    ["unit"]="tests/unit/ -x --tb=short"
    ["integration"]="tests/integration/ -x --tb=short"
    ["property"]="tests/property/ --hypothesis-show-statistics"
    ["performance"]="tests/performance/ --benchmark-only"
    ["security"]="tests/security/ -v"
)

# Run test suites
total_suites=${#test_suites[@]}
current_suite=0

for suite in "${!test_suites[@]}"; do
    current_suite=$((current_suite + 1))
    echo -e "${BLUE}🧪 Running $suite tests ($current_suite/$total_suites)...${NC}"

    start_time=$(date +%s)

    if docker-compose -f $COMPOSE_FILE run --rm \
        -v "${PWD}/${RESULTS_DIR}:/app/test-results" \
        app-test pytest ${test_suites[$suite]} \
        --junit-xml="/app/test-results/${suite}_junit.xml" \
        --cov-report="xml:/app/test-results/${suite}_coverage.xml"; then

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✅ $suite tests passed (${duration}s)${NC}"
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${RED}❌ $suite tests failed (${duration}s)${NC}"
        echo "Suite: $suite" >> "${RESULTS_DIR}/failed_suites.txt"
    fi
done

# Run load tests if performance tests passed
if [ ! -f "${RESULTS_DIR}/failed_suites.txt" ] || ! grep -q "performance" "${RESULTS_DIR}/failed_suites.txt"; then
    echo -e "${BLUE}⚡ Running load tests...${NC}"
    docker-compose -f $COMPOSE_FILE up -d app-test
    sleep 10  # Let app warm up

    docker-compose -f $COMPOSE_FILE run --rm load-test \
        --headless \
        --users 50 \
        --spawn-rate 5 \
        --run-time 60s \
        --html "/app/test-results/load_test_report.html"
fi

# Generate comprehensive report
echo -e "${BLUE}📊 Generating test report...${NC}"
cat > "${RESULTS_DIR}/test_summary.md" << EOF
# Test Execution Summary

**Timestamp**: $(date)
**Results Directory**: ${RESULTS_DIR}

## Test Suites Executed

$(for suite in "${!test_suites[@]}"; do
    if [ -f "${RESULTS_DIR}/${suite}_junit.xml" ]; then
        echo "- ✅ $suite"
    else
        echo "- ❌ $suite"
    fi
done)

## Failed Suites

$(if [ -f "${RESULTS_DIR}/failed_suites.txt" ]; then
    cat "${RESULTS_DIR}/failed_suites.txt"
else
    echo "None"
fi)

## Coverage Reports

$(ls -la "${RESULTS_DIR}"/*coverage* 2>/dev/null || echo "No coverage reports generated")

## Performance Reports

$(ls -la "${RESULTS_DIR}"/*performance* "${RESULTS_DIR}"/*load* 2>/dev/null || echo "No performance reports generated")

EOF

# Final summary
if [ -f "${RESULTS_DIR}/failed_suites.txt" ]; then
    echo -e "${RED}❌ Some test suites failed. Check ${RESULTS_DIR}/test_summary.md for details.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All test suites passed! Results in ${RESULTS_DIR}/${NC}"
    echo -e "${BLUE}🌐 View results at: http://localhost:8088/run_${TIMESTAMP}/${NC}"
fi
```

#### Service Health Check Script

```bash
#!/bin/bash
# tests/docker/wait-for-services.sh

set -euo pipefail

wait_for_service() {
    local service_name=$1
    local check_command=$2
    local timeout=${3:-60}

    echo "Waiting for $service_name..."
    local count=0
    while ! eval "$check_command" > /dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            echo "❌ Timeout waiting for $service_name"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    echo "✅ $service_name is ready"
}

# Wait for all test services
wait_for_service "PostgreSQL" "pg_isready -h localhost -p 5433 -U test_user"
wait_for_service "Redis" "redis-cli -h localhost -p 6380 ping"
wait_for_service "Mock Bluefin" "curl -f http://localhost:8080/health"

echo "🎉 All services are ready!"
```

### Test Environment Configuration Files

#### Pytest Configuration

```ini
# tests/pytest.ini
[tool:pytest]
minversion = 6.0
addopts =
    -ra
    -q
    --strict-markers
    --strict-config
    --tb=short
    --durations=10
testpaths = tests
asyncio_mode = auto
timeout = 300
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (medium, dependencies)
    property: Property-based tests (slow, comprehensive)
    performance: Performance and load tests
    security: Security and vulnerability tests
    slow: Tests that take longer than 30 seconds
    docker: Tests that require Docker containers
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

#### Coverage Configuration

```ini
# tests/.coveragerc
[run]
source = bot
omit =
    tests/*
    bot/__init__.py
    */migrations/*
    */venv/*
    */virtualenv/*
parallel = true
concurrency = multiprocessing

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
precision = 2
show_missing = true

[html]
directory = test-results/coverage

[xml]
output = test-results/coverage.xml
```

This comprehensive testing infrastructure provides all the necessary dependencies and Docker configurations to implement robust testing for the Bluefin orderbook functionality, ensuring reliability, performance, and security of the financial trading system.
