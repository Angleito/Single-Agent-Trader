#!/bin/bash
# Comprehensive Orderbook Testing Script
#
# This script orchestrates the complete orderbook testing suite including:
# - Environment setup and validation
# - Mock service startup
# - Test execution across all categories
# - Results collection and reporting
# - Cleanup and teardown
#
# Usage:
#   ./tests/docker/scripts/run_orderbook_tests.sh [test-type]
#
# Test types:
#   all          - Run all tests (default)
#   unit         - Unit tests only
#   integration  - Integration tests only
#   performance  - Performance tests only
#   stress       - Stress tests only
#   property     - Property-based tests only

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
TEST_TYPE="${1:-all}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${PROJECT_ROOT}/tests/docker/results/orderbook_${TIMESTAMP}"

# Test configuration
export TEST_MODE=true
export LOG_LEVEL=DEBUG
export PYTHONPATH="${PROJECT_ROOT}"
export PYTEST_TIMEOUT=300
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# LOGGING AND UTILITIES
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    log "Setting up test environment..."

    # Create results directory
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${PROJECT_ROOT}/tests/data/mock-bluefin"
    mkdir -p "${PROJECT_ROOT}/tests/data/mock-coinbase"
    mkdir -p "${PROJECT_ROOT}/tests/data/mock-exchange"
    mkdir -p "${PROJECT_ROOT}/logs/test"

    # Set permissions
    chmod -R 755 "${PROJECT_ROOT}/tests/docker/scripts"

    # Create test environment file
    cat > "${PROJECT_ROOT}/.env.test" << EOF
# Test Environment Configuration
TEST_MODE=true
LOG_LEVEL=DEBUG
PYTHONPATH=${PROJECT_ROOT}

# Mock service configuration
MOCK_BLUEFIN_URL=http://mock-bluefin:8080
MOCK_COINBASE_URL=http://mock-coinbase:8081
MOCK_EXCHANGE_WS_URL=ws://mock-exchange:8082/ws

# Test database configuration
TEST_DB_HOST=test-postgres
TEST_DB_PORT=5432
TEST_DB_NAME=orderbook_test
TEST_DB_USER=test_user
TEST_DB_PASSWORD=test_password

# System overrides for testing
SYSTEM__DRY_RUN=true
EXCHANGE__EXCHANGE_TYPE=mock
MCP_ENABLED=false
ENABLE_WEBSOCKET=true
FP_RUNTIME_ENABLED=true
FP_DEBUG_MODE=true

# Performance testing
BENCHMARK_ITERATIONS=1000
LOAD_TEST_DURATION=60
STRESS_TEST_CONNECTIONS=100
STRESS_TEST_MESSAGES_PER_SECOND=1000

# Test timeouts
PYTEST_TIMEOUT=300
TEST_TIMEOUT=300
WEBSOCKET_TIMEOUT=30
EOF

    log_success "Test environment setup complete"
}

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

build_test_images() {
    log "Building test Docker images..."

    cd "${PROJECT_ROOT}"

    # Build test runner image
    log "Building test runner image..."
    docker build \
        -f tests/docker/Dockerfile.test \
        -t orderbook-test-runner:latest \
        --build-arg USER_ID=$(id -u) \
        --build-arg GROUP_ID=$(id -g) \
        .

    # Build mock service images
    log "Building mock Bluefin service..."
    docker build \
        -f tests/docker/mocks/Dockerfile.mock-bluefin \
        -t mock-bluefin-service:latest \
        tests/docker/mocks/

    log "Building mock Coinbase service..."
    docker build \
        -f tests/docker/mocks/Dockerfile.mock-coinbase \
        -t mock-coinbase-service:latest \
        tests/docker/mocks/

    log "Building mock exchange WebSocket service..."
    docker build \
        -f tests/docker/mocks/Dockerfile.mock-exchange-ws \
        -t mock-exchange-ws:latest \
        tests/docker/mocks/

    log_success "All test images built successfully"
}

start_test_services() {
    log "Starting test services..."

    cd "${PROJECT_ROOT}"

    # Start infrastructure services first
    docker-compose -f docker-compose.test.yml up -d test-postgres test-redis

    # Wait for database to be ready
    log "Waiting for database to be ready..."
    timeout 60 bash -c 'until docker-compose -f docker-compose.test.yml exec test-postgres pg_isready -U test_user -d orderbook_test; do sleep 2; done'

    # Start mock services
    docker-compose -f docker-compose.test.yml up -d mock-bluefin mock-coinbase mock-exchange

    # Wait for mock services to be healthy
    log "Waiting for mock services to be ready..."
    for service in mock-bluefin mock-coinbase mock-exchange; do
        timeout 60 bash -c "until docker-compose -f docker-compose.test.yml ps ${service} | grep -q healthy; do sleep 2; done"
    done

    log_success "All test services started successfully"
}

stop_test_services() {
    log "Stopping test services..."

    cd "${PROJECT_ROOT}"
    docker-compose -f docker-compose.test.yml down -v

    log_success "Test services stopped"
}

# =============================================================================
# TEST EXECUTION
# =============================================================================

run_unit_tests() {
    log "Running orderbook unit tests..."

    cd "${PROJECT_ROOT}"

    docker-compose -f docker-compose.test.yml run --rm \
        -e PYTEST_WORKERS=4 \
        -e COVERAGE_PROCESS_START="${PROJECT_ROOT}/pyproject.toml" \
        test-runner \
        bash -c "
            python -m pytest tests/unit/fp/test_orderbook_*.py \
                -v \
                --tb=short \
                --cov=bot.fp.types.market \
                --cov-report=html:/app/test-results/coverage/unit \
                --cov-report=json:/app/test-results/coverage/unit/coverage.json \
                --junit-xml=/app/test-results/unit-tests.xml \
                --json-report --json-report-file=/app/test-results/unit-tests.json
        " 2>&1 | tee "${RESULTS_DIR}/unit-tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_success "Unit tests completed successfully"
    else
        log_error "Unit tests failed with exit code $exit_code"
    fi

    return $exit_code
}

run_integration_tests() {
    log "Running orderbook integration tests..."

    cd "${PROJECT_ROOT}"

    docker-compose -f docker-compose.test.yml run --rm \
        -e PYTEST_WORKERS=2 \
        test-runner \
        bash -c "
            python -m pytest tests/integration/test_orderbook_*.py \
                -v \
                --tb=short \
                --junit-xml=/app/test-results/integration-tests.xml \
                --json-report --json-report-file=/app/test-results/integration-tests.json
        " 2>&1 | tee "${RESULTS_DIR}/integration-tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_success "Integration tests completed successfully"
    else
        log_error "Integration tests failed with exit code $exit_code"
    fi

    return $exit_code
}

run_performance_tests() {
    log "Running orderbook performance tests..."

    cd "${PROJECT_ROOT}"

    docker-compose -f docker-compose.test.yml run --rm \
        -e PERFORMANCE_TEST_MODE=true \
        -e BENCHMARK_ITERATIONS=1000 \
        performance-test-runner \
        bash -c "
            python -m pytest tests/performance/test_orderbook_performance.py \
                -v \
                --tb=short \
                --benchmark-only \
                --benchmark-json=/app/test-results/benchmark-results.json \
                --junit-xml=/app/test-results/performance-tests.xml \
                --json-report --json-report-file=/app/test-results/performance-tests.json
        " 2>&1 | tee "${RESULTS_DIR}/performance-tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_success "Performance tests completed successfully"
    else
        log_error "Performance tests failed with exit code $exit_code"
    fi

    return $exit_code
}

run_stress_tests() {
    log "Running orderbook stress tests..."

    cd "${PROJECT_ROOT}"

    docker-compose -f docker-compose.test.yml run --rm \
        -e STRESS_TEST_MODE=true \
        -e CONCURRENT_CONNECTIONS=100 \
        -e MESSAGES_PER_SECOND=1000 \
        -e TEST_DURATION=300 \
        stress-test-runner \
        bash -c "
            python -m pytest tests/stress/test_orderbook_stress.py \
                -v \
                --tb=short \
                --junit-xml=/app/test-results/stress-tests.xml \
                --json-report --json-report-file=/app/test-results/stress-tests.json
        " 2>&1 | tee "${RESULTS_DIR}/stress-tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_success "Stress tests completed successfully"
    else
        log_error "Stress tests failed with exit code $exit_code"
    fi

    return $exit_code
}

run_property_tests() {
    log "Running orderbook property-based tests..."

    cd "${PROJECT_ROOT}"

    docker-compose -f docker-compose.test.yml run --rm \
        -e PROPERTY_TEST_MODE=true \
        -e HYPOTHESIS_MAX_EXAMPLES=1000 \
        -e HYPOTHESIS_DEADLINE=60000 \
        property-test-runner \
        bash -c "
            python -m pytest tests/property/test_orderbook_properties.py \
                -v \
                --tb=short \
                --junit-xml=/app/test-results/property-tests.xml \
                --json-report --json-report-file=/app/test-results/property-tests.json
        " 2>&1 | tee "${RESULTS_DIR}/property-tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_success "Property-based tests completed successfully"
    else
        log_error "Property-based tests failed with exit code $exit_code"
    fi

    return $exit_code
}

# =============================================================================
# RESULTS COLLECTION
# =============================================================================

collect_test_results() {
    log "Collecting test results..."

    cd "${PROJECT_ROOT}"

    # Copy test results from containers
    docker-compose -f docker-compose.test.yml run --rm \
        test-runner \
        bash -c "cp -r /app/test-results/* ${RESULTS_DIR}/ 2>/dev/null || true"

    # Copy coverage reports
    docker-compose -f docker-compose.test.yml run --rm \
        test-runner \
        bash -c "cp -r /app/test-results/coverage ${RESULTS_DIR}/ 2>/dev/null || true"

    # Generate summary report
    generate_test_summary_report

    log_success "Test results collected in ${RESULTS_DIR}"
}

generate_test_summary_report() {
    log "Generating test summary report..."

    cat > "${RESULTS_DIR}/test_summary.md" << EOF
# Orderbook Testing Summary Report

**Test Run:** ${TIMESTAMP}
**Test Type:** ${TEST_TYPE}
**Results Directory:** ${RESULTS_DIR}

## Test Results

EOF

    # Add individual test results
    for test_file in "${RESULTS_DIR}"/*.log; do
        if [[ -f "$test_file" ]]; then
            test_name=$(basename "$test_file" .log)
            echo "### ${test_name}" >> "${RESULTS_DIR}/test_summary.md"
            echo '```' >> "${RESULTS_DIR}/test_summary.md"
            tail -20 "$test_file" >> "${RESULTS_DIR}/test_summary.md"
            echo '```' >> "${RESULTS_DIR}/test_summary.md"
            echo "" >> "${RESULTS_DIR}/test_summary.md"
        fi
    done

    # Add system information
    cat >> "${RESULTS_DIR}/test_summary.md" << EOF

## System Information

- **Docker Version:** $(docker --version)
- **Docker Compose Version:** $(docker-compose --version)
- **Test Environment:** Docker containers
- **Test Framework:** pytest
- **Python Version:** 3.12

## Test Configuration

- **Test Mode:** ${TEST_MODE}
- **Log Level:** ${LOG_LEVEL}
- **Pytest Timeout:** ${PYTEST_TIMEOUT}s
- **Test Timeout:** ${TEST_TIMEOUT}s

## Files Generated

EOF

    # List all generated files
    find "${RESULTS_DIR}" -type f -exec basename {} \; | sort >> "${RESULTS_DIR}/test_summary.md"

    log_success "Test summary report generated"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

cleanup() {
    log "Performing cleanup..."
    stop_test_services

    # Remove test environment file
    rm -f "${PROJECT_ROOT}/.env.test"

    log_success "Cleanup completed"
}

main() {
    log "Starting orderbook testing suite (${TEST_TYPE})"
    log "Results will be saved to: ${RESULTS_DIR}"

    # Setup trap for cleanup
    trap cleanup EXIT

    # Setup environment
    setup_environment

    # Build images
    build_test_images

    # Start services
    start_test_services

    # Run tests based on type
    local overall_exit_code=0

    case "${TEST_TYPE}" in
        "unit")
            run_unit_tests || overall_exit_code=$?
            ;;
        "integration")
            run_integration_tests || overall_exit_code=$?
            ;;
        "performance")
            run_performance_tests || overall_exit_code=$?
            ;;
        "stress")
            run_stress_tests || overall_exit_code=$?
            ;;
        "property")
            run_property_tests || overall_exit_code=$?
            ;;
        "all")
            run_unit_tests || overall_exit_code=$?
            run_integration_tests || overall_exit_code=$?
            run_performance_tests || overall_exit_code=$?
            run_stress_tests || overall_exit_code=$?
            run_property_tests || overall_exit_code=$?
            ;;
        *)
            log_error "Unknown test type: ${TEST_TYPE}"
            log "Available types: all, unit, integration, performance, stress, property"
            exit 1
            ;;
    esac

    # Collect results
    collect_test_results

    # Final status
    if [ $overall_exit_code -eq 0 ]; then
        log_success "All orderbook tests completed successfully!"
        log "View results: ${RESULTS_DIR}/test_summary.md"
    else
        log_error "Some tests failed. Check logs in ${RESULTS_DIR}/"
    fi

    exit $overall_exit_code
}

# Print usage if help requested
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Comprehensive Orderbook Testing Script"
    echo ""
    echo "Usage: $0 [test-type]"
    echo ""
    echo "Test types:"
    echo "  all          - Run all tests (default)"
    echo "  unit         - Unit tests only"
    echo "  integration  - Integration tests only"
    echo "  performance  - Performance tests only"
    echo "  stress       - Stress tests only"
    echo "  property     - Property-based tests only"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests"
    echo "  $0 unit              # Run unit tests only"
    echo "  $0 performance       # Run performance tests only"
    exit 0
fi

# Run main function
main "$@"
