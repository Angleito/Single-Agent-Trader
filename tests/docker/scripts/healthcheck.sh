#!/bin/bash
# Health check script for test containers
#
# This script validates that the test environment is properly configured
# and all dependencies are available for testing.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
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

# Check Python environment
check_python_environment() {
    log "Checking Python environment..."
    
    # Check Python version
    python_version=$(python --version 2>&1)
    if [[ "$python_version" == *"3.12"* ]]; then
        log_success "Python version: $python_version"
    else
        log_error "Expected Python 3.12, got: $python_version"
        return 1
    fi
    
    # Check required packages
    required_packages=(
        "pytest"
        "hypothesis"
        "psutil"
        "docker"
        "websockets"
        "fastapi"
        "pydantic"
    )
    
    for package in "${required_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            log_success "Package available: $package"
        else
            log_error "Missing required package: $package"
            return 1
        fi
    done
    
    return 0
}

# Check test directories and permissions
check_test_directories() {
    log "Checking test directories and permissions..."
    
    required_dirs=(
        "/app/tests"
        "/app/test-results"
        "/app/logs"
        "/app/data"
        "/app/.pytest_cache"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            if [[ -w "$dir" ]]; then
                log_success "Directory writable: $dir"
            else
                log_warning "Directory not writable: $dir"
            fi
        else
            log_error "Missing directory: $dir"
            return 1
        fi
    done
    
    return 0
}

# Check environment variables
check_environment_variables() {
    log "Checking environment variables..."
    
    required_vars=(
        "PYTHONPATH"
        "TEST_MODE"
        "LOG_LEVEL"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            log_success "Environment variable set: $var=${!var}"
        else
            log_error "Missing environment variable: $var"
            return 1
        fi
    done
    
    return 0
}

# Check network connectivity to mock services
check_mock_services() {
    log "Checking connectivity to mock services..."
    
    # Mock services to check (if they exist)
    mock_services=(
        "mock-bluefin:8080"
        "mock-coinbase:8081"
        "mock-exchange:8082"
    )
    
    for service in "${mock_services[@]}"; do
        host=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "Mock service reachable: $service"
        else
            log_warning "Mock service not reachable: $service (may not be started yet)"
        fi
    done
    
    return 0
}

# Check database connectivity
check_database_connectivity() {
    log "Checking database connectivity..."
    
    if [[ -n "${TEST_DB_HOST:-}" ]] && [[ -n "${TEST_DB_PORT:-}" ]]; then
        if nc -z "$TEST_DB_HOST" "$TEST_DB_PORT" 2>/dev/null; then
            log_success "Database reachable: $TEST_DB_HOST:$TEST_DB_PORT"
        else
            log_warning "Database not reachable: $TEST_DB_HOST:$TEST_DB_PORT (may not be started yet)"
        fi
    else
        log_warning "Database connection variables not set"
    fi
    
    return 0
}

# Check memory and disk space
check_system_resources() {
    log "Checking system resources..."
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        available_mem=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
        log_success "Available memory: ${available_mem}GB"
    fi
    
    # Check disk space
    available_disk=$(df -h /app | awk 'NR==2{print $4}')
    log_success "Available disk space: $available_disk"
    
    return 0
}

# Check test framework functionality
check_test_framework() {
    log "Checking test framework functionality..."
    
    # Create a simple test file and run it
    cat > /tmp/health_test.py << 'EOF'
import pytest

def test_basic_functionality():
    assert True

def test_environment():
    import os
    assert os.getenv('TEST_MODE') == 'true'
    
def test_imports():
    # Test that we can import key modules
    from decimal import Decimal
    from datetime import datetime, UTC
    import json
    import asyncio
    assert True
EOF

    if python -m pytest /tmp/health_test.py -v --tb=short 2>/dev/null; then
        log_success "Test framework working correctly"
        rm -f /tmp/health_test.py
        return 0
    else
        log_error "Test framework not working correctly"
        rm -f /tmp/health_test.py
        return 1
    fi
}

# Main health check function
main() {
    log "Starting test environment health check..."
    
    local exit_code=0
    
    # Run all checks
    check_python_environment || exit_code=1
    check_test_directories || exit_code=1
    check_environment_variables || exit_code=1
    check_mock_services || exit_code=1
    check_database_connectivity || exit_code=1
    check_system_resources || exit_code=1
    check_test_framework || exit_code=1
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All health checks passed ✅"
        echo "HEALTHY"
    else
        log_error "Some health checks failed ❌"
        echo "UNHEALTHY"
    fi
    
    exit $exit_code
}

# Run health check
main "$@"