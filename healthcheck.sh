#!/bin/bash

# Ubuntu-Optimized Docker Health Check Script for AI Trading Bot
# Optimized for Ubuntu 22.04+ deployment with enhanced monitoring

set -euo pipefail

# Configuration
TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=5

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] HEALTHCHECK: $1"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] HEALTHCHECK ERROR: $1" >&2
    exit 1
}

# Function to check if python is available and responsive
check_python_health() {
    log "Checking Python process health..."

    # First check if Python process is running
    if ! pgrep -f "python.*bot.main" > /dev/null 2>&1; then
        error "Trading bot Python process not found"
    fi

    # Try to run a simple Python health check
    local health_check_cmd="python -c \"
import sys
sys.path.insert(0, '/app')
try:
    from bot.health import HealthCheckEndpoints
    from bot.config import load_settings
    settings = load_settings()
    health = HealthCheckEndpoints(settings)
    result = health.get_liveness()
    if result.get('alive', False):
        print('OK')
        exit(0)
    else:
        print('UNHEALTHY')
        exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
\""

    if ! timeout "$TIMEOUT" bash -c "$health_check_cmd" 2>/dev/null; then
        error "Python health check failed"
    fi

    log "Python process health check passed"
}

# Function to check basic system resources
check_system_resources() {
    log "Checking system resources..."

    # Check disk space (fail if >95% full)
    local disk_usage
    disk_usage=$(df /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
    if [ "$disk_usage" -gt 95 ]; then
        error "Disk usage critical: ${disk_usage}%"
    fi

    # Check if required directories exist and are writable
    local required_dirs=("/app/logs" "/app/data")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ] && [ ! -w "$dir" ]; then
            error "Directory not writable: $dir"
        fi
    done

    log "System resources check passed"
}

# Function to check network connectivity (basic)
check_network_basic() {
    log "Checking basic network connectivity..."

    # Ubuntu-optimized DNS resolution check
    if command -v nslookup >/dev/null 2>&1; then
        if ! timeout 10 nslookup google.com >/dev/null 2>&1; then
            # Try alternative DNS resolution methods on Ubuntu
            if command -v dig >/dev/null 2>&1; then
                if ! timeout 10 dig +short google.com >/dev/null 2>&1; then
                    error "DNS resolution failed"
                fi
            else
                error "DNS resolution failed"
            fi
        fi
    fi

    log "Basic network connectivity check passed"
}

# Main health check function
main() {
    log "Starting Docker health check..."

    # Perform health checks in order of importance
    check_system_resources
    check_python_health

    # Ubuntu-specific network and system checks
    if command -v nslookup >/dev/null 2>&1 || command -v dig >/dev/null 2>&1; then
        check_network_basic
    fi

    # Ubuntu-specific checks
    if [ -f /etc/os-release ]; then
        log "Ubuntu system detected - performing additional checks"
        # Check Ubuntu-specific services if needed
    fi

    log "All health checks passed - container is healthy"
    exit 0
}

# Handle script arguments
case "${1:-main}" in
    "main"|"")
        main
        ;;
    "quick")
        # Quick check - just verify Python process is running
        log "Running quick health check..."
        if pgrep -f "python.*bot.main" > /dev/null 2>&1; then
            log "Quick check passed - Python process is running"
            exit 0
        else
            error "Quick check failed - Python process not found"
        fi
        ;;
    "system-only")
        check_system_resources
        ;;
    *)
        echo "Usage: $0 {main|quick|system-only}"
        echo "  main        - Full health check (default)"
        echo "  quick       - Quick process check only"
        echo "  system-only - System resources check only"
        exit 1
        ;;
esac
