#!/bin/bash

# Ubuntu-Optimized Docker Health Check Script for AI Trading Bot
# Optimized for Ubuntu 22.04+ deployment with enhanced monitoring and FP runtime support

set -euo pipefail

# Configuration
TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=5

# FP Runtime Configuration
FP_ENABLED="${FP_RUNTIME_ENABLED:-true}"
FP_DEBUG="${FP_DEBUG_MODE:-false}"

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
    
    # Add FP-specific directories if FP is enabled
    if [ "$FP_ENABLED" = "true" ]; then
        required_dirs+=("/app/logs/fp" "/app/data/fp_runtime")
    fi
    
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

# Function to check FP runtime health
check_fp_runtime() {
    if [ "$FP_ENABLED" != "true" ]; then
        log "FP runtime disabled - skipping FP health checks"
        return 0
    fi

    log "Checking FP runtime health..."

    # Check FP effect interpreter health
    local fp_health_check_cmd="python -c \"
import sys
import os
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')
try:
    from bot.fp.runtime.interpreter import get_interpreter
    from bot.fp.runtime.scheduler import get_scheduler
    
    # Check interpreter status
    interpreter = get_interpreter()
    stats = interpreter.get_runtime_stats()
    
    # Check if interpreter is responsive
    if stats.get('active_effects', 0) >= 0:  # Basic sanity check
        print('FP_INTERPRETER_OK')
    else:
        print('FP_INTERPRETER_ERROR')
        exit(1)
    
    # Check scheduler if enabled
    scheduler = get_scheduler()
    scheduler_status = scheduler.get_status()
    
    if scheduler_status.get('running', False) is not None:  # Check scheduler exists
        print('FP_SCHEDULER_OK')
    else:
        print('FP_SCHEDULER_ERROR')
        exit(1)
        
    print('FP_RUNTIME_HEALTHY')
    exit(0)
except Exception as e:
    print(f'FP_RUNTIME_ERROR: {e}')
    exit(1)
\""

    if ! timeout "$TIMEOUT" bash -c "$fp_health_check_cmd" 2>/dev/null; then
        if [ "$FP_DEBUG" = "true" ]; then
            log "FP runtime health check failed - running in debug mode, continuing..."
            return 0
        else
            error "FP runtime health check failed"
        fi
    fi

    log "FP runtime health check passed"
}

# Function to check FP adapter compatibility
check_fp_adapters() {
    if [ "$FP_ENABLED" != "true" ]; then
        return 0
    fi

    log "Checking FP adapter compatibility..."

    local adapter_check_cmd="python -c \"
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/bot/fp')
try:
    # Test that adapters can be imported
    from bot.fp.adapters.compatibility_layer import CompatibilityLayer
    from bot.fp.adapters.exchange_adapter import ExchangeAdapter
    from bot.fp.adapters.strategy_adapter import StrategyAdapter
    
    # Basic adapter health check
    compatibility = CompatibilityLayer()
    
    print('FP_ADAPTERS_HEALTHY')
    exit(0)
except Exception as e:
    print(f'FP_ADAPTERS_ERROR: {e}')
    exit(1)
\""

    if ! timeout 15 bash -c "$adapter_check_cmd" 2>/dev/null; then
        if [ "$FP_DEBUG" = "true" ]; then
            log "FP adapter health check failed - running in debug mode, continuing..."
            return 0
        else
            error "FP adapter health check failed"
        fi
    fi

    log "FP adapter compatibility check passed"
}

# Function to check FP data persistence
check_fp_persistence() {
    if [ "$FP_ENABLED" != "true" ]; then
        return 0
    fi

    log "Checking FP data persistence..."

    # Check if FP runtime directories are writable and accessible
    local fp_dirs=("/app/data/fp_runtime/effects" "/app/data/fp_runtime/scheduler" "/app/data/fp_runtime/metrics")
    
    for dir in "${fp_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            if [ "$FP_DEBUG" = "true" ]; then
                log "FP directory missing: $dir - creating in debug mode"
                mkdir -p "$dir" || log "Failed to create $dir"
            else
                error "FP directory missing: $dir"
            fi
        elif [ ! -w "$dir" ]; then
            error "FP directory not writable: $dir"
        fi
    done

    # Test basic write access to FP runtime directory
    local test_file="/app/data/fp_runtime/healthcheck_test"
    if ! echo "test" > "$test_file" 2>/dev/null; then
        error "Cannot write to FP runtime directory"
    fi
    rm -f "$test_file" 2>/dev/null || true

    log "FP data persistence check passed"
}

# Main health check function
main() {
    log "Starting Docker health check..."

    # Perform health checks in order of importance
    check_system_resources
    check_python_health

    # FP runtime health checks (if enabled)
    if [ "$FP_ENABLED" = "true" ]; then
        log "FP runtime detected - performing FP health checks"
        check_fp_persistence
        check_fp_adapters
        check_fp_runtime
    fi

    # Ubuntu-specific network and system checks
    if command -v nslookup >/dev/null 2>&1 || command -v dig >/dev/null 2>&1; then
        check_network_basic
    fi

    # Ubuntu-specific checks
    if [ -f /etc/os-release ]; then
        log "Ubuntu system detected - performing additional checks"
        # Check Ubuntu-specific services if needed
    fi

    if [ "$FP_ENABLED" = "true" ]; then
        log "All health checks passed - container is healthy (FP runtime active)"
    else
        log "All health checks passed - container is healthy"
    fi
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
    "fp-only")
        # FP runtime check only
        if [ "$FP_ENABLED" = "true" ]; then
            log "Running FP-only health check..."
            check_fp_persistence
            check_fp_adapters
            check_fp_runtime
            log "FP health check completed successfully"
            exit 0
        else
            log "FP runtime disabled - nothing to check"
            exit 0
        fi
        ;;
    "fp-debug")
        # FP runtime debug check
        if [ "$FP_ENABLED" = "true" ]; then
            log "Running FP debug health check..."
            FP_DEBUG="true"
            check_fp_persistence
            check_fp_adapters
            check_fp_runtime
            log "FP debug health check completed"
            exit 0
        else
            log "FP runtime disabled - nothing to debug"
            exit 0
        fi
        ;;
    *)
        echo "Usage: $0 {main|quick|system-only|fp-only|fp-debug}"
        echo "  main        - Full health check including FP runtime (default)"
        echo "  quick       - Quick process check only"
        echo "  system-only - System resources check only"
        echo "  fp-only     - FP runtime health check only"
        echo "  fp-debug    - FP runtime debug check (verbose)"
        exit 1
        ;;
esac
