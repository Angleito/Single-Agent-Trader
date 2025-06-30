#!/bin/bash
"""
Container Startup Script with Integrated Monitoring
Starts the main application alongside resource monitoring
"""

set -euo pipefail

# Configuration
CONTAINER_NAME=${CONTAINER_NAME:-$(hostname)}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-30}
ENABLE_MONITORING=${ENABLE_MONITORING:-true}
METRICS_PORT=${METRICS_PORT:-9090}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_debug() {
    if [[ "${LOG_LEVEL}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $*" >&2
    fi
}

# Function to check if monitoring is enabled
is_monitoring_enabled() {
    [[ "${ENABLE_MONITORING,,}" == "true" ]]
}

# Function to start monitoring in background
start_monitoring() {
    if ! is_monitoring_enabled; then
        log_info "Monitoring disabled, skipping..."
        return 0
    fi

    log_info "Starting container monitoring for: ${CONTAINER_NAME}"

    # Check if Python and required packages are available
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found, monitoring disabled"
        return 1
    fi

    # Check if monitoring script exists
    MONITOR_SCRIPT="/app/scripts/container-monitor.py"
    if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
        log_error "Monitoring script not found at ${MONITOR_SCRIPT}"
        return 1
    fi

    # Create logs directory
    mkdir -p /app/logs

    # Set monitoring environment variables
    export CONTAINER_NAME="${CONTAINER_NAME}"
    export MONITOR_INTERVAL="${MONITOR_INTERVAL}"
    export METRICS_PORT="${METRICS_PORT}"
    export LOG_LEVEL="${LOG_LEVEL}"

    # Start monitoring in background
    log_info "Launching monitoring process (interval: ${MONITOR_INTERVAL}s, port: ${METRICS_PORT})"

    # Run monitoring with proper error handling
    (
        cd /app
        python3 "${MONITOR_SCRIPT}" 2>&1 | while IFS= read -r line; do
            echo "[MONITOR] $line"
        done
    ) &

    MONITOR_PID=$!

    # Store PID for cleanup
    echo "$MONITOR_PID" > /tmp/monitor.pid

    log_info "Monitoring started with PID: ${MONITOR_PID}"

    # Give monitoring a moment to start
    sleep 2

    # Verify monitoring is running
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        log_info "Monitoring process confirmed running"

        # Test metrics endpoint if enabled
        if command -v curl &> /dev/null; then
            log_debug "Testing metrics endpoint..."
            if curl -f -s "http://localhost:${METRICS_PORT}/health" > /dev/null 2>&1; then
                log_info "Metrics endpoint available at http://localhost:${METRICS_PORT}"
            else
                log_warn "Metrics endpoint not responding yet (may take a moment to start)"
            fi
        fi
    else
        log_error "Monitoring process failed to start"
        return 1
    fi
}

# Function to stop monitoring
stop_monitoring() {
    if [[ -f /tmp/monitor.pid ]]; then
        local monitor_pid
        monitor_pid=$(cat /tmp/monitor.pid)
        log_info "Stopping monitoring process (PID: ${monitor_pid})"

        if kill -TERM "$monitor_pid" 2>/dev/null; then
            # Wait for graceful shutdown
            sleep 2

            # Force kill if still running
            if kill -0 "$monitor_pid" 2>/dev/null; then
                log_warn "Monitoring process didn't stop gracefully, force killing..."
                kill -KILL "$monitor_pid" 2>/dev/null || true
            fi
        fi

        rm -f /tmp/monitor.pid
        log_info "Monitoring stopped"
    fi
}

# Function to setup signal handlers
setup_signal_handlers() {
    # Trap signals and cleanup
    trap 'log_info "Received SIGTERM, shutting down..."; stop_monitoring; exit 0' SIGTERM
    trap 'log_info "Received SIGINT, shutting down..."; stop_monitoring; exit 0' SIGINT
    trap 'log_info "Received SIGHUP, restarting monitoring..."; stop_monitoring; start_monitoring' SIGHUP
}

# Function to run health checks
run_health_checks() {
    log_info "Running pre-startup health checks..."

    # Check disk space
    local disk_usage
    disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        log_error "Disk usage is ${disk_usage}%, which is critically high"
        return 1
    elif [[ $disk_usage -gt 80 ]]; then
        log_warn "Disk usage is ${disk_usage}%, consider cleanup"
    fi

    # Check memory
    local mem_usage
    if command -v free &> /dev/null; then
        mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [[ $mem_usage -gt 90 ]]; then
            log_error "Memory usage is ${mem_usage}%, which is critically high"
            return 1
        elif [[ $mem_usage -gt 80 ]]; then
            log_warn "Memory usage is ${mem_usage}%, monitor closely"
        fi
    fi

    # Check if required directories exist
    for dir in /app/logs /app/data; do
        if [[ ! -d "$dir" ]]; then
            log_debug "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done

    log_info "Health checks passed"
    return 0
}

# Function to log system information
log_system_info() {
    log_info "=== Container System Information ==="
    log_info "Container: ${CONTAINER_NAME}"
    log_info "Hostname: $(hostname)"
    log_info "Uptime: $(uptime | cut -d',' -f1)"

    if command -v free &> /dev/null; then
        local total_mem
        total_mem=$(free -h | grep Mem | awk '{print $2}')
        log_info "Total Memory: ${total_mem}"
    fi

    if command -v df &> /dev/null; then
        local disk_space
        disk_space=$(df -h / | tail -1 | awk '{print $2}')
        log_info "Disk Space: ${disk_space}"
    fi

    if command -v nproc &> /dev/null; then
        local cpu_cores
        cpu_cores=$(nproc)
        log_info "CPU Cores: ${cpu_cores}"
    fi

    log_info "Python Version: $(python3 --version 2>/dev/null || echo 'Not available')"
    log_info "=== End System Information ==="
}

# Main function
main() {
    log_info "Starting container with monitoring support"
    log_info "Container: ${CONTAINER_NAME}"

    # Setup signal handlers
    setup_signal_handlers

    # Log system information
    log_system_info

    # Run health checks
    if ! run_health_checks; then
        log_error "Health checks failed, aborting startup"
        exit 1
    fi

    # Start monitoring if enabled
    if is_monitoring_enabled; then
        if ! start_monitoring; then
            log_error "Failed to start monitoring"
            # Continue without monitoring rather than failing completely
            log_warn "Continuing without monitoring..."
        fi
    else
        log_info "Monitoring disabled by configuration"
    fi

    # Execute the main command passed as arguments
    if [[ $# -eq 0 ]]; then
        log_error "No command specified to run"
        log_info "Usage: $0 <command> [args...]"
        exit 1
    fi

    log_info "Starting main application: $*"

    # Execute the main command
    # Use exec to replace the shell process
    exec "$@"
}

# Execute main function with all arguments
main "$@"
