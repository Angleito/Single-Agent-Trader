#!/bin/bash
#
# deployment-health-check.sh - Quick health check for deployments
#
# This script performs a quick health check of the deployment
# and returns appropriate exit codes for use in automation.
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Exit codes
EXIT_SUCCESS=0
EXIT_WARNING=1
EXIT_CRITICAL=2

# Check if deployment color is provided
DEPLOYMENT_COLOR="${1:-}"

# Auto-detect deployment color if not provided
if [ -z "$DEPLOYMENT_COLOR" ]; then
    if docker ps --filter "label=deployment.color=blue" -q | grep -q .; then
        DEPLOYMENT_COLOR="blue"
    elif docker ps --filter "label=deployment.color=green" -q | grep -q .; then
        DEPLOYMENT_COLOR="green"
    else
        echo "ERROR: No active deployment found"
        exit $EXIT_CRITICAL
    fi
fi

# Health check functions
check_container_running() {
    local container=$1
    docker ps --filter "name=${container}" --filter "status=running" -q | grep -q .
}

check_container_health() {
    local container=$1
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "none")
    [ "$health" = "healthy" ] || [ "$health" = "none" ]
}

check_no_restarts() {
    local container=$1
    local restarts=$(docker inspect --format='{{.RestartCount}}' "$container" 2>/dev/null || echo "0")
    [ "$restarts" -eq 0 ]
}

check_no_errors() {
    local container=$1
    ! docker logs "$container" 2>&1 | tail -50 | grep -qi "error\|exception\|critical"
}

# Main health check
main() {
    local exit_code=$EXIT_SUCCESS
    local warnings=0
    local errors=0

    # Define services to check
    local services=(
        "ai-trading-bot-${DEPLOYMENT_COLOR}"
        "bluefin-service-${DEPLOYMENT_COLOR}"
        "mcp-memory-${DEPLOYMENT_COLOR}"
        "mcp-omnisearch-${DEPLOYMENT_COLOR}"
        "dashboard-backend-${DEPLOYMENT_COLOR}"
        "dashboard-frontend-${DEPLOYMENT_COLOR}"
    )

    echo "Checking health of ${DEPLOYMENT_COLOR} deployment..."

    for service in "${services[@]}"; do
        echo -n "  ${service}: "

        if ! check_container_running "$service"; then
            echo "NOT RUNNING"
            ((errors++))
            continue
        fi

        if ! check_container_health "$service"; then
            echo "UNHEALTHY"
            ((errors++))
            continue
        fi

        if ! check_no_restarts "$service"; then
            echo "RESTARTED"
            ((warnings++))
            continue
        fi

        if ! check_no_errors "$service"; then
            echo "ERRORS IN LOGS"
            ((warnings++))
            continue
        fi

        echo "OK"
    done

    # Determine exit code
    if [ $errors -gt 0 ]; then
        exit_code=$EXIT_CRITICAL
        echo "CRITICAL: $errors services have critical issues"
    elif [ $warnings -gt 0 ]; then
        exit_code=$EXIT_WARNING
        echo "WARNING: $warnings services have warnings"
    else
        echo "SUCCESS: All services healthy"
    fi

    exit $exit_code
}

# Run health check
main
