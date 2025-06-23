#!/bin/bash
#
# zero-downtime-deploy.sh - Blue-Green Deployment for AI Trading Bot
#
# This script implements a zero-downtime deployment strategy using blue-green deployment.
# It ensures continuous trading operations by running new containers alongside old ones,
# performing health checks, and switching over only when the new deployment is healthy.
#
# Features:
# - Blue-green deployment with automated switchover
# - Comprehensive health checking before transition
# - Gradual migration with monitoring
# - Automatic rollback on failure
# - State preservation between deployments
# - Detailed logging and monitoring
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
DEPLOYMENT_LOG="${PROJECT_ROOT}/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=5
MIGRATION_GRACE_PERIOD=30
ROLLBACK_TIMEOUT=300

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${DEPLOYMENT_LOG}"
}

# Print colored output
print_status() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "ERROR" "Docker Compose is not installed"
        exit 1
    fi

    # Check if compose file exists
    if [ ! -f "${COMPOSE_FILE}" ]; then
        log "ERROR" "Docker Compose file not found: ${COMPOSE_FILE}"
        exit 1
    fi

    # Check if .env file exists
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        log "WARN" ".env file not found. Using defaults."
    fi

    log "INFO" "Prerequisites check passed"
}

# Get current deployment color (blue or green)
get_current_color() {
    # Check for running containers with color labels
    local blue_running=$(docker ps --filter "label=deployment.color=blue" --filter "label=deployment.service=ai-trading-bot" -q | wc -l)
    local green_running=$(docker ps --filter "label=deployment.color=green" --filter "label=deployment.service=ai-trading-bot" -q | wc -l)

    if [ "$blue_running" -gt 0 ]; then
        echo "blue"
    elif [ "$green_running" -gt 0 ]; then
        echo "green"
    else
        # No deployment found, default to blue
        echo "none"
    fi
}

# Get target deployment color
get_target_color() {
    local current=$1
    if [ "$current" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Build new images
build_images() {
    local deployment_color=$1
    log "INFO" "Building images for ${deployment_color} deployment..."

    # Add build-time labels
    export BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    export VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    export VERSION="${deployment_color}-$(date +%Y%m%d-%H%M%S)"

    # Build with deployment color tag
    docker-compose -f "${COMPOSE_FILE}" build \
        --build-arg DEPLOYMENT_COLOR="${deployment_color}" \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${VCS_REF}" \
        --build-arg VERSION="${VERSION}" 2>&1 | tee -a "${DEPLOYMENT_LOG}"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR" "Failed to build images"
        return 1
    fi

    log "INFO" "Images built successfully"
    return 0
}

# Create deployment-specific compose override
create_deployment_compose() {
    local deployment_color=$1
    local override_file="${PROJECT_ROOT}/docker-compose.${deployment_color}.yml"

    log "INFO" "Creating deployment compose file for ${deployment_color}..."

    cat > "${override_file}" <<EOF
# Auto-generated deployment override for ${deployment_color} deployment
version: '3.8'

services:
  ai-trading-bot:
    container_name: ai-trading-bot-${deployment_color}
    labels:
      - "deployment.color=${deployment_color}"
      - "deployment.service=ai-trading-bot"
      - "deployment.version=${VERSION}"
      - "deployment.timestamp=${BUILD_DATE}"
    environment:
      - DEPLOYMENT_COLOR=${deployment_color}
      - DEPLOYMENT_MODE=blue-green

  bluefin-service:
    container_name: bluefin-service-${deployment_color}
    labels:
      - "deployment.color=${deployment_color}"
      - "deployment.service=bluefin-service"
      - "deployment.version=${VERSION}"

  mcp-memory:
    container_name: mcp-memory-${deployment_color}
    labels:
      - "deployment.color=${deployment_color}"
      - "deployment.service=mcp-memory"
      - "deployment.version=${VERSION}"

  mcp-omnisearch:
    container_name: mcp-omnisearch-${deployment_color}
    labels:
      - "deployment.color=${deployment_color}"
      - "deployment.service=mcp-omnisearch"
      - "deployment.version=${VERSION}"

  dashboard-backend:
    container_name: dashboard-backend-${deployment_color}
    labels:
      - "deployment.color=${deployment_color}"
      - "deployment.service=dashboard-backend"
      - "deployment.version=${VERSION}"

  dashboard-frontend:
    container_name: dashboard-frontend-${deployment_color}
    labels:
      - "deployment.color=${deployment_color}"
      - "deployment.service=dashboard-frontend"
      - "deployment.version=${VERSION}"
EOF

    log "INFO" "Deployment compose file created: ${override_file}"
}

# Start new deployment
start_deployment() {
    local deployment_color=$1
    local override_file="${PROJECT_ROOT}/docker-compose.${deployment_color}.yml"

    log "INFO" "Starting ${deployment_color} deployment..."

    # Start services with override
    docker-compose -f "${COMPOSE_FILE}" -f "${override_file}" up -d 2>&1 | tee -a "${DEPLOYMENT_LOG}"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "ERROR" "Failed to start ${deployment_color} deployment"
        return 1
    fi

    log "INFO" "${deployment_color} deployment started"
    return 0
}

# Health check a specific service
health_check_service() {
    local container_name=$1
    local retries=${2:-$HEALTH_CHECK_RETRIES}

    log "INFO" "Health checking ${container_name}..."

    for i in $(seq 1 $retries); do
        # Check if container is running
        if ! docker ps --filter "name=${container_name}" --filter "status=running" -q | grep -q .; then
            log "WARN" "${container_name} is not running (attempt $i/$retries)"
            sleep $HEALTH_CHECK_INTERVAL
            continue
        fi

        # Check container health status
        local health_status=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "none")

        if [ "$health_status" = "healthy" ]; then
            log "INFO" "${container_name} is healthy"
            return 0
        elif [ "$health_status" = "none" ]; then
            # No health check defined, check if container is running for at least 10 seconds
            local running_time=$(docker inspect --format='{{.State.StartedAt}}' "${container_name}" 2>/dev/null)
            if [ -n "$running_time" ]; then
                local started_seconds=$(date -d "$running_time" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$running_time" +%s 2>/dev/null || echo 0)
                local current_seconds=$(date +%s)
                local uptime=$((current_seconds - started_seconds))

                if [ $uptime -gt 10 ]; then
                    log "INFO" "${container_name} has been running for ${uptime} seconds (no health check defined)"
                    return 0
                fi
            fi
        fi

        log "WARN" "${container_name} health check failed (attempt $i/$retries, status: $health_status)"
        sleep $HEALTH_CHECK_INTERVAL
    done

    log "ERROR" "${container_name} failed health check after $retries attempts"
    return 1
}

# Health check entire deployment
health_check_deployment() {
    local deployment_color=$1

    log "INFO" "Running health checks for ${deployment_color} deployment..."

    local services=(
        "ai-trading-bot-${deployment_color}"
        "bluefin-service-${deployment_color}"
        "mcp-memory-${deployment_color}"
        "mcp-omnisearch-${deployment_color}"
        "dashboard-backend-${deployment_color}"
        "dashboard-frontend-${deployment_color}"
    )

    local failed=0
    for service in "${services[@]}"; do
        if ! health_check_service "$service"; then
            ((failed++))
        fi
    done

    if [ $failed -gt 0 ]; then
        log "ERROR" "${failed} services failed health check"
        return 1
    fi

    log "INFO" "All services are healthy"
    return 0
}

# Gracefully stop trading on old deployment
stop_trading() {
    local deployment_color=$1
    local container_name="ai-trading-bot-${deployment_color}"

    log "INFO" "Stopping trading on ${deployment_color} deployment..."

    # Send graceful shutdown signal to trading bot
    if docker ps --filter "name=${container_name}" -q | grep -q .; then
        # First, try to close any open positions
        log "INFO" "Attempting to close open positions..."
        docker exec "${container_name}" python -m bot.utils.graceful_shutdown || true

        # Give bot time to close positions
        sleep 10

        # Stop the container gracefully
        docker stop -t 30 "${container_name}" 2>&1 | tee -a "${DEPLOYMENT_LOG}"
    fi

    log "INFO" "Trading stopped on ${deployment_color} deployment"
}

# Monitor deployment for errors
monitor_deployment() {
    local deployment_color=$1
    local duration=$2

    log "INFO" "Monitoring ${deployment_color} deployment for ${duration} seconds..."

    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    local error_count=0

    while [ $(date +%s) -lt $end_time ]; do
        # Check for container restarts
        local restart_count=$(docker ps --filter "label=deployment.color=${deployment_color}" --format '{{.Names}} {{.Status}}' | grep -c "Restarted" || true)
        if [ $restart_count -gt 0 ]; then
            log "WARN" "Detected $restart_count container restarts"
            ((error_count += restart_count))
        fi

        # Check logs for errors
        local log_errors=$(docker-compose -f "${COMPOSE_FILE}" -f "${PROJECT_ROOT}/docker-compose.${deployment_color}.yml" logs --tail=100 2>&1 | grep -iE "error|exception|critical" | wc -l || true)
        if [ $log_errors -gt 0 ]; then
            log "WARN" "Detected $log_errors errors in logs"
            ((error_count += log_errors))
        fi

        # Check if all containers are still running
        local expected_containers=6
        local running_containers=$(docker ps --filter "label=deployment.color=${deployment_color}" -q | wc -l)
        if [ $running_containers -lt $expected_containers ]; then
            log "ERROR" "Only $running_containers/$expected_containers containers are running"
            return 1
        fi

        sleep 5
    done

    if [ $error_count -gt 0 ]; then
        log "WARN" "Detected $error_count errors during monitoring period"
    else
        log "INFO" "No errors detected during monitoring period"
    fi

    return 0
}

# Switch traffic to new deployment
switch_traffic() {
    local from_color=$1
    local to_color=$2

    log "INFO" "Switching traffic from ${from_color} to ${to_color}..."

    # For trading bot, we ensure only one is active at a time
    # First, ensure new deployment is ready
    if ! health_check_deployment "$to_color"; then
        log "ERROR" "New deployment is not healthy, aborting switch"
        return 1
    fi

    # Stop trading on old deployment
    stop_trading "$from_color"

    # Update any external references (e.g., monitoring systems)
    # This is where you would update load balancers, DNS, etc.
    # For the trading bot, we might update a state file or database
    echo "$to_color" > "${PROJECT_ROOT}/.current_deployment"

    log "INFO" "Traffic switched to ${to_color} deployment"
    return 0
}

# Cleanup old deployment
cleanup_deployment() {
    local deployment_color=$1

    log "INFO" "Cleaning up ${deployment_color} deployment..."

    # Stop and remove containers
    docker-compose -f "${COMPOSE_FILE}" -f "${PROJECT_ROOT}/docker-compose.${deployment_color}.yml" down 2>&1 | tee -a "${DEPLOYMENT_LOG}"

    # Remove override file
    rm -f "${PROJECT_ROOT}/docker-compose.${deployment_color}.yml"

    log "INFO" "${deployment_color} deployment cleaned up"
}

# Rollback to previous deployment
rollback() {
    local from_color=$1
    local to_color=$2

    print_status "$RED" "ROLLING BACK TO ${to_color} DEPLOYMENT"
    log "ERROR" "Rollback initiated from ${from_color} to ${to_color}"

    # Stop problematic deployment
    docker-compose -f "${COMPOSE_FILE}" -f "${PROJECT_ROOT}/docker-compose.${from_color}.yml" stop 2>&1 | tee -a "${DEPLOYMENT_LOG}"

    # Restart old deployment if it was stopped
    docker-compose -f "${COMPOSE_FILE}" -f "${PROJECT_ROOT}/docker-compose.${to_color}.yml" start 2>&1 | tee -a "${DEPLOYMENT_LOG}"

    # Update current deployment marker
    echo "$to_color" > "${PROJECT_ROOT}/.current_deployment"

    # Cleanup failed deployment
    cleanup_deployment "$from_color"

    log "INFO" "Rollback completed"
}

# Main deployment flow
main() {
    print_status "$BLUE" "=== ZERO-DOWNTIME DEPLOYMENT STARTING ==="
    log "INFO" "Deployment started by user: $(whoami)"

    # Check prerequisites
    check_prerequisites

    # Determine current and target deployments
    local current_color=$(get_current_color)
    local target_color=$(get_target_color "$current_color")

    if [ "$current_color" = "none" ]; then
        print_status "$YELLOW" "No existing deployment found. Starting fresh with ${target_color}."
        current_color="blue"
        target_color="green"
    else
        print_status "$GREEN" "Current deployment: ${current_color}"
        print_status "$GREEN" "Target deployment: ${target_color}"
    fi

    # Save current state for potential rollback
    local rollback_file="${PROJECT_ROOT}/.rollback_state"
    echo "CURRENT=${current_color}" > "$rollback_file"
    echo "TARGET=${target_color}" >> "$rollback_file"
    echo "TIMESTAMP=$(date +%s)" >> "$rollback_file"

    # Build new images
    if ! build_images "$target_color"; then
        print_status "$RED" "Build failed! Aborting deployment."
        exit 1
    fi

    # Create deployment-specific compose file
    create_deployment_compose "$target_color"

    # Start new deployment
    print_status "$BLUE" "Starting ${target_color} deployment..."
    if ! start_deployment "$target_color"; then
        print_status "$RED" "Failed to start new deployment!"
        cleanup_deployment "$target_color"
        exit 1
    fi

    # Health check new deployment
    print_status "$BLUE" "Running health checks..."
    if ! health_check_deployment "$target_color"; then
        print_status "$RED" "Health checks failed!"
        rollback "$target_color" "$current_color"
        exit 1
    fi

    # Monitor new deployment
    print_status "$BLUE" "Monitoring new deployment for stability..."
    if ! monitor_deployment "$target_color" "$MIGRATION_GRACE_PERIOD"; then
        print_status "$RED" "Deployment monitoring detected issues!"
        rollback "$target_color" "$current_color"
        exit 1
    fi

    # Switch traffic
    print_status "$BLUE" "Switching to new deployment..."
    if ! switch_traffic "$current_color" "$target_color"; then
        print_status "$RED" "Failed to switch traffic!"
        rollback "$target_color" "$current_color"
        exit 1
    fi

    # Final monitoring period
    print_status "$BLUE" "Final monitoring period..."
    if ! monitor_deployment "$target_color" "$MIGRATION_GRACE_PERIOD"; then
        print_status "$YELLOW" "Issues detected after switch, but keeping new deployment"
        log "WARN" "Post-switch monitoring detected issues"
    fi

    # Keep old deployment running for quick rollback
    print_status "$YELLOW" "Keeping ${current_color} deployment stopped but available for rollback"
    print_status "$YELLOW" "To rollback: $0 --rollback"
    print_status "$YELLOW" "To cleanup old deployment: $0 --cleanup ${current_color}"

    # Success
    print_status "$GREEN" "=== DEPLOYMENT SUCCESSFUL ==="
    log "INFO" "Deployment completed successfully"

    # Show deployment summary
    echo
    print_status "$BLUE" "Deployment Summary:"
    echo "  Old Version: ${current_color}"
    echo "  New Version: ${target_color}"
    echo "  Duration: $(($(date +%s) - $(date -r "$rollback_file" +%s))) seconds"
    echo "  Log File: ${DEPLOYMENT_LOG}"
    echo
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    --rollback)
        if [ -f "${PROJECT_ROOT}/.rollback_state" ]; then
            source "${PROJECT_ROOT}/.rollback_state"
            rollback "$TARGET" "$CURRENT"
        else
            print_status "$RED" "No rollback state found"
            exit 1
        fi
        ;;
    --cleanup)
        if [ -z "${2:-}" ]; then
            print_status "$RED" "Usage: $0 --cleanup <color>"
            exit 1
        fi
        cleanup_deployment "$2"
        ;;
    --status)
        current=$(get_current_color)
        if [ "$current" = "none" ]; then
            print_status "$YELLOW" "No active deployment"
        else
            print_status "$GREEN" "Active deployment: $current"
            docker ps --filter "label=deployment.color=$current" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        fi
        ;;
    --help)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy       - Deploy new version with zero downtime (default)"
        echo "  --rollback   - Rollback to previous deployment"
        echo "  --cleanup    - Cleanup a specific deployment (blue/green)"
        echo "  --status     - Show current deployment status"
        echo "  --help       - Show this help message"
        ;;
    *)
        print_status "$RED" "Unknown command: $1"
        exit 1
        ;;
esac
