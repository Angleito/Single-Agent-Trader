#!/bin/bash

# Docker Infrastructure Health Check Script
# Ensures all critical services are running and connected properly

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is running
check_docker() {
    log "Checking Docker daemon..."
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    success "Docker daemon is running"
}

# Check if docker-compose is available
check_compose() {
    log "Checking Docker Compose..."
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "docker-compose is not installed or not in PATH"
        exit 1
    fi
    success "Docker Compose is available"
}

# Check network connectivity
check_networks() {
    log "Checking Docker networks..."
    if docker network ls | grep -q "trading-network"; then
        success "Trading network exists"
    else
        warning "Trading network does not exist, will be created on startup"
    fi
}

# Check service health
check_service_health() {
    local service_name=$1
    local port=$2
    local timeout=${3:-30}

    log "Checking health of $service_name on port $port..."

    # Wait for container to be running
    local container_name="${service_name//-/_}-server"
    if [ "$service_name" = "dashboard-backend" ]; then
        container_name="dashboard-backend"
    elif [ "$service_name" = "bluefin-service" ]; then
        container_name="bluefin-service"
    elif [ "$service_name" = "ai-trading-bot" ]; then
        container_name="ai-trading-bot"
    fi

    local count=0
    while [ $count -lt $timeout ]; do
        if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
            # Check if service responds
            if curl -f -s "http://localhost:$port/health" >/dev/null 2>&1 || \
               curl -f -s "http://localhost:$port/" >/dev/null 2>&1; then
                success "$service_name is healthy on port $port"
                return 0
            fi
        fi
        sleep 1
        ((count++))
    done

    warning "$service_name is not responding on port $port after ${timeout}s"
    return 1
}

# Start services in correct order
start_services() {
    log "Starting Docker services in correct order..."

    # First, start infrastructure services
    log "Starting MCP services..."
    docker-compose up -d mcp-memory mcp-omnisearch

    # Wait for MCP services to be healthy
    sleep 10

    # Start core trading services
    log "Starting core trading services..."
    docker-compose up -d bluefin-service

    # Wait for bluefin service
    sleep 15

    # Start AI trading bot
    log "Starting AI trading bot..."
    docker-compose up -d ai-trading-bot

    # Start dashboard services
    log "Starting dashboard services..."
    docker-compose up -d dashboard-backend dashboard-frontend

    success "All services started"
}

# Main health check function
run_health_checks() {
    log "Running comprehensive health checks..."

    local failed_services=()

    # Check core services
    if ! check_service_health "mcp-memory" "8765" 30; then
        failed_services+=("mcp-memory")
    fi

    if ! check_service_health "mcp-omnisearch" "8767" 30; then
        failed_services+=("mcp-omnisearch")
    fi

    if ! check_service_health "bluefin-service" "8081" 30; then  # Note: external port is 8081
        failed_services+=("bluefin-service")
    fi

    if ! check_service_health "dashboard-backend" "8000" 30; then
        failed_services+=("dashboard-backend")
    fi

    # Report results
    if [ ${#failed_services[@]} -eq 0 ]; then
        success "All critical services are healthy!"
        return 0
    else
        error "The following services are not healthy: ${failed_services[*]}"
        return 1
    fi
}

# Show service status
show_status() {
    log "Current service status:"
    docker-compose ps

    log "Network information:"
    docker network ls | grep trading

    log "Service logs (last 5 lines each):"
    for service in mcp-memory mcp-omnisearch bluefin-service dashboard-backend ai-trading-bot; do
        echo "--- $service ---"
        docker-compose logs --tail=5 "$service" 2>/dev/null || echo "Service not running"
    done
}

# Main execution
main() {
    log "Starting Docker Infrastructure Health Check..."

    check_docker
    check_compose
    check_networks

    case "${1:-check}" in
        "start")
            start_services
            sleep 30  # Give services time to start
            run_health_checks
            ;;
        "check")
            run_health_checks
            ;;
        "status")
            show_status
            ;;
        "restart")
            log "Restarting all services..."
            docker-compose down
            sleep 5
            start_services
            sleep 30
            run_health_checks
            ;;
        *)
            echo "Usage: $0 {start|check|status|restart}"
            echo "  start   - Start services in correct order and check health"
            echo "  check   - Check health of running services"
            echo "  status  - Show current status and logs"
            echo "  restart - Restart all services"
            exit 1
            ;;
    esac
}

main "$@"
