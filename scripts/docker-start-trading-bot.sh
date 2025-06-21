#!/bin/bash

# AI Trading Bot Docker Startup Script
# Comprehensive infrastructure management with service orchestration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# Logging functions
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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi

    success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."

    cd "$PROJECT_ROOT"

    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        warning ".env file not found, creating from example..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
        else
            error ".env.example not found. Please create .env file manually."
            exit 1
        fi
    fi

    # Set safe defaults in .env if not already set
    if ! grep -q "SYSTEM__DRY_RUN=true" "$ENV_FILE"; then
        log "Setting SYSTEM__DRY_RUN=true for safety..."
        sed -i.bak 's/SYSTEM__DRY_RUN=false/SYSTEM__DRY_RUN=true/' "$ENV_FILE" || true
    fi

    # Set Docker UID/GID for volume permissions
    if ! grep -q "HOST_UID=" "$ENV_FILE"; then
        echo "HOST_UID=$(id -u)" >> "$ENV_FILE"
        echo "HOST_GID=$(id -g)" >> "$ENV_FILE"
        log "Added Docker UID/GID configuration"
    fi

    success "Environment setup completed"
}

# Setup directories
setup_directories() {
    log "Setting up required directories..."

    cd "$PROJECT_ROOT"

    # Create required directories
    mkdir -p logs/{mcp,bluefin,dashboard}
    mkdir -p data/{mcp_memory,bluefin,trading}
    mkdir -p tmp
    mkdir -p config

    # Set proper permissions
    chmod 755 logs data tmp config
    chmod -R 755 logs/* data/* 2>/dev/null || true

    success "Directories setup completed"
}

# Clean up existing containers
cleanup_containers() {
    log "Cleaning up existing containers..."

    cd "$PROJECT_ROOT"

    # Stop and remove containers
    docker-compose down --remove-orphans 2>/dev/null || true

    # Clean up dangling images and networks
    docker system prune -f >/dev/null 2>&1 || true

    success "Container cleanup completed"
}

# Build services
build_services() {
    log "Building Docker services..."

    cd "$PROJECT_ROOT"

    # Build all services
    docker-compose build --no-cache --parallel || {
        error "Failed to build services"
        exit 1
    }

    success "Services built successfully"
}

# Start services in correct order
start_services() {
    log "Starting services in orchestrated order..."

    cd "$PROJECT_ROOT"

    # Start infrastructure services first
    log "Phase 1: Starting MCP infrastructure services..."
    docker-compose up -d mcp-memory mcp-omnisearch

    # Wait for MCP services to be healthy
    wait_for_service "mcp-memory" "8765" 60
    wait_for_service "mcp-omnisearch" "8767" 60

    # Start Bluefin service
    log "Phase 2: Starting Bluefin service..."
    docker-compose up -d bluefin-service
    wait_for_service "bluefin-service" "8081" 90  # External port

    # Start core trading bot
    log "Phase 3: Starting AI Trading Bot..."
    docker-compose up -d ai-trading-bot

    # Start dashboard services
    log "Phase 4: Starting Dashboard services..."
    docker-compose up -d dashboard-backend dashboard-frontend
    wait_for_service "dashboard-backend" "8000" 60

    success "All services started successfully"
}

# Wait for service to be healthy
wait_for_service() {
    local service_name=$1
    local port=$2
    local timeout=${3:-60}
    local count=0

    log "Waiting for $service_name to be healthy on port $port..."

    while [ $count -lt $timeout ]; do
        if curl -f -s --connect-timeout 5 "http://localhost:$port/health" >/dev/null 2>&1 || \
           curl -f -s --connect-timeout 5 "http://localhost:$port/" >/dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        sleep 2
        ((count += 2))

        # Show progress every 10 seconds
        if [ $((count % 10)) -eq 0 ]; then
            log "Still waiting for $service_name... (${count}s/${timeout}s)"
        fi
    done

    warning "$service_name did not become healthy within ${timeout}s"
    return 1
}

# Show service status
show_status() {
    log "Current service status:"
    echo
    docker-compose ps
    echo

    log "Service health checks:"
    for service_port in "mcp-memory:8765" "mcp-omnisearch:8767" "bluefin-service:8081" "dashboard-backend:8000"; do
        IFS=':' read -r service port <<< "$service_port"
        if curl -f -s --connect-timeout 3 "http://localhost:$port/health" >/dev/null 2>&1; then
            success "$service is healthy on port $port"
        else
            warning "$service is not responding on port $port"
        fi
    done
    echo

    log "Access URLs:"
    echo "  🚀 Dashboard Frontend: http://localhost:3000"
    echo "  📊 Dashboard API: http://localhost:8000"
    echo "  🔧 Bluefin Service: http://localhost:8081"
    echo "  🧠 MCP Memory: http://localhost:8765"
    echo "  🔍 MCP OmniSearch: http://localhost:8767"
    echo
}

# Show logs
show_logs() {
    local service=${1:-""}

    if [ -n "$service" ]; then
        log "Showing logs for $service:"
        docker-compose logs -f "$service"
    else
        log "Showing logs for all services:"
        docker-compose logs -f
    fi
}

# Main function
main() {
    local command=${1:-"start"}

    case $command in
        "start")
            log "Starting AI Trading Bot infrastructure..."
            check_prerequisites
            setup_environment
            setup_directories
            cleanup_containers
            build_services
            start_services
            show_status
            ;;
        "stop")
            log "Stopping AI Trading Bot infrastructure..."
            cd "$PROJECT_ROOT"
            docker-compose down
            success "All services stopped"
            ;;
        "restart")
            log "Restarting AI Trading Bot infrastructure..."
            check_prerequisites
            cd "$PROJECT_ROOT"
            docker-compose down
            sleep 5
            start_services
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "health")
            "$SCRIPT_DIR/docker-health-check.sh" check
            ;;
        "build")
            check_prerequisites
            setup_environment
            build_services
            ;;
        "clean")
            log "Cleaning up Docker infrastructure..."
            cd "$PROJECT_ROOT"
            docker-compose down --volumes --remove-orphans
            docker system prune -af
            docker volume prune -f
            success "Cleanup completed"
            ;;
        *)
            echo "AI Trading Bot Docker Management Script"
            echo
            echo "Usage: $0 {start|stop|restart|status|logs|health|build|clean}"
            echo
            echo "Commands:"
            echo "  start    - Start all services in correct order"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show service status and health"
            echo "  logs     - Show service logs (optionally specify service name)"
            echo "  health   - Run comprehensive health checks"
            echo "  build    - Build all Docker images"
            echo "  clean    - Clean up all containers, images, and volumes"
            echo
            echo "Examples:"
            echo "  $0 start                  # Start all services"
            echo "  $0 logs ai-trading-bot    # Show trading bot logs"
            echo "  $0 status                 # Check service status"
            echo
            exit 1
            ;;
    esac
}

main "$@"
