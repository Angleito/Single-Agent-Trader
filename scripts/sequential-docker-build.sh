#!/bin/bash

# Sequential Docker Build Script for Low-Memory Systems
# This script builds Docker services one at a time to avoid out-of-memory issues

set -e  # Exit on any error

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

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker-compose is available
if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    error "Docker Compose is not installed"
    exit 1
fi

# Determine docker compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

log "Using Docker Compose command: $DOCKER_COMPOSE"

# Check memory and add swap if needed
check_memory() {
    log "Checking system memory..."
    
    # Get available memory in MB
    AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    SWAP_MEM=$(free -m | awk 'NR==3{print $2}')
    
    log "Total Memory: ${TOTAL_MEM}MB, Available: ${AVAILABLE_MEM}MB, Swap: ${SWAP_MEM}MB"
    
    # If total memory is less than 2GB and no swap, recommend adding swap
    if [ "$TOTAL_MEM" -lt 2048 ] && [ "$SWAP_MEM" -eq 0 ]; then
        warning "Low memory detected (${TOTAL_MEM}MB). Consider adding swap:"
        echo "  sudo fallocate -l 2G /swapfile"
        echo "  sudo chmod 600 /swapfile"
        echo "  sudo mkswap /swapfile"
        echo "  sudo swapon /swapfile"
        echo ""
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Clean up any existing containers/images to free memory
cleanup_docker() {
    log "Cleaning up Docker to free memory..."
    
    # Stop all containers
    $DOCKER_COMPOSE down --remove-orphans 2>/dev/null || true
    
    # Remove unused containers, networks, images
    docker system prune -f 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f 2>/dev/null || true
    
    success "Docker cleanup completed"
}

# Build a single service with memory monitoring
build_service() {
    local service_name=$1
    log "Building service: $service_name"
    
    # Monitor memory before build
    local mem_before=$(free -m | awk 'NR==2{print $3}')
    
    # Build the service with limited resources
    if $DOCKER_COMPOSE build \
        --parallel 1 \
        --memory 512m \
        "$service_name" 2>&1 | tee "build_${service_name}.log"; then
        
        success "Successfully built $service_name"
        
        # Clean up build logs older than 1 day
        find . -name "build_*.log" -mtime +1 -delete 2>/dev/null || true
        
    else
        error "Failed to build $service_name"
        error "Check build_${service_name}.log for details"
        return 1
    fi
    
    # Monitor memory after build
    local mem_after=$(free -m | awk 'NR==2{print $3}')
    local mem_used=$((mem_after - mem_before))
    log "Memory used for $service_name build: ${mem_used}MB"
    
    # Brief pause to let memory settle
    sleep 5
}

# Get list of services from docker-compose.yml
get_services() {
    if [ -f "docker-compose.yml" ]; then
        # Extract service names from docker-compose.yml
        $DOCKER_COMPOSE config --services 2>/dev/null || {
            # Fallback: parse YAML manually
            grep -E "^  [a-zA-Z]" docker-compose.yml | grep -v "^  #" | sed 's/://g' | sed 's/^  //'
        }
    else
        error "docker-compose.yml not found"
        exit 1
    fi
}

# Main build sequence
main() {
    log "Starting sequential Docker build process..."
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi
    
    # Check system resources
    check_memory
    
    # Clean up Docker to free memory
    cleanup_docker
    
    # Get list of services
    log "Detecting services from docker-compose.yml..."
    services=$(get_services)
    
    if [ -z "$services" ]; then
        error "No services found in docker-compose.yml"
        exit 1
    fi
    
    log "Found services: $(echo $services | tr '\n' ' ')"
    
    # Build services one by one in optimal order
    # Core services first, then optional services
    declare -a build_order=(
        "bluefin-service"      # Build first (smallest, needed by others)
        "ai-trading-bot"       # Main application
        "dashboard-backend"    # Backend API
        "dashboard-frontend"   # Frontend (large Node.js build)
        "mcp-memory"          # Memory service
        "mcp-omnisearch"      # Search service (largest build)
    )
    
    local built_count=0
    local total_services=$(echo "$services" | wc -l)
    
    # Build services in order of priority
    for service in "${build_order[@]}"; do
        if echo "$services" | grep -q "^${service}$"; then
            log "Building service $((built_count + 1))/$total_services: $service"
            
            if build_service "$service"; then
                built_count=$((built_count + 1))
                success "Progress: $built_count/$total_services services built"
            else
                error "Build failed for $service. Stopping."
                exit 1
            fi
            
            # Clean up Docker layers between builds to save memory
            docker builder prune -f 2>/dev/null || true
        fi
    done
    
    # Build any remaining services not in the priority list
    for service in $services; do
        if ! printf '%s\n' "${build_order[@]}" | grep -q "^${service}$"; then
            log "Building remaining service: $service"
            
            if build_service "$service"; then
                built_count=$((built_count + 1))
                success "Progress: $built_count/$total_services services built"
            else
                warning "Failed to build optional service: $service"
            fi
        fi
    done
    
    success "All Docker services built successfully!"
    log "Starting services..."
    
    # Start the services
    if $DOCKER_COMPOSE up -d; then
        success "All services started successfully!"
        
        # Show status
        log "Service status:"
        $DOCKER_COMPOSE ps
        
        log "To view logs: $DOCKER_COMPOSE logs -f"
        log "To stop services: $DOCKER_COMPOSE down"
        
    else
        error "Failed to start services"
        exit 1
    fi
}

# Handle script interruption
trap 'error "Build interrupted"; exit 1' INT TERM

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Sequential Docker Build Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --clean        Clean Docker and rebuild everything"
        echo "  --no-start     Build only, don't start services"
        echo ""
        echo "This script builds Docker services one at a time to avoid"
        echo "out-of-memory issues on systems with limited RAM."
        exit 0
        ;;
    --clean)
        log "Performing clean rebuild..."
        $DOCKER_COMPOSE down --volumes --remove-orphans 2>/dev/null || true
        docker system prune -af 2>/dev/null || true
        ;;
    --no-start)
        NO_START=true
        ;;
esac

# Run main function
main

log "Sequential Docker build completed!"