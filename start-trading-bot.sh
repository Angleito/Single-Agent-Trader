#!/bin/bash

# Start Trading Bot Script for Low-Memory Systems
# Optimized for 1GB DigitalOcean droplets

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    error "docker-compose.yml not found. Please run from project root."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    warning ".env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warning "Please edit .env with your API keys before starting:"
        echo "  nano .env"
        exit 1
    else
        error ".env.example not found. Cannot create .env file."
        exit 1
    fi
fi

# Add swap if system has less than 2GB RAM and no swap
add_swap_if_needed() {
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    local swap_mem=$(free -m | awk 'NR==3{print $2}')
    
    if [ "$total_mem" -lt 2048 ] && [ "$swap_mem" -eq 0 ]; then
        log "Low memory detected (${total_mem}MB). Adding 2GB swap..."
        
        # Check if swap file already exists
        if [ ! -f /swapfile ]; then
            sudo fallocate -l 2G /swapfile
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
        fi
        
        # Enable swap
        sudo swapon /swapfile 2>/dev/null || true
        
        # Make permanent if not already in fstab
        if ! grep -q '/swapfile' /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        
        success "Swap added successfully"
        free -h
    fi
}

# Choose deployment mode based on available memory
choose_deployment_mode() {
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    
    echo ""
    log "Available deployment modes:"
    echo "  1) Essential only (ai-trading-bot + bluefin-service) - 512MB RAM"
    echo "  2) Core services (+ dashboard-backend) - 768MB RAM"
    echo "  3) Full services (+ dashboard-frontend + MCP) - 1GB+ RAM"
    echo ""
    
    if [ "$total_mem" -lt 1024 ]; then
        warning "Low memory detected (${total_mem}MB). Recommending Essential mode."
        echo "1" > /tmp/deployment_mode
    else
        log "System memory: ${total_mem}MB"
        read -p "Choose deployment mode (1-3): " -r deployment_mode
        echo "$deployment_mode" > /tmp/deployment_mode
    fi
}

# Start services based on deployment mode
start_services() {
    local mode=$(cat /tmp/deployment_mode 2>/dev/null || echo "2")
    
    case $mode in
        1)
            log "Starting Essential services..."
            docker compose up -d ai-trading-bot bluefin-service
            ;;
        2)
            log "Starting Core services..."
            docker compose up -d ai-trading-bot bluefin-service dashboard-backend
            ;;
        3)
            log "Starting Full services..."
            docker compose --profile full up -d
            ;;
        *)
            log "Starting Core services (default)..."
            docker compose up -d ai-trading-bot bluefin-service dashboard-backend
            ;;
    esac
}

# Main execution
main() {
    log "🚀 Starting AI Trading Bot deployment..."
    
    # Add swap if needed
    add_swap_if_needed
    
    # Choose deployment mode
    choose_deployment_mode
    
    # Use sequential build script if available
    if [ -f "scripts/sequential-docker-build.sh" ]; then
        log "Using sequential build process..."
        ./scripts/sequential-docker-build.sh --no-start
    else
        log "Building services..."
        # Build with limited parallelism
        docker compose build --parallel 1
    fi
    
    # Start services
    start_services
    
    # Show status
    echo ""
    success "Services started successfully!"
    
    log "Service status:"
    docker compose ps
    
    echo ""
    log "Useful commands:"
    echo "  View logs:     docker compose logs -f"
    echo "  Stop services: docker compose down"
    echo "  Restart:       docker compose restart"
    echo "  Update:        git pull && ./start-trading-bot.sh"
    
    # Show access URLs
    echo ""
    log "Access URLs:"
    if docker compose ps | grep -q dashboard-backend; then
        echo "  Dashboard API: http://$(hostname -I | awk '{print $1}'):8000"
    fi
    if docker compose ps | grep -q dashboard-frontend; then
        echo "  Web Dashboard: http://$(hostname -I | awk '{print $1}'):3000"
    fi
    
    # Show logs for main service
    echo ""
    log "Following AI Trading Bot logs (Ctrl+C to exit):"
    docker compose logs -f ai-trading-bot
}

# Handle interruption
trap 'log "Deployment interrupted"; exit 1' INT TERM

# Parse arguments
case "${1:-}" in
    --help|-h)
        echo "AI Trading Bot Startup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help"
        echo "  --essential    Start only essential services"
        echo "  --core         Start core services (default)"
        echo "  --full         Start all services"
        echo "  --rebuild      Rebuild all containers"
        echo ""
        exit 0
        ;;
    --essential)
        echo "1" > /tmp/deployment_mode
        ;;
    --core)
        echo "2" > /tmp/deployment_mode
        ;;
    --full)
        echo "3" > /tmp/deployment_mode
        ;;
    --rebuild)
        log "Rebuilding all containers..."
        docker compose down --volumes
        docker system prune -f
        ;;
esac

# Run main function
main