#!/bin/bash
# Enhanced Quick Launch Script for Bluefin DEX Trading Bot
# 
# OVERVIEW:
# This script provides enhanced service orchestration for the Bluefin DEX trading bot,
# supporting both Docker Compose and direct Python execution modes with proper service
# health checks, dependency management, and failure handling.
#
# FEATURES:
# - Automatic Docker detection and mode selection
# - Proper service startup order with health checks
# - Bluefin service dependency management
# - Optional MCP memory and dashboard services
# - Comprehensive error handling and recovery
# - Service monitoring and management commands
# - Graceful cleanup on exit
#
# MODES:
# 1. Docker Compose Mode (Recommended):
#    - Full service orchestration
#    - Proper dependency management
#    - Health checks and monitoring
#    - Service isolation and security
#
# 2. Direct Python Mode:
#    - Faster startup for development
#    - Bypasses Docker overhead
#    - Limited service integration
#
# USAGE EXAMPLES:
#   ./quick-launch-bluefin.sh                    # Auto-detect mode
#   ./quick-launch-bluefin.sh --docker           # Docker Compose mode
#   ./quick-launch-bluefin.sh --direct           # Direct Python mode
#   ./quick-launch-bluefin.sh --docker --memory --dashboard  # Full setup
#   ./quick-launch-bluefin.sh --status           # Check service status
#   ./quick-launch-bluefin.sh --logs             # View service logs
#   ./quick-launch-bluefin.sh --stop             # Stop all services
#
# REQUIREMENTS:
# - .env file with configured API keys
# - Docker and Docker Compose (for Docker mode)
# - Python 3.12+ and Poetry (for direct mode)
# - docker-compose.bluefin.yml file
#
# SERVICES MANAGED:
# - bluefin-service: Required Bluefin DEX service (always started first)
# - ai-trading-bot-bluefin: Main trading bot
# - mcp-memory: Optional memory service (--memory flag)
# - dashboard-backend/frontend: Optional dashboard (--dashboard flag)

set -e  # Exit on any error

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.bluefin.yml"
SERVICE_TIMEOUT=60
HEALTH_CHECK_INTERVAL=5
MAX_HEALTH_CHECKS=12

echo -e "${CYAN}🚀 QUICK LAUNCH: BLUEFIN DEX TRADING BOT${NC}"
echo "========================================"

# Function to print colored messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if Docker and Docker Compose are available
check_docker() {
    if ! command -v docker &> /dev/null; then
        return 1
    fi
    
    if ! docker info &> /dev/null 2>&1; then
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
        return 1
    fi
    
    return 0
}

# Function to wait for service health check
wait_for_service_health() {
    local service_name=$1
    local container_name=$2
    local health_url=$3
    local timeout=${4:-$SERVICE_TIMEOUT}
    
    print_info "Waiting for $service_name to become healthy..."
    
    local count=0
    local max_attempts=$((timeout / HEALTH_CHECK_INTERVAL))
    
    while [ $count -lt $max_attempts ]; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service_name" | grep -q "healthy"; then
            print_status "$service_name is healthy!"
            return 0
        elif docker-compose -f "$COMPOSE_FILE" ps "$service_name" | grep -q "unhealthy"; then
            print_error "$service_name health check failed!"
            docker-compose -f "$COMPOSE_FILE" logs --tail=20 "$service_name"
            return 1
        elif [ -n "$health_url" ] && command -v curl &> /dev/null; then
            if curl -sf "$health_url" > /dev/null 2>&1; then
                print_status "$service_name is responding to health check!"
                return 0
            fi
        fi
        
        echo -n "."
        sleep $HEALTH_CHECK_INTERVAL
        count=$((count + 1))
    done
    
    print_error "$service_name failed to become healthy within ${timeout}s"
    return 1
}

# Function to stop services gracefully
cleanup_services() {
    if [ "$DOCKER_MODE" = "true" ]; then
        print_info "Stopping Docker services..."
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    fi
}

# Trap to cleanup on exit
trap cleanup_services EXIT

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --docker, -d      Use Docker Compose (recommended)"
    echo "  --direct, -n      Use direct Python execution (faster startup)"
    echo "  --memory, -m      Enable MCP memory service (Docker mode only)"
    echo "  --dashboard, -b   Enable dashboard services (Docker mode only)"
    echo "  --help, -h        Show this help message"
    echo "  --status, -s      Check service status"
    echo "  --stop, -x        Stop running services"
    echo "  --logs, -l        Show service logs"
    echo ""
    echo "Examples:"
    echo "  $0 --docker         # Launch with Docker Compose"
    echo "  $0 --direct         # Launch with direct Python"
    echo "  $0 --docker --memory --dashboard  # Full Docker setup"
    echo "  $0 --status         # Check service status"
    echo "  $0 --stop           # Stop all services"
}

# Function to check service status
check_status() {
    if [ -f "$COMPOSE_FILE" ]; then
        echo -e "${CYAN}📊 Service Status:${NC}"
        docker-compose -f "$COMPOSE_FILE" ps
        echo ""
        echo -e "${CYAN}🔍 Health Status:${NC}"
        docker-compose -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    else
        print_error "Docker Compose file not found: $COMPOSE_FILE"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$COMPOSE_FILE" ]; then
        echo -e "${CYAN}📄 Service Logs:${NC}"
        docker-compose -f "$COMPOSE_FILE" logs --tail=50 -f
    else
        print_error "Docker Compose file not found: $COMPOSE_FILE"
    fi
}

# Function to stop services
stop_services() {
    if [ -f "$COMPOSE_FILE" ]; then
        print_info "Stopping all services..."
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans
        
        # Optionally clean up networks (only if they're empty)
        print_info "Cleaning up unused networks..."
        docker network rm bluefin-trading-network 2>/dev/null || true
        docker network rm trading-network 2>/dev/null || true
        
        print_status "All services stopped"
    else
        print_error "Docker Compose file not found: $COMPOSE_FILE"
    fi
}

# Parse command line arguments
DOCKER_MODE=""
ENABLE_MEMORY=false
ENABLE_DASHBOARD=false
SHOW_STATUS=false
STOP_SERVICES=false
SHOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker|-d)
            DOCKER_MODE="true"
            shift
            ;;
        --direct|-n)
            DOCKER_MODE="false"
            shift
            ;;
        --memory|-m)
            ENABLE_MEMORY=true
            shift
            ;;
        --dashboard|-b)
            ENABLE_DASHBOARD=true
            shift
            ;;
        --status|-s)
            SHOW_STATUS=true
            shift
            ;;
        --stop|-x)
            STOP_SERVICES=true
            shift
            ;;
        --logs|-l)
            SHOW_LOGS=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Handle status, stop, and logs commands
if [ "$SHOW_STATUS" = "true" ]; then
    check_status
    exit 0
fi

if [ "$STOP_SERVICES" = "true" ]; then
    stop_services
    exit 0
fi

if [ "$SHOW_LOGS" = "true" ]; then
    show_logs
    exit 0
fi

# Check if required files exist
if [[ ! -f ".env" ]]; then
    print_error ".env file not found"
    echo "   Please configure your API keys in .env:"
    echo "   - LLM__OPENAI_API_KEY=your_openai_api_key"
    echo "   - EXCHANGE__BLUEFIN_PRIVATE_KEY=your_sui_wallet_private_key"
    exit 1
fi

# Check if API keys are configured
if grep -q "PLEASE_REPLACE" .env; then
    print_error "API keys not configured in .env"
    echo "   Please update the following in .env:"
    echo "   - LLM__OPENAI_API_KEY=sk-your_actual_openai_key"
    echo "   - EXCHANGE__BLUEFIN_PRIVATE_KEY=0xYourActualSuiPrivateKey"
    echo ""
    echo "   Run: ./configure-api-keys.sh for help"
    exit 1
fi

print_status "Using existing .env configuration for Bluefin..."

# Create required directories
print_info "Creating required directories..."
mkdir -p logs/bluefin
mkdir -p data
mkdir -p data/mcp_memory

# Auto-detect mode if not specified
if [ -z "$DOCKER_MODE" ]; then
    if check_docker && [ -f "$COMPOSE_FILE" ]; then
        print_info "Docker detected - using Docker Compose mode (use --direct for Python mode)"
        DOCKER_MODE="true"
    else
        print_info "Docker not available or compose file missing - using direct Python mode"
        DOCKER_MODE="false"
    fi
fi

print_status "Environment configuration ready"

if [ "$DOCKER_MODE" = "true" ]; then
    # Docker Compose Mode
    echo ""
    echo -e "${CYAN}🐳 DOCKER COMPOSE MODE${NC}"
    echo "======================="
    
    # Check Docker availability
    if ! check_docker; then
        print_error "Docker or Docker Compose not available"
        echo "   Please install Docker and Docker Compose or use --direct mode"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        print_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    print_status "Docker and Docker Compose are available"
    
    # Build compose command with profiles
    compose_profiles=""
    if [ "$ENABLE_MEMORY" = "true" ]; then
        compose_profiles="$compose_profiles --profile memory"
    fi
    if [ "$ENABLE_DASHBOARD" = "true" ]; then
        compose_profiles="$compose_profiles --profile dashboard"
    fi
    
    # Stop any existing services
    print_info "Stopping any existing services..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    
    # Create required networks if they don't exist
    print_info "Ensuring Docker networks exist..."
    docker network create bluefin-trading-network 2>/dev/null || true
    docker network create trading-network 2>/dev/null || true
    
    # Build services
    print_info "Building required services..."
    docker-compose -f "$COMPOSE_FILE" build bluefin-service
    
    # Start bluefin-service first
    print_info "Starting Bluefin service..."
    docker-compose -f "$COMPOSE_FILE" up -d bluefin-service
    
    # Wait for bluefin-service to be healthy
    if ! wait_for_service_health "bluefin-service" "bluefin-service" "http://localhost:8081/health"; then
        print_error "Bluefin service failed to start properly"
        docker-compose -f "$COMPOSE_FILE" logs bluefin-service
        exit 1
    fi
    
    # Start additional services if requested
    if [ "$ENABLE_MEMORY" = "true" ]; then
        print_info "Starting MCP memory service..."
        docker-compose -f "$COMPOSE_FILE" up -d mcp-memory
        if ! wait_for_service_health "mcp-memory" "mcp-memory-server" "http://localhost:8765/health"; then
            print_warning "MCP memory service failed to start - continuing without it"
        fi
    fi
    
    # Build and start the trading bot
    print_info "Building and starting trading bot..."
    docker-compose -f "$COMPOSE_FILE" build ai-trading-bot-bluefin
    docker-compose -f "$COMPOSE_FILE" up -d ai-trading-bot-bluefin
    
    # Start dashboard services if requested
    if [ "$ENABLE_DASHBOARD" = "true" ]; then
        print_info "Starting dashboard services..."
        docker-compose -f "$COMPOSE_FILE" $compose_profiles up -d dashboard-backend dashboard-frontend
    fi
    
    # Wait for trading bot to be ready
    print_info "Waiting for trading bot to initialize..."
    sleep 15
    
    # Check final status
    echo ""
    echo -e "${CYAN}📊 Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo -e "${GREEN}✅ BLUEFIN TRADING BOT LAUNCHED SUCCESSFULLY${NC}"
    echo "============================================="
    echo -e "${BLUE}🎯 Configuration:${NC}"
    echo "   • Exchange: Bluefin DEX (Sui Network)"
    echo "   • Symbol: SUI-PERP"
    echo "   • Mode: Paper trading (safe)"
    echo "   • Data: Real market data only"
    echo "   • Deployment: Docker Compose"
    echo ""
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "   View logs:    docker-compose -f $COMPOSE_FILE logs -f ai-trading-bot-bluefin"
    echo "   Stop bot:     docker-compose -f $COMPOSE_FILE down"
    echo "   Check status: $0 --status"
    echo "   Show logs:    $0 --logs"
    echo ""
    if [ "$ENABLE_DASHBOARD" = "true" ]; then
        echo -e "${BLUE}📊 Dashboard:${NC}"
        echo "   Frontend: http://localhost:3000"
        echo "   Backend:  http://localhost:8000"
        echo ""
    fi
    echo -e "${BLUE}🔍 Service URLs:${NC}"
    echo "   Bluefin Service: http://localhost:8081/health"
    if [ "$ENABLE_MEMORY" = "true" ]; then
        echo "   MCP Memory:      http://localhost:8765/health"
    fi
    
    # Show initial logs
    echo ""
    echo -e "${CYAN}📄 Initial Trading Bot Logs:${NC}"
    echo "============================"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20 ai-trading-bot-bluefin
    
else
    # Direct Python Mode
    echo ""
    echo -e "${CYAN}🐍 DIRECT PYTHON MODE${NC}"
    echo "===================="
    
    print_warning "Running in direct mode - Bluefin service may not be available"
    print_info "For full functionality, consider using Docker mode: $0 --docker"
    
    echo ""
    echo -e "${BLUE}🎯 Configuration:${NC}"
    echo "   • Exchange: Bluefin DEX (Sui Network)"
    echo "   • Symbol: SUI-PERP"
    echo "   • Mode: Paper trading (safe)"
    echo "   • Data: Real market data only"
    echo "   • Deployment: Direct Python execution"
    echo ""
    
    # Check if Poetry is available
    if command -v poetry &> /dev/null; then
        print_info "Using Poetry to run the bot..."
        poetry run python -m bot.main live --symbol SUI-PERP --interval 1m --force
    else
        print_info "Using Python directly to run the bot..."
        python -m bot.main live --symbol SUI-PERP --interval 1m --force
    fi
fi