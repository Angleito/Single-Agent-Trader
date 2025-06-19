#!/bin/bash
# AI Trading Bot Docker Startup Script
# Handles proper network creation and service orchestration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
    esac
}

# Print header
echo "🚀 AI Trading Bot Startup"
echo "========================="

# Configuration
EXCHANGE_TYPE=${EXCHANGE_TYPE:-coinbase}
COMPOSE_FILE=""

case $EXCHANGE_TYPE in
    "coinbase")
        COMPOSE_FILE="docker-compose.yml"
        print_status "INFO" "Starting Coinbase configuration"
        ;;
    "bluefin")
        COMPOSE_FILE="docker-compose.bluefin.yml"
        print_status "INFO" "Starting Bluefin configuration"
        ;;
    *)
        print_status "ERROR" "Invalid exchange type: $EXCHANGE_TYPE. Use 'coinbase' or 'bluefin'"
        exit 1
        ;;
esac

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_status "ERROR" "Docker is not running. Please start Docker and try again."
    exit 1
fi
print_status "SUCCESS" "Docker is running"

# Create trading-network if it doesn't exist
if ! docker network ls | grep -q "trading-network"; then
    print_status "INFO" "Creating trading-network..."
    docker network create \
        --driver bridge \
        --subnet 172.21.0.0/16 \
        --gateway 172.21.0.1 \
        --opt com.docker.network.bridge.name=tr0 \
        --opt com.docker.network.bridge.enable_icc=true \
        --opt com.docker.network.bridge.enable_ip_masquerade=true \
        trading-network
    print_status "SUCCESS" "trading-network created"
else
    print_status "SUCCESS" "trading-network already exists"
fi

# Stop any existing services
print_status "INFO" "Stopping any existing services..."
docker-compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true
if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
    docker-compose -f docker-compose.bluefin.yml down --remove-orphans 2>/dev/null || true
fi

# Build and start services
print_status "INFO" "Building and starting services with $COMPOSE_FILE..."

if [ "$EXCHANGE_TYPE" = "bluefin" ]; then
    # For Bluefin, we need to ensure the main network exists and start bluefin services
    print_status "INFO" "Starting Bluefin services..."
    docker-compose -f docker-compose.bluefin.yml up -d --build
else
    # For Coinbase, use the main compose file
    print_status "INFO" "Starting Coinbase services..."
    docker-compose -f docker-compose.yml up -d --build
fi

# Wait for services to be healthy
print_status "INFO" "Waiting for services to be healthy..."
sleep 10

# Run network validation
print_status "INFO" "Running network validation..."
if [ -x "./scripts/validate-docker-network.sh" ]; then
    ./scripts/validate-docker-network.sh
else
    print_status "WARNING" "Network validation script not found or not executable"
fi

# Show running services
echo ""
print_status "SUCCESS" "Services started successfully!"
echo ""
echo "📊 Running Services:"
docker ps --filter "network=trading-network" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "📋 Useful Commands:"
echo "=================="
echo "• View logs: docker-compose -f $COMPOSE_FILE logs -f [service-name]"
echo "• Stop services: docker-compose -f $COMPOSE_FILE down"
echo "• Restart service: docker-compose -f $COMPOSE_FILE restart [service-name]"
echo "• Execute command: docker exec -it [container-name] [command]"
echo "• Network info: docker network inspect trading-network"

echo ""
print_status "SUCCESS" "Startup complete! 🎉"