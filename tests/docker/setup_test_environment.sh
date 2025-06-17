#!/bin/bash
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up WebSocket Integration Test Environment${NC}"

# Function to print status
print_status() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Phase 1: Create test environment file
print_status "Creating test environment configuration..."

if [ -f ".env.test" ]; then
    cp .env.test .env.test.backup
    print_status "Backed up existing .env.test to .env.test.backup"
fi

cp example.env .env.test

# Add WebSocket testing configuration
cat >> .env.test << EOF

# WebSocket Testing Configuration
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
SYSTEM__DRY_RUN=true
SYSTEM__WEBSOCKET_DASHBOARD_URL=ws://dashboard-backend:8000/ws
SYSTEM__WEBSOCKET_PUBLISH_INTERVAL=1.0
SYSTEM__WEBSOCKET_MAX_RETRIES=5
SYSTEM__WEBSOCKET_RETRY_DELAY=5
SYSTEM__WEBSOCKET_TIMEOUT=10
SYSTEM__WEBSOCKET_QUEUE_SIZE=100

# Test-specific settings
LOG_LEVEL=DEBUG
TEST_MODE=true
EOF

print_success "Test environment file created"

# Phase 2: Clean up existing containers and volumes
print_status "Cleaning up existing Docker resources..."

docker-compose down -v 2>/dev/null || true
docker system prune -f

print_success "Docker cleanup complete"

# Phase 3: Build fresh images
print_status "Building fresh Docker images..."

# Build with test configuration
docker-compose build --no-cache --build-arg BUILDKIT_PROGRESS=plain

if [ $? -eq 0 ]; then
    print_success "Docker images built successfully"
else
    print_error "Failed to build Docker images"
    exit 1
fi

# Phase 4: Verify network configuration
print_status "Verifying Docker network configuration..."

# Create network if it doesn't exist
docker network create trading-network 2>/dev/null || true

# Inspect network
NETWORK_INFO=$(docker network inspect trading-network 2>/dev/null)

if [ -n "$NETWORK_INFO" ]; then
    print_success "Network 'trading-network' is configured"
    echo -e "Network subnet: $(echo $NETWORK_INFO | grep -o '"Subnet": "[^"]*"' | cut -d'"' -f4)"
else
    print_error "Failed to create or inspect network"
    exit 1
fi

# Phase 5: Create test data directories
print_status "Creating test data directories..."

mkdir -p logs/test
mkdir -p data/test
mkdir -p tests/docker/results

print_success "Test directories created"

# Phase 6: Validate Docker Compose configuration
print_status "Validating Docker Compose configuration..."

docker-compose config --quiet

if [ $? -eq 0 ]; then
    print_success "Docker Compose configuration is valid"
else
    print_error "Docker Compose configuration is invalid"
    exit 1
fi

# Phase 7: Check required ports are available
print_status "Checking port availability..."

PORTS=(3000 8000 8765 8767 8081)
PORTS_AVAILABLE=true

for PORT in "${PORTS[@]}"; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_error "Port $PORT is already in use"
        PORTS_AVAILABLE=false
    else
        echo "  ✓ Port $PORT is available"
    fi
done

if [ "$PORTS_AVAILABLE" = false ]; then
    print_error "Some required ports are in use. Please free them before continuing."
    exit 1
fi

print_success "All required ports are available"

# Phase 8: Create test helper scripts
print_status "Creating test helper scripts..."

# Create container health check script
cat > tests/docker/check_health.sh << 'EOF'
#!/bin/bash

check_service_health() {
    local SERVICE=$1
    local MAX_ATTEMPTS=30
    local ATTEMPT=0

    echo -n "Checking $SERVICE health"

    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if docker-compose ps $SERVICE | grep -q "healthy"; then
            echo " ✓"
            return 0
        fi
        echo -n "."
        sleep 2
        ((ATTEMPT++))
    done

    echo " ✗"
    return 1
}

# Check all services
SERVICES=("dashboard-backend" "dashboard-frontend" "ai-trading-bot")
ALL_HEALTHY=true

for SERVICE in "${SERVICES[@]}"; do
    if ! check_service_health $SERVICE; then
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = true ]; then
    echo "All services are healthy!"
    exit 0
else
    echo "Some services failed health checks"
    exit 1
fi
EOF

chmod +x tests/docker/check_health.sh

# Create WebSocket connection test helper
cat > tests/docker/test_ws_connection.sh << 'EOF'
#!/bin/bash

# Test WebSocket connection from bot to dashboard
echo "Testing WebSocket connection..."

# Check if dashboard backend is accepting connections
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"
if [ $? -eq 0 ]; then
    echo "✓ Dashboard backend is responding"
else
    echo "✗ Dashboard backend is not responding"
    exit 1
fi

# Test WebSocket endpoint
curl -s -o /dev/null -w "%{http_code}" \
    -H "Upgrade: websocket" \
    -H "Connection: Upgrade" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
    -H "Sec-WebSocket-Version: 13" \
    http://localhost:8000/ws | grep -q "101"

if [ $? -eq 0 ]; then
    echo "✓ WebSocket endpoint is accepting connections"
else
    echo "✗ WebSocket endpoint is not responding correctly"
    exit 1
fi

echo "WebSocket connection test passed!"
EOF

chmod +x tests/docker/test_ws_connection.sh

print_success "Test helper scripts created"

# Phase 9: Summary
echo -e "\n${GREEN}✅ Test Environment Setup Complete!${NC}"
echo -e "\nNext steps:"
echo -e "1. Start services: ${YELLOW}docker-compose --env-file .env.test up -d${NC}"
echo -e "2. Check health: ${YELLOW}./tests/docker/check_health.sh${NC}"
echo -e "3. Test WebSocket: ${YELLOW}./tests/docker/test_ws_connection.sh${NC}"
echo -e "4. Run full test suite: ${YELLOW}./tests/docker/run_all_tests.sh${NC}"
echo -e "\nEnvironment details:"
echo -e "- Config file: ${YELLOW}.env.test${NC}"
echo -e "- Network: ${YELLOW}trading-network${NC}"
echo -e "- Ports: ${YELLOW}3000 (frontend), 8000 (backend), 8765 (MCP), 8767 (OmniSearch), 8081 (Bluefin)${NC}"
