#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
STRESS_TEST_ENV="${STRESS_TEST_ENV:-low_resource}"
TEST_DURATION="${TEST_DURATION:-300}"
CONCURRENT_USERS="${CONCURRENT_USERS:-3}"
OPERATIONS_PER_SECOND="${OPERATIONS_PER_SECOND:-5}"
COMPOSE_FILE="docker-compose.stress-test.yml"
RESULTS_DIR="stress_test_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

print_header() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_status() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_info() {
    echo -e "${CYAN}[INFO] $1${NC}"
}

# ASCII Art Banner
echo -e "${GREEN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║          AI Trading Bot - Low-Resource Stress Test           ║
║                                                               ║
║  🚀 Comprehensive load testing in constrained environment    ║
║  💾 Memory pressure and stability verification               ║
║  📊 Performance degradation detection                        ║
║  🔄 Recovery testing and monitoring                          ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

print_header "Pre-flight Checks"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running!"
    exit 1
fi
print_success "Docker is running"

# Check Docker Compose
if ! command -v docker-compose > /dev/null 2>&1; then
    print_error "Docker Compose is not installed!"
    exit 1
fi
print_success "Docker Compose is available"

# Check required files
REQUIRED_FILES=(
    "$COMPOSE_FILE"
    "stress_test_low_resource.py"
    "Dockerfile.stress-test"
    ".env"
)

for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        print_error "Required file missing: $FILE"
        exit 1
    fi
done
print_success "All required files present"

print_header "Environment Setup"

# Set environment variables
export STRESS_TEST_ENV="$STRESS_TEST_ENV"
export TEST_DURATION_SECONDS="$TEST_DURATION"
export CONCURRENT_USERS="$CONCURRENT_USERS"
export OPERATIONS_PER_SECOND="$OPERATIONS_PER_SECOND"

print_info "Configuration:"
echo "  - Environment: $STRESS_TEST_ENV"
echo "  - Test Duration: ${TEST_DURATION}s"
echo "  - Concurrent Users: $CONCURRENT_USERS"
echo "  - Operations/Second: $OPERATIONS_PER_SECOND"
echo "  - Results Directory: $RESULTS_DIR"

print_header "Building Images"

print_status "Building stress test images..."
docker-compose -f "$COMPOSE_FILE" build --no-cache

if [ $? -eq 0 ]; then
    print_success "Images built successfully"
else
    print_error "Failed to build images"
    exit 1
fi

print_header "Starting Services"

# Clean up any existing containers
print_status "Cleaning up existing containers..."
docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true

# Start core services (without stress test runner)
print_status "Starting core services..."
docker-compose -f "$COMPOSE_FILE" up -d ai-trading-bot bluefin-service dashboard-backend

# Wait for services to be ready
print_status "Waiting for services to be healthy..."
sleep 30

# Check service health
SERVICES=("ai-trading-bot-stress" "bluefin-service-stress" "dashboard-backend-stress")
ALL_HEALTHY=true

for SERVICE in "${SERVICES[@]}"; do
    if docker inspect "$SERVICE" | grep '"Health"' | grep -q '"healthy"'; then
        print_success "$SERVICE is healthy"
    elif docker ps | grep -q "$SERVICE.*Up"; then
        print_info "$SERVICE is running (health check may not be configured)"
    else
        print_error "$SERVICE is not running properly"
        docker logs --tail=20 "$SERVICE"
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = false ]; then
    print_error "Some services failed to start properly"
    print_info "Showing container status:"
    docker ps -a --filter "name=stress"
    exit 1
fi

print_header "Running Stress Tests"

print_status "Starting comprehensive stress test suite..."

# Option 1: Run stress test directly on host (recommended)
if command -v python3 > /dev/null 2>&1; then
    print_info "Running stress tests on host system..."

    # Install dependencies if needed
    if ! python3 -c "import aiohttp, websockets, docker, psutil" 2>/dev/null; then
        print_status "Installing required Python packages..."
        pip3 install aiohttp websockets docker psutil requests
    fi

    # Set environment variables for the test
    export BLUEFIN_ENDPOINT="http://localhost:8082"
    export DASHBOARD_ENDPOINT="http://localhost:8000"
    export WEBSOCKET_ENDPOINT="ws://localhost:8000/ws"

    # Run the stress test
    python3 stress_test_low_resource.py
    STRESS_TEST_EXIT_CODE=$?

else
    # Option 2: Run stress test in container
    print_info "Running stress tests in container..."

    # Start monitoring if requested
    if [ "${ENABLE_MONITORING:-false}" = "true" ]; then
        print_status "Starting resource monitoring..."
        docker-compose -f "$COMPOSE_FILE" --profile monitoring up -d resource-monitor
    fi

    # Run the stress test container
    docker-compose -f "$COMPOSE_FILE" --profile stress-test run --rm stress-test-runner
    STRESS_TEST_EXIT_CODE=$?
fi

print_header "Test Results"

# Copy results if they exist
if [ -d "$(pwd)" ]; then
    find . -name "stress_test_report_*.json" -newer "$COMPOSE_FILE" -exec cp {} "$RESULTS_DIR/" \; 2>/dev/null || true
fi

# Show latest results
LATEST_REPORT=$(ls -t "$RESULTS_DIR"/stress_test_report_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    print_success "Test report available: $LATEST_REPORT"

    # Extract key metrics
    if command -v jq > /dev/null 2>&1; then
        print_info "Test Summary:"
        echo "  Overall Success: $(jq -r '.summary.overall_success' "$LATEST_REPORT")"
        echo "  Success Rate: $(jq -r '.summary.success_rate_percent' "$LATEST_REPORT")%"
        echo "  Total Operations: $(jq -r '.summary.total_operations' "$LATEST_REPORT")"
        echo "  Total Failures: $(jq -r '.summary.total_failures' "$LATEST_REPORT")"
        echo "  Max Memory: $(jq -r '.summary.max_memory_mb' "$LATEST_REPORT")MB"
        echo "  Avg CPU: $(jq -r '.summary.avg_cpu_percent' "$LATEST_REPORT")%"

        # Show recommendations
        print_info "Recommendations:"
        jq -r '.summary.recommendations[]' "$LATEST_REPORT" | while read -r rec; do
            echo "  $rec"
        done
    fi
else
    print_error "No test report found"
fi

print_header "Cleanup"

print_status "Collecting container logs..."
for SERVICE in "${SERVICES[@]}"; do
    if docker ps -a | grep -q "$SERVICE"; then
        docker logs "$SERVICE" > "$RESULTS_DIR/${SERVICE}.log" 2>&1 || true
    fi
done

print_status "Stopping services..."
docker-compose -f "$COMPOSE_FILE" down --remove-orphans

# Clean up any dangling images from the build
print_status "Cleaning up build artifacts..."
docker image prune -f --filter "label=stage=build" > /dev/null 2>&1 || true

print_header "Test Complete"

if [ $STRESS_TEST_EXIT_CODE -eq 0 ]; then
    print_success "✅ Stress test completed successfully!"
    print_info "All systems demonstrated stability under low-resource constraints"
else
    print_error "❌ Stress test failed or detected issues"
    print_info "Review the test report and logs for details"
fi

print_info "Results and logs saved to: $RESULTS_DIR"
print_info "Container logs: $RESULTS_DIR/*.log"
if [ -n "$LATEST_REPORT" ]; then
    print_info "Detailed report: $LATEST_REPORT"
fi

# Final system status
print_info "Final system status:"
echo "  Docker containers: $(docker ps -q | wc -l) running"
echo "  Docker images: $(docker images -q | wc -l) total"
echo "  Disk usage: $(df -h . | awk 'NR==2 {print $5}')"

exit $STRESS_TEST_EXIT_CODE
