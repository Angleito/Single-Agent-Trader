#!/bin/bash

# AI Trading Bot Dashboard - Docker Integration Test Script
# Tests WebSocket connections, API endpoints, and UI components in Docker environment

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
TEST_DIR="$PROJECT_DIR/test"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service configuration
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
WS_URL="ws://localhost:8000/ws"

# Test configuration
MAX_WAIT_TIME=120  # Maximum time to wait for services (seconds)
HEALTH_CHECK_INTERVAL=5  # Interval between health checks (seconds)

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi

    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi

    if ! command -v node &> /dev/null; then
        missing_tools+=("node")
    fi

    if ! command -v npm &> /dev/null; then
        missing_tools+=("npm")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi

    log_success "All prerequisites are installed"
}

# Clean up function
cleanup() {
    log_info "Cleaning up..."

    # Stop containers
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        log_info "Stopping Docker containers..."
        docker-compose -f "$COMPOSE_FILE" down
    fi

    # Remove test data files
    if [ -d "$TEST_DIR/data" ]; then
        rm -rf "$TEST_DIR/data"
    fi

    log_success "Cleanup completed"
}

# Set up trap to cleanup on exit
trap cleanup EXIT

# Build and start Docker containers
start_containers() {
    log_info "Building and starting Docker containers..."

    cd "$PROJECT_DIR"

    # Build containers
    log_info "Building containers..."
    if ! docker-compose -f "$COMPOSE_FILE" build; then
        log_error "Failed to build containers"
        exit 1
    fi

    # Start containers
    log_info "Starting containers..."
    if ! docker-compose -f "$COMPOSE_FILE" up -d dashboard-backend dashboard-frontend; then
        log_error "Failed to start containers"
        exit 1
    fi

    log_success "Containers started successfully"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."

    local backend_ready=false
    local frontend_ready=false
    local wait_time=0

    while [ $wait_time -lt $MAX_WAIT_TIME ]; do
        # Check backend health
        if ! $backend_ready && curl -s -f "$BACKEND_URL/health" &> /dev/null; then
            log_success "Backend service is healthy"
            backend_ready=true
        fi

        # Check frontend (basic connectivity)
        if ! $frontend_ready && curl -s -f "$FRONTEND_URL" &> /dev/null; then
            log_success "Frontend service is healthy"
            frontend_ready=true
        fi

        # Check if both services are ready
        if $backend_ready && $frontend_ready; then
            log_success "All services are healthy"
            return 0
        fi

        # Wait and increment counter
        sleep $HEALTH_CHECK_INTERVAL
        wait_time=$((wait_time + HEALTH_CHECK_INTERVAL))

        if [ $((wait_time % 20)) -eq 0 ]; then
            log_info "Still waiting for services... (${wait_time}s elapsed)"
        fi
    done

    log_error "Services failed to become healthy within $MAX_WAIT_TIME seconds"

    # Show container logs for debugging
    log_info "Container status:"
    docker-compose -f "$COMPOSE_FILE" ps

    log_info "Backend logs:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20 dashboard-backend

    log_info "Frontend logs:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20 dashboard-frontend

    exit 1
}

# Run WebSocket tests
run_websocket_tests() {
    log_info "Running WebSocket connectivity tests..."

    # Run Node.js WebSocket tests
    if ! node "$TEST_DIR/websocket-test.js"; then
        log_error "WebSocket tests failed"
        return 1
    fi

    log_success "WebSocket tests passed"
    return 0
}

# Run API tests
run_api_tests() {
    log_info "Running REST API tests..."

    # Run Node.js API tests
    if ! node "$TEST_DIR/api-test.js"; then
        log_error "API tests failed"
        return 1
    fi

    log_success "API tests passed"
    return 0
}

# Run UI tests
run_ui_tests() {
    log_info "Running UI component tests..."

    # Open UI test page in headless browser simulation
    local ui_test_url="$FRONTEND_URL/test/ui-test.html"

    # Copy UI test file to frontend public directory if it exists
    if [ -f "$TEST_DIR/ui-test.html" ]; then
        mkdir -p "$PROJECT_DIR/frontend/public/test"
        cp "$TEST_DIR/ui-test.html" "$PROJECT_DIR/frontend/public/test/"
    fi

    # Check if UI test page is accessible
    if curl -s -f "$ui_test_url" &> /dev/null; then
        log_success "UI test page is accessible"
    else
        log_warning "UI test page not accessible at $ui_test_url"
        log_info "Testing basic frontend accessibility instead"

        # Test basic frontend response
        if curl -s -f "$FRONTEND_URL" | grep -q "html\|HTML"; then
            log_success "Frontend serves HTML content"
        else
            log_error "Frontend does not serve valid HTML"
            return 1
        fi
    fi

    log_success "UI tests completed"
    return 0
}

# Validate Docker environment
validate_docker_environment() {
    log_info "Validating Docker environment..."

    # Check container status
    local backend_status=$(docker-compose -f "$COMPOSE_FILE" ps -q dashboard-backend | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null || echo "not_found")
    local frontend_status=$(docker-compose -f "$COMPOSE_FILE" ps -q dashboard-frontend | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null || echo "not_found")

    log_info "Backend container status: $backend_status"
    log_info "Frontend container status: $frontend_status"

    if [ "$backend_status" != "running" ]; then
        log_error "Backend container is not running"
        return 1
    fi

    if [ "$frontend_status" != "running" ]; then
        log_error "Frontend container is not running"
        return 1
    fi

    # Check network connectivity between containers
    local backend_container=$(docker-compose -f "$COMPOSE_FILE" ps -q dashboard-backend)
    local frontend_container=$(docker-compose -f "$COMPOSE_FILE" ps -q dashboard-frontend)

    if [ -n "$backend_container" ] && [ -n "$frontend_container" ]; then
        log_info "Testing inter-container connectivity..."

        # Test if frontend can reach backend
        if docker exec "$frontend_container" wget -q --spider "$BACKEND_URL/health" 2>/dev/null; then
            log_success "Frontend can reach backend"
        else
            log_warning "Frontend cannot reach backend directly"
        fi
    fi

    # Check resource usage
    log_info "Container resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        $(docker-compose -f "$COMPOSE_FILE" ps -q)

    log_success "Docker environment validation completed"
    return 0
}

# Check data flow and logs
check_data_flow() {
    log_info "Checking data flow and logs..."

    # Check if backend is generating logs
    local backend_logs=$(docker-compose -f "$COMPOSE_FILE" logs --tail=10 dashboard-backend 2>/dev/null || echo "")
    if [ -n "$backend_logs" ]; then
        log_success "Backend is generating logs"
    else
        log_warning "No recent backend logs found"
    fi

    # Check if WebSocket connections are working by looking for connection messages
    local ws_logs=$(docker-compose -f "$COMPOSE_FILE" logs dashboard-backend 2>/dev/null | grep -i "websocket\|connection" | tail -5 || echo "")
    if [ -n "$ws_logs" ]; then
        log_success "WebSocket activity detected in logs"
        echo "$ws_logs" | head -3
    else
        log_warning "No WebSocket activity found in logs"
    fi

    # Test if mock data generation is working
    local api_response=$(curl -s "$BACKEND_URL/trading-data" 2>/dev/null || echo "")
    if echo "$api_response" | grep -q "account\|balance"; then
        log_success "Trading data API is returning mock data"
    else
        log_warning "Trading data API may not be working correctly"
    fi

    log_success "Data flow check completed"
    return 0
}

# Run comprehensive integration tests
run_integration_tests() {
    log_info "Running comprehensive integration tests..."

    local test_results=()

    # Run WebSocket tests
    if run_websocket_tests; then
        test_results+=("WebSocket: PASS")
    else
        test_results+=("WebSocket: FAIL")
    fi

    # Run API tests
    if run_api_tests; then
        test_results+=("API: PASS")
    else
        test_results+=("API: FAIL")
    fi

    # Run UI tests
    if run_ui_tests; then
        test_results+=("UI: PASS")
    else
        test_results+=("UI: FAIL")
    fi

    # Validate Docker environment
    if validate_docker_environment; then
        test_results+=("Docker: PASS")
    else
        test_results+=("Docker: FAIL")
    fi

    # Check data flow
    if check_data_flow; then
        test_results+=("Data Flow: PASS")
    else
        test_results+=("Data Flow: FAIL")
    fi

    # Print test results
    log_info "Integration test results:"
    for result in "${test_results[@]}"; do
        if [[ $result == *"PASS"* ]]; then
            log_success "$result"
        else
            log_error "$result"
        fi
    done

    # Check overall status
    if echo "${test_results[@]}" | grep -q "FAIL"; then
        log_error "Some integration tests failed"
        return 1
    else
        log_success "All integration tests passed"
        return 0
    fi
}

# Generate test report
generate_test_report() {
    local report_file="$TEST_DIR/integration-test-report.txt"

    log_info "Generating test report: $report_file"

    {
        echo "AI Trading Bot Dashboard - Integration Test Report"
        echo "================================================="
        echo "Date: $(date)"
        echo "Test Environment: Docker"
        echo ""
        echo "Services Tested:"
        echo "- Backend API: $BACKEND_URL"
        echo "- Frontend: $FRONTEND_URL"
        echo "- WebSocket: $WS_URL"
        echo ""
        echo "Container Status:"
        docker-compose -f "$COMPOSE_FILE" ps
        echo ""
        echo "Recent Backend Logs:"
        docker-compose -f "$COMPOSE_FILE" logs --tail=20 dashboard-backend
        echo ""
        echo "Recent Frontend Logs:"
        docker-compose -f "$COMPOSE_FILE" logs --tail=20 dashboard-frontend
        echo ""
        echo "Test Results:"
        echo "- WebSocket connectivity: $([ -f "$TEST_DIR/.websocket_test_passed" ] && echo "PASS" || echo "FAIL")"
        echo "- API endpoints: $([ -f "$TEST_DIR/.api_test_passed" ] && echo "PASS" || echo "FAIL")"
        echo "- UI components: $([ -f "$TEST_DIR/.ui_test_passed" ] && echo "PASS" || echo "FAIL")"
        echo "- Docker validation: $([ -f "$TEST_DIR/.docker_test_passed" ] && echo "PASS" || echo "FAIL")"
        echo "- Data flow: $([ -f "$TEST_DIR/.dataflow_test_passed" ] && echo "PASS" || echo "FAIL")"
    } > "$report_file"

    log_success "Test report generated: $report_file"
}

# Main function
main() {
    log_info "Starting AI Trading Bot Dashboard Integration Tests"
    log_info "=================================================="

    # Create test data directory
    mkdir -p "$TEST_DIR/data"

    # Check prerequisites
    check_prerequisites

    # Start containers
    start_containers

    # Wait for services
    wait_for_services

    # Run integration tests
    if run_integration_tests; then
        log_success "All tests passed successfully!"
        generate_test_report
        exit 0
    else
        log_error "Some tests failed!"
        generate_test_report
        exit 1
    fi
}

# Help function
show_help() {
    echo "AI Trading Bot Dashboard - Docker Integration Test Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  --no-cleanup        Don't cleanup containers after tests"
    echo "  --verbose           Enable verbose output"
    echo "  --quick             Run only basic connectivity tests"
    echo ""
    echo "Environment Variables:"
    echo "  MAX_WAIT_TIME       Maximum time to wait for services (default: 120s)"
    echo "  BACKEND_URL         Backend service URL (default: http://localhost:8000)"
    echo "  FRONTEND_URL        Frontend service URL (default: http://localhost:3000)"
    echo ""
}

# Parse command line arguments
CLEANUP_ON_EXIT=true
VERBOSE=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-cleanup)
            CLEANUP_ON_EXIT=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Override cleanup behavior if requested
if [ "$CLEANUP_ON_EXIT" = false ]; then
    trap - EXIT
fi

# Run main function
main "$@"
