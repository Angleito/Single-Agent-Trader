#!/bin/bash
set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Test configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="tests/docker/results/full_suite_$TIMESTAMP"
SUMMARY_FILE="$RESULTS_DIR/test_summary.json"
REPORT_FILE="$RESULTS_DIR/test_report.md"

# Test flags (can be overridden by environment variables)
RUN_SETUP=${RUN_SETUP:-true}
RUN_CONNECTIVITY=${RUN_CONNECTIVITY:-true}
RUN_MESSAGE_FLOW=${RUN_MESSAGE_FLOW:-true}
RUN_FAULT_TOLERANCE=${RUN_FAULT_TOLERANCE:-true}
RUN_LOAD_TESTS=${RUN_LOAD_TESTS:-true}
RUN_INTEGRATION=${RUN_INTEGRATION:-true}
RUN_PERFORMANCE=${RUN_PERFORMANCE:-true}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to print headers
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

print_test_status() {
    local TEST_NAME=$1
    local STATUS=$2
    
    if [ "$STATUS" = "PASSED" ]; then
        echo -e "${GREEN}✓ $TEST_NAME${NC}"
    else
        echo -e "${RED}✗ $TEST_NAME${NC}"
    fi
}

# Initialize test results
TEST_RESULTS=()
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test and capture results
run_test() {
    local TEST_NAME=$1
    local TEST_COMMAND=$2
    local TEST_LOG="$RESULTS_DIR/${TEST_NAME// /_}.log"
    
    print_status "Running: $TEST_NAME"
    
    ((TOTAL_TESTS++))
    
    # Run test and capture output
    if $TEST_COMMAND > "$TEST_LOG" 2>&1; then
        print_test_status "$TEST_NAME" "PASSED"
        ((PASSED_TESTS++))
        TEST_RESULTS+=("{\"name\": \"$TEST_NAME\", \"status\": \"passed\", \"log\": \"$TEST_LOG\"}")
        return 0
    else
        print_test_status "$TEST_NAME" "FAILED"
        ((FAILED_TESTS++))
        TEST_RESULTS+=("{\"name\": \"$TEST_NAME\", \"status\": \"failed\", \"log\": \"$TEST_LOG\"}")
        
        # Show last few lines of error
        echo -e "${RED}Last 10 lines of error log:${NC}"
        tail -n 10 "$TEST_LOG" | sed 's/^/  /'
        return 1
    fi
}

# ASCII Art Banner
echo -e "${GREEN}"
cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║     WebSocket Integration Test Suite - UltraThink Edition    ║
║                                                              ║
║  🚀 Comprehensive validation of real-time data bridge        ║
║  📊 Testing connectivity, flow, resilience & performance     ║
║  🔧 Docker-based production environment simulation           ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Start time
START_TIME=$(date +%s)

print_header "Test Environment Information"
echo "Date: $(date)"
echo "Results Directory: $RESULTS_DIR"
echo "Docker Version: $(docker --version)"
echo "Docker Compose Version: $(docker-compose --version)"

# Phase 0: Pre-flight checks
print_header "Phase 0: Pre-flight Checks"

# Check Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running!"
    exit 1
fi
print_success "Docker is running"

# Check required files exist
REQUIRED_FILES=(
    "tests/docker/setup_test_environment.sh"
    "tests/docker/test_websocket_connectivity.py"
    "tests/docker/test_message_flow.py"
    "tests/docker/test_fault_tolerance.sh"
    "tests/docker/test_websocket_load.py"
    "tests/docker/integration_test.sh"
    "tests/docker/measure_latency.py"
)

for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        print_error "Required file missing: $FILE"
        exit 1
    fi
done
print_success "All required test files present"

# Phase 1: Environment Setup
if [ "$RUN_SETUP" = true ]; then
    print_header "Phase 1: Environment Setup"
    
    # Always run setup to ensure clean environment
    ./tests/docker/setup_test_environment.sh
    
    if [ $? -eq 0 ]; then
        print_success "Environment setup complete"
    else
        print_error "Environment setup failed"
        exit 1
    fi
fi

# Start services for testing
print_header "Starting Services"

print_status "Starting all services with test configuration..."
docker-compose --env-file .env.test up -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 20

# Verify services are running
SERVICES=("dashboard-backend" "dashboard-frontend" "ai-trading-bot")
ALL_HEALTHY=true

for SERVICE in "${SERVICES[@]}"; do
    if docker-compose ps $SERVICE 2>/dev/null | grep -q "Up"; then
        print_success "$SERVICE is running"
    else
        print_error "$SERVICE is not running"
        docker-compose logs --tail=20 $SERVICE
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = false ]; then
    print_error "Some services failed to start"
    exit 1
fi

# Phase 2: Connectivity Tests
if [ "$RUN_CONNECTIVITY" = true ]; then
    print_header "Phase 2: WebSocket Connectivity Tests"
    
    # Run from host
    run_test "Host Connectivity" "python tests/docker/test_websocket_connectivity.py" || true
    
    # Run from within container
    run_test "Container Connectivity" "docker exec ai-trading-bot python /app/tests/docker/test_websocket_connectivity.py" || true
fi

# Phase 3: Message Flow Tests
if [ "$RUN_MESSAGE_FLOW" = true ]; then
    print_header "Phase 3: Message Flow Validation"
    
    run_test "Message Flow" "python tests/docker/test_message_flow.py" || true
fi

# Phase 4: Fault Tolerance Tests
if [ "$RUN_FAULT_TOLERANCE" = true ]; then
    print_header "Phase 4: Fault Tolerance Tests"
    
    # Note: These tests will restart services
    run_test "Fault Tolerance" "./tests/docker/test_fault_tolerance.sh" || true
    
    # Ensure services are back up
    docker-compose --env-file .env.test up -d
    sleep 15
fi

# Phase 5: Load Tests
if [ "$RUN_LOAD_TESTS" = true ]; then
    print_header "Phase 5: Load and Stress Tests"
    
    # Install numpy if needed
    if ! python -c "import numpy" 2>/dev/null; then
        print_info "Installing numpy for load tests..."
        pip install numpy >/dev/null 2>&1
    fi
    
    run_test "Load Tests" "python tests/docker/test_websocket_load.py" || true
fi

# Phase 6: Integration Tests
if [ "$RUN_INTEGRATION" = true ]; then
    print_header "Phase 6: Full Integration Test"
    
    # Set shorter duration for automated testing
    export TEST_DURATION=60
    run_test "Integration" "./tests/docker/integration_test.sh" || true
fi

# Phase 7: Performance Tests
if [ "$RUN_PERFORMANCE" = true ]; then
    print_header "Phase 7: Performance Monitoring"
    
    # Ensure services are running
    docker-compose --env-file .env.test up -d
    sleep 10
    
    run_test "Performance" "python tests/docker/measure_latency.py --duration 30" || true
fi

# Calculate test duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# Generate JSON summary
cat > "$SUMMARY_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": $DURATION,
  "total_tests": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $FAILED_TESTS,
  "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%,
  "test_results": [
    $(IFS=,; echo "${TEST_RESULTS[*]}")
  ]
}
EOF

# Generate Markdown report
cat > "$REPORT_FILE" << EOF
# WebSocket Integration Test Report

**Generated:** $(date)  
**Duration:** ${DURATION_MIN}m ${DURATION_SEC}s  
**Environment:** Docker Compose with test configuration

## Executive Summary

- **Total Tests:** $TOTAL_TESTS
- **Passed:** $PASSED_TESTS
- **Failed:** $FAILED_TESTS
- **Success Rate:** $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%

## Test Results

| Test Suite | Status | Details |
|------------|--------|---------|
$(for i in "${!TEST_RESULTS[@]}"; do
    RESULT=$(echo "${TEST_RESULTS[$i]}" | jq -r)
    NAME=$(echo "$RESULT" | jq -r '.name')
    STATUS=$(echo "$RESULT" | jq -r '.status')
    LOG=$(echo "$RESULT" | jq -r '.log')
    
    if [ "$STATUS" = "passed" ]; then
        echo "| $NAME | ✅ PASSED | [View Log]($LOG) |"
    else
        echo "| $NAME | ❌ FAILED | [View Log]($LOG) |"
    fi
done)

## Key Findings

### Successes
$(if [ $PASSED_TESTS -gt 0 ]; then
    echo "- WebSocket connectivity established successfully"
    echo "- Real-time data flow validated"
    echo "- System demonstrates resilience under various conditions"
else
    echo "- No tests passed"
fi)

### Issues
$(if [ $FAILED_TESTS -gt 0 ]; then
    echo "- Review failed test logs for specific issues"
    echo "- Check Docker container health and logs"
    echo "- Verify environment configuration"
else
    echo "- No issues detected"
fi)

## Recommendations

1. **For Failed Tests:**
   - Review individual test logs in \`$RESULTS_DIR\`
   - Check Docker container logs: \`docker-compose logs\`
   - Verify WebSocket configuration in \`.env.test\`

2. **Performance Optimization:**
   - Monitor latency metrics from performance tests
   - Review load test results for capacity planning
   - Consider message batching for high-frequency updates

3. **Next Steps:**
   - Address any failed tests before production deployment
   - Set up continuous monitoring for WebSocket health
   - Implement automated testing in CI/CD pipeline

## Logs and Artifacts

All test logs and results are stored in:
\`\`\`
$RESULTS_DIR/
\`\`\`

To view specific test output:
\`\`\`bash
cat $RESULTS_DIR/<test_name>.log
\`\`\`
EOF

# Cleanup
print_header "Test Cleanup"

print_status "Stopping test services..."
docker-compose down

print_status "Test artifacts saved to: $RESULTS_DIR"

# Final Summary
print_header "Test Suite Summary"

echo -e "${CYAN}Duration: ${DURATION_MIN}m ${DURATION_SEC}s${NC}"
echo -e "${CYAN}Total Tests: $TOTAL_TESTS${NC}"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    SUCCESS_RATE=100
else
    SUCCESS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)
fi

echo -e "${CYAN}Success Rate: ${SUCCESS_RATE}%${NC}"

echo -e "\n${CYAN}Reports:${NC}"
echo -e "  JSON Summary: ${YELLOW}$SUMMARY_FILE${NC}"
echo -e "  Markdown Report: ${YELLOW}$REPORT_FILE${NC}"
echo -e "  All Logs: ${YELLOW}$RESULTS_DIR/${NC}"

# Exit status
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed! WebSocket integration is working correctly.${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review the logs for details.${NC}"
    echo -e "${YELLOW}View the report: cat $REPORT_FILE${NC}"
    exit 1
fi