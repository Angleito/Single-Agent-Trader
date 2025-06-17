#!/bin/bash

# WebSocket Connectivity Test Script
# Tests all WebSocket endpoints and validates connectivity fixes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
BACKEND_PORT=8000
NGINX_PORT=8080
TIMEOUT=10
VERBOSE=false

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✅ ${message}${NC}"
            ((TESTS_PASSED++))
            ;;
        "FAILED")
            echo -e "${RED}❌ ${message}${NC}"
            ((TESTS_FAILED++))
            ;;
        "SKIPPED")
            echo -e "${YELLOW}⚠️  ${message}${NC}"
            ((TESTS_SKIPPED++))
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  ${message}${NC}"
            ;;
    esac
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    print_status "INFO" "Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            print_status "SUCCESS" "$service_name is ready on $host:$port"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    
    print_status "FAILED" "$service_name not ready on $host:$port after ${max_attempts}s"
    return 1
}

# Function to test HTTP connection upgrade to WebSocket
test_http_upgrade() {
    local url=$1
    local test_name=$2
    
    print_status "INFO" "Testing HTTP WebSocket upgrade: $test_name"
    
    # Test HTTP connection upgrade
    local response
    response=$(curl -s -w "%{http_code}" \
        -H "Connection: Upgrade" \
        -H "Upgrade: websocket" \
        -H "Sec-WebSocket-Version: 13" \
        -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
        --max-time $TIMEOUT \
        "$url" 2>/dev/null || echo "000")
    
    local http_code="${response: -3}"
    
    if [ "$http_code" = "101" ]; then
        print_status "SUCCESS" "$test_name - HTTP upgrade successful (101 Switching Protocols)"
        return 0
    elif [ "$http_code" = "426" ]; then
        print_status "SUCCESS" "$test_name - WebSocket upgrade required (426 Upgrade Required)"
        return 0
    elif [ "$http_code" = "000" ]; then
        print_status "FAILED" "$test_name - Connection failed (timeout or connection refused)"
        return 1
    else
        print_status "FAILED" "$test_name - Unexpected HTTP code: $http_code"
        return 1
    fi
}

# Function to test WebSocket connection with wscat
test_websocket_wscat() {
    local ws_url=$1
    local test_name=$2
    
    print_status "INFO" "Testing WebSocket connection with wscat: $test_name"
    
    # Create a temporary script to test WebSocket
    local temp_script=$(mktemp)
    cat > "$temp_script" << 'EOF'
const WebSocket = require('ws');
const url = process.argv[2];
const timeout = parseInt(process.argv[3]) * 1000;

const ws = new WebSocket(url);
let connected = false;

const timer = setTimeout(() => {
    if (!connected) {
        console.log('TIMEOUT');
        process.exit(1);
    }
}, timeout);

ws.on('open', function open() {
    connected = true;
    console.log('CONNECTED');
    clearTimeout(timer);
    ws.close();
    process.exit(0);
});

ws.on('error', function error(err) {
    console.log('ERROR:', err.message);
    clearTimeout(timer);
    process.exit(1);
});

ws.on('close', function close() {
    if (connected) {
        console.log('CLOSED');
        process.exit(0);
    }
});
EOF

    # Test with Node.js if available
    if command_exists node && [ -f package.json ]; then
        local result
        result=$(timeout $TIMEOUT node "$temp_script" "$ws_url" $TIMEOUT 2>&1 || echo "FAILED")
        
        if [[ "$result" == *"CONNECTED"* ]]; then
            print_status "SUCCESS" "$test_name - WebSocket connection established"
            rm -f "$temp_script"
            return 0
        else
            print_status "FAILED" "$test_name - WebSocket connection failed: $result"
            rm -f "$temp_script"
            return 1
        fi
    else
        print_status "SKIPPED" "$test_name - Node.js not available for WebSocket test"
        rm -f "$temp_script"
        return 0
    fi
}

# Function to test CORS preflight
test_cors_preflight() {
    local url=$1
    local test_name=$2
    
    print_status "INFO" "Testing CORS preflight: $test_name"
    
    local response
    response=$(curl -s -w "%{http_code}" \
        -X OPTIONS \
        -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: GET" \
        -H "Access-Control-Request-Headers: upgrade,connection,sec-websocket-key,sec-websocket-version" \
        --max-time $TIMEOUT \
        "$url" 2>/dev/null || echo "000")
    
    local http_code="${response: -3}"
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "204" ]; then
        print_status "SUCCESS" "$test_name - CORS preflight successful ($http_code)"
        return 0
    elif [ "$http_code" = "000" ]; then
        print_status "FAILED" "$test_name - CORS preflight failed (connection error)"
        return 1
    else
        print_status "FAILED" "$test_name - CORS preflight failed (HTTP $http_code)"
        return 1
    fi
}

# Function to test container networking
test_container_networking() {
    print_status "INFO" "Testing container networking..."
    
    # Check if we're running in a container
    if [ -f /.dockerenv ]; then
        print_status "INFO" "Running inside container - testing internal networking"
        
        # Test container-to-container communication
        if nc -z dashboard-backend 8000 2>/dev/null; then
            print_status "SUCCESS" "Container networking - dashboard-backend:8000 reachable"
        else
            print_status "FAILED" "Container networking - dashboard-backend:8000 not reachable"
        fi
        
        if nc -z dashboard-nginx 80 2>/dev/null; then
            print_status "SUCCESS" "Container networking - dashboard-nginx:80 reachable"
        else
            print_status "FAILED" "Container networking - dashboard-nginx:80 not reachable"
        fi
    else
        print_status "INFO" "Running outside container - testing Docker services"
        
        # Check if Docker services are running
        if docker ps --format "table {{.Names}}" 2>/dev/null | grep -q "dashboard-backend"; then
            print_status "SUCCESS" "Container networking - dashboard-backend container is running"
        else
            print_status "SKIPPED" "Container networking - dashboard-backend container not running"
        fi
        
        if docker ps --format "table {{.Names}}" 2>/dev/null | grep -q "dashboard-nginx"; then
            print_status "SUCCESS" "Container networking - dashboard-nginx container is running"
        else
            print_status "SKIPPED" "Container networking - dashboard-nginx container not running"
        fi
    fi
}

# Function to run all tests
run_all_tests() {
    echo "===========================================" 
    echo "WebSocket Connectivity Test Suite"
    echo "==========================================="
    echo ""
    
    # Check required tools
    print_status "INFO" "Checking required tools..."
    
    if ! command_exists curl; then
        print_status "FAILED" "curl is required but not installed"
        exit 1
    fi
    
    if ! command_exists nc; then
        print_status "SKIPPED" "netcat (nc) not available - some tests will be skipped"
    fi
    
    if ! command_exists node; then
        print_status "SKIPPED" "Node.js not available - WebSocket connection tests will be skipped"
    fi
    
    echo ""
    print_status "INFO" "Starting connectivity tests..."
    echo ""
    
    # Test 1: Direct backend HTTP upgrade
    if nc -z localhost $BACKEND_PORT 2>/dev/null; then
        test_http_upgrade "http://localhost:$BACKEND_PORT/ws" "Direct backend WebSocket (HTTP upgrade)"
    else
        print_status "FAILED" "Direct backend not reachable on localhost:$BACKEND_PORT"
    fi
    
    # Test 2: Nginx proxy HTTP upgrade
    if nc -z localhost $NGINX_PORT 2>/dev/null; then
        test_http_upgrade "http://localhost:$NGINX_PORT/api/ws" "Nginx proxy WebSocket (HTTP upgrade)"
    else
        print_status "FAILED" "Nginx proxy not reachable on localhost:$NGINX_PORT"
    fi
    
    # Test 3: Direct backend WebSocket connection
    if command_exists node && nc -z localhost $BACKEND_PORT 2>/dev/null; then
        test_websocket_wscat "ws://localhost:$BACKEND_PORT/ws" "Direct backend WebSocket (full connection)"
    else
        print_status "SKIPPED" "Direct backend WebSocket test - service not ready or Node.js unavailable"
    fi
    
    # Test 4: Nginx proxy WebSocket connection
    if command_exists node && nc -z localhost $NGINX_PORT 2>/dev/null; then
        test_websocket_wscat "ws://localhost:$NGINX_PORT/api/ws" "Nginx proxy WebSocket (full connection)"
    else
        print_status "SKIPPED" "Nginx proxy WebSocket test - service not ready or Node.js unavailable"
    fi
    
    # Test 5: CORS preflight tests
    if nc -z localhost $BACKEND_PORT 2>/dev/null; then
        test_cors_preflight "http://localhost:$BACKEND_PORT/ws" "Direct backend CORS preflight"
    fi
    
    if nc -z localhost $NGINX_PORT 2>/dev/null; then
        test_cors_preflight "http://localhost:$NGINX_PORT/api/ws" "Nginx proxy CORS preflight"
    fi
    
    # Test 6: Container networking
    test_container_networking
    
    # Test 7: Error handling tests
    echo ""
    print_status "INFO" "Testing error handling..."
    
    # Test invalid endpoint
    test_http_upgrade "http://localhost:$BACKEND_PORT/invalid-ws" "Invalid endpoint handling"
    
    # Test port that should be closed
    test_http_upgrade "http://localhost:9999/ws" "Closed port handling"
    
    echo ""
    echo "===========================================" 
    echo "Test Results Summary"
    echo "==========================================="
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "${YELLOW}Skipped: $TESTS_SKIPPED${NC}"
    echo "Total: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
    echo ""
    
    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}Some tests failed. Check the output above for details.${NC}"
        return 1
    else
        echo -e "${GREEN}All tests passed or skipped safely!${NC}"
        return 0
    fi
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "OPTIONS:"
                echo "  -v, --verbose     Enable verbose output"
                echo "  -t, --timeout N   Set timeout for tests (default: 10 seconds)"
                echo "  -h, --help        Show this help message"
                echo ""
                echo "This script tests WebSocket connectivity for the trading bot dashboard."
                echo "It validates both direct backend connections and nginx proxy routes."
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run the test suite
    if run_all_tests; then
        exit 0
    else
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"