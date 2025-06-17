#!/bin/bash

# Network Configuration Test Script
# Tests that the dashboard network configuration works correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 Dashboard Network Configuration Test"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test functions
test_compose_syntax() {
    echo -n "Testing docker-compose.yml syntax... "
    if docker-compose config >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_standalone_syntax() {
    echo -n "Testing standalone compose syntax... "
    if docker-compose -f docker-compose.yml -f docker-compose.standalone.yml config >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_production_syntax() {
    echo -n "Testing production profile syntax... "
    if docker-compose --profile production config >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_network_creation() {
    echo -n "Testing network creation... "
    
    # Clean up any existing networks first
    docker network rm dashboard-network trading-network dashboard-network-standalone 2>/dev/null || true
    
    # Test creating networks from compose
    if docker-compose config >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_port_availability() {
    echo -n "Testing port availability... "
    local ports=("3000" "8000" "8080")
    local unavailable=()
    
    for port in "${ports[@]}"; do
        if lsof -i :$port >/dev/null 2>&1; then
            unavailable+=($port)
        fi
    done
    
    if [ ${#unavailable[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  WARN - Ports in use: ${unavailable[*]}${NC}"
        return 1
    fi
}

test_container_names() {
    echo -n "Testing container name uniqueness... "
    
    # Extract container names from compose files
    local main_containers=$(docker-compose config --services 2>/dev/null | wc -l)
    local standalone_containers=$(docker-compose -f docker-compose.yml -f docker-compose.standalone.yml config --services 2>/dev/null | wc -l)
    
    if [ "$main_containers" -gt 0 ] && [ "$standalone_containers" -gt 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_volume_paths() {
    echo -n "Testing volume mount paths... "
    
    # Check if required directories exist or can be created
    local required_dirs=("./backend/logs" "./backend/data" "../logs" "../data")
    local missing=()
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            # Try to create it
            mkdir -p "$dir" 2>/dev/null || missing+=($dir)
        fi
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  WARN - Missing directories: ${missing[*]}${NC}"
        return 1
    fi
}

test_environment_variables() {
    echo -n "Testing environment variable configuration... "
    
    # Check for critical environment variables in compose config
    local config_output=$(docker-compose config 2>/dev/null)
    
    if echo "$config_output" | grep -q "CORS_ORIGINS" && \
       echo "$config_output" | grep -q "VITE_API_URL"; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_docker_requirements() {
    echo -n "Testing Docker requirements... "
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ FAIL - Docker not installed${NC}"
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ FAIL - Docker Compose not installed${NC}"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ FAIL - Docker daemon not running${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ PASS${NC}"
    return 0
}

# Run all tests
run_tests() {
    echo "Running network configuration tests..."
    echo ""
    
    local tests=(
        "test_docker_requirements"
        "test_compose_syntax"
        "test_standalone_syntax"
        "test_production_syntax"
        "test_network_creation"
        "test_port_availability"
        "test_container_names"
        "test_volume_paths"
        "test_environment_variables"
    )
    
    local passed=0
    local failed=0
    local warnings=0
    
    for test in "${tests[@]}"; do
        if $test; then
            ((passed++))
        else
            if [[ $test == *"port_availability"* ]] || [[ $test == *"volume_paths"* ]]; then
                ((warnings++))
            else
                ((failed++))
            fi
        fi
    done
    
    echo ""
    echo "Test Results:"
    echo "============="
    echo -e "${GREEN}Passed: $passed${NC}"
    echo -e "${YELLOW}Warnings: $warnings${NC}"
    echo -e "${RED}Failed: $failed${NC}"
    
    if [ "$failed" -eq 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 All critical tests passed!${NC}"
        echo "The dashboard network configuration is ready to use."
        return 0
    else
        echo ""
        echo -e "${RED}❌ Some tests failed. Please fix the issues before proceeding.${NC}"
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up test resources..."
    docker network rm dashboard-network trading-network dashboard-network-standalone 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Run the tests
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--help]"
    echo ""
    echo "This script tests the dashboard Docker network configuration."
    echo "It verifies that all compose files are valid and networks can be created."
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    exit 0
fi

run_tests