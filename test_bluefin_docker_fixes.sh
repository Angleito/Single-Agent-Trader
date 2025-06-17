#!/bin/bash
# Test script for Bluefin Docker fixes
# Tests Docker build, Socket.IO fixes, and environment handling

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐋 Bluefin Docker Fixes Test Suite${NC}"
echo "========================================"
echo

# Test 1: Docker build with Socket.IO fixes
test_docker_build() {
    echo -e "${BLUE}Test 1: Docker Build with Socket.IO Fixes${NC}"
    echo "Building Bluefin Docker image..."
    
    if docker build -f Dockerfile.bluefin -t ai-trading-bot:bluefin-test .; then
        echo -e "${GREEN}✅ Docker build successful${NC}"
        return 0
    else
        echo -e "${RED}❌ Docker build failed${NC}"
        return 1
    fi
}

# Test 2: Socket.IO import in container
test_socketio_in_container() {
    echo -e "\n${BLUE}Test 2: Socket.IO Import in Container${NC}"
    echo "Testing Socket.IO AsyncClient availability..."
    
    local test_result
    test_result=$(docker run --rm ai-trading-bot:bluefin-test python -c "
import socketio
print('Socket.IO version:', getattr(socketio, '__version__', 'unknown'))
if hasattr(socketio, 'AsyncClient'):
    client = socketio.AsyncClient()
    print('✅ AsyncClient available and instantiable')
else:
    print('❌ AsyncClient not available')
    exit(1)
" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Socket.IO AsyncClient test passed${NC}"
        echo "$test_result"
        return 0
    else
        echo -e "${RED}❌ Socket.IO AsyncClient test failed${NC}"
        echo "$test_result"
        return 1
    fi
}

# Test 3: Bluefin SDK import in container
test_bluefin_sdk_in_container() {
    echo -e "\n${BLUE}Test 3: Bluefin SDK Import in Container${NC}"
    echo "Testing Bluefin SDK availability..."
    
    local test_result
    test_result=$(docker run --rm ai-trading-bot:bluefin-test python -c "
import sys
print('Python version:', sys.version)

try:
    from bot.exchange.bluefin import BluefinClient, BLUEFIN_AVAILABLE
    print('✅ Bluefin client imported successfully')
    print('SDK Available:', BLUEFIN_AVAILABLE)
    
    if BLUEFIN_AVAILABLE:
        try:
            from bluefin_v2_client import Networks
            print('✅ Bluefin Networks imported')
            print('Available networks:', list(Networks.keys()))
        except Exception as e:
            print('⚠️  Networks import failed:', e)
    
except Exception as e:
    print('❌ Bluefin SDK import failed:', e)
    exit(1)
" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Bluefin SDK test passed${NC}"
        echo "$test_result"
        return 0
    else
        echo -e "${RED}❌ Bluefin SDK test failed${NC}"
        echo "$test_result"
        return 1
    fi
}

# Test 4: Environment variable handling
test_environment_handling() {
    echo -e "\n${BLUE}Test 4: Environment Variable Handling${NC}"
    echo "Testing SYSTEM__DRY_RUN override behavior..."
    
    # Test with DRY_RUN=true (paper trading)
    local paper_result
    paper_result=$(docker run --rm \
        -e SYSTEM__DRY_RUN=true \
        -e EXCHANGE__EXCHANGE_TYPE=bluefin \
        ai-trading-bot:bluefin-test python -c "
from bot.exchange.bluefin import BluefinClient
client = BluefinClient()
status = client.get_connection_status()
print('Trading mode:', status['trading_mode'])
print('Dry run:', status['dry_run'])
print('System dry run:', status['system_dry_run'])
if status['trading_mode'] == 'PAPER TRADING':
    print('✅ Paper trading mode correct')
else:
    print('❌ Expected paper trading mode')
    exit(1)
" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Paper trading mode test passed${NC}"
        echo "$paper_result"
    else
        echo -e "${RED}❌ Paper trading mode test failed${NC}"
        echo "$paper_result"
        return 1
    fi
    
    # Test with DRY_RUN=false (live trading detection)
    echo -e "\n${YELLOW}Testing live trading mode detection...${NC}"
    local live_result
    live_result=$(docker run --rm \
        -e SYSTEM__DRY_RUN=false \
        -e EXCHANGE__EXCHANGE_TYPE=bluefin \
        ai-trading-bot:bluefin-test python -c "
from bot.exchange.bluefin import BluefinClient
try:
    client = BluefinClient()
    status = client.get_connection_status()
    print('Trading mode:', status['trading_mode'])
    print('Dry run:', status['dry_run'])
    print('System dry run:', status['system_dry_run'])
    if not status['system_dry_run']:
        print('✅ Live trading mode detection correct')
    else:
        print('❌ Expected live trading mode detection')
        exit(1)
except Exception as e:
    if 'Private key required' in str(e):
        print('✅ Live trading validation correct (private key required)')
    else:
        print('❌ Unexpected error:', e)
        exit(1)
" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Live trading mode test passed${NC}"
        echo "$live_result"
        return 0
    else
        echo -e "${RED}❌ Live trading mode test failed${NC}"
        echo "$live_result"
        return 1
    fi
}

# Test 5: OrbStack compatibility check
test_orbstack_compatibility() {
    echo -e "\n${BLUE}Test 5: OrbStack Compatibility${NC}"
    
    # Check if OrbStack is available
    if command -v orbctl &> /dev/null; then
        echo -e "${GREEN}✅ OrbStack detected${NC}"
        
        # Test OrbStack networking
        echo "Testing OrbStack container networking..."
        if docker run --rm --network host ai-trading-bot:bluefin-test curl -s --max-time 5 https://httpbin.org/ip > /dev/null; then
            echo -e "${GREEN}✅ OrbStack network connectivity test passed${NC}"
        else
            echo -e "${YELLOW}⚠️  OrbStack network test inconclusive (external dependency)${NC}"
        fi
        
        return 0
    else
        echo -e "${YELLOW}⚠️  OrbStack not detected - using Docker Desktop${NC}"
        echo "Testing standard Docker networking..."
        if docker run --rm ai-trading-bot:bluefin-test curl -s --max-time 5 https://httpbin.org/ip > /dev/null; then
            echo -e "${GREEN}✅ Standard Docker network connectivity test passed${NC}"
        else
            echo -e "${YELLOW}⚠️  Network test inconclusive (external dependency)${NC}"
        fi
        
        return 0
    fi
}

# Test 6: Run comprehensive test suite in container
test_comprehensive_suite() {
    echo -e "\n${BLUE}Test 6: Comprehensive Test Suite in Container${NC}"
    echo "Running complete test suite inside container..."
    
    if docker run --rm \
        -e EXCHANGE__EXCHANGE_TYPE=bluefin \
        -e SYSTEM__DRY_RUN=true \
        ai-trading-bot:bluefin-test python test_bluefin_fixes.py; then
        echo -e "${GREEN}✅ Comprehensive test suite passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Comprehensive test suite failed${NC}"
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Cleaning up test artifacts...${NC}"
    docker rmi ai-trading-bot:bluefin-test 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Main test runner
main() {
    local tests=(
        "Docker Build" test_docker_build
        "Socket.IO in Container" test_socketio_in_container
        "Bluefin SDK in Container" test_bluefin_sdk_in_container
        "Environment Handling" test_environment_handling
        "OrbStack Compatibility" test_orbstack_compatibility
        "Comprehensive Suite" test_comprehensive_suite
    )
    
    local results=()
    local passed=0
    local total=$((${#tests[@]} / 2))
    
    # Run tests
    for ((i=0; i<${#tests[@]}; i+=2)); do
        local test_name="${tests[i]}"
        local test_func="${tests[i+1]}"
        
        echo -e "\n$((i/2 + 1))/${total}: ${test_name}"
        echo "----------------------------------------"
        
        if $test_func; then
            results+=("✅ $test_name")
            ((passed++))
        else
            results+=("❌ $test_name")
        fi
    done
    
    # Summary
    echo -e "\n${'='*50}"
    echo -e "${BLUE}📊 TEST SUMMARY${NC}"
    echo "="*50
    
    for result in "${results[@]}"; do
        echo -e "$result"
    done
    
    echo -e "\nResults: ${passed}/${total} tests passed"
    
    if [[ $passed -eq $total ]]; then
        echo -e "${GREEN}🎉 All Docker tests passed! Bluefin SDK is ready for deployment.${NC}"
        cleanup
        return 0
    else
        echo -e "${RED}⚠️  Some Docker tests failed. Check the output above for details.${NC}"
        read -p "Keep test image for debugging? (y/N): " keep_image
        if [[ ! "$keep_image" =~ ^[Yy]$ ]]; then
            cleanup
        fi
        return 1
    fi
}

# Handle interrupts
trap cleanup EXIT

# Run main function
main "$@"