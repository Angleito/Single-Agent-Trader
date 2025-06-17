#!/bin/bash
# End-to-End Integration Test for Bluefin Dashboard
# Tests the complete flow: Frontend -> Dashboard Backend -> Bluefin Service

set -e
echo "🚀 Starting End-to-End Integration Test"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local url=$1
    local description=$2
    local expected_content=$3

    echo -n "🔍 Testing $description... "

    response=$(curl -s -w "%{http_code}" "$url" -o /tmp/test_response)
    http_code="${response: -3}"

    if [[ "$http_code" == "200" ]]; then
        if [[ -n "$expected_content" ]]; then
            if grep -q "$expected_content" /tmp/test_response; then
                echo -e "${GREEN}✅ PASS${NC}"
                return 0
            else
                echo -e "${RED}❌ FAIL - Content not found${NC}"
                echo "Expected: $expected_content"
                echo "Response: $(cat /tmp/test_response | head -200)"
                return 1
            fi
        else
            echo -e "${GREEN}✅ PASS${NC}"
            return 0
        fi
    else
        echo -e "${RED}❌ FAIL - HTTP $http_code${NC}"
        echo "Response: $(cat /tmp/test_response)"
        return 1
    fi
}

# Function to test JSON endpoint
test_json_endpoint() {
    local url=$1
    local description=$2
    local json_key=$3

    echo -n "🔍 Testing $description... "

    response=$(curl -s "$url")
    http_code=$(curl -s -w "%{http_code}" "$url" -o /dev/null)

    if [[ "$http_code" == "200" ]]; then
        if [[ -n "$json_key" ]]; then
            if echo "$response" | jq -e "$json_key" > /dev/null 2>&1; then
                echo -e "${GREEN}✅ PASS${NC}"
                return 0
            else
                echo -e "${RED}❌ FAIL - JSON key '$json_key' not found${NC}"
                echo "Response: $response" | head -200
                return 1
            fi
        else
            if echo "$response" | jq . > /dev/null 2>&1; then
                echo -e "${GREEN}✅ PASS${NC}"
                return 0
            else
                echo -e "${RED}❌ FAIL - Invalid JSON${NC}"
                echo "Response: $response" | head -200
                return 1
            fi
        fi
    else
        echo -e "${RED}❌ FAIL - HTTP $http_code${NC}"
        return 1
    fi
}

echo ""
echo "📋 Test Plan:"
echo "1. Frontend accessibility"
echo "2. Dashboard Backend health"
echo "3. Bluefin Service connectivity"
echo "4. Frontend -> Backend API proxy"
echo "5. Backend -> Bluefin Service integration"
echo "6. End-to-end trading mode configuration"
echo ""

# Test 1: Frontend accessibility
echo "🎯 Phase 1: Frontend Tests"
test_endpoint "http://localhost:3000/" "Frontend homepage" "<!DOCTYPE html>"

# Test 2: Dashboard Backend health
echo ""
echo "🎯 Phase 2: Dashboard Backend Tests"
test_json_endpoint "http://localhost:8000/health" "Backend health check" ".status"
test_json_endpoint "http://localhost:8000/api/websocket/status" "WebSocket status" ".active_connections"

# Test 3: Bluefin Service connectivity
echo ""
echo "🎯 Phase 3: Bluefin Service Tests"
test_json_endpoint "http://localhost:8000/api/bluefin/health" "Bluefin health via backend" ".health.status"
test_json_endpoint "http://localhost:8000/api/bluefin/account" "Bluefin account via backend" ".account.address"

# Test 4: Frontend -> Backend API proxy
echo ""
echo "🎯 Phase 4: Frontend API Proxy Tests"
test_json_endpoint "http://localhost:3000/api/trading-mode" "Trading mode via frontend" ".exchange_type"
test_json_endpoint "http://localhost:3000/api/trading-data" "Trading data via frontend" ".exchange_type"
test_json_endpoint "http://localhost:3000/api/bluefin/health" "Bluefin health via frontend" ".health.status"

# Test 5: Bluefin-specific data validation
echo ""
echo "🎯 Phase 5: Bluefin Integration Validation"
echo -n "🔍 Validating Bluefin exchange type... "
response=$(curl -s "http://localhost:8000/api/trading-mode")
exchange_type=$(echo "$response" | jq -r '.exchange_type')
if [[ "$exchange_type" == "bluefin" ]]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL - Expected 'bluefin', got '$exchange_type'${NC}"
    exit 1
fi

echo -n "🔍 Validating Bluefin DEX features... "
dex_trading=$(echo "$response" | jq -r '.features.dex_trading')
sui_blockchain=$(echo "$response" | jq -r '.features.sui_blockchain')
if [[ "$dex_trading" == "true" && "$sui_blockchain" == "true" ]]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL - DEX features not enabled${NC}"
    exit 1
fi

echo -n "🔍 Validating Bluefin perpetual symbols... "
symbols=$(echo "$response" | jq -r '.supported_symbols.futures[]' | grep -c "PERP" || true)
if [[ "$symbols" -gt 0 ]]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL - No PERP symbols found${NC}"
    exit 1
fi

echo -n "🔍 Validating Bluefin service connection... "
service_status=$(echo "$response" | jq -r '.exchange_config.service_status')
if [[ "$service_status" == "connected" ]]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL - Service status: $service_status${NC}"
    exit 1
fi

# Test 6: Real account data
echo ""
echo "🎯 Phase 6: Live Data Validation"
echo -n "🔍 Validating real Bluefin account data... "
account_response=$(curl -s "http://localhost:8000/api/bluefin/account")
address=$(echo "$account_response" | jq -r '.account.address')
if [[ "$address" =~ ^0x[a-fA-F0-9]+$ ]]; then
    echo -e "${GREEN}✅ PASS${NC}"
    echo "   Account: $address"
else
    echo -e "${RED}❌ FAIL - Invalid address format: $address${NC}"
    exit 1
fi

# Clean up
rm -f /tmp/test_response

echo ""
echo "🎉 Integration Test Results:"
echo "========================================="
echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""
echo "🔗 Dashboard URLs:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   Bluefin:   http://localhost:8081 (external access)"
echo ""
echo "🛠️  Key Integration Points Verified:"
echo "   ✅ Frontend serves static content"
echo "   ✅ Frontend proxies API calls to backend"
echo "   ✅ Backend connects to Bluefin service"
echo "   ✅ Bluefin service is healthy and responding"
echo "   ✅ Exchange type correctly set to 'bluefin'"
echo "   ✅ Bluefin-specific features enabled"
echo "   ✅ Real account data accessible"
echo "   ✅ Perpetual futures symbols available"
echo ""
echo -e "${YELLOW}🚀 The Bluefin dashboard integration is working perfectly!${NC}"
echo "You can now:"
echo "1. Open http://localhost:3000 to access the dashboard"
echo "2. Monitor real Bluefin trading data"
echo "3. Use manual trading controls (if enabled)"
echo "4. View live account information and positions"
