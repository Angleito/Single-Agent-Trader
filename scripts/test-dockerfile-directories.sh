#!/bin/bash
# Test script to verify Dockerfile directory creation
# This script tests that the Dockerfile creates all required directories with proper permissions

set -e

echo "🧪 Testing Dockerfile Directory Creation"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_result() {
    case $1 in
        "PASS") echo -e "${GREEN}✅ $2${NC}" ;;
        "FAIL") echo -e "${RED}❌ $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
        "WARN") echo -e "${YELLOW}⚠️  $2${NC}" ;;
    esac
}

# Build a test container to verify directory creation
TEST_IMAGE="ai-trading-bot-directory-test"

print_result "INFO" "Building test container to verify directory creation..."

if docker build -t "${TEST_IMAGE}" -f Dockerfile . --target production; then
    print_result "PASS" "Docker build completed successfully"
else
    print_result "FAIL" "Docker build failed"
    exit 1
fi

# Test directory structure and permissions
print_result "INFO" "Testing directory structure and permissions..."

# Expected directories with their permissions
EXPECTED_DIRS=(
    "/app:755"
    "/app/config:755" 
    "/app/logs:775"
    "/app/data:775"
    "/app/prompts:755"
    "/app/tmp:775"
    "/app/data/mcp_memory:775"
    "/app/logs/mcp:775"
    "/app/logs/bluefin:775"
    "/app/logs/trades:775"
    "/app/data/orders:775"
    "/app/data/paper_trading:775"
    "/app/data/positions:775"
    "/app/data/bluefin:775"
    "/app/data/omnisearch_cache:775"
)

# Run container and test each directory
TEST_PASSED=0
TEST_TOTAL=0

for dir_perm in "${EXPECTED_DIRS[@]}"; do
    DIR_PATH="${dir_perm%:*}"
    EXPECTED_PERM="${dir_perm#*:}"
    ((TEST_TOTAL++))
    
    # Test if directory exists
    if docker run --rm "${TEST_IMAGE}" test -d "${DIR_PATH}"; then
        # Test permissions
        ACTUAL_PERM=$(docker run --rm "${TEST_IMAGE}" stat -c "%a" "${DIR_PATH}")
        if [ "${ACTUAL_PERM}" = "${EXPECTED_PERM}" ]; then
            print_result "PASS" "Directory ${DIR_PATH} exists with correct permissions (${EXPECTED_PERM})"
            ((TEST_PASSED++))
        else
            print_result "FAIL" "Directory ${DIR_PATH} has wrong permissions: expected ${EXPECTED_PERM}, got ${ACTUAL_PERM}"
        fi
    else
        print_result "FAIL" "Directory ${DIR_PATH} does not exist"
    fi
done

# Test ownership
print_result "INFO" "Testing directory ownership..."
OWNER_TEST=$(docker run --rm "${TEST_IMAGE}" stat -c "%U:%G" "/app/logs")
if [ "${OWNER_TEST}" = "botuser:botuser" ]; then
    print_result "PASS" "Directory ownership is correct (botuser:botuser)"
    ((TEST_PASSED++))
    ((TEST_TOTAL++))
else
    print_result "FAIL" "Directory ownership is wrong: expected botuser:botuser, got ${OWNER_TEST}"
    ((TEST_TOTAL++))
fi

# Test user ID
print_result "INFO" "Testing user ID mapping..."
USER_ID=$(docker run --rm "${TEST_IMAGE}" id -u botuser)
GROUP_ID=$(docker run --rm "${TEST_IMAGE}" id -g botuser)
if [ "${USER_ID}" = "1000" ] && [ "${GROUP_ID}" = "1000" ]; then
    print_result "PASS" "User ID mapping is correct (botuser = 1000:1000)"
    ((TEST_PASSED++))
    ((TEST_TOTAL++))
else
    print_result "FAIL" "User ID mapping is wrong: expected 1000:1000, got ${USER_ID}:${GROUP_ID}"
    ((TEST_TOTAL++))
fi

# Summary
echo ""
echo "📊 Test Results Summary:"
echo "======================="
echo "Tests passed: ${TEST_PASSED}/${TEST_TOTAL}"

if [ "${TEST_PASSED}" -eq "${TEST_TOTAL}" ]; then
    print_result "PASS" "All tests passed! Dockerfile directory setup is working correctly."
    echo ""
    print_result "INFO" "The Dockerfile properly creates all required directories with:"
    print_result "INFO" "- Correct ownership (botuser:botuser = 1000:1000)"
    print_result "INFO" "- Proper permissions (775 for writable, 755 for read-only)"
    print_result "INFO" "- Complete directory structure for the trading bot"
    echo ""
    print_result "INFO" "This reduces the burden on the entrypoint script and provides"
    print_result "INFO" "a solid foundation for container operations."
else
    print_result "FAIL" "Some tests failed. Please review the Dockerfile directory setup."
    exit 1
fi

# Cleanup
print_result "INFO" "Cleaning up test image..."
docker rmi "${TEST_IMAGE}" >/dev/null 2>&1 || true

echo ""
print_result "PASS" "Directory testing completed successfully! 🎉"