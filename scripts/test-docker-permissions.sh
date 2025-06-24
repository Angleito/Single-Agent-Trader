#!/bin/bash
# ==================================================================================
# Docker Permission & API Key Testing Script
# ==================================================================================
# This script validates Docker user permissions and API key configurations
# Run this before starting containers to catch permission and configuration issues
# ==================================================================================

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

echo "🧪 Docker User Permissions & API Key Test"
echo "=========================================="

# Load .env file if it exists
if [ -f .env ]; then
    print_status "INFO" "Loading environment from .env file"
    source .env
else
    print_status "WARNING" "No .env file found - using defaults"
    EXCHANGE__EXCHANGE_TYPE="coinbase"
    SYSTEM__DRY_RUN="true"
fi

# Set host user ID and group ID
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

print_status "INFO" "Host user: $HOST_UID:$HOST_GID ($(whoami):$(id -gn))"
print_status "INFO" "Exchange: ${EXCHANGE__EXCHANGE_TYPE:-coinbase}"
print_status "INFO" "Dry Run: ${SYSTEM__DRY_RUN:-true}"

# Check if logs directory exists and its permissions
if [ -d "./logs" ]; then
    LOGS_OWNER=$(stat -f "%u:%g" ./logs)
    print_status "INFO" "Logs directory owner: $LOGS_OWNER"

    if [ "$LOGS_OWNER" = "$HOST_UID:$HOST_GID" ]; then
        print_status "SUCCESS" "Logs directory permissions match host user"
    else
        print_status "WARNING" "Logs directory has different owner, but Docker user mapping will handle this"
    fi
else
    print_status "ERROR" "Logs directory not found"
    exit 1
fi

# Test write permission with a simple container
print_status "INFO" "Testing write permissions with a test container..."

# Create a simple test using a basic alpine container
docker run --rm \
    --user "$HOST_UID:$HOST_GID" \
    -v "$(pwd)/logs:/test/logs" \
    alpine:latest \
    /bin/sh -c "
        echo 'Docker permission test' > /test/logs/docker-permission-test.txt &&
        echo 'Container can write to host volume' &&
        rm /test/logs/docker-permission-test.txt &&
        echo 'Container can delete from host volume'
    "

if [ $? -eq 0 ]; then
    print_status "SUCCESS" "Docker container can write to host volumes with user $HOST_UID:$HOST_GID"
else
    print_status "ERROR" "Docker container cannot write to host volumes"
    exit 1
fi

# Test environment variable substitution in docker-compose
print_status "INFO" "Testing docker-compose environment variable substitution..."

# Check if docker-compose can resolve the environment variables
if command -v docker-compose >/dev/null 2>&1; then
    # Test config parsing
    docker-compose config > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_status "SUCCESS" "docker-compose configuration is valid"

        # Show the resolved user configuration for ai-trading-bot service
        USER_CONFIG=$(docker-compose config | grep -A 20 "ai-trading-bot:" | grep "user:" | head -1 | sed 's/.*user: //')
        print_status "INFO" "ai-trading-bot service will run as user: $USER_CONFIG"
    else
        print_status "ERROR" "docker-compose configuration is invalid"
        exit 1
    fi
else
    print_status "WARNING" "docker-compose not found, skipping configuration test"
fi

echo ""
print_status "INFO" "=== Testing API Key Configuration ==="

# Test API key configuration based on exchange type
EXCHANGE_TYPE="${EXCHANGE__EXCHANGE_TYPE:-coinbase}"
API_KEY_ERRORS=0

# Test LLM API key
if [ -z "${LLM__OPENAI_API_KEY:-}" ]; then
    print_status "ERROR" "LLM__OPENAI_API_KEY is not set"
    API_KEY_ERRORS=$((API_KEY_ERRORS + 1))
else
    if [[ "${LLM__OPENAI_API_KEY}" =~ ^sk-[A-Za-z0-9]{48,}$ ]]; then
        print_status "SUCCESS" "OpenAI API key format appears valid"
    else
        print_status "WARNING" "OpenAI API key format may be invalid (should start with 'sk-')"
    fi
fi

# Test exchange-specific API keys
case "$EXCHANGE_TYPE" in
    "coinbase")
        print_status "INFO" "Testing Coinbase API configuration..."
        if [ -z "${EXCHANGE__CDP_API_KEY_NAME:-}" ]; then
            print_status "ERROR" "EXCHANGE__CDP_API_KEY_NAME is not set"
            API_KEY_ERRORS=$((API_KEY_ERRORS + 1))
        else
            print_status "SUCCESS" "Coinbase API key name is configured"
        fi
        
        if [ -z "${EXCHANGE__CDP_PRIVATE_KEY:-}" ]; then
            print_status "ERROR" "EXCHANGE__CDP_PRIVATE_KEY is not set"
            API_KEY_ERRORS=$((API_KEY_ERRORS + 1))
        else
            if [[ "${EXCHANGE__CDP_PRIVATE_KEY}" =~ "BEGIN EC PRIVATE KEY" ]]; then
                print_status "SUCCESS" "Coinbase private key format appears valid (PEM)"
            else
                print_status "WARNING" "Coinbase private key may not be in PEM format"
            fi
        fi
        ;;
    "bluefin")
        print_status "INFO" "Testing Bluefin API configuration..."
        if [ -z "${EXCHANGE__BLUEFIN_PRIVATE_KEY:-}" ]; then
            print_status "ERROR" "EXCHANGE__BLUEFIN_PRIVATE_KEY is not set"
            API_KEY_ERRORS=$((API_KEY_ERRORS + 1))
        else
            if [[ "${EXCHANGE__BLUEFIN_PRIVATE_KEY}" =~ ^(0x)?[a-fA-F0-9]{64}$ ]]; then
                print_status "SUCCESS" "Bluefin private key format appears valid (64-char hex)"
            else
                print_status "WARNING" "Bluefin private key format may be invalid"
            fi
        fi
        
        if [ -z "${BLUEFIN_SERVICE_API_KEY:-}" ]; then
            print_status "WARNING" "BLUEFIN_SERVICE_API_KEY is not set"
        else
            print_status "SUCCESS" "Bluefin service API key is configured"
        fi
        ;;
    *)
        print_status "WARNING" "Unknown exchange type: $EXCHANGE_TYPE"
        ;;
esac

echo ""
if [ $API_KEY_ERRORS -eq 0 ]; then
    print_status "SUCCESS" "All tests passed! Docker setup and API keys are ready 🎉"
else
    if [ "${SYSTEM__DRY_RUN:-true}" = "true" ]; then
        print_status "WARNING" "API key issues detected, but paper trading mode is safe"
    else
        print_status "ERROR" "API key errors detected - cannot run in live trading mode"
        echo ""
        print_status "ERROR" "Fix API key configuration or enable paper trading (SYSTEM__DRY_RUN=true)"
    fi
fi
echo ""
echo "📋 Summary:"
echo "==========="
echo "• Host user: $HOST_UID:$HOST_GID ($(whoami):$(id -gn))"
echo "• Docker containers will run as this user to match file permissions"
echo "• Volume mounts will have correct read/write permissions"
echo ""
echo "🚀 Ready to start services:"
echo "   ./scripts/start-trading-bot.sh"
echo "   OR"
echo "   source scripts/setup-user-permissions.sh && docker-compose up -d"
