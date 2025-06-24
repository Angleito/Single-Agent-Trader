#!/bin/bash

# ==================================================================================
# Docker Configuration Validation Script
# ==================================================================================
# This script validates that Docker Compose configuration matches the .env settings
# and helps identify configuration mismatches between exchanges
# ==================================================================================

set -e

echo "🔍 Docker Configuration Validation"
echo "=================================="

# Load .env file if it exists
if [ -f .env ]; then
    echo "✅ Found .env file"
    source .env
else
    echo "⚠️  No .env file found - using defaults"
    EXCHANGE__EXCHANGE_TYPE="bluefin"
    TRADING__SYMBOL="SUI-PERP"
fi

echo
echo "📋 Current Configuration:"
echo "   Exchange Type: ${EXCHANGE__EXCHANGE_TYPE:-bluefin}"
echo "   Trading Symbol: ${TRADING__SYMBOL:-SUI-PERP}"
echo "   Trading Interval: ${TRADING__INTERVAL:-1m}"
echo "   Dry Run Mode: ${SYSTEM__DRY_RUN:-true}"

echo
echo "🔄 Validating Configuration Consistency..."

# Validate exchange and symbol compatibility
EXCHANGE_TYPE="${EXCHANGE__EXCHANGE_TYPE:-bluefin}"
SYMBOL="${TRADING__SYMBOL:-SUI-PERP}"

case "$EXCHANGE_TYPE" in
    "coinbase")
        if [[ "$SYMBOL" == *"-PERP" ]]; then
            echo "❌ ERROR: Coinbase exchange cannot trade PERP symbols ($SYMBOL)"
            echo "   Coinbase symbols should be like: BTC-USD, ETH-USD, SOL-USD"
            echo "   Update TRADING__SYMBOL in your .env file"
            exit 1
        else
            echo "✅ Coinbase configuration looks valid"
        fi
        ;;
    "bluefin")
        if [[ "$SYMBOL" != *"-PERP" && "$SYMBOL" != *"-USD" ]]; then
            echo "⚠️  WARNING: Bluefin typically uses PERP symbols ($SYMBOL)"
            echo "   Common Bluefin symbols: BTC-PERP, ETH-PERP, SUI-PERP, SOL-PERP"
        else
            echo "✅ Bluefin configuration looks valid"
        fi
        ;;
    *)
        echo "❌ ERROR: Unknown exchange type: $EXCHANGE_TYPE"
        echo "   Supported exchanges: coinbase, bluefin"
        exit 1
        ;;
esac

echo
echo "🐳 Checking Docker Compose Configuration..."

# Check if docker-compose.yml exists
if [ ! -f docker-compose.yml ]; then
    echo "❌ ERROR: docker-compose.yml not found"
    exit 1
fi

# Validate docker-compose configuration
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is available"
    
    # Check configuration syntax
    if docker-compose config > /dev/null 2>&1; then
        echo "✅ Docker Compose configuration syntax is valid"
    else
        echo "❌ ERROR: Docker Compose configuration has syntax errors"
        docker-compose config
        exit 1
    fi
else
    echo "⚠️  Docker Compose not available - skipping syntax check"
fi

echo
echo "📁 Checking Required Directories..."

# Check required directories
REQUIRED_DIRS=("logs" "data" "config")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory exists: $dir/"
    else
        echo "⚠️  Creating missing directory: $dir/"
        mkdir -p "$dir"
    fi
done

echo
echo "🔑 Checking API Key Configuration..."

# Check API keys based on exchange type
case "$EXCHANGE_TYPE" in
    "coinbase")
        if [ -z "${EXCHANGE__CDP_API_KEY_NAME:-}" ] || [ -z "${EXCHANGE__CDP_PRIVATE_KEY:-}" ]; then
            echo "⚠️  WARNING: Coinbase API credentials not configured"
            echo "   Set EXCHANGE__CDP_API_KEY_NAME and EXCHANGE__CDP_PRIVATE_KEY in .env"
        else
            echo "✅ Coinbase API credentials configured"
        fi
        ;;
    "bluefin")
        if [ -z "${EXCHANGE__BLUEFIN_PRIVATE_KEY:-}" ]; then
            echo "⚠️  WARNING: Bluefin private key not configured"
            echo "   Set EXCHANGE__BLUEFIN_PRIVATE_KEY in .env"
        else
            echo "✅ Bluefin private key configured"
        fi
        
        if [ -z "${BLUEFIN_SERVICE_API_KEY:-}" ]; then
            echo "⚠️  WARNING: Bluefin service API key not configured"
            echo "   Set BLUEFIN_SERVICE_API_KEY in .env"
        else
            echo "✅ Bluefin service API key configured"
        fi
        ;;
esac

# Check LLM API key
if [ -z "${LLM__OPENAI_API_KEY:-}" ]; then
    echo "⚠️  WARNING: OpenAI API key not configured"
    echo "   Set LLM__OPENAI_API_KEY in .env"
else
    echo "✅ OpenAI API key configured"
fi

echo
echo "🚀 Configuration Summary:"
echo "========================"

if [ "${SYSTEM__DRY_RUN:-true}" = "true" ]; then
    echo "✅ SAFE MODE: Paper trading enabled (no real money)"
else
    echo "⚠️  LIVE MODE: Real trading enabled (REAL MONEY AT RISK!)"
fi

echo "   Exchange: $EXCHANGE_TYPE"
echo "   Symbol: $SYMBOL"
echo "   Environment: ${SYSTEM__ENVIRONMENT:-development}"

echo
echo "🏁 Configuration validation complete!"

if [ "${SYSTEM__DRY_RUN:-true}" = "true" ]; then
    echo "✅ Safe to run: docker-compose up"
else
    echo "⚠️  CAUTION: Live trading mode - verify all settings before running!"
fi

echo
echo "📚 Next Steps:"
echo "   1. Review the configuration above"
echo "   2. Run: docker-compose up"
echo "   3. Monitor logs: docker-compose logs -f ai-trading-bot"
echo "   4. Access dashboard: http://localhost:8080 (if enabled)"