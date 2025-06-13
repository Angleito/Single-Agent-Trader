#!/bin/bash
# Comprehensive health check script for AI Trading Bot

set -e

echo "Starting comprehensive health check..."

# Function to check HTTP endpoint
check_endpoint() {
    local url="$1"
    local name="$2"
    if curl -sf "$url" > /dev/null 2>&1; then
        echo "✓ $name is responding"
        return 0
    else
        echo "✗ $name is not responding"
        return 1
    fi
}

# Function to check process health
check_process() {
    if python -c "
import os
import sys
try:
    # Check if the main bot process is running by looking for the import
    import bot.main
    print('✓ Bot process is accessible')
except ImportError as e:
    print(f'✗ Bot process import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ Bot process check failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        return 0
    else
        echo "✗ Bot process check failed"
        return 1
    fi
}

# Function to check configuration
check_config() {
    if python -c "
import sys
sys.path.append('/app')
try:
    from bot.config import settings
    print('✓ Configuration loaded successfully')
    
    # Check critical settings
    if not settings.llm.openai_api_key:
        print('✗ OpenAI API key not configured')
        sys.exit(1)
    
    # Check exchange credentials for non-dry-run mode
    if not settings.system.dry_run:
        has_legacy = all([
            settings.exchange.cb_api_key,
            settings.exchange.cb_api_secret,
            settings.exchange.cb_passphrase,
        ])
        has_cdp = all([
            settings.exchange.cdp_api_key_name,
            settings.exchange.cdp_private_key,
        ])
        if not (has_legacy or has_cdp):
            print('✗ Exchange API credentials not configured for live trading')
            sys.exit(1)
    
    print('✓ Configuration validation passed')
except Exception as e:
    print(f'✗ Configuration error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        return 0
    else
        echo "✗ Configuration validation failed"
        return 1
    fi
}

# Function to check market data connectivity
check_market_data() {
    if python -c "
import sys
sys.path.append('/app')
import asyncio
import aiohttp

async def check_coinbase():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.exchange.coinbase.com/time', timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print('✓ Coinbase API connectivity OK')
                    return True
                else:
                    print(f'✗ Coinbase API returned status {response.status}')
                    return False
    except Exception as e:
        print(f'✗ Coinbase API connectivity failed: {e}')
        return False

result = asyncio.run(check_coinbase())
sys.exit(0 if result else 1)
" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check log files
check_logs() {
    if [ -d "/app/logs" ] && [ "$(ls -A /app/logs 2>/dev/null)" ]; then
        echo "✓ Log directory exists and has files"
        return 0
    else
        echo "? Log directory empty or missing (may be normal on startup)"
        return 0  # Don't fail for missing logs
    fi
}

# Run all health checks
FAILED=0

echo "=== Process Health ==="
check_process || FAILED=1

echo "=== Configuration Health ==="
check_config || FAILED=1

echo "=== Market Data Connectivity ==="
check_market_data || FAILED=1

echo "=== Log Files ==="
check_logs || FAILED=1

# Summary
if [ $FAILED -eq 0 ]; then
    echo "=== Health Check Result ==="
    echo "✓ All health checks passed"
    exit 0
else
    echo "=== Health Check Result ==="
    echo "✗ One or more health checks failed"
    exit 1
fi