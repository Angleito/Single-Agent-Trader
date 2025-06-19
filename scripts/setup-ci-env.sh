#!/bin/bash

# CI/CD Environment Setup Script
# Creates a .env file with safe defaults for CI/CD environments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

echo "🔧 Setting up CI/CD environment..."

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env file already exists. Creating backup..."
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%s)"
fi

# Create .env file with CI/CD safe defaults
cat > "$ENV_FILE" << 'EOF'
# CI/CD Environment Configuration
# Auto-generated for CI/CD environments

#==============================================================================
# EXCHANGE CONFIGURATION
#==============================================================================
EXCHANGE_TYPE=coinbase

#==============================================================================
# SYSTEM CONFIGURATION
#==============================================================================
SYSTEM__DRY_RUN=true
SYSTEM__ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=false
TESTING=true

#==============================================================================
# TRADING CONFIGURATION
#==============================================================================
SYMBOL=BTC-USD
TIMEFRAME=1h
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02
ENABLE_FUTURES=false
MAX_FUTURES_LEVERAGE=1

#==============================================================================
# AI CONFIGURATION
#==============================================================================
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=4000

#==============================================================================
# MOCK CREDENTIALS (CI/CD ONLY - NON-FUNCTIONAL)
#==============================================================================
# These are placeholder values for CI/CD testing
# They are NOT real credentials and will not work for actual trading

OPENAI_API_KEY=sk-test-key-for-ci-cd-only-not-real
COINBASE_API_KEY=test-coinbase-key-not-real
COINBASE_API_SECRET=test-coinbase-secret-not-real
COINBASE_PASSPHRASE=test-passphrase-not-real
CDP_API_KEY_NAME=organizations/test/apiKeys/test
CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\ntest-key-not-real\n-----END EC PRIVATE KEY-----"
BLUEFIN_PRIVATE_KEY=test-private-key-not-real
BLUEFIN_NETWORK=testnet

#==============================================================================
# FEATURE FLAGS
#==============================================================================
ENABLE_BACKTESTING=false
ENABLE_PAPER_TRADING=true
ENABLE_LIVE_TRADING=false
TECHNICAL_INDICATORS_ENABLED=true

#==============================================================================
# SERVICE CONFIGURATION
#==============================================================================
CONFIG_FILE=./config/development.json
STRATEGY_NAME=vumanchu_cipher

# MCP Configuration
MCP_ENABLED=false
MCP_SERVER_PORT=8765
MCP_MEMORY_RETENTION_DAYS=30

# WebSocket Configuration
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=false
SYSTEM__WEBSOCKET_TIMEOUT=30
SYSTEM__WEBSOCKET_MAX_RETRIES=3

# Optional API Keys (leave empty for CI/CD)
TAVILY_API_KEY=
PERPLEXITY_API_KEY=
KAGI_API_KEY=
JINA_AI_API_KEY=
BRAVE_API_KEY=
FIRECRAWL_API_KEY=
MEM0_API_KEY=
BLUEFIN_SERVICE_API_KEY=

EOF

echo "✅ Created .env file for CI/CD environment"
echo "📍 Location: $ENV_FILE"
echo ""
echo "🔒 Security Notes:"
echo "   - All credentials are mock/placeholder values"
echo "   - DRY_RUN is enabled (no real trading)"
echo "   - Live trading is disabled"
echo "   - Using development configuration"
echo ""
echo "🧪 CI/CD Testing:"
echo "   - Run: docker-compose config"
echo "   - Test: docker-compose up --dry-run"
echo ""
echo "✅ Environment setup complete!"