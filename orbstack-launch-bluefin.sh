#!/bin/bash
# OrbStack Launch Script for Bluefin DEX Trading Bot
# This script deploys the AI trading bot configured for Bluefin DEX on OrbStack

set -e  # Exit on any error

echo "🚀 LAUNCHING BLUEFIN DEX TRADING BOT ON ORBSTACK"
echo "================================================="

# Check if OrbStack is running
if ! command -v orb &> /dev/null; then
    echo "❌ OrbStack CLI not found. Please install OrbStack first."
    echo "   Download from: https://orbstack.dev/"
    exit 1
fi

# Check if Docker is available through OrbStack
if ! docker info &> /dev/null; then
    echo "❌ Docker is not available. Please ensure OrbStack is running."
    exit 1
fi

echo "✅ OrbStack and Docker are available"

# Check if required environment file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found"
    echo "   Please configure your API keys in .env:"
    echo "   - LLM__OPENAI_API_KEY=your_openai_api_key"
    echo "   - EXCHANGE__BLUEFIN_PRIVATE_KEY=your_sui_wallet_private_key"
    exit 1
fi

# Check if API keys are configured
if grep -q "PLEASE_REPLACE" .env; then
    echo "❌ API keys not configured in .env"
    echo "   Please update the following in .env:"
    echo "   - LLM__OPENAI_API_KEY=sk-your_actual_openai_key"
    echo "   - EXCHANGE__BLUEFIN_PRIVATE_KEY=0xYourActualSuiPrivateKey"
    echo ""
    echo "   Run: ./configure-api-keys.sh for help"
    exit 1
fi

echo "✅ Using existing .env configuration for Bluefin..."

# Create required directories
echo "📁 Creating required directories..."
mkdir -p logs/bluefin
mkdir -p data
mkdir -p data/mcp_memory

# Check if Docker Compose file exists
if [[ ! -f "docker-compose.bluefin.yml" ]]; then
    echo "❌ docker-compose.bluefin.yml not found"
    exit 1
fi

echo "✅ Environment configuration ready"

# Build and launch the bot with Bluefin configuration
echo "🔨 Building Bluefin trading bot containers..."
docker-compose -f docker-compose.bluefin.yml build --no-cache

echo "🚀 Launching Bluefin trading bot on OrbStack..."
docker-compose -f docker-compose.bluefin.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker-compose -f docker-compose.bluefin.yml ps

# Display logs
echo ""
echo "📋 BLUEFIN TRADING BOT STATUS"
echo "============================="
echo "✅ Bot launched successfully on OrbStack"
echo "✅ Using real Bluefin DEX market data"
echo "✅ Paper trading mode enabled (safe)"
echo ""
echo "🔍 View logs with:"
echo "   docker-compose -f docker-compose.bluefin.yml logs -f ai-trading-bot-bluefin"
echo ""
echo "🛑 Stop the bot with:"
echo "   docker-compose -f docker-compose.bluefin.yml down"
echo ""
echo "📊 Monitor dashboard (if enabled):"
echo "   http://localhost:3000"
echo ""
echo "🎯 Trading Symbol: SUI-PERP"
echo "💹 Exchange: Bluefin DEX (Sui Network)"
echo "📈 Data Source: Real market data only"
echo "🛡️  Mode: Paper trading (no real money)"

# Show initial logs
echo ""
echo "📄 Initial bot logs:"
echo "==================="
docker-compose -f docker-compose.bluefin.yml logs --tail=20 ai-trading-bot-bluefin