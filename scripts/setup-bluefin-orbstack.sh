#!/bin/bash
# Secure Bluefin DEX Setup Script for OrbStack
# Configures environment for live SUI-PERP trading

set -e

echo "🚀 Bluefin DEX Trading Bot - OrbStack Setup"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo "Please create a .env file with your configuration"
    exit 1
fi

# Function to validate private key format
validate_private_key() {
    local key="$1"
    if [[ ${#key} -eq 64 && "$key" =~ ^[0-9a-fA-F]+$ ]]; then
        return 0
    elif [[ "$key" =~ ^suiprivkey1 ]]; then
        return 0
    else
        return 1
    fi
}

# Check for required environment variables
echo -e "${BLUE}🔍 Checking environment configuration...${NC}"

# Read current values from .env
CURRENT_DRY_RUN=$(grep "^SYSTEM__DRY_RUN=" .env | cut -d'=' -f2)
CURRENT_PRIVATE_KEY=$(grep "^EXCHANGE__BLUEFIN_PRIVATE_KEY=" .env | cut -d'=' -f2)
CURRENT_OPENAI_KEY=$(grep "^LLM__OPENAI_API_KEY=" .env | cut -d'=' -f2)

echo "Current configuration:"
echo "  • Dry Run Mode: $CURRENT_DRY_RUN"
echo "  • Private Key: ${CURRENT_PRIVATE_KEY:0:20}..."
echo "  • OpenAI Key: ${CURRENT_OPENAI_KEY:0:20}..."

# Validate private key
if [ -z "$CURRENT_PRIVATE_KEY" ] || [ "$CURRENT_PRIVATE_KEY" = "your_private_key_here" ]; then
    echo -e "${RED}❌ Error: Bluefin private key not configured${NC}"
    echo "Please set EXCHANGE__BLUEFIN_PRIVATE_KEY in your .env file"
    exit 1
fi

if ! validate_private_key "${CURRENT_PRIVATE_KEY}"; then
    echo -e "${YELLOW}⚠️  Warning: Private key format may be invalid${NC}"
    echo "Expected: 64 hex characters or suiprivkey1... format"
fi

# Check OpenAI key
if [ -z "$CURRENT_OPENAI_KEY" ] || [ "$CURRENT_OPENAI_KEY" = "your_openai_key_here" ]; then
    echo -e "${RED}❌ Error: OpenAI API key not configured${NC}"
    echo "Please set LLM__OPENAI_API_KEY in your .env file"
    exit 1
fi

# Trading mode confirmation
echo ""
echo -e "${YELLOW}🚨 TRADING MODE CONFIRMATION${NC}"
if [ "$CURRENT_DRY_RUN" = "false" ]; then
    echo -e "${RED}⚠️  LIVE TRADING MODE DETECTED${NC}"
    echo -e "${RED}   This will trade with REAL MONEY on Bluefin DEX${NC}"
    echo -e "${RED}   Symbol: SUI-PERP${NC}"
    echo -e "${RED}   Network: Mainnet${NC}"
    echo ""
    read -p "Are you sure you want to proceed with live trading? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "❌ Setup cancelled"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Paper Trading Mode - Safe simulation${NC}"
fi

# OrbStack specific checks
echo ""
echo -e "${BLUE}🐳 Checking OrbStack environment...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker/OrbStack and try again"
    exit 1
fi

# Check if using OrbStack
if docker info 2>/dev/null | grep -q "OrbStack"; then
    echo -e "${GREEN}✅ OrbStack detected${NC}"
else
    echo -e "${YELLOW}⚠️  OrbStack not detected - using standard Docker${NC}"
fi

# Network connectivity test
echo -e "${BLUE}🌐 Testing network connectivity...${NC}"
if curl -s --connect-timeout 5 https://api.sui.io/health > /dev/null; then
    echo -e "${GREEN}✅ Sui network reachable${NC}"
else
    echo -e "${YELLOW}⚠️  Sui network connectivity test failed${NC}"
fi

# Build and run
echo ""
echo -e "${BLUE}🚀 Building and starting Bluefin trading bot...${NC}"

# Stop any existing containers
docker-compose -f docker-compose.orbstack-bluefin.yml down 2>/dev/null || true

# Build with OrbStack optimizations
echo "Building OrbStack-optimized container..."
docker-compose -f docker-compose.orbstack-bluefin.yml build

# Final confirmation for live trading
if [ "$CURRENT_DRY_RUN" = "false" ]; then
    echo ""
    echo -e "${RED}🚨 FINAL CONFIRMATION${NC}"
    echo -e "${RED}   About to start LIVE TRADING with REAL MONEY${NC}"
    echo -e "${RED}   Exchange: Bluefin DEX${NC}"
    echo -e "${RED}   Symbol: SUI-PERP${NC}"
    echo -e "${RED}   Position Size: 1% (conservative)${NC}"
    echo ""
    read -p "Type 'START LIVE TRADING' to confirm: " final_confirm
    if [ "$final_confirm" != "START LIVE TRADING" ]; then
        echo "❌ Live trading cancelled"
        exit 1
    fi
fi

# Start the container
echo "Starting Bluefin trading bot..."
docker-compose -f docker-compose.orbstack-bluefin.yml up

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"

if [ "$CURRENT_DRY_RUN" = "false" ]; then
    echo -e "${RED}🚨 LIVE TRADING ACTIVE${NC}"
else
    echo -e "${GREEN}📊 Paper trading simulation active${NC}"
fi

echo ""
echo "Monitor logs with:"
echo "  docker-compose -f docker-compose.orbstack-bluefin.yml logs -f"