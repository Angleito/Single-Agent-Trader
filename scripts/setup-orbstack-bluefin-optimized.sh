#!/bin/bash
# OrbStack-Optimized Bluefin Trading Bot Setup Script
# Resolves UV dependency conflicts and configures live trading

set -e

echo "🚀 OrbStack-Optimized Bluefin DEX Trading Bot Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠️  [$(date '+%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ [$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}❌ [$(date '+%H:%M:%S')]${NC} $1"
}

# Function to validate private key format
validate_private_key() {
    local key="$1"
    if [[ ${#key} -eq 64 && "$key" =~ ^[0-9a-fA-F]+$ ]]; then
        return 0
    elif [[ "$key" =~ ^suiprivkey1 ]]; then
        return 0
    elif [[ ${#key} -gt 20 && "$key" =~ ^[a-z]+[[:space:]][a-z]+.*$ ]]; then
        # Mnemonic phrase (12+ words)
        local word_count=$(echo "$key" | wc -w)
        if [[ $word_count -ge 12 ]]; then
            return 0
        fi
    fi
    return 1
}

# Check prerequisites
log "Checking prerequisites..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    error ".env file not found"
    echo "Please create a .env file with your configuration first"
    exit 1
fi

# Check Docker/OrbStack
if ! command -v docker &> /dev/null; then
    error "Docker is not installed"
    echo "Please install Docker or OrbStack: https://orbstack.dev/"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    error "Docker is not running"
    echo "Please start Docker/OrbStack and try again"
    exit 1
fi

# Check if using OrbStack
if docker info 2>/dev/null | grep -q "OrbStack"; then
    success "OrbStack detected - optimal performance enabled"
    export ORBSTACK_DETECTED=true
else
    warn "OrbStack not detected - using standard Docker"
    export ORBSTACK_DETECTED=false
fi

# Read and validate environment configuration
log "Validating environment configuration..."

# Source .env file
set -a
source .env
set +a

# Validate required variables
if [ -z "$LLM__OPENAI_API_KEY" ] || [ "$LLM__OPENAI_API_KEY" = "your_openai_key_here" ]; then
    error "OpenAI API key not configured in .env file"
    exit 1
fi

if [ -z "$EXCHANGE__BLUEFIN_PRIVATE_KEY" ] || [ "$EXCHANGE__BLUEFIN_PRIVATE_KEY" = "your_private_key_here" ]; then
    error "Bluefin private key not configured in .env file"
    exit 1
fi

# Validate private key format
if ! validate_private_key "${EXCHANGE__BLUEFIN_PRIVATE_KEY}"; then
    error "Invalid private key format in .env file"
    echo "Expected: 64 hex characters, suiprivkey1... format, or 12+ word mnemonic phrase"
    exit 1
fi

success "Environment configuration validated"

# Display current configuration
echo ""
echo -e "${CYAN}📋 Current Configuration:${NC}"
echo "  • Exchange: ${EXCHANGE__EXCHANGE_TYPE:-bluefin}"
echo "  • Trading Symbol: ${TRADING__SYMBOL:-SUI-PERP}"
echo "  • Network: ${EXCHANGE__BLUEFIN_NETWORK:-mainnet}"
echo "  • Dry Run Mode: ${SYSTEM__DRY_RUN:-true}"
echo "  • Private Key: ${EXCHANGE__BLUEFIN_PRIVATE_KEY:0:20}..."
echo "  • OpenAI Key: ${LLM__OPENAI_API_KEY:0:20}..."

# Trading mode confirmation
echo ""
if [ "${SYSTEM__DRY_RUN:-true}" = "false" ]; then
    echo -e "${RED}🚨 LIVE TRADING MODE DETECTED${NC}"
    echo -e "${RED}   This will trade with REAL MONEY on Bluefin DEX${NC}"
    echo -e "${RED}   Symbol: ${TRADING__SYMBOL:-SUI-PERP}${NC}"
    echo -e "${RED}   Network: ${EXCHANGE__BLUEFIN_NETWORK:-mainnet}${NC}"
    echo -e "${RED}   Position Size: ${MAX_POSITION_SIZE:-0.01} (1%)${NC}"
    echo ""
    read -p "Are you absolutely sure you want to proceed with live trading? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        error "Setup cancelled by user"
        exit 1
    fi
else
    success "Paper Trading Mode - Safe simulation enabled"
fi

# Build OrbStack-optimized container
log "Building OrbStack-optimized trading bot container..."

# Create required directories
mkdir -p logs data config

# Stop any existing containers
docker-compose -f docker-compose.orbstack-optimized.yml down 2>/dev/null || true

# Build with optimizations
if [ "$ORBSTACK_DETECTED" = "true" ]; then
    log "Building with OrbStack optimizations..."
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    docker-compose -f docker-compose.orbstack-optimized.yml build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --parallel
else
    log "Building with standard Docker..."
    docker-compose -f docker-compose.orbstack-optimized.yml build
fi

success "Container built successfully"

# Network connectivity test
log "Testing network connectivity..."
if curl -s --connect-timeout 5 https://api.sui.io/health > /dev/null; then
    success "Sui network connectivity verified"
else
    warn "Sui network connectivity test failed (may affect trading)"
fi

# Final trading mode confirmation for live trading
if [ "${SYSTEM__DRY_RUN:-true}" = "false" ]; then
    echo ""
    echo -e "${RED}🚨 FINAL LIVE TRADING CONFIRMATION${NC}"
    echo -e "${RED}   About to start LIVE TRADING with REAL MONEY${NC}"
    echo -e "${RED}   Exchange: Bluefin DEX${NC}"
    echo -e "${RED}   Symbol: ${TRADING__SYMBOL:-SUI-PERP}${NC}"
    echo -e "${RED}   Network: ${EXCHANGE__BLUEFIN_NETWORK:-mainnet}${NC}"
    echo -e "${RED}   Position Size: ${MAX_POSITION_SIZE:-0.01} (conservative)${NC}"
    echo -e "${RED}   Risk per Trade: ${RISK_PER_TRADE:-0.01} (1%)${NC}"
    echo ""
    read -p "Type 'START LIVE TRADING' to confirm: " final_confirm
    if [ "$final_confirm" != "START LIVE TRADING" ]; then
        error "Live trading cancelled"
        exit 1
    fi
fi

# Start the optimized trading bot
log "Starting OrbStack-optimized Bluefin trading bot..."

if [ "$ORBSTACK_DETECTED" = "true" ]; then
    log "Launching with OrbStack optimizations (host networking, ARM64)"
else
    log "Launching with standard Docker configuration"
fi

# Start the container
docker-compose -f docker-compose.orbstack-optimized.yml up -d

# Wait for startup
sleep 10

# Check container status
if docker-compose -f docker-compose.orbstack-optimized.yml ps | grep -q "Up"; then
    success "Trading bot started successfully!"
    
    echo ""
    echo -e "${GREEN}🎉 Setup Complete!${NC}"
    echo ""
    
    if [ "${SYSTEM__DRY_RUN:-true}" = "false" ]; then
        echo -e "${RED}🚨 LIVE TRADING ACTIVE${NC}"
        echo -e "${RED}   Trading SUI-PERP with real money on Bluefin DEX${NC}"
    else
        echo -e "${GREEN}📊 Paper trading simulation active${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}📊 Monitoring Commands:${NC}"
    echo "  View logs:    docker-compose -f docker-compose.orbstack-optimized.yml logs -f"
    echo "  Check status: docker-compose -f docker-compose.orbstack-optimized.yml ps"
    echo "  Stop bot:     docker-compose -f docker-compose.orbstack-optimized.yml down"
    echo ""
    
    # Show initial logs
    echo -e "${CYAN}📝 Initial logs:${NC}"
    docker-compose -f docker-compose.orbstack-optimized.yml logs --tail=20
    
else
    error "Failed to start trading bot"
    echo "Container logs:"
    docker-compose -f docker-compose.orbstack-optimized.yml logs
    exit 1
fi