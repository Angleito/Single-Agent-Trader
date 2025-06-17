#!/bin/bash
# Secure Bluefin private key setup for Docker environments
# Best practices for Sui wallet security in containers

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bluefin DEX Secure Setup ===${NC}"
echo

# Check if running on macOS with OrbStack
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}✓ macOS detected${NC}"
    if command -v orbctl &> /dev/null; then
        echo -e "${GREEN}✓ OrbStack detected${NC}"
        USING_ORBSTACK=true
    else
        echo -e "${YELLOW}⚠ OrbStack not detected - using Docker Desktop${NC}"
        USING_ORBSTACK=false
    fi
else
    echo -e "${YELLOW}⚠ Non-macOS system detected${NC}"
    USING_ORBSTACK=false
fi

# Function to generate secure .env file
generate_secure_env() {
    local env_file=".env.bluefin"
    
    echo -e "${BLUE}Creating secure environment file: ${env_file}${NC}"
    
    cat > "$env_file" << 'EOF'
# Bluefin DEX Trading Bot Configuration
# IMPORTANT: This file contains sensitive information - never commit to version control

# ======================
# SYSTEM CONFIGURATION
# ======================
SYSTEM__DRY_RUN=true
SYSTEM__ENVIRONMENT=production
LOG_LEVEL=INFO

# ======================
# EXCHANGE CONFIGURATION
# ======================
EXCHANGE__EXCHANGE_TYPE=bluefin
EXCHANGE__BLUEFIN_NETWORK=mainnet

# ======================
# SUI WALLET CONFIGURATION (SECURITY CRITICAL)
# ======================
# IMPORTANT: Set your actual Sui private key here
# Format: 64-character hexadecimal string (without 0x prefix)
# Example: a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
EXCHANGE__BLUEFIN_PRIVATE_KEY=your_sui_private_key_here

# ======================
# TRADING CONFIGURATION
# ======================
TRADING__SYMBOL=ETH-USD
TRADING__LEVERAGE=5
TRADING__MAX_POSITION_SIZE=1000

# ======================
# LLM CONFIGURATION
# ======================
LLM__OPENAI_API_KEY=your_openai_api_key_here
LLM__MODEL=gpt-4-turbo-preview
LLM__TEMPERATURE=0.1

# ======================
# RISK MANAGEMENT
# ======================
RISK__MAX_DAILY_LOSS_PERCENT=5.0
RISK__MAX_POSITION_SIZE_PERCENT=10.0
RISK__STOP_LOSS_PERCENT=2.0
RISK__TAKE_PROFIT_PERCENT=4.0

# ======================
# DOCKER OPTIMIZATIONS
# ======================
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
DOCKER_BUILDKIT=1

# ======================
# ORBSTACK OPTIMIZATIONS (if using OrbStack)
# ======================
BLUEFIN_NETWORK_TIMEOUT=30
BLUEFIN_RETRY_ATTEMPTS=5
SOCKETIO_ASYNC_MODE=aiohttp

# ======================
# DEVELOPMENT OPTIONS
# ======================
DEBUG=false
TESTING=false
MCP_ENABLED=false
EOF

    chmod 600 "$env_file"
    echo -e "${GREEN}✓ Created $env_file with secure permissions (600)${NC}"
    echo -e "${YELLOW}⚠ IMPORTANT: Edit $env_file and set your actual private keys!${NC}"
}

# Function to set up Docker secrets (if using Docker Swarm)
setup_docker_secrets() {
    echo -e "${BLUE}Setting up Docker secrets for Sui private key...${NC}"
    
    if ! docker info | grep -q "Swarm: active"; then
        echo -e "${YELLOW}⚠ Docker Swarm not active - skipping Docker secrets setup${NC}"
        echo -e "${YELLOW}  For production, consider using Docker Swarm for secret management${NC}"
        return
    fi
    
    echo -e "${GREEN}✓ Docker Swarm detected${NC}"
    
    # Create secret for Sui private key
    read -s -p "Enter your Sui private key (will not be displayed): " SUI_PRIVATE_KEY
    echo
    
    if [[ ${#SUI_PRIVATE_KEY} -ne 64 ]]; then
        echo -e "${RED}✗ Invalid private key length. Sui private keys should be 64 hex characters${NC}"
        return 1
    fi
    
    echo "$SUI_PRIVATE_KEY" | docker secret create bluefin_private_key -
    echo -e "${GREEN}✓ Created Docker secret: bluefin_private_key${NC}"
    
    # Create secret for OpenAI API key
    read -s -p "Enter your OpenAI API key (will not be displayed): " OPENAI_API_KEY
    echo
    
    echo "$OPENAI_API_KEY" | docker secret create openai_api_key -
    echo -e "${GREEN}✓ Created Docker secret: openai_api_key${NC}"
}

# Function to validate private key format
validate_private_key() {
    local key="$1"
    
    # Check if it's 64 hex characters
    if [[ ! "$key" =~ ^[0-9a-fA-F]{64}$ ]]; then
        echo -e "${RED}✗ Invalid private key format${NC}"
        echo -e "${YELLOW}  Expected: 64 hexadecimal characters (0-9, a-f, A-F)${NC}"
        echo -e "${YELLOW}  Received: ${#key} characters${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Private key format is valid${NC}"
    return 0
}

# Function to create OrbStack-optimized compose file
create_orbstack_compose() {
    if [[ "$USING_ORBSTACK" == "true" ]]; then
        echo -e "${BLUE}Creating OrbStack-optimized Docker Compose configuration...${NC}"
        
        # Copy the OrbStack compose file as the default for Bluefin
        cp docker-compose.orbstack-bluefin.yml docker-compose.bluefin.yml
        echo -e "${GREEN}✓ Created docker-compose.bluefin.yml optimized for OrbStack${NC}"
    else
        echo -e "${BLUE}Creating standard Docker Compose configuration...${NC}"
        
        # Use the existing Bluefin compose file
        cp docker-compose.bluefin.yml docker-compose.bluefin-standard.yml 2>/dev/null || true
        echo -e "${GREEN}✓ Using standard Docker Compose configuration${NC}"
    fi
}

# Function to create secure startup script
create_startup_script() {
    local script_file="start-bluefin-secure.sh"
    
    cat > "$script_file" << 'EOF'
#!/bin/bash
# Secure startup script for Bluefin trading bot

set -euo pipefail

# Check if environment file exists
if [[ ! -f .env.bluefin ]]; then
    echo "ERROR: .env.bluefin file not found!"
    echo "Run ./scripts/setup-bluefin-secure.sh first"
    exit 1
fi

# Source environment
set -a
source .env.bluefin
set +a

# Validate critical environment variables
if [[ "$EXCHANGE__BLUEFIN_PRIVATE_KEY" == "your_sui_private_key_here" ]]; then
    echo "ERROR: Sui private key not set in .env.bluefin"
    echo "Edit .env.bluefin and set EXCHANGE__BLUEFIN_PRIVATE_KEY"
    exit 1
fi

if [[ "$LLM__OPENAI_API_KEY" == "your_openai_api_key_here" ]]; then
    echo "ERROR: OpenAI API key not set in .env.bluefin"
    echo "Edit .env.bluefin and set LLM__OPENAI_API_KEY"
    exit 1
fi

# Check trading mode
if [[ "$SYSTEM__DRY_RUN" == "false" ]]; then
    echo "⚠️  WARNING: LIVE TRADING MODE ENABLED"
    echo "   This will trade with real money on Bluefin DEX"
    echo "   Private key: ${EXCHANGE__BLUEFIN_PRIVATE_KEY:0:8}...${EXCHANGE__BLUEFIN_PRIVATE_KEY: -8}"
    echo
    read -p "Are you sure you want to continue? (type 'YES' to confirm): " confirm
    if [[ "$confirm" != "YES" ]]; then
        echo "Startup cancelled"
        exit 0
    fi
else
    echo "✓ Paper trading mode enabled (safe)"
fi

# Start the bot
echo "Starting Bluefin trading bot..."
docker-compose -f docker-compose.bluefin.yml up --build
EOF

    chmod +x "$script_file"
    echo -e "${GREEN}✓ Created secure startup script: $script_file${NC}"
}

# Main setup flow
main() {
    echo -e "${BLUE}1. Generating secure environment configuration...${NC}"
    generate_secure_env
    echo
    
    echo -e "${BLUE}2. Setting up Docker configuration...${NC}"
    create_orbstack_compose
    echo
    
    echo -e "${BLUE}3. Creating secure startup script...${NC}"
    create_startup_script
    echo
    
    echo -e "${BLUE}4. Docker secrets setup (optional)...${NC}"
    read -p "Set up Docker secrets for production? (y/N): " setup_secrets
    if [[ "$setup_secrets" =~ ^[Yy]$ ]]; then
        setup_docker_secrets
    else
        echo -e "${YELLOW}⚠ Skipping Docker secrets setup${NC}"
    fi
    echo
    
    echo -e "${GREEN}=== Setup Complete ===${NC}"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. Edit ${YELLOW}.env.bluefin${NC} and set your actual private keys"
    echo -e "2. Run ${YELLOW}./start-bluefin-secure.sh${NC} to start the bot"
    echo
    echo -e "${RED}SECURITY REMINDERS:${NC}"
    echo -e "• Never commit .env.bluefin to version control"
    echo -e "• Keep your private key secure and never share it"
    echo -e "• Start with SYSTEM__DRY_RUN=true for testing"
    echo -e "• Use strong API keys with minimal required permissions"
    echo
    
    # Add .env.bluefin to .gitignore if it exists
    if [[ -f .gitignore ]]; then
        if ! grep -q ".env.bluefin" .gitignore; then
            echo ".env.bluefin" >> .gitignore
            echo -e "${GREEN}✓ Added .env.bluefin to .gitignore${NC}"
        fi
    fi
}

# Run main function
main "$@"