# Comprehensive Bluefin Setup Guide

Complete guide to setting up and configuring the AI Trading Bot with Bluefin DEX integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Sui Wallet Setup](#sui-wallet-setup)
3. [Environment Configuration](#environment-configuration)
4. [Trading Configuration](#trading-configuration)
5. [Advanced Configuration](#advanced-configuration)
6. [Validation and Testing](#validation-and-testing)
7. [Performance Optimization](#performance-optimization)

## Prerequisites

### System Requirements
- **Docker**: Version 20.10+ with Docker Compose
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 2GB RAM, 4GB recommended
- **Storage**: 5GB free space for containers and logs
- **Network**: Stable internet connection

### Required Accounts
- **Sui Wallet**: For blockchain interaction
- **OpenAI Account**: For AI decision making
- **Bluefin Account**: Optional, for web interface monitoring

## Sui Wallet Setup

### 1. Install Wallet Extension

**Chrome/Brave Users**:
```bash
# Visit Chrome Web Store
https://chrome.google.com/webstore/detail/sui-wallet/opcgpfmipidbgpenhmajoajpbobppdil
```

**Alternative Wallets**:
- [Suiet Wallet](https://suiet.app/) - Multi-chain support
- [Ethos Wallet](https://ethoswallet.xyz/) - Advanced features

### 2. Create New Wallet

1. **Install Extension** and click "Create New Wallet"
2. **Set Strong Password** (this protects your local wallet)
3. **Save Recovery Phrase** - Write down all 12/24 words in order
   ```
   ⚠️ CRITICAL: Store your recovery phrase offline and secure
   - Never share it with anyone
   - Don't store it digitally unencrypted
   - Consider a hardware backup
   ```
4. **Verify Recovery Phrase** by entering requested words

### 3. Export Private Key

1. **Open Wallet Settings**
2. **Navigate to Security** → Export Private Key
3. **Enter Password** and copy the hex string
4. **Format Check**: Should start with `0x` and be 64 characters long
   ```
   Example: 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
   ```

### 4. Fund Your Wallet

**Get SUI Tokens** (for gas fees):
```bash
# You need ~1-2 SUI for gas fees
# Methods to get SUI:
# 1. CEX withdrawal (Binance, OKX, etc.)
# 2. Bridge from other chains
# 3. DEX swap (if you have other Sui tokens)
```

**Get USDC** (for trading):
```bash
# Bridge USDC to Sui using:
# 1. Wormhole: https://wormhole.com/
# 2. Portal Bridge: https://portalbridge.com/
# 3. LayerZero Bridge: https://layerzero.network/
```

**Verify Balances**:
```bash
# Check your wallet shows:
# - SUI: 1+ tokens (for gas)
# - USDC: Your trading capital
```

## Environment Configuration

### 1. Copy Environment Template

```bash
# Copy the Bluefin-specific template
cp .env.bluefin.example .env

# Or start with generic template
cp .env.example .env
```

### 2. Essential Configuration

Edit your `.env` file with required settings:

```bash
# ===================
# EXCHANGE CONFIGURATION
# ===================
EXCHANGE__EXCHANGE_TYPE=bluefin

# Your Sui wallet private key (from wallet export)
EXCHANGE__BLUEFIN_PRIVATE_KEY=0x1234567890abcdef...

# Network selection
EXCHANGE__BLUEFIN_NETWORK=mainnet  # or testnet for testing

# ===================
# TRADING CONFIGURATION
# ===================
# Safety first - keep true for paper trading
SYSTEM__DRY_RUN=true

# Trading pair (Bluefin uses PERP suffix)
TRADING__SYMBOL=SUI-PERP  # Options: BTC-PERP, ETH-PERP, SOL-PERP, etc.

# Leverage settings (start conservative)
TRADING__LEVERAGE=5
TRADING__MAX_FUTURES_LEVERAGE=10

# Candle interval
TRADING__INTERVAL=1m  # Supported: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w, 1M

# ===================
# LLM CONFIGURATION
# ===================
LLM__OPENAI_API_KEY=sk-...your-openai-api-key...
LLM__MODEL_NAME=gpt-4o  # or gpt-4o-mini for lower cost
LLM__PROVIDER=openai

# ===================
# BLUEFIN SERVICE
# ===================
# Internal service communication (usually defaults work)
BLUEFIN_SERVICE_URL=http://bluefin-service:8080
BLUEFIN_SERVICE_API_KEY=trading-bot-secret

# ===================
# PAPER TRADING
# ===================
PAPER_TRADING__STARTING_BALANCE=10000
PAPER_TRADING__FEE_RATE=0.0015  # 0.15% (realistic Bluefin fees)

# ===================
# LOGGING
# ===================
LOG_LEVEL=INFO
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
SYSTEM__ENABLE_PERFORMANCE_MONITORING=true
```

### 3. Optional Advanced Settings

```bash
# ===================
# MCP MEMORY (AI Learning)
# ===================
MCP_ENABLED=false  # Set to true to enable AI learning
MCP_SERVER_URL=http://mcp-memory:8765
MCP_MEMORY_RETENTION_DAYS=90

# ===================
# PERFORMANCE TUNING
# ===================
# Data fetching optimization
DATA__CANDLE_LIMIT=500
DATA__DATA_CACHE_TTL_SECONDS=30

# WebSocket optimization
EXCHANGE__WEBSOCKET_TIMEOUT=60
EXCHANGE__RATE_LIMIT_REQUESTS=30

# ===================
# SECURITY
# ===================
# Rate limiting for API calls
BLUEFIN_SERVICE_RATE_LIMIT=100

# Enable completion logging for debugging
LLM__ENABLE_COMPLETION_LOGGING=true
LLM__LOG_MARKET_CONTEXT=true
```

## Trading Configuration

### Symbol Selection

**Available Perpetual Futures**:
```bash
# Major cryptocurrencies
TRADING__SYMBOL=BTC-PERP   # Bitcoin
TRADING__SYMBOL=ETH-PERP   # Ethereum
TRADING__SYMBOL=SOL-PERP   # Solana
TRADING__SYMBOL=SUI-PERP   # Sui (native chain)

# Check current available markets:
# Visit https://trade.bluefin.io/trade/SUI-PERP to see all pairs
```

**Symbol Validation**:
```bash
# The bot will automatically validate symbols
# Invalid symbols will show error messages
# Testnet has limited symbol support
```

### Leverage Configuration

**Conservative Settings** (Recommended for beginners):
```bash
TRADING__LEVERAGE=2
TRADING__MAX_FUTURES_LEVERAGE=5
```

**Moderate Settings**:
```bash
TRADING__LEVERAGE=5
TRADING__MAX_FUTURES_LEVERAGE=10
```

**Aggressive Settings** (High risk):
```bash
TRADING__LEVERAGE=10
TRADING__MAX_FUTURES_LEVERAGE=20
```

**Risk Management**:
```bash
# Position size as percentage of account
TRADING__POSITION_SIZE_PCT=25  # 25% of account per trade

# Risk per trade
TRADING__RISK_PER_TRADE=0.02  # 2% of account risk per trade
```

### Interval Configuration

**Supported Intervals**:
```bash
# Valid intervals (Bluefin limitation)
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w, 1M

# Sub-minute intervals NOT supported
# 15s, 30s will be converted to 1m with warnings
```

**Interval Selection Guide**:
```bash
TRADING__INTERVAL=1m   # High frequency, more signals
TRADING__INTERVAL=5m   # Balanced approach
TRADING__INTERVAL=15m  # Longer timeframe, fewer but stronger signals
TRADING__INTERVAL=1h   # Long-term position trading
```

## Advanced Configuration

### Custom RPC Endpoints

If you experience slow performance, configure custom RPC:

```bash
# Add to .env
EXCHANGE__BLUEFIN_RPC_URL=https://sui-mainnet.nodeinfra.com

# Alternative RPC endpoints:
# https://fullnode.mainnet.sui.io
# https://sui-rpc.publicnode.com
# https://mainnet.suiet.app
```

### Contract Address Overrides

For custom or updated Bluefin contracts:

```bash
# Override in bot/exchange/bluefin.py if needed
CONTRACT_ADDRESSES = {
    "mainnet": {
        "exchange": "0x...",
        "clearing_house": "0x...",
    }
}
```

### Multi-Symbol Trading

Configure multiple symbols (future feature):

```bash
# Primary symbol
TRADING__SYMBOL=SUI-PERP

# Additional symbols for portfolio
TRADING__ADDITIONAL_SYMBOLS=BTC-PERP,ETH-PERP
```

## Validation and Testing

### 1. Environment Validation

```bash
# Validate configuration
python -c "from bot.config import settings; print('✓ Config loaded successfully')"

# Check Bluefin connection
python scripts/verify_exchange_config.py
```

### 2. Paper Trading Test

```bash
# Start paper trading
docker-compose -f docker-compose.bluefin.yml up ai-trading-bot-bluefin

# Look for success messages:
# ✓ Bluefin Service    Connected    mainnet
# ✓ Exchange (Bluefin) Connected    mainnet network (Sui)
# 🎯 PAPER TRADING MODE - No real trades will be executed
```

### 3. Service Health Checks

```bash
# Check Bluefin service health
curl http://localhost:8081/health

# Expected response:
# {"status": "healthy", "network": "mainnet", "service": "bluefin-sdk"}
```

### 4. Balance Validation

```bash
# The bot will show your wallet balance at startup:
# Balance: 1.234 SUI, 1000.00 USDC
# Verify these match your wallet
```

## Performance Optimization

### Container Resource Limits

Adjust Docker resource limits based on your system:

```yaml
# In docker-compose.bluefin.yml
deploy:
  resources:
    limits:
      memory: 1G      # Increase if needed
      cpus: '0.5'     # Adjust based on CPU cores
    reservations:
      memory: 512M
      cpus: '0.25'
```

### Data Caching

Optimize data fetching:

```bash
# Add to .env
DATA__CANDLE_LIMIT=500          # More data for better analysis
DATA__DATA_CACHE_TTL_SECONDS=30 # Cache data for 30 seconds
```

### WebSocket Optimization

For better real-time performance:

```bash
# Add to .env
EXCHANGE__WEBSOCKET_TIMEOUT=60
EXCHANGE__RATE_LIMIT_REQUESTS=30
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
```

### Logging Optimization

Balance logging detail with performance:

```bash
# Production settings
LOG_LEVEL=INFO                          # Less verbose than DEBUG
LLM__ENABLE_COMPLETION_LOGGING=false    # Disable in production
SYSTEM__ENABLE_PERFORMANCE_MONITORING=true
```

## Next Steps

1. **Test Setup**: Run paper trading to validate configuration
2. **Monitor Performance**: Use dashboard or logs to track bot behavior
3. **Security Review**: Check [Security Guide](BLUEFIN_SECURITY_GUIDE.md)
4. **Troubleshooting**: Reference [Troubleshooting Guide](BLUEFIN_TROUBLESHOOTING.md)
5. **Go Live**: Follow safety procedures for live trading

## Configuration Files Reference

- **Main Config**: `.env` - Your primary configuration
- **Docker Config**: `docker-compose.bluefin.yml` - Container orchestration
- **Trading Config**: `config/bluefin_trading.json` - Advanced trading settings
- **Testnet Config**: `config/bluefin_testnet.json` - Testnet-specific settings

## Support Resources

- **Bluefin Documentation**: [bluefin-exchange.readme.io](https://bluefin-exchange.readme.io)
- **Sui Documentation**: [docs.sui.io](https://docs.sui.io)
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com)
- **OpenAI API Documentation**: [platform.openai.com/docs](https://platform.openai.com/docs)
