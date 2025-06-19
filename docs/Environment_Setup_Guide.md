# Environment Setup Guide for Bluefin DEX Trading

This guide provides comprehensive instructions for setting up your environment configuration for the AI Trading Bot with Bluefin DEX support.

## Quick Start Checklist

### 1. Prerequisites
- [ ] **Sui Wallet Setup**: Install Sui Wallet browser extension and create a wallet
- [ ] **OpenAI Account**: Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- [ ] **Funding**: Have USDC on Sui for trading capital and SUI for gas fees
- [ ] **Docker**: Ensure Docker and Docker Compose are installed

### 2. Configuration Files
- [ ] Copy environment template: `cp .env.example .env` (or `cp .env.bluefin.example .env` for Bluefin-specific setup)
- [ ] Generate Bluefin service API key: `python services/generate_api_key.py`
- [ ] Configure all required variables (see sections below)
- [ ] Set file permissions: `chmod 600 .env`

### 3. Validation and Testing
- [ ] Validate configuration: `python services/scripts/validate_env.py`
- [ ] Start in paper trading mode first: `SYSTEM__DRY_RUN=true`
- [ ] Test the system: `python -m bot.main live`
- [ ] Monitor dashboard: Open `http://localhost:8000`

## Environment Variables Reference

### Core Required Variables

#### Exchange Configuration
```bash
# REQUIRED: Choose your exchange
EXCHANGE__EXCHANGE_TYPE=bluefin  # or "coinbase"

# REQUIRED for Bluefin: Your Sui wallet private key
EXCHANGE__BLUEFIN_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE

# REQUIRED for Bluefin: Network selection
EXCHANGE__BLUEFIN_NETWORK=mainnet  # or "testnet"

# REQUIRED for Bluefin: Service configuration
BLUEFIN_SERVICE_URL=http://bluefin-service:8080
BLUEFIN_SERVICE_API_KEY=your-generated-api-key
```

#### Trading Configuration
```bash
# REQUIRED: Trading pair
TRADING__SYMBOL=SUI-PERP  # Bluefin uses PERP suffix

# REQUIRED: Leverage setting
TRADING__LEVERAGE=5  # Conservative: 2-3x, Moderate: 5-10x

# REQUIRED: Enable futures (Bluefin only supports futures)
TRADING__ENABLE_FUTURES=true

# REQUIRED: Safety setting
SYSTEM__DRY_RUN=true  # ALWAYS start with true for testing
```

#### AI Configuration
```bash
# REQUIRED: OpenAI API key for AI trading decisions
LLM__OPENAI_API_KEY=sk-YOUR_OPENAI_KEY_HERE

# OPTIONAL: Model selection
LLM__MODEL_NAME=o3  # Recommended for best performance
LLM__TEMPERATURE=0.1  # Low for consistent decisions
```

#### Risk Management
```bash
# REQUIRED: Loss limits
RISK__MAX_DAILY_LOSS_PCT=5.0
RISK__DEFAULT_STOP_LOSS_PCT=0.5
RISK__DEFAULT_TAKE_PROFIT_PCT=1.0
RISK__MAX_CONCURRENT_TRADES=1
```

### Optional but Recommended Variables

#### Paper Trading Configuration
```bash
PAPER_TRADING__STARTING_BALANCE=10000
PAPER_TRADING__FEE_RATE=0.0005
PAPER_TRADING__SLIPPAGE_RATE=0.0005
PAPER_TRADING__ENABLE_DAILY_REPORTS=true
```

#### Dashboard and Monitoring
```bash
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
SYSTEM__WEBSOCKET_DASHBOARD_URL=ws://localhost:8000/ws
SYSTEM__ENABLE_PERFORMANCE_MONITORING=true
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080
```

#### Advanced Features
```bash
# Memory and learning system (optional)
MCP_ENABLED=false
MCP_SERVER_URL=http://localhost:8765

# Market sentiment analysis (optional)
OMNISEARCH_ENABLED=false
OMNISEARCH_SERVER_URL=http://localhost:8766
```

## Private Key Formats

The bot supports multiple Sui private key formats:

### 1. Hex Format (Most Common)
```bash
# With 0x prefix
EXCHANGE__BLUEFIN_PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# Without 0x prefix
EXCHANGE__BLUEFIN_PRIVATE_KEY=1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

### 2. Bech32 Format (Sui Native)
```bash
EXCHANGE__BLUEFIN_PRIVATE_KEY=suiprivkey1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
```

### 3. Mnemonic Phrase (12 or 24 words)
```bash
EXCHANGE__BLUEFIN_PRIVATE_KEY=word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12
```

## Environment Validation

### Automatic Validation Script

Use the built-in validation script to check your configuration:

```bash
python services/scripts/validate_env.py
```

This script checks:
- ✅ File existence and permissions
- ✅ Required variables are set
- ✅ API key formats
- ✅ Private key formats
- ✅ Network consistency
- ✅ Trading parameters
- ⚠️ Common configuration mistakes

### Manual Validation Checklist

#### File Security
```bash
# Check file permissions (should be 600)
ls -la .env

# Fix if needed
chmod 600 .env
```

#### API Keys Validation
```bash
# OpenAI key should start with 'sk-'
echo $LLM__OPENAI_API_KEY | head -c 3  # Should output 'sk-'

# Bluefin service key should be 32+ characters
echo $BLUEFIN_SERVICE_API_KEY | wc -c  # Should be > 32
```

#### Network Consistency
- **Development**: Use `testnet` with `SYSTEM__DRY_RUN=true`
- **Production**: Use `mainnet` with careful consideration

#### Trading Safety
- **ALWAYS** start with `SYSTEM__DRY_RUN=true`
- **NEVER** set `SYSTEM__DRY_RUN=false` until thoroughly tested
- Use conservative leverage (2-5x) for beginners

## Common Configuration Mistakes

### 1. Wrong Network for Environment
```bash
# ❌ Wrong: Production settings with testnet
SYSTEM__ENVIRONMENT=production
EXCHANGE__BLUEFIN_NETWORK=testnet

# ✅ Correct: Match network to environment
SYSTEM__ENVIRONMENT=development
EXCHANGE__BLUEFIN_NETWORK=testnet
```

### 2. Unsafe Live Trading Setup
```bash
# ❌ Dangerous: Live trading without testing
SYSTEM__DRY_RUN=false
TRADING__LEVERAGE=20

# ✅ Safe: Always test first
SYSTEM__DRY_RUN=true
TRADING__LEVERAGE=5
```

### 3. Incorrect Symbol Format
```bash
# ❌ Wrong: Coinbase format for Bluefin
EXCHANGE__EXCHANGE_TYPE=bluefin
TRADING__SYMBOL=BTC-USD

# ✅ Correct: Bluefin PERP format
EXCHANGE__EXCHANGE_TYPE=bluefin
TRADING__SYMBOL=BTC-PERP
```

### 4. Missing Required Services
```bash
# ❌ Wrong: Missing Bluefin service configuration
EXCHANGE__EXCHANGE_TYPE=bluefin
# BLUEFIN_SERVICE_URL missing

# ✅ Correct: Complete Bluefin setup
EXCHANGE__EXCHANGE_TYPE=bluefin
BLUEFIN_SERVICE_URL=http://bluefin-service:8080
BLUEFIN_SERVICE_API_KEY=your-generated-key
```

## Security Best Practices

### 1. API Key Management
- **Generate unique keys** for each environment
- **Rotate keys regularly** (monthly for production)
- **Use different keys** for development/staging/production
- **Never commit keys** to version control
- **Monitor key usage** and set up alerts

### 2. Private Key Security
- **Never share** your Sui private key
- **Use hardware wallets** for large amounts
- **Consider test amounts** first on mainnet
- **Backup your keys** securely offline
- **Use separate wallets** for different purposes

### 3. File Permissions
```bash
# Secure .env file permissions
chmod 600 .env

# Check current permissions
ls -la .env
# Should show: -rw-------
```

### 4. Environment Separation
```bash
# Development
cp .env.example .env.dev
# Configure with testnet and low amounts

# Production
cp .env.example .env.prod
# Configure with mainnet and real amounts

# Load appropriate environment
source .env.dev  # or .env.prod
```

## Troubleshooting

### Configuration Issues

#### 1. Validation Errors
```bash
# Run validator for detailed error messages
python services/scripts/validate_env.py

# Common fixes:
# - Check file permissions: chmod 600 .env
# - Verify API key formats
# - Ensure all required variables are set
```

#### 2. API Key Problems
```bash
# Test OpenAI key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $LLM__OPENAI_API_KEY"

# Generate new Bluefin service key
python services/generate_api_key.py
```

#### 3. Bluefin Service Connection
```bash
# Check if Bluefin service is running
curl http://localhost:8081/health

# Start Bluefin service if needed
docker-compose -f docker-compose.bluefin.yml up bluefin-service -d
```

### Runtime Issues

#### 1. Bot Won't Start
- Check logs: `tail -f logs/bot.log`
- Validate environment: `python services/scripts/validate_env.py`
- Verify all services are running

#### 2. No Trading Activity
- Ensure `SYSTEM__DRY_RUN=true` for testing
- Check symbol format (BTC-PERP for Bluefin)
- Verify sufficient balance and gas

#### 3. Dashboard Not Loading
- Check WebSocket settings
- Verify CORS configuration
- Ensure dashboard service is running: `http://localhost:8000`

## Environment Templates

### Minimal Development Setup
```bash
# Essential variables for testing
EXCHANGE__EXCHANGE_TYPE=bluefin
EXCHANGE__BLUEFIN_PRIVATE_KEY=0xYOUR_KEY
EXCHANGE__BLUEFIN_NETWORK=testnet
BLUEFIN_SERVICE_URL=http://bluefin-service:8080
BLUEFIN_SERVICE_API_KEY=generated-key
TRADING__SYMBOL=SUI-PERP
TRADING__LEVERAGE=2
TRADING__ENABLE_FUTURES=true
SYSTEM__DRY_RUN=true
LLM__OPENAI_API_KEY=sk-YOUR_KEY
RISK__MAX_DAILY_LOSS_PCT=2.0
```

### Production Setup
```bash
# Complete production configuration
EXCHANGE__EXCHANGE_TYPE=bluefin
EXCHANGE__BLUEFIN_PRIVATE_KEY=0xYOUR_PRODUCTION_KEY
EXCHANGE__BLUEFIN_NETWORK=mainnet
BLUEFIN_SERVICE_URL=http://bluefin-service:8080
BLUEFIN_SERVICE_API_KEY=production-api-key
TRADING__SYMBOL=BTC-PERP
TRADING__LEVERAGE=5
TRADING__ENABLE_FUTURES=true
SYSTEM__DRY_RUN=false  # Only after thorough testing!
SYSTEM__ENVIRONMENT=production
LLM__OPENAI_API_KEY=sk-PRODUCTION_KEY
RISK__MAX_DAILY_LOSS_PCT=5.0
RISK__DEFAULT_STOP_LOSS_PCT=0.5
RISK__DEFAULT_TAKE_PROFIT_PCT=1.0
PAPER_TRADING__STARTING_BALANCE=10000
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
SYSTEM__ENABLE_PERFORMANCE_MONITORING=true
```

## Getting Help

If you encounter issues:

1. **Run the validator**: `python services/scripts/validate_env.py`
2. **Check the logs**: `tail -f logs/bot.log`
3. **Verify services**: `docker-compose ps`
4. **Test connectivity**: `curl http://localhost:8081/health`
5. **Review documentation**: Check `docs/bluefin_integration.md`
6. **Start simple**: Use minimal configuration first, then add features

## Next Steps

After successfully configuring your environment:

1. **Test with paper trading**: Verify everything works with `SYSTEM__DRY_RUN=true`
2. **Monitor the dashboard**: Access `http://localhost:8000` for real-time data
3. **Review logs**: Check for any warnings or errors
4. **Optimize settings**: Adjust risk parameters based on your strategy
5. **Consider live trading**: Only after extensive testing and validation

Remember: **Always prioritize safety over profits**. Start conservatively and gradually increase your comfort level with the system.