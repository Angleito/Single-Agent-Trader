# Functional Programming Configuration Migration Guide

## Overview

This guide helps you migrate from legacy configuration to the new functional programming (FP) configuration system, which provides enhanced security, type safety, and validation.

## Migration Benefits

### Why Migrate to FP Configuration?

- **Enhanced Security**: Automatic credential masking and opaque types
- **Type Safety**: Compile-time and runtime validation
- **Better Error Messages**: Clear, actionable error reporting
- **Rate Limiting**: Built-in API protection
- **Result-based Validation**: Explicit success/failure handling
- **Future-Proof**: Foundation for advanced features

### Backward Compatibility

- ✅ **Legacy configuration continues to work**
- ✅ **Both systems can coexist**
- ✅ **No forced migration required**
- ✅ **Gradual adoption supported**

## Configuration Mapping Guide

### Exchange Configuration

#### Coinbase Configuration

**Legacy → FP Migration**:

```bash
# BEFORE (Legacy)
EXCHANGE__EXCHANGE_TYPE=coinbase
EXCHANGE__CDP_API_KEY_NAME=your_api_key
EXCHANGE__CDP_PRIVATE_KEY=your_private_key

# AFTER (FP - Enhanced Security)
EXCHANGE_TYPE=coinbase
COINBASE_API_KEY=your_api_key        # Auto-masked as APIKey(***key)
COINBASE_PRIVATE_KEY=your_private_key # Auto-masked as PrivateKey(***)
COINBASE_API_URL=https://api.coinbase.com  # Optional, has default
COINBASE_WS_URL=wss://ws.coinbase.com      # Optional, has default

# Rate Limiting (New FP Feature)
RATE_LIMIT_RPS=10      # Requests per second
RATE_LIMIT_RPM=100     # Requests per minute
RATE_LIMIT_RPH=1000    # Requests per hour
```

#### Bluefin Configuration

**Legacy → FP Migration**:

```bash
# BEFORE (Legacy)
EXCHANGE__EXCHANGE_TYPE=bluefin
EXCHANGE__BLUEFIN_PRIVATE_KEY=0x123...
EXCHANGE__BLUEFIN_NETWORK=mainnet
EXCHANGE__BLUEFIN_RPC_URL=https://custom-rpc.com

# AFTER (FP - Enhanced Security)
EXCHANGE_TYPE=bluefin
BLUEFIN_PRIVATE_KEY=0x123...           # Auto-masked as PrivateKey(***)
BLUEFIN_NETWORK=mainnet                # Validated: mainnet|testnet
BLUEFIN_RPC_URL=https://custom-rpc.com # Optional, has default

# Rate Limiting (New FP Feature)
RATE_LIMIT_RPS=10      # Requests per second
RATE_LIMIT_RPM=100     # Requests per minute
RATE_LIMIT_RPH=1000    # Requests per hour
```

### Trading Configuration

#### Basic Trading Settings

**Legacy → FP Migration**:

```bash
# BEFORE (Legacy)
SYSTEM__DRY_RUN=true
TRADING__SYMBOL=BTC-USD
TRADING__INTERVAL=1m
TRADING__LEVERAGE=5

# AFTER (FP - Enhanced Validation)
TRADING_MODE=paper                   # paper|live|backtest (validated)
TRADING_PAIRS=BTC-USD               # Comma-separated, validated
TRADING_INTERVAL=1m                 # Validated intervals
MAX_CONCURRENT_POSITIONS=3          # New: Position limits
DEFAULT_POSITION_SIZE=0.1           # New: Position sizing (10%)
```

### Strategy Configuration

#### LLM Strategy

**Legacy → FP Migration**:

```bash
# BEFORE (Legacy)
LLM__OPENAI_API_KEY=sk-...
LLM__MODEL_NAME=gpt-4
LLM__TEMPERATURE=0.1
LLM__PROVIDER=openai

# AFTER (FP - Enhanced Validation)
STRATEGY_TYPE=llm                    # Strategy selection
LLM_OPENAI_API_KEY=sk-...           # Auto-masked in logs
LLM_MODEL=gpt-4                     # Validated model names
LLM_TEMPERATURE=0.7                 # Range-validated (0.0-2.0)
LLM_MAX_CONTEXT=4000                # Context length validation
LLM_USE_MEMORY=false                # Memory features toggle
LLM_CONFIDENCE_THRESHOLD=0.7        # Confidence validation
```

#### New Strategy Types (FP Only)

**Momentum Strategy**:
```bash
STRATEGY_TYPE=momentum
MOMENTUM_LOOKBACK=20                 # Lookback period
MOMENTUM_ENTRY_THRESHOLD=0.02        # Entry threshold (2%)
MOMENTUM_EXIT_THRESHOLD=0.01         # Exit threshold (1%)
MOMENTUM_USE_VOLUME=true             # Volume confirmation
```

**Mean Reversion Strategy**:
```bash
STRATEGY_TYPE=mean_reversion
MEAN_REVERSION_WINDOW=50             # Window size
MEAN_REVERSION_STD_DEV=2.0           # Standard deviations
MEAN_REVERSION_MIN_VOL=0.001         # Minimum volatility
MEAN_REVERSION_MAX_HOLD=100          # Max holding period
```

### System Configuration

#### Feature Flags

**Legacy → FP Migration**:

```bash
# BEFORE (Legacy)
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
SYSTEM__ENABLE_PERFORMANCE_MONITORING=true
LOG_LEVEL=INFO

# AFTER (FP - Comprehensive Feature Control)
LOG_LEVEL=INFO                       # Validated levels
ENABLE_WEBSOCKET=true               # Real-time data
ENABLE_MEMORY=false                 # Learning features
ENABLE_BACKTESTING=true             # Backtesting
ENABLE_PAPER_TRADING=true           # Paper trading
ENABLE_RISK_MANAGEMENT=true         # Risk controls
ENABLE_NOTIFICATIONS=false          # Notifications
ENABLE_METRICS=true                 # Performance metrics
```

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

**Phase 1: Add FP Variables (Keep Legacy)**
```bash
# Keep existing legacy variables
EXCHANGE__EXCHANGE_TYPE=coinbase
SYSTEM__DRY_RUN=true

# Add new FP variables for enhanced features
EXCHANGE_TYPE=coinbase
TRADING_MODE=paper
RATE_LIMIT_RPS=10
```

**Phase 2: Test FP System**
```bash
# Validate FP configuration
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Ready for migration' if result.is_success() else f'❌ Fix: {result.failure()}')"
```

**Phase 3: Switch to FP (Remove Legacy)**
```bash
# Remove legacy variables
# EXCHANGE__EXCHANGE_TYPE=coinbase  # Remove
# SYSTEM__DRY_RUN=true             # Remove

# Keep only FP variables
EXCHANGE_TYPE=coinbase
TRADING_MODE=paper
```

### Strategy 2: Complete Migration

**Create New .env File**:
```bash
# Backup existing configuration
cp .env .env.legacy.backup

# Create new FP configuration
cat > .env << 'EOF'
# Functional Programming Configuration
EXCHANGE_TYPE=coinbase
COINBASE_API_KEY=your_api_key
COINBASE_PRIVATE_KEY=your_private_key
TRADING_MODE=paper
STRATEGY_TYPE=llm
TRADING_PAIRS=BTC-USD
TRADING_INTERVAL=5m
LLM_OPENAI_API_KEY=your_openai_key
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LOG_LEVEL=INFO
ENABLE_WEBSOCKET=true
ENABLE_RISK_MANAGEMENT=true
EOF
```

### Strategy 3: Hybrid Approach

**Use FP for New Features, Legacy for Existing**:
```bash
# Keep legacy for main functionality
EXCHANGE__EXCHANGE_TYPE=coinbase
SYSTEM__DRY_RUN=true
TRADING__SYMBOL=BTC-USD

# Use FP for new features
RATE_LIMIT_RPS=10
ENABLE_RISK_MANAGEMENT=true
MAX_CONCURRENT_POSITIONS=3
```

## Validation and Testing

### Pre-Migration Validation

**Test Current Configuration**:
```bash
# Validate legacy configuration still works
python -c "from bot.config import settings; print('✅ Legacy config working')"

# Test FP system readiness
python -c "from bot.fp.types.config import Config; print('✅ FP system ready')"
```

### Post-Migration Validation

**Test FP Configuration**:
```bash
# Validate complete FP configuration
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Migration successful' if result.is_success() else f'❌ Error: {result.failure()}')"

# Test specific components
python -c "from bot.fp.types.config import build_exchange_config_from_env; result = build_exchange_config_from_env(); print('✅ Exchange config' if result.is_success() else f'❌ {result.failure()}')"

python -c "from bot.fp.types.config import build_strategy_config_from_env; result = build_strategy_config_from_env(); print('✅ Strategy config' if result.is_success() else f'❌ {result.failure()}')"
```

### Security Validation

**Verify Credential Masking**:
```bash
# Check that sensitive data is masked in logs
python -c "
from bot.fp.types.config import APIKey, PrivateKey
api_key = APIKey.create('real_api_key_here')
private_key = PrivateKey.create('real_private_key_here')
print('API Key masked:', str(api_key.success()))
print('Private Key masked:', str(private_key.success()))
"
```

## Migration Checklist

### Pre-Migration
- [ ] **Backup current configuration**: `cp .env .env.backup`
- [ ] **Test current system**: Ensure bot runs with existing config
- [ ] **Review new features**: Understand FP system benefits
- [ ] **Plan migration strategy**: Choose gradual, complete, or hybrid approach

### During Migration
- [ ] **Update environment variables**: Follow mapping guide above
- [ ] **Add new FP features**: Configure rate limiting, feature flags
- [ ] **Validate configuration**: Use FP validation commands
- [ ] **Test incrementally**: Validate each component separately

### Post-Migration
- [ ] **Run full validation**: Complete FP system test
- [ ] **Test paper trading**: Ensure trading functionality works
- [ ] **Verify security**: Check credential masking
- [ ] **Monitor performance**: Ensure no performance regression
- [ ] **Update documentation**: Document any custom configurations

## Common Migration Issues

### Issue 1: "Config validation failed"

**Symptom**: FP config loading fails
**Cause**: Missing or invalid environment variables
**Solution**:
```bash
# Debug specific validation issue
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print(result.failure() if result.is_failure() else 'Config OK')"
```

### Issue 2: "Invalid private key: too short"

**Symptom**: Private key validation fails
**Solution**:
```bash
# Ensure private key format is correct
# Coinbase: PEM format or base64
# Bluefin: 0x prefixed hex string, 64 chars
```

### Issue 3: "Unknown strategy type"

**Symptom**: Strategy configuration fails
**Solution**:
```bash
# Use valid strategy types
STRATEGY_TYPE=llm              # Valid
STRATEGY_TYPE=momentum         # Valid
STRATEGY_TYPE=mean_reversion   # Valid
# STRATEGY_TYPE=invalid        # Invalid
```

### Issue 4: Mixed Configuration Conflicts

**Symptom**: Unexpected behavior with both legacy and FP vars
**Solution**:
```bash
# Remove conflicting legacy variables or use only one system
# Don't mix: EXCHANGE__EXCHANGE_TYPE and EXCHANGE_TYPE
```

## Advanced Migration Scenarios

### Docker Environment Migration

**Update docker-compose.yml**:
```yaml
services:
  ai-trading-bot:
    environment:
      # FP Configuration
      - EXCHANGE_TYPE=coinbase
      - TRADING_MODE=paper
      - STRATEGY_TYPE=llm
      # Secrets from .env file
      - COINBASE_API_KEY=${COINBASE_API_KEY}
      - COINBASE_PRIVATE_KEY=${COINBASE_PRIVATE_KEY}
      - LLM_OPENAI_API_KEY=${LLM_OPENAI_API_KEY}
```

### Multi-Environment Migration

**Development Environment**:
```bash
# .env.development
TRADING_MODE=paper
LOG_LEVEL=DEBUG
ENABLE_METRICS=true
```

**Production Environment**:
```bash
# .env.production
TRADING_MODE=live
LOG_LEVEL=INFO
ENABLE_RISK_MANAGEMENT=true
MAX_CONCURRENT_POSITIONS=1
```

### Migration Automation Script

**Create migration helper**:
```bash
#!/bin/bash
# migrate_to_fp.sh

echo "Backing up current configuration..."
cp .env .env.legacy.backup

echo "Converting legacy variables to FP format..."
# Add your specific conversion logic here

echo "Validating new configuration..."
python -c "from bot.fp.types.config import Config; result = Config.from_env(); exit(0 if result.is_success() else 1)"

if [ $? -eq 0 ]; then
    echo "✅ Migration successful!"
else
    echo "❌ Migration failed. Restoring backup..."
    cp .env.legacy.backup .env
fi
```

## Rollback Procedure

### If Migration Fails

**Immediate Rollback**:
```bash
# Restore backup
cp .env.backup .env

# Verify legacy system works
python -c "from bot.config import settings; print('✅ Rollback successful')"
```

**Investigate Issues**:
```bash
# Check what failed
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print(result.failure() if result.is_failure() else 'Should not reach here')"
```

## Migration Support

### Getting Help

1. **Check validation output**: Use FP validation commands
2. **Review error messages**: FP system provides detailed errors
3. **Test incrementally**: Migrate one component at a time
4. **Use hybrid approach**: Keep legacy while testing FP
5. **Check documentation**: Refer to component-specific guides

### Migration Timeline

- **Week 1**: Test FP system with current config
- **Week 2**: Add FP variables alongside legacy
- **Week 3**: Validate FP configuration and features
- **Week 4**: Switch to FP-only configuration
- **Week 5**: Remove legacy variables and finalize

The FP migration is **optional but recommended** for enhanced security and features. Take your time and test thoroughly at each step.