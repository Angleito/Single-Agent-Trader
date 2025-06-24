# Functional Programming Configuration Migration Guide

## Overview

This guide provides comprehensive instructions for migrating from the legacy configuration system to the new functional programming (FP) configuration system. The FP system provides enhanced type safety, security, and error handling while maintaining full backward compatibility.

## Why Migrate to Functional Programming Configuration?

### Benefits of FP Configuration

1. **Enhanced Security**
   - Opaque types prevent accidental logging of sensitive data
   - Automatic credential masking in logs and error messages
   - Strong validation prevents configuration errors

2. **Better Error Handling**
   - Result types provide explicit error handling
   - Detailed error messages for debugging
   - No silent failures or crashes from bad configuration

3. **Type Safety**
   - Sum types for strategy configuration variants
   - Compile-time type checking prevents runtime errors
   - Strong typing for all configuration parameters

4. **Improved Validation**
   - Comprehensive validation with detailed error messages
   - Consistency checks between related configuration values
   - Security warnings for dangerous configurations

5. **Better Performance**
   - Lazy loading of configuration types
   - Caching of environment variables and validation results
   - Optimized configuration loading pipeline

## Migration Strategy

### Phase 1: Understanding Both Systems

The bot supports two configuration systems simultaneously:

**Legacy System (Current)**:
```bash
EXCHANGE__EXCHANGE_TYPE=coinbase
EXCHANGE__CDP_API_KEY_NAME=my_key
TRADING__SYMBOL=BTC-USD
SYSTEM__DRY_RUN=true
```

**Functional Programming System (New)**:
```bash
EXCHANGE_TYPE=coinbase
COINBASE_API_KEY=my_key
TRADING_PAIRS=BTC-USD
TRADING_MODE=paper
```

### Phase 2: Gradual Migration

You can migrate gradually by introducing FP environment variables alongside existing ones:

**Step 1: Add FP Variables**
```bash
# Keep existing legacy variables
EXCHANGE__EXCHANGE_TYPE=coinbase
TRADING__SYMBOL=BTC-USD
SYSTEM__DRY_RUN=true

# Add new FP variables (these take precedence)
STRATEGY_TYPE=llm
EXCHANGE_TYPE=coinbase
TRADING_PAIRS=BTC-USD
TRADING_MODE=paper
```

**Step 2: Test FP Configuration**
```bash
# Test that FP configuration loads correctly
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Config valid' if result.is_success() else f'❌ Config error: {result.failure()}')"
```

**Step 3: Migrate Gradually**
```bash
# Remove legacy variables one section at a time
# Keep both during transition for safety
```

### Phase 3: Full Migration

Once you've tested the FP system, you can remove legacy variables:

**Before (Legacy)**:
```bash
EXCHANGE__EXCHANGE_TYPE=coinbase
EXCHANGE__CDP_API_KEY_NAME=my_api_key
EXCHANGE__CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"
TRADING__SYMBOL=BTC-USD
TRADING__INTERVAL=5m
TRADING__LEVERAGE=5
SYSTEM__DRY_RUN=true
LLM__OPENAI_API_KEY=sk-...
LLM__MODEL_NAME=gpt-4
LLM__TEMPERATURE=0.1
```

**After (Functional Programming)**:
```bash
# Strategy configuration
STRATEGY_TYPE=llm
TRADING_MODE=paper
TRADING_PAIRS=BTC-USD
TRADING_INTERVAL=5m

# Exchange configuration with enhanced security
EXCHANGE_TYPE=coinbase
COINBASE_API_KEY=my_api_key
COINBASE_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"

# Rate limiting
RATE_LIMIT_RPS=10
RATE_LIMIT_RPM=100
RATE_LIMIT_RPH=1000

# LLM configuration with validation
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.1
LLM_CONFIDENCE_THRESHOLD=0.7
LLM_USE_MEMORY=false

# Feature flags
ENABLE_WEBSOCKET=true
ENABLE_RISK_MANAGEMENT=true
ENABLE_METRICS=true

# System configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_POSITIONS=3
DEFAULT_POSITION_SIZE=0.1
```

## Configuration Mapping Reference

### Exchange Configuration

| Legacy | Functional Programming | Notes |
|--------|----------------------|-------|
| `EXCHANGE__EXCHANGE_TYPE` | `EXCHANGE_TYPE` | Simplified name |
| `EXCHANGE__CDP_API_KEY_NAME` | `COINBASE_API_KEY` | Direct mapping, auto-masked |
| `EXCHANGE__CDP_PRIVATE_KEY` | `COINBASE_PRIVATE_KEY` | Opaque type, never logged |
| `EXCHANGE__BLUEFIN_PRIVATE_KEY` | `BLUEFIN_PRIVATE_KEY` | Opaque type, auto-masked |
| `EXCHANGE__BLUEFIN_NETWORK` | `BLUEFIN_NETWORK` | Same name, enhanced validation |

### Trading Configuration

| Legacy | Functional Programming | Notes |
|--------|----------------------|-------|
| `TRADING__SYMBOL` | `TRADING_PAIRS` | Supports multiple pairs |
| `TRADING__INTERVAL` | `TRADING_INTERVAL` | Enhanced validation |
| `TRADING__LEVERAGE` | N/A | Handled per strategy |
| `SYSTEM__DRY_RUN` | `TRADING_MODE` | More explicit (paper/live/backtest) |

### Strategy Configuration

| Legacy | Functional Programming | Notes |
|--------|----------------------|-------|
| N/A | `STRATEGY_TYPE` | New: momentum/mean_reversion/llm |
| `LLM__MODEL_NAME` | `LLM_MODEL` | Simplified name |
| `LLM__TEMPERATURE` | `LLM_TEMPERATURE` | Enhanced validation (0.0-2.0) |
| `LLM__OPENAI_API_KEY` | `LLM_OPENAI_API_KEY` | Same (backward compatible) |
| N/A | `LLM_CONFIDENCE_THRESHOLD` | New validation feature |
| N/A | `LLM_USE_MEMORY` | Explicit memory control |

### System Configuration

| Legacy | Functional Programming | Notes |
|--------|----------------------|-------|
| `SYSTEM__LOG_LEVEL` | `LOG_LEVEL` | Simplified name |
| N/A | `MAX_CONCURRENT_POSITIONS` | New explicit control |
| N/A | `DEFAULT_POSITION_SIZE` | New explicit control |
| N/A | `ENABLE_*` variables | New feature flags |

## Security Migration

### Credential Security Enhancements

**Before (Legacy)**:
```bash
# Credentials could be accidentally logged
EXCHANGE__CDP_API_KEY_NAME=my_api_key  # Might appear in logs
```

**After (Functional Programming)**:
```bash
# Credentials are automatically protected
COINBASE_API_KEY=my_api_key  # Shows as "APIKey(***key)" in logs
COINBASE_PRIVATE_KEY=my_key  # Shows as "PrivateKey(***)" in logs
```

### Validation Security

**Legacy System**: Silent failures, potential crashes
```python
leverage = int(os.getenv('TRADING__LEVERAGE', '5'))  # Could crash on invalid input
```

**FP System**: Explicit error handling
```python
result = parse_int_env('TRADING_LEVERAGE', 5)
if isinstance(result, Failure):
    logger.error(f"Configuration error: {result.failure()}")
    return
leverage = result.success()
```

## Strategy-Specific Migration

### LLM Strategy Migration

**Legacy Configuration**:
```bash
LLM__MODEL_NAME=gpt-4
LLM__TEMPERATURE=0.1
LLM__MAX_TOKENS=4000
```

**FP Configuration**:
```bash
STRATEGY_TYPE=llm
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.1
LLM_MAX_CONTEXT=4000
LLM_CONFIDENCE_THRESHOLD=0.7
LLM_USE_MEMORY=false
```

### New Strategy Options

**Momentum Strategy**:
```bash
STRATEGY_TYPE=momentum
MOMENTUM_LOOKBACK=20
MOMENTUM_ENTRY_THRESHOLD=0.02
MOMENTUM_EXIT_THRESHOLD=0.01
MOMENTUM_USE_VOLUME=true
```

**Mean Reversion Strategy**:
```bash
STRATEGY_TYPE=mean_reversion
MEAN_REVERSION_WINDOW=50
MEAN_REVERSION_STD_DEV=2.0
MEAN_REVERSION_MIN_VOL=0.001
MEAN_REVERSION_MAX_HOLD=100
```

## Validation and Testing

### Configuration Validation Commands

```bash
# Test basic configuration loading
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Valid' if result.is_success() else f'❌ Error: {result.failure()}')"

# Test strategy configuration
python -c "from bot.fp.types.config import build_strategy_config_from_env; print(build_strategy_config_from_env())"

# Test exchange configuration
python -c "from bot.fp.types.config import build_exchange_config_from_env; print(build_exchange_config_from_env())"

# Test system configuration
python -c "from bot.fp.types.config import build_system_config_from_env; print(build_system_config_from_env())"

# Test complete configuration with validation
python -c "from bot.fp.types.config import Config, validate_config; result = Config.from_env(); validated = validate_config(result.success()) if result.is_success() else result; print(validated)"
```

### Testing Migration Steps

1. **Backup Current Configuration**:
```bash
cp .env .env.backup
cp -r config/ config_backup/
```

2. **Create Test Environment**:
```bash
cp .env .env.test
# Edit .env.test with FP variables
```

3. **Test FP Configuration**:
```bash
# Load test environment
export $(cat .env.test | xargs)

# Test configuration
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Valid' if result.is_success() else f'❌ Error: {result.failure()}')"
```

4. **Test Bot Startup**:
```bash
python -m bot.main live --dry-run
```

## Troubleshooting Migration Issues

### Common Migration Problems

**1. Configuration Not Loading**
```bash
# Error: "Module 'bot.fp.types.config' not found"
# Solution: Ensure FP system is available
python -c "import bot.fp.types.config; print('✅ FP system available')"
```

**2. Environment Variable Conflicts**
```bash
# Error: Both legacy and FP variables set
# Solution: Remove legacy variables or ensure FP takes precedence
unset EXCHANGE__EXCHANGE_TYPE  # Remove legacy
export EXCHANGE_TYPE=coinbase  # Use FP
```

**3. Validation Failures**
```bash
# Error: "Invalid API key: too short"
# Solution: Check API key format and length
echo ${#COINBASE_API_KEY}  # Check length
```

**4. Type Conversion Errors**
```bash
# Error: "Invalid float for LLM_TEMPERATURE"
# Solution: Check numeric format
echo $LLM_TEMPERATURE  # Should be decimal like 0.1, not "0.1°"
```

### Migration Verification Checklist

- [ ] **Configuration loads without errors**
- [ ] **All required environment variables are set**
- [ ] **Credentials are properly masked in logs**
- [ ] **Bot starts successfully in paper trading mode**
- [ ] **Strategy configuration is recognized**
- [ ] **Exchange configuration is valid**
- [ ] **Feature flags work as expected**
- [ ] **Rate limits are reasonable and consistent**
- [ ] **Security settings are appropriate**
- [ ] **Backup of legacy configuration exists**

## Performance Considerations

### Configuration Loading Performance

The FP system is optimized for performance:

**Lazy Loading**: FP types are only loaded when needed
```python
# FP types loaded only when accessed
functional_config = get_functional_config()  # Lazy loading
```

**Caching**: Environment variables are cached
```python
# Environment variables cached after first read
value = parse_env_var_cached('KEY', 'default')  # Cached
```

**Validation Caching**: Validation results are cached
```python
# Configuration validation cached
validated_config = validate_config(config)  # Cached
```

### Benchmarking

```python
from bot.config import benchmark_config_loading

results = benchmark_config_loading(iterations=100)
print(f"Create settings: {results['create_settings_ms']:.2f}ms")
print(f"Load from file: {results['load_from_file_ms']:.2f}ms")
print(f"Validation: {results['validation_ms']:.2f}ms")
```

## Best Practices for FP Configuration

### 1. Use Result Types for Error Handling
```python
from bot.fp.types.result import Result, Success, Failure

def load_config() -> Result[Config, str]:
    try:
        config = Config.from_env()
        return config
    except Exception as e:
        return Failure(f"Configuration error: {e}")
```

### 2. Implement Comprehensive Validation
```python
def validate_trading_config(config: TradingConfig) -> Result[TradingConfig, str]:
    if config.position_size < 0.01 or config.position_size > 1.0:
        return Failure("Position size must be between 0.01 and 1.0")
    return Success(config)
```

### 3. Use Opaque Types for Sensitive Data
```python
@dataclass(frozen=True)
class APIKey:
    _value: str

    def __str__(self) -> str:
        return f"APIKey(***{self._value[-4:]})"
```

### 4. Implement Configuration Profiles
```python
# Conservative profile
settings = create_settings(profile='conservative')

# Custom profile with overrides
settings = create_settings(overrides={
    'trading': {'leverage': 2},
    'risk': {'max_daily_loss_pct': 2.0}
})
```

## Conclusion

The functional programming configuration system provides significant improvements in security, type safety, and error handling while maintaining full backward compatibility. Migration can be done gradually, and the enhanced validation and debugging capabilities make configuration management much more robust.

For support during migration, refer to the troubleshooting section or test configuration loading with the provided validation commands.
