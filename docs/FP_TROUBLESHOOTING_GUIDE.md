# Functional Programming Integration Troubleshooting Guide

## Overview

This guide helps diagnose and resolve common issues with the functional programming (FP) configuration system and related integrations.

## Quick Diagnostic Commands

### Configuration Health Check

```bash
# Complete FP system health check
python -c "
from bot.fp.types.config import Config, validate_config
result = Config.from_env()
if result.is_success():
    validation = validate_config(result.success())
    if validation.is_success():
        print('✅ FP system fully operational')
    else:
        print(f'⚠️ Configuration issues: {validation.failure()}')
else:
    print(f'❌ Configuration failed: {result.failure()}')
"
```

### Component-Specific Checks

```bash
# Check exchange configuration
python -c "from bot.fp.types.config import build_exchange_config_from_env; result = build_exchange_config_from_env(); print('✅ Exchange OK' if result.is_success() else f'❌ Exchange: {result.failure()}')"

# Check strategy configuration  
python -c "from bot.fp.types.config import build_strategy_config_from_env; result = build_strategy_config_from_env(); print('✅ Strategy OK' if result.is_success() else f'❌ Strategy: {result.failure()}')"

# Check system configuration
python -c "from bot.fp.types.config import build_system_config_from_env; result = build_system_config_from_env(); print('✅ System OK' if result.is_success() else f'❌ System: {result.failure()}')"
```

## Common Issues and Solutions

### 1. Configuration Loading Issues

#### Issue: "Config validation failed"

**Symptoms**:
- Bot fails to start
- Error messages about configuration validation
- Import errors related to config modules

**Diagnostic Commands**:
```bash
# Get detailed error information
python -c "
from bot.fp.types.config import Config
result = Config.from_env()
if result.is_failure():
    print(f'Error: {result.failure()}')
    print('Check your environment variables')
"
```

**Common Causes & Solutions**:

1. **Missing Environment Variables**:
   ```bash
   # Check which variables are set
   env | grep -E "(EXCHANGE_TYPE|TRADING_MODE|STRATEGY_TYPE)"
   
   # Set missing variables
   export EXCHANGE_TYPE=coinbase
   export TRADING_MODE=paper
   export STRATEGY_TYPE=llm
   ```

2. **Invalid Environment Variable Values**:
   ```bash
   # Check for typos in values
   export TRADING_MODE=paper  # ✅ Correct
   # export TRADING_MODE=Paper  # ❌ Wrong case
   
   export EXCHANGE_TYPE=coinbase  # ✅ Correct
   # export EXCHANGE_TYPE=Coinbase  # ❌ Wrong case
   ```

#### Issue: "Import Error: No module named 'bot.fp'"

**Symptoms**:
- ModuleNotFoundError when importing FP modules
- Python can't find functional programming modules

**Solutions**:
```bash
# Ensure you're in the right directory
pwd  # Should show .../cursorprod

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Verify bot.fp exists
ls -la bot/fp/

# Reinstall dependencies
poetry install
```

### 2. Exchange Configuration Issues

#### Issue: Coinbase Authentication Failures

**Symptoms**:
- "COINBASE_API_KEY not set" error
- "Invalid API key: too short" error
- Authentication failures with Coinbase API

**Diagnostic Commands**:
```bash
# Check API key format
python -c "
from bot.fp.types.config import APIKey
api_key = APIKey.create('${COINBASE_API_KEY:-not_set}')
print('✅ API key valid' if api_key.is_success() else f'❌ {api_key.failure()}')
"

# Check private key format
python -c "
from bot.fp.types.config import PrivateKey
private_key = PrivateKey.create('${COINBASE_PRIVATE_KEY:-not_set}')
print('✅ Private key valid' if private_key.is_success() else f'❌ {private_key.failure()}')
"
```

**Solutions**:
1. **Set Correct Environment Variables**:
   ```bash
   # Use FP format (recommended)
   export COINBASE_API_KEY=your_actual_api_key_here
   export COINBASE_PRIVATE_KEY=your_actual_private_key_here
   
   # Or use legacy format
   export EXCHANGE__CDP_API_KEY_NAME=your_api_key
   export EXCHANGE__CDP_PRIVATE_KEY=your_private_key
   ```

2. **Verify API Key Length**:
   ```bash
   # API keys should be substantial length
   echo "API key length: ${#COINBASE_API_KEY}"  # Should be > 10
   echo "Private key length: ${#COINBASE_PRIVATE_KEY}"  # Should be > 20
   ```

#### Issue: Bluefin Authentication Failures

**Symptoms**:
- "BLUEFIN_PRIVATE_KEY not set" error
- "Invalid private key: too short" error
- Connection failures to Bluefin network

**Diagnostic Commands**:
```bash
# Check Bluefin private key format
python -c "
from bot.fp.types.config import PrivateKey
key = '${BLUEFIN_PRIVATE_KEY:-not_set}'
if key.startswith('0x') and len(key) == 66:
    result = PrivateKey.create(key)
    print('✅ Bluefin key format valid' if result.is_success() else f'❌ {result.failure()}')
else:
    print(f'❌ Invalid format: should be 0x + 64 hex chars, got {len(key)} chars')
"
```

**Solutions**:
1. **Correct Private Key Format**:
   ```bash
   # Bluefin private key should be hex format with 0x prefix
   export BLUEFIN_PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
   # Length should be exactly 66 characters (0x + 64 hex chars)
   ```

2. **Network Configuration**:
   ```bash
   # Set correct network
   export BLUEFIN_NETWORK=mainnet  # or testnet
   
   # Optional: custom RPC URL
   export BLUEFIN_RPC_URL=https://sui-mainnet.bluefin.io
   ```

### 3. Strategy Configuration Issues

#### Issue: "Unknown strategy type"

**Symptoms**:
- Strategy validation fails
- Error about invalid strategy type
- Bot fails to initialize strategy

**Diagnostic Commands**:
```bash
# Check current strategy configuration
echo "Strategy type: ${STRATEGY_TYPE:-not_set}"

# Test strategy validation
python -c "
from bot.fp.types.config import build_strategy_config_from_env
result = build_strategy_config_from_env()
print('✅ Strategy valid' if result.is_success() else f'❌ {result.failure()}')
"
```

**Solutions**:
```bash
# Use valid strategy types
export STRATEGY_TYPE=llm              # ✅ LLM strategy
export STRATEGY_TYPE=momentum         # ✅ Momentum strategy  
export STRATEGY_TYPE=mean_reversion   # ✅ Mean reversion strategy

# Not valid:
# export STRATEGY_TYPE=ai             # ❌ Invalid
# export STRATEGY_TYPE=LLM            # ❌ Wrong case
```

#### Issue: LLM Configuration Problems

**Symptoms**:
- "Invalid temperature" errors
- OpenAI API key validation failures
- LLM model validation errors

**Diagnostic Commands**:
```bash
# Check LLM configuration
python -c "
from bot.fp.types.config import build_strategy_config_from_env
import os
os.environ['STRATEGY_TYPE'] = 'llm'
result = build_strategy_config_from_env()
print('✅ LLM config valid' if result.is_success() else f'❌ {result.failure()}')
"
```

**Solutions**:
```bash
# Set correct LLM parameters
export STRATEGY_TYPE=llm
export LLM_MODEL=gpt-4                    # Valid model names
export LLM_TEMPERATURE=0.7                # Range: 0.0-2.0
export LLM_MAX_CONTEXT=4000               # Positive integer
export LLM_USE_MEMORY=false               # true/false
export LLM_CONFIDENCE_THRESHOLD=0.7       # Range: 0.0-1.0
export LLM_OPENAI_API_KEY=sk-your-key     # Must start with sk-
```

### 4. Type System Issues

#### Issue: Result Type Handling Errors

**Symptoms**:
- Errors about Result types
- "is_success()" or "is_failure()" method errors
- Type conversion issues

**Diagnostic Commands**:
```bash
# Test Result type functionality
python -c "
from bot.fp.types.result import Success, Failure
s = Success(42)
f = Failure('test error')
print(f'Success test: {s.is_success()} (should be True)')
print(f'Failure test: {f.is_failure()} (should be True)')
print(f'Success value: {s.success()} (should be 42)')
print(f'Failure value: {f.failure()} (should be test error)')
"
```

**Solutions**:
1. **Check Result handling in custom code**:
   ```python
   # ✅ Correct Result handling
   from bot.fp.types.result import Result
   
   result = some_function_returning_result()
   if result.is_success():
       value = result.success()
       # Use value
   else:
       error = result.failure()
       # Handle error
   ```

2. **Common Result type mistakes**:
   ```python
   # ❌ Don't access value directly
   # value = result.value  # Wrong
   
   # ✅ Check success first
   if result.is_success():
       value = result.success()
   ```

### 5. Import and Dependency Issues

#### Issue: Circular Import Errors

**Symptoms**:
- "ImportError: cannot import name" with circular references
- Module loading failures
- Startup errors related to imports

**Diagnostic Commands**:
```bash
# Test for circular imports
python -c "
try:
    import bot.fp.types.config
    import bot.config
    print('✅ No circular imports detected')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

**Solutions**:
1. **Check import order**:
   ```python
   # ✅ Import FP modules first
   from bot.fp.types.config import Config
   from bot.config import settings  # Then legacy
   ```

2. **Use lazy imports if needed**:
   ```python
   # In functions where circular imports occur
   def some_function():
       from bot.fp.types.config import Config  # Lazy import
       return Config.from_env()
   ```

#### Issue: Missing Dependencies

**Symptoms**:
- ModuleNotFoundError for specific packages
- Import failures for external libraries
- Version compatibility issues

**Solutions**:
```bash
# Reinstall all dependencies
poetry install

# Check for outdated packages
poetry show --outdated

# Update specific packages if needed
poetry update

# Check Python version compatibility
python --version  # Should be 3.12+
```

### 6. Runtime Issues

#### Issue: Performance Problems

**Symptoms**:
- Slow configuration loading
- High memory usage
- Timeouts during startup

**Diagnostic Commands**:
```bash
# Measure config loading time
python -c "
import time
from bot.fp.types.config import Config
start = time.time()
result = Config.from_env()
elapsed = time.time() - start
print(f'Config loading time: {elapsed:.3f}s')
if elapsed > 0.1:
    print('⚠️ Slow loading - check configuration complexity')
else:
    print('✅ Loading time acceptable')
"
```

**Solutions**:
1. **Reduce configuration complexity**:
   ```bash
   # Minimize environment variables
   # Use defaults where possible
   # Avoid complex validation in tight loops
   ```

2. **Check for memory leaks**:
   ```bash
   # Test repeated config loading
   python -c "
   import gc
   from bot.fp.types.config import Config
   for i in range(100):
       Config.from_env()
   gc.collect()
   print('✅ Memory test completed')
   "
   ```

#### Issue: Feature Flag Conflicts

**Symptoms**:
- Unexpected feature behavior
- Conflicts between legacy and FP settings
- Features not enabling/disabling correctly

**Diagnostic Commands**:
```bash
# Check feature flag status
python -c "
from bot.fp.types.config import build_system_config_from_env
result = build_system_config_from_env()
if result.is_success():
    features = result.success().features
    print(f'WebSocket: {features.enable_websocket}')
    print(f'Memory: {features.enable_memory}')
    print(f'Risk Management: {features.enable_risk_management}')
    print(f'Metrics: {features.enable_metrics}')
else:
    print(f'❌ {result.failure()}')
"
```

**Solutions**:
```bash
# Set clear feature flags
export ENABLE_WEBSOCKET=true
export ENABLE_MEMORY=false
export ENABLE_RISK_MANAGEMENT=true
export ENABLE_METRICS=true
export ENABLE_PAPER_TRADING=true
export ENABLE_BACKTESTING=true
export ENABLE_NOTIFICATIONS=false
```

## Advanced Troubleshooting

### Debug Mode

**Enable detailed debugging**:
```bash
# Set debug logging
export LOG_LEVEL=DEBUG

# Enable detailed error traces
python -c "
import traceback
try:
    from bot.fp.types.config import Config
    result = Config.from_env()
    if result.is_failure():
        print(f'Detailed error: {result.failure()}')
except Exception as e:
    traceback.print_exc()
"
```

### Configuration Dump

**Dump current configuration state**:
```bash
# Show all environment variables
env | grep -E "(EXCHANGE|TRADING|STRATEGY|LLM|ENABLE)" | sort

# Show FP configuration interpretation
python -c "
from bot.fp.types.config import Config
result = Config.from_env()
if result.is_success():
    config = result.success()
    print('Exchange type:', type(config.exchange).__name__)
    print('Strategy type:', type(config.strategy).__name__)
    print('Trading mode:', config.system.mode)
    print('Log level:', config.system.log_level)
else:
    print('Configuration failed to load')
"
```

### System Information

**Gather system information for support**:
```bash
# System info script
cat > debug_info.sh << 'EOF'
#!/bin/bash
echo "=== System Information ==="
python --version
echo "Working directory: $(pwd)"
echo "Git commit: $(git rev-parse HEAD)"
echo

echo "=== Environment Variables ==="
env | grep -E "(EXCHANGE|TRADING|STRATEGY|LLM|ENABLE)" | sort
echo

echo "=== Python Path ==="
python -c "import sys; print('\n'.join(sys.path))"
echo

echo "=== FP System Test ==="
python -c "
from bot.fp.types.config import Config
result = Config.from_env()
print('FP Config:', 'SUCCESS' if result.is_success() else result.failure())
"
echo

echo "=== Legacy System Test ==="
python -c "
try:
    from bot.config import settings
    print('Legacy Config: SUCCESS')
except Exception as e:
    print(f'Legacy Config: FAILED - {e}')
"
EOF

chmod +x debug_info.sh
./debug_info.sh
```

## Getting Help

### Information to Gather

When reporting issues, include:

1. **Error Messages**: Full error output
2. **Configuration**: Relevant environment variables (mask sensitive data)
3. **System Info**: Python version, OS, git commit
4. **Reproduction Steps**: Exact commands that cause the issue
5. **Expected vs Actual**: What should happen vs what actually happens

### Self-Help Resources

1. **Configuration Reference**: Check `CLAUDE.md` for complete config options
2. **Migration Guide**: `FP_CONFIGURATION_MIGRATION_GUIDE.md` for migration issues  
3. **Testing Checklist**: `FP_INTEGRATION_TESTING_CHECKLIST.md` for validation
4. **Exchange Guides**: `bluefin_integration.md` and `BLUEFIN_SETUP_GUIDE.md`

### Quick Fixes

**Reset to Working State**:
```bash
# Backup current config
cp .env .env.debug.backup

# Use minimal working configuration
cat > .env << 'EOF'
TRADING_MODE=paper
EXCHANGE_TYPE=coinbase
STRATEGY_TYPE=llm
COINBASE_API_KEY=test_key_1234567890
COINBASE_PRIVATE_KEY=test_private_key_123456789012345678901234567890
LLM_OPENAI_API_KEY=sk-test123
TRADING_PAIRS=BTC-USD
TRADING_INTERVAL=5m
LOG_LEVEL=INFO
EOF

# Test minimal config
python -c "from bot.fp.types.config import Config; result = Config.from_env(); print('✅ Minimal config works' if result.is_success() else f'❌ Still failing: {result.failure()}')"
```

Remember: The FP system is designed to provide clear error messages. Read the error output carefully - it usually contains the exact information needed to fix the issue.