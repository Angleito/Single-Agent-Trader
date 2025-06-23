# Integration Test Fixes - Specific Actions

## 1. Import Path Fixes

### Issue: CoinbaseClient moved from bot.main to bot.exchange.coinbase

**Files to update**:
- `tests/integration/test_startup_validation.py`

**Change**:
```python
# FROM:
patch.multiple(
    "bot.main.CoinbaseClient",
    ...
)

# TO:
patch.multiple(
    "bot.exchange.coinbase.CoinbaseClient",
    ...
)
```

### Issue: LLMAgent import path

**Change**:
```python
# FROM:
patch.multiple(
    "bot.main.LLMAgent",
    ...
)

# TO:
patch.multiple(
    "bot.strategy.llm_agent.LLMAgent",
    ...
)
```

## 2. Configuration Attribute Updates

### Issue: sandbox renamed to cb_sandbox

**Files to update**:
- `tests/integration/test_startup_validation.py` (line 119)

**Change**:
```python
# FROM:
assert settings.exchange.sandbox is True

# TO:
assert settings.exchange.cb_sandbox is True
```

**Also update mock return values**:
```python
# FROM:
return_value={"connected": True, "sandbox": True}

# TO:
return_value={"connected": True, "cb_sandbox": True}
```

## 3. Test Data Updates

### Issue: max_size_pct validation constraint

**Update test configurations**:
```json
// Change any max_size_pct > 50 to <= 50
{
  "trading": {
    "max_size_pct": 50.0  // Was 75.0
  }
}
```

### Issue: Required API credentials for live trading

**Add mock credentials to test configs**:
```json
{
  "exchange": {
    "cdp_api_key_name": "test_key",
    "cdp_private_key": "test_private_key"
  }
}
```

## 4. Environment Variable Isolation

### Add proper cleanup in test setup

```python
class TestStartupValidation:
    def setup_method(self):
        """Clear environment before each test"""
        # Store original env
        self.original_env = os.environ.copy()
        # Clear trading-related env vars
        for key in list(os.environ.keys()):
            if key.startswith(('TRADING_', 'EXCHANGE_', 'SYSTEM_')):
                del os.environ[key]

    def teardown_method(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)
```

## 5. Async Test Timeout Issues

### Add proper async cleanup

```python
@pytest.mark.asyncio
async def test_async_operation():
    try:
        # test code
    finally:
        # Ensure all tasks are cancelled
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
```

### Add test timeouts

```python
@pytest.mark.asyncio
@pytest.mark.timeout(10)  # Add timeout decorator
async def test_with_timeout():
    # test code
```

## Quick Fix Script

```bash
#!/bin/bash
# Fix import paths
find tests/integration -name "*.py" -exec sed -i '' 's/bot.main.CoinbaseClient/bot.exchange.coinbase.CoinbaseClient/g' {} \;
find tests/integration -name "*.py" -exec sed -i '' 's/bot.main.LLMAgent/bot.strategy.llm_agent.LLMAgent/g' {} \;

# Fix sandbox attribute
find tests/integration -name "*.py" -exec sed -i '' 's/\.sandbox/\.cb_sandbox/g' {} \;
find tests/integration -name "*.py" -exec sed -i '' 's/"sandbox"/"cb_sandbox"/g' {} \;

echo "Basic fixes applied. Manual review still needed for:"
echo "- Test data validation constraints"
echo "- Environment variable isolation"
echo "- Async test cleanup"
```
