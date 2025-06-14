# Paper Trading Configuration Simplification

## Overview

We have simplified the paper trading / dry run configuration to use a single, clear control mechanism instead of multiple confusing toggles.

## The Change

### Before (Confusing)
The system had multiple environment variables that could conflict:
- `DRY_RUN`
- `SYSTEM__DRY_RUN` 
- `ENABLE_PAPER_TRADING`
- `USE_PAPER_TRADING`
- `FORCE_LIVE_TRADING`

### After (Simple)
Now there is only ONE control:
- `SYSTEM__DRY_RUN` - The master toggle for trading mode

## How It Works

### Paper Trading Mode (Safe)
```bash
SYSTEM__DRY_RUN=true
```
- Uses simulated paper trading account
- No real money involved
- Perfect for testing strategies
- Default mode for safety

### Live Trading Mode (Real Money)
```bash
SYSTEM__DRY_RUN=false
```
- Uses real Coinbase account
- Real money at risk
- Requires valid API credentials
- Only use when ready!

## Implementation Details

1. **Configuration**: The `system.dry_run` setting in `bot/config.py` controls everything
2. **Paper Trading**: When `dry_run=true`, the bot automatically uses `PaperTradingAccount` class
3. **Position Manager**: Integrates with paper trading account when in dry run mode
4. **Exchange Client**: Skips real API calls when in dry run mode

## Migration Guide

If you have an existing `.env` file with the old variables:

1. Remove these lines:
   ```bash
   DRY_RUN=...
   ENABLE_PAPER_TRADING=...
   USE_PAPER_TRADING=...
   FORCE_LIVE_TRADING=...
   ```

2. Add or update:
   ```bash
   SYSTEM__DRY_RUN=true  # or false for live trading
   ```

## Benefits

1. **Clarity**: One toggle, one purpose
2. **Safety**: Default is paper trading mode
3. **Consistency**: No conflicting settings
4. **Simplicity**: Easy to understand and use

## Related Files

- `.env` - Environment configuration
- `bot/config.py` - Configuration system
- `bot/paper_trading.py` - Paper trading implementation
- `bot/main.py` - Main bot logic that initializes paper trading
- `bot/position_manager.py` - Position tracking integration