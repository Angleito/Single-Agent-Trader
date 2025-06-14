# Paper Trading Migration Guide

## Overview

We've simplified the trading bot's configuration by consolidating multiple trading toggles into a single `SYSTEM__DRY_RUN` environment variable. This change makes the system easier to understand and reduces configuration errors.

### What Changed

**Before**: Multiple potential toggles with confusing logic
- `DRY_RUN` - Old format without proper namespacing
- `ENABLE_PAPER_TRADING` - Redundant toggle
- `USE_PAPER_TRADING` - Another redundant toggle
- `FORCE_LIVE_TRADING` - Inverse toggle causing confusion
- `ENABLE_LIVE_TRADING` - Yet another redundant toggle

**After**: Single toggle with clear behavior
- `SYSTEM__DRY_RUN=true` - Paper trading mode (safe, default)
- `SYSTEM__DRY_RUN=false` - Live trading mode (real money)

## Migration Steps

### 1. Update Your .env File

Remove any old variables and ensure you're using the new format:

```bash
# Remove these lines if they exist:
DRY_RUN=true
ENABLE_PAPER_TRADING=true
USE_PAPER_TRADING=true
FORCE_LIVE_TRADING=false
ENABLE_LIVE_TRADING=false

# Use this single line instead:
SYSTEM__DRY_RUN=true  # Set to false ONLY for live trading
```

### 2. Update Docker Commands

The docker-compose files have been updated to respect the environment variable:

```bash
# Before (with CLI flags)
docker-compose run ai-trading-bot live --dry-run
docker-compose run ai-trading-bot live --no-dry-run

# After (respects .env file)
docker-compose up                          # Uses SYSTEM__DRY_RUN from .env
```

### 3. Update Command Line Usage

The CLI no longer requires dry-run flags - it reads from the environment:

```bash
# Before (various combinations)
python -m bot.main live --dry-run
python -m bot.main live --no-dry-run

# After (simple and clear)
python -m bot.main live                    # Respects SYSTEM__DRY_RUN from .env
poetry run ai-trading-bot live             # Respects SYSTEM__DRY_RUN from .env
```

## Configuration Examples

### Paper Trading Mode (Safe - Default)

**.env file:**
```bash
# This is the SINGLE control for paper trading vs live trading
SYSTEM__DRY_RUN=true  # Paper trading mode (safe, no real money)
```

**Startup display:**
```
🚀 Starting AI Trading Bot in DRY-RUN mode
📝 Mode: Paper Trading (No real orders)
```

### Live Trading Mode (Real Money - Dangerous)

**.env file:**
```bash
# WARNING: This enables REAL trading with REAL money!
SYSTEM__DRY_RUN=false  # Live trading mode (DANGER: uses real money!)
```

**Startup display:**
```
⚠️  Starting AI Trading Bot in LIVE mode
💰 Mode: Real Trading (Real money at risk!)
⚠️  LIVE TRADING CONFIRMATION REQUIRED
```

## Common Issues and Solutions

### Issue: "Unknown argument --dry-run"

**Solution**: Remove `--dry-run` or `--no-dry-run` from your commands. The bot now reads the setting from the .env file.

### Issue: Bot still in paper trading mode when you want live trading

**Solution**: 
1. Check your .env file has `SYSTEM__DRY_RUN=false`
2. Restart the bot to pick up the new setting
3. Confirm the startup message shows "LIVE mode"

### Issue: Accidentally trading with real money

**Solution**: 
1. Immediately set `SYSTEM__DRY_RUN=true` in your .env file
2. The bot defaults to paper trading for safety
3. Always check the startup logs before proceeding

### Issue: Docker not respecting .env settings

**Solution**: 
1. Ensure your .env file is in the project root
2. Use `docker-compose down` and `docker-compose up` to restart
3. Check that docker-compose.yml references `${SYSTEM__DRY_RUN:-true}`

## Safety Features

1. **Safe by Default**: If `SYSTEM__DRY_RUN` is not set, defaults to `true` (paper trading)
2. **Clear Warnings**: Live trading mode shows prominent warnings and requires confirmation
3. **Single Source of Truth**: One environment variable eliminates conflicting configurations
4. **Explicit Configuration**: Must explicitly set to `false` for live trading

## Technical Details

- Environment variable uses double underscore (`__`) for Pydantic nested configuration
- Maps to `settings.system.dry_run` in the Python code
- Paper trading uses `PaperTradingAccount` class for simulation
- Live trading uses `CoinbaseClient` for real order execution

## Quick Reference

| Mode | Environment Variable | Trading Behavior | Risk Level |
|------|---------------------|------------------|------------|
| Paper Trading | `SYSTEM__DRY_RUN=true` | Simulated trades only | No risk |
| Live Trading | `SYSTEM__DRY_RUN=false` | Real money trades | High risk |

## Need Help?

If you encounter issues during migration:
1. Check the bot's startup message for the current mode
2. Verify your .env file has the correct `SYSTEM__DRY_RUN` setting
3. Ensure you're not using old CLI flags (--dry-run, --no-dry-run)
4. Remember: when in doubt, `SYSTEM__DRY_RUN=true` is the safe choice