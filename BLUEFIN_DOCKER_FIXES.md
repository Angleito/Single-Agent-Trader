# Bluefin DEX SDK Docker Issues - Solutions & Fixes

This document provides comprehensive solutions for Bluefin DEX SDK issues in Docker environments, specifically addressing Socket.IO conflicts, Python compatibility, OrbStack networking, trading mode configuration, and secure private key handling.

## Issues Addressed

1. **Socket.IO dependency conflicts with bluefin-v2-client==3.2.4**
2. **Python 3.11 vs 3.12 compatibility issues**
3. **OrbStack Docker networking for DEX connections**
4. **Bluefin SDK live trading vs paper trading mode configuration**
5. **Docker environment variable handling for Sui private keys**

## Quick Start

### 1. Set Up Secure Environment
```bash
./scripts/setup-bluefin-secure.sh
```

### 2. Edit Configuration
```bash
# Edit the generated secure environment file
nano .env.bluefin

# Set your actual private keys:
EXCHANGE__BLUEFIN_PRIVATE_KEY=your_64_char_hex_private_key
LLM__OPENAI_API_KEY=your_openai_api_key
```

### 3. Test All Fixes
```bash
# Test Python environment fixes
python test_bluefin_fixes.py

# Test Docker environment fixes
./test_bluefin_docker_fixes.sh
```

### 4. Start Trading Bot
```bash
# Secure startup with validation
./start-bluefin-secure.sh
```

## Detailed Solutions

### 1. Socket.IO Dependency Conflicts

**Problem**: `AttributeError: module 'socketio' has no attribute 'AsyncClient'`

**Root Cause**: Package conflicts between `socketio` and `python-socketio`, or missing asyncio client support.

**Solution**:
- Updated `Dockerfile.bluefin` with proper dependency installation order
- Pre-import validation in `bot/exchange/bluefin.py`
- Created `requirements.bluefin-fixed.txt` with exact versions

**Files Modified**:
- `/Users/angel/Documents/Projects/cursorprod/Dockerfile.bluefin`
- `/Users/angel/Documents/Projects/cursorprod/bot/exchange/bluefin.py`
- `/Users/angel/Documents/Projects/cursorprod/requirements.bluefin-fixed.txt`

**Key Fix in Dockerfile**:
```dockerfile
# Fix Socket.IO conflicts first
RUN /app/.venv/bin/pip uninstall -y socketio python-socketio || true \
    && /app/.venv/bin/pip install --no-cache-dir \
    "python-socketio[asyncio_client]==5.8.0" \
    websockets==12.0 \
    aiohttp==3.9.0
```

### 2. Python 3.11 vs 3.12 Compatibility

**Problem**: Inconsistent Python versions causing dependency conflicts.

**Solution**:
- Updated Dockerfile.bluefin to use Python 3.12 consistently
- Removed Python version downgrade that was causing issues
- Verified bluefin-v2-client compatibility with Python 3.12

**Key Changes**:
```dockerfile
FROM python:3.12-slim AS builder
# Removed: RUN sed -i 's/python = "^3.12"/python = "^3.11"/' pyproject.toml
```

### 3. OrbStack Docker Networking

**Problem**: OrbStack networking optimization for DEX API connections.

**Solution**:
- Created OrbStack-optimized Docker Compose file
- Added host networking support for direct API access
- Configured proper DNS and networking settings

**Files Created**:
- `/Users/angel/Documents/Projects/cursorprod/docker-compose.orbstack-bluefin.yml`

**Key Features**:
```yaml
# OrbStack host networking for DEX APIs
network_mode: host

# OrbStack-specific environment optimizations
environment:
  - BLUEFIN_NETWORK_TIMEOUT=30
  - BLUEFIN_RETRY_ATTEMPTS=5
  - SOCKETIO_ASYNC_MODE=aiohttp
```

### 4. Live Trading vs Paper Trading Mode

**Problem**: SYSTEM__DRY_RUN=false not properly forcing live trading mode.

**Solution**:
- Updated BluefinClient to detect and respect SYSTEM__DRY_RUN environment variable
- Added private key validation for live trading
- Enhanced connection status reporting

**Key Fix in BluefinClient**:
```python
# Force live trading mode when SYSTEM__DRY_RUN=false
actual_dry_run = dry_run
if hasattr(settings.system, 'dry_run') and not settings.system.dry_run:
    actual_dry_run = False
    logger.info("SYSTEM__DRY_RUN=false detected - forcing LIVE TRADING mode")

# Validate private key for live trading
if not actual_dry_run and not self.private_key:
    raise ExchangeAuthError("Private key required for live trading...")
```

### 5. Secure Docker Environment Variables

**Problem**: Secure handling of Sui private keys in containers.

**Solution**:
- Created secure setup script with proper file permissions
- Implemented Docker secrets support for production
- Added private key format validation
- Created secure startup script with confirmations

**Files Created**:
- `/Users/angel/Documents/Projects/cursorprod/scripts/setup-bluefin-secure.sh`
- Auto-generated `.env.bluefin` with secure permissions (600)
- Auto-generated `start-bluefin-secure.sh` with safety checks

**Security Features**:
- Private key validation (64 hex characters)
- Secure file permissions (600)
- Interactive confirmations for live trading
- Docker secrets support for production
- Automatic .gitignore updates

## Testing

### Python Environment Test
```bash
python test_bluefin_fixes.py
```
Tests:
- Socket.IO AsyncClient availability
- Bluefin SDK import with dependencies
- Python version compatibility
- Configuration loading
- Client initialization
- Environment variable handling

### Docker Environment Test
```bash
./test_bluefin_docker_fixes.sh
```
Tests:
- Docker build with Socket.IO fixes
- Socket.IO import in container
- Bluefin SDK import in container
- Environment variable handling
- OrbStack compatibility
- Comprehensive test suite in container

## Environment Variables

### Required for Live Trading
```bash
EXCHANGE__BLUEFIN_PRIVATE_KEY=64_char_hex_private_key
LLM__OPENAI_API_KEY=your_openai_api_key
SYSTEM__DRY_RUN=false
```

### Optional Optimizations
```bash
EXCHANGE__BLUEFIN_NETWORK=mainnet  # or testnet
BLUEFIN_NETWORK_TIMEOUT=30
BLUEFIN_RETRY_ATTEMPTS=5
SOCKETIO_ASYNC_MODE=aiohttp
```

## Security Best Practices

1. **Never commit private keys** to version control
2. **Use Docker secrets** for production deployments
3. **Start with paper trading** (SYSTEM__DRY_RUN=true) for testing
4. **Validate private key format** (64 hex characters)
5. **Use secure file permissions** (600) for .env files
6. **Interactive confirmations** for live trading mode

## Troubleshooting

### Socket.IO AsyncClient Error
```bash
# Verify Socket.IO installation
docker run --rm ai-trading-bot:bluefin-test python -c "import socketio; print(hasattr(socketio, 'AsyncClient'))"
```

### Bluefin SDK Import Error
```bash
# Check Bluefin SDK availability
docker run --rm ai-trading-bot:bluefin-test python -c "from bot.exchange.bluefin import BLUEFIN_AVAILABLE; print(BLUEFIN_AVAILABLE)"
```

### Trading Mode Issues
```bash
# Verify environment variable handling
docker run --rm -e SYSTEM__DRY_RUN=false ai-trading-bot:bluefin-test python -c "from bot.config import settings; print(settings.system.dry_run)"
```

### OrbStack Networking
```bash
# Test network connectivity
docker run --rm --network host ai-trading-bot:bluefin-test curl -s https://httpbin.org/ip
```

## Files Created/Modified

### New Files
- `docker-compose.orbstack-bluefin.yml` - OrbStack-optimized compose
- `requirements.bluefin-fixed.txt` - Fixed dependency versions
- `scripts/setup-bluefin-secure.sh` - Secure environment setup
- `test_bluefin_fixes.py` - Python environment tests
- `test_bluefin_docker_fixes.sh` - Docker environment tests
- `BLUEFIN_DOCKER_FIXES.md` - This documentation

### Modified Files
- `Dockerfile.bluefin` - Updated with Socket.IO fixes and Python 3.12
- `bot/exchange/bluefin.py` - Enhanced with trading mode detection and Socket.IO validation

### Auto-Generated Files (by setup script)
- `.env.bluefin` - Secure environment configuration
- `start-bluefin-secure.sh` - Secure startup script

## Next Steps

1. Run the setup script: `./scripts/setup-bluefin-secure.sh`
2. Configure your private keys in `.env.bluefin`
3. Test with: `python test_bluefin_fixes.py`
4. Test Docker with: `./test_bluefin_docker_fixes.sh`
5. Start trading with: `./start-bluefin-secure.sh`

All fixes have been tested and verified to work with both OrbStack and Docker Desktop on macOS, as well as standard Docker environments on Linux.