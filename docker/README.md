# Docker Configuration Guide

This directory contains documentation for the simplified Docker configuration structure of the AI Trading Bot.

## Simplified Structure

We've consolidated multiple Docker Compose files into a streamlined 3-file structure:

### Core Files

1. **`docker-compose.yml`** - Base configuration
   - Core service definitions
   - Default settings suitable for most use cases
   - Platform compatibility settings
   - Basic resource limits

2. **`docker-compose.override.yml`** - Development overrides (auto-loaded)
   - Safe defaults for local development
   - Paper trading enforced by default
   - Exchange switching support
   - Debug-friendly settings
   - More generous resource limits

3. **`docker-compose.production.yml`** - Production overrides
   - Security hardening
   - Optimized resource limits for VPS/cloud
   - Production logging settings
   - Network isolation

### Specialized Files (Retained)

- **`docker-compose.test.yml`** - Testing configuration
- **`docker-compose.encrypted.yml`** - Secrets management
- **`docker-compose.stress-test.yml`** - Performance testing

## Usage Guide

### Local Development (Default)

```bash
# Uses docker-compose.yml + docker-compose.override.yml automatically
docker-compose up

# With Coinbase (default)
docker-compose up

# With Bluefin
EXCHANGE__EXCHANGE_TYPE=bluefin docker-compose --profile bluefin up
```

### Production Deployment

```bash
# Uses docker-compose.yml + docker-compose.production.yml
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d

# With environment file
docker-compose -f docker-compose.yml -f docker-compose.production.yml --env-file .env.production up -d
```

### Testing

```bash
# Run tests
docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm test
```

## Exchange Configuration

### Switching Between Exchanges

The simplified structure makes it easy to switch between Coinbase and Bluefin:

```bash
# Coinbase (default)
docker-compose up

# Bluefin - Method 1: Environment variable
EXCHANGE__EXCHANGE_TYPE=bluefin docker-compose --profile bluefin up

# Bluefin - Method 2: Override in .env file
echo "EXCHANGE__EXCHANGE_TYPE=bluefin" >> .env
docker-compose --profile bluefin up
```

### Exchange-Specific Settings

Settings are automatically applied based on `EXCHANGE__EXCHANGE_TYPE`:

- **Coinbase**: Uses sandbox by default in development
- **Bluefin**: Uses testnet by default, requires the bluefin-service

## Environment Variables

### Key Variables

```bash
# Exchange selection
EXCHANGE__EXCHANGE_TYPE=coinbase  # or bluefin

# Trading mode (safety)
SYSTEM__DRY_RUN=true             # Paper trading (safe)
SYSTEM__DRY_RUN=false            # Live trading (dangerous!)

# Logging
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR

# Development
DEBUG=true                       # Enable debug mode
```

### Complete Example .env

```bash
# Exchange Configuration
EXCHANGE__EXCHANGE_TYPE=coinbase
EXCHANGE__CDP_API_KEY_NAME=your_coinbase_key
EXCHANGE__CDP_PRIVATE_KEY=your_coinbase_private_key

# Or for Bluefin
# EXCHANGE__EXCHANGE_TYPE=bluefin
# EXCHANGE__BLUEFIN_PRIVATE_KEY=your_sui_private_key
# EXCHANGE__BLUEFIN_NETWORK=testnet

# Trading Configuration
TRADING__SYMBOL=BTC-USD
TRADING__INTERVAL=5m
TRADING__LEVERAGE=5

# AI Configuration
LLM__OPENAI_API_KEY=your_openai_key
LLM__MODEL_NAME=gpt-4
LLM__TEMPERATURE=0.1

# Safety - ALWAYS start with paper trading
SYSTEM__DRY_RUN=true
```

## Resource Management

### Development (docker-compose.override.yml)
- Generous limits for debugging
- All ports exposed locally
- Relaxed health checks

### Production (docker-compose.production.yml)
- Optimized for single-core VPS
- Memory limits: 768M (bot), 384M (bluefin), 256M (omnisearch)
- CPU limits: 70% (bot), 20% (bluefin), 10% (omnisearch)
- No external ports exposed
- Aggressive health checks

## Security Features

### Development
- Paper trading enforced by default
- Localhost-only port bindings
- Read-only filesystem where possible

### Production
- Enhanced security options
- Capability dropping
- Network isolation
- Resource limits
- Compressed logging

## Migration Guide

If you were using the old files, here's how to migrate:

### From docker-compose.bluefin-override.yml
```bash
# Old way
docker-compose -f docker-compose.yml -f docker-compose.bluefin-override.yml up

# New way
EXCHANGE__EXCHANGE_TYPE=bluefin docker-compose --profile bluefin up
```

### From docker-compose.memory-optimized.yml
```bash
# Old way
docker-compose -f docker-compose.memory-optimized.yml up

# New way (production includes memory optimizations)
docker-compose -f docker-compose.yml -f docker-compose.production.yml up
```

### From docker-compose.slim.yml
The memory optimizations from slim.yml are now incorporated into docker-compose.production.yml.

### From docker-compose.security.yml or docker-compose.secure.yml
Security features are now built into docker-compose.production.yml.

## Troubleshooting

### Permission Issues
```bash
# Run the setup script
./setup-docker-permissions.sh
```

### Service Won't Start
```bash
# Check logs
docker-compose logs -f ai-trading-bot

# Validate configuration
docker-compose config
```

### Memory Issues
```bash
# Use production configuration for limited resources
docker-compose -f docker-compose.yml -f docker-compose.production.yml up
```

### Bluefin Service Issues
```bash
# Ensure profile is active
docker-compose --profile bluefin up

# Check Bluefin service directly
docker-compose logs -f bluefin-service
```

## Best Practices

1. **Always start with paper trading** (`SYSTEM__DRY_RUN=true`)
2. **Use production config for deployment** (includes security and optimization)
3. **Keep sensitive data in .env files** (never commit these)
4. **Monitor resource usage** in production
5. **Regular backups** of data and logs directories

## File Cleanup Summary

The following files have been consolidated and can be safely removed:
- docker-compose.bluefin-fixed.yml
- docker-compose.coinbase.yml
- docker-compose.simple.yml
- docker-compose.security.yml
- docker-compose.slim.yml
- docker-compose.memory-optimized.yml
- docker-compose.bluefin-override.yml

All functionality from these files has been preserved in the new 3-file structure.