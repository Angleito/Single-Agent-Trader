# AI Trading Bot - Simple Deployment Guide

*Perfect for macOS with OrbStack - No Kubernetes complexity*

This guide covers deploying the AI Trading Bot using Docker with a focus on simplicity and reliability.

## Table of Contents

1. [Quick Deployment](#quick-deployment)
2. [Production Setup](#production-setup)
3. [Environment Configuration](#environment-configuration)
4. [Security Best Practices](#security-best-practices)
5. [Monitoring and Health Checks](#monitoring-and-health-checks)
6. [Backup and Recovery](#backup-and-recovery)
7. [Troubleshooting](#troubleshooting)

## Quick Deployment

### Prerequisites
- macOS with OrbStack or Docker Desktop
- Coinbase Advanced Trade API credentials
- OpenAI API key

### 3-Step Deployment

#### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd ai-trading-bot

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

#### 2. Configure API Keys
Edit `.env` file:
```bash
# Required: Coinbase API credentials
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Required: OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

#### 3. Start Trading Bot
```bash
# Start in safe dry-run mode
docker-compose up

# For development with live code reloading
docker-compose --profile dev up

# Run in background
docker-compose up -d
```

## Production Setup

### 1. Production Configuration

Create a production environment file:

```bash
# Copy example and customize
cp .env.example .env.prod
```

Edit `.env.prod` for production:
```bash
# System Configuration
ENVIRONMENT=production
DRY_RUN=false  # ONLY when ready for live trading
LOG_LEVEL=INFO
CONFIG_FILE=./config/conservative_config.json

# Trading Configuration
SYMBOL=BTC-USD
TIMEFRAME=1h
MAX_POSITION_SIZE=0.1  # Start small
RISK_PER_TRADE=0.02    # 2% risk per trade

# Safety settings (CRITICAL for production)
MAX_DAILY_LOSS=0.05    # Stop if 5% daily loss
STOP_LOSS_ENABLED=true
TAKE_PROFIT_ENABLED=true

# Exchange Configuration (LIVE TRADING)
# IMPORTANT: cb_sandbox=false for live trading
COINBASE_API_KEY=your_production_api_key
COINBASE_API_SECRET=your_production_api_secret
COINBASE_PASSPHRASE=your_production_passphrase

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
```

### 2. Start Production Bot

```bash
# Start production bot
docker-compose --env-file .env.prod up -d

# Monitor logs
docker-compose logs -f ai-trading-bot

# Check health
docker-compose ps
```

### 3. Production Validation Script

Create `scripts/validate_production.py`:

```python
#!/usr/bin/env python3
"""Production deployment validation."""

import os
import sys
sys.path.append('/app')

def validate_production_config():
    """Validate production configuration."""

    checks = []

    # Check environment
    env = os.getenv('ENVIRONMENT')
    if env == 'production':
        checks.append(("✅", "Environment set to production"))
    else:
        checks.append(("❌", f"Environment is {env}, should be production"))

    # Check dry run
    dry_run = os.getenv('DRY_RUN', 'true').lower()
    if dry_run == 'false':
        checks.append(("⚠️", "DRY_RUN disabled - LIVE TRADING ACTIVE"))
    else:
        checks.append(("✅", "DRY_RUN enabled - Safe mode"))

    # Check API keys
    if os.getenv('COINBASE_API_KEY'):
        checks.append(("✅", "Coinbase API key configured"))
    else:
        checks.append(("❌", "Coinbase API key missing"))

    if os.getenv('OPENAI_API_KEY'):
        checks.append(("✅", "OpenAI API key configured"))
    else:
        checks.append(("❌", "OpenAI API key missing"))

    # Print results
    print("🔍 Production Configuration Check:")
    for status, message in checks:
        print(f"   {status} {message}")

    # Check if any critical failures
    failures = [check for check in checks if check[0] == "❌"]
    if failures:
        print("\n❌ Critical issues found - deployment not ready")
        return False

    print("\n✅ Production configuration validated")
    return True

if __name__ == "__main__":
    success = validate_production_config()
    sys.exit(0 if success else 1)
```

Run validation:
```bash
docker exec ai-trading-bot python scripts/validate_production.py
```

## Environment Configuration

### Development vs Production

#### Development Configuration (.env)
```bash
ENVIRONMENT=development
DRY_RUN=true          # Safe paper trading
LOG_LEVEL=DEBUG       # Verbose logging
TESTING=true          # Enable test features
MAX_POSITION_SIZE=0.1 # Small positions
```

#### Production Configuration (.env.prod)
```bash
ENVIRONMENT=production
DRY_RUN=false         # Live trading (when ready)
LOG_LEVEL=INFO        # Standard logging
TESTING=false         # Disable test features
MAX_POSITION_SIZE=0.05 # Conservative start
```

### Configuration Files

The bot uses JSON configuration files in `config/`:

- `development.json` - Development settings
- `conservative_config.json` - Safe production settings

Example `conservative_config.json`:
```json
{
  "trading": {
    "leverage": 3,
    "max_size_pct": 5.0,
    "min_trade_interval": 300
  },
  "risk": {
    "max_daily_loss_pct": 2.0,
    "max_concurrent_trades": 1,
    "stop_loss_pct": 1.0,
    "take_profit_pct": 2.0
  },
  "system": {
    "health_check_interval": 60,
    "log_retention_days": 30
  }
}
```

## Security Best Practices

### 1. API Key Security

#### Secure Storage
```bash
# Never commit API keys to git
echo ".env*" >> .gitignore

# Use environment variables only
# Store keys in secure password manager

# Rotate keys monthly
# Use separate keys for dev/prod
```

#### Key Validation Script
Create `scripts/test_api_keys.py`:
```python
#!/usr/bin/env python3
"""Test API key connectivity."""

import os
import asyncio
from bot.exchange.coinbase import CoinbaseClient

async def test_coinbase_connection():
    """Test Coinbase API connectivity."""
    try:
        client = CoinbaseClient()
        await client.connect()

        # Test basic API call
        accounts = await client.get_accounts()
        print("✅ Coinbase API connection successful")
        print(f"   Found {len(accounts)} accounts")

        await client.disconnect()
        return True

    except Exception as e:
        print(f"❌ Coinbase API connection failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connectivity."""
    try:
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Test API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )

        print("✅ OpenAI API connection successful")
        return True

    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔑 Testing API connections...")

    # Test APIs
    coinbase_ok = asyncio.run(test_coinbase_connection())
    openai_ok = test_openai_connection()

    if coinbase_ok and openai_ok:
        print("\n✅ All API connections successful")
    else:
        print("\n❌ Some API connections failed")
        exit(1)
```

### 2. Container Security

#### Secure Docker Configuration
```bash
# Run as non-root user (already configured in Dockerfile)
# Use read-only filesystems where possible
# Limit resource usage

# Check container security
docker run --rm -it \
  --security-opt=no-new-privileges \
  --read-only \
  --tmpfs /tmp \
  ai-trading-bot
```

### 3. Network Security

#### Firewall Setup (if running on server)
```bash
# Basic firewall (Ubuntu/Debian)
sudo ufw allow ssh
sudo ufw allow 80/tcp   # HTTP (if needed)
sudo ufw allow 443/tcp  # HTTPS (if needed)
sudo ufw --force enable
```

## Monitoring and Health Checks

### 1. Health Monitoring

#### Check Bot Health
```bash
# Check if container is running
docker-compose ps

# Check container health status
docker inspect ai-trading-bot --format='{{.State.Health.Status}}'

# View recent logs
docker-compose logs --tail=50 ai-trading-bot

# Follow live logs
docker-compose logs -f ai-trading-bot
```

#### Health Check Script
Create `scripts/health_check.sh`:
```bash
#!/bin/bash
set -e

echo "🏥 AI Trading Bot Health Check"
echo "=============================="

# Check container status
echo "🐳 Container Status:"
if docker-compose ps | grep -q "Up"; then
    echo "   ✅ Container running"
else
    echo "   ❌ Container not running"
    exit 1
fi

# Check health endpoint (if exposed)
echo
echo "💚 Health Endpoint:"
if curl -f http://localhost:8080/health 2>/dev/null; then
    echo "   ✅ Health endpoint responding"
else
    echo "   ⚠️  Health endpoint not accessible"
fi

# Check logs for errors
echo
echo "📝 Recent Errors:"
ERROR_COUNT=$(docker-compose logs --tail=100 ai-trading-bot | grep -c "ERROR" || true)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "   ✅ No recent errors"
else
    echo "   ⚠️  Found $ERROR_COUNT recent errors"
fi

# Check disk space
echo
echo "💾 Disk Usage:"
df -h . | tail -1 | awk '{print "   📊 " $5 " used (" $4 " free)"}'

echo
echo "✅ Health check completed"
```

### 2. Log Management

#### Log Rotation
Docker handles log rotation automatically with the configuration in `docker-compose.yml`:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

#### Log Analysis
```bash
# View logs by time range
docker-compose logs --since="1h" ai-trading-bot

# Search for specific events
docker-compose logs ai-trading-bot | grep "Trade executed"

# Check for errors
docker-compose logs ai-trading-bot | grep -E "(ERROR|CRITICAL)"

# Monitor real-time trading activity
docker-compose logs -f ai-trading-bot | grep -E "(Trade|Position|P&L)"
```

## Backup and Recovery

### 1. Data Backup

#### What to Backup
- Configuration files (`config/`)
- Trading logs (`logs/`)
- Market data (`data/`)
- Environment files (`.env`)

#### Backup Script
Create `scripts/backup.sh`:
```bash
#!/bin/bash
set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="trading_bot_backup_${TIMESTAMP}"

echo "📦 Creating backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup archive
tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    config/ \
    logs/ \
    data/ \
    .env.example \
    docker-compose.yml

echo "✅ Backup created: $BACKUP_DIR/$BACKUP_NAME.tar.gz"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "trading_bot_backup_*.tar.gz" -mtime +30 -delete

echo "🧹 Old backups cleaned up"
```

#### Automated Backups
Add to crontab for automatic daily backups:
```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/trading-bot && ./scripts/backup.sh
```

### 2. Recovery Procedures

#### Restore from Backup
```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

echo "🔄 Restoring from backup: $BACKUP_FILE"

# Stop bot
docker-compose down

# Extract backup
tar -xzf "$BACKUP_FILE"

# Validate configuration
if [ -f ".env" ]; then
    echo "✅ Configuration restored"
else
    echo "⚠️  No .env file in backup - you'll need to recreate it"
fi

# Restart bot
docker-compose up -d

echo "✅ Restore completed"
```

#### Emergency Stop Procedure
```bash
#!/bin/bash
# scripts/emergency_stop.sh

echo "🚨 EMERGENCY STOP INITIATED"

# Stop bot immediately
docker-compose down

# Create emergency log entry
echo "$(date): Emergency stop executed" >> logs/emergency.log

# Send notification (if configured)
if [ -n "$SLACK_WEBHOOK" ]; then
    curl -X POST "$SLACK_WEBHOOK" \
         -H 'Content-type: application/json' \
         --data '{"text":"🚨 Trading bot emergency stop executed"}'
fi

echo "✅ Emergency stop completed"
echo "📝 Check logs/emergency.log for details"
```

## Troubleshooting

### 1. Common Issues

#### Bot Won't Start
```bash
# Check container logs
docker-compose logs ai-trading-bot

# Check configuration
docker exec ai-trading-bot python scripts/validate_production.py

# Test API connections
docker exec ai-trading-bot python scripts/test_api_keys.py

# Check environment variables
docker exec ai-trading-bot env | grep -E "(COINBASE|OPENAI)"
```

#### No Trading Activity
```bash
# Verify not in dry-run mode (if you want live trading)
docker exec ai-trading-bot env | grep DRY_RUN

# Check market data connection
docker exec ai-trading-bot python -c "
from bot.data.market import MarketDataClient
client = MarketDataClient()
print('Market data:', client.get_latest_price('BTC-USD'))
"

# Check strategy decisions
docker-compose logs ai-trading-bot | grep -E "(Decision|Strategy|Signal)"
```

#### High Memory Usage
```bash
# Check resource usage
docker stats ai-trading-bot

# Restart if needed
docker-compose restart ai-trading-bot

# Check for memory leaks in logs
docker-compose logs ai-trading-bot | grep -i memory
```

### 2. Performance Monitoring

#### Resource Usage
```bash
# Monitor in real-time
docker stats ai-trading-bot

# Check historical usage
docker exec ai-trading-bot cat /proc/meminfo
docker exec ai-trading-bot cat /proc/loadavg
```

#### Trading Performance
```bash
# Check recent trades
docker-compose logs ai-trading-bot | grep "Trade executed" | tail -10

# Check P&L
docker-compose logs ai-trading-bot | grep "P&L" | tail -5

# Check positions
docker-compose logs ai-trading-bot | grep "Position" | tail -5
```

### 3. Debugging Mode

#### Enable Debug Logging
```bash
# Temporary debug mode
docker-compose exec ai-trading-bot python -c "
import logging
logging.getLogger().setLevel(logging.DEBUG)
"

# Or restart with debug environment
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose restart ai-trading-bot
```

#### Interactive Shell
```bash
# Access container shell
docker-compose exec ai-trading-bot bash

# Python interactive session
docker-compose exec ai-trading-bot python
```

---

## Summary

This simplified deployment guide focuses on Docker-based deployment for macOS users with OrbStack. Key benefits:

- **Simple Setup**: 3 commands to get started
- **Safe Defaults**: Starts in dry-run mode
- **Clear Documentation**: Step-by-step instructions
- **Production Ready**: When you're ready to go live
- **Easy Monitoring**: Built-in health checks and logging
- **Reliable Backups**: Automated backup procedures

**Next Steps:**
1. Follow the Quick Deployment section
2. Test with dry-run mode
3. Gradually move to live trading with small positions
4. Set up monitoring and backups

Remember: Always start with `DRY_RUN=true` and small position sizes!
