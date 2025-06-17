# Environment Setup and Integration Guide

This guide covers the comprehensive environment setup, configuration validation, and monitoring integration for the AI Trading Bot.

## 🚀 Quick Start

### 1. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```bash
# Required: Choose your LLM provider
LLM__PROVIDER=openai  # or anthropic, ollama
LLM__OPENAI_API_KEY=your_actual_openai_key

# Required: Coinbase credentials (for live trading)
EXCHANGE__CB_API_KEY=your_coinbase_key
EXCHANGE__CB_API_SECRET=your_coinbase_secret
EXCHANGE__CB_PASSPHRASE=your_coinbase_passphrase

# Recommended: Start with safety settings
SYSTEM__DRY_RUN=true
EXCHANGE__CB_SANDBOX=true
TRADING__LEVERAGE=3
RISK__MAX_DAILY_LOSS_PCT=2.0
```

### 2. Validate Configuration

Run the validation script to check your setup:

```bash
python scripts/validate_config.py
```

This will:
- ✅ Validate all environment variables
- 🔍 Test API connectivity
- 🏥 Check system health
- 📊 Generate detailed reports
- 💡 Provide recommendations

### 3. Start the Bot

If validation passes, start the bot:

```bash
python -m bot.main
```

## 📋 Environment Variables Reference

### Trading Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING__SYMBOL` | BTC-USD | Trading pair symbol |
| `TRADING__INTERVAL` | 1m | Chart timeframe |
| `TRADING__LEVERAGE` | 5 | Trading leverage (1-20) |
| `TRADING__MAX_SIZE_PCT` | 20.0 | Max position size (% of equity) |

### LLM Provider Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM__PROVIDER` | openai | Provider: openai, anthropic, ollama |
| `LLM__MODEL_NAME` | gpt-4o | Model name |
| `LLM__TEMPERATURE` | 0.1 | Response randomness (0.0-2.0) |
| `LLM__OPENAI_API_KEY` | - | OpenAI API key (required for OpenAI) |
| `LLM__ANTHROPIC_API_KEY` | - | Anthropic API key (required for Anthropic) |

### Exchange Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `EXCHANGE__CB_API_KEY` | - | Coinbase API key |
| `EXCHANGE__CB_API_SECRET` | - | Coinbase API secret |
| `EXCHANGE__CB_PASSPHRASE` | - | Coinbase passphrase |
| `EXCHANGE__CB_SANDBOX` | true | Use sandbox for testing |

### Risk Management
| Variable | Default | Description |
|----------|---------|-------------|
| `RISK__MAX_DAILY_LOSS_PCT` | 5.0 | Daily loss limit (%) |
| `RISK__MAX_CONCURRENT_TRADES` | 3 | Max open positions |
| `RISK__DEFAULT_STOP_LOSS_PCT` | 2.0 | Default stop loss (%) |
| `RISK__DEFAULT_TAKE_PROFIT_PCT` | 4.0 | Default take profit (%) |

### System Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `SYSTEM__DRY_RUN` | true | Safe testing mode |
| `SYSTEM__ENVIRONMENT` | development | Environment type |
| `SYSTEM__LOG_LEVEL` | INFO | Logging verbosity |
| `SYSTEM__ENABLE_MONITORING` | true | Health monitoring |

## 🔧 Configuration Management

### Trading Profiles

The bot supports predefined risk profiles:

```python
from bot.config_utils import ConfigManager, setup_configuration
from bot.config import TradingProfile

# Load with specific profile
settings = setup_configuration(profile=TradingProfile.CONSERVATIVE)

# Switch profiles with backup
config_manager = ConfigManager()
new_settings = config_manager.switch_profile(
    settings,
    TradingProfile.AGGRESSIVE,
    save_current=True
)
```

**Available Profiles:**
- `conservative`: Low risk, low leverage (2x), 1% daily loss limit
- `moderate`: Balanced risk, medium leverage (5x), 3% daily loss limit
- `aggressive`: High risk, high leverage (10x), 5% daily loss limit
- `custom`: User-defined parameters

### Configuration Backup and Export

```python
# Create backup
backup_path = config_manager.create_config_backup(settings)

# List backups
backups = config_manager.list_config_backups()

# Export configuration
json_export = config_manager.export_configuration(settings, "json")
env_export = config_manager.export_configuration(settings, "env")

# Import configuration
imported_settings = config_manager.import_configuration(backup_path)
```

## 🏥 Health Monitoring

### Health Check Endpoints

The bot provides comprehensive health monitoring:

```python
from bot.health import create_health_endpoints

endpoints = create_health_endpoints(settings)

# Basic health check (fast)
health = endpoints.get_health()

# Detailed health check (comprehensive)
detailed_health = endpoints.get_health_detailed()

# Performance metrics
metrics = endpoints.get_metrics()

# Configuration status
config_status = endpoints.get_configuration_status()

# Readiness check
readiness = endpoints.get_readiness()
```

### Monitoring Integration

#### Prometheus Metrics Export

```python
from bot.health import create_monitoring_exporter

exporter = create_monitoring_exporter(settings)
prometheus_metrics = exporter.export_prometheus_metrics()

# Metrics include:
# - trading_bot_uptime_seconds
# - trading_bot_health_status
# - trading_bot_memory_usage_bytes
# - trading_bot_cpu_usage_percent
```

#### JSON Summary Export

```python
json_summary = exporter.export_json_summary()
snapshot_file = exporter.save_monitoring_snapshot()
```

### Health Check Components

The health monitoring system checks:

1. **System Resources**
   - CPU usage (warning >80%)
   - Memory usage (warning >85%)
   - Disk space (critical >90%)

2. **API Connectivity**
   - LLM provider accessibility
   - Exchange API connectivity
   - Network latency and timeouts

3. **File System**
   - Directory permissions
   - Disk space availability
   - Log file writability

4. **Configuration Integrity**
   - Parameter validation
   - Risk management settings
   - Environment consistency

## 🛡️ Security Best Practices

### API Key Management

1. **Never commit `.env` files** to version control
2. **Use strong, unique API keys** for all services
3. **Enable 2FA** on all exchange accounts
4. **Regularly rotate API keys** (monthly recommended)
5. **Use sandbox mode** for testing

### Configuration Security

```bash
# Set proper file permissions
chmod 600 .env
chmod 600 config/*.json

# Backup configurations securely
tar -czf config_backup.tar.gz config/
gpg -c config_backup.tar.gz  # Encrypt backup
```

### Runtime Security

1. **Start with dry-run mode** (`SYSTEM__DRY_RUN=true`)
2. **Use conservative settings** initially
3. **Monitor continuously** in production
4. **Set up alerts** for unusual activity
5. **Keep logs secure** and monitor access

## 📊 Startup Validation

### Comprehensive Validation

The startup validator performs:

- ✅ **Environment Variables**: Checks all required variables
- 🔌 **API Connectivity**: Tests LLM and exchange APIs
- 🖥️ **System Dependencies**: Validates Python version and modules
- 📁 **File Permissions**: Ensures writable directories
- ⚙️ **Configuration Integrity**: Validates parameter consistency

### Validation Levels

**Critical Errors** (prevent startup):
- Missing required API keys (for live trading)
- Invalid Python version (<3.8)
- Missing required modules
- Unwritable directories
- Extremely risky configurations

**Warnings** (allow startup with caution):
- High leverage settings
- Aggressive risk parameters
- Performance concerns
- Non-optimal configurations

### Example Validation Output

```bash
🤖 AI Trading Bot - Configuration Validation & Health Check
============================================================

📋 Loading configuration...
✅ Configuration loaded successfully
   Environment: development
   Profile: moderate
   Dry Run: true

🔍 Running comprehensive validation...

📊 Validation Results:
   Status: ✅ Valid
   Warnings: 2
     ⚠️  Leverage above 5x may be risky for beginners
     ⚠️  Consider enabling real-time data updates

🏥 Health Check Results:
   Overall Status: ✅ Healthy
   System: ✅ Healthy
   Apis: ✅ Healthy
   Filesystem: ✅ Healthy
   Configuration: ⚠️  Warning

📈 Performance Metrics:
   Uptime: 0s
   Memory Usage: 45.2 MB
   CPU Usage: 2.1%

💡 Recommendations:
   1. Currently in dry-run mode - switch to live trading when ready
   2. Consider reducing leverage for safer trading
   3. Enable real-time market data updates

✅ System is ready for operation!
   💡 Running in dry-run mode (safe for testing)
```

## 🚨 Troubleshooting

### Common Issues

**Configuration Validation Fails:**
```bash
# Check environment variables
python -c "from bot.config import create_settings; settings = create_settings(); print('OK')"

# Validate specific settings
python scripts/validate_config.py
```

**API Connectivity Issues:**
```bash
# Test OpenAI API
curl -H "Authorization: Bearer $LLM__OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Test Coinbase API (requires proper auth)
# See Coinbase Advanced Trade API documentation
```

**System Resource Issues:**
```bash
# Check available memory
free -h

# Check disk space
df -h

# Check Python modules
pip list | grep -E "(pandas|numpy|pydantic|requests)"
```

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "OpenAI API key is required" | Missing `LLM__OPENAI_API_KEY` | Set the environment variable |
| "Cannot write to data directory" | Permission issue | Check directory permissions |
| "Python 3.8+ required" | Old Python version | Upgrade Python |
| "Leverage above 20x is extremely risky" | High leverage setting | Reduce `TRADING__LEVERAGE` |

## 📈 Production Deployment

### Production Checklist

- [ ] API keys configured and tested
- [ ] `SYSTEM__DRY_RUN=false` (only when ready!)
- [ ] `EXCHANGE__CB_SANDBOX=false` (for live trading)
- [ ] `SYSTEM__ENVIRONMENT=production`
- [ ] Monitoring and alerts configured
- [ ] Backup strategy implemented
- [ ] Security review completed

### Monitoring Setup

1. **Set up health check monitoring**:
   ```bash
   # Monitor health endpoint
   curl http://localhost:8080/health
   ```

2. **Configure log monitoring**:
   ```bash
   tail -f logs/bot.log | grep -E "(ERROR|CRITICAL)"
   ```

3. **Set up alerts**:
   ```bash
   # Configure webhook alerts
   export SYSTEM__ALERT_WEBHOOK_URL="https://hooks.slack.com/..."
   ```

### Performance Tuning

- **Memory**: Ensure >512MB available
- **CPU**: Monitor usage <80% sustained
- **Disk**: Keep >1GB free space
- **Network**: Stable connection required
- **Latency**: <100ms to exchange APIs

## 🔗 Integration Examples

### Docker Integration

```dockerfile
# Health check in Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from bot.health import create_health_endpoints; \
                 from bot.config import create_settings; \
                 endpoints = create_health_endpoints(create_settings()); \
                 health = endpoints.get_health(); \
                 exit(0 if health['status'] == 'healthy' else 1)"
```

### Kubernetes Integration

```yaml
# Kubernetes readiness/liveness probes
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: trading-bot
    image: trading-bot:latest
    readinessProbe:
      exec:
        command:
        - python
        - -c
        - "from bot.health import create_health_endpoints; from bot.config import create_settings; endpoints = create_health_endpoints(create_settings()); readiness = endpoints.get_readiness(); exit(0 if readiness['ready'] else 1)"
      initialDelaySeconds: 10
      periodSeconds: 30
    livenessProbe:
      exec:
        command:
        - python
        - -c
        - "from bot.health import create_health_endpoints; from bot.config import create_settings; endpoints = create_health_endpoints(create_settings()); health = endpoints.get_liveness(); exit(0 if health['alive'] else 1)"
      initialDelaySeconds: 30
      periodSeconds: 60
```

## 📚 API Reference

### Configuration Functions

```python
# Setup and validation
from bot.config_utils import (
    setup_configuration,
    validate_configuration,
    create_startup_report,
    ConfigManager,
    HealthMonitor
)

# Health monitoring
from bot.health import (
    create_health_endpoints,
    create_monitoring_exporter,
    HealthCheckEndpoints,
    MonitoringExporter
)
```

### Configuration Classes

```python
from bot.config import (
    Settings,
    TradingProfile,
    Environment,
    create_settings
)
```

For detailed API documentation, see the docstrings in each module.
