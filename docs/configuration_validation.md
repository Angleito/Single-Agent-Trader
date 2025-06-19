# Configuration Validation System

The AI Trading Bot includes a comprehensive configuration validation system specifically designed to ensure robust Bluefin DEX integration. This system provides multiple layers of validation, from basic format checking to live network connectivity testing.

## Overview

The validation system consists of several components:

1. **Basic Configuration Validation** - Format and consistency checks
2. **Network Connectivity Testing** - Live endpoint accessibility
3. **Security Validation** - Private key format and secure handling
4. **Environment Consistency** - Cross-component configuration checks
5. **Runtime Monitoring** - Continuous configuration health monitoring

## Validation Tools

### 1. Environment Validator (`services/scripts/validate_env.py`)

The basic environment validator checks your `.env` file for common issues and validates basic configuration format.

```bash
# Basic validation
python services/scripts/validate_env.py

# Enhanced validation with comprehensive Bluefin checks
python services/scripts/validate_env.py
```

**Features:**
- ✅ Private key format validation (hex, mnemonic, Sui Bech32)
- ✅ Network configuration consistency
- ✅ URL format validation
- ✅ Environment-network consistency checks
- ✅ Security recommendations

### 2. Comprehensive Configuration Validator (`scripts/validate_config.py`)

Advanced validator with network testing and comprehensive analysis.

```bash
# Full validation including network tests
python scripts/validate_config.py --full

# Exchange-specific validation only
python scripts/validate_config.py --exchange-only

# Bluefin-specific validation
python scripts/validate_config.py --bluefin-only

# Export detailed report
python scripts/validate_config.py --full --export-report reports/config_validation.json

# Get automated fix suggestions
python scripts/validate_config.py --full --fix-suggestions

# Continuous monitoring mode
python scripts/validate_config.py --monitor
```

**Features:**
- 🌐 Live network connectivity testing
- 🔗 API endpoint accessibility validation
- 🔐 Comprehensive security checks
- 📊 Performance analysis
- 🔧 Automated fix suggestions
- 📈 Continuous monitoring
- 📄 Detailed reporting

### 3. Bluefin-Specific Tester (`scripts/test_bluefin_config.py`)

Focused testing utility specifically for Bluefin DEX configuration.

```bash
# Quick validation (no network tests)
python scripts/test_bluefin_config.py --quick

# Full validation with network tests
python scripts/test_bluefin_config.py

# Test specific components
python scripts/test_bluefin_config.py --test-api --test-rpc --test-service

# Test specific network
python scripts/test_bluefin_config.py --network testnet --verbose

# Private key validation only
python scripts/test_bluefin_config.py --validate-key

# Export results
python scripts/test_bluefin_config.py --export results/bluefin_test.json
```

**Features:**
- 🔷 Bluefin-specific validation
- 🔑 Advanced private key format checking
- 🌐 Network endpoint testing
- ⚡ RPC connectivity validation
- 🔧 Service connectivity testing
- 📊 Detailed result reporting

## Configuration Categories

### 1. Private Key Validation

The system supports multiple Sui private key formats:

#### Hex Format
```bash
# With 0x prefix
EXCHANGE__BLUEFIN_PRIVATE_KEY=0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# Without 0x prefix
EXCHANGE__BLUEFIN_PRIVATE_KEY=1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

#### Mnemonic Phrase
```bash
# 12-word mnemonic
EXCHANGE__BLUEFIN_PRIVATE_KEY="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"

# 24-word mnemonic
EXCHANGE__BLUEFIN_PRIVATE_KEY="word1 word2 ... word24"
```

#### Sui Bech32 Format
```bash
# Sui-specific Bech32 encoded private key
EXCHANGE__BLUEFIN_PRIVATE_KEY=suiprivkey1abc2def3ghi4jkl5mno6pqr7stu8vwx9yz0...
```

**Validation Checks:**
- ✅ Format detection and validation
- ✅ Character set validation
- ✅ Length validation
- ✅ Common error detection (all zeros, all F's)
- ✅ Mnemonic word validation
- ✅ Bech32 character set validation

### 2. Network Configuration

#### Network Settings
```bash
# Network selection
EXCHANGE__BLUEFIN_NETWORK=mainnet  # or testnet

# Custom RPC URL (optional)
EXCHANGE__BLUEFIN_RPC_URL=https://custom-sui-rpc.example.com

# Bluefin service URL
EXCHANGE__BLUEFIN_SERVICE_URL=http://bluefin-service:8080
```

**Validation Checks:**
- ✅ Network value validation (mainnet/testnet)
- ✅ URL format validation
- ✅ Network-URL consistency checking
- ✅ Environment-network consistency
- ✅ Service accessibility testing

### 3. Environment Consistency

The system checks for common misconfigurations:

#### Production Environment
```bash
SYSTEM__ENVIRONMENT=production
EXCHANGE__BLUEFIN_NETWORK=mainnet  # ✅ Correct
SYSTEM__DRY_RUN=false              # ✅ OK for production
```

#### Development Environment
```bash
SYSTEM__ENVIRONMENT=development
EXCHANGE__BLUEFIN_NETWORK=testnet  # ✅ Recommended
SYSTEM__DRY_RUN=true               # ✅ Safe for development
```

**Warning Scenarios:**
- ⚠️  Production environment with testnet network
- ⚠️  Live trading on testnet
- ⚠️  Development environment with mainnet (when not dry-run)

### 4. Security Validation

**Security Checks:**
- 🔐 Private key format and strength
- 🔐 Secure credential handling
- 🔐 Environment-appropriate settings
- 🔐 API key validation
- 🔐 File permission checks

## Network Connectivity Testing

### API Endpoint Testing

The system tests connectivity to Bluefin API endpoints:

```python
# Mainnet endpoints
REST API: https://dapi.api.sui-prod.bluefin.io
WebSocket: wss://dapi.api.sui-prod.bluefin.io
Notifications: wss://notifications.api.sui-prod.bluefin.io

# Testnet endpoints  
REST API: https://dapi.api.sui-staging.bluefin.io
WebSocket: wss://dapi.api.sui-staging.bluefin.io
Notifications: wss://notifications.api.sui-staging.bluefin.io
```

**Tests Performed:**
- ✅ DNS resolution
- ✅ HTTP connectivity
- ✅ API response validation
- ✅ Rate limit detection
- ✅ Error response handling

### RPC Connectivity Testing

Tests Sui RPC connectivity:

```python
# Default RPC endpoints
Mainnet: https://fullnode.mainnet.sui.io:443
Testnet: https://fullnode.testnet.sui.io:443

# Custom RPC (if configured)
Custom: EXCHANGE__BLUEFIN_RPC_URL
```

**Tests Performed:**
- ✅ RPC endpoint accessibility
- ✅ JSON-RPC response validation
- ✅ Network state verification
- ✅ Epoch information retrieval

### Service Connectivity Testing

Tests Bluefin SDK service connectivity:

```python
# Default service URL
Service: http://bluefin-service:8080

# Custom service (if configured)
Custom: EXCHANGE__BLUEFIN_SERVICE_URL
```

**Tests Performed:**
- ✅ Service health endpoint
- ✅ Docker container status
- ✅ Service response validation
- ✅ Error diagnostics

## Runtime Configuration Monitoring

### Configuration Monitor

The `ConfigurationMonitor` class provides runtime monitoring capabilities:

```python
from bot.config import settings

# Create monitor
monitor = settings.create_configuration_monitor()

# Register change callback
def on_config_change(settings, old_hash, new_hash):
    print(f"Configuration changed: {old_hash} -> {new_hash}")

monitor.register_change_callback(on_config_change)

# Check for changes
if monitor.check_for_changes():
    print("Configuration changed!")

# Get health status
health = monitor.get_health_status()
print(f"Status: {health['overall_status']}")
```

**Features:**
- 🔄 Change detection via configuration hashing
- 📞 Callback system for change notifications
- 🏥 Health status monitoring
- 💾 Validation result caching
- 📊 Export monitoring data

### Hot-Reloading Support

The system supports hot-reloading of configuration changes:

```python
# Enable monitoring in your application
monitor = settings.create_configuration_monitor()

# Register restart callback
def restart_on_critical_change(settings, old_hash, new_hash):
    critical_sections = ['exchange', 'llm']
    # Implementation would check if critical sections changed
    # and trigger application restart if needed
    pass

monitor.register_change_callback(restart_on_critical_change)
```

## Error Messages and Fix Suggestions

The validation system provides clear error messages and automated fix suggestions:

### Common Error Patterns

#### Private Key Errors
```
❌ Bluefin private key validation failed: Hex private key must be exactly 64 characters, got 62

💡 Fix Suggestions:
   • Ensure private key is exactly 64 hex characters
   • Check for missing characters or extra spaces
   • Verify the key was copied completely
```

#### Network Connectivity Errors
```
❌ Cannot reach Bluefin service at http://bluefin-service:8080: Connection refused

💡 Fix Suggestions:
   • Check if Docker is running: docker ps
   • Start Bluefin service: docker-compose up bluefin-service
   • Check service logs: docker-compose logs bluefin-service
   • Verify EXCHANGE__BLUEFIN_SERVICE_URL in .env
```

#### Environment Consistency Warnings
```
⚠️  Production environment using testnet network

💡 Fix Suggestions:
   • Set EXCHANGE__BLUEFIN_NETWORK=mainnet for production
   • Or change SYSTEM__ENVIRONMENT=development for testnet
   • Ensure consistency between environment and network
```

## Integration with Bot Startup

The validation system is integrated into the bot startup process:

```python
# In bot/main.py
async def startup_validation():
    """Run startup validation checks."""
    try:
        # Quick configuration test
        results = settings.test_bluefin_configuration()
        if results["status"] != "pass":
            logger.error("Configuration validation failed")
            for test in results["tests"]:
                if test["status"] == "fail":
                    logger.error(f"❌ {test['name']}: {test.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Comprehensive validation (optional)
        if os.getenv("VALIDATE_CONFIG_ON_STARTUP", "false").lower() == "true":
            validator_results = await settings.validate_configuration_comprehensive()
            if not validator_results["summary"]["is_valid"]:
                logger.warning("Comprehensive validation found issues")
                # Log warnings but don't exit unless critical errors
        
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        sys.exit(1)
```

## Best Practices

### 1. Development Workflow

```bash
# 1. Validate configuration before starting development
python scripts/test_bluefin_config.py --quick

# 2. Test network connectivity
python scripts/test_bluefin_config.py --test-api --test-rpc --test-service

# 3. Run comprehensive validation before deployment
python scripts/validate_config.py --full --fix-suggestions

# 4. Export validation report for documentation
python scripts/validate_config.py --full --export-report docs/deployment_validation.json
```

### 2. CI/CD Integration

```yaml
# Example GitHub Actions step
- name: Validate Configuration
  run: |
    python scripts/validate_config.py --exchange-only
    python scripts/test_bluefin_config.py --quick
```

### 3. Production Deployment

```bash
# 1. Validate production configuration
SYSTEM__ENVIRONMENT=production python scripts/validate_config.py --full

# 2. Test production network connectivity
EXCHANGE__BLUEFIN_NETWORK=mainnet python scripts/test_bluefin_config.py

# 3. Enable startup validation
export VALIDATE_CONFIG_ON_STARTUP=true

# 4. Start with monitoring
python scripts/validate_config.py --monitor &
python -m bot.main live
```

### 4. Troubleshooting

#### Configuration Issues
1. Run basic validation: `python services/scripts/validate_env.py`
2. Check specific components: `python scripts/test_bluefin_config.py --validate-key`
3. Test network connectivity: `python scripts/test_bluefin_config.py --test-api`
4. Get fix suggestions: `python scripts/validate_config.py --bluefin-only --fix-suggestions`

#### Network Issues
1. Test internet connectivity: `ping 8.8.8.8`
2. Test DNS resolution: `nslookup dapi.api.sui-prod.bluefin.io`
3. Test service connectivity: `curl http://bluefin-service:8080/health`
4. Check Docker services: `docker-compose ps`

#### Private Key Issues
1. Verify format: `python scripts/test_bluefin_config.py --validate-key`
2. Check for extra characters: `echo $EXCHANGE__BLUEFIN_PRIVATE_KEY | wc -c`
3. Test different formats (hex vs mnemonic vs Bech32)

## Configuration Backup and Recovery

### Create Configuration Backup

```python
from bot.config import settings

# Create backup without secrets
backup = settings.create_backup_configuration()

# Save to file
import json
with open("config_backup.json", "w") as f:
    json.dump(backup, f, indent=2)
```

### Validate Configuration Changes

```python
# Before making changes
old_hash = settings.generate_config_hash()

# After making changes
new_hash = settings.generate_config_hash()

if old_hash != new_hash:
    print("Configuration changed - running validation...")
    results = await settings.validate_configuration_comprehensive()
```

## Summary

The comprehensive configuration validation system provides:

1. **Multi-layered Validation** - From basic format checks to live network testing
2. **Multiple Tools** - Different tools for different use cases
3. **Clear Error Messages** - Actionable feedback with fix suggestions
4. **Runtime Monitoring** - Continuous configuration health monitoring
5. **Integration Support** - Easy integration with development and deployment workflows
6. **Security Focus** - Comprehensive security validation and recommendations

This system ensures robust and reliable Bluefin DEX integration while providing clear guidance for resolving configuration issues.