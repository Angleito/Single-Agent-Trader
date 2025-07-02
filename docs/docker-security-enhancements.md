# Docker Security Enhancements Guide

This guide documents the comprehensive security enhancements implemented for the AI Trading Bot's Docker deployment.

## Table of Contents
1. [Overview](#overview)
2. [Security Features Implemented](#security-features-implemented)
3. [SecureString Module](#securestring-module)
4. [Docker Secrets Management](#docker-secrets-management)
5. [Security Profiles](#security-profiles)
6. [Log Sanitization](#log-sanitization)
7. [Container Scanning](#container-scanning)
8. [Deployment Guide](#deployment-guide)
9. [Security Best Practices](#security-best-practices)

## Overview

The security enhancements implement defense-in-depth strategies to protect sensitive data, prevent unauthorized access, and ensure secure container operations. All sensitive data (API keys, private keys) are protected through multiple layers of security.

## Security Features Implemented

### 1. Memory Protection
- **SecureString class**: Secure handling of sensitive data in memory
- **Memory zeroing**: Automatic cleanup of sensitive data
- **Memory locking**: Prevents swapping sensitive data to disk (Linux)
- **Constant-time comparison**: Prevents timing attacks

### 2. Docker Secrets
- **External secret management**: Sensitive data stored outside containers
- **File-based secrets**: Mounted at `/run/secrets/`
- **Automatic rotation support**: Built-in secret rotation capabilities
- **Swarm mode integration**: Full Docker Swarm secrets support

### 3. Container Hardening
- **Read-only root filesystem**: Prevents runtime modifications
- **Non-root execution**: All containers run as non-privileged users
- **Capability dropping**: Removes all unnecessary Linux capabilities
- **Security profiles**: Custom seccomp and AppArmor profiles

### 4. Log Sanitization
- **Automatic redaction**: Sensitive data masked in all logs
- **Pattern-based detection**: Recognizes API keys, passwords, private keys
- **Recursive sanitization**: Works with nested data structures
- **Exception handling**: Sanitizes stack traces and error messages

### 5. Security Scanning
- **Vulnerability scanning**: Trivy, Grype for container images
- **Secret detection**: GitLeaks, detect-secrets for code
- **SAST analysis**: Bandit, Semgrep for code security
- **License compliance**: Automated license checking

## SecureString Module

The SecureString module (`bot/security/memory.py`) provides secure memory handling:

```python
from bot.security import SecureString, SecureEnvironment

# Create a secure string
api_key = SecureString("sk-1234567890abcdef")

# Use in API calls (only when necessary)
client = OpenAI(api_key=api_key.get_value())

# Automatic cleanup when object is deleted
del api_key  # Memory is securely zeroed

# Load from environment securely
secure_key = SecureEnvironment.get("API_KEY")
```

### Features:
- **Automatic memory zeroing**: Multiple overwrite patterns
- **Platform-specific security**: Uses `explicit_bzero` (Linux) or `memset_s` (macOS)
- **Context manager support**: Ensures cleanup with `with` statements
- **Masked string representation**: Shows as `SecureString(***)` in logs

## Docker Secrets Management

### Setting Up Docker Secrets

1. **Initialize Docker Swarm** (required for secrets):
```bash
docker swarm init
```

2. **Create secrets using the management script**:
```bash
./scripts/security/manage-docker-secrets.sh
```

3. **Deploy with secrets**:
```bash
docker-compose -f docker-compose.yml -f docker-compose.secrets.yml up -d
```

### Secret Loading in Application

The application automatically loads secrets from Docker or environment:

```python
from bot.security.secrets_loader import SecretsLoader

# Automatically checks Docker secrets first, then env vars
api_key = SecretsLoader.get_openai_api_key()
if api_key:
    # Use api_key.get_value() only when needed
    client = OpenAI(api_key=api_key.get_value())
```

## Security Profiles

### Seccomp Profile

Custom seccomp profile (`docker/security/seccomp-trading-bot.json`) restricts system calls:
- Allows only necessary syscalls for Python and networking
- Blocks dangerous operations (mount, kernel modules, etc.)
- Platform-aware (x86_64, ARM64)

### AppArmor Profile

AppArmor profile (`docker/security/apparmor-trading-bot`) provides:
- File access restrictions (read-only app code)
- Network access control
- Capability restrictions
- Protection against privilege escalation

### Loading Security Profiles

```bash
# Load AppArmor profile
sudo apparmor_parser -r ./docker/security/apparmor-trading-bot

# Deploy with security profiles
docker-compose -f docker-compose.secure.yml up -d
```

## Log Sanitization

### Automatic Log Sanitization

All logs are automatically sanitized:

```python
from bot.security.log_sanitizer import setup_secure_logging

# Setup logging with automatic sanitization
logger = setup_secure_logging()

# This will be automatically sanitized
logger.info(f"API Key: {api_key}")  # Logs: "API Key: sk-***REDACTED***"
```

### Custom Sanitization Patterns

Add custom patterns for domain-specific data:

```python
from bot.security.log_sanitizer import LogSanitizer
import re

sanitizer = LogSanitizer(additional_patterns=[
    (re.compile(r'(account_id=)(\d+)'), r'\1***'),
    (re.compile(r'(order_)([a-f0-9]{8})'), r'\1***')
])
```

## Container Scanning

### Local Security Scanning

Run comprehensive security scans locally:

```bash
# Run all security scans
./scripts/security/run-security-scan.sh

# Install missing security tools
./scripts/security/run-security-scan.sh install
```

### CI/CD Security Pipeline

GitHub Actions workflow (`.github/workflows/security-scan.yml`) runs:
- Dependency vulnerability scanning (Safety)
- Container vulnerability scanning (Trivy)
- Secret detection (TruffleHog, GitLeaks)
- SAST analysis (CodeQL, Semgrep)
- License compliance checking

## Deployment Guide

### Secure Deployment Steps

1. **Prepare environment**:
```bash
# Set up permissions
./setup-docker-permissions.sh

# Create required directories
mkdir -p logs data mcp_data security_logs
```

2. **Configure secrets**:
```bash
# Initialize Swarm mode
docker swarm init

# Create Docker secrets
./scripts/security/manage-docker-secrets.sh
```

3. **Load security profiles**:
```bash
# Load AppArmor profile
sudo apparmor_parser -r ./docker/security/apparmor-trading-bot

# Verify profile loaded
sudo aa-status | grep ai-trading-bot
```

4. **Deploy secure stack**:
```bash
# Deploy with all security features
docker-compose \
  -f docker-compose.secure.yml \
  -f docker-compose.secrets.yml \
  up -d
```

5. **Verify deployment**:
```bash
# Check container security
docker inspect ai-trading-bot-secure | jq '.[0].HostConfig.SecurityOpt'

# Verify read-only filesystem
docker exec ai-trading-bot-secure touch /test 2>&1 | grep -q "Read-only file system"

# Check non-root user
docker exec ai-trading-bot-secure whoami  # Should show: botuser
```

## Security Best Practices

### 1. Secret Management
- **Never commit secrets** to version control
- **Use Docker secrets** in production
- **Rotate secrets regularly** (monthly minimum)
- **Monitor secret access** in logs

### 2. Container Security
- **Update base images** regularly
- **Scan images** before deployment
- **Use minimal base images** (Alpine preferred)
- **Sign container images** for integrity

### 3. Runtime Security
- **Enable all security features** in production
- **Monitor container behavior** for anomalies
- **Set resource limits** to prevent DoS
- **Use network segmentation** between services

### 4. Operational Security
- **Regular security audits**: Run scans weekly
- **Incident response plan**: Document procedures
- **Access control**: Limit who can deploy
- **Audit logging**: Track all operations

### 5. Development Security
- **Security training**: Educate team members
- **Code reviews**: Focus on security
- **Dependency updates**: Automate with Dependabot
- **Security testing**: Include in CI/CD

## Monitoring and Alerts

### Security Monitoring

The security monitor sidecar provides:
- File integrity monitoring
- Resource usage alerts
- Unauthorized access detection

### Setting Up Alerts

```yaml
# Add to docker-compose.secure.yml
services:
  security-monitor:
    environment:
      - ALERT_WEBHOOK=https://your-webhook-url
      - ALERT_THRESHOLD_CPU=80
      - ALERT_THRESHOLD_MEMORY=80
```

## Troubleshooting

### Common Issues

1. **Secrets not loading**:
   - Verify Docker Swarm is initialized
   - Check secret exists: `docker secret ls`
   - Verify correct secret name in compose file

2. **Permission denied errors**:
   - Check user/group IDs match
   - Verify directory permissions
   - Run setup script: `./setup-docker-permissions.sh`

3. **Security profile errors**:
   - Verify AppArmor is enabled: `aa-status`
   - Check seccomp support: `docker info | grep seccomp`
   - Try running without profiles to isolate issue

4. **Memory protection not working**:
   - Some features require elevated privileges
   - Memory locking may need `ulimit` adjustments
   - Check system limits: `ulimit -l`

## Conclusion

These security enhancements provide comprehensive protection for the AI Trading Bot:
- **Data Protection**: Secure handling of all sensitive information
- **Container Security**: Multiple layers of runtime protection
- **Operational Security**: Automated scanning and monitoring
- **Compliance Ready**: Audit trails and security documentation

Regular security reviews and updates ensure continued protection against evolving threats.
