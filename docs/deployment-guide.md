# AI Trading Bot Deployment Guide

This comprehensive guide covers the complete deployment process for the AI Trading Bot, including pre-deployment checks, step-by-step deployment procedures, rollback strategies, monitoring setup, troubleshooting, and security considerations.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Step-by-Step Deployment Process](#step-by-step-deployment-process)
3. [Rollback Procedures](#rollback-procedures)
4. [Monitoring Setup](#monitoring-setup)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)
6. [Security Considerations](#security-considerations)
7. [Deployment Scripts Reference](#deployment-scripts-reference)

## Pre-Deployment Checklist

Before initiating any deployment, ensure all items in this checklist are completed:

### System Requirements

- [ ] **Operating System**: Linux (Ubuntu 20.04+ or similar) or macOS
- [ ] **Python**: Version 3.10 or higher installed
- [ ] **Docker**: Version 20.10+ with Docker Compose v2.0+
- [ ] **Memory**: Minimum 4GB RAM (8GB recommended)
- [ ] **Storage**: At least 20GB free disk space
- [ ] **Network**: Stable internet connection with outbound HTTPS access

### Environment Configuration

- [ ] `.env` file created from `.env.example` with all required variables set
- [ ] API keys configured and validated:
  - [ ] `LLM__OPENAI_API_KEY` - OpenAI API key
  - [ ] Exchange credentials (Coinbase or Bluefin)
  - [ ] Optional: `MCP_ENABLED` and related memory system keys
- [ ] Trading mode verified: `SYSTEM__DRY_RUN=true` for paper trading
- [ ] Configuration files reviewed in `config/` directory
- [ ] Trading parameters set appropriately for your risk tolerance

### Code Quality & Testing

- [ ] All tests passing: `poetry run pytest`
- [ ] Type checking passing: `poetry run mypy bot/`
- [ ] Security audit clean: `./scripts/security-audit.sh`
- [ ] Code quality checks passing: `./scripts/code-quality.sh`
- [ ] No uncommitted changes: `git status`

### Infrastructure Preparation

- [ ] Docker permissions configured: `./setup-docker-permissions.sh`
- [ ] Required directories created with proper permissions
- [ ] Backup strategy in place
- [ ] Monitoring infrastructure ready (if applicable)

### Safety Verification

- [ ] Paper trading mode confirmed (unless intentionally going live)
- [ ] Risk management parameters reviewed
- [ ] Emergency stop procedures understood
- [ ] Team notification plan in place

## Step-by-Step Deployment Process

### 1. Local Development Deployment

For initial testing and development:

```bash
# 1. Clone and setup repository
git clone <repository-url>
cd ai-trading-bot

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your configuration

# 4. Set up Docker permissions (IMPORTANT!)
./setup-docker-permissions.sh

# 5. Run security audit
./scripts/security-audit.sh

# 6. Start services
docker-compose up -d

# 7. Monitor logs
docker-compose logs -f ai-trading-bot
```

### 2. Automated Safe Deployment

Use the automated deployment script for production-ready deployments:

```bash
# Run the automated safe deployment
./scripts/automated-safe-deployment.sh

# This script will:
# - Check prerequisites
# - Create backups
# - Validate configuration
# - Run all tests
# - Build Docker images
# - Deploy services
# - Verify deployment
# - Set up monitoring
```

The script includes automatic rollback on failure and comprehensive logging.

### 3. VPS Remote Deployment

For deploying to a VPS or cloud server:

```bash
# 1. SSH into your VPS
ssh user@your-vps-ip

# 2. Clone repository
git clone <repository-url>
cd ai-trading-bot

# 3. Set up environment
cp .env.example .env
nano .env  # Configure with production values

# 4. Run VPS deployment script
./scripts/vps-deploy.sh

# This will:
# - Validate system requirements
# - Set up VPS-specific configurations
# - Optimize network settings
# - Harden security
# - Deploy all services
# - Set up automated backups
```

### 4. Multi-Platform Deployment

For x64 Linux servers (common VPS platforms):

```bash
# Use the x64 Linux deployment script
./scripts/deploy-x64-linux.sh

# This handles platform-specific optimizations
```

### 5. Production Deployment Steps

For production environments with live trading:

```bash
# 1. Final safety check
grep "SYSTEM__DRY_RUN" .env  # Ensure this is set correctly

# 2. Run comprehensive validation
poetry run python scripts/validate_config_comprehensive.py

# 3. Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 4. Enable monitoring
docker-compose up -d prometheus grafana fluent-bit

# 5. Verify health
./scripts/docker-health-check.sh
```

## Rollback Procedures

### Immediate Rollback

If deployment fails or issues are detected:

```bash
# 1. Stop all services immediately
docker-compose down

# 2. The automated deployment script creates backups
# Restore from backup if using automated deployment
cd /path/to/backup/directory
cp -r config/* /project/config/
cp .env /project/.env
cp docker-compose.yml /project/

# 3. Revert to previous Docker images
docker-compose pull  # Pull previous versions
docker-compose up -d

# 4. Verify services are running with old version
docker-compose ps
docker-compose logs --tail=100
```

### Git-Based Rollback

For code-level rollbacks:

```bash
# 1. Find the last known good commit
git log --oneline -10

# 2. Create a rollback branch
git checkout -b rollback/emergency-<date>

# 3. Reset to known good commit
git reset --hard <commit-hash>

# 4. Force push if necessary (be careful!)
git push --force-with-lease origin rollback/emergency-<date>

# 5. Redeploy
./scripts/automated-safe-deployment.sh
```

### Database/State Rollback

For memory system or persistent state:

```bash
# 1. Stop memory services
docker-compose stop mcp-memory

# 2. Backup current state
docker-compose exec mcp-memory-backup /bin/bash
tar -czf backup-$(date +%Y%m%d-%H%M%S).tar.gz /data

# 3. Restore from backup
tar -xzf /backups/memory-backup-<timestamp>.tar.gz -C /

# 4. Restart services
docker-compose up -d mcp-memory
```

## Monitoring Setup

### 1. Basic Monitoring

Built-in monitoring with Docker:

```bash
# Real-time logs
docker-compose logs -f --tail=100

# Service status
watch -n 5 'docker-compose ps'

# Resource usage
docker stats

# Health checks
./scripts/docker-health-check.sh
```

### 2. Log Aggregation

Configure Fluent Bit for centralized logging:

```bash
# Fluent Bit is configured in docker-compose
docker-compose up -d fluent-bit

# View aggregated logs
docker logs fluent-bit | jq .
```

### 3. Performance Monitoring

Monitor trading performance:

```bash
# Check trading metrics
docker-compose exec ai-trading-bot python -m bot.utils.performance_monitor

# View performance dashboard
open http://localhost:8000  # If dashboard is enabled
```

### 4. Automated Health Checks

Set up automated monitoring:

```bash
# Enable systemd timer (on Linux)
sudo systemctl enable vps-trading-monitor.timer
sudo systemctl start vps-trading-monitor.timer

# Or use cron
crontab -e
# Add: */5 * * * * /path/to/scripts/docker-health-check.sh
```

### 5. Alert Configuration

Set up alerts for critical events:

```yaml
# In config/production.json
{
  "monitoring": {
    "alert_enabled": true,
    "alert_thresholds": {
      "max_drawdown": 0.10,
      "error_rate": 0.05,
      "latency_ms": 1000
    }
  }
}
```

## Troubleshooting Common Issues

### 1. Permission Denied Errors

```bash
# Fix Docker volume permissions
./setup-docker-permissions.sh

# Manual fix if script fails
sudo chown -R $(id -u):$(id -g) ./logs ./data ./tmp
```

### 2. Service Won't Start

```bash
# Check logs for specific service
docker-compose logs ai-trading-bot | tail -50

# Common fixes:
# - Check .env file is properly formatted
# - Verify all required env vars are set
# - Ensure ports aren't already in use
lsof -i :8765  # Check if port is in use
```

### 3. Connection Issues

```bash
# Test exchange connectivity
./scripts/diagnose-bluefin-connectivity.sh

# Validate Docker network
./scripts/validate-docker-network.sh

# Check DNS resolution
docker-compose exec ai-trading-bot nslookup api.openai.com
```

### 4. Memory/Resource Issues

```bash
# Check system resources
free -h
df -h

# Increase Docker resources (Docker Desktop)
# Settings > Resources > Increase Memory/CPU

# Clean up Docker resources
docker system prune -a --volumes
```

### 5. Configuration Errors

```bash
# Validate configuration
poetry run python scripts/validate_config_comprehensive.py

# Check for type errors
poetry run python scripts/validate-types.py

# Test specific config
docker-compose exec ai-trading-bot python -c "from bot.config import get_config; print(get_config())"
```

## Security Considerations

### 1. Pre-Deployment Security Audit

Always run security checks before deployment:

```bash
# Run comprehensive security audit
./scripts/security-audit.sh

# Python security scan
poetry run bandit -r bot/ -ll

# Check for secrets in code
poetry run detect-secrets scan

# Vulnerability scanning
poetry run safety check
```

### 2. Environment Security

Protect sensitive configuration:

```bash
# Set secure permissions on .env
chmod 600 .env

# Never commit .env to git
echo ".env" >> .gitignore

# Use environment-specific configs
cp .env.example .env.production
# Edit with production values
```

### 3. Container Security

Implement container best practices:

```yaml
# In docker-compose.yml
services:
  ai-trading-bot:
    user: "1000:1000"  # Don't run as root
    read_only: true    # Read-only filesystem
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

### 4. Network Security

For VPS deployments:

```bash
# Configure firewall (Ubuntu/Debian)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 443/tcp   # HTTPS
sudo ufw default deny incoming
sudo ufw enable

# Use SSH keys instead of passwords
ssh-copy-id user@vps-ip
```

### 5. API Key Security

Best practices for API keys:

- Never hardcode keys in source code
- Use separate keys for dev/staging/production
- Rotate keys regularly
- Monitor key usage for anomalies
- Use read-only keys where possible

### 6. Monitoring Security

```bash
# Monitor for suspicious activity
grep -i "error\|fail\|denied" /var/log/trading-bot/*.log

# Set up fail2ban for repeated failures
sudo apt-get install fail2ban
sudo systemctl enable fail2ban
```

## Deployment Scripts Reference

### Core Deployment Scripts

1. **`setup-docker-permissions.sh`**
   - Sets up proper permissions for Docker volumes
   - Prevents permission denied errors
   - Run before first deployment

2. **`scripts/automated-safe-deployment.sh`**
   - Comprehensive deployment automation
   - Includes all safety checks
   - Automatic rollback on failure
   - Creates deployment logs and reports

3. **`scripts/vps-deploy.sh`**
   - VPS-specific deployment
   - Network optimization
   - Security hardening
   - Automated backup setup

4. **`scripts/deploy-x64-linux.sh`**
   - Platform-specific deployment for x64 Linux
   - Handles architecture-specific optimizations

### Validation Scripts

5. **`scripts/validate-docker-network.sh`**
   - Verifies Docker networking
   - Checks service connectivity
   - Tests inter-container communication

6. **`scripts/security-audit.sh`**
   - Comprehensive security checks
   - Scans for hardcoded secrets
   - Verifies file permissions
   - Checks container security

7. **`scripts/docker-health-check.sh`**
   - Service health verification
   - Resource usage monitoring
   - Log error detection

### Utility Scripts

8. **`scripts/vps-healthcheck.sh`**
   - Automated health monitoring for VPS
   - Service restart on failure
   - Disk and memory checks

9. **`scripts/code-quality.sh`**
   - Pre-deployment code quality checks
   - Runs linting, formatting, and tests

10. **`monitor_bot.sh`**
    - Real-time bot monitoring
    - Performance tracking
    - Alert generation

## Post-Deployment Checklist

After deployment, verify:

- [ ] All services are running: `docker-compose ps`
- [ ] No errors in logs: `docker-compose logs --tail=100`
- [ ] API connections working: Check logs for successful connections
- [ ] Paper trading confirmed (if intended): Look for "PAPER TRADING MODE" in logs
- [ ] Monitoring active: Verify health checks are running
- [ ] Backups configured: Check backup scripts are scheduled
- [ ] Documentation updated: Update deployment date and version

## Emergency Contacts and Procedures

Document your emergency procedures:

1. **Emergency Stop**: `docker-compose down`
2. **Support Contact**: [Your contact info]
3. **Escalation Path**: [Team escalation procedure]
4. **Incident Response**: [Link to incident response plan]

Remember: Always test in paper trading mode first, monitor closely after deployment, and have a rollback plan ready.
