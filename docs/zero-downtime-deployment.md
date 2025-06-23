# Zero-Downtime Deployment Guide

This guide explains how to use the zero-downtime deployment system for the AI Trading Bot. The system uses a blue-green deployment strategy to ensure continuous trading operations during updates.

## Overview

The zero-downtime deployment system allows you to:
- Deploy new versions without interrupting trading operations
- Perform health checks before switching to new versions
- Monitor deployments in real-time
- Rollback quickly if issues are detected
- Maintain trading state across deployments

## Prerequisites

1. Docker and Docker Compose installed
2. Project setup with proper permissions (run `./setup-docker-permissions.sh`)
3. Valid `.env` configuration file
4. No manual changes to container names or labels

## Basic Usage

### Deploy a New Version

```bash
# Standard deployment
./scripts/zero-downtime-deploy.sh

# The script will:
# 1. Detect current deployment (blue/green)
# 2. Build new images for the opposite color
# 3. Start new containers alongside existing ones
# 4. Run health checks on new deployment
# 5. Monitor for stability
# 6. Switch traffic to new deployment
# 7. Keep old deployment stopped for rollback
```

### Monitor Deployment Progress

In a separate terminal, run:

```bash
# Real-time monitoring
./scripts/monitor-deployment.sh
```

This shows:
- Container status for both deployments
- Resource usage (CPU/Memory)
- Recent logs
- Health status
- Transition progress

### Validate Deployment

```bash
# Validate current deployment
./scripts/validate-deployment.sh

# Validate specific deployment
./scripts/validate-deployment.sh blue
```

### Quick Health Check

```bash
# Check current deployment health
./scripts/deployment-health-check.sh

# Check specific deployment
./scripts/deployment-health-check.sh green
```

## Advanced Operations

### Rollback to Previous Version

If issues are detected after deployment:

```bash
# Automatic rollback to previous version
./scripts/zero-downtime-deploy.sh --rollback
```

### Manual Cleanup

After confirming the new deployment is stable:

```bash
# Clean up old blue deployment
./scripts/zero-downtime-deploy.sh --cleanup blue

# Clean up old green deployment
./scripts/zero-downtime-deploy.sh --cleanup green
```

### Check Deployment Status

```bash
# Show current active deployment
./scripts/zero-downtime-deploy.sh --status
```

## How It Works

### 1. Blue-Green Strategy

The system maintains two complete environments:
- **Blue**: One version of the application
- **Green**: Another version of the application

Only one environment is active at a time.

### 2. Deployment Process

```
1. Current State: Blue (Active) → Green (Preparing)
2. Build: Create new images tagged for Green
3. Launch: Start Green containers with unique names
4. Health Check: Verify Green deployment is healthy
5. Monitor: Watch for errors/issues for 30 seconds
6. Switch: Stop Blue trading, activate Green trading
7. Standby: Blue remains available for rollback
```

### 3. Container Naming

Containers are named with deployment color suffix:
- `ai-trading-bot-blue` / `ai-trading-bot-green`
- `bluefin-service-blue` / `bluefin-service-green`
- `mcp-memory-blue` / `mcp-memory-green`
- etc.

### 4. State Preservation

The system preserves:
- Open trading positions (gracefully closed before switch)
- Historical data and logs
- Memory/learning data
- Configuration state

## Safety Features

### Graceful Position Closure

Before switching deployments, the system:
1. Stops accepting new trades
2. Closes all open positions at market
3. Cancels pending orders
4. Saves state for recovery

### Health Checks

Multiple health check layers:
- Container running status
- Application health endpoints
- Resource usage thresholds
- Log error detection
- Network connectivity

### Automatic Rollback

Triggers on:
- Health check failures
- High error rates in logs
- Container crashes/restarts
- Network connectivity issues

### Monitoring Period

After switching:
- 30-second grace period monitoring
- Error rate tracking
- Performance validation
- Final health assessment

## Troubleshooting

### Deployment Fails to Start

```bash
# Check Docker logs
docker-compose logs --tail=50

# Verify permissions
./setup-docker-permissions.sh

# Check disk space
df -h
```

### Health Checks Fail

```bash
# Check specific service
docker logs ai-trading-bot-green

# Test service endpoints
curl http://localhost:8765/health  # MCP Memory
curl http://localhost:8000/health  # Dashboard
```

### Rollback Issues

```bash
# Force cleanup and restart
docker-compose down
docker-compose up -d

# Manual position closure
docker exec ai-trading-bot-blue python -m bot.utils.graceful_shutdown
```

### Both Deployments Running

This can happen if the switch process is interrupted:

```bash
# Check status
./scripts/zero-downtime-deploy.sh --status

# Stop one deployment manually
docker-compose -f docker-compose.yml -f docker-compose.blue.yml stop
```

## Best Practices

1. **Test in Paper Trading Mode First**
   - Set `SYSTEM__DRY_RUN=true` in `.env`
   - Verify deployment process works correctly

2. **Deploy During Low Activity**
   - Choose times with lower market volatility
   - Avoid deploying during news events

3. **Monitor After Deployment**
   - Watch for at least 15 minutes after switch
   - Check trading behavior is normal
   - Verify all integrations working

4. **Keep Backups**
   - Backup data directory before major updates
   - Export trading history
   - Save configuration files

5. **Clean Up Regularly**
   - Remove old deployments after stability confirmed
   - Clean up old Docker images
   - Archive old logs

## Integration with CI/CD

The deployment script can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
deploy:
  steps:
    - name: Deploy to Production
      run: |
        ssh user@server 'cd /app && ./scripts/zero-downtime-deploy.sh'

    - name: Validate Deployment
      run: |
        ssh user@server 'cd /app && ./scripts/validate-deployment.sh'
```

## Configuration

### Environment Variables

Key environment variables for deployment:

```bash
# Deployment behavior
DEPLOYMENT_LOG_LEVEL=INFO
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=5
MIGRATION_GRACE_PERIOD=30
ROLLBACK_TIMEOUT=300
```

### Docker Labels

The system uses Docker labels for deployment tracking:

```yaml
labels:
  - "deployment.color=${color}"
  - "deployment.service=${service}"
  - "deployment.version=${version}"
  - "deployment.timestamp=${timestamp}"
```

## Recovery Procedures

### From Failed Deployment

1. Run rollback: `./scripts/zero-downtime-deploy.sh --rollback`
2. Check logs: `docker-compose logs --tail=100`
3. Fix issues in code/config
4. Retry deployment

### From System Crash

1. Check which deployment was active: `cat .current_deployment`
2. Start that deployment: `docker-compose up -d`
3. Verify health: `./scripts/deployment-health-check.sh`
4. Resume normal operations

### From Data Corruption

1. Stop all containers: `docker-compose down`
2. Restore data from backups
3. Start fresh deployment
4. Verify data integrity

## Conclusion

The zero-downtime deployment system ensures continuous trading operations while allowing safe updates. Always test in paper trading mode first and monitor carefully during transitions.
