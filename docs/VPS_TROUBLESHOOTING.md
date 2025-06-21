# VPS Deployment Troubleshooting Guide

## Overview

This guide provides troubleshooting steps for VPS deployment issues with the AI Trading Bot, specifically optimized for Bluefin balance functionality and regional restrictions bypass.

## Quick Diagnostics

### 1. Check Service Status

```bash
# Check all VPS services
docker-compose -f docker-compose.vps.yml ps

# Check specific service logs
docker-compose -f docker-compose.vps.yml logs -f bluefin-service
docker-compose -f docker-compose.vps.yml logs -f ai-trading-bot
docker-compose -f docker-compose.vps.yml logs -f mcp-memory

# Check service health
docker-compose -f docker-compose.vps.yml exec bluefin-service curl -f http://localhost:8080/health
docker-compose -f docker-compose.vps.yml exec mcp-memory curl -f http://localhost:8765/health
```

### 2. Run Built-in Health Check

```bash
# Run comprehensive health check
/usr/local/bin/vps-health-check.sh

# Manual health check for specific services
docker exec bluefin-service-vps /app/vps-healthcheck.sh
docker exec ai-trading-bot-vps /app/vps-healthcheck.sh
```

### 3. Check System Resources

```bash
# Check disk usage
df -h

# Check memory usage
free -m

# Check CPU usage
top -n 1

# Check Docker resources
docker system df
docker stats --no-stream
```

## Common Issues and Solutions

### 1. Bluefin Service Issues

#### Issue: Bluefin service fails to start
**Symptoms:**
- Service exits immediately after start
- Health check fails
- Logs show connection errors

**Solutions:**
```bash
# Check environment variables
docker-compose -f docker-compose.vps.yml config

# Verify Bluefin private key format
echo $EXCHANGE__BLUEFIN_PRIVATE_KEY | grep -E '^0x[a-fA-F0-9]{64}$'

# Check network connectivity
docker-compose -f docker-compose.vps.yml exec bluefin-service curl -v https://api.bluefin.io

# Restart with debug logging
docker-compose -f docker-compose.vps.yml up -d bluefin-service
docker-compose -f docker-compose.vps.yml logs -f bluefin-service
```

#### Issue: Regional restrictions blocking Bluefin access
**Symptoms:**
- HTTP 403 errors
- "Region not supported" messages
- Connection timeouts to Bluefin API

**Solutions:**
```bash
# Check current IP geolocation
curl -s https://ipapi.co/json/ | jq

# Enable proxy if configured
docker-compose -f docker-compose.vps.yml stop bluefin-service
export PROXY_ENABLED=true
export PROXY_HOST=your-proxy-host
export PROXY_PORT=your-proxy-port
docker-compose -f docker-compose.vps.yml up -d bluefin-service

# Verify proxy is working
docker-compose -f docker-compose.vps.yml exec bluefin-service curl --proxy $PROXY_HOST:$PROXY_PORT -s https://ipapi.co/json/
```

#### Issue: Bluefin balance retrieval fails
**Symptoms:**
- Balance shows as 0 or null
- "Insufficient balance" errors
- Balance API timeouts

**Solutions:**
```bash
# Check wallet connection
docker-compose -f docker-compose.vps.yml exec bluefin-service python3 -c "
from bluefin_client_python import BluefinClient
import os
client = BluefinClient(True if os.getenv('EXCHANGE__BLUEFIN_NETWORK') == 'testnet' else False, os.getenv('EXCHANGE__BLUEFIN_PRIVATE_KEY'))
print('Balance:', client.get_public_address())
"

# Check network connectivity to Bluefin
docker-compose -f docker-compose.vps.yml exec bluefin-service nslookup api.bluefin.io
docker-compose -f docker-compose.vps.yml exec bluefin-service ping -c 3 api.bluefin.io

# Check API rate limits
docker-compose -f docker-compose.vps.yml logs bluefin-service | grep -i "rate\|limit\|429"

# Increase timeouts
export BLUEFIN_SERVICE_RATE_LIMIT=25
export CONNECTION_TIMEOUT=60
export READ_TIMEOUT=120
docker-compose -f docker-compose.vps.yml up -d bluefin-service
```

### 2. Trading Bot Issues

#### Issue: Trading bot cannot connect to Bluefin service
**Symptoms:**
- "Connection refused" errors
- Bluefin service unavailable messages
- Trading decisions not executed

**Solutions:**
```bash
# Check network connectivity between services
docker-compose -f docker-compose.vps.yml exec ai-trading-bot ping bluefin-service
docker-compose -f docker-compose.vps.yml exec ai-trading-bot curl -f http://bluefin-service:8080/health

# Check service discovery
docker network inspect vps-trading-network

# Restart services in correct order
docker-compose -f docker-compose.vps.yml stop ai-trading-bot
docker-compose -f docker-compose.vps.yml up -d bluefin-service
sleep 30
docker-compose -f docker-compose.vps.yml up -d ai-trading-bot
```

#### Issue: Trading bot high memory usage
**Symptoms:**
- OOM kills
- Slow performance
- Memory warnings in logs

**Solutions:**
```bash
# Check memory usage
docker stats ai-trading-bot-vps --no-stream

# Reduce memory footprint
export TRADING__MARKET_DATA_INTERVAL=30
export TRADING__POSITION_CHECK_INTERVAL=60
export LLM__ENABLE_COMPLETION_LOGGING=false
docker-compose -f docker-compose.vps.yml up -d ai-trading-bot

# Clear memory caches
docker-compose -f docker-compose.vps.yml exec ai-trading-bot python3 -c "
import gc
gc.collect()
"
```

### 3. Network Issues

#### Issue: VPS cannot reach external APIs
**Symptoms:**
- DNS resolution failures
- Connection timeouts
- Certificate errors

**Solutions:**
```bash
# Check DNS resolution
docker-compose -f docker-compose.vps.yml exec ai-trading-bot nslookup api.openai.com
docker-compose -f docker-compose.vps.yml exec ai-trading-bot nslookup api.bluefin.io

# Check network configuration
cat /etc/resolv.conf
ip route show

# Test external connectivity
docker-compose -f docker-compose.vps.yml exec ai-trading-bot curl -v https://api.openai.com/v1/models
docker-compose -f docker-compose.vps.yml exec bluefin-service curl -v https://api.bluefin.io
```

#### Issue: Docker network problems
**Symptoms:**
- Services cannot communicate
- Network isolation issues
- Port binding failures

**Solutions:**
```bash
# Recreate Docker network
docker-compose -f docker-compose.vps.yml down
docker network rm vps-trading-network
docker network create --driver bridge --subnet=172.20.0.0/16 vps-trading-network
docker-compose -f docker-compose.vps.yml up -d

# Check network configuration
docker network inspect vps-trading-network
docker-compose -f docker-compose.vps.yml exec ai-trading-bot ip addr show
```

### 4. Performance Issues

#### Issue: Slow response times
**Symptoms:**
- High latency
- Timeouts
- Degraded performance

**Solutions:**
```bash
# Check system load
uptime
iostat -x 1 5

# Optimize Docker configuration
echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
sysctl -p

# Increase resource limits
docker-compose -f docker-compose.vps.yml down
# Edit docker-compose.vps.yml to increase memory/CPU limits
docker-compose -f docker-compose.vps.yml up -d

# Enable performance monitoring
export MONITORING__ENABLED=true
export MONITORING__METRICS_INTERVAL=30
docker-compose -f docker-compose.vps.yml up -d
```

#### Issue: High disk I/O
**Symptoms:**
- Slow file operations
- High wait times
- Disk space filling up

**Solutions:**
```bash
# Check disk usage and I/O
df -h
iotop -o

# Clean up old logs
find /var/log/vps-* -name "*.log" -mtime +7 -delete
docker system prune -f

# Optimize logging
export LOG_LEVEL=WARN
docker-compose -f docker-compose.vps.yml up -d

# Move logs to separate volume if needed
docker volume create vps-logs
# Update docker-compose.vps.yml to use named volume
```

## Monitoring and Alerting

### 1. Set up Monitoring

```bash
# Enable comprehensive monitoring
export MONITORING__ENABLED=true
export MONITORING__METRICS_INTERVAL=60
export MONITORING__PERFORMANCE_TRACKING=true
export ALERT_ENABLED=true
export ALERT_WEBHOOK_URL=https://your-webhook-url.com/alerts

# Start monitoring dashboard
docker-compose -f docker-compose.vps.yml up -d monitoring-dashboard

# Access monitoring dashboard
curl http://localhost:8000/health
```

### 2. Set up Alerting

```bash
# Configure webhook alerts
export ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook
export ALERT_ENABLED=true

# Test alert system
curl -X POST $ALERT_WEBHOOK_URL -H 'Content-type: application/json' --data '{"text":"VPS Trading Bot Alert Test"}'
```

### 3. Log Analysis

```bash
# Analyze trading bot logs
grep -i "error\|exception\|failed" /var/log/vps-trading-bot/*.log | tail -20

# Analyze Bluefin service logs
grep -i "balance\|connection\|timeout" /var/log/vps-bluefin-service/*.log | tail -20

# Monitor real-time logs
tail -f /var/log/vps-trading-bot/*.log | grep -i "trade\|position\|balance"
```

## Regional Restrictions Bypass

### 1. Proxy Configuration

```bash
# Configure SOCKS5 proxy
export PROXY_ENABLED=true
export PROXY_HOST=your-proxy-host.com
export PROXY_PORT=1080
export PROXY_USER=username
export PROXY_PASS=password

# Test proxy connectivity
docker-compose -f docker-compose.vps.yml exec bluefin-service curl --socks5 $PROXY_HOST:$PROXY_PORT https://ipapi.co/json/
```

### 2. VPN Configuration

```bash
# Install WireGuard (if using VPN)
apt-get update && apt-get install -y wireguard

# Configure WireGuard
cat > /etc/wireguard/wg0.conf << EOF
[Interface]
PrivateKey = YOUR_PRIVATE_KEY
Address = 10.0.0.2/32
DNS = 8.8.8.8

[Peer]
PublicKey = YOUR_PEER_PUBLIC_KEY
Endpoint = YOUR_VPN_ENDPOINT:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25
EOF

# Start VPN
wg-quick up wg0

# Test connectivity
curl -s https://ipapi.co/json/ | jq
```

### 3. DNS Configuration

```bash
# Configure DNS for better routing
cat > /etc/systemd/resolved.conf << EOF
[Resolve]
DNS=8.8.8.8 1.1.1.1
FallbackDNS=8.8.4.4 1.0.0.1
EOF

systemctl restart systemd-resolved
```

## Backup and Recovery

### 1. Create Backup

```bash
# Manual backup
/usr/local/bin/vps-backup.sh

# Verify backup
ls -la /var/backups/vps-trading-bot/
```

### 2. Restore from Backup

```bash
# Stop services
docker-compose -f docker-compose.vps.yml down

# Restore data
cd /var/backups/vps-trading-bot/
tar -xzf data_backup_YYYYMMDD_HHMMSS.tar.gz -C /

# Restore configurations
tar -xzf logs_backup_YYYYMMDD_HHMMSS.tar.gz -C /

# Restart services
docker-compose -f docker-compose.vps.yml up -d
```

## Security Considerations

### 1. Check Security Configuration

```bash
# Verify firewall rules
ufw status

# Check Docker security
docker-compose -f docker-compose.vps.yml config | grep -E "user|read_only|security_opt|cap_drop"

# Verify file permissions
ls -la /var/log/vps-*
ls -la /var/lib/vps-*
```

### 2. Update Security Settings

```bash
# Update system packages
apt-get update && apt-get upgrade -y

# Update Docker images
docker-compose -f docker-compose.vps.yml pull
docker-compose -f docker-compose.vps.yml up -d

# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image ai-trading-bot:vps-latest
```

## Emergency Procedures

### 1. Emergency Stop

```bash
# Immediate stop all services
docker-compose -f docker-compose.vps.yml down

# Emergency trading halt
docker-compose -f docker-compose.vps.yml exec ai-trading-bot python3 -c "
import os
os.environ['SYSTEM__DRY_RUN'] = 'true'
print('Emergency halt activated')
"
```

### 2. Service Recovery

```bash
# Full service restart
docker-compose -f docker-compose.vps.yml down
docker system prune -f
docker-compose -f docker-compose.vps.yml build --no-cache
docker-compose -f docker-compose.vps.yml up -d

# Verify recovery
/usr/local/bin/vps-health-check.sh
```

## Getting Help

### 1. Collect Debug Information

```bash
# Create debug package
mkdir -p /tmp/vps-debug
docker-compose -f docker-compose.vps.yml config > /tmp/vps-debug/compose-config.yml
docker-compose -f docker-compose.vps.yml ps > /tmp/vps-debug/services-status.txt
docker-compose -f docker-compose.vps.yml logs --tail=100 > /tmp/vps-debug/services-logs.txt
docker system info > /tmp/vps-debug/docker-info.txt
free -m > /tmp/vps-debug/memory-usage.txt
df -h > /tmp/vps-debug/disk-usage.txt
tar -czf /tmp/vps-debug-$(date +%Y%m%d_%H%M%S).tar.gz -C /tmp vps-debug/
```

### 2. Contact Support

When contacting support, include:
- VPS provider and region
- Error messages and logs
- System specifications
- Network configuration
- Steps to reproduce the issue
- Debug package created above

## Best Practices

### 1. Regular Maintenance

```bash
# Daily health check
/usr/local/bin/vps-health-check.sh

# Weekly system update
apt-get update && apt-get upgrade -y
docker-compose -f docker-compose.vps.yml pull
docker-compose -f docker-compose.vps.yml up -d

# Monthly cleanup
docker system prune -f
find /var/log/vps-* -name "*.log" -mtime +30 -delete
```

### 2. Performance Optimization

```bash
# Monitor resource usage
docker stats --no-stream
htop

# Optimize based on usage patterns
# - Adjust resource limits in docker-compose.vps.yml
# - Tune environment variables for performance
# - Implement caching strategies
```

### 3. Security Hardening

```bash
# Regular security updates
apt-get update && apt-get upgrade -y

# Monitor for suspicious activity
grep -i "failed\|error\|unauthorized" /var/log/auth.log

# Update API keys regularly
# - Regenerate Bluefin service API key
# - Rotate OpenAI API key
# - Update webhook URLs
```

This troubleshooting guide should help you diagnose and resolve common VPS deployment issues with the AI Trading Bot.
