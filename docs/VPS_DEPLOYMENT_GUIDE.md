# VPS Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the AI Trading Bot on a VPS (Virtual Private Server) with optimizations for Bluefin balance functionality and regional restrictions bypass.

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04 LTS or later (recommended)
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 50GB SSD
- **CPU**: Minimum 2 cores, Recommended 4 cores+
- **Network**: Stable internet connection with low latency

### Required Software

- Docker 20.10 or later
- Docker Compose 2.0 or later
- Git
- curl
- ufw (Uncomplicated Firewall)

### API Keys and Configuration

- Bluefin private key (64-character hex string)
- OpenAI API key
- Bluefin service API key
- Optional: Proxy configuration for regional restrictions

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/ai-trading-bot.git
cd ai-trading-bot
```

### 2. Configure Environment

```bash
# Copy VPS environment template
cp .env.vps.template .env

# Edit configuration (see Configuration section below)
nano .env
```

### 3. Deploy

```bash
# Make deployment script executable
chmod +x scripts/vps-deploy.sh

# Run deployment
./scripts/vps-deploy.sh
```

### 4. Verify

```bash
# Check service status
docker-compose -f docker-compose.vps.yml ps

# Check logs
docker-compose -f docker-compose.vps.yml logs -f
```

## Detailed Deployment

### Step 1: VPS Setup

#### 1.1 Update System

```bash
# Update package lists
apt-get update && apt-get upgrade -y

# Install required packages
apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    htop \
    iotop \
    net-tools \
    ufw \
    fail2ban \
    ca-certificates \
    gnupg \
    lsb-release
```

#### 1.2 Install Docker

```bash
# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Enable Docker service
systemctl enable docker
systemctl start docker

# Add user to docker group (optional)
usermod -aG docker $USER
```

#### 1.3 Install Docker Compose

```bash
# Download Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

### Step 2: Security Configuration

#### 2.1 Configure Firewall

```bash
# Enable UFW
ufw --force enable

# Set default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (adjust port as needed)
ufw allow 22/tcp

# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow monitoring dashboard (optional)
ufw allow 8000/tcp

# Check status
ufw status
```

#### 2.2 Configure Fail2Ban

```bash
# Install and configure fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# Create custom jail for SSH
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

# Restart fail2ban
systemctl restart fail2ban
```

### Step 3: Project Setup

#### 3.1 Clone Repository

```bash
# Clone to /opt/trading-bot
cd /opt
git clone https://github.com/your-org/ai-trading-bot.git trading-bot
cd trading-bot

# Set proper ownership
chown -R 1000:1000 /opt/trading-bot
```

#### 3.2 Configure Environment

```bash
# Copy VPS environment template
cp .env.vps.template .env

# Edit configuration
nano .env
```

**Critical Configuration Items:**

```bash
# Trading Configuration
SYSTEM__DRY_RUN=true  # Set to false for live trading
EXCHANGE__BLUEFIN_PRIVATE_KEY=0x1234...  # Your Bluefin private key
BLUEFIN_SERVICE_API_KEY=your-api-key-here

# API Keys
LLM__OPENAI_API_KEY=sk-your-openai-key

# Geographic Settings
GEOGRAPHIC_REGION=US  # Your VPS region
VPS_DEPLOYMENT=true

# Monitoring
MONITORING__ENABLED=true
ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook
```

### Step 4: Regional Restrictions Configuration

#### 4.1 Without Proxy

If your VPS is in a supported region:

```bash
# In .env file
PROXY_ENABLED=false
GEOGRAPHIC_REGION=US  # Set to your VPS region
```

#### 4.2 With Proxy

If your VPS is in a restricted region:

```bash
# In .env file
PROXY_ENABLED=true
PROXY_HOST=your-proxy-host.com
PROXY_PORT=1080
PROXY_USER=username  # Optional
PROXY_PASS=password  # Optional
```

#### 4.3 With VPN

If using a VPN service:

```bash
# Install WireGuard
apt-get install -y wireguard

# Configure WireGuard
nano /etc/wireguard/wg0.conf

# Example configuration:
[Interface]
PrivateKey = YOUR_PRIVATE_KEY
Address = 10.0.0.2/32
DNS = 8.8.8.8

[Peer]
PublicKey = YOUR_PEER_PUBLIC_KEY
Endpoint = YOUR_VPN_ENDPOINT:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25

# Start VPN
wg-quick up wg0

# Enable on boot
systemctl enable wg-quick@wg0
```

### Step 5: Deployment

#### 5.1 Automated Deployment

```bash
# Run automated deployment script
./scripts/vps-deploy.sh

# Monitor deployment progress
tail -f /var/log/vps-deployment.log
```

#### 5.2 Manual Deployment

```bash
# Create required directories
mkdir -p /var/log/vps-trading-bot
mkdir -p /var/lib/vps-trading-bot/data
mkdir -p /etc/vps-trading-bot/config

# Set permissions
chown -R 1000:1000 /var/log/vps-*
chown -R 1000:1000 /var/lib/vps-*
chown -R 1000:1000 /etc/vps-trading-bot

# Build and start services
docker-compose -f docker-compose.vps.yml build --no-cache
docker-compose -f docker-compose.vps.yml up -d

# Wait for services to start
sleep 30

# Check service status
docker-compose -f docker-compose.vps.yml ps
```

### Step 6: Verification

#### 6.1 Service Health

```bash
# Check all services
docker-compose -f docker-compose.vps.yml ps

# Check individual service health
curl -f http://localhost:8081/health  # Bluefin service
curl -f http://localhost:8765/health  # MCP memory
curl -f http://localhost:8000/health  # Monitoring dashboard

# Run comprehensive health check
/usr/local/bin/vps-health-check.sh
```

#### 6.2 Logs Analysis

```bash
# View real-time logs
docker-compose -f docker-compose.vps.yml logs -f

# Check for errors
docker-compose -f docker-compose.vps.yml logs | grep -i "error\|exception\|failed"

# Check trading activity
docker-compose -f docker-compose.vps.yml logs ai-trading-bot | grep -i "trade\|position\|balance"
```

#### 6.3 Connectivity Tests

```bash
# Test Bluefin API connectivity
docker-compose -f docker-compose.vps.yml exec bluefin-service curl -v https://api.bluefin.io

# Test OpenAI API connectivity
docker-compose -f docker-compose.vps.yml exec ai-trading-bot curl -v https://api.openai.com/v1/models

# Test inter-service communication
docker-compose -f docker-compose.vps.yml exec ai-trading-bot curl -f http://bluefin-service:8080/health
```

## Monitoring and Maintenance

### Monitoring Setup

#### 1. Built-in Monitoring

```bash
# Access monitoring dashboard
curl http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Real-time monitoring
docker stats --no-stream
```

#### 2. External Monitoring

```bash
# Configure Slack alerts
export ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook

# Test alerts
curl -X POST $ALERT_WEBHOOK_URL -H 'Content-type: application/json' --data '{"text":"VPS Trading Bot Alert Test"}'
```

### Maintenance Tasks

#### Daily Maintenance

```bash
# Health check
/usr/local/bin/vps-health-check.sh

# Check disk space
df -h

# Check memory usage
free -m

# Check service status
docker-compose -f docker-compose.vps.yml ps
```

#### Weekly Maintenance

```bash
# Update system packages
apt-get update && apt-get upgrade -y

# Update Docker images
docker-compose -f docker-compose.vps.yml pull
docker-compose -f docker-compose.vps.yml up -d

# Clean up old containers/images
docker system prune -f
```

#### Monthly Maintenance

```bash
# Rotate API keys (if applicable)
# Update Bluefin service API key
# Update OpenAI API key

# Review and archive old logs
find /var/log/vps-* -name "*.log" -mtime +30 -delete

# Performance review
docker stats --no-stream
htop
iotop -o
```

## Backup and Recovery

### Automated Backup

Backups are automatically configured by the deployment script:

```bash
# Manual backup
/usr/local/bin/vps-backup.sh

# Check backup status
ls -la /var/backups/vps-trading-bot/

# Verify backup integrity
tar -tzf /var/backups/vps-trading-bot/data_backup_*.tar.gz
```

### Disaster Recovery

#### 1. Full Recovery

```bash
# Stop services
docker-compose -f docker-compose.vps.yml down

# Restore from backup
cd /var/backups/vps-trading-bot/
tar -xzf data_backup_YYYYMMDD_HHMMSS.tar.gz -C /

# Restart services
docker-compose -f docker-compose.vps.yml up -d

# Verify recovery
/usr/local/bin/vps-health-check.sh
```

#### 2. Partial Recovery

```bash
# Restore specific service data
docker-compose -f docker-compose.vps.yml stop ai-trading-bot
tar -xzf data_backup_*.tar.gz var/lib/vps-trading-bot/data
docker-compose -f docker-compose.vps.yml start ai-trading-bot
```

## Performance Optimization

### Resource Optimization

#### 1. Memory Optimization

```bash
# Adjust memory limits in docker-compose.vps.yml
services:
  ai-trading-bot:
    deploy:
      resources:
        limits:
          memory: 4G  # Increase if needed
          cpus: '2.0'
```

#### 2. Network Optimization

```bash
# Optimize network settings
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_congestion_control = bbr' >> /etc/sysctl.conf
sysctl -p
```

#### 3. Storage Optimization

```bash
# Use SSD for data directories
# Consider using separate volumes for logs and data
# Implement log rotation

# Example: Move logs to separate volume
docker volume create vps-logs
# Update docker-compose.vps.yml to use named volume
```

### Application Optimization

#### 1. Trading Intervals

```bash
# Optimize trading intervals based on VPS performance
TRADING__MARKET_DATA_INTERVAL=10  # Increase if CPU limited
TRADING__POSITION_CHECK_INTERVAL=20  # Increase if network limited
```

#### 2. Logging Optimization

```bash
# Reduce logging verbosity in production
LOG_LEVEL=WARN
LLM__ENABLE_COMPLETION_LOGGING=false  # Disable if not needed
```

## Security Best Practices

### 1. API Key Management

```bash
# Rotate API keys regularly
# Use environment variables, never hardcode keys
# Implement key rotation monitoring

# Example key rotation
export OLD_API_KEY=$BLUEFIN_SERVICE_API_KEY
export BLUEFIN_SERVICE_API_KEY=new-api-key
docker-compose -f docker-compose.vps.yml up -d bluefin-service
```

### 2. Network Security

```bash
# Use VPN for additional security
# Implement IP whitelisting where possible
# Monitor for unusual network activity

# Example: Monitor connections
netstat -tuln | grep ESTABLISHED
ss -tuln
```

### 3. Container Security

```bash
# Regular security updates
docker-compose -f docker-compose.vps.yml pull
docker-compose -f docker-compose.vps.yml up -d

# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image ai-trading-bot:vps-latest
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.vps.yml logs service-name

# Check resource usage
docker stats --no-stream

# Check network connectivity
docker-compose -f docker-compose.vps.yml exec service-name ping google.com
```

#### 2. Regional Restrictions

```bash
# Check current IP location
curl -s https://ipapi.co/json/ | jq

# Test with proxy
curl --proxy your-proxy:port -s https://ipapi.co/json/ | jq

# Verify Bluefin API access
curl -v https://api.bluefin.io
```

#### 3. Balance Issues

```bash
# Check Bluefin service logs
docker-compose -f docker-compose.vps.yml logs bluefin-service | grep -i balance

# Test balance endpoint
curl http://localhost:8081/balance

# Check network connectivity to Bluefin
docker-compose -f docker-compose.vps.yml exec bluefin-service ping api.bluefin.io
```

For detailed troubleshooting, see [VPS_TROUBLESHOOTING.md](VPS_TROUBLESHOOTING.md).

## Cost Optimization

### VPS Provider Selection

**Recommended VPS Providers:**
- **DigitalOcean**: Good performance, multiple regions
- **Linode**: Reliable, good support
- **Vultr**: Competitive pricing, many locations
- **AWS Lightsail**: Integrated with AWS ecosystem
- **Hetzner**: Cost-effective, European locations

**Cost Optimization Tips:**
1. Choose the right instance size (4GB RAM, 2 CPU cores minimum)
2. Use reserved instances for long-term deployments
3. Monitor resource usage and downsize if possible
4. Use block storage for data persistence
5. Implement auto-scaling for peak trading hours

### Resource Monitoring

```bash
# Monitor resource usage
docker stats --no-stream
htop
iotop -o
df -h

# Set up alerts for resource thresholds
# CPU > 80% for 5 minutes
# Memory > 85% for 5 minutes
# Disk > 90%
```

## Conclusion

This VPS deployment configuration provides:

✅ **Optimized for Bluefin**: Specialized configuration for balance functionality
✅ **Regional Restrictions Bypass**: Proxy and VPN support
✅ **Production Ready**: Security hardening and monitoring
✅ **Automated Deployment**: One-command deployment with validation
✅ **Comprehensive Monitoring**: Health checks and alerting
✅ **Disaster Recovery**: Automated backups and recovery procedures
✅ **Performance Optimized**: Resource limits and network tuning
✅ **Troubleshooting Ready**: Detailed logs and diagnostic tools

For support or questions, refer to:
- [VPS_TROUBLESHOOTING.md](VPS_TROUBLESHOOTING.md)
- [BLUEFIN_SETUP_GUIDE.md](BLUEFIN_SETUP_GUIDE.md)
- [Operations_Manual.md](Operations_Manual.md)
