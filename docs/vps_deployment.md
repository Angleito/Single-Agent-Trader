# VPS Deployment Guide

This guide explains how to deploy the AI Trading Bot to a VPS (Virtual Private Server) with support for both Coinbase and Bluefin exchanges.

## Prerequisites

### VPS Requirements
- Ubuntu 20.04+ or Debian 11+
- Minimum 2GB RAM (4GB recommended)
- 20GB storage
- Docker and Docker Compose installed
- SSH access with key authentication

### Local Requirements
- Docker installed locally for building images
- SSH key configured for VPS access
- Git repository cloned locally

## Quick Start

### 1. Prepare VPS

```bash
# Connect to your VPS
ssh ubuntu@your.vps.ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Logout and login again for docker group to take effect
exit
```

### 2. Configure Environment

Choose your exchange and create appropriate `.env` file:

**For Coinbase:**
```bash
cp .env.example .env
# Edit .env with your Coinbase credentials
```

**For Bluefin:**
```bash
cp .env.bluefin.example .env.bluefin
# Edit .env.bluefin with your Sui wallet private key
```

### 3. Deploy with Script

```bash
# Deploy Coinbase version
./scripts/deploy-vps.sh --host your.vps.ip --exchange coinbase

# Deploy Bluefin version
./scripts/deploy-vps.sh --host your.vps.ip --exchange bluefin

# With monitoring
./scripts/deploy-vps.sh --host your.vps.ip --exchange bluefin --with-monitoring

# With systemd auto-start
./scripts/deploy-vps.sh --host your.vps.ip --exchange coinbase --with-systemd
```

## Manual Deployment

If you prefer manual deployment:

### 1. Build Images Locally

```bash
# For Coinbase
docker build --build-arg EXCHANGE_TYPE=coinbase -t ai-trading-bot:coinbase-latest .

# For Bluefin
docker build --build-arg EXCHANGE_TYPE=bluefin -t ai-trading-bot:bluefin-latest .
```

### 2. Save and Transfer Images

```bash
# Save image
docker save ai-trading-bot:bluefin-latest | gzip > ai-trading-bot-bluefin.tar.gz

# Transfer to VPS
scp ai-trading-bot-bluefin.tar.gz ubuntu@your.vps.ip:/tmp/

# Load on VPS
ssh ubuntu@your.vps.ip
docker load < /tmp/ai-trading-bot-bluefin.tar.gz
```

### 3. Setup on VPS

```bash
# Create directories
mkdir -p ~/ai-trading-bot/{config,logs,data}
cd ~/ai-trading-bot

# Copy docker-compose file
# For Bluefin, use docker-compose.bluefin.yml
scp docker-compose.bluefin.yml ubuntu@your.vps.ip:~/ai-trading-bot/docker-compose.yml

# Copy .env file
scp .env.bluefin ubuntu@your.vps.ip:~/ai-trading-bot/.env

# Start services
docker compose up -d
```

## Exchange-Specific Configuration

### Coinbase Deployment

```yaml
# docker-compose.yml excerpt
environment:
  - EXCHANGE__EXCHANGE_TYPE=coinbase
  - EXCHANGE__CDP_API_KEY_NAME=${CDP_API_KEY_NAME}
  - EXCHANGE__CDP_PRIVATE_KEY=${CDP_PRIVATE_KEY}
```

### Bluefin Deployment

```yaml
# docker-compose.bluefin.yml excerpt
environment:
  - EXCHANGE__EXCHANGE_TYPE=bluefin
  - EXCHANGE__BLUEFIN_PRIVATE_KEY=${BLUEFIN_PRIVATE_KEY}
  - EXCHANGE__BLUEFIN_NETWORK=mainnet
```

## Monitoring and Maintenance

### View Logs

```bash
# All logs
docker compose logs -f

# Bot logs only
docker compose logs -f ai-trading-bot

# Last 100 lines
docker compose logs --tail=100 ai-trading-bot
```

### Check Status

```bash
# Service status
docker compose ps

# Resource usage
docker stats

# Bot health
docker compose exec ai-trading-bot /app/healthcheck.sh
```

### Update Bot

```bash
# Pull latest changes
git pull

# Rebuild and redeploy
./scripts/deploy-vps.sh --host your.vps.ip --exchange bluefin
```

### Stop/Start Services

```bash
# Stop
docker compose down

# Start
docker compose up -d

# Restart
docker compose restart ai-trading-bot
```

## Security Best Practices

### 1. Firewall Configuration

```bash
# Allow only SSH and monitoring ports
sudo ufw allow 22/tcp
sudo ufw allow 3000/tcp  # Grafana (if monitoring enabled)
sudo ufw allow 9090/tcp  # Prometheus (if monitoring enabled)
sudo ufw enable
```

### 2. SSH Hardening

```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
# Set: PubkeyAuthentication yes
sudo systemctl restart sshd
```

### 3. Environment Variables

- Never commit `.env` files to git
- Use strong, unique API keys
- Rotate keys regularly
- Set restrictive file permissions:
  ```bash
  chmod 600 .env
  ```

### 4. Docker Security

```bash
# Run containers as non-root (already configured)
# Regular updates
docker system prune -a  # Clean unused images
docker compose pull     # Update base images
```

## Monitoring Setup (Optional)

The deployment includes optional Prometheus and Grafana monitoring:

```bash
# Deploy with monitoring
./scripts/deploy-vps.sh --host your.vps.ip --with-monitoring

# Access Grafana
http://your.vps.ip:3000
# Default: admin/admin (change immediately!)

# Access Prometheus
http://your.vps.ip:9090
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs ai-trading-bot

# Check environment
docker compose config

# Verify image
docker images
```

### Connection Issues

```bash
# Test Coinbase connection
docker compose exec ai-trading-bot python -m bot.exchange.coinbase

# Test Bluefin connection
docker compose exec ai-trading-bot python scripts/test_bluefin_connection.py
```

### Memory Issues

```bash
# Check memory usage
free -h
docker stats

# Adjust Docker memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

## Backup and Recovery

### Backup Data

```bash
# Backup trading data and logs
tar -czf backup-$(date +%Y%m%d).tar.gz logs/ data/

# Download backup
scp ubuntu@your.vps.ip:~/ai-trading-bot/backup-*.tar.gz ./backups/
```

### Restore Data

```bash
# Upload backup
scp backup-20240115.tar.gz ubuntu@your.vps.ip:/tmp/

# Restore on VPS
cd ~/ai-trading-bot
tar -xzf /tmp/backup-20240115.tar.gz
```

## Production Checklist

Before going live:

- [ ] VPS secured with firewall
- [ ] SSH key-only authentication
- [ ] Environment variables configured
- [ ] Dry run mode tested
- [ ] Monitoring setup (optional)
- [ ] Backup strategy in place
- [ ] Emergency shutdown procedure documented
- [ ] API rate limits configured
- [ ] Risk parameters reviewed
- [ ] Logs rotation configured

## Emergency Shutdown

If you need to stop trading immediately:

```bash
# Quick stop
ssh ubuntu@your.vps.ip 'cd ~/ai-trading-bot && docker compose down'

# Or use emergency script
ssh ubuntu@your.vps.ip 'docker stop ai-trading-bot'
```

Remember: Always test in dry-run mode first before enabling live trading!