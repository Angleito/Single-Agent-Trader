# 🚀 1-Core VPS Deployment Guide for AI Trading Bot

This guide provides step-by-step instructions for deploying the AI Trading Bot to a 1-core VPS with real-time monitoring, optimized for memory-constrained environments.

## 📋 Prerequisites

### VPS Requirements (Minimum)
- **CPU**: 1 core (1.0 GHz or higher)
- **RAM**: 1GB (2GB recommended)
- **Storage**: 20GB SSD
- **OS**: Ubuntu 20.04 LTS or newer
- **Network**: 10 Mbps upload/download

### Local Requirements
- Docker and Docker Compose installed
- SSH access to your VPS
- Git repository cloned locally

### VPS Setup
1. **Update your VPS**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Install Docker Compose**:
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

4. **Reboot VPS**:
   ```bash
   sudo reboot
   ```

## 🔧 Configuration

### 1. Environment Setup
Copy and configure your environment file:
```bash
cp .env.example .env
# Edit .env with your actual API keys and settings
nano .env
```

**Critical Settings for 1-Core VPS**:
```env
# Memory optimization
ENABLE_MEMORY_OPTIMIZATION=true
MAX_MEMORY_MB=450
PYTHONOPTIMIZE=2

# Reduced concurrency
SYSTEM__MAX_CONCURRENT_TASKS=2
SYSTEM__THREAD_POOL_SIZE=2
FP_MAX_CONCURRENT_EFFECTS=10

# Conservative trading
RISK__MAX_CONCURRENT_TRADES=1
TRADING__ORDER_TIMEOUT_SECONDS=30

# Always start with paper trading
SYSTEM__DRY_RUN=true
```

### 2. VPS Connection
Set your VPS details:
```bash
export DEPLOY_HOST=your.vps.ip.address
export DEPLOY_USER=ubuntu  # or your VPS username
export EXCHANGE_TYPE=coinbase  # or bluefin
```

## 🚀 Deployment Process

### Option 1: Quick Deployment (Recommended)
```bash
# Deploy with monitoring enabled
./scripts/deploy-1core-vps.sh --host $DEPLOY_HOST --exchange $EXCHANGE_TYPE --enable-monitoring
```

### Option 2: Custom Deployment
```bash
# Deploy with custom memory limits
./scripts/deploy-1core-vps.sh \
  --host $DEPLOY_HOST \
  --exchange $EXCHANGE_TYPE \
  --memory-bot 384M \
  --memory-service 192M \
  --cpu-limit 0.8 \
  --enable-monitoring
```

### Option 3: Minimal Deployment (No Monitoring)
```bash
# For very low-memory VPS
./scripts/deploy-1core-vps.sh \
  --host $DEPLOY_HOST \
  --exchange $EXCHANGE_TYPE \
  --memory-bot 256M \
  --memory-service 128M \
  --disable-monitoring
```

## 📊 Real-Time Monitoring

### Start Real-Time Monitor
```bash
# SSH into your VPS
ssh $DEPLOY_USER@$DEPLOY_HOST

# Navigate to deployment directory
cd /home/$DEPLOY_USER/ai-trading-bot

# Start real-time monitoring
python3 scripts/vps-monitor.py --interval 10
```

### Monitor Dashboard Features
- **System Performance**: CPU, RAM, Disk usage with alerts
- **Container Status**: Health, restarts, resource usage
- **Trading Metrics**: Balance, P&L, positions, trades
- **Real-time Alerts**: Automatic warnings for resource issues

### Monitoring Commands
```bash
# Quick status check
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && docker compose ps'

# View real-time logs
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && docker compose logs -f ai-trading-bot'

# Check system resources
ssh $DEPLOY_USER@$DEPLOY_HOST 'free -h && df -h && top -bn1 | head -10'

# Check trading bot health
curl http://$DEPLOY_HOST:8080/health

# Check dashboard health
curl http://$DEPLOY_HOST:8000/health
```

## 🔍 Performance Monitoring

### Key Metrics to Watch

#### System Resources
- **CPU Usage**: Should stay below 85% on average
- **Memory Usage**: Should stay below 90%
- **Load Average**: Should stay below 2.0 for 1-core system
- **Disk Usage**: Should stay below 85%

#### Container Health
- **Restart Count**: Should remain at 0
- **Health Status**: Should show "healthy" or "none"
- **Memory per Container**: Should stay within limits

#### Trading Performance
- **API Response Time**: Should be under 2 seconds
- **Order Execution**: Should complete within 30 seconds
- **WebSocket Connectivity**: Should maintain stable connection

### Alert Thresholds
The monitoring system will alert you when:
- CPU usage > 85%
- Memory usage > 90%
- Disk usage > 85%
- Load average > 2.0
- Container restarts occur
- Container memory exceeds 90% of limit

## 🌐 Web Interfaces

After successful deployment, access these interfaces:

- **Trading Dashboard**: `http://$DEPLOY_HOST:8000`
  - Real-time trading data
  - Performance metrics
  - Bot configuration

- **Trading Bot API**: `http://$DEPLOY_HOST:8080`
  - Health endpoint: `/health`
  - Metrics endpoint: `/metrics`

- **Prometheus (if enabled)**: `http://$DEPLOY_HOST:9090`
  - System and application metrics
  - Custom queries and alerts

## 🚨 Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Check memory usage
ssh $DEPLOY_USER@$DEPLOY_HOST 'free -h'

# Reduce memory limits
./scripts/deploy-1core-vps.sh --memory-bot 256M --memory-service 128M

# Enable swap (emergency only)
ssh $DEPLOY_USER@$DEPLOY_HOST 'sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile'
```

#### 2. High CPU Usage
```bash
# Check CPU usage
ssh $DEPLOY_USER@$DEPLOY_HOST 'top -bn1 | head -20'

# Reduce CPU limits
./scripts/deploy-1core-vps.sh --cpu-limit 0.8

# Increase update intervals in .env
SYSTEM__UPDATE_FREQUENCY_SECONDS=60.0
```

#### 3. Container Restart Loops
```bash
# Check container logs
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && docker compose logs ai-trading-bot'

# Check system resources
ssh $DEPLOY_USER@$DEPLOY_HOST 'docker stats --no-stream'

# Restart with increased limits
./scripts/deploy-1core-vps.sh --memory-bot 512M
```

#### 4. Network Connectivity Issues
```bash
# Check container networking
ssh $DEPLOY_USER@$DEPLOY_HOST 'docker network ls && docker network inspect trading-network'

# Restart networking
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && docker compose down && docker compose up -d'
```

### Performance Optimization

#### For 512MB RAM VPS
```bash
./scripts/deploy-1core-vps.sh \
  --memory-bot 256M \
  --memory-service 128M \
  --memory-dashboard 64M \
  --disable-monitoring
```

#### For 1GB RAM VPS (Recommended)
```bash
./scripts/deploy-1core-vps.sh \
  --memory-bot 512M \
  --memory-service 256M \
  --memory-dashboard 128M \
  --enable-monitoring
```

#### For 2GB RAM VPS (Optimal)
```bash
./scripts/deploy-1core-vps.sh \
  --memory-bot 768M \
  --memory-service 384M \
  --memory-dashboard 256M \
  --enable-monitoring
```

## 📈 Scaling Recommendations

### When to Upgrade VPS

#### Upgrade to 2-Core if:
- CPU usage consistently > 90%
- Load average > 3.0
- Frequent container restarts due to resource limits
- Trading latency > 5 seconds

#### Upgrade Memory if:
- Memory usage consistently > 95%
- Frequent OOM (Out of Memory) kills
- Swap usage > 50%
- Cannot run monitoring services

### Horizontal Scaling
For high-frequency trading, consider:
- Separate VPS for each exchange
- Dedicated monitoring VPS
- Load balancer for dashboard access

## 🔐 Security Best Practices

### VPS Security
```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no

# Configure firewall
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # Dashboard
sudo ufw allow 8080/tcp  # Bot API
sudo ufw allow 9090/tcp  # Prometheus (optional)
sudo ufw enable

# Regular updates
sudo apt update && sudo apt upgrade -y
```

### Application Security
- Use strong API keys
- Enable 2FA on all accounts
- Regular backup of configuration
- Monitor access logs
- Use HTTPS in production

## 📊 Performance Benchmarks

### Expected Performance (1-Core VPS)

#### Minimum Specs (1GB RAM)
- **Order Processing**: 2-5 seconds
- **Market Data Updates**: 10-30 seconds
- **Memory Usage**: 300-600MB
- **CPU Usage**: 20-60%

#### Recommended Specs (2GB RAM)
- **Order Processing**: 1-3 seconds
- **Market Data Updates**: 5-15 seconds
- **Memory Usage**: 400-800MB
- **CPU Usage**: 15-45%

### Optimization Results
After optimization, expect:
- 40% reduction in memory usage
- 30% reduction in CPU usage
- 50% faster startup times
- 60% fewer container restarts

## 📞 Support

### Monitoring Logs
```bash
# Monitor deployment logs
tail -f /tmp/vps-monitor.log

# Monitor container logs
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && docker compose logs -f --tail=100'

# Monitor system logs
ssh $DEPLOY_USER@$DEPLOY_HOST 'sudo journalctl -f -u docker'
```

### Emergency Procedures

#### Stop All Services
```bash
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && docker compose down'
```

#### Emergency Resource Cleanup
```bash
ssh $DEPLOY_USER@$DEPLOY_HOST 'docker system prune -f && docker volume prune -f'
```

#### Backup Configuration
```bash
ssh $DEPLOY_USER@$DEPLOY_HOST 'cd /home/ubuntu/ai-trading-bot && tar -czf backup-$(date +%Y%m%d).tar.gz .env config/ data/'
```

---

## ✅ Success Checklist

- [ ] VPS meets minimum requirements
- [ ] Docker and Docker Compose installed
- [ ] SSH access configured
- [ ] Environment variables configured
- [ ] Deployment script executed successfully
- [ ] All containers running and healthy
- [ ] API endpoints responding
- [ ] Real-time monitoring active
- [ ] Trading metrics visible
- [ ] Performance within acceptable limits
- [ ] Security measures implemented
- [ ] Backup procedures established

**Remember**: Always start with `SYSTEM__DRY_RUN=true` for testing before enabling live trading!
