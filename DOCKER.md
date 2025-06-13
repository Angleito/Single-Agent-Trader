# AI Trading Bot - Docker Setup

This document provides comprehensive instructions for running the AI Trading Bot using Docker.

## Quick Start

1. **Build the Docker image:**
   ```bash
   ./scripts/docker-build.sh
   ```

2. **Set up environment:**
   ```bash
   cp example.env .env
   # Edit .env with your API keys
   ```

3. **Run in dry-run mode (safe):**
   ```bash
   ./scripts/docker-run.sh
   ```

## Docker Files Overview

### Core Files
- **`Dockerfile`** - Multi-stage build with security best practices
- **`docker-compose.yml`** - Complete service orchestration
- **`.dockerignore`** - Optimized build context

### Helper Scripts
- **`scripts/docker-build.sh`** - Build script with tagging and optimization
- **`scripts/docker-run.sh`** - Run script with environment handling
- **`scripts/docker-logs.sh`** - Log monitoring and management

## Building the Image

### Basic Build
```bash
./scripts/docker-build.sh
```

### Advanced Build Options
```bash
# Build with specific tag
./scripts/docker-build.sh -t v1.0.0

# Clean build (no cache)
./scripts/docker-build.sh -c

# Build development image
./scripts/docker-build.sh -d

# Build and push to registry
./scripts/docker-build.sh -t v1.0.0 -r your-registry.com -p
```

## Running the Container

### Safe Mode (Dry Run)
```bash
# Default dry-run mode
./scripts/docker-run.sh

# With specific symbol
./scripts/docker-run.sh --symbol ETH-USD

# Interactive mode
./scripts/docker-run.sh -i
```

### Development Mode
```bash
# Run in development mode
./scripts/docker-run.sh --dev -i

# With port mapping for web interface
./scripts/docker-run.sh --dev -p 8080
```

### Live Trading (DANGEROUS!)
```bash
# Enable live trading (requires confirmation)
./scripts/docker-run.sh --live

# Live trading with specific parameters
./scripts/docker-run.sh --live --symbol BTC-USD --interval 5m
```

### Other Commands
```bash
# Run backtest
./scripts/docker-run.sh backtest --symbol BTC-USD

# Initialize configuration
./scripts/docker-run.sh init

# Open shell in container
./scripts/docker-run.sh shell
```

## Using Docker Compose

### Basic Services
```bash
# Run main trading bot
docker-compose up ai-trading-bot

# Run in background
docker-compose up -d ai-trading-bot
```

### Development Setup
```bash
# Run development profile
docker-compose --profile dev up

# With file watching
docker-compose --profile dev up ai-trading-bot-dev
```

### Full Stack (with monitoring)
```bash
# Run all services including monitoring
docker-compose --profile full up -d

# Access Grafana dashboard at http://localhost:3000
# Access Prometheus at http://localhost:9090
```

### Available Profiles
- **`dev`** - Development environment
- **`cache`** - Redis caching service
- **`database`** - PostgreSQL database
- **`monitoring`** - Prometheus + Grafana
- **`full`** - All services

## Log Management

### Real-time Monitoring
```bash
# Follow logs in real-time
./scripts/docker-logs.sh -f

# Show last 50 lines
./scripts/docker-logs.sh -t 50

# Show logs from last hour
./scripts/docker-logs.sh --since 1h
```

### Filtering Logs
```bash
# Show only errors
./scripts/docker-logs.sh --errors

# Show only warnings
./scripts/docker-logs.sh --warnings

# Filter by pattern
./scripts/docker-logs.sh --grep "trade"

# Multiple containers
./scripts/docker-logs.sh all
```

### Log Export and Cleanup
```bash
# Export logs to file
./scripts/docker-logs.sh --export trading.log

# Show log statistics
./scripts/docker-logs.sh --stats

# Clean old logs
./scripts/docker-logs.sh --clean
```

## Environment Configuration

### Required Environment Variables
```env
# ─── OPENAI / LLM ─────────────────────────────────────────────
LLM__OPENAI_API_KEY=sk-...

# ─── COINBASE – CHOOSE ONE AUTH SCHEME ───────────────────────
# Legacy Advanced-Trade keys
# EXCHANGE__CB_API_KEY=...
# EXCHANGE__CB_API_SECRET=...
# EXCHANGE__CB_PASSPHRASE=...

# OR CDP keys (recommended going forward)
# EXCHANGE__CDP_API_KEY_NAME=orgs/.../apiKeys/...
# EXCHANGE__CDP_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"

# ─── BOT SETTINGS ────────────────────────────────────────────
SYMBOL=BTC-USD
DRY_RUN=true
```

### Docker-specific Variables
```env
# Resource limits
MEMORY_LIMIT=1g
CPU_LIMIT=0.5

# Logging
LOG_LEVEL=INFO

# Database (if using full profile)
DB_PASSWORD=secure_password
```

## Security Best Practices

### Container Security
- Runs as non-root user (`botuser`)
- Read-only root filesystem where possible
- Resource limits to prevent abuse
- Health checks for monitoring

### Environment Security
- Environment files not included in image
- Secrets managed through Docker secrets (production)
- Network isolation with custom bridge network

### Data Protection
- Logs and data stored in named volumes
- Configuration mounted read-only
- API keys never logged or exposed

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check container status
docker ps -a

# Check logs
./scripts/docker-logs.sh --errors

# Verify environment file
cat .env | grep -E "API_KEY|SECRET"
```

**Permission errors:**
```bash
# Fix directory permissions
sudo chown -R $(id -u):$(id -g) logs data config

# Check container user
docker exec ai-trading-bot whoami
```

**Network issues:**
```bash
# Recreate network
docker network rm trading-network
docker network create trading-network

# Check network connectivity
docker exec ai-trading-bot ping google.com
```

**Resource issues:**
```bash
# Check resource usage
docker stats ai-trading-bot

# Clean up unused resources
docker system prune -f
```

### Health Checks

The container includes health checks that verify:
- Python module imports successfully
- Application responds to health pings
- No critical errors in recent logs

Check health status:
```bash
docker inspect ai-trading-bot --format='{{.State.Health.Status}}'
```

### Performance Monitoring

Monitor container performance:
```bash
# Real-time stats
docker stats ai-trading-bot

# Detailed logs with timestamps
./scripts/docker-logs.sh --timestamps --details

# Resource usage over time
./scripts/docker-logs.sh --stats
```

## Production Deployment

### Registry Setup
```bash
# Tag for registry
./scripts/docker-build.sh -t v1.0.0 -r your-registry.com

# Push to registry
./scripts/docker-build.sh -t v1.0.0 -r your-registry.com -p
```

### Production Environment
```bash
# Create production environment file
cp example.env .env.prod

# Run with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Monitoring Setup
```bash
# Enable full monitoring stack
docker-compose --profile full up -d

# Configure alerts in Grafana
# Set up log aggregation
# Monitor health endpoints
```

## Advanced Configuration

### Custom Volumes
```bash
# Mount custom configuration
./scripts/docker-run.sh -v /host/config:/app/config:ro

# Mount external data source
./scripts/docker-run.sh -v /data/market-data:/app/external-data:ro
```

### Network Configuration
```bash
# Use host networking (not recommended)
docker run --network host ai-trading-bot

# Connect to external network
docker network connect external-network ai-trading-bot
```

### Multi-container Setup
```bash
# Scale services
docker-compose up --scale ai-trading-bot=3

# Load balancing with nginx
docker-compose -f docker-compose.yml -f docker-compose.nginx.yml up
```

## Support

For issues with Docker setup:
1. Check this documentation
2. Review container logs with `./scripts/docker-logs.sh --errors`
3. Verify environment configuration
4. Check Docker daemon status
5. Review container health status

For trading bot issues:
- See main README.md
- Check API connectivity
- Verify market data access
- Review trading parameters