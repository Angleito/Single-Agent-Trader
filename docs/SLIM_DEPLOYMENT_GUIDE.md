# Slim Trading Bot Deployment Guide

This guide provides comprehensive instructions for deploying the AI Trading Bot in a memory-optimized "slim" configuration, ideal for production environments with resource constraints.

## 🎯 Overview

The slim deployment configuration reduces memory usage by:
- Using Alpine Linux base image (3.12-alpine)
- Removing unnecessary dependencies
- Implementing memory optimization monitoring
- Using multi-stage Docker builds
- Optimizing Python environment

## 📊 Memory Comparison

| Configuration | Memory Usage | Container Size | Dependencies |
|---------------|-------------|----------------|--------------|
| Full Deployment | ~1.2GB RAM | ~800MB | All features |
| Slim Deployment | ~512MB RAM | ~350MB | Core features only |

## 🚀 Quick Start

### 1. Build Slim Image

```bash
# Using Makefile (recommended)
make build-slim

# Or using Docker directly
docker build \
  --file Dockerfile.slim \
  --tag ai-trading-bot-slim:latest \
  --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  .
```

### 2. Configure Environment

```bash
# Create environment file
make env-example
cp .env.slim.example .env.slim
```

Edit `.env.slim`:
```env
# Memory Optimization
ENABLE_MEMORY_OPTIMIZATION=true
MAX_MEMORY_MB=450

# Trading Configuration
EXCHANGE__EXCHANGE_TYPE=coinbase
TRADING_MODE=paper
DRY_RUN=true

# Required API Keys
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here
```

### 3. Deploy

```bash
# Start slim deployment
make run-slim

# Check status
make memory-report
make logs-slim
```

## 📁 File Structure

```
├── Dockerfile.slim              # Optimized Alpine Dockerfile
├── docker-compose.slim.yml      # Slim deployment configuration
├── requirements-minimal.txt     # Minimal Python dependencies
├── healthcheck-slim.sh          # Lightweight health check
├── Makefile.slim                # Deployment automation
└── bot/memory_optimizer.py      # Memory monitoring module
```

## 🔧 Configuration Details

### Core Dependencies (Minimal)

```txt
# Essential trading components
pandas==2.2.0
numpy==1.26.0
coinbase-advanced-py==1.7.0
websockets>=15.0.0
aiohttp>=3.12.0

# Configuration
pydantic==2.9.0
python-dotenv==1.0.0

# Functional programming (lightweight)
returns==0.22.0
toolz==0.12.0

# Basic monitoring
psutil==6.1.0
prometheus-client>=0.22.0
```

### Memory Optimization Features

The slim deployment includes automatic memory optimization:

```python
# Memory monitoring every 20 iterations
if self._memory_optimizer and loop_count % 20 == 0:
    self._memory_optimizer.check_memory_threshold()
```

### Docker Configuration

**Multi-stage build for minimal image size:**

```dockerfile
# Builder stage - includes build tools
FROM python:3.12-alpine AS builder
RUN apk add --no-cache gcc g++ make musl-dev python3-dev
# ... install dependencies

# Production stage - runtime only
FROM python:3.12-alpine AS production
RUN apk add --no-cache ca-certificates libssl1.1 openblas
# ... copy artifacts
```

**Memory constraints:**
```yaml
deploy:
  resources:
    limits:
      memory: 512M
      cpus: '1.0'
    reservations:
      memory: 256M
      cpus: '0.5'
```

## 🎛️ Available Commands

### Build Commands
```bash
make build-slim          # Build slim image (no cache)
make build-slim-cache    # Build with cache for faster development
```

### Deployment Commands
```bash
make run-slim           # Start the slim trading bot
make stop-slim          # Stop the slim trading bot
make restart-slim       # Restart the slim trading bot
```

### Monitoring Commands
```bash
make memory-report      # Show current memory usage
make size-report        # Compare image sizes
make logs-slim          # View container logs
make health-check       # Test container health
```

### Development Commands
```bash
make shell-slim         # Open shell in container
make inspect-slim       # Inspect Docker image
make test-memory        # Test memory optimization
```

### Cleanup Commands
```bash
make clean-slim         # Remove slim containers and images
make prune-all          # Remove all unused Docker resources
```

## 📈 Memory Optimization

### Automatic Memory Management

The bot includes built-in memory optimization:

```python
from bot.memory_optimizer import get_memory_optimizer

# Initialize memory optimizer
memory_optimizer = get_memory_optimizer()

# Check memory usage
memory_report = memory_optimizer.get_memory_report()
print(memory_report)

# Force optimization
memory_optimizer.optimize_memory(force=True)
```

### Monitoring Features

- **Real-time tracking**: Memory usage monitored every trading cycle
- **Automatic optimization**: Garbage collection triggered at configurable thresholds
- **Detailed reporting**: Memory statistics logged periodically
- **Leak detection**: Identifies functions causing memory growth

### Environment Variables

```env
# Memory optimization settings
ENABLE_MEMORY_OPTIMIZATION=true
MAX_MEMORY_MB=450
PYTHONOPTIMIZE=2
MALLOC_TRIM_THRESHOLD_=100000
```

## 🔍 Monitoring & Debugging

### Memory Reports

```bash
# Real-time memory usage
make memory-report

# Detailed memory analysis
docker exec ai-trading-bot-slim python -c "
from bot.memory_optimizer import get_memory_optimizer
print(get_memory_optimizer().get_memory_report())
"
```

### Health Checks

```bash
# Container health status
make health-check

# Application health endpoint
curl http://localhost:8080/health
```

### Log Analysis

```bash
# Follow logs
make logs-slim

# Memory-specific logs
docker logs ai-trading-bot-slim 2>&1 | grep -i memory
```

## 🚨 Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Check current usage
make memory-report

# Reduce memory limit
export MAX_MEMORY_MB=400
make restart-slim
```

**2. Container Won't Start**
```bash
# Check container logs
make logs-slim

# Inspect image
make inspect-slim

# Test health check
make health-check
```

**3. Performance Issues**
```bash
# Monitor resource usage
docker stats ai-trading-bot-slim

# Check memory optimization
make test-memory
```

### Memory Tuning

**For very constrained environments:**
```env
MAX_MEMORY_MB=300
PYTHONOPTIMIZE=2
MALLOC_TRIM_THRESHOLD_=50000
```

**For better performance:**
```env
MAX_MEMORY_MB=600
ENABLE_MEMORY_OPTIMIZATION=false
```

## 🔄 Deployment Strategies

### Production Deployment

```bash
# Build for production
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
VCS_REF=$(git rev-parse --short HEAD) \
VERSION=1.0.0 \
make build-slim

# Deploy with constraints
docker run -d \
  --name trading-bot-prod \
  --memory=512m \
  --cpus=1.0 \
  --restart=unless-stopped \
  --env-file .env.slim \
  ai-trading-bot-slim:latest
```

### Development Deployment

```bash
# Quick development build
make build-slim-cache
make run-slim

# Development with shell access
make shell-slim
```

### Scaling Considerations

**Horizontal Scaling:**
- Deploy multiple instances with different symbols
- Use separate containers for different exchanges
- Implement load balancing for API calls

**Vertical Scaling:**
- Increase memory limits for more symbols
- Add CPU resources for faster processing
- Enable additional features as needed

## 📋 Feature Comparison

| Feature | Full Deployment | Slim Deployment |
|---------|----------------|-----------------|
| Core Trading | ✅ | ✅ |
| Technical Indicators | ✅ | ✅ |
| Risk Management | ✅ | ✅ |
| Paper Trading | ✅ | ✅ |
| Multiple Exchanges | ✅ | ✅ |
| Memory Optimization | ⚠️ Basic | ✅ Advanced |
| AI/LLM Features | ✅ Full | ⚠️ Optional |
| Market Making | ✅ | ⚠️ Limited |
| Dashboard | ✅ | ⚠️ Basic |
| Advanced Analytics | ✅ | ❌ |
| Development Tools | ✅ | ❌ |

## 🔐 Security

The slim deployment includes security hardening:

```yaml
# Security constraints
read_only: false  # Set to true if possible
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - SETUID
  - SETGID
```

**Non-root execution:**
- Runs as user `botuser` (UID 1000)
- Minimal system privileges
- No unnecessary capabilities

## 📝 Migration Guide

### From Full to Slim Deployment

1. **Export configuration:**
   ```bash
   docker cp trading-bot:/app/config ./backup-config
   ```

2. **Build slim image:**
   ```bash
   make build-slim
   ```

3. **Update environment:**
   ```bash
   # Remove AI-specific variables if not needed
   unset OPENAI_API_KEY
   # Add memory optimization
   export ENABLE_MEMORY_OPTIMIZATION=true
   ```

4. **Deploy and test:**
   ```bash
   make run-slim
   make health-check
   ```

### Configuration Migration

**Full deployment config:**
```yaml
services:
  trading-bot:
    image: ai-trading-bot:latest
    memory: 2G
    environment:
      - ENABLE_ALL_FEATURES=true
```

**Slim deployment config:**
```yaml
services:
  trading-bot-slim:
    image: ai-trading-bot-slim:latest
    memory: 512M
    environment:
      - ENABLE_MEMORY_OPTIMIZATION=true
      - MAX_MEMORY_MB=450
```

## 📚 Additional Resources

- [Docker Optimization Guide](../Performance_Optimization.md)
- [Memory Profiling Tools](../monitoring/MEMORY_PROFILING.md)
- [Production Deployment Checklist](../deployment/PRODUCTION_CHECKLIST.md)
- [Troubleshooting Guide](../troubleshooting/COMMON_ISSUES.md)

## 🤝 Contributing

To contribute to the slim deployment:

1. Test changes in slim environment
2. Verify memory usage improvements
3. Update documentation
4. Submit pull request with memory benchmarks

## 📞 Support

For slim deployment issues:
- Check the troubleshooting section above
- Review container logs: `make logs-slim`
- Monitor memory usage: `make memory-report`
- Open an issue with memory statistics and configuration
