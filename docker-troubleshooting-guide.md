# Docker Compose Troubleshooting Guide

## Quick Diagnostics

### Check Configuration
```bash
# Validate syntax
docker-compose config --quiet

# View resolved configuration
docker-compose config

# List services
docker-compose config --services
```

### Check Service Status
```bash
# View all services
docker-compose ps

# Check specific service
docker-compose ps ai-trading-bot

# View service logs
docker-compose logs -f ai-trading-bot
```

## Common Issues and Solutions

### 1. **Permission Denied Errors**
```bash
# Fix permissions
./setup-docker-permissions.sh

# Or manually
sudo chown -R $(id -u):$(id -g) ./logs ./data
```

### 2. **Health Check Failures**
```bash
# Check individual service health
docker-compose exec ai-trading-bot /app/healthcheck.sh

# Debug health check
docker-compose exec ai-trading-bot /app/healthcheck.sh quick
```

### 3. **Network Connectivity Issues**
```bash
# Recreate network
docker-compose down
docker network rm trading-network
docker-compose up -d

# Test network connectivity
docker-compose exec ai-trading-bot ping mcp-memory
```

### 4. **Volume Mount Issues**
```bash
# Check volume permissions
ls -la ./logs ./data

# Recreate volumes
docker-compose down -v
docker volume prune
docker-compose up -d
```

### 5. **Build Failures**
```bash
# Clean build
docker-compose build --no-cache ai-trading-bot

# Build with verbose output
docker-compose build --progress=plain ai-trading-bot
```

### 6. **Dependency Startup Issues**
```bash
# Start services in order
docker-compose up -d mcp-memory mcp-omnisearch
sleep 30
docker-compose up -d bluefin-service
sleep 15
docker-compose up -d ai-trading-bot dashboard-backend
sleep 10
docker-compose up -d dashboard-frontend
```

## Service-Specific Issues

### AI Trading Bot
```bash
# Check FP runtime health
docker-compose exec ai-trading-bot /app/healthcheck.sh fp-only

# View trading logs
docker-compose logs ai-trading-bot | grep -E "(TRADE|ERROR|WARN)"
```

### Bluefin Service
```bash
# Test Bluefin connectivity
docker-compose exec bluefin-service curl -f http://localhost:8080/health

# Check Bluefin logs
docker-compose logs bluefin-service | tail -50
```

### MCP Services
```bash
# Test MCP Memory
curl -f http://localhost:8765/health

# Test MCP OmniSearch
curl -f http://localhost:8767/health
```

### Dashboard
```bash
# Test backend API
curl -f http://localhost:8000/health

# Check frontend
curl -f http://localhost:3000/
```

## Emergency Recovery

### Complete Reset
```bash
# Stop everything
docker-compose down -v

# Clean up
docker system prune -f
docker volume prune -f

# Rebuild and start
docker-compose build --no-cache
docker-compose up -d
```

### Partial Reset (Preserve Data)
```bash
# Stop services
docker-compose down

# Keep volumes, rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

## Monitoring Commands

### Resource Usage
```bash
# Monitor all services
docker stats $(docker-compose ps -q)

# Monitor specific service
docker stats ai-trading-bot
```

### Log Monitoring
```bash
# Follow all logs
docker-compose logs -f

# Follow specific service
docker-compose logs -f ai-trading-bot

# Filter logs
docker-compose logs ai-trading-bot | grep ERROR
```

### Health Status
```bash
# Check all health statuses
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"

# Watch health status
watch -n 5 'docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"'
```

## Production Deployment Checklist

- [ ] Run `./setup-docker-permissions.sh`
- [ ] Validate configuration: `docker-compose config --quiet`
- [ ] Check available resources: `free -h && df -h`
- [ ] Start core services first: `docker-compose up -d mcp-memory mcp-omnisearch`
- [ ] Wait for health checks to pass
- [ ] Start application services: `docker-compose up -d ai-trading-bot`
- [ ] Start dashboard services: `docker-compose up -d dashboard-backend dashboard-frontend`
- [ ] Verify all services healthy: `docker-compose ps`
- [ ] Test connectivity: `curl http://localhost:8000/health`

## Contact Information

For issues not covered in this guide:
1. Check Docker Compose logs: `docker-compose logs`
2. Review the validation report: `docker-compose-validation-report.md`
3. Check system resources: `docker system df`
4. Review service-specific documentation in `docs/` directory
