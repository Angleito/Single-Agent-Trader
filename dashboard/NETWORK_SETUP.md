# Dashboard Docker Network Configuration

This document explains the updated network configuration for the dashboard services and how to use them in different scenarios.

## Network Architecture

The dashboard now supports two network configurations:

1. **dashboard-network (172.20.0.0/16)** - Internal network for dashboard services
2. **trading-network (172.21.0.0/16)** - Shared network with main trading system

## Usage Scenarios

### 1. Standalone Dashboard Mode

Run only the dashboard services without the main trading system:

```bash
cd dashboard/
docker-compose up
```

Or for production with nginx:
```bash
cd dashboard/
docker-compose --profile production up
```

Or use the standalone override to avoid trading network connections:
```bash
cd dashboard/
docker-compose -f docker-compose.yml -f docker-compose.standalone.yml up
```

**Ports (Standalone):**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Production Frontend: http://localhost:3001 (with --profile production)
- Nginx Proxy: http://localhost:8080 (with --profile production)

### 2. Combined with Main Trading System

Run both the main trading system and dashboard together:

```bash
# Start main trading system (includes dashboard)
docker-compose up

# Or start them separately but on same network
docker-compose up  # Main system
cd dashboard/ && docker-compose up  # Dashboard (will connect to trading-network)
```

**Ports (Combined):**
- Main system handles port management
- Dashboard services connect via internal network names

### 3. Development Mode

For development with hot-reloading:

```bash
cd dashboard/
docker-compose up dashboard-frontend  # Development frontend only
# Backend can run locally: python backend/main.py
```

## Container Names

To avoid conflicts, standalone containers use different names:

- `dashboard-backend-standalone` (instead of `dashboard-backend`)
- `dashboard-frontend-standalone` (instead of `dashboard-frontend`)
- `dashboard-frontend-prod-standalone` (instead of `dashboard-frontend-prod`)
- `dashboard-nginx-standalone` (instead of `dashboard-nginx`)

## Network Connectivity

### Standalone Mode
- Services communicate only within `dashboard-network`
- External access via host ports (3000, 8000, 8080)
- No connection to trading bot

### Combined Mode
- Services join both `dashboard-network` and `trading-network`
- Can communicate with trading bot services
- Access to shared logs and data volumes

## Volume Mounts

### Standalone Mode
```yaml
volumes:
  - ./backend/logs:/app/logs
  - ./backend/data:/app/data
```

### Combined Mode
```yaml
volumes:
  - ./backend/logs:/app/logs
  - ./backend/data:/app/data
  - ../logs:/app/trading-logs:ro        # Access to trading bot logs
  - ../data:/app/trading-data:ro        # Access to trading bot data
```

## Environment Variables

Key environment variables for network configuration:

### Backend Service
```bash
# Service discovery
TRADING_BOT_CONTAINER=ai-trading-bot

# CORS configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001,http://localhost:8080,http://127.0.0.1:8080

# Security settings
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
SECURITY_HEADERS_ENABLED=true
```

### Frontend Service
```bash
# API endpoints (standalone)
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# API endpoints (with nginx proxy)
VITE_API_URL=/api
VITE_WS_URL=/api
```

## Health Checks

All services include health checks:

- **Backend**: `curl -f http://localhost:8000/health`
- **Nginx**: `wget --spider http://localhost/health`

## Troubleshooting

### Network Issues

1. **Standalone mode network conflicts:**
   ```bash
   docker network ls
   docker network rm dashboard-network trading-network
   docker-compose up
   ```

2. **Combined mode connection issues:**
   ```bash
   # Check if trading-network exists
   docker network inspect trading-network

   # If not, start main compose first
   cd .. && docker-compose up -d
   cd dashboard && docker-compose up
   ```

### Port Conflicts

1. **Port already in use:**
   ```bash
   # Check what's using the port
   lsof -i :8000

   # Stop conflicting services or use different ports in compose
   ```

2. **Multiple dashboard instances:**
   ```bash
   # Stop all dashboard containers
   docker-compose down
   cd .. && docker-compose down

   # Start only one instance
   ```

### Volume Mount Issues

1. **Permission denied:**
   ```bash
   # Fix permissions
   sudo chown -R 1000:1000 ./backend/logs ./backend/data
   chmod -R 755 ./backend/logs ./backend/data
   ```

2. **Missing parent directories:**
   ```bash
   # Create required directories
   mkdir -p ../logs ../data ./backend/logs ./backend/data
   ```

## Security Considerations

- Containers run as non-root user (1000:1000)
- Read-only root filesystem where possible
- Dropped capabilities except NET_BIND_SERVICE
- No privilege escalation
- Secure tmpfs mounts
- CORS restrictions in place
- Rate limiting enabled
- Security headers enabled

## Best Practices

1. **Use specific compose files:**
   - Use `-f docker-compose.standalone.yml` for standalone mode
   - Use main compose for integrated deployment

2. **Environment files:**
   - Create `.env` file in dashboard directory for local settings
   - Don't commit sensitive data

3. **Resource limits:**
   - Services have memory and CPU limits
   - Adjust based on your system capacity

4. **Logging:**
   - Logs are rotated (max 10MB, 3 files)
   - Use `docker-compose logs -f` to monitor

5. **Updates:**
   - Use `docker-compose pull` before `up` to get latest images
   - Use `docker-compose build --no-cache` for clean rebuilds
