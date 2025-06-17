# Dashboard Docker Integration

This document describes the complete Docker setup for the AI Trading Bot Dashboard system, including both backend and frontend services with production-ready configurations.

## Architecture Overview

The Docker setup includes the following services:

1. **ai-trading-bot**: Core trading bot application
2. **dashboard-backend**: FastAPI backend with Docker log access
3. **dashboard-frontend**: Vite development server with HMR
4. **dashboard-frontend-prod**: Production build with Nginx
5. **dashboard-nginx**: Reverse proxy for production

## Quick Start

### Development Environment

```bash
# Initial setup
cd dashboard
./docker-setup.sh setup

# Start development environment
./docker-setup.sh dev

# Access the dashboard
open http://localhost:3000  # Frontend
open http://localhost:8000  # Backend API
```

### Production Environment

```bash
# Start production environment
./docker-setup.sh prod

# Access the dashboard
open http://localhost:8080  # Main access through Nginx
```

## Service Configuration

### AI Trading Bot
- **Container**: `ai-trading-bot`
- **Network**: `trading-network`
- **Volumes**: Logs and data persistence
- **Health Check**: Custom health check script

### Dashboard Backend
- **Container**: `dashboard-backend`
- **Port**: `8000`
- **Features**:
  - Docker socket access for log reading
  - Access to trading bot logs and data
  - FastAPI with auto-documentation
  - Health checks and monitoring

### Dashboard Frontend (Development)
- **Container**: `dashboard-frontend`
- **Port**: `3000`
- **Features**:
  - Vite dev server with HMR
  - Source code mounting for development
  - Environment variable configuration

### Dashboard Frontend (Production)
- **Container**: `dashboard-frontend-prod`
- **Features**:
  - Optimized production build
  - Nginx serving static files
  - Compressed assets
  - Security headers

### Nginx Reverse Proxy
- **Container**: `dashboard-nginx`
- **Port**: `8080`
- **Features**:
  - Load balancing
  - SSL termination ready
  - Rate limiting
  - Caching configuration

## Volume Mounts

### Development Volumes
```yaml
# Trading bot
- ./logs:/app/logs
- ./data:/app/data

# Dashboard backend
- /var/run/docker.sock:/var/run/docker.sock:ro
- ./dashboard/backend:/app:delegated
- ./logs:/app/trading-logs:ro

# Dashboard frontend
- ./dashboard/frontend:/app
- /app/node_modules
```

### Production Volumes
```yaml
# Persistent data only
- dashboard-logs:/app/logs
- dashboard-data:/app/data
- ./logs:/app/trading-logs:ro
```

## Network Configuration

All services communicate through the `trading-network` bridge network:

- **Subnet**: `172.21.0.0/16`
- **Driver**: `bridge`
- **Internal DNS**: Service discovery by container name

## Environment Variables

### Backend Configuration
```env
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
TRADING_BOT_CONTAINER=ai-trading-bot
```

### Frontend Configuration
```env
NODE_ENV=development
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
CHOKIDAR_USEPOLLING=true
```

## Health Checks

### Backend Health Check
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Frontend Health Check
```yaml
healthcheck:
  test: ["CMD", "wget", "--spider", "http://localhost/"]
  interval: 30s
  timeout: 3s
  retries: 3
```

## Resource Limits

### Development Limits
- **Backend**: 512MB RAM, 0.3 CPU
- **Frontend**: 1GB RAM, 0.5 CPU
- **Trading Bot**: 1GB RAM, 0.5 CPU

### Production Limits
- **Backend**: 512MB RAM, 0.3 CPU
- **Frontend**: 256MB RAM, 0.2 CPU
- **Nginx**: 64MB RAM, 0.1 CPU

## Security Configuration

### Docker Socket Access
- **Read-only** Docker socket mount for backend
- **Principle of least privilege** for container access
- **Non-root user** in production containers

### Network Security
- **Internal communication** through bridge network
- **Rate limiting** on API endpoints
- **CORS configuration** for frontend access

### Headers and Security
```nginx
# Security headers in Nginx
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
```

## Logging Configuration

### Log Rotation
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Log Access
- **Backend logs**: `/app/logs`
- **Trading bot logs**: `/app/trading-logs` (read-only)
- **Container logs**: via Docker socket

## Development Workflow

### 1. Initial Setup
```bash
# Clone and setup
git clone <repository>
cd dashboard
./docker-setup.sh setup
```

### 2. Development
```bash
# Start development environment
./docker-setup.sh dev

# View logs
./docker-setup.sh logs
./docker-setup.sh logs dashboard-backend

# Make changes (auto-reload enabled)
# Frontend: HMR reload
# Backend: Auto-reload with uvicorn --reload
```

### 3. Testing
```bash
# Check service status
./docker-setup.sh status

# View specific service logs
docker-compose logs -f dashboard-backend
```

### 4. Production Deployment
```bash
# Build and start production
./docker-setup.sh prod

# Access through Nginx proxy
curl http://localhost:8080/health
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs service-name

# Rebuild service
docker-compose build --no-cache service-name
```

#### Port Conflicts
```bash
# Check port usage
lsof -i :3000
lsof -i :8000

# Stop conflicting services
./docker-setup.sh stop
```

#### Permission Issues
```bash
# Fix Docker socket permissions (macOS)
sudo chmod 666 /var/run/docker.sock

# Fix file permissions
sudo chown -R $USER:$USER ./logs ./data
```

#### Network Issues
```bash
# Recreate network
docker network rm trading-network
docker-compose up -d
```

### Debug Commands

```bash
# Enter container shell
docker exec -it dashboard-backend /bin/bash
docker exec -it dashboard-frontend /bin/sh

# Check network connectivity
docker exec dashboard-backend ping ai-trading-bot
docker exec dashboard-frontend ping dashboard-backend

# View container resources
docker stats

# Inspect container configuration
docker inspect dashboard-backend
```

## Production Considerations

### SSL/TLS Configuration
- Add SSL certificates to Nginx configuration
- Update `docker-compose.prod.yml` for HTTPS
- Configure proper domain names

### Monitoring and Alerting
- Add monitoring services (Prometheus, Grafana)
- Configure log aggregation (ELK stack)
- Set up health check alerts

### Backup and Recovery
- Implement volume backup strategy
- Database backup procedures
- Configuration backup

### Scaling
- Add load balancer for multiple backend instances
- Implement container orchestration (Kubernetes)
- Database clustering and replication

## File Structure

```
dashboard/
├── docker-compose.yml          # Main dashboard services
├── .env.docker                # Docker environment variables
├── docker-setup.sh            # Setup and management script
├── DOCKER.md                  # This documentation
├── backend/
│   ├── Dockerfile             # Backend container definition
│   └── ...
├── frontend/
│   ├── Dockerfile             # Frontend container definition
│   ├── nginx.conf            # Frontend Nginx config
│   └── ...
└── nginx/
    ├── nginx.conf            # Main Nginx configuration
    └── conf.d/
        └── default.conf      # Site configuration

# Project root
docker-compose.yml              # Main project compose file
docker-compose.override.yml     # Development overrides
docker-compose.prod.yml         # Production configuration
```

## Commands Reference

```bash
# Setup and build
./docker-setup.sh setup

# Start/stop services
./docker-setup.sh dev      # Development mode
./docker-setup.sh prod     # Production mode
./docker-setup.sh stop     # Stop all services
./docker-setup.sh restart  # Restart services

# Monitoring
./docker-setup.sh status   # Service status
./docker-setup.sh logs     # All logs
./docker-setup.sh logs dashboard-backend  # Specific service

# Maintenance
./docker-setup.sh cleanup  # Clean up resources

# Manual Docker commands
docker-compose up -d                    # Start all services
docker-compose --profile production up  # Production profile
docker-compose down -v                  # Stop and remove volumes
docker-compose build --no-cache         # Rebuild all images
```
