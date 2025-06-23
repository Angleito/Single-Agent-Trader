# Docker Infrastructure Analysis Report

## Executive Summary

This report provides a comprehensive analysis of the Docker infrastructure for the AI Trading Bot system. The infrastructure consists of multiple interconnected services orchestrated via Docker Compose, with identified issues related to network connectivity and service dependencies.

## Service Architecture

### Core Services

#### 1. **ai-trading-bot** (Main Trading Bot)
- **Role**: Core trading bot responsible for automated trading decisions
- **Container**: `ai-trading-bot`
- **Dependencies**:
  - `mcp-memory` (optional, condition: service_started)
  - `mcp-omnisearch` (optional, condition: service_started)
- **Health Check**: `/app/healthcheck.sh`
- **Network Aliases**: `ai-trading-bot`, `trading-bot`
- **Key Features**:
  - WebSocket publishing to dashboard-backend
  - MCP memory integration for AI learning
  - OmniSearch integration for market intelligence
  - Configured for 15-second intervals with SUI-PERP symbol

#### 2. **bluefin-service** (Bluefin DEX Integration)
- **Role**: Isolated container for Bluefin decentralized exchange operations
- **Container**: `bluefin-service`
- **Port Mapping**: `127.0.0.1:8081 → 8080`
- **Health Check**: `curl http://localhost:8080/health`
- **Network Aliases**: `bluefin-service`, `bluefin`
- **Key Features**:
  - Read-only root filesystem with tmpfs mounts
  - Security hardening with capability restrictions
  - Rate limiting configuration

#### 3. **mcp-memory** (AI Memory Server)
- **Role**: Persistent memory storage for AI learning and trade experiences
- **Container**: `mcp-memory-server`
- **Port Mapping**: `127.0.0.1:8765 → 8765`
- **Health Check**: `curl http://localhost:8765/health`
- **Network Aliases**: `mcp-memory`, `memory-server`
- **Key Features**:
  - 90-day memory retention
  - Persistent volume mounting at `./data/mcp_memory`

#### 4. **mcp-omnisearch** (Market Intelligence)
- **Role**: Enhanced market intelligence through web search integration
- **Container**: `mcp-omnisearch-server`
- **Port Mapping**: `127.0.0.1:8767 → 8767`
- **Health Check**: Node.js health check
- **Network Aliases**: `mcp-omnisearch`, `omnisearch`
- **Key Features**:
  - Multiple search provider support (Tavily, Perplexity, Kagi, etc.)
  - Node.js-based service

#### 5. **dashboard-backend** (Dashboard API)
- **Role**: Backend API for the trading dashboard
- **Container**: `dashboard-backend`
- **Port Mapping**: `8000 → 8000` (public access)
- **Dependencies**:
  - `bluefin-service` (optional, condition: service_started)
- **Health Check**: `curl http://localhost:8000/health`
- **Network Aliases**: `dashboard-backend`, `api`
- **Key Features**:
  - WebSocket support for real-time updates
  - CORS configuration for multiple origins
  - Rate limiting enabled

#### 6. **dashboard-frontend** (Web UI)
- **Role**: Web-based trading dashboard interface
- **Container**: `dashboard-frontend`
- **Port Mapping**: `3000 → 8080`
- **Dependencies**: `dashboard-backend`
- **Network Aliases**: `dashboard-frontend`, `frontend`
- **Key Features**:
  - Nginx-based static file serving
  - Vite-based React application

### Additional Services (Profiles)

- **dashboard-frontend-prod** (Profile: production)
- **dashboard-nginx** (Profile: production)
- **ai-trading-bot-dev** (Profile: dev)
- **ai-trading-bot-vps** (Profile: vps)

## Dependency Graph

```
ai-trading-bot
├── mcp-memory (optional)
├── mcp-omnisearch (optional)
└── → dashboard-backend (WebSocket connection)

dashboard-backend
└── bluefin-service (optional)

dashboard-frontend
└── dashboard-backend (required)

dashboard-nginx (production)
├── dashboard-backend
└── dashboard-frontend-prod
```

## Network Configuration

### Network Name: `trading-network`
- **Driver**: bridge
- **IPv6**: Disabled
- **Subnet**: 172.20.0.0/16 (from override)
- **Gateway**: 172.20.0.1

### Service Network Aliases
Each service has multiple DNS names for inter-container communication:
- `bluefin-service`: bluefin-service, bluefin
- `ai-trading-bot`: ai-trading-bot, trading-bot
- `mcp-memory`: mcp-memory, memory-server
- `mcp-omnisearch`: mcp-omnisearch, omnisearch
- `dashboard-backend`: dashboard-backend, api
- `dashboard-frontend`: dashboard-frontend, frontend

## Health Check Status

| Service | Health Check Method | Interval | Timeout | Retries | Start Period |
|---------|-------------------|----------|----------|---------|--------------|
| bluefin-service | HTTP /health | 30s→45s* | 10s→15s* | 3 | 15s→30s* |
| ai-trading-bot | Shell script | 30s→45s* | 15s→20s* | 3 | 30s→60s* |
| mcp-memory | HTTP /health | 30s | 10s | 3 | 10s→15s* |
| mcp-omnisearch | Node check→wget* | 30s | 10s | 3 | 10s→20s* |
| dashboard-backend | HTTP /health | 30s→45s* | 10s→15s* | 3 | 10s→20s* |
| dashboard-nginx | HTTP /health | 30s | 10s | 3 | 15s |

*Enhanced values from docker-compose.override.yml

## Identified Issues

### 1. Network Infrastructure
- **Issue**: `trading-network` not found when checked
- **Impact**: Services cannot communicate with each other
- **Root Cause**: Docker Compose stack not running or network not created

### 2. Service Connectivity
- **Issue**: WebSocket connection failures to dashboard-backend
- **Evidence**: `Failed to connect to dashboard WebSocket` errors
- **Configuration**:
  - Primary URL: `ws://dashboard-backend:8000/ws`
  - Fallback URLs: `ws://localhost:8000/ws`, `ws://127.0.0.1:8000/ws`
  - Max retries: 15 (enhanced from 10)
  - Connection delay: 10 seconds

### 3. Service Dependencies
- **Issue**: Optional dependencies marked as `required: false` in compose v3 syntax
- **Impact**: Services may start before dependencies are ready
- **Affected Services**:
  - ai-trading-bot → mcp-memory, mcp-omnisearch
  - dashboard-backend → bluefin-service

### 4. DNS Resolution
- **Issue**: Potential DNS resolution failures between containers
- **Configuration**: Services use network aliases for discovery
- **Enhanced Discovery**: Override file adds fallback URLs

### 5. Container Status
- **Issue**: No related containers found running
- **Impact**: Entire stack appears to be down

## Security Configuration

### Container Security Features
1. **Read-only Root Filesystem**: All services use read-only root FS
2. **Capability Restrictions**: `cap_drop: ALL` with minimal additions
3. **User Permissions**: Run as non-root users (HOST_UID:HOST_GID)
4. **Tmpfs Mounts**: Writable areas isolated to tmpfs
5. **Security Options**:
   - no-new-privileges: false (VPS compatibility)
   - seccomp: unconfined (VPS requirement)
   - apparmor: unconfined (Ubuntu compatibility)

### Network Security
1. **Port Bindings**: Most services bind to localhost only (127.0.0.1)
2. **CORS Configuration**: Comprehensive origin list for dashboard
3. **Rate Limiting**: Enabled on dashboard-backend
4. **API Keys**: Environment-based configuration

## Volume Mounts and Permissions

### Critical Volume Mounts
```
ai-trading-bot:
  ./logs:/app/logs:rw
  ./data:/app/data:rw
  ./config:/app/config:ro
  ./bot:/app/bot:ro

bluefin-service:
  ./logs/bluefin:/app/logs

mcp-memory:
  ./data/mcp_memory:/app/data
  ./logs/mcp:/app/logs

dashboard-backend:
  ./dashboard/backend/logs:/app/logs
  ./dashboard/backend/data:/app/data
  ./logs:/app/trading-logs:ro
  ./data:/app/trading-data:ro
```

### Permission Requirements
- User mapping via HOST_UID and HOST_GID environment variables
- Script available: `./setup-docker-permissions.sh`
- Default fallback: UID=1000, GID=1000

## Environment Variable Configuration

### Critical Environment Variables
1. **Exchange Configuration**:
   - `EXCHANGE__EXCHANGE_TYPE`: coinbase or bluefin
   - `EXCHANGE__BLUEFIN_NETWORK`: mainnet or testnet

2. **Service URLs**:
   - `BLUEFIN_SERVICE_URL`: http://bluefin-service:8080
   - `MCP_SERVER_URL`: http://mcp-memory:8765
   - `OMNISEARCH__SERVER_URL`: http://mcp-omnisearch:8767

3. **WebSocket Configuration**:
   - `SYSTEM__WEBSOCKET_DASHBOARD_URL`: ws://dashboard-backend:8000/ws
   - `SYSTEM__WEBSOCKET_MAX_RETRIES`: 15
   - `SYSTEM__WEBSOCKET_CONNECTION_DELAY`: 10

## Recommendations

### Immediate Actions
1. **Start Docker Stack**:
   ```bash
   docker-compose up -d
   ```

2. **Verify Network Creation**:
   ```bash
   docker network ls | grep trading-network
   ```

3. **Check Service Health**:
   ```bash
   docker-compose ps
   docker-compose logs --tail=50
   ```

### Configuration Improvements
1. **Update Docker Compose Version**: Use compose specification v3.9+ for better dependency handling
2. **Add Restart Policies**: Already implemented via override file
3. **Implement Health Check Dependencies**: Use `condition: service_healthy` where appropriate
4. **Add Network Connectivity Tests**: Implement inter-service ping tests

### Startup Order Recommendations
1. Start infrastructure services first:
   - mcp-memory
   - mcp-omnisearch
   - bluefin-service
2. Start dashboard services:
   - dashboard-backend
3. Start main application:
   - ai-trading-bot
4. Start frontend services:
   - dashboard-frontend

### Monitoring Recommendations
1. Implement centralized logging aggregation
2. Add Prometheus metrics endpoints
3. Create health check dashboard
4. Implement service discovery mechanism

## Diagnostic Tools Available

### diagnose-bluefin-connectivity.sh
A comprehensive diagnostic script is available at `./scripts/diagnose-bluefin-connectivity.sh` that performs:
- Docker daemon availability check
- Container running status verification
- Port binding validation (8080)
- Internal health check testing
- Cross-container DNS resolution
- Network connectivity testing
- Environment variable inspection

### Key Diagnostic Commands
```bash
# Check container status
docker-compose ps

# Verify network exists
docker network ls | grep trading-network

# Test bluefin service health
docker exec bluefin-service curl -f http://localhost:8080/health

# Test cross-container connectivity
docker exec ai-trading-bot curl -f http://bluefin-service:8080/health

# Check service logs
docker-compose logs --tail=50 bluefin-service
docker-compose logs --tail=50 dashboard-backend
```

## Troubleshooting Steps

### Step 1: Verify Docker Setup
```bash
# Ensure Docker is running
docker info

# Check for any existing containers
docker ps -a | grep -E "(bluefin|trading|dashboard|mcp)"
```

### Step 2: Clean Start
```bash
# Stop all services
docker-compose down

# Remove orphaned containers
docker-compose down --remove-orphans

# Start services in correct order
docker-compose up -d mcp-memory mcp-omnisearch bluefin-service
sleep 10
docker-compose up -d dashboard-backend
sleep 5
docker-compose up -d ai-trading-bot dashboard-frontend
```

### Step 3: Verify Connectivity
```bash
# Run diagnostic script
./scripts/diagnose-bluefin-connectivity.sh

# Check health endpoints
curl http://localhost:8765/health  # mcp-memory
curl http://localhost:8767/health  # mcp-omnisearch
curl http://localhost:8081/health  # bluefin-service
curl http://localhost:8000/health  # dashboard-backend
```

## Conclusion

The Docker infrastructure is well-designed with comprehensive security measures and service isolation. The primary issue appears to be that the stack is not currently running, leading to network connectivity failures. The system includes:

1. **Robust Architecture**: Multi-service design with clear separation of concerns
2. **Enhanced Resilience**: Override file provides improved health checks and retry mechanisms
3. **Security Hardening**: Read-only filesystems, capability restrictions, and user isolation
4. **Diagnostic Tools**: Comprehensive scripts for troubleshooting connectivity issues
5. **Flexible Configuration**: Support for multiple deployment profiles (dev, production, VPS)

Once started properly with the correct sequence and network creation, the services should communicate effectively through the shared `trading-network` bridge network. The diagnostic script provides a systematic approach to identifying and resolving connectivity issues.
