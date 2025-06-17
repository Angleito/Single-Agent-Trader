# Environment Variables Configuration

This document describes the environment variable patterns used across different deployment scenarios.

## Supported Environment Variables

### API Configuration
- `VITE_API_BASE_URL` - Base URL for API endpoints (preferred for consistency)
- `VITE_API_URL` - Alternative API URL (supported for compatibility)

### WebSocket Configuration
- `VITE_WS_URL` - WebSocket endpoint URL

### Environment Indicators
- `VITE_DOCKER_ENV` - Set to "true" when running in Docker containers
- `VITE_STANDALONE_MODE` - Set to "true" for standalone dashboard deployments
- `VITE_DEBUG` - Enable debug logging and development features

### TradingView Configuration
- `VITE_TRADINGVIEW_LIBRARY_PATH` - Path to TradingView charting library
- `VITE_TRADINGVIEW_THEME` - Theme for TradingView charts (dark/light)
- `VITE_TRADINGVIEW_AUTOSIZE` - Enable automatic chart resizing

## Environment Files

### `.env.development`
Used for local development with direct backend access:
- `VITE_API_BASE_URL=http://localhost:8000/api`
- `VITE_WS_URL=ws://localhost:8000/ws`

### `.env.production`
Used for production deployments with nginx proxy:
- `VITE_API_BASE_URL=/api`
- `VITE_WS_URL=/api/ws`

### `.env.docker`
Used for Docker deployments with nginx reverse proxy:
- `VITE_API_BASE_URL=/api`
- `VITE_WS_URL=/api/ws`
- `VITE_DOCKER_ENV=true`

### `.env.standalone`
Used for standalone dashboard deployments:
- `VITE_API_BASE_URL=http://dashboard-backend:8000/api`
- `VITE_WS_URL=ws://dashboard-backend:8000/ws`
- `VITE_STANDALONE_MODE=true`

## WebSocket URL Patterns

### Development
- **Direct backend**: `ws://localhost:8000/ws`
- **Vite dev server**: Proxied through Vite config

### Production (with nginx proxy)
- **Through nginx**: `/api/ws` (relative URL)
- **Direct access**: `ws://backend-host:8000/ws`

### Docker Scenarios
- **Container to container**: `ws://dashboard-backend:8000/ws`
- **Browser to nginx**: `/api/ws`
- **Browser to exposed port**: `ws://localhost:8000/ws`

## Runtime Configuration

The `inject-env.sh` script creates runtime configuration that overrides build-time variables:

```javascript
// Runtime variables (highest priority)
window.__RUNTIME_CONFIG__ = {
  API_URL: '/api',
  WS_URL: '/api/ws',
  DOCKER_ENV: 'true'
};

// Individual variables (for compatibility)
window.__API_URL__ = '/api';
window.__WS_URL__ = '/api/ws';
window.__VITE_API_URL__ = '/api';
window.__VITE_WS_URL__ = '/api/ws';
```

## Priority Order (highest to lowest)

1. Runtime configuration (`window.__RUNTIME_CONFIG__`)
2. Individual runtime variables (`window.__API_URL__`, etc.)
3. Vite environment variables (`import.meta.env.VITE_*`)
4. Default values (fallback)

## Container Environment Variables

When running in Docker, the following environment variables are injected:

```bash
# For dashboard containers
VITE_API_BASE_URL=/api
VITE_API_URL=/api
VITE_WS_URL=/api/ws
VITE_DOCKER_ENV=true

# For standalone containers
VITE_API_BASE_URL=http://dashboard-backend:8000/api
VITE_API_URL=http://dashboard-backend:8000/api
VITE_WS_URL=ws://dashboard-backend:8000/ws
VITE_STANDALONE_MODE=true
```