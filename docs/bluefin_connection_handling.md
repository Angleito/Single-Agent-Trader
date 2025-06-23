# Bluefin Service Connection Handling

This document describes the enhanced connection handling for the Bluefin service client, including automatic fallback mechanisms and graceful degradation.

## Overview

The Bluefin service client now implements robust connection handling with:
- Multiple fallback URLs
- Per-URL circuit breakers
- Automatic service discovery
- Graceful degradation for optional service
- Environment-aware URL prioritization

## Connection Fallback Mechanism

### URL Priority

The client automatically tries multiple URLs in order of priority:

#### When Running Inside Docker:
1. `http://bluefin-service:8080` - Docker service name (primary)
2. `http://host.docker.internal:8080` - Docker Desktop host access
3. `http://172.17.0.1:8080` - Default Docker bridge gateway
4. `http://localhost:8080` - Fallback to localhost
5. `http://127.0.0.1:8080` - Fallback to localhost IP

#### When Running Outside Docker:
1. `http://localhost:8080` - Local host (primary)
2. `http://127.0.0.1:8080` - Localhost IP
3. `http://bluefin-service:8080` - Try Docker service name
4. `http://host.docker.internal:8080` - Docker Desktop fallback

### Docker Environment Detection

The client automatically detects if it's running inside a Docker container by checking:
- Existence of `/.dockerenv` file
- Docker references in `/proc/self/cgroup`
- Docker-specific environment variables
- Container-style hostname patterns

## Circuit Breaker Pattern

Each URL has its own circuit breaker to prevent repeated connection attempts to failed endpoints:

- **Failure Threshold**: 3 consecutive failures
- **Recovery Timeout**: 60 seconds
- **States**: CLOSED (normal), OPEN (blocking requests)

When a URL's circuit breaker opens:
1. That URL is skipped for 60 seconds
2. The client tries the next available URL
3. After 60 seconds, the circuit breaker automatically closes

## Service Discovery

The connection process follows these steps:

1. **Initial Connection**: Try the primary URL first
2. **Fallback Sequence**: If primary fails, try each fallback URL in order
3. **Last Successful URL**: Remember and prioritize the last working URL
4. **Health Checks**: Validate service health before marking as connected
5. **Automatic Rediscovery**: If all URLs fail during operation, trigger rediscovery

## Graceful Degradation

### Paper Trading Mode (DRY_RUN=true)
- Service connection is **optional**
- Bot continues with local trade simulation
- Clear log messages indicate paper trading mode
- No real positions or orders are created

### Live Trading Mode (DRY_RUN=false)
- Service connection is **recommended but not required**
- Warning messages clearly indicate limited functionality
- Position queries and order execution won't work without service
- Bot suggests switching to paper trading mode

## Configuration

### Environment Variables

```bash
# Optional: Override default service URL
BLUEFIN_SERVICE_URL=http://custom-service:8080

# Optional: API key for service authentication
BLUEFIN_SERVICE_API_KEY=your-api-key

# Trading mode
SYSTEM__DRY_RUN=true  # Paper trading (service optional)
SYSTEM__DRY_RUN=false # Live trading (service recommended)
```

### Docker Compose Setup

```yaml
services:
  ai-trading-bot:
    depends_on:
      bluefin-service:
        condition: service_healthy
        required: false  # Service is optional
    environment:
      - BLUEFIN_SERVICE_URL=http://bluefin-service:8080
      - BLUEFIN_SERVICE_API_KEY=${BLUEFIN_SERVICE_API_KEY}

  bluefin-service:
    image: bluefin-service:latest
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Error Messages and Troubleshooting

### Connection Success
```
✅ Successfully connected to Bluefin service
   Active URL: http://localhost:8080
   Service status: healthy
```

### Paper Trading - Service Unavailable
```
📋 PAPER TRADING MODE: Continuing without Bluefin service.
   • All trades will be simulated locally
   • Market data will be fetched if available
   • No real positions or orders will be created
   Service issue: Connection refused
```

### Live Trading - Service Unavailable
```
⚠️  LIVE TRADING MODE - Bluefin service unavailable!
   • The Bluefin service container appears to be down
   • Position queries and order execution WILL NOT WORK
   • Consider switching to paper trading mode (DRY_RUN=true)
   Service issue: All URLs failed
   To fix: Ensure bluefin-service container is running
```

## Testing Connection

Use the provided test script to verify connection handling:

```bash
# Test connection with different scenarios
python test_bluefin_connection.py

# Run with custom service URL
BLUEFIN_SERVICE_URL=http://localhost:9090 python test_bluefin_connection.py
```

## Implementation Details

### Key Features

1. **Automatic URL Switching**: Seamlessly switches between URLs without interrupting operations
2. **Connection Pooling**: Maintains efficient connection pool with 10 connections
3. **Session Management**: Automatic session renewal after 30 minutes or 1000 requests
4. **Request Timeouts**: Configurable timeouts for different operation types
5. **Comprehensive Logging**: Detailed logs for debugging connection issues

### Performance Metrics

The client tracks:
- Total requests and success rate
- Response times (rolling window)
- Connection attempts and failures
- Circuit breaker states
- Per-URL failure counts

## Best Practices

1. **Development**: Run bluefin-service locally on port 8080
2. **Docker**: Use docker-compose with proper service dependencies
3. **Production**: Configure primary URL to your production service endpoint
4. **Monitoring**: Check logs for connection warnings and circuit breaker activations
5. **Testing**: Always test in paper trading mode first

## Troubleshooting Common Issues

### Service Not Starting
- Check if port 8080 is already in use
- Verify Docker network configuration
- Ensure API key is properly configured

### Connection Timeouts
- Increase timeout values for slow networks
- Check firewall rules
- Verify service health endpoint responds

### Circuit Breaker Always Open
- Check service logs for errors
- Verify URL is accessible
- Wait for recovery timeout (60s)
- Manually reset by restarting the bot
