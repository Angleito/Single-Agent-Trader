# WebSocket Integration Guide

This guide explains how to enable and use real-time WebSocket communication between the AI Trading Bot and the Dashboard.

## Overview

The WebSocket integration allows the trading bot to publish real-time events to the dashboard, enabling:
- Live market data streaming
- Real-time trading decisions and AI reasoning
- Technical indicator updates
- Position and P&L tracking
- System health monitoring

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐     WebSocket      ┌──────────────────┐
│  Trading Bot    │ ─────────────────> │ Dashboard Backend│ <───────────────── │ Dashboard Frontend│
│ (Publisher)     │  ws://backend:8000 │ (Broker)         │                    │ (Subscriber)      │
└─────────────────┘                    └──────────────────┘                    └──────────────────┘
```

## Quick Start

### 1. Enable WebSocket in Environment

Edit your `.env` file:

```bash
# Enable WebSocket publishing
SYSTEM__ENABLE_WEBSOCKET_PUBLISHING=true
```

### 2. Start Services with WebSocket Support

Using the WebSocket override configuration:

```bash
# Start all services with WebSocket enabled
docker-compose -f docker-compose.yml -f docker-compose.websocket.yml up -d

# Or set as default
export COMPOSE_FILE=docker-compose.yml:docker-compose.websocket.yml
docker-compose up -d
```

### 3. Verify WebSocket Connection

Run the test script:

```bash
./test_docker_websocket.sh
```

## Configuration

### Trading Bot Settings

The bot's WebSocket publisher is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SYSTEM__ENABLE_WEBSOCKET_PUBLISHING` | `false` | Enable/disable WebSocket publishing |
| `SYSTEM__WEBSOCKET_DASHBOARD_URL` | `ws://localhost:8000/ws` | Dashboard WebSocket URL |
| `SYSTEM__WEBSOCKET_PUBLISH_INTERVAL` | `1.0` | Publishing interval in seconds |
| `SYSTEM__WEBSOCKET_MAX_RETRIES` | `3` | Maximum reconnection attempts |
| `SYSTEM__WEBSOCKET_RETRY_DELAY` | `5` | Delay between reconnection attempts |
| `SYSTEM__WEBSOCKET_TIMEOUT` | `10` | Connection timeout in seconds |
| `SYSTEM__WEBSOCKET_QUEUE_SIZE` | `100` | Max queued messages during disconnection |

### Docker Network URLs

When running in Docker, use internal container names:
- Trading Bot → Backend: `ws://dashboard-backend:8000/ws`
- Frontend → Backend: `ws://dashboard-backend:8000/ws`

When running locally:
- All services: `ws://localhost:8000/ws`

## Message Types

The trading bot publishes the following message types:

### 1. Market Data
```json
{
  "type": "market_data",
  "data": {
    "symbol": "BTC-USD",
    "price": 45000.0,
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2. AI Decision
```json
{
  "type": "ai_decision",
  "data": {
    "action": "BUY",
    "reasoning": "Strong bullish signals detected",
    "confidence": 0.85,
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 3. Trading Decision (Detailed)
```json
{
  "type": "trading_decision",
  "data": {
    "request_id": "abc123",
    "action": "BUY",
    "confidence": 0.85,
    "reasoning": "Multiple indicators show bullish convergence",
    "price": 45000.0,
    "quantity": 0.1,
    "leverage": 5,
    "indicators": {
      "rsi": 45.2,
      "wavetrend": "bullish",
      "ema_trend": "up"
    },
    "risk_analysis": {
      "stop_loss": 44000.0,
      "take_profit": 46000.0,
      "risk_reward_ratio": 2.0
    },
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 4. System Status
```json
{
  "type": "system_status",
  "data": {
    "status": "healthy",
    "health": true,
    "errors": [],
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 5. Position Update
```json
{
  "type": "position_update",
  "data": {
    "symbol": "BTC-USD",
    "side": "long",
    "size": 0.1,
    "entry_price": 45000.0,
    "current_price": 45500.0,
    "unrealized_pnl": 50.0,
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

## Testing

### 1. Test WebSocket Client

Use the provided test client to verify connectivity:

```bash
# Test from local machine
python test_websocket_client.py --url ws://localhost:8000/ws --duration 30

# Test from inside Docker
docker-compose exec websocket-tester python test_websocket_client.py --docker-url

# Send test messages
python test_websocket_client.py --send all
```

### 2. Docker WebSocket Test

Run comprehensive Docker WebSocket tests:

```bash
./test_docker_websocket.sh
```

### 3. Monitor WebSocket Activity

Watch real-time WebSocket logs:

```bash
# Dashboard backend logs
docker-compose logs -f dashboard-backend | grep -i websocket

# Trading bot WebSocket publisher logs
docker-compose logs -f ai-trading-bot | grep -i websocket
```

## Troubleshooting

### Connection Refused

If you see "Connection refused" errors:

1. Verify services are running:
   ```bash
   docker-compose ps
   ```

2. Check WebSocket URL in bot configuration:
   ```bash
   docker-compose exec ai-trading-bot env | grep WEBSOCKET
   ```

3. Test network connectivity:
   ```bash
   docker-compose exec ai-trading-bot ping dashboard-backend
   ```

### No Messages Received

If WebSocket connects but no messages are received:

1. Ensure WebSocket publishing is enabled:
   ```bash
   docker-compose exec ai-trading-bot env | grep ENABLE_WEBSOCKET_PUBLISHING
   ```

2. Check bot is actively trading:
   ```bash
   docker-compose logs ai-trading-bot | tail -50
   ```

3. Verify message publishing in bot logs:
   ```bash
   docker-compose logs ai-trading-bot | grep "WebSocket.*publish"
   ```

### Connection Drops

If WebSocket connection drops frequently:

1. Increase reconnection attempts:
   ```yaml
   SYSTEM__WEBSOCKET_MAX_RETRIES=10
   SYSTEM__WEBSOCKET_RETRY_DELAY=10
   ```

2. Check Docker network stability:
   ```bash
   docker network inspect trading-network
   ```

3. Monitor resource usage:
   ```bash
   docker stats
   ```

## Production Considerations

### 1. Security

- Use WSS (WebSocket Secure) in production
- Implement authentication for WebSocket connections
- Rate limit WebSocket connections

### 2. Scaling

- Consider using a message broker (Redis, RabbitMQ) for multiple bot instances
- Implement WebSocket connection pooling
- Monitor WebSocket connection metrics

### 3. Monitoring

- Set up alerts for WebSocket disconnections
- Track message latency and throughput
- Monitor queue sizes for backpressure

## Advanced Usage

### Custom Message Types

To add new message types:

1. Update `WebSocketPublisher` in `bot/websocket_publisher.py`:
   ```python
   async def publish_custom_event(self, data: dict):
       message = {
           "type": "custom_event",
           "data": data
       }
       await self._send_message(message)
   ```

2. Call from trading logic:
   ```python
   if self.websocket_publisher:
       await self.websocket_publisher.publish_custom_event({
           "event": "strategy_change",
           "details": "Switched to conservative mode"
       })
   ```

3. Handle in frontend WebSocket client:
   ```typescript
   ws.on('message', (data) => {
       const message = JSON.parse(data);
       if (message.type === 'custom_event') {
           handleCustomEvent(message.data);
       }
   });
   ```

### Filtering Messages

To reduce WebSocket traffic, implement message filtering:

```python
# In WebSocketPublisher
async def publish_market_data(self, symbol: str, price: float):
    # Only publish significant price changes
    if abs(price - self.last_price) / self.last_price > 0.001:  # 0.1%
        await self._send_message({...})
        self.last_price = price
```

## Related Documentation

- [Dashboard Frontend WebSocket Client](../dashboard/frontend/src/stores/websocket.ts)
- [Dashboard Backend WebSocket Handler](../dashboard/backend/main.py)
- [Trading Bot WebSocket Publisher](../bot/websocket_publisher.py)
