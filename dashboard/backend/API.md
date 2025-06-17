# AI Trading Bot Dashboard API Documentation

## Overview

The dashboard backend provides WebSocket streaming and REST API endpoints for monitoring the AI trading bot's LLM decision-making process.

## WebSocket Endpoint

### `/ws`
Real-time streaming of bot logs and LLM events.

**Message Types:**
- `llm_event` - General LLM events (requests, responses, metrics)
- `llm_decision` - Trading decisions made by the LLM
- `echo` - Server acknowledgment of client messages

**LLM Decision Event Format:**
```json
{
  "timestamp": "2025-06-14T21:00:00.000Z",
  "type": "llm_decision",
  "event_type": "trading_decision",
  "action": "LONG",
  "size_pct": 15,
  "rationale": "Bullish signals detected",
  "symbol": "ETH-USD",
  "current_price": 2500.00,
  "indicators": {
    "cipher_a_dot": 1.0,
    "cipher_b_wave": 5.2,
    "rsi": 60.5
  },
  "session_id": "abc123",
  "source": "llm_parser"
}
```

## REST API Endpoints

### LLM Monitoring Endpoints

#### `GET /llm/status`
Get LLM completion monitoring status and summary.

**Response:**
```json
{
  "timestamp": "2025-06-14T21:00:00.000Z",
  "monitoring_active": true,
  "log_file": "/app/trading-logs/llm_completions.log",
  "total_parsed": {
    "requests": 0,
    "responses": 0,
    "decisions": 150,
    "alerts": 5,
    "performance_metrics": 10
  },
  "metrics": {
    "1_hour": { ... },
    "24_hours": { ... },
    "all_time": { ... }
  },
  "recent_activity": {
    "decisions_last_hour": 12,
    "decision_distribution": {
      "HOLD": 8,
      "LONG": 3,
      "SHORT": 1
    },
    "last_decision": { ... }
  }
}
```

#### `GET /llm/decisions`
Get recent LLM trading decisions.

**Query Parameters:**
- `limit` (int, 1-500): Number of decisions to return (default: 50)
- `action_filter` (string): Filter by action type (LONG, SHORT, CLOSE, HOLD)

**Response:**
```json
{
  "timestamp": "2025-06-14T21:00:00.000Z",
  "total_decisions": 150,
  "returned_decisions": 50,
  "action_distribution": {
    "HOLD": 80,
    "LONG": 45,
    "SHORT": 20,
    "CLOSE": 5
  },
  "decisions": [
    {
      "event_type": "trading_decision",
      "timestamp": "2025-06-14T21:00:00.000Z",
      "session_id": "abc123",
      "request_id": "llm_completion",
      "action": "LONG",
      "size_pct": 15,
      "rationale": "Bullish Cipher B, price>EMA",
      "symbol": "ETH-USD",
      "current_price": 2515.74,
      "indicators": { ... }
    }
  ]
}
```

#### `GET /llm/activity`
Get recent LLM activity across all event types.

**Query Parameters:**
- `limit` (int, 1-500): Number of events to return (default: 50)

#### `GET /llm/metrics`
Get detailed LLM performance metrics.

**Query Parameters:**
- `time_window` (string): Time window for metrics (1h, 24h, 7d, all)

**Response:**
```json
{
  "timestamp": "2025-06-14T21:00:00.000Z",
  "time_window": "24h",
  "metrics": {
    "total_decisions": 120,
    "decisions_per_hour": 5.0,
    "decision_counts": {
      "HOLD": 70,
      "LONG": 30,
      "SHORT": 15,
      "CLOSE": 5
    },
    "no_llm_logs": true
  }
}
```

### General Endpoints

#### `GET /status`
Get bot health check and status information.

#### `GET /health`
Simple health check endpoint.

#### `GET /trading-data`
Get current trading information and metrics.

### TradingView UDF Endpoints

#### `GET /udf/config`
TradingView UDF configuration.

#### `GET /udf/symbols`
Get symbol information.

#### `GET /udf/history`
Get historical OHLCV data.

#### `GET /udf/marks`
Get AI decision marks for chart annotations.

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request
- `404` - Not found
- `500` - Internal server error

Error responses include details:
```json
{
  "detail": "Error message here"
}
```

## CORS Configuration

CORS is enabled for all origins in development. Configure appropriately for production deployment.

## Docker Integration

The backend runs in a Docker container with access to:
- Trading bot logs at `/app/trading-logs/`
- Trading data at `/app/trading-data/`
- Docker socket for container management

## Testing

Use the included `test_websocket.py` script to verify WebSocket streaming and API endpoints:

```bash
python test_websocket.py
```
