# AI Trading Bot Dashboard Backend

FastAPI server providing WebSocket streaming of bot logs and REST API endpoints for monitoring the AI trading bot.

## Features

- **WebSocket Streaming**: Real-time log streaming from Docker container
- **REST API Endpoints**: Status, trading data, and control endpoints
- **Connection Management**: Handles multiple WebSocket connections
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Configured for frontend integration

## API Endpoints

### WebSocket
- `WS /ws` - Real-time log streaming

### REST Endpoints
- `GET /` - Basic service information
- `GET /status` - Bot health check and system status
- `GET /trading-data` - Current trading information and metrics
- `GET /logs?limit=100` - Recent logs from buffer
- `GET /health` - Health check endpoint
- `POST /control/restart` - Restart the trading bot container

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the startup script
./start.sh
```

## Running

```bash
# Direct run
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Using startup script
./start.sh
```

## Configuration

The server expects the AI trading bot to be running in a Docker container named `ai-trading-bot`. This can be configured in the `LogStreamer` class.

## WebSocket Usage

Connect to `ws://localhost:8000/ws` to receive real-time log streams. Messages are sent as JSON:

```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "message": "Trading bot started",
  "source": "docker-logs"
}
```

## Development

The server includes:
- Automatic log buffering (last 1000 entries)
- Connection management for WebSocket clients
- Background Docker log streaming
- Graceful shutdown handling
- Comprehensive error handling

## Production Considerations

- Configure CORS origins appropriately
- Add authentication/authorization
- Set up proper logging configuration
- Configure resource limits
- Add monitoring and metrics