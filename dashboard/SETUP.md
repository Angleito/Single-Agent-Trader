# AI Trading Bot Dashboard Setup

This guide covers setting up the FastAPI backend server for monitoring the AI trading bot.

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- AI trading bot running in container named `ai-trading-bot`

## Quick Start

### Option 1: Direct Python Run

```bash
cd dashboard/backend

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

### Option 2: Using Startup Script

```bash
cd dashboard/backend
./start.sh
```

### Option 3: Docker Compose (Recommended)

```bash
cd dashboard
docker-compose up -d
```

## Server Endpoints

Once running, the server will be available at `http://localhost:8000`

### WebSocket
- `ws://localhost:8000/ws` - Real-time log streaming

### REST API
- `GET /` - Service information
- `GET /status` - Bot health and system status
- `GET /trading-data` - Trading metrics and performance
- `GET /logs?limit=100` - Recent log entries
- `GET /health` - Health check
- `POST /control/restart` - Restart bot container

## Testing the Setup

### 1. Test Server Health
```bash
curl http://localhost:8000/health
```

### 2. Test Bot Status
```bash
curl http://localhost:8000/status
```

### 3. Test WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    console.log('Received:', JSON.parse(event.data));
};
```

### 4. Test Trading Data
```bash
curl http://localhost:8000/trading-data
```

## Configuration

### Environment Variables
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: INFO)

### Docker Container Name
The server expects the trading bot container to be named `ai-trading-bot`. To change this, modify the `LogStreamer` class in `main.py`:

```python
log_streamer = LogStreamer(container_name="your-container-name")
```

## Troubleshooting

### Common Issues

1. **FastAPI Import Error**
   ```bash
   pip install fastapi uvicorn websockets python-multipart
   ```

2. **Docker Permission Issues**
   ```bash
   sudo usermod -aG docker $USER
   # Then logout and login again
   ```

3. **Container Not Found**
   - Ensure the AI trading bot container is running
   - Check container name with `docker ps`

4. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process or use different port
   ```

### Logs and Debugging

- Server logs: Check console output when running directly
- Docker logs: `docker-compose logs dashboard-backend`
- WebSocket debugging: Use browser developer tools

## Production Deployment

For production deployment:

1. **Security**: Configure CORS origins appropriately
2. **Authentication**: Add authentication middleware
3. **SSL/TLS**: Use reverse proxy (nginx) with SSL
4. **Resource Limits**: Set Docker memory/CPU limits
5. **Monitoring**: Add metrics collection
6. **Backup**: Implement log archiving

### Example Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Development

### Code Structure
- `main.py` - FastAPI application with all endpoints
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `start.sh` - Development startup script

### Adding New Endpoints
1. Add endpoint function to `main.py`
2. Add error handling
3. Update documentation
4. Test the endpoint

### WebSocket Message Format
```json
{
  "timestamp": "ISO-8601 timestamp",
  "level": "INFO|WARN|ERROR|DEBUG",
  "message": "Log message content",
  "source": "docker-logs|dashboard-api"
}
```
