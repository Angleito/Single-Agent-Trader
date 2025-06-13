# WebSocket Client Implementation

## Overview

A robust WebSocket client implementation for real-time communication with the AI trading bot backend, featuring automatic reconnection, error handling, and message type routing.

## Files Created/Updated

### 1. `/src/websocket.ts` - Main WebSocket Client
**Complete implementation with:**
- Automatic reconnection with exponential backoff
- Connection timeout handling
- Ping/pong keepalive mechanism
- Event-driven message routing
- Type-safe message handling
- Error recovery and logging
- Clean API for UI components

### 2. `/src/types.ts` - Type Definitions
**Updated to include:**
- Enhanced WebSocket message types
- Support for new backend message formats
- Trading loop, AI decision, system status, and error message types

### 3. `/src/main.ts` - Integration with Dashboard
**Updated to use:**
- New WebSocket client API
- Type-specific message handlers
- Enhanced error handling
- Connection status monitoring

### 4. `/src/websocket-example.ts` - Usage Examples
**Complete examples demonstrating:**
- Singleton pattern usage
- Custom configuration
- React-style hooks
- Production-ready wrapper class

## Key Features

### Connection Management
```typescript
const client = new DashboardWebSocket('ws://localhost:8000/ws', {
  maxReconnectAttempts: 10,
  initialReconnectDelay: 1000,
  maxReconnectDelay: 30000,
  pingInterval: 30000,
  connectionTimeout: 10000
});

client.connect();
client.disconnect();
client.isConnected();
client.getConnectionStatus();
```

### Message Type Handling
```typescript
// Subscribe to specific message types
client.on('trading_loop', (message) => {
  console.log(`Trading signal: ${message.data.action}`);
});

client.on('ai_decision', (message) => {
  console.log(`AI Decision: ${message.data.reasoning}`);
});

client.on('system_status', (message) => {
  console.log(`System: ${message.data.status}`);
});

// Subscribe to all messages
client.on('*', (message) => {
  console.log('Received:', message);
});
```

### Error Handling & Recovery
```typescript
client.onError((error) => {
  console.error('WebSocket error:', error);
});

client.onConnectionStatusChange((status) => {
  switch (status) {
    case 'connecting': /* handle connecting */; break;
    case 'connected': /* handle connected */; break;
    case 'disconnected': /* handle disconnected */; break;
    case 'error': /* handle error */; break;
  }
});
```

## Message Types Supported

### 1. Trading Loop Messages
```typescript
{
  type: 'trading_loop',
  data: {
    price: number,
    action: string,
    confidence: number,
    timestamp?: string,
    symbol?: string
  }
}
```

### 2. AI Decision Messages
```typescript
{
  type: 'ai_decision',
  data: {
    action: string,
    reasoning: string,
    confidence?: number,
    timestamp?: string
  }
}
```

### 3. System Status Messages
```typescript
{
  type: 'system_status',
  data: {
    status: string,
    health: boolean,
    errors: string[],
    timestamp?: string
  }
}
```

### 4. Error Messages
```typescript
{
  type: 'error',
  data: {
    message: string,
    level: string,
    timestamp?: string
  }
}
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `url` | `'ws://localhost:8000/ws'` | WebSocket endpoint URL |
| `maxReconnectAttempts` | `10` | Maximum reconnection attempts |
| `initialReconnectDelay` | `1000` | Initial delay between reconnects (ms) |
| `maxReconnectDelay` | `30000` | Maximum delay between reconnects (ms) |
| `pingInterval` | `30000` | Ping interval for keepalive (ms) |
| `connectionTimeout` | `10000` | Connection timeout (ms) |

## Reconnection Strategy

The client implements exponential backoff for reconnection:
- Starts with `initialReconnectDelay` (1s)
- Doubles delay each attempt up to `maxReconnectDelay` (30s)
- Stops after `maxReconnectAttempts` (10)
- Resets counter on successful connection

## API Reference

### DashboardWebSocket Class

#### Methods
- `connect()` - Connect to WebSocket server
- `disconnect()` - Disconnect from server
- `send(message)` - Send message to server
- `on(type, handler)` - Subscribe to message type
- `off(type, handler)` - Unsubscribe from message type
- `onConnectionStatusChange(callback)` - Subscribe to status changes
- `onError(callback)` - Subscribe to errors
- `isConnected()` - Check connection status
- `getConnectionStatus()` - Get current status
- `getReconnectionInfo()` - Get reconnection details
- `destroy()` - Clean up all resources

#### Static Functions
- `createWebSocketClient(config)` - Create configured client
- `webSocketClient` - Singleton instance

## Integration with Backend

The client connects to the FastAPI backend WebSocket endpoint at `/ws` and handles:
- Real-time trading data streaming
- AI decision notifications
- System status updates
- Error reporting
- Bidirectional communication

## Usage in Production

```typescript
import { createWebSocketClient } from './websocket';

const client = createWebSocketClient({
  url: process.env.WEBSOCKET_URL || 'ws://localhost:8000/ws',
  maxReconnectAttempts: 15,
  connectionTimeout: 15000
});

// Production error handling
client.onError((error) => {
  // Log to monitoring service
  console.error('WebSocket error:', error);
});

// Production status monitoring
client.onConnectionStatusChange((status) => {
  // Update UI connection indicator
  updateConnectionIndicator(status);
});

client.connect();
```

## Testing

The implementation has been tested for:
- TypeScript compilation without errors
- Proper message type handling
- Connection lifecycle management
- Error recovery scenarios
- Memory leak prevention

## Dependencies

- Native WebSocket API (browser built-in)
- TypeScript for type safety
- No external libraries required

## Browser Compatibility

- Chrome/Edge 16+
- Firefox 11+
- Safari 7+
- All modern browsers with WebSocket support