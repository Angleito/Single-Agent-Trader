# Dashboard Resilience Features

This document describes the comprehensive error handling and resilience features added to the AI Trading Bot Dashboard.

## Overview

The dashboard has been enhanced with multiple layers of resilience to ensure it remains functional and user-friendly even when backend services are partially or completely unavailable.

## WebSocket Resilience (`websocket.ts`)

### Enhanced Error Handling
- **Malformed Message Recovery**: Automatically detects and recovers from corrupted JSON messages
- **Message Structure Validation**: Validates incoming messages and attempts to repair invalid structures
- **Error Boundaries**: Isolates message processing errors to prevent cascade failures
- **Message Type Inference**: Attempts to infer message types from data structure when type is missing

### Advanced Reconnection Logic
- **Exponential Backoff with Jitter**: Prevents thundering herd problems during reconnection
- **Fallback URLs**: Supports multiple WebSocket endpoints for redundancy
- **Persistent Reconnection**: Continues attempting to reconnect with longer delays in resilience mode
- **Connection Health Monitoring**: Tracks ping/pong timeouts and connection quality

### Message Retry System
- **Critical Message Retry**: Automatically retries important messages (ping, authentication)
- **Retry Queue Management**: Tracks retry attempts with configurable limits
- **Message Throttling**: Prevents message flooding during reconnection

### Configuration Options
```typescript
const client = createResilientWebSocketClient([
  'wss://backup1.example.com/ws',
  'wss://backup2.example.com/ws'
]);
```

## UI Resilience (`ui.ts`)

### Error Boundaries
- **Operation-Level Error Handling**: Each UI update operation is wrapped in error boundaries
- **Automatic Error Recovery**: Attempts to recover from errors using fallback data
- **Error Rate Limiting**: Prevents error spam by tracking error frequency per operation
- **User-Friendly Error Messages**: Shows meaningful messages instead of technical errors

### Data Validation and Fallback
- **Input Validation**: Validates all incoming data structures before processing
- **Fallback Data Cache**: Maintains cache of last known good data for each component
- **Graceful Degradation**: UI components continue to function with stale data when live data is unavailable
- **Data Staleness Detection**: Warns users when displayed data is outdated

### Offline Mode Support
- **Automatic Offline Detection**: Switches to offline mode when connection is lost
- **Cached Data Display**: Shows last known data with clear indicators
- **Offline Indicator**: Visual indicator when dashboard is in offline mode
- **Seamless Recovery**: Automatically switches back to live mode when connection is restored

### Safe Data Access
- **Type-Safe Conversions**: Safe number/string conversion functions that handle edge cases
- **Null/Undefined Handling**: Graceful handling of missing or null data
- **NaN/Infinity Protection**: Prevents display of invalid numeric values

## API Resilience (`api-retry.ts`)

### Circuit Breaker Pattern
- **Failure Threshold**: Opens circuit after configurable number of failures
- **Recovery Timeout**: Automatically attempts recovery after timeout period
- **Half-Open State**: Tests service recovery with limited requests

### Retry Logic
- **Exponential Backoff**: Increases delay between retry attempts
- **Jitter**: Adds randomness to prevent synchronized retries
- **Configurable Conditions**: Customizable retry conditions based on error type
- **Maximum Attempts**: Configurable maximum retry attempts

### Intelligent Caching
- **TTL-Based Cache**: Time-based cache expiration
- **Stale Data Fallback**: Returns stale cache data when API is unavailable
- **Cache Cleanup**: Automatic cleanup of expired cache entries

### Request Timeouts
- **Configurable Timeouts**: Per-request timeout configuration
- **Abort Controller**: Proper request cancellation
- **Timeout Error Handling**: Specific handling for timeout scenarios

## User Experience Features

### Visual Feedback
- **Loading States**: Clear loading indicators for all data fetching
- **Error States**: User-friendly error messages with retry options
- **Offline Indicators**: Visual indicators when in offline mode
- **Data Age Warnings**: Warnings when data is stale

### Notifications System
- **Toast Notifications**: Non-intrusive notifications for status changes
- **Auto-Dismiss**: Automatic dismissal after timeout
- **Close Buttons**: Manual dismissal option
- **Notification Types**: Info, warning, and error notification styles

### Graceful Degradation
- **Partial Functionality**: Dashboard continues to work with limited data
- **Missing Data Warnings**: Clear indicators when specific data is unavailable
- **Fallback Displays**: Sensible defaults when data is missing

## Configuration

### WebSocket Configuration
```typescript
export interface WebSocketConfig {
  url?: string;
  maxReconnectAttempts?: number;
  initialReconnectDelay?: number;
  maxReconnectDelay?: number;
  pingInterval?: number;
  connectionTimeout?: number;
  enableResilience?: boolean;
  fallbackUrls?: string[];
  maxMessageRetries?: number;
  messageRetryDelay?: number;
}
```

### API Retry Configuration
```typescript
export interface RetryConfig {
  maxAttempts?: number;
  baseDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
  jitter?: boolean;
  retryCondition?: (error: any) => boolean;
  onRetry?: (attempt: number, error: any) => void;
  timeout?: number;
}
```

### Circuit Breaker Configuration
```typescript
export interface CircuitBreakerConfig {
  failureThreshold?: number;
  recoveryTimeout?: number;
  monitoringPeriod?: number;
}
```

## Usage Examples

### Creating a Resilient WebSocket Client
```typescript
import { createResilientWebSocketClient } from './websocket';

const wsClient = createResilientWebSocketClient([
  'wss://backup1.example.com/ws',
  'wss://backup2.example.com/ws'
]);

// Monitor connection health
const health = wsClient.getConnectionHealth();
console.log('Connection health:', health);
```

### Using API with Retry
```typescript
import { apiGet, apiWithFallback } from './api-retry';

// Simple API call with caching
const marketData = await apiGet('/api/market-data', 'market-cache', 30000);

// API call with fallback data
const fallbackData = { price: 0, symbol: 'UNKNOWN' };
const result = await apiWithFallback(
  () => apiGet('/api/market-data'),
  fallbackData,
  'Market data unavailable, showing placeholder'
);
```

### Error Boundary Usage in UI
```typescript
// Error boundaries are automatically applied to all update methods
dashboard.updateMarketData(data); // Automatically wrapped with error handling
dashboard.updateBotStatus(status); // Graceful failure with fallback data
```

## Monitoring and Debugging

### Health Checks
- WebSocket connection health monitoring
- API circuit breaker status
- Cache hit/miss ratios
- Error boundary statistics

### Logging
- Structured error logging with component identification
- Retry attempt logging
- Circuit breaker state changes
- Offline mode transitions

### Debug Information
- Connection quality metrics
- Message queue sizes
- Cache statistics
- Error boundary trigger counts

## Best Practices

### When to Use Fallback Data
- Connection is temporarily lost
- API returns errors but cached data exists
- Partial system failure where some data is unavailable

### Error Message Guidelines
- Show user-friendly messages, not technical errors
- Provide actionable information when possible
- Include estimated recovery time if available
- Offer manual retry options for user control

### Performance Considerations
- Cache cleanup runs automatically every 5 minutes
- Message throttling prevents overwhelming slow connections
- DOM updates are batched for better performance
- Memory cleanup prevents resource leaks

### Testing Resilience
- Test with network disconnection
- Test with malformed WebSocket messages
- Test with API timeouts and 5xx errors
- Test with partial data availability
- Verify fallback data is displayed correctly
- Confirm recovery when services come back online

## Future Enhancements

### Potential Improvements
- Service worker for true offline functionality
- IndexedDB for persistent local storage
- Progressive Web App features
- Push notifications for critical alerts
- Advanced analytics and metrics collection

### Configuration Persistence
- Save user preferences for error handling behavior
- Persistent cache across browser sessions
- Customizable notification preferences
- Offline mode preferences