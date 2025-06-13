// Example usage of the DashboardWebSocket client
// This file demonstrates how to use the WebSocket client for real-time communication

import { 
  DashboardWebSocket, 
  createWebSocketClient,
  type TradingLoopMessage,
  type AIDecisionMessage,
  type SystemStatusMessage,
  type ErrorMessage
} from './websocket';
import type { ConnectionStatus } from './types';

// Example 1: Using the singleton instance
export async function useSingletonWebSocket() {
  const { webSocketClient } = await import('./websocket');
  
  // Connect to the backend
  webSocketClient.connect();
  
  // Subscribe to connection status changes
  webSocketClient.onConnectionStatusChange((status) => {
    console.log(`Connection status changed: ${status}`);
  });
  
  // Subscribe to trading loop messages
  webSocketClient.on('trading_loop', (message) => {
    const msg = message as TradingLoopMessage;
    console.log(`Trading signal: ${msg.data.action} at $${msg.data.price} (confidence: ${msg.data.confidence})`);
  });
  
  // Subscribe to AI decisions
  webSocketClient.on('ai_decision', (message) => {
    const msg = message as AIDecisionMessage;
    console.log(`AI Decision: ${msg.data.action} - ${msg.data.reasoning}`);
  });
  
  // Subscribe to system status updates
  webSocketClient.on('system_status', (message) => {
    const msg = message as SystemStatusMessage;
    console.log(`System status: ${msg.data.status} (healthy: ${msg.data.health})`);
  });
  
  // Subscribe to errors
  webSocketClient.on('error', (message) => {
    const msg = message as ErrorMessage;
    console.error(`System error: ${msg.data.message} (level: ${msg.data.level})`);
  });
  
  // Subscribe to all messages with wildcard
  webSocketClient.on('*', (message) => {
    console.log('Received message:', message);
  });
  
  return webSocketClient;
}

// Example 2: Creating a custom WebSocket client with configuration
export function useCustomWebSocketClient() {
  const client = createWebSocketClient({
    // Use dynamic URL detection by not specifying a URL
    maxReconnectAttempts: 15,
    initialReconnectDelay: 2000,
    maxReconnectDelay: 60000,
    pingInterval: 15000,
    connectionTimeout: 15000
  });
  
  // Error handling
  client.onError((error) => {
    console.error('WebSocket error:', error);
  });
  
  // Connection status monitoring
  client.onConnectionStatusChange((status) => {
    switch (status) {
      case 'connecting':
        console.log('Connecting to trading bot...');
        break;
      case 'connected':
        console.log('Connected to trading bot successfully!');
        break;
      case 'disconnected':
        console.log('Disconnected from trading bot');
        break;
      case 'error':
        console.error('Connection error');
        break;
    }
  });
  
  // Start the connection
  client.connect();
  
  return client;
}

// Example 3: React-style hook for WebSocket integration
export function useWebSocketData() {
  const client = new DashboardWebSocket();
  
  // State to track data
  const state = {
    connectionStatus: 'disconnected' as ConnectionStatus,
    latestTradingSignal: null as TradingLoopMessage['data'] | null,
    latestAIDecision: null as AIDecisionMessage['data'] | null,
    systemStatus: null as SystemStatusMessage['data'] | null,
    errors: [] as ErrorMessage['data'][]
  };
  
  // Set up event handlers
  client.onConnectionStatusChange((status) => {
    state.connectionStatus = status;
  });
  
  client.on('trading_loop', (message) => {
    const msg = message as TradingLoopMessage;
    state.latestTradingSignal = msg.data;
  });
  
  client.on('ai_decision', (message) => {
    const msg = message as AIDecisionMessage;
    state.latestAIDecision = msg.data;
  });
  
  client.on('system_status', (message) => {
    const msg = message as SystemStatusMessage;
    state.systemStatus = msg.data;
  });
  
  client.on('error', (message) => {
    const msg = message as ErrorMessage;
    state.errors.push(msg.data);
    // Keep only last 10 errors
    if (state.errors.length > 10) {
      state.errors.shift();
    }
  });
  
  // Connect immediately
  client.connect();
  
  return {
    client,
    state,
    // Utility functions
    isConnected: () => client.isConnected(),
    getReconnectionInfo: () => client.getReconnectionInfo(),
    disconnect: () => client.disconnect(),
    reconnect: () => {
      client.disconnect();
      setTimeout(() => client.connect(), 1000);
    },
    cleanup: () => client.destroy()
  };
}

// Example 4: Production-ready wrapper with error recovery
export class RobustWebSocketClient {
  private client: DashboardWebSocket;
  private reconnectOnError: boolean;
  private maxErrorCount: number;
  private errorCount: number = 0;
  private lastErrorTime: number = 0;
  
  constructor(
    url?: string,
    options: {
      reconnectOnError?: boolean;
      maxErrorCount?: number;
    } = {}
  ) {
    this.reconnectOnError = options.reconnectOnError ?? true;
    this.maxErrorCount = options.maxErrorCount ?? 5;
    
    const config: any = {
      maxReconnectAttempts: 20,
      initialReconnectDelay: 1000,
      maxReconnectDelay: 30000,
      pingInterval: 30000
    };
    if (url) {
      config.url = url;
    }
    this.client = createWebSocketClient(config);
    
    this.setupErrorRecovery();
  }
  
  private setupErrorRecovery() {
    this.client.onError((error) => {
      const now = Date.now();
      
      // Reset error count if it's been more than 5 minutes since last error
      if (now - this.lastErrorTime > 5 * 60 * 1000) {
        this.errorCount = 0;
      }
      
      this.errorCount++;
      this.lastErrorTime = now;
      
      console.error(`WebSocket error #${this.errorCount}:`, error);
      
      // If too many errors in short time, give up on reconnection
      if (this.errorCount >= this.maxErrorCount) {
        console.error('Too many WebSocket errors, stopping reconnection attempts');
        this.client.disconnect();
        return;
      }
      
      // Try to reconnect on error if enabled
      if (this.reconnectOnError && !this.client.isConnected()) {
        setTimeout(() => {
          if (!this.client.isConnected()) {
            console.log('Attempting error recovery reconnection...');
            this.client.connect();
          }
        }, 5000);
      }
    });
  }
  
  // Proxy methods to the underlying client
  connect() {
    this.errorCount = 0; // Reset error count on manual connect
    return this.client.connect();
  }
  
  disconnect() {
    return this.client.disconnect();
  }
  
  on(messageType: string, handler: any) {
    return this.client.on(messageType, handler);
  }
  
  off(messageType: string, handler: any) {
    return this.client.off(messageType, handler);
  }
  
  onConnectionStatusChange(callback: any) {
    return this.client.onConnectionStatusChange(callback);
  }
  
  isConnected() {
    return this.client.isConnected();
  }
  
  getStatus() {
    return {
      connected: this.client.isConnected(),
      status: this.client.getConnectionStatus(),
      errorCount: this.errorCount,
      reconnectionInfo: this.client.getReconnectionInfo()
    };
  }
  
  destroy() {
    return this.client.destroy();
  }
}