import type { WebSocketMessage, ConnectionStatus } from './types.ts';

// Enhanced message types based on backend specifications
export interface TradingLoopMessage {
  type: 'trading_loop';
  data: {
    price: number;
    action: string;
    confidence: number;
    timestamp?: string;
    symbol?: string;
  };
}

export interface AIDecisionMessage {
  type: 'ai_decision';
  data: {
    action: string;
    reasoning: string;
    confidence?: number;
    timestamp?: string;
  };
}

export interface SystemStatusMessage {
  type: 'system_status';
  data: {
    status: string;
    health: boolean;
    errors: string[];
    timestamp?: string;
  };
}

export interface ErrorMessage {
  type: 'error';
  data: {
    message: string;
    level: string;
    timestamp?: string;
  };
}

export interface LLMEventMessage {
  type: 'llm_event';
  data: {
    event_type: 'llm_request' | 'llm_response' | 'trading_decision' | 'performance_metrics' | 'alert';
    timestamp: string;
    session_id?: string;
    request_id?: string;
    model?: string;
    response_time_ms?: number;
    cost_estimate_usd?: number;
    action?: string;
    rationale?: string;
    success?: boolean;
    error?: string;
    alert_level?: 'info' | 'warning' | 'critical';
    alert_category?: string;
    alert_message?: string;
    [key: string]: any;
  };
}

export interface PerformanceUpdateMessage {
  type: 'performance_update';
  data: {
    timestamp: string;
    avg_response_time_ms: number;
    success_rate: number;
    total_cost_usd: number;
    active_alerts: number;
    hourly_cost: number;
  };
}

export interface TradingViewDecisionMessage {
  type: 'tradingview_decision';
  data: {
    symbol: string;
    timestamp: string;
    decision: {
      decision: string;
      price: number;
      confidence: number;
      reasoning: string;
    };
  };
}

export interface PingMessage {
  type: 'ping';
  timestamp: string;
}

export interface PongMessage {
  type: 'pong';
  timestamp: string;
}

// Union type for all possible messages
export type AllWebSocketMessages = 
  | WebSocketMessage 
  | TradingLoopMessage 
  | AIDecisionMessage 
  | SystemStatusMessage 
  | ErrorMessage 
  | LLMEventMessage
  | PerformanceUpdateMessage
  | TradingViewDecisionMessage
  | PingMessage 
  | PongMessage;

// Event handler type
export type MessageHandler = (message: AllWebSocketMessages) => void;

// WebSocket client configuration
export interface WebSocketConfig {
  url?: string;
  maxReconnectAttempts?: number;
  initialReconnectDelay?: number;
  maxReconnectDelay?: number;
  pingInterval?: number;
  connectionTimeout?: number;
}

export class DashboardWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private reconnectDelay: number;
  private maxReconnectDelay: number;
  private pingInterval: number | null = null;
  private pingIntervalMs: number;
  private connectionTimeout: number;
  private isManualClose = false;
  private connectionTimeoutId: number | null = null;

  // Event system for message type routing
  private eventHandlers = new Map<string, Set<MessageHandler>>();
  
  // Connection status callbacks
  private connectionStatusCallbacks = new Set<(status: ConnectionStatus) => void>();
  private errorCallbacks = new Set<(error: Event | Error) => void>();

  constructor(url?: string, config: WebSocketConfig = {}) {
    // Use dynamic URL detection if no URL provided
    if (!url && !config.url) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      url = `${protocol}//${window.location.host}/ws`;
    }
    
    // Set the URL with proper validation
    let finalUrl = config.url || url || this.getDefaultWebSocketUrl();
    
    // Validate and clean up the URL
    finalUrl = this.validateAndCleanUrl(finalUrl);
    
    this.url = finalUrl;
    this.maxReconnectAttempts = config.maxReconnectAttempts || 10;
    this.reconnectDelay = config.initialReconnectDelay || 1000;
    this.maxReconnectDelay = config.maxReconnectDelay || 30000;
    this.pingIntervalMs = config.pingInterval || 30000;
    this.connectionTimeout = config.connectionTimeout || 10000;
  }

  /**
   * Get default WebSocket URL based on current environment
   */
  private getDefaultWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    
    // In development mode, use the current host (Vite will proxy)
    if (host.includes('localhost') || host.includes('127.0.0.1')) {
      return `${protocol}//${host}/ws`;
    }
    
    // In production, use current host
    return `${protocol}//${host}/ws`;
  }

  /**
   * Validate and clean up WebSocket URL
   */
  private validateAndCleanUrl(url: string): string {
    try {
      // Handle relative paths
      if (url.startsWith('/')) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        url = `${protocol}//${host}${url}`;
      }
      
      // Ensure URL starts with ws:// or wss://
      if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
        throw new Error(`Invalid WebSocket URL protocol: ${url}`);
      }
      
      // Parse URL to validate format
      const urlObj = new URL(url);
      
      // Ensure path exists (default to /ws if only host provided)
      if (!urlObj.pathname || urlObj.pathname === '/') {
        urlObj.pathname = '/ws';
      }
      
      const cleanUrl = urlObj.toString();
      console.log(`WebSocket URL validated and cleaned: ${url} -> ${cleanUrl}`);
      return cleanUrl;
      
    } catch (error) {
      console.error('WebSocket URL validation failed:', error);
      // Fallback to default URL
      const fallbackUrl = this.getDefaultWebSocketUrl();
      console.warn(`Using fallback WebSocket URL: ${fallbackUrl}`);
      return fallbackUrl;
    }
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    this.isManualClose = false;
    this.notifyConnectionStatus('connecting');

    // Debug logging (reduced verbosity)
    const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
    if (isDebugMode) {
      console.log(`WebSocket connecting to: ${this.url}`);
      console.log(`Browser WebSocket support: ${typeof WebSocket !== 'undefined'}`);
    }

    try {
      // Check if WebSocket is available
      if (typeof WebSocket === 'undefined') {
        throw new Error('WebSocket not available in this environment');
      }

      // Additional checks for WebSocket availability
      if (!window.WebSocket && !(window as any).MozWebSocket) {
        throw new Error('WebSocket not supported by this browser');
      }

      // Check if we're in a secure context if needed
      const isSecure = this.url.startsWith('wss://');
      if (isSecure && !window.isSecureContext) {
        console.warn('Secure WebSocket requested but not in secure context');
      }

      this.ws = new WebSocket(this.url);
      if (isDebugMode) {
        console.log(`WebSocket created, readyState: ${this.ws.readyState} (${this.getReadyStateText(this.ws.readyState)})`);
      }
      
      this.setupEventListeners();
      this.setupConnectionTimeout();
    } catch (error) {
      // More concise error logging
      const errorMessage = error instanceof Error ? error.message : 'Unknown WebSocket error';
      console.error('WebSocket connection failed:', errorMessage);
      
      // Provide user-friendly error messages
      let userErrorMessage = 'Failed to create WebSocket connection';
      if (error instanceof Error) {
        if (error.message.includes('not available')) {
          userErrorMessage = 'WebSocket not available - browser may not support WebSockets';
        } else if (error.message.includes('not supported')) {
          userErrorMessage = 'WebSocket not supported by this browser';
        } else {
          userErrorMessage = error.message;
        }
      }
      
      this.notifyError(new Error(userErrorMessage));
      this.notifyConnectionStatus('error');
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    this.isManualClose = true;
    this.clearPingInterval();
    this.clearConnectionTimeout();
    
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }
    
    this.notifyConnectionStatus('disconnected');
  }

  /**
   * Send a message to the server
   */
  public send(message: any): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        const jsonMessage = JSON.stringify(message);
        this.ws.send(jsonMessage);
        
        const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
        if (isDebugMode) {
          console.debug('WebSocket message sent:', message.type || 'unknown');
        }
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.notifyError(error instanceof Error ? error : new Error('Failed to send message'));
        return false;
      }
    }
    
    const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
    if (isDebugMode) {
      console.warn('WebSocket not connected, cannot send message');
    }
    return false;
  }

  /**
   * Subscribe to specific message types
   */
  public on(messageType: string, handler: MessageHandler): void {
    if (!this.eventHandlers.has(messageType)) {
      this.eventHandlers.set(messageType, new Set());
    }
    this.eventHandlers.get(messageType)!.add(handler);
  }

  /**
   * Unsubscribe from specific message types
   */
  public off(messageType: string, handler: MessageHandler): void {
    const handlers = this.eventHandlers.get(messageType);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.eventHandlers.delete(messageType);
      }
    }
  }

  /**
   * Subscribe to connection status changes
   */
  public onConnectionStatusChange(callback: (status: ConnectionStatus) => void): void {
    this.connectionStatusCallbacks.add(callback);
  }

  /**
   * Unsubscribe from connection status changes
   */
  public offConnectionStatusChange(callback: (status: ConnectionStatus) => void): void {
    this.connectionStatusCallbacks.delete(callback);
  }

  /**
   * Subscribe to errors
   */
  public onError(callback: (error: Event | Error) => void): void {
    this.errorCallbacks.add(callback);
  }

  /**
   * Unsubscribe from errors
   */
  public offError(callback: (error: Event | Error) => void): void {
    this.errorCallbacks.delete(callback);
  }

  /**
   * Check if client is connected
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get current connection status
   */
  public getConnectionStatus(): ConnectionStatus {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
      default:
        return 'disconnected';
    }
  }

  /**
   * Get reconnection info
   */
  public getReconnectionInfo() {
    return {
      attempts: this.reconnectAttempts,
      maxAttempts: this.maxReconnectAttempts,
      nextDelay: Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay),
      isReconnecting: this.reconnectAttempts > 0 && !this.isManualClose
    };
  }

  /**
   * Set up event listeners for the WebSocket
   */
  private setupEventListeners(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
      if (isDebugMode) {
        console.log('WebSocket connected successfully to', this.url);
      }
      this.clearConnectionTimeout();
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.notifyConnectionStatus('connected');
      this.startPingInterval();
    };

    this.ws.onmessage = (event) => {
      try {
        const message: AllWebSocketMessages = JSON.parse(event.data);
        
        // Handle pong messages internally
        if (message.type === 'pong') {
          const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
          if (isDebugMode) {
            console.debug('Received pong from server');
          }
          return;
        }
        
        // Route message to type-specific handlers
        this.routeMessage(message);
        
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        this.notifyError(new Error('Failed to parse WebSocket message'));
      }
    };

    this.ws.onclose = (event) => {
      const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
      
      // Only log close details if it's an unexpected close or in debug mode
      if (event.code !== 1000 || isDebugMode) {
        console.log(`WebSocket connection closed: code ${event.code}, reason: ${event.reason || 'none'}`);
      }
      
      this.clearPingInterval();
      this.clearConnectionTimeout();
      this.ws = null;
      
      if (!this.isManualClose) {
        this.notifyConnectionStatus('disconnected');
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      // More concise error logging - avoid dumping raw error objects
      console.error('WebSocket error occurred');
      
      const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
      if (isDebugMode) {
        console.error('WebSocket error details:', {
          readyState: this.ws?.readyState,
          url: this.url,
          error: error
        });
      }
      
      this.clearConnectionTimeout();
      this.notifyError(error);
      this.notifyConnectionStatus('error');
    };
  }

  /**
   * Route incoming messages to appropriate handlers
   */
  private routeMessage(message: AllWebSocketMessages): void {
    const handlers = this.eventHandlers.get(message.type);
    if (handlers && handlers.size > 0) {
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error(`Error in message handler for type ${message.type}:`, error);
        }
      });
    } else {
      console.debug('No handlers registered for message type:', message.type);
    }

    // Also emit to generic message handlers
    const genericHandlers = this.eventHandlers.get('*');
    if (genericHandlers) {
      genericHandlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error('Error in generic message handler:', error);
        }
      });
    }
  }

  /**
   * Notify all connection status callbacks
   */
  private notifyConnectionStatus(status: ConnectionStatus): void {
    this.connectionStatusCallbacks.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        console.error('Error in connection status callback:', error);
      }
    });
  }

  /**
   * Notify all error callbacks
   */
  private notifyError(error: Event | Error): void {
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (callbackError) {
        console.error('Error in error callback:', callbackError);
      }
    });
  }

  /**
   * Setup connection timeout
   */
  private setupConnectionTimeout(): void {
    this.clearConnectionTimeout();
    
    this.connectionTimeoutId = window.setTimeout(() => {
      if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
        console.error('WebSocket connection timeout');
        this.ws.close();
        this.notifyError(new Error('Connection timeout'));
        this.notifyConnectionStatus('error');
        this.scheduleReconnect();
      }
    }, this.connectionTimeout);
  }

  /**
   * Clear connection timeout
   */
  private clearConnectionTimeout(): void {
    if (this.connectionTimeoutId) {
      clearTimeout(this.connectionTimeoutId);
      this.connectionTimeoutId = null;
    }
  }

  /**
   * Schedule a reconnection attempt with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.isManualClose || this.reconnectAttempts >= this.maxReconnectAttempts) {
      const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
      if (isDebugMode) {
        console.log('Max reconnection attempts reached or manual close');
      }
      this.notifyConnectionStatus('error');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 
      this.maxReconnectDelay
    );
    
    // Only log every few attempts to reduce spam, or in debug mode
    const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
    if (isDebugMode || this.reconnectAttempts <= 3 || this.reconnectAttempts % 5 === 0) {
      console.log(`WebSocket reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(delay/1000)}s`);
    }
    
    setTimeout(() => {
      if (!this.isManualClose) {
        this.connect();
      }
    }, delay);
  }

  /**
   * Start sending periodic ping messages to keep connection alive
   */
  private startPingInterval(): void {
    this.clearPingInterval();
    
    this.pingInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ 
          type: 'ping', 
          timestamp: new Date().toISOString() 
        });
      } else {
        this.clearPingInterval();
      }
    }, this.pingIntervalMs);
  }

  /**
   * Clear the ping interval
   */
  private clearPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Get human-readable ready state text
   */
  private getReadyStateText(readyState: number): string {
    switch (readyState) {
      case WebSocket.CONNECTING: return 'CONNECTING';
      case WebSocket.OPEN: return 'OPEN';
      case WebSocket.CLOSING: return 'CLOSING';
      case WebSocket.CLOSED: return 'CLOSED';
      default: return 'UNKNOWN';
    }
  }

  /**
   * Clean up all resources
   */
  public destroy(): void {
    this.disconnect();
    this.eventHandlers.clear();
    this.connectionStatusCallbacks.clear();
    this.errorCallbacks.clear();
  }
}

// Singleton instance with default configuration
export const webSocketClient = new DashboardWebSocket();

// Convenience function to create configured client
export function createWebSocketClient(config: WebSocketConfig = {}): DashboardWebSocket {
  return new DashboardWebSocket(config.url, config);
}