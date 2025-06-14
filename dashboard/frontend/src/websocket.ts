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

// Specific LLM event message types
export interface LLMRequestMessage {
  type: 'llm_request';
  data: {
    request_id: string;
    timestamp: string;
    model: string;
    prompt_tokens?: number;
    max_tokens?: number;
    temperature?: number;
    context?: {
      market_data?: any;
      indicators?: any;
      positions?: any;
    };
  };
}

export interface LLMResponseMessage {
  type: 'llm_response';
  data: {
    request_id: string;
    timestamp: string;
    model: string;
    response_time_ms: number;
    completion_tokens?: number;
    total_tokens?: number;
    cost_estimate_usd?: number;
    success: boolean;
    error?: string;
    raw_response?: string;
  };
}

export interface TradingDecisionMessage {
  type: 'trading_decision';
  data: {
    request_id: string;
    timestamp: string;
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    reasoning: string;
    price: number;
    quantity?: number;
    leverage?: number;
    indicators?: {
      cipher_a?: number;
      cipher_b?: number;
      wave_trend_1?: number;
      wave_trend_2?: number;
    };
    risk_analysis?: {
      stop_loss?: number;
      take_profit?: number;
      risk_reward_ratio?: number;
    };
  };
}

export interface LLMEventMessage {
  type: 'llm_event';
  event_type: 'llm_request' | 'llm_response' | 'trading_decision' | 'performance_metrics' | 'alert';
  timestamp: string;
  source: 'llm_parser';
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
  | LLMRequestMessage
  | LLMResponseMessage
  | TradingDecisionMessage
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
  private messageQueue: any[] = [];
  private maxQueueSize = 50; // Reduced from 100
  private lastPongTime: number = Date.now();
  private pongTimeout = 60000; // 60 seconds

  // Event system for message type routing
  private eventHandlers = new Map<string, Set<MessageHandler>>();
  
  // Connection status callbacks
  private connectionStatusCallbacks = new Set<(status: ConnectionStatus) => void>();
  private errorCallbacks = new Set<(error: Event | Error) => void>();

  // Memory management
  private readonly MAX_EVENT_HANDLERS = 20;
  private readonly MAX_STATUS_CALLBACKS = 10;
  private readonly MAX_ERROR_CALLBACKS = 10;
  private cleanupTimer: number | null = null;
  private readonly CLEANUP_INTERVAL = 30000; // 30 seconds

  // Message throttling for performance
  private messageThrottle = new Map<string, number>();
  private readonly MESSAGE_THROTTLE_MS = 50; // 50ms between same message types

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
    this.maxReconnectAttempts = config.maxReconnectAttempts || 5;
    this.reconnectDelay = config.initialReconnectDelay || 1000;
    this.maxReconnectDelay = config.maxReconnectDelay || 30000;
    this.pingIntervalMs = config.pingInterval || 30000;
    this.connectionTimeout = config.connectionTimeout || 10000;
    
    // Start memory cleanup
    this.startMemoryCleanup();
  }

  /**
   * Start periodic memory cleanup
   */
  private startMemoryCleanup(): void {
    this.cleanupTimer = window.setInterval(() => {
      this.performMemoryCleanup();
    }, this.CLEANUP_INTERVAL);
  }

  /**
   * Perform memory cleanup operations
   */
  private performMemoryCleanup(): void {
    // Clean up message queue
    if (this.messageQueue.length > this.maxQueueSize) {
      this.messageQueue = this.messageQueue.slice(-this.maxQueueSize);
    }

    // Clean up event handlers if too many
    if (this.eventHandlers.size > this.MAX_EVENT_HANDLERS) {
      // Keep only the most recently used handlers
      const entries = Array.from(this.eventHandlers.entries());
      const recentEntries = entries.slice(-this.MAX_EVENT_HANDLERS);
      this.eventHandlers = new Map(recentEntries);
    }

    // Clean up status callbacks
    if (this.connectionStatusCallbacks.size > this.MAX_STATUS_CALLBACKS) {
      const callbacks = Array.from(this.connectionStatusCallbacks);
      this.connectionStatusCallbacks = new Set(callbacks.slice(-this.MAX_STATUS_CALLBACKS));
    }

    // Clean up error callbacks
    if (this.errorCallbacks.size > this.MAX_ERROR_CALLBACKS) {
      const callbacks = Array.from(this.errorCallbacks);
      this.errorCallbacks = new Set(callbacks.slice(-this.MAX_ERROR_CALLBACKS));
    }

    // Clean up old message throttle entries
    const now = Date.now();
    for (const [key, timestamp] of this.messageThrottle.entries()) {
      if (now - timestamp > 10000) { // Remove entries older than 10 seconds
        this.messageThrottle.delete(key);
      }
    }
  }

  /**
   * Check if message should be throttled
   */
  private shouldThrottleMessage(messageType: string): boolean {
    const now = Date.now();
    const lastTime = this.messageThrottle.get(messageType);
    
    if (lastTime && now - lastTime < this.MESSAGE_THROTTLE_MS) {
      return true; // Throttle this message
    }
    
    this.messageThrottle.set(messageType, now);
    return false;
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
      return cleanUrl;
      
    } catch (error) {
      console.error('WebSocket URL validation failed:', error);
      // Fallback to default URL
      const fallbackUrl = this.getDefaultWebSocketUrl();
      return fallbackUrl;
    }
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.isManualClose = false;
    this.notifyConnectionStatus('connecting');

    // Debug logging (reduced verbosity)
    const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
    if (isDebugMode) {
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
      }

      this.ws = new WebSocket(this.url);
      if (isDebugMode) {
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
  public send(message: any, queueIfDisconnected = true): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        const jsonMessage = JSON.stringify(message);
        this.ws.send(jsonMessage);
        
        const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
        if (isDebugMode) {
        }
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.notifyError(error instanceof Error ? error : new Error('Failed to send message'));
        return false;
      }
    }
    
    // Queue message if disconnected and queueing is enabled
    if (queueIfDisconnected && !this.isManualClose) {
      if (this.messageQueue.length < this.maxQueueSize) {
        this.messageQueue.push(message);
        const isDebugMode = (import.meta.env.VITE_DEBUG === 'true') || (import.meta.env.VITE_LOG_LEVEL === 'debug');
        if (isDebugMode) {
        }
        return true;
      } else {
        console.warn('WebSocket message queue is full, dropping message');
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
      }
      this.clearConnectionTimeout();
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.notifyConnectionStatus('connected');
      this.startPingInterval();
      
      // Process queued messages
      if (this.messageQueue.length > 0) {
        const queuedCount = this.messageQueue.length;
        if (isDebugMode) {
        }
        
        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift();
          this.send(message, false); // Don't re-queue if send fails
        }
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const message: AllWebSocketMessages = JSON.parse(event.data);
        
        // Handle pong messages internally
        if (message.type === 'pong') {
          this.lastPongTime = Date.now();
          return;
        }
        
        // Handle ping messages (server-initiated ping)
        if (message.type === 'ping') {
          this.send({ type: 'pong', timestamp: new Date().toISOString() }, false);
          return;
        }
        
        // Check if message should be throttled
        if (this.shouldThrottleMessage(message.type)) {
          return; // Skip processing this message
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
    }
    
    // Special handling for llm_event messages - also route to sub-event handlers
    if (message.type === 'llm_event' && 'data' in message && message.data?.event_type) {
      const subEventHandlers = this.eventHandlers.get(`llm_event:${message.data.event_type}`);
      if (subEventHandlers && subEventHandlers.size > 0) {
        subEventHandlers.forEach(handler => {
          try {
            handler(message);
          } catch (error) {
            console.error(`Error in llm_event sub-handler for ${message.data.event_type}:`, error);
          }
        });
      }
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
    
    // Reset last pong time on start
    this.lastPongTime = Date.now();
    
    this.pingInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        // Check if we've received a pong recently
        const timeSinceLastPong = Date.now() - this.lastPongTime;
        if (timeSinceLastPong > this.pongTimeout) {
          console.error('WebSocket pong timeout - connection may be dead');
          this.ws.close();
          return;
        }
        
        this.send({ 
          type: 'ping', 
          timestamp: new Date().toISOString() 
        }, false); // Don't queue ping messages
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
    // Clear cleanup timer
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    
    // Disconnect WebSocket
    this.disconnect();
    
    // Clear all collections
    this.eventHandlers.clear();
    this.connectionStatusCallbacks.clear();
    this.errorCallbacks.clear();
    this.messageQueue = [];
    this.messageThrottle.clear();
    
    // Clear timeouts
    if (this.connectionTimeoutId) {
      clearTimeout(this.connectionTimeoutId);
      this.connectionTimeoutId = null;
    }
  }

  /**
   * Helper method to validate message structure
   */
  public static isValidMessage(message: any): message is AllWebSocketMessages {
    return message && 
           typeof message === 'object' && 
           typeof message.type === 'string' &&
           message.data !== undefined;
  }

  /**
   * Helper to parse LLM-specific messages
   */
  public static parseLLMMessage(message: any): LLMRequestMessage | LLMResponseMessage | TradingDecisionMessage | LLMEventMessage | null {
    if (!this.isValidMessage(message)) return null;
    
    switch (message.type) {
      case 'llm_request':
        return message as LLMRequestMessage;
      case 'llm_response':
        return message as LLMResponseMessage;
      case 'trading_decision':
        return message as TradingDecisionMessage;
      case 'llm_event':
        return message as LLMEventMessage;
      default:
        return null;
    }
  }

  /**
   * Type guard for specific message types
   */
  public static isLLMRequest(message: any): message is LLMRequestMessage {
    return message?.type === 'llm_request';
  }

  public static isLLMResponse(message: any): message is LLMResponseMessage {
    return message?.type === 'llm_response';
  }

  public static isTradingDecision(message: any): message is TradingDecisionMessage {
    return message?.type === 'trading_decision';
  }

  public static isLLMEvent(message: any): message is LLMEventMessage {
    return message?.type === 'llm_event';
  }

  /**
   * Get current queue size
   */
  public getQueueSize(): number {
    return this.messageQueue.length;
  }

  /**
   * Clear message queue
   */
  public clearQueue(): void {
    this.messageQueue = [];
  }

  /**
   * Extract LLM event data from llm_event message
   */
  public static extractLLMEventData(message: LLMEventMessage): LLMRequestMessage | LLMResponseMessage | TradingDecisionMessage | null {
    if (!message.data) return null;
    
    const eventData = message.data;
    const baseData = {
      timestamp: eventData.timestamp || message.timestamp,
    };
    
    switch (eventData.event_type) {
      case 'llm_request':
        return {
          type: 'llm_request',
          data: {
            ...baseData,
            request_id: eventData.request_id || '',
            model: eventData.model || '',
            prompt_tokens: eventData.prompt_length,
            max_tokens: eventData.max_tokens,
            temperature: eventData.temperature,
            context: eventData.market_context,
          }
        } as LLMRequestMessage;
        
      case 'llm_response':
        return {
          type: 'llm_response',
          data: {
            ...baseData,
            request_id: eventData.request_id || '',
            model: eventData.model || '',
            response_time_ms: eventData.response_time_ms || 0,
            completion_tokens: eventData.token_usage?.completion_tokens,
            total_tokens: eventData.token_usage?.total_tokens,
            cost_estimate_usd: eventData.cost_estimate_usd,
            success: eventData.success || false,
            error: eventData.error,
            raw_response: eventData.response_preview,
          }
        } as LLMResponseMessage;
        
      case 'trading_decision':
        return {
          type: 'trading_decision',
          data: {
            ...baseData,
            request_id: eventData.request_id || '',
            action: eventData.action as 'BUY' | 'SELL' | 'HOLD',
            confidence: eventData.size_pct || 0,
            reasoning: eventData.rationale || '',
            price: eventData.current_price || 0,
            indicators: eventData.indicators,
            risk_analysis: {
              risk_reward_ratio: eventData.risk_assessment ? 1.5 : undefined,
            },
          }
        } as TradingDecisionMessage;
        
      default:
        return null;
    }
  }

  /**
   * Wait for connection to be established
   */
  public async waitForConnection(timeout: number = 5000): Promise<boolean> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      if (this.isConnected()) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    return false;
  }

  /**
   * Subscribe to a specific LLM event sub-type
   */
  public onLLMEvent(eventType: 'llm_request' | 'llm_response' | 'trading_decision' | 'performance_metrics' | 'alert', handler: MessageHandler): void {
    this.on(`llm_event:${eventType}`, handler);
  }

  /**
   * Unsubscribe from a specific LLM event sub-type
   */
  public offLLMEvent(eventType: 'llm_request' | 'llm_response' | 'trading_decision' | 'performance_metrics' | 'alert', handler: MessageHandler): void {
    this.off(`llm_event:${eventType}`, handler);
  }
}

// Singleton instance with default configuration
export const webSocketClient = new DashboardWebSocket();

// Convenience function to create configured client
export function createWebSocketClient(config: WebSocketConfig = {}): DashboardWebSocket {
  return new DashboardWebSocket(config.url, config);
}

// Production-ready client factory with optimized settings
export function createProductionWebSocketClient(): DashboardWebSocket {
  return new DashboardWebSocket(undefined, {
    maxReconnectAttempts: 5,
    initialReconnectDelay: 1000,
    maxReconnectDelay: 30000,
    pingInterval: 30000,
    connectionTimeout: 10000,
  });
}

// Helper to validate and parse WebSocket messages
export function parseWebSocketMessage(data: string): AllWebSocketMessages | null {
  try {
    const message = JSON.parse(data);
    if (DashboardWebSocket.isValidMessage(message)) {
      return message as AllWebSocketMessages;
    }
    return null;
  } catch (error) {
    console.error('Failed to parse WebSocket message:', error);
    return null;
  }
}