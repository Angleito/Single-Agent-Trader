import type { WebSocketMessage, ConnectionStatus } from './types.ts'

// Enhanced message types based on backend specifications
export interface TradingLoopMessage {
  type: 'trading_loop'
  data: {
    price: number
    action: string
    confidence: number
    timestamp?: string
    symbol?: string
  }
}

export interface AIDecisionMessage {
  type: 'ai_decision'
  data: {
    action: string
    reasoning: string
    confidence?: number
    timestamp?: string
  }
}

export interface SystemStatusMessage {
  type: 'system_status'
  data: {
    status: string
    health: boolean
    errors: string[]
    timestamp?: string
  }
}

export interface ErrorMessage {
  type: 'error'
  data: {
    message: string
    level: string
    timestamp?: string
  }
}

// Specific LLM event message types
export interface LLMRequestMessage {
  type: 'llm_request'
  data: {
    request_id: string
    timestamp: string
    model: string
    prompt_tokens?: number
    max_tokens?: number
    temperature?: number
    context?: {
      market_data?: any
      indicators?: any
      positions?: any
    }
  }
}

export interface LLMResponseMessage {
  type: 'llm_response'
  data: {
    request_id: string
    timestamp: string
    model: string
    response_time_ms: number
    completion_tokens?: number
    total_tokens?: number
    cost_estimate_usd?: number
    success: boolean
    error?: string
    raw_response?: string
  }
}

export interface TradingDecisionMessage {
  type: 'trading_decision'
  data: {
    request_id: string
    timestamp: string
    action: 'BUY' | 'SELL' | 'HOLD'
    confidence: number
    reasoning: string
    price: number
    quantity?: number
    leverage?: number
    indicators?: {
      cipher_a?: number
      cipher_b?: number
      wave_trend_1?: number
      wave_trend_2?: number
    }
    risk_analysis?: {
      stop_loss?: number
      take_profit?: number
      risk_reward_ratio?: number
    }
  }
}

export interface LLMEventMessage {
  type: 'llm_event'
  event_type: 'llm_request' | 'llm_response' | 'trading_decision' | 'performance_metrics' | 'alert'
  timestamp: string
  source: 'llm_parser'
  data: {
    event_type:
      | 'llm_request'
      | 'llm_response'
      | 'trading_decision'
      | 'performance_metrics'
      | 'alert'
    timestamp: string
    session_id?: string
    request_id?: string
    model?: string
    response_time_ms?: number
    cost_estimate_usd?: number
    action?: string
    rationale?: string
    success?: boolean
    error?: string
    alert_level?: 'info' | 'warning' | 'critical'
    alert_category?: string
    alert_message?: string
    [key: string]: any
  }
}

export interface PerformanceUpdateMessage {
  type: 'performance_update'
  data: {
    timestamp: string
    avg_response_time_ms: number
    success_rate: number
    total_cost_usd: number
    active_alerts: number
    hourly_cost: number
  }
}

export interface TradingViewDecisionMessage {
  type: 'tradingview_decision'
  data: {
    symbol: string
    timestamp: string
    decision: {
      decision: string
      price: number
      confidence: number
      reasoning: string
    }
  }
}

export interface PingMessage {
  type: 'ping'
  timestamp: string
}

export interface PongMessage {
  type: 'pong'
  timestamp: string
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
  | PongMessage

// Event handler type
export type MessageHandler = (message: AllWebSocketMessages) => void

// WebSocket client configuration
export interface WebSocketConfig {
  url?: string
  maxReconnectAttempts?: number
  initialReconnectDelay?: number
  maxReconnectDelay?: number
  pingInterval?: number
  connectionTimeout?: number
  enableResilience?: boolean
  fallbackUrls?: string[]
  maxMessageRetries?: number
  messageRetryDelay?: number
}

export class DashboardWebSocket {
  private ws: WebSocket | null = null
  private url: string
  private fallbackUrls: string[] = []
  private currentUrlIndex = 0
  private reconnectAttempts = 0
  private maxReconnectAttempts: number
  private reconnectDelay: number
  private maxReconnectDelay: number
  private pingInterval: number | null = null
  private pingIntervalMs: number
  private connectionTimeout: number
  private isManualClose = false
  private connectionTimeoutId: number | null = null
  private messageQueue: any[] = []
  private maxQueueSize = 50
  private lastPongTime: number = Date.now()
  private pongTimeout = 60000
  private enableResilience: boolean
  private maxMessageRetries: number
  private messageRetryDelay: number
  private retryQueue = new Map<string, { message: any; attempts: number; maxRetries: number }>()

  // Event system for message type routing
  private eventHandlers = new Map<string, Set<MessageHandler>>()

  // Connection status callbacks
  private connectionStatusCallbacks = new Set<(status: ConnectionStatus) => void>()
  private errorCallbacks = new Set<(error: Event | Error) => void>()

  // Memory management
  private readonly MAX_EVENT_HANDLERS = 20
  private readonly MAX_STATUS_CALLBACKS = 10
  private readonly MAX_ERROR_CALLBACKS = 10
  private cleanupTimer: number | null = null
  private readonly CLEANUP_INTERVAL = 30000 // 30 seconds

  // Message throttling for performance
  private messageThrottle = new Map<string, number>()
  private readonly MESSAGE_THROTTLE_MS = 50 // 50ms between same message types

  constructor(url?: string, config: WebSocketConfig = {}) {
    // Check for runtime configuration first
    const runtimeWsUrl = (window as any).__WS_URL__ || (window as any).__RUNTIME_CONFIG__?.WS_URL
    
    // Use dynamic URL detection if no URL provided
    if (!url && !config.url && !runtimeWsUrl) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      url = `${protocol}//${window.location.host}/ws`
    }

    // Priority: explicit config.url > constructor url > runtime config > default
    let finalUrl = config.url ?? url ?? runtimeWsUrl ?? this.getDefaultWebSocketUrl()

    // Validate and clean up the URL
    finalUrl = this.validateAndCleanUrl(finalUrl)

    this.url = finalUrl
    this.fallbackUrls = config.fallbackUrls ?? []
    this.maxReconnectAttempts = config.maxReconnectAttempts ?? 5
    this.reconnectDelay = config.initialReconnectDelay ?? 1000
    this.maxReconnectDelay = config.maxReconnectDelay ?? 30000
    this.pingIntervalMs = config.pingInterval ?? 30000
    this.connectionTimeout = config.connectionTimeout ?? 10000
    this.enableResilience = config.enableResilience !== false
    this.maxMessageRetries = config.maxMessageRetries ?? 3
    this.messageRetryDelay = config.messageRetryDelay ?? 1000

    // Start memory cleanup
    this.startMemoryCleanup()
  }

  /**
   * Start periodic memory cleanup
   */
  private startMemoryCleanup(): void {
    this.cleanupTimer = window.setInterval(() => {
      this.performMemoryCleanup()
    }, this.CLEANUP_INTERVAL)
  }

  /**
   * Perform memory cleanup operations
   */
  private performMemoryCleanup(): void {
    // Clean up message queue
    if (this.messageQueue.length > this.maxQueueSize) {
      this.messageQueue = this.messageQueue.slice(-this.maxQueueSize)
    }

    // Clean up event handlers if too many
    if (this.eventHandlers.size > this.MAX_EVENT_HANDLERS) {
      // Keep only the most recently used handlers
      const entries = Array.from(this.eventHandlers.entries())
      const recentEntries = entries.slice(-this.MAX_EVENT_HANDLERS)
      this.eventHandlers = new Map(recentEntries)
    }

    // Clean up status callbacks
    if (this.connectionStatusCallbacks.size > this.MAX_STATUS_CALLBACKS) {
      const callbacks = Array.from(this.connectionStatusCallbacks)
      this.connectionStatusCallbacks = new Set(callbacks.slice(-this.MAX_STATUS_CALLBACKS))
    }

    // Clean up error callbacks
    if (this.errorCallbacks.size > this.MAX_ERROR_CALLBACKS) {
      const callbacks = Array.from(this.errorCallbacks)
      this.errorCallbacks = new Set(callbacks.slice(-this.MAX_ERROR_CALLBACKS))
    }

    // Clean up old message throttle entries
    const now = Date.now()
    for (const [key, timestamp] of this.messageThrottle.entries()) {
      if (now - timestamp > 10000) {
        // Remove entries older than 10 seconds
        this.messageThrottle.delete(key)
      }
    }

    // Clean up old retry queue entries
    for (const [messageId, queued] of this.retryQueue.entries()) {
      if (queued.attempts >= queued.maxRetries) {
        this.retryQueue.delete(messageId)
      }
    }
  }

  /**
   * Check if message should be throttled
   */
  private shouldThrottleMessage(messageType: string): boolean {
    const now = Date.now()
    const lastTime = this.messageThrottle.get(messageType)

    if (lastTime && now - lastTime < this.MESSAGE_THROTTLE_MS) {
      return true // Throttle this message
    }

    this.messageThrottle.set(messageType, now)
    return false
  }

  /**
   * Get default WebSocket URL based on current environment with Docker support
   */
  private getDefaultWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const hostname = window.location.hostname
    const port = window.location.port

    // Smart environment detection for Docker scenarios
    const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1'
    const isDevPort = port === '3000' || port === '3001' || port === '5173'
    const isNginxPort = port === '8080'
    
    // Production nginx proxy scenario (port 8080)
    if (isNginxPort) {
      // Use relative /api/ws path for nginx proxy routing
      return `${protocol}//${host}/api/ws`
    }

    // Development scenarios with Docker backend
    if (isDevPort && isLocalhost) {
      // In development, connect to backend container port
      const backendPort = '8000'  // Dashboard backend port from docker-compose
      return `${protocol}//${hostname}:${backendPort}/ws`
    }

    // Docker container frontend or localhost scenarios
    if (isLocalhost) {
      // Check if we're likely running in a containerized environment
      const isLikelyContainerized = (
        // Check for Docker-specific indicators
        (window as any).__DOCKER_ENV__ ||
        // Environment variable indicating containerization
        import.meta.env.VITE_DOCKER_ENV ||
        // Port patterns suggesting containerization
        port === '8080' || isDevPort
      )

      if (isLikelyContainerized) {
        // Frontend container needs to reach backend container via host networking
        const backendPort = '8000'  // Dashboard backend port exposed to host
        return `${protocol}//${hostname}:${backendPort}/ws`
      }
    }

    // Default fallback - use current host with /ws path
    return `${protocol}//${host}/ws`
  }

  /**
   * Validate and clean up WebSocket URL
   */
  private validateAndCleanUrl(url: string): string {
    try {
      // Handle relative paths
      if (url.startsWith('/')) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const host = window.location.host
        url = `${protocol}//${host}${url}`
      }

      // Ensure URL starts with ws:// or wss://
      if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
        throw new Error(`Invalid WebSocket URL protocol: ${url}`)
      }

      // Parse URL to validate format
      const urlObj = new URL(url)

      // Ensure path exists (default to /ws if only host provided)
      if (!urlObj.pathname || urlObj.pathname === '/') {
        urlObj.pathname = '/ws'
      }

      const cleanUrl = urlObj.toString()
      return cleanUrl
    } catch (error) {
      console.error('WebSocket URL validation failed:', error)
      // Fallback to default URL
      const fallbackUrl = this.getDefaultWebSocketUrl()
      return fallbackUrl
    }
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return
    }

    this.isManualClose = false
    this.notifyConnectionStatus('connecting')

    // Debug logging (reduced verbosity)
    const isDebugMode =
      import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
    if (isDebugMode) {
    }

    try {
      // Check if WebSocket is available
      if (typeof WebSocket === 'undefined') {
        throw new Error('WebSocket not available in this environment')
      }

      // Additional checks for WebSocket availability
      if (!window.WebSocket && !(window as any).MozWebSocket) {
        throw new Error('WebSocket not supported by this browser')
      }

      // Check if we're in a secure context if needed
      const isSecure = this.url.startsWith('wss://')
      if (isSecure && !window.isSecureContext) {
      }

      this.ws = new WebSocket(this.url)
      if (isDebugMode) {
      }

      this.setupEventListeners()
      this.setupConnectionTimeout()
    } catch (error) {
      // More concise error logging
      const errorMessage = error instanceof Error ? error.message : 'Unknown WebSocket error'
      console.error('WebSocket connection failed:', errorMessage)

      // Provide user-friendly error messages
      let userErrorMessage = 'Failed to create WebSocket connection'
      if (error instanceof Error) {
        if (error.message.includes('not available')) {
          userErrorMessage = 'WebSocket not available - browser may not support WebSockets'
        } else if (error.message.includes('not supported')) {
          userErrorMessage = 'WebSocket not supported by this browser'
        } else {
          userErrorMessage = error.message
        }
      }

      this.notifyError(new Error(userErrorMessage))
      this.notifyConnectionStatus('error')
      this.scheduleReconnect()
    }
  }

  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    this.isManualClose = true
    this.clearPingInterval()
    this.clearConnectionTimeout()

    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect')
      this.ws = null
    }

    this.notifyConnectionStatus('disconnected')
  }

  /**
   * Send a message to the server with retry capability
   */
  public send(message: any, queueIfDisconnected = true): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        const jsonMessage = JSON.stringify(message)
        this.ws.send(jsonMessage)

        const isDebugMode =
          import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
        if (isDebugMode) {
          console.log('Message sent:', message.type)
        }
        return true
      } catch (error) {
        console.error('Failed to send WebSocket message:', error)

        // Retry critical messages if resilience is enabled
        if (this.enableResilience && this.isCriticalMessage(message)) {
          this.queueForRetry(message)
        }

        this.notifyError(error instanceof Error ? error : new Error('Failed to send message'))
        return false
      }
    }

    // Queue message if disconnected and queueing is enabled
    if (queueIfDisconnected && !this.isManualClose) {
      if (this.messageQueue.length < this.maxQueueSize) {
        this.messageQueue.push(message)
        const isDebugMode =
          import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
        if (isDebugMode) {
          console.log('Message queued:', message.type)
        }
        return true
      } else {
        console.warn('WebSocket message queue is full, dropping message')
      }
    }

    const isDebugMode =
      import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
    if (isDebugMode) {
      console.warn('WebSocket not connected, cannot send message')
    }
    return false
  }

  /**
   * Subscribe to specific message types
   */
  public on(messageType: string, handler: MessageHandler): void {
    if (!this.eventHandlers.has(messageType)) {
      this.eventHandlers.set(messageType, new Set())
    }
    this.eventHandlers.get(messageType)!.add(handler)
  }

  /**
   * Unsubscribe from specific message types
   */
  public off(messageType: string, handler: MessageHandler): void {
    const handlers = this.eventHandlers.get(messageType)
    if (handlers) {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.eventHandlers.delete(messageType)
      }
    }
  }

  /**
   * Subscribe to connection status changes
   */
  public onConnectionStatusChange(callback: (status: ConnectionStatus) => void): void {
    this.connectionStatusCallbacks.add(callback)
  }

  /**
   * Unsubscribe from connection status changes
   */
  public offConnectionStatusChange(callback: (status: ConnectionStatus) => void): void {
    this.connectionStatusCallbacks.delete(callback)
  }

  /**
   * Subscribe to errors
   */
  public onError(callback: (error: Event | Error) => void): void {
    this.errorCallbacks.add(callback)
  }

  /**
   * Unsubscribe from errors
   */
  public offError(callback: (error: Event | Error) => void): void {
    this.errorCallbacks.delete(callback)
  }

  /**
   * Check if client is connected
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  /**
   * Get current connection status
   */
  public getConnectionStatus(): ConnectionStatus {
    if (!this.ws) return 'disconnected'

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting'
      case WebSocket.OPEN:
        return 'connected'
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
      default:
        return 'disconnected'
    }
  }

  /**
   * Get reconnection info
   */
  public getReconnectionInfo() {
    return {
      attempts: this.reconnectAttempts,
      maxAttempts: this.maxReconnectAttempts,
      nextDelay: Math.min(
        this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
        this.maxReconnectDelay
      ),
      isReconnecting: this.reconnectAttempts > 0 && !this.isManualClose,
    }
  }

  /**
   * Set up event listeners for the WebSocket
   */
  private setupEventListeners(): void {
    if (!this.ws) return

    this.ws.onopen = () => {
      const isDebugMode =
        import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
      if (isDebugMode) {
      }
      this.clearConnectionTimeout()
      this.reconnectAttempts = 0
      this.reconnectDelay = 1000
      this.notifyConnectionStatus('connected')
      this.startPingInterval()

      // Process queued messages
      if (this.messageQueue.length > 0) {
        const _queuedCount = this.messageQueue.length
        if (isDebugMode) {
        }

        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift()
          this.send(message, false) // Don't re-queue if send fails
        }
      }
    }

    this.ws.onmessage = (event) => {
      try {
        // Validate message data before parsing
        if (!event.data || typeof event.data !== 'string') {
          console.warn('Received invalid WebSocket message data:', event.data)
          return
        }

        let message: AllWebSocketMessages
        try {
          message = JSON.parse(event.data)
        } catch (parseError) {
          console.error('Failed to parse WebSocket message JSON:', parseError)
          console.warn(
            'Raw message data:',
            event.data.substring(0, 200) + (event.data.length > 200 ? '...' : '')
          )

          // Try to recover from malformed JSON
          if (this.enableResilience) {
            this.handleMalformedMessage(event.data)
          }
          return
        }

        // Validate message structure
        if (!this.validateMessageStructure(message)) {
          console.warn('Received message with invalid structure:', message)
          if (this.enableResilience) {
            this.handleInvalidMessage(message)
          }
          return
        }

        // Handle pong messages internally
        if (message.type === 'pong') {
          this.lastPongTime = Date.now()
          return
        }

        // Handle ping messages (server-initiated ping)
        if (message.type === 'ping') {
          this.send({ type: 'pong', timestamp: new Date().toISOString() }, false)
          return
        }

        // Check if message should be throttled
        if (this.shouldThrottleMessage(message.type)) {
          return // Skip processing this message
        }

        // Route message to type-specific handlers with error boundaries
        this.routeMessageSafely(message)
      } catch (error) {
        console.error('Critical error in WebSocket message handler:', error)
        this.notifyError(
          new Error(
            `WebSocket message handling failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          )
        )

        // Don't disconnect on message handling errors if resilience is enabled
        if (!this.enableResilience) {
          this.handleCriticalError(error)
        }
      }
    }

    this.ws.onclose = (event) => {
      const isDebugMode =
        import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'

      // Only log close details if it's an unexpected close or in debug mode
      if (event.code !== 1000 || isDebugMode) {
      }

      this.clearPingInterval()
      this.clearConnectionTimeout()
      this.ws = null

      if (!this.isManualClose) {
        this.notifyConnectionStatus('disconnected')
        this.scheduleReconnect()
      }
    }

    this.ws.onerror = (error) => {
      // More concise error logging - avoid dumping raw error objects
      console.error('WebSocket error occurred')

      const isDebugMode =
        import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
      if (isDebugMode) {
        console.error('WebSocket error details:', {
          readyState: this.ws?.readyState,
          url: this.url,
          error: error,
        })
      }

      this.clearConnectionTimeout()
      this.notifyError(error)
      this.notifyConnectionStatus('error')
    }
  }

  /**
   * Route incoming messages to appropriate handlers with enhanced error boundaries
   */
  private routeMessageSafely(message: AllWebSocketMessages): void {
    try {
      this.routeMessage(message)
    } catch (error) {
      console.error(`Critical error routing message of type ${message.type}:`, error)
      this.notifyError(
        new Error(
          `Message routing failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        )
      )
    }
  }

  /**
   * Route incoming messages to appropriate handlers
   */
  private routeMessage(message: AllWebSocketMessages): void {
    const handlers = this.eventHandlers.get(message.type)
    if (handlers && handlers.size > 0) {
      handlers.forEach((handler) => {
        try {
          handler(message)
        } catch (error) {
          console.error(`Error in message handler for type ${message.type}:`, error)
          // Continue processing other handlers even if one fails
        }
      })
    }

    // Special handling for llm_event messages - also route to sub-event handlers
    if (message.type === 'llm_event' && 'data' in message && message.data?.event_type) {
      const subEventHandlers = this.eventHandlers.get(`llm_event:${message.data.event_type}`)
      if (subEventHandlers && subEventHandlers.size > 0) {
        subEventHandlers.forEach((handler) => {
          try {
            handler(message)
          } catch (error) {
            console.error(`Error in llm_event sub-handler for ${message.data.event_type}:`, error)
          }
        })
      }
    }

    // Also emit to generic message handlers
    const genericHandlers = this.eventHandlers.get('*')
    if (genericHandlers) {
      genericHandlers.forEach((handler) => {
        try {
          handler(message)
        } catch (error) {
          console.error('Error in generic message handler:', error)
        }
      })
    }
  }

  /**
   * Notify all connection status callbacks
   */
  private notifyConnectionStatus(status: ConnectionStatus): void {
    this.connectionStatusCallbacks.forEach((callback) => {
      try {
        callback(status)
      } catch (error) {
        console.error('Error in connection status callback:', error)
      }
    })
  }

  /**
   * Notify all error callbacks
   */
  private notifyError(error: Event | Error): void {
    this.errorCallbacks.forEach((callback) => {
      try {
        callback(error)
      } catch (callbackError) {
        console.error('Error in error callback:', callbackError)
      }
    })
  }

  /**
   * Setup connection timeout
   */
  private setupConnectionTimeout(): void {
    this.clearConnectionTimeout()

    this.connectionTimeoutId = window.setTimeout(() => {
      if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
        console.error('WebSocket connection timeout')
        this.ws.close()
        this.notifyError(new Error('Connection timeout'))
        this.notifyConnectionStatus('error')
        this.scheduleReconnect()
      }
    }, this.connectionTimeout)
  }

  /**
   * Clear connection timeout
   */
  private clearConnectionTimeout(): void {
    if (this.connectionTimeoutId) {
      clearTimeout(this.connectionTimeoutId)
      this.connectionTimeoutId = null
    }
  }

  /**
   * Schedule a reconnection attempt with exponential backoff and fallback URLs
   */
  private scheduleReconnect(): void {
    if (this.isManualClose) {
      return
    }

    // Try fallback URLs if available and primary connection failed
    if (
      this.reconnectAttempts >= this.maxReconnectAttempts &&
      this.enableResilience &&
      this.fallbackUrls.length > 0
    ) {
      this.tryFallbackUrl()
      return
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      const isDebugMode =
        import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
      if (isDebugMode) {
        console.log('Max reconnection attempts reached')
      }
      this.notifyConnectionStatus('error')

      // In resilience mode, keep trying with longer delays
      if (this.enableResilience) {
        setTimeout(() => {
          this.reconnectAttempts = 0 // Reset attempts
          this.scheduleReconnect()
        }, 60000) // Wait 1 minute before resetting
      }
      return
    }

    this.reconnectAttempts++
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.maxReconnectDelay
    )

    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 1000
    const finalDelay = delay + jitter

    // Only log every few attempts to reduce spam, or in debug mode
    const isDebugMode =
      import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
    if (isDebugMode || this.reconnectAttempts <= 3 || this.reconnectAttempts % 5 === 0) {
      console.log(
        `Scheduling reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(finalDelay)}ms`
      )
    }

    setTimeout(() => {
      if (!this.isManualClose) {
        this.connect()
      }
    }, finalDelay)
  }

  /**
   * Start sending periodic ping messages to keep connection alive
   */
  private startPingInterval(): void {
    this.clearPingInterval()

    // Reset last pong time on start
    this.lastPongTime = Date.now()

    this.pingInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        // Check if we've received a pong recently
        const timeSinceLastPong = Date.now() - this.lastPongTime
        if (timeSinceLastPong > this.pongTimeout) {
          console.error('WebSocket pong timeout - connection may be dead')
          this.ws.close()
          return
        }

        this.send(
          {
            type: 'ping',
            timestamp: new Date().toISOString(),
          },
          false
        ) // Don't queue ping messages
      } else {
        this.clearPingInterval()
      }
    }, this.pingIntervalMs)
  }

  /**
   * Clear the ping interval
   */
  private clearPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval)
      this.pingInterval = null
    }
  }

  /**
   * Get human-readable ready state text
   */
  private getReadyStateText(readyState: number): string {
    switch (readyState) {
      case WebSocket.CONNECTING:
        return 'CONNECTING'
      case WebSocket.OPEN:
        return 'OPEN'
      case WebSocket.CLOSING:
        return 'CLOSING'
      case WebSocket.CLOSED:
        return 'CLOSED'
      default:
        return 'UNKNOWN'
    }
  }

  /**
   * Clean up all resources
   */
  public destroy(): void {
    // Clear cleanup timer
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }

    // Disconnect WebSocket
    this.disconnect()

    // Clear all collections
    this.eventHandlers.clear()
    this.connectionStatusCallbacks.clear()
    this.errorCallbacks.clear()
    this.messageQueue = []
    this.messageThrottle.clear()
    this.retryQueue.clear()

    // Clear timeouts
    if (this.connectionTimeoutId) {
      clearTimeout(this.connectionTimeoutId)
      this.connectionTimeoutId = null
    }
  }

  /**
   * Helper method to validate message structure
   */
  public static isValidMessage(message: any): message is AllWebSocketMessages {
    return (
      message &&
      typeof message === 'object' &&
      typeof message.type === 'string' &&
      message.data !== undefined
    )
  }

  /**
   * Helper to parse LLM-specific messages
   */
  public static parseLLMMessage(
    message: any
  ): LLMRequestMessage | LLMResponseMessage | TradingDecisionMessage | LLMEventMessage | null {
    if (!this.isValidMessage(message)) return null

    switch (message.type) {
      case 'llm_request':
        return message
      case 'llm_response':
        return message
      case 'trading_decision':
        return message
      case 'llm_event':
        return message
      default:
        return null
    }
  }

  /**
   * Type guard for specific message types
   */
  public static isLLMRequest(message: any): message is LLMRequestMessage {
    return message?.type === 'llm_request'
  }

  public static isLLMResponse(message: any): message is LLMResponseMessage {
    return message?.type === 'llm_response'
  }

  public static isTradingDecision(message: any): message is TradingDecisionMessage {
    return message?.type === 'trading_decision'
  }

  public static isLLMEvent(message: any): message is LLMEventMessage {
    return message?.type === 'llm_event'
  }

  /**
   * Get current queue size
   */
  public getQueueSize(): number {
    return this.messageQueue.length
  }

  /**
   * Clear message queue
   */
  public clearQueue(): void {
    this.messageQueue = []
  }

  /**
   * Extract LLM event data from llm_event message
   */
  public static extractLLMEventData(
    message: LLMEventMessage
  ): LLMRequestMessage | LLMResponseMessage | TradingDecisionMessage | null {
    if (!message.data) return null

    const eventData = message.data
    const baseData = {
      timestamp: eventData.timestamp || message.timestamp,
    }

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
          },
        } as LLMRequestMessage

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
          },
        } as LLMResponseMessage

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
          },
        } as TradingDecisionMessage

      default:
        return null
    }
  }

  /**
   * Wait for connection to be established
   */
  public async waitForConnection(timeout: number = 5000): Promise<boolean> {
    const startTime = Date.now()

    while (Date.now() - startTime < timeout) {
      if (this.isConnected()) {
        return true
      }
      await new Promise((resolve) => setTimeout(resolve, 100))
    }

    return false
  }

  /**
   * Subscribe to a specific LLM event sub-type
   */
  public onLLMEvent(
    eventType:
      | 'llm_request'
      | 'llm_response'
      | 'trading_decision'
      | 'performance_metrics'
      | 'alert',
    handler: MessageHandler
  ): void {
    this.on(`llm_event:${eventType}`, handler)
  }

  /**
   * Unsubscribe from a specific LLM event sub-type
   */
  public offLLMEvent(
    eventType:
      | 'llm_request'
      | 'llm_response'
      | 'trading_decision'
      | 'performance_metrics'
      | 'alert',
    handler: MessageHandler
  ): void {
    this.off(`llm_event:${eventType}`, handler)
  }

  /**
   * Validate message structure for resilience
   */
  private validateMessageStructure(message: any): message is AllWebSocketMessages {
    if (!message || typeof message !== 'object') {
      return false
    }

    if (!message.type || typeof message.type !== 'string') {
      return false
    }

    // Allow undefined data for simple messages like ping/pong
    if (message.data !== undefined && typeof message.data !== 'object') {
      return false
    }

    return true
  }

  /**
   * Handle malformed messages gracefully
   */
  private handleMalformedMessage(data: string): void {
    console.warn('Attempting to recover from malformed message')

    // Try to extract a valid message structure
    try {
      // Common malformed patterns to recover from
      const patterns = [
        // Truncated JSON - try to find the last complete object
        { regex: /\{[^}]*\}(?=\s*$)/, name: 'truncated' },
        // Double-encoded JSON
        { regex: /"(\{.*\})"/, name: 'double-encoded' },
        // Concatenated messages
        { regex: /\}(\{)/g, name: 'concatenated' },
      ]

      for (const pattern of patterns) {
        if (pattern.name === 'concatenated') {
          // Split concatenated messages
          const messages = data.split(pattern.regex)
          messages.forEach((msgStr, index, array) => {
            if (index < array.length - 1) msgStr += '}'
            if (index > 0) msgStr = '{' + msgStr

            try {
              const msg = JSON.parse(msgStr)
              if (this.validateMessageStructure(msg)) {
                this.routeMessageSafely(msg)
              }
            } catch (e) {
              // Ignore individual parse failures
            }
          })
          return
        } else {
          const match = data.match(pattern.regex)
          if (match) {
            const recoveredData = pattern.name === 'double-encoded' ? match[1] : match[0]
            const message = JSON.parse(recoveredData)
            if (this.validateMessageStructure(message)) {
              console.log(`Recovered message using ${pattern.name} pattern`)
              this.routeMessageSafely(message)
              return
            }
          }
        }
      }
    } catch (error) {
      console.warn('Could not recover malformed message:', error)
    }

    // If recovery fails, log and continue
    console.error('Failed to recover malformed message, data length:', data.length)
  }

  /**
   * Handle invalid message structure
   */
  private handleInvalidMessage(message: any): void {
    console.warn('Handling invalid message structure:', message)

    // Try to create a valid message structure
    if (message && typeof message === 'object') {
      // Add missing type if data exists
      if (!message.type && message.data) {
        // Try to infer type from data structure
        const inferredType = this.inferMessageType(message.data)
        if (inferredType) {
          message.type = inferredType
          console.log(`Inferred message type: ${inferredType}`)
          this.routeMessageSafely(message)
          return
        }
      }

      // Add missing data as empty object
      if (message.type && !message.data) {
        message.data = {}
        this.routeMessageSafely(message)
        return
      }
    }

    console.warn('Could not repair invalid message structure')
  }

  /**
   * Infer message type from data structure
   */
  private inferMessageType(data: any): string | null {
    if (!data || typeof data !== 'object') return null

    // Check for common data patterns
    if (data.status && data.symbol) return 'bot_status'
    if (data.price && data.symbol) return 'market_data'
    if (data.action && data.confidence) return 'trade_action'
    if (data.cipher_a !== undefined || data.cipher_b !== undefined) return 'indicators'
    if (data.side && data.entry_price) return 'position'
    if (data.total_portfolio_value !== undefined) return 'risk_metrics'
    if (data.health !== undefined) return 'system_status'
    if (data.message && data.level) return 'error'

    return null
  }

  /**
   * Try fallback URLs when primary connection fails
   */
  private tryFallbackUrl(): void {
    if (this.fallbackUrls.length === 0) {
      console.warn('No fallback URLs available')
      this.notifyConnectionStatus('error')
      return
    }

    this.currentUrlIndex = (this.currentUrlIndex + 1) % this.fallbackUrls.length
    const fallbackUrl = this.fallbackUrls[this.currentUrlIndex]

    console.log(
      `Trying fallback URL ${this.currentUrlIndex + 1}/${this.fallbackUrls.length}: ${fallbackUrl}`
    )

    // Switch to fallback URL and reset attempts
    this.url = fallbackUrl
    this.reconnectAttempts = 0
    this.scheduleReconnect()
  }

  /**
   * Handle critical errors that might require connection reset
   */
  private handleCriticalError(error: any): void {
    console.error('Critical WebSocket error, considering connection reset:', error)

    // Close current connection
    if (this.ws) {
      this.ws.close(1006, 'Critical error occurred')
      this.ws = null
    }

    // Schedule reconnection
    this.notifyConnectionStatus('error')
    this.scheduleReconnect()
  }

  /**
   * Check if a message is critical and should be retried
   */
  private isCriticalMessage(message: any): boolean {
    if (!message?.type) return false

    const criticalTypes = ['ping', 'pong', 'subscription', 'authentication']
    return criticalTypes.includes(message.type)
  }

  /**
   * Queue message for retry
   */
  private queueForRetry(message: any): void {
    const messageId = this.generateMessageId(message)
    const existing = this.retryQueue.get(messageId)

    if (existing) {
      existing.attempts++
      if (existing.attempts >= existing.maxRetries) {
        console.warn(`Message retry limit reached for ${message.type}`)
        this.retryQueue.delete(messageId)
        return
      }
    } else {
      this.retryQueue.set(messageId, {
        message,
        attempts: 1,
        maxRetries: this.maxMessageRetries,
      })
    }

    // Schedule retry
    setTimeout(() => {
      this.retryQueuedMessage(messageId)
    }, this.messageRetryDelay)
  }

  /**
   * Retry a queued message
   */
  private retryQueuedMessage(messageId: string): void {
    const queued = this.retryQueue.get(messageId)
    if (!queued) return

    console.log(
      `Retrying message ${queued.message.type} (attempt ${queued.attempts}/${queued.maxRetries})`
    )

    if (this.send(queued.message, false)) {
      // Success - remove from retry queue
      this.retryQueue.delete(messageId)
    } else {
      // Failed - will be handled by queueForRetry if called again
      console.warn(`Retry failed for message ${queued.message.type}`)
    }
  }

  /**
   * Generate unique ID for message retry tracking
   */
  private generateMessageId(message: any): string {
    const type = message.type || 'unknown'
    const data = JSON.stringify(message.data || {})
    const timestamp = message.timestamp || Date.now()
    return `${type}-${timestamp}-${data.slice(0, 20)}`
  }

  /**
   * Get connection health status
   */
  public getConnectionHealth(): {
    isHealthy: boolean
    lastPong: number
    queueSize: number
    retryQueueSize: number
    reconnectAttempts: number
    currentUrl: string
    issues: string[]
  } {
    const now = Date.now()
    const timeSinceLastPong = now - this.lastPongTime
    const issues: string[] = []

    if (timeSinceLastPong > this.pongTimeout) {
      issues.push('Ping timeout')
    }

    if (this.messageQueue.length > this.maxQueueSize * 0.8) {
      issues.push('Message queue almost full')
    }

    if (this.retryQueue.size > 5) {
      issues.push('High retry queue size')
    }

    if (this.reconnectAttempts > 2) {
      issues.push('Multiple reconnection attempts')
    }

    return {
      isHealthy: issues.length === 0 && this.isConnected(),
      lastPong: this.lastPongTime,
      queueSize: this.messageQueue.length,
      retryQueueSize: this.retryQueue.size,
      reconnectAttempts: this.reconnectAttempts,
      currentUrl: this.url,
      issues,
    }
  }
}

// Singleton instance with default configuration
export const webSocketClient = new DashboardWebSocket()

// Convenience function to create configured client
export function createWebSocketClient(config: WebSocketConfig = {}): DashboardWebSocket {
  return new DashboardWebSocket(config.url, config)
}

// Production-ready client factory with optimized settings
export function createProductionWebSocketClient(): DashboardWebSocket {
  return new DashboardWebSocket(undefined, {
    maxReconnectAttempts: 5,
    initialReconnectDelay: 1000,
    maxReconnectDelay: 30000,
    pingInterval: 30000,
    connectionTimeout: 10000,
    enableResilience: true,
    maxMessageRetries: 3,
    messageRetryDelay: 1000,
  })
}

// Resilient client factory with fallback URLs
export function createResilientWebSocketClient(fallbackUrls: string[] = []): DashboardWebSocket {
  return new DashboardWebSocket(undefined, {
    maxReconnectAttempts: 10,
    initialReconnectDelay: 500,
    maxReconnectDelay: 60000,
    pingInterval: 15000,
    connectionTimeout: 15000,
    enableResilience: true,
    fallbackUrls,
    maxMessageRetries: 5,
    messageRetryDelay: 2000,
  })
}

// Helper to validate and parse WebSocket messages
export function parseWebSocketMessage(data: string): AllWebSocketMessages | null {
  try {
    const message = JSON.parse(data)
    if (DashboardWebSocket.isValidMessage(message)) {
      return message
    }
    return null
  } catch (error) {
    console.error('Failed to parse WebSocket message:', error)
    return null
  }
}
