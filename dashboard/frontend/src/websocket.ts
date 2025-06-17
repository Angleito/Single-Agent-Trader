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
  clientTime?: number
}

export interface PongMessage {
  type: 'pong'
  timestamp: string
  clientTime?: number
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
  // Enhanced configuration
  enableJitter?: boolean
  jitterMaxMs?: number
  enableCircuitBreaker?: boolean
  circuitBreakerThreshold?: number
  circuitBreakerResetTimeMs?: number
  enableOfflineMode?: boolean
  connectionHealthCheckInterval?: number
  maxConsecutiveFailures?: number
  backoffMultiplier?: number
  enableNetworkStatusDetection?: boolean
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

  // Enhanced error handling and connection resilience
  private enableJitter: boolean
  private jitterMaxMs: number
  private enableCircuitBreaker: boolean
  private circuitBreakerThreshold: number
  private circuitBreakerResetTimeMs: number
  private enableOfflineMode: boolean
  private connectionHealthCheckInterval: number
  private maxConsecutiveFailures: number
  private backoffMultiplier: number
  private enableNetworkStatusDetection: boolean

  // Connection health monitoring
  private consecutiveFailures = 0
  private lastSuccessfulConnection: number = Date.now()
  private connectionQuality: 'excellent' | 'good' | 'poor' | 'critical' = 'excellent'
  private averageLatency = 0
  private latencyHistory: number[] = []
  private circuitBreakerState: 'closed' | 'open' | 'half-open' = 'closed'
  private circuitBreakerOpenTime: number | null = null
  private healthCheckTimer: number | null = null
  private networkStatusListener: (() => void) | null = null

  // Error categorization
  private errorHistory: Array<{ type: string; timestamp: number; recoverable: boolean }> = []
  private readonly ERROR_HISTORY_SIZE = 20

  // Offline mode support
  private isOffline = false
  private offlineModeCallbacks = new Set<(isOffline: boolean) => void>()
  private offlineMessageQueue: any[] = []
  private readonly MAX_OFFLINE_QUEUE_SIZE = 100

  constructor(url?: string, config: WebSocketConfig = {}) {
    // Check for runtime configuration first (multiple patterns for compatibility)
    const runtimeWsUrl =
      (window as any).__WS_URL__ ||
      (window as any).__VITE_WS_URL__ ||
      (window as any).__RUNTIME_CONFIG__?.WS_URL

    // Use dynamic URL detection if no URL provided
    if (!url && !config.url && !runtimeWsUrl) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const port = window.location.port
      const isNginxPort = port === '8080'
      
      // Use correct path based on environment
      const wsPath = isNginxPort ? '/api/ws' : '/ws'
      url = `${protocol}//${window.location.host}${wsPath}`
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

    // Enhanced configuration
    this.enableJitter = config.enableJitter !== false
    this.jitterMaxMs = config.jitterMaxMs ?? 1000
    this.enableCircuitBreaker = config.enableCircuitBreaker !== false
    this.circuitBreakerThreshold = config.circuitBreakerThreshold ?? 5
    this.circuitBreakerResetTimeMs = config.circuitBreakerResetTimeMs ?? 60000
    this.enableOfflineMode = config.enableOfflineMode !== false
    this.connectionHealthCheckInterval = config.connectionHealthCheckInterval ?? 30000
    this.maxConsecutiveFailures = config.maxConsecutiveFailures ?? 3
    this.backoffMultiplier = config.backoffMultiplier ?? 2
    this.enableNetworkStatusDetection = config.enableNetworkStatusDetection !== false

    // Start memory cleanup
    this.startMemoryCleanup()

    // Setup network status detection
    if (this.enableNetworkStatusDetection) {
      this.setupNetworkStatusDetection()
    }

    // Setup connection health monitoring
    this.startHealthCheck()
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
    const currentTime = Date.now()
    const throttleEntries = Array.from(this.messageThrottle.entries())
    for (const [key, timestamp] of throttleEntries) {
      if (currentTime - timestamp > 10000) {
        // Remove entries older than 10 seconds
        this.messageThrottle.delete(key)
      }
    }

    // Clean up old retry queue entries
    const retryEntries = Array.from(this.retryQueue.entries())
    for (const [messageId, queued] of retryEntries) {
      if (queued.attempts >= queued.maxRetries) {
        this.retryQueue.delete(messageId)
      }
    }

    // Clean up old error history
    this.errorHistory = this.errorHistory.filter(e => currentTime - e.timestamp < 86400000) // Keep 24 hours

    // Clean up latency history (keep reasonable size)
    if (this.latencyHistory.length > 50) {
      this.latencyHistory = this.latencyHistory.slice(-20)
    }

    // Clean up offline message queue if too large
    if (this.offlineMessageQueue.length > this.MAX_OFFLINE_QUEUE_SIZE) {
      this.offlineMessageQueue = this.offlineMessageQueue.slice(-this.MAX_OFFLINE_QUEUE_SIZE)
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
   * Connect to the WebSocket server with enhanced error handling
   */
  public connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return
    }

    // Check circuit breaker state
    if (this.enableCircuitBreaker && this.circuitBreakerState === 'open') {
      const timeSinceOpen = Date.now() - (this.circuitBreakerOpenTime || 0)
      if (timeSinceOpen < this.circuitBreakerResetTimeMs) {
        console.warn('Circuit breaker is open, connection attempt blocked')
        return
      } else {
        // Try to move to half-open state
        this.circuitBreakerState = 'half-open'
        console.log('Circuit breaker moving to half-open state')
      }
    }

    this.isManualClose = false
    this.notifyConnectionStatus('connecting')

    // Debug logging (reduced verbosity)
    const isDebugMode =
      import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'

    try {
      // Check if WebSocket is available
      if (typeof WebSocket === 'undefined') {
        throw new Error('WebSocket not available in this environment')
      }

      // Additional checks for WebSocket availability
      if (!window.WebSocket && !(window as any).MozWebSocket) {
        throw new Error('WebSocket not supported by this browser')
      }

      // Check network connectivity
      if (this.enableOfflineMode && !navigator.onLine) {
        throw new Error('No network connection available')
      }

      // Check if we're in a secure context if needed
      const isSecure = this.url.startsWith('wss://')
      if (isSecure && !window.isSecureContext) {
        console.warn('Secure WebSocket requested but not in secure context')
      }

      this.ws = new WebSocket(this.url)
      if (isDebugMode) {
        console.debug('WebSocket connection attempt to:', this.url)
      }

      this.setupEventListeners()
      this.setupConnectionTimeout()
    } catch (error) {
      // Enhanced error categorization
      const errorType = this.categorizeError(error)
      this.recordError(errorType, error instanceof Error)

      // More concise error logging
      const errorMessage = error instanceof Error ? error.message : 'Unknown WebSocket error'
      console.error('WebSocket connection failed:', errorMessage)

      // Provide user-friendly error messages
      const userErrorMessage = this.getUserFriendlyErrorMessage(error)

      this.notifyError(new Error(userErrorMessage))
      this.notifyConnectionStatus('error')
      this.handleConnectionFailure()
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
   * Send a message to the server with enhanced retry and offline support
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

    // Handle offline mode
    if (this.enableOfflineMode && this.isOffline && queueIfDisconnected) {
      if (this.offlineMessageQueue.length < this.MAX_OFFLINE_QUEUE_SIZE) {
        this.offlineMessageQueue.push(message)
        const isDebugMode =
          import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
        if (isDebugMode) {
          console.log('Message queued for offline mode:', message.type)
        }
        return true
      } else {
        console.warn('Offline message queue is full, dropping message')
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
        console.debug('WebSocket connection opened successfully')
      }

      this.clearConnectionTimeout()

      // Reset connection tracking
      this.reconnectAttempts = 0
      this.reconnectDelay = 1000
      this.consecutiveFailures = 0
      this.lastSuccessfulConnection = Date.now()

      // Update circuit breaker state
      if (this.enableCircuitBreaker) {
        this.closeCircuitBreaker()
      }

      // Update connection quality
      this.updateConnectionQuality('excellent')

      // Exit offline mode if enabled
      if (this.isOffline) {
        this.exitOfflineMode()
      }

      this.notifyConnectionStatus('connected')
      this.startPingInterval()

      // Process queued messages
      if (this.messageQueue.length > 0) {
        const queuedCount = this.messageQueue.length
        if (isDebugMode) {
          console.debug(`Processing ${queuedCount} queued messages`)
        }

        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift()
          this.send(message, false) // Don't re-queue if send fails
        }
      }

      // Process offline message queue if available
      if (this.enableOfflineMode && this.offlineMessageQueue.length > 0) {
        this.flushOfflineQueue()
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

        // Handle pong messages internally with latency tracking
        if (message.type === 'pong') {
          this.lastPongTime = Date.now()

          // Calculate latency if clientTime is available
          if ('clientTime' in message && typeof message.clientTime === 'number') {
            const latency = Date.now() - message.clientTime
            this.updateLatencyHistory(latency)
            this.updateConnectionQualityFromLatency(latency)
          }

          // Record successful ping-pong for circuit breaker
          if (this.enableCircuitBreaker && this.circuitBreakerState === 'half-open') {
            this.closeCircuitBreaker()
          }

          return
        }

        // Handle ping messages (server-initiated ping)
        if (message.type === 'ping') {
          const pongMessage: any = {
            type: 'pong',
            timestamp: new Date().toISOString()
          }

          // Echo back clientTime if present for latency calculation
          if ('clientTime' in message) {
            pongMessage.clientTime = message.clientTime
          }

          this.send(pongMessage, false)
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

      // Categorize the close event
      const closeReason = this.categorizeCloseEvent(event)

      // Only log close details if it's an unexpected close or in debug mode
      if (event.code !== 1000 || isDebugMode) {
        console.log(`WebSocket closed: ${closeReason} (code: ${event.code})`)
      }

      // Record the error if it wasn't a clean close
      if (event.code !== 1000) {
        this.recordError('connection_closed', event.code < 1003)
      }

      this.clearPingInterval()
      this.clearConnectionTimeout()
      this.ws = null

      if (!this.isManualClose) {
        this.notifyConnectionStatus('disconnected')
        this.handleConnectionFailure()
      }
    }

    this.ws.onerror = (error) => {
      // Enhanced error handling with categorization
      const errorType = this.categorizeError(error)
      this.recordError(errorType, false)

      console.error(`WebSocket error occurred: ${errorType}`)

      const isDebugMode =
        import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
      if (isDebugMode) {
        console.error('WebSocket error details:', {
          readyState: this.ws?.readyState,
          url: this.url,
          errorType,
          error: error,
        })
      }

      this.clearConnectionTimeout()

      // Update connection quality
      this.updateConnectionQuality('critical')

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
   * Schedule a reconnection attempt with enhanced exponential backoff and circuit breaker
   */
  private scheduleReconnect(): void {
    if (this.isManualClose) {
      return
    }

    // Check if we should enter offline mode
    if (this.enableOfflineMode && !navigator.onLine) {
      this.enterOfflineMode()
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

      // Update circuit breaker state
      if (this.enableCircuitBreaker) {
        this.openCircuitBreaker()
      }

      this.notifyConnectionStatus('error')

      // In resilience mode, keep trying with longer delays
      if (this.enableResilience) {
        const resetDelay = this.enableCircuitBreaker
          ? this.circuitBreakerResetTimeMs
          : 60000

        setTimeout(() => {
          this.reconnectAttempts = 0 // Reset attempts
          this.consecutiveFailures = 0
          this.scheduleReconnect()
        }, resetDelay)
      }
      return
    }

    this.reconnectAttempts++

    // Enhanced exponential backoff with configurable multiplier
    const baseDelay = this.reconnectDelay * Math.pow(this.backoffMultiplier, this.reconnectAttempts - 1)
    const delay = Math.min(baseDelay, this.maxReconnectDelay)

    // Add jitter to prevent thundering herd (configurable)
    const jitter = this.enableJitter ? Math.random() * this.jitterMaxMs : 0
    const finalDelay = delay + jitter

    // Adaptive delay based on connection history
    const adaptiveMultiplier = this.getAdaptiveDelayMultiplier()
    const adaptedDelay = finalDelay * adaptiveMultiplier

    // Only log every few attempts to reduce spam, or in debug mode
    const isDebugMode =
      import.meta.env.VITE_DEBUG === 'true' || import.meta.env.VITE_LOG_LEVEL === 'debug'
    if (isDebugMode || this.reconnectAttempts <= 3 || this.reconnectAttempts % 5 === 0) {
      console.log(
        `Scheduling reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(adaptedDelay)}ms (quality: ${this.connectionQuality})`
      )
    }

    setTimeout(() => {
      if (!this.isManualClose) {
        this.connect()
      }
    }, adaptedDelay)
  }

  /**
   * Start sending periodic ping messages with enhanced health monitoring
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
          this.recordError('ping_timeout', true)
          this.updateConnectionQuality('critical')
          this.ws.close()
          return
        }

        // Update connection quality based on ping responsiveness
        this.updateConnectionQualityFromPing(timeSinceLastPong)

        // Send ping with timestamp for latency calculation
        const pingTime = Date.now()
        this.send(
          {
            type: 'ping',
            timestamp: new Date().toISOString(),
            clientTime: pingTime,
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
   * Clean up all resources including enhanced features
   */
  public destroy(): void {
    // Clear cleanup timer
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }

    // Clear health check timer
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer)
      this.healthCheckTimer = null
    }

    // Remove network status listener
    if (this.networkStatusListener) {
      window.removeEventListener('online', this.networkStatusListener)
      window.removeEventListener('offline', this.networkStatusListener)
      this.networkStatusListener = null
    }

    // Disconnect WebSocket
    this.disconnect()

    // Clear all collections
    this.eventHandlers.clear()
    this.connectionStatusCallbacks.clear()
    this.errorCallbacks.clear()
    this.offlineModeCallbacks.clear()
    this.messageQueue = []
    this.offlineMessageQueue = []
    this.messageThrottle.clear()
    this.retryQueue.clear()
    this.errorHistory = []
    this.latencyHistory = []

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
   * Get enhanced connection health status
   */
  public getConnectionHealth(): {
    isHealthy: boolean
    lastPong: number
    queueSize: number
    retryQueueSize: number
    reconnectAttempts: number
    currentUrl: string
    issues: string[]
    connectionQuality: 'excellent' | 'good' | 'poor' | 'critical'
    averageLatency: number
    consecutiveFailures: number
    circuitBreakerState: 'closed' | 'open' | 'half-open'
    isOffline: boolean
    timeSinceLastSuccess: number
  } {
    const now = Date.now()
    const timeSinceLastPong = now - this.lastPongTime
    const timeSinceLastSuccess = now - this.lastSuccessfulConnection
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

    if (this.consecutiveFailures > this.maxConsecutiveFailures) {
      issues.push('Multiple consecutive failures')
    }

    if (this.circuitBreakerState === 'open') {
      issues.push('Circuit breaker is open')
    }

    if (this.isOffline) {
      issues.push('Offline mode active')
    }

    if (this.averageLatency > 1000) {
      issues.push('High latency')
    }

    return {
      isHealthy: issues.length === 0 && this.isConnected(),
      lastPong: this.lastPongTime,
      queueSize: this.messageQueue.length,
      retryQueueSize: this.retryQueue.size,
      reconnectAttempts: this.reconnectAttempts,
      currentUrl: this.url,
      issues,
      connectionQuality: this.connectionQuality,
      averageLatency: this.averageLatency,
      consecutiveFailures: this.consecutiveFailures,
      circuitBreakerState: this.circuitBreakerState,
      isOffline: this.isOffline,
      timeSinceLastSuccess,
    }
  }

  /**
   * Setup network status detection
   */
  private setupNetworkStatusDetection(): void {
    if (typeof window === 'undefined' || !window.navigator) {
      return
    }

    this.networkStatusListener = () => {
      const wasOffline = this.isOffline
      const isNowOnline = navigator.onLine

      if (wasOffline && isNowOnline) {
        console.log('Network connection restored, attempting to reconnect')
        this.exitOfflineMode()
        if (!this.isConnected()) {
          this.connect()
        }
      } else if (!wasOffline && !isNowOnline) {
        console.log('Network connection lost, entering offline mode')
        this.enterOfflineMode()
      }
    }

    window.addEventListener('online', this.networkStatusListener)
    window.addEventListener('offline', this.networkStatusListener)
  }

  /**
   * Start connection health monitoring
   */
  private startHealthCheck(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer)
    }

    this.healthCheckTimer = window.setInterval(() => {
      this.performHealthCheck()
    }, this.connectionHealthCheckInterval)
  }

  /**
   * Perform periodic health check
   */
  private performHealthCheck(): void {
    const health = this.getConnectionHealth()

    // Log health issues if they exist
    if (health.issues.length > 0 && import.meta.env.VITE_DEBUG === 'true') {
      console.debug('Connection health issues:', health.issues)
    }

    // Update connection quality based on overall health
    if (health.issues.length >= 3) {
      this.updateConnectionQuality('critical')
    } else if (health.issues.length >= 2) {
      this.updateConnectionQuality('poor')
    } else if (health.issues.length >= 1) {
      this.updateConnectionQuality('good')
    }
  }

  /**
   * Categorize different types of errors
   */
  private categorizeError(error: any): string {
    if (!error) return 'unknown'

    const errorMessage = error.message || error.toString().toLowerCase()

    if (errorMessage.includes('timeout')) return 'timeout'
    if (errorMessage.includes('network')) return 'network'
    if (errorMessage.includes('refused') || errorMessage.includes('unreachable')) return 'connection_refused'
    if (errorMessage.includes('not available') || errorMessage.includes('not supported')) return 'browser_support'
    if (errorMessage.includes('security') || errorMessage.includes('origin')) return 'security'
    if (errorMessage.includes('rate limit') || errorMessage.includes('too many')) return 'rate_limit'

    return 'unknown'
  }

  /**
   * Categorize WebSocket close events
   */
  private categorizeCloseEvent(event: CloseEvent): string {
    switch (event.code) {
      case 1000: return 'Normal closure'
      case 1001: return 'Going away'
      case 1002: return 'Protocol error'
      case 1003: return 'Unsupported data'
      case 1005: return 'No status received'
      case 1006: return 'Abnormal closure'
      case 1007: return 'Invalid frame payload data'
      case 1008: return 'Policy violation'
      case 1009: return 'Message too large'
      case 1010: return 'Missing extension'
      case 1011: return 'Internal error'
      case 1012: return 'Service restart'
      case 1013: return 'Try again later'
      case 1014: return 'Bad gateway'
      case 1015: return 'TLS handshake'
      default: return `Unknown (${event.code})`
    }
  }

  /**
   * Record error in history for analysis
   */
  private recordError(errorType: string, recoverable: boolean): void {
    this.errorHistory.push({
      type: errorType,
      timestamp: Date.now(),
      recoverable
    })

    // Keep history size manageable
    if (this.errorHistory.length > this.ERROR_HISTORY_SIZE) {
      this.errorHistory = this.errorHistory.slice(-this.ERROR_HISTORY_SIZE)
    }

    // Update consecutive failures counter
    if (!recoverable) {
      this.consecutiveFailures++
    }
  }

  /**
   * Get user-friendly error message
   */
  private getUserFriendlyErrorMessage(error: any): string {
    if (!error) return 'Unknown connection error'

    if (error instanceof Error) {
      const errorType = this.categorizeError(error)

      switch (errorType) {
        case 'network':
          return 'Network connection error - please check your internet connection'
        case 'connection_refused':
          return 'Unable to connect to server - server may be unavailable'
        case 'timeout':
          return 'Connection timeout - server is not responding'
        case 'browser_support':
          return 'WebSocket not supported by this browser'
        case 'security':
          return 'Security error - check HTTPS/WSS configuration'
        case 'rate_limit':
          return 'Too many connection attempts - please wait before retrying'
        default:
          return error.message || 'Connection failed'
      }
    }

    return 'Unknown connection error'
  }

  /**
   * Handle connection failure with enhanced logic
   */
  private handleConnectionFailure(): void {
    this.consecutiveFailures++

    // Check if we should open circuit breaker
    if (this.enableCircuitBreaker &&
        this.consecutiveFailures >= this.circuitBreakerThreshold) {
      this.openCircuitBreaker()
    }

    // Update connection quality
    this.updateConnectionQuality('critical')

    // Schedule reconnection or enter offline mode
    if (this.enableOfflineMode && !navigator.onLine) {
      this.enterOfflineMode()
    } else {
      this.scheduleReconnect()
    }
  }

  /**
   * Update connection quality based on various factors
   */
  private updateConnectionQuality(quality: 'excellent' | 'good' | 'poor' | 'critical'): void {
    if (this.connectionQuality !== quality) {
      this.connectionQuality = quality

      if (import.meta.env.VITE_DEBUG === 'true') {
        console.debug(`Connection quality updated to: ${quality}`)
      }
    }
  }

  /**
   * Update connection quality based on ping responsiveness
   */
  private updateConnectionQualityFromPing(timeSinceLastPong: number): void {
    if (timeSinceLastPong < this.pingIntervalMs * 1.5) {
      this.updateConnectionQuality('excellent')
    } else if (timeSinceLastPong < this.pingIntervalMs * 2) {
      this.updateConnectionQuality('good')
    } else if (timeSinceLastPong < this.pongTimeout * 0.8) {
      this.updateConnectionQuality('poor')
    } else {
      this.updateConnectionQuality('critical')
    }
  }

  /**
   * Update latency history and calculate average
   */
  private updateLatencyHistory(latency: number): void {
    this.latencyHistory.push(latency)

    // Keep history size manageable
    if (this.latencyHistory.length > 20) {
      this.latencyHistory = this.latencyHistory.slice(-20)
    }

    // Calculate running average
    this.averageLatency = this.latencyHistory.reduce((sum, lat) => sum + lat, 0) / this.latencyHistory.length
  }

  /**
   * Update connection quality based on latency
   */
  private updateConnectionQualityFromLatency(latency: number): void {
    if (latency < 100) {
      this.updateConnectionQuality('excellent')
    } else if (latency < 300) {
      this.updateConnectionQuality('good')
    } else if (latency < 1000) {
      this.updateConnectionQuality('poor')
    } else {
      this.updateConnectionQuality('critical')
    }
  }

  /**
   * Get adaptive delay multiplier based on connection history
   */
  private getAdaptiveDelayMultiplier(): number {
    if (this.consecutiveFailures === 0) return 1
    if (this.consecutiveFailures < 3) return 1.5
    if (this.consecutiveFailures < 5) return 2
    return 3
  }

  /**
   * Circuit breaker methods
   */
  private openCircuitBreaker(): void {
    this.circuitBreakerState = 'open'
    this.circuitBreakerOpenTime = Date.now()
    console.warn('Circuit breaker opened due to consecutive failures')
  }

  private closeCircuitBreaker(): void {
    if (this.circuitBreakerState !== 'closed') {
      this.circuitBreakerState = 'closed'
      this.circuitBreakerOpenTime = null
      this.consecutiveFailures = 0
      console.log('Circuit breaker closed - connection restored')
    }
  }

  /**
   * Offline mode methods
   */
  private enterOfflineMode(): void {
    if (!this.isOffline) {
      this.isOffline = true
      console.log('Entering offline mode')
      this.notifyOfflineMode(true)
    }
  }

  private exitOfflineMode(): void {
    if (this.isOffline) {
      this.isOffline = false
      console.log('Exiting offline mode')
      this.notifyOfflineMode(false)
    }
  }

  private notifyOfflineMode(isOffline: boolean): void {
    this.offlineModeCallbacks.forEach((callback) => {
      try {
        callback(isOffline)
      } catch (error) {
        console.error('Error in offline mode callback:', error)
      }
    })
  }

  private flushOfflineQueue(): void {
    if (this.offlineMessageQueue.length === 0) return

    console.log(`Flushing ${this.offlineMessageQueue.length} offline messages`)

    while (this.offlineMessageQueue.length > 0) {
      const message = this.offlineMessageQueue.shift()
      this.send(message, false)
    }
  }

  /**
   * Public methods for offline mode
   */
  public onOfflineModeChange(callback: (isOffline: boolean) => void): void {
    this.offlineModeCallbacks.add(callback)
  }

  public offOfflineModeChange(callback: (isOffline: boolean) => void): void {
    this.offlineModeCallbacks.delete(callback)
  }

  public isInOfflineMode(): boolean {
    return this.isOffline
  }

  /**
   * Get current connection quality
   */
  public getConnectionQuality(): 'excellent' | 'good' | 'poor' | 'critical' {
    return this.connectionQuality
  }

  /**
   * Get average latency
   */
  public getAverageLatency(): number {
    return this.averageLatency
  }

  /**
   * Get circuit breaker state
   */
  public getCircuitBreakerState(): 'closed' | 'open' | 'half-open' {
    return this.circuitBreakerState
  }

  /**
   * Get consecutive failures count
   */
  public getConsecutiveFailures(): number {
    return this.consecutiveFailures
  }

  /**
   * Force circuit breaker reset (for admin/debugging purposes)
   */
  public resetCircuitBreaker(): void {
    this.closeCircuitBreaker()
    console.log('Circuit breaker manually reset')
  }

  /**
   * Get error history for debugging
   */
  public getErrorHistory(): Array<{ type: string; timestamp: number; recoverable: boolean }> {
    return [...this.errorHistory]
  }

  /**
   * Clear error history
   */
  public clearErrorHistory(): void {
    this.errorHistory = []
    this.consecutiveFailures = 0
    console.log('Error history cleared')
  }

  /**
   * Get latency history
   */
  public getLatencyHistory(): number[] {
    return [...this.latencyHistory]
  }

  /**
   * Force connection quality update (for testing)
   */
  public setConnectionQuality(quality: 'excellent' | 'good' | 'poor' | 'critical'): void {
    this.updateConnectionQuality(quality)
  }

  /**
   * Get comprehensive connection statistics
   */
  public getConnectionStats(): {
    uptime: number
    totalReconnects: number
    averageLatency: number
    connectionQuality: string
    messagesQueued: number
    offlineMessagesQueued: number
    errorRate: number
    lastSuccessfulConnection: number
    circuitBreakerTrips: number
  } {
    const now = Date.now()
    const recentErrors = this.errorHistory.filter(e => now - e.timestamp < 3600000) // Last hour
    const errorRate = recentErrors.length / 60 // Errors per minute

    // Count circuit breaker trips
    const circuitBreakerTrips = this.errorHistory.filter(e =>
      e.type === 'circuit_breaker_open'
    ).length

    return {
      uptime: now - this.lastSuccessfulConnection,
      totalReconnects: this.reconnectAttempts,
      averageLatency: this.averageLatency,
      connectionQuality: this.connectionQuality,
      messagesQueued: this.messageQueue.length,
      offlineMessagesQueued: this.offlineMessageQueue.length,
      errorRate,
      lastSuccessfulConnection: this.lastSuccessfulConnection,
      circuitBreakerTrips,
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
    enableJitter: true,
    jitterMaxMs: 1000,
    enableCircuitBreaker: true,
    circuitBreakerThreshold: 5,
    circuitBreakerResetTimeMs: 60000,
    enableOfflineMode: true,
    connectionHealthCheckInterval: 30000,
    maxConsecutiveFailures: 3,
    backoffMultiplier: 2,
    enableNetworkStatusDetection: true,
  })
}

// Resilient client factory with enhanced fallback and error handling
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
    enableJitter: true,
    jitterMaxMs: 2000,
    enableCircuitBreaker: true,
    circuitBreakerThreshold: 3,
    circuitBreakerResetTimeMs: 30000,
    enableOfflineMode: true,
    connectionHealthCheckInterval: 15000,
    maxConsecutiveFailures: 2,
    backoffMultiplier: 2.5,
    enableNetworkStatusDetection: true,
  })
}

// High-performance client factory for low-latency requirements
export function createHighPerformanceWebSocketClient(): DashboardWebSocket {
  return new DashboardWebSocket(undefined, {
    maxReconnectAttempts: 3,
    initialReconnectDelay: 250,
    maxReconnectDelay: 5000,
    pingInterval: 10000,
    connectionTimeout: 5000,
    enableResilience: true,
    maxMessageRetries: 2,
    messageRetryDelay: 500,
    enableJitter: false, // Disable jitter for predictable timing
    enableCircuitBreaker: false, // Disable for immediate reconnection
    enableOfflineMode: false, // Disable for performance
    connectionHealthCheckInterval: 10000,
    maxConsecutiveFailures: 5,
    backoffMultiplier: 1.5,
    enableNetworkStatusDetection: false,
  })
}

// Development client factory with extensive logging and debugging
export function createDevelopmentWebSocketClient(): DashboardWebSocket {
  return new DashboardWebSocket(undefined, {
    maxReconnectAttempts: 20,
    initialReconnectDelay: 1000,
    maxReconnectDelay: 10000,
    pingInterval: 5000,
    connectionTimeout: 30000,
    enableResilience: true,
    maxMessageRetries: 10,
    messageRetryDelay: 1000,
    enableJitter: true,
    jitterMaxMs: 500,
    enableCircuitBreaker: true,
    circuitBreakerThreshold: 10,
    circuitBreakerResetTimeMs: 30000,
    enableOfflineMode: true,
    connectionHealthCheckInterval: 5000,
    maxConsecutiveFailures: 10,
    backoffMultiplier: 1.5,
    enableNetworkStatusDetection: true,
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
