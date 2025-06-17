/**
 * Enterprise-Grade WebSocket Manager
 *
 * Provides robust real-time communication with:
 * - Automatic reconnection with exponential backoff
 * - Message queuing and replay capabilities
 * - Multiple subscription management
 * - Health monitoring and diagnostics
 * - Comprehensive error handling and recovery
 * - Connection state management
 * - Message compression and batching
 * - Performance monitoring and metrics
 */

export interface WebSocketConfig {
  url: string
  protocols?: string[]
  reconnect: boolean
  maxReconnectAttempts: number
  reconnectInterval: number
  maxReconnectInterval: number
  reconnectBackoffMultiplier: number
  heartbeatInterval: number
  messageQueueSize: number
  enableCompression: boolean
  enableBatching: boolean
  batchInterval: number
  timeout: number
  debug: boolean
}

export interface WebSocketMessage {
  id: string
  type: string
  channel?: string
  data: any
  timestamp: number
  priority: 'low' | 'normal' | 'high' | 'critical'
  retry?: boolean
  maxRetries?: number
  retryCount?: number
}

export interface WebSocketSubscription {
  id: string
  channel: string
  callback: (data: any) => void
  errorCallback?: (error: Error) => void
  filter?: (data: any) => boolean
  transform?: (data: any) => any
  active: boolean
  lastMessageTime?: number
  messageCount: number
}

export interface ConnectionMetrics {
  connectTime: number
  lastMessageTime: number
  messagesSent: number
  messagesReceived: number
  reconnectCount: number
  errorCount: number
  averageLatency: number
  connectionDuration: number
  bytesTransferred: number
  compressionRatio: number
}

export interface HealthStatus {
  connected: boolean
  connecting: boolean
  lastPing: number
  latency: number
  quality: 'excellent' | 'good' | 'fair' | 'poor' | 'critical'
  errors: string[]
  warnings: string[]
}

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error'
  | 'closed'

export class WebSocketManager {
  private config: WebSocketConfig
  private socket: WebSocket | null = null
  private state: ConnectionState = 'disconnected'
  private subscriptions = new Map<string, WebSocketSubscription>()
  private messageQueue: WebSocketMessage[] = []
  private pendingMessages = new Map<string, WebSocketMessage>()
  private reconnectAttempts = 0
  private reconnectTimer: number | null = null
  private heartbeatTimer: number | null = null
  private batchTimer: number | null = null
  private batchQueue: WebSocketMessage[] = []
  private metrics: ConnectionMetrics
  private healthStatus: HealthStatus
  private listeners = new Map<string, Set<Function>>()
  private messageId = 0
  private compressionEnabled = false

  // Performance monitoring
  private latencyMeasurements: number[] = []
  private maxLatencyMeasurements = 100
  private performanceTimer: number | null = null

  // Error tracking
  private errorHistory: Array<{ timestamp: number; error: string; context: string }> = []
  private maxErrorHistory = 50

  constructor(config: Partial<WebSocketConfig>) {
    this.config = {
      url: '',
      protocols: [],
      reconnect: true,
      maxReconnectAttempts: 10,
      reconnectInterval: 1000,
      maxReconnectInterval: 30000,
      reconnectBackoffMultiplier: 1.5,
      heartbeatInterval: 30000,
      messageQueueSize: 1000,
      enableCompression: true,
      enableBatching: false,
      batchInterval: 100,
      timeout: 30000,
      debug: false,
      ...config,
    }

    this.metrics = {
      connectTime: 0,
      lastMessageTime: 0,
      messagesSent: 0,
      messagesReceived: 0,
      reconnectCount: 0,
      errorCount: 0,
      averageLatency: 0,
      connectionDuration: 0,
      bytesTransferred: 0,
      compressionRatio: 1.0,
    }

    this.healthStatus = {
      connected: false,
      connecting: false,
      lastPing: 0,
      latency: 0,
      quality: 'critical',
      errors: [],
      warnings: [],
    }

    this.startPerformanceMonitoring()
  }

  /**
   * Connect to WebSocket server
   */
  public async connect(): Promise<void> {
    if (this.state === 'connected' || this.state === 'connecting') {
      this.log('Already connected or connecting')
      return
    }

    return new Promise((resolve, reject) => {
      this.setState('connecting')
      this.healthStatus.connecting = true

      try {
        this.socket = new WebSocket(this.config.url, this.config.protocols)
        this.setupSocketHandlers(resolve, reject)

        // Connection timeout
        const timeout = setTimeout(() => {
          if (this.state === 'connecting') {
            this.handleError(new Error('Connection timeout'))
            reject(new Error('Connection timeout'))
          }
        }, this.config.timeout)

        this.socket.addEventListener('open', () => {
          clearTimeout(timeout)
        })
      } catch (error) {
        this.handleError(error as Error)
        reject(error)
      }
    })
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    this.setState('disconnected')
    this.stopReconnecting()
    this.stopHeartbeat()
    this.stopBatching()

    if (this.socket) {
      this.socket.close(1000, 'Client disconnect')
      this.socket = null
    }

    this.clearMessageQueue()
    this.updateHealthStatus()
  }

  /**
   * Subscribe to a channel
   */
  public subscribe(
    channel: string,
    callback: (data: any) => void,
    options: {
      errorCallback?: (error: Error) => void
      filter?: (data: any) => boolean
      transform?: (data: any) => any
    } = {}
  ): string {
    const id = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

    const subscription: WebSocketSubscription = {
      id,
      channel,
      callback,
      errorCallback: options.errorCallback,
      filter: options.filter,
      transform: options.transform,
      active: true,
      messageCount: 0,
    }

    this.subscriptions.set(id, subscription)

    // Send subscription message if connected
    if (this.state === 'connected') {
      this.sendMessage({
        type: 'subscribe',
        channel,
        data: { subscriptionId: id },
      })
    }

    this.log(`Subscribed to channel: ${channel} (${id})`)
    return id
  }

  /**
   * Unsubscribe from a channel
   */
  public unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId)
    if (!subscription) return

    subscription.active = false
    this.subscriptions.delete(subscriptionId)

    if (this.state === 'connected') {
      this.sendMessage({
        type: 'unsubscribe',
        channel: subscription.channel,
        data: { subscriptionId },
      })
    }

    this.log(`Unsubscribed from channel: ${subscription.channel} (${subscriptionId})`)
  }

  /**
   * Send a message
   */
  public sendMessage(
    message: Partial<WebSocketMessage>,
    priority: 'low' | 'normal' | 'high' | 'critical' = 'normal'
  ): Promise<void> {
    const fullMessage: WebSocketMessage = {
      id: `msg_${++this.messageId}`,
      type: message.type || 'data',
      channel: message.channel,
      data: message.data,
      timestamp: Date.now(),
      priority,
      retry: priority === 'critical' || priority === 'high',
      maxRetries: priority === 'critical' ? 5 : 3,
      retryCount: 0,
      ...message,
    }

    return new Promise((resolve, reject) => {
      if (this.state !== 'connected') {
        if (fullMessage.retry) {
          this.queueMessage(fullMessage)
          this.log(`Message queued for retry: ${fullMessage.id}`)
          resolve()
        } else {
          reject(new Error('WebSocket not connected'))
        }
        return
      }

      try {
        if (this.config.enableBatching && priority !== 'critical') {
          this.batchMessage(fullMessage)
          resolve()
        } else {
          this.sendMessageNow(fullMessage)
          resolve()
        }
      } catch (error) {
        if (fullMessage.retry && (fullMessage.retryCount || 0) < (fullMessage.maxRetries || 0)) {
          fullMessage.retryCount = (fullMessage.retryCount || 0) + 1
          this.queueMessage(fullMessage)
          this.log(
            `Message queued for retry (${fullMessage.retryCount}/${fullMessage.maxRetries}): ${fullMessage.id}`
          )
          resolve()
        } else {
          this.handleError(error as Error, `Failed to send message: ${fullMessage.id}`)
          reject(error)
        }
      }
    })
  }

  /**
   * Get connection state
   */
  public getState(): ConnectionState {
    return this.state
  }

  /**
   * Get connection metrics
   */
  public getMetrics(): ConnectionMetrics {
    return { ...this.metrics }
  }

  /**
   * Get health status
   */
  public getHealthStatus(): HealthStatus {
    return { ...this.healthStatus }
  }

  /**
   * Get active subscriptions
   */
  public getSubscriptions(): WebSocketSubscription[] {
    return Array.from(this.subscriptions.values())
  }

  /**
   * Add event listener
   */
  public addEventListener(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set())
    }
    this.listeners.get(event)!.add(callback)
  }

  /**
   * Remove event listener
   */
  public removeEventListener(event: string, callback: Function): void {
    const listeners = this.listeners.get(event)
    if (listeners) {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.listeners.delete(event)
      }
    }
  }

  /**
   * Setup socket event handlers
   */
  private setupSocketHandlers(resolve: Function, reject: Function): void {
    if (!this.socket) return

    this.socket.addEventListener('open', (event) => {
      this.handleOpen(event)
      resolve()
    })

    this.socket.addEventListener('message', (event) => {
      this.handleMessage(event)
    })

    this.socket.addEventListener('close', (event) => {
      this.handleClose(event)
    })

    this.socket.addEventListener('error', (_event) => {
      this.handleError(new Error('WebSocket error'), 'Socket error event')
      reject(new Error('WebSocket connection failed'))
    })
  }

  /**
   * Handle WebSocket open event
   */
  private handleOpen(event: Event): void {
    this.setState('connected')
    this.healthStatus.connecting = false
    this.healthStatus.connected = true
    this.metrics.connectTime = Date.now()
    this.reconnectAttempts = 0

    this.log('WebSocket connected')
    this.emit('connected', { event, metrics: this.metrics })

    // Resubscribe to channels
    this.resubscribeChannels()

    // Send queued messages
    this.processMessageQueue()

    // Start heartbeat
    this.startHeartbeat()

    // Start batching if enabled
    if (this.config.enableBatching) {
      this.startBatching()
    }

    this.updateHealthStatus()
  }

  /**
   * Handle WebSocket message event
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const startTime = performance.now()
      this.metrics.messagesReceived++
      this.metrics.lastMessageTime = Date.now()
      this.metrics.bytesTransferred += event.data.length

      let data: any

      // Handle compressed messages
      if (
        this.compressionEnabled &&
        typeof event.data === 'string' &&
        event.data.startsWith('compressed:')
      ) {
        data = this.decompressMessage(event.data.substring(11))
      } else {
        data = typeof event.data === 'string' ? JSON.parse(event.data) : event.data
      }

      // Handle system messages
      if (data.type === 'pong') {
        this.handlePong(data)
        return
      }

      if (data.type === 'error') {
        this.handleServerError(data)
        return
      }

      if (data.type === 'compression') {
        this.compressionEnabled = data.enabled
        this.log(`Compression ${data.enabled ? 'enabled' : 'disabled'}`)
        return
      }

      // Route message to subscribers
      if (data.channel) {
        this.routeMessage(data)
      }

      // Measure latency if message has timestamp
      if (data.timestamp) {
        const latency = Date.now() - data.timestamp
        this.recordLatency(latency)
      }

      const processingTime = performance.now() - startTime
      this.log(`Message processed in ${processingTime.toFixed(2)}ms`)

      this.emit('message', data)
      this.updateHealthStatus()
    } catch (error) {
      this.handleError(error as Error, 'Failed to process message')
    }
  }

  /**
   * Handle WebSocket close event
   */
  private handleClose(event: CloseEvent): void {
    this.setState('disconnected')
    this.healthStatus.connected = false
    this.healthStatus.connecting = false

    this.log(`WebSocket closed: ${event.code} - ${event.reason}`)
    this.emit('disconnected', { code: event.code, reason: event.reason })

    this.stopHeartbeat()
    this.stopBatching()

    // Auto-reconnect if enabled and not a clean close
    if (
      this.config.reconnect &&
      event.code !== 1000 &&
      this.reconnectAttempts < this.config.maxReconnectAttempts
    ) {
      this.scheduleReconnect()
    }

    this.updateHealthStatus()
  }

  /**
   * Handle errors
   */
  private handleError(error: Error, context: string = 'Unknown'): void {
    this.metrics.errorCount++

    const errorRecord = {
      timestamp: Date.now(),
      error: error.message,
      context,
    }

    this.errorHistory.push(errorRecord)
    if (this.errorHistory.length > this.maxErrorHistory) {
      this.errorHistory.shift()
    }

    this.healthStatus.errors.unshift(error.message)
    if (this.healthStatus.errors.length > 10) {
      this.healthStatus.errors = this.healthStatus.errors.slice(0, 10)
    }

    this.log(`Error: ${error.message} (${context})`)
    this.emit('error', { error, context, metrics: this.metrics })
    this.updateHealthStatus()
  }

  /**
   * Handle server error messages
   */
  private handleServerError(data: any): void {
    const error = new Error(data.message || 'Server error')
    this.handleError(error, 'Server error')
  }

  /**
   * Handle pong messages
   */
  private handlePong(data: any): void {
    if (data.timestamp) {
      const latency = Date.now() - data.timestamp
      this.healthStatus.lastPing = Date.now()
      this.healthStatus.latency = latency
      this.recordLatency(latency)
    }
  }

  /**
   * Record latency measurement
   */
  private recordLatency(latency: number): void {
    this.latencyMeasurements.push(latency)
    if (this.latencyMeasurements.length > this.maxLatencyMeasurements) {
      this.latencyMeasurements.shift()
    }

    this.metrics.averageLatency =
      this.latencyMeasurements.reduce((a, b) => a + b, 0) / this.latencyMeasurements.length
  }

  /**
   * Route message to appropriate subscribers
   */
  private routeMessage(data: any): void {
    for (const subscription of this.subscriptions.values()) {
      if (!subscription.active || subscription.channel !== data.channel) {
        continue
      }

      try {
        // Apply filter if present
        if (subscription.filter && !subscription.filter(data.data)) {
          continue
        }

        // Apply transform if present
        let processedData = data.data
        if (subscription.transform) {
          processedData = subscription.transform(processedData)
        }

        subscription.messageCount++
        subscription.lastMessageTime = Date.now()
        subscription.callback(processedData)
      } catch (error) {
        if (subscription.errorCallback) {
          subscription.errorCallback(error as Error)
        } else {
          this.handleError(error as Error, `Subscription callback error: ${subscription.id}`)
        }
      }
    }
  }

  /**
   * Set connection state
   */
  private setState(state: ConnectionState): void {
    const previousState = this.state
    this.state = state

    if (previousState !== state) {
      this.log(`State changed: ${previousState} -> ${state}`)
      this.emit('stateChange', { previousState, currentState: state })
    }
  }

  /**
   * Schedule reconnection
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimer) return

    this.setState('reconnecting')
    this.reconnectAttempts++
    this.metrics.reconnectCount++

    const delay = Math.min(
      this.config.reconnectInterval *
        Math.pow(this.config.reconnectBackoffMultiplier, this.reconnectAttempts - 1),
      this.config.maxReconnectInterval
    )

    this.log(
      `Scheduling reconnect attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts} in ${delay}ms`
    )

    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null
      this.connect().catch((error) => {
        this.log(`Reconnect attempt ${this.reconnectAttempts} failed: ${error.message}`)

        if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
          this.scheduleReconnect()
        } else {
          this.setState('error')
          this.emit('maxReconnectAttemptsReached', { attempts: this.reconnectAttempts })
        }
      })
    }, delay)
  }

  /**
   * Stop reconnecting
   */
  private stopReconnecting(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.reconnectAttempts = 0
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    if (this.heartbeatTimer) return

    this.heartbeatTimer = window.setInterval(() => {
      if (this.state === 'connected') {
        this.sendMessage(
          {
            type: 'ping',
            data: { timestamp: Date.now() },
          },
          'high'
        )
      }
    }, this.config.heartbeatInterval)
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  /**
   * Start message batching
   */
  private startBatching(): void {
    if (this.batchTimer || !this.config.enableBatching) return

    this.batchTimer = window.setInterval(() => {
      if (this.batchQueue.length > 0) {
        this.sendBatch()
      }
    }, this.config.batchInterval)
  }

  /**
   * Stop message batching
   */
  private stopBatching(): void {
    if (this.batchTimer) {
      clearInterval(this.batchTimer)
      this.batchTimer = null
    }

    // Send any remaining batched messages
    if (this.batchQueue.length > 0) {
      this.sendBatch()
    }
  }

  /**
   * Add message to batch queue
   */
  private batchMessage(message: WebSocketMessage): void {
    this.batchQueue.push(message)

    // Send immediately if batch is full
    if (this.batchQueue.length >= 10) {
      this.sendBatch()
    }
  }

  /**
   * Send batched messages
   */
  private sendBatch(): void {
    if (this.batchQueue.length === 0) return

    const batch = {
      type: 'batch',
      messages: this.batchQueue.splice(0),
      timestamp: Date.now(),
    }

    this.sendMessageNow(batch as any)
  }

  /**
   * Send message immediately
   */
  private sendMessageNow(message: WebSocketMessage | any): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected')
    }

    let payload = JSON.stringify(message)

    // Compress if enabled and message is large
    if (this.config.enableCompression && this.compressionEnabled && payload.length > 1024) {
      payload = 'compressed:' + this.compressMessage(payload)
    }

    this.socket.send(payload)
    this.metrics.messagesSent++
    this.metrics.bytesTransferred += payload.length

    this.log(`Message sent: ${message.type} (${payload.length} bytes)`)
  }

  /**
   * Queue message for later sending
   */
  private queueMessage(message: WebSocketMessage): void {
    this.messageQueue.push(message)

    // Remove oldest messages if queue is full
    if (this.messageQueue.length > this.config.messageQueueSize) {
      this.messageQueue.shift()
      this.log('Message queue full, removing oldest message')
    }
  }

  /**
   * Process queued messages
   */
  private processMessageQueue(): void {
    const messages = this.messageQueue.splice(0)

    for (const message of messages) {
      try {
        this.sendMessageNow(message)
      } catch (error) {
        this.handleError(error as Error, `Failed to send queued message: ${message.id}`)
      }
    }

    this.log(`Processed ${messages.length} queued messages`)
  }

  /**
   * Clear message queue
   */
  private clearMessageQueue(): void {
    this.messageQueue = []
    this.batchQueue = []
    this.pendingMessages.clear()
  }

  /**
   * Resubscribe to all channels
   */
  private resubscribeChannels(): void {
    for (const subscription of this.subscriptions.values()) {
      if (subscription.active) {
        this.sendMessage({
          type: 'subscribe',
          channel: subscription.channel,
          data: { subscriptionId: subscription.id },
        })
      }
    }

    this.log(`Resubscribed to ${this.subscriptions.size} channels`)
  }

  /**
   * Update health status
   */
  private updateHealthStatus(): void {
    const now = Date.now()

    // Update connection duration
    if (this.metrics.connectTime > 0) {
      this.metrics.connectionDuration = now - this.metrics.connectTime
    }

    // Determine connection quality
    if (!this.healthStatus.connected) {
      this.healthStatus.quality = 'critical'
    } else if (this.healthStatus.latency > 1000) {
      this.healthStatus.quality = 'poor'
    } else if (this.healthStatus.latency > 500) {
      this.healthStatus.quality = 'fair'
    } else if (this.healthStatus.latency > 200) {
      this.healthStatus.quality = 'good'
    } else {
      this.healthStatus.quality = 'excellent'
    }

    // Check for warnings
    this.healthStatus.warnings = []

    if (this.messageQueue.length > this.config.messageQueueSize * 0.8) {
      this.healthStatus.warnings.push('Message queue nearly full')
    }

    if (this.reconnectAttempts > 0) {
      this.healthStatus.warnings.push(`Reconnect attempts: ${this.reconnectAttempts}`)
    }

    if (this.healthStatus.latency > 200) {
      this.healthStatus.warnings.push(`High latency: ${this.healthStatus.latency}ms`)
    }
  }

  /**
   * Start performance monitoring
   */
  private startPerformanceMonitoring(): void {
    this.performanceTimer = window.setInterval(() => {
      this.updateHealthStatus()
      this.emit('metrics', this.getMetrics())
    }, 5000)
  }

  /**
   * Compress message
   */
  private compressMessage(message: string): string {
    // Simple compression simulation - in real implementation use actual compression
    return btoa(message)
  }

  /**
   * Decompress message
   */
  private decompressMessage(compressed: string): any {
    try {
      const decompressed = atob(compressed)
      return JSON.parse(decompressed)
    } catch (error) {
      throw new Error('Failed to decompress message')
    }
  }

  /**
   * Emit event to listeners
   */
  private emit(event: string, data?: any): void {
    const listeners = this.listeners.get(event)
    if (listeners) {
      for (const callback of listeners) {
        try {
          callback(data)
        } catch (error) {
          this.log(`Error in event listener for ${event}: ${error}`)
        }
      }
    }
  }

  /**
   * Log debug messages
   */
  private log(_message: string): void {
    if (this.config.debug) {
      // DEBUG: WebSocket manager debug logging
      // console.log(`[WebSocketManager] ${message}`)
    }
  }

  /**
   * Clean up resources
   */
  public destroy(): void {
    this.disconnect()

    if (this.performanceTimer) {
      clearInterval(this.performanceTimer)
    }

    this.listeners.clear()
    this.subscriptions.clear()
    this.clearMessageQueue()
  }
}
