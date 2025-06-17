/**
 * Comprehensive Error Handling and Recovery System
 *
 * Provides enterprise-grade error management and recovery:
 * - Centralized error capture and reporting
 * - Automatic error recovery strategies
 * - User-friendly error presentation
 * - Error analytics and monitoring
 * - Circuit breaker pattern implementation
 * - Retry mechanisms with backoff
 * - Error context preservation
 * - Health monitoring and diagnostics
 * - Performance impact tracking
 * - Integration with external monitoring services
 */

export interface ErrorContext {
  timestamp: Date
  userId?: string
  sessionId: string
  component: string
  action: string
  environment: 'development' | 'production' | 'staging'
  version: string
  userAgent: string
  url: string
  stackTrace?: string
  additionalData?: Record<string, any>
}

export interface ErrorDetails {
  id: string
  type: ErrorType
  severity: ErrorSeverity
  message: string
  code?: string
  originalError?: Error
  context: ErrorContext
  recoveryAttempts: number
  recovered: boolean
  reportedToService: boolean
  userNotified: boolean
  metadata: {
    fingerprint: string
    category: string
    tags: string[]
    breadcrumbs: Breadcrumb[]
  }
}

export type ErrorType =
  | 'network'
  | 'api'
  | 'validation'
  | 'authentication'
  | 'authorization'
  | 'runtime'
  | 'ui'
  | 'data'
  | 'external'
  | 'system'
  | 'configuration'
  | 'performance'

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical'

export interface Breadcrumb {
  id: string
  timestamp: Date
  category: 'navigation' | 'user' | 'http' | 'debug' | 'error' | 'info' | 'warning'
  message: string
  data?: Record<string, any>
  level: 'debug' | 'info' | 'warning' | 'error'
}

export interface RecoveryStrategy {
  id: string
  name: string
  description: string
  applicable: (error: ErrorDetails) => boolean
  execute: (error: ErrorDetails) => Promise<RecoveryResult>
  priority: number
  maxAttempts: number
  backoffStrategy: 'linear' | 'exponential' | 'fixed'
  baseDelay: number
  maxDelay: number
}

export interface RecoveryResult {
  success: boolean
  message: string
  data?: any
  nextStrategy?: string
  shouldRetry: boolean
  delay?: number
}

export interface CircuitBreakerState {
  state: 'closed' | 'open' | 'half-open'
  failures: number
  lastFailure: Date
  nextAttempt: Date
  halfOpenCalls: number
}

export interface CircuitBreakerConfig {
  failureThreshold: number
  recoveryTimeout: number
  monitoringPeriod: number
  halfOpenMaxCalls: number
}

export interface RetryConfig {
  maxAttempts: number
  baseDelay: number
  maxDelay: number
  backoffMultiplier: number
  jitter: boolean
}

export interface ErrorHandlerConfig {
  enableAutomaticRecovery: boolean
  enableUserNotifications: boolean
  enableAnalytics: boolean
  enableExternalReporting: boolean
  maxErrorsPerSession: number
  maxBreadcrumbs: number
  reportingEndpoint?: string
  reportingApiKey?: string
  ignoredErrors: string[]
  circuitBreaker: CircuitBreakerConfig
  defaultRetry: RetryConfig
}

export interface ErrorReport {
  errorId: string
  sessionId: string
  timestamp: Date
  environment: string
  version: string
  error: {
    type: string
    severity: string
    message: string
    stack?: string
    fingerprint: string
  }
  context: ErrorContext
  recovery: {
    attempted: boolean
    successful: boolean
    strategies: string[]
  }
  performance: {
    memoryUsage: number
    timing: Record<string, number>
  }
  breadcrumbs: Breadcrumb[]
}

export class ErrorHandlingManager {
  private config: ErrorHandlerConfig
  private errors = new Map<string, ErrorDetails>()
  private breadcrumbs: Breadcrumb[] = []
  private recoveryStrategies = new Map<string, RecoveryStrategy>()
  private circuitBreakers = new Map<string, CircuitBreakerState>()
  private sessionId: string
  private eventListeners = new Map<string, Set<(...args: any[]) => void>>()
  private isInitialized = false

  // Performance monitoring
  private performance = {
    errorsHandled: 0,
    recoveryAttempts: 0,
    successfulRecoveries: 0,
    userNotifications: 0,
    reportsGenerated: 0,
    averageRecoveryTime: 0,
  }

  // Circuit breaker state tracking
  private circuitBreakerStates = new Map<
    string,
    {
      state: 'closed' | 'open' | 'half-open'
      failures: number
      lastFailure: Date
      nextAttempt: Date
      halfOpenCalls: number
    }
  >()

  constructor(config: ErrorHandlerConfig) {
    this.config = config
    this.sessionId = this.generateSessionId()
    this.setupDefaultStrategies()
    this.setupGlobalHandlers()
  }

  /**
   * Initialize the error handling system
   */
  public initialize(): void {
    if (this.isInitialized) return

    try {
      this.addBreadcrumb('debug', 'Error handling system initializing')
      this.setupPerformanceMonitoring()
      this.setupHealthChecks()

      this.isInitialized = true
      this.addBreadcrumb('info', 'Error handling system initialized')
      this.emit('initialized')
    } catch (error) {
      console.error('Failed to initialize error handling system:', error)
      throw error
    }
  }

  /**
   * Handle an error with automatic recovery
   */
  public async handleError(
    error: Error | string,
    context: Partial<ErrorContext> = {},
    options: {
      type?: ErrorType
      severity?: ErrorSeverity
      component?: string
      action?: string
      skipRecovery?: boolean
      skipNotification?: boolean
    } = {}
  ): Promise<ErrorDetails> {
    const startTime = performance.now()

    try {
      // Create error details
      const errorDetails = this.createErrorDetails(error, context, options)

      // Store error
      this.errors.set(errorDetails.id, errorDetails)
      this.performance.errorsHandled++

      // Add breadcrumb
      this.addBreadcrumb('error', errorDetails.message, {
        errorId: errorDetails.id,
        type: errorDetails.type,
        severity: errorDetails.severity,
      })

      // Check circuit breaker
      if (this.isCircuitOpen(errorDetails.type)) {
        this.addBreadcrumb('warning', 'Circuit breaker is open, skipping recovery')
        await this.notifyUser(errorDetails)
        return errorDetails
      }

      // Attempt recovery if enabled
      if (!options.skipRecovery && this.config.enableAutomaticRecovery) {
        await this.attemptRecovery(errorDetails)
      }

      // Notify user if needed
      if (!options.skipNotification && this.shouldNotifyUser(errorDetails)) {
        await this.notifyUser(errorDetails)
      }

      // Report to external services
      if (this.config.enableExternalReporting) {
        await this.reportError(errorDetails)
      }

      // Update performance metrics
      const recoveryTime = performance.now() - startTime
      this.updatePerformanceMetrics(recoveryTime)

      // Emit events
      this.emit('errorHandled', errorDetails)

      return errorDetails
    } catch (handlingError) {
      console.error('Error in error handling:', handlingError)
      throw handlingError
    }
  }

  /**
   * Add a recovery strategy
   */
  public addRecoveryStrategy(strategy: RecoveryStrategy): void {
    this.recoveryStrategies.set(strategy.id, strategy)
    this.addBreadcrumb('debug', `Recovery strategy added: ${strategy.name}`)
  }

  /**
   * Remove a recovery strategy
   */
  public removeRecoveryStrategy(strategyId: string): void {
    this.recoveryStrategies.delete(strategyId)
    this.addBreadcrumb('debug', `Recovery strategy removed: ${strategyId}`)
  }

  /**
   * Add breadcrumb for debugging
   */
  public addBreadcrumb(
    category: Breadcrumb['category'],
    message: string,
    data?: Record<string, any>,
    level: Breadcrumb['level'] = 'info'
  ): void {
    const breadcrumb: Breadcrumb = {
      id: this.generateId(),
      timestamp: new Date(),
      category,
      message,
      data,
      level,
    }

    this.breadcrumbs.push(breadcrumb)

    // Maintain breadcrumb limit
    if (this.breadcrumbs.length > this.config.maxBreadcrumbs) {
      this.breadcrumbs.shift()
    }

    this.emit('breadcrumbAdded', breadcrumb)
  }

  /**
   * Get error by ID
   */
  public getError(errorId: string): ErrorDetails | undefined {
    return this.errors.get(errorId)
  }

  /**
   * Get all errors
   */
  public getAllErrors(): ErrorDetails[] {
    return Array.from(this.errors.values())
  }

  /**
   * Get errors by type
   */
  public getErrorsByType(type: ErrorType): ErrorDetails[] {
    return Array.from(this.errors.values()).filter((error) => error.type === type)
  }

  /**
   * Get recent breadcrumbs
   */
  public getBreadcrumbs(limit?: number): Breadcrumb[] {
    return limit ? this.breadcrumbs.slice(-limit) : [...this.breadcrumbs]
  }

  /**
   * Clear all errors
   */
  public clearErrors(): void {
    this.errors.clear()
    this.addBreadcrumb('debug', 'All errors cleared')
  }

  /**
   * Get performance metrics
   */
  public getPerformanceMetrics(): typeof this.performance {
    return { ...this.performance }
  }

  /**
   * Get circuit breaker status
   */
  public getCircuitBreakerStatus(): { [key: string]: any } {
    const status: { [key: string]: any } = {}

    for (const [key, state] of this.circuitBreakerStates.entries()) {
      status[key] = {
        state: state.state,
        failures: state.failures,
        lastFailure: state.lastFailure,
        nextAttempt: state.nextAttempt,
      }
    }

    return status
  }

  /**
   * Manual recovery trigger
   */
  public async triggerRecovery(errorId: string, strategyId?: string): Promise<boolean> {
    const error = this.errors.get(errorId)
    if (!error) {
      throw new Error(`Error not found: ${errorId}`)
    }

    if (strategyId) {
      const strategy = this.recoveryStrategies.get(strategyId)
      if (!strategy) {
        throw new Error(`Recovery strategy not found: ${strategyId}`)
      }

      return this.executeRecoveryStrategy(error, strategy)
    } else {
      return this.attemptRecovery(error)
    }
  }

  /**
   * Private methods
   */
  private createErrorDetails(
    error: Error | string,
    context: Partial<ErrorContext>,
    options: any
  ): ErrorDetails {
    const errorMessage = typeof error === 'string' ? error : error.message
    const originalError = typeof error === 'string' ? undefined : error

    const fullContext: ErrorContext = {
      timestamp: new Date(),
      sessionId: this.sessionId,
      component: options.component || 'unknown',
      action: options.action || 'unknown',
      environment: (process.env.NODE_ENV as any) ?? 'development',
      version: process.env.REACT_APP_VERSION || '1.0.0',
      userAgent: navigator.userAgent,
      url: window.location.href,
      stackTrace: originalError?.stack,
      ...context,
    }

    const errorDetails: ErrorDetails = {
      id: this.generateId(),
      type: options.type ?? this.inferErrorType(errorMessage, originalError),
      severity: options.severity || this.inferErrorSeverity(errorMessage, originalError),
      message: errorMessage,
      code: (originalError as any)?.code,
      originalError,
      context: fullContext,
      recoveryAttempts: 0,
      recovered: false,
      reportedToService: false,
      userNotified: false,
      metadata: {
        fingerprint: this.generateFingerprint(errorMessage, fullContext),
        category: this.categorizeError(errorMessage, originalError),
        tags: this.generateTags(errorMessage, fullContext),
        breadcrumbs: [...this.breadcrumbs],
      },
    }

    return errorDetails
  }

  private async attemptRecovery(error: ErrorDetails): Promise<boolean> {
    this.performance.recoveryAttempts++

    // Get applicable strategies
    const strategies = Array.from(this.recoveryStrategies.values())
      .filter((strategy) => strategy.applicable(error))
      .sort((a, b) => b.priority - a.priority)

    if (strategies.length === 0) {
      this.addBreadcrumb('warning', 'No recovery strategies available', {
        errorId: error.id,
        type: error.type,
      })
      return false
    }

    // Try each strategy
    for (const strategy of strategies) {
      if (error.recoveryAttempts >= strategy.maxAttempts) {
        continue
      }

      try {
        const success = await this.executeRecoveryStrategy(error, strategy)
        if (success) {
          error.recovered = true
          this.performance.successfulRecoveries++
          this.addBreadcrumb('info', 'Recovery successful', {
            errorId: error.id,
            strategy: strategy.name,
          })
          return true
        }
      } catch (recoveryError) {
        this.addBreadcrumb('error', 'Recovery strategy failed', {
          errorId: error.id,
          strategy: strategy.name,
          error: (recoveryError as Error).message,
        })
      }
    }

    return false
  }

  private async executeRecoveryStrategy(
    error: ErrorDetails,
    strategy: RecoveryStrategy
  ): Promise<boolean> {
    error.recoveryAttempts++

    this.addBreadcrumb('debug', `Executing recovery strategy: ${strategy.name}`, {
      errorId: error.id,
      attempt: error.recoveryAttempts,
    })

    // Calculate delay
    const delay = this.calculateDelay(strategy, error.recoveryAttempts)
    if (delay > 0) {
      await this.sleep(delay)
    }

    // Execute strategy
    const result = await strategy.execute(error)

    if (result.success) {
      this.recordCircuitBreakerSuccess(error.type)
      return true
    } else {
      this.recordCircuitBreakerFailure(error.type)

      if (result.shouldRetry && error.recoveryAttempts < strategy.maxAttempts) {
        const retryDelay = result.delay ?? this.calculateDelay(strategy, error.recoveryAttempts + 1)
        setTimeout(() => {
          void this.executeRecoveryStrategy(error, strategy)
        }, retryDelay)
      }

      return false
    }
  }

  private calculateDelay(strategy: RecoveryStrategy, attempt: number): number {
    let delay: number

    switch (strategy.backoffStrategy) {
      case 'linear':
        delay = strategy.baseDelay * attempt
        break
      case 'exponential':
        delay = strategy.baseDelay * Math.pow(2, attempt - 1)
        break
      case 'fixed':
      default:
        delay = strategy.baseDelay
        break
    }

    return Math.min(delay, strategy.maxDelay)
  }

  private shouldNotifyUser(error: ErrorDetails): boolean {
    if (!this.config.enableUserNotifications) return false
    if (error.userNotified) return false
    if (error.severity === 'low') return false
    if (error.recovered) return false

    return true
  }

  private notifyUser(error: ErrorDetails): void {
    try {
      const notification = this.createUserNotification(error)

      // Show browser notification if available
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(notification.title, {
          body: notification.message,
          icon: '/favicon.ico',
          tag: `error-${error.id}`,
        })
      }

      // Emit event for UI notification
      this.emit('userNotification', notification)

      error.userNotified = true
      this.performance.userNotifications++
    } catch (notificationError) {
      console.error('Failed to notify user:', notificationError)
    }
  }

  private createUserNotification(error: ErrorDetails): any {
    const messages = {
      network: {
        title: 'Connection Issue',
        message: 'Unable to connect to the server. Please check your internet connection.',
      },
      api: {
        title: 'Service Error',
        message: 'There was an issue with the trading service. Please try again.',
      },
      authentication: {
        title: 'Authentication Required',
        message: 'Please log in to continue using the trading dashboard.',
      },
      authorization: {
        title: 'Access Denied',
        message: "You don't have permission to perform this action.",
      },
      validation: {
        title: 'Invalid Input',
        message: 'Please check your input and try again.',
      },
      default: {
        title: 'Something went wrong',
        message: 'An unexpected error occurred. Our team has been notified.',
      },
    }

    const notification = messages[error.type] || messages.default

    return {
      id: error.id,
      type: error.type,
      severity: error.severity,
      title: notification.title,
      message: notification.message,
      actions: this.getNotificationActions(error),
      timestamp: new Date(),
    }
  }

  private getNotificationActions(error: ErrorDetails): any[] {
    const actions = []

    if (error.type === 'network') {
      actions.push({
        id: 'retry',
        label: 'Retry',
        action: () => this.triggerRecovery(error.id),
      })
    }

    if (error.severity === 'high' || error.severity === 'critical') {
      actions.push({
        id: 'details',
        label: 'Details',
        action: () => this.emit('showErrorDetails', error),
      })
    }

    return actions
  }

  private async reportError(error: ErrorDetails): Promise<void> {
    if (!this.config.reportingEndpoint || error.reportedToService) return

    try {
      const report: ErrorReport = {
        errorId: error.id,
        sessionId: this.sessionId,
        timestamp: new Date(),
        environment: error.context.environment,
        version: error.context.version,
        error: {
          type: error.type,
          severity: error.severity,
          message: error.message,
          stack: error.context.stackTrace,
          fingerprint: error.metadata.fingerprint,
        },
        context: error.context,
        recovery: {
          attempted: error.recoveryAttempts > 0,
          successful: error.recovered,
          strategies: Array.from(this.recoveryStrategies.keys()),
        },
        performance: {
          memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
          timing: this.getPerformanceTiming(),
        },
        breadcrumbs: error.metadata.breadcrumbs,
      }

      const response = await fetch(this.config.reportingEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.config.reportingApiKey}`,
          'X-Session-ID': this.sessionId,
        },
        body: JSON.stringify(report),
      })

      if (response.ok) {
        error.reportedToService = true
        this.performance.reportsGenerated++
        this.addBreadcrumb('debug', 'Error reported successfully', {
          errorId: error.id,
        })
      }
    } catch (reportingError) {
      console.error('Failed to report error:', reportingError)
    }
  }

  /**
   * Circuit breaker implementation
   */
  private isCircuitOpen(type: string): boolean {
    const state = this.circuitBreakerStates.get(type)
    if (!state) return false

    if (state.state === 'open') {
      if (Date.now() > state.nextAttempt.getTime()) {
        state.state = 'half-open'
        state.halfOpenCalls = 0
        return false
      }
      return true
    }

    return false
  }

  private recordCircuitBreakerFailure(type: string): void {
    let state = this.circuitBreakerStates.get(type)

    if (!state) {
      state = {
        state: 'closed',
        failures: 0,
        lastFailure: new Date(),
        nextAttempt: new Date(),
        halfOpenCalls: 0,
      }
      this.circuitBreakerStates.set(type, state)
    }

    state.failures++
    state.lastFailure = new Date()

    if (state.failures >= this.config.circuitBreaker.failureThreshold) {
      state.state = 'open'
      state.nextAttempt = new Date(Date.now() + this.config.circuitBreaker.recoveryTimeout)

      this.addBreadcrumb('warning', `Circuit breaker opened for ${type}`, {
        failures: state.failures,
        nextAttempt: state.nextAttempt,
      })
    }
  }

  private recordCircuitBreakerSuccess(type: string): void {
    const state = this.circuitBreakerStates.get(type)
    if (!state) return

    if (state.state === 'half-open') {
      state.halfOpenCalls++

      if (state.halfOpenCalls >= this.config.circuitBreaker.halfOpenMaxCalls) {
        state.state = 'closed'
        state.failures = 0
        state.halfOpenCalls = 0

        this.addBreadcrumb('info', `Circuit breaker closed for ${type}`)
      }
    } else if (state.state === 'closed') {
      state.failures = Math.max(0, state.failures - 1)
    }
  }

  /**
   * Setup and utility methods
   */
  private setupDefaultStrategies(): void {
    // Network retry strategy
    this.addRecoveryStrategy({
      id: 'network-retry',
      name: 'Network Retry',
      description: 'Retry network requests with exponential backoff',
      applicable: (error) => error.type === 'network' || error.type === 'api',
      execute: async (error) => {
        try {
          // Re-attempt the last network request
          if (error.context.additionalData?.lastRequest) {
            const request = error.context.additionalData.lastRequest
            const response = await fetch(request.url, request.options)

            if (response.ok) {
              return {
                success: true,
                message: 'Network request succeeded on retry',
                shouldRetry: false,
              }
            }
          }

          return { success: false, message: 'Network retry failed', shouldRetry: true }
        } catch (retryError) {
          return { success: false, message: 'Network retry failed', shouldRetry: true }
        }
      },
      priority: 10,
      maxAttempts: 3,
      backoffStrategy: 'exponential',
      baseDelay: 1000,
      maxDelay: 10000,
    })

    // Page refresh strategy
    this.addRecoveryStrategy({
      id: 'page-refresh',
      name: 'Page Refresh',
      description: 'Refresh the page to recover from critical errors',
      applicable: (error) => error.severity === 'critical' && error.type === 'runtime',
      execute: async (_error) => {
        if (confirm('A critical error occurred. Would you like to refresh the page?')) {
          window.location.reload()
          return { success: true, message: 'Page refresh initiated', shouldRetry: false }
        }
        return { success: false, message: 'User declined page refresh', shouldRetry: false }
      },
      priority: 1,
      maxAttempts: 1,
      backoffStrategy: 'fixed',
      baseDelay: 0,
      maxDelay: 0,
    })

    // Authentication retry strategy
    this.addRecoveryStrategy({
      id: 'auth-retry',
      name: 'Authentication Retry',
      description: 'Attempt to refresh authentication tokens',
      applicable: (error) => error.type === 'authentication',
      execute: async (_error) => {
        try {
          // Attempt to refresh auth tokens
          const refreshResult = await this.attemptAuthRefresh()
          if (refreshResult) {
            return {
              success: true,
              message: 'Authentication refreshed successfully',
              shouldRetry: false,
            }
          }
          return { success: false, message: 'Authentication refresh failed', shouldRetry: false }
        } catch (authError) {
          return { success: false, message: 'Authentication retry failed', shouldRetry: false }
        }
      },
      priority: 15,
      maxAttempts: 2,
      backoffStrategy: 'fixed',
      baseDelay: 2000,
      maxDelay: 2000,
    })
  }

  private attemptAuthRefresh(): Promise<boolean> {
    // Implementation would depend on your auth system
    // This is a placeholder
    return false
  }

  private setupGlobalHandlers(): void {
    // Global error handler
    window.addEventListener('error', (event) => {
      void this.handleError(
        event.error ?? event.message,
        {
          component: 'global',
          action: 'runtime-error',
          additionalData: {
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
          },
        },
        {
          type: 'runtime',
          severity: 'high',
        }
      )
    })

    // Promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      void this.handleError(
        event.reason,
        {
          component: 'global',
          action: 'unhandled-promise-rejection',
        },
        {
          type: 'runtime',
          severity: 'high',
        }
      )
    })

    // Network error detection
    window.addEventListener('online', () => {
      this.addBreadcrumb('info', 'Network connection restored')
      this.emit('networkOnline')
    })

    window.addEventListener('offline', () => {
      this.addBreadcrumb('warning', 'Network connection lost')
      this.emit('networkOffline')
    })
  }

  private setupPerformanceMonitoring(): void {
    // Monitor performance and detect issues
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        for (const entry of entries) {
          if (entry.entryType === 'navigation') {
            const navigation = entry as PerformanceNavigationTiming
            if (navigation.loadEventEnd - navigation.loadEventStart > 10000) {
              void this.handleError(
                'Slow page load detected',
                {
                  component: 'performance',
                  action: 'page-load',
                  additionalData: {
                    loadTime: navigation.loadEventEnd - navigation.loadEventStart,
                    timing: {
                      dns: navigation.domainLookupEnd - navigation.domainLookupStart,
                      tcp: navigation.connectEnd - navigation.connectStart,
                      request: navigation.responseStart - navigation.requestStart,
                      response: navigation.responseEnd - navigation.responseStart,
                      dom: navigation.domContentLoadedEventEnd - navigation.responseEnd,
                    },
                  },
                },
                {
                  type: 'performance',
                  severity: 'medium',
                }
              )
            }
          }
        }
      })

      observer.observe({ entryTypes: ['navigation', 'measure'] })
    }
  }

  private setupHealthChecks(): void {
    // Regular health checks
    setInterval(() => {
      void this.performHealthCheck()
    }, 60000) // Every minute
  }

  private performHealthCheck(): void {
    const checks = {
      memory: this.checkMemoryUsage(),
      errors: this.checkErrorRate(),
      performance: this.checkPerformance(),
    }

    for (const [check, result] of Object.entries(checks)) {
      if (!result.healthy) {
        void this.handleError(
          `Health check failed: ${check}`,
          {
            component: 'health-monitor',
            action: 'health-check',
            additionalData: result,
          },
          {
            type: 'system',
            severity: result.severity as ErrorSeverity,
          }
        )
      }
    }
  }

  private checkMemoryUsage(): { healthy: boolean; severity?: string; details?: any } {
    if (!(performance as any).memory) {
      return { healthy: true }
    }

    const memory = (performance as any).memory
    const usageRatio = memory.usedJSHeapSize / memory.jsHeapSizeLimit

    if (usageRatio > 0.9) {
      return {
        healthy: false,
        severity: 'critical',
        details: { usageRatio, usedMB: memory.usedJSHeapSize / 1024 / 1024 },
      }
    } else if (usageRatio > 0.7) {
      return {
        healthy: false,
        severity: 'medium',
        details: { usageRatio, usedMB: memory.usedJSHeapSize / 1024 / 1024 },
      }
    }

    return { healthy: true }
  }

  private checkErrorRate(): { healthy: boolean; severity?: string; details?: any } {
    const recentErrors = Array.from(this.errors.values()).filter(
      (error) => Date.now() - error.context.timestamp.getTime() < 300000 // Last 5 minutes
    )

    if (recentErrors.length > 10) {
      return {
        healthy: false,
        severity: 'high',
        details: { errorCount: recentErrors.length, timeWindow: '5 minutes' },
      }
    } else if (recentErrors.length > 5) {
      return {
        healthy: false,
        severity: 'medium',
        details: { errorCount: recentErrors.length, timeWindow: '5 minutes' },
      }
    }

    return { healthy: true }
  }

  private checkPerformance(): { healthy: boolean; severity?: string; details?: any } {
    if (this.performance.averageRecoveryTime > 10000) {
      return {
        healthy: false,
        severity: 'medium',
        details: { averageRecoveryTime: this.performance.averageRecoveryTime },
      }
    }

    return { healthy: true }
  }

  /**
   * Utility methods
   */
  private inferErrorType(message: string, error?: Error): ErrorType {
    const lowerMessage = message.toLowerCase()

    if (lowerMessage.includes('network') || lowerMessage.includes('fetch')) return 'network'
    if (lowerMessage.includes('unauthorized') || lowerMessage.includes('auth'))
      return 'authentication'
    if (lowerMessage.includes('forbidden') || lowerMessage.includes('permission'))
      return 'authorization'
    if (lowerMessage.includes('validation') || lowerMessage.includes('invalid')) return 'validation'
    if (lowerMessage.includes('api') || lowerMessage.includes('server')) return 'api'
    if (error && error.name === 'TypeError') return 'runtime'

    return 'system'
  }

  private inferErrorSeverity(message: string, _error?: Error): ErrorSeverity {
    const lowerMessage = message.toLowerCase()

    if (lowerMessage.includes('critical') || lowerMessage.includes('fatal')) return 'critical'
    if (lowerMessage.includes('unauthorized') || lowerMessage.includes('auth')) return 'high'
    if (lowerMessage.includes('network') || lowerMessage.includes('api')) return 'medium'
    if (lowerMessage.includes('validation')) return 'low'

    return 'medium'
  }

  private categorizeError(message: string, error?: Error): string {
    if (error?.name) return error.name

    const lowerMessage = message.toLowerCase()
    if (lowerMessage.includes('network')) return 'NetworkError'
    if (lowerMessage.includes('auth')) return 'AuthenticationError'
    if (lowerMessage.includes('validation')) return 'ValidationError'

    return 'GenericError'
  }

  private generateFingerprint(message: string, context: ErrorContext): string {
    const key = `${message}-${context.component}-${context.action}`
    return btoa(key).replace(/[+/=]/g, '').substring(0, 16)
  }

  private generateTags(message: string, context: ErrorContext): string[] {
    const tags = [context.component, context.action, context.environment]

    if (context.userId) tags.push(`user:${context.userId}`)
    if (context.url) tags.push(`page:${new URL(context.url).pathname}`)

    return tags.filter(Boolean)
  }

  private generateSessionId(): string {
    return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private generateId(): string {
    return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private getPerformanceTiming(): Record<string, number> {
    const timing = performance.timing
    return {
      domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
      loadComplete: timing.loadEventEnd - timing.navigationStart,
      domReady: timing.domComplete - timing.navigationStart,
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  private updatePerformanceMetrics(recoveryTime: number): void {
    const totalRecoveries =
      this.performance.successfulRecoveries + this.performance.recoveryAttempts
    if (totalRecoveries > 0) {
      this.performance.averageRecoveryTime =
        (this.performance.averageRecoveryTime * (totalRecoveries - 1) + recoveryTime) /
        totalRecoveries
    }
  }

  /**
   * Event handling
   */
  public addEventListener(event: string, callback: (...args: any[]) => void): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set())
    }
    this.eventListeners.get(event)!.add(callback)
  }

  public removeEventListener(event: string, callback: (...args: any[]) => void): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.delete(callback)
    }
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach((callback) => {
        try {
          callback(data)
        } catch (error) {
          console.error(`Error in error handling event listener for ${event}:`, error)
        }
      })
    }
  }

  /**
   * Cleanup
   */
  public destroy(): void {
    this.errors.clear()
    this.breadcrumbs = []
    this.eventListeners.clear()
    this.recoveryStrategies.clear()
    this.circuitBreakerStates.clear()
    this.isInitialized = false
  }
}

export default ErrorHandlingManager
