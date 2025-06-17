import './style.css'
import { DashboardUI } from './ui.ts'
import { DashboardWebSocket, type AllWebSocketMessages } from './websocket.ts'
import { TradingViewChart } from './tradingview.ts'
import { LLMDecisionCard, type LLMDecisionData } from './components/llm-decision-card.ts'
import {
  StatusIndicators,
  type ConnectionStatus,
  type BotStatus as IndicatorBotStatus,
  type MarketStatus,
  type PositionStatus,
} from './components/status-indicators.ts'
import { LLMMonitorDashboard } from './llm-monitor.ts'

// Phase 4 Enterprise Services
import { WebSocketManager } from './services/websocket-manager.ts'
import { NotificationSystem } from './services/notification-system.ts'
import { DashboardOrchestrator } from './services/dashboard-orchestrator.ts'
import { MobileOptimizationManager } from './services/mobile-optimization.ts'
import { DataPersistenceManager } from './services/data-persistence.ts'
import { ErrorHandlingManager } from './services/error-handling.ts'
import { SecurityManager } from './services/security-manager.ts'
import { PerformanceOptimizer } from './services/performance-optimizer.ts'
import { TestSuiteManager } from './tests/test-suite-manager.ts'

import type {
  DashboardConfig,
  BotStatus,
  MarketData,
  TradeAction,
  VuManchuIndicators,
  Position,
  RiskMetrics,
} from './types.ts'

/**
 * Debounce utility for performance optimization
 */
class Debouncer {
  private timers = new Map<string, number>()
  private readonly maxTimers = 20

  public debounce<T extends (...args: any[]) => any>(
    key: string,
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    return (...args: Parameters<T>) => {
      const existingTimer = this.timers.get(key)
      if (existingTimer) {
        clearTimeout(existingTimer)
      }

      const timer = window.setTimeout(() => {
        this.timers.delete(key)
        fn(...args)
      }, delay)

      this.timers.set(key, timer)

      // Cleanup old timers to prevent memory leaks
      if (this.timers.size > this.maxTimers) {
        const oldestKey = this.timers.keys().next().value
        const oldTimer = this.timers.get(oldestKey)
        if (oldTimer) clearTimeout(oldTimer)
        this.timers.delete(oldestKey)
      }
    }
  }

  public throttle<T extends (...args: any[]) => any>(
    key: string,
    fn: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    let lastCall = 0
    return (...args: Parameters<T>) => {
      const now = Date.now()
      if (now - lastCall >= delay) {
        lastCall = now
        fn(...args)
      }
    }
  }

  public clear(key: string): void {
    const timer = this.timers.get(key)
    if (timer) {
      clearTimeout(timer)
      this.timers.delete(key)
    }
  }

  public destroy(): void {
    this.timers.forEach((timer) => clearTimeout(timer))
    this.timers.clear()
  }
}

/**
 * Memory management utility
 */
class MemoryManager {
  private readonly MAX_MEMORY_MB = 100
  private readonly CHECK_INTERVAL = 30000 // 30 seconds
  private checkTimer: number | null = null
  private memoryWarningThreshold = 0.8 // 80% of max memory

  constructor() {
    this.startMemoryMonitoring()
  }

  private startMemoryMonitoring(): void {
    if (!this.supportsMemoryAPI()) return

    this.checkTimer = window.setInterval(() => {
      this.checkMemoryUsage()
    }, this.CHECK_INTERVAL)
  }

  private supportsMemoryAPI(): boolean {
    return 'memory' in performance && 'usedJSHeapSize' in (performance as any).memory
  }

  private checkMemoryUsage(): void {
    if (!this.supportsMemoryAPI()) return

    const memoryInfo = (performance as any).memory
    const usedMB = memoryInfo.usedJSHeapSize / (1024 * 1024)
    const totalMB = memoryInfo.totalJSHeapSize / (1024 * 1024)

    if (usedMB > this.MAX_MEMORY_MB * this.memoryWarningThreshold) {
      if (__DEV__) {
        console.warn(`Memory usage high: ${usedMB.toFixed(2)}MB / ${totalMB.toFixed(2)}MB`)
      }
      this.triggerGarbageCollection()
    }
  }

  private triggerGarbageCollection(): void {
    // Force garbage collection hints
    if (typeof window !== 'undefined' && 'gc' in window) {
      (window as any).gc()
    }

    // Clear caches that might be holding references
    if (typeof queueMicrotask !== 'undefined') {
      queueMicrotask(() => {
        // This can help trigger GC in some browsers
      })
    }
  }

  public getMemoryUsage(): { used: number; total: number; limit: number } | null {
    if (!this.supportsMemoryAPI()) return null

    const memoryInfo = (performance as any).memory
    return {
      used: memoryInfo.usedJSHeapSize / (1024 * 1024),
      total: memoryInfo.totalJSHeapSize / (1024 * 1024),
      limit: this.MAX_MEMORY_MB,
    }
  }

  public destroy(): void {
    if (this.checkTimer) {
      clearInterval(this.checkTimer)
      this.checkTimer = null
    }
  }
}

/**
 * Performance monitoring utility with memory optimization
 */
class PerformanceMonitor {
  private metrics = new Map<string, number>()
  private readonly maxMetrics = 50 // Reduced from 100
  private cleanupTimer: number | null = null
  private readonly CLEANUP_INTERVAL = 60000 // Clean up every minute

  constructor() {
    this.startPeriodicCleanup()
  }

  private startPeriodicCleanup(): void {
    this.cleanupTimer = window.setInterval(() => {
      this.performCleanup()
    }, this.CLEANUP_INTERVAL)
  }

  private performCleanup(): void {
    if (this.metrics.size > this.maxMetrics) {
      const entries = Array.from(this.metrics.entries())
      const toDelete = entries.slice(0, entries.length - this.maxMetrics)
      toDelete.forEach(([key]) => {
        if (!key.endsWith('_start')) {
          this.metrics.delete(key)
        }
      })
    }
  }

  public startTiming(label: string): void {
    this.metrics.set(`${label}_start`, performance.now())
  }

  public endTiming(label: string): number {
    const start = this.metrics.get(`${label}_start`)
    if (start === undefined) return 0

    const duration = performance.now() - start
    this.metrics.set(label, duration)
    this.metrics.delete(`${label}_start`)

    return duration
  }

  public getMetric(label: string): number | undefined {
    return this.metrics.get(label)
  }

  public logMetrics(): void {
    if (__DEV__) {
      console.group('Performance Metrics')
      this.metrics.forEach((value, key) => {
        if (!key.endsWith('_start')) {
        }
      })
      console.groupEnd()
    }
  }

  public destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    this.metrics.clear()
  }
}

/**
 * Page visibility handler for connection management
 */
class VisibilityHandler {
  private callbacks = new Set<(visible: boolean) => void>()
  private isVisible = !document.hidden
  private isInitialized = false
  private initializationTimeout: number | null = null
  private debounceTimer: number | null = null
  private lastVisibilityChange = 0
  private readonly DEBOUNCE_DELAY = 500 // 500ms minimum between visibility changes
  private readonly INIT_DELAY = 100 // Minimal delay to prevent false triggers during page load
  private processVisibilityChange!: (newVisible: boolean, source: string) => void

  constructor() {
    // Check browser compatibility for Page Visibility API
    if (typeof document.hidden === 'undefined') {
      console.warn('Page Visibility API not supported - using fallback visibility detection')
      this.isVisible = document.hasFocus()
    }

    // Delay initialization to avoid false triggers during page load
    this.initializationTimeout = window.setTimeout(() => {
      this.setupVisibilityListeners()
      this.isInitialized = true
    }, this.INIT_DELAY) // Longer delay to let TradingView and page fully settle
  }

  private setupVisibilityListeners(): void {
    const debouncedVisibilityChange = (newVisible: boolean, source: string): void => {
      if (!this.isInitialized) return

      const now = Date.now()

      // Clear existing debounce timer
      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer)
        this.debounceTimer = null
      }

      // Only proceed if visibility state actually changed and enough time has passed
      if (newVisible === this.isVisible) {
        return // No actual change, ignore
      }

      // Check if enough time has passed since last change
      if (now - this.lastVisibilityChange < this.DEBOUNCE_DELAY) {
        // Schedule the change for later
        this.debounceTimer = window.setTimeout(
          () => {
            this.processVisibilityChange(newVisible, source)
          },
          this.DEBOUNCE_DELAY - (now - this.lastVisibilityChange)
        )
        return
      }

      // Process the change immediately
      this.processVisibilityChange(newVisible, source)
    }

    const processVisibilityChange = (newVisible: boolean, _source: string): void => {
      this.lastVisibilityChange = Date.now()
      this.isVisible = newVisible
      this.callbacks.forEach((callback) => callback(newVisible))
    }

    this.processVisibilityChange = processVisibilityChange

    const handleVisibilityChange = (): void => {
      // Use the Page Visibility API as primary source (most reliable)
      const visible = !document.hidden
      debouncedVisibilityChange(visible, 'visibilitychange')
    }

    const handleFocus = (): void => {
      // Only trust focus events if the page is currently marked as hidden
      // and the document is not actually hidden
      if (!this.isVisible && !document.hidden) {
        debouncedVisibilityChange(true, 'focus')
      }
    }

    const handleBlur = (): void => {
      // For blur events, add extra delay and checks to avoid false positives
      // especially during TradingView interactions
      setTimeout(() => {
        if (!this.isInitialized) return

        const activeElement = document.activeElement
        const isIframeActive = activeElement && activeElement.tagName === 'IFRAME'

        // Only consider page hidden if:
        // 1. Document is actually hidden OR
        // 2. Document doesn't have focus AND active element is not an iframe
        const shouldBeHidden = document.hidden || (!document.hasFocus() && !isIframeActive)

        if (shouldBeHidden && this.isVisible) {
          debouncedVisibilityChange(false, 'blur')
        }
      }, 100) // Longer delay for blur to avoid TradingView focus issues
    }

    // Primary visibility API (most reliable)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    // Secondary focus/blur events (with enhanced safeguards)
    window.addEventListener('focus', handleFocus)
    window.addEventListener('blur', handleBlur)

    // Page lifecycle events (less frequent, more reliable)
    window.addEventListener('pageshow', () => {
      if (this.isInitialized && !this.isVisible) {
        debouncedVisibilityChange(true, 'pageshow')
      }
    })

    window.addEventListener('pagehide', () => {
      if (this.isInitialized && this.isVisible) {
        debouncedVisibilityChange(false, 'pagehide')
      }
    })
  }

  public onVisibilityChange(callback: (visible: boolean) => void): void {
    this.callbacks.add(callback)

    // If not initialized yet, don't trigger immediately
    // If initialized and the current state is different from expected, trigger once
    if (this.isInitialized && this.callbacks.size === 1) {
      // First callback - verify current state is correct
      const actualVisible = !document.hidden
      if (actualVisible !== this.isVisible) {
        this.isVisible = actualVisible
        callback(actualVisible)
      }
    }
  }

  public removeVisibilityCallback(callback: (visible: boolean) => void): void {
    this.callbacks.delete(callback)
  }

  public get visible(): boolean {
    return this.isVisible
  }

  public destroy(): void {
    if (this.initializationTimeout) {
      clearTimeout(this.initializationTimeout)
      this.initializationTimeout = null
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer)
      this.debounceTimer = null
    }
    this.callbacks.clear()
    this.isInitialized = false
  }
}

/**
 * AI Trading Bot Dashboard Main Application with Phase 4 Enterprise Services
 */
class DashboardApp {
  public ui: DashboardUI
  private websocket: DashboardWebSocket
  public chart: TradingViewChart | null = null
  private llmDecisionCard: LLMDecisionCard | null = null
  private llmMonitorDashboard: LLMMonitorDashboard | null = null
  private config: DashboardConfig
  private isInitialized = false
  private initializationPromise: Promise<void> | null = null
  public performanceMonitor: PerformanceMonitor
  private visibilityHandler: VisibilityHandler
  private debouncer: Debouncer
  private memoryManager: MemoryManager
  private lastMarketData: MarketData | null = null
  private lastIndicators: VuManchuIndicators | null = null
  private statusIndicators: StatusIndicators | null = null
  private systemHealthInterval: number | null = null
  private headerElements: any = {}
  private sidebarElements: any = {}

  // Phase 4 Enterprise Services
  private enterpriseWebSocketManager: WebSocketManager | null = null
  private notificationSystem: NotificationSystem | null = null
  private dashboardOrchestrator: DashboardOrchestrator | null = null
  private mobileOptimizer: MobileOptimizationManager | null = null
  private dataManager: DataPersistenceManager | null = null
  private errorHandler: ErrorHandlingManager | null = null
  private securityManager: SecurityManager | null = null
  private performanceOptimizer: PerformanceOptimizer | null = null
  private testManager: TestSuiteManager | null = null

  // Debounced methods for performance
  private debouncedUpdateMarketData: (data: MarketData) => void
  private debouncedUpdateIndicators: (indicators: VuManchuIndicators) => void
  private throttledLogUpdate: (entry: any) => void

  constructor() {
    // Configuration
    this.config = {
      websocket_url: this.getWebSocketUrl(),
      api_base_url: this.getApiBaseUrl(),
      default_symbol: 'BTC-USD',
      refresh_interval: 1000,
      chart_config: {
        container_id: 'tradingview-chart',
        symbol: 'DOGE-USD',
        interval: '1',
        library_path: '/charting_library/',
        theme: 'dark',
        autosize: true,
      },
    }

    // Initialize utilities
    this.performanceMonitor = new PerformanceMonitor()
    this.visibilityHandler = new VisibilityHandler()
    this.debouncer = new Debouncer()
    this.memoryManager = new MemoryManager()

    // Initialize components
    this.ui = new DashboardUI()
    this.websocket = new DashboardWebSocket(this.config.websocket_url)

    // Initialize Phase 4 Enterprise Services (lazy initialization in performInitialization)
    // Services will be initialized during performInitialization() for better error handling

    // Setup debounced methods for performance
    this.debouncedUpdateMarketData = this.debouncer.debounce(
      'market_data_update',
      (data: MarketData) => {
        this.ui.updateMarketData(data)
        this.chart?.updateMarketData(data)
      },
      100 // 100ms debounce
    )

    this.debouncedUpdateIndicators = this.debouncer.debounce(
      'indicators_update',
      (indicators: VuManchuIndicators) => {
        if (this.chart) {
          this.chart.updateIndicators(indicators)
        }
      },
      150 // 150ms debounce for indicators
    )

    this.throttledLogUpdate = this.debouncer.throttle(
      'log_update',
      (entry: any) => {
        if (__DEV__) {
          console.log('Throttled log update:', entry)
        }
      },
      500 // 500ms throttle for logs
    )

    // Setup page visibility handling
    this.setupVisibilityHandling()
  }

  /**
   * Initialize the dashboard application
   */
  public async initialize(): Promise<void> {
    // Prevent multiple initialization attempts
    if (this.isInitialized) {
      console.warn('Dashboard already initialized')
      return
    }

    if (this.initializationPromise) {
      return this.initializationPromise
    }

    this.initializationPromise = this.performInitialization()

    try {
      await this.initializationPromise
      this.isInitialized = true
    } catch (error) {
      // Reset so retry is possible
      this.initializationPromise = null
      throw error
    }
  }

  /**
   * Perform the actual initialization steps
   */
  private async performInitialization(): Promise<void> {
    this.performanceMonitor.startTiming('total_initialization')

    try {
      this.updateLoadingProgress('Starting dashboard...', 10)

      // Small delay for smoother progress animation
      await new Promise((resolve) => setTimeout(resolve, 100))
      this.updateLoadingProgress('Initializing UI components...', 25)

      // Step 1: Initialize UI and Status Indicators
      this.performanceMonitor.startTiming('ui_initialization')
      this.ui.initialize()
      this.initializeStatusIndicators()
      this.setupUIEventHandlers()

      // Initialize LLM Decision Card
      try {
        this.llmDecisionCard = new LLMDecisionCard('llm-decision-container')
      } catch (error) {
        console.warn('Failed to initialize LLM Decision Card:', error)
        // Continue without the decision card
      }

      const _uiTime = this.performanceMonitor.endTiming('ui_initialization')
      this.updateLoadingProgress('UI ready', 40)

      // Step 2: Initialize TradingView chart (non-blocking)
      this.updateLoadingProgress('Loading charts...', 50)
      this.performanceMonitor.startTiming('chart_initialization')

      // Start chart initialization in background without blocking
      const chartInitPromise = this.initializeChartNonBlocking().catch((error) => {
        console.warn('Chart initialization failed:', error)
        return null // Continue without chart
      })

      // Step 3: Initialize Phase 4 Enterprise Services
      this.updateLoadingProgress('Initializing enterprise services...', 60)
      this.performanceMonitor.startTiming('enterprise_services_setup')
      await this.initializeEnterpriseServices()
      const _servicesTime = this.performanceMonitor.endTiming('enterprise_services_setup')

      // Step 4: Set up WebSocket connection
      this.updateLoadingProgress('Connecting to bot...', 75)
      this.performanceMonitor.startTiming('websocket_setup')
      this.setupWebSocketHandlers()

      // WebSocket connection with non-blocking timeout
      const connectionPromise = new Promise<void>((resolve) => {
        let connected = false
        const timeout = setTimeout(() => {
          if (!connected) {
            console.warn('WebSocket connection timeout - continuing in offline mode')
            resolve() // Resolve instead of reject to continue initialization
          }
        }, 5000) // 5 second timeout

        const handleConnection = (status: string) => {
          if (status === 'connected' && !connected) {
            connected = true
            clearTimeout(timeout)
            resolve()
          }
        }

        // Listen for connection status
        this.websocket.onConnectionStatusChange(handleConnection)
        this.websocket.connect()
      })

      await connectionPromise
      const _wsTime = this.performanceMonitor.endTiming('websocket_setup')

      // Step 5: Finalize initialization
      this.updateLoadingProgress('Finalizing...', 90)

      await new Promise((resolve) => setTimeout(resolve, 200)) // Brief pause for smooth UX

      this.updateLoadingProgress('Ready!', 100)
      await new Promise((resolve) => setTimeout(resolve, 150))

      this.hideLoadingScreen()

      const _totalTime = this.performanceMonitor.endTiming('total_initialization')

      // Log performance metrics in development
      if (window.location.hostname === 'localhost') {
        this.performanceMonitor.logMetrics()
      }

      this.ui.log('info', 'Dashboard fully initialized and ready', 'System')

      // Handle chart initialization result in background
      chartInitPromise
        .then(() => {
          const _chartTime = this.performanceMonitor.endTiming('chart_initialization')
          this.ui.log('info', 'TradingView chart loaded successfully', 'Chart')
        })
        .catch((chartError) => {
          console.warn('Chart background initialization failed:', chartError)
          this.performanceMonitor.endTiming('chart_initialization')
          this.ui.log(
            'warn',
            'Chart failed to load - dashboard continues with limited functionality',
            'Chart'
          )
        })
    } catch (error) {
      this.performanceMonitor.endTiming('total_initialization')
      console.error('❌ Failed to initialize dashboard:', error)

      const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error'
      this.showInitializationError(errorMessage)
      throw error
    }
  }

  /**
   * Initialize Phase 4 Enterprise Services
   */
  private async initializeEnterpriseServices(): Promise<void> {
    try {
      // Initialize Error Handler first (handles other service failures)
      this.errorHandler = new ErrorHandlingManager({
        enableAutomaticRecovery: true,
        enableUserNotifications: true,
        enableAnalytics: false,
        enableExternalReporting: false,
        maxErrorsPerSession: 100,
        maxBreadcrumbs: 50,
        ignoredErrors: ['ChunkLoadError', 'NetworkError'],
        circuitBreaker: {
          failureThreshold: 5,
          recoveryTimeout: 60000,
          monitoringPeriod: 300000,
          halfOpenMaxCalls: 3,
        },
        defaultRetry: {
          maxAttempts: 3,
          baseDelay: 1000,
          maxDelay: 10000,
          backoffMultiplier: 2,
          jitter: true,
        },
      })

      // Initialize Security Manager
      this.securityManager = new SecurityManager({
        sessionTimeout: 3600000, // 1 hour
        maxFailedAttempts: 5,
        lockoutDuration: 300000, // 5 minutes
        enableBiometric: true,
        enableMFA: false,
        tokenRefreshThreshold: 300000, // 5 minutes
        enableCSRF: true,
        enableRateLimiting: true,
        rateLimitRequests: 100,
        rateLimitWindow: 60000,
      })

      // Initialize Performance Optimizer
      this.performanceOptimizer = new PerformanceOptimizer({
        enableCaching: true,
        enableCompression: true,
        enableMetrics: true,
        cacheSize: 100,
        compressionThreshold: 1024,
        metricsInterval: 30000,
      })

      // Initialize Data Persistence Manager
      this.dataManager = new DataPersistenceManager({
        dbName: 'TradingBotDashboard',
        version: 1,
        enableCompression: true,
        enableEncryption: false,
        maxCacheSize: 50 * 1024 * 1024, // 50MB
      })
      await this.dataManager.initialize()

      // Initialize Enterprise WebSocket Manager (enhanced WebSocket system)
      this.enterpriseWebSocketManager = new WebSocketManager({
        url: this.config.websocket_url,
        maxReconnectAttempts: 10,
        reconnectInterval: 1000,
        heartbeatInterval: 30000,
        messageQueueSize: 1000,
        enableCompression: true,
        enableBatching: true,
        batchInterval: 100,
      })

      // Initialize Notification System
      this.notificationSystem = new NotificationSystem(
        this.config.api_base_url || window.location.origin,
        ''
      )

      // Initialize Mobile Optimization Manager
      this.mobileOptimizer = new MobileOptimizationManager({
        enableTouchGestures: true,
        enableOfflineMode: true,
        enablePWA: true,
        touchSensitivity: 10,
        batteryThreshold: 0.2,
        connectionThreshold: 'slow-2g',
      })

      // Initialize Dashboard Orchestrator (coordinates all services)
      this.dashboardOrchestrator = new DashboardOrchestrator({
        enableAutoSync: true,
        syncInterval: 30000,
        enableMetrics: true,
        metricsInterval: 60000,
        enableHealthChecks: true,
        healthCheckInterval: 30000,
      })

      // Initialize Test Suite Manager (in development mode)
      if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        this.testManager = new TestSuiteManager({
          enableCoverage: true,
          enablePerformanceTests: true,
          enableIntegrationTests: true,
          testTimeout: 30000,
        })
      }

      this.ui.log('info', 'Phase 4 Enterprise Services initialized successfully', 'Enterprise')
    } catch (error) {
      console.error('Failed to initialize enterprise services:', error)
      this.ui.log(
        'error',
        `Enterprise services initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        'Enterprise'
      )

      // Continue without enterprise services - core functionality should still work
      this.handleGracefulDegradation(
        'enterprise-services',
        error instanceof Error ? error : new Error('Unknown enterprise services error')
      )
    }
  }

  /**
   * Initialize Status Indicators
   */
  private initializeStatusIndicators(): void {
    // Create status indicators instance
    this.statusIndicators = new StatusIndicators()

    // Initialize with default states
    this.statusIndicators.connectionStatus = { websocket: 'disconnected' }
    this.statusIndicators.botStatus = { state: 'initializing' }
    this.statusIndicators.marketStatus = { state: 'closed' }
    this.statusIndicators.positionStatus = { state: 'flat' }

    // Render status indicators in designated layout areas
    this.renderStatusIndicatorsInLayout()

    // Start system health monitoring
    this.startSystemHealthMonitoring()
  }

  /**
   * Render status indicators in their designated layout areas
   */
  private renderStatusIndicatorsInLayout(): void {
    if (!this.statusIndicators) return

    // Create dedicated status indicator components for different layout areas
    this.createHeaderStatusIndicators()
    this.createSidebarHealthIndicators()
  }

  /**
   * Create status indicators for the header
   */
  private createHeaderStatusIndicators(): void {
    const headerContainer = document.getElementById('header-status-indicators')
    if (!headerContainer) return

    // Create individual status elements
    const connectionBadge = this.createConnectionBadge()
    const marketStatus = this.createMarketStatus()
    const positionStatus = this.createPositionStatus()

    headerContainer.style.cssText = 'display: flex; align-items: center; gap: 12px;'
    headerContainer.appendChild(connectionBadge)
    headerContainer.appendChild(marketStatus)
    headerContainer.appendChild(positionStatus)

    // Store references for updates
    this.headerElements = { connectionBadge, marketStatus, positionStatus }
  }

  /**
   * Create health indicators for the sidebar
   */
  private createSidebarHealthIndicators(): void {
    const sidebarContainer = document.getElementById('sidebar-health-indicators')
    if (!sidebarContainer) return

    // Create individual health elements
    const botHealth = this.createBotHealth()
    const systemHealth = this.createSystemHealth()
    const soundToggle = this.createSoundToggle()
    const alertHistory = this.createAlertHistory()

    sidebarContainer.style.cssText = 'display: flex; flex-direction: column; gap: 12px;'
    sidebarContainer.appendChild(botHealth)
    sidebarContainer.appendChild(systemHealth)
    sidebarContainer.appendChild(soundToggle)
    sidebarContainer.appendChild(alertHistory)

    // Store references for updates
    this.sidebarElements = { botHealth, systemHealth, soundToggle, alertHistory }
  }

  /**
   * Create connection badge element
   */
  private createConnectionBadge(): HTMLElement {
    const element = document.createElement('div')
    element.className = 'connection-badge disconnected'
    element.innerHTML = `
      <span class="pulse"></span>
      <span>LIVE</span>
      <span class="latency-indicator" id="latency-display"></span>
    `
    return element
  }

  /**
   * Create market status element
   */
  private createMarketStatus(): HTMLElement {
    const element = document.createElement('div')
    element.className = 'market-status'
    element.innerHTML = `
      <span class="market-dot closed"></span>
      <span>Market closed</span>
    `
    return element
  }

  /**
   * Create position status element
   */
  private createPositionStatus(): HTMLElement {
    const element = document.createElement('div')
    element.className = 'position-badge flat'
    element.innerHTML = `
      <span>💤</span>
      <span>flat</span>
    `
    return element
  }

  /**
   * Create bot health element
   */
  private createBotHealth(): HTMLElement {
    const element = document.createElement('div')
    element.className = 'bot-health'
    element.innerHTML = `
      <div class="health-icon initializing">🔄</div>
      <div>
        <div style="font-weight: 500;">Bot initializing</div>
        <div style="font-size: 12px; color: #9ca3af;" id="bot-status-message"></div>
      </div>
    `
    return element
  }

  /**
   * Create system health element
   */
  private createSystemHealth(): HTMLElement {
    const element = document.createElement('div')
    element.className = 'system-health'
    element.innerHTML = `
      <div class="health-meter" id="cpu-meter" style="display: none;">
        <span class="meter-label">CPU</span>
        <div class="meter-bar">
          <div class="meter-fill low" id="cpu-fill"></div>
        </div>
        <span class="meter-value" id="cpu-value">0%</span>
      </div>
      <div class="health-meter" id="memory-meter" style="display: none;">
        <span class="meter-label">Memory</span>
        <div class="meter-bar">
          <div class="meter-fill low" id="memory-fill"></div>
        </div>
        <span class="meter-value" id="memory-value">0%</span>
      </div>
    `
    return element
  }

  /**
   * Create sound toggle element
   */
  private createSoundToggle(): HTMLElement {
    const element = document.createElement('button')
    element.className = 'sound-toggle enabled'
    element.innerHTML = `
      <span>🔊</span>
      <span>Sounds On</span>
    `
    element.addEventListener('click', () => {
      if (this.statusIndicators) {
        this.statusIndicators.soundEnabled = !this.statusIndicators.soundEnabled
        this.updateSoundToggle()
      }
    })
    return element
  }

  /**
   * Create alert history element
   */
  private createAlertHistory(): HTMLElement {
    const element = document.createElement('button')
    element.className = 'sound-toggle'
    element.innerHTML = `
      <span>📜</span>
      <span id="alert-count">History (0)</span>
    `
    element.addEventListener('click', () => {
      if (this.statusIndicators) {
        this.statusIndicators.showAlertHistory = !this.statusIndicators.showAlertHistory
      }
    })
    return element
  }

  /**
   * Start monitoring system health metrics
   */
  private startSystemHealthMonitoring(): void {
    // Simulate system metrics (in production, these would come from the backend)
    this.systemHealthInterval = window.setInterval(() => {
      if (this.statusIndicators && this.isInitialized) {
        // Get memory usage estimate
        const memoryUsage = (performance as any).memory
          ? Math.round(
              ((performance as any).memory.usedJSHeapSize /
                (performance as any).memory.totalJSHeapSize) *
                100
            )
          : undefined

        // Update system health in status indicators component
        this.statusIndicators.systemHealth = {
          memory: memoryUsage,
          // CPU usage would come from backend
        }

        // Update visual display
        this.updateSystemHealthDisplay(undefined, memoryUsage)

        // Update alert count
        this.updateAlertCount(this.statusIndicators.alerts.length)
      }
    }, 5000)
  }

  /**
   * Update WebSocket latency
   */
  private updateLatency(latency: number): void {
    if (this.statusIndicators) {
      this.statusIndicators.connectionStatus = {
        ...this.statusIndicators.connectionStatus,
        latency,
      }
    }

    // Update latency display in header
    this.updateLatencyDisplay(latency)
  }

  /**
   * Update latency display in header
   */
  private updateLatencyDisplay(latency: number): void {
    const latencyDisplay = document.getElementById('latency-display')
    if (!latencyDisplay) return

    const latencyClass =
      latency < 100 ? 'latency-good' : latency < 500 ? 'latency-medium' : 'latency-poor'
    const latencyText = latency < 1000 ? `${latency}ms` : `${(latency / 1000).toFixed(1)}s`

    latencyDisplay.className = `latency-indicator ${latencyClass}`
    latencyDisplay.innerHTML = `
      <span class="latency-bar active"></span>
      <span class="latency-bar ${latency < 500 ? 'active' : ''}"></span>
      <span class="latency-bar ${latency < 100 ? 'active' : ''}"></span>
      <span>${latencyText}</span>
    `
  }

  /**
   * Update connection status display
   */
  private updateConnectionDisplay(status: ConnectionStatus['websocket']): void {
    if (this.headerElements.connectionBadge) {
      this.headerElements.connectionBadge.className = `connection-badge ${status}`
    }
  }

  /**
   * Update bot status display
   */
  private updateBotStatusDisplay(state: IndicatorBotStatus['state'], message?: string): void {
    const healthIcon = document.querySelector('.health-icon')
    const statusMessage = document.getElementById('bot-status-message')

    if (healthIcon) {
      const icons = {
        active: '🟢',
        paused: '⏸️',
        error: '🔴',
        initializing: '🔄',
      }
      healthIcon.className = `health-icon ${state}`
      healthIcon.textContent = icons[state] || '🔄'
      const statusText = healthIcon.parentElement?.querySelector('div')?.querySelector('div')
      if (statusText) statusText.textContent = `Bot ${state}`
    }

    if (statusMessage) {
      statusMessage.textContent = message ?? ''
      statusMessage.style.display = message ? 'block' : 'none'
    }
  }

  /**
   * Update system health display
   */
  private updateSystemHealthDisplay(cpu?: number, memory?: number): void {
    // Update CPU meter
    if (cpu !== undefined) {
      const cpuMeter = document.getElementById('cpu-meter')
      const cpuFill = document.getElementById('cpu-fill')
      const cpuValue = document.getElementById('cpu-value')

      if (cpuMeter && cpuFill && cpuValue) {
        cpuMeter.style.display = 'flex'
        cpuFill.style.width = `${cpu}%`
        cpuFill.className = `meter-fill ${cpu < 50 ? 'low' : cpu < 80 ? 'medium' : 'high'}`
        cpuValue.textContent = `${cpu}%`
      }
    }

    // Update Memory meter
    if (memory !== undefined) {
      const memoryMeter = document.getElementById('memory-meter')
      const memoryFill = document.getElementById('memory-fill')
      const memoryValue = document.getElementById('memory-value')

      if (memoryMeter && memoryFill && memoryValue) {
        memoryMeter.style.display = 'flex'
        memoryFill.style.width = `${memory}%`
        memoryFill.className = `meter-fill ${memory < 50 ? 'low' : memory < 80 ? 'medium' : 'high'}`
        memoryValue.textContent = `${memory}%`
      }
    }
  }

  /**
   * Update sound toggle display
   */
  private updateSoundToggle(): void {
    const soundToggle = this.sidebarElements.soundToggle
    if (!soundToggle || !this.statusIndicators) return

    const enabled = this.statusIndicators.soundEnabled
    soundToggle.className = `sound-toggle ${enabled ? 'enabled' : 'disabled'}`
    soundToggle.innerHTML = `
      <span>${enabled ? '🔊' : '🔇'}</span>
      <span>Sounds ${enabled ? 'On' : 'Off'}</span>
    `
  }

  /**
   * Update alert count display
   */
  private updateAlertCount(count: number): void {
    const alertCount = document.getElementById('alert-count')
    if (alertCount) {
      alertCount.textContent = `History (${count})`
    }
  }

  /**
   * Update market status display
   */
  private updateMarketStatusDisplay(state: MarketStatus['state']): void {
    if (this.headerElements.marketStatus) {
      const dot = this.headerElements.marketStatus.querySelector('.market-dot')
      const text = this.headerElements.marketStatus.querySelector('span:last-child')

      if (dot) dot.className = `market-dot ${state}`
      if (text) text.textContent = `Market ${state}`
    }
  }

  /**
   * Update position status display
   */
  private updatePositionStatusDisplay(state: PositionStatus['state'], count?: number): void {
    if (this.headerElements.positionStatus) {
      const icons = {
        'in-position': '📊',
        'pending-entry': '⏳',
        'pending-exit': '🚪',
        flat: '💤',
      }

      this.headerElements.positionStatus.className = `position-badge ${state}`
      this.headerElements.positionStatus.innerHTML = `
        <span>${icons[state] || '💤'}</span>
        <span>${state.replace(/-/g, ' ')}</span>
        ${count ? `<span>(${count})</span>` : ''}
      `
    }
  }

  /**
   * Initialize TradingView chart with enhanced error handling
   */
  private async initializeChart(): Promise<void> {
    try {
      // Chart URL Construction: Use the configured API base URL
      // Chart makes calls to: /udf/config, /udf/symbols, /udf/history, /udf/marks
      // The getApiBaseUrl() method already handles environment variables and proxy routing
      let baseUrl = this.config.api_base_url

      // Ensure the base URL ends with /api for proper chart API calls
      if (!baseUrl.endsWith('/api')) {
        baseUrl = `${baseUrl}/api`
      }

      // Check network connectivity before initialization
      if (!navigator.onLine) {
        throw new Error('No network connection - chart requires internet access')
      }

      this.chart = new TradingViewChart(this.config.chart_config, baseUrl)
      const success = await this.chart.initialize()

      if (!success) {
        throw new Error('TradingView chart initialization returned false')
      }

      this.ui.log('info', 'TradingView chart initialized successfully', 'Chart')
      this.hideChartError()
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown chart error'
      console.error('Chart initialization failed:', errorMessage)

      // Show user-friendly error message
      if (errorMessage.includes('timeout')) {
        this.ui.log('error', 'Chart loading timed out - check internet connection', 'Chart')
        this.showChartError(
          'Chart loading timed out. Please check your internet connection and try again.'
        )
      } else if (errorMessage.includes('network') || errorMessage.includes('internet')) {
        this.ui.log('error', 'Network issue preventing chart load', 'Chart')
        this.showChartError('Network connection required for chart. Please check your connection.')
      } else {
        this.ui.log('error', `Chart initialization failed: ${errorMessage}`, 'Chart')
        this.showChartError(
          'Failed to load trading chart. Dashboard will continue with limited functionality.'
        )
      }

      // Continue without chart - the rest of the dashboard should still work
      this.chart = null
    }
  }

  /**
   * Initialize TradingView chart in non-blocking mode with aggressive timeout
   */
  private async initializeChartNonBlocking(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Set aggressive timeout for chart initialization (max 15 seconds)
      const chartTimeout = setTimeout(() => {
        console.warn('Chart initialization timeout - continuing without chart')
        this.chart = null
        this.showChartError('Chart loading timed out. Dashboard continues with full functionality.')
        reject(new Error('Chart initialization timeout'))
      }, 5000) // 5 second timeout for faster startup

      // Start chart initialization
      this.initializeChart()
        .then(() => {
          clearTimeout(chartTimeout)
          resolve()
        })
        .catch((error) => {
          clearTimeout(chartTimeout)
          // Don't propagate chart errors as fatal - dashboard should continue
          console.warn('Chart initialization failed, dashboard continues without chart:', error)
          this.chart = null
          this.showChartError(
            'Chart failed to load. All other dashboard features remain available.'
          )
          reject(error)
        })
    })
  }

  /**
   * Show chart error message to user
   */
  private showChartError(message: string): void {
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement
    if (chartError) {
      chartError.style.display = 'block'
      chartError.setAttribute('data-chart-error', 'visible')

      const errorText = chartError.querySelector('p')
      if (errorText) {
        errorText.textContent = message
      }
    }
  }

  /**
   * Hide chart error message
   */
  private hideChartError(): void {
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement
    if (chartError) {
      chartError.style.display = 'none'
      chartError.setAttribute('data-chart-error', 'hidden')
    }
  }

  /**
   * Set up UI event handlers
   */
  private setupUIEventHandlers(): void {
    // Handle symbol changes
    this.ui.onSymbolChanged((symbol: string) => {
      this.config.chart_config.symbol = symbol
      this.chart?.changeSymbol(symbol)
      this.ui.log('info', `Changed symbol to ${symbol}`, 'UI')
    })

    // Handle interval changes
    this.ui.onIntervalChanged((interval: string) => {
      this.config.chart_config.interval = interval
      this.chart?.changeInterval(interval)
      this.ui.log('info', `Changed interval to ${interval}`, 'UI')
    })

    // Handle chart fullscreen toggle
    this.ui.onChartFullscreenToggle(() => {
      this.chart?.toggleFullscreen()
    })

    // Handle chart retry
    this.ui.onChartRetryRequested(() => {
      void this.retryChartInitialization()
    })

    // Handle error retry
    this.ui.onErrorRetryRequested(() => {
      void this.handleErrorRetry()
    })
  }

  /**
   * Set up WebSocket event handlers
   */
  private setupWebSocketHandlers(): void {
    // Connection status changes
    this.websocket.onConnectionStatusChange((status) => {
      this.ui.updateConnectionStatus(status)

      // Update status indicators
      if (this.statusIndicators) {
        const wsStatus: ConnectionStatus['websocket'] =
          status === 'connected'
            ? 'connected'
            : status === 'connecting'
              ? 'reconnecting'
              : 'disconnected'

        this.statusIndicators.connectionStatus = {
          ...this.statusIndicators.connectionStatus,
          websocket: wsStatus,
          lastHeartbeat: new Date(),
        }

        // Update visual display
        this.updateConnectionDisplay(wsStatus)
      }

      if (status === 'connected') {
        this.ui.log('info', 'Connected to trading bot', 'WebSocket')
        this.statusIndicators?.addAlert({
          type: 'success',
          title: 'Connected',
          message: 'Successfully connected to trading bot',
          sound: true,
        })
      } else if (status === 'disconnected') {
        this.ui.log('warn', 'Disconnected from trading bot', 'WebSocket')
        this.statusIndicators?.addAlert({
          type: 'warning',
          title: 'Disconnected',
          message: 'Lost connection to trading bot',
          sound: true,
        })
      } else if (status === 'error') {
        this.ui.log('error', 'WebSocket connection error', 'WebSocket')
        this.statusIndicators?.addAlert({
          type: 'error',
          title: 'Connection Error',
          message: 'Failed to connect to trading bot',
          sound: true,
        })
      }
    })

    // Subscribe to all message types with a generic handler
    this.websocket.on('*', (message) => {
      this.handleWebSocketMessage(message)
    })

    // Subscribe to specific message types
    this.websocket.on('trading_loop', (message) => {
      const msg = message as any
      if (msg.data) {
        this.ui.log('info', `Trading signal: ${msg.data.action} at $${msg.data.price}`, 'Trading')

        // Add alert for new trading signals
        if (msg.data.action !== 'HOLD') {
          this.statusIndicators?.addAlert({
            type: 'info',
            title: 'Trading Signal',
            message: `${msg.data.action} signal at $${msg.data.price}`,
            sound: true,
          })
        }
      }
    })

    this.websocket.on('ai_decision', (message) => {
      const msg = message as any
      if (msg.data) {
        this.ui.log('info', `AI Decision: ${msg.data.action} - ${msg.data.reasoning}`, 'AI')

        // Alert for AI decisions
        if (msg.data.action !== 'HOLD') {
          this.statusIndicators?.addAlert({
            type: 'success',
            title: 'AI Decision',
            message: `${msg.data.action} - ${msg.data.reasoning}`,
            sound: false,
          })
        }

        // Update LLM Decision Card with AI decision
        if (msg.data.action && msg.data.reasoning !== undefined) {
          const tradeAction: TradeAction = {
            action: msg.data.action,
            confidence: msg.data.confidence || 0.5,
            reasoning: msg.data.reasoning,
            timestamp: msg.data.timestamp || new Date().toISOString(),
            price: msg.data.price,
            quantity: msg.data.quantity,
            leverage: msg.data.leverage,
          }
          this.updateLLMDecisionCard(tradeAction)
        }
      }
    })

    this.websocket.on('system_status', (message) => {
      const msg = message as any
      if (msg.data) {
        const level = msg.data.health ? 'info' : 'warn'
        this.ui.log(level, `System status: ${msg.data.status}`, 'System')

        // Update bot status indicator
        if (this.statusIndicators) {
          this.statusIndicators.botStatus = {
            state: msg.data.health ? 'active' : 'error',
            message: msg.data.status,
          }
        }
      }
    })

    this.websocket.on('error', (message) => {
      const msg = message as any
      if (msg.data) {
        this.ui.log('error', msg.data.message, 'System')

        // Error alert
        this.statusIndicators?.addAlert({
          type: 'error',
          title: 'System Error',
          message: msg.data.message,
          sound: true,
        })
      }
    })

    // Error handling
    this.websocket.onError((error: Event | Error) => {
      console.error('WebSocket error:', error)
      const errorMessage = error instanceof Error ? error.message : 'WebSocket connection error'
      this.ui.showError(errorMessage)
    })
  }

  /**
   * Handle incoming WebSocket messages with error boundaries
   */
  private handleWebSocketMessage(message: AllWebSocketMessages): void {
    try {
      // Validate message structure
      if (!message?.type) {
        console.warn('Invalid WebSocket message structure:', message)
        return
      }

      // Handle messages that don't have data property (like ping/pong)
      if (!('data' in message) && message.type !== 'ping' && message.type !== 'pong') {
        console.warn('WebSocket message missing data property:', message)
        return
      }

      switch (message.type) {
        case 'bot_status':
          if ('data' in message && message.data) {
            const botStatus = message.data as BotStatus
            this.ui.updateBotStatus(botStatus)

            // Update bot status indicator
            if (this.statusIndicators) {
              const indicatorState: IndicatorBotStatus['state'] = botStatus.is_active
                ? 'active'
                : botStatus.is_paused
                  ? 'paused'
                  : 'error'

              this.statusIndicators.botStatus = {
                state: indicatorState,
                message: botStatus.status_message,
              }

              // Update visual display
              this.updateBotStatusDisplay(indicatorState, botStatus.status_message)
            }
          }
          break

        case 'market_data':
          if ('data' in message && message.data) {
            const marketData = message.data as MarketData
            this.lastMarketData = marketData
            // Use debounced method to prevent excessive updates
            this.debouncedUpdateMarketData(marketData)

            // Update market status based on time (simplified)
            if (this.statusIndicators) {
              const hour = new Date().getHours()
              const day = new Date().getDay()
              const isWeekend = day === 0 || day === 6

              let marketState: MarketStatus['state'] = 'open'
              if (isWeekend) {
                marketState = 'closed'
              } else if (hour < 9 || hour >= 16) {
                marketState = hour < 9 ? 'pre-market' : 'after-hours'
              }

              this.statusIndicators.marketStatus = { state: marketState }

              // Update visual display
              this.updateMarketStatusDisplay(marketState)
            }
          }
          break

        case 'trade_action':
          if ('data' in message && message.data) {
            const tradeAction = message.data as TradeAction
            this.ui.updateLatestAction(tradeAction)
            // Add AI decision marker to chart if available
            if (this.chart && 'addAIDecisionMarker' in this.chart) {
              this.chart.addAIDecisionMarker(tradeAction)
            }
            // Update LLM Decision Card
            this.updateLLMDecisionCard(tradeAction)

            // Alert for executed trades
            if (tradeAction.action !== 'HOLD' && tradeAction.executed) {
              this.statusIndicators?.addAlert({
                type: 'success',
                title: 'Trade Executed',
                message: `${tradeAction.action} ${tradeAction.quantity} @ $${tradeAction.price}`,
                sound: true,
              })
            }
          }
          break

        case 'indicators':
          if ('data' in message && message.data) {
            const indicators = message.data as VuManchuIndicators
            this.lastIndicators = indicators
            // Use debounced method to prevent excessive chart updates
            this.debouncedUpdateIndicators(indicators)
            this.throttledLogUpdate({ type: 'indicators', message: 'VuManChu indicators updated' })
          }
          break

        case 'position':
          if ('data' in message && message.data) {
            // Handle single position update - convert to array for UI
            const position = message.data as Position
            this.ui.updatePositions([position])

            // Update position status indicator
            if (this.statusIndicators) {
              let positionState: PositionStatus['state'] = 'flat'
              const positionSize = position.quantity ?? position.size ?? 0
              if (position.side && positionSize > 0) {
                positionState = 'in-position'
              }

              this.statusIndicators.positionStatus = {
                state: positionState,
                count: positionSize > 0 ? 1 : 0,
              }

              // Update visual display
              this.updatePositionStatusDisplay(positionState, positionSize > 0 ? 1 : 0)

              // Alert for significant P&L changes
              if (position.unrealized_pnl !== undefined) {
                const averagePrice = position.average_price ?? position.entry_price ?? 0
                const quantity = position.quantity ?? position.size ?? 0
                if (averagePrice > 0 && quantity > 0) {
                  const pnlPercent = Math.abs(
                    (position.unrealized_pnl / (averagePrice * quantity)) * 100
                  )
                  if (pnlPercent > 5) {
                    this.statusIndicators.addAlert({
                      type: position.unrealized_pnl > 0 ? 'success' : 'warning',
                      title: 'Significant P&L Change',
                      message: `${position.unrealized_pnl > 0 ? 'Profit' : 'Loss'}: $${position.unrealized_pnl.toFixed(2)} (${pnlPercent.toFixed(1)}%)`,
                      sound: pnlPercent > 10,
                    })
                  }
                }
              }
            }
          }
          break

        case 'risk_metrics':
          if ('data' in message && message.data) {
            this.ui.updateRiskMetrics(message.data as RiskMetrics)
          }
          break

        case 'trading_loop':
        case 'ai_decision':
        case 'system_status':
        case 'error':
          // These are handled by specific handlers
          break

        case 'ping':
        case 'pong':
          // Ping/pong messages are handled internally
          // Update latency on pong
          if (
            message.type === 'pong' &&
            'timestamp' in message &&
            typeof message.timestamp === 'number'
          ) {
            const latency = Date.now() - message.timestamp
            this.updateLatency(latency)
          }
          break

        default:
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error)
      // Don't log UI errors to prevent infinite loops
    }
  }

  /**
   * Get WebSocket URL from environment or default
   *
   * Simplified URL Construction Logic:
   * 1. Use VITE_WS_URL environment variable if available
   * 2. Fallback to relative '/ws' path for nginx proxy compatibility
   * 3. Automatically handle protocol (ws/wss) based on current page protocol
   */
  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host

    // Check for environment variable first
    const envWsUrl = import.meta.env.VITE_WS_URL ?? (window as any).__WS_URL__
    if (envWsUrl) {
      // Handle absolute URLs (already include protocol and host)
      if (envWsUrl.startsWith('ws://') || envWsUrl.startsWith('wss://')) {
        return envWsUrl
      }

      // Handle protocol-relative URLs
      if (envWsUrl.startsWith('//')) {
        return `${protocol}${envWsUrl}`
      }

      // Handle relative paths from environment variables
      if (envWsUrl.startsWith('/')) {
        return `${protocol}//${host}${envWsUrl}`
      }

      // Fallback: treat as relative path
      return `${protocol}//${host}/${envWsUrl.replace(/^\/+/, '')}`
    }

    // Default fallback - use relative '/ws' path for nginx proxy compatibility
    return `${protocol}//${host}/ws`
  }

  /**
   * Get API base URL from environment or default
   *
   * Simplified URL Construction Logic:
   * 1. Use VITE_API_URL or VITE_API_BASE_URL environment variable if available
   * 2. Fallback to relative '/api' path for nginx proxy compatibility
   * 3. No hardcoded host detection - let environment variables or proxy handle routing
   */
  private getApiBaseUrl(): string {
    const protocol = window.location.protocol
    const host = window.location.host

    // Check for environment variable first
    const envApiUrl =
      import.meta.env.VITE_API_URL ||
      import.meta.env.VITE_API_BASE_URL ||
      (window as any).__API_URL__
    if (envApiUrl) {
      // Handle absolute URLs (already include protocol and host)
      if (envApiUrl.startsWith('http://') || envApiUrl.startsWith('https://')) {
        return envApiUrl
      }

      // Handle protocol-relative URLs
      if (envApiUrl.startsWith('//')) {
        return `${protocol}${envApiUrl}`
      }

      // Handle relative paths from environment variables
      if (envApiUrl.startsWith('/')) {
        return `${protocol}//${host}${envApiUrl}`
      }

      // Fallback: treat as relative path
      return `${protocol}//${host}/${envApiUrl.replace(/^\/+/, '')}`
    }

    // Default fallback - use relative '/api' path for nginx proxy compatibility
    return `${protocol}//${host}/api`
  }

  /**
   * Update loading progress with message and percentage
   */
  private updateLoadingProgress(message: string, percentage: number): void {
    const loadingEl = document.getElementById('loading')
    if (!loadingEl) return

    const messageEl = loadingEl.querySelector('.loading-message')
    const barEl = loadingEl.querySelector('.progress-bar')
    const percentageEl = loadingEl.querySelector('.loading-percentage')

    if (messageEl) {
      messageEl.textContent = message
    }

    if (barEl) {
      (barEl as HTMLElement).style.width = `${percentage}%`
    }

    if (percentageEl) {
      percentageEl.textContent = `${percentage}%`
    }
  }

  /**
   * Show initialization error
   */
  private showInitializationError(errorMessage: string): void {
    const loadingEl = document.getElementById('loading')
    if (!loadingEl) return

    loadingEl.innerHTML = `
      <div class="error-message">
        <div class="error-icon">⚠️</div>
        <h2>Failed to Load Dashboard</h2>
        <p class="error-details">${errorMessage}</p>
        <div class="error-actions">
          <button onclick="window.location.reload()" class="retry-button">
            🔄 Retry
          </button>
          <button onclick="this.showErrorDetails()" class="details-button">
            📋 Details
          </button>
        </div>
      </div>
    `
  }

  /**
   * Hide the loading screen
   */
  private hideLoadingScreen(): void {
    const loadingEl = document.getElementById('loading')
    const dashboardEl = document.getElementById('dashboard')

    if (loadingEl) {
      loadingEl.style.opacity = '0'
      loadingEl.style.transition = 'opacity 0.3s ease'
      setTimeout(() => {
        loadingEl.style.display = 'none'
      }, 300)
    }

    if (dashboardEl) {
      dashboardEl.style.opacity = '1'
      dashboardEl.style.visibility = 'visible'
    }
  }

  /**
   * Setup page visibility handling for connection management
   */
  private setupVisibilityHandling(): void {
    this.visibilityHandler.onVisibilityChange((visible: boolean) => {
      if (visible) {
        // Page became visible - restore normal operations
        if (this.websocket && this.isInitialized) {
          this.websocket.connect()
          this.ui.log('info', 'Page became visible - checking connection', 'Visibility')
        }

        // Restore performance optimization
        this.optimizePerformance(true)

        if (this.ui) {
          this.ui.log('info', 'Page became visible - full functionality restored', 'Visibility')
        }
      } else {
        // Page hidden - reduce activity
        // Only apply performance optimization if we're actually initialized
        if (this.isInitialized) {
          this.optimizePerformance(false)

          if (this.ui) {
            this.ui.log('debug', 'Page hidden - reducing background activity', 'Visibility')
          }
        }
      }
    })
  }

  /**
   * Retry chart initialization with enhanced recovery
   */
  private async retryChartInitialization(): Promise<void> {
    try {
      this.ui.log('info', 'Retrying chart initialization...', 'Chart')

      // Show loading state
      const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement
      if (chartLoading) {
        chartLoading.style.display = 'flex'
        chartLoading.setAttribute('data-chart-loading', 'true')
      }

      this.hideChartError()

      // Clean up existing chart
      if (this.chart) {
        this.chart.destroy()
        this.chart = null
      }

      // Check network connectivity
      if (!navigator.onLine) {
        throw new Error('No network connection available')
      }

      // Wait a moment before retrying
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Retry initialization
      await this.initializeChart()

      // Hide loading state on success
      if (chartLoading) {
        chartLoading.style.display = 'none'
        chartLoading.setAttribute('data-chart-loading', 'false')
      }

      this.ui.log('info', 'Chart retry successful', 'Chart')
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown retry error'
      console.error('Chart retry failed:', errorMessage)
      this.ui.log('error', `Chart retry failed: ${errorMessage}`, 'Chart')

      // Hide loading state on failure
      const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement
      if (chartLoading) {
        chartLoading.style.display = 'none'
        chartLoading.setAttribute('data-chart-loading', 'false')
      }

      // Show specific error message
      if (errorMessage.includes('network') || errorMessage.includes('connection')) {
        this.showChartError('No internet connection. Please check your network and try again.')
      } else {
        this.showChartError('Failed to load chart after retry. Please refresh the page.')
      }
    }
  }

  /**
   * Handle general error retry
   */
  private async handleErrorRetry(): Promise<void> {
    try {
      this.ui.log('info', 'Attempting error recovery...', 'System')

      // Clear any existing errors
      this.ui.clearError()

      // Try to reconnect WebSocket if disconnected
      if (this.websocket && !this.websocket.isConnected()) {
        this.ui.log('info', 'Reconnecting WebSocket...', 'WebSocket')
        this.websocket.connect()
      }

      // Retry chart if it failed
      if (!this.chart) {
        await this.retryChartInitialization()
      }

      this.ui.log('info', 'Error recovery completed', 'System')
    } catch (error) {
      console.error('Error recovery failed:', error)
      this.ui.log('error', 'Error recovery failed', 'System')
    }
  }

  /**
   * Graceful degradation when components fail
   */
  public handleGracefulDegradation(component: string, error: Error): void {
    console.warn(`Component ${component} failed, continuing with degraded functionality:`, error)

    const degradationMap: Record<string, string> = {
      chart: 'Trading chart unavailable - market data and controls still functional',
      websocket: 'Real-time updates unavailable - dashboard in read-only mode',
      ui: 'Some UI features may be limited',
    }

    const message = degradationMap[component] || `${component} functionality limited`
    this.ui.log('warn', message, 'System')

    // Store degradation state for recovery attempts
    ;(this as any)[`${component}Failed`] = true
  }

  /**
   * Performance optimization based on page visibility
   */
  private optimizePerformance(visible: boolean): void {
    if (!visible) {
      // Reduce update frequency when page is hidden
      this.config.refresh_interval = Math.max(this.config.refresh_interval * 2, 5000)

      // Pause non-critical chart updates (if method exists)
      if (this.chart && 'pauseUpdates' in this.chart) {
        (this.chart as any).pauseUpdates()
      }
    } else {
      // Restore normal update frequency
      this.config.refresh_interval = 1000

      // Resume chart updates (if method exists)
      if (this.chart && 'resumeUpdates' in this.chart) {
        (this.chart as any).resumeUpdates()
      }
    }
  }

  /**
   * Update LLM Decision Card with new trade action
   */
  private updateLLMDecisionCard(tradeAction: TradeAction): void {
    if (!this.llmDecisionCard) return

    try {
      // Determine risk level based on confidence
      let riskLevel: 'low' | 'medium' | 'high' = 'medium'
      if (tradeAction.confidence >= 0.8) {
        riskLevel = 'low'
      } else if (tradeAction.confidence <= 0.4) {
        riskLevel = 'high'
      }

      // Create decision data
      const decisionData: LLMDecisionData = {
        action: tradeAction,
        marketData: this.lastMarketData || {
          symbol: this.config.default_symbol,
          price: tradeAction.price || 0,
          timestamp: new Date().toISOString(),
        },
        indicators: this.lastIndicators || undefined,
        riskLevel,
        positionSize: tradeAction.quantity,
      }

      // Update the card
      this.llmDecisionCard.updateDecision(decisionData)

      this.ui.log('debug', 'LLM Decision Card updated', 'UI')
    } catch (error) {
      console.error('Failed to update LLM Decision Card:', error)
      this.ui.log('error', 'Failed to update AI decision display', 'UI')
    }
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    console.log('🧹 Cleaning up dashboard resources...')

    try {
      // Disconnect WebSocket
      if (this.websocket) {
        this.websocket.disconnect()
      }

      // Destroy chart
      if (this.chart) {
        this.chart.destroy()
        this.chart = null
      }

      // Destroy LLM Monitor Dashboard
      if (this.llmMonitorDashboard) {
        this.llmMonitorDashboard.destroy()
        this.llmMonitorDashboard = null
      }

      // Clear LLM Decision Card (it doesn't have destroy method)
      if (this.llmDecisionCard) {
        this.llmDecisionCard.clear()
        this.llmDecisionCard = null
      }

      // Destroy Phase 4 Enterprise Services
      if (this.testManager) {
        void this.testManager.destroy()
        this.testManager = null
      }

      if (this.dashboardOrchestrator) {
        this.dashboardOrchestrator.destroy()
        this.dashboardOrchestrator = null
      }

      if (this.mobileOptimizer) {
        this.mobileOptimizer.destroy()
        this.mobileOptimizer = null
      }

      if (this.notificationSystem) {
        this.notificationSystem.destroy()
        this.notificationSystem = null
      }

      if (this.enterpriseWebSocketManager) {
        this.enterpriseWebSocketManager.disconnect()
        this.enterpriseWebSocketManager = null
      }

      if (this.dataManager) {
        void this.dataManager.destroy()
        this.dataManager = null
      }

      if (this.performanceOptimizer) {
        this.performanceOptimizer.destroy()
        this.performanceOptimizer = null
      }

      if (this.securityManager) {
        this.securityManager.destroy()
        this.securityManager = null
      }

      if (this.errorHandler) {
        this.errorHandler.destroy()
        this.errorHandler = null
      }

      // Clear all utilities
      this.visibilityHandler.destroy()
      this.performanceMonitor.destroy()
      this.debouncer.destroy()
      this.memoryManager.destroy()
      this.ui.destroy()

      // Clear system health monitoring
      if (this.systemHealthInterval) {
        clearInterval(this.systemHealthInterval)
        this.systemHealthInterval = null
      }

      // Final performance log
      if (__DEV__) {
        this.performanceMonitor.logMetrics()
      }

      // Clear references to prevent memory leaks
      this.lastMarketData = null
      this.lastIndicators = null
      this.statusIndicators = null
      this.headerElements = {}
      this.sidebarElements = {}

      // Reset state
      this.isInitialized = false
      this.initializationPromise = null

      this.ui.log('info', 'Dashboard destroyed cleanly', 'App')
    } catch (error) {
      console.error('Error during cleanup:', error)
    }
  }

  /**
   * Get current application health status
   */
  public getHealthStatus(): Record<string, any> {
    return {
      initialized: this.isInitialized,
      websocket_connected: this.websocket?.isConnected() || false,
      chart_loaded: this.chart !== null,
      page_visible: this.visibilityHandler.visible,
      performance_metrics: Object.fromEntries(this.performanceMonitor['metrics'] || []),
    }
  }

  /**
   * Force reconnection of all services
   */
  public async forceReconnect(): Promise<void> {
    this.ui.log('info', 'Forcing full reconnection...', 'System')

    try {
      // Reconnect WebSocket
      if (this.websocket) {
        this.websocket.disconnect()
        await new Promise((resolve) => setTimeout(resolve, 1000))
        this.websocket.connect()
      }

      // Retry chart initialization if it failed
      if (!this.chart) {
        await this.retryChartInitialization()
      }

      this.ui.log('info', 'Full reconnection completed', 'System')
    } catch (error) {
      console.error('Force reconnection failed:', error)
      this.ui.log('error', 'Force reconnection failed', 'System')
    }
  }
}

// Global application instance for debugging and external access
let globalDashboardApp: DashboardApp | null = null

// Enhanced global error handling
window.addEventListener('error', (event) => {
  console.error('🚨 Global JavaScript error:', {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    error: event.error,
  })

  // Try to log to dashboard if available
  if (globalDashboardApp?.ui) {
    globalDashboardApp.ui.log('error', `Global error: ${event.message}`, 'System')
  }
})

window.addEventListener('unhandledrejection', (event) => {
  console.error('🚨 Unhandled Promise rejection:', event.reason)

  // Try to log to dashboard if available
  if (globalDashboardApp?.ui) {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason)
    globalDashboardApp.ui.log('error', `Unhandled Promise: ${reason}`, 'System')
  }

  // Prevent default behavior that logs to console
  event.preventDefault()
})

// Network connectivity monitoring
window.addEventListener('online', () => {
  if (globalDashboardApp?.ui) {
    globalDashboardApp.ui.log('info', 'Network connection restored', 'Network')
    // Attempt to reconnect services
    globalDashboardApp.forceReconnect().catch(console.error)
  }
})

window.addEventListener('offline', () => {
  if (globalDashboardApp?.ui) {
    globalDashboardApp.ui.log(
      'warn',
      'Network connection lost - dashboard in offline mode',
      'Network'
    )
  }
})

// Performance monitoring
if ('PerformanceObserver' in window) {
  try {
    const perfObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries()
      entries.forEach((entry) => {
        if (entry.entryType === 'navigation') {
        }
      })
    })

    perfObserver.observe({ entryTypes: ['navigation', 'measure'] })
  } catch (error) {
    console.warn('Performance monitoring not available:', error)
  }
}

/**
 * Service Worker cleanup utility - simplified for reliability
 */
class ServiceWorkerCleaner {
  async performCleanup(): Promise<void> {
    if (!('serviceWorker' in navigator)) {
      return
    }

    try {
      const registrations = await navigator.serviceWorker.getRegistrations()

      if (registrations.length > 0) {
        await Promise.all(
          registrations.map((registration) => registration.unregister().catch(() => {}))
        )
      }
    } catch (error) {
      // Silently fail - not critical for dashboard operation
    }
  }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  // Quick service worker cleanup - non-blocking
  const swCleaner = new ServiceWorkerCleaner()
  swCleaner.performCleanup().catch(() => {}) // Don't wait for cleanup

  // Track page load performance
  const pageLoadStart = performance.now()

  try {
    globalDashboardApp = new DashboardApp()

    // Show loading screen immediately
    const loadingEl = document.getElementById('loading')
    if (loadingEl) {
      loadingEl.style.display = 'flex'
      loadingEl.innerHTML = `
        <div class="loading-container">
          <div class="loading-spinner"></div>
          <div class="loading-text">Loading AI Trading Dashboard...</div>
          <div class="loading-progress">
            <div class="loading-message">Initializing...</div>
            <div class="progress-container">
              <div class="progress-bar" style="width: 0%"></div>
            </div>
            <div class="loading-percentage">0%</div>
          </div>
        </div>
      `
    }

    // Initialize with timeout protection (reduced from 30s to 20s)
    const initTimeout = setTimeout(() => {
      console.error('⏰ Dashboard initialization timeout')
      if (loadingEl) {
        loadingEl.innerHTML = `
          <div class="error-message">
            <div class="error-icon">⏰</div>
            <h2>Initialization Timeout</h2>
            <p class="error-details">Dashboard core components took too long to load. This may be due to network issues or server problems.</p>
            <div class="error-actions">
              <button onclick="window.location.reload()" class="retry-button">
                🔄 Retry
              </button>
              <button onclick="window.dashboard?.app()?.forceReconnect()" class="reconnect-button">
                🔌 Reconnect
              </button>
            </div>
          </div>
        `
      }
    }, 8000) // 8 second timeout for initialization

    await globalDashboardApp.initialize()
    clearTimeout(initTimeout)

    const _pageLoadTime = performance.now() - pageLoadStart
  } catch (error) {
    console.error('❌ Failed to start dashboard:', error)

    // Show comprehensive error information
    const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error'
    const loadingEl = document.getElementById('loading')

    if (loadingEl) {
      loadingEl.innerHTML = `
        <div class="error-message">
          <div class="error-icon">💥</div>
          <h2>Dashboard Initialization Failed</h2>
          <p class="error-details">${errorMessage}</p>
          <div class="error-info">
            <p><strong>Time:</strong> ${new Date().toLocaleString()}</p>
            <p><strong>User Agent:</strong> ${navigator.userAgent}</p>
            <p><strong>URL:</strong> ${window.location.href}</p>
          </div>
          <div class="error-actions">
            <button onclick="window.location.reload()" class="retry-button">
              🔄 Retry
            </button>
            <button onclick="navigator.clipboard?.writeText(document.querySelector('.error-info').textContent || '')" class="copy-button">
              📋 Copy Info
            </button>
          </div>
        </div>
      `
    }
  }
})

// Cleanup and lifecycle management
window.addEventListener('beforeunload', () => {
  if (globalDashboardApp) {
    globalDashboardApp.destroy()
  }
})

// Detect page refresh vs close
window.addEventListener('pagehide', (event) => {
  if (event.persisted) {
  } else {
  }
})

// Handle critical resource errors
window.addEventListener(
  'error',
  (event) => {
    // Check if it's a resource loading error
    if (event.target && event.target !== window) {
      const element = event.target as HTMLElement
      console.error(`📎 Resource failed to load:`, {
        tagName: element.tagName,
        src: (element as any).src || (element as any).href,
        message: event.message,
      })

      if (globalDashboardApp?.ui) {
        globalDashboardApp.ui.log(
          'error',
          `Resource failed to load: ${element.tagName}`,
          'Resources'
        )
      }
    }
  },
  true
)

// Export enhanced debugging interface
;(window as any).dashboard = {
  app: () => globalDashboardApp,
  health: () => globalDashboardApp?.getHealthStatus(),
  reconnect: () => globalDashboardApp?.forceReconnect(),
  performance: () => globalDashboardApp?.performanceMonitor?.logMetrics(),

  // Debugging utilities
  classes: { DashboardApp, DashboardUI, DashboardWebSocket, TradingViewChart },
  utils: { PerformanceMonitor, VisibilityHandler },
}

// Development helpers
if (window.location.hostname === 'localhost') {
  // Development mode indicators available in browser console

  // Add schema compliance testing to debug interface
  (window as any).dashboard.testSchemaCompliance = () => {
    const chart = globalDashboardApp?.chart
    if (chart && 'testSchemaCompliance' in chart) {
      return (chart as any).testSchemaCompliance()
    } else {
      console.warn('Chart not available or schema testing not supported')
      return { success: false, issues: ['Chart not initialized'], validations: {} }
    }
  }
}
