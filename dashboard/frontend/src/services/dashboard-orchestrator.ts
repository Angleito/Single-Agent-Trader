/**
 * Comprehensive Dashboard Integration and Orchestration System
 *
 * Provides centralized management and coordination of all dashboard components:
 * - Component lifecycle management and coordination
 * - Centralized state management and synchronization
 * - Event-driven communication between components
 * - Real-time data flow orchestration
 * - Performance monitoring and optimization
 * - Error handling and recovery coordination
 * - Layout and UI state management
 * - Plugin architecture for extensibility
 * - Configuration management
 * - Analytics and usage tracking
 */

import { WebSocketManager } from './websocket-manager'
import { NotificationSystem } from './notification-system'
import { ManualTradingInterface } from '../components/manual-trading'
import { PositionMonitor } from '../components/position-monitor'
import { RiskManagementDashboard } from '../components/risk-management-dashboard'
import { PerformanceAnalyticsDashboard } from '../components/performance-analytics-dashboard'
import type {
  DashboardState,
  BotStatus,
  MarketData,
  Position,
  RiskMetrics,
  ConnectionStatus,
  TradeAction,
  TradingModeConfig,
  TradingMode,
} from '../types'

export interface ComponentConfig {
  id: string
  type: string
  containerId: string
  enabled: boolean
  priority: number
  dependencies: string[]
  settings: Record<string, any>
  lifecycle: {
    initialized: boolean
    loaded: boolean
    active: boolean
    error?: string
  }
}

export interface DashboardConfig {
  apiBaseUrl: string
  websocketUrl: string
  apiKey: string
  theme: 'dark' | 'light' | 'auto'
  layout: 'grid' | 'tabs' | 'accordion' | 'sidebar'
  realTimeEnabled: boolean
  notificationsEnabled: boolean
  autoRefreshInterval: number
  maxRetries: number
  debugMode: boolean
  performance: {
    enableMetrics: boolean
    enableProfiling: boolean
    maxMemoryUsage: number
    maxCpuUsage: number
  }
  components: ComponentConfig[]
  features: {
    manualTrading: boolean
    positionMonitoring: boolean
    riskManagement: boolean
    performanceAnalytics: boolean
    marketAnalysis: boolean
    strategyTuning: boolean
    alertSystem: boolean
    reporting: boolean
  }
}

export interface PerformanceMetrics {
  memoryUsage: number
  cpuUsage: number
  renderTime: number
  updateFrequency: number
  errorRate: number
  latency: number
  throughput: number
  activeComponents: number
  totalEvents: number
  cacheHitRate: number
}

export interface DashboardAnalytics {
  sessionStart: Date
  sessionDuration: number
  pageViews: number
  componentInteractions: number
  errorsEncountered: number
  featuresUsed: string[]
  userPreferences: Record<string, any>
  performanceMetrics: PerformanceMetrics
}

export class DashboardOrchestrator {
  private config: DashboardConfig
  private state: DashboardState
  private components = new Map<string, any>()
  private websocketManager: WebSocketManager | null = null
  private notificationSystem: NotificationSystem | null = null
  private eventBus = new Map<string, Set<Function>>()
  private updateQueue: Array<{ component: string; data: any; timestamp: number }> = []
  private performanceMetrics: PerformanceMetrics
  private analytics: DashboardAnalytics
  private intervals: NodeJS.Timeout[] = []
  private errorRecovery = new Map<string, { count: number; lastError: Date }>()
  private dataCache = new Map<string, { data: any; timestamp: number; ttl: number }>()
  private stateHistory: DashboardState[] = []
  private maxStateHistory = 50

  // Performance monitoring
  private performanceObserver: PerformanceObserver | null = null
  private memoryMonitor: number | null = null
  private renderProfiler = new Map<string, number[]>()

  // Plugin system
  private plugins = new Map<string, any>()
  private pluginHooks = new Map<string, Function[]>()

  constructor(config: DashboardConfig) {
    this.config = config
    this.state = {
      bot_status: null,
      market_data: null,
      latest_action: null,
      indicators: null,
      positions: [],
      risk_metrics: null,
      connection_status: 'disconnected',
      error_message: null,
      trading_mode_config: undefined,
    }

    this.performanceMetrics = {
      memoryUsage: 0,
      cpuUsage: 0,
      renderTime: 0,
      updateFrequency: 0,
      errorRate: 0,
      latency: 0,
      throughput: 0,
      activeComponents: 0,
      totalEvents: 0,
      cacheHitRate: 0,
    }

    this.analytics = {
      sessionStart: new Date(),
      sessionDuration: 0,
      pageViews: 1,
      componentInteractions: 0,
      errorsEncountered: 0,
      featuresUsed: [],
      userPreferences: {},
      performanceMetrics: this.performanceMetrics,
    }

    this.setupPerformanceMonitoring()
    this.setupErrorHandling()
    this.init()
  }

  /**
   * Initialize the dashboard orchestrator
   */
  private async init(): Promise<void> {
    try {
      this.log('Initializing Dashboard Orchestrator')

      // Initialize core services
      await this.initializeWebSocket()
      await this.initializeNotifications()

      // Fetch trading mode configuration
      await this.fetchTradingModeConfig()

      // Initialize components
      await this.initializeComponents()

      // Setup data flow
      this.setupDataFlow()

      // Start monitoring
      this.startPerformanceMonitoring()
      this.startAnalyticsTracking()

      // Setup auto-refresh
      if (this.config.autoRefreshInterval > 0) {
        this.startAutoRefresh()
      }

      this.emit('orchestrator:initialized', { config: this.config })
      this.log('Dashboard Orchestrator initialized successfully')
    } catch (error) {
      this.handleError('initialization', error as Error)
      throw error
    }
  }

  /**
   * Fetch trading mode configuration
   */
  private async fetchTradingModeConfig(): Promise<void> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/api/trading-mode`)
      if (response.ok) {
        const config = await response.json()
        this.state.trading_mode_config = config
        this.emit('trading_mode:updated', config)
        this.log(
          `Trading mode configured: ${config.trading_mode} (futures: ${config.futures_enabled})`
        )
      } else {
        this.log('Failed to fetch trading mode configuration')
      }
    } catch (error) {
      this.log(`Error fetching trading mode config: ${error}`)
      // Set default config
      this.state.trading_mode_config = {
        trading_mode: 'spot' as TradingMode,
        futures_enabled: false,
        exchange_type: 'coinbase',
        account_type: 'CBI',
        leverage_available: false,
        features: {
          margin_trading: false,
          stop_loss: true,
          take_profit: true,
          contracts: false,
          liquidation_price: false,
        },
        supported_symbols: {
          spot: ['BTC-USD', 'ETH-USD'],
          futures: [],
        },
      }
    }
  }

  /**
   * Initialize WebSocket connection
   */
  private async initializeWebSocket(): Promise<void> {
    if (!this.config.realTimeEnabled) return

    try {
      this.websocketManager = new WebSocketManager({
        url: this.config.websocketUrl,
        reconnect: true,
        maxReconnectAttempts: this.config.maxRetries,
        debug: this.config.debugMode,
      })

      // Setup WebSocket event handlers
      this.websocketManager.addEventListener('connected', () => {
        this.updateConnectionStatus('connected')
        this.emit('websocket:connected')
      })

      this.websocketManager.addEventListener('disconnected', () => {
        this.updateConnectionStatus('disconnected')
        this.emit('websocket:disconnected')
      })

      this.websocketManager.addEventListener('error', (data) => {
        this.handleError('websocket', data.error)
        this.updateConnectionStatus('error')
      })

      this.websocketManager.addEventListener('message', (data) => {
        this.handleWebSocketMessage(data)
      })

      // Subscribe to data streams
      this.setupWebSocketSubscriptions()

      await this.websocketManager.connect()
      this.log('WebSocket initialized successfully')
    } catch (error) {
      this.log(`Failed to initialize WebSocket: ${error}`)
      throw error
    }
  }

  /**
   * Initialize notification system
   */
  private async initializeNotifications(): Promise<void> {
    if (!this.config.notificationsEnabled) return

    try {
      this.notificationSystem = new NotificationSystem(this.config.apiBaseUrl, this.config.apiKey)

      // Register default notification channels
      this.registerDefaultNotificationChannels()

      // Setup trading alert templates
      this.setupTradingAlertTemplates()

      this.log('Notification system initialized successfully')
    } catch (error) {
      this.log(`Failed to initialize notifications: ${error}`)
      throw error
    }
  }

  /**
   * Initialize dashboard components
   */
  private async initializeComponents(): Promise<void> {
    const enabledComponents = this.config.components.filter((c) => c.enabled)

    // Sort by priority (higher priority first)
    enabledComponents.sort((a, b) => b.priority - a.priority)

    for (const componentConfig of enabledComponents) {
      try {
        await this.initializeComponent(componentConfig)
      } catch (error) {
        this.log(`Failed to initialize component ${componentConfig.id}: ${error}`)
        componentConfig.lifecycle.error = (error as Error).message
        this.handleComponentError(componentConfig.id, error as Error)
      }
    }

    this.performanceMetrics.activeComponents = this.components.size
    this.log(`Initialized ${this.components.size} components successfully`)
  }

  /**
   * Initialize individual component
   */
  private async initializeComponent(config: ComponentConfig): Promise<void> {
    this.log(`Initializing component: ${config.id} (${config.type})`)

    // Check dependencies
    for (const depId of config.dependencies) {
      if (!this.components.has(depId)) {
        throw new Error(`Dependency not met: ${depId} required for ${config.id}`)
      }
    }

    let component: any

    try {
      switch (config.type) {
        case 'manual-trading':
          if (this.config.features.manualTrading) {
            component = new ManualTradingInterface(config.containerId, this.config.apiBaseUrl)
          }
          break

        case 'position-monitor':
          if (this.config.features.positionMonitoring) {
            component = new PositionMonitor(config.containerId)
          }
          break

        case 'risk-management':
          if (this.config.features.riskManagement) {
            component = new RiskManagementDashboard(config.containerId, this.config.apiBaseUrl)
          }
          break

        case 'performance-analytics':
          if (this.config.features.performanceAnalytics) {
            component = new PerformanceAnalyticsDashboard(
              config.containerId,
              this.config.apiBaseUrl
            )
          }
          break

        default:
          // Try to load from plugins
          component = await this.loadPluginComponent(config)
      }

      if (component) {
        this.components.set(config.id, component)
        config.lifecycle.initialized = true
        config.lifecycle.loaded = true
        config.lifecycle.active = true

        // Setup component event handlers
        this.setupComponentEventHandlers(config.id, component)

        this.emit('component:initialized', { componentId: config.id, config })
      }
    } catch (error) {
      config.lifecycle.error = (error as Error).message
      throw error
    }
  }

  /**
   * Setup data flow between components and services
   */
  private setupDataFlow(): void {
    // Subscribe to state changes
    this.addEventListener('state:updated', (data) => {
      this.propagateStateUpdate(data)
    })

    // Setup component data updates
    this.addEventListener('data:market', (data) => {
      this.updateComponents('market_data', data)
    })

    this.addEventListener('data:position', (data) => {
      this.updateComponents('positions', data)
    })

    this.addEventListener('data:risk', (data) => {
      this.updateComponents('risk_metrics', data)
    })

    this.addEventListener('data:bot_status', (data) => {
      this.updateComponents('bot_status', data)
    })

    // Setup trading mode updates
    this.addEventListener('trading_mode:updated', (config) => {
      this.updateComponentsTradingMode(config)
    })

    // Setup cross-component communication
    this.setupCrossComponentCommunication()
  }

  /**
   * Setup WebSocket subscriptions
   */
  private setupWebSocketSubscriptions(): void {
    if (!this.websocketManager) return

    // Subscribe to market data
    this.websocketManager.subscribe('market_data', (data) => {
      this.updateMarketData(data)
    })

    // Subscribe to position updates
    this.websocketManager.subscribe('positions', (data) => {
      this.updatePositions(data)
    })

    // Subscribe to risk metrics
    this.websocketManager.subscribe('risk_metrics', (data) => {
      this.updateRiskMetrics(data)
    })

    // Subscribe to bot status
    this.websocketManager.subscribe('bot_status', (data) => {
      this.updateBotStatus(data)
    })

    // Subscribe to trade executions
    this.websocketManager.subscribe('trade_executed', (data) => {
      this.handleTradeExecution(data)
    })

    // Subscribe to alerts
    this.websocketManager.subscribe('alerts', (data) => {
      this.handleAlert(data)
    })
  }

  /**
   * Handle WebSocket messages
   */
  private handleWebSocketMessage(data: any): void {
    try {
      this.performanceMetrics.totalEvents++

      // Add to update queue for batching
      this.updateQueue.push({
        component: 'websocket',
        data,
        timestamp: Date.now(),
      })

      // Process queue if it's getting large
      if (this.updateQueue.length > 100) {
        this.processUpdateQueue()
      }
    } catch (error) {
      this.handleError('websocket_message', error as Error)
    }
  }

  /**
   * Update market data
   */
  private updateMarketData(data: MarketData): void {
    this.state.market_data = data
    this.addToStateHistory()
    this.emit('data:market', data)

    // Update components
    const positionMonitor = this.components.get('position-monitor')
    if (positionMonitor?.updateMarketData) {
      positionMonitor.updateMarketData(data)
    }

    const manualTrading = this.components.get('manual-trading')
    if (manualTrading?.updateMarketData) {
      manualTrading.updateMarketData(data)
    }
  }

  /**
   * Update positions
   */
  private updatePositions(data: Position[]): void {
    this.state.positions = data
    this.addToStateHistory()
    this.emit('data:position', data)

    // Update components
    const positionMonitor = this.components.get('position-monitor')
    if (positionMonitor?.updatePosition) {
      data.forEach((position) => positionMonitor.updatePosition(position))
    }

    const manualTrading = this.components.get('manual-trading')
    if (manualTrading?.updatePosition) {
      data.forEach((position) => manualTrading.updatePosition(position))
    }
  }

  /**
   * Update risk metrics
   */
  private updateRiskMetrics(data: RiskMetrics): void {
    this.state.risk_metrics = data
    this.addToStateHistory()
    this.emit('data:risk', data)

    // Update components
    const riskManagement = this.components.get('risk-management')
    if (riskManagement?.updateRiskMetrics) {
      riskManagement.updateRiskMetrics(data)
    }

    const manualTrading = this.components.get('manual-trading')
    if (manualTrading?.updateRiskMetrics) {
      manualTrading.updateRiskMetrics(data)
    }
  }

  /**
   * Update bot status
   */
  private updateBotStatus(data: BotStatus): void {
    this.state.bot_status = data
    this.addToStateHistory()
    this.emit('data:bot_status', data)
  }

  /**
   * Update connection status
   */
  private updateConnectionStatus(status: ConnectionStatus): void {
    this.state.connection_status = status
    this.emit('connection:status', status)

    // Update all components
    this.components.forEach((component, _id) => {
      if (component.setConnectionStatus) {
        component.setConnectionStatus(status)
      }
    })
  }

  /**
   * Handle trade execution
   */
  private handleTradeExecution(data: TradeAction): void {
    this.state.latest_action = data
    this.addToStateHistory()
    this.emit('trade:executed', data)

    // Send notification
    if (this.notificationSystem) {
      this.notificationSystem.sendTradingAlert(
        'trade_executed',
        {
          symbol: data.symbol,
          side: data.action,
          quantity: data.size_percentage,
          price: data.confidence || 0,
          pnl: 0, // Would be calculated
        },
        'normal'
      )
    }

    // Update analytics
    this.analytics.componentInteractions++
    this.trackFeatureUsage('trade_execution')
  }

  /**
   * Handle alert
   */
  private handleAlert(data: any): void {
    this.emit('alert:received', data)

    // Send notification if critical
    if (this.notificationSystem && data.priority === 'critical') {
      this.notificationSystem.sendQuickNotification(
        'critical_alert',
        data.title,
        data.message,
        'critical'
      )
    }
  }

  /**
   * Setup component event handlers
   */
  private setupComponentEventHandlers(componentId: string, component: any): void {
    // Manual Trading events
    if (component.onTradeExecuted) {
      component.onTradeExecuted((trade: any) => {
        this.emit('component:trade_executed', { componentId, trade })
        this.analytics.componentInteractions++
      })
    }

    if (component.onError) {
      component.onError((error: string) => {
        this.handleComponentError(componentId, new Error(error))
      })
    }

    // Risk Management events
    if (component.onRiskAlert) {
      component.onRiskAlert((alert: any) => {
        this.emit('component:risk_alert', { componentId, alert })
        this.handleAlert(alert)
      })
    }

    if (component.onConfigUpdate) {
      component.onConfigUpdate((config: any) => {
        this.emit('component:config_updated', { componentId, config })
      })
    }

    // Position Monitor events
    if (component.onPositionAlert) {
      component.onPositionAlert((alert: any) => {
        this.emit('component:position_alert', { componentId, alert })
        this.handleAlert(alert)
      })
    }
  }

  /**
   * Setup cross-component communication
   */
  private setupCrossComponentCommunication(): void {
    // Manual trading → Risk management
    this.addEventListener('component:trade_executed', (data) => {
      const riskComponent = this.components.get('risk-management')
      if (riskComponent?.validateTrade) {
        riskComponent.validateTrade(data.trade)
      }
    })

    // Risk management → Position monitor
    this.addEventListener('component:risk_alert', (data) => {
      const positionComponent = this.components.get('position-monitor')
      if (positionComponent?.addAlert) {
        positionComponent.addAlert(data.alert)
      }
    })

    // Position monitor → Performance analytics
    this.addEventListener('component:position_alert', (data) => {
      const analyticsComponent = this.components.get('performance-analytics')
      if (analyticsComponent?.recordEvent) {
        analyticsComponent.recordEvent('position_alert', data.alert)
      }
    })
  }

  /**
   * Update components with new data
   */
  private updateComponents(dataType: string, data: any): void {
    const startTime = performance.now()

    this.components.forEach((component, id) => {
      try {
        const componentStartTime = performance.now()

        switch (dataType) {
          case 'market_data':
            if (component.updateMarketData) {
              component.updateMarketData(data)
            }
            break
          case 'positions':
            if (component.updatePosition) {
              data.forEach((position: Position) => component.updatePosition(position))
            }
            break
          case 'risk_metrics':
            if (component.updateRiskMetrics) {
              component.updateRiskMetrics(data)
            }
            break
          case 'bot_status':
            if (component.updateBotStatus) {
              component.updateBotStatus(data)
            }
            break
        }

        const componentTime = performance.now() - componentStartTime
        this.recordComponentRenderTime(id, componentTime)
      } catch (error) {
        this.handleComponentError(id, error as Error)
      }
    })

    const _totalTime = performance.now() - startTime
    this.performanceMetrics.renderTime = totalTime
  }

  /**
   * Update components with trading mode configuration
   */
  private updateComponentsTradingMode(config: TradingModeConfig): void {
    this.components.forEach((component, id) => {
      try {
        if (component.setTradingModeConfig) {
          component.setTradingModeConfig(config)
        }
      } catch (error) {
        this.log(`Error updating trading mode for component ${id}: ${error}`)
      }
    })
  }

  /**
   * Propagate state updates to all components
   */
  private propagateStateUpdate(_data: any): void {
    this.components.forEach((component, _id) => {
      if (component.onStateUpdate) {
        try {
          component.onStateUpdate(this.state)
        } catch (error) {
          this.handleComponentError(id, error as Error)
        }
      }
    })
  }

  /**
   * Process update queue in batches
   */
  private processUpdateQueue(): void {
    const updates = this.updateQueue.splice(0)
    const batches = new Map<string, any[]>()

    // Group updates by type
    updates.forEach((update) => {
      if (!batches.has(update.component)) {
        batches.set(update.component, [])
      }
      batches.get(update.component)!.push(update.data)
    })

    // Process batches
    batches.forEach((batch, component) => {
      this.emit(`batch:${component}`, batch)
    })

    this.performanceMetrics.updateFrequency = updates.length
  }

  /**
   * Setup performance monitoring
   */
  private setupPerformanceMonitoring(): void {
    if (!this.config.performance.enableMetrics) return

    // Setup Performance Observer
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        entries.forEach((entry) => {
          if (entry.entryType === 'measure') {
            this.recordPerformanceMeasure(entry)
          }
        })
      })

      this.performanceObserver.observe({ entryTypes: ['measure', 'navigation'] })
    }

    // Setup memory monitoring
    if ('memory' in performance) {
      this.memoryMonitor = window.setInterval(() => {
        const memory = (performance as any).memory
        this.performanceMetrics.memoryUsage = memory.usedJSHeapSize

        if (memory.usedJSHeapSize > this.config.performance.maxMemoryUsage) {
          this.handlePerformanceIssue('memory', memory.usedJSHeapSize)
        }
      }, 5000)
    }
  }

  /**
   * Start performance monitoring
   */
  private startPerformanceMonitoring(): void {
    const interval = setInterval(() => {
      this.updatePerformanceMetrics()
      this.emit('performance:update', this.performanceMetrics)
    }, 10000)

    this.intervals.push(interval)
  }

  /**
   * Start analytics tracking
   */
  private startAnalyticsTracking(): void {
    const interval = setInterval(() => {
      this.updateAnalytics()
      this.emit('analytics:update', this.analytics)
    }, 30000)

    this.intervals.push(interval)
  }

  /**
   * Start auto-refresh
   */
  private startAutoRefresh(): void {
    const interval = setInterval(() => {
      this.refreshData()
    }, this.config.autoRefreshInterval)

    this.intervals.push(interval)
  }

  /**
   * Refresh data from API
   */
  private async refreshData(): Promise<void> {
    try {
      // Refresh trading mode
      const modeResponse = await fetch(`${this.config.apiBaseUrl}/api/trading-mode`)
      if (modeResponse.ok) {
        const modeConfig = await modeResponse.json()
        this.state.trading_mode_config = modeConfig
        this.emit('trading_mode:updated', modeConfig)
      }

      // Refresh market data
      const marketResponse = await fetch(`${this.config.apiBaseUrl}/api/bot/market-data`)
      if (marketResponse.ok) {
        const marketData = await marketResponse.json()
        this.updateMarketData(marketData)
      }

      // Refresh positions
      const positionsResponse = await fetch(`${this.config.apiBaseUrl}/api/bot/positions`)
      if (positionsResponse.ok) {
        const positions = await positionsResponse.json()
        this.updatePositions(positions)
      }

      // Refresh risk metrics
      const riskResponse = await fetch(`${this.config.apiBaseUrl}/api/bot/risk`)
      if (riskResponse.ok) {
        const riskData = await riskResponse.json()
        this.updateRiskMetrics(riskData)
      }
    } catch (error) {
      this.handleError('data_refresh', error as Error)
    }
  }

  /**
   * Register default notification channels
   */
  private registerDefaultNotificationChannels(): void {
    if (!this.notificationSystem) return

    const channels = [
      {
        id: 'browser',
        type: 'browser' as const,
        name: 'Browser Notifications',
        enabled: true,
        priority: 1,
        config: {
          icon: '/favicon.ico',
        },
      },
      {
        id: 'email',
        type: 'email' as const,
        name: 'Email Notifications',
        enabled: true,
        priority: 2,
        config: {},
        rateLimit: {
          maxMessages: 10,
          windowMs: 3600000, // 1 hour
        },
      },
      {
        id: 'webhook',
        type: 'webhook' as const,
        name: 'Webhook Notifications',
        enabled: false,
        priority: 3,
        config: {
          url: '',
          headers: {},
        },
      },
    ]

    this.notificationSystem.registerChannels(channels)
  }

  /**
   * Setup trading alert templates
   */
  private setupTradingAlertTemplates(): void {
    if (!this.notificationSystem) return

    const templates = [
      {
        name: 'Trade Executed',
        type: 'trading_alert_trade_executed',
        subject: 'Trade Executed: {{side}} {{symbol}}',
        body: 'Trade executed: {{side}} {{quantity}} {{symbol}} at ${{price}}\nP&L: ${{pnl}}',
        variables: ['side', 'symbol', 'quantity', 'price', 'pnl'],
        channels: ['browser', 'email'],
        priority: 'normal' as const,
        tags: ['trading', 'execution'],
      },
      {
        name: 'Risk Alert',
        type: 'trading_alert_risk_warning',
        subject: '⚠️ Risk Alert: {{warning}}',
        body: 'Risk warning: {{message}}\nCurrent exposure: {{exposure}}\nRecommended action: {{action}}',
        variables: ['warning', 'message', 'exposure', 'action'],
        channels: ['browser', 'email', 'webhook'],
        priority: 'high' as const,
        tags: ['risk', 'warning'],
      },
      {
        name: 'System Error',
        type: 'trading_alert_system_error',
        subject: '🚨 System Error: {{error}}',
        body: 'System error occurred: {{message}}\nComponent: {{component}}\nTimestamp: {{timestamp}}',
        variables: ['error', 'message', 'component', 'timestamp'],
        channels: ['browser', 'email', 'webhook'],
        priority: 'critical' as const,
        tags: ['system', 'error', 'critical'],
      },
    ]

    templates.forEach((template) => {
      this.notificationSystem!.createTemplate(template)
    })
  }

  /**
   * Handle component errors
   */
  private handleComponentError(componentId: string, error: Error): void {
    this.analytics.errorsEncountered++

    const errorInfo = this.errorRecovery.get(componentId) || { count: 0, lastError: new Date() }
    errorInfo.count++
    errorInfo.lastError = new Date()
    this.errorRecovery.set(componentId, errorInfo)

    this.log(`Component error in ${componentId}: ${error.message}`)
    this.emit('component:error', { componentId, error, count: errorInfo.count })

    // Attempt recovery if error count is manageable
    if (errorInfo.count <= 3) {
      this.attemptComponentRecovery(componentId)
    } else {
      this.disableComponent(componentId)
    }
  }

  /**
   * Attempt to recover a failed component
   */
  private async attemptComponentRecovery(componentId: string): Promise<void> {
    try {
      const config = this.config.components.find((c) => c.id === componentId)
      if (!config) return

      this.log(`Attempting to recover component: ${componentId}`)

      // Destroy existing component
      const existingComponent = this.components.get(componentId)
      if (existingComponent?.destroy) {
        existingComponent.destroy()
      }

      this.components.delete(componentId)

      // Reinitialize component
      await this.initializeComponent(config)

      this.log(`Component ${componentId} recovered successfully`)
      this.emit('component:recovered', { componentId })
    } catch (error) {
      this.log(`Failed to recover component ${componentId}: ${error}`)
      this.disableComponent(componentId)
    }
  }

  /**
   * Disable a component
   */
  private disableComponent(componentId: string): void {
    const config = this.config.components.find((c) => c.id === componentId)
    if (config) {
      config.enabled = false
      config.lifecycle.active = false
      config.lifecycle.error = 'Component disabled due to repeated errors'
    }

    const component = this.components.get(componentId)
    if (component?.destroy) {
      component.destroy()
    }

    this.components.delete(componentId)
    this.performanceMetrics.activeComponents = this.components.size

    this.log(`Component ${componentId} disabled`)
    this.emit('component:disabled', { componentId })
  }

  /**
   * Handle general errors
   */
  private handleError(context: string, error: Error): void {
    this.analytics.errorsEncountered++
    this.performanceMetrics.errorRate =
      (this.analytics.errorsEncountered / this.analytics.sessionDuration) * 1000

    this.log(`Error in ${context}: ${error.message}`)
    this.emit('error', { context, error })

    // Send critical error notification
    if (this.notificationSystem) {
      this.notificationSystem.sendTradingAlert(
        'system_error',
        {
          error: error.name,
          message: error.message,
          component: context,
          timestamp: new Date().toISOString(),
        },
        'critical'
      )
    }
  }

  /**
   * Handle performance issues
   */
  private handlePerformanceIssue(type: string, value: number): void {
    this.log(`Performance issue detected: ${type} = ${value}`)
    this.emit('performance:issue', { type, value })

    // Take corrective action
    switch (type) {
      case 'memory':
        this.optimizeMemoryUsage()
        break
      case 'cpu':
        this.reduceCpuUsage()
        break
      case 'render':
        this.optimizeRendering()
        break
    }
  }

  /**
   * Performance optimization methods
   */
  private optimizeMemoryUsage(): void {
    // Clear old cache entries
    const now = Date.now()
    for (const [key, entry] of this.dataCache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.dataCache.delete(key)
      }
    }

    // Clear old state history
    if (this.stateHistory.length > this.maxStateHistory) {
      this.stateHistory = this.stateHistory.slice(-this.maxStateHistory / 2)
    }

    // Force garbage collection if available
    if ('gc' in window) {
      ;(window as any).gc()
    }
  }

  private reduceCpuUsage(): void {
    // Increase update intervals
    this.intervals.forEach((interval) => {
      clearInterval(interval)
    })
    this.intervals = []

    // Restart with longer intervals
    this.startPerformanceMonitoring()
    this.startAnalyticsTracking()
  }

  private optimizeRendering(): void {
    // Reduce update frequency for non-critical components
    this.components.forEach((component, _id) => {
      if (component.setUpdateFrequency) {
        component.setUpdateFrequency('low')
      }
    })
  }

  /**
   * Utility methods
   */
  private updatePerformanceMetrics(): void {
    this.performanceMetrics.activeComponents = this.components.size
    this.performanceMetrics.latency = this.calculateAverageLatency()
    this.performanceMetrics.throughput = this.calculateThroughput()
    this.performanceMetrics.cacheHitRate = this.calculateCacheHitRate()
  }

  private updateAnalytics(): void {
    this.analytics.sessionDuration = Date.now() - this.analytics.sessionStart.getTime()
    this.analytics.performanceMetrics = { ...this.performanceMetrics }
  }

  private calculateAverageLatency(): number {
    // Calculate from WebSocket metrics if available
    if (this.websocketManager) {
      const metrics = this.websocketManager.getMetrics()
      return metrics.averageLatency
    }
    return 0
  }

  private calculateThroughput(): number {
    // Calculate messages per second
    const duration = this.analytics.sessionDuration / 1000
    return duration > 0 ? this.performanceMetrics.totalEvents / duration : 0
  }

  private calculateCacheHitRate(): number {
    // Simple cache hit rate calculation
    const totalRequests = this.dataCache.size
    return totalRequests > 0 ? 0.85 : 0 // Placeholder
  }

  private recordComponentRenderTime(componentId: string, time: number): void {
    if (!this.renderProfiler.has(componentId)) {
      this.renderProfiler.set(componentId, [])
    }

    const times = this.renderProfiler.get(componentId)!
    times.push(time)

    // Keep only last 50 measurements
    if (times.length > 50) {
      times.shift()
    }
  }

  private recordPerformanceMeasure(entry: PerformanceEntry): void {
    // Record performance measurements for analysis
    this.log(`Performance measure: ${entry.name} = ${entry.duration}ms`)
  }

  private addToStateHistory(): void {
    this.stateHistory.push({ ...this.state })
    if (this.stateHistory.length > this.maxStateHistory) {
      this.stateHistory.shift()
    }
  }

  private trackFeatureUsage(feature: string): void {
    if (!this.analytics.featuresUsed.includes(feature)) {
      this.analytics.featuresUsed.push(feature)
    }
  }

  private setupErrorHandling(): void {
    window.addEventListener('error', (event) => {
      this.handleError('global', event.error)
    })

    window.addEventListener('unhandledrejection', (event) => {
      this.handleError('promise', new Error(event.reason))
    })
  }

  private loadPluginComponent(config: ComponentConfig): Promise<any> {
    // Plugin loading logic would go here
    throw new Error(`Unknown component type: ${config.type}`)
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventBus.get(event)
    if (listeners) {
      listeners.forEach((callback) => {
        try {
          callback(data)
        } catch (error) {
          this.log(`Error in event listener for ${event}: ${error}`)
        }
      })
    }
  }

  private log(message: string): void {
    if (this.config.debugMode) {
      // DEBUG: Dashboard orchestrator debug logging
      // console.log(`[DashboardOrchestrator] ${message}`)
    }
  }

  /**
   * Public API methods
   */
  public addEventListener(event: string, callback: Function): void {
    if (!this.eventBus.has(event)) {
      this.eventBus.set(event, new Set())
    }
    this.eventBus.get(event)!.add(callback)
  }

  public removeEventListener(event: string, callback: Function): void {
    const listeners = this.eventBus.get(event)
    if (listeners) {
      listeners.delete(callback)
      if (listeners.size === 0) {
        this.eventBus.delete(event)
      }
    }
  }

  public getState(): DashboardState {
    return { ...this.state }
  }

  public getStateHistory(): DashboardState[] {
    return [...this.stateHistory]
  }

  public getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics }
  }

  public getAnalytics(): DashboardAnalytics {
    return { ...this.analytics }
  }

  public getTradingModeConfig(): TradingModeConfig | undefined {
    return this.state.trading_mode_config
  }

  public getTradingMode(): TradingMode {
    return this.state.trading_mode_config?.trading_mode || 'spot'
  }

  public isFuturesEnabled(): boolean {
    return this.state.trading_mode_config?.futures_enabled || false
  }

  public getComponent(id: string): any {
    return this.components.get(id)
  }

  public getAllComponents(): Map<string, any> {
    return new Map(this.components)
  }

  public enableComponent(id: string): void {
    const config = this.config.components.find((c) => c.id === id)
    if (config) {
      config.enabled = true
      this.initializeComponent(config)
    }
  }

  public disableComponentById(id: string): void {
    this.disableComponent(id)
  }

  public updateConfig(newConfig: Partial<DashboardConfig>): void {
    this.config = { ...this.config, ...newConfig }
    this.emit('config:updated', this.config)
  }

  public sendNotification(message: any): Promise<any> {
    if (this.notificationSystem) {
      return this.notificationSystem.sendNotification(message)
    }
    return Promise.reject(new Error('Notification system not available'))
  }

  public executeCommand(command: string, args: any = {}): Promise<any> {
    this.emit('command:execute', { command, args })

    switch (command) {
      case 'refresh_data':
        return this.refreshData()
      case 'optimize_performance':
        this.optimizeMemoryUsage()
        this.optimizeRendering()
        return Promise.resolve()
      case 'export_analytics':
        return Promise.resolve(this.analytics)
      default:
        return Promise.reject(new Error(`Unknown command: ${command}`))
    }
  }

  /**
   * Clean up resources
   */
  public destroy(): void {
    // Stop all intervals
    this.intervals.forEach((interval) => clearInterval(interval))
    this.intervals = []

    // Stop performance monitoring
    if (this.performanceObserver) {
      this.performanceObserver.disconnect()
    }

    if (this.memoryMonitor) {
      clearInterval(this.memoryMonitor)
    }

    // Destroy all components
    this.components.forEach((component, _id) => {
      if (component.destroy) {
        component.destroy()
      }
    })
    this.components.clear()

    // Clean up services
    if (this.websocketManager) {
      this.websocketManager.destroy()
    }

    if (this.notificationSystem) {
      this.notificationSystem.destroy()
    }

    // Clear caches and data
    this.dataCache.clear()
    this.eventBus.clear()
    this.stateHistory = []
    this.updateQueue = []

    this.log('Dashboard Orchestrator destroyed')
  }
}
