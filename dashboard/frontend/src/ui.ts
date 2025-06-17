import type {
  DashboardState,
  BotStatus,
  MarketData,
  TradeAction,
  Position,
  RiskMetrics,
  ConnectionStatus,
  LogEntry,
  VuManchuIndicators,
} from './types'
import { PerformanceCharts } from './components/performance-charts'

export class DashboardUI {
  private state: DashboardState
  private logEntries: LogEntry[] = []
  private maxLogEntries = 50
  private logPaused = false
  private animationQueue: Set<string> = new Set()
  private updateThrottles: Map<string, number> = new Map()
  private readonly maxUpdateThrottles = 50
  private performanceMetrics = {
    totalReturn: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    maxDrawdown: 0,
    sharpeRatio: 0,
  }
  private performanceCharts: PerformanceCharts | null = null
  private performanceChartsVisible = false

  // Error handling and resilience
  private errorBoundaries = new Map<string, { count: number; lastError: Date; maxErrors: number }>()
  private fallbackData = new Map<string, unknown>()
  private readonly maxErrorsPerBoundary = 5
  private readonly errorResetTime = 5 * 60 * 1000 // 5 minutes
  private retryTimeouts = new Map<string, number>()
  private readonly defaultRetryDelays = [1000, 2000, 5000, 10000] // Exponential backoff
  private isOfflineMode = false
  private lastDataUpdate = new Map<string, Date>()
  private readonly dataStaleThreshold = 30000 // 30 seconds

  // DOM cache for performance
  private domCache = new Map<string, HTMLElement>()
  private readonly maxDomCacheSize = 30

  // Update batching for performance
  private updateBatch = new Set<() => void>()
  private batchTimer: number | null = null
  private readonly BATCH_DELAY = 16 // ~60fps

  // Memory cleanup timer
  private cleanupTimer: number | null = null
  private readonly CLEANUP_INTERVAL = 30000 // 30 seconds

  constructor() {
    this.state = {
      bot_status: null,
      market_data: null,
      latest_action: null,
      indicators: null,
      positions: [],
      risk_metrics: null,
      connection_status: 'disconnected',
      error_message: null,
    }

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
    // Clean up log entries
    if (this.logEntries.length > this.maxLogEntries) {
      this.logEntries = this.logEntries.slice(-this.maxLogEntries)
    }

    // Clean up animation queue
    this.animationQueue.clear()

    // Clean up update throttles
    if (this.updateThrottles.size > this.maxUpdateThrottles) {
      const entries = Array.from(this.updateThrottles.entries())
      const now = Date.now()
      const activeThrottles = entries.filter(([_, timestamp]) => now - timestamp < 5000)
      this.updateThrottles = new Map(activeThrottles.slice(-this.maxUpdateThrottles))
    }

    // Clean up DOM cache
    if (this.domCache.size > this.maxDomCacheSize) {
      const entries = Array.from(this.domCache.entries())
      this.domCache = new Map(entries.slice(-this.maxDomCacheSize))
    }

    // Clear update batch if stale
    if (this.updateBatch.size > 0 && !this.batchTimer) {
      this.updateBatch.clear()
    }
  }

  /**
   * Get cached DOM element
   */
  private getCachedElement(selector: string): HTMLElement | null {
    if (this.domCache.has(selector)) {
      return this.domCache.get(selector)!
    }

    const element = document.querySelector(selector) as HTMLElement
    if (element) {
      this.domCache.set(selector, element)
    }

    return element
  }

  /**
   * Batch DOM updates for better performance
   */
  private batchDOMUpdate(updateFn: () => void): void {
    this.updateBatch.add(updateFn)

    if (!this.batchTimer) {
      this.batchTimer = window.setTimeout(() => {
        this.flushUpdateBatch()
      }, this.BATCH_DELAY)
    }
  }

  /**
   * Flush batched updates
   */
  private flushUpdateBatch(): void {
    if (this.updateBatch.size === 0) return

    // Use requestAnimationFrame for smooth updates
    requestAnimationFrame(() => {
      this.updateBatch.forEach((updateFn) => {
        try {
          updateFn()
        } catch (error) {
          console.error('Error in batched update:', error)
        }
      })

      this.updateBatch.clear()
      this.batchTimer = null
    })
  }

  /**
   * Initialize the dashboard UI
   */
  public initialize(): void {
    this.setupEventListeners()
    this.showLoadingScreen()
    this.updateConnectionStatus('disconnected')
    this.log('info', 'Dashboard UI initialized')
    this.startPerformanceMonitoring()
  }

  /**
   * Show the loading screen
   */
  public showLoadingScreen(): void {
    const loading = document.getElementById('loading')
    const dashboard = document.getElementById('dashboard')

    if (loading) loading.setAttribute('data-loading', 'true')
    if (dashboard) dashboard.setAttribute('data-dashboard', 'hidden')
  }

  /**
   * Hide loading screen and show dashboard
   */
  public hideLoadingScreen(): void {
    const loading = document.getElementById('loading')
    const dashboard = document.getElementById('dashboard')

    if (loading) {
      loading.setAttribute('data-loading', 'false')
      setTimeout(() => (loading.style.display = 'none'), 500)
    }
    if (dashboard) {
      setTimeout(() => dashboard.setAttribute('data-dashboard', 'visible'), 200)
    }
  }

  /**
   * Update loading progress
   */
  public updateLoadingProgress(progress: number, message?: string): void {
    const progressBar = document.querySelector('[data-progress]') as HTMLElement
    const loadingMessage = document.querySelector('.loading-message') as HTMLElement

    if (progressBar) {
      progressBar.setAttribute('data-progress', progress.toString())
      progressBar.style.setProperty('--progress', `${progress}%`)
    }

    if (loadingMessage && message) {
      loadingMessage.textContent = message
    }
  }

  /**
   * Set up event listeners for UI interactions
   */
  private setupEventListeners(): void {
    // Chart fullscreen toggle
    const fullscreenBtn = document.querySelector(
      '[data-chart-action="fullscreen"]'
    ) as HTMLButtonElement
    fullscreenBtn?.addEventListener('click', () => {
      this.toggleChartFullscreen()
      this.log('info', 'Chart fullscreen toggled', 'UI')
    })

    // Chart retry button
    const retryBtn = document.querySelector('[data-chart-retry]') as HTMLButtonElement
    retryBtn?.addEventListener('click', () => {
      this.onChartRetry?.()
      this.log('info', 'Chart retry requested', 'UI')
    })

    // Error modal close buttons
    const errorModalClose = document.querySelectorAll('[data-modal-close]')
    errorModalClose.forEach((btn) => {
      btn.addEventListener('click', () => {
        this.hideError()
      })
    })

    // Error retry button
    const errorRetryBtn = document.querySelector('[data-error-retry]') as HTMLButtonElement
    errorRetryBtn?.addEventListener('click', () => {
      this.hideError()
      this.onErrorRetry?.()
    })

    // Log controls
    const logClearBtn = document.querySelector('[data-log-action="clear"]') as HTMLButtonElement
    logClearBtn?.addEventListener('click', () => {
      this.clearLogs()
    })

    const logPauseBtn = document.querySelector('[data-log-action="pause"]') as HTMLButtonElement
    logPauseBtn?.addEventListener('click', () => {
      this.toggleLogPause()
    })

    // Modal overlay clicks
    const modalOverlay = document.querySelector('[data-modal-overlay]')
    modalOverlay?.addEventListener('click', () => {
      this.hideError()
    })

    // Performance charts toggle
    const performanceBtn = document.querySelector('[data-toggle-performance]') as HTMLButtonElement
    performanceBtn?.addEventListener('click', () => {
      this.togglePerformanceCharts()
    })
  }

  // Event callbacks
  private onChartRetry: (() => void) | null = null
  private onErrorRetry: (() => void) | null = null

  /**
   * Update connection status with animation and offline mode handling
   */
  public updateConnectionStatus(status: ConnectionStatus): void {
    this.withErrorBoundary(
      'updateConnectionStatus',
      () => {
        this.state.connection_status = status

        // Handle offline mode
        const wasOffline = this.isOfflineMode
        this.isOfflineMode = status === 'disconnected' || status === 'error'

        if (this.isOfflineMode && !wasOffline) {
          this.enterOfflineMode()
        } else if (!this.isOfflineMode && wasOffline) {
          this.exitOfflineMode()
        }

        const connectionEl = this.getCachedElement('[data-connection]')
        const statusText = connectionEl?.querySelector('.status-text') as HTMLElement
        const statusIndicator = connectionEl?.querySelector('.status-indicator') as HTMLElement

        if (connectionEl && statusText && statusIndicator) {
          // Add animation class
          this.addAnimation(connectionEl, 'pulse')

          connectionEl.setAttribute('data-connection', status)
          statusText.textContent = this.getConnectionStatusText(status)
          statusIndicator.setAttribute('aria-label', `Connection status: ${status}`)

          // Show offline indicator if needed
          this.updateOfflineIndicator()
        }

        // Update last update time
        this.updateLastUpdateTime()
        this.log('info', `Connection status: ${status}`)
      },
      { status }
    )
  }

  /**
   * Update bot status with visual feedback and error handling
   */
  public updateBotStatus(status: BotStatus | null): void {
    this.withErrorBoundary(
      'updateBotStatus',
      () => {
        if (!status) {
          // Use fallback data if available
          const fallback = this.fallbackData.get('bot_status') as BotStatus
          if (fallback) {
            status = fallback
            this.showDataFallbackIndicator('bot_status')
          } else {
            this.showMissingDataWarning('Bot Status')
            return
          }
        } else {
          // Cache valid data
          this.fallbackData.set('bot_status', status)
          this.lastDataUpdate.set('bot_status', new Date())
          this.hideMissingDataWarning('Bot Status')
        }

        this.state.bot_status = status

        // Update status bar with safe access
        const elements = {
          botStatusEl: this.getCachedElement('[data-bot-status]'),
          botIndicator: this.getCachedElement('[data-bot-indicator]'),
          tradingModeEl: this.getCachedElement('[data-trading-mode]'),
          tradingSymbolEl: this.getCachedElement('[data-trading-symbol]'),
          leverageEl: this.getCachedElement('[data-leverage]'),
        }

        if (elements.botStatusEl && status.status) {
          this.addAnimation(elements.botStatusEl, 'flash')
          elements.botStatusEl.textContent = this.safeString(status.status).toUpperCase()
        }

        if (elements.botIndicator && status.status) {
          elements.botIndicator.setAttribute(
            'data-bot-indicator',
            this.getBotIndicatorState(status.status)
          )
        }

        if (elements.tradingModeEl) {
          const isDryRun = Boolean(status.dry_run)
          elements.tradingModeEl.textContent = isDryRun ? 'Dry Run' : 'Live Trading'
          elements.tradingModeEl.className = isDryRun ? 'status-value safe' : 'status-value danger'
        }

        if (elements.tradingSymbolEl && status.symbol) {
          elements.tradingSymbolEl.textContent = this.safeString(status.symbol)
        }

        if (elements.leverageEl && status.leverage != null) {
          elements.leverageEl.textContent = `${this.safeNumber(status.leverage)}x`
        }

        this.updateLastUpdateTime()
        const logMsg = `Bot status: ${status.status} (${status.symbol}, leverage: ${status.leverage}x, dry_run: ${status.dry_run})`
        this.log('info', logMsg)
      },
      status
    )
  }

  /**
   * Update market data with price change animations and error handling
   */
  public updateMarketData(data: MarketData | null): void {
    this.withErrorBoundary(
      'updateMarketData',
      () => {
        if (!data || this.shouldThrottleUpdate('market_data', 1000)) {
          // Check if we have stale data and should show warning
          if (!data) {
            this.checkDataStaleness('market_data')
          }
          return
        }

        // Validate data structure
        if (!this.validateMarketData(data)) {
          this.log('warn', 'Invalid market data received, using fallback', 'UI')
          const fallback = this.fallbackData.get('market_data') as MarketData
          if (fallback) {
            data = fallback
            this.showDataFallbackIndicator('market_data')
          } else {
            this.showMissingDataWarning('Market Data')
            return
          }
        } else {
          // Cache valid data
          this.fallbackData.set('market_data', data)
          this.lastDataUpdate.set('market_data', new Date())
          this.hideMissingDataWarning('Market Data')
        }

        this.state.market_data = data

        const priceEl = this.getCachedElement('[data-current-price]')
        const changeEl = this.getCachedElement('[data-price-change]')

        if (priceEl && data.price != null) {
          const oldPriceText = priceEl.textContent?.replace(/[$,]/g, '') || '0'
          const oldPrice = this.safeNumber(parseFloat(oldPriceText))
          const newPrice = this.safeNumber(data.price)

          priceEl.textContent = this.formatPrice(newPrice)

          // Add price change animation
          if (oldPrice > 0 && oldPrice !== newPrice) {
            const changeClass = newPrice > oldPrice ? 'price-up' : 'price-down'
            this.addAnimation(priceEl, changeClass)
          }
        }

        if (changeEl && data.change_percent_24h != null) {
          const change = this.safeNumber(data.change_percent_24h)
          changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`
          changeEl.className = `stat-change ${change >= 0 ? 'positive' : 'negative'}`
          changeEl.setAttribute('data-price-change', change.toString())
        }

        this.updateLastUpdateTime()
      },
      data
    )
  }

  /**
   * Update latest trade action with comprehensive display and error handling
   */
  public updateLatestAction(action: TradeAction | null): void {
    this.withErrorBoundary(
      'updateLatestAction',
      () => {
        if (!action) {
          const fallback = this.fallbackData.get('trade_action') as TradeAction
          if (fallback) {
            action = fallback
            this.showDataFallbackIndicator('trade_action')
          } else {
            this.log('warn', 'No trade action data available', 'UI')
            return
          }
        } else {
          // Validate and cache data
          if (this.validateTradeAction(action)) {
            this.fallbackData.set('trade_action', action)
            this.lastDataUpdate.set('trade_action', new Date())
          } else {
            this.log('warn', 'Invalid trade action data received', 'UI')
            return
          }
        }

        this.state.latest_action = action

        // Add to AI decision log with error handling
        try {
          this.addAIDecision(action)
        } catch (error) {
          this.log('error', `Failed to add AI decision to log: ${error}`, 'UI')
        }

        const confidence = this.safeNumber(action.confidence || 0)
        const logMsg = `Trade action: ${action.action} (${(confidence * 100).toFixed(1)}% confidence) - ${this.safeString(action.reasoning)}`
        this.log('info', logMsg)
      },
      action
    )
  }

  /**
   * Add AI decision to the log with formatting
   */
  private addAIDecision(action: TradeAction): void {
    const logContainer = document.querySelector('[data-log-content]') as HTMLElement
    const logEmpty = document.querySelector('[data-log-empty]') as HTMLElement

    if (!logContainer) return

    // Hide empty state
    if (logEmpty) {
      logEmpty.setAttribute('data-log-empty', 'false')
    }

    const decision = document.createElement('div')
    decision.className = 'ai-decision-entry'
    decision.innerHTML = `
      <div class="decision-header">
        <span class="decision-action ${action.action.toLowerCase()}">${action.action}</span>
        <span class="decision-confidence">${(action.confidence * 100).toFixed(1)}%</span>
        <span class="decision-time">${new Date(action.timestamp).toLocaleTimeString()}</span>
      </div>
      <div class="decision-reasoning">
        ${this.formatReasoning(action.reasoning)}
      </div>
      ${action.price ? `<div class="decision-price">Price: ${this.formatPrice(action.price)}</div>` : ''}
    `

    // Add animation
    decision.style.opacity = '0'
    decision.style.transform = 'translateY(-10px)'

    logContainer.insertBefore(decision, logContainer.firstChild)

    // Animate in
    requestAnimationFrame(() => {
      decision.style.transition = 'all 0.3s ease'
      decision.style.opacity = '1'
      decision.style.transform = 'translateY(0)'
    })

    // Remove old entries
    const entries = logContainer.querySelectorAll('.ai-decision-entry')
    if (entries.length > 20) {
      const oldEntry = entries[entries.length - 1] as HTMLElement
      oldEntry.style.opacity = '0'
      setTimeout(() => oldEntry.remove(), 300)
    }
  }

  /**
   * Update positions display with error handling
   */
  public updatePositions(positions: Position[] | null): void {
    this.withErrorBoundary(
      'updatePositions',
      () => {
        if (!positions || !Array.isArray(positions)) {
          const fallback = this.fallbackData.get('positions') as Position[]
          if (fallback) {
            positions = fallback
            this.showDataFallbackIndicator('positions')
          } else {
            this.showMissingDataWarning('Positions')
            return
          }
        } else {
          // Validate and cache data
          const validPositions = positions.filter((pos) => this.validatePosition(pos))
          if (validPositions.length !== positions.length) {
            this.log(
              'warn',
              `Filtered ${positions.length - validPositions.length} invalid positions`,
              'UI'
            )
          }
          this.fallbackData.set('positions', validPositions)
          this.lastDataUpdate.set('positions', new Date())
          positions = validPositions
          this.hideMissingDataWarning('Positions')
        }

        this.state.positions = positions

        // Update position size in quick stats with safe calculations
        const positionSizeEl = this.getCachedElement('[data-position-size]')
        if (positionSizeEl) {
          const totalSize = positions.reduce((sum, pos) => {
            const size = this.safeNumber(pos.size || 0)
            return sum + Math.abs(size)
          }, 0)
          positionSizeEl.textContent = totalSize.toFixed(4)
        }

        // Update P&L in quick stats with safe calculations
        const pnlEl = this.getCachedElement('[data-pnl]')
        const pnlChangeEl = this.getCachedElement('[data-pnl-change]')

        if (pnlEl && pnlChangeEl) {
          const totalPnl = positions.reduce((sum, pos) => {
            return sum + this.safeNumber(pos.pnl || 0)
          }, 0)

          const totalPnlPercent =
            positions.length > 0
              ? positions.reduce((sum, pos) => {
                  return sum + this.safeNumber(pos.pnl_percent || 0)
                }, 0) / positions.length
              : 0

          pnlEl.textContent = this.formatCurrency(totalPnl)
          pnlChangeEl.textContent = `${totalPnlPercent >= 0 ? '+' : ''}${totalPnlPercent.toFixed(2)}%`
          pnlChangeEl.className = `stat-change ${totalPnlPercent >= 0 ? 'positive' : 'negative'}`
          pnlChangeEl.setAttribute('data-pnl-change', totalPnlPercent.toString())
        }

        // Update performance charts if visible with error handling
        if (this.performanceCharts && this.performanceChartsVisible) {
          try {
            this.performanceCharts.updateData(positions, this.state.risk_metrics || undefined)
          } catch (error) {
            this.log('error', `Failed to update performance charts: ${error}`, 'UI')
          }
        }

        this.updateLastUpdateTime()
      },
      positions
    )
  }

  /**
   * Update risk metrics and gauges with error handling
   */
  public updateRiskMetrics(metrics: RiskMetrics | null): void {
    this.withErrorBoundary(
      'updateRiskMetrics',
      () => {
        if (!metrics) {
          const fallback = this.fallbackData.get('risk_metrics') as RiskMetrics
          if (fallback) {
            metrics = fallback
            this.showDataFallbackIndicator('risk_metrics')
          } else {
            this.showMissingDataWarning('Risk Metrics')
            return
          }
        } else {
          // Validate and cache data
          if (this.validateRiskMetrics(metrics)) {
            this.fallbackData.set('risk_metrics', metrics)
            this.lastDataUpdate.set('risk_metrics', new Date())
            this.hideMissingDataWarning('Risk Metrics')
          } else {
            this.log('warn', 'Invalid risk metrics data received', 'UI')
            return
          }
        }

        this.state.risk_metrics = metrics

        // Update risk level gauge with safe calculations
        const elements = {
          riskFill: this.getCachedElement('[data-risk-percentage]'),
          riskText: this.getCachedElement('[data-risk-text]'),
          riskLevelEl: this.getCachedElement('[data-risk-level]'),
          riskColorEl: this.getCachedElement('[data-risk-color]'),
        }

        const riskPercentage = this.calculateRiskPercentage(metrics)
        const riskLevel = this.getRiskLevel(riskPercentage)
        const riskColor = this.getRiskColor(riskLevel)

        if (elements.riskFill) {
          elements.riskFill.style.setProperty('--risk-percentage', `${riskPercentage}%`)
          elements.riskFill.setAttribute('data-risk-percentage', riskPercentage.toString())
        }

        if (elements.riskText) {
          elements.riskText.textContent = riskLevel
          elements.riskText.setAttribute('data-risk-text', riskLevel)
        }

        if (elements.riskLevelEl) {
          elements.riskLevelEl.textContent = riskLevel
          elements.riskLevelEl.setAttribute('data-risk-level', riskLevel.toLowerCase())
        }

        if (elements.riskColorEl) {
          elements.riskColorEl.setAttribute('data-risk-color', riskColor)
        }

        // Update risk details with error handling
        try {
          this.updateRiskDetails(metrics)
        } catch (error) {
          this.log('error', `Failed to update risk details: ${error}`, 'UI')
        }

        // Update performance metrics with error handling
        try {
          this.updatePerformanceMetrics(metrics)
        } catch (error) {
          this.log('error', `Failed to update performance metrics: ${error}`, 'UI')
        }

        // Update performance charts if visible with error handling
        if (this.performanceCharts && this.performanceChartsVisible) {
          try {
            this.performanceCharts.updateData(this.state.positions, metrics)
          } catch (error) {
            this.log('error', `Failed to update performance charts: ${error}`, 'UI')
          }
        }

        this.updateLastUpdateTime()
      },
      metrics
    )
  }

  /**
   * Update risk details panel
   */
  private updateRiskDetails(metrics: RiskMetrics): void {
    const positionValueEl = document.querySelector('[data-position-value]') as HTMLElement
    const riskPerTradeEl = document.querySelector('[data-risk-per-trade]') as HTMLElement

    if (positionValueEl && this.state.positions) {
      const totalPositionValue = this.state.positions.reduce(
        (sum, pos) => sum + Math.abs((pos.size || 0) * (pos.current_price || 0)),
        0
      )
      positionValueEl.textContent = this.formatCurrency(totalPositionValue)
    }

    if (riskPerTradeEl && metrics.total_portfolio_value > 0) {
      const riskPerTrade = (metrics.margin_used / metrics.total_portfolio_value) * 100
      riskPerTradeEl.textContent = `${riskPerTrade.toFixed(1)}%`
    }
  }

  /**
   * Update performance metrics panel
   */
  private updatePerformanceMetrics(metrics: RiskMetrics): void {
    const totalReturnEl = document.querySelector('[data-total-return]') as HTMLElement
    const winRateEl = document.querySelector('[data-win-rate]') as HTMLElement
    const avgWinEl = document.querySelector('[data-avg-win]') as HTMLElement
    const avgLossEl = document.querySelector('[data-avg-loss]') as HTMLElement
    const maxDrawdownEl = document.querySelector('[data-max-drawdown]') as HTMLElement
    const sharpeRatioEl = document.querySelector('[data-sharpe-ratio]') as HTMLElement

    if (totalReturnEl && metrics.total_portfolio_value > 0) {
      const totalReturn = (metrics.total_pnl / metrics.total_portfolio_value) * 100
      totalReturnEl.textContent = `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`
      totalReturnEl.className = `metric-value ${totalReturn >= 0 ? 'positive' : 'negative'}`
    }

    if (winRateEl && metrics.win_rate != null) {
      winRateEl.textContent = `${(metrics.win_rate * 100).toFixed(0)}%`
    }

    if (avgWinEl) {
      avgWinEl.textContent = this.formatCurrency(this.performanceMetrics.avgWin)
    }

    if (avgLossEl) {
      avgLossEl.textContent = this.formatCurrency(this.performanceMetrics.avgLoss)
    }

    if (maxDrawdownEl && metrics.max_drawdown != null) {
      maxDrawdownEl.textContent = `${(metrics.max_drawdown * 100).toFixed(2)}%`
      maxDrawdownEl.className = 'metric-value negative'
    }

    if (sharpeRatioEl) {
      sharpeRatioEl.textContent = this.performanceMetrics.sharpeRatio.toFixed(2)
    }
  }

  /**
   * Update system health indicators
   */
  public updateSystemHealth(healthData: {
    api: 'connected' | 'disconnected' | 'error'
    websocket: 'connected' | 'disconnected' | 'error'
    llm: 'ready' | 'error' | 'loading'
    indicators: 'calculating' | 'ready' | 'error'
  }): void {
    const healthItems = [
      { key: 'api', selector: '[data-api-status]', indicator: '[data-api-indicator]' },
      {
        key: 'websocket',
        selector: '[data-websocket-status]',
        indicator: '[data-websocket-indicator]',
      },
      { key: 'llm', selector: '[data-llm-status]', indicator: '[data-llm-indicator]' },
      {
        key: 'indicators',
        selector: '[data-indicators-status]',
        indicator: '[data-indicators-indicator]',
      },
    ]

    healthItems.forEach(({ key, selector, indicator }) => {
      const statusEl = document.querySelector(selector) as HTMLElement
      const indicatorEl = document.querySelector(indicator) as HTMLElement
      const status = healthData[key as keyof typeof healthData]

      if (statusEl) {
        statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1)
        statusEl.setAttribute(`data-${key}-status`, status)
      }

      if (indicatorEl) {
        const color = this.getHealthColor(status)
        indicatorEl.setAttribute(`data-${key}-indicator`, color)
      }
    })
  }

  /**
   * Update footer system information
   */
  public updateSystemInfo(info: {
    version?: string
    uptime?: number
    memoryUsage?: number
    serverTime?: Date
    marketStatus?: 'open' | 'closed'
  }): void {
    const versionEl = document.querySelector('[data-version]') as HTMLElement
    const uptimeEl = document.querySelector('[data-uptime]') as HTMLElement
    const memoryEl = document.querySelector('[data-memory-usage]') as HTMLElement
    const serverTimeEl = document.querySelector('[data-server-time]') as HTMLElement
    const marketStatusEl = document.querySelector('[data-market-status]') as HTMLElement

    if (versionEl && info.version) {
      versionEl.textContent = info.version
    }

    if (uptimeEl && info.uptime) {
      uptimeEl.textContent = this.formatUptime(info.uptime)
    }

    if (memoryEl && info.memoryUsage) {
      memoryEl.textContent = `${info.memoryUsage.toFixed(0)}MB`
    }

    if (serverTimeEl && info.serverTime) {
      serverTimeEl.textContent = info.serverTime.toLocaleTimeString()
    }

    if (marketStatusEl && info.marketStatus) {
      marketStatusEl.textContent =
        info.marketStatus.charAt(0).toUpperCase() + info.marketStatus.slice(1)
      marketStatusEl.className = `timestamp-value ${info.marketStatus}`
    }
  }

  /**
   * Update API status indicators in footer
   */
  public updateAPIStatus(apiStatus: {
    coinbase: 'connected' | 'disconnected' | 'error'
    openai: 'connected' | 'disconnected' | 'error'
  }): void {
    const coinbaseIndicator = document.querySelector('[data-coinbase-status]') as HTMLElement
    const openaiIndicator = document.querySelector('[data-openai-status]') as HTMLElement

    if (coinbaseIndicator) {
      coinbaseIndicator.className = `api-indicator ${apiStatus.coinbase}`
      coinbaseIndicator.setAttribute('data-coinbase-status', apiStatus.coinbase)
    }

    if (openaiIndicator) {
      openaiIndicator.className = `api-indicator ${apiStatus.openai}`
      openaiIndicator.setAttribute('data-openai-status', apiStatus.openai)
    }
  }

  /**
   * Show error modal with details
   */
  public showError(message: string, details?: string): void {
    this.state.error_message = message

    const errorModal = document.getElementById('error-modal')
    const errorMessage = document.querySelector('[data-error-message]') as HTMLElement
    const errorStack = document.querySelector('[data-error-stack]') as HTMLElement

    if (errorModal) {
      errorModal.setAttribute('data-modal', 'visible')
      document.body.style.overflow = 'hidden' // Prevent background scroll
    }

    if (errorMessage) {
      errorMessage.textContent = message
    }

    if (errorStack && details) {
      errorStack.textContent = details
    }

    this.log('error', message, 'System')
  }

  /**
   * Hide error modal
   */
  public hideError(): void {
    this.state.error_message = null

    const errorModal = document.getElementById('error-modal')
    if (errorModal) {
      errorModal.setAttribute('data-modal', 'hidden')
      document.body.style.overflow = '' // Restore scroll
    }
  }

  /**
   * Show chart loading state
   */
  public showChartLoading(): void {
    const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement

    if (chartLoading) chartLoading.setAttribute('data-chart-loading', 'true')
    if (chartError) chartError.setAttribute('data-chart-error', 'hidden')
  }

  /**
   * Hide chart loading state
   */
  public hideChartLoading(): void {
    const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement
    if (chartLoading) chartLoading.setAttribute('data-chart-loading', 'false')
  }

  /**
   * Show chart error state
   */
  public showChartError(message: string = 'Failed to load chart'): void {
    const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement
    const errorMessage = chartError?.querySelector('p')

    if (chartLoading) chartLoading.setAttribute('data-chart-loading', 'false')
    if (chartError) chartError.setAttribute('data-chart-error', 'visible')
    if (errorMessage) errorMessage.textContent = message

    this.log('error', `Chart error: ${message}`, 'Chart')
  }

  /**
   * Hide chart error state
   */
  public hideChartError(): void {
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement
    if (chartError) chartError.setAttribute('data-chart-error', 'hidden')
  }

  /**
   * Toggle chart fullscreen mode
   */
  private toggleChartFullscreen(): void {
    const chartSection = document.querySelector('.chart-section') as HTMLElement
    const dashboard = document.querySelector('.dashboard-container') as HTMLElement

    if (chartSection && dashboard) {
      const isFullscreen = chartSection.hasAttribute('data-fullscreen')

      if (isFullscreen) {
        chartSection.removeAttribute('data-fullscreen')
        dashboard.classList.remove('chart-fullscreen')
      } else {
        chartSection.setAttribute('data-fullscreen', 'true')
        dashboard.classList.add('chart-fullscreen')
      }
    }
  }

  /**
   * Add log entry with improved formatting
   */
  public log(
    level: 'info' | 'warn' | 'error' | 'debug',
    message: string,
    component?: string
  ): void {
    if (this.logPaused) return

    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date().toISOString(),
      component,
    }

    this.logEntries.unshift(entry)

    // Keep only the latest entries with memory optimization
    if (this.logEntries.length > this.maxLogEntries) {
      this.logEntries = this.logEntries.slice(0, this.maxLogEntries)
    }

    // Batch log UI updates for better performance
    this.batchDOMUpdate(() => {
      this.updateLogDisplay()
    })
  }

  /**
   * Update log display in DOM
   */
  private updateLogDisplay(): void {
    const logContent = this.getCachedElement('[data-log-content]')
    const logEmpty = this.getCachedElement('[data-log-empty]')

    if (!logContent) return

    // Only show recent entries for performance
    const recentEntries = this.logEntries.slice(0, 20)

    // Use document fragment for efficient DOM updates
    const fragment = document.createDocumentFragment()

    recentEntries.forEach((entry, index) => {
      if (index < 20) {
        // Limit visible entries
        const logItem = this.createLogElement(entry)
        fragment.appendChild(logItem)
      }
    })

    // Clear and update content efficiently
    if (logContent.children.length !== fragment.children.length) {
      logContent.innerHTML = ''
      logContent.appendChild(fragment)
    }

    if (logEmpty) {
      logEmpty.setAttribute('data-log-empty', this.logEntries.length === 0 ? 'true' : 'false')
    }
  }

  /**
   * Create optimized log element
   */
  private createLogElement(entry: LogEntry): HTMLElement {
    const logItem = document.createElement('div')
    logItem.className = `log-item log-${entry.level}`

    // Use efficient innerHTML for simple content
    logItem.innerHTML = `
      <span class="log-time">${new Date(entry.timestamp).toLocaleTimeString()}</span>
      <span class="log-level">${entry.level.toUpperCase()}</span>
      ${entry.component ? `<span class="log-component">${entry.component}</span>` : ''}
      <span class="log-message">${entry.message}</span>
    `

    return logItem
  }

  /**
   * Clear all log entries
   */
  public clearLogs(): void {
    this.logEntries = []
    const logContent = this.getCachedElement('[data-log-content]')
    const logEmpty = this.getCachedElement('[data-log-empty]')

    if (logContent) {
      logContent.innerHTML = ''
    }

    if (logEmpty) {
      logEmpty.setAttribute('data-log-empty', 'true')
    }

    this.log('info', 'Logs cleared', 'UI')
  }

  /**
   * Destroy UI and cleanup resources
   */
  public destroy(): void {
    // Clear timers
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }

    if (this.batchTimer) {
      clearTimeout(this.batchTimer)
      this.batchTimer = null
    }

    // Clear retry timeouts
    for (const timeoutId of this.retryTimeouts.values()) {
      clearTimeout(timeoutId)
    }
    this.retryTimeouts.clear()

    // Clear collections
    this.logEntries = []
    this.animationQueue.clear()
    this.updateThrottles.clear()
    this.domCache.clear()
    this.updateBatch.clear()
    this.errorBoundaries.clear()
    this.fallbackData.clear()
    this.lastDataUpdate.clear()

    // Remove dynamic UI elements
    const elementsToRemove = [
      'notification-area',
      'offline-indicator',
      'missing-data-warnings',
      'performance-modal',
    ]

    elementsToRemove.forEach((id) => {
      const element = document.getElementById(id)
      if (element) {
        element.remove()
      }
    })

    // Destroy performance charts
    if (this.performanceCharts) {
      this.performanceCharts.destroy()
      this.performanceCharts = null
    }

    // Reset state
    this.state = {
      bot_status: null,
      market_data: null,
      latest_action: null,
      indicators: null,
      positions: [],
      risk_metrics: null,
      connection_status: 'disconnected',
      error_message: null,
    }

    // Reset flags
    this.isOfflineMode = false
    this.performanceChartsVisible = false
  }

  /**
   * Toggle log pause state
   */
  public toggleLogPause(): void {
    this.logPaused = !this.logPaused
    const pauseBtn = document.querySelector('[data-log-action="pause"]') as HTMLButtonElement
    if (pauseBtn) {
      pauseBtn.textContent = this.logPaused ? 'Resume' : 'Pause'
      pauseBtn.setAttribute(
        'aria-label',
        this.logPaused ? 'Resume log updates' : 'Pause log updates'
      )
    }
    this.log('info', `Log updates ${this.logPaused ? 'paused' : 'resumed'}`, 'UI')
  }

  // Helper methods
  private updateLastUpdateTime(): void {
    const lastUpdateEl = document.querySelector('.update-time') as HTMLElement
    if (lastUpdateEl) {
      lastUpdateEl.textContent = new Date().toLocaleTimeString()
    }
  }

  private addAnimation(element: HTMLElement, animationClass: string): void {
    // Use requestAnimationFrame for better performance
    requestAnimationFrame(() => {
      const key = element.id || element.className
      if (this.animationQueue.has(key)) return

      this.animationQueue.add(key)
      element.classList.add(animationClass)

      setTimeout(() => {
        element.classList.remove(animationClass)
        this.animationQueue.delete(key)
      }, 1000)
    })
  }

  private shouldThrottleUpdate(key: string, throttleMs: number): boolean {
    const now = Date.now()
    const lastUpdate = this.updateThrottles.get(key) || 0

    if (now - lastUpdate < throttleMs) {
      return true
    }

    this.updateThrottles.set(key, now)
    return false
  }

  private formatPrice(price: number): string {
    return `$${price.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`
  }

  private formatCurrency(amount: number): string {
    return `$${amount.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`
  }

  private formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  private formatReasoning(reasoning: string): string {
    // Basic HTML escaping and formatting
    return reasoning
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>')
  }

  private getConnectionStatusText(status: ConnectionStatus): string {
    switch (status) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting...'
      case 'disconnected':
        return 'Disconnected'
      case 'error':
        return 'Connection Error'
      default:
        return 'Unknown'
    }
  }

  private getBotIndicatorState(status: string): string {
    switch (status) {
      case 'running':
        return 'green'
      case 'stopped':
        return 'red'
      case 'error':
        return 'red'
      case 'dry_run':
        return 'yellow'
      default:
        return 'gray'
    }
  }

  private calculateRiskPercentage(metrics: RiskMetrics): number {
    if (!metrics || metrics.total_portfolio_value <= 0) return 0
    const riskRatio = metrics.margin_used / metrics.total_portfolio_value
    return Math.min(riskRatio * 100, 100)
  }

  private getRiskLevel(percentage: number): string {
    if (percentage < 20) return 'Low'
    if (percentage < 50) return 'Medium'
    if (percentage < 80) return 'High'
    return 'Critical'
  }

  private getRiskColor(level: string): string {
    switch (level.toLowerCase()) {
      case 'low':
        return 'green'
      case 'medium':
        return 'yellow'
      case 'high':
        return 'orange'
      case 'critical':
        return 'red'
      default:
        return 'gray'
    }
  }

  private getHealthColor(status: string): string {
    switch (status) {
      case 'connected':
      case 'ready':
        return 'green'
      case 'calculating':
      case 'loading':
        return 'yellow'
      case 'disconnected':
      case 'error':
        return 'red'
      default:
        return 'gray'
    }
  }

  private startPerformanceMonitoring(): void {
    // Monitor and update performance metrics periodically
    setInterval(() => {
      this.updateSystemInfo({
        serverTime: new Date(),
        memoryUsage: (performance as any).memory?.usedJSHeapSize / 1024 / 1024 || 0,
      })
    }, 1000)
  }

  // Event callback setters
  public onChartRetryRequested(callback: () => void): void {
    this.onChartRetry = callback
  }

  public onErrorRetryRequested(callback: () => void): void {
    this.onErrorRetry = callback
  }

  /**
   * Clear error message (alias for hideError)
   */
  public clearError(): void {
    this.hideError()
  }

  /**
   * Set symbol change callback (placeholder for compatibility)
   */
  public onSymbolChanged(_callback: (symbol: string) => void): void {
    // This would be implemented if we had symbol selection controls
  }

  /**
   * Set interval change callback (placeholder for compatibility)
   */
  public onIntervalChanged(_callback: (interval: string) => void): void {
    // This would be implemented if we had interval selection controls
  }

  /**
   * Set chart fullscreen callback (placeholder for compatibility)
   */
  public onChartFullscreenToggle(_callback: () => void): void {
    // This would be implemented if we had additional fullscreen handling
  }

  /**
   * Get current dashboard state
   */
  public getState(): DashboardState {
    return { ...this.state }
  }

  /**
   * Update indicators display
   */
  public updateIndicators(indicators: VuManchuIndicators): void {
    this.state.indicators = indicators
    // Update any indicator-specific UI elements if needed
    this.updateLastUpdateTime()
  }

  /**
   * Toggle performance charts visibility
   */
  private togglePerformanceCharts(): void {
    this.performanceChartsVisible = !this.performanceChartsVisible

    const modal = document.getElementById('performance-modal')
    if (!modal) {
      this.createPerformanceModal()
      return
    }

    if (this.performanceChartsVisible) {
      this.showPerformanceCharts()
    } else {
      this.hidePerformanceCharts()
    }
  }

  /**
   * Create performance charts modal
   */
  private createPerformanceModal(): void {
    const modal = document.createElement('div')
    modal.id = 'performance-modal'
    modal.className = 'performance-modal'
    modal.setAttribute('data-modal', 'hidden')
    modal.innerHTML = `
      <div class="modal-overlay" data-performance-overlay=""></div>
      <div class="modal-content performance-content">
        <div class="modal-header">
          <h2 class="modal-title">Performance Analytics</h2>
          <button class="modal-close" data-performance-close="" aria-label="Close performance charts">&times;</button>
        </div>
        <div class="modal-body performance-body">
          <div id="performance-charts-container"></div>
        </div>
      </div>
    `

    document.getElementById('app')?.appendChild(modal)

    // Add event listeners
    const closeBtn = modal.querySelector('[data-performance-close]')
    const overlay = modal.querySelector('[data-performance-overlay]')

    closeBtn?.addEventListener('click', () => this.hidePerformanceCharts())
    overlay?.addEventListener('click', () => this.hidePerformanceCharts())

    // Show the modal
    this.showPerformanceCharts()
  }

  /**
   * Show performance charts
   */
  private showPerformanceCharts(): void {
    const modal = document.getElementById('performance-modal')
    const container = document.getElementById('performance-charts-container')

    if (!modal || !container) return

    modal.setAttribute('data-modal', 'visible')
    document.body.style.overflow = 'hidden'

    // Initialize charts if not already done
    if (!this.performanceCharts) {
      this.performanceCharts = new PerformanceCharts(container)

      // Update with current data
      if (this.state.positions && this.state.risk_metrics) {
        this.performanceCharts.updateData(this.state.positions, this.state.risk_metrics)
      }
    }

    this.performanceChartsVisible = true
    this.log('info', 'Performance charts opened', 'UI')
  }

  /**
   * Hide performance charts
   */
  private hidePerformanceCharts(): void {
    const modal = document.getElementById('performance-modal')

    if (modal) {
      modal.setAttribute('data-modal', 'hidden')
      document.body.style.overflow = ''
    }

    this.performanceChartsVisible = false
    this.log('info', 'Performance charts closed', 'UI')
  }

  /**
   * Batch update multiple components for better performance
   */
  public batchUpdate(updates: {
    botStatus?: BotStatus
    marketData?: MarketData
    tradeAction?: TradeAction
    positions?: Position[]
    riskMetrics?: RiskMetrics
    indicators?: VuManchuIndicators
  }): void {
    // Use requestAnimationFrame to batch DOM updates
    requestAnimationFrame(() => {
      if (updates.botStatus) this.updateBotStatus(updates.botStatus)
      if (updates.marketData) this.updateMarketData(updates.marketData)
      if (updates.tradeAction) this.updateLatestAction(updates.tradeAction)
      if (updates.positions) this.updatePositions(updates.positions)
      if (updates.riskMetrics) this.updateRiskMetrics(updates.riskMetrics)
      if (updates.indicators) this.updateIndicators(updates.indicators)
    })
  }

  // Error Handling and Resilience Methods

  /**
   * Execute function within error boundary
   */
  private withErrorBoundary<T>(operation: string, fn: () => T, fallbackData?: any): T | undefined {
    try {
      const result = fn()
      this.resetErrorBoundary(operation)
      return result
    } catch (error) {
      return this.handleBoundaryError(operation, error, fallbackData)
    }
  }

  /**
   * Handle error within boundary
   */
  private handleBoundaryError(operation: string, error: any, fallbackData?: any): undefined {
    const boundary = this.errorBoundaries.get(operation) || {
      count: 0,
      lastError: new Date(),
      maxErrors: this.maxErrorsPerBoundary,
    }

    boundary.count++
    boundary.lastError = new Date()
    this.errorBoundaries.set(operation, boundary)

    const errorMsg = error instanceof Error ? error.message : String(error)
    this.log('error', `Error in ${operation}: ${errorMsg}`, 'UI')

    // Show user-friendly error if too many failures
    if (boundary.count >= boundary.maxErrors) {
      this.showUserFriendlyError(operation, boundary.count)
    }

    // Attempt recovery with fallback data
    if (fallbackData && boundary.count < boundary.maxErrors) {
      this.attemptRecovery(operation, fallbackData)
    }

    return undefined
  }

  /**
   * Reset error boundary after successful operation
   */
  private resetErrorBoundary(operation: string): void {
    const boundary = this.errorBoundaries.get(operation)
    if (boundary && boundary.count > 0) {
      // Only reset if enough time has passed
      const timeSinceLastError = Date.now() - boundary.lastError.getTime()
      if (timeSinceLastError > this.errorResetTime) {
        this.errorBoundaries.delete(operation)
      }
    }
  }

  /**
   * Show user-friendly error message
   */
  private showUserFriendlyError(operation: string, _errorCount: number): void {
    const messages: Record<string, string> = {
      updateConnectionStatus: 'Connection status updates are experiencing issues',
      updateBotStatus: 'Bot status information is temporarily unavailable',
      updateMarketData: 'Market data updates are experiencing delays',
      updateLatestAction: 'Trade action updates are temporarily unavailable',
      updatePositions: 'Position data is temporarily unavailable',
      updateRiskMetrics: 'Risk metrics are temporarily unavailable',
    }

    const message = messages[operation] || `${operation} is experiencing issues`
    this.showNotification('warning', `${message}. Using cached data where possible.`)
  }

  /**
   * Attempt recovery with fallback data
   */
  private attemptRecovery(operation: string, fallbackData: any): void {
    this.log('info', `Attempting recovery for ${operation} with fallback data`, 'UI')

    // Schedule retry with exponential backoff
    const retryAttempts = this.getRetryAttempts(operation)
    const delay =
      this.defaultRetryDelays[Math.min(retryAttempts, this.defaultRetryDelays.length - 1)]

    const timeoutId = window.setTimeout(() => {
      this.retryOperation(operation, fallbackData)
    }, delay)

    this.retryTimeouts.set(operation, timeoutId)
  }

  /**
   * Retry failed operation
   */
  private retryOperation(operation: string, data: any): void {
    this.log('info', `Retrying operation: ${operation}`, 'UI')

    try {
      switch (operation) {
        case 'updateBotStatus':
          this.updateBotStatus(data)
          break
        case 'updateMarketData':
          this.updateMarketData(data)
          break
        case 'updateLatestAction':
          this.updateLatestAction(data)
          break
        case 'updatePositions':
          this.updatePositions(data)
          break
        case 'updateRiskMetrics':
          this.updateRiskMetrics(data)
          break
      }
    } catch (error) {
      this.log('warn', `Retry failed for ${operation}: ${error}`, 'UI')
    }
  }

  /**
   * Get retry attempts for operation
   */
  private getRetryAttempts(operation: string): number {
    const boundary = this.errorBoundaries.get(operation)
    return boundary ? boundary.count - 1 : 0
  }

  /**
   * Show notification to user
   */
  private showNotification(type: 'info' | 'warning' | 'error', message: string): void {
    // Create or update notification area
    let notificationArea = document.getElementById('notification-area')
    if (!notificationArea) {
      notificationArea = document.createElement('div')
      notificationArea.id = 'notification-area'
      notificationArea.className = 'notification-area'
      document.body.appendChild(notificationArea)
    }

    const notification = document.createElement('div')
    notification.className = `notification notification-${type}`
    notification.innerHTML = `
      <span class="notification-message">${message}</span>
      <button class="notification-close" aria-label="Close notification">&times;</button>
    `

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove()
      }
    }, 5000)

    // Close button functionality
    const closeBtn = notification.querySelector('.notification-close')
    closeBtn?.addEventListener('click', () => {
      notification.remove()
    })

    notificationArea.appendChild(notification)
  }

  /**
   * Enter offline mode with fallback UI
   */
  private enterOfflineMode(): void {
    this.log('warn', 'Entering offline mode - using cached data', 'UI')
    this.showNotification('warning', 'Connection lost. Displaying cached data.')

    // Show offline indicator
    this.updateOfflineIndicator()

    // Use fallback data for all components
    this.loadFallbackData()
  }

  /**
   * Exit offline mode
   */
  private exitOfflineMode(): void {
    this.log('info', 'Exiting offline mode - live data restored', 'UI')
    this.showNotification('info', 'Connection restored. Live data available.')
    this.updateOfflineIndicator()
  }

  /**
   * Update offline indicator
   */
  private updateOfflineIndicator(): void {
    let indicator = document.getElementById('offline-indicator')
    if (!indicator) {
      indicator = document.createElement('div')
      indicator.id = 'offline-indicator'
      indicator.className = 'offline-indicator'
      indicator.innerHTML = '<span>⚠️ Offline Mode - Cached Data</span>'
      document.body.appendChild(indicator)
    }

    indicator.style.display = this.isOfflineMode ? 'block' : 'none'
  }

  /**
   * Load fallback data for all components
   */
  private loadFallbackData(): void {
    for (const [key, data] of this.fallbackData.entries()) {
      switch (key) {
        case 'bot_status':
          this.updateBotStatus(data)
          break
        case 'market_data':
          this.updateMarketData(data)
          break
        case 'trade_action':
          this.updateLatestAction(data)
          break
        case 'positions':
          this.updatePositions(data)
          break
        case 'risk_metrics':
          this.updateRiskMetrics(data)
          break
      }
    }
  }

  /**
   * Check if data is stale and show warnings
   */
  private checkDataStaleness(dataType: string): void {
    const lastUpdate = this.lastDataUpdate.get(dataType)
    if (lastUpdate) {
      const age = Date.now() - lastUpdate.getTime()
      if (age > this.dataStaleThreshold) {
        this.showStaleDataWarning(dataType, Math.floor(age / 1000))
      }
    }
  }

  /**
   * Show stale data warning
   */
  private showStaleDataWarning(dataType: string, ageSeconds: number): void {
    const humanReadableType = dataType.replace('_', ' ').toLowerCase()
    const ageText = ageSeconds < 60 ? `${ageSeconds}s` : `${Math.floor(ageSeconds / 60)}m`
    this.showNotification('warning', `${humanReadableType} is ${ageText} old`)
  }

  /**
   * Show missing data warning
   */
  private showMissingDataWarning(dataType: string): void {
    let warningArea = document.getElementById('missing-data-warnings')
    if (!warningArea) {
      warningArea = document.createElement('div')
      warningArea.id = 'missing-data-warnings'
      warningArea.className = 'missing-data-warnings'
      document.body.appendChild(warningArea)
    }

    const warningId = `warning-${dataType.toLowerCase().replace(' ', '-')}`
    if (!document.getElementById(warningId)) {
      const warning = document.createElement('div')
      warning.id = warningId
      warning.className = 'missing-data-warning'
      warning.textContent = `${dataType} unavailable`
      warningArea.appendChild(warning)
    }
  }

  /**
   * Hide missing data warning
   */
  private hideMissingDataWarning(dataType: string): void {
    const warningId = `warning-${dataType.toLowerCase().replace(' ', '-')}`
    const warning = document.getElementById(warningId)
    if (warning) {
      warning.remove()
    }
  }

  /**
   * Show data fallback indicator
   */
  private showDataFallbackIndicator(dataType: string): void {
    this.log('info', `Using fallback data for ${dataType}`, 'UI')
    // Could add visual indicator in UI if needed
  }

  // Data Validation Methods

  /**
   * Validate market data structure
   */
  private validateMarketData(data: any): data is MarketData {
    return (
      data &&
      typeof data === 'object' &&
      typeof data.symbol === 'string' &&
      typeof data.price === 'number' &&
      !isNaN(data.price) &&
      data.price > 0
    )
  }

  /**
   * Validate trade action structure
   */
  private validateTradeAction(data: any): data is TradeAction {
    return (
      data &&
      typeof data === 'object' &&
      ['BUY', 'SELL', 'HOLD'].includes(data.action) &&
      typeof data.confidence === 'number' &&
      !isNaN(data.confidence) &&
      typeof data.reasoning === 'string'
    )
  }

  /**
   * Validate position structure
   */
  private validatePosition(data: any): data is Position {
    return (
      data &&
      typeof data === 'object' &&
      typeof data.symbol === 'string' &&
      ['long', 'short'].includes(data.side) &&
      typeof data.size === 'number' &&
      !isNaN(data.size) &&
      typeof data.entry_price === 'number' &&
      !isNaN(data.entry_price)
    )
  }

  /**
   * Validate risk metrics structure
   */
  private validateRiskMetrics(data: any): data is RiskMetrics {
    return (
      data &&
      typeof data === 'object' &&
      typeof data.total_portfolio_value === 'number' &&
      !isNaN(data.total_portfolio_value) &&
      typeof data.available_balance === 'number' &&
      !isNaN(data.available_balance)
    )
  }

  // Safe Data Access Methods

  /**
   * Safely convert value to number
   */
  private safeNumber(value: any, defaultValue: number = 0): number {
    if (typeof value === 'number' && !isNaN(value) && isFinite(value)) {
      return value
    }
    if (typeof value === 'string') {
      const parsed = parseFloat(value)
      if (!isNaN(parsed) && isFinite(parsed)) {
        return parsed
      }
    }
    return defaultValue
  }

  /**
   * Safely convert value to string
   */
  private safeString(value: any, defaultValue: string = ''): string {
    if (typeof value === 'string') {
      return value
    }
    if (value != null) {
      return String(value)
    }
    return defaultValue
  }
}
