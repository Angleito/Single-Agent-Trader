/**
 * Trading Performance Analytics Dashboard Component
 *
 * Provides comprehensive performance analysis with:
 * - Real-time performance metrics tracking
 * - Historical performance charts and trends
 * - Trade analytics and win/loss analysis
 * - Risk-adjusted returns and Sharpe ratio
 * - Drawdown analysis and recovery periods
 * - Strategy performance comparison
 * - Benchmark comparisons
 * - Performance attribution analysis
 */

// Removed unused imports

export interface PerformanceMetrics {
  total_return: number
  total_return_percentage: number
  annualized_return: number
  win_rate: number
  profit_factor: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  max_drawdown_duration: number // in days
  avg_trade_duration: number // in minutes
  avg_win: number
  avg_loss: number
  largest_win: number
  largest_loss: number
  consecutive_wins: number
  consecutive_losses: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  total_fees: number
  net_profit: number
  gross_profit: number
  gross_loss: number
  volatility: number
  calmar_ratio: number
  recovery_factor: number
}

export interface TradeAnalytics {
  trade_id: string
  symbol: string
  side: 'long' | 'short'
  entry_time: string
  exit_time?: string
  entry_price: number
  exit_price?: number
  quantity: number
  pnl: number
  pnl_percentage: number
  duration: number // in minutes
  fees: number
  strategy: string
  market_conditions: string
  confidence_score?: number
  tags: string[]
}

export interface PerformancePeriod {
  period: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly'
  start_date: string
  end_date: string
  return_pct: number
  trades_count: number
  win_rate: number
  max_drawdown: number
  sharpe_ratio: number
  best_trade: number
  worst_trade: number
}

export interface BenchmarkComparison {
  benchmark_name: string
  benchmark_return: number
  strategy_return: number
  alpha: number
  beta: number
  correlation: number
  tracking_error: number
  information_ratio: number
}

export interface DrawdownPeriod {
  start_date: string
  end_date?: string
  peak_value: number
  trough_value: number
  drawdown_pct: number
  duration_days: number
  recovery_date?: string
  recovery_duration?: number
  is_underwater: boolean
}

export class PerformanceAnalyticsDashboard {
  private container: HTMLElement
  private apiBaseUrl: string
  private performanceMetrics: PerformanceMetrics | null = null
  private tradeAnalytics: TradeAnalytics[] = []
  private performancePeriods: PerformancePeriod[] = []
  private benchmarkComparisons: BenchmarkComparison[] = []
  private drawdownPeriods: DrawdownPeriod[] = []
  private chartCanvas: HTMLCanvasElement | null = null
  private updateInterval: number | null = null
  private selectedTimeframe: string = '30d'
  private selectedMetric: string = 'cumulative_return'
  private chartType: string = 'equity_curve'

  constructor(containerId: string, apiBaseUrl: string) {
    const container = document.getElementById(containerId)
    if (!container) {
      throw new Error(`Container element with ID ${containerId} not found`)
    }

    this.container = container
    this.apiBaseUrl = apiBaseUrl
    this.render()
    void this.loadPerformanceData()
    this.startRealtimeUpdates()
  }

  /**
   * Update performance metrics
   */
  public updatePerformanceMetrics(metrics: PerformanceMetrics): void {
    this.performanceMetrics = metrics
    this.updateMetricsDisplay()
  }

  /**
   * Add new trade analytics
   */
  public addTradeAnalytics(trade: TradeAnalytics): void {
    this.tradeAnalytics.unshift(trade)
    // Keep only last 1000 trades for performance
    if (this.tradeAnalytics.length > 1000) {
      this.tradeAnalytics = this.tradeAnalytics.slice(0, 1000)
    }
    this.updateTradeAnalyticsDisplay()
  }

  /**
   * Render the main interface
   */
  private render(): void {
    this.container.innerHTML = `
      <div class="performance-analytics-dashboard">
        <!-- Header -->
        <div class="analytics-header">
          <div class="header-title">
            <h3>Performance Analytics</h3>
            <div class="performance-summary">
              <span class="summary-item">
                <span class="label">Total Return:</span>
                <span class="value" id="total-return">--</span>
              </span>
              <span class="summary-item">
                <span class="label">Win Rate:</span>
                <span class="value" id="win-rate">--</span>
              </span>
              <span class="summary-item">
                <span class="label">Sharpe Ratio:</span>
                <span class="value" id="sharpe-ratio">--</span>
              </span>
            </div>
          </div>
          <div class="header-controls">
            <select id="timeframe-selector" class="form-control">
              <option value="1d">1 Day</option>
              <option value="7d">7 Days</option>
              <option value="30d" selected>30 Days</option>
              <option value="90d">90 Days</option>
              <option value="1y">1 Year</option>
              <option value="all">All Time</option>
            </select>
            <button class="export-btn" id="export-analytics">📊 Export Report</button>
            <button class="refresh-btn" id="refresh-analytics">🔄 Refresh</button>
          </div>
        </div>

        <!-- Key Metrics Grid -->
        <div class="metrics-grid">
          <div class="metric-card primary">
            <div class="metric-header">
              <h4>Returns</h4>
              <span class="metric-trend" id="returns-trend">--</span>
            </div>
            <div class="metric-values">
              <div class="metric-row">
                <span class="label">Total Return</span>
                <span class="value" id="total-return-detailed">$0.00</span>
              </div>
              <div class="metric-row">
                <span class="label">Return %</span>
                <span class="value" id="total-return-pct">0.00%</span>
              </div>
              <div class="metric-row">
                <span class="label">Annualized</span>
                <span class="value" id="annualized-return">0.00%</span>
              </div>
            </div>
          </div>

          <div class="metric-card">
            <div class="metric-header">
              <h4>Risk Metrics</h4>
              <span class="metric-trend" id="risk-trend">--</span>
            </div>
            <div class="metric-values">
              <div class="metric-row">
                <span class="label">Max Drawdown</span>
                <span class="value negative" id="max-drawdown-detailed">0.00%</span>
              </div>
              <div class="metric-row">
                <span class="label">Volatility</span>
                <span class="value" id="volatility">0.00%</span>
              </div>
              <div class="metric-row">
                <span class="label">Sharpe</span>
                <span class="value" id="sharpe-detailed">0.00</span>
              </div>
            </div>
          </div>

          <div class="metric-card">
            <div class="metric-header">
              <h4>Trade Statistics</h4>
              <span class="metric-trend" id="trades-trend">--</span>
            </div>
            <div class="metric-values">
              <div class="metric-row">
                <span class="label">Total Trades</span>
                <span class="value" id="total-trades">0</span>
              </div>
              <div class="metric-row">
                <span class="label">Win Rate</span>
                <span class="value" id="win-rate-detailed">0.00%</span>
              </div>
              <div class="metric-row">
                <span class="label">Profit Factor</span>
                <span class="value" id="profit-factor">0.00</span>
              </div>
            </div>
          </div>

          <div class="metric-card">
            <div class="metric-header">
              <h4>Efficiency</h4>
              <span class="metric-trend" id="efficiency-trend">--</span>
            </div>
            <div class="metric-values">
              <div class="metric-row">
                <span class="label">Avg Win</span>
                <span class="value positive" id="avg-win">$0.00</span>
              </div>
              <div class="metric-row">
                <span class="label">Avg Loss</span>
                <span class="value negative" id="avg-loss">$0.00</span>
              </div>
              <div class="metric-row">
                <span class="label">Recovery Factor</span>
                <span class="value" id="recovery-factor">0.00</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Charts Section -->
        <div class="charts-section">
          <div class="chart-container">
            <div class="chart-header">
              <h4>Performance Charts</h4>
              <div class="chart-controls">
                <select id="chart-type-selector" class="form-control">
                  <option value="equity_curve">Equity Curve</option>
                  <option value="drawdown">Drawdown Chart</option>
                  <option value="monthly_returns">Monthly Returns</option>
                  <option value="rolling_sharpe">Rolling Sharpe</option>
                  <option value="underwater">Underwater Chart</option>
                </select>
                <select id="chart-period" class="form-control">
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
                <button class="fullscreen-btn" id="fullscreen-chart">⛶</button>
              </div>
            </div>
            <div class="chart-canvas-container">
              <canvas id="performance-chart" class="performance-chart"></canvas>
              <div class="chart-overlay" id="chart-overlay"></div>
            </div>
            <div class="chart-legend" id="chart-legend">
              <!-- Chart legend will be populated here -->
            </div>
          </div>
        </div>

        <!-- Analytics Tabs -->
        <div class="analytics-content">
          <div class="content-tabs">
            <button class="tab-btn active" data-tab="trade-analysis">Trade Analysis</button>
            <button class="tab-btn" data-tab="period-performance">Period Performance</button>
            <button class="tab-btn" data-tab="drawdown-analysis">Drawdown Analysis</button>
            <button class="tab-btn" data-tab="benchmark-comparison">Benchmark Comparison</button>
          </div>

          <!-- Trade Analysis Tab -->
          <div class="tab-content active" id="trade-analysis-tab">
            <div class="trade-analysis-section">
              <div class="analysis-header">
                <h5>Trade Analytics</h5>
                <div class="analysis-controls">
                  <select id="trade-filter" class="form-control">
                    <option value="all">All Trades</option>
                    <option value="winning">Winning Trades</option>
                    <option value="losing">Losing Trades</option>
                    <option value="long">Long Trades</option>
                    <option value="short">Short Trades</option>
                  </select>
                  <select id="trade-sort" class="form-control">
                    <option value="recent">Most Recent</option>
                    <option value="pnl-desc">Highest P&L</option>
                    <option value="pnl-asc">Lowest P&L</option>
                    <option value="duration-desc">Longest Duration</option>
                    <option value="duration-asc">Shortest Duration</option>
                  </select>
                </div>
              </div>
              
              <div class="trade-stats-grid">
                <div class="stat-item">
                  <span class="stat-label">Best Trade</span>
                  <span class="stat-value positive" id="best-trade">$0.00</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Worst Trade</span>
                  <span class="stat-value negative" id="worst-trade">$0.00</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Avg Duration</span>
                  <span class="stat-value" id="avg-duration">0h 0m</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Consecutive Wins</span>
                  <span class="stat-value" id="consecutive-wins">0</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Consecutive Losses</span>
                  <span class="stat-value" id="consecutive-losses">0</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Total Fees</span>
                  <span class="stat-value" id="total-fees">$0.00</span>
                </div>
              </div>

              <div class="trades-table-container">
                <table class="trades-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Entry Time</th>
                      <th>Exit Time</th>
                      <th>P&L</th>
                      <th>P&L %</th>
                      <th>Duration</th>
                      <th>Strategy</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody id="trades-table-body">
                    <tr class="empty-state">
                      <td colspan="9">No trades to display</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Period Performance Tab -->
          <div class="tab-content" id="period-performance-tab">
            <div class="period-performance-section">
              <div class="periods-grid" id="periods-grid">
                <!-- Period performance cards will be populated here -->
              </div>
            </div>
          </div>

          <!-- Drawdown Analysis Tab -->
          <div class="tab-content" id="drawdown-analysis-tab">
            <div class="drawdown-analysis-section">
              <div class="drawdown-stats">
                <h5>Drawdown Statistics</h5>
                <div class="drawdown-summary">
                  <div class="summary-item">
                    <span class="label">Current Drawdown:</span>
                    <span class="value negative" id="current-drawdown">0.00%</span>
                  </div>
                  <div class="summary-item">
                    <span class="label">Days Underwater:</span>
                    <span class="value" id="days-underwater">0</span>
                  </div>
                  <div class="summary-item">
                    <span class="label">Max DD Duration:</span>
                    <span class="value" id="max-dd-duration">0 days</span>
                  </div>
                </div>
              </div>
              
              <div class="drawdown-periods-list" id="drawdown-periods-list">
                <!-- Drawdown periods will be populated here -->
              </div>
            </div>
          </div>

          <!-- Benchmark Comparison Tab -->
          <div class="tab-content" id="benchmark-comparison-tab">
            <div class="benchmark-comparison-section">
              <div class="benchmark-grid" id="benchmark-grid">
                <!-- Benchmark comparisons will be populated here -->
              </div>
            </div>
          </div>
        </div>
      </div>
    `

    this.attachEventListeners()
    this.initializeChart()
  }

  /**
   * Attach event listeners
   */
  private attachEventListeners(): void {
    // Header controls
    const timeframeSelector = document.getElementById('timeframe-selector')
    const exportBtn = document.getElementById('export-analytics')
    const refreshBtn = document.getElementById('refresh-analytics')

    timeframeSelector?.addEventListener('change', (e) => {
      this.selectedTimeframe = (e.target as HTMLSelectElement).value
      this.loadPerformanceData()
    })

    exportBtn?.addEventListener('click', () => this.exportAnalyticsReport())
    refreshBtn?.addEventListener('click', () => this.loadPerformanceData())

    // Chart controls
    const chartTypeSelector = document.getElementById('chart-type-selector')
    const chartPeriodSelector = document.getElementById('chart-period')
    const fullscreenBtn = document.getElementById('fullscreen-chart')

    chartTypeSelector?.addEventListener('change', (e) => {
      this.chartType = (e.target as HTMLSelectElement).value
      this.updateChart()
    })

    chartPeriodSelector?.addEventListener('change', () => this.updateChart())
    fullscreenBtn?.addEventListener('click', () => this.toggleChartFullscreen())

    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn')
    tabBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const tabName = (e.target as HTMLElement).dataset.tab
        if (tabName) this.switchTab(tabName)
      })
    })

    // Trade analysis controls
    const tradeFilter = document.getElementById('trade-filter')
    const tradeSort = document.getElementById('trade-sort')

    tradeFilter?.addEventListener('change', () => this.filterTrades())
    tradeSort?.addEventListener('change', () => this.sortTrades())
  }

  /**
   * Initialize chart canvas
   */
  private initializeChart(): void {
    const canvas = document.getElementById('performance-chart') as HTMLCanvasElement
    if (!canvas) return

    this.chartCanvas = canvas
    this.resizeChart()

    // Add resize listener
    window.addEventListener('resize', () => this.resizeChart())
  }

  /**
   * Resize chart canvas
   */
  private resizeChart(): void {
    if (!this.chartCanvas) return

    const container = this.chartCanvas.parentElement
    if (!container) return

    const rect = container.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    this.chartCanvas.width = rect.width * dpr
    this.chartCanvas.height = 400 * dpr
    this.chartCanvas.style.width = rect.width + 'px'
    this.chartCanvas.style.height = '400px'

    const ctx = this.chartCanvas.getContext('2d')
    if (ctx) {
      ctx.scale(dpr, dpr)
    }

    this.updateChart()
  }

  /**
   * Load performance data from API
   */
  private async loadPerformanceData(): Promise<void> {
    try {
      const response = await fetch(
        `${this.apiBaseUrl}/api/bot/analytics/performance?timeframe=${this.selectedTimeframe}`
      )
      if (response.ok) {
        const data = await response.json()
        this.performanceMetrics = data.metrics
        this.tradeAnalytics = data.trades || []
        this.performancePeriods = data.periods || []
        this.benchmarkComparisons = data.benchmarks || []
        this.drawdownPeriods = data.drawdowns || []

        this.updateMetricsDisplay()
        this.updateTradeAnalyticsDisplay()
        this.updateChart()
      }
    } catch (error) {
      console.warn('Failed to load performance data:', error)
      this.loadMockData()
    }
  }

  /**
   * Load mock data for development
   */
  private loadMockData(): void {
    this.performanceMetrics = {
      total_return: 2543.67,
      total_return_percentage: 15.24,
      annualized_return: 18.45,
      win_rate: 62.5,
      profit_factor: 1.85,
      sharpe_ratio: 1.42,
      sortino_ratio: 1.67,
      max_drawdown: -8.33,
      max_drawdown_duration: 12,
      avg_trade_duration: 125,
      avg_win: 145.23,
      avg_loss: -78.45,
      largest_win: 567.89,
      largest_loss: -234.56,
      consecutive_wins: 8,
      consecutive_losses: 4,
      total_trades: 96,
      winning_trades: 60,
      losing_trades: 36,
      total_fees: 234.56,
      net_profit: 2309.11,
      gross_profit: 2543.67,
      gross_loss: -1234.56,
      volatility: 12.34,
      calmar_ratio: 2.21,
      recovery_factor: 3.85,
    }

    this.updateMetricsDisplay()
    this.updateChart()
  }

  /**
   * Update metrics display
   */
  private updateMetricsDisplay(): void {
    if (!this.performanceMetrics) return

    const metrics = this.performanceMetrics

    // Update summary
    this.updateElement('total-return', `$${metrics.total_return.toFixed(2)}`)
    this.updateElement('win-rate', `${metrics.win_rate.toFixed(1)}%`)
    this.updateElement('sharpe-ratio', metrics.sharpe_ratio.toFixed(2))

    // Update detailed metrics
    this.updateElement('total-return-detailed', `$${metrics.total_return.toFixed(2)}`)
    this.updateElement('total-return-pct', `${metrics.total_return_percentage.toFixed(2)}%`)
    this.updateElement('annualized-return', `${metrics.annualized_return.toFixed(2)}%`)

    this.updateElement('max-drawdown-detailed', `${metrics.max_drawdown.toFixed(2)}%`)
    this.updateElement('volatility', `${metrics.volatility.toFixed(2)}%`)
    this.updateElement('sharpe-detailed', metrics.sharpe_ratio.toFixed(2))

    this.updateElement('total-trades', metrics.total_trades.toString())
    this.updateElement('win-rate-detailed', `${metrics.win_rate.toFixed(1)}%`)
    this.updateElement('profit-factor', metrics.profit_factor.toFixed(2))

    this.updateElement('avg-win', `$${metrics.avg_win.toFixed(2)}`)
    this.updateElement('avg-loss', `$${Math.abs(metrics.avg_loss).toFixed(2)}`)
    this.updateElement('recovery-factor', metrics.recovery_factor.toFixed(2))

    // Update trade stats
    this.updateElement('best-trade', `$${metrics.largest_win.toFixed(2)}`)
    this.updateElement('worst-trade', `$${metrics.largest_loss.toFixed(2)}`)
    this.updateElement('avg-duration', this.formatDuration(metrics.avg_trade_duration))
    this.updateElement('consecutive-wins', metrics.consecutive_wins.toString())
    this.updateElement('consecutive-losses', metrics.consecutive_losses.toString())
    this.updateElement('total-fees', `$${metrics.total_fees.toFixed(2)}`)

    // Update trends
    this.updateTrendIndicators()
  }

  /**
   * Update trend indicators
   */
  private updateTrendIndicators(): void {
    // Simplified trend calculation - in reality this would compare to previous periods
    const trends: Record<string, 'up' | 'down'> = {
      returns: this.performanceMetrics!.total_return_percentage > 0 ? 'up' : 'down',
      risk: this.performanceMetrics!.max_drawdown > -10 ? 'up' : 'down',
      trades: this.performanceMetrics!.win_rate > 50 ? 'up' : 'down',
      efficiency: this.performanceMetrics!.profit_factor > 1 ? 'up' : 'down',
    }

    this.updateTrendElement('returns-trend', trends.returns)
    this.updateTrendElement('risk-trend', trends.risk)
    this.updateTrendElement('trades-trend', trends.trades)
    this.updateTrendElement('efficiency-trend', trends.efficiency)
  }

  /**
   * Update trend element
   */
  private updateTrendElement(id: string, trend: 'up' | 'down'): void {
    const element = document.getElementById(id)
    if (!element) return

    element.className = `metric-trend trend-${trend}`
    element.textContent = trend === 'up' ? '↗' : '↘'
  }

  /**
   * Update trade analytics display
   */
  private updateTradeAnalyticsDisplay(): void {
    const tableBody = document.getElementById('trades-table-body')
    if (!tableBody) return

    if (this.tradeAnalytics.length === 0) {
      tableBody.innerHTML = '<tr class="empty-state"><td colspan="9">No trades to display</td></tr>'
      return
    }

    tableBody.innerHTML = this.tradeAnalytics
      .slice(0, 50)
      .map(
        (trade) => `
      <tr class="trade-row ${trade.pnl >= 0 ? 'winning' : 'losing'}">
        <td class="symbol-cell">${trade.symbol}</td>
        <td class="side-cell">
          <span class="side-badge side-${trade.side}">${trade.side.toUpperCase()}</span>
        </td>
        <td class="time-cell">${this.formatDateTime(trade.entry_time)}</td>
        <td class="time-cell">${trade.exit_time ? this.formatDateTime(trade.exit_time) : 'Open'}</td>
        <td class="pnl-cell ${trade.pnl >= 0 ? 'positive' : 'negative'}">
          $${trade.pnl.toFixed(2)}
        </td>
        <td class="pnl-pct-cell ${trade.pnl_percentage >= 0 ? 'positive' : 'negative'}">
          ${trade.pnl_percentage.toFixed(2)}%
        </td>
        <td class="duration-cell">${this.formatDuration(trade.duration)}</td>
        <td class="strategy-cell">${trade.strategy}</td>
        <td class="actions-cell">
          <button class="action-btn view-btn" onclick="this.viewTradeDetails('${trade.trade_id}')">👁</button>
        </td>
      </tr>
    `
      )
      .join('')
  }

  /**
   * Update chart based on selected type
   */
  private updateChart(): void {
    if (!this.chartCanvas) return

    const ctx = this.chartCanvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, this.chartCanvas.width, this.chartCanvas.height)

    switch (this.chartType) {
      case 'equity_curve':
        this.drawEquityCurve(ctx)
        break
      case 'drawdown':
        this.drawDrawdownChart(ctx)
        break
      case 'monthly_returns':
        this.drawMonthlyReturns(ctx)
        break
      case 'rolling_sharpe':
        this.drawRollingSharpe(ctx)
        break
      case 'underwater':
        this.drawUnderwaterChart(ctx)
        break
    }
  }

  /**
   * Draw equity curve chart
   */
  private drawEquityCurve(ctx: CanvasRenderingContext2D): void {
    const _width = ctx.canvas.width / (window.devicePixelRatio || 1)
    const _height = ctx.canvas.height / (window.devicePixelRatio || 1)
    const padding = { top: 20, right: 60, bottom: 40, left: 60 }

    // Generate sample equity curve data
    const data = this.generateEquityCurveData()

    if (data.length === 0) return

    // Calculate bounds
    const _xMin = 0
    const xMax = data.length - 1
    const yMin = Math.min(...data.map((d) => d.value))
    const yMax = Math.max(...data.map((d) => d.value))
    const yRange = yMax - yMin

    // Draw grid
    this.drawGrid(ctx, width, height, padding)

    // Draw equity curve
    ctx.strokeStyle = '#22c55e'
    ctx.lineWidth = 2
    ctx.beginPath()

    data.forEach((point, index) => {
      const x = padding.left + (index / xMax) * (width - padding.left - padding.right)
      const y =
        padding.top + (1 - (point.value - yMin) / yRange) * (height - padding.top - padding.bottom)

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()

    // Draw axes labels
    this.drawAxesLabels(ctx, width, height, padding, yMin, yMax)
  }

  /**
   * Draw drawdown chart
   */
  private drawDrawdownChart(ctx: CanvasRenderingContext2D): void {
    const _width = ctx.canvas.width / (window.devicePixelRatio || 1)
    const _height = ctx.canvas.height / (window.devicePixelRatio || 1)
    const padding = { top: 20, right: 60, bottom: 40, left: 60 }

    // Generate sample drawdown data
    const data = this.generateDrawdownData()

    if (data.length === 0) return

    // Calculate bounds
    const xMax = data.length - 1
    const yMin = Math.min(...data.map((d) => d.drawdown))
    const yMax = 0

    // Draw grid
    this.drawGrid(ctx, width, height, padding)

    // Fill drawdown area
    ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'
    ctx.beginPath()

    data.forEach((point, index) => {
      const x = padding.left + (index / xMax) * (width - padding.left - padding.right)
      const yTop = padding.top
      const _yBottom =
        padding.top +
        (Math.abs(point.drawdown) / Math.abs(yMin)) * (height - padding.top - padding.bottom)

      if (index === 0) {
        ctx.moveTo(x, yTop)
      } else {
        ctx.lineTo(x, yTop)
      }
    })

    data.reverse().forEach((point, index) => {
      const x = padding.left + ((xMax - index) / xMax) * (width - padding.left - padding.right)
      const y =
        padding.top +
        (Math.abs(point.drawdown) / Math.abs(yMin)) * (height - padding.top - padding.bottom)
      ctx.lineTo(x, y)
    })

    ctx.closePath()
    ctx.fill()

    // Draw axes labels
    this.drawAxesLabels(ctx, width, height, padding, yMin, yMax)
  }

  /**
   * Draw monthly returns heatmap
   */
  private drawMonthlyReturns(ctx: CanvasRenderingContext2D): void {
    // Implementation for monthly returns heatmap
    ctx.fillStyle = '#9ca3af'
    ctx.font = '14px Inter'
    ctx.textAlign = 'center'
    ctx.fillText('Monthly Returns Chart', ctx.canvas.width / 2, ctx.canvas.height / 2)
  }

  /**
   * Draw rolling Sharpe ratio
   */
  private drawRollingSharpe(ctx: CanvasRenderingContext2D): void {
    // Implementation for rolling Sharpe ratio
    ctx.fillStyle = '#9ca3af'
    ctx.font = '14px Inter'
    ctx.textAlign = 'center'
    ctx.fillText('Rolling Sharpe Ratio Chart', ctx.canvas.width / 2, ctx.canvas.height / 2)
  }

  /**
   * Draw underwater chart
   */
  private drawUnderwaterChart(ctx: CanvasRenderingContext2D): void {
    // Implementation for underwater chart
    ctx.fillStyle = '#9ca3af'
    ctx.font = '14px Inter'
    ctx.textAlign = 'center'
    ctx.fillText('Underwater Chart', ctx.canvas.width / 2, ctx.canvas.height / 2)
  }

  /**
   * Draw grid
   */
  private drawGrid(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    padding: any
  ): void {
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * (height - padding.top - padding.bottom)
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (i / 10) * (width - padding.left - padding.right)
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, height - padding.bottom)
      ctx.stroke()
    }
  }

  /**
   * Draw axes labels
   */
  private drawAxesLabels(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    padding: any,
    yMin: number,
    yMax: number
  ): void {
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px Inter'
    ctx.textAlign = 'right'

    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * (height - padding.top - padding.bottom)
      const value = yMax - (i / 5) * (yMax - yMin)
      ctx.fillText(value.toFixed(0), padding.left - 10, y + 4)
    }

    // X-axis labels (simplified)
    ctx.textAlign = 'center'
    const labels = ['30d ago', '20d ago', '10d ago', 'Today']
    labels.forEach((label, index) => {
      const x =
        padding.left + (index / (labels.length - 1)) * (width - padding.left - padding.right)
      ctx.fillText(label, x, height - padding.bottom + 20)
    })
  }

  /**
   * Generate sample equity curve data
   */
  private generateEquityCurveData(): Array<{ timestamp: number; value: number }> {
    const data = []
    let value = 10000
    const now = Date.now()

    for (let i = 0; i < 30; i++) {
      value += (Math.random() - 0.45) * 100 // Slight upward bias
      data.push({
        timestamp: now - (29 - i) * 24 * 60 * 60 * 1000,
        value: value,
      })
    }

    return data
  }

  /**
   * Generate sample drawdown data
   */
  private generateDrawdownData(): Array<{ timestamp: number; drawdown: number }> {
    const data = []
    let peak = 10000
    let current = 10000
    const now = Date.now()

    for (let i = 0; i < 30; i++) {
      current += (Math.random() - 0.45) * 100
      if (current > peak) peak = current

      const drawdown = ((current - peak) / peak) * 100
      data.push({
        timestamp: now - (29 - i) * 24 * 60 * 60 * 1000,
        drawdown: drawdown,
      })
    }

    return data
  }

  /**
   * Tab management
   */
  private switchTab(tabName: string): void {
    // Update tab buttons
    const tabBtns = document.querySelectorAll('.tab-btn')
    tabBtns.forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabName)
    })

    // Update tab content
    const tabContents = document.querySelectorAll('.tab-content')
    tabContents.forEach((content) => {
      content.classList.toggle('active', content.id === `${tabName}-tab`)
    })

    // Load tab-specific data
    this.loadTabData(tabName)
  }

  /**
   * Load tab-specific data
   */
  private loadTabData(tabName: string): void {
    switch (tabName) {
      case 'period-performance':
        this.updatePeriodPerformanceDisplay()
        break
      case 'drawdown-analysis':
        this.updateDrawdownAnalysisDisplay()
        break
      case 'benchmark-comparison':
        this.updateBenchmarkComparisonDisplay()
        break
    }
  }

  /**
   * Update period performance display
   */
  private updatePeriodPerformanceDisplay(): void {
    const periodsGrid = document.getElementById('periods-grid')
    if (!periodsGrid) return

    // Mock period data
    const periods = [
      { period: 'This Month', return: 8.45, trades: 24, winRate: 65.5 },
      { period: 'Last Month', return: -2.34, trades: 28, winRate: 52.8 },
      { period: 'This Quarter', return: 15.67, trades: 72, winRate: 62.5 },
      { period: 'Last Quarter', return: 12.34, trades: 68, winRate: 58.9 },
    ]

    periodsGrid.innerHTML = periods
      .map(
        (period) => `
      <div class="period-card">
        <h6>${period.period}</h6>
        <div class="period-metrics">
          <div class="metric">
            <span class="label">Return</span>
            <span class="value ${period.return >= 0 ? 'positive' : 'negative'}">
              ${period.return.toFixed(2)}%
            </span>
          </div>
          <div class="metric">
            <span class="label">Trades</span>
            <span class="value">${period.trades}</span>
          </div>
          <div class="metric">
            <span class="label">Win Rate</span>
            <span class="value">${period.winRate.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    `
      )
      .join('')
  }

  /**
   * Update drawdown analysis display
   */
  private updateDrawdownAnalysisDisplay(): void {
    if (!this.performanceMetrics) return

    this.updateElement('current-drawdown', `${this.performanceMetrics.max_drawdown.toFixed(2)}%`)
    this.updateElement('days-underwater', '0') // Would be calculated from actual data
    this.updateElement('max-dd-duration', `${this.performanceMetrics.max_drawdown_duration} days`)

    const periodsList = document.getElementById('drawdown-periods-list')
    if (!periodsList) return

    periodsList.innerHTML = '<div class="empty-state">No significant drawdown periods</div>'
  }

  /**
   * Update benchmark comparison display
   */
  private updateBenchmarkComparisonDisplay(): void {
    const benchmarkGrid = document.getElementById('benchmark-grid')
    if (!benchmarkGrid) return

    benchmarkGrid.innerHTML =
      '<div class="empty-state">Benchmark comparison data not available</div>'
  }

  /**
   * Utility methods
   */
  private filterTrades(): void {
    // Implementation for filtering trades
    this.updateTradeAnalyticsDisplay()
  }

  private sortTrades(): void {
    // Implementation for sorting trades
    this.updateTradeAnalyticsDisplay()
  }

  private toggleChartFullscreen(): void {
    // Implementation for fullscreen chart
    console.log('Toggle fullscreen chart')
  }

  private exportAnalyticsReport(): void {
    // Implementation for exporting analytics report
    console.log('Export analytics report')
  }

  private formatDuration(minutes: number): string {
    const hours = Math.floor(minutes / 60)
    const mins = minutes % 60
    return `${hours}h ${mins}m`
  }

  private formatDateTime(dateString: string): string {
    return new Date(dateString).toLocaleString()
  }

  private updateElement(id: string, value: string): void {
    const element = document.getElementById(id)
    if (element) element.textContent = value
  }

  private startRealtimeUpdates(): void {
    this.updateInterval = window.setInterval(() => {
      this.loadPerformanceData()
    }, 30000) // Update every 30 seconds
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
    }

    window.removeEventListener('resize', () => this.resizeChart())
  }
}
