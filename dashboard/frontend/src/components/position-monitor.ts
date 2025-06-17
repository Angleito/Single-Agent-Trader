/**
 * Real-time Position Monitoring Component with P&L Visualization
 *
 * Features:
 * - Live position tracking with real-time updates
 * - Interactive P&L charts and metrics
 * - Position risk analysis and alerts
 * - Historical performance tracking
 * - Multi-timeframe P&L analysis
 * - Position entry/exit visualization
 */

import type { Position, MarketData, TradingModeConfig } from '../types'

export interface PositionMetrics {
  unrealized_pnl: number
  realized_pnl: number
  total_pnl: number
  daily_pnl: number
  weekly_pnl: number
  monthly_pnl: number
  pnl_percentage: number
  max_profit: number
  max_loss: number
  duration: number // in minutes
  entry_price: number
  current_price: number
  quantity: number
  side: 'long' | 'short' | 'flat'
  leverage: number
  margin_used: number
  liquidation_price?: number
}

export interface PnLDataPoint {
  timestamp: number
  unrealized_pnl: number
  price: number
  cumulative_pnl: number
}

export interface PositionAlert {
  type: 'profit_target' | 'stop_loss' | 'liquidation_warning' | 'margin_call'
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  threshold: number
  current_value: number
}

export class PositionMonitor {
  private container: HTMLElement
  private currentPosition: Position | null = null
  private currentMarketData: MarketData | null = null
  private positionMetrics: PositionMetrics | null = null
  private pnlHistory: PnLDataPoint[] = []
  private alerts: PositionAlert[] = []
  private charts: { [key: string]: any } = {}
  private updateInterval: number | null = null
  private _onPositionAlert?: (alert: PositionAlert) => void
  private chartCanvas: HTMLCanvasElement | null = null
  private lastUpdate = 0
  private tradingModeConfig: TradingModeConfig | null = null

  // Chart configuration
  private readonly chartConfig = {
    width: 800,
    height: 300,
    padding: { top: 20, right: 60, bottom: 40, left: 60 },
    gridColor: '#333',
    axisColor: '#666',
    profitColor: '#22c55e',
    lossColor: '#ef4444',
    neutralColor: '#6b7280',
    lineWidth: 2,
    pointRadius: 3,
  }

  constructor(containerId: string) {
    const container = document.getElementById(containerId)
    if (!container) {
      throw new Error(`Container element with ID ${containerId} not found`)
    }

    this.container = container
    this.render()
    this.startRealtimeUpdates()
  }

  /**
   * Update position data
   */
  public updatePosition(position: Position): void {
    this.currentPosition = position
    this.updatePositionMetrics()
    this.updatePositionDisplay()
    this.checkAlerts()
  }

  /**
   * Update market data
   */
  public updateMarketData(marketData: MarketData): void {
    this.currentMarketData = marketData

    // Add new P&L data point
    if (this.currentPosition && this.positionMetrics) {
      this.addPnLDataPoint(marketData.price)
    }

    this.updatePositionMetrics()
    this.updateCharts()
    this.updateRealTimeMetrics()
  }

  /**
   * Set trading mode configuration
   */
  public setTradingModeConfig(config: TradingModeConfig): void {
    this.tradingModeConfig = config
    this.render() // Re-render to update UI based on trading mode
  }

  /**
   * Set alert handler
   */
  public onPositionAlert(callback: (alert: PositionAlert) => void): void {
    this._onPositionAlert = callback
  }

  /**
   * Render the main interface
   */
  private render(): void {
    this.container.innerHTML = `
      <div class="position-monitor">
        <!-- Header -->
        <div class="monitor-header">
          <h3>Position Monitor</h3>
          <div class="position-status" id="position-status">
            <span class="status-indicator flat"></span>
            <span class="status-text">No Position</span>
          </div>
        </div>

        <!-- Quick Metrics Grid -->
        <div class="metrics-grid">
          <div class="metric-card pnl-card">
            <div class="metric-header">
              <h4>Unrealized P&L</h4>
              <div class="metric-trend" id="pnl-trend">--</div>
            </div>
            <div class="metric-value" id="unrealized-pnl">$0.00</div>
            <div class="metric-details">
              <span id="pnl-percentage">0.00%</span>
              <span class="metric-separator">•</span>
              <span id="pnl-duration">--</span>
            </div>
          </div>

          <div class="metric-card entry-card">
            <div class="metric-header">
              <h4>Entry Details</h4>
            </div>
            <div class="metric-value" id="entry-price">$0.00</div>
            <div class="metric-details">
              <span id="position-size">0</span>
              <span class="metric-separator">•</span>
              <span id="leverage-info">1x</span>
            </div>
          </div>

          <div class="metric-card current-card">
            <div class="metric-header">
              <h4>Current Price</h4>
              <div class="price-change" id="price-change">--</div>
            </div>
            <div class="metric-value" id="current-price">$0.00</div>
            <div class="metric-details">
              <span id="price-distance">--</span>
              <span class="metric-separator">•</span>
              <span id="last-update">--</span>
            </div>
          </div>

          <div class="metric-card risk-card">
            <div class="metric-header">
              <h4>Risk Metrics</h4>
              <div class="risk-level" id="risk-level">Low</div>
            </div>
            <div class="metric-value" id="margin-used">$0.00</div>
            <div class="metric-details">
              <span id="liquidation-distance">--</span>
              <span class="metric-separator">•</span>
              <span id="max-loss">--</span>
            </div>
          </div>
        </div>

        <!-- P&L Chart -->
        <div class="chart-section">
          <div class="chart-header">
            <h4>Real-time P&L Chart</h4>
            <div class="chart-controls">
              <div class="timeframe-selector">
                <button class="timeframe-btn active" data-timeframe="1h">1H</button>
                <button class="timeframe-btn" data-timeframe="4h">4H</button>
                <button class="timeframe-btn" data-timeframe="1d">1D</button>
                <button class="timeframe-btn" data-timeframe="all">ALL</button>
              </div>
              <button class="chart-reset-btn" id="reset-chart">🔄 Reset</button>
            </div>
          </div>
          <div class="chart-container">
            <canvas id="pnl-chart" class="pnl-chart"></canvas>
            <div class="chart-overlay" id="chart-overlay">
              <div class="chart-crosshair" id="chart-crosshair"></div>
              <div class="chart-tooltip" id="chart-tooltip"></div>
            </div>
          </div>
          <div class="chart-legend">
            <div class="legend-item">
              <span class="legend-color profit"></span>
              <span>Profit Zone</span>
            </div>
            <div class="legend-item">
              <span class="legend-color loss"></span>
              <span>Loss Zone</span>
            </div>
            <div class="legend-item">
              <span class="legend-color entry"></span>
              <span>Entry Price</span>
            </div>
          </div>
        </div>

        <!-- Performance Summary -->
        <div class="performance-section">
          <h4>Position Performance</h4>
          <div class="performance-grid">
            <div class="perf-item">
              <span class="perf-label">Daily P&L</span>
              <span class="perf-value" id="daily-pnl">$0.00</span>
            </div>
            <div class="perf-item">
              <span class="perf-label">Weekly P&L</span>
              <span class="perf-value" id="weekly-pnl">$0.00</span>
            </div>
            <div class="perf-item">
              <span class="perf-label">Monthly P&L</span>
              <span class="perf-value" id="monthly-pnl">$0.00</span>
            </div>
            <div class="perf-item">
              <span class="perf-label">Max Profit</span>
              <span class="perf-value positive" id="max-profit">$0.00</span>
            </div>
            <div class="perf-item">
              <span class="perf-label">Max Loss</span>
              <span class="perf-value negative" id="max-loss-value">$0.00</span>
            </div>
            <div class="perf-item">
              <span class="perf-label">Realized P&L</span>
              <span class="perf-value" id="realized-pnl">$0.00</span>
            </div>
          </div>
        </div>

        <!-- Alerts Panel -->
        <div class="alerts-section" id="alerts-section" style="display: none;">
          <div class="alerts-header">
            <h4>Position Alerts</h4>
            <button class="clear-alerts-btn" id="clear-alerts">Clear All</button>
          </div>
          <div class="alerts-list" id="alerts-list"></div>
        </div>

        <!-- Position Actions -->
        <div class="actions-section">
          <button class="action-btn close-position-btn" id="close-position" disabled>
            Close Position
          </button>
          <button class="action-btn add-to-position-btn" id="add-to-position" disabled>
            Add to Position
          </button>
          <button class="action-btn set-stop-loss-btn" id="set-stop-loss" disabled>
            Set Stop Loss
          </button>
          <button class="action-btn set-take-profit-btn" id="set-take-profit" disabled>
            Set Take Profit
          </button>
        </div>
      </div>
    `

    this.initializeChart()
    this.attachEventListeners()
  }

  /**
   * Initialize the P&L chart
   */
  private initializeChart(): void {
    this.chartCanvas = document.getElementById('pnl-chart') as HTMLCanvasElement
    if (!this.chartCanvas) return

    const container = this.chartCanvas.parentElement
    if (container) {
      this.chartCanvas.width = container.clientWidth ?? this.chartConfig.width
      this.chartCanvas.height = this.chartConfig.height
    }

    this.drawChart()
  }

  /**
   * Draw the P&L chart
   */
  private drawChart(): void {
    if (!this.chartCanvas) return

    const ctx = this.chartCanvas.getContext('2d')
    if (!ctx) return

    const { padding } = this.chartConfig
    const chartWidth = this.chartCanvas.width - padding.left - padding.right
    const chartHeight = this.chartCanvas.height - padding.top - padding.bottom

    // Clear canvas
    ctx.clearRect(0, 0, this.chartCanvas.width, this.chartCanvas.height)

    // Draw background
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, this.chartCanvas.width, this.chartCanvas.height)

    if (this.pnlHistory.length === 0) {
      this.drawEmptyChart(ctx, chartWidth, chartHeight)
      return
    }

    // Calculate data ranges
    const timeRange = this.getTimeRange()
    const filteredData = this.filterDataByTimeframe(timeRange)

    if (filteredData.length === 0) {
      this.drawEmptyChart(ctx, chartWidth, chartHeight)
      return
    }

    const minTime = Math.min(...filteredData.map((d) => d.timestamp))
    const maxTime = Math.max(...filteredData.map((d) => d.timestamp))
    const minPnL = Math.min(...filteredData.map((d) => d.unrealized_pnl))
    const maxPnL = Math.max(...filteredData.map((d) => d.unrealized_pnl))

    // Add padding to PnL range
    const pnlRange = maxPnL - minPnL
    const pnlPadding = pnlRange * 0.1
    const adjustedMinPnL = minPnL - pnlPadding
    const adjustedMaxPnL = maxPnL + pnlPadding

    // Draw grid
    this.drawGrid(ctx, chartWidth, chartHeight, adjustedMinPnL, adjustedMaxPnL)

    // Draw zero line
    this.drawZeroLine(ctx, chartWidth, chartHeight, adjustedMinPnL, adjustedMaxPnL)

    // Draw P&L area and line
    this.drawPnLArea(
      ctx,
      filteredData,
      chartWidth,
      chartHeight,
      minTime,
      maxTime,
      adjustedMinPnL,
      adjustedMaxPnL
    )
    this.drawPnLLine(
      ctx,
      filteredData,
      chartWidth,
      chartHeight,
      minTime,
      maxTime,
      adjustedMinPnL,
      adjustedMaxPnL
    )

    // Draw entry price line
    if (this.positionMetrics && this.positionMetrics.entry_price > 0) {
      this.drawEntryLine(ctx, chartWidth, chartHeight)
    }

    // Draw axes
    this.drawAxes(ctx, chartWidth, chartHeight, minTime, maxTime, adjustedMinPnL, adjustedMaxPnL)
  }

  /**
   * Draw empty chart state
   */
  private drawEmptyChart(
    ctx: CanvasRenderingContext2D,
    chartWidth: number,
    chartHeight: number
  ): void {
    ctx.fillStyle = '#6b7280'
    ctx.font = '14px Inter'
    ctx.textAlign = 'center'
    ctx.fillText(
      'No position data to display',
      this.chartConfig.padding.left + chartWidth / 2,
      this.chartConfig.padding.top + chartHeight / 2
    )
  }

  /**
   * Draw chart grid
   */
  private drawGrid(
    ctx: CanvasRenderingContext2D,
    chartWidth: number,
    chartHeight: number,
    _minPnL: number,
    _maxPnL: number
  ): void {
    ctx.strokeStyle = this.chartConfig.gridColor
    ctx.lineWidth = 1

    // Horizontal grid lines (P&L levels)
    const pnlSteps = 5
    for (let i = 0; i <= pnlSteps; i++) {
      const y = this.chartConfig.padding.top + (chartHeight * i) / pnlSteps
      ctx.beginPath()
      ctx.moveTo(this.chartConfig.padding.left, y)
      ctx.lineTo(this.chartConfig.padding.left + chartWidth, y)
      ctx.stroke()
    }

    // Vertical grid lines (time)
    const timeSteps = 6
    for (let i = 0; i <= timeSteps; i++) {
      const x = this.chartConfig.padding.left + (chartWidth * i) / timeSteps
      ctx.beginPath()
      ctx.moveTo(x, this.chartConfig.padding.top)
      ctx.lineTo(x, this.chartConfig.padding.top + chartHeight)
      ctx.stroke()
    }
  }

  /**
   * Draw zero P&L line
   */
  private drawZeroLine(
    ctx: CanvasRenderingContext2D,
    chartWidth: number,
    chartHeight: number,
    minPnL: number,
    maxPnL: number
  ): void {
    if (minPnL <= 0 && maxPnL >= 0) {
      const zeroY =
        this.chartConfig.padding.top + chartHeight * (1 - (0 - minPnL) / (maxPnL - minPnL))

      ctx.strokeStyle = this.chartConfig.neutralColor
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])

      ctx.beginPath()
      ctx.moveTo(this.chartConfig.padding.left, zeroY)
      ctx.lineTo(this.chartConfig.padding.left + chartWidth, zeroY)
      ctx.stroke()

      ctx.setLineDash([])
    }
  }

  /**
   * Draw P&L area (filled background)
   */
  private drawPnLArea(
    ctx: CanvasRenderingContext2D,
    data: PnLDataPoint[],
    chartWidth: number,
    chartHeight: number,
    minTime: number,
    maxTime: number,
    minPnL: number,
    maxPnL: number
  ): void {
    if (data.length < 2) return

    const timeRange = maxTime - minTime
    const pnlRange = maxPnL - minPnL

    ctx.beginPath()

    // Start from bottom-left
    const firstPoint = data[0]
    const firstX =
      this.chartConfig.padding.left + ((firstPoint.timestamp - minTime) / timeRange) * chartWidth
    const firstY =
      this.chartConfig.padding.top +
      chartHeight * (1 - (firstPoint.unrealized_pnl - minPnL) / pnlRange)

    ctx.moveTo(firstX, this.chartConfig.padding.top + chartHeight)
    ctx.lineTo(firstX, firstY)

    // Draw the area path
    for (let i = 1; i < data.length; i++) {
      const point = data[i]
      const x =
        this.chartConfig.padding.left + ((point.timestamp - minTime) / timeRange) * chartWidth
      const y =
        this.chartConfig.padding.top +
        chartHeight * (1 - (point.unrealized_pnl - minPnL) / pnlRange)
      ctx.lineTo(x, y)
    }

    // Close the area at the bottom
    const lastPoint = data[data.length - 1]
    const lastX =
      this.chartConfig.padding.left + ((lastPoint.timestamp - minTime) / timeRange) * chartWidth
    ctx.lineTo(lastX, this.chartConfig.padding.top + chartHeight)
    ctx.closePath()

    // Fill with gradient
    const gradient = ctx.createLinearGradient(
      0,
      this.chartConfig.padding.top,
      0,
      this.chartConfig.padding.top + chartHeight
    )
    const latestPnL = data[data.length - 1].unrealized_pnl

    if (latestPnL >= 0) {
      gradient.addColorStop(0, 'rgba(34, 197, 94, 0.3)')
      gradient.addColorStop(1, 'rgba(34, 197, 94, 0.05)')
    } else {
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0.3)')
      gradient.addColorStop(1, 'rgba(239, 68, 68, 0.05)')
    }

    ctx.fillStyle = gradient
    ctx.fill()
  }

  /**
   * Draw P&L line
   */
  private drawPnLLine(
    ctx: CanvasRenderingContext2D,
    data: PnLDataPoint[],
    chartWidth: number,
    chartHeight: number,
    minTime: number,
    maxTime: number,
    minPnL: number,
    maxPnL: number
  ): void {
    if (data.length < 2) return

    const timeRange = maxTime - minTime
    const pnlRange = maxPnL - minPnL
    const latestPnL = data[data.length - 1].unrealized_pnl

    ctx.strokeStyle = latestPnL >= 0 ? this.chartConfig.profitColor : this.chartConfig.lossColor
    ctx.lineWidth = this.chartConfig.lineWidth

    ctx.beginPath()

    for (let i = 0; i < data.length; i++) {
      const point = data[i]
      const x =
        this.chartConfig.padding.left + ((point.timestamp - minTime) / timeRange) * chartWidth
      const y =
        this.chartConfig.padding.top +
        chartHeight * (1 - (point.unrealized_pnl - minPnL) / pnlRange)

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }

    ctx.stroke()

    // Draw current point
    const lastPoint = data[data.length - 1]
    const lastX =
      this.chartConfig.padding.left + ((lastPoint.timestamp - minTime) / timeRange) * chartWidth
    const lastY =
      this.chartConfig.padding.top +
      chartHeight * (1 - (lastPoint.unrealized_pnl - minPnL) / pnlRange)

    ctx.beginPath()
    ctx.arc(lastX, lastY, this.chartConfig.pointRadius, 0, 2 * Math.PI)
    ctx.fillStyle = latestPnL >= 0 ? this.chartConfig.profitColor : this.chartConfig.lossColor
    ctx.fill()
  }

  /**
   * Draw entry price line
   */
  private drawEntryLine(
    ctx: CanvasRenderingContext2D,
    chartWidth: number,
    chartHeight: number
  ): void {
    // This would show the entry price as a horizontal reference line
    // Implementation depends on having price data in the chart

    ctx.strokeStyle = '#f59e0b'
    ctx.lineWidth = 1
    ctx.setLineDash([3, 3])

    // For now, draw a middle line as placeholder
    const entryY = this.chartConfig.padding.top + chartHeight / 2

    ctx.beginPath()
    ctx.moveTo(this.chartConfig.padding.left, entryY)
    ctx.lineTo(this.chartConfig.padding.left + chartWidth, entryY)
    ctx.stroke()

    ctx.setLineDash([])

    // Add entry label
    ctx.fillStyle = '#f59e0b'
    ctx.font = '11px Inter'
    ctx.textAlign = 'left'
    ctx.fillText('Entry', this.chartConfig.padding.left + 5, entryY - 5)
  }

  /**
   * Draw chart axes with labels
   */
  private drawAxes(
    ctx: CanvasRenderingContext2D,
    chartWidth: number,
    chartHeight: number,
    minTime: number,
    maxTime: number,
    minPnL: number,
    maxPnL: number
  ): void {
    ctx.strokeStyle = this.chartConfig.axisColor
    ctx.fillStyle = this.chartConfig.axisColor
    ctx.font = '11px Inter'
    ctx.lineWidth = 1

    // Y-axis (P&L values)
    const pnlSteps = 5
    for (let i = 0; i <= pnlSteps; i++) {
      const pnlValue = minPnL + ((maxPnL - minPnL) * i) / pnlSteps
      const y = this.chartConfig.padding.top + chartHeight * (1 - i / pnlSteps)

      ctx.textAlign = 'right'
      ctx.fillText(`$${pnlValue.toFixed(2)}`, this.chartConfig.padding.left - 10, y + 4)
    }

    // X-axis (time labels)
    const timeSteps = 4
    for (let i = 0; i <= timeSteps; i++) {
      const timeValue = minTime + ((maxTime - minTime) * i) / timeSteps
      const x = this.chartConfig.padding.left + (chartWidth * i) / timeSteps

      ctx.textAlign = 'center'
      ctx.fillText(
        new Date(timeValue).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        x,
        this.chartConfig.padding.top + chartHeight + 25
      )
    }
  }

  /**
   * Attach event listeners
   */
  private attachEventListeners(): void {
    // Timeframe selection
    const timeframeBtns = document.querySelectorAll('.timeframe-btn')
    timeframeBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        timeframeBtns.forEach((b) => b.classList.remove('active'))
        ;(e.target as HTMLElement).classList.add('active')
        this.drawChart()
      })
    })

    // Chart reset
    const resetBtn = document.getElementById('reset-chart')
    resetBtn?.addEventListener('click', () => {
      this.pnlHistory = []
      this.drawChart()
    })

    // Position actions
    const closePositionBtn = document.getElementById('close-position')
    const addToPositionBtn = document.getElementById('add-to-position')
    const setStopLossBtn = document.getElementById('set-stop-loss')
    const setTakeProfitBtn = document.getElementById('set-take-profit')

    closePositionBtn?.addEventListener('click', () => this.closePosition())
    addToPositionBtn?.addEventListener('click', () => this.addToPosition())
    setStopLossBtn?.addEventListener('click', () => this.setStopLoss())
    setTakeProfitBtn?.addEventListener('click', () => this.setTakeProfit())

    // Clear alerts
    const clearAlertsBtn = document.getElementById('clear-alerts')
    clearAlertsBtn?.addEventListener('click', () => this.clearAlerts())

    // Chart interactions (mouse events for tooltip)
    if (this.chartCanvas) {
      this.chartCanvas.addEventListener('mousemove', (e) => this.handleChartMouseMove(e))
      this.chartCanvas.addEventListener('mouseleave', () => this.hideChartTooltip())
    }
  }

  /**
   * Handle chart mouse movement for tooltip
   */
  private handleChartMouseMove(event: MouseEvent): void {
    if (!this.chartCanvas || this.pnlHistory.length === 0) return

    const rect = this.chartCanvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Calculate which data point we're hovering over
    const chartWidth =
      this.chartCanvas.width - this.chartConfig.padding.left - this.chartConfig.padding.right
    const relativeX = (x - this.chartConfig.padding.left) / chartWidth

    if (relativeX >= 0 && relativeX <= 1) {
      const timeRange = this.getTimeRange()
      const filteredData = this.filterDataByTimeframe(timeRange)

      if (filteredData.length > 0) {
        const dataIndex = Math.round(relativeX * (filteredData.length - 1))
        const dataPoint = filteredData[dataIndex]

        if (dataPoint) {
          this.showChartTooltip(dataPoint, x, y)
        }
      }
    }
  }

  /**
   * Show chart tooltip
   */
  private showChartTooltip(dataPoint: PnLDataPoint, x: number, y: number): void {
    const tooltip = document.getElementById('chart-tooltip')
    if (!tooltip) return

    tooltip.innerHTML = `
      <div class="tooltip-time">${new Date(dataPoint.timestamp).toLocaleTimeString()}</div>
      <div class="tooltip-pnl ${dataPoint.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
        P&L: $${dataPoint.unrealized_pnl.toFixed(2)}
      </div>
      <div class="tooltip-price">Price: $${dataPoint.price.toFixed(2)}</div>
    `

    tooltip.style.display = 'block'
    tooltip.style.left = `${x + 10}px`
    tooltip.style.top = `${y - 10}px`
  }

  /**
   * Hide chart tooltip
   */
  private hideChartTooltip(): void {
    const tooltip = document.getElementById('chart-tooltip')
    if (tooltip) {
      tooltip.style.display = 'none'
    }
  }

  /**
   * Get current timeframe selection
   */
  private getTimeRange(): string {
    const activeBtn = document.querySelector('.timeframe-btn.active')
    return activeBtn?.getAttribute('data-timeframe') ?? '1h'
  }

  /**
   * Filter P&L data by timeframe
   */
  private filterDataByTimeframe(timeframe: string): PnLDataPoint[] {
    const now = Date.now()
    let cutoffTime = 0

    switch (timeframe) {
      case '1h':
        cutoffTime = now - 60 * 60 * 1000
        break
      case '4h':
        cutoffTime = now - 4 * 60 * 60 * 1000
        break
      case '1d':
        cutoffTime = now - 24 * 60 * 60 * 1000
        break
      case 'all':
      default:
        cutoffTime = 0
        break
    }

    return this.pnlHistory.filter((point) => point.timestamp >= cutoffTime)
  }

  /**
   * Add new P&L data point
   */
  private addPnLDataPoint(currentPrice: number): void {
    if (!this.positionMetrics) return

    const now = Date.now()

    // Don't add points too frequently (max once per second)
    if (now - this.lastUpdate < 1000) return

    this.lastUpdate = now

    const dataPoint: PnLDataPoint = {
      timestamp: now,
      unrealized_pnl: this.positionMetrics.unrealized_pnl,
      price: currentPrice,
      cumulative_pnl: this.positionMetrics.total_pnl,
    }

    this.pnlHistory.push(dataPoint)

    // Keep only last 1000 points to prevent memory issues
    if (this.pnlHistory.length > 1000) {
      this.pnlHistory = this.pnlHistory.slice(-1000)
    }
  }

  /**
   * Update position metrics calculations
   */
  private updatePositionMetrics(): void {
    if (!this.currentPosition || !this.currentMarketData) return

    const position = this.currentPosition
    const currentPrice = this.currentMarketData.price
    const quantity = position.quantity ?? position.size ?? 0
    const entryPrice = position.average_price ?? position.entry_price ?? 0

    if (quantity === 0) {
      this.positionMetrics = null
      return
    }

    // Calculate unrealized P&L
    let unrealizedPnL = 0
    if (position.side?.toLowerCase() === 'long') {
      unrealizedPnL = (currentPrice - entryPrice) * Math.abs(quantity)
    } else if (position.side?.toLowerCase() === 'short') {
      unrealizedPnL = (entryPrice - currentPrice) * Math.abs(quantity)
    }

    // Calculate other metrics
    const leverage = position.leverage ?? 1
    const marginUsed = (entryPrice * Math.abs(quantity)) / leverage
    const liquidationPrice = this.calculateLiquidationPrice(position, leverage)

    this.positionMetrics = {
      unrealized_pnl: unrealizedPnL,
      realized_pnl: position.unrealized_pnl ?? 0,
      total_pnl: unrealizedPnL + (position.unrealized_pnl ?? 0),
      daily_pnl: 0, // Would need daily tracking
      weekly_pnl: 0, // Would need weekly tracking
      monthly_pnl: 0, // Would need monthly tracking
      pnl_percentage:
        entryPrice > 0 ? (unrealizedPnL / (entryPrice * Math.abs(quantity))) * 100 : 0,
      max_profit: Math.max(...this.pnlHistory.map((p) => p.unrealized_pnl), unrealizedPnL),
      max_loss: Math.min(...this.pnlHistory.map((p) => p.unrealized_pnl), unrealizedPnL),
      duration: this.calculatePositionDuration(),
      entry_price: entryPrice,
      current_price: currentPrice,
      quantity: Math.abs(quantity),
      side: quantity > 0 ? 'long' : quantity < 0 ? 'short' : 'flat',
      leverage: leverage,
      margin_used: marginUsed,
      liquidation_price: liquidationPrice,
    }
  }

  /**
   * Calculate liquidation price
   */
  private calculateLiquidationPrice(position: Position, leverage: number): number | undefined {
    // Simplified liquidation calculation
    // In practice, this would depend on exchange-specific formulas
    const entryPrice = position.average_price ?? position.entry_price ?? 0
    const liquidationThreshold = 0.8 // 80% of margin

    if (position.side?.toLowerCase() === 'long') {
      return entryPrice * (1 - liquidationThreshold / leverage)
    } else if (position.side?.toLowerCase() === 'short') {
      return entryPrice * (1 + liquidationThreshold / leverage)
    }

    return undefined
  }

  /**
   * Calculate position duration in minutes
   */
  private calculatePositionDuration(): number {
    // This would require position entry timestamp
    // For now, return approximate duration based on P&L history
    if (this.pnlHistory.length > 0) {
      const firstPoint = this.pnlHistory[0]
      return (Date.now() - firstPoint.timestamp) / (1000 * 60)
    }
    return 0
  }

  /**
   * Update position display elements
   */
  private updatePositionDisplay(): void {
    this.updatePositionStatus()
    this.updateMetricCards()
    this.updatePerformanceGrid()
    this.updateActionButtons()
  }

  /**
   * Update position status indicator
   */
  private updatePositionStatus(): void {
    const statusEl = document.getElementById('position-status')
    if (!statusEl) return

    const indicator = statusEl.querySelector('.status-indicator')
    const text = statusEl.querySelector('.status-text')

    if (!this.positionMetrics || this.positionMetrics.side === 'flat') {
      indicator?.setAttribute('class', 'status-indicator flat')
      if (text) text.textContent = 'No Position'
    } else {
      indicator?.setAttribute('class', `status-indicator ${this.positionMetrics.side}`)
      if (text) text.textContent = `${this.positionMetrics.side.toUpperCase()} Position`
    }
  }

  /**
   * Update metric cards
   */
  private updateMetricCards(): void {
    if (!this.positionMetrics) {
      this.clearMetricCards()
      return
    }

    const metrics = this.positionMetrics

    // Unrealized P&L card
    this.updateElement(
      'unrealized-pnl',
      `$${metrics.unrealized_pnl.toFixed(2)}`,
      metrics.unrealized_pnl >= 0 ? 'positive' : 'negative'
    )
    this.updateElement('pnl-percentage', `${metrics.pnl_percentage.toFixed(2)}%`)
    this.updateElement('pnl-duration', this.formatDuration(metrics.duration))
    this.updateElement('pnl-trend', metrics.unrealized_pnl >= 0 ? '↗' : '↘')

    // Entry card
    this.updateElement('entry-price', `$${metrics.entry_price.toFixed(2)}`)
    this.updateElement('position-size', metrics.quantity.toFixed(4))
    this.updateElement('leverage-info', `${metrics.leverage}x`)

    // Current price card
    this.updateElement('current-price', `$${metrics.current_price.toFixed(2)}`)
    const priceChange = ((metrics.current_price - metrics.entry_price) / metrics.entry_price) * 100
    this.updateElement(
      'price-change',
      `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`,
      priceChange >= 0 ? 'positive' : 'negative'
    )
    this.updateElement(
      'price-distance',
      `$${Math.abs(metrics.current_price - metrics.entry_price).toFixed(2)}`
    )
    this.updateElement('last-update', new Date().toLocaleTimeString())

    // Risk card
    this.updateElement('margin-used', `$${metrics.margin_used.toFixed(2)}`)
    if (metrics.liquidation_price) {
      const liquidationDistance = Math.abs(metrics.current_price - metrics.liquidation_price)
      const liquidationPercent = (liquidationDistance / metrics.current_price) * 100
      this.updateElement('liquidation-distance', `${liquidationPercent.toFixed(1)}%`)
    }
    this.updateElement('max-loss', `$${Math.abs(metrics.max_loss).toFixed(2)}`)

    // Risk level
    const riskLevel = this.calculateRiskLevel(metrics)
    this.updateElement('risk-level', riskLevel.label, riskLevel.class)
  }

  /**
   * Update performance grid
   */
  private updatePerformanceGrid(): void {
    if (!this.positionMetrics) return

    const metrics = this.positionMetrics

    this.updateElement(
      'daily-pnl',
      `$${metrics.daily_pnl.toFixed(2)}`,
      metrics.daily_pnl >= 0 ? 'positive' : 'negative'
    )
    this.updateElement(
      'weekly-pnl',
      `$${metrics.weekly_pnl.toFixed(2)}`,
      metrics.weekly_pnl >= 0 ? 'positive' : 'negative'
    )
    this.updateElement(
      'monthly-pnl',
      `$${metrics.monthly_pnl.toFixed(2)}`,
      metrics.monthly_pnl >= 0 ? 'positive' : 'negative'
    )
    this.updateElement('max-profit', `$${metrics.max_profit.toFixed(2)}`)
    this.updateElement('max-loss-value', `$${Math.abs(metrics.max_loss).toFixed(2)}`)
    this.updateElement(
      'realized-pnl',
      `$${metrics.realized_pnl.toFixed(2)}`,
      metrics.realized_pnl >= 0 ? 'positive' : 'negative'
    )
  }

  /**
   * Update action buttons
   */
  private updateActionButtons(): void {
    const hasPosition = this.positionMetrics && this.positionMetrics.side !== 'flat'

    const buttons = ['close-position', 'add-to-position', 'set-stop-loss', 'set-take-profit']

    buttons.forEach((buttonId) => {
      const button = document.getElementById(buttonId) as HTMLButtonElement
      if (button) {
        button.disabled = !hasPosition
      }
    })
  }

  /**
   * Clear metric cards when no position
   */
  private clearMetricCards(): void {
    const elements = [
      'unrealized-pnl',
      'pnl-percentage',
      'pnl-duration',
      'pnl-trend',
      'entry-price',
      'position-size',
      'leverage-info',
      'current-price',
      'price-change',
      'price-distance',
      'last-update',
      'margin-used',
      'liquidation-distance',
      'max-loss',
      'risk-level',
    ]

    elements.forEach((id) => {
      this.updateElement(id, '--', 'neutral')
    })
  }

  /**
   * Update real-time metrics (called frequently)
   */
  private updateRealTimeMetrics(): void {
    if (!this.positionMetrics) return

    // Update only time-sensitive elements
    this.updateElement('last-update', new Date().toLocaleTimeString())

    if (this.currentMarketData) {
      this.updateElement('current-price', `$${this.currentMarketData.price.toFixed(2)}`)
    }
  }

  /**
   * Update charts
   */
  private updateCharts(): void {
    this.drawChart()
  }

  /**
   * Check for position alerts
   */
  private checkAlerts(): void {
    if (!this.positionMetrics) return

    const metrics = this.positionMetrics
    const newAlerts: PositionAlert[] = []

    // P&L percentage alerts
    if (Math.abs(metrics.pnl_percentage) > 10) {
      newAlerts.push({
        type: metrics.pnl_percentage > 0 ? 'profit_target' : 'stop_loss',
        severity: 'medium',
        message: `Position has ${metrics.pnl_percentage > 0 ? 'gained' : 'lost'} ${Math.abs(metrics.pnl_percentage).toFixed(1)}%`,
        threshold: 10,
        current_value: Math.abs(metrics.pnl_percentage),
      })
    }

    // Liquidation warning
    if (metrics.liquidation_price) {
      const liquidationDistance = Math.abs(metrics.current_price - metrics.liquidation_price)
      const liquidationPercent = (liquidationDistance / metrics.current_price) * 100

      if (liquidationPercent < 20) {
        newAlerts.push({
          type: 'liquidation_warning',
          severity: liquidationPercent < 5 ? 'critical' : 'high',
          message: `Position is ${liquidationPercent.toFixed(1)}% away from liquidation`,
          threshold: 20,
          current_value: liquidationPercent,
        })
      }
    }

    // Add new alerts
    newAlerts.forEach((alert) => {
      if (!this.alerts.find((a) => a.type === alert.type && a.message === alert.message)) {
        this.alerts.push(alert)
        this._onPositionAlert?.(alert)
      }
    })

    this.updateAlertsDisplay()
  }

  /**
   * Update alerts display
   */
  private updateAlertsDisplay(): void {
    const alertsSection = document.getElementById('alerts-section')
    const alertsList = document.getElementById('alerts-list')

    if (!alertsSection || !alertsList) return

    if (this.alerts.length === 0) {
      alertsSection.style.display = 'none'
      return
    }

    alertsSection.style.display = 'block'
    alertsList.innerHTML = this.alerts
      .map(
        (alert, index) => `
      <div class="alert-item severity-${alert.severity}">
        <div class="alert-icon">${this.getAlertIcon(alert.type)}</div>
        <div class="alert-content">
          <div class="alert-message">${alert.message}</div>
          <div class="alert-details">
            ${alert.type.replace('_', ' ').toUpperCase()} • ${alert.severity.toUpperCase()}
          </div>
        </div>
        <button class="alert-dismiss" data-index="${index}">✕</button>
      </div>
    `
      )
      .join('')

    // Attach dismiss handlers
    const dismissBtns = alertsList.querySelectorAll('.alert-dismiss')
    dismissBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const index = parseInt((e.target as HTMLElement).dataset.index ?? '0')
        this.dismissAlert(index)
      })
    })
  }

  /**
   * Get alert icon
   */
  private getAlertIcon(type: string): string {
    const icons = {
      profit_target: '🎯',
      stop_loss: '⚠️',
      liquidation_warning: '🚨',
      margin_call: '📢',
    }
    return icons[type as keyof typeof icons] ?? '⚠️'
  }

  /**
   * Dismiss alert
   */
  private dismissAlert(index: number): void {
    this.alerts.splice(index, 1)
    this.updateAlertsDisplay()
  }

  /**
   * Clear all alerts
   */
  private clearAlerts(): void {
    this.alerts = []
    this.updateAlertsDisplay()
  }

  /**
   * Start real-time updates
   */
  private startRealtimeUpdates(): void {
    this.updateInterval = window.setInterval(() => {
      this.updateRealTimeMetrics()
    }, 1000) // Update every second
  }

  /**
   * Position actions
   */
  private closePosition(): void {
    // Implementation would trigger close position API call
    console.log('Close position requested')
  }

  private addToPosition(): void {
    // Implementation would show add to position dialog
    console.log('Add to position requested')
  }

  private setStopLoss(): void {
    // Implementation would show stop loss dialog
    console.log('Set stop loss requested')
  }

  private setTakeProfit(): void {
    // Implementation would show take profit dialog
    console.log('Set take profit requested')
  }

  /**
   * Utility methods
   */
  private updateElement(id: string, text: string, className?: string): void {
    const element = document.getElementById(id)
    if (element) {
      element.textContent = text
      if (className) {
        element.className = element.className.replace(/\b(positive|negative|neutral)\b/g, '')
        element.classList.add(className)
      }
    }
  }

  private formatDuration(minutes: number): string {
    if (minutes < 60) {
      return `${Math.round(minutes)}m`
    } else if (minutes < 1440) {
      return `${Math.round(minutes / 60)}h`
    } else {
      return `${Math.round(minutes / 1440)}d`
    }
  }

  private calculateRiskLevel(metrics: PositionMetrics): { label: string; class: string } {
    const leverage = metrics.leverage
    const pnlPercent = Math.abs(metrics.pnl_percentage)

    if (leverage > 10 || pnlPercent > 20) {
      return { label: 'High', class: 'risk-high' }
    } else if (leverage > 5 || pnlPercent > 10) {
      return { label: 'Medium', class: 'risk-medium' }
    } else {
      return { label: 'Low', class: 'risk-low' }
    }
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  }
}
