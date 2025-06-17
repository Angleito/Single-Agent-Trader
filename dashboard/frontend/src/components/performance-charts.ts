import type { Position, RiskMetrics } from '../types'

interface ChartData {
  labels: string[]
  values: number[]
  timestamps: Date[]
}

interface PerformanceMetrics {
  totalPnL: number
  totalPnLPercent: number
  winRate: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
  avgWin: number
  avgLoss: number
  maxDrawdown: number
  sharpeRatio: number
  dailyPnL: ChartData
  weeklyPnL: ChartData
  monthlyPnL: ChartData
}

export class PerformanceCharts {
  private container: HTMLElement
  private metrics: PerformanceMetrics
  private charts: Map<string, HTMLCanvasElement> = new Map()
  private contexts: Map<string, CanvasRenderingContext2D> = new Map()
  private animationFrames: Map<string, number> = new Map()
  private resizeObserver: ResizeObserver

  // Chart colors and styles
  private readonly colors = {
    profit: '#00d4aa',
    loss: '#ff5252',
    neutral: '#787b86',
    grid: '#2a2e39',
    text: '#b2b5be',
    background: '#131722',
    chartBg: '#0b0e11',
    neonGreen: '#00ffaa',
    neonRed: '#ff3366',
    neonBlue: '#2962ff',
  }

  constructor(container: HTMLElement) {
    this.container = container
    this.metrics = this.initializeMetrics()

    // Set up resize observer
    this.resizeObserver = new ResizeObserver(() => {
      this.handleResize()
    })
    this.resizeObserver.observe(this.container)

    this.render()
  }

  private initializeMetrics(): PerformanceMetrics {
    return {
      totalPnL: 0,
      totalPnLPercent: 0,
      winRate: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      avgWin: 0,
      avgLoss: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      dailyPnL: { labels: [], values: [], timestamps: [] },
      weeklyPnL: { labels: [], values: [], timestamps: [] },
      monthlyPnL: { labels: [], values: [], timestamps: [] },
    }
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="performance-charts-container">
        <!-- Performance Summary -->
        <div class="performance-summary">
          <div class="summary-header">
            <h3>Performance Overview</h3>
            <div class="time-selector">
              <button class="time-btn active" data-period="daily">Daily</button>
              <button class="time-btn" data-period="weekly">Weekly</button>
              <button class="time-btn" data-period="monthly">Monthly</button>
            </div>
          </div>
          
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">Total P&L</div>
              <div class="metric-value" data-metric="total-pnl">$0.00</div>
              <div class="metric-change" data-metric="total-pnl-percent">0.00%</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Win Rate</div>
              <div class="metric-value" data-metric="win-rate">0%</div>
              <div class="metric-subtext" data-metric="win-loss">0W / 0L</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Average Win</div>
              <div class="metric-value positive" data-metric="avg-win">$0.00</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Average Loss</div>
              <div class="metric-value negative" data-metric="avg-loss">$0.00</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Max Drawdown</div>
              <div class="metric-value negative" data-metric="max-drawdown">0.00%</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Sharpe Ratio</div>
              <div class="metric-value" data-metric="sharpe-ratio">0.00</div>
            </div>
          </div>
        </div>

        <!-- Charts Container -->
        <div class="charts-grid">
          <!-- P&L Chart -->
          <div class="chart-wrapper" data-chart="pnl">
            <div class="chart-header">
              <h4>Profit & Loss Timeline</h4>
              <div class="chart-legend">
                <span class="legend-item profit">
                  <span class="legend-dot"></span>Profit
                </span>
                <span class="legend-item loss">
                  <span class="legend-dot"></span>Loss
                </span>
              </div>
            </div>
            <div class="chart-container">
              <canvas id="pnl-chart" data-chart-canvas="pnl"></canvas>
            </div>
          </div>

          <!-- Win/Loss Donut Chart -->
          <div class="chart-wrapper" data-chart="win-loss">
            <div class="chart-header">
              <h4>Win/Loss Distribution</h4>
            </div>
            <div class="chart-container">
              <canvas id="win-loss-chart" data-chart-canvas="win-loss"></canvas>
              <div class="donut-center">
                <div class="donut-value" data-donut-value="0">0%</div>
                <div class="donut-label">Win Rate</div>
              </div>
            </div>
          </div>

          <!-- Position Timeline -->
          <div class="chart-wrapper full-width" data-chart="timeline">
            <div class="chart-header">
              <h4>Position Timeline</h4>
              <div class="chart-info">Entry/Exit Points & P&L</div>
            </div>
            <div class="chart-container">
              <canvas id="timeline-chart" data-chart-canvas="timeline"></canvas>
            </div>
          </div>

          <!-- Daily Performance Comparison -->
          <div class="chart-wrapper full-width" data-chart="comparison">
            <div class="chart-header">
              <h4>Performance Comparison</h4>
            </div>
            <div class="chart-container">
              <canvas id="comparison-chart" data-chart-canvas="comparison"></canvas>
            </div>
          </div>
        </div>
      </div>
    `

    this.setupCharts()
    this.setupEventListeners()
    this.applyStyles()
  }

  private setupCharts(): void {
    const chartConfigs = [
      { id: 'pnl', canvas: 'pnl-chart' },
      { id: 'win-loss', canvas: 'win-loss-chart' },
      { id: 'timeline', canvas: 'timeline-chart' },
      { id: 'comparison', canvas: 'comparison-chart' },
    ]

    chartConfigs.forEach((config) => {
      const canvas = this.container.querySelector(`#${config.canvas}`) as HTMLCanvasElement
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          this.charts.set(config.id, canvas)
          this.contexts.set(config.id, ctx)
          this.setupCanvas(canvas)
        }
      }
    })

    // Initial draw
    this.drawAllCharts()
  }

  private setupCanvas(canvas: HTMLCanvasElement): void {
    const container = canvas.parentElement
    if (!container) return

    const rect = container.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    canvas.style.width = `${rect.width}px`
    canvas.style.height = `${rect.height}px`

    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.scale(dpr, dpr)
    }
  }

  private setupEventListeners(): void {
    // Time period selector
    const timeButtons = this.container.querySelectorAll('.time-btn')
    timeButtons.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const target = e.target as HTMLButtonElement
        const period = target.dataset.period

        timeButtons.forEach((b) => b.classList.remove('active'))
        target.classList.add('active')

        if (period) {
          this.switchTimePeriod(period)
        }
      })
    })
  }

  private drawAllCharts(): void {
    this.drawPnLChart()
    this.drawWinLossDonut()
    this.drawPositionTimeline()
    this.drawComparisonChart()
  }

  private drawPnLChart(): void {
    const canvas = this.charts.get('pnl')
    const ctx = this.contexts.get('pnl')
    if (!canvas || !ctx) return

    const _width = canvas.width / (window.devicePixelRatio || 1)
    const _height = canvas.height / (window.devicePixelRatio || 1)
    const padding = 40

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = this.colors.chartBg
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    this.drawGrid(ctx, width, height, padding)

    // Get chart data
    const data = this.metrics.dailyPnL
    if (data.values.length === 0) {
      // Draw empty state
      ctx.fillStyle = this.colors.text
      ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('No data available', width / 2, height / 2)
      return
    }

    // Calculate chart dimensions
    const chartWidth = width - padding * 2
    const chartHeight = height - padding * 2
    const barWidth = (chartWidth / data.values.length) * 0.7
    const spacing = (chartWidth / data.values.length) * 0.3

    // Find max value for scaling
    const maxValue = Math.max(...data.values.map(Math.abs))
    const scale = maxValue > 0 ? chartHeight / 2 / maxValue : 1

    // Draw zero line
    const zeroY = height / 2
    ctx.strokeStyle = this.colors.grid
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(padding, zeroY)
    ctx.lineTo(width - padding, zeroY)
    ctx.stroke()

    // Draw bars with animation
    data.values.forEach((value, index) => {
      const x = padding + index * (barWidth + spacing) + spacing / 2
      const barHeight = Math.abs(value) * scale
      const y = value >= 0 ? zeroY - barHeight : zeroY

      // Create gradient
      const gradient = ctx.createLinearGradient(x, y, x, y + barHeight)
      if (value >= 0) {
        gradient.addColorStop(0, this.colors.neonGreen)
        gradient.addColorStop(1, this.colors.profit)
      } else {
        gradient.addColorStop(0, this.colors.neonRed)
        gradient.addColorStop(1, this.colors.loss)
      }

      // Draw bar
      ctx.fillStyle = gradient
      ctx.fillRect(x, y, barWidth, barHeight)

      // Add glow effect
      ctx.shadowBlur = 10
      ctx.shadowColor = value >= 0 ? this.colors.neonGreen : this.colors.neonRed
      ctx.fillRect(x, y, barWidth, barHeight)
      ctx.shadowBlur = 0

      // Draw value label
      if (index % Math.ceil(data.values.length / 10) === 0) {
        ctx.fillStyle = this.colors.text
        ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(`$${this.formatNumber(value)}`, x + barWidth / 2, y - 5)
      }
    })

    // Draw axis labels
    this.drawAxisLabels(ctx, width, height, padding, data.labels)
  }

  private drawWinLossDonut(): void {
    const canvas = this.charts.get('win-loss')
    const ctx = this.contexts.get('win-loss')
    if (!canvas || !ctx) return

    const _width = canvas.width / (window.devicePixelRatio || 1)
    const _height = canvas.height / (window.devicePixelRatio || 1)
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 2 - 30
    const innerRadius = radius * 0.65

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Calculate angles
    const winRate = this.metrics.winRate || 0
    const winAngle = (winRate / 100) * Math.PI * 2
    const lossAngle = Math.PI * 2 - winAngle

    // Draw donut segments
    const segments = [
      { value: winAngle, color: this.colors.profit, glow: this.colors.neonGreen },
      { value: lossAngle, color: this.colors.loss, glow: this.colors.neonRed },
    ]

    let currentAngle = -Math.PI / 2 // Start at top

    segments.forEach((segment) => {
      if (segment.value === 0) return

      // Draw segment
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + segment.value)
      ctx.arc(centerX, centerY, innerRadius, currentAngle + segment.value, currentAngle, true)
      ctx.closePath()

      // Add gradient
      const gradient = ctx.createRadialGradient(
        centerX,
        centerY,
        innerRadius,
        centerX,
        centerY,
        radius
      )
      gradient.addColorStop(0, segment.color)
      gradient.addColorStop(1, segment.glow)
      ctx.fillStyle = gradient
      ctx.fill()

      // Add glow
      ctx.shadowBlur = 20
      ctx.shadowColor = segment.glow
      ctx.fill()
      ctx.shadowBlur = 0

      currentAngle += segment.value
    })

    // Update center text
    const donutValue = this.container.querySelector('[data-donut-value]')
    if (donutValue) {
      donutValue.textContent = `${Math.round(winRate)}%`
    }
  }

  private drawPositionTimeline(): void {
    const canvas = this.charts.get('timeline')
    const ctx = this.contexts.get('timeline')
    if (!canvas || !ctx) return

    const _width = canvas.width / (window.devicePixelRatio || 1)
    const _height = canvas.height / (window.devicePixelRatio || 1)
    const padding = 40

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = this.colors.chartBg
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    this.drawGrid(ctx, width, height, padding)

    // Mock data for demonstration
    const positions = [
      { entry: 100, exit: 120, profit: true, timestamp: new Date() },
      { entry: 150, exit: 140, profit: false, timestamp: new Date() },
      { entry: 180, exit: 200, profit: true, timestamp: new Date() },
    ]

    if (positions.length === 0) {
      ctx.fillStyle = this.colors.text
      ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('No position history', width / 2, height / 2)
      return
    }

    // Draw timeline
    const timelineY = height / 2
    const segmentWidth = (width - padding * 2) / positions.length

    positions.forEach((pos, index) => {
      const x = padding + index * segmentWidth + segmentWidth / 2

      // Draw position line
      ctx.strokeStyle = pos.profit ? this.colors.profit : this.colors.loss
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(x - 20, timelineY)
      ctx.lineTo(x + 20, timelineY)
      ctx.stroke()

      // Draw entry point
      ctx.fillStyle = this.colors.neonBlue
      ctx.beginPath()
      ctx.arc(x - 20, timelineY, 6, 0, Math.PI * 2)
      ctx.fill()

      // Draw exit point
      ctx.fillStyle = pos.profit ? this.colors.neonGreen : this.colors.neonRed
      ctx.beginPath()
      ctx.arc(x + 20, timelineY, 6, 0, Math.PI * 2)
      ctx.fill()

      // Draw P&L label
      ctx.fillStyle = pos.profit ? this.colors.profit : this.colors.loss
      ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif'
      ctx.textAlign = 'center'
      const pnl = pos.exit - pos.entry
      ctx.fillText(`${pnl > 0 ? '+' : ''}$${pnl}`, x, timelineY - 20)
    })
  }

  private drawComparisonChart(): void {
    const canvas = this.charts.get('comparison')
    const ctx = this.contexts.get('comparison')
    if (!canvas || !ctx) return

    const _width = canvas.width / (window.devicePixelRatio || 1)
    const _height = canvas.height / (window.devicePixelRatio || 1)
    const padding = 40

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = this.colors.chartBg
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    this.drawGrid(ctx, width, height, padding)

    // Draw comparison bars for daily, weekly, monthly
    const periods = ['Daily', 'Weekly', 'Monthly']
    const values = [150, 850, 2400] // Mock data
    const barHeight = 30
    const spacing = 20
    const startY = (height - (barHeight * 3 + spacing * 2)) / 2

    const maxValue = Math.max(...values)
    const scale = (width - padding * 2 - 100) / maxValue

    periods.forEach((period, index) => {
      const y = startY + index * (barHeight + spacing)
      const barWidth = values[index] * scale

      // Draw label
      ctx.fillStyle = this.colors.text
      ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif'
      ctx.textAlign = 'right'
      ctx.fillText(period, padding + 80, y + barHeight / 2 + 5)

      // Draw bar
      const gradient = ctx.createLinearGradient(padding + 100, y, padding + 100 + barWidth, y)
      gradient.addColorStop(0, this.colors.neonBlue)
      gradient.addColorStop(1, values[index] >= 0 ? this.colors.neonGreen : this.colors.neonRed)

      ctx.fillStyle = gradient
      ctx.fillRect(padding + 100, y, barWidth, barHeight)

      // Add glow
      ctx.shadowBlur = 10
      ctx.shadowColor = values[index] >= 0 ? this.colors.neonGreen : this.colors.neonRed
      ctx.fillRect(padding + 100, y, barWidth, barHeight)
      ctx.shadowBlur = 0

      // Draw value
      ctx.fillStyle = this.colors.text
      ctx.textAlign = 'left'
      ctx.fillText(
        `$${this.formatNumber(values[index])}`,
        padding + 110 + barWidth,
        y + barHeight / 2 + 5
      )
    })
  }

  private drawGrid(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    padding: number
  ): void {
    ctx.strokeStyle = this.colors.grid
    ctx.lineWidth = 1
    ctx.setLineDash([5, 5])

    // Horizontal lines
    const hLines = 5
    for (let i = 0; i <= hLines; i++) {
      const y = padding + ((height - padding * 2) / hLines) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    // Vertical lines
    const vLines = 10
    for (let i = 0; i <= vLines; i++) {
      const x = padding + ((width - padding * 2) / vLines) * i
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.stroke()
    }

    ctx.setLineDash([])
  }

  private drawAxisLabels(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    padding: number,
    labels: string[]
  ): void {
    ctx.fillStyle = this.colors.text
    ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif'
    ctx.textAlign = 'center'

    const step = Math.ceil(labels.length / 10)
    labels.forEach((label, index) => {
      if (index % step === 0) {
        const x = padding + ((width - padding * 2) / labels.length) * index
        ctx.fillText(label, x, height - padding + 20)
      }
    })
  }

  private formatNumber(value: number): string {
    if (Math.abs(value) >= 1000) {
      return `${(value / 1000).toFixed(1)}k`
    }
    return value.toFixed(2)
  }

  private switchTimePeriod(period: string): void {
    // Update chart data based on selected period
    switch (period) {
      case 'daily':
        this.metrics.dailyPnL = this.generateMockData(30, 'day')
        break
      case 'weekly':
        this.metrics.weeklyPnL = this.generateMockData(12, 'week')
        break
      case 'monthly':
        this.metrics.monthlyPnL = this.generateMockData(12, 'month')
        break
    }
    this.drawAllCharts()
  }

  private generateMockData(count: number, unit: string): ChartData {
    const labels: string[] = []
    const values: number[] = []
    const timestamps: Date[] = []

    for (let i = 0; i < count; i++) {
      const date = new Date()
      if (unit === 'day') date.setDate(date.getDate() - (count - i))
      else if (unit === 'week') date.setDate(date.getDate() - (count - i) * 7)
      else if (unit === 'month') date.setMonth(date.getMonth() - (count - i))

      labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }))
      values.push((Math.random() - 0.5) * 1000)
      timestamps.push(date)
    }

    return { labels, values, timestamps }
  }

  private handleResize(): void {
    // Debounce resize
    const resizeTimer = this.animationFrames.get('resize')
    if (resizeTimer) {
      cancelAnimationFrame(resizeTimer)
    }

    const frame = requestAnimationFrame(() => {
      this.charts.forEach((canvas, _id) => {
        this.setupCanvas(canvas)
      })
      this.drawAllCharts()
    })

    this.animationFrames.set('resize', frame)
  }

  public updateData(positions: Position[], riskMetrics: RiskMetrics): void {
    // Calculate performance metrics from positions and risk data
    if (positions && positions.length > 0) {
      const wins = positions.filter((p) => p.pnl > 0)
      const losses = positions.filter((p) => p.pnl < 0)

      this.metrics.totalPnL = positions.reduce((sum, p) => sum + p.pnl, 0)
      this.metrics.totalPnLPercent =
        positions.reduce((sum, p) => sum + p.pnl_percent, 0) / positions.length
      this.metrics.winningTrades = wins.length
      this.metrics.losingTrades = losses.length
      this.metrics.totalTrades = positions.length
      this.metrics.winRate = (wins.length / positions.length) * 100
      this.metrics.avgWin =
        wins.length > 0 ? wins.reduce((sum, p) => sum + p.pnl, 0) / wins.length : 0
      this.metrics.avgLoss =
        losses.length > 0 ? Math.abs(losses.reduce((sum, p) => sum + p.pnl, 0) / losses.length) : 0
    }

    if (riskMetrics) {
      this.metrics.maxDrawdown = riskMetrics.max_drawdown * 100
      // Sharpe ratio would need to be calculated from historical returns
    }

    // Update UI
    this.updateMetricsDisplay()
    this.drawAllCharts()
  }

  private updateMetricsDisplay(): void {
    const updates = [
      {
        selector: '[data-metric="total-pnl"]',
        value: `$${this.formatNumber(this.metrics.totalPnL)}`,
      },
      {
        selector: '[data-metric="total-pnl-percent"]',
        value: `${this.metrics.totalPnLPercent >= 0 ? '+' : ''}${this.metrics.totalPnLPercent.toFixed(2)}%`,
      },
      { selector: '[data-metric="win-rate"]', value: `${Math.round(this.metrics.winRate)}%` },
      {
        selector: '[data-metric="win-loss"]',
        value: `${this.metrics.winningTrades}W / ${this.metrics.losingTrades}L`,
      },
      { selector: '[data-metric="avg-win"]', value: `$${this.formatNumber(this.metrics.avgWin)}` },
      {
        selector: '[data-metric="avg-loss"]',
        value: `$${this.formatNumber(this.metrics.avgLoss)}`,
      },
      {
        selector: '[data-metric="max-drawdown"]',
        value: `${this.metrics.maxDrawdown.toFixed(2)}%`,
      },
      { selector: '[data-metric="sharpe-ratio"]', value: this.metrics.sharpeRatio.toFixed(2) },
    ]

    updates.forEach((update) => {
      const element = this.container.querySelector(update.selector)
      if (element) {
        element.textContent = update.value
      }
    })

    // Update colors based on values
    const pnlElement = this.container.querySelector('[data-metric="total-pnl"]')
    const pnlPercentElement = this.container.querySelector('[data-metric="total-pnl-percent"]')

    if (pnlElement && pnlPercentElement) {
      const colorClass = this.metrics.totalPnL >= 0 ? 'positive' : 'negative'
      pnlElement.className = `metric-value ${colorClass}`
      pnlPercentElement.className = `metric-change ${colorClass}`
    }
  }

  private applyStyles(): void {
    const style = document.createElement('style')
    style.textContent = `
      .performance-charts-container {
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: var(--spacing-lg);
        padding: var(--spacing-lg);
        background: var(--bg-secondary);
        border-radius: var(--radius-lg);
        overflow: auto;
      }

      .performance-summary {
        background: var(--bg-tertiary);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        border: 1px solid var(--border-color);
      }

      .summary-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--spacing-lg);
      }

      .summary-header h3 {
        margin: 0;
        font-size: var(--font-size-xl);
        font-weight: var(--font-weight-semibold);
        color: var(--text-primary);
      }

      .time-selector {
        display: flex;
        gap: var(--spacing-xs);
        background: var(--bg-elevated);
        padding: 2px;
        border-radius: var(--radius-md);
      }

      .time-btn {
        padding: var(--spacing-xs) var(--spacing-md);
        background: transparent;
        border: none;
        color: var(--text-secondary);
        font-size: var(--font-size-sm);
        cursor: pointer;
        border-radius: var(--radius-sm);
        transition: var(--transition-fast);
      }

      .time-btn:hover {
        color: var(--text-primary);
      }

      .time-btn.active {
        background: var(--accent-primary);
        color: var(--text-white);
      }

      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: var(--spacing-md);
      }

      .metric-card {
        background: var(--bg-elevated);
        padding: var(--spacing-md);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        transition: var(--transition-fast);
      }

      .metric-card:hover {
        border-color: var(--border-light);
        transform: translateY(-2px);
      }

      .metric-label {
        font-size: var(--font-size-sm);
        color: var(--text-secondary);
        margin-bottom: var(--spacing-xs);
      }

      .metric-value {
        font-size: var(--font-size-2xl);
        font-weight: var(--font-weight-semibold);
        color: var(--text-primary);
        margin-bottom: var(--spacing-xs);
      }

      .metric-value.positive {
        color: var(--bull-color);
      }

      .metric-value.negative {
        color: var(--bear-color);
      }

      .metric-change {
        font-size: var(--font-size-sm);
        font-weight: var(--font-weight-medium);
      }

      .metric-change.positive {
        color: var(--bull-color);
      }

      .metric-change.negative {
        color: var(--bear-color);
      }

      .metric-subtext {
        font-size: var(--font-size-sm);
        color: var(--text-muted);
      }

      .charts-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--spacing-lg);
        flex: 1;
      }

      .chart-wrapper {
        background: var(--bg-tertiary);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        border: 1px solid var(--border-color);
        display: flex;
        flex-direction: column;
      }

      .chart-wrapper.full-width {
        grid-column: 1 / -1;
      }

      .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--spacing-md);
      }

      .chart-header h4 {
        margin: 0;
        font-size: var(--font-size-lg);
        font-weight: var(--font-weight-medium);
        color: var(--text-primary);
      }

      .chart-info {
        font-size: var(--font-size-sm);
        color: var(--text-secondary);
      }

      .chart-legend {
        display: flex;
        gap: var(--spacing-md);
      }

      .legend-item {
        display: flex;
        align-items: center;
        gap: var(--spacing-xs);
        font-size: var(--font-size-sm);
        color: var(--text-secondary);
      }

      .legend-item.profit .legend-dot {
        background: var(--bull-color);
      }

      .legend-item.loss .legend-dot {
        background: var(--bear-color);
      }

      .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
      }

      .chart-container {
        position: relative;
        flex: 1;
        min-height: 200px;
      }

      .chart-container canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100% !important;
        height: 100% !important;
      }

      .donut-center {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        pointer-events: none;
      }

      .donut-value {
        font-size: var(--font-size-3xl);
        font-weight: var(--font-weight-bold);
        color: var(--text-primary);
      }

      .donut-label {
        font-size: var(--font-size-sm);
        color: var(--text-secondary);
      }

      @media (max-width: 1200px) {
        .charts-grid {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 768px) {
        .metrics-grid {
          grid-template-columns: 1fr 1fr;
        }
        
        .performance-charts-container {
          padding: var(--spacing-md);
        }
      }
    `

    document.head.appendChild(style)
  }

  public destroy(): void {
    // Clean up
    this.animationFrames.forEach((frame) => cancelAnimationFrame(frame))
    this.resizeObserver.disconnect()
    this.charts.clear()
    this.contexts.clear()
  }
}
