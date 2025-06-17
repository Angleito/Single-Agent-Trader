/**
 * Advanced Manual Trading Interface Component
 *
 * Provides sophisticated manual trading controls with:
 * - Real-time position management
 * - Risk validation and warnings
 * - Order preview and confirmation
 * - Emergency controls
 * - Trading history
 */

import type { RiskMetrics, TradingModeConfig } from '../types'

export interface TradeRequest {
  action: 'buy' | 'sell' | 'close'
  symbol: string
  size_percentage: number
  reason?: string
}

export interface RiskLimits {
  max_position_size: number
  stop_loss_percentage: number
  max_daily_loss: number
}

export interface TradingCommand {
  id: string
  type: string
  status: 'pending' | 'executed' | 'failed' | 'cancelled'
  timestamp: string
  parameters?: any
}

export class ManualTradingInterface {
  private container: HTMLElement
  private apiBaseUrl: string
  private currentPosition: Position | null = null
  private currentMarketData: MarketData | null = null
  private currentRiskMetrics: RiskMetrics | null = null
  private pendingCommands: TradingCommand[] = []
  private _onTradeExecuted?: (trade: TradeRequest) => void
  private _onError?: (error: string) => void
  private isEmergencyMode = false
  private connectionStatus: 'connected' | 'disconnected' | 'error' = 'disconnected'
  private tradingModeConfig: TradingModeConfig | null = null

  constructor(containerId: string, apiBaseUrl: string) {
    const container = document.getElementById(containerId)
    if (!container) {
      throw new Error(`Container element with ID ${containerId} not found`)
    }

    this.container = container
    this.apiBaseUrl = apiBaseUrl
    this.render()
    this.startCommandPolling()
  }

  /**
   * Update position data
   */
  public updatePosition(position: Position): void {
    this.currentPosition = position
    this.updatePositionDisplay()
    this.validateRiskLimits()
  }

  /**
   * Update market data
   */
  public updateMarketData(marketData: MarketData): void {
    this.currentMarketData = marketData
    this.updateMarketDisplay()
    this.updateUnrealizedPnL()
  }

  /**
   * Update risk metrics
   */
  public updateRiskMetrics(riskMetrics: RiskMetrics): void {
    this.currentRiskMetrics = riskMetrics
    this.updateRiskDisplay()
  }

  /**
   * Set connection status
   */
  public setConnectionStatus(status: 'connected' | 'disconnected' | 'error'): void {
    this.connectionStatus = status
    this.updateConnectionIndicator()
  }

  /**
   * Set trading mode configuration
   */
  public setTradingModeConfig(config: TradingModeConfig): void {
    this.tradingModeConfig = config
    this.render() // Re-render to update UI based on trading mode
  }

  /**
   * Set emergency mode
   */
  public setEmergencyMode(emergency: boolean): void {
    this.isEmergencyMode = emergency
    this.updateEmergencyState()
  }

  /**
   * Set event handlers
   */
  public onTradeExecuted(callback: (trade: TradeRequest) => void): void {
    this._onTradeExecuted = callback
  }

  public onError(callback: (error: string) => void): void {
    this._onError = callback
  }

  /**
   * Render the main interface
   */
  private render(): void {
    const isSpot = this.tradingModeConfig?.trading_mode === 'spot'
    const isFutures = this.tradingModeConfig?.futures_enabled ?? false

    this.container.innerHTML = `
      <div class="manual-trading-interface">
        <!-- Header with Status -->
        <div class="trading-header">
          <div class="header-title">
            <h3>Manual Trading Control</h3>
            <div class="trading-mode-badge">${isSpot ? '💰 Spot Trading' : '📊 Futures Trading'}</div>
            <div class="connection-status disconnected" id="connection-indicator">
              <span class="status-dot"></span>
              <span class="status-text">Disconnected</span>
            </div>
          </div>
          <div class="emergency-controls">
            <button class="emergency-stop-btn" id="emergency-stop">
              🚨 EMERGENCY STOP
            </button>
            <button class="pause-trading-btn" id="pause-trading">
              ⏸️ Pause Trading
            </button>
            <button class="resume-trading-btn" id="resume-trading" disabled>
              ▶️ Resume Trading
            </button>
          </div>
        </div>

        <!-- Quick Stats Dashboard -->
        <div class="quick-stats">
          <div class="stat-card">
            <div class="stat-label">Current Position</div>
            <div class="stat-value" id="current-position">No Position</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Unrealized P&L</div>
            <div class="stat-value" id="unrealized-pnl">$0.00</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Current Price</div>
            <div class="stat-value" id="current-price">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Daily P&L</div>
            <div class="stat-value" id="daily-pnl">$0.00</div>
          </div>
        </div>

        <!-- Trading Controls -->
        <div class="trading-controls">
          <div class="control-section">
            <h4>Execute Trade</h4>
            <div class="trade-form">
              <div class="form-row">
                <div class="form-group">
                  <label for="trade-symbol">Symbol</label>
                  <select id="trade-symbol" class="form-control">
                    <option value="BTC-USD">BTC-USD</option>
                    <option value="ETH-USD">ETH-USD</option>
                    <option value="DOGE-USD">DOGE-USD</option>
                    <option value="SUI-PERP">SUI-PERP</option>
                  </select>
                </div>
                <div class="form-group">
                  <label for="trade-action">Action</label>
                  <select id="trade-action" class="form-control">
                    <option value="buy">Buy (Long)</option>
                    <option value="sell">Sell (Short)</option>
                    <option value="close">Close Position</option>
                  </select>
                </div>
              </div>

              <div class="form-row">
                <div class="form-group">
                  <label for="position-size">Position Size (%)</label>
                  <div class="size-input-group">
                    <input type="range" id="position-size-slider" min="1" max="100" value="10" class="size-slider">
                    <input type="number" id="position-size" min="0.1" max="100" step="0.1" value="10" class="form-control size-input">
                    <span class="size-unit">%</span>
                  </div>
                  <div class="size-presets">
                    <button class="preset-btn" data-size="5">5%</button>
                    <button class="preset-btn" data-size="10">10%</button>
                    <button class="preset-btn" data-size="25">25%</button>
                    <button class="preset-btn" data-size="50">50%</button>
                  </div>
                </div>
                ${
                  isFutures
                    ? `
                <div class="form-group">
                  <label for="leverage">Leverage</label>
                  <div class="leverage-input-group">
                    <input type="range" id="leverage-slider" min="1" max="${this.tradingModeConfig?.max_leverage ?? 20}" value="${this.tradingModeConfig?.default_leverage ?? 5}" class="leverage-slider">
                    <input type="number" id="leverage" min="1" max="${this.tradingModeConfig?.max_leverage ?? 20}" step="1" value="${this.tradingModeConfig?.default_leverage ?? 5}" class="form-control leverage-input">
                    <span class="leverage-unit">x</span>
                  </div>
                </div>
                `
                    : `
                <div class="form-group">
                  <label for="trade-reason">Reason (Optional)</label>
                  <input type="text" id="trade-reason" placeholder="Manual trade reason..." class="form-control">
                </div>
                `
                }
              </div>

              ${
                isFutures
                  ? `
              <div class="form-row">
                <div class="form-group full-width">
                  <label for="trade-reason">Reason (Optional)</label>
                  <input type="text" id="trade-reason" placeholder="Manual trade reason..." class="form-control">
                </div>
              </div>
              `
                  : ''
              }

              <!-- Risk Validation Display -->
              <div class="risk-validation" id="risk-validation">
                <div class="validation-item">
                  <span class="validation-label">Estimated Cost:</span>
                  <span class="validation-value" id="estimated-cost">$0.00</span>
                </div>
                <div class="validation-item">
                  <span class="validation-label">Trading Fee:</span>
                  <span class="validation-value" id="trading-fee">$0.00 (0.00%)</span>
                </div>
                ${
                  isFutures
                    ? `
                <div class="validation-item">
                  <span class="validation-label">Margin Required:</span>
                  <span class="validation-value" id="margin-required">$0.00</span>
                </div>
                <div class="validation-item">
                  <span class="validation-label">Liquidation Price:</span>
                  <span class="validation-value" id="liquidation-price">--</span>
                </div>
                `
                    : `
                <div class="validation-item">
                  <span class="validation-label">Fee Tier:</span>
                  <span class="validation-value" id="fee-tier">Basic</span>
                </div>
                <div class="validation-item">
                  <span class="validation-label">Min Profit Move:</span>
                  <span class="validation-value" id="min-profit-move">0.00%</span>
                </div>
                `
                }
                <div class="validation-item">
                  <span class="validation-label">Risk Level:</span>
                  <span class="validation-value risk-low" id="risk-level">Low</span>
                </div>
                <div class="validation-item">
                  <span class="validation-label">Portfolio Impact:</span>
                  <span class="validation-value" id="portfolio-impact">0.0%</span>
                </div>
              </div>

              <!-- Execute Button -->
              <div class="execute-section">
                <button class="execute-btn" id="execute-trade" disabled>
                  <span class="btn-icon">🎯</span>
                  <span class="btn-text">Execute Trade</span>
                </button>
                <div class="execution-warning" id="execution-warning">
                  ⚠️ This will execute a real trade! Review all parameters carefully.
                </div>
              </div>
            </div>
          </div>

          <!-- Risk Management Controls -->
          <div class="control-section">
            <h4>Risk Management</h4>
            <div class="risk-controls">
              <div class="risk-limit-group">
                <label for="max-position-size">Max Position Size (%)</label>
                <input type="number" id="max-position-size" min="1" max="100" step="1" value="50" class="form-control">
              </div>
              <div class="risk-limit-group">
                <label for="stop-loss-pct">Stop Loss (%)</label>
                <input type="number" id="stop-loss-pct" min="0.1" max="50" step="0.1" value="2.0" class="form-control">
              </div>
              <div class="risk-limit-group">
                <label for="max-daily-loss">Max Daily Loss ($)</label>
                <input type="number" id="max-daily-loss" min="1" max="10000" step="1" value="500" class="form-control">
              </div>
              <button class="update-risk-btn" id="update-risk-limits">
                Update Risk Limits
              </button>
            </div>
          </div>
        </div>

        <!-- Command Queue and History -->
        <div class="command-section">
          <div class="section-tabs">
            <button class="tab-btn active" data-tab="queue">Command Queue</button>
            <button class="tab-btn" data-tab="history">Trade History</button>
          </div>

          <div class="tab-content active" id="queue-tab">
            <div class="command-queue">
              <div class="queue-header">
                <h5>Pending Commands</h5>
                <button class="clear-queue-btn" id="clear-queue">Clear All</button>
              </div>
              <div class="command-list" id="command-list">
                <div class="empty-state">No pending commands</div>
              </div>
            </div>
          </div>

          <div class="tab-content" id="history-tab">
            <div class="trade-history">
              <div class="history-header">
                <h5>Recent Trades</h5>
                <button class="refresh-history-btn" id="refresh-history">🔄 Refresh</button>
              </div>
              <div class="history-list" id="history-list">
                <div class="empty-state">No recent trades</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `

    this.attachEventListeners()
    this.updateTradeValidation()
  }

  /**
   * Attach event listeners
   */
  private attachEventListeners(): void {
    // Emergency controls
    const emergencyStopBtn = document.getElementById('emergency-stop')
    const pauseTradingBtn = document.getElementById('pause-trading')
    const resumeTradingBtn = document.getElementById('resume-trading')

    emergencyStopBtn?.addEventListener('click', () => this.executeEmergencyStop())
    pauseTradingBtn?.addEventListener('click', () => this.pauseTrading())
    resumeTradingBtn?.addEventListener('click', () => this.resumeTrading())

    // Trade form controls
    const positionSizeSlider = document.getElementById('position-size-slider') as HTMLInputElement
    const positionSizeInput = document.getElementById('position-size') as HTMLInputElement
    const tradeActionSelect = document.getElementById('trade-action') as HTMLSelectElement
    const tradeSymbolSelect = document.getElementById('trade-symbol') as HTMLSelectElement

    // Sync slider and input
    positionSizeSlider?.addEventListener('input', (e) => {
      const value = (e.target as HTMLInputElement).value
      if (positionSizeInput) positionSizeInput.value = value
      this.updateTradeValidation()
    })

    positionSizeInput?.addEventListener('input', (e) => {
      const value = (e.target as HTMLInputElement).value
      if (positionSizeSlider) positionSizeSlider.value = value
      this.updateTradeValidation()
    })

    // Size presets
    const presetBtns = document.querySelectorAll('.preset-btn')
    presetBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const size = (e.target as HTMLElement).dataset.size
        if (size && positionSizeSlider && positionSizeInput) {
          positionSizeSlider.value = size
          positionSizeInput.value = size
          this.updateTradeValidation()
        }
      })
    })

    // Form change validation
    tradeActionSelect?.addEventListener('change', () => this.updateTradeValidation())
    tradeSymbolSelect?.addEventListener('change', () => this.updateTradeValidation())

    // Execute trade
    const executeBtn = document.getElementById('execute-trade')
    executeBtn?.addEventListener('click', () => this.executeTrade())

    // Risk management
    const updateRiskBtn = document.getElementById('update-risk-limits')
    updateRiskBtn?.addEventListener('click', () => this.updateRiskLimits())

    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn')
    tabBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const tabName = (e.target as HTMLElement).dataset.tab
        if (tabName) this.switchTab(tabName)
      })
    })

    // Queue management
    const clearQueueBtn = document.getElementById('clear-queue')
    const refreshHistoryBtn = document.getElementById('refresh-history')

    clearQueueBtn?.addEventListener('click', () => this.clearCommandQueue())
    refreshHistoryBtn?.addEventListener('click', () => this.refreshTradeHistory())
  }

  /**
   * Update trade validation and risk assessment
   */
  private updateTradeValidation(): void {
    const sizeInput = document.getElementById('position-size') as HTMLInputElement
    const actionSelect = document.getElementById('trade-action') as HTMLSelectElement
    const symbolSelect = document.getElementById('trade-symbol') as HTMLSelectElement
    const executeBtn = document.getElementById('execute-trade') as HTMLButtonElement

    if (!sizeInput || !actionSelect || !symbolSelect || !executeBtn) return

    const size = parseFloat(sizeInput.value)
    const _action = actionSelect.value
    const _symbol = symbolSelect.value

    // Calculate estimated cost
    const currentPrice = this.currentMarketData?.price ?? 0
    const estimatedCost = currentPrice * (size / 100) * 1000 // Assuming $1000 portfolio

    // Calculate fees based on trading mode
    let feeRate = 0.012 // Default to spot taker fee (1.2%)
    let feeAmount = 0
    let feeTier = 'Basic'
    let minProfitMove = 0

    if (this.tradingModeConfig) {
      if (this.tradingModeConfig.mode === 'spot') {
        // Spot trading fees (market orders use taker rate)
        feeRate = this.tradingModeConfig.taker_fee_rate ?? 0.012
        feeAmount = estimatedCost * feeRate * 2 // Round trip (entry + exit)
        minProfitMove = feeRate * 2 * 100 // As percentage

        // Determine fee tier based on rate
        if (feeRate >= 0.012) feeTier = 'Basic'
        else if (feeRate >= 0.004) feeTier = 'Standard'
        else if (feeRate >= 0.002) feeTier = 'Pro'
        else feeTier = 'VIP'
      } else {
        // Futures trading
        feeRate = this.tradingModeConfig.futures_fee_rate ?? 0.0015
        const leverage = parseFloat(
          (document.getElementById('leverage') as HTMLInputElement)?.value ?? '5'
        )
        feeAmount = estimatedCost * feeRate * 2 // Round trip
        minProfitMove = (feeRate * 2 * 100) / leverage // Adjusted for leverage
      }
    }

    // Determine risk level
    let riskLevel = 'low'
    let riskClass = 'risk-low'

    if (size > 50) {
      riskLevel = 'high'
      riskClass = 'risk-high'
    } else if (size > 25) {
      riskLevel = 'medium'
      riskClass = 'risk-medium'
    }

    // Update display
    const estimatedCostEl = document.getElementById('estimated-cost')
    const tradingFeeEl = document.getElementById('trading-fee')
    const feeTierEl = document.getElementById('fee-tier')
    const minProfitMoveEl = document.getElementById('min-profit-move')
    const riskLevelEl = document.getElementById('risk-level')
    const portfolioImpactEl = document.getElementById('portfolio-impact')

    if (estimatedCostEl) estimatedCostEl.textContent = `$${estimatedCost.toFixed(2)}`
    if (tradingFeeEl)
      tradingFeeEl.textContent = `$${feeAmount.toFixed(2)} (${(feeRate * 100).toFixed(2)}%)`
    if (feeTierEl) feeTierEl.textContent = feeTier
    if (minProfitMoveEl) minProfitMoveEl.textContent = `${minProfitMove.toFixed(3)}%`
    if (riskLevelEl) {
      riskLevelEl.textContent = riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)
      riskLevelEl.className = `validation-value ${riskClass}`
    }
    if (portfolioImpactEl) portfolioImpactEl.textContent = `${size.toFixed(1)}%`

    // Enable/disable execute button
    const canExecute =
      this.connectionStatus === 'connected' &&
      !this.isEmergencyMode &&
      size > 0 &&
      size <= 100 &&
      currentPrice > 0

    executeBtn.disabled = !canExecute

    if (!canExecute) {
      executeBtn.classList.add('disabled')
    } else {
      executeBtn.classList.remove('disabled')
    }
  }

  /**
   * Execute a manual trade
   */
  private async executeTrade(): Promise<void> {
    const sizeInput = document.getElementById('position-size') as HTMLInputElement
    const actionSelect = document.getElementById('trade-action') as HTMLSelectElement
    const symbolSelect = document.getElementById('trade-symbol') as HTMLSelectElement
    const reasonInput = document.getElementById('trade-reason') as HTMLInputElement
    const executeBtn = document.getElementById('execute-trade') as HTMLButtonElement

    if (!sizeInput || !actionSelect || !symbolSelect) return

    const tradeRequest: TradeRequest = {
      action: actionSelect.value as 'buy' | 'sell' | 'close',
      symbol: symbolSelect.value,
      size_percentage: parseFloat(sizeInput.value),
      reason: reasonInput?.value ?? 'Manual trade from dashboard',
    }

    // Show confirmation dialog
    const confirmed = await this.showTradeConfirmation(tradeRequest)
    if (!confirmed) return

    executeBtn.disabled = true
    executeBtn.innerHTML = '<span class="loading-spinner"></span> Executing...'

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/manual-trade`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          action: tradeRequest.action,
          symbol: tradeRequest.symbol,
          size_percentage: tradeRequest.size_percentage.toString(),
          reason: tradeRequest.reason ?? '',
        }),
      })

      const result = await response.json()

      if (response.ok) {
        this.showSuccessMessage(`Trade command sent: ${result.message}`)
        this._onTradeExecuted?.(tradeRequest)
        this.refreshCommandQueue()

        // Reset form
        if (reasonInput) reasonInput.value = ''
      } else {
        throw new Error(result.detail ?? 'Trade execution failed')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.showErrorMessage(`Trade failed: ${errorMessage}`)
      this._onError?.(errorMessage)
    } finally {
      executeBtn.disabled = false
      executeBtn.innerHTML =
        '<span class="btn-icon">🎯</span><span class="btn-text">Execute Trade</span>'
      this.updateTradeValidation()
    }
  }

  /**
   * Show trade confirmation dialog
   */
  private async showTradeConfirmation(trade: TradeRequest): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = document.createElement('div')
      modal.className = 'trade-confirmation-modal'
      modal.innerHTML = `
        <div class="modal-overlay">
          <div class="modal-content">
            <h3>Confirm Trade Execution</h3>
            <div class="trade-summary">
              <div class="summary-item">
                <strong>Action:</strong> ${trade.action.toUpperCase()}
              </div>
              <div class="summary-item">
                <strong>Symbol:</strong> ${trade.symbol}
              </div>
              <div class="summary-item">
                <strong>Size:</strong> ${trade.size_percentage}% of portfolio
              </div>
              <div class="summary-item">
                <strong>Reason:</strong> ${trade.reason ?? 'Manual trade'}
              </div>
            </div>
            <div class="warning-message">
              ⚠️ This will execute a real trade with actual money!
            </div>
            <div class="modal-actions">
              <button class="cancel-btn" id="cancel-trade">Cancel</button>
              <button class="confirm-btn" id="confirm-trade">Execute Trade</button>
            </div>
          </div>
        </div>
      `

      document.body.appendChild(modal)

      const cancelBtn = modal.querySelector('#cancel-trade')
      const confirmBtn = modal.querySelector('#confirm-trade')

      const cleanup = () => {
        document.body.removeChild(modal)
      }

      cancelBtn?.addEventListener('click', () => {
        cleanup()
        resolve(false)
      })

      confirmBtn?.addEventListener('click', () => {
        cleanup()
        resolve(true)
      })

      // Close on overlay click
      modal.querySelector('.modal-overlay')?.addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
          cleanup()
          resolve(false)
        }
      })
    })
  }

  /**
   * Execute emergency stop
   */
  private async executeEmergencyStop(): Promise<void> {
    const confirmed = confirm(
      'Are you sure you want to execute an EMERGENCY STOP? This will immediately halt all trading and close positions.'
    )
    if (!confirmed) return

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/emergency-stop`, {
        method: 'POST',
      })

      const result = await response.json()

      if (response.ok) {
        this.setEmergencyMode(true)
        this.showSuccessMessage('Emergency stop executed successfully')
      } else {
        throw new Error(result.detail ?? 'Emergency stop failed')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.showErrorMessage(`Emergency stop failed: ${errorMessage}`)
    }
  }

  /**
   * Pause trading
   */
  private async pauseTrading(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/pause-trading`, {
        method: 'POST',
      })

      const result = await response.json()

      if (response.ok) {
        this.showSuccessMessage('Trading paused successfully')
        this.updateTradingControlsState('paused')
      } else {
        throw new Error(result.detail ?? 'Pause trading failed')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.showErrorMessage(`Pause trading failed: ${errorMessage}`)
    }
  }

  /**
   * Resume trading
   */
  private async resumeTrading(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/resume-trading`, {
        method: 'POST',
      })

      const result = await response.json()

      if (response.ok) {
        this.setEmergencyMode(false)
        this.showSuccessMessage('Trading resumed successfully')
        this.updateTradingControlsState('active')
      } else {
        throw new Error(result.detail ?? 'Resume trading failed')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.showErrorMessage(`Resume trading failed: ${errorMessage}`)
    }
  }

  /**
   * Update risk limits
   */
  private async updateRiskLimits(): Promise<void> {
    const maxPositionSizeInput = document.getElementById('max-position-size') as HTMLInputElement
    const stopLossPctInput = document.getElementById('stop-loss-pct') as HTMLInputElement
    const maxDailyLossInput = document.getElementById('max-daily-loss') as HTMLInputElement

    if (!maxPositionSizeInput || !stopLossPctInput || !maxDailyLossInput) return

    const params = new URLSearchParams()

    if (maxPositionSizeInput.value) {
      params.append('max_position_size', maxPositionSizeInput.value)
    }
    if (stopLossPctInput.value) {
      params.append('stop_loss_percentage', stopLossPctInput.value)
    }
    if (maxDailyLossInput.value) {
      params.append('max_daily_loss', maxDailyLossInput.value)
    }

    try {
      const response = await fetch(
        `${this.apiBaseUrl}/api/bot/commands/update-risk-limits?${params}`,
        {
          method: 'POST',
        }
      )

      const result = await response.json()

      if (response.ok) {
        this.showSuccessMessage('Risk limits updated successfully')
      } else {
        throw new Error(result.detail ?? 'Risk limit update failed')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.showErrorMessage(`Risk limit update failed: ${errorMessage}`)
    }
  }

  /**
   * Update various displays
   */
  private updatePositionDisplay(): void {
    const positionEl = document.getElementById('current-position')
    if (!positionEl || !this.currentPosition) return

    const position = this.currentPosition
    const size = position.quantity ?? position.size ?? 0

    if (size === 0) {
      positionEl.textContent = 'No Position'
      positionEl.className = 'stat-value neutral'
    } else {
      positionEl.textContent = `${position.side} ${Math.abs(size).toFixed(4)}`
      positionEl.className = `stat-value ${position.side?.toLowerCase() === 'long' ? 'positive' : 'negative'}`
    }
  }

  private updateMarketDisplay(): void {
    const priceEl = document.getElementById('current-price')
    if (!priceEl || !this.currentMarketData) return

    priceEl.textContent = `$${this.currentMarketData.price.toFixed(2)}`
  }

  private updateUnrealizedPnL(): void {
    const pnlEl = document.getElementById('unrealized-pnl')
    if (!pnlEl || !this.currentPosition) return

    const unrealizedPnL = this.currentPosition.unrealized_pnl ?? 0
    pnlEl.textContent = `$${unrealizedPnL.toFixed(2)}`
    pnlEl.className = `stat-value ${unrealizedPnL >= 0 ? 'positive' : 'negative'}`
  }

  private updateRiskDisplay(): void {
    const dailyPnLEl = document.getElementById('daily-pnl')
    if (!dailyPnLEl || !this.currentRiskMetrics) return

    const dailyPnL = this.currentRiskMetrics.daily_pnl ?? 0
    dailyPnLEl.textContent = `$${dailyPnL.toFixed(2)}`
    dailyPnLEl.className = `stat-value ${dailyPnL >= 0 ? 'positive' : 'negative'}`
  }

  private updateConnectionIndicator(): void {
    const indicator = document.getElementById('connection-indicator')
    if (!indicator) return

    indicator.className = `connection-status ${this.connectionStatus}`
    const statusText = indicator.querySelector('.status-text')
    if (statusText) {
      statusText.textContent =
        this.connectionStatus.charAt(0).toUpperCase() + this.connectionStatus.slice(1)
    }
  }

  private updateEmergencyState(): void {
    const executeBtn = document.getElementById('execute-trade') as HTMLButtonElement
    const pauseBtn = document.getElementById('pause-trading') as HTMLButtonElement
    const resumeBtn = document.getElementById('resume-trading') as HTMLButtonElement

    if (this.isEmergencyMode) {
      executeBtn?.setAttribute('disabled', 'true')
      pauseBtn?.setAttribute('disabled', 'true')
      resumeBtn?.removeAttribute('disabled')

      this.container.classList.add('emergency-mode')
    } else {
      resumeBtn?.setAttribute('disabled', 'true')
      pauseBtn?.removeAttribute('disabled')

      this.container.classList.remove('emergency-mode')
    }

    this.updateTradeValidation()
  }

  private updateTradingControlsState(state: 'active' | 'paused'): void {
    const pauseBtn = document.getElementById('pause-trading') as HTMLButtonElement
    const resumeBtn = document.getElementById('resume-trading') as HTMLButtonElement

    if (state === 'paused') {
      pauseBtn?.setAttribute('disabled', 'true')
      resumeBtn?.removeAttribute('disabled')
    } else {
      pauseBtn?.removeAttribute('disabled')
      resumeBtn?.setAttribute('disabled', 'true')
    }
  }

  /**
   * Command queue management
   */
  private async startCommandPolling(): Promise<void> {
    const pollCommands = async () => {
      try {
        const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/queue`)
        if (response.ok) {
          const data = await response.json()
          this.pendingCommands = data.pending_commands ?? []
          this.updateCommandQueueDisplay()
        }
      } catch (error) {
        console.warn('Failed to poll command queue:', error)
      }
    }

    // Poll every 2 seconds
    setInterval(pollCommands, 2000)
    pollCommands() // Initial poll
  }

  private updateCommandQueueDisplay(): void {
    const commandList = document.getElementById('command-list')
    if (!commandList) return

    if (this.pendingCommands.length === 0) {
      commandList.innerHTML = '<div class="empty-state">No pending commands</div>'
      return
    }

    commandList.innerHTML = this.pendingCommands
      .map(
        (cmd) => `
      <div class="command-item">
        <div class="command-header">
          <span class="command-type">${cmd.type}</span>
          <span class="command-status status-${cmd.status}">${cmd.status}</span>
          <button class="cancel-command-btn" data-command-id="${cmd.id}">✕</button>
        </div>
        <div class="command-details">
          <small>${new Date(cmd.timestamp).toLocaleTimeString()}</small>
          ${cmd.parameters ? `<div class="command-params">${JSON.stringify(cmd.parameters)}</div>` : ''}
        </div>
      </div>
    `
      )
      .join('')

    // Attach cancel handlers
    const cancelBtns = commandList.querySelectorAll('.cancel-command-btn')
    cancelBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const commandId = (e.target as HTMLElement).dataset.commandId
        if (commandId) this.cancelCommand(commandId)
      })
    })
  }

  private async cancelCommand(commandId: string): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/${commandId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        this.showSuccessMessage('Command cancelled successfully')
        this.refreshCommandQueue()
      } else {
        const result = await response.json()
        throw new Error(result.detail ?? 'Failed to cancel command')
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      this.showErrorMessage(`Cancel failed: ${errorMessage}`)
    }
  }

  private async clearCommandQueue(): Promise<void> {
    for (const cmd of this.pendingCommands) {
      await this.cancelCommand(cmd.id)
    }
  }

  private async refreshCommandQueue(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/queue`)
      if (response.ok) {
        const data = await response.json()
        this.pendingCommands = data.pending_commands ?? []
        this.updateCommandQueueDisplay()
      }
    } catch (error) {
      console.warn('Failed to refresh command queue:', error)
    }
  }

  private async refreshTradeHistory(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/commands/history`)
      if (response.ok) {
        const data = await response.json()
        this.updateTradeHistoryDisplay(data.history ?? [])
      }
    } catch (error) {
      console.warn('Failed to refresh trade history:', error)
    }
  }

  private updateTradeHistoryDisplay(history: any[]): void {
    const historyList = document.getElementById('history-list')
    if (!historyList) return

    if (history.length === 0) {
      historyList.innerHTML = '<div class="empty-state">No recent trades</div>'
      return
    }

    historyList.innerHTML = history
      .map(
        (trade) => `
      <div class="history-item">
        <div class="trade-header">
          <span class="trade-type">${trade.type}</span>
          <span class="trade-status status-${trade.status}">${trade.status}</span>
          <span class="trade-time">${new Date(trade.created_at).toLocaleString()}</span>
        </div>
        ${trade.cancelled_at ? `<div class="trade-cancelled">Cancelled at ${new Date(trade.cancelled_at).toLocaleString()}</div>` : ''}
      </div>
    `
      )
      .join('')
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

    // Refresh data when switching to history tab
    if (tabName === 'history') {
      this.refreshTradeHistory()
    }
  }

  /**
   * Utility methods
   */
  private validateRiskLimits(): void {
    // Add risk validation logic here
    // This would check current position against risk limits
    // and show warnings if approaching limits
  }

  private showSuccessMessage(message: string): void {
    this.showToast(message, 'success')
  }

  private showErrorMessage(message: string): void {
    this.showToast(message, 'error')
  }

  private showToast(message: string, type: 'success' | 'error' | 'warning' = 'success'): void {
    const toast = document.createElement('div')
    toast.className = `toast toast-${type}`
    toast.textContent = message

    document.body.appendChild(toast)

    // Animate in
    requestAnimationFrame(() => {
      toast.classList.add('show')
    })

    // Remove after 3 seconds
    setTimeout(() => {
      toast.classList.remove('show')
      setTimeout(() => {
        document.body.removeChild(toast)
      }, 300)
    }, 3000)
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    // Clear any polling intervals
    // Remove event listeners
    // Clean up DOM
  }
}
