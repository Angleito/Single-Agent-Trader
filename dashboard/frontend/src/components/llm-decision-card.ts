import type { TradeAction, MarketData, VuManchuIndicators } from '../types'

export interface LLMDecisionData {
  action: TradeAction
  marketData: MarketData
  indicators?: VuManchuIndicators
  riskLevel?: 'low' | 'medium' | 'high'
  positionSize?: number
}

export class LLMDecisionCard {
  private container: HTMLElement
  private currentDecision: LLMDecisionData | null = null
  private updateAnimation: Animation | null = null

  constructor(containerId: string) {
    const element = document.getElementById(containerId)
    if (!element) {
      throw new Error(`Container element with id '${containerId}' not found`)
    }
    this.container = element
    this.render()
  }

  /**
   * Initialize the decision card UI
   */
  private render(): void {
    this.container.innerHTML = `
      <div class="llm-decision-card" data-state="empty">
        <div class="card-header">
          <h2 class="card-title">
            <span class="ai-icon">🤖</span>
            AI Trading Decision
          </h2>
          <div class="timestamp-badge">
            <span class="timestamp-label">Last Update</span>
            <span class="timestamp-value">--:--:--</span>
          </div>
        </div>

        <div class="decision-content">
          <!-- Main Decision Display -->
          <div class="decision-main">
            <div class="action-display" data-action="hold">
              <span class="action-text">HOLD</span>
              <span class="action-subtext">Awaiting market analysis...</span>
            </div>

            <div class="confidence-meter">
              <div class="confidence-label">Confidence Level</div>
              <div class="confidence-bar">
                <div class="confidence-fill" style="width: 0%"></div>
                <div class="confidence-markers">
                  <span class="marker" style="left: 25%"></span>
                  <span class="marker" style="left: 50%"></span>
                  <span class="marker" style="left: 75%"></span>
                </div>
              </div>
              <div class="confidence-value">0%</div>
            </div>
          </div>

          <!-- AI Reasoning Section -->
          <div class="reasoning-section">
            <h3 class="section-title">AI Reasoning</h3>
            <div class="reasoning-content">
              <p class="reasoning-text">Waiting for AI analysis...</p>
            </div>
          </div>

          <!-- Market Context Grid -->
          <div class="market-context">
            <h3 class="section-title">Market Context</h3>
            <div class="context-grid">
              <div class="context-item">
                <span class="context-label">Current Price</span>
                <span class="context-value price-value">--</span>
              </div>
              <div class="context-item">
                <span class="context-label">24h Change</span>
                <span class="context-value change-value">--</span>
              </div>
              <div class="context-item">
                <span class="context-label">Risk Level</span>
                <span class="context-value risk-value" data-risk="medium">Medium</span>
              </div>
              <div class="context-item">
                <span class="context-label">Position Size</span>
                <span class="context-value position-value">--</span>
              </div>
            </div>
          </div>

          <!-- Key Indicators -->
          <div class="indicators-section">
            <h3 class="section-title">Key Indicators</h3>
            <div class="indicators-grid">
              <div class="indicator-item">
                <span class="indicator-name">Cipher A</span>
                <span class="indicator-value" data-trend="neutral">--</span>
              </div>
              <div class="indicator-item">
                <span class="indicator-name">Cipher B</span>
                <span class="indicator-value" data-trend="neutral">--</span>
              </div>
              <div class="indicator-item">
                <span class="indicator-name">Wave Trend</span>
                <span class="indicator-value" data-trend="neutral">--</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Update Indicator -->
        <div class="update-indicator" data-updating="false">
          <div class="update-pulse"></div>
          <span class="update-text">Live</span>
        </div>
      </div>
    `

    this.addStyles()
  }

  /**
   * Add component styles
   */
  private addStyles(): void {
    const styleId = 'llm-decision-card-styles'
    if (document.getElementById(styleId)) return

    const style = document.createElement('style')
    style.id = styleId
    style.textContent = `
      .llm-decision-card {
        background: rgba(17, 24, 39, 0.95);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
      }

      .llm-decision-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg,
          transparent 0%,
          rgba(59, 130, 246, 0.5) 50%,
          transparent 100%);
        opacity: 0.8;
      }

      .llm-decision-card[data-state="updating"] {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4),
                    0 0 20px rgba(59, 130, 246, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
      }

      .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(75, 85, 99, 0.3);
      }

      .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e5e7eb;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .ai-icon {
        font-size: 1.5rem;
        animation: pulse 2s ease-in-out infinite;
      }

      @keyframes pulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
      }

      .timestamp-badge {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
      }

      .timestamp-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .timestamp-value {
        font-size: 0.875rem;
        color: #e5e7eb;
        font-family: 'Monaco', 'Consolas', monospace;
      }

      .decision-main {
        margin-bottom: 32px;
      }

      .action-display {
        text-align: center;
        margin-bottom: 24px;
        padding: 24px;
        border-radius: 12px;
        background: rgba(31, 41, 55, 0.5);
        border: 2px solid transparent;
        transition: all 0.3s ease;
      }

      .action-display[data-action="buy"],
      .action-display[data-action="long"] {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
      }

      .action-display[data-action="sell"],
      .action-display[data-action="short"] {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
      }

      .action-display[data-action="hold"] {
        background: rgba(251, 191, 36, 0.1);
        border-color: rgba(251, 191, 36, 0.3);
      }

      .action-text {
        display: block;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
      }

      .action-display[data-action="buy"] .action-text,
      .action-display[data-action="long"] .action-text {
        color: #10b981;
      }

      .action-display[data-action="sell"] .action-text,
      .action-display[data-action="short"] .action-text {
        color: #ef4444;
      }

      .action-display[data-action="hold"] .action-text {
        color: #fbbf24;
      }

      .action-subtext {
        font-size: 0.875rem;
        color: #9ca3af;
      }

      .confidence-meter {
        max-width: 400px;
        margin: 0 auto;
      }

      .confidence-label {
        font-size: 0.875rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 8px;
      }

      .confidence-bar {
        height: 12px;
        background: rgba(31, 41, 55, 0.5);
        border-radius: 6px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(75, 85, 99, 0.3);
      }

      .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        transition: width 0.5s ease;
        position: relative;
      }

      .confidence-markers {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
      }

      .confidence-markers .marker {
        position: absolute;
        top: 0;
        width: 1px;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
      }

      .confidence-value {
        text-align: center;
        font-size: 1.125rem;
        font-weight: 600;
        color: #60a5fa;
        margin-top: 8px;
      }

      .reasoning-section,
      .market-context,
      .indicators-section {
        margin-bottom: 24px;
      }

      .section-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
      }

      .reasoning-content {
        background: rgba(31, 41, 55, 0.3);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid rgba(75, 85, 99, 0.2);
      }

      .reasoning-text {
        font-size: 0.875rem;
        color: #e5e7eb;
        line-height: 1.6;
        margin: 0;
      }

      .context-grid,
      .indicators-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 16px;
      }

      .context-item,
      .indicator-item {
        background: rgba(31, 41, 55, 0.3);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(75, 85, 99, 0.2);
        transition: all 0.2s ease;
      }

      .context-item:hover,
      .indicator-item:hover {
        background: rgba(31, 41, 55, 0.5);
        border-color: rgba(75, 85, 99, 0.3);
      }

      .context-label,
      .indicator-name {
        display: block;
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 4px;
      }

      .context-value,
      .indicator-value {
        display: block;
        font-size: 1rem;
        font-weight: 600;
        color: #e5e7eb;
      }

      .change-value.positive {
        color: #10b981;
      }

      .change-value.negative {
        color: #ef4444;
      }

      .risk-value[data-risk="low"] {
        color: #10b981;
      }

      .risk-value[data-risk="medium"] {
        color: #fbbf24;
      }

      .risk-value[data-risk="high"] {
        color: #ef4444;
      }

      .indicator-value[data-trend="bullish"] {
        color: #10b981;
      }

      .indicator-value[data-trend="bearish"] {
        color: #ef4444;
      }

      .indicator-value[data-trend="neutral"] {
        color: #9ca3af;
      }

      .update-indicator {
        position: absolute;
        top: 24px;
        right: 24px;
        display: flex;
        align-items: center;
        gap: 6px;
        opacity: 0;
        transition: opacity 0.3s ease;
      }

      .update-indicator[data-updating="true"] {
        opacity: 1;
      }

      .update-pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: updatePulse 1.5s ease-in-out infinite;
      }

      @keyframes updatePulse {
        0%, 100% {
          transform: scale(1);
          opacity: 1;
        }
        50% {
          transform: scale(1.5);
          opacity: 0.5;
        }
      }

      .update-text {
        font-size: 0.75rem;
        color: #10b981;
        font-weight: 500;
      }

      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .llm-decision-card[data-state="updating"] .decision-content {
        animation: fadeIn 0.5s ease;
      }
    `

    document.head.appendChild(style)
  }

  /**
   * Update the decision card with new data
   */
  public updateDecision(data: LLMDecisionData): void {
    this.currentDecision = data

    // Show updating state
    this.setUpdatingState(true)

    // Update action display
    this.updateActionDisplay(data.action)

    // Update confidence meter
    this.updateConfidenceMeter(data.action.confidence)

    // Update reasoning
    this.updateReasoning(data.action.reasoning)

    // Update market context
    this.updateMarketContext(data)

    // Update indicators
    if (data.indicators) {
      this.updateIndicators(data.indicators)
    }

    // Update timestamp
    this.updateTimestamp()

    // Remove updating state after animation
    setTimeout(() => {
      this.setUpdatingState(false)
    }, 1000)
  }

  /**
   * Update the action display
   */
  private updateActionDisplay(action: TradeAction): void {
    const actionDisplay = this.container.querySelector('.action-display') as HTMLElement
    const actionText = this.container.querySelector('.action-text') as HTMLElement
    const actionSubtext = this.container.querySelector('.action-subtext') as HTMLElement

    if (!actionDisplay || !actionText || !actionSubtext) return

    const actionType = action.action.toLowerCase()
    actionDisplay.setAttribute('data-action', actionType)
    actionText.textContent = action.action.toUpperCase()

    // Set appropriate subtext
    const subtextMap: Record<string, string> = {
      buy: 'Bullish signal detected',
      sell: 'Bearish signal detected',
      hold: 'No clear signal',
      long: 'Opening long position',
      short: 'Opening short position',
    }

    actionSubtext.textContent = subtextMap[actionType] || 'Processing...'
  }

  /**
   * Update the confidence meter
   */
  private updateConfidenceMeter(confidence: number): void {
    const fill = this.container.querySelector('.confidence-fill') as HTMLElement
    const value = this.container.querySelector('.confidence-value') as HTMLElement

    if (!fill || !value) return

    const percentage = Math.round(confidence * 100)
    fill.style.width = `${percentage}%`
    value.textContent = `${percentage}%`

    // Update fill color based on confidence level
    if (percentage >= 80) {
      fill.style.background = 'linear-gradient(90deg, #10b981 0%, #34d399 100%)'
    } else if (percentage >= 60) {
      fill.style.background = 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)'
    } else {
      fill.style.background = 'linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)'
    }
  }

  /**
   * Update the reasoning section
   */
  private updateReasoning(reasoning: string): void {
    const reasoningText = this.container.querySelector('.reasoning-text') as HTMLElement
    if (!reasoningText) return

    reasoningText.textContent = reasoning
  }

  /**
   * Update market context
   */
  private updateMarketContext(data: LLMDecisionData): void {
    const priceElement = this.container.querySelector('.price-value') as HTMLElement
    const changeElement = this.container.querySelector('.change-value') as HTMLElement
    const riskElement = this.container.querySelector('.risk-value') as HTMLElement
    const positionElement = this.container.querySelector('.position-value') as HTMLElement

    if (priceElement && data.marketData.price) {
      priceElement.textContent = `$${data.marketData.price.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`
    }

    if (changeElement && data.marketData.change_percent_24h !== undefined) {
      const change = data.marketData.change_percent_24h
      changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`
      changeElement.className = `context-value change-value ${change >= 0 ? 'positive' : 'negative'}`
    }

    if (riskElement && data.riskLevel) {
      riskElement.textContent = data.riskLevel.charAt(0).toUpperCase() + data.riskLevel.slice(1)
      riskElement.setAttribute('data-risk', data.riskLevel)
    }

    if (positionElement && data.positionSize !== undefined) {
      positionElement.textContent =
        data.positionSize > 0 ? `${data.positionSize.toFixed(4)} units` : 'No position'
    }
  }

  /**
   * Update indicators display
   */
  private updateIndicators(indicators: VuManchuIndicators): void {
    const indicatorElements = this.container.querySelectorAll('.indicator-item')

    if (indicatorElements.length >= 3) {
      // Cipher A
      const cipherAValue = indicatorElements[0].querySelector('.indicator-value') as HTMLElement
      if (cipherAValue && indicators.cipher_a !== null) {
        cipherAValue.textContent = indicators.cipher_a.toFixed(2)
        const trend =
          indicators.cipher_a > 50 ? 'bullish' : indicators.cipher_a < -50 ? 'bearish' : 'neutral'
        cipherAValue.setAttribute('data-trend', trend)
      }

      // Cipher B
      const cipherBValue = indicatorElements[1].querySelector('.indicator-value') as HTMLElement
      if (cipherBValue && indicators.cipher_b !== null) {
        cipherBValue.textContent = indicators.cipher_b.toFixed(2)
        const trend =
          indicators.cipher_b > 0 ? 'bullish' : indicators.cipher_b < 0 ? 'bearish' : 'neutral'
        cipherBValue.setAttribute('data-trend', trend)
      }

      // Wave Trend
      const waveTrendValue = indicatorElements[2].querySelector('.indicator-value') as HTMLElement
      if (waveTrendValue && indicators.wave_trend_1 !== null) {
        waveTrendValue.textContent = indicators.wave_trend_1.toFixed(2)
        const trend =
          indicators.wave_trend_1 > 0
            ? 'bullish'
            : indicators.wave_trend_1 < 0
              ? 'bearish'
              : 'neutral'
        waveTrendValue.setAttribute('data-trend', trend)
      }
    }
  }

  /**
   * Update timestamp
   */
  private updateTimestamp(): void {
    const timestampValue = this.container.querySelector('.timestamp-value') as HTMLElement
    if (!timestampValue) return

    const now = new Date()
    timestampValue.textContent = now.toLocaleTimeString()
  }

  /**
   * Set updating state
   */
  private setUpdatingState(updating: boolean): void {
    const card = this.container.querySelector('.llm-decision-card') as HTMLElement
    const indicator = this.container.querySelector('.update-indicator') as HTMLElement

    if (card) {
      card.setAttribute('data-state', updating ? 'updating' : 'active')
    }

    if (indicator) {
      indicator.setAttribute('data-updating', updating.toString())
    }
  }

  /**
   * Clear the decision display
   */
  public clear(): void {
    this.currentDecision = null
    const card = this.container.querySelector('.llm-decision-card') as HTMLElement
    if (card) {
      card.setAttribute('data-state', 'empty')
    }
  }

  /**
   * Get current decision data
   */
  public getCurrentDecision(): LLMDecisionData | null {
    return this.currentDecision
  }
}
