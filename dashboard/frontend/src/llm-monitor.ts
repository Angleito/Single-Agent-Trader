/**
 * LLM Monitoring Dashboard Component
 *
 * Provides real-time monitoring of LLM completions, performance metrics,
 * cost tracking, and alert management for the AI trading bot.
 */

import type {
  LLMEventMessage,
  PerformanceUpdateMessage,
} from './websocket.ts'
import { DashboardWebSocket } from './websocket.ts'

// LLM Event Types
interface LLMRequest {
  timestamp: string
  request_id: string
  session_id: string
  model: string
  temperature: number
  max_tokens: number
  prompt_length: number
  market_context: Record<string, any>
  event_type?: string
}

interface LLMResponse {
  timestamp: string
  request_id: string
  session_id: string
  success: boolean
  response_time_ms: number
  cost_estimate_usd: number
  error?: string
  event_type?: string
}

interface TradingDecision {
  timestamp: string
  request_id: string
  session_id: string
  action: string
  size_pct: number
  rationale: string
  symbol: string
  current_price: number
  indicators: Record<string, number>
  event_type?: string
}

interface LLMAlert {
  timestamp: string
  level: 'info' | 'warning' | 'critical'
  category: string
  message: string
  details: Record<string, any>
  event_type?: string
}

// Base interface for all LLM events
interface LLMEvent {
  event_type: string
  timestamp: string
  action?: string
}

interface LLMMetrics {
  total_requests: number
  total_responses: number
  success_rate: number
  avg_response_time_ms: number
  total_cost_usd: number
  active_alerts: number
}

class LLMMonitorDashboard {
  private container: HTMLElement
  private websocket: DashboardWebSocket | null = null
  private isConnected: boolean = false
  private isPaused: boolean = false
  private successfulResponses: number = 0
  private responseTimes: number[] = []
  private decisionCounts = {
    long: 0,
    short: 0,
    hold: 0,
    close: 0,
  }
  private metrics: LLMMetrics = {
    total_requests: 0,
    total_responses: 0,
    success_rate: 0,
    avg_response_time_ms: 0,
    total_cost_usd: 0,
    active_alerts: 0,
  }
  private recentEvents: Array<LLMRequest | LLMResponse | TradingDecision | LLMAlert> = []
  private maxEvents: number = 100

  constructor(containerId: string, websocketInstance?: DashboardWebSocket) {
    const container = document.getElementById(containerId)
    if (!container) {
      throw new Error(`Container with ID '${containerId}' not found`)
    }
    this.container = container
    this.websocket = websocketInstance ?? null
    this.initialize()
  }

  private initialize(): void {
    this.createDashboardHTML()
    this.setupWebSocketConnection()
    this.setupEventListeners()
    this.loadInitialData()
    this.startPeriodicUpdates()
  }

  private createDashboardHTML(): void {
    this.container.innerHTML = `
      <div class="llm-monitor-dashboard">
        <!-- Header with Connection Status -->
        <div class="dashboard-header">
          <h2>LLM Completion Monitor</h2>
          <div class="connection-status" id="llm-connection-status">
            <span class="status-indicator disconnected"></span>
            <span class="status-text">Disconnected</span>
          </div>
        </div>

        <!-- Metrics Overview -->
        <div class="metrics-grid">
          <div class="metric-card">
            <h3>Total Requests</h3>
            <div class="metric-value" id="total-requests">0</div>
            <div class="metric-change" id="requests-change"></div>
          </div>
          <div class="metric-card">
            <h3>Success Rate</h3>
            <div class="metric-value" id="success-rate">0%</div>
            <div class="metric-change" id="success-change"></div>
          </div>
          <div class="metric-card">
            <h3>Avg Response Time</h3>
            <div class="metric-value" id="avg-response-time">0ms</div>
            <div class="metric-change" id="response-time-change"></div>
          </div>
          <div class="metric-card">
            <h3>Total Cost</h3>
            <div class="metric-value" id="total-cost">$0.00</div>
            <div class="metric-change" id="cost-change"></div>
          </div>
          <div class="metric-card alerts">
            <h3>Active Alerts</h3>
            <div class="metric-value" id="active-alerts">0</div>
            <div class="metric-change" id="alerts-change"></div>
          </div>
        </div>

        <!-- Charts and Visualizations -->
        <div class="charts-container">
          <div class="chart-panel">
            <h3>Response Time Trend</h3>
            <canvas id="response-time-chart" width="400" height="200"></canvas>
          </div>
          <div class="chart-panel">
            <h3>Cost Analysis</h3>
            <canvas id="cost-chart" width="400" height="200"></canvas>
          </div>
        </div>

        <!-- Real-time Activity Feed -->
        <div class="activity-section">
          <h3>Real-time Activity</h3>
          <div class="activity-controls">
            <button id="pause-feed" class="btn-secondary">Pause</button>
            <button id="clear-feed" class="btn-secondary">Clear</button>
            <select id="event-filter">
              <option value="all">All Events</option>
              <option value="requests">Requests</option>
              <option value="responses">Responses</option>
              <option value="decisions">Decisions</option>
              <option value="alerts">Alerts</option>
            </select>
          </div>
          <div class="activity-feed" id="activity-feed"></div>
        </div>

        <!-- Alerts Panel -->
        <div class="alerts-section">
          <h3>Recent Alerts</h3>
          <div class="alerts-list" id="alerts-list"></div>
        </div>

        <!-- Trading Decisions Analysis -->
        <div class="decisions-section">
          <h3>Trading Decisions</h3>
          <div class="decisions-stats">
            <div class="decision-stat">
              <span class="label">Long:</span>
              <span class="value" id="long-count">0</span>
            </div>
            <div class="decision-stat">
              <span class="label">Short:</span>
              <span class="value" id="short-count">0</span>
            </div>
            <div class="decision-stat">
              <span class="label">Hold:</span>
              <span class="value" id="hold-count">0</span>
            </div>
            <div class="decision-stat">
              <span class="label">Close:</span>
              <span class="value" id="close-count">0</span>
            </div>
          </div>
          <div class="recent-decisions" id="recent-decisions"></div>
        </div>
      </div>

      <!-- Styles -->
      <style>
        .llm-monitor-dashboard {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: #1a1a1a;
          color: #e0e0e0;
          padding: 20px;
          min-height: 100vh;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 30px;
          padding-bottom: 20px;
          border-bottom: 1px solid #333;
        }

        .dashboard-header h2 {
          margin: 0;
          color: #fff;
          font-size: 24px;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: #ff4444;
        }

        .status-indicator.connected {
          background: #44ff44;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
          margin-bottom: 30px;
        }

        .metric-card {
          background: #2a2a2a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          text-align: center;
        }

        .metric-card.alerts {
          border-color: #ff6b6b;
        }

        .metric-card h3 {
          margin: 0 0 10px 0;
          font-size: 14px;
          color: #888;
          text-transform: uppercase;
        }

        .metric-value {
          font-size: 32px;
          font-weight: bold;
          color: #fff;
          margin-bottom: 5px;
        }

        .metric-change {
          font-size: 12px;
          color: #888;
        }

        .metric-change.positive {
          color: #44ff44;
        }

        .metric-change.negative {
          color: #ff4444;
        }

        .charts-container {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-bottom: 30px;
        }

        .chart-panel {
          background: #2a2a2a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
        }

        .chart-panel h3 {
          margin: 0 0 15px 0;
          color: #fff;
        }

        .activity-section, .alerts-section, .decisions-section {
          background: #2a2a2a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }

        .activity-section h3, .alerts-section h3, .decisions-section h3 {
          margin: 0 0 15px 0;
          color: #fff;
        }

        .activity-controls {
          display: flex;
          gap: 10px;
          margin-bottom: 15px;
        }

        .btn-secondary {
          background: #333;
          border: 1px solid #555;
          color: #e0e0e0;
          padding: 8px 16px;
          border-radius: 4px;
          cursor: pointer;
        }

        .btn-secondary:hover {
          background: #444;
        }

        .activity-feed {
          max-height: 400px;
          overflow-y: auto;
          border: 1px solid #333;
          border-radius: 4px;
        }

        .activity-item {
          padding: 10px;
          border-bottom: 1px solid #333;
          font-family: monospace;
          font-size: 12px;
        }

        .activity-item.request {
          background: #1a2332;
        }

        .activity-item.response {
          background: #233218;
        }

        .activity-item.decision {
          background: #321a32;
        }

        .activity-item.alert {
          background: #332318;
        }

        .activity-item .timestamp {
          color: #888;
          margin-right: 10px;
        }

        .activity-item .type {
          font-weight: bold;
          margin-right: 10px;
        }

        .decisions-stats {
          display: flex;
          gap: 20px;
          margin-bottom: 15px;
        }

        .decision-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .decision-stat .label {
          font-size: 12px;
          color: #888;
        }

        .decision-stat .value {
          font-size: 24px;
          font-weight: bold;
          color: #fff;
        }

        .recent-decisions {
          max-height: 400px;
          overflow-y: auto;
        }

        .decision-item {
          padding: 12px;
          margin-bottom: 10px;
          border: 1px solid #333;
          border-radius: 6px;
          background: #2a2a2a;
          transition: all 0.2s ease;
        }

        .decision-item:hover {
          border-color: #555;
          transform: translateY(-1px);
        }

        .decision-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .action {
          font-weight: bold;
          font-size: 14px;
          padding: 4px 10px;
          border-radius: 4px;
          text-transform: uppercase;
        }

        .action-buy, .action-long {
          background: #28a745;
          color: #fff;
        }

        .action-sell, .action-short {
          background: #dc3545;
          color: #fff;
        }

        .action-hold {
          background: #ffc107;
          color: #000;
        }

        .action-close {
          background: #6c757d;
          color: #fff;
        }

        .decision-reasoning {
          font-size: 14px;
          line-height: 1.6;
          color: #e0e0e0;
          margin-bottom: 8px;
          padding: 8px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 4px;
        }

        .decision-details {
          font-size: 12px;
          color: #888;
          display: flex;
          gap: 16px;
          flex-wrap: wrap;
        }

        .decision-details span {
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .decision-details strong {
          color: #ddd;
        }

        .decision-indicators {
          margin-top: 8px;
          font-size: 11px;
          color: #888;
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }

        .activity-error {
          margin-top: 8px;
          padding: 8px;
          background: rgba(255, 68, 68, 0.1);
          border-radius: 4px;
          color: #ff8888;
          font-size: 12px;
        }

        .alert-message {
          margin-top: 8px;
          font-size: 14px;
          line-height: 1.5;
          color: #e0e0e0;
        }

        .alert-level {
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 11px;
          font-weight: 600;
        }

        .alert-level.info {
          background: #4a9eff;
          color: #fff;
        }

        .alert-level.warning {
          background: #ffa500;
          color: #000;
        }

        .alert-level.critical {
          background: #ff4444;
          color: #fff;
        }

        .success {
          color: #44ff44;
        }

        .failed {
          color: #ff4444;
        }

        .alerts-list {
          max-height: 300px;
          overflow-y: auto;
        }

        .alert-item {
          padding: 10px;
          margin-bottom: 10px;
          border-radius: 4px;
          border-left: 4px solid;
        }

        .alert-item.info {
          background: #1a2332;
          border-left-color: #4a9eff;
        }

        .alert-item.warning {
          background: #332318;
          border-left-color: #ffa500;
        }

        .alert-item.critical {
          background: #331818;
          border-left-color: #ff4444;
        }

        .alert-item .alert-time {
          font-size: 12px;
          color: #888;
        }

        .alert-item .alert-message {
          font-weight: bold;
          margin: 5px 0;
        }

        .alert-item .alert-category {
          font-size: 12px;
          color: #aaa;
        }

        .error-state {
          color: #ff4444;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        /* Smooth transitions for metric updates */
        .metric-value {
          transition: all 0.3s ease;
        }

        .metric-value.updating {
          transform: scale(1.05);
          color: #4a9eff;
        }

        /* Enhanced activity feed scrollbar */
        .activity-feed::-webkit-scrollbar {
          width: 8px;
        }

        .activity-feed::-webkit-scrollbar-track {
          background: #1a1a1a;
          border-radius: 4px;
        }

        .activity-feed::-webkit-scrollbar-thumb {
          background: #444;
          border-radius: 4px;
        }

        .activity-feed::-webkit-scrollbar-thumb:hover {
          background: #555;
        }

        @media (max-width: 768px) {
          .charts-container {
            grid-template-columns: 1fr;
          }

          .metrics-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          }
        }
      </style>
    `
  }

  private setupWebSocketConnection(): void {
    if (!this.websocket) {
      console.warn('No WebSocket instance provided to LLM Monitor')
      return
    }

    // Listen for connection status changes
    this.websocket.onConnectionStatusChange((status) => {
      this.updateConnectionStatus(status === 'connected')
    })

    // Listen for various LLM-related events
    this.websocket.on('llm_event', (message: LLMEventMessage) => {
      if (message.data) {
        this.handleLLMEvent(message.data)
      }
    })

    // Also listen for specific event types that might come through
    this.websocket.on('llm_request', (message: any) => {
      if (message.data) {
        this.handleLLMEvent({ ...message.data, event_type: 'llm_request' })
      }
    })

    this.websocket.on('llm_response', (message: any) => {
      if (message.data) {
        this.handleLLMEvent({ ...message.data, event_type: 'llm_response' })
      }
    })

    this.websocket.on('trading_decision', (message: any) => {
      if (message.data) {
        this.handleLLMEvent({ ...message.data, event_type: 'trading_decision' })
      }
    })

    this.websocket.on('llm_decision', (message: any) => {
      // Backend sends trading_decision events as llm_decision
      if (message.data) {
        this.handleLLMEvent(message.data)
      }
    })

    this.websocket.on('ai_decision', (message: any) => {
      // Map ai_decision to trading_decision for consistency
      if (message.data) {
        this.handleLLMEvent({
          ...message.data,
          event_type: 'trading_decision',
          rationale: message.data.reasoning,
          timestamp: message.data.timestamp ?? new Date().toISOString(),
        })
      }
    })

    // Listen for performance updates
    this.websocket.on('performance_update', (message: PerformanceUpdateMessage) => {
      if (message.data) {
        this.updateMetrics(message.data)
      }
    })

    // Connect if not already connected
    if (!this.websocket.isConnected()) {
      this.websocket.connect()
    }
  }

  private setupEventListeners(): void {
    // Pause/resume feed
    const pauseBtn = document.getElementById('pause-feed')
    pauseBtn?.addEventListener('click', () => {
      this.isPaused = !this.isPaused
      pauseBtn.textContent = this.isPaused ? 'Resume' : 'Pause'
    })

    // Clear feed
    document.getElementById('clear-feed')?.addEventListener('click', () => {
      this.clearActivityFeed()
    })

    // Event filter
    document.getElementById('event-filter')?.addEventListener('change', (e) => {
      const target = e.target as HTMLSelectElement
      this.filterEvents(target.value)
    })
  }

  private updateConnectionStatus(connected: boolean): void {
    this.isConnected = connected
    const statusIndicator = document.querySelector('.status-indicator')
    const statusText = document.querySelector('.status-text')

    if (statusIndicator && statusText) {
      if (connected) {
        statusIndicator.classList.add('connected')
        statusText.textContent = 'Connected'
      } else {
        statusIndicator.classList.remove('connected')
        statusText.textContent = 'Disconnected'
      }
    }
  }

  private handleLLMEvent(eventData: any): void {
    // Ensure we have a timestamp
    if (!eventData.timestamp) {
      eventData.timestamp = new Date().toISOString()
    }

    // Add to recent events
    this.recentEvents.unshift(eventData)
    if (this.recentEvents.length > this.maxEvents) {
      this.recentEvents.pop()
    }

    // Update activity feed if not paused
    if (!this.isPaused) {
      this.addToActivityFeed(eventData)
    }

    // Handle specific event types
    switch (eventData.event_type) {
      case 'llm_request':
        this.handleRequest(eventData)
        break
      case 'llm_response':
        this.handleResponse(eventData)
        break
      case 'trading_decision':
        this.handleDecision(eventData)
        break
      case 'alert':
        this.handleAlert(eventData)
        break
    }
  }

  private addToActivityFeed(eventData: any): void {
    const feed = document.getElementById('activity-feed')
    if (!feed) return

    // Remove loading state if present
    const loadingState = document.getElementById('feed-loading')
    if (loadingState) {
      loadingState.remove()
    }

    const item = document.createElement('div')
    item.className = `activity-item ${eventData.event_type}`

    const timestamp = new Date(eventData.timestamp).toLocaleTimeString()
    const typeLabel = eventData.event_type.replace(/_/g, ' ').toUpperCase()

    let content = `
      <div class="activity-header">
        <div class="activity-meta">
          <span class="timestamp">${timestamp}</span>
          <span class="type">${typeLabel}</span>
        </div>
      </div>
      <div class="activity-content">
    `

    switch (eventData.event_type) {
      case 'llm_request':
        content += `
          <div class="activity-details">
            <div class="activity-detail">
              <strong>Model:</strong> ${eventData.model ?? 'N/A'}
            </div>
            <div class="activity-detail">
              <strong>Tokens:</strong> ${eventData.prompt_tokens ?? eventData.prompt_length ?? 'N/A'}
            </div>
            ${
              eventData.temperature !== undefined
                ? `
              <div class="activity-detail">
                <strong>Temperature:</strong> ${eventData.temperature}
              </div>
            `
                : ''
            }
          </div>
        `
        break

      case 'llm_response': {
        const statusClass = eventData.success ? 'success' : 'failed'
        content += `
          <div class="activity-details">
            <div class="activity-detail">
              <strong>Status:</strong> <span class="${statusClass}">${eventData.success ? 'SUCCESS' : 'FAILED'}</span>
            </div>
            <div class="activity-detail">
              <strong>Response Time:</strong> ${eventData.response_time_ms}ms
            </div>
            ${
              eventData.cost_estimate_usd !== undefined
                ? `
              <div class="activity-detail">
                <strong>Cost:</strong> $${eventData.cost_estimate_usd.toFixed(4)}
              </div>
            `
                : ''
            }
            ${
              eventData.total_tokens
                ? `
              <div class="activity-detail">
                <strong>Total Tokens:</strong> ${eventData.total_tokens}
              </div>
            `
                : ''
            }
          </div>
          ${
            eventData.error
              ? `
            <div class="activity-error">
              <strong>Error:</strong> ${eventData.error}
            </div>
          `
              : ''
          }
        `
        break
      }

      case 'trading_decision': {
        const actionClass = this.getActionClass(eventData.action)
        content += `
          <div class="activity-details">
            <div class="activity-detail">
              <strong>Action:</strong> <span class="action ${actionClass}">${eventData.action?.toUpperCase() ?? 'N/A'}</span>
            </div>
            ${
              eventData.confidence !== undefined
                ? `
              <div class="activity-detail">
                <strong>Confidence:</strong> ${(eventData.confidence * 100).toFixed(1)}%
              </div>
            `
                : ''
            }
            ${
              eventData.price ?? eventData.current_price
                ? `
              <div class="activity-detail">
                <strong>Price:</strong> $${(eventData.price ?? eventData.current_price).toFixed(2)}
              </div>
            `
                : ''
            }
          </div>
          ${
            eventData.reasoning ?? eventData.rationale
              ? `
            <div class="activity-reasoning">
              <strong>Reasoning:</strong> ${eventData.reasoning ?? eventData.rationale}
            </div>
          `
              : ''
          }
        `
        break
      }

      case 'alert':
        content += `
          <div class="activity-details">
            <div class="activity-detail">
              <strong>Level:</strong> <span class="alert-level ${eventData.alert_level}">${eventData.alert_level?.toUpperCase()}</span>
            </div>
            <div class="activity-detail">
              <strong>Category:</strong> ${eventData.alert_category ?? 'General'}
            </div>
          </div>
          <div class="alert-message">
            ${eventData.alert_message}
          </div>
        `
        break
    }

    content += '</div>'
    item.innerHTML = content

    // Insert at top
    feed.insertBefore(item, feed.firstChild)

    // Remove old items if too many
    while (feed.children.length > 100) {
      feed.removeChild(feed.lastChild!)
    }
  }

  private getActionClass(action: string): string {
    const actionLower = action?.toLowerCase()
    if (actionLower === 'buy' || actionLower === 'long') return 'action-buy'
    if (actionLower === 'sell' || actionLower === 'short') return 'action-sell'
    if (actionLower === 'hold') return 'action-hold'
    if (actionLower === 'close') return 'action-close'
    return ''
  }

  private handleRequest(_data: any): void {
    this.metrics.total_requests++
    this.updateMetricsDisplay()
  }

  private handleResponse(data: any): void {
    this.metrics.total_responses++

    // Track success
    if (data.success) {
      this.successfulResponses++
    }

    // Update success rate
    this.metrics.success_rate = (this.successfulResponses / this.metrics.total_responses) * 100

    // Track response time
    if (data.response_time_ms !== undefined) {
      this.responseTimes.push(data.response_time_ms)
      // Keep only last 100 response times
      if (this.responseTimes.length > 100) {
        this.responseTimes.shift()
      }
      // Calculate average
      this.metrics.avg_response_time_ms =
        this.responseTimes.reduce((a, b) => a + b, 0) / this.responseTimes.length
    }

    // Update total cost
    if (data.cost_estimate_usd) {
      this.metrics.total_cost_usd += data.cost_estimate_usd
    }

    this.updateMetricsDisplay()
  }

  private handleDecision(data: any): void {
    // Update decision counters
    const _action = data.action?.toLowerCase()

    // Map action to counter
    if (action === 'buy' || action === 'long') {
      this.decisionCounts.long++
      document.getElementById('long-count')!.textContent = this.decisionCounts.long.toString()
    } else if (action === 'sell' || action === 'short') {
      this.decisionCounts.short++
      document.getElementById('short-count')!.textContent = this.decisionCounts.short.toString()
    } else if (action === 'hold') {
      this.decisionCounts.hold++
      document.getElementById('hold-count')!.textContent = this.decisionCounts.hold.toString()
    } else if (action === 'close') {
      this.decisionCounts.close++
      document.getElementById('close-count')!.textContent = this.decisionCounts.close.toString()
    }

    // Add to recent decisions display
    this.addToRecentDecisions(data)
  }

  private handleAlert(data: any): void {
    this.metrics.active_alerts++
    this.updateMetricsDisplay()
    this.addToAlertsPanel(data)
  }

  private addToRecentDecisions(data: any): void {
    const container = document.getElementById('recent-decisions')
    if (!container) return

    const actionClass = this.getActionClass(data.action)
    const item = document.createElement('div')
    item.className = 'decision-item'
    item.innerHTML = `
      <div class="decision-header">
        <span class="action ${actionClass}">${data.action}</span>
        <span class="timestamp">${new Date(data.timestamp).toLocaleTimeString()}</span>
      </div>
      ${
        data.reasoning ?? data.rationale
          ? `
        <div class="decision-reasoning">
          ${data.reasoning ?? data.rationale}
        </div>
      `
          : ''
      }
      <div class="decision-details">
        ${data.symbol ? `<span><strong>Symbol:</strong> ${data.symbol}</span>` : ''}
        ${data.price ?? data.current_price ? `<span><strong>Price:</strong> $${(data.price ?? data.current_price).toFixed(2)}</span>` : ''}
        ${data.confidence !== undefined ? `<span><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</span>` : ''}
        ${data.leverage ? `<span><strong>Leverage:</strong> ${data.leverage}x</span>` : ''}
      </div>
      ${
        data.indicators
          ? `
        <div class="decision-indicators">
          ${Object.entries(data.indicators)
            .map(([key, value]) => `<span><strong>${key}:</strong> ${value}</span>`)
            .join('')}
        </div>
      `
          : ''
      }
    `

    container.insertBefore(item, container.firstChild)

    // Keep only last 10 decisions
    while (container.children.length > 10) {
      container.removeChild(container.lastChild!)
    }
  }

  private addToAlertsPanel(data: any): void {
    const container = document.getElementById('alerts-list')
    if (!container) return

    const item = document.createElement('div')
    item.className = `alert-item ${data.alert_level}`
    item.innerHTML = `
      <div class="alert-time">${new Date(data.timestamp).toLocaleString()}</div>
      <div class="alert-message">${data.alert_message}</div>
      <div class="alert-category">${data.alert_category}</div>
    `

    container.insertBefore(item, container.firstChild)

    // Keep only last 20 alerts
    while (container.children.length > 20) {
      container.removeChild(container.lastChild!)
    }
  }

  private updateMetricsDisplay(): void {
    // Update metric values
    const totalRequestsEl = document.getElementById('total-requests')
    const successRateEl = document.getElementById('success-rate')
    const avgResponseTimeEl = document.getElementById('avg-response-time')
    const totalCostEl = document.getElementById('total-cost')
    const activeAlertsEl = document.getElementById('active-alerts')

    if (totalRequestsEl) {
      const prevValue = parseInt(totalRequestsEl.textContent ?? '0')
      totalRequestsEl.textContent = this.metrics.total_requests.toString()
      this.updateMetricChange('requests-change', this.metrics.total_requests - prevValue)
    }

    if (successRateEl) {
      const prevValue = parseFloat(successRateEl.textContent ?? '0')
      successRateEl.textContent = `${this.metrics.success_rate.toFixed(1)}%`
      this.updateMetricChange('success-change', this.metrics.success_rate - prevValue, '%')
    }

    if (avgResponseTimeEl) {
      const prevValue = parseInt(avgResponseTimeEl.textContent ?? '0')
      avgResponseTimeEl.textContent = `${Math.round(this.metrics.avg_response_time_ms)}ms`
      this.updateMetricChange(
        'response-time-change',
        Math.round(this.metrics.avg_response_time_ms) - prevValue,
        'ms',
        true
      )
    }

    if (totalCostEl) {
      const prevValue = parseFloat(totalCostEl.textContent?.replace('$', '') ?? '0')
      totalCostEl.textContent = `$${this.metrics.total_cost_usd.toFixed(4)}`
      this.updateMetricChange('cost-change', this.metrics.total_cost_usd - prevValue, '$')
    }

    if (activeAlertsEl) {
      const prevValue = parseInt(activeAlertsEl.textContent ?? '0')
      activeAlertsEl.textContent = this.metrics.active_alerts.toString()
      this.updateMetricChange('alerts-change', this.metrics.active_alerts - prevValue)
    }
  }

  private updateMetricChange(
    elementId: string,
    change: number,
    suffix: string = '',
    inverse: boolean = false
  ): void {
    const element = document.getElementById(elementId)
    if (!element) return

    if (change === 0) {
      element.textContent = ''
      element.className = 'metric-change'
    } else if (change > 0) {
      element.textContent = `+${change.toFixed(suffix === '$' ? 4 : suffix === '%' ? 1 : 0)}${suffix}`
      element.className = inverse ? 'metric-change negative' : 'metric-change positive'
    } else {
      element.textContent = `${change.toFixed(suffix === '$' ? 4 : suffix === '%' ? 1 : 0)}${suffix}`
      element.className = inverse ? 'metric-change positive' : 'metric-change negative'
    }
  }

  private clearActivityFeed(): void {
    const feed = document.getElementById('activity-feed')
    if (feed) {
      feed.innerHTML = ''
    }
  }

  private filterEvents(filter: string): void {
    const feed = document.getElementById('activity-feed')
    if (!feed) return

    const items = feed.children
    for (let i = 0; i < items.length; i++) {
      const item = items[i] as HTMLElement
      if (item.classList.contains('loading-state')) continue

      if (filter === 'all') {
        item.style.display = 'block'
      } else {
        // Check if the item's classes contain the filter type
        const hasFilterType =
          item.classList.contains(`llm_${filter}`) ||
          item.classList.contains(filter) ||
          item.classList.contains(`trading_${filter}`)
        item.style.display = hasFilterType ? 'block' : 'none'
      }
    }
  }

  private async loadInitialData(): Promise<void> {
    try {
      // Show loading state
      const loadingState = document.getElementById('feed-loading')
      if (loadingState) {
        loadingState.style.display = 'flex'
      }

      // Load initial metrics
      const response = await fetch('/api/llm/status')
      if (response.ok) {
        const data = await response.json()
        if (data.metrics?.['24_hours']) {
          const metrics = data.metrics['24_hours']
          this.metrics = {
            total_requests: metrics.total_requests ?? 0,
            total_responses: metrics.total_responses ?? 0,
            success_rate: (metrics.success_rate ?? 0) * 100,
            avg_response_time_ms: metrics.avg_response_time_ms ?? 0,
            total_cost_usd: metrics.total_cost_usd ?? 0,
            active_alerts: data.active_alerts ?? 0,
          }

          // Initialize counters for accurate tracking
          this.successfulResponses = Math.round(
            (metrics.total_responses ?? 0) * (metrics.success_rate ?? 0)
          )

          this.updateMetricsDisplay()
        }
      }

      // Load recent activity
      const activityResponse = await fetch('/api/llm/activity?limit=50')
      if (activityResponse.ok) {
        const activityData = await activityResponse.json()
        this.recentEvents = activityData.activity ?? []

        // Process recent events to update counters
        this.recentEvents.forEach((event) => {
          const eventWithType = event as LLMEvent
          if (eventWithType.event_type === 'trading_decision') {
            const _action = eventWithType.action?.toLowerCase()
            if (action === 'buy' || action === 'long') this.decisionCounts.long++
            else if (action === 'sell' || action === 'short') this.decisionCounts.short++
            else if (action === 'hold') this.decisionCounts.hold++
            else if (action === 'close') this.decisionCounts.close++
          }
        })

        // Update decision counters
        document.getElementById('long-count')!.textContent = this.decisionCounts.long.toString()
        document.getElementById('short-count')!.textContent = this.decisionCounts.short.toString()
        document.getElementById('hold-count')!.textContent = this.decisionCounts.hold.toString()
        document.getElementById('close-count')!.textContent = this.decisionCounts.close.toString()

        // Display recent events (newest first)
        this.recentEvents
          .slice()
          .reverse()
          .forEach((event) => {
            this.addToActivityFeed(event)
          })
      }

      // Hide loading state
      if (loadingState) {
        loadingState.style.display = 'none'
      }
    } catch (error) {
      console.error('Failed to load initial LLM data:', error)

      // Hide loading state and show error
      const loadingState = document.getElementById('feed-loading')
      if (loadingState) {
        loadingState.innerHTML = `
          <div class="error-state">
            <span>Failed to load activity data</span>
          </div>
        `
      }
    }
  }

  private startPeriodicUpdates(): void {
    // Update metrics every 30 seconds
    setInterval(async () => {
      try {
        const response = await fetch('/api/llm/metrics?time_window=1h')
        if (response.ok) {
          const data = await response.json()
          if (data.metrics) {
            this.updateMetrics(data.metrics)
          }
        }
      } catch (error) {
        console.error('Failed to update metrics:', error)
      }
    }, 30000)

    // Also add a visual indicator when metrics update
    setInterval(() => {
      // Animate metric values briefly
      const metricValues = document.querySelectorAll('.metric-value')
      metricValues.forEach((el) => {
        el.classList.add('updating')
        setTimeout(() => el.classList.remove('updating'), 300)
      })
    }, 30000)
  }

  private updateMetrics(metricsData: any): void {
    if (metricsData.total_requests !== undefined) {
      this.metrics.total_requests = metricsData.total_requests
    }
    if (metricsData.total_responses !== undefined) {
      this.metrics.total_responses = metricsData.total_responses
    }
    if (metricsData.success_rate !== undefined) {
      this.metrics.success_rate = metricsData.success_rate * 100
    }
    if (metricsData.avg_response_time_ms !== undefined) {
      this.metrics.avg_response_time_ms = metricsData.avg_response_time_ms
    }
    if (metricsData.total_cost_usd !== undefined) {
      this.metrics.total_cost_usd = metricsData.total_cost_usd
    }
    if (metricsData.active_alerts !== undefined) {
      this.metrics.active_alerts = metricsData.active_alerts
    }

    this.updateMetricsDisplay()
  }

  // Public methods
  public destroy(): void {
    // Clean up all WebSocket listeners if websocket exists
    if (this.websocket) {
      // Note: We can't use bound methods in off() since they create new function references
      // Instead, we'll need to store the bound methods or use a different approach
      // For now, we'll rely on the component being destroyed completely
    }

    // Clear the container
    this.container.innerHTML = ''
    this.websocket = null
  }

  public refresh(): void {
    this.loadInitialData()
  }

  public exportData(): string {
    return JSON.stringify(
      {
        metrics: this.metrics,
        recentEvents: this.recentEvents,
        timestamp: new Date().toISOString(),
      },
      null,
      2
    )
  }
}

// Export for use in other modules
export {
  LLMMonitorDashboard,

  // Global initialization function
}
;(window as any).initLLMMonitor = function (
  containerId: string,
  websocketInstance?: DashboardWebSocket
) {
  return new LLMMonitorDashboard(containerId, websocketInstance)
}

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('llm-monitor-container')
  if (container) {
    // Note: Auto-initialization without WebSocket instance will show warning
    // Main app should call initLLMMonitor with websocket instance
    new LLMMonitorDashboard('llm-monitor-container')
  }
})
