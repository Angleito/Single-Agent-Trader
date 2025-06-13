/**
 * LLM Monitoring Dashboard Component
 * 
 * Provides real-time monitoring of LLM completions, performance metrics,
 * cost tracking, and alert management for the AI trading bot.
 */

import type { 
  AllWebSocketMessages, 
  LLMEventMessage, 
  PerformanceUpdateMessage 
} from './websocket.ts';
import { webSocketClient } from './websocket.ts';

// LLM Event Types
interface LLMRequest {
  timestamp: string;
  request_id: string;
  session_id: string;
  model: string;
  temperature: number;
  max_tokens: number;
  prompt_length: number;
  market_context: Record<string, any>;
}

interface LLMResponse {
  timestamp: string;
  request_id: string;
  session_id: string;
  success: boolean;
  response_time_ms: number;
  cost_estimate_usd: number;
  error?: string;
}

interface TradingDecision {
  timestamp: string;
  request_id: string;
  session_id: string;
  action: string;
  size_pct: number;
  rationale: string;
  symbol: string;
  current_price: number;
  indicators: Record<string, number>;
}

interface LLMAlert {
  timestamp: string;
  level: 'info' | 'warning' | 'critical';
  category: string;
  message: string;
  details: Record<string, any>;
}

interface LLMMetrics {
  total_requests: number;
  total_responses: number;
  success_rate: number;
  avg_response_time_ms: number;
  total_cost_usd: number;
  active_alerts: number;
}

class LLMMonitorDashboard {
  private container: HTMLElement;
  private isConnected: boolean = false;
  private metrics: LLMMetrics = {
    total_requests: 0,
    total_responses: 0,
    success_rate: 0,
    avg_response_time_ms: 0,
    total_cost_usd: 0,
    active_alerts: 0
  };
  private recentEvents: Array<LLMRequest | LLMResponse | TradingDecision | LLMAlert> = [];
  private maxEvents: number = 100;
  
  constructor(containerId: string) {
    const container = document.getElementById(containerId);
    if (!container) {
      throw new Error(`Container with ID '${containerId}' not found`);
    }
    this.container = container;
    this.initialize();
  }

  private initialize(): void {
    this.createDashboardHTML();
    this.setupWebSocketConnection();
    this.setupEventListeners();
    this.loadInitialData();
    this.startPeriodicUpdates();
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

        @media (max-width: 768px) {
          .charts-container {
            grid-template-columns: 1fr;
          }
          
          .metrics-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          }
        }
      </style>
    `;
  }

  private setupWebSocketConnection(): void {
    // Listen for connection status changes
    webSocketClient.onConnectionStatusChange((status) => {
      this.updateConnectionStatus(status === 'connected');
    });

    // Listen for LLM events
    webSocketClient.on('llm_event', (message: LLMEventMessage) => {
      this.handleLLMEvent(message.data);
    });

    // Listen for performance updates
    webSocketClient.on('performance_update', (message: PerformanceUpdateMessage) => {
      this.updateMetrics(message.data);
    });

    // Connect if not already connected
    if (!webSocketClient.isConnected()) {
      webSocketClient.connect();
    }
  }

  private setupEventListeners(): void {
    // Pause/resume feed
    const pauseBtn = document.getElementById('pause-feed');
    let isPaused = false;
    pauseBtn?.addEventListener('click', () => {
      isPaused = !isPaused;
      pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
    });

    // Clear feed
    document.getElementById('clear-feed')?.addEventListener('click', () => {
      this.clearActivityFeed();
    });

    // Event filter
    document.getElementById('event-filter')?.addEventListener('change', (e) => {
      const target = e.target as HTMLSelectElement;
      this.filterEvents(target.value);
    });
  }

  private updateConnectionStatus(connected: boolean): void {
    this.isConnected = connected;
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');
    
    if (statusIndicator && statusText) {
      if (connected) {
        statusIndicator.classList.add('connected');
        statusText.textContent = 'Connected';
      } else {
        statusIndicator.classList.remove('connected');
        statusText.textContent = 'Disconnected';
      }
    }
  }

  private handleLLMEvent(eventData: any): void {
    // Add to recent events
    this.recentEvents.unshift(eventData);
    if (this.recentEvents.length > this.maxEvents) {
      this.recentEvents.pop();
    }

    // Update activity feed
    this.addToActivityFeed(eventData);

    // Handle specific event types
    switch (eventData.event_type) {
      case 'llm_request':
        this.handleRequest(eventData);
        break;
      case 'llm_response':
        this.handleResponse(eventData);
        break;
      case 'trading_decision':
        this.handleDecision(eventData);
        break;
      case 'alert':
        this.handleAlert(eventData);
        break;
    }
  }

  private addToActivityFeed(eventData: any): void {
    const feed = document.getElementById('activity-feed');
    if (!feed) return;

    const item = document.createElement('div');
    item.className = `activity-item ${eventData.event_type}`;
    
    const timestamp = new Date(eventData.timestamp).toLocaleTimeString();
    const type = eventData.event_type.replace('_', ' ').toUpperCase();
    
    let content = `<span class="timestamp">${timestamp}</span><span class="type">${type}</span>`;
    
    switch (eventData.event_type) {
      case 'llm_request':
        content += `Model: ${eventData.model}, Tokens: ${eventData.prompt_length}`;
        break;
      case 'llm_response':
        content += `${eventData.success ? 'SUCCESS' : 'FAILED'} - ${eventData.response_time_ms}ms`;
        if (eventData.cost_estimate_usd) {
          content += ` ($${eventData.cost_estimate_usd.toFixed(4)})`;
        }
        break;
      case 'trading_decision':
        content += `${eventData.action} - ${eventData.rationale}`;
        break;
      case 'alert':
        content += `${eventData.alert_level?.toUpperCase()}: ${eventData.alert_message}`;
        break;
    }
    
    item.innerHTML = content;
    
    // Insert at top
    feed.insertBefore(item, feed.firstChild);
    
    // Remove old items if too many
    while (feed.children.length > 100) {
      feed.removeChild(feed.lastChild!);
    }
  }

  private handleRequest(data: any): void {
    this.metrics.total_requests++;
    this.updateMetricsDisplay();
  }

  private handleResponse(data: any): void {
    this.metrics.total_responses++;
    
    // Update success rate
    const successfulResponses = this.recentEvents
      .filter(e => e.event_type === 'llm_response' && e.success)
      .length;
    this.metrics.success_rate = (successfulResponses / this.metrics.total_responses) * 100;
    
    // Update average response time
    const responseTimes = this.recentEvents
      .filter(e => e.event_type === 'llm_response')
      .map(e => e.response_time_ms)
      .filter(t => t !== undefined);
    
    if (responseTimes.length > 0) {
      this.metrics.avg_response_time_ms = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
    }
    
    // Update total cost
    if (data.cost_estimate_usd) {
      this.metrics.total_cost_usd += data.cost_estimate_usd;
    }
    
    this.updateMetricsDisplay();
  }

  private handleDecision(data: any): void {
    // Update decision counters
    const action = data.action?.toLowerCase();
    const counter = document.getElementById(`${action}-count`);
    if (counter) {
      const current = parseInt(counter.textContent || '0');
      counter.textContent = (current + 1).toString();
    }
    
    // Add to recent decisions display
    this.addToRecentDecisions(data);
  }

  private handleAlert(data: any): void {
    this.metrics.active_alerts++;
    this.updateMetricsDisplay();
    this.addToAlertsPanel(data);
  }

  private addToRecentDecisions(data: any): void {
    const container = document.getElementById('recent-decisions');
    if (!container) return;
    
    const item = document.createElement('div');
    item.className = 'decision-item';
    item.innerHTML = `
      <div class="decision-header">
        <span class="action action-${data.action?.toLowerCase()}">${data.action}</span>
        <span class="timestamp">${new Date(data.timestamp).toLocaleTimeString()}</span>
      </div>
      <div class="decision-rationale">${data.rationale}</div>
      <div class="decision-details">
        Symbol: ${data.symbol} | Price: $${data.current_price}
      </div>
    `;
    
    container.insertBefore(item, container.firstChild);
    
    // Keep only last 10 decisions
    while (container.children.length > 10) {
      container.removeChild(container.lastChild!);
    }
  }

  private addToAlertsPanel(data: any): void {
    const container = document.getElementById('alerts-list');
    if (!container) return;
    
    const item = document.createElement('div');
    item.className = `alert-item ${data.alert_level}`;
    item.innerHTML = `
      <div class="alert-time">${new Date(data.timestamp).toLocaleString()}</div>
      <div class="alert-message">${data.alert_message}</div>
      <div class="alert-category">${data.alert_category}</div>
    `;
    
    container.insertBefore(item, container.firstChild);
    
    // Keep only last 20 alerts
    while (container.children.length > 20) {
      container.removeChild(container.lastChild!);
    }
  }

  private updateMetricsDisplay(): void {
    document.getElementById('total-requests')!.textContent = this.metrics.total_requests.toString();
    document.getElementById('success-rate')!.textContent = `${this.metrics.success_rate.toFixed(1)}%`;
    document.getElementById('avg-response-time')!.textContent = `${Math.round(this.metrics.avg_response_time_ms)}ms`;
    document.getElementById('total-cost')!.textContent = `$${this.metrics.total_cost_usd.toFixed(4)}`;
    document.getElementById('active-alerts')!.textContent = this.metrics.active_alerts.toString();
  }

  private clearActivityFeed(): void {
    const feed = document.getElementById('activity-feed');
    if (feed) {
      feed.innerHTML = '';
    }
  }

  private filterEvents(filter: string): void {
    const feed = document.getElementById('activity-feed');
    if (!feed) return;
    
    const items = feed.children;
    for (let i = 0; i < items.length; i++) {
      const item = items[i] as HTMLElement;
      if (filter === 'all') {
        item.style.display = 'block';
      } else {
        const eventType = item.className.split(' ')[1];
        item.style.display = eventType.includes(filter) ? 'block' : 'none';
      }
    }
  }

  private async loadInitialData(): Promise<void> {
    try {
      // Load initial metrics
      const response = await fetch('/llm/status');
      if (response.ok) {
        const data = await response.json();
        if (data.metrics?.['24_hours']) {
          const metrics = data.metrics['24_hours'];
          this.metrics = {
            total_requests: metrics.total_requests || 0,
            total_responses: metrics.total_responses || 0,
            success_rate: (metrics.success_rate || 0) * 100,
            avg_response_time_ms: metrics.avg_response_time_ms || 0,
            total_cost_usd: metrics.total_cost_usd || 0,
            active_alerts: data.active_alerts || 0
          };
          this.updateMetricsDisplay();
        }
      }
      
      // Load recent activity
      const activityResponse = await fetch('/llm/activity?limit=50');
      if (activityResponse.ok) {
        const activityData = await activityResponse.json();
        this.recentEvents = activityData.activity || [];
        
        // Display recent events
        this.recentEvents.reverse().forEach(event => {
          this.addToActivityFeed(event);
        });
      }
      
    } catch (error) {
      console.error('Failed to load initial LLM data:', error);
    }
  }

  private startPeriodicUpdates(): void {
    // Update metrics every 30 seconds
    setInterval(async () => {
      try {
        const response = await fetch('/llm/metrics?time_window=1h');
        if (response.ok) {
          const data = await response.json();
          if (data.metrics) {
            this.updateMetrics(data.metrics);
          }
        }
      } catch (error) {
        console.error('Failed to update metrics:', error);
      }
    }, 30000);
  }

  private updateMetrics(metricsData: any): void {
    if (metricsData.total_requests !== undefined) {
      this.metrics.total_requests = metricsData.total_requests;
    }
    if (metricsData.total_responses !== undefined) {
      this.metrics.total_responses = metricsData.total_responses;
    }
    if (metricsData.success_rate !== undefined) {
      this.metrics.success_rate = metricsData.success_rate * 100;
    }
    if (metricsData.avg_response_time_ms !== undefined) {
      this.metrics.avg_response_time_ms = metricsData.avg_response_time_ms;
    }
    if (metricsData.total_cost_usd !== undefined) {
      this.metrics.total_cost_usd = metricsData.total_cost_usd;
    }
    if (metricsData.active_alerts !== undefined) {
      this.metrics.active_alerts = metricsData.active_alerts;
    }
    
    this.updateMetricsDisplay();
  }

  // Public methods
  public destroy(): void {
    // Clean up WebSocket listeners
    webSocketClient.off('llm_event', this.handleLLMEvent.bind(this));
    
    // Clear the container
    this.container.innerHTML = '';
  }

  public refresh(): void {
    this.loadInitialData();
  }

  public exportData(): string {
    return JSON.stringify({
      metrics: this.metrics,
      recentEvents: this.recentEvents,
      timestamp: new Date().toISOString()
    }, null, 2);
  }
}

// Export for use in other modules
export { LLMMonitorDashboard };

// Global initialization function
(window as any).initLLMMonitor = function(containerId: string) {
  return new LLMMonitorDashboard(containerId);
};

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('llm-monitor-container');
  if (container) {
    new LLMMonitorDashboard('llm-monitor-container');
  }
});