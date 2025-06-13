import type { 
  DashboardState, 
  BotStatus, 
  MarketData, 
  TradeAction, 
  Position, 
  RiskMetrics, 
  ConnectionStatus,
  LogEntry,
  VuManchuIndicators
} from './types';

export class DashboardUI {
  private state: DashboardState;
  private logEntries: LogEntry[] = [];
  private maxLogEntries = 100;
  private logPaused = false;
  private animationQueue: Set<string> = new Set();
  private updateThrottles: Map<string, number> = new Map();
  private performanceMetrics = {
    totalReturn: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    maxDrawdown: 0,
    sharpeRatio: 0
  };

  constructor() {
    this.state = {
      bot_status: null,
      market_data: null,
      latest_action: null,
      indicators: null,
      positions: [],
      risk_metrics: null,
      connection_status: 'disconnected',
      error_message: null
    };
  }

  /**
   * Initialize the dashboard UI
   */
  public initialize(): void {
    this.setupEventListeners();
    this.showLoadingScreen();
    this.updateConnectionStatus('disconnected');
    this.log('info', 'Dashboard UI initialized');
    this.startPerformanceMonitoring();
  }

  /**
   * Show the loading screen
   */
  public showLoadingScreen(): void {
    const loading = document.getElementById('loading');
    const dashboard = document.getElementById('dashboard');
    
    if (loading) loading.setAttribute('data-loading', 'true');
    if (dashboard) dashboard.setAttribute('data-dashboard', 'hidden');
  }

  /**
   * Hide loading screen and show dashboard
   */
  public hideLoadingScreen(): void {
    const loading = document.getElementById('loading');
    const dashboard = document.getElementById('dashboard');
    
    if (loading) {
      loading.setAttribute('data-loading', 'false');
      setTimeout(() => loading.style.display = 'none', 500);
    }
    if (dashboard) {
      setTimeout(() => dashboard.setAttribute('data-dashboard', 'visible'), 200);
    }
  }

  /**
   * Update loading progress
   */
  public updateLoadingProgress(progress: number, message?: string): void {
    const progressBar = document.querySelector('[data-progress]') as HTMLElement;
    const loadingMessage = document.querySelector('.loading-message') as HTMLElement;
    
    if (progressBar) {
      progressBar.setAttribute('data-progress', progress.toString());
      progressBar.style.setProperty('--progress', `${progress}%`);
    }
    
    if (loadingMessage && message) {
      loadingMessage.textContent = message;
    }
  }

  /**
   * Set up event listeners for UI interactions
   */
  private setupEventListeners(): void {
    // Chart fullscreen toggle
    const fullscreenBtn = document.querySelector('[data-chart-action="fullscreen"]') as HTMLButtonElement;
    fullscreenBtn?.addEventListener('click', () => {
      this.toggleChartFullscreen();
      this.log('info', 'Chart fullscreen toggled', 'UI');
    });
    
    // Chart retry button
    const retryBtn = document.querySelector('[data-chart-retry]') as HTMLButtonElement;
    retryBtn?.addEventListener('click', () => {
      this.onChartRetry?.();
      this.log('info', 'Chart retry requested', 'UI');
    });
    
    // Error modal close buttons
    const errorModalClose = document.querySelectorAll('[data-modal-close]');
    errorModalClose.forEach(btn => {
      btn.addEventListener('click', () => {
        this.hideError();
      });
    });
    
    // Error retry button
    const errorRetryBtn = document.querySelector('[data-error-retry]') as HTMLButtonElement;
    errorRetryBtn?.addEventListener('click', () => {
      this.hideError();
      this.onErrorRetry?.();
    });
    
    // Log controls
    const logClearBtn = document.querySelector('[data-log-action="clear"]') as HTMLButtonElement;
    logClearBtn?.addEventListener('click', () => {
      this.clearLogs();
    });
    
    const logPauseBtn = document.querySelector('[data-log-action="pause"]') as HTMLButtonElement;
    logPauseBtn?.addEventListener('click', () => {
      this.toggleLogPause();
    });

    // Modal overlay clicks
    const modalOverlay = document.querySelector('[data-modal-overlay]');
    modalOverlay?.addEventListener('click', () => {
      this.hideError();
    });
  }

  // Event callbacks
  private onChartRetry: (() => void) | null = null;
  private onErrorRetry: (() => void) | null = null;

  /**
   * Update connection status with animation
   */
  public updateConnectionStatus(status: ConnectionStatus): void {
    this.state.connection_status = status;
    
    const connectionEl = document.querySelector('[data-connection]') as HTMLElement;
    const statusText = connectionEl?.querySelector('.status-text') as HTMLElement;
    const statusIndicator = connectionEl?.querySelector('.status-indicator') as HTMLElement;
    
    if (connectionEl && statusText && statusIndicator) {
      // Add animation class
      this.addAnimation(connectionEl, 'pulse');
      
      connectionEl.setAttribute('data-connection', status);
      statusText.textContent = this.getConnectionStatusText(status);
      statusIndicator.setAttribute('aria-label', `Connection status: ${status}`);
    }

    // Update last update time
    this.updateLastUpdateTime();
    this.log('info', `Connection status: ${status}`);
  }

  /**
   * Update bot status with visual feedback
   */
  public updateBotStatus(status: BotStatus): void {
    this.state.bot_status = status;
    
    // Update status bar
    const botStatusEl = document.querySelector('[data-bot-status]') as HTMLElement;
    const botIndicator = document.querySelector('[data-bot-indicator]') as HTMLElement;
    const tradingModeEl = document.querySelector('[data-trading-mode]') as HTMLElement;
    const tradingSymbolEl = document.querySelector('[data-trading-symbol]') as HTMLElement;
    const leverageEl = document.querySelector('[data-leverage]') as HTMLElement;
    
    if (botStatusEl) {
      this.addAnimation(botStatusEl, 'flash');
      botStatusEl.textContent = status.status.toUpperCase();
    }
    
    if (botIndicator) {
      botIndicator.setAttribute('data-bot-indicator', this.getBotIndicatorState(status.status));
    }
    
    if (tradingModeEl) {
      tradingModeEl.textContent = status.dry_run ? 'Dry Run' : 'Live Trading';
      tradingModeEl.className = status.dry_run ? 'status-value safe' : 'status-value danger';
    }
    
    if (tradingSymbolEl) {
      tradingSymbolEl.textContent = status.symbol;
    }
    
    if (leverageEl) {
      leverageEl.textContent = `${status.leverage}x`;
    }

    this.updateLastUpdateTime();
    this.log('info', `Bot status: ${status.status} (${status.symbol}, leverage: ${status.leverage}x, dry_run: ${status.dry_run})`);
  }

  /**
   * Update market data with price change animations
   */
  public updateMarketData(data: MarketData): void {
    if (this.shouldThrottleUpdate('market_data', 1000)) return;
    
    this.state.market_data = data;
    
    const priceEl = document.querySelector('[data-current-price]') as HTMLElement;
    const changeEl = document.querySelector('[data-price-change]') as HTMLElement;
    
    if (priceEl) {
      const oldPrice = parseFloat(priceEl.textContent?.replace(/[$,]/g, '') || '0');
      const newPrice = data.price;
      
      priceEl.textContent = this.formatPrice(newPrice);
      
      // Add price change animation
      if (oldPrice > 0 && oldPrice !== newPrice) {
        const changeClass = newPrice > oldPrice ? 'price-up' : 'price-down';
        this.addAnimation(priceEl, changeClass);
      }
    }
    
    if (changeEl && data.change_percent_24h !== undefined) {
      const change = data.change_percent_24h;
      changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
      changeEl.className = `stat-change ${change >= 0 ? 'positive' : 'negative'}`;
      changeEl.setAttribute('data-price-change', change.toString());
    }

    this.updateLastUpdateTime();
  }

  /**
   * Update latest trade action with comprehensive display
   */
  public updateLatestAction(action: TradeAction): void {
    this.state.latest_action = action;
    
    // Add to AI decision log
    this.addAIDecision(action);
    
    this.log('info', `Trade action: ${action.action} (${(action.confidence * 100).toFixed(1)}% confidence) - ${action.reasoning}`);
  }

  /**
   * Add AI decision to the log with formatting
   */
  private addAIDecision(action: TradeAction): void {
    const logContainer = document.querySelector('[data-log-content]') as HTMLElement;
    const logEmpty = document.querySelector('[data-log-empty]') as HTMLElement;
    
    if (!logContainer) return;
    
    // Hide empty state
    if (logEmpty) {
      logEmpty.setAttribute('data-log-empty', 'false');
    }
    
    const decision = document.createElement('div');
    decision.className = 'ai-decision-entry';
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
    `;
    
    // Add animation
    decision.style.opacity = '0';
    decision.style.transform = 'translateY(-10px)';
    
    logContainer.insertBefore(decision, logContainer.firstChild);
    
    // Animate in
    requestAnimationFrame(() => {
      decision.style.transition = 'all 0.3s ease';
      decision.style.opacity = '1';
      decision.style.transform = 'translateY(0)';
    });
    
    // Remove old entries
    const entries = logContainer.querySelectorAll('.ai-decision-entry');
    if (entries.length > 20) {
      const oldEntry = entries[entries.length - 1] as HTMLElement;
      oldEntry.style.opacity = '0';
      setTimeout(() => oldEntry.remove(), 300);
    }
  }

  /**
   * Update positions display
   */
  public updatePositions(positions: Position[]): void {
    this.state.positions = positions;
    
    // Update position size in quick stats
    const positionSizeEl = document.querySelector('[data-position-size]') as HTMLElement;
    if (positionSizeEl) {
      const totalSize = positions.reduce((sum, pos) => sum + Math.abs(pos.size), 0);
      positionSizeEl.textContent = totalSize.toFixed(4);
    }
    
    // Update P&L in quick stats
    const pnlEl = document.querySelector('[data-pnl]') as HTMLElement;
    const pnlChangeEl = document.querySelector('[data-pnl-change]') as HTMLElement;
    
    if (pnlEl && pnlChangeEl) {
      const totalPnl = positions.reduce((sum, pos) => sum + pos.pnl, 0);
      const totalPnlPercent = positions.length > 0 ? 
        positions.reduce((sum, pos) => sum + pos.pnl_percent, 0) / positions.length : 0;
      
      pnlEl.textContent = this.formatCurrency(totalPnl);
      pnlChangeEl.textContent = `${totalPnlPercent >= 0 ? '+' : ''}${totalPnlPercent.toFixed(2)}%`;
      pnlChangeEl.className = `stat-change ${totalPnlPercent >= 0 ? 'positive' : 'negative'}`;
      pnlChangeEl.setAttribute('data-pnl-change', totalPnlPercent.toString());
    }

    this.updateLastUpdateTime();
  }

  /**
   * Update risk metrics and gauges
   */
  public updateRiskMetrics(metrics: RiskMetrics): void {
    this.state.risk_metrics = metrics;
    
    // Update risk level gauge
    const riskFill = document.querySelector('[data-risk-percentage]') as HTMLElement;
    const riskText = document.querySelector('[data-risk-text]') as HTMLElement;
    const riskLevelEl = document.querySelector('[data-risk-level]') as HTMLElement;
    const riskColorEl = document.querySelector('[data-risk-color]') as HTMLElement;
    
    const riskPercentage = this.calculateRiskPercentage(metrics);
    const riskLevel = this.getRiskLevel(riskPercentage);
    const riskColor = this.getRiskColor(riskLevel);
    
    if (riskFill) {
      riskFill.style.setProperty('--risk-percentage', `${riskPercentage}%`);
      riskFill.setAttribute('data-risk-percentage', riskPercentage.toString());
    }
    
    if (riskText) {
      riskText.textContent = riskLevel;
      riskText.setAttribute('data-risk-text', riskLevel);
    }
    
    if (riskLevelEl) {
      riskLevelEl.textContent = riskLevel;
      riskLevelEl.setAttribute('data-risk-level', riskLevel.toLowerCase());
    }
    
    if (riskColorEl) {
      riskColorEl.setAttribute('data-risk-color', riskColor);
    }
    
    // Update risk details
    this.updateRiskDetails(metrics);
    
    // Update performance metrics
    this.updatePerformanceMetrics(metrics);

    this.updateLastUpdateTime();
  }

  /**
   * Update risk details panel
   */
  private updateRiskDetails(metrics: RiskMetrics): void {
    const positionValueEl = document.querySelector('[data-position-value]') as HTMLElement;
    const riskPerTradeEl = document.querySelector('[data-risk-per-trade]') as HTMLElement;
    
    if (positionValueEl) {
      const totalPositionValue = this.state.positions.reduce((sum, pos) => 
        sum + Math.abs(pos.size * pos.current_price), 0);
      positionValueEl.textContent = this.formatCurrency(totalPositionValue);
    }
    
    if (riskPerTradeEl) {
      const riskPerTrade = (metrics.margin_used / metrics.total_portfolio_value) * 100;
      riskPerTradeEl.textContent = `${riskPerTrade.toFixed(1)}%`;
    }
  }

  /**
   * Update performance metrics panel
   */
  private updatePerformanceMetrics(metrics: RiskMetrics): void {
    const totalReturnEl = document.querySelector('[data-total-return]') as HTMLElement;
    const winRateEl = document.querySelector('[data-win-rate]') as HTMLElement;
    const avgWinEl = document.querySelector('[data-avg-win]') as HTMLElement;
    const avgLossEl = document.querySelector('[data-avg-loss]') as HTMLElement;
    const maxDrawdownEl = document.querySelector('[data-max-drawdown]') as HTMLElement;
    const sharpeRatioEl = document.querySelector('[data-sharpe-ratio]') as HTMLElement;
    
    if (totalReturnEl) {
      const totalReturn = (metrics.total_pnl / metrics.total_portfolio_value) * 100;
      totalReturnEl.textContent = `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`;
      totalReturnEl.className = `metric-value ${totalReturn >= 0 ? 'positive' : 'negative'}`;
    }
    
    if (winRateEl) {
      winRateEl.textContent = `${(metrics.win_rate * 100).toFixed(0)}%`;
    }
    
    if (avgWinEl) {
      avgWinEl.textContent = this.formatCurrency(this.performanceMetrics.avgWin);
    }
    
    if (avgLossEl) {
      avgLossEl.textContent = this.formatCurrency(this.performanceMetrics.avgLoss);
    }
    
    if (maxDrawdownEl) {
      maxDrawdownEl.textContent = `${(metrics.max_drawdown * 100).toFixed(2)}%`;
      maxDrawdownEl.className = 'metric-value negative';
    }
    
    if (sharpeRatioEl) {
      sharpeRatioEl.textContent = this.performanceMetrics.sharpeRatio.toFixed(2);
    }
  }

  /**
   * Update system health indicators
   */
  public updateSystemHealth(healthData: {
    api: 'connected' | 'disconnected' | 'error';
    websocket: 'connected' | 'disconnected' | 'error';
    llm: 'ready' | 'error' | 'loading';
    indicators: 'calculating' | 'ready' | 'error';
  }): void {
    const healthItems = [
      { key: 'api', selector: '[data-api-status]', indicator: '[data-api-indicator]' },
      { key: 'websocket', selector: '[data-websocket-status]', indicator: '[data-websocket-indicator]' },
      { key: 'llm', selector: '[data-llm-status]', indicator: '[data-llm-indicator]' },
      { key: 'indicators', selector: '[data-indicators-status]', indicator: '[data-indicators-indicator]' }
    ];

    healthItems.forEach(({ key, selector, indicator }) => {
      const statusEl = document.querySelector(selector) as HTMLElement;
      const indicatorEl = document.querySelector(indicator) as HTMLElement;
      const status = healthData[key as keyof typeof healthData];
      
      if (statusEl) {
        statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        statusEl.setAttribute(`data-${key}-status`, status);
      }
      
      if (indicatorEl) {
        const color = this.getHealthColor(status);
        indicatorEl.setAttribute(`data-${key}-indicator`, color);
      }
    });
  }

  /**
   * Update footer system information
   */
  public updateSystemInfo(info: {
    version?: string;
    uptime?: number;
    memoryUsage?: number;
    serverTime?: Date;
    marketStatus?: 'open' | 'closed';
  }): void {
    const versionEl = document.querySelector('[data-version]') as HTMLElement;
    const uptimeEl = document.querySelector('[data-uptime]') as HTMLElement;
    const memoryEl = document.querySelector('[data-memory-usage]') as HTMLElement;
    const serverTimeEl = document.querySelector('[data-server-time]') as HTMLElement;
    const marketStatusEl = document.querySelector('[data-market-status]') as HTMLElement;
    
    if (versionEl && info.version) {
      versionEl.textContent = info.version;
    }
    
    if (uptimeEl && info.uptime) {
      uptimeEl.textContent = this.formatUptime(info.uptime);
    }
    
    if (memoryEl && info.memoryUsage) {
      memoryEl.textContent = `${info.memoryUsage.toFixed(0)}MB`;
    }
    
    if (serverTimeEl && info.serverTime) {
      serverTimeEl.textContent = info.serverTime.toLocaleTimeString();
    }
    
    if (marketStatusEl && info.marketStatus) {
      marketStatusEl.textContent = info.marketStatus.charAt(0).toUpperCase() + info.marketStatus.slice(1);
      marketStatusEl.className = `timestamp-value ${info.marketStatus}`;
    }
  }

  /**
   * Update API status indicators in footer
   */
  public updateAPIStatus(apiStatus: {
    coinbase: 'connected' | 'disconnected' | 'error';
    openai: 'connected' | 'disconnected' | 'error';
  }): void {
    const coinbaseIndicator = document.querySelector('[data-coinbase-status]') as HTMLElement;
    const openaiIndicator = document.querySelector('[data-openai-status]') as HTMLElement;
    
    if (coinbaseIndicator) {
      coinbaseIndicator.className = `api-indicator ${apiStatus.coinbase}`;
      coinbaseIndicator.setAttribute('data-coinbase-status', apiStatus.coinbase);
    }
    
    if (openaiIndicator) {
      openaiIndicator.className = `api-indicator ${apiStatus.openai}`;
      openaiIndicator.setAttribute('data-openai-status', apiStatus.openai);
    }
  }

  /**
   * Show error modal with details
   */
  public showError(message: string, details?: string): void {
    this.state.error_message = message;
    
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.querySelector('[data-error-message]') as HTMLElement;
    const errorStack = document.querySelector('[data-error-stack]') as HTMLElement;
    
    if (errorModal) {
      errorModal.setAttribute('data-modal', 'visible');
      document.body.style.overflow = 'hidden'; // Prevent background scroll
    }
    
    if (errorMessage) {
      errorMessage.textContent = message;
    }
    
    if (errorStack && details) {
      errorStack.textContent = details;
    }
    
    this.log('error', message, 'System');
  }

  /**
   * Hide error modal
   */
  public hideError(): void {
    this.state.error_message = null;
    
    const errorModal = document.getElementById('error-modal');
    if (errorModal) {
      errorModal.setAttribute('data-modal', 'hidden');
      document.body.style.overflow = ''; // Restore scroll
    }
  }

  /**
   * Show chart loading state
   */
  public showChartLoading(): void {
    const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement;
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement;
    
    if (chartLoading) chartLoading.setAttribute('data-chart-loading', 'true');
    if (chartError) chartError.setAttribute('data-chart-error', 'hidden');
  }

  /**
   * Hide chart loading state
   */
  public hideChartLoading(): void {
    const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement;
    if (chartLoading) chartLoading.setAttribute('data-chart-loading', 'false');
  }

  /**
   * Show chart error state
   */
  public showChartError(message: string = 'Failed to load chart'): void {
    const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement;
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement;
    const errorMessage = chartError?.querySelector('p');
    
    if (chartLoading) chartLoading.setAttribute('data-chart-loading', 'false');
    if (chartError) chartError.setAttribute('data-chart-error', 'visible');
    if (errorMessage) errorMessage.textContent = message;
    
    this.log('error', `Chart error: ${message}`, 'Chart');
  }

  /**
   * Hide chart error state
   */
  public hideChartError(): void {
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement;
    if (chartError) chartError.setAttribute('data-chart-error', 'hidden');
  }

  /**
   * Toggle chart fullscreen mode
   */
  private toggleChartFullscreen(): void {
    const chartSection = document.querySelector('.chart-section') as HTMLElement;
    const dashboard = document.querySelector('.dashboard-container') as HTMLElement;
    
    if (chartSection && dashboard) {
      const isFullscreen = chartSection.hasAttribute('data-fullscreen');
      
      if (isFullscreen) {
        chartSection.removeAttribute('data-fullscreen');
        dashboard.classList.remove('chart-fullscreen');
      } else {
        chartSection.setAttribute('data-fullscreen', 'true');
        dashboard.classList.add('chart-fullscreen');
      }
    }
  }

  /**
   * Add log entry with improved formatting
   */
  public log(level: 'info' | 'warn' | 'error' | 'debug', message: string, component?: string): void {
    if (this.logPaused) return;
    
    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date().toISOString(),
      component
    };
    
    this.logEntries.unshift(entry);
    
    // Keep only the latest entries
    if (this.logEntries.length > this.maxLogEntries) {
      this.logEntries = this.logEntries.slice(0, this.maxLogEntries);
    }
  }

  /**
   * Clear all log entries
   */
  public clearLogs(): void {
    this.logEntries = [];
    const logContent = document.querySelector('[data-log-content]') as HTMLElement;
    const logEmpty = document.querySelector('[data-log-empty]') as HTMLElement;
    
    if (logContent) {
      logContent.innerHTML = '';
    }
    
    if (logEmpty) {
      logEmpty.setAttribute('data-log-empty', 'true');
    }
    
    this.log('info', 'Logs cleared', 'UI');
  }

  /**
   * Toggle log pause state
   */
  public toggleLogPause(): void {
    this.logPaused = !this.logPaused;
    const pauseBtn = document.querySelector('[data-log-action="pause"]') as HTMLButtonElement;
    if (pauseBtn) {
      pauseBtn.textContent = this.logPaused ? 'Resume' : 'Pause';
      pauseBtn.setAttribute('aria-label', this.logPaused ? 'Resume log updates' : 'Pause log updates');
    }
    this.log('info', `Log updates ${this.logPaused ? 'paused' : 'resumed'}`, 'UI');
  }

  // Helper methods
  private updateLastUpdateTime(): void {
    const lastUpdateEl = document.querySelector('.update-time') as HTMLElement;
    if (lastUpdateEl) {
      lastUpdateEl.textContent = new Date().toLocaleTimeString();
    }
  }

  private addAnimation(element: HTMLElement, animationClass: string): void {
    if (this.animationQueue.has(element.id || element.className)) return;
    
    this.animationQueue.add(element.id || element.className);
    element.classList.add(animationClass);
    
    setTimeout(() => {
      element.classList.remove(animationClass);
      this.animationQueue.delete(element.id || element.className);
    }, 1000);
  }

  private shouldThrottleUpdate(key: string, throttleMs: number): boolean {
    const now = Date.now();
    const lastUpdate = this.updateThrottles.get(key) || 0;
    
    if (now - lastUpdate < throttleMs) {
      return true;
    }
    
    this.updateThrottles.set(key, now);
    return false;
  }

  private formatPrice(price: number): string {
    return `$${price.toLocaleString(undefined, { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    })}`;
  }

  private formatCurrency(amount: number): string {
    return `$${amount.toLocaleString(undefined, { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    })}`;
  }

  private formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  private formatReasoning(reasoning: string): string {
    // Basic HTML escaping and formatting
    return reasoning
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>');
  }

  private getConnectionStatusText(status: ConnectionStatus): string {
    switch (status) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'disconnected': return 'Disconnected';
      case 'error': return 'Connection Error';
      default: return 'Unknown';
    }
  }

  private getBotIndicatorState(status: string): string {
    switch (status) {
      case 'running': return 'green';
      case 'stopped': return 'red';
      case 'error': return 'red';
      case 'dry_run': return 'yellow';
      default: return 'gray';
    }
  }

  private calculateRiskPercentage(metrics: RiskMetrics): number {
    const riskRatio = metrics.margin_used / metrics.total_portfolio_value;
    return Math.min(riskRatio * 100, 100);
  }

  private getRiskLevel(percentage: number): string {
    if (percentage < 20) return 'Low';
    if (percentage < 50) return 'Medium';
    if (percentage < 80) return 'High';
    return 'Critical';
  }

  private getRiskColor(level: string): string {
    switch (level.toLowerCase()) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'orange';
      case 'critical': return 'red';
      default: return 'gray';
    }
  }

  private getHealthColor(status: string): string {
    switch (status) {
      case 'connected':
      case 'ready': return 'green';
      case 'calculating':
      case 'loading': return 'yellow';
      case 'disconnected':
      case 'error': return 'red';
      default: return 'gray';
    }
  }

  private startPerformanceMonitoring(): void {
    // Monitor and update performance metrics periodically
    setInterval(() => {
      this.updateSystemInfo({
        serverTime: new Date(),
        memoryUsage: (performance as any).memory?.usedJSHeapSize / 1024 / 1024 || 0
      });
    }, 1000);
  }

  // Event callback setters
  public onChartRetryRequested(callback: () => void): void {
    this.onChartRetry = callback;
  }

  public onErrorRetryRequested(callback: () => void): void {
    this.onErrorRetry = callback;
  }

  /**
   * Clear error message (alias for hideError)
   */
  public clearError(): void {
    this.hideError();
  }

  /**
   * Set symbol change callback (placeholder for compatibility)
   */
  public onSymbolChanged(callback: (symbol: string) => void): void {
    // This would be implemented if we had symbol selection controls
    console.log('Symbol change callback set:', callback);
  }

  /**
   * Set interval change callback (placeholder for compatibility)
   */
  public onIntervalChanged(callback: (interval: string) => void): void {
    // This would be implemented if we had interval selection controls
    console.log('Interval change callback set:', callback);
  }

  /**
   * Set chart fullscreen callback (placeholder for compatibility)
   */
  public onChartFullscreenToggle(callback: () => void): void {
    // This would be implemented if we had additional fullscreen handling
    console.log('Chart fullscreen callback set:', callback);
  }

  /**
   * Get current dashboard state
   */
  public getState(): DashboardState {
    return { ...this.state };
  }

  /**
   * Update indicators display
   */
  public updateIndicators(indicators: VuManchuIndicators): void {
    this.state.indicators = indicators;
    // Update any indicator-specific UI elements if needed
    this.updateLastUpdateTime();
  }

  /**
   * Batch update multiple components for better performance
   */
  public batchUpdate(updates: {
    botStatus?: BotStatus;
    marketData?: MarketData;
    tradeAction?: TradeAction;
    positions?: Position[];
    riskMetrics?: RiskMetrics;
    indicators?: VuManchuIndicators;
  }): void {
    // Use requestAnimationFrame to batch DOM updates
    requestAnimationFrame(() => {
      if (updates.botStatus) this.updateBotStatus(updates.botStatus);
      if (updates.marketData) this.updateMarketData(updates.marketData);
      if (updates.tradeAction) this.updateLatestAction(updates.tradeAction);
      if (updates.positions) this.updatePositions(updates.positions);
      if (updates.riskMetrics) this.updateRiskMetrics(updates.riskMetrics);
      if (updates.indicators) this.updateIndicators(updates.indicators);
    });
  }
}