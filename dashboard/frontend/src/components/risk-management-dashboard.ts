/**
 * Advanced Risk Management Dashboard Component
 * 
 * Provides comprehensive risk management tools with:
 * - Dynamic risk limit configuration
 * - Real-time risk monitoring and alerts
 * - Portfolio exposure analysis
 * - Stress testing and scenario analysis
 * - Risk metrics visualization
 * - Automated risk controls
 */

import type { Position, MarketData, RiskMetrics } from '../types';

export interface RiskConfiguration {
  max_position_size: number;
  max_portfolio_exposure: number;
  stop_loss_percentage: number;
  take_profit_percentage: number;
  max_daily_loss: number;
  max_weekly_loss: number;
  max_monthly_loss: number;
  max_drawdown: number;
  leverage_limit: number;
  correlation_limit: number;
  concentration_limit: number;
  volatility_threshold: number;
}

export interface RiskAlert {
  id: string;
  type: 'exposure' | 'loss' | 'drawdown' | 'volatility' | 'correlation' | 'concentration';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  threshold: number;
  current_value: number;
  timestamp: string;
  acknowledged: boolean;
}

export interface PortfolioExposure {
  symbol: string;
  position_size: number;
  market_value: number;
  percentage_of_portfolio: number;
  unrealized_pnl: number;
  risk_score: number;
  correlation_score: number;
}

export interface RiskScenario {
  name: string;
  description: string;
  market_shock: number; // percentage
  expected_loss: number;
  probability: number;
  var_95: number; // Value at Risk 95%
  var_99: number; // Value at Risk 99%
  expected_shortfall: number;
}

export class RiskManagementDashboard {
  private container: HTMLElement;
  private apiBaseUrl: string;
  private currentRiskConfig: RiskConfiguration | null = null;
  private activeAlerts: RiskAlert[] = [];
  private portfolioExposures: PortfolioExposure[] = [];
  private riskScenarios: RiskScenario[] = [];
  private currentRiskMetrics: RiskMetrics | null = null;
  private updateInterval: number | null = null;
  private onRiskAlert?: (alert: RiskAlert) => void;
  private onConfigUpdate?: (config: RiskConfiguration) => void;

  constructor(containerId: string, apiBaseUrl: string) {
    const container = document.getElementById(containerId);
    if (!container) {
      throw new Error(`Container element with ID ${containerId} not found`);
    }
    
    this.container = container;
    this.apiBaseUrl = apiBaseUrl;
    this.render();
    this.loadRiskConfiguration();
    this.startRealtimeUpdates();
  }

  /**
   * Set event handlers
   */
  public onRiskAlert(callback: (alert: RiskAlert) => void): void {
    this.onRiskAlert = callback;
  }

  public onConfigUpdate(callback: (config: RiskConfiguration) => void): void {
    this.onConfigUpdate = callback;
  }

  /**
   * Update risk metrics
   */
  public updateRiskMetrics(riskMetrics: RiskMetrics): void {
    this.currentRiskMetrics = riskMetrics;
    this.updateRiskMetricsDisplay();
    this.checkRiskThresholds();
  }

  /**
   * Update portfolio exposures
   */
  public updatePortfolioExposures(exposures: PortfolioExposure[]): void {
    this.portfolioExposures = exposures;
    this.updateExposureDisplay();
  }

  /**
   * Render the main interface
   */
  private render(): void {
    this.container.innerHTML = `
      <div class="risk-management-dashboard">
        <!-- Header -->
        <div class="risk-header">
          <div class="header-title">
            <h3>Risk Management Dashboard</h3>
            <div class="risk-status-indicator">
              <span class="status-dot" id="risk-status-dot"></span>
              <span class="status-text" id="risk-status-text">Monitoring</span>
            </div>
          </div>
          <div class="header-controls">
            <button class="stress-test-btn" id="run-stress-test">🧪 Stress Test</button>
            <button class="reset-alerts-btn" id="reset-alerts">🔔 Reset Alerts</button>
            <button class="export-config-btn" id="export-config">📥 Export Config</button>
          </div>
        </div>

        <!-- Risk Overview Cards -->
        <div class="risk-overview">
          <div class="risk-card">
            <div class="card-header">
              <h4>Portfolio Risk</h4>
              <span class="risk-score" id="portfolio-risk-score">--</span>
            </div>
            <div class="card-metrics">
              <div class="metric">
                <span class="label">VaR (95%)</span>
                <span class="value" id="var-95">$0.00</span>
              </div>
              <div class="metric">
                <span class="label">Max Drawdown</span>
                <span class="value" id="max-drawdown">0.0%</span>
              </div>
              <div class="metric">
                <span class="label">Sharpe Ratio</span>
                <span class="value" id="sharpe-ratio">0.00</span>
              </div>
            </div>
          </div>

          <div class="risk-card">
            <div class="card-header">
              <h4>Exposure Analysis</h4>
              <span class="exposure-level" id="exposure-level">--</span>
            </div>
            <div class="card-metrics">
              <div class="metric">
                <span class="label">Total Exposure</span>
                <span class="value" id="total-exposure">$0.00</span>
              </div>
              <div class="metric">
                <span class="label">Concentration</span>
                <span class="value" id="concentration-ratio">0.0%</span>
              </div>
              <div class="metric">
                <span class="label">Correlation</span>
                <span class="value" id="avg-correlation">0.00</span>
              </div>
            </div>
          </div>

          <div class="risk-card">
            <div class="card-header">
              <h4>P&L Limits</h4>
              <span class="pnl-status" id="pnl-status">--</span>
            </div>
            <div class="card-metrics">
              <div class="metric">
                <span class="label">Daily P&L</span>
                <span class="value" id="daily-pnl-limit">$0.00</span>
              </div>
              <div class="metric">
                <span class="label">Weekly P&L</span>
                <span class="value" id="weekly-pnl-limit">$0.00</span>
              </div>
              <div class="metric">
                <span class="label">Monthly P&L</span>
                <span class="value" id="monthly-pnl-limit">$0.00</span>
              </div>
            </div>
          </div>

          <div class="risk-card">
            <div class="card-header">
              <h4>Active Alerts</h4>
              <span class="alert-count" id="alert-count">0</span>
            </div>
            <div class="card-metrics">
              <div class="metric">
                <span class="label">Critical</span>
                <span class="value alert-critical" id="critical-alerts">0</span>
              </div>
              <div class="metric">
                <span class="label">High</span>
                <span class="value alert-high" id="high-alerts">0</span>
              </div>
              <div class="metric">
                <span class="label">Medium</span>
                <span class="value alert-medium" id="medium-alerts">0</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Main Content Tabs -->
        <div class="risk-content">
          <div class="content-tabs">
            <button class="tab-btn active" data-tab="configuration">Risk Configuration</button>
            <button class="tab-btn" data-tab="exposure">Portfolio Exposure</button>
            <button class="tab-btn" data-tab="scenarios">Scenario Analysis</button>
            <button class="tab-btn" data-tab="alerts">Risk Alerts</button>
          </div>

          <!-- Risk Configuration Tab -->
          <div class="tab-content active" id="configuration-tab">
            <div class="config-section">
              <div class="config-grid">
                <div class="config-group">
                  <h5>Position Limits</h5>
                  <div class="form-row">
                    <div class="form-group">
                      <label for="max-position-size">Max Position Size (%)</label>
                      <input type="number" id="max-position-size" min="1" max="100" step="1" class="form-control">
                    </div>
                    <div class="form-group">
                      <label for="max-portfolio-exposure">Max Portfolio Exposure (%)</label>
                      <input type="number" id="max-portfolio-exposure" min="1" max="500" step="1" class="form-control">
                    </div>
                  </div>
                  <div class="form-row">
                    <div class="form-group">
                      <label for="leverage-limit">Max Leverage</label>
                      <input type="number" id="leverage-limit" min="1" max="100" step="1" class="form-control">
                    </div>
                    <div class="form-group">
                      <label for="concentration-limit">Concentration Limit (%)</label>
                      <input type="number" id="concentration-limit" min="1" max="100" step="1" class="form-control">
                    </div>
                  </div>
                </div>

                <div class="config-group">
                  <h5>Stop Loss & Take Profit</h5>
                  <div class="form-row">
                    <div class="form-group">
                      <label for="stop-loss-pct">Stop Loss (%)</label>
                      <input type="number" id="stop-loss-pct" min="0.1" max="50" step="0.1" class="form-control">
                    </div>
                    <div class="form-group">
                      <label for="take-profit-pct">Take Profit (%)</label>
                      <input type="number" id="take-profit-pct" min="0.1" max="100" step="0.1" class="form-control">
                    </div>
                  </div>
                  <div class="form-row">
                    <div class="form-group">
                      <label for="max-drawdown">Max Drawdown (%)</label>
                      <input type="number" id="max-drawdown" min="1" max="50" step="0.1" class="form-control">
                    </div>
                    <div class="form-group">
                      <label for="volatility-threshold">Volatility Threshold (%)</label>
                      <input type="number" id="volatility-threshold" min="1" max="100" step="1" class="form-control">
                    </div>
                  </div>
                </div>

                <div class="config-group">
                  <h5>P&L Limits</h5>
                  <div class="form-row">
                    <div class="form-group">
                      <label for="max-daily-loss">Max Daily Loss ($)</label>
                      <input type="number" id="max-daily-loss" min="1" max="50000" step="1" class="form-control">
                    </div>
                    <div class="form-group">
                      <label for="max-weekly-loss">Max Weekly Loss ($)</label>
                      <input type="number" id="max-weekly-loss" min="1" max="100000" step="1" class="form-control">
                    </div>
                  </div>
                  <div class="form-row">
                    <div class="form-group">
                      <label for="max-monthly-loss">Max Monthly Loss ($)</label>
                      <input type="number" id="max-monthly-loss" min="1" max="500000" step="1" class="form-control">
                    </div>
                    <div class="form-group">
                      <label for="correlation-limit">Correlation Limit</label>
                      <input type="number" id="correlation-limit" min="0" max="1" step="0.01" class="form-control">
                    </div>
                  </div>
                </div>
              </div>

              <div class="config-actions">
                <button class="save-config-btn" id="save-config">💾 Save Configuration</button>
                <button class="reset-config-btn" id="reset-config">🔄 Reset to Defaults</button>
                <button class="validate-config-btn" id="validate-config">✅ Validate Config</button>
              </div>
            </div>
          </div>

          <!-- Portfolio Exposure Tab -->
          <div class="tab-content" id="exposure-tab">
            <div class="exposure-section">
              <div class="exposure-header">
                <h5>Portfolio Exposure Analysis</h5>
                <div class="exposure-controls">
                  <button class="refresh-exposure-btn" id="refresh-exposure">🔄 Refresh</button>
                  <select id="exposure-timeframe" class="form-control">
                    <option value="1d">1 Day</option>
                    <option value="7d">7 Days</option>
                    <option value="30d">30 Days</option>
                  </select>
                </div>
              </div>
              <div class="exposure-table-container">
                <table class="exposure-table" id="exposure-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Position Size</th>
                      <th>Market Value</th>
                      <th>% of Portfolio</th>
                      <th>Unrealized P&L</th>
                      <th>Risk Score</th>
                      <th>Correlation</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody id="exposure-table-body">
                    <tr class="empty-state">
                      <td colspan="8">No positions to analyze</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Scenario Analysis Tab -->
          <div class="tab-content" id="scenarios-tab">
            <div class="scenarios-section">
              <div class="scenarios-header">
                <h5>Stress Testing & Scenario Analysis</h5>
                <button class="add-scenario-btn" id="add-scenario">➕ Add Scenario</button>
              </div>
              <div class="scenarios-grid" id="scenarios-grid">
                <!-- Scenarios will be populated here -->
              </div>
            </div>
          </div>

          <!-- Risk Alerts Tab -->
          <div class="tab-content" id="alerts-tab">
            <div class="alerts-section">
              <div class="alerts-header">
                <h5>Active Risk Alerts</h5>
                <div class="alerts-controls">
                  <button class="acknowledge-all-btn" id="acknowledge-all">✅ Acknowledge All</button>
                  <select id="alert-filter" class="form-control">
                    <option value="all">All Alerts</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
              </div>
              <div class="alerts-list" id="alerts-list">
                <div class="empty-state">No active alerts</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    this.attachEventListeners();
  }

  /**
   * Attach event listeners
   */
  private attachEventListeners(): void {
    // Header controls
    const stressTestBtn = document.getElementById('run-stress-test');
    const resetAlertsBtn = document.getElementById('reset-alerts');
    const exportConfigBtn = document.getElementById('export-config');

    stressTestBtn?.addEventListener('click', () => this.runStressTest());
    resetAlertsBtn?.addEventListener('click', () => this.resetAlerts());
    exportConfigBtn?.addEventListener('click', () => this.exportConfiguration());

    // Configuration controls
    const saveConfigBtn = document.getElementById('save-config');
    const resetConfigBtn = document.getElementById('reset-config');
    const validateConfigBtn = document.getElementById('validate-config');

    saveConfigBtn?.addEventListener('click', () => this.saveConfiguration());
    resetConfigBtn?.addEventListener('click', () => this.resetConfiguration());
    validateConfigBtn?.addEventListener('click', () => this.validateConfiguration());

    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tabName = (e.target as HTMLElement).dataset.tab;
        if (tabName) this.switchTab(tabName);
      });
    });

    // Exposure controls
    const refreshExposureBtn = document.getElementById('refresh-exposure');
    const exposureTimeframeSelect = document.getElementById('exposure-timeframe');

    refreshExposureBtn?.addEventListener('click', () => this.refreshExposureAnalysis());
    exposureTimeframeSelect?.addEventListener('change', () => this.refreshExposureAnalysis());

    // Alerts controls
    const acknowledgeAllBtn = document.getElementById('acknowledge-all');
    const alertFilterSelect = document.getElementById('alert-filter');

    acknowledgeAllBtn?.addEventListener('click', () => this.acknowledgeAllAlerts());
    alertFilterSelect?.addEventListener('change', () => this.filterAlerts());

    // Scenario controls
    const addScenarioBtn = document.getElementById('add-scenario');
    addScenarioBtn?.addEventListener('click', () => this.showAddScenarioDialog());

    // Form validation on input changes
    const configInputs = document.querySelectorAll('#configuration-tab input');
    configInputs.forEach(input => {
      input.addEventListener('input', () => this.validateConfigurationInput());
    });
  }

  /**
   * Load current risk configuration
   */
  private async loadRiskConfiguration(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/risk/configuration`);
      if (response.ok) {
        const config = await response.json();
        this.currentRiskConfig = config;
        this.populateConfigurationForm(config);
      }
    } catch (error) {
      console.warn('Failed to load risk configuration:', error);
      this.loadDefaultConfiguration();
    }
  }

  /**
   * Load default configuration
   */
  private loadDefaultConfiguration(): void {
    const defaultConfig: RiskConfiguration = {
      max_position_size: 50,
      max_portfolio_exposure: 200,
      stop_loss_percentage: 2.0,
      take_profit_percentage: 4.0,
      max_daily_loss: 1000,
      max_weekly_loss: 5000,
      max_monthly_loss: 15000,
      max_drawdown: 10.0,
      leverage_limit: 5,
      correlation_limit: 0.7,
      concentration_limit: 30,
      volatility_threshold: 50
    };

    this.currentRiskConfig = defaultConfig;
    this.populateConfigurationForm(defaultConfig);
  }

  /**
   * Populate configuration form
   */
  private populateConfigurationForm(config: RiskConfiguration): void {
    const inputs = [
      { id: 'max-position-size', value: config.max_position_size },
      { id: 'max-portfolio-exposure', value: config.max_portfolio_exposure },
      { id: 'stop-loss-pct', value: config.stop_loss_percentage },
      { id: 'take-profit-pct', value: config.take_profit_percentage },
      { id: 'max-daily-loss', value: config.max_daily_loss },
      { id: 'max-weekly-loss', value: config.max_weekly_loss },
      { id: 'max-monthly-loss', value: config.max_monthly_loss },
      { id: 'max-drawdown', value: config.max_drawdown },
      { id: 'leverage-limit', value: config.leverage_limit },
      { id: 'correlation-limit', value: config.correlation_limit },
      { id: 'concentration-limit', value: config.concentration_limit },
      { id: 'volatility-threshold', value: config.volatility_threshold }
    ];

    inputs.forEach(({ id, value }) => {
      const input = document.getElementById(id) as HTMLInputElement;
      if (input) input.value = value.toString();
    });
  }

  /**
   * Save configuration
   */
  private async saveConfiguration(): Promise<void> {
    const config = this.getConfigurationFromForm();
    if (!config) return;

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/risk/configuration`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        this.currentRiskConfig = config;
        this.onConfigUpdate?.(config);
        this.showSuccessMessage('Risk configuration saved successfully');
      } else {
        const result = await response.json();
        throw new Error(result.detail || 'Failed to save configuration');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.showErrorMessage(`Failed to save configuration: ${errorMessage}`);
    }
  }

  /**
   * Get configuration from form
   */
  private getConfigurationFromForm(): RiskConfiguration | null {
    try {
      const getValue = (id: string): number => {
        const input = document.getElementById(id) as HTMLInputElement;
        return parseFloat(input?.value || '0');
      };

      return {
        max_position_size: getValue('max-position-size'),
        max_portfolio_exposure: getValue('max-portfolio-exposure'),
        stop_loss_percentage: getValue('stop-loss-pct'),
        take_profit_percentage: getValue('take-profit-pct'),
        max_daily_loss: getValue('max-daily-loss'),
        max_weekly_loss: getValue('max-weekly-loss'),
        max_monthly_loss: getValue('max-monthly-loss'),
        max_drawdown: getValue('max-drawdown'),
        leverage_limit: getValue('leverage-limit'),
        correlation_limit: getValue('correlation-limit'),
        concentration_limit: getValue('concentration-limit'),
        volatility_threshold: getValue('volatility-threshold')
      };
    } catch (error) {
      this.showErrorMessage('Invalid configuration values');
      return null;
    }
  }

  /**
   * Validate configuration
   */
  private validateConfiguration(): void {
    const config = this.getConfigurationFromForm();
    if (!config) return;

    const validationResults = [];

    // Validate ranges
    if (config.max_position_size <= 0 || config.max_position_size > 100) {
      validationResults.push('Max position size must be between 1% and 100%');
    }
    if (config.stop_loss_percentage <= 0 || config.stop_loss_percentage > 50) {
      validationResults.push('Stop loss must be between 0.1% and 50%');
    }
    if (config.max_daily_loss <= 0) {
      validationResults.push('Max daily loss must be positive');
    }
    if (config.leverage_limit < 1 || config.leverage_limit > 100) {
      validationResults.push('Leverage limit must be between 1 and 100');
    }
    if (config.correlation_limit < 0 || config.correlation_limit > 1) {
      validationResults.push('Correlation limit must be between 0 and 1');
    }

    // Validate logical relationships
    if (config.take_profit_percentage <= config.stop_loss_percentage) {
      validationResults.push('Take profit should be greater than stop loss');
    }
    if (config.max_weekly_loss <= config.max_daily_loss) {
      validationResults.push('Weekly loss limit should be greater than daily limit');
    }
    if (config.max_monthly_loss <= config.max_weekly_loss) {
      validationResults.push('Monthly loss limit should be greater than weekly limit');
    }

    if (validationResults.length === 0) {
      this.showSuccessMessage('Configuration validation passed');
    } else {
      this.showErrorMessage(`Validation failed:\n${validationResults.join('\n')}`);
    }
  }

  /**
   * Validate configuration input in real-time
   */
  private validateConfigurationInput(): void {
    // Real-time validation logic
    const config = this.getConfigurationFromForm();
    if (!config) return;

    // Update save button state based on validation
    const saveBtn = document.getElementById('save-config') as HTMLButtonElement;
    if (saveBtn) {
      const isValid = this.isConfigurationValid(config);
      saveBtn.disabled = !isValid;
    }
  }

  /**
   * Check if configuration is valid
   */
  private isConfigurationValid(config: RiskConfiguration): boolean {
    return config.max_position_size > 0 && 
           config.max_position_size <= 100 &&
           config.stop_loss_percentage > 0 &&
           config.max_daily_loss > 0 &&
           config.take_profit_percentage > config.stop_loss_percentage;
  }

  /**
   * Update risk metrics display
   */
  private updateRiskMetricsDisplay(): void {
    if (!this.currentRiskMetrics) return;

    const metrics = this.currentRiskMetrics;
    
    // Update overview cards
    const portfolioRiskScore = this.calculatePortfolioRiskScore(metrics);
    this.updateElement('portfolio-risk-score', this.formatRiskScore(portfolioRiskScore));
    this.updateElement('var-95', `$${(metrics.value_at_risk_95 || 0).toFixed(2)}`);
    this.updateElement('max-drawdown', `${(metrics.max_drawdown || 0).toFixed(1)}%`);
    this.updateElement('sharpe-ratio', (metrics.sharpe_ratio || 0).toFixed(2));

    this.updateElement('total-exposure', `$${(metrics.total_exposure || 0).toFixed(2)}`);
    this.updateElement('concentration-ratio', `${(metrics.concentration_ratio || 0).toFixed(1)}%`);
    this.updateElement('avg-correlation', (metrics.avg_correlation || 0).toFixed(2));

    this.updateElement('daily-pnl-limit', `$${(metrics.daily_pnl || 0).toFixed(2)}`);
    this.updateElement('weekly-pnl-limit', `$${(metrics.weekly_pnl || 0).toFixed(2)}`);
    this.updateElement('monthly-pnl-limit', `$${(metrics.monthly_pnl || 0).toFixed(2)}`);

    // Update status indicators
    this.updateRiskStatusIndicator(portfolioRiskScore);
    this.updateExposureLevelIndicator(metrics.total_exposure || 0);
    this.updatePnLStatusIndicator(metrics.daily_pnl || 0);
  }

  /**
   * Calculate portfolio risk score
   */
  private calculatePortfolioRiskScore(metrics: RiskMetrics): number {
    // Simplified risk scoring algorithm
    let score = 0;
    
    score += Math.min((metrics.max_drawdown || 0) * 2, 40); // Max 40 points
    score += Math.min((metrics.volatility || 0), 30); // Max 30 points
    score += Math.min((metrics.concentration_ratio || 0), 20); // Max 20 points
    score += Math.min(Math.abs(metrics.daily_pnl || 0) / 100, 10); // Max 10 points
    
    return Math.min(score, 100);
  }

  /**
   * Format risk score
   */
  private formatRiskScore(score: number): string {
    if (score < 30) return 'Low';
    if (score < 60) return 'Medium';
    if (score < 80) return 'High';
    return 'Critical';
  }

  /**
   * Update status indicators
   */
  private updateRiskStatusIndicator(riskScore: number): void {
    const dot = document.getElementById('risk-status-dot');
    const text = document.getElementById('risk-status-text');
    
    if (!dot || !text) return;

    if (riskScore < 30) {
      dot.className = 'status-dot status-low';
      text.textContent = 'Low Risk';
    } else if (riskScore < 60) {
      dot.className = 'status-dot status-medium';
      text.textContent = 'Medium Risk';
    } else if (riskScore < 80) {
      dot.className = 'status-dot status-high';
      text.textContent = 'High Risk';
    } else {
      dot.className = 'status-dot status-critical';
      text.textContent = 'Critical Risk';
    }
  }

  private updateExposureLevelIndicator(exposure: number): void {
    const exposureLevel = document.getElementById('exposure-level');
    if (!exposureLevel) return;

    if (exposure < 1000) {
      exposureLevel.textContent = 'Low';
      exposureLevel.className = 'exposure-level exposure-low';
    } else if (exposure < 5000) {
      exposureLevel.textContent = 'Medium';
      exposureLevel.className = 'exposure-level exposure-medium';
    } else {
      exposureLevel.textContent = 'High';
      exposureLevel.className = 'exposure-level exposure-high';
    }
  }

  private updatePnLStatusIndicator(dailyPnL: number): void {
    const pnlStatus = document.getElementById('pnl-status');
    if (!pnlStatus) return;

    if (dailyPnL >= 0) {
      pnlStatus.textContent = 'Positive';
      pnlStatus.className = 'pnl-status pnl-positive';
    } else if (dailyPnL > -500) {
      pnlStatus.textContent = 'Warning';
      pnlStatus.className = 'pnl-status pnl-warning';
    } else {
      pnlStatus.textContent = 'Critical';
      pnlStatus.className = 'pnl-status pnl-critical';
    }
  }

  /**
   * Check risk thresholds and generate alerts
   */
  private checkRiskThresholds(): void {
    if (!this.currentRiskMetrics || !this.currentRiskConfig) return;

    const newAlerts: RiskAlert[] = [];
    const metrics = this.currentRiskMetrics;
    const config = this.currentRiskConfig;

    // Check daily loss limit
    if (Math.abs(metrics.daily_pnl || 0) > config.max_daily_loss) {
      newAlerts.push({
        id: `daily-loss-${Date.now()}`,
        type: 'loss',
        severity: 'critical',
        title: 'Daily Loss Limit Exceeded',
        message: `Daily loss of $${Math.abs(metrics.daily_pnl || 0).toFixed(2)} exceeds limit of $${config.max_daily_loss}`,
        threshold: config.max_daily_loss,
        current_value: Math.abs(metrics.daily_pnl || 0),
        timestamp: new Date().toISOString(),
        acknowledged: false
      });
    }

    // Check drawdown limit
    if ((metrics.max_drawdown || 0) > config.max_drawdown) {
      newAlerts.push({
        id: `drawdown-${Date.now()}`,
        type: 'drawdown',
        severity: 'high',
        title: 'Max Drawdown Exceeded',
        message: `Current drawdown of ${(metrics.max_drawdown || 0).toFixed(1)}% exceeds limit of ${config.max_drawdown}%`,
        threshold: config.max_drawdown,
        current_value: metrics.max_drawdown || 0,
        timestamp: new Date().toISOString(),
        acknowledged: false
      });
    }

    // Check concentration limit
    if ((metrics.concentration_ratio || 0) > config.concentration_limit) {
      newAlerts.push({
        id: `concentration-${Date.now()}`,
        type: 'concentration',
        severity: 'medium',
        title: 'Concentration Limit Exceeded',
        message: `Portfolio concentration of ${(metrics.concentration_ratio || 0).toFixed(1)}% exceeds limit of ${config.concentration_limit}%`,
        threshold: config.concentration_limit,
        current_value: metrics.concentration_ratio || 0,
        timestamp: new Date().toISOString(),
        acknowledged: false
      });
    }

    // Add new alerts
    newAlerts.forEach(alert => {
      if (!this.activeAlerts.find(a => a.type === alert.type && a.severity === alert.severity)) {
        this.activeAlerts.push(alert);
        this.onRiskAlert?.(alert);
      }
    });

    this.updateAlertsDisplay();
  }

  /**
   * Update alerts display
   */
  private updateAlertsDisplay(): void {
    // Update alert counts
    const criticalCount = this.activeAlerts.filter(a => a.severity === 'critical').length;
    const highCount = this.activeAlerts.filter(a => a.severity === 'high').length;
    const mediumCount = this.activeAlerts.filter(a => a.severity === 'medium').length;

    this.updateElement('alert-count', this.activeAlerts.length.toString());
    this.updateElement('critical-alerts', criticalCount.toString());
    this.updateElement('high-alerts', highCount.toString());
    this.updateElement('medium-alerts', mediumCount.toString());

    // Update alerts list
    const alertsList = document.getElementById('alerts-list');
    if (!alertsList) return;

    if (this.activeAlerts.length === 0) {
      alertsList.innerHTML = '<div class="empty-state">No active alerts</div>';
      return;
    }

    alertsList.innerHTML = this.activeAlerts.map(alert => `
      <div class="alert-item severity-${alert.severity}" data-alert-id="${alert.id}">
        <div class="alert-icon">
          ${this.getAlertIcon(alert.type)}
        </div>
        <div class="alert-content">
          <h6>${alert.title}</h6>
          <p>${alert.message}</p>
          <div class="alert-meta">
            <span class="alert-time">${new Date(alert.timestamp).toLocaleString()}</span>
            <span class="alert-values">
              Current: ${alert.current_value.toFixed(2)} | Threshold: ${alert.threshold.toFixed(2)}
            </span>
          </div>
        </div>
        <div class="alert-actions">
          <button class="acknowledge-btn" onclick="this.acknowledgeAlert('${alert.id}')">✓</button>
          <button class="dismiss-btn" onclick="this.dismissAlert('${alert.id}')">✕</button>
        </div>
      </div>
    `).join('');
  }

  /**
   * Get alert icon
   */
  private getAlertIcon(type: string): string {
    const icons = {
      exposure: '📊',
      loss: '💸',
      drawdown: '📉',
      volatility: '📈',
      correlation: '🔗',
      concentration: '🎯'
    };
    return icons[type as keyof typeof icons] || '⚠️';
  }

  /**
   * Update exposure display
   */
  private updateExposureDisplay(): void {
    const tableBody = document.getElementById('exposure-table-body');
    if (!tableBody) return;

    if (this.portfolioExposures.length === 0) {
      tableBody.innerHTML = '<tr class="empty-state"><td colspan="8">No positions to analyze</td></tr>';
      return;
    }

    tableBody.innerHTML = this.portfolioExposures.map(exposure => `
      <tr>
        <td class="symbol-cell">${exposure.symbol}</td>
        <td class="size-cell">${exposure.position_size.toFixed(4)}</td>
        <td class="value-cell">$${exposure.market_value.toFixed(2)}</td>
        <td class="percentage-cell">${exposure.percentage_of_portfolio.toFixed(1)}%</td>
        <td class="pnl-cell ${exposure.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
          $${exposure.unrealized_pnl.toFixed(2)}
        </td>
        <td class="risk-cell">
          <span class="risk-score score-${this.getRiskScoreClass(exposure.risk_score)}">
            ${exposure.risk_score.toFixed(1)}
          </span>
        </td>
        <td class="correlation-cell">${exposure.correlation_score.toFixed(2)}</td>
        <td class="actions-cell">
          <button class="action-btn reduce-btn" onclick="this.reducePosition('${exposure.symbol}')">Reduce</button>
          <button class="action-btn close-btn" onclick="this.closePosition('${exposure.symbol}')">Close</button>
        </td>
      </tr>
    `).join('');
  }

  /**
   * Get risk score class
   */
  private getRiskScoreClass(score: number): string {
    if (score < 3) return 'low';
    if (score < 6) return 'medium';
    if (score < 8) return 'high';
    return 'critical';
  }

  /**
   * Tab management
   */
  private switchTab(tabName: string): void {
    // Update tab buttons
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabName);
    });

    // Update tab content
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
      content.classList.toggle('active', content.id === `${tabName}-tab`);
    });

    // Load tab-specific data
    if (tabName === 'exposure') {
      this.refreshExposureAnalysis();
    } else if (tabName === 'scenarios') {
      this.loadRiskScenarios();
    }
  }

  /**
   * Stress testing and scenario analysis
   */
  private async runStressTest(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/bot/risk/stress-test`, {
        method: 'POST'
      });

      if (response.ok) {
        const results = await response.json();
        this.displayStressTestResults(results);
        this.showSuccessMessage('Stress test completed successfully');
      } else {
        throw new Error('Stress test failed');
      }
    } catch (error) {
      this.showErrorMessage('Failed to run stress test');
    }
  }

  private displayStressTestResults(results: any): void {
    // Implementation for displaying stress test results
    console.log('Stress test results:', results);
  }

  private async loadRiskScenarios(): Promise<void> {
    // Load and display risk scenarios
    const defaultScenarios: RiskScenario[] = [
      {
        name: 'Market Crash',
        description: '30% market decline across all assets',
        market_shock: -30,
        expected_loss: -5000,
        probability: 0.05,
        var_95: -3500,
        var_99: -4800,
        expected_shortfall: -5500
      },
      {
        name: 'High Volatility',
        description: 'Volatility spike to 80%',
        market_shock: 0,
        expected_loss: -1500,
        probability: 0.15,
        var_95: -1200,
        var_99: -1400,
        expected_shortfall: -1600
      }
    ];

    this.riskScenarios = defaultScenarios;
    this.displayRiskScenarios();
  }

  private displayRiskScenarios(): void {
    const scenariosGrid = document.getElementById('scenarios-grid');
    if (!scenariosGrid) return;

    scenariosGrid.innerHTML = this.riskScenarios.map(scenario => `
      <div class="scenario-card">
        <div class="scenario-header">
          <h6>${scenario.name}</h6>
          <span class="scenario-probability">${(scenario.probability * 100).toFixed(1)}%</span>
        </div>
        <div class="scenario-content">
          <p>${scenario.description}</p>
          <div class="scenario-metrics">
            <div class="metric">
              <span class="label">Market Shock:</span>
              <span class="value">${scenario.market_shock.toFixed(1)}%</span>
            </div>
            <div class="metric">
              <span class="label">Expected Loss:</span>
              <span class="value">$${scenario.expected_loss.toFixed(0)}</span>
            </div>
            <div class="metric">
              <span class="label">VaR 95%:</span>
              <span class="value">$${scenario.var_95.toFixed(0)}</span>
            </div>
            <div class="metric">
              <span class="label">VaR 99%:</span>
              <span class="value">$${scenario.var_99.toFixed(0)}</span>
            </div>
          </div>
        </div>
        <div class="scenario-actions">
          <button class="run-scenario-btn" onclick="this.runScenario('${scenario.name}')">Run Test</button>
        </div>
      </div>
    `).join('');
  }

  /**
   * Alert management
   */
  private acknowledgeAlert(alertId: string): void {
    const alert = this.activeAlerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      this.updateAlertsDisplay();
    }
  }

  private dismissAlert(alertId: string): void {
    this.activeAlerts = this.activeAlerts.filter(a => a.id !== alertId);
    this.updateAlertsDisplay();
  }

  private acknowledgeAllAlerts(): void {
    this.activeAlerts.forEach(alert => alert.acknowledged = true);
    this.updateAlertsDisplay();
  }

  private resetAlerts(): void {
    this.activeAlerts = [];
    this.updateAlertsDisplay();
  }

  private filterAlerts(): void {
    const filterSelect = document.getElementById('alert-filter') as HTMLSelectElement;
    const filter = filterSelect?.value || 'all';
    
    // Implementation for filtering alerts
    this.updateAlertsDisplay();
  }

  /**
   * Utility methods
   */
  private async refreshExposureAnalysis(): Promise<void> {
    try {
      const timeframe = (document.getElementById('exposure-timeframe') as HTMLSelectElement)?.value || '1d';
      const response = await fetch(`${this.apiBaseUrl}/api/bot/risk/exposure?timeframe=${timeframe}`);
      
      if (response.ok) {
        const exposures = await response.json();
        this.updatePortfolioExposures(exposures);
      }
    } catch (error) {
      console.warn('Failed to refresh exposure analysis:', error);
    }
  }

  private startRealtimeUpdates(): void {
    this.updateInterval = window.setInterval(() => {
      this.refreshExposureAnalysis();
    }, 10000); // Update every 10 seconds
  }

  private resetConfiguration(): void {
    this.loadDefaultConfiguration();
  }

  private exportConfiguration(): void {
    if (!this.currentRiskConfig) return;

    const dataStr = JSON.stringify(this.currentRiskConfig, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `risk-config-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  }

  private showAddScenarioDialog(): void {
    // Implementation for adding custom scenarios
    this.showSuccessMessage('Add scenario dialog would open here');
  }

  private updateElement(id: string, value: string): void {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  }

  private showSuccessMessage(message: string): void {
    console.log('Success:', message);
  }

  private showErrorMessage(message: string): void {
    console.error('Error:', message);
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
  }
}