import './style.css';
import { DashboardUI } from './ui.ts';
import { DashboardWebSocket, type AllWebSocketMessages } from './websocket.ts';
import { TradingViewChart } from './tradingview.ts';
import type { 
  DashboardConfig,
  BotStatus,
  MarketData,
  TradeAction,
  VuManchuIndicators,
  Position,
  RiskMetrics
} from './types.ts';

/**
 * Performance monitoring utility
 */
class PerformanceMonitor {
  private metrics = new Map<string, number>();
  private readonly maxMetrics = 100;

  public startTiming(label: string): void {
    this.metrics.set(`${label}_start`, performance.now());
  }

  public endTiming(label: string): number {
    const start = this.metrics.get(`${label}_start`);
    if (start === undefined) return 0;
    
    const duration = performance.now() - start;
    this.metrics.set(label, duration);
    this.metrics.delete(`${label}_start`);
    
    // Clean up old metrics to prevent memory bloat
    if (this.metrics.size > this.maxMetrics) {
      const entries = Array.from(this.metrics.entries());
      entries.slice(0, entries.length - this.maxMetrics).forEach(([key]) => {
        this.metrics.delete(key);
      });
    }
    
    return duration;
  }

  public getMetric(label: string): number | undefined {
    return this.metrics.get(label);
  }

  public logMetrics(): void {
    console.group('Performance Metrics');
    this.metrics.forEach((value, key) => {
      if (!key.endsWith('_start')) {
        console.log(`${key}: ${value.toFixed(2)}ms`);
      }
    });
    console.groupEnd();
  }
}

/**
 * Page visibility handler for connection management
 */
class VisibilityHandler {
  private callbacks = new Set<(visible: boolean) => void>();
  private isVisible = !document.hidden;
  private isInitialized = false;
  private initializationTimeout: number | null = null;
  private debounceTimer: number | null = null;
  private lastVisibilityChange = 0;
  private readonly DEBOUNCE_DELAY = 500; // 500ms minimum between visibility changes
  private readonly INIT_DELAY = 2000; // 2s delay to prevent false triggers during page load
  private processVisibilityChange!: (newVisible: boolean, source: string) => void;

  constructor() {
    // Check browser compatibility for Page Visibility API
    if (typeof document.hidden === 'undefined') {
      console.warn('Page Visibility API not supported - using fallback visibility detection');
      this.isVisible = document.hasFocus();
    }
    
    // Delay initialization to avoid false triggers during page load
    this.initializationTimeout = window.setTimeout(() => {
      this.setupVisibilityListeners();
      this.isInitialized = true;
      console.log(`VisibilityHandler initialized - initial state: ${this.isVisible ? 'visible' : 'hidden'}`);
    }, this.INIT_DELAY); // Longer delay to let TradingView and page fully settle
  }

  private setupVisibilityListeners(): void {
    const debouncedVisibilityChange = (newVisible: boolean, source: string): void => {
      if (!this.isInitialized) return;
      
      const now = Date.now();
      
      // Clear existing debounce timer
      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = null;
      }
      
      // Only proceed if visibility state actually changed and enough time has passed
      if (newVisible === this.isVisible) {
        return; // No actual change, ignore
      }
      
      // Check if enough time has passed since last change
      if (now - this.lastVisibilityChange < this.DEBOUNCE_DELAY) {
        // Schedule the change for later
        this.debounceTimer = window.setTimeout(() => {
          this.processVisibilityChange(newVisible, source);
        }, this.DEBOUNCE_DELAY - (now - this.lastVisibilityChange));
        return;
      }
      
      // Process the change immediately
      this.processVisibilityChange(newVisible, source);
    };
    
    const processVisibilityChange = (newVisible: boolean, source: string): void => {
      this.lastVisibilityChange = Date.now();
      this.isVisible = newVisible;
      console.log(`Visibility changed: ${newVisible ? 'visible' : 'hidden'} (source: ${source}, document.hidden: ${document.hidden}, hasFocus: ${document.hasFocus()})`);
      this.callbacks.forEach(callback => callback(newVisible));
    };
    
    this.processVisibilityChange = processVisibilityChange;

    const handleVisibilityChange = (): void => {
      // Use the Page Visibility API as primary source (most reliable)
      const visible = !document.hidden;
      debouncedVisibilityChange(visible, 'visibilitychange');
    };

    const handleFocus = (): void => {
      // Only trust focus events if the page is currently marked as hidden
      // and the document is not actually hidden
      if (!this.isVisible && !document.hidden) {
        debouncedVisibilityChange(true, 'focus');
      }
    };

    const handleBlur = (): void => {
      // For blur events, add extra delay and checks to avoid false positives
      // especially during TradingView interactions
      setTimeout(() => {
        if (!this.isInitialized) return;
        
        const activeElement = document.activeElement;
        const isIframeActive = activeElement && activeElement.tagName === 'IFRAME';
        
        // Only consider page hidden if:
        // 1. Document is actually hidden OR
        // 2. Document doesn't have focus AND active element is not an iframe
        const shouldBeHidden = document.hidden || (!document.hasFocus() && !isIframeActive);
        
        if (shouldBeHidden && this.isVisible) {
          debouncedVisibilityChange(false, 'blur');
        }
      }, 100); // Longer delay for blur to avoid TradingView focus issues
    };

    // Primary visibility API (most reliable)
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Secondary focus/blur events (with enhanced safeguards)
    window.addEventListener('focus', handleFocus);
    window.addEventListener('blur', handleBlur);
    
    // Page lifecycle events (less frequent, more reliable)
    window.addEventListener('pageshow', () => {
      if (this.isInitialized && !this.isVisible) {
        debouncedVisibilityChange(true, 'pageshow');
      }
    });
    
    window.addEventListener('pagehide', () => {
      if (this.isInitialized && this.isVisible) {
        debouncedVisibilityChange(false, 'pagehide');
      }
    });
  }

  public onVisibilityChange(callback: (visible: boolean) => void): void {
    this.callbacks.add(callback);
    
    // If not initialized yet, don't trigger immediately
    // If initialized and the current state is different from expected, trigger once
    if (this.isInitialized && this.callbacks.size === 1) {
      // First callback - verify current state is correct
      const actualVisible = !document.hidden;
      if (actualVisible !== this.isVisible) {
        this.isVisible = actualVisible;
        callback(actualVisible);
      }
    }
  }

  public removeVisibilityCallback(callback: (visible: boolean) => void): void {
    this.callbacks.delete(callback);
  }

  public get visible(): boolean {
    return this.isVisible;
  }
  
  public destroy(): void {
    if (this.initializationTimeout) {
      clearTimeout(this.initializationTimeout);
      this.initializationTimeout = null;
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    this.callbacks.clear();
    this.isInitialized = false;
  }
}

/**
 * AI Trading Bot Dashboard Main Application
 */
class DashboardApp {
  public ui: DashboardUI;
  private websocket: DashboardWebSocket;
  public chart: TradingViewChart | null = null;
  private config: DashboardConfig;
  private isInitialized = false;
  private initializationPromise: Promise<void> | null = null;
  public performanceMonitor: PerformanceMonitor;
  private visibilityHandler: VisibilityHandler;

  constructor() {
    // Configuration
    this.config = {
      websocket_url: this.getWebSocketUrl(),
      api_base_url: this.getApiBaseUrl(),
      default_symbol: 'BTC-USD',
      refresh_interval: 1000,
      chart_config: {
        container_id: 'tradingview-chart',
        symbol: 'DOGE-USD',
        interval: '1',
        library_path: '/charting_library/',
        theme: 'dark',
        autosize: true
      }
    };

    // Initialize utilities
    this.performanceMonitor = new PerformanceMonitor();
    this.visibilityHandler = new VisibilityHandler();

    // Initialize components
    this.ui = new DashboardUI();
    this.websocket = new DashboardWebSocket(this.config.websocket_url);

    // Setup page visibility handling
    this.setupVisibilityHandling();
  }

  /**
   * Initialize the dashboard application
   */
  public async initialize(): Promise<void> {
    // Prevent multiple initialization attempts
    if (this.isInitialized) {
      console.warn('Dashboard already initialized');
      return;
    }

    if (this.initializationPromise) {
      console.log('Initialization already in progress, waiting...');
      return this.initializationPromise;
    }

    this.initializationPromise = this.performInitialization();
    
    try {
      await this.initializationPromise;
      this.isInitialized = true;
    } catch (error) {
      // Reset so retry is possible
      this.initializationPromise = null;
      throw error;
    }
  }

  /**
   * Perform the actual initialization steps
   */
  private async performInitialization(): Promise<void> {
    this.performanceMonitor.startTiming('total_initialization');
    
    try {
      console.log('🚀 Initializing AI Trading Bot Dashboard...');
      this.updateLoadingProgress('Initializing UI components...', 20);

      // Step 1: Initialize UI
      this.performanceMonitor.startTiming('ui_initialization');
      this.ui.initialize();
      this.setupUIEventHandlers();
      const uiTime = this.performanceMonitor.endTiming('ui_initialization');
      console.log(`✅ UI initialized in ${uiTime.toFixed(2)}ms`);
      this.updateLoadingProgress('UI components loaded', 40);

      // Step 2: Initialize TradingView chart (truly non-blocking with timeout)
      this.updateLoadingProgress('Loading TradingView chart...', 60);
      this.performanceMonitor.startTiming('chart_initialization');
      
      // Start chart initialization in background without blocking main initialization
      const chartInitPromise = this.initializeChartNonBlocking();
      
      // Don't wait for chart - continue with core dashboard functionality
      this.updateLoadingProgress('Connecting to trading bot...', 80);

      // Step 3: Set up WebSocket connection
      this.performanceMonitor.startTiming('websocket_setup');
      this.setupWebSocketHandlers();
      
      // Wait for initial connection or timeout (with reduced timeout)
      const connectionPromise = new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('WebSocket connection timeout'));
        }, 8000); // 8 second timeout (reduced from 10s)

        const cleanup = () => {
          clearTimeout(timeout);
        };

        const handleConnection = (status: string) => {
          if (status === 'connected') {
            cleanup();
            resolve();
          } else if (status === 'error') {
            cleanup();
            reject(new Error('WebSocket connection failed'));
          }
        };

        // Listen for connection status
        this.websocket.onConnectionStatusChange(handleConnection);
        this.websocket.connect();
      });

      try {
        await connectionPromise;
        const wsTime = this.performanceMonitor.endTiming('websocket_setup');
        console.log(`✅ WebSocket connected in ${wsTime.toFixed(2)}ms`);
      } catch (wsError) {
        console.warn('WebSocket connection failed, continuing in offline mode:', wsError);
        this.performanceMonitor.endTiming('websocket_setup');
        this.ui.log('warn', 'Connected in offline mode - limited functionality', 'Connection');
      }

      // Step 4: Finalize initialization
      this.updateLoadingProgress('Finalizing dashboard...', 95);
      
      await new Promise(resolve => setTimeout(resolve, 300)); // Brief pause for smooth UX
      
      this.updateLoadingProgress('Dashboard ready!', 100);
      await new Promise(resolve => setTimeout(resolve, 200));
      
      this.hideLoadingScreen();

      const totalTime = this.performanceMonitor.endTiming('total_initialization');
      console.log(`🎉 Dashboard initialized successfully in ${totalTime.toFixed(2)}ms`);
      
      // Log performance metrics in development
      if (window.location.hostname === 'localhost') {
        this.performanceMonitor.logMetrics();
      }

      this.ui.log('info', 'Dashboard fully initialized and ready', 'System');
      
      // Handle chart initialization result in background
      chartInitPromise.then(() => {
        const chartTime = this.performanceMonitor.endTiming('chart_initialization');
        console.log(`✅ Chart initialized in background in ${chartTime.toFixed(2)}ms`);
        this.ui.log('info', 'TradingView chart loaded successfully', 'Chart');
      }).catch((chartError) => {
        console.warn('Chart background initialization failed:', chartError);
        this.performanceMonitor.endTiming('chart_initialization');
        this.ui.log('warn', 'Chart failed to load - dashboard continues with limited functionality', 'Chart');
      });
      
    } catch (error) {
      this.performanceMonitor.endTiming('total_initialization');
      console.error('❌ Failed to initialize dashboard:', error);
      
      const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error';
      this.showInitializationError(errorMessage);
      throw error;
    }
  }

  /**
   * Initialize TradingView chart with enhanced error handling
   */
  private async initializeChart(): Promise<void> {
    try {
      const baseUrl = this.config.api_base_url.endsWith('/api') 
        ? this.config.api_base_url.slice(0, -4) 
        : this.config.api_base_url;
      
      // Check network connectivity before initialization
      if (!navigator.onLine) {
        throw new Error('No network connection - chart requires internet access');
      }
      
      this.chart = new TradingViewChart(this.config.chart_config, baseUrl);
      const success = await this.chart.initialize();
      
      if (!success) {
        throw new Error('TradingView chart initialization returned false');
      }
      
      this.ui.log('info', 'TradingView chart initialized successfully', 'Chart');
      this.hideChartError();
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown chart error';
      console.error('Chart initialization failed:', errorMessage);
      
      // Show user-friendly error message
      if (errorMessage.includes('timeout')) {
        this.ui.log('error', 'Chart loading timed out - check internet connection', 'Chart');
        this.showChartError('Chart loading timed out. Please check your internet connection and try again.');
      } else if (errorMessage.includes('network') || errorMessage.includes('internet')) {
        this.ui.log('error', 'Network issue preventing chart load', 'Chart');
        this.showChartError('Network connection required for chart. Please check your connection.');
      } else {
        this.ui.log('error', `Chart initialization failed: ${errorMessage}`, 'Chart');
        this.showChartError('Failed to load trading chart. Dashboard will continue with limited functionality.');
      }
      
      // Continue without chart - the rest of the dashboard should still work
      this.chart = null;
    }
  }

  /**
   * Initialize TradingView chart in non-blocking mode with aggressive timeout
   */
  private async initializeChartNonBlocking(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Set aggressive timeout for chart initialization (max 15 seconds)
      const chartTimeout = setTimeout(() => {
        console.warn('Chart initialization timeout - continuing without chart');
        this.chart = null;
        this.showChartError('Chart loading timed out. Dashboard continues with full functionality.');
        reject(new Error('Chart initialization timeout'));
      }, 12000); // 12 second timeout (reduced from 15s)
      
      // Start chart initialization
      this.initializeChart()
        .then(() => {
          clearTimeout(chartTimeout);
          resolve();
        })
        .catch((error) => {
          clearTimeout(chartTimeout);
          // Don't propagate chart errors as fatal - dashboard should continue
          console.warn('Chart initialization failed, dashboard continues without chart:', error);
          this.chart = null;
          this.showChartError('Chart failed to load. All other dashboard features remain available.');
          reject(error);
        });
    });
  }

  /**
   * Show chart error message to user
   */
  private showChartError(message: string): void {
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement;
    if (chartError) {
      chartError.style.display = 'block';
      chartError.setAttribute('data-chart-error', 'visible');
      
      const errorText = chartError.querySelector('p');
      if (errorText) {
        errorText.textContent = message;
      }
    }
  }

  /**
   * Hide chart error message
   */
  private hideChartError(): void {
    const chartError = document.querySelector('[data-chart-error]') as HTMLElement;
    if (chartError) {
      chartError.style.display = 'none';
      chartError.setAttribute('data-chart-error', 'hidden');
    }
  }

  /**
   * Set up UI event handlers
   */
  private setupUIEventHandlers(): void {
    // Handle symbol changes
    this.ui.onSymbolChanged((symbol: string) => {
      this.config.chart_config.symbol = symbol;
      this.chart?.changeSymbol(symbol);
      this.ui.log('info', `Changed symbol to ${symbol}`, 'UI');
    });

    // Handle interval changes
    this.ui.onIntervalChanged((interval: string) => {
      this.config.chart_config.interval = interval;
      this.chart?.changeInterval(interval);
      this.ui.log('info', `Changed interval to ${interval}`, 'UI');
    });

    // Handle chart fullscreen toggle
    this.ui.onChartFullscreenToggle(() => {
      this.chart?.toggleFullscreen();
    });

    // Handle chart retry
    this.ui.onChartRetryRequested(() => {
      this.retryChartInitialization();
    });

    // Handle error retry
    this.ui.onErrorRetryRequested(() => {
      this.handleErrorRetry();
    });
  }

  /**
   * Set up WebSocket event handlers
   */
  private setupWebSocketHandlers(): void {
    // Connection status changes
    this.websocket.onConnectionStatusChange((status) => {
      this.ui.updateConnectionStatus(status);
      
      if (status === 'connected') {
        this.ui.log('info', 'Connected to trading bot', 'WebSocket');
      } else if (status === 'disconnected') {
        this.ui.log('warn', 'Disconnected from trading bot', 'WebSocket');
      } else if (status === 'error') {
        this.ui.log('error', 'WebSocket connection error', 'WebSocket');
      }
    });

    // Subscribe to all message types with a generic handler
    this.websocket.on('*', (message) => {
      this.handleWebSocketMessage(message);
    });

    // Subscribe to specific message types
    this.websocket.on('trading_loop', (message) => {
      const msg = message as any;
      if (msg.data) {
        this.ui.log('info', `Trading signal: ${msg.data.action} at $${msg.data.price}`, 'Trading');
      }
    });

    this.websocket.on('ai_decision', (message) => {
      const msg = message as any;
      if (msg.data) {
        this.ui.log('info', `AI Decision: ${msg.data.action} - ${msg.data.reasoning}`, 'AI');
      }
    });

    this.websocket.on('system_status', (message) => {
      const msg = message as any;
      if (msg.data) {
        const level = msg.data.health ? 'info' : 'warn';
        this.ui.log(level, `System status: ${msg.data.status}`, 'System');
      }
    });

    this.websocket.on('error', (message) => {
      const msg = message as any;
      if (msg.data) {
        this.ui.log('error', msg.data.message, 'System');
      }
    });

    // Error handling
    this.websocket.onError((error: Event | Error) => {
      console.error('WebSocket error:', error);
      const errorMessage = error instanceof Error ? error.message : 'WebSocket connection error';
      this.ui.showError(errorMessage);
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleWebSocketMessage(message: AllWebSocketMessages): void {
    try {
      switch (message.type) {
        case 'bot_status':
          this.ui.updateBotStatus(message.data as BotStatus);
          break;

        case 'market_data':
          const marketData = message.data as MarketData;
          this.ui.updateMarketData(marketData);
          this.chart?.updateMarketData(marketData);
          break;

        case 'trade_action':
          const tradeAction = message.data as TradeAction;
          this.ui.updateLatestAction(tradeAction);
          // Add AI decision marker to chart
          this.chart?.addAIDecisionMarker(tradeAction);
          break;

        case 'indicators':
          const indicators = message.data as VuManchuIndicators;
          this.chart?.updateIndicators(indicators);
          this.ui.log('debug', 'VuManChu indicators updated', 'Indicators');
          break;

        case 'position':
          // Handle single position update - convert to array for UI
          this.ui.updatePositions([message.data as Position]);
          break;

        case 'risk_metrics':
          this.ui.updateRiskMetrics(message.data as RiskMetrics);
          break;

        case 'trading_loop':
          // Trading loop messages are handled by specific handler above
          console.debug('Trading loop message processed');
          break;

        case 'ai_decision':
          // AI decision messages are handled by specific handler above
          console.debug('AI decision message processed');
          break;

        case 'system_status':
          // System status messages are handled by specific handler above
          console.debug('System status message processed');
          break;

        case 'error':
          // Error messages are handled by specific handler above
          console.debug('Error message processed');
          break;

        case 'ping':
        case 'pong':
          // Ping/pong messages are handled internally by WebSocket client
          break;

        default:
          console.log('Unknown message type:', (message as any).type);
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      this.ui.log('error', `Failed to process ${(message as any).type} message`, 'WebSocket');
    }
  }

  /**
   * Get WebSocket URL from environment or default
   */
  private getWebSocketUrl(): string {
    // Check for environment variable first
    const envWsUrl = (import.meta.env.VITE_WS_URL as string) || (window as any).__WS_URL__;
    if (envWsUrl) {
      console.log('Using WebSocket URL from environment:', envWsUrl);
      
      // Handle absolute URLs (already include protocol and host)
      if (envWsUrl.startsWith('ws://') || envWsUrl.startsWith('wss://')) {
        return envWsUrl;
      }
      
      // Handle protocol-relative URLs
      if (envWsUrl.startsWith('//')) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}${envWsUrl}`;
      }
      
      // Handle relative paths from environment variables
      if (envWsUrl.startsWith('/')) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const fullUrl = `${protocol}//${host}${envWsUrl}`;
        console.log('Converted relative WebSocket URL to absolute:', fullUrl);
        return fullUrl;
      }
      
      // Fallback: treat as relative path
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const fullUrl = `${protocol}//${host}/${envWsUrl.replace(/^\/+/, '')}`;
      console.log('Constructed WebSocket URL from environment variable:', fullUrl);
      return fullUrl;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    
    // In development mode (Vite dev server on port 3000), use proxy to backend
    if (host.includes('localhost:3000') || host.includes('127.0.0.1:3000')) {
      // Vite proxy will forward /ws to the backend automatically
      console.log('Development mode detected, using Vite proxy for WebSocket');
      return `${protocol}//${host}/ws`;
    }
    
    // In production or other environments, construct URL based on current host
    console.log('Production mode detected, using current host for WebSocket');
    return `${protocol}//${host}/ws`;
  }

  /**
   * Get API base URL from environment or default
   */
  private getApiBaseUrl(): string {
    // Check for environment variable first
    const envApiUrl = (import.meta.env.VITE_API_BASE_URL as string) || (import.meta.env.VITE_API_URL as string) || (window as any).__API_URL__;
    if (envApiUrl) {
      console.log('Using API URL from environment:', envApiUrl);
      return envApiUrl.endsWith('/api') ? envApiUrl : `${envApiUrl}/api`;
    }

    const protocol = window.location.protocol;
    const host = window.location.host;
    
    // In development mode (Vite dev server on port 3000), use proxy to backend
    if (host.includes('localhost:3000') || host.includes('127.0.0.1:3000')) {
      // Vite proxy will forward /api to the backend automatically
      console.log('Development mode detected, using Vite proxy for API');
      return `${protocol}//${host}/api`;
    }
    
    // In production or other environments, construct URL based on current host
    console.log('Production mode detected, using current host for API');
    return `${protocol}//${host}/api`;
  }

  /**
   * Update loading progress with message and percentage
   */
  private updateLoadingProgress(message: string, percentage: number): void {
    const loadingEl = document.getElementById('loading');
    if (!loadingEl) return;

    const progressEl = loadingEl.querySelector('.loading-progress');
    const messageEl = loadingEl.querySelector('.loading-message');
    const barEl = loadingEl.querySelector('.progress-bar');

    if (messageEl) {
      messageEl.textContent = message;
    }

    if (barEl) {
      (barEl as HTMLElement).style.width = `${percentage}%`;
    }

    // Create progress elements if they don't exist
    if (!progressEl) {
      const progressHTML = `
        <div class="loading-progress">
          <div class="loading-message">${message}</div>
          <div class="progress-container">
            <div class="progress-bar" style="width: ${percentage}%"></div>
          </div>
          <div class="loading-percentage">${percentage}%</div>
        </div>
      `;
      loadingEl.innerHTML += progressHTML;
    }
  }

  /**
   * Show initialization error
   */
  private showInitializationError(errorMessage: string): void {
    const loadingEl = document.getElementById('loading');
    if (!loadingEl) return;

    loadingEl.innerHTML = `
      <div class="error-message">
        <div class="error-icon">⚠️</div>
        <h2>Failed to Load Dashboard</h2>
        <p class="error-details">${errorMessage}</p>
        <div class="error-actions">
          <button onclick="window.location.reload()" class="retry-button">
            🔄 Retry
          </button>
          <button onclick="this.showErrorDetails()" class="details-button">
            📋 Details
          </button>
        </div>
      </div>
    `;
  }

  /**
   * Hide the loading screen
   */
  private hideLoadingScreen(): void {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
      loadingEl.style.opacity = '0';
      loadingEl.style.transition = 'opacity 0.3s ease';
      setTimeout(() => {
        loadingEl.style.display = 'none';
      }, 300);
    }
  }

  /**
   * Setup page visibility handling for connection management
   */
  private setupVisibilityHandling(): void {
    this.visibilityHandler.onVisibilityChange((visible: boolean) => {
      if (visible) {
        // Page became visible - restore normal operations
        if (this.websocket && this.isInitialized) {
          this.websocket.connect();
          this.ui.log('info', 'Page became visible - checking connection', 'Visibility');
        }
        
        // Restore performance optimization
        this.optimizePerformance(true);
        
        if (this.ui) {
          this.ui.log('info', 'Page became visible - full functionality restored', 'Visibility');
        }
      } else {
        // Page hidden - reduce activity
        // Only apply performance optimization if we're actually initialized
        if (this.isInitialized) {
          this.optimizePerformance(false);
          
          if (this.ui) {
            this.ui.log('debug', 'Page hidden - reducing background activity', 'Visibility');
          }
        }
      }
    });
  }

  /**
   * Retry chart initialization with enhanced recovery
   */
  private async retryChartInitialization(): Promise<void> {
    try {
      this.ui.log('info', 'Retrying chart initialization...', 'Chart');
      
      // Show loading state
      const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement;
      if (chartLoading) {
        chartLoading.style.display = 'flex';
        chartLoading.setAttribute('data-chart-loading', 'true');
      }
      
      this.hideChartError();
      
      // Clean up existing chart
      if (this.chart) {
        this.chart.destroy();
        this.chart = null;
      }

      // Check network connectivity
      if (!navigator.onLine) {
        throw new Error('No network connection available');
      }

      // Wait a moment before retrying
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Retry initialization
      await this.initializeChart();
      
      // Hide loading state on success
      if (chartLoading) {
        chartLoading.style.display = 'none';
        chartLoading.setAttribute('data-chart-loading', 'false');
      }
      
      this.ui.log('info', 'Chart retry successful', 'Chart');
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown retry error';
      console.error('Chart retry failed:', errorMessage);
      this.ui.log('error', `Chart retry failed: ${errorMessage}`, 'Chart');
      
      // Hide loading state on failure
      const chartLoading = document.querySelector('[data-chart-loading]') as HTMLElement;
      if (chartLoading) {
        chartLoading.style.display = 'none';
        chartLoading.setAttribute('data-chart-loading', 'false');
      }
      
      // Show specific error message
      if (errorMessage.includes('network') || errorMessage.includes('connection')) {
        this.showChartError('No internet connection. Please check your network and try again.');
      } else {
        this.showChartError('Failed to load chart after retry. Please refresh the page.');
      }
    }
  }

  /**
   * Handle general error retry
   */
  private async handleErrorRetry(): Promise<void> {
    try {
      this.ui.log('info', 'Attempting error recovery...', 'System');
      
      // Clear any existing errors
      this.ui.clearError();
      
      // Try to reconnect WebSocket if disconnected
      if (this.websocket && !this.websocket.isConnected()) {
        this.ui.log('info', 'Reconnecting WebSocket...', 'WebSocket');
        this.websocket.connect();
      }

      // Retry chart if it failed
      if (!this.chart) {
        await this.retryChartInitialization();
      }

      this.ui.log('info', 'Error recovery completed', 'System');
      
    } catch (error) {
      console.error('Error recovery failed:', error);
      this.ui.log('error', 'Error recovery failed', 'System');
    }
  }

  /**
   * Graceful degradation when components fail
   */
  public handleGracefulDegradation(component: string, error: Error): void {
    console.warn(`Component ${component} failed, continuing with degraded functionality:`, error);
    
    const degradationMap: Record<string, string> = {
      'chart': 'Trading chart unavailable - market data and controls still functional',
      'websocket': 'Real-time updates unavailable - dashboard in read-only mode',
      'ui': 'Some UI features may be limited'
    };

    const message = degradationMap[component] || `${component} functionality limited`;
    this.ui.log('warn', message, 'System');
    
    // Store degradation state for recovery attempts
    (this as any)[`${component}Failed`] = true;
  }

  /**
   * Performance optimization based on page visibility
   */
  private optimizePerformance(visible: boolean): void {
    if (!visible) {
      // Reduce update frequency when page is hidden
      this.config.refresh_interval = Math.max(this.config.refresh_interval * 2, 5000);
      
      // Pause non-critical chart updates (if method exists)
      if (this.chart && 'pauseUpdates' in this.chart) {
        (this.chart as any).pauseUpdates();
      }
    } else {
      // Restore normal update frequency
      this.config.refresh_interval = 1000;
      
      // Resume chart updates (if method exists)
      if (this.chart && 'resumeUpdates' in this.chart) {
        (this.chart as any).resumeUpdates();
      }
    }
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    console.log('🧹 Cleaning up dashboard resources...');
    
    try {
      // Disconnect WebSocket
      if (this.websocket) {
        this.websocket.disconnect();
      }

      // Destroy chart
      if (this.chart) {
        this.chart.destroy();
        this.chart = null;
      }

      // Clear visibility handler
      this.visibilityHandler.destroy();

      // Final performance log
      if (window.location.hostname === 'localhost') {
        this.performanceMonitor.logMetrics();
      }

      // Reset state
      this.isInitialized = false;
      this.initializationPromise = null;

      this.ui.log('info', 'Dashboard destroyed cleanly', 'App');
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }

  /**
   * Get current application health status
   */
  public getHealthStatus(): Record<string, any> {
    return {
      initialized: this.isInitialized,
      websocket_connected: this.websocket?.isConnected() || false,
      chart_loaded: this.chart !== null,
      page_visible: this.visibilityHandler.visible,
      performance_metrics: Object.fromEntries(this.performanceMonitor['metrics'] || [])
    };
  }

  /**
   * Force reconnection of all services
   */
  public async forceReconnect(): Promise<void> {
    this.ui.log('info', 'Forcing full reconnection...', 'System');
    
    try {
      // Reconnect WebSocket
      if (this.websocket) {
        this.websocket.disconnect();
        await new Promise(resolve => setTimeout(resolve, 1000));
        this.websocket.connect();
      }

      // Retry chart initialization if it failed
      if (!this.chart) {
        await this.retryChartInitialization();
      }

      this.ui.log('info', 'Full reconnection completed', 'System');
    } catch (error) {
      console.error('Force reconnection failed:', error);
      this.ui.log('error', 'Force reconnection failed', 'System');
    }
  }
}

// Global application instance for debugging and external access
let globalDashboardApp: DashboardApp | null = null;

// Enhanced global error handling
window.addEventListener('error', (event) => {
  console.error('🚨 Global JavaScript error:', {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    error: event.error
  });
  
  // Try to log to dashboard if available
  if (globalDashboardApp?.ui) {
    globalDashboardApp.ui.log('error', `Global error: ${event.message}`, 'System');
  }
});

window.addEventListener('unhandledrejection', (event) => {
  console.error('🚨 Unhandled Promise rejection:', event.reason);
  
  // Try to log to dashboard if available
  if (globalDashboardApp?.ui) {
    const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
    globalDashboardApp.ui.log('error', `Unhandled Promise: ${reason}`, 'System');
  }
  
  // Prevent default behavior that logs to console
  event.preventDefault();
});

// Network connectivity monitoring
window.addEventListener('online', () => {
  console.log('📶 Network connection restored');
  if (globalDashboardApp?.ui) {
    globalDashboardApp.ui.log('info', 'Network connection restored', 'Network');
    // Attempt to reconnect services
    globalDashboardApp.forceReconnect().catch(console.error);
  }
});

window.addEventListener('offline', () => {
  console.log('📵 Network connection lost');
  if (globalDashboardApp?.ui) {
    globalDashboardApp.ui.log('warn', 'Network connection lost - dashboard in offline mode', 'Network');
  }
});

// Performance monitoring
if ('PerformanceObserver' in window) {
  try {
    const perfObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        if (entry.entryType === 'navigation') {
          console.log(`📊 Page load performance: ${(entry as PerformanceNavigationTiming).loadEventEnd}ms`);
        }
      });
    });
    
    perfObserver.observe({ entryTypes: ['navigation', 'measure'] });
  } catch (error) {
    console.warn('Performance monitoring not available:', error);
  }
}

/**
 * Service Worker cleanup utility
 */
class ServiceWorkerCleaner {
  private cleanupAttempts = 0;
  private maxCleanupAttempts = 3;

  async performCleanup(): Promise<void> {
    if (!('serviceWorker' in navigator)) {
      console.log('[SW Cleaner] Service Worker API not available');
      return;
    }

    try {
      this.cleanupAttempts++;
      console.log(`[SW Cleaner] Cleanup attempt ${this.cleanupAttempts}/${this.maxCleanupAttempts}`);

      const registrations = await navigator.serviceWorker.getRegistrations();
      
      if (registrations.length === 0) {
        console.log('[SW Cleaner] No service workers found');
        return;
      }

      console.log(`[SW Cleaner] Found ${registrations.length} service worker(s) to clean up`);
      
      const unregisterPromises = registrations.map(async (registration) => {
        try {
          console.log(`[SW Cleaner] Unregistering SW from scope: ${registration.scope}`);
          const success = await registration.unregister();
          if (success) {
            console.log(`[SW Cleaner] Successfully unregistered: ${registration.scope}`);
          } else {
            console.warn(`[SW Cleaner] Failed to unregister: ${registration.scope}`);
          }
          return success;
        } catch (error) {
          console.error(`[SW Cleaner] Error unregistering ${registration.scope}:`, error);
          return false;
        }
      });

      const results = await Promise.all(unregisterPromises);
      const successCount = results.filter(Boolean).length;
      
      console.log(`[SW Cleaner] Cleanup complete: ${successCount}/${registrations.length} unregistered`);

      // If we still have registrations and haven't exceeded max attempts, try again
      if (successCount < registrations.length && this.cleanupAttempts < this.maxCleanupAttempts) {
        console.log('[SW Cleaner] Some service workers persist, retrying in 1 second...');
        setTimeout(() => this.performCleanup(), 1000);
      }

    } catch (error) {
      console.error('[SW Cleaner] Service worker cleanup failed:', error);
    }
  }

  preventNewRegistrations(): void {
    if (!('serviceWorker' in navigator)) return;

    // Store the original register function for potential future use
    (window as any).__originalServiceWorkerRegister = navigator.serviceWorker.register;
    
    navigator.serviceWorker.register = function(...args) {
      console.warn('[SW Cleaner] Blocked attempt to register service worker:', args);
      return Promise.reject(new Error('Service worker registration blocked by AI Trading Dashboard'));
    };

    console.log('[SW Cleaner] Service worker registration blocking active');
  }

  monitorServiceWorkers(): void {
    if (!('serviceWorker' in navigator)) return;

    // Monitor for new registrations every 5 seconds
    const monitorInterval = setInterval(async () => {
      try {
        const registrations = await navigator.serviceWorker.getRegistrations();
        if (registrations.length > 0) {
          console.warn(`[SW Cleaner] Detected ${registrations.length} service worker(s) during monitoring`);
          await this.performCleanup();
        }
      } catch (error) {
        console.error('[SW Cleaner] Monitoring error:', error);
      }
    }, 5000);

    // Stop monitoring after 60 seconds (12 checks)
    setTimeout(() => {
      clearInterval(monitorInterval);
      console.log('[SW Cleaner] Service worker monitoring stopped');
    }, 60000);
  }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  console.log('🎯 DOM loaded, initializing dashboard...');
  
  // Service worker cleanup - run before any other initialization
  const swCleaner = new ServiceWorkerCleaner();
  await swCleaner.performCleanup();
  swCleaner.preventNewRegistrations();
  swCleaner.monitorServiceWorkers();
  
  // Track page load performance
  const pageLoadStart = performance.now();
  
  try {
    globalDashboardApp = new DashboardApp();
    
    // Show loading screen immediately
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
      loadingEl.style.display = 'flex';
      loadingEl.innerHTML = `
        <div class="loading-container">
          <div class="loading-spinner"></div>
          <div class="loading-text">Loading AI Trading Dashboard...</div>
          <div class="loading-progress">
            <div class="loading-message">Initializing...</div>
            <div class="progress-container">
              <div class="progress-bar" style="width: 0%"></div>
            </div>
            <div class="loading-percentage">0%</div>
          </div>
        </div>
      `;
    }
    
    // Initialize with timeout protection (reduced from 30s to 20s)
    const initTimeout = setTimeout(() => {
      console.error('⏰ Dashboard initialization timeout');
      if (loadingEl) {
        loadingEl.innerHTML = `
          <div class="error-message">
            <div class="error-icon">⏰</div>
            <h2>Initialization Timeout</h2>
            <p class="error-details">Dashboard core components took too long to load. This may be due to network issues or server problems.</p>
            <div class="error-actions">
              <button onclick="window.location.reload()" class="retry-button">
                🔄 Retry
              </button>
              <button onclick="window.dashboard?.app()?.forceReconnect()" class="reconnect-button">
                🔌 Reconnect
              </button>
            </div>
          </div>
        `;
      }
    }, 20000); // 20 second timeout (reduced from 30s)
    
    await globalDashboardApp.initialize();
    clearTimeout(initTimeout);
    
    const pageLoadTime = performance.now() - pageLoadStart;
    console.log(`🎉 Dashboard fully loaded in ${pageLoadTime.toFixed(2)}ms`);
    
  } catch (error) {
    console.error('❌ Failed to start dashboard:', error);
    
    // Show comprehensive error information
    const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error';
    const loadingEl = document.getElementById('loading');
    
    if (loadingEl) {
      loadingEl.innerHTML = `
        <div class="error-message">
          <div class="error-icon">💥</div>
          <h2>Dashboard Initialization Failed</h2>
          <p class="error-details">${errorMessage}</p>
          <div class="error-info">
            <p><strong>Time:</strong> ${new Date().toLocaleString()}</p>
            <p><strong>User Agent:</strong> ${navigator.userAgent}</p>
            <p><strong>URL:</strong> ${window.location.href}</p>
          </div>
          <div class="error-actions">
            <button onclick="window.location.reload()" class="retry-button">
              🔄 Retry
            </button>
            <button onclick="navigator.clipboard?.writeText(document.querySelector('.error-info').textContent || '')" class="copy-button">
              📋 Copy Info
            </button>
          </div>
        </div>
      `;
    }
  }
});

// Cleanup and lifecycle management
window.addEventListener('beforeunload', () => {
  if (globalDashboardApp) {
    console.log('🚪 Page unloading, cleaning up...');
    globalDashboardApp.destroy();
  }
});

// Detect page refresh vs close
window.addEventListener('pagehide', (event) => {
  if (event.persisted) {
    console.log('📄 Page cached (back/forward navigation)');
  } else {
    console.log('🔄 Page being discarded');
  }
});

// Handle critical resource errors
window.addEventListener('error', (event) => {
  // Check if it's a resource loading error
  if (event.target && event.target !== window) {
    const element = event.target as HTMLElement;
    console.error(`📎 Resource failed to load:`, {
      tagName: element.tagName,
      src: (element as any).src || (element as any).href,
      message: event.message
    });
    
    if (globalDashboardApp?.ui) {
      globalDashboardApp.ui.log('error', `Resource failed to load: ${element.tagName}`, 'Resources');
    }
  }
}, true);

// Export enhanced debugging interface
(window as any).dashboard = {
  app: () => globalDashboardApp,
  health: () => globalDashboardApp?.getHealthStatus(),
  reconnect: () => globalDashboardApp?.forceReconnect(),
  performance: () => globalDashboardApp?.performanceMonitor?.logMetrics(),
  
  // Debugging utilities
  classes: { DashboardApp, DashboardUI, DashboardWebSocket, TradingViewChart },
  utils: { PerformanceMonitor, VisibilityHandler }
};

// Development helpers
if (window.location.hostname === 'localhost') {
  console.log(`
🔧 Development Mode Active
Available debugging commands:
- dashboard.app() - Get app instance
- dashboard.health() - Get health status  
- dashboard.reconnect() - Force reconnection
- dashboard.performance() - Show performance metrics
- dashboard.testSchemaCompliance() - Test TradingView schema compliance

Dashboard will be available at: window.dashboard
  `);
  
  // Add schema compliance testing to debug interface
  (window as any).dashboard.testSchemaCompliance = () => {
    const chart = globalDashboardApp?.chart;
    if (chart && 'testSchemaCompliance' in chart) {
      return (chart as any).testSchemaCompliance();
    } else {
      console.warn('Chart not available or schema testing not supported');
      return { success: false, issues: ['Chart not initialized'], validations: {} };
    }
  };
}
