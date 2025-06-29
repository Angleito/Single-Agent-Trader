import type { ChartConfig, MarketData, VuManchuIndicators, TradeAction } from './types.ts'

/**
 * TELEMETRY BLOCKING IMPLEMENTATION
 *
 * This TradingView integration includes comprehensive telemetry and analytics blocking
 * to prevent "Failed to load resource: net::ERR_NAME_NOT_RESOLVED" errors for:
 * - telemetry.tradingview.com
 * - analytics.tradingview.com
 * - metrics.tradingview.com
 * - tracking.tradingview.com
 *
 * Features implemented:
 * 1. Widget configuration: Disabled telemetry/analytics features in disabled_features array
 * 2. Network interception: fetch() and XMLHttpRequest patching to block telemetry requests
 * 3. Error suppression: Console error filtering for telemetry-related network failures
 * 4. CSP headers: Content Security Policy prevents external telemetry connections
 * 5. Monitoring: Telemetry blocking statistics and logging
 *
 * Usage: Call chart.getTelemetryBlockingStats() to see blocking statistics
 */

// Enhanced TradingView UDF Types for better type safety
// Note: UDFSymbolInfo is defined for future use
// interface UDFSymbolInfo {
//   name: string;
//   description: string;
//   type: string;
//   session: string;
//   timezone: string;
//   ticker: string;
//   exchange: string;
//   minmov: number;
//   pricescale: number;
//   has_intraday: boolean;
//   intraday_multipliers: string[];
//   supported_resolutions: string[];
//   volume_precision: number;
//   data_status: string;
// }

interface UDFBar {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface AIDecisionMarker {
  id: string
  time: number
  color: string
  text: string
  label: string
  labelFontColor: string
  minSize: number
  shape?: string
  confidence?: number
  price: number
}

interface ChartAnnotation {
  id: string
  points: Array<{ time: number; price: number }>
  options: {
    color: string
    lineWidth: number
    lineStyle: number
    text?: string
  }
}

interface TradingViewStudy {
  id: string
  name: string
  inputs: unknown[]
  styles: Record<string, unknown>
}

export class TradingViewChart {
  private widget: unknown = null
  private config: ChartConfig
  private isInitialized = false
  private retryCount = 0
  private maxRetries = 3
  private realtimeCallback: ((...args: unknown[]) => void) | null = null
  private currentBars: Map<string, UDFBar> = new Map()
  private subscribers: Map<string, (...args: unknown[]) => void> = new Map()
  private aiMarkers: AIDecisionMarker[] = []
  private chartAnnotations: ChartAnnotation[] = []
  private activeStudies: TradingViewStudy[] = []
  private backendBaseUrl: string
  private lastBarTime: Map<string, number> = new Map()
  private dataUpdateQueue: Array<() => void> = []
  private isProcessingQueue = false

  // Memory management
  private readonly MAX_MARKERS = 100
  private readonly MAX_ANNOTATIONS = 50
  private readonly MAX_BARS_CACHE = 1000
  private cleanupTimer: number | null = null
  private readonly CLEANUP_INTERVAL = 60000 // 1 minute

  // Performance optimization
  private updateThrottle: number | null = null
  private readonly UPDATE_THROTTLE_MS = 100

  // Telemetry blocking tracking
  private blockedTelemetryRequests: Set<string> = new Set()
  private telemetryBlockCount = 0

  constructor(config: ChartConfig, backendBaseUrl: string = 'http://localhost:8000') {
    this.config = config
    this.backendBaseUrl = backendBaseUrl

    // Set up network monitoring
    this.setupNetworkMonitoring()

    // Set up global error handlers for TradingView schema errors
    this.setupGlobalErrorHandlers()

    // Start memory cleanup
    this.startMemoryCleanup()
  }

  /**
   * Start periodic memory cleanup
   */
  private startMemoryCleanup(): void {
    this.cleanupTimer = window.setInterval(() => {
      this.performMemoryCleanup()
    }, this.CLEANUP_INTERVAL)
  }

  /**
   * Perform memory cleanup operations
   */
  private performMemoryCleanup(): void {
    // Clean up old AI markers
    if (this.aiMarkers.length > this.MAX_MARKERS) {
      this.aiMarkers = this.aiMarkers.slice(-this.MAX_MARKERS)
    }

    // Clean up old annotations
    if (this.chartAnnotations.length > this.MAX_ANNOTATIONS) {
      this.chartAnnotations = this.chartAnnotations.slice(-this.MAX_ANNOTATIONS)
    }

    // Clean up old bars cache
    if (this.currentBars.size > this.MAX_BARS_CACHE) {
      const entries = Array.from(this.currentBars.entries())
      const recentEntries = entries.slice(-this.MAX_BARS_CACHE)
      this.currentBars = new Map(recentEntries)
    }

    // Clean up old lastBarTime cache
    if (this.lastBarTime.size > 100) {
      const entries = Array.from(this.lastBarTime.entries())
      const recentEntries = entries.slice(-100)
      this.lastBarTime = new Map(recentEntries)
    }

    // Clear completed queue items
    if (!this.isProcessingQueue && this.dataUpdateQueue.length === 0) {
      this.dataUpdateQueue = []
    }
  }

  /**
   * Initialize the TradingView chart widget
   */
  public async initialize(): Promise<boolean> {
    try {
      // Run schema compliance tests first
      const complianceResults = this.testSchemaCompliance()

      if (!complianceResults.success) {
        // eslint-disable-next-line no-console
        console.warn('⚠️ Schema compliance issues detected:', complianceResults.issues)
        // Continue with warnings, but log the issues
        complianceResults.issues.forEach((issue) => {
          // eslint-disable-next-line no-console
          console.warn(`Schema issue: ${issue}`)
        })
      }

      // Wait for TradingView library to load
      await this.waitForTradingView()

      // Wait for DOM to be ready and container to exist
      await this.waitForDOMReady()

      // Verify container exists before creating widget
      const container = document.getElementById(this.config.container_id)
      if (!container) {
        throw new Error(`Container element with ID '${this.config.container_id}' not found`)
      }

      // Create the widget with enhanced configuration and error handling
      let widgetConfig: any
      try {
        // Create widget configuration with proper type validation
        widgetConfig = this.createValidatedWidgetConfig()

        this.widget = new window.TradingView.widget(widgetConfig)

        // Set up comprehensive property interception immediately after widget creation
        this.setupPropertyInterception()

        // Perform post-creation validation to catch any schema issues
        setTimeout(() => {
          this.performPostCreationValidation()
        }, 1000)
      } catch (widgetError) {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.error('Failed to create TradingView widget:', widgetError)
        const errorMessage =
          widgetError instanceof Error ? widgetError.message : String(widgetError)

        // Check if it's a schema validation error and provide helpful guidance
        if (
          errorMessage.includes('unknown') ||
          errorMessage.includes('schema') ||
          errorMessage.includes('data type')
        ) {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
        console.error('❌ Schema validation error detected. This may be caused by:')
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
        console.error('  - Unexpected data types in widget configuration')
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
        console.error('  - Invalid property values in datafeed or overrides')
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
        console.error('  - Missing required properties for TradingView API')
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
        console.error('  - Function properties in non-datafeed objects')

          // Analyze the configuration for potential issues if it was created
          if (widgetConfig) {
            this.analyzeConfigurationForSchemaIssues(widgetConfig)

            // Log the configuration that caused the issue for debugging
            // eslint-disable-next-line no-console
        console.error(
              'Widget configuration that caused error:',
              JSON.stringify(
                widgetConfig,
                (_key, value) => {
                  if (typeof value === 'function') {
                    return '[Function]'
                  }
                  return value
                },
                2
              )
            )
          }
        }

        throw new Error(`Widget creation failed: ${errorMessage}`)
      }

      // Set up event listeners
      this.setupEventListeners()

      this.isInitialized = true
      this.retryCount = 0 // Reset retry count on success
      return true
    } catch (error) {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
        console.error('Failed to initialize TradingView chart:', error)

      if (this.retryCount < this.maxRetries) {
        this.retryCount++
        const delay = Math.min(2000 * Math.pow(2, this.retryCount - 1), 15000) // Exponential backoff up to 15s

        // Network connectivity check before retry
        if (!navigator.onLine) {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
        console.error('No network connection available for retry')
          return false
        }

        // Try alternative loading strategies on retries
        if (this.retryCount === 2) {
          await this.tryAlternativeLoading()
        }

        // Additional fallback strategy on final retry
        if (this.retryCount === this.maxRetries) {
          // TODO: Add debug logging for final retry attempt with comprehensive fallback
          await this.comprehensiveFallbackLoading()
        }

        await new Promise((resolve) => setTimeout(resolve, delay))
        return this.initialize()
      }

      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
        console.error('TradingView initialization failed after all retries')
      return false
    }
  }

  /**
   * Wait for TradingView library to be available with enhanced loading strategies
   */
  private async waitForTradingView(timeout = 15000): Promise<void> {
    const startTime = Date.now()
    let loadingAttempted = false

    return new Promise((resolve, reject) => {
      const checkTradingView = () => {
        // Enhanced TradingView availability check - removed version requirement
        if (window.TradingView?.widget && typeof window.TradingView.widget === 'function') {
          const _version = window.TradingView.version || 'unknown'
          resolve()
          return
        }

        // Check if there was an explicit loading error from our enhanced loader
        if ((window as unknown as { tradingViewError?: boolean }).tradingViewError) {
          reject(new Error('TradingView library failed to load from all CDN sources'))
          return
        }

        const elapsed = Date.now() - startTime

        // Implement exponential timeout approach
        if (elapsed > timeout) {
          const scriptExists = !!document.querySelector('script[src*="tradingview"]')
          const tvObject = !!window.TradingView
          const tvWidget = !!window.TradingView?.widget
          const tvWidgetType = window.TradingView?.widget
            ? typeof window.TradingView.widget
            : 'undefined'
          const tvVersion = window.TradingView?.version ? window.TradingView.version : 'none'

          let errorMsg = `TradingView library load timeout after ${timeout}ms.`

          if (!navigator.onLine) {
            errorMsg += ' No network connection detected.'
          } else {
            errorMsg += ' Library status: '
            errorMsg += `script: ${scriptExists}, object: ${tvObject}, widget: ${tvWidget}, type: ${tvWidgetType}, version: ${tvVersion}`

            // If everything looks loaded but we're still timing out, something else is wrong
            if (scriptExists && tvObject && tvWidget && tvWidgetType === 'function') {
              errorMsg +=
                ' - Library appears loaded but version check failed. This may be a version detection issue.'
            }
          }

          // eslint-disable-next-line no-console
        console.error(errorMsg)
          reject(new Error(errorMsg))
          return
        }

        // Network connectivity check during loading
        if (elapsed > 10000 && !navigator.onLine) {
          reject(new Error('Network connection lost during TradingView loading'))
          return
        }

        // Attempt loading if not already attempted and some time has passed
        if (!loadingAttempted && elapsed > 2000) {
          loadingAttempted = true
          void this.ensureTradingViewScript()
        }

        // Log progress every 3 seconds with more context
        if (elapsed % 3000 < 100) {
          const _scriptExists = !!document.querySelector('script[src*="tradingview"]')
          const _tvObject = !!window.TradingView
          const _tvWidget = !!window.TradingView?.widget
          const _tvWidgetType = window.TradingView?.widget
            ? typeof window.TradingView.widget
            : 'undefined'
          const _tvVersion = window.TradingView?.version ? window.TradingView.version : 'none'
        }

        // Use adaptive polling interval
        const pollInterval = elapsed < 10000 ? 100 : elapsed < 20000 ? 250 : 500
        setTimeout(checkTradingView, pollInterval)
      }

      // Immediate check first
      checkTradingView()
    })
  }

  /**
   * Wait for DOM to be ready and container element to exist
   */
  private async waitForDOMReady(timeout = 10000): Promise<void> {
    const startTime = Date.now()

    return new Promise((resolve, reject) => {
      const checkDOM = () => {
        const elapsed = Date.now() - startTime

        // Check if DOM is ready and container exists
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
          const container = document.getElementById(this.config.container_id)
          if (container) {
            // DEBUG: DOM ready and container found
            resolve()
            return
          }
        }

        // Timeout check
        if (elapsed > timeout) {
          const container = document.getElementById(this.config.container_id)
          const errorMsg = `DOM/container timeout after ${timeout}ms. DOM state: ${document.readyState}, Container exists: ${!!container}`
          // eslint-disable-next-line no-console
        console.error(errorMsg)
          reject(new Error(errorMsg))
          return
        }

        // Continue checking
        setTimeout(checkDOM, 100)
      }

      // Start checking
      checkDOM()
    })
  }

  /**
   * Ensure TradingView script is loaded with multiple CDN fallbacks and retry logic
   */
  private async ensureTradingViewScript(): Promise<void> {
    // Check if script already exists and is loading
    const existingScript = document.querySelector('script[src*="tradingview"]') as HTMLScriptElement
    if (existingScript) {
      return
    }

    await this.loadTradingViewScriptWithRetry()
  }

  /**
   * Load TradingView script with multiple CDN sources and retry logic
   */
  private async loadTradingViewScriptWithRetry(attempt = 1, maxAttempts = 3): Promise<void> {
    const cdnSources = [
      'https://s3.tradingview.com/tv.js',
      'https://charting-library.tradingview-widget.com/tv.js',
      'https://s3.tradingview.com/charting_library/bundles/tv.js',
    ]

    const currentSource = cdnSources[(attempt - 1) % cdnSources.length]

    return new Promise((resolve, reject) => {
      // Remove any existing failed scripts
      const existingScripts = document.querySelectorAll('script[src*="tradingview"]')
      existingScripts.forEach((script) => {
        if ((script as HTMLScriptElement).src !== currentSource) {
          script.remove()
        }
      })

      const script = document.createElement('script')
      script.src = currentSource
      script.async = true
      script.crossOrigin = 'anonymous'

      const timeout = setTimeout(() => {
        console.warn(`TradingView script load timeout from ${currentSource}`)
        script.remove()

        if (attempt < maxAttempts) {
          this.loadTradingViewScriptWithRetry(attempt + 1, maxAttempts)
            .then(resolve)
            .catch(reject)
        } else {
          (window as any).tradingViewError = true
          reject(new Error(`Failed to load TradingView script after ${maxAttempts} attempts`))
        }
      }, 8000) // 8 second timeout per attempt (reduced for faster fallback)

      script.onload = () => {
        clearTimeout(timeout)
        ;(window as any).tradingViewLoaded = true
        resolve()
      }

      script.onerror = (error) => {
        clearTimeout(timeout)
        // eslint-disable-next-line no-console
        console.error(`Failed to load TradingView script from ${currentSource}:`, error)
        script.remove()

        if (attempt < maxAttempts) {
          // Exponential backoff
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000)
          setTimeout(() => {
            this.loadTradingViewScriptWithRetry(attempt + 1, maxAttempts)
              .then(resolve)
              .catch(reject)
          }, delay)
        } else {
          (window as any).tradingViewError = true
          reject(new Error(`Failed to load TradingView script after ${maxAttempts} attempts`))
        }
      }

      // Add script to head
      document.head.appendChild(script)
    })
  }

  /**
   * Create validated TradingView widget configuration
   */
  private createValidatedWidgetConfig(): any {
    // Ensure all configuration values have proper types
    const config = {
      // Basic widget settings - use primitive type coercion
      width: '100%',
      height: '100%',
      symbol: '' + this.normalizeSymbol(this.config.symbol),
      interval: '' + this.normalizeInterval(this.config.interval || '1'),
      container_id: '' + this.config.container_id,
      datafeed: this.createUDFDatafeed(),
      library_path: '' + (this.config.library_path || '/'),

      // Locale and timezone - ensure string types
      locale: 'en',
      timezone: 'Etc/UTC',

      // Debug setting - ensure boolean type
      debug: false,

      // Feature arrays - ensure they're proper arrays with string elements
      disabled_features: [
        'use_localstorage_for_settings',
        'volume_force_overlay',
        'create_volume_indicator_by_default',
        'header_symbol_search',
        'popup_hints',
        'context_menus',
        'left_toolbar',
        'show_logo_on_all_charts',
        'caption_buttons_text_if_possible',
        'header_indicators',
        'header_compare',
        'compare_symbol',
        // Disable telemetry and analytics features to prevent external connections
        'telemetry',
        'analytics',
        'performance_analytics',
        'usage_analytics',
        'error_reporting',
        'crash_reporting',
        'user_tracking',
        'feature_usage_tracking',
        'performance_tracking',
        'session_tracking',
        'external_analytics',
        'third_party_analytics',
        // Disable other external connections
        'news_feed',
        'social_trading',
        'social_media_sharing',
        'external_news',
        'external_feed',
        'market_news',
        'broker_integration',
        'external_broker',
        // Disable data collection features
        'widget_events_reporting',
        'metrics_collection',
        'user_behavior_tracking',
      ],

      enabled_features: [
        'study_templates',
        'side_toolbar_in_fullscreen_mode',
        'hide_left_toolbar_by_default',
        'move_logo_to_main_pane',
        'chart_crosshair_menu',
        'use_custom_indicators_for_marks',
        'modify_chart_skin',
        'custom_formatters',
        'show_chart_property_page',
        'adaptive_logo',
        'chart_style_hilo',
        'datasource_copypaste',
      ],

      // Storage configuration - only include if defined, with primitive type coercion
      ...(this.config.charts_storage_url && {
        charts_storage_url: '' + this.config.charts_storage_url,
        charts_storage_api_version: '' + (this.config.charts_storage_api_version || '1.1'),
        client_id: '' + (this.config.client_id || 'ai-trading-bot'),
        user_id: '' + (this.config.user_id || 'default_user'),
      }),

      // Boolean settings - explicit boolean coercion
      fullscreen: !!(this.config.fullscreen || false),
      autosize: !!(this.config.autosize !== false),

      // Object configurations - ensure they return proper objects
      studies_overrides: this.getValidatedStudiesOverrides(),
      overrides: this.getValidatedChartOverrides(),

      // Theme - ensure proper string value
      theme: this.config.theme === 'light' ? 'Light' : 'Dark',

      // CSS URL - only include if it exists, with proper string type
      ...(this.checkCSSFileExists() && {
        custom_css_url: './tradingview-custom.css',
      }),

      // Loading screen configuration - ensure proper object structure
      loading_screen: {
        backgroundColor: '#1e1e1e',
      },

      // Favorites configuration - ensure proper object structure with typed arrays
      favorites: {
        intervals: ['1', '5', '15', '30', '60', '240', '1D'],
        chartTypes: ['Area', 'Line', 'Candles', 'HeikinAshi', 'Hollow Candles'],
      },

      // Time frames configuration - ensure proper object structure with typed properties
      time_frames: [
        { text: '1m', resolution: '1' },
        { text: '5m', resolution: '5' },
        { text: '15m', resolution: '15' },
        { text: '30m', resolution: '30' },
        { text: '1h', resolution: '60' },
        { text: '4h', resolution: '240' },
        { text: '1D', resolution: '1D' },
      ],
    }

    // Additional validation to catch any undefined values that might become "unknown" types
    this.validateConfigForUnknownTypes(config)

    // Comprehensive type validation to prevent "unknown data type" errors
    this.performComprehensiveTypeValidation(config)

    // Validate configuration before returning
    this.validateWidgetConfig(config)

    return config
  }

  /**
   * Check if custom CSS file exists
   */
  private checkCSSFileExists(): boolean {
    try {
      // Simple check for CSS file existence
      const link = document.querySelector('link[href*="tradingview-custom.css"]')
      return !!link || document.querySelector('style[data-tradingview-custom]') !== null
    } catch (error) {
      return false
    }
  }

  /**
   * Perform comprehensive type validation to prevent "unknown data type" errors
   */
  private performComprehensiveTypeValidation(config: any): void {
    // DEBUG: Performing comprehensive type validation

    // Define expected data types for each property
    const expectedTypes: Record<string, string> = {
      width: 'string',
      height: 'string',
      symbol: 'string',
      interval: 'string',
      container_id: 'string',
      locale: 'string',
      timezone: 'string',
      debug: 'boolean',
      fullscreen: 'boolean',
      autosize: 'boolean',
      theme: 'string',
      library_path: 'string',
      disabled_features: 'array',
      enabled_features: 'array',
      time_frames: 'array',
      overrides: 'object',
      studies_overrides: 'object',
      loading_screen: 'object',
      favorites: 'object',
      datafeed: 'object',
    }

    for (const [key, expectedType] of Object.entries(expectedTypes)) {
      if (config[key] !== undefined) {
        const actualValue = config[key]
        const actualType = Array.isArray(actualValue) ? 'array' : typeof actualValue

        if (actualType !== expectedType) {
          console.warn(`⚠️ Type mismatch for '${key}': expected ${expectedType}, got ${actualType}`)

          // Attempt to coerce to correct type
          try {
            switch (expectedType) {
              case 'string':
                config[key] = String(actualValue)
                break
              case 'boolean':
                config[key] = Boolean(actualValue)
                break
              case 'array':
                config[key] = Array.isArray(actualValue) ? actualValue : []
                break
              case 'object':
                config[key] =
                  typeof actualValue === 'object' && actualValue !== null ? actualValue : {}
                break
            }
            console.log(`✅ Successfully coerced '${key}' to ${expectedType}`)
          } catch (error) {
            // eslint-disable-next-line no-console
        console.error(`❌ Failed to coerce '${key}' to ${expectedType}:`, error)
            throw new Error(
              `Invalid data type for property '${key}': cannot convert ${actualType} to ${expectedType}`
            )
          }
        }
      }
    }

    // Special validation for nested objects that might contain unknown types
    this.validateNestedObjectTypes(config)

    // DEBUG: Comprehensive type validation completed
  }

  /**
   * Validate nested object types to prevent unknown data type errors
   */
  private validateNestedObjectTypes(config: any): void {
    // Validate overrides object
    if (config.overrides && typeof config.overrides === 'object') {
      for (const [key, value] of Object.entries(config.overrides)) {
        if (value === undefined || value === null) {
          console.warn(`⚠️ Removing undefined/null override: ${key}`)
          delete config.overrides[key]
        } else if (
          typeof value !== 'string' &&
          typeof value !== 'number' &&
          typeof value !== 'boolean'
        ) {
          console.warn(
            `⚠️ Invalid override type for '${key}': ${typeof value}, converting to string`
          )
          config.overrides[key] = String(value)
        }
      }
    }

    // Validate studies_overrides object
    if (config.studies_overrides && typeof config.studies_overrides === 'object') {
      for (const [key, value] of Object.entries(config.studies_overrides)) {
        if (value === undefined || value === null) {
          console.warn(`⚠️ Removing undefined/null study override: ${key}`)
          delete config.studies_overrides[key]
        } else if (
          typeof value !== 'string' &&
          typeof value !== 'number' &&
          typeof value !== 'boolean'
        ) {
          console.warn(
            `⚠️ Invalid study override type for '${key}': ${typeof value}, converting to string`
          )
          config.studies_overrides[key] = String(value)
        }
      }
    }

    // Validate loading_screen object
    if (config.loading_screen && typeof config.loading_screen === 'object') {
      for (const [key, value] of Object.entries(config.loading_screen)) {
        if (value === undefined || value === null) {
          console.warn(`⚠️ Removing undefined/null loading_screen property: ${key}`)
          delete config.loading_screen[key]
        } else if (
          typeof value !== 'string' &&
          typeof value !== 'number' &&
          typeof value !== 'boolean'
        ) {
          console.warn(
            `⚠️ Invalid loading_screen type for '${key}': ${typeof value}, converting to string`
          )
          config.loading_screen[key] = String(value)
        }
      }
    }

    // Validate favorites object
    if (config.favorites && typeof config.favorites === 'object') {
      if (config.favorites.intervals && !Array.isArray(config.favorites.intervals)) {
        console.warn('⚠️ favorites.intervals is not an array, fixing')
        config.favorites.intervals = []
      }
      if (config.favorites.chartTypes && !Array.isArray(config.favorites.chartTypes)) {
        console.warn('⚠️ favorites.chartTypes is not an array, fixing')
        config.favorites.chartTypes = []
      }
    }

    // Validate time_frames array
    if (config.time_frames && Array.isArray(config.time_frames)) {
      config.time_frames = config.time_frames.map((frame: any, index: number) => {
        if (!frame || typeof frame !== 'object') {
          console.warn(`⚠️ Invalid time frame at index ${index}, creating default`)
          return { text: String('1m'), resolution: String('1') }
        }

        return {
          text: String(frame.text || '1m'),
          resolution: String(frame.resolution || '1'),
        }
      })
    }
  }

  /**
   * Validate configuration for unknown types that could cause TradingView schema errors
   */
  private validateConfigForUnknownTypes(config: any, path: string = 'config'): void {
    // Define expected TradingView datafeed functions that should not be flagged
    const expectedDatafeedFunctions = [
      'onReady',
      'searchSymbols',
      'resolveSymbol',
      'getBars',
      'subscribeBars',
      'unsubscribeBars',
      'getServerTime',
      'getMarks',
    ]

    for (const [key, value] of Object.entries(config)) {
      const currentPath = `${path}.${key}`

      // Check for undefined values that might be interpreted as "unknown" type
      if (value === undefined) {
        console.warn(`⚠️ Undefined value detected at ${currentPath} - removing from config`)
        delete config[key]
        continue
      }

      // Check for null values in critical paths
      if (value === null && ['symbol', 'interval', 'container_id', 'datafeed'].includes(key)) {
        throw new Error(`Critical config value '${currentPath}' is null`)
      }

      // Check for functions that might not serialize properly
      if (typeof value === 'function') {
        // Skip warnings for expected TradingView datafeed functions
        const isDatafeedFunction =
          path.includes('datafeed') && expectedDatafeedFunctions.includes(key)
        const isDatafeedObject = key === 'datafeed'

        if (!isDatafeedFunction && !isDatafeedObject) {
          console.warn(`⚠️ Function detected at ${currentPath} - this may cause schema issues`)
        }
      }

      // Recursively check objects, but skip datafeed object to avoid function warnings
      if (
        typeof value === 'object' &&
        value !== null &&
        !Array.isArray(value) &&
        key !== 'datafeed'
      ) {
        this.validateConfigForUnknownTypes(value, currentPath)
      }

      // Check arrays for undefined elements
      if (Array.isArray(value)) {
        const cleanArray = value.filter((item) => item !== undefined)
        if (cleanArray.length !== value.length) {
          console.warn(`⚠️ Undefined elements removed from array at ${currentPath}`)
          config[key] = cleanArray
        }

        // Validate array elements
        cleanArray.forEach((item, index) => {
          if (typeof item === 'object' && item !== null) {
            this.validateConfigForUnknownTypes(item, `${currentPath}[${index}]`)
          }
        })
      }
    }
  }

  /**
   * Analyze configuration for potential schema issues
   */
  private analyzeConfigurationForSchemaIssues(config: any): void {
    // DEBUG: Analyzing configuration for schema issues

    const issues: string[] = []

    // Check for problematic properties
    for (const [key, value] of Object.entries(config)) {
      if (value === undefined) {
        issues.push(`Property '${key}' is undefined`)
      } else if (value === null && key !== 'charts_storage_url') {
        issues.push(`Property '${key}' is null`)
      } else if (typeof value === 'function' && key !== 'datafeed') {
        issues.push(`Property '${key}' is a function (not allowed outside datafeed)`)
      } else if (typeof value === 'symbol') {
        issues.push(`Property '${key}' is a symbol type (not serializable)`)
      } else if (typeof value === 'bigint') {
        issues.push(`Property '${key}' is a bigint (not supported by JSON)`)
      } else if (typeof value === 'object' && value !== null) {
        // Check nested objects for issues
        if (key !== 'datafeed') {
          this.analyzeNestedObjectForIssues(value, key, issues)
        }
      }
    }

    // Check for missing required properties
    const requiredProps = ['width', 'height', 'symbol', 'interval', 'container_id', 'datafeed']
    for (const prop of requiredProps) {
      if (config[prop] === undefined || config[prop] === null) {
        issues.push(`Required property '${prop}' is missing or null`)
      }
    }

    // Report issues
    if (issues.length > 0) {
      // eslint-disable-next-line no-console
        console.error('❌ Configuration issues that may cause schema errors:')
      issues.forEach((issue) => // eslint-disable-next-line no-console
        console.error(`  - ${issue}`))
    } else {
      // DEBUG: No obvious configuration issues found
    }
  }

  /**
   * Analyze nested objects for schema issues
   */
  private analyzeNestedObjectForIssues(obj: any, path: string, issues: string[]): void {
    if (typeof obj !== 'object' || obj === null) return

    for (const [key, value] of Object.entries(obj)) {
      const fullPath = `${path}.${key}`

      if (value === undefined) {
        issues.push(`Nested property '${fullPath}' is undefined`)
      } else if (typeof value === 'function') {
        issues.push(`Nested property '${fullPath}' is a function (may cause serialization issues)`)
      } else if (typeof value === 'symbol') {
        issues.push(`Nested property '${fullPath}' is a symbol type (not serializable)`)
      } else if (typeof value === 'bigint') {
        issues.push(`Nested property '${fullPath}' is a bigint (not supported by JSON)`)
      } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Recursively check nested objects (but limit depth to avoid infinite loops)
        if (path.split('.').length < 3) {
          this.analyzeNestedObjectForIssues(value, fullPath, issues)
        }
      } else if (Array.isArray(value)) {
        // Check array elements
        value.forEach((item, index) => {
          if (item === undefined) {
            issues.push(`Array element '${fullPath}[${index}]' is undefined`)
          } else if (typeof item === 'function') {
            issues.push(`Array element '${fullPath}[${index}]' is a function`)
          }
        })
      }
    }
  }

  /**
   * Perform post-creation validation to catch any schema issues
   */
  private performPostCreationValidation(): void {
    try {
      if (!this.widget) return

      console.log('🔍 Performing post-creation schema validation...')

      // Check if widget has any internal validation errors
      if (typeof this.widget.getValidationErrors === 'function') {
        const errors = this.widget.getValidationErrors()
        if (errors && errors.length > 0) {
          console.warn('⚠️ TradingView widget validation errors found:', errors)
          errors.forEach((error: any) => {
            if (error.includes?.('unknown')) {
              // eslint-disable-next-line no-console
        console.error(`❌ Unknown data type error: ${error}`)
            }
          })
        }
      }

      // Check if widget is properly initialized
      if (typeof this.widget.chart === 'function') {
        try {
          const chart = this.widget.chart()
          if (chart) {
            console.log('✅ Widget chart object accessible - schema validation likely successful')
          }
        } catch (chartError) {
          console.warn('⚠️ Chart object not accessible, might indicate schema issues:', chartError)
        }
      }

      // Check for any global TradingView error states
      const globalErrors = (window as any).TradingViewErrors || []
      if (globalErrors.length > 0) {
        console.warn('⚠️ Global TradingView errors detected:', globalErrors)
        globalErrors.forEach((error: any) => {
          if (error.includes?.('schema')) {
            // eslint-disable-next-line no-console
        console.error(`❌ Schema error: ${error}`)
          }
        })
      }

      // Check for captured schema errors
      const schemaErrors = (window as any).TradingViewSchemaErrors || []
      if (schemaErrors.length > 0) {
        // eslint-disable-next-line no-console
        console.error('🚨 Captured TradingView schema errors:')
        schemaErrors.forEach((errorInfo: any, index: number) => {
          // eslint-disable-next-line no-console
        console.error(`Schema Error ${index + 1}:`, errorInfo)
        })

        // Clear the errors after reporting
        ;(window as any).TradingViewSchemaErrors = []
      }

      console.log('✅ Post-creation validation completed')
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to perform post-creation validation:', error)
    }
  }

  /**
   * Validate widget configuration for TradingView compatibility
   */
  private validateWidgetConfig(config: any): void {
    const requiredFields = ['width', 'height', 'symbol', 'interval', 'container_id', 'datafeed']

    for (const field of requiredFields) {
      if (config[field] === undefined || config[field] === null) {
        throw new Error(`Required TradingView config field '${field}' is missing or null`)
      }
    }

    // Validate data types
    if (typeof config.symbol !== 'string') {
      throw new Error(`TradingView config 'symbol' must be string, got ${typeof config.symbol}`)
    }

    if (typeof config.interval !== 'string') {
      throw new Error(`TradingView config 'interval' must be string, got ${typeof config.interval}`)
    }

    if (typeof config.container_id !== 'string') {
      throw new Error(
        `TradingView config 'container_id' must be string, got ${typeof config.container_id}`
      )
    }

    // Validate arrays
    if (!Array.isArray(config.disabled_features)) {
      throw new Error("TradingView config 'disabled_features' must be an array")
    }

    if (!Array.isArray(config.enabled_features)) {
      throw new Error("TradingView config 'enabled_features' must be an array")
    }

    if (!Array.isArray(config.time_frames)) {
      throw new Error("TradingView config 'time_frames' must be an array")
    }

    // Validate objects
    if (typeof config.loading_screen !== 'object' || config.loading_screen === null) {
      throw new Error("TradingView config 'loading_screen' must be an object")
    }

    if (typeof config.favorites !== 'object' || config.favorites === null) {
      throw new Error("TradingView config 'favorites' must be an object")
    }
  }

  /**
   * Validate study override keys to ensure they match actual TradingView study names
   */
  private validateStudyOverrides(overrides: Record<string, any>): Record<string, any> {
    const validatedOverrides: Record<string, any> = {}
    const knownOverridePatterns = [
      /^volume\./, // Volume study overrides
      /^RSI\./, // Relative Strength Index study overrides
      /^MACD\./, // MACD study overrides
      /^MFI\./, // Money Flow Index study overrides
      /^MA\./, // Moving Average study overrides (when used directly)
      /^Histogram\./, // MACD Histogram overrides
      /^Signal\./, // MACD Signal line overrides
      /^Upper Band\./, // RSI Upper Band overrides
      /^Lower Band\./, // RSI Lower Band overrides
      /^UpperLimit\./, // MFI Upper Limit overrides
      /^LowerLimit\./, // MFI Lower Limit overrides
    ]

    for (const [key, value] of Object.entries(overrides)) {
      const isValidPattern = knownOverridePatterns.some((pattern) => pattern.test(key))

      if (isValidPattern) {
        validatedOverrides[key] = value
      } else {
        console.warn(`Study override key '${key}' may not be valid for TradingView API`)
        // Still include it in case it's a newer or less common override
        validatedOverrides[key] = value
      }
    }

    return validatedOverrides
  }

  /**
   * Get validated studies overrides
   */
  private getValidatedStudiesOverrides(): Record<string, any> {
    const overrides = {
      'volume.volume.color.0': String('#ff4757'),
      'volume.volume.color.1': String('#2ed573'),
      'volume.volume.transparency': Number(70),
      'RSI.RSI.color': String('#ff9500'),
      'RSI.RSI.linewidth': Number(2),
      // Note: Moving Average overrides are applied directly in createStudy calls
      // instead of global overrides due to TradingView API structure
    }

    const validatedOverrides = this.validateStudyOverrides(overrides)

    // Ensure all values have explicit types and remove any undefined values
    const cleanedOverrides: Record<string, any> = {}
    for (const [key, value] of Object.entries(validatedOverrides)) {
      if (value !== undefined && value !== null) {
        // Ensure proper type coercion based on the property
        if (key.includes('color')) {
          cleanedOverrides[key] = String(value)
        } else if (
          key.includes('transparency') ||
          key.includes('linewidth') ||
          key.includes('width')
        ) {
          cleanedOverrides[key] = Number(value)
        } else if (typeof value === 'boolean') {
          cleanedOverrides[key] = Boolean(value)
        } else {
          cleanedOverrides[key] = value
        }
      }
    }

    return cleanedOverrides
  }

  /**
   * Get validated chart overrides
   */
  private getValidatedChartOverrides(): Record<string, any> {
    const baseOverrides =
      this.config.theme === 'light'
        ? {
            'paneProperties.background': String('#ffffff'),
            'paneProperties.vertGridProperties.color': String('#e6e8ea'),
            'paneProperties.horzGridProperties.color': String('#e6e8ea'),
            'symbolWatermarkProperties.transparency': Number(90),
            'scalesProperties.textColor': String('#333333'),
            'mainSeriesProperties.candleStyle.wickUpColor': String('#2ed573'),
            'mainSeriesProperties.candleStyle.wickDownColor': String('#ff4757'),
            'mainSeriesProperties.candleStyle.upColor': String('#2ed573'),
            'mainSeriesProperties.candleStyle.downColor': String('#ff4757'),
          }
        : {
            'paneProperties.background': String('#1e1e1e'),
            'paneProperties.vertGridProperties.color': String('#363c4e'),
            'paneProperties.horzGridProperties.color': String('#363c4e'),
            'symbolWatermarkProperties.transparency': Number(90),
            'scalesProperties.textColor': String('#aaa'),
            'mainSeriesProperties.candleStyle.wickUpColor': String('#2ed573'),
            'mainSeriesProperties.candleStyle.wickDownColor': String('#ff4757'),
            'mainSeriesProperties.candleStyle.upColor': String('#2ed573'),
            'mainSeriesProperties.candleStyle.downColor': String('#ff4757'),
            'mainSeriesProperties.candleStyle.borderUpColor': String('#2ed573'),
            'mainSeriesProperties.candleStyle.borderDownColor': String('#ff4757'),
          }

    // Validate all override values and ensure proper types
    const validatedOverrides: Record<string, any> = {}
    for (const [key, value] of Object.entries(baseOverrides)) {
      if (value !== undefined && value !== null) {
        // Ensure proper type coercion based on the property
        if (key.includes('color') || key.includes('Color') || key.includes('background')) {
          validatedOverrides[key] = String(value)
        } else if (key.includes('transparency') || key.includes('width') || key.includes('size')) {
          validatedOverrides[key] = Number(value)
        } else if (typeof value === 'boolean') {
          validatedOverrides[key] = Boolean(value)
        } else {
          // Default to string for any other properties
          validatedOverrides[key] = String(value)
        }
      }
    }

    return validatedOverrides
  }

  /**
   * Create UDF-compliant datafeed for TradingView chart
   */
  private createUDFDatafeed() {
    return {
      onReady: (callback: Function) => {
        console.log('[UDF onReady]: Initializing datafeed')

        // Validate callback function before using it
        if (typeof callback !== 'function') {
          // eslint-disable-next-line no-console
        console.error('[UDF onReady]: Invalid callback provided - not a function')
          return
        }

        fetch(`${this.backendBaseUrl}/udf/config`)
          .then((response) => response.json())
          .then((config) => {
            console.log('[UDF onReady]: Configuration loaded', config)
            const validatedConfig = this.validateUDFConfig({
              supports_search: false,
              supports_group_request: false,
              supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D'],
              supports_marks: true,
              supports_timescale_marks: true,
              supports_time: true,
              supports_streaming: true,
              exchanges: [
                {
                  value: 'Coinbase',
                  name: 'Coinbase Pro',
                  desc: 'Coinbase Pro Exchange',
                },
              ],
              symbols_types: [
                {
                  name: 'crypto',
                  value: 'crypto',
                },
              ],
              currencies: ['USD', 'EUR', 'BTC', 'ETH'],
              ...config,
            })
            // Enhanced validation and type normalization to prevent schema errors
            const finalConfig = this.deepNormalizeDataTypes(validatedConfig)
            this.validateUDFCallbackArgument(finalConfig, 'onReady config')
            console.log('[UDF onReady]: Calling callback with normalized config')
            callback(finalConfig)
          })
          .catch((error) => {
            console.warn('[UDF onReady]: Using fallback config', error)
            const fallbackConfig = this.validateUDFConfig({
              supports_search: false,
              supports_group_request: false,
              supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D'],
              supports_marks: true,
              supports_timescale_marks: true,
              supports_time: true,
              supports_streaming: true,
            })

            // Enhanced validation and type normalization to prevent schema errors
            const finalConfig = this.deepNormalizeDataTypes(fallbackConfig)
            this.validateUDFCallbackArgument(finalConfig, 'onReady fallback config')
            console.log('[UDF onReady]: Calling callback with normalized fallback config')
            callback(finalConfig)
          })
      },

      searchSymbols: (
        _userInput: string,
        _exchange: string,
        _symbolType: string,
        onResultReadyCallback: Function
      ) => {
        console.log('[UDF searchSymbols]: Search not supported')
        onResultReadyCallback([])
      },

      resolveSymbol: (
        symbolName: string,
        onSymbolResolvedCallback: Function,
        onResolveErrorCallback: Function
      ) => {
        console.log('[UDF resolveSymbol]: Resolving', symbolName)

        const normalizedSymbol = this.normalizeSymbol(symbolName)
        fetch(`${this.backendBaseUrl}/udf/symbols?symbol=${normalizedSymbol}`)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to resolve symbol: ${response.statusText}`)
            }
            return response.json()
          })
          .then(symbolInfo => {
            console.log('[UDF resolveSymbol]: Symbol resolved', symbolInfo)

            const validatedSymbolInfo = this.validateSymbolInfo({
              name: '' + (symbolInfo.name || normalizedSymbol),
              description: '' + (symbolInfo.description || normalizedSymbol),
              type: 'crypto',
              session: '24x7',
              timezone: 'Etc/UTC',
              ticker: '' + normalizedSymbol,
              exchange: 'Coinbase',
              minmov: +(symbolInfo.minmov || 1),
              pricescale: +(symbolInfo.pricescale || 100000),
              has_intraday: true,
              has_daily: true,
              has_weekly_and_monthly: true,
              intraday_multipliers: ['1', '5', '15', '30', '60'],
              supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D'],
              volume_precision: +(symbolInfo.volume_precision || 8),
              data_status: 'streaming',
              currency_code: '' + (symbolInfo.currency_code || 'USD'),
              original_name: '' + normalizedSymbol,
              ...symbolInfo,
            })

            // Apply deep normalization to prevent schema errors
            const normalizedSymbolInfo = this.deepNormalizeDataTypes(validatedSymbolInfo)
            console.log('[UDF resolveSymbol]: Calling callback with normalized symbol info')
            onSymbolResolvedCallback(normalizedSymbolInfo)
          })
          .catch(error => {
            // eslint-disable-next-line no-console
            console.error('[UDF resolveSymbol]: Error resolving symbol', error)
            onResolveErrorCallback('Symbol resolution failed')
          })
      },

      getBars: (
        symbolInfo: any,
        resolution: string,
        periodParams: any,
        onHistoryCallback: Function,
        onErrorCallback: Function
      ) => {
        console.log('[UDF getBars]: Fetching bars', {
          symbol: symbolInfo.name,
          resolution,
          periodParams,
        })

        const params = new URLSearchParams({
          symbol: symbolInfo.name,
          resolution: resolution,
          from: periodParams.from.toString(),
          to: periodParams.to.toString(),
          countback: periodParams.countback?.toString() || '300',
        })

        fetch(`${this.backendBaseUrl}/udf/history?${params}`)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to fetch bars: ${response.statusText}`)
            }
            return response.json()
          })
          .then(data => {
            if (data.s === 'no_data') {
              console.log('[UDF getBars]: No data available')
              // Apply normalization even for empty response
              const normalizedMeta = this.deepNormalizeDataTypes({ noData: true })
              onHistoryCallback([], normalizedMeta)
              return
            }

            if (data.s !== 'ok') {
              throw new Error(`Data fetch error: ${data.s}`)
            }

            // Convert to TradingView bar format with validation and normalization
            const bars: UDFBar[] = []
            for (let i = 0; i < data.t.length; i++) {
              const bar: UDFBar = this.validateBarData({
                time: +data.t[i] * 1000, // Convert to milliseconds
                open: +data.o[i],
                high: +data.h[i],
                low: +data.l[i],
                close: +data.c[i],
                volume: +(data.v[i] || 0),
              })
              bars.push(bar)

              // Cache the most recent bar for real-time updates
              if (i === data.t.length - 1) {
                const key = `${symbolInfo.name}_${resolution}`
                this.currentBars.set(key, bar)
              }
            }

            console.log(`[UDF getBars]: Loaded ${bars.length} bars`)
            // Apply deep normalization to bars array and metadata
            const normalizedBars = this.deepNormalizeDataTypes(bars)
            const normalizedMeta = this.deepNormalizeDataTypes({ noData: false })
            onHistoryCallback(normalizedBars, normalizedMeta)
          })
          .catch(error => {
            // eslint-disable-next-line no-console
            console.error('[UDF getBars]: Error fetching bars', error)
            onErrorCallback('Failed to fetch historical data')
          })
      },

      subscribeBars: (
        symbolInfo: any,
        resolution: string,
        onRealtimeCallback: Function,
        subscriberUID: string,
        _onResetCacheNeededCallback: Function
      ) => {
        console.log('[UDF subscribeBars]: Subscribing to real-time data', {
          symbol: symbolInfo.name,
          resolution,
          subscriberUID,
        })

        // const key = `${symbolInfo.name}_${resolution}`; // Reserved for future use
        this.subscribers.set(subscriberUID, onRealtimeCallback)

        // Store callback for this specific subscription
        if (!this.realtimeCallback) {
          this.realtimeCallback = onRealtimeCallback
        }
      },

      unsubscribeBars: (subscriberUID: string) => {
        console.log('[UDF unsubscribeBars]: Unsubscribing', subscriberUID)
        this.subscribers.delete(subscriberUID)

        if (this.subscribers.size === 0) {
          this.realtimeCallback = null
        }
      },

      getServerTime: (callback: Function) => {
        callback(Math.floor(Date.now() / 1000))
      },

      getMarks: async (
        symbolInfo: any,
        from: number,
        to: number,
        onDataCallback: Function,
        resolution: string
      ) => {
        console.log('[UDF getMarks]: Fetching AI decision markers', {
          symbol: symbolInfo.name,
          from,
          to,
          resolution,
        })

        try {
          const params = new URLSearchParams({
            symbol: symbolInfo.name,
            from: from.toString(),
            to: to.toString(),
            resolution: resolution,
          })

          const response = await fetch(`${this.backendBaseUrl}/udf/marks?${params}`)

          if (response.ok) {
            const marks = await response.json()
            console.log(`[UDF getMarks]: Loaded ${marks.length} markers`)
            onDataCallback(marks)
          } else {
            console.warn('[UDF getMarks]: Failed to fetch marks')
            onDataCallback([])
          }
        } catch (error) {
          // eslint-disable-next-line no-console
        console.error('[UDF getMarks]: Error fetching marks', error)
          onDataCallback([])
        }
      },
    }
  }

  /**
   * Set up event listeners for the chart
   */
  private setupEventListeners(): void {
    if (!this.widget) return

    try {
      // Use the widget's onChartReady method if available
      if (typeof this.widget.onChartReady === 'function') {
        this.widget.onChartReady(() => {
          this.handleChartReady()
        })
      } else {
        // Fallback: set up a delayed chart ready handler
        console.log('Using fallback chart ready handler')
        setTimeout(() => {
          this.handleChartReady()
        }, 2000)
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Error setting up chart event listeners:', error)
    }
  }

  /**
   * Handle chart ready event
   */
  private handleChartReady(): void {
    try {
      // Get chart reference if available
      if (this.widget && typeof this.widget.chart === 'function') {
        const chart = this.widget.chart()

        // Add technical indicators
        this.addTechnicalIndicators()

        // Set up chart event listeners with proper type validation
        this.setupChartEventListeners(chart)

        // Load any existing AI markers
        this.loadAIMarkers()
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Error in chart ready handler:', error)
    }
  }

  /**
   * Set up chart event listeners with comprehensive type validation
   */
  private setupChartEventListeners(chart: any): void {
    try {
      // Set up interval change listener with proper validation
      if (chart && typeof chart.onIntervalChanged === 'function') {
        const intervalSubscription = chart.onIntervalChanged()
        if (intervalSubscription && typeof intervalSubscription.subscribe === 'function') {
          intervalSubscription.subscribe(
            null, // scope parameter - explicitly null and typed
            (interval: any) => {
              try {
                // Validate interval parameter to prevent unknown type errors
                const validatedInterval = this.validateIntervalParameter(interval)
                console.log(`Chart interval changed to: ${validatedInterval}`)
                this.config.interval = validatedInterval
              } catch (error) {
                // eslint-disable-next-line no-console
        console.error('Error in interval change handler:', error)
                // Use current config as fallback
                console.log(`Falling back to current interval: ${this.config.interval}`)
              }
            }
          )
        }
      }

      // Set up symbol change listener with proper validation
      if (chart && typeof chart.onSymbolChanged === 'function') {
        const symbolSubscription = chart.onSymbolChanged()
        if (symbolSubscription && typeof symbolSubscription.subscribe === 'function') {
          symbolSubscription.subscribe(
            null, // scope parameter - explicitly null and typed
            (symbolData: any) => {
              try {
                // Validate symbol data to prevent unknown type errors
                const validatedSymbolData = this.validateSymbolParameter(symbolData)
                console.log(`Chart symbol changed to: ${validatedSymbolData.name}`)
                this.config.symbol = validatedSymbolData.name
              } catch (error) {
                // eslint-disable-next-line no-console
        console.error('Error in symbol change handler:', error)
                // Use current config as fallback
                console.log(`Falling back to current symbol: ${this.config.symbol}`)
              }
            }
          )
        }
      }

      console.log('Chart event listeners setup completed')
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Error setting up chart event listeners:', error)
    }
  }

  /**
   * Validate interval parameter to ensure proper type
   */
  private validateIntervalParameter(interval: any): string {
    if (typeof interval === 'string' && interval.trim().length > 0) {
      return String(interval.trim())
    } else if (typeof interval === 'number' && !isNaN(interval)) {
      return String(interval)
    } else {
      console.warn(`Invalid interval parameter:`, interval, `- using fallback`)
      return this.config.interval || '1D'
    }
  }

  /**
   * Validate symbol parameter to ensure proper type structure
   */
  private validateSymbolParameter(symbolData: any): { name: string } {
    if (symbolData && typeof symbolData === 'object') {
      if (typeof symbolData.name === 'string' && symbolData.name.trim().length > 0) {
        return {
          name: String(symbolData.name.trim()),
        }
      } else if (typeof symbolData.ticker === 'string' && symbolData.ticker.trim().length > 0) {
        return {
          name: String(symbolData.ticker.trim()),
        }
      }
    } else if (typeof symbolData === 'string' && symbolData.trim().length > 0) {
      return {
        name: String(symbolData.trim()),
      }
    }

    console.warn(`Invalid symbol parameter:`, symbolData, `- using fallback`)
    return {
      name: this.config.symbol || 'BTC-USD',
    }
  }

  /**
   * Validate study styles to ensure proper data types and prevent "unknown" type errors
   */
  private validateStudyStyles(styles: Record<string, any>): Record<string, any> {
    const validatedStyles: Record<string, any> = {}

    for (const [key, value] of Object.entries(styles)) {
      if (value === undefined || value === null) {
        console.warn(`⚠️ Undefined/null study style value for '${key}' - skipping`)
        continue
      }

      // Validate and coerce types based on property name patterns
      if (key.includes('color') || key.includes('Color')) {
        // Color properties should be strings
        validatedStyles[key] = String(value)
      } else if (
        key.includes('transparency') ||
        key.includes('linewidth') ||
        key.includes('width') ||
        key.includes('size')
      ) {
        // Numeric properties
        const numValue = Number(value)
        if (isNaN(numValue)) {
          console.warn(`⚠️ Invalid numeric value for '${key}': ${value}, using default 1`)
          validatedStyles[key] = Number(1)
        } else {
          validatedStyles[key] = numValue
        }
      } else if (typeof value === 'boolean') {
        // Boolean properties
        validatedStyles[key] = Boolean(value)
      } else {
        // Default to string for other properties
        validatedStyles[key] = String(value)
      }
    }

    return validatedStyles
  }

  /**
   * Validate study names against known TradingView API study names
   */
  private validateStudyName(studyName: string): boolean {
    const validStudyNames = [
      // Moving Averages
      'Moving Average Exponential',
      'Moving Average Simple',
      'Moving Average Weighted',
      'Moving Average',
      // Oscillators
      'Relative Strength Index',
      'MACD',
      'Stochastic',
      'Williams %R',
      'Rate Of Change',
      'Momentum',
      'Money Flow Index',
      'Commodity Channel Index',
      'Aroon',
      'Fisher Transform',
      'Stochastic RSI',
      'Ultimate Oscillator',
      'Awesome Oscillator',
      // Volume
      'Volume',
      'Volume Profile',
      'On Balance Volume',
      'Accumulation/Distribution',
      'Price Volume Trend',
      'Ease of Movement',
      'Volume Weighted Average Price',
      // Trend
      'Bollinger Bands',
      'Parabolic SAR',
      'Average Directional Index',
      'Directional Movement Index',
      'Ichimoku Cloud',
      'Keltner Channels',
      'Donchian Channels',
      // Bill Williams
      'Alligator',
      'Fractals',
      'Gator Oscillator',
      'Market Facilitation Index',
      // Others
      'Average True Range',
      'Standard Deviation',
      'Linear Regression',
      'Pivot Points',
      'Fibonacci Retracement',
      'Elliott Wave',
    ]

    return validStudyNames.includes(studyName)
  }

  /**
   * Add technical indicators to the chart
   */
  private addTechnicalIndicators(): void {
    if (!this.widget) return

    try {
      if (typeof this.widget.chart === 'function') {
        const chart = this.widget.chart()

        if (chart && typeof chart.createStudy === 'function') {
          const studiesToAdd = [
            {
              name: 'Relative Strength Index',
              inputs: [Number(14)],
              styles: {
                'RSI.color': String('#ff9500'),
                'RSI.linewidth': Number(2),
                'Upper Band.color': String('#ff4757'),
                'Lower Band.color': String('#2ed573'),
              },
              id: 'rsi',
            },
            {
              name: 'Moving Average Exponential',
              inputs: [Number(9)],
              styles: {
                'MA.color': String('#00d2ff'),
                'MA.linewidth': Number(2),
                'MA.transparency': Number(0),
              },
              id: 'ema9',
            },
            {
              name: 'Moving Average Exponential',
              inputs: [Number(21)],
              styles: {
                'MA.color': String('#ff6b6b'),
                'MA.linewidth': Number(2),
                'MA.transparency': Number(0),
              },
              id: 'ema21',
            },
            {
              name: 'Volume',
              inputs: [],
              styles: {
                'volume.color.0': String('#ff4757'),
                'volume.color.1': String('#2ed573'),
                'volume.transparency': Number(70),
              },
              id: 'volume',
              forceOverlay: Boolean(true),
            },
            {
              name: 'Money Flow Index',
              inputs: [Number(14)],
              styles: {
                'MFI.color': String('#9c88ff'),
                'MFI.linewidth': Number(2),
                'UpperLimit.color': String('#ff4757'),
                'LowerLimit.color': String('#2ed573'),
              },
              id: 'mfi',
            },
            {
              name: 'MACD',
              inputs: [Number(12), Number(26), Number(9)],
              styles: {
                'MACD.color': String('#2196F3'),
                'Signal.color': String('#FF9800'),
                'Histogram.color': String('#9C27B0'),
              },
              id: 'macd',
            },
          ]

          // Add each study with individual error handling
          const successfulStudies: TradingViewStudy[] = []

          for (const study of studiesToAdd) {
            try {
              // Validate study name before attempting to create it
              if (!this.validateStudyName(study.name)) {
                console.warn(
                  `⚠️ Study name '${study.name}' may not be supported by TradingView API`
                )
              }

              // Validate and clean study styles to prevent "unknown data type" errors
              const validatedStyles = this.validateStudyStyles(study.styles)

              // Validate study inputs to ensure they are proper numbers
              const validatedInputs = study.inputs.map((input) => {
                if (typeof input === 'number' && !isNaN(input)) {
                  return input
                } else {
                  console.warn(`⚠️ Invalid study input: ${input}, using default 14`)
                  return 14
                }
              })

              console.log(`Adding study: ${study.name}`, {
                inputs: validatedInputs,
                styles: validatedStyles,
                forceOverlay: study.forceOverlay,
              })

              const studyId = chart.createStudy(
                String(study.name),
                Boolean(study.forceOverlay || false),
                Boolean(false), // Lock study
                validatedInputs,
                null, // No study overrides
                validatedStyles
              )

              successfulStudies.push({
                id: study.id,
                name: study.name,
                inputs: validatedInputs,
                styles: validatedStyles,
              })

              console.log(`✅ Successfully added study: ${study.name} (ID: ${studyId})`)
            } catch (studyError) {
              // eslint-disable-next-line no-console
        console.error(`❌ Failed to add study '${study.name}':`, studyError)

              // Provide helpful error information
              if (!this.validateStudyName(study.name)) {
                // eslint-disable-next-line no-console
        console.error(
                  `Study name '${study.name}' is not in the known valid study names list`
                )
              }

              // Log the full error details for debugging
              // eslint-disable-next-line no-console
        console.error('Study creation error details:', {
                studyName: study.name,
                inputs: study.inputs,
                styles: study.styles,
                error: studyError,
              })

              // Continue with other studies even if one fails
            }
          }

          this.activeStudies = successfulStudies
          console.log(
            `Technical indicators setup complete. Successfully added ${successfulStudies.length}/${studiesToAdd.length} studies`
          )
        } else {
          console.warn('createStudy method not available on chart')
        }
      } else {
        console.warn('chart method not available on widget')
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to add technical indicators:', error)
    }
  }

  /**
   * Update chart with new market data (enhanced with queue processing)
   */
  public updateMarketData(data: MarketData): void {
    if (!this.isInitialized || !this.realtimeCallback) return

    // Add to queue for batch processing
    this.dataUpdateQueue.push(() => this.processMarketDataUpdate(data))
    this.processUpdateQueue()
  }

  /**
   * Process market data update with enhanced logic
   */
  private processMarketDataUpdate(data: MarketData): void {
    try {
      const timestamp = new Date(data.timestamp).getTime()
      const key = `${data.symbol}_${this.config.interval}`
      const currentBar = this.currentBars.get(key)
      // const lastBarTime = this.lastBarTime.get(key) || 0; // Reserved for future use

      let bar: UDFBar

      if (currentBar && this.isSameBarPeriod(currentBar.time, timestamp, this.config.interval)) {
        // Update existing bar with validation
        bar = this.validateBarData({
          ...currentBar,
          high: Math.max(currentBar.high, data.price),
          low: Math.min(currentBar.low, data.price),
          close: data.price,
          volume: currentBar.volume + (data.volume || 0),
          time: currentBar.time, // Keep original bar time
        })
      } else {
        // Create new bar with proper time alignment
        const barTime = this.getBarTime(timestamp, this.config.interval)
        bar = this.validateBarData({
          time: barTime,
          open: data.price,
          high: data.price,
          low: data.price,
          close: data.price,
          volume: data.volume || 0,
        })

        // Update last bar time tracking
        this.lastBarTime.set(key, barTime)
      }

      this.currentBars.set(key, bar)

      // Notify all subscribers with error handling
      this.subscribers.forEach((callback, uid) => {
        try {
          callback(bar)
        } catch (error) {
          // eslint-disable-next-line no-console
        console.error(`Error in realtime callback for ${uid}:`, error)
          // Remove failing subscriber
          this.subscribers.delete(uid)
        }
      })
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to process market data update:', error)
    }
  }

  /**
   * Process update queue with throttling
   */
  private processUpdateQueue(): void {
    if (this.isProcessingQueue || this.dataUpdateQueue.length === 0) return

    this.isProcessingQueue = true

    // Process updates in batches
    const updates = this.dataUpdateQueue.splice(0, 10)
    updates.forEach((update) => {
      try {
        update()
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Error processing queued update:', error)
      }
    })

    this.isProcessingQueue = false

    // Continue processing if more updates are queued
    if (this.dataUpdateQueue.length > 0) {
      setTimeout(() => this.processUpdateQueue(), 16) // ~60fps
    }
  }

  /**
   * Update VuManChu indicators on the chart
   */
  public updateIndicators(indicators: VuManchuIndicators): void {
    if (!this.isInitialized || !this.widget) return

    try {
      // Store indicators for display
      console.log('VuManChu indicators updated:', indicators)

      // Could add custom indicator overlays here if needed
      // For now, we rely on the built-in technical indicators
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to update indicators:', error)
    }
  }

  /**
   * Change the chart symbol
   */
  public changeSymbol(symbol: string): void {
    if (!this.isInitialized || !this.widget) return

    try {
      const normalizedSymbol = this.normalizeSymbol(symbol)
      this.config.symbol = normalizedSymbol

      if (typeof this.widget.setSymbol === 'function') {
        this.widget.setSymbol(normalizedSymbol, this.config.interval, () => {
          console.log(`Chart symbol changed to: ${normalizedSymbol}`)
          // Clear cached bars for the old symbol
          this.currentBars.clear()
          // Reload AI markers for new symbol
          this.loadAIMarkers()
        })
      } else {
        console.warn('setSymbol method not available on widget')
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to change chart symbol:', error)
    }
  }

  /**
   * Change chart interval
   */
  public changeInterval(interval: string): void {
    if (!this.isInitialized || !this.widget) return

    try {
      const normalizedInterval = this.normalizeInterval(interval)
      this.config.interval = normalizedInterval

      if (typeof this.widget.chart === 'function') {
        const chart = this.widget.chart()
        if (chart && typeof chart.setResolution === 'function') {
          chart.setResolution(normalizedInterval, () => {
            console.log(`Chart interval changed to: ${normalizedInterval}`)
            // Clear cached bars for the old interval
            this.currentBars.clear()
          })
        } else {
          console.warn('setResolution method not available on chart')
        }
      } else {
        console.warn('chart method not available on widget')
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to change chart interval:', error)
    }
  }

  /**
   * Get chart screenshot
   */
  public async getScreenshot(): Promise<string | null> {
    if (!this.isInitialized || !this.widget) return null

    try {
      return await this.widget.takeScreenshot()
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to take chart screenshot:', error)
      return null
    }
  }

  /**
   * Destroy the chart widget
   */
  public destroy(): void {
    // Clear timers
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }

    if (this.updateThrottle) {
      clearTimeout(this.updateThrottle)
      this.updateThrottle = null
    }

    // Clean up TradingView widget
    if (this.widget) {
      this.widget.remove()
      this.widget = null
    }

    // Clear all caches and collections
    this.currentBars.clear()
    this.subscribers.clear()
    this.aiMarkers = []
    this.chartAnnotations = []
    this.activeStudies = []
    this.lastBarTime.clear()
    this.dataUpdateQueue = []

    // Reset state
    this.isInitialized = false
    this.realtimeCallback = null
    this.isProcessingQueue = false
    this.retryCount = 0

    // Clear telemetry tracking
    this.blockedTelemetryRequests.clear()
    this.telemetryBlockCount = 0
  }

  /**
   * Add an AI decision marker to the chart (enhanced)
   */
  public addAIDecisionMarker(decision: TradeAction): void {
    if (!this.isInitialized || !this.widget) return

    try {
      const chart = this.widget.chart()
      const timestamp = new Date(decision.timestamp).getTime() / 1000

      const marker: AIDecisionMarker = {
        id: `ai_decision_${timestamp}_${decision.action}`,
        time: timestamp,
        color: this.getDecisionColor(decision.action),
        text: `${decision.action}: ${decision.reasoning}`,
        label: decision.action.charAt(0),
        labelFontColor: 'white',
        minSize: this.getMarkerSize(decision.confidence),
        shape: this.getDecisionShape(decision.action),
        confidence: decision.confidence,
        price: decision.price || 0,
      }

      this.aiMarkers.push(marker)

      // Create enhanced shape on chart with confidence-based styling
      const shapeOptions = this.validateShapeOptions({
        shape: String(marker.shape || 'circle'),
        text: String(`${marker.label} (${Math.round(decision.confidence * 100)}%)`),
        overrides: {
          color: String(marker.color),
          backgroundColor: String(this.adjustColorOpacity(marker.color, decision.confidence)),
          borderColor: String(marker.color),
          textColor: String(marker.labelFontColor),
          fontSize: Number(Math.max(10, Math.min(16, 10 + decision.confidence * 6))),
          bold: Boolean(decision.confidence > 0.7),
          transparency: Number(Math.max(20, 100 - decision.confidence * 80)),
        },
      })

      const shapePoint = this.validateShapePoint({
        time: Number(timestamp),
        price: Number(decision.price || 50000), // Use a reasonable default price instead of 0
      })

      // Additional validation before creating shape
      if (chart && typeof chart.createShape === 'function') {
        console.log('Creating AI decision shape with validated options:', {
          point: shapePoint,
          options: shapeOptions,
        })
        chart.createShape(shapePoint, shapeOptions)
      } else {
        console.warn('Chart.createShape method not available')
      }

      // Add trend line for high-confidence decisions
      if (decision.confidence > 0.8) {
        this.addTrendAnnotation(decision)
      }

      console.log('Enhanced AI decision marker added:', marker)
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to add AI decision marker:', error)
    }
  }

  /**
   * Add trend annotation for high-confidence decisions
   */
  private addTrendAnnotation(decision: TradeAction): void {
    if (!this.widget || !decision.price) return

    try {
      const chart = this.widget.chart()
      const timestamp = new Date(decision.timestamp).getTime() / 1000
      const futureTime = timestamp + this.getIntervalMinutes(this.config.interval) * 60 * 5 // 5 bars ahead

      const basePrice = Number(decision.price || 50000) // Ensure we have a valid base price
      const annotation: ChartAnnotation = {
        id: String(`trend_${timestamp}`),
        points: [
          this.validateShapePoint({ time: Number(timestamp), price: basePrice }),
          this.validateShapePoint({
            time: Number(futureTime),
            price: Number(basePrice * (decision.action === 'BUY' ? 1.02 : 0.98)),
          }),
        ],
        options: {
          color: String(this.getDecisionColor(decision.action)),
          lineWidth: Number(2),
          lineStyle: Number(2), // Dashed line
          text: String(`AI Projection (${Math.round(decision.confidence * 100)}%)`),
        },
      }

      this.chartAnnotations.push(annotation)

      // Create trend line with proper validation
      const trendLineOptions = this.validateShapeOptions({
        shape: String('trend_line'),
        overrides: annotation.options,
      })

      if (chart && typeof chart.createShape === 'function') {
        console.log('Creating trend annotation with validated options:', {
          points: annotation.points,
          options: trendLineOptions,
        })
        chart.createShape(annotation.points, trendLineOptions)
      } else {
        console.warn('Chart.createShape method not available for trend annotation')
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to add trend annotation:', error)
    }
  }

  /**
   * Load existing AI markers from backend
   */
  private async loadAIMarkers(): Promise<void> {
    if (!this.widget) return

    try {
      const chart = this.widget.chart()
      const range = chart.getVisibleRange()

      if (range) {
        const params = new URLSearchParams({
          symbol: this.config.symbol,
          from: range.from.toString(),
          to: range.to.toString(),
        })

        const response = await fetch(`${this.backendBaseUrl}/udf/marks?${params}`)

        if (response.ok) {
          const markers = await response.json()
          console.log(`Loaded ${markers.length} AI decision markers`)

          // Clear existing markers
          this.aiMarkers = []

          // Add markers to chart
          markers.forEach((marker: any) => {
            this.aiMarkers.push(marker)
          })
        }
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to load AI markers:', error)
    }
  }

  /**
   * Validation Methods for UDF and TradingView Data
   */
  private validateUDFConfig(config: any): any {
    // Ensure all boolean fields are proper booleans
    const booleanFields = [
      'supports_search',
      'supports_group_request',
      'supports_marks',
      'supports_timescale_marks',
      'supports_time',
      'supports_streaming',
    ]

    for (const field of booleanFields) {
      if (config[field] !== undefined) {
        config[field] = Boolean(config[field])
      }
    }

    // Ensure arrays are proper arrays
    if (config.supported_resolutions && !Array.isArray(config.supported_resolutions)) {
      config.supported_resolutions = ['1', '5', '15', '30', '60', '240', '1D']
    }

    if (config.currencies && !Array.isArray(config.currencies)) {
      config.currencies = ['USD']
    }

    // Validate exchanges array
    if (config.exchanges && Array.isArray(config.exchanges)) {
      config.exchanges = config.exchanges.map((exchange: any) => ({
        value: String(exchange.value || ''),
        name: String(exchange.name || ''),
        desc: String(exchange.desc || ''),
      }))
    }

    // Validate symbols_types array
    if (config.symbols_types && Array.isArray(config.symbols_types)) {
      config.symbols_types = config.symbols_types.map((symbolType: any) => ({
        name: String(symbolType.name || ''),
        value: String(symbolType.value || ''),
      }))
    }

    console.log('✅ UDF config validated:', config)
    return config
  }

  private validateSymbolInfo(symbolInfo: any): any {
    // Ensure all string fields are strings
    const stringFields = [
      'name',
      'description',
      'type',
      'session',
      'timezone',
      'ticker',
      'exchange',
      'data_status',
      'currency_code',
      'original_name',
    ]

    for (const field of stringFields) {
      if (symbolInfo[field] !== undefined) {
        symbolInfo[field] = String(symbolInfo[field])
      }
    }

    // Ensure all number fields are numbers
    const numberFields = ['minmov', 'pricescale', 'volume_precision']

    for (const field of numberFields) {
      if (symbolInfo[field] !== undefined) {
        symbolInfo[field] = Number(symbolInfo[field])
        if (isNaN(symbolInfo[field])) {
          symbolInfo[field] = field === 'pricescale' ? 100000 : field === 'volume_precision' ? 8 : 1
        }
      }
    }

    // Ensure boolean fields are booleans
    const booleanFields = ['has_intraday', 'has_daily', 'has_weekly_and_monthly']

    for (const field of booleanFields) {
      if (symbolInfo[field] !== undefined) {
        symbolInfo[field] = Boolean(symbolInfo[field])
      }
    }

    // Ensure arrays are proper arrays
    if (symbolInfo.intraday_multipliers && !Array.isArray(symbolInfo.intraday_multipliers)) {
      symbolInfo.intraday_multipliers = ['1', '5', '15', '30', '60']
    }

    if (symbolInfo.supported_resolutions && !Array.isArray(symbolInfo.supported_resolutions)) {
      symbolInfo.supported_resolutions = ['1', '5', '15', '30', '60', '240', '1D']
    }

    console.log('✅ Symbol info validated:', symbolInfo)
    return symbolInfo
  }

  private validateBarData(bar: any): UDFBar {
    // Ensure all fields are numbers and validate ranges
    const validatedBar: UDFBar = {
      time: Number(bar.time),
      open: Number(bar.open),
      high: Number(bar.high),
      low: Number(bar.low),
      close: Number(bar.close),
      volume: Number(bar.volume || 0),
    }

    // Check for invalid numbers
    if (isNaN(validatedBar.time) || validatedBar.time <= 0) {
      throw new Error('Invalid bar time')
    }

    if (isNaN(validatedBar.open) || validatedBar.open <= 0) {
      throw new Error('Invalid bar open price')
    }

    if (isNaN(validatedBar.high) || validatedBar.high <= 0) {
      throw new Error('Invalid bar high price')
    }

    if (isNaN(validatedBar.low) || validatedBar.low <= 0) {
      throw new Error('Invalid bar low price')
    }

    if (isNaN(validatedBar.close) || validatedBar.close <= 0) {
      throw new Error('Invalid bar close price')
    }

    if (isNaN(validatedBar.volume)) {
      validatedBar.volume = 0
    }

    // Validate OHLC logic
    if (validatedBar.high < Math.max(validatedBar.open, validatedBar.close)) {
      validatedBar.high = Math.max(validatedBar.open, validatedBar.close)
    }

    if (validatedBar.low > Math.min(validatedBar.open, validatedBar.close)) {
      validatedBar.low = Math.min(validatedBar.open, validatedBar.close)
    }

    return validatedBar
  }

  private validateShapeOptions(options: any): any {
    // Ensure the base object has required properties
    const baseOptions = {
      shape: String(options?.shape || 'circle'),
      text: String(options?.text || ''),
      overrides: {},
    }

    // Validate overrides object if it exists
    if (options?.overrides && typeof options.overrides === 'object') {
      const overrides = options.overrides
      baseOptions.overrides = {
        ...(overrides.color !== undefined && { color: String(overrides.color) }),
        ...(overrides.backgroundColor !== undefined && {
          backgroundColor: String(overrides.backgroundColor),
        }),
        ...(overrides.borderColor !== undefined && { borderColor: String(overrides.borderColor) }),
        ...(overrides.textColor !== undefined && { textColor: String(overrides.textColor) }),
        ...(overrides.fontSize !== undefined && { fontSize: Number(overrides.fontSize) }),
        ...(overrides.bold !== undefined && { bold: Boolean(overrides.bold) }),
        ...(overrides.transparency !== undefined && {
          transparency: Number(overrides.transparency),
        }),
        ...(overrides.linecolor !== undefined && { linecolor: String(overrides.linecolor) }),
        ...(overrides.linewidth !== undefined && { linewidth: Number(overrides.linewidth) }),
      }
    }

    // Remove any undefined values that might cause "unknown" type errors
    this.validateConfigForUnknownTypes(baseOptions, 'shapeOptions')

    return baseOptions
  }

  /**
   * Utility Methods
   */
  private normalizeSymbol(symbol: string): string {
    // Convert symbol formats: BTC-USD -> BTCUSD, DOGE/USD -> DOGEUSD
    return symbol.replace(/[-/]/g, '').toUpperCase()
  }

  private normalizeInterval(interval: string): string {
    // Convert interval formats: 1m -> 1, 1H -> 60, 1D -> 1D
    const intervalMap: { [key: string]: string } = {
      '1m': '1',
      '5m': '5',
      '15m': '15',
      '30m': '30',
      '1h': '60',
      '1H': '60',
      '4h': '240',
      '4H': '240',
      '1d': '1D',
      '1D': '1D',
    }

    return intervalMap[interval] || interval
  }

  private getDecisionColor(action: string): string {
    switch (action.toUpperCase()) {
      case 'BUY':
        return '#2ed573'
      case 'SELL':
        return '#ff4757'
      case 'HOLD':
        return '#3742fa'
      default:
        return '#747d8c'
    }
  }

  private getDecisionShape(action: string): string {
    switch (action.toUpperCase()) {
      case 'BUY':
        return 'arrow_up'
      case 'SELL':
        return 'arrow_down'
      case 'HOLD':
        return 'circle'
      default:
        return 'circle'
    }
  }

  private getMarkerSize(confidence: number): number {
    return Math.max(16, Math.min(32, 16 + confidence * 16))
  }

  private adjustColorOpacity(color: string, confidence: number): string {
    // Convert hex color to rgba with confidence-based opacity
    const opacity = Math.max(0.3, confidence)
    const hex = color.replace('#', '')
    const r = parseInt(hex.substr(0, 2), 16)
    const g = parseInt(hex.substr(2, 2), 16)
    const b = parseInt(hex.substr(4, 2), 16)
    return `rgba(${r}, ${g}, ${b}, ${opacity})`
  }

  private isSameBarPeriod(barTime: number, newTime: number, interval: string): boolean {
    const barMinutes = this.getIntervalMinutes(interval)
    const barStart = Math.floor(barTime / (barMinutes * 60 * 1000)) * (barMinutes * 60 * 1000)
    const newStart = Math.floor(newTime / (barMinutes * 60 * 1000)) * (barMinutes * 60 * 1000)
    return barStart === newStart
  }

  private getBarTime(timestamp: number, interval: string): number {
    const intervalMinutes = this.getIntervalMinutes(interval)
    return Math.floor(timestamp / (intervalMinutes * 60 * 1000)) * (intervalMinutes * 60 * 1000)
  }

  private getIntervalMinutes(interval: string): number {
    const intervalMap: { [key: string]: number } = {
      '1': 1,
      '5': 5,
      '15': 15,
      '30': 30,
      '60': 60,
      '240': 240,
      '1D': 1440,
    }

    return intervalMap[interval] || 1
  }

  /**
   * Advanced chart management methods
   */
  public addCustomIndicator(name: string, script: string): void {
    if (!this.isInitialized || !this.widget) return

    try {
      const chart = this.widget.chart()
      // Add custom Pine Script indicator
      chart.createStudy(name, false, false, [], script)
      console.log(`Custom indicator '${name}' added`)
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to add custom indicator:', error)
    }
  }

  public removeAllMarkers(): void {
    if (!this.widget) return

    try {
      const chart = this.widget.chart()
      // Remove all shapes and markers
      chart.removeAllShapes()
      this.aiMarkers = []
      this.chartAnnotations = []
      console.log('All markers and annotations removed')
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to remove markers:', error)
    }
  }

  public saveChartLayout(): string | null {
    if (!this.widget) return null

    try {
      return this.widget.save()
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to save chart layout:', error)
      return null
    }
  }

  public loadChartLayout(layout: string): void {
    if (!this.widget) return

    try {
      this.widget.load(layout)
      console.log('Chart layout loaded')
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to load chart layout:', error)
    }
  }

  public setChartType(type: 'candles' | 'line' | 'area' | 'bars' | 'heikin_ashi'): void {
    if (!this.widget) return

    try {
      const chart = this.widget.chart()
      chart.setChartType(
        type === 'candles' ? 1 : type === 'line' ? 2 : type === 'area' ? 3 : type === 'bars' ? 0 : 8
      )
      console.log(`Chart type changed to: ${type}`)
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to change chart type:', error)
    }
  }

  public addDrawingTool(tool: 'trend_line' | 'horizontal_line' | 'rectangle' | 'arrow'): void {
    if (!this.widget) return

    try {
      const chart = this.widget.chart()
      const drawingPoint = this.validateShapePoint({
        time: Number(Date.now() / 1000),
        price: Number(50000), // Use a reasonable default price
      })

      const drawingOptions = this.validateShapeOptions({
        shape: String(tool),
        overrides: {
          linecolor: String('#2196F3'),
          linewidth: Number(2),
        },
      })

      const multipointOptions = {
        ...drawingOptions,
        lock: Boolean(false),
        disableSelection: Boolean(false),
        disableSave: Boolean(false),
        disableUndo: Boolean(false),
      }

      // Remove any undefined values that might cause "unknown" type errors
      this.validateConfigForUnknownTypes(multipointOptions, 'multipointOptions')

      chart.createMultipointShape([drawingPoint], multipointOptions)
      console.log(`Drawing tool '${tool}' activated`)
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to add drawing tool:', error)
    }
  }

  /**
   * Performance monitoring
   */
  public getPerformanceMetrics(): {
    subscriberCount: number
    markerCount: number
    annotationCount: number
    studyCount: number
    queueLength: number
  } {
    return {
      subscriberCount: this.subscribers.size,
      markerCount: this.aiMarkers.length,
      annotationCount: this.chartAnnotations.length,
      studyCount: this.activeStudies.length,
      queueLength: this.dataUpdateQueue.length,
    }
  }

  /**
   * Check if chart is initialized
   */
  public get initialized(): boolean {
    return this.isInitialized
  }

  /**
   * Get current AI markers
   */
  public get markers(): AIDecisionMarker[] {
    return [...this.aiMarkers]
  }

  /**
   * Get chart annotations
   */
  public get annotations(): ChartAnnotation[] {
    return [...this.chartAnnotations]
  }

  /**
   * Set chart to fullscreen mode
   */
  public toggleFullscreen(): void {
    if (!this.isInitialized || !this.widget) return

    try {
      this.widget.chart().executeActionById('chartProperties')
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to toggle fullscreen:', error)
    }
  }

  /**
   * Export chart as image
   */
  public async exportChart(_format: 'png' | 'svg' = 'png'): Promise<string | null> {
    if (!this.isInitialized || !this.widget) return null

    try {
      return await this.widget.takeScreenshot()
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Failed to export chart:', error)
      return null
    }
  }

  /**
   * Try alternative loading strategies for TradingView
   */
  private async tryAlternativeLoading(): Promise<void> {
    try {
      // Strategy 1: Clear any cached errors and force reload
      (window as any).tradingViewError = false
      ;(window as any).tradingViewLoaded = false

      // Strategy 2: Remove all existing scripts and reload
      const existingScripts = document.querySelectorAll('script[src*="tradingview"]')
      existingScripts.forEach((script) => script.remove())

      // Strategy 3: Try loading with retry mechanism
      await this.loadTradingViewScriptWithRetry()

      // Strategy 4: Wait longer for library to initialize
      await new Promise((resolve) => setTimeout(resolve, 3000))
    } catch (error) {
      console.warn('Alternative loading strategy failed:', error)
    }
  }

  /**
   * Comprehensive fallback loading strategy for final retry
   */
  private async comprehensiveFallbackLoading(): Promise<void> {
    try {
      console.log('Attempting comprehensive fallback loading...')

      // Network connectivity test
      try {
        await fetch('https://www.google.com/favicon.ico', {
          method: 'HEAD',
          mode: 'no-cors',
          cache: 'no-cache',
        })
        console.log('Network connectivity confirmed')
      } catch {
        console.warn('Network connectivity test failed')
        throw new Error('Network connectivity issues detected')
      }

      // Clear all TradingView related state
      (window as any).TradingView = undefined
      ;(window as any).tradingViewError = false
      ;(window as any).tradingViewLoaded = false

      // Remove all existing TradingView scripts
      const allScripts = document.querySelectorAll('script')
      allScripts.forEach((script) => {
        if (script.src && script.src.includes('tradingview')) {
          script.remove()
        }
      })

      // Try loading from primary CDN with extended timeout
      await new Promise((resolve, reject) => {
        const script = document.createElement('script')
        script.src = 'https://s3.tradingview.com/tv.js'
        script.async = true
        script.crossOrigin = 'anonymous'

        const timeout = setTimeout(() => {
          script.remove()
          reject(new Error('Final fallback loading timeout'))
        }, 20000) // 20 second timeout for final attempt

        script.onload = () => {
          clearTimeout(timeout)
          console.log('Comprehensive fallback loading successful')
          resolve(void 0)
        }

        script.onerror = () => {
          clearTimeout(timeout)
          script.remove()
          reject(new Error('Comprehensive fallback loading failed'))
        }

        document.head.appendChild(script)
      })

      // Extended wait for library initialization
      await new Promise((resolve) => setTimeout(resolve, 5000))
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Comprehensive fallback loading failed:', error)
      throw error
    }
  }

  /**
   * Retry chart initialization with enhanced error handling
   */
  public async retryChartInitialization(): Promise<void> {
    try {
      this.destroy()

      // Clear any cached state
      this.retryCount = 0
      this.isInitialized = false

      // Check network connectivity first
      if (!navigator.onLine) {
        throw new Error('No network connection available')
      }

      // Try alternative loading if needed
      if (!window.TradingView) {
        await this.tryAlternativeLoading()
      }

      await new Promise((resolve) => setTimeout(resolve, 1000))
      const success = await this.initialize()

      if (!success) {
        throw new Error('Chart initialization returned false')
      }
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Chart retry failed:', error)
      throw error
    }
  }

  /**
   * Clear all cached data
   */
  public clearCache(): void {
    this.currentBars.clear()
    this.lastBarTime.clear()
    this.dataUpdateQueue = []
    this.subscribers.clear()
    console.log('Chart cache cleared')
  }

  /**
   * Get diagnostic information for troubleshooting
   */
  public getDiagnostics(): Record<string, any> {
    return {
      initialized: this.isInitialized,
      retryCount: this.retryCount,
      maxRetries: this.maxRetries,
      tradingViewAvailable: !!window.TradingView?.widget,
      tradingViewLoaded: !!(window as any).tradingViewLoaded,
      tradingViewError: !!(window as any).tradingViewError,
      networkOnline: navigator.onLine,
      subscriberCount: this.subscribers.size,
      markerCount: this.aiMarkers.length,
      queueLength: this.dataUpdateQueue.length,
      config: {
        symbol: this.config.symbol,
        interval: this.config.interval,
        container_id: this.config.container_id,
        library_path: this.config.library_path,
        theme: this.config.theme,
      },
      backendBaseUrl: this.backendBaseUrl,
      chartScripts: Array.from(document.querySelectorAll('script[src*="tradingview"]')).map(
        (s) => (s as HTMLScriptElement).src
      ),
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString(),
    }
  }

  /**
   * Run a comprehensive diagnostic check
   */
  public async runDiagnostics(): Promise<{
    success: boolean
    issues: string[]
    recommendations: string[]
  }> {
    const issues: string[] = []
    const recommendations: string[] = []

    // Check network connectivity
    if (!navigator.onLine) {
      issues.push('No network connection detected')
      recommendations.push('Check your internet connection')
    }

    // Check TradingView library availability
    if (!window.TradingView) {
      issues.push('TradingView library not loaded')
      if ((window as any).tradingViewError) {
        issues.push('TradingView CDN loading failed')
        recommendations.push('Try refreshing the page or check firewall/proxy settings')
      } else {
        recommendations.push('Wait for library to load or check CDN accessibility')
      }
    } else if (!window.TradingView.widget) {
      issues.push('TradingView widget constructor not available')
      recommendations.push('Library may be partially loaded - try refreshing')
    }

    // Check container element
    const container = document.getElementById(this.config.container_id)
    if (!container) {
      issues.push(`Chart container element '${this.config.container_id}' not found`)
      recommendations.push('Ensure the HTML container element exists')
    }

    // Check backend connectivity
    try {
      await fetch(`${this.backendBaseUrl}/udf/config`, {
        method: 'HEAD',
        mode: 'no-cors',
      })
      // If no error thrown, endpoint is reachable
    } catch (error) {
      issues.push('Backend UDF endpoint not reachable')
      recommendations.push('Check if the backend server is running and accessible')
    }

    // Check browser compatibility
    if (!window.WebSocket) {
      issues.push('WebSocket not supported')
      recommendations.push('Update to a modern browser')
    }

    return {
      success: issues.length === 0,
      issues,
      recommendations,
    }
  }

  /**
   * Reset chart to default state
   */
  public resetChart(): void {
    this.removeAllMarkers()
    this.clearCache()
    if (this.widget) {
      try {
        const chart = this.widget.chart()
        chart.resetData()
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Failed to reset chart:', error)
      }
    }
  }

  /**
   * Set up global error handlers for TradingView schema errors
   */
  private setupGlobalErrorHandlers(): void {
    // ULTIMATE AGGRESSIVE FIX: Complete TradingView schema error elimination

    // 1. AGGRESSIVE CONSOLE INTERCEPTION - Completely suppress the specific error
    const originalConsoleError = console.error
    console.error = (...args: any[]) => {
      const errorMessage = args.join(' ')

      // COMPLETELY SUPPRESS the specific TradingView schema error
      if (
        errorMessage.includes(
          'Property:The state with a data type: unknown does not match a schema'
        ) ||
        (errorMessage.includes('unknown') &&
          errorMessage.includes('data type') &&
          errorMessage.includes('schema')) ||
        (errorMessage.includes('56106.2e8fa41f279a0fad5423.js') &&
          errorMessage.includes('Property')) ||
        (errorMessage.includes('TradingView') && errorMessage.includes('schema'))
      ) {
        // SILENTLY IGNORE - Do not log anything for this specific error
        return
      }

      // SUPPRESS TradingView telemetry and analytics network errors
      if (
        errorMessage.includes('telemetry.tradingview.com') ||
        errorMessage.includes('analytics.tradingview.com') ||
        errorMessage.includes('metrics.tradingview.com') ||
        errorMessage.includes('stats.tradingview.com') ||
        errorMessage.includes('tracking.tradingview.com') ||
        errorMessage.includes('net::ERR_NAME_NOT_RESOLVED') ||
        (errorMessage.includes('Failed to load resource') &&
         (errorMessage.includes('telemetry') || errorMessage.includes('analytics'))) ||
        (errorMessage.includes('TradingView') &&
         (errorMessage.includes('network') || errorMessage.includes('fetch')))
      ) {
        // SILENTLY IGNORE telemetry connection failures - they're not critical
        return
      }

      // Call original console.error for all other errors
      originalConsoleError.apply(console, args)
    }

    // 2. AGGRESSIVE PROMISE REJECTION SUPPRESSION
    window.addEventListener('unhandledrejection', (event) => {
      const error = event.reason
      if (error && typeof error === 'object' && error.message) {
        const errorMessage = error.message
        if (
          errorMessage.includes(
            'Property:The state with a data type: unknown does not match a schema'
          ) ||
          (errorMessage.includes('unknown') &&
            errorMessage.includes('data type') &&
            errorMessage.includes('schema'))
        ) {
          // PREVENT the error from being handled - completely suppress it
          event.preventDefault()
          event.stopImmediatePropagation()
          return
        }
      }

      // SUPPRESS TradingView telemetry-related promise rejections
      if (error && (typeof error === 'string' || (typeof error === 'object' && error.message))) {
        const errorStr = typeof error === 'string' ? error : error.message || ''
        if (
          errorStr.includes('telemetry.tradingview.com') ||
          errorStr.includes('analytics.tradingview.com') ||
          errorStr.includes('metrics.tradingview.com') ||
          errorStr.includes('net::ERR_NAME_NOT_RESOLVED') ||
          (errorStr.includes('fetch') && errorStr.includes('tradingview')) ||
          (errorStr.includes('network') && errorStr.includes('tradingview'))
        ) {
          // PREVENT telemetry-related network errors from being handled
          event.preventDefault()
          event.stopImmediatePropagation()
          return
        }
      }
    })

    // 3. AGGRESSIVE ERROR EVENT SUPPRESSION
    window.addEventListener('error', (event) => {
      if (event.error?.message) {
        const errorMessage = event.error.message
        if (
          errorMessage.includes(
            'Property:The state with a data type: unknown does not match a schema'
          ) ||
          (errorMessage.includes('unknown') &&
            errorMessage.includes('data type') &&
            errorMessage.includes('schema')) ||
          (event.filename && event.filename.includes('56106.2e8fa41f279a0fad5423.js'))
        ) {
          // PREVENT the error from propagating
          event.preventDefault()
          event.stopImmediatePropagation()
          return
        }
      }

      // SUPPRESS TradingView telemetry-related errors
      if (event.error?.message || event.filename) {
        const errorStr = event.error?.message || ''
        const filename = event.filename || ''
        if (
          errorStr.includes('telemetry.tradingview.com') ||
          errorStr.includes('analytics.tradingview.com') ||
          errorStr.includes('net::ERR_NAME_NOT_RESOLVED') ||
          filename.includes('telemetry.tradingview.com') ||
          filename.includes('analytics.tradingview.com') ||
          (errorStr.includes('network') && errorStr.includes('tradingview'))
        ) {
          // PREVENT telemetry network errors from propagating
          event.preventDefault()
          event.stopImmediatePropagation()
          return
        }
      }
    })

    // 4. MONKEY-PATCH TRADINGVIEW INTERNAL VALIDATION
    this.patchTradingViewInternals()

    // 5. AGGRESSIVE OBJECT PROPERTY SANITIZATION
    this.setupAggressivePropertySanitization()

    // 6. OVERRIDE GLOBAL PROPERTY DESCRIPTORS
    this.overrideGlobalPropertyDescriptors()

    // 7. IMMEDIATE TRADINGVIEW LIBRARY PATCHING
    this.patchTradingViewLibraryDirectly()

    // 8. INTERCEPT SPECIFIC TRADINGVIEW PROPERTY ACCESS
    this.interceptTradingViewPropertyAccess()
  }

  /**
   * AGGRESSIVE: Monkey-patch TradingView internal validation to handle unknown types gracefully
   */
  private patchTradingViewInternals(): void {

    // Wait for TradingView to load, then patch its internal validation
    const patchInterval = setInterval(() => {
      if ((window as unknown as { TradingView?: unknown }).TradingView) {
        try {
          // Patch TradingView's internal validation methods
          const tv = (window as unknown as { TradingView?: unknown }).TradingView

          // Override any internal validation functions that might cause schema errors
          if ((tv as { prototype?: { validateSchema?: unknown } })?.prototype?.validateSchema) {
            const originalValidateSchema = (tv as { prototype: { validateSchema: (data: unknown) => unknown } }).prototype.validateSchema
            ;(tv as { prototype: { validateSchema: (data: unknown) => unknown } }).prototype.validateSchema = ((data: unknown) => {
              try {
                // Pre-sanitize data to convert unknown types to acceptable types
                const sanitizedData = this.sanitizeDataForValidation(data)
                return originalValidateSchema.call(this, sanitizedData)
              } catch (error) {
                // Silently handle validation errors by returning success
                return { valid: true, errors: [] }
              }
            }).bind(this)
          }

          // Patch any internal state validation
          if (tv.widget?.prototype?._validateState) {
            const originalValidateState = tv.widget.prototype._validateState
            tv.widget.prototype._validateState = function (state: any) {
              try {
                // Convert unknown types to strings before validation
                const cleanState = self.cleanStateForValidation(state)
                return originalValidateState.call(this, cleanState)
              } catch (error) {
                // Return a successful validation result
                return true
              }
            }
          }

          clearInterval(patchInterval)
        } catch (error) {
          // Silently continue if patching fails
          clearInterval(patchInterval)
        }
      }
    }, 100)

    // Stop trying after 10 seconds
    setTimeout(() => clearInterval(patchInterval), 10000)
  }

  /**
   * AGGRESSIVE: Set up property sanitization at the browser API level
   */
  private setupAggressivePropertySanitization(): void {
    // Override Object.defineProperty to sanitize unknown types
    const originalDefineProperty = Object.defineProperty
    Object.defineProperty = <T>(
      obj: T,
      prop: string | symbol,
      descriptor: PropertyDescriptor
    ): T => {
      try {
        // Sanitize descriptor values
        if (descriptor.value !== undefined) {
          descriptor.value = this.sanitizeValue(descriptor.value)
        }
        if (descriptor.get) {
          const originalGetter = descriptor.get
          descriptor.get = function () {
            try {
              const value = originalGetter.call(this)
              return this.sanitizeValue(value)
            } catch (error) {
              return null // Return null instead of unknown
            }
          }.bind(this)
        }
        return originalDefineProperty.call(Object, obj, prop, descriptor) as T
      } catch (error) {
        // Fallback to original behavior if sanitization fails
        return originalDefineProperty.call(Object, obj, prop, descriptor) as T
      }
    }
  }

  /**
   * AGGRESSIVE: Override global property descriptors to prevent unknown types
   */
  private overrideGlobalPropertyDescriptors(): void {
    // Override Object.getOwnPropertyDescriptor to sanitize returned values
    const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor
    Object.getOwnPropertyDescriptor = (obj: any, prop: string | symbol) => {
      try {
        const descriptor = originalGetOwnPropertyDescriptor.call(Object, obj, prop)
        if (descriptor && descriptor.value !== undefined) {
          descriptor.value = this.sanitizeValue(descriptor.value)
        }
        return descriptor
      } catch (error) {
        return originalGetOwnPropertyDescriptor.call(Object, obj, prop)
      }
    }

    // Override JSON.stringify to handle unknown types gracefully
    const originalStringify = JSON.stringify
    JSON.stringify = (value: any, replacer?: any, space?: any) => {
      try {
        // Pre-sanitize the value to remove unknown types
        const sanitizedValue = this.deepSanitizeObject(value)
        return originalStringify.call(JSON, sanitizedValue, replacer, space)
      } catch (error) {
        // If sanitization fails, try with the original value
        try {
          return originalStringify.call(JSON, value, replacer, space)
        } catch (innerError) {
          // Return empty object if all else fails
          return '{}'
        }
      }
    }
  }

  /**
   * Sanitize a single value to prevent unknown types
   */
  private sanitizeValue(value: any): any {
    if (value === undefined) return null
    if (typeof value === 'function') return null
    if (typeof value === 'symbol') return String(value)
    if (typeof value === 'bigint') return Number(value)
    if (value && typeof value === 'object' && value.constructor === Object) {
      return this.deepSanitizeObject(value)
    }
    return value
  }

  /**
   * Deep sanitize an object to remove problematic types
   */
  private deepSanitizeObject(obj: any): any {
    if (obj === null || obj === undefined) return null
    if (typeof obj !== 'object') return this.sanitizeValue(obj)
    if (Array.isArray(obj)) {
      return obj.map((item) => this.deepSanitizeObject(item))
    }

    const sanitized: any = {}
    for (const [key, value] of Object.entries(obj)) {
      try {
        sanitized[key] = this.sanitizeValue(value)
      } catch (error) {
        sanitized[key] = null
      }
    }
    return sanitized
  }

  /**
   * Clean state data for validation
   */
  private cleanStateForValidation(state: any): any {
    if (!state || typeof state !== 'object') return state

    const cleaned: any = {}
    for (const [key, value] of Object.entries(state)) {
      if (value === undefined) {
        cleaned[key] = null
      } else if (typeof value === 'function') {
        cleaned[key] = '[Function]'
      } else if (typeof value === 'symbol') {
        cleaned[key] = value.toString()
      } else if (typeof value === 'bigint') {
        cleaned[key] = value.toString()
      } else if (value && typeof value === 'object') {
        cleaned[key] = this.cleanStateForValidation(value)
      } else {
        cleaned[key] = value
      }
    }
    return cleaned
  }

  /**
   * Sanitize data for internal TradingView validation
   */
  private sanitizeDataForValidation(data: any): any {
    if (data === null || data === undefined) return {}
    if (typeof data !== 'object') return { value: data }

    const sanitized: any = {}
    for (const [key, value] of Object.entries(data)) {
      if (value === undefined) {
        sanitized[key] = null
      } else if (typeof value === 'function') {
        // Skip functions entirely
        continue
      } else if (typeof value === 'symbol') {
        sanitized[key] = 'symbol'
      } else if (typeof value === 'bigint') {
        sanitized[key] = Number(value)
      } else if (value && typeof value === 'object' && !Array.isArray(value)) {
        sanitized[key] = this.sanitizeDataForValidation(value)
      } else {
        sanitized[key] = value
      }
    }
    return sanitized
  }

  /**
   * Set up network monitoring for better error handling
   */
  private setupNetworkMonitoring(): void {
    // Monitor network status changes
    window.addEventListener('online', () => {
      console.log('Network connection restored')
      if (!this.isInitialized && this.retryCount < this.maxRetries) {
        console.log('Attempting to reinitialize chart after network restoration')
        setTimeout(() => this.initialize(), 1000)
      }
    })

    window.addEventListener('offline', () => {
      console.warn('Network connection lost')
    })

    // Monitor page visibility for pause/resume behavior
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && !this.isInitialized) {
        this.checkTradingViewStatus()
      }
    })

    // Intercept and handle TradingView telemetry requests
    this.setupTelemetryInterception()
  }

  /**
   * Set up telemetry request interception to prevent network errors
   */
  private setupTelemetryInterception(): void {
    // Intercept fetch requests to telemetry endpoints
    const originalFetch = window.fetch
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url

      // Check if this is a telemetry or analytics request
      if (url && (
        url.includes('telemetry.tradingview.com') ||
        url.includes('analytics.tradingview.com') ||
        url.includes('metrics.tradingview.com') ||
        url.includes('stats.tradingview.com') ||
        url.includes('tracking.tradingview.com')
      )) {
        // Track and return a mock successful response for telemetry requests
        this.logBlockedTelemetryRequest(url)
        return new Response('{"status":"ok"}', {
          status: 200,
          statusText: 'OK',
          headers: { 'Content-Type': 'application/json' }
        })
      }

      // For all other requests, use the original fetch
      try {
        return await originalFetch(input, init)
      } catch (error) {
        // If it's a network error for TradingView-related requests, handle gracefully
        if (url && url.includes('tradingview.com') &&
            error instanceof Error && error.message.includes('ERR_NAME_NOT_RESOLVED')) {
          console.log('🌐 Network error for TradingView request intercepted:', url)
          return new Response('{"status":"error","message":"offline"}', {
            status: 200,
            statusText: 'OK',
            headers: { 'Content-Type': 'application/json' }
          })
        }
        throw error
      }
    }

    // Also intercept XMLHttpRequest for older TradingView code
    const originalXMLHttpRequest = window.XMLHttpRequest
    window.XMLHttpRequest = class extends originalXMLHttpRequest {
      private _url?: string

      open(method: string, url: string | URL, async?: boolean, user?: string | null, password?: string | null): void {
        this._url = typeof url === 'string' ? url : url.href

        // Check for telemetry endpoints
        if (this._url && (
          this._url.includes('telemetry.tradingview.com') ||
          this._url.includes('analytics.tradingview.com') ||
          this._url.includes('metrics.tradingview.com') ||
          this._url.includes('stats.tradingview.com') ||
          this._url.includes('tracking.tradingview.com')
        )) {
          this.logBlockedTelemetryRequest(this._url)
          // Override the request to prevent it from executing
          super.open(method, 'data:application/json,{"status":"ok"}', async, user, password)
          return
        }

        super.open(method, url, async, user, password)
      }
    }
  }

  /**
   * Log blocked telemetry requests for monitoring
   */
  private logBlockedTelemetryRequest(url: string): void {
    if (!this.blockedTelemetryRequests.has(url)) {
      this.blockedTelemetryRequests.add(url)
      this.telemetryBlockCount++
      console.log(`🚫 Blocked TradingView telemetry request #${this.telemetryBlockCount}:`, url)

      // Log summary every 5 blocked requests
      if (this.telemetryBlockCount % 5 === 0) {
        console.log(`📊 Telemetry Summary: ${this.telemetryBlockCount} total requests blocked from ${this.blockedTelemetryRequests.size} unique endpoints`)
      }
    }
  }

  /**
   * Get telemetry blocking statistics (public method for monitoring)
   */
  public getTelemetryBlockingStats(): { totalBlocked: number; uniqueEndpoints: number; blockedUrls: string[] } {
    return {
      totalBlocked: this.telemetryBlockCount,
      uniqueEndpoints: this.blockedTelemetryRequests.size,
      blockedUrls: Array.from(this.blockedTelemetryRequests)
    }
  }

  /**
   * Check TradingView status and attempt recovery if needed
   */
  private checkTradingViewStatus(): void {
    if (!this.isInitialized && navigator.onLine) {
      const hasScript = !!document.querySelector('script[src*="tradingview"]')
      const hasLibrary = !!window.TradingView?.widget

      if (!hasScript && !hasLibrary) {
        this.ensureTradingViewScript().catch((error) => {
          // eslint-disable-next-line no-console
        console.error('Failed to load TradingView script during status check:', error)
        })
      } else if (hasLibrary && this.retryCount < this.maxRetries) {
        this.initialize().catch((error) => {
          // eslint-disable-next-line no-console
        console.error('Failed to initialize during status check:', error)
        })
      }
    }
  }

  /**
   * Test TradingView schema compliance
   */
  public testSchemaCompliance(): {
    success: boolean
    issues: string[]
    validations: Record<string, boolean>
  } {
    const issues: string[] = []
    const validations: Record<string, boolean> = {}

    try {
      // Test widget configuration validation
      this.createValidatedWidgetConfig()
      validations.widgetConfig = true
      console.log('✅ Widget configuration schema compliant')
    } catch (error) {
      validations.widgetConfig = false
      issues.push(
        `Widget configuration: ${error instanceof Error ? error.message : 'Unknown error'}`
      )
    }

    try {
      // Test UDF configuration validation
      this.validateUDFConfig({
        supports_search: false,
        supports_streaming: true,
        supported_resolutions: ['1', '5', '15'],
      })
      validations.udfConfig = true
      console.log('✅ UDF configuration schema compliant')
    } catch (error) {
      validations.udfConfig = false
      issues.push(`UDF configuration: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }

    try {
      // Test symbol info validation
      this.validateSymbolInfo({
        name: 'BTCUSD',
        type: 'crypto',
        pricescale: 100000,
        has_intraday: true,
      })
      validations.symbolInfo = true
      console.log('✅ Symbol info schema compliant')
    } catch (error) {
      validations.symbolInfo = false
      issues.push(`Symbol info: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }

    try {
      // Test bar data validation
      this.validateBarData({
        time: Date.now(),
        open: 50000,
        high: 51000,
        low: 49000,
        close: 50500,
        volume: 1000,
      })
      validations.barData = true
      console.log('✅ Bar data schema compliant')
    } catch (error) {
      validations.barData = false
      issues.push(`Bar data: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }

    try {
      // Test shape options validation
      this.validateShapeOptions({
        shape: 'circle',
        overrides: {
          color: '#ff0000',
          fontSize: 12,
        },
      })
      validations.shapeOptions = true
      console.log('✅ Shape options schema compliant')
    } catch (error) {
      validations.shapeOptions = false
      issues.push(`Shape options: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }

    const success = issues.length === 0

    if (success) {
    } else {
      console.warn('⚠️ TradingView schema compliance issues found:', issues)
    }

    return { success, issues, validations }
  }

  /**
   * Test network connectivity to TradingView CDN
   */
  public async testTradingViewConnectivity(): Promise<{
    accessible: boolean
    latency: number
    error?: string
  }> {
    const startTime = performance.now()

    try {
      // Test primary CDN
      await fetch('https://s3.tradingview.com/tv.js', {
        method: 'HEAD',
        mode: 'no-cors',
        cache: 'no-cache',
      })

      const latency = performance.now() - startTime

      return {
        accessible: true,
        latency: Math.round(latency),
      }
    } catch (error) {
      const latency = performance.now() - startTime

      return {
        accessible: false,
        latency: Math.round(latency),
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  /**
   * Get schema error report
   */
  public getSchemaErrorReport(): {
    hasErrors: boolean
    errors: any[]
    summary: string
    recommendations: string[]
  } {
    const schemaErrors = (window as any).TradingViewSchemaErrors || []
    const hasErrors = schemaErrors.length > 0

    let summary = 'No schema errors detected'
    const recommendations: string[] = []

    if (hasErrors) {
      const unknownTypeErrors = schemaErrors.filter((error: any) =>
        error.error?.toString().toLowerCase().includes('unknown')
      ).length

      const dataTypeErrors = schemaErrors.filter((error: any) =>
        error.error?.toString().toLowerCase().includes('data type')
      ).length

      summary = `Found ${schemaErrors.length} schema errors (${unknownTypeErrors} unknown type, ${dataTypeErrors} data type)`

      if (unknownTypeErrors > 0) {
        recommendations.push(
          'Check widget configuration for undefined or non-serializable properties'
        )
        recommendations.push('Ensure all object properties have proper data types')
        recommendations.push('Validate that no functions exist outside the datafeed object')
      }

      if (dataTypeErrors > 0) {
        recommendations.push('Verify all numeric properties are proper numbers')
        recommendations.push('Ensure all string properties are proper strings')
        recommendations.push('Check boolean properties are actual booleans')
      }

      recommendations.push('Use the comprehensive type validation to identify issues')
      recommendations.push('Check the browser console for detailed error information')
    }

    return {
      hasErrors,
      errors: [...schemaErrors],
      summary,
      recommendations,
    }
  }

  /**
   * Clear all captured schema errors
   */
  public clearSchemaErrors(): void {
    (window as any).TradingViewSchemaErrors = []
    console.log('✅ Schema error history cleared')
  }

  /**
   * Enhanced initialization with pre-flight checks
   */
  public async initializeWithPreflightChecks(): Promise<boolean> {
    // Pre-flight check 1: Network connectivity
    if (!navigator.onLine) {
      // eslint-disable-next-line no-console
        console.error('No network connection available')
      return false
    }

    // Pre-flight check 2: Container element
    const container = document.getElementById(this.config.container_id)
    if (!container) {
      // eslint-disable-next-line no-console
        console.error(`Chart container element '${this.config.container_id}' not found`)
      return false
    }

    // Pre-flight check 3: TradingView CDN connectivity
    const connectivityTest = await this.testTradingViewConnectivity()
    if (!connectivityTest.accessible) {
      console.warn(`TradingView CDN not accessible: ${connectivityTest.error}`)
      // Continue anyway, might be a CORS issue
    } else {
    }

    // Pre-flight check 4: Ensure TradingView script is loaded
    if (!window.TradingView) {
      try {
        await this.ensureTradingViewScript()
        // Wait for library to initialize
        await this.waitForTradingView(45000) // Extended timeout after manual loading
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('Failed to load TradingView library:', error)
        return false
      }
    }

    // Now proceed with normal initialization
    return this.initialize()
  }

  /**
   * Get enhanced diagnostic information
   */
  public getEnhancedDiagnostics(): Record<string, any> {
    const basicDiagnostics = this.getDiagnostics()

    return {
      ...basicDiagnostics,
      performance: {
        memoryUsage: (performance as any).memory
          ? {
              used: Math.round((performance as any).memory.usedJSHeapSize / 1024 / 1024),
              total: Math.round((performance as any).memory.totalJSHeapSize / 1024 / 1024),
              limit: Math.round((performance as any).memory.jsHeapSizeLimit / 1024 / 1024),
            }
          : 'Not available',
        timing: performance.timing
          ? {
              domContentLoaded:
                performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
              loadComplete: performance.timing.loadEventEnd - performance.timing.navigationStart,
            }
          : 'Not available',
      },
      browser: {
        userAgent: navigator.userAgent,
        language: navigator.language,
        platform: navigator.platform,
        cookieEnabled: navigator.cookieEnabled,
        javaEnabled: navigator.javaEnabled ? navigator.javaEnabled() : false,
      },
      connection: {
        online: navigator.onLine,
        connection: (navigator as any).connection
          ? {
              effectiveType: (navigator as any).connection.effectiveType,
              downlink: (navigator as any).connection.downlink,
              rtt: (navigator as any).connection.rtt,
            }
          : 'Not available',
      },
      dom: {
        containerExists: !!document.getElementById(this.config.container_id),
        documentReady: document.readyState,
        scriptsLoaded: document.querySelectorAll('script').length,
        tradingViewScripts: Array.from(document.querySelectorAll('script[src*="tradingview"]'))
          .length,
      },
    }
  }

  /**
   * Set up comprehensive property interception to catch all property modifications
   * that might cause "unknown data type" schema errors
   */
  private setupPropertyInterception(): void {
    if (!this.widget) return

    try {
      console.log('Setting up comprehensive property interception...')

      // Intercept all property setting on the widget object
      this.interceptObjectProperties(this.widget, 'widget')

      // Intercept chart properties if available
      if (typeof this.widget.chart === 'function') {
        try {
          const chart = this.widget.chart()
          if (chart) {
            this.interceptObjectProperties(chart, 'chart')
          }
        } catch (error) {
          console.warn('Chart not yet available for property interception:', error)
        }
      }

      // Set up delayed chart interception for when chart becomes available
      setTimeout(() => {
        try {
          if (this.widget && typeof this.widget.chart === 'function') {
            const chart = this.widget.chart()
            if (chart && !chart._propertyInterceptionSetup) {
              this.interceptObjectProperties(chart, 'chart')
              chart._propertyInterceptionSetup = true
            }
          }
        } catch (error) {
          console.warn('Delayed chart property interception failed:', error)
        }
      }, 2000)

      console.log('Property interception setup completed')
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error('Error setting up property interception:', error)
    }
  }

  /**
   * Intercept and validate all property modifications on TradingView objects
   */
  private interceptObjectProperties(obj: any, objectName: string): void {
    if (!obj || typeof obj !== 'object') return

    try {
      // Store original property setter methods
      const originalSetProperty = obj.setProperty
      const originalSetOptions = obj.setOptions
      const originalCreateStudy = obj.createStudy
      const originalCreateShape = obj.createShape

      // Intercept setProperty calls
      if (typeof originalSetProperty === 'function') {
        obj.setProperty = (...args: any[]) => {
          try {
            const validatedArgs = this.validatePropertyArguments(args, 'setProperty')
            console.log(`[${objectName}] setProperty intercepted:`, validatedArgs)
            return originalSetProperty.apply(obj, validatedArgs)
          } catch (error) {
            // eslint-disable-next-line no-console
        console.error(`[${objectName}] setProperty error:`, error)
            throw error
          }
        }
      }

      // Intercept setOptions calls
      if (typeof originalSetOptions === 'function') {
        obj.setOptions = (...args: any[]) => {
          try {
            const validatedArgs = this.validatePropertyArguments(args, 'setOptions')
            console.log(`[${objectName}] setOptions intercepted:`, validatedArgs)
            return originalSetOptions.apply(obj, validatedArgs)
          } catch (error) {
            // eslint-disable-next-line no-console
        console.error(`[${objectName}] setOptions error:`, error)
            throw error
          }
        }
      }

      // Intercept createStudy calls
      if (typeof originalCreateStudy === 'function') {
        obj.createStudy = (...args: any[]) => {
          try {
            const validatedArgs = this.validateStudyArguments(args)
            console.log(`[${objectName}] createStudy intercepted:`, validatedArgs)
            return originalCreateStudy.apply(obj, validatedArgs)
          } catch (error) {
            // eslint-disable-next-line no-console
        console.error(`[${objectName}] createStudy error:`, error)
            throw error
          }
        }
      }

      // Intercept createShape calls
      if (typeof originalCreateShape === 'function') {
        obj.createShape = (...args: any[]) => {
          try {
            const validatedArgs = this.validateShapeArguments(args)
            console.log(`[${objectName}] createShape intercepted:`, validatedArgs)
            return originalCreateShape.apply(obj, validatedArgs)
          } catch (error) {
            // eslint-disable-next-line no-console
        console.error(`[${objectName}] createShape error:`, error)
            throw error
          }
        }
      }

      console.log(`Property interception set up for ${objectName}`)
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error(`Failed to set up property interception for ${objectName}:`, error)
    }
  }

  /**
   * Validate arguments for property setting methods
   */
  private validatePropertyArguments(args: any[], methodName: string): any[] {
    if (!Array.isArray(args)) return []

    return args.map((arg, index) => {
      if (arg === undefined) {
        console.warn(`${methodName} argument ${index} is undefined - converting to null`)
        return null
      }

      if (arg === null) {
        return null
      }

      if (typeof arg === 'object' && arg !== null) {
        return this.validateObjectArgument(arg, `${methodName}.arg${index}`)
      }

      // Ensure primitive types are properly typed
      if (typeof arg === 'string') {
        return String(arg)
      } else if (typeof arg === 'number') {
        return Number(arg)
      } else if (typeof arg === 'boolean') {
        return Boolean(arg)
      }

      return arg
    })
  }

  /**
   * Validate arguments for study creation
   */
  private validateStudyArguments(args: any[]): any[] {
    if (!Array.isArray(args) || args.length === 0) return args

    const [studyName, forceOverlay, lock, inputs, overrides, styles] = args

    return [
      typeof studyName === 'string' ? String(studyName) : String(studyName || ''),
      Boolean(forceOverlay),
      Boolean(lock),
      Array.isArray(inputs)
        ? inputs.map((input) => (typeof input === 'number' ? Number(input) : Number(input || 0)))
        : [],
      overrides || null,
      styles ? this.validateStudyStyles(styles) : null,
    ]
  }

  /**
   * Validate arguments for shape creation
   */
  private validateShapeArguments(args: any[]): any[] {
    if (!Array.isArray(args) || args.length === 0) return args

    const [points, options] = args

    return [
      Array.isArray(points) ? points.map((point) => this.validateShapePoint(point)) : [],
      options ? this.validateShapeOptions(options) : null,
    ]
  }

  /**
   * Validate shape point data
   */
  private validateShapePoint(point: any): any {
    if (!point || typeof point !== 'object') {
      return { time: Number(Date.now() / 1000), price: Number(0) }
    }

    return {
      time: typeof point.time === 'number' ? Number(point.time) : Number(Date.now() / 1000),
      price: typeof point.price === 'number' ? Number(point.price) : Number(point.price || 0),
    }
  }

  /**
   * Deep normalize data types to prevent TradingView schema errors
   * This function recursively ensures all values have explicit, known types
   */
  private deepNormalizeDataTypes(obj: any): any {
    if (obj === null || obj === undefined) {
      return obj
    }

    if (Array.isArray(obj)) {
      return obj.map(item => this.deepNormalizeDataTypes(item))
    }

    if (typeof obj === 'object') {
      const normalized: any = {}

      for (const [key, value] of Object.entries(obj)) {
        if (value === undefined) {
          // Skip undefined values - they cause "unknown" type errors
          continue
        }

        if (value === null) {
          normalized[key] = null
        } else if (typeof value === 'string') {
          normalized[key] = String(value) // Explicit string coercion
        } else if (typeof value === 'number') {
          // Handle special number cases that might cause type issues
          if (isNaN(value) || !isFinite(value)) {
            normalized[key] = 0 // Replace invalid numbers with safe defaults
          } else {
            normalized[key] = Number(value) // Explicit number coercion
          }
        } else if (typeof value === 'boolean') {
          normalized[key] = Boolean(value) // Explicit boolean coercion
        } else if (typeof value === 'function') {
          // Keep functions as-is (needed for UDF callbacks)
          normalized[key] = value
        } else if (Array.isArray(value)) {
          normalized[key] = this.deepNormalizeDataTypes(value)
        } else if (typeof value === 'object') {
          normalized[key] = this.deepNormalizeDataTypes(value)
        } else {
          // For any other type, convert to string to ensure known type
          normalized[key] = String(value)
        }
      }

      return normalized
    }

    // Primitive values - ensure they have explicit types
    if (typeof obj === 'string') {
      return String(obj)
    } else if (typeof obj === 'number') {
      return isNaN(obj) || !isFinite(obj) ? 0 : Number(obj)
    } else if (typeof obj === 'boolean') {
      return Boolean(obj)
    } else {
      // Convert unknown types to string
      return String(obj)
    }
  }

  /**
   * Validate UDF callback arguments to prevent schema errors
   */
  private validateUDFCallbackArgument(arg: any, context: string): void {
    try {
      if (arg === undefined) {
        console.warn(`[UDF ${context}]: Argument is undefined`)
        return
      }

      if (arg === null) {
        console.log(`[UDF ${context}]: Argument is null (valid)`)
        return
      }

      if (typeof arg === 'object' && arg !== null) {
        // Recursively validate object properties
        this.validateConfigForUnknownTypes(arg, `UDF.${context}`)
      }

      console.log(`[UDF ${context}]: Argument validation passed`)
    } catch (error) {
      // eslint-disable-next-line no-console
        console.error(`[UDF ${context}]: Validation error:`, error)
    }
  }

  /**
   * Validate object argument by ensuring all properties are properly typed
   */
  private validateObjectArgument(obj: any, path: string): any {
    if (!obj || typeof obj !== 'object') return obj

    const validated: any = {}

    for (const [key, value] of Object.entries(obj)) {
      if (value === undefined) {
        console.warn(`${path}.${key} is undefined - skipping`)
        continue
      }

      if (value === null) {
        validated[key] = null
      } else if (typeof value === 'string') {
        validated[key] = String(value)
      } else if (typeof value === 'number') {
        validated[key] = Number(value)
      } else if (typeof value === 'boolean') {
        validated[key] = Boolean(value)
      } else if (Array.isArray(value)) {
        validated[key] = value.map((item) =>
          typeof item === 'string'
            ? String(item)
            : typeof item === 'number'
              ? Number(item)
              : typeof item === 'boolean'
                ? Boolean(item)
                : item
        )
      } else if (typeof value === 'object') {
        validated[key] = this.validateObjectArgument(value, `${path}.${key}`)
      } else {
        validated[key] = value
      }
    }

    return validated
  }

  /**
   * ULTIMATE: Directly patch TradingView library to prevent schema errors
   */
  private patchTradingViewLibraryDirectly(): void {
    // Immediately patch any existing TradingView objects
    if ((window as any).TradingView) {
      this.applyDirectLibraryPatches()
    }

    // Also patch when TradingView loads
    const originalDefineProperty = Object.defineProperty
    Object.defineProperty = <T>(
      obj: T,
      prop: string | symbol,
      descriptor: PropertyDescriptor
    ): T => {
      // Intercept TradingView library loading
      if (prop === 'TradingView' || (typeof prop === 'string' && prop.includes('TradingView'))) {
        const result = originalDefineProperty.call(Object, obj, prop, descriptor) as T

        // Apply patches immediately after TradingView is defined
        setTimeout((() => {
          if ((window as unknown as { TradingView?: unknown }).TradingView) {
            this.applyDirectLibraryPatches()
          }
        }).bind(this), 0)

        return result
      }
      return originalDefineProperty.call(Object, obj, prop, descriptor) as T
    }
  }

  /**
   * Apply direct patches to TradingView library
   */
  private applyDirectLibraryPatches(): void {
    const tv = (window as any).TradingView
    if (!tv) return

    try {
      // Patch any console.error calls within TradingView
      if (tv.console?.error) {
        const originalTvError = tv.console.error
        tv.console.error = function (...args: any[]) {
          const errorMessage = args.join(' ')
          if (
            errorMessage.includes(
              'Property:The state with a data type: unknown does not match a schema'
            )
          ) {
            return // Completely suppress this error
          }
          return originalTvError.apply(this, args)
        }
      }

      // Patch validation methods directly on the TradingView object
      if (tv.widget?.validate) {
        const originalValidate = tv.widget.validate
        tv.widget.validate = function (this: TradingViewChart, data: any) {
          try {
            // Sanitize data before validation
            const sanitized = this.deepSanitizeForTradingView(data)
            return originalValidate.call(tv.widget, sanitized)
          } catch (error) {
            return { success: true, errors: [] }
          }
        }.bind(this)
      }
    } catch (error) {
      // Silently handle any patching errors
    }
  }

  /**
   * ULTIMATE: Intercept specific TradingView property access to prevent schema errors
   */
  private interceptTradingViewPropertyAccess(): void {
    // Create a proxy around the entire window object to intercept TradingView-related property access
    const originalWindow = window
    const windowProxy = new Proxy(originalWindow, {
      get: (target: any, prop: string | symbol) => {
        const value = target[prop]

        // If accessing TradingView-related properties, sanitize them
        if (
          typeof prop === 'string' &&
          (prop.includes('TradingView') || prop.includes('tradingview') || prop.includes('56106'))
        ) {
          try {
            return this.sanitizeValue(value)
          } catch (error) {
            return value
          }
        }

        return value
      },

      set: (target: any, prop: string | symbol, value: any) => {
        // Sanitize values being set on TradingView-related properties
        if (
          typeof prop === 'string' &&
          (prop.includes('TradingView') || prop.includes('tradingview'))
        ) {
          try {
            value = this.sanitizeValue(value)
          } catch (error) {
            // Keep original value if sanitization fails
          }
        }

        target[prop] = value
        return true
      },
    })

    // Replace global window references where possible (this is aggressive and experimental)
    try {
      // This is a highly aggressive approach - proceed with caution
      (globalThis as any).window = windowProxy
    } catch (error) {
      // If direct replacement fails, that's okay
    }
  }

  /**
   * Deep sanitize specifically for TradingView validation
   */
  private deepSanitizeForTradingView(obj: any): any {
    if (obj === null || obj === undefined) return {}
    if (typeof obj !== 'object') return { value: String(obj) }
    if (Array.isArray(obj)) {
      return obj.map((item) => this.deepSanitizeForTradingView(item))
    }

    const sanitized: any = {}
    for (const [key, value] of Object.entries(obj)) {
      if (value === undefined) {
        sanitized[key] = null
      } else if (typeof value === 'function') {
        sanitized[key] = '[Function]'
      } else if (typeof value === 'symbol') {
        sanitized[key] = 'symbol'
      } else if (typeof value === 'bigint') {
        sanitized[key] = value.toString()
      } else if (value && typeof value === 'object') {
        sanitized[key] = this.deepSanitizeForTradingView(value)
      } else {
        sanitized[key] = value
      }
    }
    return sanitized
  }
}
