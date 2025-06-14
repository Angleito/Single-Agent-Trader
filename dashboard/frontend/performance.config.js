/**
 * Performance Configuration for AI Trading Bot Dashboard
 * This file contains settings and optimizations for production builds
 */

export const PERFORMANCE_CONFIG = {
  // Memory limits
  MAX_LOG_ENTRIES: 50,
  MAX_CHART_DATA_POINTS: 1000,
  MAX_WEBSOCKET_QUEUE: 50,
  MAX_DOM_CACHE_SIZE: 30,
  MAX_PERFORMANCE_METRICS: 50,
  
  // Update intervals (milliseconds)
  DOM_UPDATE_BATCH_DELAY: 16, // ~60fps
  MEMORY_CLEANUP_INTERVAL: 30000, // 30 seconds
  PERFORMANCE_MONITOR_INTERVAL: 60000, // 1 minute
  WEBSOCKET_THROTTLE_DELAY: 50, // 50ms
  UI_UPDATE_THROTTLE: 100, // 100ms
  
  // Chart optimization
  CHART_UPDATE_THROTTLE: 150,
  CHART_CLEANUP_INTERVAL: 60000,
  MAX_CHART_MARKERS: 100,
  MAX_CHART_ANNOTATIONS: 50,
  
  // WebSocket optimization
  WEBSOCKET_RECONNECT_DELAY: 1000,
  WEBSOCKET_MAX_RECONNECT_ATTEMPTS: 5,
  WEBSOCKET_PING_INTERVAL: 30000,
  WEBSOCKET_PONG_TIMEOUT: 60000,
  
  // Virtual scrolling
  VIRTUAL_SCROLL_ITEM_HEIGHT: 32,
  VIRTUAL_SCROLL_BUFFER_SIZE: 10,
  VIRTUAL_SCROLL_MAX_RENDERED: 100,
  
  // Animation and transitions
  ANIMATION_DURATION: {
    FAST: 200,
    NORMAL: 300,
    SLOW: 500
  },
  
  // Bundle optimization targets
  BUNDLE_SIZE_LIMITS: {
    MAIN_CHUNK: 1000, // 1MB
    VENDOR_CHUNK: 500, // 500KB
    CSS_CHUNK: 100 // 100KB
  },
  
  // Browser performance targets
  PERFORMANCE_TARGETS: {
    FIRST_CONTENTFUL_PAINT: 1500, // 1.5s
    LARGEST_CONTENTFUL_PAINT: 3000, // 3s
    CUMULATIVE_LAYOUT_SHIFT: 0.1,
    FIRST_INPUT_DELAY: 100, // 100ms
    MEMORY_USAGE_MB: 100 // 100MB max
  },
  
  // Feature flags for performance
  FEATURES: {
    ENABLE_VIRTUAL_SCROLLING: true,
    ENABLE_DOM_CACHING: true,
    ENABLE_MEMORY_MONITORING: true,
    ENABLE_UPDATE_BATCHING: true,
    ENABLE_WEBSOCKET_THROTTLING: true,
    ENABLE_CHART_OPTIMIZATION: true,
    ENABLE_SERVICE_WORKER: false, // Disabled for now
    ENABLE_PERFORMANCE_OBSERVER: true
  },
  
  // Development vs Production settings
  isDevelopment: () => process.env.NODE_ENV === 'development',
  isProduction: () => process.env.NODE_ENV === 'production',
  
  // Get optimized config based on environment
  getConfig() {
    const base = { ...this };
    
    if (this.isDevelopment()) {
      // More relaxed limits for development
      return {
        ...base,
        MAX_LOG_ENTRIES: 100,
        MEMORY_CLEANUP_INTERVAL: 60000,
        PERFORMANCE_MONITOR_INTERVAL: 30000
      };
    }
    
    // Production optimizations
    return {
      ...base,
      MAX_LOG_ENTRIES: 30,
      MEMORY_CLEANUP_INTERVAL: 20000,
      PERFORMANCE_MONITOR_INTERVAL: 120000
    };
  }
};

// Export individual optimizations for tree-shaking
export const MEMORY_LIMITS = PERFORMANCE_CONFIG.MAX_LOG_ENTRIES;
export const UPDATE_INTERVALS = {
  DOM_BATCH: PERFORMANCE_CONFIG.DOM_UPDATE_BATCH_DELAY,
  MEMORY_CLEANUP: PERFORMANCE_CONFIG.MEMORY_CLEANUP_INTERVAL,
  WEBSOCKET_THROTTLE: PERFORMANCE_CONFIG.WEBSOCKET_THROTTLE_DELAY
};

export const CHART_CONFIG = {
  UPDATE_THROTTLE: PERFORMANCE_CONFIG.CHART_UPDATE_THROTTLE,
  MAX_MARKERS: PERFORMANCE_CONFIG.MAX_CHART_MARKERS,
  MAX_ANNOTATIONS: PERFORMANCE_CONFIG.MAX_CHART_ANNOTATIONS
};

export const WEBSOCKET_CONFIG = {
  THROTTLE_DELAY: PERFORMANCE_CONFIG.WEBSOCKET_THROTTLE_DELAY,
  MAX_QUEUE_SIZE: PERFORMANCE_CONFIG.MAX_WEBSOCKET_QUEUE,
  RECONNECT_DELAY: PERFORMANCE_CONFIG.WEBSOCKET_RECONNECT_DELAY
};

export default PERFORMANCE_CONFIG;