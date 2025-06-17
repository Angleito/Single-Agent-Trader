/**
 * Advanced Service Worker for Trading Bot Dashboard
 * 
 * Provides comprehensive PWA capabilities:
 * - Intelligent caching strategies with cache versioning
 * - Offline-first architecture with background sync
 * - Push notification handling
 * - Real-time data synchronization
 * - Performance optimization with preloading
 * - Cache management and cleanup
 * - Network resilience and fallbacks
 * - Analytics and monitoring
 */

declare var self: ServiceWorkerGlobalScope;

// Cache configuration
const CACHE_CONFIG = {
  APP_CACHE: 'trading-bot-app-v1.0.0',
  API_CACHE: 'trading-bot-api-v1.0.0',
  STATIC_CACHE: 'trading-bot-static-v1.0.0',
  RUNTIME_CACHE: 'trading-bot-runtime-v1.0.0',
  MAX_ENTRIES: 100,
  MAX_AGE_SECONDS: 60 * 60 * 24 * 7, // 7 days
  STALE_WHILE_REVALIDATE_AGE: 60 * 60 * 24 // 24 hours
};

// Network strategies
const CACHE_STRATEGIES = {
  CACHE_FIRST: 'cache-first',
  NETWORK_FIRST: 'network-first',
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate',
  NETWORK_ONLY: 'network-only',
  CACHE_ONLY: 'cache-only'
};

// Routes configuration
const ROUTES_CONFIG = [
  {
    pattern: /^https:\/\/.*\/api\/bot\/market-data/,
    strategy: CACHE_STRATEGIES.STALE_WHILE_REVALIDATE,
    cache: CACHE_CONFIG.API_CACHE,
    maxAge: 30 * 1000, // 30 seconds
    networkTimeout: 3000
  },
  {
    pattern: /^https:\/\/.*\/api\/bot\/positions/,
    strategy: CACHE_STRATEGIES.NETWORK_FIRST,
    cache: CACHE_CONFIG.API_CACHE,
    maxAge: 60 * 1000, // 1 minute
    networkTimeout: 5000
  },
  {
    pattern: /^https:\/\/.*\/api\/bot\/risk/,
    strategy: CACHE_STRATEGIES.NETWORK_FIRST,
    cache: CACHE_CONFIG.API_CACHE,
    maxAge: 60 * 1000, // 1 minute
    networkTimeout: 5000
  },
  {
    pattern: /^https:\/\/.*\/api\/bot\/analytics/,
    strategy: CACHE_STRATEGIES.STALE_WHILE_REVALIDATE,
    cache: CACHE_CONFIG.API_CACHE,
    maxAge: 5 * 60 * 1000, // 5 minutes
    networkTimeout: 10000
  },
  {
    pattern: /^https:\/\/.*\/api\/bot\/commands/,
    strategy: CACHE_STRATEGIES.NETWORK_ONLY,
    cache: null,
    maxAge: 0,
    networkTimeout: 30000
  },
  {
    pattern: /\.(js|css|woff2?|ttf|eot)$/,
    strategy: CACHE_STRATEGIES.CACHE_FIRST,
    cache: CACHE_CONFIG.STATIC_CACHE,
    maxAge: CACHE_CONFIG.MAX_AGE_SECONDS * 1000
  },
  {
    pattern: /\.(png|jpg|jpeg|gif|svg|ico|webp)$/,
    strategy: CACHE_STRATEGIES.CACHE_FIRST,
    cache: CACHE_CONFIG.STATIC_CACHE,
    maxAge: CACHE_CONFIG.MAX_AGE_SECONDS * 1000
  },
  {
    pattern: /^https:\/\/.*\/(dashboard|trade|risk|analytics)/,
    strategy: CACHE_STRATEGIES.NETWORK_FIRST,
    cache: CACHE_CONFIG.APP_CACHE,
    maxAge: 60 * 60 * 1000, // 1 hour
    networkTimeout: 5000
  }
];

// Static assets to precache
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/dashboard',
  '/trade',
  '/risk',
  '/analytics',
  '/styles/main.css',
  '/scripts/main.js',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
  '/manifest.json'
];

// Background sync configuration
const SYNC_CONFIG = {
  QUEUE_NAME: 'tradingbot-sync',
  MAX_RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 5000,
  BATCH_SIZE: 10
};

// Push notification configuration
const NOTIFICATION_CONFIG = {
  DEFAULT_ICON: '/icons/icon-192x192.png',
  DEFAULT_BADGE: '/icons/badge-72x72.png',
  MAX_NOTIFICATIONS: 5,
  AUTO_CLOSE_DELAY: 10000
};

// Global variables for tracking
let backgroundSyncQueue: any[] = [];
let notificationQueue: any[] = [];
let performanceMetrics = {
  cacheHits: 0,
  cacheMisses: 0,
  networkRequests: 0,
  backgroundSyncs: 0,
  pushNotifications: 0
};

/**
 * Service Worker Installation
 */
self.addEventListener('install', (event: ExtendableEvent) => {
  console.log('[SW] Installing service worker...');
  
  event.waitUntil(
    Promise.all([
      precacheAssets(),
      setupCaches(),
      self.skipWaiting()
    ])
  );
});

/**
 * Service Worker Activation
 */
self.addEventListener('activate', (event: ExtendableEvent) => {
  console.log('[SW] Activating service worker...');
  
  event.waitUntil(
    Promise.all([
      cleanupOldCaches(),
      setupBackgroundSync(),
      self.clients.claim()
    ])
  );
});

/**
 * Fetch Event Handler
 */
self.addEventListener('fetch', (event: FetchEvent) => {
  // Only handle GET requests for now
  if (event.request.method !== 'GET') {
    return;
  }

  const url = new URL(event.request.url);
  const route = findMatchingRoute(url.href);
  
  if (route) {
    event.respondWith(handleRequest(event.request, route));
  }
});

/**
 * Background Sync Event Handler
 */
self.addEventListener('sync', (event: any) => {
  console.log('[SW] Background sync triggered:', event.tag);
  
  if (event.tag === SYNC_CONFIG.QUEUE_NAME) {
    event.waitUntil(processBackgroundSync());
  }
});

/**
 * Push Notification Event Handler
 */
self.addEventListener('push', (event: PushEvent) => {
  console.log('[SW] Push notification received');
  
  if (!event.data) return;
  
  try {
    const data = event.data.json();
    event.waitUntil(handlePushNotification(data));
  } catch (error) {
    console.error('[SW] Error parsing push notification data:', error);
  }
});

/**
 * Notification Click Event Handler
 */
self.addEventListener('notificationclick', (event: NotificationEvent) => {
  console.log('[SW] Notification clicked:', event.notification.tag);
  
  event.notification.close();
  event.waitUntil(handleNotificationClick(event));
});

/**
 * Message Event Handler (from main thread)
 */
self.addEventListener('message', (event: ExtendableMessageEvent) => {
  console.log('[SW] Message received:', event.data);
  
  const { type, payload } = event.data;
  
  switch (type) {
    case 'CACHE_UPDATE':
      event.waitUntil(updateCache(payload));
      break;
    case 'SYNC_DATA':
      event.waitUntil(queueBackgroundSync(payload));
      break;
    case 'GET_METRICS':
      event.ports[0]?.postMessage(performanceMetrics);
      break;
    case 'CLEAR_CACHE':
      event.waitUntil(clearCache(payload.cacheName));
      break;
    default:
      console.warn('[SW] Unknown message type:', type);
  }
});

/**
 * Precache essential assets
 */
async function precacheAssets(): Promise<void> {
  try {
    const cache = await caches.open(CACHE_CONFIG.APP_CACHE);
    await cache.addAll(PRECACHE_ASSETS);
    console.log('[SW] Precached assets successfully');
  } catch (error) {
    console.error('[SW] Failed to precache assets:', error);
  }
}

/**
 * Setup initial caches
 */
async function setupCaches(): Promise<void> {
  const cacheNames = Object.values(CACHE_CONFIG).filter(name => typeof name === 'string');
  
  await Promise.all(
    cacheNames.map(cacheName => caches.open(cacheName))
  );
  
  console.log('[SW] Caches initialized');
}

/**
 * Clean up old caches
 */
async function cleanupOldCaches(): Promise<void> {
  const cacheNames = await caches.keys();
  const currentCaches = new Set(Object.values(CACHE_CONFIG));
  
  const deletePromises = cacheNames
    .filter(name => !currentCaches.has(name))
    .map(name => caches.delete(name));
  
  await Promise.all(deletePromises);
  console.log('[SW] Old caches cleaned up');
}

/**
 * Find matching route configuration
 */
function findMatchingRoute(url: string): any {
  return ROUTES_CONFIG.find(route => route.pattern.test(url));
}

/**
 * Handle fetch requests based on route configuration
 */
async function handleRequest(request: Request, route: any): Promise<Response> {
  performanceMetrics.networkRequests++;
  
  switch (route.strategy) {
    case CACHE_STRATEGIES.CACHE_FIRST:
      return handleCacheFirst(request, route);
    case CACHE_STRATEGIES.NETWORK_FIRST:
      return handleNetworkFirst(request, route);
    case CACHE_STRATEGIES.STALE_WHILE_REVALIDATE:
      return handleStaleWhileRevalidate(request, route);
    case CACHE_STRATEGIES.NETWORK_ONLY:
      return handleNetworkOnly(request, route);
    case CACHE_STRATEGIES.CACHE_ONLY:
      return handleCacheOnly(request, route);
    default:
      return fetch(request);
  }
}

/**
 * Cache First Strategy
 */
async function handleCacheFirst(request: Request, route: any): Promise<Response> {
  const cache = await caches.open(route.cache);
  const cachedResponse = await cache.match(request);
  
  if (cachedResponse && !isExpired(cachedResponse, route.maxAge)) {
    performanceMetrics.cacheHits++;
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetchWithTimeout(request, route.networkTimeout);
    
    if (networkResponse.ok) {
      cache.put(request.clone(), networkResponse.clone());
    }
    
    performanceMetrics.cacheMisses++;
    return networkResponse;
  } catch (error) {
    if (cachedResponse) {
      performanceMetrics.cacheHits++;
      return cachedResponse;
    }
    throw error;
  }
}

/**
 * Network First Strategy
 */
async function handleNetworkFirst(request: Request, route: any): Promise<Response> {
  try {
    const networkResponse = await fetchWithTimeout(request, route.networkTimeout);
    
    if (networkResponse.ok && route.cache) {
      const cache = await caches.open(route.cache);
      cache.put(request.clone(), networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    if (route.cache) {
      const cache = await caches.open(route.cache);
      const cachedResponse = await cache.match(request);
      
      if (cachedResponse) {
        performanceMetrics.cacheHits++;
        return cachedResponse;
      }
    }
    
    // Return offline fallback for API requests
    if (request.url.includes('/api/')) {
      return createOfflineApiResponse(request);
    }
    
    throw error;
  }
}

/**
 * Stale While Revalidate Strategy
 */
async function handleStaleWhileRevalidate(request: Request, route: any): Promise<Response> {
  const cache = await caches.open(route.cache);
  const cachedResponse = await cache.match(request);
  
  // Always try to fetch fresh data in the background
  const fetchPromise = fetchWithTimeout(request, route.networkTimeout)
    .then(response => {
      if (response.ok) {
        cache.put(request.clone(), response.clone());
      }
      return response;
    })
    .catch(error => {
      console.warn('[SW] Background fetch failed:', error);
      return null;
    });
  
  // Return cached response immediately if available and not too stale
  if (cachedResponse && !isStale(cachedResponse, route.maxAge)) {
    performanceMetrics.cacheHits++;
    // Don't await the fetch promise - let it update cache in background
    fetchPromise;
    return cachedResponse;
  }
  
  // Wait for network response if no cache or cache is stale
  try {
    const networkResponse = await fetchPromise;
    if (networkResponse) {
      return networkResponse;
    }
  } catch (error) {
    // Network failed, fall back to cache if available
  }
  
  if (cachedResponse) {
    performanceMetrics.cacheHits++;
    return cachedResponse;
  }
  
  // No cache and network failed - return offline response
  if (request.url.includes('/api/')) {
    return createOfflineApiResponse(request);
  }
  
  throw new Error('No cache available and network failed');
}

/**
 * Network Only Strategy
 */
async function handleNetworkOnly(request: Request, route: any): Promise<Response> {
  try {
    return await fetchWithTimeout(request, route.networkTimeout);
  } catch (error) {
    // For critical API calls, queue for background sync
    if (request.url.includes('/api/bot/commands')) {
      await queueBackgroundSync({
        url: request.url,
        method: request.method,
        headers: Object.fromEntries(request.headers.entries()),
        body: request.method !== 'GET' ? await request.text() : null
      });
    }
    
    throw error;
  }
}

/**
 * Cache Only Strategy
 */
async function handleCacheOnly(request: Request, route: any): Promise<Response> {
  const cache = await caches.open(route.cache);
  const cachedResponse = await cache.match(request);
  
  if (cachedResponse) {
    performanceMetrics.cacheHits++;
    return cachedResponse;
  }
  
  performanceMetrics.cacheMisses++;
  throw new Error('Resource not available in cache');
}

/**
 * Fetch with timeout
 */
async function fetchWithTimeout(request: Request, timeout: number): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(request, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Check if cached response is expired
 */
function isExpired(response: Response, maxAge: number): boolean {
  const cachedTime = response.headers.get('sw-cached-time');
  if (!cachedTime) return true;
  
  const age = Date.now() - parseInt(cachedTime);
  return age > maxAge;
}

/**
 * Check if cached response is stale
 */
function isStale(response: Response, maxAge: number): boolean {
  const cachedTime = response.headers.get('sw-cached-time');
  if (!cachedTime) return true;
  
  const age = Date.now() - parseInt(cachedTime);
  return age > maxAge;
}

/**
 * Create offline API response
 */
function createOfflineApiResponse(request: Request): Response {
  const url = new URL(request.url);
  const pathSegments = url.pathname.split('/');
  const endpoint = pathSegments[pathSegments.length - 1];
  
  let offlineData: any = {
    error: 'Offline',
    message: 'This data is not available offline',
    timestamp: new Date().toISOString()
  };
  
  // Provide some fallback data for critical endpoints
  switch (endpoint) {
    case 'market-data':
      offlineData = {
        price: 0,
        change_24h: 0,
        volume: 0,
        timestamp: new Date().toISOString(),
        offline: true
      };
      break;
    case 'positions':
      offlineData = {
        positions: [],
        total_value: 0,
        offline: true
      };
      break;
    case 'risk':
      offlineData = {
        total_exposure: 0,
        max_drawdown: 0,
        daily_pnl: 0,
        offline: true
      };
      break;
  }
  
  return new Response(JSON.stringify(offlineData), {
    status: 200,
    statusText: 'OK (Offline)',
    headers: {
      'Content-Type': 'application/json',
      'SW-Offline': 'true'
    }
  });
}

/**
 * Setup background sync
 */
async function setupBackgroundSync(): Promise<void> {
  try {
    await self.registration.sync.register(SYNC_CONFIG.QUEUE_NAME);
    console.log('[SW] Background sync registered');
  } catch (error) {
    console.warn('[SW] Background sync not supported:', error);
  }
}

/**
 * Queue item for background sync
 */
async function queueBackgroundSync(data: any): Promise<void> {
  backgroundSyncQueue.push({
    id: Date.now().toString(),
    data,
    timestamp: Date.now(),
    attempts: 0
  });
  
  try {
    await self.registration.sync.register(SYNC_CONFIG.QUEUE_NAME);
  } catch (error) {
    console.warn('[SW] Failed to register background sync:', error);
  }
}

/**
 * Process background sync queue
 */
async function processBackgroundSync(): Promise<void> {
  console.log('[SW] Processing background sync queue:', backgroundSyncQueue.length);
  performanceMetrics.backgroundSyncs++;
  
  const batch = backgroundSyncQueue.splice(0, SYNC_CONFIG.BATCH_SIZE);
  
  for (const item of batch) {
    try {
      await processBackgroundSyncItem(item);
    } catch (error) {
      console.error('[SW] Failed to process sync item:', error);
      
      item.attempts++;
      if (item.attempts < SYNC_CONFIG.MAX_RETRY_ATTEMPTS) {
        // Re-queue for retry
        setTimeout(() => {
          backgroundSyncQueue.push(item);
        }, SYNC_CONFIG.RETRY_DELAY * item.attempts);
      }
    }
  }
  
  // Continue processing if there are more items
  if (backgroundSyncQueue.length > 0) {
    setTimeout(() => {
      processBackgroundSync();
    }, 1000);
  }
}

/**
 * Process individual background sync item
 */
async function processBackgroundSyncItem(item: any): Promise<void> {
  const { data } = item;
  
  const request = new Request(data.url, {
    method: data.method,
    headers: data.headers,
    body: data.body
  });
  
  const response = await fetch(request);
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  console.log('[SW] Background sync item processed successfully');
}

/**
 * Handle push notifications
 */
async function handlePushNotification(data: any): Promise<void> {
  console.log('[SW] Handling push notification:', data);
  performanceMetrics.pushNotifications++;
  
  const options: NotificationOptions = {
    body: data.body || 'Trading Bot Alert',
    icon: data.icon || NOTIFICATION_CONFIG.DEFAULT_ICON,
    badge: data.badge || NOTIFICATION_CONFIG.DEFAULT_BADGE,
    tag: data.tag || 'trading-alert',
    data: data.data || {},
    requireInteraction: data.priority === 'critical',
    vibrate: data.priority === 'critical' ? [200, 100, 200] : [100],
    actions: data.actions || []
  };
  
  // Add to notification queue for management
  notificationQueue.push({
    title: data.title || 'Trading Bot',
    options,
    timestamp: Date.now()
  });
  
  // Clean up old notifications
  if (notificationQueue.length > NOTIFICATION_CONFIG.MAX_NOTIFICATIONS) {
    notificationQueue.shift();
  }
  
  await self.registration.showNotification(
    data.title || 'Trading Bot',
    options
  );
  
  // Auto-close non-critical notifications
  if (data.priority !== 'critical') {
    setTimeout(async () => {
      const notifications = await self.registration.getNotifications({
        tag: options.tag
      });
      
      notifications.forEach(notification => {
        const age = Date.now() - (data.timestamp || Date.now());
        if (age > NOTIFICATION_CONFIG.AUTO_CLOSE_DELAY) {
          notification.close();
        }
      });
    }, NOTIFICATION_CONFIG.AUTO_CLOSE_DELAY);
  }
}

/**
 * Handle notification clicks
 */
async function handleNotificationClick(event: NotificationEvent): Promise<void> {
  const notification = event.notification;
  const action = event.action;
  const data = notification.data;
  
  console.log('[SW] Notification clicked:', { action, data });
  
  let url = '/';
  
  if (action) {
    // Handle specific action
    switch (action) {
      case 'view_trade':
        url = `/trade?id=${data.tradeId}`;
        break;
      case 'view_risk':
        url = `/risk`;
        break;
      case 'view_analytics':
        url = `/analytics`;
        break;
      default:
        url = data.url || '/';
    }
  } else {
    // Default click action
    url = data.url || '/dashboard';
  }
  
  // Open or focus the app
  const clients = await self.clients.matchAll({
    type: 'window',
    includeUncontrolled: true
  });
  
  // Check if app is already open
  for (const client of clients) {
    if (client.url.includes(self.location.origin)) {
      client.focus();
      client.postMessage({
        type: 'NOTIFICATION_CLICK',
        action,
        data,
        url
      });
      return;
    }
  }
  
  // Open new window
  await self.clients.openWindow(url);
}

/**
 * Update cache with new data
 */
async function updateCache(payload: any): Promise<void> {
  const { cacheName, url, data } = payload;
  
  if (!cacheName || !url || !data) return;
  
  const cache = await caches.open(cacheName);
  const response = new Response(JSON.stringify(data), {
    headers: {
      'Content-Type': 'application/json',
      'sw-cached-time': Date.now().toString()
    }
  });
  
  await cache.put(url, response);
  console.log('[SW] Cache updated:', { cacheName, url });
}

/**
 * Clear specific cache
 */
async function clearCache(cacheName: string): Promise<void> {
  if (cacheName) {
    await caches.delete(cacheName);
    console.log('[SW] Cache cleared:', cacheName);
  } else {
    // Clear all caches
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map(name => caches.delete(name)));
    console.log('[SW] All caches cleared');
  }
}

/**
 * Periodic cache cleanup
 */
async function performPeriodicCleanup(): Promise<void> {
  console.log('[SW] Performing periodic cleanup');
  
  const cacheNames = await caches.keys();
  
  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();
    
    for (const request of requests) {
      const response = await cache.match(request);
      if (response && isExpired(response, CACHE_CONFIG.MAX_AGE_SECONDS * 1000)) {
        await cache.delete(request);
        console.log('[SW] Expired cache entry removed:', request.url);
      }
    }
  }
}

// Schedule periodic cleanup
setInterval(performPeriodicCleanup, 60 * 60 * 1000); // Every hour

console.log('[SW] Service worker script loaded');

export {};