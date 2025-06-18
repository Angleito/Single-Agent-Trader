/**
 * Test setup file for Vitest
 */
import { vi } from 'vitest'

// Mock browser APIs for testing
global.navigator = {
  ...global.navigator,
  onLine: true,
  userAgent: 'test-agent',
  language: 'en-US',
  languages: ['en-US'],
  platform: 'test',
  hardwareConcurrency: 4,
  deviceMemory: 8,
  serviceWorker: {
    ready: Promise.resolve({
      sync: {
        register: vi.fn(),
      },
    }),
  },
} as Partial<Navigator>

global.window = {
  ...global.window,
  location: {
    href: 'http://localhost:3000',
    protocol: 'http:',
    host: 'localhost:3000',
    hostname: 'localhost',
  },
  crypto: {
    getRandomValues: (arr: Uint8Array) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256)
      }
      return arr
    },
    subtle: {
      generateKey: vi.fn(),
      encrypt: vi.fn(),
      decrypt: vi.fn(),
      digest: vi.fn(),
    },
  },
  performance: {
    now: () => Date.now(),
    memory: {
      usedJSHeapSize: 10000000,
      totalJSHeapSize: 20000000,
      jsHeapSizeLimit: 100000000,
    },
  },
  Notification: {
    permission: 'granted',
    requestPermission: vi.fn().mockResolvedValue('granted'),
  },
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
} as Partial<Window> & {
  location: {
    href: string
    protocol: string
    host: string
    hostname: string
  }
  crypto: {
    getRandomValues: (arr: Uint8Array) => Uint8Array
    subtle: {
      generateKey: () => void
      encrypt: () => void
      decrypt: () => void
      digest: () => void
    }
  }
  performance: {
    now: () => number
    memory: {
      usedJSHeapSize: number
      totalJSHeapSize: number
      jsHeapSizeLimit: number
    }
  }
  Notification: {
    permission: NotificationPermission
    requestPermission: () => Promise<NotificationPermission>
  }
  addEventListener: (type: string, listener: EventListener) => void
  removeEventListener: (type: string, listener: EventListener) => void
}

// Mock IndexedDB
global.indexedDB = {
  open: vi.fn(),
  deleteDatabase: vi.fn(),
} as Partial<IDBFactory>

// Mock WebSocket
global.WebSocket = vi.fn(() => ({
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: 1,
})) as unknown as typeof WebSocket

// Mock fetch
global.fetch = vi.fn()

// Mock ResizeObserver
global.ResizeObserver = vi.fn(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
  root: null,
  rootMargin: '0px',
  thresholds: [0],
  takeRecords: vi.fn().mockReturnValue([]),
})) as unknown as typeof IntersectionObserver

// Mock PerformanceObserver
global.PerformanceObserver = vi.fn(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
})) as unknown as typeof PerformanceObserver

// Add supportedEntryTypes as a static property
;(global.PerformanceObserver as unknown as { supportedEntryTypes: string[] }).supportedEntryTypes =
  ['measure', 'mark', 'resource', 'navigation']

// Mock MutationObserver
global.MutationObserver = vi.fn(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn().mockReturnValue([]),
})) as unknown as typeof MutationObserver
