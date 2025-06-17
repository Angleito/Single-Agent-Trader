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
} as any

global.window = {
  ...global.window,
  location: {
    href: 'http://localhost:3000',
    protocol: 'http:',
    host: 'localhost:3000',
    hostname: 'localhost',
  },
  crypto: {
    getRandomValues: (arr: any) => {
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
} as any

// Mock IndexedDB
global.indexedDB = {
  open: vi.fn(),
  deleteDatabase: vi.fn(),
} as any

// Mock WebSocket
global.WebSocket = vi.fn(() => ({
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: 1,
})) as any

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
})) as any

// Mock PerformanceObserver
global.PerformanceObserver = vi.fn(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
})) as any

// Add supportedEntryTypes as a static property
;(global.PerformanceObserver as any).supportedEntryTypes = [
  'measure',
  'mark',
  'resource',
  'navigation',
]

// Mock MutationObserver
global.MutationObserver = vi.fn(() => ({
  observe: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn().mockReturnValue([]),
})) as any
