/// <reference types="vite/client" />

// Global development flag
declare const __DEV__: boolean;

interface ImportMetaEnv {
  readonly VITE_APP_TITLE: string
  readonly VITE_API_BASE_URL: string
  readonly VITE_API_URL: string
  readonly VITE_WS_URL: string
  readonly VITE_DEFAULT_SYMBOL: string
  readonly VITE_REFRESH_INTERVAL: string
  readonly VITE_LOG_LEVEL: string
  readonly VITE_TRADINGVIEW_LIBRARY_PATH: string
  readonly VITE_TRADINGVIEW_THEME: string
  readonly VITE_TRADINGVIEW_AUTOSIZE: string
  readonly VITE_ENVIRONMENT: string
  readonly VITE_DEBUG: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
