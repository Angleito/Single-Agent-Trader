// Global type definitions for the AI Trading Bot Dashboard

// TradingView Charting Library types
declare global {
  interface Window {
    TradingView: any;
    tradingViewLoaded?: boolean;
    tradingViewError?: boolean;
  }
}

// Trading Bot Data Types
export interface BotStatus {
  status: 'running' | 'stopped' | 'error' | 'dry_run';
  uptime: number;
  last_update: string;
  dry_run: boolean;
  symbol: string;
  leverage: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  timestamp: string;
  volume?: number;
  change_24h?: number;
  change_percent_24h?: number;
}

export interface TradeAction {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  timestamp: string;
  price?: number;
  quantity?: number;
  leverage?: number;
}

export interface VuManchuIndicators {
  cipher_a: number | null;
  cipher_b: number | null;
  wave_trend_1: number | null;
  wave_trend_2: number | null;
  timestamp: string;
}

export interface Position {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
  leverage: number;
  liquidation_price?: number;
  timestamp: string;
}

export interface RiskMetrics {
  total_portfolio_value: number;
  available_balance: number;
  margin_used: number;
  margin_available: number;
  total_pnl: number;
  daily_pnl: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: 'bot_status' | 'market_data' | 'trade_action' | 'indicators' | 'position' | 'risk_metrics' | 'trading_loop' | 'ai_decision' | 'system_status' | 'error';
  data: BotStatus | MarketData | TradeAction | VuManchuIndicators | Position | RiskMetrics | any;
  timestamp: string;
}

// Dashboard UI State
export interface DashboardState {
  bot_status: BotStatus | null;
  market_data: MarketData | null;
  latest_action: TradeAction | null;
  indicators: VuManchuIndicators | null;
  positions: Position[];
  risk_metrics: RiskMetrics | null;
  connection_status: ConnectionStatus;
  error_message: string | null;
}

// TradingView Chart Configuration
export interface ChartConfig {
  container_id: string;
  symbol: string;
  interval: string;
  library_path: string;
  charts_storage_url?: string;
  charts_storage_api_version?: string;
  client_id?: string;
  user_id?: string;
  fullscreen?: boolean;
  autosize?: boolean;
  studies_overrides?: Record<string, any>;
  theme?: 'light' | 'dark';
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// Configuration Types
export interface DashboardConfig {
  websocket_url: string;
  api_base_url: string;
  default_symbol: string;
  refresh_interval: number;
  chart_config: ChartConfig;
}

// Utility Types
export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error';
export type LogLevel = 'info' | 'warn' | 'error' | 'debug';

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: string;
  component?: string | undefined;
}

// Note: All types are already exported above, no need to re-export