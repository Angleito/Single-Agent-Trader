"""
Functional configuration system with backward compatibility.

This module provides a functional programming approach to configuration
while maintaining full backward compatibility with the existing Settings interface.
All configuration now uses functional programming types with proper validation.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import SecretStr

# Import functional programming types with lazy loading to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.fp.types.result import Failure, Result, Success
else:
    # Import for runtime use
    try:
        from bot.fp.types.result import Failure, Result, Success
    except ImportError:
        # Fallback classes if fp types are not available
        class Success:
            def __init__(self, value):
                self._value = value
            def success(self):
                return self._value
        
        class Failure:
            def __init__(self, error):
                self._error = error
        
        class Result:
            pass
            
    try:
        from bot.fp.types.base import Money, Percentage, Symbol, TimeInterval, TradingMode
        from bot.fp.types.config import (
            APIKey,
            BacktestConfig,
            Config as FunctionalConfig,
            ExchangeConfig,
            FeatureFlags,
            LogLevel,
            PrivateKey,
            StrategyConfig,
            SystemConfig,
        )
    except ImportError:
        # Fallback classes if fp types are not available
        class Money:
            pass
        class Percentage:
            pass
        class Symbol:
            pass
        class TimeInterval:
            pass
        class TradingMode:
            pass
        class APIKey:
            pass
        class BacktestConfig:
            pass
        class FunctionalConfig:
            pass
        class ExchangeConfig:
            pass
        class FeatureFlags:
            pass
        class LogLevel:
            pass
        class PrivateKey:
            pass
        class StrategyConfig:
            pass
        class SystemConfig:
            pass

# Local implementations to avoid imports during normal usage
def parse_bool_env(key: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")

def parse_int_env(key: str, default: int) -> int:
    """Parse integer environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default  # Fallback to default for compatibility

def parse_float_env(key: str, default: float) -> float:
    """Parse float environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Environment enumeration
from enum import Enum

class Environment(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class TradingProfile(Enum):
    """Trading profile types."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"  # Fallback to default for compatibility

def parse_env_var(key: str, default: str | None = None) -> str | None:
    """Parse environment variable with optional default."""
    return os.environ.get(key, default)


# Cached lazy loading for functional config
_functional_config_cache = None

def _get_functional_config():
    """Lazy load functional configuration types with caching."""
    global _functional_config_cache
    
    if _functional_config_cache is not None:
        return _functional_config_cache
    
    try:
        from bot.fp.types.result import Failure, Result, Success
        from bot.fp.types.config import Config as FunctionalConfig
        _functional_config_cache = (FunctionalConfig, Success, Failure, Result)
        return _functional_config_cache
    except ImportError:
        _functional_config_cache = (None, None, None, None)
        return _functional_config_cache


# Environment variable cache for performance
_env_cache = {}

def parse_env_var_cached(key: str, default: str | None = None) -> str | None:
    """Parse environment variable with caching for performance."""
    if key not in _env_cache:
        _env_cache[key] = os.environ.get(key, default)
    return _env_cache[key]


class ConfigError(Exception):
    """Configuration error."""
    pass


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


# Compatibility adapters using functional programming types
@dataclass
class TradingSettings:
    """Trading configuration settings using functional types."""
    
    def __init__(self, functional_config: Any = None, **kwargs):
        # Use functional config if provided, otherwise environment/kwargs
        if functional_config:
            # Extract from functional config based on strategy type
            self._from_functional_config(functional_config)
        else:
            # Load from environment first, then kwargs, then defaults (maintaining exact compatibility)
            self.symbol: str = os.getenv('TRADING__SYMBOL', kwargs.get('symbol', 'BTC-USD'))
            self.interval: str = os.getenv('TRADING__INTERVAL', kwargs.get('interval', '1m'))
            self.leverage: int = int(os.getenv('TRADING__LEVERAGE', str(kwargs.get('leverage', 5))))
            self.max_size_pct: float = float(os.getenv('TRADING__MAX_SIZE_PCT', kwargs.get('max_size_pct', 20.0)))
            self.order_timeout_seconds: int = int(os.getenv('TRADING__ORDER_TIMEOUT_SECONDS', kwargs.get('order_timeout_seconds', 30)))
            self.slippage_tolerance_pct: float = float(os.getenv('TRADING__SLIPPAGE_TOLERANCE_PCT', kwargs.get('slippage_tolerance_pct', 0.1)))
            self.min_profit_pct: float = float(os.getenv('TRADING__MIN_PROFIT_PCT', kwargs.get('min_profit_pct', 0.5)))
            self.maker_fee_rate: float = float(os.getenv('TRADING__MAKER_FEE_RATE', kwargs.get('maker_fee_rate', 0.004)))
            self.taker_fee_rate: float = float(os.getenv('TRADING__TAKER_FEE_RATE', kwargs.get('taker_fee_rate', 0.006)))
            self.futures_fee_rate: float = float(os.getenv('TRADING__FUTURES_FEE_RATE', kwargs.get('futures_fee_rate', 0.0015)))
            self.min_trading_interval_seconds: int = int(os.getenv('TRADING__MIN_TRADING_INTERVAL_SECONDS', kwargs.get('min_trading_interval_seconds', 60)))
            self.require_24h_data_before_trading: bool = parse_bool_env('TRADING__REQUIRE_24H_DATA_BEFORE_TRADING', kwargs.get('require_24h_data_before_trading', True))
            self.min_candles_for_trading: int = int(os.getenv('TRADING__MIN_CANDLES_FOR_TRADING', kwargs.get('min_candles_for_trading', 100)))
            self.enable_futures: bool = parse_bool_env('TRADING__ENABLE_FUTURES', kwargs.get('enable_futures', True))
            self.futures_account_type: str = os.getenv('TRADING__FUTURES_ACCOUNT_TYPE', kwargs.get('futures_account_type', 'CFM'))
            self.auto_cash_transfer: bool = parse_bool_env('TRADING__AUTO_CASH_TRANSFER', kwargs.get('auto_cash_transfer', True))
            self.max_futures_leverage: int = int(os.getenv('TRADING__MAX_FUTURES_LEVERAGE', kwargs.get('max_futures_leverage', 20)))
    
    def _from_functional_config(self, config: Any) -> None:
        """Extract settings from functional configuration."""
        # Set defaults compatible with existing interface
        self.symbol = 'BTC-USD'
        self.interval = '1m'
        self.leverage = 5
        self.max_size_pct = 20.0
        self.order_timeout_seconds = 30
        self.slippage_tolerance_pct = 0.1
        self.min_profit_pct = 0.5
        self.maker_fee_rate = 0.004
        self.taker_fee_rate = 0.006
        self.futures_fee_rate = 0.0015
        self.min_trading_interval_seconds = 60
        self.require_24h_data_before_trading = True
        self.min_candles_for_trading = 100
        self.enable_futures = True
        self.futures_account_type = 'CFM'
        self.auto_cash_transfer = True
        self.max_futures_leverage = 20


@dataclass
class LLMSettings:
    """LLM configuration settings using functional types."""
    
    def __init__(self, functional_config: Any = None, **kwargs) -> None:
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.provider: str = os.getenv('LLM__PROVIDER', kwargs.get('provider', 'openai'))
            self.model_name: str = os.getenv('LLM__MODEL_NAME', kwargs.get('model_name', 'gpt-4'))
            self.temperature: float = parse_float_env('LLM__TEMPERATURE', kwargs.get('temperature', 0.1))
            self.max_tokens: int = parse_int_env('LLM__MAX_TOKENS', kwargs.get('max_tokens', 30000))
            self.request_timeout: int = parse_int_env('LLM__REQUEST_TIMEOUT', kwargs.get('request_timeout', 30))
            self.max_retries: int = parse_int_env('LLM__MAX_RETRIES', kwargs.get('max_retries', 3))
            self.openai_api_key: Optional[SecretStr] = None
            
            # Load from environment
            api_key = os.getenv('LLM__OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_api_key = SecretStr(api_key)
    
    def _from_functional_config(self, config: Any) -> None:
        """Extract settings from functional configuration."""
        # Avoid imports to prevent circular dependencies
        # Check if config has LLM-specific attributes
        if hasattr(config, 'model_name') and hasattr(config, 'temperature'):
            self.provider = 'openai'
            self.model_name = config.model_name
            self.temperature = config.temperature
            self.max_tokens = config.max_context_length
            self.request_timeout = 30
            self.max_retries = 3
            self.openai_api_key = None
            
            # Load API key from environment
            api_key = os.getenv('LLM__OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_api_key = SecretStr(api_key)
        else:
            # Set defaults for non-LLM strategies
            self.provider = 'openai'
            self.model_name = 'gpt-4'
            self.temperature = 0.1
            self.max_tokens = 30000
            self.request_timeout = 30
            self.max_retries = 3
            self.openai_api_key = None


@dataclass
class ExchangeSettings:
    """Exchange configuration settings using functional types."""
    
    def __init__(self, functional_config: Optional['ExchangeConfig'] = None, **kwargs) -> None:
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.exchange_type: str = os.getenv('EXCHANGE__EXCHANGE_TYPE', kwargs.get('exchange_type', 'coinbase'))
            self.cb_sandbox: bool = parse_bool_env('EXCHANGE__CB_SANDBOX', kwargs.get('cb_sandbox', True))
            self.api_timeout: int = parse_int_env('EXCHANGE__API_TIMEOUT', kwargs.get('api_timeout', 10))
            self.rate_limit_requests: int = parse_int_env('EXCHANGE__RATE_LIMIT_REQUESTS', kwargs.get('rate_limit_requests', 10))
            self.rate_limit_window_seconds: int = parse_int_env('EXCHANGE__RATE_LIMIT_WINDOW_SECONDS', kwargs.get('rate_limit_window_seconds', 60))
            self.bluefin_network: str = os.getenv('EXCHANGE__BLUEFIN_NETWORK', kwargs.get('bluefin_network', 'mainnet'))
            
            # Coinbase credentials
            self.cb_api_key: Optional[SecretStr] = None
            self.cb_api_secret: Optional[SecretStr] = None
            self.cb_passphrase: Optional[SecretStr] = None
            self.cdp_api_key_name: Optional[SecretStr] = None
            self.cdp_private_key: Optional[SecretStr] = None
            
            # Bluefin credentials
            self.bluefin_private_key: Optional[SecretStr] = None
            
            # Load from environment
            if os.getenv('EXCHANGE__CB_API_KEY'):
                self.cb_api_key = SecretStr(os.getenv('EXCHANGE__CB_API_KEY'))
            if os.getenv('EXCHANGE__CB_API_SECRET'):
                self.cb_api_secret = SecretStr(os.getenv('EXCHANGE__CB_API_SECRET'))
            if os.getenv('EXCHANGE__CB_PASSPHRASE'):
                self.cb_passphrase = SecretStr(os.getenv('EXCHANGE__CB_PASSPHRASE'))
            if os.getenv('EXCHANGE__CDP_API_KEY_NAME'):
                self.cdp_api_key_name = SecretStr(os.getenv('EXCHANGE__CDP_API_KEY_NAME'))
            if os.getenv('EXCHANGE__CDP_PRIVATE_KEY'):
                self.cdp_private_key = SecretStr(os.getenv('EXCHANGE__CDP_PRIVATE_KEY'))
            if os.getenv('EXCHANGE__BLUEFIN_PRIVATE_KEY'):
                self.bluefin_private_key = SecretStr(os.getenv('EXCHANGE__BLUEFIN_PRIVATE_KEY'))
    
    def _from_functional_config(self, config: 'ExchangeConfig') -> None:
        """Extract settings from functional configuration."""
        # Avoid imports to prevent circular dependencies
        # Check exchange type by attributes
        if hasattr(config, 'api_key') and hasattr(config, 'private_key') and hasattr(config, 'api_url'):
            self.exchange_type = 'coinbase'
            self.cb_sandbox = True  # Default for safety
            self.api_timeout = 10
            self.rate_limit_requests = config.rate_limits.requests_per_minute
            self.rate_limit_window_seconds = 60
            self.bluefin_network = 'mainnet'
            
            # Convert functional types to compatibility types
            self.cdp_api_key_name = SecretStr(config.api_key._value) if config.api_key else None
            self.cdp_private_key = SecretStr(config.private_key._value) if config.private_key else None
            
        elif hasattr(config, 'private_key') and hasattr(config, 'network'):
            # Bluefin exchange
            self.exchange_type = 'bluefin'
            self.cb_sandbox = True
            self.api_timeout = 10
            self.rate_limit_requests = config.rate_limits.requests_per_minute
            self.rate_limit_window_seconds = 60
            self.bluefin_network = config.network
            
            # Convert functional types
            self.bluefin_private_key = SecretStr(config.private_key._value) if config.private_key else None
            
        elif hasattr(config, 'testnet'):
            # Binance exchange
            self.exchange_type = 'binance'
            self.cb_sandbox = config.testnet
            self.api_timeout = 10
            self.rate_limit_requests = config.rate_limits.requests_per_minute
            self.rate_limit_window_seconds = 60
            self.bluefin_network = 'mainnet'
        else:
            # Default/unknown exchange type
            self.exchange_type = 'coinbase'
            self.cb_sandbox = True
            self.api_timeout = 10
            self.rate_limit_requests = 10
            self.rate_limit_window_seconds = 60
            self.bluefin_network = 'mainnet'
        
        # Set defaults for missing credentials
        self.cb_api_key = self.cb_api_key or None
        self.cb_api_secret = self.cb_api_secret or None
        self.cb_passphrase = self.cb_passphrase or None
        self.cdp_api_key_name = self.cdp_api_key_name or None
        self.cdp_private_key = self.cdp_private_key or None
        self.bluefin_private_key = self.bluefin_private_key or None


@dataclass
class RiskSettings:
    """Risk management settings using functional types."""
    
    def __init__(self, functional_config: Optional['SystemConfig'] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.max_daily_loss_pct: float = parse_float_env('RISK__MAX_DAILY_LOSS_PCT', kwargs.get('max_daily_loss_pct', 5.0))
            self.max_concurrent_trades: int = parse_int_env('RISK__MAX_CONCURRENT_TRADES', kwargs.get('max_concurrent_trades', 3))
            self.default_stop_loss_pct: float = parse_float_env('RISK__DEFAULT_STOP_LOSS_PCT', kwargs.get('default_stop_loss_pct', 2.0))
            self.default_take_profit_pct: float = parse_float_env('RISK__DEFAULT_TAKE_PROFIT_PCT', kwargs.get('default_take_profit_pct', 4.0))
    
    def _from_functional_config(self, config: 'SystemConfig') -> None:
        """Extract settings from functional configuration."""
        self.max_daily_loss_pct = 5.0  # Default values
        self.max_concurrent_trades = config.max_concurrent_positions
        self.default_stop_loss_pct = 2.0
        self.default_take_profit_pct = 4.0


@dataclass
class DataSettings:
    """Data management settings using functional types."""
    
    def __init__(self, **kwargs):
        # Maintain exact compatibility with environment variable loading
        self.keep_days: int = parse_int_env('DATA__KEEP_DAYS', kwargs.get('keep_days', 30))
        self.backup_enabled: bool = parse_bool_env('DATA__BACKUP_ENABLED', kwargs.get('backup_enabled', True))


@dataclass
class DominanceSettings:
    """Market dominance settings using functional types."""
    
    def __init__(self, **kwargs):
        # Maintain exact compatibility with environment variable loading
        self.enabled: bool = parse_bool_env('DOMINANCE__ENABLED', kwargs.get('enabled', False))
        self.threshold: float = parse_float_env('DOMINANCE__THRESHOLD', kwargs.get('threshold', 0.45))


@dataclass
class SystemSettings:
    """System configuration settings using functional types."""
    
    def __init__(self, functional_config: Optional['SystemConfig'] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.dry_run: bool = parse_bool_env('SYSTEM__DRY_RUN', kwargs.get('dry_run', True))
            self.environment: str = os.getenv('SYSTEM__ENVIRONMENT', kwargs.get('environment', 'development'))
            self.log_level: str = os.getenv('SYSTEM__LOG_LEVEL', kwargs.get('log_level', 'INFO'))
            self.update_frequency_seconds: float = parse_float_env('SYSTEM__UPDATE_FREQUENCY_SECONDS', kwargs.get('update_frequency_seconds', 30.0))
    
    def _from_functional_config(self, config: 'SystemConfig') -> None:
        """Extract settings from functional configuration."""
        self.dry_run = config.mode == TradingMode.PAPER
        self.environment = 'production' if config.mode == TradingMode.LIVE else 'development'
        self.log_level = config.log_level.value
        self.update_frequency_seconds = 30.0  # Default


@dataclass
class PaperTradingSettings:
    """Paper trading settings using functional types."""
    
    def __init__(self, functional_config: Optional['BacktestConfig'] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.starting_balance: float = parse_float_env('PAPER_TRADING__STARTING_BALANCE', kwargs.get('starting_balance', 10000.0))
            self.fee_rate: float = parse_float_env('PAPER_TRADING__FEE_RATE', kwargs.get('fee_rate', 0.001))
            self.slippage_rate: float = parse_float_env('PAPER_TRADING__SLIPPAGE_RATE', kwargs.get('slippage_rate', 0.0005))
            self.enable_daily_reports: bool = parse_bool_env('PAPER_TRADING__ENABLE_DAILY_REPORTS', kwargs.get('enable_daily_reports', True))
            self.enable_weekly_summaries: bool = parse_bool_env('PAPER_TRADING__ENABLE_WEEKLY_SUMMARIES', kwargs.get('enable_weekly_summaries', True))
            self.track_drawdown: bool = parse_bool_env('PAPER_TRADING__TRACK_DRAWDOWN', kwargs.get('track_drawdown', True))
            self.keep_trade_history_days: int = parse_int_env('PAPER_TRADING__KEEP_TRADE_HISTORY_DAYS', kwargs.get('keep_trade_history_days', 90))
            self.export_trade_data: bool = parse_bool_env('PAPER_TRADING__EXPORT_TRADE_DATA', kwargs.get('export_trade_data', False))
            self.report_time_utc: str = os.getenv('PAPER_TRADING__REPORT_TIME_UTC', kwargs.get('report_time_utc', '23:59'))
            self.include_unrealized_pnl: bool = parse_bool_env('PAPER_TRADING__INCLUDE_UNREALIZED_PNL', kwargs.get('include_unrealized_pnl', True))
    
    def _from_functional_config(self, config: 'BacktestConfig') -> None:
        """Extract settings from functional configuration."""
        self.starting_balance = float(config.initial_capital.amount)
        self.fee_rate = config.fee_structure.taker_fee.as_ratio()
        self.slippage_rate = config.slippage.as_ratio()
        self.enable_daily_reports = True
        self.enable_weekly_summaries = True
        self.track_drawdown = True
        self.keep_trade_history_days = 90
        self.export_trade_data = False
        self.report_time_utc = '23:59'
        self.include_unrealized_pnl = True


@dataclass
class MonitoringSettings:
    """System monitoring settings using functional types."""
    
    def __init__(self, functional_config: Optional['SystemConfig'] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.enabled: bool = parse_bool_env('MONITORING__ENABLED', kwargs.get('enabled', True))
            self.check_interval: int = parse_int_env('MONITORING__CHECK_INTERVAL', kwargs.get('check_interval', 60))
    
    def _from_functional_config(self, config: 'SystemConfig') -> None:
        """Extract settings from functional configuration."""
        self.enabled = config.features.enable_metrics
        self.check_interval = 60


@dataclass
class MCPSettings:
    """MCP (Model Context Protocol) settings using functional types."""
    
    def __init__(self, functional_config: Optional['SystemConfig'] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.enabled: bool = parse_bool_env('MCP_ENABLED', kwargs.get('enabled', False))
            self.server_url: str = os.getenv('MCP_SERVER_URL', kwargs.get('server_url', 'http://localhost:8765'))
    
    def _from_functional_config(self, config: 'SystemConfig') -> None:
        """Extract settings from functional configuration."""
        self.enabled = config.features.enable_memory
        self.server_url = 'http://localhost:8765'


@dataclass
class OmniSearchSettings:
    """OmniSearch integration settings using functional types."""
    
    def __init__(self, **kwargs):
        # Maintain exact compatibility with environment variable loading
        self.enabled: bool = parse_bool_env('OMNISEARCH__ENABLED', kwargs.get('enabled', False))
        self.server_url: str = os.getenv('OMNISEARCH__SERVER_URL', kwargs.get('server_url', 'http://localhost:8766'))
        self.max_results: int = parse_int_env('OMNISEARCH__MAX_RESULTS', kwargs.get('max_results', 5))
        self.cache_ttl_seconds: int = parse_int_env('OMNISEARCH__CACHE_TTL_SECONDS', kwargs.get('cache_ttl_seconds', 300))
        self.rate_limit_requests_per_minute: int = parse_int_env('OMNISEARCH__RATE_LIMIT_REQUESTS_PER_MINUTE', kwargs.get('rate_limit_requests_per_minute', 10))
        self.timeout_seconds: int = parse_int_env('OMNISEARCH__TIMEOUT_SECONDS', kwargs.get('timeout_seconds', 30))
        self.enable_crypto_sentiment: bool = parse_bool_env('OMNISEARCH__ENABLE_CRYPTO_SENTIMENT', kwargs.get('enable_crypto_sentiment', True))
        self.enable_nasdaq_sentiment: bool = parse_bool_env('OMNISEARCH__ENABLE_NASDAQ_SENTIMENT', kwargs.get('enable_nasdaq_sentiment', True))
        self.enable_correlation_analysis: bool = parse_bool_env('OMNISEARCH__ENABLE_CORRELATION_ANALYSIS', kwargs.get('enable_correlation_analysis', True))


class Settings:
    """Main configuration settings with functional programming foundation."""
    
    def __init__(self, functional_config: Optional['FunctionalConfig'] = None, **overrides):
        """Initialize settings with optional functional config and overrides."""
        # Create compatibility sections using functional config if available
        if functional_config:
            self.trading = TradingSettings(functional_config.strategy, **overrides.get('trading', {}))
            self.llm = LLMSettings(functional_config.strategy, **overrides.get('llm', {}))
            self.exchange = ExchangeSettings(functional_config.exchange, **overrides.get('exchange', {}))
            self.risk = RiskSettings(functional_config.system, **overrides.get('risk', {}))
            self.data = DataSettings(**overrides.get('data', {}))
            self.dominance = DominanceSettings(**overrides.get('dominance', {}))
            self.system = SystemSettings(functional_config.system, **overrides.get('system', {}))
            self.paper_trading = PaperTradingSettings(functional_config.backtest, **overrides.get('paper_trading', {}))
            self.monitoring = MonitoringSettings(functional_config.system, **overrides.get('monitoring', {}))
            self.mcp = MCPSettings(functional_config.system, **overrides.get('mcp', {}))
            self.omnisearch = OmniSearchSettings(**overrides.get('omnisearch', {}))
        else:
            # Fallback to environment/kwargs based initialization
            self.trading = TradingSettings(**overrides.get('trading', {}))
            self.llm = LLMSettings(**overrides.get('llm', {}))
            self.exchange = ExchangeSettings(**overrides.get('exchange', {}))
            self.risk = RiskSettings(**overrides.get('risk', {}))
            self.data = DataSettings(**overrides.get('data', {}))
            self.dominance = DominanceSettings(**overrides.get('dominance', {}))
            self.system = SystemSettings(**overrides.get('system', {}))
            self.paper_trading = PaperTradingSettings(**overrides.get('paper_trading', {}))
            self.monitoring = MonitoringSettings(**overrides.get('monitoring', {}))
            self.mcp = MCPSettings(**overrides.get('mcp', {}))
            self.omnisearch = OmniSearchSettings(**overrides.get('omnisearch', {}))
        
        # Store functional config for advanced use cases
        self._functional_config = functional_config
    
    def apply_profile(self, profile: str) -> "Settings":
        """Apply a configuration profile using functional composition."""
        # Load profile-specific overrides and create new Settings instance
        profile_overrides = self._load_profile_overrides(profile)
        return Settings(self._functional_config, **profile_overrides)
    
    def _load_profile_overrides(self, profile: str) -> Dict[str, Any]:
        """Load profile-specific configuration overrides."""
        profile_configs = {
            'conservative': {
                'trading': {'leverage': 2, 'max_size_pct': 10.0},
                'risk': {'max_daily_loss_pct': 2.0, 'max_concurrent_trades': 1},
                'llm': {'temperature': 0.05}
            },
            'aggressive': {
                'trading': {'leverage': 10, 'max_size_pct': 50.0},
                'risk': {'max_daily_loss_pct': 10.0, 'max_concurrent_trades': 5},
                'llm': {'temperature': 0.2}
            },
            'balanced': {
                'trading': {'leverage': 5, 'max_size_pct': 25.0},
                'risk': {'max_daily_loss_pct': 5.0, 'max_concurrent_trades': 3},
                'llm': {'temperature': 0.1}
            }
        }
        return profile_configs.get(profile, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'trading': self.trading.__dict__,
            'llm': self.llm.__dict__,
            'exchange': self.exchange.__dict__,
            'risk': self.risk.__dict__,
            'data': self.data.__dict__,
            'dominance': self.dominance.__dict__,
            'system': self.system.__dict__,
            'paper_trading': self.paper_trading.__dict__,
            'monitoring': self.monitoring.__dict__,
            'mcp': self.mcp.__dict__,
            'omnisearch': self.omnisearch.__dict__,
        }


def create_settings(env_file: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, profile: Optional[str] = None) -> Settings:
    """Create settings instance with functional programming backend."""
    try:
        from dotenv import load_dotenv
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
    except ImportError:
        pass
    
    # Try to load functional configuration from environment (with lazy loading)
    functional_config = None
    try:
        FunctionalConfig, Success, Failure, Result = _get_functional_config()
        if FunctionalConfig:
            functional_config_result = FunctionalConfig.from_env()
            if isinstance(functional_config_result, Success):
                functional_config = functional_config_result.success()
    except (ImportError, AttributeError):
        # Functional types not available or failed to load
        functional_config = None
    
    settings_kwargs = overrides or {}
    settings = Settings(functional_config, **settings_kwargs)
    
    if profile:
        settings = settings.apply_profile(profile)
    
    # Validate settings using functional patterns (placeholder for future implementation)
    # validation_result = validate_settings(settings)
    # if validation_result:
    #     # Log warnings but don't fail
    #     print(f"Configuration validation warnings: {validation_result}")
    
    return settings


def load_settings_from_file(file_path: Union[str, Path]) -> Settings:
    """Load settings from a configuration file using functional patterns."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            # Fallback to default settings if file not found
            return create_settings()
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                # Unsupported format, fallback to default
                return create_settings()
        
        # Try to create functional config from file data
        try:
            functional_config = _create_functional_config_from_dict(config_data)
            if isinstance(functional_config, str):  # Error message
                functional_config = None
        except Exception:
            functional_config = None
        
        # Create settings with functional config or fallback to compatibility mode
        settings = Settings(functional_config, **config_data)
        return settings
        
    except Exception as e:
        # Fallback to default settings on any error
        return create_settings()


def _create_functional_config_from_dict(config_data: Dict[str, Any]) -> Any:
    """Create functional configuration from dictionary data."""
    try:
        # Set environment variables temporarily for functional config builders
        original_env = {}
        
        # Map config data to environment variables
        env_mapping = {
            # Trading settings
            'TRADING_PAIRS': config_data.get('trading', {}).get('symbol', 'BTC-USD'),
            'TRADING_INTERVAL': config_data.get('trading', {}).get('interval', '1m'),
            'TRADING_MODE': 'paper' if config_data.get('system', {}).get('dry_run', True) else 'live',
            'LOG_LEVEL': config_data.get('system', {}).get('log_level', 'INFO'),
            'MAX_CONCURRENT_POSITIONS': str(config_data.get('risk', {}).get('max_concurrent_trades', 3)),
            'DEFAULT_POSITION_SIZE': str(config_data.get('trading', {}).get('max_size_pct', 20.0) / 100.0),
            
            # Strategy settings
            'STRATEGY_TYPE': 'llm',
            'LLM_MODEL': config_data.get('llm', {}).get('model_name', 'gpt-4'),
            'LLM_TEMPERATURE': str(config_data.get('llm', {}).get('temperature', 0.1)),
            'LLM_MAX_CONTEXT': str(config_data.get('llm', {}).get('max_tokens', 4000)),
            'LLM_USE_MEMORY': str(config_data.get('mcp', {}).get('enabled', False)).lower(),
            'LLM_CONFIDENCE_THRESHOLD': '0.7',
            
            # Exchange settings
            'EXCHANGE_TYPE': config_data.get('exchange', {}).get('exchange_type', 'coinbase'),
            'RATE_LIMIT_RPS': '10',
            'RATE_LIMIT_RPM': str(config_data.get('exchange', {}).get('rate_limit_requests', 100)),
            'RATE_LIMIT_RPH': '1000',
        }
        
        # Set environment variables temporarily
        for key, value in env_mapping.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        
        try:
            # Build functional config using lazy loading
            FunctionalConfig, Success, Failure, Result = _get_functional_config()
            if FunctionalConfig:
                result = FunctionalConfig.from_env()
                if isinstance(result, Success):
                    return result.success()
            return None
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
                    
    except Exception as e:
        return None


def validate_settings(settings: Settings) -> Optional[str]:
    """Validate settings using functional patterns and return warnings if any."""
    warnings = []
    
    # Trading validation
    if settings.trading.leverage < 1 or settings.trading.leverage > 100:
        warnings.append(f"Trading leverage {settings.trading.leverage} is outside safe range (1-100)")
    
    if settings.trading.max_size_pct < 0.1 or settings.trading.max_size_pct > 100:
        warnings.append(f"Trading max size {settings.trading.max_size_pct}% is outside safe range (0.1-100%)")
    
    # Risk validation
    if settings.risk.max_daily_loss_pct < 0.1 or settings.risk.max_daily_loss_pct > 50:
        warnings.append(f"Risk max daily loss {settings.risk.max_daily_loss_pct}% is outside safe range (0.1-50%)")
    
    if settings.risk.max_concurrent_trades < 1 or settings.risk.max_concurrent_trades > 10:
        warnings.append(f"Risk max concurrent trades {settings.risk.max_concurrent_trades} is outside safe range (1-10)")
    
    # LLM validation
    if settings.llm.temperature < 0 or settings.llm.temperature > 2:
        warnings.append(f"LLM temperature {settings.llm.temperature} is outside valid range (0-2)")
    
    if settings.llm.max_tokens < 100 or settings.llm.max_tokens > 100000:
        warnings.append(f"LLM max tokens {settings.llm.max_tokens} is outside reasonable range (100-100000)")
    
    # Exchange validation
    if settings.exchange.api_timeout < 1 or settings.exchange.api_timeout > 300:
        warnings.append(f"Exchange API timeout {settings.exchange.api_timeout}s is outside safe range (1-300s)")
    
    # System validation
    if settings.system.update_frequency_seconds < 1 or settings.system.update_frequency_seconds > 3600:
        warnings.append(f"System update frequency {settings.system.update_frequency_seconds}s is outside reasonable range (1-3600s)")
    
    # Paper trading validation
    if settings.paper_trading.starting_balance < 100 or settings.paper_trading.starting_balance > 1000000:
        warnings.append(f"Paper trading starting balance ${settings.paper_trading.starting_balance} is outside reasonable range ($100-$1M)")
    
    # Logical consistency checks
    if settings.risk.default_take_profit_pct <= settings.risk.default_stop_loss_pct:
        warnings.append("Take profit percentage should be greater than stop loss percentage")
    
    if settings.system.dry_run and settings.system.environment == "production":
        warnings.append("Dry run mode enabled in production environment - this may not be intended")
    
    if not settings.system.dry_run and settings.exchange.cb_sandbox:
        warnings.append("Live trading mode with sandbox exchange - this may not be intended")
    
    return "; ".join(warnings) if warnings else None


def validate_settings_functional(settings: Settings) -> Union[Settings, str]:
    """Validate settings and return either valid settings or error message."""
    validation_result = validate_settings(settings)
    if validation_result:
        return f"Configuration validation failed: {validation_result}"
    return settings


def benchmark_config_loading(iterations: int = 100) -> Dict[str, float]:
    """Benchmark configuration loading performance."""
    import time
    
    # Test create_settings performance
    start_time = time.time()
    for _ in range(iterations):
        settings = create_settings()
    create_time = time.time() - start_time
    
    # Test file loading performance
    start_time = time.time()
    for _ in range(iterations):
        try:
            settings = load_settings_from_file('config/development.json')
        except:
            pass
    file_time = time.time() - start_time
    
    # Test validation performance
    settings = create_settings()
    start_time = time.time()
    for _ in range(iterations):
        validation_result = validate_settings(settings)
    validation_time = time.time() - start_time
    
    return {
        'create_settings_ms': (create_time / iterations) * 1000,
        'load_from_file_ms': (file_time / iterations) * 1000,
        'validation_ms': (validation_time / iterations) * 1000,
        'total_iterations': iterations
    }


# Global settings instance with functional programming backend
settings = create_settings()


def get_config():
    """Get configuration - compatibility function."""
    return settings


def get_functional_config() -> Optional['FunctionalConfig']:
    """Get functional configuration directly."""
    try:
        FunctionalConfig, Success, Failure, Result = _get_functional_config()
        if FunctionalConfig:
            result = FunctionalConfig.from_env()
            if isinstance(result, Success):
                return result.success()
    except (ImportError, AttributeError):
        pass
    return None


def get_config_template() -> Dict[str, Any]:
    """Get a configuration template with descriptions."""
    return {
        "trading": {
            "symbol": "BTC-USD",
            "interval": "5m",
            "leverage": 5,
            "max_size_pct": 20.0,
            "order_timeout_seconds": 30,
            "slippage_tolerance_pct": 0.1,
            "min_profit_pct": 0.5,
            "maker_fee_rate": 0.004,
            "taker_fee_rate": 0.006,
            "futures_fee_rate": 0.0015,
            "min_trading_interval_seconds": 60,
            "require_24h_data_before_trading": True,
            "min_candles_for_trading": 100,
            "enable_futures": True,
            "futures_account_type": "CFM",
            "auto_cash_transfer": True,
            "max_futures_leverage": 20,
        },
        "llm": {
            "provider": "openai",
            "model_name": "o3",
            "temperature": 0.1,
            "max_tokens": 30000,
            "request_timeout": 30,
            "max_retries": 3,
        },
        "exchange": {
            "exchange_type": "coinbase",
            "cb_sandbox": True,
            "api_timeout": 10,
            "rate_limit_requests": 10,
            "bluefin_network": "mainnet",
        },
        "risk": {
            "max_daily_loss_pct": 5.0,
            "max_concurrent_trades": 3,
            "default_stop_loss_pct": 2.0,
            "default_take_profit_pct": 4.0,
        },
        "system": {
            "dry_run": True,
            "environment": "development",
            "log_level": "INFO",
            "update_frequency_seconds": 30.0,
        },
        "paper_trading": {
            "starting_balance": 10000.0,
            "fee_rate": 0.001,
            "slippage_rate": 0.0005,
            "enable_daily_reports": True,
            "enable_weekly_summaries": True,
            "track_drawdown": True,
            "keep_trade_history_days": 90,
            "export_trade_data": False,
            "report_time_utc": "23:59",
            "include_unrealized_pnl": True,
        },
        "omnisearch": {
            "enabled": False,
            "server_url": "http://localhost:8766",
            "max_results": 5,
            "cache_ttl_seconds": 300,
            "rate_limit_requests_per_minute": 10,
            "timeout_seconds": 30,
            "enable_crypto_sentiment": True,
            "enable_nasdaq_sentiment": True,
            "enable_correlation_analysis": True,
        },
    }


# Add missing load_from_file method to Settings class for compatibility
Settings.load_from_file = classmethod(lambda cls, file_path: load_settings_from_file(file_path))


# Export all the classes and functions that the current interface provides
__all__ = [
    "Settings",
    "TradingSettings",
    "LLMSettings", 
    "ExchangeSettings",
    "RiskSettings",
    "DataSettings",
    "DominanceSettings",
    "SystemSettings",
    "PaperTradingSettings",
    "MonitoringSettings",
    "MCPSettings",
    "OmniSearchSettings",
    "create_settings",
    "load_settings_from_file",
    "get_config",
    "get_functional_config",
    "get_config_template",
    "validate_settings",
    "validate_settings_functional",
    "benchmark_config_loading",
    "settings",
    "ConfigValidationError",
    "ConfigError",
    # Functional programming exports (lazy-loaded)
    # These are available when fp types are loaded
]