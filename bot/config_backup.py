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

# Import functional programming types
from bot.fp.types.result import Failure, Result, Success
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
    build_backtest_config_from_env,
    build_exchange_config_from_env,
    build_strategy_config_from_env,
    build_system_config_from_env,
    validate_config,
    parse_bool_env,
    parse_env_var,
    parse_float_env,
    parse_int_env,
)


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
    
    def __init__(self, functional_config: Optional[StrategyConfig] = None, **kwargs):
        # Use functional config if provided, otherwise environment/kwargs
        if functional_config:
            # Extract from functional config based on strategy type
            self._from_functional_config(functional_config)
        else:
            # Load from environment first, then kwargs, then defaults (maintaining exact compatibility)
            self.symbol: str = os.getenv('TRADING__SYMBOL', kwargs.get('symbol', 'BTC-USD'))
            self.interval: str = os.getenv('TRADING__INTERVAL', kwargs.get('interval', '1m'))
            self.leverage: int = int(os.getenv('TRADING__LEVERAGE', kwargs.get('leverage', 5)))
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
    
    def _from_functional_config(self, config: StrategyConfig) -> None:
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
    
    def __init__(self, functional_config: Optional[StrategyConfig] = None, **kwargs) -> None:
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.provider: str = os.getenv('LLM__PROVIDER', kwargs.get('provider', 'openai'))
            self.model_name: str = os.getenv('LLM__MODEL_NAME', kwargs.get('model_name', 'gpt-4'))
            temp_result = parse_float_env('LLM__TEMPERATURE', kwargs.get('temperature', 0.1))
            self.temperature: float = temp_result.success() if isinstance(temp_result, Success) else 0.1
            max_tokens_result = parse_int_env('LLM__MAX_TOKENS', kwargs.get('max_tokens', 30000))
            self.max_tokens: int = max_tokens_result.success() if isinstance(max_tokens_result, Success) else 30000
            timeout_result = parse_int_env('LLM__REQUEST_TIMEOUT', kwargs.get('request_timeout', 30))
            self.request_timeout: int = timeout_result.success() if isinstance(timeout_result, Success) else 30
            retries_result = parse_int_env('LLM__MAX_RETRIES', kwargs.get('max_retries', 3))
            self.max_retries: int = retries_result.success() if isinstance(retries_result, Success) else 3
            self.openai_api_key: Optional[SecretStr] = None
            
            # Load from environment
            api_key = os.getenv('LLM__OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_api_key = SecretStr(api_key)
    
    def _from_functional_config(self, config: StrategyConfig) -> None:
        """Extract settings from functional configuration."""
        # Import here to avoid circular imports
        from bot.fp.types.config import LLMStrategyConfig
        
        if isinstance(config, LLMStrategyConfig):
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
    """Exchange configuration settings."""
    
    def __init__(self, **kwargs) -> None:
        self.exchange_type: str = os.getenv('EXCHANGE__EXCHANGE_TYPE', kwargs.get('exchange_type', 'coinbase'))
        self.cb_sandbox: bool = os.getenv('EXCHANGE__CB_SANDBOX', str(kwargs.get('cb_sandbox', True))).lower() in ('true', '1', 'yes')
        self.api_timeout: int = int(os.getenv('EXCHANGE__API_TIMEOUT', kwargs.get('api_timeout', 10)))
        self.rate_limit_requests: int = int(os.getenv('EXCHANGE__RATE_LIMIT_REQUESTS', kwargs.get('rate_limit_requests', 10)))
        self.rate_limit_window_seconds: int = int(os.getenv('EXCHANGE__RATE_LIMIT_WINDOW_SECONDS', kwargs.get('rate_limit_window_seconds', 60)))
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


@dataclass
class RiskSettings:
    """Risk management settings."""
    
    def __init__(self, **kwargs):
        self.max_daily_loss_pct: float = float(os.getenv('RISK__MAX_DAILY_LOSS_PCT', kwargs.get('max_daily_loss_pct', 5.0)))
        self.max_concurrent_trades: int = int(os.getenv('RISK__MAX_CONCURRENT_TRADES', kwargs.get('max_concurrent_trades', 3)))
        self.default_stop_loss_pct: float = float(os.getenv('RISK__DEFAULT_STOP_LOSS_PCT', kwargs.get('default_stop_loss_pct', 2.0)))
        self.default_take_profit_pct: float = float(os.getenv('RISK__DEFAULT_TAKE_PROFIT_PCT', kwargs.get('default_take_profit_pct', 4.0)))


@dataclass
class DataSettings:
    """Data management settings."""
    
    def __init__(self, **kwargs):
        self.keep_days: int = int(os.getenv('DATA__KEEP_DAYS', kwargs.get('keep_days', 30)))
        self.backup_enabled: bool = os.getenv('DATA__BACKUP_ENABLED', str(kwargs.get('backup_enabled', True))).lower() in ('true', '1', 'yes')


@dataclass
class DominanceSettings:
    """Market dominance settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = os.getenv('DOMINANCE__ENABLED', str(kwargs.get('enabled', False))).lower() in ('true', '1', 'yes')
        self.threshold: float = float(os.getenv('DOMINANCE__THRESHOLD', kwargs.get('threshold', 0.45)))


@dataclass
class SystemSettings:
    """System configuration settings."""
    
    def __init__(self, **kwargs):
        self.dry_run: bool = os.getenv('SYSTEM__DRY_RUN', str(kwargs.get('dry_run', True))).lower() in ('true', '1', 'yes')
        self.environment: str = os.getenv('SYSTEM__ENVIRONMENT', kwargs.get('environment', 'development'))
        self.log_level: str = os.getenv('SYSTEM__LOG_LEVEL', kwargs.get('log_level', 'INFO'))
        self.update_frequency_seconds: float = float(os.getenv('SYSTEM__UPDATE_FREQUENCY_SECONDS', kwargs.get('update_frequency_seconds', 30.0)))


@dataclass
class PaperTradingSettings:
    """Paper trading settings."""
    
    def __init__(self, **kwargs):
        self.starting_balance: float = float(os.getenv('PAPER_TRADING__STARTING_BALANCE', kwargs.get('starting_balance', 10000.0)))
        self.fee_rate: float = float(os.getenv('PAPER_TRADING__FEE_RATE', kwargs.get('fee_rate', 0.001)))
        self.slippage_rate: float = float(os.getenv('PAPER_TRADING__SLIPPAGE_RATE', kwargs.get('slippage_rate', 0.0005)))
        self.enable_daily_reports: bool = os.getenv('PAPER_TRADING__ENABLE_DAILY_REPORTS', str(kwargs.get('enable_daily_reports', True))).lower() in ('true', '1', 'yes')
        self.enable_weekly_summaries: bool = os.getenv('PAPER_TRADING__ENABLE_WEEKLY_SUMMARIES', str(kwargs.get('enable_weekly_summaries', True))).lower() in ('true', '1', 'yes')
        self.track_drawdown: bool = os.getenv('PAPER_TRADING__TRACK_DRAWDOWN', str(kwargs.get('track_drawdown', True))).lower() in ('true', '1', 'yes')
        self.keep_trade_history_days: int = int(os.getenv('PAPER_TRADING__KEEP_TRADE_HISTORY_DAYS', kwargs.get('keep_trade_history_days', 90)))
        self.export_trade_data: bool = os.getenv('PAPER_TRADING__EXPORT_TRADE_DATA', str(kwargs.get('export_trade_data', False))).lower() in ('true', '1', 'yes')
        self.report_time_utc: str = os.getenv('PAPER_TRADING__REPORT_TIME_UTC', kwargs.get('report_time_utc', '23:59'))
        self.include_unrealized_pnl: bool = os.getenv('PAPER_TRADING__INCLUDE_UNREALIZED_PNL', str(kwargs.get('include_unrealized_pnl', True))).lower() in ('true', '1', 'yes')


@dataclass
class MonitoringSettings:
    """System monitoring settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = os.getenv('MONITORING__ENABLED', str(kwargs.get('enabled', True))).lower() in ('true', '1', 'yes')
        self.check_interval: int = int(os.getenv('MONITORING__CHECK_INTERVAL', kwargs.get('check_interval', 60)))


@dataclass
class MCPSettings:
    """MCP (Model Context Protocol) settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = os.getenv('MCP_ENABLED', str(kwargs.get('enabled', False))).lower() in ('true', '1', 'yes')
        self.server_url: str = os.getenv('MCP_SERVER_URL', kwargs.get('server_url', 'http://localhost:8765'))


@dataclass
class OmniSearchSettings:
    """OmniSearch integration settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = os.getenv('OMNISEARCH__ENABLED', str(kwargs.get('enabled', False))).lower() in ('true', '1', 'yes')
        self.server_url: str = os.getenv('OMNISEARCH__SERVER_URL', kwargs.get('server_url', 'http://localhost:8766'))
        self.max_results: int = int(os.getenv('OMNISEARCH__MAX_RESULTS', kwargs.get('max_results', 5)))
        self.cache_ttl_seconds: int = int(os.getenv('OMNISEARCH__CACHE_TTL_SECONDS', kwargs.get('cache_ttl_seconds', 300)))
        self.rate_limit_requests_per_minute: int = int(os.getenv('OMNISEARCH__RATE_LIMIT_REQUESTS_PER_MINUTE', kwargs.get('rate_limit_requests_per_minute', 10)))
        self.timeout_seconds: int = int(os.getenv('OMNISEARCH__TIMEOUT_SECONDS', kwargs.get('timeout_seconds', 30)))
        self.enable_crypto_sentiment: bool = os.getenv('OMNISEARCH__ENABLE_CRYPTO_SENTIMENT', str(kwargs.get('enable_crypto_sentiment', True))).lower() in ('true', '1', 'yes')
        self.enable_nasdaq_sentiment: bool = os.getenv('OMNISEARCH__ENABLE_NASDAQ_SENTIMENT', str(kwargs.get('enable_nasdaq_sentiment', True))).lower() in ('true', '1', 'yes')
        self.enable_correlation_analysis: bool = os.getenv('OMNISEARCH__ENABLE_CORRELATION_ANALYSIS', str(kwargs.get('enable_correlation_analysis', True))).lower() in ('true', '1', 'yes')


class Settings:
    """Main configuration settings with functional programming foundation."""
    
    def __init__(self, **overrides):
        """Initialize settings with optional overrides."""
        # Create compatibility sections
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
    
    def apply_profile(self, profile: str) -> "Settings":
        """Apply a configuration profile."""
        # For now, return self - profiles will be implemented later
        return self
    
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
    """Create settings instance with optional environment file and overrides."""
    try:
        from dotenv import load_dotenv
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
    except ImportError:
        pass
    
    settings_kwargs = overrides or {}
    settings = Settings(**settings_kwargs)
    
    if profile:
        settings = settings.apply_profile(profile)
    
    return settings


# Global settings instance
settings = create_settings()


def get_config():
    """Get configuration - compatibility function."""
    return settings


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


# Export validation classes for compatibility
class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


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
    "get_config",
    "get_config_template",
    "settings",
    "ConfigValidationError",
    "ConfigError",
]