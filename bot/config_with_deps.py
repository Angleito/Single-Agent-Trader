"""
Functional configuration system with backward compatibility.

This module provides a functional programming approach to configuration
while maintaining full backward compatibility with the existing Settings interface.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import SecretStr
from bot.fp.types.result import Failure, Result, Success

from bot.fp.types.config import (
    APIKey,
    BacktestConfig,
    Config as FunctionalConfig,
    ExchangeConfig,
    ExchangeType,
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
)

T = TypeVar('T')


class ConfigError(Exception):
    """Configuration error."""
    pass


# Compatibility adapters for current Settings interface
@dataclass
class TradingSettings:
    """Trading configuration settings."""
    
    def __init__(self, **kwargs):
        # Default values that match current system
        self.symbol: str = kwargs.get('symbol', 'BTC-USD')
        self.interval: str = kwargs.get('interval', '1m')
        self.leverage: int = kwargs.get('leverage', 5)
        self.max_size_pct: float = kwargs.get('max_size_pct', 20.0)
        self.order_timeout_seconds: int = kwargs.get('order_timeout_seconds', 30)
        self.slippage_tolerance_pct: float = kwargs.get('slippage_tolerance_pct', 0.1)
        self.min_profit_pct: float = kwargs.get('min_profit_pct', 0.5)
        self.maker_fee_rate: float = kwargs.get('maker_fee_rate', 0.004)
        self.taker_fee_rate: float = kwargs.get('taker_fee_rate', 0.006)
        self.futures_fee_rate: float = kwargs.get('futures_fee_rate', 0.0015)
        self.min_trading_interval_seconds: int = kwargs.get('min_trading_interval_seconds', 60)
        self.require_24h_data_before_trading: bool = kwargs.get('require_24h_data_before_trading', True)
        self.min_candles_for_trading: int = kwargs.get('min_candles_for_trading', 100)
        self.enable_futures: bool = kwargs.get('enable_futures', True)
        self.futures_account_type: str = kwargs.get('futures_account_type', 'CFM')
        self.auto_cash_transfer: bool = kwargs.get('auto_cash_transfer', True)
        self.max_futures_leverage: int = kwargs.get('max_futures_leverage', 20)


@dataclass
class LLMSettings:
    """LLM configuration settings."""
    
    def __init__(self, **kwargs):
        self.provider: str = kwargs.get('provider', 'openai')
        self.model_name: str = kwargs.get('model_name', 'gpt-4')
        self.temperature: float = kwargs.get('temperature', 0.1)
        self.max_tokens: int = kwargs.get('max_tokens', 30000)
        self.request_timeout: int = kwargs.get('request_timeout', 30)
        self.max_retries: int = kwargs.get('max_retries', 3)
        self.openai_api_key: Optional[SecretStr] = None
        
        # Load from environment
        api_key = os.getenv('LLM__OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_api_key = SecretStr(api_key)


@dataclass
class ExchangeSettings:
    """Exchange configuration settings."""
    
    def __init__(self, **kwargs):
        self.exchange_type: str = kwargs.get('exchange_type', 'coinbase')
        self.cb_sandbox: bool = kwargs.get('cb_sandbox', True)
        self.api_timeout: int = kwargs.get('api_timeout', 10)
        self.rate_limit_requests: int = kwargs.get('rate_limit_requests', 10)
        self.bluefin_network: str = kwargs.get('bluefin_network', 'mainnet')
        
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
        self.max_daily_loss_pct: float = kwargs.get('max_daily_loss_pct', 5.0)
        self.max_concurrent_trades: int = kwargs.get('max_concurrent_trades', 3)
        self.default_stop_loss_pct: float = kwargs.get('default_stop_loss_pct', 2.0)
        self.default_take_profit_pct: float = kwargs.get('default_take_profit_pct', 4.0)


@dataclass
class DataSettings:
    """Data management settings."""
    
    def __init__(self, **kwargs):
        self.keep_days: int = kwargs.get('keep_days', 30)
        self.backup_enabled: bool = kwargs.get('backup_enabled', True)


@dataclass
class DominanceSettings:
    """Market dominance settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', False)
        self.threshold: float = kwargs.get('threshold', 0.45)


@dataclass
class SystemSettings:
    """System configuration settings."""
    
    def __init__(self, **kwargs):
        self.dry_run: bool = kwargs.get('dry_run', True)
        self.environment: str = kwargs.get('environment', 'development')
        self.log_level: str = kwargs.get('log_level', 'INFO')
        self.update_frequency_seconds: float = kwargs.get('update_frequency_seconds', 30.0)
        
        # Load from environment
        self.dry_run = os.getenv('SYSTEM__DRY_RUN', 'true').lower() in ('true', '1', 'yes')
        self.environment = os.getenv('SYSTEM__ENVIRONMENT', 'development')
        self.log_level = os.getenv('SYSTEM__LOG_LEVEL', 'INFO')


@dataclass
class PaperTradingSettings:
    """Paper trading settings."""
    
    def __init__(self, **kwargs):
        self.starting_balance: float = kwargs.get('starting_balance', 10000.0)
        self.fee_rate: float = kwargs.get('fee_rate', 0.001)
        self.slippage_rate: float = kwargs.get('slippage_rate', 0.0005)
        self.enable_daily_reports: bool = kwargs.get('enable_daily_reports', True)
        self.enable_weekly_summaries: bool = kwargs.get('enable_weekly_summaries', True)
        self.track_drawdown: bool = kwargs.get('track_drawdown', True)
        self.keep_trade_history_days: int = kwargs.get('keep_trade_history_days', 90)
        self.export_trade_data: bool = kwargs.get('export_trade_data', False)
        self.report_time_utc: str = kwargs.get('report_time_utc', '23:59')
        self.include_unrealized_pnl: bool = kwargs.get('include_unrealized_pnl', True)


@dataclass
class MonitoringSettings:
    """System monitoring settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', True)
        self.check_interval: int = kwargs.get('check_interval', 60)


@dataclass
class MCPSettings:
    """MCP (Model Context Protocol) settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', False)
        self.server_url: str = kwargs.get('server_url', 'http://localhost:8765')
        
        # Load from environment
        self.enabled = os.getenv('MCP_ENABLED', 'false').lower() in ('true', '1', 'yes')
        self.server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8765')


@dataclass
class OmniSearchSettings:
    """OmniSearch integration settings."""
    
    def __init__(self, **kwargs):
        self.enabled: bool = kwargs.get('enabled', False)
        self.server_url: str = kwargs.get('server_url', 'http://localhost:8766')
        self.max_results: int = kwargs.get('max_results', 5)
        self.cache_ttl_seconds: int = kwargs.get('cache_ttl_seconds', 300)
        self.rate_limit_requests_per_minute: int = kwargs.get('rate_limit_requests_per_minute', 10)
        self.timeout_seconds: int = kwargs.get('timeout_seconds', 30)
        self.enable_crypto_sentiment: bool = kwargs.get('enable_crypto_sentiment', True)
        self.enable_nasdaq_sentiment: bool = kwargs.get('enable_nasdaq_sentiment', True)
        self.enable_correlation_analysis: bool = kwargs.get('enable_correlation_analysis', True)


class Settings:
    """Main configuration settings using functional programming internally."""
    
    def __init__(self, **overrides):
        """Initialize settings with optional overrides."""
        # Load functional config
        self._functional_config = self._load_functional_config()
        
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
    
    def _load_functional_config(self) -> Optional[FunctionalConfig]:
        """Load functional configuration from environment."""
        try:
            config_result = FunctionalConfig.from_env()
            if isinstance(config_result, Success):
                validated = validate_config(config_result.success())
                if isinstance(validated, Success):
                    return validated.success()
                else:
                    print(f"Config validation failed: {validated.failure()}")
                    return None
            else:
                print(f"Config loading failed: {config_result.failure()}")
                return None
        except Exception as e:
            print(f"Failed to load functional config: {e}")
            return None
    
    def apply_profile(self, profile: str) -> "Settings":
        """Apply a configuration profile."""
        # For now, return self as profiles are handled at the functional level
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


def get_config():
    """Get configuration - compatibility function."""
    return settings


# Global settings instance
settings = create_settings()


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