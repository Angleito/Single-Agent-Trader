"""
Type-safe configuration models with strong validation and safety defaults.

This module provides Pydantic models for all configuration types with comprehensive
validation rules that enforce safety-first defaults, especially for paper trading.
"""

import os
import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

# Constants for validation
MIN_LEVERAGE = 1
MAX_LEVERAGE = 20
SAFE_LEVERAGE = 5
MAX_POSITION_SIZE_PCT = 100.0
SAFE_POSITION_SIZE_PCT = 25.0
MIN_STOP_LOSS_PCT = 0.1
MAX_STOP_LOSS_PCT = 10.0
MIN_TAKE_PROFIT_PCT = 0.1
MAX_TAKE_PROFIT_PCT = 50.0
MAX_DAILY_LOSS_PCT = 10.0
MAX_CONSECUTIVE_LOSSES = 5


class ExchangeConfig(BaseModel):
    """Type-safe exchange configuration with validation."""

    exchange_type: Literal["bluefin", "coinbase"] = Field(
        default="coinbase",
        description="Exchange to use for trading",
    )
    service_url: str | None = Field(
        default=None,
        description="Exchange service URL",
    )
    bluefin_private_key: SecretStr | None = Field(
        default=None,
        description="Sui wallet private key for Bluefin (hex format)",
    )
    bluefin_network: Literal["mainnet", "testnet"] = Field(
        default="mainnet",
        description="Bluefin network to connect to",
    )
    cdp_api_key_name: SecretStr | None = Field(
        default=None,
        description="Coinbase CDP API key name",
    )
    cdp_private_key: SecretStr | None = Field(
        default=None,
        description="Coinbase CDP private key (PEM format)",
    )
    use_trade_aggregation: bool = Field(
        default=True,
        description="Enable trade aggregation for sub-minute intervals",
    )

    @field_validator("service_url")
    @classmethod
    def validate_service_url(cls, v: str | None) -> str | None:
        """Validate service URL format."""
        if v is None:
            return v

        # Allow localhost for development
        if v.startswith(("http://localhost", "http://127.0.0.1")):
            return v

        # Validate URL format
        url_pattern = r"^https?://[a-zA-Z0-9\-\.]+(?::\d+)?(?:/.*)?$"
        if not re.match(url_pattern, v):
            raise ValueError(
                f"Invalid service URL format: {v}. Must be a valid HTTP(S) URL."
            )

        # Parse URL for additional validation
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL structure: {v}")
        except Exception as e:
            raise ValueError(f"Failed to parse URL: {e}")

        return v

    @field_validator("bluefin_private_key")
    @classmethod
    def validate_bluefin_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Bluefin private key format."""
        if v is None:
            return v

        key = v.get_secret_value()

        # Check for hex format (64 chars for 32 bytes)
        if not re.match(r"^[a-fA-F0-9]{64}$", key):
            # Check if it starts with 0x
            if key.startswith("0x") and re.match(r"^0x[a-fA-F0-9]{64}$", key):
                # Strip 0x prefix and return
                return SecretStr(key[2:])
            raise ValueError("Bluefin private key must be 64 hex characters (32 bytes)")

        return v

    @field_validator("cdp_private_key")
    @classmethod
    def validate_cdp_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate Coinbase CDP private key format."""
        if v is None:
            return v

        key = v.get_secret_value()

        # Check for PEM format
        if not (
            "-----BEGIN EC PRIVATE KEY-----" in key
            or "-----BEGIN RSA PRIVATE KEY-----" in key
        ):
            raise ValueError("CDP private key must be in PEM format")

        return v

    @model_validator(mode="after")
    def validate_exchange_credentials(self) -> "ExchangeConfig":
        """Cross-validate exchange type and credentials."""
        if self.exchange_type == "bluefin":
            if not self.bluefin_private_key:
                raise ValueError("Bluefin exchange requires bluefin_private_key")
        elif self.exchange_type == "coinbase":
            if not self.cdp_api_key_name or not self.cdp_private_key:
                raise ValueError(
                    "Coinbase exchange requires both cdp_api_key_name and cdp_private_key"
                )

        return self


class TradingConfig(BaseModel):
    """Type-safe trading configuration with safety validation."""

    symbol: str = Field(
        default="BTC-USD",
        description="Trading pair symbol",
    )
    interval: str = Field(
        default="1m",
        description="Candle interval for analysis",
    )
    leverage: int = Field(
        default=SAFE_LEVERAGE,
        ge=MIN_LEVERAGE,
        le=MAX_LEVERAGE,
        description="Futures leverage multiplier",
    )
    dry_run: bool = Field(
        default=True,  # Safety first!
        description="Enable paper trading mode (no real money)",
    )
    position_mode: Literal["hedge", "one-way"] = Field(
        default="hedge",
        description="Position mode for futures trading",
    )
    max_size_pct: float = Field(
        default=SAFE_POSITION_SIZE_PCT,
        gt=0,
        le=MAX_POSITION_SIZE_PCT,
        description="Maximum position size as percentage of balance",
    )
    enable_shorts: bool = Field(
        default=True,
        description="Allow short positions",
    )
    enable_longs: bool = Field(
        default=True,
        description="Allow long positions",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        # Basic pattern for crypto pairs
        symbol_pattern = r"^[A-Z0-9]+-[A-Z0-9]+$"
        if not re.match(symbol_pattern, v):
            raise ValueError(
                f"Invalid symbol format: {v}. Must be like 'BTC-USD' or 'ETH-PERP'"
            )
        return v.upper()

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate candle interval."""
        valid_intervals = [
            "1s",
            "5s",
            "15s",
            "30s",
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "4h",
            "1d",
        ]
        if v not in valid_intervals:
            raise ValueError(
                f"Invalid interval: {v}. Must be one of {', '.join(valid_intervals)}"
            )
        return v

    @field_validator("leverage")
    @classmethod
    def validate_leverage_safety(cls, v: int) -> int:
        """Validate leverage with safety warnings."""
        if v > 10:
            # High leverage warning (not an error, but logged later)
            pass
        return v

    @model_validator(mode="after")
    def validate_trading_safety(self) -> "TradingConfig":
        """Cross-validate trading parameters for safety."""
        # Ensure at least one direction is enabled
        if not self.enable_longs and not self.enable_shorts:
            raise ValueError("At least one trading direction must be enabled")

        # Warn about high risk combinations (handled by logging layer)
        if not self.dry_run and self.leverage > 10:
            # This would be logged as a warning in the application
            pass

        return self


class RiskConfig(BaseModel):
    """Type-safe risk management configuration."""

    default_stop_loss_pct: float = Field(
        default=2.0,
        ge=MIN_STOP_LOSS_PCT,
        le=MAX_STOP_LOSS_PCT,
        description="Default stop loss percentage",
    )
    default_take_profit_pct: float = Field(
        default=4.0,
        ge=MIN_TAKE_PROFIT_PCT,
        le=MAX_TAKE_PROFIT_PCT,
        description="Default take profit percentage",
    )
    max_daily_loss_pct: float = Field(
        default=5.0,
        gt=0,
        le=MAX_DAILY_LOSS_PCT,
        description="Maximum daily loss percentage",
    )
    max_consecutive_losses: int = Field(
        default=3,
        ge=1,
        le=MAX_CONSECUTIVE_LOSSES,
        description="Maximum consecutive losses before stopping",
    )
    risk_per_trade: float = Field(
        default=1.0,
        gt=0,
        le=5.0,
        description="Risk percentage per trade",
    )
    min_risk_reward_ratio: float = Field(
        default=1.5,
        ge=0.5,
        le=10.0,
        description="Minimum risk/reward ratio",
    )
    max_open_positions: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum number of open positions",
    )
    margin_call_threshold_pct: float = Field(
        default=80.0,
        ge=50.0,
        le=95.0,
        description="Margin usage threshold for warnings",
    )
    emergency_stop_loss_pct: float = Field(
        default=10.0,
        ge=5.0,
        le=20.0,
        description="Emergency stop loss for extreme moves",
    )

    @model_validator(mode="after")
    def validate_risk_parameters(self) -> "RiskConfig":
        """Cross-validate risk parameters."""
        # Ensure take profit is greater than stop loss
        if self.default_take_profit_pct <= self.default_stop_loss_pct:
            raise ValueError("Take profit must be greater than stop loss")

        # Calculate and validate risk/reward ratio
        risk_reward = self.default_take_profit_pct / self.default_stop_loss_pct
        if risk_reward < self.min_risk_reward_ratio:
            raise ValueError(
                f"Risk/reward ratio ({risk_reward:.2f}) is below minimum "
                f"({self.min_risk_reward_ratio})"
            )

        return self


class SystemConfig(BaseModel):
    """Type-safe system configuration with safety defaults."""

    dry_run: bool = Field(
        default=True,  # ALWAYS default to paper trading
        description="Enable paper trading mode",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug logging",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    timezone: str = Field(
        default="UTC",
        description="System timezone",
    )
    auto_restart: bool = Field(
        default=True,
        description="Auto-restart on errors",
    )
    health_check_interval: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Health check interval in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for operations",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay between retries in seconds",
    )

    @model_validator(mode="after")
    def validate_production_safety(self) -> "SystemConfig":
        """Validate production environment safety."""
        if self.environment == "production":
            # Production should have stricter settings
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")
            if self.log_level == "DEBUG":
                raise ValueError("DEBUG log level not recommended for production")

        return self


class APIKeyConfig(BaseModel):
    """Type-safe API key configuration with validation."""

    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for LLM",
    )
    mem0_api_key: SecretStr | None = Field(
        default=None,
        description="Mem0 API key for memory storage",
    )
    coingecko_api_key: SecretStr | None = Field(
        default=None,
        description="CoinGecko API key for market data",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate OpenAI API key format."""
        if v is None:
            return v

        key = v.get_secret_value()
        if not key.startswith("sk-") or len(key) < 20:
            raise ValueError("Invalid OpenAI API key format")

        return v

    @model_validator(mode="after")
    def check_required_keys(self) -> "APIKeyConfig":
        """Check that required API keys are provided."""
        # OpenAI key is required for LLM trading
        if not self.openai_api_key:
            # This is a warning, not an error - handled by app logic
            pass

        return self


class WebSocketConfig(BaseModel):
    """Type-safe WebSocket configuration."""

    heartbeat_interval: int = Field(
        default=30,
        ge=5,
        le=120,
        description="WebSocket heartbeat interval in seconds",
    )
    reconnect_delay: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Delay before reconnection attempts",
    )
    max_reconnect_attempts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reconnection attempts",
    )
    message_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Message timeout in seconds",
    )
    buffer_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Message buffer size",
    )
    compression: bool = Field(
        default=True,
        description="Enable WebSocket compression",
    )

    @model_validator(mode="after")
    def validate_websocket_timings(self) -> "WebSocketConfig":
        """Validate WebSocket timing parameters."""
        # Heartbeat should be less than message timeout
        if self.heartbeat_interval >= self.message_timeout:
            raise ValueError("Heartbeat interval must be less than message timeout")

        return self


class CompleteConfig(BaseModel):
    """Complete configuration with all sections."""

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    api_keys: APIKeyConfig = Field(default_factory=APIKeyConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)

    @model_validator(mode="after")
    def validate_complete_config(self) -> "CompleteConfig":
        """Cross-validate all configuration sections."""
        # Ensure dry_run consistency
        if hasattr(self.trading, "dry_run") and hasattr(self.system, "dry_run"):
            if self.trading.dry_run != self.system.dry_run:
                raise ValueError(
                    "Inconsistent dry_run settings between trading and system config"
                )

        # Production environment checks
        if self.system.environment == "production":
            if self.system.dry_run:
                # Warning: production with dry_run
                pass
            else:
                # Live production trading - enforce strict limits
                if self.trading.leverage > 10:
                    raise ValueError(
                        "Production live trading limited to 10x leverage for safety"
                    )
                if self.trading.max_size_pct > 50:
                    raise ValueError(
                        "Production live trading limited to 50% position size for safety"
                    )

        # API key requirements
        if not self.system.dry_run and not self.api_keys.openai_api_key:
            raise ValueError("OpenAI API key required for live trading")

        return self


def load_config_from_env() -> CompleteConfig:
    """
    Load configuration from environment variables with validation.

    Returns:
        CompleteConfig: Validated configuration object

    Raises:
        ValidationError: If configuration is invalid
    """
    # Load from environment with prefix mapping
    config_data = {
        "exchange": {
            "exchange_type": os.getenv("EXCHANGE__EXCHANGE_TYPE", "coinbase"),
            "service_url": os.getenv("EXCHANGE__SERVICE_URL"),
            "bluefin_private_key": os.getenv("EXCHANGE__BLUEFIN_PRIVATE_KEY"),
            "bluefin_network": os.getenv("EXCHANGE__BLUEFIN_NETWORK", "mainnet"),
            "cdp_api_key_name": os.getenv("EXCHANGE__CDP_API_KEY_NAME"),
            "cdp_private_key": os.getenv("EXCHANGE__CDP_PRIVATE_KEY"),
            "use_trade_aggregation": os.getenv(
                "EXCHANGE__USE_TRADE_AGGREGATION", "true"
            ).lower()
            == "true",
        },
        "trading": {
            "symbol": os.getenv("TRADING__SYMBOL", "BTC-USD"),
            "interval": os.getenv("TRADING__INTERVAL", "1m"),
            "leverage": int(os.getenv("TRADING__LEVERAGE", str(SAFE_LEVERAGE))),
            "dry_run": os.getenv("SYSTEM__DRY_RUN", "true").lower() == "true",
            "max_size_pct": float(
                os.getenv("TRADING__MAX_SIZE_PCT", str(SAFE_POSITION_SIZE_PCT))
            ),
        },
        "risk": {
            "default_stop_loss_pct": float(
                os.getenv("RISK__DEFAULT_STOP_LOSS_PCT", "2.0")
            ),
            "default_take_profit_pct": float(
                os.getenv("RISK__DEFAULT_TAKE_PROFIT_PCT", "4.0")
            ),
            "max_daily_loss_pct": float(os.getenv("RISK__MAX_DAILY_LOSS_PCT", "5.0")),
            "max_consecutive_losses": int(
                os.getenv("RISK__MAX_CONSECUTIVE_LOSSES", "3")
            ),
        },
        "system": {
            "dry_run": os.getenv("SYSTEM__DRY_RUN", "true").lower() == "true",
            "environment": os.getenv("SYSTEM__ENVIRONMENT", "development"),
            "debug": os.getenv("SYSTEM__DEBUG", "false").lower() == "true",
            "log_level": os.getenv("SYSTEM__LOG_LEVEL", "INFO"),
        },
        "api_keys": {
            "openai_api_key": os.getenv("LLM__OPENAI_API_KEY"),
            "mem0_api_key": os.getenv("MEM0_API_KEY"),
        },
        "websocket": {
            "heartbeat_interval": int(os.getenv("WEBSOCKET__HEARTBEAT_INTERVAL", "30")),
            "reconnect_delay": float(os.getenv("WEBSOCKET__RECONNECT_DELAY", "5.0")),
            "max_reconnect_attempts": int(
                os.getenv("WEBSOCKET__MAX_RECONNECT_ATTEMPTS", "10")
            ),
        },
    }

    # Remove None values from nested dicts
    for section in config_data.values():
        keys_to_remove = [k for k, v in section.items() if v is None]
        for key in keys_to_remove:
            del section[key]

    return CompleteConfig(**config_data)


# Export commonly used types
__all__ = [
    "MAX_LEVERAGE",
    "MAX_POSITION_SIZE_PCT",
    # Constants
    "MIN_LEVERAGE",
    "SAFE_LEVERAGE",
    "SAFE_POSITION_SIZE_PCT",
    "APIKeyConfig",
    "CompleteConfig",
    "ExchangeConfig",
    "RiskConfig",
    "SystemConfig",
    "TradingConfig",
    "WebSocketConfig",
    "load_config_from_env",
]
