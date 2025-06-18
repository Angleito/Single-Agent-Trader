"""Configuration settings for the AI Trading Bot."""

import json
import logging
import os
import re
import secrets
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingProfile(str, Enum):
    """Trading profile types for different risk levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class TradingSettings(BaseModel):
    """Trading-specific configuration."""

    model_config = ConfigDict(frozen=True)

    # Core Trading Parameters
    symbol: str = Field(
        default="BTC-USD",
        description="Trading symbol (e.g., BTC-USD, ETH-USD)",
        pattern=r"^[A-Z]+-[A-Z]+$",
    )
    interval: str = Field(
        default="15s",
        description="Candle interval for analysis (scalping mode: 1s-15s)",
    )
    leverage: int = Field(
        default=5, ge=1, le=100, description="Trading leverage multiplier"
    )
    max_size_pct: float = Field(
        default=20.0,
        ge=0.1,
        le=50.0,
        description="Maximum position size as percentage of equity",
    )

    # Futures Trading Configuration
    enable_futures: bool = Field(
        default=True, description="Enable futures trading (vs spot trading)"
    )
    futures_account_type: Literal["CFM", "CBI"] = Field(
        default="CFM", description="Futures account type - CFM (futures) or CBI (spot)"
    )
    auto_cash_transfer: bool = Field(
        default=True,
        description="Automatically transfer cash from spot to futures for margin",
    )
    intraday_margin_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Intraday margin requirement multiplier",
    )
    overnight_margin_multiplier: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Overnight margin requirement multiplier",
    )
    max_futures_leverage: int = Field(
        default=20, ge=1, le=100, description="Maximum leverage for futures positions"
    )
    fixed_contract_size: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Fixed number of contracts to trade (e.g., 10 for 1 ETH in ETH-USD futures)",
    )

    # Position Tracking Configuration
    use_fifo_accounting: bool = Field(
        default=True,
        description="Use FIFO (First In First Out) accounting for position tracking",
    )

    # Order Configuration
    order_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Order timeout in seconds"
    )
    slippage_tolerance_pct: float = Field(
        default=0.1,
        ge=0.0,
        le=5.0,
        description="Maximum acceptable slippage percentage",
    )
    min_profit_pct: float = Field(
        default=0.5, ge=0.1, le=10.0, description="Minimum profit target percentage"
    )

    # Trading Fee Configuration
    # Spot trading fees - default to basic tier (< $10K volume)
    spot_maker_fee_rate: float = Field(
        default=0.006,  # 0.6% maker fee (basic tier)
        ge=0.0,
        le=0.02,
        description="Spot maker fee rate (for limit orders)",
    )
    spot_taker_fee_rate: float = Field(
        default=0.012,  # 1.2% taker fee (basic tier)
        ge=0.0,
        le=0.02,
        description="Spot taker fee rate (for market orders)",
    )
    # Legacy fee names for backward compatibility
    maker_fee_rate: float = Field(
        default=0.006,  # Default to spot maker fee
        ge=0.0,
        le=0.02,
        description="Maker fee rate (for limit orders)",
    )
    taker_fee_rate: float = Field(
        default=0.012,  # Default to spot taker fee
        ge=0.0,
        le=0.02,
        description="Taker fee rate (for market orders)",
    )
    # Futures trading fees
    futures_fee_rate: float = Field(
        default=0.0015,  # 0.15% for futures
        ge=0.0,
        le=0.01,
        description="Futures trading fee rate",
    )

    # Volume-based fee tiers for Coinbase (monthly USD volume)
    fee_tier_thresholds: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"volume": 0, "maker": 0.006, "taker": 0.012},  # < $10K
            {"volume": 10000, "maker": 0.0025, "taker": 0.004},  # $10K+
            {"volume": 50000, "maker": 0.0015, "taker": 0.0025},  # $50K+
            {"volume": 100000, "maker": 0.001, "taker": 0.002},  # $100K+
            {"volume": 1000000, "maker": 0.0007, "taker": 0.0012},  # $1M+
            {"volume": 15000000, "maker": 0.0004, "taker": 0.0008},  # $15M+
            {"volume": 75000000, "maker": 0.0002, "taker": 0.0005},  # $75M+
            {"volume": 250000000, "maker": 0.0, "taker": 0.0005},  # $250M+
        ],
        description="Volume-based fee tiers for Coinbase spot trading",
    )

    # Trading Interval Configuration - Scalping Mode
    min_trading_interval_seconds: int = Field(
        default=15,  # 15 seconds minimum for high-frequency scalping
        ge=1,
        le=300,
        description="Minimum interval between trades in seconds (15s for scalping)",
    )
    require_24h_data_before_trading: bool = Field(
        default=False,
        description="Require at least 24 hours of market data before first trade (disabled for 8h trading)",
    )
    min_candles_for_trading: int = Field(
        default=1,
        ge=1,
        le=2000,
        description="Minimum number of candles required before trading (1 = start immediately with historical data)",
    )

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate trading interval format."""
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
            raise ValueError(f"Invalid interval. Must be one of: {valid_intervals}")
        return v


class LLMSettings(BaseModel):
    """LLM and AI configuration."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    # Provider Configuration
    provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai", description="LLM provider to use"
    )
    model_name: str = Field(default="o3", description="Specific model name/version")
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for response creativity",
    )
    max_tokens: int = Field(
        default=30000,
        ge=100,
        le=50000,
        description="Maximum tokens in LLM response (o3 supports up to 30000 per request)",
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for token repetition",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for topic repetition",
    )

    # API Configuration
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    openai_org_id: str | None = Field(
        default=None, description="OpenAI organization ID"
    )
    openai_base_url: AnyHttpUrl | None = Field(
        default=None, description="Custom OpenAI API base URL"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, description="Anthropic API key"
    )
    anthropic_base_url: AnyHttpUrl | None = Field(
        default=None, description="Custom Anthropic API base URL"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL for local models"
    )

    # Request Configuration
    request_timeout: int = Field(
        default=30, ge=5, le=120, description="API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum API request retries"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Delay between retries in seconds"
    )
    retry_exponential_base: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff base for retries"
    )

    # Response Caching
    enable_caching: bool = Field(
        default=True, description="Enable response caching for identical prompts"
    )
    cache_ttl_seconds: int = Field(
        default=300, ge=0, le=3600, description="Cache TTL in seconds (0 = no expiry)"
    )

    # LLM Logging Configuration
    enable_completion_logging: bool = Field(
        default=True, description="Enable detailed LLM completion logging"
    )
    completion_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level for LLM completions"
    )
    completion_log_file: str = Field(
        default="logs/llm_completions.log",
        description="Path to LLM completion log file",
    )
    log_prompt_preview_length: int = Field(
        default=500, ge=100, le=2000, description="Length of prompt preview in logs"
    )
    log_response_preview_length: int = Field(
        default=1000, ge=100, le=5000, description="Length of response preview in logs"
    )
    enable_performance_tracking: bool = Field(
        default=True, description="Enable LLM performance and cost tracking"
    )
    enable_langchain_callbacks: bool = Field(
        default=True,
        description="Enable LangChain callback handlers for detailed tracing",
    )
    log_market_context: bool = Field(
        default=True, description="Include market context in LLM completion logs"
    )
    enable_token_usage_tracking: bool = Field(
        default=True, description="Track and log token usage for cost analysis"
    )
    performance_log_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Log performance metrics every N completions",
    )

    @field_validator("openai_api_key", "anthropic_api_key")
    @classmethod
    def validate_api_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate API key format."""
        if v is None:
            return v

        key = v.get_secret_value()
        if not key.strip():
            raise ValueError("API key cannot be empty")

        # Basic format validation
        if len(key) < 20:
            raise ValueError("API key seems too short")

        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str, info) -> str:
        """Validate model name based on provider."""
        provider = info.data.get("provider", "openai")

        if provider == "openai":
            valid_models = [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-3.5-turbo",
                "o3",
                "o3-mini",
            ]
            if not any(v.startswith(model) for model in valid_models):
                raise ValueError(f"Invalid OpenAI model: {v}")
        elif provider == "anthropic":
            valid_models = ["claude-3", "claude-2", "claude-instant"]
            if not any(v.startswith(model) for model in valid_models):
                raise ValueError(f"Invalid Anthropic model: {v}")

        return v


class MCPSettings(BaseModel):
    """MCP (Model Context Protocol) configuration for memory and learning."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False, description="Enable MCP memory server for learning"
    )
    server_url: str = Field(
        default="http://localhost:8765", description="MCP memory server URL"
    )
    memory_api_key: SecretStr | None = Field(
        default=None, description="API key for memory server (if required)"
    )

    # Memory Configuration
    max_memories_per_query: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum similar experiences to retrieve per query",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for memory retrieval",
    )
    memory_retention_days: int = Field(
        default=90, ge=7, le=365, description="Days to retain trading memories"
    )

    # Learning Configuration
    enable_pattern_learning: bool = Field(
        default=True, description="Enable pattern recognition and learning"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Learning rate for strategy adjustments",
    )
    min_samples_for_pattern: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Minimum samples needed to identify a pattern",
    )
    confidence_decay_rate: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Decay rate for pattern confidence over time",
    )

    # Experience Collection
    track_trade_lifecycle: bool = Field(
        default=True, description="Track complete trade lifecycle for learning"
    )
    store_market_snapshots: bool = Field(
        default=True, description="Store market snapshots at key decision points"
    )
    reflection_delay_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Minutes to wait before analyzing trade outcome",
    )


class OmniSearchSettings(BaseModel):
    """OmniSearch MCP configuration for enhanced market intelligence and sentiment analysis."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="Enable OmniSearch MCP server for market intelligence",
    )
    api_key: SecretStr | None = Field(
        default=None, description="API key for OmniSearch service"
    )
    server_url: str = Field(
        default="http://localhost:8766", description="OmniSearch MCP server URL"
    )

    # Search Configuration
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to retrieve per query",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cache TTL for search results in seconds",
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Maximum requests per minute to prevent API throttling",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Request timeout in seconds",
    )

    # Feature Toggles
    enable_crypto_sentiment: bool = Field(
        default=True, description="Enable cryptocurrency sentiment analysis"
    )
    enable_nasdaq_sentiment: bool = Field(
        default=True, description="Enable NASDAQ/traditional market sentiment analysis"
    )
    enable_correlation_analysis: bool = Field(
        default=True, description="Enable cross-market correlation analysis"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate OmniSearch API key format."""
        if v is None:
            return v

        key = v.get_secret_value()
        if not key.strip():
            raise ValueError("OmniSearch API key cannot be empty")

        # Basic format validation
        if len(key) < 10:
            raise ValueError("OmniSearch API key seems too short")

        return v


class ExchangeSettings(BaseModel):
    """Exchange API configuration."""

    model_config = ConfigDict(frozen=True)

    # Exchange Selection
    exchange_type: Literal["coinbase", "bluefin"] = Field(
        default="coinbase", description="Exchange to use for trading"
    )

    # Coinbase Configuration - Legacy Advanced Trade API
    cb_api_key: SecretStr | None = Field(
        default=None, description="Coinbase Advanced Trade API key (legacy)"
    )
    cb_api_secret: SecretStr | None = Field(
        default=None, description="Coinbase Advanced Trade API secret (legacy)"
    )
    cb_passphrase: SecretStr | None = Field(
        default=None, description="Coinbase Advanced Trade passphrase (legacy)"
    )

    # Coinbase Configuration - CDP API Keys
    cdp_api_key_name: SecretStr | None = Field(
        default=None,
        description="Coinbase CDP API key name (from organizations/.../apiKeys/...)",
    )
    cdp_private_key: SecretStr | None = Field(
        default=None, description="Coinbase CDP private key (PEM format)"
    )
    cb_sandbox: bool = Field(
        default=True, description="Use Coinbase sandbox environment"
    )
    cb_base_url: AnyHttpUrl | None = Field(
        default=None, description="Custom Coinbase API base URL"
    )

    # API Configuration
    api_timeout: int = Field(
        default=10, ge=1, le=60, description="API request timeout in seconds"
    )
    rate_limit_requests: int = Field(
        default=10, ge=1, le=100, description="Maximum requests per minute"
    )
    rate_limit_window_seconds: int = Field(
        default=60, ge=1, le=3600, description="Rate limit window in seconds"
    )
    websocket_reconnect_attempts: int = Field(
        default=5, ge=1, le=20, description="WebSocket reconnection attempts"
    )
    websocket_timeout: int = Field(
        default=30, ge=5, le=300, description="WebSocket connection timeout in seconds"
    )

    # Connection Health
    health_check_interval: int = Field(
        default=300, ge=60, le=3600, description="API health check interval in seconds"
    )

    # Bluefin Configuration
    bluefin_private_key: SecretStr | None = Field(
        default=None, description="Bluefin Sui wallet private key"
    )
    bluefin_network: Literal["mainnet", "testnet"] = Field(
        default="mainnet", description="Bluefin network to connect to"
    )
    bluefin_rpc_url: str | None = Field(
        default=None, description="Custom Sui RPC endpoint for Bluefin"
    )
    bluefin_service_url: str = Field(
        default="http://bluefin-service:8080",
        description="Bluefin microservice URL for SDK operations",
    )

    @field_validator(
        "cb_api_key",
        "cb_api_secret",
        "cb_passphrase",
        "cdp_api_key_name",
        "cdp_private_key",
        "bluefin_private_key",
    )
    @classmethod
    def validate_api_credentials(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate API credential format."""
        if v is not None and len(v.get_secret_value().strip()) == 0:
            raise ValueError("API credentials cannot be empty strings")
        return v

    @model_validator(mode="after")
    def validate_exchange_credentials(self) -> "ExchangeSettings":
        """Validate exchange-specific credentials based on exchange type."""
        if self.exchange_type == "coinbase":
            # Validate Coinbase credentials only when using Coinbase
            has_legacy = all(
                [
                    self.cb_api_key and self.cb_api_key.get_secret_value().strip(),
                    self.cb_api_secret
                    and self.cb_api_secret.get_secret_value().strip(),
                    self.cb_passphrase
                    and self.cb_passphrase.get_secret_value().strip(),
                ]
            )

            has_cdp = all(
                [
                    self.cdp_api_key_name
                    and self.cdp_api_key_name.get_secret_value().strip(),
                    self.cdp_private_key
                    and self.cdp_private_key.get_secret_value().strip(),
                ]
            )

            # Allow neither (for dry run mode), but not both
            if has_legacy and has_cdp:
                raise ValueError(
                    "Cannot use both legacy and CDP credentials. Choose one method."
                )

            # Validate CDP private key format ONLY if provided AND we're using Coinbase
            if has_cdp and self.cdp_private_key:
                private_key = self.cdp_private_key.get_secret_value()
                if private_key and not private_key.startswith(
                    "-----BEGIN EC PRIVATE KEY-----"
                ):
                    raise ValueError(
                        "CDP private key must be in PEM format starting with '-----BEGIN EC PRIVATE KEY-----'"
                    )

        elif self.exchange_type == "bluefin":
            # Validate Bluefin credentials only when using Bluefin
            if (
                self.bluefin_private_key
                and self.bluefin_private_key.get_secret_value().strip()
            ):
                # Support multiple Sui private key formats with proper validation
                private_key = self.bluefin_private_key.get_secret_value().strip()

                # Check for mnemonic phrase format (12 or 24 words)
                words = private_key.split()
                if len(words) in [12, 24]:
                    # This is a mnemonic phrase - validate word count and basic structure
                    if all(word.isalpha() and len(word) > 2 for word in words):
                        return self
                    else:
                        raise ValueError(
                            "Bluefin mnemonic phrase contains invalid words"
                        )

                # Remove common prefixes for other formats
                if private_key.startswith("0x"):
                    private_key = private_key[2:]
                elif private_key.startswith("suiprivkey"):
                    # This is a Bech32-encoded Sui private key, which is valid
                    # Basic length validation for Bech32 format
                    if len(private_key) < 20:
                        raise ValueError(
                            "Bluefin Sui private key in Bech32 format appears too short"
                        )
                    return self

                # Validate hex format (after removing 0x prefix)
                if private_key:
                    # Check for valid hex characters
                    if not all(c in "0123456789abcdefABCDEF" for c in private_key):
                        raise ValueError(
                            "Bluefin private key must be a valid hexadecimal string, Sui private key format (suiprivkey...), or mnemonic phrase"
                        )
                    # Check for reasonable length (32 bytes = 64 hex chars)
                    if len(private_key) != 64:
                        raise ValueError(
                            "Bluefin hex private key must be exactly 64 characters (32 bytes)"
                        )

        return self


class RiskSettings(BaseModel):
    """Risk management configuration."""

    model_config = ConfigDict(frozen=True)

    # Loss Limits
    max_daily_loss_pct: float = Field(
        default=5.0, ge=0.1, le=50.0, description="Maximum daily loss percentage"
    )
    max_weekly_loss_pct: float = Field(
        default=15.0, ge=0.1, le=75.0, description="Maximum weekly loss percentage"
    )
    max_monthly_loss_pct: float = Field(
        default=30.0, ge=0.1, le=90.0, description="Maximum monthly loss percentage"
    )

    # Position Limits
    max_concurrent_trades: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Maximum concurrent positions (default: 1 for single position rule)",
    )
    max_position_hold_hours: int = Field(
        default=1,
        ge=1,
        le=24,  # 1 day max for scalping
        description="Maximum hours to hold a position (short for scalping)",
    )

    # Stop Loss and Take Profit - Scalping Mode
    default_stop_loss_pct: float = Field(
        default=0.3,
        ge=0.05,
        le=2.0,
        description="Default stop loss percentage (tight for scalping)",
    )
    default_take_profit_pct: float = Field(
        default=0.5,
        ge=0.1,
        le=3.0,
        description="Default take profit percentage (quick scalping targets)",
    )

    # Account Protection
    min_account_balance: Decimal = Field(
        default=Decimal("100"),
        ge=Decimal("10"),
        description="Minimum account balance to continue trading",
    )
    emergency_stop_loss_pct: float = Field(
        default=10.0, ge=1.0, le=25.0, description="Emergency stop loss percentage"
    )


class DataSettings(BaseModel):
    """Data and indicator configuration."""

    model_config = ConfigDict(frozen=True)

    # Data Fetching - Scalping Mode
    candle_limit: int = Field(
        default=1000,
        ge=200,
        le=5000,
        description="Number of historical candles to fetch (1000 = ~4 hours for 15s intervals, optimal for fast indicators)",
    )
    real_time_updates: bool = Field(
        default=True, description="Enable real-time data updates"
    )
    data_cache_ttl_seconds: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Data cache TTL in seconds (fast refresh for scalping)",
    )

    # Indicator Configuration
    indicator_warmup: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Indicator warmup period (minimum 100 for VuManChu indicators)",
    )

    # VuManChu Cipher Settings - Scalping Mode (Faster Periods)
    cipher_a_ema_length: int = Field(
        default=7, ge=3, le=21, description="Cipher A EMA length (faster for scalping)"
    )
    cipher_b_vwap_length: int = Field(
        default=10,
        ge=3,
        le=20,
        description="Cipher B VWAP length (faster for scalping)",
    )

    # Cipher B Signal Filter Configuration
    enable_cipher_b_filter: bool = Field(
        default=True,
        description="Enable Cipher B signal filtering for trade validation",
    )
    cipher_b_wave_bullish_threshold: float = Field(
        default=0.0, description="Cipher B wave threshold for bullish signals"
    )
    cipher_b_wave_bearish_threshold: float = Field(
        default=0.0, description="Cipher B wave threshold for bearish signals"
    )
    cipher_b_money_flow_bullish_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Cipher B money flow threshold for bullish signals",
    )
    cipher_b_money_flow_bearish_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Cipher B money flow threshold for bearish signals",
    )

    # Storage Configuration
    data_storage_path: Path = Field(
        default_factory=lambda: Path("data"), description="Path for data storage"
    )
    keep_historical_days: int = Field(
        default=30, ge=1, le=365, description="Days of historical data to keep"
    )


class DominanceSettings(BaseModel):
    """Stablecoin dominance data configuration."""

    model_config = ConfigDict(frozen=True)

    # Feature Toggle
    enable_dominance_data: bool = Field(
        default=True, description="Enable stablecoin dominance data integration"
    )

    # Data Source Configuration
    data_source: Literal["coingecko", "coinmarketcap", "custom"] = Field(
        default="coingecko", description="Dominance data source"
    )
    api_key: SecretStr | None = Field(
        default=None, description="API key for premium data sources"
    )

    # Update Configuration
    update_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Dominance data update interval in seconds",
    )
    cache_ttl: int = Field(
        default=300, ge=30, le=3600, description="Dominance data cache TTL in seconds"
    )

    # Analysis Configuration
    dominance_weight_in_decisions: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight of dominance data in trading decisions (0-1)",
    )

    # Alert Thresholds
    dominance_alert_threshold: float = Field(
        default=12.0,
        ge=5.0,
        le=20.0,
        description="Stablecoin dominance percentage to trigger alerts",
    )
    dominance_change_alert_threshold: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Dominance change percentage (24h) to trigger alerts",
    )


class PaperTradingSettings(BaseModel):
    """Paper trading configuration."""

    model_config = ConfigDict(frozen=True)

    # Account Configuration
    starting_balance: Decimal = Field(
        default=Decimal("10000"),
        ge=Decimal("100"),
        description="Starting balance for paper trading",
    )
    fee_rate: float = Field(
        default=0.001, ge=0.0, le=0.01, description="Trading fee rate (0.1% default)"
    )
    slippage_rate: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.005,
        description="Slippage simulation rate (0.05% default)",
    )

    # Performance Tracking
    enable_daily_reports: bool = Field(
        default=True, description="Enable daily performance reports"
    )
    enable_weekly_summaries: bool = Field(
        default=True, description="Enable weekly performance summaries"
    )
    track_drawdown: bool = Field(default=True, description="Track maximum drawdown")

    # Data Retention
    keep_trade_history_days: int = Field(
        default=90, ge=7, le=365, description="Days to keep trade history"
    )
    export_trade_data: bool = Field(
        default=False, description="Export trade data for external analysis"
    )

    # Reporting Configuration
    report_time_utc: str = Field(
        default="23:59", description="UTC time for daily reports (HH:MM)"
    )
    include_unrealized_pnl: bool = Field(
        default=True, description="Include unrealized P&L in reports"
    )


class SystemSettings(BaseModel):
    """System and operational configuration."""

    model_config = ConfigDict(frozen=True)

    # Execution Mode
    dry_run: bool = Field(
        default=True,
        description="Single control for paper trading/dry run mode - when True, no real trades are executed",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    instance_id: str = Field(
        default_factory=lambda: secrets.token_hex(8),
        description="Unique instance identifier",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_file_path: Path | None = Field(
        default=Path("logs/bot.log"), description="Log file path"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    log_to_console: bool = Field(default=True, description="Enable console logging")
    log_to_file: bool = Field(default=True, description="Enable file logging")
    max_log_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Maximum log file size in MB"
    )
    log_backup_count: int = Field(
        default=5, ge=1, le=50, description="Number of log backup files to keep"
    )
    log_retention_days: int = Field(
        default=30, ge=1, le=365, description="Log retention period in days"
    )

    # Performance - Scalping Mode
    update_frequency_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Main loop update frequency (1s for high-frequency scalping)",
    )
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing where possible"
    )
    max_worker_threads: int = Field(
        default=4, ge=1, le=20, description="Maximum worker threads"
    )
    memory_limit_mb: int | None = Field(
        default=None, ge=100, le=8192, description="Memory usage limit in MB"
    )

    # Monitoring and Alerts
    enable_monitoring: bool = Field(
        default=True, description="Enable system monitoring"
    )
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_export_interval: int = Field(
        default=60, ge=10, le=3600, description="Metrics export interval in seconds"
    )
    alert_webhook_url: AnyHttpUrl | None = Field(
        default=None, description="Webhook URL for alerts"
    )
    alert_email: str | None = Field(
        default=None, description="Email address for alerts"
    )
    health_check_interval: int = Field(
        default=300, ge=60, le=3600, description="Health check interval in seconds"
    )

    # Graceful Shutdown
    shutdown_timeout: int = Field(
        default=30, ge=5, le=300, description="Graceful shutdown timeout in seconds"
    )

    # Security
    enable_api_auth: bool = Field(default=True, description="Enable API authentication")
    api_secret_key: SecretStr | None = Field(
        default=None, description="Secret key for API authentication"
    )

    # WebSocket Publishing for Real-time Dashboard Integration
    enable_websocket_publishing: bool = Field(
        default=False, description="Enable real-time WebSocket publishing to dashboard"
    )
    websocket_dashboard_url: str = Field(
        default="ws://localhost:8000/ws",
        description="Dashboard WebSocket URL for real-time data",
    )
    websocket_publish_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="WebSocket publishing interval in seconds",
    )
    websocket_max_retries: int = Field(
        default=10, ge=1, le=20, description="Maximum WebSocket reconnection attempts"
    )
    websocket_retry_delay: int = Field(
        default=3, ge=1, le=60, description="WebSocket reconnection base delay in seconds"
    )
    websocket_timeout: int = Field(
        default=30, ge=10, le=120, description="WebSocket connection timeout in seconds"
    )
    websocket_queue_size: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum queued messages during connection issues",
    )
    # Additional WebSocket resilience settings
    websocket_ping_interval: int = Field(
        default=15, ge=5, le=60, description="WebSocket ping interval in seconds"
    )
    websocket_ping_timeout: int = Field(
        default=8, ge=3, le=30, description="WebSocket ping timeout in seconds"
    )
    websocket_health_check_interval: int = Field(
        default=30, ge=10, le=300, description="WebSocket health check interval in seconds"
    )

    @field_validator("alert_email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        """Validate email address format."""
        if v is None:
            return v

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email address format")

        return v


class Settings(BaseSettings):
    """Main configuration settings for the AI Trading Bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        frozen=True,
        extra="allow",
    )

    # Configuration Sections
    trading: TradingSettings = Field(default_factory=TradingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    dominance: DominanceSettings = Field(default_factory=DominanceSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    paper_trading: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    omnisearch: OmniSearchSettings = Field(default_factory=OmniSearchSettings)

    # Profile Configuration
    profile: TradingProfile = Field(
        default=TradingProfile.MODERATE, description="Trading risk profile"
    )

    @computed_field
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.system.environment == Environment.PRODUCTION

    @computed_field
    def requires_api_keys(self) -> bool:
        """Check if API keys are required for current configuration."""
        return (
            not self.system.dry_run or self.system.environment == Environment.PRODUCTION
        )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Validate required API keys based on configuration."""
        if self.requires_api_keys:
            # Validate LLM API keys
            if self.llm.provider == "openai" and not self.llm.openai_api_key:
                raise ValueError("OpenAI API key required for live trading")
            elif self.llm.provider == "anthropic" and not self.llm.anthropic_api_key:
                raise ValueError("Anthropic API key required for live trading")

            # Validate exchange API keys
            if not self.system.dry_run:
                if self.exchange.exchange_type == "coinbase":
                    has_legacy = all(
                        [
                            self.exchange.cb_api_key,
                            self.exchange.cb_api_secret,
                            self.exchange.cb_passphrase,
                        ]
                    )

                    has_cdp = all(
                        [
                            self.exchange.cdp_api_key_name,
                            self.exchange.cdp_private_key,
                        ]
                    )

                    if not (has_legacy or has_cdp):
                        raise ValueError(
                            "Coinbase API credentials required for live trading (either legacy or CDP keys)"
                        )
                elif self.exchange.exchange_type == "bluefin":
                    if not self.exchange.bluefin_private_key:
                        raise ValueError(
                            "Bluefin private key required for live trading"
                        )

        return self

    @model_validator(mode="after")
    def validate_environment_consistency(self) -> "Settings":
        """Validate environment-specific configuration consistency."""
        env = self.system.environment

        # Production environment validations
        if env == Environment.PRODUCTION:
            # Allow dry-run override for testing purposes
            if self.system.dry_run:
                logger.warning(
                    "Production environment running in dry-run mode - this should only be for testing"
                )

            if self.exchange.cb_sandbox:
                raise ValueError("Production environment cannot use sandbox exchange")

            if self.trading.leverage > 10:
                raise ValueError("Production leverage should not exceed 10x")

            if self.risk.max_daily_loss_pct > 10.0:
                raise ValueError("Production daily loss limit should not exceed 10%")

        # Development environment validations
        elif env == Environment.DEVELOPMENT:
            if not self.system.dry_run:
                raise ValueError("Development environment should use dry-run mode")

        return self

    @model_validator(mode="after")
    def validate_risk_parameters(self) -> "Settings":
        """Validate risk management parameters."""
        if self.risk.default_take_profit_pct <= self.risk.default_stop_loss_pct:
            raise ValueError("Take profit must be greater than stop loss")

        if self.risk.max_weekly_loss_pct <= self.risk.max_daily_loss_pct:
            raise ValueError("Weekly loss limit must be greater than daily loss limit")

        if self.risk.max_monthly_loss_pct <= self.risk.max_weekly_loss_pct:
            raise ValueError(
                "Monthly loss limit must be greater than weekly loss limit"
            )

        return self

    def apply_profile(self, profile: TradingProfile) -> "Settings":
        """Apply a trading profile to adjust risk settings."""
        profile_configs = {
            TradingProfile.CONSERVATIVE: {
                "max_size_pct": 10.0,
                "leverage": 2,
                "max_daily_loss_pct": 2.0,
                "max_concurrent_trades": 1,
                "default_stop_loss_pct": 1.5,
                "default_take_profit_pct": 3.0,
            },
            TradingProfile.MODERATE: {
                "max_size_pct": 20.0,
                "leverage": 5,
                "max_daily_loss_pct": 5.0,
                "max_concurrent_trades": 3,
                "default_stop_loss_pct": 2.0,
                "default_take_profit_pct": 4.0,
            },
            TradingProfile.AGGRESSIVE: {
                "max_size_pct": 40.0,
                "leverage": 10,
                "max_daily_loss_pct": 10.0,
                "max_concurrent_trades": 5,
                "default_stop_loss_pct": 3.0,
                "default_take_profit_pct": 6.0,
            },
        }

        if profile == TradingProfile.CUSTOM:
            return self

        config_updates = profile_configs[profile]

        # Create new instances with updated values
        new_trading = self.trading.model_copy(
            update={
                "max_size_pct": config_updates["max_size_pct"],
                "leverage": config_updates["leverage"],
            }
        )

        new_risk = self.risk.model_copy(
            update={
                "max_daily_loss_pct": config_updates["max_daily_loss_pct"],
                "max_concurrent_trades": config_updates["max_concurrent_trades"],
                "default_stop_loss_pct": config_updates["default_stop_loss_pct"],
                "default_take_profit_pct": config_updates["default_take_profit_pct"],
            }
        )

        return self.model_copy(
            update={
                "trading": new_trading,
                "risk": new_risk,
                "profile": profile,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Convert settings to JSON string."""
        return self.model_dump_json(indent=indent)

    def save_to_file(
        self, file_path: str | Path, include_secrets: bool = False
    ) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Export with or without secrets
        if include_secrets:
            data = self.model_dump(mode="json")
        else:
            data = self.model_dump(
                mode="json",
                exclude={
                    "llm": {"openai_api_key", "anthropic_api_key"},
                    "exchange": {
                        "cb_api_key",
                        "cb_api_secret",
                        "cb_passphrase",
                        "cdp_api_key_name",
                        "cdp_private_key",
                        "bluefin_private_key",
                    },
                    "system": {"api_secret_key"},
                    "mcp": {"memory_api_key"},
                    "omnisearch": {"api_key"},
                },
            )

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Add metadata
        metadata = {
            "exported_at": datetime.now(UTC).isoformat(),
            "version": "1.0",
            "environment": self.system.environment.value,
            "profile": self.profile.value,
            "includes_secrets": include_secrets,
        }

        metadata_path = path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "Settings":
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            config_data = json.load(f)

        return cls(**config_data)

    def validate_trading_environment(self) -> list[str]:
        """Validate configuration for trading environment."""
        warnings = []

        if self.system.dry_run and self.system.environment == Environment.PRODUCTION:
            warnings.append("Running in dry-run mode in production environment")

        if not self.system.dry_run and self.exchange.cb_sandbox:
            warnings.append("Live trading enabled but using sandbox environment")

        if (
            self.trading.leverage > 10
            and self.system.environment == Environment.PRODUCTION
        ):
            warnings.append("High leverage detected in production environment")

        if self.risk.max_daily_loss_pct > 10.0:
            warnings.append("High daily loss limit may pose significant risk")

        # Additional validations
        if self.trading.max_size_pct > 50.0:
            warnings.append("Position size over 50% of equity is highly risky")

        if self.risk.max_concurrent_trades > 10:
            warnings.append("High number of concurrent trades may impact performance")

        risk_reward_ratio = (
            self.risk.default_take_profit_pct / self.risk.default_stop_loss_pct
        )
        if risk_reward_ratio < 1.5:
            warnings.append(
                f"Risk/reward ratio of {risk_reward_ratio:.2f} may not be profitable"
            )

        if self.llm.temperature > 0.3:
            warnings.append(
                "High LLM temperature may lead to inconsistent trading decisions"
            )

        return warnings

    def export_for_environment(self, target_env: Environment) -> dict[str, Any]:
        """Export configuration optimized for a specific environment."""
        config = self.model_dump()

        # Environment-specific adjustments
        if target_env == Environment.PRODUCTION:
            config["system"]["dry_run"] = False
            config["system"]["environment"] = Environment.PRODUCTION.value
            config["system"]["log_level"] = "INFO"
            config["exchange"]["cb_sandbox"] = False
            # Remove sensitive data
            for section in ["llm", "exchange", "system", "mcp", "omnisearch"]:
                if section in config:
                    for key in list(config[section].keys()):
                        if (
                            "key" in key.lower()
                            or "secret" in key.lower()
                            or "passphrase" in key.lower()
                            or "private" in key.lower()
                        ):
                            config[section][key] = None

        elif target_env == Environment.DEVELOPMENT:
            config["system"]["dry_run"] = True
            config["system"]["environment"] = Environment.DEVELOPMENT.value
            config["system"]["log_level"] = "DEBUG"
            config["exchange"]["cb_sandbox"] = True
            config["trading"]["leverage"] = min(config["trading"]["leverage"], 3)
            config["risk"]["max_daily_loss_pct"] = min(
                config["risk"]["max_daily_loss_pct"], 5.0
            )

        elif target_env == Environment.STAGING:
            config["system"]["dry_run"] = True
            config["system"]["environment"] = Environment.STAGING.value
            config["system"]["log_level"] = "INFO"
            config["exchange"]["cb_sandbox"] = True

        return config

    def get_effective_config(self) -> dict[str, Any]:
        """Get the effective configuration with all computed values."""
        base_config = self.model_dump()

        # Add computed fields
        base_config["computed"] = {
            "is_production": self.is_production,
            "requires_api_keys": self.requires_api_keys,
            "risk_reward_ratio": self.risk.default_take_profit_pct
            / self.risk.default_stop_loss_pct,
            "max_position_value": f"{self.trading.max_size_pct}% * {self.trading.leverage}x leverage",
            "effective_update_frequency": self.system.update_frequency_seconds,
        }

        return base_config

    def generate_config_hash(self) -> str:
        """Generate a hash of the current configuration for change detection."""
        import hashlib

        # Create a normalized config without secrets and timestamps
        config_for_hash = self.model_dump(
            exclude={
                "llm": {"openai_api_key", "anthropic_api_key"},
                "exchange": {
                    "cb_api_key",
                    "cb_api_secret",
                    "cb_passphrase",
                    "cdp_api_key_name",
                    "cdp_private_key",
                    "bluefin_private_key",
                },
                "system": {"api_secret_key", "instance_id"},
                "mcp": {"memory_api_key"},
                "omnisearch": {"api_key"},
            }
        )

        config_str = json.dumps(config_for_hash, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def create_settings(
    env_file: str | None = None, profile: TradingProfile | None = None, **overrides: Any
) -> Settings:
    """Factory function to create settings with optional overrides."""
    # Ensure .env file is loaded
    try:
        from dotenv import load_dotenv

        env_path = env_file or ".env"
        if Path(env_path).exists():
            load_dotenv(env_path)
    except ImportError:
        # dotenv not available, rely on pydantic-settings
        pass

    # Set environment file if provided
    if env_file:
        os.environ.setdefault("ENV_FILE", env_file)

    # Create base settings
    settings = Settings(**overrides)

    # Apply profile if specified
    if profile:
        settings = settings.apply_profile(profile)

    return settings


# Global settings instance
settings = create_settings()


def get_config_template() -> dict[str, Any]:
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
