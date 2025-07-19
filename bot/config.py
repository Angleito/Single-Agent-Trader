"""
Functional configuration system with backward compatibility.

This module provides a functional programming approach to configuration
while maintaining full backward compatibility with the existing Settings interface.
All configuration now uses functional programming types with proper validation.
"""

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

# Import functional programming types with lazy loading to avoid circular imports
from typing import TYPE_CHECKING, Any, Optional

from pydantic import SecretStr

# Import SecureString for private key handling
try:
    from bot.security.memory import SecureString
except ImportError:
    # Fallback to string if SecureString not available
    SecureString = str

if TYPE_CHECKING:
    from bot.fp.types.result import Failure, Result, Success
else:
    # Import for runtime use
    try:
        from bot.fp.types.result import Failure, Result, Success
    except ImportError:
        # Minimal fallback for missing dependencies
        Success, Failure, Result = None, None, None

    try:
        from bot.fp.types.base import (
            Money,
            Percentage,
            Symbol,
            TimeInterval,
            TradingMode,
        )
        from bot.fp.types.config import (
            APIKey,
            BacktestConfig,
            ExchangeConfig,
            FeatureFlags,
            LogLevel,
            PrivateKey,
            StrategyConfig,
            SystemConfig,
        )
        from bot.fp.types.config import (
            Config as FunctionalConfig,
        )
    except ImportError:
        # Minimal fallback for missing dependencies
        (
            Money,
            Percentage,
            Symbol,
            TimeInterval,
            TradingMode,
            APIKey,
            BacktestConfig,
            ExchangeConfig,
            FeatureFlags,
            LogLevel,
            PrivateKey,
            StrategyConfig,
            SystemConfig,
            FunctionalConfig,
        ) = (None,) * 14


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


def _get_functional_config():
    """Lazy load functional configuration types."""
    try:
        from bot.fp.types.config import Config as FunctionalConfig
        from bot.fp.types.result import Failure, Result, Success

        return (FunctionalConfig, Success, Failure, Result)
    except ImportError:
        return (None, None, None, None)


class ConfigError(Exception):
    """Configuration error."""


class ConfigValidationError(Exception):
    """Configuration validation error."""


class StartupValidator:
    """Comprehensive startup validation for the trading bot."""

    def __init__(self, settings: Settings):
        """Initialize startup validator with settings."""
        self.settings = settings
        self.validation_results: dict[str, list[str]] = {
            "environment_vars": [],
            "api_connectivity": [],
            "configuration": [],
            "system_dependencies": [],
            "file_permissions": [],
            "warnings": [],
            "errors": [],
            "critical_errors": [],
        }

    def validate_environment_variables(self) -> list[str]:
        """Validate all required environment variables are present."""
        issues = []

        # LLM provider specific validation
        if self.settings.llm.provider == "openai":
            if not self.settings.llm.openai_api_key:
                issues.append("OpenAI API key is required when using OpenAI provider")
        elif self.settings.llm.provider == "anthropic":
            if not self.settings.llm.anthropic_api_key:
                issues.append("Anthropic API key is required when using Anthropic provider")
        elif self.settings.llm.provider == "ollama" and not self.settings.llm.ollama_base_url:
            issues.append("Ollama base URL is required when using Ollama provider")

        # Exchange credentials validation (for live trading)
        if not self.settings.system.dry_run:
            if self.settings.exchange.exchange_type == "coinbase":
                if not self.settings.exchange.cdp_api_key_name:
                    issues.append("Coinbase CDP API key is required for live trading")
                if not self.settings.exchange.cdp_private_key:
                    issues.append("Coinbase CDP private key is required for live trading")
            elif self.settings.exchange.exchange_type == "bluefin":
                if not self.settings.exchange.bluefin_private_key:
                    issues.append("Bluefin private key is required for live trading")

        # Critical system settings
        if self.settings.system.environment == Environment.PRODUCTION:
            if self.settings.system.dry_run:
                issues.append("Production environment should not use dry run mode")

        self.validation_results["environment_vars"] = issues
        return issues

    def validate_configuration_integrity(self) -> list[str]:
        """Validate configuration parameter integrity and consistency."""
        issues = []

        # Trading parameter validation
        if self.settings.trading.leverage > 20:
            issues.append("Leverage above 20x is extremely risky")

        if self.settings.trading.max_size_pct > 50:
            issues.append("Position size above 50% is extremely risky")

        # Risk management validation
        if self.settings.risk.default_stop_loss_pct >= self.settings.risk.default_take_profit_pct:
            issues.append("Stop loss should be smaller than take profit for positive risk/reward")

        if self.settings.risk.max_concurrent_trades > 20:
            issues.append("Too many concurrent trades may lead to overexposure")

        # Data configuration validation
        if self.settings.data.candle_limit < 50:
            issues.append("Candle limit below 50 may not provide enough data for analysis")

        if self.settings.data.candle_limit > 1000:
            issues.append("Candle limit above 1000 may impact performance")

        self.validation_results["configuration"] = issues
        return issues

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run all validation checks and return comprehensive results."""
        # Run validation checks
        env_issues = self.validate_environment_variables()
        config_issues = self.validate_configuration_integrity()

        # Categorize issues by severity
        critical_issues = []
        warning_issues = []

        for issue in env_issues:
            if any(keyword in issue.lower() for keyword in ["required", "failed", "cannot"]):
                critical_issues.append(issue)
            else:
                warning_issues.append(issue)

        # Configuration issues are typically warnings unless critical
        for issue in config_issues:
            if any(keyword in issue.lower() for keyword in ["extremely risky"]):
                critical_issues.append(issue)
            else:
                warning_issues.append(issue)

        # Build final results
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "valid": len(critical_issues) == 0,
            "critical_errors": critical_issues,
            "warnings": warning_issues,
            "details": self.validation_results,
            "configuration_summary": {
                "environment": self.settings.system.environment.value,
                "dry_run": self.settings.system.dry_run,
                "profile": getattr(self.settings, 'profile', TradingProfile.BALANCED).value,
                "llm_provider": self.settings.llm.provider,
                "trading_symbol": self.settings.trading.symbol,
                "leverage": self.settings.trading.leverage,
            },
        }

        return results


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
            # Load from kwargs first (overrides), then environment, then defaults
            self.symbol: str = kwargs.get("symbol") or os.getenv(
                "TRADING__SYMBOL", "BTC-USD"
            )
            self.interval: str = kwargs.get("interval") or os.getenv(
                "TRADING__INTERVAL", "1m"
            )
            self.leverage: int = (
                kwargs.get("leverage")
                if kwargs.get("leverage") is not None
                else parse_int_env("TRADING__LEVERAGE", 5)
            )
            self.max_size_pct: float = (
                kwargs.get("max_size_pct")
                if kwargs.get("max_size_pct") is not None
                else parse_float_env("TRADING__MAX_SIZE_PCT", 20.0)
            )
            self.order_timeout_seconds: int = parse_int_env(
                "TRADING__ORDER_TIMEOUT_SECONDS",
                kwargs.get("order_timeout_seconds", 30),
            )
            self.slippage_tolerance_pct: float = parse_float_env(
                "TRADING__SLIPPAGE_TOLERANCE_PCT",
                kwargs.get("slippage_tolerance_pct", 0.1),
            )
            self.min_profit_pct: float = parse_float_env(
                "TRADING__MIN_PROFIT_PCT", kwargs.get("min_profit_pct", 0.5)
            )
            self.maker_fee_rate: float = parse_float_env(
                "TRADING__MAKER_FEE_RATE", kwargs.get("maker_fee_rate", 0.004)
            )
            self.taker_fee_rate: float = parse_float_env(
                "TRADING__TAKER_FEE_RATE", kwargs.get("taker_fee_rate", 0.006)
            )
            self.futures_fee_rate: float = parse_float_env(
                "TRADING__FUTURES_FEE_RATE", kwargs.get("futures_fee_rate", 0.0015)
            )
            self.min_trading_interval_seconds: int = parse_int_env(
                "TRADING__MIN_TRADING_INTERVAL_SECONDS",
                kwargs.get("min_trading_interval_seconds", 60),
            )
            self.require_24h_data_before_trading: bool = parse_bool_env(
                "TRADING__REQUIRE_24H_DATA_BEFORE_TRADING",
                kwargs.get("require_24h_data_before_trading", True),
            )
            self.min_candles_for_trading: int = parse_int_env(
                "TRADING__MIN_CANDLES_FOR_TRADING",
                kwargs.get("min_candles_for_trading", 100),
            )
            self.enable_futures: bool = parse_bool_env(
                "TRADING__ENABLE_FUTURES", kwargs.get("enable_futures", True)
            )
            self.futures_account_type: str = kwargs.get(
                "futures_account_type"
            ) or os.getenv("TRADING__FUTURES_ACCOUNT_TYPE", "CFM")
            self.auto_cash_transfer: bool = parse_bool_env(
                "TRADING__AUTO_CASH_TRANSFER", kwargs.get("auto_cash_transfer", True)
            )
            self.max_futures_leverage: int = parse_int_env(
                "TRADING__MAX_FUTURES_LEVERAGE",
                kwargs.get("max_futures_leverage", 20),
            )

    def _from_functional_config(self, _config: Any) -> None:
        """Extract settings from functional configuration."""
        # Set defaults compatible with existing interface
        self.symbol = "BTC-USD"
        self.interval = "1m"
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
        self.futures_account_type = "CFM"
        self.auto_cash_transfer = True
        self.max_futures_leverage = 20


@dataclass
class LLMSettings:
    """LLM configuration settings using functional types."""

    def __init__(self, functional_config: Any = None, **kwargs) -> None:
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Load from kwargs first (overrides), then environment, then defaults
            self.provider: str = kwargs.get("provider") or os.getenv(
                "LLM__PROVIDER", "openai"
            )
            self.model_name: str = kwargs.get("model_name") or os.getenv(
                "LLM__MODEL_NAME", "gpt-4"
            )
            self.temperature: float = parse_float_env(
                "LLM__TEMPERATURE", kwargs.get("temperature", 0.1)
            )
            self.max_tokens: int = parse_int_env(
                "LLM__MAX_TOKENS", kwargs.get("max_tokens", 30000)
            )
            self.request_timeout: int = parse_int_env(
                "LLM__REQUEST_TIMEOUT", kwargs.get("request_timeout", 30)
            )
            self.max_retries: int = parse_int_env(
                "LLM__MAX_RETRIES", kwargs.get("max_retries", 3)
            )
            self.openai_api_key: SecretStr | None = None

            # Load from environment
            api_key = os.getenv("LLM__OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_api_key = SecretStr(api_key)

    def _from_functional_config(self, config: Any) -> None:
        """Extract settings from functional configuration."""
        # Avoid imports to prevent circular dependencies
        # Check if config has LLM-specific attributes
        if hasattr(config, "model_name") and hasattr(config, "temperature"):
            self.provider = "openai"
            self.model_name = config.model_name
            self.temperature = config.temperature
            self.max_tokens = config.max_context_length
            self.request_timeout = 30
            self.max_retries = 3
            self.openai_api_key = None

            # Load API key from environment
            api_key = os.getenv("LLM__OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_api_key = SecretStr(api_key)
        else:
            # Set defaults for non-LLM strategies
            self.provider = "openai"
            self.model_name = "gpt-4"
            self.temperature = 0.1
            self.max_tokens = 30000
            self.request_timeout = 30
            self.max_retries = 3
            self.openai_api_key = None


@dataclass
class ExchangeSettings:
    """Exchange configuration settings using functional types."""

    def __init__(
        self, functional_config: Optional["ExchangeConfig"] = None, **kwargs
    ) -> None:
        # Initialize credentials to None first
        self.cb_api_key: SecretStr | None = None
        self.cb_api_secret: SecretStr | None = None
        self.cb_passphrase: SecretStr | None = None
        self.cdp_api_key_name: SecretStr | None = None
        self.cdp_private_key: SecretStr | None = None
        self.bluefin_private_key: SecretStr | None = None

        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Load from kwargs first (overrides), then environment, then defaults
            self.exchange_type: str = kwargs.get("exchange_type") or os.getenv(
                "EXCHANGE__EXCHANGE_TYPE", "coinbase"
            )
            self.cb_sandbox: bool = parse_bool_env(
                "EXCHANGE__CB_SANDBOX", kwargs.get("cb_sandbox", True)
            )
            self.api_timeout: int = parse_int_env(
                "EXCHANGE__API_TIMEOUT", kwargs.get("api_timeout", 10)
            )
            self.rate_limit_requests: int = parse_int_env(
                "EXCHANGE__RATE_LIMIT_REQUESTS", kwargs.get("rate_limit_requests", 10)
            )
            self.rate_limit_window_seconds: int = parse_int_env(
                "EXCHANGE__RATE_LIMIT_WINDOW_SECONDS",
                kwargs.get("rate_limit_window_seconds", 60),
            )
            self.bluefin_network: str = kwargs.get("bluefin_network") or os.getenv(
                "EXCHANGE__BLUEFIN_NETWORK", "mainnet"
            )

            # WebSocket settings
            self.websocket_reconnect_attempts: int = parse_int_env(
                "EXCHANGE__WEBSOCKET_RECONNECT_ATTEMPTS",
                kwargs.get("websocket_reconnect_attempts", 5),
            )
            self.websocket_timeout: int = parse_int_env(
                "EXCHANGE__WEBSOCKET_TIMEOUT", kwargs.get("websocket_timeout", 30)
            )

            # Load credentials from environment
            if os.getenv("EXCHANGE__CB_API_KEY"):
                self.cb_api_key = SecretStr(os.getenv("EXCHANGE__CB_API_KEY"))
            if os.getenv("EXCHANGE__CB_API_SECRET"):
                self.cb_api_secret = SecretStr(os.getenv("EXCHANGE__CB_API_SECRET"))
            if os.getenv("EXCHANGE__CB_PASSPHRASE"):
                self.cb_passphrase = SecretStr(os.getenv("EXCHANGE__CB_PASSPHRASE"))
            if os.getenv("EXCHANGE__CDP_API_KEY_NAME"):
                self.cdp_api_key_name = SecretStr(
                    os.getenv("EXCHANGE__CDP_API_KEY_NAME")
                )
            if os.getenv("EXCHANGE__CDP_PRIVATE_KEY"):
                self.cdp_private_key = SecretStr(os.getenv("EXCHANGE__CDP_PRIVATE_KEY"))
            if os.getenv("EXCHANGE__BLUEFIN_PRIVATE_KEY"):
                # Auto-convert Sui private key format if needed
                raw_key = os.getenv("EXCHANGE__BLUEFIN_PRIVATE_KEY")
                converted_key = self._convert_sui_private_key_legacy(raw_key)
                self.bluefin_private_key = SecretStr(converted_key)

    def _from_functional_config(self, config: "ExchangeConfig") -> None:
        """Extract settings from functional configuration."""
        # Avoid imports to prevent circular dependencies
        # Check exchange type by attributes
        if (
            hasattr(config, "api_key")
            and hasattr(config, "private_key")
            and hasattr(config, "api_url")
        ):
            self.exchange_type = "coinbase"
            self.cb_sandbox = True  # Default for safety
            self.api_timeout = 10
            self.rate_limit_requests = config.rate_limits.requests_per_minute
            self.rate_limit_window_seconds = 60
            self.bluefin_network = "mainnet"

            # Convert functional types to compatibility types
            self.cdp_api_key_name = (
                SecretStr(config.api_key._value) if config.api_key else None
            )
            self.cdp_private_key = (
                SecretStr(config.private_key._value) if config.private_key else None
            )

        elif hasattr(config, "private_key") and hasattr(config, "network"):
            # Bluefin exchange
            self.exchange_type = "bluefin"
            self.cb_sandbox = True
            self.api_timeout = 10
            self.rate_limit_requests = config.rate_limits.requests_per_minute
            self.rate_limit_window_seconds = 60
            self.bluefin_network = config.network

            # Convert functional types
            self.bluefin_private_key = (
                SecretStr(config.private_key._value) if config.private_key else None
            )

        elif hasattr(config, "testnet"):
            # Binance exchange
            self.exchange_type = "binance"
            self.cb_sandbox = config.testnet
            self.api_timeout = 10
            self.rate_limit_requests = config.rate_limits.requests_per_minute
            self.rate_limit_window_seconds = 60
            self.bluefin_network = "mainnet"
        else:
            # Default/unknown exchange type
            self.exchange_type = "coinbase"
            self.cb_sandbox = True
            self.api_timeout = 10
            self.rate_limit_requests = 10
            self.rate_limit_window_seconds = 60
            self.bluefin_network = "mainnet"

        # Set defaults for missing credentials
        self.cb_api_key = self.cb_api_key or None
        self.cb_api_secret = self.cb_api_secret or None
        self.cb_passphrase = self.cb_passphrase or None
        self.cdp_api_key_name = self.cdp_api_key_name or None
        self.cdp_private_key = self.cdp_private_key or None
        self.bluefin_private_key = self.bluefin_private_key or None

    def _convert_sui_private_key_legacy(self, private_key_str: str) -> str:
        """Convert Sui private key from any format to hex format for legacy config.

        Args:
            private_key_str: Private key in any supported format

        Returns:
            Hex format private key

        Raises:
            ValueError: If the key format is invalid or cannot be converted
        """
        try:
            # Import the converter utility
            from bot.utils.sui_key_converter import (
                auto_convert_private_key,
            )

            converted_key, format_detected, message = auto_convert_private_key(
                private_key_str
            )

            if converted_key is None:
                raise ValueError(f"Invalid Sui private key format: {message}")

            return converted_key
        except ImportError:
            # Fallback implementation if converter module is not available
            private_key_str = private_key_str.strip()

            # Check if already in hex format
            if private_key_str.startswith("0x") and len(private_key_str) == 66:
                return private_key_str
            if len(private_key_str) == 64 and all(
                c in "0123456789abcdefABCDEF" for c in private_key_str
            ):
                return f"0x{private_key_str}"

            # Check if bech32 format and attempt automatic conversion
            if private_key_str.startswith("suiprivkey"):
                # Try to automatically convert the bech32 key
                try:
                    from bot.utils.sui_key_converter import (
                        bech32_to_hex,
                    )

                    print(
                        "🔄 Bech32 format detected, attempting automatic conversion..."
                    )
                    converted_key = bech32_to_hex(private_key_str)
                    if converted_key:
                        print("✅ Successfully converted bech32 to hex format")
                        return SecureString(converted_key)
                    print("❌ Automatic conversion failed")
                except ImportError:
                    print("⚠️ Converter utility not available")

                # If conversion failed, provide manual instructions
                raise ValueError(
                    "Bech32 format detected (suiprivkey...) but automatic conversion failed.\n"
                    "Please convert manually:\n"
                    "1. Open your Sui wallet → Settings → Export Private Key\n"
                    "2. Choose 'Raw Private Key' or 'Hex Format'\n"
                    "3. Copy the hex string (should start with 0x)\n"
                    "4. Update your EXCHANGE__BLUEFIN_PRIVATE_KEY in .env with the hex format"
                ) from None

            # Check if mnemonic format and attempt automatic conversion
            words = private_key_str.split()
            if len(words) in [12, 24] and all(word.isalpha() for word in words):
                # Try to automatically convert the mnemonic
                try:
                    from bot.utils.sui_key_converter import (
                        mnemonic_to_hex,
                    )

                    print(
                        "🔄 Mnemonic phrase detected, attempting automatic conversion..."
                    )
                    converted_key = mnemonic_to_hex(private_key_str)
                    if converted_key:
                        print("✅ Successfully converted mnemonic to hex format")
                        return SecureString(converted_key)
                    print("❌ Automatic conversion failed")
                except ImportError:
                    print("⚠️ Converter utility not available")

                # If conversion failed, provide manual instructions
                raise ValueError(
                    "Mnemonic phrase detected but automatic conversion failed.\n"
                    "Please convert manually:\n"
                    '1. Use Sui CLI: sui keytool import "<your mnemonic>" ed25519\n'
                    "2. Then export as hex: sui keytool export <address> --key-scheme ed25519\n"
                    "3. Update your EXCHANGE__BLUEFIN_PRIVATE_KEY in .env with the hex format"
                ) from None

            # Unknown format
            raise ValueError(
                "Unknown private key format. Supported formats:\n"
                "• Hex: 0x1234...abcd (64 hex characters with 0x prefix)\n"
                "• Bech32: suiprivkey... (Sui wallet export format)\n"
                "• Mnemonic: 12 or 24 word seed phrase"
            ) from None


@dataclass
class RiskSettings:
    """Risk management settings using functional types."""

    def __init__(self, functional_config: Optional["SystemConfig"] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.max_daily_loss_pct: float = parse_float_env(
                "RISK__MAX_DAILY_LOSS_PCT", kwargs.get("max_daily_loss_pct", 5.0)
            )
            self.max_concurrent_trades: int = parse_int_env(
                "RISK__MAX_CONCURRENT_TRADES", kwargs.get("max_concurrent_trades", 3)
            )
            self.default_stop_loss_pct: float = parse_float_env(
                "RISK__DEFAULT_STOP_LOSS_PCT", kwargs.get("default_stop_loss_pct", 2.0)
            )
            self.default_take_profit_pct: float = parse_float_env(
                "RISK__DEFAULT_TAKE_PROFIT_PCT",
                kwargs.get("default_take_profit_pct", 4.0),
            )

    def _from_functional_config(self, config: "SystemConfig") -> None:
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
        self.keep_days: int = parse_int_env(
            "DATA__KEEP_DAYS", kwargs.get("keep_days", 30)
        )
        self.backup_enabled: bool = parse_bool_env(
            "DATA__BACKUP_ENABLED", kwargs.get("backup_enabled", True)
        )
        self.candle_limit: int = parse_int_env(
            "DATA__CANDLE_LIMIT", kwargs.get("candle_limit", 1000)
        )
        self.indicator_warmup: int = parse_int_env(
            "DATA__INDICATOR_WARMUP", kwargs.get("indicator_warmup", 100)
        )
        self.data_cache_ttl_seconds: int = parse_int_env(
            "DATA__DATA_CACHE_TTL_SECONDS", kwargs.get("data_cache_ttl_seconds", 30)
        )
        self.use_real_data: bool = parse_bool_env(
            "DATA__USE_REAL_DATA", kwargs.get("use_real_data", True)
        )


@dataclass
class DominanceSettings:
    """Market dominance settings using functional types."""

    def __init__(self, **kwargs):
        # Maintain exact compatibility with environment variable loading
        self.enabled: bool = parse_bool_env(
            "DOMINANCE__ENABLED", kwargs.get("enabled", False)
        )
        self.threshold: float = parse_float_env(
            "DOMINANCE__THRESHOLD", kwargs.get("threshold", 0.45)
        )


@dataclass
class SystemSettings:
    """System configuration settings using functional types."""

    def __init__(self, functional_config: Optional["SystemConfig"] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Load from kwargs first (overrides), then environment, then defaults
            self.dry_run: bool = (
                kwargs.get("dry_run")
                if kwargs.get("dry_run") is not None
                else parse_bool_env("SYSTEM__DRY_RUN", True)
            )
            self.environment: str = kwargs.get("environment") or os.getenv(
                "SYSTEM__ENVIRONMENT", "development"
            )
            self.log_level: str = kwargs.get("log_level") or os.getenv(
                "SYSTEM__LOG_LEVEL", "INFO"
            )
            self.update_frequency_seconds: float = parse_float_env(
                "SYSTEM__UPDATE_FREQUENCY_SECONDS",
                kwargs.get("update_frequency_seconds", 30.0),
            )

            # Concurrency optimization settings
            self.max_concurrent_tasks: int = parse_int_env(
                "SYSTEM__MAX_CONCURRENT_TASKS",
                kwargs.get("max_concurrent_tasks", 4),
            )
            self.thread_pool_size: int = parse_int_env(
                "SYSTEM__THREAD_POOL_SIZE",
                kwargs.get("thread_pool_size", 2),
            )
            self.async_timeout: float = parse_float_env(
                "SYSTEM__ASYNC_TIMEOUT",
                kwargs.get("async_timeout", 15.0),
            )
            self.task_batch_size: int = parse_int_env(
                "SYSTEM__TASK_BATCH_SIZE",
                kwargs.get("task_batch_size", 2),
            )

            # Add missing logging attributes
            self.log_to_console: bool = parse_bool_env(
                "SYSTEM__LOG_TO_CONSOLE",
                kwargs.get("log_to_console", True),
            )
            self.log_to_file: bool = parse_bool_env(
                "SYSTEM__LOG_TO_FILE",
                kwargs.get("log_to_file", True),
            )
            self.log_file_path: str = kwargs.get("log_file_path") or os.getenv(
                "SYSTEM__LOG_FILE_PATH", "logs/bot.log"
            )
            self.max_log_size_mb: int = int(
                parse_float_env(
                    "SYSTEM__MAX_LOG_SIZE_MB",
                    kwargs.get("max_log_size_mb", 100),
                )
            )
            self.log_backup_count: int = int(
                parse_float_env(
                    "SYSTEM__LOG_BACKUP_COUNT",
                    kwargs.get("log_backup_count", 5),
                )
            )

    def _from_functional_config(self, config: "SystemConfig") -> None:
        """Extract settings from functional configuration."""
        self.dry_run = config.mode == TradingMode.PAPER
        self.environment = (
            "production" if config.mode == TradingMode.LIVE else "development"
        )
        self.log_level = config.log_level.value
        self.update_frequency_seconds = 30.0  # Default

        # Add missing logging attributes with functional config defaults
        self.log_to_console = True  # Default for functional config
        self.log_to_file = True  # Default for functional config
        self.log_file_path = "logs/bot.log"  # Default
        self.max_log_size_mb = 100  # Default
        self.log_backup_count = 5  # Default


@dataclass
class PaperTradingSettings:
    """Paper trading settings using functional types."""

    def __init__(self, functional_config: Optional["BacktestConfig"] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.starting_balance: float = parse_float_env(
                "PAPER_TRADING__STARTING_BALANCE",
                kwargs.get("starting_balance", 10000.0),
            )
            self.fee_rate: float = parse_float_env(
                "PAPER_TRADING__FEE_RATE", kwargs.get("fee_rate", 0.001)
            )
            self.slippage_rate: float = parse_float_env(
                "PAPER_TRADING__SLIPPAGE_RATE", kwargs.get("slippage_rate", 0.0005)
            )
            self.enable_daily_reports: bool = parse_bool_env(
                "PAPER_TRADING__ENABLE_DAILY_REPORTS",
                kwargs.get("enable_daily_reports", True),
            )
            self.enable_weekly_summaries: bool = parse_bool_env(
                "PAPER_TRADING__ENABLE_WEEKLY_SUMMARIES",
                kwargs.get("enable_weekly_summaries", True),
            )
            self.track_drawdown: bool = parse_bool_env(
                "PAPER_TRADING__TRACK_DRAWDOWN", kwargs.get("track_drawdown", True)
            )
            self.keep_trade_history_days: int = parse_int_env(
                "PAPER_TRADING__KEEP_TRADE_HISTORY_DAYS",
                kwargs.get("keep_trade_history_days", 90),
            )
            self.export_trade_data: bool = parse_bool_env(
                "PAPER_TRADING__EXPORT_TRADE_DATA",
                kwargs.get("export_trade_data", False),
            )
            self.report_time_utc: str = os.getenv(
                "PAPER_TRADING__REPORT_TIME_UTC", kwargs.get("report_time_utc", "23:59")
            )
            self.include_unrealized_pnl: bool = parse_bool_env(
                "PAPER_TRADING__INCLUDE_UNREALIZED_PNL",
                kwargs.get("include_unrealized_pnl", True),
            )

    def _from_functional_config(self, config: "BacktestConfig") -> None:
        """Extract settings from functional configuration."""
        self.starting_balance = float(config.initial_capital.amount)
        self.fee_rate = config.fee_structure.taker_fee.as_ratio()
        self.slippage_rate = config.slippage.as_ratio()
        self.enable_daily_reports = True
        self.enable_weekly_summaries = True
        self.track_drawdown = True
        self.keep_trade_history_days = 90
        self.export_trade_data = False
        self.report_time_utc = "23:59"
        self.include_unrealized_pnl = True


@dataclass
class MonitoringSettings:
    """System monitoring settings using functional types."""

    def __init__(self, functional_config: Optional["SystemConfig"] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.enabled: bool = parse_bool_env(
                "MONITORING__ENABLED", kwargs.get("enabled", True)
            )
            self.check_interval: int = parse_int_env(
                "MONITORING__CHECK_INTERVAL", kwargs.get("check_interval", 60)
            )

    def _from_functional_config(self, config: "SystemConfig") -> None:
        """Extract settings from functional configuration."""
        self.enabled = config.features.enable_metrics
        self.check_interval = 60


@dataclass
class MCPSettings:
    """MCP (Model Context Protocol) settings using functional types."""

    def __init__(self, functional_config: Optional["SystemConfig"] = None, **kwargs):
        if functional_config:
            self._from_functional_config(functional_config)
        else:
            # Maintain exact compatibility with environment variable loading
            self.enabled: bool = parse_bool_env(
                "MCP_ENABLED", kwargs.get("enabled", False)
            )
            self.server_url: str = os.getenv(
                "MCP_SERVER_URL", kwargs.get("server_url", "http://localhost:8765")
            )

    def _from_functional_config(self, config: "SystemConfig") -> None:
        """Extract settings from functional configuration."""
        self.enabled = config.features.enable_memory
        self.server_url = "http://localhost:8765"


@dataclass
class OrderbookSettings:
    """Orderbook configuration settings using functional types."""

    def __init__(self, **kwargs):
        # Maintain exact compatibility with environment variable loading
        self.depth_levels: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__DEPTH_LEVELS", kwargs.get("depth_levels", 20)
        )
        self.refresh_interval_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__REFRESH_INTERVAL_MS",
            kwargs.get("refresh_interval_ms", 100),
        )
        self.max_age_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__MAX_AGE_MS", kwargs.get("max_age_ms", 1000)
        )
        self.min_liquidity_threshold: str = os.getenv(
            "MARKET_MAKING__ORDERBOOK__MIN_LIQUIDITY_THRESHOLD",
            kwargs.get("min_liquidity_threshold", "500"),
        )
        self.max_spread_bps: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__MAX_SPREAD_BPS",
            kwargs.get("max_spread_bps", 200),
        )
        self.quality_threshold: float = parse_float_env(
            "MARKET_MAKING__ORDERBOOK__QUALITY_THRESHOLD",
            kwargs.get("quality_threshold", 0.8),
        )
        self.staleness_threshold_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__STALENESS_THRESHOLD_MS",
            kwargs.get("staleness_threshold_ms", 2000),
        )
        self.aggregation_levels: list[int] = kwargs.get(
            "aggregation_levels", [1, 5, 10, 20]
        )
        self.price_precision: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__PRICE_PRECISION",
            kwargs.get("price_precision", 6),
        )
        self.size_precision: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__SIZE_PRECISION", kwargs.get("size_precision", 4)
        )
        self.enable_snapshot_recovery: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__ENABLE_SNAPSHOT_RECOVERY",
            kwargs.get("enable_snapshot_recovery", True),
        )
        self.snapshot_recovery_interval_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__SNAPSHOT_RECOVERY_INTERVAL_MS",
            kwargs.get("snapshot_recovery_interval_ms", 5000),
        )
        self.enable_incremental_updates: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__ENABLE_INCREMENTAL_UPDATES",
            kwargs.get("enable_incremental_updates", True),
        )
        self.buffer_size: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__BUFFER_SIZE", kwargs.get("buffer_size", 1000)
        )
        self.compression_enabled: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__COMPRESSION_ENABLED",
            kwargs.get("compression_enabled", False),
        )
        self.websocket_timeout_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__WEBSOCKET_TIMEOUT_MS",
            kwargs.get("websocket_timeout_ms", 30000),
        )
        self.heartbeat_interval_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__HEARTBEAT_INTERVAL_MS",
            kwargs.get("heartbeat_interval_ms", 15000),
        )
        self.reconnect_delay_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__RECONNECT_DELAY_MS",
            kwargs.get("reconnect_delay_ms", 1000),
        )
        self.max_reconnect_attempts: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__MAX_RECONNECT_ATTEMPTS",
            kwargs.get("max_reconnect_attempts", 10),
        )
        self.enable_order_flow_analysis: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__ENABLE_ORDER_FLOW_ANALYSIS",
            kwargs.get("enable_order_flow_analysis", True),
        )
        self.imbalance_detection_threshold: float = parse_float_env(
            "MARKET_MAKING__ORDERBOOK__IMBALANCE_DETECTION_THRESHOLD",
            kwargs.get("imbalance_detection_threshold", 0.3),
        )
        # Market data validation settings
        self.enable_price_validation: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__ENABLE_PRICE_VALIDATION",
            kwargs.get("enable_price_validation", True),
        )
        self.max_price_deviation_pct: float = parse_float_env(
            "MARKET_MAKING__ORDERBOOK__MAX_PRICE_DEVIATION_PCT",
            kwargs.get("max_price_deviation_pct", 5.0),
        )
        self.enable_size_validation: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__ENABLE_SIZE_VALIDATION",
            kwargs.get("enable_size_validation", True),
        )
        self.min_order_size: str = os.getenv(
            "MARKET_MAKING__ORDERBOOK__MIN_ORDER_SIZE",
            kwargs.get("min_order_size", "10"),
        )
        self.max_order_size: str = os.getenv(
            "MARKET_MAKING__ORDERBOOK__MAX_ORDER_SIZE",
            kwargs.get("max_order_size", "50000"),
        )
        self.enable_time_validation: bool = parse_bool_env(
            "MARKET_MAKING__ORDERBOOK__ENABLE_TIME_VALIDATION",
            kwargs.get("enable_time_validation", True),
        )
        self.max_timestamp_drift_ms: int = parse_int_env(
            "MARKET_MAKING__ORDERBOOK__MAX_TIMESTAMP_DRIFT_MS",
            kwargs.get("max_timestamp_drift_ms", 5000),
        )


@dataclass
class OmniSearchSettings:
    """OmniSearch integration settings using functional types."""

    def __init__(self, **kwargs):
        # Maintain exact compatibility with environment variable loading
        self.enabled: bool = parse_bool_env(
            "OMNISEARCH__ENABLED", kwargs.get("enabled", False)
        )
        self.server_url: str = os.getenv(
            "OMNISEARCH__SERVER_URL", kwargs.get("server_url", "http://localhost:8766")
        )
        self.max_results: int = parse_int_env(
            "OMNISEARCH__MAX_RESULTS", kwargs.get("max_results", 5)
        )
        self.cache_ttl_seconds: int = parse_int_env(
            "OMNISEARCH__CACHE_TTL_SECONDS", kwargs.get("cache_ttl_seconds", 300)
        )
        self.rate_limit_requests_per_minute: int = parse_int_env(
            "OMNISEARCH__RATE_LIMIT_REQUESTS_PER_MINUTE",
            kwargs.get("rate_limit_requests_per_minute", 10),
        )
        self.timeout_seconds: int = parse_int_env(
            "OMNISEARCH__TIMEOUT_SECONDS", kwargs.get("timeout_seconds", 30)
        )
        self.enable_crypto_sentiment: bool = parse_bool_env(
            "OMNISEARCH__ENABLE_CRYPTO_SENTIMENT",
            kwargs.get("enable_crypto_sentiment", True),
        )
        self.enable_nasdaq_sentiment: bool = parse_bool_env(
            "OMNISEARCH__ENABLE_NASDAQ_SENTIMENT",
            kwargs.get("enable_nasdaq_sentiment", True),
        )
        self.enable_correlation_analysis: bool = parse_bool_env(
            "OMNISEARCH__ENABLE_CORRELATION_ANALYSIS",
            kwargs.get("enable_correlation_analysis", True),
        )


class Settings:
    """Main configuration settings with functional programming foundation."""

    def __init__(
        self, functional_config: Optional["FunctionalConfig"] = None, **overrides
    ):
        """Initialize settings with optional functional config and overrides."""
        # Create compatibility sections using functional config if available
        if functional_config:
            self.trading = TradingSettings(
                functional_config.strategy, **overrides.get("trading", {})
            )
            self.llm = LLMSettings(
                functional_config.strategy, **overrides.get("llm", {})
            )
            self.exchange = ExchangeSettings(
                functional_config.exchange, **overrides.get("exchange", {})
            )
            self.risk = RiskSettings(
                functional_config.system, **overrides.get("risk", {})
            )
            self.data = DataSettings(**overrides.get("data", {}))
            self.dominance = DominanceSettings(**overrides.get("dominance", {}))
            self.system = SystemSettings(
                functional_config.system, **overrides.get("system", {})
            )
            self.paper_trading = PaperTradingSettings(
                functional_config.backtest, **overrides.get("paper_trading", {})
            )
            self.monitoring = MonitoringSettings(
                functional_config.system, **overrides.get("monitoring", {})
            )
            self.mcp = MCPSettings(functional_config.system, **overrides.get("mcp", {}))
            self.orderbook = OrderbookSettings(**overrides.get("orderbook", {}))
            self.omnisearch = OmniSearchSettings(**overrides.get("omnisearch", {}))
        else:
            # Fallback to environment/kwargs based initialization
            self.trading = TradingSettings(**overrides.get("trading", {}))
            self.llm = LLMSettings(**overrides.get("llm", {}))
            self.exchange = ExchangeSettings(**overrides.get("exchange", {}))
            self.risk = RiskSettings(**overrides.get("risk", {}))
            self.data = DataSettings(**overrides.get("data", {}))
            self.dominance = DominanceSettings(**overrides.get("dominance", {}))
            self.system = SystemSettings(**overrides.get("system", {}))
            self.paper_trading = PaperTradingSettings(
                **overrides.get("paper_trading", {})
            )
            self.monitoring = MonitoringSettings(**overrides.get("monitoring", {}))
            self.mcp = MCPSettings(**overrides.get("mcp", {}))
            self.orderbook = OrderbookSettings(**overrides.get("orderbook", {}))
            self.omnisearch = OmniSearchSettings(**overrides.get("omnisearch", {}))

        # Store functional config for advanced use cases
        self._functional_config = functional_config
        
        # Set default profile
        self.profile = TradingProfile.BALANCED

    def apply_profile(self, profile: str | TradingProfile) -> "Settings":
        """Apply a configuration profile using functional composition."""
        # Convert string to TradingProfile enum if needed
        if isinstance(profile, str):
            try:
                profile_enum = TradingProfile(profile.lower())
            except ValueError:
                # Fallback to balanced if invalid profile
                profile_enum = TradingProfile.BALANCED
        else:
            profile_enum = profile
            
        # Load profile-specific overrides and create new Settings instance
        profile_overrides = self._load_profile_overrides(profile_enum.value)
        new_settings = Settings(self._functional_config, **profile_overrides)
        new_settings.profile = profile_enum
        return new_settings

    def _load_profile_overrides(self, profile: str) -> dict[str, Any]:
        """Load profile-specific configuration overrides."""
        profile_configs = {
            "conservative": {
                "trading": {"leverage": 2, "max_size_pct": 10.0},
                "risk": {"max_daily_loss_pct": 2.0, "max_concurrent_trades": 1},
                "llm": {"temperature": 0.05},
            },
            "aggressive": {
                "trading": {"leverage": 10, "max_size_pct": 50.0},
                "risk": {"max_daily_loss_pct": 10.0, "max_concurrent_trades": 5},
                "llm": {"temperature": 0.2},
            },
            "balanced": {
                "trading": {"leverage": 5, "max_size_pct": 25.0},
                "risk": {"max_daily_loss_pct": 5.0, "max_concurrent_trades": 3},
                "llm": {"temperature": 0.1},
            },
        }
        return profile_configs.get(profile, {})

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "trading": self.trading.__dict__,
            "llm": self.llm.__dict__,
            "exchange": self.exchange.__dict__,
            "risk": self.risk.__dict__,
            "data": self.data.__dict__,
            "dominance": self.dominance.__dict__,
            "system": self.system.__dict__,
            "paper_trading": self.paper_trading.__dict__,
            "monitoring": self.monitoring.__dict__,
            "mcp": self.mcp.__dict__,
            "orderbook": self.orderbook.__dict__,
            "omnisearch": self.omnisearch.__dict__,
        }


def create_settings(
    env_file: str | None = None,
    overrides: dict[str, Any] | None = None,
    profile: str | None = None,
) -> Settings:
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

    return settings


def load_settings_from_file(file_path: str | Path) -> Settings:
    """Load settings from a configuration file using functional patterns."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            # Fallback to default settings if file not found
            return create_settings()

        with open(file_path) as f:
            if file_path.suffix.lower() == ".json":
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
        return Settings(functional_config, **config_data)

    except Exception:
        # Fallback to default settings on any error
        return create_settings()


def _create_functional_config_from_dict(config_data: dict[str, Any]) -> Any:
    """Create functional configuration from dictionary data."""
    try:
        # Set environment variables temporarily for functional config builders
        original_env = {}

        # Map config data to environment variables
        env_mapping = {
            # Trading settings
            "TRADING_PAIRS": config_data.get("trading", {}).get("symbol", "BTC-USD"),
            "TRADING_INTERVAL": config_data.get("trading", {}).get("interval", "1m"),
            "TRADING_MODE": (
                "paper"
                if config_data.get("system", {}).get("dry_run", True)
                else "live"
            ),
            "LOG_LEVEL": config_data.get("system", {}).get("log_level", "INFO"),
            "MAX_CONCURRENT_POSITIONS": str(
                config_data.get("risk", {}).get("max_concurrent_trades", 3)
            ),
            "DEFAULT_POSITION_SIZE": str(
                config_data.get("trading", {}).get("max_size_pct", 20.0) / 100.0
            ),
            # Strategy settings
            "STRATEGY_TYPE": "llm",
            "LLM_MODEL": config_data.get("llm", {}).get("model_name", "gpt-4"),
            "LLM_TEMPERATURE": str(config_data.get("llm", {}).get("temperature", 0.1)),
            "LLM_MAX_CONTEXT": str(config_data.get("llm", {}).get("max_tokens", 4000)),
            "LLM_USE_MEMORY": str(
                config_data.get("mcp", {}).get("enabled", False)
            ).lower(),
            "LLM_CONFIDENCE_THRESHOLD": "0.7",
            # Exchange settings
            "EXCHANGE_TYPE": config_data.get("exchange", {}).get(
                "exchange_type", "coinbase"
            ),
            "RATE_LIMIT_RPS": "10",
            "RATE_LIMIT_RPM": str(
                config_data.get("exchange", {}).get("rate_limit_requests", 100)
            ),
            "RATE_LIMIT_RPH": "1000",
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

    except Exception:
        return None


def save_settings_to_file(settings: Settings, file_path: str | Path) -> None:
    """Save settings to a configuration file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_data = settings.to_dict()
    
    with open(file_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)


def validate_settings(settings: Settings) -> str | None:
    """Validate settings using functional patterns and return warnings if any."""
    warnings = []

    # Trading validation
    if settings.trading.leverage < 1 or settings.trading.leverage > 100:
        warnings.append(
            f"Trading leverage {settings.trading.leverage} is outside safe range (1-100)"
        )

    if settings.trading.max_size_pct < 0.1 or settings.trading.max_size_pct > 100:
        warnings.append(
            f"Trading max size {settings.trading.max_size_pct}% is outside safe range (0.1-100%)"
        )

    # Risk validation
    if settings.risk.max_daily_loss_pct < 0.1 or settings.risk.max_daily_loss_pct > 50:
        warnings.append(
            f"Risk max daily loss {settings.risk.max_daily_loss_pct}% is outside safe range (0.1-50%)"
        )

    if (
        settings.risk.max_concurrent_trades < 1
        or settings.risk.max_concurrent_trades > 10
    ):
        warnings.append(
            f"Risk max concurrent trades {settings.risk.max_concurrent_trades} is outside safe range (1-10)"
        )

    # LLM validation
    if settings.llm.temperature < 0 or settings.llm.temperature > 2:
        warnings.append(
            f"LLM temperature {settings.llm.temperature} is outside valid range (0-2)"
        )

    if settings.llm.max_tokens < 100 or settings.llm.max_tokens > 100000:
        warnings.append(
            f"LLM max tokens {settings.llm.max_tokens} is outside reasonable range (100-100000)"
        )

    # Exchange validation
    if settings.exchange.api_timeout < 1 or settings.exchange.api_timeout > 300:
        warnings.append(
            f"Exchange API timeout {settings.exchange.api_timeout}s is outside safe range (1-300s)"
        )

    # System validation
    if (
        settings.system.update_frequency_seconds < 1
        or settings.system.update_frequency_seconds > 3600
    ):
        warnings.append(
            f"System update frequency {settings.system.update_frequency_seconds}s is outside reasonable range (1-3600s)"
        )

    # Paper trading validation
    if (
        settings.paper_trading.starting_balance < 100
        or settings.paper_trading.starting_balance > 1000000
    ):
        warnings.append(
            f"Paper trading starting balance ${settings.paper_trading.starting_balance} is outside reasonable range ($100-$1M)"
        )

    # Logical consistency checks
    if settings.risk.default_take_profit_pct <= settings.risk.default_stop_loss_pct:
        warnings.append(
            "Take profit percentage should be greater than stop loss percentage"
        )

    if settings.system.dry_run and settings.system.environment == "production":
        warnings.append(
            "Dry run mode enabled in production environment - this may not be intended"
        )

    if not settings.system.dry_run and settings.exchange.cb_sandbox:
        warnings.append(
            "Live trading mode with sandbox exchange - this may not be intended"
        )

    # Orderbook validation
    if settings.orderbook.depth_levels < 1 or settings.orderbook.depth_levels > 100:
        warnings.append(
            f"Orderbook depth levels {settings.orderbook.depth_levels} is outside reasonable range (1-100)"
        )

    if (
        settings.orderbook.refresh_interval_ms < 10
        or settings.orderbook.refresh_interval_ms > 10000
    ):
        warnings.append(
            f"Orderbook refresh interval {settings.orderbook.refresh_interval_ms}ms is outside reasonable range (10-10000ms)"
        )

    if (
        settings.orderbook.quality_threshold < 0.1
        or settings.orderbook.quality_threshold > 1.0
    ):
        warnings.append(
            f"Orderbook quality threshold {settings.orderbook.quality_threshold} is outside valid range (0.1-1.0)"
        )

    if (
        settings.orderbook.imbalance_detection_threshold < 0.1
        or settings.orderbook.imbalance_detection_threshold > 1.0
    ):
        warnings.append(
            f"Orderbook imbalance detection threshold {settings.orderbook.imbalance_detection_threshold} is outside valid range (0.1-1.0)"
        )

    if (
        settings.orderbook.max_price_deviation_pct < 0.1
        or settings.orderbook.max_price_deviation_pct > 50.0
    ):
        warnings.append(
            f"Orderbook max price deviation {settings.orderbook.max_price_deviation_pct}% is outside reasonable range (0.1-50%)"
        )

    return "; ".join(warnings) if warnings else None


# Global settings instance with functional programming backend
settings = create_settings()


def get_config():
    """Get configuration - compatibility function."""
    return settings


def get_functional_config() -> Optional["FunctionalConfig"]:
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
        "orderbook": {
            "depth_levels": 20,
            "refresh_interval_ms": 100,
            "max_age_ms": 1000,
            "min_liquidity_threshold": "500",
            "max_spread_bps": 200,
            "quality_threshold": 0.8,
            "staleness_threshold_ms": 2000,
            "aggregation_levels": [1, 5, 10, 20],
            "price_precision": 6,
            "size_precision": 4,
            "enable_snapshot_recovery": True,
            "snapshot_recovery_interval_ms": 5000,
            "enable_incremental_updates": True,
            "buffer_size": 1000,
            "compression_enabled": False,
            "websocket_timeout_ms": 30000,
            "heartbeat_interval_ms": 15000,
            "reconnect_delay_ms": 1000,
            "max_reconnect_attempts": 10,
            "enable_order_flow_analysis": True,
            "imbalance_detection_threshold": 0.3,
            "enable_price_validation": True,
            "max_price_deviation_pct": 5.0,
            "enable_size_validation": True,
            "min_order_size": "10",
            "max_order_size": "50000",
            "enable_time_validation": True,
            "max_timestamp_drift_ms": 5000,
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


# Add missing load_from_file and save_to_file methods to Settings class for compatibility
Settings.load_from_file = classmethod(
    lambda _cls, file_path: load_settings_from_file(file_path)
)
Settings.save_to_file = lambda self, file_path: save_settings_to_file(self, file_path)


# Export all the classes and functions that the current interface provides
__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "DataSettings",
    "DominanceSettings",
    "Environment",
    "ExchangeSettings",
    "LLMSettings",
    "MCPSettings",
    "MonitoringSettings",
    "OmniSearchSettings",
    "OrderbookSettings",
    "PaperTradingSettings",
    "RiskSettings",
    "Settings",
    "StartupValidator",
    "SystemSettings",
    "TradingProfile",
    "TradingSettings",
    "create_settings",
    "get_config",
    "get_config_template",
    "get_functional_config",
    "load_settings_from_file",
    "save_settings_to_file",
    "settings",
    "validate_settings",
]
