"""
Configuration types for the trading bot.

This module defines configuration structures using functional programming patterns,
including sum types for strategy selection, opaque types for sensitive data,
and comprehensive validation with environment variable parsing.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Union

from .base import Money, Percentage, Symbol, TimeInterval, TradingMode
from .result import Failure, Result, Success


# Opaque types for sensitive data
@dataclass(frozen=True)
class APIKey:
    """Opaque type for API keys."""

    _value: str

    @classmethod
    def create(cls, value: str) -> Result["APIKey", str]:
        """Create an APIKey with validation."""
        if not value or len(value) < 10:
            return Failure("Invalid API key: too short")
        return Success(cls(_value=value))

    def __str__(self) -> str:
        """Return masked representation."""
        return f"APIKey(***{self._value[-4:]})"


@dataclass(frozen=True)
class PrivateKey:
    """Opaque type for private keys."""

    _value: str

    @classmethod
    def create(cls, value: str) -> Result["PrivateKey", str]:
        """Create a PrivateKey with validation."""
        if not value or len(value) < 20:
            return Failure("Invalid private key: too short")
        return Success(cls(_value=value))

    def __str__(self) -> str:
        """Return masked representation."""
        return "PrivateKey(***)"


# Strategy configurations
@dataclass(frozen=True)
class MomentumStrategyConfig:
    """Configuration for momentum strategy."""

    lookback_period: int
    entry_threshold: Percentage
    exit_threshold: Percentage
    use_volume_confirmation: bool = True

    @classmethod
    def create(
        cls,
        lookback_period: int,
        entry_threshold: float,
        exit_threshold: float,
        use_volume_confirmation: bool = True,
    ) -> Result["MomentumStrategyConfig", str]:
        """Create config with validation."""
        if lookback_period < 1:
            return Failure("Lookback period must be positive")

        entry_pct = Percentage.create(entry_threshold)
        if isinstance(entry_pct, Failure):
            return Failure(f"Invalid entry threshold: {entry_pct.failure()}")

        exit_pct = Percentage.create(exit_threshold)
        if isinstance(exit_pct, Failure):
            return Failure(f"Invalid exit threshold: {exit_pct.failure()}")

        return Success(
            cls(
                lookback_period=lookback_period,
                entry_threshold=entry_pct.success(),
                exit_threshold=exit_pct.success(),
                use_volume_confirmation=use_volume_confirmation,
            )
        )


@dataclass(frozen=True)
class MeanReversionStrategyConfig:
    """Configuration for mean reversion strategy."""

    window_size: int
    std_deviations: float
    min_volatility: Percentage
    max_holding_period: int

    @classmethod
    def create(
        cls,
        window_size: int,
        std_deviations: float,
        min_volatility: float,
        max_holding_period: int,
    ) -> Result["MeanReversionStrategyConfig", str]:
        """Create config with validation."""
        if window_size < 2:
            return Failure("Window size must be at least 2")
        if std_deviations <= 0:
            return Failure("Standard deviations must be positive")
        if max_holding_period < 1:
            return Failure("Max holding period must be positive")

        min_vol = Percentage.create(min_volatility)
        if isinstance(min_vol, Failure):
            return Failure(f"Invalid min volatility: {min_vol.failure()}")

        return Success(
            cls(
                window_size=window_size,
                std_deviations=std_deviations,
                min_volatility=min_vol.success(),
                max_holding_period=max_holding_period,
            )
        )


@dataclass(frozen=True)
class LLMStrategyConfig:
    """Configuration for LLM-based strategy."""

    model_name: str
    temperature: float
    max_context_length: int
    use_memory: bool
    confidence_threshold: Percentage

    @classmethod
    def create(
        cls,
        model_name: str,
        temperature: float,
        max_context_length: int,
        use_memory: bool,
        confidence_threshold: float,
    ) -> Result["LLMStrategyConfig", str]:
        """Create config with validation."""
        if not model_name:
            return Failure("Model name cannot be empty")
        if not 0 <= temperature <= 2:
            return Failure("Temperature must be between 0 and 2")
        if max_context_length < 100:
            return Failure("Max context length too small")

        conf_threshold = Percentage.create(confidence_threshold)
        if isinstance(conf_threshold, Failure):
            return Failure(f"Invalid confidence threshold: {conf_threshold.failure()}")

        return Success(
            cls(
                model_name=model_name,
                temperature=temperature,
                max_context_length=max_context_length,
                use_memory=use_memory,
                confidence_threshold=conf_threshold.success(),
            )
        )


# Sum type for strategy configuration
StrategyConfig = Union[
    MomentumStrategyConfig, MeanReversionStrategyConfig, LLMStrategyConfig
]


# Exchange configurations
class ExchangeType(Enum):
    """Supported exchange types."""

    COINBASE = "coinbase"
    BLUEFIN = "bluefin"
    BINANCE = "binance"


class AccountType(Enum):
    """Account types for trading."""

    CFM = "CFM"  # Coinbase futures margin account
    CBI = "CBI"  # Coinbase instant account
    SPOT = "SPOT"  # Spot trading account
    FUTURES = "FUTURES"  # Futures trading account


class Environment(Enum):
    """Environment types for configuration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass(frozen=True)
class RateLimits:
    """Rate limit configuration."""

    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int

    @classmethod
    def create(
        cls, requests_per_second: int, requests_per_minute: int, requests_per_hour: int
    ) -> Result["RateLimits", str]:
        """Create rate limits with validation."""
        if any(
            x <= 0
            for x in [requests_per_second, requests_per_minute, requests_per_hour]
        ):
            return Failure("All rate limits must be positive")

        # Consistency check
        if requests_per_second * 60 > requests_per_minute:
            return Failure("Inconsistent rate limits: per-second * 60 > per-minute")
        if requests_per_minute * 60 > requests_per_hour:
            return Failure("Inconsistent rate limits: per-minute * 60 > per-hour")

        return Success(
            cls(
                requests_per_second=requests_per_second,
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour,
            )
        )


@dataclass(frozen=True)
class CoinbaseExchangeConfig:
    """Configuration for Coinbase exchange."""

    api_key: APIKey
    private_key: PrivateKey
    api_url: str
    websocket_url: str
    rate_limits: RateLimits


@dataclass(frozen=True)
class BluefinExchangeConfig:
    """Configuration for Bluefin exchange."""

    private_key: PrivateKey
    network: Literal["mainnet", "testnet"]
    rpc_url: str
    rate_limits: RateLimits


@dataclass(frozen=True)
class BinanceExchangeConfig:
    """Configuration for Binance exchange."""

    api_key: APIKey
    api_secret: APIKey
    testnet: bool
    rate_limits: RateLimits


# Sum type for exchange configuration
ExchangeConfig = Union[
    CoinbaseExchangeConfig, BluefinExchangeConfig, BinanceExchangeConfig
]


# System configuration
class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class FeatureFlags:
    """Feature flags for the system."""

    enable_websocket: bool = True
    enable_memory: bool = False
    enable_backtesting: bool = True
    enable_paper_trading: bool = True
    enable_risk_management: bool = True
    enable_notifications: bool = False
    enable_metrics: bool = True


@dataclass(frozen=True)
class SystemConfig:
    """System-wide configuration."""

    trading_pairs: list[Symbol]
    interval: TimeInterval
    mode: TradingMode
    log_level: LogLevel
    features: FeatureFlags
    max_concurrent_positions: int
    default_position_size: Percentage

    @classmethod
    def create(
        cls,
        trading_pairs: list[str],
        interval: str,
        mode: str,
        log_level: str,
        features: dict[str, bool],
        max_concurrent_positions: int,
        default_position_size: float,
    ) -> Result["SystemConfig", str]:
        """Create system config with validation."""
        # Parse trading pairs
        symbols: list[Symbol] = []
        for pair in trading_pairs:
            symbol = Symbol.create(pair)
            if isinstance(symbol, Failure):
                return Failure(f"Invalid trading pair {pair}: {symbol.failure()}")
            symbols.append(symbol.success())

        if not symbols:
            return Failure("At least one trading pair required")

        # Parse interval
        time_interval = TimeInterval.create(interval)
        if isinstance(time_interval, Failure):
            return Failure(f"Invalid interval: {time_interval.failure()}")

        # Parse mode
        try:
            trading_mode = TradingMode[mode.upper()]
        except KeyError:
            return Failure(f"Invalid trading mode: {mode}")

        # Parse log level
        try:
            level = LogLevel[log_level.upper()]
        except KeyError:
            return Failure(f"Invalid log level: {log_level}")

        # Parse position size
        position_size = Percentage.create(default_position_size)
        if isinstance(position_size, Failure):
            return Failure(f"Invalid position size: {position_size.failure()}")

        # Validate other parameters
        if max_concurrent_positions < 1:
            return Failure("Max concurrent positions must be at least 1")

        return Success(
            cls(
                trading_pairs=symbols,
                interval=time_interval.success(),
                mode=trading_mode,
                log_level=level,
                features=FeatureFlags(**features),
                max_concurrent_positions=max_concurrent_positions,
                default_position_size=position_size.success(),
            )
        )


# Backtest configuration
@dataclass(frozen=True)
class FeeStructure:
    """Trading fee structure."""

    maker_fee: Percentage
    taker_fee: Percentage

    @classmethod
    def create(cls, maker_fee: float, taker_fee: float) -> Result["FeeStructure", str]:
        """Create fee structure with validation."""
        maker = Percentage.create(maker_fee)
        if isinstance(maker, Failure):
            return Failure(f"Invalid maker fee: {maker.failure()}")

        taker = Percentage.create(taker_fee)
        if isinstance(taker, Failure):
            return Failure(f"Invalid taker fee: {taker.failure()}")

        return Success(cls(maker_fee=maker.success(), taker_fee=taker.success()))


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration."""

    start_date: datetime
    end_date: datetime
    initial_capital: Money
    fee_structure: FeeStructure
    slippage: Percentage
    use_limit_orders: bool

    @classmethod
    def create(
        cls,
        start_date: str,
        end_date: str,
        initial_capital: float,
        currency: str,
        maker_fee: float,
        taker_fee: float,
        slippage: float,
        use_limit_orders: bool = True,
    ) -> Result["BacktestConfig", str]:
        """Create backtest config with validation."""
        # Parse dates
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except ValueError as e:
            return Failure(f"Invalid date format: {e}")

        if start >= end:
            return Failure("Start date must be before end date")

        # Parse money
        capital = Money.create(initial_capital, currency)
        if isinstance(capital, Failure):
            return Failure(f"Invalid initial capital: {capital.failure()}")

        # Parse fees
        fees = FeeStructure.create(maker_fee, taker_fee)
        if isinstance(fees, Failure):
            return Failure(f"Invalid fee structure: {fees.failure()}")

        # Parse slippage
        slip = Percentage.create(slippage)
        if isinstance(slip, Failure):
            return Failure(f"Invalid slippage: {slip.failure()}")

        return Success(
            cls(
                start_date=start,
                end_date=end,
                initial_capital=capital.success(),
                fee_structure=fees.success(),
                slippage=slip.success(),
                use_limit_orders=use_limit_orders,
            )
        )


# Environment variable parsing
def parse_env_var(key: str, default: str | None = None) -> str | None:
    """Parse environment variable with optional default."""
    return os.environ.get(key, default)


def parse_bool_env(key: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    value = parse_env_var(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def parse_int_env(key: str, default: int) -> Result[int, str]:
    """Parse integer environment variable."""
    value = parse_env_var(key)
    if value is None or value == "":
        return Success(default)
    try:
        return Success(int(value))
    except ValueError:
        return Failure(f"Invalid integer for {key}: {value}")


def parse_float_env(key: str, default: float) -> Result[float, str]:
    """Parse float environment variable."""
    value = parse_env_var(key)
    if value is None or value == "":
        return Success(default)
    try:
        return Success(float(value))
    except ValueError:
        return Failure(f"Invalid float for {key}: {value}")


def parse_list_env(key: str, delimiter: str = ",") -> list[str]:
    """Parse list from environment variable."""
    value = parse_env_var(key)
    if not value:
        return []
    return [item.strip() for item in value.split(delimiter) if item.strip()]


# Sui Private Key Conversion Helper
def _convert_sui_private_key(private_key_str: str) -> tuple[str | None, str, str]:
    """Convert Sui private key from any format to hex format.

    Args:
        private_key_str: Private key in any supported format

    Returns:
        Tuple of (converted_hex_key, format_detected, message)
    """
    try:
        # Import the converter utility
        from bot.utils.sui_key_converter import auto_convert_private_key

        return auto_convert_private_key(private_key_str)
    except ImportError:
        # Fallback implementation if converter module is not available
        private_key_str = private_key_str.strip()

        # Check if already in hex format
        if private_key_str.startswith("0x") and len(private_key_str) == 66:
            return private_key_str, "hex", "✅ Hex format private key validated"
        if len(private_key_str) == 64 and all(
            c in "0123456789abcdefABCDEF" for c in private_key_str
        ):
            return (
                f"0x{private_key_str}",
                "hex",
                "✅ Hex format private key validated (added 0x prefix)",
            )

        # Check if bech32 format
        if private_key_str.startswith("suiprivkey"):
            return (
                None,
                "bech32",
                (
                    "🔧 Bech32 format detected (suiprivkey...). Please convert to hex format:\n"
                    "1. Open your Sui wallet → Settings → Export Private Key\n"
                    "2. Choose 'Raw Private Key' or 'Hex Format'\n"
                    "3. Copy the hex string (should start with 0x)\n"
                    "4. Update your BLUEFIN_PRIVATE_KEY in .env with the hex format"
                ),
            )

        # Check if mnemonic format
        words = private_key_str.split()
        if len(words) in [12, 24] and all(word.isalpha() for word in words):
            return (
                None,
                "mnemonic",
                (
                    "🔧 Mnemonic phrase detected. Please convert to private key:\n"
                    '1. Use Sui CLI: sui keytool import "<your mnemonic>" ed25519\n'
                    "2. Then export as hex: sui keytool export <address> --key-scheme ed25519\n"
                    "3. Update your BLUEFIN_PRIVATE_KEY in .env with the hex format"
                ),
            )

        # Unknown format
        return (
            None,
            "unknown",
            (
                "❌ Unknown private key format. Supported formats:\n"
                "• Hex: 0x1234...abcd (64 hex characters with 0x prefix)\n"
                "• Bech32: suiprivkey... (Sui wallet export format)\n"
                "• Mnemonic: 12 or 24 word seed phrase"
            ),
        )


# Configuration builders
def build_strategy_config_from_env() -> Result[StrategyConfig, str]:
    """Build strategy configuration from environment variables."""
    strategy_type = parse_env_var("STRATEGY_TYPE", "llm")

    if strategy_type == "momentum":
        lookback = parse_int_env("MOMENTUM_LOOKBACK", 20)
        if isinstance(lookback, Failure):
            return lookback

        entry = parse_float_env("MOMENTUM_ENTRY_THRESHOLD", 0.02)
        if isinstance(entry, Failure):
            return entry

        exit = parse_float_env("MOMENTUM_EXIT_THRESHOLD", 0.01)
        if isinstance(exit, Failure):
            return exit

        use_volume = parse_bool_env("MOMENTUM_USE_VOLUME", True)

        return MomentumStrategyConfig.create(
            lookback_period=lookback.success(),
            entry_threshold=entry.success(),
            exit_threshold=exit.success(),
            use_volume_confirmation=use_volume,
        )

    if strategy_type == "mean_reversion":
        window = parse_int_env("MEAN_REVERSION_WINDOW", 50)
        if isinstance(window, Failure):
            return window

        std_dev = parse_float_env("MEAN_REVERSION_STD_DEV", 2.0)
        if isinstance(std_dev, Failure):
            return std_dev

        min_vol = parse_float_env("MEAN_REVERSION_MIN_VOL", 0.001)
        if isinstance(min_vol, Failure):
            return min_vol

        max_hold = parse_int_env("MEAN_REVERSION_MAX_HOLD", 100)
        if isinstance(max_hold, Failure):
            return max_hold

        return MeanReversionStrategyConfig.create(
            window_size=window.success(),
            std_deviations=std_dev.success(),
            min_volatility=min_vol.success(),
            max_holding_period=max_hold.success(),
        )

    if strategy_type == "llm":
        model = parse_env_var("LLM_MODEL", "gpt-4")

        temp = parse_float_env("LLM_TEMPERATURE", 0.7)
        if isinstance(temp, Failure):
            return temp

        context = parse_int_env("LLM_MAX_CONTEXT", 4000)
        if isinstance(context, Failure):
            return context

        use_memory = parse_bool_env("LLM_USE_MEMORY", False)

        confidence = parse_float_env("LLM_CONFIDENCE_THRESHOLD", 0.7)
        if isinstance(confidence, Failure):
            return confidence

        return LLMStrategyConfig.create(
            model_name=model,
            temperature=temp.success(),
            max_context_length=context.success(),
            use_memory=use_memory,
            confidence_threshold=confidence.success(),
        )

    return Failure(f"Unknown strategy type: {strategy_type}")


def build_exchange_config_from_env() -> Result[ExchangeConfig, str]:
    """Build exchange configuration from environment variables."""
    exchange_type = parse_env_var("EXCHANGE_TYPE", "coinbase")

    # Parse rate limits (common to all exchanges)
    rps = parse_int_env("RATE_LIMIT_RPS", 10)
    if isinstance(rps, Failure):
        return rps

    rpm = parse_int_env("RATE_LIMIT_RPM", 600)
    if isinstance(rpm, Failure):
        return rpm

    rph = parse_int_env("RATE_LIMIT_RPH", 36000)
    if isinstance(rph, Failure):
        return rph

    rate_limits = RateLimits.create(
        requests_per_second=rps.success(),
        requests_per_minute=rpm.success(),
        requests_per_hour=rph.success(),
    )
    if isinstance(rate_limits, Failure):
        return rate_limits

    if exchange_type == "coinbase":
        api_key_str = parse_env_var("COINBASE_API_KEY")
        if not api_key_str:
            return Failure("COINBASE_API_KEY not set")

        api_key = APIKey.create(api_key_str)
        if isinstance(api_key, Failure):
            return api_key

        private_key_str = parse_env_var("COINBASE_PRIVATE_KEY")
        if not private_key_str:
            return Failure("COINBASE_PRIVATE_KEY not set")

        private_key = PrivateKey.create(private_key_str)
        if isinstance(private_key, Failure):
            return private_key

        return Success(
            CoinbaseExchangeConfig(
                api_key=api_key.success(),
                private_key=private_key.success(),
                api_url=parse_env_var("COINBASE_API_URL", "https://api.coinbase.com"),
                websocket_url=parse_env_var("COINBASE_WS_URL", "wss://ws.coinbase.com"),
                rate_limits=rate_limits.success(),
            )
        )

    if exchange_type == "bluefin":
        private_key_str = parse_env_var("BLUEFIN_PRIVATE_KEY")
        if not private_key_str:
            return Failure("BLUEFIN_PRIVATE_KEY not set")

        # Auto-convert Sui private key format if needed
        converted_key, format_detected, message = _convert_sui_private_key(
            private_key_str
        )

        if converted_key is None:
            return Failure(f"Invalid Sui private key format: {message}")

        # Use the converted hex format key
        private_key = PrivateKey.create(converted_key)
        if isinstance(private_key, Failure):
            return private_key

        network = parse_env_var("BLUEFIN_NETWORK", "mainnet")
        if network not in ["mainnet", "testnet"]:
            return Failure(f"Invalid Bluefin network: {network}")

        return Success(
            BluefinExchangeConfig(
                private_key=private_key.success(),
                network=network,  # type: ignore
                rpc_url=parse_env_var(
                    "BLUEFIN_RPC_URL", "https://sui-mainnet.bluefin.io"
                ),
                rate_limits=rate_limits.success(),
            )
        )

    if exchange_type == "binance":
        api_key_str = parse_env_var("BINANCE_API_KEY")
        if not api_key_str:
            return Failure("BINANCE_API_KEY not set")

        api_key = APIKey.create(api_key_str)
        if isinstance(api_key, Failure):
            return api_key

        api_secret_str = parse_env_var("BINANCE_API_SECRET")
        if not api_secret_str:
            return Failure("BINANCE_API_SECRET not set")

        api_secret = APIKey.create(api_secret_str)
        if isinstance(api_secret, Failure):
            return api_secret

        return Success(
            BinanceExchangeConfig(
                api_key=api_key.success(),
                api_secret=api_secret.success(),
                testnet=parse_bool_env("BINANCE_TESTNET", False),
                rate_limits=rate_limits.success(),
            )
        )

    return Failure(f"Unknown exchange type: {exchange_type}")


def build_system_config_from_env() -> Result[SystemConfig, str]:
    """Build system configuration from environment variables."""
    trading_pairs = parse_list_env("TRADING_PAIRS")
    if not trading_pairs:
        trading_pairs = ["BTC-USD"]

    features = {
        "enable_websocket": parse_bool_env("ENABLE_WEBSOCKET", True),
        "enable_memory": parse_bool_env("ENABLE_MEMORY", False),
        "enable_backtesting": parse_bool_env("ENABLE_BACKTESTING", True),
        "enable_paper_trading": parse_bool_env("ENABLE_PAPER_TRADING", True),
        "enable_risk_management": parse_bool_env("ENABLE_RISK_MANAGEMENT", True),
        "enable_notifications": parse_bool_env("ENABLE_NOTIFICATIONS", False),
        "enable_metrics": parse_bool_env("ENABLE_METRICS", True),
    }

    max_positions = parse_int_env("MAX_CONCURRENT_POSITIONS", 3)
    if isinstance(max_positions, Failure):
        return max_positions

    position_size = parse_float_env("DEFAULT_POSITION_SIZE", 0.1)
    if isinstance(position_size, Failure):
        return position_size

    return SystemConfig.create(
        trading_pairs=trading_pairs,
        interval=parse_env_var("TRADING_INTERVAL", "1m"),
        mode=parse_env_var("TRADING_MODE", "paper"),
        log_level=parse_env_var("LOG_LEVEL", "INFO"),
        features=features,
        max_concurrent_positions=max_positions.success(),
        default_position_size=position_size.success(),
    )


def build_backtest_config_from_env() -> Result[BacktestConfig, str]:
    """Build backtest configuration from environment variables."""
    start_date = parse_env_var("BACKTEST_START_DATE", "2024-01-01")
    end_date = parse_env_var("BACKTEST_END_DATE", "2024-12-31")

    initial_capital = parse_float_env("BACKTEST_INITIAL_CAPITAL", 10000.0)
    if isinstance(initial_capital, Failure):
        return initial_capital

    currency = parse_env_var("BACKTEST_CURRENCY", "USD")

    maker_fee = parse_float_env("BACKTEST_MAKER_FEE", 0.001)
    if isinstance(maker_fee, Failure):
        return maker_fee

    taker_fee = parse_float_env("BACKTEST_TAKER_FEE", 0.002)
    if isinstance(taker_fee, Failure):
        return taker_fee

    slippage = parse_float_env("BACKTEST_SLIPPAGE", 0.0005)
    if isinstance(slippage, Failure):
        return slippage

    use_limit = parse_bool_env("BACKTEST_USE_LIMIT_ORDERS", True)

    return BacktestConfig.create(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital.success(),
        currency=currency,
        maker_fee=maker_fee.success(),
        taker_fee=taker_fee.success(),
        slippage=slippage.success(),
        use_limit_orders=use_limit,
    )


# Main configuration structure
@dataclass(frozen=True)
class Config:
    """Complete bot configuration."""

    strategy: StrategyConfig
    exchange: ExchangeConfig
    system: SystemConfig
    backtest: BacktestConfig | None = None

    @classmethod
    def from_env(cls) -> Result["Config", str]:
        """Load configuration from environment variables."""
        strategy = build_strategy_config_from_env()
        if isinstance(strategy, Failure):
            return strategy

        exchange = build_exchange_config_from_env()
        if isinstance(exchange, Failure):
            return exchange

        system = build_system_config_from_env()
        if isinstance(system, Failure):
            return system

        # Backtest config is optional
        backtest = None
        if system.success().features.enable_backtesting:
            backtest_result = build_backtest_config_from_env()
            if isinstance(backtest_result, Failure):
                return backtest_result
            backtest = backtest_result.success()

        return Success(
            cls(
                strategy=strategy.success(),
                exchange=exchange.success(),
                system=system.success(),
                backtest=backtest,
            )
        )


# Config validation helpers
def validate_config(config: Config) -> Result[Config, str]:
    """Validate the complete configuration."""
    # Check strategy-specific validations
    if isinstance(config.strategy, LLMStrategyConfig):
        if config.strategy.use_memory and not config.system.features.enable_memory:
            return Failure(
                "LLM strategy requires memory but it's disabled in system features"
            )

    # Check exchange-specific validations
    if isinstance(config.exchange, BluefinExchangeConfig) and (
        config.exchange.network == "testnet" and config.system.mode == TradingMode.LIVE
    ):
        return Failure("Cannot use testnet for live trading")

    # Check backtest validations
    if config.backtest and config.system.mode != TradingMode.BACKTEST:
        return Failure("Backtest config provided but not in backtest mode")

    return Success(config)


# Export main types
__all__ = [
    "APIKey",
    "AccountType",
    "BacktestConfig",
    "BinanceExchangeConfig",
    "BluefinExchangeConfig",
    "CoinbaseExchangeConfig",
    "Config",
    "Environment",
    "ExchangeConfig",
    "ExchangeType",
    "FeatureFlags",
    "FeeStructure",
    "LLMStrategyConfig",
    "LogLevel",
    "MeanReversionStrategyConfig",
    "MomentumStrategyConfig",
    "PrivateKey",
    "RateLimits",
    "StrategyConfig",
    "SystemConfig",
    "validate_config",
]
