"""
Unit tests for functional programming configuration types.

This module tests the immutable configuration types, opaque types for sensitive data,
validation functions, and environment variable parsing with Result/Either error handling.
"""

import os
from datetime import datetime
from unittest.mock import patch

import pytest

from bot.fp.types.config import (
    APIKey,
    BacktestConfig,
    CoinbaseExchangeConfig,
    Config,
    ExchangeType,
    FeatureFlags,
    FeeStructure,
    LLMStrategyConfig,
    LogLevel,
    MeanReversionStrategyConfig,
    MomentumStrategyConfig,
    PrivateKey,
    RateLimits,
    SystemConfig,
    parse_bool_env,
    parse_float_env,
    parse_int_env,
    parse_list_env,
    validate_config,
)
from bot.fp.types.result import Failure, Success


class TestOpaqueTypes:
    """Test opaque types for sensitive data."""

    def test_api_key_creation_success(self):
        """Test successful API key creation."""
        api_key_result = APIKey.create("sk-1234567890abcdefghij")
        assert isinstance(api_key_result, Success)

        api_key = api_key_result.success()
        assert isinstance(api_key, APIKey)

        # Test string representation masks the key
        str_repr = str(api_key)
        assert "APIKey(***" in str_repr
        assert "ghij)" in str_repr
        assert "sk-1234567890abcdef" not in str_repr

    def test_api_key_creation_failure_empty(self):
        """Test API key creation failure with empty string."""
        result = APIKey.create("")
        assert isinstance(result, Failure)
        assert "too short" in result.failure()

    def test_api_key_creation_failure_too_short(self):
        """Test API key creation failure with too short key."""
        result = APIKey.create("short")
        assert isinstance(result, Failure)
        assert "too short" in result.failure()

    def test_private_key_creation_success(self):
        """Test successful private key creation."""
        private_key_result = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIExamplePrivateKey123\n-----END EC PRIVATE KEY-----"
        )
        assert isinstance(private_key_result, Success)

        private_key = private_key_result.success()
        assert isinstance(private_key, PrivateKey)

        # Test string representation masks the key completely
        str_repr = str(private_key)
        assert str_repr == "PrivateKey(***)"

    def test_private_key_creation_failure_empty(self):
        """Test private key creation failure with empty string."""
        result = PrivateKey.create("")
        assert isinstance(result, Failure)
        assert "too short" in result.failure()

    def test_private_key_creation_failure_too_short(self):
        """Test private key creation failure with too short key."""
        result = PrivateKey.create("short")
        assert isinstance(result, Failure)
        assert "too short" in result.failure()


class TestStrategyConfigurations:
    """Test strategy configuration types."""

    def test_momentum_strategy_config_creation_success(self):
        """Test successful momentum strategy config creation."""
        result = MomentumStrategyConfig.create(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=1.0,
            use_volume_confirmation=True,
        )

        assert isinstance(result, Success)
        config = result.success()
        assert config.lookback_period == 20
        assert config.entry_threshold.value == 2.0
        assert config.exit_threshold.value == 1.0
        assert config.use_volume_confirmation is True

    def test_momentum_strategy_config_invalid_lookback(self):
        """Test momentum strategy config with invalid lookback period."""
        result = MomentumStrategyConfig.create(
            lookback_period=0, entry_threshold=2.0, exit_threshold=1.0
        )

        assert isinstance(result, Failure)
        assert "Lookback period must be positive" in result.failure()

    def test_momentum_strategy_config_invalid_threshold(self):
        """Test momentum strategy config with invalid threshold."""
        result = MomentumStrategyConfig.create(
            lookback_period=20,
            entry_threshold=-1.0,  # Invalid negative threshold
            exit_threshold=1.0,
        )

        assert isinstance(result, Failure)
        assert "Invalid entry threshold" in result.failure()

    def test_mean_reversion_strategy_config_creation_success(self):
        """Test successful mean reversion strategy config creation."""
        result = MeanReversionStrategyConfig.create(
            window_size=50,
            std_deviations=2.0,
            min_volatility=0.1,
            max_holding_period=100,
        )

        assert isinstance(result, Success)
        config = result.success()
        assert config.window_size == 50
        assert config.std_deviations == 2.0
        assert config.min_volatility.value == 0.1
        assert config.max_holding_period == 100

    def test_mean_reversion_strategy_config_invalid_window(self):
        """Test mean reversion strategy config with invalid window size."""
        result = MeanReversionStrategyConfig.create(
            window_size=1,  # Too small
            std_deviations=2.0,
            min_volatility=0.1,
            max_holding_period=100,
        )

        assert isinstance(result, Failure)
        assert "Window size must be at least 2" in result.failure()

    def test_llm_strategy_config_creation_success(self):
        """Test successful LLM strategy config creation."""
        result = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=70.0,
        )

        assert isinstance(result, Success)
        config = result.success()
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_context_length == 4000
        assert config.use_memory is True
        assert config.confidence_threshold.value == 70.0

    def test_llm_strategy_config_invalid_temperature(self):
        """Test LLM strategy config with invalid temperature."""
        result = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=3.0,  # Too high
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=70.0,
        )

        assert isinstance(result, Failure)
        assert "Temperature must be between 0 and 2" in result.failure()

    def test_llm_strategy_config_invalid_context_length(self):
        """Test LLM strategy config with invalid context length."""
        result = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=50,  # Too small
            use_memory=True,
            confidence_threshold=70.0,
        )

        assert isinstance(result, Failure)
        assert "Max context length too small" in result.failure()


class TestExchangeConfigurations:
    """Test exchange configuration types."""

    def test_rate_limits_creation_success(self):
        """Test successful rate limits creation."""
        result = RateLimits.create(
            requests_per_second=10, requests_per_minute=100, requests_per_hour=1000
        )

        assert isinstance(result, Success)
        limits = result.success()
        assert limits.requests_per_second == 10
        assert limits.requests_per_minute == 100
        assert limits.requests_per_hour == 1000

    def test_rate_limits_creation_invalid_negative(self):
        """Test rate limits creation with negative values."""
        result = RateLimits.create(
            requests_per_second=-1, requests_per_minute=100, requests_per_hour=1000
        )

        assert isinstance(result, Failure)
        assert "All rate limits must be positive" in result.failure()

    def test_rate_limits_creation_inconsistent(self):
        """Test rate limits creation with inconsistent values."""
        result = RateLimits.create(
            requests_per_second=100,  # 100 * 60 = 6000 > 1000
            requests_per_minute=1000,
            requests_per_hour=1000,
        )

        assert isinstance(result, Failure)
        assert "Inconsistent rate limits" in result.failure()


class TestSystemConfiguration:
    """Test system configuration."""

    def test_system_config_creation_success(self):
        """Test successful system config creation."""
        result = SystemConfig.create(
            trading_pairs=["BTC-USD", "ETH-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={
                "enable_websocket": True,
                "enable_memory": False,
                "enable_backtesting": True,
                "enable_paper_trading": True,
                "enable_risk_management": True,
                "enable_notifications": False,
                "enable_metrics": True,
            },
            max_concurrent_positions=3,
            default_position_size=10.0,
        )

        assert isinstance(result, Success)
        config = result.success()
        assert len(config.trading_pairs) == 2
        assert config.interval.value == "1m"
        assert config.max_concurrent_positions == 3
        assert config.default_position_size.value == 10.0

    def test_system_config_invalid_trading_pair(self):
        """Test system config with invalid trading pair."""
        result = SystemConfig.create(
            trading_pairs=["INVALID"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        )

        assert isinstance(result, Failure)
        assert "Invalid trading pair" in result.failure()

    def test_system_config_invalid_mode(self):
        """Test system config with invalid trading mode."""
        result = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="invalid_mode",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        )

        assert isinstance(result, Failure)
        assert "Invalid trading mode" in result.failure()

    def test_system_config_zero_positions(self):
        """Test system config with zero max positions."""
        result = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=0,  # Invalid
            default_position_size=10.0,
        )

        assert isinstance(result, Failure)
        assert "Max concurrent positions must be at least 1" in result.failure()


class TestBacktestConfiguration:
    """Test backtest configuration."""

    def test_backtest_config_creation_success(self):
        """Test successful backtest config creation."""
        result = BacktestConfig.create(
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0,
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005,
            use_limit_orders=True,
        )

        assert isinstance(result, Success)
        config = result.success()
        assert config.start_date == datetime.fromisoformat("2024-01-01")
        assert config.end_date == datetime.fromisoformat("2024-12-31")
        assert config.initial_capital.amount == 10000.0
        assert config.use_limit_orders is True

    def test_backtest_config_invalid_date_order(self):
        """Test backtest config with invalid date order."""
        result = BacktestConfig.create(
            start_date="2024-12-31",
            end_date="2024-01-01",  # End before start
            initial_capital=10000.0,
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005,
        )

        assert isinstance(result, Failure)
        assert "Start date must be before end date" in result.failure()

    def test_backtest_config_invalid_date_format(self):
        """Test backtest config with invalid date format."""
        result = BacktestConfig.create(
            start_date="invalid-date",
            end_date="2024-12-31",
            initial_capital=10000.0,
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005,
        )

        assert isinstance(result, Failure)
        assert "Invalid date format" in result.failure()

    def test_fee_structure_creation_success(self):
        """Test successful fee structure creation."""
        result = FeeStructure.create(0.001, 0.002)

        assert isinstance(result, Success)
        fees = result.success()
        assert fees.maker_fee.value == 0.001
        assert fees.taker_fee.value == 0.002

    def test_fee_structure_invalid_fee(self):
        """Test fee structure with invalid fee."""
        result = FeeStructure.create(-0.001, 0.002)  # Negative fee

        assert isinstance(result, Failure)
        assert "Invalid maker fee" in result.failure()


class TestEnvironmentVariableParsing:
    """Test environment variable parsing functions."""

    def test_parse_bool_env_true_values(self):
        """Test parsing boolean environment variables for true values."""
        true_values = ["true", "1", "yes", "on", "True", "YES", "ON"]

        for value in true_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                result = parse_bool_env("TEST_BOOL", False)
                assert result is True, f"Value '{value}' should parse as True"

    def test_parse_bool_env_false_values(self):
        """Test parsing boolean environment variables for false values."""
        false_values = ["false", "0", "no", "off", "False", "NO", "OFF", ""]

        for value in false_values:
            with patch.dict(os.environ, {"TEST_BOOL": value}):
                result = parse_bool_env("TEST_BOOL", True)
                assert result is False, f"Value '{value}' should parse as False"

    def test_parse_bool_env_missing_uses_default(self):
        """Test parsing boolean environment variable with missing value."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_bool_env("MISSING_BOOL", True)
            assert result is True

    def test_parse_int_env_success(self):
        """Test successful integer parsing."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = parse_int_env("TEST_INT", 0)
            assert isinstance(result, Success)
            assert result.success() == 42

    def test_parse_int_env_failure(self):
        """Test integer parsing failure."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            result = parse_int_env("TEST_INT", 0)
            assert isinstance(result, Failure)
            assert "Invalid integer" in result.failure()

    def test_parse_int_env_missing_uses_default(self):
        """Test integer parsing with missing value."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_int_env("MISSING_INT", 42)
            assert isinstance(result, Success)
            assert result.success() == 42

    def test_parse_float_env_success(self):
        """Test successful float parsing."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = parse_float_env("TEST_FLOAT", 0.0)
            assert isinstance(result, Success)
            assert result.success() == 3.14

    def test_parse_float_env_failure(self):
        """Test float parsing failure."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            result = parse_float_env("TEST_FLOAT", 0.0)
            assert isinstance(result, Failure)
            assert "Invalid float" in result.failure()

    def test_parse_list_env_success(self):
        """Test successful list parsing."""
        with patch.dict(os.environ, {"TEST_LIST": "a,b,c"}):
            result = parse_list_env("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_parse_list_env_with_spaces(self):
        """Test list parsing with spaces."""
        with patch.dict(os.environ, {"TEST_LIST": "a, b , c "}):
            result = parse_list_env("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_parse_list_env_empty(self):
        """Test list parsing with empty value."""
        with patch.dict(os.environ, {"TEST_LIST": ""}):
            result = parse_list_env("TEST_LIST")
            assert result == []

    def test_parse_list_env_missing(self):
        """Test list parsing with missing value."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_list_env("MISSING_LIST")
            assert result == []


class TestConfigurationValidation:
    """Test configuration validation functions."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        # Create valid configurations
        strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0,
        ).success()

        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        rate_limits = RateLimits.create(10, 100, 1000).success()

        exchange = CoinbaseExchangeConfig(
            api_key=api_key,
            private_key=private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=rate_limits,
        )

        system = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": False},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(strategy=strategy, exchange=exchange, system=system)

        result = validate_config(config)
        assert isinstance(result, Success)

    def test_validate_config_memory_mismatch(self):
        """Test configuration validation with memory feature mismatch."""
        # LLM strategy wants memory but system doesn't enable it
        strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,  # Wants memory
            confidence_threshold=70.0,
        ).success()

        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        rate_limits = RateLimits.create(10, 100, 1000).success()

        exchange = CoinbaseExchangeConfig(
            api_key=api_key,
            private_key=private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=rate_limits,
        )

        system = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": False},  # Memory disabled
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(strategy=strategy, exchange=exchange, system=system)

        result = validate_config(config)
        assert isinstance(result, Failure)
        assert "memory but it's disabled" in result.failure()


class TestFeatureFlags:
    """Test feature flags configuration."""

    def test_feature_flags_defaults(self):
        """Test feature flags default values."""
        flags = FeatureFlags()

        assert flags.enable_websocket is True
        assert flags.enable_memory is False
        assert flags.enable_backtesting is True
        assert flags.enable_paper_trading is True
        assert flags.enable_risk_management is True
        assert flags.enable_notifications is False
        assert flags.enable_metrics is True

    def test_feature_flags_custom_values(self):
        """Test feature flags with custom values."""
        flags = FeatureFlags(
            enable_websocket=False, enable_memory=True, enable_notifications=True
        )

        assert flags.enable_websocket is False
        assert flags.enable_memory is True
        assert flags.enable_notifications is True
        # Defaults should be preserved
        assert flags.enable_paper_trading is True
        assert flags.enable_risk_management is True


class TestLogLevel:
    """Test log level enumeration."""

    def test_log_level_values(self):
        """Test log level enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_log_level_iteration(self):
        """Test log level enum iteration."""
        levels = [level.value for level in LogLevel]
        expected = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert levels == expected


class TestExchangeType:
    """Test exchange type enumeration."""

    def test_exchange_type_values(self):
        """Test exchange type enum values."""
        assert ExchangeType.COINBASE.value == "coinbase"
        assert ExchangeType.BLUEFIN.value == "bluefin"
        assert ExchangeType.BINANCE.value == "binance"

    def test_exchange_type_iteration(self):
        """Test exchange type enum iteration."""
        types = [exchange_type.value for exchange_type in ExchangeType]
        expected = ["coinbase", "bluefin", "binance"]
        assert types == expected


class TestImmutability:
    """Test that configuration objects are immutable."""

    def test_api_key_immutable(self):
        """Test that APIKey is immutable."""
        api_key = APIKey.create("sk-1234567890abcdefghij").success()

        with pytest.raises(AttributeError):
            api_key._value = "new_value"  # type: ignore

    def test_strategy_config_immutable(self):
        """Test that strategy configs are immutable."""
        config = MomentumStrategyConfig.create(
            lookback_period=20, entry_threshold=2.0, exit_threshold=1.0
        ).success()

        with pytest.raises(AttributeError):
            config.lookback_period = 30  # type: ignore

    def test_system_config_immutable(self):
        """Test that system config is immutable."""
        config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        with pytest.raises(AttributeError):
            config.max_concurrent_positions = 5  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
