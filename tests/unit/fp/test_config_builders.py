"""
Unit tests for functional programming configuration builders and environment parsing.

This module tests the configuration builder functions that construct FP config objects
from environment variables, ensuring proper Result/Either error handling and validation.
"""

import os
from unittest.mock import patch

import pytest

from bot.fp.types.config import (
    BinanceExchangeConfig,
    BluefinExchangeConfig,
    CoinbaseExchangeConfig,
    LLMStrategyConfig,
    MeanReversionStrategyConfig,
    MomentumStrategyConfig,
    build_backtest_config_from_env,
    build_exchange_config_from_env,
    build_strategy_config_from_env,
    build_system_config_from_env,
)
from bot.fp.types.result import Failure, Success


class TestStrategyConfigBuilders:
    """Test strategy configuration builders from environment variables."""

    def test_build_momentum_strategy_from_env_success(self):
        """Test building momentum strategy config from environment."""
        env_vars = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "20",
            "MOMENTUM_ENTRY_THRESHOLD": "0.02",
            "MOMENTUM_EXIT_THRESHOLD": "0.01",
            "MOMENTUM_USE_VOLUME": "true"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, MomentumStrategyConfig)
            assert config.lookback_period == 20
            assert config.entry_threshold.value == 0.02
            assert config.exit_threshold.value == 0.01
            assert config.use_volume_confirmation is True

    def test_build_momentum_strategy_invalid_lookback(self):
        """Test building momentum strategy with invalid lookback period."""
        env_vars = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "invalid",
            "MOMENTUM_ENTRY_THRESHOLD": "0.02",
            "MOMENTUM_EXIT_THRESHOLD": "0.01"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid integer" in result.failure()

    def test_build_momentum_strategy_with_defaults(self):
        """Test building momentum strategy with default values."""
        env_vars = {"STRATEGY_TYPE": "momentum"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, MomentumStrategyConfig)
            assert config.lookback_period == 20  # Default
            assert config.use_volume_confirmation is True  # Default

    def test_build_mean_reversion_strategy_from_env_success(self):
        """Test building mean reversion strategy config from environment."""
        env_vars = {
            "STRATEGY_TYPE": "mean_reversion",
            "MEAN_REVERSION_WINDOW": "50",
            "MEAN_REVERSION_STD_DEV": "2.0",
            "MEAN_REVERSION_MIN_VOL": "0.001",
            "MEAN_REVERSION_MAX_HOLD": "100"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, MeanReversionStrategyConfig)
            assert config.window_size == 50
            assert config.std_deviations == 2.0
            assert config.min_volatility.value == 0.001
            assert config.max_holding_period == 100

    def test_build_llm_strategy_from_env_success(self):
        """Test building LLM strategy config from environment."""
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "LLM_MODEL": "gpt-4",
            "LLM_TEMPERATURE": "0.7",
            "LLM_MAX_CONTEXT": "4000",
            "LLM_USE_MEMORY": "false",
            "LLM_CONFIDENCE_THRESHOLD": "0.7"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, LLMStrategyConfig)
            assert config.model_name == "gpt-4"
            assert config.temperature == 0.7
            assert config.max_context_length == 4000
            assert config.use_memory is False
            assert config.confidence_threshold.value == 0.7

    def test_build_llm_strategy_with_defaults(self):
        """Test building LLM strategy with default values."""
        env_vars = {"STRATEGY_TYPE": "llm"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, LLMStrategyConfig)
            assert config.model_name == "gpt-4"  # Default
            assert config.temperature == 0.7  # Default
            assert config.use_memory is False  # Default

    def test_build_strategy_unknown_type(self):
        """Test building strategy with unknown type."""
        env_vars = {"STRATEGY_TYPE": "unknown_strategy"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Unknown strategy type" in result.failure()

    def test_build_strategy_default_type(self):
        """Test building strategy with default type (LLM)."""
        with patch.dict(os.environ, {}, clear=True):
            result = build_strategy_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, LLMStrategyConfig)


class TestExchangeConfigBuilders:
    """Test exchange configuration builders from environment variables."""

    def test_build_coinbase_exchange_from_env_success(self):
        """Test building Coinbase exchange config from environment."""
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "COINBASE_API_URL": "https://api.coinbase.com",
            "COINBASE_WS_URL": "wss://ws.coinbase.com",
            "RATE_LIMIT_RPS": "10",
            "RATE_LIMIT_RPM": "100",
            "RATE_LIMIT_RPH": "1000"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, CoinbaseExchangeConfig)
            assert config.api_url == "https://api.coinbase.com"
            assert config.websocket_url == "wss://ws.coinbase.com"
            assert config.rate_limits.requests_per_second == 10

    def test_build_coinbase_exchange_missing_api_key(self):
        """Test building Coinbase exchange with missing API key."""
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Failure)
            assert "COINBASE_API_KEY not set" in result.failure()

    def test_build_coinbase_exchange_invalid_api_key(self):
        """Test building Coinbase exchange with invalid API key."""
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "short",  # Too short
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Failure)
            assert "too short" in result.failure()

    def test_build_bluefin_exchange_from_env_success(self):
        """Test building Bluefin exchange config from environment."""
        env_vars = {
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "testnet",
            "BLUEFIN_RPC_URL": "https://sui-testnet.bluefin.io",
            "RATE_LIMIT_RPS": "5",
            "RATE_LIMIT_RPM": "50",
            "RATE_LIMIT_RPH": "500"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, BluefinExchangeConfig)
            assert config.network == "testnet"
            assert config.rpc_url == "https://sui-testnet.bluefin.io"
            assert config.rate_limits.requests_per_second == 5

    def test_build_bluefin_exchange_invalid_network(self):
        """Test building Bluefin exchange with invalid network."""
        env_vars = {
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "invalid_network"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid Bluefin network" in result.failure()

    def test_build_bluefin_exchange_with_defaults(self):
        """Test building Bluefin exchange with default values."""
        env_vars = {
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, BluefinExchangeConfig)
            assert config.network == "mainnet"  # Default
            assert "sui-mainnet.bluefin.io" in config.rpc_url  # Default

    def test_build_binance_exchange_from_env_success(self):
        """Test building Binance exchange config from environment."""
        env_vars = {
            "EXCHANGE_TYPE": "binance",
            "BINANCE_API_KEY": "binance_api_key_1234567890",
            "BINANCE_API_SECRET": "binance_secret_1234567890abcdef",
            "BINANCE_TESTNET": "true",
            "RATE_LIMIT_RPS": "20",
            "RATE_LIMIT_RPM": "200",
            "RATE_LIMIT_RPH": "2000"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, BinanceExchangeConfig)
            assert config.testnet is True
            assert config.rate_limits.requests_per_second == 20

    def test_build_exchange_unknown_type(self):
        """Test building exchange with unknown type."""
        env_vars = {"EXCHANGE_TYPE": "unknown_exchange"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Unknown exchange type" in result.failure()

    def test_build_exchange_default_type(self):
        """Test building exchange with default type (Coinbase)."""
        env_vars = {
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config, CoinbaseExchangeConfig)

    def test_build_exchange_invalid_rate_limits(self):
        """Test building exchange with invalid rate limits."""
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "RATE_LIMIT_RPS": "invalid"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid integer" in result.failure()


class TestSystemConfigBuilder:
    """Test system configuration builder from environment variables."""

    def test_build_system_config_from_env_success(self):
        """Test building system config from environment."""
        env_vars = {
            "TRADING_PAIRS": "BTC-USD,ETH-USD,SOL-USD",
            "TRADING_INTERVAL": "5m",
            "TRADING_MODE": "paper",
            "LOG_LEVEL": "WARNING",
            "ENABLE_WEBSOCKET": "true",
            "ENABLE_MEMORY": "true",
            "ENABLE_BACKTESTING": "false",
            "MAX_CONCURRENT_POSITIONS": "5",
            "DEFAULT_POSITION_SIZE": "0.2"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert len(config.trading_pairs) == 3
            assert config.interval.value == "5m"
            assert config.log_level.value == "WARNING"
            assert config.features.enable_websocket is True
            assert config.features.enable_memory is True
            assert config.features.enable_backtesting is False
            assert config.max_concurrent_positions == 5
            assert config.default_position_size.value == 0.2

    def test_build_system_config_with_defaults(self):
        """Test building system config with default values."""
        with patch.dict(os.environ, {}, clear=True):
            result = build_system_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert len(config.trading_pairs) == 1
            assert config.trading_pairs[0].value == "BTC-USD"  # Default
            assert config.interval.value == "1m"  # Default
            assert config.max_concurrent_positions == 3  # Default
            assert config.default_position_size.value == 0.1  # Default

    def test_build_system_config_invalid_trading_pairs(self):
        """Test building system config with invalid trading pairs."""
        env_vars = {"TRADING_PAIRS": "INVALID_PAIR"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid trading pair" in result.failure()

    def test_build_system_config_invalid_max_positions(self):
        """Test building system config with invalid max positions."""
        env_vars = {"MAX_CONCURRENT_POSITIONS": "invalid"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid integer" in result.failure()

    def test_build_system_config_invalid_position_size(self):
        """Test building system config with invalid position size."""
        env_vars = {"DEFAULT_POSITION_SIZE": "invalid"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid float" in result.failure()


class TestBacktestConfigBuilder:
    """Test backtest configuration builder from environment variables."""

    def test_build_backtest_config_from_env_success(self):
        """Test building backtest config from environment."""
        env_vars = {
            "BACKTEST_START_DATE": "2024-01-01",
            "BACKTEST_END_DATE": "2024-06-30",
            "BACKTEST_INITIAL_CAPITAL": "50000.0",
            "BACKTEST_CURRENCY": "EUR",
            "BACKTEST_MAKER_FEE": "0.0005",
            "BACKTEST_TAKER_FEE": "0.001",
            "BACKTEST_SLIPPAGE": "0.0002",
            "BACKTEST_USE_LIMIT_ORDERS": "false"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_backtest_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert config.start_date.year == 2024
            assert config.start_date.month == 1
            assert config.end_date.month == 6
            assert config.initial_capital.amount == 50000.0
            assert config.initial_capital.currency == "EUR"
            assert config.fee_structure.maker_fee.value == 0.0005
            assert config.use_limit_orders is False

    def test_build_backtest_config_with_defaults(self):
        """Test building backtest config with default values."""
        with patch.dict(os.environ, {}, clear=True):
            result = build_backtest_config_from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert config.start_date.year == 2024
            assert config.end_date.year == 2024
            assert config.initial_capital.amount == 10000.0  # Default
            assert config.initial_capital.currency == "USD"  # Default
            assert config.use_limit_orders is True  # Default

    def test_build_backtest_config_invalid_date(self):
        """Test building backtest config with invalid date."""
        env_vars = {"BACKTEST_START_DATE": "invalid-date"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_backtest_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid date format" in result.failure()

    def test_build_backtest_config_invalid_capital(self):
        """Test building backtest config with invalid capital."""
        env_vars = {"BACKTEST_INITIAL_CAPITAL": "invalid"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_backtest_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid float" in result.failure()

    def test_build_backtest_config_invalid_fee(self):
        """Test building backtest config with invalid fee."""
        env_vars = {"BACKTEST_MAKER_FEE": "invalid"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_backtest_config_from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid float" in result.failure()


class TestCompleteConfigBuilder:
    """Test building complete configuration from environment."""

    def test_build_complete_config_from_env_success(self):
        """Test building complete configuration from environment."""
        env_vars = {
            # Strategy config
            "STRATEGY_TYPE": "llm",
            "LLM_MODEL": "gpt-4",
            
            # Exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            
            # System config
            "TRADING_PAIRS": "BTC-USD",
            "TRADING_MODE": "paper",
            
            # Backtest config (optional)
            "ENABLE_BACKTESTING": "true",
            "BACKTEST_START_DATE": "2024-01-01",
            "BACKTEST_END_DATE": "2024-12-31"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import Config
            
            result = Config.from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config.strategy, LLMStrategyConfig)
            assert isinstance(config.exchange, CoinbaseExchangeConfig)
            assert config.backtest is not None  # Backtest enabled

    def test_build_complete_config_without_backtest(self):
        """Test building complete configuration without backtest."""
        env_vars = {
            # Strategy config
            "STRATEGY_TYPE": "momentum",
            
            # Exchange config
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            
            # System config
            "ENABLE_BACKTESTING": "false"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import Config
            
            result = Config.from_env()
            
            assert isinstance(result, Success)
            config = result.success()
            assert isinstance(config.strategy, MomentumStrategyConfig)
            assert isinstance(config.exchange, BluefinExchangeConfig)
            assert config.backtest is None  # Backtest disabled

    def test_build_complete_config_strategy_failure(self):
        """Test building complete configuration with strategy failure."""
        env_vars = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "invalid",  # Invalid integer
            
            # Valid exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import Config
            
            result = Config.from_env()
            
            assert isinstance(result, Failure)
            assert "Invalid integer" in result.failure()

    def test_build_complete_config_exchange_failure(self):
        """Test building complete configuration with exchange failure."""
        env_vars = {
            # Valid strategy config
            "STRATEGY_TYPE": "llm",
            
            # Invalid exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "short"  # Too short
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import Config
            
            result = Config.from_env()
            
            assert isinstance(result, Failure)
            assert "too short" in result.failure()


class TestConfigBuilderEdgeCases:
    """Test edge cases in configuration builders."""

    def test_empty_environment_variables(self):
        """Test builders with completely empty environment."""
        with patch.dict(os.environ, {}, clear=True):
            # Strategy should default to LLM
            strategy_result = build_strategy_config_from_env()
            assert isinstance(strategy_result, Success)
            
            # Exchange should fail due to missing API keys
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            
            # System should work with defaults
            system_result = build_system_config_from_env()
            assert isinstance(system_result, Success)

    def test_partial_environment_variables(self):
        """Test builders with partial environment variables."""
        env_vars = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "30",  # Only partial config
            "EXCHANGE_TYPE": "coinbase"  # Missing API keys
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Strategy should work with defaults for missing values
            strategy_result = build_strategy_config_from_env()
            assert isinstance(strategy_result, Success)
            
            # Exchange should fail due to missing API keys
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)

    def test_malformed_environment_variables(self):
        """Test builders with malformed environment variables."""
        env_vars = {
            "RATE_LIMIT_RPS": "not_a_number",
            "LLM_TEMPERATURE": "not_a_float",
            "ENABLE_MEMORY": "not_a_boolean"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Should handle malformed values gracefully
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            
            strategy_result = build_strategy_config_from_env()
            assert isinstance(strategy_result, Failure)
            
            # Boolean parsing is more forgiving, defaults to False
            system_result = build_system_config_from_env()
            assert isinstance(system_result, Success)
            config = system_result.success()
            assert config.features.enable_memory is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])