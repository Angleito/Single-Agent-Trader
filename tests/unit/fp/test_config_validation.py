"""
Unit tests for functional programming configuration validation.

This module tests the validation pipeline for functional configuration with
immutable data structures and Result/Either error handling patterns.
"""

import os
from unittest.mock import patch

import pytest

from bot.fp.types.config import (
    APIKey,
    BacktestConfig,
    BluefinExchangeConfig,
    CoinbaseExchangeConfig,
    Config,
    LLMStrategyConfig,
    PrivateKey,
    RateLimits,
    SystemConfig,
    validate_config,
)
from bot.fp.types.result import Failure, Success


class TestConfigValidation:
    """Test configuration validation functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Valid API key and private key
        self.api_key = APIKey.create("sk-1234567890abcdefghij").success()
        self.private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        self.rate_limits = RateLimits.create(10, 100, 1000).success()

        # Valid strategy config
        self.llm_strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0,
        ).success()

        # Valid exchange config
        self.coinbase_exchange = CoinbaseExchangeConfig(
            api_key=self.api_key,
            private_key=self.private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=self.rate_limits,
        )

        # Valid system config
        self.system_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": False},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = Config(
            strategy=self.llm_strategy,
            exchange=self.coinbase_exchange,
            system=self.system_config,
        )

        result = validate_config(config)
        assert isinstance(result, Success)
        validated_config = result.success()
        assert validated_config == config  # Should return the same config

    def test_validate_config_llm_memory_mismatch_failure(self):
        """Test validation failure when LLM strategy requires memory but system disables it."""
        # Create LLM strategy that wants memory
        strategy_with_memory = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,  # Wants memory
            confidence_threshold=70.0,
        ).success()

        # System config with memory disabled
        system_without_memory = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": False},  # Memory disabled
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=strategy_with_memory,
            exchange=self.coinbase_exchange,
            system=system_without_memory,
        )

        result = validate_config(config)
        assert isinstance(result, Failure)
        assert "memory but it's disabled" in result.failure()

    def test_validate_config_llm_memory_match_success(self):
        """Test validation success when LLM strategy and system memory settings match."""
        # Create LLM strategy that wants memory
        strategy_with_memory = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,  # Wants memory
            confidence_threshold=70.0,
        ).success()

        # System config with memory enabled
        system_with_memory = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": True},  # Memory enabled
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=strategy_with_memory,
            exchange=self.coinbase_exchange,
            system=system_with_memory,
        )

        result = validate_config(config)
        assert isinstance(result, Success)

    def test_validate_config_bluefin_testnet_live_trading_failure(self):
        """Test validation failure when using Bluefin testnet for live trading."""
        # Create Bluefin testnet exchange config
        bluefin_testnet = BluefinExchangeConfig(
            private_key=self.private_key,
            network="testnet",
            rpc_url="https://sui-testnet.bluefin.io",
            rate_limits=self.rate_limits,
        )

        # System config with live trading mode
        system_live = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="live",  # Live trading
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=self.llm_strategy, exchange=bluefin_testnet, system=system_live
        )

        result = validate_config(config)
        assert isinstance(result, Failure)
        assert "Cannot use testnet for live trading" in result.failure()

    def test_validate_config_bluefin_testnet_paper_trading_success(self):
        """Test validation success when using Bluefin testnet for paper trading."""
        # Create Bluefin testnet exchange config
        bluefin_testnet = BluefinExchangeConfig(
            private_key=self.private_key,
            network="testnet",
            rpc_url="https://sui-testnet.bluefin.io",
            rate_limits=self.rate_limits,
        )

        # System config with paper trading mode
        system_paper = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",  # Paper trading
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=self.llm_strategy, exchange=bluefin_testnet, system=system_paper
        )

        result = validate_config(config)
        assert isinstance(result, Success)

    def test_validate_config_backtest_without_backtest_mode_failure(self):
        """Test validation failure when backtest config is provided but not in backtest mode."""
        # Create backtest config
        backtest_config = BacktestConfig.create(
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0,
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005,
        ).success()

        # System config not in backtest mode
        system_paper = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",  # Not backtest mode
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=self.llm_strategy,
            exchange=self.coinbase_exchange,
            system=system_paper,
            backtest=backtest_config,
        )

        result = validate_config(config)
        assert isinstance(result, Failure)
        assert "Backtest config provided but not in backtest mode" in result.failure()

    def test_validate_config_backtest_mode_with_backtest_config_success(self):
        """Test validation success when in backtest mode with backtest config."""
        # Create backtest config
        backtest_config = BacktestConfig.create(
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0,
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005,
        ).success()

        # System config in backtest mode
        system_backtest = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="backtest",  # Backtest mode
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=self.llm_strategy,
            exchange=self.coinbase_exchange,
            system=system_backtest,
            backtest=backtest_config,
        )

        result = validate_config(config)
        assert isinstance(result, Success)

    def test_validate_config_backtest_mode_without_backtest_config_success(self):
        """Test validation success when in backtest mode without backtest config."""
        # System config in backtest mode
        system_backtest = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="backtest",  # Backtest mode
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config = Config(
            strategy=self.llm_strategy,
            exchange=self.coinbase_exchange,
            system=system_backtest,
            backtest=None,  # No backtest config
        )

        result = validate_config(config)
        assert isinstance(result, Success)


class TestConfigValidationEdgeCases:
    """Test edge cases in configuration validation."""

    def test_validate_config_with_all_strategy_types(self):
        """Test configuration validation with different strategy types."""
        from bot.fp.types.config import (
            MeanReversionStrategyConfig,
            MomentumStrategyConfig,
        )

        # Setup common components
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
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        # Test LLM strategy
        llm_strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0,
        ).success()

        config_llm = Config(strategy=llm_strategy, exchange=exchange, system=system)
        result_llm = validate_config(config_llm)
        assert isinstance(result_llm, Success)

        # Test Momentum strategy
        momentum_strategy = MomentumStrategyConfig.create(
            lookback_period=20, entry_threshold=2.0, exit_threshold=1.0
        ).success()

        config_momentum = Config(
            strategy=momentum_strategy, exchange=exchange, system=system
        )
        result_momentum = validate_config(config_momentum)
        assert isinstance(result_momentum, Success)

        # Test Mean Reversion strategy
        mean_reversion_strategy = MeanReversionStrategyConfig.create(
            window_size=50,
            std_deviations=2.0,
            min_volatility=0.1,
            max_holding_period=100,
        ).success()

        config_mean_reversion = Config(
            strategy=mean_reversion_strategy, exchange=exchange, system=system
        )
        result_mean_reversion = validate_config(config_mean_reversion)
        assert isinstance(result_mean_reversion, Success)

    def test_validate_config_with_all_exchange_types(self):
        """Test configuration validation with different exchange types."""
        from bot.fp.types.config import BinanceExchangeConfig

        # Setup common components
        private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        rate_limits = RateLimits.create(10, 100, 1000).success()

        strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0,
        ).success()

        system = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        # Test Coinbase exchange
        coinbase_exchange = CoinbaseExchangeConfig(
            api_key=api_key,
            private_key=private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=rate_limits,
        )

        config_coinbase = Config(
            strategy=strategy, exchange=coinbase_exchange, system=system
        )
        result_coinbase = validate_config(config_coinbase)
        assert isinstance(result_coinbase, Success)

        # Test Bluefin exchange
        bluefin_exchange = BluefinExchangeConfig(
            private_key=private_key,
            network="mainnet",
            rpc_url="https://sui-mainnet.bluefin.io",
            rate_limits=rate_limits,
        )

        config_bluefin = Config(
            strategy=strategy, exchange=bluefin_exchange, system=system
        )
        result_bluefin = validate_config(config_bluefin)
        assert isinstance(result_bluefin, Success)

        # Test Binance exchange
        binance_exchange = BinanceExchangeConfig(
            api_key=api_key,
            api_secret=api_key,  # Using same key for simplicity
            testnet=False,
            rate_limits=rate_limits,
        )

        config_binance = Config(
            strategy=strategy, exchange=binance_exchange, system=system
        )
        result_binance = validate_config(config_binance)
        assert isinstance(result_binance, Success)

    def test_validate_config_with_all_trading_modes(self):
        """Test configuration validation with different trading modes."""
        # Setup common components
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        rate_limits = RateLimits.create(10, 100, 1000).success()

        strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0,
        ).success()

        exchange = CoinbaseExchangeConfig(
            api_key=api_key,
            private_key=private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=rate_limits,
        )

        # Test paper trading mode
        system_paper = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config_paper = Config(strategy=strategy, exchange=exchange, system=system_paper)
        result_paper = validate_config(config_paper)
        assert isinstance(result_paper, Success)

        # Test live trading mode
        system_live = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="live",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config_live = Config(strategy=strategy, exchange=exchange, system=system_live)
        result_live = validate_config(config_live)
        assert isinstance(result_live, Success)

        # Test backtest mode
        system_backtest = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="backtest",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config_backtest = Config(
            strategy=strategy, exchange=exchange, system=system_backtest
        )
        result_backtest = validate_config(config_backtest)
        assert isinstance(result_backtest, Success)

    def test_validate_config_complex_feature_combinations(self):
        """Test validation with complex feature flag combinations."""
        # Setup common components
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

        # Test all features enabled
        strategy_with_memory = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=70.0,
        ).success()

        system_all_features = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={
                "enable_websocket": True,
                "enable_memory": True,
                "enable_backtesting": True,
                "enable_paper_trading": True,
                "enable_risk_management": True,
                "enable_notifications": True,
                "enable_metrics": True,
            },
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config_all_features = Config(
            strategy=strategy_with_memory, exchange=exchange, system=system_all_features
        )

        result_all_features = validate_config(config_all_features)
        assert isinstance(result_all_features, Success)

        # Test minimal features
        strategy_no_memory = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0,
        ).success()

        system_minimal_features = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={
                "enable_websocket": False,
                "enable_memory": False,
                "enable_backtesting": False,
                "enable_paper_trading": True,  # Must be enabled for paper trading
                "enable_risk_management": True,  # Must be enabled for safety
                "enable_notifications": False,
                "enable_metrics": False,
            },
            max_concurrent_positions=3,
            default_position_size=10.0,
        ).success()

        config_minimal_features = Config(
            strategy=strategy_no_memory,
            exchange=exchange,
            system=system_minimal_features,
        )

        result_minimal_features = validate_config(config_minimal_features)
        assert isinstance(result_minimal_features, Success)


class TestConfigValidationIntegration:
    """Test configuration validation integration with builders."""

    def test_validate_config_from_env_success(self):
        """Test validation of configuration built from environment variables."""
        env_vars = {
            # Strategy config
            "STRATEGY_TYPE": "llm",
            "LLM_MODEL": "gpt-4",
            "LLM_USE_MEMORY": "false",
            # Exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            # System config
            "TRADING_PAIRS": "BTC-USD",
            "TRADING_MODE": "paper",
            "ENABLE_MEMORY": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import Config

            # Build config from environment
            build_result = Config.from_env()
            assert isinstance(build_result, Success)

            # Validate the built config
            config = build_result.success()
            validation_result = validate_config(config)
            assert isinstance(validation_result, Success)

    def test_validate_config_from_env_memory_mismatch(self):
        """Test validation failure from environment with memory mismatch."""
        env_vars = {
            # Strategy config wants memory
            "STRATEGY_TYPE": "llm",
            "LLM_USE_MEMORY": "true",
            # Exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            # System config disables memory
            "ENABLE_MEMORY": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import Config

            # Build config from environment
            build_result = Config.from_env()
            assert isinstance(build_result, Success)

            # Validation should fail
            config = build_result.success()
            validation_result = validate_config(config)
            assert isinstance(validation_result, Failure)
            assert "memory but it's disabled" in validation_result.failure()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
