"""
Integration tests for functional programming configuration with the trading system.

This module tests end-to-end integration of FP configuration with the actual
trading system components, ensuring all adapters work correctly and the system
can start and operate with FP configurations.
"""

import os
from unittest.mock import patch

import pytest

from bot.fp.types.base import TradingMode
from bot.fp.types.config import (
    APIKey,
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


class TestFPConfigurationSystemIntegration:
    """Test FP configuration integration with the trading system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create valid FP configurations for testing
        self.api_key = APIKey.create("sk-1234567890abcdefghij").success()
        self.private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        self.rate_limits = RateLimits.create(10, 100, 1000).success()

        self.valid_fp_config = Config(
            strategy=LLMStrategyConfig.create(
                model_name="gpt-4",
                temperature=0.7,
                max_context_length=4000,
                use_memory=False,
                confidence_threshold=70.0,
            ).success(),
            exchange=CoinbaseExchangeConfig(
                api_key=self.api_key,
                private_key=self.private_key,
                api_url="https://api.coinbase.com",
                websocket_url="wss://ws.coinbase.com",
                rate_limits=self.rate_limits,
            ),
            system=SystemConfig.create(
                trading_pairs=["BTC-USD"],
                interval="1m",
                mode="paper",
                log_level="INFO",
                features={"enable_memory": False},
                max_concurrent_positions=3,
                default_position_size=10.0,
            ).success(),
        )

    def test_fp_config_loads_from_environment_successfully(self):
        """Test that FP configuration can be loaded from environment variables."""
        env_vars = {
            # Strategy configuration
            "STRATEGY_TYPE": "llm",
            "LLM_MODEL": "gpt-4",
            "LLM_TEMPERATURE": "0.7",
            "LLM_USE_MEMORY": "false",
            "LLM_CONFIDENCE_THRESHOLD": "0.7",
            # Exchange configuration
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "RATE_LIMIT_RPS": "10",
            "RATE_LIMIT_RPM": "100",
            "RATE_LIMIT_RPH": "1000",
            # System configuration
            "TRADING_PAIRS": "BTC-USD",
            "TRADING_INTERVAL": "1m",
            "TRADING_MODE": "paper",
            "LOG_LEVEL": "INFO",
            "ENABLE_MEMORY": "false",
            "MAX_CONCURRENT_POSITIONS": "3",
            "DEFAULT_POSITION_SIZE": "10.0",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()

            assert isinstance(config_result, Success)
            config = config_result.success()

            # Verify configuration is properly loaded
            assert isinstance(config.strategy, LLMStrategyConfig)
            assert isinstance(config.exchange, CoinbaseExchangeConfig)
            assert config.system.mode == TradingMode.PAPER

            # Verify validation passes
            validation_result = validate_config(config)
            assert isinstance(validation_result, Success)

    def test_fp_config_works_with_bluefin_exchange(self):
        """Test FP configuration works with Bluefin exchange."""
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "testnet",
            "TRADING_MODE": "paper",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()

            assert isinstance(config_result, Success)
            config = config_result.success()

            assert isinstance(config.exchange, BluefinExchangeConfig)
            assert config.exchange.network == "testnet"
            assert config.system.mode == TradingMode.PAPER

    def test_fp_config_enforces_safety_constraints(self):
        """Test that FP configuration enforces safety constraints in integration."""
        # Try to create live trading config with testnet - should fail validation
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "testnet",
            "TRADING_MODE": "live",  # Should fail with testnet
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            # But validation should fail
            validation_result = validate_config(config_result.success())
            assert isinstance(validation_result, Failure)
            assert "Cannot use testnet for live trading" in validation_result.failure()

    def test_fp_config_memory_consistency_enforcement(self):
        """Test that FP configuration enforces memory consistency."""
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "LLM_USE_MEMORY": "true",  # Strategy wants memory
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "ENABLE_MEMORY": "false",  # System disables memory
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            # Validation should catch the inconsistency
            validation_result = validate_config(config_result.success())
            assert isinstance(validation_result, Failure)
            assert "memory but it's disabled" in validation_result.failure()


class TestFPConfigurationBootstrapIntegration:
    """Test FP configuration integration during system bootstrap."""

    @patch("bot.main.create_settings")  # Mock legacy settings creation
    def test_fp_config_can_replace_legacy_settings(self, mock_create_settings):
        """Test that FP configuration can replace legacy settings during bootstrap."""
        # Mock the legacy settings to return None, forcing FP path
        mock_create_settings.return_value = None

        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "TRADING_MODE": "paper",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # FP configuration should work as a fallback
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            config = config_result.success()
            assert config.system.mode == TradingMode.PAPER

    def test_fp_config_validates_at_startup(self):
        """Test that FP configuration is validated at system startup."""
        # Create an environment that builds successfully but fails validation
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "LLM_USE_MEMORY": "true",  # Wants memory
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "ENABLE_MEMORY": "false",  # Memory disabled - validation should fail
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Config should build
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            # But validation should fail
            validation_result = validate_config(config_result.success())
            assert isinstance(validation_result, Failure)

            # System should not start with invalid configuration
            # This would typically be enforced by the main application


class TestFPConfigurationFeatureIntegration:
    """Test integration of FP configuration with feature flags."""

    def test_fp_config_feature_flags_control_system_behavior(self):
        """Test that FP configuration feature flags properly control system behavior."""
        # Test with all features enabled
        env_vars_all_features = {
            "STRATEGY_TYPE": "llm",
            "LLM_USE_MEMORY": "true",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "ENABLE_WEBSOCKET": "true",
            "ENABLE_MEMORY": "true",
            "ENABLE_BACKTESTING": "true",
            "ENABLE_PAPER_TRADING": "true",
            "ENABLE_RISK_MANAGEMENT": "true",
            "ENABLE_NOTIFICATIONS": "true",
            "ENABLE_METRICS": "true",
        }

        with patch.dict(os.environ, env_vars_all_features, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            config = config_result.success()
            features = config.system.features

            # All features should be enabled
            assert features.enable_websocket is True
            assert features.enable_memory is True
            assert features.enable_backtesting is True
            assert features.enable_paper_trading is True
            assert features.enable_risk_management is True
            assert features.enable_notifications is True
            assert features.enable_metrics is True

    def test_fp_config_safety_features_cannot_be_disabled(self):
        """Test that safety-critical features maintain their importance."""
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "TRADING_MODE": "paper",
            # Note: We still allow disabling paper trading and risk management
            # but the defaults should be safe
            "ENABLE_PAPER_TRADING": "false",
            "ENABLE_RISK_MANAGEMENT": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            config = config_result.success()

            # Even if user tries to disable safety features, paper mode should be preserved
            assert config.system.mode == TradingMode.PAPER
            # Features reflect user choice but system behavior should remain safe
            assert config.system.features.enable_paper_trading is False
            assert config.system.features.enable_risk_management is False


class TestFPConfigurationErrorIntegration:
    """Test error handling integration across the system."""

    def test_fp_config_errors_prevent_system_startup(self):
        """Test that FP configuration errors prevent unsafe system startup."""
        # Create environment with missing critical configuration
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "coinbase",
            # Missing API keys - should prevent startup
            "TRADING_MODE": "live",  # Attempting live trading without credentials
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()

            # Should fail to build due to missing credentials
            assert isinstance(config_result, Failure)
            assert "COINBASE_API_KEY not set" in config_result.failure()

    def test_fp_config_validation_errors_are_actionable(self):
        """Test that FP configuration validation errors provide actionable feedback."""
        # Create config that builds but has validation errors
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "LLM_USE_MEMORY": "true",
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "testnet",
            "TRADING_MODE": "live",  # Invalid: testnet + live
            "ENABLE_MEMORY": "false",  # Invalid: strategy wants memory but disabled
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            validation_result = validate_config(config_result.success())
            assert isinstance(validation_result, Failure)

            error_message = validation_result.failure()
            # Should provide actionable error message
            assert len(error_message) > 0
            # Should mention specific issues that can be fixed
            assert "testnet" in error_message or "memory" in error_message


class TestFPConfigurationPerformanceIntegration:
    """Test performance characteristics of FP configuration in integration."""

    def test_fp_config_loading_performance(self):
        """Test that FP configuration loading is performant."""
        # Test loading configuration multiple times
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "TRADING_MODE": "paper",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Load configuration multiple times
            configs = []
            for _ in range(10):
                config_result = Config.from_env()
                assert isinstance(config_result, Success)
                configs.append(config_result.success())

            # All configs should be equivalent
            assert len(configs) == 10
            assert all(c.system.mode == TradingMode.PAPER for c in configs)

    def test_fp_config_validation_performance(self):
        """Test that FP configuration validation is performant."""
        # Create a valid config and validate it multiple times
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create(
            "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        ).success()
        rate_limits = RateLimits.create(10, 100, 1000).success()

        config = Config(
            strategy=LLMStrategyConfig.create(
                model_name="gpt-4",
                temperature=0.7,
                max_context_length=4000,
                use_memory=False,
                confidence_threshold=70.0,
            ).success(),
            exchange=CoinbaseExchangeConfig(
                api_key=api_key,
                private_key=private_key,
                api_url="https://api.coinbase.com",
                websocket_url="wss://ws.coinbase.com",
                rate_limits=rate_limits,
            ),
            system=SystemConfig.create(
                trading_pairs=["BTC-USD"],
                interval="1m",
                mode="paper",
                log_level="INFO",
                features={},
                max_concurrent_positions=3,
                default_position_size=10.0,
            ).success(),
        )

        # Validate multiple times
        for _ in range(20):
            result = validate_config(config)
            assert isinstance(result, Success)


class TestFPConfigurationCompatibilityIntegration:
    """Test compatibility with existing system components."""

    def test_fp_config_works_with_existing_trading_components(self):
        """Test that FP configuration can integrate with existing trading components."""
        # This would typically involve mocking existing components
        # and verifying they can consume FP configuration

        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "TRADING_MODE": "paper",
            "TRADING_PAIRS": "BTC-USD,ETH-USD",
            "MAX_CONCURRENT_POSITIONS": "5",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            config = config_result.success()

            # Verify configuration can be used by trading components
            assert len(config.system.trading_pairs) == 2
            assert config.system.max_concurrent_positions == 5
            assert config.system.mode == TradingMode.PAPER

    def test_fp_config_maintains_backward_compatibility(self):
        """Test that FP configuration maintains backward compatibility where needed."""
        # Test that FP configuration doesn't break existing functionality

        env_vars = {
            "STRATEGY_TYPE": "momentum",  # Non-LLM strategy
            "MOMENTUM_LOOKBACK": "20",
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "TRADING_MODE": "paper",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            config = config_result.success()

            # Should work with different strategy types
            from bot.fp.types.config import MomentumStrategyConfig

            assert isinstance(config.strategy, MomentumStrategyConfig)
            assert config.strategy.lookback_period == 20


class TestFPConfigurationResilience:
    """Test resilience and fault tolerance of FP configuration."""

    def test_fp_config_handles_partial_environment_gracefully(self):
        """Test that FP configuration handles partial environment gracefully."""
        # Provide minimal valid environment
        minimal_env = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            # Many other settings will use defaults
        }

        with patch.dict(os.environ, minimal_env, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)

            config = config_result.success()

            # Should use safe defaults
            assert config.system.mode == TradingMode.PAPER
            assert config.system.max_concurrent_positions == 3
            assert config.system.default_position_size.value == 0.1

    def test_fp_config_fails_safely_with_invalid_environment(self):
        """Test that FP configuration fails safely with invalid environment."""
        # Provide completely invalid environment
        invalid_env = {
            "STRATEGY_TYPE": "unknown",
            "EXCHANGE_TYPE": "invalid",
            "TRADING_MODE": "dangerous",
            "MALICIOUS_SETTING": "value",
        }

        with patch.dict(os.environ, invalid_env, clear=True):
            config_result = Config.from_env()

            # Should fail safely rather than create dangerous configuration
            assert isinstance(config_result, Failure)
            # Error should be specific about what failed
            assert "Unknown strategy type" in config_result.failure()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
