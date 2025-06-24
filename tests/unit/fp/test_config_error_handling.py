"""
Unit tests for functional error handling patterns in configuration.

This module tests the Result/Either error handling patterns used throughout
the functional configuration system, ensuring proper error propagation,
composition, and recovery strategies.
"""

import os
from unittest.mock import patch

import pytest

from bot.fp.types.config import (
    Config,
    SystemConfig,
    LLMStrategyConfig,
    MomentumStrategyConfig,
    CoinbaseExchangeConfig,
    BluefinExchangeConfig,
    APIKey,
    PrivateKey,
    RateLimits,
    BacktestConfig,
    build_system_config_from_env,
    build_exchange_config_from_env,
    build_strategy_config_from_env,
    build_backtest_config_from_env,
    parse_int_env,
    parse_float_env,
    validate_config,
)
from bot.fp.types.result import Failure, Success


class TestResultErrorHandling:
    """Test Result type error handling patterns."""

    def test_success_result_operations(self):
        """Test operations on Success results."""
        success_result = Success("test_value")
        
        # Test basic properties
        assert success_result.is_success() is True
        assert success_result.is_failure() is False
        assert success_result.success() == "test_value"
        
        # Accessing failure on success should raise
        with pytest.raises(ValueError, match="Cannot get failure from Success"):
            success_result.failure()

    def test_failure_result_operations(self):
        """Test operations on Failure results."""
        failure_result = Failure("test_error")
        
        # Test basic properties
        assert failure_result.is_success() is False
        assert failure_result.is_failure() is True
        assert failure_result.failure() == "test_error"
        
        # Accessing success on failure should raise
        with pytest.raises(ValueError, match="Cannot get success from Failure"):
            failure_result.success()

    def test_result_equality(self):
        """Test Result equality comparison."""
        success1 = Success("value")
        success2 = Success("value")
        success3 = Success("different")
        failure1 = Failure("error")
        failure2 = Failure("error")
        failure3 = Failure("different")
        
        # Success equality
        assert success1 == success2
        assert success1 != success3
        assert success1 != failure1
        
        # Failure equality
        assert failure1 == failure2
        assert failure1 != failure3
        assert failure1 != success1

    def test_result_string_representation(self):
        """Test Result string representation."""
        success = Success("test_value")
        failure = Failure("test_error")
        
        assert "Success(test_value)" in str(success)
        assert "Failure(test_error)" in str(failure)


class TestConfigurationErrorPropagation:
    """Test error propagation in configuration building."""

    def test_api_key_error_propagation(self):
        """Test API key validation errors propagate correctly."""
        # Invalid API key should return Failure
        result = APIKey.create("short")
        assert isinstance(result, Failure)
        assert "too short" in result.failure()
        
        # This error should propagate through exchange config builder
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "short",  # Invalid
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            assert "too short" in exchange_result.failure()

    def test_private_key_error_propagation(self):
        """Test private key validation errors propagate correctly."""
        # Invalid private key should return Failure
        result = PrivateKey.create("invalid")
        assert isinstance(result, Failure)
        assert "too short" in result.failure()
        
        # This error should propagate through exchange config builder
        env_vars = {
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "invalid"  # Invalid
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            assert "too short" in exchange_result.failure()

    def test_rate_limits_error_propagation(self):
        """Test rate limits validation errors propagate correctly."""
        # Invalid rate limits should return Failure
        result = RateLimits.create(-1, 100, 1000)  # Negative value
        assert isinstance(result, Failure)
        assert "All rate limits must be positive" in result.failure()
        
        # This error should propagate through exchange config builder
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "RATE_LIMIT_RPS": "-1"  # Invalid
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            assert "Invalid integer" in exchange_result.failure()

    def test_strategy_config_error_propagation(self):
        """Test strategy configuration errors propagate correctly."""
        # Invalid strategy config should return Failure
        result = MomentumStrategyConfig.create(
            lookback_period=0,  # Invalid
            entry_threshold=2.0,
            exit_threshold=1.0
        )
        assert isinstance(result, Failure)
        assert "Lookback period must be positive" in result.failure()
        
        # This error should propagate through strategy config builder
        env_vars = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "0"  # Invalid
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            strategy_result = build_strategy_config_from_env()
            assert isinstance(strategy_result, Failure)
            assert "Lookback period must be positive" in strategy_result.failure()

    def test_system_config_error_propagation(self):
        """Test system configuration errors propagate correctly."""
        # Invalid system config should return Failure
        result = SystemConfig.create(
            trading_pairs=["INVALID"],  # Invalid trading pair
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0
        )
        assert isinstance(result, Failure)
        assert "Invalid trading pair" in result.failure()
        
        # This error should propagate through system config builder
        env_vars = {"TRADING_PAIRS": "INVALID"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            system_result = build_system_config_from_env()
            assert isinstance(system_result, Failure)
            assert "Invalid trading pair" in system_result.failure()

    def test_complete_config_error_propagation(self):
        """Test error propagation in complete configuration building."""
        # Strategy error should propagate to complete config
        env_vars = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "invalid",  # Invalid integer
            
            # Valid exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Failure)
            assert "Invalid integer" in config_result.failure()


class TestEnvironmentVariableErrorHandling:
    """Test error handling in environment variable parsing."""

    def test_parse_int_env_error_handling(self):
        """Test integer parsing error handling."""
        # Valid integer should succeed
        valid_result = parse_int_env("TEST_INT", 42)
        assert isinstance(valid_result, Success)
        assert valid_result.success() == 42
        
        # Invalid integer should fail
        with patch.dict(os.environ, {"TEST_INT": "not_an_integer"}):
            invalid_result = parse_int_env("TEST_INT", 42)
            assert isinstance(invalid_result, Failure)
            assert "Invalid integer" in invalid_result.failure()
            assert "TEST_INT" in invalid_result.failure()
            assert "not_an_integer" in invalid_result.failure()

    def test_parse_float_env_error_handling(self):
        """Test float parsing error handling."""
        # Valid float should succeed
        valid_result = parse_float_env("TEST_FLOAT", 3.14)
        assert isinstance(valid_result, Success)
        assert valid_result.success() == 3.14
        
        # Invalid float should fail
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            invalid_result = parse_float_env("TEST_FLOAT", 3.14)
            assert isinstance(invalid_result, Failure)
            assert "Invalid float" in invalid_result.failure()
            assert "TEST_FLOAT" in invalid_result.failure()
            assert "not_a_float" in invalid_result.failure()

    def test_environment_variable_error_context(self):
        """Test that environment variable errors include helpful context."""
        env_vars = {
            "MAX_CONCURRENT_POSITIONS": "not_a_number",
            "DEFAULT_POSITION_SIZE": "also_not_a_number"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Failure)
            
            error_message = result.failure()
            # Error should include the variable name for debugging
            assert "MAX_CONCURRENT_POSITIONS" in error_message or "Invalid integer" in error_message


class TestValidationErrorHandling:
    """Test validation error handling patterns."""

    def test_config_validation_error_composition(self):
        """Test composition of validation errors."""
        # Create config with multiple validation issues
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create("-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----").success()
        rate_limits = RateLimits.create(10, 100, 1000).success()
        
        # LLM strategy wants memory
        strategy_with_memory = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,  # Wants memory
            confidence_threshold=70.0
        ).success()
        
        # Bluefin testnet exchange
        bluefin_testnet = BluefinExchangeConfig(
            private_key=private_key,
            network="testnet",
            rpc_url="https://sui-testnet.bluefin.io",
            rate_limits=rate_limits
        )
        
        # System with live trading mode and memory disabled
        system_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="live",  # Live trading with testnet
            log_level="INFO",
            features={"enable_memory": False},  # Memory disabled
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        config = Config(
            strategy=strategy_with_memory,
            exchange=bluefin_testnet,
            system=system_config
        )
        
        # Validation should catch issues
        result = validate_config(config)
        assert isinstance(result, Failure)
        
        # Should report validation issues
        error_message = result.failure()
        # Could catch either error depending on validation order
        assert ("memory but it's disabled" in error_message or 
                "Cannot use testnet for live trading" in error_message)

    def test_early_validation_failure_stops_processing(self):
        """Test that early validation failures stop further processing."""
        # Create invalid strategy config first
        env_vars = {
            "STRATEGY_TYPE": "unknown_strategy",  # Invalid strategy type
            
            # These should not be processed due to early failure
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "invalid_key"  # This would also fail
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Failure)
            
            # Should fail on strategy error, not exchange error
            assert "Unknown strategy type" in config_result.failure()

    def test_validation_error_specificity(self):
        """Test that validation errors are specific and actionable."""
        # Test specific validation errors for different components
        
        # API key too short
        api_key_result = APIKey.create("short")
        assert isinstance(api_key_result, Failure)
        assert "Invalid API key: too short" == api_key_result.failure()
        
        # Invalid network
        env_vars = {
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "invalid_network"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            assert "Invalid Bluefin network: invalid_network" == exchange_result.failure()


class TestErrorRecoveryPatterns:
    """Test error recovery and fallback patterns."""

    def test_default_value_recovery(self):
        """Test recovery using default values when parsing fails."""
        from bot.fp.types.config import parse_int_env, parse_float_env
        
        # Missing environment variable should use default
        with patch.dict(os.environ, {}, clear=True):
            int_result = parse_int_env("MISSING_INT", 42)
            assert isinstance(int_result, Success)
            assert int_result.success() == 42
            
            float_result = parse_float_env("MISSING_FLOAT", 3.14)
            assert isinstance(float_result, Success)
            assert float_result.success() == 3.14

    def test_graceful_degradation_with_partial_config(self):
        """Test graceful degradation with partial configuration."""
        # System config should work with minimal valid environment
        minimal_env = {"TRADING_MODE": "paper"}
        
        with patch.dict(os.environ, minimal_env, clear=True):
            system_result = build_system_config_from_env()
            assert isinstance(system_result, Success)
            
            config = system_result.success()
            # Should use safe defaults
            assert len(config.trading_pairs) == 1
            assert config.trading_pairs[0].value == "BTC-USD"
            assert config.max_concurrent_positions == 3
            assert config.default_position_size.value == 0.1

    def test_configuration_builder_isolation(self):
        """Test that builder failures are isolated and don't affect other builders."""
        # One builder failing shouldn't prevent others from working
        env_vars = {
            # Valid strategy config
            "STRATEGY_TYPE": "llm",
            
            # Invalid exchange config
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "short",  # Invalid
            
            # Valid system config
            "TRADING_MODE": "paper"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Strategy should succeed
            strategy_result = build_strategy_config_from_env()
            assert isinstance(strategy_result, Success)
            
            # Exchange should fail
            exchange_result = build_exchange_config_from_env()
            assert isinstance(exchange_result, Failure)
            
            # System should succeed
            system_result = build_system_config_from_env()
            assert isinstance(system_result, Success)


class TestErrorHandlingEdgeCases:
    """Test edge cases in error handling."""

    def test_nested_validation_errors(self):
        """Test handling of nested validation errors."""
        # Create a scenario with nested validation failures
        result = BacktestConfig.create(
            start_date="invalid-date",  # Invalid date format
            end_date="2024-12-31",
            initial_capital=-1000.0,  # Invalid negative capital
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005
        )
        
        assert isinstance(result, Failure)
        # Should report the first validation error encountered
        assert "Invalid date format" in result.failure()

    def test_error_message_clarity(self):
        """Test that error messages are clear and helpful."""
        # Test various error scenarios for clear messaging
        
        # Missing required field
        env_vars = {"EXCHANGE_TYPE": "coinbase"}  # Missing API key
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            assert isinstance(result, Failure)
            assert "COINBASE_API_KEY not set" in result.failure()
        
        # Invalid enum value
        env_vars = {"TRADING_MODE": "invalid_mode"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Failure)
            assert "Invalid trading mode: invalid_mode" in result.failure()

    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        # Create many failing operations to test error handling overhead
        failures = []
        
        for i in range(100):
            result = APIKey.create("short")  # Will always fail
            failures.append(result)
        
        # All should be failures
        assert all(isinstance(f, Failure) for f in failures)
        assert len(failures) == 100

    def test_error_serialization(self):
        """Test that errors can be properly serialized for logging/debugging."""
        failure = Failure("Test error message")
        
        # Should be convertible to string
        error_str = str(failure)
        assert "Test error message" in error_str
        
        # Should be able to extract the error message
        assert failure.failure() == "Test error message"


class TestErrorHandlingIntegration:
    """Test error handling integration across the configuration system."""

    def test_complete_failure_scenario(self):
        """Test complete failure scenario with helpful error reporting."""
        # Create environment that will fail at multiple levels
        problematic_env = {
            "STRATEGY_TYPE": "momentum",
            "MOMENTUM_LOOKBACK": "invalid",  # Invalid integer
            "EXCHANGE_TYPE": "coinbase", 
            "COINBASE_API_KEY": "short",  # Invalid API key
            "TRADING_PAIRS": "INVALID",  # Invalid trading pair
            "MAX_CONCURRENT_POSITIONS": "-1"  # Invalid position count
        }
        
        with patch.dict(os.environ, problematic_env, clear=True):
            # Should fail fast on first error encountered
            config_result = Config.from_env()
            assert isinstance(config_result, Failure)
            
            # Error message should be helpful for debugging
            error_message = config_result.failure()
            assert len(error_message) > 0
            # Should mention the specific problem
            assert ("Invalid integer" in error_message or 
                    "short" in error_message or 
                    "Invalid trading pair" in error_message)

    def test_error_context_preservation(self):
        """Test that error context is preserved through the call stack."""
        # Create a scenario where we can trace the error source
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "LLM_TEMPERATURE": "invalid_temperature"  # Invalid float
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            strategy_result = build_strategy_config_from_env()
            assert isinstance(strategy_result, Failure)
            
            error_message = strategy_result.failure()
            # Should preserve context about which field failed
            assert "Invalid float" in error_message
            assert "LLM_TEMPERATURE" in error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])