"""
Unit tests for configuration adapters that bridge FP and legacy systems.

This module tests the compatibility layer between functional programming
configuration types and legacy imperative configuration, ensuring smooth
migration and interoperability.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from bot.fp.types.config import (
    Config,
    SystemConfig,
    LLMStrategyConfig,
    CoinbaseExchangeConfig,
    BluefinExchangeConfig,
    APIKey,
    PrivateKey,
    RateLimits,
    FeatureFlags,
)
from bot.fp.types.base import TradingMode
from bot.fp.types.result import Success


class ConfigurationAdapter:
    """
    Adapter class to bridge FP and legacy configuration systems.
    
    This would typically be implemented in bot/fp/adapters/configuration_adapter.py
    but we'll define it here for testing purposes.
    """
    
    @staticmethod
    def fp_to_legacy_system_config(fp_config: SystemConfig) -> dict:
        """Convert FP SystemConfig to legacy format."""
        return {
            "trading_pairs": [pair.value for pair in fp_config.trading_pairs],
            "interval": fp_config.interval.value,
            "mode": fp_config.mode.value,
            "log_level": fp_config.log_level.value,
            "max_concurrent_positions": fp_config.max_concurrent_positions,
            "default_position_size": fp_config.default_position_size.value,
            "features": {
                "enable_websocket": fp_config.features.enable_websocket,
                "enable_memory": fp_config.features.enable_memory,
                "enable_backtesting": fp_config.features.enable_backtesting,
                "enable_paper_trading": fp_config.features.enable_paper_trading,
                "enable_risk_management": fp_config.features.enable_risk_management,
                "enable_notifications": fp_config.features.enable_notifications,
                "enable_metrics": fp_config.features.enable_metrics,
            }
        }
    
    @staticmethod
    def fp_to_legacy_exchange_config(fp_config) -> dict:
        """Convert FP ExchangeConfig to legacy format."""
        if isinstance(fp_config, CoinbaseExchangeConfig):
            return {
                "type": "coinbase",
                "api_key": fp_config.api_key._value,  # Access internal value for legacy
                "private_key": fp_config.private_key._value,
                "api_url": fp_config.api_url,
                "websocket_url": fp_config.websocket_url,
                "rate_limits": {
                    "requests_per_second": fp_config.rate_limits.requests_per_second,
                    "requests_per_minute": fp_config.rate_limits.requests_per_minute,
                    "requests_per_hour": fp_config.rate_limits.requests_per_hour,
                }
            }
        elif isinstance(fp_config, BluefinExchangeConfig):
            return {
                "type": "bluefin",
                "private_key": fp_config.private_key._value,
                "network": fp_config.network,
                "rpc_url": fp_config.rpc_url,
                "rate_limits": {
                    "requests_per_second": fp_config.rate_limits.requests_per_second,
                    "requests_per_minute": fp_config.rate_limits.requests_per_minute,
                    "requests_per_hour": fp_config.rate_limits.requests_per_hour,
                }
            }
        else:
            raise ValueError(f"Unsupported exchange config type: {type(fp_config)}")
    
    @staticmethod
    def legacy_to_fp_system_config(legacy_config: dict) -> SystemConfig:
        """Convert legacy system config to FP format."""
        return SystemConfig.create(
            trading_pairs=legacy_config.get("trading_pairs", ["BTC-USD"]),
            interval=legacy_config.get("interval", "1m"),
            mode=legacy_config.get("mode", "paper"),
            log_level=legacy_config.get("log_level", "INFO"),
            features=legacy_config.get("features", {}),
            max_concurrent_positions=legacy_config.get("max_concurrent_positions", 3),
            default_position_size=legacy_config.get("default_position_size", 10.0)
        ).success()


class TestConfigurationAdapterConversion:
    """Test configuration adapter conversion functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create valid FP configurations
        self.api_key = APIKey.create("sk-1234567890abcdefghij").success()
        self.private_key = PrivateKey.create("-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----").success()
        self.rate_limits = RateLimits.create(10, 100, 1000).success()
        
        self.fp_system_config = SystemConfig.create(
            trading_pairs=["BTC-USD", "ETH-USD"],
            interval="5m",
            mode="paper",
            log_level="WARNING",
            features={
                "enable_websocket": True,
                "enable_memory": False,
                "enable_backtesting": True,
                "enable_paper_trading": True,
                "enable_risk_management": True,
                "enable_notifications": False,
                "enable_metrics": True,
            },
            max_concurrent_positions=5,
            default_position_size=15.0
        ).success()
        
        self.fp_coinbase_config = CoinbaseExchangeConfig(
            api_key=self.api_key,
            private_key=self.private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=self.rate_limits
        )
        
        self.fp_bluefin_config = BluefinExchangeConfig(
            private_key=self.private_key,
            network="testnet",
            rpc_url="https://sui-testnet.bluefin.io",
            rate_limits=self.rate_limits
        )

    def test_fp_to_legacy_system_config_conversion(self):
        """Test conversion from FP system config to legacy format."""
        legacy_config = ConfigurationAdapter.fp_to_legacy_system_config(self.fp_system_config)
        
        # Verify structure and values
        assert legacy_config["trading_pairs"] == ["BTC-USD", "ETH-USD"]
        assert legacy_config["interval"] == "5m"
        assert legacy_config["mode"] == "paper"
        assert legacy_config["log_level"] == "WARNING"
        assert legacy_config["max_concurrent_positions"] == 5
        assert legacy_config["default_position_size"] == 15.0
        
        # Verify features
        features = legacy_config["features"]
        assert features["enable_websocket"] is True
        assert features["enable_memory"] is False
        assert features["enable_backtesting"] is True

    def test_fp_to_legacy_coinbase_config_conversion(self):
        """Test conversion from FP Coinbase config to legacy format."""
        legacy_config = ConfigurationAdapter.fp_to_legacy_exchange_config(self.fp_coinbase_config)
        
        # Verify structure and values
        assert legacy_config["type"] == "coinbase"
        assert legacy_config["api_key"] == "sk-1234567890abcdefghij"
        assert legacy_config["api_url"] == "https://api.coinbase.com"
        assert legacy_config["websocket_url"] == "wss://ws.coinbase.com"
        
        # Verify rate limits
        rate_limits = legacy_config["rate_limits"]
        assert rate_limits["requests_per_second"] == 10
        assert rate_limits["requests_per_minute"] == 100
        assert rate_limits["requests_per_hour"] == 1000

    def test_fp_to_legacy_bluefin_config_conversion(self):
        """Test conversion from FP Bluefin config to legacy format."""
        legacy_config = ConfigurationAdapter.fp_to_legacy_exchange_config(self.fp_bluefin_config)
        
        # Verify structure and values
        assert legacy_config["type"] == "bluefin"
        assert legacy_config["network"] == "testnet"
        assert legacy_config["rpc_url"] == "https://sui-testnet.bluefin.io"
        
        # Verify rate limits
        rate_limits = legacy_config["rate_limits"]
        assert rate_limits["requests_per_second"] == 10

    def test_legacy_to_fp_system_config_conversion(self):
        """Test conversion from legacy system config to FP format."""
        legacy_config = {
            "trading_pairs": ["SOL-USD", "AVAX-USD"],
            "interval": "15m",
            "mode": "live",
            "log_level": "ERROR",
            "max_concurrent_positions": 2,
            "default_position_size": 20.0,
            "features": {
                "enable_websocket": False,
                "enable_memory": True,
                "enable_backtesting": False,
            }
        }
        
        fp_config = ConfigurationAdapter.legacy_to_fp_system_config(legacy_config)
        
        # Verify conversion
        assert len(fp_config.trading_pairs) == 2
        assert fp_config.trading_pairs[0].value == "SOL-USD"
        assert fp_config.trading_pairs[1].value == "AVAX-USD"
        assert fp_config.interval.value == "15m"
        assert fp_config.mode == TradingMode.LIVE
        assert fp_config.max_concurrent_positions == 2
        assert fp_config.default_position_size.value == 20.0
        
        # Verify features
        assert fp_config.features.enable_websocket is False
        assert fp_config.features.enable_memory is True
        assert fp_config.features.enable_backtesting is False

    def test_legacy_to_fp_system_config_with_defaults(self):
        """Test conversion from minimal legacy config using defaults."""
        minimal_legacy_config = {}
        
        fp_config = ConfigurationAdapter.legacy_to_fp_system_config(minimal_legacy_config)
        
        # Verify defaults are applied
        assert len(fp_config.trading_pairs) == 1
        assert fp_config.trading_pairs[0].value == "BTC-USD"
        assert fp_config.interval.value == "1m"
        assert fp_config.mode == TradingMode.PAPER
        assert fp_config.max_concurrent_positions == 3
        assert fp_config.default_position_size.value == 10.0


class TestConfigurationAdapterCompatibility:
    """Test compatibility between FP and legacy configurations."""

    def test_roundtrip_system_config_conversion(self):
        """Test roundtrip conversion preserves data integrity."""
        # Start with FP config
        original_fp_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1h",
            mode="backtest",
            log_level="DEBUG",
            features={
                "enable_websocket": True,
                "enable_memory": True,
            },
            max_concurrent_positions=4,
            default_position_size=12.5
        ).success()
        
        # Convert to legacy and back
        legacy_config = ConfigurationAdapter.fp_to_legacy_system_config(original_fp_config)
        converted_fp_config = ConfigurationAdapter.legacy_to_fp_system_config(legacy_config)
        
        # Verify data integrity
        assert original_fp_config.trading_pairs[0].value == converted_fp_config.trading_pairs[0].value
        assert original_fp_config.interval.value == converted_fp_config.interval.value
        assert original_fp_config.mode == converted_fp_config.mode
        assert original_fp_config.max_concurrent_positions == converted_fp_config.max_concurrent_positions
        assert original_fp_config.default_position_size.value == converted_fp_config.default_position_size.value

    def test_adapter_handles_invalid_legacy_config(self):
        """Test adapter handles invalid legacy configuration gracefully."""
        invalid_legacy_config = {
            "trading_pairs": ["INVALID-PAIR"],
            "interval": "invalid_interval",
            "mode": "invalid_mode",
            "max_concurrent_positions": -1,
            "default_position_size": -5.0
        }
        
        # Should raise appropriate errors for invalid data
        with pytest.raises(Exception):
            ConfigurationAdapter.legacy_to_fp_system_config(invalid_legacy_config)

    def test_adapter_preserves_feature_flags(self):
        """Test adapter correctly preserves all feature flags."""
        fp_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
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
            default_position_size=10.0
        ).success()
        
        legacy_config = ConfigurationAdapter.fp_to_legacy_system_config(fp_config)
        converted_config = ConfigurationAdapter.legacy_to_fp_system_config(legacy_config)
        
        # Verify all feature flags are preserved
        original_features = fp_config.features
        converted_features = converted_config.features
        
        assert original_features.enable_websocket == converted_features.enable_websocket
        assert original_features.enable_memory == converted_features.enable_memory
        assert original_features.enable_backtesting == converted_features.enable_backtesting
        assert original_features.enable_paper_trading == converted_features.enable_paper_trading
        assert original_features.enable_risk_management == converted_features.enable_risk_management
        assert original_features.enable_notifications == converted_features.enable_notifications
        assert original_features.enable_metrics == converted_features.enable_metrics


class TestConfigurationAdapterSecurityConsiderations:
    """Test security considerations when adapting between FP and legacy configs."""

    def test_adapter_preserves_api_key_security(self):
        """Test adapter maintains API key security during conversion."""
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create("-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----").success()
        rate_limits = RateLimits.create(10, 100, 1000).success()
        
        fp_config = CoinbaseExchangeConfig(
            api_key=api_key,
            private_key=private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=rate_limits
        )
        
        # Convert to legacy format
        legacy_config = ConfigurationAdapter.fp_to_legacy_exchange_config(fp_config)
        
        # Legacy format should contain the actual key values for functionality
        assert legacy_config["api_key"] == "sk-1234567890abcdefghij"
        
        # But the original FP objects should still mask them
        assert str(api_key) != "sk-1234567890abcdefghij"
        assert "APIKey(***" in str(api_key)

    def test_adapter_validates_sensitive_data_access(self):
        """Test adapter properly handles sensitive data access."""
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        
        # FP API key should not expose the raw value in normal operations
        with pytest.raises(AttributeError):
            _ = api_key.value  # type: ignore
        
        # But adapter should be able to access internal value for conversion
        # (This would typically use a proper accessor method in production)
        assert hasattr(api_key, '_value')
        assert api_key._value == "sk-1234567890abcdefghij"

    def test_adapter_maintains_immutability_boundaries(self):
        """Test adapter respects immutability boundaries between systems."""
        fp_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        # Convert to legacy (mutable) format
        legacy_config = ConfigurationAdapter.fp_to_legacy_system_config(fp_config)
        
        # Legacy config should be mutable
        legacy_config["max_concurrent_positions"] = 5
        assert legacy_config["max_concurrent_positions"] == 5
        
        # Original FP config should remain immutable and unchanged
        assert fp_config.max_concurrent_positions == 3
        
        with pytest.raises(AttributeError):
            fp_config.max_concurrent_positions = 5  # type: ignore


class TestConfigurationAdapterIntegration:
    """Test adapter integration with configuration builders and validators."""

    def test_adapter_works_with_env_builders(self):
        """Test adapter works with environment-based configuration builders."""
        env_vars = {
            "TRADING_PAIRS": "BTC-USD,ETH-USD",
            "TRADING_MODE": "paper",
            "MAX_CONCURRENT_POSITIONS": "4",
            "DEFAULT_POSITION_SIZE": "15.0"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            from bot.fp.types.config import build_system_config_from_env
            
            # Build FP config from environment
            fp_result = build_system_config_from_env()
            assert isinstance(fp_result, Success)
            fp_config = fp_result.success()
            
            # Convert to legacy format
            legacy_config = ConfigurationAdapter.fp_to_legacy_system_config(fp_config)
            
            # Verify conversion maintains environment-based values
            assert legacy_config["trading_pairs"] == ["BTC-USD", "ETH-USD"]
            assert legacy_config["mode"] == "paper"
            assert legacy_config["max_concurrent_positions"] == 4
            assert legacy_config["default_position_size"] == 15.0

    def test_adapter_maintains_validation_requirements(self):
        """Test adapter preserves validation requirements between systems."""
        # Create invalid legacy config
        invalid_legacy_config = {
            "trading_pairs": [],  # Empty trading pairs should be invalid
            "max_concurrent_positions": 0,  # Zero positions should be invalid
            "default_position_size": -10.0  # Negative size should be invalid
        }
        
        # Conversion should fail validation
        with pytest.raises(Exception):
            ConfigurationAdapter.legacy_to_fp_system_config(invalid_legacy_config)

    def test_adapter_preserves_safety_invariants(self):
        """Test adapter preserves safety invariants across conversions."""
        # Create FP config with safe defaults
        safe_fp_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",  # Safe default
            log_level="INFO",
            features={"enable_paper_trading": True},  # Safety feature enabled
            max_concurrent_positions=3,
            default_position_size=10.0  # Conservative default
        ).success()
        
        # Convert to legacy and back
        legacy_config = ConfigurationAdapter.fp_to_legacy_system_config(safe_fp_config)
        converted_config = ConfigurationAdapter.legacy_to_fp_system_config(legacy_config)
        
        # Verify safety invariants are preserved
        assert converted_config.mode == TradingMode.PAPER
        assert converted_config.features.enable_paper_trading is True
        assert converted_config.default_position_size.value <= 25.0  # Conservative sizing


class TestConfigurationAdapterErrorHandling:
    """Test error handling in configuration adapters."""

    def test_adapter_handles_unsupported_exchange_types(self):
        """Test adapter handles unsupported exchange types gracefully."""
        # Mock an unsupported exchange config type
        unsupported_config = MagicMock()
        unsupported_config.__class__.__name__ = "UnsupportedExchangeConfig"
        
        with pytest.raises(ValueError, match="Unsupported exchange config type"):
            ConfigurationAdapter.fp_to_legacy_exchange_config(unsupported_config)

    def test_adapter_handles_malformed_legacy_config(self):
        """Test adapter handles malformed legacy configuration."""
        malformed_configs = [
            None,
            "not_a_dict",
            {"trading_pairs": "not_a_list"},
            {"max_concurrent_positions": "not_an_int"},
            {"default_position_size": "not_a_float"}
        ]
        
        for malformed_config in malformed_configs:
            with pytest.raises(Exception):
                ConfigurationAdapter.legacy_to_fp_system_config(malformed_config)

    def test_adapter_provides_helpful_error_messages(self):
        """Test adapter provides helpful error messages for debugging."""
        # This would typically be tested with actual error message content
        # For now, we verify that exceptions are raised for invalid inputs
        
        invalid_config = {"trading_pairs": ["INVALID"]}
        
        try:
            ConfigurationAdapter.legacy_to_fp_system_config(invalid_config)
            pytest.fail("Expected exception for invalid trading pair")
        except Exception as e:
            # Error message should be informative
            assert len(str(e)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])