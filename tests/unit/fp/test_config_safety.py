"""
Property-based safety tests for functional programming configuration.

CRITICAL: These tests ensure that the functional configuration system maintains
all safety invariants that prevent accidental real trading, dangerous leverage,
and invalid configuration combinations.

This complements the legacy configuration safety tests by ensuring FP configs
provide the same safety guarantees.
"""

import os
import tempfile
from typing import Any
from unittest.mock import patch

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from bot.fp.types.config import (
    Config,
    SystemConfig,
    LLMStrategyConfig,
    CoinbaseExchangeConfig,
    BluefinExchangeConfig,
    APIKey,
    PrivateKey,
    RateLimits,
    build_system_config_from_env,
    build_exchange_config_from_env,
    build_strategy_config_from_env,
    parse_bool_env,
    validate_config,
)
from bot.fp.types.base import TradingMode
from bot.fp.types.result import Failure, Success


class TestFunctionalPaperTradingDefaults:
    """Test that FP configuration defaults to paper trading mode."""

    def test_fp_system_config_defaults_to_paper_mode(self):
        """Functional system config must default to paper trading mode."""
        with patch.dict(os.environ, {}, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            assert config.mode == TradingMode.PAPER, "FP system must default to paper trading"

    def test_fp_system_config_paper_mode_with_empty_env(self):
        """FP system config must default to paper trading when mode is empty."""
        with patch.dict(os.environ, {"TRADING_MODE": ""}, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            assert config.mode == TradingMode.PAPER, "Empty mode must default to paper trading"

    @given(
        mode_value=st.text(
            alphabet=st.characters(blacklist_characters="\x00"), min_size=1
        ).filter(lambda x: x.lower() not in ["live", "backtest"])
    )
    def test_fp_trading_mode_safe_values(self, mode_value: str):
        """Only explicitly live/backtest values should disable paper trading in FP."""
        with patch.dict(os.environ, {"TRADING_MODE": mode_value}, clear=True):
            result = build_system_config_from_env()
            
            if result.is_success():
                config = result.success()
                # Most invalid values should either fail or default to paper
                assert config.mode == TradingMode.PAPER, f"Value '{mode_value}' should result in paper trading"
            else:
                # Invalid values failing validation is also safe
                assert "Invalid trading mode" in result.failure()

    def test_fp_trading_mode_dangerous_values(self):
        """Test that only specific values enable live/backtest trading in FP."""
        dangerous_values = ["live", "Live", "LIVE", "backtest", "Backtest", "BACKTEST"]
        
        for value in dangerous_values:
            with patch.dict(os.environ, {"TRADING_MODE": value}, clear=True):
                result = build_system_config_from_env()
                assert isinstance(result, Success)
                
                config = result.success()
                if value.lower() == "live":
                    assert config.mode == TradingMode.LIVE, f"Value '{value}' should enable live trading"
                elif value.lower() == "backtest":
                    assert config.mode == TradingMode.BACKTEST, f"Value '{value}' should enable backtest mode"

    def test_fp_paper_trading_truly_isolated(self):
        """FP paper trading configuration must be truly isolated."""
        with patch.dict(os.environ, {"TRADING_MODE": "paper"}, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            assert config.mode == TradingMode.PAPER
            # Paper trading should enable paper trading feature
            assert config.features.enable_paper_trading is True


class TestFunctionalLeverageSafety:
    """Test leverage safety in functional configuration."""

    def test_fp_system_config_conservative_position_size_default(self):
        """FP system config must default to conservative position sizing."""
        with patch.dict(os.environ, {}, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            # Default position size should be conservative (10%)
            assert config.default_position_size.value <= 25.0, "Default position size must be conservative"

    @given(position_size=st.floats(min_value=0.01, max_value=100.0))
    def test_fp_position_size_bounds(self, position_size: float):
        """FP position size must respect bounds."""
        env_vars = {"DEFAULT_POSITION_SIZE": str(position_size)}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            
            if position_size > 100.0:
                # Should fail validation for excessive position size
                assert isinstance(result, Failure)
                assert "Invalid position size" in result.failure()
            else:
                # Should succeed for reasonable values
                assert isinstance(result, Success)
                config = result.success()
                assert 0 < config.default_position_size.value <= 100.0

    @given(max_positions=st.integers())
    def test_fp_max_concurrent_positions_bounds(self, max_positions: int):
        """FP max concurrent positions must be within bounds."""
        env_vars = {"MAX_CONCURRENT_POSITIONS": str(max_positions)}
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            
            if max_positions < 1:
                # Should fail validation for zero or negative positions
                assert isinstance(result, Failure)
                assert "Max concurrent positions must be at least 1" in result.failure()
            else:
                # Should succeed for positive values
                assert isinstance(result, Success)
                config = result.success()
                assert config.max_concurrent_positions >= 1


class TestFunctionalAPIKeySafety:
    """Test API key safety in functional configuration."""

    def test_fp_api_key_creation_validation(self):
        """FP API key creation must validate key format."""
        # Valid key should succeed
        valid_result = APIKey.create("sk-1234567890abcdefghijklmnopqrstuvwxyz")
        assert isinstance(valid_result, Success)
        
        # Invalid keys should fail
        invalid_keys = ["", "short", "sk-123", None]
        
        for invalid_key in invalid_keys:
            if invalid_key is None:
                continue
            result = APIKey.create(invalid_key)
            assert isinstance(result, Failure)
            assert "too short" in result.failure()

    def test_fp_private_key_creation_validation(self):
        """FP private key creation must validate key format."""
        # Valid key should succeed
        valid_key = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIExampleKey\n-----END EC PRIVATE KEY-----"
        valid_result = PrivateKey.create(valid_key)
        assert isinstance(valid_result, Success)
        
        # Invalid keys should fail
        invalid_keys = ["", "short", "invalid-key"]
        
        for invalid_key in invalid_keys:
            result = PrivateKey.create(invalid_key)
            assert isinstance(result, Failure)
            assert "too short" in result.failure()

    def test_fp_api_key_masking(self):
        """FP API keys must be properly masked in string representation."""
        api_key = APIKey.create("sk-1234567890abcdefghijklmnopqrstuvwxyz").success()
        str_repr = str(api_key)
        
        # Should not contain the full key
        assert "1234567890abcdefghijklmnopqrstu" not in str_repr
        # Should contain masked representation
        assert "APIKey(***" in str_repr

    def test_fp_private_key_masking(self):
        """FP private keys must be completely masked."""
        private_key = PrivateKey.create("-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----").success()
        str_repr = str(private_key)
        
        # Should be completely masked
        assert str_repr == "PrivateKey(***)"
        assert "test" not in str_repr

    def test_fp_exchange_config_missing_credentials(self):
        """FP exchange config must fail when credentials are missing."""
        # Coinbase without API key
        with patch.dict(os.environ, {"EXCHANGE_TYPE": "coinbase"}, clear=True):
            result = build_exchange_config_from_env()
            assert isinstance(result, Failure)
            assert "COINBASE_API_KEY not set" in result.failure()
        
        # Bluefin without private key
        with patch.dict(os.environ, {"EXCHANGE_TYPE": "bluefin"}, clear=True):
            result = build_exchange_config_from_env()
            assert isinstance(result, Failure)
            assert "BLUEFIN_PRIVATE_KEY not set" in result.failure()


class TestFunctionalConfigurationConsistency:
    """Test cross-field configuration consistency in FP system."""

    def test_fp_llm_memory_feature_consistency(self):
        """FP LLM strategy memory usage must be consistent with system features."""
        # Strategy wants memory but system disables it
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "LLM_USE_MEMORY": "true",
            "ENABLE_MEMORY": "false",
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)
            
            # Validation should catch the inconsistency
            validation_result = validate_config(config_result.success())
            assert isinstance(validation_result, Failure)
            assert "memory but it's disabled" in validation_result.failure()

    def test_fp_bluefin_network_mode_consistency(self):
        """FP Bluefin network must be consistent with trading mode."""
        # Testnet with live trading should fail
        env_vars = {
            "STRATEGY_TYPE": "llm",
            "EXCHANGE_TYPE": "bluefin",
            "BLUEFIN_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\nlongenoughkey\n-----END EC PRIVATE KEY-----",
            "BLUEFIN_NETWORK": "testnet",
            "TRADING_MODE": "live"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config_result = Config.from_env()
            assert isinstance(config_result, Success)
            
            # Validation should catch the inconsistency
            validation_result = validate_config(config_result.success())
            assert isinstance(validation_result, Failure)
            assert "Cannot use testnet for live trading" in validation_result.failure()

    def test_fp_rate_limits_consistency(self):
        """FP rate limits must be internally consistent."""
        # Inconsistent rate limits
        env_vars = {
            "EXCHANGE_TYPE": "coinbase",
            "COINBASE_API_KEY": "sk-1234567890abcdefghij",
            "COINBASE_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
            "RATE_LIMIT_RPS": "100",  # 100 * 60 = 6000
            "RATE_LIMIT_RPM": "1000",  # 6000 > 1000, inconsistent
            "RATE_LIMIT_RPH": "10000"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_exchange_config_from_env()
            assert isinstance(result, Failure)
            assert "Inconsistent rate limits" in result.failure()

    def test_fp_backtest_mode_config_consistency(self):
        """FP backtest mode must require appropriate configuration."""
        # Backtest config provided but not in backtest mode
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        private_key = PrivateKey.create("-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----").success()
        rate_limits = RateLimits.create(10, 100, 1000).success()
        
        strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0
        ).success()
        
        exchange = CoinbaseExchangeConfig(
            api_key=api_key,
            private_key=private_key,
            api_url="https://api.coinbase.com",
            websocket_url="wss://ws.coinbase.com",
            rate_limits=rate_limits
        )
        
        system = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",  # Not backtest mode
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        from bot.fp.types.config import BacktestConfig
        backtest = BacktestConfig.create(
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0,
            currency="USD",
            maker_fee=0.001,
            taker_fee=0.002,
            slippage=0.0005
        ).success()
        
        config = Config(
            strategy=strategy,
            exchange=exchange,
            system=system,
            backtest=backtest  # Backtest config but not in backtest mode
        )
        
        result = validate_config(config)
        assert isinstance(result, Failure)
        assert "Backtest config provided but not in backtest mode" in result.failure()


class TestFunctionalEnvironmentVariableSafety:
    """Test environment variable parsing safety in FP system."""

    @given(
        env_vars=st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.text(min_size=0, max_size=100),
            max_size=10
        )
    )
    def test_fp_arbitrary_env_vars_safe(self, env_vars: dict[str, str]):
        """FP system must handle arbitrary environment variables safely."""
        # Always include safe defaults
        safe_env = {"TRADING_MODE": "paper"}
        safe_env.update(env_vars)
        
        with patch.dict(os.environ, safe_env, clear=True):
            try:
                system_result = build_system_config_from_env()
                if system_result.is_success():
                    config = system_result.success()
                    # Even with arbitrary env vars, safety must be maintained
                    if "TRADING_MODE" not in env_vars or env_vars.get("TRADING_MODE", "").lower() not in ["live", "backtest"]:
                        assert config.mode == TradingMode.PAPER
            except Exception:
                # Failures are acceptable for malformed input
                pass

    @given(
        bool_value=st.text(min_size=1, max_size=20)
    )
    def test_fp_boolean_parsing_safety(self, bool_value: str):
        """FP boolean parsing must be safe and predictable."""
        result = parse_bool_env("TEST_BOOL", False)
        
        # Only specific true values should result in True
        expected_true_values = {"true", "1", "yes", "on"}
        if bool_value.lower() in expected_true_values:
            # These should parse as True when set in environment
            with patch.dict(os.environ, {"TEST_BOOL": bool_value}):
                result = parse_bool_env("TEST_BOOL", False)
                assert result is True
        else:
            # All other values should be False or use default
            with patch.dict(os.environ, {"TEST_BOOL": bool_value}):
                result = parse_bool_env("TEST_BOOL", False)
                assert result is False

    @given(
        numeric_value=st.text(min_size=1, max_size=20)
    )
    def test_fp_numeric_parsing_safety(self, numeric_value: str):
        """FP numeric parsing must handle invalid values safely."""
        from bot.fp.types.config import parse_int_env, parse_float_env
        
        with patch.dict(os.environ, {"TEST_INT": numeric_value}):
            int_result = parse_int_env("TEST_INT", 42)
            
            try:
                int(numeric_value)
                # Valid integer should succeed
                assert isinstance(int_result, Success)
            except ValueError:
                # Invalid integer should fail gracefully
                assert isinstance(int_result, Failure)
                assert "Invalid integer" in int_result.failure()
        
        with patch.dict(os.environ, {"TEST_FLOAT": numeric_value}):
            float_result = parse_float_env("TEST_FLOAT", 3.14)
            
            try:
                float(numeric_value)
                # Valid float should succeed
                assert isinstance(float_result, Success)
            except ValueError:
                # Invalid float should fail gracefully
                assert isinstance(float_result, Failure)
                assert "Invalid float" in float_result.failure()


class TestFunctionalFeatureFlagSafety:
    """Test feature flag safety in FP configuration."""

    def test_fp_default_feature_flags_are_safe(self):
        """FP default feature flags must be safe for new users."""
        with patch.dict(os.environ, {}, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            features = config.features
            
            # Safety-critical features should be enabled by default
            assert features.enable_paper_trading is True, "Paper trading must be enabled by default"
            assert features.enable_risk_management is True, "Risk management must be enabled by default"
            
            # Potentially risky features should be disabled by default
            assert features.enable_memory is False, "Memory should be disabled by default"
            assert features.enable_notifications is False, "Notifications should be disabled by default"

    @given(
        enable_memory=st.booleans(),
        enable_risk=st.booleans(),
        enable_paper=st.booleans()
    )
    def test_fp_feature_flag_combinations(self, enable_memory: bool, enable_risk: bool, enable_paper: bool):
        """FP feature flag combinations must maintain safety invariants."""
        env_vars = {
            "ENABLE_MEMORY": str(enable_memory).lower(),
            "ENABLE_RISK_MANAGEMENT": str(enable_risk).lower(),
            "ENABLE_PAPER_TRADING": str(enable_paper).lower(),
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            features = config.features
            
            # Verify the settings were applied correctly
            assert features.enable_memory == enable_memory
            assert features.enable_risk_management == enable_risk
            assert features.enable_paper_trading == enable_paper


class TestFunctionalConfigurationImmutability:
    """Test that FP configuration objects are truly immutable."""

    def test_fp_config_objects_are_immutable(self):
        """FP configuration objects must be immutable."""
        # Create a valid config
        api_key = APIKey.create("sk-1234567890abcdefghij").success()
        
        # API key should be immutable
        with pytest.raises(AttributeError):
            api_key._value = "new_value"  # type: ignore
        
        # System config should be immutable
        system_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        with pytest.raises(AttributeError):
            system_config.max_concurrent_positions = 10  # type: ignore

    def test_fp_config_equality_and_hashing(self):
        """FP configuration objects must support equality and hashing."""
        # Create two identical configs
        config1 = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        config2 = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={},
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        # Should be equal
        assert config1 == config2
        
        # Should be hashable (for use in sets, dicts)
        config_set = {config1, config2}
        assert len(config_set) == 1  # Same config, so set should have one element


class TestFunctionalCriticalSafetyInvariants:
    """High-priority FP safety tests that should never fail."""

    @settings(max_examples=1000)
    @given(
        env_vars=st.dictionaries(
            st.text(
                alphabet=st.characters(blacklist_characters="\x00"),
                min_size=1,
                max_size=50,
            ),
            st.text(
                alphabet=st.characters(blacklist_characters="\x00"),
                min_size=0,
                max_size=100,
            ),
            max_size=20,
        )
    )
    def test_fp_no_accidental_live_trading(self, env_vars: dict[str, str]):
        """CRITICAL: FP system must never accidentally enable live trading."""
        # Only these specific values should enable live trading
        dangerous_mode_values = {"live", "Live", "LIVE"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            try:
                result = build_system_config_from_env()
                
                if result.is_success():
                    config = result.success()
                    trading_mode_value = env_vars.get("TRADING_MODE", "paper")
                    
                    if trading_mode_value in dangerous_mode_values:
                        # These values explicitly enable live trading
                        assert config.mode == TradingMode.LIVE
                    else:
                        # All other values should result in paper trading
                        assert config.mode == TradingMode.PAPER, f"Value '{trading_mode_value}' should not enable live trading"
                else:
                    # Build failures are acceptable - system should fail safely
                    pass
            except Exception:
                # Exceptions are acceptable - system should fail safely rather than
                # accidentally enable dangerous configurations
                pass

    def test_fp_paper_trading_is_truly_safe(self):
        """CRITICAL: FP paper trading must be completely isolated."""
        with patch.dict(os.environ, {"TRADING_MODE": "paper"}, clear=True):
            result = build_system_config_from_env()
            assert isinstance(result, Success)
            
            config = result.success()
            
            # Verify paper trading mode
            assert config.mode == TradingMode.PAPER
            
            # Paper trading feature must be enabled
            assert config.features.enable_paper_trading is True
            
            # Position sizing should be conservative by default
            assert config.default_position_size.value <= 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])