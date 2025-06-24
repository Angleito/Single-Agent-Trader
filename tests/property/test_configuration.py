"""Property-based tests for configuration safety invariants.

CRITICAL: These tests ensure the trading bot cannot accidentally:
1. Trade real money without explicit confirmation
2. Use dangerous leverage settings
3. Start without valid API keys
4. Accept invalid configuration combinations
"""

import json
import os
import tempfile
from typing import Any
from unittest.mock import patch

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import ValidationError

# Legacy configuration imports (maintained for compatibility)
from bot.config import (
    Environment,
    RiskSettings,
    Settings,
    SystemSettings,
    TradingSettings,
)

# Functional configuration imports (added for migration to functional programming patterns)
try:
    from bot.fp.types.config import (
        Config,
        SystemConfig,
        ExchangeConfig,
        StrategyConfig,
        validate_config,
        build_system_config_from_env,
    )
    FUNCTIONAL_CONFIG_AVAILABLE = True
except ImportError:
    # Functional implementations not available, continue with legacy
    FUNCTIONAL_CONFIG_AVAILABLE = False


class TestPaperTradingDefaults:
    """Test that paper trading is always the default to prevent accidental real trading."""

    def test_paper_trading_default_no_env(self):
        """System must default to paper trading when no environment is set."""
        # Clear any existing DRY_RUN environment variable
        with patch.dict(os.environ, {}, clear=True):
            system_settings = SystemSettings()
            assert (
                system_settings.dry_run is True
            ), "System must default to paper trading"

    def test_paper_trading_default_empty_env(self):
        """System must default to paper trading when DRY_RUN is empty."""
        with patch.dict(os.environ, {"SYSTEM__DRY_RUN": ""}):
            system_settings = SystemSettings()
            assert (
                system_settings.dry_run is True
            ), "Empty DRY_RUN must default to paper trading"

    @given(
        env_value=st.text(
            alphabet=st.characters(blacklist_characters="\x00"), min_size=1
        ).filter(lambda x: x.lower() not in ["false", "0", "no"])
    )
    def test_paper_trading_safe_values(self, env_value: str):
        """Only explicitly false values should disable paper trading."""
        with patch.dict(os.environ, {"SYSTEM__DRY_RUN": env_value}):
            try:
                settings = Settings()
                # Most values should result in dry_run=True
                assert (
                    settings.system.dry_run is True
                ), f"Value '{env_value}' should enable paper trading"
            except ValidationError:
                # Some random text might cause validation errors, which is safe
                pass

    def test_paper_trading_dangerous_values(self):
        """Test that only specific values disable paper trading."""
        dangerous_values = ["false", "False", "FALSE", "0", "no", "No", "NO"]

        for value in dangerous_values:
            with patch.dict(
                os.environ,
                {
                    "SYSTEM__DRY_RUN": value,
                    # Add mock API credentials since dry_run=false requires them
                    "EXCHANGE__CDP_API_KEY_NAME": "test-key",
                    "EXCHANGE__CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
                    "LLM__OPENAI_API_KEY": "sk-test1234567890abcdefghijklmnopqrstuvwxyz",
                },
            ):
                settings = Settings()
                assert (
                    settings.system.dry_run is False
                ), f"Value '{value}' should disable paper trading"

    def test_development_enforces_dry_run(self):
        """Development environment must enforce dry-run mode."""
        with patch.dict(
            os.environ,
            {
                "SYSTEM__ENVIRONMENT": "development",
                "SYSTEM__DRY_RUN": "false",
                # Add mock API credentials to bypass API validation
                "EXCHANGE__CDP_API_KEY_NAME": "test-key",
                "EXCHANGE__CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
                "LLM__OPENAI_API_KEY": "sk-test1234567890abcdefghijklmnopqrstuvwxyz",
            },
        ):
            with pytest.raises(
                ValidationError, match="Development environment should use dry-run mode"
            ):
                Settings()


class TestLeverageSafety:
    """Test leverage limits based on trading mode and environment."""

    @given(leverage=st.integers(min_value=1, max_value=100))
    def test_leverage_bounds(self, leverage: int):
        """Leverage must respect defined bounds."""
        trading_settings = TradingSettings(leverage=leverage)
        assert 1 <= trading_settings.leverage <= 100

    @given(leverage=st.integers(min_value=101))
    def test_excessive_leverage_rejected(self, leverage: int):
        """Excessive leverage must be rejected."""
        with pytest.raises(ValidationError):
            TradingSettings(leverage=leverage)

    @given(leverage=st.integers(max_value=0))
    def test_zero_negative_leverage_rejected(self, leverage: int):
        """Zero or negative leverage must be rejected."""
        with pytest.raises(ValidationError):
            TradingSettings(leverage=leverage)

    def test_production_leverage_limit(self):
        """Production environment must enforce leverage limits."""
        with patch.dict(
            os.environ,
            {
                "SYSTEM__ENVIRONMENT": "production",
                "SYSTEM__DRY_RUN": "false",
                "TRADING__LEVERAGE": "15",
            },
        ):
            with pytest.raises(
                ValidationError, match="Production leverage should not exceed 10x"
            ):
                Settings()

    @given(leverage=st.integers(min_value=1, max_value=10))
    def test_production_safe_leverage(self, leverage: int):
        """Production should accept leverage up to 10x."""
        with patch.dict(
            os.environ,
            {
                "SYSTEM__ENVIRONMENT": "production",
                "SYSTEM__DRY_RUN": "false",
                "TRADING__LEVERAGE": str(leverage),
                "LLM__OPENAI_API_KEY": "sk-test1234567890abcdefghijklmnopqrstuvwxyz",
                "EXCHANGE__CDP_API_KEY_NAME": "test",
                "EXCHANGE__CDP_PRIVATE_KEY": "test-key",
            },
        ):
            settings = Settings()
            assert settings.trading.leverage == leverage

    @given(
        leverage=st.integers(min_value=1, max_value=100),
        max_futures_leverage=st.integers(min_value=1, max_value=100),
    )
    def test_futures_leverage_consistency(
        self, leverage: int, max_futures_leverage: int
    ):
        """Futures leverage settings must be consistent."""
        trading_settings = TradingSettings(
            leverage=leverage, max_futures_leverage=max_futures_leverage
        )
        assert trading_settings.leverage <= 100
        assert trading_settings.max_futures_leverage <= 100


class TestAPIKeyValidation:
    """Test API key validation properties."""

    def test_dry_run_no_api_keys_required(self):
        """Dry-run mode should not require API keys."""
        with patch.dict(
            os.environ,
            {"SYSTEM__DRY_RUN": "true", "SYSTEM__ENVIRONMENT": "development"},
            clear=True,
        ):
            settings = Settings()
            assert settings.system.dry_run is True
            assert not settings.requires_api_keys()

    def test_live_trading_requires_llm_keys(self):
        """Live trading must require LLM API keys."""
        with patch.dict(
            os.environ,
            {
                "SYSTEM__DRY_RUN": "false",
                "SYSTEM__ENVIRONMENT": "staging",
                "LLM__PROVIDER": "openai",
            },
        ):
            with pytest.raises(ValidationError, match="OpenAI API key required"):
                Settings()

    @given(provider=st.sampled_from(["openai", "anthropic"]), has_key=st.booleans())
    def test_llm_provider_key_validation(self, provider: str, has_key: bool):
        """LLM provider must have corresponding API key for live trading."""
        env_vars = {
            "SYSTEM__DRY_RUN": "false",
            "SYSTEM__ENVIRONMENT": "staging",
            "LLM__PROVIDER": provider,
            "EXCHANGE__CDP_API_KEY_NAME": "test",
            "EXCHANGE__CDP_PRIVATE_KEY": "test-key",
        }

        if has_key:
            if provider == "openai":
                env_vars["LLM__OPENAI_API_KEY"] = "test-key"
            else:
                env_vars["LLM__ANTHROPIC_API_KEY"] = "test-key"

        with patch.dict(os.environ, env_vars):
            if has_key:
                settings = Settings()
                assert settings.llm.provider == provider
            else:
                with pytest.raises(
                    ValidationError, match=f"{provider.capitalize()} API key required"
                ):
                    Settings()

    @given(exchange_type=st.sampled_from(["coinbase", "bluefin"]))
    def test_exchange_key_validation(self, exchange_type: str):
        """Live trading must require exchange API keys."""
        base_env = {
            "SYSTEM__DRY_RUN": "false",
            "SYSTEM__ENVIRONMENT": "staging",
            "LLM__OPENAI_API_KEY": "test-key",
            "EXCHANGE__EXCHANGE_TYPE": exchange_type,
        }

        with patch.dict(os.environ, base_env):
            if exchange_type == "coinbase":
                with pytest.raises(
                    ValidationError, match="Coinbase API credentials required"
                ):
                    Settings()
            else:
                with pytest.raises(
                    ValidationError, match="Bluefin private key required"
                ):
                    Settings()


class TestConfigurationConsistency:
    """Test cross-field configuration consistency."""

    @given(
        stop_loss=st.floats(min_value=0.1, max_value=10.0),
        take_profit=st.floats(min_value=0.1, max_value=20.0),
    )
    def test_risk_parameters_consistency(self, stop_loss: float, take_profit: float):
        """Take profit must be greater than stop loss."""
        if take_profit <= stop_loss:
            with pytest.raises(
                ValidationError, match="Take profit must be greater than stop loss"
            ):
                RiskSettings(
                    default_stop_loss_pct=stop_loss, default_take_profit_pct=take_profit
                )
        else:
            risk_settings = RiskSettings(
                default_stop_loss_pct=stop_loss, default_take_profit_pct=take_profit
            )
            assert (
                risk_settings.default_take_profit_pct
                > risk_settings.default_stop_loss_pct
            )

    @given(
        daily_loss=st.floats(min_value=1.0, max_value=50.0),
        weekly_loss=st.floats(min_value=1.0, max_value=50.0),
        monthly_loss=st.floats(min_value=1.0, max_value=50.0),
    )
    def test_loss_limit_hierarchy(
        self, daily_loss: float, weekly_loss: float, monthly_loss: float
    ):
        """Loss limits must follow hierarchy: daily < weekly < monthly."""
        if weekly_loss <= daily_loss or monthly_loss <= weekly_loss:
            with pytest.raises(ValidationError):
                RiskSettings(
                    max_daily_loss_pct=daily_loss,
                    max_weekly_loss_pct=weekly_loss,
                    max_monthly_loss_pct=monthly_loss,
                )
        else:
            risk_settings = RiskSettings(
                max_daily_loss_pct=daily_loss,
                max_weekly_loss_pct=weekly_loss,
                max_monthly_loss_pct=monthly_loss,
            )
            assert (
                risk_settings.max_daily_loss_pct
                < risk_settings.max_weekly_loss_pct
                < risk_settings.max_monthly_loss_pct
            )

    @given(
        interval=st.sampled_from(["1s", "5s", "15s", "30s", "1m", "5m", "1h"]),
        use_aggregation=st.booleans(),
    )
    def test_interval_aggregation_requirement(
        self, interval: str, use_aggregation: bool
    ):
        """Sub-minute intervals must require trade aggregation."""
        sub_minute_intervals = ["1s", "5s", "15s", "30s"]

        env_vars = {
            "TRADING__INTERVAL": interval,
            "EXCHANGE__USE_TRADE_AGGREGATION": str(use_aggregation).lower(),
        }

        with patch.dict(os.environ, env_vars):
            if interval in sub_minute_intervals and not use_aggregation:
                with pytest.raises(
                    ValidationError, match="requires trade aggregation to be enabled"
                ):
                    Settings()
            else:
                settings = Settings()
                assert settings.trading.interval == interval


class TestConfigurationFileParsing:
    """Test configuration file parsing safety."""

    @given(
        config_data=st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.text(),
            ),
        )
    )
    def test_config_file_parsing(self, config_data: dict[str, Any]):
        """Configuration files should be parsed safely."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Test that arbitrary config data doesn't crash the system
            with patch.dict(os.environ, {"CONFIG_FILE": temp_path}):
                try:
                    settings = Settings()
                    # If it parses successfully, verify defaults are safe
                    assert settings.system.dry_run is True
                except (ValidationError, json.JSONDecodeError):
                    # Invalid config should be rejected safely
                    pass
        finally:
            os.unlink(temp_path)

    def test_malformed_json_handled(self):
        """Malformed JSON should be handled gracefully."""
        malformed_jsons = [
            '{"key": "value"',  # Missing closing brace
            '{"key": "value",}',  # Trailing comma
            '{key: "value"}',  # Unquoted key
            '{"key": undefined}',  # JavaScript undefined
            '{"key": NaN}',  # JavaScript NaN
        ]

        for malformed in malformed_jsons:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(malformed)
                temp_path = f.name

            try:
                with patch.dict(os.environ, {"CONFIG_FILE": temp_path}):
                    try:
                        Settings()
                    except (json.JSONDecodeError, ValidationError):
                        # Expected - malformed JSON should raise an error
                        pass
            finally:
                os.unlink(temp_path)


class TestEnvironmentVariableInjection:
    """Test protection against environment variable injection attacks."""

    @given(key=st.text(min_size=1).filter(lambda x: "__" in x), value=st.text())
    def test_nested_env_delimiter_safety(self, key: str, value: str):
        """Test that nested delimiter is handled safely."""
        # The env_nested_delimiter="__" should safely parse nested configs
        with patch.dict(os.environ, {key: value}):
            try:
                settings = Settings()
                # Verify critical defaults remain safe
                assert settings.system.dry_run is True
            except (ValidationError, AttributeError):
                # Invalid keys should be safely rejected
                pass

    @given(
        env_vars=st.dictionaries(
            st.text(min_size=1, max_size=100),
            st.text(min_size=0, max_size=1000),
            max_size=10,
        )
    )
    def test_arbitrary_env_vars_safe(self, env_vars: dict[str, str]):
        """Arbitrary environment variables should not compromise safety."""
        # Always include safe defaults
        safe_env = {"SYSTEM__DRY_RUN": "true"}
        safe_env.update(env_vars)

        with patch.dict(os.environ, safe_env, clear=True):
            try:
                settings = Settings()
                # Even with arbitrary env vars, critical safety must be maintained
                if "SYSTEM__DRY_RUN" not in env_vars or env_vars.get(
                    "SYSTEM__DRY_RUN", ""
                ).lower() not in ["false", "0", "no"]:
                    assert settings.system.dry_run is True
            except ValidationError:
                # Invalid configurations should fail safely
                pass


class TestDefaultValueSafety:
    """Test that all default values are conservative and safe."""

    def test_all_defaults_are_safe(self):
        """All configuration defaults must be safe for new users."""
        settings = Settings()

        # System defaults
        assert settings.system.dry_run is True, "Must default to paper trading"
        assert (
            settings.system.environment == Environment.DEVELOPMENT
        ), "Must default to development"

        # Trading defaults
        assert settings.trading.leverage <= 10, "Default leverage must be conservative"
        assert (
            settings.trading.max_size_pct <= 25.0
        ), "Default position size must be conservative"

        # Risk defaults
        assert (
            settings.risk.max_position_risk_pct <= 2.0
        ), "Default position risk must be conservative"
        assert (
            settings.risk.max_daily_loss_pct <= 10.0
        ), "Default daily loss must be reasonable"
        assert (
            settings.risk.default_stop_loss_pct >= 1.0
        ), "Default stop loss must be reasonable"

        # Paper trading defaults
        assert (
            settings.paper_trading.starting_balance >= 10000
        ), "Paper trading should have reasonable starting balance"

    @given(
        dry_run=st.just(None),  # Simulate missing value
        environment=st.just(None),
    )
    def test_missing_critical_values_default_safe(self, dry_run, environment):
        """Missing critical values must default to safe options."""
        env_vars = {}
        if dry_run is not None:
            env_vars["SYSTEM__DRY_RUN"] = str(dry_run)
        if environment is not None:
            env_vars["SYSTEM__ENVIRONMENT"] = str(environment)

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.system.dry_run is True
            assert settings.system.environment == Environment.DEVELOPMENT


# Run specific safety-critical tests with more examples
class TestCriticalSafetyInvariants:
    """High-priority safety tests that should never fail."""

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
    def test_no_accidental_live_trading(self, env_vars: dict[str, str]):
        """CRITICAL: System must never accidentally enable live trading."""
        # Only these specific values should disable dry_run
        dangerous_dry_run_values = {"false", "False", "FALSE", "0", "no", "No", "NO"}

        with patch.dict(os.environ, env_vars, clear=True):
            try:
                settings = Settings()
                dry_run_value = env_vars.get("SYSTEM__DRY_RUN", "true")

                if dry_run_value in dangerous_dry_run_values:
                    # These values explicitly disable dry_run
                    if settings.system.environment == Environment.DEVELOPMENT:
                        # Development should have failed validation
                        pytest.fail("Development environment allowed dry_run=false")
                else:
                    # All other values should result in dry_run=true
                    assert (
                        settings.system.dry_run is True
                    ), f"Value '{dry_run_value}' should not disable paper trading"
            except ValidationError as e:
                # Validation errors are acceptable
                error_message = str(e)
                if any(
                    msg in error_message
                    for msg in [
                        "Development environment should use dry-run mode",
                        "Coinbase API credentials required",
                        "Bluefin private key required",
                    ]
                ):
                    pass  # Expected errors
                else:
                    # Re-raise unexpected validation errors
                    raise

    def test_paper_trading_truly_isolated(self):
        """CRITICAL: Paper trading must not affect real money."""
        settings = Settings()

        # Verify paper trading is isolated
        assert settings.system.dry_run is True
        assert settings.paper_trading.starting_balance > 0
        # Paper trading is isolated by design - it doesn't interact with real exchange

        # Verify no API keys are required in paper mode
        assert not settings.requires_api_keys()

    @given(leverage=st.integers())
    def test_leverage_always_bounded(self, leverage: int):
        """CRITICAL: Leverage must always be within safe bounds."""
        if leverage < 1 or leverage > 100:
            with pytest.raises(ValidationError):
                TradingSettings(leverage=leverage)
        else:
            trading = TradingSettings(leverage=leverage)
            assert 1 <= trading.leverage <= 100


if __name__ == "__main__":
    # Run safety-critical tests with verbose output
    pytest.main([__file__, "-v", "-k", "TestCriticalSafetyInvariants"])
