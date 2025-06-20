"""
Configuration and startup validation integration tests.

Tests the complete startup process including configuration loading,
environment validation, component initialization, and health checks.
"""

import json
import os
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from bot.config import Environment, Settings, TradingProfile, create_settings
from bot.main import TradingEngine
from bot.trading_types import MarketData


class TestStartupValidation:
    """Test configuration loading and startup validation."""

    @pytest.fixture()
    def temp_config_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for test configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture()
    def valid_config_data(self) -> dict[str, Any]:
        """Create valid configuration data for testing."""
        return {
            "system": {
                "environment": "development",
                "dry_run": True,
                "log_level": "INFO",
                "log_to_console": True,
                "log_to_file": False,
                "update_frequency_seconds": 10,
            },
            "trading": {
                "symbol": "BTC-USD",
                "interval": "1m",
                "leverage": 5,
                "max_size_pct": 20.0,
                "order_timeout_seconds": 30,
                "slippage_tolerance_pct": 0.1,
                "min_profit_pct": 0.5,
            },
            "risk": {
                "max_daily_loss_pct": 5.0,
                "max_position_size_pct": 25.0,
                "max_open_positions": 3,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0,
                "risk_per_trade_pct": 1.0,
            },
            "llm": {
                "provider": "openai",
                "model_name": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 500,
                "timeout_seconds": 30,
            },
            "exchange": {
                "name": "coinbase",
                "sandbox": True,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "rate_limit_per_second": 10,
            },
        }

    @pytest.fixture()
    def invalid_config_data(self):
        """Create invalid configuration data for testing."""
        return {
            "system": {
                "environment": "invalid_env",  # Invalid environment
                "dry_run": "not_boolean",  # Wrong type
                "log_level": "INVALID_LEVEL",  # Invalid log level
                "update_frequency_seconds": -5,  # Invalid value
            },
            "trading": {
                "symbol": "invalid-symbol",  # Invalid format
                "interval": "invalid",  # Invalid interval
                "leverage": 0,  # Invalid leverage
                "max_size_pct": 150.0,  # Over 100%
            },
            "risk": {
                "max_daily_loss_pct": -1.0,  # Negative value
                "max_position_size_pct": 200.0,  # Over 100%
            },
        }

    def test_configuration_loading_from_file(self, temp_config_dir, valid_config_data):
        """Test loading configuration from file."""
        # Create config file
        config_file = temp_config_dir / "test_config.json"
        with config_file.open("w") as f:
            json.dump(valid_config_data, f)

        # Load configuration
        settings = Settings.load_from_file(str(config_file))

        # Verify configuration loaded correctly
        assert settings.system.environment == Environment.DEVELOPMENT
        assert settings.system.dry_run is True
        assert settings.trading.symbol == "BTC-USD"
        assert settings.trading.leverage == 5
        assert settings.risk.max_daily_loss_pct == 5.0
        assert settings.llm.provider == "openai"
        assert settings.exchange.sandbox is True

    def test_configuration_validation_errors(
        self, temp_config_dir, invalid_config_data
    ):
        """Test configuration validation catches errors."""
        # Create invalid config file
        config_file = temp_config_dir / "invalid_config.json"
        with config_file.open("w") as f:
            json.dump(invalid_config_data, f)

        # Loading should fail with validation errors
        with pytest.raises(ValidationError):  # Pydantic validation error
            Settings.load_from_file(str(config_file))

    def test_environment_variable_override(self, temp_config_dir, valid_config_data):
        """Test environment variables override configuration file."""
        # Create config file
        config_file = temp_config_dir / "test_config.json"
        with config_file.open("w") as f:
            json.dump(valid_config_data, f)

        # Set environment variables
        test_env_vars = {
            "TRADING_SYMBOL": "ETH-USD",
            "TRADING_LEVERAGE": "10",
            "SYSTEM_DRY_RUN": "false",
            "LLM_TEMPERATURE": "0.5",
        }

        with patch.dict(os.environ, test_env_vars):
            settings = Settings.load_from_file(str(config_file))

            # Verify environment variables took precedence
            assert settings.trading.symbol == "ETH-USD"
            assert settings.trading.leverage == 10
            assert settings.system.dry_run is False
            assert settings.llm.temperature == 0.5

    def test_trading_profile_configuration(self):
        """Test different trading profile configurations."""
        profiles_to_test = [
            TradingProfile.CONSERVATIVE,
            TradingProfile.MODERATE,
            TradingProfile.AGGRESSIVE,
        ]

        for profile in profiles_to_test:
            with patch.dict(os.environ, {"TRADING_PROFILE": profile.value}):
                settings = create_settings()

                # Verify profile-specific settings
                if profile == TradingProfile.CONSERVATIVE:
                    assert settings.trading.max_size_pct <= 15.0
                    assert settings.risk.max_daily_loss_pct <= 3.0
                    assert settings.risk.risk_per_trade_pct <= 1.0
                elif profile == TradingProfile.MODERATE:
                    assert 15.0 < settings.trading.max_size_pct <= 25.0
                    assert 3.0 < settings.risk.max_daily_loss_pct <= 5.0
                elif profile == TradingProfile.AGGRESSIVE:
                    assert settings.trading.max_size_pct > 25.0
                    assert settings.risk.max_daily_loss_pct > 5.0

    def test_configuration_profile_switching(self, temp_config_dir):
        """Test switching between configuration profiles."""
        # Create multiple config files
        profiles = {
            "conservative": {
                "trading": {"max_size_pct": 10.0, "leverage": 2},
                "risk": {"max_daily_loss_pct": 2.0, "risk_per_trade_pct": 0.5},
            },
            "aggressive": {
                "trading": {"max_size_pct": 50.0, "leverage": 20},
                "risk": {"max_daily_loss_pct": 10.0, "risk_per_trade_pct": 3.0},
            },
        }

        config_files = {}
        for profile, config in profiles.items():
            config_file = temp_config_dir / f"{profile}_config.json"
            with config_file.open("w") as f:
                json.dump(config, f)
            config_files[profile] = str(config_file)

        # Test conservative profile
        conservative_settings = Settings.load_from_file(config_files["conservative"])
        assert conservative_settings.trading.max_size_pct == 10.0
        assert conservative_settings.trading.leverage == 2

        # Test aggressive profile
        aggressive_settings = Settings.load_from_file(config_files["aggressive"])
        assert aggressive_settings.trading.max_size_pct == 50.0
        assert aggressive_settings.trading.leverage == 20

        # Verify profiles are different
        assert (
            conservative_settings.trading.max_size_pct
            != aggressive_settings.trading.max_size_pct
        )

    @pytest.mark.asyncio()
    async def test_trading_engine_startup_sequence(
        self, temp_config_dir, valid_config_data
    ):
        """Test complete trading engine startup sequence."""
        # Create config file
        config_file = temp_config_dir / "startup_test.json"
        with config_file.open("w") as f:
            json.dump(valid_config_data, f)

        # Mock all external dependencies
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                connect=AsyncMock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 50}
                ),
                get_latest_ohlcv=Mock(
                    return_value=[
                        MarketData(
                            symbol="BTC-USD",
                            timestamp=datetime.now(UTC),
                            open=Decimal(50000),
                            high=Decimal(50100),
                            low=Decimal(49900),
                            close=Decimal(50050),
                            volume=Decimal(100),
                        )
                        for _ in range(50)
                    ]
                ),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                get_connection_status=Mock(
                    return_value={"connected": True, "sandbox": True}
                ),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                is_available=Mock(return_value=True),
                get_status=Mock(
                    return_value={
                        "llm_available": True,
                        "model_provider": "openai",
                        "model_name": "gpt-4",
                    }
                ),
            ),
        ):
            # Create and initialize trading engine
            engine = TradingEngine(
                symbol="BTC-USD",
                interval="1m",
                config_file=str(config_file),
                dry_run=True,
            )

            # Test initialization
            await engine._initialize_components()

            # Verify all components initialized
            assert engine.market_data is not None
            assert engine.indicator_calc is not None
            assert engine.llm_agent is not None
            assert engine.validator is not None
            assert engine.risk_manager is not None
            assert engine.exchange_client is not None

            # Verify configuration loaded
            assert engine.settings.trading.symbol == "BTC-USD"
            assert engine.settings.system.dry_run is True

            # Test shutdown
            await engine._shutdown()

    def test_startup_validation_warnings(self, temp_config_dir):
        """Test startup validation produces appropriate warnings."""
        # Create config with potential issues
        warning_config = {
            "system": {
                "environment": "production",
                "dry_run": False,  # Live trading - should warn
            },
            "trading": {
                "leverage": 50,  # High leverage - should warn
                "max_size_pct": 75.0,  # High position size - should warn
            },
            "risk": {"max_daily_loss_pct": 15.0},  # High daily loss - should warn
            "exchange": {"sandbox": False},  # Live exchange - should warn
        }

        config_file = temp_config_dir / "warning_config.json"
        with config_file.open("w") as f:
            json.dump(warning_config, f)

        settings = Settings.load_from_file(str(config_file))
        warnings = settings.validate_trading_environment()

        # Should have multiple warnings
        assert len(warnings) > 0

        # Check for specific warning types
        warning_text = " ".join(warnings).lower()
        assert any(
            word in warning_text for word in ["live", "production", "leverage", "risk"]
        )

    @pytest.mark.asyncio()
    async def test_health_monitoring_initialization(self):
        """Test health monitoring initialization during startup."""
        with patch("bot.health.HealthMonitor") as mock_health_monitor:
            mock_instance = Mock()
            mock_health_monitor.return_value = mock_instance
            mock_instance.start_monitoring = AsyncMock()
            mock_instance.stop_monitoring = AsyncMock()
            mock_instance.get_system_health = Mock(
                return_value={
                    "overall_status": "healthy",
                    "components": {},
                    "timestamp": datetime.now(UTC),
                }
            )

            with (
                patch.multiple(
                    "bot.main.MarketDataProvider",
                    connect=AsyncMock(return_value=True),
                    get_data_status=Mock(
                        return_value={"connected": True, "cached_candles": 50}
                    ),
                    get_latest_ohlcv=Mock(return_value=[Mock() for _ in range(50)]),
                ),
                patch.multiple(
                    "bot.main.CoinbaseClient",
                    connect=AsyncMock(return_value=True),
                    get_connection_status=Mock(
                        return_value={"connected": True, "sandbox": True}
                    ),
                ),
                patch.multiple(
                    "bot.main.LLMAgent",
                    is_available=Mock(return_value=True),
                    get_status=Mock(
                        return_value={
                            "llm_available": True,
                            "model_provider": "openai",
                            "model_name": "gpt-4",
                        }
                    ),
                ),
            ):
                engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

                # Initialize with health monitoring
                await engine._initialize_components()

                # Health monitor should be created and started
                mock_health_monitor.assert_called_once()

                await engine._shutdown()

    def test_missing_required_environment_variables(self):
        """Test handling of missing required environment variables."""
        # Clear important environment variables
        important_vars = ["COINBASE_API_KEY", "COINBASE_API_SECRET", "OPENAI_API_KEY"]

        original_values = {}
        for var in important_vars:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        try:
            # Create settings without required vars
            settings = create_settings()

            # Should load with defaults but may have warnings
            warnings = settings.validate_trading_environment()

            # Should warn about missing API keys for live trading
            if not settings.system.dry_run:
                assert len(warnings) > 0
                warning_text = " ".join(warnings).lower()
                assert any(word in warning_text for word in ["api", "key", "missing"])

        finally:
            # Restore environment variables
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_configuration_backward_compatibility(self, temp_config_dir):
        """Test configuration backward compatibility."""
        # Create old format config (simulated)
        old_format_config = {
            "symbol": "BTC-USD",
            "leverage": 5,
            "max_position_size": 20.0,  # Old field name
            "dry_run": True,
        }

        config_file = temp_config_dir / "old_format.json"
        with config_file.open("w") as f:
            json.dump(old_format_config, f)

        # Should handle old format gracefully (or provide clear error)
        try:
            settings = Settings.load_from_file(str(config_file))
            # If successful, verify key settings
            assert settings.system.dry_run is True
        except Exception as e:
            # If it fails, should be a clear validation error
            error_msg = str(e).lower()
            assert "validation" in error_msg or "field" in error_msg

    @pytest.mark.asyncio()
    async def test_startup_failure_recovery(self):
        """Test startup failure scenarios and recovery."""
        # Test market data connection failure
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                connect=AsyncMock(
                    side_effect=ConnectionError("Market data connection failed")
                ),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient", connect=AsyncMock(return_value=True)
            ),
            patch.multiple("bot.main.LLMAgent", is_available=Mock(return_value=True)),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Should raise exception during initialization
            with pytest.raises(ConnectionError):
                await engine._initialize_components()

        # Test exchange connection failure
        with (
            patch.multiple(
                "bot.main.MarketDataProvider", connect=AsyncMock(return_value=True)
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=False),  # Connection failed
            ),
        ):
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)

            # Should raise RuntimeError for failed exchange connection
            with pytest.raises(RuntimeError, match="Failed to connect to exchange"):
                await engine._initialize_components()

    def test_dry_run_mode_enforcement(self, temp_config_dir, valid_config_data):
        """Test dry run mode enforcement in configuration."""
        # Test explicit dry run override
        config_file = temp_config_dir / "dry_run_test.json"

        # Config with dry_run=False
        live_config = valid_config_data.copy()
        live_config["system"]["dry_run"] = False

        with config_file.open("w") as f:
            json.dump(live_config, f)

        # Create engine with explicit dry_run=True override
        engine = TradingEngine(
            symbol="BTC-USD",
            interval="1m",
            config_file=str(config_file),
            dry_run=True,  # Override config file
        )

        # Should use the override value
        assert engine.settings.system.dry_run is True
        assert engine.dry_run is True

        # Test that live mode requires explicit confirmation
        engine_live = TradingEngine(
            symbol="BTC-USD",
            interval="1m",
            config_file=str(config_file),
            dry_run=False,  # Live mode
        )

        # Should use live mode when explicitly set
        assert engine_live.settings.system.dry_run is False
        assert engine_live.dry_run is False
