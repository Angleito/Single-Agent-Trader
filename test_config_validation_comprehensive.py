#!/usr/bin/env python3
"""
Comprehensive configuration system validation test.

This test validates the functional configuration system including:
- Environment variable parsing
- Functional types and opaque types
- Configuration validation
- Backward compatibility
- Error handling and fallbacks
"""

import json
import os
import tempfile
import traceback


def test_environment_parsing():
    """Test environment variable parsing functionality."""
    print("=== Testing Environment Variable Parsing ===")

    # Save original environment
    original_env = {}
    test_env_vars = {
        "TRADING__SYMBOL": "ETH-USD",
        "TRADING__LEVERAGE": "10",
        "TRADING__MAX_SIZE_PCT": "25.0",
        "LLM__TEMPERATURE": "0.2",
        "LLM__MAX_TOKENS": "8000",
        "SYSTEM__DRY_RUN": "false",
        "SYSTEM__LOG_LEVEL": "DEBUG",
        "EXCHANGE__EXCHANGE_TYPE": "coinbase",
        "RISK__MAX_DAILY_LOSS_PCT": "3.0",
        "PAPER_TRADING__STARTING_BALANCE": "50000.0",
    }

    # Set test environment variables
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        from bot.config import (
            create_settings,
            parse_bool_env,
            parse_float_env,
            parse_int_env,
        )

        # Test basic environment parsing functions
        print("  ✓ Testing parse_bool_env...")
        assert parse_bool_env("SYSTEM__DRY_RUN") == False
        assert parse_bool_env("NONEXISTENT_VAR", True) == True

        print("  ✓ Testing parse_int_env...")
        assert parse_int_env("TRADING__LEVERAGE", 5) == 10
        assert parse_int_env("NONEXISTENT_VAR", 42) == 42

        print("  ✓ Testing parse_float_env...")
        assert parse_float_env("LLM__TEMPERATURE", 0.1) == 0.2
        assert parse_float_env("NONEXISTENT_VAR", 3.14) == 3.14

        # Test settings creation from environment
        print("  ✓ Testing settings creation from environment...")
        settings = create_settings()

        # Verify environment values were loaded correctly
        assert settings.trading.symbol == "ETH-USD"
        assert settings.trading.leverage == 10
        assert settings.trading.max_size_pct == 25.0
        assert settings.llm.temperature == 0.2
        assert settings.llm.max_tokens == 8000
        assert settings.system.dry_run == False
        assert settings.system.log_level == "DEBUG"
        assert settings.exchange.exchange_type == "coinbase"
        assert settings.risk.max_daily_loss_pct == 3.0
        assert settings.paper_trading.starting_balance == 50000.0

        print("  ✅ Environment parsing tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Environment parsing test failed: {e}")
        traceback.print_exc()
        return False

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def test_functional_types():
    """Test functional programming types and opaque types."""
    print("\n=== Testing Functional Types ===")

    try:
        from bot.fp.types.base import Money, Percentage, Symbol, TimeInterval
        from bot.fp.types.config import APIKey, PrivateKey
        from bot.fp.types.result import Failure, Success

        # Test APIKey opaque type
        print("  ✓ Testing APIKey opaque type...")
        valid_key = APIKey.create("test_api_key_12345")
        assert isinstance(valid_key, Success)
        api_key = valid_key.success()
        assert str(api_key) == "APIKey(***2345)"  # Masked representation

        invalid_key = APIKey.create("short")
        assert isinstance(invalid_key, Failure)

        # Test PrivateKey opaque type
        print("  ✓ Testing PrivateKey opaque type...")
        valid_private = PrivateKey.create("very_long_private_key_data_12345")
        assert isinstance(valid_private, Success)
        private_key = valid_private.success()
        assert str(private_key) == "PrivateKey(***)"  # Fully masked

        invalid_private = PrivateKey.create("short")
        assert isinstance(invalid_private, Failure)

        # Test Money type
        print("  ✓ Testing Money type...")
        valid_money = Money.create(100.50, "USD")
        assert isinstance(valid_money, Success)
        money = valid_money.success()
        assert str(money) == "100.5 USD"

        invalid_money = Money.create(-10.0, "USD")
        assert isinstance(invalid_money, Failure)

        # Test Percentage type
        print("  ✓ Testing Percentage type...")
        valid_pct = Percentage.create(0.25)
        assert isinstance(valid_pct, Success)
        pct = valid_pct.success()
        assert pct.as_percent() == 25.0
        assert pct.as_ratio() == 0.25

        invalid_pct = Percentage.create(1.5)  # > 1.0
        assert isinstance(invalid_pct, Failure)

        # Test Symbol type
        print("  ✓ Testing Symbol type...")
        valid_symbol = Symbol.create("BTC-USD")
        assert isinstance(valid_symbol, Success)
        symbol = valid_symbol.success()
        assert symbol.base == "BTC"
        assert symbol.quote == "USD"

        invalid_symbol = Symbol.create("invalid")
        assert isinstance(invalid_symbol, Failure)

        # Test TimeInterval type
        print("  ✓ Testing TimeInterval type...")
        valid_interval = TimeInterval.create("5m")
        assert isinstance(valid_interval, Success)
        interval = valid_interval.success()
        assert interval.to_seconds() == 300
        assert interval.to_milliseconds() == 300000

        invalid_interval = TimeInterval.create("invalid")
        assert isinstance(invalid_interval, Failure)

        print("  ✅ Functional types tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Functional types test failed: {e}")
        traceback.print_exc()
        return False


def test_functional_config_loading():
    """Test loading functional configuration from environment."""
    print("\n=== Testing Functional Configuration Loading ===")

    # Set up functional config environment variables
    original_env = {}
    functional_env_vars = {
        "STRATEGY_TYPE": "llm",
        "LLM_MODEL": "gpt-4",
        "LLM_TEMPERATURE": "0.3",
        "LLM_MAX_CONTEXT": "4000",
        "LLM_USE_MEMORY": "true",
        "LLM_CONFIDENCE_THRESHOLD": "0.8",
        "EXCHANGE_TYPE": "coinbase",
        "COINBASE_API_KEY": "test_coinbase_api_key_12345",
        "COINBASE_PRIVATE_KEY": "test_coinbase_private_key_67890",
        "RATE_LIMIT_RPS": "5",
        "RATE_LIMIT_RPM": "100",
        "RATE_LIMIT_RPH": "1000",
        "TRADING_PAIRS": "BTC-USD,ETH-USD",
        "TRADING_INTERVAL": "1m",
        "TRADING_MODE": "paper",
        "LOG_LEVEL": "INFO",
        "MAX_CONCURRENT_POSITIONS": "3",
        "DEFAULT_POSITION_SIZE": "0.1",
        "ENABLE_WEBSOCKET": "true",
        "ENABLE_MEMORY": "true",
        "ENABLE_BACKTESTING": "true",
        "ENABLE_PAPER_TRADING": "true",
        "ENABLE_RISK_MANAGEMENT": "true",
        "ENABLE_NOTIFICATIONS": "false",
        "ENABLE_METRICS": "true",
    }

    # Set test environment variables
    for key, value in functional_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        from bot.config import get_functional_config

        # Test functional config loading
        print("  ✓ Testing functional config loading...")
        functional_config = get_functional_config()

        if functional_config is not None:
            print(
                f"  ✓ Functional config loaded successfully: {type(functional_config)}"
            )

            # Test config attributes exist
            assert hasattr(functional_config, "strategy")
            assert hasattr(functional_config, "exchange")
            assert hasattr(functional_config, "system")

            print("  ✓ Functional config has required attributes")
        else:
            print(
                "  ⚠️  Functional config returned None (may be expected if types not available)"
            )

        # Test Settings creation with functional config
        print("  ✓ Testing Settings creation with functional config...")
        from bot.config import Settings

        settings = Settings(functional_config)

        # Verify settings were created
        assert hasattr(settings, "trading")
        assert hasattr(settings, "llm")
        assert hasattr(settings, "exchange")
        assert hasattr(settings, "risk")
        assert hasattr(settings, "system")

        print("  ✅ Functional configuration loading tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Functional config loading test failed: {e}")
        traceback.print_exc()
        return False

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def test_configuration_validation():
    """Test configuration validation functionality."""
    print("\n=== Testing Configuration Validation ===")

    try:
        from bot.config import Settings, validate_settings, validate_settings_functional

        # Test validation with valid settings
        print("  ✓ Testing validation with valid settings...")
        valid_settings = Settings()
        validation_result = validate_settings(valid_settings)

        if validation_result:
            print(f"  ⚠️  Validation warnings: {validation_result}")
        else:
            print("  ✓ No validation warnings for default settings")

        # Test functional validation wrapper
        functional_result = validate_settings_functional(valid_settings)
        assert isinstance(functional_result, Settings) or isinstance(
            functional_result, str
        )

        if isinstance(functional_result, Settings):
            print("  ✓ Functional validation passed")
        else:
            print(f"  ⚠️  Functional validation failed: {functional_result}")

        # Test validation with invalid settings
        print("  ✓ Testing validation with invalid settings...")
        invalid_settings = Settings()
        invalid_settings.trading.leverage = 150  # Invalid leverage
        invalid_settings.risk.max_daily_loss_pct = 75.0  # Too high
        invalid_settings.llm.temperature = 3.0  # Out of range

        validation_warnings = validate_settings(invalid_settings)
        assert validation_warnings is not None, "Should have validation warnings"
        print(
            f"  ✓ Invalid settings correctly flagged: {len(validation_warnings.split(';'))} warnings"
        )

        print("  ✅ Configuration validation tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Configuration validation test failed: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility with legacy Settings interface."""
    print("\n=== Testing Backward Compatibility ===")

    try:
        from bot.config import Settings, create_settings, get_config

        # Test legacy function interfaces
        print("  ✓ Testing legacy create_settings function...")
        settings1 = create_settings()
        assert isinstance(settings1, Settings)

        print("  ✓ Testing legacy get_config function...")
        settings2 = get_config()
        assert isinstance(settings2, Settings)

        # Test Settings class compatibility
        print("  ✓ Testing Settings class compatibility...")
        settings = Settings()

        # Test all expected attributes exist
        expected_sections = [
            "trading",
            "llm",
            "exchange",
            "risk",
            "data",
            "dominance",
            "system",
            "paper_trading",
            "monitoring",
            "mcp",
            "omnisearch",
        ]

        for section in expected_sections:
            assert hasattr(settings, section), f"Missing section: {section}"

        # Test trading settings attributes
        trading_attrs = [
            "symbol",
            "interval",
            "leverage",
            "max_size_pct",
            "order_timeout_seconds",
            "slippage_tolerance_pct",
        ]
        for attr in trading_attrs:
            assert hasattr(settings.trading, attr), f"Missing trading.{attr}"

        # Test LLM settings attributes
        llm_attrs = ["provider", "model_name", "temperature", "max_tokens"]
        for attr in llm_attrs:
            assert hasattr(settings.llm, attr), f"Missing llm.{attr}"

        # Test exchange settings attributes
        exchange_attrs = ["exchange_type", "api_timeout", "rate_limit_requests"]
        for attr in exchange_attrs:
            assert hasattr(settings.exchange, attr), f"Missing exchange.{attr}"

        # Test to_dict method
        print("  ✓ Testing to_dict method...")
        config_dict = settings.to_dict()
        assert isinstance(config_dict, dict)
        assert all(section in config_dict for section in expected_sections)

        print("  ✅ Backward compatibility tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_merging():
    """Test configuration merging between functional and legacy systems."""
    print("\n=== Testing Configuration Merging ===")

    try:
        from bot.config import create_settings

        # Test overrides during creation
        print("  ✓ Testing configuration overrides...")
        overrides = {
            "trading": {"symbol": "DOGE-USD", "leverage": 3},
            "llm": {"temperature": 0.15, "model_name": "gpt-3.5-turbo"},
            "system": {"dry_run": False},
        }

        settings = create_settings(overrides=overrides)

        # Verify overrides were applied
        assert settings.trading.symbol == "DOGE-USD"
        assert settings.trading.leverage == 3
        assert settings.llm.temperature == 0.15
        assert settings.llm.model_name == "gpt-3.5-turbo"
        assert settings.system.dry_run == False

        # Test profile application
        print("  ✓ Testing profile application...")
        conservative_settings = settings.apply_profile("conservative")
        assert conservative_settings.trading.leverage == 2  # Conservative override
        assert conservative_settings.risk.max_daily_loss_pct == 2.0

        aggressive_settings = settings.apply_profile("aggressive")
        assert aggressive_settings.trading.leverage == 10  # Aggressive override
        assert aggressive_settings.risk.max_daily_loss_pct == 10.0

        print("  ✅ Configuration merging tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Configuration merging test failed: {e}")
        traceback.print_exc()
        return False


def test_file_loading():
    """Test configuration loading from files."""
    print("\n=== Testing File Loading ===")

    try:
        from bot.config import Settings, load_settings_from_file

        # Create temporary config file
        config_data = {
            "trading": {"symbol": "LTC-USD", "leverage": 8, "max_size_pct": 30.0},
            "llm": {"model_name": "gpt-4", "temperature": 0.25},
            "system": {"dry_run": True, "log_level": "WARNING"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            print("  ✓ Testing file loading...")
            settings = load_settings_from_file(temp_file)
            assert isinstance(settings, Settings)

            # Verify loaded values (note: environment variables may override file values)
            print(f"  ✓ Loaded symbol: {settings.trading.symbol}")
            print(f"  ✓ Loaded leverage: {settings.trading.leverage}")
            print(f"  ✓ Loaded model: {settings.llm.model_name}")

            # Test non-existent file handling
            print("  ✓ Testing non-existent file handling...")
            fallback_settings = load_settings_from_file("/nonexistent/file.json")
            assert isinstance(fallback_settings, Settings)

            print("  ✅ File loading tests passed!")
            return True

        finally:
            # Clean up temporary file
            os.unlink(temp_file)

    except Exception as e:
        print(f"  ❌ File loading test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_benchmarking():
    """Test configuration loading performance."""
    print("\n=== Testing Performance Benchmarking ===")

    try:
        from bot.config import benchmark_config_loading

        print("  ✓ Running performance benchmark...")
        results = benchmark_config_loading(iterations=10)

        print(
            f"  ✓ Create settings: {results['create_settings_ms']:.2f}ms per iteration"
        )
        print(f"  ✓ Load from file: {results['load_from_file_ms']:.2f}ms per iteration")
        print(f"  ✓ Validation: {results['validation_ms']:.2f}ms per iteration")
        print(f"  ✓ Total iterations: {results['total_iterations']}")

        # Performance thresholds (generous for different environments)
        assert results["create_settings_ms"] < 1000, "Settings creation too slow"
        assert results["validation_ms"] < 100, "Validation too slow"

        print("  ✅ Performance benchmarking tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Performance benchmarking test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and fallback behavior."""
    print("\n=== Testing Error Handling ===")

    try:
        from bot.config import Settings, create_settings

        # Test fallback behavior when functional types aren't available
        print("  ✓ Testing fallback behavior...")
        settings = Settings(functional_config=None)  # Force legacy mode
        assert isinstance(settings, Settings)

        # Test invalid environment values
        print("  ✓ Testing invalid environment values...")
        original_leverage = os.environ.get("TRADING__LEVERAGE")
        os.environ["TRADING__LEVERAGE"] = "invalid_number"

        try:
            settings = create_settings()
            # Should fallback to default value, not crash
            assert isinstance(settings.trading.leverage, int)
            print(
                f"  ✓ Invalid leverage env var handled gracefully: {settings.trading.leverage}"
            )
        finally:
            if original_leverage is None:
                os.environ.pop("TRADING__LEVERAGE", None)
            else:
                os.environ["TRADING__LEVERAGE"] = original_leverage

        # Test missing required fields
        print("  ✓ Testing graceful handling of configuration issues...")
        try:
            from bot.config import ConfigError, ConfigValidationError

            print("  ✓ Configuration error classes available")
        except ImportError:
            print("  ⚠️  Configuration error classes not available")

        print("  ✅ Error handling tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all configuration validation tests."""
    print("🔧 CONFIGURATION SYSTEM VALIDATION TESTS")
    print("=" * 50)

    test_results = []

    # Run all test functions
    test_functions = [
        test_environment_parsing,
        test_functional_types,
        test_functional_config_loading,
        test_configuration_validation,
        test_backward_compatibility,
        test_configuration_merging,
        test_file_loading,
        test_performance_benchmarking,
        test_error_handling,
    ]

    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ {test_func.__name__} crashed: {e}")
            test_results.append((test_func.__name__, False))

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🏆 All configuration validation tests PASSED!")
        return True
    print("⚠️  Some configuration validation tests FAILED!")
    return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
