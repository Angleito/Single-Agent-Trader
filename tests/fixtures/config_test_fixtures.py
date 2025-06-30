"""
Configuration Test Fixtures

This module provides comprehensive test fixtures for configuration testing
including various configuration scenarios, validation test cases, and
environment simulation for robust configuration management testing.

Features:
- Valid and invalid configuration samples
- Environment variable simulation
- Configuration migration testing
- Edge case configuration scenarios
- Security and validation testing
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Safe imports with fallbacks for FP types
try:
    from bot.fp.types.config import (
        BacktestConfig,
        Config,
        ExchangeConfig,
        LLMConfig,
        RiskConfig,
        TradingConfig,
    )

    FP_CONFIG_AVAILABLE = True
except ImportError:
    FP_CONFIG_AVAILABLE = False

# Legacy config import
try:
    from bot.config import Settings

    LEGACY_CONFIG_AVAILABLE = True
except ImportError:
    LEGACY_CONFIG_AVAILABLE = False


@dataclass
class ConfigTestScenario:
    """Test scenario for configuration testing."""

    name: str
    description: str
    config_data: dict[str, Any]
    env_vars: dict[str, str] = field(default_factory=dict)
    expected_valid: bool = True
    expected_errors: list[str] = field(default_factory=list)
    test_type: str = "functional"  # functional, security, edge_case, migration


class ConfigTestFixtures:
    """Generator for configuration test fixtures."""

    def __init__(self):
        self.temp_dir = None
        self.temp_files = []

    def create_temp_config_file(
        self, config_data: dict[str, Any], filename: str = None
    ) -> Path:
        """Create temporary configuration file."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()

        filename = filename or f"test_config_{len(self.temp_files)}.json"
        filepath = Path(self.temp_dir) / filename

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)

        self.temp_files.append(filepath)
        return filepath

    def cleanup_temp_files(self):
        """Clean up temporary files."""
        for filepath in self.temp_files:
            try:
                filepath.unlink()
            except FileNotFoundError:
                pass

        if self.temp_dir:
            try:
                Path(self.temp_dir).rmdir()
            except OSError:
                pass

    def generate_valid_config_scenarios(self) -> list[ConfigTestScenario]:
        """Generate valid configuration test scenarios."""
        scenarios = []

        # Scenario 1: Minimal valid configuration
        scenarios.append(
            ConfigTestScenario(
                name="minimal_valid_config",
                description="Minimal configuration with only required fields",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-coinbase-key",
                    "COINBASE_PRIVATE_KEY": "test-private-key",
                },
                expected_valid=True,
            )
        )

        # Scenario 2: Complete configuration
        scenarios.append(
            ConfigTestScenario(
                name="complete_valid_config",
                description="Complete configuration with all optional fields",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "live",
                    "exchange_type": "coinbase",
                    "trading_pairs": ["BTC-USD", "ETH-USD"],
                    "trading_interval": "5m",
                    "max_concurrent_positions": 3,
                    "default_position_size": 0.25,
                    "enable_websocket": True,
                    "enable_risk_management": True,
                    "log_level": "INFO",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "live",
                    "EXCHANGE_TYPE": "coinbase",
                    "TRADING_PAIRS": "BTC-USD,ETH-USD",
                    "TRADING_INTERVAL": "5m",
                    "MAX_CONCURRENT_POSITIONS": "3",
                    "DEFAULT_POSITION_SIZE": "0.25",
                    "ENABLE_WEBSOCKET": "true",
                    "ENABLE_RISK_MANAGEMENT": "true",
                    "LOG_LEVEL": "INFO",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-coinbase-key",
                    "COINBASE_PRIVATE_KEY": "test-private-key",
                },
                expected_valid=True,
            )
        )

        # Scenario 3: Bluefin exchange configuration
        scenarios.append(
            ConfigTestScenario(
                name="bluefin_exchange_config",
                description="Valid configuration for Bluefin exchange",
                config_data={
                    "strategy_type": "momentum",
                    "trading_mode": "paper",
                    "exchange_type": "bluefin",
                },
                env_vars={
                    "STRATEGY_TYPE": "momentum",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "bluefin",
                    "BLUEFIN_PRIVATE_KEY": "test-bluefin-key",
                    "BLUEFIN_NETWORK": "testnet",
                },
                expected_valid=True,
            )
        )

        # Scenario 4: Backtest configuration
        scenarios.append(
            ConfigTestScenario(
                name="backtest_config",
                description="Valid backtest configuration",
                config_data={
                    "strategy_type": "mean_reversion",
                    "trading_mode": "backtest",
                },
                env_vars={
                    "STRATEGY_TYPE": "mean_reversion",
                    "TRADING_MODE": "backtest",
                    "BACKTEST_START_DATE": "2024-01-01",
                    "BACKTEST_END_DATE": "2024-12-31",
                    "BACKTEST_INITIAL_CAPITAL": "100000.0",
                    "BACKTEST_CURRENCY": "USD",
                },
                expected_valid=True,
            )
        )

        # Scenario 5: High precision configuration
        scenarios.append(
            ConfigTestScenario(
                name="high_precision_config",
                description="Configuration with high precision values",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                    "default_position_size": 0.123456789,
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "DEFAULT_POSITION_SIZE": "0.123456789",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-coinbase-key",
                    "COINBASE_PRIVATE_KEY": "test-private-key",
                },
                expected_valid=True,
            )
        )

        return scenarios

    def generate_invalid_config_scenarios(self) -> list[ConfigTestScenario]:
        """Generate invalid configuration test scenarios."""
        scenarios = []

        # Scenario 1: Missing required fields
        scenarios.append(
            ConfigTestScenario(
                name="missing_required_fields",
                description="Configuration missing required strategy_type",
                config_data={"trading_mode": "paper", "exchange_type": "coinbase"},
                env_vars={"TRADING_MODE": "paper", "EXCHANGE_TYPE": "coinbase"},
                expected_valid=False,
                expected_errors=["strategy_type is required"],
                test_type="edge_case",
            )
        )

        # Scenario 2: Invalid enum values
        scenarios.append(
            ConfigTestScenario(
                name="invalid_enum_values",
                description="Configuration with invalid enum values",
                config_data={
                    "strategy_type": "invalid_strategy",
                    "trading_mode": "invalid_mode",
                    "exchange_type": "invalid_exchange",
                },
                env_vars={
                    "STRATEGY_TYPE": "invalid_strategy",
                    "TRADING_MODE": "invalid_mode",
                    "EXCHANGE_TYPE": "invalid_exchange",
                },
                expected_valid=False,
                expected_errors=[
                    "invalid strategy_type",
                    "invalid trading_mode",
                    "invalid exchange_type",
                ],
                test_type="edge_case",
            )
        )

        # Scenario 3: Invalid numeric ranges
        scenarios.append(
            ConfigTestScenario(
                name="invalid_numeric_ranges",
                description="Configuration with out-of-range numeric values",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                    "max_concurrent_positions": -1,
                    "default_position_size": 1.5,  # > 1.0
                    "llm_temperature": 5.0,  # > 2.0
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "MAX_CONCURRENT_POSITIONS": "-1",
                    "DEFAULT_POSITION_SIZE": "1.5",
                    "LLM_TEMPERATURE": "5.0",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=False,
                expected_errors=[
                    "max_concurrent_positions must be positive",
                    "default_position_size must be <= 1.0",
                    "llm_temperature must be <= 2.0",
                ],
                test_type="edge_case",
            )
        )

        # Scenario 4: Missing credentials
        scenarios.append(
            ConfigTestScenario(
                name="missing_credentials",
                description="Configuration missing required API credentials",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "live",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "live",
                    "EXCHANGE_TYPE": "coinbase",
                    # Missing: LLM_OPENAI_API_KEY, COINBASE_API_KEY, COINBASE_PRIVATE_KEY
                },
                expected_valid=False,
                expected_errors=[
                    "LLM_OPENAI_API_KEY is required",
                    "COINBASE_API_KEY is required",
                    "COINBASE_PRIVATE_KEY is required",
                ],
                test_type="security",
            )
        )

        # Scenario 5: Invalid date formats
        scenarios.append(
            ConfigTestScenario(
                name="invalid_date_formats",
                description="Configuration with invalid date formats",
                config_data={"strategy_type": "llm", "trading_mode": "backtest"},
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "backtest",
                    "BACKTEST_START_DATE": "invalid-date",
                    "BACKTEST_END_DATE": "2024-13-45",  # Invalid month/day
                },
                expected_valid=False,
                expected_errors=[
                    "invalid start_date format",
                    "invalid end_date format",
                ],
                test_type="edge_case",
            )
        )

        return scenarios

    def generate_security_test_scenarios(self) -> list[ConfigTestScenario]:
        """Generate security-focused test scenarios."""
        scenarios = []

        # Scenario 1: Credential exposure in logs
        scenarios.append(
            ConfigTestScenario(
                name="credential_exposure_test",
                description="Test that credentials are not exposed in logs or errors",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "LLM_OPENAI_API_KEY": "sk-super-secret-api-key-12345",
                    "COINBASE_API_KEY": "coinbase-secret-key-67890",
                    "COINBASE_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----\nVERY_SECRET_PRIVATE_KEY\n-----END PRIVATE KEY-----",
                },
                expected_valid=True,
                test_type="security",
            )
        )

        # Scenario 2: Injection attack simulation
        scenarios.append(
            ConfigTestScenario(
                name="injection_attack_simulation",
                description="Test resistance to injection attacks in config values",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm'; DROP TABLE users; --",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=False,
                expected_errors=["invalid strategy_type"],
                test_type="security",
            )
        )

        # Scenario 3: Path traversal simulation
        scenarios.append(
            ConfigTestScenario(
                name="path_traversal_simulation",
                description="Test resistance to path traversal attacks",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "CONFIG_FILE_PATH": "../../../etc/passwd",
                    "LOG_FILE_PATH": "../../sensitive_logs.txt",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=False,
                expected_errors=["invalid file path"],
                test_type="security",
            )
        )

        return scenarios

    def generate_edge_case_scenarios(self) -> list[ConfigTestScenario]:
        """Generate edge case test scenarios."""
        scenarios = []

        # Scenario 1: Empty values
        scenarios.append(
            ConfigTestScenario(
                name="empty_values",
                description="Configuration with empty string values",
                config_data={
                    "strategy_type": "",
                    "trading_mode": "",
                    "exchange_type": "",
                },
                env_vars={"STRATEGY_TYPE": "", "TRADING_MODE": "", "EXCHANGE_TYPE": ""},
                expected_valid=False,
                expected_errors=["empty values not allowed"],
                test_type="edge_case",
            )
        )

        # Scenario 2: Very long values
        scenarios.append(
            ConfigTestScenario(
                name="very_long_values",
                description="Configuration with extremely long values",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "LLM_OPENAI_API_KEY": "x" * 10000,  # Very long key
                    "COINBASE_API_KEY": "y" * 5000,
                    "COINBASE_PRIVATE_KEY": "z" * 20000,
                },
                expected_valid=False,
                expected_errors=["value too long"],
                test_type="edge_case",
            )
        )

        # Scenario 3: Unicode and special characters
        scenarios.append(
            ConfigTestScenario(
                name="unicode_special_chars",
                description="Configuration with unicode and special characters",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "SYMBOL_PREFIX": "🚀💰",
                    "CUSTOM_NOTE": "测试配置 with émojis! @#$%^&*()",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=True,  # Should handle unicode gracefully
                test_type="edge_case",
            )
        )

        # Scenario 4: Boundary values
        scenarios.append(
            ConfigTestScenario(
                name="boundary_values",
                description="Configuration with boundary numeric values",
                config_data={
                    "strategy_type": "llm",
                    "trading_mode": "paper",
                    "exchange_type": "coinbase",
                    "max_concurrent_positions": 1,  # Minimum
                    "default_position_size": 0.0001,  # Very small
                    "llm_temperature": 0.0,  # Minimum
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "MAX_CONCURRENT_POSITIONS": "1",
                    "DEFAULT_POSITION_SIZE": "0.0001",
                    "LLM_TEMPERATURE": "0.0",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=True,
                test_type="edge_case",
            )
        )

        return scenarios

    def generate_migration_test_scenarios(self) -> list[ConfigTestScenario]:
        """Generate configuration migration test scenarios."""
        scenarios = []

        # Scenario 1: Legacy to FP migration
        scenarios.append(
            ConfigTestScenario(
                name="legacy_to_fp_migration",
                description="Migration from legacy config format to FP format",
                config_data={
                    # Legacy format
                    "SYSTEM__DRY_RUN": "true",
                    "EXCHANGE__EXCHANGE_TYPE": "coinbase",
                    "TRADING__SYMBOL": "BTC-USD",
                    "LLM__MODEL_NAME": "gpt-4",
                },
                env_vars={
                    # New FP format
                    "TRADING_MODE": "paper",
                    "EXCHANGE_TYPE": "coinbase",
                    "TRADING_PAIRS": "BTC-USD",
                    "LLM_MODEL": "gpt-4",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=True,
                test_type="migration",
            )
        )

        # Scenario 2: Mixed format compatibility
        scenarios.append(
            ConfigTestScenario(
                name="mixed_format_compatibility",
                description="Mixed legacy and FP format configuration",
                config_data={
                    "strategy_type": "llm",  # FP format
                    "SYSTEM__DRY_RUN": "true",  # Legacy format
                    "exchange_type": "coinbase",  # FP format
                },
                env_vars={
                    "STRATEGY_TYPE": "llm",
                    "SYSTEM__DRY_RUN": "true",  # Legacy
                    "EXCHANGE_TYPE": "coinbase",
                    "LLM_OPENAI_API_KEY": "test-key",
                    "COINBASE_API_KEY": "test-key",
                    "COINBASE_PRIVATE_KEY": "test-key",
                },
                expected_valid=True,
                test_type="migration",
            )
        )

        return scenarios

    def create_environment_context(self, env_vars: dict[str, str]):
        """Create environment variable context for testing."""
        return patch.dict(os.environ, env_vars, clear=False)

    def generate_all_scenarios(self) -> dict[str, list[ConfigTestScenario]]:
        """Generate all configuration test scenarios."""
        return {
            "valid_configs": self.generate_valid_config_scenarios(),
            "invalid_configs": self.generate_invalid_config_scenarios(),
            "security_tests": self.generate_security_test_scenarios(),
            "edge_cases": self.generate_edge_case_scenarios(),
            "migration_tests": self.generate_migration_test_scenarios(),
        }


class ConfigValidationTestSuite:
    """Test suite for configuration validation."""

    def __init__(self):
        self.fixtures = ConfigTestFixtures()
        self.results = []

    def run_validation_tests(
        self, scenarios: list[ConfigTestScenario]
    ) -> dict[str, Any]:
        """Run validation tests on provided scenarios."""
        results = {
            "total_scenarios": len(scenarios),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "scenario_results": [],
        }

        for scenario in scenarios:
            try:
                result = self._test_scenario(scenario)
                results["scenario_results"].append(result)

                if result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["errors"] += 1
                results["scenario_results"].append(
                    {
                        "scenario_name": scenario.name,
                        "passed": False,
                        "error": str(e),
                        "exception_type": type(e).__name__,
                    }
                )

        return results

    def _test_scenario(self, scenario: ConfigTestScenario) -> dict[str, Any]:
        """Test a single configuration scenario."""
        result = {
            "scenario_name": scenario.name,
            "description": scenario.description,
            "test_type": scenario.test_type,
            "passed": False,
            "validation_errors": [],
            "security_issues": [],
            "performance_issues": [],
        }

        # Test with environment variables
        with self.fixtures.create_environment_context(scenario.env_vars):
            try:
                if FP_CONFIG_AVAILABLE:
                    # Test FP configuration
                    config_result = self._test_fp_config(scenario)
                    result.update(config_result)

                if LEGACY_CONFIG_AVAILABLE:
                    # Test legacy configuration
                    legacy_result = self._test_legacy_config(scenario)
                    result["legacy_config_result"] = legacy_result

            except Exception as e:
                result["validation_errors"].append(str(e))

        # Check if result matches expectation
        if scenario.expected_valid:
            result["passed"] = len(result["validation_errors"]) == 0
        else:
            # For invalid scenarios, we expect validation errors
            result["passed"] = len(result["validation_errors"]) > 0

            # Check if expected errors are present
            expected_found = all(
                any(expected in error for error in result["validation_errors"])
                for expected in scenario.expected_errors
            )
            result["expected_errors_found"] = expected_found

        return result

    def _test_fp_config(self, scenario: ConfigTestScenario) -> dict[str, Any]:
        """Test FP configuration."""
        try:
            from bot.fp.types.config import Config

            config_result = Config.from_env()

            if config_result.is_success():
                config = config_result.value()
                return {
                    "fp_config_valid": True,
                    "fp_config": config,
                    "validation_errors": [],
                }
            return {
                "fp_config_valid": False,
                "validation_errors": [str(config_result.failure())],
            }

        except Exception as e:
            return {
                "fp_config_valid": False,
                "validation_errors": [f"FP config error: {e!s}"],
            }

    def _test_legacy_config(self, scenario: ConfigTestScenario) -> dict[str, Any]:
        """Test legacy configuration."""
        try:
            from bot.config import Settings

            settings = Settings()
            return {
                "legacy_config_valid": True,
                "legacy_config": settings,
                "validation_errors": [],
            }

        except Exception as e:
            return {
                "legacy_config_valid": False,
                "validation_errors": [f"Legacy config error: {e!s}"],
            }

    def cleanup(self):
        """Cleanup test fixtures."""
        self.fixtures.cleanup_temp_files()


# Factory function for easy access
def create_config_test_suite() -> dict[str, Any]:
    """Create complete configuration test suite."""
    fixtures = ConfigTestFixtures()
    test_suite = ConfigValidationTestSuite()

    all_scenarios = fixtures.generate_all_scenarios()

    return {
        # Test scenarios by category
        "scenarios": all_scenarios,
        # Flattened list of all scenarios
        "all_scenarios": [
            scenario
            for scenario_list in all_scenarios.values()
            for scenario in scenario_list
        ],
        # Test utilities
        "fixtures": fixtures,
        "test_suite": test_suite,
        # Summary information
        "summary": {
            "total_scenarios": sum(
                len(scenarios) for scenarios in all_scenarios.values()
            ),
            "scenario_categories": list(all_scenarios.keys()),
            "fp_config_available": FP_CONFIG_AVAILABLE,
            "legacy_config_available": LEGACY_CONFIG_AVAILABLE,
        },
        # Generation metadata
        "generation_timestamp": datetime.now(UTC),
        "test_capabilities": [
            "environment_variable_testing",
            "config_file_testing",
            "validation_testing",
            "security_testing",
            "edge_case_testing",
            "migration_testing",
        ],
    }


# Export main classes and functions
__all__ = [
    "ConfigTestFixtures",
    "ConfigTestScenario",
    "ConfigValidationTestSuite",
    "create_config_test_suite",
]
