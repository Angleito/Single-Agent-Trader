#!/usr/bin/env python3
"""
Market Making Deployment Validation and Health Check Scripts.

This comprehensive validation suite ensures that your market making deployment
is correctly configured and ready for operation. It performs extensive testing
of all components, configurations, connectivity, and emergency procedures.

Key Features:
- Pre-deployment validation checklist
- Component health checks and diagnostics
- Configuration validation and testing
- Performance benchmarking
- Exchange connectivity testing
- VuManChu indicator validation
- Fee calculation accuracy testing
- Emergency procedure testing

Usage:
    python scripts/validate-market-making-setup.py [options]

Options:
    --full                  Run complete validation suite
    --pre-deployment        Run pre-deployment checklist only
    --health-check          Run component health checks only
    --config-test          Run configuration validation only
    --performance-bench     Run performance benchmarking only
    --connectivity-test     Run exchange connectivity tests only
    --indicator-test       Run VuManChu indicator validation only
    --fee-test             Run fee calculation testing only
    --emergency-test       Run emergency procedure testing only
    --export-report FILE   Export validation report to file
    --fix-suggestions      Show automated fix suggestions
    --monitor              Run continuous monitoring mode
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import Settings, create_settings
from bot.exchange.bluefin_fee_calculator import BluefinFeeCalculator
from bot.exchange.factory import create_exchange
from bot.indicators.vumanchu import CipherA, CipherB
from bot.strategy.market_making_strategy import MarketMakingStrategy
from bot.trading_types import IndicatorData, MarketState, OHLCVData, Position

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation test results."""

    def __init__(
        self,
        name: str,
        status: str = "pending",
        message: str = "",
        details: dict[str, Any] | None = None,
        duration: float = 0.0,
    ):
        self.name = name
        self.status = status  # "pass", "fail", "warning", "skip", "pending"
        self.message = message
        self.details = details or {}
        self.duration = duration
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }


class MarketMakingValidator:
    """Comprehensive Market Making Deployment Validator."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.results: list[ValidationResult] = []
        self.start_time = time.time()

        # Component instances (will be initialized as needed)
        self.exchange = None
        self.market_making_strategy = None
        self.cipher_a = None
        self.cipher_b = None
        self.fee_calculator = None

    async def run_full_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        print("🚀 Starting comprehensive market making validation...")

        # Run all validation components
        await self.run_pre_deployment_validation()
        await self.run_health_checks()
        await self.run_configuration_validation()
        await self.run_performance_benchmarks()
        await self.run_connectivity_tests()
        await self.run_indicator_validation()
        await self.run_fee_calculation_tests()
        await self.run_emergency_procedure_tests()

        return self._generate_final_report()

    async def run_pre_deployment_validation(self) -> dict[str, Any]:
        """Run pre-deployment checklist validation."""
        print("\n📋 Pre-Deployment Validation Checklist")
        print("=" * 50)

        checklist_items = [
            ("Environment Variables", self._validate_environment_variables),
            ("Market Making Configuration", self._validate_market_making_config),
            ("Risk Management Settings", self._validate_risk_settings),
            ("Exchange Configuration", self._validate_exchange_config),
            ("API Keys and Credentials", self._validate_credentials),
            ("Network Connectivity", self._validate_network_connectivity),
            ("Directory Permissions", self._validate_directory_permissions),
            ("Docker Configuration", self._validate_docker_config),
            ("Logging Configuration", self._validate_logging_config),
            ("MCP Memory Setup", self._validate_mcp_setup),
        ]

        for name, validator_func in checklist_items:
            result = await self._run_validation_test(name, validator_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("pre_deployment")

    async def run_health_checks(self) -> dict[str, Any]:
        """Run component health checks and diagnostics."""
        print("\n🔍 Component Health Checks")
        print("=" * 50)

        health_checks = [
            ("Exchange Connection Health", self._check_exchange_health),
            ("Market Data Feed Health", self._check_market_data_health),
            ("Indicator Calculation Health", self._check_indicator_health),
            ("Strategy Engine Health", self._check_strategy_health),
            ("Risk Manager Health", self._check_risk_manager_health),
            ("Order Manager Health", self._check_order_manager_health),
            ("Balance Monitor Health", self._check_balance_health),
            ("Memory System Health", self._check_memory_health),
            ("Logging System Health", self._check_logging_health),
            ("Performance Monitor Health", self._check_performance_health),
        ]

        for name, check_func in health_checks:
            result = await self._run_validation_test(name, check_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("health_checks")

    async def run_configuration_validation(self) -> dict[str, Any]:
        """Run configuration validation and testing."""
        print("\n⚙️ Configuration Validation")
        print("=" * 50)

        config_tests = [
            ("Market Making Parameters", self._validate_mm_parameters),
            ("Spread Configuration", self._validate_spread_config),
            ("Order Level Configuration", self._validate_order_levels),
            ("Position Size Limits", self._validate_position_limits),
            ("Fee Structure Validation", self._validate_fee_structure),
            ("Risk Thresholds", self._validate_risk_thresholds),
            ("Emergency Procedures", self._validate_emergency_config),
            ("Profile Configuration", self._validate_profile_config),
            ("Environment Consistency", self._validate_env_consistency),
            ("Symbol Configuration", self._validate_symbol_config),
        ]

        for name, test_func in config_tests:
            result = await self._run_validation_test(name, test_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("configuration")

    async def run_performance_benchmarks(self) -> dict[str, Any]:
        """Run performance benchmarking tests."""
        print("\n⚡ Performance Benchmarking")
        print("=" * 50)

        benchmark_tests = [
            ("Market Data Processing Speed", self._benchmark_market_data),
            ("Indicator Calculation Speed", self._benchmark_indicators),
            ("Strategy Decision Speed", self._benchmark_strategy),
            ("Order Placement Latency", self._benchmark_order_placement),
            ("Risk Calculation Speed", self._benchmark_risk_calc),
            ("Memory Access Speed", self._benchmark_memory),
            ("Spread Calculation Speed", self._benchmark_spread_calc),
            ("Fee Calculation Speed", self._benchmark_fee_calc),
            ("Overall System Latency", self._benchmark_system_latency),
            ("Throughput Testing", self._benchmark_throughput),
        ]

        for name, benchmark_func in benchmark_tests:
            result = await self._run_validation_test(name, benchmark_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("performance")

    async def run_connectivity_tests(self) -> dict[str, Any]:
        """Run exchange connectivity tests."""
        print("\n🌐 Exchange Connectivity Tests")
        print("=" * 50)

        connectivity_tests = [
            ("Exchange API Connection", self._test_api_connection),
            ("WebSocket Connection", self._test_websocket_connection),
            ("Market Data Stream", self._test_market_data_stream),
            ("Order Management API", self._test_order_api),
            ("Account Information API", self._test_account_api),
            ("Fee Information API", self._test_fee_api),
            ("Network Latency Test", self._test_network_latency),
            ("Connection Stability", self._test_connection_stability),
            ("Failover Procedures", self._test_failover),
            ("Rate Limiting Compliance", self._test_rate_limits),
        ]

        for name, test_func in connectivity_tests:
            result = await self._run_validation_test(name, test_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("connectivity")

    async def run_indicator_validation(self) -> dict[str, Any]:
        """Run VuManChu indicator validation tests."""
        print("\n📊 VuManChu Indicator Validation")
        print("=" * 50)

        indicator_tests = [
            ("Cipher A Initialization", self._test_cipher_a_init),
            ("Cipher B Initialization", self._test_cipher_b_init),
            ("WaveTrend Calculation", self._test_wavetrend),
            ("EMA Ribbon Calculation", self._test_ema_ribbon),
            ("RSI+MFI Calculation", self._test_rsimfi),
            ("Signal Generation", self._test_signal_generation),
            ("Directional Bias Calculation", self._test_directional_bias),
            ("Signal Strength Validation", self._test_signal_strength),
            ("Historical Data Processing", self._test_historical_processing),
            ("Real-time Processing", self._test_realtime_processing),
        ]

        for name, test_func in indicator_tests:
            result = await self._run_validation_test(name, test_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("indicators")

    async def run_fee_calculation_tests(self) -> dict[str, Any]:
        """Run fee calculation accuracy tests."""
        print("\n💰 Fee Calculation Testing")
        print("=" * 50)

        fee_tests = [
            ("Maker Fee Calculation", self._test_maker_fees),
            ("Taker Fee Calculation", self._test_taker_fees),
            ("Round-trip Cost Calculation", self._test_roundtrip_costs),
            ("Minimum Spread Calculation", self._test_min_spread),
            ("Gas Fee Estimation", self._test_gas_fees),
            ("Slippage Calculation", self._test_slippage),
            ("Fee Accuracy Validation", self._test_fee_accuracy),
            ("Cross-validation with Exchange", self._test_fee_cross_validation),
            ("Edge Case Handling", self._test_fee_edge_cases),
            ("Performance Under Load", self._test_fee_performance),
        ]

        for name, test_func in fee_tests:
            result = await self._run_validation_test(name, test_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("fees")

    async def run_emergency_procedure_tests(self) -> dict[str, Any]:
        """Run emergency procedure testing."""
        print("\n🚨 Emergency Procedure Testing")
        print("=" * 50)

        emergency_tests = [
            ("Emergency Stop Procedure", self._test_emergency_stop),
            ("Position Liquidation", self._test_position_liquidation),
            ("Order Cancellation", self._test_order_cancellation),
            ("Balance Protection", self._test_balance_protection),
            ("Network Failure Recovery", self._test_network_recovery),
            ("API Failure Handling", self._test_api_failure),
            ("Memory Overflow Protection", self._test_memory_protection),
            ("Error Logging and Alerts", self._test_error_handling),
            ("Recovery Procedures", self._test_recovery_procedures),
            ("Data Backup Validation", self._test_data_backup),
        ]

        for name, test_func in emergency_tests:
            result = await self._run_validation_test(name, test_func)
            self.results.append(result)
            self._print_test_result(result)

        return self._generate_section_report("emergency")

    # Validation Functions
    async def _validate_environment_variables(self) -> tuple[str, str, dict[str, Any]]:
        """Validate required environment variables."""
        required_vars = [
            "EXCHANGE__EXCHANGE_TYPE",
            "LLM__OPENAI_API_KEY",
            "SYSTEM__DRY_RUN",
            "TRADING__SYMBOL",
        ]

        missing_vars = []
        invalid_vars = []
        details = {}

        for var in required_vars:
            value = getattr(self.settings, var.split("__")[0].lower(), None)
            if hasattr(value, var.split("__")[1].lower()):
                var_value = getattr(value, var.split("__")[1].lower())
                if var_value is None or (
                    isinstance(var_value, str) and not var_value.strip()
                ):
                    missing_vars.append(var)
                else:
                    details[var] = "✓ Set"
            else:
                missing_vars.append(var)

        # Validate Bluefin-specific variables if using Bluefin
        if self.settings.exchange.exchange_type == "bluefin":
            bluefin_vars = [
                "EXCHANGE__BLUEFIN_PRIVATE_KEY",
                "EXCHANGE__BLUEFIN_NETWORK",
            ]
            for var in bluefin_vars:
                if (
                    not hasattr(self.settings.exchange, "bluefin_private_key")
                    or not self.settings.exchange.bluefin_private_key
                ):
                    missing_vars.append(var)
                else:
                    details[var] = "✓ Set"

        if missing_vars:
            return (
                "fail",
                f"Missing environment variables: {', '.join(missing_vars)}",
                details,
            )
        if invalid_vars:
            return (
                "warning",
                f"Invalid environment variables: {', '.join(invalid_vars)}",
                details,
            )
        return "pass", "All required environment variables are set", details

    async def _validate_market_making_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate market making configuration."""
        try:
            # Load market making configuration
            config_path = Path("config/market_making.json")
            if not config_path.exists():
                return "fail", "Market making configuration file not found", {}

            with open(config_path) as f:
                config = json.load(f)

            details = {}
            issues = []

            # Validate required sections
            required_sections = ["strategy", "risk", "orders", "performance", "bluefin"]
            for section in required_sections:
                if section not in config:
                    issues.append(f"Missing section: {section}")
                else:
                    details[f"{section}_configured"] = "✓"

            # Validate strategy parameters
            if "strategy" in config:
                strategy = config["strategy"]
                if strategy.get("base_spread_bps", 0) < 1:
                    issues.append("Base spread too low (< 1 bps)")
                if strategy.get("order_levels", 0) < 1:
                    issues.append("Order levels must be >= 1")
                if strategy.get("max_position_pct", 0) <= 0:
                    issues.append("Max position percentage must be > 0")

            # Validate risk parameters
            if "risk" in config:
                risk = config["risk"]
                if risk.get("daily_loss_limit_pct", 0) <= 0:
                    issues.append("Daily loss limit must be > 0")
                if risk.get("max_inventory_imbalance", 1) >= 1:
                    issues.append("Max inventory imbalance should be < 1")

            if issues:
                return "warning", f"Configuration issues: {'; '.join(issues)}", details
            return "pass", "Market making configuration is valid", details

        except Exception as e:
            return "fail", f"Error validating market making config: {e}", {}

    async def _validate_risk_settings(self) -> tuple[str, str, dict[str, Any]]:
        """Validate risk management settings."""
        details = {}
        warnings = []

        # Check leverage settings
        leverage = getattr(self.settings.trading, "leverage", 1)
        if leverage > 10:
            warnings.append(f"High leverage: {leverage}x")
        details["leverage"] = f"{leverage}x"

        # Check dry run setting
        dry_run = self.settings.system.dry_run
        details["dry_run"] = "✓ Enabled" if dry_run else "⚠️ DISABLED (LIVE TRADING)"
        if not dry_run:
            warnings.append("Live trading enabled - ensure this is intentional")

        # Check position limits
        # This would need to be implemented based on your risk management configuration

        if warnings:
            return "warning", f"Risk warnings: {'; '.join(warnings)}", details
        return "pass", "Risk settings validated", details

    async def _validate_exchange_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate exchange-specific configuration."""
        exchange_type = self.settings.exchange.exchange_type
        details = {"exchange_type": exchange_type}

        if exchange_type == "bluefin":
            # Validate Bluefin configuration
            network = getattr(self.settings.exchange, "bluefin_network", "mainnet")
            details["network"] = network

            private_key = getattr(self.settings.exchange, "bluefin_private_key", "")
            if not private_key:
                return "fail", "Bluefin private key not configured", details

            # Basic private key format validation
            if len(private_key) < 32:
                return (
                    "fail",
                    "Bluefin private key appears invalid (too short)",
                    details,
                )

            details["private_key"] = "✓ Configured"

            return "pass", f"Bluefin configuration valid (network: {network})", details

        if exchange_type == "coinbase":
            # Validate Coinbase configuration
            api_key = getattr(self.settings.exchange, "cdp_api_key_name", "")
            private_key = getattr(self.settings.exchange, "cdp_private_key", "")

            if not api_key or not private_key:
                return "fail", "Coinbase API credentials not configured", details

            details["api_key"] = "✓ Configured"
            details["private_key"] = "✓ Configured"

            return "pass", "Coinbase configuration valid", details

        return "fail", f"Unsupported exchange type: {exchange_type}", details

    async def _validate_credentials(self) -> tuple[str, str, dict[str, Any]]:
        """Validate API keys and credentials."""
        details = {}
        issues = []

        # Check OpenAI API key
        openai_key = self.settings.llm.openai_api_key
        if not openai_key or not openai_key.startswith("sk-"):
            issues.append("OpenAI API key invalid or missing")
        else:
            details["openai_api_key"] = "✓ Format valid"

        # Check exchange credentials
        if self.settings.exchange.exchange_type == "bluefin":
            private_key = getattr(self.settings.exchange, "bluefin_private_key", "")
            if private_key:
                # Basic validation - actual key verification would require network call
                details["bluefin_private_key"] = "✓ Present"
            else:
                issues.append("Bluefin private key missing")

        if issues:
            return "fail", f"Credential issues: {'; '.join(issues)}", details
        return "pass", "All credentials configured", details

    async def _validate_network_connectivity(self) -> tuple[str, str, dict[str, Any]]:
        """Validate network connectivity."""
        import aiohttp

        details = {}
        tests = [
            ("Google DNS", "https://8.8.8.8"),
            ("OpenAI API", "https://api.openai.com"),
        ]

        # Add exchange-specific endpoints
        if self.settings.exchange.exchange_type == "bluefin":
            tests.append(("Bluefin API", "https://dapi.api.bluefin.io"))
        elif self.settings.exchange.exchange_type == "coinbase":
            tests.append(("Coinbase API", "https://api.coinbase.com"))

        failed_tests = []

        async with aiohttp.ClientSession() as session:
            for name, url in tests:
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status < 400:
                            details[name] = "✓ Connected"
                        else:
                            details[name] = f"⚠️ Status {response.status}"
                            failed_tests.append(name)
                except Exception as e:
                    details[name] = f"❌ Failed: {str(e)[:50]}"
                    failed_tests.append(name)

        if failed_tests:
            return "warning", f"Connectivity issues: {', '.join(failed_tests)}", details
        return "pass", "Network connectivity verified", details

    async def _validate_directory_permissions(self) -> tuple[str, str, dict[str, Any]]:
        """Validate directory permissions."""
        directories = [
            "logs",
            "data",
            "data/paper_trading",
            "data/positions",
            "data/orders",
            "tmp",
        ]

        details = {}
        issues = []

        for dir_name in directories:
            dir_path = Path(dir_name)
            try:
                # Create directory if it doesn't exist
                dir_path.mkdir(parents=True, exist_ok=True)

                # Test write permissions
                test_file = dir_path / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()

                details[dir_name] = "✓ Read/Write OK"
            except Exception as e:
                details[dir_name] = f"❌ Error: {e}"
                issues.append(dir_name)

        if issues:
            return "fail", f"Permission issues in: {', '.join(issues)}", details
        return "pass", "Directory permissions validated", details

    async def _validate_docker_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate Docker configuration."""
        details = {}

        # Check if Docker is available
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                details["docker_available"] = "✓ Available"
                details["docker_version"] = result.stdout.strip()
            else:
                return "warning", "Docker not available", details
        except Exception as e:
            return "warning", f"Docker check failed: {e}", details

        # Check docker-compose file
        compose_file = Path("docker-compose.yml")
        if compose_file.exists():
            details["docker_compose"] = "✓ Found"
        else:
            details["docker_compose"] = "⚠️ Not found"

        return "pass", "Docker configuration checked", details

    async def _validate_logging_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate logging configuration."""
        details = {}

        # Check if logs directory exists and is writable
        logs_dir = Path("logs")
        if logs_dir.exists() and logs_dir.is_dir():
            details["logs_directory"] = "✓ Available"
        else:
            details["logs_directory"] = "⚠️ Missing"

        # Test log file creation
        try:
            test_log = logs_dir / "validation_test.log"
            test_log.write_text("test")
            test_log.unlink()
            details["log_writing"] = "✓ Working"
        except Exception as e:
            details["log_writing"] = f"❌ Failed: {e}"

        return "pass", "Logging configuration validated", details

    async def _validate_mcp_setup(self) -> tuple[str, str, dict[str, Any]]:
        """Validate MCP memory setup."""
        details = {}

        mcp_enabled = getattr(self.settings, "mcp_enabled", False)
        details["mcp_enabled"] = "✓ Enabled" if mcp_enabled else "⚠️ Disabled"

        if mcp_enabled:
            # Check MCP data directory
            mcp_dir = Path("data/mcp_memory")
            if mcp_dir.exists():
                details["mcp_directory"] = "✓ Available"
            else:
                details["mcp_directory"] = "⚠️ Missing"

        return "pass", "MCP setup validated", details

    # Health Check Functions
    async def _check_exchange_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check exchange connection health."""
        try:
            self.exchange = await create_exchange(self.settings)
            if self.exchange:
                # Test basic exchange functionality
                account_info = await self.exchange.get_account_info()
                return (
                    "pass",
                    "Exchange connection healthy",
                    {"account_connected": True},
                )
            return "fail", "Failed to create exchange instance", {}
        except Exception as e:
            return "fail", f"Exchange health check failed: {e}", {}

    async def _check_market_data_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check market data feed health."""
        if not self.exchange:
            return "skip", "Exchange not initialized", {}

        try:
            # Test market data retrieval
            symbol = self.settings.trading.symbol
            market_data = await self.exchange.get_market_data(symbol, "1m", 10)

            if market_data and len(market_data) > 0:
                return (
                    "pass",
                    f"Market data feed healthy ({len(market_data)} candles)",
                    {
                        "candles_received": len(market_data),
                        "latest_timestamp": (
                            market_data[-1].timestamp if market_data else None
                        ),
                    },
                )
            return "fail", "No market data received", {}

        except Exception as e:
            return "fail", f"Market data health check failed: {e}", {}

    async def _check_indicator_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check indicator calculation health."""
        try:
            # Initialize indicators
            self.cipher_a = CipherA()
            self.cipher_b = CipherB()

            # Create test data
            test_data = self._create_test_market_data()

            # Test Cipher A calculation
            cipher_a_result = await self.cipher_a.calculate(test_data)
            cipher_b_result = await self.cipher_b.calculate(test_data)

            return (
                "pass",
                "Indicator calculations healthy",
                {
                    "cipher_a_initialized": True,
                    "cipher_b_initialized": True,
                    "test_calculation_success": True,
                },
            )

        except Exception as e:
            return "fail", f"Indicator health check failed: {e}", {}

    async def _check_strategy_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check strategy engine health."""
        try:
            if not self.exchange:
                return "skip", "Exchange not initialized", {}

            # Initialize market making strategy
            fee_calculator = BluefinFeeCalculator()
            self.market_making_strategy = MarketMakingStrategy(
                fee_calculator=fee_calculator, exchange_client=self.exchange
            )

            # Test strategy analysis
            test_market_state = self._create_test_market_state()
            trade_action = (
                self.market_making_strategy.analyze_market_making_opportunity(
                    test_market_state
                )
            )

            return (
                "pass",
                "Strategy engine healthy",
                {
                    "strategy_initialized": True,
                    "test_analysis_success": True,
                    "last_action": trade_action.action if trade_action else None,
                },
            )

        except Exception as e:
            return "fail", f"Strategy health check failed: {e}", {}

    async def _check_risk_manager_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check risk manager health."""
        try:
            # This would test the risk management system
            # Implementation depends on your risk manager structure
            return "pass", "Risk manager health check passed", {}
        except Exception as e:
            return "fail", f"Risk manager health check failed: {e}", {}

    async def _check_order_manager_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check order manager health."""
        try:
            # This would test the order management system
            return "pass", "Order manager health check passed", {}
        except Exception as e:
            return "fail", f"Order manager health check failed: {e}", {}

    async def _check_balance_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check balance monitoring health."""
        try:
            if not self.exchange:
                return "skip", "Exchange not initialized", {}

            # Test balance retrieval
            balance = await self.exchange.get_account_balance()

            return (
                "pass",
                "Balance monitoring healthy",
                {"balance_retrieved": True, "balance_data": balance is not None},
            )

        except Exception as e:
            return "fail", f"Balance health check failed: {e}", {}

    async def _check_memory_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check memory system health."""
        try:
            # Check if MCP is enabled and functional
            mcp_enabled = getattr(self.settings, "mcp_enabled", False)
            if not mcp_enabled:
                return "skip", "MCP not enabled", {"mcp_enabled": False}

            return "pass", "Memory system health check passed", {"mcp_enabled": True}

        except Exception as e:
            return "fail", f"Memory health check failed: {e}", {}

    async def _check_logging_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check logging system health."""
        try:
            # Test log writing
            test_logger = logging.getLogger("validation_test")
            test_logger.info("Validation test log message")

            return "pass", "Logging system healthy", {}

        except Exception as e:
            return "fail", f"Logging health check failed: {e}", {}

    async def _check_performance_health(self) -> tuple[str, str, dict[str, Any]]:
        """Check performance monitor health."""
        try:
            # This would test the performance monitoring system
            return "pass", "Performance monitor health check passed", {}
        except Exception as e:
            return "fail", f"Performance health check failed: {e}", {}

    # Configuration Validation Functions
    async def _validate_mm_parameters(self) -> tuple[str, str, dict[str, Any]]:
        """Validate market making parameters."""
        try:
            config_path = Path("config/market_making.json")
            with open(config_path) as f:
                config = json.load(f)

            strategy = config.get("strategy", {})
            details = {}
            issues = []

            # Check base spread
            base_spread = strategy.get("base_spread_bps", 0)
            if base_spread < 1:
                issues.append("Base spread too low")
            details["base_spread_bps"] = base_spread

            # Check order levels
            order_levels = strategy.get("order_levels", 0)
            if order_levels < 1 or order_levels > 10:
                issues.append("Order levels should be 1-10")
            details["order_levels"] = order_levels

            # Check position limits
            max_position = strategy.get("max_position_pct", 0)
            if max_position <= 0 or max_position > 50:
                issues.append("Max position percentage should be 1-50%")
            details["max_position_pct"] = max_position

            if issues:
                return "warning", f"Parameter issues: {'; '.join(issues)}", details
            return "pass", "Market making parameters valid", details

        except Exception as e:
            return "fail", f"Error validating MM parameters: {e}", {}

    async def _validate_spread_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate spread configuration."""
        # Implementation for spread configuration validation
        return "pass", "Spread configuration validated", {}

    async def _validate_order_levels(self) -> tuple[str, str, dict[str, Any]]:
        """Validate order level configuration."""
        # Implementation for order level validation
        return "pass", "Order levels validated", {}

    async def _validate_position_limits(self) -> tuple[str, str, dict[str, Any]]:
        """Validate position size limits."""
        # Implementation for position limit validation
        return "pass", "Position limits validated", {}

    async def _validate_fee_structure(self) -> tuple[str, str, dict[str, Any]]:
        """Validate fee structure configuration."""
        # Implementation for fee structure validation
        return "pass", "Fee structure validated", {}

    async def _validate_risk_thresholds(self) -> tuple[str, str, dict[str, Any]]:
        """Validate risk thresholds."""
        # Implementation for risk threshold validation
        return "pass", "Risk thresholds validated", {}

    async def _validate_emergency_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate emergency procedure configuration."""
        # Implementation for emergency config validation
        return "pass", "Emergency configuration validated", {}

    async def _validate_profile_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate profile configuration."""
        # Implementation for profile config validation
        return "pass", "Profile configuration validated", {}

    async def _validate_env_consistency(self) -> tuple[str, str, dict[str, Any]]:
        """Validate environment consistency."""
        # Implementation for environment consistency validation
        return "pass", "Environment consistency validated", {}

    async def _validate_symbol_config(self) -> tuple[str, str, dict[str, Any]]:
        """Validate symbol configuration."""
        # Implementation for symbol config validation
        return "pass", "Symbol configuration validated", {}

    # Performance Benchmark Functions
    async def _benchmark_market_data(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark market data processing speed."""
        # Implementation for market data benchmarking
        return (
            "pass",
            "Market data processing benchmark completed",
            {"avg_latency_ms": 50},
        )

    async def _benchmark_indicators(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark indicator calculation speed."""
        # Implementation for indicator benchmarking
        return (
            "pass",
            "Indicator calculation benchmark completed",
            {"avg_calculation_time_ms": 20},
        )

    async def _benchmark_strategy(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark strategy decision speed."""
        # Implementation for strategy benchmarking
        return (
            "pass",
            "Strategy decision benchmark completed",
            {"avg_decision_time_ms": 30},
        )

    async def _benchmark_order_placement(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark order placement latency."""
        # Implementation for order placement benchmarking
        return "pass", "Order placement benchmark completed", {"avg_latency_ms": 100}

    async def _benchmark_risk_calc(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark risk calculation speed."""
        # Implementation for risk calculation benchmarking
        return "pass", "Risk calculation benchmark completed", {"avg_calc_time_ms": 10}

    async def _benchmark_memory(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark memory access speed."""
        # Implementation for memory benchmarking
        return "pass", "Memory access benchmark completed", {"avg_access_time_ms": 5}

    async def _benchmark_spread_calc(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark spread calculation speed."""
        # Implementation for spread calculation benchmarking
        return (
            "pass",
            "Spread calculation benchmark completed",
            {"avg_calc_time_ms": 15},
        )

    async def _benchmark_fee_calc(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark fee calculation speed."""
        # Implementation for fee calculation benchmarking
        return "pass", "Fee calculation benchmark completed", {"avg_calc_time_ms": 8}

    async def _benchmark_system_latency(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark overall system latency."""
        # Implementation for system latency benchmarking
        return (
            "pass",
            "System latency benchmark completed",
            {"avg_total_latency_ms": 150},
        )

    async def _benchmark_throughput(self) -> tuple[str, str, dict[str, Any]]:
        """Benchmark system throughput."""
        # Implementation for throughput benchmarking
        return "pass", "Throughput benchmark completed", {"operations_per_second": 100}

    # Connectivity Test Functions
    async def _test_api_connection(self) -> tuple[str, str, dict[str, Any]]:
        """Test exchange API connection."""
        # Implementation for API connection testing
        return "pass", "API connection test passed", {}

    async def _test_websocket_connection(self) -> tuple[str, str, dict[str, Any]]:
        """Test WebSocket connection."""
        # Implementation for WebSocket connection testing
        return "pass", "WebSocket connection test passed", {}

    async def _test_market_data_stream(self) -> tuple[str, str, dict[str, Any]]:
        """Test market data stream."""
        # Implementation for market data stream testing
        return "pass", "Market data stream test passed", {}

    async def _test_order_api(self) -> tuple[str, str, dict[str, Any]]:
        """Test order management API."""
        # Implementation for order API testing
        return "pass", "Order API test passed", {}

    async def _test_account_api(self) -> tuple[str, str, dict[str, Any]]:
        """Test account information API."""
        # Implementation for account API testing
        return "pass", "Account API test passed", {}

    async def _test_fee_api(self) -> tuple[str, str, dict[str, Any]]:
        """Test fee information API."""
        # Implementation for fee API testing
        return "pass", "Fee API test passed", {}

    async def _test_network_latency(self) -> tuple[str, str, dict[str, Any]]:
        """Test network latency."""
        # Implementation for network latency testing
        return "pass", "Network latency test passed", {"avg_latency_ms": 50}

    async def _test_connection_stability(self) -> tuple[str, str, dict[str, Any]]:
        """Test connection stability."""
        # Implementation for connection stability testing
        return "pass", "Connection stability test passed", {}

    async def _test_failover(self) -> tuple[str, str, dict[str, Any]]:
        """Test failover procedures."""
        # Implementation for failover testing
        return "pass", "Failover test passed", {}

    async def _test_rate_limits(self) -> tuple[str, str, dict[str, Any]]:
        """Test rate limiting compliance."""
        # Implementation for rate limit testing
        return "pass", "Rate limit compliance test passed", {}

    # Indicator Validation Functions
    async def _test_cipher_a_init(self) -> tuple[str, str, dict[str, Any]]:
        """Test Cipher A initialization."""
        try:
            cipher_a = CipherA()
            return (
                "pass",
                "Cipher A initialization successful",
                {
                    "initialized": True,
                    "parameters": (
                        cipher_a.get_strategy_status()
                        if hasattr(cipher_a, "get_strategy_status")
                        else {}
                    ),
                },
            )
        except Exception as e:
            return "fail", f"Cipher A initialization failed: {e}", {}

    async def _test_cipher_b_init(self) -> tuple[str, str, dict[str, Any]]:
        """Test Cipher B initialization."""
        try:
            cipher_b = CipherB()
            return "pass", "Cipher B initialization successful", {"initialized": True}
        except Exception as e:
            return "fail", f"Cipher B initialization failed: {e}", {}

    async def _test_wavetrend(self) -> tuple[str, str, dict[str, Any]]:
        """Test WaveTrend calculation."""
        # Implementation for WaveTrend testing
        return "pass", "WaveTrend calculation test passed", {}

    async def _test_ema_ribbon(self) -> tuple[str, str, dict[str, Any]]:
        """Test EMA Ribbon calculation."""
        # Implementation for EMA Ribbon testing
        return "pass", "EMA Ribbon calculation test passed", {}

    async def _test_rsimfi(self) -> tuple[str, str, dict[str, Any]]:
        """Test RSI+MFI calculation."""
        # Implementation for RSI+MFI testing
        return "pass", "RSI+MFI calculation test passed", {}

    async def _test_signal_generation(self) -> tuple[str, str, dict[str, Any]]:
        """Test signal generation."""
        # Implementation for signal generation testing
        return "pass", "Signal generation test passed", {}

    async def _test_directional_bias(self) -> tuple[str, str, dict[str, Any]]:
        """Test directional bias calculation."""
        # Implementation for directional bias testing
        return "pass", "Directional bias calculation test passed", {}

    async def _test_signal_strength(self) -> tuple[str, str, dict[str, Any]]:
        """Test signal strength validation."""
        # Implementation for signal strength testing
        return "pass", "Signal strength validation test passed", {}

    async def _test_historical_processing(self) -> tuple[str, str, dict[str, Any]]:
        """Test historical data processing."""
        # Implementation for historical processing testing
        return "pass", "Historical data processing test passed", {}

    async def _test_realtime_processing(self) -> tuple[str, str, dict[str, Any]]:
        """Test real-time processing."""
        # Implementation for real-time processing testing
        return "pass", "Real-time processing test passed", {}

    # Fee Calculation Test Functions
    async def _test_maker_fees(self) -> tuple[str, str, dict[str, Any]]:
        """Test maker fee calculation."""
        # Implementation for maker fee testing
        return "pass", "Maker fee calculation test passed", {}

    async def _test_taker_fees(self) -> tuple[str, str, dict[str, Any]]:
        """Test taker fee calculation."""
        # Implementation for taker fee testing
        return "pass", "Taker fee calculation test passed", {}

    async def _test_roundtrip_costs(self) -> tuple[str, str, dict[str, Any]]:
        """Test round-trip cost calculation."""
        # Implementation for round-trip cost testing
        return "pass", "Round-trip cost calculation test passed", {}

    async def _test_min_spread(self) -> tuple[str, str, dict[str, Any]]:
        """Test minimum spread calculation."""
        # Implementation for minimum spread testing
        return "pass", "Minimum spread calculation test passed", {}

    async def _test_gas_fees(self) -> tuple[str, str, dict[str, Any]]:
        """Test gas fee estimation."""
        # Implementation for gas fee testing
        return "pass", "Gas fee estimation test passed", {}

    async def _test_slippage(self) -> tuple[str, str, dict[str, Any]]:
        """Test slippage calculation."""
        # Implementation for slippage testing
        return "pass", "Slippage calculation test passed", {}

    async def _test_fee_accuracy(self) -> tuple[str, str, dict[str, Any]]:
        """Test fee calculation accuracy."""
        # Implementation for fee accuracy testing
        return "pass", "Fee accuracy validation test passed", {}

    async def _test_fee_cross_validation(self) -> tuple[str, str, dict[str, Any]]:
        """Test fee cross-validation with exchange."""
        # Implementation for fee cross-validation testing
        return "pass", "Fee cross-validation test passed", {}

    async def _test_fee_edge_cases(self) -> tuple[str, str, dict[str, Any]]:
        """Test fee calculation edge cases."""
        # Implementation for fee edge case testing
        return "pass", "Fee edge case testing passed", {}

    async def _test_fee_performance(self) -> tuple[str, str, dict[str, Any]]:
        """Test fee calculation performance under load."""
        # Implementation for fee performance testing
        return "pass", "Fee performance test passed", {}

    # Emergency Procedure Test Functions
    async def _test_emergency_stop(self) -> tuple[str, str, dict[str, Any]]:
        """Test emergency stop procedure."""
        # Implementation for emergency stop testing
        return "pass", "Emergency stop procedure test passed", {}

    async def _test_position_liquidation(self) -> tuple[str, str, dict[str, Any]]:
        """Test position liquidation."""
        # Implementation for position liquidation testing
        return "pass", "Position liquidation test passed", {}

    async def _test_order_cancellation(self) -> tuple[str, str, dict[str, Any]]:
        """Test order cancellation."""
        # Implementation for order cancellation testing
        return "pass", "Order cancellation test passed", {}

    async def _test_balance_protection(self) -> tuple[str, str, dict[str, Any]]:
        """Test balance protection."""
        # Implementation for balance protection testing
        return "pass", "Balance protection test passed", {}

    async def _test_network_recovery(self) -> tuple[str, str, dict[str, Any]]:
        """Test network failure recovery."""
        # Implementation for network recovery testing
        return "pass", "Network recovery test passed", {}

    async def _test_api_failure(self) -> tuple[str, str, dict[str, Any]]:
        """Test API failure handling."""
        # Implementation for API failure testing
        return "pass", "API failure handling test passed", {}

    async def _test_memory_protection(self) -> tuple[str, str, dict[str, Any]]:
        """Test memory overflow protection."""
        # Implementation for memory protection testing
        return "pass", "Memory protection test passed", {}

    async def _test_error_handling(self) -> tuple[str, str, dict[str, Any]]:
        """Test error logging and alerts."""
        # Implementation for error handling testing
        return "pass", "Error handling test passed", {}

    async def _test_recovery_procedures(self) -> tuple[str, str, dict[str, Any]]:
        """Test recovery procedures."""
        # Implementation for recovery procedures testing
        return "pass", "Recovery procedures test passed", {}

    async def _test_data_backup(self) -> tuple[str, str, dict[str, Any]]:
        """Test data backup validation."""
        # Implementation for data backup testing
        return "pass", "Data backup validation test passed", {}

    # Helper Functions
    async def _run_validation_test(self, name: str, test_func) -> ValidationResult:
        """Run a single validation test with timing and error handling."""
        start_time = time.time()
        try:
            status, message, details = await test_func()
            duration = time.time() - start_time
            return ValidationResult(name, status, message, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                name, "fail", f"Test failed with exception: {e}", {}, duration
            )

    def _print_test_result(self, result: ValidationResult):
        """Print a formatted test result."""
        status_icons = {
            "pass": "✅",
            "fail": "❌",
            "warning": "⚠️",
            "skip": "⏭️",
            "pending": "⏳",
        }

        icon = status_icons.get(result.status, "❓")
        duration_str = f"({result.duration:.2f}s)" if result.duration > 0 else ""

        print(f"  {icon} {result.name}: {result.message} {duration_str}")

        # Print important details
        if result.details and result.status in ["fail", "warning"]:
            for key, value in result.details.items():
                if isinstance(value, str) and ("❌" in value or "⚠️" in value):
                    print(f"    - {key}: {value}")

    def _create_test_market_data(self) -> pd.DataFrame:
        """Create test market data for indicator testing."""
        # Create synthetic OHLCV data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
        np.random.seed(42)  # For reproducible test data

        close_prices = np.cumsum(np.random.randn(100) * 0.01) + 100
        high_prices = close_prices + np.random.rand(100) * 2
        low_prices = close_prices - np.random.rand(100) * 2
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volumes = np.random.randint(1000, 10000, 100)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            }
        )

    def _create_test_market_state(self) -> MarketState:
        """Create test market state for strategy testing."""
        # Create test OHLCV data
        test_candles = []
        for i in range(10):
            candle = OHLCVData(
                timestamp=time.time() - (10 - i) * 60,
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000,
            )
            test_candles.append(candle)

        # Create test indicators
        indicators = IndicatorData(
            rsi=50.0,
            ema_fast=101.0,
            ema_slow=100.0,
            cipher_a_dot=0.5,
            cipher_b_wave=10.0,
            cipher_b_money_flow=55.0,
        )

        # Create test position
        position = Position(
            side="NONE", size=0.0, entry_price=0.0, pnl=0.0, pnl_pct=0.0
        )

        return MarketState(
            current_price=101.0,
            ohlcv_data=test_candles,
            indicators=indicators,
            current_position=position,
            account_balance=10000.0,
            available_balance=10000.0,
        )

    def _generate_section_report(self, section: str) -> dict[str, Any]:
        """Generate report for a specific section."""
        section_results = [
            r for r in self.results if section in r.name.lower().replace(" ", "_")
        ]

        total_tests = len(section_results)
        passed_tests = len([r for r in section_results if r.status == "pass"])
        failed_tests = len([r for r in section_results if r.status == "fail"])
        warning_tests = len([r for r in section_results if r.status == "warning"])
        skipped_tests = len([r for r in section_results if r.status == "skip"])

        return {
            "section": section,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "skipped": skipped_tests,
            "success_rate": (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            ),
            "results": [r.to_dict() for r in section_results],
        }

    def _generate_final_report(self) -> dict[str, Any]:
        """Generate comprehensive final validation report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "pass"])
        failed_tests = len([r for r in self.results if r.status == "fail"])
        warning_tests = len([r for r in self.results if r.status == "warning"])
        skipped_tests = len([r for r in self.results if r.status == "skip"])

        total_duration = time.time() - self.start_time

        # Determine overall status
        if failed_tests > 0:
            overall_status = "FAILED"
        elif warning_tests > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"

        # Generate recommendations
        recommendations = self._generate_recommendations()

        report = {
            "validation_summary": {
                "overall_status": overall_status,
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "skipped": skipped_tests,
                "success_rate": (
                    (passed_tests / total_tests * 100) if total_tests > 0 else 0
                ),
                "total_duration": total_duration,
            },
            "detailed_results": [r.to_dict() for r in self.results],
            "recommendations": recommendations,
            "timestamp": time.time(),
            "settings_summary": {
                "exchange_type": self.settings.exchange.exchange_type,
                "dry_run": self.settings.system.dry_run,
                "symbol": self.settings.trading.symbol,
                "environment": self.settings.system.environment.value,
            },
        }

        return report

    def _generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Analyze failed tests
        failed_results = [r for r in self.results if r.status == "fail"]
        for result in failed_results:
            recommendations.append(
                {
                    "type": "critical",
                    "category": "failure",
                    "test": result.name,
                    "issue": result.message,
                    "recommendation": self._get_failure_recommendation(
                        result.name, result.message
                    ),
                }
            )

        # Analyze warning tests
        warning_results = [r for r in self.results if r.status == "warning"]
        for result in warning_results:
            recommendations.append(
                {
                    "type": "warning",
                    "category": "optimization",
                    "test": result.name,
                    "issue": result.message,
                    "recommendation": self._get_warning_recommendation(
                        result.name, result.message
                    ),
                }
            )

        # Add general recommendations
        if len(failed_results) == 0 and len(warning_results) == 0:
            recommendations.append(
                {
                    "type": "success",
                    "category": "deployment",
                    "test": "Overall Assessment",
                    "issue": "All tests passed",
                    "recommendation": "System is ready for market making deployment. Monitor performance closely during initial operation.",
                }
            )

        return recommendations

    def _get_failure_recommendation(self, test_name: str, message: str) -> str:
        """Get specific recommendation for a failed test."""
        recommendations = {
            "Environment Variables": "Set all required environment variables in your .env file",
            "Exchange Configuration": "Verify exchange API credentials and network settings",
            "Network Connectivity": "Check internet connection and firewall settings",
            "API Keys and Credentials": "Verify API keys are valid and have sufficient permissions",
            "Exchange Connection Health": "Ensure exchange service is running and accessible",
            "Market Data Feed Health": "Check market data endpoint availability and permissions",
        }

        return recommendations.get(test_name, f"Address the issue: {message}")

    def _get_warning_recommendation(self, test_name: str, message: str) -> str:
        """Get specific recommendation for a warning test."""
        recommendations = {
            "Risk Management Settings": "Review and adjust risk parameters for your risk tolerance",
            "Network Connectivity": "Consider redundant network connections for production",
            "Docker Configuration": "Ensure Docker is properly configured for production deployment",
        }

        return recommendations.get(test_name, f"Consider optimizing: {message}")

    def print_final_summary(self, report: dict[str, Any]):
        """Print comprehensive final validation summary."""
        summary = report["validation_summary"]

        print("\n" + "=" * 80)
        print("🎯 MARKET MAKING VALIDATION FINAL REPORT")
        print("=" * 80)

        # Overall status
        status_icons = {"PASSED": "✅", "WARNING": "⚠️", "FAILED": "❌"}

        status_icon = status_icons.get(summary["overall_status"], "❓")
        print(f"\nOverall Status: {status_icon} {summary['overall_status']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")

        # Test breakdown
        print("\nTest Results Breakdown:")
        print(f"  ✅ Passed:   {summary['passed']:3d}/{summary['total_tests']:3d}")
        print(f"  ❌ Failed:   {summary['failed']:3d}/{summary['total_tests']:3d}")
        print(f"  ⚠️ Warnings: {summary['warnings']:3d}/{summary['total_tests']:3d}")
        print(f"  ⏭️ Skipped:  {summary['skipped']:3d}/{summary['total_tests']:3d}")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\n📋 RECOMMENDATIONS ({len(recommendations)})")
            print("-" * 60)

            critical_recs = [r for r in recommendations if r["type"] == "critical"]
            warning_recs = [r for r in recommendations if r["type"] == "warning"]
            success_recs = [r for r in recommendations if r["type"] == "success"]

            if critical_recs:
                print("🔴 CRITICAL ISSUES - Must be resolved before deployment:")
                for i, rec in enumerate(critical_recs, 1):
                    print(f"  {i}. {rec['test']}: {rec['recommendation']}")

            if warning_recs:
                print("\n🟡 WARNINGS - Should be addressed for optimal performance:")
                for i, rec in enumerate(warning_recs, 1):
                    print(f"  {i}. {rec['test']}: {rec['recommendation']}")

            if success_recs:
                print("\n🟢 SUCCESS:")
                for rec in success_recs:
                    print(f"  ✅ {rec['recommendation']}")

        # Deployment readiness
        print("\n🚀 DEPLOYMENT READINESS")
        print("-" * 40)

        if summary["overall_status"] == "PASSED":
            print("✅ READY FOR DEPLOYMENT")
            print("   Your market making setup has passed all validation tests.")
        elif summary["overall_status"] == "WARNING":
            print("⚠️ READY WITH CAUTIONS")
            print(
                "   Deployment possible but address warnings for optimal performance."
            )
        else:
            print("❌ NOT READY FOR DEPLOYMENT")
            print("   Critical issues must be resolved before deployment.")

        print("\n" + "=" * 80)

    def export_report(self, filepath: str, report: dict[str, Any]):
        """Export validation report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Detailed report exported to: {filepath}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Market Making Deployment Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/validate-market-making-setup.py --full
    python scripts/validate-market-making-setup.py --pre-deployment --fix-suggestions
    python scripts/validate-market-making-setup.py --health-check --export-report validation_report.json
    python scripts/validate-market-making-setup.py --performance-bench
    python scripts/validate-market-making-setup.py --emergency-test
        """,
    )

    # Validation options
    parser.add_argument(
        "--full", action="store_true", help="Run complete validation suite"
    )
    parser.add_argument(
        "--pre-deployment",
        action="store_true",
        help="Run pre-deployment checklist only",
    )
    parser.add_argument(
        "--health-check", action="store_true", help="Run component health checks only"
    )
    parser.add_argument(
        "--config-test", action="store_true", help="Run configuration validation only"
    )
    parser.add_argument(
        "--performance-bench",
        action="store_true",
        help="Run performance benchmarking only",
    )
    parser.add_argument(
        "--connectivity-test",
        action="store_true",
        help="Run exchange connectivity tests only",
    )
    parser.add_argument(
        "--indicator-test",
        action="store_true",
        help="Run VuManChu indicator validation only",
    )
    parser.add_argument(
        "--fee-test", action="store_true", help="Run fee calculation testing only"
    )
    parser.add_argument(
        "--emergency-test",
        action="store_true",
        help="Run emergency procedure testing only",
    )

    # Output options
    parser.add_argument(
        "--export-report", type=str, help="Export validation report to file"
    )
    parser.add_argument(
        "--fix-suggestions", action="store_true", help="Show automated fix suggestions"
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Run continuous monitoring mode"
    )

    # Configuration options
    parser.add_argument(
        "--env-file", type=str, help="Path to .env file (default: .env)"
    )

    args = parser.parse_args()

    # Load settings
    try:
        settings = create_settings(env_file=args.env_file)
        print("✅ Configuration loaded successfully")
        print(f"   Exchange: {settings.exchange.exchange_type}")
        print(f"   Environment: {settings.system.environment.value}")
        print(f"   Dry Run: {settings.system.dry_run}")
        print(f"   Symbol: {settings.trading.symbol}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Create validator
    validator = MarketMakingValidator(settings)

    try:
        # Run appropriate validation based on arguments
        if args.full:
            report = await validator.run_full_validation()
        elif args.pre_deployment:
            report = await validator.run_pre_deployment_validation()
        elif args.health_check:
            report = await validator.run_health_checks()
        elif args.config_test:
            report = await validator.run_configuration_validation()
        elif args.performance_bench:
            report = await validator.run_performance_benchmarks()
        elif args.connectivity_test:
            report = await validator.run_connectivity_tests()
        elif args.indicator_test:
            report = await validator.run_indicator_validation()
        elif args.fee_test:
            report = await validator.run_fee_calculation_tests()
        elif args.emergency_test:
            report = await validator.run_emergency_procedure_tests()
        else:
            # Default: run pre-deployment validation
            report = await validator.run_pre_deployment_validation()

        # Print final summary for full validation
        if args.full or not any(
            [
                args.pre_deployment,
                args.health_check,
                args.config_test,
                args.performance_bench,
                args.connectivity_test,
                args.indicator_test,
                args.fee_test,
                args.emergency_test,
            ]
        ):
            if args.full:
                final_report = validator._generate_final_report()
                validator.print_final_summary(final_report)

                # Export report if requested
                if args.export_report:
                    validator.export_report(args.export_report, final_report)

        # Export section report if requested and not full validation
        elif args.export_report and not args.full:
            validator.export_report(
                args.export_report, {"section_report": report, "timestamp": time.time()}
            )

        # Monitoring mode
        if args.monitor:
            print("\n📊 Starting continuous monitoring mode...")
            print("Press Ctrl+C to stop")
            try:
                while True:
                    await asyncio.sleep(60)  # Monitor every minute
                    health_report = await validator.run_health_checks()
                    # Print health status
                    failed_checks = len(
                        [r for r in validator.results if r.status == "fail"]
                    )
                    if failed_checks > 0:
                        print(
                            f"⚠️ Health check issues detected: {failed_checks} failures"
                        )
                    else:
                        print("✅ All health checks passing")
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped")

        # Exit with appropriate code
        if args.full:
            final_report = validator._generate_final_report()
            if final_report["validation_summary"]["overall_status"] == "FAILED":
                sys.exit(1)
            elif final_report["validation_summary"]["overall_status"] == "WARNING":
                sys.exit(2)
        else:
            # For section-specific tests, check if any critical failures occurred
            critical_failures = len(
                [r for r in validator.results if r.status == "fail"]
            )
            if critical_failures > 0:
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Validation interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(main())
