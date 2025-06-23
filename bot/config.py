"""Configuration settings for the AI Trading Bot."""

import hashlib
import json
import logging
import os
import re
import secrets
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv

from bot.market_making_config import MarketMakingConfig
from bot.utils.path_utils import (
    ensure_directory_exists,
    get_config_directory,
    get_data_directory,
    get_data_file_path,
    get_logs_file_path,
)

# Try to import Bluefin endpoints, but don't fail if it's not available
if TYPE_CHECKING:
    from bot.exchange.bluefin_endpoints import BluefinEndpointConfig

try:
    from bot.exchange.bluefin_endpoints import BluefinEndpointConfig

    BLUEFIN_ENDPOINTS_AVAILABLE = True
except ImportError:
    BLUEFIN_ENDPOINTS_AVAILABLE = False

# Optional dependency for network testing
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

    # Create a mock aiohttp for type hints
    class MockAiohttp:
        class ClientSession:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def __aenter__(self) -> "MockAiohttp.ClientSession":
                return self

            async def __aexit__(self, *args: object) -> None:
                pass

            def get(self, *_args: object, **_kwargs: object) -> "MockResponse":
                return MockResponse()

            def post(self, *_args: object, **_kwargs: object) -> "MockResponse":
                return MockResponse()

        class ClientTimeout:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

    class MockResponse:
        def __init__(self) -> None:
            self.status = 200

        async def __aenter__(self) -> "MockResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def json(self) -> dict[str, object]:
            return {}

    # Assign the mock class instance to aiohttp for type compatibility
    aiohttp = MockAiohttp()  # type: ignore[assignment]


from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# Type definitions for validation results
class ValidationCheck(TypedDict):
    """Single validation check result."""

    name: str
    status: Literal["pass", "error", "warning", "skip"]
    message: str | None


class ValidationResult(TypedDict):
    """Result of a single validation test."""

    name: str
    status: Literal["pass", "error", "warning", "skip"]
    message: str


class ValidationResultWithChecks(TypedDict):
    """Validation result with sub-checks."""

    status: Literal["pass", "error", "warning"]
    checks: list[ValidationCheck]


class ValidationSummary(TypedDict):
    """Summary of all validation results."""

    total_errors: int
    total_warnings: int
    errors: list[str]
    warnings: list[str]
    is_valid: bool


class TestResult(TypedDict):
    """Result of a test."""

    name: str
    status: Literal["pass", "error", "warning"]
    warnings: list[str] | None


class TestValidationResult(TypedDict):
    """Validation result with tests."""

    status: Literal["pass", "skipped"]
    reason: str | None
    tests: list[TestResult] | None


class FullValidationResults(TypedDict):
    """Complete validation results."""

    environment: ValidationResultWithChecks
    network_connectivity: ValidationResultWithChecks
    exchange_config: ValidationResultWithChecks
    security: ValidationResultWithChecks
    trading_parameters: ValidationResultWithChecks
    llm_config: ValidationResultWithChecks
    sui_network: ValidationResultWithChecks | None
    summary: ValidationSummary


class ConfigSummary(TypedDict):
    """Configuration summary structure."""

    basic_info: dict[str, str | int | bool]
    security_status: dict[str, bool]
    risk_parameters: dict[str, float]
    network_config: dict[str, str | bool | dict[str, str] | None]
    warnings: list[str]
    config_hash: str


class BackupConfig(TypedDict):
    """Backup configuration structure."""

    metadata: dict[str, str]
    configuration: dict[str, object]


class HealthStatus(TypedDict):
    """Health status structure."""

    overall_status: str
    last_check: float
    config_hash: str
    issues: list[str]


class MonitoringData(TypedDict):
    """Monitoring data structure."""

    monitor_info: dict[str, str | float | int]
    current_config: ConfigSummary
    health_status: HealthStatus
    validation_cache: dict[str, object]


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


class ConfigurationValidator:
    """Comprehensive configuration validation and testing utilities."""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.validation_results: (
            FullValidationResults
            | dict[str, ValidationResultWithChecks | ValidationSummary | None]
        ) = {}

    def _check_aiohttp_available(self, test_name: str) -> ValidationResult | None:
        """Check if aiohttp is available for network testing."""
        if not AIOHTTP_AVAILABLE:
            return ValidationResult(
                name=test_name,
                status="skip",
                message="aiohttp not available for network testing",
            )
        return None

    async def validate_all(self) -> FullValidationResults:
        """Run all configuration validations."""
        results = {
            "environment": await self._validate_environment(),
            "network_connectivity": await self._validate_network_connectivity(),
            "exchange_config": await self._validate_exchange_configuration(),
            "security": await self._validate_security_settings(),
            "trading_parameters": await self._validate_trading_parameters(),
            "llm_config": await self._validate_llm_configuration(),
            "sui_network": (
                await self._validate_sui_network()
                if self.settings.exchange.exchange_type == "bluefin"
                else None
            ),
        }

        summary = ValidationSummary(
            total_errors=len(self.errors),
            total_warnings=len(self.warnings),
            errors=self.errors,
            warnings=self.warnings,
            is_valid=len(self.errors) == 0,
        )
        # Cast to the expected return type
        full_results = FullValidationResults(
            environment=results["environment"],
            network_connectivity=results["network_connectivity"],
            exchange_config=results["exchange_config"],
            security=results["security"],
            trading_parameters=results["trading_parameters"],
            llm_config=results["llm_config"],
            sui_network=results.get("sui_network"),
            summary=summary,
        )

        self.validation_results = full_results
        return full_results

    async def _validate_environment(self) -> ValidationResultWithChecks:
        """Validate environment configuration consistency."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        # Environment-network consistency
        if self.settings.exchange.exchange_type == "bluefin":
            env = self.settings.system.environment
            network = self.settings.exchange.bluefin_network
            dry_run = self.settings.system.dry_run

            if env.value == "production" and network == "testnet":
                self.warnings.append("Production environment using testnet network")
                results["checks"].append(
                    ValidationCheck(
                        name="prod_testnet_mismatch",
                        status="warning",
                        message="Production environment using testnet network",
                    )
                )

            if not dry_run and network == "testnet":
                self.warnings.append("Live trading enabled on testnet network")
                results["checks"].append(
                    ValidationCheck(
                        name="live_testnet_mismatch",
                        status="warning",
                        message="Live trading enabled on testnet network",
                    )
                )

            if env.value == "development" and network == "mainnet" and not dry_run:
                self.errors.append(
                    "Development environment should not use live mainnet trading"
                )
                results["status"] = "error"
                results["checks"].append(
                    ValidationCheck(
                        name="dev_mainnet_live",
                        status="error",
                        message="Development environment should not use live mainnet trading",
                    )
                )

        return results

    async def _validate_network_connectivity(self) -> ValidationResultWithChecks:
        """Test network connectivity to required services."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        # Test basic internet connectivity
        internet_check = await self._test_internet_connectivity()
        if internet_check["status"] == "error":
            results["status"] = "error"
        results["checks"].append(
            ValidationCheck(
                name=internet_check["name"],
                status=internet_check["status"],
                message=internet_check["message"],
            )
        )

        # Test DNS resolution
        dns_check = await self._test_dns_resolution()
        if dns_check["status"] == "error":
            results["status"] = "error"
        results["checks"].append(
            ValidationCheck(
                name=dns_check["name"],
                status=dns_check["status"],
                message=dns_check["message"],
            )
        )

        return results

    async def _validate_exchange_configuration(self) -> ValidationResultWithChecks:
        """Validate exchange-specific configuration."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        if self.settings.exchange.exchange_type == "bluefin":
            # Validate Bluefin service URL
            service_url_check = await self._test_bluefin_service_connectivity()
            if service_url_check["status"] == "error":
                results["status"] = "error"
            results["checks"].append(
                ValidationCheck(
                    name=service_url_check["name"],
                    status=service_url_check["status"],
                    message=service_url_check["message"],
                )
            )

            # Validate private key format comprehensively
            key_validation = self._validate_bluefin_private_key_format()
            if key_validation["status"] == "error":
                results["status"] = "error"
            results["checks"].append(
                ValidationCheck(
                    name=key_validation["name"],
                    status=key_validation["status"],
                    message=key_validation["message"],
                )
            )

            # Validate network endpoints
            endpoint_validation = await self._validate_bluefin_endpoints()
            if endpoint_validation["status"] == "error":
                results["status"] = "error"
            results["checks"].append(
                ValidationCheck(
                    name=endpoint_validation["name"],
                    status=endpoint_validation["status"],
                    message=endpoint_validation["message"],
                )
            )

        return results

    async def _validate_security_settings(self) -> ValidationResultWithChecks:
        """Validate security-related configuration."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        # Check for secure private key handling
        if (
            self.settings.exchange.exchange_type == "bluefin"
            and self.settings.exchange.bluefin_private_key
        ):
            key = self.settings.exchange.bluefin_private_key.get_secret_value()

            # Check if key appears to be exposed in logs or config
            if len(key) < 32:
                self.errors.append(
                    "Bluefin private key appears to be truncated or invalid"
                )
                results["status"] = "error"
                results["checks"].append(
                    ValidationCheck(
                        name="key_length",
                        status="error",
                        message="Key must be 32 bytes for security",
                    )
                )

            # Security recommendations
            if not self.settings.system.dry_run:
                self.warnings.append(
                    "Live trading enabled - ensure private keys are securely stored"
                )
                results["checks"].append(
                    ValidationCheck(
                        name="live_trading_security",
                        status="warning",
                        message="Live trading enabled - ensure secure key storage",
                    )
                )

        return results

    async def _validate_trading_parameters(self) -> ValidationResultWithChecks:
        """Validate trading parameter bounds and consistency."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        # Leverage validation
        if self.settings.trading.leverage > 20:
            self.warnings.append(
                f"High leverage ({self.settings.trading.leverage}x) detected"
            )
            results["checks"].append(
                ValidationCheck(
                    name="high_leverage",
                    status="warning",
                    message=f"High leverage ({self.settings.trading.leverage}x) increases risk",
                )
            )

        # Position size validation
        if self.settings.trading.max_size_pct > 50.0:
            self.warnings.append(
                f"High position size ({self.settings.trading.max_size_pct}%) detected"
            )
            results["checks"].append(
                ValidationCheck(
                    name="high_position_size",
                    status="warning",
                    message=f"High position size ({self.settings.trading.max_size_pct}%) increases risk",
                )
            )

        # Risk-reward ratio validation
        risk_reward = (
            self.settings.risk.default_take_profit_pct
            / self.settings.risk.default_stop_loss_pct
        )
        if risk_reward < 1.5:
            self.warnings.append(
                f"Low risk-reward ratio ({risk_reward:.2f}) may impact profitability"
            )
            results["checks"].append(
                ValidationCheck(
                    name="low_risk_reward",
                    status="warning",
                    message=f"Low risk-reward ratio ({risk_reward:.2f}) may impact profitability",
                )
            )

        return results

    async def _validate_llm_configuration(self) -> ValidationResultWithChecks:
        """Validate LLM configuration and test connectivity."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        # Test LLM API connectivity if key is provided
        if self.settings.llm.openai_api_key:
            api_test = await self._test_openai_api_connectivity()
            results["checks"].append(
                ValidationCheck(
                    name=api_test["name"],
                    status=api_test["status"],
                    message=api_test["message"],
                )
            )
            if api_test["status"] == "error":
                results["status"] = "error"

        # Validate temperature setting for trading
        if self.settings.llm.temperature > 0.3:
            self.warnings.append(
                f"High LLM temperature ({self.settings.llm.temperature}) may cause inconsistent decisions"
            )
            results["checks"].append(
                ValidationCheck(
                    name="high_temperature",
                    status="warning",
                    message=f"High LLM temperature ({self.settings.llm.temperature}) may cause inconsistent decisions",
                )
            )

        return results

    async def _validate_sui_network(self) -> ValidationResultWithChecks:
        """Validate Sui network connectivity and configuration."""
        results = ValidationResultWithChecks(status="pass", checks=[])

        # Test Sui RPC connectivity
        rpc_test = await self._test_sui_rpc_connectivity()
        if rpc_test["status"] == "error":
            results["status"] = "error"
        results["checks"].append(
            ValidationCheck(
                name=rpc_test["name"],
                status=rpc_test["status"],
                message=rpc_test["message"],
            )
        )

        # Test Bluefin API endpoints
        bluefin_api_test = await self._test_bluefin_api_connectivity()
        if bluefin_api_test["status"] == "error":
            results["status"] = "error"
        results["checks"].append(
            ValidationCheck(
                name=bluefin_api_test["name"],
                status=bluefin_api_test["status"],
                message=bluefin_api_test["message"],
            )
        )

        return results

    async def _test_internet_connectivity(self) -> ValidationResult:
        """Test basic internet connectivity."""
        skip_result = self._check_aiohttp_available("internet_connectivity")
        if skip_result:
            return skip_result

        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session,
                session.get("https://8.8.8.8") as response,
            ):
                if response.status == 200:
                    return {
                        "name": "internet_connectivity",
                        "status": "pass",
                        "message": "Internet connectivity OK",
                    }
        except Exception as e:
            self.errors.append(f"No internet connectivity: {e!s}")
            return {
                "name": "internet_connectivity",
                "status": "error",
                "message": f"No internet: {e!s}",
            }

        self.errors.append("Internet connectivity test failed")
        return {
            "name": "internet_connectivity",
            "status": "error",
            "message": "Internet test failed",
        }

    async def _test_dns_resolution(self) -> ValidationResult:
        """Test DNS resolution for key domains."""
        domains_to_test = ["api.openai.com"]

        if self.settings.exchange.exchange_type == "bluefin":
            network = self.settings.exchange.bluefin_network
            if network == "mainnet":
                domains_to_test.extend(
                    ["dapi.api.sui-prod.bluefin.io", "fullnode.mainnet.sui.io"]
                )
            else:
                domains_to_test.extend(
                    ["dapi.api.sui-staging.bluefin.io", "fullnode.testnet.sui.io"]
                )

        failed_domains = []
        for domain in domains_to_test:
            try:
                socket.gethostbyname(domain)
            except socket.gaierror:
                failed_domains.append(domain)

        if failed_domains:
            self.errors.append(
                f"DNS resolution failed for: {', '.join(failed_domains)}"
            )
            return {
                "name": "dns_resolution",
                "status": "error",
                "message": f"DNS failed: {failed_domains}",
            }

        return {
            "name": "dns_resolution",
            "status": "pass",
            "message": "DNS resolution OK",
        }

    async def _test_bluefin_service_connectivity(self) -> ValidationResult:
        """Test connectivity to Bluefin service."""
        service_url = self.settings.exchange.bluefin_service_url

        def _validate_url_format(url: str) -> None:
            """Validate URL format."""
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid service URL format")

        try:
            # Parse URL to check if it's valid
            _validate_url_format(service_url)

            # Test connectivity
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                health_url = f"{service_url.rstrip('/')}/health"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return {
                            "name": "bluefin_service",
                            "status": "pass",
                            "message": "Bluefin service reachable",
                        }
                    self.warnings.append(
                        f"Bluefin service returned status {response.status}"
                    )
                    return {
                        "name": "bluefin_service",
                        "status": "warning",
                        "message": f"Service status: {response.status}",
                    }
        except Exception as e:
            self.errors.append(f"Cannot reach Bluefin service at {service_url}: {e!s}")
            return {
                "name": "bluefin_service",
                "status": "error",
                "message": f"Service unreachable: {e!s}",
            }

    def _validate_bluefin_private_key_format(self) -> ValidationResult:
        """Comprehensively validate Bluefin private key format."""
        if not self.settings.exchange.bluefin_private_key:
            self.errors.append("Bluefin private key is required")
            return {
                "name": "private_key_format",
                "status": "error",
                "message": "Private key missing",
            }

        key = self.settings.exchange.bluefin_private_key.get_secret_value().strip()

        # Check for common formats
        if not key:
            self.errors.append("Bluefin private key is empty")
            return {
                "name": "private_key_format",
                "status": "error",
                "message": "Private key empty",
            }

        # Validate different supported formats
        formats_detected = []

        # 1. Mnemonic phrase (12 or 24 words)
        words = key.split()
        if len(words) in [12, 24] and all(
            word.isalpha() and len(word) > 2 for word in words
        ):
            formats_detected.append("mnemonic")

        # 2. Bech32-encoded Sui private key
        if key.startswith("suiprivkey") and len(key) > 20:
            formats_detected.append("sui_bech32")

        # 3. Hex format (with or without 0x prefix)
        hex_key = key.removeprefix("0x")
        if len(hex_key) == 64 and all(c in "0123456789abcdefABCDEF" for c in hex_key):
            formats_detected.append("hex")

        if not formats_detected:
            self.errors.append(
                "Bluefin private key format not recognized (expected: hex, mnemonic, or Sui bech32)"
            )
            return {
                "name": "private_key_format",
                "status": "error",
                "message": "Invalid key format",
            }

        return {
            "name": "private_key_format",
            "status": "pass",
            "message": f"Valid format: {', '.join(formats_detected)}",
        }

    async def _validate_bluefin_endpoints(self) -> ValidationResult:
        """Validate Bluefin API endpoints."""
        if not BLUEFIN_ENDPOINTS_AVAILABLE:
            return {
                "name": "bluefin_endpoints",
                "status": "skip",
                "message": "Bluefin endpoints module not available",
            }

        try:
            network = self.settings.exchange.bluefin_network
            endpoints = BluefinEndpointConfig.get_endpoints(network)

            # Test if endpoints are reachable
            test_url = f"{endpoints.rest_api}/ping"

            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session,
                session.get(test_url) as response,
            ):
                if response.status == 200:
                    return {
                        "name": "bluefin_endpoints",
                        "status": "pass",
                        "message": f"Endpoints reachable ({network})",
                    }
                self.warnings.append(
                    f"Bluefin {network} endpoint returned status {response.status}"
                )
                return {
                    "name": "bluefin_endpoints",
                    "status": "warning",
                    "message": f"Endpoint status: {response.status}",
                }
        except Exception as e:
            self.errors.append(f"Cannot validate Bluefin endpoints: {e!s}")
            return {
                "name": "bluefin_endpoints",
                "status": "error",
                "message": f"Endpoint test failed: {e!s}",
            }

    async def _test_openai_api_connectivity(self) -> ValidationResult:
        """Test OpenAI API connectivity."""
        if not self.settings.llm.openai_api_key:
            return {
                "name": "openai_api",
                "status": "skip",
                "message": "No API key provided",
            }

        try:
            headers = {
                "Authorization": f"Bearer {self.settings.llm.openai_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

            # Test with a minimal request to models endpoint
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as session,
                session.get(
                    "https://api.openai.com/v1/models", headers=headers
                ) as response,
            ):
                if response.status == 200:
                    return {
                        "name": "openai_api",
                        "status": "pass",
                        "message": "OpenAI API accessible",
                    }
                if response.status == 401:
                    self.errors.append("OpenAI API key is invalid")
                    return {
                        "name": "openai_api",
                        "status": "error",
                        "message": "Invalid API key",
                    }
                self.warnings.append(f"OpenAI API returned status {response.status}")
                return {
                    "name": "openai_api",
                    "status": "warning",
                    "message": f"API status: {response.status}",
                }
        except Exception as e:
            self.errors.append(f"Cannot reach OpenAI API: {e!s}")
            return {
                "name": "openai_api",
                "status": "error",
                "message": f"API unreachable: {e!s}",
            }

    async def _test_sui_rpc_connectivity(self) -> ValidationResult:
        """Test Sui RPC connectivity."""
        # Determine RPC URL based on network
        network = self.settings.exchange.bluefin_network

        if self.settings.exchange.bluefin_rpc_url:
            rpc_url = self.settings.exchange.bluefin_rpc_url
        elif network == "mainnet":
            rpc_url = "https://fullnode.mainnet.sui.io:443"
        else:
            rpc_url = "https://fullnode.testnet.sui.io:443"

        try:
            # Test with a simple RPC call
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sui_getLatestSuiSystemState",
                "params": [],
            }

            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as session,
                session.post(rpc_url, json=payload) as response,
            ):
                if response.status == 200:
                    data = await response.json()  # type: ignore[misc]
                    if "result" in data:  # type: ignore[operator]
                        return {
                            "name": "sui_rpc",
                            "status": "pass",
                            "message": f"Sui RPC accessible ({network})",
                        }
                    self.warnings.append(
                        f"Sui RPC returned unexpected response: {data}"  # type: ignore[str-format]
                    )
                    return {
                        "name": "sui_rpc",
                        "status": "warning",
                        "message": "Unexpected RPC response",
                    }
                self.errors.append(f"Sui RPC returned status {response.status}")
                return {
                    "name": "sui_rpc",
                    "status": "error",
                    "message": f"RPC status: {response.status}",
                }
        except Exception as e:
            self.errors.append(f"Cannot reach Sui RPC at {rpc_url}: {e!s}")
            return {
                "name": "sui_rpc",
                "status": "error",
                "message": f"RPC unreachable: {e!s}",
            }

    async def _test_bluefin_api_connectivity(self) -> ValidationResult:
        """Test Bluefin API connectivity."""
        if not BLUEFIN_ENDPOINTS_AVAILABLE:
            return {
                "name": "bluefin_api_connectivity",
                "status": "skip",
                "message": "Bluefin endpoints module not available",
            }

        try:
            network = self.settings.exchange.bluefin_network
            endpoints = BluefinEndpointConfig.get_endpoints(network)

            # Test ticker endpoint (public, no auth required)
            test_url = f"{endpoints.rest_api}/ticker24hr"

            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as session,
                session.get(test_url) as response,
            ):
                if response.status == 200:
                    return {
                        "name": "bluefin_api",
                        "status": "pass",
                        "message": f"Bluefin API accessible ({network})",
                    }
                if response.status == 429:
                    self.warnings.append("Bluefin API rate limited - this is normal")
                    return {
                        "name": "bluefin_api",
                        "status": "warning",
                        "message": "API rate limited",
                    }
                self.warnings.append(f"Bluefin API returned status {response.status}")
                return {
                    "name": "bluefin_api",
                    "status": "warning",
                    "message": f"API status: {response.status}",
                }
        except Exception as e:
            self.errors.append(f"Cannot reach Bluefin API: {e!s}")
            return {
                "name": "bluefin_api",
                "status": "error",
                "message": f"API unreachable: {e!s}",
            }

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."

        report = ["\n=== CONFIGURATION VALIDATION REPORT ==="]
        report.append(f"Generated at: {datetime.now(UTC).isoformat()}")
        report.append(f"Environment: {self.settings.system.environment.value}")
        report.append(f"Exchange: {self.settings.exchange.exchange_type}")

        if self.settings.exchange.exchange_type == "bluefin":
            report.append(f"Network: {self.settings.exchange.bluefin_network}")

        report.append(f"Dry Run: {self.settings.system.dry_run}")
        report.append("")

        # Summary
        summary = self.validation_results["summary"]
        report.append("=== SUMMARY ===")
        report.append(f"Status: {'✓ PASS' if summary['is_valid'] else '✗ FAIL'}")
        report.append(f"Errors: {summary['total_errors']}")
        report.append(f"Warnings: {summary['total_warnings']}")
        report.append("")

        # Detailed results
        for section, results in self.validation_results.items():
            if section == "summary" or results is None:
                continue

            report.append(f"=== {section.upper().replace('_', ' ')} ===")
            report.append(f"Status: {results['status'].upper()}")

            if "checks" in results:
                for check in results["checks"]:
                    status_symbol = {
                        "pass": "✓",
                        "warning": "⚠",
                        "error": "✗",
                        "skip": "-",
                    }[check["status"]]
                    report.append(
                        f"  {status_symbol} {check['name']}: {check['message']}"
                    )

            report.append("")

        # Errors and warnings
        if summary["errors"]:
            report.append("=== ERRORS ===")
            for error in summary["errors"]:
                report.append(f"  ✗ {error}")
            report.append("")

        if summary["warnings"]:
            report.append("=== WARNINGS ===")
            for warning in summary["warnings"]:
                report.append(f"  ⚠ {warning}")
            report.append("")

        return "\n".join(report)


class Environment(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingProfile(str, Enum):
    """Trading profile types for different risk levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class TradingSettings(BaseModel):
    """Trading-specific configuration."""

    model_config = ConfigDict(frozen=True)

    # Core Trading Parameters
    symbol: str = Field(
        default="BTC-USD",
        description="Trading symbol (e.g., BTC-USD, ETH-USD)",
        pattern=r"^[A-Z]+-[A-Z]+$",
    )
    interval: str = Field(
        default="1m",
        description="Candle interval for analysis. Sub-minute intervals (1s, 5s, 15s, 30s) require trade aggregation to be enabled in exchange settings.",
    )
    leverage: int = Field(
        default=5, ge=1, le=100, description="Trading leverage multiplier"
    )
    max_size_pct: float = Field(
        default=20.0,
        ge=0.1,
        le=50.0,
        description="Maximum position size as percentage of equity",
    )

    # Futures Trading Configuration
    enable_futures: bool = Field(
        default=True, description="Enable futures trading (vs spot trading)"
    )
    futures_account_type: Literal["CFM", "CBI"] = Field(
        default="CFM", description="Futures account type - CFM (futures) or CBI (spot)"
    )
    auto_cash_transfer: bool = Field(
        default=True,
        description="Automatically transfer cash from spot to futures for margin",
    )
    intraday_margin_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Intraday margin requirement multiplier",
    )
    overnight_margin_multiplier: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Overnight margin requirement multiplier",
    )
    max_futures_leverage: int = Field(
        default=20, ge=1, le=100, description="Maximum leverage for futures positions"
    )
    fixed_contract_size: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Fixed number of contracts to trade (e.g., 10 for 1 ETH in ETH-USD futures)",
    )

    # Position Tracking Configuration
    use_fifo_accounting: bool = Field(
        default=True,
        description="Use FIFO (First In First Out) accounting for position tracking",
    )

    # Order Configuration
    order_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Order timeout in seconds"
    )
    slippage_tolerance_pct: float = Field(
        default=0.1,
        ge=0.0,
        le=5.0,
        description="Maximum acceptable slippage percentage",
    )
    min_profit_pct: float = Field(
        default=0.5, ge=0.1, le=10.0, description="Minimum profit target percentage"
    )

    # Trading Fee Configuration
    # Spot trading fees - default to basic tier (< $10K volume)
    spot_maker_fee_rate: float = Field(
        default=0.006,  # 0.6% maker fee (basic tier)
        ge=0.0,
        le=0.02,
        description="Spot maker fee rate (for limit orders)",
    )
    spot_taker_fee_rate: float = Field(
        default=0.012,  # 1.2% taker fee (basic tier)
        ge=0.0,
        le=0.02,
        description="Spot taker fee rate (for market orders)",
    )
    # Legacy fee names for backward compatibility
    maker_fee_rate: float = Field(
        default=0.006,  # Default to spot maker fee
        ge=0.0,
        le=0.02,
        description="Maker fee rate (for limit orders)",
    )
    taker_fee_rate: float = Field(
        default=0.012,  # Default to spot taker fee
        ge=0.0,
        le=0.02,
        description="Taker fee rate (for market orders)",
    )
    # Futures trading fees
    futures_fee_rate: float = Field(
        default=0.0015,  # 0.15% for futures
        ge=0.0,
        le=0.01,
        description="Futures trading fee rate",
    )

    # Volume-based fee tiers for Coinbase (monthly USD volume)
    fee_tier_thresholds: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {"volume": 0, "maker": 0.006, "taker": 0.012},  # < $10K
            {"volume": 10000, "maker": 0.0025, "taker": 0.004},  # $10K+
            {"volume": 50000, "maker": 0.0015, "taker": 0.0025},  # $50K+
            {"volume": 100000, "maker": 0.001, "taker": 0.002},  # $100K+
            {"volume": 1000000, "maker": 0.0007, "taker": 0.0012},  # $1M+
            {"volume": 15000000, "maker": 0.0004, "taker": 0.0008},  # $15M+
            {"volume": 75000000, "maker": 0.0002, "taker": 0.0005},  # $75M+
            {"volume": 250000000, "maker": 0.0, "taker": 0.0005},  # $250M+
        ],
        description="Volume-based fee tiers for Coinbase spot trading",
    )

    # Trading Interval Configuration - Scalping Mode
    min_trading_interval_seconds: int = Field(
        default=15,  # 15 seconds minimum for high-frequency scalping
        ge=1,
        le=300,
        description="Minimum interval between trades in seconds (Note: Bluefin API constraints may affect actual execution timing)",
    )
    require_24h_data_before_trading: bool = Field(
        default=False,
        description="Require at least 24 hours of market data before first trade (disabled for 8h trading)",
    )
    min_candles_for_trading: int = Field(
        default=1,
        ge=1,
        le=2000,
        description="Minimum number of candles required before trading (1 = start immediately with historical data)",
    )

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate trading interval format.

        Note: Sub-minute intervals (1s, 5s, 15s, 30s) require trade aggregation
        to be enabled in exchange settings to function properly.
        """
        # Standard intervals supported by most exchanges
        standard_intervals = [
            "1m",  # Supported by all exchanges
            "3m",  # Supported by all exchanges
            "5m",  # Supported by all exchanges
            "15m",  # Supported by all exchanges
            "30m",
            "1h",
            "4h",
            "1d",
        ]

        # Sub-minute intervals that require trade aggregation
        sub_minute_intervals = [
            "1s",  # Requires trade aggregation
            "5s",  # Requires trade aggregation
            "15s",  # Requires trade aggregation
            "30s",  # Requires trade aggregation
        ]

        all_valid_intervals = standard_intervals + sub_minute_intervals

        if v not in all_valid_intervals:
            raise ValueError(
                f"Invalid interval '{v}'. Must be one of: {all_valid_intervals}\n"
                f"Note: Sub-minute intervals ({sub_minute_intervals}) require "
                f"trade aggregation to be enabled in exchange settings."
            )
        return v


class LLMSettings(BaseModel):
    """LLM and AI configuration."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    # Provider Configuration
    provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai", description="LLM provider to use"
    )
    model_name: str = Field(default="o3", description="Specific model name/version")
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature for response creativity",
    )
    max_tokens: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum tokens in LLM response (reduced from 30K to 5K for trading decisions performance)",
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for token repetition",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for topic repetition",
    )

    # API Configuration
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    openai_org_id: str | None = Field(
        default=None, description="OpenAI organization ID"
    )
    openai_base_url: AnyHttpUrl | None = Field(
        default=None, description="Custom OpenAI API base URL"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, description="Anthropic API key"
    )
    anthropic_base_url: AnyHttpUrl | None = Field(
        default=None, description="Custom Anthropic API base URL"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL for local models"
    )

    # Request Configuration
    request_timeout: int = Field(
        default=30, ge=5, le=120, description="API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum API request retries"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Delay between retries in seconds"
    )
    retry_exponential_base: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff base for retries"
    )

    # Response Caching
    enable_caching: bool = Field(
        default=True, description="Enable response caching for identical prompts"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        le=3600,
        description="Cache TTL in seconds (increased to 5 minutes for better hit rates)",
    )

    # Performance Optimization Configuration
    use_optimized_prompts: bool = Field(
        default=True,
        description="Use optimized prompt templates for ~50% performance improvement",
    )
    enable_api_call_parallelization: bool = Field(
        default=True, description="Enable parallel API calls for faster response times"
    )

    # LLM Logging Configuration
    enable_completion_logging: bool = Field(
        default=True, description="Enable detailed LLM completion logging"
    )
    completion_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level for LLM completions"
    )
    completion_log_file: str = Field(
        default="logs/llm_completions.log",
        description="Path to LLM completion log file",
    )
    log_prompt_preview_length: int = Field(
        default=500, ge=100, le=2000, description="Length of prompt preview in logs"
    )
    log_response_preview_length: int = Field(
        default=1000, ge=100, le=5000, description="Length of response preview in logs"
    )
    enable_performance_tracking: bool = Field(
        default=True, description="Enable LLM performance and cost tracking"
    )
    enable_langchain_callbacks: bool = Field(
        default=True,
        description="Enable LangChain callback handlers for detailed tracing",
    )
    log_market_context: bool = Field(
        default=True, description="Include market context in LLM completion logs"
    )
    enable_token_usage_tracking: bool = Field(
        default=True, description="Track and log token usage for cost analysis"
    )
    performance_log_interval: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Log performance metrics every N completions",
    )

    @field_validator("openai_api_key", "anthropic_api_key")
    @classmethod
    def validate_api_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate API key format."""
        if v is None:
            return v

        key = v.get_secret_value()
        if not key.strip():
            raise ValueError("API key cannot be empty")

        # Basic format validation
        if len(key) < 20:
            raise ValueError("API key seems too short")

        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str, info: object) -> str:
        """Validate model name based on provider."""
        provider = info.data.get("provider", "openai")

        if provider == "openai":
            valid_models = [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-3.5-turbo",
                "o3",
                "o3-mini",
            ]
            if not any(v.startswith(model) for model in valid_models):
                raise ValueError(f"Invalid OpenAI model: {v}")
        elif provider == "anthropic":
            valid_models = ["claude-3", "claude-2", "claude-instant"]
            if not any(v.startswith(model) for model in valid_models):
                raise ValueError(f"Invalid Anthropic model: {v}")

        return v


class MCPSettings(BaseModel):
    """MCP (Model Context Protocol) configuration for memory and learning."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False, description="Enable MCP memory server for learning"
    )
    server_url: str = Field(
        default="http://localhost:8765", description="MCP memory server URL"
    )
    memory_api_key: SecretStr | None = Field(
        default=None, description="API key for memory server (if required)"
    )

    # Memory Configuration
    max_memories_per_query: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum similar experiences to retrieve per query",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for memory retrieval",
    )
    memory_retention_days: int = Field(
        default=90, ge=7, le=365, description="Days to retain trading memories"
    )

    # Learning Configuration
    enable_pattern_learning: bool = Field(
        default=True, description="Enable pattern recognition and learning"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Learning rate for strategy adjustments",
    )
    min_samples_for_pattern: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Minimum samples needed to identify a pattern",
    )
    confidence_decay_rate: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Decay rate for pattern confidence over time",
    )

    # Experience Collection
    track_trade_lifecycle: bool = Field(
        default=True, description="Track complete trade lifecycle for learning"
    )
    store_market_snapshots: bool = Field(
        default=True, description="Store market snapshots at key decision points"
    )
    reflection_delay_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        description="Minutes to wait before analyzing trade outcome",
    )


class OmniSearchSettings(BaseModel):
    """OmniSearch MCP configuration for enhanced market intelligence and sentiment analysis."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description="Enable OmniSearch MCP server for market intelligence",
    )
    server_url: str = Field(
        default="http://localhost:8766", description="OmniSearch MCP server URL"
    )

    # API Keys for Different Providers
    # Search Provider API Keys
    tavily_api_key: SecretStr | None = Field(
        default=None, description="Tavily API key for premium search and news"
    )
    brave_api_key: SecretStr | None = Field(
        default=None, description="Brave Search API key for privacy-focused search"
    )
    kagi_api_key: SecretStr | None = Field(
        default=None, description="Kagi API key for premium search and AI services"
    )

    # AI Response Provider API Keys
    perplexity_api_key: SecretStr | None = Field(
        default=None,
        description="Perplexity API key for AI-powered search and reasoning",
    )

    # Content Processing API Keys
    jina_ai_api_key: SecretStr | None = Field(
        default=None, description="Jina AI API key for text processing and grounding"
    )
    firecrawl_api_key: SecretStr | None = Field(
        default=None,
        description="Firecrawl API key for web scraping and content extraction",
    )

    # Search Configuration
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to retrieve per query",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cache TTL for search results in seconds",
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Maximum requests per minute to prevent API throttling",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Request timeout in seconds",
    )

    # Feature Toggles
    enable_crypto_sentiment: bool = Field(
        default=True, description="Enable cryptocurrency sentiment analysis"
    )
    enable_nasdaq_sentiment: bool = Field(
        default=True, description="Enable NASDAQ/traditional market sentiment analysis"
    )
    enable_correlation_analysis: bool = Field(
        default=True, description="Enable cross-market correlation analysis"
    )

    # Legacy compatibility
    api_key: SecretStr | None = Field(
        default=None, description="Legacy API key field for backward compatibility"
    )

    @property
    @computed_field
    def has_any_api_key(self) -> bool:
        """Check if any API key is configured."""
        api_keys = [
            self.tavily_api_key,
            self.brave_api_key,
            self.kagi_api_key,
            self.perplexity_api_key,
            self.jina_ai_api_key,
            self.firecrawl_api_key,
            self.api_key,  # Legacy
        ]
        return any(key and key.get_secret_value().strip() for key in api_keys if key)

    @field_validator(
        "tavily_api_key",
        "brave_api_key",
        "kagi_api_key",
        "perplexity_api_key",
        "jina_ai_api_key",
        "firecrawl_api_key",
        "api_key",
    )
    @classmethod
    def validate_api_key_format(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate API key format."""
        if v is None:
            return v

        key = v.get_secret_value()
        if not key.strip():
            return None  # Allow empty keys

        # Basic format validation
        if len(key) < 10:
            raise ValueError("API key seems too short (minimum 10 characters)")

        return v


class ExchangeSettings(BaseModel):
    """Exchange API configuration."""

    model_config = ConfigDict(frozen=True)

    # Exchange Selection
    exchange_type: Literal["coinbase", "bluefin"] = Field(
        default="coinbase", description="Exchange to use for trading"
    )

    # Coinbase Configuration - Legacy Advanced Trade API
    cb_api_key: SecretStr | None = Field(
        default=None, description="Coinbase Advanced Trade API key (legacy)"
    )
    cb_api_secret: SecretStr | None = Field(
        default=None, description="Coinbase Advanced Trade API secret (legacy)"
    )
    cb_passphrase: SecretStr | None = Field(
        default=None, description="Coinbase Advanced Trade passphrase (legacy)"
    )

    # Coinbase Configuration - CDP API Keys
    cdp_api_key_name: SecretStr | None = Field(
        default=None,
        description="Coinbase CDP API key name (from organizations/.../apiKeys/...)",
    )
    cdp_private_key: SecretStr | None = Field(
        default=None, description="Coinbase CDP private key (PEM format)"
    )
    cb_sandbox: bool = Field(
        default=True, description="Use Coinbase sandbox environment"
    )
    cb_base_url: AnyHttpUrl | None = Field(
        default=None, description="Custom Coinbase API base URL"
    )

    # API Configuration
    api_timeout: int = Field(
        default=10, ge=1, le=60, description="API request timeout in seconds"
    )
    rate_limit_requests: int = Field(
        default=10, ge=1, le=100, description="Maximum requests per minute"
    )
    rate_limit_window_seconds: int = Field(
        default=60, ge=1, le=3600, description="Rate limit window in seconds"
    )
    websocket_reconnect_attempts: int = Field(
        default=5, ge=1, le=20, description="WebSocket reconnection attempts"
    )
    websocket_timeout: int = Field(
        default=30, ge=5, le=300, description="WebSocket connection timeout in seconds"
    )

    # Connection Health
    health_check_interval: int = Field(
        default=300, ge=60, le=3600, description="API health check interval in seconds"
    )

    # Bluefin Configuration
    bluefin_private_key: SecretStr | None = Field(
        default=None, description="Bluefin Sui wallet private key"
    )
    bluefin_network: Literal["mainnet", "testnet"] = Field(
        default="mainnet", description="Bluefin network to connect to"
    )
    bluefin_rpc_url: str | None = Field(
        default=None, description="Custom Sui RPC endpoint for Bluefin"
    )
    bluefin_service_url: str = Field(
        default="http://bluefin-service:8080",
        description="Bluefin microservice URL for SDK operations",
    )
    use_trade_aggregation: bool = Field(
        default=True,
        description="Enable trade-to-candle aggregation for sub-minute intervals (1s, 5s, 15s, 30s). "
        "Required for sub-minute trading intervals to work properly. "
        "When enabled, individual trades are aggregated into candles at the specified interval.",
    )

    @field_validator(
        "cb_api_key",
        "cb_api_secret",
        "cb_passphrase",
        "cdp_api_key_name",
        "cdp_private_key",
        "bluefin_private_key",
    )
    @classmethod
    def validate_api_credentials(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate API credential format."""
        if v is not None and len(v.get_secret_value().strip()) == 0:
            raise ValueError("API credentials cannot be empty strings")
        return v

    @field_validator("bluefin_network")
    @classmethod
    def validate_bluefin_network(cls, v: str) -> str:
        """Validate Bluefin network configuration."""
        valid_networks = {"mainnet", "testnet"}
        if v.lower() not in valid_networks:
            raise ValueError(
                f"Invalid Bluefin network '{v}'. Must be one of: {valid_networks}"
            )
        return v.lower()

    @model_validator(mode="after")
    def validate_exchange_credentials(self) -> "ExchangeSettings":
        """Validate exchange-specific credentials based on exchange type."""
        if self.exchange_type == "coinbase":
            # Validate Coinbase credentials only when using Coinbase
            has_legacy = all(
                [
                    self.cb_api_key and self.cb_api_key.get_secret_value().strip(),
                    self.cb_api_secret
                    and self.cb_api_secret.get_secret_value().strip(),
                    self.cb_passphrase
                    and self.cb_passphrase.get_secret_value().strip(),
                ]
            )

            has_cdp = all(
                [
                    self.cdp_api_key_name
                    and self.cdp_api_key_name.get_secret_value().strip(),
                    self.cdp_private_key
                    and self.cdp_private_key.get_secret_value().strip(),
                ]
            )

            # Allow neither (for dry run mode), but not both
            if has_legacy and has_cdp:
                raise ValueError(
                    "Cannot use both legacy and CDP credentials. Choose one method."
                )

            # Validate CDP private key format ONLY if provided AND we're using Coinbase
            if has_cdp and self.cdp_private_key:
                private_key = self.cdp_private_key.get_secret_value()
                if private_key and not private_key.startswith(
                    "-----BEGIN EC PRIVATE KEY-----"
                ):
                    raise ValueError(
                        "CDP private key must be in PEM format starting with '-----BEGIN EC PRIVATE KEY-----'"
                    )

        elif self.exchange_type == "bluefin":
            # Validate Bluefin credentials only when using Bluefin
            if (
                self.bluefin_private_key
                and self.bluefin_private_key.get_secret_value().strip()
            ):
                self._validate_bluefin_private_key_comprehensive()

        return self

    def _validate_bluefin_private_key_comprehensive(self) -> None:
        """Comprehensive validation of Bluefin private key formats."""
        if not self.bluefin_private_key:
            return

        private_key = self.bluefin_private_key.get_secret_value().strip()
        validation_errors = []
        detected_formats = []

        # Use helper methods to reduce complexity
        mnemonic_result = self._validate_mnemonic_key_format(private_key)
        if mnemonic_result:
            validation_errors.extend(mnemonic_result["errors"])
            detected_formats.extend(mnemonic_result["formats"])
        else:
            bech32_result = self._validate_bech32_key_format(private_key)
            if bech32_result:
                validation_errors.extend(bech32_result["errors"])
                detected_formats.extend(bech32_result["formats"])
            else:
                hex_result = self._validate_hex_key_format(private_key)
                validation_errors.extend(hex_result["errors"])
                detected_formats.extend(hex_result["formats"])

        self._handle_validation_results(validation_errors, detected_formats)

    def _validate_mnemonic_key_format(
        self, private_key: str
    ) -> dict[str, list[str]] | None:
        """Validate mnemonic key format."""
        words = private_key.split()
        if len(words) not in [12, 24]:
            return None

        errors = []
        formats = []

        if all(word.isalpha() and len(word) > 2 for word in words):
            formats.append("mnemonic")
            if not self._validate_mnemonic_checksum(words):
                errors.append("Mnemonic phrase appears to have invalid checksum")
        else:
            errors.append(
                "Mnemonic phrase contains invalid words (must be alphabetic, >2 chars)"
            )

        return {"errors": errors, "formats": formats}

    def _validate_bech32_key_format(
        self, private_key: str
    ) -> dict[str, list[str]] | None:
        """Validate Bech32 key format."""
        if not private_key.startswith("suiprivkey"):
            return None

        errors = []
        formats = []

        if len(private_key) < 50:
            errors.append("Sui Bech32 private key appears too short")
        else:
            formats.append("sui_bech32")
            if not self._validate_bech32_format(private_key):
                errors.append("Sui Bech32 private key has invalid format")

        return {"errors": errors, "formats": formats}

    def _validate_hex_key_format(self, private_key: str) -> dict[str, list[str]]:
        """Validate hex key format."""
        errors = []
        formats = []
        hex_key = private_key.removeprefix("0x")

        if len(hex_key) == 64:
            if all(c in "0123456789abcdefABCDEF" for c in hex_key):
                formats.append("hex")
                if hex_key == "0" * 64:
                    errors.append("Private key cannot be all zeros")
                elif hex_key.upper() == "F" * 64:
                    errors.append("Private key cannot be all F's")
            else:
                errors.append("Hex private key contains invalid characters")
        else:
            errors.append(
                f"Hex private key must be exactly 64 characters, got {len(hex_key)}"
            )

        return {"errors": errors, "formats": formats}

    def _handle_validation_results(
        self, validation_errors: list[str], detected_formats: list[str]
    ) -> None:
        """Handle validation results and raise errors if needed."""
        if not detected_formats:
            validation_errors.append(
                "Private key format not recognized. Expected: 64-char hex (with/without 0x), "
                "Sui Bech32 (suiprivkey...), or 12/24-word mnemonic phrase"
            )

        if validation_errors:
            error_msg = "Bluefin private key validation failed: " + "; ".join(
                validation_errors
            )
            if detected_formats:
                error_msg += f" (detected formats: {', '.join(detected_formats)})"
            raise ValueError(error_msg)

    def _validate_mnemonic_checksum(self, words: list[str]) -> bool:
        """Basic mnemonic phrase validation (simplified BIP39 check)."""
        # This is a simplified check - in production, you'd use a proper BIP39 library
        # For now, just check word count and basic structure
        if len(words) not in [12, 24]:
            return False

        # Check for common invalid patterns
        if len(set(words)) < len(words) * 0.8:  # Too many repeated words
            return False

        # Check for reasonable word lengths
        return not any(len(word) < 3 or len(word) > 12 for word in words)

    def _validate_bech32_format(self, key: str) -> bool:
        """Basic Bech32 format validation for Sui private keys."""
        if not key.startswith("suiprivkey"):
            return False

        # Check length (Bech32 encoded keys should be reasonably long)
        if len(key) < 50 or len(key) > 120:
            return False

        # Check character set (Bech32 uses specific characters)
        bech32_chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
        key_part = key[10:]  # Remove "suiprivkey" prefix

        # Allow uppercase and lowercase, but check valid Bech32 chars
        return all(c.lower() in bech32_chars for c in key_part)

    def validate_network_endpoints(self) -> list[str]:
        """Validate network endpoint configuration and accessibility."""
        validation_issues = []

        if self.exchange_type == "bluefin":
            # Validate network setting
            if self.bluefin_network not in ["mainnet", "testnet"]:
                validation_issues.append(
                    f"Invalid Bluefin network: {self.bluefin_network}"
                )

            # Validate custom RPC URL if provided
            if self.bluefin_rpc_url:
                if not self._validate_url_format(self.bluefin_rpc_url):
                    validation_issues.append(
                        f"Invalid Bluefin RPC URL format: {self.bluefin_rpc_url}"
                    )

                # Check if URL matches network
                if (
                    self.bluefin_network == "mainnet"
                    and "testnet" in self.bluefin_rpc_url.lower()
                ):
                    validation_issues.append(
                        "Mainnet network configured with testnet RPC URL"
                    )
                elif (
                    self.bluefin_network == "testnet"
                    and "mainnet" in self.bluefin_rpc_url.lower()
                ):
                    validation_issues.append(
                        "Testnet network configured with mainnet RPC URL"
                    )

            # Validate service URL
            if not self._validate_url_format(self.bluefin_service_url):
                validation_issues.append(
                    f"Invalid Bluefin service URL format: {self.bluefin_service_url}"
                )

        return validation_issues

    def _validate_url_format(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def get_effective_endpoints(self) -> dict[str, str]:
        """Get effective endpoints for the current configuration."""
        endpoints = {}

        if self.exchange_type == "bluefin":
            try:
                # Try to import and use the endpoint config
                if BLUEFIN_ENDPOINTS_AVAILABLE:
                    bluefin_endpoints = BluefinEndpointConfig.get_endpoints(
                        self.bluefin_network
                    )
                endpoints.update(
                    {
                        "rest_api": bluefin_endpoints.rest_api,
                        "websocket_api": bluefin_endpoints.websocket_api,
                        "websocket_notifications": bluefin_endpoints.websocket_notifications,
                        "service_url": self.bluefin_service_url,
                    }
                )

                # Add custom RPC if specified
                if self.bluefin_rpc_url:
                    endpoints["custom_rpc"] = self.bluefin_rpc_url
                elif self.bluefin_network == "mainnet":
                    endpoints["default_rpc"] = "https://fullnode.mainnet.sui.io:443"
                else:
                    endpoints["default_rpc"] = "https://fullnode.testnet.sui.io:443"

            except ImportError:
                # Fallback if endpoint config is not available
                if self.bluefin_network == "mainnet":
                    endpoints.update(
                        {
                            "rest_api": "https://dapi.api.sui-prod.bluefin.io",
                            "websocket_api": "wss://dapi.api.sui-prod.bluefin.io",
                            "default_rpc": "https://fullnode.mainnet.sui.io:443",
                        }
                    )
                else:
                    endpoints.update(
                        {
                            "rest_api": "https://dapi.api.sui-staging.bluefin.io",
                            "websocket_api": "wss://dapi.api.sui-staging.bluefin.io",
                            "default_rpc": "https://fullnode.testnet.sui.io:443",
                        }
                    )

                endpoints["service_url"] = self.bluefin_service_url

        return endpoints


class RiskSettings(BaseModel):
    """Risk management configuration."""

    model_config = ConfigDict(frozen=True)

    # Loss Limits
    max_daily_loss_pct: float = Field(
        default=5.0, ge=0.1, le=50.0, description="Maximum daily loss percentage"
    )
    max_weekly_loss_pct: float = Field(
        default=15.0, ge=0.1, le=75.0, description="Maximum weekly loss percentage"
    )
    max_monthly_loss_pct: float = Field(
        default=30.0, ge=0.1, le=90.0, description="Maximum monthly loss percentage"
    )

    # Position Limits
    max_concurrent_trades: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Maximum concurrent positions (default: 1 for single position rule)",
    )
    max_position_hold_hours: int = Field(
        default=1,
        ge=1,
        le=24,  # 1 day max for scalping
        description="Maximum hours to hold a position (short for scalping)",
    )

    # Stop Loss and Take Profit - Scalping Mode
    default_stop_loss_pct: float = Field(
        default=0.3,
        ge=0.05,
        le=2.0,
        description="Default stop loss percentage (tight for scalping)",
    )
    default_take_profit_pct: float = Field(
        default=0.5,
        ge=0.1,
        le=3.0,
        description="Default take profit percentage (quick scalping targets)",
    )

    # Account Protection
    min_account_balance: Decimal = Field(
        default=Decimal(100),
        ge=Decimal(10),
        description="Minimum account balance to continue trading",
    )
    emergency_stop_loss_pct: float = Field(
        default=10.0, ge=1.0, le=25.0, description="Emergency stop loss percentage"
    )


class DataSettings(BaseModel):
    """Data and indicator configuration."""

    model_config = ConfigDict(frozen=True)

    # Data Fetching - Scalping Mode
    candle_limit: int = Field(
        default=1000,
        ge=200,
        le=5000,
        description="Number of historical candles to fetch (1000 = ~16.7 hours for 1m intervals, optimal for indicators)",
    )
    real_time_updates: bool = Field(
        default=True, description="Enable real-time data updates"
    )
    data_cache_ttl_seconds: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Data cache TTL in seconds (fast refresh for scalping)",
    )

    # Indicator Configuration
    indicator_warmup: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Indicator warmup period (minimum 100 for VuManChu indicators)",
    )

    # VuManChu Cipher Settings - Scalping Mode (Faster Periods)
    cipher_a_ema_length: int = Field(
        default=7, ge=3, le=21, description="Cipher A EMA length (faster for scalping)"
    )
    cipher_b_vwap_length: int = Field(
        default=10,
        ge=3,
        le=20,
        description="Cipher B VWAP length (faster for scalping)",
    )

    # Cipher B Signal Filter Configuration
    enable_cipher_b_filter: bool = Field(
        default=True,
        description="Enable Cipher B signal filtering for trade validation",
    )
    cipher_b_wave_bullish_threshold: float = Field(
        default=0.0, description="Cipher B wave threshold for bullish signals"
    )
    cipher_b_wave_bearish_threshold: float = Field(
        default=0.0, description="Cipher B wave threshold for bearish signals"
    )
    cipher_b_money_flow_bullish_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Cipher B money flow threshold for bullish signals",
    )
    cipher_b_money_flow_bearish_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Cipher B money flow threshold for bearish signals",
    )

    # Storage Configuration
    data_storage_path: Path = Field(
        default_factory=get_data_directory, description="Path for data storage"
    )
    keep_historical_days: int = Field(
        default=30, ge=1, le=365, description="Days of historical data to keep"
    )


class DominanceSettings(BaseModel):
    """Stablecoin dominance data configuration."""

    model_config = ConfigDict(frozen=True)

    # Feature Toggle
    enable_dominance_data: bool = Field(
        default=True, description="Enable stablecoin dominance data integration"
    )

    # Data Source Configuration
    data_source: Literal["coingecko", "coinmarketcap", "custom"] = Field(
        default="coingecko", description="Dominance data source"
    )
    api_key: SecretStr | None = Field(
        default=None, description="API key for premium data sources"
    )

    # Update Configuration
    update_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Dominance data update interval in seconds",
    )
    cache_ttl: int = Field(
        default=300, ge=30, le=3600, description="Dominance data cache TTL in seconds"
    )

    # Analysis Configuration
    dominance_weight_in_decisions: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight of dominance data in trading decisions (0-1)",
    )

    # Alert Thresholds
    dominance_alert_threshold: float = Field(
        default=12.0,
        ge=5.0,
        le=20.0,
        description="Stablecoin dominance percentage to trigger alerts",
    )
    dominance_change_alert_threshold: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Dominance change percentage (24h) to trigger alerts",
    )


class PaperTradingSettings(BaseModel):
    """Paper trading configuration."""

    model_config = ConfigDict(frozen=True)

    # Account Configuration
    starting_balance: Decimal = Field(
        default=Decimal(10000),
        ge=Decimal(100),
        description="Starting balance for paper trading",
    )
    fee_rate: float = Field(
        default=0.001, ge=0.0, le=0.01, description="Trading fee rate (0.1% default)"
    )
    slippage_rate: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.005,
        description="Slippage simulation rate (0.05% default)",
    )

    # Performance Tracking
    enable_daily_reports: bool = Field(
        default=True, description="Enable daily performance reports"
    )
    enable_weekly_summaries: bool = Field(
        default=True, description="Enable weekly performance summaries"
    )
    track_drawdown: bool = Field(default=True, description="Track maximum drawdown")

    # Data Retention
    keep_trade_history_days: int = Field(
        default=90, ge=7, le=365, description="Days to keep trade history"
    )
    export_trade_data: bool = Field(
        default=False, description="Export trade data for external analysis"
    )

    # Reporting Configuration
    report_time_utc: str = Field(
        default="23:59", description="UTC time for daily reports (HH:MM)"
    )
    include_unrealized_pnl: bool = Field(
        default=True, description="Include unrealized P&L in reports"
    )


class MonitoringSettings(BaseModel):
    """Balance monitoring and alerting configuration."""

    model_config = ConfigDict(frozen=True)

    # Core Monitoring Configuration
    enabled: bool = Field(
        default=True,
        description="Enable comprehensive balance monitoring and metrics collection",
    )

    # Metrics Collection Settings
    metrics_retention_limit: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum number of metrics to retain in memory",
    )
    timing_window_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Time window for timing statistics in minutes",
    )
    enable_prometheus_metrics: bool = Field(
        default=True, description="Enable Prometheus-compatible metrics exposition"
    )

    # Metrics Collection Performance
    collection_overhead_limit_ms: float = Field(
        default=5.0,
        ge=1.0,
        le=50.0,
        description="Maximum allowed overhead per monitoring operation in milliseconds",
    )

    # Alerting Configuration
    enable_alerting: bool = Field(
        default=True, description="Enable balance operation alerting system"
    )
    alert_history_limit: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum number of alerts to keep in history",
    )
    alert_config_file: str = Field(
        default_factory=lambda: str(get_data_file_path("alerts/config.json")),
        description="Path to alert configuration file",
    )

    # Default Alert Thresholds
    error_rate_warning_threshold: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Error rate percentage threshold for warnings",
    )
    error_rate_critical_threshold: float = Field(
        default=25.0,
        ge=10.0,
        le=75.0,
        description="Error rate percentage threshold for critical alerts",
    )
    response_time_warning_threshold_ms: float = Field(
        default=5000.0,
        ge=1000.0,
        le=30000.0,
        description="Response time threshold in milliseconds for warnings",
    )
    response_time_critical_threshold_ms: float = Field(
        default=10000.0,
        ge=5000.0,
        le=60000.0,
        description="Response time threshold in milliseconds for critical alerts",
    )
    consecutive_errors_warning_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of consecutive errors before warning alert",
    )
    consecutive_errors_critical_threshold: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Number of consecutive errors before critical alert",
    )

    # Balance Change Alert Thresholds
    large_balance_change_threshold: float = Field(
        default=1000.0,
        ge=100.0,
        le=100000.0,
        description="Balance change amount in USD for large change alerts",
    )
    huge_balance_change_threshold: float = Field(
        default=5000.0,
        ge=1000.0,
        le=500000.0,
        description="Balance change amount in USD for huge change alerts",
    )

    # Notification Settings
    enable_email_notifications: bool = Field(
        default=False, description="Enable email notifications for alerts"
    )
    email_smtp_server: str = Field(
        default="", description="SMTP server for email notifications"
    )
    email_smtp_port: int = Field(
        default=587, ge=1, le=65535, description="SMTP port for email notifications"
    )
    email_username: str = Field(
        default="", description="SMTP username for email notifications"
    )
    email_password: SecretStr = Field(
        default=SecretStr(""), description="SMTP password for email notifications"
    )
    email_from_address: str = Field(
        default="", description="From email address for notifications"
    )
    email_to_addresses: list[str] = Field(
        default_factory=list,
        description="List of email addresses to send notifications to",
    )

    # Webhook Notifications
    enable_webhook_notifications: bool = Field(
        default=False, description="Enable webhook notifications for alerts"
    )
    webhook_url: str = Field(
        default="", description="Webhook URL for alert notifications"
    )
    webhook_timeout_seconds: int = Field(
        default=10, ge=1, le=60, description="Timeout for webhook requests in seconds"
    )

    # File Logging
    enable_file_logging: bool = Field(
        default=True, description="Enable file logging for alerts"
    )
    alert_log_file: str = Field(
        default_factory=lambda: str(get_data_file_path("alerts/alerts.log")),
        description="Path to alert log file",
    )

    # Background Tasks
    enable_background_monitoring: bool = Field(
        default=True, description="Enable background monitoring and cleanup tasks"
    )
    monitoring_interval_seconds: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Interval for background monitoring evaluation in seconds",
    )
    cleanup_interval_minutes: int = Field(
        default=30, ge=5, le=240, description="Interval for cleanup tasks in minutes"
    )

    @field_validator("email_to_addresses")
    @classmethod
    def validate_email_addresses(cls, v: list[str]) -> list[str]:
        """Validate email addresses format."""
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        for email in v:
            if email and not email_pattern.match(email):
                raise ValueError(f"Invalid email address format: {email}")

        return v

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        """Validate webhook URL format."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v


class SystemSettings(BaseModel):
    """System and operational configuration."""

    model_config = ConfigDict(frozen=True)

    # Execution Mode
    dry_run: bool = Field(
        default=True,
        description="Single control for paper trading/dry run mode - when True, no real trades are executed",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    instance_id: str = Field(
        default_factory=lambda: secrets.token_hex(8),
        description="Unique instance identifier",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_file_path: Path | None = Field(
        default_factory=lambda: get_logs_file_path("bot.log"),
        description="Log file path",
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    log_to_console: bool = Field(default=True, description="Enable console logging")
    log_to_file: bool = Field(default=True, description="Enable file logging")
    max_log_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Maximum log file size in MB"
    )
    log_backup_count: int = Field(
        default=5, ge=1, le=50, description="Number of log backup files to keep"
    )
    log_retention_days: int = Field(
        default=30, ge=1, le=365, description="Log retention period in days"
    )

    # Performance - Scalping Mode
    update_frequency_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Main loop update frequency (1s for high-frequency scalping)",
    )
    parallel_processing: bool = Field(
        default=True, description="Enable parallel processing where possible"
    )
    max_worker_threads: int = Field(
        default=4, ge=1, le=20, description="Maximum worker threads"
    )
    memory_limit_mb: int | None = Field(
        default=None, ge=100, le=8192, description="Memory usage limit in MB"
    )

    # Monitoring and Alerts
    enable_monitoring: bool = Field(
        default=True, description="Enable system monitoring"
    )
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_export_interval: int = Field(
        default=60, ge=10, le=3600, description="Metrics export interval in seconds"
    )
    alert_webhook_url: AnyHttpUrl | None = Field(
        default=None, description="Webhook URL for alerts"
    )
    alert_email: str | None = Field(
        default=None, description="Email address for alerts"
    )
    health_check_interval: int = Field(
        default=300, ge=60, le=3600, description="Health check interval in seconds"
    )

    # Graceful Shutdown
    shutdown_timeout: int = Field(
        default=30, ge=5, le=300, description="Graceful shutdown timeout in seconds"
    )

    # Security
    enable_api_auth: bool = Field(default=True, description="Enable API authentication")
    api_secret_key: SecretStr | None = Field(
        default=None, description="Secret key for API authentication"
    )

    # WebSocket Publishing for Real-time Dashboard Integration
    enable_websocket_publishing: bool = Field(
        default=False, description="Enable real-time WebSocket publishing to dashboard"
    )
    websocket_dashboard_url: str = Field(
        default="ws://localhost:8000/ws",
        description="Primary dashboard WebSocket URL for real-time data",
    )
    websocket_fallback_urls: str = Field(
        default="",
        description="Comma-separated list of fallback WebSocket URLs",
    )
    websocket_publish_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="WebSocket publishing interval in seconds",
    )
    websocket_max_retries: int = Field(
        default=15, ge=1, le=30, description="Maximum WebSocket reconnection attempts"
    )
    websocket_retry_delay: int = Field(
        default=5,
        ge=1,
        le=60,
        description="WebSocket reconnection base delay in seconds",
    )
    websocket_timeout: int = Field(
        default=45, ge=10, le=120, description="WebSocket connection timeout in seconds"
    )
    websocket_initial_connect_timeout: int = Field(
        default=60,
        ge=30,
        le=180,
        description="Initial WebSocket connection timeout in seconds",
    )
    websocket_connection_delay: int = Field(
        default=0,
        ge=0,
        le=60,
        description="Delay before initial WebSocket connection attempt in seconds",
    )
    websocket_queue_size: int = Field(
        default=2000,
        ge=50,
        le=5000,
        description="Maximum queued messages during connection issues",
    )
    # Additional WebSocket resilience settings
    websocket_ping_interval: int = Field(
        default=20, ge=5, le=60, description="WebSocket ping interval in seconds"
    )
    websocket_ping_timeout: int = Field(
        default=10, ge=3, le=30, description="WebSocket ping timeout in seconds"
    )
    websocket_health_check_interval: int = Field(
        default=45,
        ge=10,
        le=300,
        description="WebSocket health check interval in seconds",
    )

    @field_validator("alert_email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        """Validate email address format."""
        if v is None:
            return v

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email address format")

        return v


class Settings(BaseSettings):
    """Main configuration settings for the AI Trading Bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        frozen=True,
        extra="allow",
    )

    # Configuration Sections
    trading: TradingSettings = Field(default_factory=TradingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    dominance: DominanceSettings = Field(default_factory=DominanceSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    paper_trading: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    omnisearch: OmniSearchSettings = Field(default_factory=OmniSearchSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    market_making: MarketMakingConfig = Field(default_factory=MarketMakingConfig)

    # Profile Configuration
    profile: TradingProfile = Field(
        default=TradingProfile.MODERATE, description="Trading risk profile"
    )

    @computed_field
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.system.environment == Environment.PRODUCTION

    def requires_api_keys(self) -> bool:
        """Check if API keys are required for current configuration."""
        return (
            not self.system.dry_run or self.system.environment == Environment.PRODUCTION
        )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Validate required API keys based on configuration."""
        if self.requires_api_keys():
            # Validate LLM API keys
            if self.llm.provider == "openai" and not self.llm.openai_api_key:
                raise ValueError("OpenAI API key required for live trading")
            if self.llm.provider == "anthropic" and not self.llm.anthropic_api_key:
                raise ValueError("Anthropic API key required for live trading")

            # Validate exchange API keys
            if not self.system.dry_run:
                if self.exchange.exchange_type == "coinbase":
                    has_legacy = all(
                        [
                            self.exchange.cb_api_key,
                            self.exchange.cb_api_secret,
                            self.exchange.cb_passphrase,
                        ]
                    )

                    has_cdp = all(
                        [
                            self.exchange.cdp_api_key_name,
                            self.exchange.cdp_private_key,
                        ]
                    )

                    if not (has_legacy or has_cdp):
                        raise ValueError(
                            "Coinbase API credentials required for live trading (either legacy or CDP keys)"
                        )
                elif self.exchange.exchange_type == "bluefin":
                    if not self.exchange.bluefin_private_key:
                        raise ValueError(
                            "Bluefin private key required for live trading"
                        )

        return self

    @model_validator(mode="after")
    def validate_environment_consistency(self) -> "Settings":
        """Validate environment-specific configuration consistency."""
        env = self.system.environment

        # Production environment validations
        if env == Environment.PRODUCTION:
            # Allow dry-run override for testing purposes
            if self.system.dry_run:
                logger.warning(
                    "Production environment running in dry-run mode - this should only be for testing"
                )

            if self.exchange.cb_sandbox:
                raise ValueError("Production environment cannot use sandbox exchange")

            if self.trading.leverage > 10:
                raise ValueError("Production leverage should not exceed 10x")

            if self.risk.max_daily_loss_pct > 10.0:
                raise ValueError("Production daily loss limit should not exceed 10%")

        # Development environment validations
        elif env == Environment.DEVELOPMENT:
            if not self.system.dry_run:
                raise ValueError("Development environment should use dry-run mode")

        return self

    @model_validator(mode="after")
    def validate_risk_parameters(self) -> "Settings":
        """Validate risk management parameters."""
        if self.risk.default_take_profit_pct <= self.risk.default_stop_loss_pct:
            raise ValueError("Take profit must be greater than stop loss")

        if self.risk.max_weekly_loss_pct <= self.risk.max_daily_loss_pct:
            raise ValueError("Weekly loss limit must be greater than daily loss limit")

        if self.risk.max_monthly_loss_pct <= self.risk.max_weekly_loss_pct:
            raise ValueError(
                "Monthly loss limit must be greater than weekly loss limit"
            )

        return self

    @model_validator(mode="after")
    def validate_interval_trade_aggregation(self) -> "Settings":
        """Validate interval and trade aggregation compatibility."""
        # Sub-minute intervals that require trade aggregation
        sub_minute_intervals = ["1s", "5s", "15s", "30s"]

        # Check if using a sub-minute interval without trade aggregation
        if (
            self.trading.interval in sub_minute_intervals
            and not self.exchange.use_trade_aggregation
        ):
            raise ValueError(
                f"Trading interval '{self.trading.interval}' requires trade aggregation to be enabled. "
                f"Please set EXCHANGE__USE_TRADE_AGGREGATION=true in your environment configuration "
                f"or switch to a standard interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)."
            )

        return self

    @model_validator(mode="after")
    def validate_market_making_config(self) -> "Settings":
        """Validate market making configuration consistency."""
        if self.market_making.enabled:
            # Ensure market making is compatible with current exchange
            if self.exchange.exchange_type != "bluefin":
                logger.warning(
                    "Market making is optimized for Bluefin but exchange is set to %s",
                    self.exchange.exchange_type,
                )

            # Ensure market making symbol is compatible
            if not self.market_making.symbol.endswith("-PERP"):
                logger.warning(
                    "Market making symbol %s may not be optimal for perpetual futures",
                    self.market_making.symbol,
                )

            # Validate position size compatibility
            if self.market_making.strategy.max_position_pct > self.trading.max_size_pct:
                logger.warning(
                    "Market making max position (%s%%) exceeds trading max size (%s%%)",
                    self.market_making.strategy.max_position_pct,
                    self.trading.max_size_pct,
                )

            # Ensure reasonable cycle intervals
            if self.market_making.cycle_interval_seconds < 0.5:
                logger.warning(
                    "Market making cycle interval (%ss) may be too aggressive for stable operation",
                    self.market_making.cycle_interval_seconds,
                )

        return self

    def apply_profile(self, profile: TradingProfile) -> "Settings":
        """Apply a trading profile to adjust risk settings."""
        profile_configs = {
            TradingProfile.CONSERVATIVE: {
                "max_size_pct": 10.0,
                "leverage": 2,
                "max_daily_loss_pct": 2.0,
                "max_concurrent_trades": 1,
                "default_stop_loss_pct": 1.5,
                "default_take_profit_pct": 3.0,
            },
            TradingProfile.MODERATE: {
                "max_size_pct": 20.0,
                "leverage": 5,
                "max_daily_loss_pct": 5.0,
                "max_concurrent_trades": 3,
                "default_stop_loss_pct": 2.0,
                "default_take_profit_pct": 4.0,
            },
            TradingProfile.AGGRESSIVE: {
                "max_size_pct": 40.0,
                "leverage": 10,
                "max_daily_loss_pct": 10.0,
                "max_concurrent_trades": 5,
                "default_stop_loss_pct": 3.0,
                "default_take_profit_pct": 6.0,
            },
        }

        if profile == TradingProfile.CUSTOM:
            return self

        config_updates = profile_configs[profile]

        # Create new instances with updated values
        new_trading = self.trading.model_copy(
            update={
                "max_size_pct": config_updates["max_size_pct"],
                "leverage": config_updates["leverage"],
            }
        )

        new_risk = self.risk.model_copy(
            update={
                "max_daily_loss_pct": config_updates["max_daily_loss_pct"],
                "max_concurrent_trades": config_updates["max_concurrent_trades"],
                "default_stop_loss_pct": config_updates["default_stop_loss_pct"],
                "default_take_profit_pct": config_updates["default_take_profit_pct"],
            }
        )

        return self.model_copy(
            update={
                "trading": new_trading,
                "risk": new_risk,
                "profile": profile,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Convert settings to JSON string."""
        return self.model_dump_json(indent=indent)

    def save_to_file(
        self, file_path: str | Path, include_secrets: bool = False
    ) -> None:
        """Save configuration to JSON file with fallback directory support."""

        path = Path(file_path)
        # Use fallback-aware directory creation
        ensure_directory_exists(path.parent)

        # Export with or without secrets
        if include_secrets:
            data = self.model_dump(mode="json")
        else:
            data = self.model_dump(
                mode="json",
                exclude={
                    "llm": {"openai_api_key", "anthropic_api_key"},
                    "exchange": {
                        "cb_api_key",
                        "cb_api_secret",
                        "cb_passphrase",
                        "cdp_api_key_name",
                        "cdp_private_key",
                        "bluefin_private_key",
                    },
                    "system": {"api_secret_key"},
                    "mcp": {"memory_api_key"},
                    "omnisearch": {"api_key"},
                },
            )

        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        # Add metadata
        metadata = {
            "exported_at": datetime.now(UTC).isoformat(),
            "version": "1.0",
            "environment": self.system.environment.value,
            "profile": self.profile.value,
            "includes_secrets": include_secrets,
        }

        metadata_path = path.with_suffix(".metadata.json")
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "Settings":
        """Load configuration from JSON file with fallback directory support."""

        path = Path(file_path)

        # If the path doesn't exist and it's a relative path, try fallback directories
        if not path.exists() and not path.is_absolute():
            # Try in config directory with fallback support
            try:
                config_dir_path = get_config_directory() / path
                if config_dir_path.exists():
                    path = config_dir_path
            except OSError:
                pass  # Continue with original path

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open() as f:
            config_data = json.load(f)

        return cls(**config_data)

    def validate_network_consistency(self) -> list[str]:
        """Validate network configuration consistency across components."""
        warnings = []

        if self.exchange.exchange_type == "bluefin":
            # Check for network consistency
            network = self.exchange.bluefin_network

            # Warn about production networks
            if network == "mainnet" and self.system.dry_run:
                warnings.append(
                    "Using Bluefin mainnet in paper trading mode - "
                    "consider using testnet for development"
                )
            elif network == "testnet" and not self.system.dry_run:
                warnings.append(
                    "Using Bluefin testnet in live trading mode - "
                    "ensure this is intentional"
                )

            # Environment-specific recommendations
            if (
                self.system.environment == Environment.PRODUCTION
                and network == "testnet"
            ):
                warnings.append(
                    "Production environment configured with Bluefin testnet - "
                    "consider using mainnet for production"
                )
            elif (
                self.system.environment == Environment.DEVELOPMENT
                and network == "mainnet"
            ):
                warnings.append(
                    "Development environment configured with Bluefin mainnet - "
                    "consider using testnet for development"
                )

        return warnings

    def validate_trading_environment(self) -> list[str]:
        """Validate configuration for trading environment."""
        warnings = []

        # Include network consistency warnings
        warnings.extend(self.validate_network_consistency())

        if self.system.dry_run and self.system.environment == Environment.PRODUCTION:
            warnings.append("Running in dry-run mode in production environment")

        if not self.system.dry_run and self.exchange.cb_sandbox:
            warnings.append("Live trading enabled but using sandbox environment")

        if (
            self.trading.leverage > 10
            and self.system.environment == Environment.PRODUCTION
        ):
            warnings.append("High leverage detected in production environment")

        if self.risk.max_daily_loss_pct > 10.0:
            warnings.append("High daily loss limit may pose significant risk")

        # Additional validations
        if self.trading.max_size_pct > 50.0:
            warnings.append("Position size over 50% of equity is highly risky")

        if self.risk.max_concurrent_trades > 10:
            warnings.append("High number of concurrent trades may impact performance")

        risk_reward_ratio = (
            self.risk.default_take_profit_pct / self.risk.default_stop_loss_pct
        )
        if risk_reward_ratio < 1.5:
            warnings.append(
                f"Risk/reward ratio of {risk_reward_ratio:.2f} may not be profitable"
            )

        if self.llm.temperature > 0.3:
            warnings.append(
                "High LLM temperature may lead to inconsistent trading decisions"
            )

        return warnings

    def export_for_environment(self, target_env: Environment) -> dict[str, Any]:
        """Export configuration optimized for a specific environment."""
        config = self.model_dump()

        # Environment-specific adjustments
        if target_env == Environment.PRODUCTION:
            config["system"]["dry_run"] = False
            config["system"]["environment"] = Environment.PRODUCTION.value
            config["system"]["log_level"] = "INFO"
            config["exchange"]["cb_sandbox"] = False
            # Remove sensitive data
            for section in ["llm", "exchange", "system", "mcp", "omnisearch"]:
                if section in config:
                    for key in list(config[section].keys()):
                        if (
                            "key" in key.lower()
                            or "secret" in key.lower()
                            or "passphrase" in key.lower()
                            or "private" in key.lower()
                        ):
                            config[section][key] = None

        elif target_env == Environment.DEVELOPMENT:
            config["system"]["dry_run"] = True
            config["system"]["environment"] = Environment.DEVELOPMENT.value
            config["system"]["log_level"] = "DEBUG"
            config["exchange"]["cb_sandbox"] = True
            config["trading"]["leverage"] = min(config["trading"]["leverage"], 3)
            config["risk"]["max_daily_loss_pct"] = min(
                config["risk"]["max_daily_loss_pct"], 5.0
            )

        elif target_env == Environment.STAGING:
            config["system"]["dry_run"] = True
            config["system"]["environment"] = Environment.STAGING.value
            config["system"]["log_level"] = "INFO"
            config["exchange"]["cb_sandbox"] = True

        return config

    def get_effective_config(self) -> dict[str, Any]:
        """Get the effective configuration with all computed values."""
        base_config = self.model_dump()

        # Add computed fields
        base_config["computed"] = {
            "is_production": self.is_production,
            "requires_api_keys": self.requires_api_keys,
            "risk_reward_ratio": self.risk.default_take_profit_pct
            / self.risk.default_stop_loss_pct,
            "max_position_value": f"{self.trading.max_size_pct}% * {self.trading.leverage}x leverage",
            "effective_update_frequency": self.system.update_frequency_seconds,
        }

        return base_config

    def generate_config_hash(self) -> str:
        """Generate a hash of the current configuration for change detection."""
        # Create a normalized config without secrets and timestamps
        config_for_hash = self.model_dump(
            exclude={
                "llm": {"openai_api_key", "anthropic_api_key"},
                "exchange": {
                    "cb_api_key",
                    "cb_api_secret",
                    "cb_passphrase",
                    "cdp_api_key_name",
                    "cdp_private_key",
                    "bluefin_private_key",
                },
                "system": {"api_secret_key", "instance_id"},
                "mcp": {"memory_api_key"},
                "omnisearch": {"api_key"},
            }
        )

        config_str = json.dumps(config_for_hash, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def validate_configuration_comprehensive(self) -> FullValidationResults:
        """Run comprehensive configuration validation with network testing."""
        validator = ConfigurationValidator(self)
        return await validator.validate_all()

    def create_configuration_monitor(self) -> "ConfigurationMonitor":
        """Create a configuration monitor for runtime change detection."""
        return ConfigurationMonitor(self)

    def test_bluefin_configuration(self) -> TestValidationResult:
        """Test Bluefin-specific configuration synchronously."""
        if self.exchange.exchange_type != "bluefin":
            return {"status": "skipped", "reason": "Not using Bluefin exchange"}

        results: TestValidationResult = {"status": "pass", "reason": None, "tests": []}

        # Test private key format
        try:
            if self.exchange.bluefin_private_key:
                self.exchange._validate_bluefin_private_key_comprehensive()
                if results["tests"] is not None:
                    results["tests"].append(
                        {
                            "name": "private_key_format",
                            "status": "pass",
                            "warnings": None,
                        }
                    )
            elif results["tests"] is not None:
                results["tests"].append(
                    {
                        "name": "private_key_format",
                        "status": "warning",
                        "warnings": ["No private key provided"],
                    }
                )
        except ValueError as e:
            results["status"] = "skipped"
            if results["tests"] is not None:
                results["tests"].append(
                    {
                        "name": "private_key_format",
                        "status": "error",
                        "warnings": [str(e)],
                    }
                )

        # Test network endpoint validation
        endpoint_issues = self.exchange.validate_network_endpoints()
        if endpoint_issues:
            results["status"] = "skipped"
            if results["tests"] is not None:
                results["tests"].append(
                    {
                        "name": "network_endpoints",
                        "status": "error",
                        "warnings": endpoint_issues,
                    }
                )
        elif results["tests"] is not None:
            results["tests"].append(
                {"name": "network_endpoints", "status": "pass", "warnings": None}
            )

        # Test environment consistency
        consistency_warnings = self.validate_network_consistency()
        if consistency_warnings:
            if results["tests"] is not None:
                results["tests"].append(
                    {
                        "name": "environment_consistency",
                        "status": "warning",
                        "warnings": consistency_warnings,
                    }
                )
        elif results["tests"] is not None:
            results["tests"].append(
                {"name": "environment_consistency", "status": "pass", "warnings": None}
            )

        return results

    def get_configuration_summary(self) -> ConfigSummary:
        """Get a comprehensive configuration summary."""
        summary = ConfigSummary(
            basic_info={
                "environment": self.system.environment.value,
                "exchange": self.exchange.exchange_type,
                "dry_run": str(self.system.dry_run),
                "trading_symbol": self.trading.symbol,
                "leverage": str(self.trading.leverage),
                "profile": self.profile.value,
            },
            security_status={
                "has_llm_key": bool(self.llm.openai_api_key),
                "has_exchange_credentials": self._has_exchange_credentials(),
                "dry_run_enabled": self.system.dry_run,
            },
            risk_parameters={
                "max_daily_loss_pct": self.risk.max_daily_loss_pct,
                "max_position_size_pct": self.trading.max_size_pct,
                "stop_loss_pct": self.risk.default_stop_loss_pct,
                "take_profit_pct": self.risk.default_take_profit_pct,
                "risk_reward_ratio": self.risk.default_take_profit_pct
                / self.risk.default_stop_loss_pct,
            },
            network_config={},
            warnings=[],
            config_hash=self.generate_config_hash(),
        )

        # Add exchange-specific info
        if self.exchange.exchange_type == "bluefin":
            summary["network_config"] = {
                "network": self.exchange.bluefin_network,
                "service_url": self.exchange.bluefin_service_url,
                "custom_rpc": bool(self.exchange.bluefin_rpc_url),
                "endpoints": self.exchange.get_effective_endpoints(),
            }

            # Add network consistency warnings
            summary["warnings"].extend(self.validate_network_consistency())

        # Add trading environment warnings
        summary["warnings"].extend(self.validate_trading_environment())

        return summary

    def _has_exchange_credentials(self) -> bool:
        """Check if exchange credentials are configured."""
        if self.exchange.exchange_type == "coinbase":
            has_legacy = all(
                [
                    self.exchange.cb_api_key,
                    self.exchange.cb_api_secret,
                    self.exchange.cb_passphrase,
                ]
            )
            has_cdp = all(
                [
                    self.exchange.cdp_api_key_name,
                    self.exchange.cdp_private_key,
                ]
            )
            return has_legacy or has_cdp
        if self.exchange.exchange_type == "bluefin":
            return bool(self.exchange.bluefin_private_key)

        return False

    def create_backup_configuration(self) -> BackupConfig:
        """Create a backup of the current configuration without secrets."""
        return {
            "metadata": {
                "created_at": datetime.now(UTC).isoformat(),
                "config_hash": self.generate_config_hash(),
                "environment": self.system.environment.value,
                "exchange": self.exchange.exchange_type,
                "version": "1.0",
            },
            "configuration": self.model_dump(
                exclude={
                    "llm": {"openai_api_key", "anthropic_api_key"},
                    "exchange": {
                        "cb_api_key",
                        "cb_api_secret",
                        "cb_passphrase",
                        "cdp_api_key_name",
                        "cdp_private_key",
                        "bluefin_private_key",
                    },
                    "system": {"api_secret_key"},
                    "mcp": {"memory_api_key"},
                    "omnisearch": {"api_key"},
                }
            ),
        }


class ConfigurationMonitor:
    """Monitor configuration changes and provide hot-reloading capabilities."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.initial_hash = settings.generate_config_hash()
        self.last_check = time.time()
        self.change_callbacks: list[Callable[[Settings, str, str], None]] = []
        self.validation_cache: dict[str, FullValidationResults | float | str] = {}
        self.cache_ttl = 300  # 5 minutes

    def register_change_callback(
        self, callback: Callable[[Settings, str, str], None]
    ) -> None:
        """Register a callback to be called when configuration changes."""
        self.change_callbacks.append(callback)

    def check_for_changes(self) -> bool:
        """Check if configuration has changed since last check."""
        current_hash = self.settings.generate_config_hash()

        if current_hash != self.initial_hash:
            logger.info(
                "Configuration change detected: %s -> %s",
                self.initial_hash,
                current_hash,
            )

            # Notify callbacks
            for callback in self.change_callbacks:
                try:
                    callback(self.settings, self.initial_hash, current_hash)
                except Exception:
                    logger.exception("Error in configuration change callback")

            self.initial_hash = current_hash
            return True

        return False

    async def validate_and_cache(self) -> FullValidationResults:
        """Validate configuration and cache results."""
        current_time = time.time()

        # Check if cache is still valid
        if (
            "validation_results" in self.validation_cache
            and current_time - self.validation_cache.get("cached_at", 0)
            < self.cache_ttl
        ):
            return self.validation_cache["validation_results"]

        # Run validation
        validation_results = await self.settings.validate_configuration_comprehensive()

        # Cache results
        self.validation_cache = {
            "validation_results": validation_results,
            "cached_at": current_time,
            "config_hash": self.settings.generate_config_hash(),
        }

        return validation_results

    def get_health_status(self) -> HealthStatus:
        """Get overall configuration health status."""
        status = HealthStatus(
            overall_status="healthy",
            last_check=self.last_check,
            config_hash=self.settings.generate_config_hash(),
            issues=[],
        )

        # Check basic configuration issues
        if self.settings.exchange.exchange_type == "bluefin":
            try:
                endpoint_issues = self.settings.exchange.validate_network_endpoints()
                if endpoint_issues:
                    status["issues"].extend(endpoint_issues)
                    status["overall_status"] = "degraded"
            except Exception as e:
                status["issues"].append(f"Endpoint validation error: {e!s}")
                status["overall_status"] = "unhealthy"

        # Check environment consistency
        env_warnings = self.settings.validate_trading_environment()
        if env_warnings:
            status["issues"].extend(env_warnings)
            if status["overall_status"] == "healthy":
                status["overall_status"] = "warning"

        return status

    def export_monitoring_data(self) -> MonitoringData:
        """Export monitoring data for external analysis."""
        return {
            "monitor_info": {
                "initial_hash": self.initial_hash,
                "last_check": self.last_check,
                "cache_ttl": self.cache_ttl,
                "callback_count": len(self.change_callbacks),
            },
            "current_config": self.settings.get_configuration_summary(),
            "health_status": self.get_health_status(),
            "validation_cache": self.validation_cache,
        }


def create_settings(
    env_file: str | None = None,
    profile: TradingProfile | None = None,
    **overrides: dict[str, object],
) -> Settings:
    """Factory function to create settings with optional overrides."""
    # Ensure .env file is loaded
    try:
        env_path = env_file or ".env"
        if Path(env_path).exists():
            load_dotenv(env_path)
    except ImportError:
        # dotenv not available, rely on pydantic-settings
        pass

    # Set environment file if provided
    if env_file:
        os.environ.setdefault("ENV_FILE", env_file)

    # Create base settings
    # Type-safe settings creation
    settings_kwargs: dict[str, object] = {}
    if overrides:
        settings_kwargs.update(overrides)
    settings = Settings(**settings_kwargs)  # type: ignore[arg-type]

    # Apply profile if specified
    if profile:
        settings = settings.apply_profile(profile)

    return settings


# Global settings instance
settings = create_settings()


def get_config_template() -> dict[str, object]:
    """Get a configuration template with descriptions."""
    return {
        "trading": {
            "symbol": "BTC-USD",
            "interval": "5m",
            "leverage": 5,
            "max_size_pct": 20.0,
            "order_timeout_seconds": 30,
            "slippage_tolerance_pct": 0.1,
            "min_profit_pct": 0.5,
            "maker_fee_rate": 0.004,
            "taker_fee_rate": 0.006,
            "futures_fee_rate": 0.0015,
            "min_trading_interval_seconds": 60,
            "require_24h_data_before_trading": True,
            "min_candles_for_trading": 100,
            "enable_futures": True,
            "futures_account_type": "CFM",
            "auto_cash_transfer": True,
            "max_futures_leverage": 20,
        },
        "llm": {
            "provider": "openai",
            "model_name": "o3",
            "temperature": 0.1,
            "max_tokens": 30000,
            "request_timeout": 30,
            "max_retries": 3,
        },
        "exchange": {
            "exchange_type": "coinbase",
            "cb_sandbox": True,
            "api_timeout": 10,
            "rate_limit_requests": 10,
            "bluefin_network": "mainnet",
        },
        "risk": {
            "max_daily_loss_pct": 5.0,
            "max_concurrent_trades": 3,
            "default_stop_loss_pct": 2.0,
            "default_take_profit_pct": 4.0,
        },
        "system": {
            "dry_run": True,
            "environment": "development",
            "log_level": "INFO",
            "update_frequency_seconds": 30.0,
        },
        "paper_trading": {
            "starting_balance": 10000.0,
            "fee_rate": 0.001,
            "slippage_rate": 0.0005,
            "enable_daily_reports": True,
            "enable_weekly_summaries": True,
            "track_drawdown": True,
            "keep_trade_history_days": 90,
            "export_trade_data": False,
            "report_time_utc": "23:59",
            "include_unrealized_pnl": True,
        },
        "omnisearch": {
            "enabled": False,
            "server_url": "http://localhost:8766",
            "max_results": 5,
            "cache_ttl_seconds": 300,
            "rate_limit_requests_per_minute": 10,
            "timeout_seconds": 30,
            "enable_crypto_sentiment": True,
            "enable_nasdaq_sentiment": True,
            "enable_correlation_analysis": True,
        },
    }
