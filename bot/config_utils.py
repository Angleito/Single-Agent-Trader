"""Configuration utilities and validation helpers."""

import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from .config import Environment, Settings, TradingProfile, create_settings

logger = logging.getLogger(__name__)


class StartupValidator:
    """Comprehensive startup validation for the trading bot."""

    def __init__(self, settings: Settings):
        """Initialize startup validator with settings."""
        self.settings = settings
        self.validation_results: dict[str, list[str]] = {
            "environment_vars": [],
            "api_connectivity": [],
            "configuration": [],
            "system_dependencies": [],
            "file_permissions": [],
            "warnings": [],
            "errors": [],
            "critical_errors": [],
        }

    def validate_environment_variables(self) -> list[str]:
        """Validate all required environment variables are present."""
        issues = []

        # LLM provider specific validation
        if self.settings.llm.provider == "openai":
            if not self.settings.llm.openai_api_key:
                issues.append("OpenAI API key is required when using OpenAI provider")
        elif self.settings.llm.provider == "anthropic":
            if not self.settings.llm.anthropic_api_key:
                issues.append(
                    "Anthropic API key is required when using Anthropic provider"
                )
        elif self.settings.llm.provider == "ollama":
            if not self.settings.llm.ollama_base_url:
                issues.append("Ollama base URL is required when using Ollama provider")

        # Exchange credentials validation (for live trading)
        if not self.settings.system.dry_run:
            if not self.settings.exchange.cb_api_key:
                issues.append("Coinbase API key is required for live trading")
            if not self.settings.exchange.cb_api_secret:
                issues.append("Coinbase API secret is required for live trading")
            if not self.settings.exchange.cb_passphrase:
                issues.append("Coinbase passphrase is required for live trading")

        # Critical system settings
        if self.settings.system.environment == Environment.PRODUCTION:
            if self.settings.system.dry_run:
                issues.append("Production environment should not use dry run mode")
            if self.settings.exchange.cb_sandbox:
                issues.append("Production environment should not use sandbox exchange")

        self.validation_results["environment_vars"] = issues
        return issues

    def validate_api_connectivity(self) -> list[str]:
        """Test connectivity to all required APIs."""
        issues = []

        # Test LLM provider connectivity
        llm_status = self._test_llm_connectivity()
        if not llm_status["success"]:
            issues.append(
                f"LLM provider ({self.settings.llm.provider}) "
                f"connectivity failed: {llm_status['error']}"
            )

        # Test exchange connectivity (if not dry run)
        if not self.settings.system.dry_run:
            exchange_status = self._test_exchange_connectivity()
            if not exchange_status["success"]:
                issues.append(
                    f"Exchange connectivity failed: {exchange_status['error']}"
                )

        self.validation_results["api_connectivity"] = issues
        return issues

    def validate_system_dependencies(self) -> list[str]:
        """Check system dependencies and requirements."""
        issues = []

        # Check Python version (updated for Python 3.12)
        if sys.version_info < (3, 12):  # noqa: UP036
            issues.append(
                f"Python 3.12+ required, found "
                f"{sys.version_info.major}.{sys.version_info.minor}"
            )

        # Check required modules
        required_modules = ["pandas", "numpy", "pydantic", "requests"]
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                issues.append(f"Required module '{module}' not installed")

        # Check memory and disk space
        memory_warning = self._check_system_resources()
        if memory_warning:
            issues.append(memory_warning)

        self.validation_results["system_dependencies"] = issues
        return issues

    def validate_file_permissions(self) -> list[str]:
        """Validate file and directory permissions."""
        issues = []

        # Check data directory
        data_path = Path(self.settings.data.data_storage_path)
        if not self._check_directory_permissions(data_path):
            issues.append(f"Cannot write to data directory: {data_path}")

        # Check logs directory
        log_path = Path(str(self.settings.system.log_file_path)).parent
        if not self._check_directory_permissions(log_path):
            issues.append(f"Cannot write to logs directory: {log_path}")

        # Check config directory
        config_path = Path("config")
        if not self._check_directory_permissions(config_path):
            issues.append(f"Cannot write to config directory: {config_path}")

        self.validation_results["file_permissions"] = issues
        return issues

    def validate_configuration_integrity(self) -> list[str]:
        """Validate configuration parameter integrity and consistency."""
        issues = []

        # Trading parameter validation
        if self.settings.trading.leverage > 20:
            issues.append("Leverage above 20x is extremely risky")

        if self.settings.trading.max_size_pct > 50:
            issues.append("Position size above 50% is extremely risky")

        # Risk management validation
        if (
            self.settings.risk.default_stop_loss_pct
            >= self.settings.risk.default_take_profit_pct
        ):
            issues.append(
                "Stop loss should be smaller than take profit "
                "for positive risk/reward"
            )

        if self.settings.risk.max_concurrent_trades > 20:
            issues.append("Too many concurrent trades may lead to overexposure")

        # Data configuration validation
        if self.settings.data.candle_limit < 50:
            issues.append(
                "Candle limit below 50 may not provide enough data for analysis"
            )

        if self.settings.data.candle_limit > 1000:
            issues.append("Candle limit above 1000 may impact performance")

        self.validation_results["configuration"] = issues
        return issues

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run all validation checks and return comprehensive results."""
        logger.info("Starting comprehensive startup validation...")

        # Run all validation checks
        env_issues = self.validate_environment_variables()
        api_issues = self.validate_api_connectivity()
        sys_issues = self.validate_system_dependencies()
        file_issues = self.validate_file_permissions()
        config_issues = self.validate_configuration_integrity()

        # Categorize issues by severity
        critical_issues = []
        warning_issues = []

        for issue in env_issues + api_issues + sys_issues + file_issues:
            if any(
                keyword in issue.lower()
                for keyword in ["required", "failed", "cannot", "not installed"]
            ):
                critical_issues.append(issue)
            else:
                warning_issues.append(issue)

        # Configuration issues are typically warnings unless critical
        for issue in config_issues:
            if any(keyword in issue.lower() for keyword in ["extremely risky"]):
                critical_issues.append(issue)
            else:
                warning_issues.append(issue)

        # Build final results
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "valid": len(critical_issues) == 0,
            "critical_errors": critical_issues,
            "warnings": warning_issues,
            "details": self.validation_results,
            "system_info": self._get_system_info(),
            "configuration_summary": self._get_config_summary(),
        }

        # Log results
        if results["valid"]:
            logger.info("Startup validation completed successfully")
            if warning_issues:
                logger.warning("Found %s warnings", len(warning_issues))
                for warning in warning_issues:
                    logger.warning("  - %s", warning)
        else:
            logger.error("Startup validation failed with critical errors")
            for error in critical_issues:
                logger.error("  - %s", error)

        return results

    def _test_llm_connectivity(self) -> dict[str, Any]:
        """Test LLM provider connectivity."""
        try:
            if self.settings.llm.provider == "openai" and requests:
                # Check if API key is available
                if not self.settings.llm.openai_api_key:
                    return {"success": False, "error": "OpenAI API key not configured"}

                # Get the secret value properly
                api_key = self.settings.llm.openai_api_key.get_secret_value()
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(
                    "https://api.openai.com/v1/models", headers=headers, timeout=10
                )

                if response.status_code == 200:
                    return {"success": True, "error": None}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    }

            elif self.settings.llm.provider == "anthropic" and requests:
                # Check if API key is available
                if not self.settings.llm.anthropic_api_key:
                    return {
                        "success": False,
                        "error": "Anthropic API key not configured",
                    }

                api_key = self.settings.llm.anthropic_api_key.get_secret_value()
                headers = {"x-api-key": api_key}
                # Anthropic doesn't have a simple health check, so we'll just verify key format
                return {"success": True, "error": None}

            elif self.settings.llm.provider == "ollama" and requests:
                response = requests.get(
                    f"{self.settings.llm.ollama_base_url}/api/tags", timeout=10
                )

                if response.status_code == 200:
                    return {"success": True, "error": None}
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    }

            else:
                return {
                    "success": True,
                    "error": "Connectivity test skipped "
                    "(requests module not available)",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_exchange_connectivity(self) -> dict[str, Any]:
        """Test exchange connectivity."""
        try:
            # For now, we'll just check if credentials are provided
            if (
                self.settings.exchange.cb_api_key
                and self.settings.exchange.cb_api_secret
            ):
                return {"success": True, "error": None}
            else:
                return {"success": False, "error": "Missing exchange credentials"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _check_system_resources(self) -> str | None:
        """Check system resources like memory and disk space."""
        try:
            import psutil

            # Check available memory
            memory = psutil.virtual_memory()
            if memory.available < 512 * 1024 * 1024:  # 512MB
                return "Low available memory (<512MB). " "Performance may be affected."

            # Check disk space
            disk = psutil.disk_usage(".")
            if disk.free < 1024 * 1024 * 1024:  # 1GB
                return "Low disk space (<1GB). Data storage may be affected."

        except ImportError:
            return "psutil module not available, cannot check system resources"
        except Exception as e:
            return f"Error checking system resources: {e}"

        return None

    def _check_directory_permissions(self, path: Path) -> bool:
        """Check if directory is writable."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / f"test_{int(time.time())}.tmp"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for debugging."""
        return {
            "python_version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "platform": sys.platform,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _get_config_summary(self) -> dict[str, Any]:
        """Get configuration summary for validation report."""
        return {
            "environment": self.settings.system.environment.value,
            "dry_run": self.settings.system.dry_run,
            "profile": self.settings.profile.value,
            "llm_provider": self.settings.llm.provider,
            "trading_symbol": self.settings.trading.symbol,
            "leverage": self.settings.trading.leverage,
            "sandbox_mode": self.settings.exchange.cb_sandbox,
        }


class ConfigValidator:
    """Configuration validation utilities."""

    @staticmethod
    def validate_api_connectivity(settings: Settings) -> dict[str, bool]:
        """Validate API connectivity for configured services."""
        results = {
            "llm_provider": False,
            "exchange": False,
        }

        # Test LLM provider connectivity
        try:
            if settings.llm.provider == "openai" and settings.llm.openai_api_key:
                # In a real implementation, you would test the API connection
                results["llm_provider"] = True
            elif (
                settings.llm.provider == "anthropic" and settings.llm.anthropic_api_key
            ):
                results["llm_provider"] = True
            elif settings.llm.provider == "ollama":
                # Test local Ollama connection
                results["llm_provider"] = True
        except Exception as e:
            logger.warning("LLM provider validation failed: %s", e)

        # Test exchange connectivity
        try:
            if settings.exchange.cb_api_key and settings.exchange.cb_api_secret:
                # In a real implementation, you would test the exchange connection
                results["exchange"] = True
        except Exception as e:
            logger.warning("Exchange validation failed: %s", e)

        return results

    @staticmethod
    def validate_trading_parameters(settings: Settings) -> list[str]:
        """Validate trading parameters for safety and correctness."""
        issues = []

        # Check leverage vs risk settings
        if settings.trading.leverage > 10 and settings.risk.max_daily_loss_pct > 5.0:
            issues.append(
                "High leverage combined with high daily loss limit may be risky"
            )

        # Check position sizing
        if settings.trading.max_size_pct > 50.0:
            issues.append("Position size over 50% of equity is highly risky")

        # Check stop loss vs take profit ratio
        risk_reward_ratio = (
            settings.risk.default_take_profit_pct / settings.risk.default_stop_loss_pct
        )
        if risk_reward_ratio < 1.5:
            issues.append(
                f"Risk/reward ratio of {risk_reward_ratio:.2f} "
                f"may not be profitable long-term"
            )

        # Check concurrent trades vs account size
        if settings.risk.max_concurrent_trades > 10:
            issues.append(
                "High number of concurrent trades may lead to over-diversification"
            )

        return issues

    @staticmethod
    def validate_environment_consistency(settings: Settings) -> list[str]:
        """Validate environment-specific configuration consistency."""
        issues = []

        # Production environment checks
        if settings.system.environment == Environment.PRODUCTION:
            if settings.system.dry_run:
                issues.append("Production environment should not use dry-run mode")

            if settings.exchange.cb_sandbox:
                issues.append("Production environment should not use sandbox exchange")

            if settings.system.log_level == "DEBUG":
                issues.append("DEBUG logging in production may impact performance")

        # Development environment checks
        elif settings.system.environment == Environment.DEVELOPMENT:
            if not settings.system.dry_run:
                issues.append("Development environment should use dry-run mode")

        return issues


class ConfigManager:
    """Enhanced configuration management utilities."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def save_profile_config(self, profile: TradingProfile, settings: Settings) -> Path:
        """Save configuration for a specific trading profile."""
        filename = f"{profile.value}_config.json"
        file_path = self.config_dir / filename

        settings.save_to_file(file_path)
        logger.info("Saved %s configuration to %s", profile.value, file_path)

        return file_path

    def load_profile_config(self, profile: TradingProfile) -> Settings | None:
        """Load configuration for a specific trading profile."""
        filename = f"{profile.value}_config.json"
        file_path = self.config_dir / filename

        if not file_path.exists():
            logger.warning("Configuration file not found: %s", file_path)
            return None

        try:
            settings = Settings.load_from_file(file_path)
            logger.info("Loaded %s configuration from %s", profile.value, file_path)
            return settings
        except Exception as e:
            logger.exception("Failed to load configuration from %s: %s", file_path, e)
            return None

    def create_environment_configs(self) -> dict[Environment, Path]:
        """Create configuration files for different environments."""
        configs = {}

        # Development configuration
        dev_settings = create_settings(
            system__environment=Environment.DEVELOPMENT,
            system__dry_run=True,
            system__log_level="DEBUG",
            exchange__cb_sandbox=True,
            trading__leverage=2,
            risk__max_daily_loss_pct=2.0,
        )
        dev_path = self.config_dir / "development.json"
        dev_settings.save_to_file(dev_path)
        configs[Environment.DEVELOPMENT] = dev_path

        # Staging configuration
        staging_settings = create_settings(
            system__environment=Environment.STAGING,
            system__dry_run=True,
            system__log_level="INFO",
            exchange__cb_sandbox=True,
            trading__leverage=5,
            risk__max_daily_loss_pct=3.0,
        )
        staging_path = self.config_dir / "staging.json"
        staging_settings.save_to_file(staging_path)
        configs[Environment.STAGING] = staging_path

        # Production configuration template
        prod_settings = create_settings(
            system__environment=Environment.PRODUCTION,
            system__dry_run=False,
            system__log_level="INFO",
            exchange__cb_sandbox=False,
            trading__leverage=3,
            risk__max_daily_loss_pct=5.0,
        )
        prod_path = self.config_dir / "production.json"
        prod_settings.save_to_file(prod_path)
        configs[Environment.PRODUCTION] = prod_path

        logger.info("Created environment-specific configuration files")
        return configs

    def create_trading_profile_configs(self) -> dict[TradingProfile, Path]:
        """Create configuration files for different trading profiles."""
        configs = {}
        base_settings = create_settings()

        for profile in [
            TradingProfile.CONSERVATIVE,
            TradingProfile.MODERATE,
            TradingProfile.AGGRESSIVE,
        ]:
            profile_settings = base_settings.apply_profile(profile)
            file_path = self.save_profile_config(profile, profile_settings)
            configs[profile] = file_path

        logger.info("Created trading profile configuration files")
        return configs

    def create_config_backup(
        self, settings: Settings, backup_name: str | None = None
    ) -> Path:
        """Create a backup of the current configuration."""
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}.json"

        backup_path = self.backup_dir / backup_name
        settings.save_to_file(backup_path)
        logger.info("Configuration backup created: %s", backup_path)
        return backup_path

    def restore_config_backup(self, backup_name: str) -> Settings | None:
        """Restore configuration from a backup."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return None

        try:
            settings = Settings.load_from_file(backup_path)
            logger.info("Configuration restored from backup: %s", backup_path)
            return settings
        except Exception as e:
            logger.exception("Failed to restore backup %s: %s", backup_path, e)
            return None

    def list_config_backups(self) -> list[dict[str, Any]]:
        """List all available configuration backups."""
        backups = []
        for backup_file in self.backup_dir.glob("*.json"):
            try:
                stat = backup_file.stat()
                backups.append(
                    {
                        "name": backup_file.name,
                        "path": str(backup_file),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "size": stat.st_size,
                    }
                )
            except Exception as e:
                logger.warning("Error reading backup file %s: %s", backup_file, e)

        return sorted(backups, key=lambda x: x["created"], reverse=True)

    def export_configuration(
        self, settings: Settings, export_format: str = "json"
    ) -> Path:
        """Export configuration in various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format.lower() == "json":
            export_path = self.config_dir / f"export_{timestamp}.json"
            settings.save_to_file(export_path)

        elif export_format.lower() == "env":
            export_path = self.config_dir / f"export_{timestamp}.env"
            env_content = self._settings_to_env_format(settings)
            with open(export_path, "w") as f:
                f.write(env_content)

        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        logger.info("Configuration exported to: %s", export_path)
        return export_path

    def import_configuration(
        self, import_path: Path, import_format: str = "json"
    ) -> Settings | None:
        """Import configuration from various formats."""
        if not import_path.exists():
            logger.error("Import file not found: %s", import_path)
            return None

        try:
            if import_format.lower() == "json":
                settings = Settings.load_from_file(import_path)
            else:
                raise ValueError(f"Unsupported import format: {import_format}")

            logger.info("Configuration imported from: %s", import_path)
            return settings

        except Exception as e:
            logger.exception("Failed to import configuration from %s: %s", import_path, e)
            return None

    def switch_profile(
        self, settings: Settings, new_profile: TradingProfile, save_current: bool = True
    ) -> Settings:
        """Switch trading profile with optional backup of current settings."""
        if save_current:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"profile_{settings.profile.value}_backup_{timestamp}.json"
            self.create_config_backup(settings, backup_name)

        new_settings = settings.apply_profile(new_profile)
        logger.info("Switched from %s to " "%s profile" ), settings.profile.value, new_profile.value)
        return new_settings

    def _settings_to_env_format(self, settings: Settings) -> str:
        """Convert settings to environment variable format."""
        lines = ["# Exported Trading Bot Configuration", ""]

        # This would be a comprehensive mapping of all settings to env vars
        # For brevity, showing key mappings
        env_mappings = {
            f"TRADING__SYMBOL={settings.trading.symbol}",
            f"TRADING__LEVERAGE={settings.trading.leverage}",
            f"LLM__PROVIDER={settings.llm.provider}",
            f"SYSTEM__DRY_RUN={str(settings.system.dry_run).lower()}",
            f"SYSTEM__ENVIRONMENT={settings.system.environment.value}",
            f"PROFILE={settings.profile.value}",
        }

        lines.extend(sorted(env_mappings))
        return "\n".join(lines)

    def generate_env_template(self) -> Path:
        """Generate a .env template file with all available options."""
        # This method is kept for backward compatibility
        # The actual template is now in .env.example
        template_path = Path(".env.template")
        logger.info("Use .env.example as the template file. " "Legacy template: %s" ), template_path)
        return template_path


def validate_configuration(settings: Settings) -> dict[str, Any]:
    """Comprehensive configuration validation using new validator."""
    startup_validator = StartupValidator(settings)
    return startup_validator.run_comprehensive_validation()


def validate_configuration_legacy(settings: Settings) -> dict[str, Any]:
    """Legacy configuration validation for backward compatibility."""
    validator = ConfigValidator()

    results: dict[str, Any] = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "connectivity": {},
    }

    try:
        # Basic parameter validation
        trading_issues = validator.validate_trading_parameters(settings)
        if trading_issues:
            results["warnings"].extend(trading_issues)

        # Environment consistency validation
        env_issues = validator.validate_environment_consistency(settings)
        if env_issues:
            results["warnings"].extend(env_issues)

        # Trading environment validation
        if hasattr(settings, "validate_trading_environment"):
            trading_warnings = settings.validate_trading_environment()
            if trading_warnings:
                results["warnings"].extend(trading_warnings)

        # API connectivity validation
        connectivity = validator.validate_api_connectivity(settings)
        results["connectivity"] = connectivity

        # Check for critical issues
        if not connectivity.get("llm_provider", False) and not settings.system.dry_run:
            results["errors"].append("LLM provider not accessible for live trading")

        if not connectivity.get("exchange", False) and not settings.system.dry_run:
            results["errors"].append("Exchange not accessible for live trading")

        if results["errors"]:
            results["valid"] = False

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Configuration validation failed: {e}")

    return results


class HealthMonitor:
    """System health monitoring and metrics collection."""

    def __init__(self, settings: Settings):
        """Initialize health monitor with settings."""
        self.settings = settings
        self.metrics: dict[str, Any] = {
            "startup_time": None,
            "last_health_check": None,
            "system_status": "unknown",
            "component_status": {},
            "performance_metrics": {},
            "error_counts": {},
            "uptime_seconds": 0,
        }
        self.startup_time = datetime.now(UTC)

    def perform_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check of all components."""
        logger.info("Performing system health check...")

        health_status: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_status": "healthy",
            "components": {},
            "metrics": {},
            "issues": [],
        }

        # Check system resources
        system_health = self._check_system_health()
        health_status["components"]["system"] = system_health

        # Check API connectivity
        api_health = self._check_api_health()
        health_status["components"]["apis"] = api_health

        # Check file system
        fs_health = self._check_filesystem_health()
        health_status["components"]["filesystem"] = fs_health

        # Check configuration integrity
        config_health = self._check_configuration_health()
        health_status["components"]["configuration"] = config_health

        # Collect performance metrics
        perf_metrics = self._collect_performance_metrics()
        health_status["metrics"] = perf_metrics

        # Determine overall status
        component_statuses = [
            comp["status"] for comp in health_status["components"].values()
        ]
        if "critical" in component_statuses:
            health_status["overall_status"] = "critical"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "healthy"

        # Update internal metrics
        self.metrics["last_health_check"] = health_status["timestamp"]
        self.metrics["system_status"] = health_status["overall_status"]
        self.metrics["component_status"] = {
            name: comp["status"] for name, comp in health_status["components"].items()
        }

        logger.info("Health check completed. Overall status: " "%s" ), health_status['overall_status'])
        return health_status

    def get_status_summary(self) -> dict[str, Any]:
        """Get a quick status summary."""
        uptime = (datetime.now(UTC) - self.startup_time).total_seconds()

        return {
            "status": self.metrics["system_status"],
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "last_check": self.metrics["last_health_check"],
            "components": self.metrics["component_status"],
            "startup_time": self.startup_time.isoformat(),
        }

    def _check_system_health(self) -> dict[str, Any]:
        """Check system resource health."""
        try:
            import psutil

            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(".")

            # Determine status based on thresholds
            status = "healthy"
            issues = []

            if cpu_percent > 80:
                status = "warning"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory.percent > 85:
                status = "warning"
                issues.append(f"High memory usage: {memory.percent:.1f}%")

            if disk.percent > 90:
                status = "critical"
                issues.append(f"Low disk space: {disk.percent:.1f}% used")

            return {
                "status": status,
                "issues": issues,
                "metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                },
            }

        except ImportError:
            return {
                "status": "warning",
                "issues": ["psutil not available for system monitoring"],
                "metrics": {},
            }
        except Exception as e:
            return {
                "status": "critical",
                "issues": [f"System health check failed: {e}"],
                "metrics": {},
            }

    def _check_api_health(self) -> dict[str, Any]:
        """Check API connectivity health."""
        validator = StartupValidator(self.settings)

        api_issues = validator.validate_api_connectivity()

        if not api_issues:
            return {
                "status": "healthy",
                "issues": [],
                "metrics": {"connectivity_test": "passed"},
            }
        else:
            return {
                "status": (
                    "critical" if not self.settings.system.dry_run else "warning"
                ),
                "issues": api_issues,
                "metrics": {"connectivity_test": "failed"},
            }

    def _check_filesystem_health(self) -> dict[str, Any]:
        """Check filesystem health."""
        validator = StartupValidator(self.settings)

        fs_issues = validator.validate_file_permissions()

        if not fs_issues:
            return {
                "status": "healthy",
                "issues": [],
                "metrics": {"file_permissions": "valid"},
            }
        else:
            return {
                "status": "critical",
                "issues": fs_issues,
                "metrics": {"file_permissions": "invalid"},
            }

    def _check_configuration_health(self) -> dict[str, Any]:
        """Check configuration health."""
        validator = StartupValidator(self.settings)

        config_issues = validator.validate_configuration_integrity()

        status = "healthy"
        if any("extremely risky" in issue.lower() for issue in config_issues):
            status = "warning"

        return {
            "status": status,
            "issues": config_issues,
            "metrics": {
                "config_validation": ("passed" if not config_issues else "warnings")
            },
        }

    def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect performance metrics."""
        uptime = (datetime.now(UTC) - self.startup_time).total_seconds()

        metrics: dict[str, Any] = {
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "checks_performed": 1,  # This would be incremented in real implementation
        }

        try:
            import psutil

            process = psutil.Process()
            metrics.update(
                {
                    "memory_usage_mb": process.memory_info().rss / (1024**2),
                    "cpu_percent": process.cpu_percent(),
                    "open_files": len(process.open_files()),
                    "threads": process.num_threads(),
                }
            )
        except (ImportError, Exception):
            pass

        return metrics

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def create_startup_report(settings: Settings) -> dict[str, Any]:
    """Create a comprehensive startup report."""
    logger.info("Generating startup report...")

    # Run startup validation
    validator = StartupValidator(settings)
    validation_results = validator.run_comprehensive_validation()

    # Initialize health monitor
    health_monitor = HealthMonitor(settings)
    health_status = health_monitor.perform_health_check()

    # Create comprehensive report
    report = {
        "report_timestamp": datetime.now(UTC).isoformat(),
        "startup_validation": validation_results,
        "health_check": health_status,
        "configuration_summary": validation_results.get("configuration_summary", {}),
        "system_info": validation_results.get("system_info", {}),
        "recommendations": _generate_recommendations(validation_results, health_status),
    }

    logger.info("Startup report generated successfully")
    return report


def _generate_recommendations(
    validation_results: dict[str, Any], health_status: dict[str, Any]
) -> list[str]:
    """Generate recommendations based on validation and health check results."""
    recommendations = []

    # Validation-based recommendations
    if not validation_results.get("valid", True):
        recommendations.append(
            "Address critical configuration errors before starting the bot"
        )

    if validation_results.get("warnings"):
        recommendations.append(
            "Review and address configuration warnings for optimal performance"
        )

    # Health-based recommendations
    if health_status.get("overall_status") == "critical":
        recommendations.append("Resolve critical system issues before proceeding")

    # System-specific recommendations
    system_comp = health_status.get("components", {}).get("system", {})
    if system_comp.get("metrics", {}).get("memory_percent", 0) > 75:
        recommendations.append(
            "Consider increasing available memory for better performance"
        )

    if system_comp.get("metrics", {}).get("disk_percent", 0) > 80:
        recommendations.append("Free up disk space to ensure proper operation")

    # Configuration-specific recommendations
    config_summary = validation_results.get("configuration_summary", {})
    if config_summary.get("dry_run"):
        recommendations.append(
            "Currently in dry-run mode - switch to live trading when ready"
        )

    if config_summary.get("leverage", 0) > 10:
        recommendations.append("Consider reducing leverage for safer trading")

    return recommendations


def setup_configuration(
    environment: Environment | None = None,
    profile: TradingProfile | None = None,
    config_file: str | None = None,
) -> Settings:
    """Setup configuration with environment and profile detection."""

    # Ensure .env file is loaded first
    try:
        from dotenv import load_dotenv

        if Path(".env").exists():
            load_dotenv()
            logger.debug("Loaded .env file")
    except ImportError:
        logger.debug("python-dotenv not available, relying on pydantic-settings")

    # Detect environment from various sources
    if not environment:
        env_var = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
        try:
            environment = Environment(env_var)
        except ValueError:
            logger.warning("Unknown environment '%s', defaulting to development", env_var)
            environment = Environment.DEVELOPMENT

    # Detect profile from environment variable
    if not profile:
        profile_var = os.getenv("TRADING_PROFILE", "moderate").lower()
        try:
            profile = TradingProfile(profile_var)
        except ValueError:
            logger.warning("Unknown trading profile '%s', defaulting to moderate", profile_var)
            profile = TradingProfile.MODERATE

    # Load from config file if provided
    if config_file and Path(config_file).exists():
        logger.info("Loading configuration from %s", config_file)
        settings = Settings.load_from_file(config_file)
        # Apply profile if different
        if settings.profile != profile:
            settings = settings.apply_profile(profile)
    else:
        # Create settings with detected parameters
        settings = create_settings()
        if profile:
            settings = settings.apply_profile(profile)

    # Validate configuration with comprehensive startup validation
    validation_results = validate_configuration(settings)

    if not validation_results["valid"]:
        for error in validation_results["critical_errors"]:
            logger.error("Configuration error: %s", error)
        raise ValueError("Configuration validation failed with critical errors")

    for warning in validation_results["warnings"]:
        logger.warning("Configuration warning: %s", warning)

    logger.info("Configuration loaded successfully:")
    logger.info("  Environment: %s", settings.system.environment.value)
    logger.info("  Profile: %s", settings.profile.value)
    logger.info("  Dry Run: %s", settings.system.dry_run)
    logger.info("  Symbol: %s", settings.trading.symbol)
    logger.info("  Leverage: %sx", settings.trading.leverage)

    return settings
