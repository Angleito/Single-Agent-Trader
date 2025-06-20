"""
Health check and monitoring endpoints for the AI Trading Bot.

This module provides HTTP-like endpoints and utilities for monitoring
the bot's health, configuration status, and performance metrics.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .config_utils import HealthMonitor, StartupValidator, create_startup_report


class HealthCheckEndpoints:
    """HTTP-like health check endpoints for monitoring integration."""

    def __init__(self, settings: Settings):
        """Initialize health check endpoints."""
        self.settings = settings
        self.health_monitor = HealthMonitor(settings)
        self.startup_time: datetime = datetime.now(UTC)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl: int = 30  # seconds

    def get_health(self) -> dict[str, Any]:
        """Get basic health status (lightweight endpoint)."""
        return {
            "status": "healthy",  # This would be determined by quick checks
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": (datetime.now(UTC) - self.startup_time).total_seconds(),
            "version": "1.0.0",  # This would come from package metadata
            "environment": self.settings.system.environment.value,
            "dry_run": self.settings.system.dry_run,
        }

    def get_health_detailed(self, force_refresh: bool = False) -> dict[str, Any]:
        """Get detailed health status (comprehensive check)."""
        cache_key = "detailed_health"

        # Check cache
        if not force_refresh and self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        # Perform comprehensive health check
        health_status = self.health_monitor.perform_health_check()

        # Cache result
        self._cache[cache_key] = {"data": health_status, "timestamp": time.time()}

        return health_status

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return self.health_monitor._collect_performance_metrics()

    def get_configuration_status(self) -> dict[str, Any]:
        """Get configuration validation status."""
        cache_key = "config_status"

        # Check cache
        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        # Run configuration validation
        validator = StartupValidator(self.settings)
        validation_results = validator.run_comprehensive_validation()

        # Simplified status for API consumption
        status = {
            "valid": validation_results["valid"],
            "environment": validation_results["configuration_summary"]["environment"],
            "profile": validation_results["configuration_summary"]["profile"],
            "dry_run": validation_results["configuration_summary"]["dry_run"],
            "critical_errors_count": len(validation_results["critical_errors"]),
            "warnings_count": len(validation_results["warnings"]),
            "last_validated": validation_results["timestamp"],
        }

        # Cache result
        self._cache[cache_key] = {"data": status, "timestamp": time.time()}

        return status

    def get_startup_report(self) -> dict[str, Any]:
        """Get comprehensive startup report."""
        return create_startup_report(self.settings)

    def get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        try:
            import platform

            import psutil

            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage(".").total / (1024**3),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except ImportError:
            return {
                "platform": "unknown",
                "python_version": "unknown",
                "timestamp": datetime.now(UTC).isoformat(),
                "note": "psutil not available for detailed system info",
            }

    def get_readiness(self) -> dict[str, Any]:
        """Check if the bot is ready to start trading."""
        validator = StartupValidator(self.settings)
        validation_results = validator.run_comprehensive_validation()

        # Determine readiness based on validation
        validation_results["valid"]

        # Additional readiness checks
        readiness_checks = {
            "configuration_valid": validation_results["valid"],
            "no_critical_errors": len(validation_results["critical_errors"]) == 0,
            "apis_accessible": True,  # This would check actual API connectivity
            "filesystem_writable": True,  # This would check file permissions
            "sufficient_resources": True,  # This would check system resources
        }

        # Overall readiness
        overall_ready = all(readiness_checks.values())

        return {
            "ready": overall_ready,
            "checks": readiness_checks,
            "message": (
                "Ready to start trading"
                if overall_ready
                else "Not ready - check failed validations"
            ),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_liveness(self) -> dict[str, Any]:
        """Check if the bot process is alive and responding."""
        return {
            "alive": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": (datetime.now(UTC) - self.startup_time).total_seconds(),
        }

    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid."""
        if key not in self._cache:
            return False

        age = time.time() - self._cache[key]["timestamp"]
        return age < self._cache_ttl


class MonitoringExporter:
    """Export monitoring data in various formats for external systems."""

    def __init__(self, settings: Settings):
        """Initialize monitoring exporter."""
        self.settings = settings
        self.endpoints = HealthCheckEndpoints(settings)

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.endpoints.get_metrics()
        health = self.endpoints.get_health()

        lines = [
            "# HELP trading_bot_uptime_seconds Total uptime of the trading bot",
            "# TYPE trading_bot_uptime_seconds counter",
            f"trading_bot_uptime_seconds {metrics.get('uptime_seconds', 0)}",
            "",
            "# HELP trading_bot_health_status Health status of the trading bot (1=healthy, 0=unhealthy)",
            "# TYPE trading_bot_health_status gauge",
            f'trading_bot_health_status{{environment="{self.settings.system.environment.value}"}} {1 if health['status'] == 'healthy' else 0}',
            "",
        ]

        # Add memory metrics if available
        if "memory_usage_mb" in metrics:
            lines.extend(
                [
                    "# HELP trading_bot_memory_usage_bytes Memory usage in bytes",
                    "# TYPE trading_bot_memory_usage_bytes gauge",
                    f"trading_bot_memory_usage_bytes {metrics['memory_usage_mb'] * 1024 * 1024}",
                    "",
                ]
            )

        # Add CPU metrics if available
        if "cpu_percent" in metrics:
            lines.extend(
                [
                    "# HELP trading_bot_cpu_usage_percent CPU usage percentage",
                    "# TYPE trading_bot_cpu_usage_percent gauge",
                    f"trading_bot_cpu_usage_percent {metrics['cpu_percent']}",
                    "",
                ]
            )

        return "\n".join(lines)

    def export_json_summary(self) -> str:
        """Export summary data in JSON format."""
        data = {
            "health": self.endpoints.get_health(),
            "metrics": self.endpoints.get_metrics(),
            "configuration": self.endpoints.get_configuration_status(),
            "readiness": self.endpoints.get_readiness(),
            "system_info": self.endpoints.get_system_info(),
        }

        return json.dumps(data, indent=2, default=str)

    def save_monitoring_snapshot(self, output_dir: Path | None = None) -> Path:
        """Save a complete monitoring snapshot to disk."""
        if not output_dir:
            output_dir = Path("logs") / "monitoring"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = output_dir / f"monitoring_snapshot_{timestamp}.json"

        # Collect all monitoring data
        snapshot_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "basic_health": self.endpoints.get_health(),
            "detailed_health": self.endpoints.get_health_detailed(),
            "metrics": self.endpoints.get_metrics(),
            "configuration_status": self.endpoints.get_configuration_status(),
            "system_info": self.endpoints.get_system_info(),
            "readiness": self.endpoints.get_readiness(),
            "startup_report": self.endpoints.get_startup_report(),
        }

        # Save to file
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

        return snapshot_file


def create_health_endpoints(settings: Settings) -> HealthCheckEndpoints:
    """Factory function to create health check endpoints."""
    return HealthCheckEndpoints(settings)


def create_monitoring_exporter(settings: Settings) -> MonitoringExporter:
    """Factory function to create monitoring exporter."""
    return MonitoringExporter(settings)
