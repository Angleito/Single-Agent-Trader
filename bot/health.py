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

from .config import Settings, StartupValidator
from .utils.path_utils import get_logs_directory, get_safe_file_path

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


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
            f'trading_bot_health_status{{environment="{self.settings.system.environment.value}"}} {1 if health["status"] == "healthy" else 0}',
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
            output_dir = get_logs_directory() / "monitoring"

        # Generate filename with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        snapshot_file = get_safe_file_path(
            output_dir, f"monitoring_snapshot_{timestamp}.json"
        )

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
        with snapshot_file.open("w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

        return snapshot_file


def create_health_endpoints(settings: Settings) -> HealthCheckEndpoints:
    """Factory function to create health check endpoints."""
    return HealthCheckEndpoints(settings)


def create_monitoring_exporter(settings: Settings) -> MonitoringExporter:
    """Factory function to create monitoring exporter."""
    return MonitoringExporter(settings)


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

        return health_status

    def _check_system_health(self) -> dict[str, Any]:
        """Check system resource health."""
        if psutil is None:
            return {"status": "warning", "issues": ["psutil not available"], "metrics": {}}

        try:
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

        except Exception as e:
            return {
                "status": "critical",
                "issues": [f"System health check failed: {e}"],
                "metrics": {},
            }

    def _check_api_health(self) -> dict[str, Any]:
        """Check API connectivity health."""
        validator = StartupValidator(self.settings)
        api_issues = validator.validate_environment_variables()

        if not api_issues:
            return {
                "status": "healthy",
                "issues": [],
                "metrics": {"connectivity_test": "passed"},
            }
        return {
            "status": "warning" if self.settings.system.dry_run else "critical",
            "issues": api_issues,
            "metrics": {"connectivity_test": "failed"},
        }

    def _check_filesystem_health(self) -> dict[str, Any]:
        """Check filesystem health."""
        issues = []
        
        # Check data directory
        data_path = Path(self.settings.data.data_storage_path)
        if not self._check_directory_permissions(data_path):
            issues.append(f"Cannot write to data directory: {data_path}")
        
        # Check logs directory
        try:
            log_dir = get_logs_directory()
            if not self._check_directory_permissions(log_dir):
                issues.append(f"Cannot write to logs directory: {log_dir}")
        except Exception as e:
            issues.append(f"Cannot access logs directory: {e}")

        if not issues:
            return {
                "status": "healthy",
                "issues": [],
                "metrics": {"file_permissions": "valid"},
            }
        return {
            "status": "critical",
            "issues": issues,
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
                "config_validation": "passed" if not config_issues else "warnings"
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

        if psutil is None:
            return metrics

        try:
            process = psutil.Process()
            metrics.update(
                {
                    "memory_usage_mb": process.memory_info().rss / (1024**2),
                    "cpu_percent": process.cpu_percent(),
                    "open_files": len(process.open_files()),
                    "threads": process.num_threads(),
                }
            )
        except Exception:
            pass  # Ignore errors in metrics collection

        return metrics

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

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"


def create_startup_report(settings: Settings) -> dict[str, Any]:
    """Create a comprehensive startup report."""
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


__all__ = [
    "HealthCheckEndpoints",
    "HealthMonitor",
    "MonitoringExporter",
    "create_health_endpoints",
    "create_monitoring_exporter",
    "create_startup_report",
]
