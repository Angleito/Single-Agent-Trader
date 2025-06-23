#!/usr/bin/env python3
"""
Continuous Health Monitoring Script for AI Trading Bot

This script monitors the health of all services and trading operations,
alerting on issues and providing detailed diagnostics.

Features:
- Service availability monitoring
- Performance metrics tracking
- Error rate monitoring
- Trading safety checks
- Resource usage monitoring
- Automated alerting
"""

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import docker
import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.config import Settings
from bot.types.services import ServiceHealth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(project_root / "logs" / "health-monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

console = Console()


class HealthMonitor:
    """Monitors system health and trading safety."""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.settings = Settings()
        self.start_time = datetime.now(UTC)
        self.alerts: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {
            "services": {},
            "trading": {
                "mode": "unknown",
                "positions": 0,
                "pnl": 0.0,
                "errors": 0,
            },
            "system": {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_usage": 0.0,
            },
        }

    async def check_service_health(self, service_name: str, url: str) -> ServiceHealth:
        """Check health of a service endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = (time.time() - start_time) * 1000  # ms

                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "last_check": time.time(),
                            "response_time_ms": response_time,
                        }
                    return {
                        "status": "unhealthy",
                        "last_check": time.time(),
                        "error": f"HTTP {response.status}",
                        "response_time_ms": response_time,
                    }
        except TimeoutError:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "error": str(e),
            }

    async def monitor_docker_services(self):
        """Monitor Docker container health."""
        services = {
            "ai-trading-bot": {"critical": True, "health_endpoint": None},
            "mcp-memory": {
                "critical": True,
                "health_endpoint": "http://localhost:8765/health",
            },
            "mcp-omnisearch": {"critical": True, "health_endpoint": None},
            "bluefin-service": {
                "critical": False,
                "health_endpoint": "http://localhost:8081/health",
            },
            "dashboard-backend": {
                "critical": False,
                "health_endpoint": "http://localhost:8000/health",
            },
            "dashboard-frontend": {"critical": False, "health_endpoint": None},
        }

        for service_name, config in services.items():
            try:
                container = self.docker_client.containers.get(service_name)
                status = container.status

                # Get container stats
                stats = container.stats(stream=False)
                cpu_percent = self._calculate_cpu_percent(stats)
                memory_usage = stats["memory_stats"].get("usage", 0) / (
                    1024 * 1024
                )  # MB

                # Check health endpoint if available
                if config["health_endpoint"]:
                    health = await self.check_service_health(
                        service_name, config["health_endpoint"]
                    )
                else:
                    health = {
                        "status": "healthy" if status == "running" else "unhealthy",
                        "last_check": time.time(),
                    }

                self.metrics["services"][service_name] = {
                    "status": status,
                    "health": health,
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_usage,
                    "critical": config["critical"],
                    "restarts": container.attrs["RestartCount"],
                }

                # Alert on critical service issues
                if config["critical"] and health["status"] != "healthy":
                    self.add_alert(
                        "critical", f"Service {service_name} is unhealthy", health
                    )

            except docker.errors.NotFound:
                self.metrics["services"][service_name] = {
                    "status": "not_found",
                    "health": {"status": "unknown", "last_check": time.time()},
                    "critical": config["critical"],
                }
                if config["critical"]:
                    self.add_alert("critical", f"Service {service_name} not found")

    def _calculate_cpu_percent(self, stats: dict[str, Any]) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"]
                - stats["precpu_stats"]["system_cpu_usage"]
            )
            num_cpus = len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))

            if system_delta > 0:
                return (cpu_delta / system_delta) * num_cpus * 100.0
        except Exception:
            pass
        return 0.0

    async def monitor_trading_safety(self):
        """Monitor trading safety and risk parameters."""
        try:
            # Check trading mode
            self.metrics["trading"]["mode"] = (
                "PAPER" if self.settings.system.dry_run else "LIVE"
            )

            # Read recent logs for trading activity
            log_path = project_root / "logs" / "trading-bot.log"
            if log_path.exists():
                with open(log_path) as f:
                    recent_logs = f.readlines()[-100:]  # Last 100 lines

                    # Count errors
                    error_count = sum(1 for line in recent_logs if "ERROR" in line)
                    self.metrics["trading"]["errors"] = error_count

                    # Check for position info
                    for line in reversed(recent_logs):
                        if "position" in line.lower():
                            # Parse position info (simplified)
                            if "LONG" in line:
                                self.metrics["trading"]["positions"] = 1
                            elif "SHORT" in line:
                                self.metrics["trading"]["positions"] = -1
                            break

                    # Alert on high error rate
                    if error_count > 10:
                        self.add_alert(
                            "warning",
                            f"High error rate: {error_count} errors in recent logs",
                        )

            # Safety checks for live trading
            if not self.settings.system.dry_run:
                # Check leverage
                if self.settings.trading.leverage > 10:
                    self.add_alert(
                        "warning",
                        f"High leverage detected: {self.settings.trading.leverage}x",
                    )

                # Check risk parameters
                if self.settings.risk.max_position_risk_pct > 5:
                    self.add_alert(
                        "warning",
                        f"High position risk: {self.settings.risk.max_position_risk_pct}%",
                    )

        except Exception as e:
            logger.error(f"Error monitoring trading safety: {e}")

    def monitor_system_resources(self):
        """Monitor system resource usage."""
        try:
            # CPU usage
            self.metrics["system"]["cpu_percent"] = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["system"]["memory_percent"] = memory.percent
            self.metrics["system"]["memory_available_gb"] = memory.available / (1024**3)

            # Disk usage
            disk = psutil.disk_usage("/")
            self.metrics["system"]["disk_usage"] = disk.percent
            self.metrics["system"]["disk_free_gb"] = disk.free / (1024**3)

            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics["system"]["network_sent_mb"] = net_io.bytes_sent / (1024**2)
            self.metrics["system"]["network_recv_mb"] = net_io.bytes_recv / (1024**2)

            # Alerts for resource issues
            if self.metrics["system"]["cpu_percent"] > 80:
                self.add_alert(
                    "warning",
                    f"High CPU usage: {self.metrics['system']['cpu_percent']:.1f}%",
                )

            if self.metrics["system"]["memory_percent"] > 85:
                self.add_alert(
                    "warning",
                    f"High memory usage: {self.metrics['system']['memory_percent']:.1f}%",
                )

            if self.metrics["system"]["disk_usage"] > 90:
                self.add_alert(
                    "critical",
                    f"Low disk space: {self.metrics['system']['disk_usage']:.1f}% used",
                )

        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")

    def add_alert(
        self, severity: str, message: str, details: dict[str, Any] | None = None
    ):
        """Add an alert to the alert list."""
        alert = {
            "timestamp": datetime.now(UTC),
            "severity": severity,
            "message": message,
            "details": details or {},
        }
        self.alerts.append(alert)

        # Keep only recent alerts (last hour)
        cutoff_time = datetime.now(UTC) - timedelta(hours=1)
        self.alerts = [a for a in self.alerts if a["timestamp"] > cutoff_time]

        # Log critical alerts
        if severity == "critical":
            logger.error(f"CRITICAL ALERT: {message}")

    def create_dashboard(self) -> Layout:
        """Create a rich dashboard layout."""
        layout = Layout()

        # Main layout structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="alerts", size=10),
        )

        # Header
        uptime = datetime.now(UTC) - self.start_time
        header_text = Text()
        header_text.append("🤖 AI Trading Bot Health Monitor", style="bold cyan")
        header_text.append(f"\nUptime: {uptime}", style="dim")
        header_text.append(" | Mode: ", style="dim")

        mode = self.metrics["trading"]["mode"]
        mode_style = "green" if mode == "PAPER" else "red bold"
        header_text.append(mode, style=mode_style)

        layout["header"].update(Panel(header_text))

        # Body split
        layout["body"].split_row(
            Layout(name="services", ratio=2),
            Layout(name="metrics", ratio=1),
        )

        # Services table
        services_table = Table(title="Services Status")
        services_table.add_column("Service", style="cyan")
        services_table.add_column("Status", style="green")
        services_table.add_column("Health", style="green")
        services_table.add_column("CPU %", style="yellow")
        services_table.add_column("Memory MB", style="yellow")

        for service, data in self.metrics["services"].items():
            status_style = "green" if data["status"] == "running" else "red"
            health_style = "green" if data["health"]["status"] == "healthy" else "red"

            services_table.add_row(
                service,
                Text(data["status"], style=status_style),
                Text(data["health"]["status"], style=health_style),
                f"{data.get('cpu_percent', 0):.1f}",
                f"{data.get('memory_mb', 0):.1f}",
            )

        layout["services"].update(Panel(services_table))

        # Metrics panel
        metrics_text = Text()
        metrics_text.append("System Resources\n", style="bold")
        metrics_text.append(f"CPU: {self.metrics['system']['cpu_percent']:.1f}%\n")
        metrics_text.append(
            f"Memory: {self.metrics['system']['memory_percent']:.1f}%\n"
        )
        metrics_text.append(f"Disk: {self.metrics['system']['disk_usage']:.1f}%\n\n")

        metrics_text.append("Trading Metrics\n", style="bold")
        metrics_text.append(f"Positions: {self.metrics['trading']['positions']}\n")
        metrics_text.append(f"Recent Errors: {self.metrics['trading']['errors']}\n")

        layout["metrics"].update(Panel(metrics_text, title="Metrics"))

        # Alerts
        alerts_table = Table(title="Recent Alerts")
        alerts_table.add_column("Time", style="dim")
        alerts_table.add_column("Severity", style="yellow")
        alerts_table.add_column("Message", style="white")

        for alert in sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[
            :5
        ]:
            severity_style = "red bold" if alert["severity"] == "critical" else "yellow"
            alerts_table.add_row(
                alert["timestamp"].strftime("%H:%M:%S"),
                Text(alert["severity"].upper(), style=severity_style),
                alert["message"],
            )

        layout["alerts"].update(Panel(alerts_table))

        return layout

    async def run_monitoring_loop(self):
        """Run the main monitoring loop."""
        with Live(self.create_dashboard(), refresh_per_second=1) as live:
            while True:
                try:
                    # Run all monitoring tasks
                    await self.monitor_docker_services()
                    await self.monitor_trading_safety()
                    self.monitor_system_resources()

                    # Update dashboard
                    live.update(self.create_dashboard())

                    # Save metrics snapshot
                    self.save_metrics_snapshot()

                    # Wait before next update
                    await asyncio.sleep(5)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(5)

    def save_metrics_snapshot(self):
        """Save current metrics to file for historical analysis."""
        snapshot = {
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": self.metrics,
            "alerts": [
                {
                    "timestamp": a["timestamp"].isoformat(),
                    "severity": a["severity"],
                    "message": a["message"],
                }
                for a in self.alerts[-10:]  # Last 10 alerts
            ],
        }

        snapshot_path = project_root / "logs" / "health-snapshots.jsonl"
        with open(snapshot_path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")


async def main():
    """Main entry point."""
    console.print("[bold cyan]Starting AI Trading Bot Health Monitor...[/bold cyan]")

    monitor = HealthMonitor()

    try:
        await monitor.run_monitoring_loop()
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Fatal error in health monitor")


if __name__ == "__main__":
    asyncio.run(main())
