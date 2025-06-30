#!/usr/bin/env python3
"""
VPS Real-time Monitoring Script for AI Trading Bot
Monitors CPU, RAM, trading performance, and container health
Optimized for 1-core VPS environments
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/vps-monitor.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""

    timestamp: str
    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    load_average: list[float]
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class ContainerMetrics:
    """Docker container metrics"""

    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    memory_limit_mb: float
    memory_percent: float
    network_io: dict[str, float]
    restart_count: int
    health_status: str


@dataclass
class TradingMetrics:
    """Trading performance metrics"""

    timestamp: str
    balance: float
    pnl: float
    pnl_percent: float
    positions_count: int
    orders_count: int
    last_trade_time: str | None
    trades_today: int
    win_rate: float
    avg_trade_duration: float


class VPSMonitor:
    """Real-time VPS monitoring system"""

    def __init__(self, config_path: str = ".env"):
        self.config_path = config_path
        self.monitoring_active = True
        self.alerts_sent = set()
        self.metrics_history = []
        self.max_history = 1000  # Keep last 1000 data points

        # Load configuration
        self.load_config()

        # Performance thresholds for alerts
        self.cpu_threshold = 85.0
        self.memory_threshold = 90.0
        self.disk_threshold = 85.0

        # Container names to monitor
        self.containers = [
            "ai-trading-bot",
            "bluefin-service",
            "dashboard-backend",
            "dashboard-frontend",
            "mcp-memory",
            "prometheus",
            "grafana",
        ]

        # Trading API endpoints
        self.api_base = os.getenv("DASHBOARD_API_URL", "http://localhost:8000")

    def load_config(self):
        """Load environment configuration"""
        try:
            if os.path.exists(self.config_path):
                # Load from .env file if available
                with open(self.config_path) as f:
                    for line in f:
                        if "=" in line and not line.startswith("#"):
                            key, value = line.strip().split("=", 1)
                            os.environ[key] = value
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")

    def get_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            memory_total_mb = memory.total / 1024 / 1024

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_used_gb = disk.used / 1024 / 1024 / 1024
            disk_total_gb = disk.total / 1024 / 1024 / 1024

            # Load average
            load_avg = os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0]

            # Network I/O
            net_io = psutil.net_io_counters()
            network_sent_mb = net_io.bytes_sent / 1024 / 1024
            network_recv_mb = net_io.bytes_recv / 1024 / 1024

            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                load_average=list(load_avg),
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None

    def get_container_metrics(self) -> list[ContainerMetrics]:
        """Collect Docker container metrics"""
        metrics = []

        try:
            # Get container list
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning("Could not get container stats")
                return metrics

            lines = result.stdout.strip().split("\n")[1:]  # Skip header

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split("\t")
                if len(parts) < 6:
                    continue

                name = parts[0].strip()
                if not any(container in name for container in self.containers):
                    continue

                try:
                    # Parse CPU percentage
                    cpu_str = parts[1].strip().rstrip("%")
                    cpu_percent = float(cpu_str) if cpu_str != "--" else 0.0

                    # Parse memory usage
                    mem_usage = parts[2].strip()
                    if "/" in mem_usage:
                        used_str, limit_str = mem_usage.split("/")
                        memory_mb = self._parse_memory(used_str.strip())
                        memory_limit_mb = self._parse_memory(limit_str.strip())
                    else:
                        memory_mb = memory_limit_mb = 0

                    # Parse memory percentage
                    mem_perc_str = parts[3].strip().rstrip("%")
                    memory_percent = (
                        float(mem_perc_str) if mem_perc_str != "--" else 0.0
                    )

                    # Get additional container info
                    container_info = self.get_container_info(name)

                    metrics.append(
                        ContainerMetrics(
                            name=name,
                            status=container_info.get("status", "unknown"),
                            cpu_percent=cpu_percent,
                            memory_mb=memory_mb,
                            memory_limit_mb=memory_limit_mb,
                            memory_percent=memory_percent,
                            network_io={"sent": 0, "received": 0},  # Simplified
                            restart_count=container_info.get("restart_count", 0),
                            health_status=container_info.get("health", "unknown"),
                        )
                    )

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing container stats for {name}: {e}")
                    continue

        except subprocess.TimeoutExpired:
            logger.warning("Docker stats command timed out")
        except Exception as e:
            logger.error(f"Error collecting container metrics: {e}")

        return metrics

    def _parse_memory(self, mem_str: str) -> float:
        """Parse memory string like '123.4MiB' to MB"""
        mem_str = mem_str.strip()
        if mem_str.endswith("GiB"):
            return float(mem_str[:-3]) * 1024
        if mem_str.endswith("MiB"):
            return float(mem_str[:-3])
        if mem_str.endswith("KiB"):
            return float(mem_str[:-3]) / 1024
        if mem_str.endswith("B"):
            return float(mem_str[:-1]) / 1024 / 1024
        try:
            return float(mem_str)
        except ValueError:
            return 0.0

    def get_container_info(self, container_name: str) -> dict[str, Any]:
        """Get detailed container information"""
        try:
            result = subprocess.run(
                ["docker", "inspect", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return {"status": "not_found"}

            info = json.loads(result.stdout)[0]
            state = info.get("State", {})

            return {
                "status": "running" if state.get("Running") else "stopped",
                "restart_count": state.get("RestartCount", 0),
                "health": state.get("Health", {}).get("Status", "none"),
                "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
            }

        except Exception as e:
            logger.warning(f"Error getting container info for {container_name}: {e}")
            return {"status": "unknown"}

    def get_trading_metrics(self) -> TradingMetrics | None:
        """Collect trading performance metrics from API"""
        try:
            # Try to get metrics from dashboard API
            response = requests.get(f"{self.api_base}/api/trading/metrics", timeout=5)

            if response.status_code == 200:
                data = response.json()
                return TradingMetrics(
                    timestamp=datetime.now().isoformat(),
                    balance=data.get("balance", 0.0),
                    pnl=data.get("pnl", 0.0),
                    pnl_percent=data.get("pnl_percent", 0.0),
                    positions_count=data.get("positions_count", 0),
                    orders_count=data.get("orders_count", 0),
                    last_trade_time=data.get("last_trade_time"),
                    trades_today=data.get("trades_today", 0),
                    win_rate=data.get("win_rate", 0.0),
                    avg_trade_duration=data.get("avg_trade_duration", 0.0),
                )

        except requests.RequestException as e:
            logger.debug(f"Could not fetch trading metrics: {e}")
        except Exception as e:
            logger.warning(f"Error collecting trading metrics: {e}")

        return None

    def check_alerts(
        self, system_metrics: SystemMetrics, container_metrics: list[ContainerMetrics]
    ):
        """Check for alert conditions and log warnings"""
        alerts = []

        # System alerts
        if system_metrics.cpu_percent > self.cpu_threshold:
            alert = f"HIGH CPU: {system_metrics.cpu_percent:.1f}% (threshold: {self.cpu_threshold}%)"
            alerts.append(alert)

        if system_metrics.memory_percent > self.memory_threshold:
            alert = f"HIGH MEMORY: {system_metrics.memory_percent:.1f}% (threshold: {self.memory_threshold}%)"
            alerts.append(alert)

        if system_metrics.disk_percent > self.disk_threshold:
            alert = f"HIGH DISK: {system_metrics.disk_percent:.1f}% (threshold: {self.disk_threshold}%)"
            alerts.append(alert)

        # Load average alert (for single core)
        if system_metrics.load_average[0] > 2.0:
            alert = f"HIGH LOAD: {system_metrics.load_average[0]:.2f} (1-min avg)"
            alerts.append(alert)

        # Container alerts
        for container in container_metrics:
            if container.status != "running":
                alert = f"CONTAINER DOWN: {container.name} ({container.status})"
                alerts.append(alert)

            if container.memory_percent > 90:
                alert = f"CONTAINER HIGH MEMORY: {container.name} ({container.memory_percent:.1f}%)"
                alerts.append(alert)

            if container.restart_count > 0:
                alert = f"CONTAINER RESTARTS: {container.name} ({container.restart_count} restarts)"
                if alert not in self.alerts_sent:
                    alerts.append(alert)
                    self.alerts_sent.add(alert)

        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")

    def print_dashboard(
        self,
        system_metrics: SystemMetrics,
        container_metrics: list[ContainerMetrics],
        trading_metrics: TradingMetrics | None,
    ):
        """Print real-time dashboard to console"""

        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print(
            f"🚀 AI Trading Bot VPS Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 80)

        # System overview
        print("\n📊 SYSTEM PERFORMANCE")
        print(
            f"CPU:      {system_metrics.cpu_percent:5.1f}% | Cores: {system_metrics.cpu_count}"
        )
        print(
            f"Memory:   {system_metrics.memory_percent:5.1f}% | {system_metrics.memory_used_mb:6.0f}MB / {system_metrics.memory_total_mb:6.0f}MB"
        )
        print(
            f"Disk:     {system_metrics.disk_percent:5.1f}% | {system_metrics.disk_used_gb:5.1f}GB / {system_metrics.disk_total_gb:5.1f}GB"
        )
        print(
            f"Load Avg: {system_metrics.load_average[0]:5.2f} | {system_metrics.load_average[1]:5.2f} | {system_metrics.load_average[2]:5.2f}"
        )
        print(
            f"Network:  ↑{system_metrics.network_sent_mb:6.1f}MB ↓{system_metrics.network_recv_mb:6.1f}MB"
        )

        # Container status
        print("\n🐳 CONTAINER STATUS")
        print(
            f"{'Container':<20} {'Status':<10} {'CPU%':<8} {'Memory':<15} {'Health':<10}"
        )
        print("-" * 70)

        for container in container_metrics:
            status_icon = "✅" if container.status == "running" else "❌"
            health_icon = {
                "healthy": "💚",
                "unhealthy": "❤️",
                "none": "⚪",
                "unknown": "⚪",
            }.get(container.health_status, "⚪")

            print(
                f"{container.name:<20} {status_icon}{container.status:<9} {container.cpu_percent:6.1f}% {container.memory_mb:6.0f}MB/{container.memory_limit_mb:6.0f}MB {health_icon}{container.health_status:<9}"
            )

        # Trading performance
        if trading_metrics:
            print("\n💰 TRADING PERFORMANCE")
            pnl_icon = "📈" if trading_metrics.pnl >= 0 else "📉"
            print(f"Balance:       ${trading_metrics.balance:,.2f}")
            print(
                f"P&L:           {pnl_icon} ${trading_metrics.pnl:,.2f} ({trading_metrics.pnl_percent:+.2f}%)"
            )
            print(f"Positions:     {trading_metrics.positions_count}")
            print(f"Pending Orders:{trading_metrics.orders_count}")
            print(f"Trades Today:  {trading_metrics.trades_today}")
            print(f"Win Rate:      {trading_metrics.win_rate:.1f}%")
            if trading_metrics.last_trade_time:
                print(f"Last Trade:    {trading_metrics.last_trade_time}")
        else:
            print("\n💰 TRADING PERFORMANCE")
            print("📊 Trading metrics unavailable (dashboard API not accessible)")

        # Performance indicators
        print("\n🚨 STATUS INDICATORS")
        cpu_status = "🔥" if system_metrics.cpu_percent > 80 else "✅"
        mem_status = "🔥" if system_metrics.memory_percent > 85 else "✅"
        containers_running = sum(1 for c in container_metrics if c.status == "running")
        container_status = "✅" if containers_running >= 2 else "⚠️"

        print(
            f"CPU Load:      {cpu_status} {'Critical' if system_metrics.cpu_percent > 80 else 'Normal'}"
        )
        print(
            f"Memory Usage:  {mem_status} {'Critical' if system_metrics.memory_percent > 85 else 'Normal'}"
        )
        print(
            f"Containers:    {container_status} {containers_running}/{len(container_metrics)} running"
        )

        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)

    def save_metrics(
        self,
        system_metrics: SystemMetrics,
        container_metrics: list[ContainerMetrics],
        trading_metrics: TradingMetrics | None,
    ):
        """Save metrics to file for historical analysis"""
        try:
            metrics_data = {
                "timestamp": system_metrics.timestamp,
                "system": asdict(system_metrics),
                "containers": [asdict(c) for c in container_metrics],
                "trading": asdict(trading_metrics) if trading_metrics else None,
            }

            # Add to history
            self.metrics_history.append(metrics_data)

            # Keep only recent history
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history :]

            # Save to file every 10 minutes
            if len(self.metrics_history) % 60 == 0:  # Assuming 10s intervals
                with open("/tmp/vps-metrics-history.json", "w") as f:
                    json.dump(
                        self.metrics_history[-100:], f, indent=2
                    )  # Last 100 entries

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    async def monitor_loop(self, interval: int = 10):
        """Main monitoring loop"""
        logger.info("Starting VPS monitoring...")

        try:
            while self.monitoring_active:
                start_time = time.time()

                # Collect metrics
                system_metrics = self.get_system_metrics()
                container_metrics = self.get_container_metrics()
                trading_metrics = self.get_trading_metrics()

                if system_metrics:
                    # Check for alerts
                    self.check_alerts(system_metrics, container_metrics)

                    # Display dashboard
                    self.print_dashboard(
                        system_metrics, container_metrics, trading_metrics
                    )

                    # Save metrics
                    self.save_metrics(
                        system_metrics, container_metrics, trading_metrics
                    )

                # Wait for next iteration
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.monitoring_active = False

    def stop(self):
        """Stop monitoring"""
        self.monitoring_active = False


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="VPS Real-time Monitor for AI Trading Bot"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Monitoring interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--config", type=str, default=".env", help="Configuration file path"
    )
    parser.add_argument(
        "--cpu-threshold",
        type=float,
        default=85.0,
        help="CPU alert threshold percentage",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=90.0,
        help="Memory alert threshold percentage",
    )
    parser.add_argument(
        "--disk-threshold",
        type=float,
        default=85.0,
        help="Disk alert threshold percentage",
    )

    args = parser.parse_args()

    # Create monitor
    monitor = VPSMonitor(config_path=args.config)
    monitor.cpu_threshold = args.cpu_threshold
    monitor.memory_threshold = args.memory_threshold
    monitor.disk_threshold = args.disk_threshold

    # Run monitoring
    try:
        await monitor.monitor_loop(interval=args.interval)
    except KeyboardInterrupt:
        logger.info("Shutting down monitor...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
