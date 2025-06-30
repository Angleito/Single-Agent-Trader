#!/usr/bin/env python3
"""
Lightweight Container Resource Monitor
Embeddable monitoring script for individual containers to track resource usage
and generate alerts when thresholds are exceeded.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


# Configure logging
def setup_logging(container_name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup container-specific logging"""
    # Create logs directory
    log_dir = Path("/app/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger
    logger = logging.getLogger(f"monitor.{container_name}")
    logger.setLevel(getattr(logging, log_level.upper()))

    # File handler for persistent logging
    file_handler = logging.FileHandler(log_dir / f"{container_name}-monitor.log")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s - MONITOR[%(name)s] - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


@dataclass
class ResourceSnapshot:
    """Snapshot of container resource usage at a point in time"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_limit_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_rx_bytes: int
    network_tx_bytes: int
    process_count: int
    load_avg: list[float]
    open_files: int
    tcp_connections: int


@dataclass
class AlertConfig:
    """Alert threshold configuration"""

    cpu_warning: float = 70.0
    cpu_critical: float = 85.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    process_warning: int = 100
    process_critical: int = 200
    network_warning_mbps: float = 50.0
    network_critical_mbps: float = 100.0


class ContainerMonitor:
    """Lightweight container resource monitor"""

    def __init__(
        self,
        container_name: str,
        interval: int = 30,
        alert_config: AlertConfig | None = None,
        enable_metrics_endpoint: bool = True,
    ):
        self.container_name = container_name
        self.interval = interval
        self.alert_config = alert_config or AlertConfig()
        self.enable_metrics_endpoint = enable_metrics_endpoint

        # Setup logging
        self.logger = setup_logging(container_name, os.environ.get("LOG_LEVEL", "INFO"))

        # Initialize state
        self.running = False
        self.start_time = time.time()
        self.snapshots = deque(maxlen=100)  # Keep last 100 snapshots
        self.alert_history = deque(maxlen=50)

        # Track baseline metrics for anomaly detection
        self.baseline_cpu = None
        self.baseline_memory = None
        self.last_network_stats = None

        # Docker container resource limits detection
        self.memory_limit = self._detect_memory_limit()

        self.logger.info(f"Container monitor initialized for {container_name}")
        self.logger.info(
            f"Memory limit detected: {self.memory_limit / 1024 / 1024:.0f}MB"
        )

    def _detect_memory_limit(self) -> int:
        """Detect container memory limit from cgroup"""
        try:
            # Try Docker cgroup v2 first
            cgroup_files = [
                "/sys/fs/cgroup/memory.max",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/proc/1/cgroup",
            ]

            for file_path in cgroup_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path) as f:
                            content = f.read().strip()
                            if file_path.endswith("memory.max"):
                                if content != "max":
                                    return int(content)
                            elif file_path.endswith("memory.limit_in_bytes"):
                                limit = int(content)
                                # Filter out unrealistic high values (like 9223372036854775807)
                                if limit < 1024**4:  # Less than 1TB
                                    return limit
                    except (ValueError, OSError):
                        continue

            # Fallback to system memory
            return psutil.virtual_memory().total

        except Exception as e:
            self.logger.warning(f"Could not detect memory limit: {e}")
            return psutil.virtual_memory().total

    def collect_snapshot(self) -> ResourceSnapshot:
        """Collect current resource usage snapshot"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_limit_mb = self.memory_limit / (1024 * 1024)
            memory_percent = (memory.used / self.memory_limit) * 100

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)

            # Network I/O
            net_io = psutil.net_io_counters()

            # Process information
            process_count = len(psutil.pids())

            # Load average
            try:
                load_avg = list(os.getloadavg())
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]

            # File descriptors
            try:
                process = psutil.Process()
                open_files = process.num_fds() if hasattr(process, "num_fds") else 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                open_files = 0

            # TCP connections
            try:
                tcp_connections = len(
                    [
                        c
                        for c in psutil.net_connections()
                        if c.type == psutil.SOCK_STREAM
                    ]
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                tcp_connections = 0

            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_limit_mb=memory_limit_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_rx_bytes=net_io.bytes_recv,
                network_tx_bytes=net_io.bytes_sent,
                process_count=process_count,
                load_avg=load_avg,
                open_files=open_files,
                tcp_connections=tcp_connections,
            )

            return snapshot

        except Exception as e:
            self.logger.error(f"Error collecting snapshot: {e}")
            raise

    def check_alerts(self, snapshot: ResourceSnapshot) -> list[str]:
        """Check for alert conditions and return alert messages"""
        alerts = []
        timestamp = datetime.fromtimestamp(snapshot.timestamp).isoformat()

        # CPU alerts
        if snapshot.cpu_percent >= self.alert_config.cpu_critical:
            alert = f"CRITICAL: CPU usage {snapshot.cpu_percent:.1f}% >= {self.alert_config.cpu_critical}%"
            alerts.append(alert)
            self.logger.critical(alert)
        elif snapshot.cpu_percent >= self.alert_config.cpu_warning:
            alert = f"WARNING: CPU usage {snapshot.cpu_percent:.1f}% >= {self.alert_config.cpu_warning}%"
            alerts.append(alert)
            self.logger.warning(alert)

        # Memory alerts
        if snapshot.memory_percent >= self.alert_config.memory_critical:
            alert = f"CRITICAL: Memory usage {snapshot.memory_percent:.1f}% >= {self.alert_config.memory_critical}% ({snapshot.memory_used_mb:.0f}MB)"
            alerts.append(alert)
            self.logger.critical(alert)
        elif snapshot.memory_percent >= self.alert_config.memory_warning:
            alert = f"WARNING: Memory usage {snapshot.memory_percent:.1f}% >= {self.alert_config.memory_warning}% ({snapshot.memory_used_mb:.0f}MB)"
            alerts.append(alert)
            self.logger.warning(alert)

        # Disk alerts
        if snapshot.disk_usage_percent >= self.alert_config.disk_critical:
            alert = f"CRITICAL: Disk usage {snapshot.disk_usage_percent:.1f}% >= {self.alert_config.disk_critical}%"
            alerts.append(alert)
            self.logger.critical(alert)
        elif snapshot.disk_usage_percent >= self.alert_config.disk_warning:
            alert = f"WARNING: Disk usage {snapshot.disk_usage_percent:.1f}% >= {self.alert_config.disk_warning}%"
            alerts.append(alert)
            self.logger.warning(alert)

        # Process count alerts
        if snapshot.process_count >= self.alert_config.process_critical:
            alert = f"CRITICAL: Process count {snapshot.process_count} >= {self.alert_config.process_critical}"
            alerts.append(alert)
            self.logger.critical(alert)
        elif snapshot.process_count >= self.alert_config.process_warning:
            alert = f"WARNING: Process count {snapshot.process_count} >= {self.alert_config.process_warning}"
            alerts.append(alert)
            self.logger.warning(alert)

        # Network rate alerts (if we have previous data)
        if self.last_network_stats and len(self.snapshots) > 0:
            time_delta = snapshot.timestamp - self.snapshots[-1].timestamp
            if time_delta > 0:
                rx_rate_mbps = (
                    (snapshot.network_rx_bytes - self.last_network_stats["rx"])
                    / time_delta
                    / (1024 * 1024)
                )
                tx_rate_mbps = (
                    (snapshot.network_tx_bytes - self.last_network_stats["tx"])
                    / time_delta
                    / (1024 * 1024)
                )

                if (
                    max(rx_rate_mbps, tx_rate_mbps)
                    >= self.alert_config.network_critical_mbps
                ):
                    alert = f"CRITICAL: Network usage {max(rx_rate_mbps, tx_rate_mbps):.1f} Mbps >= {self.alert_config.network_critical_mbps} Mbps"
                    alerts.append(alert)
                    self.logger.critical(alert)
                elif (
                    max(rx_rate_mbps, tx_rate_mbps)
                    >= self.alert_config.network_warning_mbps
                ):
                    alert = f"WARNING: Network usage {max(rx_rate_mbps, tx_rate_mbps):.1f} Mbps >= {self.alert_config.network_warning_mbps} Mbps"
                    alerts.append(alert)
                    self.logger.warning(alert)

        # Store network stats for next comparison
        self.last_network_stats = {
            "rx": snapshot.network_rx_bytes,
            "tx": snapshot.network_tx_bytes,
        }

        # Record alerts in history
        if alerts:
            self.alert_history.append({"timestamp": timestamp, "alerts": alerts})

        return alerts

    def save_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Save snapshot to JSON file for external analysis"""
        try:
            log_file = Path("/app/logs") / f"{self.container_name}-metrics.jsonl"
            with open(log_file, "a") as f:
                json.dump(asdict(snapshot), f)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Error saving snapshot: {e}")

    def get_performance_summary(self, minutes: int = 60) -> dict[str, Any]:
        """Get performance summary for the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {"error": "No recent data available"}

        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]

        current = recent_snapshots[-1]

        return {
            "container": self.container_name,
            "period_minutes": minutes,
            "uptime_seconds": int(time.time() - self.start_time),
            "current": {
                "cpu_percent": current.cpu_percent,
                "memory_percent": current.memory_percent,
                "memory_used_mb": current.memory_used_mb,
                "disk_usage_percent": current.disk_usage_percent,
                "process_count": current.process_count,
                "tcp_connections": current.tcp_connections,
            },
            "averages": {
                "cpu_percent": sum(cpu_values) / len(cpu_values),
                "memory_percent": sum(memory_values) / len(memory_values),
            },
            "peaks": {
                "cpu_percent": max(cpu_values),
                "memory_percent": max(memory_values),
            },
            "sample_count": len(recent_snapshots),
            "recent_alerts": len(
                [
                    a
                    for a in self.alert_history
                    if time.time()
                    - float(datetime.fromisoformat(a["timestamp"]).timestamp())
                    <= minutes * 60
                ]
            ),
        }

    def generate_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics"""
        if not self.snapshots:
            return "# No metrics available\n"

        latest = self.snapshots[-1]
        container = self.container_name

        metrics = f"""# HELP container_cpu_percent Container CPU usage percentage
# TYPE container_cpu_percent gauge
container_cpu_percent{{container="{container}"}} {latest.cpu_percent}

# HELP container_memory_percent Container memory usage percentage
# TYPE container_memory_percent gauge
container_memory_percent{{container="{container}"}} {latest.memory_percent}

# HELP container_memory_used_mb Container memory used in MB
# TYPE container_memory_used_mb gauge
container_memory_used_mb{{container="{container}"}} {latest.memory_used_mb}

# HELP container_memory_limit_mb Container memory limit in MB
# TYPE container_memory_limit_mb gauge
container_memory_limit_mb{{container="{container}"}} {latest.memory_limit_mb}

# HELP container_disk_usage_percent Container disk usage percentage
# TYPE container_disk_usage_percent gauge
container_disk_usage_percent{{container="{container}"}} {latest.disk_usage_percent}

# HELP container_process_count Container process count
# TYPE container_process_count gauge
container_process_count{{container="{container}"}} {latest.process_count}

# HELP container_tcp_connections Container TCP connections count
# TYPE container_tcp_connections gauge
container_tcp_connections{{container="{container}"}} {latest.tcp_connections}

# HELP container_open_files Container open files count
# TYPE container_open_files gauge
container_open_files{{container="{container}"}} {latest.open_files}

# HELP container_network_rx_bytes Container network bytes received
# TYPE container_network_rx_bytes counter
container_network_rx_bytes{{container="{container}"}} {latest.network_rx_bytes}

# HELP container_network_tx_bytes Container network bytes transmitted
# TYPE container_network_tx_bytes counter
container_network_tx_bytes{{container="{container}"}} {latest.network_tx_bytes}

# HELP container_uptime_seconds Container uptime in seconds
# TYPE container_uptime_seconds gauge
container_uptime_seconds{{container="{container}"}} {time.time() - self.start_time}
"""
        return metrics

    async def serve_metrics_endpoint(self, port: int = 9090) -> None:
        """Serve metrics via HTTP endpoint"""
        from aiohttp import web

        async def health_handler(request):
            summary = self.get_performance_summary(5)  # Last 5 minutes
            return web.json_response(summary)

        async def metrics_handler(request):
            summary = self.get_performance_summary(60)  # Last hour
            return web.json_response(summary)

        async def prometheus_handler(request):
            metrics = self.generate_prometheus_metrics()
            return web.Response(text=metrics, content_type="text/plain")

        app = web.Application()
        app.router.add_get("/health", health_handler)
        app.router.add_get("/metrics", metrics_handler)
        app.router.add_get("/prometheus", prometheus_handler)

        try:
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()
            self.logger.info(f"Metrics endpoint available on port {port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics endpoint: {e}")

    def run_monitoring_loop(self) -> None:
        """Main monitoring loop"""
        self.logger.info(
            f"Starting resource monitoring for container '{self.container_name}'"
        )
        self.logger.info(f"Monitoring interval: {self.interval} seconds")
        self.logger.info(
            f"Alert thresholds: CPU {self.alert_config.cpu_warning}%/{self.alert_config.cpu_critical}%, "
            f"Memory {self.alert_config.memory_warning}%/{self.alert_config.memory_critical}%"
        )

        self.running = True
        iteration = 0

        while self.running:
            try:
                # Collect resource snapshot
                snapshot = self.collect_snapshot()

                # Store snapshot
                self.snapshots.append(snapshot)

                # Save to file
                self.save_snapshot(snapshot)

                # Check for alerts
                alerts = self.check_alerts(snapshot)

                # Log summary periodically
                iteration += 1
                if iteration % 10 == 0:  # Every 10 iterations
                    self.logger.info(
                        f"Resource summary - CPU: {snapshot.cpu_percent:.1f}%, "
                        f"Memory: {snapshot.memory_percent:.1f}% ({snapshot.memory_used_mb:.0f}MB), "
                        f"Disk: {snapshot.disk_usage_percent:.1f}%, "
                        f"Processes: {snapshot.process_count}, "
                        f"Connections: {snapshot.tcp_connections}"
                    )

                time.sleep(self.interval)

            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)

    def stop(self) -> None:
        """Stop the monitoring loop"""
        self.running = False
        self.logger.info("Monitoring stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Received signal {signum}, shutting down...")
    monitor.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    global monitor

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration from environment
    container_name = os.environ.get(
        "CONTAINER_NAME", os.environ.get("HOSTNAME", "unknown")
    )
    interval = int(os.environ.get("MONITOR_INTERVAL", "30"))
    enable_endpoint = (
        os.environ.get("ENABLE_METRICS_ENDPOINT", "true").lower() == "true"
    )
    metrics_port = int(os.environ.get("METRICS_PORT", "9090"))

    # Create alert configuration from environment
    alert_config = AlertConfig(
        cpu_warning=float(os.environ.get("ALERT_CPU_WARNING", "70")),
        cpu_critical=float(os.environ.get("ALERT_CPU_CRITICAL", "85")),
        memory_warning=float(os.environ.get("ALERT_MEMORY_WARNING", "75")),
        memory_critical=float(os.environ.get("ALERT_MEMORY_CRITICAL", "90")),
        disk_warning=float(os.environ.get("ALERT_DISK_WARNING", "80")),
        disk_critical=float(os.environ.get("ALERT_DISK_CRITICAL", "95")),
        process_warning=int(os.environ.get("ALERT_PROCESS_WARNING", "100")),
        process_critical=int(os.environ.get("ALERT_PROCESS_CRITICAL", "200")),
    )

    # Create monitor
    monitor = ContainerMonitor(
        container_name=container_name,
        interval=interval,
        alert_config=alert_config,
        enable_metrics_endpoint=enable_endpoint,
    )

    # Start metrics endpoint if enabled
    if enable_endpoint:

        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(monitor.serve_metrics_endpoint(metrics_port))
            loop.run_forever()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

    # Start monitoring
    monitor.run_monitoring_loop()


if __name__ == "__main__":
    main()
