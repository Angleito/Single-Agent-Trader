#!/usr/bin/env python3
"""
AI Trading Bot - Network Security Monitoring System
Comprehensive network monitoring with real-time threat detection and alerting
"""

import json
import logging
import queue
import re
import smtplib
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from pathlib import Path

import psutil

# Configuration
MONITORING_CONFIG = {
    "check_interval": 30,  # seconds
    "alert_thresholds": {
        "max_connections": 100,
        "max_bandwidth_mbps": 50,
        "max_failed_requests": 50,
        "max_cpu_percent": 80,
        "max_memory_percent": 85,
        "max_disk_percent": 90,
    },
    "log_file": "/var/log/trading-bot/security/network-monitor.log",
    "db_file": "/var/log/trading-bot/security/network-monitor.db",
    "alert_email": "admin@localhost",
    "smtp_server": "localhost",
    "smtp_port": 587,
}


@dataclass
class NetworkMetrics:
    timestamp: str
    active_connections: int
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    docker_containers: int
    firewall_active: bool
    failed_requests: int
    suspicious_ips: list[str]


@dataclass
class SecurityAlert:
    timestamp: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    metrics: dict
    action_taken: str | None = None


class NetworkSecurityMonitor:
    def __init__(self):
        self.setup_logging()
        self.setup_database()
        self.alert_queue = queue.Queue()
        self.suspicious_ips = set()
        self.connection_counts = {}
        self.last_metrics = None

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(MONITORING_CONFIG["log_file"]).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(MONITORING_CONFIG["log_file"]),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup SQLite database for metrics storage"""
        db_dir = Path(MONITORING_CONFIG["db_file"]).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            MONITORING_CONFIG["db_file"], check_same_thread=False
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS network_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                active_connections INTEGER,
                bytes_sent INTEGER,
                bytes_recv INTEGER,
                packets_sent INTEGER,
                packets_recv INTEGER,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                docker_containers INTEGER,
                firewall_active BOOLEAN,
                failed_requests INTEGER,
                suspicious_ips TEXT
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS security_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                metrics TEXT,
                action_taken TEXT
            )
        """
        )
        self.conn.commit()

    def get_network_stats(self) -> NetworkMetrics:
        """Collect comprehensive network statistics"""
        # Basic network stats
        net_io = psutil.net_io_counters()
        connections = psutil.net_connections()
        active_connections = len([c for c in connections if c.status == "ESTABLISHED"])

        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Docker container count
        docker_containers = self.get_docker_container_count()

        # Firewall status
        firewall_active = self.check_firewall_status()

        # Failed requests from logs
        failed_requests = self.count_failed_requests()

        # Suspicious IPs
        suspicious_ips = list(self.detect_suspicious_ips())

        return NetworkMetrics(
            timestamp=datetime.now().isoformat(),
            active_connections=active_connections,
            bytes_sent=net_io.bytes_sent,
            bytes_recv=net_io.bytes_recv,
            packets_sent=net_io.packets_sent,
            packets_recv=net_io.packets_recv,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            docker_containers=docker_containers,
            firewall_active=firewall_active,
            failed_requests=failed_requests,
            suspicious_ips=suspicious_ips,
        )

    def get_docker_container_count(self) -> int:
        """Get number of running Docker containers"""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return (
                len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            )
        except:
            return 0

    def check_firewall_status(self) -> bool:
        """Check if UFW firewall is active"""
        try:
            result = subprocess.run(
                ["ufw", "status"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "Status: active" in result.stdout
        except:
            return False

    def count_failed_requests(self) -> int:
        """Count failed requests in the last minute from access logs"""
        try:
            # Check various log files for failed requests
            log_files = [
                "/var/log/trading-bot/access.log",
                "/var/log/nginx/access.log",
                "/var/log/apache2/access.log",
            ]

            failed_count = 0
            cutoff_time = datetime.now() - timedelta(minutes=1)

            for log_file in log_files:
                if Path(log_file).exists():
                    failed_count += self.parse_access_log(log_file, cutoff_time)

            return failed_count
        except Exception as e:
            self.logger.warning(f"Error counting failed requests: {e}")
            return 0

    def parse_access_log(self, log_file: str, cutoff_time: datetime) -> int:
        """Parse access log for failed requests"""
        failed_count = 0
        try:
            with open(log_file) as f:
                for line in f.readlines()[-1000:]:  # Check last 1000 lines
                    if any(
                        code in line
                        for code in [
                            "400",
                            "401",
                            "403",
                            "404",
                            "429",
                            "500",
                            "502",
                            "503",
                        ]
                    ):
                        failed_count += 1
        except:
            pass
        return failed_count

    def detect_suspicious_ips(self) -> set:
        """Detect suspicious IP addresses from connections and logs"""
        suspicious = set()

        # Check for too many connections from single IP
        connection_counts = {}
        connections = psutil.net_connections()

        for conn in connections:
            if conn.raddr and conn.status == "ESTABLISHED":
                ip = conn.raddr.ip
                connection_counts[ip] = connection_counts.get(ip, 0) + 1

        # Flag IPs with too many connections
        for ip, count in connection_counts.items():
            if count > 20:  # Threshold for suspicious activity
                suspicious.add(ip)

        # Check fail2ban banned IPs
        suspicious.update(self.get_fail2ban_banned_ips())

        return suspicious

    def get_fail2ban_banned_ips(self) -> set:
        """Get currently banned IPs from fail2ban"""
        banned_ips = set()
        try:
            result = subprocess.run(
                ["fail2ban-client", "status"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse fail2ban output for banned IPs
                for line in result.stdout.split("\n"):
                    if "Currently banned:" in line:
                        ip_match = re.findall(
                            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", line
                        )
                        banned_ips.update(ip_match)
        except:
            pass
        return banned_ips

    def analyze_metrics(self, metrics: NetworkMetrics) -> list[SecurityAlert]:
        """Analyze metrics and generate security alerts"""
        alerts = []
        thresholds = MONITORING_CONFIG["alert_thresholds"]

        # Check connection count
        if metrics.active_connections > thresholds["max_connections"]:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="HIGH_CONNECTION_COUNT",
                    severity="HIGH",
                    message=f"High connection count: {metrics.active_connections} > {thresholds['max_connections']}",
                    metrics=asdict(metrics),
                )
            )

        # Check resource usage
        if metrics.cpu_percent > thresholds["max_cpu_percent"]:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="HIGH_CPU_USAGE",
                    severity="MEDIUM",
                    message=f"High CPU usage: {metrics.cpu_percent}% > {thresholds['max_cpu_percent']}%",
                    metrics=asdict(metrics),
                )
            )

        if metrics.memory_percent > thresholds["max_memory_percent"]:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="HIGH_MEMORY_USAGE",
                    severity="MEDIUM",
                    message=f"High memory usage: {metrics.memory_percent}% > {thresholds['max_memory_percent']}%",
                    metrics=asdict(metrics),
                )
            )

        if metrics.disk_percent > thresholds["max_disk_percent"]:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="HIGH_DISK_USAGE",
                    severity="HIGH",
                    message=f"High disk usage: {metrics.disk_percent}% > {thresholds['max_disk_percent']}%",
                    metrics=asdict(metrics),
                )
            )

        # Check firewall status
        if not metrics.firewall_active:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="FIREWALL_DOWN",
                    severity="CRITICAL",
                    message="UFW firewall is not active",
                    metrics=asdict(metrics),
                )
            )

        # Check failed requests
        if metrics.failed_requests > thresholds["max_failed_requests"]:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="HIGH_FAILED_REQUESTS",
                    severity="MEDIUM",
                    message=f"High failed request count: {metrics.failed_requests} > {thresholds['max_failed_requests']}",
                    metrics=asdict(metrics),
                )
            )

        # Check suspicious IPs
        if metrics.suspicious_ips:
            alerts.append(
                SecurityAlert(
                    timestamp=metrics.timestamp,
                    alert_type="SUSPICIOUS_IPS",
                    severity="MEDIUM",
                    message=f"Suspicious IPs detected: {', '.join(metrics.suspicious_ips)}",
                    metrics=asdict(metrics),
                )
            )

        # Check bandwidth usage (if previous metrics available)
        if self.last_metrics:
            time_diff = datetime.fromisoformat(
                metrics.timestamp
            ) - datetime.fromisoformat(self.last_metrics.timestamp)
            seconds = time_diff.total_seconds()
            if seconds > 0:
                bytes_diff = (metrics.bytes_sent + metrics.bytes_recv) - (
                    self.last_metrics.bytes_sent + self.last_metrics.bytes_recv
                )
                mbps = (bytes_diff * 8) / (seconds * 1024 * 1024)  # Convert to Mbps

                if mbps > thresholds["max_bandwidth_mbps"]:
                    alerts.append(
                        SecurityAlert(
                            timestamp=metrics.timestamp,
                            alert_type="HIGH_BANDWIDTH",
                            severity="MEDIUM",
                            message=f"High bandwidth usage: {mbps:.2f} Mbps > {thresholds['max_bandwidth_mbps']} Mbps",
                            metrics=asdict(metrics),
                        )
                    )

        return alerts

    def handle_alert(self, alert: SecurityAlert):
        """Handle security alert with appropriate actions"""
        self.logger.warning(
            f"SECURITY ALERT [{alert.severity}] {alert.alert_type}: {alert.message}"
        )

        # Store alert in database
        self.conn.execute(
            """
            INSERT INTO security_alerts (timestamp, alert_type, severity, message, metrics, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                alert.timestamp,
                alert.alert_type,
                alert.severity,
                alert.message,
                json.dumps(alert.metrics),
                alert.action_taken,
            ),
        )
        self.conn.commit()

        # Take automated actions based on alert type
        action_taken = None

        if alert.alert_type == "FIREWALL_DOWN":
            action_taken = self.restart_firewall()
        elif alert.alert_type == "HIGH_CONNECTION_COUNT":
            action_taken = self.apply_connection_limits()
        elif alert.alert_type == "SUSPICIOUS_IPS":
            action_taken = self.block_suspicious_ips(
                alert.metrics.get("suspicious_ips", [])
            )

        if action_taken:
            alert.action_taken = action_taken
            self.logger.info(f"Action taken for {alert.alert_type}: {action_taken}")

        # Send email alert for high/critical severity
        if alert.severity in ["HIGH", "CRITICAL"]:
            self.send_email_alert(alert)

    def restart_firewall(self) -> str:
        """Restart UFW firewall"""
        try:
            subprocess.run(["ufw", "--force", "enable"], check=True, timeout=30)
            return "UFW firewall restarted"
        except Exception as e:
            return f"Failed to restart firewall: {e}"

    def apply_connection_limits(self) -> str:
        """Apply iptables connection limits"""
        try:
            # Apply connection limits per IP
            subprocess.run(
                [
                    "iptables",
                    "-A",
                    "INPUT",
                    "-p",
                    "tcp",
                    "--dport",
                    "3000",
                    "-m",
                    "connlimit",
                    "--connlimit-above",
                    "10",
                    "--connlimit-mask",
                    "32",
                    "-j",
                    "REJECT",
                ],
                check=False,
                timeout=10,
            )

            subprocess.run(
                [
                    "iptables",
                    "-A",
                    "INPUT",
                    "-p",
                    "tcp",
                    "--dport",
                    "8000",
                    "-m",
                    "connlimit",
                    "--connlimit-above",
                    "20",
                    "--connlimit-mask",
                    "32",
                    "-j",
                    "REJECT",
                ],
                check=False,
                timeout=10,
            )

            return "Connection limits applied"
        except Exception as e:
            return f"Failed to apply connection limits: {e}"

    def block_suspicious_ips(self, ips: list[str]) -> str:
        """Block suspicious IPs using iptables"""
        blocked = []
        for ip in ips:
            try:
                subprocess.run(
                    ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                    timeout=10,
                    check=True,
                )
                blocked.append(ip)
            except:
                pass
        return f"Blocked IPs: {', '.join(blocked)}" if blocked else "No IPs blocked"

    def send_email_alert(self, alert: SecurityAlert):
        """Send email alert for critical issues"""
        try:
            msg = MimeMultipart()
            msg["From"] = "trading-bot-monitor@localhost"
            msg["To"] = MONITORING_CONFIG["alert_email"]
            msg["Subject"] = (
                f"[{alert.severity}] Trading Bot Security Alert: {alert.alert_type}"
            )

            body = f"""
Security Alert Detected:

Alert Type: {alert.alert_type}
Severity: {alert.severity}
Timestamp: {alert.timestamp}
Message: {alert.message}

System Metrics:
- Active Connections: {alert.metrics.get("active_connections", "N/A")}
- CPU Usage: {alert.metrics.get("cpu_percent", "N/A")}%
- Memory Usage: {alert.metrics.get("memory_percent", "N/A")}%
- Disk Usage: {alert.metrics.get("disk_percent", "N/A")}%
- Firewall Active: {alert.metrics.get("firewall_active", "N/A")}
- Docker Containers: {alert.metrics.get("docker_containers", "N/A")}

Action Taken: {alert.action_taken or "None"}

Please investigate immediately.
            """

            msg.attach(MimeText(body, "plain"))

            server = smtplib.SMTP(
                MONITORING_CONFIG["smtp_server"], MONITORING_CONFIG["smtp_port"]
            )
            server.send_message(msg)
            server.quit()

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def store_metrics(self, metrics: NetworkMetrics):
        """Store metrics in database"""
        self.conn.execute(
            """
            INSERT INTO network_metrics (
                timestamp, active_connections, bytes_sent, bytes_recv,
                packets_sent, packets_recv, cpu_percent, memory_percent,
                disk_percent, docker_containers, firewall_active,
                failed_requests, suspicious_ips
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.timestamp,
                metrics.active_connections,
                metrics.bytes_sent,
                metrics.bytes_recv,
                metrics.packets_sent,
                metrics.packets_recv,
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.disk_percent,
                metrics.docker_containers,
                metrics.firewall_active,
                metrics.failed_requests,
                json.dumps(metrics.suspicious_ips),
            ),
        )
        self.conn.commit()

    def cleanup_old_data(self):
        """Clean up old data to prevent database bloat"""
        cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

        self.conn.execute(
            "DELETE FROM network_metrics WHERE timestamp < ?", (cutoff_date,)
        )
        self.conn.execute(
            "DELETE FROM security_alerts WHERE timestamp < ?", (cutoff_date,)
        )
        self.conn.commit()

        self.logger.info("Cleaned up old monitoring data")

    def get_security_dashboard_data(self) -> dict:
        """Get data for security dashboard display"""
        # Get recent metrics
        cursor = self.conn.execute(
            """
            SELECT * FROM network_metrics
            ORDER BY timestamp DESC
            LIMIT 100
        """
        )
        metrics_data = cursor.fetchall()

        # Get recent alerts
        cursor = self.conn.execute(
            """
            SELECT * FROM security_alerts
            ORDER BY timestamp DESC
            LIMIT 20
        """
        )
        alerts_data = cursor.fetchall()

        return {
            "recent_metrics": metrics_data,
            "recent_alerts": alerts_data,
            "current_status": {
                "monitoring_active": True,
                "last_check": datetime.now().isoformat(),
            },
        }

    def run(self):
        """Main monitoring loop"""
        self.logger.info("Starting network security monitoring...")

        try:
            while True:
                # Collect metrics
                metrics = self.get_network_stats()
                self.store_metrics(metrics)

                # Analyze for alerts
                alerts = self.analyze_metrics(metrics)
                for alert in alerts:
                    self.handle_alert(alert)

                # Store current metrics for comparison
                self.last_metrics = metrics

                # Periodic cleanup
                if (
                    datetime.now().hour == 2 and datetime.now().minute < 5
                ):  # Daily at 2 AM
                    self.cleanup_old_data()

                # Wait for next check
                time.sleep(MONITORING_CONFIG["check_interval"])

        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            raise
        finally:
            self.conn.close()


def main():
    """Main entry point"""
    monitor = NetworkSecurityMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
