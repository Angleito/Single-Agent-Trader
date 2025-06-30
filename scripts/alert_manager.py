#!/usr/bin/env python3
"""
Alert Manager for Test Execution and System Monitoring
Manages alerts for test failures, performance issues, and system problems.
"""

import json
import logging
import smtplib
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import requests
import yaml


@dataclass
class Alert:
    """Alert definition"""

    id: str
    title: str
    severity: str  # critical, high, medium, low, info
    source: str
    message: str
    timestamp: datetime
    metadata: dict[str, Any]
    resolved: bool = False
    resolved_at: datetime | None = None
    notification_sent: bool = False


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    condition: str
    severity: str
    description: str
    cooldown_minutes: int = 5
    escalation_minutes: int = 15
    enabled: bool = True
    tags: list[str] = None


@dataclass
class NotificationChannel:
    """Notification channel configuration"""

    name: str
    type: str  # email, webhook, slack, console
    config: dict[str, Any]
    severity_filter: list[str] = None  # Filter by severity levels
    enabled: bool = True


class AlertEvaluator:
    """Evaluates alert conditions against system metrics"""

    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.logger = logging.getLogger("alert_evaluator")

    def evaluate_test_failure_rate(
        self, hours: int = 1, threshold: float = 0.2
    ) -> Alert | None:
        """Check if test failure rate exceeds threshold"""
        try:
            test_files = list(self.logs_dir.glob("*-tests.jsonl"))
            if not test_files:
                return None

            cutoff_time = datetime.now() - timedelta(hours=hours)
            total_tests = 0
            failed_tests = 0

            for test_file in test_files:
                container = test_file.stem.replace("-tests", "")

                with open(test_file) as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            event_time = datetime.fromisoformat(event["timestamp"])

                            if event_time >= cutoff_time and event.get(
                                "event_type"
                            ) in ["pass", "fail"]:
                                total_tests += 1
                                if event["event_type"] == "fail":
                                    failed_tests += 1
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

            if total_tests == 0:
                return None

            failure_rate = failed_tests / total_tests

            if failure_rate > threshold:
                return Alert(
                    id=f"test_failure_rate_{int(time.time())}",
                    title="High Test Failure Rate",
                    severity="high" if failure_rate > 0.5 else "medium",
                    source="test_monitor",
                    message=f"Test failure rate is {failure_rate:.1%} ({failed_tests}/{total_tests}) in the last {hours}h, exceeding threshold of {threshold:.1%}",
                    timestamp=datetime.now(),
                    metadata={
                        "failure_rate": failure_rate,
                        "failed_tests": failed_tests,
                        "total_tests": total_tests,
                        "hours": hours,
                        "threshold": threshold,
                    },
                )

        except Exception as e:
            self.logger.error(f"Error evaluating test failure rate: {e}")

        return None

    def evaluate_container_health(self) -> list[Alert]:
        """Check container health status"""
        alerts = []

        try:
            # Get container status from Docker
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                check=True,
            )

            running_containers = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    container_info = json.loads(line)
                    running_containers.append(container_info["Names"])

            # Check for expected containers
            expected_containers = [
                "ai-trading-bot",
                "dashboard",
                "mcp-memory",
                "postgres",
            ]

            for expected in expected_containers:
                if not any(expected in name for name in running_containers):
                    alerts.append(
                        Alert(
                            id=f"container_down_{expected}_{int(time.time())}",
                            title=f"Container Down: {expected}",
                            severity="critical",
                            source="container_monitor",
                            message=f"Expected container '{expected}' is not running",
                            timestamp=datetime.now(),
                            metadata={
                                "container_name": expected,
                                "running_containers": running_containers,
                            },
                        )
                    )

        except subprocess.CalledProcessError as e:
            alerts.append(
                Alert(
                    id=f"docker_check_failed_{int(time.time())}",
                    title="Docker Status Check Failed",
                    severity="high",
                    source="container_monitor",
                    message=f"Failed to check Docker container status: {e}",
                    timestamp=datetime.now(),
                    metadata={"error": str(e)},
                )
            )
        except Exception as e:
            self.logger.error(f"Error evaluating container health: {e}")

        return alerts

    def evaluate_performance_metrics(self, hours: int = 1) -> list[Alert]:
        """Check performance metrics for threshold violations"""
        alerts = []

        try:
            metrics_files = list(self.logs_dir.glob("*-metrics.jsonl"))
            cutoff_time = datetime.now() - timedelta(hours=hours)

            for metrics_file in metrics_files:
                container = metrics_file.stem.replace("-metrics", "")

                cpu_values = []
                memory_values = []

                with open(metrics_file) as f:
                    for line in f:
                        try:
                            snapshot = json.loads(line)
                            snapshot_time = datetime.fromtimestamp(
                                snapshot["timestamp"]
                            )

                            if snapshot_time >= cutoff_time:
                                cpu_values.append(snapshot["cpu_percent"])
                                memory_values.append(snapshot["memory_percent"])
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

                # Check CPU usage
                if cpu_values:
                    avg_cpu = sum(cpu_values) / len(cpu_values)
                    max_cpu = max(cpu_values)

                    if max_cpu > 90:
                        alerts.append(
                            Alert(
                                id=f"high_cpu_{container}_{int(time.time())}",
                                title=f"High CPU Usage: {container}",
                                severity="critical" if max_cpu > 95 else "high",
                                source="performance_monitor",
                                message=f"Container {container} CPU usage peaked at {max_cpu:.1f}% (avg: {avg_cpu:.1f}%) in the last {hours}h",
                                timestamp=datetime.now(),
                                metadata={
                                    "container": container,
                                    "max_cpu": max_cpu,
                                    "avg_cpu": avg_cpu,
                                    "hours": hours,
                                },
                            )
                        )

                # Check memory usage
                if memory_values:
                    avg_memory = sum(memory_values) / len(memory_values)
                    max_memory = max(memory_values)

                    if max_memory > 85:
                        alerts.append(
                            Alert(
                                id=f"high_memory_{container}_{int(time.time())}",
                                title=f"High Memory Usage: {container}",
                                severity="critical" if max_memory > 95 else "high",
                                source="performance_monitor",
                                message=f"Container {container} memory usage peaked at {max_memory:.1f}% (avg: {avg_memory:.1f}%) in the last {hours}h",
                                timestamp=datetime.now(),
                                metadata={
                                    "container": container,
                                    "max_memory": max_memory,
                                    "avg_memory": avg_memory,
                                    "hours": hours,
                                },
                            )
                        )

        except Exception as e:
            self.logger.error(f"Error evaluating performance metrics: {e}")

        return alerts

    def evaluate_log_errors(
        self, hours: int = 1, error_threshold: int = 10
    ) -> list[Alert]:
        """Check for excessive log errors"""
        alerts = []

        try:
            log_files = list(self.logs_dir.glob("*-logs.jsonl"))
            cutoff_time = datetime.now() - timedelta(hours=hours)

            for log_file in log_files:
                container = log_file.stem.replace("-logs", "")
                error_count = 0
                critical_errors = []

                with open(log_file) as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            event_time = datetime.fromisoformat(event["timestamp"])

                            if event_time >= cutoff_time:
                                if event["level"] in ["ERROR", "CRITICAL"]:
                                    error_count += 1
                                    if event["level"] == "CRITICAL":
                                        critical_errors.append(event["message"])
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

                if error_count > error_threshold:
                    alerts.append(
                        Alert(
                            id=f"excessive_errors_{container}_{int(time.time())}",
                            title=f"Excessive Log Errors: {container}",
                            severity=(
                                "high"
                                if error_count > error_threshold * 2
                                else "medium"
                            ),
                            source="log_monitor",
                            message=f"Container {container} has {error_count} error log entries in the last {hours}h (threshold: {error_threshold})",
                            timestamp=datetime.now(),
                            metadata={
                                "container": container,
                                "error_count": error_count,
                                "hours": hours,
                                "threshold": error_threshold,
                                "critical_errors": critical_errors[
                                    :5
                                ],  # First 5 critical errors
                            },
                        )
                    )

        except Exception as e:
            self.logger.error(f"Error evaluating log errors: {e}")

        return alerts


class NotificationManager:
    """Manages different notification channels"""

    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels
        self.logger = logging.getLogger("notification_manager")

    def send_alert(self, alert: Alert) -> bool:
        """Send alert through appropriate channels"""
        success = True

        for channel in self.channels:
            if not channel.enabled:
                continue

            # Check severity filter
            if (
                channel.severity_filter
                and alert.severity not in channel.severity_filter
            ):
                continue

            try:
                if channel.type == "email":
                    self._send_email(alert, channel)
                elif channel.type == "webhook":
                    self._send_webhook(alert, channel)
                elif channel.type == "slack":
                    self._send_slack(alert, channel)
                elif channel.type == "console":
                    self._send_console(alert, channel)
                else:
                    self.logger.warning(
                        f"Unknown notification channel type: {channel.type}"
                    )
                    continue

                self.logger.info(f"Alert sent via {channel.name}: {alert.title}")

            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.name}: {e}")
                success = False

        return success

    def _send_email(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        config = channel.config

        msg = MIMEMultipart()
        msg["From"] = config["from"]
        msg["To"] = ", ".join(config["to"])
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"

        body = f"""
Alert Details:
- Title: {alert.title}
- Severity: {alert.severity}
- Source: {alert.source}
- Time: {alert.timestamp.isoformat()}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(config["smtp_host"], config.get("smtp_port", 587))
        if config.get("use_tls", True):
            server.starttls()
        if config.get("username") and config.get("password"):
            server.login(config["username"], config["password"])

        server.send_message(msg)
        server.quit()

    def _send_webhook(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        config = channel.config

        payload = {"alert": asdict(alert), "timestamp": alert.timestamp.isoformat()}

        headers = config.get("headers", {"Content-Type": "application/json"})

        response = requests.post(
            config["url"],
            json=payload,
            headers=headers,
            timeout=config.get("timeout", 10),
        )
        response.raise_for_status()

    def _send_slack(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        config = channel.config

        color_map = {
            "critical": "danger",
            "high": "warning",
            "medium": "warning",
            "low": "good",
            "info": "good",
        }

        payload = {
            "channel": config.get("channel", "#alerts"),
            "username": config.get("username", "Alert Bot"),
            "icon_emoji": config.get("icon", ":warning:"),
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "warning"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.upper(),
                            "short": True,
                        },
                        {"title": "Source", "value": alert.source, "short": True},
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "short": True,
                        },
                    ],
                    "ts": int(alert.timestamp.timestamp()),
                }
            ],
        }

        response = requests.post(
            config["webhook_url"], json=payload, timeout=config.get("timeout", 10)
        )
        response.raise_for_status()

    def _send_console(self, alert: Alert, channel: NotificationChannel):
        """Send console notification"""
        severity_colors = {
            "critical": "\033[91m",  # Red
            "high": "\033[93m",  # Yellow
            "medium": "\033[94m",  # Blue
            "low": "\033[92m",  # Green
            "info": "\033[96m",  # Cyan
        }

        reset_color = "\033[0m"
        color = severity_colors.get(alert.severity, "")

        print(f"\n{color}🚨 ALERT [{alert.severity.upper()}]{reset_color}")
        print(f"Title: {alert.title}")
        print(f"Source: {alert.source}")
        print(f"Time: {alert.timestamp.isoformat()}")
        print(f"Message: {alert.message}")
        print()


class AlertManager:
    """Main alert management system"""

    def __init__(self, config_path: Path = None, logs_dir: Path = None):
        self.config = self._load_config(config_path)
        self.logs_dir = logs_dir or Path("/app/logs")

        self.evaluator = AlertEvaluator(self.logs_dir)
        self.notification_manager = NotificationManager(
            self._load_notification_channels()
        )

        self.active_alerts = {}
        self.alert_history = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("alert_manager")

    def _load_config(self, config_path: Path = None) -> dict[str, Any]:
        """Load alert configuration"""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

        return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default alert configuration"""
        return {
            "evaluation_interval_seconds": 60,
            "alert_rules": [
                {
                    "name": "test_failure_rate",
                    "condition": "test_failure_rate > 0.2",
                    "severity": "high",
                    "description": "Test failure rate exceeds 20%",
                    "cooldown_minutes": 5,
                },
                {
                    "name": "container_down",
                    "condition": "container_health_check",
                    "severity": "critical",
                    "description": "Required container is not running",
                    "cooldown_minutes": 1,
                },
                {
                    "name": "high_cpu",
                    "condition": "cpu_usage > 90",
                    "severity": "high",
                    "description": "CPU usage exceeds 90%",
                    "cooldown_minutes": 10,
                },
                {
                    "name": "high_memory",
                    "condition": "memory_usage > 85",
                    "severity": "high",
                    "description": "Memory usage exceeds 85%",
                    "cooldown_minutes": 10,
                },
                {
                    "name": "excessive_errors",
                    "condition": "error_count > 10",
                    "severity": "medium",
                    "description": "Excessive error log entries",
                    "cooldown_minutes": 15,
                },
            ],
            "notification_channels": [
                {
                    "name": "console",
                    "type": "console",
                    "config": {},
                    "severity_filter": ["critical", "high", "medium"],
                    "enabled": True,
                }
            ],
        }

    def _load_notification_channels(self) -> list[NotificationChannel]:
        """Load notification channels from config"""
        channels = []

        for channel_config in self.config.get("notification_channels", []):
            channels.append(
                NotificationChannel(
                    name=channel_config["name"],
                    type=channel_config["type"],
                    config=channel_config["config"],
                    severity_filter=channel_config.get("severity_filter"),
                    enabled=channel_config.get("enabled", True),
                )
            )

        return channels

    def evaluate_alerts(self) -> list[Alert]:
        """Evaluate all alert conditions"""
        alerts = []

        # Test failure rate
        test_failure_alert = self.evaluator.evaluate_test_failure_rate()
        if test_failure_alert:
            alerts.append(test_failure_alert)

        # Container health
        container_alerts = self.evaluator.evaluate_container_health()
        alerts.extend(container_alerts)

        # Performance metrics
        performance_alerts = self.evaluator.evaluate_performance_metrics()
        alerts.extend(performance_alerts)

        # Log errors
        log_error_alerts = self.evaluator.evaluate_log_errors()
        alerts.extend(log_error_alerts)

        return alerts

    def process_alert(self, alert: Alert) -> bool:
        """Process a single alert"""
        # Check if this is a duplicate of an active alert
        for active_id, active_alert in self.active_alerts.items():
            if (
                active_alert.title == alert.title
                and active_alert.source == alert.source
                and not active_alert.resolved
            ):
                # Check cooldown
                time_since_active = (
                    datetime.now() - active_alert.timestamp
                ).total_seconds() / 60
                cooldown = 5  # Default cooldown in minutes

                if time_since_active < cooldown:
                    self.logger.debug(f"Alert in cooldown: {alert.title}")
                    return False

        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Send notification
        success = self.notification_manager.send_alert(alert)
        alert.notification_sent = success

        return success

    def run_monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting alert monitoring loop")

        interval = self.config.get("evaluation_interval_seconds", 60)

        try:
            while True:
                try:
                    # Evaluate all alerts
                    new_alerts = self.evaluate_alerts()

                    # Process each alert
                    for alert in new_alerts:
                        self.process_alert(alert)

                    # Clean up resolved alerts (older than 1 hour)
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    self.active_alerts = {
                        alert_id: alert
                        for alert_id, alert in self.active_alerts.items()
                        if alert.timestamp >= cutoff_time
                    }

                    self.logger.debug(
                        f"Alert evaluation complete. Active alerts: {len(self.active_alerts)}"
                    )

                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")

                time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("Alert monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error in monitoring loop: {e}")

    def get_alert_status(self) -> dict[str, Any]:
        """Get current alert status"""
        active_count = len([a for a in self.active_alerts.values() if not a.resolved])

        severity_counts = {}
        for alert in self.active_alerts.values():
            if not alert.resolved:
                severity_counts[alert.severity] = (
                    severity_counts.get(alert.severity, 0) + 1
                )

        recent_alerts = [asdict(alert) for alert in self.alert_history[-10:]]

        return {
            "active_alerts": active_count,
            "severity_distribution": severity_counts,
            "recent_alerts": recent_alerts,
            "total_alerts_today": len(
                [
                    a
                    for a in self.alert_history
                    if a.timestamp
                    >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                ]
            ),
        }


def create_sample_config(output_path: Path):
    """Create a sample alert configuration file"""
    config = {
        "evaluation_interval_seconds": 60,
        "alert_rules": [
            {
                "name": "test_failure_rate",
                "condition": "test_failure_rate > 0.2",
                "severity": "high",
                "description": "Test failure rate exceeds 20%",
                "cooldown_minutes": 5,
                "enabled": True,
            },
            {
                "name": "container_down",
                "condition": "container_health_check",
                "severity": "critical",
                "description": "Required container is not running",
                "cooldown_minutes": 1,
                "enabled": True,
            },
        ],
        "notification_channels": [
            {
                "name": "console",
                "type": "console",
                "config": {},
                "severity_filter": ["critical", "high", "medium"],
                "enabled": True,
            },
            {
                "name": "email_alerts",
                "type": "email",
                "config": {
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "your-email@gmail.com",
                    "password": "your-app-password",
                    "from": "alerts@yourdomain.com",
                    "to": ["admin@yourdomain.com"],
                    "use_tls": True,
                },
                "severity_filter": ["critical", "high"],
                "enabled": False,
            },
            {
                "name": "slack_alerts",
                "type": "slack",
                "config": {
                    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                    "channel": "#alerts",
                    "username": "Alert Bot",
                    "icon": ":warning:",
                },
                "severity_filter": ["critical", "high", "medium"],
                "enabled": False,
            },
            {
                "name": "webhook_alerts",
                "type": "webhook",
                "config": {
                    "url": "https://your-monitoring-system.com/webhooks/alerts",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer your-token",
                    },
                    "timeout": 10,
                },
                "enabled": False,
            },
        ],
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sample configuration created at: {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Alert Manager for Test Execution")
    parser.add_argument("--config", "-c", type=Path, help="Configuration file path")
    parser.add_argument("--logs-dir", "-d", type=Path, help="Logs directory")
    parser.add_argument(
        "--create-config", type=Path, help="Create sample configuration file"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show alert status and exit"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.create_config:
        create_sample_config(args.create_config)
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    alert_manager = AlertManager(args.config, args.logs_dir)

    if args.status:
        status = alert_manager.get_alert_status()
        print(json.dumps(status, indent=2, default=str))
        return

    alert_manager.run_monitoring_loop()


if __name__ == "__main__":
    main()
