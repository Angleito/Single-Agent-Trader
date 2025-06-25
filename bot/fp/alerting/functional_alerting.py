"""
Functional Alerting System

This module provides a functional programming approach to alerting with pure
calculations, composable notification effects, and immutable alert state.
All alert operations are pure functions that return IO effects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from bot.fp.effects.io import IO

if TYPE_CHECKING:
    from bot.fp.effects.monitoring import HealthCheck, MetricPoint

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"
    SMS = "sms"


class AlertSeverity(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AlertRule:
    """Immutable alert rule definition"""

    id: str
    name: str
    description: str
    metric_name: str
    threshold: float
    operator: str  # >, <, >=, <=, ==, !=
    severity: AlertSeverity
    channels: list[NotificationChannel]
    cooldown_minutes: int = 15
    enabled: bool = True
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertCondition:
    """Immutable alert condition evaluation"""

    rule_id: str
    metric_value: float
    threshold: float
    operator: str
    triggered: bool
    evaluation_time: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertEvent:
    """Immutable alert event"""

    id: str
    rule: AlertRule
    condition: AlertCondition
    timestamp: datetime
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: datetime | None = None


@dataclass(frozen=True)
class NotificationConfig:
    """Immutable notification configuration"""

    channel: NotificationChannel
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    rate_limit_per_hour: int = 60


@dataclass(frozen=True)
class NotificationResult:
    """Immutable notification delivery result"""

    channel: NotificationChannel
    success: bool
    timestamp: datetime
    message: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlertingState:
    """Immutable alerting system state"""

    timestamp: datetime
    active_alerts: list[AlertEvent]
    recent_notifications: list[NotificationResult]
    rule_states: dict[str, dict[str, Any]]
    metrics_evaluated: int
    alerts_triggered: int
    notifications_sent: int


class NotificationProvider(Protocol):
    """Protocol for notification providers"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        """Send notification and return result"""
        ...


class AlertingEngine:
    """
    Functional alerting engine with pure alert evaluation and
    composable notification effects.
    """

    def __init__(self):
        """Initialize the alerting engine"""
        self._rules: dict[str, AlertRule] = {}
        self._notification_configs: dict[NotificationChannel, NotificationConfig] = {}
        self._notification_providers: dict[
            NotificationChannel, NotificationProvider
        ] = {}
        self._alert_history: list[AlertEvent] = []
        self._rule_last_triggered: dict[str, datetime] = {}
        self._notification_history: list[NotificationResult] = []

        # Initialize default notification providers
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize default notification providers"""
        self._notification_providers.update(
            {
                NotificationChannel.EMAIL: EmailNotificationProvider(),
                NotificationChannel.SLACK: SlackNotificationProvider(),
                NotificationChannel.WEBHOOK: WebhookNotificationProvider(),
                NotificationChannel.LOG: LogNotificationProvider(),
                NotificationChannel.CONSOLE: ConsoleNotificationProvider(),
                NotificationChannel.SMS: SMSNotificationProvider(),
            }
        )

    # ==============================================================================
    # Pure Alert Rule Management
    # ==============================================================================

    def add_alert_rule(self, rule: AlertRule) -> IO[None]:
        """Add an alert rule to the engine"""

        def add_rule() -> None:
            self._rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.id})")

        return IO(add_rule)

    def remove_alert_rule(self, rule_id: str) -> IO[bool]:
        """Remove an alert rule from the engine"""

        def remove_rule() -> bool:
            if rule_id in self._rules:
                rule = self._rules.pop(rule_id)
                logger.info(f"Removed alert rule: {rule.name} ({rule_id})")
                return True
            return False

        return IO(remove_rule)

    def get_alert_rule(self, rule_id: str) -> IO[AlertRule | None]:
        """Get an alert rule by ID"""

        def get_rule() -> AlertRule | None:
            return self._rules.get(rule_id)

        return IO(get_rule)

    def list_alert_rules(self) -> IO[list[AlertRule]]:
        """List all alert rules"""

        def list_rules() -> list[AlertRule]:
            return list(self._rules.values())

        return IO(list_rules)

    # ==============================================================================
    # Pure Alert Evaluation Functions
    # ==============================================================================

    def evaluate_metric(self, metric: MetricPoint) -> IO[list[AlertCondition]]:
        """Evaluate a metric against all applicable alert rules"""

        def evaluate() -> list[AlertCondition]:
            conditions = []
            evaluation_time = datetime.now(UTC)

            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                # Check if rule applies to this metric
                if rule.metric_name != metric.name:
                    continue

                # Evaluate condition
                triggered = self._evaluate_condition(
                    metric.value, rule.threshold, rule.operator
                )

                condition = AlertCondition(
                    rule_id=rule.id,
                    metric_value=metric.value,
                    threshold=rule.threshold,
                    operator=rule.operator,
                    triggered=triggered,
                    evaluation_time=evaluation_time,
                    details={
                        "metric_name": metric.name,
                        "metric_tags": metric.tags,
                        "metric_unit": metric.unit,
                        "metric_timestamp": metric.timestamp.isoformat(),
                    },
                )

                conditions.append(condition)

            return conditions

        return IO(evaluate)

    def _evaluate_condition(
        self, value: float, threshold: float, operator: str
    ) -> bool:
        """Pure function to evaluate alert condition"""
        if operator == ">":
            return value > threshold
        if operator == "<":
            return value < threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<=":
            return value <= threshold
        if operator == "==":
            return abs(value - threshold) < 1e-9  # Float equality with epsilon
        if operator == "!=":
            return abs(value - threshold) >= 1e-9
        logger.warning(f"Unknown operator: {operator}")
        return False

    def evaluate_health_check(
        self, health_check: HealthCheck
    ) -> IO[list[AlertCondition]]:
        """Evaluate health check against health-based alert rules"""

        def evaluate() -> list[AlertCondition]:
            conditions = []
            evaluation_time = datetime.now(UTC)

            # Convert health status to numeric value for evaluation
            health_value = self._health_status_to_value(health_check.status)

            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                # Check if rule applies to this health check
                if rule.metric_name not in (
                    "health_status",
                    f"health_{health_check.component}",
                ):
                    continue

                # Evaluate condition
                triggered = self._evaluate_condition(
                    health_value, rule.threshold, rule.operator
                )

                condition = AlertCondition(
                    rule_id=rule.id,
                    metric_value=health_value,
                    threshold=rule.threshold,
                    operator=rule.operator,
                    triggered=triggered,
                    evaluation_time=evaluation_time,
                    details={
                        "component": health_check.component,
                        "health_status": health_check.status.value,
                        "response_time_ms": health_check.response_time_ms,
                        "health_details": health_check.details,
                    },
                )

                conditions.append(condition)

            return conditions

        return IO(evaluate)

    def _health_status_to_value(self, status) -> float:
        """Convert health status to numeric value for evaluation"""
        from bot.fp.effects.monitoring import HealthStatus

        if status == HealthStatus.HEALTHY:
            return 1.0
        if status == HealthStatus.DEGRADED:
            return 0.5
        # UNHEALTHY
        return 0.0

    def create_alert_event(
        self, condition: AlertCondition, context: dict[str, Any] | None = None
    ) -> IO[AlertEvent | None]:
        """Create an alert event from a triggered condition"""

        def create_event() -> AlertEvent | None:
            if not condition.triggered:
                return None

            rule = self._rules.get(condition.rule_id)
            if not rule:
                return None

            # Check cooldown period
            last_triggered = self._rule_last_triggered.get(rule.id)
            if last_triggered:
                cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
                if datetime.now(UTC) - last_triggered < cooldown_delta:
                    return None  # Still in cooldown

            # Generate alert message
            message = self._generate_alert_message(rule, condition)

            # Create unique alert ID
            alert_id = f"{rule.id}_{int(condition.evaluation_time.timestamp())}"

            alert_event = AlertEvent(
                id=alert_id,
                rule=rule,
                condition=condition,
                timestamp=condition.evaluation_time,
                message=message,
                context=context or {},
            )

            # Update last triggered time
            self._rule_last_triggered[rule.id] = condition.evaluation_time

            return alert_event

        return IO(create_event)

    def _generate_alert_message(
        self, rule: AlertRule, condition: AlertCondition
    ) -> str:
        """Generate human-readable alert message"""
        return (
            f"Alert: {rule.name} - "
            f"{rule.metric_name} is {condition.metric_value} "
            f"({rule.operator} {rule.threshold})"
        )

    # ==============================================================================
    # Notification Management
    # ==============================================================================

    def configure_notification_channel(
        self, channel: NotificationChannel, config: NotificationConfig
    ) -> IO[None]:
        """Configure a notification channel"""

        def configure() -> None:
            self._notification_configs[channel] = config
            logger.info(f"Configured notification channel: {channel.value}")

        return IO(configure)

    def send_alert_notifications(
        self, alert_event: AlertEvent
    ) -> IO[list[NotificationResult]]:
        """Send notifications for an alert event to all configured channels"""

        def send_notifications() -> list[NotificationResult]:
            results = []

            for channel in alert_event.rule.channels:
                # Get channel configuration
                config = self._notification_configs.get(channel)
                if not config or not config.enabled:
                    continue

                # Check rate limiting
                if not self._check_rate_limit(channel, config):
                    continue

                # Get notification provider
                provider = self._notification_providers.get(channel)
                if not provider:
                    logger.warning(f"No provider found for channel: {channel}")
                    continue

                try:
                    # Send notification
                    result = provider.send_notification(alert_event, config).run()
                    results.append(result)

                    # Add to history
                    self._notification_history.append(result)

                except Exception as e:
                    error_result = NotificationResult(
                        channel=channel,
                        success=False,
                        timestamp=datetime.now(UTC),
                        message="Failed to send notification",
                        error=str(e),
                    )
                    results.append(error_result)
                    self._notification_history.append(error_result)

            return results

        return IO(send_notifications)

    def _check_rate_limit(
        self, channel: NotificationChannel, config: NotificationConfig
    ) -> bool:
        """Check if channel is within rate limits"""
        if config.rate_limit_per_hour <= 0:
            return True

        # Count notifications sent in the last hour
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        recent_notifications = [
            n
            for n in self._notification_history
            if n.channel == channel and n.timestamp >= one_hour_ago and n.success
        ]

        return len(recent_notifications) < config.rate_limit_per_hour

    # ==============================================================================
    # Batch Alert Processing
    # ==============================================================================

    def process_metrics_batch(self, metrics: list[MetricPoint]) -> IO[list[AlertEvent]]:
        """Process a batch of metrics and generate alert events"""

        def process_batch() -> list[AlertEvent]:
            alert_events = []

            for metric in metrics:
                try:
                    # Evaluate metric against rules
                    conditions = self.evaluate_metric(metric).run()

                    # Create alert events for triggered conditions
                    for condition in conditions:
                        alert_event = self.create_alert_event(condition).run()
                        if alert_event:
                            alert_events.append(alert_event)

                            # Send notifications
                            notification_results = self.send_alert_notifications(
                                alert_event
                            ).run()

                            logger.info(
                                f"Alert triggered: {alert_event.rule.name}, "
                                f"sent {len(notification_results)} notifications"
                            )

                except Exception as e:
                    logger.exception(f"Error processing metric {metric.name}: {e}")

            # Add to alert history
            self._alert_history.extend(alert_events)

            return alert_events

        return IO(process_batch)

    def process_health_checks_batch(
        self, health_checks: list[HealthCheck]
    ) -> IO[list[AlertEvent]]:
        """Process a batch of health checks and generate alert events"""

        def process_batch() -> list[AlertEvent]:
            alert_events = []

            for health_check in health_checks:
                try:
                    # Evaluate health check against rules
                    conditions = self.evaluate_health_check(health_check).run()

                    # Create alert events for triggered conditions
                    for condition in conditions:
                        alert_event = self.create_alert_event(
                            condition, {"health_check": health_check}
                        ).run()
                        if alert_event:
                            alert_events.append(alert_event)

                            # Send notifications
                            notification_results = self.send_alert_notifications(
                                alert_event
                            ).run()

                            logger.info(
                                f"Health alert triggered: {alert_event.rule.name}, "
                                f"sent {len(notification_results)} notifications"
                            )

                except Exception as e:
                    logger.exception(
                        f"Error processing health check {health_check.component}: {e}"
                    )

            # Add to alert history
            self._alert_history.extend(alert_events)

            return alert_events

        return IO(process_batch)

    # ==============================================================================
    # Alert State Management
    # ==============================================================================

    def get_active_alerts(self) -> IO[list[AlertEvent]]:
        """Get currently active (unresolved) alerts"""

        def get_active() -> list[AlertEvent]:
            return [alert for alert in self._alert_history if not alert.resolved]

        return IO(get_active)

    def resolve_alert(self, alert_id: str) -> IO[bool]:
        """Mark an alert as resolved"""

        def resolve() -> bool:
            for i, alert in enumerate(self._alert_history):
                if alert.id == alert_id and not alert.resolved:
                    # Create resolved version of alert
                    resolved_alert = AlertEvent(
                        id=alert.id,
                        rule=alert.rule,
                        condition=alert.condition,
                        timestamp=alert.timestamp,
                        message=alert.message,
                        context=alert.context,
                        resolved=True,
                        resolved_at=datetime.now(UTC),
                    )
                    self._alert_history[i] = resolved_alert
                    logger.info(f"Resolved alert: {alert_id}")
                    return True
            return False

        return IO(resolve)

    def get_alerting_state(self) -> IO[AlertingState]:
        """Get comprehensive alerting system state"""

        def get_state() -> AlertingState:
            active_alerts = [
                alert for alert in self._alert_history if not alert.resolved
            ]
            recent_notifications = [
                n
                for n in self._notification_history
                if n.timestamp >= datetime.now(UTC) - timedelta(hours=1)
            ]

            # Rule state statistics
            rule_states = {}
            for rule_id, rule in self._rules.items():
                last_triggered = self._rule_last_triggered.get(rule_id)
                triggered_count = len(
                    [a for a in self._alert_history if a.rule.id == rule_id]
                )

                rule_states[rule_id] = {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "last_triggered": (
                        last_triggered.isoformat() if last_triggered else None
                    ),
                    "triggered_count": triggered_count,
                    "severity": rule.severity.value,
                }

            return AlertingState(
                timestamp=datetime.now(UTC),
                active_alerts=active_alerts,
                recent_notifications=recent_notifications,
                rule_states=rule_states,
                metrics_evaluated=len(self._alert_history),  # Approximation
                alerts_triggered=len(self._alert_history),
                notifications_sent=len(self._notification_history),
            )

        return IO(get_state)

    # ==============================================================================
    # Alert Analytics
    # ==============================================================================

    def get_alert_statistics(
        self, duration: timedelta = timedelta(hours=24)
    ) -> IO[dict[str, Any]]:
        """Get alert statistics for a time period"""

        def get_stats() -> dict[str, Any]:
            cutoff_time = datetime.now(UTC) - duration
            recent_alerts = [
                alert for alert in self._alert_history if alert.timestamp >= cutoff_time
            ]

            # Alert frequency by rule
            rule_frequencies = {}
            for alert in recent_alerts:
                rule_name = alert.rule.name
                if rule_name not in rule_frequencies:
                    rule_frequencies[rule_name] = 0
                rule_frequencies[rule_name] += 1

            # Alert frequency by severity
            severity_frequencies = {}
            for alert in recent_alerts:
                severity = alert.rule.severity.value
                if severity not in severity_frequencies:
                    severity_frequencies[severity] = 0
                severity_frequencies[severity] += 1

            # Resolution statistics
            resolved_alerts = [alert for alert in recent_alerts if alert.resolved]
            resolution_times = []
            for alert in resolved_alerts:
                if alert.resolved_at:
                    resolution_time = (
                        alert.resolved_at - alert.timestamp
                    ).total_seconds()
                    resolution_times.append(resolution_time)

            avg_resolution_time = (
                sum(resolution_times) / len(resolution_times) if resolution_times else 0
            )

            # Notification statistics
            recent_notifications = [
                n for n in self._notification_history if n.timestamp >= cutoff_time
            ]

            successful_notifications = [n for n in recent_notifications if n.success]
            notification_success_rate = (
                len(successful_notifications) / len(recent_notifications) * 100
                if recent_notifications
                else 0
            )

            return {
                "period_hours": duration.total_seconds() / 3600,
                "total_alerts": len(recent_alerts),
                "active_alerts": len([a for a in recent_alerts if not a.resolved]),
                "resolved_alerts": len(resolved_alerts),
                "resolution_rate": (
                    len(resolved_alerts) / len(recent_alerts) * 100
                    if recent_alerts
                    else 0
                ),
                "average_resolution_time_seconds": avg_resolution_time,
                "alerts_by_rule": rule_frequencies,
                "alerts_by_severity": severity_frequencies,
                "total_notifications": len(recent_notifications),
                "notification_success_rate": notification_success_rate,
                "most_frequent_alerts": sorted(
                    rule_frequencies.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }

        return IO(get_stats)

    def export_alert_data(
        self, format_type: str = "json", duration: timedelta | None = None
    ) -> IO[str]:
        """Export alert data in specified format"""

        def export_data() -> str:
            if duration:
                cutoff_time = datetime.now(UTC) - duration
                alerts = [
                    alert
                    for alert in self._alert_history
                    if alert.timestamp >= cutoff_time
                ]
            else:
                alerts = self._alert_history

            if format_type.lower() == "json":
                alert_data = []
                for alert in alerts:
                    alert_data.append(
                        {
                            "id": alert.id,
                            "rule_name": alert.rule.name,
                            "severity": alert.rule.severity.value,
                            "metric_name": alert.rule.metric_name,
                            "metric_value": alert.condition.metric_value,
                            "threshold": alert.condition.threshold,
                            "operator": alert.condition.operator,
                            "timestamp": alert.timestamp.isoformat(),
                            "message": alert.message,
                            "resolved": alert.resolved,
                            "resolved_at": (
                                alert.resolved_at.isoformat()
                                if alert.resolved_at
                                else None
                            ),
                            "context": alert.context,
                        }
                    )

                return json.dumps(alert_data, indent=2)

            if format_type.lower() == "csv":
                lines = [
                    "id,rule_name,severity,metric_name,metric_value,threshold,operator,timestamp,message,resolved"
                ]
                for alert in alerts:
                    lines.append(
                        f"{alert.id},{alert.rule.name},{alert.rule.severity.value},"
                        f"{alert.rule.metric_name},{alert.condition.metric_value},"
                        f"{alert.condition.threshold},{alert.condition.operator},"
                        f"{alert.timestamp.isoformat()},{alert.message},{alert.resolved}"
                    )
                return "\\n".join(lines)

            return f"Unsupported format: {format_type}"

        return IO(export_data)


# ==============================================================================
# Notification Provider Implementations
# ==============================================================================


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        def send_email() -> NotificationResult:
            try:
                # Extract email configuration
                config.config.get("smtp_server", "localhost")
                config.config.get("smtp_port", 587)
                config.config.get("username", "")
                config.config.get("password", "")
                from_email = config.config.get("from_email", "alerts@example.com")
                to_emails = config.config.get("to_emails", [])

                if not to_emails:
                    return NotificationResult(
                        channel=NotificationChannel.EMAIL,
                        success=False,
                        timestamp=datetime.now(UTC),
                        message="No recipient emails configured",
                        error="Missing to_emails configuration",
                    )

                # Create email message
                msg = MIMEMultipart()
                msg["From"] = from_email
                msg["To"] = ", ".join(to_emails)
                msg["Subject"] = (
                    f"[{alert_event.rule.severity.value.upper()}] {alert_event.rule.name}"
                )

                body = f"""
Alert: {alert_event.rule.name}
Severity: {alert_event.rule.severity.value}
Time: {alert_event.timestamp.isoformat()}

Message: {alert_event.message}

Metric: {alert_event.rule.metric_name}
Value: {alert_event.condition.metric_value}
Threshold: {alert_event.condition.threshold}
Operator: {alert_event.condition.operator}

Alert ID: {alert_event.id}
                """

                msg.attach(MIMEText(body, "plain"))

                # Send email (simulation for safety)
                logger.info(f"Would send email to {to_emails}: {alert_event.rule.name}")

                return NotificationResult(
                    channel=NotificationChannel.EMAIL,
                    success=True,
                    timestamp=datetime.now(UTC),
                    message=f"Email sent to {len(to_emails)} recipients",
                    metadata={"recipients": to_emails},
                )

            except Exception as e:
                return NotificationResult(
                    channel=NotificationChannel.EMAIL,
                    success=False,
                    timestamp=datetime.now(UTC),
                    message="Failed to send email",
                    error=str(e),
                )

        return IO(send_email)


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        def send_slack() -> NotificationResult:
            try:
                webhook_url = config.config.get("webhook_url", "")
                channel = config.config.get("channel", "#alerts")

                if not webhook_url:
                    return NotificationResult(
                        channel=NotificationChannel.SLACK,
                        success=False,
                        timestamp=datetime.now(UTC),
                        message="No Slack webhook URL configured",
                        error="Missing webhook_url configuration",
                    )

                # Create Slack message
                color = {
                    AlertSeverity.LOW: "good",
                    AlertSeverity.MEDIUM: "warning",
                    AlertSeverity.HIGH: "danger",
                    AlertSeverity.CRITICAL: "danger",
                }.get(alert_event.rule.severity, "warning")

                {
                    "channel": channel,
                    "username": "AlertBot",
                    "attachments": [
                        {
                            "color": color,
                            "title": f"{alert_event.rule.name}",
                            "text": alert_event.message,
                            "fields": [
                                {
                                    "title": "Severity",
                                    "value": alert_event.rule.severity.value,
                                    "short": True,
                                },
                                {
                                    "title": "Metric",
                                    "value": alert_event.rule.metric_name,
                                    "short": True,
                                },
                                {
                                    "title": "Value",
                                    "value": str(alert_event.condition.metric_value),
                                    "short": True,
                                },
                                {
                                    "title": "Threshold",
                                    "value": str(alert_event.condition.threshold),
                                    "short": True,
                                },
                            ],
                            "timestamp": int(alert_event.timestamp.timestamp()),
                        }
                    ],
                }

                # Send to Slack (simulation for safety)
                logger.info(
                    f"Would send Slack message to {channel}: {alert_event.rule.name}"
                )

                return NotificationResult(
                    channel=NotificationChannel.SLACK,
                    success=True,
                    timestamp=datetime.now(UTC),
                    message=f"Slack message sent to {channel}",
                    metadata={"channel": channel},
                )

            except Exception as e:
                return NotificationResult(
                    channel=NotificationChannel.SLACK,
                    success=False,
                    timestamp=datetime.now(UTC),
                    message="Failed to send Slack message",
                    error=str(e),
                )

        return IO(send_slack)


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        def send_webhook() -> NotificationResult:
            try:
                url = config.config.get("url", "")
                method = config.config.get("method", "POST")
                config.config.get("headers", {})

                if not url:
                    return NotificationResult(
                        channel=NotificationChannel.WEBHOOK,
                        success=False,
                        timestamp=datetime.now(UTC),
                        message="No webhook URL configured",
                        error="Missing url configuration",
                    )

                # Create webhook payload
                {
                    "alert_id": alert_event.id,
                    "rule_name": alert_event.rule.name,
                    "severity": alert_event.rule.severity.value,
                    "message": alert_event.message,
                    "metric_name": alert_event.rule.metric_name,
                    "metric_value": alert_event.condition.metric_value,
                    "threshold": alert_event.condition.threshold,
                    "operator": alert_event.condition.operator,
                    "timestamp": alert_event.timestamp.isoformat(),
                    "context": alert_event.context,
                }

                # Send webhook (simulation for safety)
                logger.info(f"Would send webhook to {url}: {alert_event.rule.name}")

                return NotificationResult(
                    channel=NotificationChannel.WEBHOOK,
                    success=True,
                    timestamp=datetime.now(UTC),
                    message=f"Webhook sent to {url}",
                    metadata={"url": url, "method": method},
                )

            except Exception as e:
                return NotificationResult(
                    channel=NotificationChannel.WEBHOOK,
                    success=False,
                    timestamp=datetime.now(UTC),
                    message="Failed to send webhook",
                    error=str(e),
                )

        return IO(send_webhook)


class LogNotificationProvider(NotificationProvider):
    """Log-based notification provider"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        def send_log() -> NotificationResult:
            try:
                log_level = config.config.get("level", "warning").upper()

                # Log the alert
                log_message = (
                    f"ALERT [{alert_event.rule.severity.value.upper()}] "
                    f"{alert_event.rule.name}: {alert_event.message} "
                    f"(ID: {alert_event.id})"
                )

                if log_level == "ERROR":
                    logger.error(log_message)
                elif log_level == "WARNING":
                    logger.warning(log_message)
                elif log_level == "INFO":
                    logger.info(log_message)
                else:
                    logger.debug(log_message)

                return NotificationResult(
                    channel=NotificationChannel.LOG,
                    success=True,
                    timestamp=datetime.now(UTC),
                    message=f"Alert logged at {log_level} level",
                    metadata={"log_level": log_level},
                )

            except Exception as e:
                return NotificationResult(
                    channel=NotificationChannel.LOG,
                    success=False,
                    timestamp=datetime.now(UTC),
                    message="Failed to log alert",
                    error=str(e),
                )

        return IO(send_log)


class ConsoleNotificationProvider(NotificationProvider):
    """Console notification provider"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        def send_console() -> NotificationResult:
            try:
                # Print to console
                severity_color = {
                    AlertSeverity.LOW: "\\033[32m",  # Green
                    AlertSeverity.MEDIUM: "\\033[33m",  # Yellow
                    AlertSeverity.HIGH: "\\033[31m",  # Red
                    AlertSeverity.CRITICAL: "\\033[35m",  # Magenta
                }.get(
                    alert_event.rule.severity, "\\033[37m"
                )  # White

                reset_color = "\\033[0m"

                console_message = (
                    f"{severity_color}ALERT [{alert_event.rule.severity.value.upper()}]{reset_color} "
                    f"{alert_event.rule.name}: {alert_event.message}"
                )

                print(console_message)

                return NotificationResult(
                    channel=NotificationChannel.CONSOLE,
                    success=True,
                    timestamp=datetime.now(UTC),
                    message="Alert printed to console",
                )

            except Exception as e:
                return NotificationResult(
                    channel=NotificationChannel.CONSOLE,
                    success=False,
                    timestamp=datetime.now(UTC),
                    message="Failed to print to console",
                    error=str(e),
                )

        return IO(send_console)


class SMSNotificationProvider(NotificationProvider):
    """SMS notification provider"""

    def send_notification(
        self, alert_event: AlertEvent, config: NotificationConfig
    ) -> IO[NotificationResult]:
        def send_sms() -> NotificationResult:
            try:
                phone_numbers = config.config.get("phone_numbers", [])
                service = config.config.get("service", "twilio")

                if not phone_numbers:
                    return NotificationResult(
                        channel=NotificationChannel.SMS,
                        success=False,
                        timestamp=datetime.now(UTC),
                        message="No phone numbers configured",
                        error="Missing phone_numbers configuration",
                    )

                # Create SMS message
                (
                    f"ALERT [{alert_event.rule.severity.value.upper()}] "
                    f"{alert_event.rule.name}: {alert_event.message}"
                )

                # Send SMS (simulation for safety)
                logger.info(
                    f"Would send SMS to {len(phone_numbers)} numbers: {alert_event.rule.name}"
                )

                return NotificationResult(
                    channel=NotificationChannel.SMS,
                    success=True,
                    timestamp=datetime.now(UTC),
                    message=f"SMS sent to {len(phone_numbers)} numbers",
                    metadata={"recipients": len(phone_numbers), "service": service},
                )

            except Exception as e:
                return NotificationResult(
                    channel=NotificationChannel.SMS,
                    success=False,
                    timestamp=datetime.now(UTC),
                    message="Failed to send SMS",
                    error=str(e),
                )

        return IO(send_sms)


# ==============================================================================
# Factory Functions and Utilities
# ==============================================================================


def create_alerting_engine() -> AlertingEngine:
    """Factory function to create an alerting engine"""
    return AlertingEngine()


def create_simple_threshold_rule(
    rule_id: str,
    name: str,
    metric_name: str,
    threshold: float,
    operator: str = ">",
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    channels: list[NotificationChannel] | None = None,
) -> AlertRule:
    """Create a simple threshold-based alert rule"""
    return AlertRule(
        id=rule_id,
        name=name,
        description=f"Alert when {metric_name} {operator} {threshold}",
        metric_name=metric_name,
        threshold=threshold,
        operator=operator,
        severity=severity,
        channels=channels or [NotificationChannel.LOG, NotificationChannel.CONSOLE],
    )


def create_health_check_rule(
    rule_id: str,
    name: str,
    component: str,
    severity: AlertSeverity = AlertSeverity.HIGH,
    channels: list[NotificationChannel] | None = None,
) -> AlertRule:
    """Create a health check alert rule"""
    return AlertRule(
        id=rule_id,
        name=name,
        description=f"Alert when {component} is unhealthy",
        metric_name=f"health_{component}",
        threshold=0.5,  # Less than DEGRADED
        operator="<",
        severity=severity,
        channels=channels or [NotificationChannel.LOG, NotificationChannel.CONSOLE],
    )
