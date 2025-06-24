"""
Functional Alerting Module

This module provides functional programming approaches to alerting,
notification management, and alert rule evaluation.
"""

from .functional_alerting import (
    AlertEvent,
    AlertRule,
    AlertSeverity,
    AlertingEngine,
    AlertingState,
    NotificationChannel,
    NotificationConfig,
    NotificationResult,
    create_alerting_engine,
    create_health_check_rule,
    create_simple_threshold_rule,
)

__all__ = [
    "AlertEvent",
    "AlertRule", 
    "AlertSeverity",
    "AlertingEngine",
    "AlertingState",
    "NotificationChannel",
    "NotificationConfig",
    "NotificationResult",
    "create_alerting_engine",
    "create_health_check_rule",
    "create_simple_threshold_rule",
]