"""
Emergency stop system for critical trading conditions.

This module provides the EmergencyStopManager class which monitors multiple
risk factors and triggers automatic trading halts when dangerous conditions
are detected.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from bot.types.base_types import EmergencyStopStatus

logger = logging.getLogger(__name__)


class EmergencyStopManager:
    """
    Emergency stop system for critical trading conditions.

    Monitors multiple risk factors and triggers automatic trading halts
    when dangerous conditions are detected.
    """

    def __init__(self):
        """Initialize emergency stop manager."""
        self.stop_triggers = {
            "rapid_loss": {"threshold": 0.05, "timeframe": 300},  # 5% in 5 min
            "api_failures": {
                "threshold": 10,
                "timeframe": 600,
            },  # 10 failures in 10 min
            "position_errors": {"threshold": 3, "timeframe": 60},  # 3 errors in 1 min
            "margin_critical": {"threshold": 0.95, "timeframe": 0},  # 95% margin usage
            "consecutive_losses": {
                "threshold": 5,
                "timeframe": 3600,
            },  # 5 losses in 1 hour
        }
        self.is_stopped = False
        self.stop_reason: str | None = None
        self.stop_timestamp: datetime | None = None
        self.trigger_history: list[dict[str, Any]] = []

        logger.info("Emergency stop manager initialized")

    def check_emergency_conditions(self, metrics: dict[str, Any]) -> bool:
        """
        Check if emergency stop conditions are met.

        Args:
            metrics: Current system metrics

        Returns:
            True if emergency stop should be triggered
        """
        for trigger_name, config in self.stop_triggers.items():
            if self._evaluate_trigger(trigger_name, metrics, config):
                self.trigger_emergency_stop(
                    trigger_name, metrics.get(trigger_name, "unknown")
                )
                return True
        return False

    def _evaluate_trigger(
        self, trigger_name: str, metrics: dict[str, Any], config: dict[str, Any]
    ) -> bool:
        """Evaluate if a specific trigger condition is met."""
        current_value = metrics.get(trigger_name, 0)
        threshold = config["threshold"]
        config["timeframe"]

        if trigger_name == "rapid_loss":
            # Check if loss percentage exceeds threshold in timeframe
            recent_loss = current_value
            return recent_loss >= threshold

        if trigger_name == "api_failures":
            # Check API failure rate
            failure_count = current_value
            return failure_count >= threshold

        if trigger_name == "position_errors":
            # Check position validation errors
            error_count = current_value
            return error_count >= threshold

        if trigger_name == "margin_critical":
            # Check margin usage
            margin_usage = current_value
            return margin_usage >= threshold

        if trigger_name == "consecutive_losses":
            # Check consecutive losing trades
            consecutive_losses = current_value
            return consecutive_losses >= threshold

        return False

    def trigger_emergency_stop(self, reason: str, trigger_value: Any = None):
        """
        Trigger emergency stop with specified reason.

        Args:
            reason: Reason for emergency stop
            trigger_value: Value that triggered the stop
        """
        self.is_stopped = True
        self.stop_reason = reason
        self.stop_timestamp = datetime.now(UTC)

        trigger_record = {
            "timestamp": self.stop_timestamp,
            "reason": reason,
            "trigger_value": str(trigger_value) if trigger_value is not None else "N/A",
        }
        self.trigger_history.append(trigger_record)

        logger.critical(
            "🚨 EMERGENCY STOP TRIGGERED: %s\nTrigger value: %s\nAll trading operations halted immediately!",
            reason,
            trigger_value,
        )

        # Keep only recent trigger history
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)
        self.trigger_history = [
            t for t in self.trigger_history if t["timestamp"] >= cutoff_time
        ]

    def reset_emergency_stop(self, manual_reset: bool = False):
        """
        Reset emergency stop (should only be done manually after investigation).

        Args:
            manual_reset: Whether this is a manual reset (requires admin action)
        """
        if manual_reset:
            self.is_stopped = False
            self.stop_reason = None
            self.stop_timestamp = None
            logger.warning("⚠️  Emergency stop manually reset - trading resumed")
        else:
            logger.error("❌ Emergency stop reset requires manual intervention")

    def get_status(self) -> EmergencyStopStatus:
        """Get emergency stop status information."""
        return {
            "is_stopped": self.is_stopped,
            "stop_reason": self.stop_reason,
            "stop_timestamp": (
                self.stop_timestamp.isoformat() if self.stop_timestamp else None
            ),
            "trigger_count_24h": len(self.trigger_history),
            "last_trigger": (
                self.trigger_history[-1]["timestamp"].isoformat()
                if self.trigger_history
                else None
            ),
        }
