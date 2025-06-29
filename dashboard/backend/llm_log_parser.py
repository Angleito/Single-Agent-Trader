#!/usr/bin/env python3
"""
Enhanced LLM Completion Log Parser for AI Trading Bot Dashboard

Specialized parser for LLM completion logs with real-time streaming,
performance metrics aggregation, and cost tracking capabilities.
"""

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Set up logging for this module
logger = logging.getLogger(__name__)


def _resolve_log_file_path(relative_path: str) -> Path | None:
    """
    Resolve log file path with fallback directory support.

    This function tries to find log files in multiple locations to handle
    scenarios where logs might be written to fallback directories due to
    permission issues.

    Args:
        relative_path: Relative path from logs directory (e.g., "llm_completions.log")

    Returns:
        Path object for the first existing log file found, or None if not found
    """
    # List of potential log file locations to check
    candidate_paths = []

    # 1. Original logs directory (relative to current working directory)
    original_logs_dir = Path("logs")
    candidate_paths.append(original_logs_dir / relative_path)

    # 2. Fallback logs directory from environment variable
    fallback_logs_dir = os.getenv("FALLBACK_LOGS_DIR")
    if fallback_logs_dir:
        candidate_paths.append(Path(fallback_logs_dir) / relative_path)

    # 3. System temp directory variations
    import tempfile

    temp_base = Path(tempfile.gettempdir())
    candidate_paths.extend(
        [
            temp_base / "ai_trading_bot_logs" / relative_path,
            temp_base / "logs" / relative_path,
        ]
    )

    # 4. Dashboard backend directory logs (if logs are written locally)
    dashboard_logs = Path(__file__).parent / "logs"
    candidate_paths.append(dashboard_logs / relative_path)

    # 5. Project root logs directory (in case we're running from a subdirectory)
    # Try to find project root by looking for pyproject.toml
    current_dir = Path(__file__).parent
    for _ in range(5):  # Limit search depth
        if (current_dir / "pyproject.toml").exists():
            candidate_paths.append(current_dir / "logs" / relative_path)
            break
        if current_dir.parent == current_dir:  # Reached filesystem root
            break
        current_dir = current_dir.parent

    # Try each candidate path and return the first one that exists
    for path in candidate_paths:
        if path.exists() and path.is_file():
            logger.debug("Found log file at: %s", path)
            return path
        logger.debug("Log file not found at: %s", path)

    # Log all attempted paths for debugging
    logger.warning(
        "Log file '%s' not found in any of these locations: %s",
        relative_path,
        [str(p) for p in candidate_paths],
    )
    return None


class EventType(Enum):
    """LLM log event types."""

    COMPLETION_REQUEST = "completion_request"
    COMPLETION_RESPONSE = "completion_response"
    TRADING_DECISION = "trading_decision"
    PERFORMANCE_METRICS = "performance_metrics"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class LLMRequest:
    """Parsed LLM request entry."""

    timestamp: datetime
    session_id: str
    request_id: str
    completion_number: int
    model: str
    temperature: float
    max_tokens: int
    prompt_length: int
    prompt_preview: str
    market_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "llm_request",
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "request_id": self.request_id,
            "completion_number": self.completion_number,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_length": self.prompt_length,
            "prompt_preview": self.prompt_preview,
            "market_context": self.market_context,
        }


@dataclass
class LLMResponse:
    """Parsed LLM response entry."""

    timestamp: datetime
    session_id: str
    request_id: str
    success: bool
    response_time_ms: float
    token_usage: dict[str, int]
    cost_estimate_usd: float
    error: str | None
    response_preview: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "llm_response",
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "request_id": self.request_id,
            "success": self.success,
            "response_time_ms": self.response_time_ms,
            "token_usage": self.token_usage,
            "cost_estimate_usd": self.cost_estimate_usd,
            "error": self.error,
            "response_preview": self.response_preview,
        }


@dataclass
class TradingDecision:
    """Parsed trading decision entry."""

    timestamp: datetime
    session_id: str
    request_id: str
    action: str
    size_pct: int
    rationale: str
    symbol: str
    current_price: float
    validation_result: str | None
    risk_assessment: str | None
    indicators: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "trading_decision",
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "request_id": self.request_id,
            "action": self.action,
            "size_pct": self.size_pct,
            "rationale": self.rationale,
            "symbol": self.symbol,
            "current_price": self.current_price,
            "validation_result": self.validation_result,
            "risk_assessment": self.risk_assessment,
            "indicators": self.indicators,
        }


@dataclass
class PerformanceMetrics:
    """Parsed performance metrics entry."""

    timestamp: datetime
    session_id: str
    total_completions: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    total_tokens: int
    total_cost_estimate_usd: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "performance_metrics",
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "total_completions": self.total_completions,
            "avg_response_time_ms": self.avg_response_time_ms,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "total_tokens": self.total_tokens,
            "total_cost_estimate_usd": self.total_cost_estimate_usd,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass
class Alert:
    """System alert for unusual patterns."""

    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "alert",
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class AlertThresholds:
    """Configurable alert thresholds."""

    max_response_time_ms: float = 30000  # 30 seconds
    max_cost_per_hour: float = 5.0  # $5/hour
    min_success_rate: float = 0.95  # 95%
    max_consecutive_failures: int = 3
    max_avg_response_time_ms: float = 10000  # 10 seconds


class LLMLogParser:
    """Enhanced parser for LLM completion logs."""

    # Regex patterns for different log types
    LOG_TIMESTAMP_PATTERN = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    )

    LLM_REQUEST_PATTERN = re.compile(r"LLM_REQUEST: (?P<json>\{.*\})")
    LLM_RESPONSE_PATTERN = re.compile(r"LLM_RESPONSE: (?P<json>\{.*\})")
    TRADING_DECISION_PATTERN = re.compile(r"TRADING_DECISION: (?P<json>\{.*\})")
    PERFORMANCE_PATTERN = re.compile(r"PERFORMANCE: (?P<json>\{.*\})")

    def __init__(
        self,
        log_file: str = "llm_completions.log",
        alert_thresholds: AlertThresholds | None = None,
    ):
        """Initialize the LLM log parser."""
        # Resolve log file path with fallback directory support
        resolved_path = _resolve_log_file_path(log_file)
        if resolved_path:
            self.log_file = resolved_path
            logger.info("Using LLM log file: %s", self.log_file)
        else:
            # Fallback to original behavior if no file found
            # Use just the filename (not full path) as it might be created later
            self.log_file = Path("logs") / log_file
            logger.warning(
                "LLM log file not found, will monitor: %s (may be created later)",
                self.log_file,
            )

        self.alert_thresholds = alert_thresholds or AlertThresholds()

        # Parsed data storage
        self.requests: list[LLMRequest] = []
        self.responses: list[LLMResponse] = []
        self.decisions: list[TradingDecision] = []
        self.metrics: list[PerformanceMetrics] = []
        self.alerts: list[Alert] = []

        # Performance tracking
        self._response_times = deque(maxlen=100)  # Last 100 response times
        self._costs_by_hour = defaultdict(float)  # Cost per hour tracking
        self._success_counts = deque(maxlen=50)  # Last 50 success/failure
        self._consecutive_failures = 0

        # Real-time streaming
        self._stop_event = threading.Event()
        self._streaming_thread: threading.Thread | None = None
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []

        # File monitoring
        self._last_position = 0
        self._last_check = time.time()

    def parse_log_file(self) -> dict[str, int]:
        """Parse the entire log file and return counts."""
        try:
            # If the log file doesn't exist, try to re-resolve it in case it was created
            # in a fallback directory after initialization
            if not self.log_file.exists():
                filename = (
                    self.log_file.name if self.log_file else "llm_completions.log"
                )
                new_resolved_path = _resolve_log_file_path(filename)

                if new_resolved_path:
                    logger.info("Found log file at new location: %s", new_resolved_path)
                    self.log_file = new_resolved_path
                else:
                    logger.warning("Log file not found: %s", self.log_file)
                    return {"requests": 0, "responses": 0, "decisions": 0, "metrics": 0}

            with self.log_file.open() as f:
                content = f.read()

            lines = content.splitlines()

            for line in lines:
                self._parse_line(line)

            return {
                "requests": len(self.requests),
                "responses": len(self.responses),
                "decisions": len(self.decisions),
                "metrics": len(self.metrics),
                "alerts": len(self.alerts),
            }

        except Exception:
            logger.exception("Error parsing log file")
            return {"requests": 0, "responses": 0, "decisions": 0, "metrics": 0}

    def _parse_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single log line."""
        try:
            # Extract JSON from different log types
            for pattern, event_type in [
                (self.LLM_REQUEST_PATTERN, EventType.COMPLETION_REQUEST),
                (self.LLM_RESPONSE_PATTERN, EventType.COMPLETION_RESPONSE),
                (self.TRADING_DECISION_PATTERN, EventType.TRADING_DECISION),
                (self.PERFORMANCE_PATTERN, EventType.PERFORMANCE_METRICS),
            ]:
                match = pattern.search(line)
                if match:
                    json_data = json.loads(match.group("json"))
                    parsed_entry = self._create_typed_entry(event_type, json_data)

                    if parsed_entry:
                        # Store the entry
                        self._store_entry(event_type, parsed_entry)

                        # Check for alerts
                        self._check_alerts(event_type, parsed_entry)

                        # Notify callbacks
                        self._notify_callbacks(parsed_entry.to_dict())

                        return parsed_entry.to_dict()

        except Exception:
            logger.exception("Error parsing line '%s...'", line[:100])
            return None

    def _create_typed_entry(self, event_type: EventType, data: dict[str, Any]):
        """Create typed entry from JSON data."""
        try:
            timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

            if event_type == EventType.COMPLETION_REQUEST:
                return LLMRequest(
                    timestamp=timestamp,
                    session_id=data["session_id"],
                    request_id=data["request_id"],
                    completion_number=data["completion_number"],
                    model=data["model"],
                    temperature=data["temperature"],
                    max_tokens=data["max_tokens"],
                    prompt_length=data["prompt_length"],
                    prompt_preview=data["prompt_preview"],
                    market_context=data.get("market_context", {}),
                )

            if event_type == EventType.COMPLETION_RESPONSE:
                return LLMResponse(
                    timestamp=timestamp,
                    session_id=data["session_id"],
                    request_id=data["request_id"],
                    success=data["success"],
                    response_time_ms=data["response_time_ms"],
                    token_usage=data.get("token_usage", {}),
                    cost_estimate_usd=data.get("cost_estimate_usd", 0.0),
                    error=data.get("error"),
                    response_preview=data.get("response_preview"),
                )

            if event_type == EventType.TRADING_DECISION:
                return TradingDecision(
                    timestamp=timestamp,
                    session_id=data["session_id"],
                    request_id=data["request_id"],
                    action=data["action"],
                    size_pct=data["size_pct"],
                    rationale=data["rationale"],
                    symbol=data["symbol"],
                    current_price=data["current_price"],
                    validation_result=data.get("validation_result"),
                    risk_assessment=data.get("risk_assessment"),
                    indicators=data.get("indicators", {}),
                )

            if event_type == EventType.PERFORMANCE_METRICS:
                return PerformanceMetrics(
                    timestamp=timestamp,
                    session_id=data["session_id"],
                    total_completions=data["total_completions"],
                    avg_response_time_ms=data["avg_response_time_ms"],
                    min_response_time_ms=data["min_response_time_ms"],
                    max_response_time_ms=data["max_response_time_ms"],
                    total_tokens=data["total_tokens"],
                    total_cost_estimate_usd=data["total_cost_estimate_usd"],
                    tokens_per_second=data["tokens_per_second"],
                )

        except Exception:
            logger.exception("Error creating typed entry for %s", event_type)
            return None

    def _store_entry(self, event_type: EventType, entry):
        """Store parsed entry in appropriate list."""
        if event_type == EventType.COMPLETION_REQUEST:
            self.requests.append(entry)
        elif event_type == EventType.COMPLETION_RESPONSE:
            self.responses.append(entry)
            self._response_times.append(entry.response_time_ms)
            self._success_counts.append(entry.success)

            # Track consecutive failures
            if entry.success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            # Track hourly costs
            hour_key = entry.timestamp.strftime("%Y-%m-%d-%H")
            self._costs_by_hour[hour_key] += entry.cost_estimate_usd

        elif event_type == EventType.TRADING_DECISION:
            self.decisions.append(entry)
        elif event_type == EventType.PERFORMANCE_METRICS:
            self.metrics.append(entry)

    def _check_alerts(self, event_type: EventType, entry):
        """Check for alert conditions."""
        alerts = []

        if event_type == EventType.COMPLETION_RESPONSE:
            # High response time alert
            if entry.response_time_ms > self.alert_thresholds.max_response_time_ms:
                alerts.append(
                    Alert(
                        timestamp=entry.timestamp,
                        level=AlertLevel.WARNING,
                        category="performance",
                        message=f"High response time: {entry.response_time_ms:.0f}ms",
                        details={
                            "response_time_ms": entry.response_time_ms,
                            "threshold": self.alert_thresholds.max_response_time_ms,
                            "request_id": entry.request_id,
                        },
                    )
                )

            # Consecutive failures alert
            if (
                self._consecutive_failures
                >= self.alert_thresholds.max_consecutive_failures
            ):
                alerts.append(
                    Alert(
                        timestamp=entry.timestamp,
                        level=AlertLevel.CRITICAL,
                        category="reliability",
                        message=f"Consecutive failures: {self._consecutive_failures}",
                        details={
                            "consecutive_failures": self._consecutive_failures,
                            "threshold": self.alert_thresholds.max_consecutive_failures,
                            "error": entry.error,
                        },
                    )
                )

            # High cost alert (hourly)
            hour_key = entry.timestamp.strftime("%Y-%m-%d-%H")
            hourly_cost = self._costs_by_hour[hour_key]
            if hourly_cost > self.alert_thresholds.max_cost_per_hour:
                alerts.append(
                    Alert(
                        timestamp=entry.timestamp,
                        level=AlertLevel.WARNING,
                        category="cost",
                        message=f"High hourly cost: ${hourly_cost:.2f}",
                        details={
                            "hourly_cost": hourly_cost,
                            "threshold": self.alert_thresholds.max_cost_per_hour,
                            "hour": hour_key,
                        },
                    )
                )

            # Low success rate alert
            if len(self._success_counts) >= 10:
                success_rate = sum(self._success_counts) / len(self._success_counts)
                if success_rate < self.alert_thresholds.min_success_rate:
                    alerts.append(
                        Alert(
                            timestamp=entry.timestamp,
                            level=AlertLevel.CRITICAL,
                            category="reliability",
                            message=f"Low success rate: {success_rate:.1%}",
                            details={
                                "success_rate": success_rate,
                                "threshold": self.alert_thresholds.min_success_rate,
                                "sample_size": len(self._success_counts),
                            },
                        )
                    )

        # Store and notify about alerts
        for alert in alerts:
            self.alerts.append(alert)
            self._notify_callbacks(alert.to_dict())

    def _notify_callbacks(self, data: dict[str, Any]):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception:
                logger.exception("Error in callback")

    def add_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Add callback for real-time notifications."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start_real_time_monitoring(self, poll_interval: float = 1.0):
        """Start real-time log monitoring."""
        if self._streaming_thread and self._streaming_thread.is_alive():
            return

        self._stop_event.clear()
        self._streaming_thread = threading.Thread(
            target=self._monitor_log_file, args=(poll_interval,), daemon=True
        )
        self._streaming_thread.start()
        logger.info("Started real-time LLM log monitoring")

    def stop_real_time_monitoring(self):
        """Stop real-time log monitoring."""
        self._stop_event.set()
        if self._streaming_thread:
            self._streaming_thread.join(timeout=5)
        logger.info("Stopped real-time LLM log monitoring")

    def _monitor_log_file(self, poll_interval: float):
        """Monitor log file for new entries."""
        try:
            # Start from end of file if it exists
            if self.log_file.exists():
                self._last_position = self.log_file.stat().st_size
                logger.info(
                    "Starting log monitoring from position %s", self._last_position
                )
            else:
                self._last_position = 0

            # Track when we last checked for new log files
            last_fallback_check = time.time()
            fallback_check_interval = 30  # Check for new log files every 30 seconds

            while not self._stop_event.is_set():
                current_time = time.time()

                # Periodically check if a new log file has appeared in fallback directories
                if (current_time - last_fallback_check) >= fallback_check_interval:
                    # Re-resolve log file path to check for new files
                    filename = (
                        self.log_file.name if self.log_file else "llm_completions.log"
                    )
                    new_resolved_path = _resolve_log_file_path(filename)

                    if new_resolved_path and new_resolved_path != self.log_file:
                        logger.info("Found new log file at: %s", new_resolved_path)
                        self.log_file = new_resolved_path
                        self._last_position = 0  # Start from beginning of new file

                    last_fallback_check = current_time

                if self.log_file.exists():
                    try:
                        # Check if file has grown
                        current_size = self.log_file.stat().st_size
                        if current_size > self._last_position:
                            # Read new content
                            with self.log_file.open() as f:
                                f.seek(self._last_position)
                                new_content = f.read()
                                self._last_position = f.tell()

                            # Parse new lines
                            new_lines = new_content.splitlines()
                            for line in new_lines:
                                if line.strip():
                                    self._parse_line(line)

                    except Exception:
                        logger.exception("Error reading log file")
                else:
                    logger.debug("Log file not found: %s", self.log_file)

                time.sleep(poll_interval)

        except Exception:
            logger.exception("Error in log monitoring")

    def get_aggregated_metrics(
        self, time_window: timedelta | None = None
    ) -> dict[str, Any]:
        """Get aggregated performance metrics."""
        if time_window:
            cutoff = datetime.now(UTC) - time_window
            responses = [r for r in self.responses if r.timestamp >= cutoff]
            requests = [r for r in self.requests if r.timestamp >= cutoff]
            decisions = [d for d in self.decisions if d.timestamp >= cutoff]
            [m for m in self.metrics if m.timestamp >= cutoff]
        else:
            responses = self.responses
            requests = self.requests
            decisions = self.decisions

        # Decision analysis
        action_counts = defaultdict(int)
        for decision in decisions:
            action_counts[decision.action] += 1

        # If we have responses, use them for metrics
        if responses:
            # Calculate metrics from responses
            response_times = [r.response_time_ms for r in responses]
            success_count = sum(1 for r in responses if r.success)
            total_cost = sum(r.cost_estimate_usd for r in responses)

            return {
                "time_window": str(time_window) if time_window else "all_time",
                "total_requests": len(requests),
                "total_responses": len(responses),
                "total_decisions": len(decisions),
                "success_rate": success_count / len(responses) if responses else 0,
                "avg_response_time_ms": (
                    sum(response_times) / len(response_times) if response_times else 0
                ),
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "total_cost_usd": total_cost,
                "avg_cost_per_request": total_cost / len(responses) if responses else 0,
                "decision_counts": dict(action_counts),
                "active_alerts": len(
                    [
                        a
                        for a in self.alerts
                        if a.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                    ]
                ),
            }

        # If we only have decisions (no request/response logs), provide decision-based metrics
        if decisions:
            # Calculate decision rate
            if decisions and len(decisions) >= 2:
                time_span = (
                    decisions[-1].timestamp - decisions[0].timestamp
                ).total_seconds()
                decisions_per_hour = (
                    len(decisions) / (time_span / 3600) if time_span > 0 else 0
                )
            else:
                decisions_per_hour = 0

            return {
                "time_window": str(time_window) if time_window else "all_time",
                "total_requests": 0,
                "total_responses": 0,
                "total_decisions": len(decisions),
                "decisions_per_hour": round(decisions_per_hour, 2),
                "decision_counts": dict(action_counts),
                "last_decision": decisions[-1].to_dict() if decisions else None,
                "active_alerts": len(
                    [
                        a
                        for a in self.alerts
                        if a.timestamp >= datetime.now(UTC) - timedelta(hours=1)
                    ]
                ),
                "no_llm_logs": True,  # Indicate that we only have decision logs
            }

        # No data at all
        return {"no_data": True, "total_decisions": 0}

    def get_recent_activity(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent activity across all event types."""
        all_events = []

        # Combine all events with timestamps
        for req in self.requests[-limit:]:
            all_events.append(req.to_dict())
        for resp in self.responses[-limit:]:
            all_events.append(resp.to_dict())
        for decision in self.decisions[-limit:]:
            all_events.append(decision.to_dict())
        for alert in self.alerts[-limit:]:
            all_events.append(alert.to_dict())

        # Sort by timestamp and return most recent
        all_events.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_events[:limit]

    def export_data(self, format_type: str = "json") -> str:
        """Export all parsed data."""
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": {
                "requests": len(self.requests),
                "responses": len(self.responses),
                "decisions": len(self.decisions),
                "metrics": len(self.metrics),
                "alerts": len(self.alerts),
            },
            "requests": [r.to_dict() for r in self.requests],
            "responses": [r.to_dict() for r in self.responses],
            "decisions": [d.to_dict() for d in self.decisions],
            "metrics": [m.to_dict() for m in self.metrics],
            "alerts": [a.to_dict() for a in self.alerts],
        }

        if format_type.lower() == "json":
            return json.dumps(data, indent=2, default=str)

        raise ValueError(f"Unsupported format: {format_type}")


# Factory function
def create_llm_log_parser(
    log_file: str = "llm_completions.log",
    alert_thresholds: AlertThresholds | None = None,
) -> LLMLogParser:
    """Create configured LLM log parser."""
    return LLMLogParser(log_file, alert_thresholds)


if __name__ == "__main__":
    # Example usage
    parser = create_llm_log_parser()

    # Parse existing logs
    counts = parser.parse_log_file()
    print(f"Parsed logs: {counts}")

    # Get metrics
    metrics = parser.get_aggregated_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Start real-time monitoring
    def log_callback(data):
        print(f"New event: {data['event_type']} at {data['timestamp']}")

    parser.add_callback(log_callback)
    parser.start_real_time_monitoring()

    # Keep running for demo
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        parser.stop_real_time_monitoring()
