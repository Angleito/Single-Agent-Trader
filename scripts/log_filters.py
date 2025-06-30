"""
Log Filters for Container Monitoring
Provides filtering functionality for structured logging
"""

import logging
import re
from re import Pattern


class PerformanceFilter(logging.Filter):
    """Filter to include only performance-related log messages"""

    def __init__(self):
        super().__init__()
        # Keywords that indicate performance-related messages
        self.performance_keywords: set[str] = {
            "performance",
            "metric",
            "cpu",
            "memory",
            "disk",
            "network",
            "latency",
            "response_time",
            "throughput",
            "bandwidth",
            "execution_time",
            "duration",
            "timing",
            "benchmark",
            "load",
            "usage",
            "utilization",
            "resource",
            "efficiency",
            "speed",
            "rate",
            "bytes_per_second",
            "requests_per_second",
        }

        # Regex patterns for performance metrics
        self.performance_patterns: set[Pattern] = {
            re.compile(r"\b\d+(\.\d+)?\s*(ms|seconds?|minutes?)\b", re.IGNORECASE),
            re.compile(r"\b\d+(\.\d+)?\s*%\b"),  # Percentage values
            re.compile(r"\b\d+(\.\d+)?\s*(MB|GB|KB|bytes?)\b", re.IGNORECASE),
            re.compile(r"\bCPU\s*:\s*\d+", re.IGNORECASE),
            re.compile(r"\bMemory\s*:\s*\d+", re.IGNORECASE),
            re.compile(r"\bDisk\s*:\s*\d+", re.IGNORECASE),
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if record should be logged"""
        message = record.getMessage().lower()

        # Check for performance keywords
        for keyword in self.performance_keywords:
            if keyword in message:
                return True

        # Check for performance patterns
        original_message = record.getMessage()
        for pattern in self.performance_patterns:
            if pattern.search(original_message):
                return True

        # Check logger name for performance-related loggers
        if any(
            perf_logger in record.name.lower()
            for perf_logger in ["performance", "monitor", "metric"]
        ):
            return True

        return False


class AlertFilter(logging.Filter):
    """Filter to include only alert and critical messages"""

    def __init__(self):
        super().__init__()
        # Keywords that indicate alert conditions
        self.alert_keywords: set[str] = {
            "alert",
            "critical",
            "error",
            "failure",
            "failed",
            "exception",
            "warning",
            "danger",
            "emergency",
            "urgent",
            "high",
            "exceeded",
            "limit",
            "threshold",
            "overload",
            "unavailable",
            "down",
            "timeout",
            "crash",
            "abort",
            "kill",
            "dead",
            "leak",
            "overflow",
            "underflow",
            "bottleneck",
            "congestion",
        }

        # Regex patterns for alert conditions
        self.alert_patterns: set[Pattern] = {
            re.compile(
                r"\b(critical|error|fail|exception|alert|warning)\b", re.IGNORECASE
            ),
            re.compile(
                r"\b\d+(\.\d+)?\s*%\s*(usage|used).*?(>|>=|above|exceeds?)\s*\d+",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(memory|cpu|disk).*?(high|low|critical|full)\b", re.IGNORECASE
            ),
            re.compile(
                r"\b(connection|network|service).*?(lost|failed|down|unavailable)\b",
                re.IGNORECASE,
            ),
        }

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if record should be logged"""
        # Always include WARNING and above
        if record.levelno >= logging.WARNING:
            return True

        message = record.getMessage().lower()

        # Check for alert keywords
        for keyword in self.alert_keywords:
            if keyword in message:
                return True

        # Check for alert patterns
        original_message = record.getMessage()
        for pattern in self.alert_patterns:
            if pattern.search(original_message):
                return True

        # Check logger name for alert-related loggers
        if any(
            alert_logger in record.name.lower()
            for alert_logger in ["alert", "error", "critical"]
        ):
            return True

        return False


class RateLimitFilter(logging.Filter):
    """Filter to rate limit high-frequency log messages"""

    def __init__(self, max_rate: int = 10, time_window: int = 60):
        super().__init__()
        self.max_rate = max_rate
        self.time_window = time_window
        self.message_counts = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if record should be logged (not rate limited)"""
        import time

        current_time = time.time()
        message_key = f"{record.name}:{record.levelno}:{record.getMessage()[:100]}"

        # Clean old entries
        cutoff_time = current_time - self.time_window
        self.message_counts = {
            key: (count, timestamp)
            for key, (count, timestamp) in self.message_counts.items()
            if timestamp > cutoff_time
        }

        # Check current message rate
        if message_key in self.message_counts:
            count, first_timestamp = self.message_counts[message_key]
            if count >= self.max_rate:
                return False  # Rate limited
            self.message_counts[message_key] = (count + 1, first_timestamp)
        else:
            self.message_counts[message_key] = (1, current_time)

        return True


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from log messages"""

    def __init__(self):
        super().__init__()
        # Patterns for sensitive data
        self.sensitive_patterns = [
            (re.compile(r"\b[A-Za-z0-9]{64}\b"), "[REDACTED_API_KEY]"),  # API keys
            (re.compile(r"\b[A-Za-z0-9+/]{88}==\b"), "[REDACTED_TOKEN]"),  # JWT tokens
            (
                re.compile(r"\bsk-[A-Za-z0-9]{48}\b"),
                "[REDACTED_OPENAI_KEY]",
            ),  # OpenAI keys
            (
                re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
                "[REDACTED_ETH_ADDRESS]",
            ),  # Ethereum addresses
            (
                re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"),
                "[REDACTED_BTC_ADDRESS]",
            ),  # Bitcoin addresses
            (
                re.compile(r'"password"\s*:\s*"[^"]*"', re.IGNORECASE),
                '"password": "[REDACTED]"',
            ),
            (
                re.compile(r'"secret"\s*:\s*"[^"]*"', re.IGNORECASE),
                '"secret": "[REDACTED]"',
            ),
            (re.compile(r'"key"\s*:\s*"[^"]*"', re.IGNORECASE), '"key": "[REDACTED]"'),
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log record"""
        # Modify the message to redact sensitive information
        message = record.getMessage()

        for pattern, replacement in self.sensitive_patterns:
            message = pattern.sub(replacement, message)

        # Update the record's message
        record.msg = message
        record.args = ()

        return True


class ContainerContextFilter(logging.Filter):
    """Add container context information to log records"""

    def __init__(self, container_name: str = None):
        super().__init__()
        import os

        self.container_name = container_name or os.environ.get(
            "CONTAINER_NAME", os.environ.get("HOSTNAME", "unknown")
        )
        self.process_id = os.getpid()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add container context to log record"""
        record.container_name = self.container_name
        record.process_id = self.process_id

        return True
