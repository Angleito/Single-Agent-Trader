"""
Secure logging utilities to prevent sensitive data exposure.
"""

import logging
import re
from re import Pattern
from typing import Any, ClassVar


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that redacts sensitive information from log messages.
    """

    # Patterns for sensitive data
    SENSITIVE_PATTERNS: ClassVar[list[Pattern[str]]] = [
        # API Keys
        re.compile(
            r'(api[_-]?key|apikey|api_secret)[\s:=]+[\'""]?([A-Za-z0-9\-_]{20,})[\'""]?',
            re.IGNORECASE,
        ),
        re.compile(r"(sk-[A-Za-z0-9]{48,})", re.IGNORECASE),  # OpenAI keys
        re.compile(r"(pk-[A-Za-z0-9]{48,})", re.IGNORECASE),  # Public keys
        re.compile(r"(tvly-[A-Za-z0-9]{32,})", re.IGNORECASE),  # Tavily keys
        re.compile(r"(pplx-[A-Za-z0-9]{32,})", re.IGNORECASE),  # Perplexity keys
        re.compile(r"(jina_[A-Za-z0-9]{32,})", re.IGNORECASE),  # Jina keys
        re.compile(r"(fc-[A-Za-z0-9]{32,})", re.IGNORECASE),  # Firecrawl keys
        # Private Keys
        re.compile(
            r'(private[_-]?key)[\s:=]+[\'""]?([A-Za-z0-9\-_/+=]{32,})[\'""]?',
            re.IGNORECASE,
        ),
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"
        ),
        # Mnemonics and Seeds
        re.compile(
            r'(mnemonic|seed[_-]?phrase)[\s:=]+[\'""]?([a-z\s]{20,})[\'""]?',
            re.IGNORECASE,
        ),
        # Passwords
        re.compile(
            r'(password|passwd|pwd)[\s:=]+[\'""]?([^\s\'"",]+)[\'""]?', re.IGNORECASE
        ),
        # Bearer Tokens
        re.compile(r"(bearer|token)[\s:]+([A-Za-z0-9\-_.]+)", re.IGNORECASE),
        # URLs with credentials
        re.compile(r"(https?://)([^:]+):([^@]+)@", re.IGNORECASE),
        # Base64 encoded potential secrets (min 40 chars)
        re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
        # Ethereum/Sui addresses (but preserve partial for debugging)
        re.compile(r"(0x[a-fA-F0-9]{40})", re.IGNORECASE),  # Ethereum addresses
        re.compile(r"(sui[a-zA-Z0-9]{32,})", re.IGNORECASE),  # Sui addresses
        # Private keys in hex format
        re.compile(r"(0x[a-fA-F0-9]{64})", re.IGNORECASE),  # 64-char hex keys
        # Balance amounts (preserve structure but redact if large)
        re.compile(r'("balance":\s*")([0-9]{10,})(\.?[0-9]*")', re.IGNORECASE),
        # Account IDs and similar identifiers
        re.compile(r'("account[_-]?id":\s*")([a-zA-Z0-9]{16,})(")', re.IGNORECASE),
    ]

    # Safe placeholder for redacted content
    REDACTED = "[REDACTED]"

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and redact sensitive information from log records.
        """
        # Redact message
        if hasattr(record, "msg"):
            record.msg = self._redact_sensitive_data(str(record.msg))

        # Redact args if present
        if hasattr(record, "args") and record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact_if_string(v) for k, v in record.args.items()
                }
            elif isinstance(record.args, list | tuple):
                record.args = type(record.args)(
                    self._redact_if_string(arg) for arg in record.args
                )

        return True

    def _redact_if_string(self, value: Any) -> Any:
        """
        Redact sensitive data only if the value is a string.
        Preserves numeric types and other non-string values.
        """
        if isinstance(value, str):
            return self._redact_sensitive_data(value)
        return value

    def _redact_sensitive_data(self, text: str) -> str:
        """
        Redact sensitive data from text using compiled patterns.
        """
        if not text:
            return text

        for pattern in self.SENSITIVE_PATTERNS:
            text = pattern.sub(self._replacement_func, text)

        return text

    def _replacement_func(self, match: re.Match[str]) -> str:
        """
        Replacement function for regex substitution.
        """
        groups = match.groups()
        full_match = match.group(0)

        # Handle Ethereum addresses - preserve first 6 and last 4 chars
        if full_match.startswith("0x") and len(full_match) == 42:
            return f"{full_match[:6]}...{full_match[-4:]}"

        # Handle Sui addresses - preserve first 8 and last 4 chars
        if full_match.startswith("sui") and len(full_match) > 20:
            return f"{full_match[:8]}...{full_match[-4:]}"

        # Handle private keys - completely redact
        if full_match.startswith("0x") and len(full_match) == 66:
            return "[PRIVATE_KEY_REDACTED]"

        # Handle balance amounts - preserve structure but redact large amounts
        if '"balance":' in full_match.lower():
            if len(groups) >= 3:
                balance_value = groups[1]
                if len(balance_value) >= 10:  # Large balance
                    return f"{groups[0]}[LARGE_BALANCE]{groups[2]}"
                return full_match  # Keep small balances

        # Handle account IDs - preserve first 4 and last 4 chars
        elif '"account' in full_match.lower() and len(groups) >= 3:
            account_id = groups[1]
            if len(account_id) > 8:
                return f"{groups[0]}{account_id[:4]}...{account_id[-4:]}{groups[2]}"

        # For patterns with groups, keep the key name but redact the value
        if len(groups) >= 2:
            return f"{groups[0]}={self.REDACTED}"

        return self.REDACTED


def setup_secure_logging(logger_name: str | None = None) -> logging.Logger:
    """
    Set up a logger with secure filtering.

    Args:
        logger_name: Name of the logger (None for root logger)

    Returns:
        Configured logger with security filter
    """
    logger = logging.getLogger(logger_name)

    # Add sensitive data filter
    secure_filter = SensitiveDataFilter()

    # Add filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(secure_filter)

    # Also add to the logger itself
    logger.addFilter(secure_filter)

    return logger


def create_secure_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create a new secure logger with rotation support.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        max_bytes: Max size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured secure logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add secure filter
    secure_filter = SensitiveDataFilter()
    console_handler.addFilter(secure_filter)

    logger.addHandler(console_handler)

    # File handler with rotation if specified
    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(secure_filter)

        logger.addHandler(file_handler)

    return logger


def create_balance_operation_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
    include_performance_metrics: bool = True,
) -> logging.Logger:
    """
    Create a specialized secure logger for balance operations.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path for balance operations
        include_performance_metrics: Whether to include performance data in logs

    Returns:
        Configured secure logger optimized for balance operations
    """
    logger = create_secure_logger(name, level, log_file)

    # Add balance-specific formatter
    balance_formatter = BalanceOperationFormatter(include_performance_metrics)

    # Update all handlers with balance formatter
    for handler in logger.handlers:
        handler.setFormatter(balance_formatter)

    logger.info(
        "Balance operation logger initialized",
        extra={
            "logger_name": name,
            "security_enabled": True,
            "performance_metrics": include_performance_metrics,
            "sensitive_data_filtering": True,
        },
    )

    return logger


class BalanceOperationFormatter(logging.Formatter):
    """
    Custom formatter for balance operations that highlights important fields.
    """

    def __init__(self, include_performance: bool = True):
        super().__init__()
        self.include_performance = include_performance

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with balance operation emphasis."""

        # Base format
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Extract key balance fields from extra
        extra = getattr(record, "__dict__", {})
        correlation_id = extra.get("correlation_id", "N/A")
        operation = extra.get("operation", "unknown")
        duration_ms = extra.get("duration_ms")
        balance_amount = extra.get("balance_amount")
        error_category = extra.get("error_category")
        success = extra.get("success")

        # Build formatted message
        parts = [
            f"{timestamp}",
            f"[{record.levelname}]",
            f"[{record.name}]",
            f"CID:{correlation_id[:8]}",  # Short correlation ID
            f"OP:{operation}",
        ]

        # Add performance info if enabled
        if self.include_performance and duration_ms is not None:
            parts.append(f"{duration_ms:.1f}ms")

        # Add balance amount if present
        if balance_amount and balance_amount != "0":
            # Apply same security filtering as SensitiveDataFilter
            filter_instance = SensitiveDataFilter()
            safe_balance = filter_instance._redact_sensitive_data(str(balance_amount))
            parts.append(f"BAL:{safe_balance}")

        # Add status indicator
        if success is not None:
            status = "✓" if success else "✗"
            parts.append(status)
        elif error_category:
            parts.append(f"ERR:{error_category}")

        # Add the actual message
        parts.append(f"- {record.getMessage()}")

        formatted = " ".join(parts)

        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def get_balance_operation_context(
    correlation_id: str,
    operation: str,
    balance_amount: str | None = None,
    duration_ms: float | None = None,
    success: bool | None = None,
    error_category: str | None = None,
    additional_context: dict | None = None,
) -> dict[str, Any]:
    """
    Create standardized context for balance operation logging.

    Args:
        correlation_id: Unique operation identifier
        operation: Operation name
        balance_amount: Balance amount (will be filtered for security)
        duration_ms: Operation duration
        success: Whether operation succeeded
        error_category: Category of error if failed
        additional_context: Additional context fields

    Returns:
        Dictionary with standardized balance operation context
    """
    from datetime import UTC, datetime

    context: dict[str, Any] = {
        "correlation_id": correlation_id,
        "operation": operation,
        "operation_type": "balance_operation",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if balance_amount is not None:
        context["balance_amount"] = balance_amount

    if duration_ms is not None:
        context["duration_ms"] = duration_ms
        context["performance_category"] = (
            "fast"
            if duration_ms < 100
            else (
                "normal"
                if duration_ms < 1000
                else "slow"
                if duration_ms < 5000
                else "very_slow"
            )
        )

    if success is not None:
        context["success"] = success

    if error_category:
        context["error_category"] = error_category

    if additional_context:
        context.update(additional_context)

    return context
