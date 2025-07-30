"""Log sanitization to prevent sensitive data leakage."""

import logging
import re
import traceback
from functools import wraps
from re import Pattern
from typing import Any, ClassVar


class SensitiveDataPatterns:
    """Regex patterns for identifying sensitive data."""

    # API Keys and tokens
    API_KEY_PATTERNS: ClassVar[list[tuple[Pattern, str]]] = [
        # OpenAI keys
        (re.compile(r"(sk-[a-zA-Z0-9]{48})", re.IGNORECASE), "sk-***REDACTED***"),
        # Generic API keys
        (
            re.compile(r"(api[_-]?key\s*[:=]\s*)([a-zA-Z0-9_\-]{20,})", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
        (
            re.compile(r"(bearer\s+)([a-zA-Z0-9_\-\.]{20,})", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
        (
            re.compile(r"(token\s*[:=]\s*)([a-zA-Z0-9_\-]{20,})", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
    ]

    # Private keys and secrets
    PRIVATE_KEY_PATTERNS: ClassVar[list[tuple[Pattern, str]]] = [
        # Private key blocks
        (
            re.compile(
                r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                re.DOTALL | re.IGNORECASE,
            ),
            "-----BEGIN PRIVATE KEY-----\n***REDACTED***\n-----END PRIVATE KEY-----",
        ),
        # Hex private keys (64 chars)
        (
            re.compile(
                r"((?:private[_-]?key|priv[_-]?key)\s*[:=]\s*)([a-fA-F0-9]{64})",
                re.IGNORECASE,
            ),
            r"\1***REDACTED***",
        ),
        # Mnemonic phrases (12-24 words)
        (
            re.compile(
                r"((?:mnemonic|seed[_-]?phrase)\s*[:=]\s*)((?:\w+\s+){11,23}\w+)",
                re.IGNORECASE,
            ),
            r"\1***REDACTED***",
        ),
    ]

    # Passwords and credentials
    PASSWORD_PATTERNS: ClassVar[list[tuple[Pattern, str]]] = [
        (
            re.compile(r"(password\s*[:=]\s*)([^\s]+)", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
        (re.compile(r"(pass\s*[:=]\s*)([^\s]+)", re.IGNORECASE), r"\1***REDACTED***"),
        (re.compile(r"(secret\s*[:=]\s*)([^\s]+)", re.IGNORECASE), r"\1***REDACTED***"),
        (
            re.compile(r"(credential\s*[:=]\s*)([^\s]+)", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
    ]

    # URLs with credentials
    URL_PATTERNS: ClassVar[list[tuple[Pattern, str]]] = [
        (re.compile(r"(https?://)([^:]+):([^@]+)@"), r"\1***:***@"),
        (re.compile(r"(mongodb://)([^:]+):([^@]+)@"), r"\1***:***@"),
        (re.compile(r"(redis://)([^:]+):([^@]+)@"), r"\1***:***@"),
        (re.compile(r"(postgresql://)([^:]+):([^@]+)@"), r"\1***:***@"),
    ]

    # Wallet addresses (partial masking)
    WALLET_PATTERNS: ClassVar[list[tuple[Pattern, str | callable]]] = [
        # Ethereum addresses
        (
            re.compile(r"(0x[a-fA-F0-9]{40})"),
            lambda m: f"{m.group(1)[:6]}...{m.group(1)[-4:]}",
        ),
        # Bitcoin addresses
        (
            re.compile(r"([13][a-km-zA-HJ-NP-Z1-9]{25,34})"),
            lambda m: f"{m.group(1)[:6]}...{m.group(1)[-4:]}",
        ),
        # Sui addresses
        (
            re.compile(r"(0x[a-fA-F0-9]{64})"),
            lambda m: f"{m.group(1)[:6]}...{m.group(1)[-4:]}",
        ),
    ]

    @classmethod
    def get_all_patterns(cls) -> list[tuple[Pattern, str | callable]]:
        """Get all sanitization patterns."""
        all_patterns = []
        all_patterns.extend(cls.API_KEY_PATTERNS)
        all_patterns.extend(cls.PRIVATE_KEY_PATTERNS)
        all_patterns.extend(cls.PASSWORD_PATTERNS)
        all_patterns.extend(cls.URL_PATTERNS)
        all_patterns.extend(cls.WALLET_PATTERNS)
        return all_patterns


class LogSanitizer:
    """Sanitize logs to remove sensitive information."""

    def __init__(
        self, additional_patterns: list[tuple[Pattern, str | callable]] | None = None
    ):
        """Initialize the log sanitizer.

        Args:
            additional_patterns: Additional regex patterns to sanitize
        """
        self.patterns = SensitiveDataPatterns.get_all_patterns()
        if additional_patterns:
            self.patterns.extend(additional_patterns)

    def sanitize(self, text: str) -> str:
        """Sanitize a text string.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            text = str(text)

        sanitized = text
        for pattern, replacement in self.patterns:
            if callable(replacement):
                sanitized = pattern.sub(replacement, sanitized)
            else:
                sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def sanitize_dict(self, data: dict[str, Any], depth: int = 10) -> dict[str, Any]:
        """Recursively sanitize a dictionary.

        Args:
            data: Dictionary to sanitize
            depth: Maximum recursion depth

        Returns:
            Sanitized dictionary
        """
        if depth <= 0:
            return {"error": "Max recursion depth reached"}

        sanitized = {}
        sensitive_keys = {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "private_key",
            "privatekey",
            "credential",
            "auth",
            "authorization",
            "mnemonic",
            "seed_phrase",
        }

        for key, value in data.items():
            # Check if key indicates sensitive data
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value, depth - 1)
            elif isinstance(value, list):
                sanitized[key] = [
                    (
                        self.sanitize_dict(item, depth - 1)
                        if isinstance(item, dict)
                        else self.sanitize(str(item))
                        if isinstance(item, str)
                        else item
                    )
                    for item in value
                ]
            elif isinstance(value, str):
                sanitized[key] = self.sanitize(value)
            else:
                sanitized[key] = value

        return sanitized

    def sanitize_exception(self, exc_info: tuple) -> str:
        """Sanitize exception traceback.

        Args:
            exc_info: Exception info from sys.exc_info()

        Returns:
            Sanitized traceback string
        """
        tb_lines = traceback.format_exception(*exc_info)
        sanitized_lines = [self.sanitize(line) for line in tb_lines]
        return "".join(sanitized_lines)


class SanitizingFilter(logging.Filter):
    """Logging filter that sanitizes sensitive data."""

    def __init__(self, sanitizer: LogSanitizer | None = None):
        """Initialize the filter.

        Args:
            sanitizer: LogSanitizer instance to use
        """
        super().__init__()
        self.sanitizer = sanitizer or LogSanitizer()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize log records.

        Args:
            record: Log record to filter

        Returns:
            True (always allows the record through after sanitizing)
        """
        # Sanitize message
        if hasattr(record, "msg"):
            record.msg = self.sanitizer.sanitize(str(record.msg))

        # Sanitize args
        if hasattr(record, "args") and record.args:
            if isinstance(record.args, dict):
                record.args = self.sanitizer.sanitize_dict(record.args)
            else:
                record.args = tuple(
                    self.sanitizer.sanitize(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        # Sanitize exception info
        if record.exc_info:
            record.exc_text = self.sanitizer.sanitize_exception(record.exc_info)
            record.exc_info = None  # Prevent double formatting

        return True


class SanitizingFormatter(logging.Formatter):
    """Logging formatter that sanitizes output."""

    def __init__(self, *args, sanitizer: LogSanitizer | None = None, **kwargs):
        """Initialize the formatter.

        Args:
            sanitizer: LogSanitizer instance to use
        """
        super().__init__(*args, **kwargs)
        self.sanitizer = sanitizer or LogSanitizer()

    def format(self, record: logging.LogRecord) -> str:
        """Format and sanitize log record.

        Args:
            record: Log record to format

        Returns:
            Sanitized formatted string
        """
        formatted = super().format(record)
        return self.sanitizer.sanitize(formatted)


def setup_secure_logging(
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """Set up logging with automatic sanitization.

    Args:
        logger: Logger to configure (uses root logger if None)
        level: Logging level
        format_string: Log format string

    Returns:
        Configured logger
    """
    if logger is None:
        logger = logging.getLogger()

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler with sanitization
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Add sanitizing filter
    sanitizer = LogSanitizer()
    handler.addFilter(SanitizingFilter(sanitizer))

    # Set formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = SanitizingFormatter(format_string, sanitizer=sanitizer)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def sanitize_log_message(func):
    """Decorator to sanitize function arguments and return values in logs.

    Usage:
        @sanitize_log_message
        def my_function(api_key: str) -> dict:
            # Function implementation
            pass
    """
    sanitizer = LogSanitizer()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Sanitize args for logging
        safe_args = [sanitizer.sanitize(str(arg)) for arg in args]
        safe_kwargs = {k: sanitizer.sanitize(str(v)) for k, v in kwargs.items()}

        # Log function call with sanitized args
        logger = logging.getLogger(func.__module__)
        logger.debug(
            "Calling %s with args=%s, kwargs=%s", func.__name__, safe_args, safe_kwargs
        )

        try:
            result = func(*args, **kwargs)

            # Sanitize result for logging
            if isinstance(result, dict):
                safe_result = sanitizer.sanitize_dict(result)
            else:
                safe_result = sanitizer.sanitize(str(result))

            logger.debug("%s returned: %s", func.__name__, safe_result)
            return result

        except Exception as e:
            # Log sanitized exception
            logger.exception(
                "Error in %s: %s", func.__name__, sanitizer.sanitize(str(e))
            )
            raise

    return wrapper
