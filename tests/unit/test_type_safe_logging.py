"""Unit tests for type-safe logging utilities."""

import logging
from unittest.mock import Mock, patch

from bot.utils.log_format_parser import FormatSpec
from bot.utils.logger_factory import TypeSafeLogger
from bot.utils.type_safe_logging import (
    log_reconnect,
    log_required_failures,
    log_service_available,
    log_service_count,
    log_service_start,
    log_service_wait,
    log_startup_attempt,
)
from bot.utils.typed_config import ensure_float, ensure_int, ensure_str, get_typed


class TestFormatSpec:
    """Test format string parsing."""

    def test_parse_integer_formats(self):
        """Test parsing integer format specifiers."""
        fmt = "Count: %d, Value: %i, Hex: %x"
        types = FormatSpec.parse_format(fmt)
        assert types == [int, int, int]

    def test_parse_float_formats(self):
        """Test parsing float format specifiers."""
        fmt = "Price: %.2f, Rate: %e, General: %g"
        types = FormatSpec.parse_format(fmt)
        assert types == [float, float, float]

    def test_parse_string_formats(self):
        """Test parsing string format specifiers."""
        fmt = "Name: %s, Char: %c"
        types = FormatSpec.parse_format(fmt)
        assert types == [str, str]

    def test_parse_mixed_formats(self):
        """Test parsing mixed format specifiers."""
        fmt = "User %s has %d items worth $%.2f"
        types = FormatSpec.parse_format(fmt)
        assert types == [str, int, float]

    def test_convert_args_integers(self):
        """Test converting arguments to integers."""
        fmt = "Count: %d"
        args = ("10",)
        converted = FormatSpec.convert_args(fmt, args)
        assert converted == (10,)
        assert isinstance(converted[0], int)

    def test_convert_args_floats(self):
        """Test converting arguments to floats."""
        fmt = "Price: %.2f"
        args = ("10.5",)
        converted = FormatSpec.convert_args(fmt, args)
        assert converted == (10.5,)
        assert isinstance(converted[0], float)

    def test_convert_args_mixed(self):
        """Test converting mixed arguments."""
        fmt = "%s has %d items worth $%.2f"
        args = ("Alice", "5", "12.50")
        converted = FormatSpec.convert_args(fmt, args)
        assert converted == ("Alice", 5, 12.50)
        assert isinstance(converted[0], str)
        assert isinstance(converted[1], int)
        assert isinstance(converted[2], float)

    def test_convert_args_invalid(self):
        """Test converting invalid arguments."""
        fmt = "Count: %d"
        args = ("not_a_number",)
        converted = FormatSpec.convert_args(fmt, args)
        # Should fallback to string
        assert converted == ("not_a_number",)


class TestTypeSafeLogger:
    """Test type-safe logger wrapper."""

    def test_info_with_type_conversion(self):
        """Test info logging with type conversion."""
        mock_logger = Mock(spec=logging.Logger)
        safe_logger = TypeSafeLogger(mock_logger)

        safe_logger.info("Count: %d", "10")
        mock_logger.info.assert_called_once_with("Count: %d", 10)

    def test_warning_with_float_conversion(self):
        """Test warning logging with float conversion."""
        mock_logger = Mock(spec=logging.Logger)
        safe_logger = TypeSafeLogger(mock_logger)

        safe_logger.warning("Delay: %.1f seconds", "5.5")
        mock_logger.warning.assert_called_once_with("Delay: %.1f seconds", 5.5)

    def test_error_with_mixed_types(self):
        """Test error logging with mixed types."""
        mock_logger = Mock(spec=logging.Logger)
        safe_logger = TypeSafeLogger(mock_logger)

        safe_logger.error(
            "User %s failed %d times with %.2f%% rate", "test", "3", "75.5"
        )
        mock_logger.error.assert_called_once_with(
            "User %s failed %d times with %.2f%% rate", "test", 3, 75.5
        )

    def test_logging_without_args(self):
        """Test logging without arguments."""
        mock_logger = Mock(spec=logging.Logger)
        safe_logger = TypeSafeLogger(mock_logger)

        safe_logger.info("Simple message")
        mock_logger.info.assert_called_once_with("Simple message")

    def test_exception_logging(self):
        """Test exception logging."""
        mock_logger = Mock(spec=logging.Logger)
        safe_logger = TypeSafeLogger(mock_logger)

        safe_logger.exception("Error: %d", "5")
        mock_logger.exception.assert_called_once_with("Error: %d", 5)

    def test_fallback_on_conversion_error(self):
        """Test fallback when conversion fails."""
        mock_logger = Mock(spec=logging.Logger)
        safe_logger = TypeSafeLogger(mock_logger)

        # Simulate conversion failure
        with patch.object(
            FormatSpec, "convert_args", side_effect=Exception("Test error")
        ):
            safe_logger.info("Count: %d", "invalid")
            # Should fallback to string conversion
            mock_logger.info.assert_called_once_with("Count: %d", "invalid")


class TestTypedConfig:
    """Test typed configuration access."""

    def test_get_typed_int(self):
        """Test getting integer with type conversion."""
        obj = {"count": "10"}
        assert get_typed(obj, "count", 0) == 10
        assert isinstance(get_typed(obj, "count", 0), int)

    def test_get_typed_float(self):
        """Test getting float with type conversion."""
        obj = {"rate": "5.5"}
        assert get_typed(obj, "rate", 0.0) == 5.5
        assert isinstance(get_typed(obj, "rate", 0.0), float)

    def test_get_typed_string(self):
        """Test getting string."""
        obj = {"name": 123}
        assert get_typed(obj, "name", "") == "123"
        assert isinstance(get_typed(obj, "name", ""), str)

    def test_get_typed_bool(self):
        """Test getting boolean with conversion."""
        obj = {"enabled": "true"}
        assert get_typed(obj, "enabled", False) is True

        obj = {"enabled": "false"}
        assert get_typed(obj, "enabled", True) is False

    def test_get_typed_missing_key(self):
        """Test getting missing key returns default."""
        obj = {}
        assert get_typed(obj, "missing", 42) == 42

    def test_get_typed_none_value(self):
        """Test getting None value returns default."""
        obj = {"value": None}
        assert get_typed(obj, "value", 10) == 10

    def test_ensure_int(self):
        """Test ensure_int function."""
        assert ensure_int("10") == 10
        assert ensure_int(10.5) == 10
        assert ensure_int("invalid", 0) == 0

    def test_ensure_float(self):
        """Test ensure_float function."""
        assert ensure_float("10.5") == 10.5
        assert ensure_float(10) == 10.0
        assert ensure_float("invalid", 0.0) == 0.0

    def test_ensure_str(self):
        """Test ensure_str function."""
        assert ensure_str(123) == "123"
        assert ensure_str(None, "default") == "default"
        assert ensure_str("test") == "test"


class TestTypeSafeLoggingFunctions:
    """Test type-safe logging convenience functions."""

    def test_log_reconnect(self):
        """Test log_reconnect function."""
        mock_logger = Mock(spec=logging.Logger)
        log_reconnect(mock_logger, "5.5", "1", "10", "3")
        mock_logger.warning.assert_called_once_with(
            "Reconnecting in %.1fs (attempt %d/%d, failures: %d)", 5.5, 1, 10, 3
        )

    def test_log_service_wait(self):
        """Test log_service_wait function."""
        mock_logger = Mock(spec=logging.Logger)
        log_service_wait(mock_logger, "2.0", "omnisearch")
        mock_logger.info.assert_called_once_with(
            "Waiting %.1fs before starting %s...", 2.0, "omnisearch"
        )

    def test_log_service_start(self):
        """Test log_service_start function."""
        mock_logger = Mock(spec=logging.Logger)
        log_service_start(mock_logger, "bluefin", "0.005")
        mock_logger.info.assert_called_once_with(
            "✓ %s service started successfully (%.1fs)", "bluefin", 0.005
        )

    def test_log_service_available(self):
        """Test log_service_available function."""
        mock_logger = Mock(spec=logging.Logger)
        log_service_available(mock_logger, "websocket_publisher", "14.03")
        mock_logger.info.assert_called_once_with(
            "✓ %-20s: Available (%.1fs)", "websocket_publisher", 14.03
        )

    def test_log_service_count(self):
        """Test log_service_count function."""
        mock_logger = Mock(spec=logging.Logger)
        log_service_count(mock_logger, "2", "3", 66.7)
        mock_logger.info.assert_called_once_with(
            "Services available: %s/%s (%.0f%%)", 2, 3, 66.7
        )

    def test_log_required_failures(self):
        """Test log_required_failures function."""
        mock_logger = Mock(spec=logging.Logger)
        log_required_failures(mock_logger, "2")
        mock_logger.error.assert_called_once_with(
            "ERROR: %s required service(s) failed to start!", 2
        )

    def test_log_startup_attempt(self):
        """Test log_startup_attempt function."""
        mock_logger = Mock(spec=logging.Logger)
        log_startup_attempt(mock_logger, "1", "3", "Connection failed")
        mock_logger.warning.assert_called_once_with(
            "Service startup failed (attempt %s/%s): %s", 1, 3, "Connection failed"
        )
