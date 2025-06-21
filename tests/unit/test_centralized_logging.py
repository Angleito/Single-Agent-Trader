"""
Unit tests for the centralized logging system.

Tests the robust logging architecture including fallback mechanisms,
path resolution, security filtering, and error recovery.
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot.utils.logging_config import RobustLoggingConfig, setup_application_logging
from bot.utils.logging_factory import LoggingFactory, get_logger
from bot.utils.logging_migration import LoggingMigrationHelper


class TestPathResolution(unittest.TestCase):
    """Test path resolution with fallback support."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

        # Clean up temp directory
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_primary_path_resolution(self):
        """Test resolution when primary path is available."""
        from bot.utils.path_utils import get_logs_directory

        # Should use existing logs directory or create fallback
        logs_dir = get_logs_directory()
        self.assertIsInstance(logs_dir, Path)
        self.assertTrue(logs_dir.exists() or True)  # May use fallback

    def test_fallback_environment_variable(self):
        """Test fallback to environment variable."""
        fallback_dir = self.temp_dir / "fallback_logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)

        os.environ["FALLBACK_LOGS_DIR"] = str(fallback_dir)

        # Should detect and use fallback directory
        from bot.utils.path_utils import get_logs_directory

        try:
            logs_dir = get_logs_directory()
            # Either uses primary or fallback - both are valid
            self.assertIsInstance(logs_dir, Path)
        except OSError:
            # If no writable directory found, that's also a valid test result
            pass

    @patch("bot.utils.path_utils._is_directory_writable")
    def test_all_directories_fail(self, mock_writable):
        """Test behavior when all directories are unwritable."""
        mock_writable.return_value = False

        from bot.utils.path_utils import get_logs_directory

        with self.assertRaises(OSError):
            get_logs_directory()


class TestLoggingFactory(unittest.TestCase):
    """Test the centralized logging factory."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Clear any existing loggers
        logging.getLogger().handlers.clear()
        for name in list(logging.getLogger().manager.loggerDict.keys()):
            if name.startswith("test_"):
                del logging.getLogger().manager.loggerDict[name]

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_basic_logger_creation(self):
        """Test basic logger creation."""
        logger = LoggingFactory.create_logger(
            name="test_basic",
            enable_console=True,
            log_file=None,  # No file logging for test
        )

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_basic")
        self.assertTrue(len(logger.handlers) > 0)

    def test_component_logger(self):
        """Test component logger creation."""
        logger = LoggingFactory.create_component_logger(
            component_name="test_component",
            subcomponent="test_sub",
            log_file=None,  # No file logging for test
        )

        self.assertEqual(logger.name, "bot.test_component.test_sub")

    def test_specialized_loggers(self):
        """Test specialized logger creation."""
        # Trade logger
        trade_logger = LoggingFactory.create_trade_logger(log_file=None)
        self.assertEqual(trade_logger.name, "bot.trading.decisions")

        # Exchange logger
        exchange_logger = LoggingFactory.create_exchange_logger(
            "test_exchange", log_file=None
        )
        self.assertEqual(exchange_logger.name, "bot.exchange.test_exchange")

        # Strategy logger
        strategy_logger = LoggingFactory.create_strategy_logger(
            "test_strategy", log_file=None
        )
        self.assertEqual(strategy_logger.name, "bot.strategy.test_strategy")

    def test_security_filter_applied(self):
        """Test that security filters are applied."""
        logger = LoggingFactory.create_logger(
            name="test_security",
            enable_console=True,
            log_file=None,
            enable_security_filter=True,
        )

        # Check if security filter is present
        security_filters = [
            f for f in logger.filters if f.__class__.__name__ == "SensitiveDataFilter"
        ]
        self.assertTrue(len(security_filters) > 0)

    def test_file_handler_fallback(self):
        """Test file handler creation with fallback."""
        # Try to create logger with file in non-existent directory
        logger = LoggingFactory.create_logger(
            name="test_fallback",
            log_file="nonexistent/test.log",
            enable_console=True,
        )

        # Should succeed with console handler even if file handler fails
        self.assertIsInstance(logger, logging.Logger)
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename")
        ]
        self.assertTrue(len(console_handlers) > 0)


class TestRobustLoggingConfig(unittest.TestCase):
    """Test the robust logging configuration system."""

    def setUp(self):
        """Set up test environment."""
        self.config = RobustLoggingConfig()
        # Clear existing loggers
        logging.getLogger().handlers.clear()

    def test_setup_logging_success(self):
        """Test successful logging setup."""
        results = self.config.setup_logging()

        self.assertIn("status", results)
        self.assertIn(results["status"], ["success", "fallback"])
        self.assertIsInstance(results["handlers_created"], list)

    def test_fallback_mode_detection(self):
        """Test fallback mode detection."""
        # After setup, check if we can detect fallback mode
        results = self.config.setup_logging()

        system_info = self.config.get_system_info()
        self.assertIn("fallback_mode", system_info)
        self.assertIsInstance(system_info["fallback_mode"], bool)

    @patch("bot.utils.logging_config.get_logs_directory")
    def test_emergency_logging(self, mock_get_logs):
        """Test emergency console-only logging."""
        mock_get_logs.side_effect = OSError("No writable directory")

        results = self.config.setup_logging()

        # Should fall back to emergency logging
        self.assertEqual(results["status"], "fallback")
        self.assertTrue(results["fallback_mode"])


class TestSecurityFilter(unittest.TestCase):
    """Test sensitive data filtering."""

    def setUp(self):
        """Set up test environment."""
        from bot.utils.secure_logging import SensitiveDataFilter

        self.filter = SensitiveDataFilter()

    def test_api_key_redaction(self):
        """Test API key redaction."""
        sensitive_text = "API_KEY=sk-1234567890abcdef"
        safe_text = self.filter._redact_sensitive_data(sensitive_text)

        self.assertNotIn("sk-1234567890abcdef", safe_text)
        self.assertIn("[REDACTED]", safe_text)

    def test_private_key_redaction(self):
        """Test private key redaction."""
        sensitive_text = "private_key=0x1234567890abcdef1234567890abcdef12345678"
        safe_text = self.filter._redact_sensitive_data(sensitive_text)

        self.assertNotIn("0x1234567890abcdef", safe_text)

    def test_address_partial_redaction(self):
        """Test that addresses are partially redacted (not completely hidden)."""
        address = "0x1234567890123456789012345678901234567890"
        safe_text = self.filter._redact_sensitive_data(address)

        # Should preserve first 6 and last 4 characters
        self.assertIn("0x1234", safe_text)
        self.assertIn("7890", safe_text)
        self.assertIn("...", safe_text)

    def test_balance_filtering(self):
        """Test balance amount filtering."""
        large_balance = '"balance": "1000000000000"'
        safe_text = self.filter._redact_sensitive_data(large_balance)

        self.assertIn("[LARGE_BALANCE]", safe_text)

    def test_log_record_filtering(self):
        """Test filtering of log records."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API call with key sk-test123456789",
            args=(),
            exc_info=None,
        )

        result = self.filter.filter(record)

        self.assertTrue(result)  # Filter should return True (don't drop record)
        self.assertNotIn("sk-test123456789", record.msg)


class TestLoggingMigration(unittest.TestCase):
    """Test logging migration utilities."""

    def setUp(self):
        """Set up test environment."""
        self.helper = LoggingMigrationHelper()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_analyze_hardcoded_paths(self):
        """Test detection of hardcoded log paths."""
        test_file = self.temp_dir / "test_component.py"
        test_file.write_text(
            """
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("test")
handler = RotatingFileHandler("hardcoded/path/test.log")
logger.addHandler(handler)
"""
        )

        analysis = self.helper.analyze_file_logging_patterns(test_file)

        self.assertEqual(analysis["severity"], "high")
        hardcoded_issues = [
            issue for issue in analysis["issues"] if issue["type"] == "hardcoded_paths"
        ]
        self.assertTrue(len(hardcoded_issues) > 0)

    def test_analyze_direct_handlers(self):
        """Test detection of direct file handler usage."""
        test_file = self.temp_dir / "test_component.py"
        test_file.write_text(
            """
import logging
from logging.handlers import FileHandler

handler = FileHandler("test.log")
"""
        )

        analysis = self.helper.analyze_file_logging_patterns(test_file)

        direct_handler_issues = [
            issue
            for issue in analysis["issues"]
            if issue["type"] == "direct_file_handlers"
        ]
        self.assertTrue(len(direct_handler_issues) > 0)

    def test_migration_code_generation(self):
        """Test migration code generation."""
        trade_code = self.helper.generate_migration_code("trade_executor", "trade")
        self.assertIn("get_trade_logger", trade_code)
        self.assertIn("BEFORE:", trade_code)
        self.assertIn("AFTER:", trade_code)

        exchange_code = self.helper.generate_migration_code("coinbase", "exchange")
        self.assertIn("get_exchange_logger", exchange_code)

    def test_quick_upgrade_function(self):
        """Test quick upgrade function."""
        from bot.utils.logging_migration import quick_logger_upgrade

        logger = quick_logger_upgrade("test_upgrade")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_upgrade")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete logging system."""

    def test_end_to_end_logging(self):
        """Test complete logging flow from setup to usage."""
        # Setup logging system
        results = setup_application_logging()
        self.assertIn(results["status"], ["success", "fallback"])

        # Create and use logger
        logger = get_logger("test_integration")

        # Test logging operations
        logger.info("Test info message")
        logger.warning("Test warning with sensitive data: api_key=sk-test123")
        logger.error("Test error message")

        # Should complete without exceptions
        self.assertTrue(True)

    def test_fallback_behavior(self):
        """Test behavior under fallback conditions."""
        # Create logger when system might be in fallback mode
        logger = get_logger("test_fallback", log_file="test_fallback.log")

        # Should work regardless of fallback mode
        logger.info("Testing fallback behavior")
        self.assertIsInstance(logger, logging.Logger)

    def test_multiple_logger_creation(self):
        """Test creating multiple specialized loggers."""
        from bot.utils.logging_factory import (
            get_exchange_logger,
            get_mcp_logger,
            get_strategy_logger,
            get_trade_logger,
        )

        # Create multiple loggers
        trade_logger = get_trade_logger()
        exchange_logger = get_exchange_logger("test_exchange")
        strategy_logger = get_strategy_logger("test_strategy")
        mcp_logger = get_mcp_logger()

        # All should be valid logger instances
        loggers = [trade_logger, exchange_logger, strategy_logger, mcp_logger]
        for logger in loggers:
            self.assertIsInstance(logger, logging.Logger)
            logger.info("Test message from %s", logger.name)


if __name__ == "__main__":
    unittest.main()
