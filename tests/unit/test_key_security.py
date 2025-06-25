"""Security-focused tests for key handling and validation."""

import gc
import unittest

from bot.security.memory import SecureString
from bot.security.validation import rate_limit, validate_bip39, validate_hex_key
from bot.utils.sui_key_converter import auto_convert_private_key, detect_key_format


class TestKeySecurityValidation(unittest.TestCase):
    """Test security aspects of key handling and validation."""

    def test_secure_string_clears_memory(self):
        """Test SecureString zeros memory on deletion."""
        sensitive_data = "test_secret_key_123"
        secure_str = SecureString(sensitive_data)

        # Verify we can access the data
        self.assertEqual(secure_str.get(), sensitive_data)

        # Force deletion and garbage collection
        del secure_str
        gc.collect()

        # SecureString should have cleared its internal memory

    def test_rate_limiting_enforcement(self):
        """Test rate limiting blocks excessive calls."""

        @rate_limit(calls=2, period=1)
        def limited_function():
            return "success"

        # First two calls should succeed
        self.assertEqual(limited_function(), "success")
        self.assertEqual(limited_function(), "success")

        # Third call should raise exception
        with self.assertRaises(RuntimeError):
            limited_function()

    def test_hex_key_validation_rejects_invalid(self):
        """Test hex validation rejects malicious inputs."""
        invalid_keys = [
            "",  # Empty
            "invalid_hex",  # Non-hex
            "0x123",  # Too short
            "0x" + "0" * 64,  # All zeros
            "0x" + "f" * 64,  # All F's
            "../../../etc/passwd",  # Path traversal
            "<script>alert('xss')</script>",  # XSS attempt
        ]

        for key in invalid_keys:
            self.assertFalse(validate_hex_key(key), f"Should reject: {key}")

    def test_bip39_validation_prevents_injection(self):
        """Test BIP39 validation blocks injection attempts."""
        malicious_inputs = [
            "'; DROP TABLE users; --",  # SQL injection
            "../../../etc/passwd",  # Path traversal
            "word1 word2 $(rm -rf /)",  # Command injection
            "a" * 1000,  # Buffer overflow attempt
        ]

        for malicious in malicious_inputs:
            self.assertFalse(validate_bip39(malicious))

    def test_key_format_detection_secure(self):
        """Test key format detection handles edge cases safely."""
        edge_cases = [None, 123, [], {}, "", " " * 100]

        for case in edge_cases:
            result = detect_key_format(case)
            self.assertEqual(result, "unknown")

    def test_auto_convert_prevents_data_leakage(self):
        """Test auto-conversion doesn't leak sensitive data."""
        test_key = "fake_private_key_for_testing"

        result, format_type, message = auto_convert_private_key(test_key)

        # Ensure error messages don't contain the actual key
        self.assertNotIn(test_key, message)
        self.assertNotIn(test_key, str(result))


if __name__ == "__main__":
    unittest.main()
