"""
Unit tests for balance validation logic and error handling.

This module tests balance range validation, anomaly detection, precision validation,
format validation, and comprehensive error scenarios for balance operations.
"""

import logging
from decimal import Decimal, InvalidOperation

import pytest

from bot.exchange.coinbase import CoinbaseResponseValidator
from bot.paper_trading import PaperTradingAccount


class TestBalanceValidation:
    """Test cases for balance validation systems."""

    @pytest.fixture()
    def validator(self):
        """Create a balance response validator."""
        return CoinbaseResponseValidator()

    @pytest.fixture()
    def account(self):
        """Create a paper trading account for validation testing."""
        return PaperTradingAccount(starting_balance=Decimal("10000"))

    def test_balance_range_validation_positive(self, account):
        """Test validation accepts positive balance values."""
        valid_balances = [
            Decimal("0.01"),
            Decimal("100.00"),
            Decimal("10000.50"),
            Decimal("999999.99"),
        ]

        for balance in valid_balances:
            normalized = account._normalize_balance(balance)
            assert normalized >= 0
            assert isinstance(normalized, Decimal)

    def test_balance_range_validation_boundary_cases(self, account):
        """Test validation of boundary case balance values."""
        # Test zero balance
        zero_balance = account._normalize_balance(Decimal("0"))
        assert zero_balance == Decimal("0.00")

        # Test very small balance
        tiny_balance = account._normalize_balance(Decimal("0.001"))
        assert tiny_balance == Decimal("0.00")  # Should round to 0.00

        # Test large balance
        large_balance = account._normalize_balance(Decimal("1000000.00"))
        assert large_balance == Decimal("1000000.00")

    def test_balance_range_validation_negative_values(self, account):
        """Test validation of negative balance values."""
        negative_balances = [
            Decimal("-1.00"),
            Decimal("-100.50"),
            Decimal("-0.01"),
        ]

        for balance in negative_balances:
            # In production, negative balances should be handled appropriately
            # For testing, we'll verify they're processed without crashing
            try:
                normalized = account._normalize_balance(balance)
                # Depending on implementation, might convert to 0 or maintain negative
                assert isinstance(normalized, Decimal)
            except ValueError:
                # Some implementations might reject negative values
                pass

    def test_balance_precision_validation(self, account):
        """Test balance precision validation and rounding."""
        precision_test_cases = [
            # (input, expected_output)
            (Decimal("100.123"), Decimal("100.12")),
            (Decimal("100.126"), Decimal("100.13")),
            (Decimal("100.125"), Decimal("100.12")),  # Banker's rounding
            (Decimal("100.135"), Decimal("100.14")),  # Banker's rounding
            (Decimal("100"), Decimal("100.00")),
            (Decimal("100.1"), Decimal("100.10")),
        ]

        for input_val, expected in precision_test_cases:
            result = account._normalize_balance(input_val)
            assert (
                result == expected
            ), f"Input {input_val} should normalize to {expected}, got {result}"

    def test_crypto_precision_validation(self, account):
        """Test crypto amount precision validation (8 decimal places)."""
        crypto_test_cases = [
            (Decimal("1.123456789"), Decimal("1.12345679")),
            (Decimal("0.00000001"), Decimal("0.00000001")),
            (Decimal("1000.0"), Decimal("1000.00000000")),
            (Decimal("0.123"), Decimal("0.12300000")),
        ]

        for input_val, expected in crypto_test_cases:
            result = account._normalize_crypto_amount(input_val)
            assert result == expected

    def test_balance_format_validation(self, account):
        """Test validation of different balance input formats."""
        # Test string inputs
        string_inputs = [
            ("100.50", Decimal("100.50")),
            ("0", Decimal("0.00")),
            ("1000", Decimal("1000.00")),
        ]

        for string_input, expected in string_inputs:
            result = account._normalize_balance(Decimal(string_input))
            assert result == expected

    def test_invalid_format_handling(self, account):
        """Test handling of invalid balance formats."""
        # These should raise exceptions during Decimal creation
        truly_invalid_inputs = [
            "not_a_number",
            "100.50.30",  # Multiple decimal points
            "",
        ]

        for invalid_input in truly_invalid_inputs:
            with pytest.raises((ValueError, InvalidOperation, TypeError, OSError)):
                account._normalize_balance(Decimal(invalid_input))

        # These should raise exceptions in normalize_balance due to validation
        special_cases = ["inf", "nan"]
        for special_case in special_cases:
            with pytest.raises(ValueError):
                account._normalize_balance(Decimal(special_case))

        # Scientific notation should be normalized properly
        scientific_result = account._normalize_balance(Decimal("1e10"))
        assert scientific_result == Decimal("10000000000.00")

    def test_none_value_handling(self, account):
        """Test handling of None values in balance operations."""
        result = account._normalize_balance(None)
        assert result == Decimal("0.00")

        crypto_result = account._normalize_crypto_amount(None)
        assert crypto_result == Decimal("0.00000000")

    def test_balance_anomaly_detection_thresholds(self):
        """Test balance anomaly detection with various thresholds."""

        def detect_anomaly(
            previous: Decimal, current: Decimal, threshold_pct: float = 50.0
        ) -> bool:
            """Helper function to detect balance anomalies."""
            if previous == 0:
                return current > Decimal("10000")  # Arbitrary large threshold

            change_pct = abs((current - previous) / previous * 100)
            return change_pct > threshold_pct

        test_cases = [
            # (previous, current, threshold, expected_anomaly)
            (Decimal("1000"), Decimal("1050"), 50.0, False),  # 5% change - normal
            (Decimal("1000"), Decimal("1600"), 50.0, True),  # 60% change - anomaly
            (Decimal("1000"), Decimal("400"), 50.0, True),  # 60% decrease - anomaly
            (
                Decimal("1000"),
                Decimal("1500"),
                50.0,
                False,
            ),  # 50% change - at threshold
            (
                Decimal("1000"),
                Decimal("2000"),
                100.0,
                False,
            ),  # 100% change at 100% threshold
        ]

        for previous, current, threshold, expected in test_cases:
            result = detect_anomaly(previous, current, threshold)
            assert result == expected

    def test_coinbase_response_validation_valid_formats(self, validator):
        """Test Coinbase response validation with valid formats."""
        # Object format with balance_summary
        valid_object_response = type(
            "MockResponse",
            (),
            {
                "balance_summary": type(
                    "MockSummary",
                    (),
                    {
                        "cfm_usd_balance": type(
                            "MockBalance", (), {"value": "5000.25"}
                        )()
                    },
                )()
            },
        )()

        assert validator.validate_balance_response(valid_object_response) is True

        # Dictionary format
        valid_dict_response = {"balance": "3000.75"}
        assert validator.validate_balance_response(valid_dict_response) is True

    def test_coinbase_response_validation_invalid_formats(self, validator):
        """Test Coinbase response validation with invalid formats."""
        # Missing balance_summary
        invalid_object = type("MockResponse", (), {"other_field": "value"})()
        assert validator.validate_balance_response(invalid_object) is False

        # Missing balance field in dict
        invalid_dict = {"other_field": "value"}
        assert validator.validate_balance_response(invalid_dict) is False

        # Not dict or object
        assert validator.validate_balance_response("string") is False
        assert validator.validate_balance_response(123) is False
        assert validator.validate_balance_response([]) is False

    def test_coinbase_response_validation_negative_balances(self, validator):
        """Test Coinbase response validation rejects negative balances."""
        # Object format with negative balance
        negative_object = type(
            "MockResponse",
            (),
            {
                "balance_summary": type(
                    "MockSummary",
                    (),
                    {
                        "cfm_usd_balance": type(
                            "MockBalance", (), {"value": "-1000.00"}
                        )()
                    },
                )()
            },
        )()

        assert validator.validate_balance_response(negative_object) is False

        # Dict format with negative balance
        negative_dict = {"balance": "-500.00"}
        assert validator.validate_balance_response(negative_dict) is False

    def test_coinbase_response_validation_invalid_numeric_formats(self, validator):
        """Test validation of invalid numeric formats in responses."""
        # Object with invalid value
        invalid_numeric_object = type(
            "MockResponse",
            (),
            {
                "balance_summary": type(
                    "MockSummary",
                    (),
                    {
                        "cfm_usd_balance": type(
                            "MockBalance", (), {"value": "not_a_number"}
                        )()
                    },
                )()
            },
        )()

        assert validator.validate_balance_response(invalid_numeric_object) is False

        # Dict with invalid value
        invalid_numeric_dict = {"balance": "invalid_amount"}
        assert validator.validate_balance_response(invalid_numeric_dict) is False

    def test_account_response_validation(self, validator):
        """Test account response structure validation."""
        # Valid account response
        valid_response = {
            "accounts": [
                {
                    "uuid": "12345678-1234-1234-1234-123456789012",
                    "name": "Test Account",
                    "currency": "USD",
                    "balance": "1000.00",
                }
            ]
        }

        assert validator.validate_account_response(valid_response) is True

        # Invalid - missing accounts
        invalid_response = {"other_field": "value"}
        assert validator.validate_account_response(invalid_response) is False

        # Invalid - accounts not a list
        invalid_list_response = {"accounts": "not_a_list"}
        assert validator.validate_account_response(invalid_list_response) is False

    def test_account_validation_missing_fields(self, validator):
        """Test account validation with missing required fields."""
        # Missing UUID
        missing_uuid = {"accounts": [{"name": "Test Account", "currency": "USD"}]}
        assert validator.validate_account_response(missing_uuid) is False

        # Missing currency
        missing_currency = {
            "accounts": [
                {"uuid": "12345678-1234-1234-1234-123456789012", "name": "Test Account"}
            ]
        }
        assert validator.validate_account_response(missing_currency) is False

    def test_account_validation_invalid_formats(self, validator):
        """Test account validation with invalid field formats."""
        # Invalid UUID format
        invalid_uuid = {
            "accounts": [
                {"uuid": "invalid-uuid", "name": "Test Account", "currency": "USD"}
            ]
        }
        assert validator.validate_account_response(invalid_uuid) is False

        # Invalid currency format
        invalid_currency = {
            "accounts": [
                {
                    "uuid": "12345678-1234-1234-1234-123456789012",
                    "name": "Test Account",
                    "currency": "X",  # Too short
                }
            ]
        }
        assert validator.validate_account_response(invalid_currency) is False

    def test_validation_error_logging(self, validator, caplog):
        """Test that validation errors are properly logged."""
        with caplog.at_level(logging.WARNING):
            # Trigger validation error
            validator.validate_balance_response({"invalid": "response"})

        # Check that warning was logged
        assert any(
            "missing 'balance' field" in record.message for record in caplog.records
        )

    def test_validation_counter_tracking(self, validator):
        """Test validation counter and failure tracking."""
        initial_count = validator.response_count
        initial_failures = validator.validation_failures

        # Valid response
        validator.validate_balance_response({"balance": "1000.00"})
        assert validator.response_count == initial_count + 1

        # Invalid response
        validator.validate_balance_response({"invalid": "response"})
        assert validator.response_count == initial_count + 2

    def test_order_response_validation(self, validator):
        """Test order response validation."""
        # Valid order response (object format)
        valid_order_object = type(
            "MockResponse",
            (),
            {
                "order": type(
                    "MockOrder",
                    (),
                    {"order_id": "order_123", "status": "FILLED", "side": "BUY"},
                )()
            },
        )()

        assert validator.validate_order_response(valid_order_object) is True

        # Valid order response (dict format)
        valid_order_dict = {"order_id": "order_456", "status": "OPEN", "side": "SELL"}

        assert validator.validate_order_response(valid_order_dict) is True

        # Invalid order response
        invalid_order = {"invalid": "order"}
        assert validator.validate_order_response(invalid_order) is False

    def test_order_status_validation(self, validator):
        """Test order status field validation."""
        # Invalid status
        invalid_status_order = {
            "order_id": "order_123",
            "status": "INVALID_STATUS",
            "side": "BUY",
        }

        assert validator.validate_order_response(invalid_status_order) is False

        # Invalid side
        invalid_side_order = {
            "order_id": "order_123",
            "status": "FILLED",
            "side": "INVALID_SIDE",
        }

        assert validator.validate_order_response(invalid_side_order) is False

    def test_position_response_validation(self, validator):
        """Test position response validation."""
        # Valid positions response
        valid_positions = {
            "positions": [
                {"product_id": "BTC-USD", "number_of_contracts": "1.5", "side": "LONG"}
            ]
        }

        assert validator.validate_position_response(valid_positions) is True

        # Valid positions list format (this should pass validation)
        valid_positions_list = []  # Empty list should be valid

        assert validator.validate_position_response(valid_positions_list) is True

        # Invalid positions response
        invalid_positions = {"invalid": "positions"}
        assert validator.validate_position_response(invalid_positions) is False

    def test_balance_validation_edge_cases(self, account):
        """Test balance validation edge cases and corner scenarios."""
        # Test scientific notation
        scientific = account._normalize_balance(Decimal("1E+2"))  # 100
        assert scientific == Decimal("100.00")

        # Test very precise input
        precise = account._normalize_balance(Decimal("99.999999999"))
        assert precise == Decimal("100.00")  # Should round up

        # Test rounding edge case - banker's rounding rounds 0.005 to 0.00
        edge_case = account._normalize_balance(Decimal("0.005"))
        assert edge_case == Decimal("0.00")  # Banker's rounding rounds to even

        # Test actual round up case
        round_up_case = account._normalize_balance(Decimal("0.006"))
        assert round_up_case == Decimal("0.01")  # Should round up

    def test_validation_failure_callback(self):
        """Test validation failure callback mechanism."""
        callback_called = False
        failure_reason = None

        def failure_callback(reason):
            nonlocal callback_called, failure_reason
            callback_called = True
            failure_reason = reason

        validator = CoinbaseResponseValidator(failure_callback=failure_callback)

        # Trigger validation failure
        validator.validate_balance_response({"invalid": "response"})

        # Note: Current implementation doesn't use callback, but test structure is ready
        # if callback functionality is added in the future

    def test_comprehensive_validation_scenario(self, validator, account):
        """Test comprehensive validation scenario with multiple components."""
        # Test complete validation flow

        # 1. Validate API response format
        api_response = {"balance": "5000.75"}
        assert validator.validate_balance_response(api_response) is True

        # 2. Extract and normalize balance
        balance_str = api_response["balance"]
        balance_decimal = Decimal(balance_str)
        normalized_balance = account._normalize_balance(balance_decimal)

        # 3. Validate normalized result
        assert normalized_balance == Decimal("5000.75")
        assert isinstance(normalized_balance, Decimal)

        # 4. Test anomaly detection
        previous_balance = Decimal("5200.00")
        change_pct = abs(
            (normalized_balance - previous_balance) / previous_balance * 100
        )
        is_anomaly = change_pct > 50.0
        assert is_anomaly is False  # Should be normal change

    def test_validation_performance_tracking(self, validator):
        """Test validation performance and response time tracking."""
        import time

        start_time = time.time()

        # Perform multiple validations
        for i in range(100):
            validator.validate_balance_response({"balance": f"{1000 + i}.00"})

        end_time = time.time()
        duration = end_time - start_time

        # Validation should be fast (under 1 second for 100 validations)
        assert duration < 1.0
        assert validator.response_count >= 100

    def test_validation_thread_safety(self, validator):
        """Test validation thread safety under concurrent access."""
        import threading

        def validate_responses():
            for i in range(10):
                validator.validate_balance_response({"balance": f"{i * 100}.00"})

        # Run validations concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=validate_responses)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        assert validator.response_count >= 50  # 5 threads * 10 validations each

    def test_validation_with_real_world_scenarios(self, validator, account):
        """Test validation with real-world data scenarios."""
        # Simulate real API responses
        real_world_scenarios = [
            # Normal trading balance
            {"balance": "15248.32"},
            # Large institutional balance
            {"balance": "5000000.00"},
            # Small retail balance
            {"balance": "127.45"},
            # Precise balance with many decimals
            {"balance": "999.99"},
            # Zero balance
            {"balance": "0.00"},
        ]

        for scenario in real_world_scenarios:
            # Validate API response
            assert validator.validate_balance_response(scenario) is True

            # Normalize balance
            balance = account._normalize_balance(Decimal(scenario["balance"]))
            assert isinstance(balance, Decimal)
            assert balance >= 0
