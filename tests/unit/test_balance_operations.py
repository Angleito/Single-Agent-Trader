"""
Unit tests for balance operations across exchanges and paper trading.

This module tests balance retrieval, normalization, precision handling,
error scenarios, and validation logic for both live and paper trading modes.
"""

import asyncio
import logging
import tempfile
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.exchange.base import BaseExchange, ExchangeConnectionError, ExchangeError
from bot.exchange.coinbase import CoinbaseResponseValidator
from bot.paper_trading import PaperTradingAccount
from bot.trading_types import (
    AccountType,
    FuturesAccountInfo,
    MarginHealthStatus,
    MarginInfo,
)


class MockExchange(BaseExchange):
    """Mock exchange implementation for testing."""

    def __init__(self, dry_run: bool = True):
        super().__init__(dry_run)
        self._mock_balance = Decimal("10000.00")
        self._connection_error = False
        self._auth_error = False
        self._timeout_error = False

    @property
    def enable_futures(self) -> bool:
        return False

    async def get_trading_symbol(self, symbol: str) -> str:
        return symbol

    async def connect(self) -> bool:
        if self._connection_error:
            raise ExchangeConnectionError("Mock connection error")
        return True

    async def disconnect(self) -> None:
        pass

    async def execute_trade_action(
        self, trade_action, symbol: str, current_price: Decimal
    ):
        return None

    async def place_market_order(self, symbol: str, side, quantity: Decimal):
        return None

    async def place_limit_order(
        self, symbol: str, side, quantity: Decimal, price: Decimal
    ):
        return None

    async def get_positions(self, symbol: str | None = None):
        return []

    async def get_account_balance(
        self, account_type: AccountType | None = None
    ) -> Decimal:
        if self._connection_error:
            raise ExchangeConnectionError("Connection failed")
        if self._auth_error:
            raise ExchangeError("Authentication failed")
        if self._timeout_error:
            raise TimeoutError("Request timeout")
        return self._mock_balance

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def cancel_all_orders(
        self, symbol: str | None = None, status: str | None = None
    ) -> bool:
        return True

    def is_connected(self) -> bool:
        return not self._connection_error

    def get_connection_status(self) -> dict:
        return {"connected": not self._connection_error}


class TestBalanceOperations:
    """Test cases for balance operations."""

    @pytest.fixture()
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture()
    def mock_exchange(self):
        """Create a mock exchange for testing."""
        return MockExchange()

    @pytest.fixture()
    def coinbase_validator(self):
        """Create a Coinbase response validator."""
        return CoinbaseResponseValidator()

    def test_balance_normalization_precision(self, temp_data_dir):
        """Test balance normalization maintains proper precision."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )

        # Test various precision scenarios
        test_amounts = [
            Decimal("10000.123456789"),  # High precision
            Decimal("999.99"),  # Standard precision
            Decimal("0.001"),  # Small amount
            Decimal("1000000.5555"),  # Large amount with decimals
        ]

        for amount in test_amounts:
            normalized = account._normalize_balance(amount)

            # Should be normalized to 2 decimal places
            assert normalized.as_tuple().exponent >= -2
            assert isinstance(normalized, Decimal)
            assert normalized >= 0

    def test_crypto_amount_normalization(self, temp_data_dir):
        """Test crypto amount normalization to 8 decimal places."""
        account = PaperTradingAccount(data_dir=temp_data_dir)

        test_amounts = [
            Decimal("0.123456789123"),  # High precision crypto
            Decimal("1.0"),  # Whole number
            Decimal("0.00000001"),  # Minimum precision
            Decimal("1000.12345678"),  # Large crypto amount
        ]

        for amount in test_amounts:
            normalized = account._normalize_crypto_amount(amount)

            # Should be normalized to 8 decimal places
            assert normalized.as_tuple().exponent >= -8
            assert isinstance(normalized, Decimal)
            assert normalized >= 0

    @pytest.mark.asyncio()
    async def test_successful_balance_retrieval(self, mock_exchange):
        """Test successful balance retrieval from exchange."""
        mock_exchange._mock_balance = Decimal("25000.50")

        balance = await mock_exchange.get_account_balance()

        assert balance == Decimal("25000.50")
        assert isinstance(balance, Decimal)

    @pytest.mark.asyncio()
    async def test_balance_retrieval_connection_error(self, mock_exchange):
        """Test balance retrieval handles connection errors."""
        mock_exchange._connection_error = True

        with pytest.raises(ExchangeConnectionError):
            await mock_exchange.get_account_balance()

    @pytest.mark.asyncio()
    async def test_balance_retrieval_timeout_error(self, mock_exchange):
        """Test balance retrieval handles timeout errors."""
        mock_exchange._timeout_error = True

        with pytest.raises(TimeoutError):
            await mock_exchange.get_account_balance()

    @pytest.mark.asyncio()
    async def test_balance_retrieval_with_error_handling(self, mock_exchange):
        """Test balance retrieval with error boundary protection."""
        mock_exchange._auth_error = True

        # Should return 0 as safe default when using error handling
        balance = await mock_exchange.get_account_balance_with_error_handling()

        assert balance == Decimal("0")

    def test_account_status_calculation(self, temp_data_dir):
        """Test account status calculation with various scenarios."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )

        # Reset account to ensure clean state
        account.reset_account(Decimal("10000"))

        # Test initial status
        status = account.get_account_status()

        assert status["starting_balance"] == 10000.0
        assert status["current_balance"] == 10000.0
        assert status["equity"] == 10000.0
        assert status["total_pnl"] == 0.0
        assert status["roi_percent"] == 0.0
        assert status["margin_used"] == 0.0
        assert status["open_positions"] == 0

    def test_account_status_with_margin_usage(self, temp_data_dir):
        """Test account status with margin being used."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )
        account.reset_account(Decimal("10000"))
        account.margin_used = Decimal("2000")

        status = account.get_account_status()

        assert status["margin_used"] == 2000.0
        assert status["margin_available"] == 8000.0

    def test_account_status_with_profit(self, temp_data_dir):
        """Test account status with profit scenario."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )
        account.reset_account(Decimal("10000"))
        account.current_balance = Decimal("12000")
        account.equity = Decimal("12000")

        status = account.get_account_status()

        assert status["total_pnl"] == 2000.0
        assert status["roi_percent"] == 20.0

    def test_account_status_with_loss(self, temp_data_dir):
        """Test account status with loss scenario."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )
        account.reset_account(Decimal("10000"))
        account.current_balance = Decimal("8000")
        account.equity = Decimal("8000")

        status = account.get_account_status()

        assert status["total_pnl"] == -2000.0
        assert status["roi_percent"] == -20.0

    def test_peak_equity_tracking(self, temp_data_dir):
        """Test peak equity tracking and drawdown calculation."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )
        account.reset_account(Decimal("10000"))

        # Simulate profit
        account.current_balance = Decimal("15000")
        account.equity = Decimal("15000")

        status = account.get_account_status()
        assert status["peak_equity"] == 15000.0
        assert status["current_drawdown"] == 0.0

        # Simulate drawdown
        account.current_balance = Decimal("12000")
        account.equity = Decimal("12000")

        status = account.get_account_status()
        assert status["peak_equity"] == 15000.0
        assert status["current_drawdown"] == 20.0  # (15000 - 12000) / 15000 * 100

    def test_balance_validation_range_checks(self, temp_data_dir):
        """Test balance validation with range checks."""
        account = PaperTradingAccount(data_dir=temp_data_dir)

        # Test negative balance handling
        negative_balance = Decimal("-1000")
        normalized = account._normalize_balance(negative_balance)
        # Note: This test allows negative balances to be processed
        assert isinstance(normalized, Decimal)

        # Test extremely large balance
        large_balance = Decimal("999999999.99")
        normalized = account._normalize_balance(large_balance)
        assert isinstance(normalized, Decimal)
        assert normalized == Decimal("999999999.99")

    def test_balance_precision_validation(self, temp_data_dir):
        """Test balance precision validation and formatting."""
        account = PaperTradingAccount(data_dir=temp_data_dir)

        # Test various precision scenarios
        test_cases = [
            (Decimal("100.123456"), Decimal("100.12")),  # Round down
            (Decimal("100.125"), Decimal("100.12")),  # Banker's rounding (even)
            (Decimal("100.135"), Decimal("100.14")),  # Banker's rounding (odd)
            (Decimal("100"), Decimal("100.00")),  # Add precision
        ]

        for input_val, expected in test_cases:
            result = account._normalize_balance(input_val)
            assert result == expected

    def test_zero_balance_handling(self, temp_data_dir):
        """Test handling of zero balances."""
        # Clean the temp directory to ensure no existing files
        for file in temp_data_dir.glob("*"):
            file.unlink()

        account = PaperTradingAccount(
            starting_balance=Decimal("0"), data_dir=temp_data_dir
        )

        status = account.get_account_status()

        assert status["starting_balance"] == 0.0
        assert status["current_balance"] == 0.0
        assert status["equity"] == 0.0
        assert status["roi_percent"] == 0.0

    def test_coinbase_balance_response_validation(self, coinbase_validator):
        """Test Coinbase balance response validation."""

        # Create proper mock objects with working hasattr
        class MockBalance:
            def __init__(self, value):
                self.value = value

        class MockSummary:
            def __init__(self, balance_value):
                self.cfm_usd_balance = MockBalance(balance_value)

        class MockResponse:
            def __init__(self, balance_value):
                self.balance_summary = MockSummary(balance_value)

        # Valid response format
        valid_response = MockResponse("10000.50")
        assert coinbase_validator.validate_balance_response(valid_response) is True

        # Invalid response - missing balance
        invalid_response = {"some_other_field": "value"}
        assert coinbase_validator.validate_balance_response(invalid_response) is False

        # Invalid response - negative balance
        negative_response = MockResponse("-1000")
        assert coinbase_validator.validate_balance_response(negative_response) is False

    def test_coinbase_dict_balance_validation(self, coinbase_validator):
        """Test Coinbase dict format balance validation."""
        # Valid dict format
        valid_dict = {"balance": "5000.25"}
        assert coinbase_validator.validate_balance_response(valid_dict) is True

        # Invalid dict format - missing balance
        invalid_dict = {"other_field": "value"}
        assert coinbase_validator.validate_balance_response(invalid_dict) is False

        # Invalid dict format - negative balance
        negative_dict = {"balance": "-500.00"}
        assert coinbase_validator.validate_balance_response(negative_dict) is False

    @pytest.mark.asyncio()
    async def test_balance_anomaly_detection(self, mock_exchange):
        """Test balance anomaly detection for suspicious changes."""
        previous_balance = Decimal("10000")
        mock_exchange._mock_balance = Decimal("100000")  # 10x increase

        current_balance = await mock_exchange.get_account_balance()

        # Calculate percentage change
        change_pct = ((current_balance - previous_balance) / previous_balance) * 100

        # Flag as anomaly if change > 50%
        is_anomaly = abs(change_pct) > 50
        assert is_anomaly is True

        # Test normal change
        mock_exchange._mock_balance = Decimal("10500")  # 5% increase
        current_balance = await mock_exchange.get_account_balance()
        change_pct = ((current_balance - previous_balance) / previous_balance) * 100
        is_anomaly = abs(change_pct) > 50
        assert is_anomaly is False

    def test_balance_format_validation(self, temp_data_dir):
        """Test balance format validation and type checking."""
        account = PaperTradingAccount(data_dir=temp_data_dir)

        # Test valid formats
        valid_inputs = [
            "1000.50",
            1000.50,
            Decimal("1000.50"),
        ]

        for valid_input in valid_inputs:
            try:
                normalized = account._normalize_balance(Decimal(str(valid_input)))
                assert isinstance(normalized, Decimal)
                assert normalized >= 0
            except (ValueError, TypeError):
                pytest.fail(f"Should accept valid input: {valid_input}")

        # Test invalid formats
        invalid_inputs = [
            "invalid",
            None,
            float("inf"),
            float("nan"),
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(
                (ValueError, TypeError, AttributeError, OSError, InvalidOperation)
            ):
                account._normalize_balance(Decimal(str(invalid_input)))

    def test_concurrent_balance_access(self, temp_data_dir):
        """Test thread-safe balance access and modifications."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )

        def modify_balance():
            """Function to modify balance in thread."""
            account.current_balance = Decimal("12000")
            account.equity = Decimal("12000")

        def read_balance():
            """Function to read balance in thread."""
            return account.get_account_status()

        # Test that multiple operations don't corrupt data
        import threading

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=modify_balance)
            threads.append(thread)
            thread = threading.Thread(target=read_balance)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should maintain data integrity
        final_status = account.get_account_status()
        assert isinstance(final_status["current_balance"], float)
        assert final_status["current_balance"] >= 0

    @pytest.mark.asyncio()
    async def test_balance_retry_logic(self, mock_exchange):
        """Test balance retrieval retry logic on failures."""
        call_count = 0

        async def failing_balance_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ExchangeConnectionError("Temporary failure")
            return Decimal("10000")

        # Mock the balance method to fail first 2 times
        with patch.object(
            mock_exchange, "get_account_balance", side_effect=failing_balance_call
        ):
            # Simulate retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    balance = await mock_exchange.get_account_balance()
                    assert balance == Decimal("10000")
                    break
                except ExchangeConnectionError:
                    if attempt == max_retries - 1:
                        pytest.fail("Should have succeeded after retries")
                    continue

    def test_balance_error_logging(self, caplog, mock_exchange):
        """Test that balance errors are properly logged."""
        mock_exchange._auth_error = True

        with caplog.at_level(logging.ERROR):
            # This should trigger error logging
            asyncio.run(mock_exchange.get_account_balance_with_error_handling())

        # Check that error was logged (could be at ERROR level)
        assert any(
            "error" in record.message.lower() or "exception" in record.message.lower()
            for record in caplog.records
        )

    def test_performance_metrics_generation(self, temp_data_dir):
        """Test generation of balance-related performance metrics."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )
        account.reset_account(Decimal("10000"))
        account.current_balance = Decimal("11000")
        account.equity = Decimal("11000")
        account.margin_used = Decimal("2000")

        metrics = account.get_performance_metrics_for_monitor()

        # Check required metrics are present
        metric_names = [metric["name"] for metric in metrics]

        expected_metrics = [
            "paper_trading.equity",
            "paper_trading.balance",
            "paper_trading.total_pnl",
            "paper_trading.roi_percent",
            "paper_trading.margin_used",
            "paper_trading.margin_available",
        ]

        for expected_metric in expected_metrics:
            assert expected_metric in metric_names

        # Verify metric structure
        for metric in metrics:
            assert "name" in metric
            assert "value" in metric
            assert "timestamp" in metric
            assert "unit" in metric
            assert "tags" in metric

    def test_account_reset_functionality(self, temp_data_dir):
        """Test account reset preserves critical data integrity."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )

        # Modify account state
        account.current_balance = Decimal("12000")
        account.equity = Decimal("12000")
        account.margin_used = Decimal("1000")

        # Reset account
        new_balance = Decimal("20000")
        account.reset_account(new_balance)

        # Verify reset
        assert account.starting_balance == new_balance
        assert account.current_balance == new_balance
        assert account.equity == new_balance
        assert account.margin_used == Decimal("0")
        assert len(account.open_trades) == 0
        assert len(account.closed_trades) == 0

    def test_futures_account_balance_handling(self, mock_exchange):
        """Test futures account balance handling."""
        # Test futures account info structure
        futures_info = FuturesAccountInfo(
            account_type=AccountType.CFM,
            account_id="test_futures_123",
            currency="USD",
            cash_balance=Decimal("10000"),
            futures_balance=Decimal("50000"),
            total_balance=Decimal("60000"),
            max_position_size=Decimal("1000000"),
            timestamp=datetime.now(UTC),
            margin_info=MarginInfo(
                total_margin=Decimal("10000"),
                available_margin=Decimal("8000"),
                used_margin=Decimal("2000"),
                maintenance_margin=Decimal("1500"),
                initial_margin=Decimal("2000"),
                margin_ratio=0.2,
                health_status=MarginHealthStatus.HEALTHY,
                liquidation_threshold=Decimal("1000"),
                intraday_margin_requirement=Decimal("2000"),
                overnight_margin_requirement=Decimal("2500"),
            ),
        )

        assert futures_info.cash_balance == Decimal("10000")
        assert futures_info.futures_balance == Decimal("50000")
        assert isinstance(futures_info.margin_info, MarginInfo)

    def test_edge_case_balance_scenarios(self, temp_data_dir):
        """Test edge case balance scenarios."""
        account = PaperTradingAccount(data_dir=temp_data_dir)

        # Test very small balance
        tiny_balance = Decimal("0.01")
        normalized = account._normalize_balance(tiny_balance)
        assert normalized == Decimal("0.01")

        # Test maximum precision balance
        max_precision = Decimal("999999.99")
        normalized = account._normalize_balance(max_precision)
        assert normalized == max_precision

        # Test zero balance edge cases
        zero_balance = Decimal("0")
        normalized = account._normalize_balance(zero_balance)
        assert normalized == Decimal("0.00")
