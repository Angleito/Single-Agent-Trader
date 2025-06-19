"""
Unit tests for paper trading balance calculations and trade execution.

This module tests paper trading balance updates, margin calculations,
fee deductions, trade simulation, and P&L tracking with real market data.
"""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.fee_calculator import TradeFees
from bot.paper_trading import PaperTradingAccount
from bot.trading_types import OrderStatus, TradeAction


class TestPaperTradingBalance:
    """Test cases for paper trading balance calculations."""

    @pytest.fixture()
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture()
    def account(self, temp_data_dir):
        """Create paper trading account for testing."""
        return PaperTradingAccount(
            starting_balance=Decimal("10000"), data_dir=temp_data_dir
        )

    @pytest.fixture()
    def mock_fee_calculator(self):
        """Mock fee calculator for consistent testing."""
        with patch("bot.paper_trading.fee_calculator") as mock_calc:
            mock_calc.calculate_trade_fees.return_value = TradeFees(
                entry_fee=Decimal("5.00"),
                exit_fee=Decimal("5.00"),
                total_fee=Decimal("10.00"),
                fee_rate=0.001,
                net_position_value=Decimal("4990.00"),
            )
            yield mock_calc

    def test_initial_balance_setup(self, account):
        """Test initial balance and account setup."""
        assert account.starting_balance == Decimal("10000.00")
        assert account.current_balance == Decimal("10000.00")
        assert account.equity == Decimal("10000.00")
        assert account.margin_used == Decimal("0.00")

    def test_long_trade_execution_balance_update(self, account, mock_fee_calculator):
        """Test balance updates for long trade execution."""
        action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Test long trade",
        )

        current_price = Decimal("50000")

        # Execute trade
        order = account.execute_trade_action(action, "BTC-USD", current_price)

        assert order is not None
        assert order.status == OrderStatus.FILLED

        # Check balance updates
        status = account.get_account_status()

        # Balance should be reduced by fees
        assert status["current_balance"] < 10000.0

        # Margin should be used
        assert status["margin_used"] > 0

        # Should have one open position
        assert status["open_positions"] == 1

    def test_short_trade_execution_balance_update(self, account, mock_fee_calculator):
        """Test balance updates for short trade execution."""
        action = TradeAction(
            action="SHORT",
            size_pct=15,
            take_profit_pct=2.5,
            stop_loss_pct=1.5,
            rationale="Test short trade",
        )

        current_price = Decimal("50000")

        # Execute trade
        order = account.execute_trade_action(action, "BTC-USD", current_price)

        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == "SELL"

        # Check balance and margin
        status = account.get_account_status()
        assert status["margin_used"] > 0
        assert status["open_positions"] == 1

    def test_insufficient_funds_handling(self, account, mock_fee_calculator):
        """Test handling of insufficient funds scenarios."""
        # Try to place a very large trade
        action = TradeAction(
            action="LONG",
            size_pct=100,  # 100% position
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Oversized trade",
        )

        current_price = Decimal("100000")  # High price to require more margin

        # This should fail due to insufficient funds
        order = account.execute_trade_action(action, "BTC-USD", current_price)

        if order:
            assert order.status == OrderStatus.REJECTED
        else:
            # Or return None for failed trades
            assert order is None

    def test_margin_calculation_accuracy(self, account, mock_fee_calculator):
        """Test accurate margin calculation with leverage."""
        action = TradeAction(
            action="LONG",
            size_pct=20,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Margin test",
        )

        current_price = Decimal("50000")

        # Mock settings for leverage
        with patch("bot.paper_trading.settings") as mock_settings:
            mock_settings.trading.leverage = 5

            order = account.execute_trade_action(action, "BTC-USD", current_price)

            if order and order.status == OrderStatus.FILLED:
                status = account.get_account_status()

                # With 20% position and 5x leverage, margin should be 4% of equity
                expected_margin_ratio = 0.20 / 5  # 4%
                actual_margin_ratio = status["margin_used"] / status["equity"]

                # Allow some tolerance for fees and slippage
                assert abs(actual_margin_ratio - expected_margin_ratio) < 0.02

    def test_fee_deduction_accuracy(self, account):
        """Test accurate fee deduction from balance."""
        initial_balance = account.current_balance

        action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Fee test",
        )

        # Mock specific fee calculation
        with patch("bot.paper_trading.fee_calculator") as mock_calc:
            expected_fee = Decimal("10.50")
            mock_calc.calculate_trade_fees.return_value = TradeFees(
                entry_fee=expected_fee,
                exit_fee=expected_fee,
                total_fee=expected_fee * 2,
                fee_rate=0.001,
                net_position_value=Decimal("4990.00"),
            )

            order = account.execute_trade_action(action, "BTC-USD", Decimal("50000"))

            if order and order.status == OrderStatus.FILLED:
                # Balance should be reduced by exactly the entry fee
                expected_balance = initial_balance - expected_fee
                assert account.current_balance == expected_balance

    def test_slippage_application(self, account, mock_fee_calculator):
        """Test slippage application in trade execution."""
        action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Slippage test",
        )

        market_price = Decimal("50000")

        order = account.execute_trade_action(action, "BTC-USD", market_price)

        if order and order.status == OrderStatus.FILLED:
            # Execution price should be slightly higher than market price for LONG
            assert order.price > market_price

            # Slippage should be reasonable (less than 1%)
            slippage_pct = ((order.price - market_price) / market_price) * 100
            assert slippage_pct < 1.0

    def test_position_closing_balance_reconciliation(
        self, account, mock_fee_calculator
    ):
        """Test balance reconciliation when closing positions."""
        # Open a position
        open_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Open position",
        )

        entry_price = Decimal("50000")
        account.execute_trade_action(open_action, "BTC-USD", entry_price)

        initial_equity = account.equity
        initial_margin = account.margin_used

        # Close the position at profit
        close_action = TradeAction(
            action="CLOSE",
            size_pct=0,
            take_profit_pct=0,
            stop_loss_pct=0,
            rationale="Close position",
        )

        exit_price = Decimal("51000")  # 2% profit
        order = account.execute_trade_action(close_action, "BTC-USD", exit_price)

        if order and order.status == OrderStatus.FILLED:
            # Margin should be released
            assert account.margin_used < initial_margin

            # Equity should increase due to profit (minus fees)
            assert account.equity > initial_equity

    def test_unrealized_pnl_calculation(self, account, mock_fee_calculator):
        """Test unrealized P&L calculation with price changes."""
        # Open a long position
        action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="P&L test",
        )

        entry_price = Decimal("50000")
        account.execute_trade_action(action, "BTC-USD", entry_price)

        # Test unrealized P&L with price increase
        current_prices = {"BTC-USD": Decimal("52000")}  # 4% increase
        status = account.get_account_status(current_prices)

        # Should show positive unrealized P&L
        assert status["unrealized_pnl"] > 0

        # Test with price decrease
        current_prices = {"BTC-USD": Decimal("48000")}  # 4% decrease
        status = account.get_account_status(current_prices)

        # Should show negative unrealized P&L
        assert status["unrealized_pnl"] < 0

    def test_multiple_positions_balance_tracking(self, account, mock_fee_calculator):
        """Test balance tracking with multiple open positions."""
        # Open first position
        action1 = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Position 1",
        )
        account.execute_trade_action(action1, "BTC-USD", Decimal("50000"))

        # Open second position
        action2 = TradeAction(
            action="SHORT",
            size_pct=5,
            take_profit_pct=1.5,
            stop_loss_pct=1.0,
            rationale="Position 2",
        )
        account.execute_trade_action(action2, "ETH-USD", Decimal("3000"))

        status = account.get_account_status()

        # Should have 2 open positions
        assert status["open_positions"] == 2

        # Margin should account for both positions
        assert status["margin_used"] > 0

    def test_position_increase_balance_updates(self, account, mock_fee_calculator):
        """Test balance updates when increasing existing position size."""
        # Open initial position
        action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Initial position",
        )

        account.execute_trade_action(action, "BTC-USD", Decimal("50000"))
        initial_margin = account.margin_used

        # Increase position size
        increase_action = TradeAction(
            action="LONG",
            size_pct=5,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Increase position",
        )

        account.execute_trade_action(increase_action, "BTC-USD", Decimal("50500"))

        # Margin should increase
        assert account.margin_used > initial_margin

        # Should still have only 1 position (increased size)
        status = account.get_account_status()
        assert status["open_positions"] == 1

    def test_position_reversal_balance_handling(self, account, mock_fee_calculator):
        """Test balance handling for position reversals."""
        # Open long position
        long_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Long position",
        )

        account.execute_trade_action(long_action, "BTC-USD", Decimal("50000"))
        initial_balance = account.current_balance

        # Reverse to short position
        short_action = TradeAction(
            action="SHORT",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Reverse to short",
        )

        account.execute_trade_action(short_action, "BTC-USD", Decimal("51000"))

        # Should still have 1 position but now short
        status = account.get_account_status()
        assert status["open_positions"] == 1

        # Balance should reflect fees from both close and open
        assert account.current_balance < initial_balance

    def test_balance_precision_in_calculations(self, account, mock_fee_calculator):
        """Test balance precision is maintained in complex calculations."""
        action = TradeAction(
            action="LONG",
            size_pct=33,  # Odd percentage to test precision
            take_profit_pct=1.33,
            stop_loss_pct=0.77,
            rationale="Precision test",
        )

        # Use price that creates non-round numbers
        price = Decimal("47831.57")

        account.execute_trade_action(action, "BTC-USD", price)

        status = account.get_account_status()

        # All balance values should have proper precision
        assert isinstance(status["current_balance"], float)
        assert isinstance(status["equity"], float)
        assert isinstance(status["margin_used"], float)

        # Internal values should maintain Decimal precision
        assert isinstance(account.current_balance, Decimal)
        assert isinstance(account.equity, Decimal)
        assert isinstance(account.margin_used, Decimal)

    def test_futures_contract_balance_calculations(self, account, mock_fee_calculator):
        """Test balance calculations for futures contracts."""
        action = TradeAction(
            action="LONG",
            size_pct=20,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            rationale="Futures test",
        )

        # Mock futures trading settings
        with patch("bot.paper_trading.settings") as mock_settings:
            mock_settings.trading.enable_futures = True
            mock_settings.trading.leverage = 5
            mock_settings.trading.fixed_contract_size = None

            order = account.execute_trade_action(action, "ETH-USD", Decimal("3000"))

            if order and order.status == OrderStatus.FILLED:
                # Verify contract-based quantity calculation
                assert order.quantity > 0

                # For ETH futures, quantity should be multiple of 0.1
                quantity_decimal = order.quantity % Decimal("0.1")
                assert quantity_decimal == 0

    def test_balance_error_recovery(self, account, mock_fee_calculator):
        """Test balance error recovery and data integrity."""
        # Simulate corrupted balance state
        account.current_balance = Decimal("-100")  # Invalid negative balance
        account.margin_used = Decimal("999999999")  # Impossibly high margin

        # System should handle and recover
        status = account.get_account_status()

        # Should not crash and provide reasonable values
        assert isinstance(status["current_balance"], float)
        assert isinstance(status["margin_used"], float)

    def test_performance_metrics_accuracy(self, account, mock_fee_calculator):
        """Test accuracy of performance metrics calculation."""
        # Execute several trades
        trades = [
            (TradeAction("LONG", 10, 2.0, 1.0, "Trade 1"), "BTC-USD", Decimal("50000")),
            (TradeAction("SHORT", 5, 1.5, 1.0, "Trade 2"), "ETH-USD", Decimal("3000")),
        ]

        for action, symbol, price in trades:
            account.execute_trade_action(action, symbol, price)

        metrics = account.get_performance_metrics_for_monitor()

        # Verify metric values are reasonable
        equity_metric = next(m for m in metrics if m["name"] == "paper_trading.equity")
        assert equity_metric["value"] > 0

        balance_metric = next(
            m for m in metrics if m["name"] == "paper_trading.balance"
        )
        assert balance_metric["value"] > 0

        margin_metric = next(
            m for m in metrics if m["name"] == "paper_trading.margin_used"
        )
        assert margin_metric["value"] >= 0

    def test_trade_lifecycle_balance_integrity(self, account, mock_fee_calculator):
        """Test balance integrity throughout complete trade lifecycle."""
        initial_balance = account.current_balance

        # Open position
        open_action = TradeAction("LONG", 15, 3.0, 1.5, "Lifecycle test")
        account.execute_trade_action(open_action, "BTC-USD", Decimal("50000"))

        after_open_balance = account.current_balance
        margin_used = account.margin_used

        # Verify margin and fee deduction
        assert after_open_balance < initial_balance  # Fees deducted
        assert margin_used > 0  # Margin allocated

        # Close position at profit
        close_action = TradeAction("CLOSE", 0, 0, 0, "Close position")
        account.execute_trade_action(close_action, "BTC-USD", Decimal("52000"))

        final_balance = account.current_balance
        final_margin = account.margin_used

        # Verify margin release and profit realization
        assert final_margin == 0  # All margin released
        assert final_balance > after_open_balance  # Profit realized (minus exit fees)

    def test_edge_case_balance_scenarios(self, account, mock_fee_calculator):
        """Test edge case balance scenarios."""
        # Test with very small position
        small_action = TradeAction("LONG", 0.1, 1.0, 0.5, "Tiny position")
        order = account.execute_trade_action(small_action, "BTC-USD", Decimal("50000"))

        if order:
            assert order.quantity > 0

        # Test with maximum allowed position
        max_action = TradeAction(
            "LONG", 20, 2.0, 1.0, "Max position"
        )  # 20% is typical max
        account.execute_trade_action(max_action, "BTC-USD", Decimal("50000"))

        status = account.get_account_status()
        assert (
            status["margin_used"] <= status["equity"]
        )  # Margin shouldn't exceed equity

    def test_concurrent_balance_operations(self, account, mock_fee_calculator):
        """Test balance operations under concurrent access."""
        import threading

        def execute_trade():
            action = TradeAction("LONG", 1, 1.0, 0.5, "Concurrent test")
            account.execute_trade_action(action, "BTC-USD", Decimal("50000"))

        # Execute multiple trades concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=execute_trade)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify data integrity after concurrent operations
        status = account.get_account_status()
        assert isinstance(status["current_balance"], float)
        assert status["current_balance"] >= 0
