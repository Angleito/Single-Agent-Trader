"""
End-to-end balance functionality integration tests.

This module tests complete balance workflows across all components:
- Account initialization and balance loading
- Trade execution with balance updates
- Balance reconciliation after trades
- Cross-component balance consistency
- Performance monitoring for balance operations
"""

import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot.main import TradingEngine
from bot.paper_trading import PaperTradingAccount
from bot.trading_types import (
    MarketData,
    Order,
    OrderStatus,
    TradeAction,
)


class TestBalanceIntegration:
    """Test end-to-end balance functionality integration."""

    @pytest.fixture()
    def mock_market_data(self):
        """Create realistic mock market data for testing."""
        base_price = 50000
        timestamps = [
            datetime.now(UTC) - timedelta(minutes=i) for i in range(200, 0, -1)
        ]

        market_data = []
        for i, timestamp in enumerate(timestamps):
            current_price = base_price + (i * 10)
            market_data.append(
                MarketData(
                    symbol="BTC-USD",
                    timestamp=timestamp,
                    open=Decimal(str(current_price - 5)),
                    high=Decimal(str(current_price + 10)),
                    low=Decimal(str(current_price - 10)),
                    close=Decimal(str(current_price)),
                    volume=Decimal("100"),
                )
            )
        return market_data

    @pytest.fixture()
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory for testing."""
        data_dir = tmp_path / "paper_trading"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @pytest.mark.asyncio()
    async def test_account_initialization_and_balance_loading(self, temp_data_dir):
        """Test account initialization and balance loading from persistent storage."""
        starting_balance = Decimal("10000.00")

        # Create initial account
        account = PaperTradingAccount(
            starting_balance=starting_balance, data_dir=temp_data_dir
        )

        # Verify initial state
        assert account.starting_balance == starting_balance
        assert account.current_balance == starting_balance
        assert account.equity == starting_balance
        assert account.margin_used == Decimal("0.00")
        assert len(account.open_trades) == 0
        assert len(account.closed_trades) == 0

        # Make some balance changes
        account.current_balance = Decimal("9500.00")
        account.margin_used = Decimal("1000.00")
        account.equity = Decimal("10500.00")
        account._save_state()

        # Create new account instance to test loading
        account2 = PaperTradingAccount(data_dir=temp_data_dir)

        # Verify state was loaded correctly
        assert account2.starting_balance == starting_balance
        assert account2.current_balance == Decimal("9500.00")
        assert account2.margin_used == Decimal("1000.00")
        assert account2.equity == Decimal("10500.00")

    @pytest.mark.asyncio()
    async def test_trade_execution_with_balance_updates(
        self, mock_market_data, temp_data_dir
    ):
        """Test trade execution with real-time balance updates."""
        with (
            patch.multiple(
                "bot.main.MarketDataProvider",
                get_latest_ohlcv=Mock(return_value=mock_market_data),
                to_dataframe=Mock(
                    return_value=self._create_mock_dataframe(mock_market_data)
                ),
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
                get_data_status=Mock(
                    return_value={"connected": True, "cached_candles": 200}
                ),
            ),
            patch.multiple(
                "bot.main.LLMAgent",
                analyze_market=AsyncMock(
                    return_value=TradeAction(
                        action="LONG",
                        size_pct=20,
                        take_profit_pct=3.0,
                        stop_loss_pct=2.0,
                        rationale="Test bullish trade",
                    )
                ),
                is_available=Mock(return_value=True),
            ),
            patch.multiple(
                "bot.main.CoinbaseClient",
                connect=AsyncMock(return_value=True),
                disconnect=AsyncMock(return_value=True),
                is_connected=Mock(return_value=True),
                _get_account_balance=AsyncMock(return_value=Decimal("10000")),
            ),
        ):
            # Override the paper account data directory
            engine = TradingEngine(symbol="BTC-USD", interval="1m", dry_run=True)
            await engine._initialize_components()

            # Replace paper account with our test instance
            engine.paper_account = PaperTradingAccount(
                starting_balance=Decimal("10000.00"), data_dir=temp_data_dir
            )

            initial_balance = engine.paper_account.current_balance
            initial_equity = engine.paper_account.equity
            initial_margin = engine.paper_account.margin_used

            # Execute a trade
            current_price = mock_market_data[-1].close
            trade_action = TradeAction(
                action="LONG",
                size_pct=20,
                take_profit_pct=3.0,
                stop_loss_pct=2.0,
                rationale="Balance test trade",
            )

            # Mock successful order execution
            successful_order = Order(
                id="test_order_123",
                symbol="BTC-USD",
                side="BUY",
                type="MARKET",
                quantity=Decimal("0.04"),  # 20% of 10k at 50k price
                price=current_price,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(UTC),
                filled_quantity=Decimal("0.04"),
            )
            engine.paper_account.execute_trade_action = Mock(
                return_value=successful_order
            )

            await engine._execute_trade(trade_action, current_price)

            # Verify balance changes
            assert (
                engine.paper_account.current_balance < initial_balance
            )  # Fees deducted
            assert engine.paper_account.margin_used > initial_margin  # Margin allocated
            assert len(engine.paper_account.open_trades) == 1  # Position opened

            # Verify balance consistency
            account_status = engine.paper_account.get_account_status()
            assert account_status["current_balance"] == float(
                engine.paper_account.current_balance
            )
            assert account_status["margin_used"] == float(
                engine.paper_account.margin_used
            )
            assert account_status["equity"] >= float(
                initial_equity
            )  # Should include unrealized P&L

            await engine._shutdown()

    @pytest.mark.asyncio()
    async def test_balance_reconciliation_after_trades(
        self, mock_market_data, temp_data_dir
    ):
        """Test balance reconciliation after multiple trades."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000.00"), data_dir=temp_data_dir
        )

        current_price = Decimal("50000.00")

        # Execute multiple trades to test reconciliation
        trades = [
            ("LONG", Decimal("0.02"), Decimal("50000.00")),
            ("SHORT", Decimal("0.01"), Decimal("51000.00")),
            ("LONG", Decimal("0.03"), Decimal("49000.00")),
        ]

        expected_balance_changes = []
        initial_balance = account.current_balance

        for action, size, price in trades:
            trade_action = TradeAction(
                action=action,
                size_pct=10,  # Will be overridden by our direct execution
                take_profit_pct=2.0,
                stop_loss_pct=1.5,
                rationale=f"Test {action} trade",
            )

            # Calculate expected fees
            trade_value = size * price
            expected_fee = trade_value * account.fee_rate
            expected_balance_changes.append(-expected_fee)

            # Execute trade
            order = account.execute_trade_action(trade_action, "BTC-USD", price)

            assert order is not None
            assert order.status == OrderStatus.FILLED

        # Verify total balance change matches expected fees
        total_expected_fee_impact = sum(expected_balance_changes)
        actual_balance_change = account.current_balance - initial_balance

        # Balance should decrease by at least the total fees (may have additional margin impacts)
        assert actual_balance_change <= total_expected_fee_impact

        # Verify account status consistency
        account_status = account.get_account_status({current_price: current_price})

        # All monetary values should be properly normalized
        assert account_status["current_balance"] == float(account.current_balance)
        assert account_status["margin_used"] == float(account.margin_used)
        assert isinstance(account_status["equity"], float)
        assert isinstance(account_status["total_pnl"], float)

    @pytest.mark.asyncio()
    async def test_balance_consistency_across_restarts(
        self, mock_market_data, temp_data_dir
    ):
        """Test balance consistency across account restarts."""
        starting_balance = Decimal("10000.00")

        # Create account and execute some trades
        account1 = PaperTradingAccount(
            starting_balance=starting_balance, data_dir=temp_data_dir
        )

        # Execute a trade
        trade_action = TradeAction(
            action="LONG",
            size_pct=15,
            take_profit_pct=2.5,
            stop_loss_pct=1.5,
            rationale="Persistence test trade",
        )

        current_price = Decimal("50000.00")
        order = account1.execute_trade_action(trade_action, "BTC-USD", current_price)

        assert order is not None
        assert len(account1.open_trades) == 1

        # Capture state before restart
        pre_restart_status = account1.get_account_status()
        pre_restart_balance = account1.current_balance
        pre_restart_margin = account1.margin_used
        pre_restart_open_trades = len(account1.open_trades)

        # Save state explicitly
        account1._save_state()

        # Create new account instance (simulating restart)
        account2 = PaperTradingAccount(data_dir=temp_data_dir)

        # Verify state was restored correctly
        assert account2.current_balance == pre_restart_balance
        assert account2.margin_used == pre_restart_margin
        assert len(account2.open_trades) == pre_restart_open_trades

        # Verify account status consistency
        post_restart_status = account2.get_account_status()

        assert (
            post_restart_status["current_balance"]
            == pre_restart_status["current_balance"]
        )
        assert post_restart_status["margin_used"] == pre_restart_status["margin_used"]
        assert (
            post_restart_status["open_positions"]
            == pre_restart_status["open_positions"]
        )

    @pytest.mark.asyncio()
    async def test_multi_trade_balance_reconciliation(self, temp_data_dir):
        """Test balance reconciliation across multiple trades with different outcomes."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000.00"), data_dir=temp_data_dir
        )

        current_prices = {
            "BTC-USD": Decimal("50000.00"),
            "ETH-USD": Decimal("3000.00"),
        }

        # Execute trades with known outcomes
        trade_scenarios = [
            # Profitable LONG trade
            {
                "action": "LONG",
                "symbol": "BTC-USD",
                "entry_price": Decimal("50000.00"),
                "exit_price": Decimal("52000.00"),
                "size": Decimal("0.02"),
            },
            # Loss-making SHORT trade
            {
                "action": "SHORT",
                "symbol": "ETH-USD",
                "entry_price": Decimal("3000.00"),
                "exit_price": Decimal("3200.00"),
                "size": Decimal("0.5"),
            },
        ]

        initial_balance = account.current_balance
        total_fees_paid = Decimal("0.00")
        total_realized_pnl = Decimal("0.00")

        for scenario in trade_scenarios:
            # Open position
            open_action = TradeAction(
                action=scenario["action"],
                size_pct=10,
                take_profit_pct=3.0,
                stop_loss_pct=2.0,
                rationale=f"Test {scenario['action']} trade",
            )

            open_order = account.execute_trade_action(
                open_action, scenario["symbol"], scenario["entry_price"]
            )

            assert open_order.status == OrderStatus.FILLED

            # Track fees
            total_fees_paid += account.fee_rate * (
                scenario["size"] * scenario["entry_price"]
            )

            # Close position
            close_action = TradeAction(
                action="CLOSE",
                size_pct=0,
                take_profit_pct=0,
                stop_loss_pct=0,
                rationale="Close test position",
            )

            close_order = account.execute_trade_action(
                close_action, scenario["symbol"], scenario["exit_price"]
            )

            assert close_order.status == OrderStatus.FILLED

            # Track additional closing fees
            total_fees_paid += account.fee_rate * (
                scenario["size"] * scenario["exit_price"]
            )

            # Calculate expected P&L
            if scenario["action"] == "LONG":
                pnl = scenario["size"] * (
                    scenario["exit_price"] - scenario["entry_price"]
                )
            else:  # SHORT
                pnl = scenario["size"] * (
                    scenario["entry_price"] - scenario["exit_price"]
                )

            total_realized_pnl += pnl

        # Verify no open positions remain
        assert len(account.open_trades) == 0
        assert account.margin_used == Decimal("0.00")

        # Verify balance reconciliation
        expected_balance = initial_balance + total_realized_pnl - total_fees_paid
        actual_balance = account.current_balance

        # Allow for small rounding differences
        balance_diff = abs(expected_balance - actual_balance)
        assert balance_diff < Decimal(
            "0.01"
        ), f"Balance mismatch: expected {expected_balance}, got {actual_balance}"

        # Verify account status consistency
        account_status = account.get_account_status(current_prices)
        assert account_status["current_balance"] == float(account.current_balance)
        assert account_status["unrealized_pnl"] == 0.0  # No open positions

    @pytest.mark.asyncio()
    async def test_balance_validation_integration(self, temp_data_dir):
        """Test balance validation across different scenarios."""
        account = PaperTradingAccount(
            starting_balance=Decimal("1000.00"),  # Small balance for testing limits
            data_dir=temp_data_dir,
        )

        # Test insufficient funds scenario
        large_trade = TradeAction(
            action="LONG",
            size_pct=200,  # Requesting 200% of balance
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Insufficient funds test",
        )

        current_price = Decimal("50000.00")
        order = account.execute_trade_action(large_trade, "BTC-USD", current_price)

        # Should fail due to insufficient funds
        assert order.status == OrderStatus.REJECTED

        # Balance should remain unchanged
        assert account.current_balance == Decimal("1000.00")
        assert account.margin_used == Decimal("0.00")
        assert len(account.open_trades) == 0

        # Test valid trade within limits
        valid_trade = TradeAction(
            action="LONG",
            size_pct=10,  # Reasonable 10%
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Valid trade test",
        )

        order = account.execute_trade_action(valid_trade, "BTC-USD", current_price)

        # Should succeed
        assert order.status == OrderStatus.FILLED
        assert account.current_balance < Decimal("1000.00")  # Fees deducted
        assert account.margin_used > Decimal("0.00")  # Margin allocated
        assert len(account.open_trades) == 1

    @pytest.mark.asyncio()
    async def test_balance_performance_monitoring(
        self, mock_market_data, temp_data_dir
    ):
        """Test performance monitoring for balance operations."""
        account = PaperTradingAccount(
            starting_balance=Decimal("10000.00"), data_dir=temp_data_dir
        )

        # Measure performance of balance operations
        performance_metrics = {}

        # Test account status performance
        start_time = time.perf_counter()
        for _ in range(100):
            account.get_account_status()
        status_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["account_status_100_calls"] = status_time

        # Test trade execution performance
        start_time = time.perf_counter()
        for i in range(10):
            trade_action = TradeAction(
                action="LONG" if i % 2 == 0 else "SHORT",
                size_pct=5,
                take_profit_pct=2.0,
                stop_loss_pct=1.5,
                rationale=f"Performance test trade {i}",
            )
            account.execute_trade_action(trade_action, "BTC-USD", Decimal("50000.00"))
        trade_execution_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["trade_execution_10_trades"] = trade_execution_time

        # Test state save performance
        start_time = time.perf_counter()
        account._save_state()
        save_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["state_save"] = save_time

        # Test state load performance
        start_time = time.perf_counter()
        new_account = PaperTradingAccount(data_dir=temp_data_dir)
        load_time = (time.perf_counter() - start_time) * 1000
        performance_metrics["state_load"] = load_time

        # Performance assertions (reasonable thresholds)
        assert (
            performance_metrics["account_status_100_calls"] < 1000
        ), "Account status calls too slow"
        assert (
            performance_metrics["trade_execution_10_trades"] < 2000
        ), "Trade execution too slow"
        assert performance_metrics["state_save"] < 500, "State save too slow"
        assert performance_metrics["state_load"] < 500, "State load too slow"

        # Log performance metrics for monitoring
        print("\nBalance Performance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"  {metric}: {value:.2f}ms")

    @pytest.mark.asyncio()
    async def test_balance_edge_cases(self, temp_data_dir):
        """Test balance handling in edge cases."""
        account = PaperTradingAccount(
            starting_balance=Decimal("100.00"),  # Very small balance
            data_dir=temp_data_dir,
        )

        # Test zero-size trade
        zero_trade = TradeAction(
            action="LONG",
            size_pct=0,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Zero size test",
        )

        order = account.execute_trade_action(zero_trade, "BTC-USD", Decimal("50000.00"))
        assert order is None  # Should return None for zero-size trades

        # Test extremely small trade
        micro_trade = TradeAction(
            action="LONG",
            size_pct=0.01,  # 0.01% of $100 = $0.01
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Micro trade test",
        )

        order = account.execute_trade_action(
            micro_trade, "BTC-USD", Decimal("50000.00")
        )
        # Should handle micro trades gracefully (might fail due to minimum size requirements)

        # Test precision edge cases with very precise decimal values
        precise_trade = TradeAction(
            action="LONG",
            size_pct=33.333333,  # Non-terminating decimal
            take_profit_pct=2.123456,
            stop_loss_pct=1.987654,
            rationale="Precision test",
        )

        order = account.execute_trade_action(
            precise_trade, "BTC-USD", Decimal("50000.123456")
        )

        # Verify all balance values are properly normalized
        account_status = account.get_account_status()
        for key, value in account_status.items():
            if isinstance(value, float):
                # Check that float values have reasonable precision
                assert (
                    abs(value - round(value, 8)) < 1e-6
                ), f"Excessive precision in {key}: {value}"

    def _create_mock_dataframe(self, market_data):
        """Convert market data to DataFrame for testing."""
        import pandas as pd

        data = []
        for candle in market_data:
            data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
