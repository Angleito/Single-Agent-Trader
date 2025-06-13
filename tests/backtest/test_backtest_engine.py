"""Tests for the backtest engine."""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bot.backtest.engine import BacktestEngine, BacktestResults, BacktestTrade


class TestBacktestEngine:
    """Test cases for the backtest engine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for backtesting."""
        # Create 1000 candles of sample data
        dates = pd.date_range("2024-01-01", periods=1000, freq="1h")

        # Generate realistic price data with some trend
        base_price = 50000
        price_change = np.random.normal(
            0, 0.002, 1000
        ).cumsum()  # 0.2% std dev per hour
        prices = base_price * (1 + price_change)

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + np.random.uniform(0, 0.01, 1000)),
                "low": prices * (1 - np.random.uniform(0, 0.01, 1000)),
                "close": prices * (1 + np.random.normal(0, 0.005, 1000)),
                "volume": np.random.uniform(10, 100, 1000),
            },
            index=dates,
        )

        # Ensure high >= max(open, close) and low <= min(open, close)
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))

        return data

    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization."""
        engine = BacktestEngine(initial_balance=Decimal("10000"))

        assert engine.initial_balance == Decimal("10000")
        assert engine.current_balance == Decimal("10000")
        assert len(engine.trades) == 0
        assert engine.current_position is None

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, sample_data):
        """Test basic backtest execution."""
        engine = BacktestEngine(initial_balance=Decimal("10000"))

        # Run backtest on subset of data
        start_date = sample_data.index[100]  # Skip some data for indicator warmup
        end_date = sample_data.index[500]

        results = await engine.run_backtest(
            historical_data=sample_data,
            start_date=start_date,
            end_date=end_date,
            strategy_type="core",
        )

        # Verify results structure
        assert isinstance(results, BacktestResults)
        assert results.start_date == start_date
        assert results.end_date == end_date
        assert results.total_trades >= 0
        assert isinstance(results.trades, list)

    @pytest.mark.asyncio
    async def test_backtest_with_llm_strategy(self, sample_data):
        """Test backtest with LLM strategy (fallback mode)."""
        engine = BacktestEngine()

        # Run short backtest
        start_date = sample_data.index[50]
        end_date = sample_data.index[150]

        results = await engine.run_backtest(
            historical_data=sample_data,
            start_date=start_date,
            end_date=end_date,
            strategy_type="llm",
        )

        # Should complete without errors
        assert isinstance(results, BacktestResults)

    def test_prepare_data(self, sample_data):
        """Test data preparation for backtesting."""
        engine = BacktestEngine()

        start_date = sample_data.index[100]
        end_date = sample_data.index[200]

        prepared_data = engine._prepare_data(sample_data, start_date, end_date)

        assert len(prepared_data) == 101  # Inclusive range
        assert prepared_data.index[0] == start_date
        assert prepared_data.index[-1] == end_date

    def test_prepare_data_no_date_filter(self, sample_data):
        """Test data preparation without date filtering."""
        engine = BacktestEngine()

        prepared_data = engine._prepare_data(sample_data)

        assert len(prepared_data) == len(sample_data)

    def test_reset_state(self):
        """Test backtest state reset."""
        engine = BacktestEngine(initial_balance=Decimal("10000"))

        # Modify state
        engine.current_balance = Decimal("15000")
        engine.trades = [BacktestTrade(entry_time=datetime.now(), symbol="BTC-USD")]
        engine.current_position = BacktestTrade(
            entry_time=datetime.now(), symbol="BTC-USD"
        )

        # Reset
        engine._reset_state()

        assert engine.current_balance == Decimal("10000")
        assert len(engine.trades) == 0
        assert engine.current_position is None

    @pytest.mark.asyncio
    async def test_trade_execution_flow(self):
        """Test trade execution in backtest."""
        engine = BacktestEngine()

        from bot.types import TradeAction

        # Test opening a position
        long_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Test long",
        )

        timestamp = datetime.now()
        market_data = pd.Series(
            {"open": 50000, "high": 50100, "low": 49900, "close": 50050, "volume": 25}
        )

        await engine._execute_backtest_trade(long_action, timestamp, market_data)

        # Should have opened a position
        assert engine.current_position is not None
        assert engine.current_position.side == "LONG"
        assert engine.current_position.entry_price == Decimal("50050")

    @pytest.mark.asyncio
    async def test_position_closing(self):
        """Test position closing in backtest."""
        engine = BacktestEngine()

        # Manually create a position
        engine.current_position = BacktestTrade(
            entry_time=datetime.now(),
            symbol="BTC-USD",
            side="LONG",
            entry_price=Decimal("50000"),
            size=Decimal("1.0"),
        )

        initial_balance = engine.current_balance

        # Close position at profit
        await engine._close_backtest_position(
            timestamp=datetime.now(),
            price=Decimal("51000"),  # 2% profit
            reason="Take Profit",
        )

        # Position should be closed
        assert engine.current_position is None
        assert len(engine.trades) == 1
        assert engine.current_balance > initial_balance

        # Check trade record
        trade = engine.trades[0]
        assert trade.exit_price == Decimal("51000")
        assert trade.pnl > 0
        assert trade.exit_reason == "Take Profit"

    @pytest.mark.asyncio
    async def test_stop_loss_execution(self):
        """Test stop loss execution."""
        engine = BacktestEngine()

        # Create a long position
        engine.current_position = BacktestTrade(
            entry_time=datetime.now(),
            symbol="BTC-USD",
            side="LONG",
            entry_price=Decimal("50000"),
            size=Decimal("1.0"),
        )

        # Market data that should trigger stop loss
        market_data = pd.Series(
            {
                "open": 49000,
                "high": 49100,
                "low": 48900,
                "close": 49000,  # 2% loss
                "volume": 25,
            }
        )

        await engine._check_exit_conditions(datetime.now(), market_data)

        # Position should be closed
        assert engine.current_position is None
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "Stop Loss"

    @pytest.mark.asyncio
    async def test_take_profit_execution(self):
        """Test take profit execution."""
        engine = BacktestEngine()

        # Create a short position
        engine.current_position = BacktestTrade(
            entry_time=datetime.now(),
            symbol="BTC-USD",
            side="SHORT",
            entry_price=Decimal("50000"),
            size=Decimal("1.0"),
        )

        # Market data that should trigger take profit
        market_data = pd.Series(
            {
                "open": 49000,
                "high": 49100,
                "low": 48900,
                "close": 49000,  # 2% profit for short
                "volume": 25,
            }
        )

        await engine._check_exit_conditions(datetime.now(), market_data)

        # Position should be closed
        assert engine.current_position is None
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "Take Profit"

    def test_calculate_results_empty(self):
        """Test results calculation with no trades."""
        engine = BacktestEngine()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        results = engine._calculate_results(start_date, end_date)

        assert results.total_trades == 0
        assert results.win_rate == 0
        assert results.total_pnl == Decimal("0")

    def test_calculate_results_with_trades(self):
        """Test results calculation with sample trades."""
        engine = BacktestEngine()

        # Add sample trades
        engine.trades = [
            BacktestTrade(
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 1, 1),
                symbol="BTC-USD",
                side="LONG",
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                size=Decimal("1.0"),
                pnl=Decimal("1000"),
                duration_minutes=60,
            ),
            BacktestTrade(
                entry_time=datetime(2024, 1, 2),
                exit_time=datetime(2024, 1, 2, 1),
                symbol="BTC-USD",
                side="SHORT",
                entry_price=Decimal("51000"),
                exit_price=Decimal("51500"),
                size=Decimal("1.0"),
                pnl=Decimal("-500"),
                duration_minutes=60,
            ),
        ]

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        results = engine._calculate_results(start_date, end_date)

        assert results.total_trades == 2
        assert results.winning_trades == 1
        assert results.losing_trades == 1
        assert results.win_rate == 50.0
        assert results.total_pnl == Decimal("500")
        assert results.avg_trade_duration == 60
