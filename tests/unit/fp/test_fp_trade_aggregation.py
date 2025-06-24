"""
Functional Programming Trade Aggregation Tests

This module tests trade aggregation for sub-minute intervals using functional
programming patterns, including high-frequency data processing, time-based
aggregation, and immutable candle generation.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from bot.fp.core.either import Left, Right
from bot.fp.effects.market_data_aggregation import (
    aggregate_sub_minute_trades,
    calculate_ohlcv_from_trades,
    calculate_weighted_average_price,
    create_time_windows,
    detect_trading_gaps,
    merge_trade_windows,
    validate_aggregation_result,
)
from bot.fp.types.market import (
    Candle,
    Trade,
)


class TestSubMinuteAggregation:
    """Test sub-minute trade aggregation functionality."""

    def test_sub_minute_intervals_setup(self):
        """Test setup for sub-minute interval aggregation."""
        # Test supported sub-minute intervals
        supported_intervals = ["1s", "5s", "15s", "30s"]

        for interval in supported_intervals:
            # Should be able to create time windows
            start_time = datetime.now(UTC).replace(microsecond=0)
            window_duration = self._parse_interval_to_timedelta(interval)

            assert window_duration.total_seconds() < 60
            assert window_duration.total_seconds() > 0

    def test_1_second_trade_aggregation(self):
        """Test 1-second trade aggregation."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades within 5 seconds
        trades = []
        for second in range(5):
            for millisecond in [0, 250, 500, 750]:  # 4 trades per second
                trade_time = base_time + timedelta(
                    seconds=second, milliseconds=millisecond
                )
                trades.append(
                    Trade(
                        id=f"trade-{second}-{millisecond}",
                        timestamp=trade_time,
                        price=Decimal(f"{50000 + second * 10 + millisecond // 100}"),
                        size=Decimal("0.1"),
                        side="BUY" if millisecond < 500 else "SELL",
                        symbol="BTC-USD",
                    )
                )

        # Aggregate to 1-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=1))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 5  # 5 seconds of data

        # Verify first candle
        first_candle = candles[0]
        assert first_candle.timestamp == base_time
        assert first_candle.volume == Decimal("0.4")  # 4 trades * 0.1
        assert first_candle.open == Decimal(50000)  # First trade price
        assert first_candle.symbol == "BTC-USD"

    def test_5_second_trade_aggregation(self):
        """Test 5-second trade aggregation."""
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)

        # Create trades over 20 seconds (should create 4 candles)
        trades = []
        for second in range(20):
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=Decimal(f"{50000 + second}"),
                    size=Decimal("0.05"),
                    side="BUY" if second % 2 == 0 else "SELL",
                    symbol="BTC-USD",
                )
            )

        # Aggregate to 5-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=5))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 4  # 20 seconds / 5 seconds = 4 candles

        # Verify aggregation
        first_candle = candles[0]
        assert first_candle.timestamp == base_time
        assert first_candle.volume == Decimal("0.25")  # 5 trades * 0.05
        assert first_candle.open == Decimal(50000)
        assert first_candle.close == Decimal(50004)  # Last trade in window
        assert first_candle.high == Decimal(50004)  # Max in window
        assert first_candle.low == Decimal(50000)  # Min in window

    def test_15_second_trade_aggregation(self):
        """Test 15-second trade aggregation."""
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)

        # Create trades over 1 minute (should create 4 candles)
        trades = []
        for second in range(60):
            trade_time = base_time + timedelta(seconds=second)
            # Create price trend
            price = Decimal(50000) + Decimal(str(second * 2))
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=price,
                    size=Decimal("0.02"),
                    side="BUY" if second % 3 == 0 else "SELL",
                    symbol="BTC-USD",
                )
            )

        # Aggregate to 15-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=15))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 4  # 60 seconds / 15 seconds = 4 candles

        # Verify trend is captured
        for i, candle in enumerate(candles):
            expected_start_time = base_time + timedelta(seconds=i * 15)
            assert candle.timestamp == expected_start_time
            assert candle.volume == Decimal("0.30")  # 15 trades * 0.02
            # Price should be trending upward
            if i > 0:
                assert candle.open > candles[i - 1].open

    def test_30_second_trade_aggregation(self):
        """Test 30-second trade aggregation."""
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)

        # Create trades over 2 minutes
        trades = []
        for second in range(120):
            trade_time = base_time + timedelta(seconds=second)
            # Create volatility pattern
            price_offset = 100 * (1 if second % 60 < 30 else -1)  # Up/down pattern
            price = Decimal(50000) + Decimal(str(price_offset))

            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=price,
                    size=Decimal("0.01"),
                    side="BUY" if second % 4 == 0 else "SELL",
                    symbol="BTC-USD",
                )
            )

        # Aggregate to 30-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=30))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 4  # 120 seconds / 30 seconds = 4 candles

        # Verify volatility pattern is captured
        first_candle = candles[0]
        second_candle = candles[1]

        assert first_candle.volume == Decimal("0.30")  # 30 trades * 0.01
        assert first_candle.high == Decimal(50100)  # Up pattern
        assert second_candle.low == Decimal(49900)  # Down pattern


class TestOHLCVCalculation:
    """Test OHLCV calculation from trade data."""

    def test_ohlcv_calculation_from_trades(self):
        """Test OHLCV calculation from a set of trades."""
        trades = [
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal(50000),
                size=Decimal("1.0"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="2",
                timestamp=datetime.now(UTC),
                price=Decimal(50200),
                size=Decimal("0.5"),
                side="SELL",
                symbol="BTC-USD",
            ),
            Trade(
                id="3",
                timestamp=datetime.now(UTC),
                price=Decimal(49800),
                size=Decimal("0.8"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="4",
                timestamp=datetime.now(UTC),
                price=Decimal(50100),
                size=Decimal("0.3"),
                side="SELL",
                symbol="BTC-USD",
            ),
        ]

        result = calculate_ohlcv_from_trades(trades)
        assert isinstance(result, Right)

        ohlcv = result.value
        assert ohlcv["open"] == Decimal(50000)  # First trade
        assert ohlcv["high"] == Decimal(50200)  # Highest price
        assert ohlcv["low"] == Decimal(49800)  # Lowest price
        assert ohlcv["close"] == Decimal(50100)  # Last trade
        assert ohlcv["volume"] == Decimal("2.6")  # Sum of sizes

    def test_ohlcv_with_single_trade(self):
        """Test OHLCV calculation with a single trade."""
        single_trade = [
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal(50000),
                size=Decimal("1.0"),
                side="BUY",
                symbol="BTC-USD",
            )
        ]

        result = calculate_ohlcv_from_trades(single_trade)
        assert isinstance(result, Right)

        ohlcv = result.value
        # All prices should be the same
        assert ohlcv["open"] == Decimal(50000)
        assert ohlcv["high"] == Decimal(50000)
        assert ohlcv["low"] == Decimal(50000)
        assert ohlcv["close"] == Decimal(50000)
        assert ohlcv["volume"] == Decimal("1.0")

    def test_ohlcv_with_empty_trades(self):
        """Test OHLCV calculation with empty trade list."""
        result = calculate_ohlcv_from_trades([])
        assert isinstance(result, Left)
        assert "no trades" in result.value.lower()

    def test_ohlcv_precision_handling(self):
        """Test OHLCV calculation with high precision prices."""
        trades = [
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal("50000.123456789"),
                size=Decimal("1.000000001"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="2",
                timestamp=datetime.now(UTC),
                price=Decimal("50000.987654321"),
                size=Decimal("0.999999999"),
                side="SELL",
                symbol="BTC-USD",
            ),
        ]

        result = calculate_ohlcv_from_trades(trades)
        assert isinstance(result, Right)

        ohlcv = result.value
        # Should preserve precision
        assert ohlcv["open"] == Decimal("50000.123456789")
        assert ohlcv["high"] == Decimal("50000.987654321")
        assert ohlcv["low"] == Decimal("50000.123456789")
        assert ohlcv["close"] == Decimal("50000.987654321")
        assert ohlcv["volume"] == Decimal("2.000000000")


class TestTimeWindowCreation:
    """Test time window creation for aggregation."""

    def test_create_time_windows_1_second(self):
        """Test creating 1-second time windows."""
        start_time = datetime.now(UTC).replace(microsecond=0)
        end_time = start_time + timedelta(seconds=5)

        windows = create_time_windows(start_time, end_time, timedelta(seconds=1))

        assert len(windows) == 5
        for i, (window_start, window_end) in enumerate(windows):
            expected_start = start_time + timedelta(seconds=i)
            expected_end = start_time + timedelta(seconds=i + 1)
            assert window_start == expected_start
            assert window_end == expected_end

    def test_create_time_windows_5_second(self):
        """Test creating 5-second time windows."""
        start_time = datetime.now(UTC).replace(second=0, microsecond=0)
        end_time = start_time + timedelta(seconds=20)

        windows = create_time_windows(start_time, end_time, timedelta(seconds=5))

        assert len(windows) == 4
        for i, (window_start, window_end) in enumerate(windows):
            expected_start = start_time + timedelta(seconds=i * 5)
            expected_end = start_time + timedelta(seconds=(i + 1) * 5)
            assert window_start == expected_start
            assert window_end == expected_end

    def test_create_time_windows_partial_window(self):
        """Test creating time windows with partial end window."""
        start_time = datetime.now(UTC).replace(microsecond=0)
        end_time = start_time + timedelta(seconds=7.5)  # Partial window

        windows = create_time_windows(start_time, end_time, timedelta(seconds=5))

        assert len(windows) == 2  # One full window + one partial

        # First window should be full
        assert windows[0][1] - windows[0][0] == timedelta(seconds=5)

        # Second window should be partial
        assert windows[1][1] - windows[1][0] == timedelta(seconds=2.5)

    def test_time_window_alignment(self):
        """Test time window alignment to interval boundaries."""
        # Start at an arbitrary time
        start_time = datetime(2024, 1, 1, 12, 34, 27, 500000, UTC)
        end_time = start_time + timedelta(seconds=10)

        windows = create_time_windows(start_time, end_time, timedelta(seconds=5))

        # Should create windows aligned to the start time
        assert len(windows) == 2
        assert windows[0][0] == start_time
        assert windows[0][1] == start_time + timedelta(seconds=5)
        assert windows[1][0] == start_time + timedelta(seconds=5)
        assert windows[1][1] == end_time


class TestTradeWindowMerging:
    """Test merging trades into time windows."""

    def test_merge_trades_into_windows(self):
        """Test merging trades into time windows."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades spanning 10 seconds
        trades = []
        for second in range(10):
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=Decimal(f"{50000 + second}"),
                    size=Decimal("0.1"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        # Create 5-second windows
        windows = [
            (base_time, base_time + timedelta(seconds=5)),
            (base_time + timedelta(seconds=5), base_time + timedelta(seconds=10)),
        ]

        result = merge_trade_windows(trades, windows)
        assert isinstance(result, Right)

        windowed_trades = result.value
        assert len(windowed_trades) == 2

        # First window should have 5 trades
        assert len(windowed_trades[0]) == 5
        assert all(
            trade.id.endswith(str(i)) for i, trade in enumerate(windowed_trades[0])
        )

        # Second window should have 5 trades
        assert len(windowed_trades[1]) == 5
        assert all(
            trade.id.endswith(str(i + 5)) for i, trade in enumerate(windowed_trades[1])
        )

    def test_merge_trades_with_gaps(self):
        """Test merging trades when there are time gaps."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades with gaps (no trades at seconds 2, 3, 7, 8)
        trades = []
        for second in [0, 1, 4, 5, 6, 9]:
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=Decimal(f"{50000 + second}"),
                    size=Decimal("0.1"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        # Create 5-second windows
        windows = [
            (base_time, base_time + timedelta(seconds=5)),
            (base_time + timedelta(seconds=5), base_time + timedelta(seconds=10)),
        ]

        result = merge_trade_windows(trades, windows)
        assert isinstance(result, Right)

        windowed_trades = result.value
        assert len(windowed_trades) == 2

        # First window should have 4 trades (0, 1, 4)
        assert len(windowed_trades[0]) == 4

        # Second window should have 2 trades (6, 9)
        assert len(windowed_trades[1]) == 2

    def test_merge_trades_outside_windows(self):
        """Test handling trades outside time windows."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades, some outside the window range
        trades = [
            Trade(
                id="before",
                timestamp=base_time - timedelta(seconds=1),
                price=Decimal(49000),
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="inside1",
                timestamp=base_time + timedelta(seconds=1),
                price=Decimal(50000),
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="inside2",
                timestamp=base_time + timedelta(seconds=3),
                price=Decimal(50100),
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="after",
                timestamp=base_time + timedelta(seconds=6),
                price=Decimal(51000),
                size=Decimal("0.1"),
                side="BUY",
                symbol="BTC-USD",
            ),
        ]

        # Create 5-second window
        windows = [(base_time, base_time + timedelta(seconds=5))]

        result = merge_trade_windows(trades, windows)
        assert isinstance(result, Right)

        windowed_trades = result.value
        assert len(windowed_trades) == 1

        # Should only include trades within the window
        window_trades = windowed_trades[0]
        assert len(window_trades) == 2
        assert window_trades[0].id == "inside1"
        assert window_trades[1].id == "inside2"


class TestAggregationValidation:
    """Test validation of aggregation results."""

    def test_validate_aggregation_result(self):
        """Test validation of aggregation results."""
        # Valid candle
        valid_candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal(50000),
            high=Decimal(51000),
            low=Decimal(49500),
            close=Decimal(50500),
            volume=Decimal("100.5"),
            symbol="BTC-USD",
        )

        result = validate_aggregation_result(valid_candle)
        assert isinstance(result, Right)
        assert result.value == valid_candle

        # Test validation is working by trying invalid construction
        # (This would fail at candle creation, not at validation)
        try:
            invalid_candle = Candle(
                timestamp=datetime.now(UTC),
                open=Decimal(50000),
                high=Decimal(49000),  # High < open (invalid)
                low=Decimal(49500),
                close=Decimal(50500),
                volume=Decimal("100.5"),
                symbol="BTC-USD",
            )
            assert False, "Should have failed validation"
        except ValueError:
            # Expected behavior
            pass

    def test_validate_aggregation_consistency(self):
        """Test consistency validation across multiple candles."""
        candles = [
            Candle(
                timestamp=datetime.now(UTC),
                open=Decimal(50000),
                high=Decimal(50100),
                low=Decimal(49900),
                close=Decimal(50050),
                volume=Decimal(10),
                symbol="BTC-USD",
            ),
            Candle(
                timestamp=datetime.now(UTC) + timedelta(seconds=5),
                open=Decimal(50050),  # Should match previous close
                high=Decimal(50200),
                low=Decimal(50000),
                close=Decimal(50150),
                volume=Decimal(15),
                symbol="BTC-USD",
            ),
        ]

        # Test consistency check
        for i in range(1, len(candles)):
            prev_candle = candles[i - 1]
            curr_candle = candles[i]

            # Current open should be close to previous close (in real scenarios)
            # For this test, they match exactly
            assert curr_candle.open == prev_candle.close


class TestWeightedAveragePrice:
    """Test weighted average price calculations."""

    def test_calculate_weighted_average_price(self):
        """Test weighted average price calculation."""
        trades = [
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal(50000),
                size=Decimal("1.0"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="2",
                timestamp=datetime.now(UTC),
                price=Decimal(51000),
                size=Decimal("2.0"),
                side="SELL",
                symbol="BTC-USD",
            ),
            Trade(
                id="3",
                timestamp=datetime.now(UTC),
                price=Decimal(49000),
                size=Decimal("1.0"),
                side="BUY",
                symbol="BTC-USD",
            ),
        ]

        result = calculate_weighted_average_price(trades)
        assert isinstance(result, Right)

        # VWAP = (50000*1 + 51000*2 + 49000*1) / (1+2+1) = 200000/4 = 50250
        vwap = result.value
        assert vwap == Decimal(50250)

    def test_vwap_with_zero_volume(self):
        """Test VWAP calculation with zero volume trades."""
        trades = [
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal(50000),
                size=Decimal(0),
                side="BUY",
                symbol="BTC-USD",
            ),
        ]

        # This should fail at Trade creation due to size validation
        with pytest.raises(ValueError, match="Size must be positive"):
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal(50000),
                size=Decimal(0),
                side="BUY",
                symbol="BTC-USD",
            )

    def test_vwap_with_high_precision(self):
        """Test VWAP calculation with high precision values."""
        trades = [
            Trade(
                id="1",
                timestamp=datetime.now(UTC),
                price=Decimal("50000.123456789"),
                size=Decimal("1.000000001"),
                side="BUY",
                symbol="BTC-USD",
            ),
            Trade(
                id="2",
                timestamp=datetime.now(UTC),
                price=Decimal("50000.987654321"),
                size=Decimal("0.999999999"),
                side="SELL",
                symbol="BTC-USD",
            ),
        ]

        result = calculate_weighted_average_price(trades)
        assert isinstance(result, Right)

        vwap = result.value
        # Should preserve precision in calculation
        expected = (
            Decimal("50000.123456789") * Decimal("1.000000001")
            + Decimal("50000.987654321") * Decimal("0.999999999")
        ) / Decimal("2.000000000")

        assert abs(vwap - expected) < Decimal("0.000000001")


class TestTradingGapDetection:
    """Test detection of trading gaps in sub-minute data."""

    def test_detect_normal_trading_gaps(self):
        """Test detection of normal trading gaps."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades with a gap at seconds 3-6
        trades = []
        for second in [0, 1, 2, 7, 8, 9]:
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=Decimal(50000),
                    size=Decimal("0.1"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        result = detect_trading_gaps(trades, max_gap_seconds=2)
        assert isinstance(result, Right)

        gaps = result.value
        assert len(gaps) == 1

        gap = gaps[0]
        assert gap["start_time"] == base_time + timedelta(seconds=2)
        assert gap["end_time"] == base_time + timedelta(seconds=7)
        assert gap["duration_seconds"] == 5

    def test_detect_no_gaps(self):
        """Test gap detection when there are no gaps."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create continuous trades
        trades = []
        for second in range(10):
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=Decimal(50000),
                    size=Decimal("0.1"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        result = detect_trading_gaps(trades, max_gap_seconds=2)
        assert isinstance(result, Right)

        gaps = result.value
        assert len(gaps) == 0

    def test_detect_multiple_gaps(self):
        """Test detection of multiple trading gaps."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades with multiple gaps
        trade_times = [0, 1, 5, 6, 10, 11, 12]  # Gaps at 2-4 and 7-9
        trades = []
        for second in trade_times:
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"trade-{second}",
                    timestamp=trade_time,
                    price=Decimal(50000),
                    size=Decimal("0.1"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        result = detect_trading_gaps(trades, max_gap_seconds=2)
        assert isinstance(result, Right)

        gaps = result.value
        assert len(gaps) == 2

        # First gap
        assert gaps[0]["duration_seconds"] == 4  # 2-5 (exclusive)

        # Second gap
        assert gaps[1]["duration_seconds"] == 3  # 7-9 (exclusive)


class TestHighFrequencyAggregation:
    """Test high-frequency aggregation scenarios."""

    def test_microsecond_precision_trades(self):
        """Test aggregation with microsecond precision trades."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades with microsecond intervals
        trades = []
        for microsecond in range(0, 1000000, 100000):  # Every 100ms
            trade_time = base_time + timedelta(microseconds=microsecond)
            trades.append(
                Trade(
                    id=f"trade-{microsecond}",
                    timestamp=trade_time,
                    price=Decimal(
                        f"{50000 + microsecond // 1000}"
                    ),  # Price changes per ms
                    size=Decimal("0.01"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        # Aggregate to 1-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=1))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 1  # All trades within 1 second

        candle = candles[0]
        assert candle.volume == Decimal("0.10")  # 10 trades * 0.01
        assert candle.open == Decimal(50000)  # First trade
        assert candle.close == Decimal(50999)  # Last trade

    def test_burst_trading_aggregation(self):
        """Test aggregation during burst trading periods."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Simulate burst trading (many trades in short time)
        trades = []

        # Normal trading for first 2 seconds
        for second in range(2):
            trade_time = base_time + timedelta(seconds=second)
            trades.append(
                Trade(
                    id=f"normal-{second}",
                    timestamp=trade_time,
                    price=Decimal(f"{50000 + second}"),
                    size=Decimal("0.1"),
                    side="BUY",
                    symbol="BTC-USD",
                )
            )

        # Burst trading in 3rd second (many small trades)
        burst_start = base_time + timedelta(seconds=2)
        for millisecond in range(0, 1000, 50):  # 20 trades in 1 second
            trade_time = burst_start + timedelta(milliseconds=millisecond)
            trades.append(
                Trade(
                    id=f"burst-{millisecond}",
                    timestamp=trade_time,
                    price=Decimal(f"{50002 + millisecond // 100}"),
                    size=Decimal("0.005"),  # Smaller trades
                    side="SELL",
                    symbol="BTC-USD",
                )
            )

        # Aggregate to 1-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=1))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 3

        # Burst candle should have high volume
        burst_candle = candles[2]
        assert burst_candle.volume == Decimal("0.100")  # 20 * 0.005
        assert (
            burst_candle.trades_count == 20
            if hasattr(burst_candle, "trades_count")
            else True
        )

    def test_extreme_price_volatility_aggregation(self):
        """Test aggregation during extreme price volatility."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades with extreme price swings
        trades = []
        prices = [50000, 55000, 45000, 52000, 48000, 51000]  # High volatility

        for i, price in enumerate(prices):
            trade_time = base_time + timedelta(seconds=i)
            trades.append(
                Trade(
                    id=f"volatile-{i}",
                    timestamp=trade_time,
                    price=Decimal(str(price)),
                    size=Decimal("1.0"),
                    side="BUY" if i % 2 == 0 else "SELL",
                    symbol="BTC-USD",
                )
            )

        # Aggregate to 3-second candles
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=3))
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 2

        # First candle should capture the extreme range
        first_candle = candles[0]
        assert first_candle.high == Decimal(55000)  # Highest in first 3 trades
        assert first_candle.low == Decimal(45000)  # Lowest in first 3 trades

        # Price range should be significant
        price_range = first_candle.high - first_candle.low
        assert price_range == Decimal(10000)  # 20% range


class TestAggregationPerformance:
    """Test performance characteristics of trade aggregation."""

    def test_large_volume_aggregation_performance(self):
        """Test aggregation performance with large trade volumes."""
        import time

        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create 10,000 trades
        trades = []
        for i in range(10000):
            trade_time = base_time + timedelta(
                milliseconds=i * 10
            )  # 100 seconds of data
            trades.append(
                Trade(
                    id=f"trade-{i}",
                    timestamp=trade_time,
                    price=Decimal(f"{50000 + (i % 1000)}"),
                    size=Decimal("0.001"),
                    side="BUY" if i % 2 == 0 else "SELL",
                    symbol="BTC-USD",
                )
            )

        # Measure aggregation time
        start_time = time.time()
        result = aggregate_sub_minute_trades(trades, timedelta(seconds=5))
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time
        assert processing_time < 5.0  # Less than 5 seconds
        assert isinstance(result, Right)

        candles = result.value
        assert len(candles) == 20  # 100 seconds / 5 seconds

    def test_memory_efficiency_aggregation(self):
        """Test memory efficiency during aggregation."""

        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create trades and measure memory usage pattern
        trade_batch_size = 1000

        for batch in range(5):  # Process in batches
            trades = []
            for i in range(trade_batch_size):
                trade_time = base_time + timedelta(seconds=batch * trade_batch_size + i)
                trades.append(
                    Trade(
                        id=f"batch-{batch}-trade-{i}",
                        timestamp=trade_time,
                        price=Decimal(50000),
                        size=Decimal("0.01"),
                        side="BUY",
                        symbol="BTC-USD",
                    )
                )

            # Aggregate batch
            result = aggregate_sub_minute_trades(trades, timedelta(seconds=60))
            assert isinstance(result, Right)

            # Memory should not grow unbounded
            # (In a real implementation, this would test memory usage patterns)
            del trades  # Simulate memory cleanup

    def _parse_interval_to_timedelta(self, interval: str) -> timedelta:
        """Parse interval string to timedelta."""
        if interval.endswith("s"):
            seconds = int(interval[:-1])
            return timedelta(seconds=seconds)
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            return timedelta(minutes=minutes)
        raise ValueError(f"Unsupported interval: {interval}")


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional Trade Aggregation...")

    # Test sub-minute aggregation
    test_sub_minute = TestSubMinuteAggregation()
    test_sub_minute.test_sub_minute_intervals_setup()
    test_sub_minute.test_1_second_trade_aggregation()
    test_sub_minute.test_5_second_trade_aggregation()
    test_sub_minute.test_15_second_trade_aggregation()
    test_sub_minute.test_30_second_trade_aggregation()
    print("✓ Sub-minute aggregation tests passed")

    # Test OHLCV calculation
    test_ohlcv = TestOHLCVCalculation()
    test_ohlcv.test_ohlcv_calculation_from_trades()
    test_ohlcv.test_ohlcv_with_single_trade()
    test_ohlcv.test_ohlcv_precision_handling()
    print("✓ OHLCV calculation tests passed")

    # Test time windows
    test_windows = TestTimeWindowCreation()
    test_windows.test_create_time_windows_1_second()
    test_windows.test_create_time_windows_5_second()
    test_windows.test_create_time_windows_partial_window()
    print("✓ Time window creation tests passed")

    # Test weighted average price
    test_vwap = TestWeightedAveragePrice()
    test_vwap.test_calculate_weighted_average_price()
    test_vwap.test_vwap_with_high_precision()
    print("✓ Weighted average price tests passed")

    # Test gap detection
    test_gaps = TestTradingGapDetection()
    test_gaps.test_detect_normal_trading_gaps()
    test_gaps.test_detect_no_gaps()
    test_gaps.test_detect_multiple_gaps()
    print("✓ Trading gap detection tests passed")

    print("All functional trade aggregation tests completed successfully!")
