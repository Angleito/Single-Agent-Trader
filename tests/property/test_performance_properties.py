"""
Property-based tests for performance characteristics and computational properties.

Tests algorithmic complexity, memory usage patterns, and performance invariants
for market data processing and orderbook operations.
"""

import sys
import time
from datetime import UTC, datetime
from decimal import Decimal

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from bot.fp.types.market import Candle, MarketData, OrderBook, Trade


# Performance measurement utilities
class PerformanceTimer:
    """Context manager for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


def get_object_size(obj) -> int:
    """Get approximate size of object in bytes."""
    return sys.getsizeof(obj)


# Strategies for performance testing
@st.composite
def large_orderbook_strategy(draw, min_levels=10, max_levels=1000):
    """Generate large orderbooks for performance testing."""
    num_bid_levels = draw(st.integers(min_value=min_levels, max_value=max_levels))
    num_ask_levels = draw(st.integers(min_value=min_levels, max_value=max_levels))

    base_price = Decimal("100.0")

    # Generate bids (descending prices)
    bids = []
    current_price = base_price - Decimal("1.0")
    for i in range(num_bid_levels):
        size = draw(st.floats(min_value=0.1, max_value=100.0))
        bids.append((current_price, Decimal(str(size))))
        current_price -= Decimal("0.01")

    # Generate asks (ascending prices)
    asks = []
    current_price = base_price + Decimal("1.0")
    for i in range(num_ask_levels):
        size = draw(st.floats(min_value=0.1, max_value=100.0))
        asks.append((current_price, Decimal(str(size))))
        current_price += Decimal("0.01")

    timestamp = datetime.now(UTC)
    return OrderBook(bids=bids, asks=asks, timestamp=timestamp)


@st.composite
def trade_sequence_strategy(draw, min_trades=10, max_trades=10000):
    """Generate sequences of trades for performance testing."""
    num_trades = draw(st.integers(min_value=min_trades, max_value=max_trades))

    trades = []
    base_time = datetime.now(UTC)

    for i in range(num_trades):
        trade_id = f"trade_{i}"
        timestamp = base_time
        price = draw(st.floats(min_value=1.0, max_value=1000.0))
        size = draw(st.floats(min_value=0.001, max_value=100.0))
        side = draw(st.sampled_from(["BUY", "SELL"]))

        trade = Trade(
            id=trade_id,
            timestamp=timestamp,
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            side=side,
        )
        trades.append(trade)

    return trades


class TestOrderBookPerformanceProperties:
    """Property-based tests for orderbook performance characteristics."""

    @given(large_orderbook_strategy(min_levels=10, max_levels=100))
    @settings(max_examples=20, deadline=5000)
    def test_orderbook_creation_time_complexity(self, orderbook: OrderBook):
        """Property: Orderbook creation should be O(n) in the number of levels."""
        num_levels = len(orderbook.bids) + len(orderbook.asks)

        with PerformanceTimer() as timer:
            # Test operations that should be O(1) or O(log n)
            _ = orderbook.best_bid
            _ = orderbook.best_ask
            _ = orderbook.mid_price
            _ = orderbook.spread

        # Operations should complete quickly regardless of orderbook size
        assert (
            timer.duration < 0.01
        ), f"Basic operations took too long: {timer.duration}s for {num_levels} levels"

    @given(large_orderbook_strategy(min_levels=50, max_levels=500))
    @settings(max_examples=15, deadline=5000)
    def test_orderbook_depth_calculation_performance(self, orderbook: OrderBook):
        """Property: Depth calculations should be linear in the number of levels."""
        num_levels = len(orderbook.bids) + len(orderbook.asks)

        with PerformanceTimer() as timer:
            bid_depth = orderbook.bid_depth
            ask_depth = orderbook.ask_depth

        # Depth calculation should be reasonable even for large orderbooks
        assert (
            timer.duration < 0.1
        ), f"Depth calculation took too long: {timer.duration}s for {num_levels} levels"

        # Results should be meaningful
        assert bid_depth > 0
        assert ask_depth > 0

    @given(
        large_orderbook_strategy(min_levels=20, max_levels=200),
        st.floats(min_value=0.1, max_value=1000.0),
    )
    @settings(max_examples=10, deadline=5000)
    def test_vwap_calculation_performance(
        self, orderbook: OrderBook, order_size: float
    ):
        """Property: VWAP calculation should be efficient for reasonable order sizes."""
        order_size_decimal = Decimal(str(order_size))

        with PerformanceTimer() as timer:
            buy_vwap = orderbook.get_volume_weighted_price(order_size_decimal, "BUY")
            sell_vwap = orderbook.get_volume_weighted_price(order_size_decimal, "SELL")

        # VWAP calculation should complete quickly
        assert (
            timer.duration < 0.05
        ), f"VWAP calculation took too long: {timer.duration}s"

        # If results are not None, they should be positive
        if buy_vwap is not None:
            assert buy_vwap > 0
        if sell_vwap is not None:
            assert sell_vwap > 0

    @given(large_orderbook_strategy(min_levels=100, max_levels=1000))
    @settings(max_examples=10, deadline=8000)
    def test_orderbook_memory_usage_properties(self, orderbook: OrderBook):
        """Property: Memory usage should scale reasonably with orderbook size."""
        num_levels = len(orderbook.bids) + len(orderbook.asks)
        memory_size = get_object_size(orderbook)

        # Memory usage should be reasonable (rough heuristic)
        # Each level has price + size (2 Decimals) plus tuple overhead
        # Estimate ~200 bytes per level as reasonable upper bound
        max_expected_size = num_levels * 200

        assert (
            memory_size < max_expected_size
        ), f"Memory usage too high: {memory_size} bytes for {num_levels} levels"

        # Memory should scale roughly linearly
        bytes_per_level = memory_size / num_levels if num_levels > 0 else 0
        assert (
            bytes_per_level < 500
        ), f"Memory per level too high: {bytes_per_level} bytes"


class TestTradeProcessingPerformanceProperties:
    """Property-based tests for trade processing performance."""

    @given(trade_sequence_strategy(min_trades=100, max_trades=1000))
    @settings(max_examples=10, deadline=5000)
    def test_trade_sequence_processing_performance(self, trades: list[Trade]):
        """Property: Processing trade sequences should scale linearly."""
        num_trades = len(trades)

        with PerformanceTimer() as timer:
            # Simulate common trade processing operations
            total_volume = sum(trade.size for trade in trades)
            total_value = sum(trade.value for trade in trades)
            buy_trades = [t for t in trades if t.is_buy()]
            sell_trades = [t for t in trades if t.is_sell()]

        # Processing should complete quickly
        assert (
            timer.duration < 0.5
        ), f"Trade processing took too long: {timer.duration}s for {num_trades} trades"

        # Results should be meaningful
        assert total_volume > 0
        assert total_value > 0
        assert len(buy_trades) + len(sell_trades) == num_trades

    @given(trade_sequence_strategy(min_trades=500, max_trades=5000))
    @settings(max_examples=5, deadline=10000)
    def test_vwap_calculation_from_trades_performance(self, trades: list[Trade]):
        """Property: VWAP calculation from large trade sequences should be efficient."""
        if not trades:
            return

        with PerformanceTimer() as timer:
            # Calculate VWAP manually
            total_value = sum(trade.price * trade.size for trade in trades)
            total_volume = sum(trade.size for trade in trades)
            vwap = total_value / total_volume if total_volume > 0 else Decimal(0)

        num_trades = len(trades)
        assert (
            timer.duration < 1.0
        ), f"VWAP calculation took too long: {timer.duration}s for {num_trades} trades"

        # VWAP should be within reasonable range of trade prices
        if trades:
            prices = [trade.price for trade in trades]
            min_price = min(prices)
            max_price = max(prices)
            assert (
                min_price <= vwap <= max_price
            ), f"VWAP {vwap} should be within price range [{min_price}, {max_price}]"


class TestCandleProcessingPerformanceProperties:
    """Property-based tests for candle processing performance."""

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=1.0, max_value=1000.0),  # open
                st.floats(min_value=1.0, max_value=1000.0),  # high
                st.floats(min_value=1.0, max_value=1000.0),  # low
                st.floats(min_value=1.0, max_value=1000.0),  # close
                st.floats(min_value=0.1, max_value=10000.0),  # volume
            ),
            min_size=100,
            max_size=10000,
        )
    )
    @settings(max_examples=5, deadline=10000)
    def test_candle_sequence_processing_performance(
        self, candle_data: list[tuple[float, float, float, float, float]]
    ):
        """Property: Processing large candle sequences should be efficient."""
        num_candles = len(candle_data)

        with PerformanceTimer() as timer:
            candles = []
            base_time = datetime.now(UTC)

            for i, (o, h, l, c, v) in enumerate(candle_data):
                # Ensure OHLC relationships are valid
                high = max(o, h, l, c)
                low = min(o, h, l, c)

                candle = Candle(
                    timestamp=base_time,
                    open=Decimal(str(o)),
                    high=Decimal(str(high)),
                    low=Decimal(str(low)),
                    close=Decimal(str(c)),
                    volume=Decimal(str(v)),
                )
                candles.append(candle)

        assert (
            timer.duration < 2.0
        ), f"Candle creation took too long: {timer.duration}s for {num_candles} candles"
        assert len(candles) == num_candles

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=1.0, max_value=1000.0),  # price
                st.floats(min_value=0.1, max_value=1000.0),  # volume
            ),
            min_size=1000,
            max_size=10000,
        )
    )
    @settings(max_examples=3, deadline=15000)
    def test_market_data_aggregation_performance(
        self, price_volume_data: list[tuple[float, float]]
    ):
        """Property: Aggregating large amounts of market data should be efficient."""
        num_points = len(price_volume_data)

        with PerformanceTimer() as timer:
            market_data_points = []
            base_time = datetime.now(UTC)

            for i, (price, volume) in enumerate(price_volume_data):
                market_data = MarketData(
                    symbol="BTC-USD",
                    price=Decimal(str(price)),
                    volume=Decimal(str(volume)),
                    timestamp=base_time,
                )
                market_data_points.append(market_data)

            # Perform some aggregation operations
            total_volume = sum(md.volume for md in market_data_points)
            avg_price = sum(md.price for md in market_data_points) / len(
                market_data_points
            )
            vwap = (
                sum(md.price * md.volume for md in market_data_points) / total_volume
                if total_volume > 0
                else Decimal(0)
            )

        assert (
            timer.duration < 3.0
        ), f"Market data aggregation took too long: {timer.duration}s for {num_points} points"

        # Results should be meaningful
        assert total_volume > 0
        assert avg_price > 0
        assert vwap > 0


class TestMemoryUsageProperties:
    """Property-based tests for memory usage characteristics."""

    @given(st.integers(min_value=10, max_value=1000))
    @settings(max_examples=20, deadline=3000)
    def test_orderbook_memory_scaling(self, num_levels: int):
        """Property: Memory usage should scale predictably with orderbook size."""
        # Create orderbooks of different sizes
        small_orderbook = self._create_test_orderbook(num_levels // 10)
        medium_orderbook = self._create_test_orderbook(num_levels // 2)
        large_orderbook = self._create_test_orderbook(num_levels)

        small_size = get_object_size(small_orderbook)
        medium_size = get_object_size(medium_orderbook)
        large_size = get_object_size(large_orderbook)

        # Memory should increase with size (but not necessarily linearly due to overhead)
        assert small_size <= medium_size <= large_size

        # Memory per level should be reasonable
        large_memory_per_level = large_size / (num_levels * 2)  # bids + asks
        assert (
            large_memory_per_level < 1000
        ), f"Memory per level too high: {large_memory_per_level} bytes"

    def _create_test_orderbook(self, num_levels: int) -> OrderBook:
        """Helper to create test orderbook with specified number of levels."""
        base_price = Decimal("100.0")

        bids = [
            (base_price - Decimal(str(i * 0.01)), Decimal("10.0"))
            for i in range(1, num_levels + 1)
        ]
        asks = [
            (base_price + Decimal(str(i * 0.01)), Decimal("10.0"))
            for i in range(1, num_levels + 1)
        ]

        return OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))

    @given(st.integers(min_value=100, max_value=10000))
    @settings(max_examples=10, deadline=5000)
    def test_trade_sequence_memory_usage(self, num_trades: int):
        """Property: Memory usage for trade sequences should be reasonable."""
        trades = []
        base_time = datetime.now(UTC)

        # Create trade sequence
        for i in range(num_trades):
            trade = Trade(
                id=f"trade_{i}",
                timestamp=base_time,
                price=Decimal("100.0"),
                size=Decimal("1.0"),
                side="BUY",
            )
            trades.append(trade)

        total_memory = get_object_size(trades)
        memory_per_trade = total_memory / num_trades

        # Memory per trade should be reasonable (rough estimate)
        assert (
            memory_per_trade < 2000
        ), f"Memory per trade too high: {memory_per_trade} bytes"


class TestAlgorithmicComplexityProperties:
    """Property-based tests for algorithmic complexity characteristics."""

    @given(
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20, deadline=5000)
    def test_orderbook_operation_complexity(self, bid_levels: int, ask_levels: int):
        """Property: Core orderbook operations should have predictable complexity."""
        orderbook = self._create_test_orderbook(max(bid_levels, ask_levels))

        # Test O(1) operations
        with PerformanceTimer() as timer:
            for _ in range(1000):  # Repeat to get measurable time
                _ = orderbook.best_bid
                _ = orderbook.best_ask
                _ = orderbook.mid_price
                _ = orderbook.spread

        # These operations should be very fast
        assert timer.duration < 0.1, f"O(1) operations took too long: {timer.duration}s"

        # Test O(n) operations
        with PerformanceTimer() as timer:
            for _ in range(10):  # Fewer repetitions for O(n) operations
                _ = orderbook.bid_depth
                _ = orderbook.ask_depth

        # These should still be reasonable
        total_levels = bid_levels + ask_levels
        assert (
            timer.duration < 0.1
        ), f"O(n) operations took too long: {timer.duration}s for {total_levels} levels"

    @given(
        large_orderbook_strategy(min_levels=50, max_levels=200),
        st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=10, max_size=50),
    )
    @settings(max_examples=10, deadline=5000)
    def test_multiple_vwap_calculation_complexity(
        self, orderbook: OrderBook, order_sizes: list[float]
    ):
        """Property: Multiple VWAP calculations should have predictable complexity."""
        num_calculations = len(order_sizes)

        with PerformanceTimer() as timer:
            for size in order_sizes:
                size_decimal = Decimal(str(size))
                _ = orderbook.get_volume_weighted_price(size_decimal, "BUY")
                _ = orderbook.get_volume_weighted_price(size_decimal, "SELL")

        # Time should scale reasonably with number of calculations
        time_per_calculation = (
            timer.duration / (num_calculations * 2) if num_calculations > 0 else 0
        )
        assert (
            time_per_calculation < 0.01
        ), f"VWAP calculation too slow: {time_per_calculation}s per calculation"


class TestConcurrencyProperties:
    """Property-based tests for concurrent access patterns (simulation)."""

    @given(large_orderbook_strategy(min_levels=100, max_levels=500))
    @settings(max_examples=5, deadline=5000)
    def test_concurrent_read_performance_simulation(self, orderbook: OrderBook):
        """Property: Simulated concurrent reads should not degrade performance significantly."""
        num_operations = 1000

        # Simulate rapid read operations
        with PerformanceTimer() as timer:
            for i in range(num_operations):
                # Simulate different threads reading different properties
                if i % 4 == 0:
                    _ = orderbook.best_bid
                elif i % 4 == 1:
                    _ = orderbook.best_ask
                elif i % 4 == 2:
                    _ = orderbook.mid_price
                else:
                    _ = orderbook.spread

        # Operations should remain fast even with many reads
        time_per_operation = timer.duration / num_operations
        assert (
            time_per_operation < 0.0001
        ), f"Read operations too slow under load: {time_per_operation}s per operation"


if __name__ == "__main__":
    # Run performance property tests
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
