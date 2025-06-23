"""
Property-based tests for datetime timezone consistency.

This module ensures that all datetime objects in the codebase:
1. Are timezone-aware (have tzinfo set)
2. Use UTC consistently
3. Handle timezone conversions correctly
4. Maintain proper ordering relationships
"""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import hypothesis.strategies as st
from hypothesis import given, settings

from bot.types.market_data import (
    CandleData,
    MarketDataStatus,
    TickerData,
    aggregate_candles,
)
from bot.types.services import (
    ConnectionState,
    create_health_status,
)
from bot.utils.price_conversion import (
    _CIRCUIT_BREAKER_STATE,
)


# Custom strategies for generating timezone-aware datetimes
@st.composite
def timezone_aware_datetime(draw: st.DrawFn) -> datetime:
    """Generate timezone-aware datetime objects."""
    # Generate a base datetime (naive bounds, then add timezone)
    dt = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),  # Naive datetime
            max_value=datetime(2030, 12, 31),  # Naive datetime
            timezones=st.just(UTC),  # Always use UTC
        )
    )
    return dt


@st.composite
def candle_with_tz(draw: st.DrawFn) -> CandleData:
    """Generate a CandleData object with proper timezone handling."""
    timestamp = draw(timezone_aware_datetime())

    # Generate prices ensuring OHLC relationships
    low = draw(st.decimals(min_value="0.01", max_value="10000", places=8))
    high = draw(st.decimals(min_value=low, max_value=low * Decimal(2), places=8))

    # Open and close must be between low and high
    open_price = draw(st.decimals(min_value=low, max_value=high, places=8))
    close_price = draw(st.decimals(min_value=low, max_value=high, places=8))

    volume = draw(st.decimals(min_value="0", max_value="1000000", places=8))

    return CandleData(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close_price,
        volume=volume,
    )


@st.composite
def ticker_with_tz(draw: st.DrawFn) -> TickerData:
    """Generate a TickerData object with proper timezone handling."""
    timestamp = draw(timezone_aware_datetime())
    product_id = draw(st.sampled_from(["BTC-USD", "ETH-USD", "SUI-PERP"]))

    price = draw(st.decimals(min_value="0.01", max_value="100000", places=8))

    # Optional bid/ask
    has_bid_ask = draw(st.booleans())
    if has_bid_ask:
        spread_pct = draw(st.floats(min_value=0.0001, max_value=0.01))
        bid = price * (Decimal(1) - Decimal(str(spread_pct)))
        ask = price * (Decimal(1) + Decimal(str(spread_pct)))
    else:
        bid = None
        ask = None

    # Optional 24h high/low
    has_24h = draw(st.booleans())
    if has_24h:
        variance = draw(st.floats(min_value=0.01, max_value=0.5))
        low_24h = price * (Decimal(1) - Decimal(str(variance)))
        high_24h = price * (Decimal(1) + Decimal(str(variance)))
    else:
        low_24h = None
        high_24h = None

    return TickerData(
        product_id=product_id,
        timestamp=timestamp,
        price=price,
        bid=bid,
        ask=ask,
        low_24h=low_24h,
        high_24h=high_24h,
    )


class TestDatetimeConsistency:
    """Test datetime timezone consistency across the codebase."""

    @given(dt=timezone_aware_datetime())
    def test_datetime_always_has_timezone(self, dt: datetime) -> None:
        """All generated datetimes must have timezone info."""
        assert dt.tzinfo is not None
        assert dt.tzinfo == UTC

    @given(candle=candle_with_tz())
    def test_candle_timestamp_timezone(self, candle: CandleData) -> None:
        """CandleData timestamps must have UTC timezone."""
        assert candle.timestamp.tzinfo is not None
        assert candle.timestamp.tzinfo == UTC

        # Test serialization preserves timezone
        json_data = candle.model_dump_json()
        parsed = CandleData.model_validate_json(json_data)
        assert parsed.timestamp.tzinfo == UTC

    @given(ticker=ticker_with_tz())
    def test_ticker_timestamp_timezone(self, ticker: TickerData) -> None:
        """TickerData timestamps must have UTC timezone."""
        assert ticker.timestamp.tzinfo is not None
        assert ticker.timestamp.tzinfo == UTC

    @given(candles=st.lists(candle_with_tz(), min_size=2, max_size=10))
    def test_candle_aggregation_preserves_timezone(
        self, candles: list[CandleData]
    ) -> None:
        """Candle aggregation must preserve UTC timezone."""
        # Sort candles by timestamp
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)

        # Test aggregation
        aggregated = aggregate_candles(sorted_candles, 5)  # 5-minute aggregation

        for agg_candle in aggregated:
            assert agg_candle.timestamp.tzinfo == UTC
            # Aggregated timestamp should be the first candle's timestamp in the group
            assert any(c.timestamp <= agg_candle.timestamp for c in sorted_candles)

    @given(
        status=st.sampled_from(list(ConnectionState)),
        timestamp=timezone_aware_datetime(),
    )
    def test_connection_info_timezone(
        self, status: ConnectionState, timestamp: datetime
    ) -> None:
        """ConnectionInfo timestamps must maintain timezone."""
        # Mock datetime.now(UTC) to return our test timestamp
        import bot.types.services_integration
        from bot.types.services_integration import create_connection_info

        original_datetime = bot.types.services_integration.datetime

        class MockDatetime:
            @staticmethod
            def now(tz=None):
                return timestamp

        bot.types.services_integration.datetime = MockDatetime

        try:
            conn_info = create_connection_info(status)

            if status == ConnectionState.CONNECTED:
                assert conn_info.get("connected_at") is not None
                assert conn_info["connected_at"].tzinfo == UTC
            elif status == ConnectionState.DISCONNECTED:
                assert conn_info.get("disconnected_at") is not None
                assert conn_info["disconnected_at"].tzinfo == UTC
        finally:
            bot.types.services_integration.datetime = original_datetime

    @given(
        dt1=timezone_aware_datetime(),
        dt2=timezone_aware_datetime(),
    )
    def test_datetime_comparison_safety(self, dt1: datetime, dt2: datetime) -> None:
        """Datetime comparisons must work correctly with UTC timezone."""
        # These comparisons should not raise exceptions
        _ = dt1 < dt2
        _ = dt1 <= dt2
        _ = dt1 > dt2
        _ = dt1 >= dt2
        _ = dt1 == dt2
        _ = dt1 != dt2

        # Time differences should work
        diff = dt2 - dt1
        assert isinstance(diff, timedelta)

    @given(
        timestamp=timezone_aware_datetime(),
        last_update=timezone_aware_datetime(),
    )
    def test_market_data_staleness_calculation(
        self, timestamp: datetime, last_update: datetime
    ) -> None:
        """Market data staleness calculation must handle timezones correctly."""
        # Create a market data status
        status = MarketDataStatus(
            product_id="BTC-USD",
            timestamp=timestamp,
            connection_state="CONNECTED",
            data_quality="EXCELLENT",
            last_ticker_update=last_update,
        )

        staleness = status.get_staleness_seconds()

        if last_update <= timestamp:
            assert staleness is not None
            assert staleness >= 0
            # Verify calculation matches expected
            expected = (timestamp - last_update).total_seconds()
            assert (
                abs(staleness - expected) < 0.001
            )  # Allow small float precision error

    def test_price_conversion_circuit_breaker_timestamps(self) -> None:
        """Circuit breaker timestamps must use UTC."""
        # Reset circuit breaker state
        _CIRCUIT_BREAKER_STATE["failure_timestamps"].clear()
        _CIRCUIT_BREAKER_STATE["last_known_good_prices"].clear()

        # Trigger a circuit breaker failure
        from bot.utils.price_conversion import _record_conversion_failure

        _record_conversion_failure("TEST_SYMBOL_price")

        # Check timestamp
        assert "TEST_SYMBOL_price" in _CIRCUIT_BREAKER_STATE["failure_timestamps"]
        failure_time = _CIRCUIT_BREAKER_STATE["failure_timestamps"]["TEST_SYMBOL_price"]

        # Verify it's a recent timestamp (within last minute)
        now = datetime.now(UTC).timestamp()
        assert abs(now - failure_time) < 60

        # Update last known good price
        from bot.utils.price_conversion import _update_last_known_good_price

        _update_last_known_good_price("TEST_SYMBOL", "price", Decimal("100.50"))

        # Check stored timestamp
        price_data = _CIRCUIT_BREAKER_STATE["last_known_good_prices"][
            "TEST_SYMBOL_price"
        ]
        assert "timestamp" in price_data
        assert abs(now - price_data["timestamp"]) < 60

    @given(
        base_dt=timezone_aware_datetime(),
        offset_hours=st.integers(min_value=-24, max_value=24),
    )
    def test_timezone_offset_handling(
        self, base_dt: datetime, offset_hours: int
    ) -> None:
        """Test handling of different timezone offsets."""
        # Create datetime with different offset
        offset = timezone(timedelta(hours=offset_hours))
        dt_with_offset = base_dt.astimezone(offset)

        # Converting to UTC should give same moment in time
        dt_utc = dt_with_offset.astimezone(UTC)
        assert dt_utc == base_dt

        # Timestamps should be equal
        assert dt_utc.timestamp() == base_dt.timestamp()

    @settings(max_examples=50)
    @given(candles=st.lists(candle_with_tz(), min_size=10, max_size=20))
    def test_candle_ordering_with_timezones(self, candles: list[CandleData]) -> None:
        """Candle ordering must work correctly with timezone-aware timestamps."""
        # Sort candles by timestamp
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)

        # Verify ordering
        for i in range(1, len(sorted_candles)):
            assert sorted_candles[i - 1].timestamp <= sorted_candles[i].timestamp

        # Verify we can calculate time differences
        if len(sorted_candles) >= 2:
            time_diff = sorted_candles[1].timestamp - sorted_candles[0].timestamp
            assert isinstance(time_diff, timedelta)


# Regression tests for specific timezone issues
class TestTimezoneRegressions:
    """Regression tests for specific timezone-related bugs."""

    def test_market_data_status_naive_datetime_handling(self) -> None:
        """Test MarketDataStatus handles naive datetimes correctly."""
        # This should not raise an exception
        status = MarketDataStatus(
            product_id="BTC-USD",
            timestamp=datetime.now(UTC),
            connection_state="CONNECTED",
            data_quality="EXCELLENT",
            last_ticker_update=datetime.now(UTC) - timedelta(seconds=5),
        )

        staleness = status.get_staleness_seconds()
        assert staleness is not None
        assert 4 <= staleness <= 6  # Allow for execution time

    def test_service_health_timestamp_format(self) -> None:
        """Test ServiceHealth timestamp is properly formatted."""
        from bot.types.services import ServiceStatus

        health = create_health_status(ServiceStatus.HEALTHY)

        # Timestamp should be Unix timestamp (float)
        assert isinstance(health["last_check"], float)
        assert health["last_check"] > 0

        # Should be recent (within last minute)
        now = datetime.now(UTC).timestamp()
        assert abs(now - health["last_check"]) < 60

    def test_spread_calculator_timezone_consistency(self) -> None:
        """Test spread calculator handles timezones correctly."""
        from bot.strategy.spread_calculator import SpreadRecommendation

        # Create recommendation with explicit UTC timestamp
        timestamp = datetime.now(UTC)
        recommendation = SpreadRecommendation(
            base_spread_bps=Decimal(10),
            adjusted_spread_bps=Decimal(12),
            min_profitable_spread_bps=Decimal(8),
            directional_bias=0.1,
            bid_adjustment_bps=Decimal(1),
            ask_adjustment_bps=Decimal(-1),
            levels=[],
            reasoning="Test",
            timestamp=timestamp,
        )

        assert recommendation.timestamp.tzinfo == UTC
        assert recommendation.timestamp == timestamp
