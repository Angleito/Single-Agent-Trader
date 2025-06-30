"""
Property-based tests for orderbook data validation using hypothesis.

This module provides comprehensive property testing for orderbook invariants,
mathematical properties, and edge cases using hypothesis-generated test data.
"""

from datetime import UTC, datetime
from decimal import Decimal

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from bot.fp.types.market import Candle, OrderBook
from bot.fp.types.result import Failure, Success


# Custom strategies for generating realistic market data
@st.composite
def price_strategy(draw, min_price=0.01, max_price=100000.0):
    """Generate realistic price values as Decimal."""
    price = draw(
        st.floats(
            min_value=min_price,
            max_value=max_price,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return Decimal(str(round(price, 8)))


@st.composite
def volume_strategy(draw, min_volume=0.0, max_volume=1000000.0):
    """Generate realistic volume values as Decimal."""
    volume = draw(
        st.floats(
            min_value=min_volume,
            max_value=max_volume,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return Decimal(str(round(volume, 8)))


@st.composite
def price_size_tuple_strategy(
    draw, min_price=0.01, max_price=100000.0, min_size=0.001, max_size=10000.0
):
    """Generate (price, size) tuples for orderbook entries."""
    price = draw(price_strategy(min_price=min_price, max_price=max_price))
    size = draw(volume_strategy(min_volume=min_size, max_volume=max_size))
    return (price, size)


@st.composite
def sorted_bids_strategy(draw, min_entries=1, max_entries=20):
    """Generate sorted bid entries (descending order)."""
    num_entries = draw(st.integers(min_value=min_entries, max_value=max_entries))

    # Generate prices in descending order
    max_price = draw(price_strategy(min_price=50.0, max_price=10000.0))
    prices = []
    current_price = max_price

    for _ in range(num_entries):
        # Ensure each price is strictly less than the previous
        next_price = draw(
            price_strategy(min_price=0.01, max_price=float(current_price) - 0.01)
        )
        prices.append(next_price)
        current_price = next_price

    # Generate sizes for each price
    bids = []
    for price in prices:
        size = draw(volume_strategy(min_volume=0.001, max_volume=1000.0))
        bids.append((price, size))

    return bids


@st.composite
def sorted_asks_strategy(draw, min_entries=1, max_entries=20, min_price=None):
    """Generate sorted ask entries (ascending order)."""
    num_entries = draw(st.integers(min_value=min_entries, max_value=max_entries))

    # Start from a minimum price (usually above the best bid)
    base_price = (
        min_price
        if min_price
        else draw(price_strategy(min_price=0.02, max_price=100.0))
    )
    current_price = base_price

    asks = []
    for _ in range(num_entries):
        # Generate size first
        size = draw(volume_strategy(min_volume=0.001, max_volume=1000.0))
        asks.append((current_price, size))

        # Next price must be higher
        current_price = draw(
            price_strategy(
                min_price=float(current_price) + 0.01,
                max_price=float(current_price) + 10.0,
            )
        )

    return asks


@st.composite
def valid_orderbook_strategy(draw):
    """Generate valid orderbook with proper bid/ask ordering and spread."""
    # Generate bids (descending order)
    bids = draw(sorted_bids_strategy(min_entries=1, max_entries=10))

    # Generate asks starting above the best bid
    best_bid_price = bids[0][0] if bids else Decimal("50.0")
    min_ask_price = best_bid_price + Decimal("0.01")  # Ensure positive spread

    asks = draw(
        sorted_asks_strategy(min_entries=1, max_entries=10, min_price=min_ask_price)
    )

    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        )
    )

    return OrderBook(bids=bids, asks=asks, timestamp=timestamp)


@st.composite
def candle_strategy(draw):
    """Generate valid OHLCV candle data."""
    # Generate base price range
    base_price = draw(price_strategy(min_price=1.0, max_price=1000.0))

    # Generate OHLC ensuring proper relationships
    open_price = base_price

    # High must be >= max(open, close)
    # Low must be <= min(open, close)
    price_range = draw(st.floats(min_value=0.0, max_value=float(base_price) * 0.1))

    high = open_price + Decimal(str(price_range))
    low = open_price - Decimal(str(price_range))

    # Close can be anywhere between low and high
    close = draw(price_strategy(min_price=float(low), max_price=float(high)))

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = max(high, open_price, close)
    low = min(low, open_price, close)

    volume = draw(volume_strategy(min_volume=0.001, max_volume=10000.0))
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        )
    )

    return Candle(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestOrderBookProperties:
    """Property-based tests for OrderBook validation and invariants."""

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_orderbook_creation_succeeds_with_valid_data(self, orderbook: OrderBook):
        """Property: Valid orderbook data should always create successful OrderBook instances."""
        # Should not raise any exceptions
        assert isinstance(orderbook, OrderBook)
        assert orderbook.bids or orderbook.asks  # At least one side should have entries

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_bid_prices_descending_invariant(self, orderbook: OrderBook):
        """Property: Bid prices must always be in strictly descending order."""
        if len(orderbook.bids) <= 1:
            return  # Trivially true for 0-1 entries

        for i in range(len(orderbook.bids) - 1):
            current_price = orderbook.bids[i][0]
            next_price = orderbook.bids[i + 1][0]
            assert (
                current_price > next_price
            ), f"Bid price {current_price} should be > {next_price} at position {i}"

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_ask_prices_ascending_invariant(self, orderbook: OrderBook):
        """Property: Ask prices must always be in strictly ascending order."""
        if len(orderbook.asks) <= 1:
            return  # Trivially true for 0-1 entries

        for i in range(len(orderbook.asks) - 1):
            current_price = orderbook.asks[i][0]
            next_price = orderbook.asks[i + 1][0]
            assert (
                current_price < next_price
            ), f"Ask price {current_price} should be < {next_price} at position {i}"

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_positive_spread_invariant(self, orderbook: OrderBook):
        """Property: Spread between best bid and ask must always be positive."""
        if not orderbook.bids or not orderbook.asks:
            return  # No spread without both sides

        best_bid = orderbook.best_bid[0]
        best_ask = orderbook.best_ask[0]
        spread = best_ask - best_bid

        assert (
            spread > 0
        ), f"Spread {spread} must be positive (bid: {best_bid}, ask: {best_ask})"
        assert (
            orderbook.spread == spread
        ), "OrderBook.spread property should match calculated spread"

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_mid_price_calculation_property(self, orderbook: OrderBook):
        """Property: Mid price should be the average of best bid and ask."""
        if not orderbook.bids or not orderbook.asks:
            assert orderbook.mid_price is None
            return

        best_bid_price = orderbook.best_bid[0]
        best_ask_price = orderbook.best_ask[0]
        expected_mid = (best_bid_price + best_ask_price) / 2

        assert orderbook.mid_price == expected_mid
        assert orderbook.mid_price > best_bid_price
        assert orderbook.mid_price < best_ask_price

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_depth_calculation_properties(self, orderbook: OrderBook):
        """Property: Depth calculations should sum all sizes correctly."""
        expected_bid_depth = sum(size for _, size in orderbook.bids)
        expected_ask_depth = sum(size for _, size in orderbook.asks)

        assert orderbook.bid_depth == expected_bid_depth
        assert orderbook.ask_depth == expected_ask_depth
        assert orderbook.bid_depth >= 0
        assert orderbook.ask_depth >= 0

    @given(valid_orderbook_strategy(), st.floats(min_value=0.001, max_value=1000.0))
    @settings(max_examples=50, deadline=3000)
    def test_volume_weighted_price_properties(
        self, orderbook: OrderBook, order_size: float
    ):
        """Property: VWAP should be within the price range of utilized orders."""
        order_size_decimal = Decimal(str(order_size))

        # Test buy side VWAP
        buy_vwap = orderbook.get_volume_weighted_price(order_size_decimal, "BUY")
        if buy_vwap is not None:
            # VWAP should be >= lowest ask price used
            min_ask_price = orderbook.asks[0][0] if orderbook.asks else None
            if min_ask_price:
                assert (
                    buy_vwap >= min_ask_price
                ), f"Buy VWAP {buy_vwap} should be >= min ask {min_ask_price}"

        # Test sell side VWAP
        sell_vwap = orderbook.get_volume_weighted_price(order_size_decimal, "SELL")
        if sell_vwap is not None:
            # VWAP should be <= highest bid price used
            max_bid_price = orderbook.bids[0][0] if orderbook.bids else None
            if max_bid_price:
                assert (
                    sell_vwap <= max_bid_price
                ), f"Sell VWAP {sell_vwap} should be <= max bid {max_bid_price}"

    @given(valid_orderbook_strategy(), st.floats(min_value=0.001, max_value=10000.0))
    @settings(max_examples=50, deadline=3000)
    def test_price_impact_bounded_property(
        self, orderbook: OrderBook, order_size: float
    ):
        """Property: Price impact should be bounded by available liquidity."""
        order_size_decimal = Decimal(str(order_size))

        # Test buy side
        buy_impact = orderbook.price_impact("buy", order_size_decimal)
        if buy_impact is not None and orderbook.asks:
            min_ask = orderbook.asks[0][0]
            max_ask = orderbook.asks[-1][0]
            assert (
                min_ask <= buy_impact <= max_ask
            ), f"Buy impact {buy_impact} should be between {min_ask} and {max_ask}"

        # Test sell side
        sell_impact = orderbook.price_impact("sell", order_size_decimal)
        if sell_impact is not None and orderbook.bids:
            min_bid = orderbook.bids[-1][0]
            max_bid = orderbook.bids[0][0]
            assert (
                min_bid <= sell_impact <= max_bid
            ), f"Sell impact {sell_impact} should be between {min_bid} and {max_bid}"

    @given(valid_orderbook_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_spread_basis_points_calculation(self, orderbook: OrderBook):
        """Property: Spread in basis points should be consistent with absolute spread."""
        if not orderbook.bids or not orderbook.asks:
            assert orderbook.get_spread_bps() == 0
            return

        spread_bps = orderbook.get_spread_bps()
        absolute_spread = orderbook.spread
        mid_price = orderbook.mid_price

        if mid_price and mid_price > 0:
            expected_bps = (absolute_spread / mid_price) * 10000
            assert abs(spread_bps - expected_bps) < Decimal(
                "0.01"
            ), f"BPS calculation mismatch: {spread_bps} vs {expected_bps}"
            assert spread_bps >= 0, "Spread BPS should be non-negative"


class TestCandleProperties:
    """Property-based tests for OHLCV candle data validation."""

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_candle_creation_succeeds_with_valid_data(self, candle: Candle):
        """Property: Valid candle data should always create successful Candle instances."""
        assert isinstance(candle, Candle)
        assert candle.open > 0
        assert candle.high > 0
        assert candle.low > 0
        assert candle.close > 0
        assert candle.volume >= 0

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_ohlc_relationships_invariant(self, candle: Candle):
        """Property: OHLC relationships must always be valid."""
        # High must be the highest price
        assert (
            candle.high >= candle.open
        ), f"High {candle.high} should be >= open {candle.open}"
        assert (
            candle.high >= candle.close
        ), f"High {candle.high} should be >= close {candle.close}"
        assert (
            candle.high >= candle.low
        ), f"High {candle.high} should be >= low {candle.low}"

        # Low must be the lowest price
        assert (
            candle.low <= candle.open
        ), f"Low {candle.low} should be <= open {candle.open}"
        assert (
            candle.low <= candle.close
        ), f"Low {candle.low} should be <= close {candle.close}"
        assert (
            candle.low <= candle.high
        ), f"Low {candle.low} should be <= high {candle.high}"

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_price_range_calculation_property(self, candle: Candle):
        """Property: Price range should equal high minus low."""
        expected_range = candle.high - candle.low
        assert candle.price_range == expected_range
        assert candle.price_range >= 0, "Price range should be non-negative"

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_bullish_bearish_classification_property(self, candle: Candle):
        """Property: Candle classification should be mutually exclusive and exhaustive."""
        is_bullish = candle.is_bullish
        is_bearish = candle.is_bearish

        if candle.close > candle.open:
            assert is_bullish and not is_bearish, "Close > Open should be bullish"
        elif candle.close < candle.open:
            assert is_bearish and not is_bullish, "Close < Open should be bearish"
        else:  # close == open
            assert (
                not is_bullish and not is_bearish
            ), "Close == Open should be neither bullish nor bearish"

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_shadow_calculations_property(self, candle: Candle):
        """Property: Shadow calculations should be consistent with OHLC values."""
        upper_shadow = candle.upper_shadow
        lower_shadow = candle.lower_shadow

        # Upper shadow: high - max(open, close)
        expected_upper = candle.high - max(candle.open, candle.close)
        assert (
            upper_shadow == expected_upper
        ), f"Upper shadow mismatch: {upper_shadow} vs {expected_upper}"

        # Lower shadow: min(open, close) - low
        expected_lower = min(candle.open, candle.close) - candle.low
        assert (
            lower_shadow == expected_lower
        ), f"Lower shadow mismatch: {lower_shadow} vs {expected_lower}"

        # Shadows should be non-negative
        assert upper_shadow >= 0, "Upper shadow should be non-negative"
        assert lower_shadow >= 0, "Lower shadow should be non-negative"

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_body_size_calculation_property(self, candle: Candle):
        """Property: Body size should be absolute difference between open and close."""
        expected_body_size = abs(candle.close - candle.open)
        assert candle.body_size == expected_body_size
        assert candle.body_size >= 0, "Body size should be non-negative"

    @given(candle_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_vwap_calculation_property(self, candle: Candle):
        """Property: VWAP should be the typical price (HLC/3)."""
        expected_vwap = (candle.high + candle.low + candle.close) / 3
        assert candle.vwap() == expected_vwap

        # VWAP should be within the price range
        assert (
            candle.low <= candle.vwap() <= candle.high
        ), "VWAP should be within price range"


class TestMarketDataFuzzing:
    """Fuzzing tests for edge cases and boundary conditions."""

    @given(
        st.lists(price_size_tuple_strategy(), min_size=0, max_size=0),  # Empty bids
        st.lists(price_size_tuple_strategy(), min_size=0, max_size=0),  # Empty asks
    )
    @settings(max_examples=10, deadline=1000)
    def test_empty_orderbook_handling(
        self, bids: list[tuple[Decimal, Decimal]], asks: list[tuple[Decimal, Decimal]]
    ):
        """Fuzz test: Empty orderbook should raise appropriate errors."""
        timestamp = datetime.now(UTC)

        with pytest.raises(ValueError, match="Order book cannot be empty"):
            OrderBook(bids=bids, asks=asks, timestamp=timestamp)

    @given(
        st.floats(min_value=-1000.0, max_value=0.0),  # Non-positive prices
        st.floats(min_value=-1000.0, max_value=1000.0),
        st.floats(min_value=-1000.0, max_value=1000.0),
        st.floats(min_value=-1000.0, max_value=1000.0),
        st.floats(min_value=-1000.0, max_value=1000.0),
    )
    @settings(max_examples=20, deadline=1000)
    def test_invalid_candle_prices_fuzzing(
        self,
        open_val: float,
        high_val: float,
        low_val: float,
        close_val: float,
        volume_val: float,
    ):
        """Fuzz test: Invalid price values should raise appropriate errors."""
        timestamp = datetime.now(UTC)
        volume = Decimal(str(abs(volume_val)))  # Ensure volume is non-negative

        # Test cases where any price is non-positive
        if any(price <= 0 for price in [open_val, high_val, low_val, close_val]):
            with pytest.raises(ValueError, match="All prices must be positive"):
                Candle(
                    timestamp=timestamp,
                    open=Decimal(str(open_val)),
                    high=Decimal(str(high_val)),
                    low=Decimal(str(low_val)),
                    close=Decimal(str(close_val)),
                    volume=volume,
                )

    @given(
        price_strategy(min_price=0.01, max_price=100.0),
        st.floats(min_value=-1000.0, max_value=-0.01),  # Negative volume
    )
    @settings(max_examples=20, deadline=1000)
    def test_negative_volume_fuzzing(self, price: Decimal, volume_val: float):
        """Fuzz test: Negative volume should raise appropriate errors."""
        timestamp = datetime.now(UTC)
        volume = Decimal(str(volume_val))

        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Candle(
                timestamp=timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
            )

    @given(
        st.floats(
            min_value=0.01, max_value=1000000.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_extreme_order_sizes_property(self, order_size: float):
        """Property test: Extreme order sizes should be handled gracefully."""
        # Create a simple orderbook
        timestamp = datetime.now(UTC)
        bids = [(Decimal("100.0"), Decimal("10.0"))]
        asks = [(Decimal("101.0"), Decimal("10.0"))]

        orderbook = OrderBook(bids=bids, asks=asks, timestamp=timestamp)
        order_size_decimal = Decimal(str(order_size))

        # Test that extreme sizes either return None (insufficient liquidity) or valid prices
        buy_vwap = orderbook.get_volume_weighted_price(order_size_decimal, "BUY")
        sell_vwap = orderbook.get_volume_weighted_price(order_size_decimal, "SELL")

        if buy_vwap is not None:
            assert buy_vwap > 0, "Buy VWAP should be positive when not None"

        if sell_vwap is not None:
            assert sell_vwap > 0, "Sell VWAP should be positive when not None"


class TestConfigurationProperties:
    """Property-based tests for configuration parameter validation."""

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=50, deadline=1000)
    def test_api_key_validation_property(self, key_text: str):
        """Property: API key validation should be consistent with length requirements."""
        result = APIKey.create(key_text)

        if len(key_text) >= 10:
            assert isinstance(
                result, Success
            ), f"Valid key {key_text[:10]}... should succeed"
            api_key = result.success()
            assert "***" in str(
                api_key
            ), "API key string representation should be masked"
        else:
            assert isinstance(result, Failure), f"Invalid key {key_text} should fail"
            assert "too short" in result.failure()

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=50, deadline=1000)
    def test_private_key_validation_property(self, key_text: str):
        """Property: Private key validation should be consistent with length requirements."""
        result = PrivateKey.create(key_text)

        if len(key_text) >= 20:
            assert isinstance(result, Success), "Valid private key should succeed"
            private_key = result.success()
            assert (
                str(private_key) == "PrivateKey(***)"
            ), "Private key should be fully masked"
        else:
            assert isinstance(result, Failure), "Invalid private key should fail"
            assert "too short" in result.failure()

    @given(
        st.integers(min_value=1, max_value=1000),
        st.floats(
            min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        st.booleans(),
    )
    @settings(max_examples=50, deadline=1000)
    def test_momentum_config_validation_property(
        self,
        lookback: int,
        entry_threshold: float,
        exit_threshold: float,
        use_volume: bool,
    ):
        """Property: Momentum strategy config validation should be consistent."""
        from bot.fp.types.config import MomentumStrategyConfig

        result = MomentumStrategyConfig.create(
            lookback_period=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_volume_confirmation=use_volume,
        )

        # Should succeed with valid parameters
        if lookback >= 1 and 0 < entry_threshold <= 1 and 0 < exit_threshold <= 1:
            assert isinstance(
                result, Success
            ), f"Valid config should succeed: lookback={lookback}, entry={entry_threshold}, exit={exit_threshold}"
            config = result.success()
            assert config.lookback_period == lookback
            assert config.use_volume_confirmation == use_volume
        else:
            assert isinstance(
                result, Failure
            ), f"Invalid config should fail: lookback={lookback}, entry={entry_threshold}, exit={exit_threshold}"


if __name__ == "__main__":
    # Run property tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
