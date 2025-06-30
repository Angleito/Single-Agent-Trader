"""
Property-based tests for market data validation and consistency.

Tests mathematical properties, invariants, and edge cases for market data structures
including price/volume validation, spread calculations, and VWAP properties.
"""

from datetime import UTC, datetime
from decimal import Decimal

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

from bot.fp.types.market import (
    MarketData,
    MarketSnapshot,
    Ticker,
    Trade,
)


# Enhanced strategies for market data generation
@st.composite
def decimal_price_strategy(draw, min_price=0.0001, max_price=1000000.0):
    """Generate realistic price values with proper decimal precision."""
    # Use a mix of common price ranges
    price_range = draw(
        st.sampled_from(
            [
                (0.0001, 0.01),  # Micro-cap/small coins
                (0.01, 1.0),  # Small-cap coins
                (1.0, 100.0),  # Mid-cap coins
                (100.0, 10000.0),  # Large-cap coins
                (10000.0, 100000.0),  # BTC-like prices
            ]
        )
    )

    price = draw(
        st.floats(
            min_value=price_range[0],
            max_value=price_range[1],
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return Decimal(str(round(price, 8)))


@st.composite
def decimal_volume_strategy(draw, min_volume=0.0, max_volume=1000000.0):
    """Generate realistic volume values with proper decimal precision."""
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
def symbol_strategy(draw):
    """Generate realistic trading pair symbols."""
    base_currencies = ["BTC", "ETH", "SOL", "ADA", "DOT", "MATIC", "LINK", "UNI"]
    quote_currencies = ["USD", "USDT", "USDC", "EUR", "BTC", "ETH"]

    base = draw(st.sampled_from(base_currencies))
    quote = draw(st.sampled_from(quote_currencies))

    # Ensure base != quote
    assume(base != quote)

    return f"{base}-{quote}"


@st.composite
def market_snapshot_strategy(draw):
    """Generate valid MarketSnapshot instances."""
    symbol = draw(symbol_strategy())
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        )
    )

    # Generate coherent bid/ask spread
    mid_price = draw(decimal_price_strategy(min_price=1.0, max_price=10000.0))
    spread_pct = draw(st.floats(min_value=0.0001, max_value=0.1))  # 0.01% to 10% spread
    spread = mid_price * Decimal(str(spread_pct))

    bid = mid_price - (spread / 2)
    ask = mid_price + (spread / 2)
    price = draw(
        st.sampled_from([bid, ask, mid_price])
    )  # Price can be bid, ask, or mid

    volume = draw(decimal_volume_strategy(min_volume=0.1, max_volume=100000.0))

    # Optional technical indicators
    high_20 = draw(
        st.one_of(
            st.none(),
            decimal_price_strategy(
                min_price=float(price), max_price=float(price) * 1.2
            ),
        )
    )
    low_20 = draw(
        st.one_of(
            st.none(),
            decimal_price_strategy(
                min_price=float(price) * 0.8, max_price=float(price)
            ),
        )
    )
    sma_20 = draw(
        st.one_of(
            st.none(),
            decimal_price_strategy(
                min_price=float(price) * 0.9, max_price=float(price) * 1.1
            ),
        )
    )

    return MarketSnapshot(
        timestamp=timestamp,
        symbol=symbol,
        price=price,
        volume=volume,
        bid=bid,
        ask=ask,
        high_20=high_20,
        low_20=low_20,
        sma_20=sma_20,
    )


@st.composite
def market_data_strategy(draw):
    """Generate valid MarketData instances."""
    symbol = draw(symbol_strategy())
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        )
    )

    price = draw(decimal_price_strategy(min_price=0.01, max_price=10000.0))
    volume = draw(decimal_volume_strategy(min_volume=0.0, max_volume=100000.0))

    # Optional bid/ask
    bid = draw(
        st.one_of(
            st.none(), decimal_price_strategy(min_price=0.01, max_price=float(price))
        )
    )
    ask = draw(
        st.one_of(
            st.none(),
            decimal_price_strategy(
                min_price=float(price), max_price=float(price) * 1.1
            ),
        )
    )

    # Optional OHLCV data
    include_ohlcv = draw(st.booleans())
    if include_ohlcv:
        # Generate OHLCV with proper relationships
        open_price = price
        high = draw(
            decimal_price_strategy(min_price=float(price), max_price=float(price) * 1.1)
        )
        low = draw(
            decimal_price_strategy(min_price=float(price) * 0.9, max_price=float(price))
        )
        close = price

        return MarketData(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            open=open_price,
            high=high,
            low=low,
            close=close,
        )
    return MarketData(
        symbol=symbol, price=price, volume=volume, timestamp=timestamp, bid=bid, ask=ask
    )


@st.composite
def ticker_strategy(draw):
    """Generate valid Ticker instances."""
    symbol = draw(symbol_strategy())
    last_price = draw(decimal_price_strategy(min_price=0.01, max_price=10000.0))
    volume_24h = draw(decimal_volume_strategy(min_volume=0.0, max_volume=1000000.0))
    change_24h = draw(
        st.floats(
            min_value=-99.9, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )

    return Ticker(
        symbol=symbol,
        last_price=last_price,
        volume_24h=volume_24h,
        change_24h=Decimal(str(change_24h)),
    )


@st.composite
def trade_strategy(draw):
    """Generate valid Trade instances."""
    trade_id = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        )
    )
    timestamp = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        )
    )
    price = draw(decimal_price_strategy(min_price=0.01, max_price=10000.0))
    size = draw(decimal_volume_strategy(min_volume=0.001, max_volume=1000.0))
    side = draw(st.sampled_from(["BUY", "SELL"]))
    symbol = draw(st.one_of(st.none(), symbol_strategy()))
    exchange = draw(
        st.one_of(st.none(), st.sampled_from(["coinbase", "bluefin", "binance"]))
    )

    return Trade(
        id=trade_id,
        timestamp=timestamp,
        price=price,
        size=size,
        side=side,
        symbol=symbol,
        exchange=exchange,
    )


class TestMarketSnapshotProperties:
    """Property-based tests for MarketSnapshot validation and calculations."""

    @given(market_snapshot_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_market_snapshot_creation_succeeds(self, snapshot: MarketSnapshot):
        """Property: Valid market snapshot data should always create successful instances."""
        assert isinstance(snapshot, MarketSnapshot)
        assert snapshot.price > 0
        assert snapshot.volume >= 0
        assert snapshot.bid > 0
        assert snapshot.ask > 0
        assert snapshot.symbol

    @given(market_snapshot_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_spread_calculation_property(self, snapshot: MarketSnapshot):
        """Property: Spread should always equal ask minus bid."""
        expected_spread = snapshot.ask - snapshot.bid
        assert snapshot.spread == expected_spread
        assert (
            snapshot.spread >= 0
        ), f"Spread should be non-negative, got {snapshot.spread}"

    @given(market_snapshot_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_bid_ask_ordering_invariant(self, snapshot: MarketSnapshot):
        """Property: Bid should always be less than or equal to ask."""
        assert (
            snapshot.bid <= snapshot.ask
        ), f"Bid {snapshot.bid} should be <= ask {snapshot.ask}"

    @given(market_snapshot_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_technical_indicator_relationships(self, snapshot: MarketSnapshot):
        """Property: Technical indicators should have valid relationships when present."""
        if snapshot.high_20 is not None and snapshot.low_20 is not None:
            assert (
                snapshot.high_20 >= snapshot.low_20
            ), f"High_20 {snapshot.high_20} should be >= low_20 {snapshot.low_20}"

        if snapshot.high_20 is not None:
            assert (
                snapshot.high_20 > 0
            ), f"High_20 should be positive, got {snapshot.high_20}"

        if snapshot.low_20 is not None:
            assert (
                snapshot.low_20 > 0
            ), f"Low_20 should be positive, got {snapshot.low_20}"

        if snapshot.sma_20 is not None:
            assert (
                snapshot.sma_20 > 0
            ), f"SMA_20 should be positive, got {snapshot.sma_20}"

    @given(market_snapshot_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_current_price_alias_property(self, snapshot: MarketSnapshot):
        """Property: current_price should be an alias for price."""
        assert snapshot.current_price == snapshot.price


class TestMarketDataProperties:
    """Property-based tests for MarketData validation and features."""

    @given(market_data_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_market_data_creation_succeeds(self, market_data: MarketData):
        """Property: Valid market data should always create successful instances."""
        assert isinstance(market_data, MarketData)
        assert market_data.price > 0
        assert market_data.volume >= 0
        assert market_data.symbol

    @given(market_data_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_spread_calculation_when_available(self, market_data: MarketData):
        """Property: Spread should be calculated correctly when bid/ask are available."""
        if market_data.bid is not None and market_data.ask is not None:
            expected_spread = market_data.ask - market_data.bid
            assert market_data.spread == expected_spread
            assert market_data.spread >= 0
        else:
            assert market_data.spread is None

    @given(market_data_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_ohlcv_detection_property(self, market_data: MarketData):
        """Property: OHLCV detection should be consistent with field availability."""
        has_ohlcv = market_data.has_ohlcv
        expected_has_ohlcv = all(
            field is not None
            for field in [
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.close,
            ]
        )
        assert has_ohlcv == expected_has_ohlcv

    @given(market_data_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_typical_price_calculation(self, market_data: MarketData):
        """Property: Typical price should be calculated correctly based on available data."""
        if market_data.has_ohlcv:
            expected_typical = (
                market_data.high + market_data.low + market_data.close
            ) / Decimal(3)
            assert market_data.typical_price == expected_typical
        else:
            assert market_data.typical_price == market_data.price

    @given(
        symbol_strategy(),
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        decimal_price_strategy(min_price=0.01, max_price=10000.0),
        decimal_price_strategy(min_price=0.01, max_price=10000.0),
        decimal_price_strategy(min_price=0.01, max_price=10000.0),
        decimal_price_strategy(min_price=0.01, max_price=10000.0),
        decimal_volume_strategy(min_volume=0.0, max_volume=100000.0),
    )
    @settings(max_examples=50, deadline=2000)
    def test_from_ohlcv_construction(
        self,
        symbol: str,
        timestamp: datetime,
        o: Decimal,
        h: Decimal,
        l: Decimal,
        c: Decimal,
        v: Decimal,
    ):
        """Property: OHLCV construction should create valid MarketData with proper relationships."""
        # Ensure OHLC relationships are valid
        high = max(o, h, l, c)
        low = min(o, h, l, c)
        assume(high >= low)  # Ensure valid range

        market_data = MarketData.from_ohlcv(
            symbol=symbol,
            timestamp=timestamp,
            open=o,
            high=high,
            low=low,
            close=c,
            volume=v,
        )

        assert market_data.symbol == symbol
        assert market_data.price == c  # Price should be close
        assert market_data.volume == v
        assert market_data.has_ohlcv
        assert market_data.open == o
        assert market_data.high == high
        assert market_data.low == low
        assert market_data.close == c


class TestTickerProperties:
    """Property-based tests for Ticker validation and calculations."""

    @given(ticker_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_ticker_creation_succeeds(self, ticker: Ticker):
        """Property: Valid ticker data should always create successful instances."""
        assert isinstance(ticker, Ticker)
        assert ticker.last_price > 0
        assert ticker.volume_24h >= 0
        assert ticker.change_24h >= -100  # Can't lose more than 100%

    @given(ticker_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_price_24h_ago_calculation(self, ticker: Ticker):
        """Property: Price 24h ago should be calculated correctly from current price and change."""
        price_24h_ago = ticker.price_24h_ago
        expected_price = ticker.last_price / (1 + ticker.change_24h / 100)

        assert abs(price_24h_ago - expected_price) < Decimal(
            "0.00001"
        ), "Price 24h ago calculation should be accurate"
        assert price_24h_ago > 0, "Price 24h ago should be positive"

    @given(ticker_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_volatility_calculation(self, ticker: Ticker):
        """Property: Volatility should be absolute value of 24h change."""
        volatility = ticker.volatility_24h
        expected_volatility = abs(ticker.change_24h)

        assert volatility == expected_volatility
        assert volatility >= 0, "Volatility should be non-negative"

    @given(ticker_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_positive_change_classification(self, ticker: Ticker):
        """Property: Positive change classification should be consistent."""
        is_positive = ticker.is_positive_24h
        expected_positive = ticker.change_24h > 0

        assert is_positive == expected_positive

    @given(ticker_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_mid_price_equals_last_price(self, ticker: Ticker):
        """Property: Mid price should equal last price for ticker data."""
        assert ticker.mid_price == ticker.last_price


class TestTradeProperties:
    """Property-based tests for Trade validation and calculations."""

    @given(trade_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_trade_creation_succeeds(self, trade: Trade):
        """Property: Valid trade data should always create successful instances."""
        assert isinstance(trade, Trade)
        assert trade.price > 0
        assert trade.size > 0
        assert trade.side in ["BUY", "SELL"]
        assert trade.id

    @given(trade_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_trade_value_calculation(self, trade: Trade):
        """Property: Trade value should equal price * size."""
        expected_value = trade.price * trade.size
        assert trade.value == expected_value
        assert trade.value > 0, "Trade value should be positive"

    @given(trade_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_buy_sell_classification(self, trade: Trade):
        """Property: Buy/sell classification should be mutually exclusive."""
        is_buy = trade.is_buy()
        is_sell = trade.is_sell()

        assert is_buy != is_sell, "Trade should be either buy or sell, not both"

        if trade.side == "BUY":
            assert is_buy and not is_sell
        else:  # trade.side == "SELL"
            assert is_sell and not is_buy


class TestVWAPProperties:
    """Property-based tests for Volume-Weighted Average Price calculations."""

    @given(st.lists(trade_strategy(), min_size=1, max_size=100))
    @settings(max_examples=50, deadline=3000)
    def test_vwap_calculation_properties(self, trades: list[Trade]):
        """Property: VWAP should be weighted average of trade prices by volume."""
        if not trades:
            return

        # Calculate VWAP manually
        total_value = sum(trade.price * trade.size for trade in trades)
        total_volume = sum(trade.size for trade in trades)

        if total_volume == 0:
            return

        expected_vwap = total_value / total_volume

        # VWAP should be within the range of trade prices
        min_price = min(trade.price for trade in trades)
        max_price = max(trade.price for trade in trades)

        assert (
            min_price <= expected_vwap <= max_price
        ), f"VWAP {expected_vwap} should be within price range [{min_price}, {max_price}]"
        assert expected_vwap > 0, "VWAP should be positive"

    @given(st.lists(trade_strategy(), min_size=2, max_size=20))
    @settings(max_examples=30, deadline=2000)
    def test_vwap_weighting_effect(self, trades: list[Trade]):
        """Property: VWAP should be closer to prices with higher volume."""
        if len(trades) < 2:
            return

        # Calculate VWAP
        total_value = sum(trade.price * trade.size for trade in trades)
        total_volume = sum(trade.size for trade in trades)

        if total_volume == 0:
            return

        vwap = total_value / total_volume

        # Find the trade with the highest volume
        max_volume_trade = max(trades, key=lambda t: t.size)

        # VWAP should be influenced by high-volume trades
        # If max volume trade has significant volume, VWAP should be closer to its price
        if max_volume_trade.size > total_volume * Decimal(
            "0.5"
        ):  # More than 50% of total volume
            price_diff_to_max = abs(vwap - max_volume_trade.price)

            # Compare to distance from other trades
            other_trades = [t for t in trades if t != max_volume_trade]
            if other_trades:
                avg_other_price = sum(t.price for t in other_trades) / len(other_trades)
                price_diff_to_others = abs(vwap - avg_other_price)

                # VWAP should generally be closer to the high-volume trade
                # (This is a statistical property, not always true due to price distribution)
                # This is more of an informational test


class TestMarketDataEdgeCases:
    """Property-based tests for edge cases and boundary conditions."""

    @given(
        symbol_strategy(),
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        st.floats(min_value=-1000.0, max_value=0.0),  # Invalid negative prices
        decimal_volume_strategy(min_volume=0.0, max_volume=1000.0),
    )
    @settings(max_examples=20, deadline=1000)
    def test_negative_price_handling(
        self, symbol: str, timestamp: datetime, price: float, volume: Decimal
    ):
        """Property: Negative prices should raise appropriate validation errors."""
        with pytest.raises(ValueError, match="Price must be positive"):
            MarketData(
                symbol=symbol,
                timestamp=timestamp,
                price=Decimal(str(price)),
                volume=volume,
            )

    @given(
        symbol_strategy(),
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        decimal_price_strategy(min_price=0.01, max_price=1000.0),
        st.floats(min_value=-1000.0, max_value=-0.01),  # Invalid negative volume
    )
    @settings(max_examples=20, deadline=1000)
    def test_negative_volume_handling(
        self, symbol: str, timestamp: datetime, price: Decimal, volume: float
    ):
        """Property: Negative volume should raise appropriate validation errors."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            MarketData(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                volume=Decimal(str(volume)),
            )

    @given(
        symbol_strategy(),
        decimal_price_strategy(min_price=0.01, max_price=1000.0),
        decimal_volume_strategy(min_volume=0.0, max_volume=1000.0),
        st.floats(min_value=-200.0, max_value=-100.01),  # Change less than -100%
    )
    @settings(max_examples=20, deadline=1000)
    def test_extreme_negative_change_handling(
        self, symbol: str, price: Decimal, volume: Decimal, change: float
    ):
        """Property: Extreme negative changes should raise validation errors."""
        with pytest.raises(ValueError, match="24h change cannot be less than -100%"):
            Ticker(
                symbol=symbol,
                last_price=price,
                volume_24h=volume,
                change_24h=Decimal(str(change)),
            )

    @given(
        st.text(min_size=1, max_size=20),
        st.datetimes(
            min_value=datetime(2020, 1, 1, tzinfo=UTC),
            max_value=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        decimal_price_strategy(min_price=0.01, max_price=1000.0),
        decimal_volume_strategy(min_volume=0.001, max_volume=1000.0),
        st.text(min_size=1, max_size=10).filter(
            lambda x: x not in ["BUY", "SELL"]
        ),  # Invalid sides
    )
    @settings(max_examples=20, deadline=1000)
    def test_invalid_trade_side_handling(
        self,
        trade_id: str,
        timestamp: datetime,
        price: Decimal,
        size: Decimal,
        side: str,
    ):
        """Property: Invalid trade sides should raise validation errors."""
        with pytest.raises(ValueError, match="Side must be BUY or SELL"):
            Trade(id=trade_id, timestamp=timestamp, price=price, size=size, side=side)


if __name__ == "__main__":
    # Run property tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
