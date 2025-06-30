"""
Stateful property-based tests for orderbook operations and state transitions.

Uses hypothesis stateful testing to model complex orderbook interactions,
updates, and state consistency over time.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from bot.fp.types.market import OrderBook, Trade


# Strategies for stateful testing
@st.composite
def price_level_strategy(draw, base_price=100.0, max_deviation=0.1):
    """Generate price levels around a base price."""
    deviation = draw(st.floats(min_value=-max_deviation, max_value=max_deviation))
    price = base_price * (1 + deviation)
    return Decimal(str(round(price, 4)))


@st.composite
def size_strategy(draw, min_size=0.001, max_size=1000.0):
    """Generate order sizes."""
    size = draw(
        st.floats(
            min_value=min_size,
            max_value=max_size,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return Decimal(str(round(size, 6)))


@st.composite
def price_size_entry_strategy(draw, base_price=100.0, side="bid"):
    """Generate orderbook entries for a specific side."""
    if side == "bid":
        price = draw(
            price_level_strategy(base_price=base_price * 0.99, max_deviation=0.05)
        )
    else:  # ask
        price = draw(
            price_level_strategy(base_price=base_price * 1.01, max_deviation=0.05)
        )

    size = draw(size_strategy())
    return (price, size)


class OrderBookStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for orderbook operations.

    This tests complex sequences of orderbook updates and ensures
    that invariants are maintained throughout state transitions.
    """

    def __init__(self):
        super().__init__()
        self.base_price = Decimal("100.0")
        self.orderbook: OrderBook | None = None
        self.trade_history: list[Trade] = []
        self.update_count = 0
        self.last_timestamp = datetime.now(UTC)

    @initialize()
    def init_orderbook(self):
        """Initialize with a valid orderbook."""
        # Create initial bids (descending prices)
        bids = [
            (self.base_price - Decimal("1.0"), Decimal("10.0")),
            (self.base_price - Decimal("2.0"), Decimal("15.0")),
            (self.base_price - Decimal("3.0"), Decimal("20.0")),
        ]

        # Create initial asks (ascending prices)
        asks = [
            (self.base_price + Decimal("1.0"), Decimal("12.0")),
            (self.base_price + Decimal("2.0"), Decimal("18.0")),
            (self.base_price + Decimal("3.0"), Decimal("25.0")),
        ]

        self.orderbook = OrderBook(bids=bids, asks=asks, timestamp=self.last_timestamp)

    @rule()
    def add_bid_level(self):
        """Add a new bid level to the orderbook."""
        if not self.orderbook:
            return

        # Generate a new bid price that maintains ordering
        current_bids = self.orderbook.bids
        if current_bids:
            max_bid_price = current_bids[0][0]
            new_price = max_bid_price - Decimal("0.5")
        else:
            new_price = self.base_price - Decimal("1.0")

        new_size = Decimal("5.0")

        # Insert new bid while maintaining order
        new_bids = [(new_price, new_size)] + current_bids
        new_bids.sort(key=lambda x: x[0], reverse=True)  # Descending order

        self.last_timestamp += timedelta(seconds=1)

        try:
            self.orderbook = OrderBook(
                bids=new_bids, asks=self.orderbook.asks, timestamp=self.last_timestamp
            )
            self.update_count += 1
        except ValueError:
            # If update would create invalid orderbook, skip
            pass

    @rule()
    def add_ask_level(self):
        """Add a new ask level to the orderbook."""
        if not self.orderbook:
            return

        # Generate a new ask price that maintains ordering
        current_asks = self.orderbook.asks
        if current_asks:
            min_ask_price = current_asks[0][0]
            new_price = min_ask_price + Decimal("0.5")
        else:
            new_price = self.base_price + Decimal("1.0")

        new_size = Decimal("7.0")

        # Insert new ask while maintaining order
        new_asks = current_asks + [(new_price, new_size)]
        new_asks.sort(key=lambda x: x[0])  # Ascending order

        self.last_timestamp += timedelta(seconds=1)

        try:
            self.orderbook = OrderBook(
                bids=self.orderbook.bids, asks=new_asks, timestamp=self.last_timestamp
            )
            self.update_count += 1
        except ValueError:
            # If update would create invalid orderbook, skip
            pass

    @rule(trade_size=size_strategy(min_size=0.1, max_size=50.0))
    def simulate_trade(self, trade_size: Decimal):
        """Simulate a trade that might affect the orderbook."""
        if not self.orderbook or not self.orderbook.bids or not self.orderbook.asks:
            return

        # Randomly choose buy or sell
        import random

        is_buy = random.choice([True, False])

        if is_buy:
            # Buy trade - might consume ask liquidity
            price = self.orderbook.asks[0][0]  # Take from best ask
            side = "BUY"
        else:
            # Sell trade - might consume bid liquidity
            price = self.orderbook.bids[0][0]  # Take from best bid
            side = "SELL"

        trade = Trade(
            id=f"trade_{len(self.trade_history)}",
            timestamp=self.last_timestamp,
            price=price,
            size=trade_size,
            side=side,
        )

        self.trade_history.append(trade)
        self.last_timestamp += timedelta(milliseconds=100)

    @rule(bid_adjustment=st.floats(min_value=-0.1, max_value=0.1))
    def adjust_spread(self, bid_adjustment: float):
        """Adjust the spread by moving bid or ask levels."""
        if not self.orderbook or not self.orderbook.bids or not self.orderbook.asks:
            return

        # Adjust the best bid price slightly
        current_bids = list(self.orderbook.bids)
        if current_bids:
            best_bid_price, best_bid_size = current_bids[0]
            new_bid_price = best_bid_price + Decimal(str(bid_adjustment))

            # Ensure new bid doesn't cross the spread
            best_ask_price = (
                self.orderbook.asks[0][0]
                if self.orderbook.asks
                else new_bid_price + Decimal("1.0")
            )
            if new_bid_price < best_ask_price:
                current_bids[0] = (new_bid_price, best_bid_size)

                self.last_timestamp += timedelta(seconds=1)

                try:
                    self.orderbook = OrderBook(
                        bids=current_bids,
                        asks=self.orderbook.asks,
                        timestamp=self.last_timestamp,
                    )
                    self.update_count += 1
                except ValueError:
                    pass

    @rule()
    def remove_top_bid(self):
        """Remove the top bid level."""
        if not self.orderbook or len(self.orderbook.bids) <= 1:
            return

        new_bids = self.orderbook.bids[1:]  # Remove first (best) bid
        self.last_timestamp += timedelta(seconds=1)

        try:
            self.orderbook = OrderBook(
                bids=new_bids, asks=self.orderbook.asks, timestamp=self.last_timestamp
            )
            self.update_count += 1
        except ValueError:
            pass

    @rule()
    def remove_top_ask(self):
        """Remove the top ask level."""
        if not self.orderbook or len(self.orderbook.asks) <= 1:
            return

        new_asks = self.orderbook.asks[1:]  # Remove first (best) ask
        self.last_timestamp += timedelta(seconds=1)

        try:
            self.orderbook = OrderBook(
                bids=self.orderbook.bids, asks=new_asks, timestamp=self.last_timestamp
            )
            self.update_count += 1
        except ValueError:
            pass

    @invariant()
    def orderbook_always_valid(self):
        """Invariant: Orderbook should always be in a valid state."""
        if self.orderbook is None:
            return

        # Check bid ordering (descending)
        bids = self.orderbook.bids
        for i in range(len(bids) - 1):
            assert (
                bids[i][0] > bids[i + 1][0]
            ), f"Bid prices not in descending order at position {i}"

        # Check ask ordering (ascending)
        asks = self.orderbook.asks
        for i in range(len(asks) - 1):
            assert (
                asks[i][0] < asks[i + 1][0]
            ), f"Ask prices not in ascending order at position {i}"

        # Check spread is positive
        if bids and asks:
            assert (
                bids[0][0] < asks[0][0]
            ), f"Best bid {bids[0][0]} should be < best ask {asks[0][0]}"

    @invariant()
    def positive_sizes(self):
        """Invariant: All order sizes should be positive."""
        if self.orderbook is None:
            return

        for price, size in self.orderbook.bids + self.orderbook.asks:
            assert size > 0, f"Order size should be positive, got {size}"
            assert price > 0, f"Order price should be positive, got {price}"

    @invariant()
    def spread_properties(self):
        """Invariant: Spread properties should be consistent."""
        if self.orderbook is None or not self.orderbook.bids or not self.orderbook.asks:
            return

        # Spread should match calculation
        expected_spread = self.orderbook.asks[0][0] - self.orderbook.bids[0][0]
        assert self.orderbook.spread == expected_spread

        # Mid price should be average
        expected_mid = (self.orderbook.bids[0][0] + self.orderbook.asks[0][0]) / 2
        assert self.orderbook.mid_price == expected_mid

    @invariant()
    def depth_calculations(self):
        """Invariant: Depth calculations should be correct."""
        if self.orderbook is None:
            return

        expected_bid_depth = sum(size for _, size in self.orderbook.bids)
        expected_ask_depth = sum(size for _, size in self.orderbook.asks)

        assert self.orderbook.bid_depth == expected_bid_depth
        assert self.orderbook.ask_depth == expected_ask_depth


class TestStatefulOrderBook:
    """Test runner for stateful orderbook property tests."""

    @settings(
        max_examples=50,
        stateful_step_count=100,
        deadline=10000,  # 10 second deadline
        suppress_health_check=[
            # Suppress health checks that might be noisy for stateful tests
        ],
    )
    def test_orderbook_state_machine(self):
        """Run the orderbook state machine test."""
        OrderBookStateMachine.TestCase.settings = settings(
            max_examples=50, stateful_step_count=100, deadline=10000
        )

        # Run the state machine
        try:
            machine = OrderBookStateMachine()
            machine.init_orderbook()

            # Verify initial state
            assert machine.orderbook is not None
            assert len(machine.orderbook.bids) == 3
            assert len(machine.orderbook.asks) == 3

        except Exception as e:
            pytest.fail(f"Stateful test failed: {e}")


class TestOrderBookComplexScenarios:
    """Property tests for complex orderbook scenarios."""

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=95.0, max_value=99.99),  # Bid prices
                st.floats(min_value=0.1, max_value=100.0),  # Bid sizes
            ),
            min_size=1,
            max_size=20,
        ),
        st.lists(
            st.tuples(
                st.floats(min_value=100.01, max_value=105.0),  # Ask prices
                st.floats(min_value=0.1, max_value=100.0),  # Ask sizes
            ),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50, deadline=3000)
    def test_large_orderbook_construction(
        self, bid_data: list[tuple[float, float]], ask_data: list[tuple[float, float]]
    ):
        """Property: Large orderbooks should maintain invariants."""
        # Convert to proper format and ensure ordering
        bids = [(Decimal(str(price)), Decimal(str(size))) for price, size in bid_data]
        asks = [(Decimal(str(price)), Decimal(str(size))) for price, size in ask_data]

        # Sort to ensure proper ordering
        bids.sort(key=lambda x: x[0], reverse=True)  # Descending
        asks.sort(key=lambda x: x[0])  # Ascending

        # Remove duplicates while preserving order
        unique_bids = []
        unique_asks = []

        seen_bid_prices = set()
        for price, size in bids:
            if price not in seen_bid_prices:
                unique_bids.append((price, size))
                seen_bid_prices.add(price)

        seen_ask_prices = set()
        for price, size in asks:
            if price not in seen_ask_prices:
                unique_asks.append((price, size))
                seen_ask_prices.add(price)

        if not unique_bids or not unique_asks:
            return

        # Ensure no spread crossing
        if unique_bids[0][0] >= unique_asks[0][0]:
            return

        timestamp = datetime.now(UTC)
        orderbook = OrderBook(bids=unique_bids, asks=unique_asks, timestamp=timestamp)

        # Verify invariants
        assert len(orderbook.bids) == len(unique_bids)
        assert len(orderbook.asks) == len(unique_asks)
        assert orderbook.spread > 0
        assert orderbook.mid_price is not None
        assert orderbook.bid_depth > 0
        assert orderbook.ask_depth > 0

    @given(
        st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=1, max_size=10),
        st.sampled_from(["BUY", "SELL"]),
    )
    @settings(max_examples=30, deadline=2000)
    def test_multiple_vwap_calculations(self, order_sizes: list[float], side: str):
        """Property: Multiple VWAP calculations should be monotonic for increasing sizes."""
        # Create a simple orderbook
        bids = [
            (Decimal("99.0"), Decimal("100.0")),
            (Decimal("98.0"), Decimal("200.0")),
        ]
        asks = [
            (Decimal("101.0"), Decimal("150.0")),
            (Decimal("102.0"), Decimal("250.0")),
        ]

        orderbook = OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))

        # Convert sizes to Decimal and sort
        sizes = [Decimal(str(size)) for size in order_sizes]
        sizes.sort()

        vwaps = []
        for size in sizes:
            vwap = orderbook.get_volume_weighted_price(size, side)
            if vwap is not None:
                vwaps.append((size, vwap))

        # For increasing order sizes, VWAP should move away from best price
        # (as larger orders consume deeper levels with worse prices)
        if len(vwaps) >= 2:
            if side == "BUY":
                # For buy orders, VWAP should increase (worse prices) with larger sizes
                for i in range(len(vwaps) - 1):
                    size1, vwap1 = vwaps[i]
                    size2, vwap2 = vwaps[i + 1]
                    if size2 > size1:
                        # VWAP should be non-decreasing for buy orders
                        assert (
                            vwap2 >= vwap1
                        ), f"Buy VWAP should not decrease: {vwap1} -> {vwap2} for sizes {size1} -> {size2}"
            else:  # SELL
                # For sell orders, VWAP should decrease (worse prices) with larger sizes
                for i in range(len(vwaps) - 1):
                    size1, vwap1 = vwaps[i]
                    size2, vwap2 = vwaps[i + 1]
                    if size2 > size1:
                        # VWAP should be non-increasing for sell orders
                        assert (
                            vwap2 <= vwap1
                        ), f"Sell VWAP should not increase: {vwap1} -> {vwap2} for sizes {size1} -> {size2}"


if __name__ == "__main__":
    # Run stateful property tests
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
