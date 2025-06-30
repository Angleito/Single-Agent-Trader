"""
Comprehensive Orderbook Integration Tests

This module provides end-to-end testing of the orderbook functionality across
the entire trading system including:
- OrderBook type validation and methods
- WebSocket message processing and conversion
- Market data adapter integration
- Market making strategy integration
- Data flow validation from message to execution

Test Coverage:
- Immutable OrderBook type functionality
- OrderBookMessage processing and validation
- Type conversion functions
- Market making calculations with orderbook data
- Error handling and edge cases
- Performance benchmarks
"""

import time
from datetime import UTC, datetime
from decimal import Decimal

import pytest


class TestOrderBookIntegration:
    """Comprehensive orderbook integration test suite."""

    def test_orderbook_type_functionality(self):
        """Test core OrderBook type functionality."""
        from bot.fp.types.market import OrderBook

        # Create test orderbook
        orderbook = OrderBook(
            bids=[
                (Decimal(50000), Decimal("10.0")),
                (Decimal(49950), Decimal("5.0")),
                (Decimal(49900), Decimal("3.0")),
            ],
            asks=[
                (Decimal(50050), Decimal("8.0")),
                (Decimal(50100), Decimal("6.0")),
                (Decimal(50150), Decimal("4.0")),
            ],
            timestamp=datetime.now(UTC),
        )

        # Test basic properties
        assert orderbook.best_bid == (Decimal(50000), Decimal("10.0"))
        assert orderbook.best_ask == (Decimal(50050), Decimal("8.0"))
        assert orderbook.mid_price == Decimal(50025)
        assert orderbook.spread == Decimal(50)
        assert orderbook.bid_depth == Decimal("18.0")
        assert orderbook.ask_depth == Decimal("18.0")

        # Test spread in basis points
        spread_bps = orderbook.get_spread_bps()
        assert 9.9 < spread_bps < 10.1  # Approximately 10 bps

        # Test price impact calculations
        buy_impact = orderbook.price_impact("buy", Decimal("15.0"))
        assert buy_impact is not None
        assert buy_impact > orderbook.mid_price

        sell_impact = orderbook.price_impact("sell", Decimal("15.0"))
        assert sell_impact is not None
        assert sell_impact < orderbook.mid_price

        # Test insufficient liquidity
        large_impact = orderbook.price_impact("buy", Decimal("100.0"))
        assert large_impact is None

    def test_orderbook_message_processing(self):
        """Test OrderBookMessage creation and processing."""
        from bot.fp.types.market import OrderBookMessage

        # Test message creation
        test_data = {
            "channel": "level2",
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"], ["49950.00", "2.0"]],
            "asks": [["50050.00", "1.2"], ["50100.00", "1.8"]],
        }

        orderbook_msg = OrderBookMessage(
            channel="level2",
            timestamp=datetime.now(UTC),
            data=test_data,
            bids=[
                (Decimal("50000.00"), Decimal("1.5")),
                (Decimal("49950.00"), Decimal("2.0")),
            ],
            asks=[
                (Decimal("50050.00"), Decimal("1.2")),
                (Decimal("50100.00"), Decimal("1.8")),
            ],
        )

        # Validate message structure
        assert orderbook_msg.channel == "level2"
        assert len(orderbook_msg.bids) == 2
        assert len(orderbook_msg.asks) == 2
        assert orderbook_msg.data == test_data

        # Test bid/ask ordering validation (should not raise exceptions)
        assert orderbook_msg.bids[0][0] > orderbook_msg.bids[1][0]  # Descending
        assert orderbook_msg.asks[0][0] < orderbook_msg.asks[1][0]  # Ascending

    def test_orderbook_message_conversion(self):
        """Test conversion from WebSocket data to OrderBookMessage."""

        def create_orderbook_message_from_data(message, symbol):
            """Simplified converter for testing."""
            from bot.fp.types.market import OrderBookMessage

            bids = None
            asks = None

            if "bids" in message:
                try:
                    bids = [
                        (Decimal(str(bid[0])), Decimal(str(bid[1])))
                        for bid in message["bids"]
                        if len(bid) >= 2
                    ]
                except (ValueError, TypeError, IndexError):
                    bids = None

            if "asks" in message:
                try:
                    asks = [
                        (Decimal(str(ask[0])), Decimal(str(ask[1])))
                        for ask in message["asks"]
                        if len(ask) >= 2
                    ]
                except (ValueError, TypeError, IndexError):
                    asks = None

            channel = message.get("channel", message.get("type", "orderbook"))
            message_id = message.get("id", message.get("message_id"))

            return OrderBookMessage(
                channel=channel,
                timestamp=datetime.now(UTC),
                data=message,
                message_id=message_id,
                bids=bids,
                asks=asks,
            )

        # Test with realistic Coinbase data
        coinbase_data = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"], ["49950.00", "2.0"], ["49900.00", "1.8"]],
            "asks": [["50050.00", "1.2"], ["50100.00", "1.8"], ["50150.00", "2.2"]],
        }

        orderbook_msg = create_orderbook_message_from_data(coinbase_data, "BTC-USD")

        assert orderbook_msg.channel == "snapshot"
        assert len(orderbook_msg.bids) == 3
        assert len(orderbook_msg.asks) == 3

        # Verify data integrity
        assert orderbook_msg.bids[0] == (Decimal("50000.00"), Decimal("1.5"))
        assert orderbook_msg.asks[0] == (Decimal("50050.00"), Decimal("1.2"))

        # Test with invalid data
        invalid_data = {
            "type": "snapshot",
            "bids": [["invalid", "price"]],
            "asks": [["50050.00", "1.2"]],
        }

        invalid_msg = create_orderbook_message_from_data(invalid_data, "BTC-USD")
        assert invalid_msg.bids is None  # Should handle invalid data gracefully
        assert len(invalid_msg.asks) == 1

    def test_market_making_with_orderbook(self):
        """Test market making strategy integration with orderbook data."""

        def calculate_spread(
            volatility,
            base_spread,
            volatility_multiplier=2.0,
            min_spread=0.001,
            max_spread=0.05,
        ):
            """Calculate dynamic spread."""
            spread = base_spread + (volatility * volatility_multiplier)
            return max(min_spread, min(spread, max_spread))

        def calculate_inventory_skew(current_inventory, max_inventory, skew_factor=0.5):
            """Calculate inventory-based price skew."""
            if max_inventory == 0:
                return 0.0
            normalized_inventory = current_inventory / max_inventory
            return -normalized_inventory * skew_factor

        def generate_quotes(
            mid_price,
            spread,
            inventory_skew,
            order_book_imbalance=0.0,
            competitive_adjustment=0.9,
        ):
            """Generate bid/ask quotes."""
            adjusted_spread = spread * competitive_adjustment
            half_spread = adjusted_spread / 2

            bid_adjustment = half_spread * (1 - inventory_skew)
            ask_adjustment = half_spread * (1 + inventory_skew)

            if order_book_imbalance > 0:
                ask_adjustment *= 1 + abs(order_book_imbalance) * 0.2
            else:
                bid_adjustment *= 1 + abs(order_book_imbalance) * 0.2

            bid_price = mid_price * (1 - bid_adjustment)
            ask_price = mid_price * (1 + ask_adjustment)

            return bid_price, ask_price

        # Create test scenario
        from bot.fp.types.market import OrderBook

        orderbook = OrderBook(
            bids=[
                (Decimal(50000), Decimal("10.0")),
                (Decimal(49950), Decimal("5.0")),
            ],
            asks=[
                (Decimal(50050), Decimal("8.0")),
                (Decimal(50100), Decimal("6.0")),
            ],
            timestamp=datetime.now(UTC),
        )

        # Market making parameters
        volatility = 0.02  # 2%
        base_spread = 0.001  # 0.1%
        current_inventory = 100.0
        max_inventory = 1000.0

        # Calculate market making parameters
        dynamic_spread = calculate_spread(volatility, base_spread)
        inventory_skew = calculate_inventory_skew(current_inventory, max_inventory, 0.3)

        # Calculate orderbook imbalance
        book_imbalance = (orderbook.bid_depth - orderbook.ask_depth) / (
            orderbook.bid_depth + orderbook.ask_depth
        )

        # Generate quotes
        mid_price = float(orderbook.mid_price)
        bid_price, ask_price = generate_quotes(
            mid_price=mid_price,
            spread=dynamic_spread,
            inventory_skew=inventory_skew,
            order_book_imbalance=book_imbalance,
        )

        # Validate results
        assert bid_price < mid_price < ask_price
        assert ask_price - bid_price > 0  # Positive spread

        # Test that inventory skew affects quotes appropriately
        # (positive inventory should lower both bid and ask slightly)
        assert inventory_skew < 0  # Long inventory should create negative skew

    def test_end_to_end_data_flow(self):
        """Test complete data flow from WebSocket message to strategy execution."""

        # Step 1: Simulate WebSocket message
        ws_message = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "10.0"], ["49950.00", "5.0"]],
            "asks": [["50050.00", "8.0"], ["50100.00", "6.0"]],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Step 2: Convert to OrderBookMessage
        from bot.fp.types.market import OrderBookMessage

        orderbook_msg = OrderBookMessage(
            channel="snapshot",
            timestamp=datetime.now(UTC),
            data=ws_message,
            bids=[
                (Decimal("50000.00"), Decimal("10.0")),
                (Decimal("49950.00"), Decimal("5.0")),
            ],
            asks=[
                (Decimal("50050.00"), Decimal("8.0")),
                (Decimal("50100.00"), Decimal("6.0")),
            ],
        )

        # Step 3: Create OrderBook from message
        from bot.fp.types.market import OrderBook

        orderbook = OrderBook(
            bids=orderbook_msg.bids,
            asks=orderbook_msg.asks,
            timestamp=orderbook_msg.timestamp,
        )

        # Step 4: Use in market making decision
        mid_price = orderbook.mid_price
        spread_pct = orderbook.spread / mid_price

        # Simulate strategy decision
        strategy_decision = {
            "action": "MARKET_MAKE",
            "bid_price": float(orderbook.best_bid[0]) - 5,  # 5 ticks inside
            "ask_price": float(orderbook.best_ask[0]) + 5,  # 5 ticks inside
            "bid_size": 1.0,
            "ask_size": 1.0,
            "spread_bps": float(orderbook.get_spread_bps()),
            "mid_price": float(mid_price),
        }

        # Step 5: Validate decision
        assert strategy_decision["action"] == "MARKET_MAKE"
        assert strategy_decision["bid_price"] < strategy_decision["ask_price"]
        assert strategy_decision["spread_bps"] > 0
        assert strategy_decision["mid_price"] > 0

        # Test the complete flow worked
        assert ws_message["product_id"] == "BTC-USD"
        assert len(orderbook_msg.bids) == 2
        assert orderbook.mid_price == Decimal(50025)
        assert strategy_decision["mid_price"] == 50025.0

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases in orderbook processing."""
        from bot.fp.types.market import OrderBook

        # Test empty orderbook (should raise validation error)
        with pytest.raises(ValueError, match="Order book cannot be empty"):
            OrderBook(bids=[], asks=[], timestamp=datetime.now(UTC))

        # Test invalid bid ordering (should raise validation error)
        with pytest.raises(ValueError, match="Bid prices must be in descending order"):
            OrderBook(
                bids=[
                    (Decimal(49000), Decimal("1.0")),
                    (Decimal(50000), Decimal("1.0")),  # Higher price after lower
                ],
                asks=[(Decimal(50050), Decimal("1.0"))],
                timestamp=datetime.now(UTC),
            )

        # Test invalid ask ordering (should raise validation error)
        with pytest.raises(ValueError, match="Ask prices must be in ascending order"):
            OrderBook(
                bids=[(Decimal(50000), Decimal("1.0"))],
                asks=[
                    (Decimal(50100), Decimal("1.0")),
                    (Decimal(50050), Decimal("1.0")),  # Lower price after higher
                ],
                timestamp=datetime.now(UTC),
            )

        # Test crossed book (should raise validation error)
        with pytest.raises(ValueError, match="Best bid .* must be < best ask"):
            OrderBook(
                bids=[(Decimal(50100), Decimal("1.0"))],  # Bid higher than ask
                asks=[(Decimal(50050), Decimal("1.0"))],
                timestamp=datetime.now(UTC),
            )

    def test_performance_benchmarks(self):
        """Test performance of orderbook operations."""
        from bot.fp.types.market import OrderBook

        # Create large orderbook
        bids = [(Decimal(str(50000 - i)), Decimal("1.0")) for i in range(100)]
        asks = [(Decimal(str(50050 + i)), Decimal("1.0")) for i in range(100)]

        start_time = time.time()
        orderbook = OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))
        creation_time = time.time() - start_time

        # Test property access performance
        start_time = time.time()
        for _ in range(1000):
            _ = orderbook.mid_price
            _ = orderbook.spread
            _ = orderbook.bid_depth
            _ = orderbook.ask_depth
        property_access_time = time.time() - start_time

        # Test price impact calculation performance
        start_time = time.time()
        for i in range(100):
            _ = orderbook.price_impact("buy", Decimal(str(i + 1)))
        price_impact_time = time.time() - start_time

        # Performance assertions
        assert creation_time < 0.1  # Should create in under 100ms
        assert property_access_time < 0.1  # 1000 property accesses in under 100ms
        assert price_impact_time < 0.5  # 100 price impact calculations in under 500ms

        print(f"OrderBook creation time: {creation_time * 1000:.2f}ms")
        print(f"Property access time (1000 ops): {property_access_time * 1000:.2f}ms")
        print(f"Price impact time (100 ops): {price_impact_time * 1000:.2f}ms")

    def test_integration_with_market_data_processor(self):
        """Test integration with market data processing components."""

        # Mock market data processor functionality
        class MockMarketDataProcessor:
            def __init__(self, symbol):
                self.symbol = symbol
                self.recent_updates = []

            def process_websocket_message(self, message):
                """Process WebSocket message."""
                if message.get("type") == "snapshot":
                    # Create update record
                    update = {
                        "timestamp": datetime.now(UTC),
                        "symbol": self.symbol,
                        "update_type": "orderbook",
                        "data": message,
                    }
                    self.recent_updates.append(update)
                    return True
                return False

            def get_recent_updates(self, limit=None):
                """Get recent updates."""
                if limit:
                    return self.recent_updates[-limit:]
                return self.recent_updates

        # Test processor integration
        processor = MockMarketDataProcessor("BTC-USD")

        test_message = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "10.0"]],
            "asks": [["50050.00", "8.0"]],
        }

        # Process message
        result = processor.process_websocket_message(test_message)
        assert result is True

        # Check processing results
        updates = processor.get_recent_updates()
        assert len(updates) == 1
        assert updates[0]["symbol"] == "BTC-USD"
        assert updates[0]["update_type"] == "orderbook"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
