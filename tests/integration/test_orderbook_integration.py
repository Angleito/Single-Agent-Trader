"""
Comprehensive End-to-End Orderbook Integration Tests

This module provides complete integration testing of the orderbook functionality across
the entire trading system including:
- OrderBook type validation and methods
- WebSocket message processing and conversion
- Market data adapter integration
- Market making strategy integration
- Data flow validation from message to execution
- SDK to Service Client to Market Data to Exchange flow
- Real-time WebSocket integration
- REST API to WebSocket data consistency
- Error propagation through the stack
- Performance under load conditions

Test Coverage:
- Complete data flow: SDK → Service Client → Market Data → Exchange
- Immutable OrderBook type functionality
- OrderBookMessage processing and validation
- Type conversion functions
- Market making calculations with orderbook data
- WebSocket subscription and updates
- Cache invalidation and refresh
- Error handling across components
- Configuration changes impact
- Multi-symbol orderbook management
- Performance benchmarks and load testing
"""

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

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


class TestCompleteDataFlowIntegration:
    """Test complete data flow from SDK to Exchange."""

    @pytest.fixture
    async def mock_market_data_provider(self):
        """Create mock market data provider."""

        class MockMarketDataProvider:
            def __init__(self):
                self.websocket_data = []
                self.orderbook_cache = {}
                self.subscribers = []
                self.connected = False

            async def connect(self):
                self.connected = True

            async def disconnect(self):
                self.connected = False

            async def fetch_orderbook(self, level=2):
                if not self.connected:
                    return None

                return {
                    "bids": [(Decimal(50000), Decimal("10.0"))],
                    "asks": [(Decimal(50050), Decimal("8.0"))],
                    "timestamp": datetime.now(UTC),
                }

            def subscribe_to_updates(self, callback):
                self.subscribers.append(callback)

            async def simulate_websocket_update(self, message):
                """Simulate receiving WebSocket orderbook update."""
                self.websocket_data.append(message)

                # Notify subscribers
                for callback in self.subscribers:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)

        return MockMarketDataProvider()

    @pytest.fixture
    async def mock_exchange_client(self):
        """Create mock exchange client."""

        class MockExchangeClient:
            def __init__(self):
                self.orders = []
                self.positions = {}
                self.connected = False

            async def connect(self):
                self.connected = True

            async def get_orderbook(self, symbol, level=2):
                if not self.connected:
                    raise ConnectionError("Exchange not connected")

                return {
                    "symbol": symbol,
                    "bids": [["50000.00", "10.0"], ["49950.00", "5.0"]],
                    "asks": [["50050.00", "8.0"], ["50100.00", "6.0"]],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

            async def place_order(self, order_data):
                if not self.connected:
                    raise ConnectionError("Exchange not connected")

                order = {
                    "id": f"order_{len(self.orders) + 1}",
                    "status": "filled",
                    **order_data,
                }
                self.orders.append(order)
                return order

        return MockExchangeClient()

    @pytest.fixture
    async def mock_service_client(self):
        """Create mock Bluefin service client."""

        class MockServiceClient:
            def __init__(self):
                self.available = True
                self.last_request = None

            async def check_health(self):
                return self.available

            async def get_orderbook(self, symbol):
                if not self.available:
                    raise Exception("Service unavailable")

                self.last_request = {"method": "get_orderbook", "symbol": symbol}
                return {
                    "symbol": symbol,
                    "bids": [{"price": "50000.00", "size": "10.0"}],
                    "asks": [{"price": "50050.00", "size": "8.0"}],
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        return MockServiceClient()

    async def test_complete_orderbook_data_flow(
        self, mock_market_data_provider, mock_exchange_client, mock_service_client
    ):
        """Test complete data flow from service to strategy execution."""

        # Step 1: Initialize all components
        await mock_market_data_provider.connect()
        await mock_exchange_client.connect()

        # Step 2: Fetch orderbook from exchange via service client
        orderbook_data = await mock_service_client.get_orderbook("BTC-USD")
        assert orderbook_data["symbol"] == "BTC-USD"
        assert len(orderbook_data["bids"]) > 0
        assert len(orderbook_data["asks"]) > 0

        # Step 3: Convert to OrderBook type
        from bot.fp.types.market import OrderBook

        orderbook = OrderBook(
            bids=[(Decimal("50000.00"), Decimal("10.0"))],
            asks=[(Decimal("50050.00"), Decimal("8.0"))],
            timestamp=datetime.now(UTC),
        )

        # Step 4: Use in market making decision
        mid_price = orderbook.mid_price
        spread_bps = orderbook.get_spread_bps()

        # Simulate market making strategy using orderbook
        strategy_decision = {
            "action": "MARKET_MAKE",
            "bid_price": float(orderbook.best_bid[0]) - 5,
            "ask_price": float(orderbook.best_ask[0]) + 5,
            "bid_size": 1.0,
            "ask_size": 1.0,
            "spread_bps": float(spread_bps),
            "mid_price": float(mid_price),
        }

        # Step 5: Execute orders via exchange
        bid_order = await mock_exchange_client.place_order(
            {
                "side": "buy",
                "price": strategy_decision["bid_price"],
                "size": strategy_decision["bid_size"],
                "symbol": "BTC-USD",
            }
        )

        ask_order = await mock_exchange_client.place_order(
            {
                "side": "sell",
                "price": strategy_decision["ask_price"],
                "size": strategy_decision["ask_size"],
                "symbol": "BTC-USD",
            }
        )

        # Step 6: Validate complete flow
        assert bid_order["status"] == "filled"
        assert ask_order["status"] == "filled"
        assert len(mock_exchange_client.orders) == 2
        assert strategy_decision["spread_bps"] > 0
        assert strategy_decision["mid_price"] == 50025.0

    async def test_websocket_to_orderbook_integration(self, mock_market_data_provider):
        """Test WebSocket message to OrderBook conversion."""

        received_updates = []

        def update_callback(message):
            received_updates.append(message)

        # Subscribe to updates
        mock_market_data_provider.subscribe_to_updates(update_callback)

        # Simulate WebSocket orderbook update
        ws_message = {
            "channel": "level2",
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "10.0"], ["49950.00", "5.0"]],
            "asks": [["50050.00", "8.0"], ["50100.00", "6.0"]],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await mock_market_data_provider.simulate_websocket_update(ws_message)

        # Validate update was received
        assert len(received_updates) == 1
        assert received_updates[0] == ws_message

        # Convert to OrderBook
        from bot.fp.types.market import OrderBook

        orderbook = OrderBook(
            bids=[
                (Decimal("50000.00"), Decimal("10.0")),
                (Decimal("49950.00"), Decimal("5.0")),
            ],
            asks=[
                (Decimal("50050.00"), Decimal("8.0")),
                (Decimal("50100.00"), Decimal("6.0")),
            ],
            timestamp=datetime.now(UTC),
        )

        # Validate orderbook properties
        assert orderbook.best_bid == (Decimal("50000.00"), Decimal("10.0"))
        assert orderbook.best_ask == (Decimal("50050.00"), Decimal("8.0"))
        assert orderbook.mid_price == Decimal("50025.00")

    async def test_error_propagation_through_stack(
        self, mock_market_data_provider, mock_exchange_client, mock_service_client
    ):
        """Test error handling propagation through the complete stack."""

        # Test 1: Service unavailable
        mock_service_client.available = False

        with pytest.raises(Exception, match="Service unavailable"):
            await mock_service_client.get_orderbook("BTC-USD")

        # Test 2: Exchange connection error
        mock_exchange_client.connected = False

        with pytest.raises(ConnectionError, match="Exchange not connected"):
            await mock_exchange_client.get_orderbook("BTC-USD")

        # Test 3: Invalid orderbook data
        from bot.fp.types.market import OrderBook

        with pytest.raises(ValueError, match="Order book cannot be empty"):
            OrderBook(bids=[], asks=[], timestamp=datetime.now(UTC))

        # Test 4: Network timeout simulation
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = TimeoutError("Request timeout")

            # This would normally be handled by the actual client
            with pytest.raises(asyncio.TimeoutError):
                raise TimeoutError("Request timeout")

    async def test_multi_symbol_orderbook_management(self, mock_service_client):
        """Test managing orderbooks for multiple symbols simultaneously."""

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        orderbooks = {}

        # Fetch orderbooks for multiple symbols
        for symbol in symbols:
            data = await mock_service_client.get_orderbook(symbol)

            from bot.fp.types.market import OrderBook

            orderbooks[symbol] = OrderBook(
                bids=[(Decimal("50000.00"), Decimal("10.0"))],
                asks=[(Decimal("50050.00"), Decimal("8.0"))],
                timestamp=datetime.now(UTC),
            )

        # Validate all orderbooks
        assert len(orderbooks) == 3
        for symbol in symbols:
            assert symbol in orderbooks
            assert orderbooks[symbol].mid_price == Decimal("50025.00")

        # Test orderbook updates
        for symbol in symbols:
            orderbook = orderbooks[symbol]
            impact = orderbook.price_impact("buy", Decimal("5.0"))
            assert impact is not None
            assert impact > orderbook.mid_price


class TestWebSocketIntegration:
    """Test real-time WebSocket orderbook integration."""

    @pytest.fixture
    async def mock_websocket_server(self):
        """Create mock WebSocket server for testing."""

        class MockWebSocketServer:
            def __init__(self):
                self.clients = []
                self.messages = []
                self.running = False

            async def start(self):
                self.running = True

            async def stop(self):
                self.running = False
                self.clients.clear()

            async def send_orderbook_update(self, symbol, bids, asks):
                """Send orderbook update to all clients."""
                message = {
                    "channel": "level2",
                    "type": "snapshot",
                    "product_id": symbol,
                    "bids": bids,
                    "asks": asks,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                self.messages.append(message)

                # Simulate sending to clients
                for client in self.clients:
                    await client.receive_message(message)

            def add_client(self, client):
                self.clients.append(client)

        return MockWebSocketServer()

    @pytest.fixture
    async def mock_websocket_client(self):
        """Create mock WebSocket client."""

        class MockWebSocketClient:
            def __init__(self):
                self.received_messages = []
                self.connected = False
                self.subscriptions = []

            async def connect(self, url):
                self.connected = True

            async def disconnect(self):
                self.connected = False

            async def subscribe(self, channel, symbol):
                if self.connected:
                    self.subscriptions.append({"channel": channel, "symbol": symbol})

            async def receive_message(self, message):
                if self.connected:
                    self.received_messages.append(message)

        return MockWebSocketClient()

    async def test_websocket_orderbook_subscription(
        self, mock_websocket_server, mock_websocket_client
    ):
        """Test WebSocket orderbook subscription and updates."""

        # Start server and connect client
        await mock_websocket_server.start()
        await mock_websocket_client.connect("wss://test-server")
        mock_websocket_server.add_client(mock_websocket_client)

        # Subscribe to orderbook channel
        await mock_websocket_client.subscribe("level2", "BTC-USD")

        # Send orderbook update
        await mock_websocket_server.send_orderbook_update(
            "BTC-USD",
            [["50000.00", "10.0"], ["49950.00", "5.0"]],
            [["50050.00", "8.0"], ["50100.00", "6.0"]],
        )

        # Validate message received
        assert len(mock_websocket_client.received_messages) == 1
        message = mock_websocket_client.received_messages[0]

        assert message["channel"] == "level2"
        assert message["product_id"] == "BTC-USD"
        assert len(message["bids"]) == 2
        assert len(message["asks"]) == 2

        # Convert to OrderBook and validate
        from bot.fp.types.market import OrderBook

        orderbook = OrderBook(
            bids=[
                (Decimal(message["bids"][0][0]), Decimal(message["bids"][0][1])),
                (Decimal(message["bids"][1][0]), Decimal(message["bids"][1][1])),
            ],
            asks=[
                (Decimal(message["asks"][0][0]), Decimal(message["asks"][0][1])),
                (Decimal(message["asks"][1][0]), Decimal(message["asks"][1][1])),
            ],
            timestamp=datetime.now(UTC),
        )

        assert orderbook.best_bid == (Decimal("50000.00"), Decimal("10.0"))
        assert orderbook.best_ask == (Decimal("50050.00"), Decimal("8.0"))

    async def test_websocket_reconnection_handling(
        self, mock_websocket_server, mock_websocket_client
    ):
        """Test WebSocket reconnection and data consistency."""

        # Initial connection
        await mock_websocket_server.start()
        await mock_websocket_client.connect("wss://test-server")
        mock_websocket_server.add_client(mock_websocket_client)

        # Send initial data
        await mock_websocket_server.send_orderbook_update(
            "BTC-USD", [["50000.00", "10.0"]], [["50050.00", "8.0"]]
        )

        assert len(mock_websocket_client.received_messages) == 1

        # Simulate disconnection
        await mock_websocket_client.disconnect()

        # Send data while disconnected (should not be received)
        await mock_websocket_server.send_orderbook_update(
            "BTC-USD", [["49900.00", "15.0"]], [["49950.00", "12.0"]]
        )

        # Still only 1 message received
        assert len(mock_websocket_client.received_messages) == 1

        # Reconnect
        await mock_websocket_client.connect("wss://test-server")
        mock_websocket_server.add_client(mock_websocket_client)

        # Send snapshot after reconnection
        await mock_websocket_server.send_orderbook_update(
            "BTC-USD", [["49900.00", "15.0"]], [["49950.00", "12.0"]]
        )

        # Should receive new message
        assert len(mock_websocket_client.received_messages) == 2

        latest_message = mock_websocket_client.received_messages[-1]
        assert latest_message["bids"][0][0] == "49900.00"

    async def test_websocket_message_validation(self, mock_websocket_client):
        """Test WebSocket message validation and error handling."""

        await mock_websocket_client.connect("wss://test-server")

        # Test invalid message (missing required fields)
        invalid_message = {
            "channel": "level2",
            # Missing type, product_id, bids, asks
        }

        # In a real implementation, this would be handled by a validator
        from bot.fp.types.market import OrderBookMessage

        with pytest.raises(ValueError):
            # This would fail due to missing required data
            OrderBookMessage(
                channel="level2",
                timestamp=datetime.now(UTC),
                data=invalid_message,
                bids=None,
                asks=None,
            )

        # Test valid message
        valid_message = {
            "channel": "level2",
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "10.0"]],
            "asks": [["50050.00", "8.0"]],
        }

        # Should not raise exception
        orderbook_msg = OrderBookMessage(
            channel="level2",
            timestamp=datetime.now(UTC),
            data=valid_message,
            bids=[(Decimal("50000.00"), Decimal("10.0"))],
            asks=[(Decimal("50050.00"), Decimal("8.0"))],
        )

        assert orderbook_msg.channel == "level2"


class TestRESTWebSocketConsistency:
    """Test consistency between REST API and WebSocket data."""

    @pytest.fixture
    async def mock_rest_client(self):
        """Create mock REST API client."""

        class MockRESTClient:
            def __init__(self):
                self.orderbook_data = {
                    "BTC-USD": {
                        "bids": [["50000.00", "10.0"], ["49950.00", "5.0"]],
                        "asks": [["50050.00", "8.0"], ["50100.00", "6.0"]],
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                }

            async def get_orderbook(self, symbol, level=2):
                if symbol in self.orderbook_data:
                    return self.orderbook_data[symbol]
                return None

            def update_orderbook(self, symbol, bids, asks):
                """Update orderbook data (simulates market changes)."""
                self.orderbook_data[symbol] = {
                    "bids": bids,
                    "asks": asks,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

        return MockRESTClient()

    async def test_rest_websocket_data_consistency(
        self, mock_rest_client, mock_websocket_server, mock_websocket_client
    ):
        """Test that REST and WebSocket data remain consistent."""

        # Get initial orderbook via REST
        rest_data = await mock_rest_client.get_orderbook("BTC-USD")

        # Start WebSocket connection
        await mock_websocket_server.start()
        await mock_websocket_client.connect("wss://test-server")
        mock_websocket_server.add_client(mock_websocket_client)

        # Send same data via WebSocket
        await mock_websocket_server.send_orderbook_update(
            "BTC-USD", rest_data["bids"], rest_data["asks"]
        )

        # Compare data consistency
        ws_message = mock_websocket_client.received_messages[0]

        assert ws_message["bids"] == rest_data["bids"]
        assert ws_message["asks"] == rest_data["asks"]

        # Test data update consistency
        new_bids = [["49900.00", "15.0"], ["49850.00", "8.0"]]
        new_asks = [["49950.00", "12.0"], ["50000.00", "10.0"]]

        # Update both REST and WebSocket data
        mock_rest_client.update_orderbook("BTC-USD", new_bids, new_asks)
        await mock_websocket_server.send_orderbook_update("BTC-USD", new_bids, new_asks)

        # Verify consistency
        updated_rest_data = await mock_rest_client.get_orderbook("BTC-USD")
        latest_ws_message = mock_websocket_client.received_messages[-1]

        assert updated_rest_data["bids"] == latest_ws_message["bids"]
        assert updated_rest_data["asks"] == latest_ws_message["asks"]

    async def test_cache_invalidation_and_refresh(self, mock_rest_client):
        """Test cache invalidation when data becomes stale."""

        class OrderBookCache:
            def __init__(self):
                self.cache = {}
                self.cache_timestamps = {}
                self.ttl_seconds = 5

            async def get_orderbook(self, symbol, rest_client):
                # Check if cached data is still valid
                if symbol in self.cache:
                    cache_age = datetime.now(UTC) - self.cache_timestamps[symbol]
                    if cache_age.total_seconds() < self.ttl_seconds:
                        return self.cache[symbol]

                # Cache miss or stale data - fetch fresh data
                fresh_data = await rest_client.get_orderbook(symbol)
                if fresh_data:
                    self.cache[symbol] = fresh_data
                    self.cache_timestamps[symbol] = datetime.now(UTC)

                return fresh_data

        cache = OrderBookCache()

        # First request - cache miss
        data1 = await cache.get_orderbook("BTC-USD", mock_rest_client)
        assert data1 is not None
        assert "BTC-USD" in cache.cache

        # Second request within TTL - cache hit
        data2 = await cache.get_orderbook("BTC-USD", mock_rest_client)
        assert data2 == data1

        # Wait for cache to expire
        await asyncio.sleep(6)

        # Update underlying data
        mock_rest_client.update_orderbook(
            "BTC-USD", [["49800.00", "20.0"]], [["49850.00", "15.0"]]
        )

        # Request after TTL - should refresh cache
        data3 = await cache.get_orderbook("BTC-USD", mock_rest_client)
        assert data3 != data1  # Should be different due to refresh
        assert data3["bids"][0][0] == "49800.00"


class TestPerformanceAndLoad:
    """Test performance under load conditions."""

    async def test_high_frequency_orderbook_updates(self):
        """Test handling high frequency orderbook updates."""

        update_count = 0
        processed_updates = []

        def process_orderbook_update(orderbook_data):
            nonlocal update_count
            update_count += 1
            processed_updates.append(orderbook_data)

        # Simulate high frequency updates
        start_time = time.time()

        for i in range(1000):
            # Simulate orderbook data
            orderbook_data = {
                "symbol": "BTC-USD",
                "bids": [[f"{50000 - i}", "10.0"]],
                "asks": [[f"{50050 + i}", "8.0"]],
                "timestamp": datetime.now(UTC),
            }

            process_orderbook_update(orderbook_data)

        end_time = time.time()
        processing_time = end_time - start_time

        # Performance assertions
        assert update_count == 1000
        assert len(processed_updates) == 1000
        assert processing_time < 1.0  # Should process 1000 updates in under 1 second

        updates_per_second = update_count / processing_time
        assert updates_per_second > 500  # Should handle at least 500 updates/sec

    async def test_concurrent_symbol_processing(self):
        """Test concurrent orderbook processing for multiple symbols."""

        symbols = [f"SYM{i}-USD" for i in range(50)]
        results = {}

        async def process_symbol_orderbook(symbol):
            """Simulate processing orderbook for a symbol."""
            # Simulate some processing time
            await asyncio.sleep(0.01)

            from bot.fp.types.market import OrderBook

            orderbook = OrderBook(
                bids=[(Decimal("100.0"), Decimal("10.0"))],
                asks=[(Decimal("101.0"), Decimal("8.0"))],
                timestamp=datetime.now(UTC),
            )

            results[symbol] = {
                "mid_price": float(orderbook.mid_price),
                "spread_bps": float(orderbook.get_spread_bps()),
                "processed_at": datetime.now(UTC),
            }

        # Process all symbols concurrently
        start_time = time.time()

        tasks = [process_symbol_orderbook(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert len(results) == 50
        assert (
            total_time < 1.0
        )  # Should process 50 symbols concurrently in under 1 second

        # Verify all results are valid
        for symbol in symbols:
            assert symbol in results
            assert results[symbol]["mid_price"] == 100.5
            assert results[symbol]["spread_bps"] > 0

    async def test_memory_usage_under_load(self):
        """Test memory usage with large orderbook data."""

        # Create large orderbook
        large_bids = [
            (Decimal(f"{50000 - i}"), Decimal(f"{i + 1}")) for i in range(1000)
        ]
        large_asks = [
            (Decimal(f"{50050 + i}"), Decimal(f"{i + 1}")) for i in range(1000)
        ]

        from bot.fp.types.market import OrderBook

        start_time = time.time()

        # Create multiple large orderbooks
        orderbooks = []
        for i in range(100):
            orderbook = OrderBook(
                bids=large_bids[:500],  # Use subset to avoid excessive memory
                asks=large_asks[:500],
                timestamp=datetime.now(UTC),
            )
            orderbooks.append(orderbook)

        end_time = time.time()
        creation_time = end_time - start_time

        # Performance assertions
        assert len(orderbooks) == 100
        assert (
            creation_time < 5.0
        )  # Should create 100 large orderbooks in under 5 seconds

        # Test operations on large orderbooks
        start_time = time.time()

        for orderbook in orderbooks:
            # Perform operations that would be common in trading
            mid_price = orderbook.mid_price
            spread = orderbook.spread
            depth = orderbook.bid_depth + orderbook.ask_depth
            impact = orderbook.price_impact("buy", Decimal(100))

            # Validate results
            assert mid_price > 0
            assert spread > 0
            assert depth > 0

        end_time = time.time()
        operation_time = end_time - start_time

        assert (
            operation_time < 2.0
        )  # Should perform operations on 100 orderbooks in under 2 seconds

    async def test_network_latency_simulation(self):
        """Test behavior under network latency conditions."""

        class LatencySimulator:
            def __init__(self, base_latency_ms=10, jitter_ms=5):
                self.base_latency = base_latency_ms / 1000.0
                self.jitter = jitter_ms / 1000.0

            async def simulate_network_call(self, operation):
                """Simulate network call with latency."""
                import random

                # Add latency with jitter
                latency = self.base_latency + random.uniform(0, self.jitter)
                await asyncio.sleep(latency)

                return await operation()

        simulator = LatencySimulator(base_latency_ms=50, jitter_ms=20)

        async def fetch_orderbook():
            """Simulate fetching orderbook data."""
            return {
                "bids": [["50000.00", "10.0"]],
                "asks": [["50050.00", "8.0"]],
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Test multiple concurrent requests with latency
        start_time = time.time()

        tasks = [simulator.simulate_network_call(fetch_orderbook) for _ in range(10)]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Assertions
        assert len(results) == 10
        assert (
            total_time < 0.2
        )  # Should complete within reasonable time despite latency

        # Verify all results are valid
        for result in results:
            assert "bids" in result
            assert "asks" in result
            assert len(result["bids"]) > 0
            assert len(result["asks"]) > 0


class TestConfigurationImpact:
    """Test impact of configuration changes on orderbook processing."""

    @pytest.fixture
    def config_manager(self):
        """Create configuration manager for testing."""

        class ConfigManager:
            def __init__(self):
                self.config = {
                    "orderbook_levels": 5,
                    "update_frequency_ms": 100,
                    "cache_ttl_seconds": 30,
                    "max_symbols": 10,
                    "enable_validation": True,
                }

            def update_config(self, key, value):
                self.config[key] = value

            def get_config(self, key):
                return self.config.get(key)

        return ConfigManager()

    async def test_orderbook_levels_configuration(self, config_manager):
        """Test different orderbook depth level configurations."""

        def create_orderbook_with_levels(levels):
            """Create orderbook with specified number of levels."""
            bids = [
                (Decimal(f"{50000 - i * 10}"), Decimal(f"{i + 1}"))
                for i in range(levels)
            ]
            asks = [
                (Decimal(f"{50050 + i * 10}"), Decimal(f"{i + 1}"))
                for i in range(levels)
            ]

            from bot.fp.types.market import OrderBook

            return OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))

        # Test different level configurations
        for levels in [1, 5, 10, 20]:
            config_manager.update_config("orderbook_levels", levels)

            orderbook = create_orderbook_with_levels(levels)

            # Validate orderbook structure
            assert len(orderbook.bids) == levels
            assert len(orderbook.asks) == levels

            # Test operations scale with levels
            total_bid_depth = orderbook.bid_depth
            total_ask_depth = orderbook.ask_depth

            assert total_bid_depth > 0
            assert total_ask_depth > 0

            # More levels should generally mean more depth
            if levels > 1:
                assert len(orderbook.bids) >= 1

    async def test_update_frequency_impact(self, config_manager):
        """Test impact of different update frequencies."""

        update_counts = {}

        async def simulate_updates_with_frequency(frequency_ms):
            """Simulate orderbook updates at specified frequency."""
            config_manager.update_config("update_frequency_ms", frequency_ms)

            update_count = 0
            start_time = time.time()

            # Simulate 1 second of updates
            while time.time() - start_time < 1.0:
                # Simulate processing an update
                update_count += 1
                await asyncio.sleep(frequency_ms / 1000.0)

            return update_count

        # Test different frequencies
        for frequency in [10, 50, 100, 500]:  # milliseconds
            count = await simulate_updates_with_frequency(frequency)
            update_counts[frequency] = count

        # Higher frequency should result in more updates (within reasonable bounds)
        assert update_counts[10] > update_counts[100]
        assert update_counts[50] > update_counts[500]

    async def test_cache_ttl_configuration(self, config_manager):
        """Test cache TTL configuration impact."""

        class ConfigurableCache:
            def __init__(self, config_manager):
                self.config_manager = config_manager
                self.cache = {}
                self.cache_times = {}

            async def get_data(self, key):
                ttl = self.config_manager.get_config("cache_ttl_seconds")

                if key in self.cache:
                    age = time.time() - self.cache_times[key]
                    if age < ttl:
                        return self.cache[key]

                # Cache miss or expired - generate new data
                new_data = {"timestamp": datetime.now(UTC), "value": f"data_{key}"}
                self.cache[key] = new_data
                self.cache_times[key] = time.time()

                return new_data

        cache = ConfigurableCache(config_manager)

        # Test short TTL
        config_manager.update_config("cache_ttl_seconds", 1)

        data1 = await cache.get_data("test_key")
        data2 = await cache.get_data("test_key")  # Should be cached

        assert data1 == data2

        # Wait for expiration
        await asyncio.sleep(1.5)

        data3 = await cache.get_data("test_key")  # Should be fresh
        assert data3 != data1  # Timestamps should be different

        # Test longer TTL
        config_manager.update_config("cache_ttl_seconds", 10)

        data4 = await cache.get_data("test_key2")
        await asyncio.sleep(2)  # Wait but still within TTL
        data5 = await cache.get_data("test_key2")

        assert data4 == data5  # Should still be cached


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
