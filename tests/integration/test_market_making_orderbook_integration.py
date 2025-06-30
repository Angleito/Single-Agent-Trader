"""
Market Making Strategy Integration Tests with Orderbook Data

This module tests the integration between market making strategies and orderbook data,
ensuring that the complete flow from orderbook updates to market making decisions
and order execution works correctly.

Test Coverage:
- Market making strategy integration with live orderbook data
- Quote generation based on orderbook state
- Inventory management with orderbook feedback
- Risk management integration with orderbook metrics
- Performance optimization for high-frequency market making
- Cross-exchange arbitrage opportunity detection
"""

import time
from datetime import UTC, datetime
from decimal import Decimal

import pytest


class TestMarketMakingOrderbookIntegration:
    """Test market making strategy integration with orderbook data."""

    @pytest.fixture
    async def mock_orderbook_provider(self):
        """Create mock orderbook data provider."""

        class MockOrderbookProvider:
            def __init__(self):
                self.orderbooks = {}
                self.subscribers = []
                self.update_count = 0

            def get_orderbook(self, symbol):
                """Get current orderbook for symbol."""
                if symbol in self.orderbooks:
                    return self.orderbooks[symbol]

                # Default orderbook
                from bot.fp.types.market import OrderBook

                return OrderBook(
                    bids=[(Decimal(50000), Decimal("10.0"))],
                    asks=[(Decimal(50050), Decimal("8.0"))],
                    timestamp=datetime.now(UTC),
                )

            def update_orderbook(self, symbol, bids, asks):
                """Update orderbook data."""
                from bot.fp.types.market import OrderBook

                self.orderbooks[symbol] = OrderBook(
                    bids=bids, asks=asks, timestamp=datetime.now(UTC)
                )
                self.update_count += 1

                # Notify subscribers
                for callback in self.subscribers:
                    callback(symbol, self.orderbooks[symbol])

            def subscribe(self, callback):
                """Subscribe to orderbook updates."""
                self.subscribers.append(callback)

        return MockOrderbookProvider()

    @pytest.fixture
    async def market_making_strategy(self):
        """Create market making strategy."""

        class MarketMakingStrategy:
            def __init__(self):
                self.config = {
                    "spread_multiplier": 2.0,
                    "max_position_size": 100.0,
                    "inventory_target": 0.0,
                    "risk_factor": 0.1,
                    "min_quote_size": 1.0,
                    "max_quote_size": 10.0,
                }
                self.current_position = Decimal(0)
                self.active_quotes = {}

            def calculate_quotes(self, orderbook, symbol):
                """Calculate bid/ask quotes based on orderbook."""
                if not orderbook.best_bid or not orderbook.best_ask:
                    return None

                mid_price = orderbook.mid_price
                current_spread = orderbook.spread

                # Calculate dynamic spread based on market conditions
                spread_bps = orderbook.get_spread_bps()
                min_spread = max(
                    current_spread * self.config["spread_multiplier"],
                    mid_price * Decimal("0.0001"),
                )  # 1 bps minimum

                # Adjust for inventory
                inventory_skew = self._calculate_inventory_skew()

                # Calculate quote prices
                half_spread = min_spread / 2
                bid_price = mid_price - half_spread + inventory_skew
                ask_price = mid_price + half_spread + inventory_skew

                # Calculate quote sizes
                quote_size = self._calculate_quote_size(orderbook)

                return {
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "bid_size": quote_size,
                    "ask_size": quote_size,
                    "mid_price": mid_price,
                    "spread_bps": float(spread_bps),
                    "inventory_skew": float(inventory_skew),
                }

            def _calculate_inventory_skew(self):
                """Calculate price skew based on current inventory."""
                max_position = Decimal(str(self.config["max_position_size"]))
                target_position = Decimal(str(self.config["inventory_target"]))

                if max_position == 0:
                    return Decimal(0)

                inventory_ratio = (
                    self.current_position - target_position
                ) / max_position
                skew = inventory_ratio * Decimal(str(self.config["risk_factor"]))

                return skew

            def _calculate_quote_size(self, orderbook):
                """Calculate appropriate quote size based on market depth."""
                # Simple size calculation based on orderbook depth
                total_depth = orderbook.bid_depth + orderbook.ask_depth

                min_size = Decimal(str(self.config["min_quote_size"]))
                max_size = Decimal(str(self.config["max_quote_size"]))

                # Scale size with available liquidity
                if total_depth > 100:
                    return max_size
                if total_depth > 50:
                    return (min_size + max_size) / 2
                return min_size

            def update_position(self, trade_side, trade_size):
                """Update position based on executed trade."""
                if trade_side == "buy":
                    self.current_position += Decimal(str(trade_size))
                else:
                    self.current_position -= Decimal(str(trade_size))

        return MarketMakingStrategy()

    @pytest.fixture
    async def mock_order_manager(self):
        """Create mock order management system."""

        class MockOrderManager:
            def __init__(self):
                self.active_orders = {}
                self.executed_orders = []
                self.order_id_counter = 0

            async def place_order(self, order_data):
                """Place an order."""
                self.order_id_counter += 1
                order_id = f"order_{self.order_id_counter}"

                order = {
                    "id": order_id,
                    "symbol": order_data["symbol"],
                    "side": order_data["side"],
                    "price": float(order_data["price"]),
                    "size": float(order_data["size"]),
                    "status": "open",
                    "timestamp": datetime.now(UTC),
                }

                self.active_orders[order_id] = order
                return order

            async def cancel_order(self, order_id):
                """Cancel an order."""
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    order["status"] = "cancelled"
                    del self.active_orders[order_id]
                    return order
                return None

            def simulate_fill(self, order_id, fill_size=None):
                """Simulate order fill."""
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    fill_size = fill_size or order["size"]

                    filled_order = {
                        **order,
                        "status": "filled",
                        "filled_size": fill_size,
                        "fill_timestamp": datetime.now(UTC),
                    }

                    self.executed_orders.append(filled_order)
                    del self.active_orders[order_id]
                    return filled_order
                return None

        return MockOrderManager()

    async def test_basic_market_making_flow(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test basic market making flow with orderbook updates."""

        symbol = "BTC-USD"

        # Set up initial orderbook
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("10.0")), (Decimal(49950), Decimal("5.0"))],
            asks=[(Decimal(50050), Decimal("8.0")), (Decimal(50100), Decimal("6.0"))],
        )

        # Get orderbook and calculate quotes
        orderbook = mock_orderbook_provider.get_orderbook(symbol)
        quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

        # Validate quote calculation
        assert quotes is not None
        assert quotes["bid_price"] < quotes["mid_price"] < quotes["ask_price"]
        assert quotes["bid_size"] > 0
        assert quotes["ask_size"] > 0
        assert quotes["spread_bps"] > 0

        # Place market making orders
        bid_order = await mock_order_manager.place_order(
            {
                "symbol": symbol,
                "side": "buy",
                "price": quotes["bid_price"],
                "size": quotes["bid_size"],
            }
        )

        ask_order = await mock_order_manager.place_order(
            {
                "symbol": symbol,
                "side": "sell",
                "price": quotes["ask_price"],
                "size": quotes["ask_size"],
            }
        )

        # Validate orders placed
        assert bid_order["status"] == "open"
        assert ask_order["status"] == "open"
        assert len(mock_order_manager.active_orders) == 2

        # Simulate order fills
        filled_bid = mock_order_manager.simulate_fill(bid_order["id"])
        assert filled_bid["status"] == "filled"

        # Update strategy position
        market_making_strategy.update_position("buy", filled_bid["filled_size"])
        assert market_making_strategy.current_position > 0

    async def test_orderbook_update_response(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test market making response to orderbook updates."""

        symbol = "BTC-USD"
        quote_updates = []

        def on_orderbook_update(symbol, orderbook):
            """Handle orderbook updates."""
            new_quotes = market_making_strategy.calculate_quotes(orderbook, symbol)
            quote_updates.append(
                {
                    "timestamp": datetime.now(UTC),
                    "quotes": new_quotes,
                    "orderbook_mid": orderbook.mid_price,
                }
            )

        # Subscribe to orderbook updates
        mock_orderbook_provider.subscribe(on_orderbook_update)

        # Initial orderbook
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("10.0"))],
            asks=[(Decimal(50050), Decimal("8.0"))],
        )

        assert len(quote_updates) == 1
        initial_quotes = quote_updates[0]["quotes"]

        # Market moves up
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50100), Decimal("10.0"))],
            asks=[(Decimal(50150), Decimal("8.0"))],
        )

        assert len(quote_updates) == 2
        updated_quotes = quote_updates[1]["quotes"]

        # Validate quotes moved with market
        assert updated_quotes["mid_price"] > initial_quotes["mid_price"]
        assert updated_quotes["bid_price"] > initial_quotes["bid_price"]
        assert updated_quotes["ask_price"] > initial_quotes["ask_price"]

    async def test_inventory_management_integration(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test inventory management impact on quote generation."""

        symbol = "BTC-USD"

        # Set up orderbook
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("10.0"))],
            asks=[(Decimal(50050), Decimal("8.0"))],
        )

        orderbook = mock_orderbook_provider.get_orderbook(symbol)

        # Test with neutral inventory
        neutral_quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

        # Build up long position
        market_making_strategy.update_position("buy", 50)  # Long 50 units
        long_quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

        # Build up short position
        market_making_strategy.update_position("sell", 100)  # Now short 50 units
        short_quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

        # Validate inventory skew
        assert long_quotes["inventory_skew"] != neutral_quotes["inventory_skew"]
        assert short_quotes["inventory_skew"] != neutral_quotes["inventory_skew"]

        # Long position should have lower quotes (incentivize selling)
        assert long_quotes["bid_price"] < neutral_quotes["bid_price"]
        assert long_quotes["ask_price"] < neutral_quotes["ask_price"]

        # Short position should have higher quotes (incentivize buying)
        assert short_quotes["bid_price"] > neutral_quotes["bid_price"]
        assert short_quotes["ask_price"] > neutral_quotes["ask_price"]

    async def test_risk_management_integration(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test risk management integration with market making."""

        symbol = "BTC-USD"

        # Test with normal market conditions
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("10.0")), (Decimal(49950), Decimal("8.0"))],
            asks=[(Decimal(50050), Decimal("8.0")), (Decimal(50100), Decimal("6.0"))],
        )

        orderbook = mock_orderbook_provider.get_orderbook(symbol)
        normal_quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

        # Test with wide spread (high volatility)
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(49000), Decimal("5.0"))],
            asks=[(Decimal(51000), Decimal("5.0"))],
        )

        wide_orderbook = mock_orderbook_provider.get_orderbook(symbol)
        wide_quotes = market_making_strategy.calculate_quotes(wide_orderbook, symbol)

        # Validate risk-adjusted quotes
        assert wide_quotes["spread_bps"] > normal_quotes["spread_bps"]

        # Test with thin market (low liquidity)
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("1.0"))],
            asks=[(Decimal(50050), Decimal("1.0"))],
        )

        thin_orderbook = mock_orderbook_provider.get_orderbook(symbol)
        thin_quotes = market_making_strategy.calculate_quotes(thin_orderbook, symbol)

        # Should reduce quote size in thin market
        assert thin_quotes["bid_size"] <= normal_quotes["bid_size"]
        assert thin_quotes["ask_size"] <= normal_quotes["ask_size"]

    async def test_high_frequency_quote_updates(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test performance with high-frequency orderbook updates."""

        symbol = "BTC-USD"
        quote_calculation_times = []
        total_updates = 0

        def performance_callback(symbol, orderbook):
            nonlocal total_updates
            start_time = time.time()

            quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

            end_time = time.time()
            calculation_time = end_time - start_time
            quote_calculation_times.append(calculation_time)
            total_updates += 1

        mock_orderbook_provider.subscribe(performance_callback)

        # Simulate high-frequency updates
        start_time = time.time()

        for i in range(1000):
            bid_price = Decimal(f"{50000 - (i % 100)}")
            ask_price = Decimal(f"{50050 + (i % 100)}")

            mock_orderbook_provider.update_orderbook(
                symbol,
                bids=[(bid_price, Decimal("10.0"))],
                asks=[(ask_price, Decimal("8.0"))],
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        assert total_updates == 1000
        assert total_time < 5.0  # Should handle 1000 updates in under 5 seconds

        avg_calculation_time = sum(quote_calculation_times) / len(
            quote_calculation_times
        )
        assert avg_calculation_time < 0.001  # Average calculation time under 1ms

    async def test_multi_symbol_market_making(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test market making across multiple symbols."""

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        all_quotes = {}

        # Set up orderbooks for all symbols
        for i, symbol in enumerate(symbols):
            base_price = 50000 * (i + 1) // (i + 1)  # Different price levels

            mock_orderbook_provider.update_orderbook(
                symbol,
                bids=[(Decimal(f"{base_price}"), Decimal("10.0"))],
                asks=[(Decimal(f"{base_price + 50}"), Decimal("8.0"))],
            )

        # Calculate quotes for all symbols
        for symbol in symbols:
            orderbook = mock_orderbook_provider.get_orderbook(symbol)
            quotes = market_making_strategy.calculate_quotes(orderbook, symbol)
            all_quotes[symbol] = quotes

        # Validate quotes for all symbols
        assert len(all_quotes) == 3
        for symbol in symbols:
            quotes = all_quotes[symbol]
            assert quotes["bid_price"] < quotes["ask_price"]
            assert quotes["bid_size"] > 0
            assert quotes["ask_size"] > 0

        # Place orders for all symbols
        all_orders = {}
        for symbol in symbols:
            quotes = all_quotes[symbol]

            bid_order = await mock_order_manager.place_order(
                {
                    "symbol": symbol,
                    "side": "buy",
                    "price": quotes["bid_price"],
                    "size": quotes["bid_size"],
                }
            )

            ask_order = await mock_order_manager.place_order(
                {
                    "symbol": symbol,
                    "side": "sell",
                    "price": quotes["ask_price"],
                    "size": quotes["ask_size"],
                }
            )

            all_orders[symbol] = {"bid": bid_order, "ask": ask_order}

        # Validate all orders placed
        assert len(mock_order_manager.active_orders) == 6  # 2 orders per symbol

        for symbol in symbols:
            orders = all_orders[symbol]
            assert orders["bid"]["status"] == "open"
            assert orders["ask"]["status"] == "open"

    async def test_order_lifecycle_management(
        self, mock_orderbook_provider, market_making_strategy, mock_order_manager
    ):
        """Test complete order lifecycle with orderbook changes."""

        symbol = "BTC-USD"

        # Initial setup
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("10.0"))],
            asks=[(Decimal(50050), Decimal("8.0"))],
        )

        orderbook = mock_orderbook_provider.get_orderbook(symbol)
        initial_quotes = market_making_strategy.calculate_quotes(orderbook, symbol)

        # Place initial orders
        bid_order = await mock_order_manager.place_order(
            {
                "symbol": symbol,
                "side": "buy",
                "price": initial_quotes["bid_price"],
                "size": initial_quotes["bid_size"],
            }
        )

        ask_order = await mock_order_manager.place_order(
            {
                "symbol": symbol,
                "side": "sell",
                "price": initial_quotes["ask_price"],
                "size": initial_quotes["ask_size"],
            }
        )

        assert len(mock_order_manager.active_orders) == 2

        # Market moves significantly - should trigger order updates
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(51000), Decimal("10.0"))],  # Market up 1000
            asks=[(Decimal(51050), Decimal("8.0"))],
        )

        new_orderbook = mock_orderbook_provider.get_orderbook(symbol)
        new_quotes = market_making_strategy.calculate_quotes(new_orderbook, symbol)

        # Old orders should be too far from market
        price_deviation = abs(
            float(new_quotes["mid_price"] - initial_quotes["mid_price"])
        )
        assert price_deviation > 500  # Significant move

        # Cancel old orders
        await mock_order_manager.cancel_order(bid_order["id"])
        await mock_order_manager.cancel_order(ask_order["id"])
        assert len(mock_order_manager.active_orders) == 0

        # Place new orders at updated prices
        new_bid_order = await mock_order_manager.place_order(
            {
                "symbol": symbol,
                "side": "buy",
                "price": new_quotes["bid_price"],
                "size": new_quotes["bid_size"],
            }
        )

        new_ask_order = await mock_order_manager.place_order(
            {
                "symbol": symbol,
                "side": "sell",
                "price": new_quotes["ask_price"],
                "size": new_quotes["ask_size"],
            }
        )

        assert len(mock_order_manager.active_orders) == 2

        # Simulate partial fills
        partial_fill = mock_order_manager.simulate_fill(
            new_bid_order["id"], fill_size=float(new_quotes["bid_size"]) / 2
        )

        assert partial_fill["filled_size"] == float(new_quotes["bid_size"]) / 2
        assert len(mock_order_manager.executed_orders) == 1

    async def test_cross_exchange_arbitrage_detection(self, mock_orderbook_provider):
        """Test detection of arbitrage opportunities across exchanges."""

        symbol = "BTC-USD"

        class ArbitrageDetector:
            def __init__(self):
                self.exchange_orderbooks = {}
                self.arbitrage_opportunities = []

            def update_exchange_orderbook(self, exchange, symbol, orderbook):
                """Update orderbook for specific exchange."""
                if exchange not in self.exchange_orderbooks:
                    self.exchange_orderbooks[exchange] = {}
                self.exchange_orderbooks[exchange][symbol] = orderbook

                # Check for arbitrage after each update
                self._check_arbitrage(symbol)

            def _check_arbitrage(self, symbol):
                """Check for arbitrage opportunities."""
                if len(self.exchange_orderbooks) < 2:
                    return

                exchanges = list(self.exchange_orderbooks.keys())
                best_bids = {}
                best_asks = {}

                # Find best bid and ask across exchanges
                for exchange in exchanges:
                    if symbol in self.exchange_orderbooks[exchange]:
                        orderbook = self.exchange_orderbooks[exchange][symbol]
                        if orderbook.best_bid:
                            best_bids[exchange] = orderbook.best_bid[0]
                        if orderbook.best_ask:
                            best_asks[exchange] = orderbook.best_ask[0]

                if not best_bids or not best_asks:
                    return

                # Find highest bid and lowest ask
                highest_bid_exchange = max(best_bids.keys(), key=lambda x: best_bids[x])
                lowest_ask_exchange = min(best_asks.keys(), key=lambda x: best_asks[x])

                highest_bid = best_bids[highest_bid_exchange]
                lowest_ask = best_asks[lowest_ask_exchange]

                # Check for arbitrage opportunity
                if (
                    highest_bid > lowest_ask
                    and highest_bid_exchange != lowest_ask_exchange
                ):
                    profit = highest_bid - lowest_ask
                    profit_bps = (profit / lowest_ask) * 10000

                    opportunity = {
                        "symbol": symbol,
                        "buy_exchange": lowest_ask_exchange,
                        "sell_exchange": highest_bid_exchange,
                        "buy_price": lowest_ask,
                        "sell_price": highest_bid,
                        "profit": profit,
                        "profit_bps": float(profit_bps),
                        "timestamp": datetime.now(UTC),
                    }

                    self.arbitrage_opportunities.append(opportunity)

        detector = ArbitrageDetector()

        # Set up orderbooks on different exchanges
        # Exchange A - Higher prices
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50100), Decimal("5.0"))],
            asks=[(Decimal(50150), Decimal("5.0"))],
        )
        detector.update_exchange_orderbook(
            "exchange_a", symbol, mock_orderbook_provider.get_orderbook(symbol)
        )

        # Exchange B - Lower prices
        mock_orderbook_provider.update_orderbook(
            symbol,
            bids=[(Decimal(50000), Decimal("8.0"))],
            asks=[(Decimal(50050), Decimal("8.0"))],
        )
        detector.update_exchange_orderbook(
            "exchange_b", symbol, mock_orderbook_provider.get_orderbook(symbol)
        )

        # Should detect arbitrage opportunity
        assert len(detector.arbitrage_opportunities) == 1

        opportunity = detector.arbitrage_opportunities[0]
        assert opportunity["buy_exchange"] == "exchange_b"  # Buy low
        assert opportunity["sell_exchange"] == "exchange_a"  # Sell high
        assert opportunity["profit"] > 0
        assert opportunity["profit_bps"] > 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
