#!/usr/bin/env python3
"""
Mock Exchange WebSocket Service for High-Frequency Orderbook Testing

This service provides a generic high-frequency WebSocket feed for orderbook data
to stress test the trading system's ability to handle rapid market data updates.

Features:
- High-frequency orderbook updates (configurable rate)
- Multiple symbol support
- Realistic price movement simulation
- WebSocket connection management
- Configurable market depth and volatility
"""

import asyncio
import json
import logging
import os
import random
import signal
from datetime import UTC, datetime
from typing import Any

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# =============================================================================
# CONFIGURATION
# =============================================================================


class MockConfig:
    """Configuration for mock exchange WebSocket service."""

    def __init__(self):
        self.port = int(os.getenv("WEBSOCKET_PORT", "8082"))
        self.update_frequency = float(
            os.getenv("UPDATE_FREQUENCY", "100")
        )  # Updates per second
        self.orderbook_levels = int(os.getenv("ORDERBOOK_LEVELS", "20"))
        self.symbols = os.getenv("SYMBOLS", "BTC-USD,ETH-USD,SOL-USD").split(",")
        self.price_volatility = float(os.getenv("PRICE_VOLATILITY", "0.015"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()


# =============================================================================
# MARKET DATA SIMULATOR
# =============================================================================


class MarketDataSimulator:
    """Simulates realistic market data with configurable parameters."""

    def __init__(self, config: MockConfig):
        self.config = config
        self.sequence_number = 0

        # Base market parameters for each symbol
        self.market_params = {
            "BTC-USD": {
                "base_price": 50000.0,
                "tick_size": 0.01,
                "min_size": 0.00001,
                "volatility": 0.02,
                "spread_bps": 10,
            },
            "ETH-USD": {
                "base_price": 3000.0,
                "tick_size": 0.01,
                "min_size": 0.001,
                "volatility": 0.025,
                "spread_bps": 15,
            },
            "SOL-USD": {
                "base_price": 100.0,
                "tick_size": 0.01,
                "min_size": 0.01,
                "volatility": 0.03,
                "spread_bps": 20,
            },
        }

        # Current market state
        self.current_prices = {
            symbol: params["base_price"]
            for symbol, params in self.market_params.items()
            if symbol in self.config.symbols
        }

        # Order book state
        self.orderbooks = {}
        self._initialize_orderbooks()

    def _initialize_orderbooks(self):
        """Initialize orderbooks for all symbols."""
        for symbol in self.config.symbols:
            if symbol in self.market_params:
                self.orderbooks[symbol] = self._generate_initial_orderbook(symbol)

    def _generate_initial_orderbook(self, symbol: str) -> dict[str, Any]:
        """Generate initial orderbook for a symbol."""
        params = self.market_params[symbol]
        mid_price = self.current_prices[symbol]
        tick_size = params["tick_size"]
        spread_bps = params["spread_bps"]

        # Calculate spread
        spread = mid_price * (spread_bps / 10000)  # Convert bps to decimal
        half_spread = spread / 2

        # Generate bids (descending prices)
        bids = []
        for i in range(self.config.orderbook_levels):
            price = mid_price - half_spread - (i * tick_size * random.uniform(1, 3))
            price = round(price / tick_size) * tick_size  # Round to tick size
            size = random.uniform(0.1, 10.0)
            bids.append([price, size])

        # Generate asks (ascending prices)
        asks = []
        for i in range(self.config.orderbook_levels):
            price = mid_price + half_spread + (i * tick_size * random.uniform(1, 3))
            price = round(price / tick_size) * tick_size  # Round to tick size
            size = random.uniform(0.1, 10.0)
            asks.append([price, size])

        return {
            "symbol": symbol,
            "bids": sorted(bids, key=lambda x: x[0], reverse=True),  # Descending
            "asks": sorted(asks, key=lambda x: x[0]),  # Ascending
            "timestamp": datetime.now(UTC).isoformat(),
            "sequence": self.get_sequence(),
        }

    def get_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_number += 1
        return self.sequence_number

    def update_price(self, symbol: str) -> float:
        """Update price for a symbol using random walk."""
        if symbol not in self.market_params:
            return self.current_prices.get(symbol, 0.0)

        params = self.market_params[symbol]
        current_price = self.current_prices[symbol]

        # Generate price change using geometric Brownian motion
        dt = 1.0 / self.config.update_frequency  # Time step
        drift = 0.0  # No drift for testing
        volatility = params["volatility"] * self.config.price_volatility

        # Random shock
        shock = np.random.normal(0, 1)

        # Price change
        price_change = current_price * (drift * dt + volatility * shock * np.sqrt(dt))
        new_price = current_price + price_change

        # Ensure price doesn't go negative
        new_price = max(new_price, params["tick_size"])

        self.current_prices[symbol] = new_price
        return new_price

    def generate_orderbook_update(self, symbol: str) -> dict[str, Any]:
        """Generate an orderbook update for a symbol."""
        if symbol not in self.orderbooks:
            return self._generate_initial_orderbook(symbol)

        # Update price
        new_price = self.update_price(symbol)
        params = self.market_params[symbol]
        tick_size = params["tick_size"]

        # Get current orderbook
        orderbook = self.orderbooks[symbol]

        # Decide on update type
        update_type = random.choices(
            ["price_update", "size_update", "level_change"], weights=[0.5, 0.3, 0.2]
        )[0]

        updates = []

        if update_type == "price_update":
            # Shift all levels based on new mid price
            spread_bps = params["spread_bps"]
            spread = new_price * (spread_bps / 10000)
            half_spread = spread / 2

            # Update a few bid levels
            for i in range(min(3, len(orderbook["bids"]))):
                old_price, old_size = orderbook["bids"][i]
                new_level_price = (
                    new_price - half_spread - (i * tick_size * random.uniform(1, 2))
                )
                new_level_price = round(new_level_price / tick_size) * tick_size

                orderbook["bids"][i] = [new_level_price, old_size]
                updates.append(
                    {
                        "side": "bid",
                        "price": str(new_level_price),
                        "size": str(old_size),
                        "type": "update",
                    }
                )

            # Update a few ask levels
            for i in range(min(3, len(orderbook["asks"]))):
                old_price, old_size = orderbook["asks"][i]
                new_level_price = (
                    new_price + half_spread + (i * tick_size * random.uniform(1, 2))
                )
                new_level_price = round(new_level_price / tick_size) * tick_size

                orderbook["asks"][i] = [new_level_price, old_size]
                updates.append(
                    {
                        "side": "ask",
                        "price": str(new_level_price),
                        "size": str(old_size),
                        "type": "update",
                    }
                )

        elif update_type == "size_update":
            # Update sizes at existing price levels
            side = random.choice(["bids", "asks"])
            side_name = "bid" if side == "bids" else "ask"

            if orderbook[side]:
                level_idx = random.randint(0, min(4, len(orderbook[side]) - 1))
                price, old_size = orderbook[side][level_idx]

                # Change size (could be increase, decrease, or remove)
                size_change = random.uniform(-0.5, 1.0)
                new_size = max(0, old_size + size_change)

                if new_size > 0:
                    orderbook[side][level_idx] = [price, new_size]
                    updates.append(
                        {
                            "side": side_name,
                            "price": str(price),
                            "size": str(new_size),
                            "type": "update",
                        }
                    )
                else:
                    # Remove level
                    orderbook[side].pop(level_idx)
                    updates.append(
                        {
                            "side": side_name,
                            "price": str(price),
                            "size": "0",
                            "type": "remove",
                        }
                    )

        elif update_type == "level_change":
            # Add new levels or remove existing ones
            side = random.choice(["bids", "asks"])
            side_name = "bid" if side == "bids" else "ask"

            if random.random() < 0.7:  # 70% chance to add level
                # Add new level
                if side == "bids":
                    # Add bid below current best
                    if orderbook["bids"]:
                        best_bid = orderbook["bids"][0][0]
                        new_price = best_bid - tick_size * random.randint(1, 5)
                    else:
                        new_price = new_price - new_price * 0.001
                # Add ask above current best
                elif orderbook["asks"]:
                    best_ask = orderbook["asks"][0][0]
                    new_price = best_ask + tick_size * random.randint(1, 5)
                else:
                    new_price = new_price + new_price * 0.001

                new_size = random.uniform(0.1, 5.0)
                new_level = [new_price, new_size]

                orderbook[side].append(new_level)

                # Re-sort
                if side == "bids":
                    orderbook[side].sort(key=lambda x: x[0], reverse=True)
                else:
                    orderbook[side].sort(key=lambda x: x[0])

                # Limit depth
                orderbook[side] = orderbook[side][: self.config.orderbook_levels]

                updates.append(
                    {
                        "side": side_name,
                        "price": str(new_price),
                        "size": str(new_size),
                        "type": "add",
                    }
                )

            # Remove level
            elif len(orderbook[side]) > 1:  # Keep at least one level
                level_idx = random.randint(
                    1, len(orderbook[side]) - 1
                )  # Don't remove best level
                price, size = orderbook[side].pop(level_idx)

                updates.append(
                    {
                        "side": side_name,
                        "price": str(price),
                        "size": "0",
                        "type": "remove",
                    }
                )

        # Update timestamp and sequence
        orderbook["timestamp"] = datetime.now(UTC).isoformat()
        orderbook["sequence"] = self.get_sequence()

        return {
            "type": "orderbook_update",
            "symbol": symbol,
            "updates": updates,
            "timestamp": orderbook["timestamp"],
            "sequence": orderbook["sequence"],
        }

    def get_orderbook_snapshot(self, symbol: str) -> dict[str, Any]:
        """Get current orderbook snapshot for a symbol."""
        if symbol not in self.orderbooks:
            return self._generate_initial_orderbook(symbol)

        orderbook = self.orderbooks[symbol]

        return {
            "type": "orderbook_snapshot",
            "symbol": symbol,
            "bids": [[str(price), str(size)] for price, size in orderbook["bids"]],
            "asks": [[str(price), str(size)] for price, size in orderbook["asks"]],
            "timestamp": orderbook["timestamp"],
            "sequence": orderbook["sequence"],
        }


# =============================================================================
# WEBSOCKET SERVER
# =============================================================================


class MockExchangeWebSocketServer:
    """High-frequency WebSocket server for orderbook data."""

    def __init__(self):
        self.config = MockConfig()
        self.simulator = MarketDataSimulator(self.config)
        self.connections: set[WebSocketServerProtocol] = set()
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("mock-exchange-ws")

        # Background tasks
        self.broadcast_task = None

    async def register_connection(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket connection."""
        self.connections.add(websocket)
        self.logger.info(
            f"New connection registered. Total connections: {len(self.connections)}"
        )

        # Send initial snapshots for all symbols
        for symbol in self.config.symbols:
            if symbol in self.simulator.market_params:
                snapshot = self.simulator.get_orderbook_snapshot(symbol)
                try:
                    await websocket.send(json.dumps(snapshot))
                except websockets.exceptions.ConnectionClosed:
                    break

    async def unregister_connection(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket connection."""
        self.connections.discard(websocket)
        self.logger.info(
            f"Connection unregistered. Total connections: {len(self.connections)}"
        )

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket connection."""
        await self.register_connection(websocket)

        try:
            # Keep connection alive
            async for message in websocket:
                # Echo back any messages (for ping/pong)
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        pong = {
                            "type": "pong",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        await websocket.send(json.dumps(pong))
                except json.JSONDecodeError:
                    # Ignore invalid JSON
                    pass

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection closed by client")
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            await self.unregister_connection(websocket)

    async def broadcast_updates(self):
        """Continuously broadcast orderbook updates to all connections."""
        self.logger.info(
            f"Starting broadcast updates at {self.config.update_frequency} Hz"
        )

        while self.running:
            try:
                if self.connections:
                    # Generate updates for random symbols
                    symbol = random.choice(self.config.symbols)
                    if symbol in self.simulator.market_params:
                        update = self.simulator.generate_orderbook_update(symbol)

                        # Broadcast to all connections
                        message = json.dumps(update)
                        disconnected = set()

                        for websocket in self.connections.copy():
                            try:
                                await websocket.send(message)
                            except websockets.exceptions.ConnectionClosed:
                                disconnected.add(websocket)
                            except Exception as e:
                                self.logger.warning(f"Error sending to connection: {e}")
                                disconnected.add(websocket)

                        # Remove disconnected connections
                        for websocket in disconnected:
                            self.connections.discard(websocket)

                # Wait for next update
                await asyncio.sleep(1.0 / self.config.update_frequency)

            except Exception as e:
                self.logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(1.0)

    async def start_server(self):
        """Start the WebSocket server."""
        self.running = True

        # Start broadcast task
        self.broadcast_task = asyncio.create_task(self.broadcast_updates())

        # Start WebSocket server
        self.logger.info(f"Starting WebSocket server on port {self.config.port}")

        async with websockets.serve(
            self.handle_connection,
            "0.0.0.0",
            self.config.port,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
        ):
            self.logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever

    async def stop_server(self):
        """Stop the WebSocket server."""
        self.logger.info("Stopping WebSocket server")
        self.running = False

        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        if self.connections:
            self.logger.info(f"Closing {len(self.connections)} connections")
            await asyncio.gather(
                *[websocket.close() for websocket in self.connections],
                return_exceptions=True,
            )

        self.logger.info("WebSocket server stopped")


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================


async def main():
    """Main application entry point."""
    server = MockExchangeWebSocketServer()

    # Setup signal handlers
    def signal_handler(signum, frame):
        server.logger.info(f"Received signal {signum}")
        asyncio.create_task(server.stop_server())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await server.start_server()
    except KeyboardInterrupt:
        server.logger.info("Received keyboard interrupt")
    except Exception as e:
        server.logger.error(f"Server error: {e}")
    finally:
        await server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
