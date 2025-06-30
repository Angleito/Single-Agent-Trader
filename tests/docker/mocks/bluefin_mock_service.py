#!/usr/bin/env python3
"""
Mock Bluefin DEX Service for Orderbook Testing

This service simulates the Bluefin DEX API and WebSocket feeds to provide
realistic orderbook data for testing without requiring actual blockchain connections.

Features:
- RESTful API endpoints matching Bluefin interface
- Real-time WebSocket orderbook feeds
- Configurable market data simulation
- Error simulation for fault tolerance testing
- Performance metrics and monitoring
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import UTC, datetime
from decimal import Decimal

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================


class MockConfig:
    """Configuration for mock Bluefin service."""

    def __init__(self):
        self.port = int(os.getenv("MOCK_SERVICE_PORT", "8080"))
        self.service_name = os.getenv("MOCK_SERVICE_NAME", "bluefin")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.simulate_latency = os.getenv("SIMULATE_LATENCY", "true").lower() == "true"
        self.simulate_errors = os.getenv("SIMULATE_ERRORS", "false").lower() == "true"
        self.orderbook_depth = int(os.getenv("ORDERBOOK_DEPTH", "100"))
        self.price_volatility = float(os.getenv("PRICE_VOLATILITY", "0.02"))
        self.update_frequency = float(
            os.getenv("UPDATE_FREQUENCY", "10.0")
        )  # Updates per second


# =============================================================================
# DATA MODELS
# =============================================================================


class OrderBookLevel(BaseModel):
    """Represents a single orderbook level."""

    price: Decimal
    size: Decimal


class OrderBookSnapshot(BaseModel):
    """Represents a complete orderbook snapshot."""

    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime
    sequence: int


class MarketTicker(BaseModel):
    """Represents market ticker data."""

    symbol: str
    price: Decimal
    volume_24h: Decimal
    change_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime


class OrderRequest(BaseModel):
    """Represents an order placement request."""

    symbol: str
    side: str  # "buy" or "sell"
    size: Decimal
    price: Decimal | None = None
    order_type: str = "limit"  # "market" or "limit"


class OrderResponse(BaseModel):
    """Represents an order placement response."""

    order_id: str
    symbol: str
    side: str
    size: Decimal
    price: Decimal | None
    status: str
    timestamp: datetime


# =============================================================================
# MOCK DATA GENERATOR
# =============================================================================


class MockDataGenerator:
    """Generates realistic mock market data for testing."""

    def __init__(self, config: MockConfig):
        self.config = config
        self.sequence_number = 0

        # Base prices for different symbols
        self.base_prices = {
            "BTC-USD": Decimal("50000.00"),
            "ETH-USD": Decimal("3000.00"),
            "SOL-USD": Decimal("100.00"),
            "BTC-PERP": Decimal("50100.00"),
            "ETH-PERP": Decimal("3010.00"),
            "SOL-PERP": Decimal("101.00"),
        }

        # Current prices (start at base prices)
        self.current_prices = self.base_prices.copy()

        # Price update threads
        self.price_update_tasks = []

    def get_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_number += 1
        return self.sequence_number

    def generate_orderbook(self, symbol: str) -> OrderBookSnapshot:
        """Generate a realistic orderbook for a symbol."""
        if symbol not in self.current_prices:
            raise ValueError(f"Unknown symbol: {symbol}")

        mid_price = self.current_prices[symbol]
        spread = mid_price * Decimal("0.001")  # 0.1% spread

        # Generate bids (descending prices)
        bids = []
        for i in range(self.config.orderbook_depth):
            price = mid_price - spread * (1 + i * 0.5)
            size = Decimal(str(random.uniform(0.1, 10.0)))
            bids.append(OrderBookLevel(price=price, size=size))

        # Generate asks (ascending prices)
        asks = []
        for i in range(self.config.orderbook_depth):
            price = mid_price + spread * (1 + i * 0.5)
            size = Decimal(str(random.uniform(0.1, 10.0)))
            asks.append(OrderBookLevel(price=price, size=size))

        return OrderBookSnapshot(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(UTC),
            sequence=self.get_sequence(),
        )

    def generate_ticker(self, symbol: str) -> MarketTicker:
        """Generate market ticker data for a symbol."""
        if symbol not in self.current_prices:
            raise ValueError(f"Unknown symbol: {symbol}")

        current_price = self.current_prices[symbol]
        base_price = self.base_prices[symbol]

        # Calculate 24h statistics
        change_24h = (current_price - base_price) / base_price * 100
        high_24h = current_price * Decimal("1.05")
        low_24h = current_price * Decimal("0.95")
        volume_24h = Decimal(str(random.uniform(1000000, 10000000)))

        return MarketTicker(
            symbol=symbol,
            price=current_price,
            volume_24h=volume_24h,
            change_24h=change_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            timestamp=datetime.now(UTC),
        )

    async def update_prices(self):
        """Continuously update prices with realistic movement."""
        while True:
            try:
                for symbol in self.current_prices:
                    # Generate price movement
                    change_pct = random.gauss(0, self.config.price_volatility)
                    change_pct = max(-0.05, min(0.05, change_pct))  # Limit to ±5%

                    current = self.current_prices[symbol]
                    new_price = current * (1 + Decimal(str(change_pct)))

                    # Ensure price doesn't go negative
                    self.current_prices[symbol] = max(Decimal("0.01"), new_price)

                await asyncio.sleep(1.0 / self.config.update_frequency)

            except Exception as e:
                logging.exception(f"Error updating prices: {e}")
                await asyncio.sleep(1.0)


# =============================================================================
# MOCK BLUEFIN SERVICE
# =============================================================================


class MockBluefinService:
    """Main mock Bluefin service implementation."""

    def __init__(self):
        self.config = MockConfig()
        self.data_generator = MockDataGenerator(self.config)
        self.websocket_connections = set()
        self.app = self._create_app()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("mock-bluefin")

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Mock Bluefin Service",
            description="Mock Bluefin DEX service for orderbook testing",
            version="1.0.0",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add routes
        self._add_routes(app)

        return app

    def _add_routes(self, app: FastAPI):
        """Add API routes to the application."""

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": self.config.service_name,
                "timestamp": datetime.now(UTC).isoformat(),
                "connections": len(self.websocket_connections),
            }

        @app.get("/orderbook/{symbol}")
        async def get_orderbook(symbol: str):
            """Get orderbook snapshot for a symbol."""
            await self._simulate_latency()

            if self.config.simulate_errors and random.random() < 0.05:  # 5% error rate
                raise HTTPException(status_code=500, detail="Simulated error")

            try:
                orderbook = self.data_generator.generate_orderbook(symbol)
                return {
                    "symbol": orderbook.symbol,
                    "bids": [
                        [str(level.price), str(level.size)] for level in orderbook.bids
                    ],
                    "asks": [
                        [str(level.price), str(level.size)] for level in orderbook.asks
                    ],
                    "timestamp": orderbook.timestamp.isoformat(),
                    "sequence": orderbook.sequence,
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.get("/ticker/{symbol}")
        async def get_ticker(symbol: str):
            """Get ticker data for a symbol."""
            await self._simulate_latency()

            try:
                ticker = self.data_generator.generate_ticker(symbol)
                return {
                    "symbol": ticker.symbol,
                    "price": str(ticker.price),
                    "volume_24h": str(ticker.volume_24h),
                    "change_24h": str(ticker.change_24h),
                    "high_24h": str(ticker.high_24h),
                    "low_24h": str(ticker.low_24h),
                    "timestamp": ticker.timestamp.isoformat(),
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.get("/symbols")
        async def get_symbols():
            """Get list of available symbols."""
            await self._simulate_latency()

            return {
                "symbols": list(self.data_generator.base_prices.keys()),
                "count": len(self.data_generator.base_prices),
            }

        @app.post("/orders")
        async def place_order(order: OrderRequest):
            """Place a new order (mock implementation)."""
            await self._simulate_latency()

            if self.config.simulate_errors and random.random() < 0.02:  # 2% error rate
                raise HTTPException(status_code=400, detail="Simulated order error")

            # Generate mock order response
            order_id = f"mock_order_{int(time.time() * 1000)}"

            response = OrderResponse(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                price=order.price,
                status="pending",
                timestamp=datetime.now(UTC),
            )

            return response.model_dump()

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time orderbook updates."""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            self.logger.info(
                f"WebSocket connection established. Total connections: {len(self.websocket_connections)}"
            )

            try:
                # Send initial orderbook snapshots
                for symbol in self.data_generator.base_prices.keys():
                    orderbook = self.data_generator.generate_orderbook(symbol)
                    message = {
                        "type": "orderbook_snapshot",
                        "data": {
                            "symbol": orderbook.symbol,
                            "bids": [
                                [str(level.price), str(level.size)]
                                for level in orderbook.bids[:10]
                            ],
                            "asks": [
                                [str(level.price), str(level.size)]
                                for level in orderbook.asks[:10]
                            ],
                            "timestamp": orderbook.timestamp.isoformat(),
                            "sequence": orderbook.sequence,
                        },
                    }
                    await websocket.send_text(json.dumps(message))

                # Keep connection alive and send periodic updates
                while True:
                    # Send orderbook updates for random symbols
                    symbol = random.choice(list(self.data_generator.base_prices.keys()))
                    orderbook = self.data_generator.generate_orderbook(symbol)

                    message = {
                        "type": "orderbook_update",
                        "data": {
                            "symbol": orderbook.symbol,
                            "bids": [
                                [str(level.price), str(level.size)]
                                for level in orderbook.bids[:5]
                            ],
                            "asks": [
                                [str(level.price), str(level.size)]
                                for level in orderbook.asks[:5]
                            ],
                            "timestamp": orderbook.timestamp.isoformat(),
                            "sequence": orderbook.sequence,
                        },
                    }

                    await websocket.send_text(json.dumps(message))
                    await asyncio.sleep(1.0 / self.config.update_frequency)

            except WebSocketDisconnect:
                self.logger.info("WebSocket connection disconnected")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.discard(websocket)
                self.logger.info(
                    f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}"
                )

    async def _simulate_latency(self):
        """Simulate network latency."""
        if self.config.simulate_latency:
            # Simulate 10-50ms latency
            latency = random.uniform(0.01, 0.05)
            await asyncio.sleep(latency)

    async def start_background_tasks(self):
        """Start background tasks."""
        # Start price update task
        task = asyncio.create_task(self.data_generator.update_prices())
        self.data_generator.price_update_tasks.append(task)

    async def stop_background_tasks(self):
        """Stop background tasks."""
        for task in self.data_generator.price_update_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================


async def main():
    """Main application entry point."""
    service = MockBluefinService()

    # Start background tasks
    await service.start_background_tasks()

    try:
        # Start the server
        config = uvicorn.Config(
            service.app,
            host="0.0.0.0",
            port=service.config.port,
            log_level=service.config.log_level.lower(),
            access_log=True,
        )
        server = uvicorn.Server(config)

        service.logger.info(
            f"Starting mock Bluefin service on port {service.config.port}"
        )
        await server.serve()

    except KeyboardInterrupt:
        service.logger.info("Received shutdown signal")
    finally:
        # Stop background tasks
        await service.stop_background_tasks()
        service.logger.info("Mock Bluefin service stopped")


if __name__ == "__main__":
    asyncio.run(main())
