#!/usr/bin/env python3
"""
Mock Coinbase Advanced Trade API Service for Orderbook Testing

This service simulates the Coinbase Advanced Trade API and WebSocket feeds to provide
realistic orderbook data for testing without requiring actual API credentials.

Features:
- RESTful API endpoints matching Coinbase Advanced Trade interface
- Real-time WebSocket orderbook feeds matching Coinbase format
- Realistic market data simulation with proper Coinbase data structures
- Error simulation for fault tolerance testing
- Authentication simulation (without requiring real credentials)
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================


class MockConfig:
    """Configuration for mock Coinbase service."""

    def __init__(self):
        self.port = int(os.getenv("MOCK_SERVICE_PORT", "8081"))
        self.service_name = os.getenv("MOCK_SERVICE_NAME", "coinbase")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.simulate_latency = os.getenv("SIMULATE_LATENCY", "true").lower() == "true"
        self.simulate_errors = os.getenv("SIMULATE_ERRORS", "false").lower() == "true"
        self.orderbook_depth = int(os.getenv("ORDERBOOK_DEPTH", "50"))
        self.tick_size = float(os.getenv("TICK_SIZE", "0.01"))
        self.update_frequency = float(
            os.getenv("UPDATE_FREQUENCY", "5.0")
        )  # Updates per second


# =============================================================================
# DATA MODELS (COINBASE FORMAT)
# =============================================================================


class CoinbaseOrderBookLevel(BaseModel):
    """Represents a single Coinbase orderbook level."""

    price_level: Decimal
    new_quantity: Decimal


class CoinbaseOrderBook(BaseModel):
    """Represents a Coinbase orderbook snapshot."""

    product_id: str
    bids: list[list[str]]  # [price, size] pairs as strings
    asks: list[list[str]]  # [price, size] pairs as strings
    time: str  # ISO timestamp


class CoinbaseProduct(BaseModel):
    """Represents a Coinbase product."""

    id: str
    base_currency: str
    quote_currency: str
    base_min_size: str
    base_max_size: str
    quote_min_size: str
    quote_max_size: str
    base_increment: str
    quote_increment: str
    display_name: str
    min_market_funds: str
    max_market_funds: str
    margin_enabled: bool
    post_only: bool
    limit_only: bool
    cancel_only: bool
    trading_disabled: bool
    status: str
    status_message: str


class CoinbaseTicker(BaseModel):
    """Represents Coinbase ticker data."""

    product_id: str
    price: str
    price_percentage_change_24h: str
    volume_24h: str
    volume_percentage_change_24h: str
    base_increment: str
    quote_increment: str
    quote_min_size: str
    quote_max_size: str
    base_min_size: str
    base_max_size: str
    base_name: str
    quote_name: str
    watched: bool
    is_disabled: bool
    new: bool
    status: str
    cancel_only: bool
    limit_only: bool
    post_only: bool
    trading_disabled: bool


class CoinbaseOrder(BaseModel):
    """Represents a Coinbase order."""

    order_id: str
    product_id: str
    user_id: str
    order_configuration: dict[str, Any]
    side: str
    client_order_id: str
    status: str
    time_in_force: str
    created_time: str
    completion_percentage: str
    filled_size: str
    average_filled_price: str
    fee: str
    number_of_fills: str
    filled_value: str
    pending_cancel: bool
    size_in_quote: bool
    total_fees: str
    size_inclusive_of_fees: bool
    total_value_after_fees: str
    trigger_status: str
    order_type: str
    reject_reason: str
    settled: bool
    product_type: str
    reject_message: str
    cancel_message: str


# =============================================================================
# MOCK DATA GENERATOR
# =============================================================================


class MockCoinbaseDataGenerator:
    """Generates realistic mock Coinbase market data for testing."""

    def __init__(self, config: MockConfig):
        self.config = config
        self.sequence_number = 0

        # Coinbase-style product definitions
        self.products = {
            "BTC-USD": {
                "base_currency": "BTC",
                "quote_currency": "USD",
                "base_price": Decimal("50000.00"),
                "tick_size": Decimal("0.01"),
                "min_size": Decimal("0.00001"),
                "max_size": Decimal(10000),
            },
            "ETH-USD": {
                "base_currency": "ETH",
                "quote_currency": "USD",
                "base_price": Decimal("3000.00"),
                "tick_size": Decimal("0.01"),
                "min_size": Decimal("0.001"),
                "max_size": Decimal(10000),
            },
            "SOL-USD": {
                "base_currency": "SOL",
                "quote_currency": "USD",
                "base_price": Decimal("100.00"),
                "tick_size": Decimal("0.01"),
                "min_size": Decimal("0.01"),
                "max_size": Decimal(10000),
            },
        }

        # Current prices (start at base prices)
        self.current_prices = {
            product_id: info["base_price"] for product_id, info in self.products.items()
        }

        # Price update tasks
        self.price_update_tasks = []

    def get_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_number += 1
        return self.sequence_number

    def generate_orderbook(self, product_id: str) -> CoinbaseOrderBook:
        """Generate a realistic Coinbase orderbook for a product."""
        if product_id not in self.products:
            raise ValueError(f"Unknown product: {product_id}")

        product_info = self.products[product_id]
        mid_price = self.current_prices[product_id]
        tick_size = product_info["tick_size"]

        # Calculate spread (wider for less liquid markets)
        base_spread = mid_price * Decimal("0.001")  # 0.1% base spread
        spread_ticks = max(1, int(base_spread / tick_size))
        actual_spread = spread_ticks * tick_size

        # Generate bids (descending prices)
        bids = []
        for i in range(self.config.orderbook_depth):
            price = mid_price - actual_spread * (1 + i * 0.5)
            price = (price // tick_size) * tick_size  # Round to tick size
            size = Decimal(str(random.uniform(0.1, 10.0)))
            bids.append([str(price), str(size)])

        # Generate asks (ascending prices)
        asks = []
        for i in range(self.config.orderbook_depth):
            price = mid_price + actual_spread * (1 + i * 0.5)
            price = (price // tick_size) * tick_size  # Round to tick size
            size = Decimal(str(random.uniform(0.1, 10.0)))
            asks.append([str(price), str(size)])

        return CoinbaseOrderBook(
            product_id=product_id,
            bids=bids,
            asks=asks,
            time=datetime.now(UTC).isoformat() + "Z",
        )

    def generate_product(self, product_id: str) -> CoinbaseProduct:
        """Generate Coinbase product information."""
        if product_id not in self.products:
            raise ValueError(f"Unknown product: {product_id}")

        info = self.products[product_id]

        return CoinbaseProduct(
            id=product_id,
            base_currency=info["base_currency"],
            quote_currency=info["quote_currency"],
            base_min_size=str(info["min_size"]),
            base_max_size=str(info["max_size"]),
            quote_min_size="1.00",
            quote_max_size="1000000.00",
            base_increment=str(info["min_size"]),
            quote_increment=str(info["tick_size"]),
            display_name=f"{info['base_currency']}-{info['quote_currency']}",
            min_market_funds="10.00",
            max_market_funds="1000000.00",
            margin_enabled=False,
            post_only=False,
            limit_only=False,
            cancel_only=False,
            trading_disabled=False,
            status="online",
            status_message="",
        )

    def generate_ticker(self, product_id: str) -> CoinbaseTicker:
        """Generate Coinbase ticker data."""
        if product_id not in self.products:
            raise ValueError(f"Unknown product: {product_id}")

        info = self.products[product_id]
        current_price = self.current_prices[product_id]
        base_price = info["base_price"]

        # Calculate 24h change
        change_24h = (current_price - base_price) / base_price * 100
        volume_24h = Decimal(str(random.uniform(1000000, 10000000)))

        return CoinbaseTicker(
            product_id=product_id,
            price=str(current_price),
            price_percentage_change_24h=str(change_24h),
            volume_24h=str(volume_24h),
            volume_percentage_change_24h=str(random.uniform(-20, 20)),
            base_increment=str(info["min_size"]),
            quote_increment=str(info["tick_size"]),
            quote_min_size="1.00",
            quote_max_size="1000000.00",
            base_min_size=str(info["min_size"]),
            base_max_size=str(info["max_size"]),
            base_name=info["base_currency"],
            quote_name=info["quote_currency"],
            watched=False,
            is_disabled=False,
            new=False,
            status="online",
            cancel_only=False,
            limit_only=False,
            post_only=False,
            trading_disabled=False,
        )

    async def update_prices(self):
        """Continuously update prices with realistic movement."""
        while True:
            try:
                for product_id in self.products:
                    # Generate price movement
                    change_pct = random.gauss(0, 0.02)  # 2% volatility
                    change_pct = max(-0.05, min(0.05, change_pct))  # Limit to ±5%

                    current = self.current_prices[product_id]
                    new_price = current * (1 + Decimal(str(change_pct)))

                    # Round to tick size
                    tick_size = self.products[product_id]["tick_size"]
                    new_price = (new_price // tick_size) * tick_size

                    # Ensure price doesn't go negative
                    self.current_prices[product_id] = max(tick_size, new_price)

                await asyncio.sleep(1.0 / self.config.update_frequency)

            except Exception as e:
                logging.exception(f"Error updating prices: {e}")
                await asyncio.sleep(1.0)


# =============================================================================
# MOCK COINBASE SERVICE
# =============================================================================


class MockCoinbaseService:
    """Main mock Coinbase Advanced Trade API service implementation."""

    def __init__(self):
        self.config = MockConfig()
        self.data_generator = MockCoinbaseDataGenerator(self.config)
        self.websocket_connections = set()
        self.app = self._create_app()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("mock-coinbase")

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Mock Coinbase Advanced Trade API",
            description="Mock Coinbase Advanced Trade API service for orderbook testing",
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

        @app.get("/api/v3/brokerage/products")
        async def get_products():
            """Get list of available products."""
            await self._simulate_latency()

            products = []
            for product_id in self.data_generator.products:
                product = self.data_generator.generate_product(product_id)
                products.append(product.model_dump())

            return {"products": products}

        @app.get("/api/v3/brokerage/products/{product_id}")
        async def get_product(product_id: str):
            """Get specific product information."""
            await self._simulate_latency()

            try:
                product = self.data_generator.generate_product(product_id)
                return product.model_dump()
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.get("/api/v3/brokerage/products/{product_id}/book")
        async def get_orderbook(product_id: str, limit: int = 50):
            """Get orderbook for a product."""
            await self._simulate_latency()

            if self.config.simulate_errors and random.random() < 0.05:  # 5% error rate
                raise HTTPException(status_code=500, detail="Simulated error")

            try:
                orderbook = self.data_generator.generate_orderbook(product_id)

                # Limit depth if requested
                limited_bids = orderbook.bids[:limit] if limit else orderbook.bids
                limited_asks = orderbook.asks[:limit] if limit else orderbook.asks

                return {
                    "pricebook": {
                        "product_id": orderbook.product_id,
                        "bids": limited_bids,
                        "asks": limited_asks,
                        "time": orderbook.time,
                    }
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.get("/api/v3/brokerage/products/{product_id}/ticker")
        async def get_ticker(product_id: str):
            """Get ticker data for a product."""
            await self._simulate_latency()

            try:
                ticker = self.data_generator.generate_ticker(product_id)
                return ticker.model_dump()
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.post("/api/v3/brokerage/orders")
        async def create_order(order_request: dict):
            """Create a new order (mock implementation)."""
            await self._simulate_latency()

            if self.config.simulate_errors and random.random() < 0.02:  # 2% error rate
                raise HTTPException(status_code=400, detail="Simulated order error")

            # Generate mock order response
            order_id = f"mock_order_{int(time.time() * 1000)}"

            return {
                "success": True,
                "order_id": order_id,
                "success_response": {
                    "order_id": order_id,
                    "product_id": order_request.get("product_id", "BTC-USD"),
                    "side": order_request.get("side", "buy"),
                    "client_order_id": order_request.get("client_order_id", ""),
                },
                "order_configuration": order_request.get("order_configuration", {}),
            }

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data feeds."""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            self.logger.info(
                f"WebSocket connection established. Total connections: {len(self.websocket_connections)}"
            )

            try:
                # Send initial heartbeat
                await websocket.send_text(
                    json.dumps(
                        {
                            "channel": "heartbeats",
                            "client_id": "",
                            "timestamp": datetime.now(UTC).isoformat() + "Z",
                            "sequence_num": self.data_generator.get_sequence(),
                            "events": [
                                {
                                    "type": "heartbeat",
                                    "current_time": datetime.now(UTC).isoformat() + "Z",
                                }
                            ],
                        }
                    )
                )

                # Send initial orderbook snapshots
                for product_id in self.data_generator.products:
                    orderbook = self.data_generator.generate_orderbook(product_id)

                    message = {
                        "channel": "level2",
                        "client_id": "",
                        "timestamp": orderbook.time,
                        "sequence_num": self.data_generator.get_sequence(),
                        "events": [
                            {
                                "type": "snapshot",
                                "product_id": product_id,
                                "updates": [
                                    {
                                        "side": "bid",
                                        "event_time": orderbook.time,
                                        "price_level": bid[0],
                                        "new_quantity": bid[1],
                                    }
                                    for bid in orderbook.bids[:10]
                                ]
                                + [
                                    {
                                        "side": "offer",
                                        "event_time": orderbook.time,
                                        "price_level": ask[0],
                                        "new_quantity": ask[1],
                                    }
                                    for ask in orderbook.asks[:10]
                                ],
                            }
                        ],
                    }

                    await websocket.send_text(json.dumps(message))

                # Keep connection alive and send periodic updates
                while True:
                    # Send orderbook updates for random products
                    product_id = random.choice(
                        list(self.data_generator.products.keys())
                    )
                    orderbook = self.data_generator.generate_orderbook(product_id)

                    # Generate random updates
                    updates = []
                    for i in range(random.randint(1, 5)):
                        side = random.choice(["bid", "offer"])
                        levels = orderbook.bids if side == "bid" else orderbook.asks
                        if levels:
                            level = random.choice(levels[:5])
                            updates.append(
                                {
                                    "side": side,
                                    "event_time": orderbook.time,
                                    "price_level": level[0],
                                    "new_quantity": str(
                                        Decimal(level[1])
                                        * Decimal(str(random.uniform(0.5, 1.5)))
                                    ),
                                }
                            )

                    message = {
                        "channel": "level2",
                        "client_id": "",
                        "timestamp": orderbook.time,
                        "sequence_num": self.data_generator.get_sequence(),
                        "events": [
                            {
                                "type": "update",
                                "product_id": product_id,
                                "updates": updates,
                            }
                        ],
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
            # Simulate 20-100ms latency (Coinbase is typically slower than DEX)
            latency = random.uniform(0.02, 0.1)
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
    service = MockCoinbaseService()

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
            f"Starting mock Coinbase service on port {service.config.port}"
        )
        await server.serve()

    except KeyboardInterrupt:
        service.logger.info("Received shutdown signal")
    finally:
        # Stop background tasks
        await service.stop_background_tasks()
        service.logger.info("Mock Coinbase service stopped")


if __name__ == "__main__":
    asyncio.run(main())
