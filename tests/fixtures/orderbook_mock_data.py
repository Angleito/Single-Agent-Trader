"""
Comprehensive Orderbook Mock Data Generator

This module provides realistic orderbook mock data and test fixtures for comprehensive
orderbook testing across various market conditions and scenarios.

Features:
- Realistic orderbook data with proper price/volume distributions
- Various market conditions (normal, volatile, illiquid)
- WebSocket message fixtures
- Error response fixtures
- Performance test datasets
- Configuration test fixtures
"""

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

from bot.types.market_data import OrderBook, OrderBookLevel


@dataclass
class OrderBookMockConfig:
    """Configuration for orderbook mock data generation."""

    # Symbol configuration
    symbol: str = "BTC-USD"
    base_price: float = 50000.0

    # Orderbook structure parameters
    depth_levels: int = 50  # Number of price levels per side
    min_spread_bps: int = 1  # Minimum spread in basis points (0.01%)
    max_spread_bps: int = 10  # Maximum spread in basis points (0.1%)

    # Volume parameters
    base_volume: float = 10.0
    volume_variance: float = 0.8  # Volume variance multiplier

    # Price parameters
    price_precision: int = 2  # Number of decimal places for prices
    volume_precision: int = 8  # Number of decimal places for volumes

    # Market condition parameters
    volatility: float = 0.02  # Base volatility
    trend_strength: float = 0.0  # Positive = bullish, negative = bearish

    # Special conditions
    illiquid_mode: bool = False  # Low liquidity conditions
    volatile_mode: bool = False  # High volatility conditions
    wide_spread_mode: bool = False  # Wide spread conditions

    # Sequence numbering
    sequence_start: int = 1000000

    # Random seed for reproducibility
    random_seed: int | None = 42


class OrderBookMockGenerator:
    """Generator for realistic orderbook mock data."""

    def __init__(self, config: OrderBookMockConfig = None):
        self.config = config or OrderBookMockConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self.rng = np.random.default_rng(self.config.random_seed)
        self.current_sequence = self.config.sequence_start

    def generate_normal_orderbook(self, timestamp: datetime = None) -> OrderBook:
        """Generate normal market conditions orderbook."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        mid_price = self._get_current_mid_price()
        spread_bps = self.rng.uniform(
            self.config.min_spread_bps, self.config.max_spread_bps
        )
        spread = mid_price * (spread_bps / 10000)  # Convert bps to decimal

        best_bid = mid_price - (spread / 2)
        best_ask = mid_price + (spread / 2)

        # Generate bid levels (descending prices)
        bids = self._generate_price_levels(
            best_bid, direction="down", levels=self.config.depth_levels
        )

        # Generate ask levels (ascending prices)
        asks = self._generate_price_levels(
            best_ask, direction="up", levels=self.config.depth_levels
        )

        return OrderBook(
            product_id=self.config.symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            sequence=self._next_sequence(),
        )

    def generate_illiquid_orderbook(self, timestamp: datetime = None) -> OrderBook:
        """Generate illiquid market conditions orderbook."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Wider spreads and lower volumes for illiquid conditions
        mid_price = self._get_current_mid_price()
        spread_bps = self.rng.uniform(20, 100)  # 0.2% to 1% spread
        spread = mid_price * (spread_bps / 10000)

        best_bid = mid_price - (spread / 2)
        best_ask = mid_price + (spread / 2)

        # Fewer levels with lower volumes
        depth_levels = max(5, self.config.depth_levels // 3)
        volume_multiplier = 0.2  # Much lower volumes

        bids = self._generate_price_levels(
            best_bid,
            direction="down",
            levels=depth_levels,
            volume_multiplier=volume_multiplier,
        )

        asks = self._generate_price_levels(
            best_ask,
            direction="up",
            levels=depth_levels,
            volume_multiplier=volume_multiplier,
        )

        return OrderBook(
            product_id=self.config.symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            sequence=self._next_sequence(),
        )

    def generate_volatile_orderbook(self, timestamp: datetime = None) -> OrderBook:
        """Generate volatile market conditions orderbook."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Higher volatility affects spread and volume distribution
        mid_price = self._get_current_mid_price() * (1 + self.rng.normal(0, 0.005))
        spread_bps = self.rng.uniform(5, 30)  # Wider spreads during volatility
        spread = mid_price * (spread_bps / 10000)

        best_bid = mid_price - (spread / 2)
        best_ask = mid_price + (spread / 2)

        # More irregular volume distribution
        volume_multiplier = self.rng.uniform(0.5, 3.0)  # Highly variable volumes

        bids = self._generate_price_levels(
            best_bid,
            direction="down",
            levels=self.config.depth_levels,
            volume_multiplier=volume_multiplier,
            irregular_volumes=True,
        )

        asks = self._generate_price_levels(
            best_ask,
            direction="up",
            levels=self.config.depth_levels,
            volume_multiplier=volume_multiplier,
            irregular_volumes=True,
        )

        return OrderBook(
            product_id=self.config.symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            sequence=self._next_sequence(),
        )

    def generate_crossed_orderbook(self, timestamp: datetime = None) -> dict[str, Any]:
        """Generate invalid crossed orderbook for error testing."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        mid_price = self._get_current_mid_price()

        # Create crossed book (best bid > best ask)
        best_bid = mid_price + 10  # Bid higher than mid
        best_ask = mid_price - 10  # Ask lower than mid

        # This should fail validation
        return {
            "product_id": self.config.symbol,
            "timestamp": timestamp.isoformat(),
            "bids": [[str(best_bid), str(self.config.base_volume)]],
            "asks": [[str(best_ask), str(self.config.base_volume)]],
            "sequence": self._next_sequence(),
        }

    def generate_empty_orderbook(self, timestamp: datetime = None) -> OrderBook:
        """Generate empty orderbook for edge case testing."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        return OrderBook(
            product_id=self.config.symbol,
            timestamp=timestamp,
            bids=[],
            asks=[],
            sequence=self._next_sequence(),
        )

    def generate_single_side_orderbook(
        self, side: str = "bids", timestamp: datetime = None
    ) -> OrderBook:
        """Generate orderbook with only one side for edge case testing."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        mid_price = self._get_current_mid_price()

        if side == "bids":
            bids = self._generate_price_levels(
                mid_price - 50, direction="down", levels=self.config.depth_levels
            )
            asks = []
        else:
            bids = []
            asks = self._generate_price_levels(
                mid_price + 50, direction="up", levels=self.config.depth_levels
            )

        return OrderBook(
            product_id=self.config.symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            sequence=self._next_sequence(),
        )

    def generate_orderbook_sequence(self, count: int = 100) -> list[OrderBook]:
        """Generate sequence of orderbooks simulating market evolution."""
        orderbooks = []
        base_time = datetime.now(UTC)

        for i in range(count):
            timestamp = base_time + timedelta(milliseconds=i * 100)

            # Gradually evolve market conditions
            if i < count * 0.3:
                # Start normal
                orderbook = self.generate_normal_orderbook(timestamp)
            elif i < count * 0.6:
                # Become volatile
                orderbook = self.generate_volatile_orderbook(timestamp)
            else:
                # End illiquid
                orderbook = self.generate_illiquid_orderbook(timestamp)

            orderbooks.append(orderbook)

        return orderbooks

    def _generate_price_levels(
        self,
        start_price: float,
        direction: str,
        levels: int,
        volume_multiplier: float = 1.0,
        irregular_volumes: bool = False,
    ) -> list[OrderBookLevel]:
        """Generate price levels for one side of the orderbook."""
        price_levels = []
        current_price = start_price

        # Price step size (gets larger as we move away from best price)
        base_step = start_price * 0.0001  # 0.01% of price

        for i in range(levels):
            # Calculate price step (increases with distance from best price)
            step_multiplier = 1 + (i * 0.1)  # Increasing steps
            price_step = base_step * step_multiplier

            if direction == "down":
                current_price -= price_step
            else:
                current_price += price_step

            # Round to appropriate precision
            rounded_price = round(current_price, self.config.price_precision)

            # Generate volume for this level
            if irregular_volumes:
                # Highly variable volumes for volatile conditions
                volume = (
                    self.config.base_volume
                    * volume_multiplier
                    * self.rng.exponential(1.0)
                )
            else:
                # Normal volume distribution (decreases with distance from best price)
                distance_factor = max(0.1, 1.0 - (i * 0.1))  # Decreasing with distance
                volume_factor = self.rng.gamma(2, 0.5) * distance_factor
                volume = self.config.base_volume * volume_multiplier * volume_factor

            # Round volume to appropriate precision
            rounded_volume = round(volume, self.config.volume_precision)

            # Create order count estimate (higher volumes = more orders)
            order_count = max(1, int(volume / (self.config.base_volume * 0.1)))

            price_levels.append(
                OrderBookLevel(
                    price=Decimal(str(rounded_price)),
                    size=Decimal(str(rounded_volume)),
                    order_count=order_count,
                )
            )

        return price_levels

    def _get_current_mid_price(self) -> float:
        """Get current mid price with trend and noise."""
        # Apply trend
        trend_component = self.config.trend_strength * self.config.base_price

        # Apply volatility noise
        noise_component = self.rng.normal(
            0, self.config.volatility * self.config.base_price
        )

        return self.config.base_price + trend_component + noise_component

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self.current_sequence += 1
        return self.current_sequence


class WebSocketMessageFixtures:
    """Generator for WebSocket message fixtures."""

    def __init__(self, config: OrderBookMockConfig = None):
        self.config = config or OrderBookMockConfig()
        self.generator = OrderBookMockGenerator(config)

    def generate_subscription_message(self) -> dict[str, Any]:
        """Generate orderbook subscription message."""
        return {
            "type": "subscribe",
            "product_ids": [self.config.symbol],
            "channels": [{"channel": "level2", "product_ids": [self.config.symbol]}],
        }

    def generate_subscription_ack(self) -> dict[str, Any]:
        """Generate subscription acknowledgment."""
        return {
            "type": "subscriptions",
            "channels": [{"channel": "level2", "product_ids": [self.config.symbol]}],
        }

    def generate_snapshot_message(self, orderbook: OrderBook = None) -> dict[str, Any]:
        """Generate orderbook snapshot WebSocket message."""
        if orderbook is None:
            orderbook = self.generator.generate_normal_orderbook()

        return {
            "type": "snapshot",
            "product_id": orderbook.product_id,
            "timestamp": orderbook.timestamp.isoformat(),
            "bids": [[str(level.price), str(level.size)] for level in orderbook.bids],
            "asks": [[str(level.price), str(level.size)] for level in orderbook.asks],
        }

    def generate_update_message(
        self, changes: list[tuple[str, str, str]]
    ) -> dict[str, Any]:
        """Generate orderbook update message."""
        return {
            "type": "l2update",
            "product_id": self.config.symbol,
            "timestamp": datetime.now(UTC).isoformat(),
            "changes": changes,
        }

    def generate_heartbeat_message(self) -> dict[str, Any]:
        """Generate heartbeat message."""
        return {
            "type": "heartbeat",
            "timestamp": datetime.now(UTC).isoformat(),
            "last_trade_id": self.generator._next_sequence(),
            "product_id": self.config.symbol,
            "sequence": self.generator._next_sequence(),
        }

    def generate_error_message(
        self, error_type: str = "invalid_request"
    ) -> dict[str, Any]:
        """Generate error message."""
        error_messages = {
            "invalid_request": "Invalid request format",
            "unauthorized": "Authentication required",
            "rate_limit": "Rate limit exceeded",
            "invalid_symbol": f"Invalid product_id: {self.config.symbol}",
            "connection_error": "WebSocket connection error",
        }

        return {
            "type": "error",
            "message": error_messages.get(error_type, "Unknown error"),
            "reason": error_type,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def generate_message_sequence(self, count: int = 50) -> list[dict[str, Any]]:
        """Generate realistic sequence of WebSocket messages."""
        messages = []

        # Start with subscription
        messages.append(self.generate_subscription_message())
        messages.append(self.generate_subscription_ack())

        # Initial snapshot
        messages.append(self.generate_snapshot_message())

        # Series of updates with occasional heartbeats
        for i in range(count):
            if i % 10 == 0:
                # Heartbeat every 10 messages
                messages.append(self.generate_heartbeat_message())
            else:
                # Random updates
                changes = self._generate_random_changes()
                messages.append(self.generate_update_message(changes))

        return messages

    def _generate_random_changes(self) -> list[tuple[str, str, str]]:
        """Generate random orderbook changes."""
        changes = []
        num_changes = random.randint(1, 5)

        for _ in range(num_changes):
            side = random.choice(["buy", "sell"])
            price = str(round(self.config.base_price * random.uniform(0.999, 1.001), 2))
            size = str(round(random.uniform(0, 10), 6))  # 0 size = removal

            changes.append([side, price, size])

        return changes


class ErrorResponseFixtures:
    """Generator for error response fixtures."""

    def __init__(self, config: OrderBookMockConfig = None):
        self.config = config or OrderBookMockConfig()

    def generate_http_errors(self) -> dict[int, dict[str, Any]]:
        """Generate HTTP error responses."""
        return {
            400: {
                "error": "Bad Request",
                "message": "Invalid request parameters",
                "details": {"product_id": "Invalid product_id format"},
            },
            401: {"error": "Unauthorized", "message": "API key required"},
            403: {"error": "Forbidden", "message": "Insufficient permissions"},
            429: {
                "error": "Rate Limit Exceeded",
                "message": "Too many requests",
                "retry_after": 60,
            },
            500: {
                "error": "Internal Server Error",
                "message": "Temporary server error",
            },
            503: {
                "error": "Service Unavailable",
                "message": "Market data service temporarily unavailable",
            },
        }

    def generate_websocket_errors(self) -> list[dict[str, Any]]:
        """Generate WebSocket error scenarios."""
        return [
            {"type": "error", "message": "Connection timeout", "code": "TIMEOUT"},
            {
                "type": "error",
                "message": "Invalid authentication",
                "code": "AUTH_FAILED",
            },
            {
                "type": "error",
                "message": "Market data unavailable",
                "code": "DATA_UNAVAILABLE",
            },
            {
                "type": "error",
                "message": "Invalid product_id",
                "code": "INVALID_PRODUCT",
            },
        ]

    def generate_malformed_data(self) -> list[dict[str, Any]]:
        """Generate malformed data for error handling tests."""
        return [
            # Missing required fields
            {
                "type": "snapshot",
                "bids": [[50000, 1.0]],  # Missing asks, product_id
            },
            # Invalid price format
            {
                "type": "snapshot",
                "product_id": self.config.symbol,
                "bids": [["not_a_number", "1.0"]],
                "asks": [["50100", "1.0"]],
            },
            # Negative prices
            {
                "type": "snapshot",
                "product_id": self.config.symbol,
                "bids": [["-100", "1.0"]],
                "asks": [["50100", "1.0"]],
            },
            # Invalid timestamp
            {
                "type": "snapshot",
                "product_id": self.config.symbol,
                "timestamp": "not_a_timestamp",
                "bids": [["50000", "1.0"]],
                "asks": [["50100", "1.0"]],
            },
        ]


class PerformanceTestDatasets:
    """Generator for performance testing datasets."""

    def __init__(self, config: OrderBookMockConfig = None):
        self.config = config or OrderBookMockConfig()
        self.generator = OrderBookMockGenerator(config)

    def generate_large_orderbook(self, depth_per_side: int = 1000) -> OrderBook:
        """Generate large orderbook for performance testing."""
        # Temporarily override depth for large book
        original_depth = self.config.depth_levels
        self.config.depth_levels = depth_per_side

        try:
            orderbook = self.generator.generate_normal_orderbook()
            return orderbook
        finally:
            self.config.depth_levels = original_depth

    def generate_high_frequency_updates(
        self, count: int = 10000
    ) -> list[dict[str, Any]]:
        """Generate high-frequency orderbook updates."""
        ws_fixtures = WebSocketMessageFixtures(self.config)
        updates = []

        for _ in range(count):
            changes = []
            # Generate 1-3 changes per update
            for _ in range(random.randint(1, 3)):
                side = random.choice(["buy", "sell"])
                price = round(self.config.base_price * random.uniform(0.99, 1.01), 2)
                size = round(random.uniform(0.1, 5.0), 6)
                changes.append([side, str(price), str(size)])

            updates.append(ws_fixtures.generate_update_message(changes))

        return updates

    def generate_stress_test_data(self) -> dict[str, Any]:
        """Generate comprehensive stress test dataset."""
        return {
            "large_orderbooks": [
                self.generate_large_orderbook(depth) for depth in [100, 500, 1000, 2000]
            ],
            "high_frequency_updates": self.generate_high_frequency_updates(10000),
            "concurrent_symbols": self._generate_multi_symbol_data(),
            "memory_test_sequence": [
                self.generator.generate_normal_orderbook()
                for _ in range(50000)  # Large number for memory testing
            ],
        }

    def _generate_multi_symbol_data(self) -> dict[str, list[OrderBook]]:
        """Generate data for multiple symbols simultaneously."""
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOT-USD"]
        multi_data = {}

        for symbol in symbols:
            # Create config for each symbol
            symbol_config = OrderBookMockConfig(
                symbol=symbol,
                base_price=50000.0 * random.uniform(0.1, 2.0),  # Different price ranges
                random_seed=hash(symbol) % 10000,
            )

            symbol_generator = OrderBookMockGenerator(symbol_config)
            multi_data[symbol] = [
                symbol_generator.generate_normal_orderbook() for _ in range(100)
            ]

        return multi_data


class ConfigurationTestFixtures:
    """Generator for configuration test fixtures."""

    def generate_test_configs(self) -> dict[str, OrderBookMockConfig]:
        """Generate various test configurations."""
        return {
            "default": OrderBookMockConfig(),
            "high_precision": OrderBookMockConfig(
                price_precision=8, volume_precision=12
            ),
            "wide_spreads": OrderBookMockConfig(
                min_spread_bps=50, max_spread_bps=200, wide_spread_mode=True
            ),
            "illiquid": OrderBookMockConfig(
                depth_levels=10, base_volume=1.0, illiquid_mode=True
            ),
            "volatile": OrderBookMockConfig(volatility=0.1, volatile_mode=True),
            "trending_bull": OrderBookMockConfig(trend_strength=0.001, volatility=0.03),
            "trending_bear": OrderBookMockConfig(
                trend_strength=-0.001, volatility=0.03
            ),
            "micro_structure": OrderBookMockConfig(
                depth_levels=5, base_volume=0.1, min_spread_bps=1, max_spread_bps=3
            ),
        }

    def generate_symbol_configs(self) -> dict[str, OrderBookMockConfig]:
        """Generate configs for different trading pairs."""
        return {
            "BTC-USD": OrderBookMockConfig(
                symbol="BTC-USD",
                base_price=50000.0,
                price_precision=2,
                min_spread_bps=1,
                max_spread_bps=5,
            ),
            "ETH-USD": OrderBookMockConfig(
                symbol="ETH-USD",
                base_price=3000.0,
                price_precision=2,
                min_spread_bps=2,
                max_spread_bps=8,
            ),
            "SOL-USD": OrderBookMockConfig(
                symbol="SOL-USD",
                base_price=100.0,
                price_precision=3,
                min_spread_bps=5,
                max_spread_bps=20,
            ),
            "AVAX-USD": OrderBookMockConfig(
                symbol="AVAX-USD",
                base_price=25.0,
                price_precision=3,
                min_spread_bps=8,
                max_spread_bps=25,
            ),
        }


# Factory function for easy access
def create_orderbook_mock_suite(config: OrderBookMockConfig = None) -> dict[str, Any]:
    """Create complete orderbook mock data suite."""
    config = config or OrderBookMockConfig()

    generator = OrderBookMockGenerator(config)
    ws_fixtures = WebSocketMessageFixtures(config)
    error_fixtures = ErrorResponseFixtures(config)
    perf_datasets = PerformanceTestDatasets(config)
    config_fixtures = ConfigurationTestFixtures()

    return {
        # Basic orderbooks
        "normal_orderbook": generator.generate_normal_orderbook(),
        "illiquid_orderbook": generator.generate_illiquid_orderbook(),
        "volatile_orderbook": generator.generate_volatile_orderbook(),
        "empty_orderbook": generator.generate_empty_orderbook(),
        "single_side_bids": generator.generate_single_side_orderbook("bids"),
        "single_side_asks": generator.generate_single_side_orderbook("asks"),
        # Sequences
        "orderbook_sequence": generator.generate_orderbook_sequence(100),
        # WebSocket messages
        "websocket_messages": ws_fixtures.generate_message_sequence(50),
        "subscription_message": ws_fixtures.generate_subscription_message(),
        "heartbeat_message": ws_fixtures.generate_heartbeat_message(),
        # Error fixtures
        "http_errors": error_fixtures.generate_http_errors(),
        "websocket_errors": error_fixtures.generate_websocket_errors(),
        "malformed_data": error_fixtures.generate_malformed_data(),
        "crossed_orderbook": generator.generate_crossed_orderbook(),
        # Performance datasets
        "stress_test_data": perf_datasets.generate_stress_test_data(),
        "large_orderbook": perf_datasets.generate_large_orderbook(500),
        # Configuration fixtures
        "test_configs": config_fixtures.generate_test_configs(),
        "symbol_configs": config_fixtures.generate_symbol_configs(),
        # Metadata
        "config": config,
        "generation_timestamp": datetime.now(UTC),
        "data_summary": {
            "total_orderbooks": 7,
            "total_websocket_messages": 50,
            "error_scenarios": 15,
            "performance_datasets": 4,
            "test_configurations": 8,
        },
    }


# Export main classes and functions
__all__ = [
    "ConfigurationTestFixtures",
    "ErrorResponseFixtures",
    "OrderBookMockConfig",
    "OrderBookMockGenerator",
    "PerformanceTestDatasets",
    "WebSocketMessageFixtures",
    "create_orderbook_mock_suite",
]
