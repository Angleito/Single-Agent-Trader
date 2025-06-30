"""
WebSocket Message Factory for Testing

This module provides factories for generating realistic WebSocket messages
for comprehensive testing of WebSocket handling, message parsing, and
error scenarios in trading systems.

Features:
- Realistic message timing and sequencing
- Multiple exchange format support (Coinbase, Bluefin, generic)
- Error and edge case message generation
- Performance test message streams
- Message validation fixtures
"""

import random
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """WebSocket message types."""

    # Connection management
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTIONS = "subscriptions"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

    # Market data
    SNAPSHOT = "snapshot"
    L2UPDATE = "l2update"
    TICKER = "ticker"
    TRADE = "match"

    # Order management
    RECEIVED = "received"
    OPEN = "open"
    DONE = "done"
    CHANGE = "change"

    # Exchange specific
    BLUEFIN_ORDERBOOK = "orderbook"
    BLUEFIN_TRADE = "trade"
    BLUEFIN_TICKER = "ticker_24hr"


@dataclass
class MessageFactoryConfig:
    """Configuration for WebSocket message factory."""

    # Exchange configuration
    exchange: str = "coinbase"  # coinbase, bluefin, generic

    # Symbol configuration
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD"])
    base_prices: dict[str, float] = field(default_factory=lambda: {"BTC-USD": 50000.0})

    # Message timing
    message_interval_ms: int = 100  # Milliseconds between messages
    heartbeat_interval_s: int = 30  # Seconds between heartbeats

    # Market parameters
    volatility: float = 0.02
    spread_bps: float = 5.0  # Basis points
    volume_base: float = 10.0

    # Sequence numbering
    sequence_start: int = 1000000
    trade_id_start: int = 500000

    # Error simulation
    error_rate: float = 0.01  # 1% of messages are errors
    malformed_rate: float = 0.005  # 0.5% are malformed

    # Random seed
    random_seed: int | None = 42


class WebSocketMessageFactory:
    """Factory for generating WebSocket messages."""

    def __init__(self, config: MessageFactoryConfig = None):
        self.config = config or MessageFactoryConfig()
        if self.config.random_seed:
            random.seed(self.config.random_seed)

        self.sequence = self.config.sequence_start
        self.trade_id = self.config.trade_id_start
        self.last_heartbeat = datetime.now(UTC)

        # Track current state for each symbol
        self.symbol_state = {}
        for symbol in self.config.symbols:
            self.symbol_state[symbol] = {
                "current_price": self.config.base_prices.get(symbol, 50000.0),
                "last_trade_price": self.config.base_prices.get(symbol, 50000.0),
                "volume_24h": random.uniform(1000, 10000),
                "price_24h": self.config.base_prices.get(symbol, 50000.0)
                * random.uniform(0.95, 1.05),
            }

    def generate_subscription_message(
        self, channels: list[str] = None
    ) -> dict[str, Any]:
        """Generate subscription message."""
        if channels is None:
            channels = ["level2", "ticker", "matches"]

        if self.config.exchange == "coinbase":
            return {
                "type": "subscribe",
                "product_ids": self.config.symbols,
                "channels": [
                    {"channel": channel, "product_ids": self.config.symbols}
                    for channel in channels
                ],
            }
        if self.config.exchange == "bluefin":
            return {
                "method": "SUBSCRIBE",
                "params": self.config.symbols,
                "id": str(uuid.uuid4()),
            }
        return {
            "action": "subscribe",
            "symbols": self.config.symbols,
            "channels": channels,
        }

    def generate_subscription_ack(self, channels: list[str] = None) -> dict[str, Any]:
        """Generate subscription acknowledgment."""
        if channels is None:
            channels = ["level2", "ticker", "matches"]

        if self.config.exchange == "coinbase":
            return {
                "type": "subscriptions",
                "channels": [
                    {"channel": channel, "product_ids": self.config.symbols}
                    for channel in channels
                ],
            }
        if self.config.exchange == "bluefin":
            return {"result": "success", "id": str(uuid.uuid4())}
        return {
            "action": "subscribed",
            "symbols": self.config.symbols,
            "channels": channels,
        }

    def generate_snapshot_message(self, symbol: str = None) -> dict[str, Any]:
        """Generate orderbook snapshot message."""
        symbol = symbol or self.config.symbols[0]
        current_price = self.symbol_state[symbol]["current_price"]

        # Generate spread
        spread = current_price * (self.config.spread_bps / 10000)
        best_bid = current_price - (spread / 2)
        best_ask = current_price + (spread / 2)

        # Generate levels
        bids = self._generate_order_levels(best_bid, "down", 20)
        asks = self._generate_order_levels(best_ask, "up", 20)

        if self.config.exchange == "coinbase":
            return {
                "type": "snapshot",
                "product_id": symbol,
                "timestamp": datetime.now(UTC).isoformat(),
                "bids": bids,
                "asks": asks,
            }
        if self.config.exchange == "bluefin":
            return {
                "stream": f"{symbol.lower()}@depth",
                "data": {
                    "symbol": symbol,
                    "bids": bids,
                    "asks": asks,
                    "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                },
            }
        return {
            "type": "snapshot",
            "symbol": symbol,
            "timestamp": datetime.now(UTC).isoformat(),
            "bids": bids,
            "asks": asks,
        }

    def generate_l2update_message(self, symbol: str = None) -> dict[str, Any]:
        """Generate L2 orderbook update message."""
        symbol = symbol or self.config.symbols[0]

        # Generate 1-3 random changes
        changes = []
        for _ in range(random.randint(1, 3)):
            side = random.choice(["buy", "sell"])
            price = self._get_random_price_near_mid(symbol)
            size = str(round(random.uniform(0, self.config.volume_base * 2), 6))
            changes.append([side, str(price), size])

        if self.config.exchange == "coinbase":
            return {
                "type": "l2update",
                "product_id": symbol,
                "timestamp": datetime.now(UTC).isoformat(),
                "changes": changes,
            }
        if self.config.exchange == "bluefin":
            return {
                "stream": f"{symbol.lower()}@depth",
                "data": {
                    "symbol": symbol,
                    "changes": changes,
                    "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                },
            }
        return {
            "type": "orderbook_update",
            "symbol": symbol,
            "timestamp": datetime.now(UTC).isoformat(),
            "changes": changes,
        }

    def generate_trade_message(self, symbol: str = None) -> dict[str, Any]:
        """Generate trade execution message."""
        symbol = symbol or self.config.symbols[0]

        # Update symbol state with trade
        price_change = random.uniform(-0.005, 0.005)  # ±0.5% price change
        new_price = self.symbol_state[symbol]["current_price"] * (1 + price_change)
        self.symbol_state[symbol]["current_price"] = new_price
        self.symbol_state[symbol]["last_trade_price"] = new_price

        trade_size = round(random.uniform(0.01, 2.0), 6)
        side = random.choice(["buy", "sell"])

        if self.config.exchange == "coinbase":
            return {
                "type": "match",
                "trade_id": self._next_trade_id(),
                "maker_order_id": str(uuid.uuid4()),
                "taker_order_id": str(uuid.uuid4()),
                "side": side,
                "size": str(trade_size),
                "price": str(round(new_price, 2)),
                "product_id": symbol,
                "sequence": self._next_sequence(),
                "time": datetime.now(UTC).isoformat(),
            }
        if self.config.exchange == "bluefin":
            return {
                "stream": f"{symbol.lower()}@trade",
                "data": {
                    "symbol": symbol,
                    "price": str(round(new_price, 2)),
                    "quantity": str(trade_size),
                    "side": side,
                    "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                    "trade_id": str(self._next_trade_id()),
                },
            }
        return {
            "type": "trade",
            "symbol": symbol,
            "price": str(round(new_price, 2)),
            "size": str(trade_size),
            "side": side,
            "trade_id": str(self._next_trade_id()),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def generate_ticker_message(self, symbol: str = None) -> dict[str, Any]:
        """Generate ticker message."""
        symbol = symbol or self.config.symbols[0]
        state = self.symbol_state[symbol]

        # Calculate 24h change
        price_24h_change = (
            (state["current_price"] - state["price_24h"]) / state["price_24h"]
        ) * 100

        if self.config.exchange == "coinbase":
            return {
                "type": "ticker",
                "sequence": self._next_sequence(),
                "product_id": symbol,
                "price": str(round(state["current_price"], 2)),
                "open_24h": str(round(state["price_24h"], 2)),
                "volume_24h": str(round(state["volume_24h"], 2)),
                "low_24h": str(round(state["price_24h"] * 0.95, 2)),
                "high_24h": str(round(state["price_24h"] * 1.05, 2)),
                "volume_30d": str(round(state["volume_24h"] * 30, 2)),
                "best_bid": str(round(state["current_price"] * 0.9995, 2)),
                "best_ask": str(round(state["current_price"] * 1.0005, 2)),
                "side": random.choice(["buy", "sell"]),
                "time": datetime.now(UTC).isoformat(),
                "trade_id": self._next_trade_id(),
            }
        if self.config.exchange == "bluefin":
            return {
                "stream": f"{symbol.lower()}@ticker",
                "data": {
                    "symbol": symbol,
                    "price": str(round(state["current_price"], 2)),
                    "priceChange": str(round(price_24h_change, 2)),
                    "priceChangePercent": str(round(price_24h_change, 4)),
                    "volume": str(round(state["volume_24h"], 2)),
                    "count": random.randint(1000, 5000),
                    "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                },
            }
        return {
            "type": "ticker",
            "symbol": symbol,
            "price": str(round(state["current_price"], 2)),
            "change_24h": str(round(price_24h_change, 2)),
            "volume_24h": str(round(state["volume_24h"], 2)),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def generate_heartbeat_message(self) -> dict[str, Any]:
        """Generate heartbeat message."""
        self.last_heartbeat = datetime.now(UTC)

        if self.config.exchange == "coinbase":
            return {
                "type": "heartbeat",
                "sequence": self._next_sequence(),
                "last_trade_id": self.trade_id,
                "product_id": self.config.symbols[0],
                "time": self.last_heartbeat.isoformat(),
            }
        if self.config.exchange == "bluefin":
            return {
                "method": "ping",
                "timestamp": int(self.last_heartbeat.timestamp() * 1000),
            }
        return {"type": "heartbeat", "timestamp": self.last_heartbeat.isoformat()}

    def generate_error_message(self, error_type: str = "generic") -> dict[str, Any]:
        """Generate error message."""
        error_templates = {
            "generic": "An error occurred",
            "auth": "Authentication failed",
            "rate_limit": "Rate limit exceeded",
            "invalid_symbol": "Invalid symbol",
            "connection": "Connection error",
            "parse": "Message parsing error",
            "timeout": "Request timeout",
        }

        message = error_templates.get(error_type, error_templates["generic"])

        if self.config.exchange == "coinbase":
            return {"type": "error", "message": message, "reason": error_type}
        if self.config.exchange == "bluefin":
            return {"error": {"code": random.randint(1000, 9999), "msg": message}}
        return {"type": "error", "error": message, "code": error_type}

    def generate_malformed_message(self) -> dict[str, Any] | str | None:
        """Generate malformed message for error testing."""
        malformed_types = [
            # Missing required fields
            {
                "type": "snapshot",
                "bids": [["50000", "1.0"]],
            },  # Missing asks, product_id
            # Invalid JSON (return as string)
            '{"type": "snapshot", "invalid": json}',
            # Invalid data types
            {"type": "l2update", "product_id": 12345, "changes": "not_a_list"},
            # None/null message
            None,
            # Empty message
            {},
            # Invalid timestamp
            {
                "type": "ticker",
                "product_id": "BTC-USD",
                "timestamp": "not_a_timestamp",
                "price": "50000",
            },
            # Invalid numbers
            {
                "type": "match",
                "product_id": "BTC-USD",
                "price": "not_a_number",
                "size": "1.0",
            },
        ]

        return random.choice(malformed_types)

    def generate_message_stream(
        self, duration_seconds: int = 60, message_types: list[str] = None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate continuous stream of messages."""
        if message_types is None:
            message_types = ["l2update", "trade", "ticker", "heartbeat"]

        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(seconds=duration_seconds)

        # Send initial snapshot
        for symbol in self.config.symbols:
            yield self.generate_snapshot_message(symbol)

        while datetime.now(UTC) < end_time:
            # Check if it's time for heartbeat
            if (
                datetime.now(UTC) - self.last_heartbeat
            ).total_seconds() >= self.config.heartbeat_interval_s:
                yield self.generate_heartbeat_message()
                continue

            # Simulate error messages
            if random.random() < self.config.error_rate:
                yield self.generate_error_message(
                    random.choice(["generic", "rate_limit", "timeout"])
                )
                continue

            # Simulate malformed messages
            if random.random() < self.config.malformed_rate:
                malformed = self.generate_malformed_message()
                if malformed is not None:
                    yield malformed
                continue

            # Generate normal message
            message_type = random.choice(message_types)
            symbol = random.choice(self.config.symbols)

            if message_type == "l2update":
                yield self.generate_l2update_message(symbol)
            elif message_type == "trade":
                yield self.generate_trade_message(symbol)
            elif message_type == "ticker":
                yield self.generate_ticker_message(symbol)
            elif message_type == "heartbeat":
                yield self.generate_heartbeat_message()

            # Wait for next message
            # Note: In real implementation, this would be handled by event loop

    def generate_burst_messages(self, count: int = 100) -> list[dict[str, Any]]:
        """Generate burst of messages for load testing."""
        messages = []

        for i in range(count):
            symbol = random.choice(self.config.symbols)

            # Higher frequency of updates during burst
            message_type = random.choices(
                ["l2update", "trade", "ticker"], weights=[0.6, 0.3, 0.1], k=1
            )[0]

            if message_type == "l2update":
                messages.append(self.generate_l2update_message(symbol))
            elif message_type == "trade":
                messages.append(self.generate_trade_message(symbol))
            elif message_type == "ticker":
                messages.append(self.generate_ticker_message(symbol))

        return messages

    def _generate_order_levels(
        self, start_price: float, direction: str, count: int
    ) -> list[list[str]]:
        """Generate orderbook levels."""
        levels = []
        current_price = start_price

        for i in range(count):
            # Price step increases with distance
            step = start_price * 0.0001 * (1 + i * 0.1)

            if direction == "down":
                current_price -= step
            else:
                current_price += step

            # Volume decreases with distance
            volume = self.config.volume_base * (1 - i * 0.05) * random.uniform(0.5, 1.5)

            levels.append(
                [str(round(current_price, 2)), str(round(max(volume, 0.01), 6))]
            )

        return levels

    def _get_random_price_near_mid(self, symbol: str) -> float:
        """Get random price near current mid price."""
        current_price = self.symbol_state[symbol]["current_price"]
        # ±0.1% of current price
        return current_price * random.uniform(0.999, 1.001)

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence += 1
        return self.sequence

    def _next_trade_id(self) -> int:
        """Get next trade ID."""
        self.trade_id += 1
        return self.trade_id


class MessageValidationFixtures:
    """Fixtures for message validation testing."""

    def __init__(self, config: MessageFactoryConfig = None):
        self.config = config or MessageFactoryConfig()
        self.factory = WebSocketMessageFactory(config)

    def generate_valid_message_samples(self) -> dict[str, list[dict[str, Any]]]:
        """Generate samples of valid messages for each type."""
        return {
            "subscription": [self.factory.generate_subscription_message()],
            "subscription_ack": [self.factory.generate_subscription_ack()],
            "snapshot": [
                self.factory.generate_snapshot_message(symbol)
                for symbol in self.config.symbols
            ],
            "l2update": [
                self.factory.generate_l2update_message(symbol)
                for symbol in self.config.symbols
            ],
            "trade": [
                self.factory.generate_trade_message(symbol)
                for symbol in self.config.symbols
            ],
            "ticker": [
                self.factory.generate_ticker_message(symbol)
                for symbol in self.config.symbols
            ],
            "heartbeat": [self.factory.generate_heartbeat_message()],
            "error": [
                self.factory.generate_error_message(error_type)
                for error_type in ["auth", "rate_limit", "invalid_symbol"]
            ],
        }

    def generate_invalid_message_samples(self) -> dict[str, list[Any]]:
        """Generate samples of invalid messages for validation testing."""
        return {
            "malformed_json": [
                '{"type": "snapshot", "invalid": json}',
                '{"incomplete": ',
                "not json at all",
                "",
            ],
            "missing_fields": [
                {"type": "snapshot"},  # Missing required fields
                {"product_id": "BTC-USD"},  # Missing type
                {"type": "l2update", "product_id": "BTC-USD"},  # Missing changes
            ],
            "invalid_types": [
                {"type": 123, "product_id": "BTC-USD"},
                {"type": "snapshot", "product_id": ["BTC-USD"]},
                {"type": "l2update", "changes": "not_a_list"},
            ],
            "invalid_values": [
                {"type": "snapshot", "product_id": "INVALID-SYMBOL"},
                {"type": "trade", "price": "-100", "size": "1.0"},
                {"type": "ticker", "timestamp": "invalid_timestamp"},
            ],
            "edge_cases": [
                None,
                {},
                [],
                {"type": None},
                {"type": ""},
                {"type": "unknown_type"},
            ],
        }


# Factory function for easy access
def create_websocket_message_suite(
    config: MessageFactoryConfig = None,
) -> dict[str, Any]:
    """Create complete WebSocket message test suite."""
    config = config or MessageFactoryConfig()

    factory = WebSocketMessageFactory(config)
    validation_fixtures = MessageValidationFixtures(config)

    return {
        # Basic messages
        "subscription": factory.generate_subscription_message(),
        "subscription_ack": factory.generate_subscription_ack(),
        "heartbeat": factory.generate_heartbeat_message(),
        # Market data messages
        "snapshots": [
            factory.generate_snapshot_message(symbol) for symbol in config.symbols
        ],
        "l2updates": [
            factory.generate_l2update_message(symbol) for symbol in config.symbols
        ],
        "trades": [factory.generate_trade_message(symbol) for symbol in config.symbols],
        "tickers": [
            factory.generate_ticker_message(symbol) for symbol in config.symbols
        ],
        # Error scenarios
        "errors": [
            factory.generate_error_message(error_type)
            for error_type in [
                "auth",
                "rate_limit",
                "invalid_symbol",
                "connection",
                "timeout",
            ]
        ],
        "malformed_messages": [factory.generate_malformed_message() for _ in range(10)],
        # Performance datasets
        "burst_messages": factory.generate_burst_messages(1000),
        "message_stream_sample": list(factory.generate_message_stream(10)),
        # Validation fixtures
        "valid_samples": validation_fixtures.generate_valid_message_samples(),
        "invalid_samples": validation_fixtures.generate_invalid_message_samples(),
        # Metadata
        "config": config,
        "generation_timestamp": datetime.now(UTC),
        "summary": {
            "total_message_types": 8,
            "error_scenarios": 5,
            "malformed_samples": 10,
            "burst_size": 1000,
            "symbols": len(config.symbols),
        },
    }


# Export main classes and functions
__all__ = [
    "MessageFactoryConfig",
    "MessageType",
    "MessageValidationFixtures",
    "WebSocketMessageFactory",
    "create_websocket_message_suite",
]
