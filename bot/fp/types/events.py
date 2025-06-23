"""Event sourcing types for the trading system.

This module defines immutable event types that capture all state changes
in the trading system, enabling event sourcing and audit trails.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, ClassVar
from uuid import UUID, uuid4

from bot.types import OrderSide, OrderType


@dataclass(frozen=True)
class EventMetadata:
    """Metadata common to all events."""

    version: str = "1.0.0"
    source: str = "trading-bot"
    correlation_id: UUID | None = None
    causation_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "version": self.version,
            "source": self.source,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventMetadata":
        """Create metadata from dictionary."""
        return cls(
            version=data["version"],
            source=data["source"],
            correlation_id=(
                UUID(data["correlation_id"]) if data.get("correlation_id") else None
            ),
            causation_id=(
                UUID(data["causation_id"]) if data.get("causation_id") else None
            ),
        )


@dataclass(frozen=True)
class TradingEvent:
    """Base class for all trading events."""

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: EventMetadata = field(default_factory=EventMetadata)

    # Event type for polymorphic deserialization
    event_type: ClassVar[str] = "TradingEvent"

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradingEvent":
        """Create event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
        )


@dataclass(frozen=True)
class MarketDataReceived(TradingEvent):
    """Event for market data snapshots."""

    event_type: ClassVar[str] = "MarketDataReceived"

    symbol: str
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    high_24h: Decimal
    low_24h: Decimal
    open_24h: Decimal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "symbol": self.symbol,
                "price": str(self.price),
                "volume": str(self.volume),
                "bid": str(self.bid),
                "ask": str(self.ask),
                "high_24h": str(self.high_24h),
                "low_24h": str(self.low_24h),
                "open_24h": str(self.open_24h),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketDataReceived":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            symbol=data["symbol"],
            price=Decimal(data["price"]),
            volume=Decimal(data["volume"]),
            bid=Decimal(data["bid"]),
            ask=Decimal(data["ask"]),
            high_24h=Decimal(data["high_24h"]),
            low_24h=Decimal(data["low_24h"]),
            open_24h=Decimal(data["open_24h"]),
        )


@dataclass(frozen=True)
class OrderPlaced(TradingEvent):
    """Event for order placement."""

    event_type: ClassVar[str] = "OrderPlaced"

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "GTC"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side.value,
                "order_type": self.order_type.value,
                "size": str(self.size),
                "price": str(self.price) if self.price else None,
                "stop_price": str(self.stop_price) if self.stop_price else None,
                "time_in_force": self.time_in_force,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderPlaced":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            size=Decimal(data["size"]),
            price=Decimal(data["price"]) if data.get("price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            time_in_force=data["time_in_force"],
        )


@dataclass(frozen=True)
class OrderFilled(TradingEvent):
    """Event for order execution."""

    event_type: ClassVar[str] = "OrderFilled"

    order_id: str
    symbol: str
    side: OrderSide
    fill_price: Decimal
    fill_size: Decimal
    fees: Decimal
    is_partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side.value,
                "fill_price": str(self.fill_price),
                "fill_size": str(self.fill_size),
                "fees": str(self.fees),
                "is_partial": self.is_partial,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderFilled":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            fill_price=Decimal(data["fill_price"]),
            fill_size=Decimal(data["fill_size"]),
            fees=Decimal(data["fees"]),
            is_partial=data["is_partial"],
        )


@dataclass(frozen=True)
class OrderCancelled(TradingEvent):
    """Event for order cancellation."""

    event_type: ClassVar[str] = "OrderCancelled"

    order_id: str
    symbol: str
    reason: str
    remaining_size: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "reason": self.reason,
                "remaining_size": (
                    str(self.remaining_size) if self.remaining_size else None
                ),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderCancelled":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            order_id=data["order_id"],
            symbol=data["symbol"],
            reason=data["reason"],
            remaining_size=(
                Decimal(data["remaining_size"]) if data.get("remaining_size") else None
            ),
        )


@dataclass(frozen=True)
class PositionOpened(TradingEvent):
    """Event for position opening."""

    event_type: ClassVar[str] = "PositionOpened"

    position_id: UUID
    symbol: str
    side: OrderSide
    size: Decimal
    entry_price: Decimal
    leverage: int
    initial_margin: Decimal

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "position_id": str(self.position_id),
                "symbol": self.symbol,
                "side": self.side.value,
                "size": str(self.size),
                "entry_price": str(self.entry_price),
                "leverage": self.leverage,
                "initial_margin": str(self.initial_margin),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionOpened":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            position_id=UUID(data["position_id"]),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            size=Decimal(data["size"]),
            entry_price=Decimal(data["entry_price"]),
            leverage=data["leverage"],
            initial_margin=Decimal(data["initial_margin"]),
        )


@dataclass(frozen=True)
class PositionClosed(TradingEvent):
    """Event for position closing."""

    event_type: ClassVar[str] = "PositionClosed"

    position_id: UUID
    symbol: str
    side: OrderSide
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    realized_pnl: Decimal
    fees: Decimal
    duration_seconds: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "position_id": str(self.position_id),
                "symbol": self.symbol,
                "side": self.side.value,
                "entry_price": str(self.entry_price),
                "exit_price": str(self.exit_price),
                "size": str(self.size),
                "realized_pnl": str(self.realized_pnl),
                "fees": str(self.fees),
                "duration_seconds": self.duration_seconds,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionClosed":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            position_id=UUID(data["position_id"]),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            entry_price=Decimal(data["entry_price"]),
            exit_price=Decimal(data["exit_price"]),
            size=Decimal(data["size"]),
            realized_pnl=Decimal(data["realized_pnl"]),
            fees=Decimal(data["fees"]),
            duration_seconds=data["duration_seconds"],
        )


@dataclass(frozen=True)
class StrategySignal(TradingEvent):
    """Event for strategy signals."""

    event_type: ClassVar[str] = "StrategySignal"

    signal_type: str  # "LONG", "SHORT", "CLOSE", "HOLD"
    confidence: float  # 0.0 to 1.0
    symbol: str
    current_price: Decimal
    indicators: dict[str, Any]
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "signal_type": self.signal_type,
                "confidence": self.confidence,
                "symbol": self.symbol,
                "current_price": str(self.current_price),
                "indicators": self.indicators,
                "reasoning": self.reasoning,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategySignal":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            signal_type=data["signal_type"],
            confidence=data["confidence"],
            symbol=data["symbol"],
            current_price=Decimal(data["current_price"]),
            indicators=data["indicators"],
            reasoning=data.get("reasoning"),
        )


# Event type registry for deserialization
EVENT_REGISTRY: dict[str, type[TradingEvent]] = {
    "TradingEvent": TradingEvent,
    "MarketDataReceived": MarketDataReceived,
    "OrderPlaced": OrderPlaced,
    "OrderFilled": OrderFilled,
    "OrderCancelled": OrderCancelled,
    "PositionOpened": PositionOpened,
    "PositionClosed": PositionClosed,
    "StrategySignal": StrategySignal,
}


def deserialize_event(data: dict[str, Any]) -> TradingEvent:
    """Deserialize an event from dictionary based on event_type."""
    event_type = data.get("event_type")
    if not event_type:
        raise ValueError("Missing event_type in event data")

    event_class = EVENT_REGISTRY.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.from_dict(data)
