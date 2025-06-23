"""Trading action sum types using algebraic data types."""

from dataclasses import dataclass
from typing import Literal, Union
from uuid import uuid4


# Trade Signal ADT
@dataclass(frozen=True)
class Long:
    """Long position signal."""

    confidence: float
    size: float
    reason: str

    def __post_init__(self) -> None:
        """Validate signal parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
        if not 0 < self.size <= 1:
            raise ValueError(f"Size must be between 0 and 1, got {self.size}")


@dataclass(frozen=True)
class Short:
    """Short position signal."""

    confidence: float
    size: float
    reason: str

    def __post_init__(self) -> None:
        """Validate signal parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
        if not 0 < self.size <= 1:
            raise ValueError(f"Size must be between 0 and 1, got {self.size}")


@dataclass(frozen=True)
class Hold:
    """Hold current position signal."""

    reason: str


@dataclass(frozen=True)
class MarketMake:
    """Market making signal with bid/ask prices."""

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float

    def __post_init__(self) -> None:
        """Validate market making parameters."""
        if self.bid_price >= self.ask_price:
            raise ValueError(
                f"Bid price {self.bid_price} must be less than ask price {self.ask_price}"
            )
        if self.bid_size <= 0 or self.ask_size <= 0:
            raise ValueError("Bid and ask sizes must be positive")

    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Calculate the mid price."""
        return (self.bid_price + self.ask_price) / 2


# Type alias for all trade signals
TradeSignal = Union[Long, Short, Hold, MarketMake]


# Order Types ADT
@dataclass(frozen=True)
class LimitOrder:
    """Limit order with specific price."""

    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    order_id: str = ""

    def __post_init__(self) -> None:
        """Generate order ID if not provided and validate."""
        if not self.order_id:
            # Use object.__setattr__ to modify frozen dataclass
            object.__setattr__(self, "order_id", str(uuid4()))
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")

    @property
    def value(self) -> float:
        """Calculate order value."""
        return self.price * self.size


@dataclass(frozen=True)
class MarketOrder:
    """Market order executed at current market price."""

    symbol: str
    side: Literal["buy", "sell"]
    size: float
    order_id: str = ""

    def __post_init__(self) -> None:
        """Generate order ID if not provided and validate."""
        if not self.order_id:
            object.__setattr__(self, "order_id", str(uuid4()))
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")


@dataclass(frozen=True)
class StopOrder:
    """Stop order triggered at specific price."""

    symbol: str
    side: Literal["buy", "sell"]
    stop_price: float
    size: float
    order_id: str = ""

    def __post_init__(self) -> None:
        """Generate order ID if not provided and validate."""
        if not self.order_id:
            object.__setattr__(self, "order_id", str(uuid4()))
        if self.stop_price <= 0:
            raise ValueError(f"Stop price must be positive, got {self.stop_price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")


# Type alias for all order types
Order = Union[LimitOrder, MarketOrder, StopOrder]


# Type aliases for common patterns
BuyOrder = Union[LimitOrder, MarketOrder, StopOrder]  # Orders with side="buy"
SellOrder = Union[LimitOrder, MarketOrder, StopOrder]  # Orders with side="sell"
PendingOrder = Union[LimitOrder, StopOrder]  # Orders that wait for price
ImmediateOrder = MarketOrder  # Orders executed immediately


# Helper functions for pattern matching
def is_directional_signal(signal: TradeSignal) -> bool:
    """Check if signal is directional (Long or Short)."""
    return isinstance(signal, (Long, Short))


def is_pending_order(order: Order) -> bool:
    """Check if order is pending execution."""
    return isinstance(order, (LimitOrder, StopOrder))


def get_signal_confidence(signal: TradeSignal) -> float:
    """Get confidence from signal if available, 0 otherwise."""
    if isinstance(signal, (Long, Short)):
        return signal.confidence
    return 0.0


def get_signal_size(signal: TradeSignal) -> float:
    """Get position size from signal if available, 0 otherwise."""
    if isinstance(signal, (Long, Short)):
        return signal.size
    if isinstance(signal, MarketMake):
        return max(signal.bid_size, signal.ask_size)
    return 0.0


def signal_to_side(signal: TradeSignal) -> Literal["buy", "sell", "none"]:
    """Convert trade signal to order side."""
    if isinstance(signal, Long):
        return "buy"
    if isinstance(signal, Short):
        return "sell"
    return "none"


def create_market_order_from_signal(
    signal: TradeSignal, symbol: str, base_size: float
) -> MarketOrder | None:
    """Create a market order from a trade signal."""
    if isinstance(signal, Long):
        return MarketOrder(symbol=symbol, side="buy", size=base_size * signal.size)
    if isinstance(signal, Short):
        return MarketOrder(symbol=symbol, side="sell", size=base_size * signal.size)
    return None


def create_limit_orders_from_market_make(
    signal: MarketMake, symbol: str
) -> tuple[LimitOrder, LimitOrder]:
    """Create bid and ask limit orders from market making signal."""
    bid_order = LimitOrder(
        symbol=symbol, side="buy", price=signal.bid_price, size=signal.bid_size
    )
    ask_order = LimitOrder(
        symbol=symbol, side="sell", price=signal.ask_price, size=signal.ask_size
    )
    return bid_order, ask_order
