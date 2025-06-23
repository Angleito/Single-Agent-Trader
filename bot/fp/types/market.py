"""Immutable market data types for functional programming approach."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable representation of market state at a specific point in time.

    Attributes:
        timestamp: When the snapshot was taken
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        price: Current market price
        volume: Trading volume
        bid: Best bid price
        ask: Best ask price
        spread: Calculated spread between bid and ask
    """

    timestamp: datetime
    symbol: str
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    spread: Decimal = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived fields and validate data."""
        # Use object.__setattr__ to set frozen field
        object.__setattr__(self, "spread", self.ask - self.bid)

        # Validation
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("Bid and ask prices must be positive")
        if self.bid > self.ask:
            raise ValueError(f"Bid {self.bid} cannot be greater than ask {self.ask}")


@dataclass(frozen=True)
class OHLCV:
    """Immutable OHLCV (Open, High, Low, Close, Volume) candle data.

    Attributes:
        open: Opening price
        high: Highest price during period
        low: Lowest price during period
        close: Closing price
        volume: Total volume during period
        timestamp: Start time of the candle
    """

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate OHLCV data consistency."""
        if any(price <= 0 for price in [self.open, self.high, self.low, self.close]):
            raise ValueError("All prices must be positive")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"High {self.high} must be >= all other prices")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Low {self.low} must be <= all other prices")

    @property
    def price_range(self) -> Decimal:
        """Calculate the price range of the candle."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (green) candle."""
        return self.close > self.open

    @property
    def body_size(self) -> Decimal:
        """Calculate the size of the candle body."""
        return abs(self.close - self.open)


@dataclass(frozen=True)
class OrderBook:
    """Immutable order book representation with bids and asks.

    Attributes:
        bids: List of (price, size) tuples sorted by price descending
        asks: List of (price, size) tuples sorted by price ascending
        timestamp: When the order book snapshot was taken
    """

    bids: list[tuple[Decimal, Decimal]]
    asks: list[tuple[Decimal, Decimal]]
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate order book structure."""
        if not self.bids and not self.asks:
            raise ValueError("Order book cannot be empty")

        # Validate bid prices are descending
        for i in range(1, len(self.bids)):
            if self.bids[i][0] >= self.bids[i - 1][0]:
                raise ValueError("Bid prices must be in descending order")

        # Validate ask prices are ascending
        for i in range(1, len(self.asks)):
            if self.asks[i][0] <= self.asks[i - 1][0]:
                raise ValueError("Ask prices must be in ascending order")

        # Validate best bid < best ask
        if self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            raise ValueError(
                f"Best bid {self.bids[0][0]} must be < best ask {self.asks[0][0]}"
            )

    @property
    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        """Get the best (highest) bid price and size."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        """Get the best (lowest) ask price and size."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        """Calculate the mid-point price between best bid and ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid[0] + self.best_ask[0]) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        """Calculate the spread between best bid and ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask[0] - self.best_bid[0]
        return None

    @property
    def bid_depth(self) -> Decimal:
        """Calculate total bid side depth (sum of all bid sizes)."""
        return sum(size for _, size in self.bids)

    @property
    def ask_depth(self) -> Decimal:
        """Calculate total ask side depth (sum of all ask sizes)."""
        return sum(size for _, size in self.asks)

    def price_impact(self, side: str, size: Decimal) -> Decimal | None:
        """Calculate the average price impact of a market order.

        Args:
            side: 'buy' or 'sell'
            size: Order size to calculate impact for

        Returns:
            Average execution price, or None if insufficient liquidity
        """
        orders = self.asks if side == "buy" else self.bids
        remaining = size
        total_cost = Decimal(0)

        for price, available in orders:
            if remaining <= 0:
                break

            filled = min(remaining, available)
            total_cost += price * filled
            remaining -= filled

        if remaining > 0:
            return None  # Insufficient liquidity

        return total_cost / size if size > 0 else Decimal(0)


@dataclass(frozen=True)
class Ticker:
    """Immutable ticker information for a trading pair.

    Attributes:
        symbol: Trading pair symbol (e.g., 'BTC-USD')
        last_price: Most recent trade price
        volume_24h: 24-hour trading volume
        change_24h: 24-hour price change percentage
    """

    symbol: str
    last_price: Decimal
    volume_24h: Decimal
    change_24h: Decimal

    def __post_init__(self) -> None:
        """Validate ticker data."""
        if self.last_price <= 0:
            raise ValueError(f"Last price must be positive, got {self.last_price}")
        if self.volume_24h < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume_24h}")
        if self.change_24h < -100:
            raise ValueError(
                f"24h change cannot be less than -100%, got {self.change_24h}%"
            )

    @property
    def price_24h_ago(self) -> Decimal:
        """Calculate the price 24 hours ago based on current price and change."""
        return self.last_price / (1 + self.change_24h / 100)

    @property
    def is_positive_24h(self) -> bool:
        """Check if the 24-hour change is positive."""
        return self.change_24h > 0
