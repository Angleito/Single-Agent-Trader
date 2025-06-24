"""Immutable market data types for functional programming approach."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


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
    
    def get_spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        if self.best_bid and self.best_ask:
            mid = (self.best_bid[0] + self.best_ask[0]) / 2
            if mid > 0:
                spread = self.best_ask[0] - self.best_bid[0]
                return (spread / mid) * 10000
        return Decimal(0)

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
    
    def get_depth_at_price(self, price: Decimal, side: str) -> Decimal:
        """Get total depth at or better than a specific price."""
        if side.upper() == "BUY":
            return sum(size for p, size in self.asks if p <= price)
        else:
            return sum(size for p, size in self.bids if p >= price)
    
    def get_volume_weighted_price(self, size: Decimal, side: str) -> Decimal | None:
        """Calculate volume-weighted average price for a given order size."""
        orders = self.asks if side.upper() == "BUY" else self.bids
        
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
    
    @property
    def volatility_24h(self) -> Decimal:
        """Calculate 24-hour volatility as percentage."""
        return abs(self.change_24h)
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid-point price (same as last_price for ticker)."""
        return self.last_price


@dataclass(frozen=True)
class Candle:
    """Immutable candle/kline data for functional processing"""
    
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    symbol: str | None = None
    interval: str | None = None
    trades_count: int | None = None
    
    def __post_init__(self) -> None:
        """Validate candle data consistency."""
        if any(price <= 0 for price in [self.open, self.high, self.low, self.close]):
            raise ValueError("All prices must be positive")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"High {self.high} must be >= all other prices")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Low {self.low} must be <= all other prices")
        if self.trades_count is not None and self.trades_count < 0:
            raise ValueError(f"Trades count cannot be negative, got {self.trades_count}")
    
    @property
    def price_range(self) -> Decimal:
        """Calculate the price range of the candle."""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish (green) candle."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish (red) candle."""
        return self.close < self.open
    
    @property
    def is_doji(self, threshold_pct: float = 0.1) -> bool:
        """Check if this is a doji candle (small body)."""
        if self.price_range == 0:
            return True
        body_size = abs(self.close - self.open)
        return (body_size / self.price_range) < (threshold_pct / 100)
    
    @property
    def body_size(self) -> Decimal:
        """Calculate the size of the candle body."""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> Decimal:
        """Calculate the upper shadow length."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> Decimal:
        """Calculate the lower shadow length."""
        return min(self.open, self.close) - self.low
    
    def vwap(self) -> Decimal:
        """Calculate volume-weighted average price."""
        return (self.high + self.low + self.close) / 3


@dataclass(frozen=True)
class Trade:
    """Immutable trade/tick data"""
    
    id: str
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: str  # "BUY" or "SELL"
    symbol: str | None = None
    exchange: str | None = None
    
    def __post_init__(self) -> None:
        """Validate trade data."""
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")
        if self.side not in ["BUY", "SELL"]:
            raise ValueError(f"Side must be BUY or SELL, got {self.side}")
    
    @property
    def value(self) -> Decimal:
        """Calculate trade value (price * size)."""
        return self.price * self.size
    
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side == "BUY"
    
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side == "SELL"


@dataclass(frozen=True)
class Subscription:
    """Immutable WebSocket subscription state"""
    
    symbol: str
    channels: list[str]
    active: bool
    created_at: datetime
    exchange: str | None = None
    subscription_id: str | None = None
    
    def is_subscribed_to(self, channel: str) -> bool:
        """Check if subscribed to a specific channel."""
        return channel in self.channels and self.active


class ConnectionStatus(str, Enum):
    """WebSocket connection status"""
    
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    AUTHENTICATED = "AUTHENTICATED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"
    RECONNECTING = "RECONNECTING"


@dataclass(frozen=True)
class ConnectionState:
    """Immutable WebSocket connection state"""
    
    status: ConnectionStatus
    url: str
    reconnect_attempts: int = 0
    last_error: str | None = None
    connected_at: datetime | None = None
    last_message_at: datetime | None = None
    
    def is_healthy(self, max_staleness_seconds: int = 30) -> bool:
        """Check if connection is healthy."""
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        if self.last_message_at is None:
            return False
        
        # Import timezone to ensure consistent timezone handling
        from datetime import UTC
        
        # Get current time with timezone info
        current_time = datetime.now(UTC)
        
        # Ensure last_message_at has timezone info
        if self.last_message_at.tzinfo is None:
            # If no timezone, assume UTC
            last_message = self.last_message_at.replace(tzinfo=UTC)
        else:
            last_message = self.last_message_at
        
        staleness = (current_time - last_message).total_seconds()
        return staleness <= max_staleness_seconds


@dataclass(frozen=True)
class DataQuality:
    """Immutable data quality metrics"""
    
    timestamp: datetime
    messages_received: int
    messages_processed: int
    validation_failures: int
    average_latency_ms: float | None = None
    
    @property
    def success_rate(self) -> float:
        """Calculate message processing success rate."""
        if self.messages_received == 0:
            return 100.0
        return (self.messages_processed / self.messages_received) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate validation error rate."""
        if self.messages_received == 0:
            return 0.0
        return (self.validation_failures / self.messages_received) * 100


@dataclass(frozen=True)
class WebSocketMessage:
    """Immutable WebSocket message"""
    
    channel: str
    timestamp: datetime
    data: dict[str, Any]
    message_id: str | None = None
    
    def __post_init__(self) -> None:
        """Validate message structure."""
        if not self.channel:
            raise ValueError("Channel cannot be empty")
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary")


@dataclass(frozen=True)
class TickerMessage:
    """Immutable ticker WebSocket message"""
    
    channel: str
    timestamp: datetime
    data: dict[str, Any]
    price: Decimal
    volume_24h: Decimal | None = None
    message_id: str | None = None
    
    def __post_init__(self) -> None:
        """Validate ticker message."""
        if not self.channel:
            raise ValueError("Channel cannot be empty")
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary")
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.volume_24h is not None and self.volume_24h < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume_24h}")


@dataclass(frozen=True)
class TradeMessage(WebSocketMessage):
    """Immutable trade WebSocket message"""
    
    trade_id: str | None = None
    price: Decimal | None = None
    size: Decimal | None = None
    side: str | None = None
    
    def __post_init__(self) -> None:
        """Validate trade message."""
        super().__post_init__()
        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.size is not None and self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")
        if self.side is not None and self.side not in ["BUY", "SELL"]:
            raise ValueError(f"Side must be BUY or SELL, got {self.side}")


@dataclass(frozen=True)
class OrderBookMessage(WebSocketMessage):
    """Immutable order book WebSocket message"""
    
    bids: list[tuple[Decimal, Decimal]] | None = None
    asks: list[tuple[Decimal, Decimal]] | None = None
    
    def __post_init__(self) -> None:
        """Validate order book message."""
        super().__post_init__()
        
        # Validate bid prices are descending
        if self.bids:
            for i in range(1, len(self.bids)):
                if self.bids[i][0] >= self.bids[i - 1][0]:
                    raise ValueError("Bid prices must be in descending order")
        
        # Validate ask prices are ascending
        if self.asks:
            for i in range(1, len(self.asks)):
                if self.asks[i][0] <= self.asks[i - 1][0]:
                    raise ValueError("Ask prices must be in ascending order")
        
        # Validate spread
        if self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            raise ValueError(f"Best bid {self.bids[0][0]} must be < best ask {self.asks[0][0]}")


@runtime_checkable
class StreamProcessor(Protocol):
    """Protocol for processing real-time market data streams."""
    
    def process_ticker(self, message: TickerMessage) -> None:
        """Process ticker message."""
        ...
    
    def process_trade(self, message: TradeMessage) -> None:
        """Process trade message."""
        ...
    
    def process_orderbook(self, message: OrderBookMessage) -> None:
        """Process order book message."""
        ...


@dataclass(frozen=True)
class MarketDataStream:
    """Immutable multi-exchange market data stream"""
    
    symbol: str
    exchanges: list[str]
    connection_states: dict[str, ConnectionState]
    data_quality: DataQuality
    active: bool
    
    def get_healthy_exchanges(self) -> list[str]:
        """Get list of exchanges with healthy connections."""
        return [
            exchange for exchange, state in self.connection_states.items()
            if state.is_healthy()
        ]
    
    @property
    def overall_health(self) -> bool:
        """Check overall stream health."""
        healthy_exchanges = self.get_healthy_exchanges()
        return len(healthy_exchanges) > 0 and self.data_quality.success_rate > 95.0


@dataclass(frozen=True)
class RealtimeUpdate:
    """Immutable real-time market data update"""
    
    symbol: str
    timestamp: datetime
    update_type: str  # 'ticker', 'trade', 'orderbook'
    data: dict[str, Any]
    exchange: str | None = None
    latency_ms: float | None = None
    
    def __post_init__(self) -> None:
        """Validate update data."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.update_type not in ['ticker', 'trade', 'orderbook', 'heartbeat']:
            raise ValueError(f"Invalid update type: {self.update_type}")
        if not isinstance(self.data, dict):
            raise ValueError("Data must be a dictionary")


@dataclass(frozen=True)
class AggregatedData:
    """Immutable aggregated market data over a time period"""
    
    symbol: str
    start_time: datetime
    end_time: datetime
    candles: list[Candle]
    trades: list[Trade]
    volume_total: Decimal
    trade_count: int
    
    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def average_trade_size(self) -> Decimal:
        """Calculate average trade size."""
        if self.trade_count == 0:
            return Decimal(0)
        return self.volume_total / Decimal(self.trade_count)
    
    @property
    def vwap(self) -> Decimal:
        """Calculate volume-weighted average price."""
        if not self.trades:
            return Decimal(0)
        
        total_value = sum(trade.price * trade.size for trade in self.trades)
        total_volume = sum(trade.size for trade in self.trades)
        
        if total_volume == 0:
            return Decimal(0)
        
        return total_value / total_volume
    
    def get_price_range(self) -> tuple[Decimal, Decimal]:
        """Get min and max prices during the period."""
        if not self.candles:
            return Decimal(0), Decimal(0)
        
        all_prices = []
        for candle in self.candles:
            all_prices.extend([candle.high, candle.low])
        
        return min(all_prices), max(all_prices)


# Enhanced MarketData class for backward compatibility
@dataclass(frozen=True)
class MarketData:
    """Simple market data point for functional programming.
    
    Supports both simple price/volume usage and OHLCV construction patterns
    for backward compatibility with different adapter layers.
    """
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    
    # OHLCV fields for backward compatibility (optional)
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    
    def __post_init__(self) -> None:
        """Validate market data and derive price from OHLCV if needed."""
        # If OHLCV fields provided but price is not meaningful, derive price from close
        if (self.open is not None and self.high is not None and 
            self.low is not None and self.close is not None):
            
            # Validate OHLCV relationships
            if any(p <= 0 for p in [self.open, self.high, self.low, self.close]):
                raise ValueError("All OHLCV prices must be positive")
            if self.high < max(self.open, self.close, self.low):
                raise ValueError(f"High {self.high} must be >= all other prices")
            if self.low > min(self.open, self.close, self.high):
                raise ValueError(f"Low {self.low} must be <= all other prices")
            
            # If price not explicitly set or is zero, use close price
            if self.price == Decimal(0):
                object.__setattr__(self, "price", self.close)
        
        # Validate required fields
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread"""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def has_ohlcv(self) -> bool:
        """Check if this market data has OHLCV fields."""
        return all(field is not None for field in [self.open, self.high, self.low, self.close])
    
    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3) if OHLCV available, else return price."""
        if self.has_ohlcv:
            return (self.high + self.low + self.close) / Decimal("3")
        return self.price
    
    @classmethod
    def from_ohlcv(
        cls,
        symbol: str,
        timestamp: datetime,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        bid: Optional[Decimal] = None,
        ask: Optional[Decimal] = None,
    ) -> "MarketData":
        """Create MarketData from OHLCV fields."""
        return cls(
            symbol=symbol,
            price=close,  # Use close as the primary price
            volume=volume,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            open=open,
            high=high,
            low=low,
            close=close,
        )
    
    @classmethod
    def from_price(
        cls,
        symbol: str,
        timestamp: datetime,
        price: Decimal,
        volume: Decimal,
        bid: Optional[Decimal] = None,
        ask: Optional[Decimal] = None,
    ) -> "MarketData":
        """Create MarketData from simple price/volume fields."""
        return cls(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
        )


# Type aliases for functional market data
MarketDataMessage = WebSocketMessage | TickerMessage | TradeMessage | OrderBookMessage
PriceData = Candle | Trade | Ticker
StreamingData = RealtimeUpdate | AggregatedData
