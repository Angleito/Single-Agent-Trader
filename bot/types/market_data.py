"""
Market data type definitions with comprehensive validation.

This module provides type-safe definitions for all market data structures including
OHLCV candles, order books, ticker data, trade executions, and market status.
All types include comprehensive validation to ensure data integrity.
"""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, NewType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Custom type definitions using NewType for additional type safety
Timestamp = NewType("Timestamp", int)  # Unix timestamp in milliseconds
Price = NewType("Price", Decimal)
Volume = NewType("Volume", Decimal)
OrderId = NewType("OrderId", str)
TradeId = NewType("TradeId", str)
ProductId = NewType("ProductId", str)

# Type aliases for clarity
PriceLevel = tuple[Price, Volume]  # Price and size at that level
Spread = Decimal  # Bid-ask spread


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    AUTHENTICATED = "AUTHENTICATED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"
    RECONNECTING = "RECONNECTING"


class MarketDataQuality(str, Enum):
    """Data quality indicators."""

    EXCELLENT = "EXCELLENT"  # Real-time, validated data
    GOOD = "GOOD"  # Minor delays or gaps
    DEGRADED = "DEGRADED"  # Significant delays or missing data
    STALE = "STALE"  # Data older than threshold
    INVALID = "INVALID"  # Failed validation


class TradeSide(str, Enum):
    """Trade side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class CandleData(BaseModel):
    """
    Type-safe OHLCV candle with comprehensive validation.

    Ensures price relationships are valid and volumes are non-negative.
    """

    timestamp: datetime = Field(description="Candle timestamp with timezone")
    open: Decimal = Field(gt=0, description="Opening price")
    high: Decimal = Field(gt=0, description="Highest price in period")
    low: Decimal = Field(gt=0, description="Lowest price in period")
    close: Decimal = Field(gt=0, description="Closing price")
    volume: Decimal = Field(ge=0, description="Trading volume in base currency")

    # Optional fields for enhanced data
    trades_count: int | None = Field(
        default=None, ge=0, description="Number of trades in this candle"
    )
    vwap: Decimal | None = Field(
        default=None, gt=0, description="Volume-weighted average price"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={Decimal: str, datetime: lambda v: v.isoformat()},
        str_strip_whitespace=True,
    )

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_have_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp has timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        return v

    @model_validator(mode="after")
    def validate_price_relationships(self) -> "CandleData":
        """
        Validate OHLCV price relationships.

        Ensures:
        - High is the highest price
        - Low is the lowest price
        - Open and Close are within High/Low range
        """
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(
                f"High price {self.high} must be >= all other prices. "
                f"Open: {self.open}, Close: {self.close}, Low: {self.low}"
            )

        if self.low > min(self.open, self.close, self.high):
            raise ValueError(
                f"Low price {self.low} must be <= all other prices. "
                f"Open: {self.open}, Close: {self.close}, High: {self.high}"
            )

        if self.open > self.high or self.open < self.low:
            raise ValueError(
                f"Open price {self.open} must be between Low {self.low} and High {self.high}"
            )

        if self.close > self.high or self.close < self.low:
            raise ValueError(
                f"Close price {self.close} must be between Low {self.low} and High {self.high}"
            )

        # Validate VWAP if provided
        if self.vwap is not None:
            if self.vwap > self.high or self.vwap < self.low:
                raise ValueError(
                    f"VWAP {self.vwap} must be between Low {self.low} and High {self.high}"
                )

        return self

    @field_validator("open", "high", "low", "close", "vwap")
    @classmethod
    def validate_price_precision(cls, v: Decimal | None) -> Decimal | None:
        """Validate price precision (max 8 decimal places)."""
        if v is not None:
            # Check for excessive decimal places
            if v.as_tuple().exponent < -8:
                raise ValueError(
                    f"Price {v} has too many decimal places (max 8 allowed)"
                )
        return v

    def get_price_range(self) -> Decimal:
        """Calculate the price range (high - low)."""
        return self.high - self.low

    def get_body_size(self) -> Decimal:
        """Calculate the candle body size (abs(close - open))."""
        return abs(self.close - self.open)

    def is_bullish(self) -> bool:
        """Check if this is a bullish candle."""
        return self.close > self.open

    def is_doji(self, threshold_pct: float = 0.1) -> bool:
        """
        Check if this is a doji candle.

        Args:
            threshold_pct: Body size threshold as percentage of range

        Returns:
            True if candle body is very small relative to range
        """
        if self.get_price_range() == 0:
            return True
        body_ratio = self.get_body_size() / self.get_price_range()
        return float(body_ratio) < threshold_pct


class TickerData(BaseModel):
    """Real-time ticker data with current market prices."""

    product_id: str = Field(description="Trading pair symbol")
    timestamp: datetime = Field(description="Data timestamp")
    price: Decimal = Field(gt=0, description="Current price")
    size: Decimal | None = Field(default=None, ge=0, description="Size of last trade")
    bid: Decimal | None = Field(default=None, gt=0, description="Best bid price")
    ask: Decimal | None = Field(default=None, gt=0, description="Best ask price")
    volume_24h: Decimal | None = Field(default=None, ge=0, description="24-hour volume")
    low_24h: Decimal | None = Field(default=None, gt=0, description="24-hour low price")
    high_24h: Decimal | None = Field(
        default=None, gt=0, description="24-hour high price"
    )
    low_52w: Decimal | None = Field(default=None, gt=0, description="52-week low price")
    high_52w: Decimal | None = Field(
        default=None, gt=0, description="52-week high price"
    )
    price_percent_chg_24h: float | None = Field(
        default=None, description="24-hour price change percentage"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_encoders={Decimal: str}
    )

    @model_validator(mode="after")
    def validate_bid_ask_spread(self) -> "TickerData":
        """Validate bid/ask relationship."""
        if self.bid is not None and self.ask is not None:
            if self.bid >= self.ask:
                raise ValueError(
                    f"Bid price {self.bid} must be less than ask price {self.ask}"
                )
        return self

    @model_validator(mode="after")
    def validate_24h_range(self) -> "TickerData":
        """Validate 24-hour price range."""
        if self.low_24h is not None and self.high_24h is not None:
            if self.low_24h > self.high_24h:
                raise ValueError(
                    f"24h low {self.low_24h} cannot be greater than 24h high {self.high_24h}"
                )
            # Current price should be within 24h range (with some tolerance for rapid moves)
            tolerance = Decimal("0.02")  # 2% tolerance
            adjusted_high = self.high_24h * (1 + tolerance)
            adjusted_low = self.low_24h * (1 - tolerance)
            if self.price > adjusted_high or self.price < adjusted_low:
                raise ValueError(
                    f"Current price {self.price} is outside 24h range "
                    f"[{self.low_24h}, {self.high_24h}] with tolerance"
                )
        return self

    def get_spread(self) -> Decimal | None:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    def get_spread_percentage(self) -> float | None:
        """Calculate spread as percentage of mid price."""
        if self.bid is not None and self.ask is not None:
            mid_price = (self.bid + self.ask) / 2
            spread = self.ask - self.bid
            return float(spread / mid_price * 100)
        return None


class OrderBookLevel(BaseModel):
    """Single level in the order book."""

    price: Decimal = Field(gt=0, description="Price level")
    size: Decimal = Field(gt=0, description="Total size at this level")
    order_count: int | None = Field(
        default=None, ge=1, description="Number of orders at this level"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OrderBook(BaseModel):
    """Order book snapshot with bids and asks."""

    product_id: str = Field(description="Trading pair symbol")
    timestamp: datetime = Field(description="Snapshot timestamp")
    bids: list[OrderBookLevel] = Field(
        description="Bid levels sorted by price descending"
    )
    asks: list[OrderBookLevel] = Field(
        description="Ask levels sorted by price ascending"
    )
    sequence: int | None = Field(
        default=None, description="Sequence number for order book updates"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_order_book_integrity(self) -> "OrderBook":
        """Validate order book structure and price levels."""
        # Validate bid prices are descending
        if len(self.bids) > 1:
            for i in range(1, len(self.bids)):
                if self.bids[i].price >= self.bids[i - 1].price:
                    raise ValueError("Bid prices must be in descending order")

        # Validate ask prices are ascending
        if len(self.asks) > 1:
            for i in range(1, len(self.asks)):
                if self.asks[i].price <= self.asks[i - 1].price:
                    raise ValueError("Ask prices must be in ascending order")

        # Validate bid/ask spread
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            if best_bid >= best_ask:
                raise ValueError(
                    f"Best bid {best_bid} must be less than best ask {best_ask}"
                )

        return self

    def get_best_bid(self) -> OrderBookLevel | None:
        """Get the best bid level."""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> OrderBookLevel | None:
        """Get the best ask level."""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> Decimal | None:
        """Calculate the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    def get_mid_price(self) -> Decimal | None:
        """Calculate the mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None

    def get_depth_imbalance(self, levels: int = 5) -> float | None:
        """
        Calculate order book imbalance.

        Args:
            levels: Number of levels to consider

        Returns:
            Imbalance ratio (-1 to 1, negative = more selling pressure)
        """
        bid_volume = sum(
            level.size for level in self.bids[:levels] if level.size is not None
        )
        ask_volume = sum(
            level.size for level in self.asks[:levels] if level.size is not None
        )

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return None

        return float((bid_volume - ask_volume) / total_volume)


class TradeExecution(BaseModel):
    """Individual trade execution data."""

    trade_id: str = Field(description="Unique trade identifier")
    product_id: str = Field(description="Trading pair symbol")
    timestamp: datetime = Field(description="Execution timestamp")
    price: Decimal = Field(gt=0, description="Execution price")
    size: Decimal = Field(gt=0, description="Trade size")
    side: TradeSide = Field(description="Trade side (BUY/SELL)")
    maker_order_id: str | None = Field(
        default=None, description="Maker order ID if available"
    )
    taker_order_id: str | None = Field(
        default=None, description="Taker order ID if available"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, use_enum_values=True, json_encoders={Decimal: str}
    )

    def get_trade_value(self) -> Decimal:
        """Calculate the trade value (price * size)."""
        return self.price * self.size


class MarketDepth(BaseModel):
    """Market depth information with liquidity metrics."""

    product_id: str = Field(description="Trading pair symbol")
    timestamp: datetime = Field(description="Snapshot timestamp")
    bid_depth: list[PriceLevel] = Field(
        description="Aggregated bid liquidity at price levels"
    )
    ask_depth: list[PriceLevel] = Field(
        description="Aggregated ask liquidity at price levels"
    )
    total_bid_volume: Decimal = Field(ge=0, description="Total bid volume")
    total_ask_volume: Decimal = Field(ge=0, description="Total ask volume")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_liquidity_at_distance(self, distance_pct: float) -> tuple[Decimal, Decimal]:
        """
        Get liquidity within a percentage distance from mid price.

        Args:
            distance_pct: Percentage distance from mid price

        Returns:
            Tuple of (bid_liquidity, ask_liquidity)
        """
        if not self.bid_depth or not self.ask_depth:
            return Decimal(0), Decimal(0)

        best_bid = self.bid_depth[0][0]
        best_ask = self.ask_depth[0][0]
        mid_price = (best_bid + best_ask) / 2

        distance = mid_price * Decimal(str(distance_pct / 100))
        min_bid_price = mid_price - distance
        max_ask_price = mid_price + distance

        bid_liquidity = sum(
            volume for price, volume in self.bid_depth if price >= min_bid_price
        )
        ask_liquidity = sum(
            volume for price, volume in self.ask_depth if price <= max_ask_price
        )

        # Ensure we return Decimal types
        return Decimal(bid_liquidity), Decimal(ask_liquidity)


class MarketDataStatus(BaseModel):
    """
    Comprehensive market data connection and quality status.

    Tracks connection state, data quality, and health metrics.
    """

    product_id: str = Field(description="Trading pair symbol")
    timestamp: datetime = Field(description="Status timestamp")
    connection_state: ConnectionState = Field(description="WebSocket connection state")
    data_quality: MarketDataQuality = Field(description="Overall data quality")

    # Connection metrics
    connected_since: datetime | None = Field(
        default=None, description="Connection establishment time"
    )
    reconnect_count: int = Field(
        default=0, ge=0, description="Number of reconnection attempts"
    )
    last_error: str | None = Field(
        default=None, description="Last error message if any"
    )

    # Data freshness
    last_ticker_update: datetime | None = Field(
        default=None, description="Last ticker data received"
    )
    last_trade_update: datetime | None = Field(
        default=None, description="Last trade data received"
    )
    last_orderbook_update: datetime | None = Field(
        default=None, description="Last order book update"
    )

    # Data statistics
    messages_received: int = Field(
        default=0, ge=0, description="Total messages received"
    )
    messages_processed: int = Field(
        default=0, ge=0, description="Successfully processed messages"
    )
    validation_failures: int = Field(default=0, ge=0, description="Failed validations")

    # Performance metrics
    avg_latency_ms: float | None = Field(
        default=None, ge=0, description="Average message latency in milliseconds"
    )
    message_rate_per_sec: float | None = Field(
        default=None, ge=0, description="Current message rate"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)

    def get_success_rate(self) -> float | None:
        """Calculate message processing success rate."""
        if self.messages_received == 0:
            return None
        return float(self.messages_processed / self.messages_received * 100)

    def is_healthy(self) -> bool:
        """Check if market data connection is healthy."""
        success_rate = self.get_success_rate()
        return (
            self.connection_state == ConnectionState.CONNECTED
            and self.data_quality
            in [MarketDataQuality.EXCELLENT, MarketDataQuality.GOOD]
            and success_rate is not None
            and success_rate > 95.0
        )

    def get_staleness_seconds(self) -> float | None:
        """Get seconds since last update."""
        latest_update = max(
            filter(
                None,
                [
                    self.last_ticker_update,
                    self.last_trade_update,
                    self.last_orderbook_update,
                ],
            ),
            default=None,
        )
        if latest_update is None:
            return None

        # Ensure both timestamps have timezone info
        if self.timestamp.tzinfo is None:
            current = self.timestamp.replace(tzinfo=UTC)
        else:
            current = self.timestamp

        if latest_update.tzinfo is None:
            latest = latest_update.replace(tzinfo=UTC)
        else:
            latest = latest_update

        return (current - latest).total_seconds()


class AggregatedMarketData(BaseModel):
    """Aggregated market data across multiple time periods."""

    product_id: str = Field(description="Trading pair symbol")
    timestamp: datetime = Field(description="Aggregation timestamp")

    # Current data
    current_price: Decimal = Field(gt=0, description="Current market price")
    current_volume: Decimal = Field(ge=0, description="Current period volume")

    # Time-based aggregations
    candles_1m: list[CandleData] | None = Field(
        default=None, description="1-minute candles"
    )
    candles_5m: list[CandleData] | None = Field(
        default=None, description="5-minute candles"
    )
    candles_1h: list[CandleData] | None = Field(
        default=None, description="1-hour candles"
    )

    # Volume profile
    volume_profile: list[tuple[Decimal, Decimal]] | None = Field(
        default=None, description="Price levels and volumes"
    )

    # Market microstructure
    average_trade_size: Decimal | None = Field(
        default=None, gt=0, description="Average trade size"
    )
    trade_count: int | None = Field(default=None, ge=0, description="Number of trades")
    buy_volume_ratio: float | None = Field(
        default=None, ge=0, le=1, description="Buy volume as ratio of total"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_volatility(self, period: str = "1h") -> float | None:
        """
        Calculate price volatility for a given period.

        Args:
            period: Time period (1m, 5m, 1h)

        Returns:
            Volatility as standard deviation of returns
        """
        candles_map = {
            "1m": self.candles_1m,
            "5m": self.candles_5m,
            "1h": self.candles_1h,
        }

        candles = candles_map.get(period)
        if not candles or len(candles) < 2:
            return None

        returns = []
        for i in range(1, len(candles)):
            if candles[i - 1].close > 0:
                ret = float(
                    (candles[i].close - candles[i - 1].close) / candles[i - 1].close
                )
                returns.append(ret)

        if not returns:
            return None

        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return float(variance**0.5)


# Type guards for runtime type checking
def is_valid_price(value: Any) -> bool:
    """Check if value is a valid price."""
    try:
        price = Decimal(str(value))
        exponent = price.as_tuple().exponent
        # Check for special values (infinity, NaN)
        if isinstance(exponent, str):
            return False
        return price > 0 and exponent >= -8
    except:
        return False


def is_valid_volume(value: Any) -> bool:
    """Check if value is a valid volume."""
    try:
        volume = Decimal(str(value))
        return volume >= 0
    except:
        return False


def is_valid_timestamp(value: Any) -> bool:
    """Check if value is a valid timestamp."""
    if isinstance(value, datetime):
        return value.tzinfo is not None
    if isinstance(value, (int, float)):
        return value > 0
    return False


# Aggregation helpers
def aggregate_candles(
    candles: list[CandleData], target_interval_minutes: int
) -> list[CandleData]:
    """
    Aggregate fine-grained candles into larger intervals.

    Args:
        candles: List of candles to aggregate
        target_interval_minutes: Target interval in minutes

    Returns:
        List of aggregated candles
    """
    if not candles:
        return []

    aggregated: list[CandleData] = []
    current_group: list[CandleData] = []

    for candle in sorted(candles, key=lambda x: x.timestamp):
        if not current_group:
            current_group.append(candle)
            continue

        # Check if candle belongs to current group
        first_timestamp = current_group[0].timestamp
        time_diff = (candle.timestamp - first_timestamp).total_seconds() / 60

        if time_diff < target_interval_minutes:
            current_group.append(candle)
        else:
            # Create aggregated candle from group
            if current_group:
                agg_candle = CandleData(
                    timestamp=current_group[0].timestamp,
                    open=current_group[0].open,
                    high=max(c.high for c in current_group),
                    low=min(c.low for c in current_group),
                    close=current_group[-1].close,
                    volume=Decimal(sum(c.volume for c in current_group)),
                    trades_count=sum(c.trades_count or 0 for c in current_group),
                )
                aggregated.append(agg_candle)

            # Start new group
            current_group = [candle]

    # Handle last group
    if current_group:
        agg_candle = CandleData(
            timestamp=current_group[0].timestamp,
            open=current_group[0].open,
            high=max(c.high for c in current_group),
            low=min(c.low for c in current_group),
            close=current_group[-1].close,
            volume=sum(c.volume for c in current_group),
            trades_count=sum(c.trades_count or 0 for c in current_group),
        )
        aggregated.append(agg_candle)

    return aggregated
