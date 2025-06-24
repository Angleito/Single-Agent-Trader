"""Trading action sum types using algebraic data types."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Union
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


# Futures-specific Order Types
@dataclass(frozen=True)
class FuturesLimitOrder:
    """Futures limit order with leverage and margin requirements."""

    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    leverage: int
    margin_required: Decimal
    reduce_only: bool = False
    post_only: bool = False
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"
    order_id: str = ""

    def __post_init__(self) -> None:
        """Validate futures order and generate ID."""
        if not self.order_id:
            object.__setattr__(self, "order_id", str(uuid4()))
        if self.price <= 0:
            raise ValueError(f"Price must be positive: {self.price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive: {self.size}")
        if self.leverage < 1 or self.leverage > 100:
            raise ValueError(f"Leverage must be between 1 and 100: {self.leverage}")
        if self.margin_required < 0:
            raise ValueError(
                f"Margin required cannot be negative: {self.margin_required}"
            )

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the position."""
        return Decimal(str(self.price)) * Decimal(str(self.size))

    @property
    def position_value(self) -> Decimal:
        """Calculate leveraged position value."""
        return self.notional_value * self.leverage


@dataclass(frozen=True)
class FuturesMarketOrder:
    """Futures market order with leverage and margin requirements."""

    symbol: str
    side: Literal["buy", "sell"]
    size: float
    leverage: int
    margin_required: Decimal
    reduce_only: bool = False
    order_id: str = ""

    def __post_init__(self) -> None:
        """Validate futures market order and generate ID."""
        if not self.order_id:
            object.__setattr__(self, "order_id", str(uuid4()))
        if self.size <= 0:
            raise ValueError(f"Size must be positive: {self.size}")
        if self.leverage < 1 or self.leverage > 100:
            raise ValueError(f"Leverage must be between 1 and 100: {self.leverage}")
        if self.margin_required < 0:
            raise ValueError(
                f"Margin required cannot be negative: {self.margin_required}"
            )


@dataclass(frozen=True)
class FuturesStopOrder:
    """Futures stop order with leverage and margin requirements."""

    symbol: str
    side: Literal["buy", "sell"]
    stop_price: float
    size: float
    leverage: int
    margin_required: Decimal
    reduce_only: bool = False
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"
    order_id: str = ""

    def __post_init__(self) -> None:
        """Validate futures stop order and generate ID."""
        if not self.order_id:
            object.__setattr__(self, "order_id", str(uuid4()))
        if self.stop_price <= 0:
            raise ValueError(f"Stop price must be positive: {self.stop_price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive: {self.size}")
        if self.leverage < 1 or self.leverage > 100:
            raise ValueError(f"Leverage must be between 1 and 100: {self.leverage}")
        if self.margin_required < 0:
            raise ValueError(
                f"Margin required cannot be negative: {self.margin_required}"
            )


# Type alias for all order types
Order = Union[LimitOrder, MarketOrder, StopOrder]
FuturesOrder = Union[FuturesLimitOrder, FuturesMarketOrder, FuturesStopOrder]
AnyOrder = Union[Order, FuturesOrder]


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


# Account and Balance Types
@dataclass(frozen=True)
class AccountBalance:
    """Account balance information."""

    currency: str
    available: Decimal
    held: Decimal
    total: Decimal

    def __post_init__(self) -> None:
        """Validate balance values."""
        if self.total != self.available + self.held:
            raise ValueError(
                f"Total balance {self.total} must equal available {self.available} + held {self.held}"
            )


@dataclass(frozen=True)
class AccountType:
    """Immutable account type representation."""

    value: Literal["CFM", "CBI"]  # CFM: Futures Commission Merchant, CBI: Coinbase Inc

    def is_futures(self) -> bool:
        """Check if this is a futures account."""
        return self.value == "CFM"

    def is_spot(self) -> bool:
        """Check if this is a spot account."""
        return self.value == "CBI"


# Account type constants
CFM_ACCOUNT = AccountType("CFM")
CBI_ACCOUNT = AccountType("CBI")


@dataclass(frozen=True)
class MarginHealthStatus:
    """Immutable margin health status."""

    value: Literal["HEALTHY", "WARNING", "CRITICAL", "LIQUIDATION_RISK"]

    def is_healthy(self) -> bool:
        """Check if margin is healthy."""
        return self.value == "HEALTHY"

    def needs_attention(self) -> bool:
        """Check if margin needs attention."""
        return self.value in ["WARNING", "CRITICAL", "LIQUIDATION_RISK"]

    def is_critical(self) -> bool:
        """Check if margin is in critical state."""
        return self.value in ["CRITICAL", "LIQUIDATION_RISK"]


# Margin health constants
HEALTHY_MARGIN = MarginHealthStatus("HEALTHY")
WARNING_MARGIN = MarginHealthStatus("WARNING")
CRITICAL_MARGIN = MarginHealthStatus("CRITICAL")
LIQUIDATION_RISK_MARGIN = MarginHealthStatus("LIQUIDATION_RISK")


@dataclass(frozen=True)
class MarginInfo:
    """Immutable margin information for futures trading."""

    total_margin: Decimal
    available_margin: Decimal
    used_margin: Decimal
    maintenance_margin: Decimal
    initial_margin: Decimal
    health_status: MarginHealthStatus
    liquidation_threshold: Decimal
    intraday_margin_requirement: Decimal
    overnight_margin_requirement: Decimal
    is_overnight_position: bool = False

    def __post_init__(self) -> None:
        """Validate margin values."""
        if self.total_margin < 0:
            raise ValueError(f"Total margin cannot be negative: {self.total_margin}")
        if self.available_margin < 0:
            raise ValueError(
                f"Available margin cannot be negative: {self.available_margin}"
            )
        if self.used_margin < 0:
            raise ValueError(f"Used margin cannot be negative: {self.used_margin}")
        if self.total_margin != self.available_margin + self.used_margin:
            raise ValueError(
                f"Total margin {self.total_margin} must equal available {self.available_margin} + used {self.used_margin}"
            )

    @property
    def margin_ratio(self) -> float:
        """Calculate margin utilization ratio."""
        if self.total_margin == 0:
            return 0.0
        return float(self.used_margin / self.total_margin)

    @property
    def margin_usage_percentage(self) -> float:
        """Calculate margin usage as percentage."""
        return self.margin_ratio * 100

    @property
    def free_margin_percentage(self) -> float:
        """Calculate free margin as percentage."""
        return (1 - self.margin_ratio) * 100

    def can_open_position(self, required_margin: Decimal) -> bool:
        """Check if enough margin is available to open a position."""
        return self.available_margin >= required_margin

    def margin_call_distance(self) -> Decimal:
        """Calculate distance to margin call."""
        return self.available_margin


@dataclass(frozen=True)
class FuturesAccountBalance:
    """Immutable futures account balance information."""

    account_type: AccountType
    account_id: str
    currency: str
    cash_balance: Decimal
    futures_balance: Decimal
    total_balance: Decimal
    margin_info: MarginInfo
    auto_cash_transfer_enabled: bool = True
    min_cash_transfer_amount: Decimal = Decimal(100)
    max_cash_transfer_amount: Decimal = Decimal(10000)
    max_leverage: int = 20
    max_position_size: Decimal = Decimal(1000000)
    current_positions_count: int = 0
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Validate account balance."""
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now())

        if self.cash_balance < 0:
            raise ValueError(f"Cash balance cannot be negative: {self.cash_balance}")
        if self.futures_balance < 0:
            raise ValueError(
                f"Futures balance cannot be negative: {self.futures_balance}"
            )
        if not self.account_id:
            raise ValueError("Account ID cannot be empty")
        if self.max_leverage < 1 or self.max_leverage > 100:
            raise ValueError(
                f"Max leverage must be between 1 and 100: {self.max_leverage}"
            )

    @property
    def equity(self) -> Decimal:
        """Calculate total equity."""
        return self.cash_balance + self.futures_balance

    @property
    def buying_power(self) -> Decimal:
        """Calculate buying power based on available margin and leverage."""
        return self.margin_info.available_margin * self.max_leverage

    def can_transfer_cash(
        self, amount: Decimal, direction: Literal["to_futures", "to_spot"]
    ) -> bool:
        """Check if cash transfer is possible."""
        if not self.auto_cash_transfer_enabled:
            return False
        if (
            amount < self.min_cash_transfer_amount
            or amount > self.max_cash_transfer_amount
        ):
            return False

        if direction == "to_futures":
            return self.cash_balance >= amount
        # to_spot
        return self.futures_balance >= amount


@dataclass(frozen=True)
class CashTransferRequest:
    """Immutable cash transfer request between accounts."""

    from_account: AccountType
    to_account: AccountType
    amount: Decimal
    currency: str = "USD"
    reason: Literal["MARGIN_CALL", "MANUAL", "AUTO_REBALANCE"] = "MANUAL"
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Validate transfer request."""
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now())

        if self.amount <= 0:
            raise ValueError(f"Transfer amount must be positive: {self.amount}")
        if self.from_account.value == self.to_account.value:
            raise ValueError("Cannot transfer between same account types")

    @property
    def is_to_futures(self) -> bool:
        """Check if transfer is to futures account."""
        return self.to_account.is_futures()

    @property
    def is_to_spot(self) -> bool:
        """Check if transfer is to spot account."""
        return self.to_account.is_spot()

    @property
    def is_automated(self) -> bool:
        """Check if transfer is automated."""
        return self.reason in ["MARGIN_CALL", "AUTO_REBALANCE"]


# Order Status and Results
class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class OrderResult:
    """Result of order execution."""

    order_id: str
    status: OrderStatus
    filled_size: Decimal
    average_price: Decimal | None
    fees: Decimal
    created_at: datetime
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate order result."""
        if self.filled_size < 0:
            raise ValueError(f"Filled size cannot be negative: {self.filled_size}")
        if self.fees < 0:
            raise ValueError(f"Fees cannot be negative: {self.fees}")


# Position Management
@dataclass(frozen=True)
class Position:
    """Trading position information."""

    symbol: str
    side: Literal["long", "short"]
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    entry_time: datetime

    def __post_init__(self) -> None:
        """Validate position data."""
        if self.size <= 0:
            raise ValueError(f"Position size must be positive: {self.size}")
        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {self.entry_price}")
        if self.current_price <= 0:
            raise ValueError(f"Current price must be positive: {self.current_price}")

    @property
    def value(self) -> Decimal:
        """Calculate position value at current price."""
        return self.size * self.current_price

    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL as percentage of entry value."""
        entry_value = self.size * self.entry_price
        if entry_value == 0:
            return 0.0
        return float(self.unrealized_pnl / entry_value * 100)


# Enhanced Market Data Types
@dataclass(frozen=True)
class FunctionalMarketData:
    """Immutable market data with enhanced validation."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        """Validate OHLCV relationships and values."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.open <= 0:
            raise ValueError(f"Open price must be positive: {self.open}")
        if self.high <= 0:
            raise ValueError(f"High price must be positive: {self.high}")
        if self.low <= 0:
            raise ValueError(f"Low price must be positive: {self.low}")
        if self.close <= 0:
            raise ValueError(f"Close price must be positive: {self.close}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")

        # Validate OHLCV relationships
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(
                f"High {self.high} must be >= all other prices. "
                f"Open: {self.open}, Close: {self.close}, Low: {self.low}"
            )
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(
                f"Low {self.low} must be <= all other prices. "
                f"Open: {self.open}, Close: {self.close}, High: {self.high}"
            )
        if self.open > self.high or self.open < self.low:
            raise ValueError(
                f"Open {self.open} must be between Low {self.low} and High {self.high}"
            )
        if self.close > self.high or self.close < self.low:
            raise ValueError(
                f"Close {self.close} must be between Low {self.low} and High {self.high}"
            )

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / Decimal(3)

    @property
    def weighted_price(self) -> Decimal:
        """Calculate weighted price (OHLC/4)."""
        return (self.open + self.high + self.low + self.close) / Decimal(4)

    @property
    def price_range(self) -> Decimal:
        """Calculate price range (High - Low)."""
        return self.high - self.low

    @property
    def price_change(self) -> Decimal:
        """Calculate price change (Close - Open)."""
        return self.close - self.open

    @property
    def price_change_percentage(self) -> float:
        """Calculate price change as percentage."""
        if self.open == 0:
            return 0.0
        return float(self.price_change / self.open * 100)

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """Check if candle is a doji (open ≈ close)."""
        return abs(self.close - self.open) <= (
            self.price_range * Decimal("0.01")
        )  # 1% of range

    def update_timestamp(self, new_timestamp: datetime) -> "FunctionalMarketData":
        """Create new market data with updated timestamp."""
        return FunctionalMarketData(
            symbol=self.symbol,
            timestamp=new_timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass(frozen=True)
class FuturesMarketData:
    """Enhanced market data for futures trading."""

    base_data: FunctionalMarketData
    open_interest: Decimal
    funding_rate: float
    next_funding_time: datetime | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate futures-specific data."""
        if self.open_interest < 0:
            raise ValueError(f"Open interest cannot be negative: {self.open_interest}")
        if abs(self.funding_rate) > 1.0:  # 100% funding rate seems unrealistic
            raise ValueError(f"Funding rate seems unrealistic: {self.funding_rate}")
        if self.mark_price is not None and self.mark_price <= 0:
            raise ValueError(f"Mark price must be positive: {self.mark_price}")
        if self.index_price is not None and self.index_price <= 0:
            raise ValueError(f"Index price must be positive: {self.index_price}")

    @property
    def symbol(self) -> str:
        """Get symbol from base data."""
        return self.base_data.symbol

    @property
    def timestamp(self) -> datetime:
        """Get timestamp from base data."""
        return self.base_data.timestamp

    @property
    def close_price(self) -> Decimal:
        """Get close price from base data."""
        return self.base_data.close

    @property
    def effective_price(self) -> Decimal:
        """Get effective price (mark price if available, otherwise close)."""
        return self.mark_price if self.mark_price is not None else self.base_data.close

    @property
    def basis(self) -> Decimal:
        """Calculate basis (mark price - index price) if both available."""
        if self.mark_price is not None and self.index_price is not None:
            return self.mark_price - self.index_price
        return Decimal(0)

    @property
    def funding_rate_8h_annualized(self) -> float:
        """Convert 8-hour funding rate to annualized rate."""
        return self.funding_rate * 365 * 3  # 3 funding periods per day

    def with_updated_funding(
        self, new_rate: float, next_time: datetime
    ) -> "FuturesMarketData":
        """Create new futures data with updated funding information."""
        return FuturesMarketData(
            base_data=self.base_data,
            open_interest=self.open_interest,
            funding_rate=new_rate,
            next_funding_time=next_time,
            mark_price=self.mark_price,
            index_price=self.index_price,
        )


@dataclass(frozen=True)
class TradingIndicators:
    """Immutable trading indicators with validation."""

    timestamp: datetime
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    volume_sma: float | None = None
    atr: float | None = None

    # VuManChu Cipher indicators
    cipher_a_dot: float | None = None
    cipher_b_wave: float | None = None
    cipher_b_money_flow: float | None = None

    # Dominance indicators
    usdt_dominance: float | None = None
    usdc_dominance: float | None = None
    stablecoin_dominance: float | None = None
    dominance_trend: float | None = None
    dominance_rsi: float | None = None

    def __post_init__(self) -> None:
        """Validate indicator values."""
        # RSI should be between 0 and 100
        if self.rsi is not None and (self.rsi < 0 or self.rsi > 100):
            raise ValueError(f"RSI must be between 0 and 100: {self.rsi}")

        # Dominance values should be between 0 and 1 (or 0-100 if percentage)
        for field_name in ["usdt_dominance", "usdc_dominance", "stablecoin_dominance"]:
            value = getattr(self, field_name)
            if value is not None:
                if value < 0 or value > 100:  # Assuming percentage format
                    raise ValueError(f"{field_name} must be between 0 and 100: {value}")

        # Dominance RSI should be between 0 and 100
        if self.dominance_rsi is not None and (
            self.dominance_rsi < 0 or self.dominance_rsi > 100
        ):
            raise ValueError(
                f"Dominance RSI must be between 0 and 100: {self.dominance_rsi}"
            )

        # ATR should be positive
        if self.atr is not None and self.atr < 0:
            raise ValueError(f"ATR cannot be negative: {self.atr}")

    @property
    def has_vumanchu_signals(self) -> bool:
        """Check if VuManChu indicators are available."""
        return (
            self.cipher_a_dot is not None
            and self.cipher_b_wave is not None
            and self.cipher_b_money_flow is not None
        )

    @property
    def has_dominance_data(self) -> bool:
        """Check if dominance data is available."""
        return (
            self.stablecoin_dominance is not None and self.dominance_trend is not None
        )

    def rsi_signal(self) -> Literal["overbought", "oversold", "neutral"]:
        """Get RSI signal based on common thresholds."""
        if self.rsi is None:
            return "neutral"
        if self.rsi >= 70:
            return "overbought"
        if self.rsi <= 30:
            return "oversold"
        return "neutral"

    def dominance_signal(self) -> Literal["risk_on", "risk_off", "neutral"]:
        """Get market sentiment signal based on stablecoin dominance."""
        if self.stablecoin_dominance is None or self.dominance_trend is None:
            return "neutral"

        # High stablecoin dominance + increasing = risk off
        if self.stablecoin_dominance > 10 and self.dominance_trend > 0.5:
            return "risk_off"
        # Low stablecoin dominance + decreasing = risk on
        if self.stablecoin_dominance < 5 and self.dominance_trend < -0.5:
            return "risk_on"
        return "neutral"


# Risk Management Types
@dataclass(frozen=True)
class RiskLimits:
    """Immutable risk limits for trading operations."""

    max_position_size: Decimal
    max_daily_loss: Decimal
    max_drawdown_percentage: float
    max_leverage: int
    max_open_positions: int
    max_correlation_exposure: float
    stop_loss_percentage: float
    take_profit_percentage: float

    def __post_init__(self) -> None:
        """Validate risk limits."""
        if self.max_position_size <= 0:
            raise ValueError(
                f"Max position size must be positive: {self.max_position_size}"
            )
        if self.max_daily_loss <= 0:
            raise ValueError(f"Max daily loss must be positive: {self.max_daily_loss}")
        if self.max_drawdown_percentage <= 0 or self.max_drawdown_percentage > 100:
            raise ValueError(
                f"Max drawdown percentage must be between 0 and 100: {self.max_drawdown_percentage}"
            )
        if self.max_leverage < 1 or self.max_leverage > 100:
            raise ValueError(
                f"Max leverage must be between 1 and 100: {self.max_leverage}"
            )
        if self.max_open_positions < 1:
            raise ValueError(
                f"Max open positions must be positive: {self.max_open_positions}"
            )
        if self.max_correlation_exposure < 0 or self.max_correlation_exposure > 1:
            raise ValueError(
                f"Max correlation exposure must be between 0 and 1: {self.max_correlation_exposure}"
            )
        if self.stop_loss_percentage <= 0 or self.stop_loss_percentage > 100:
            raise ValueError(
                f"Stop loss percentage must be between 0 and 100: {self.stop_loss_percentage}"
            )
        if self.take_profit_percentage <= 0 or self.take_profit_percentage > 500:
            raise ValueError(
                f"Take profit percentage must be between 0 and 500: {self.take_profit_percentage}"
            )


@dataclass(frozen=True)
class RiskMetrics:
    """Immutable risk metrics calculation."""

    account_balance: Decimal
    available_margin: Decimal
    used_margin: Decimal
    daily_pnl: Decimal
    total_exposure: Decimal
    current_positions: int
    max_daily_loss_reached: bool = False
    value_at_risk_95: Decimal | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None

    def __post_init__(self) -> None:
        """Validate risk metrics."""
        if self.account_balance < 0:
            raise ValueError(
                f"Account balance cannot be negative: {self.account_balance}"
            )
        if self.available_margin < 0:
            raise ValueError(
                f"Available margin cannot be negative: {self.available_margin}"
            )
        if self.used_margin < 0:
            raise ValueError(f"Used margin cannot be negative: {self.used_margin}")
        if self.total_exposure < 0:
            raise ValueError(
                f"Total exposure cannot be negative: {self.total_exposure}"
            )
        if self.current_positions < 0:
            raise ValueError(
                f"Current positions cannot be negative: {self.current_positions}"
            )

    @property
    def margin_utilization(self) -> float:
        """Calculate margin utilization ratio."""
        total_margin = self.available_margin + self.used_margin
        if total_margin == 0:
            return 0.0
        return float(self.used_margin / total_margin)

    @property
    def exposure_ratio(self) -> float:
        """Calculate exposure to account balance ratio."""
        if self.account_balance == 0:
            return 0.0
        return float(self.total_exposure / self.account_balance)

    @property
    def daily_return_percentage(self) -> float:
        """Calculate daily return as percentage."""
        if self.account_balance == 0:
            return 0.0
        return float(self.daily_pnl / self.account_balance * 100)

    def is_within_risk_limits(self, limits: RiskLimits) -> bool:
        """Check if current metrics are within risk limits."""
        return (
            self.total_exposure <= limits.max_position_size
            and abs(self.daily_pnl) <= limits.max_daily_loss
            and self.current_positions <= limits.max_open_positions
            and not self.max_daily_loss_reached
        )

    def risk_score(self) -> float:
        """Calculate overall risk score (0-100, higher is riskier)."""
        score = 0.0

        # Margin utilization component (30% weight)
        score += self.margin_utilization * 30

        # Exposure ratio component (40% weight)
        score += min(self.exposure_ratio, 2.0) * 20  # Cap at 2x leverage

        # Daily loss component (30% weight)
        if self.account_balance > 0:
            daily_loss_ratio = abs(
                float(min(self.daily_pnl, Decimal(0)) / self.account_balance)
            )
            score += min(daily_loss_ratio * 100, 30)  # Cap at 30 points

        return min(score, 100.0)


# Comprehensive Market State
@dataclass(frozen=True)
class FunctionalMarketState:
    """Immutable comprehensive market state for trading decisions."""

    symbol: str
    timestamp: datetime
    market_data: FunctionalMarketData
    indicators: TradingIndicators
    position: Position
    futures_data: FuturesMarketData | None = None
    account_balance: FuturesAccountBalance | None = None
    risk_metrics: RiskMetrics | None = None

    def __post_init__(self) -> None:
        """Validate market state consistency."""
        if self.symbol != self.market_data.symbol:
            raise ValueError(
                f"Symbol mismatch: state={self.symbol}, market_data={self.market_data.symbol}"
            )
        if self.symbol != self.position.symbol:
            raise ValueError(
                f"Symbol mismatch: state={self.symbol}, position={self.position.symbol}"
            )
        if self.futures_data and self.symbol != self.futures_data.symbol:
            raise ValueError(
                f"Symbol mismatch: state={self.symbol}, futures_data={self.futures_data.symbol}"
            )

    @property
    def current_price(self) -> Decimal:
        """Get current effective price."""
        if self.futures_data:
            return self.futures_data.effective_price
        return self.market_data.close

    @property
    def is_futures_market(self) -> bool:
        """Check if this is a futures market."""
        return self.futures_data is not None

    @property
    def has_position(self) -> bool:
        """Check if there's an active position."""
        return self.position.side != "FLAT" and self.position.size > 0

    @property
    def position_value(self) -> Decimal:
        """Calculate current position value."""
        if not self.has_position:
            return Decimal(0)
        return self.position.size * self.current_price

    def with_updated_price(self, new_price: Decimal) -> "FunctionalMarketState":
        """Create new market state with updated price."""
        # Update market data
        updated_market_data = FunctionalMarketData(
            symbol=self.market_data.symbol,
            timestamp=datetime.now(),
            open=self.market_data.open,
            high=max(self.market_data.high, new_price),
            low=min(self.market_data.low, new_price),
            close=new_price,
            volume=self.market_data.volume,
        )

        # Update position PnL if there's a position
        updated_position = self.position
        if self.has_position and self.position.entry_price:
            price_diff = new_price - self.position.entry_price
            if self.position.side == "LONG":
                unrealized_pnl = self.position.size * price_diff
            else:  # SHORT
                unrealized_pnl = self.position.size * (-price_diff)

            updated_position = Position(
                symbol=self.position.symbol,
                side=self.position.side,
                size=self.position.size,
                entry_price=self.position.entry_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=self.position.realized_pnl,
                timestamp=datetime.now(),
            )

        return FunctionalMarketState(
            symbol=self.symbol,
            timestamp=datetime.now(),
            market_data=updated_market_data,
            indicators=self.indicators,
            position=updated_position,
            futures_data=self.futures_data,
            account_balance=self.account_balance,
            risk_metrics=self.risk_metrics,
        )


# Type Converters for Backward Compatibility
def convert_pydantic_to_functional_market_data(
    pydantic_data: Any,
) -> FunctionalMarketData:
    """Convert Pydantic MarketData to functional equivalent."""
    return FunctionalMarketData(
        symbol=pydantic_data.symbol,
        timestamp=pydantic_data.timestamp,
        open=pydantic_data.open,
        high=pydantic_data.high,
        low=pydantic_data.low,
        close=pydantic_data.close,
        volume=pydantic_data.volume,
    )


def convert_pydantic_to_functional_position(pydantic_position: Any) -> Position:
    """Convert Pydantic Position to functional equivalent."""
    return Position(
        symbol=pydantic_position.symbol,
        side=pydantic_position.side,
        size=pydantic_position.size,
        entry_price=pydantic_position.entry_price,
        unrealized_pnl=pydantic_position.unrealized_pnl,
        realized_pnl=pydantic_position.realized_pnl,
        timestamp=pydantic_position.timestamp,
    )


def convert_functional_to_pydantic_position(functional_position: Position) -> Any:
    """Convert functional Position to Pydantic equivalent."""
    # Import here to avoid circular dependencies
    from bot.trading_types import Position as PydanticPosition

    return PydanticPosition(
        symbol=functional_position.symbol,
        side=functional_position.side,
        size=functional_position.size,
        entry_price=functional_position.entry_price,
        unrealized_pnl=functional_position.unrealized_pnl,
        realized_pnl=functional_position.realized_pnl,
        timestamp=functional_position.timestamp,
    )


def convert_order_to_functional(order_data: Any) -> Order:
    """Convert various order formats to functional equivalent."""
    if hasattr(order_data, "type"):
        order_type = order_data.type.upper()
    else:
        order_type = getattr(order_data, "order_type", "MARKET").upper()

    if order_type == "LIMIT":
        return LimitOrder(
            symbol=order_data.symbol,
            side=order_data.side.lower(),
            price=float(order_data.price),
            size=float(
                order_data.quantity
                if hasattr(order_data, "quantity")
                else order_data.size
            ),
            order_id=getattr(order_data, "id", ""),
        )
    if order_type == "STOP":
        return StopOrder(
            symbol=order_data.symbol,
            side=order_data.side.lower(),
            stop_price=float(order_data.stop_price),
            size=float(
                order_data.quantity
                if hasattr(order_data, "quantity")
                else order_data.size
            ),
            order_id=getattr(order_data, "id", ""),
        )
    # MARKET
    return MarketOrder(
        symbol=order_data.symbol,
        side=order_data.side.lower(),
        size=float(
            order_data.quantity if hasattr(order_data, "quantity") else order_data.size
        ),
        order_id=getattr(order_data, "id", ""),
    )


def convert_trade_signal_to_orders(
    signal: TradeSignal, symbol: str, current_price: float
) -> list[Order]:
    """Convert functional trade signal to executable orders."""
    orders = []

    if isinstance(signal, Long):
        # Create market buy order
        order = MarketOrder(symbol=symbol, side="buy", size=signal.size)
        orders.append(order)

        # Add stop loss and take profit if needed
        if current_price > 0:
            stop_price = current_price * 0.95  # 5% stop loss
            take_profit_price = current_price * 1.10  # 10% take profit

            stop_order = StopOrder(
                symbol=symbol, side="sell", stop_price=stop_price, size=signal.size
            )

            profit_order = LimitOrder(
                symbol=symbol, side="sell", price=take_profit_price, size=signal.size
            )

            orders.extend([stop_order, profit_order])

    elif isinstance(signal, Short):
        # Create market sell order
        order = MarketOrder(symbol=symbol, side="sell", size=signal.size)
        orders.append(order)

        # Add stop loss and take profit if needed
        if current_price > 0:
            stop_price = current_price * 1.05  # 5% stop loss
            take_profit_price = current_price * 0.90  # 10% take profit

            stop_order = StopOrder(
                symbol=symbol, side="buy", stop_price=stop_price, size=signal.size
            )

            profit_order = LimitOrder(
                symbol=symbol, side="buy", price=take_profit_price, size=signal.size
            )

            orders.extend([stop_order, profit_order])

    elif isinstance(signal, MarketMake):
        # Create bid and ask orders
        bid_order, ask_order = create_limit_orders_from_market_make(signal, symbol)
        orders.extend([bid_order, ask_order])

    return orders


# Factory Functions for Common Patterns
def create_conservative_risk_limits() -> RiskLimits:
    """Create conservative risk limits for safe trading."""
    return RiskLimits(
        max_position_size=Decimal(10000),
        max_daily_loss=Decimal(500),
        max_drawdown_percentage=10.0,
        max_leverage=3,
        max_open_positions=5,
        max_correlation_exposure=0.3,
        stop_loss_percentage=5.0,
        take_profit_percentage=15.0,
    )


def create_aggressive_risk_limits() -> RiskLimits:
    """Create aggressive risk limits for high-risk trading."""
    return RiskLimits(
        max_position_size=Decimal(50000),
        max_daily_loss=Decimal(2000),
        max_drawdown_percentage=25.0,
        max_leverage=10,
        max_open_positions=15,
        max_correlation_exposure=0.8,
        stop_loss_percentage=10.0,
        take_profit_percentage=30.0,
    )


def create_default_margin_info() -> MarginInfo:
    """Create default margin info for testing."""
    return MarginInfo(
        total_margin=Decimal(10000),
        available_margin=Decimal(8000),
        used_margin=Decimal(2000),
        maintenance_margin=Decimal(1000),
        initial_margin=Decimal(1500),
        health_status=HEALTHY_MARGIN,
        liquidation_threshold=Decimal(500),
        intraday_margin_requirement=Decimal(1000),
        overnight_margin_requirement=Decimal(1500),
    )


# Type aliases for backward compatibility and learning module
MarketData = FunctionalMarketData  # Alias for learning module compatibility
MarketState = FunctionalMarketState  # Alias for consistency


@dataclass(frozen=True)
class TradeDecision:
    """Immutable trade decision with reasoning."""

    signal: TradeSignal
    symbol: str
    timestamp: datetime
    confidence: float
    reasoning: str
    market_context: dict[str, Any] = None
    risk_assessment: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Validate trade decision."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not self.reasoning:
            raise ValueError("Reasoning cannot be empty")
        if self.market_context is None:
            object.__setattr__(self, "market_context", {})
        if self.risk_assessment is None:
            object.__setattr__(self, "risk_assessment", {})


@dataclass(frozen=True)
class TradingParams:
    """Parameters for trading operations."""

    max_position_size: float = 0.25  # Maximum position size as fraction of portfolio
    min_confidence: float = 0.7  # Minimum confidence threshold for trades
    stop_loss_pct: float = 0.02  # Stop loss percentage
    take_profit_pct: float = 0.04  # Take profit percentage
    max_leverage: int = 5  # Maximum leverage allowed
    min_trade_amount: float = 10.0  # Minimum trade amount
    risk_per_trade: float = 0.02  # Risk per trade as fraction of portfolio

    def __post_init__(self) -> None:
        """Validate trading parameters."""
        if not 0 < self.max_position_size <= 1:
            raise ValueError(
                f"Max position size must be between 0 and 1: {self.max_position_size}"
            )
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(
                f"Min confidence must be between 0 and 1: {self.min_confidence}"
            )
        if self.stop_loss_pct <= 0:
            raise ValueError(
                f"Stop loss percentage must be positive: {self.stop_loss_pct}"
            )
        if self.take_profit_pct <= 0:
            raise ValueError(
                f"Take profit percentage must be positive: {self.take_profit_pct}"
            )
        if self.max_leverage < 1:
            raise ValueError(f"Max leverage must be at least 1: {self.max_leverage}")
        if self.min_trade_amount <= 0:
            raise ValueError(
                f"Min trade amount must be positive: {self.min_trade_amount}"
            )
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(
                f"Risk per trade must be between 0 and 1: {self.risk_per_trade}"
            )


@dataclass(frozen=True)
class MarketState:
    """Current market state for trading decisions."""

    symbol: str
    current_price: float
    timestamp: datetime
    volume: float
    price_change_24h: float = 0.0
    volatility: float = 0.0
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        """Validate market state."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.current_price <= 0:
            raise ValueError(f"Current price must be positive: {self.current_price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")
        if self.bid is not None and self.bid <= 0:
            raise ValueError(f"Bid must be positive: {self.bid}")
        if self.ask is not None and self.ask <= 0:
            raise ValueError(f"Ask must be positive: {self.ask}")
        if self.bid is not None and self.ask is not None and self.bid >= self.ask:
            raise ValueError(f"Bid {self.bid} must be less than ask {self.ask}")
