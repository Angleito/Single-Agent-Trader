"""Type definitions and data models for the AI Trading Bot."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TradeAction(BaseModel):
    """LLM output for trading decisions."""

    action: Literal["LONG", "SHORT", "CLOSE", "HOLD"] = Field(
        description="Trading action to take"
    )
    size_pct: float = Field(
        ge=0, le=100, description="Position size as percentage of available capital"
    )
    take_profit_pct: float = Field(
        ge=0, le=50.0, description="Take profit level as percentage (0 for HOLD/CLOSE)"
    )
    stop_loss_pct: float = Field(
        ge=0, le=50.0, description="Stop loss level as percentage (0 for HOLD/CLOSE)"
    )
    rationale: str = Field(
        max_length=500, description="Brief explanation for the trading decision"
    )

    @model_validator(mode="after")
    def validate_trade_action(self) -> TradeAction:
        """Validate the entire trade action for consistency."""
        action = self.action
        take_profit_pct = self.take_profit_pct
        stop_loss_pct = self.stop_loss_pct

        # For HOLD and CLOSE actions, allow 0 values
        if action in ["HOLD", "CLOSE"]:
            # Ensure percentages are reasonable for these actions
            if take_profit_pct is not None and (
                take_profit_pct < 0 or take_profit_pct > 50
            ):
                raise ValueError("Take profit percentage must be between 0 and 50")
            if stop_loss_pct is not None and (stop_loss_pct < 0 or stop_loss_pct > 50):
                raise ValueError("Stop loss percentage must be between 0 and 50")
        else:
            # For LONG/SHORT actions, require > 0
            if take_profit_pct is not None and (
                take_profit_pct <= 0 or take_profit_pct > 50
            ):
                raise ValueError(
                    "Take profit percentage must be between 0 and 50, and greater than 0 for trading actions"
                )
            if stop_loss_pct is not None and (stop_loss_pct <= 0 or stop_loss_pct > 50):
                raise ValueError(
                    "Stop loss percentage must be between 0 and 50, and greater than 0 for trading actions"
                )

        return self

    # Optional futures-specific fields
    leverage: int = Field(
        default=1, ge=1, le=100, description="Leverage for futures positions"
    )
    reduce_only: bool = Field(
        default=False, description="Whether this is a reduce-only order for futures"
    )


class AccountType(str, Enum):
    """Coinbase account type enumeration."""

    CFM = "CFM"  # Futures Commission Merchant (futures account)
    CBI = "CBI"  # Coinbase Inc (spot account)


class MarginHealthStatus(str, Enum):
    """Margin health status enumeration."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    LIQUIDATION_RISK = "LIQUIDATION_RISK"


class Position(BaseModel):
    """Current trading position information."""

    symbol: str
    side: Literal["LONG", "SHORT", "FLAT"]
    size: Decimal
    entry_price: Decimal | None = None
    unrealized_pnl: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    timestamp: datetime

    # Futures-specific fields
    is_futures: bool = False
    leverage: int | None = None
    margin_used: Decimal | None = None
    liquidation_price: Decimal | None = None
    margin_health: MarginHealthStatus | None = None

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class MarketData(BaseModel):
    """Market data container."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IndicatorData(BaseModel):
    """Technical indicator values."""

    timestamp: datetime
    cipher_a_dot: float | None = None
    cipher_b_wave: float | None = None
    cipher_b_money_flow: float | None = None
    rsi: float | None = None
    ema_fast: float | None = None
    ema_slow: float | None = None
    vwap: float | None = None

    # Stablecoin dominance indicators
    usdt_dominance: float | None = None
    usdc_dominance: float | None = None
    stablecoin_dominance: float | None = None
    dominance_trend: float | None = None  # 24h change in dominance
    dominance_rsi: float | None = None  # RSI of dominance
    stablecoin_velocity: float | None = None  # Trading volume / market cap
    market_sentiment: str | None = None  # BULLISH/BEARISH/NEUTRAL based on dominance

    # Dominance candlestick data for trend analysis
    dominance_candles: list[Any] | None = (
        None  # Will be list[DominanceCandleData] at runtime
    )

    model_config = ConfigDict(use_enum_values=True)


class MarketState(BaseModel):
    """Complete market state for LLM analysis."""

    symbol: str
    interval: str
    timestamp: datetime
    current_price: Decimal
    ohlcv_data: list[MarketData]
    indicators: IndicatorData
    current_position: Position

    # Stablecoin dominance data for sentiment analysis
    dominance_data: Any | None = None  # Will be DominanceData when available

    # Dominance candlesticks for technical analysis
    dominance_candles: list[Any] | None = (
        None  # Will be list[DominanceCandleData] at runtime
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class Order(BaseModel):
    """Trading order representation."""

    id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus
    timestamp: datetime
    filled_quantity: Decimal = Decimal(0)

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class MarginInfo(BaseModel):
    """Futures margin information."""

    total_margin: Decimal
    available_margin: Decimal
    used_margin: Decimal
    maintenance_margin: Decimal
    initial_margin: Decimal

    # Margin health metrics
    margin_ratio: float  # used_margin / total_margin
    health_status: MarginHealthStatus
    liquidation_threshold: Decimal

    # Intraday vs overnight requirements
    intraday_margin_requirement: Decimal
    overnight_margin_requirement: Decimal
    is_overnight_position: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FuturesAccountInfo(BaseModel):
    """Futures account information."""

    account_type: AccountType
    account_id: str
    currency: str = "USD"

    # Account balances
    cash_balance: Decimal
    futures_balance: Decimal
    total_balance: Decimal

    # Margin information
    margin_info: MarginInfo

    # Auto-transfer settings
    auto_cash_transfer_enabled: bool = True
    min_cash_transfer_amount: Decimal = Decimal(100)
    max_cash_transfer_amount: Decimal = Decimal(10000)

    # Position limits
    max_leverage: int = 20
    max_position_size: Decimal
    current_positions_count: int = 0

    timestamp: datetime

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class FuturesOrder(BaseModel):
    """Futures-specific order representation."""

    id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus
    timestamp: datetime
    filled_quantity: Decimal = Decimal(0)

    # Futures-specific fields
    leverage: int
    margin_required: Decimal
    reduce_only: bool = False  # True for closing positions
    post_only: bool = False  # True for maker-only orders
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class RiskMetrics(BaseModel):
    """Risk management metrics."""

    account_balance: Decimal
    available_margin: Decimal
    used_margin: Decimal
    daily_pnl: Decimal
    max_position_size: Decimal
    current_positions: int
    max_daily_loss_reached: bool = False

    # Futures-specific risk metrics
    futures_account_info: FuturesAccountInfo | None = None
    total_leverage_exposure: Decimal | None = None
    margin_health_status: MarginHealthStatus | None = None
    liquidation_risk_level: float | None = None  # 0.0 to 1.0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CashTransferRequest(BaseModel):
    """Request for cash transfer between spot and futures accounts."""

    from_account: AccountType
    to_account: AccountType
    amount: Decimal
    currency: str = "USD"
    reason: Literal["MARGIN_CALL", "MANUAL", "AUTO_REBALANCE"] = "MANUAL"

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


class FuturesMarketState(BaseModel):
    """Extended market state for futures trading."""

    symbol: str
    interval: str
    timestamp: datetime
    current_price: Decimal
    ohlcv_data: list[MarketData]
    indicators: IndicatorData
    current_position: Position

    # Futures-specific market state
    futures_account: FuturesAccountInfo | None = None
    margin_requirements: MarginInfo | None = None
    funding_rate: float | None = None
    next_funding_time: datetime | None = None
    open_interest: Decimal | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StablecoinDominance(BaseModel):
    """Stablecoin market dominance data."""

    timestamp: datetime
    stablecoin_dominance: float
    usdt_dominance: float
    usdc_dominance: float
    dominance_24h_change: float
    dominance_rsi: float

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Type aliases for common data structures
OHLCV = dict[str, Any]  # Raw OHLCV data from exchange
OHLCVData = MarketData  # Alias for backward compatibility
IndicatorValues = dict[str, float]  # Calculated indicator values
MarketSnapshot = dict[str, Any]  # Complete market snapshot
FuturesSnapshot = dict[str, Any]  # Futures-specific market snapshot

# Rebuild models to ensure they're fully defined with all dependencies
MarketData.model_rebuild()
IndicatorData.model_rebuild()
Position.model_rebuild()
MarketState.model_rebuild()
FuturesMarketState.model_rebuild()
StablecoinDominance.model_rebuild()
