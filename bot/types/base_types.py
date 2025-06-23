"""
Base type definitions for the trading bot.

This module provides type aliases and TypedDict definitions to replace
generic Any types throughout the codebase for improved type safety.
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal, NotRequired, TypeAlias, TypedDict, Union

# Type aliases for common types
Price: TypeAlias = Decimal
Quantity: TypeAlias = Decimal
Percentage: TypeAlias = float
Timestamp: TypeAlias = datetime
Symbol: TypeAlias = str
OrderId: TypeAlias = str
AccountId: TypeAlias = str

# Type for validation details
ValidationDetail: TypeAlias = Union[str, int, float, bool, Decimal, datetime, list[str]]
ValidationDetails: TypeAlias = dict[str, ValidationDetail]

# Type for error details
ErrorDetail: TypeAlias = Union[str, int, float, bool, list[str], dict[str, str]]
ErrorDetails: TypeAlias = dict[str, ErrorDetail]


class MarketDataDict(TypedDict):
    """Type-safe dictionary for market data."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class DominanceCandleData(TypedDict):
    """Type-safe dictionary for dominance candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: NotRequired[float]
    rsi: NotRequired[float]
    trend_signal: NotRequired[str]


class IndicatorDict(TypedDict, total=False):
    """Type-safe dictionary for indicator values."""

    timestamp: datetime
    cipher_a_dot: float
    cipher_b_wave: float
    cipher_b_money_flow: float
    rsi: float
    ema_fast: float
    ema_slow: float
    vwap: float
    usdt_dominance: float
    usdc_dominance: float
    stablecoin_dominance: float
    dominance_trend: float
    dominance_rsi: float
    stablecoin_velocity: float
    market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    dominance_candles: list[DominanceCandleData]
    dominance_cipher_a_signal: int
    dominance_cipher_a_confidence: float
    dominance_cipher_b_signal: int
    dominance_cipher_b_confidence: float
    dominance_wt1: float
    dominance_wt2: float
    dominance_price_divergence: Literal[
        "NONE", "BULLISH", "BEARISH", "HIDDEN_BULLISH", "HIDDEN_BEARISH"
    ]
    dominance_sentiment: Literal[
        "STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"
    ]


class ValidationResult(TypedDict):
    """Type-safe dictionary for validation results."""

    valid: bool
    message: NotRequired[str]
    error: NotRequired[dict[str, str]]
    details: NotRequired[ValidationDetails]
    timestamp: NotRequired[str]


class BalanceValidationResult(ValidationResult):
    """Extended validation result for balance operations."""

    balance: NotRequired[Decimal]
    previous_balance: NotRequired[Decimal]
    operation_type: NotRequired[str]
    account_type: NotRequired[str]


class MarginValidationResult(ValidationResult):
    """Extended validation result for margin calculations."""

    balance: NotRequired[Decimal]
    used_margin: NotRequired[Decimal]
    available_margin: NotRequired[Decimal]
    margin_ratio_pct: NotRequired[float]
    position_value: NotRequired[Decimal]
    leverage: NotRequired[int]


class PositionValidationResult(ValidationResult):
    """Extended validation result for position validation."""

    reason: str
    severity: Literal["none", "low", "medium", "high", "critical"]
    position_side: NotRequired[str]
    position_size: NotRequired[Decimal]
    entry_price: NotRequired[Decimal]


class RiskMetricsDict(TypedDict):
    """Type-safe dictionary for risk metrics."""

    consecutive_losses: int
    rapid_loss: float
    api_failures: int
    position_errors: int
    margin_critical: float
    circuit_breaker_failures: int
    daily_loss_pct: float
    timestamp: datetime
    account_balance: NotRequired[float]


class ErrorContext(TypedDict):
    """Type-safe dictionary for error context."""

    error_type: str
    message: str
    component: NotRequired[str]
    operation: NotRequired[str]
    timestamp: str
    details: NotRequired[ErrorDetails]


class CircuitBreakerStatus(TypedDict):
    """Type-safe dictionary for circuit breaker status."""

    state: Literal["CLOSED", "OPEN", "HALF_OPEN"]
    failure_count: int
    failure_threshold: int
    last_failure: str | None
    timeout_remaining: float
    recent_failures: int


class EmergencyStopStatus(TypedDict):
    """Type-safe dictionary for emergency stop status."""

    is_stopped: bool
    stop_reason: str | None
    stop_timestamp: str | None
    trigger_count_24h: int
    last_trigger: str | None


class APIHealthStatus(TypedDict):
    """Type-safe dictionary for API health status."""

    consecutive_failures: int
    total_failures: int
    last_success: str | None
    health_score: float
