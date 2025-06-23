"""
Risk management types for functional programming architecture.

This module defines immutable data structures and pure functions for risk management,
including risk parameters, limits, metrics, margin information, and alerts.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Union


@dataclass(frozen=True)
class RiskParameters:
    """Risk management parameters for trading."""

    max_position_size: Decimal  # Maximum position size as percentage of portfolio
    max_leverage: Decimal  # Maximum allowed leverage
    stop_loss_pct: Decimal  # Stop loss percentage from entry
    take_profit_pct: Decimal  # Take profit percentage from entry


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits for risk management."""

    daily_loss_limit: Decimal  # Maximum daily loss in USD
    position_limit: int  # Maximum number of concurrent positions
    margin_requirement: Decimal  # Minimum margin requirement percentage


@dataclass(frozen=True)
class RiskMetrics:
    """Current risk metrics and statistics."""

    current_exposure: Decimal  # Total exposure in USD
    var_95: Decimal  # Value at Risk at 95% confidence
    max_drawdown: Decimal  # Maximum drawdown percentage
    sharpe_ratio: Decimal  # Risk-adjusted return metric


@dataclass(frozen=True)
class MarginInfo:
    """Margin and balance information."""

    total_balance: Decimal  # Total account balance
    used_margin: Decimal  # Margin currently in use
    free_margin: Decimal  # Available margin
    margin_ratio: Decimal  # Used margin / total balance ratio


# Risk Alert Types (Sum Type)
class RiskAlertType(Enum):
    """Types of risk alerts."""

    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    MARGIN_CALL = "margin_call"
    DAILY_LOSS_LIMIT = "daily_loss_limit"


@dataclass(frozen=True)
class PositionLimitExceeded:
    """Alert when position limit is exceeded."""

    current_positions: int
    limit: int
    alert_type: RiskAlertType = RiskAlertType.POSITION_LIMIT_EXCEEDED


@dataclass(frozen=True)
class MarginCall:
    """Alert when margin ratio is too high."""

    margin_ratio: Decimal
    threshold: Decimal
    alert_type: RiskAlertType = RiskAlertType.MARGIN_CALL


@dataclass(frozen=True)
class DailyLossLimit:
    """Alert when daily loss limit is reached."""

    current_loss: Decimal
    limit: Decimal
    alert_type: RiskAlertType = RiskAlertType.DAILY_LOSS_LIMIT


# Union type for all risk alerts
RiskAlert = Union[PositionLimitExceeded, MarginCall, DailyLossLimit]


# Pure Risk Calculation Functions


def calculate_position_size(
    balance: Decimal, risk_per_trade: Decimal, stop_loss_pct: Decimal
) -> Decimal:
    """
    Calculate position size based on risk parameters.

    Args:
        balance: Total account balance
        risk_per_trade: Risk percentage per trade
        stop_loss_pct: Stop loss percentage

    Returns:
        Position size in base currency
    """
    if stop_loss_pct <= Decimal(0):
        return Decimal(0)

    risk_amount = balance * risk_per_trade / Decimal(100)
    position_size = risk_amount / (stop_loss_pct / Decimal(100))
    return position_size


def calculate_margin_ratio(used_margin: Decimal, total_balance: Decimal) -> Decimal:
    """
    Calculate margin ratio.

    Args:
        used_margin: Currently used margin
        total_balance: Total account balance

    Returns:
        Margin ratio as decimal
    """
    if total_balance <= Decimal(0):
        return Decimal(0)

    return used_margin / total_balance


def calculate_free_margin(total_balance: Decimal, used_margin: Decimal) -> Decimal:
    """
    Calculate available free margin.

    Args:
        total_balance: Total account balance
        used_margin: Currently used margin

    Returns:
        Free margin amount
    """
    return total_balance - used_margin


def calculate_max_position_value(
    balance: Decimal, max_position_size_pct: Decimal, leverage: Decimal
) -> Decimal:
    """
    Calculate maximum allowed position value.

    Args:
        balance: Total account balance
        max_position_size_pct: Maximum position size as percentage
        leverage: Trading leverage

    Returns:
        Maximum position value
    """
    max_position = balance * max_position_size_pct / Decimal(100)
    return max_position * leverage


def calculate_required_margin(
    position_size: Decimal, entry_price: Decimal, leverage: Decimal
) -> Decimal:
    """
    Calculate required margin for a position.

    Args:
        position_size: Size of the position
        entry_price: Entry price
        leverage: Trading leverage

    Returns:
        Required margin amount
    """
    if leverage <= Decimal(0):
        return Decimal(0)

    position_value = position_size * entry_price
    return position_value / leverage


def calculate_stop_loss_price(
    entry_price: Decimal, stop_loss_pct: Decimal, is_long: bool
) -> Decimal:
    """
    Calculate stop loss price.

    Args:
        entry_price: Position entry price
        stop_loss_pct: Stop loss percentage
        is_long: True for long position, False for short

    Returns:
        Stop loss price
    """
    stop_loss_amount = entry_price * stop_loss_pct / Decimal(100)

    if is_long:
        return entry_price - stop_loss_amount
    return entry_price + stop_loss_amount


def calculate_take_profit_price(
    entry_price: Decimal, take_profit_pct: Decimal, is_long: bool
) -> Decimal:
    """
    Calculate take profit price.

    Args:
        entry_price: Position entry price
        take_profit_pct: Take profit percentage
        is_long: True for long position, False for short

    Returns:
        Take profit price
    """
    take_profit_amount = entry_price * take_profit_pct / Decimal(100)

    if is_long:
        return entry_price + take_profit_amount
    return entry_price - take_profit_amount


def check_risk_alerts(
    margin_info: MarginInfo,
    limits: RiskLimits,
    current_positions: int,
    daily_pnl: Decimal,
) -> list[RiskAlert]:
    """
    Check for risk alerts based on current state.

    Args:
        margin_info: Current margin information
        limits: Risk limits
        current_positions: Number of open positions
        daily_pnl: Daily profit/loss

    Returns:
        List of active risk alerts
    """
    alerts: list[RiskAlert] = []

    # Check position limit
    if current_positions >= limits.position_limit:
        alerts.append(
            PositionLimitExceeded(
                current_positions=current_positions, limit=limits.position_limit
            )
        )

    # Check margin ratio
    margin_threshold = limits.margin_requirement / Decimal(100)
    if margin_info.margin_ratio > margin_threshold:
        alerts.append(
            MarginCall(
                margin_ratio=margin_info.margin_ratio, threshold=margin_threshold
            )
        )

    # Check daily loss limit
    if daily_pnl < Decimal(0) and abs(daily_pnl) >= limits.daily_loss_limit:
        alerts.append(
            DailyLossLimit(current_loss=abs(daily_pnl), limit=limits.daily_loss_limit)
        )

    return alerts


def calculate_position_risk(
    position_size: Decimal, entry_price: Decimal, stop_loss_price: Decimal
) -> Decimal:
    """
    Calculate risk amount for a position.

    Args:
        position_size: Size of the position
        entry_price: Entry price
        stop_loss_price: Stop loss price

    Returns:
        Risk amount in USD
    """
    price_diff = abs(entry_price - stop_loss_price)
    return position_size * price_diff


def is_within_risk_limits(
    proposed_risk: Decimal,
    current_exposure: Decimal,
    balance: Decimal,
    max_risk_pct: Decimal,
) -> bool:
    """
    Check if proposed trade is within risk limits.

    Args:
        proposed_risk: Risk of proposed trade
        current_exposure: Current total exposure
        balance: Account balance
        max_risk_pct: Maximum risk percentage

    Returns:
        True if within limits, False otherwise
    """
    total_risk = current_exposure + proposed_risk
    max_allowed_risk = balance * max_risk_pct / Decimal(100)
    return total_risk <= max_allowed_risk
