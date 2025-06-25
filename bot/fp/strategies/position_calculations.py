"""Pure functions for position P&L calculations and risk metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from bot.fp.types.base import Maybe, Nothing
from bot.fp.types.result import Failure, Result, Success

if TYPE_CHECKING:
    from bot.fp.types.positions import (
        FunctionalPosition,
        LotSale,
        PositionSnapshot,
    )


@dataclass(frozen=True)
class PnLComponents:
    """Immutable P&L breakdown for analysis."""

    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    realized_trades: int
    winning_trades: int
    losing_trades: int

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.realized_trades == 0:
            return 0.0
        return (self.winning_trades / self.realized_trades) * 100

    @property
    def average_win(self) -> Decimal:
        """Calculate average winning trade P&L."""
        if self.winning_trades == 0:
            return Decimal(0)

        # This would need trade-level data to calculate properly
        # For now, return simplified calculation
        return (
            self.realized_pnl / self.winning_trades
            if self.winning_trades > 0
            else Decimal(0)
        )

    @property
    def average_loss(self) -> Decimal:
        """Calculate average losing trade P&L."""
        if self.losing_trades == 0:
            return Decimal(0)

        # This would need trade-level data to calculate properly
        # For now, return simplified calculation
        return (
            abs(self.realized_pnl) / self.losing_trades
            if self.losing_trades > 0
            else Decimal(0)
        )


@dataclass(frozen=True)
class RiskMetrics:
    """Immutable risk metrics for position analysis."""

    position_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    time_in_position_seconds: float
    exposure_risk_pct: float
    max_lot_age_seconds: float
    concentration_risk: float

    @property
    def time_in_position_hours(self) -> float:
        """Get time in position in hours."""
        return self.time_in_position_seconds / 3600

    @property
    def time_in_position_days(self) -> float:
        """Get time in position in days."""
        return self.time_in_position_seconds / 86400

    @property
    def max_lot_age_hours(self) -> float:
        """Get maximum lot age in hours."""
        return self.max_lot_age_seconds / 3600


@dataclass(frozen=True)
class PositionPerformance:
    """Immutable position performance metrics."""

    symbol: str
    pnl_components: PnLComponents
    risk_metrics: RiskMetrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.pnl_components.total_pnl > Decimal(0)

    @property
    def risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return score."""
        if self.risk_metrics.position_value == Decimal(0):
            return 0.0

        # Simple risk-adjusted return combining multiple metrics
        base_return = float(
            self.pnl_components.total_pnl / self.risk_metrics.position_value
        )
        time_penalty = min(
            1.0, self.risk_metrics.time_in_position_hours / 24
        )  # Penalty for holding too long
        concentration_penalty = max(0.5, 1.0 - self.risk_metrics.concentration_risk)

        return base_return * concentration_penalty * (1.0 - time_penalty * 0.1)


# Pure P&L Calculation Functions


def calculate_unrealized_pnl(
    position: FunctionalPosition, current_price: Decimal
) -> Decimal:
    """
    Calculate unrealized P&L for a position at current price.

    Args:
        position: The position to calculate P&L for
        current_price: Current market price

    Returns:
        Unrealized P&L amount
    """
    if position.is_flat or position.total_quantity == Decimal(0):
        return Decimal(0)

    avg_price = position.average_price
    if avg_price.is_nothing():
        return Decimal(0)

    price_diff = current_price - avg_price.value

    if position.side.is_long():
        return position.total_quantity * price_diff
    if position.side.is_short():
        return position.total_quantity * (-price_diff)
    return Decimal(0)


def calculate_position_value(
    position: FunctionalPosition, current_price: Decimal
) -> Decimal:
    """
    Calculate current position value.

    Args:
        position: The position to value
        current_price: Current market price

    Returns:
        Current position value
    """
    if position.is_flat:
        return Decimal(0)

    return position.total_quantity * current_price


def calculate_cost_basis(position: FunctionalPosition) -> Decimal:
    """
    Calculate total cost basis of position.

    Args:
        position: The position to calculate cost basis for

    Returns:
        Total cost basis
    """
    return sum(lot.remaining_cost_basis for lot in position.lots)


def calculate_breakeven_price(position: FunctionalPosition) -> Maybe[Decimal]:
    """
    Calculate breakeven price for position.

    Args:
        position: The position to calculate breakeven for

    Returns:
        Breakeven price or Nothing if flat
    """
    if position.is_flat or position.total_quantity == Decimal(0):
        return Nothing()

    return position.average_price


def calculate_lot_level_pnl(
    position: FunctionalPosition, current_price: Decimal
) -> list[tuple[str, Decimal]]:
    """
    Calculate P&L for each individual lot.

    Args:
        position: The position containing lots
        current_price: Current market price

    Returns:
        List of (lot_id, unrealized_pnl) tuples
    """
    lot_pnls = []

    for lot in position.lots:
        if lot.remaining_quantity <= Decimal(0):
            continue

        price_diff = current_price - lot.purchase_price

        if position.side.is_long():
            lot_pnl = lot.remaining_quantity * price_diff
        elif position.side.is_short():
            lot_pnl = lot.remaining_quantity * (-price_diff)
        else:
            lot_pnl = Decimal(0)

        lot_pnls.append((lot.lot_id, lot_pnl))

    return lot_pnls


def calculate_fifo_tax_impact(sales: tuple[LotSale, ...]) -> Decimal:
    """
    Calculate tax impact of FIFO sales (simplified).

    Args:
        sales: Tuple of lot sales

    Returns:
        Total taxable gain/loss
    """
    return sum(sale.realized_pnl for sale in sales)


def calculate_position_pnl_components(
    position: FunctionalPosition, current_price: Decimal
) -> PnLComponents:
    """
    Calculate comprehensive P&L components for position.

    Args:
        position: The position to analyze
        current_price: Current market price

    Returns:
        PnL components breakdown
    """
    realized_pnl = position.total_realized_pnl
    unrealized_pnl = calculate_unrealized_pnl(position, current_price)
    total_pnl = realized_pnl + unrealized_pnl

    # Analyze sales for win/loss statistics
    winning_sales = [
        sale for sale in position.sales_history if sale.realized_pnl > Decimal(0)
    ]
    losing_sales = [
        sale for sale in position.sales_history if sale.realized_pnl < Decimal(0)
    ]

    return PnLComponents(
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=total_pnl,
        realized_trades=len(position.sales_history),
        winning_trades=len(winning_sales),
        losing_trades=len(losing_sales),
    )


def calculate_position_risk_metrics(
    position: FunctionalPosition,
    current_price: Decimal,
    account_balance: Decimal,
    current_time: datetime | None = None,
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for position.

    Args:
        position: The position to analyze
        current_price: Current market price
        account_balance: Total account balance for exposure calculation
        current_time: Current timestamp (defaults to now)

    Returns:
        Risk metrics
    """
    if current_time is None:
        current_time = datetime.now()

    position_value = calculate_position_value(position, current_price)
    unrealized_pnl = calculate_unrealized_pnl(position, current_price)

    # Calculate unrealized P&L percentage
    cost_basis = calculate_cost_basis(position)
    unrealized_pnl_pct = 0.0
    if cost_basis > Decimal(0):
        unrealized_pnl_pct = float(unrealized_pnl / cost_basis * 100)

    # Calculate time in position
    if position.lots:
        oldest_lot_time = min(lot.purchase_date for lot in position.lots)
        time_in_position = (current_time - oldest_lot_time).total_seconds()
        max_lot_age = time_in_position
    else:
        time_in_position = 0.0
        max_lot_age = 0.0

    # Calculate exposure risk
    exposure_risk_pct = 0.0
    if account_balance > Decimal(0):
        exposure_risk_pct = float(position_value / account_balance * 100)

    # Calculate concentration risk (simplified)
    concentration_risk = min(1.0, exposure_risk_pct / 100)

    return RiskMetrics(
        position_value=position_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        time_in_position_seconds=time_in_position,
        exposure_risk_pct=exposure_risk_pct,
        max_lot_age_seconds=max_lot_age,
        concentration_risk=concentration_risk,
    )


def calculate_sharpe_ratio(
    sales: tuple[LotSale, ...], risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe ratio from sale history.

    Args:
        sales: Historical sales data
        risk_free_rate: Risk-free rate for calculation

    Returns:
        Sharpe ratio
    """
    if len(sales) < 2:
        return 0.0

    returns = [float(sale.realized_pnl) for sale in sales]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)

    # Calculate standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_dev


def calculate_sortino_ratio(
    sales: tuple[LotSale, ...], risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sortino ratio from sale history (downside deviation).

    Args:
        sales: Historical sales data
        risk_free_rate: Risk-free rate for calculation

    Returns:
        Sortino ratio
    """
    if len(sales) < 2:
        return 0.0

    returns = [float(sale.realized_pnl) for sale in sales]

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)

    # Calculate downside deviation (only negative returns)
    negative_returns = [r for r in returns if r < 0]

    if not negative_returns:
        return float("inf") if mean_return > risk_free_rate else 0.0

    downside_variance = sum((r - 0) ** 2 for r in negative_returns) / len(
        negative_returns
    )
    downside_deviation = math.sqrt(downside_variance)

    if downside_deviation == 0:
        return 0.0

    return (mean_return - risk_free_rate) / downside_deviation


def calculate_max_drawdown(
    sales: tuple[LotSale, ...], current_unrealized: Decimal = Decimal(0)
) -> float:
    """
    Calculate maximum drawdown from sale history.

    Args:
        sales: Historical sales data
        current_unrealized: Current unrealized P&L

    Returns:
        Maximum drawdown percentage
    """
    if not sales:
        return 0.0

    # Build cumulative P&L series
    cumulative_pnl = []
    running_total = Decimal(0)

    for sale in sorted(sales, key=lambda s: s.sale_date):
        running_total += sale.realized_pnl
        cumulative_pnl.append(running_total)

    # Add current unrealized
    cumulative_pnl.append(running_total + current_unrealized)

    if not cumulative_pnl:
        return 0.0

    # Calculate maximum drawdown
    peak = cumulative_pnl[0]
    max_drawdown = 0.0

    for pnl in cumulative_pnl:
        peak = max(peak, pnl)
        if peak > Decimal(0):
            drawdown = float((peak - pnl) / peak * 100)
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def calculate_profit_factor(sales: tuple[LotSale, ...]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        sales: Historical sales data

    Returns:
        Profit factor
    """
    if not sales:
        return 0.0

    gross_profit = sum(
        sale.realized_pnl for sale in sales if sale.realized_pnl > Decimal(0)
    )
    gross_loss = abs(
        sum(sale.realized_pnl for sale in sales if sale.realized_pnl < Decimal(0))
    )

    if gross_loss == Decimal(0):
        return float("inf") if gross_profit > Decimal(0) else 0.0

    return float(gross_profit / gross_loss)


def calculate_position_performance(
    position: FunctionalPosition,
    current_price: Decimal,
    account_balance: Decimal,
    current_time: datetime | None = None,
) -> PositionPerformance:
    """
    Calculate comprehensive position performance metrics.

    Args:
        position: The position to analyze
        current_price: Current market price
        account_balance: Total account balance
        current_time: Current timestamp

    Returns:
        Complete position performance analysis
    """
    pnl_components = calculate_position_pnl_components(position, current_price)
    risk_metrics = calculate_position_risk_metrics(
        position, current_price, account_balance, current_time
    )

    sharpe = calculate_sharpe_ratio(position.sales_history)
    sortino = calculate_sortino_ratio(position.sales_history)
    max_dd = calculate_max_drawdown(
        position.sales_history, pnl_components.unrealized_pnl
    )
    profit_factor = calculate_profit_factor(position.sales_history)

    return PositionPerformance(
        symbol=position.symbol,
        pnl_components=pnl_components,
        risk_metrics=risk_metrics,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd,
        profit_factor=profit_factor,
    )


# Portfolio-level calculations


def calculate_portfolio_pnl(
    snapshot: PositionSnapshot, current_prices: dict[str, Decimal]
) -> Result[str, PnLComponents]:
    """
    Calculate portfolio-level P&L components.

    Args:
        snapshot: Position snapshot
        current_prices: Current prices for all symbols

    Returns:
        Result containing portfolio P&L components
    """
    try:
        total_realized = Decimal(0)
        total_unrealized = Decimal(0)
        total_trades = 0
        total_winning = 0
        total_losing = 0

        for position in snapshot.positions:
            if position.is_flat:
                continue

            current_price = current_prices.get(position.symbol)
            if current_price is None:
                return Failure(f"No current price available for {position.symbol}")

            pnl_components = calculate_position_pnl_components(position, current_price)

            total_realized += pnl_components.realized_pnl
            total_unrealized += pnl_components.unrealized_pnl
            total_trades += pnl_components.realized_trades
            total_winning += pnl_components.winning_trades
            total_losing += pnl_components.losing_trades

        portfolio_pnl = PnLComponents(
            realized_pnl=total_realized,
            unrealized_pnl=total_unrealized,
            total_pnl=total_realized + total_unrealized,
            realized_trades=total_trades,
            winning_trades=total_winning,
            losing_trades=total_losing,
        )

        return Success(portfolio_pnl)

    except Exception as e:
        return Failure(f"Failed to calculate portfolio P&L: {e!s}")


def calculate_portfolio_risk(
    snapshot: PositionSnapshot,
    current_prices: dict[str, Decimal],
    account_balance: Decimal,
) -> Result[str, RiskMetrics]:
    """
    Calculate portfolio-level risk metrics.

    Args:
        snapshot: Position snapshot
        current_prices: Current prices for all symbols
        account_balance: Total account balance

    Returns:
        Result containing portfolio risk metrics
    """
    try:
        total_value = Decimal(0)
        total_unrealized = Decimal(0)
        max_age = 0.0
        concentration_sum = 0.0

        current_time = datetime.now()

        for position in snapshot.positions:
            if position.is_flat:
                continue

            current_price = current_prices.get(position.symbol)
            if current_price is None:
                return Failure(f"No current price available for {position.symbol}")

            risk_metrics = calculate_position_risk_metrics(
                position, current_price, account_balance, current_time
            )

            total_value += risk_metrics.position_value
            total_unrealized += risk_metrics.unrealized_pnl
            max_age = max(max_age, risk_metrics.max_lot_age_seconds)
            concentration_sum += risk_metrics.concentration_risk

        # Calculate portfolio-level metrics
        exposure_pct = (
            float(total_value / account_balance * 100)
            if account_balance > Decimal(0)
            else 0.0
        )
        unrealized_pct = (
            float(total_unrealized / total_value * 100)
            if total_value > Decimal(0)
            else 0.0
        )
        avg_concentration = (
            concentration_sum / len(snapshot.positions) if snapshot.positions else 0.0
        )

        portfolio_risk = RiskMetrics(
            position_value=total_value,
            unrealized_pnl=total_unrealized,
            unrealized_pnl_pct=unrealized_pct,
            time_in_position_seconds=max_age,
            exposure_risk_pct=exposure_pct,
            max_lot_age_seconds=max_age,
            concentration_risk=avg_concentration,
        )

        return Success(portfolio_risk)

    except Exception as e:
        return Failure(f"Failed to calculate portfolio risk: {e!s}")


# Position comparison and ranking functions


def rank_positions_by_performance(
    positions: list[FunctionalPosition],
    current_prices: dict[str, Decimal],
    account_balance: Decimal,
) -> list[tuple[str, PositionPerformance]]:
    """
    Rank positions by risk-adjusted performance.

    Args:
        positions: List of positions to rank
        current_prices: Current prices for all symbols
        account_balance: Account balance for risk calculations

    Returns:
        List of (symbol, performance) tuples sorted by performance
    """
    performances = []

    for position in positions:
        if position.is_flat:
            continue

        current_price = current_prices.get(position.symbol)
        if current_price is None:
            continue

        performance = calculate_position_performance(
            position, current_price, account_balance
        )
        performances.append((position.symbol, performance))

    # Sort by risk-adjusted return (descending)
    return sorted(performances, key=lambda x: x[1].risk_adjusted_return, reverse=True)


def find_underperforming_positions(
    positions: list[FunctionalPosition],
    current_prices: dict[str, Decimal],
    account_balance: Decimal,
    max_drawdown_threshold: float = 10.0,
    min_sharpe_ratio: float = 0.5,
) -> list[tuple[str, PositionPerformance]]:
    """
    Find positions that may need attention based on performance criteria.

    Args:
        positions: List of positions to analyze
        current_prices: Current prices for all symbols
        account_balance: Account balance for risk calculations
        max_drawdown_threshold: Maximum acceptable drawdown percentage
        min_sharpe_ratio: Minimum acceptable Sharpe ratio

    Returns:
        List of underperforming positions
    """
    underperforming = []

    for position in positions:
        if position.is_flat:
            continue

        current_price = current_prices.get(position.symbol)
        if current_price is None:
            continue

        performance = calculate_position_performance(
            position, current_price, account_balance
        )

        # Check underperformance criteria
        is_underperforming = (
            performance.max_drawdown_pct > max_drawdown_threshold
            or performance.sharpe_ratio < min_sharpe_ratio
            or performance.pnl_components.total_pnl < Decimal(0)
        )

        if is_underperforming:
            underperforming.append((position.symbol, performance))

    return underperforming
