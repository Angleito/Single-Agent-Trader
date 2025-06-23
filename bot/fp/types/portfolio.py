"""Immutable portfolio representation types for functional programming."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class Position:
    """Immutable position representation."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    size: Decimal
    entry_price: Decimal
    current_price: Decimal

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.size
        # SHORT
        return (self.entry_price - self.current_price) * self.size

    @property
    def value(self) -> Decimal:
        """Calculate current position value."""
        return self.current_price * self.size

    def with_price(self, new_price: Decimal) -> "Position":
        """Return new Position with updated price."""
        return replace(self, current_price=new_price)


@dataclass(frozen=True)
class Portfolio:
    """Immutable portfolio representation."""

    positions: tuple[Position, ...]
    cash_balance: Decimal

    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        position_value = sum(pos.value for pos in self.positions)
        return self.cash_balance + position_value

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions)

    def with_position(self, position: Position) -> "Portfolio":
        """Return new Portfolio with added position."""
        # Check if position already exists
        existing_positions = tuple(
            pos for pos in self.positions if pos.symbol != position.symbol
        )
        return replace(
            self,
            positions=existing_positions + (position,),
            cash_balance=self.cash_balance - position.value,
        )

    def without_position(self, symbol: str) -> "Portfolio":
        """Return new Portfolio with position removed."""
        position_to_remove = next(
            (pos for pos in self.positions if pos.symbol == symbol), None
        )
        if not position_to_remove:
            return self

        new_positions = tuple(pos for pos in self.positions if pos.symbol != symbol)
        # Add back the cash from closing the position
        cash_from_close = position_to_remove.value + position_to_remove.unrealized_pnl
        return replace(
            self,
            positions=new_positions,
            cash_balance=self.cash_balance + cash_from_close,
        )

    def calculate_pnl(self) -> Decimal:
        """Calculate total realized and unrealized P&L."""
        return self.unrealized_pnl

    def update_prices(self, price_updates: dict[str, Decimal]) -> "Portfolio":
        """Return new Portfolio with updated position prices."""
        new_positions = tuple(
            pos.with_price(price_updates.get(pos.symbol, pos.current_price))
            for pos in self.positions
        )
        return replace(self, positions=new_positions)


@dataclass(frozen=True)
class TradeResult:
    """Immutable trade result representation."""

    trade_id: str
    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    entry_time: datetime
    exit_time: datetime

    @property
    def pnl(self) -> Decimal:
        """Calculate P&L for the trade."""
        if self.side == "LONG":
            return (self.exit_price - self.entry_price) * self.size
        # SHORT
        return (self.entry_price - self.exit_price) * self.size

    @property
    def duration(self) -> float:
        """Calculate trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def return_pct(self) -> Decimal:
        """Calculate percentage return."""
        if self.side == "LONG":
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        # SHORT
        return ((self.entry_price - self.exit_price) / self.entry_price) * 100


@dataclass(frozen=True)
class PortfolioMetrics:
    """Immutable portfolio metrics representation."""

    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    sharpe_ratio: float
    max_drawdown: float
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float

    @classmethod
    def from_trades(
        cls,
        trades: tuple[TradeResult, ...],
        current_unrealized: Decimal = Decimal(0),
        risk_free_rate: float = 0.02,
    ) -> "PortfolioMetrics":
        """Calculate metrics from trade results."""
        if not trades:
            return cls(
                total_pnl=current_unrealized,
                realized_pnl=Decimal(0),
                unrealized_pnl=current_unrealized,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_win=Decimal(0),
                avg_loss=Decimal(0),
                profit_factor=0.0,
            )

        # Calculate basic metrics
        winning_trades = tuple(t for t in trades if t.pnl > 0)
        losing_trades = tuple(t for t in trades if t.pnl < 0)

        realized_pnl = sum(t.pnl for t in trades)
        total_pnl = realized_pnl + current_unrealized

        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        avg_win = (
            sum(t.pnl for t in winning_trades) / len(winning_trades)
            if winning_trades
            else Decimal(0)
        )
        avg_loss = (
            sum(abs(t.pnl) for t in losing_trades) / len(losing_trades)
            if losing_trades
            else Decimal(0)
        )

        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = sum(abs(t.pnl) for t in losing_trades)
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # Calculate Sharpe ratio (simplified)
        returns = [float(t.return_pct) for t in trades]
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (
                sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
            ) ** 0.5
            sharpe_ratio = (
                (avg_return - risk_free_rate) / std_return if std_return > 0 else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Calculate max drawdown
        cumulative_pnl = []
        running_total = Decimal(0)
        for trade in sorted(trades, key=lambda t: t.exit_time):
            running_total += trade.pnl
            cumulative_pnl.append(running_total)

        if cumulative_pnl:
            peak = cumulative_pnl[0]
            max_dd = 0.0
            for pnl in cumulative_pnl:
                peak = max(peak, pnl)
                drawdown = float((peak - pnl) / peak) if peak > 0 else 0.0
                max_dd = max(max_dd, drawdown)
            max_drawdown = max_dd * 100  # Convert to percentage
        else:
            max_drawdown = 0.0

        return cls(
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=current_unrealized,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
        )


# Example usage functions (pure)
def close_position(
    portfolio: Portfolio, symbol: str, exit_price: Decimal, exit_time: datetime
) -> tuple[Portfolio, TradeResult | None]:
    """Close a position and return updated portfolio and trade result."""
    position = next((pos for pos in portfolio.positions if pos.symbol == symbol), None)
    if not position:
        return portfolio, None

    # Create trade result
    trade_result = TradeResult(
        trade_id=f"{symbol}_{exit_time.isoformat()}",
        symbol=symbol,
        side=position.side,
        entry_price=position.entry_price,
        exit_price=exit_price,
        size=position.size,
        entry_time=exit_time,  # This should be tracked separately in real implementation
        exit_time=exit_time,
    )

    # Update portfolio
    new_portfolio = portfolio.without_position(symbol)

    return new_portfolio, trade_result


def open_position(
    portfolio: Portfolio, symbol: str, side: str, size: Decimal, entry_price: Decimal
) -> Portfolio:
    """Open a new position and return updated portfolio."""
    position = Position(
        symbol=symbol,
        side=side,
        size=size,
        entry_price=entry_price,
        current_price=entry_price,
    )
    return portfolio.with_position(position)


def calculate_portfolio_metrics(
    portfolio: Portfolio, historical_trades: tuple[TradeResult, ...]
) -> PortfolioMetrics:
    """Calculate comprehensive portfolio metrics."""
    return PortfolioMetrics.from_trades(
        trades=historical_trades, current_unrealized=portfolio.unrealized_pnl
    )
