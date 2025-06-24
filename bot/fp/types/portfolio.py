"""Immutable portfolio representation types for functional programming."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple
from enum import Enum

from bot.fp.types.base import Money, Percentage
from bot.fp.types.result import Result, Success, Failure


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


# Enhanced Portfolio Types for Asset Allocation and Balance Management

class AccountType(Enum):
    """Account type enumeration."""
    SPOT = "SPOT"
    FUTURES = "FUTURES" 
    MARGIN = "MARGIN"


class BalanceType(Enum):
    """Balance type enumeration."""
    AVAILABLE = "AVAILABLE"
    LOCKED = "LOCKED"
    TOTAL = "TOTAL"
    MARGIN_USED = "MARGIN_USED"
    MARGIN_AVAILABLE = "MARGIN_AVAILABLE"


@dataclass(frozen=True)
class AssetBalance:
    """Immutable asset balance representation."""
    
    asset: str
    available: Decimal
    locked: Decimal
    total: Decimal
    account_type: AccountType
    last_updated: datetime
    
    def __post_init__(self) -> None:
        """Validate balance consistency."""
        if self.available < 0:
            raise ValueError(f"Available balance cannot be negative: {self.available}")
        if self.locked < 0:
            raise ValueError(f"Locked balance cannot be negative: {self.locked}")
        if abs(self.total - (self.available + self.locked)) > Decimal("0.000001"):
            raise ValueError(f"Total balance {self.total} does not equal available + locked {self.available + self.locked}")
    
    @property
    def utilization_ratio(self) -> Decimal:
        """Calculate balance utilization ratio (locked / total)."""
        if self.total == Decimal("0"):
            return Decimal("0")
        return self.locked / self.total
    
    def with_available(self, new_available: Decimal) -> "AssetBalance":
        """Return new balance with updated available amount."""
        new_total = new_available + self.locked
        return replace(
            self, 
            available=new_available, 
            total=new_total,
            last_updated=datetime.now()
        )
    
    def with_locked(self, new_locked: Decimal) -> "AssetBalance":
        """Return new balance with updated locked amount."""
        new_total = self.available + new_locked
        return replace(
            self, 
            locked=new_locked, 
            total=new_total,
            last_updated=datetime.now()
        )
    
    def lock_amount(self, amount: Decimal) -> Result["AssetBalance", str]:
        """Lock specified amount from available balance."""
        if amount <= 0:
            return Failure("Lock amount must be positive")
        if amount > self.available:
            return Failure(f"Insufficient available balance: {self.available} < {amount}")
        
        new_balance = AssetBalance(
            asset=self.asset,
            available=self.available - amount,
            locked=self.locked + amount,
            total=self.total,
            account_type=self.account_type,
            last_updated=datetime.now()
        )
        return Success(new_balance)
    
    def unlock_amount(self, amount: Decimal) -> Result["AssetBalance", str]:
        """Unlock specified amount to available balance."""
        if amount <= 0:
            return Failure("Unlock amount must be positive")
        if amount > self.locked:
            return Failure(f"Insufficient locked balance: {self.locked} < {amount}")
        
        new_balance = AssetBalance(
            asset=self.asset,
            available=self.available + amount,
            locked=self.locked - amount,
            total=self.total,
            account_type=self.account_type,
            last_updated=datetime.now()
        )
        return Success(new_balance)


@dataclass(frozen=True)
class AssetAllocation:
    """Immutable asset allocation configuration."""
    
    asset: str
    target_percentage: Percentage
    min_percentage: Percentage
    max_percentage: Percentage
    rebalance_threshold: Percentage
    
    def __post_init__(self) -> None:
        """Validate allocation constraints."""
        if self.min_percentage.value > self.target_percentage.value:
            raise ValueError("Minimum percentage cannot exceed target percentage")
        if self.target_percentage.value > self.max_percentage.value:
            raise ValueError("Target percentage cannot exceed maximum percentage")
        if self.rebalance_threshold.value <= 0:
            raise ValueError("Rebalance threshold must be positive")
    
    def is_within_bounds(self, current_percentage: Percentage) -> bool:
        """Check if current allocation is within bounds."""
        return (
            self.min_percentage.value <= current_percentage.value <= 
            self.max_percentage.value
        )
    
    def needs_rebalancing(self, current_percentage: Percentage) -> bool:
        """Check if allocation needs rebalancing."""
        deviation = abs(current_percentage.value - self.target_percentage.value)
        return deviation > self.rebalance_threshold.value


@dataclass(frozen=True)
class MarginInfo:
    """Immutable margin information for futures/margin accounts."""
    
    initial_margin: Decimal
    maintenance_margin: Decimal
    margin_used: Decimal
    margin_available: Decimal
    equity: Decimal
    leverage: Decimal
    margin_ratio: Decimal
    liquidation_threshold: Decimal
    last_updated: datetime
    
    @property
    def is_healthy(self) -> bool:
        """Check if margin health is good."""
        return self.margin_ratio > self.liquidation_threshold * Decimal("1.5")
    
    @property
    def is_warning(self) -> bool:
        """Check if margin is in warning zone."""
        return (
            self.margin_ratio > self.liquidation_threshold and 
            self.margin_ratio <= self.liquidation_threshold * Decimal("1.5")
        )
    
    @property
    def is_critical(self) -> bool:
        """Check if margin is in critical zone."""
        return self.margin_ratio <= self.liquidation_threshold
    
    def with_updated_equity(self, new_equity: Decimal) -> "MarginInfo":
        """Return new margin info with updated equity."""
        new_margin_ratio = new_equity / self.margin_used if self.margin_used > 0 else Decimal("0")
        new_margin_available = new_equity - self.margin_used
        
        return replace(
            self,
            equity=new_equity,
            margin_ratio=new_margin_ratio,
            margin_available=max(Decimal("0"), new_margin_available),
            last_updated=datetime.now()
        )


@dataclass(frozen=True)
class AccountSnapshot:
    """Immutable account snapshot with balances and positions."""
    
    account_type: AccountType
    balances: Tuple[AssetBalance, ...]
    base_currency: str
    total_equity: Decimal
    margin_info: Optional[MarginInfo]
    timestamp: datetime
    
    @property
    def total_available_value(self) -> Decimal:
        """Calculate total available value across all assets."""
        # In a real implementation, this would need price conversion
        # For now, assuming base currency values
        return sum(
            balance.available for balance in self.balances 
            if balance.asset == self.base_currency
        )
    
    @property
    def total_locked_value(self) -> Decimal:
        """Calculate total locked value across all assets."""
        return sum(
            balance.locked for balance in self.balances 
            if balance.asset == self.base_currency
        )
    
    def get_balance(self, asset: str) -> Optional[AssetBalance]:
        """Get balance for specific asset."""
        for balance in self.balances:
            if balance.asset == asset:
                return balance
        return None
    
    def with_updated_balance(self, updated_balance: AssetBalance) -> "AccountSnapshot":
        """Return new snapshot with updated balance."""
        new_balances = []
        found = False
        
        for balance in self.balances:
            if balance.asset == updated_balance.asset:
                new_balances.append(updated_balance)
                found = True
            else:
                new_balances.append(balance)
        
        if not found:
            new_balances.append(updated_balance)
        
        # Recalculate equity
        new_equity = sum(balance.total for balance in new_balances if balance.asset == self.base_currency)
        
        return replace(
            self,
            balances=tuple(new_balances),
            total_equity=new_equity,
            timestamp=datetime.now()
        )


@dataclass(frozen=True)
class PortfolioAllocation:
    """Immutable portfolio allocation state."""
    
    allocations: Tuple[AssetAllocation, ...]
    current_values: Dict[str, Decimal]
    total_portfolio_value: Decimal
    last_rebalance: Optional[datetime]
    
    def get_current_allocation(self, asset: str) -> Optional[Percentage]:
        """Get current allocation percentage for asset."""
        if self.total_portfolio_value == Decimal("0"):
            return None
        
        current_value = self.current_values.get(asset, Decimal("0"))
        percentage_value = current_value / self.total_portfolio_value
        return Percentage.create(float(percentage_value)).success() if percentage_value <= 1 else None
    
    def get_target_allocation(self, asset: str) -> Optional[AssetAllocation]:
        """Get target allocation for asset."""
        for allocation in self.allocations:
            if allocation.asset == asset:
                return allocation
        return None
    
    def calculate_rebalancing_needs(self) -> Dict[str, Decimal]:
        """Calculate rebalancing requirements for all assets."""
        rebalancing_needs = {}
        
        for allocation in self.allocations:
            current_pct = self.get_current_allocation(allocation.asset)
            if current_pct and allocation.needs_rebalancing(current_pct):
                target_value = self.total_portfolio_value * allocation.target_percentage.value
                current_value = self.current_values.get(allocation.asset, Decimal("0"))
                rebalancing_needs[allocation.asset] = target_value - current_value
        
        return rebalancing_needs
    
    def is_rebalancing_needed(self) -> bool:
        """Check if any asset needs rebalancing."""
        return len(self.calculate_rebalancing_needs()) > 0


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Immutable performance snapshot for a point in time."""
    
    timestamp: datetime
    total_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    daily_return: Optional[Decimal]
    benchmark_return: Optional[Decimal]
    drawdown: Decimal
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_return_pct(self) -> Decimal:
        """Calculate total return percentage."""
        if self.total_value == Decimal("0"):
            return Decimal("0")
        return (self.total_pnl / self.total_value) * 100


@dataclass(frozen=True)
class RiskMetrics:
    """Immutable risk metrics for portfolio."""
    
    var_95: Decimal  # Value at Risk (95% confidence)
    max_drawdown: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    beta: Optional[Decimal]
    correlation_to_benchmark: Optional[Decimal]
    concentration_risk: Decimal
    timestamp: datetime
    
    @property
    def risk_score(self) -> int:
        """Calculate overall risk score (1-10 scale)."""
        # Simplified risk scoring
        score = 5  # Base score
        
        # Adjust for volatility
        if self.volatility > Decimal("0.3"):
            score += 2
        elif self.volatility < Decimal("0.1"):
            score -= 1
        
        # Adjust for max drawdown
        if self.max_drawdown > Decimal("0.2"):
            score += 2
        elif self.max_drawdown < Decimal("0.05"):
            score -= 1
        
        # Adjust for Sharpe ratio
        if self.sharpe_ratio > Decimal("2.0"):
            score -= 2
        elif self.sharpe_ratio < Decimal("0.5"):
            score += 1
        
        return max(1, min(10, score))


# Factory functions for creating portfolio components

def create_spot_account(
    balances: Dict[str, Decimal], 
    base_currency: str = "USD"
) -> AccountSnapshot:
    """Create a spot trading account snapshot."""
    balance_objects = []
    for asset, amount in balances.items():
        balance_objects.append(AssetBalance(
            asset=asset,
            available=amount,
            locked=Decimal("0"),
            total=amount,
            account_type=AccountType.SPOT,
            last_updated=datetime.now()
        ))
    
    total_equity = sum(
        balance.total for balance in balance_objects 
        if balance.asset == base_currency
    )
    
    return AccountSnapshot(
        account_type=AccountType.SPOT,
        balances=tuple(balance_objects),
        base_currency=base_currency,
        total_equity=total_equity,
        margin_info=None,
        timestamp=datetime.now()
    )


def create_futures_account(
    balances: Dict[str, Decimal],
    margin_used: Decimal,
    leverage: Decimal,
    base_currency: str = "USD"
) -> AccountSnapshot:
    """Create a futures trading account snapshot."""
    balance_objects = []
    for asset, amount in balances.items():
        balance_objects.append(AssetBalance(
            asset=asset,
            available=amount,
            locked=Decimal("0"),
            total=amount,
            account_type=AccountType.FUTURES,
            last_updated=datetime.now()
        ))
    
    total_equity = sum(
        balance.total for balance in balance_objects 
        if balance.asset == base_currency
    )
    
    margin_info = MarginInfo(
        initial_margin=margin_used,
        maintenance_margin=margin_used * Decimal("0.8"),  # Simplified
        margin_used=margin_used,
        margin_available=total_equity - margin_used,
        equity=total_equity,
        leverage=leverage,
        margin_ratio=total_equity / margin_used if margin_used > 0 else Decimal("0"),
        liquidation_threshold=Decimal("1.1"),  # 110% margin ratio threshold
        last_updated=datetime.now()
    )
    
    return AccountSnapshot(
        account_type=AccountType.FUTURES,
        balances=tuple(balance_objects),
        base_currency=base_currency,
        total_equity=total_equity,
        margin_info=margin_info,
        timestamp=datetime.now()
    )


def create_balanced_allocation(
    assets: Tuple[str, ...], 
    rebalance_threshold: float = 0.05
) -> Tuple[AssetAllocation, ...]:
    """Create balanced allocation across multiple assets."""
    if not assets:
        return tuple()
    
    equal_weight = 1.0 / len(assets)
    allocations = []
    
    for asset in assets:
        allocations.append(AssetAllocation(
            asset=asset,
            target_percentage=Percentage.create(equal_weight).success(),
            min_percentage=Percentage.create(max(0.0, equal_weight - 0.1)).success(),
            max_percentage=Percentage.create(min(1.0, equal_weight + 0.1)).success(),
            rebalance_threshold=Percentage.create(rebalance_threshold).success()
        ))
    
    return tuple(allocations)
