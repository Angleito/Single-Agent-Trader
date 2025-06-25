"""Immutable paper trading state types for functional programming."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Union
from uuid import uuid4

from .portfolio import Position, TradeResult


# Simple aliases for backward compatibility
@dataclass(frozen=True)
class PaperTrade:
    """Simple paper trade for backward compatibility"""

    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: Decimal
    price: Decimal
    timestamp: datetime
    status: str = "filled"
    pnl: Decimal | None = None

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L"""
        if self.side == "buy":
            return (current_price - self.price) * self.size
        return (self.price - current_price) * self.size


@dataclass(frozen=True)
class PaperPosition:
    """Simple paper position for backward compatibility"""

    symbol: str
    side: str  # 'long' or 'short'
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal

    @classmethod
    def from_trade(cls, trade: PaperTrade, current_price: Decimal) -> "PaperPosition":
        """Create position from trade"""
        side = "long" if trade.side == "buy" else "short"
        pnl = trade.calculate_pnl(current_price)

        return cls(
            symbol=trade.symbol,
            side=side,
            size=trade.size,
            entry_price=trade.price,
            current_price=current_price,
            unrealized_pnl=pnl,
        )


@dataclass(frozen=True)
class PaperTradeState:
    """Immutable paper trade state representation."""

    id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: Decimal
    size: Decimal
    exit_time: datetime | None = None
    exit_price: Decimal | None = None
    realized_pnl: Decimal | None = None
    fees: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    status: str = "OPEN"  # "OPEN", "CLOSED", "PARTIAL"

    def __post_init__(self) -> None:
        """Validate trade state parameters."""
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")
        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {self.entry_price}")
        if self.exit_price is not None and self.exit_price <= 0:
            raise ValueError(f"Exit price must be positive, got {self.exit_price}")

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L based on current price (pure function)."""
        if self.status == "CLOSED":
            return Decimal(0)

        price_diff = current_price - self.entry_price
        if self.side == "LONG":
            return self.size * price_diff - self.fees
        # SHORT
        return self.size * (-price_diff) - self.fees

    def close_trade(
        self,
        exit_price: Decimal,
        exit_time: datetime,
        additional_fees: Decimal = Decimal(0),
    ) -> "PaperTradeState":
        """Close the trade and return new state with realized P&L (pure function)."""
        price_diff = exit_price - self.entry_price
        if self.side == "LONG":
            gross_pnl = self.size * price_diff
        else:  # SHORT
            gross_pnl = self.size * (-price_diff)

        total_fees = self.fees + additional_fees
        realized_pnl = gross_pnl - total_fees

        return replace(
            self,
            exit_time=exit_time,
            exit_price=exit_price,
            status="CLOSED",
            realized_pnl=realized_pnl,
        )

    def to_position(self, current_price: Decimal | None = None) -> Position:
        """Convert to immutable Position type."""
        price = current_price or self.exit_price or self.entry_price
        return Position(
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            entry_price=self.entry_price,
            current_price=price,
        )

    def to_trade_result(self) -> TradeResult | None:
        """Convert to TradeResult if trade is closed."""
        if self.status != "CLOSED" or not self.exit_time or not self.exit_price:
            return None

        return TradeResult(
            trade_id=self.id,
            symbol=self.symbol,
            side=self.side,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            size=self.size,
            entry_time=self.entry_time,
            exit_time=self.exit_time,
        )


@dataclass(frozen=True)
class PaperTradingAccountState:
    """Immutable paper trading account state."""

    starting_balance: Decimal
    current_balance: Decimal
    equity: Decimal
    margin_used: Decimal
    open_trades: tuple[PaperTradeState, ...]
    closed_trades: tuple[PaperTradeState, ...]
    trade_counter: int
    peak_equity: Decimal
    max_drawdown: Decimal
    session_start_time: datetime

    def __post_init__(self) -> None:
        """Validate account state parameters."""
        if self.starting_balance <= 0:
            raise ValueError(
                f"Starting balance must be positive, got {self.starting_balance}"
            )
        if self.trade_counter < 0:
            raise ValueError(
                f"Trade counter must be non-negative, got {self.trade_counter}"
            )

    @classmethod
    def create_initial(
        cls, starting_balance: Decimal, session_start_time: datetime | None = None
    ) -> "PaperTradingAccountState":
        """Create initial account state (pure function)."""
        start_time = session_start_time or datetime.now()
        return cls(
            starting_balance=starting_balance,
            current_balance=starting_balance,
            equity=starting_balance,
            margin_used=Decimal(0),
            open_trades=(),
            closed_trades=(),
            trade_counter=0,
            peak_equity=starting_balance,
            max_drawdown=Decimal(0),
            session_start_time=start_time,
        )

    def update_equity(
        self, current_prices: dict[str, Decimal]
    ) -> "PaperTradingAccountState":
        """Update equity based on current prices (pure function)."""
        unrealized_pnl = sum(
            trade.calculate_unrealized_pnl(
                current_prices.get(trade.symbol, trade.entry_price)
            )
            for trade in self.open_trades
        )
        new_equity = self.current_balance + unrealized_pnl

        # Update peak equity and drawdown
        new_peak_equity = max(self.peak_equity, new_equity)
        current_drawdown = (
            (new_peak_equity - new_equity) / new_peak_equity * 100
            if new_peak_equity > 0
            else Decimal(0)
        )
        new_max_drawdown = max(self.max_drawdown, current_drawdown)

        return replace(
            self,
            equity=new_equity,
            peak_equity=new_peak_equity,
            max_drawdown=new_max_drawdown,
        )

    def add_trade(self, trade: PaperTradeState) -> "PaperTradingAccountState":
        """Add a new trade to the account (pure function)."""
        return replace(
            self,
            open_trades=(*self.open_trades, trade),
            trade_counter=self.trade_counter + 1,
        )

    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_time: datetime,
        additional_fees: Decimal = Decimal(0),
    ) -> "PaperTradingAccountState":
        """Close a trade and update account state (pure function)."""
        # Find the trade to close
        trade_to_close = next(
            (trade for trade in self.open_trades if trade.id == trade_id), None
        )
        if not trade_to_close:
            return self  # No-op if trade not found

        # Close the trade
        closed_trade = trade_to_close.close_trade(
            exit_price, exit_time, additional_fees
        )

        # Update trades
        remaining_open = tuple(
            trade for trade in self.open_trades if trade.id != trade_id
        )
        updated_closed = (*self.closed_trades, closed_trade)

        # Update balance and margin
        realized_pnl = closed_trade.realized_pnl or Decimal(0)
        new_balance = self.current_balance + realized_pnl

        # Calculate released margin
        trade_value = trade_to_close.size * trade_to_close.entry_price
        # Assuming 5x leverage as default, this should be configurable
        released_margin = trade_value / Decimal(5)
        new_margin_used = self.margin_used - released_margin

        return replace(
            self,
            current_balance=new_balance,
            open_trades=remaining_open,
            closed_trades=updated_closed,
            margin_used=max(Decimal(0), new_margin_used),  # Ensure non-negative
        )

    def update_balance(
        self, amount: Decimal, reason: str = ""
    ) -> "PaperTradingAccountState":
        """Update balance by amount (pure function)."""
        return replace(self, current_balance=self.current_balance + amount)

    def update_margin(self, additional_margin: Decimal) -> "PaperTradingAccountState":
        """Update margin usage (pure function)."""
        return replace(self, margin_used=self.margin_used + additional_margin)

    def get_available_margin(self) -> Decimal:
        """Calculate available margin (pure function)."""
        return max(Decimal(0), self.equity - self.margin_used)

    def get_total_pnl(self) -> Decimal:
        """Calculate total P&L (pure function)."""
        return self.equity - self.starting_balance

    def get_roi_percent(self) -> Decimal:
        """Calculate ROI percentage (pure function)."""
        if self.starting_balance <= 0:
            return Decimal(0)
        return (self.get_total_pnl() / self.starting_balance) * 100

    def get_open_position_count(self) -> int:
        """Get number of open positions (pure function)."""
        return len(self.open_trades)

    def get_total_trade_count(self) -> int:
        """Get total number of trades (pure function)."""
        return len(self.closed_trades)

    def find_open_trade(self, symbol: str) -> PaperTradeState | None:
        """Find open trade for symbol (pure function)."""
        return next(
            (trade for trade in self.open_trades if trade.symbol == symbol), None
        )

    def to_portfolio(self, current_prices: dict[str, Decimal]) -> "Portfolio":
        """Convert to immutable Portfolio representation."""
        from .portfolio import Portfolio

        positions = tuple(
            trade.to_position(current_prices.get(trade.symbol))
            for trade in self.open_trades
        )
        return Portfolio(positions=positions, cash_balance=self.current_balance)


@dataclass(frozen=True)
class TradingFees:
    """Immutable trading fees representation."""

    entry_fee: Decimal
    exit_fee: Decimal
    slippage_cost: Decimal
    fee_rate: Decimal

    @property
    def total_fees(self) -> Decimal:
        """Calculate total fees (pure function)."""
        return self.entry_fee + self.exit_fee + self.slippage_cost


@dataclass(frozen=True)
class TradeExecution:
    """Immutable trade execution result."""

    success: bool
    trade_state: PaperTradeState | None
    fees: TradingFees
    execution_price: Decimal
    slippage_amount: Decimal
    reason: str = ""

    @classmethod
    def success_result(
        cls,
        trade_state: PaperTradeState,
        fees: TradingFees,
        execution_price: Decimal,
        slippage_amount: Decimal,
    ) -> "TradeExecution":
        """Create successful execution result."""
        return cls(
            success=True,
            trade_state=trade_state,
            fees=fees,
            execution_price=execution_price,
            slippage_amount=slippage_amount,
        )

    @classmethod
    def failure_result(cls, reason: str) -> "TradeExecution":
        """Create failed execution result."""
        return cls(
            success=False,
            trade_state=None,
            fees=TradingFees(Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
            execution_price=Decimal(0),
            slippage_amount=Decimal(0),
            reason=reason,
        )


@dataclass(frozen=True)
class AccountUpdate:
    """Immutable account update result."""

    success: bool
    new_state: PaperTradingAccountState | None
    operation: str
    reason: str = ""

    @classmethod
    def success_update(
        cls, new_state: PaperTradingAccountState, operation: str
    ) -> "AccountUpdate":
        """Create successful update result."""
        return cls(success=True, new_state=new_state, operation=operation)

    @classmethod
    def failure_update(cls, operation: str, reason: str) -> "AccountUpdate":
        """Create failed update result."""
        return cls(success=False, new_state=None, operation=operation, reason=reason)


# Union types for pattern matching
TradeStateTransition = Union[PaperTradeState, TradeExecution]
AccountStateTransition = Union[PaperTradingAccountState, AccountUpdate]


# Helper functions for creating trades
def create_paper_trade(
    symbol: str,
    side: str,
    size: Decimal,
    entry_price: Decimal,
    entry_time: datetime | None = None,
    fees: Decimal = Decimal(0),
) -> PaperTradeState:
    """Create a new paper trade (pure function)."""
    trade_id = f"paper_{uuid4().hex[:8]}"
    time = entry_time or datetime.now()

    return PaperTradeState(
        id=trade_id,
        symbol=symbol,
        side=side,
        entry_time=time,
        entry_price=entry_price,
        size=size,
        fees=fees,
    )


def calculate_trade_fees(
    trade_value: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    has_exit_fee: bool = False,
) -> TradingFees:
    """Calculate trading fees (pure function)."""
    entry_fee = trade_value * fee_rate
    exit_fee = trade_value * fee_rate if has_exit_fee else Decimal(0)
    slippage_cost = trade_value * slippage_rate

    return TradingFees(
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        slippage_cost=slippage_cost,
        fee_rate=fee_rate,
    )


def apply_slippage(price: Decimal, side: str, slippage_rate: Decimal) -> Decimal:
    """Apply slippage to execution price (pure function)."""
    slippage_amount = price * slippage_rate

    if side in ["LONG", "BUY"]:
        return price + slippage_amount  # Pay slightly more when buying
    # SHORT, SELL
    return price - slippage_amount  # Receive slightly less when selling


def calculate_required_margin(trade_value: Decimal, leverage: Decimal) -> Decimal:
    """Calculate required margin for trade (pure function)."""
    return trade_value / leverage


def validate_trade_size(
    size: Decimal, available_balance: Decimal, price: Decimal, leverage: Decimal
) -> bool:
    """Validate if trade size is affordable (pure function)."""
    trade_value = size * price
    required_margin = calculate_required_margin(trade_value, leverage)
    return required_margin <= available_balance
