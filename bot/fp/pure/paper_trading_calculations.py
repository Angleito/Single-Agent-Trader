"""Pure functions for paper trading calculations."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple

from ..types.paper_trading import (
    AccountUpdate,
    PaperTradingAccountState,
    PaperTradeState,
    TradeExecution,
    TradingFees,
    apply_slippage,
    calculate_required_margin,
    calculate_trade_fees,
    create_paper_trade,
    validate_trade_size,
)
from ..types.portfolio import Portfolio, PortfolioMetrics
from ..types.effects import Result, Ok, Err


def calculate_position_size(
    equity: Decimal,
    size_percentage: Decimal,
    leverage: Decimal,
    current_price: Decimal,
    is_futures: bool = False,
    contract_size: Optional[Decimal] = None,
    fixed_contracts: Optional[int] = None,
) -> Decimal:
    """
    Calculate position size based on equity and parameters (pure function).
    
    Args:
        equity: Available equity
        size_percentage: Percentage of equity to use (0-100)
        leverage: Trading leverage multiplier
        current_price: Current asset price
        is_futures: Whether this is futures trading
        contract_size: Size per contract for futures (e.g., 0.1 ETH)
        fixed_contracts: Fixed number of contracts to use
        
    Returns:
        Calculated position size
    """
    if size_percentage <= 0 or size_percentage > 100:
        return Decimal(0)
    
    # Calculate position value based on percentage of equity
    position_value = equity * (size_percentage / 100)
    leveraged_value = position_value * leverage
    
    if is_futures and contract_size is not None:
        if fixed_contracts is not None:
            # Use fixed number of contracts
            return contract_size * fixed_contracts
        else:
            # Calculate quantity in base asset
            quantity_in_asset = leveraged_value / current_price
            # Convert to number of contracts and round down
            num_contracts = int(quantity_in_asset / contract_size)
            num_contracts = max(1, num_contracts)  # Minimum 1 contract
            return contract_size * num_contracts
    else:
        # For spot trading or non-futures
        return leveraged_value / current_price


def simulate_trade_execution(
    account_state: PaperTradingAccountState,
    symbol: str,
    side: str,
    size_percentage: Decimal,
    current_price: Decimal,
    leverage: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    execution_time: Optional[datetime] = None,
    is_futures: bool = False,
    contract_size: Optional[Decimal] = None,
    fixed_contracts: Optional[int] = None,
) -> Tuple[TradeExecution, Optional[PaperTradingAccountState]]:
    """
    Simulate trade execution and return results (pure function).
    
    Args:
        account_state: Current account state
        symbol: Trading symbol
        side: Trade side (LONG/SHORT)
        size_percentage: Percentage of equity to use
        current_price: Current market price
        leverage: Trading leverage
        fee_rate: Trading fee rate
        slippage_rate: Slippage rate
        execution_time: Trade execution time
        is_futures: Whether futures trading
        contract_size: Contract size for futures
        fixed_contracts: Fixed contract count
        
    Returns:
        Tuple of (TradeExecution, Optional[PaperTradingAccountState])
    """
    time = execution_time or datetime.now()
    
    # Calculate position size
    trade_size = calculate_position_size(
        equity=account_state.equity,
        size_percentage=size_percentage,
        leverage=leverage,
        current_price=current_price,
        is_futures=is_futures,
        contract_size=contract_size,
        fixed_contracts=fixed_contracts,
    )
    
    if trade_size <= 0:
        return TradeExecution.failure_result("Invalid trade size calculated"), None
    
    # Apply slippage
    execution_price = apply_slippage(current_price, side, slippage_rate)
    slippage_amount = abs(execution_price - current_price)
    
    # Calculate trade value and fees
    trade_value = trade_size * execution_price
    required_margin = calculate_required_margin(trade_value, leverage)
    fees = calculate_trade_fees(trade_value, fee_rate, slippage_rate)
    
    # Check if trade is affordable
    available_margin = account_state.get_available_margin()
    total_cost = required_margin + fees.total_fees
    
    if total_cost > available_margin:
        return (
            TradeExecution.failure_result(
                f"Insufficient funds: need {total_cost}, have {available_margin}"
            ),
            None,
        )
    
    # Check for existing position
    existing_trade = account_state.find_open_trade(symbol)
    if existing_trade:
        if existing_trade.side == side:
            # Same direction - would need to handle position increase
            return (
                TradeExecution.failure_result(
                    f"Position increase not supported in pure calculations"
                ),
                None,
            )
        else:
            # Opposite direction - would need to close existing first
            return (
                TradeExecution.failure_result(
                    f"Position reversal requires closing existing position first"
                ),
                None,
            )
    
    # Create new trade
    new_trade = create_paper_trade(
        symbol=symbol,
        side=side,
        size=trade_size,
        entry_price=execution_price,
        entry_time=time,
        fees=fees.entry_fee,
    )
    
    # Update account state
    new_account_state = (
        account_state
        .add_trade(new_trade)
        .update_balance(-fees.total_fees, f"Trade fees for {symbol}")
        .update_margin(required_margin)
    )
    
    execution = TradeExecution.success_result(
        trade_state=new_trade,
        fees=fees,
        execution_price=execution_price,
        slippage_amount=slippage_amount,
    )
    
    return execution, new_account_state


def simulate_position_close(
    account_state: PaperTradingAccountState,
    symbol: str,
    exit_price: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    exit_time: Optional[datetime] = None,
) -> Tuple[TradeExecution, Optional[PaperTradingAccountState]]:
    """
    Simulate closing a position (pure function).
    
    Args:
        account_state: Current account state
        symbol: Symbol to close
        exit_price: Exit price
        fee_rate: Trading fee rate
        slippage_rate: Slippage rate
        exit_time: Exit time
        
    Returns:
        Tuple of (TradeExecution, Optional[PaperTradingAccountState])
    """
    time = exit_time or datetime.now()
    
    # Find existing trade
    existing_trade = account_state.find_open_trade(symbol)
    if not existing_trade:
        return TradeExecution.failure_result(f"No open position for {symbol}"), None
    
    # Apply slippage to exit price
    # For closing, reverse the slippage direction
    opposite_side = "SELL" if existing_trade.side == "LONG" else "BUY"
    execution_price = apply_slippage(exit_price, opposite_side, slippage_rate)
    slippage_amount = abs(execution_price - exit_price)
    
    # Calculate exit fees
    trade_value = existing_trade.size * execution_price
    exit_fees = calculate_trade_fees(
        trade_value, fee_rate, slippage_rate, has_exit_fee=True
    )
    
    # Close the trade and update account
    new_account_state = account_state.close_trade(
        trade_id=existing_trade.id,
        exit_price=execution_price,
        exit_time=time,
        additional_fees=exit_fees.total_fees,
    )
    
    # Create closed trade state for the execution result
    closed_trade = existing_trade.close_trade(
        exit_price=execution_price,
        exit_time=time,
        additional_fees=exit_fees.total_fees,
    )
    
    execution = TradeExecution.success_result(
        trade_state=closed_trade,
        fees=exit_fees,
        execution_price=execution_price,
        slippage_amount=slippage_amount,
    )
    
    return execution, new_account_state


def calculate_unrealized_pnl(
    account_state: PaperTradingAccountState, current_prices: dict[str, Decimal]
) -> Decimal:
    """Calculate total unrealized P&L for all open positions (pure function)."""
    return sum(
        trade.calculate_unrealized_pnl(
            current_prices.get(trade.symbol, trade.entry_price)
        )
        for trade in account_state.open_trades
    )


def calculate_account_metrics(
    account_state: PaperTradingAccountState, current_prices: dict[str, Decimal]
) -> dict[str, Decimal]:
    """Calculate comprehensive account metrics (pure function)."""
    unrealized_pnl = calculate_unrealized_pnl(account_state, current_prices)
    total_pnl = account_state.get_total_pnl()
    equity = account_state.current_balance + unrealized_pnl
    
    return {
        "starting_balance": account_state.starting_balance,
        "current_balance": account_state.current_balance,
        "equity": equity,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "roi_percent": account_state.get_roi_percent(),
        "margin_used": account_state.margin_used,
        "margin_available": account_state.get_available_margin(),
        "peak_equity": account_state.peak_equity,
        "max_drawdown": account_state.max_drawdown,
    }


def calculate_portfolio_performance(
    account_state: PaperTradingAccountState, current_prices: dict[str, Decimal]
) -> PortfolioMetrics:
    """Calculate portfolio performance metrics (pure function)."""
    # Convert closed trades to TradeResult tuples
    trade_results = tuple(
        result for trade in account_state.closed_trades
        if (result := trade.to_trade_result()) is not None
    )
    
    # Calculate current unrealized P&L
    current_unrealized = calculate_unrealized_pnl(account_state, current_prices)
    
    return PortfolioMetrics.from_trades(
        trades=trade_results,
        current_unrealized=current_unrealized,
    )


def calculate_win_rate(account_state: PaperTradingAccountState) -> float:
    """Calculate win rate from closed trades (pure function)."""
    if not account_state.closed_trades:
        return 0.0
    
    winning_trades = sum(
        1 for trade in account_state.closed_trades
        if trade.realized_pnl is not None and trade.realized_pnl > 0
    )
    
    return (winning_trades / len(account_state.closed_trades)) * 100


def calculate_average_trade_duration(account_state: PaperTradingAccountState) -> float:
    """Calculate average trade duration in hours (pure function)."""
    if not account_state.closed_trades:
        return 0.0
    
    durations = []
    for trade in account_state.closed_trades:
        if trade.exit_time and trade.entry_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            durations.append(duration)
    
    return sum(durations) / len(durations) if durations else 0.0


def calculate_largest_win_loss(
    account_state: PaperTradingAccountState,
) -> Tuple[Decimal, Decimal]:
    """Calculate largest win and loss (pure function)."""
    if not account_state.closed_trades:
        return Decimal(0), Decimal(0)
    
    pnl_values = [
        trade.realized_pnl for trade in account_state.closed_trades
        if trade.realized_pnl is not None
    ]
    
    if not pnl_values:
        return Decimal(0), Decimal(0)
    
    largest_win = max(pnl_values)
    largest_loss = min(pnl_values)
    
    return largest_win, largest_loss


def calculate_profit_factor(account_state: PaperTradingAccountState) -> float:
    """Calculate profit factor (pure function)."""
    if not account_state.closed_trades:
        return 0.0
    
    gross_profit = sum(
        trade.realized_pnl for trade in account_state.closed_trades
        if trade.realized_pnl is not None and trade.realized_pnl > 0
    )
    
    gross_loss = sum(
        abs(trade.realized_pnl) for trade in account_state.closed_trades
        if trade.realized_pnl is not None and trade.realized_pnl < 0
    )
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


def validate_account_state(account_state: PaperTradingAccountState) -> bool:
    """Validate account state consistency (pure function)."""
    try:
        # Check basic constraints
        if account_state.starting_balance <= 0:
            return False
        
        if account_state.margin_used < 0:
            return False
        
        if account_state.trade_counter < 0:
            return False
        
        # Check trade consistency
        if len(account_state.open_trades) + len(account_state.closed_trades) > account_state.trade_counter:
            return False
        
        # Check that all closed trades have required fields
        for trade in account_state.closed_trades:
            if trade.status != "CLOSED":
                return False
            if trade.exit_time is None or trade.exit_price is None:
                return False
        
        # Check that open trades don't have exit data
        for trade in account_state.open_trades:
            if trade.status == "CLOSED":
                return False
            if trade.exit_time is not None or trade.exit_price is not None:
                return False
        
        return True
    
    except Exception:
        return False


def normalize_decimal_precision(value: Decimal, decimal_places: int = 8) -> Decimal:
    """Normalize decimal to specified precision (pure function)."""
    if decimal_places == 2:
        return value.quantize(Decimal("0.01"))
    elif decimal_places == 8:
        return value.quantize(Decimal("0.00000001"))
    else:
        quantizer = Decimal(10) ** (-decimal_places)
        return value.quantize(quantizer)


def calculate_drawdown_series(account_state: PaperTradingAccountState) -> list[float]:
    """Calculate drawdown series over time (pure function)."""
    if not account_state.closed_trades:
        return []
    
    # Sort trades by exit time
    sorted_trades = sorted(
        account_state.closed_trades,
        key=lambda t: t.exit_time or t.entry_time
    )
    
    # Calculate cumulative balance
    running_balance = float(account_state.starting_balance)
    peak_balance = running_balance
    drawdowns = []
    
    for trade in sorted_trades:
        if trade.realized_pnl is not None:
            running_balance += float(trade.realized_pnl)
            peak_balance = max(peak_balance, running_balance)
            
            if peak_balance > 0:
                drawdown = ((peak_balance - running_balance) / peak_balance) * 100
                drawdowns.append(drawdown)
            else:
                drawdowns.append(0.0)
    
    return drawdowns


# Additional pure functions for functional position management tests

def calculate_position_value(size: Decimal, price: Decimal) -> Decimal:
    """
    Calculate the total value of a position.
    
    Args:
        size: Position size in units
        price: Current price per unit
        
    Returns:
        Total position value
    """
    return size * price


def calculate_unrealized_pnl_simple(
    side: str, size: Decimal, entry_price: Decimal, current_price: Decimal
) -> Decimal:
    """
    Calculate unrealized profit/loss for a position (simple version).
    
    Args:
        side: Position side ("LONG", "SHORT", or "FLAT")
        size: Position size in units
        entry_price: Entry price per unit
        current_price: Current market price per unit
        
    Returns:
        Unrealized P&L amount
    """
    if side == "FLAT" or size == 0:
        return Decimal("0.0")
    
    if side == "LONG":
        return size * (current_price - entry_price)
    elif side == "SHORT":
        return size * (entry_price - current_price)
    else:
        return Decimal("0.0")


def calculate_margin_requirement_simple(position_value: Decimal, leverage: Decimal) -> Decimal:
    """
    Calculate margin requirement for a position (simple version).
    
    Args:
        position_value: Total value of the position
        leverage: Trading leverage (e.g., 5.0 for 5x leverage)
        
    Returns:
        Required margin amount
    """
    if leverage <= 0:
        return position_value  # No leverage = full margin
    
    return position_value / leverage


def calculate_fees_simple(position_value: Decimal, fee_rate: Decimal) -> Decimal:
    """
    Calculate trading fees for a position (simple version).
    
    Args:
        position_value: Total value of the position
        fee_rate: Fee rate as decimal (e.g., 0.001 for 0.1%)
        
    Returns:
        Fee amount
    """
    return position_value * fee_rate


def validate_position_size_simple(size: Decimal) -> Result[Decimal, str]:
    """
    Validate position size.
    
    Args:
        size: Position size to validate
        
    Returns:
        Result containing validated size or error message
    """
    if size < 0:
        return Err("Invalid position size: cannot be negative")
    
    if size == 0:
        return Err("Invalid position size: cannot be zero")
    
    return Ok(size)


def calculate_stop_loss_distance(entry_price: Decimal, stop_loss_pct: Decimal) -> Decimal:
    """
    Calculate stop loss distance from entry price.
    
    Args:
        entry_price: Position entry price
        stop_loss_pct: Stop loss percentage
        
    Returns:
        Stop loss distance amount
    """
    return entry_price * (stop_loss_pct / Decimal("100.0"))


def calculate_take_profit_distance(entry_price: Decimal, take_profit_pct: Decimal) -> Decimal:
    """
    Calculate take profit distance from entry price.
    
    Args:
        entry_price: Position entry price
        take_profit_pct: Take profit percentage
        
    Returns:
        Take profit distance amount
    """
    return entry_price * (take_profit_pct / Decimal("100.0"))