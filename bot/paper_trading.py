"""
Enhanced paper trading system for realistic trading simulation.

This module provides a comprehensive paper trading system that simulates
real trading conditions including:
- Realistic account balance tracking
- Trade execution with fees and slippage
- Position management and P&L tracking
- Performance analytics and reporting
- State persistence between sessions
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from .config import settings
from .fee_calculator import fee_calculator
from .trading_types import Order, OrderStatus, TradeAction

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a paper trade with entry and exit information."""

    id: str
    symbol: str
    side: str  # LONG or SHORT
    entry_time: datetime
    entry_price: Decimal
    size: Decimal
    exit_time: datetime | None = None
    exit_price: Decimal | None = None
    realized_pnl: Decimal | None = None
    fees: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    status: str = "OPEN"  # OPEN, CLOSED, PARTIAL

    def calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L based on current price."""
        if self.status == "CLOSED":
            return Decimal("0")

        price_diff = current_price - self.entry_price
        if self.side == "LONG":
            return self.size * price_diff - self.fees
        else:  # SHORT
            return self.size * (-price_diff) - self.fees

    def close_trade(
        self,
        exit_price: Decimal,
        exit_time: datetime,
        additional_fees: Decimal = Decimal("0"),
    ) -> Decimal:
        """Close the trade and calculate realized P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = "CLOSED"

        price_diff = exit_price - self.entry_price
        if self.side == "LONG":
            gross_pnl = self.size * price_diff
        else:  # SHORT
            gross_pnl = self.size * (-price_diff)

        total_fees = self.fees + additional_fees
        self.realized_pnl = gross_pnl - total_fees
        return self.realized_pnl


@dataclass
class DailyPerformance:
    """Daily performance metrics."""

    date: str
    starting_balance: Decimal
    ending_balance: Decimal
    trades_opened: int
    trades_closed: int
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    fees_paid: Decimal
    win_rate: float
    largest_win: Decimal
    largest_loss: Decimal
    drawdown: Decimal


class PaperTradingAccount:
    """
    Paper trading account simulation with realistic features.

    Simulates a trading account with:
    - Starting balance and equity tracking
    - Position management with leverage
    - Fee calculation and slippage simulation
    - Performance tracking and analytics
    - State persistence between sessions
    """

    def __init__(self, starting_balance: Decimal = None, data_dir: Path | None = None):
        """
        Initialize paper trading account.

        Args:
            starting_balance: Initial account balance (default from config)
            data_dir: Directory for state persistence
        """
        self.data_dir = data_dir or Path("data/paper_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Account state
        self.starting_balance = starting_balance or Decimal(
            str(getattr(settings.paper_trading, "starting_balance", 10000))
        )
        self.current_balance = self.starting_balance
        self.equity = self.starting_balance
        self.margin_used = Decimal("0")

        # Trading state
        self.open_trades: dict[str, PaperTrade] = {}
        self.closed_trades: list[PaperTrade] = []
        self.trade_counter = 0

        # Performance tracking
        self.session_start_time = datetime.now(UTC)
        self.daily_metrics: dict[str, DailyPerformance] = {}
        self.peak_equity = self.starting_balance
        self.max_drawdown = Decimal("0")

        # Configuration
        self.fee_rate = Decimal(
            str(getattr(settings.paper_trading, "fee_rate", 0.001))
        )  # 0.1%
        self.slippage_rate = Decimal(
            str(getattr(settings.paper_trading, "slippage_rate", 0.0005))
        )  # 0.05%

        # Thread safety
        self._lock = threading.RLock()

        # State files
        self.account_file = self.data_dir / "account.json"
        self.trades_file = self.data_dir / "trades.json"
        self.performance_file = self.data_dir / "performance.json"

        # Load persisted state
        self._load_state()

        logger.info(
            f"Initialized paper trading account with ${self.current_balance:,.2f}"
        )

    def get_account_status(
        self, current_prices: dict[str, Decimal] = None
    ) -> dict[str, Any]:
        """Get current account status."""
        with self._lock:
            # Calculate unrealized P&L for open trades
            unrealized_pnl = Decimal("0")
            for trade in self.open_trades.values():
                if current_prices and trade.symbol in current_prices:
                    price = current_prices[trade.symbol]
                else:
                    price = self._get_current_price(trade.symbol)
                unrealized_pnl += trade.calculate_unrealized_pnl(price)

            self.equity = self.current_balance + unrealized_pnl

            # Calculate performance metrics
            total_pnl = self.equity - self.starting_balance
            roi = (
                (total_pnl / self.starting_balance * 100)
                if self.starting_balance > 0
                else Decimal("0")
            )

            # Update drawdown
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            current_drawdown = (
                (self.peak_equity - self.equity) / self.peak_equity * 100
                if self.peak_equity > 0
                else Decimal("0")
            )
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            return {
                "starting_balance": float(self.starting_balance),
                "current_balance": float(self.current_balance),
                "equity": float(self.equity),
                "unrealized_pnl": float(unrealized_pnl),
                "total_pnl": float(total_pnl),
                "roi_percent": float(roi),
                "margin_used": float(self.margin_used),
                "margin_available": float(self.equity - self.margin_used),
                "open_positions": len(self.open_trades),
                "total_trades": len(self.closed_trades),
                "peak_equity": float(self.peak_equity),
                "current_drawdown": float(current_drawdown),
                "max_drawdown": float(self.max_drawdown),
            }

    def execute_trade_action(
        self, action: TradeAction, symbol: str, current_price: Decimal
    ) -> Order | None:
        """
        Execute a trade action in paper trading mode using real market data.

        This method simulates what would happen if the trade were placed on the real exchange,
        using actual market prices and comprehensive logging to show the trading decisions.

        Args:
            action: Trade action to execute
            symbol: Trading symbol
            current_price: Current market price (from real market data)

        Returns:
            Order object representing the simulated trade
        """
        with self._lock:
            try:
                if action.action == "HOLD":
                    logger.info(
                        f"🛑 PAPER TRADING DECISION: HOLD | Symbol: {symbol} | "
                        f"Current Price: ${current_price} | Reason: {action.rationale}"
                    )
                    return None

                # Log the trading decision with full context
                logger.info(
                    f"🎯 PAPER TRADING DECISION: {action.action} | Symbol: {symbol} | "
                    f"Current Price: ${current_price} | Size: {action.size_pct}% | "
                    f"Reason: {action.rationale}"
                )

                # Calculate trade size based on action
                trade_size = self._calculate_trade_size(action, current_price, symbol)
                if trade_size <= 0:
                    logger.warning(f"❌ Invalid trade size calculated: {trade_size}")
                    return None

                # Log what would happen with real market execution
                trade_value = trade_size * current_price
                required_margin = trade_value / Decimal(str(settings.trading.leverage))

                logger.info(
                    f"📊 TRADE SIMULATION DETAILS:"
                    f"\n  • Symbol: {symbol}"
                    f"\n  • Action: {action.action}"
                    f"\n  • Current Real Price: ${current_price}"
                    f"\n  • Position Size: {trade_size} {symbol.split('-')[0]}"
                    f"\n  • Position Value: ${trade_value:,.2f}"
                    f"\n  • Leverage: {settings.trading.leverage}x"
                    f"\n  • Required Margin: ${required_margin:,.2f}"
                    f"\n  • Available Balance: ${self.equity - self.margin_used:,.2f}"
                    f"\n  • Stop Loss: {action.stop_loss_pct}%"
                    f"\n  • Take Profit: {action.take_profit_pct}%"
                )

                # Apply realistic slippage based on real market conditions
                execution_price = self._apply_slippage(current_price, action.action)
                slippage_amount = abs(execution_price - current_price)
                slippage_pct = (slippage_amount / current_price) * 100

                logger.info(
                    f"📈 EXECUTION SIMULATION:"
                    f"\n  • Market Price: ${current_price}"
                    f"\n  • Execution Price: ${execution_price} (slippage: {slippage_pct:.3f}%)"
                    f"\n  • Slippage Cost: ${slippage_amount * trade_size:.2f}"
                )

                # Calculate realistic trading fees using the fee calculator
                trade_fees = fee_calculator.calculate_trade_fees(
                    action, trade_value, execution_price, is_market_order=True
                )

                # For paper trading, we only charge entry fee for now
                fees = trade_fees.entry_fee

                logger.info(
                    f"💰 FEE CALCULATION:"
                    f"\n  • Fee Rate: {trade_fees.fee_rate:.4%}"
                    f"\n  • Entry Fee: ${fees:.4f}"
                    f"\n  • Trade Value: ${trade_value:.2f}"
                )

                # Check available balance
                if action.action in ["LONG", "SHORT"]:
                    # Opening new position
                    if required_margin + fees > (self.equity - self.margin_used):
                        logger.warning(
                            f"❌ INSUFFICIENT FUNDS SIMULATION:"
                            f"\n  • Required: ${required_margin + fees:,.2f}"
                            f"\n  • Available: ${self.equity - self.margin_used:,.2f}"
                            f"\n  • Shortfall: ${(required_margin + fees) - (self.equity - self.margin_used):,.2f}"
                        )
                        return self._create_failed_order(
                            action, symbol, "INSUFFICIENT_FUNDS"
                        )

                # Execute the simulated trade
                order = self._execute_paper_trade(
                    action, symbol, execution_price, trade_size, fees
                )

                if order and order.status == OrderStatus.FILLED:
                    # Calculate potential profit/loss scenarios
                    stop_loss_price = (
                        execution_price * (1 - Decimal(str(action.stop_loss_pct)) / 100)
                        if action.action == "LONG"
                        else execution_price
                        * (1 + Decimal(str(action.stop_loss_pct)) / 100)
                    )
                    take_profit_price = (
                        execution_price
                        * (1 + Decimal(str(action.take_profit_pct)) / 100)
                        if action.action == "LONG"
                        else execution_price
                        * (1 - Decimal(str(action.take_profit_pct)) / 100)
                    )

                    stop_loss_pnl = (
                        (stop_loss_price - execution_price) * trade_size
                        if action.action == "LONG"
                        else (execution_price - stop_loss_price) * trade_size
                    )
                    take_profit_pnl = (
                        (take_profit_price - execution_price) * trade_size
                        if action.action == "LONG"
                        else (execution_price - take_profit_price) * trade_size
                    )

                    # Add comprehensive execution logging
                    if settings.trading.enable_futures and symbol == "ETH-USD":
                        num_contracts = int(trade_size / Decimal("0.1"))
                        logger.info(
                            f"✅ PAPER TRADING FUTURES EXECUTION COMPLETE:"
                            f"\n  🎯 Action: {action.action}"
                            f"\n  📊 Contracts: {num_contracts} contracts ({trade_size} ETH)"
                            f"\n  💵 Price: ${execution_price}"
                            f"\n  💸 Value: ${trade_size * execution_price:.2f}"
                            f"\n  🏷️ Fees: ${fees:.4f} @ {trade_fees.fee_rate:.4%}"
                            f"\n  🔴 Stop Loss: ${stop_loss_price:.2f} (${stop_loss_pnl:+.2f})"
                            f"\n  🟢 Take Profit: ${take_profit_price:.2f} (${take_profit_pnl:+.2f})"
                            f"\n  🏦 New Balance: ${self.current_balance:.2f}"
                            f"\n  📈 New Equity: ${self.equity:.2f}"
                            f"\n  🔒 Margin Used: ${self.margin_used:.2f}"
                            f"\n  💳 Available: ${self.equity - self.margin_used:.2f}"
                            f"\n  📋 Order ID: {order.id}"
                        )
                    else:
                        logger.info(
                            f"✅ PAPER TRADING EXECUTION COMPLETE:"
                            f"\n  🎯 Action: {action.action}"
                            f"\n  📊 Size: {trade_size} {symbol}"
                            f"\n  💵 Price: ${execution_price}"
                            f"\n  💸 Value: ${trade_size * execution_price:.2f}"
                            f"\n  🏷️ Fees: ${fees:.4f} @ {trade_fees.fee_rate:.4%}"
                            f"\n  🔴 Stop Loss: ${stop_loss_price:.2f} (${stop_loss_pnl:+.2f})"
                            f"\n  🟢 Take Profit: ${take_profit_price:.2f} (${take_profit_pnl:+.2f})"
                            f"\n  🏦 New Balance: ${self.current_balance:.2f}"
                            f"\n  📈 New Equity: ${self.equity:.2f}"
                            f"\n  🔒 Margin Used: ${self.margin_used:.2f}"
                            f"\n  💳 Available: ${self.equity - self.margin_used:.2f}"
                            f"\n  📋 Order ID: {order.id}"
                        )

                    # Also log a summary for easy scanning
                    logger.info(
                        f"🎬 TRADE SUMMARY: {action.action} {trade_size} {symbol} @ ${execution_price} | "
                        f"Value: ${trade_value:.2f} | Balance: ${self.current_balance:.2f} → ${self.equity:.2f}"
                    )

                return order

            except Exception as e:
                logger.error(f"❌ Error simulating paper trade: {e}")
                return self._create_failed_order(action, symbol, f"ERROR: {str(e)}")

    def _calculate_trade_size(
        self, action: TradeAction, current_price: Decimal, symbol: str
    ) -> Decimal:
        """Calculate trade size based on action parameters."""
        if action.action == "CLOSE":
            # Close existing position
            existing_position = self._find_open_position(symbol)
            return existing_position.size if existing_position else Decimal("0")

        # Calculate position size based on percentage of equity
        position_value = self.equity * (Decimal(str(action.size_pct)) / 100)
        leveraged_value = position_value * Decimal(str(settings.trading.leverage))

        # Check if this is a futures trade for ETH
        symbol = action.symbol if hasattr(action, "symbol") else settings.trading.symbol
        if settings.trading.enable_futures and symbol == "ETH-USD":
            # Apply futures contract size logic for ETH
            CONTRACT_SIZE = Decimal("0.1")  # 0.1 ETH per contract

            # Check if we're using fixed contract size from config
            if settings.trading.fixed_contract_size:
                # Use fixed number of contracts
                num_contracts = settings.trading.fixed_contract_size
                trade_size = CONTRACT_SIZE * num_contracts
                logger.debug(
                    f"Using fixed contract size: {num_contracts} contracts = {trade_size} ETH"
                )
            else:
                # Calculate quantity in ETH based on position value
                quantity_in_eth = leveraged_value / current_price

                # Convert to number of contracts and round down
                num_contracts = int(quantity_in_eth / CONTRACT_SIZE)
                num_contracts = max(1, num_contracts)  # Minimum 1 contract

                # Return the actual quantity in ETH (multiples of 0.1)
                trade_size = CONTRACT_SIZE * num_contracts

                logger.debug(
                    f"Futures contract calculation: {quantity_in_eth:.6f} ETH -> "
                    f"{num_contracts} contracts = {trade_size} ETH"
                )
        else:
            # For spot trading or non-ETH futures, use the original calculation
            trade_size = leveraged_value / current_price
            # Round to reasonable precision
            trade_size = trade_size.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)

        return trade_size

    def _apply_slippage(self, price: Decimal, action: str) -> Decimal:
        """Apply realistic slippage to trade execution."""
        slippage_amount = price * self.slippage_rate

        # Market orders typically get worse fill prices
        if action in ["LONG", "BUY"]:
            return price + slippage_amount  # Pay slightly more when buying
        else:  # SHORT, SELL
            return price - slippage_amount  # Receive slightly less when selling

    def _execute_paper_trade(
        self,
        action: TradeAction,
        symbol: str,
        price: Decimal,
        size: Decimal,
        fees: Decimal,
    ) -> Order:
        """Execute the actual paper trade."""
        trade_id = f"paper_{self.trade_counter:06d}"
        self.trade_counter += 1
        current_time = datetime.now(UTC)

        if action.action == "CLOSE":
            # Close existing position
            return self._close_position(symbol, price, current_time, fees)

        elif action.action in ["LONG", "SHORT"]:
            # Check if we already have an open position for this symbol
            existing_position = self._find_open_position(symbol)
            if existing_position:
                # Handle different scenarios for existing positions
                if existing_position.side == action.action:
                    # Same direction - increase position size
                    logger.info(
                        f"Adding to existing {action.action} position for {symbol}"
                    )
                    return self._increase_position(
                        existing_position, price, size, fees, current_time
                    )
                else:
                    # Opposite direction - close existing position and open new one
                    logger.info(
                        f"Closing existing {existing_position.side} position and opening new {action.action} position for {symbol}"
                    )
                    # First close the existing position
                    close_order = self._close_position(
                        symbol, price, current_time, fees
                    )
                    if close_order.status != OrderStatus.FILLED:
                        logger.error(
                            f"Failed to close existing position: {close_order.status}"
                        )
                        return self._create_failed_order(action, symbol, "CLOSE_FAILED")

                    # Now continue to open the new position (fall through to position opening logic)

            # Open new position
            trade = PaperTrade(
                id=trade_id,
                symbol=symbol,
                side=action.action,
                entry_time=current_time,
                entry_price=price,
                size=size,
                fees=fees,
                slippage=abs(price - (price / (1 + self.slippage_rate))),
            )

            self.open_trades[trade_id] = trade

            # Update account balance and margin
            trade_value = size * price
            required_margin = trade_value / Decimal(str(settings.trading.leverage))
            self.margin_used += required_margin
            self.current_balance -= fees  # Deduct fees immediately

            # Create order object
            order = Order(
                id=trade_id,
                symbol=symbol,
                side="BUY" if action.action == "LONG" else "SELL",
                type="MARKET",
                quantity=size,
                price=price,
                filled_quantity=size,
                status=OrderStatus.FILLED,
                timestamp=current_time,
            )

            # Log with contract information for futures
            if settings.trading.enable_futures and symbol == "ETH-USD":
                num_contracts = int(size / Decimal("0.1"))
                logger.info(
                    f"📈 Paper Trading FUTURES: Opened {action.action} position | "
                    f"{num_contracts} contracts ({size} ETH) @ ${price} | Trade ID: {trade_id}"
                )
            else:
                logger.info(
                    f"📈 Paper Trading: Opened {action.action} position | {size} {symbol} @ ${price} | Trade ID: {trade_id}"
                )

            # Save state immediately after opening position
            self._save_state()
            return order

        return self._create_failed_order(action, symbol, "INVALID_ACTION")

    def _close_position(
        self, symbol: str, price: Decimal, close_time: datetime, fees: Decimal
    ) -> Order:
        """Close an existing position."""
        # Find open position for symbol
        trade_to_close = None
        for trade in self.open_trades.values():
            if trade.symbol == symbol:
                trade_to_close = trade
                break

        if not trade_to_close:
            logger.warning(f"No open position found for {symbol}")
            return self._create_failed_order(
                TradeAction(
                    action="CLOSE",
                    size_pct=0,
                    take_profit_pct=0.0,
                    stop_loss_pct=0.0,
                    rationale="No position",
                ),
                symbol,
                "NO_POSITION",
            )

        # Calculate realized P&L
        realized_pnl = trade_to_close.close_trade(price, close_time, fees)

        # Update account
        self.current_balance += realized_pnl - fees  # Add P&L, subtract fees
        trade_value = trade_to_close.size * trade_to_close.entry_price
        released_margin = trade_value / Decimal(str(settings.trading.leverage))
        self.margin_used -= released_margin

        # Move to closed trades
        self.closed_trades.append(trade_to_close)
        del self.open_trades[trade_to_close.id]

        # Create order
        order = Order(
            id=f"close_{trade_to_close.id}",
            symbol=symbol,
            side="SELL" if trade_to_close.side == "LONG" else "BUY",
            type="MARKET",
            quantity=trade_to_close.size,
            price=price,
            filled_quantity=trade_to_close.size,
            status=OrderStatus.FILLED,
            timestamp=close_time,
        )

        logger.info(
            f"📊 Paper Trading: Closed {trade_to_close.side} position | {trade_to_close.size} {symbol} @ ${price} | "
            f"P&L: ${realized_pnl:.2f} ({'+' if realized_pnl > 0 else ''}{realized_pnl/trade_to_close.entry_price*100:.2f}%) | "
            f"{'✅ WIN' if realized_pnl > 0 else '❌ LOSS'}"
        )

        # Save state immediately after closing position
        self._save_state()
        return order

    def _find_open_position(self, symbol: str) -> PaperTrade | None:
        """Find open position for a symbol."""
        for trade in self.open_trades.values():
            if trade.symbol == symbol:
                return trade
        return None

    def _increase_position(
        self,
        existing_position: PaperTrade,
        price: Decimal,
        size: Decimal,
        fees: Decimal,
        current_time: datetime,
    ) -> Order:
        """Increase the size of an existing position."""
        trade_id = f"paper_{self.trade_counter:06d}"
        self.trade_counter += 1

        # Calculate average entry price (weighted by position sizes)
        current_value = existing_position.size * existing_position.entry_price
        new_value = size * price
        total_size = existing_position.size + size
        average_price = (current_value + new_value) / total_size

        # Update existing position
        existing_position.size = total_size
        existing_position.entry_price = average_price
        existing_position.fees += fees

        # Update account balance and margin
        trade_value = size * price
        required_margin = trade_value / Decimal(str(settings.trading.leverage))
        self.margin_used += required_margin
        self.current_balance -= fees  # Deduct fees immediately

        # Create order object for the increase
        order = Order(
            id=trade_id,
            symbol=existing_position.symbol,
            side="BUY" if existing_position.side == "LONG" else "SELL",
            type="MARKET",
            quantity=size,
            price=price,
            filled_quantity=size,
            status=OrderStatus.FILLED,
            timestamp=current_time,
        )

        logger.info(
            f"📈 Paper Trading: Increased {existing_position.side} position | "
            f"+{size} {existing_position.symbol} @ ${price} | "
            f"Total size: {total_size} @ avg ${average_price:.4f} | Trade ID: {trade_id}"
        )

        # Save state immediately after increasing position
        self._save_state()
        return order

    def _create_failed_order(
        self, action: TradeAction, symbol: str, reason: str
    ) -> Order:
        """Create a failed order object."""
        return Order(
            id=f"failed_{self.trade_counter}",
            symbol=symbol,
            side="BUY" if action.action == "LONG" else "SELL",
            type="MARKET",
            quantity=Decimal("0"),
            price=Decimal("0"),
            filled_quantity=Decimal("0"),
            status=OrderStatus.REJECTED,
            timestamp=datetime.now(UTC),
        )

    def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol from real market data."""
        try:
            # Try to get real market data from the price conversion utility
            from .utils.price_conversion import get_current_real_price

            real_price = get_current_real_price(symbol)
            if real_price:
                logger.debug(f"📊 Using real market price for {symbol}: ${real_price}")
                return Decimal(str(real_price))
        except (ImportError, Exception) as e:
            logger.debug(f"Could not fetch real price for {symbol}: {e}")

        # Fallback to realistic prices based on corrected price data
        if "SUI" in symbol:
            price = Decimal("3.50")  # Realistic SUI price post-fix
        elif "ETH" in symbol:
            price = Decimal("2500")  # Realistic ETH price
        elif "BTC" in symbol:
            price = Decimal("50000")  # Realistic BTC price
        else:
            price = Decimal("100")  # Generic reasonable price

        logger.debug(f"📊 Using fallback price for {symbol}: ${price}")
        return price

    def update_daily_performance(self) -> None:
        """Update daily performance metrics."""
        with self._lock:
            today = datetime.now(UTC).date().isoformat()

            if today in self.daily_metrics:
                return  # Already updated today

            # Calculate daily metrics
            account_status = self.get_account_status()

            # Count trades for today
            today_trades_opened = len(
                [
                    t
                    for t in self.open_trades.values()
                    if t.entry_time.date().isoformat() == today
                ]
            )
            today_trades_closed = len(
                [
                    t
                    for t in self.closed_trades
                    if t.exit_time and t.exit_time.date().isoformat() == today
                ]
            )

            # Calculate P&L for today
            today_realized = sum(
                t.realized_pnl
                for t in self.closed_trades
                if t.exit_time
                and t.exit_time.date().isoformat() == today
                and t.realized_pnl is not None
            ) or Decimal("0")
            today_fees = sum(
                t.fees
                for t in self.closed_trades
                if t.exit_time
                and t.exit_time.date().isoformat() == today
                and t.fees is not None
            ) or Decimal("0")

            # Win rate calculation
            today_closed_trades = [
                t
                for t in self.closed_trades
                if t.exit_time and t.exit_time.date().isoformat() == today
            ]
            winning_trades = len(
                [
                    t
                    for t in today_closed_trades
                    if t.realized_pnl is not None and t.realized_pnl > 0
                ]
            )
            win_rate = (
                (winning_trades / len(today_closed_trades) * 100)
                if today_closed_trades
                else 0
            )

            # Largest win/loss
            if today_closed_trades:
                pnl_values = [
                    t.realized_pnl
                    for t in today_closed_trades
                    if t.realized_pnl is not None
                ]
                if pnl_values:
                    largest_win = max(pnl_values)
                    largest_loss = min(pnl_values)
                else:
                    largest_win = Decimal("0")
                    largest_loss = Decimal("0")
            else:
                largest_win = Decimal("0")
                largest_loss = Decimal("0")

            # Get previous day's balance
            yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
            yesterday_metrics = self.daily_metrics.get(yesterday)
            if yesterday_metrics:
                prev_balance = yesterday_metrics.ending_balance
            else:
                prev_balance = self.starting_balance

            daily_perf = DailyPerformance(
                date=today,
                starting_balance=Decimal(str(prev_balance)),
                ending_balance=Decimal(str(account_status["equity"])),
                trades_opened=today_trades_opened,
                trades_closed=today_trades_closed,
                realized_pnl=Decimal(str(today_realized)),
                unrealized_pnl=Decimal(str(account_status["unrealized_pnl"])),
                total_pnl=today_realized
                + Decimal(str(account_status["unrealized_pnl"])),
                fees_paid=today_fees,
                win_rate=win_rate,
                largest_win=Decimal(str(largest_win)),
                largest_loss=Decimal(str(largest_loss)),
                drawdown=Decimal(str(account_status["current_drawdown"])),
            )

            self.daily_metrics[today] = daily_perf
            self._save_state()

    def get_trade_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get trade history for the last N days."""
        with self._lock:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            recent_trades = [
                trade
                for trade in self.closed_trades
                if trade.exit_time and trade.exit_time >= cutoff_date
            ]

            return [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": (
                        trade.exit_time.isoformat() if trade.exit_time else None
                    ),
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price) if trade.exit_price else None,
                    "size": float(trade.size),
                    "realized_pnl": (
                        float(trade.realized_pnl) if trade.realized_pnl else None
                    ),
                    "fees": float(trade.fees),
                    "duration_hours": (
                        (trade.exit_time - trade.entry_time).total_seconds() / 3600
                        if trade.exit_time
                        else None
                    ),
                }
                for trade in recent_trades
            ]

    def get_performance_summary(self, days: int = 7) -> dict[str, Any]:
        """Get performance summary for the last N days."""
        with self._lock:
            cutoff_date = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()

            # Get relevant daily metrics
            relevant_metrics = {
                date: metrics
                for date, metrics in self.daily_metrics.items()
                if date >= cutoff_date
            }

            if not relevant_metrics:
                return {
                    "message": "No performance data available for the specified period"
                }

            # Calculate summary statistics
            total_trades = sum(m.trades_closed for m in relevant_metrics.values())
            total_realized_pnl = sum(m.realized_pnl for m in relevant_metrics.values())
            total_fees = sum(m.fees_paid for m in relevant_metrics.values())

            winning_days = len(
                [m for m in relevant_metrics.values() if m.total_pnl > 0]
            )
            total_days = len(relevant_metrics)

            avg_daily_pnl = (
                total_realized_pnl / total_days if total_days > 0 else Decimal("0")
            )
            max_daily_gain = (
                max(m.total_pnl for m in relevant_metrics.values())
                if relevant_metrics
                else Decimal("0")
            )
            max_daily_loss = (
                min(m.total_pnl for m in relevant_metrics.values())
                if relevant_metrics
                else Decimal("0")
            )

            # Win rate across all trades
            closed_trades_period = [
                t
                for t in self.closed_trades
                if t.exit_time and t.exit_time.date().isoformat() >= cutoff_date
            ]
            winning_trades = len(
                [
                    t
                    for t in closed_trades_period
                    if t.realized_pnl is not None and t.realized_pnl > 0
                ]
            )
            overall_win_rate = (
                (winning_trades / len(closed_trades_period) * 100)
                if closed_trades_period
                else 0
            )

            account_status = self.get_account_status()

            return {
                "period_days": days,
                "total_trades": total_trades,
                "total_realized_pnl": float(total_realized_pnl),
                "total_fees_paid": float(total_fees),
                "net_pnl": float(total_realized_pnl - total_fees),
                "avg_daily_pnl": float(avg_daily_pnl),
                "max_daily_gain": float(max_daily_gain),
                "max_daily_loss": float(max_daily_loss),
                "winning_days": winning_days,
                "total_days": total_days,
                "win_day_rate": (
                    (winning_days / total_days * 100) if total_days > 0 else 0
                ),
                "overall_win_rate": overall_win_rate,
                "current_equity": float(account_status["equity"]),
                "roi_percent": float(account_status["roi_percent"]),
                "max_drawdown": float(account_status["max_drawdown"]),
                "sharp_ratio": self._calculate_sharpe_ratio(relevant_metrics),
            }

    def _calculate_sharpe_ratio(
        self, daily_metrics: dict[str, DailyPerformance]
    ) -> float:
        """Calculate Sharpe ratio for the period."""
        if len(daily_metrics) < 2:
            return 0.0

        daily_returns = [
            float(m.total_pnl / m.starting_balance) for m in daily_metrics.values()
        ]

        import statistics

        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0

        # Annualized Sharpe ratio (assuming 252 trading days)
        return (mean_return * 252 / std_return) if std_return > 0 else 0

    def get_performance_metrics_for_monitor(self) -> list[dict[str, Any]]:
        """
        Get performance metrics formatted for the PerformanceMonitor system.

        Returns:
            List of metric dictionaries with name, value, timestamp, unit, and tags
        """
        with self._lock:
            metrics = []
            timestamp = datetime.now(UTC)
            account_status = self.get_account_status()

            # Core financial metrics
            metrics.extend(
                [
                    {
                        "name": "paper_trading.equity",
                        "value": account_status["equity"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.balance",
                        "value": account_status["current_balance"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.total_pnl",
                        "value": account_status["total_pnl"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.roi_percent",
                        "value": account_status["roi_percent"],
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.drawdown_percent",
                        "value": account_status["current_drawdown"],
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                    {
                        "name": "paper_trading.max_drawdown_percent",
                        "value": account_status["max_drawdown"],
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "account": "trading"},
                    },
                ]
            )

            # Trading activity metrics
            metrics.extend(
                [
                    {
                        "name": "paper_trading.open_positions",
                        "value": account_status["open_positions"],
                        "timestamp": timestamp,
                        "unit": "count",
                        "tags": {"mode": "paper", "type": "positions"},
                    },
                    {
                        "name": "paper_trading.total_trades",
                        "value": account_status["total_trades"],
                        "timestamp": timestamp,
                        "unit": "count",
                        "tags": {"mode": "paper", "type": "trades"},
                    },
                    {
                        "name": "paper_trading.margin_used",
                        "value": account_status["margin_used"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "type": "margin"},
                    },
                    {
                        "name": "paper_trading.margin_available",
                        "value": account_status["margin_available"],
                        "timestamp": timestamp,
                        "unit": "dollars",
                        "tags": {"mode": "paper", "type": "margin"},
                    },
                ]
            )

            # Calculate win rate if we have closed trades
            if self.closed_trades:
                winning_trades = len(
                    [
                        t
                        for t in self.closed_trades
                        if t.realized_pnl is not None and t.realized_pnl > 0
                    ]
                )
                win_rate = (winning_trades / len(self.closed_trades)) * 100

                metrics.append(
                    {
                        "name": "paper_trading.win_rate_percent",
                        "value": win_rate,
                        "timestamp": timestamp,
                        "unit": "percent",
                        "tags": {"mode": "paper", "type": "performance"},
                    }
                )

                # Average trade duration in hours
                durations = []
                for trade in self.closed_trades:
                    if trade.exit_time and trade.entry_time:
                        duration = (
                            trade.exit_time - trade.entry_time
                        ).total_seconds() / 3600
                        durations.append(duration)

                if durations:
                    avg_duration = sum(durations) / len(durations)
                    metrics.append(
                        {
                            "name": "paper_trading.avg_trade_duration_hours",
                            "value": avg_duration,
                            "timestamp": timestamp,
                            "unit": "hours",
                            "tags": {"mode": "paper", "type": "performance"},
                        }
                    )

            return metrics

    def generate_daily_report(self, date: str = None) -> str:
        """Generate a daily performance report."""
        target_date = date or datetime.now(UTC).date().isoformat()

        if target_date not in self.daily_metrics:
            return f"No trading data available for {target_date}"

        metrics = self.daily_metrics[target_date]
        account_status = self.get_account_status()

        report = f"""
🗓️  Daily Trading Report - {target_date}
{'=' * 50}

💰 Account Status:
   Starting Balance: ${metrics.starting_balance:,.2f}
   Ending Balance:   ${metrics.ending_balance:,.2f}
   Current Equity:   ${account_status['equity']:,.2f}
   Daily P&L:        ${metrics.total_pnl:,.2f}

📊 Trading Activity:
   Trades Opened:    {metrics.trades_opened}
   Trades Closed:    {metrics.trades_closed}
   Win Rate:         {metrics.win_rate:.1f}%
   Fees Paid:        ${metrics.fees_paid:.2f}

📈 Performance:
   Largest Win:      ${metrics.largest_win:.2f}
   Largest Loss:     ${metrics.largest_loss:.2f}
   Drawdown:         {metrics.drawdown:.2f}%
   ROI (Total):      {account_status['roi_percent']:.2f}%

🔄 Open Positions:   {account_status['open_positions']}
💸 Margin Used:      ${account_status['margin_used']:,.2f}
💳 Available:        ${account_status['margin_available']:,.2f}
"""
        return report

    def _save_state(self) -> None:
        """Save current state to files with robust error handling and logging."""
        save_start_time = time.perf_counter()

        # Use file-level locking to prevent race conditions
        with self._lock:
            try:
                logger.info("📝 Starting paper trading state save...")

                # Ensure data directory exists with proper permissions
                try:
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                    # Check write permissions
                    test_file = self.data_dir / ".write_test"
                    test_file.write_text("test")
                    test_file.unlink()
                except PermissionError as e:
                    raise OSError(
                        f"Permission denied writing to {self.data_dir}: {e}"
                    ) from e
                except Exception as e:
                    raise OSError(
                        f"Cannot access data directory {self.data_dir}: {e}"
                    ) from e

                # Save account state with detailed error context
                try:
                    account_data = {
                        "starting_balance": str(self.starting_balance),
                        "current_balance": str(self.current_balance),
                        "equity": str(self.equity),
                        "margin_used": str(self.margin_used),
                        "trade_counter": self.trade_counter,
                        "session_start_time": self.session_start_time.isoformat(),
                        "peak_equity": str(self.peak_equity),
                        "max_drawdown": str(self.max_drawdown),
                        "last_save_time": datetime.now(UTC).isoformat(),
                    }

                    # Write to temporary file first, then rename for atomic operation
                    temp_account_file = self.account_file.with_suffix(".tmp")
                    with open(temp_account_file, "w") as f:
                        json.dump(account_data, f, indent=2)
                    temp_account_file.rename(self.account_file)
                    logger.info(
                        f"✅ Account state saved: balance=${self.current_balance}, trades={self.trade_counter}"
                    )

                except Exception as e:
                    logger.error(
                        f"❌ Failed to save account state to {self.account_file}: {type(e).__name__}: {e}"
                    )
                    raise OSError(f"Account save failed: {e}") from e

                # Save trades with enhanced serialization error handling
                try:
                    trades_data = {
                        "open_trades": {
                            trade_id: asdict(trade)
                            for trade_id, trade in self.open_trades.items()
                        },
                        "closed_trades": [
                            asdict(trade) for trade in self.closed_trades
                        ],
                        "save_metadata": {
                            "save_time": datetime.now(UTC).isoformat(),
                            "open_count": len(self.open_trades),
                            "closed_count": len(self.closed_trades),
                        },
                    }

                    # Enhanced serialization with better error handling
                    def serialize_value(obj):
                        try:
                            if isinstance(obj, datetime):
                                return obj.isoformat()
                            elif isinstance(obj, Decimal):
                                return str(obj)
                            elif obj is None:
                                return None
                            else:
                                # Test if object is JSON serializable
                                json.dumps(obj)
                                return obj
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"⚠️ Serialization issue with {type(obj).__name__} '{obj}': {e}"
                            )
                            return str(obj)  # Fallback to string representation

                    def serialize_dict(d):
                        if isinstance(d, dict):
                            return {k: serialize_dict(v) for k, v in d.items()}
                        elif isinstance(d, list):
                            return [serialize_dict(item) for item in d]
                        else:
                            return serialize_value(d)

                    trades_data = serialize_dict(trades_data)

                    # Atomic write for trades file
                    temp_trades_file = self.trades_file.with_suffix(".tmp")
                    with open(temp_trades_file, "w") as f:
                        json.dump(trades_data, f, indent=2)
                    temp_trades_file.rename(self.trades_file)
                    logger.info(
                        f"✅ Trades saved: {len(self.open_trades)} open, {len(self.closed_trades)} closed"
                    )

                except Exception as e:
                    logger.error(
                        f"❌ Failed to save trades to {self.trades_file}: {type(e).__name__}: {e}"
                    )
                    raise OSError(f"Trades save failed: {e}") from e

                # Save performance data with error handling
                try:
                    performance_data = {
                        date: asdict(metrics)
                        for date, metrics in self.daily_metrics.items()
                    }
                    performance_data = serialize_dict(performance_data)
                    performance_data["save_metadata"] = {
                        "save_time": datetime.now(UTC).isoformat(),
                        "metrics_count": len(self.daily_metrics),
                    }

                    # Atomic write for performance file
                    temp_perf_file = self.performance_file.with_suffix(".tmp")
                    with open(temp_perf_file, "w") as f:
                        json.dump(performance_data, f, indent=2)
                    temp_perf_file.rename(self.performance_file)
                    logger.info(
                        f"✅ Performance data saved: {len(self.daily_metrics)} daily metrics"
                    )

                except Exception as e:
                    logger.error(
                        f"❌ Failed to save performance data to {self.performance_file}: {type(e).__name__}: {e}"
                    )
                    raise OSError(f"Performance save failed: {e}") from e

                # Save session trades for dashboard access
                try:
                    self.save_session_trades()
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to save session trades (non-critical): {e}"
                    )
                    # Don't fail the entire save operation for this

                # Calculate and log save performance
                save_duration = (time.perf_counter() - save_start_time) * 1000
                logger.info(
                    f"💾 Paper trading state saved successfully in {save_duration:.1f}ms"
                )

                # Clean up any temporary files that might have been left behind
                for temp_file in self.data_dir.glob("*.tmp"):
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass  # Ignore cleanup errors

            except OSError as e:
                # IO-specific errors (permissions, disk space, etc.)
                logger.error(
                    f"🚨 CRITICAL: Paper trading state save failed due to IO error: {e}"
                )
                logger.error(f"🚨 Data directory: {self.data_dir}")
                logger.error(
                    f"🚨 Files: account={self.account_file.exists()}, trades={self.trades_file.exists()}, perf={self.performance_file.exists()}"
                )
                raise  # Re-raise IO errors as they're critical

            except Exception as e:
                # Catch-all for unexpected errors
                logger.error(
                    f"🚨 CRITICAL: Unexpected error during paper trading state save: {type(e).__name__}: {e}"
                )
                logger.error(
                    f"🚨 Account state: balance={self.current_balance}, trades={self.trade_counter}"
                )
                logger.error(
                    f"🚨 Data state: open_trades={len(self.open_trades)}, closed_trades={len(self.closed_trades)}"
                )

                # Log stack trace for debugging
                import traceback

                logger.error(f"🚨 Stack trace:\n{traceback.format_exc()}")

                raise  # Re-raise unexpected errors

    def save_session_trades(self) -> None:
        """Save session trades to session_trades.json file for dashboard access."""
        try:
            session_trades_file = self.data_dir / "session_trades.json"

            # Get all trades (both open and closed) for this session
            session_trades = []

            # Add open trades with error handling
            for trade in self.open_trades.values():
                try:
                    session_trades.append(
                        {
                            "id": trade.id,
                            "symbol": trade.symbol,
                            "side": trade.side,
                            "entry_time": trade.entry_time.isoformat(),
                            "exit_time": None,
                            "entry_price": float(trade.entry_price),
                            "exit_price": None,
                            "size": float(trade.size),
                            "realized_pnl": None,
                            "unrealized_pnl": float(
                                trade.calculate_unrealized_pnl(trade.entry_price)
                            ),  # Placeholder
                            "fees": float(trade.fees),
                            "status": trade.status,
                            "duration_hours": None,
                        }
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Error serializing open trade {trade.id}: {e}")
                    continue

            # Add closed trades with error handling
            for trade in self.closed_trades:
                try:
                    duration_hours = None
                    if trade.exit_time:
                        duration_hours = (
                            trade.exit_time - trade.entry_time
                        ).total_seconds() / 3600

                    session_trades.append(
                        {
                            "id": trade.id,
                            "symbol": trade.symbol,
                            "side": trade.side,
                            "entry_time": trade.entry_time.isoformat(),
                            "exit_time": (
                                trade.exit_time.isoformat() if trade.exit_time else None
                            ),
                            "entry_price": float(trade.entry_price),
                            "exit_price": (
                                float(trade.exit_price) if trade.exit_price else None
                            ),
                            "size": float(trade.size),
                            "realized_pnl": (
                                float(trade.realized_pnl)
                                if trade.realized_pnl
                                else None
                            ),
                            "unrealized_pnl": 0.0,  # Closed trades have no unrealized P&L
                            "fees": float(trade.fees),
                            "status": trade.status,
                            "duration_hours": duration_hours,
                        }
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Error serializing closed trade {trade.id}: {e}")
                    continue

            # Add metadata for debugging
            session_data = {
                "trades": session_trades,
                "metadata": {
                    "save_time": datetime.now(UTC).isoformat(),
                    "total_trades": len(session_trades),
                    "open_count": len(self.open_trades),
                    "closed_count": len(self.closed_trades),
                },
            }

            # Atomic write for session trades
            temp_session_file = session_trades_file.with_suffix(".tmp")
            with open(temp_session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            temp_session_file.rename(session_trades_file)

            logger.info(
                f"📊 Session trades saved: {len(session_trades)} trades to {session_trades_file}"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to save session trades to {session_trades_file}: {type(e).__name__}: {e}"
            )

    def _load_state(self) -> None:
        """Load state from files."""
        try:
            # Load account state
            if self.account_file.exists():
                with open(self.account_file) as f:
                    account_data = json.load(f)

                self.starting_balance = Decimal(
                    account_data.get("starting_balance", str(self.starting_balance))
                )
                self.current_balance = Decimal(
                    account_data.get("current_balance", str(self.current_balance))
                )
                self.equity = Decimal(account_data.get("equity", str(self.equity)))
                self.margin_used = Decimal(account_data.get("margin_used", "0"))
                self.trade_counter = account_data.get("trade_counter", 0)
                self.peak_equity = Decimal(
                    account_data.get("peak_equity", str(self.peak_equity))
                )
                self.max_drawdown = Decimal(account_data.get("max_drawdown", "0"))

                if "session_start_time" in account_data:
                    self.session_start_time = datetime.fromisoformat(
                        account_data["session_start_time"]
                    )

                logger.info(
                    f"Loaded paper trading account: ${self.current_balance:,.2f} balance"
                )

            # Load trades
            if self.trades_file.exists():
                with open(self.trades_file) as f:
                    trades_data = json.load(f)

                # Load open trades
                for trade_id, trade_data in trades_data.get("open_trades", {}).items():
                    trade = PaperTrade(
                        id=trade_data["id"],
                        symbol=trade_data["symbol"],
                        side=trade_data["side"],
                        entry_time=datetime.fromisoformat(trade_data["entry_time"]),
                        entry_price=Decimal(trade_data["entry_price"]),
                        size=Decimal(trade_data["size"]),
                        exit_time=(
                            datetime.fromisoformat(trade_data["exit_time"])
                            if trade_data.get("exit_time")
                            else None
                        ),
                        exit_price=(
                            Decimal(trade_data["exit_price"])
                            if trade_data.get("exit_price")
                            else None
                        ),
                        realized_pnl=(
                            Decimal(trade_data["realized_pnl"])
                            if trade_data.get("realized_pnl")
                            else None
                        ),
                        fees=Decimal(trade_data.get("fees", "0")),
                        slippage=Decimal(trade_data.get("slippage", "0")),
                        status=trade_data.get("status", "OPEN"),
                    )
                    self.open_trades[trade_id] = trade

                # Load closed trades
                for trade_data in trades_data.get("closed_trades", []):
                    trade = PaperTrade(
                        id=trade_data["id"],
                        symbol=trade_data["symbol"],
                        side=trade_data["side"],
                        entry_time=datetime.fromisoformat(trade_data["entry_time"]),
                        entry_price=Decimal(trade_data["entry_price"]),
                        size=Decimal(trade_data["size"]),
                        exit_time=(
                            datetime.fromisoformat(trade_data["exit_time"])
                            if trade_data.get("exit_time")
                            else None
                        ),
                        exit_price=(
                            Decimal(trade_data["exit_price"])
                            if trade_data.get("exit_price")
                            else None
                        ),
                        realized_pnl=(
                            Decimal(trade_data["realized_pnl"])
                            if trade_data.get("realized_pnl")
                            else None
                        ),
                        fees=Decimal(trade_data.get("fees", "0")),
                        slippage=Decimal(trade_data.get("slippage", "0")),
                        status=trade_data.get("status", "CLOSED"),
                    )
                    self.closed_trades.append(trade)

                logger.info(
                    f"Loaded {len(self.open_trades)} open and {len(self.closed_trades)} closed trades"
                )

            # Load performance data
            if self.performance_file.exists():
                with open(self.performance_file) as f:
                    performance_data = json.load(f)

                for date, metrics_data in performance_data.items():
                    # Skip metadata entries that don't contain daily performance data
                    if (
                        date == "save_metadata"
                        or not isinstance(metrics_data, dict)
                        or "date" not in metrics_data
                    ):
                        continue

                    metrics = DailyPerformance(
                        date=metrics_data["date"],
                        starting_balance=Decimal(metrics_data["starting_balance"]),
                        ending_balance=Decimal(metrics_data["ending_balance"]),
                        trades_opened=metrics_data["trades_opened"],
                        trades_closed=metrics_data["trades_closed"],
                        realized_pnl=Decimal(metrics_data["realized_pnl"]),
                        unrealized_pnl=Decimal(metrics_data["unrealized_pnl"]),
                        total_pnl=Decimal(metrics_data["total_pnl"]),
                        fees_paid=Decimal(metrics_data["fees_paid"]),
                        win_rate=metrics_data["win_rate"],
                        largest_win=Decimal(metrics_data["largest_win"]),
                        largest_loss=Decimal(metrics_data["largest_loss"]),
                        drawdown=Decimal(metrics_data["drawdown"]),
                    )
                    self.daily_metrics[date] = metrics

                logger.info(
                    f"Loaded performance data for {len(self.daily_metrics)} days"
                )

        except Exception as e:
            logger.error(f"Failed to load paper trading state: {e}")
            # Continue with default state

    def reset_account(self, new_balance: Decimal = None) -> None:
        """Reset the paper trading account."""
        with self._lock:
            self.starting_balance = new_balance or Decimal("10000")
            self.current_balance = self.starting_balance
            self.equity = self.starting_balance
            self.margin_used = Decimal("0")
            self.open_trades.clear()
            self.closed_trades.clear()
            self.daily_metrics.clear()
            self.trade_counter = 0
            self.session_start_time = datetime.now(UTC)
            self.peak_equity = self.starting_balance
            self.max_drawdown = Decimal("0")

            self._save_state()
            logger.info(f"Paper trading account reset to ${self.starting_balance:,.2f}")
