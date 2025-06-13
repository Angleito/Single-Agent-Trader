"""
Example integration showing how to use the position and order management system.

This module demonstrates how to integrate the PositionManager and OrderManager
with the existing trading bot components.
"""

import asyncio
import logging
from decimal import Decimal
from pathlib import Path

from .exchange.coinbase import CoinbaseClient
from .order_manager import OrderManager
from .position_manager import PositionManager
from .risk import RiskManager
from .types import TradeAction

logger = logging.getLogger(__name__)


class TradingBotExample:
    """
    Example trading bot integration with position and order management.

    Demonstrates how to wire together all the components for a complete
    trading system with proper position tracking and risk management.
    """

    def __init__(self, data_dir: Path = None):
        """
        Initialize the trading bot with integrated components.

        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir or Path("data")

        # Initialize managers
        self.position_manager = PositionManager(self.data_dir / "positions")
        self.order_manager = OrderManager(self.data_dir / "orders")

        # Initialize risk manager with position manager
        self.risk_manager = RiskManager(position_manager=self.position_manager)

        # Initialize exchange client with managers
        self.exchange = CoinbaseClient(
            order_manager=self.order_manager, position_manager=self.position_manager
        )

        logger.info(
            "TradingBotExample initialized with integrated position and order management"
        )

    async def start(self) -> None:
        """Start the trading bot."""
        # Start the order manager
        await self.order_manager.start()

        # Connect to exchange
        await self.exchange.connect()

        logger.info("Trading bot started successfully")

    async def stop(self) -> None:
        """Stop the trading bot."""
        # Stop the order manager
        await self.order_manager.stop()

        # Disconnect from exchange
        await self.exchange.disconnect()

        logger.info("Trading bot stopped")

    async def execute_trade(
        self, symbol: str, trade_action: TradeAction, current_price: Decimal
    ) -> bool:
        """
        Execute a trade with full risk management and position tracking.

        Args:
            symbol: Trading symbol
            trade_action: Trade action to execute
            current_price: Current market price

        Returns:
            True if trade executed successfully
        """
        try:
            # Get current position
            current_position = self.position_manager.get_position(symbol)

            # Evaluate risk
            approved, modified_action, reason = self.risk_manager.evaluate_risk(
                trade_action, current_position, current_price
            )

            if not approved:
                logger.warning(f"Trade rejected by risk manager: {reason}")
                return False

            # Execute trade through exchange
            order = await self.exchange.execute_trade_action(
                modified_action, symbol, current_price
            )

            if order:
                logger.info(
                    f"Trade executed: {order.id} - {order.side} {order.quantity} {symbol}"
                )

                # Register callback for order updates
                self.order_manager.register_callback(order.id, self._on_order_update)

                return True
            else:
                logger.error("Failed to execute trade")
                return False

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def _on_order_update(self, order_id: str, event) -> None:
        """
        Handle order updates.

        Args:
            order_id: Order ID
            event: Order event
        """
        logger.info(f"Order {order_id} event: {event}")

        # Get updated order
        order = self.order_manager.get_order(order_id)
        if order:
            logger.info(
                f"Order status: {order.status}, Filled: {order.filled_quantity}"
            )

    async def update_positions(self, symbol: str, current_price: Decimal) -> None:
        """
        Update position P&L and check risk levels.

        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        # Update unrealized P&L
        unrealized_pnl = self.position_manager.update_unrealized_pnl(
            symbol, current_price
        )

        # Check if position should be closed due to risk
        should_close, reason = self.position_manager.should_close_position(
            symbol, current_price
        )

        if should_close:
            logger.warning(f"Position should be closed: {reason}")

            # Create close action
            close_action = TradeAction(
                action="CLOSE",
                size_pct=0,
                take_profit_pct=2.0,
                stop_loss_pct=1.5,
                rationale=f"Risk management: {reason}",
            )

            # Execute close trade
            await self.execute_trade(symbol, close_action, current_price)

    def get_portfolio_summary(self) -> dict:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        # Position summary
        position_summary = self.position_manager.get_position_summary()

        # Risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()

        # Order statistics
        order_stats = self.order_manager.get_order_statistics()

        return {
            "positions": position_summary,
            "risk": {
                "account_balance": float(risk_metrics.account_balance),
                "daily_pnl": float(risk_metrics.daily_pnl),
                "available_margin": float(risk_metrics.available_margin),
                "current_positions": risk_metrics.current_positions,
                "max_daily_loss_reached": risk_metrics.max_daily_loss_reached,
            },
            "orders": order_stats,
            "timestamp": position_summary.get("timestamp", "N/A"),
        }

    async def cleanup_old_data(self) -> None:
        """Clean up old position and order history."""
        # Clean old position history (keep 30 days)
        self.position_manager.clear_old_history(days_to_keep=30)

        # Clean old order history (keep 7 days)
        self.order_manager.clear_old_history(days_to_keep=7)

        logger.info("Old data cleanup completed")


async def main():
    """Example usage of the integrated trading bot."""
    bot = TradingBotExample()

    try:
        # Start the bot
        await bot.start()

        # Example trade
        trade_action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=3.0,
            stop_loss_pct=2.0,
            rationale="Example long position",
        )

        current_price = Decimal("50000.00")
        symbol = "BTC-USD"

        # Execute trade
        success = await bot.execute_trade(symbol, trade_action, current_price)

        if success:
            logger.info("Trade executed successfully")

            # Simulate price update
            new_price = Decimal("51000.00")
            await bot.update_positions(symbol, new_price)

            # Get portfolio summary
            summary = bot.get_portfolio_summary()
            logger.info(f"Portfolio summary: {summary}")

        # Clean up old data
        await bot.cleanup_old_data()

    finally:
        # Stop the bot
        await bot.stop()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the example
    asyncio.run(main())
