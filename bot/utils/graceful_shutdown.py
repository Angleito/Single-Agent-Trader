#!/usr/bin/env python3
"""
Graceful shutdown utility for AI Trading Bot.

This module provides functionality to safely shutdown the trading bot
by closing open positions and saving state before termination.
Used during zero-downtime deployments to ensure clean transitions.
"""

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.config import get_config
from bot.exchange.factory import ExchangeFactory

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handles graceful shutdown of the trading bot."""

    def __init__(self):
        """Initialize graceful shutdown handler."""
        self.config = get_config()
        self.exchange = None
        self.state_file = Path("data/shutdown_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    async def initialize_exchange(self):
        """Initialize exchange connection."""
        try:
            exchange_factory = ExchangeFactory(self.config)
            self.exchange = await exchange_factory.create()
            logger.info(f"Connected to {self.config.exchange.exchange_type} exchange")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    async def get_open_positions(self) -> list:
        """Get all open positions."""
        try:
            positions = await self.exchange.get_open_positions()
            logger.info(f"Found {len(positions)} open positions")
            return positions
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    async def close_position(self, position: dict[str, Any]) -> bool:
        """Close a single position."""
        try:
            symbol = position.get("symbol", position.get("product_id"))
            size = position.get("size", position.get("quantity", 0))
            side = position.get("side", "").upper()

            if not symbol or not size:
                logger.warning(f"Invalid position data: {position}")
                return False

            # Determine close side
            close_side = "SELL" if side == "BUY" else "BUY"

            logger.info(f"Closing position: {symbol} {side} {size}")

            # Place market order to close position
            order = await self.exchange.place_order(
                symbol=symbol,
                side=close_side,
                size=abs(float(size)),
                order_type="MARKET",
                reduce_only=True,  # Ensure we're only closing
            )

            logger.info(f"Position closed successfully: {order}")
            return True

        except Exception as e:
            logger.error(f"Failed to close position {position}: {e}")
            return False

    async def save_state(self, positions: list):
        """Save current state for recovery."""
        try:
            state = {
                "timestamp": datetime.now(UTC).isoformat(),
                "positions": positions,
                "config": {
                    "symbol": self.config.trading.symbol,
                    "interval": self.config.trading.interval,
                    "leverage": self.config.trading.leverage,
                    "dry_run": self.config.system.dry_run,
                },
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"State saved to {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def cancel_open_orders(self) -> int:
        """Cancel all open orders."""
        try:
            orders = await self.exchange.get_open_orders()
            cancelled = 0

            for order in orders:
                try:
                    order_id = order.get("id", order.get("order_id"))
                    if order_id:
                        await self.exchange.cancel_order(order_id)
                        cancelled += 1
                        logger.info(f"Cancelled order: {order_id}")
                except Exception as e:
                    logger.error(f"Failed to cancel order {order}: {e}")

            logger.info(f"Cancelled {cancelled} open orders")
            return cancelled

        except Exception as e:
            logger.error(f"Failed to get/cancel open orders: {e}")
            return 0

    async def shutdown(self) -> bool:
        """Execute graceful shutdown sequence."""
        logger.info("Starting graceful shutdown sequence...")

        try:
            # Initialize exchange if not already done
            if not self.exchange:
                await self.initialize_exchange()

            # Cancel all open orders first
            await self.cancel_open_orders()

            # Get open positions
            positions = await self.get_open_positions()

            # Save state before closing positions
            if positions:
                await self.save_state(positions)

            # Close all positions
            closed = 0
            failed = 0

            for position in positions:
                if await self.close_position(position):
                    closed += 1
                else:
                    failed += 1

                # Small delay between closures
                await asyncio.sleep(0.5)

            logger.info(
                f"Shutdown complete: {closed} positions closed, {failed} failed"
            )

            # Final state save
            await self.save_state([])

            return failed == 0

        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
            return False
        finally:
            # Cleanup
            if self.exchange:
                await self.exchange.close()


async def main():
    """Main entry point for graceful shutdown."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create shutdown handler
    shutdown_handler = GracefulShutdown()

    # Execute shutdown
    success = await shutdown_handler.shutdown()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Run the shutdown sequence
    asyncio.run(main())
