"""
Functional adapter for PositionManager to enable gradual migration to functional portfolio types.

This adapter bridges the gap between the existing imperative PositionManager and the new
functional portfolio types, allowing for seamless migration without breaking existing APIs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from bot.fp.strategies.position_calculations import (
    PositionPerformance,
    calculate_portfolio_pnl,
    calculate_position_performance,
)
from bot.fp.types.portfolio import (
    AccountSnapshot,
    AccountType,
    PerformanceSnapshot,
    create_futures_account,
    create_spot_account,
)
from bot.fp.types.positions import (
    FunctionalPosition,
    PositionSnapshot,
    create_empty_position,
    create_position_from_lot,
)
from bot.fp.types.result import Failure, Result, Success

if TYPE_CHECKING:
    from bot.position_manager import PositionManager
    from bot.trading_types import Order
    from bot.trading_types import Position as LegacyPosition

logger = logging.getLogger(__name__)


class FunctionalPositionManagerAdapter:
    """
    Functional adapter for PositionManager that provides functional portfolio operations.

    This adapter converts between legacy Position objects and functional types,
    enabling gradual migration to functional programming patterns.
    """

    def __init__(self, position_manager: PositionManager) -> None:
        """
        Initialize the adapter with an existing position manager.

        Args:
            position_manager: The legacy position manager to adapt
        """
        self.position_manager = position_manager
        self._last_snapshot: PositionSnapshot | None = None
        self._last_account_snapshot: AccountSnapshot | None = None

    def get_functional_position(self, symbol: str) -> FunctionalPosition:
        """
        Get a functional position for the given symbol.

        Args:
            symbol: Trading symbol

        Returns:
            FunctionalPosition object
        """
        legacy_position = self.position_manager.get_position(symbol)
        return self._convert_to_functional_position(legacy_position)

    def get_all_functional_positions(self) -> list[FunctionalPosition]:
        """
        Get all active positions as functional types.

        Returns:
            List of functional positions
        """
        legacy_positions = self.position_manager.get_all_positions()
        return [self._convert_to_functional_position(pos) for pos in legacy_positions]

    def get_position_snapshot(self) -> PositionSnapshot:
        """
        Get current position snapshot in functional form.

        Returns:
            PositionSnapshot with all current positions
        """
        functional_positions = self.get_all_functional_positions()

        # Calculate totals
        total_unrealized = sum(pos.unrealized_pnl for pos in functional_positions)
        total_realized = sum(pos.total_realized_pnl for pos in functional_positions)

        snapshot = PositionSnapshot(
            timestamp=datetime.now(),
            positions=tuple(functional_positions),
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
        )

        self._last_snapshot = snapshot
        return snapshot

    def get_account_snapshot(
        self,
        current_prices: dict[str, Decimal],
        account_type: AccountType = AccountType.SPOT,
        base_currency: str = "USD",
    ) -> Result[str, AccountSnapshot]:
        """
        Get account snapshot with balance information.

        Args:
            current_prices: Current market prices for valuation
            account_type: Type of account (SPOT/FUTURES)
            base_currency: Base currency for calculations

        Returns:
            Result containing AccountSnapshot or error
        """
        try:
            # Get total P&L from position manager
            realized_pnl, unrealized_pnl = self.position_manager.calculate_total_pnl()

            # For this adapter, we'll simulate account balances
            # In a real implementation, this would come from the exchange
            total_equity = (
                realized_pnl + unrealized_pnl + Decimal(10000)
            )  # Simulated starting balance

            if account_type == AccountType.SPOT:
                # Create spot account snapshot
                balances = {base_currency: max(Decimal(0), total_equity)}
                account_snapshot = create_spot_account(balances, base_currency)
            else:
                # Create futures account snapshot
                balances = {base_currency: max(Decimal(0), total_equity)}
                margin_used = sum(
                    self._estimate_margin_used(
                        pos, current_prices.get(pos.symbol, Decimal(0))
                    )
                    for pos in self.get_all_functional_positions()
                    if not pos.is_flat
                )
                leverage = Decimal(5)  # Default leverage

                account_snapshot = create_futures_account(
                    balances, margin_used, leverage, base_currency
                )

            self._last_account_snapshot = account_snapshot
            return Success(account_snapshot)

        except Exception as e:
            return Failure(f"Failed to create account snapshot: {e!s}")

    def get_portfolio_performance(
        self, current_prices: dict[str, Decimal], account_balance: Decimal
    ) -> Result[str, list[PositionPerformance]]:
        """
        Get performance analysis for all positions.

        Args:
            current_prices: Current market prices
            account_balance: Total account balance

        Returns:
            Result containing list of position performances
        """
        try:
            functional_positions = self.get_all_functional_positions()
            performances = []

            for position in functional_positions:
                if position.is_flat:
                    continue

                current_price = current_prices.get(position.symbol)
                if current_price is None:
                    logger.warning(f"No current price for {position.symbol}")
                    continue

                performance = calculate_position_performance(
                    position, current_price, account_balance
                )
                performances.append(performance)

            return Success(performances)

        except Exception as e:
            return Failure(f"Failed to calculate portfolio performance: {e!s}")

    def update_position_from_order_functional(
        self, order: Order, fill_price: Decimal
    ) -> Result[str, FunctionalPosition]:
        """
        Update position using functional types and return the result.

        Args:
            order: Order that was filled
            fill_price: Actual fill price

        Returns:
            Result containing updated functional position
        """
        try:
            # Update the legacy position manager first
            legacy_position = self.position_manager.update_position_from_order(
                order, fill_price
            )

            # Convert to functional position
            functional_position = self._convert_to_functional_position(legacy_position)

            return Success(functional_position)

        except Exception as e:
            return Failure(f"Failed to update position from order: {e!s}")

    def calculate_portfolio_metrics(
        self, current_prices: dict[str, Decimal]
    ) -> Result[str, dict[str, Decimal]]:
        """
        Calculate comprehensive portfolio metrics using functional types.

        Args:
            current_prices: Current market prices

        Returns:
            Result containing portfolio metrics
        """
        try:
            snapshot = self.get_position_snapshot()

            # Calculate P&L components
            pnl_result = calculate_portfolio_pnl(snapshot, current_prices)
            if pnl_result.is_failure():
                return Failure(pnl_result.failure())

            pnl_components = pnl_result.success()

            # Calculate additional metrics
            total_value = sum(
                pos.position_value
                for pos in snapshot.positions
                if current_prices.get(pos.symbol) is not None
            )

            metrics = {
                "total_pnl": pnl_components.total_pnl,
                "realized_pnl": pnl_components.realized_pnl,
                "unrealized_pnl": pnl_components.unrealized_pnl,
                "total_value": total_value,
                "win_rate": Decimal(str(pnl_components.win_rate)),
                "total_trades": Decimal(str(pnl_components.realized_trades)),
                "position_count": Decimal(
                    str(len([p for p in snapshot.positions if not p.is_flat]))
                ),
            }

            return Success(metrics)

        except Exception as e:
            return Failure(f"Failed to calculate portfolio metrics: {e!s}")

    def validate_portfolio_consistency(
        self, current_prices: dict[str, Decimal]
    ) -> Result[str, dict[str, bool]]:
        """
        Validate consistency between legacy and functional representations.

        Args:
            current_prices: Current market prices for validation

        Returns:
            Result containing validation results
        """
        try:
            validation_results = {}

            # Compare total P&L calculations
            legacy_realized, legacy_unrealized = (
                self.position_manager.calculate_total_pnl()
            )

            snapshot = self.get_position_snapshot()
            functional_realized = snapshot.total_realized_pnl
            functional_unrealized = snapshot.total_unrealized_pnl

            # Check if P&L calculations match (within tolerance)
            tolerance = Decimal("0.01")
            realized_match = abs(legacy_realized - functional_realized) <= tolerance
            unrealized_match = (
                abs(legacy_unrealized - functional_unrealized) <= tolerance
            )

            validation_results["realized_pnl_match"] = realized_match
            validation_results["unrealized_pnl_match"] = unrealized_match

            # Check position count consistency
            legacy_positions = self.position_manager.get_all_positions()
            functional_positions = self.get_all_functional_positions()

            active_legacy = len([p for p in legacy_positions if p.side != "FLAT"])
            active_functional = len([p for p in functional_positions if not p.is_flat])

            validation_results["position_count_match"] = (
                active_legacy == active_functional
            )

            # Overall consistency
            validation_results["overall_consistent"] = all(validation_results.values())

            if not validation_results["overall_consistent"]:
                logger.warning(
                    "Portfolio consistency check failed: %s",
                    {k: v for k, v in validation_results.items() if not v},
                )

            return Success(validation_results)

        except Exception as e:
            return Failure(f"Failed to validate portfolio consistency: {e!s}")

    def _convert_to_functional_position(
        self, legacy_position: LegacyPosition
    ) -> FunctionalPosition:
        """
        Convert legacy Position to FunctionalPosition.

        Args:
            legacy_position: Legacy position object

        Returns:
            FunctionalPosition equivalent
        """
        if legacy_position.side == "FLAT":
            return create_empty_position(
                legacy_position.symbol, legacy_position.timestamp
            )

        # Convert side to functional type
        if legacy_position.side in {"LONG", "SHORT"}:
            pass
        else:
            pass

        # Create functional position with a single lot
        if legacy_position.entry_price is not None and legacy_position.size > 0:
            result = create_position_from_lot(
                symbol=legacy_position.symbol,
                quantity=legacy_position.size,
                price=legacy_position.entry_price,
                timestamp=legacy_position.timestamp,
            )

            if result.is_success():
                position = result.success()
                # Update unrealized P&L to match legacy calculation
                return position.update_unrealized_pnl(
                    legacy_position.entry_price
                    + (
                        legacy_position.unrealized_pnl / legacy_position.size
                        if legacy_position.size > 0
                        else Decimal(0)
                    )
                )

        # Fallback to empty position
        return create_empty_position(legacy_position.symbol, legacy_position.timestamp)

    def _estimate_margin_used(
        self, position: FunctionalPosition, current_price: Decimal
    ) -> Decimal:
        """
        Estimate margin used for a position (simplified calculation).

        Args:
            position: Functional position
            current_price: Current market price

        Returns:
            Estimated margin used
        """
        if position.is_flat or current_price == Decimal(0):
            return Decimal(0)

        # Simplified margin calculation (position value / default leverage)
        position_value = position.total_quantity * current_price
        default_leverage = Decimal(5)
        return position_value / default_leverage

    def get_functional_summary(self) -> dict[str, any]:
        """
        Get comprehensive summary using functional types.

        Returns:
            Dictionary with functional portfolio summary
        """
        snapshot = self.get_position_snapshot()

        return {
            "timestamp": snapshot.timestamp,
            "active_positions": snapshot.position_count,
            "total_realized_pnl": float(snapshot.total_realized_pnl),
            "total_unrealized_pnl": float(snapshot.total_unrealized_pnl),
            "total_pnl": float(snapshot.total_pnl),
            "symbols": list(snapshot.symbols),
            "functional_types_enabled": True,
            "adapter_version": "1.0.0",
        }


# Utility functions for migration support


def migrate_legacy_positions_to_functional(
    legacy_positions: list[LegacyPosition],
) -> list[FunctionalPosition]:
    """
    Convert a list of legacy positions to functional positions.

    Args:
        legacy_positions: List of legacy Position objects

    Returns:
        List of FunctionalPosition objects
    """
    functional_positions = []

    for legacy_pos in legacy_positions:
        # Create temporary adapter to use conversion logic
        adapter = FunctionalPositionManagerAdapter(None)  # type: ignore
        functional_pos = adapter._convert_to_functional_position(legacy_pos)
        functional_positions.append(functional_pos)

    return functional_positions


def create_performance_snapshot(
    portfolio_value: Decimal,
    realized_pnl: Decimal,
    unrealized_pnl: Decimal,
    daily_return: Decimal | None = None,
    benchmark_return: Decimal | None = None,
    drawdown: Decimal = Decimal(0),
) -> PerformanceSnapshot:
    """
    Create a performance snapshot for portfolio analysis.

    Args:
        portfolio_value: Total portfolio value
        realized_pnl: Realized P&L
        unrealized_pnl: Unrealized P&L
        daily_return: Daily return percentage
        benchmark_return: Benchmark return for comparison
        drawdown: Current drawdown percentage

    Returns:
        PerformanceSnapshot object
    """
    return PerformanceSnapshot(
        timestamp=datetime.now(),
        total_value=portfolio_value,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        daily_return=daily_return,
        benchmark_return=benchmark_return,
        drawdown=drawdown,
    )


def validate_functional_migration(
    adapter: FunctionalPositionManagerAdapter, current_prices: dict[str, Decimal]
) -> bool:
    """
    Validate that functional migration is working correctly.

    Args:
        adapter: The functional adapter to validate
        current_prices: Current market prices

    Returns:
        True if migration is valid, False otherwise
    """
    try:
        # Test basic functionality
        snapshot = adapter.get_position_snapshot()
        if snapshot is None:
            return False

        # Test portfolio calculations
        metrics_result = adapter.calculate_portfolio_metrics(current_prices)
        if metrics_result.is_failure():
            return False

        # Test consistency validation
        validation_result = adapter.validate_portfolio_consistency(current_prices)
        if validation_result.is_failure():
            return False

        consistency = validation_result.success()
        return consistency.get("overall_consistent", False)

    except Exception as e:
        logger.exception(f"Functional migration validation failed: {e}")
        return False
