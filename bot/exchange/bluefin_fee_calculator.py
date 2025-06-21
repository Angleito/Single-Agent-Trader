"""
Bluefin exchange fee calculator for perpetual futures trading.

This module provides precise fee calculations for Bluefin DEX operations,
including maker/taker fees, round-trip costs, and minimum profitable spreads.
"""

import logging
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import NamedTuple

logger = logging.getLogger(__name__)


class BluefinFees(NamedTuple):
    """Calculated Bluefin trading fees for a trade."""

    maker_fee: Decimal
    taker_fee: Decimal
    round_trip_cost: Decimal
    notional_value: Decimal
    fee_percentage: Decimal


class BluefinFeeCalculator:
    """
    Calculate trading fees for Bluefin DEX perpetual futures.

    Bluefin DEX fee structure:
    - Maker Fee: 0.010% (0.0001 as decimal)
    - Taker Fee: 0.035% (0.00035 as decimal)

    All fees are calculated on notional value (position_size x price).
    """

    # Bluefin DEX fee rates (as decimals)
    MAKER_FEE_RATE = Decimal("0.0001")  # 0.010%
    TAKER_FEE_RATE = Decimal("0.00035")  # 0.035%

    # Safety margin for minimum profitable spread calculations
    SAFETY_MARGIN = Decimal("0.0002")  # 0.02%

    def __init__(self):
        """Initialize the Bluefin fee calculator."""
        # Ensure fee rates are Decimal objects
        self.maker_fee_rate = self._ensure_decimal(
            self.MAKER_FEE_RATE, Decimal("0.0001")
        )
        self.taker_fee_rate = self._ensure_decimal(
            self.TAKER_FEE_RATE, Decimal("0.00035")
        )

        # Convert to percentages for logging
        try:
            maker_rate_pct = float(self.maker_fee_rate * 100)
            taker_rate_pct = float(self.taker_fee_rate * 100)
        except (TypeError, ValueError) as e:
            # Handle edge cases where rates might be strings or None
            logger.warning("Fee rate conversion error: %s. Using defaults.", e)
            maker_rate_pct = 0.01  # Default 0.01%
            taker_rate_pct = 0.035  # Default 0.035%

        logger.info(
            "Initialized BluefinFeeCalculator - Maker: %.4f%%, Taker: %.4f%%",
            maker_rate_pct,
            taker_rate_pct,
        )

    def _ensure_decimal(self, value, default: Decimal) -> Decimal:
        """Ensure a value is a Decimal object."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (TypeError, ValueError, InvalidOperation):
            logger.warning(
                "Invalid fee rate value: %s (type: %s). Using default: %s",
                value,
                type(value),
                default,
            )
            return default

    def _normalize_decimal(self, value: Decimal, decimal_places: int = 8) -> Decimal:
        """
        Normalize a decimal value to the specified number of decimal places.

        Args:
            value: Decimal value to normalize
            decimal_places: Number of decimal places to round to

        Returns:
            Normalized decimal value

        Raises:
            ValueError: If value is invalid (NaN, infinite)
        """
        if value is None:
            return Decimal(0)

        if not isinstance(value, Decimal):
            try:
                value = Decimal(str(value))
            except (ValueError, TypeError) as e:
                logger.exception(
                    "Invalid decimal value (value: %s, type: %s)", value, type(value)
                )
                raise ValueError(f"Cannot convert to Decimal: {value}") from e

        if value.is_nan():
            logger.error("Value is NaN")
            raise ValueError("Value cannot be NaN")

        if value.is_infinite():
            logger.error("Value is infinite")
            raise ValueError("Value cannot be infinite")

        # Quantize to specified decimal places using banker's rounding
        quantize_exp = Decimal("0.1") ** decimal_places
        return value.quantize(quantize_exp, rounding=ROUND_HALF_EVEN)

    def calculate_maker_fee(self, notional_value: Decimal) -> Decimal:
        """
        Calculate maker fee for a limit order.

        Args:
            notional_value: The notional value of the trade (position_size x price)

        Returns:
            Maker fee in USDC

        Raises:
            ValueError: If notional_value is invalid
        """
        try:
            notional_value = self._normalize_decimal(notional_value)

            if notional_value <= 0:
                logger.debug("Notional value must be positive for fee calculation")
                return Decimal(0)

            maker_fee = notional_value * self.maker_fee_rate
            maker_fee = self._normalize_decimal(
                maker_fee, 6
            )  # USDC has 6 decimal places

            logger.debug(
                "Calculated maker fee: $%.6f (%.4f%% of $%.2f)",
                maker_fee,
                float(self.maker_fee_rate * 100),
                notional_value,
            )

            return maker_fee

        except Exception as e:
            logger.exception(
                "Error calculating maker fee for notional value: %s", notional_value
            )
            raise ValueError(f"Failed to calculate maker fee: {e}") from e

    def calculate_taker_fee(self, notional_value: Decimal) -> Decimal:
        """
        Calculate taker fee for a market order.

        Args:
            notional_value: The notional value of the trade (position_size x price)

        Returns:
            Taker fee in USDC

        Raises:
            ValueError: If notional_value is invalid
        """
        try:
            notional_value = self._normalize_decimal(notional_value)

            if notional_value <= 0:
                logger.debug("Notional value must be positive for fee calculation")
                return Decimal(0)

            taker_fee = notional_value * self.taker_fee_rate
            taker_fee = self._normalize_decimal(
                taker_fee, 6
            )  # USDC has 6 decimal places

            logger.debug(
                "Calculated taker fee: $%.6f (%.4f%% of $%.2f)",
                taker_fee,
                float(self.taker_fee_rate * 100),
                notional_value,
            )

            return taker_fee

        except Exception as e:
            logger.exception(
                "Error calculating taker fee for notional value: %s", notional_value
            )
            raise ValueError(f"Failed to calculate taker fee: {e}") from e

    def calculate_round_trip_cost(
        self, notional_value: Decimal, use_limit_orders: bool = True
    ) -> Decimal:
        """
        Calculate the total cost of a round-trip trade (entry + exit).

        Args:
            notional_value: The notional value of the trade (position_size x price)
            use_limit_orders: Whether to use limit orders (maker fees) or market orders (taker fees)

        Returns:
            Total round-trip cost in USDC

        Raises:
            ValueError: If notional_value is invalid
        """
        try:
            notional_value = self._normalize_decimal(notional_value)

            if notional_value <= 0:
                logger.debug(
                    "Notional value must be positive for round-trip calculation"
                )
                return Decimal(0)

            if use_limit_orders:
                # Use maker fees for both entry and exit (assuming limit orders)
                entry_fee = self.calculate_maker_fee(notional_value)
                exit_fee = self.calculate_maker_fee(notional_value)
                round_trip_cost = entry_fee + exit_fee
                fee_type = "maker"
            else:
                # Use taker fees for both entry and exit (market orders)
                entry_fee = self.calculate_taker_fee(notional_value)
                exit_fee = self.calculate_taker_fee(notional_value)
                round_trip_cost = entry_fee + exit_fee
                fee_type = "taker"

            round_trip_cost = self._normalize_decimal(round_trip_cost, 6)

            logger.debug(
                "Calculated round-trip cost (%s orders): $%.6f for $%.2f notional",
                fee_type,
                round_trip_cost,
                notional_value,
            )

            return round_trip_cost

        except Exception as e:
            logger.exception(
                "Error calculating round-trip cost for notional value: %s",
                notional_value,
            )
            raise ValueError(f"Failed to calculate round-trip cost: {e}") from e

    def get_minimum_profitable_spread(self, use_limit_orders: bool = True) -> Decimal:
        """
        Calculate the minimum spread needed to be profitable after fees.

        Args:
            use_limit_orders: Whether to use limit orders (maker fees) or market orders (taker fees)

        Returns:
            Minimum profitable spread as a percentage (e.g., 0.0004 = 0.04%)
        """
        try:
            if use_limit_orders:
                # For limit orders: 2 x maker fee + safety margin
                base_cost = self.maker_fee_rate * 2
                fee_type = "maker"
            else:
                # For market orders: 2 x taker fee + safety margin
                base_cost = self.taker_fee_rate * 2
                fee_type = "taker"

            minimum_spread = base_cost + self.SAFETY_MARGIN
            minimum_spread = self._normalize_decimal(minimum_spread, 8)

            logger.debug(
                "Minimum profitable spread (%s orders): %.4f%% (%.6f as decimal)",
                fee_type,
                float(minimum_spread * 100),
                minimum_spread,
            )

            return minimum_spread

        except Exception as e:
            logger.exception("Error calculating minimum profitable spread")
            raise ValueError(
                f"Failed to calculate minimum profitable spread: {e}"
            ) from e

    def calculate_fee_breakdown(
        self, notional_value: Decimal, use_limit_orders: bool = True
    ) -> BluefinFees:
        """
        Calculate a complete fee breakdown for a trade.

        Args:
            notional_value: The notional value of the trade (position_size x price)
            use_limit_orders: Whether to use limit orders (maker fees) or market orders (taker fees)

        Returns:
            BluefinFees object with complete fee breakdown

        Raises:
            ValueError: If notional_value is invalid
        """
        try:
            notional_value = self._normalize_decimal(notional_value)

            if notional_value <= 0:
                logger.debug("Notional value must be positive for fee breakdown")
                return BluefinFees(
                    maker_fee=Decimal(0),
                    taker_fee=Decimal(0),
                    round_trip_cost=Decimal(0),
                    notional_value=Decimal(0),
                    fee_percentage=Decimal(0),
                )

            # Calculate individual fees
            maker_fee = self.calculate_maker_fee(notional_value)
            taker_fee = self.calculate_taker_fee(notional_value)

            # Calculate round-trip cost based on order type preference
            round_trip_cost = self.calculate_round_trip_cost(
                notional_value, use_limit_orders
            )

            # Calculate fee percentage of notional value
            fee_percentage = self._normalize_decimal(
                round_trip_cost / notional_value, 8
            )

            fees = BluefinFees(
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                round_trip_cost=round_trip_cost,
                notional_value=notional_value,
                fee_percentage=fee_percentage,
            )

            logger.debug(
                "Complete fee breakdown for $%.2f notional: Maker: $%.6f, Taker: $%.6f, "
                "Round-trip: $%.6f (%.4f%%)",
                notional_value,
                maker_fee,
                taker_fee,
                round_trip_cost,
                float(fee_percentage * 100),
            )

            return fees

        except Exception as e:
            logger.exception("Error calculating fee breakdown")
            raise ValueError(f"Failed to calculate fee breakdown: {e}") from e

    def get_fee_rates(self) -> dict[str, Decimal]:
        """
        Get current fee rates as decimals.

        Returns:
            Dictionary with maker and taker fee rates
        """
        return {
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "maker_fee_percentage": self.maker_fee_rate * 100,
            "taker_fee_percentage": self.taker_fee_rate * 100,
        }

    def get_fee_summary(self) -> dict[str, float | str]:
        """
        Get a human-readable summary of fee information.

        Returns:
            Dictionary with fee information formatted for display
        """
        min_spread_maker = self.get_minimum_profitable_spread(use_limit_orders=True)
        min_spread_taker = self.get_minimum_profitable_spread(use_limit_orders=False)

        return {
            "exchange": "Bluefin DEX",
            "maker_fee_rate": float(self.maker_fee_rate),
            "taker_fee_rate": float(self.taker_fee_rate),
            "maker_fee_percentage": float(self.maker_fee_rate * 100),
            "taker_fee_percentage": float(self.taker_fee_rate * 100),
            "min_profitable_spread_maker_pct": float(min_spread_maker * 100),
            "min_profitable_spread_taker_pct": float(min_spread_taker * 100),
            "currency": "USDC",
            "calculation_basis": "notional_value",
        }

    def estimate_trading_costs(
        self,
        position_size: Decimal,
        entry_price: Decimal,
        exit_price: Decimal | None = None,
        use_limit_orders: bool = True,
    ) -> dict[str, Decimal | str]:
        """
        Estimate total trading costs for a complete trade scenario.

        Args:
            position_size: Size of the position (in base asset units)
            entry_price: Entry price per unit
            exit_price: Exit price per unit (if None, assumes same as entry)
            use_limit_orders: Whether to use limit orders (maker fees) or market orders (taker fees)

        Returns:
            Dictionary with detailed cost breakdown

        Raises:
            ValueError: If any input values are invalid
        """
        try:
            position_size = self._normalize_decimal(position_size)
            entry_price = self._normalize_decimal(entry_price)
            exit_price = self._normalize_decimal(exit_price or entry_price)

            # Calculate notional values
            entry_notional = position_size * entry_price
            exit_notional = position_size * exit_price

            # Calculate fees for entry and exit
            if use_limit_orders:
                entry_fee = self.calculate_maker_fee(entry_notional)
                exit_fee = self.calculate_maker_fee(exit_notional)
            else:
                entry_fee = self.calculate_taker_fee(entry_notional)
                exit_fee = self.calculate_taker_fee(exit_notional)

            total_fees = entry_fee + exit_fee

            # Calculate cost as percentage of entry notional
            fee_percentage = (
                self._normalize_decimal(total_fees / entry_notional, 8)
                if entry_notional > 0
                else Decimal(0)
            )

            cost_breakdown = {
                "position_size": position_size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_notional": entry_notional,
                "exit_notional": exit_notional,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "total_fees": total_fees,
                "fee_percentage": fee_percentage,
                "order_type": "limit" if use_limit_orders else "market",
            }

            logger.debug(
                "Trading cost estimate: %s %s @ $%.2f -> $%.2f, Total fees: $%.6f (%.4f%%)",
                position_size,
                "limit" if use_limit_orders else "market",
                entry_price,
                exit_price,
                total_fees,
                float(fee_percentage * 100),
            )

            return cost_breakdown

        except Exception as e:
            logger.exception("Error estimating trading costs")
            raise ValueError(f"Failed to estimate trading costs: {e}") from e


# Global Bluefin fee calculator instance
bluefin_fee_calculator = BluefinFeeCalculator()
