"""
Fee calculation utilities for trading operations.

This module provides functions to calculate trading fees before position sizing
and to adjust position sizes to account for fees.
"""

import logging
from decimal import Decimal
from typing import NamedTuple

from .config import settings
from .trading_types import TradeAction

logger = logging.getLogger(__name__)


class TradeFees(NamedTuple):
    """Calculated trading fees for a trade."""

    entry_fee: Decimal
    exit_fee: Decimal
    total_fee: Decimal
    fee_rate: float
    net_position_value: Decimal


class FeeCalculator:
    """
    Calculate trading fees and adjust position sizes accordingly.

    This calculator takes into account different fee structures for:
    - Spot trading (maker/taker fees)
    - Futures trading (standard futures fees)
    - Market vs limit orders
    """

    def __init__(self):
        """Initialize the fee calculator with current settings."""
        # Use spot-specific fees if available, otherwise fall back to legacy names
        self.spot_maker_fee_rate = getattr(
            settings.trading, "spot_maker_fee_rate", settings.trading.maker_fee_rate
        )
        self.spot_taker_fee_rate = getattr(
            settings.trading, "spot_taker_fee_rate", settings.trading.taker_fee_rate
        )
        self.futures_fee_rate = settings.trading.futures_fee_rate
        self.enable_futures = settings.trading.enable_futures
        self.fee_tier_thresholds = getattr(settings.trading, "fee_tier_thresholds", [])

        # Legacy support
        self.maker_fee_rate = self.spot_maker_fee_rate
        self.taker_fee_rate = self.spot_taker_fee_rate

        # Track current volume tier
        self.current_volume = 0
        self.current_tier = self._get_fee_tier(0)

        logger.info(
            "Initialized FeeCalculator with rates: Spot Maker: %.4f, Spot Taker: %.4f, Futures: %.4f",
            self.spot_maker_fee_rate,
            self.spot_taker_fee_rate,
            self.futures_fee_rate,
        )

    def calculate_trade_fees(
        self,
        _trade_action: TradeAction,
        position_value: Decimal,
        current_price: Decimal,
        is_market_order: bool = True,
    ) -> TradeFees:
        """
        Calculate total fees for a complete trade (entry + exit).

        Args:
            trade_action: The trade action being executed
            position_value: The value of the position in USD
            current_price: Current market price
            is_market_order: Whether this is a market order (True) or limit order (False)

        Returns:
            TradeFees object with detailed fee breakdown
        """
        try:
            # Check for zero or negative position value
            if position_value <= 0:
                logger.debug(
                    "Position value must be positive for fee calculation"
                )  # Reduced from warning to debug
                return TradeFees(
                    entry_fee=Decimal(0),
                    exit_fee=Decimal(0),
                    total_fee=Decimal(0),
                    fee_rate=0.0,
                    net_position_value=Decimal(0),
                )

            # Check for zero or negative current price
            if current_price <= 0:
                logger.warning("Current price must be positive for fee calculation")
                return TradeFees(
                    entry_fee=Decimal(0),
                    exit_fee=Decimal(0),
                    total_fee=Decimal(0),
                    fee_rate=0.0,
                    net_position_value=Decimal(0),
                )

            # Determine the appropriate fee rate
            if self.enable_futures:
                fee_rate = self.futures_fee_rate
                logger.debug("Using futures fee rate: %.4f", fee_rate)
            else:
                # Use current tier rates for spot trading
                fee_rate = (
                    self.taker_fee_rate if is_market_order else self.maker_fee_rate
                )
                logger.debug(
                    "Using spot %s fee rate: %.4f (Volume: $%,.2f)",
                    "taker" if is_market_order else "maker",
                    fee_rate,
                    self.current_volume,
                )

            # Calculate entry fee
            entry_fee = position_value * Decimal(str(fee_rate))

            # Calculate exit fee (assume same fee rate for exit)
            exit_fee = position_value * Decimal(str(fee_rate))

            # Total fees
            total_fee = entry_fee + exit_fee

            # Net position value after fees
            net_position_value = position_value - total_fee

            fees = TradeFees(
                entry_fee=entry_fee,
                exit_fee=exit_fee,
                total_fee=total_fee,
                fee_rate=fee_rate,
                net_position_value=net_position_value,
            )

            logger.debug(
                "Calculated fees for $%.2f position: Entry: $%.2f, Exit: $%.2f, Total: $%.2f, Net: $%.2f",
                position_value,
                entry_fee,
                exit_fee,
                total_fee,
                net_position_value,
            )

        except Exception:
            logger.exception("Error calculating trade fees")
            # Return zero fees as fallback
            return TradeFees(
                entry_fee=Decimal(0),
                exit_fee=Decimal(0),
                total_fee=Decimal(0),
                fee_rate=0.0,
                net_position_value=position_value,
            )
        else:
            return fees

    def _create_zero_fee_result(
        self, trade_action: TradeAction, size_pct: float | None = None
    ) -> tuple[TradeAction, TradeFees]:
        """Helper method to create a zero fee result with adjusted trade action."""
        adjusted_action = trade_action.copy()
        if size_pct is not None:
            adjusted_action.size_pct = size_pct
        else:
            adjusted_action.size_pct = 0

        return adjusted_action, TradeFees(
            entry_fee=Decimal(0),
            exit_fee=Decimal(0),
            total_fee=Decimal(0),
            fee_rate=0.0,
            net_position_value=Decimal(0),
        )

    def _validate_position_inputs(
        self,
        trade_action: TradeAction,
        account_balance: Decimal,
        current_price: Decimal,
        leverage: int,
    ) -> str | None:
        """Validate inputs for position adjustment. Returns error message if invalid, None if valid."""
        if trade_action.action in ["HOLD", "CLOSE"]:
            return (
                "hold_close"  # Special case - not an error but needs different handling
            )

        if account_balance <= 0:
            logger.warning("Account balance must be positive for position sizing")
            return "invalid_balance"

        if trade_action.size_pct <= 0:
            logger.warning("Position size percentage must be positive")
            return "invalid_size"

        if current_price <= 0:
            logger.warning("Current price must be positive")
            return "invalid_price"

        if leverage == 0 and self.enable_futures:
            logger.warning("Leverage cannot be zero for futures trading")
            return "invalid_leverage"

        return None

    def adjust_position_size_for_fees(
        self,
        trade_action: TradeAction,
        account_balance: Decimal,
        current_price: Decimal,
        is_market_order: bool = True,
    ) -> tuple[TradeAction, TradeFees]:
        """
        Adjust position size to account for trading fees.

        This ensures that the total cost (position + fees) doesn't exceed
        the intended position size percentage of the account.

        Args:
            trade_action: Original trade action
            account_balance: Current account balance
            current_price: Current market price
            is_market_order: Whether this is a market order

        Returns:
            Tuple of (adjusted_trade_action, calculated_fees)
        """
        try:
            # Apply leverage first for validation
            leverage = trade_action.leverage or settings.trading.leverage

            # Validate all inputs
            validation_result = self._validate_position_inputs(
                trade_action, account_balance, current_price, leverage
            )

            if validation_result == "hold_close":
                return self._create_zero_fee_result(trade_action, trade_action.size_pct)
            if validation_result is not None:
                return self._create_zero_fee_result(trade_action)

            # Calculate intended position value
            intended_position_value = account_balance * Decimal(
                str(trade_action.size_pct / 100)
            )

            if intended_position_value == 0:
                logger.warning("Intended position value cannot be zero")
                return self._create_zero_fee_result(trade_action)

            leveraged_position_value = intended_position_value * Decimal(str(leverage))

            # Calculate fees for the intended position
            initial_fees = self.calculate_trade_fees(
                trade_action, leveraged_position_value, current_price, is_market_order
            )

            # Calculate available position value after fees
            available_for_position = intended_position_value - initial_fees.total_fee

            if available_for_position <= 0:
                logger.warning("Insufficient funds after accounting for fees")
                return self._create_zero_fee_result(trade_action)

            # Calculate new position size (same logic for both futures and spot)
            adjustment_ratio = available_for_position / intended_position_value
            new_size_pct = trade_action.size_pct * float(adjustment_ratio)

            # Create adjusted trade action
            adjusted_action = trade_action.copy()
            adjusted_action.size_pct = max(0, int(new_size_pct))

            # Recalculate fees for the adjusted position
            adjusted_position_value = account_balance * Decimal(
                str(adjusted_action.size_pct / 100)
            )
            adjusted_leveraged_value = adjusted_position_value * Decimal(str(leverage))

            final_fees = self.calculate_trade_fees(
                adjusted_action,
                adjusted_leveraged_value,
                current_price,
                is_market_order,
            )

            logger.info(
                "Adjusted position size for fees: %s%% -> %s%% ($%.2f total fees)",
                trade_action.size_pct,
                adjusted_action.size_pct,
                initial_fees.total_fee,
            )

        except Exception:
            logger.exception("Error adjusting position size for fees")
            # Return original action as fallback
            return self._create_zero_fee_result(trade_action, trade_action.size_pct)
        else:
            return adjusted_action, final_fees

    def calculate_minimum_profitable_move(
        self,
        position_value: Decimal,
        leverage: int | None = None,
        is_market_order: bool = True,
    ) -> Decimal:
        """
        Calculate the minimum price move needed to break even after fees.

        Args:
            position_value: Value of the position
            leverage: Trading leverage
            is_market_order: Whether using market orders

        Returns:
            Minimum percentage move needed to break even
        """
        try:
            leverage = leverage or settings.trading.leverage

            # Calculate total fees
            fees = self.calculate_trade_fees(
                # Dummy trade action for calculation
                TradeAction(
                    action="LONG",
                    size_pct=10,
                    take_profit_pct=2.0,
                    stop_loss_pct=1.0,
                    rationale="Fee calculation dummy action",
                ),
                position_value,
                Decimal(1000),  # Dummy price
                is_market_order,
            )

            # Check for zero position value to prevent division by zero
            if position_value == 0:
                logger.debug(
                    "Position value cannot be zero for minimum profitable move calculation"
                )  # Reduced from warning to debug
                return Decimal("0.001")  # 0.1% fallback

            # Check for zero leverage to prevent division by zero
            if leverage == 0:
                logger.warning(
                    "Leverage cannot be zero for minimum profitable move calculation"
                )
                return Decimal("0.001")  # 0.1% fallback

            # The minimum move needed is the fee percentage times 2 (round trip)
            # divided by leverage (since leverage amplifies the move)
            fee_percentage = float(fees.total_fee / position_value)
            min_move_percentage = fee_percentage / leverage

            logger.debug(
                "Minimum profitable move for $%.2f position with %sx leverage: %.4f%%",
                position_value,
                leverage,
                min_move_percentage * 100,
            )

        except Exception:
            logger.exception("Error calculating minimum profitable move")
            return Decimal("0.001")  # 0.1% fallback
        else:
            return Decimal(str(min_move_percentage))

    def validate_trade_profitability(
        self,
        trade_action: TradeAction,
        position_value: Decimal,
        _current_price: Decimal,
        leverage: int | None = None,
    ) -> tuple[bool, str]:
        """
        Validate that a trade has sufficient profit targets to cover fees.

        Args:
            trade_action: Trade action to validate
            position_value: Position value
            current_price: Current market price
            leverage: Trading leverage

        Returns:
            Tuple of (is_profitable, reason)
        """
        try:
            leverage = leverage or settings.trading.leverage

            # Calculate minimum profitable move
            min_move = self.calculate_minimum_profitable_move(
                position_value, leverage, is_market_order=True
            )

            # Convert percentages to decimal for comparison
            take_profit_decimal = Decimal(str(trade_action.take_profit_pct / 100))
            stop_loss_decimal = Decimal(str(trade_action.stop_loss_pct / 100))

            # Check if take profit is sufficient
            if take_profit_decimal < min_move * Decimal(
                "1.2"
            ):  # 1.2x safety margin (more aggressive for active trading)
                return (
                    False,
                    f"Take profit {trade_action.take_profit_pct:.2f}% too low to cover fees (min: {float(min_move * Decimal('1.2')):.4%})",
                )

            # Check if stop loss gives reasonable risk/reward
            if stop_loss_decimal == 0:
                logger.warning("Stop loss percentage cannot be zero")
                return (
                    False,
                    "Stop loss percentage cannot be zero for risk/reward calculation",
                )

            risk_reward_ratio = float(take_profit_decimal / stop_loss_decimal)
            if (
                risk_reward_ratio < 1.0
            ):  # Minimum 1:1 ratio after fees (more aggressive for active trading)
                return (
                    False,
                    f"Risk/reward ratio {risk_reward_ratio:.2f} too low after accounting for fees",
                )

        except Exception:
            logger.exception("Error validating trade profitability")
            return True, "Validation error - allowing trade"
        else:
            return True, "Trade profitability validated"

    def _get_fee_tier(self, volume: float) -> dict[str, float]:
        """
        Get the fee tier based on trading volume.

        Args:
            volume: Monthly trading volume in USD

        Returns:
            Fee tier dictionary with maker and taker rates
        """
        if not self.fee_tier_thresholds:
            return {
                "maker": self.spot_maker_fee_rate,
                "taker": self.spot_taker_fee_rate,
            }

        # Find the appropriate tier
        tier = self.fee_tier_thresholds[0]
        for threshold in self.fee_tier_thresholds:
            if volume >= threshold["volume"]:
                tier = threshold
            else:
                break

        return tier

    def update_volume_tier(self, monthly_volume: float):
        """
        Update the fee calculator with current monthly trading volume.

        Args:
            monthly_volume: Monthly trading volume in USD
        """
        self.current_volume = monthly_volume
        self.current_tier = self._get_fee_tier(monthly_volume)

        # Update fee rates based on current tier
        if not self.enable_futures:
            self.maker_fee_rate = self.current_tier["maker"]
            self.taker_fee_rate = self.current_tier["taker"]

            logger.info(
                "Updated fee tier based on $%,.2f volume: Maker: %.4f%%, Taker: %.4f%%",
                monthly_volume,
                self.maker_fee_rate * 100,
                self.taker_fee_rate * 100,
            )

    def get_fee_summary(self) -> dict[str, float]:
        """
        Get a summary of current fee rates.

        Returns:
            Dictionary with fee rate information
        """
        return {
            "spot_maker_fee_rate": self.spot_maker_fee_rate,
            "spot_taker_fee_rate": self.spot_taker_fee_rate,
            "current_maker_fee_rate": self.maker_fee_rate,
            "current_taker_fee_rate": self.taker_fee_rate,
            "futures_fee_rate": self.futures_fee_rate,
            "active_fee_rate": (
                self.futures_fee_rate if self.enable_futures else self.taker_fee_rate
            ),
            "current_volume": self.current_volume,
            "current_tier": self.current_tier,
        }


# Global fee calculator instance
fee_calculator = FeeCalculator()
