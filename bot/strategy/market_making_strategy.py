"""
Market Making Strategy Core Implementation.

This module implements a market making strategy that integrates with VuManChu indicators
to provide directional bias and dynamic spread adjustments for optimal market making.

Key Features:
- Integration with VuManChu Cipher A & B indicators for directional bias
- Dynamic spread calculation based on market conditions and signal strength
- Order level generation with bias-adjusted positioning
- Risk-aware spread calculations using fee structures
- Comprehensive error handling and logging
"""

import logging
from decimal import Decimal
from typing import Any, NamedTuple

from bot.exchange.bluefin_fee_calculator import BluefinFeeCalculator
from bot.indicators.vumanchu import CipherA, CipherB
from bot.trading_types import IndicatorData, MarketState, TradeAction

from .core import CoreStrategy

logger = logging.getLogger(__name__)


class DirectionalBias(NamedTuple):
    """Directional bias calculated from VuManChu indicators."""

    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signals: dict[str, Any]  # Individual signal values


class SpreadCalculation(NamedTuple):
    """Calculated optimal spread for market making."""

    base_spread: Decimal
    adjusted_spread: Decimal
    bid_adjustment: Decimal
    ask_adjustment: Decimal
    min_profitable_spread: Decimal


class OrderLevel(NamedTuple):
    """Individual order level for market making."""

    side: str  # 'BUY' or 'SELL'
    price: Decimal
    size: Decimal
    level: int  # Order book level (0 = best bid/ask)


class MarketMakingStrategy:
    """
    Core Market Making Strategy with VuManChu Integration.

    This strategy creates liquidity by placing buy and sell orders around the current
    market price, using VuManChu indicators to determine directional bias and adjust
    spreads dynamically.

    Features:
    - VuManChu signal integration for directional bias
    - Dynamic spread adjustment based on signal strength
    - Multi-level order placement with bias adjustments
    - Fee-aware minimum profitable spread calculations
    - Risk management and position size controls
    """

    def __init__(
        self,
        fee_calculator: BluefinFeeCalculator,
        exchange_client: Any,  # Exchange client interface
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Market Making Strategy.

        Args:
            fee_calculator: Bluefin fee calculator for spread calculations
            exchange_client: Exchange client for order operations
            config: Optional configuration overrides
        """
        self.fee_calculator = fee_calculator
        self.exchange_client = exchange_client
        self.config = config or {}

        # Initialize core strategy for fallback logic
        self.core_strategy = CoreStrategy()

        # Market making parameters (configurable)
        self.base_spread_bps = self.config.get("base_spread_bps", 10)  # 0.10%
        self.max_spread_bps = self.config.get("max_spread_bps", 50)  # 0.50%
        self.min_spread_bps = self.config.get("min_spread_bps", 5)  # 0.05%
        self.order_levels = self.config.get("order_levels", 3)  # Number of order levels
        self.max_position_pct = self.config.get(
            "max_position_pct", 25
        )  # Max 25% position
        self.bias_adjustment_factor = self.config.get(
            "bias_adjustment_factor", 0.3
        )  # 30% adjustment

        # VuManChu indicators (will be initialized when needed)
        self._cipher_a: CipherA | None = None
        self._cipher_b: CipherB | None = None

        # Strategy state
        self._last_market_state: MarketState | None = None
        self._last_bias: DirectionalBias | None = None

        logger.info(
            "Initialized MarketMakingStrategy with base_spread=%.2f bps, levels=%d",
            self.base_spread_bps,
            self.order_levels,
        )

    def _initialize_indicators(self) -> None:
        """Initialize VuManChu indicators if not already done."""
        if self._cipher_a is None:
            try:
                # Use scalping-optimized parameters for market making
                self._cipher_a = CipherA(
                    wt_channel_length=6,
                    wt_average_length=8,
                    wt_ma_length=3,
                    overbought_level=45.0,
                    oversold_level=-45.0,
                )
                logger.info("Initialized VuManChu Cipher A for market making")
            except Exception as e:
                logger.exception("Failed to initialize Cipher A: %s", e)

        if self._cipher_b is None:
            try:
                # Initialize Cipher B with default parameters
                self._cipher_b = CipherB()
                logger.info("Initialized VuManChu Cipher B for market making")
            except Exception as e:
                logger.exception("Failed to initialize Cipher B: %s", e)

    def calculate_optimal_spread(
        self, market_state: MarketState, vumanchu_signals: IndicatorData | None = None
    ) -> SpreadCalculation:
        """
        Calculate optimal spread based on market conditions and VuManChu signals.

        Args:
            market_state: Current market state with OHLCV data
            vumanchu_signals: Optional pre-calculated VuManChu signals

        Returns:
            SpreadCalculation with optimal spreads and adjustments

        Raises:
            ValueError: If market state is invalid
            RuntimeError: If fee calculation fails
        """
        try:
            current_price = Decimal(str(market_state.current_price))

            # Get directional bias from VuManChu signals
            bias = self.determine_directional_bias(
                vumanchu_signals or market_state.indicators
            )

            # Calculate base spread in price terms
            base_spread_decimal = Decimal(str(self.base_spread_bps)) / Decimal(10000)
            base_spread_price = current_price * base_spread_decimal

            # Calculate minimum profitable spread using fee calculator
            try:
                # Estimate notional value for fee calculation (using 1 unit for calculation)
                notional_value = current_price * Decimal(1)
                round_trip_fee = self.fee_calculator.calculate_round_trip_cost(
                    notional_value
                )
                min_profitable_spread = round_trip_fee.round_trip_cost * Decimal(
                    2
                )  # 2x safety margin

                # Convert to price terms
                min_profitable_spread_price = min_profitable_spread

            except Exception as e:
                logger.warning("Fee calculation failed, using fallback: %s", e)
                min_spread_decimal = Decimal(str(self.min_spread_bps)) / Decimal(10000)
                min_profitable_spread_price = current_price * min_spread_decimal

            # Adjust spread based on directional bias
            if bias.direction == "bullish":
                # Bullish bias: tighter ask, wider bid
                bid_adjustment = base_spread_price * Decimal(
                    str(bias.strength * self.bias_adjustment_factor)
                )
                ask_adjustment = -base_spread_price * Decimal(
                    str(bias.strength * self.bias_adjustment_factor * 0.5)
                )
            elif bias.direction == "bearish":
                # Bearish bias: tighter bid, wider ask
                bid_adjustment = -base_spread_price * Decimal(
                    str(bias.strength * self.bias_adjustment_factor * 0.5)
                )
                ask_adjustment = base_spread_price * Decimal(
                    str(bias.strength * self.bias_adjustment_factor)
                )
            else:
                # Neutral: symmetric spread
                bid_adjustment = Decimal(0)
                ask_adjustment = Decimal(0)

            # Calculate final adjusted spread
            adjusted_spread = max(
                base_spread_price + abs(bid_adjustment) + abs(ask_adjustment),
                min_profitable_spread_price,
            )

            # Apply maximum spread limit
            max_spread_decimal = Decimal(str(self.max_spread_bps)) / Decimal(10000)
            max_spread_price = current_price * max_spread_decimal
            adjusted_spread = min(adjusted_spread, max_spread_price)

            result = SpreadCalculation(
                base_spread=base_spread_price,
                adjusted_spread=adjusted_spread,
                bid_adjustment=bid_adjustment,
                ask_adjustment=ask_adjustment,
                min_profitable_spread=min_profitable_spread_price,
            )

            logger.debug(
                "Calculated spread: base=%.6f, adjusted=%.6f, bias=%s (%.2f)",
                float(base_spread_price),
                float(adjusted_spread),
                bias.direction,
                bias.strength,
            )

            return result

        except Exception as e:
            logger.exception("Error calculating optimal spread: %s", e)
            # Return safe default spread
            safe_spread = (
                Decimal(str(current_price))
                * Decimal(str(self.base_spread_bps))
                / Decimal(10000)
            )
            return SpreadCalculation(
                base_spread=safe_spread,
                adjusted_spread=safe_spread,
                bid_adjustment=Decimal(0),
                ask_adjustment=Decimal(0),
                min_profitable_spread=safe_spread,
            )

    def determine_directional_bias(
        self, vumanchu_signals: IndicatorData
    ) -> DirectionalBias:
        """
        Determine directional bias from VuManChu indicators.

        Args:
            vumanchu_signals: VuManChu indicator signals

        Returns:
            DirectionalBias with direction, strength, and confidence
        """
        try:
            signals = {}
            bullish_weight = 0.0
            bearish_weight = 0.0
            total_signals = 0

            # Cipher A Dot signal
            if vumanchu_signals.cipher_a_dot is not None:
                signals["cipher_a_dot"] = vumanchu_signals.cipher_a_dot
                if vumanchu_signals.cipher_a_dot > 0:
                    bullish_weight += abs(vumanchu_signals.cipher_a_dot)
                elif vumanchu_signals.cipher_a_dot < 0:
                    bearish_weight += abs(vumanchu_signals.cipher_a_dot)
                total_signals += 1

            # Cipher B Wave signal
            if vumanchu_signals.cipher_b_wave is not None:
                signals["cipher_b_wave"] = vumanchu_signals.cipher_b_wave
                if vumanchu_signals.cipher_b_wave > 0:
                    bullish_weight += min(
                        abs(vumanchu_signals.cipher_b_wave) / 100.0, 1.0
                    )
                elif vumanchu_signals.cipher_b_wave < 0:
                    bearish_weight += min(
                        abs(vumanchu_signals.cipher_b_wave) / 100.0, 1.0
                    )
                total_signals += 1

            # Cipher B Money Flow signal
            if vumanchu_signals.cipher_b_money_flow is not None:
                signals["cipher_b_money_flow"] = vumanchu_signals.cipher_b_money_flow
                if vumanchu_signals.cipher_b_money_flow > 55:
                    bullish_weight += (vumanchu_signals.cipher_b_money_flow - 50) / 50.0
                elif vumanchu_signals.cipher_b_money_flow < 45:
                    bearish_weight += (50 - vumanchu_signals.cipher_b_money_flow) / 50.0
                total_signals += 1

            # RSI for momentum confirmation
            if vumanchu_signals.rsi is not None:
                signals["rsi"] = vumanchu_signals.rsi
                if 30 <= vumanchu_signals.rsi <= 70:  # Not in extreme zones
                    if vumanchu_signals.rsi > 50:
                        bullish_weight += (
                            (vumanchu_signals.rsi - 50) / 50.0 * 0.5
                        )  # Lower weight for RSI
                    else:
                        bearish_weight += (50 - vumanchu_signals.rsi) / 50.0 * 0.5
                total_signals += 0.5  # RSI gets half weight

            # EMA trend confirmation
            if (
                vumanchu_signals.ema_fast is not None
                and vumanchu_signals.ema_slow is not None
            ):
                signals["ema_trend"] = (
                    vumanchu_signals.ema_fast - vumanchu_signals.ema_slow
                )
                ema_diff_pct = (
                    vumanchu_signals.ema_fast - vumanchu_signals.ema_slow
                ) / vumanchu_signals.ema_slow
                if ema_diff_pct > 0.001:  # 0.1% threshold
                    bullish_weight += min(abs(ema_diff_pct) * 10, 0.5)  # Cap at 0.5
                elif ema_diff_pct < -0.001:
                    bearish_weight += min(abs(ema_diff_pct) * 10, 0.5)
                total_signals += 0.5  # EMA gets half weight

            # Calculate direction and strength
            if total_signals == 0:
                direction = "neutral"
                strength = 0.0
                confidence = 0.0
            else:
                net_weight = bullish_weight - bearish_weight
                total_weight = bullish_weight + bearish_weight

                if abs(net_weight) < 0.1:  # Very close to neutral
                    direction = "neutral"
                    strength = 0.0
                elif net_weight > 0:
                    direction = "bullish"
                    strength = min(bullish_weight / total_signals, 1.0)
                else:
                    direction = "bearish"
                    strength = min(bearish_weight / total_signals, 1.0)

                # Confidence based on signal agreement
                if total_weight > 0:
                    confidence = min(max(abs(net_weight) / total_weight, 0.0), 1.0)
                else:
                    confidence = 0.0

            bias = DirectionalBias(
                direction=direction,
                strength=strength,
                confidence=confidence,
                signals=signals,
            )

            # Cache the bias for logging
            self._last_bias = bias

            logger.debug(
                "Directional bias: %s (strength=%.2f, confidence=%.2f)",
                direction.upper(),
                strength,
                confidence,
            )

            return bias

        except Exception as e:
            logger.exception("Error determining directional bias: %s", e)
            # Return neutral bias on error
            return DirectionalBias(
                direction="neutral", strength=0.0, confidence=0.0, signals={}
            )

    def generate_order_levels(
        self, current_price: Decimal, spread: SpreadCalculation, bias: DirectionalBias
    ) -> list[OrderLevel]:
        """
        Generate multiple order levels for market making.

        Args:
            current_price: Current market price
            spread: Calculated spread information
            bias: Directional bias from VuManChu

        Returns:
            List of OrderLevel objects for bid and ask sides
        """
        try:
            order_levels = []

            # Calculate base position size per level
            base_size_pct = min(
                self.max_position_pct / (self.order_levels * 2), 5.0
            )  # Max 5% per level

            for level in range(self.order_levels):
                # Calculate price offsets with increasing distance
                level_multiplier = Decimal(str(1 + level * 0.5))  # 1.0, 1.5, 2.0, ...

                # Bid side (buy orders)
                bid_offset = spread.adjusted_spread * level_multiplier / Decimal(2)
                bid_offset += spread.bid_adjustment  # Apply bias adjustment
                bid_price = current_price - bid_offset

                # Ask side (sell orders)
                ask_offset = spread.adjusted_spread * level_multiplier / Decimal(2)
                ask_offset += spread.ask_adjustment  # Apply bias adjustment
                ask_price = current_price + ask_offset

                # Adjust size based on bias and level
                if bias.direction == "bullish":
                    # More aggressive on bid side
                    bid_size_adjustment = 1.0 + (bias.strength * 0.3)
                    ask_size_adjustment = 1.0 - (bias.strength * 0.2)
                elif bias.direction == "bearish":
                    # More aggressive on ask side
                    bid_size_adjustment = 1.0 - (bias.strength * 0.2)
                    ask_size_adjustment = 1.0 + (bias.strength * 0.3)
                else:
                    bid_size_adjustment = ask_size_adjustment = 1.0

                # Calculate final sizes
                bid_size = Decimal(str(base_size_pct * bid_size_adjustment))
                ask_size = Decimal(str(base_size_pct * ask_size_adjustment))

                # Add bid order
                order_levels.append(
                    OrderLevel(side="BUY", price=bid_price, size=bid_size, level=level)
                )

                # Add ask order
                order_levels.append(
                    OrderLevel(side="SELL", price=ask_price, size=ask_size, level=level)
                )

            logger.debug(
                "Generated %d order levels (price=%.6f, spread=%.6f)",
                len(order_levels),
                float(current_price),
                float(spread.adjusted_spread),
            )

            return order_levels

        except Exception as e:
            logger.exception("Error generating order levels: %s", e)
            return []

    def analyze_market_making_opportunity(
        self, market_state: MarketState
    ) -> TradeAction:
        """
        Analyze market for market making opportunities.

        Args:
            market_state: Current market state

        Returns:
            TradeAction with market making recommendations
        """
        try:
            self._last_market_state = market_state

            # Initialize indicators if needed
            self._initialize_indicators()

            # Determine directional bias
            bias = self.determine_directional_bias(market_state.indicators)

            # Calculate optimal spread
            spread = self.calculate_optimal_spread(
                market_state, market_state.indicators
            )

            # Generate order levels
            current_price = Decimal(str(market_state.current_price))
            order_levels = self.generate_order_levels(current_price, spread, bias)

            # Create market making action
            if len(order_levels) > 0:
                # Use the first level as the primary action
                primary_level = order_levels[0]

                # Determine action based on bias and current position
                current_pos = market_state.current_position
                if bias.direction == "bullish" and current_pos.side != "LONG":
                    action = "LONG"
                    size_pct = float(primary_level.size)
                elif bias.direction == "bearish" and current_pos.side != "SHORT":
                    action = "SHORT"
                    size_pct = float(primary_level.size)
                else:
                    action = "HOLD"
                    size_pct = 0.0

                # Calculate TP/SL based on spread
                spread_pct = float(
                    spread.adjusted_spread / current_price * Decimal(100)
                )
                take_profit_pct = max(spread_pct * 2, 1.0)  # At least 1%
                stop_loss_pct = max(spread_pct, 0.5)  # At least 0.5%

                rationale = (
                    f"Market making: {bias.direction} bias (str={bias.strength:.2f}, "
                    f"conf={bias.confidence:.2f}), spread={spread_pct:.3f}%"
                )

            else:
                # Fallback to hold if no valid levels
                action = "HOLD"
                size_pct = 0.0
                take_profit_pct = 1.0
                stop_loss_pct = 1.0
                rationale = "Market making: No valid order levels generated"

            trade_action = TradeAction(
                action=action,  # type: ignore
                size_pct=size_pct,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                rationale=rationale,
            )

            logger.info("Market making analysis: %s - %s", action, rationale)

            return trade_action

        except Exception as e:
            logger.exception("Error in market making analysis: %s", e)
            # Fallback to core strategy
            return self.core_strategy.analyze_market(market_state)

    def get_current_bias(self) -> DirectionalBias | None:
        """
        Get the last calculated directional bias.

        Returns:
            Last DirectionalBias or None if not calculated yet
        """
        return self._last_bias

    def get_strategy_status(self) -> dict[str, Any]:
        """
        Get current strategy status and configuration.

        Returns:
            Dictionary with strategy status information
        """
        return {
            "strategy_type": "market_making",
            "base_spread_bps": self.base_spread_bps,
            "max_spread_bps": self.max_spread_bps,
            "min_spread_bps": self.min_spread_bps,
            "order_levels": self.order_levels,
            "max_position_pct": self.max_position_pct,
            "bias_adjustment_factor": self.bias_adjustment_factor,
            "cipher_a_initialized": self._cipher_a is not None,
            "cipher_b_initialized": self._cipher_b is not None,
            "last_bias": self._last_bias._asdict() if self._last_bias else None,
            "fee_calculator_type": type(self.fee_calculator).__name__,
        }

    def validate_market_conditions(self, market_state: MarketState) -> bool:
        """
        Validate that market conditions are suitable for market making.

        Args:
            market_state: Current market state

        Returns:
            True if conditions are suitable for market making
        """
        try:
            # Check if we have sufficient market data
            if len(market_state.ohlcv_data) < 10:
                logger.warning("Insufficient OHLCV data for market making")
                return False

            # Check for reasonable volatility (not too high, not too low)
            recent_candles = market_state.ohlcv_data[-10:]
            price_changes = [
                abs(candle.close - candle.open) / candle.open
                for candle in recent_candles
                if candle.open > 0
            ]

            if price_changes:
                avg_volatility = sum(price_changes) / len(price_changes)
                # Avoid market making in extremely volatile or dead markets
                if avg_volatility > 0.05 or avg_volatility < 0.001:  # 5% or 0.1%
                    logger.warning(
                        "Market volatility (%.3f%%) not suitable for market making",
                        avg_volatility * 100,
                    )
                    return False

            # Check if we have valid indicators
            indicators = market_state.indicators
            if (
                indicators.cipher_a_dot is None
                and indicators.cipher_b_wave is None
                and indicators.rsi is None
            ):
                logger.warning("No valid indicators for market making bias calculation")
                return False

            return True

        except Exception as e:
            logger.exception("Error validating market conditions: %s", e)
            return False
