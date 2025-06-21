"""
Dynamic Spread Calculator for Market Making Strategy.

This module implements an intelligent spread calculator that uses VuManChu signals,
market conditions, and volatility analysis to determine optimal bid-ask spreads
for market making operations.

Key Features:
- Dynamic spread calculation based on multiple market factors
- VuManChu signal integration for directional bias adjustments
- Volatility-based spread sizing
- Liquidity-aware spread adjustments
- Multi-level spread recommendations
- Integration with BluefinFeeCalculator for minimum profitable spreads
- Real-time recalculation capabilities
"""

import logging
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, NamedTuple

from ..exchange.bluefin_fee_calculator import BluefinFeeCalculator
from ..indicators.vumanchu import CipherA, CipherB
from ..trading_types import IndicatorData

logger = logging.getLogger(__name__)


class VuManChuSignals(NamedTuple):
    """Aggregated VuManChu signals for spread calculation."""

    cipher_a_bullish: bool
    cipher_a_bearish: bool
    cipher_b_bullish: bool
    cipher_b_bearish: bool
    wavetrend_momentum: float  # -1.0 to 1.0
    signal_strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    trend_direction: str  # 'bullish', 'bearish', 'neutral'


class MarketConditions(NamedTuple):
    """Current market conditions affecting spread calculations."""

    volatility: float  # Current volatility (0.0 to 1.0)
    volume_ratio: float  # Current volume vs average (0.0 to 5.0+)
    liquidity_score: float  # Order book depth score (0.0 to 1.0)
    market_session: str  # 'active', 'quiet', 'volatile'
    time_factor: float  # Time-based adjustment (0.5 to 2.0)


class SpreadLevel(NamedTuple):
    """Individual spread level configuration."""

    level: int  # Level number (0 = best bid/ask)
    bid_spread_bps: Decimal  # Bid spread in basis points
    ask_spread_bps: Decimal  # Ask spread in basis points
    size_multiplier: float  # Size multiplier for this level


class SpreadRecommendation(NamedTuple):
    """Complete spread recommendation with all levels."""

    base_spread_bps: Decimal  # Base spread before adjustments
    adjusted_spread_bps: Decimal  # Final adjusted spread
    min_profitable_spread_bps: Decimal  # Minimum profitable spread
    directional_bias: float  # -1.0 (bearish) to 1.0 (bullish)
    bid_adjustment_bps: Decimal  # Bid-side adjustment
    ask_adjustment_bps: Decimal  # Ask-side adjustment
    levels: list[SpreadLevel]  # Multi-level spread configuration
    reasoning: str  # Calculation reasoning
    timestamp: datetime  # Calculation timestamp


class DynamicSpreadCalculator:
    """
    Intelligent spread calculator using VuManChu signals and market conditions.

    This calculator determines optimal bid-ask spreads for market making by analyzing:
    - Market volatility and liquidity conditions
    - VuManChu indicator signals for directional bias
    - Order book depth and trading volume
    - Time-based factors (market sessions, hours)
    - Minimum profitable spread requirements
    - Multi-level spread recommendations

    The calculator provides real-time spread adjustments based on changing market
    conditions and signal strength.
    """

    def __init__(
        self,
        fee_calculator: BluefinFeeCalculator,
        cipher_a: CipherA | None = None,
        cipher_b: CipherB | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Dynamic Spread Calculator.

        Args:
            fee_calculator: Bluefin fee calculator for minimum spreads
            cipher_a: VuManChu Cipher A indicator instance
            cipher_b: VuManChu Cipher B indicator instance
            config: Optional configuration overrides
        """
        self.fee_calculator = fee_calculator
        self.cipher_a = cipher_a
        self.cipher_b = cipher_b
        self.config = config or {}

        # Spread calculation parameters
        self.base_spread_bps = Decimal(
            str(self.config.get("base_spread_bps", 10))
        )  # 0.10%
        self.min_spread_bps = Decimal(
            str(self.config.get("min_spread_bps", 5))
        )  # 0.05%
        self.max_spread_bps = Decimal(
            str(self.config.get("max_spread_bps", 100))
        )  # 1.00%

        # Volatility adjustment parameters
        self.volatility_multiplier = self.config.get("volatility_multiplier", 2.0)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.3)

        # VuManChu signal adjustment parameters
        self.signal_adjustment_factor = self.config.get("signal_adjustment_factor", 0.5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)

        # Multi-level parameters
        self.num_levels = self.config.get("num_levels", 3)
        self.level_multiplier = self.config.get("level_multiplier", 1.5)

        # Market condition parameters
        self.volume_adjustment_factor = self.config.get("volume_adjustment_factor", 0.3)
        self.liquidity_adjustment_factor = self.config.get(
            "liquidity_adjustment_factor", 0.4
        )

        logger.info(
            "Initialized DynamicSpreadCalculator - Base: %s bps, Range: %s-%s bps",
            self.base_spread_bps,
            self.min_spread_bps,
            self.max_spread_bps,
        )

    def calculate_base_spread(
        self, volatility: float, volume: float, liquidity: float
    ) -> Decimal:
        """
        Calculate base spread based on market conditions.

        Args:
            volatility: Current market volatility (0.0 to 1.0)
            volume: Current volume ratio vs average (0.0 to 5.0+)
            liquidity: Order book liquidity score (0.0 to 1.0)

        Returns:
            Base spread in basis points
        """
        try:
            # Start with configured base spread
            base_spread = self.base_spread_bps

            # Volatility adjustment - increase spread in volatile markets
            if volatility > self.volatility_threshold:
                volatility_factor = (
                    1
                    + (volatility - self.volatility_threshold)
                    * self.volatility_multiplier
                )
                base_spread *= Decimal(str(volatility_factor))
                logger.debug(
                    "Applied volatility adjustment: %.2f%% -> %s bps",
                    volatility * 100,
                    base_spread,
                )

            # Volume adjustment - decrease spread in high volume
            if volume > 1.0:  # Above average volume
                volume_factor = 1 - min(
                    0.3, (volume - 1.0) * self.volume_adjustment_factor
                )
                base_spread *= Decimal(str(max(0.7, volume_factor)))
                logger.debug(
                    "Applied volume adjustment: %.2fx -> %s bps", volume, base_spread
                )

            # Liquidity adjustment - decrease spread in liquid markets
            if liquidity > 0.5:  # Good liquidity
                liquidity_factor = (
                    1 - (liquidity - 0.5) * self.liquidity_adjustment_factor
                )
                base_spread *= Decimal(str(max(0.8, liquidity_factor)))
                logger.debug(
                    "Applied liquidity adjustment: %.2f -> %s bps",
                    liquidity,
                    base_spread,
                )

            # Ensure spread is within bounds
            base_spread = max(
                self.min_spread_bps, min(self.max_spread_bps, base_spread)
            )
            base_spread = base_spread.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            logger.debug(
                "Calculated base spread: %s bps (vol: %.2f%%, vol_ratio: %.2fx, liq: %.2f)",
                base_spread,
                volatility * 100,
                volume,
                liquidity,
            )

            return base_spread

        except Exception as e:
            logger.error("Error calculating base spread: %s", e)
            return self.base_spread_bps

    def adjust_for_vumanchu_bias(
        self, base_spread: Decimal, signals: VuManChuSignals, confidence: float
    ) -> tuple[Decimal, Decimal, Decimal]:
        """
        Adjust spread for VuManChu directional bias.

        Args:
            base_spread: Base spread in basis points
            signals: VuManChu signals data
            confidence: Signal confidence (0.0 to 1.0)

        Returns:
            Tuple of (adjusted_spread, bid_adjustment, ask_adjustment)
        """
        try:
            if confidence < self.confidence_threshold:
                # Low confidence - use symmetric spread
                return base_spread, Decimal(0), Decimal(0)

            # Calculate directional bias adjustment
            bias_strength = abs(signals.wavetrend_momentum) * confidence
            adjustment_bps = base_spread * Decimal(
                str(bias_strength * self.signal_adjustment_factor)
            )

            bid_adjustment = Decimal(0)
            ask_adjustment = Decimal(0)

            if signals.trend_direction == "bullish":
                # Bullish bias - tighter ask spread, wider bid spread
                ask_adjustment = -adjustment_bps  # Tighter ask
                bid_adjustment = adjustment_bps * Decimal("0.5")  # Slightly wider bid

                logger.debug(
                    "Applied bullish bias: ask -%s bps, bid +%s bps",
                    adjustment_bps,
                    bid_adjustment,
                )

            elif signals.trend_direction == "bearish":
                # Bearish bias - tighter bid spread, wider ask spread
                bid_adjustment = -adjustment_bps  # Tighter bid
                ask_adjustment = adjustment_bps * Decimal("0.5")  # Slightly wider ask

                logger.debug(
                    "Applied bearish bias: bid -%s bps, ask +%s bps",
                    adjustment_bps,
                    ask_adjustment,
                )

            # Calculate final adjusted spread
            adjusted_spread = (
                base_spread + (abs(bid_adjustment) + abs(ask_adjustment)) / 2
            )
            adjusted_spread = max(
                self.min_spread_bps, min(self.max_spread_bps, adjusted_spread)
            )

            return adjusted_spread, bid_adjustment, ask_adjustment

        except Exception as e:
            logger.error("Error adjusting spread for VuManChu bias: %s", e)
            return base_spread, Decimal(0), Decimal(0)

    def adjust_for_market_conditions(
        self, spread: Decimal, market_state: MarketConditions
    ) -> Decimal:
        """
        Adjust spread based on current market conditions.

        Args:
            spread: Current spread in basis points
            market_state: Current market conditions

        Returns:
            Market-adjusted spread in basis points
        """
        try:
            adjusted_spread = spread

            # Market session adjustment
            if market_state.market_session == "volatile":
                adjusted_spread *= Decimal("1.2")  # 20% wider in volatile sessions
            elif market_state.market_session == "quiet":
                adjusted_spread *= Decimal("0.8")  # 20% tighter in quiet sessions

            # Time factor adjustment
            adjusted_spread *= Decimal(str(market_state.time_factor))

            # Ensure within bounds
            adjusted_spread = max(
                self.min_spread_bps, min(self.max_spread_bps, adjusted_spread)
            )
            adjusted_spread = adjusted_spread.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            logger.debug(
                "Market condition adjustment: %s -> %s bps (session: %s, time_factor: %.2f)",
                spread,
                adjusted_spread,
                market_state.market_session,
                market_state.time_factor,
            )

            return adjusted_spread

        except Exception as e:
            logger.error("Error adjusting spread for market conditions: %s", e)
            return spread

    def calculate_level_spreads(
        self,
        base_spread: Decimal,
        bid_adjustment: Decimal,
        ask_adjustment: Decimal,
        num_levels: int,
    ) -> list[SpreadLevel]:
        """
        Calculate multi-level spread configuration.

        Args:
            base_spread: Base spread in basis points
            bid_adjustment: Bid-side adjustment in basis points
            ask_adjustment: Ask-side adjustment in basis points
            num_levels: Number of levels to generate

        Returns:
            List of spread levels
        """
        try:
            levels = []

            for level in range(num_levels):
                # Calculate level multiplier (exponential growth)
                level_factor = Decimal(str(self.level_multiplier**level))

                # Calculate spreads for this level
                bid_spread = (base_spread + bid_adjustment) * level_factor
                ask_spread = (base_spread + ask_adjustment) * level_factor

                # Size decreases with level distance
                size_multiplier = 1.0 / (1 + level * 0.3)

                # Ensure minimum spreads
                bid_spread = max(self.min_spread_bps, bid_spread)
                ask_spread = max(self.min_spread_bps, ask_spread)

                # Quantize to reasonable precision
                bid_spread = bid_spread.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                ask_spread = ask_spread.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                levels.append(
                    SpreadLevel(
                        level=level,
                        bid_spread_bps=bid_spread,
                        ask_spread_bps=ask_spread,
                        size_multiplier=size_multiplier,
                    )
                )

                logger.debug(
                    "Level %d: bid %s bps, ask %s bps, size %.2fx",
                    level,
                    bid_spread,
                    ask_spread,
                    size_multiplier,
                )

            return levels

        except Exception as e:
            logger.error("Error calculating level spreads: %s", e)
            return []

    def get_minimum_profitable_spread(self, notional_value: Decimal) -> Decimal:
        """
        Calculate minimum profitable spread using fee calculator.

        Args:
            notional_value: Notional value of the trade

        Returns:
            Minimum profitable spread in basis points
        """
        try:
            # Calculate round-trip cost using fee calculator
            fees = self.fee_calculator.calculate_fees(notional_value)
            round_trip_cost = fees.round_trip_cost

            # Convert to basis points
            min_spread_bps = (round_trip_cost / notional_value) * Decimal(10000)

            # Add safety margin
            safety_margin_bps = min_spread_bps * Decimal("0.2")  # 20% safety margin
            min_profitable_spread = min_spread_bps + safety_margin_bps

            min_profitable_spread = min_profitable_spread.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            logger.debug(
                "Minimum profitable spread: %s bps (fees: %s, safety: %s bps)",
                min_profitable_spread,
                round_trip_cost,
                safety_margin_bps,
            )

            return min_profitable_spread

        except Exception as e:
            logger.error("Error calculating minimum profitable spread: %s", e)
            return self.min_spread_bps

    def extract_vumanchu_signals(
        self, indicator_data: IndicatorData
    ) -> VuManChuSignals:
        """
        Extract VuManChu signals from indicator data.

        Args:
            indicator_data: Latest indicator data

        Returns:
            Aggregated VuManChu signals
        """
        try:
            # Extract Cipher A signals
            cipher_a_data = (
                indicator_data.cipher_a if hasattr(indicator_data, "cipher_a") else {}
            )
            cipher_a_bullish = cipher_a_data.get("bullish_signal", False)
            cipher_a_bearish = cipher_a_data.get("bearish_signal", False)

            # Extract Cipher B signals
            cipher_b_data = (
                indicator_data.cipher_b if hasattr(indicator_data, "cipher_b") else {}
            )
            cipher_b_bullish = cipher_b_data.get("bullish_signal", False)
            cipher_b_bearish = cipher_b_data.get("bearish_signal", False)

            # Extract WaveTrend momentum
            wavetrend_data = cipher_a_data.get("wavetrend", {})
            wt1 = wavetrend_data.get("wt1", 0.0)
            wt2 = wavetrend_data.get("wt2", 0.0)
            wavetrend_momentum = float(wt1 - wt2) / 100.0  # Normalize to -1.0 to 1.0
            wavetrend_momentum = max(-1.0, min(1.0, wavetrend_momentum))

            # Calculate signal strength
            signal_count = sum(
                [cipher_a_bullish, cipher_a_bearish, cipher_b_bullish, cipher_b_bearish]
            )
            signal_strength = min(1.0, signal_count / 2.0)  # Normalize to 0.0-1.0

            # Calculate confidence based on signal alignment
            bullish_signals = cipher_a_bullish + cipher_b_bullish
            bearish_signals = cipher_a_bearish + cipher_b_bearish

            if bullish_signals > bearish_signals:
                trend_direction = "bullish"
                confidence = bullish_signals / 2.0
            elif bearish_signals > bullish_signals:
                trend_direction = "bearish"
                confidence = bearish_signals / 2.0
            else:
                trend_direction = "neutral"
                confidence = 0.0

            # Adjust confidence based on momentum strength
            momentum_strength = abs(wavetrend_momentum)
            confidence = min(1.0, confidence + momentum_strength * 0.3)

            return VuManChuSignals(
                cipher_a_bullish=cipher_a_bullish,
                cipher_a_bearish=cipher_a_bearish,
                cipher_b_bullish=cipher_b_bullish,
                cipher_b_bearish=cipher_b_bearish,
                wavetrend_momentum=wavetrend_momentum,
                signal_strength=signal_strength,
                confidence=confidence,
                trend_direction=trend_direction,
            )

        except Exception as e:
            logger.error("Error extracting VuManChu signals: %s", e)
            return VuManChuSignals(
                cipher_a_bullish=False,
                cipher_a_bearish=False,
                cipher_b_bullish=False,
                cipher_b_bearish=False,
                wavetrend_momentum=0.0,
                signal_strength=0.0,
                confidence=0.0,
                trend_direction="neutral",
            )

    def analyze_market_conditions(
        self, market_data: dict[str, Any], current_time: datetime | None = None
    ) -> MarketConditions:
        """
        Analyze current market conditions.

        Args:
            market_data: Current market data
            current_time: Current timestamp (defaults to now)

        Returns:
            Market conditions analysis
        """
        try:
            if current_time is None:
                current_time = datetime.now()

            # Extract volatility (using ATR or price change)
            volatility = market_data.get(
                "volatility", 0.2
            )  # Default moderate volatility
            volatility = max(0.0, min(1.0, volatility))

            # Extract volume ratio
            current_volume = market_data.get("volume", 1000)
            avg_volume = market_data.get("avg_volume", 1000)
            volume_ratio = current_volume / max(1, avg_volume)
            volume_ratio = max(0.0, min(5.0, volume_ratio))

            # Calculate liquidity score (based on order book depth)
            bid_depth = market_data.get("bid_depth", 0.5)
            ask_depth = market_data.get("ask_depth", 0.5)
            liquidity_score = min(1.0, (bid_depth + ask_depth) / 2.0)

            # Determine market session
            hour = current_time.hour
            if 8 <= hour <= 16:  # Active trading hours
                if volatility > 0.4:
                    market_session = "volatile"
                else:
                    market_session = "active"
            else:  # Off-hours
                market_session = "quiet"

            # Calculate time factor
            if 9 <= hour <= 15:  # Peak hours
                time_factor = 1.0
            elif 8 <= hour <= 16:  # Active hours
                time_factor = 0.9
            else:  # Off-hours
                time_factor = 1.2  # Wider spreads during off-hours

            return MarketConditions(
                volatility=volatility,
                volume_ratio=volume_ratio,
                liquidity_score=liquidity_score,
                market_session=market_session,
                time_factor=time_factor,
            )

        except Exception as e:
            logger.error("Error analyzing market conditions: %s", e)
            return MarketConditions(
                volatility=0.2,
                volume_ratio=1.0,
                liquidity_score=0.5,
                market_session="active",
                time_factor=1.0,
            )

    def get_spread_recommendations(
        self,
        indicator_data: IndicatorData,
        market_data: dict[str, Any],
        notional_value: Decimal,
        current_time: datetime | None = None,
    ) -> SpreadRecommendation:
        """
        Generate complete spread recommendations.

        Args:
            indicator_data: Latest indicator data
            market_data: Current market data
            notional_value: Notional value for minimum spread calculation
            current_time: Current timestamp (defaults to now)

        Returns:
            Complete spread recommendation
        """
        try:
            if current_time is None:
                current_time = datetime.now()

            # Extract signals and market conditions
            signals = self.extract_vumanchu_signals(indicator_data)
            market_conditions = self.analyze_market_conditions(
                market_data, current_time
            )

            logger.info(
                "Calculating spreads - Trend: %s (%.2f confidence), Volatility: %.2f%%, Volume: %.2fx",
                signals.trend_direction,
                signals.confidence,
                market_conditions.volatility * 100,
                market_conditions.volume_ratio,
            )

            # Calculate base spread
            base_spread = self.calculate_base_spread(
                market_conditions.volatility,
                market_conditions.volume_ratio,
                market_conditions.liquidity_score,
            )

            # Adjust for VuManChu bias
            adjusted_spread, bid_adjustment, ask_adjustment = (
                self.adjust_for_vumanchu_bias(base_spread, signals, signals.confidence)
            )

            # Adjust for market conditions
            final_spread = self.adjust_for_market_conditions(
                adjusted_spread, market_conditions
            )

            # Calculate minimum profitable spread
            min_profitable_spread = self.get_minimum_profitable_spread(notional_value)

            # Ensure spread meets minimum requirements
            if final_spread < min_profitable_spread:
                logger.warning(
                    "Adjusting spread to meet minimum profitable requirement: %s -> %s bps",
                    final_spread,
                    min_profitable_spread,
                )
                final_spread = min_profitable_spread
                # Recalculate adjustments proportionally
                adjustment_factor = final_spread / adjusted_spread
                bid_adjustment *= adjustment_factor
                ask_adjustment *= adjustment_factor

            # Calculate multi-level spreads
            levels = self.calculate_level_spreads(
                final_spread, bid_adjustment, ask_adjustment, self.num_levels
            )

            # Calculate directional bias
            directional_bias = signals.wavetrend_momentum * signals.confidence

            # Generate reasoning
            reasoning_parts = [
                f"Base spread: {base_spread} bps",
                f"Volatility: {market_conditions.volatility:.1%}",
                f"Volume ratio: {market_conditions.volume_ratio:.1f}x",
                f"Trend: {signals.trend_direction} ({signals.confidence:.1%} confidence)",
                f"Session: {market_conditions.market_session}",
            ]
            reasoning = " | ".join(reasoning_parts)

            recommendation = SpreadRecommendation(
                base_spread_bps=base_spread,
                adjusted_spread_bps=final_spread,
                min_profitable_spread_bps=min_profitable_spread,
                directional_bias=directional_bias,
                bid_adjustment_bps=bid_adjustment,
                ask_adjustment_bps=ask_adjustment,
                levels=levels,
                reasoning=reasoning,
                timestamp=current_time,
            )

            logger.info(
                "Spread recommendation: %s bps (bias: %.2f, levels: %d) - %s",
                final_spread,
                directional_bias,
                len(levels),
                reasoning,
            )

            return recommendation

        except Exception as e:
            logger.error("Error generating spread recommendations: %s", e)
            # Return conservative fallback recommendation
            return SpreadRecommendation(
                base_spread_bps=self.base_spread_bps,
                adjusted_spread_bps=self.base_spread_bps,
                min_profitable_spread_bps=self.min_spread_bps,
                directional_bias=0.0,
                bid_adjustment_bps=Decimal(0),
                ask_adjustment_bps=Decimal(0),
                levels=[
                    SpreadLevel(0, self.base_spread_bps, self.base_spread_bps, 1.0)
                ],
                reasoning="Fallback recommendation due to calculation error",
                timestamp=current_time or datetime.now(),
            )
