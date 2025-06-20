"""
Core strategy logic and fallback rule-based trading with enterprise-grade error handling.

This module provides fallback trading logic when LLM is unavailable
and core strategy utilities. Includes error boundaries, circuit breakers,
and comprehensive recovery mechanisms.
"""

import logging
from typing import Any, Literal, cast

from bot.config import settings
from bot.error_handling import ErrorBoundary
from bot.system_monitor import error_recovery_manager, system_monitor
from bot.trading_types import IndicatorData, MarketState, TradeAction

logger = logging.getLogger(__name__)


class CoreStrategy:
    """
    Core strategy implementation with rule-based trading logic and enterprise-grade error handling.

    Provides fallback trading decisions and strategy utilities
    that can operate independently of LLM components. Includes error boundaries,
    automatic recovery, and comprehensive failure handling.
    """

    def __init__(self):
        """Initialize the core strategy with error handling capabilities."""
        self.max_size_pct = settings.trading.max_size_pct
        self.default_tp_pct = 2.0
        self.default_sl_pct = 1.5

        # Error handling components
        self._error_boundary = ErrorBoundary(
            component_name="core_strategy",
            fallback_behavior=self._strategy_error_fallback,
            max_retries=2,
            retry_delay=1.0,
        )

        # Strategy health tracking
        self._analysis_error_count = 0
        self._last_successful_analysis = None
        self._total_analyses = 0
        self._failed_analyses = 0

        # Register health check with system monitor
        system_monitor.register_component(
            "core_strategy",
            self._health_check,
            [
                error_recovery_manager._get_recovery_action(
                    "strategy_reset", self._reset_strategy_state
                )
            ],
        )

        logger.info("Initialized CoreStrategy with error handling")

    async def _strategy_error_fallback(self, error: Exception, context: dict) -> None:
        """Fallback behavior for strategy errors."""
        logger.warning("Strategy error fallback triggered: %s", error)

        # Increment error count
        self._analysis_error_count += 1

        # Attempt specific recovery based on error type
        if "indicator" in str(error).lower():
            await error_recovery_manager.recover_from_error(
                "data_error",
                {"component": "indicators", "error": str(error)},
                "core_strategy",
            )
        else:
            await error_recovery_manager.recover_from_error(
                "strategy_error",
                {"error": str(error), "context": context},
                "core_strategy",
            )

    async def _health_check(self) -> bool:
        """Health check for core strategy component."""
        try:
            # Check if too many recent failures
            error_rate = self._failed_analyses / max(self._total_analyses, 1)

            # Consider unhealthy if error rate > 50% or too many consecutive errors
            is_healthy = error_rate < 0.5 and self._analysis_error_count < 5

            if not is_healthy:
                logger.warning(
                    "Strategy health check failed: error_rate=%.2f%%, consecutive_errors=%s",
                    error_rate * 100,
                    self._analysis_error_count,
                )

            return is_healthy

        except Exception:
            logger.exception("Strategy health check error")
        else:
            return False

    async def _reset_strategy_state(self, _component_name: str, _health: Any) -> None:
        """Recovery action to reset strategy state."""
        logger.info("Resetting core strategy state")

        self._analysis_error_count = 0
        self._failed_analyses = max(0, self._failed_analyses - 1)

        logger.info("Core strategy state reset completed")

    def analyze_market(self, market_state: MarketState) -> TradeAction:
        """
        Analyze market using rule-based logic.

        Args:
            market_state: Complete market state

        Returns:
            TradeAction based on technical analysis rules
        """
        try:
            indicators = market_state.indicators
            current_position = market_state.current_position

            # Determine market bias
            market_bias = self._get_market_bias(indicators)

            # Generate action based on bias and current position
            action = self._determine_action(market_bias, current_position)

            # Calculate position size
            size_pct = self._calculate_position_size(market_bias, indicators)

            # Determine TP/SL levels
            tp_pct, sl_pct = self._calculate_risk_levels(market_bias, indicators)

            # Create rationale
            rationale = self._create_rationale(market_bias, action)

            trade_action = TradeAction(
                action=cast("Literal['LONG', 'SHORT', 'CLOSE', 'HOLD']", action),
                size_pct=size_pct,
                take_profit_pct=tp_pct,
                stop_loss_pct=sl_pct,
                rationale=rationale,
            )

            logger.info("Core strategy decision: %s (%s)", action, rationale)
            return trade_action

        except Exception:
            logger.exception("Error in core strategy analysis")
        else:
            return self._get_safe_action()

    def _get_market_bias(self, indicators: IndicatorData) -> str:
        """
        Determine overall market bias from indicators.

        Args:
            indicators: Technical indicator values

        Returns:
            Market bias: 'bullish', 'bearish', 'neutral'
        """
        bullish_signals = 0.0
        bearish_signals = 0.0

        # Cipher A analysis
        if indicators.cipher_a_dot is not None:
            if indicators.cipher_a_dot > 0:
                bullish_signals += 1.0
            elif indicators.cipher_a_dot < 0:
                bearish_signals += 1.0

        # Cipher B analysis
        if indicators.cipher_b_wave is not None:
            if indicators.cipher_b_wave > 0:
                bullish_signals += 1.0
            elif indicators.cipher_b_wave < 0:
                bearish_signals += 1.0

        # Money flow analysis
        if indicators.cipher_b_money_flow is not None:
            if indicators.cipher_b_money_flow > 60:
                bullish_signals += 1.0
            elif indicators.cipher_b_money_flow < 40:
                bearish_signals += 1.0

        # RSI analysis
        if (
            indicators.rsi is not None
            and 30 < indicators.rsi < 70
            and indicators.ema_fast
            and indicators.ema_slow
        ):
            # Neutral RSI, look at trend
            if indicators.ema_fast > indicators.ema_slow:
                bullish_signals += 0.5
            else:
                bearish_signals += 0.5

        # Determine bias
        if bullish_signals > bearish_signals + 1:
            return "bullish"
        if bearish_signals > bullish_signals + 1:
            return "bearish"
        return "neutral"

    def _determine_action(self, market_bias: str, current_position) -> str:
        """
        Determine trading action based on bias and current position.

        Args:
            market_bias: Market bias from indicator analysis
            current_position: Current trading position

        Returns:
            Trading action: 'LONG', 'SHORT', 'CLOSE', 'HOLD'
        """
        pos_side = current_position.side

        if market_bias == "bullish":
            if pos_side == "FLAT":
                return "LONG"
            if pos_side == "SHORT":
                return "CLOSE"
            # Already LONG
            return "HOLD"

        if market_bias == "bearish":
            if pos_side == "FLAT":
                return "SHORT"
            if pos_side == "LONG":
                return "CLOSE"
            # Already SHORT
            return "HOLD"

        if pos_side != "FLAT":
            # Consider closing position in neutral market
            return "CLOSE"
        return "HOLD"

    def _calculate_position_size(
        self, market_bias: str, indicators: IndicatorData
    ) -> int:
        """
        Calculate position size based on market conditions.

        Args:
            market_bias: Market bias
            indicators: Technical indicators

        Returns:
            Position size as percentage (0-100)
        """
        if market_bias == "neutral":
            return 0

        base_size = 15  # Base position size

        # Adjust size based on signal strength
        signal_strength = self._assess_signal_strength(indicators)

        if signal_strength >= 0.8:
            size_pct = min(base_size + 5, self.max_size_pct)
        elif signal_strength >= 0.6:
            size_pct = base_size
        elif signal_strength >= 0.4:
            size_pct = max(base_size - 5, 5)
        else:
            size_pct = 0

        return size_pct

    def _assess_signal_strength(self, indicators: IndicatorData) -> float:
        """
        Assess the strength of trading signals.

        Args:
            indicators: Technical indicator values

        Returns:
            Signal strength between 0.0 and 1.0
        """
        strength_factors = []

        # Cipher A strength
        if indicators.cipher_a_dot is not None:
            strength_factors.append(min(abs(indicators.cipher_a_dot), 1.0))

        # Cipher B wave strength
        if indicators.cipher_b_wave is not None:
            normalized_wave = (
                abs(indicators.cipher_b_wave) / 100.0
            )  # Assuming typical wave range
            strength_factors.append(min(normalized_wave, 1.0))

        # Money flow conviction
        if indicators.cipher_b_money_flow is not None:
            mf_strength = abs(indicators.cipher_b_money_flow - 50) / 50.0
            strength_factors.append(mf_strength)

        if not strength_factors:
            return 0.0

        return sum(strength_factors) / len(strength_factors)

    def _calculate_risk_levels(
        self, _market_bias: str, indicators: IndicatorData
    ) -> tuple[float, float]:
        """
        Calculate take profit and stop loss levels.

        Args:
            market_bias: Market bias
            indicators: Technical indicators

        Returns:
            Tuple of (take_profit_pct, stop_loss_pct)
        """
        # Base levels
        tp_pct = self.default_tp_pct
        sl_pct = self.default_sl_pct

        # Adjust based on market conditions
        signal_strength = self._assess_signal_strength(indicators)

        if signal_strength >= 0.8:
            # Strong signal - wider targets
            tp_pct = 3.0
            sl_pct = 1.0
        elif signal_strength <= 0.4:
            # Weak signal - tighter stops
            tp_pct = 1.5
            sl_pct = 2.0

        return tp_pct, sl_pct

    def _create_rationale(self, market_bias: str, action: str) -> str:
        """
        Create a brief rationale for the trading decision.

        Args:
            market_bias: Market bias
            action: Trading action

        Returns:
            Brief rationale string
        """
        bias_desc = {
            "bullish": "uptrend signals",
            "bearish": "downtrend signals",
            "neutral": "mixed signals",
        }

        if action in ["LONG", "SHORT"]:
            return f"Core strategy: {bias_desc[market_bias]} favor {action.lower()}"
        if action == "CLOSE":
            return f"Core strategy: {bias_desc[market_bias]} suggest close"
        return f"Core strategy: {bias_desc[market_bias]} recommend hold"

    def _get_safe_action(self) -> TradeAction:
        """
        Get a safe default action in case of errors.

        Returns:
            Safe TradeAction (HOLD)
        """
        return TradeAction(
            action="HOLD",
            size_pct=0,
            take_profit_pct=self.default_tp_pct,
            stop_loss_pct=self.default_sl_pct,
            rationale="Core strategy: Error occurred - holding safe",
        )

    def validate_action(self, action: TradeAction) -> TradeAction:
        """
        Validate and potentially modify a trade action.

        Args:
            action: Original trade action

        Returns:
            Validated/modified trade action
        """
        validated = action.copy()

        # Ensure size doesn't exceed maximum
        if validated.size_pct > self.max_size_pct:
            validated.size_pct = self.max_size_pct
            logger.warning("Position size capped at %s%", self.max_size_pct)

        # Ensure reasonable TP/SL levels
        if validated.take_profit_pct > 10.0:
            validated.take_profit_pct = 10.0
            logger.warning("Take profit capped at 10%")

        if validated.stop_loss_pct > 5.0:
            validated.stop_loss_pct = 5.0
            logger.warning("Stop loss capped at 5%")

        return validated
