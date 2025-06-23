"""
Market state analyzer for LLM input preparation.

This module extracts market analysis methods from the LLMAgent class
to provide a focused, reusable market state preparation interface.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from ...trading_types import IndicatorData, MarketState

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Analyzes market state and prepares data for LLM consumption."""

    @staticmethod
    async def prepare_llm_input(
        market_state: MarketState,
        omnisearch_client: Any | None = None,
        omnisearch_enabled: bool = False,
    ) -> dict[str, Any]:
        """
        Prepare market state data for LLM input.

        Args:
            market_state: Market state object containing all market data
            omnisearch_client: Optional OmniSearch client for financial context
            omnisearch_enabled: Whether OmniSearch is enabled

        Returns:
            Dictionary formatted for prompt template
        """
        # Get recent OHLCV data (show summary of 24h data)
        all_candles = market_state.ohlcv_data

        # Show detailed last 10 candles
        recent_candles = all_candles[-10:] if len(all_candles) >= 10 else all_candles

        ohlcv_lines = []
        ohlcv_lines.append(f"=== Last 10 candles (of {len(all_candles)} total) ===")
        for candle in recent_candles:
            ohlcv_lines.append(
                f"{candle.timestamp.strftime('%H:%M')}: "
                f"O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}"
            )

        # Add 24h summary statistics
        if len(all_candles) > 10:
            ohlcv_lines.append("\n=== 24h Summary ===")
            high_24h = max(c.high for c in all_candles)
            low_24h = min(c.low for c in all_candles)
            volume_24h = sum(c.volume for c in all_candles)
            open_24h = all_candles[0].open
            close_24h = all_candles[-1].close
            change_24h = ((close_24h - open_24h) / open_24h) * 100

            ohlcv_lines.append(f"24h High: {high_24h}, Low: {low_24h}")
            ohlcv_lines.append(f"24h Change: {change_24h:+.2f}%")
            ohlcv_lines.append(f"24h Volume: {volume_24h}")
            ohlcv_lines.append(f"Total candles in context: {len(all_candles)}")

        ohlcv_tail = "\n".join(ohlcv_lines)

        # Format current position
        if market_state.current_position.side == "FLAT":
            position_str = "No position (flat)"
        else:
            position_str = f"{market_state.current_position.side} {market_state.current_position.size} @ {market_state.current_position.entry_price}"

        # Get futures-specific information if available
        margin_health = "N/A"
        available_margin: str | float = "N/A"

        if hasattr(market_state, "futures_account") and market_state.futures_account:
            margin_health = market_state.futures_account.margin_info.health_status.value
            available_margin = float(
                market_state.futures_account.margin_info.available_margin
            )
        elif (
            hasattr(market_state, "margin_requirements")
            and market_state.margin_requirements
        ):
            margin_health = market_state.margin_requirements.health_status.value
            available_margin = float(market_state.margin_requirements.available_margin)

        # Extract dominance data if available
        dominance_data = {
            "usdt_dominance": market_state.indicators.usdt_dominance or "N/A",
            "usdc_dominance": market_state.indicators.usdc_dominance or "N/A",
            "stablecoin_dominance": market_state.indicators.stablecoin_dominance
            or "N/A",
            "dominance_trend": market_state.indicators.dominance_trend or "N/A",
            "dominance_rsi": market_state.indicators.dominance_rsi or "N/A",
            "market_sentiment": market_state.indicators.market_sentiment or "UNKNOWN",
        }

        # Format dominance candlestick data for analysis
        dominance_candles_analysis = MarketAnalyzer.format_dominance_candles(
            market_state
        )

        # Format VuManChu dominance technical analysis
        dominance_vumanchu_analysis = MarketAnalyzer.format_dominance_vumanchu_analysis(
            market_state
        )

        # Calculate Cipher B signal alignment
        cipher_b_alignment = MarketAnalyzer.calculate_cipher_b_alignment(
            market_state.indicators
        )

        # Get financial context from OmniSearch if enabled
        financial_context = (
            "OmniSearch disabled - no external market intelligence available"
        )
        if omnisearch_enabled and omnisearch_client:
            try:
                # Extract base symbol for searches
                base_symbol = market_state.symbol.split("-")[0]  # "BTC-USD" -> "BTC"

                # This would be implemented in the actual LLMAgent
                # For now, we just return a placeholder
                financial_context = (
                    f"Financial context for {base_symbol} would be fetched here"
                )
            except Exception as e:
                logger.warning(f"Error getting financial context: {e}")

        # Placeholder for scalping analysis
        scalping_analysis = {
            "scalping_enabled": False,
            "scalping_signals": "Scalping analysis not available",
        }

        return {
            "symbol": market_state.symbol,
            "interval": market_state.interval,
            "current_price": float(market_state.current_price),
            "current_position": position_str,
            "margin_health": margin_health,
            "available_margin": available_margin,
            "cipher_a_dot": market_state.indicators.cipher_a_dot,
            "cipher_b_wave": market_state.indicators.cipher_b_wave,
            "cipher_b_money_flow": market_state.indicators.cipher_b_money_flow,
            "rsi": market_state.indicators.rsi,
            "ema_fast": market_state.indicators.ema_fast,
            "ema_slow": market_state.indicators.ema_slow,
            "ohlcv_tail": ohlcv_tail,
            "dominance_candles_analysis": dominance_candles_analysis,
            "dominance_vumanchu_analysis": dominance_vumanchu_analysis,
            "cipher_b_alignment": cipher_b_alignment,
            "max_size_pct": settings.trading.max_size_pct,
            "leverage": settings.trading.leverage,
            "max_leverage": settings.trading.max_futures_leverage,
            "futures_enabled": settings.trading.enable_futures,
            "auto_cash_transfer": settings.trading.auto_cash_transfer,
            "financial_context": financial_context,
            **dominance_data,
            **scalping_analysis,
        }

    @staticmethod
    def format_dominance_candles(market_state: MarketState) -> str:
        """
        Format dominance candlestick data for LLM analysis.

        Args:
            market_state: Market state containing dominance candles

        Returns:
            Formatted string with dominance candlestick analysis
        """
        try:
            # Try to get dominance candles from MarketState
            dominance_candles = getattr(market_state, "dominance_candles", None)

            if not dominance_candles:
                return "No dominance candle data available for analysis"

            # Take last 5 candles for analysis (like OHLCV)
            recent_candles = (
                dominance_candles[-5:]
                if len(dominance_candles) >= 5
                else dominance_candles
            )

            if not recent_candles:
                return "Insufficient dominance candle data for analysis"

            # Format candlestick data
            candle_lines = []
            for i, candle in enumerate(recent_candles):
                # Determine candle color/direction
                direction = (
                    "🟢"
                    if candle.close > candle.open
                    else "🔴" if candle.close < candle.open else "⚪"
                )
                change_pct = (
                    ((candle.close - candle.open) / candle.open * 100)
                    if candle.open > 0
                    else 0
                )

                # Format time
                time_str = (
                    candle.timestamp.strftime("%H:%M")
                    if hasattr(candle, "timestamp")
                    else f"T-{len(recent_candles) - i}"
                )

                # Create candle summary
                candle_line = (
                    f"{time_str}: {direction} O:{candle.open:.2f}% H:{candle.high:.2f}% "
                    f"L:{candle.low:.2f}% C:{candle.close:.2f}% ({change_pct:+.2f}%)"
                )

                # Add technical indicators if available
                indicators = []
                if hasattr(candle, "rsi") and candle.rsi is not None:
                    indicators.append(f"RSI:{candle.rsi:.1f}")
                if hasattr(candle, "trend_signal") and candle.trend_signal:
                    indicators.append(f"Signal:{candle.trend_signal}")

                if indicators:
                    candle_line += f" [{', '.join(indicators)}]"

                candle_lines.append(candle_line)

            # Calculate overall trend
            if len(recent_candles) >= 2:
                first_close = recent_candles[0].close
                last_close = recent_candles[-1].close
                overall_trend = (
                    ((last_close - first_close) / first_close * 100)
                    if first_close > 0
                    else 0
                )

                trend_direction = (
                    "RISING"
                    if overall_trend > 0.1
                    else "FALLING" if overall_trend < -0.1 else "SIDEWAYS"
                )
                trend_line = f"Overall Trend: {trend_direction} ({overall_trend:+.2f}% over {len(recent_candles)} candles)"
            else:
                trend_line = "Overall Trend: Insufficient data"

            # Create analysis summary
            analysis_lines = [
                f"Last {len(recent_candles)} Dominance Candles (3-minute intervals):",
                *candle_lines,
                trend_line,
                f"Latest Dominance: {recent_candles[-1].close:.2f}% ({'increasing stablecoin inflows' if recent_candles[-1].close > recent_candles[-1].open else 'decreasing dominance'})",
            ]

            return "\n".join(analysis_lines)

        except Exception as e:
            logger.warning(f"Error formatting dominance candles: {e}")
            return f"Error formatting dominance candles: {e!s}"

    @staticmethod
    def format_dominance_vumanchu_analysis(market_state: MarketState) -> str:
        """
        Format VuManChu dominance technical analysis for LLM consumption.

        Args:
            market_state: Market state containing dominance VuManChu indicators

        Returns:
            Formatted string with VuManChu dominance analysis
        """
        try:
            indicators = market_state.indicators

            # Check if we have VuManChu dominance analysis
            if not hasattr(indicators, "dominance_cipher_a_signal"):
                return "VuManChu dominance analysis not available"

            analysis_lines = []

            # Cipher A Dominance Analysis
            cipher_a_signal = getattr(indicators, "dominance_cipher_a_signal", 0)
            cipher_a_confidence = getattr(
                indicators, "dominance_cipher_a_confidence", 0.0
            )

            if cipher_a_signal != 0:
                signal_text = (
                    "🔴 BEARISH for crypto"
                    if cipher_a_signal > 0
                    else "🟢 BULLISH for crypto"
                )
                analysis_lines.append(
                    f"Dominance Cipher A: {signal_text} (confidence: {cipher_a_confidence:.1f}%)"
                )
            else:
                analysis_lines.append("Dominance Cipher A: ⚪ NEUTRAL")

            # Cipher B Dominance Analysis
            cipher_b_signal = getattr(indicators, "dominance_cipher_b_signal", 0)
            cipher_b_confidence = getattr(
                indicators, "dominance_cipher_b_confidence", 0.0
            )

            if cipher_b_signal != 0:
                signal_text = (
                    "🔴 BEARISH for crypto"
                    if cipher_b_signal > 0
                    else "🟢 BULLISH for crypto"
                )
                analysis_lines.append(
                    f"Dominance Cipher B: {signal_text} (confidence: {cipher_b_confidence:.1f}%)"
                )
            else:
                analysis_lines.append("Dominance Cipher B: ⚪ NEUTRAL")

            # WaveTrend on Dominance
            wt1 = getattr(indicators, "dominance_wt1", None)
            wt2 = getattr(indicators, "dominance_wt2", None)
            if wt1 is not None and wt2 is not None:
                wt_condition = ""
                if wt2 > 60:
                    wt_condition = "📈 Overbought dominance (bearish for crypto)"
                elif wt2 < -60:
                    wt_condition = "📉 Oversold dominance (bullish for crypto)"
                else:
                    wt_condition = "📊 Neutral zone"
                analysis_lines.append(
                    f"Dominance WaveTrend: WT1={wt1:.1f}, WT2={wt2:.1f} - {wt_condition}"
                )

            # Price vs Dominance Divergence
            price_divergence = getattr(indicators, "dominance_price_divergence", "NONE")
            if price_divergence != "NONE":
                divergence_emoji = {
                    "BULLISH": "🚀",
                    "BEARISH": "🔻",
                    "HIDDEN_BULLISH": "💎",
                    "HIDDEN_BEARISH": "⚠️",
                }.get(price_divergence, "❓")
                analysis_lines.append(
                    f"Price-Dominance Divergence: {divergence_emoji} {price_divergence}"
                )
            else:
                analysis_lines.append(
                    "Price-Dominance Divergence: ➡️ No divergence detected"
                )

            # Overall Dominance Sentiment
            dominance_sentiment = getattr(indicators, "dominance_sentiment", "NEUTRAL")
            sentiment_emoji = {
                "STRONG_BULLISH": "🚀🚀",
                "BULLISH": "🚀",
                "NEUTRAL": "➡️",
                "BEARISH": "🔻",
                "STRONG_BEARISH": "🔻🔻",
            }.get(dominance_sentiment, "❓")
            analysis_lines.append(
                f"VuManChu Dominance Sentiment: {sentiment_emoji} {dominance_sentiment}"
            )

            # Key Insight
            analysis_lines.append("")
            analysis_lines.append(
                "📝 Key Insight: Dominance signals are INVERTED for crypto:"
            )
            analysis_lines.append(
                "   • Rising dominance = Money flowing to stables = Bearish for crypto"
            )
            analysis_lines.append(
                "   • Falling dominance = Money leaving stables = Bullish for crypto"
            )
            analysis_lines.append(
                "   • Cipher signals on dominance predict stablecoin flow direction"
            )

            return "\n".join(analysis_lines)

        except Exception as e:
            logger.warning(f"Error formatting VuManChu dominance analysis: {e}")
            return f"Error formatting VuManChu dominance analysis: {e!s}"

    @staticmethod
    def calculate_cipher_b_alignment(indicators: IndicatorData) -> str:
        """
        Calculate and describe Cipher B signal alignment.

        Args:
            indicators: IndicatorData object

        Returns:
            String describing Cipher B signal alignment
        """
        if indicators.cipher_b_wave is None or indicators.cipher_b_money_flow is None:
            return "Cipher B indicators not available"

        wave = indicators.cipher_b_wave
        money_flow = indicators.cipher_b_money_flow

        # Determine signal states
        wave_bullish = wave > 0.0
        wave_bearish = wave < 0.0
        money_flow_bullish = money_flow > 50.0
        money_flow_bearish = money_flow < 50.0

        # Check alignments
        bullish_aligned = wave_bullish and money_flow_bullish
        bearish_aligned = wave_bearish and money_flow_bearish

        # Build alignment description
        lines = []
        lines.append(f"Wave: {wave:.2f} ({'bullish' if wave_bullish else 'bearish'})")
        lines.append(
            f"Money Flow: {money_flow:.2f} ({'bullish' if money_flow_bullish else 'bearish'})"
        )

        if bullish_aligned:
            lines.append("✓ BULLISH ALIGNMENT - Both signals confirm upward momentum")
            lines.append("Traditional Cipher B would trigger LONG here")
        elif bearish_aligned:
            lines.append("✓ BEARISH ALIGNMENT - Both signals confirm downward momentum")
            lines.append("Traditional Cipher B would trigger SHORT here")
        else:
            lines.append("⚠ MIXED SIGNALS - Wave and Money Flow disagree")
            lines.append("Traditional Cipher B would wait for alignment")

        # Add signal strength
        wave_strength = abs(wave)
        if wave_strength > 60:
            lines.append(f"Wave strength: STRONG ({wave_strength:.1f})")
        elif wave_strength > 30:
            lines.append(f"Wave strength: MODERATE ({wave_strength:.1f})")
        else:
            lines.append(f"Wave strength: WEAK ({wave_strength:.1f})")

        return "\n".join(lines)
