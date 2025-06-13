"""
VuManChu Cipher A & B indicator implementations.

This module provides complete Python implementations of the VuManChu Cipher indicators,
originally written in Pine Script. Integrates all newly implemented components:
- WaveTrend Oscillator as core component
- Cipher A Signals (Diamond patterns, Yellow Cross, Candle patterns)
- Cipher B Signals (Buy/Sell circles, Gold signals, Divergence)
- 8-EMA Ribbon system
- RSI+MFI Combined indicator
- Stochastic RSI and Schaff Trend Cycle
- Sommi Pattern recognition
- Complete divergence detection system

Provides 100% Pine Script compatibility with exact parameter matching.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from .cipher_a_signals import CipherASignals
from .cipher_b_signals import CipherBSignals
from .divergence_detector import DivergenceDetector
from .ema_ribbon import EMAribbon
from .rsimfi import RSIMFIIndicator
from .schaff_trend_cycle import SchaffTrendCycle
from .sommi_patterns import SommiPatterns
from .stochastic_rsi import StochasticRSI

# Import all newly implemented components
from .wavetrend import WaveTrend

logger = logging.getLogger(__name__)


class CipherA:
    """
    VuManChu Cipher A Complete Implementation.

    Integrates all advanced Cipher A components:
    - WaveTrend Oscillator (Channel=9, Average=13, MA=3)
    - 8-EMA Ribbon system [5,11,15,18,21,24,28,34]
    - RSI+MFI Combined indicator (Period=60, Multiplier=150)
    - Advanced Signal Patterns:
      * Diamond patterns (Red/Green diamonds)
      * Yellow Cross signals with precise conditions
      * Extreme diamonds (Dump/Moon)
      * Bull/Bear candle patterns
    - Stochastic RSI and Schaff Trend Cycle
    - Complete divergence detection
    - Signal strength and confidence analysis

    Maintains exact Pine Script parameter compatibility.
    """

    def __init__(
        self,
        # WaveTrend parameters (Pine Script defaults for Cipher A)
        wt_channel_length: int = 9,
        wt_average_length: int = 13,
        wt_ma_length: int = 3,
        # Overbought/Oversold levels
        overbought_level: float = 60.0,
        overbought_level2: float = 53.0,
        overbought_level3: float = 100.0,
        oversold_level: float = -60.0,
        oversold_level2: float = -53.0,
        oversold_level3: float = -100.0,
        # RSI parameters
        rsi_length: int = 14,
        # RSI+MFI parameters (Pine Script defaults)
        rsimfi_period: int = 60,
        rsimfi_multiplier: float = 150.0,
        # EMA Ribbon lengths (Pine Script defaults)
        ema_ribbon_lengths: list[int] | None = None,
        # Stochastic RSI parameters
        stoch_rsi_length: int = 14,
        stoch_k_smooth: int = 3,
        stoch_d_smooth: int = 3,
        # Schaff Trend Cycle parameters
        stc_length: int = 10,
        stc_fast_length: int = 23,
        stc_slow_length: int = 50,
        stc_factor: float = 0.5,
    ):
        """
        Initialize Cipher A with all integrated components.

        Args:
            wt_channel_length: WaveTrend channel length (Pine Script: 9)
            wt_average_length: WaveTrend average length (Pine Script: 13)
            wt_ma_length: WaveTrend MA length (Pine Script: 3)
            overbought_level: Primary overbought level (60.0)
            overbought_level2: Secondary overbought level (53.0)
            overbought_level3: Extreme overbought level (100.0)
            oversold_level: Primary oversold level (-60.0)
            oversold_level2: Secondary oversold level (-53.0)
            oversold_level3: Extreme oversold level (-100.0)
            rsi_length: RSI calculation period (14)
            rsimfi_period: RSI+MFI period (Pine Script: 60)
            rsimfi_multiplier: RSI+MFI multiplier (Pine Script: 150.0)
            ema_ribbon_lengths: EMA ribbon lengths (default: [5,11,15,18,21,24,28,34])
            stoch_rsi_length: Stochastic RSI length (14)
            stoch_k_smooth: Stochastic K smoothing (3)
            stoch_d_smooth: Stochastic D smoothing (3)
            stc_length: Schaff Trend Cycle length (10)
            stc_fast_length: STC fast length (23)
            stc_slow_length: STC slow length (50)
            stc_factor: STC factor (0.5)
        """
        # Initialize core components
        self.wavetrend = WaveTrend(
            channel_length=wt_channel_length,
            average_length=wt_average_length,
            ma_length=wt_ma_length,
            overbought_level=overbought_level,
            oversold_level=oversold_level,
        )

        self.cipher_a_signals = CipherASignals(
            wt_channel_length=wt_channel_length,
            wt_average_length=wt_average_length,
            wt_ma_length=wt_ma_length,
            overbought_level=overbought_level,
            overbought_level2=overbought_level2,
            overbought_level3=overbought_level3,
            oversold_level=oversold_level,
            oversold_level2=oversold_level2,
            oversold_level3=oversold_level3,
            rsi_length=rsi_length,
            rsimfi_period=rsimfi_period,
            rsimfi_multiplier=rsimfi_multiplier,
        )

        self.ema_ribbon = EMAribbon(
            lengths=ema_ribbon_lengths or [5, 11, 15, 18, 21, 24, 28, 34]
        )

        self.rsimfi = RSIMFIIndicator()

        self.stochastic_rsi = StochasticRSI(
            rsi_length=stoch_rsi_length,
            smooth_k=stoch_k_smooth,
            smooth_d=stoch_d_smooth,
        )

        self.schaff_trend_cycle = SchaffTrendCycle(
            length=stc_length,
            fast_length=stc_fast_length,
            slow_length=stc_slow_length,
            factor=stc_factor,
        )

        self.divergence_detector = DivergenceDetector()

        # Store parameters for compatibility
        self.wt_channel_length = wt_channel_length
        self.wt_average_length = wt_average_length
        self.wt_ma_length = wt_ma_length
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.rsi_length = rsi_length
        self.rsimfi_period = rsimfi_period
        self.rsimfi_multiplier = rsimfi_multiplier

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate complete Cipher A indicators with all integrated components.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with comprehensive Cipher A indicators:

            Core WaveTrend:
            - wt1, wt2: WaveTrend oscillator values
            - wt_overbought, wt_oversold: OB/OS conditions
            - wt_cross_up, wt_cross_down: Basic cross signals

            Advanced Signal Patterns:
            - red_diamond, green_diamond: Diamond patterns
            - yellow_cross_up, yellow_cross_down: Yellow cross signals
            - dump_diamond, moon_diamond: Extreme diamonds
            - bull_candle, bear_candle: Candle patterns

            8-EMA Ribbon:
            - ema1 through ema8: All ribbon EMAs
            - ema_ribbon_bullish, ema_ribbon_bearish: Ribbon direction
            - ema_cross_signals: Various EMA crossover signals

            Additional Indicators:
            - rsi: Standard RSI
            - rsimfi: RSI+MFI combined indicator
            - stoch_rsi_k, stoch_rsi_d: Stochastic RSI
            - stc: Schaff Trend Cycle

            Signal Analysis:
            - cipher_a_signal: Overall signal (-1, 0, 1)
            - cipher_a_bullish_strength: Bullish signal strength
            - cipher_a_bearish_strength: Bearish signal strength
            - cipher_a_confidence: Signal confidence (0-100)

            Divergence Detection:
            - divergence_bullish, divergence_bearish: Divergence signals
        """
        # Validate input data
        min_length = (
            max(
                self.wt_channel_length,
                self.wt_average_length,
                max(self.ema_ribbon.lengths),
                self.rsimfi_period,
                self.rsi_length,
            )
            + 20
        )  # Buffer for lookbacks

        if len(df) < min_length:
            logger.warning(
                f"Insufficient data for Cipher A calculation. Need {min_length}, got {len(df)}"
            )
            return df.copy()

        result = df.copy()

        # Ensure all input columns are proper float64 dtype
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype(
                    "float64"
                )

        try:
            # Step 1: Calculate 8-EMA Ribbon system
            logger.debug("Calculating 8-EMA Ribbon system")
            result = self.ema_ribbon.calculate_ema_ribbon(result)
            result = self.ema_ribbon.calculate_ribbon_direction(result)
            result = self.ema_ribbon.calculate_crossover_signals(result)

            # Map ribbon column names to expected names
            if "ribbon_bullish" in result.columns:
                result["ema_ribbon_bullish"] = result["ribbon_bullish"]
            if "ribbon_bearish" in result.columns:
                result["ema_ribbon_bearish"] = result["ribbon_bearish"]

            # Step 2: Calculate core WaveTrend oscillator
            logger.debug("Calculating WaveTrend oscillator")
            result = self.wavetrend.calculate(result)

            # Step 3: Calculate RSI+MFI combined indicator
            logger.debug("Calculating RSI+MFI combined indicator")
            rsimfi_values = self.rsimfi.calculate_rsimfi(
                result, period=self.rsimfi_period, multiplier=self.rsimfi_multiplier
            )
            result["rsimfi"] = rsimfi_values.astype("float64")

            # Step 4: Calculate standard RSI
            logger.debug("Calculating standard RSI")
            rsi_values = ta.rsi(result["close"], length=self.rsi_length)
            result["rsi"] = (
                rsi_values.astype("float64")
                if rsi_values is not None
                else pd.Series(50.0, index=result.index)
            )

            # Step 5: Calculate Stochastic RSI
            logger.debug("Calculating Stochastic RSI")
            result = self.stochastic_rsi.calculate(result)

            # Step 6: Calculate Schaff Trend Cycle
            logger.debug("Calculating Schaff Trend Cycle")
            result = self.schaff_trend_cycle.calculate(result)

            # Step 7: Calculate all advanced Cipher A signal patterns
            logger.debug("Calculating advanced Cipher A signal patterns")
            result = self.cipher_a_signals.get_all_cipher_a_signals(result)

            # Step 8: Calculate divergence signals
            logger.debug("Calculating divergence signals")
            # Use the correct method names for divergence detection
            try:
                wt_regular_divs = self.divergence_detector.detect_regular_divergences(
                    result["wt2"], result["close"], 60.0, -60.0
                )
                wt_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    result["wt2"], result["close"]
                )

                rsi_regular_divs = self.divergence_detector.detect_regular_divergences(
                    result["rsi"], result["close"], 70.0, 30.0
                )
                rsi_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    result["rsi"], result["close"]
                )

                # Convert divergence signals to boolean series
                from .divergence_detector import DivergenceType

                # Initialize divergence series
                result["wt_divergence_bullish"] = pd.Series(False, index=result.index)
                result["wt_divergence_bearish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bullish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bearish"] = pd.Series(False, index=result.index)

                # Process WT divergences
                for div in wt_regular_divs + wt_hidden_divs:
                    if div.end_fractal.index < len(result):
                        if div.type in [
                            DivergenceType.REGULAR_BULLISH,
                            DivergenceType.HIDDEN_BULLISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("wt_divergence_bullish"),
                            ] = True
                        elif div.type in [
                            DivergenceType.REGULAR_BEARISH,
                            DivergenceType.HIDDEN_BEARISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("wt_divergence_bearish"),
                            ] = True

                # Process RSI divergences
                for div in rsi_regular_divs + rsi_hidden_divs:
                    if div.end_fractal.index < len(result):
                        if div.type in [
                            DivergenceType.REGULAR_BULLISH,
                            DivergenceType.HIDDEN_BULLISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("rsi_divergence_bullish"),
                            ] = True
                        elif div.type in [
                            DivergenceType.REGULAR_BEARISH,
                            DivergenceType.HIDDEN_BEARISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("rsi_divergence_bearish"),
                            ] = True

            except Exception as div_error:
                logger.warning(
                    f"Error calculating divergences in Cipher A: {div_error}"
                )
                # Set default divergence columns
                result["wt_divergence_bullish"] = pd.Series(False, index=result.index)
                result["wt_divergence_bearish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bullish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bearish"] = pd.Series(False, index=result.index)

            # Combine all divergence signals
            result["divergence_bullish"] = (
                result["wt_divergence_bullish"] | result["rsi_divergence_bullish"]
            )
            result["divergence_bearish"] = (
                result["wt_divergence_bearish"] | result["rsi_divergence_bearish"]
            )

            # Step 9: Enhanced signal generation with all components
            result["cipher_a_signal"] = self._generate_enhanced_signals(result)

            # Step 10: Add compatibility indicators for backward compatibility
            self._add_compatibility_indicators(result)

            logger.debug("Cipher A calculation completed successfully")

        except Exception as e:
            logger.error(f"Error in Cipher A calculation: {str(e)}")
            # Add error indicators
            result["cipher_a_error"] = True
            result["cipher_a_signal"] = 0
            if "cipher_a_confidence" not in result.columns:
                result["cipher_a_confidence"] = 0.0

        return result

    def _generate_enhanced_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate enhanced buy/sell signals using all Cipher A components.

        Uses advanced signal patterns with weighted strength and confidence.

        Args:
            df: DataFrame with all calculated indicators

        Returns:
            Series with signals: 1 = buy, -1 = sell, 0 = hold
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        try:
            # Get pre-calculated signal from cipher_a_signals if available
            if "cipher_a_signal" in df.columns:
                df["cipher_a_signal"].fillna(0)
            else:
                pd.Series(0, index=df.index)

            # Get confidence score
            confidence = df.get("cipher_a_confidence", pd.Series(0.0, index=df.index))

            # Get signal strengths
            bullish_strength = df.get(
                "cipher_a_bullish_strength", pd.Series(0.0, index=df.index)
            )
            bearish_strength = df.get(
                "cipher_a_bearish_strength", pd.Series(0.0, index=df.index)
            )

            # Enhanced signal conditions with multiple confirmations

            # Strong Bullish Signals (require high confidence)
            strong_bullish = (
                (confidence >= 50.0)
                & (bullish_strength >= 2.0)
                & (
                    df.get("yellow_cross_up", False)
                    | df.get("moon_diamond", False)
                    | (
                        df.get("green_diamond", False)
                        & df.get("ema_ribbon_bullish", False)
                        & df.get("wt_cross_up", False)
                    )
                )
            )

            # Strong Bearish Signals (require high confidence)
            strong_bearish = (
                (confidence >= 50.0)
                & (bearish_strength >= 2.0)
                & (
                    df.get("yellow_cross_down", False)
                    | df.get("dump_diamond", False)
                    | (
                        df.get("red_diamond", False)
                        & df.get("ema_ribbon_bearish", False)
                        & df.get("wt_cross_down", False)
                    )
                )
            )

            # Moderate Bullish Signals (medium confidence)
            moderate_bullish = (
                (confidence >= 25.0)
                & (confidence < 50.0)
                & (bullish_strength >= 1.0)
                & (
                    df.get("bull_candle", False)
                    | (
                        df.get("wt_cross_up", False)
                        & df.get("ema_ribbon_bullish", False)
                    )
                    | df.get("divergence_bullish", False)
                )
            )

            # Moderate Bearish Signals (medium confidence)
            moderate_bearish = (
                (confidence >= 25.0)
                & (confidence < 50.0)
                & (bearish_strength >= 1.0)
                & (
                    df.get("bear_candle", False)
                    | (
                        df.get("wt_cross_down", False)
                        & df.get("ema_ribbon_bearish", False)
                    )
                    | df.get("divergence_bearish", False)
                )
            )

            # Apply signals with priority (strong signals override moderate ones)
            signals[strong_bullish] = 1
            signals[strong_bearish] = -1

            # Apply moderate signals only where no strong signals exist
            no_strong_signal = signals == 0
            signals[moderate_bullish & no_strong_signal] = 1
            signals[moderate_bearish & no_strong_signal] = -1

            # Filter out conflicting signals (both bullish and bearish at same time)
            conflicting = (strong_bullish | moderate_bullish) & (
                strong_bearish | moderate_bearish
            )
            signals[conflicting] = 0

            logger.debug(
                f"Generated {(signals == 1).sum()} bullish and {(signals == -1).sum()} bearish signals"
            )

        except Exception as e:
            logger.error(f"Error generating enhanced signals: {str(e)}")
            signals = pd.Series(0, index=df.index, dtype=int)

        return signals

    def _add_compatibility_indicators(self, df: pd.DataFrame) -> None:
        """
        Add compatibility indicators for backward compatibility with existing code.

        Args:
            df: DataFrame to add compatibility indicators to
        """
        try:
            # Legacy EMA indicators (map to ribbon EMAs)
            if "ema2" in df.columns and "ema8" in df.columns:
                df["ema_fast"] = df["ema2"].copy()  # Map to 11-period EMA
                df["ema_slow"] = df["ema8"].copy()  # Map to 34-period EMA
                df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
                df["trend_dot"] = np.where(
                    df["ema_diff"] > 0, 1, np.where(df["ema_diff"] < 0, -1, 0)
                ).astype(int)

            # Legacy RSI indicators
            if "rsi" in df.columns:
                df["rsi_overbought"] = df["rsi"] > 70.0  # Standard RSI overbought
                df["rsi_oversold"] = df["rsi"] < 30.0  # Standard RSI oversold

        except Exception as e:
            logger.warning(f"Error adding compatibility indicators: {str(e)}")

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive latest Cipher A values with all indicators.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with all latest indicator values organized by category
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        # Core signal analysis
        core_signals = {
            "cipher_a_signal": latest.get("cipher_a_signal", 0),
            "cipher_a_bullish_strength": latest.get("cipher_a_bullish_strength", 0.0),
            "cipher_a_bearish_strength": latest.get("cipher_a_bearish_strength", 0.0),
            "cipher_a_confidence": latest.get("cipher_a_confidence", 0.0),
        }

        # WaveTrend indicators
        wavetrend_values = {
            "wt1": latest.get("wt1"),
            "wt2": latest.get("wt2"),
            "wt_overbought": latest.get("wt_overbought", False),
            "wt_oversold": latest.get("wt_oversold", False),
            "wt_cross_up": latest.get("wt_cross_up", False),
            "wt_cross_down": latest.get("wt_cross_down", False),
        }

        # Diamond pattern signals
        diamond_signals = {
            "red_diamond": latest.get("red_diamond", False),
            "green_diamond": latest.get("green_diamond", False),
            "dump_diamond": latest.get("dump_diamond", False),
            "moon_diamond": latest.get("moon_diamond", False),
        }

        # Yellow cross signals
        yellow_cross_signals = {
            "yellow_cross_up": latest.get("yellow_cross_up", False),
            "yellow_cross_down": latest.get("yellow_cross_down", False),
        }

        # Candle pattern signals
        candle_patterns = {
            "bull_candle": latest.get("bull_candle", False),
            "bear_candle": latest.get("bear_candle", False),
        }

        # EMA Ribbon values
        ema_ribbon_values = {
            "ema1": latest.get("ema1"),
            "ema2": latest.get("ema2"),
            "ema3": latest.get("ema3"),
            "ema4": latest.get("ema4"),
            "ema5": latest.get("ema5"),
            "ema6": latest.get("ema6"),
            "ema7": latest.get("ema7"),
            "ema8": latest.get("ema8"),
            "ema_ribbon_bullish": latest.get("ema_ribbon_bullish", False),
            "ema_ribbon_bearish": latest.get("ema_ribbon_bearish", False),
        }

        # Additional indicators
        additional_indicators = {
            "rsi": latest.get("rsi"),
            "rsimfi": latest.get("rsimfi"),
            "stoch_rsi_k": latest.get("stoch_rsi_k"),
            "stoch_rsi_d": latest.get("stoch_rsi_d"),
            "stc": latest.get("stc"),
        }

        # Divergence signals
        divergence_signals = {
            "divergence_bullish": latest.get("divergence_bullish", False),
            "divergence_bearish": latest.get("divergence_bearish", False),
            "wt_divergence_bullish": latest.get("wt_divergence_bullish", False),
            "wt_divergence_bearish": latest.get("wt_divergence_bearish", False),
            "rsi_divergence_bullish": latest.get("rsi_divergence_bullish", False),
            "rsi_divergence_bearish": latest.get("rsi_divergence_bearish", False),
        }

        # Legacy compatibility values
        legacy_values = {
            "ema_fast": latest.get("ema_fast"),
            "ema_slow": latest.get("ema_slow"),
            "trend_dot": latest.get("trend_dot"),
            "rsi_overbought": latest.get("rsi_overbought", False),
            "rsi_oversold": latest.get("rsi_oversold", False),
        }

        # Combine all values
        result = {
            "timestamp": latest.name if hasattr(latest, "name") else None,
            **core_signals,
            **wavetrend_values,
            **diamond_signals,
            **yellow_cross_signals,
            **candle_patterns,
            **ema_ribbon_values,
            **additional_indicators,
            **divergence_signals,
            **legacy_values,
        }

        return result

    def get_all_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive signal analysis for the latest data point.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with detailed signal analysis
        """
        if df.empty:
            return {"error": "No data available"}

        latest_values = self.get_latest_values(df)

        # Analyze signal strength
        bullish_strength = latest_values.get("cipher_a_bullish_strength", 0.0)
        bearish_strength = latest_values.get("cipher_a_bearish_strength", 0.0)
        confidence = latest_values.get("cipher_a_confidence", 0.0)
        signal = latest_values.get("cipher_a_signal", 0)

        # Generate signal interpretation
        signal_interpretation = self._interpret_signals(latest_values)

        return {
            "latest_values": latest_values,
            "signal_strength": {
                "bullish": bullish_strength,
                "bearish": bearish_strength,
                "net": bullish_strength - bearish_strength,
                "confidence": confidence,
            },
            "overall_signal": {
                "direction": (
                    "BULLISH" if signal > 0 else "BEARISH" if signal < 0 else "NEUTRAL"
                ),
                "strength": (
                    "STRONG"
                    if confidence >= 50
                    else "MODERATE" if confidence >= 25 else "WEAK"
                ),
                "value": signal,
            },
            "interpretation": signal_interpretation,
        }

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Get overall signal strength score.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Signal strength score (-100 to +100)
        """
        if df.empty:
            return 0.0

        latest = df.iloc[-1]
        bullish_strength = latest.get("cipher_a_bullish_strength", 0.0)
        bearish_strength = latest.get("cipher_a_bearish_strength", 0.0)
        confidence = latest.get("cipher_a_confidence", 0.0)

        # Calculate net strength with confidence weighting
        net_strength = bullish_strength - bearish_strength
        strength_score = net_strength * (confidence / 100.0)

        # Normalize to -100 to +100 range
        return max(-100.0, min(100.0, strength_score * 20.0))

    def interpret_signals(self, df: pd.DataFrame) -> str:
        """
        Generate human-readable interpretation of current signals.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Human-readable signal interpretation
        """
        if df.empty:
            return "No data available for signal interpretation."

        latest_values = self.get_latest_values(df)
        return self._interpret_signals(latest_values)

    def _interpret_signals(self, values: dict[str, Any]) -> str:
        """
        Internal method to interpret signal values.

        Args:
            values: Dictionary with latest indicator values

        Returns:
            Human-readable interpretation string
        """
        try:
            signal = values.get("cipher_a_signal", 0)
            confidence = values.get("cipher_a_confidence", 0.0)
            bullish_strength = values.get("cipher_a_bullish_strength", 0.0)
            bearish_strength = values.get("cipher_a_bearish_strength", 0.0)

            # Generate base interpretation
            if signal > 0:
                base = f"BULLISH signal with {confidence:.1f}% confidence"
            elif signal < 0:
                base = f"BEARISH signal with {confidence:.1f}% confidence"
            else:
                base = f"NEUTRAL - no clear signal (confidence: {confidence:.1f}%)"

            # Add specific signal details
            active_signals = []

            if values.get("yellow_cross_up"):
                active_signals.append("Yellow Cross Up (strong bullish)")
            if values.get("yellow_cross_down"):
                active_signals.append("Yellow Cross Down (strong bearish)")
            if values.get("moon_diamond"):
                active_signals.append("Moon Diamond (extreme bullish)")
            if values.get("dump_diamond"):
                active_signals.append("Dump Diamond (extreme bearish)")
            if values.get("green_diamond"):
                active_signals.append("Green Diamond (bullish)")
            if values.get("red_diamond"):
                active_signals.append("Red Diamond (bearish)")
            if values.get("bull_candle"):
                active_signals.append("Bull Candle Pattern")
            if values.get("bear_candle"):
                active_signals.append("Bear Candle Pattern")
            if values.get("divergence_bullish"):
                active_signals.append("Bullish Divergence")
            if values.get("divergence_bearish"):
                active_signals.append("Bearish Divergence")

            # Combine interpretation
            interpretation = base
            if active_signals:
                interpretation += f". Active patterns: {', '.join(active_signals)}"

            # Add strength information
            if bullish_strength > 0 or bearish_strength > 0:
                interpretation += f". Strength - Bullish: {bullish_strength:.1f}, Bearish: {bearish_strength:.1f}"

            return interpretation

        except Exception as e:
            logger.error(f"Error interpreting signals: {str(e)}")
            return "Error interpreting signals."


class CipherB:
    """
    VuManChu Cipher B Complete Implementation.

    Integrates all advanced Cipher B components:
    - WaveTrend Oscillator (Channel=9, Average=12, MA=3)
    - 8-EMA Ribbon system [5,11,15,18,21,24,28,34]
    - Advanced Signal Patterns:
      * Buy/Sell circles with exact Pine Script timing
      * Gold buy signals with RSI and fractal conditions
      * Divergence-based signals integration
      * Small circle signals for wave trend crosses
    - Stochastic RSI integration
    - Sommi Flag/Diamond patterns
    - Complete divergence detection system
    - Quality-based signal filtering and ranking

    Maintains exact Pine Script parameter compatibility.
    """

    def __init__(
        self,
        # WaveTrend parameters (Pine Script defaults for Cipher B)
        wt_channel_length: int = 9,
        wt_average_length: int = 12,
        wt_ma_length: int = 3,
        # Overbought/Oversold levels for Cipher B
        ob_level: float = 53.0,
        ob_level2: float = 60.0,
        ob_level3: float = 100.0,
        os_level: float = -53.0,
        os_level2: float = -60.0,
        os_level3: float = -75.0,
        # Divergence-specific levels
        wt_div_ob_level: float = 45.0,
        wt_div_os_level: float = -65.0,
        rsi_div_ob_level: float = 60.0,
        rsi_div_os_level: float = 30.0,
        # RSI parameters
        rsi_length: int = 14,
        # Stochastic RSI parameters
        stoch_length: int = 14,
        stoch_rsi_length: int = 14,
        stoch_k_smooth: int = 3,
        stoch_d_smooth: int = 3,
        # EMA Ribbon lengths (Pine Script defaults)
        ema_ribbon_lengths: list[int] | None = None,
        # Legacy compatibility
        vwap_length: int = 14,
        mfi_length: int = 14,
        wave_length: int = 10,
        wave_mult: float = 3.7,
    ):
        """
        Initialize Cipher B with all integrated components.

        Args:
            wt_channel_length: WaveTrend channel length (Pine Script: 9)
            wt_average_length: WaveTrend average length (Pine Script: 12)
            wt_ma_length: WaveTrend MA length (Pine Script: 3)
            ob_level: Overbought level 1 (53.0)
            ob_level2: Overbought level 2 (60.0)
            ob_level3: Overbought level 3 (100.0)
            os_level: Oversold level 1 (-53.0)
            os_level2: Oversold level 2 (-60.0)
            os_level3: Oversold level 3 (-75.0)
            wt_div_ob_level: WaveTrend divergence overbought (45.0)
            wt_div_os_level: WaveTrend divergence oversold (-65.0)
            rsi_div_ob_level: RSI divergence overbought (60.0)
            rsi_div_os_level: RSI divergence oversold (30.0)
            rsi_length: RSI calculation period (14)
            stoch_length: Stochastic length (14)
            stoch_rsi_length: Stochastic RSI length (14)
            stoch_k_smooth: Stochastic K smoothing (3)
            stoch_d_smooth: Stochastic D smoothing (3)
            ema_ribbon_lengths: EMA ribbon lengths (default: [5,11,15,18,21,24,28,34])
            vwap_length: VWAP calculation period (legacy)
            mfi_length: Money Flow Index period (legacy)
            wave_length: Wave calculation period (legacy)
            wave_mult: Wave multiplier for sensitivity (legacy)
        """
        # Initialize core components
        self.wavetrend = WaveTrend(
            channel_length=wt_channel_length,
            average_length=wt_average_length,
            ma_length=wt_ma_length,
            overbought_level=ob_level,
            oversold_level=os_level,
        )

        self.cipher_b_signals = CipherBSignals(
            wt_channel_length=wt_channel_length,
            wt_average_length=wt_average_length,
            wt_ma_length=wt_ma_length,
            ob_level=ob_level,
            ob_level2=ob_level2,
            ob_level3=ob_level3,
            os_level=os_level,
            os_level2=os_level2,
            os_level3=os_level3,
            wt_div_ob_level=wt_div_ob_level,
            wt_div_os_level=wt_div_os_level,
            rsi_div_ob_level=rsi_div_ob_level,
            rsi_div_os_level=rsi_div_os_level,
            rsi_length=rsi_length,
            stoch_length=stoch_length,
            stoch_rsi_length=stoch_rsi_length,
            stoch_k_smooth=stoch_k_smooth,
            stoch_d_smooth=stoch_d_smooth,
        )

        self.ema_ribbon = EMAribbon(
            lengths=ema_ribbon_lengths or [5, 11, 15, 18, 21, 24, 28, 34]
        )

        self.stochastic_rsi = StochasticRSI(
            rsi_length=stoch_rsi_length,
            smooth_k=stoch_k_smooth,
            smooth_d=stoch_d_smooth,
        )

        self.sommi_patterns = SommiPatterns()

        self.divergence_detector = DivergenceDetector()

        # Store parameters
        self.wt_channel_length = wt_channel_length
        self.wt_average_length = wt_average_length
        self.wt_ma_length = wt_ma_length
        self.ob_level = ob_level
        self.os_level = os_level
        self.rsi_length = rsi_length

        # Legacy parameters for compatibility
        self.vwap_length = vwap_length
        self.mfi_length = mfi_length
        self.wave_length = wave_length
        self.wave_mult = wave_mult

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate complete Cipher B indicators with all integrated components.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with comprehensive Cipher B indicators:

            Core WaveTrend:
            - wt1, wt2: WaveTrend oscillator values
            - wt_overbought, wt_oversold: OB/OS conditions
            - wt_cross_up, wt_cross_down: Basic cross signals

            Advanced Signal Patterns:
            - buy_circle, sell_circle: Buy/Sell circle signals
            - gold_buy: Gold buy signals
            - divergence_buy, divergence_sell: Divergence-based signals
            - small_circle_up, small_circle_down: Small circle signals

            8-EMA Ribbon:
            - ema1 through ema8: All ribbon EMAs
            - ema_ribbon_bullish, ema_ribbon_bearish: Ribbon direction
            - ema_cross_signals: Various EMA crossover signals

            Additional Indicators:
            - rsi: Standard RSI
            - stoch_rsi_k, stoch_rsi_d: Stochastic RSI

            Sommi Patterns:
            - sommi_flag_up, sommi_flag_down: Sommi flag patterns
            - sommi_diamond_up, sommi_diamond_down: Sommi diamonds

            Signal Analysis:
            - cipher_b_signal: Overall signal (-1, 0, 1)
            - cipher_b_strength: Signal strength score
            - cipher_b_confidence: Signal confidence (0-100)

            Legacy Indicators (for compatibility):
            - vwap: Volume Weighted Average Price
            - money_flow: Money Flow Index
            - wave: Zero-lag wave indicator
        """
        # Validate input data
        min_length = (
            max(
                self.wt_channel_length,
                self.wt_average_length,
                max(self.ema_ribbon.lengths),
                self.rsi_length,
            )
            + 20
        )  # Buffer for lookbacks

        if len(df) < min_length:
            logger.warning(
                f"Insufficient data for Cipher B calculation. Need {min_length}, got {len(df)}"
            )
            return df.copy()

        result = df.copy()

        # Ensure all input columns are proper float64 dtype
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype(
                    "float64"
                )

        try:
            # Step 1: Calculate 8-EMA Ribbon system
            logger.debug("Calculating 8-EMA Ribbon system for Cipher B")
            result = self.ema_ribbon.calculate_ema_ribbon(result)
            result = self.ema_ribbon.calculate_ribbon_direction(result)
            result = self.ema_ribbon.calculate_crossover_signals(result)

            # Map ribbon column names to expected names
            if "ribbon_bullish" in result.columns:
                result["ema_ribbon_bullish"] = result["ribbon_bullish"]
            if "ribbon_bearish" in result.columns:
                result["ema_ribbon_bearish"] = result["ribbon_bearish"]

            # Step 2: Calculate core WaveTrend oscillator
            logger.debug("Calculating WaveTrend oscillator for Cipher B")
            result = self.wavetrend.calculate(result)

            # Step 3: Calculate standard RSI
            logger.debug("Calculating RSI for Cipher B")
            rsi_values = ta.rsi(result["close"], length=self.rsi_length)
            result["rsi"] = (
                rsi_values.astype("float64")
                if rsi_values is not None
                else pd.Series(50.0, index=result.index)
            )

            # Step 4: Calculate Stochastic RSI
            logger.debug("Calculating Stochastic RSI for Cipher B")
            result = self.stochastic_rsi.calculate(result)

            # Step 5: Calculate Sommi patterns
            logger.debug("Calculating Sommi patterns")
            result = self.sommi_patterns.calculate(result)

            # Step 6: Calculate all advanced Cipher B signal patterns
            logger.debug("Calculating advanced Cipher B signal patterns")
            cipher_b_signals_data = self.cipher_b_signals.get_all_cipher_b_signals(
                result
            )

            # Extract signal data and add to result DataFrame
            result = self._integrate_cipher_b_signals(result, cipher_b_signals_data)

            # Step 7: Calculate divergence signals
            logger.debug("Calculating divergence signals for Cipher B")
            # Use the correct method names for divergence detection
            try:
                wt_regular_divs = self.divergence_detector.detect_regular_divergences(
                    result["wt2"], result["close"], 45.0, -65.0
                )
                wt_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    result["wt2"], result["close"]
                )

                rsi_regular_divs = self.divergence_detector.detect_regular_divergences(
                    result["rsi"], result["close"], 60.0, 30.0
                )
                rsi_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    result["rsi"], result["close"]
                )

                # Convert divergence signals to boolean series
                from .divergence_detector import DivergenceType

                # Initialize divergence series
                result["wt_divergence_bullish"] = pd.Series(False, index=result.index)
                result["wt_divergence_bearish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bullish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bearish"] = pd.Series(False, index=result.index)

                # Process WT divergences
                for div in wt_regular_divs + wt_hidden_divs:
                    if div.end_fractal.index < len(result):
                        if div.type in [
                            DivergenceType.REGULAR_BULLISH,
                            DivergenceType.HIDDEN_BULLISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("wt_divergence_bullish"),
                            ] = True
                        elif div.type in [
                            DivergenceType.REGULAR_BEARISH,
                            DivergenceType.HIDDEN_BEARISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("wt_divergence_bearish"),
                            ] = True

                # Process RSI divergences
                for div in rsi_regular_divs + rsi_hidden_divs:
                    if div.end_fractal.index < len(result):
                        if div.type in [
                            DivergenceType.REGULAR_BULLISH,
                            DivergenceType.HIDDEN_BULLISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("rsi_divergence_bullish"),
                            ] = True
                        elif div.type in [
                            DivergenceType.REGULAR_BEARISH,
                            DivergenceType.HIDDEN_BEARISH,
                        ]:
                            result.iloc[
                                div.end_fractal.index,
                                result.columns.get_loc("rsi_divergence_bearish"),
                            ] = True

            except Exception as div_error:
                logger.warning(f"Error calculating divergences: {div_error}")
                # Set default divergence columns
                result["wt_divergence_bullish"] = pd.Series(False, index=result.index)
                result["wt_divergence_bearish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bullish"] = pd.Series(False, index=result.index)
                result["rsi_divergence_bearish"] = pd.Series(False, index=result.index)

            # Step 8: Enhanced signal generation with all components
            result["cipher_b_signal"] = self._generate_enhanced_signals(result)

            # Step 9: Add legacy indicators for backward compatibility
            self._add_legacy_indicators(result)

            logger.debug("Cipher B calculation completed successfully")

        except Exception as e:
            logger.error(f"Error in Cipher B calculation: {str(e)}")
            # Add error indicators
            result["cipher_b_error"] = True
            result["cipher_b_signal"] = 0
            if "cipher_b_confidence" not in result.columns:
                result["cipher_b_confidence"] = 0.0

        return result

    def _integrate_cipher_b_signals(
        self, df: pd.DataFrame, signals_data: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Integrate Cipher B signals data into the main DataFrame.

        Args:
            df: Main DataFrame with indicators
            signals_data: Signal data from cipher_b_signals

        Returns:
            DataFrame with integrated signal columns
        """
        result = df.copy()

        try:
            # Initialize signal columns with False/0 defaults
            signal_columns = {
                "buy_circle": False,
                "sell_circle": False,
                "gold_buy": False,
                "divergence_buy": False,
                "divergence_sell": False,
                "small_circle_up": False,
                "small_circle_down": False,
                "cipher_b_strength": 0.0,
                "cipher_b_confidence": 0.0,
            }

            for col, default in signal_columns.items():
                result[col] = pd.Series(default, index=result.index)

            # Extract and process signal data if available
            if signals_data and "signals" in signals_data:
                signals = signals_data["signals"]

                # Process each signal and set the appropriate flags
                for signal in signals:
                    idx = signal.index
                    if idx < len(result):
                        signal_type = signal.signal_type.value

                        # Map signal types to DataFrame columns
                        if signal_type in result.columns:
                            result.loc[result.index[idx], signal_type] = True

                        # Set strength and confidence
                        result.loc[result.index[idx], "cipher_b_strength"] = max(
                            result.loc[result.index[idx], "cipher_b_strength"],
                            signal.strength.value,
                        )
                        result.loc[result.index[idx], "cipher_b_confidence"] = max(
                            result.loc[result.index[idx], "cipher_b_confidence"],
                            signal.confidence * 100,  # Convert to percentage
                        )

            # Set quality score if available
            if signals_data and "quality_score" in signals_data:
                result["cipher_b_quality"] = signals_data["quality_score"]
            else:
                result["cipher_b_quality"] = 0.0

            # Ensure all expected signal columns exist
            expected_signal_columns = {
                "buy_circle": False,
                "sell_circle": False,
                "gold_buy": False,
                "divergence_buy": False,
                "divergence_sell": False,
                "small_circle_up": False,
                "small_circle_down": False,
            }

            for col, default in expected_signal_columns.items():
                if col not in result.columns:
                    result[col] = pd.Series(default, index=result.index)

        except Exception as e:
            logger.warning(f"Error integrating Cipher B signals: {str(e)}")
            # Ensure minimum required columns exist
            default_columns = {
                "buy_circle": False,
                "sell_circle": False,
                "gold_buy": False,
                "divergence_buy": False,
                "divergence_sell": False,
                "small_circle_up": False,
                "small_circle_down": False,
                "cipher_b_strength": 0.0,
                "cipher_b_confidence": 0.0,
            }
            for col, default in default_columns.items():
                if col not in result.columns:
                    result[col] = pd.Series(default, index=result.index)

        return result

    def _generate_enhanced_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate enhanced buy/sell signals using all Cipher B components.

        Uses advanced signal patterns with weighted strength and confidence.

        Args:
            df: DataFrame with all calculated indicators

        Returns:
            Series with signals: 1 = buy, -1 = sell, 0 = hold
        """
        signals = pd.Series(0, index=df.index, dtype=int)

        try:
            # Get pre-calculated signals from cipher_b_signals if available
            if "cipher_b_signal" in df.columns:
                df["cipher_b_signal"].fillna(0)
            else:
                pd.Series(0, index=df.index)

            # Get signal strength and confidence if available
            df.get("cipher_b_strength", pd.Series(0.0, index=df.index))
            confidence = df.get("cipher_b_confidence", pd.Series(0.0, index=df.index))

            # High priority signals (strongest)
            gold_buy = df.get("gold_buy", pd.Series(False, index=df.index))
            divergence_buy = df.get("divergence_buy", pd.Series(False, index=df.index))
            divergence_sell = df.get(
                "divergence_sell", pd.Series(False, index=df.index)
            )

            # Medium priority signals
            buy_circle = df.get("buy_circle", pd.Series(False, index=df.index))
            sell_circle = df.get("sell_circle", pd.Series(False, index=df.index))

            # Low priority signals
            small_circle_up = df.get(
                "small_circle_up", pd.Series(False, index=df.index)
            )
            small_circle_down = df.get(
                "small_circle_down", pd.Series(False, index=df.index)
            )

            # Sommi pattern signals
            sommi_flag_up = df.get("sommi_flag_up", pd.Series(False, index=df.index))
            sommi_flag_down = df.get(
                "sommi_flag_down", pd.Series(False, index=df.index)
            )

            # WaveTrend and EMA signals
            wt_cross_up = df.get("wt_cross_up", pd.Series(False, index=df.index))
            wt_cross_down = df.get("wt_cross_down", pd.Series(False, index=df.index))
            ema_ribbon_bullish = df.get(
                "ema_ribbon_bullish", pd.Series(False, index=df.index)
            )
            ema_ribbon_bearish = df.get(
                "ema_ribbon_bearish", pd.Series(False, index=df.index)
            )

            # Apply high priority signals first
            high_priority_bullish = (
                gold_buy | divergence_buy | (buy_circle & (confidence >= 50.0))
            )

            high_priority_bearish = divergence_sell | (
                sell_circle & (confidence >= 50.0)
            )

            signals[high_priority_bullish] = 1
            signals[high_priority_bearish] = -1

            # Apply medium priority signals where no high priority exists
            no_high_priority = signals == 0

            medium_priority_bullish = no_high_priority & (
                buy_circle
                | (small_circle_up & wt_cross_up & ema_ribbon_bullish)
                | (sommi_flag_up & (confidence >= 25.0))
            )

            medium_priority_bearish = no_high_priority & (
                sell_circle
                | (small_circle_down & wt_cross_down & ema_ribbon_bearish)
                | (sommi_flag_down & (confidence >= 25.0))
            )

            signals[medium_priority_bullish] = 1
            signals[medium_priority_bearish] = -1

            # Apply low priority signals where no other signals exist
            no_signal = signals == 0

            low_priority_bullish = no_signal & (
                small_circle_up
                | (wt_cross_up & ema_ribbon_bullish & (confidence >= 15.0))
            )

            low_priority_bearish = no_signal & (
                small_circle_down
                | (wt_cross_down & ema_ribbon_bearish & (confidence >= 15.0))
            )

            signals[low_priority_bullish] = 1
            signals[low_priority_bearish] = -1

            # Filter out conflicting signals
            bullish_signals = (
                high_priority_bullish | medium_priority_bullish | low_priority_bullish
            )
            bearish_signals = (
                high_priority_bearish | medium_priority_bearish | low_priority_bearish
            )

            conflicting = bullish_signals & bearish_signals
            signals[conflicting] = 0

            logger.debug(
                f"Generated {(signals == 1).sum()} bullish and {(signals == -1).sum()} bearish Cipher B signals"
            )

        except Exception as e:
            logger.error(f"Error generating enhanced Cipher B signals: {str(e)}")
            signals = pd.Series(0, index=df.index, dtype=int)

        return signals

    def _add_legacy_indicators(self, df: pd.DataFrame) -> None:
        """
        Add legacy indicators for backward compatibility.

        Args:
            df: DataFrame to add legacy indicators to
        """
        try:
            # Calculate VWAP using pandas-ta
            if "volume" in df.columns and not df["volume"].isna().all():
                logger.debug("Calculating VWAP")
                vwap_values = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
                df["vwap"] = (
                    vwap_values.astype("float64")
                    if vwap_values is not None
                    else df["close"].rolling(20).mean()
                )

                # Calculate Money Flow Index (MFI) using pandas-ta
                logger.debug("Calculating MFI")
                mfi_values = ta.mfi(
                    df["high"],
                    df["low"],
                    df["close"],
                    df["volume"],
                    length=self.mfi_length,
                )
                df["money_flow"] = (
                    mfi_values.astype("float64")
                    if mfi_values is not None
                    else pd.Series(50.0, index=df.index)
                )
            else:
                # Fallback when no volume data
                logger.debug("No volume data, using price-based VWAP approximation")
                df["vwap"] = (
                    df["close"].rolling(20).mean()
                )  # Simple price average as fallback
                df["money_flow"] = pd.Series(50.0, index=df.index)  # Neutral MFI

            # Calculate zero-lag wave (legacy)
            logger.debug("Calculating wave indicator")
            wave_values = self._calculate_wave(df)
            df["wave"] = (
                wave_values.astype("float64")
                if wave_values is not None
                else pd.Series(0.0, index=df.index)
            )

        except Exception as e:
            logger.warning(f"Error adding legacy indicators: {str(e)}")
            # Ensure these columns exist even if calculation fails
            if "vwap" not in df.columns:
                df["vwap"] = (
                    df["close"].rolling(20).mean()
                    if "close" in df.columns
                    else pd.Series(dtype="float64", index=df.index)
                )
            if "money_flow" not in df.columns:
                df["money_flow"] = pd.Series(50.0, index=df.index)
            if "wave" not in df.columns:
                df["wave"] = pd.Series(0.0, index=df.index)

    def _calculate_wave(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the zero-lag wave indicator (legacy method).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with wave values
        """
        try:
            # Simplified wave calculation based on EMA differences
            ema1 = ta.ema(df["close"], length=self.wave_length)
            ema2 = ta.ema(df["close"], length=self.wave_length * 2)

            if ema1 is None or ema2 is None:
                return pd.Series(dtype=float, index=df.index)

            ema1 = ema1.astype(float)
            ema2 = ema2.astype(float)

            wave = (ema1 - ema2) * self.wave_mult

            return wave.astype(float)
        except Exception as e:
            logger.error(f"Error calculating wave indicator: {str(e)}")
            return pd.Series(dtype=float, index=df.index)

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive latest Cipher B values with all indicators.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with all latest indicator values organized by category
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        # Core signal analysis
        core_signals = {
            "cipher_b_signal": latest.get("cipher_b_signal", 0),
            "cipher_b_strength": latest.get("cipher_b_strength", 0.0),
            "cipher_b_confidence": latest.get("cipher_b_confidence", 0.0),
        }

        # WaveTrend indicators
        wavetrend_values = {
            "wt1": latest.get("wt1"),
            "wt2": latest.get("wt2"),
            "wt_overbought": latest.get("wt_overbought", False),
            "wt_oversold": latest.get("wt_oversold", False),
            "wt_cross_up": latest.get("wt_cross_up", False),
            "wt_cross_down": latest.get("wt_cross_down", False),
        }

        # Advanced signal patterns
        signal_patterns = {
            "buy_circle": latest.get("buy_circle", False),
            "sell_circle": latest.get("sell_circle", False),
            "gold_buy": latest.get("gold_buy", False),
            "divergence_buy": latest.get("divergence_buy", False),
            "divergence_sell": latest.get("divergence_sell", False),
            "small_circle_up": latest.get("small_circle_up", False),
            "small_circle_down": latest.get("small_circle_down", False),
        }

        # EMA Ribbon values
        ema_ribbon_values = {
            "ema1": latest.get("ema1"),
            "ema2": latest.get("ema2"),
            "ema3": latest.get("ema3"),
            "ema4": latest.get("ema4"),
            "ema5": latest.get("ema5"),
            "ema6": latest.get("ema6"),
            "ema7": latest.get("ema7"),
            "ema8": latest.get("ema8"),
            "ema_ribbon_bullish": latest.get("ema_ribbon_bullish", False),
            "ema_ribbon_bearish": latest.get("ema_ribbon_bearish", False),
        }

        # Additional indicators
        additional_indicators = {
            "rsi": latest.get("rsi"),
            "stoch_rsi_k": latest.get("stoch_rsi_k"),
            "stoch_rsi_d": latest.get("stoch_rsi_d"),
        }

        # Sommi patterns
        sommi_patterns = {
            "sommi_flag_up": latest.get("sommi_flag_up", False),
            "sommi_flag_down": latest.get("sommi_flag_down", False),
            "sommi_diamond_up": latest.get("sommi_diamond_up", False),
            "sommi_diamond_down": latest.get("sommi_diamond_down", False),
        }

        # Divergence signals
        divergence_signals = {
            "wt_divergence_bullish": latest.get("wt_divergence_bullish", False),
            "wt_divergence_bearish": latest.get("wt_divergence_bearish", False),
            "rsi_divergence_bullish": latest.get("rsi_divergence_bullish", False),
            "rsi_divergence_bearish": latest.get("rsi_divergence_bearish", False),
        }

        # Legacy values
        legacy_values = {
            "vwap": latest.get("vwap"),
            "money_flow": latest.get("money_flow"),
            "wave": latest.get("wave"),
        }

        # Combine all values
        result = {
            "timestamp": latest.name if hasattr(latest, "name") else None,
            **core_signals,
            **wavetrend_values,
            **signal_patterns,
            **ema_ribbon_values,
            **additional_indicators,
            **sommi_patterns,
            **divergence_signals,
            **legacy_values,
        }

        return result

    def get_all_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive signal analysis for the latest data point.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with detailed signal analysis
        """
        if df.empty:
            return {"error": "No data available"}

        latest_values = self.get_latest_values(df)

        # Analyze signal strength
        strength = latest_values.get("cipher_b_strength", 0.0)
        confidence = latest_values.get("cipher_b_confidence", 0.0)
        signal = latest_values.get("cipher_b_signal", 0)

        # Generate signal interpretation
        signal_interpretation = self._interpret_signals(latest_values)

        return {
            "latest_values": latest_values,
            "signal_strength": {
                "overall": strength,
                "confidence": confidence,
            },
            "overall_signal": {
                "direction": (
                    "BULLISH" if signal > 0 else "BEARISH" if signal < 0 else "NEUTRAL"
                ),
                "strength": (
                    "STRONG"
                    if confidence >= 50
                    else "MODERATE" if confidence >= 25 else "WEAK"
                ),
                "value": signal,
            },
            "interpretation": signal_interpretation,
        }

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Get overall signal strength score.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Signal strength score (-100 to +100)
        """
        if df.empty:
            return 0.0

        latest = df.iloc[-1]
        strength = latest.get("cipher_b_strength", 0.0)
        confidence = latest.get("cipher_b_confidence", 0.0)
        signal = latest.get("cipher_b_signal", 0)

        # Calculate strength score with confidence weighting
        base_strength = strength * signal  # Positive for bullish, negative for bearish
        strength_score = base_strength * (confidence / 100.0)

        # Normalize to -100 to +100 range
        return max(-100.0, min(100.0, strength_score * 20.0))

    def interpret_signals(self, df: pd.DataFrame) -> str:
        """
        Generate human-readable interpretation of current signals.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Human-readable signal interpretation
        """
        if df.empty:
            return "No data available for signal interpretation."

        latest_values = self.get_latest_values(df)
        return self._interpret_signals(latest_values)

    def _interpret_signals(self, values: dict[str, Any]) -> str:
        """
        Internal method to interpret signal values.

        Args:
            values: Dictionary with latest indicator values

        Returns:
            Human-readable interpretation string
        """
        try:
            signal = values.get("cipher_b_signal", 0)
            confidence = values.get("cipher_b_confidence", 0.0)
            strength = values.get("cipher_b_strength", 0.0)

            # Generate base interpretation
            if signal > 0:
                base = f"BULLISH signal with {confidence:.1f}% confidence"
            elif signal < 0:
                base = f"BEARISH signal with {confidence:.1f}% confidence"
            else:
                base = f"NEUTRAL - no clear signal (confidence: {confidence:.1f}%)"

            # Add specific signal details
            active_signals = []

            if values.get("gold_buy"):
                active_signals.append("Gold Buy (strongest bullish)")
            if values.get("buy_circle"):
                active_signals.append("Buy Circle")
            if values.get("sell_circle"):
                active_signals.append("Sell Circle")
            if values.get("divergence_buy"):
                active_signals.append("Divergence Buy")
            if values.get("divergence_sell"):
                active_signals.append("Divergence Sell")
            if values.get("small_circle_up"):
                active_signals.append("Small Circle Up")
            if values.get("small_circle_down"):
                active_signals.append("Small Circle Down")
            if values.get("sommi_flag_up"):
                active_signals.append("Sommi Flag Up")
            if values.get("sommi_flag_down"):
                active_signals.append("Sommi Flag Down")

            # Combine interpretation
            interpretation = base
            if active_signals:
                interpretation += f". Active patterns: {', '.join(active_signals)}"

            # Add strength information
            if strength > 0:
                interpretation += f". Signal strength: {strength:.1f}"

            return interpretation

        except Exception as e:
            logger.error(f"Error interpreting Cipher B signals: {str(e)}")
            return "Error interpreting signals."


class VuManChuIndicators:
    """
    Main indicator calculator that combines comprehensive Cipher A & B with utility indicators.

    Provides complete VuManChu Cipher implementation with 100% Pine Script compatibility.
    Integrates all advanced components and maintains backward compatibility.

    Enhanced with dominance candlestick analysis support for TradingView-style
    stablecoin dominance technical analysis.
    """

    def __init__(self):
        """Initialize the indicator calculator with all components."""
        self.cipher_a = CipherA()
        self.cipher_b = CipherB()

    def calculate_all(self, df: pd.DataFrame, dominance_candles=None) -> pd.DataFrame:
        """
        Calculate all indicators for the given DataFrame.

        This is the main entry point for complete VuManChu Cipher analysis.
        Calculates both Cipher A and B with all integrated components, and optionally
        processes dominance candlestick data for enhanced market sentiment analysis.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            dominance_candles: Optional list of DominanceCandleData for sentiment analysis

        Returns:
            DataFrame with comprehensive indicator suite:

            Cipher A Indicators:
            - All WaveTrend, EMA Ribbon, Signal Patterns
            - Diamond patterns, Yellow Cross, Candle patterns
            - RSI+MFI, Stochastic RSI, Schaff Trend Cycle
            - Signal strength and confidence analysis

            Cipher B Indicators:
            - All WaveTrend, EMA Ribbon, Signal Patterns
            - Buy/Sell circles, Gold signals, Divergence signals
            - Sommi patterns, Stochastic RSI
            - Quality-based signal filtering

            Utility Indicators:
            - EMA 200, Bollinger Bands, ATR
            - Combined signal analysis
            - Overall market sentiment

            Dominance Analysis (if dominance_candles provided):
            - Dominance trend indicators applied to candlestick data
            - Dominance-based market sentiment signals
            - Dominance divergence analysis vs price action
            - TradingView-style dominance technical analysis
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for indicator calculation")
            return df.copy()

        # Start with original data
        result = df.copy()

        try:
            logger.info("Starting comprehensive VuManChu Cipher calculation")

            # Calculate Cipher A with all integrated components
            logger.info("Calculating Cipher A indicators")
            result = self.cipher_a.calculate(result)

            # Calculate Cipher B with all integrated components
            logger.info("Calculating Cipher B indicators")
            result = self.cipher_b.calculate(result)

            # Add utility indicators
            logger.info("Adding utility indicators")
            result = self._add_utility_indicators(result)

            # Add combined analysis
            logger.info("Generating combined signal analysis")
            result = self._add_combined_analysis(result)

            # Process dominance candles if provided
            if dominance_candles:
                logger.info("Processing dominance candlestick analysis")
                result = self._add_dominance_analysis(result, dominance_candles)

            logger.info("VuManChu Cipher calculation completed successfully")

        except Exception as e:
            logger.error(f"Error in comprehensive indicator calculation: {str(e)}")
            # Add error indicators
            result["calculation_error"] = True

        return result

    def _add_utility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common utility indicators for market context.

        Args:
            df: DataFrame with existing data

        Returns:
            DataFrame with added utility indicators
        """
        result = df.copy()

        try:
            # Ensure all input columns are proper float64 dtype
            for col in ["open", "high", "low", "close", "volume"]:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors="coerce").astype(
                        "float64"
                    )

            # Long-term trend indicator (EMA 200)
            ema_200 = ta.ema(result["close"], length=200)
            result["ema_200"] = (
                ema_200.astype(float)
                if ema_200 is not None
                else pd.Series(dtype=float, index=df.index)
            )

            # Bollinger Bands for volatility context
            bb = ta.bbands(result["close"], length=20)
            if bb is not None:
                result = pd.concat([result, bb], axis=1)

            # Average True Range for volatility measurement
            atr_values = ta.atr(
                result["high"], result["low"], result["close"], length=14
            )
            result["atr"] = (
                atr_values.astype(float)
                if atr_values is not None
                else pd.Series(dtype=float, index=df.index)
            )

            # Volume indicators (if volume data available)
            if "volume" in result.columns:
                # Volume Moving Average
                volume_ma = ta.sma(result["volume"], length=20)
                result["volume_ma"] = (
                    volume_ma.astype(float)
                    if volume_ma is not None
                    else pd.Series(dtype=float, index=df.index)
                )

                # Volume ratio (current vs average)
                if "volume_ma" in result.columns:
                    result["volume_ratio"] = np.where(
                        result["volume_ma"] > 0,
                        result["volume"] / result["volume_ma"],
                        1.0,
                    )

        except Exception as e:
            logger.warning(f"Error adding utility indicators: {str(e)}")

        return result

    def _add_combined_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add combined analysis from both Cipher A and B signals.

        Args:
            df: DataFrame with Cipher A and B indicators

        Returns:
            DataFrame with combined analysis columns
        """
        result = df.copy()

        try:
            # Get individual signals
            cipher_a_signal = result.get(
                "cipher_a_signal", pd.Series(0, index=result.index)
            )
            cipher_b_signal = result.get(
                "cipher_b_signal", pd.Series(0, index=result.index)
            )

            # Get confidence scores
            cipher_a_confidence = result.get(
                "cipher_a_confidence", pd.Series(0.0, index=result.index)
            )
            cipher_b_confidence = result.get(
                "cipher_b_confidence", pd.Series(0.0, index=result.index)
            )

            # Calculate combined signal (weighted by confidence)
            total_confidence = cipher_a_confidence + cipher_b_confidence
            combined_signal = np.where(
                total_confidence > 0,
                (
                    cipher_a_signal * cipher_a_confidence
                    + cipher_b_signal * cipher_b_confidence
                )
                / total_confidence,
                0,
            )

            # Round to discrete signal values
            result["combined_signal"] = np.where(
                combined_signal > 0.5, 1, np.where(combined_signal < -0.5, -1, 0)
            ).astype(int)

            # Calculate overall confidence (average of both)
            result["combined_confidence"] = (
                cipher_a_confidence + cipher_b_confidence
            ) / 2.0

            # Signal agreement indicator
            result["signal_agreement"] = (
                (cipher_a_signal == cipher_b_signal) & (cipher_a_signal != 0)
            ).astype(bool)

            # Overall market sentiment
            sentiment_score = (
                result.get("cipher_a_bullish_strength", 0)
                + result.get("cipher_b_strength", 0)
                * np.where(result["cipher_b_signal"] > 0, 1, -1)
            ) / 2.0

            result["market_sentiment"] = np.where(
                sentiment_score > 1.0,
                "STRONG_BULLISH",
                np.where(
                    sentiment_score > 0.3,
                    "BULLISH",
                    np.where(
                        sentiment_score > -0.3,
                        "NEUTRAL",
                        np.where(sentiment_score > -1.0, "BEARISH", "STRONG_BEARISH"),
                    ),
                ),
            )

        except Exception as e:
            logger.warning(f"Error adding combined analysis: {str(e)}")

        return result

    def _add_dominance_analysis(
        self, df: pd.DataFrame, dominance_candles
    ) -> pd.DataFrame:
        """
        Add dominance candlestick analysis to the indicators DataFrame.

        Processes dominance candles with technical indicators similar to price action,
        providing TradingView-style analysis of stablecoin dominance patterns.

        Args:
            df: DataFrame with existing indicators
            dominance_candles: List of DominanceCandleData objects

        Returns:
            DataFrame with added dominance analysis columns
        """
        result = df.copy()

        try:
            if not dominance_candles:
                logger.debug("No dominance candles provided for analysis")
                return result

            logger.debug(f"Processing {len(dominance_candles)} dominance candles")

            # Convert dominance candles to DataFrame for analysis
            dominance_data = []
            for candle in dominance_candles:
                dominance_data.append(
                    {
                        "timestamp": candle.timestamp,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": float(candle.volume),
                        "rsi": candle.rsi,
                        "ema_fast": candle.ema_fast,
                        "ema_slow": candle.ema_slow,
                        "momentum": candle.momentum,
                        "trend_signal": candle.trend_signal,
                    }
                )

            if not dominance_data:
                logger.warning("Empty dominance data for analysis")
                return result

            dominance_df = pd.DataFrame(dominance_data)
            dominance_df.set_index("timestamp", inplace=True)

            # Apply VuManChu-style analysis to dominance data
            logger.debug("Applying Cipher A indicators to dominance candles")
            dominance_with_cipher_a = self.cipher_a.calculate(dominance_df.copy())

            logger.debug("Applying Cipher B indicators to dominance candles")
            dominance_with_cipher_b = self.cipher_b.calculate(dominance_df.copy())

            # Extract latest dominance indicator values
            if not dominance_with_cipher_a.empty and not dominance_with_cipher_b.empty:
                latest_dom_a = dominance_with_cipher_a.iloc[-1]
                latest_dom_b = dominance_with_cipher_b.iloc[-1]

                # Add dominance-specific indicators to the main DataFrame
                # These will be available for the latest values in get_latest_state
                dominance_indicators = {
                    "dominance_cipher_a_signal": latest_dom_a.get("cipher_a_signal", 0),
                    "dominance_cipher_a_confidence": latest_dom_a.get(
                        "cipher_a_confidence", 0.0
                    ),
                    "dominance_cipher_b_signal": latest_dom_b.get("cipher_b_signal", 0),
                    "dominance_cipher_b_confidence": latest_dom_b.get(
                        "cipher_b_confidence", 0.0
                    ),
                    "dominance_wt1": latest_dom_a.get("wt1"),
                    "dominance_wt2": latest_dom_a.get("wt2"),
                    "dominance_rsi": latest_dom_a.get("rsi"),
                    "dominance_ema_fast": latest_dom_a.get("ema_fast"),
                    "dominance_ema_slow": latest_dom_a.get("ema_slow"),
                    "dominance_trend": latest_dom_a.get("trend_dot", 0),
                }

                # Advanced dominance pattern signals
                dominance_patterns = {
                    "dominance_red_diamond": latest_dom_a.get("red_diamond", False),
                    "dominance_green_diamond": latest_dom_a.get("green_diamond", False),
                    "dominance_yellow_cross_up": latest_dom_a.get(
                        "yellow_cross_up", False
                    ),
                    "dominance_yellow_cross_down": latest_dom_a.get(
                        "yellow_cross_down", False
                    ),
                    "dominance_buy_circle": latest_dom_b.get("buy_circle", False),
                    "dominance_sell_circle": latest_dom_b.get("sell_circle", False),
                    "dominance_gold_buy": latest_dom_b.get("gold_buy", False),
                    "dominance_divergence_bullish": latest_dom_a.get(
                        "divergence_bullish", False
                    ),
                    "dominance_divergence_bearish": latest_dom_a.get(
                        "divergence_bearish", False
                    ),
                }

                # Calculate dominance vs price divergence
                dominance_divergence = self._calculate_dominance_price_divergence(
                    result, dominance_with_cipher_a, dominance_with_cipher_b
                )

                # Combined dominance sentiment analysis
                dominance_sentiment = self._analyze_dominance_sentiment(
                    latest_dom_a, latest_dom_b, dominance_candles
                )

                # Add all dominance indicators as the latest value in the main DataFrame
                result.index[-1] if not result.empty else 0

                # Initialize dominance columns with default values
                for col, default_val in {
                    **dominance_indicators,
                    **dominance_patterns,
                    **dominance_divergence,
                    **dominance_sentiment,
                }.items():
                    if col not in result.columns:
                        result[col] = pd.Series(
                            dtype=type(default_val), index=result.index
                        )

                # Set the latest values
                for col, val in {
                    **dominance_indicators,
                    **dominance_patterns,
                    **dominance_divergence,
                    **dominance_sentiment,
                }.items():
                    if not result.empty:
                        result.loc[result.index[-1], col] = val

                logger.debug(
                    f"Added dominance analysis with {len(dominance_indicators) + len(dominance_patterns)} indicators"
                )

        except Exception as e:
            logger.error(f"Error in dominance analysis: {str(e)}")
            # Ensure basic dominance columns exist even on error
            default_dominance_cols = {
                "dominance_cipher_a_signal": 0,
                "dominance_cipher_b_signal": 0,
                "dominance_sentiment": "NEUTRAL",
                "dominance_trend": 0,
                "dominance_price_divergence": "NONE",
            }
            for col, default_val in default_dominance_cols.items():
                if col not in result.columns:
                    result[col] = pd.Series(default_val, index=result.index)

        return result

    def _calculate_dominance_price_divergence(
        self,
        price_df: pd.DataFrame,
        dominance_cipher_a: pd.DataFrame,
        dominance_cipher_b: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate divergence between dominance indicators and price action.

        Args:
            price_df: Main price DataFrame with indicators
            dominance_cipher_a: Dominance data with Cipher A indicators
            dominance_cipher_b: Dominance data with Cipher B indicators

        Returns:
            Dictionary with divergence analysis
        """
        try:
            # Get recent price and dominance data for comparison
            if len(price_df) < 10 or len(dominance_cipher_a) < 5:
                return {"dominance_price_divergence": "INSUFFICIENT_DATA"}

            # Compare price trend vs dominance trend over recent periods
            recent_price_close = (
                price_df["close"].iloc[-10:] if "close" in price_df.columns else None
            )
            recent_dominance_close = (
                dominance_cipher_a["close"].iloc[-5:]
                if "close" in dominance_cipher_a.columns
                else None
            )

            if recent_price_close is None or recent_dominance_close is None:
                return {"dominance_price_divergence": "NO_DATA"}

            # Calculate trends using linear regression
            import numpy as np

            price_trend = np.polyfit(
                range(len(recent_price_close)), recent_price_close, 1
            )[0]
            dominance_trend = np.polyfit(
                range(len(recent_dominance_close)), recent_dominance_close, 1
            )[0]

            # Determine divergence type
            divergence_type = "NONE"
            divergence_strength = 0.0

            # Bullish divergence: Price declining, dominance declining (money leaving stables back to crypto)
            if price_trend < -0.001 and dominance_trend < -0.01:
                divergence_type = "BULLISH"
                divergence_strength = min(abs(price_trend), abs(dominance_trend)) * 1000

            # Bearish divergence: Price rising, dominance rising (money flowing to stables)
            elif price_trend > 0.001 and dominance_trend > 0.01:
                divergence_type = "BEARISH"
                divergence_strength = min(abs(price_trend), abs(dominance_trend)) * 1000

            # Hidden bullish: Price making higher lows, dominance making lower lows
            elif price_trend > 0 and dominance_trend < -0.005:
                divergence_type = "HIDDEN_BULLISH"
                divergence_strength = abs(dominance_trend) * 500

            # Hidden bearish: Price making lower highs, dominance making higher highs
            elif price_trend < 0 and dominance_trend > 0.005:
                divergence_type = "HIDDEN_BEARISH"
                divergence_strength = abs(dominance_trend) * 500

            return {
                "dominance_price_divergence": divergence_type,
                "dominance_divergence_strength": round(divergence_strength, 2),
                "price_trend": round(price_trend, 6),
                "dominance_trend": round(dominance_trend, 4),
            }

        except Exception as e:
            logger.warning(f"Error calculating dominance-price divergence: {e}")
            return {"dominance_price_divergence": "ERROR"}

    def _analyze_dominance_sentiment(
        self, cipher_a_latest: pd.Series, cipher_b_latest: pd.Series, dominance_candles
    ) -> dict[str, Any]:
        """
        Analyze overall market sentiment based on dominance patterns.

        Args:
            cipher_a_latest: Latest Cipher A values for dominance
            cipher_b_latest: Latest Cipher B values for dominance
            dominance_candles: Original dominance candle data

        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # Get dominance signals
            cipher_a_signal = cipher_a_latest.get("cipher_a_signal", 0)
            cipher_b_signal = cipher_b_latest.get("cipher_b_signal", 0)

            # Get latest dominance level
            latest_dominance = dominance_candles[-1].close if dominance_candles else 0

            # Calculate sentiment score
            sentiment_score = 0
            sentiment_factors = []

            # Cipher signals on dominance (inverted logic - rising dominance is bearish for crypto)
            if cipher_a_signal > 0:  # Bullish dominance signal = bearish for crypto
                sentiment_score -= 2
                sentiment_factors.append(
                    "Dominance Cipher A bullish (bearish for crypto)"
                )
            elif cipher_a_signal < 0:  # Bearish dominance signal = bullish for crypto
                sentiment_score += 2
                sentiment_factors.append(
                    "Dominance Cipher A bearish (bullish for crypto)"
                )

            if cipher_b_signal > 0:  # Bullish dominance signal = bearish for crypto
                sentiment_score -= 1.5
                sentiment_factors.append(
                    "Dominance Cipher B bullish (bearish for crypto)"
                )
            elif cipher_b_signal < 0:  # Bearish dominance signal = bullish for crypto
                sentiment_score += 1.5
                sentiment_factors.append(
                    "Dominance Cipher B bearish (bullish for crypto)"
                )

            # Absolute dominance level analysis
            if latest_dominance > 12:  # Very high dominance
                sentiment_score -= 1
                sentiment_factors.append("Very high stablecoin dominance (risk-off)")
            elif latest_dominance < 5:  # Very low dominance
                sentiment_score += 1
                sentiment_factors.append("Low stablecoin dominance (risk-on)")

            # Determine overall sentiment
            if sentiment_score >= 2:
                sentiment = "STRONG_BULLISH"
            elif sentiment_score >= 1:
                sentiment = "BULLISH"
            elif sentiment_score <= -2:
                sentiment = "STRONG_BEARISH"
            elif sentiment_score <= -1:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"

            return {
                "dominance_sentiment": sentiment,
                "dominance_sentiment_score": round(sentiment_score, 2),
                "dominance_sentiment_factors": sentiment_factors,
                "current_dominance_level": round(latest_dominance, 2),
            }

        except Exception as e:
            logger.warning(f"Error analyzing dominance sentiment: {e}")
            return {
                "dominance_sentiment": "NEUTRAL",
                "dominance_sentiment_score": 0.0,
                "dominance_sentiment_factors": [],
                "current_dominance_level": 0.0,
            }

    def get_latest_state(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get the comprehensive latest state of all indicators.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with latest values from all indicators organized by category
        """
        if df.empty:
            return {"error": "No data available"}

        # Get individual component values
        cipher_a_values = self.cipher_a.get_latest_values(df)
        cipher_b_values = self.cipher_b.get_latest_values(df)

        # Get latest row
        latest = df.iloc[-1]

        # Basic market data
        market_data = {
            "close": latest.get("close"),
            "volume": latest.get("volume"),
            "timestamp": latest.name if hasattr(latest, "name") else None,
        }

        # Combined analysis
        combined_analysis = {
            "combined_signal": latest.get("combined_signal", 0),
            "combined_confidence": latest.get("combined_confidence", 0.0),
            "signal_agreement": latest.get("signal_agreement", False),
            "market_sentiment": latest.get("market_sentiment", "NEUTRAL"),
        }

        # Utility indicators
        utility_indicators = {
            "ema_200": latest.get("ema_200"),
            "atr": latest.get("atr"),
            "volume_ratio": latest.get("volume_ratio", 1.0),
        }

        # Add Bollinger Bands if available
        bb_columns = [col for col in df.columns if "BB" in col]
        for col in bb_columns:
            utility_indicators[col.lower()] = latest.get(col)

        # Dominance analysis (if available)
        dominance_analysis = {}
        dominance_columns = [col for col in df.columns if col.startswith("dominance_")]
        if dominance_columns:
            for col in dominance_columns:
                # Remove 'dominance_' prefix for cleaner keys
                clean_key = col.replace("dominance_", "")
                dominance_analysis[clean_key] = latest.get(col)

        # Combine all values
        latest_state = {
            **market_data,
            **combined_analysis,
            **utility_indicators,
            "cipher_a": cipher_a_values,
            "cipher_b": cipher_b_values,
        }

        # Add dominance analysis if present
        if dominance_analysis:
            latest_state["dominance_analysis"] = dominance_analysis

        return latest_state

    def get_all_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive signal analysis from all components.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with complete signal analysis
        """
        if df.empty:
            return {"error": "No data available"}

        # Get individual signal analyses
        cipher_a_signals = self.cipher_a.get_all_signals(df)
        cipher_b_signals = self.cipher_b.get_all_signals(df)

        # Get combined state
        latest_state = self.get_latest_state(df)

        return {
            "cipher_a_analysis": cipher_a_signals,
            "cipher_b_analysis": cipher_b_signals,
            "combined_analysis": {
                "overall_signal": {
                    "direction": (
                        "BULLISH"
                        if latest_state.get("combined_signal", 0) > 0
                        else (
                            "BEARISH"
                            if latest_state.get("combined_signal", 0) < 0
                            else "NEUTRAL"
                        )
                    ),
                    "confidence": latest_state.get("combined_confidence", 0.0),
                    "agreement": latest_state.get("signal_agreement", False),
                    "sentiment": latest_state.get("market_sentiment", "NEUTRAL"),
                },
                "interpretation": self._generate_combined_interpretation(latest_state),
            },
            "latest_state": latest_state,
        }

    def get_signal_strength(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Get signal strength from all components.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with signal strengths
        """
        if df.empty:
            return {"cipher_a": 0.0, "cipher_b": 0.0, "combined": 0.0}

        cipher_a_strength = self.cipher_a.get_signal_strength(df)
        cipher_b_strength = self.cipher_b.get_signal_strength(df)

        # Combined strength (average)
        combined_strength = (cipher_a_strength + cipher_b_strength) / 2.0

        return {
            "cipher_a": cipher_a_strength,
            "cipher_b": cipher_b_strength,
            "combined": combined_strength,
        }

    def interpret_signals(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive human-readable interpretation.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Human-readable signal interpretation
        """
        if df.empty:
            return "No data available for signal interpretation."

        latest_state = self.get_latest_state(df)
        return self._generate_combined_interpretation(latest_state)

    def _generate_combined_interpretation(self, state: dict[str, Any]) -> str:
        """
        Generate combined interpretation from all signals.

        Args:
            state: Latest state dictionary

        Returns:
            Human-readable interpretation
        """
        try:
            combined_signal = state.get("combined_signal", 0)
            combined_confidence = state.get("combined_confidence", 0.0)
            signal_agreement = state.get("signal_agreement", False)
            market_sentiment = state.get("market_sentiment", "NEUTRAL")

            # Base interpretation
            if combined_signal > 0:
                base = f"COMBINED BULLISH signal with {combined_confidence:.1f}% confidence"
            elif combined_signal < 0:
                base = f"COMBINED BEARISH signal with {combined_confidence:.1f}% confidence"
            else:
                base = f"COMBINED NEUTRAL - no clear signal (confidence: {combined_confidence:.1f}%)"

            # Add agreement info
            agreement_text = (
                "Cipher A & B signals AGREE"
                if signal_agreement
                else "Cipher A & B signals DISAGREE"
            )

            # Add sentiment
            sentiment_text = f"Market sentiment: {market_sentiment}"

            # Combine all
            interpretation = f"{base}. {agreement_text}. {sentiment_text}."

            # Add specific recommendations
            if combined_confidence >= 50.0 and signal_agreement:
                if combined_signal > 0:
                    interpretation += " Strong bullish setup with high confidence - consider long positions."
                else:
                    interpretation += " Strong bearish setup with high confidence - consider short positions."
            elif combined_confidence >= 25.0:
                interpretation += " Moderate signal strength - proceed with caution."
            else:
                interpretation += " Low confidence signals - wait for better setup."

            return interpretation

        except Exception as e:
            logger.error(f"Error generating combined interpretation: {str(e)}")
            return "Error generating signal interpretation."
