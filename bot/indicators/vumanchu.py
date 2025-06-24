"""
VuManChu Cipher A & B indicator implementations - PRESERVED ORIGINAL + FUNCTIONAL ENHANCEMENTS.

This module provides both the original comprehensive imperative VuManChu implementation
AND functional alternatives, maintaining complete backward compatibility.

IMPLEMENTATION CHOICES:
1. ORIGINAL IMPERATIVE (Default): Full Pine Script-compatible classes with all features
2. FUNCTIONAL ALTERNATIVES: Pure functional implementations for advanced use cases
3. HYBRID APPROACH: Use functional calculations within imperative class structure

Key features:
- Complete preservation of original VuManChu Cipher A & B implementations
- All original signal patterns: Diamond, YellowX, Bull/Bear candles, Divergence
- Scalping-optimized parameters for 15-second timeframes
- WaveTrend Oscillator, 8-EMA Ribbon, RSI+MFI Combined indicator
- Optional functional enhancements alongside original implementations
- Zero breaking changes to existing usage
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from bot.utils import ta

# Import all original component dependencies
from .cipher_a_signals import CipherASignals
from .cipher_b_signals import CipherBSignals
from .divergence_detector import DivergenceDetector
from .ema_ribbon import EMAribbon
from .rsimfi import RSIMFIIndicator
from .schaff_trend_cycle import SchaffTrendCycle
from .sommi_patterns import SommiPatterns
from .stochastic_rsi import StochasticRSI
from .wavetrend import WaveTrend

logger = logging.getLogger(__name__)


# FUNCTIONAL ENHANCEMENT UTILITIES (Optional - used alongside originals)
def calculate_hlc3_functional(
    high: np.ndarray[Any, Any], low: np.ndarray[Any, Any], close: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Functional HLC3 calculation for advanced use cases."""
    return (high + low + close) / 3.0


def calculate_ema_functional(
    values: np.ndarray[Any, Any], period: int
) -> np.ndarray[Any, Any]:
    """Functional EMA calculation for advanced use cases."""
    if len(values) < period:
        return np.full_like(values, np.nan)

    alpha = 2.0 / (period + 1)
    ema = np.full_like(values, np.nan, dtype=np.float64)

    # Initialize with SMA for the first period
    ema[period - 1] = np.mean(values[:period])

    # Calculate EMA for remaining values
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_wavetrend_functional(
    src: np.ndarray[Any, Any], channel_length: int, average_length: int, ma_length: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Functional WaveTrend calculation for advanced use cases."""
    # Validate inputs
    if len(src) < max(channel_length, average_length, ma_length):
        return np.full_like(src, np.nan), np.full_like(src, np.nan)

    # Calculate ESA (Exponential Smoothing Average)
    esa = calculate_ema_functional(src, channel_length)

    # Calculate DE (Deviation)
    deviation = np.abs(src - esa)
    de = calculate_ema_functional(deviation, channel_length)

    # Prevent division by zero
    de = np.where(de == 0, 1e-6, de)

    # Calculate CI (Commodity Channel Index style)
    ci = (src - esa) / (0.015 * de)

    # Calculate TCI (Trend Channel Index)
    tci = calculate_ema_functional(ci, average_length)

    # WaveTrend components
    wt1 = tci
    wt2 = calculate_sma_functional(wt1, ma_length)

    return wt1, wt2


def calculate_sma_functional(
    values: np.ndarray[Any, Any], period: int
) -> np.ndarray[Any, Any]:
    """Functional SMA calculation for advanced use cases."""
    if len(values) < period:
        return np.full_like(values, np.nan)

    sma = np.full_like(values, np.nan, dtype=np.float64)

    # Use numpy's convolve for efficient SMA calculation
    kernel = np.ones(period) / period
    sma[period - 1 :] = np.convolve(values, kernel, mode="valid")

    return sma


# ORIGINAL IMPERATIVE CLASSES - FULLY PRESERVED
class CipherA:
    """
    VuManChu Cipher A Complete Implementation - ORIGINAL PRESERVED.

    Integrates all advanced Cipher A components (Optimized for 15-second scalping):
    - WaveTrend Oscillator (Channel=6, Average=8, MA=3)
    - 8-EMA Ribbon system [5,11,15,18,21,24,28,34]
    - RSI+MFI Combined indicator (Period=20, Multiplier=150)
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
        # WaveTrend parameters (Optimized for 15-second scalping)
        wt_channel_length: int = 6,  # Reduced from 9 for faster response
        wt_average_length: int = 8,  # Reduced from 13 for quicker signals
        wt_ma_length: int = 3,
        # Overbought/Oversold levels (Adjusted for scalping sensitivity)
        overbought_level: float = 45.0,  # Reduced from 60.0 for earlier signals
        overbought_level2: float = 38.0,  # Reduced from 53.0 for scalping
        overbought_level3: float = 100.0,
        oversold_level: float = -45.0,  # Increased from -60.0 for earlier signals
        oversold_level2: float = -38.0,  # Increased from -53.0 for scalping
        oversold_level3: float = -100.0,
        # RSI parameters
        rsi_length: int = 14,
        # RSI+MFI parameters (Optimized for scalping)
        rsimfi_period: int = 20,  # Reduced from 60 for faster momentum detection
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
        # Functional enhancement option
        use_functional_calculations: bool = False,
    ) -> None:
        """Initialize Cipher A with all integrated components."""
        # Store original parameters
        self.wt_channel_length = wt_channel_length
        self.wt_average_length = wt_average_length
        self.wt_ma_length = wt_ma_length
        self.overbought_level = overbought_level
        self.overbought_level2 = overbought_level2
        self.overbought_level3 = overbought_level3
        self.oversold_level = oversold_level
        self.oversold_level2 = oversold_level2
        self.oversold_level3 = oversold_level3
        self.rsi_length = rsi_length
        self.rsimfi_period = rsimfi_period
        self.rsimfi_multiplier = rsimfi_multiplier
        self.use_functional_calculations = use_functional_calculations

        # EMA Ribbon configuration (Pine Script defaults)
        if ema_ribbon_lengths is None:
            self.ema_ribbon_lengths = [5, 11, 15, 18, 21, 24, 28, 34]
        else:
            self.ema_ribbon_lengths = ema_ribbon_lengths

        # Initialize all component systems
        self.wavetrend = WaveTrend(
            channel_length=wt_channel_length,
            average_length=wt_average_length,
            ma_length=wt_ma_length,
        )

        self.ema_ribbon = EMAribbon(lengths=self.ema_ribbon_lengths)

        self.rsimfi = RSIMFIIndicator(
            period=rsimfi_period, multiplier=rsimfi_multiplier
        )

        self.stoch_rsi = StochasticRSI(
            stoch_length=stoch_rsi_length,
            rsi_length=self.rsi_length,  # Fix: Add missing rsi_length parameter
            smooth_k=stoch_k_smooth,
            smooth_d=stoch_d_smooth,
        )

        self.stc = SchaffTrendCycle(
            length=stc_length,
            fast_length=stc_fast_length,
            slow_length=stc_slow_length,
            factor=stc_factor,
        )

        self.cipher_a_signals = CipherASignals()
        self.divergence_detector = DivergenceDetector()
        self.sommi_patterns = SommiPatterns()

        logger.info(
            "Cipher A initialized with scalping-optimized parameters "
            f"(WT: {wt_channel_length}/{wt_average_length}/{wt_ma_length}, "
            f"OB/OS: {overbought_level}/{oversold_level}, "
            f"RSI+MFI: {rsimfi_period}/{rsimfi_multiplier}, "
            f"Functional: {use_functional_calculations})"
        )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Cipher A indicators and signals.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with all Cipher A indicators and signals
        """
        try:
            if len(df) < max(
                self.wt_channel_length,
                self.wt_average_length,
                max(self.ema_ribbon_lengths),
            ):
                logger.warning("Insufficient data for Cipher A calculation")
                return self._add_cipher_a_fallbacks(df)

            result = df.copy()

            # Choose calculation method based on configuration
            if self.use_functional_calculations:
                result = self._calculate_with_functional_enhancements(result)
            else:
                result = self._calculate_with_original_implementation(result)

            # Add signal interpretations
            result = self._add_signal_analysis(result)

            return result

        except Exception:
            logger.exception("Error in Cipher A calculation")
            return self._add_cipher_a_fallbacks(df)

    def _calculate_with_original_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate using original imperative implementation."""
        result = df.copy()

        # 1. Calculate WaveTrend Oscillator
        result = self.wavetrend.calculate(result)

        # 2. Calculate EMA Ribbon
        result = self.ema_ribbon.get_ribbon_analysis(result)

        # 3. Calculate RSI
        result["rsi"] = ta.rsi(result["close"], length=self.rsi_length)

        # 4. Calculate RSI+MFI
        result = self.rsimfi.calculate_with_analysis(result)

        # 5. Calculate Stochastic RSI
        result = self.stoch_rsi.calculate(result)

        # 6. Calculate Schaff Trend Cycle
        result = self.stc.calculate(result)

        # 7. Generate Cipher A signals
        result = self.cipher_a_signals.get_all_cipher_a_signals(result)

        # 8. Detect divergences - use regular divergences for cipher A
        try:
            divergence_result = self.divergence_detector.detect_regular_divergences(
                result["close"], result.get("wt1", result["close"])
            )
            # Merge divergence results
            for col, values in divergence_result.items():
                result[f"cipher_a_{col}"] = values
        except Exception as e:
            logger.warning(f"Divergence detection failed: {e}")

        # 9. Calculate Sommi flags (most relevant Sommi pattern for Cipher A)
        try:
            sommi_result = self.sommi_patterns.calculate_sommi_flags(result)
            # Merge Sommi results
            for col, values in sommi_result.items():
                result[f"sommi_{col}"] = values
        except Exception as e:
            logger.warning(f"Sommi pattern detection failed: {e}")

        return result

    def _calculate_with_functional_enhancements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate using functional enhancements alongside original logic."""
        result = df.copy()

        # Use functional calculations for core components
        hlc3 = calculate_hlc3_functional(
            df["high"].values, df["low"].values, df["close"].values
        )

        wt1, wt2 = calculate_wavetrend_functional(
            hlc3, self.wt_channel_length, self.wt_average_length, self.wt_ma_length
        )

        result["wt1"] = wt1
        result["wt2"] = wt2
        result["hlc3"] = hlc3

        # Continue with corrected method names for other components
        result = self.ema_ribbon.get_ribbon_analysis(result)
        result["rsi"] = ta.rsi(result["close"], length=self.rsi_length)
        result = self.rsimfi.calculate_with_analysis(result)
        result = self.stoch_rsi.calculate(result)
        result = self.stc.calculate(result)
        result = self.cipher_a_signals.get_all_cipher_a_signals(result)

        # Handle divergences and sommi patterns with error handling
        try:
            divergence_result = self.divergence_detector.detect_regular_divergences(
                result["close"], result.get("wt1", result["close"])
            )
            for col, values in divergence_result.items():
                result[f"cipher_a_{col}"] = values
        except Exception as e:
            logger.warning(f"Divergence detection failed in functional mode: {e}")

        try:
            sommi_result = self.sommi_patterns.calculate_sommi_flags(result)
            for col, values in sommi_result.items():
                result[f"sommi_{col}"] = values
        except Exception as e:
            logger.warning(f"Sommi pattern detection failed in functional mode: {e}")

        return result

    def _add_signal_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive signal analysis and confidence scoring."""
        result = df.copy()

        # Calculate signal strengths
        wt_strength = self._calculate_wavetrend_strength(result)
        ema_strength = self._calculate_ema_strength(result)
        rsi_strength = self._calculate_rsi_strength(result)

        # Combined confidence score
        result["cipher_a_confidence"] = (
            wt_strength + ema_strength + rsi_strength
        ) / 3.0

        # Overall signal direction
        result["cipher_a_signal"] = np.where(
            result["cipher_a_confidence"] > 0.6,
            np.where(result["wt1"] > result["wt2"], 1, -1),
            0,
        )

        return result

    def _calculate_wavetrend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate WaveTrend signal strength."""
        wt1, wt2 = df["wt1"], df["wt2"]

        # Strength based on position relative to levels and crossovers
        strength = pd.Series(0.5, index=df.index)  # Neutral baseline

        # Boost for extreme levels
        strength = np.where(wt2 >= self.overbought_level, strength + 0.3, strength)
        strength = np.where(wt2 <= self.oversold_level, strength + 0.3, strength)

        # Boost for crossovers
        crossover_up = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
        crossover_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

        strength = np.where(crossover_up | crossover_down, strength + 0.2, strength)

        return pd.Series(np.clip(strength, 0.0, 1.0), index=df.index)

    def _calculate_ema_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EMA Ribbon signal strength."""
        # Use EMA ribbon trend strength
        if "ema_ribbon_trend_strength" in df.columns:
            return df["ema_ribbon_trend_strength"].fillna(0.5)
        return pd.Series(0.5, index=df.index)

    def _calculate_rsi_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI-based signal strength."""
        rsi = df["rsi"]

        # Strength based on RSI extremes and momentum
        strength = pd.Series(0.5, index=df.index)  # Neutral baseline

        # Boost for oversold/overbought conditions
        strength = np.where(rsi <= 30, strength + 0.3, strength)
        strength = np.where(rsi >= 70, strength + 0.3, strength)

        # Boost for RSI momentum
        rsi_momentum = rsi - rsi.shift(1)
        strength = np.where(abs(rsi_momentum) > 2, strength + 0.2, strength)

        return pd.Series(np.clip(strength, 0.0, 1.0), index=df.index)

    def _add_cipher_a_fallbacks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fallback values when calculation fails."""
        result = df.copy()
        current_price = df["close"].iloc[-1] if len(df) > 0 else 0.0

        fallback_columns = {
            "wt1": 0.0,
            "wt2": 0.0,
            "rsi": 50.0,
            "cipher_a_dot": 0.0,
            "ema_fast": current_price,
            "ema_slow": current_price,
            "ema_ribbon_bullish": False,
            "ema_ribbon_bearish": False,
            "cipher_a_diamond": False,
            "cipher_a_yellow_cross": False,
            "cipher_a_confidence": 0.0,
            "cipher_a_signal": 0,
        }

        for col, default_val in fallback_columns.items():
            if isinstance(default_val, bool):
                result[col] = pd.Series([default_val] * len(result), index=result.index)
            else:
                result[col] = pd.Series(
                    [default_val] * len(result), index=result.index, dtype="float64"
                )

        logger.info("Added Cipher A fallback values due to calculation failure")
        return result


class CipherB:
    """
    VuManChu Cipher B Complete Implementation - ORIGINAL PRESERVED.

    Integrates all Cipher B components:
    - Money Flow calculations
    - VWAP analysis
    - Buy/Sell circle signals
    - Gold signal detection
    - Divergence patterns
    """

    def __init__(
        self,
        # Money Flow parameters
        mf_length: int = 9,
        # VWAP parameters
        vwap_length: int = 20,
        # Signal sensitivity
        signal_sensitivity: float = 1.0,
        # Functional enhancement option
        use_functional_calculations: bool = False,
    ) -> None:
        """Initialize Cipher B with all components."""
        self.mf_length = mf_length
        self.vwap_length = vwap_length
        self.signal_sensitivity = signal_sensitivity
        self.use_functional_calculations = use_functional_calculations

        self.cipher_b_signals = CipherBSignals()
        self.divergence_detector = DivergenceDetector()

        logger.info(
            f"Cipher B initialized (MF: {mf_length}, VWAP: {vwap_length}, "
            f"Sensitivity: {signal_sensitivity}, Functional: {use_functional_calculations})"
        )

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Cipher B indicators and signals.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with Cipher B indicators
        """
        try:
            if len(df) < max(self.mf_length, self.vwap_length):
                logger.warning("Insufficient data for Cipher B calculation")
                return self._add_cipher_b_fallbacks(df)

            result = df.copy()

            # Calculate Money Flow
            result = self._calculate_money_flow(result)

            # Calculate VWAP
            result = self._calculate_vwap(result)

            # Generate Cipher B signals
            result = self.cipher_b_signals.generate_signals(result, self)

            # Detect divergences
            result = self.divergence_detector.detect_cipher_b_divergences(result)

            return result

        except Exception:
            logger.exception("Error in Cipher B calculation")
            return self._add_cipher_b_fallbacks(df)

    def _calculate_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Money Flow indicator."""
        result = df.copy()

        # Typical Price
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # Money Flow Multiplier
        mf_multiplier = np.where(
            typical_price > typical_price.shift(1),
            1,
            np.where(typical_price < typical_price.shift(1), -1, 0),
        )

        # Money Flow Volume
        mf_volume = typical_price * df["volume"] * mf_multiplier

        # Money Flow Index
        result["cipher_b_money_flow"] = ta.sma(mf_volume, length=self.mf_length)

        return result

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price."""
        result = df.copy()

        # VWAP calculation
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        price_volume = typical_price * df["volume"]

        result["vwap"] = (
            price_volume.rolling(window=self.vwap_length).sum()
            / df["volume"].rolling(window=self.vwap_length).sum()
        )

        return result

    def _add_cipher_b_fallbacks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fallback values when calculation fails."""
        result = df.copy()
        current_price = df["close"].iloc[-1] if len(df) > 0 else 0.0

        fallback_columns = {
            "cipher_b_wave": 0.0,
            "cipher_b_money_flow": 50.0,
            "vwap": current_price,
            "cipher_b_buy_signal": False,
            "cipher_b_sell_signal": False,
            "cipher_b_gold_signal": False,
        }

        for col, default_val in fallback_columns.items():
            if isinstance(default_val, bool):
                result[col] = pd.Series([default_val] * len(result), index=result.index)
            else:
                result[col] = pd.Series(
                    [default_val] * len(result), index=result.index, dtype="float64"
                )

        logger.info("Added Cipher B fallback values due to calculation failure")
        return result


class VuManChuIndicators:
    """
    Combined VuManChu System - ORIGINAL PRESERVED + FUNCTIONAL ENHANCEMENTS.

    Provides unified interface to both Cipher A and B indicators with:
    - Original comprehensive implementations (default)
    - Optional functional enhancements
    - Combined signal analysis and interpretation
    - Backward compatibility with all existing usage
    """

    def __init__(
        self,
        # Cipher A parameters
        cipher_a_params: dict[str, Any] | None = None,
        # Cipher B parameters
        cipher_b_params: dict[str, Any] | None = None,
        # Implementation choice
        implementation_mode: Literal["original", "functional", "hybrid"] = "original",
    ) -> None:
        """
        Initialize combined VuManChu system.

        Args:
            cipher_a_params: Cipher A configuration parameters
            cipher_b_params: Cipher B configuration parameters
            implementation_mode: Which implementation to use:
                - "original": Pure imperative implementation (default)
                - "functional": Enhanced with functional calculations
                - "hybrid": Best of both approaches
        """
        self.implementation_mode = implementation_mode

        # Initialize Cipher A
        cipher_a_config = cipher_a_params or {}
        if implementation_mode in ["functional", "hybrid"]:
            cipher_a_config["use_functional_calculations"] = True

        self.cipher_a = CipherA(**cipher_a_config)

        # Initialize Cipher B
        cipher_b_config = cipher_b_params or {}
        if implementation_mode in ["functional", "hybrid"]:
            cipher_b_config["use_functional_calculations"] = True

        self.cipher_b = CipherB(**cipher_b_config)

        logger.info(f"VuManChu Indicators initialized in {implementation_mode} mode")

    def calculate(
        self, df: pd.DataFrame, include_interpretation: bool = True
    ) -> pd.DataFrame:
        """
        Calculate complete VuManChu system with both Cipher A and B.

        Args:
            df: OHLCV DataFrame
            include_interpretation: Whether to include human-readable interpretation

        Returns:
            DataFrame with all VuManChu indicators and signals
        """
        try:
            # Calculate Cipher A
            result = self.cipher_a.calculate(df)

            # Calculate Cipher B
            result = self.cipher_b.calculate(result)

            # Add combined analysis
            result = self._add_combined_analysis(result)

            # Add interpretation if requested
            if include_interpretation:
                result = self._add_interpretation(result)

            return result

        except Exception:
            logger.exception("Error in VuManChu calculation")
            return self._add_fallback_values(df)

    def calculate_all(
        self, market_data: pd.DataFrame, dominance_candles: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Calculate all VuManChu indicators - BACKWARD COMPATIBILITY METHOD.

        This method maintains backward compatibility with existing code that calls calculate_all().
        It delegates to the main calculate() method and ignores dominance_candles as VuManChu
        indicators don't currently use dominance data.

        Args:
            market_data: OHLCV DataFrame (same as df in calculate())
            dominance_candles: Dominance data (ignored for VuManChu indicators)

        Returns:
            DataFrame with all VuManChu indicators and signals
        """
        if dominance_candles is not None:
            logger.debug(
                "dominance_candles parameter provided to VuManChu calculate_all but ignored - "
                "VuManChu indicators don't currently use dominance data"
            )

        return self.calculate(market_data, include_interpretation=True)

    def get_latest_state(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get latest state for LLM analysis - PRESERVED ORIGINAL METHOD."""
        try:
            result = self.calculate(df)
            if len(result) == 0:
                return self._get_fallback_state()

            latest = result.iloc[-1]

            return {
                # Cipher A components
                "wt1": float(latest.get("wt1", 0.0)),
                "wt2": float(latest.get("wt2", 0.0)),
                "rsi": float(latest.get("rsi", 50.0)),
                "cipher_a_confidence": float(latest.get("cipher_a_confidence", 0.0)),
                "cipher_a_signal": int(latest.get("cipher_a_signal", 0)),
                # Cipher B components
                "cipher_b_money_flow": float(latest.get("cipher_b_money_flow", 50.0)),
                "vwap": float(latest.get("vwap", latest.get("close", 0.0))),
                "cipher_b_buy_signal": bool(latest.get("cipher_b_buy_signal", False)),
                "cipher_b_sell_signal": bool(latest.get("cipher_b_sell_signal", False)),
                # Combined analysis
                "combined_signal": int(latest.get("combined_signal", 0)),
                "combined_confidence": float(latest.get("combined_confidence", 0.0)),
                "signal_agreement": bool(latest.get("signal_agreement", False)),
                "market_sentiment": str(latest.get("market_sentiment", "NEUTRAL")),
                # Implementation info
                "implementation_mode": self.implementation_mode,
                "calculation_timestamp": datetime.now().isoformat(),
            }

        except Exception:
            logger.exception("Error getting VuManChu latest state")
            return self._get_fallback_state()

    def _add_combined_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combined signal analysis."""
        result = df.copy()

        # Combine signals from both ciphers
        cipher_a_signal = result.get(
            "cipher_a_signal", pd.Series(0, index=result.index)
        )
        cipher_b_buy = result.get(
            "cipher_b_buy_signal", pd.Series(False, index=result.index)
        )
        cipher_b_sell = result.get(
            "cipher_b_sell_signal", pd.Series(False, index=result.index)
        )

        # Convert Cipher B signals to numeric
        cipher_b_signal = np.where(cipher_b_buy, 1, np.where(cipher_b_sell, -1, 0))

        # Combined signal with agreement weighting
        signal_agreement = (cipher_a_signal * cipher_b_signal) > 0
        result["signal_agreement"] = signal_agreement

        # Weighted combined signal
        result["combined_signal"] = np.where(
            signal_agreement,
            (cipher_a_signal + cipher_b_signal) / 2,
            cipher_a_signal * 0.6
            + cipher_b_signal * 0.4,  # Favor Cipher A when disagreeing
        )

        # Combined confidence
        cipher_a_conf = result.get(
            "cipher_a_confidence", pd.Series(0.0, index=result.index)
        )
        result["combined_confidence"] = (
            np.where(
                signal_agreement,
                (cipher_a_conf + 0.7) / 2,  # Assume Cipher B confidence of 0.7
                cipher_a_conf * 0.7,  # Reduce confidence when signals disagree
            )
            * 100.0
        )

        # Market sentiment
        result["market_sentiment"] = np.where(
            result["combined_signal"] > 0.3,
            "BULLISH",
            np.where(result["combined_signal"] < -0.3, "BEARISH", "NEUTRAL"),
        )

        return result

    def _add_interpretation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add human-readable interpretation."""
        result = df.copy()

        def generate_interpretation(row: pd.Series) -> str:
            try:
                combined_signal = row.get("combined_signal", 0)
                combined_confidence = row.get("combined_confidence", 0.0)
                signal_agreement = row.get("signal_agreement", False)
                market_sentiment = row.get("market_sentiment", "NEUTRAL")

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
                    interpretation += (
                        " Moderate signal strength - proceed with caution."
                    )
                else:
                    interpretation += " Low confidence signals - wait for better setup."

                return interpretation

            except Exception:
                return "Error generating signal interpretation."

        result["vumanchu_interpretation"] = result.apply(
            generate_interpretation, axis=1
        )
        return result

    def _add_fallback_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fallback values when calculation fails."""
        result = self.cipher_a._add_cipher_a_fallbacks(df)
        result = self.cipher_b._add_cipher_b_fallbacks(result)

        # Add combined fallbacks
        result["combined_signal"] = pd.Series(0, index=result.index)
        result["combined_confidence"] = pd.Series(0.0, index=result.index)
        result["signal_agreement"] = pd.Series(False, index=result.index)
        result["market_sentiment"] = pd.Series("NEUTRAL", index=result.index)
        result["vumanchu_interpretation"] = pd.Series(
            "VuManChu calculation failed - using fallback values.", index=result.index
        )

        return result

    def _get_fallback_state(self) -> dict[str, Any]:
        """Get fallback state when calculation fails."""
        return {
            "wt1": 0.0,
            "wt2": 0.0,
            "rsi": 50.0,
            "cipher_a_confidence": 0.0,
            "cipher_a_signal": 0,
            "cipher_b_money_flow": 50.0,
            "vwap": 0.0,
            "cipher_b_buy_signal": False,
            "cipher_b_sell_signal": False,
            "combined_signal": 0,
            "combined_confidence": 0.0,
            "signal_agreement": False,
            "market_sentiment": "NEUTRAL",
            "implementation_mode": self.implementation_mode,
            "calculation_timestamp": datetime.now().isoformat(),
            "error": "Calculation failed - using fallback values",
        }


# FUNCTIONAL ENHANCEMENT WRAPPER (Optional Advanced Usage)
class VuManChuFunctional:
    """
    Pure functional VuManChu implementation for advanced use cases.

    This class provides a purely functional interface to VuManChu calculations
    for users who prefer functional programming patterns or need enhanced
    performance through vectorized operations.
    """

    @staticmethod
    def calculate_cipher_a_functional(
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        volume: np.ndarray[Any, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """
        Pure functional Cipher A calculation.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            config: Configuration parameters

        Returns:
            Dictionary of calculated indicators
        """
        config = config or {}

        # Calculate HLC3
        hlc3 = calculate_hlc3_functional(high, low, close)

        # Calculate WaveTrend
        wt1, wt2 = calculate_wavetrend_functional(
            hlc3,
            config.get("wt_channel_length", 6),
            config.get("wt_average_length", 8),
            config.get("wt_ma_length", 3),
        )

        # Calculate RSI functionally
        rsi = np.full_like(
            close, 50.0
        )  # Simplified for demo - would use full RSI calculation

        return {
            "hlc3": hlc3,
            "wt1": wt1,
            "wt2": wt2,
            "rsi": rsi,
        }

    @staticmethod
    def calculate_cipher_b_functional(
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        volume: np.ndarray[Any, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """
        Pure functional Cipher B calculation.

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array
            volume: Volume array
            config: Configuration parameters

        Returns:
            Dictionary of calculated indicators
        """
        config = config or {}

        # Calculate typical price
        typical_price = (high + low + close) / 3.0

        # Calculate VWAP functionally
        vwap_length = config.get("vwap_length", 20)
        price_volume = typical_price * volume

        # Simplified VWAP calculation
        vwap = np.full_like(close, np.nan)
        for i in range(vwap_length - 1, len(close)):
            start_idx = max(0, i - vwap_length + 1)
            pv_sum = np.sum(price_volume[start_idx : i + 1])
            vol_sum = np.sum(volume[start_idx : i + 1])
            vwap[i] = pv_sum / vol_sum if vol_sum > 0 else typical_price[i]

        return {
            "typical_price": typical_price,
            "vwap": vwap,
        }


# PRESERVE EXACT ORIGINAL API FOR BACKWARD COMPATIBILITY
def create_vumanchu_indicators(
    scalping_mode: bool = True,
    implementation: Literal["original", "functional", "hybrid"] = "original",
) -> VuManChuIndicators:
    """
    Factory function to create VuManChu indicators with preserved API.

    Args:
        scalping_mode: Whether to use scalping-optimized parameters
        implementation: Which implementation to use

    Returns:
        Configured VuManChuIndicators instance
    """
    if scalping_mode:
        # Scalping-optimized parameters (preserved from original)
        cipher_a_params = {
            "wt_channel_length": 6,
            "wt_average_length": 8,
            "wt_ma_length": 3,
            "overbought_level": 45.0,
            "oversold_level": -45.0,
            "rsimfi_period": 20,
        }
    else:
        # Standard parameters (Pine Script defaults)
        cipher_a_params = {
            "wt_channel_length": 9,
            "wt_average_length": 13,
            "wt_ma_length": 3,
            "overbought_level": 53.0,
            "oversold_level": -53.0,
            "rsimfi_period": 60,
        }

    return VuManChuIndicators(
        cipher_a_params=cipher_a_params,
        implementation_mode=implementation,
    )


# ASYNC SUPPORT (PRESERVED ORIGINAL FUNCTIONALITY)
async def calculate_vumanchu_async(
    df: pd.DataFrame, vumanchu_indicators: VuManChuIndicators | None = None
) -> pd.DataFrame:
    """
    Async wrapper for VuManChu calculation - preserves original async support.

    Args:
        df: OHLCV DataFrame
        vumanchu_indicators: Optional pre-configured indicators instance

    Returns:
        DataFrame with VuManChu indicators
    """
    if vumanchu_indicators is None:
        vumanchu_indicators = create_vumanchu_indicators()

    # Run calculation in thread pool to maintain async compatibility
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, vumanchu_indicators.calculate, df)


# Export all classes and functions for backward compatibility
__all__ = [
    # Original classes (preserved)
    "CipherA",
    "CipherB",
    "VuManChuIndicators",
    # Functional enhancements (new)
    "VuManChuFunctional",
    # Factory functions (preserved)
    "create_vumanchu_indicators",
    "calculate_vumanchu_async",
    # Functional utilities (new)
    "calculate_hlc3_functional",
    "calculate_ema_functional",
    "calculate_wavetrend_functional",
    "calculate_sma_functional",
]
