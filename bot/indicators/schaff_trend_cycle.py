"""
Schaff Trend Cycle (STC) indicator implementation.

This module provides a Python implementation of the Schaff Trend Cycle indicator,
originally developed by Doug Schaff. The STC is a cyclical oscillator that combines
the concepts of slow stochastics and the MACD to provide early signals of trend changes.

The implementation follows the Pine Script reference with proper handling of recursive
smoothing calculations and state management for historical value dependencies.
"""

import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from bot.utils import ta

logger = logging.getLogger(__name__)


class SchaffTrendCycle:
    """
    Schaff Trend Cycle implementation.

    The STC indicator combines slow stochastics with MACD-like calculations to create
    a cyclical oscillator that can identify trend changes with less lag than traditional
    moving average-based indicators.

    Features:
    - Exact implementation of Pine Script algorithm
    - Proper handling of recursive smoothing calculations
    - State management for historical value dependencies
    - Signal generation capabilities
    - Performance optimized for large datasets
    """

    def __init__(
        self,
        length: int = 10,
        fast_length: int = 23,
        slow_length: int = 50,
        factor: float = 0.5,
        overbought: float = 75.0,
        oversold: float = 25.0,
        gap_threshold_multiplier: float | None = None,
    ):
        """
        Initialize Schaff Trend Cycle parameters.

        Args:
            length: Period for stochastic calculations
            fast_length: Fast EMA period for MACD calculation
            slow_length: Slow EMA period for MACD calculation
            factor: Smoothing factor for recursive calculations (tcfactor in Pine Script)
            overbought: Overbought threshold level
            oversold: Oversold threshold level
            gap_threshold_multiplier: Multiplier for time gap detection threshold.
                None (default) uses auto-detection based on interval:
                - High frequency (≤60s): 10x multiplier
                - Medium frequency (≤300s): 5x multiplier
                - Low frequency (>300s): 3x multiplier
        """
        self.length = length
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.factor = factor
        self.overbought = overbought
        self.oversold = oversold
        self.gap_threshold_multiplier = gap_threshold_multiplier

        # State variables for recursive calculations
        self._prev_delta: float | None = None
        self._prev_stc: float | None = None

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._last_data_quality_check = None

        logger.info(
            "Schaff Trend Cycle indicator initialized",
            extra={
                "indicator": "schaff_trend_cycle",
                "parameters": {
                    "length": length,
                    "fast_length": fast_length,
                    "slow_length": slow_length,
                    "factor": factor,
                    "overbought": overbought,
                    "oversold": oversold,
                    "gap_threshold_multiplier": gap_threshold_multiplier,
                },
            },
        )

    def calculate_stc(
        self,
        src: pd.Series,
        length: int | None = None,
        fast_length: int | None = None,
        slow_length: int | None = None,
        factor: float | None = None,
    ) -> pd.Series:
        """
        Calculate STC values following the Pine Script algorithm exactly.

        Args:
            src: Source price series (typically close prices)
            length: Override default length parameter
            fast_length: Override default fast_length parameter
            slow_length: Override default slow_length parameter
            factor: Override default factor parameter

        Returns:
            Series with STC values (0-100 range)
        """
        # Use instance defaults if not provided
        length = length or self.length
        fast_length = fast_length or self.fast_length
        slow_length = slow_length or self.slow_length
        factor = factor or self.factor

        start_time = time.perf_counter()
        tracemalloc.start()

        logger.debug(
            "Starting Schaff Trend Cycle calculation",
            extra={
                "indicator": "schaff_trend_cycle",
                "step": "stc_calculation",
                "data_points": len(src),
                "parameters": {
                    "length": length,
                    "fast_length": fast_length,
                    "slow_length": slow_length,
                    "factor": factor,
                },
                "calculation_id": self._calculation_count,
            },
        )

        min_required = max(length, fast_length, slow_length)
        if len(src) < min_required:
            logger.warning(
                "Insufficient data for STC calculation",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "issue": "insufficient_data",
                    "data_points": len(src),
                    "min_required": min_required,
                    "shortage": min_required - len(src),
                },
            )
            return pd.Series(dtype="float64", index=src.index)

        # Data quality validation
        self._validate_input_data_quality(src)

        try:
            # Step 1: Calculate EMAs
            logger.debug(
                "Calculating EMAs for STC",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "step": "ema_calculation",
                    "fast_length": fast_length,
                    "slow_length": slow_length,
                },
            )
            ema1 = ta.ema(src, length=fast_length)
            ema2 = ta.ema(src, length=slow_length)

            if ema1 is None or ema2 is None:
                logger.error(
                    "EMA calculation failed for STC",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "error_type": "ema_calculation_failed",
                        "ema1_failed": ema1 is None,
                        "ema2_failed": ema2 is None,
                    },
                )
                return pd.Series(dtype="float64", index=src.index)

            # Step 2: Calculate MACD
            logger.debug(
                "Calculating MACD for STC",
                extra={"indicator": "schaff_trend_cycle", "step": "macd_calculation"},
            )
            macd_val = ema1 - ema2

            # Step 3: First stochastic calculation
            logger.debug(
                "First stochastic calculation for STC",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "step": "first_stochastic",
                    "length": length,
                },
            )
            alpha = macd_val.rolling(window=length, min_periods=1).min()

            beta = macd_val.rolling(window=length, min_periods=1).max() - alpha

            # Add minimum threshold to prevent zero division
            min_beta_threshold = 1e-8  # Minimum threshold for beta values
            zero_beta_mask = beta.abs() < min_beta_threshold
            zero_beta_count = zero_beta_mask.sum()

            if zero_beta_count > 0:
                logger.debug(
                    "Small beta values detected in STC calculation - applying threshold",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "issue": "small_beta_values",
                        "small_count": int(zero_beta_count),
                        "total_points": len(beta),
                        "threshold_applied": min_beta_threshold,
                    },
                )
                # Apply minimum threshold to prevent division by zero
                beta = beta.where(~zero_beta_mask, min_beta_threshold)

            # All beta values are now guaranteed to be non-zero
            gamma = (macd_val - alpha) / beta * 100

            # Handle any remaining NaN values
            gamma = gamma.ffill().bfill()

            # Step 4: First smoothing (delta calculation)
            logger.debug(
                "First smoothing (delta) for STC",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "step": "first_smoothing",
                    "factor": factor,
                },
            )
            delta = self._apply_recursive_smoothing(gamma, factor)

            # Step 5: Second stochastic calculation
            logger.debug(
                "Second stochastic calculation for STC",
                extra={"indicator": "schaff_trend_cycle", "step": "second_stochastic"},
            )
            epsilon = delta.rolling(window=length, min_periods=1).min()

            zeta = delta.rolling(window=length, min_periods=1).max() - epsilon

            # Add minimum threshold to prevent zero division
            min_zeta_threshold = 1e-8  # Minimum threshold for zeta values
            zero_zeta_mask = zeta.abs() < min_zeta_threshold
            zero_zeta_count = zero_zeta_mask.sum()

            if zero_zeta_count > 0:
                logger.debug(
                    "Small zeta values detected in STC calculation - applying threshold",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "issue": "small_zeta_values",
                        "small_count": int(zero_zeta_count),
                        "total_points": len(zeta),
                        "threshold_applied": min_zeta_threshold,
                    },
                )
                # Apply minimum threshold to prevent division by zero
                zeta = zeta.where(~zero_zeta_mask, min_zeta_threshold)

            # All zeta values are now guaranteed to be non-zero
            eta = (delta - epsilon) / zeta * 100

            # Handle any remaining NaN values
            eta = eta.ffill().bfill()

            # Step 6: Final smoothing (STC calculation)
            logger.debug(
                "Final smoothing (STC) calculation",
                extra={"indicator": "schaff_trend_cycle", "step": "final_smoothing"},
            )
            stc_return = self._apply_recursive_smoothing(eta, factor)

            # Handle flat market conditions
            stc_return = self._handle_flat_market_conditions(stc_return)

            # Performance logging
            end_time = time.perf_counter()
            calculation_duration = (end_time - start_time) * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self._calculation_count += 1
            self._total_calculation_time += calculation_duration
            avg_calculation_time = (
                self._total_calculation_time / self._calculation_count
            )

            # Final validation
            valid_count = (~stc_return.isna()).sum()

            # Calculate value statistics
            stc_stats = {
                "min": float(stc_return.min()) if valid_count > 0 else 0,
                "max": float(stc_return.max()) if valid_count > 0 else 0,
                "mean": float(stc_return.mean()) if valid_count > 0 else 0,
                "std": float(stc_return.std()) if valid_count > 0 else 0,
            }

            # Count values in different ranges
            overbought_count = (stc_return > self.overbought).sum()
            oversold_count = (stc_return < self.oversold).sum()

            logger.debug(
                "Schaff Trend Cycle calculation completed",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "duration_ms": round(calculation_duration, 2),
                    "memory_current_mb": round(current / 1024 / 1024, 2),
                    "memory_peak_mb": round(peak / 1024 / 1024, 2),
                    "data_points": len(src),
                    "valid_count": int(valid_count),
                    "calculation_count": self._calculation_count,
                    "avg_duration_ms": round(avg_calculation_time, 2),
                    "stc_statistics": stc_stats,
                    "level_analysis": {
                        "overbought_periods": int(overbought_count),
                        "oversold_periods": int(oversold_count),
                        "neutral_periods": int(
                            valid_count - overbought_count - oversold_count
                        ),
                    },
                },
            )

            return stc_return.astype("float64")

        except Exception as e:
            logger.exception(
                "Schaff Trend Cycle calculation failed with exception",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "error_type": "calculation_exception",
                    "error_message": str(e),
                    "data_points": len(src),
                    "parameters": {
                        "length": length,
                        "fast_length": fast_length,
                        "slow_length": slow_length,
                        "factor": factor,
                    },
                },
            )
            return pd.Series(dtype="float64", index=src.index)

    def _apply_recursive_smoothing(self, series: pd.Series, factor: float) -> pd.Series:
        """
        Apply recursive smoothing as done in Pine Script.

        This implements: value := na(value[1]) ? value : value[1] + factor * (input - value[1])

        Args:
            series: Input series to smooth
            factor: Smoothing factor

        Returns:
            Smoothed series
        """
        if series.empty:
            return series.copy()

        result = pd.Series(dtype="float64", index=series.index)

        # Handle flat market conditions by checking for minimal variance
        series_variance = series.var()
        if pd.isna(series_variance) or series_variance < 1e-10:
            # For very flat series, return a smoothed version of the mean
            series_mean = series.mean()
            if pd.isna(series_mean):
                series_mean = 0.0
            result[:] = series_mean
            return result

        for i in range(len(series)):
            if i == 0 or pd.isna(result.iloc[i - 1]):
                # First value or previous value is NaN
                current_val = series.iloc[i]
                result.iloc[i] = current_val if not pd.isna(current_val) else 0.0
            else:
                # Recursive smoothing formula
                prev_val = result.iloc[i - 1]
                current_input = series.iloc[i]
                if not pd.isna(current_input):
                    result.iloc[i] = prev_val + factor * (current_input - prev_val)
                else:
                    result.iloc[i] = prev_val

        return result

    def get_trend_signals(
        self, stc_values: pd.Series, threshold_levels: dict[str, float] | None = None
    ) -> pd.Series:
        """
        Generate trend change signals based on STC values.

        Args:
            stc_values: Series of STC values
            threshold_levels: Dictionary with 'overbought' and 'oversold' keys

        Returns:
            Series with signals: 1 = bullish, -1 = bearish, 0 = neutral
        """
        if threshold_levels is None:
            threshold_levels = {
                "overbought": self.overbought,
                "oversold": self.oversold,
            }

        overbought = threshold_levels.get("overbought", self.overbought)
        oversold = threshold_levels.get("oversold", self.oversold)

        signals = pd.Series(0, index=stc_values.index, dtype="int64")

        # Bullish signal: STC crosses above oversold level
        bullish_cross = (stc_values.shift(1) <= oversold) & (stc_values > oversold)

        # Bearish signal: STC crosses below overbought level
        bearish_cross = (stc_values.shift(1) >= overbought) & (stc_values < overbought)

        signals.loc[bullish_cross] = 1
        signals.loc[bearish_cross] = -1

        return signals

    def get_cycle_analysis(self, stc_values: pd.Series) -> dict[str, pd.Series]:
        """
        Perform cycle phase analysis on STC values.

        Args:
            stc_values: Series of STC values

        Returns:
            Dictionary containing cycle analysis components:
            - phase: Current cycle phase (0=bottom, 1=rising, 2=top, 3=falling)
            - momentum: Rate of change in STC
            - cycle_position: Normalized position in cycle (0-1)
            - trend_strength: Measure of trend strength
        """
        analysis = {}

        # Calculate momentum (rate of change)
        momentum = stc_values.diff()
        analysis["momentum"] = momentum

        # Determine cycle phase
        phase = pd.Series(0, index=stc_values.index, dtype="int64")

        # Phase 0: Bottom (STC < 25 and rising)
        bottom_phase = (stc_values < 25) & (momentum > 0)
        phase.loc[bottom_phase] = 0

        # Phase 1: Rising (25 <= STC < 75 and rising)
        rising_phase = (stc_values >= 25) & (stc_values < 75) & (momentum > 0)
        phase.loc[rising_phase] = 1

        # Phase 2: Top (STC >= 75 and falling)
        top_phase = (stc_values >= 75) & (momentum < 0)
        phase.loc[top_phase] = 2

        # Phase 3: Falling (25 <= STC < 75 and falling)
        falling_phase = (stc_values >= 25) & (stc_values < 75) & (momentum < 0)
        phase.loc[falling_phase] = 3

        analysis["phase"] = phase

        # Normalize cycle position (0 = oversold extreme, 1 = overbought extreme)
        cycle_position = (
            stc_values - stc_values.rolling(window=50, min_periods=1).min()
        ) / (
            stc_values.rolling(window=50, min_periods=1).max()
            - stc_values.rolling(window=50, min_periods=1).min()
        )
        analysis["cycle_position"] = cycle_position.fillna(0.5)

        # Calculate trend strength based on momentum consistency
        momentum_sign = np.sign(momentum)
        trend_strength = momentum_sign.rolling(window=5, min_periods=1).mean().abs()
        analysis["trend_strength"] = trend_strength

        return analysis

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate STC indicator for a DataFrame.

        Args:
            df: DataFrame with OHLCV data (must contain 'close' column)

        Returns:
            DataFrame with added STC columns:
            - stc: STC values
            - stc_signal: Trend signals
            - stc_phase: Cycle phase
            - stc_momentum: STC momentum
            - stc_trend_strength: Trend strength
        """
        start_time = time.perf_counter()

        logger.debug(
            "Starting Schaff Trend Cycle DataFrame calculation",
            extra={
                "indicator": "schaff_trend_cycle",
                "input_data_points": len(df),
                "calculation_id": self._calculation_count + 1,
            },
        )

        if df.empty or "close" not in df.columns:
            logger.warning(
                "Invalid DataFrame for STC calculation",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "issue": "invalid_dataframe",
                    "is_empty": df.empty,
                    "has_close": "close" in df.columns if not df.empty else False,
                    "available_columns": list(df.columns) if not df.empty else [],
                },
            )
            return df.copy()

        result = df.copy()

        # Data quality validation
        self._validate_input_dataframe_quality(result)

        # Ensure close column is proper float64 dtype
        result["close"] = pd.to_numeric(result["close"], errors="coerce").astype(
            "float64"
        )

        try:
            # Calculate STC
            logger.debug(
                "Calculating STC values",
                extra={"indicator": "schaff_trend_cycle", "step": "stc_calculation"},
            )
            stc_values = self.calculate_stc(result["close"])
            result["stc"] = stc_values

            # Generate signals
            logger.debug(
                "Generating trend signals",
                extra={"indicator": "schaff_trend_cycle", "step": "signal_generation"},
            )
            signals = self.get_trend_signals(stc_values)
            result["stc_signal"] = signals

            # Perform cycle analysis
            logger.debug(
                "Performing cycle analysis",
                extra={"indicator": "schaff_trend_cycle", "step": "cycle_analysis"},
            )
            cycle_analysis = self.get_cycle_analysis(stc_values)
            result["stc_phase"] = cycle_analysis["phase"]
            result["stc_momentum"] = cycle_analysis["momentum"]
            result["stc_trend_strength"] = cycle_analysis["trend_strength"]

            # Count signals for logging
            bullish_signals = (signals == 1).sum()
            bearish_signals = (signals == -1).sum()

            if bullish_signals > 0:
                logger.debug(
                    "STC bullish signals generated",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "signal_type": "bullish_signals",
                        "signal_count": int(bullish_signals),
                        "signal_frequency": (
                            round((bullish_signals / len(df)) * 100, 2)
                            if len(df) > 0
                            else 0
                        ),
                    },
                )

            if bearish_signals > 0:
                logger.debug(
                    "STC bearish signals generated",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "signal_type": "bearish_signals",
                        "signal_count": int(bearish_signals),
                        "signal_frequency": (
                            round((bearish_signals / len(df)) * 100, 2)
                            if len(df) > 0
                            else 0
                        ),
                    },
                )

            # Final validation and performance logging
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000

            # Count valid outputs
            stc_valid = (~result["stc"].isna()).sum()

            # Generate comprehensive summary
            summary = self._generate_calculation_summary(result)

            logger.debug(
                "Schaff Trend Cycle DataFrame calculation completed",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "duration_ms": round(total_duration, 2),
                    "input_data_points": len(df),
                    "output_data_points": len(result),
                    "stc_valid_count": int(stc_valid),
                    "signal_summary": {
                        "bullish": int(bullish_signals),
                        "bearish": int(bearish_signals),
                        "total": int(bullish_signals + bearish_signals),
                    },
                    "analysis_summary": summary,
                    "calculation_success": True,
                },
            )

            return result

        except Exception as e:
            logger.exception(
                "Schaff Trend Cycle DataFrame calculation failed",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "error_type": "dataframe_calculation_failed",
                    "error_message": str(e),
                    "input_data_points": len(df),
                },
            )
            return df.copy()

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get the latest STC values.

        Args:
            df: DataFrame with calculated STC indicators

        Returns:
            Dictionary with latest STC values
        """
        if df.empty:
            logger.warning(
                "Empty DataFrame provided for latest values",
                extra={"indicator": "schaff_trend_cycle", "issue": "empty_dataframe"},
            )
            return {}

        latest = df.iloc[-1]

        stc_value = latest.get("stc", 0)
        signal_value = latest.get("stc_signal", 0)
        phase_value = latest.get("stc_phase", 0)

        latest_values = {
            "stc": stc_value,
            "stc_signal": signal_value,
            "stc_phase": phase_value,
            "stc_momentum": latest.get("stc_momentum"),
            "stc_trend_strength": latest.get("stc_trend_strength"),
            "is_overbought": (
                stc_value > self.overbought if stc_value is not None else False
            ),
            "is_oversold": (
                stc_value < self.oversold if stc_value is not None else False
            ),
        }

        # Log any active signals or conditions
        active_conditions = []
        if latest_values.get("is_overbought"):
            active_conditions.append("overbought")
        if latest_values.get("is_oversold"):
            active_conditions.append("oversold")
        if signal_value == 1:
            active_conditions.append("bullish_signal")
        elif signal_value == -1:
            active_conditions.append("bearish_signal")

        phase_names = {0: "bottom", 1: "rising", 2: "top", 3: "falling"}
        current_phase = phase_names.get(phase_value, "unknown")

        if active_conditions or stc_value is not None:
            logger.info(
                "Active Schaff Trend Cycle signals detected",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "signal_type": "current_signals",
                    "active_conditions": active_conditions,
                    "stc_value": float(stc_value) if stc_value is not None else None,
                    "current_phase": current_phase,
                    "signal_value": (
                        int(signal_value) if signal_value is not None else 0
                    ),
                    "trend_strength": (
                        float(latest_values.get("stc_trend_strength", 0))
                        if latest_values.get("stc_trend_strength") is not None
                        else 0
                    ),
                },
            )

        return latest_values

    def get_interpretation(self, stc_value: float) -> str:
        """
        Get human-readable interpretation of STC value.

        Args:
            stc_value: Current STC value

        Returns:
            String interpretation of the current STC state
        """
        if stc_value > self.overbought:
            return "Overbought - Potential bearish reversal"
        if stc_value < self.oversold:
            return "Oversold - Potential bullish reversal"
        if stc_value > 50:
            return "Bullish territory"
        return "Bearish territory"

    def get_trade_suggestions(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Generate trade suggestions based on STC analysis.

        Args:
            df: DataFrame with calculated STC indicators

        Returns:
            List of trade suggestion dictionaries
        """
        if df.empty or len(df) < 2:
            return []

        suggestions = []
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        stc_current = latest.get("stc", 0)
        previous.get("stc", 0)
        signal = latest.get("stc_signal", 0)
        phase = latest.get("stc_phase", 0)

        # Strong buy signal
        if signal == 1 and stc_current < 30 and phase == 0:
            suggestions.append(
                {
                    "action": "BUY",
                    "strength": "STRONG",
                    "reason": "STC crossed above oversold with bullish momentum",
                    "confidence": 0.8,
                }
            )

        # Strong sell signal
        elif signal == -1 and stc_current > 70 and phase == 2:
            suggestions.append(
                {
                    "action": "SELL",
                    "strength": "STRONG",
                    "reason": "STC crossed below overbought with bearish momentum",
                    "confidence": 0.8,
                }
            )

        # Moderate signals
        elif signal == 1:
            suggestions.append(
                {
                    "action": "BUY",
                    "strength": "MODERATE",
                    "reason": "STC bullish crossover",
                    "confidence": 0.6,
                }
            )

        elif signal == -1:
            suggestions.append(
                {
                    "action": "SELL",
                    "strength": "MODERATE",
                    "reason": "STC bearish crossover",
                    "confidence": 0.6,
                }
            )

        # Divergence warnings
        if len(df) >= 10:
            price_trend = df["close"].iloc[-10:].diff().sum()
            stc_trend = df["stc"].iloc[-10:].diff().sum()

            if price_trend > 0 and stc_trend < 0:
                suggestions.append(
                    {
                        "action": "CAUTION",
                        "strength": "WARNING",
                        "reason": "Bearish divergence detected - price rising but STC falling",
                        "confidence": 0.7,
                    }
                )
            elif price_trend < 0 and stc_trend > 0:
                suggestions.append(
                    {
                        "action": "OPPORTUNITY",
                        "strength": "WARNING",
                        "reason": "Bullish divergence detected - price falling but STC rising",
                        "confidence": 0.7,
                    }
                )

        return suggestions

    def _validate_input_data_quality(self, src: pd.Series) -> None:
        """
        Validate input data quality and log issues.

        Args:
            src: Input price series to validate
        """
        # Check for NaN values
        nan_count = src.isna().sum()
        if nan_count > 0:
            logger.warning(
                "NaN values in Schaff Trend Cycle input data",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "issue": "input_nan_values",
                    "nan_count": int(nan_count),
                    "total_points": len(src),
                    "nan_percentage": round((nan_count / len(src)) * 100, 2),
                },
            )

        # Check for zero or negative prices
        invalid_prices = (src <= 0).sum()
        if invalid_prices > 0:
            logger.warning(
                "Invalid price values in Schaff Trend Cycle input data",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "issue": "invalid_prices",
                    "invalid_count": int(invalid_prices),
                    "min_price": float(src.min()) if not src.empty else 0,
                },
            )

        # Check for flat market conditions (minimal price variance)
        if len(src) > 1:
            price_variance = src.var()
            if pd.isna(price_variance) or price_variance < 1e-10:
                logger.info(
                    "Very low price variance detected in STC input - calculations may produce minimal signals",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "issue": "low_price_variance",
                        "variance": (
                            float(price_variance) if not pd.isna(price_variance) else 0
                        ),
                        "price_range": float(src.max() - src.min()),
                        "mean_price": float(src.mean()),
                    },
                )

        # Check for extreme price changes
        if len(src) > 1:
            price_changes = src.pct_change().abs()
            extreme_changes = (price_changes > 0.3).sum()  # More than 30% change
            if extreme_changes > 0:
                logger.warning(
                    "Extreme price changes in Schaff Trend Cycle input data",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "issue": "extreme_price_changes",
                        "extreme_change_count": int(extreme_changes),
                        "max_change_pct": (
                            round(float(price_changes.max()) * 100, 2)
                            if not price_changes.empty
                            else 0
                        ),
                    },
                )

    def _validate_input_dataframe_quality(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame quality and log issues.

        Args:
            df: Input DataFrame to validate
        """
        if "close" not in df.columns:
            return

        close_series = df["close"]

        # Check for duplicate timestamps
        if hasattr(df.index, "duplicated"):
            duplicate_count = df.index.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(
                    "Duplicate timestamps in Schaff Trend Cycle input data",
                    extra={
                        "indicator": "schaff_trend_cycle",
                        "issue": "duplicate_timestamps",
                        "duplicate_count": int(duplicate_count),
                    },
                )

        # Check data continuity
        if len(close_series) > 1:
            # Check for gaps in data
            time_diffs = pd.Series(df.index).diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()

                # Dynamic threshold based on interval or user-specified multiplier
                median_seconds = (
                    median_diff.total_seconds()
                    if hasattr(median_diff, "total_seconds")
                    else 0
                )

                if self.gap_threshold_multiplier is not None:
                    threshold_multiplier = self.gap_threshold_multiplier
                elif median_seconds <= 60:  # High frequency (≤60s)
                    threshold_multiplier = 10
                elif median_seconds <= 300:  # Medium frequency (≤300s)
                    threshold_multiplier = 5
                else:  # Low frequency (>300s)
                    threshold_multiplier = 3

                large_gaps = (time_diffs > median_diff * threshold_multiplier).sum()
                if large_gaps > 0:
                    logger.warning(
                        "Large time gaps detected in Schaff Trend Cycle input data",
                        extra={
                            "indicator": "schaff_trend_cycle",
                            "issue": "large_time_gaps",
                            "gap_count": int(large_gaps),
                            "median_diff": str(median_diff),
                            "threshold_multiplier": threshold_multiplier,
                            "median_seconds": median_seconds,
                        },
                    )

    def _generate_calculation_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate summary of calculation results for logging.

        Args:
            df: DataFrame with calculated STC analysis

        Returns:
            Dictionary with calculation summary
        """
        summary: dict[str, Any] = {"total_data_points": len(df)}

        # STC value statistics
        if "stc" in df.columns:
            stc_series = df["stc"].dropna()
            if not stc_series.empty:
                summary.update(
                    {
                        "stc_valid_count": len(stc_series),
                        "stc_min": round(float(stc_series.min()), 2),
                        "stc_max": round(float(stc_series.max()), 2),
                        "stc_mean": round(float(stc_series.mean()), 2),
                        "stc_std": round(float(stc_series.std()), 2),
                    }
                )

        # Phase distribution
        if "stc_phase" in df.columns:
            phase_counts = df["stc_phase"].value_counts().to_dict()
            phase_names = {0: "bottom", 1: "rising", 2: "top", 3: "falling"}
            phase_distribution = {}
            for phase_num, count in phase_counts.items():
                phase_name = phase_names.get(phase_num, f"unknown_{phase_num}")
                phase_distribution[phase_name] = int(count)
            summary["phase_distribution"] = phase_distribution

        # Signal distribution
        if "stc_signal" in df.columns:
            signal_counts = df["stc_signal"].value_counts().to_dict()
            summary["signal_distribution"] = {
                "bullish": signal_counts.get(1, 0),
                "bearish": signal_counts.get(-1, 0),
                "neutral": signal_counts.get(0, 0),
            }

        # Trend strength statistics
        if "stc_trend_strength" in df.columns:
            strength_series = df["stc_trend_strength"].dropna()
            if not strength_series.empty:
                summary.update(
                    {
                        "avg_trend_strength": round(float(strength_series.mean()), 4),
                        "max_trend_strength": round(float(strength_series.max()), 4),
                    }
                )

        return summary

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for monitoring.

        Returns:
            Dictionary with performance metrics
        """
        avg_time = (
            self._total_calculation_time / self._calculation_count
            if self._calculation_count > 0
            else 0
        )

        metrics = {
            "calculation_count": self._calculation_count,
            "total_calculation_time_ms": round(self._total_calculation_time, 2),
            "average_calculation_time_ms": round(avg_time, 2),
            "parameters": {
                "length": self.length,
                "fast_length": self.fast_length,
                "slow_length": self.slow_length,
                "factor": self.factor,
                "overbought": self.overbought,
                "oversold": self.oversold,
                "gap_threshold_multiplier": self.gap_threshold_multiplier,
            },
            "last_data_quality_check": self._last_data_quality_check,
        }

        logger.debug(
            "Schaff Trend Cycle performance metrics",
            extra={"indicator": "schaff_trend_cycle", "metrics": metrics},
        )

        return metrics

    def _handle_flat_market_conditions(
        self, series: pd.Series, default_value: float = 50.0
    ) -> pd.Series:
        """
        Handle flat market conditions by providing appropriate fallback values.

        Args:
            series: The calculated STC series
            default_value: Default value to use for flat market periods

        Returns:
            Series with flat market periods handled
        """
        if series.empty:
            return series

        # Detect flat periods (very small changes)
        series_changes = series.diff().abs()
        flat_threshold = 1e-6  # Very small change threshold
        flat_mask = series_changes < flat_threshold

        # Count consecutive flat periods
        consecutive_flat = flat_mask.groupby((~flat_mask).cumsum()).cumsum()
        long_flat_mask = consecutive_flat > 10  # More than 10 consecutive flat periods

        if long_flat_mask.sum() > 0:
            logger.debug(
                "Long flat market periods detected in STC - applying fallback values",
                extra={
                    "indicator": "schaff_trend_cycle",
                    "flat_periods": int(long_flat_mask.sum()),
                    "total_periods": len(series),
                    "fallback_value": default_value,
                },
            )
            # Set long flat periods to the default value (neutral)
            series = series.copy()
            series.loc[long_flat_mask] = default_value

        return series
