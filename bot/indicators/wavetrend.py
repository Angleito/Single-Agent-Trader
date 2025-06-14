"""
WaveTrend Oscillator implementation.

This module provides a Python implementation of the WaveTrend oscillator,
which is the core component of the VuManChu Cipher indicators.
Uses vectorized operations for performance.
"""

import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from ..utils import ta

logger = logging.getLogger(__name__)


class WaveTrend:
    """
    WaveTrend Oscillator implementation.

    The WaveTrend oscillator is based on the following Pine Script formula:
    _esa = ema(_src, _chlen)
    _de = ema(abs(_src - _esa), _chlen)
    _ci = (_src - _esa) / (0.015 * _de)
    _tci = ema(_ci, _avg)
    _wt1 = _tci
    _wt2 = sma(_wt1, _malen)

    Features:
    - WaveTrend oscillator calculation (wt1, wt2)
    - Overbought/oversold condition detection
    - Cross pattern identification
    """

    def __init__(
        self,
        channel_length: int = 10,
        average_length: int = 21,
        ma_length: int = 4,
        overbought_level: float = 60.0,
        oversold_level: float = -60.0,
    ):
        """
        Initialize WaveTrend parameters.

        Args:
            channel_length: Channel length for ESA and DE calculations
            average_length: Average length for TCI calculation
            ma_length: Moving average length for wt2 calculation
            overbought_level: Overbought threshold level
            oversold_level: Oversold threshold level
        """
        self.channel_length = channel_length
        self.average_length = average_length
        self.ma_length = ma_length
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._last_data_quality_check = None

        logger.info(
            "WaveTrend indicator initialized",
            extra={
                "channel_length": channel_length,
                "average_length": average_length,
                "ma_length": ma_length,
                "overbought_level": overbought_level,
                "oversold_level": oversold_level,
                "indicator": "wavetrend",
            },
        )

    def calculate_wavetrend(
        self,
        src: pd.Series,
        channel_len: int | None = None,
        average_len: int | None = None,
        ma_len: int | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate WaveTrend oscillator values.

        Args:
            src: Source price series (typically close prices)
            channel_len: Channel length override (optional)
            average_len: Average length override (optional)
            ma_len: MA length override (optional)

        Returns:
            Tuple of (wt1, wt2) series

        Raises:
            ValueError: If input data is invalid
        """
        start_time = time.perf_counter()
        tracemalloc.start()

        if src is None or src.empty:
            logger.error(
                "WaveTrend calculation failed: empty source data",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "empty_data",
                    "data_points": 0,
                },
            )
            raise ValueError("Source data cannot be empty")

        # Use instance parameters if not overridden
        ch_len = channel_len or self.channel_length
        avg_len = average_len or self.average_length
        ma_length = ma_len or self.ma_length

        # Log calculation start
        logger.debug(
            "Starting WaveTrend calculation",
            extra={
                "indicator": "wavetrend",
                "data_points": len(src),
                "channel_length": ch_len,
                "average_length": avg_len,
                "ma_length": ma_length,
                "calculation_id": self._calculation_count,
            },
        )

        # Validate parameters
        if ch_len <= 0 or avg_len <= 0 or ma_length <= 0:
            logger.error(
                "WaveTrend calculation failed: invalid parameters",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "invalid_parameters",
                    "channel_length": ch_len,
                    "average_length": avg_len,
                    "ma_length": ma_length,
                },
            )
            raise ValueError("All length parameters must be positive")

        min_required = max(ch_len, avg_len, ma_length)
        if len(src) < min_required:
            logger.warning(
                "Insufficient data for WaveTrend calculation",
                extra={
                    "indicator": "wavetrend",
                    "issue": "insufficient_data",
                    "data_points": len(src),
                    "min_required": min_required,
                    "shortage": min_required - len(src),
                },
            )
            return pd.Series(dtype=float, index=src.index), pd.Series(
                dtype=float, index=src.index
            )

        # Data quality validation
        nan_count = src.isna().sum()
        nan_percentage = (nan_count / len(src)) * 100 if len(src) > 0 else 0

        if nan_percentage > 10:
            logger.warning(
                "High NaN percentage in WaveTrend source data",
                extra={
                    "indicator": "wavetrend",
                    "issue": "high_nan_percentage",
                    "nan_count": int(nan_count),
                    "total_points": len(src),
                    "nan_percentage": round(nan_percentage, 2),
                },
            )

        # Check for price range anomalies
        price_std = src.std()
        price_mean = src.mean()
        if price_std > 0:
            outlier_threshold = 3 * price_std
            outliers = abs(src - price_mean) > outlier_threshold
            outlier_count = outliers.sum()

            if outlier_count > 0:
                logger.warning(
                    "Price outliers detected in WaveTrend source data",
                    extra={
                        "indicator": "wavetrend",
                        "issue": "price_outliers",
                        "outlier_count": int(outlier_count),
                        "outlier_percentage": round(
                            (outlier_count / len(src)) * 100, 2
                        ),
                        "price_std": round(float(price_std), 4),
                        "price_mean": round(float(price_mean), 4),
                    },
                )

        # Ensure src is float64 to prevent pandas warnings
        src = pd.to_numeric(src, errors="coerce").astype("float64")

        try:
            # Calculate ESA (Exponential Smoothing Average)
            logger.debug(
                "Calculating ESA (Exponential Smoothing Average)",
                extra={
                    "indicator": "wavetrend",
                    "step": "esa_calculation",
                    "length": ch_len,
                },
            )
            esa = ta.ema(src, length=ch_len)
            if esa is None:
                logger.error(
                    "Failed to calculate ESA",
                    extra={
                        "indicator": "wavetrend",
                        "error_type": "esa_calculation_failed",
                        "channel_length": ch_len,
                    },
                )
                return pd.Series(dtype=float, index=src.index), pd.Series(
                    dtype=float, index=src.index
                )

            esa = esa.astype("float64")

            # Calculate DE (Deviation)
            logger.debug(
                "Calculating DE (Deviation)",
                extra={
                    "indicator": "wavetrend",
                    "step": "de_calculation",
                    "length": ch_len,
                },
            )
            de = ta.ema(abs(src - esa), length=ch_len)
            if de is None:
                logger.error(
                    "Failed to calculate DE",
                    extra={
                        "indicator": "wavetrend",
                        "error_type": "de_calculation_failed",
                        "channel_length": ch_len,
                    },
                )
                return pd.Series(dtype=float, index=src.index), pd.Series(
                    dtype=float, index=src.index
                )

            de = de.astype("float64")

            # Check for zero DE values which would cause division issues
            zero_de_count = (de == 0).sum()
            if zero_de_count > 0:
                logger.warning(
                    "Zero DE values detected in WaveTrend calculation",
                    extra={
                        "indicator": "wavetrend",
                        "issue": "zero_de_values",
                        "zero_count": int(zero_de_count),
                        "total_points": len(de),
                    },
                )

            # Calculate CI (Commodity Channel Index style)
            logger.debug(
                "Calculating CI (Commodity Channel Index)",
                extra={"indicator": "wavetrend", "step": "ci_calculation"},
            )
            # Avoid division by zero
            ci = np.where(de != 0, (src - esa) / (0.015 * de), 0)
            ci = pd.Series(ci, index=src.index, dtype="float64")

            # Validate CI values
            ci_nan_count = ci.isna().sum()
            ci_inf_count = np.isinf(ci).sum()
            if ci_nan_count > 0 or ci_inf_count > 0:
                logger.warning(
                    "Invalid CI values detected",
                    extra={
                        "indicator": "wavetrend",
                        "issue": "invalid_ci_values",
                        "nan_count": int(ci_nan_count),
                        "inf_count": int(ci_inf_count),
                    },
                )

            # Calculate TCI (True Commodity Index)
            logger.debug(
                "Calculating TCI (True Commodity Index)",
                extra={
                    "indicator": "wavetrend",
                    "step": "tci_calculation",
                    "length": avg_len,
                },
            )
            tci = ta.ema(ci, length=avg_len)
            if tci is None:
                logger.error(
                    "Failed to calculate TCI",
                    extra={
                        "indicator": "wavetrend",
                        "error_type": "tci_calculation_failed",
                        "average_length": avg_len,
                    },
                )
                return pd.Series(dtype=float, index=src.index), pd.Series(
                    dtype=float, index=src.index
                )

            tci = tci.astype("float64")

            # WaveTrend 1 = TCI
            wt1 = tci.copy()

            # WaveTrend 2 = SMA of WaveTrend 1
            logger.debug(
                "Calculating WT2 (SMA of WT1)",
                extra={
                    "indicator": "wavetrend",
                    "step": "wt2_calculation",
                    "length": ma_length,
                },
            )
            wt2 = ta.sma(wt1, length=ma_length)
            if wt2 is None:
                logger.error(
                    "Failed to calculate WT2",
                    extra={
                        "indicator": "wavetrend",
                        "error_type": "wt2_calculation_failed",
                        "ma_length": ma_length,
                    },
                )
                return pd.Series(dtype=float, index=src.index), pd.Series(
                    dtype=float, index=src.index
                )

            wt2 = wt2.astype("float64")

            # Performance logging
            end_time = time.perf_counter()
            calculation_duration = (
                end_time - start_time
            ) * 1000  # Convert to milliseconds

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self._calculation_count += 1
            self._total_calculation_time += calculation_duration
            avg_calculation_time = (
                self._total_calculation_time / self._calculation_count
            )

            # Final validation
            wt1_valid_count = (~wt1.isna()).sum()
            wt2_valid_count = (~wt2.isna()).sum()

            logger.info(
                "WaveTrend calculation completed",
                extra={
                    "indicator": "wavetrend",
                    "duration_ms": round(calculation_duration, 2),
                    "memory_current_mb": round(current / 1024 / 1024, 2),
                    "memory_peak_mb": round(peak / 1024 / 1024, 2),
                    "data_points": len(src),
                    "wt1_valid_count": int(wt1_valid_count),
                    "wt2_valid_count": int(wt2_valid_count),
                    "calculation_count": self._calculation_count,
                    "avg_duration_ms": round(avg_calculation_time, 2),
                    "wt1_range": (
                        [round(float(wt1.min()), 4), round(float(wt1.max()), 4)]
                        if wt1_valid_count > 0
                        else [0, 0]
                    ),
                    "wt2_range": (
                        [round(float(wt2.min()), 4), round(float(wt2.max()), 4)]
                        if wt2_valid_count > 0
                        else [0, 0]
                    ),
                },
            )

            return wt1, wt2

        except Exception as e:
            logger.error(
                "WaveTrend calculation failed with exception",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "calculation_exception",
                    "error_message": str(e),
                    "data_points": len(src),
                    "parameters": {
                        "channel_length": ch_len,
                        "average_length": avg_len,
                        "ma_length": ma_length,
                    },
                },
            )
            return pd.Series(dtype=float, index=src.index), pd.Series(
                dtype=float, index=src.index
            )

    def get_overbought_oversold_conditions(
        self,
        wt2: pd.Series,
        ob_level: float | None = None,
        os_level: float | None = None,
    ) -> dict[str, pd.Series]:
        """
        Get overbought and oversold conditions.

        Args:
            wt2: WaveTrend 2 series
            ob_level: Overbought level override (optional)
            os_level: Oversold level override (optional)

        Returns:
            Dictionary with overbought/oversold conditions:
            - 'overbought': Boolean series for overbought conditions
            - 'oversold': Boolean series for oversold conditions
            - 'overbought_cross_down': Boolean series for overbought exit
            - 'oversold_cross_up': Boolean series for oversold exit
        """
        logger.debug(
            "Calculating overbought/oversold conditions",
            extra={
                "indicator": "wavetrend",
                "step": "ob_os_conditions",
                "wt2_data_points": len(wt2) if wt2 is not None else 0,
            },
        )

        if wt2 is None or wt2.empty:
            logger.warning(
                "Empty WT2 data for overbought/oversold conditions",
                extra={"indicator": "wavetrend", "issue": "empty_wt2_data"},
            )
            empty_series = pd.Series(dtype=bool, index=pd.Index([]))
            return {
                "overbought": empty_series,
                "oversold": empty_series,
                "overbought_cross_down": empty_series,
                "oversold_cross_up": empty_series,
            }

        # Use instance parameters if not overridden
        ob_threshold = ob_level if ob_level is not None else self.overbought_level
        os_threshold = os_level if os_level is not None else self.oversold_level

        logger.debug(
            "Using OB/OS thresholds",
            extra={
                "indicator": "wavetrend",
                "overbought_threshold": ob_threshold,
                "oversold_threshold": os_threshold,
            },
        )

        try:
            # Basic conditions
            overbought = wt2 > ob_threshold
            oversold = wt2 < os_threshold

            # Cross conditions
            overbought_cross_down = (wt2.shift(1) > ob_threshold) & (
                wt2 <= ob_threshold
            )
            oversold_cross_up = (wt2.shift(1) < os_threshold) & (wt2 >= os_threshold)

            # Count conditions for logging
            ob_count = overbought.sum()
            os_count = oversold.sum()
            ob_exit_count = overbought_cross_down.sum()
            os_exit_count = oversold_cross_up.sum()

            logger.debug(
                "OB/OS conditions calculated",
                extra={
                    "indicator": "wavetrend",
                    "overbought_count": int(ob_count),
                    "oversold_count": int(os_count),
                    "overbought_exit_count": int(ob_exit_count),
                    "oversold_exit_count": int(os_exit_count),
                    "total_points": len(wt2),
                },
            )

            return {
                "overbought": overbought,
                "oversold": oversold,
                "overbought_cross_down": overbought_cross_down,
                "oversold_cross_up": oversold_cross_up,
            }

        except Exception as e:
            logger.error(
                "Failed to calculate OB/OS conditions",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "ob_os_calculation_failed",
                    "error_message": str(e),
                },
            )
            empty_series = pd.Series(
                dtype=bool, index=wt2.index if wt2 is not None else pd.Index([])
            )
            return {
                "overbought": empty_series,
                "oversold": empty_series,
                "overbought_cross_down": empty_series,
                "oversold_cross_up": empty_series,
            }

    def get_cross_conditions(
        self, wt1: pd.Series, wt2: pd.Series
    ) -> dict[str, pd.Series]:
        """
        Get cross conditions between WaveTrend 1 and WaveTrend 2.

        Args:
            wt1: WaveTrend 1 series
            wt2: WaveTrend 2 series

        Returns:
            Dictionary with cross conditions:
            - 'wt1_cross_above_wt2': Boolean series for bullish cross
            - 'wt1_cross_below_wt2': Boolean series for bearish cross
            - 'wt1_above_wt2': Boolean series for wt1 > wt2
            - 'wt1_below_wt2': Boolean series for wt1 < wt2
        """
        logger.debug(
            "Calculating cross conditions",
            extra={
                "indicator": "wavetrend",
                "step": "cross_conditions",
                "wt1_data_points": len(wt1) if wt1 is not None else 0,
                "wt2_data_points": len(wt2) if wt2 is not None else 0,
            },
        )

        if wt1 is None or wt1.empty or wt2 is None or wt2.empty:
            logger.warning(
                "Empty WT1 or WT2 data for cross conditions",
                extra={
                    "indicator": "wavetrend",
                    "issue": "empty_wt_data",
                    "wt1_empty": wt1 is None or wt1.empty,
                    "wt2_empty": wt2 is None or wt2.empty,
                },
            )
            empty_series = pd.Series(dtype=bool, index=pd.Index([]))
            return {
                "wt1_cross_above_wt2": empty_series,
                "wt1_cross_below_wt2": empty_series,
                "wt1_above_wt2": empty_series,
                "wt1_below_wt2": empty_series,
            }

        try:
            # Basic position conditions
            wt1_above_wt2 = wt1 > wt2
            wt1_below_wt2 = wt1 < wt2

            # Cross conditions
            wt1_cross_above_wt2 = (wt1.shift(1) <= wt2.shift(1)) & (wt1 > wt2)
            wt1_cross_below_wt2 = (wt1.shift(1) >= wt2.shift(1)) & (wt1 < wt2)

            # Count conditions for logging
            cross_up_count = wt1_cross_above_wt2.sum()
            cross_down_count = wt1_cross_below_wt2.sum()
            above_count = wt1_above_wt2.sum()
            below_count = wt1_below_wt2.sum()

            # Log signal generation events
            if cross_up_count > 0:
                logger.info(
                    "WaveTrend bullish cross signals generated",
                    extra={
                        "indicator": "wavetrend",
                        "signal_type": "bullish_cross",
                        "signal_count": int(cross_up_count),
                        "signal_frequency": (
                            round((cross_up_count / len(wt1)) * 100, 2)
                            if len(wt1) > 0
                            else 0
                        ),
                    },
                )

            if cross_down_count > 0:
                logger.info(
                    "WaveTrend bearish cross signals generated",
                    extra={
                        "indicator": "wavetrend",
                        "signal_type": "bearish_cross",
                        "signal_count": int(cross_down_count),
                        "signal_frequency": (
                            round((cross_down_count / len(wt1)) * 100, 2)
                            if len(wt1) > 0
                            else 0
                        ),
                    },
                )

            logger.debug(
                "Cross conditions calculated",
                extra={
                    "indicator": "wavetrend",
                    "cross_up_count": int(cross_up_count),
                    "cross_down_count": int(cross_down_count),
                    "wt1_above_count": int(above_count),
                    "wt1_below_count": int(below_count),
                    "total_points": len(wt1),
                },
            )

            return {
                "wt1_cross_above_wt2": wt1_cross_above_wt2,
                "wt1_cross_below_wt2": wt1_cross_below_wt2,
                "wt1_above_wt2": wt1_above_wt2,
                "wt1_below_wt2": wt1_below_wt2,
            }

        except Exception as e:
            logger.error(
                "Failed to calculate cross conditions",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "cross_calculation_failed",
                    "error_message": str(e),
                },
            )
            empty_series = pd.Series(
                dtype=bool, index=wt1.index if wt1 is not None else pd.Index([])
            )
            return {
                "wt1_cross_above_wt2": empty_series,
                "wt1_cross_below_wt2": empty_series,
                "wt1_above_wt2": empty_series,
                "wt1_below_wt2": empty_series,
            }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WaveTrend indicators for a DataFrame.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with added WaveTrend columns:
            - wt1: WaveTrend 1 values
            - wt2: WaveTrend 2 values
            - wt_overbought: Overbought condition
            - wt_oversold: Oversold condition
            - wt_cross_up: Bullish cross signal
            - wt_cross_down: Bearish cross signal
        """
        start_time = time.perf_counter()

        logger.info(
            "Starting WaveTrend DataFrame calculation",
            extra={
                "indicator": "wavetrend",
                "input_data_points": len(df),
                "calculation_id": self._calculation_count + 1,
            },
        )

        min_required = max(self.channel_length, self.average_length, self.ma_length)
        if len(df) < min_required:
            logger.warning(
                "Insufficient data for WaveTrend calculation",
                extra={
                    "indicator": "wavetrend",
                    "issue": "insufficient_data",
                    "data_points": len(df),
                    "min_required": min_required,
                    "shortage": min_required - len(df),
                },
            )
            return df.copy()

        result = df.copy()

        # Data quality validation
        self._validate_input_data_quality(result)

        # Ensure close column is proper float64 dtype
        if "close" in result.columns:
            result["close"] = pd.to_numeric(result["close"], errors="coerce").astype(
                "float64"
            )
        else:
            logger.error(
                "Close column not found in DataFrame",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "missing_close_column",
                    "available_columns": list(result.columns),
                },
            )
            return result

        try:
            # Calculate WaveTrend oscillator
            logger.debug(
                "Calculating WaveTrend oscillator values",
                extra={"indicator": "wavetrend", "step": "oscillator_calculation"},
            )
            wt1, wt2 = self.calculate_wavetrend(result["close"])
            result["wt1"] = wt1
            result["wt2"] = wt2

            # Get overbought/oversold conditions
            logger.debug(
                "Calculating overbought/oversold conditions",
                extra={"indicator": "wavetrend", "step": "ob_os_conditions"},
            )
            ob_os_conditions = self.get_overbought_oversold_conditions(wt2)
            result["wt_overbought"] = ob_os_conditions["overbought"]
            result["wt_oversold"] = ob_os_conditions["oversold"]
            result["wt_overbought_exit"] = ob_os_conditions["overbought_cross_down"]
            result["wt_oversold_exit"] = ob_os_conditions["oversold_cross_up"]

            # Get cross conditions
            logger.debug(
                "Calculating cross conditions",
                extra={"indicator": "wavetrend", "step": "cross_conditions"},
            )
            cross_conditions = self.get_cross_conditions(wt1, wt2)
            result["wt_cross_up"] = cross_conditions["wt1_cross_above_wt2"]
            result["wt_cross_down"] = cross_conditions["wt1_cross_below_wt2"]
            result["wt1_above_wt2"] = cross_conditions["wt1_above_wt2"]

            # Final validation and performance logging
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000

            # Count valid outputs
            wt1_valid = (~result["wt1"].isna()).sum()
            wt2_valid = (~result["wt2"].isna()).sum()

            # Count signal occurrences
            signal_counts = {
                "overbought": result["wt_overbought"].sum(),
                "oversold": result["wt_oversold"].sum(),
                "cross_up": result["wt_cross_up"].sum(),
                "cross_down": result["wt_cross_down"].sum(),
            }

            logger.info(
                "WaveTrend DataFrame calculation completed",
                extra={
                    "indicator": "wavetrend",
                    "duration_ms": round(total_duration, 2),
                    "input_data_points": len(df),
                    "output_data_points": len(result),
                    "wt1_valid_count": int(wt1_valid),
                    "wt2_valid_count": int(wt2_valid),
                    "signal_counts": {k: int(v) for k, v in signal_counts.items()},
                    "calculation_success": True,
                },
            )

            # Store data quality info for monitoring
            self._last_data_quality_check = {
                "timestamp": time.time(),
                "data_points": len(df),
                "valid_outputs": int(min(wt1_valid, wt2_valid)),
                "signal_counts": signal_counts,
            }

            return result

        except Exception as e:
            logger.error(
                "WaveTrend DataFrame calculation failed",
                extra={
                    "indicator": "wavetrend",
                    "error_type": "dataframe_calculation_failed",
                    "error_message": str(e),
                    "input_data_points": len(df),
                },
            )
            return df.copy()

    def _validate_input_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate input data quality and log issues.

        Args:
            df: Input DataFrame to validate
        """
        if "close" not in df.columns:
            return

        close_series = df["close"]

        # Check for NaN values
        nan_count = close_series.isna().sum()
        if nan_count > 0:
            logger.warning(
                "NaN values in input data",
                extra={
                    "indicator": "wavetrend",
                    "issue": "input_nan_values",
                    "nan_count": int(nan_count),
                    "total_points": len(close_series),
                    "nan_percentage": round((nan_count / len(close_series)) * 100, 2),
                },
            )

        # Check for duplicate timestamps
        if hasattr(df.index, "duplicated"):
            duplicate_count = df.index.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(
                    "Duplicate timestamps in input data",
                    extra={
                        "indicator": "wavetrend",
                        "issue": "duplicate_timestamps",
                        "duplicate_count": int(duplicate_count),
                    },
                )

        # Check for zero or negative prices
        invalid_prices = (close_series <= 0).sum()
        if invalid_prices > 0:
            logger.warning(
                "Invalid price values in input data",
                extra={
                    "indicator": "wavetrend",
                    "issue": "invalid_prices",
                    "invalid_count": int(invalid_prices),
                    "min_price": (
                        float(close_series.min()) if not close_series.empty else 0
                    ),
                },
            )

        # Check data continuity (gaps)
        if len(close_series) > 1:
            price_changes = close_series.pct_change().abs()
            extreme_changes = (price_changes > 0.1).sum()  # More than 10% change
            if extreme_changes > 0:
                logger.warning(
                    "Extreme price changes detected",
                    extra={
                        "indicator": "wavetrend",
                        "issue": "extreme_price_changes",
                        "extreme_change_count": int(extreme_changes),
                        "max_change_pct": (
                            round(float(price_changes.max()) * 100, 2)
                            if not price_changes.empty
                            else 0
                        ),
                    },
                )

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get the latest WaveTrend values.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with latest indicator values
        """
        if df.empty:
            logger.warning(
                "Empty DataFrame provided for latest values",
                extra={"indicator": "wavetrend", "issue": "empty_dataframe"},
            )
            return {}

        latest = df.iloc[-1]

        latest_values = {
            "wt1": latest.get("wt1"),
            "wt2": latest.get("wt2"),
            "wt_overbought": latest.get("wt_overbought"),
            "wt_oversold": latest.get("wt_oversold"),
            "wt_cross_up": latest.get("wt_cross_up"),
            "wt_cross_down": latest.get("wt_cross_down"),
            "wt1_above_wt2": latest.get("wt1_above_wt2"),
        }

        # Log any active signals
        active_signals = []
        if latest_values.get("wt_overbought"):
            active_signals.append("overbought")
        if latest_values.get("wt_oversold"):
            active_signals.append("oversold")
        if latest_values.get("wt_cross_up"):
            active_signals.append("cross_up")
        if latest_values.get("wt_cross_down"):
            active_signals.append("cross_down")

        if active_signals:
            logger.info(
                "Active WaveTrend signals detected",
                extra={
                    "indicator": "wavetrend",
                    "signal_type": "current_signals",
                    "active_signals": active_signals,
                    "wt1_value": (
                        float(latest_values["wt1"])
                        if latest_values["wt1"] is not None
                        else None
                    ),
                    "wt2_value": (
                        float(latest_values["wt2"])
                        if latest_values["wt2"] is not None
                        else None
                    ),
                },
            )

        return latest_values

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
            "last_data_quality_check": self._last_data_quality_check,
        }

        logger.debug(
            "WaveTrend performance metrics",
            extra={"indicator": "wavetrend", "metrics": metrics},
        )

        return metrics
