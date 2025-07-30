"""
8-EMA Ribbon System implementation.

This module provides a Python implementation of the 8-EMA Ribbon system with crossover signals,
matching the Pine Script reference. The system uses 8 EMAs with lengths [5,11,15,18,21,24,28,34]
and generates various crossover signals for trading decisions.
"""

import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from bot.utils import ta

logger = logging.getLogger(__name__)


class EMAribbon:
    """
    8-EMA Ribbon System implementation.

    Features:
    - 8 EMAs with configurable lengths [5,11,15,18,21,24,28,34]
    - Ribbon direction analysis (bullish/bearish)
    - Long/Short EMA crossover signals (EMA2 vs EMA8)
    - Red/Green cross patterns (EMA1 vs EMA2)
    - Blue triangle signals (EMA2 vs EMA3)
    - Ribbon trend strength calculations
    """

    def __init__(self, lengths: list[int] | None = None) -> None:
        """
        Initialize EMA Ribbon parameters.

        Args:
            lengths: List of EMA lengths (default: [5,11,15,18,21,24,28,34])
        """
        self.lengths = lengths or [5, 11, 15, 18, 21, 24, 28, 34]

        if len(self.lengths) != 8:
            logger.error(
                "EMA Ribbon initialization failed: invalid length count",
                extra={
                    "indicator": "ema_ribbon",
                    "error_type": "invalid_length_count",
                    "provided_lengths": len(self.lengths),
                    "required_lengths": 8,
                    "lengths": self.lengths,
                },
            )
            raise ValueError("EMA Ribbon requires exactly 8 EMA lengths")

        # Sort lengths to ensure proper ordering
        self.lengths = sorted(self.lengths)

        # Assign specific EMAs for signal calculations (matching Pine Script)
        self.ema1_length: int = self.lengths[0]  # 5
        self.ema2_length: int = self.lengths[1]  # 11
        self.ema3_length: int = self.lengths[2]  # 15
        self.ema8_length: int = self.lengths[7]  # 34

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._signal_history: list[dict[str, Any]] = []
        self._last_data_quality_check = None

        logger.info(
            "EMA Ribbon indicator initialized",
            extra={
                "indicator": "ema_ribbon",
                "ema_lengths": self.lengths,
                "key_emas": {
                    "ema1": self.ema1_length,
                    "ema2": self.ema2_length,
                    "ema3": self.ema3_length,
                    "ema8": self.ema8_length,
                },
            },
        )

    def calculate_ema_ribbon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 8 EMAs for the ribbon.

        Args:
            df: DataFrame with OHLC data (must contain 'close' column)

        Returns:
            DataFrame with added EMA columns (ema1 through ema8)
        """
        start_time = time.perf_counter()
        tracemalloc.start()

        logger.debug(
            "Starting EMA ribbon calculation",
            extra={
                "indicator": "ema_ribbon",
                "step": "ema_calculation",
                "data_points": len(df),
                "calculation_id": self._calculation_count,
            },
        )

        if df.empty:
            logger.warning(
                "Empty DataFrame provided for EMA ribbon calculation",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "empty_dataframe",
                    "data_points": 0,
                },
            )
            return df.copy()

        if "close" not in df.columns:
            logger.error(
                "Missing close column for EMA ribbon calculation",
                extra={
                    "indicator": "ema_ribbon",
                    "error_type": "missing_close_column",
                    "available_columns": list(df.columns),
                },
            )
            raise ValueError("DataFrame must contain 'close' column")

        max_length = max(self.lengths)
        min_data_points = (
            max_length * 2
        )  # Require 2x the longest EMA for proper convergence

        if len(df) < min_data_points:
            logger.warning(
                "Insufficient data for reliable EMA ribbon calculation",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "insufficient_data_for_convergence",
                    "data_points": len(df),
                    "max_ema_length": max_length,
                    "recommended_min_points": min_data_points,
                    "shortage": min_data_points - len(df),
                },
            )
            # Still calculate but with warning
        elif len(df) < max_length:
            logger.info(
                "Limited data for EMA ribbon - calculations may have reduced accuracy",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "limited_data",
                    "data_points": len(df),
                    "max_ema_length": max_length,
                    "shortage": max_length - len(df),
                },
            )

        result = df.copy()

        # Data quality validation
        self._validate_input_data_quality(result)

        # Ensure close column is proper float64 dtype
        result["close"] = pd.to_numeric(result["close"], errors="coerce").astype(
            "float64"
        )

        try:
            # Calculate all 8 EMAs
            ema_calculation_times = []
            successful_emas = 0

            for i, length in enumerate(self.lengths, 1):
                ema_start = time.perf_counter()

                logger.debug(
                    "Calculating EMA%s",
                    i,
                    extra={
                        "indicator": "ema_ribbon",
                        "step": f"ema{i}_calculation",
                        "ema_length": length,
                    },
                )

                ema_values = ta.ema(result["close"], length=length)

                if ema_values is not None:
                    result[f"ema{i}"] = ema_values.astype("float64")
                    successful_emas += 1

                    # Validate EMA values with improved thresholds
                    valid_count = (~ema_values.isna()).sum()
                    total_count = len(ema_values)
                    valid_percentage = (
                        (valid_count / total_count) * 100 if total_count > 0 else 0
                    )

                    # Calculate expected convergence point for this EMA length
                    convergence_point = min(
                        length * 3, total_count
                    )  # EMAs typically converge after 3x their period
                    expected_valid_from_convergence = max(
                        0, total_count - convergence_point
                    )
                    expected_valid_percentage = (
                        (expected_valid_from_convergence / total_count) * 100
                        if total_count > 0
                        else 0
                    )

                    # Only warn if valid percentage is significantly below expected
                    if valid_percentage < max(
                        50.0, expected_valid_percentage * 0.8
                    ):  # At least 50% or 80% of expected
                        logger.warning(
                            "Low valid data percentage in EMA%s",
                            i,
                            extra={
                                "indicator": "ema_ribbon",
                                "issue": "low_valid_ema_data",
                                "ema_number": i,
                                "ema_length": length,
                                "valid_count": int(valid_count),
                                "total_count": total_count,
                                "valid_percentage": round(valid_percentage, 2),
                                "expected_valid_percentage": round(
                                    expected_valid_percentage, 2
                                ),
                                "convergence_point": convergence_point,
                            },
                        )
                    else:
                        logger.debug(
                            "EMA%s validation passed",
                            i,
                            extra={
                                "indicator": "ema_ribbon",
                                "ema_number": i,
                                "ema_length": length,
                                "valid_percentage": round(valid_percentage, 2),
                                "convergence_point": convergence_point,
                            },
                        )
                else:
                    result[f"ema{i}"] = pd.Series(dtype="float64", index=df.index)
                    logger.error(
                        "Failed to calculate EMA%s",
                        i,
                        extra={
                            "indicator": "ema_ribbon",
                            "error_type": "ema_calculation_failed",
                            "ema_number": i,
                            "ema_length": length,
                        },
                    )

                ema_duration = (time.perf_counter() - ema_start) * 1000
                ema_calculation_times.append(ema_duration)

            # Performance logging
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self._calculation_count += 1
            self._total_calculation_time += total_duration
            avg_calculation_time = (
                self._total_calculation_time / self._calculation_count
            )

            logger.debug(
                "EMA ribbon calculation completed",
                extra={
                    "indicator": "ema_ribbon",
                    "duration_ms": round(total_duration, 2),
                    "memory_current_mb": round(current / 1024 / 1024, 2),
                    "memory_peak_mb": round(peak / 1024 / 1024, 2),
                    "data_points": len(df),
                    "successful_emas": successful_emas,
                    "total_emas": len(self.lengths),
                    "calculation_count": self._calculation_count,
                    "avg_duration_ms": round(avg_calculation_time, 2),
                    "individual_ema_times_ms": [
                        round(t, 2) for t in ema_calculation_times
                    ],
                },
            )

        except Exception as e:
            logger.exception(
                "EMA ribbon calculation failed with exception",
                extra={
                    "indicator": "ema_ribbon",
                    "error_type": "calculation_exception",
                    "error_message": str(e),
                    "data_points": len(df),
                    "ema_lengths": self.lengths,
                },
            )
            return df.copy()
        else:
            return result

    def calculate_ribbon_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ribbon direction analysis.

        Pine Script logic: bullish when ema8 < ema2 (faster EMA above slower EMA)

        Args:
            df: DataFrame with calculated EMA values

        Returns:
            DataFrame with added ribbon direction columns
        """
        logger.debug(
            "Calculating ribbon direction analysis",
            extra={
                "indicator": "ema_ribbon",
                "step": "ribbon_direction",
                "data_points": len(df),
            },
        )

        result = df.copy()

        # Check if EMAs are calculated
        if "ema2" not in df.columns or "ema8" not in df.columns:
            logger.warning(
                "EMAs not calculated for ribbon direction",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "missing_emas",
                    "missing_columns": [
                        col for col in ["ema2", "ema8"] if col not in df.columns
                    ],
                },
            )
            return result

        try:
            # Ribbon direction: bullish when ema8 < ema2 (shorter period EMA above longer period EMA)
            result["ribbon_bullish"] = result["ema2"] > result["ema8"]
            result["ribbon_bearish"] = result["ema2"] < result["ema8"]
            result["ribbon_neutral"] = result["ema2"] == result["ema8"]

            # Count direction periods
            bullish_count = result["ribbon_bullish"].sum()
            bearish_count = result["ribbon_bearish"].sum()
            neutral_count = result["ribbon_neutral"].sum()

            # Ribbon direction signal: 1 = bullish, -1 = bearish, 0 = neutral
            result["ribbon_direction"] = np.where(
                result["ribbon_bullish"], 1, np.where(result["ribbon_bearish"], -1, 0)
            ).astype("int64")

            # Calculate ribbon trend strength (distance between fastest and slowest EMA)
            if "ema1" in df.columns and "ema8" in df.columns:
                ema_spread = abs(result["ema1"] - result["ema8"])
                # Prevent division by very small close prices
                close_safe = result["close"].where(result["close"] > 1e-8, 1e-8)
                result["ribbon_strength"] = (
                    ema_spread / close_safe * 100
                )  # As percentage

                # Handle flat market conditions
                low_variance_mask = ema_spread < (
                    result["close"] * 1e-6
                )  # Less than 0.0001% of price
                if low_variance_mask.sum() > 0:
                    result.loc[low_variance_mask, "ribbon_strength"] = 0.0
                    logger.debug(
                        "Flat market conditions detected - setting ribbon strength to zero",
                        extra={
                            "indicator": "ema_ribbon",
                            "flat_periods": int(low_variance_mask.sum()),
                            "total_periods": len(result),
                        },
                    )

                avg_strength = result["ribbon_strength"].mean()
                max_strength = result["ribbon_strength"].max()

                logger.debug(
                    "Ribbon strength calculated",
                    extra={
                        "indicator": "ema_ribbon",
                        "avg_strength_pct": (
                            round(float(avg_strength), 4)
                            if not pd.isna(avg_strength)
                            else 0
                        ),
                        "max_strength_pct": (
                            round(float(max_strength), 4)
                            if not pd.isna(max_strength)
                            else 0
                        ),
                    },
                )
            else:
                result["ribbon_strength"] = pd.Series(0.0, index=df.index)
                logger.warning(
                    "Cannot calculate ribbon strength: missing EMA1 or EMA8",
                    extra={
                        "indicator": "ema_ribbon",
                        "issue": "missing_emas_for_strength",
                        "has_ema1": "ema1" in df.columns,
                        "has_ema8": "ema8" in df.columns,
                    },
                )

            logger.debug(
                "Ribbon direction analysis completed",
                extra={
                    "indicator": "ema_ribbon",
                    "bullish_periods": int(bullish_count),
                    "bearish_periods": int(bearish_count),
                    "neutral_periods": int(neutral_count),
                    "total_periods": len(result),
                    "bullish_percentage": (
                        round((bullish_count / len(result)) * 100, 2)
                        if len(result) > 0
                        else 0
                    ),
                    "bearish_percentage": (
                        round((bearish_count / len(result)) * 100, 2)
                        if len(result) > 0
                        else 0
                    ),
                },
            )

        except Exception as e:
            logger.exception(
                "Ribbon direction calculation failed",
                extra={
                    "indicator": "ema_ribbon",
                    "error_type": "direction_calculation_failed",
                    "error_message": str(e),
                },
            )
            return result

        return result

    def calculate_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate long/short EMA crossover signals.

        Pine Script logic:
        - longEma = crossover(ema2, ema8)
        - shortEma = crossover(ema8, ema2)

        Args:
            df: DataFrame with calculated EMA values

        Returns:
            DataFrame with added crossover signal columns
        """
        logger.debug(
            "Calculating EMA crossover signals",
            extra={
                "indicator": "ema_ribbon",
                "step": "crossover_signals",
                "data_points": len(df),
            },
        )

        result = df.copy()

        # Check if EMAs are calculated
        if "ema2" not in df.columns or "ema8" not in df.columns:
            logger.warning(
                "EMAs not calculated for crossover signals",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "missing_emas_for_crossover",
                    "missing_columns": [
                        col for col in ["ema2", "ema8"] if col not in df.columns
                    ],
                },
            )
            return result

        try:
            # Calculate crossovers
            result["long_ema_signal"] = self._detect_crossover(
                result["ema2"], result["ema8"]
            )
            result["short_ema_signal"] = self._detect_crossover(
                result["ema8"], result["ema2"]
            )

            # Count signals
            long_signal_count = result["long_ema_signal"].sum()
            short_signal_count = result["short_ema_signal"].sum()

            # Combined signal: 1 = long, -1 = short, 0 = no signal
            result["ema_crossover_signal"] = np.where(
                result["long_ema_signal"],
                1,
                np.where(result["short_ema_signal"], -1, 0),
            ).astype("int64")

            # Log signal generation events
            if long_signal_count > 0:
                logger.debug(
                    "EMA long crossover signals generated",
                    extra={
                        "indicator": "ema_ribbon",
                        "signal_type": "long_ema_crossover",
                        "signal_count": int(long_signal_count),
                        "signal_frequency": (
                            round((long_signal_count / len(df)) * 100, 2)
                            if len(df) > 0
                            else 0
                        ),
                    },
                )

            if short_signal_count > 0:
                logger.debug(
                    "EMA short crossover signals generated",
                    extra={
                        "indicator": "ema_ribbon",
                        "signal_type": "short_ema_crossover",
                        "signal_count": int(short_signal_count),
                        "signal_frequency": (
                            round((short_signal_count / len(df)) * 100, 2)
                            if len(df) > 0
                            else 0
                        ),
                    },
                )

            logger.debug(
                "EMA crossover signals calculated",
                extra={
                    "indicator": "ema_ribbon",
                    "long_signals": int(long_signal_count),
                    "short_signals": int(short_signal_count),
                    "total_crossover_signals": int(
                        long_signal_count + short_signal_count
                    ),
                },
            )

        except Exception as e:
            logger.exception(
                "EMA crossover signal calculation failed",
                extra={
                    "indicator": "ema_ribbon",
                    "error_type": "crossover_calculation_failed",
                    "error_message": str(e),
                },
            )
            return result

        return result

    def calculate_cross_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate red/green cross patterns.

        Pine Script logic:
        - redCross = crossunder(ema1, ema2)
        - greenCross = crossunder(ema2, ema1)

        Args:
            df: DataFrame with calculated EMA values

        Returns:
            DataFrame with added cross pattern columns
        """
        result = df.copy()

        # Check if EMAs are calculated
        if "ema1" not in df.columns or "ema2" not in df.columns:
            logger.warning("EMAs not calculated. Run calculate_ema_ribbon first.")
            return result

        # Calculate cross patterns
        result["red_cross"] = self._detect_crossunder(result["ema1"], result["ema2"])
        result["green_cross"] = self._detect_crossunder(result["ema2"], result["ema1"])

        # Combined cross signal: 1 = green cross (bullish), -1 = red cross (bearish), 0 = no signal
        result["cross_pattern_signal"] = np.where(
            result["green_cross"], 1, np.where(result["red_cross"], -1, 0)
        ).astype("int64")

        return result

    def calculate_triangle_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate blue triangle signals.

        Pine Script logic:
        - blueTriangleUp = crossover(ema2, ema3)
        - blueTriangleDown = crossover(ema3, ema2)

        Args:
            df: DataFrame with calculated EMA values

        Returns:
            DataFrame with added triangle signal columns
        """
        result = df.copy()

        # Check if EMAs are calculated
        if "ema2" not in df.columns or "ema3" not in df.columns:
            logger.warning("EMAs not calculated. Run calculate_ema_ribbon first.")
            return result

        # Calculate triangle signals
        result["blue_triangle_up"] = self._detect_crossover(
            result["ema2"], result["ema3"]
        )
        result["blue_triangle_down"] = self._detect_crossover(
            result["ema3"], result["ema2"]
        )

        # Combined triangle signal: 1 = up (bullish), -1 = down (bearish), 0 = no signal
        result["triangle_signal"] = np.where(
            result["blue_triangle_up"], 1, np.where(result["blue_triangle_down"], -1, 0)
        ).astype("int64")

        return result

    def get_ribbon_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive ribbon state analysis.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with complete EMA ribbon analysis
        """
        start_time = time.perf_counter()

        logger.info(
            "Starting comprehensive EMA ribbon analysis",
            extra={
                "indicator": "ema_ribbon",
                "input_data_points": len(df),
                "analysis_components": [
                    "ema_calculation",
                    "direction_analysis",
                    "crossover_signals",
                    "cross_patterns",
                    "triangle_signals",
                    "overall_signal",
                    "flat_market_handling",
                ],
            },
        )

        try:
            # Calculate all EMAs
            logger.debug(
                "Step 1: Calculating EMAs",
                extra={"indicator": "ema_ribbon", "step": "ema_calculation"},
            )
            result = self.calculate_ema_ribbon(df)

            # Add ribbon direction analysis
            logger.debug(
                "Step 2: Analyzing ribbon direction",
                extra={"indicator": "ema_ribbon", "step": "direction_analysis"},
            )
            result = self.calculate_ribbon_direction(result)

            # Add all signal types
            logger.debug(
                "Step 3: Calculating crossover signals",
                extra={"indicator": "ema_ribbon", "step": "crossover_signals"},
            )
            result = self.calculate_crossover_signals(result)

            logger.debug(
                "Step 4: Calculating cross patterns",
                extra={"indicator": "ema_ribbon", "step": "cross_patterns"},
            )
            result = self.calculate_cross_patterns(result)

            logger.debug(
                "Step 5: Calculating triangle signals",
                extra={"indicator": "ema_ribbon", "step": "triangle_signals"},
            )
            result = self.calculate_triangle_signals(result)

            # Add overall ribbon signal (combination of all signals)
            logger.debug(
                "Step 6: Calculating overall signal",
                extra={"indicator": "ema_ribbon", "step": "overall_signal"},
            )
            result = self._calculate_overall_signal(result)

            # Handle flat market conditions
            logger.debug(
                "Step 7: Handling flat market conditions",
                extra={"indicator": "ema_ribbon", "step": "flat_market_handling"},
            )
            result = self._handle_flat_market_conditions(result)

            # Performance and summary logging
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000

            # Count different signal types
            signal_summary = self._generate_signal_summary(result)

            logger.info(
                "Comprehensive EMA ribbon analysis completed",
                extra={
                    "indicator": "ema_ribbon",
                    "duration_ms": round(total_duration, 2),
                    "input_data_points": len(df),
                    "output_data_points": len(result),
                    "signal_summary": signal_summary,
                    "analysis_success": True,
                },
            )

        except Exception as e:
            logger.exception(
                "Comprehensive EMA ribbon analysis failed",
                extra={
                    "indicator": "ema_ribbon",
                    "error_type": "comprehensive_analysis_failed",
                    "error_message": str(e),
                    "input_data_points": len(df),
                },
            )
            return df.copy()

        return result

    def _calculate_overall_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall ribbon signal combining all individual signals.

        Args:
            df: DataFrame with all calculated signals

        Returns:
            DataFrame with added overall signal column
        """
        result = df.copy()

        # Weight different signals (crossover signals are most important)
        crossover_weight = 0.4
        pattern_weight = 0.3
        triangle_weight = 0.2
        direction_weight = 0.1

        # Calculate weighted signal
        weighted_signal = (
            result.get("ema_crossover_signal", 0) * crossover_weight
            + result.get("cross_pattern_signal", 0) * pattern_weight
            + result.get("triangle_signal", 0) * triangle_weight
            + result.get("ribbon_direction", 0) * direction_weight
        )

        # Convert to discrete signal: 1 = bullish, -1 = bearish, 0 = neutral
        result["ribbon_overall_signal"] = np.where(
            weighted_signal > 0.3, 1, np.where(weighted_signal < -0.3, -1, 0)
        ).astype("int64")

        return result

    def _detect_crossover(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect crossover events (series1 crosses above series2).

        Args:
            series1: First series
            series2: Second series

        Returns:
            Boolean series indicating crossover points
        """
        if len(series1) < 2 or len(series2) < 2:
            return pd.Series(False, index=series1.index)

        # Previous values
        prev_series1 = series1.shift(1)
        prev_series2 = series2.shift(1)

        # Crossover: was below, now above
        crossover = (prev_series1 <= prev_series2) & (series1 > series2)

        return crossover.fillna(False)

    def _detect_crossunder(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect crossunder events (series1 crosses below series2).

        Args:
            series1: First series
            series2: Second series

        Returns:
            Boolean series indicating crossunder points
        """
        if len(series1) < 2 or len(series2) < 2:
            return pd.Series(False, index=series1.index)

        # Previous values
        prev_series1 = series1.shift(1)
        prev_series2 = series2.shift(1)

        # Crossunder: was above, now below
        crossunder = (prev_series1 >= prev_series2) & (series1 < series2)

        return crossunder.fillna(False)

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get the latest EMA ribbon values and signals.

        Args:
            df: DataFrame with calculated EMA ribbon indicators

        Returns:
            Dictionary with latest indicator values and signals
        """
        if df.empty:
            logger.warning(
                "Empty DataFrame provided for latest values",
                extra={"indicator": "ema_ribbon", "issue": "empty_dataframe"},
            )
            return {}

        latest = df.iloc[-1]

        # Base EMA values
        values = {
            "timestamp": latest.name if hasattr(latest, "name") else None,
        }

        # Add all EMA values
        for i in range(1, 9):
            ema_col = f"ema{i}"
            if ema_col in df.columns:
                values[ema_col] = latest.get(ema_col)

        # Add direction analysis
        direction_cols = [
            "ribbon_bullish",
            "ribbon_bearish",
            "ribbon_neutral",
            "ribbon_direction",
            "ribbon_strength",
        ]
        for col in direction_cols:
            if col in df.columns:
                values[col] = latest.get(col)

        # Add signal values
        signal_cols = [
            "long_ema_signal",
            "short_ema_signal",
            "ema_crossover_signal",
            "red_cross",
            "green_cross",
            "cross_pattern_signal",
            "blue_triangle_up",
            "blue_triangle_down",
            "triangle_signal",
            "ribbon_overall_signal",
        ]
        for col in signal_cols:
            if col in df.columns:
                values[col] = latest.get(col)

        # Log any active signals
        active_signals = []
        if values.get("long_ema_signal"):
            active_signals.append("long_ema_signal")
        if values.get("short_ema_signal"):
            active_signals.append("short_ema_signal")
        if values.get("green_cross"):
            active_signals.append("green_cross")
        if values.get("red_cross"):
            active_signals.append("red_cross")
        if values.get("blue_triangle_up"):
            active_signals.append("blue_triangle_up")
        if values.get("blue_triangle_down"):
            active_signals.append("blue_triangle_down")

        if active_signals:
            logger.info(
                "Active EMA ribbon signals detected",
                extra={
                    "indicator": "ema_ribbon",
                    "signal_type": "current_signals",
                    "active_signals": active_signals,
                    "ribbon_direction": values.get("ribbon_direction", 0),
                    "ribbon_strength": (
                        float(values.get("ribbon_strength", 0) or 0)
                        if values.get("ribbon_strength") is not None
                        else 0.0
                    ),
                    "overall_signal": values.get("ribbon_overall_signal", 0),
                },
            )

        return values

    def validate_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate data for EMA ribbon calculation.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df.empty:
            return False, "DataFrame is empty"

        if "close" not in df.columns:
            return False, "Missing required 'close' column"

        max_length = max(self.lengths)
        min_recommended = max_length * 2  # Recommend 2x for good convergence

        if len(df) < max_length:
            return (
                False,
                f"Insufficient data. Need at least {max_length} rows, got {len(df)}",
            )
        if len(df) < min_recommended:
            logger.info(
                "Limited data for optimal EMA convergence. Have %s rows, recommend %s for best results",
                len(df),
                min_recommended,
                extra={
                    "indicator": "ema_ribbon",
                    "data_points": len(df),
                    "recommended_minimum": min_recommended,
                },
            )

        # Check for valid close values
        try:
            if not pd.api.types.is_numeric_dtype(df["close"]):
                return False, "'close' column is not numeric"

            if df["close"].isna().all():
                return False, "'close' column contains only NaN values"

            if (df["close"] <= 0).any():
                return False, "'close' column contains non-positive values"

            # Check for minimum price variance to avoid flat market issues
            price_variance = df["close"].var()
            if price_variance < 1e-10:  # Extremely small variance
                logger.warning(
                    "Very low price variance detected - EMA calculations may be less meaningful",
                    extra={
                        "indicator": "ema_ribbon",
                        "issue": "low_price_variance",
                        "variance": float(price_variance),
                        "price_range": float(df["close"].max() - df["close"].min()),
                    },
                )

        except Exception as e:
            return False, f"Data validation error: {e!s}"

        return True, ""

    def get_signal_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get a summary of all current signals.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with signal summary
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        summary = {
            "ribbon_direction": (
                "bullish"
                if latest.get("ribbon_direction", 0) > 0
                else "bearish"
                if latest.get("ribbon_direction", 0) < 0
                else "neutral"
            ),
            "ribbon_strength": latest.get("ribbon_strength", 0),
            "signals": {
                "ema_crossover": latest.get("ema_crossover_signal", 0),
                "cross_pattern": latest.get("cross_pattern_signal", 0),
                "triangle": latest.get("triangle_signal", 0),
                "overall": latest.get("ribbon_overall_signal", 0),
            },
            "active_signals": [],
        }

        # Identify active signals
        if latest.get("long_ema_signal", False):
            summary["active_signals"].append("Long EMA Crossover")
        if latest.get("short_ema_signal", False):
            summary["active_signals"].append("Short EMA Crossover")
        if latest.get("green_cross", False):
            summary["active_signals"].append("Green Cross")
        if latest.get("red_cross", False):
            summary["active_signals"].append("Red Cross")
        if latest.get("blue_triangle_up", False):
            summary["active_signals"].append("Blue Triangle Up")
        if latest.get("blue_triangle_down", False):
            summary["active_signals"].append("Blue Triangle Down")

        return summary

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
                "NaN values in EMA ribbon input data",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "input_nan_values",
                    "nan_count": int(nan_count),
                    "total_points": len(close_series),
                    "nan_percentage": round((nan_count / len(close_series)) * 100, 2),
                },
            )

        # Check for zero or negative prices
        invalid_prices = (close_series <= 0).sum()
        if invalid_prices > 0:
            logger.warning(
                "Invalid price values in EMA ribbon input data",
                extra={
                    "indicator": "ema_ribbon",
                    "issue": "invalid_prices",
                    "invalid_count": int(invalid_prices),
                    "min_price": (
                        float(close_series.min()) if not close_series.empty else 0
                    ),
                },
            )

        # Check for extreme price changes
        if len(close_series) > 1:
            price_changes = close_series.pct_change().abs()
            extreme_changes = (price_changes > 0.2).sum()  # More than 20% change
            if extreme_changes > 0:
                logger.warning(
                    "Extreme price changes in EMA ribbon input data",
                    extra={
                        "indicator": "ema_ribbon",
                        "issue": "extreme_price_changes",
                        "extreme_change_count": int(extreme_changes),
                        "max_change_pct": (
                            round(float(price_changes.max()) * 100, 2)
                            if not price_changes.empty
                            else 0
                        ),
                    },
                )

    def _generate_signal_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate summary of signal counts for logging.

        Args:
            df: DataFrame with calculated signals

        Returns:
            Dictionary with signal counts and statistics
        """
        summary: dict[str, Any] = {"total_data_points": len(df)}

        # Count different signal types
        signal_columns = [
            "long_ema_signal",
            "short_ema_signal",
            "ema_crossover_signal",
            "red_cross",
            "green_cross",
            "cross_pattern_signal",
            "blue_triangle_up",
            "blue_triangle_down",
            "triangle_signal",
            "ribbon_overall_signal",
        ]

        for col in signal_columns:
            if col in df.columns:
                if df[col].dtype == bool:
                    summary[f"{col}_count"] = int(df[col].sum())
                else:
                    # For integer signals, count non-zero values
                    summary[f"{col}_count"] = int((df[col] != 0).sum())

        # Direction analysis
        if "ribbon_direction" in df.columns:
            bullish_periods = (df["ribbon_direction"] == 1).sum()
            bearish_periods = (df["ribbon_direction"] == -1).sum()
            neutral_periods = (df["ribbon_direction"] == 0).sum()

            summary.update(
                {
                    "bullish_periods": int(bullish_periods),
                    "bearish_periods": int(bearish_periods),
                    "neutral_periods": int(neutral_periods),
                }
            )

        # Ribbon strength statistics
        if "ribbon_strength" in df.columns:
            strength_series = df["ribbon_strength"].dropna()
            if not strength_series.empty:
                summary.update(
                    {
                        "avg_ribbon_strength": round(float(strength_series.mean()), 4),
                        "max_ribbon_strength": round(float(strength_series.max()), 4),
                        "min_ribbon_strength": round(float(strength_series.min()), 4),
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
            "ema_lengths": self.lengths,
            "last_data_quality_check": self._last_data_quality_check,
        }

        logger.debug(
            "EMA ribbon performance metrics",
            extra={"indicator": "ema_ribbon", "metrics": metrics},
        )

        return metrics

    def _handle_flat_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle flat market conditions by adjusting signals appropriately.

        Args:
            df: DataFrame with calculated EMA values

        Returns:
            DataFrame with flat market conditions handled
        """
        result = df.copy()

        if "close" not in result.columns:
            return result

        # Detect flat market periods
        price_changes = result["close"].pct_change().abs()
        flat_threshold = 1e-5  # 0.001% change threshold
        flat_mask = price_changes < flat_threshold

        # Count consecutive flat periods
        consecutive_flat = flat_mask.groupby((~flat_mask).cumsum()).cumsum()
        long_flat_mask = consecutive_flat > 10  # More than 10 consecutive flat periods

        if long_flat_mask.sum() > 0:
            logger.debug(
                "Long flat market periods detected in EMA ribbon - suppressing signals",
                extra={
                    "indicator": "ema_ribbon",
                    "flat_periods": int(long_flat_mask.sum()),
                    "total_periods": len(result),
                },
            )

            # Suppress signals during flat periods
            signal_columns = [
                "long_ema_signal",
                "short_ema_signal",
                "ema_crossover_signal",
                "red_cross",
                "green_cross",
                "cross_pattern_signal",
                "blue_triangle_up",
                "blue_triangle_down",
                "triangle_signal",
                "ribbon_overall_signal",
            ]

            for col in signal_columns:
                if col in result.columns:
                    if result[col].dtype == bool:
                        result.loc[long_flat_mask, col] = False
                    else:
                        result.loc[long_flat_mask, col] = 0

        return result
