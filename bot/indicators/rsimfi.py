"""
RSI+MFI Combined Indicator implementation.

This module provides a Python implementation of the custom RSI+MFI hybrid indicator
from VuManChu Cipher indicators, originally written in Pine Script. The formula is:
f_rsimfi(_period, _multiplier, _tf) =>
    security(syminfo.tickerid, _tf, sma(((close - open) / (high - low)) * _multiplier, _period))

This differs from standard MFI as it uses a custom price ratio calculation
combined with a simple moving average, rather than the traditional MFI formula.
"""

import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RSIMFIIndicator:
    """
    RSI+MFI Combined Indicator from VuManChu Cipher indicators.

    This implements the custom RSI+MFI formula used in both Cipher A and B:
    - Price ratio: (close - open) / (high - low)
    - Multiplied by sensitivity factor
    - Smoothed with simple moving average
    - Offset by position adjustment (pos_y)

    This is NOT the standard MFI indicator, but a custom hybrid oscillator
    that combines price action and momentum characteristics.
    """

    def __init__(
        self,
        rsi_length: int = 14,
        mfi_length: int = 14,
        period: int = 60,
        multiplier: float = 150.0,
    ) -> None:
        """
        Initialize the RSI+MFI indicator calculator.

        Args:
            rsi_length: RSI calculation period (used for minimum data requirements)
            mfi_length: MFI calculation period (used for minimum data requirements)
            period: Default period for the SMA calculation
            multiplier: Default sensitivity multiplier for the price ratio
        """
        # Store indicator parameters for compatibility with other components
        self.rsi_length = rsi_length
        self.mfi_length = mfi_length
        self.period = period
        self.multiplier = multiplier

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._last_data_quality_check = None

        logger.info(
            "RSI+MFI indicator initialized",
            extra={
                "indicator": "rsimfi",
                "description": "Custom RSI+MFI hybrid oscillator from VuManChu Cipher",
                "parameters": {
                    "rsi_length": rsi_length,
                    "mfi_length": mfi_length,
                    "period": period,
                    "multiplier": multiplier,
                },
            },
        )

    def calculate_rsimfi(
        self,
        df: pd.DataFrame,
        period: int | None = None,
        multiplier: float | None = None,
        pos_y: float = 0.0,
    ) -> pd.Series:
        """
        Calculate the RSI+MFI combined indicator.

        Pine Script formula:
        f_rsimfi(_period, _multiplier, _tf) =>
            security(syminfo.tickerid, _tf, sma(((close - open) / (high - low)) * _multiplier, _period))

        Args:
            df: DataFrame with OHLC data (columns: open, high, low, close)
            period: Period for the simple moving average (None = use default from __init__)
            multiplier: Sensitivity multiplier for the price ratio (None = use default from __init__)
            pos_y: Y-position offset for display purposes (default: 0.0)

        Returns:
            Series with RSI+MFI values

        Raises:
            ValueError: If required OHLC columns are missing
            ValueError: If period is less than 1
        """
        # Use default parameters if None provided
        period = period if period is not None else self.period
        multiplier = multiplier if multiplier is not None else self.multiplier

        start_time = time.perf_counter()
        tracemalloc.start()

        logger.debug(
            "Starting RSI+MFI calculation",
            extra={
                "indicator": "rsimfi",
                "step": "rsimfi_calculation",
                "data_points": len(df),
                "period": period,
                "multiplier": multiplier,
                "pos_y": pos_y,
                "calculation_id": self._calculation_count,
            },
        )

        if period < 1:
            logger.error(
                "Invalid period for RSI+MFI calculation",
                extra={
                    "indicator": "rsimfi",
                    "error_type": "invalid_period",
                    "period": period,
                },
            )
            raise ValueError("Period must be at least 1")

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                "Missing required columns for RSI+MFI calculation",
                extra={
                    "indicator": "rsimfi",
                    "error_type": "missing_columns",
                    "missing_columns": missing_columns,
                    "available_columns": list(df.columns),
                },
            )
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(df) < period:
            logger.warning(
                "Insufficient data for RSI+MFI calculation",
                extra={
                    "indicator": "rsimfi",
                    "issue": "insufficient_data",
                    "data_points": len(df),
                    "required_period": period,
                    "shortage": period - len(df),
                },
            )
            return pd.Series(dtype="float64", index=df.index)

        # Data quality validation
        self._validate_input_data_quality(df)

        try:
            # Ensure proper data types
            df_clean = df.copy()
            for col in required_columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype(
                    "float64"
                )

            logger.debug(
                "Calculating price ratio components",
                extra={"indicator": "rsimfi", "step": "price_ratio_calculation"},
            )

            # Calculate price ratio: (close - open) / (high - low)
            # Handle division by zero where high == low (rare but possible)
            high_low_diff = df_clean["high"] - df_clean["low"]
            close_open_diff = df_clean["close"] - df_clean["open"]

            # Check for zero high-low differences
            zero_hl_count = (high_low_diff == 0).sum()
            if zero_hl_count > 0:
                logger.warning(
                    "Zero high-low differences detected in RSI+MFI data",
                    extra={
                        "indicator": "rsimfi",
                        "issue": "zero_high_low_diff",
                        "zero_count": int(zero_hl_count),
                        "total_points": len(df_clean),
                        "zero_percentage": round(
                            (zero_hl_count / len(df_clean)) * 100, 2
                        ),
                    },
                )

            # Replace zero high-low differences with small epsilon to avoid division by zero
            high_low_diff = high_low_diff.replace(0, np.finfo(float).eps)

            price_ratio = close_open_diff / high_low_diff

            # Validate price ratio values
            ratio_nan_count = price_ratio.isna().sum()
            ratio_inf_count = np.isinf(price_ratio).sum()
            if ratio_nan_count > 0 or ratio_inf_count > 0:
                logger.warning(
                    "Invalid price ratio values detected",
                    extra={
                        "indicator": "rsimfi",
                        "issue": "invalid_price_ratios",
                        "nan_count": int(ratio_nan_count),
                        "inf_count": int(ratio_inf_count),
                    },
                )

            # Apply multiplier
            logger.debug(
                "Applying multiplier to price ratio",
                extra={
                    "indicator": "rsimfi",
                    "step": "multiplier_application",
                    "multiplier": multiplier,
                },
            )
            weighted_ratio = price_ratio * multiplier

            # Apply simple moving average
            logger.debug(
                "Calculating simple moving average",
                extra={
                    "indicator": "rsimfi",
                    "step": "sma_calculation",
                    "period": period,
                },
            )
            rsimfi_values = weighted_ratio.rolling(window=period, min_periods=1).mean()

            # Apply position offset
            if pos_y != 0.0:
                logger.debug(
                    "Applying position offset",
                    extra={
                        "indicator": "rsimfi",
                        "step": "position_offset",
                        "pos_y": pos_y,
                    },
                )
            rsimfi_values = rsimfi_values - pos_y

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
            valid_count = (~rsimfi_values.isna()).sum()

            # Calculate value statistics
            value_stats = {
                "min": float(rsimfi_values.min()) if valid_count > 0 else 0,
                "max": float(rsimfi_values.max()) if valid_count > 0 else 0,
                "mean": float(rsimfi_values.mean()) if valid_count > 0 else 0,
                "std": float(rsimfi_values.std()) if valid_count > 0 else 0,
            }

            logger.debug(
                "RSI+MFI calculation completed",
                extra={
                    "indicator": "rsimfi",
                    "duration_ms": round(calculation_duration, 2),
                    "memory_current_mb": round(current / 1024 / 1024, 2),
                    "memory_peak_mb": round(peak / 1024 / 1024, 2),
                    "data_points": len(df),
                    "valid_count": int(valid_count),
                    "calculation_count": self._calculation_count,
                    "avg_duration_ms": round(avg_calculation_time, 2),
                    "value_statistics": value_stats,
                    "parameters": {
                        "period": period,
                        "multiplier": multiplier,
                        "pos_y": pos_y,
                    },
                },
            )

            return rsimfi_values.astype("float64")

        except Exception as e:
            logger.error(
                "RSI+MFI calculation failed with exception",
                extra={
                    "indicator": "rsimfi",
                    "error_type": "calculation_exception",
                    "error_message": str(e),
                    "data_points": len(df),
                    "parameters": {
                        "period": period,
                        "multiplier": multiplier,
                        "pos_y": pos_y,
                    },
                },
            )
            return pd.Series(dtype="float64", index=df.index)

    def get_color_conditions(self, rsimfi_values: pd.Series) -> pd.DataFrame:
        """
        Get color conditions for RSI+MFI visualization.

        Args:
            rsimfi_values: Series with RSI+MFI values

        Returns:
            DataFrame with color condition columns:
            - is_positive: True when RSI+MFI > 0 (bullish)
            - is_negative: True when RSI+MFI < 0 (bearish)
            - color_signal: 1 for bullish, -1 for bearish, 0 for neutral
        """
        logger.debug(
            "Calculating RSI+MFI color conditions",
            extra={
                "indicator": "rsimfi",
                "step": "color_conditions",
                "data_points": len(rsimfi_values),
            },
        )

        if rsimfi_values.empty:
            logger.warning(
                "Empty RSI+MFI values for color conditions",
                extra={"indicator": "rsimfi", "issue": "empty_values"},
            )
            return pd.DataFrame(index=rsimfi_values.index)

        try:
            conditions = pd.DataFrame(index=rsimfi_values.index)

            conditions["is_positive"] = rsimfi_values > 0
            conditions["is_negative"] = rsimfi_values < 0

            # Color signal: 1 = bullish (green), -1 = bearish (red), 0 = neutral
            conditions["color_signal"] = np.where(
                rsimfi_values > 0, 1, np.where(rsimfi_values < 0, -1, 0)
            ).astype("int64")

            # Count conditions for logging
            positive_count = conditions["is_positive"].sum()
            negative_count = conditions["is_negative"].sum()
            neutral_count = len(conditions) - positive_count - negative_count

            logger.debug(
                "RSI+MFI color conditions calculated",
                extra={
                    "indicator": "rsimfi",
                    "positive_count": int(positive_count),
                    "negative_count": int(negative_count),
                    "neutral_count": int(neutral_count),
                    "total_points": len(conditions),
                    "positive_percentage": (
                        round((positive_count / len(conditions)) * 100, 2)
                        if len(conditions) > 0
                        else 0
                    ),
                    "negative_percentage": (
                        round((negative_count / len(conditions)) * 100, 2)
                        if len(conditions) > 0
                        else 0
                    ),
                },
            )

            return conditions

        except Exception as e:
            logger.error(
                "Failed to calculate RSI+MFI color conditions",
                extra={
                    "indicator": "rsimfi",
                    "error_type": "color_conditions_failed",
                    "error_message": str(e),
                },
            )
            return pd.DataFrame(index=rsimfi_values.index)

    def get_area_conditions(
        self,
        rsimfi_values: pd.Series,
        bear_level: float = 0.0,
        bull_level: float = 0.0,
    ) -> pd.DataFrame:
        """
        Get area-based conditions for RSI+MFI analysis.

        Args:
            rsimfi_values: Series with RSI+MFI values
            bear_level: Level below which conditions are considered bearish
            bull_level: Level above which conditions are considered bullish

        Returns:
            DataFrame with area condition columns:
            - in_bear_area: True when below bear_level
            - in_bull_area: True when above bull_level
            - in_neutral_area: True when between levels
            - area_signal: 1 for bull area, -1 for bear area, 0 for neutral
        """
        if rsimfi_values.empty:
            return pd.DataFrame(index=rsimfi_values.index)

        conditions = pd.DataFrame(index=rsimfi_values.index)

        conditions["in_bear_area"] = rsimfi_values < bear_level
        conditions["in_bull_area"] = rsimfi_values > bull_level
        conditions["in_neutral_area"] = (rsimfi_values >= bear_level) & (
            rsimfi_values <= bull_level
        )

        # Area signal: 1 = bull area, -1 = bear area, 0 = neutral area
        conditions["area_signal"] = np.where(
            rsimfi_values > bull_level, 1, np.where(rsimfi_values < bear_level, -1, 0)
        ).astype("int64")

        return conditions

    def calculate_with_analysis(
        self,
        df: pd.DataFrame,
        period: int | None = None,
        multiplier: float | None = None,
        pos_y: float = 0.0,
        bear_level: float = 0.0,
        bull_level: float = 0.0,
    ) -> pd.DataFrame:
        """
        Calculate RSI+MFI with full analysis including color and area conditions.

        Args:
            df: DataFrame with OHLC data
            period: Period for SMA calculation (None = use default from __init__)
            multiplier: Sensitivity multiplier (None = use default from __init__)
            pos_y: Y-position offset
            bear_level: Bearish area threshold
            bull_level: Bullish area threshold

        Returns:
            DataFrame with RSI+MFI values and analysis columns
        """
        # Use default parameters if None provided
        period = period if period is not None else self.period
        multiplier = multiplier if multiplier is not None else self.multiplier

        start_time = time.perf_counter()

        logger.info(
            "Starting RSI+MFI comprehensive analysis",
            extra={
                "indicator": "rsimfi",
                "input_data_points": len(df),
                "parameters": {
                    "period": period,
                    "multiplier": multiplier,
                    "pos_y": pos_y,
                    "bear_level": bear_level,
                    "bull_level": bull_level,
                },
            },
        )

        try:
            result = df.copy()

            # Calculate RSI+MFI values
            logger.debug(
                "Step 1: Calculating RSI+MFI values",
                extra={"indicator": "rsimfi", "step": "rsimfi_calculation"},
            )
            rsimfi_values = self.calculate_rsimfi(df, period, multiplier, pos_y)
            result["rsimfi"] = rsimfi_values

            # Add color conditions
            logger.debug(
                "Step 2: Calculating color conditions",
                extra={"indicator": "rsimfi", "step": "color_conditions"},
            )
            color_conditions = self.get_color_conditions(rsimfi_values)
            result = pd.concat([result, color_conditions], axis=1)

            # Add area conditions
            logger.debug(
                "Step 3: Calculating area conditions",
                extra={"indicator": "rsimfi", "step": "area_conditions"},
            )
            area_conditions = self.get_area_conditions(
                rsimfi_values, bear_level, bull_level
            )
            result = pd.concat([result, area_conditions], axis=1)

            # Performance and summary logging
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000

            # Generate analysis summary
            analysis_summary = self._generate_analysis_summary(result)

            logger.info(
                "RSI+MFI comprehensive analysis completed",
                extra={
                    "indicator": "rsimfi",
                    "duration_ms": round(total_duration, 2),
                    "input_data_points": len(df),
                    "output_data_points": len(result),
                    "analysis_summary": analysis_summary,
                    "analysis_success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "RSI+MFI comprehensive analysis failed",
                extra={
                    "indicator": "rsimfi",
                    "error_type": "comprehensive_analysis_failed",
                    "error_message": str(e),
                    "input_data_points": len(df),
                },
            )
            return df.copy()

    def get_latest_values(
        self, df: pd.DataFrame
    ) -> dict[str, float | bool | int | None]:
        """
        Get the latest RSI+MFI values and conditions.

        Args:
            df: DataFrame with calculated RSI+MFI indicators

        Returns:
            Dictionary with latest indicator values and conditions
        """
        if df.empty:
            logger.warning(
                "Empty DataFrame provided for latest values",
                extra={"indicator": "rsimfi", "issue": "empty_dataframe"},
            )
            return {}

        latest = df.iloc[-1]

        # Base values
        values = {
            "rsimfi": latest.get("rsimfi"),
            "timestamp": latest.name if hasattr(latest, "name") else None,
        }

        # Color conditions (if available)
        if "is_positive" in df.columns:
            values.update(
                {
                    "is_positive": latest.get("is_positive"),
                    "is_negative": latest.get("is_negative"),
                    "color_signal": latest.get("color_signal"),
                }
            )

        # Area conditions (if available)
        if "in_bear_area" in df.columns:
            values.update(
                {
                    "in_bear_area": latest.get("in_bear_area"),
                    "in_bull_area": latest.get("in_bull_area"),
                    "in_neutral_area": latest.get("in_neutral_area"),
                    "area_signal": latest.get("area_signal"),
                }
            )

        # Log any active signals or conditions
        active_conditions = []
        if values.get("is_positive"):
            active_conditions.append("positive")
        if values.get("is_negative"):
            active_conditions.append("negative")
        if values.get("in_bull_area"):
            active_conditions.append("bull_area")
        if values.get("in_bear_area"):
            active_conditions.append("bear_area")

        if active_conditions or values.get("rsimfi") is not None:
            logger.info(
                "RSI+MFI current state",
                extra={
                    "indicator": "rsimfi",
                    "signal_type": "current_state",
                    "rsimfi_value": (
                        float(values["rsimfi"])
                        if values["rsimfi"] is not None
                        else None
                    ),
                    "active_conditions": active_conditions,
                    "color_signal": values.get("color_signal", 0),
                    "area_signal": values.get("area_signal", 0),
                },
            )

        return values

    def validate_ohlc_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate OHLC data for RSI+MFI calculation.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df.empty:
            return False, "DataFrame is empty"

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check for valid OHLC relationships
        try:
            # Convert to numeric and check for valid values
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False, f"Column {col} is not numeric"

                if df[col].isna().all():
                    return False, f"Column {col} contains only NaN values"

            # Check OHLC relationships (high >= low, etc.)
            invalid_rows = (df["high"] < df["low"]).sum()
            if invalid_rows > 0:
                return False, f"Found {invalid_rows} rows where high < low"

        except Exception as e:
            return False, f"Data validation error: {str(e)}"

        return True, ""

    def _validate_input_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate input data quality and log issues.

        Args:
            df: Input DataFrame to validate
        """
        required_columns = ["open", "high", "low", "close"]

        for col in required_columns:
            if col not in df.columns:
                continue

            col_series = df[col]

            # Check for NaN values
            nan_count = col_series.isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"NaN values in {col} for RSI+MFI calculation",
                    extra={
                        "indicator": "rsimfi",
                        "issue": "input_nan_values",
                        "column": col,
                        "nan_count": int(nan_count),
                        "total_points": len(col_series),
                        "nan_percentage": round((nan_count / len(col_series)) * 100, 2),
                    },
                )

            # Check for zero or negative prices
            invalid_prices = (col_series <= 0).sum()
            if invalid_prices > 0:
                logger.warning(
                    f"Invalid price values in {col} for RSI+MFI calculation",
                    extra={
                        "indicator": "rsimfi",
                        "issue": "invalid_prices",
                        "column": col,
                        "invalid_count": int(invalid_prices),
                        "min_price": (
                            float(col_series.min()) if not col_series.empty else 0
                        ),
                    },
                )

        # Check OHLC relationships
        if all(col in df.columns for col in required_columns):
            # Check high >= low
            invalid_hl = (df["high"] < df["low"]).sum()
            if invalid_hl > 0:
                logger.warning(
                    "Invalid high-low relationships in RSI+MFI input data",
                    extra={
                        "indicator": "rsimfi",
                        "issue": "invalid_high_low_relationship",
                        "invalid_count": int(invalid_hl),
                    },
                )

            # Check for doji candles (high == low)
            doji_count = (df["high"] == df["low"]).sum()
            if doji_count > 0:
                logger.info(
                    "Doji candles detected in RSI+MFI input data",
                    extra={
                        "indicator": "rsimfi",
                        "info": "doji_candles_detected",
                        "doji_count": int(doji_count),
                        "doji_percentage": round((doji_count / len(df)) * 100, 2),
                    },
                )

    def _generate_analysis_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate summary of analysis results for logging.

        Args:
            df: DataFrame with calculated analysis

        Returns:
            Dictionary with analysis summary
        """
        summary: dict[str, Any] = {"total_data_points": len(df)}

        # RSI+MFI value statistics
        if "rsimfi" in df.columns:
            rsimfi_series = df["rsimfi"].dropna()
            if not rsimfi_series.empty:
                summary.update(
                    {
                        "rsimfi_valid_count": len(rsimfi_series),
                        "rsimfi_min": round(float(rsimfi_series.min()), 4),
                        "rsimfi_max": round(float(rsimfi_series.max()), 4),
                        "rsimfi_mean": round(float(rsimfi_series.mean()), 4),
                        "rsimfi_std": round(float(rsimfi_series.std()), 4),
                    }
                )

        # Color condition counts
        if "color_signal" in df.columns:
            positive_count = (df["color_signal"] == 1).sum()
            negative_count = (df["color_signal"] == -1).sum()
            neutral_count = (df["color_signal"] == 0).sum()

            summary.update(
                {
                    "positive_periods": int(positive_count),
                    "negative_periods": int(negative_count),
                    "neutral_periods": int(neutral_count),
                }
            )

        # Area condition counts
        if "area_signal" in df.columns:
            bull_area_count = (df["area_signal"] == 1).sum()
            bear_area_count = (df["area_signal"] == -1).sum()
            neutral_area_count = (df["area_signal"] == 0).sum()

            summary.update(
                {
                    "bull_area_periods": int(bull_area_count),
                    "bear_area_periods": int(bear_area_count),
                    "neutral_area_periods": int(neutral_area_count),
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
            "last_data_quality_check": self._last_data_quality_check,
        }

        logger.debug(
            "RSI+MFI performance metrics",
            extra={"indicator": "rsimfi", "metrics": metrics},
        )

        return metrics
