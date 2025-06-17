"""
Stochastic RSI indicator implementation.

This module provides Python implementation of the Stochastic RSI indicator,
matching the Pine Script formula exactly. Uses vectorized operations for performance.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..utils import ta

logger = logging.getLogger(__name__)


class StochasticRSI:
    """
    Stochastic RSI implementation matching Pine Script formula.

    Pine Script Reference:
    ```pine
    f_stochrsi(_src, _stochlen, _rsilen, _smoothk, _smoothd, _log, _avg) =>
        src = _log ? log(_src) : _src
        rsi = rsi(src, _rsilen)
        kk = sma(stoch(rsi, rsi, rsi, _stochlen), _smoothk)
        d1 = sma(kk, _smoothd)
        avg_1 = avg(kk, d1)
        k = _avg ? avg_1 : kk
        [k, d1]
    ```

    Features:
    - Exact Pine Script formula implementation
    - Support for logarithmic source transformation
    - K and D line averaging option
    - Divergence detection
    - Cross signal detection
    """

    def __init__(
        self,
        stoch_length: int = 14,
        rsi_length: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        use_log: bool = False,
        use_avg: bool = False,
    ):
        """
        Initialize Stochastic RSI parameters.

        Args:
            stoch_length: Stochastic calculation period
            rsi_length: RSI calculation period
            smooth_k: K line smoothing period
            smooth_d: D line smoothing period
            use_log: Apply logarithmic transformation to source
            use_avg: Use average of K and D lines for K output
        """
        self.stoch_length = stoch_length
        self.rsi_length = rsi_length
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        self.use_log = use_log
        self.use_avg = use_avg

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._last_data_quality_check: dict[str, Any] | None = None

    def calculate_stoch_rsi(
        self,
        src: pd.Series,
        stoch_len: int | None = None,
        rsi_len: int | None = None,
        smooth_k: int | None = None,
        smooth_d: int | None = None,
        use_log: bool | None = None,
        use_avg: bool | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI K and D lines.

        Args:
            src: Source price series (typically close prices)
            stoch_len: Override stochastic length
            rsi_len: Override RSI length
            smooth_k: Override K smoothing
            smooth_d: Override D smoothing
            use_log: Override log transformation
            use_avg: Override averaging option

        Returns:
            Tuple of (K line, D line) as pandas Series
        """
        # Use instance parameters if not overridden
        stoch_len = stoch_len or self.stoch_length
        rsi_len = rsi_len or self.rsi_length
        smooth_k = smooth_k or self.smooth_k
        smooth_d = smooth_d or self.smooth_d
        use_log = use_log if use_log is not None else self.use_log
        use_avg = use_avg if use_avg is not None else self.use_avg

        # Check for sufficient data
        min_required = max(stoch_len, rsi_len) + max(smooth_k, smooth_d)
        if len(src) < min_required:
            logger.warning(
                f"Insufficient data for Stochastic RSI: need {min_required}, got {len(src)}"
            )
            return pd.Series(dtype=float, index=src.index), pd.Series(
                dtype=float, index=src.index
            )

        # Step 1: Apply log transformation if requested
        if use_log:
            # Handle negative/zero values by adding small epsilon
            src_transformed = np.log(np.maximum(src, 1e-8))
        else:
            src_transformed = src.copy()

        # Step 2: Calculate RSI
        rsi_values = ta.rsi(src_transformed, length=rsi_len)
        if rsi_values is None:
            logger.warning("RSI calculation failed")
            return pd.Series(dtype=float, index=src.index), pd.Series(
                dtype=float, index=src.index
            )

        # Step 3: Calculate Stochastic of RSI
        # stoch(rsi, rsi, rsi, _stochlen) means using RSI as high, low, and close
        stoch_rsi = self._calculate_stochastic(
            rsi_values, rsi_values, rsi_values, stoch_len
        )

        # Step 4: Smooth K line (kk = sma(stoch(rsi, rsi, rsi, _stochlen), _smoothk))
        kk = ta.sma(stoch_rsi, length=smooth_k)
        if kk is None:
            logger.warning("K line smoothing failed")
            return pd.Series(dtype=float, index=src.index), pd.Series(
                dtype=float, index=src.index
            )

        # Step 5: Calculate D line (d1 = sma(kk, _smoothd))
        d1 = ta.sma(kk, length=smooth_d)
        if d1 is None:
            logger.warning("D line calculation failed")
            return pd.Series(dtype=float, index=src.index), pd.Series(
                dtype=float, index=src.index
            )

        # Step 6: Calculate final K line
        if use_avg:
            # avg_1 = avg(kk, d1) = (kk + d1) / 2
            k = (kk + d1) / 2
        else:
            k = kk

        return k.astype(float), d1.astype(float)

    def _calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, length: int
    ) -> pd.Series:
        """
        Calculate Stochastic oscillator.

        Args:
            high: High values series
            low: Low values series
            close: Close values series
            length: Lookback period

        Returns:
            Stochastic values as pandas Series
        """
        # Calculate rolling highest high and lowest low
        highest_high = high.rolling(window=length, min_periods=1).max()
        lowest_low = low.rolling(window=length, min_periods=1).min()

        # Calculate %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        denominator = highest_high - lowest_low

        # Handle division by zero
        stoch = pd.Series(50.0, index=close.index)  # Default to 50 when range is 0
        valid_mask = denominator != 0
        stoch[valid_mask] = (
            100 * (close[valid_mask] - lowest_low[valid_mask]) / denominator[valid_mask]
        )

        return stoch

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic RSI indicators for a DataFrame.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with added Stochastic RSI columns:
            - stoch_rsi_k: K line values
            - stoch_rsi_d: D line values
            - stoch_rsi_signal: Cross signals (1=bullish, -1=bearish, 0=neutral)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for Stochastic RSI calculation")
            return df.copy()

        result = df.copy()

        # Ensure close column is proper float64 dtype
        if "close" in result.columns:
            result["close"] = pd.to_numeric(result["close"], errors="coerce").astype(
                "float64"
            )
        else:
            logger.error("Close column not found in DataFrame")
            return result

        # Calculate K and D lines
        k_line, d_line = self.calculate_stoch_rsi(result["close"])

        result["stoch_rsi_k"] = k_line
        result["stoch_rsi_d"] = d_line

        # Generate cross signals
        result["stoch_rsi_signal"] = self.get_cross_signals(k_line, d_line)

        return result

    def get_divergence_points(
        self, stoch_k: pd.Series, price_data: pd.Series
    ) -> pd.DataFrame:
        """
        Detect divergence points between Stochastic RSI K line and price.

        Args:
            stoch_k: Stochastic RSI K line values
            price_data: Price series (typically close prices)

        Returns:
            DataFrame with divergence information:
            - bullish_divergence: Boolean series marking bullish divergence points
            - bearish_divergence: Boolean series marking bearish divergence points
            - divergence_strength: Numerical strength of divergence (0-100)
        """
        if len(stoch_k) != len(price_data):
            logger.error("Stochastic K and price data must have same length")
            return pd.DataFrame(index=stoch_k.index)

        result = pd.DataFrame(index=stoch_k.index)
        result["bullish_divergence"] = False
        result["bearish_divergence"] = False
        result["divergence_strength"] = 0.0

        # Need minimum lookback for divergence detection
        lookback = 10
        if len(stoch_k) < lookback * 2:
            return result

        # Find local peaks and troughs in both series
        self._find_local_extrema(price_data, lookback, find_peaks=True)
        self._find_local_extrema(price_data, lookback, find_peaks=False)
        stoch_peaks = self._find_local_extrema(stoch_k, lookback, find_peaks=True)
        stoch_troughs = self._find_local_extrema(stoch_k, lookback, find_peaks=False)

        # Detect bullish divergence: price makes lower lows, stoch makes higher lows
        for i in range(1, len(stoch_troughs)):
            if stoch_troughs.iloc[i] and i > 0:
                # Find previous trough
                prev_trough_idx = None
                for j in range(i - 1, -1, -1):
                    if stoch_troughs.iloc[j]:
                        prev_trough_idx = j
                        break

                if prev_trough_idx is not None:
                    # Check for divergence pattern
                    price_current = price_data.iloc[i]
                    price_prev = price_data.iloc[prev_trough_idx]
                    stoch_current = stoch_k.iloc[i]
                    stoch_prev = stoch_k.iloc[prev_trough_idx]

                    if price_current < price_prev and stoch_current > stoch_prev:
                        result.iloc[i, result.columns.get_loc("bullish_divergence")] = (
                            True
                        )
                        # Calculate strength based on difference magnitudes
                        price_diff = abs(
                            (price_current - price_prev) / price_prev * 100
                        )
                        stoch_diff = abs(stoch_current - stoch_prev)
                        result.iloc[
                            i, result.columns.get_loc("divergence_strength")
                        ] = min(price_diff + stoch_diff, 100)

        # Detect bearish divergence: price makes higher highs, stoch makes lower highs
        for i in range(1, len(stoch_peaks)):
            if stoch_peaks.iloc[i] and i > 0:
                # Find previous peak
                prev_peak_idx = None
                for j in range(i - 1, -1, -1):
                    if stoch_peaks.iloc[j]:
                        prev_peak_idx = j
                        break

                if prev_peak_idx is not None:
                    # Check for divergence pattern
                    price_current = price_data.iloc[i]
                    price_prev = price_data.iloc[prev_peak_idx]
                    stoch_current = stoch_k.iloc[i]
                    stoch_prev = stoch_k.iloc[prev_peak_idx]

                    if price_current > price_prev and stoch_current < stoch_prev:
                        result.iloc[i, result.columns.get_loc("bearish_divergence")] = (
                            True
                        )
                        # Calculate strength based on difference magnitudes
                        price_diff = abs(
                            (price_current - price_prev) / price_prev * 100
                        )
                        stoch_diff = abs(stoch_current - stoch_prev)
                        result.iloc[
                            i, result.columns.get_loc("divergence_strength")
                        ] = min(price_diff + stoch_diff, 100)

        return result

    def _find_local_extrema(
        self, series: pd.Series, lookback: int, find_peaks: bool = True
    ) -> pd.Series:
        """
        Find local peaks or troughs in a series.

        Args:
            series: Input series
            lookback: Lookback period for comparison
            find_peaks: True for peaks, False for troughs

        Returns:
            Boolean series marking extrema points
        """
        extrema = pd.Series(False, index=series.index)

        for i in range(lookback, len(series) - lookback):
            window = series.iloc[i - lookback : i + lookback + 1]
            current_value = series.iloc[i]

            if find_peaks:
                if (
                    current_value == window.max()
                    and current_value > series.iloc[i - 1]
                    and current_value > series.iloc[i + 1]
                ):
                    extrema.iloc[i] = True
            else:
                if (
                    current_value == window.min()
                    and current_value < series.iloc[i - 1]
                    and current_value < series.iloc[i + 1]
                ):
                    extrema.iloc[i] = True

        return extrema

    def get_cross_signals(self, k: pd.Series, d: pd.Series) -> pd.Series:
        """
        Generate crossover signals between K and D lines.

        Args:
            k: K line values
            d: D line values

        Returns:
            Series with signals: 1 = bullish cross (K crosses above D),
                               -1 = bearish cross (K crosses below D),
                                0 = no cross
        """
        logger.debug(
            "Calculating Stochastic RSI cross signals",
            extra={
                "indicator": "stochastic_rsi",
                "step": "cross_signals",
                "k_data_points": len(k),
                "d_data_points": len(d),
            },
        )

        if len(k) != len(d):
            logger.error(
                "K and D series length mismatch for cross signals",
                extra={
                    "indicator": "stochastic_rsi",
                    "error_type": "length_mismatch",
                    "k_length": len(k),
                    "d_length": len(d),
                },
            )
            return pd.Series(0, index=k.index, dtype=int)

        signals = pd.Series(0, index=k.index, dtype=int)

        if len(k) < 2:
            logger.warning(
                "Insufficient data for cross signal calculation",
                extra={
                    "indicator": "stochastic_rsi",
                    "issue": "insufficient_data_for_crosses",
                    "data_points": len(k),
                },
            )
            return signals

        try:
            # Bullish cross: K crosses above D
            bullish_cross = (k.shift(1) <= d.shift(1)) & (k > d)

            # Bearish cross: K crosses below D
            bearish_cross = (k.shift(1) >= d.shift(1)) & (k < d)

            signals[bullish_cross] = 1
            signals[bearish_cross] = -1

            # Count signals for logging
            bullish_count = bullish_cross.sum()
            bearish_count = bearish_cross.sum()

            logger.debug(
                "Stochastic RSI cross signals calculated",
                extra={
                    "indicator": "stochastic_rsi",
                    "bullish_crosses": int(bullish_count),
                    "bearish_crosses": int(bearish_count),
                    "total_crosses": int(bullish_count + bearish_count),
                    "cross_frequency": (
                        round(((bullish_count + bearish_count) / len(k)) * 100, 2)
                        if len(k) > 0
                        else 0
                    ),
                },
            )

            return signals

        except Exception as e:
            logger.error(
                "Failed to calculate Stochastic RSI cross signals",
                extra={
                    "indicator": "stochastic_rsi",
                    "error_type": "cross_signal_calculation_failed",
                    "error_message": str(e),
                },
            )
            return signals

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get the latest Stochastic RSI values.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with latest indicator values
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        return {
            "stoch_rsi_k": latest.get("stoch_rsi_k"),
            "stoch_rsi_d": latest.get("stoch_rsi_d"),
            "stoch_rsi_signal": latest.get("stoch_rsi_signal"),
            "stoch_rsi_overbought": latest.get("stoch_rsi_k", 0) > 80,
            "stoch_rsi_oversold": latest.get("stoch_rsi_k", 0) < 20,
        }

    def get_overbought_oversold_levels(
        self, k: pd.Series, d: pd.Series, overbought: float = 80, oversold: float = 20
    ) -> pd.DataFrame:
        """
        Get overbought/oversold signals based on K and D line levels.

        Args:
            k: K line values
            d: D line values
            overbought: Overbought threshold (default 80)
            oversold: Oversold threshold (default 20)

        Returns:
            DataFrame with level-based signals
        """
        result = pd.DataFrame(index=k.index)

        result["overbought"] = (k > overbought) & (d > overbought)
        result["oversold"] = (k < oversold) & (d < oversold)
        result["overbought_exit"] = (k.shift(1) > overbought) & (k <= overbought)
        result["oversold_exit"] = (k.shift(1) < oversold) & (k >= oversold)

        return result

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
                "NaN values in Stochastic RSI input data",
                extra={
                    "indicator": "stochastic_rsi",
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
                "Invalid price values in Stochastic RSI input data",
                extra={
                    "indicator": "stochastic_rsi",
                    "issue": "invalid_prices",
                    "invalid_count": int(invalid_prices),
                    "min_price": float(src.min()) if not src.empty else 0,
                },
            )

        # Check for extreme price changes
        if len(src) > 1:
            price_changes = src.pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
            if extreme_changes > 0:
                logger.warning(
                    "Extreme price changes in Stochastic RSI input data",
                    extra={
                        "indicator": "stochastic_rsi",
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
                    "Duplicate timestamps in Stochastic RSI input data",
                    extra={
                        "indicator": "stochastic_rsi",
                        "issue": "duplicate_timestamps",
                        "duplicate_count": int(duplicate_count),
                    },
                )

        # Check data continuity
        if len(close_series) > 1:
            # Check for gaps in data (this is a simple heuristic)
            time_diffs = pd.Series(df.index).diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > median_diff * 3).sum()
                if large_gaps > 0:
                    logger.warning(
                        "Large time gaps detected in Stochastic RSI input data",
                        extra={
                            "indicator": "stochastic_rsi",
                            "issue": "large_time_gaps",
                            "gap_count": int(large_gaps),
                            "median_diff": str(median_diff),
                        },
                    )

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
                "stoch_length": self.stoch_length,
                "rsi_length": self.rsi_length,
                "smooth_k": self.smooth_k,
                "smooth_d": self.smooth_d,
                "use_log": self.use_log,
                "use_avg": self.use_avg,
            },
            "last_data_quality_check": self._last_data_quality_check,
        }

        logger.debug(
            "Stochastic RSI performance metrics",
            extra={"indicator": "stochastic_rsi", "metrics": metrics},
        )

        return metrics
