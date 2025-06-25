"""
Functional Indicator Adapter

This module provides adapters that bridge existing indicator calculations
with functional programming types, enabling indicator processing using
functional market data while maintaining compatibility with existing systems.
"""

import logging
from collections.abc import Callable

import pandas as pd

from bot.fp.types.indicators import (
    BollingerBandsResult,
    IndicatorConfig,
    IndicatorResult,
    MACDResult,
    MovingAverageResult,
    ROCResult,
    RSIResult,
    StochasticResult,
    TimeSeries,
    VuManchuResult,
)
from bot.fp.types.market import Candle as FPCandle
from bot.indicators.vumanchu import VuManchuCipher
from bot.trading_types import IndicatorData as CurrentIndicatorData

from .type_converters import (
    convert_fp_candle_list_to_current,
)

logger = logging.getLogger(__name__)


class FunctionalIndicatorProcessor:
    """
    Functional processor for technical indicators.

    This class processes indicators using functional types while maintaining
    compatibility with the existing indicator calculations.
    """

    def __init__(self, config: IndicatorConfig | None = None):
        """
        Initialize the functional indicator processor.

        Args:
            config: Indicator configuration (uses defaults if None)
        """
        self.config = config or IndicatorConfig()

        # Initialize existing indicator calculators
        self.vumanchu = VuManchuCipher()

        # Functional state
        self._recent_results: dict[str, list[IndicatorResult]] = {}

        # Callbacks for indicator updates
        self._indicator_callbacks: list[Callable[[str, IndicatorResult], None]] = []

        logger.info("Initialized FunctionalIndicatorProcessor")

    def process_candles(
        self, candles: list[FPCandle]
    ) -> dict[str, IndicatorResult | None]:
        """
        Process functional candles and calculate all indicators.

        Args:
            candles: List of functional candles

        Returns:
            Dictionary of indicator results
        """
        if not candles:
            return {}

        try:
            # Convert to current format for existing calculations
            current_candles = convert_fp_candle_list_to_current(candles)

            # Create DataFrame for calculations
            df = self._create_dataframe(current_candles)

            if df.empty or len(df) < 50:  # Need sufficient data for indicators
                logger.warning(f"Insufficient data for indicators: {len(df)} candles")
                return {}

            # Calculate all indicators
            results = {}

            # VuManchu Cipher
            vumanchu_result = self._calculate_vumanchu(df)
            if vumanchu_result:
                results["vumanchu"] = vumanchu_result
                self._store_result("vumanchu", vumanchu_result)

            # Moving Averages
            ma_result = self._calculate_moving_average(df)
            if ma_result:
                results["moving_average"] = ma_result
                self._store_result("moving_average", ma_result)

            # RSI
            rsi_result = self._calculate_rsi(df)
            if rsi_result:
                results["rsi"] = rsi_result
                self._store_result("rsi", rsi_result)

            # MACD
            macd_result = self._calculate_macd(df)
            if macd_result:
                results["macd"] = macd_result
                self._store_result("macd", macd_result)

            # Bollinger Bands
            bb_result = self._calculate_bollinger_bands(df)
            if bb_result:
                results["bollinger_bands"] = bb_result
                self._store_result("bollinger_bands", bb_result)

            # Stochastic
            stoch_result = self._calculate_stochastic(df)
            if stoch_result:
                results["stochastic"] = stoch_result
                self._store_result("stochastic", stoch_result)

            # ROC
            roc_result = self._calculate_roc(df)
            if roc_result:
                results["roc"] = roc_result
                self._store_result("roc", roc_result)

            # Notify callbacks
            for indicator_name, result in results.items():
                if result:
                    self._notify_callbacks(indicator_name, result)

            return results

        except Exception as e:
            logger.exception(f"Error processing indicators: {e}")
            return {}

    def _create_dataframe(self, candles) -> pd.DataFrame:
        """Create pandas DataFrame from market data."""
        if not candles:
            return pd.DataFrame()

        data = []
        for candle in candles:
            data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        df = pd.DataFrame(data)
        df = df.set_index("timestamp")
        return df.sort_index()

    def _calculate_vumanchu(self, df: pd.DataFrame) -> VuManchuResult | None:
        """Calculate VuManchu Cipher indicators."""
        try:
            # Use existing VuManchu implementation
            latest_indicators = self.vumanchu.calculate(df)

            if not latest_indicators:
                return None

            # Extract latest values
            wave_a = latest_indicators.get("cipher_a_dot", 0.0)
            wave_b = latest_indicators.get("cipher_b_wave", 0.0)

            return VuManchuResult(
                timestamp=df.index[-1].to_pydatetime(),
                wave_a=float(wave_a) if wave_a is not None else 0.0,
                wave_b=float(wave_b) if wave_b is not None else 0.0,
            )

        except Exception as e:
            logger.exception(f"Error calculating VuManchu: {e}")
            return None

    def _calculate_moving_average(self, df: pd.DataFrame) -> MovingAverageResult | None:
        """Calculate Moving Average."""
        try:
            if len(df) < self.config.ma_period:
                return None

            if self.config.ma_type == "SMA":
                ma_value = (
                    df["close"].rolling(window=self.config.ma_period).mean().iloc[-1]
                )
            else:  # EMA
                ma_value = df["close"].ewm(span=self.config.ma_period).mean().iloc[-1]

            if pd.isna(ma_value):
                return None

            return MovingAverageResult(
                timestamp=df.index[-1].to_pydatetime(),
                value=float(ma_value),
                period=self.config.ma_period,
            )

        except Exception as e:
            logger.exception(f"Error calculating Moving Average: {e}")
            return None

    def _calculate_rsi(self, df: pd.DataFrame) -> RSIResult | None:
        """Calculate RSI."""
        try:
            if len(df) < self.config.rsi_period + 1:
                return None

            delta = df["close"].diff()
            gain = (
                (delta.where(delta > 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            rsi_value = rsi.iloc[-1]
            if pd.isna(rsi_value):
                return None

            return RSIResult(
                timestamp=df.index[-1].to_pydatetime(),
                value=float(rsi_value),
                overbought=self.config.rsi_overbought,
                oversold=self.config.rsi_oversold,
            )

        except Exception as e:
            logger.exception(f"Error calculating RSI: {e}")
            return None

    def _calculate_macd(self, df: pd.DataFrame) -> MACDResult | None:
        """Calculate MACD."""
        try:
            min_periods = max(self.config.macd_slow, self.config.macd_signal) + 1
            if len(df) < min_periods:
                return None

            ema_fast = df["close"].ewm(span=self.config.macd_fast).mean()
            ema_slow = df["close"].ewm(span=self.config.macd_slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.config.macd_signal).mean()
            histogram = macd_line - signal_line

            macd_value = macd_line.iloc[-1]
            signal_value = signal_line.iloc[-1]
            hist_value = histogram.iloc[-1]

            if pd.isna(macd_value) or pd.isna(signal_value) or pd.isna(hist_value):
                return None

            return MACDResult(
                timestamp=df.index[-1].to_pydatetime(),
                macd=float(macd_value),
                signal=float(signal_value),
                histogram=float(hist_value),
            )

        except Exception as e:
            logger.exception(f"Error calculating MACD: {e}")
            return None

    def _calculate_bollinger_bands(
        self, df: pd.DataFrame
    ) -> BollingerBandsResult | None:
        """Calculate Bollinger Bands."""
        try:
            if len(df) < self.config.bb_period:
                return None

            rolling_mean = df["close"].rolling(window=self.config.bb_period).mean()
            rolling_std = df["close"].rolling(window=self.config.bb_period).std()

            upper_band = rolling_mean + (rolling_std * self.config.bb_std_dev)
            lower_band = rolling_mean - (rolling_std * self.config.bb_std_dev)

            upper_value = upper_band.iloc[-1]
            middle_value = rolling_mean.iloc[-1]
            lower_value = lower_band.iloc[-1]

            if pd.isna(upper_value) or pd.isna(middle_value) or pd.isna(lower_value):
                return None

            return BollingerBandsResult(
                timestamp=df.index[-1].to_pydatetime(),
                upper=float(upper_value),
                middle=float(middle_value),
                lower=float(lower_value),
            )

        except Exception as e:
            logger.exception(f"Error calculating Bollinger Bands: {e}")
            return None

    def _calculate_stochastic(self, df: pd.DataFrame) -> StochasticResult | None:
        """Calculate Stochastic Oscillator."""
        try:
            if len(df) < self.config.stoch_k_period:
                return None

            low_min = df["low"].rolling(window=self.config.stoch_k_period).min()
            high_max = df["high"].rolling(window=self.config.stoch_k_period).max()

            k_percent = 100 * ((df["close"] - low_min) / (high_max - low_min))

            # Smooth %K
            if self.config.stoch_smooth_k > 1:
                k_percent = k_percent.rolling(window=self.config.stoch_smooth_k).mean()

            # Calculate %D
            d_percent = k_percent.rolling(window=self.config.stoch_d_period).mean()

            k_value = k_percent.iloc[-1]
            d_value = d_percent.iloc[-1]

            if pd.isna(k_value) or pd.isna(d_value):
                return None

            return StochasticResult(
                timestamp=df.index[-1].to_pydatetime(),
                k_percent=float(k_value),
                d_percent=float(d_value),
                overbought=self.config.stoch_overbought,
                oversold=self.config.stoch_oversold,
            )

        except Exception as e:
            logger.exception(f"Error calculating Stochastic: {e}")
            return None

    def _calculate_roc(self, df: pd.DataFrame) -> ROCResult | None:
        """Calculate Rate of Change."""
        try:
            if len(df) < self.config.roc_period + 1:
                return None

            roc = (
                (df["close"] - df["close"].shift(self.config.roc_period))
                / df["close"].shift(self.config.roc_period)
            ) * 100

            roc_value = roc.iloc[-1]
            if pd.isna(roc_value):
                return None

            return ROCResult(
                timestamp=df.index[-1].to_pydatetime(),
                value=float(roc_value),
                period=self.config.roc_period,
            )

        except Exception as e:
            logger.exception(f"Error calculating ROC: {e}")
            return None

    def _store_result(self, indicator_name: str, result: IndicatorResult) -> None:
        """Store indicator result in history."""
        if indicator_name not in self._recent_results:
            self._recent_results[indicator_name] = []

        self._recent_results[indicator_name].append(result)

        # Keep only recent results (last 1000)
        if len(self._recent_results[indicator_name]) > 1000:
            self._recent_results[indicator_name] = self._recent_results[indicator_name][
                -1000:
            ]

    def _notify_callbacks(self, indicator_name: str, result: IndicatorResult) -> None:
        """Notify all indicator callbacks."""
        for callback in self._indicator_callbacks:
            try:
                callback(indicator_name, result)
            except Exception as e:
                logger.exception(f"Error in indicator callback: {e}")

    # Callback management

    def add_indicator_callback(
        self, callback: Callable[[str, IndicatorResult], None]
    ) -> None:
        """Add a callback for indicator updates."""
        self._indicator_callbacks.append(callback)

    def remove_indicator_callback(
        self, callback: Callable[[str, IndicatorResult], None]
    ) -> None:
        """Remove an indicator callback."""
        if callback in self._indicator_callbacks:
            self._indicator_callbacks.remove(callback)

    # State access

    def get_indicator_history(
        self, indicator_name: str, limit: int | None = None
    ) -> TimeSeries[IndicatorResult]:
        """Get indicator history as time series."""
        results = self._recent_results.get(indicator_name, [])

        if limit:
            results = results[-limit:]

        return TimeSeries(
            data=results,
            symbol="",  # Not symbol-specific
            interval=f"{indicator_name}_results",
        )

    def get_latest_result(self, indicator_name: str) -> IndicatorResult | None:
        """Get the latest result for an indicator."""
        results = self._recent_results.get(indicator_name, [])
        return results[-1] if results else None

    def get_all_latest_results(self) -> dict[str, IndicatorResult | None]:
        """Get latest results for all indicators."""
        return {
            name: (results[-1] if results else None)
            for name, results in self._recent_results.items()
        }

    def update_config(self, new_config: IndicatorConfig) -> None:
        """Update indicator configuration."""
        self.config = new_config
        logger.info("Updated indicator configuration")


def convert_to_current_indicators(
    fp_results: dict[str, IndicatorResult | None],
) -> CurrentIndicatorData:
    """
    Convert functional indicator results to current IndicatorData format.

    Args:
        fp_results: Dictionary of functional indicator results

    Returns:
        Current IndicatorData object
    """
    from datetime import UTC, datetime

    # Extract individual indicator values
    vumanchu = fp_results.get("vumanchu")
    ma = fp_results.get("moving_average")
    rsi = fp_results.get("rsi")
    fp_results.get("macd")

    return CurrentIndicatorData(
        timestamp=datetime.now(UTC),
        cipher_a_dot=vumanchu.wave_a if isinstance(vumanchu, VuManchuResult) else None,
        cipher_b_wave=vumanchu.wave_b if isinstance(vumanchu, VuManchuResult) else None,
        rsi=rsi.value if isinstance(rsi, RSIResult) else None,
        ema_fast=(
            ma.value
            if isinstance(ma, MovingAverageResult) and ma.period <= 20
            else None
        ),
        ema_slow=(
            ma.value if isinstance(ma, MovingAverageResult) and ma.period > 20 else None
        ),
        vwap=None,  # Not calculated in this adapter
    )


def create_indicator_processor(
    config: IndicatorConfig | None = None,
) -> FunctionalIndicatorProcessor:
    """
    Create a functional indicator processor.

    Args:
        config: Indicator configuration

    Returns:
        Configured functional indicator processor
    """
    return FunctionalIndicatorProcessor(config)


# Integration helpers


class IndicatorIntegrationAdapter:
    """
    Adapter that integrates functional indicators with existing systems.
    """

    def __init__(self, processor: FunctionalIndicatorProcessor):
        """
        Initialize with a functional indicator processor.

        Args:
            processor: The functional indicator processor
        """
        self.processor = processor
        self._current_indicators: CurrentIndicatorData | None = None

    def process_candles_and_get_current(
        self, candles: list[FPCandle]
    ) -> CurrentIndicatorData:
        """
        Process functional candles and return current format indicators.

        Args:
            candles: List of functional candles

        Returns:
            Current format indicator data
        """
        # Process with functional processor
        fp_results = self.processor.process_candles(candles)

        # Convert to current format
        current_indicators = convert_to_current_indicators(fp_results)

        # Store for reference
        self._current_indicators = current_indicators

        return current_indicators

    def get_latest_current_indicators(self) -> CurrentIndicatorData | None:
        """Get the latest indicators in current format."""
        return self._current_indicators


def create_integrated_indicator_system(
    config: IndicatorConfig | None = None,
) -> tuple[FunctionalIndicatorProcessor, IndicatorIntegrationAdapter]:
    """
    Create an integrated indicator system with both functional and current components.

    Args:
        config: Indicator configuration

    Returns:
        Tuple of (functional_processor, integration_adapter)
    """
    functional_processor = create_indicator_processor(config)
    integration_adapter = IndicatorIntegrationAdapter(functional_processor)

    return functional_processor, integration_adapter
