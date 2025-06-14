"""
Cipher B Advanced Signal Patterns Implementation.

This module implements the sophisticated signal patterns from VuManChu Cipher B
including Buy/Sell circles, Gold signals, and divergence-based patterns.
Implements exact Pine Script signal logic and timing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .divergence_detector import DivergenceDetector, DivergenceSignal, DivergenceType
from .stochastic_rsi import StochasticRSI
from .wavetrend import WaveTrend

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of Cipher B signals."""

    BUY_CIRCLE = "buy_circle"
    SELL_CIRCLE = "sell_circle"
    GOLD_BUY = "gold_buy"
    DIVERGENCE_BUY = "divergence_buy"
    DIVERGENCE_SELL = "divergence_sell"
    SMALL_CIRCLE_UP = "small_circle_up"
    SMALL_CIRCLE_DOWN = "small_circle_down"


class SignalStrength(Enum):
    """Signal strength levels."""

    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class CipherBSignal:
    """Represents a Cipher B signal."""

    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    value: float
    index: int
    timestamp: pd.Timestamp | None = None
    conditions: dict[str, Any] = None
    divergence_info: DivergenceSignal | None = None


class CipherBSignals:
    """
    Cipher B Advanced Signal Patterns Implementation.

    Implements sophisticated signal patterns from Cipher B including:
    - Buy/Sell circles with exact Pine Script timing
    - Gold buy signals with RSI and fractal conditions
    - Divergence-based signals integration
    - Small circle signals for wave trend crosses
    - Quality-based signal filtering and ranking
    """

    def __init__(
        self,
        wt_channel_length: int = 9,
        wt_average_length: int = 12,
        wt_ma_length: int = 3,
        ob_level: float = 53.0,
        ob_level2: float = 60.0,
        ob_level3: float = 100.0,
        os_level: float = -53.0,
        os_level2: float = -60.0,
        os_level3: float = -75.0,
        wt_div_ob_level: float = 45.0,
        wt_div_os_level: float = -65.0,
        wt_div_ob_level_add: float = 15.0,
        wt_div_os_level_add: float = -40.0,
        rsi_length: int = 14,
        rsi_div_ob_level: float = 60.0,
        rsi_div_os_level: float = 30.0,
        stoch_length: int = 14,
        stoch_rsi_length: int = 14,
        stoch_k_smooth: int = 3,
        stoch_d_smooth: int = 3,
        stoch_use_log: bool = True,
        stoch_use_avg: bool = False,
    ):
        """
        Initialize Cipher B Signals calculator.

        Args:
            wt_channel_length: WaveTrend channel length
            wt_average_length: WaveTrend average length
            wt_ma_length: WaveTrend MA length
            ob_level: Overbought level 1
            ob_level2: Overbought level 2
            ob_level3: Overbought level 3
            os_level: Oversold level 1
            os_level2: Oversold level 2
            os_level3: Oversold level 3
            wt_div_ob_level: WT divergence overbought minimum
            wt_div_os_level: WT divergence oversold minimum
            wt_div_ob_level_add: WT 2nd divergence overbought
            wt_div_os_level_add: WT 2nd divergence oversold
            rsi_length: RSI calculation length
            rsi_div_ob_level: RSI divergence overbought minimum
            rsi_div_os_level: RSI divergence oversold minimum
            stoch_length: Stochastic length
            stoch_rsi_length: Stochastic RSI length
            stoch_k_smooth: Stochastic K smoothing
            stoch_d_smooth: Stochastic D smoothing
            stoch_use_log: Use logarithmic transformation for stoch
            stoch_use_avg: Use average of K and D for stoch
        """
        # WaveTrend parameters
        self.wt_channel_length = wt_channel_length
        self.wt_average_length = wt_average_length
        self.wt_ma_length = wt_ma_length

        # Overbought/Oversold levels
        self.ob_level = ob_level
        self.ob_level2 = ob_level2
        self.ob_level3 = ob_level3
        self.os_level = os_level
        self.os_level2 = os_level2
        self.os_level3 = os_level3

        # Divergence levels
        self.wt_div_ob_level = wt_div_ob_level
        self.wt_div_os_level = wt_div_os_level
        self.wt_div_ob_level_add = wt_div_ob_level_add
        self.wt_div_os_level_add = wt_div_os_level_add

        # RSI parameters
        self.rsi_length = rsi_length
        self.rsi_div_ob_level = rsi_div_ob_level
        self.rsi_div_os_level = rsi_div_os_level

        # Stochastic RSI parameters
        self.stoch_length = stoch_length
        self.stoch_rsi_length = stoch_rsi_length
        self.stoch_k_smooth = stoch_k_smooth
        self.stoch_d_smooth = stoch_d_smooth
        self.stoch_use_log = stoch_use_log
        self.stoch_use_avg = stoch_use_avg

        # Initialize component indicators
        self.wavetrend = WaveTrend(
            channel_length=wt_channel_length,
            average_length=wt_average_length,
            ma_length=wt_ma_length,
            overbought_level=ob_level,
            oversold_level=os_level,
        )

        self.stoch_rsi = StochasticRSI(
            stoch_length=stoch_length,
            rsi_length=stoch_rsi_length,
            smooth_k=stoch_k_smooth,
            smooth_d=stoch_d_smooth,
            use_log=stoch_use_log,
            use_avg=stoch_use_avg,
        )

        self.divergence_detector = DivergenceDetector(
            lookback_period=5,
            min_fractal_distance=5,
            max_lookback_bars=60,
            use_limits=True,
        )

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._signal_generation_count = 0
        self._last_data_quality_check = None

        logger.info(
            "Cipher B Signals calculator initialized",
            extra={
                "indicator": "cipher_b_signals",
                "parameters": {
                    "wt_lengths": [wt_channel_length, wt_average_length, wt_ma_length],
                    "ob_levels": [ob_level, ob_level2, ob_level3],
                    "os_levels": [os_level, os_level2, os_level3],
                    "wt_div_levels": [wt_div_ob_level, wt_div_os_level],
                    "rsi_length": rsi_length,
                    "stoch_parameters": {
                        "stoch_length": stoch_length,
                        "stoch_rsi_length": stoch_rsi_length,
                        "smooth_k": stoch_k_smooth,
                        "smooth_d": stoch_d_smooth,
                        "use_log": stoch_use_log,
                        "use_avg": stoch_use_avg,
                    },
                },
            },
        )

    def calculate_basic_signals(
        self,
        wt1: pd.Series,
        wt2: pd.Series,
        wt_conditions: dict[str, pd.Series],
        ob_level: float | None = None,
        os_level: float | None = None,
    ) -> dict[str, pd.Series]:
        """
        Calculate basic buy/sell circle signals.

        Pine Script reference:
        buySignal = wtCross and wtCrossUp and wtOversold
        sellSignal = wtCross and wtCrossDown and wtOverbought

        Args:
            wt1: WaveTrend 1 series
            wt2: WaveTrend 2 series
            wt_conditions: Dictionary with cross and overbought/oversold conditions
            ob_level: Override overbought level
            os_level: Override oversold level

        Returns:
            Dictionary with basic signal conditions
        """
        ob_threshold = ob_level if ob_level is not None else self.ob_level
        os_threshold = os_level if os_level is not None else self.os_level

        # Extract cross conditions
        wt_cross = wt_conditions.get(
            "wt_cross_up", pd.Series(False, index=wt1.index)
        ) | wt_conditions.get("wt_cross_down", pd.Series(False, index=wt1.index))
        wt_cross_up = wt_conditions.get(
            "wt_cross_up", pd.Series(False, index=wt1.index)
        )
        wt_cross_down = wt_conditions.get(
            "wt_cross_down", pd.Series(False, index=wt1.index)
        )

        # Calculate overbought/oversold conditions
        wt_oversold = wt2 <= os_threshold
        wt_overbought = wt2 >= ob_threshold

        # Buy signal: wtCross and wtCrossUp and wtOversold
        buy_signal = wt_cross & wt_cross_up & wt_oversold

        # Sell signal: wtCross and wtCrossDown and wtOverbought
        sell_signal = wt_cross & wt_cross_down & wt_overbought

        return {
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "wt_oversold": wt_oversold,
            "wt_overbought": wt_overbought,
            "wt_cross": wt_cross,
            "wt_cross_up": wt_cross_up,
            "wt_cross_down": wt_cross_down,
        }

    def calculate_divergence_signals(
        self,
        wt_divs: dict[str, list[DivergenceSignal]],
        stoch_divs: dict[str, list[DivergenceSignal]],
        rsi_divs: dict[str, list[DivergenceSignal]],
    ) -> dict[str, pd.Series]:
        """
        Calculate divergence-based signals.

        Pine Script reference:
        buySignalDiv = (wtShowDiv and wtBullDiv) or (wtShowDiv and wtBullDiv_add) or
                       (stochShowDiv and stochBullDiv) or (rsiShowDiv and rsiBullDiv)
        sellSignalDiv = (wtShowDiv and wtBearDiv) or (wtShowDiv and wtBearDiv_add) or
                        (stochShowDiv and stochBearDiv) or (rsiShowDiv and rsiBearDiv)

        Args:
            wt_divs: WaveTrend divergences
            stoch_divs: Stochastic RSI divergences
            rsi_divs: RSI divergences

        Returns:
            Dictionary with divergence signal conditions
        """
        # Get all divergences and determine the required index
        all_divs = []
        all_divs.extend(wt_divs.get("regular_bullish", []))
        all_divs.extend(wt_divs.get("regular_bearish", []))
        all_divs.extend(wt_divs.get("hidden_bullish", []))
        all_divs.extend(wt_divs.get("hidden_bearish", []))
        all_divs.extend(stoch_divs.get("regular_bullish", []))
        all_divs.extend(stoch_divs.get("regular_bearish", []))
        all_divs.extend(stoch_divs.get("hidden_bullish", []))
        all_divs.extend(stoch_divs.get("hidden_bearish", []))
        all_divs.extend(rsi_divs.get("regular_bullish", []))
        all_divs.extend(rsi_divs.get("regular_bearish", []))
        all_divs.extend(rsi_divs.get("hidden_bullish", []))
        all_divs.extend(rsi_divs.get("hidden_bearish", []))

        # Determine index length from divergences or use default
        if all_divs:
            max_index = max(div.end_fractal.index for div in all_divs)
            index_length = max_index + 10  # Add buffer
        else:
            index_length = 100  # Default length

        # Initialize result series
        buy_signal_div = pd.Series(False, index=range(index_length))
        sell_signal_div = pd.Series(False, index=range(index_length))

        # Process WT divergences
        wt_bull_divs = wt_divs.get("regular_bullish", []) + wt_divs.get(
            "hidden_bullish", []
        )
        wt_bear_divs = wt_divs.get("regular_bearish", []) + wt_divs.get(
            "hidden_bearish", []
        )

        for div in wt_bull_divs:
            if div.end_fractal.index < len(buy_signal_div):
                buy_signal_div.iloc[div.end_fractal.index] = True

        for div in wt_bear_divs:
            if div.end_fractal.index < len(sell_signal_div):
                sell_signal_div.iloc[div.end_fractal.index] = True

        # Process Stochastic RSI divergences
        stoch_bull_divs = stoch_divs.get("regular_bullish", []) + stoch_divs.get(
            "hidden_bullish", []
        )
        stoch_bear_divs = stoch_divs.get("regular_bearish", []) + stoch_divs.get(
            "hidden_bearish", []
        )

        for div in stoch_bull_divs:
            if div.end_fractal.index < len(buy_signal_div):
                buy_signal_div.iloc[div.end_fractal.index] = True

        for div in stoch_bear_divs:
            if div.end_fractal.index < len(sell_signal_div):
                sell_signal_div.iloc[div.end_fractal.index] = True

        # Process RSI divergences
        rsi_bull_divs = rsi_divs.get("regular_bullish", []) + rsi_divs.get(
            "hidden_bullish", []
        )
        rsi_bear_divs = rsi_divs.get("regular_bearish", []) + rsi_divs.get(
            "hidden_bearish", []
        )

        for div in rsi_bull_divs:
            if div.end_fractal.index < len(buy_signal_div):
                buy_signal_div.iloc[div.end_fractal.index] = True

        for div in rsi_bear_divs:
            if div.end_fractal.index < len(sell_signal_div):
                sell_signal_div.iloc[div.end_fractal.index] = True

        return {
            "buy_signal_div": buy_signal_div,
            "sell_signal_div": sell_signal_div,
            "wt_bull_div_count": len(wt_bull_divs),
            "wt_bear_div_count": len(wt_bear_divs),
            "stoch_bull_div_count": len(stoch_bull_divs),
            "stoch_bear_div_count": len(stoch_bear_divs),
            "rsi_bull_div_count": len(rsi_bull_divs),
            "rsi_bear_div_count": len(rsi_bear_divs),
        }

    def calculate_gold_signals(
        self,
        wt_data: dict[str, pd.Series],
        rsi_data: pd.Series,
        fractal_data: dict[str, pd.Series],
        os_level3: float | None = None,
    ) -> pd.Series:
        """
        Calculate gold buy signals.

        Pine Script reference:
        lastRsi = valuewhen(wtFractalBot, rsi[2], 0)[2]
        wtGoldBuy = ((wtShowDiv and wtBullDiv) or (rsiShowDiv and rsiBullDiv)) and
                   wtLow_prev <= osLevel3 and
                   wt2 > osLevel3 and
                   wtLow_prev - wt2 <= -5 and
                   lastRsi < 30

        Args:
            wt_data: WaveTrend data including wt1, wt2, fractal info
            rsi_data: RSI values
            fractal_data: Fractal bottom detection data
            os_level3: Override oversold level 3

        Returns:
            Boolean series for gold buy signals
        """
        os_threshold = os_level3 if os_level3 is not None else self.os_level3

        wt2 = wt_data.get("wt2", pd.Series(dtype=float))
        wt_fractal_bot = fractal_data.get(
            "wt_fractal_bot", pd.Series(False, index=wt2.index)
        )
        wt_low_prev = fractal_data.get(
            "wt_low_prev", pd.Series(dtype=float, index=wt2.index)
        )

        if wt2.empty or rsi_data.empty:
            return pd.Series(
                False, index=wt2.index if not wt2.empty else rsi_data.index
            )

        # Calculate lastRsi: valuewhen(wtFractalBot, rsi[2], 0)[2]
        last_rsi = pd.Series(np.nan, index=wt2.index)

        for i in range(2, len(wt_fractal_bot)):
            if wt_fractal_bot.iloc[i]:
                # Find the RSI value when fractal bottom occurred (offset by 2)
                if i >= 2 and i - 2 < len(rsi_data):
                    rsi_value = rsi_data.iloc[i - 2]
                    # Propagate this value forward until next fractal
                    last_rsi.iloc[i:] = rsi_value

        # Fill forward the last known RSI value
        last_rsi = last_rsi.ffill()

        # Gold buy conditions
        # Note: We need divergence info from calculate_divergence_signals
        # For now, we'll implement the other conditions
        wt_low_prev_condition = wt_low_prev <= os_threshold
        wt2_condition = wt2 > os_threshold
        wt_diff_condition = (wt_low_prev - wt2) <= -5
        rsi_condition = last_rsi < 30

        # Combine conditions (excluding divergence for now)
        gold_buy_base = (
            wt_low_prev_condition & wt2_condition & wt_diff_condition & rsi_condition
        )

        return gold_buy_base.fillna(False)

    def calculate_small_circles(
        self, wt_cross_data: dict[str, pd.Series]
    ) -> dict[str, pd.Series]:
        """
        Calculate small circle signals for wave trend crosses.

        Pine Script reference:
        signalColor = wt2 - wt1 > 0 ? color.red : color.lime
        wtCross and wtCrossUp -> small green circle
        wtCross and wtCrossDown -> small red circle

        Args:
            wt_cross_data: Dictionary with cross conditions and wt1, wt2 values

        Returns:
            Dictionary with small circle signal conditions
        """
        wt1 = wt_cross_data.get("wt1", pd.Series(dtype=float))
        wt2 = wt_cross_data.get("wt2", pd.Series(dtype=float))
        wt_cross = wt_cross_data.get("wt_cross", pd.Series(False, index=wt1.index))
        wt_cross_up = wt_cross_data.get(
            "wt_cross_up", pd.Series(False, index=wt1.index)
        )
        wt_cross_down = wt_cross_data.get(
            "wt_cross_down", pd.Series(False, index=wt1.index)
        )

        # Small circle up: wtCross and wtCrossUp
        small_circle_up = wt_cross & wt_cross_up

        # Small circle down: wtCross and wtCrossDown
        small_circle_down = wt_cross & wt_cross_down

        # Signal color logic: wt2 - wt1 > 0 ? red : green
        signal_bearish = (wt2 - wt1) > 0
        signal_bullish = (wt2 - wt1) <= 0

        return {
            "small_circle_up": small_circle_up,
            "small_circle_down": small_circle_down,
            "signal_bearish": signal_bearish,
            "signal_bullish": signal_bullish,
            "wt_cross_any": wt_cross,
        }

    def get_all_cipher_b_signals(
        self,
        df: pd.DataFrame,
        show_wt_div: bool = True,
        show_rsi_div: bool = False,
        show_stoch_div: bool = False,
        show_wt_div_add: bool = True,
    ) -> dict[str, Any]:
        """
        Comprehensive Cipher B signal analysis.

        Args:
            df: DataFrame with OHLCV data
            show_wt_div: Show WaveTrend divergences
            show_rsi_div: Show RSI divergences
            show_stoch_div: Show Stochastic RSI divergences
            show_wt_div_add: Show additional WaveTrend divergences

        Returns:
            Dictionary with all Cipher B signals and analysis
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for Cipher B signals analysis")
            return {}

        result = {
            "signals": [],
            "signal_counts": {},
            "latest_signals": {},
            "signal_strength_distribution": {},
            "divergence_analysis": {},
            "quality_score": 0.0,
        }

        try:
            # Calculate base indicators
            df_wt = self.wavetrend.calculate(df.copy())
            df_stoch = self.stoch_rsi.calculate(df.copy())

            # Calculate RSI
            from ..utils import ta

            rsi = ta.rsi(df["close"], length=self.rsi_length)

            # Extract WaveTrend data
            wt1 = df_wt["wt1"]
            wt2 = df_wt["wt2"]

            # Get WaveTrend conditions
            wt_conditions = self.wavetrend.get_cross_conditions(wt1, wt2)
            wt_ob_os_conditions = self.wavetrend.get_overbought_oversold_conditions(wt2)
            wt_conditions.update(wt_ob_os_conditions)

            # Calculate basic signals
            basic_signals = self.calculate_basic_signals(wt1, wt2, wt_conditions)

            # Detect divergences
            wt_divs = {}
            rsi_divs = {}
            stoch_divs = {}

            if show_wt_div:
                wt_regular_divs = self.divergence_detector.detect_regular_divergences(
                    wt2, df["close"], self.wt_div_ob_level, self.wt_div_os_level
                )
                wt_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    wt2, df["close"]
                )

                wt_divs = {
                    "regular_bullish": [
                        d
                        for d in wt_regular_divs
                        if d.type == DivergenceType.REGULAR_BULLISH
                    ],
                    "regular_bearish": [
                        d
                        for d in wt_regular_divs
                        if d.type == DivergenceType.REGULAR_BEARISH
                    ],
                    "hidden_bullish": [
                        d
                        for d in wt_hidden_divs
                        if d.type == DivergenceType.HIDDEN_BULLISH
                    ],
                    "hidden_bearish": [
                        d
                        for d in wt_hidden_divs
                        if d.type == DivergenceType.HIDDEN_BEARISH
                    ],
                }

            if show_rsi_div and rsi is not None:
                rsi_regular_divs = self.divergence_detector.detect_regular_divergences(
                    rsi, df["close"], self.rsi_div_ob_level, self.rsi_div_os_level
                )
                rsi_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    rsi, df["close"]
                )

                rsi_divs = {
                    "regular_bullish": [
                        d
                        for d in rsi_regular_divs
                        if d.type == DivergenceType.REGULAR_BULLISH
                    ],
                    "regular_bearish": [
                        d
                        for d in rsi_regular_divs
                        if d.type == DivergenceType.REGULAR_BEARISH
                    ],
                    "hidden_bullish": [
                        d
                        for d in rsi_hidden_divs
                        if d.type == DivergenceType.HIDDEN_BULLISH
                    ],
                    "hidden_bearish": [
                        d
                        for d in rsi_hidden_divs
                        if d.type == DivergenceType.HIDDEN_BEARISH
                    ],
                }

            if show_stoch_div:
                stoch_k = df_stoch["stoch_rsi_k"]
                stoch_regular_divs = (
                    self.divergence_detector.detect_regular_divergences(
                        stoch_k, df["close"], 80, 20  # Standard stoch levels
                    )
                )
                stoch_hidden_divs = self.divergence_detector.detect_hidden_divergences(
                    stoch_k, df["close"]
                )

                stoch_divs = {
                    "regular_bullish": [
                        d
                        for d in stoch_regular_divs
                        if d.type == DivergenceType.REGULAR_BULLISH
                    ],
                    "regular_bearish": [
                        d
                        for d in stoch_regular_divs
                        if d.type == DivergenceType.REGULAR_BEARISH
                    ],
                    "hidden_bullish": [
                        d
                        for d in stoch_hidden_divs
                        if d.type == DivergenceType.HIDDEN_BULLISH
                    ],
                    "hidden_bearish": [
                        d
                        for d in stoch_hidden_divs
                        if d.type == DivergenceType.HIDDEN_BEARISH
                    ],
                }

            # Calculate divergence signals
            divergence_signals = self.calculate_divergence_signals(
                wt_divs, stoch_divs, rsi_divs
            )

            # Calculate fractal data for gold signals
            wt_fractals = self.divergence_detector.find_fractals(wt2)

            # Create fractal data with proper indexing
            fractal_indices = []
            fractal_values = []
            bottom_indices = []
            bottom_values = []

            for f in wt_fractals:
                # Use integer index for array-like operations
                if isinstance(f.index, int) and f.index < len(df):
                    fractal_indices.append(f.index)
                    fractal_values.append(f.fractal_type.value == -1)

                    if f.fractal_type.value == -1:  # Bottom fractal
                        bottom_indices.append(f.index)
                        bottom_values.append(f.value)

            # Create series with integer indices that match DataFrame positions
            fractal_bot_series = pd.Series(False, index=range(len(df)))
            wt_low_prev_series = pd.Series(np.nan, index=range(len(df)))

            # Set fractal bottom positions
            for idx in fractal_indices:
                if idx < len(fractal_bot_series):
                    fractal_bot_series.iloc[idx] = fractal_values[
                        fractal_indices.index(idx)
                    ]

            # Set bottom fractal values
            for i, idx in enumerate(bottom_indices):
                if idx < len(wt_low_prev_series):
                    wt_low_prev_series.iloc[idx] = bottom_values[i]

            # Forward fill the low previous values
            wt_low_prev_series = wt_low_prev_series.ffill()

            fractal_data = {
                "wt_fractal_bot": fractal_bot_series,
                "wt_low_prev": wt_low_prev_series,
            }

            # Calculate gold signals
            wt_data = {"wt1": wt1, "wt2": wt2}
            gold_signals = self.calculate_gold_signals(wt_data, rsi, fractal_data)

            # Calculate small circles
            wt_cross_data = {
                "wt1": wt1,
                "wt2": wt2,
                "wt_cross": wt_conditions.get(
                    "wt1_cross_above_wt2", pd.Series(False, index=wt1.index)
                )
                | wt_conditions.get(
                    "wt1_cross_below_wt2", pd.Series(False, index=wt1.index)
                ),
                "wt_cross_up": wt_conditions.get(
                    "wt1_cross_above_wt2", pd.Series(False, index=wt1.index)
                ),
                "wt_cross_down": wt_conditions.get(
                    "wt1_cross_below_wt2", pd.Series(False, index=wt1.index)
                ),
            }
            small_circle_signals = self.calculate_small_circles(wt_cross_data)

            # Create CipherBSignal objects
            signals = []

            # Process buy signals - ensure all series have same length
            buy_signal_series = basic_signals["buy_signal"]
            div_buy_series = divergence_signals.get(
                "buy_signal_div", pd.Series(False, index=range(len(df)))
            )

            # Align series lengths
            min_length = min(
                len(buy_signal_series), len(div_buy_series), len(gold_signals)
            )

            for i in range(min_length):
                timestamp = df.index[i] if hasattr(df.index, "__getitem__") else None

                # Get signal values for this index
                buy_sig = (
                    buy_signal_series.iloc[i] if i < len(buy_signal_series) else False
                )
                div_sig = div_buy_series.iloc[i] if i < len(div_buy_series) else False
                gold_sig = gold_signals.iloc[i] if i < len(gold_signals) else False

                if buy_sig:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.BUY_CIRCLE,
                            strength=SignalStrength.STRONG,
                            confidence=0.8,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                            conditions={
                                "wt_cross_up": (
                                    wt_conditions.get(
                                        "wt1_cross_above_wt2", pd.Series(False)
                                    ).iloc[i]
                                    if i
                                    < len(
                                        wt_conditions.get(
                                            "wt1_cross_above_wt2", pd.Series(False)
                                        )
                                    )
                                    else False
                                ),
                                "wt_oversold": (
                                    wt_conditions.get(
                                        "oversold", pd.Series(False)
                                    ).iloc[i]
                                    if i
                                    < len(
                                        wt_conditions.get("oversold", pd.Series(False))
                                    )
                                    else False
                                ),
                            },
                        )
                    )

                if div_sig:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.DIVERGENCE_BUY,
                            strength=SignalStrength.VERY_STRONG,
                            confidence=0.9,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                        )
                    )

                if gold_sig:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.GOLD_BUY,
                            strength=SignalStrength.VERY_STRONG,
                            confidence=0.95,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                            conditions={
                                "rsi_below_30": (
                                    rsi.iloc[i] < 30
                                    if i < len(rsi) and not pd.isna(rsi.iloc[i])
                                    else False
                                ),
                                "wt_oversold_extreme": (
                                    wt2.iloc[i] <= self.os_level3
                                    if i < len(wt2)
                                    else False
                                ),
                            },
                        )
                    )

            # Process sell signals
            sell_signal_series = basic_signals["sell_signal"]
            div_sell_series = divergence_signals.get(
                "sell_signal_div", pd.Series(False, index=range(len(df)))
            )

            sell_min_length = min(len(sell_signal_series), len(div_sell_series))

            for i in range(sell_min_length):
                timestamp = df.index[i] if hasattr(df.index, "__getitem__") else None

                # Get signal values for this index
                sell_sig = (
                    sell_signal_series.iloc[i] if i < len(sell_signal_series) else False
                )
                div_sig = div_sell_series.iloc[i] if i < len(div_sell_series) else False

                if sell_sig:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.SELL_CIRCLE,
                            strength=SignalStrength.STRONG,
                            confidence=0.8,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                            conditions={
                                "wt_cross_down": (
                                    wt_conditions.get(
                                        "wt1_cross_below_wt2", pd.Series(False)
                                    ).iloc[i]
                                    if i
                                    < len(
                                        wt_conditions.get(
                                            "wt1_cross_below_wt2", pd.Series(False)
                                        )
                                    )
                                    else False
                                ),
                                "wt_overbought": (
                                    wt_conditions.get(
                                        "overbought", pd.Series(False)
                                    ).iloc[i]
                                    if i
                                    < len(
                                        wt_conditions.get(
                                            "overbought", pd.Series(False)
                                        )
                                    )
                                    else False
                                ),
                            },
                        )
                    )

                if div_sig:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.DIVERGENCE_SELL,
                            strength=SignalStrength.VERY_STRONG,
                            confidence=0.9,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                        )
                    )

            # Process small circle signals
            small_up_series = small_circle_signals["small_circle_up"]
            small_down_series = small_circle_signals["small_circle_down"]

            small_min_length = min(len(small_up_series), len(small_down_series))

            for i in range(small_min_length):
                timestamp = df.index[i] if hasattr(df.index, "__getitem__") else None

                # Get signal values for this index
                small_up = (
                    small_up_series.iloc[i] if i < len(small_up_series) else False
                )
                small_down = (
                    small_down_series.iloc[i] if i < len(small_down_series) else False
                )

                if small_up:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.SMALL_CIRCLE_UP,
                            strength=SignalStrength.WEAK,
                            confidence=0.6,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                        )
                    )

                if small_down:
                    signals.append(
                        CipherBSignal(
                            signal_type=SignalType.SMALL_CIRCLE_DOWN,
                            strength=SignalStrength.WEAK,
                            confidence=0.6,
                            value=wt2.iloc[i] if i < len(wt2) else 0,
                            index=i,
                            timestamp=timestamp,
                        )
                    )

            # Sort signals by index (chronological order)
            signals.sort(key=lambda x: x.index)

            # Calculate signal counts
            signal_counts = {}
            for signal_type in SignalType:
                signal_counts[signal_type.value] = len(
                    [s for s in signals if s.signal_type == signal_type]
                )

            # Get latest signals (last 10 bars)
            latest_signals = {}
            recent_signals = [s for s in signals if s.index >= len(df) - 10]
            for signal in recent_signals:
                signal_type_key = signal.signal_type.value
                if signal_type_key not in latest_signals:
                    latest_signals[signal_type_key] = []
                latest_signals[signal_type_key].append(
                    {
                        "index": signal.index,
                        "value": signal.value,
                        "confidence": signal.confidence,
                        "strength": signal.strength.value,
                        "timestamp": signal.timestamp,
                    }
                )

            # Calculate signal strength distribution
            strength_distribution = {}
            for strength in SignalStrength:
                strength_distribution[strength.value] = len(
                    [s for s in signals if s.strength == strength]
                )

            # Calculate overall quality score
            quality_score = self._calculate_quality_score(signals, divergence_signals)

            # Prepare divergence analysis
            divergence_analysis = {
                "wt_divergences": {
                    "bullish_count": len(wt_divs.get("regular_bullish", []))
                    + len(wt_divs.get("hidden_bullish", [])),
                    "bearish_count": len(wt_divs.get("regular_bearish", []))
                    + len(wt_divs.get("hidden_bearish", [])),
                },
                "rsi_divergences": {
                    "bullish_count": len(rsi_divs.get("regular_bullish", []))
                    + len(rsi_divs.get("hidden_bullish", [])),
                    "bearish_count": len(rsi_divs.get("regular_bearish", []))
                    + len(rsi_divs.get("hidden_bearish", [])),
                },
                "stoch_divergences": {
                    "bullish_count": len(stoch_divs.get("regular_bullish", []))
                    + len(stoch_divs.get("hidden_bullish", [])),
                    "bearish_count": len(stoch_divs.get("regular_bearish", []))
                    + len(stoch_divs.get("hidden_bearish", [])),
                },
            }

            result.update(
                {
                    "signals": signals,
                    "signal_counts": signal_counts,
                    "latest_signals": latest_signals,
                    "signal_strength_distribution": strength_distribution,
                    "divergence_analysis": divergence_analysis,
                    "quality_score": quality_score,
                    "basic_signals": basic_signals,
                    "divergence_signals": divergence_signals,
                    "gold_signals": gold_signals,
                    "small_circle_signals": small_circle_signals,
                    "wt_data": wt_data,
                    "component_data": {
                        "wavetrend": df_wt,
                        "stochastic_rsi": df_stoch,
                        "rsi": rsi,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in Cipher B signals calculation: {e}")

        return result

    def _calculate_quality_score(
        self, signals: list[CipherBSignal], divergence_signals: dict[str, Any]
    ) -> float:
        """
        Calculate overall quality score for signals.

        Args:
            signals: List of CipherBSignal objects
            divergence_signals: Divergence signal data

        Returns:
            Quality score from 0.0 to 1.0
        """
        if not signals:
            return 0.0

        # Base score from signal strengths and confidences
        total_strength = sum(s.strength.value for s in signals)
        max_possible_strength = len(signals) * SignalStrength.VERY_STRONG.value
        strength_score = (
            total_strength / max_possible_strength if max_possible_strength > 0 else 0
        )

        confidence_score = sum(s.confidence for s in signals) / len(signals)

        # Bonus for divergence presence
        divergence_bonus = 0.0
        div_counts = [
            divergence_signals.get("wt_bull_div_count", 0),
            divergence_signals.get("wt_bear_div_count", 0),
            divergence_signals.get("rsi_bull_div_count", 0),
            divergence_signals.get("rsi_bear_div_count", 0),
            divergence_signals.get("stoch_bull_div_count", 0),
            divergence_signals.get("stoch_bear_div_count", 0),
        ]

        if sum(div_counts) > 0:
            divergence_bonus = 0.2

        # Bonus for gold signals
        gold_signals = [s for s in signals if s.signal_type == SignalType.GOLD_BUY]
        gold_bonus = 0.1 if gold_signals else 0.0

        # Combine scores
        quality_score = (
            strength_score * 0.4
            + confidence_score * 0.4
            + divergence_bonus
            + gold_bonus
        )

        return min(1.0, quality_score)

    def get_latest_signal_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get a summary of the latest Cipher B signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with latest signal summary
        """
        signals_data = self.get_all_cipher_b_signals(df)

        if not signals_data or not signals_data.get("signals"):
            return {
                "has_signals": False,
                "latest_signal": None,
                "signal_counts": {},
                "quality_score": 0.0,
                "recommendation": "HOLD",
            }

        signals = signals_data["signals"]
        latest_signal = signals[-1] if signals else None

        # Determine recommendation based on latest strong signals
        recommendation = "HOLD"
        recent_strong_signals = [
            s
            for s in signals[-5:]
            if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
        ]

        if recent_strong_signals:
            last_strong = recent_strong_signals[-1]
            if last_strong.signal_type in [
                SignalType.BUY_CIRCLE,
                SignalType.DIVERGENCE_BUY,
                SignalType.GOLD_BUY,
            ]:
                recommendation = "BUY"
            elif last_strong.signal_type in [
                SignalType.SELL_CIRCLE,
                SignalType.DIVERGENCE_SELL,
            ]:
                recommendation = "SELL"

        return {
            "has_signals": True,
            "latest_signal": (
                {
                    "type": latest_signal.signal_type.value,
                    "strength": latest_signal.strength.value,
                    "confidence": latest_signal.confidence,
                    "value": latest_signal.value,
                    "index": latest_signal.index,
                }
                if latest_signal
                else None
            ),
            "signal_counts": signals_data.get("signal_counts", {}),
            "quality_score": signals_data.get("quality_score", 0.0),
            "recommendation": recommendation,
            "divergence_summary": signals_data.get("divergence_analysis", {}),
            "recent_strong_signals_count": len(recent_strong_signals),
        }
