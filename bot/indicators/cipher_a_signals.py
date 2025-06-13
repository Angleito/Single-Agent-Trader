"""
Cipher A Advanced Signal Patterns Implementation.

This module implements the complex signal patterns from the VuManChu Cipher A indicator,
including Yellow Cross signals, Diamond patterns, EMA-based signals, and candle patterns.
Based on the original Pine Script implementation with exact logic matching.

Key Signals Implemented:
- Red Diamond: wtGreenCross and wtCrossDown
- Green Diamond: wtRedCross and wtCrossUp  
- Yellow Cross Up: redDiamond and wt2 < 45 and wt2 > osLevel3 and rsi < 30 and rsi > 15
- Yellow Cross Down: greenDiamond and wt2 > 55 and wt2 < obLevel3 and rsi < 70 and rsi < 85
- Dump Diamond: redDiamond and redCross
- Moon Diamond: greenDiamond and greenCross
- Bull Candle: open > ema2 and open > ema8 and (close[1] > open[1]) and (close > open) and not redDiamond and not redCross
- Bear Candle: open < ema2 and open < ema8 and (close[1] < open[1]) and (close < open) and not greenDiamond and not redCross
"""

import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from .rsimfi import RSIMFIIndicator
from .wavetrend import WaveTrend

logger = logging.getLogger(__name__)


class CipherASignals:
    """
    Cipher A Advanced Signal Patterns implementation.

    This class implements all the sophisticated signal patterns from VuManChu Cipher A
    including diamond patterns, yellow cross signals, extreme diamonds, and candle patterns.
    Integrates with WaveTrend and RSI+MFI indicators for comprehensive signal analysis.
    """

    def __init__(
        self,
        # WaveTrend parameters
        wt_channel_length: int = 10,
        wt_average_length: int = 21,
        wt_ma_length: int = 4,
        # Overbought/Oversold levels
        overbought_level: float = 60.0,
        overbought_level2: float = 53.0,
        overbought_level3: float = 100.0,
        oversold_level: float = -60.0,
        oversold_level2: float = -53.0,
        oversold_level3: float = -100.0,
        # RSI parameters
        rsi_length: int = 14,
        # RSI+MFI parameters
        rsimfi_period: int = 60,
        rsimfi_multiplier: float = 150.0,
        # EMA parameters for candle patterns
        ema2_length: int = 2,
        ema8_length: int = 8,
        # Yellow cross signal thresholds
        yellow_cross_up_wt2_max: float = 45.0,
        yellow_cross_up_rsi_max: float = 30.0,
        yellow_cross_up_rsi_min: float = 15.0,
        yellow_cross_down_wt2_min: float = 55.0,
        yellow_cross_down_rsi_max: float = 70.0,
        yellow_cross_down_rsi_min: float = 85.0,
    ):
        """
        Initialize Cipher A Signals calculator.

        Args:
            wt_channel_length: WaveTrend channel length
            wt_average_length: WaveTrend average length
            wt_ma_length: WaveTrend MA length
            overbought_level: Primary overbought level
            overbought_level2: Secondary overbought level
            overbought_level3: Extreme overbought level
            oversold_level: Primary oversold level
            oversold_level2: Secondary oversold level
            oversold_level3: Extreme oversold level
            rsi_length: RSI calculation period
            rsimfi_period: RSI+MFI calculation period
            rsimfi_multiplier: RSI+MFI multiplier
            ema2_length: Fast EMA period for candle patterns
            ema8_length: Slow EMA period for candle patterns
            yellow_cross_up_wt2_max: Max WT2 level for yellow cross up
            yellow_cross_up_rsi_max: Max RSI level for yellow cross up
            yellow_cross_up_rsi_min: Min RSI level for yellow cross up
            yellow_cross_down_wt2_min: Min WT2 level for yellow cross down
            yellow_cross_down_rsi_max: Max RSI level for yellow cross down
            yellow_cross_down_rsi_min: Min RSI level for yellow cross down
        """
        # Initialize component indicators
        self.wavetrend = WaveTrend(
            channel_length=wt_channel_length,
            average_length=wt_average_length,
            ma_length=wt_ma_length,
            overbought_level=overbought_level,
            oversold_level=oversold_level,
        )

        self.rsimfi = RSIMFIIndicator()

        # Store parameters
        self.overbought_level = overbought_level
        self.overbought_level2 = overbought_level2
        self.overbought_level3 = overbought_level3
        self.oversold_level = oversold_level
        self.oversold_level2 = oversold_level2
        self.oversold_level3 = oversold_level3

        self.rsi_length = rsi_length
        self.rsimfi_period = rsimfi_period
        self.rsimfi_multiplier = rsimfi_multiplier

        self.ema2_length = ema2_length
        self.ema8_length = ema8_length

        # Yellow cross thresholds
        self.yellow_cross_up_wt2_max = yellow_cross_up_wt2_max
        self.yellow_cross_up_rsi_max = yellow_cross_up_rsi_max
        self.yellow_cross_up_rsi_min = yellow_cross_up_rsi_min
        self.yellow_cross_down_wt2_min = yellow_cross_down_wt2_min
        self.yellow_cross_down_rsi_max = yellow_cross_down_rsi_max
        self.yellow_cross_down_rsi_min = yellow_cross_down_rsi_min

        # Initialize performance tracking
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._signal_generation_count = 0
        self._last_data_quality_check = None

        logger.info(
            "Cipher A Signals calculator initialized",
            extra={
                "indicator": "cipher_a_signals",
                "parameters": {
                    "wt_channel_length": wt_channel_length,
                    "wt_average_length": wt_average_length,
                    "wt_ma_length": wt_ma_length,
                    "overbought_levels": [
                        overbought_level,
                        overbought_level2,
                        overbought_level3,
                    ],
                    "oversold_levels": [
                        oversold_level,
                        oversold_level2,
                        oversold_level3,
                    ],
                    "rsi_length": rsi_length,
                    "rsimfi_period": rsimfi_period,
                    "ema_lengths": [ema2_length, ema8_length],
                    "yellow_cross_thresholds": {
                        "up_wt2_max": yellow_cross_up_wt2_max,
                        "up_rsi_max": yellow_cross_up_rsi_max,
                        "up_rsi_min": yellow_cross_up_rsi_min,
                        "down_wt2_min": yellow_cross_down_wt2_min,
                        "down_rsi_max": yellow_cross_down_rsi_max,
                        "down_rsi_min": yellow_cross_down_rsi_min,
                    },
                },
            },
        )

    def calculate_diamond_patterns(
        self,
        wt1: pd.Series,
        wt2: pd.Series,
        ob_level: float | None = None,
        os_level: float | None = None,
    ) -> dict[str, pd.Series]:
        """
        Calculate diamond patterns based on WaveTrend cross conditions.

        Pine Script logic:
        redDiamond = wtGreenCross and wtCrossDown
        greenDiamond = wtRedCross and wtCrossUp

        Where:
        - wtGreenCross = wt1 crosses above wt2 in oversold area
        - wtRedCross = wt1 crosses below wt2 in overbought area
        - wtCrossDown = wt2 crosses down from overbought
        - wtCrossUp = wt2 crosses up from oversold

        Args:
            wt1: WaveTrend 1 series
            wt2: WaveTrend 2 series
            ob_level: Overbought level (uses instance default if None)
            os_level: Oversold level (uses instance default if None)

        Returns:
            Dictionary with diamond pattern conditions:
            - 'red_diamond': Boolean series for red diamond signals
            - 'green_diamond': Boolean series for green diamond signals
            - 'wt_green_cross': Boolean series for green cross conditions
            - 'wt_red_cross': Boolean series for red cross conditions
            - 'wt_cross_down': Boolean series for cross down conditions
            - 'wt_cross_up': Boolean series for cross up conditions
        """
        start_time = time.perf_counter()
        tracemalloc.start()

        logger.debug(
            "Starting diamond patterns calculation",
            extra={
                "indicator": "cipher_a_signals",
                "step": "diamond_patterns",
                "wt1_data_points": len(wt1) if wt1 is not None else 0,
                "wt2_data_points": len(wt2) if wt2 is not None else 0,
                "ob_level": ob_level,
                "os_level": os_level,
                "calculation_id": self._calculation_count,
            },
        )
        if wt1 is None or wt1.empty or wt2 is None or wt2.empty:
            empty_series = pd.Series(dtype=bool, index=pd.Index([]))
            return {
                "red_diamond": empty_series,
                "green_diamond": empty_series,
                "wt_green_cross": empty_series,
                "wt_red_cross": empty_series,
                "wt_cross_down": empty_series,
                "wt_cross_up": empty_series,
            }

        # Use instance levels if not provided
        ob_threshold = ob_level if ob_level is not None else self.overbought_level
        os_threshold = os_level if os_level is not None else self.oversold_level

        # Get basic cross conditions
        cross_conditions = self.wavetrend.get_cross_conditions(wt1, wt2)
        wt1_cross_above_wt2 = cross_conditions["wt1_cross_above_wt2"]
        wt1_cross_below_wt2 = cross_conditions["wt1_cross_below_wt2"]

        # Get overbought/oversold conditions
        ob_os_conditions = self.wavetrend.get_overbought_oversold_conditions(
            wt2, ob_threshold, os_threshold
        )
        overbought_cross_down = ob_os_conditions["overbought_cross_down"]
        oversold_cross_up = ob_os_conditions["oversold_cross_up"]
        overbought = ob_os_conditions["overbought"]
        oversold = ob_os_conditions["oversold"]

        # Calculate specific cross patterns
        # wtGreenCross: wt1 crosses above wt2 when in oversold territory
        wt_green_cross = wt1_cross_above_wt2 & (wt2 < os_threshold)

        # wtRedCross: wt1 crosses below wt2 when in overbought territory
        wt_red_cross = wt1_cross_below_wt2 & (wt2 > ob_threshold)

        # wtCrossDown: wt2 crosses down from overbought level
        wt_cross_down = overbought_cross_down

        # wtCrossUp: wt2 crosses up from oversold level
        wt_cross_up = oversold_cross_up

        # Calculate diamond patterns
        # redDiamond = wtGreenCross and wtCrossDown
        red_diamond = wt_green_cross & wt_cross_down

        # greenDiamond = wtRedCross and wtCrossUp
        green_diamond = wt_red_cross & wt_cross_up

        # Performance logging
        end_time = time.perf_counter()
        calculation_duration = (end_time - start_time) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self._calculation_count += 1
        self._total_calculation_time += calculation_duration

        # Count signals for analytics
        red_diamond_count = red_diamond.sum() if not red_diamond.empty else 0
        green_diamond_count = green_diamond.sum() if not green_diamond.empty else 0
        wt_green_cross_count = wt_green_cross.sum() if not wt_green_cross.empty else 0
        wt_red_cross_count = wt_red_cross.sum() if not wt_red_cross.empty else 0

        # Calculate pattern frequency
        total_points = len(wt1) if wt1 is not None and not wt1.empty else 0

        logger.info(
            "Diamond patterns calculation completed",
            extra={
                "indicator": "cipher_a_signals",
                "duration_ms": round(calculation_duration, 2),
                "memory_current_mb": round(current / 1024 / 1024, 2),
                "memory_peak_mb": round(peak / 1024 / 1024, 2),
                "data_points": total_points,
                "pattern_signals": {
                    "red_diamond": int(red_diamond_count),
                    "green_diamond": int(green_diamond_count),
                    "wt_green_cross": int(wt_green_cross_count),
                    "wt_red_cross": int(wt_red_cross_count),
                    "total_diamonds": int(red_diamond_count + green_diamond_count),
                },
                "pattern_frequency": {
                    "red_diamond_pct": (
                        round((red_diamond_count / total_points) * 100, 2)
                        if total_points > 0
                        else 0
                    ),
                    "green_diamond_pct": (
                        round((green_diamond_count / total_points) * 100, 2)
                        if total_points > 0
                        else 0
                    ),
                },
                "calculation_count": self._calculation_count,
            },
        )

        return {
            "red_diamond": red_diamond,
            "green_diamond": green_diamond,
            "wt_green_cross": wt_green_cross,
            "wt_red_cross": wt_red_cross,
            "wt_cross_down": wt_cross_down,
            "wt_cross_up": wt_cross_up,
        }

    def calculate_yellow_cross_signals(
        self,
        diamonds: dict[str, pd.Series],
        wt2: pd.Series,
        rsi: pd.Series,
        rsimfi: pd.Series,
        ob_level3: float | None = None,
        os_level3: float | None = None,
    ) -> dict[str, pd.Series]:
        """
        Calculate Yellow Cross signals based on diamond patterns and additional conditions.

        Pine Script logic:
        yellowCrossUp = redDiamond and wt2 < 45 and wt2 > osLevel3 and rsi < 30 and rsi > 15 //and rsiMFI < -5
        yellowCrossDown = greenDiamond and wt2 > 55 and wt2 < obLevel3 and rsi < 70 and rsi < 85 //and rsiMFI > 95

        Args:
            diamonds: Dictionary with diamond pattern series from calculate_diamond_patterns
            wt2: WaveTrend 2 series
            rsi: RSI series
            rsimfi: RSI+MFI series
            ob_level3: Extreme overbought level (uses instance default if None)
            os_level3: Extreme oversold level (uses instance default if None)

        Returns:
            Dictionary with yellow cross signals:
            - 'yellow_cross_up': Boolean series for bullish yellow cross signals
            - 'yellow_cross_down': Boolean series for bearish yellow cross signals
            - 'yellow_cross_up_conditions': Dict with individual condition breakdown
            - 'yellow_cross_down_conditions': Dict with individual condition breakdown
        """
        start_time = time.perf_counter()

        logger.debug(
            "Starting yellow cross signals calculation",
            extra={
                "indicator": "cipher_a_signals",
                "step": "yellow_cross_signals",
                "diamonds_available": list(diamonds.keys()) if diamonds else [],
                "wt2_data_points": len(wt2) if wt2 is not None else 0,
                "rsi_data_points": len(rsi) if rsi is not None else 0,
                "rsimfi_data_points": len(rsimfi) if rsimfi is not None else 0,
                "ob_level3": ob_level3,
                "os_level3": os_level3,
                "yellow_cross_thresholds": {
                    "up_wt2_max": self.yellow_cross_up_wt2_max,
                    "up_rsi_max": self.yellow_cross_up_rsi_max,
                    "down_wt2_min": self.yellow_cross_down_wt2_min,
                    "down_rsi_max": self.yellow_cross_down_rsi_max,
                },
            },
        )
        if not diamonds or wt2 is None or wt2.empty or rsi is None or rsi.empty:
            empty_series = pd.Series(dtype=bool, index=pd.Index([]))
            return {
                "yellow_cross_up": empty_series,
                "yellow_cross_down": empty_series,
                "yellow_cross_up_conditions": {},
                "yellow_cross_down_conditions": {},
            }

        # Use instance levels if not provided
        ob_level3_val = ob_level3 if ob_level3 is not None else self.overbought_level3
        os_level3_val = os_level3 if os_level3 is not None else self.oversold_level3

        red_diamond = diamonds.get(
            "red_diamond", pd.Series(dtype=bool, index=wt2.index)
        )
        green_diamond = diamonds.get(
            "green_diamond", pd.Series(dtype=bool, index=wt2.index)
        )

        # Yellow Cross Up conditions
        wt2_below_max = wt2 < self.yellow_cross_up_wt2_max
        wt2_above_os3 = wt2 > os_level3_val
        rsi_below_max = rsi < self.yellow_cross_up_rsi_max
        rsi_above_min = rsi > self.yellow_cross_up_rsi_min

        # Optional RSI+MFI condition (commented in original Pine Script)
        rsimfi_condition_up = True  # rsimfi < -5 (currently disabled as in Pine Script)
        if rsimfi is not None and not rsimfi.empty:
            rsimfi_condition_up = rsimfi < -5

        yellow_cross_up = (
            red_diamond
            & wt2_below_max
            & wt2_above_os3
            & rsi_below_max
            & rsi_above_min
            # & rsimfi_condition_up  # Uncomment to enable RSI+MFI filter
        )

        # Yellow Cross Down conditions
        wt2_above_min = wt2 > self.yellow_cross_down_wt2_min
        wt2_below_ob3 = wt2 < ob_level3_val
        rsi_below_max_down = rsi < self.yellow_cross_down_rsi_max
        rsi_above_min_down = rsi > self.yellow_cross_down_rsi_min

        # Optional RSI+MFI condition (commented in original Pine Script)
        rsimfi_condition_down = (
            True  # rsimfi > 95 (currently disabled as in Pine Script)
        )
        if rsimfi is not None and not rsimfi.empty:
            rsimfi_condition_down = rsimfi > 95

        yellow_cross_down = (
            green_diamond
            & wt2_above_min
            & wt2_below_ob3
            & rsi_below_max_down
            & rsi_above_min_down
            # & rsimfi_condition_down  # Uncomment to enable RSI+MFI filter
        )

        # Performance logging
        end_time = time.perf_counter()
        calculation_duration = (end_time - start_time) * 1000

        self._signal_generation_count += 1

        # Count signals for analytics
        yellow_up_count = yellow_cross_up.sum() if not yellow_cross_up.empty else 0
        yellow_down_count = (
            yellow_cross_down.sum() if not yellow_cross_down.empty else 0
        )
        red_diamond_count = red_diamond.sum() if not red_diamond.empty else 0
        green_diamond_count = green_diamond.sum() if not green_diamond.empty else 0

        # Calculate condition satisfaction rates
        total_points = len(wt2) if wt2 is not None and not wt2.empty else 0

        # Condition analysis for yellow cross up
        up_condition_stats = {}
        if total_points > 0:
            up_condition_stats = {
                "red_diamond_rate": round((red_diamond_count / total_points) * 100, 2),
                "wt2_below_max_rate": (
                    round((wt2_below_max.sum() / total_points) * 100, 2)
                    if not wt2_below_max.empty
                    else 0
                ),
                "wt2_above_os3_rate": (
                    round((wt2_above_os3.sum() / total_points) * 100, 2)
                    if not wt2_above_os3.empty
                    else 0
                ),
                "rsi_below_max_rate": (
                    round((rsi_below_max.sum() / total_points) * 100, 2)
                    if not rsi_below_max.empty
                    else 0
                ),
                "rsi_above_min_rate": (
                    round((rsi_above_min.sum() / total_points) * 100, 2)
                    if not rsi_above_min.empty
                    else 0
                ),
            }

        # Condition analysis for yellow cross down
        down_condition_stats = {}
        if total_points > 0:
            down_condition_stats = {
                "green_diamond_rate": round(
                    (green_diamond_count / total_points) * 100, 2
                ),
                "wt2_above_min_rate": (
                    round((wt2_above_min.sum() / total_points) * 100, 2)
                    if not wt2_above_min.empty
                    else 0
                ),
                "wt2_below_ob3_rate": (
                    round((wt2_below_ob3.sum() / total_points) * 100, 2)
                    if not wt2_below_ob3.empty
                    else 0
                ),
                "rsi_below_max_down_rate": (
                    round((rsi_below_max_down.sum() / total_points) * 100, 2)
                    if not rsi_below_max_down.empty
                    else 0
                ),
                "rsi_above_min_down_rate": (
                    round((rsi_above_min_down.sum() / total_points) * 100, 2)
                    if not rsi_above_min_down.empty
                    else 0
                ),
            }

        logger.info(
            "Yellow cross signals calculation completed",
            extra={
                "indicator": "cipher_a_signals",
                "duration_ms": round(calculation_duration, 2),
                "data_points": total_points,
                "yellow_cross_signals": {
                    "yellow_cross_up": int(yellow_up_count),
                    "yellow_cross_down": int(yellow_down_count),
                    "total_yellow_crosses": int(yellow_up_count + yellow_down_count),
                },
                "signal_frequency": {
                    "yellow_up_pct": (
                        round((yellow_up_count / total_points) * 100, 2)
                        if total_points > 0
                        else 0
                    ),
                    "yellow_down_pct": (
                        round((yellow_down_count / total_points) * 100, 2)
                        if total_points > 0
                        else 0
                    ),
                },
                "diamond_requirements": {
                    "red_diamond_signals": int(red_diamond_count),
                    "green_diamond_signals": int(green_diamond_count),
                },
                "condition_satisfaction_up": up_condition_stats,
                "condition_satisfaction_down": down_condition_stats,
                "signal_generation_count": self._signal_generation_count,
            },
        )

        # Log any high-confidence signals
        if yellow_up_count > 0:
            logger.info(
                "Yellow Cross Up signals detected",
                extra={
                    "indicator": "cipher_a_signals",
                    "signal_type": "yellow_cross_up",
                    "signal_count": int(yellow_up_count),
                    "confidence_factors": {
                        "has_red_diamond": True,
                        "wt2_in_range": f"<{self.yellow_cross_up_wt2_max} and >{os_level3_val}",
                        "rsi_in_range": f"<{self.yellow_cross_up_rsi_max} and >{self.yellow_cross_up_rsi_min}",
                    },
                },
            )

        if yellow_down_count > 0:
            logger.info(
                "Yellow Cross Down signals detected",
                extra={
                    "indicator": "cipher_a_signals",
                    "signal_type": "yellow_cross_down",
                    "signal_count": int(yellow_down_count),
                    "confidence_factors": {
                        "has_green_diamond": True,
                        "wt2_in_range": f">{self.yellow_cross_down_wt2_min} and <{ob_level3_val}",
                        "rsi_in_range": f"<{self.yellow_cross_down_rsi_max} and >{self.yellow_cross_down_rsi_min}",
                    },
                },
            )

        return {
            "yellow_cross_up": yellow_cross_up,
            "yellow_cross_down": yellow_cross_down,
            "yellow_cross_up_conditions": {
                "red_diamond": red_diamond,
                "wt2_below_max": wt2_below_max,
                "wt2_above_os3": wt2_above_os3,
                "rsi_below_max": rsi_below_max,
                "rsi_above_min": rsi_above_min,
            },
            "yellow_cross_down_conditions": {
                "green_diamond": green_diamond,
                "wt2_above_min": wt2_above_min,
                "wt2_below_ob3": wt2_below_ob3,
                "rsi_below_max_down": rsi_below_max_down,
                "rsi_above_min_down": rsi_above_min_down,
            },
        }

    def calculate_extreme_diamonds(
        self,
        diamonds: dict[str, pd.Series],
        wt1: pd.Series,
        wt2: pd.Series,
        ob_level2: float | None = None,
        os_level2: float | None = None,
    ) -> dict[str, pd.Series]:
        """
        Calculate extreme diamond patterns (Dump and Moon diamonds).

        Pine Script logic:
        dumpDiamond = redDiamond and redCross
        moonDiamond = greenDiamond and greenCross

        Where:
        - redCross = wt1 crosses below wt2 in overbought area (level2)
        - greenCross = wt1 crosses above wt2 in oversold area (level2)

        Args:
            diamonds: Dictionary with diamond pattern series
            wt1: WaveTrend 1 series
            wt2: WaveTrend 2 series
            ob_level2: Secondary overbought level (uses instance default if None)
            os_level2: Secondary oversold level (uses instance default if None)

        Returns:
            Dictionary with extreme diamond patterns:
            - 'dump_diamond': Boolean series for dump diamond signals
            - 'moon_diamond': Boolean series for moon diamond signals
            - 'red_cross': Boolean series for red cross conditions
            - 'green_cross': Boolean series for green cross conditions
        """
        if not diamonds or wt1 is None or wt1.empty or wt2 is None or wt2.empty:
            empty_series = pd.Series(dtype=bool, index=pd.Index([]))
            return {
                "dump_diamond": empty_series,
                "moon_diamond": empty_series,
                "red_cross": empty_series,
                "green_cross": empty_series,
            }

        # Use instance levels if not provided
        ob_level2_val = ob_level2 if ob_level2 is not None else self.overbought_level2
        os_level2_val = os_level2 if os_level2 is not None else self.oversold_level2

        red_diamond = diamonds.get(
            "red_diamond", pd.Series(dtype=bool, index=wt2.index)
        )
        green_diamond = diamonds.get(
            "green_diamond", pd.Series(dtype=bool, index=wt2.index)
        )

        # Get basic cross conditions
        cross_conditions = self.wavetrend.get_cross_conditions(wt1, wt2)
        wt1_cross_above_wt2 = cross_conditions["wt1_cross_above_wt2"]
        wt1_cross_below_wt2 = cross_conditions["wt1_cross_below_wt2"]

        # Calculate EMA cross conditions using secondary levels
        # redCross: bearish cross in overbought territory (secondary level)
        red_cross = wt1_cross_below_wt2 & (wt2 > ob_level2_val)

        # greenCross: bullish cross in oversold territory (secondary level)
        green_cross = wt1_cross_above_wt2 & (wt2 < os_level2_val)

        # Calculate extreme diamonds
        # dumpDiamond = redDiamond and redCross
        dump_diamond = red_diamond & red_cross

        # moonDiamond = greenDiamond and greenCross
        moon_diamond = green_diamond & green_cross

        return {
            "dump_diamond": dump_diamond,
            "moon_diamond": moon_diamond,
            "red_cross": red_cross,
            "green_cross": green_cross,
        }

    def calculate_candle_patterns(
        self,
        ohlc_data: pd.DataFrame,
        ema2: pd.Series | None = None,
        ema8: pd.Series | None = None,
        diamonds: dict[str, pd.Series] | None = None,
        extreme_diamonds: dict[str, pd.Series] | None = None,
    ) -> dict[str, pd.Series]:
        """
        Calculate Bull and Bear candle patterns based on EMA and candle conditions.

        Pine Script logic:
        bullCandle = open > ema2 and open > ema8 and (close[1] > open[1]) and (close > open) and not redDiamond and not redCross
        bearCandle = open < ema2 and open < ema8 and (close[1] < open[1]) and (close < open) and not greenDiamond and not redCross

        Args:
            ohlc_data: DataFrame with OHLC data (open, high, low, close columns)
            ema2: Fast EMA series (calculated if None)
            ema8: Slow EMA series (calculated if None)
            diamonds: Dictionary with diamond patterns (optional for filtering)
            extreme_diamonds: Dictionary with extreme diamond patterns (optional for filtering)

        Returns:
            Dictionary with candle patterns:
            - 'bull_candle': Boolean series for bull candle patterns
            - 'bear_candle': Boolean series for bear candle patterns
            - 'ema2': Fast EMA series used
            - 'ema8': Slow EMA series used
            - 'bull_candle_conditions': Dict with individual condition breakdown
            - 'bear_candle_conditions': Dict with individual condition breakdown
        """
        if ohlc_data is None or ohlc_data.empty:
            empty_series = pd.Series(dtype=bool, index=pd.Index([]))
            return {
                "bull_candle": empty_series,
                "bear_candle": empty_series,
                "ema2": empty_series,
                "ema8": empty_series,
                "bull_candle_conditions": {},
                "bear_candle_conditions": {},
            }

        required_columns = ["open", "high", "low", "close"]
        if not all(col in ohlc_data.columns for col in required_columns):
            logger.error(f"Missing required OHLC columns: {required_columns}")
            empty_series = pd.Series(dtype=bool, index=ohlc_data.index)
            return {
                "bull_candle": empty_series,
                "bear_candle": empty_series,
                "ema2": empty_series,
                "ema8": empty_series,
                "bull_candle_conditions": {},
                "bear_candle_conditions": {},
            }

        # Calculate EMAs if not provided
        if ema2 is None:
            ema2 = ta.ema(ohlc_data["close"], length=self.ema2_length)
        if ema8 is None:
            ema8 = ta.ema(ohlc_data["close"], length=self.ema8_length)

        if ema2 is None or ema8 is None:
            logger.error("Failed to calculate EMAs for candle patterns")
            empty_series = pd.Series(dtype=bool, index=ohlc_data.index)
            return {
                "bull_candle": empty_series,
                "bear_candle": empty_series,
                "ema2": empty_series,
                "ema8": empty_series,
                "bull_candle_conditions": {},
                "bear_candle_conditions": {},
            }

        open_prices = ohlc_data["open"]
        close_prices = ohlc_data["close"]

        # Basic candle conditions
        prev_close_above_open = close_prices.shift(1) > open_prices.shift(
            1
        )  # close[1] > open[1]
        prev_close_below_open = close_prices.shift(1) < open_prices.shift(
            1
        )  # close[1] < open[1]
        current_close_above_open = close_prices > open_prices  # close > open
        current_close_below_open = close_prices < open_prices  # close < open

        # EMA conditions
        open_above_ema2 = open_prices > ema2
        open_above_ema8 = open_prices > ema8
        open_below_ema2 = open_prices < ema2
        open_below_ema8 = open_prices < ema8

        # Diamond filter conditions (if provided)
        red_diamond_filter = pd.Series(False, index=ohlc_data.index)
        green_diamond_filter = pd.Series(False, index=ohlc_data.index)
        red_cross_filter = pd.Series(False, index=ohlc_data.index)

        if diamonds is not None:
            red_diamond_filter = diamonds.get("red_diamond", red_diamond_filter).fillna(
                False
            )
            green_diamond_filter = diamonds.get(
                "green_diamond", green_diamond_filter
            ).fillna(False)

        if extreme_diamonds is not None:
            red_cross_filter = extreme_diamonds.get(
                "red_cross", red_cross_filter
            ).fillna(False)

        # Bull Candle Pattern
        # bullCandle = open > ema2 and open > ema8 and (close[1] > open[1]) and (close > open) and not redDiamond and not redCross
        bull_candle = (
            open_above_ema2
            & open_above_ema8
            & prev_close_above_open
            & current_close_above_open
            & ~red_diamond_filter
            & ~red_cross_filter
        )

        # Bear Candle Pattern
        # bearCandle = open < ema2 and open < ema8 and (close[1] < open[1]) and (close < open) and not greenDiamond and not redCross
        bear_candle = (
            open_below_ema2
            & open_below_ema8
            & prev_close_below_open
            & current_close_below_open
            & ~green_diamond_filter
            & ~red_cross_filter
        )

        return {
            "bull_candle": bull_candle,
            "bear_candle": bear_candle,
            "ema2": ema2,
            "ema8": ema8,
            "bull_candle_conditions": {
                "open_above_ema2": open_above_ema2,
                "open_above_ema8": open_above_ema8,
                "prev_close_above_open": prev_close_above_open,
                "current_close_above_open": current_close_above_open,
                "not_red_diamond": ~red_diamond_filter,
                "not_red_cross": ~red_cross_filter,
            },
            "bear_candle_conditions": {
                "open_below_ema2": open_below_ema2,
                "open_below_ema8": open_below_ema8,
                "prev_close_below_open": prev_close_below_open,
                "current_close_below_open": current_close_below_open,
                "not_green_diamond": ~green_diamond_filter,
                "not_red_cross": ~red_cross_filter,
            },
        }

    def get_all_cipher_a_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Cipher A signal patterns for a DataFrame.

        This is the main method that calculates all signal patterns and combines them
        into a comprehensive analysis with signal strength and confidence scoring.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with all Cipher A signals and analysis:
            - Basic WaveTrend indicators (wt1, wt2, etc.)
            - Diamond patterns (red_diamond, green_diamond, etc.)
            - Yellow cross signals (yellow_cross_up, yellow_cross_down)
            - Extreme diamonds (dump_diamond, moon_diamond)
            - Candle patterns (bull_candle, bear_candle)
            - Signal strength and confidence scores
            - EMAs used for analysis
        """
        start_time = time.perf_counter()
        tracemalloc.start()

        logger.info(
            "Starting complete Cipher A signals calculation",
            extra={
                "indicator": "cipher_a_signals",
                "step": "complete_calculation",
                "input_data_points": len(df),
                "input_columns": list(df.columns) if not df.empty else [],
                "calculation_id": self._calculation_count + 1,
            },
        )
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for Cipher A signals calculation")
            return df.copy() if df is not None else pd.DataFrame()

        result = df.copy()

        # Ensure required columns exist
        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in result.columns]
        if missing_columns:
            logger.error(
                f"Missing required columns for Cipher A signals: {missing_columns}"
            )
            return result

        try:
            # Data quality validation
            self._validate_input_data_quality(result)

            # Step 1: Calculate base WaveTrend indicators
            logger.debug(
                "Calculating WaveTrend indicators",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "wavetrend_calculation",
                },
            )
            result = self.wavetrend.calculate(result)
            wt1, wt2 = result["wt1"], result["wt2"]

            # Step 2: Calculate RSI
            logger.debug(
                "Calculating RSI indicator",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "rsi_calculation",
                    "rsi_length": self.rsi_length,
                },
            )
            rsi = ta.rsi(result["close"], length=self.rsi_length)
            if rsi is not None:
                result["rsi"] = rsi.astype("float64")
                logger.debug(
                    "RSI calculation successful",
                    extra={
                        "indicator": "cipher_a_signals",
                        "rsi_valid_count": (~rsi.isna()).sum(),
                        "rsi_range": (
                            [float(rsi.min()), float(rsi.max())]
                            if not rsi.empty
                            else [0, 0]
                        ),
                    },
                )
            else:
                logger.warning(
                    "RSI calculation failed - using default neutral values",
                    extra={
                        "indicator": "cipher_a_signals",
                        "issue": "rsi_calculation_failed",
                        "fallback_value": 50.0,
                    },
                )
                rsi = pd.Series(50.0, index=result.index)  # Default neutral RSI
                result["rsi"] = rsi

            # Step 3: Calculate RSI+MFI
            logger.debug(
                "Calculating RSI+MFI indicator",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "rsimfi_calculation",
                    "rsimfi_period": self.rsimfi_period,
                    "rsimfi_multiplier": self.rsimfi_multiplier,
                },
            )
            rsimfi = self.rsimfi.calculate_rsimfi(
                result, period=self.rsimfi_period, multiplier=self.rsimfi_multiplier
            )
            result["rsimfi"] = rsimfi.astype("float64")

            # Step 4: Calculate diamond patterns
            logger.debug(
                "Calculating diamond patterns",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "diamond_patterns_calculation",
                },
            )
            diamonds = self.calculate_diamond_patterns(wt1, wt2)
            for key, series in diamonds.items():
                result[key] = series

            # Step 5: Calculate yellow cross signals
            logger.debug(
                "Calculating yellow cross signals",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "yellow_cross_calculation",
                },
            )
            yellow_crosses = self.calculate_yellow_cross_signals(
                diamonds, wt2, rsi, rsimfi
            )
            for key, series in yellow_crosses.items():
                if isinstance(series, pd.Series):
                    result[key] = series

            # Step 6: Calculate extreme diamonds
            logger.debug(
                "Calculating extreme diamonds",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "extreme_diamonds_calculation",
                },
            )
            extreme_diamonds = self.calculate_extreme_diamonds(diamonds, wt1, wt2)
            for key, series in extreme_diamonds.items():
                result[key] = series

            # Step 7: Calculate candle patterns
            logger.debug(
                "Calculating candle patterns",
                extra={
                    "indicator": "cipher_a_signals",
                    "step": "candle_patterns_calculation",
                },
            )
            candle_patterns = self.calculate_candle_patterns(
                result, diamonds=diamonds, extreme_diamonds=extreme_diamonds
            )
            for key, series in candle_patterns.items():
                if isinstance(series, pd.Series):
                    result[key] = series

            # Step 8: Calculate combined signal strength and confidence
            logger.debug(
                "Calculating signal strength and confidence",
                extra={"indicator": "cipher_a_signals", "step": "signal_analysis"},
            )
            signal_analysis = self._calculate_signal_strength_and_confidence(result)
            for key, series in signal_analysis.items():
                result[key] = series

            # Step 9: Add signal summary
            logger.debug(
                "Generating signal summary",
                extra={"indicator": "cipher_a_signals", "step": "signal_summary"},
            )
            result["cipher_a_signal"] = self._generate_signal_summary(result)

            # Performance logging and final validation
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self._calculation_count += 1
            self._total_calculation_time += total_duration
            avg_calculation_time = (
                self._total_calculation_time / self._calculation_count
            )

            # Generate comprehensive summary
            summary = self._generate_complete_signal_summary(result)

            logger.info(
                "Complete Cipher A signals calculation successful",
                extra={
                    "indicator": "cipher_a_signals",
                    "duration_ms": round(total_duration, 2),
                    "memory_current_mb": round(current / 1024 / 1024, 2),
                    "memory_peak_mb": round(peak / 1024 / 1024, 2),
                    "input_data_points": len(df),
                    "output_data_points": len(result),
                    "calculation_count": self._calculation_count,
                    "avg_calculation_time_ms": round(avg_calculation_time, 2),
                    "signal_summary": summary,
                    "calculation_success": True,
                },
            )

        except Exception as e:
            end_time = time.perf_counter()
            error_duration = (end_time - start_time) * 1000

            logger.error(
                "Cipher A signals calculation failed with exception",
                extra={
                    "indicator": "cipher_a_signals",
                    "error_type": "complete_calculation_failed",
                    "error_message": str(e),
                    "error_duration_ms": round(error_duration, 2),
                    "input_data_points": len(df),
                    "calculation_count": self._calculation_count,
                },
            )
            # Return original DataFrame with error indicators
            result["cipher_a_error"] = True
            result["cipher_a_signal"] = 0

        return result

    def _calculate_signal_strength_and_confidence(
        self, df: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """
        Calculate signal strength and confidence scores for all patterns.

        Args:
            df: DataFrame with calculated signals

        Returns:
            Dictionary with strength and confidence series
        """
        # Initialize scores
        bullish_strength = pd.Series(0.0, index=df.index)
        bearish_strength = pd.Series(0.0, index=df.index)
        confidence_score = pd.Series(0.0, index=df.index)

        # Weight different signal types
        weights = {
            "yellow_cross_up": 3.0,
            "yellow_cross_down": 3.0,
            "moon_diamond": 2.5,
            "dump_diamond": 2.5,
            "green_diamond": 2.0,
            "red_diamond": 2.0,
            "bull_candle": 1.5,
            "bear_candle": 1.5,
            "wt_cross_up": 1.0,
            "wt_cross_down": 1.0,
        }

        # Calculate bullish strength
        if "yellow_cross_up" in df.columns:
            bullish_strength += (
                df["yellow_cross_up"].astype(float) * weights["yellow_cross_up"]
            )
        if "moon_diamond" in df.columns:
            bullish_strength += (
                df["moon_diamond"].astype(float) * weights["moon_diamond"]
            )
        if "green_diamond" in df.columns:
            bullish_strength += (
                df["green_diamond"].astype(float) * weights["green_diamond"]
            )
        if "bull_candle" in df.columns:
            bullish_strength += df["bull_candle"].astype(float) * weights["bull_candle"]
        if "wt_cross_up" in df.columns:
            bullish_strength += df["wt_cross_up"].astype(float) * weights["wt_cross_up"]

        # Calculate bearish strength
        if "yellow_cross_down" in df.columns:
            bearish_strength += (
                df["yellow_cross_down"].astype(float) * weights["yellow_cross_down"]
            )
        if "dump_diamond" in df.columns:
            bearish_strength += (
                df["dump_diamond"].astype(float) * weights["dump_diamond"]
            )
        if "red_diamond" in df.columns:
            bearish_strength += df["red_diamond"].astype(float) * weights["red_diamond"]
        if "bear_candle" in df.columns:
            bearish_strength += df["bear_candle"].astype(float) * weights["bear_candle"]
        if "wt_cross_down" in df.columns:
            bearish_strength += (
                df["wt_cross_down"].astype(float) * weights["wt_cross_down"]
            )

        # Calculate confidence based on multiple signal confluence
        signal_count = pd.Series(0, index=df.index)
        for signal_col in [
            "yellow_cross_up",
            "yellow_cross_down",
            "moon_diamond",
            "dump_diamond",
            "green_diamond",
            "red_diamond",
            "bull_candle",
            "bear_candle",
        ]:
            if signal_col in df.columns:
                signal_count += df[signal_col].astype(int)

        # Confidence increases with multiple signals
        confidence_score = np.minimum(signal_count / 3.0 * 100, 100.0)

        # Adjust confidence based on WaveTrend position
        if "wt2" in df.columns:
            wt2 = df["wt2"]
            # Higher confidence when WT2 is in extreme zones
            extreme_zone_boost = np.where(
                (wt2 > self.overbought_level) | (wt2 < self.oversold_level), 20.0, 0.0
            )
            confidence_score = np.minimum(confidence_score + extreme_zone_boost, 100.0)

        return {
            "cipher_a_bullish_strength": bullish_strength,
            "cipher_a_bearish_strength": bearish_strength,
            "cipher_a_confidence": confidence_score,
        }

    def _generate_signal_summary(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate overall signal summary combining all patterns.

        Args:
            df: DataFrame with all calculated signals

        Returns:
            Series with signal summary: 1=bullish, -1=bearish, 0=neutral
        """
        signal_summary = pd.Series(0, index=df.index, dtype=int)

        bullish_strength = df.get(
            "cipher_a_bullish_strength", pd.Series(0.0, index=df.index)
        )
        bearish_strength = df.get(
            "cipher_a_bearish_strength", pd.Series(0.0, index=df.index)
        )
        confidence = df.get("cipher_a_confidence", pd.Series(0.0, index=df.index))

        # Require minimum confidence for signal generation
        min_confidence = 25.0
        sufficient_confidence = confidence >= min_confidence

        # Generate signals based on strength comparison
        net_strength = bullish_strength - bearish_strength

        # Strong bullish signal
        strong_bullish = sufficient_confidence & (net_strength >= 2.0)

        # Strong bearish signal
        strong_bearish = sufficient_confidence & (net_strength <= -2.0)

        signal_summary[strong_bullish] = 1
        signal_summary[strong_bearish] = -1

        return signal_summary

    def get_latest_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get the latest Cipher A signal values and analysis.

        Args:
            df: DataFrame with calculated Cipher A indicators

        Returns:
            Dictionary with latest signal values, strengths, and confidence
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        # Core signal values
        values = {
            "timestamp": latest.name if hasattr(latest, "name") else None,
            "cipher_a_signal": latest.get("cipher_a_signal", 0),
            "cipher_a_bullish_strength": latest.get("cipher_a_bullish_strength", 0.0),
            "cipher_a_bearish_strength": latest.get("cipher_a_bearish_strength", 0.0),
            "cipher_a_confidence": latest.get("cipher_a_confidence", 0.0),
        }

        # Diamond patterns
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

        # Candle patterns
        candle_patterns = {
            "bull_candle": latest.get("bull_candle", False),
            "bear_candle": latest.get("bear_candle", False),
        }

        # WaveTrend values
        wavetrend_values = {
            "wt1": latest.get("wt1"),
            "wt2": latest.get("wt2"),
            "wt_cross_up": latest.get("wt_cross_up", False),
            "wt_cross_down": latest.get("wt_cross_down", False),
        }

        # Additional indicators
        other_indicators = {
            "rsi": latest.get("rsi"),
            "rsimfi": latest.get("rsimfi"),
            "ema2": latest.get("ema2"),
            "ema8": latest.get("ema8"),
        }

        # Combine all values
        values.update(diamond_signals)
        values.update(yellow_cross_signals)
        values.update(candle_patterns)
        values.update(wavetrend_values)
        values.update(other_indicators)

        return values

    def validate_signal_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate input data for Cipher A signal calculation.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None:
            return False, "DataFrame is None"

        if df.empty:
            return False, "DataFrame is empty"

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check minimum data length
        min_length = (
            max(
                self.wavetrend.channel_length,
                self.wavetrend.average_length,
                self.wavetrend.ma_length,
                self.rsi_length,
                self.rsimfi_period,
                self.ema2_length,
                self.ema8_length,
            )
            + 10
        )  # Add buffer for lookbacks

        if len(df) < min_length:
            return (
                False,
                f"Insufficient data length. Need at least {min_length}, got {len(df)}",
            )

        # Validate OHLC relationships
        try:
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False, f"Column {col} is not numeric"

                if df[col].isna().all():
                    return False, f"Column {col} contains only NaN values"

            # Check OHLC logic
            invalid_rows = (df["high"] < df["low"]).sum()
            if invalid_rows > 0:
                return False, f"Found {invalid_rows} rows where high < low"

        except Exception as e:
            return False, f"Data validation error: {str(e)}"

        return True, ""

    def _validate_input_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame quality and log issues.

        Args:
            df: Input DataFrame to validate
        """
        # Check required columns
        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(
                "Missing required columns in Cipher A input data",
                extra={
                    "indicator": "cipher_a_signals",
                    "issue": "missing_columns",
                    "missing_columns": missing_columns,
                    "available_columns": list(df.columns),
                },
            )

        # Check for NaN values in price data
        for col in required_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(
                        f"NaN values in {col} column",
                        extra={
                            "indicator": "cipher_a_signals",
                            "issue": "nan_values_in_price_data",
                            "column": col,
                            "nan_count": int(nan_count),
                            "total_points": len(df),
                            "nan_percentage": round((nan_count / len(df)) * 100, 2),
                        },
                    )

        # Check for invalid price relationships
        if all(col in df.columns for col in required_columns):
            invalid_ohlc = (df["high"] < df["low"]).sum()
            if invalid_ohlc > 0:
                logger.warning(
                    "Invalid OHLC relationships in Cipher A input data",
                    extra={
                        "indicator": "cipher_a_signals",
                        "issue": "invalid_ohlc_relationships",
                        "invalid_count": int(invalid_ohlc),
                    },
                )

            # Check for zero or negative prices
            for col in required_columns:
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    logger.warning(
                        f"Invalid {col} prices detected",
                        extra={
                            "indicator": "cipher_a_signals",
                            "issue": "invalid_prices",
                            "column": col,
                            "invalid_count": int(invalid_prices),
                            "min_price": (
                                float(df[col].min()) if not df[col].empty else 0
                            ),
                        },
                    )

        # Check data continuity
        if hasattr(df.index, "duplicated"):
            duplicate_count = df.index.duplicated().sum()
            if duplicate_count > 0:
                logger.warning(
                    "Duplicate timestamps in Cipher A input data",
                    extra={
                        "indicator": "cipher_a_signals",
                        "issue": "duplicate_timestamps",
                        "duplicate_count": int(duplicate_count),
                    },
                )

        # Check for extreme price volatility
        if "close" in df.columns and len(df) > 1:
            price_changes = df["close"].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
            if extreme_changes > 0:
                logger.warning(
                    "Extreme price volatility in Cipher A input data",
                    extra={
                        "indicator": "cipher_a_signals",
                        "issue": "extreme_price_volatility",
                        "extreme_change_count": int(extreme_changes),
                        "max_change_pct": (
                            round(float(price_changes.max()) * 100, 2)
                            if not price_changes.empty
                            else 0
                        ),
                    },
                )

        self._last_data_quality_check = time.time()

    def _generate_complete_signal_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate comprehensive summary of all Cipher A signals for logging.

        Args:
            df: DataFrame with calculated Cipher A signals

        Returns:
            Dictionary with complete signal summary
        """
        summary = {
            "total_data_points": len(df),
            "signal_counts": {},
            "signal_frequencies": {},
            "strength_analysis": {},
            "pattern_analysis": {},
        }

        # Count all signal types
        signal_columns = [
            "red_diamond",
            "green_diamond",
            "dump_diamond",
            "moon_diamond",
            "yellow_cross_up",
            "yellow_cross_down",
            "bull_candle",
            "bear_candle",
            "wt_cross_up",
            "wt_cross_down",
        ]

        for col in signal_columns:
            if col in df.columns:
                count = df[col].sum() if df[col].dtype == bool else (df[col] == 1).sum()
                summary["signal_counts"][col] = int(count)
                summary["signal_frequencies"][col] = (
                    round((count / len(df)) * 100, 2) if len(df) > 0 else 0
                )

        # Analyze signal strengths
        if "cipher_a_bullish_strength" in df.columns:
            bullish_strength = df["cipher_a_bullish_strength"].dropna()
            if not bullish_strength.empty:
                summary["strength_analysis"]["bullish"] = {
                    "mean": round(float(bullish_strength.mean()), 2),
                    "max": round(float(bullish_strength.max()), 2),
                    "strong_signals": int((bullish_strength >= 2.0).sum()),
                }

        if "cipher_a_bearish_strength" in df.columns:
            bearish_strength = df["cipher_a_bearish_strength"].dropna()
            if not bearish_strength.empty:
                summary["strength_analysis"]["bearish"] = {
                    "mean": round(float(bearish_strength.mean()), 2),
                    "max": round(float(bearish_strength.max()), 2),
                    "strong_signals": int((bearish_strength >= 2.0).sum()),
                }

        if "cipher_a_confidence" in df.columns:
            confidence = df["cipher_a_confidence"].dropna()
            if not confidence.empty:
                summary["strength_analysis"]["confidence"] = {
                    "mean": round(float(confidence.mean()), 2),
                    "max": round(float(confidence.max()), 2),
                    "high_confidence": int((confidence >= 75.0).sum()),
                }

        # Pattern combination analysis
        diamond_signals = summary["signal_counts"].get("red_diamond", 0) + summary[
            "signal_counts"
        ].get("green_diamond", 0)
        extreme_signals = summary["signal_counts"].get("dump_diamond", 0) + summary[
            "signal_counts"
        ].get("moon_diamond", 0)
        yellow_signals = summary["signal_counts"].get("yellow_cross_up", 0) + summary[
            "signal_counts"
        ].get("yellow_cross_down", 0)
        candle_signals = summary["signal_counts"].get("bull_candle", 0) + summary[
            "signal_counts"
        ].get("bear_candle", 0)

        summary["pattern_analysis"] = {
            "total_diamond_patterns": diamond_signals,
            "total_extreme_patterns": extreme_signals,
            "total_yellow_crosses": yellow_signals,
            "total_candle_patterns": candle_signals,
            "pattern_diversity": sum(
                1
                for count in [
                    diamond_signals,
                    extreme_signals,
                    yellow_signals,
                    candle_signals,
                ]
                if count > 0
            ),
        }

        # Signal distribution
        if "cipher_a_signal" in df.columns:
            signal_dist = df["cipher_a_signal"].value_counts().to_dict()
            summary["signal_distribution"] = {
                "bullish": signal_dist.get(1, 0),
                "bearish": signal_dist.get(-1, 0),
                "neutral": signal_dist.get(0, 0),
            }

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
            "signal_generation_count": self._signal_generation_count,
            "total_calculation_time_ms": round(self._total_calculation_time, 2),
            "average_calculation_time_ms": round(avg_time, 2),
            "parameters": {
                "wt_channel_length": self.wavetrend.channel_length,
                "wt_average_length": self.wavetrend.average_length,
                "wt_ma_length": self.wavetrend.ma_length,
                "overbought_levels": [
                    self.overbought_level,
                    self.overbought_level2,
                    self.overbought_level3,
                ],
                "oversold_levels": [
                    self.oversold_level,
                    self.oversold_level2,
                    self.oversold_level3,
                ],
                "rsi_length": self.rsi_length,
                "rsimfi_period": self.rsimfi_period,
                "ema_lengths": [self.ema2_length, self.ema8_length],
            },
            "last_data_quality_check": self._last_data_quality_check,
        }

        logger.debug(
            "Cipher A Signals performance metrics",
            extra={"indicator": "cipher_a_signals", "metrics": metrics},
        )

        return metrics
