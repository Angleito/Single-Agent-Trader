"""Technical indicators and analysis modules."""

from .ema_ribbon import EMAribbon
from .fast_ema import FastEMA, ScalpingEMASignals
from .rsimfi import RSIMFIIndicator
from .scalping_momentum import FastRSI, FastMACD, WilliamsPercentR, ScalpingMomentumSignals
from .scalping_volume import (
    ScalpingVWAP,
    OnBalanceVolume,
    VolumeMovingAverage,
    VolumeProfile,
    ScalpingVolumeSignals,
)
from .schaff_trend_cycle import SchaffTrendCycle
from .stochastic_rsi import StochasticRSI
from .vumanchu import CipherA, CipherB, VuManChuIndicators
from .unified_framework import (
    UnifiedIndicatorFramework,
    TimeframeType,
    IndicatorType,
    unified_framework,
    calculate_indicators_for_strategy,
    get_available_indicators_for_timeframe,
    get_framework_performance,
)

__all__ = [
    "CipherA",
    "CipherB", 
    "VuManChuIndicators",
    "RSIMFIIndicator",
    "StochasticRSI",
    "SchaffTrendCycle",
    "EMAribbon",
    "FastEMA",
    "ScalpingEMASignals",
    "FastRSI",
    "FastMACD",
    "WilliamsPercentR",
    "ScalpingMomentumSignals",
    "ScalpingVWAP",
    "OnBalanceVolume",
    "VolumeMovingAverage",
    "VolumeProfile",
    "ScalpingVolumeSignals",
    # Unified Framework
    "UnifiedIndicatorFramework",
    "TimeframeType",
    "IndicatorType",
    "unified_framework",
    "calculate_indicators_for_strategy",
    "get_available_indicators_for_timeframe",
    "get_framework_performance",
]

# Utility functions for strategy manager
def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range for volatility measurement."""
    import numpy as np
    
    if len(highs) < period + 1:
        return 0.0
        
    true_ranges = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i-1])
        low_close_prev = abs(lows[i] - closes[i-1])
        
        true_range = max(high_low, high_close_prev, low_close_prev)
        true_ranges.append(true_range)
    
    if len(true_ranges) >= period:
        return np.mean(true_ranges[-period:])
    else:
        return np.mean(true_ranges) if true_ranges else 0.0

def calculate_support_resistance(highs, lows, window=20):
    """Calculate basic support and resistance levels."""
    import numpy as np
    
    if len(highs) < window or len(lows) < window:
        return {'support': 0.0, 'resistance': 0.0}
    
    recent_highs = highs[-window:]
    recent_lows = lows[-window:]
    
    resistance = np.max(recent_highs)
    support = np.min(recent_lows)
    
    return {'support': support, 'resistance': resistance}
