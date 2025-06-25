"""
Pure functional divergence detection for technical indicators.

This module provides immutable, side-effect-free functions for detecting
divergences between price action and oscillator indicators (RSI, MACD, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DivergenceType(Enum):
    """Types of divergences that can be detected."""

    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"


class FractalType(Enum):
    """Types of fractals in price/indicator data."""

    TOP = 1
    BOTTOM = -1
    NONE = 0


@dataclass(frozen=True)
class Fractal:
    """Represents a fractal point in the data."""

    index: int
    value: float
    fractal_type: FractalType


@dataclass(frozen=True)
class DivergenceSignal:
    """Represents a detected divergence signal."""

    divergence_type: DivergenceType
    strength: float  # 0.0 to 1.0
    start_index: int
    end_index: int
    start_value: float
    end_value: float
    price_trend: str  # "higher_high", "lower_low", etc.
    indicator_trend: str
    confidence: float  # 0.0 to 1.0


class DivergenceResult(NamedTuple):
    """Result from divergence detection."""

    regular_bullish: list[DivergenceSignal]
    regular_bearish: list[DivergenceSignal]
    hidden_bullish: list[DivergenceSignal]
    hidden_bearish: list[DivergenceSignal]
    all_signals: list[DivergenceSignal]


def find_fractals(
    data: NDArray[np.float64],
    lookback_left: int = 2,
    lookback_right: int = 2,
) -> list[Fractal]:
    """
    Identify fractal points in a data series using pure numpy operations.

    A top fractal is a point higher than its neighbors.
    A bottom fractal is a point lower than its neighbors.

    Args:
        data: Array of values to analyze
        lookback_left: Number of bars to look left
        lookback_right: Number of bars to look right

    Returns:
        List of Fractal objects
    """
    if len(data) < lookback_left + lookback_right + 1:
        return []

    fractals: list[Fractal] = []
    lookback_left + lookback_right + 1

    # Vectorized approach for finding local extrema
    for i in range(lookback_left, len(data) - lookback_right):
        window = data[i - lookback_left : i + lookback_right + 1]
        center_idx = lookback_left
        center_value = window[center_idx]

        # Check for top fractal
        is_top = np.all(window[:center_idx] < center_value) and np.all(
            window[center_idx + 1 :] < center_value
        )

        # Check for bottom fractal
        is_bottom = np.all(window[:center_idx] > center_value) and np.all(
            window[center_idx + 1 :] > center_value
        )

        if is_top:
            fractals.append(
                Fractal(index=i, value=center_value, fractal_type=FractalType.TOP)
            )
        elif is_bottom:
            fractals.append(
                Fractal(index=i, value=center_value, fractal_type=FractalType.BOTTOM)
            )

    return fractals


def filter_fractals_by_distance(
    fractals: list[Fractal], min_distance: int = 5
) -> list[Fractal]:
    """
    Filter fractals to maintain minimum distance between them.

    Args:
        fractals: List of fractal points
        min_distance: Minimum bars between fractals of same type

    Returns:
        Filtered list of fractals
    """
    if not fractals:
        return []

    # Group by fractal type
    tops = [f for f in fractals if f.fractal_type == FractalType.TOP]
    bottoms = [f for f in fractals if f.fractal_type == FractalType.BOTTOM]

    # Filter each group
    filtered_tops = _filter_fractal_group(tops, min_distance, keep_highest=True)
    filtered_bottoms = _filter_fractal_group(bottoms, min_distance, keep_highest=False)

    # Combine and sort by index
    return sorted(filtered_tops + filtered_bottoms, key=lambda f: f.index)


def _filter_fractal_group(
    fractals: list[Fractal], min_distance: int, keep_highest: bool
) -> list[Fractal]:
    """Filter a group of fractals maintaining minimum distance."""
    if not fractals:
        return []

    filtered: list[Fractal] = []

    for fractal in fractals:
        if not filtered:
            filtered.append(fractal)
            continue

        # Check distance from all previous fractals
        too_close = any(abs(fractal.index - f.index) < min_distance for f in filtered)

        if not too_close:
            filtered.append(fractal)
        else:
            # Replace if more extreme
            for i, f in enumerate(filtered):
                if abs(fractal.index - f.index) < min_distance:
                    if (keep_highest and fractal.value > f.value) or (
                        not keep_highest and fractal.value < f.value
                    ):
                        filtered[i] = fractal
                    break

    return filtered


def calculate_divergence_strength(
    indicator_change: float,
    price_change: float,
    time_distance: int,
    max_distance: int = 50,
) -> float:
    """
    Calculate the strength of a divergence.

    Args:
        indicator_change: Relative change in indicator
        price_change: Relative change in price
        time_distance: Bars between fractals
        max_distance: Maximum distance for normalization

    Returns:
        Strength value between 0.0 and 1.0
    """
    # Divergence magnitude component
    if abs(price_change) > 1e-8:
        magnitude = min(abs(indicator_change / price_change), 2.0) / 2.0
    else:
        magnitude = min(abs(indicator_change), 1.0)

    # Time proximity component (closer = stronger)
    proximity = 1.0 - min(time_distance / max_distance, 1.0)

    # Combine components
    strength = magnitude * 0.7 + proximity * 0.3

    return max(0.0, min(1.0, strength))


def detect_regular_bullish_divergence(
    price_fractals: list[Fractal],
    indicator_fractals: list[Fractal],
    price_data: NDArray[np.float64],
    indicator_data: NDArray[np.float64],
) -> list[DivergenceSignal]:
    """
    Detect regular bullish divergences.

    Regular bullish: Price makes lower lows, indicator makes higher lows.

    Args:
        price_fractals: Price fractal points
        indicator_fractals: Indicator fractal points
        price_data: Full price array
        indicator_data: Full indicator array

    Returns:
        List of bullish divergence signals
    """
    signals: list[DivergenceSignal] = []

    # Get bottom fractals only
    price_bottoms = [f for f in price_fractals if f.fractal_type == FractalType.BOTTOM]
    ind_bottoms = [
        f for f in indicator_fractals if f.fractal_type == FractalType.BOTTOM
    ]

    if len(price_bottoms) < 2 or len(ind_bottoms) < 2:
        return signals

    # Compare consecutive bottom fractals
    for i in range(len(price_bottoms) - 1):
        price1, price2 = price_bottoms[i], price_bottoms[i + 1]

        # Find corresponding indicator fractals
        ind_fractal = _find_nearest_fractal(ind_bottoms, price2.index, window=10)

        if ind_fractal and ind_fractal.index > price1.index:
            # Find previous indicator fractal
            prev_ind = _find_previous_fractal(
                ind_bottoms, ind_fractal.index, price1.index
            )

            if prev_ind:
                # Check for divergence pattern
                price_lower_low = price2.value < price1.value
                ind_higher_low = ind_fractal.value > prev_ind.value

                if price_lower_low and ind_higher_low:
                    # Calculate metrics
                    price_change = (price2.value - price1.value) / abs(price1.value)
                    ind_change = (ind_fractal.value - prev_ind.value) / abs(
                        prev_ind.value + 1e-8
                    )
                    time_dist = price2.index - price1.index

                    strength = calculate_divergence_strength(
                        ind_change, price_change, time_dist
                    )

                    # Calculate confidence based on clarity and consistency
                    confidence = _calculate_confidence(
                        price1.index, price2.index, price_data, indicator_data
                    )

                    signal = DivergenceSignal(
                        divergence_type=DivergenceType.REGULAR_BULLISH,
                        strength=strength,
                        start_index=price1.index,
                        end_index=price2.index,
                        start_value=prev_ind.value,
                        end_value=ind_fractal.value,
                        price_trend="lower_low",
                        indicator_trend="higher_low",
                        confidence=confidence,
                    )
                    signals.append(signal)

    return signals


def detect_regular_bearish_divergence(
    price_fractals: list[Fractal],
    indicator_fractals: list[Fractal],
    price_data: NDArray[np.float64],
    indicator_data: NDArray[np.float64],
) -> list[DivergenceSignal]:
    """
    Detect regular bearish divergences.

    Regular bearish: Price makes higher highs, indicator makes lower highs.

    Args:
        price_fractals: Price fractal points
        indicator_fractals: Indicator fractal points
        price_data: Full price array
        indicator_data: Full indicator array

    Returns:
        List of bearish divergence signals
    """
    signals: list[DivergenceSignal] = []

    # Get top fractals only
    price_tops = [f for f in price_fractals if f.fractal_type == FractalType.TOP]
    ind_tops = [f for f in indicator_fractals if f.fractal_type == FractalType.TOP]

    if len(price_tops) < 2 or len(ind_tops) < 2:
        return signals

    # Compare consecutive top fractals
    for i in range(len(price_tops) - 1):
        price1, price2 = price_tops[i], price_tops[i + 1]

        # Find corresponding indicator fractals
        ind_fractal = _find_nearest_fractal(ind_tops, price2.index, window=10)

        if ind_fractal and ind_fractal.index > price1.index:
            # Find previous indicator fractal
            prev_ind = _find_previous_fractal(ind_tops, ind_fractal.index, price1.index)

            if prev_ind:
                # Check for divergence pattern
                price_higher_high = price2.value > price1.value
                ind_lower_high = ind_fractal.value < prev_ind.value

                if price_higher_high and ind_lower_high:
                    # Calculate metrics
                    price_change = (price2.value - price1.value) / abs(price1.value)
                    ind_change = (ind_fractal.value - prev_ind.value) / abs(
                        prev_ind.value + 1e-8
                    )
                    time_dist = price2.index - price1.index

                    strength = calculate_divergence_strength(
                        ind_change, price_change, time_dist
                    )

                    confidence = _calculate_confidence(
                        price1.index, price2.index, price_data, indicator_data
                    )

                    signal = DivergenceSignal(
                        divergence_type=DivergenceType.REGULAR_BEARISH,
                        strength=strength,
                        start_index=price1.index,
                        end_index=price2.index,
                        start_value=prev_ind.value,
                        end_value=ind_fractal.value,
                        price_trend="higher_high",
                        indicator_trend="lower_high",
                        confidence=confidence,
                    )
                    signals.append(signal)

    return signals


def detect_hidden_bullish_divergence(
    price_fractals: list[Fractal],
    indicator_fractals: list[Fractal],
    price_data: NDArray[np.float64],
    indicator_data: NDArray[np.float64],
) -> list[DivergenceSignal]:
    """
    Detect hidden bullish divergences.

    Hidden bullish: Price makes higher lows, indicator makes lower lows.

    Args:
        price_fractals: Price fractal points
        indicator_fractals: Indicator fractal points
        price_data: Full price array
        indicator_data: Full indicator array

    Returns:
        List of hidden bullish divergence signals
    """
    signals: list[DivergenceSignal] = []

    # Get bottom fractals only
    price_bottoms = [f for f in price_fractals if f.fractal_type == FractalType.BOTTOM]
    ind_bottoms = [
        f for f in indicator_fractals if f.fractal_type == FractalType.BOTTOM
    ]

    if len(price_bottoms) < 2 or len(ind_bottoms) < 2:
        return signals

    # Compare consecutive bottom fractals
    for i in range(len(price_bottoms) - 1):
        price1, price2 = price_bottoms[i], price_bottoms[i + 1]

        # Find corresponding indicator fractals
        ind_fractal = _find_nearest_fractal(ind_bottoms, price2.index, window=10)

        if ind_fractal and ind_fractal.index > price1.index:
            prev_ind = _find_previous_fractal(
                ind_bottoms, ind_fractal.index, price1.index
            )

            if prev_ind:
                # Check for divergence pattern
                price_higher_low = price2.value > price1.value
                ind_lower_low = ind_fractal.value < prev_ind.value

                if price_higher_low and ind_lower_low:
                    # Calculate metrics
                    price_change = (price2.value - price1.value) / abs(price1.value)
                    ind_change = (ind_fractal.value - prev_ind.value) / abs(
                        prev_ind.value + 1e-8
                    )
                    time_dist = price2.index - price1.index

                    strength = calculate_divergence_strength(
                        ind_change, price_change, time_dist
                    )

                    confidence = _calculate_confidence(
                        price1.index, price2.index, price_data, indicator_data
                    )

                    signal = DivergenceSignal(
                        divergence_type=DivergenceType.HIDDEN_BULLISH,
                        strength=strength,
                        start_index=price1.index,
                        end_index=price2.index,
                        start_value=prev_ind.value,
                        end_value=ind_fractal.value,
                        price_trend="higher_low",
                        indicator_trend="lower_low",
                        confidence=confidence,
                    )
                    signals.append(signal)

    return signals


def detect_hidden_bearish_divergence(
    price_fractals: list[Fractal],
    indicator_fractals: list[Fractal],
    price_data: NDArray[np.float64],
    indicator_data: NDArray[np.float64],
) -> list[DivergenceSignal]:
    """
    Detect hidden bearish divergences.

    Hidden bearish: Price makes lower highs, indicator makes higher highs.

    Args:
        price_fractals: Price fractal points
        indicator_fractals: Indicator fractal points
        price_data: Full price array
        indicator_data: Full indicator array

    Returns:
        List of hidden bearish divergence signals
    """
    signals: list[DivergenceSignal] = []

    # Get top fractals only
    price_tops = [f for f in price_fractals if f.fractal_type == FractalType.TOP]
    ind_tops = [f for f in indicator_fractals if f.fractal_type == FractalType.TOP]

    if len(price_tops) < 2 or len(ind_tops) < 2:
        return signals

    # Compare consecutive top fractals
    for i in range(len(price_tops) - 1):
        price1, price2 = price_tops[i], price_tops[i + 1]

        # Find corresponding indicator fractals
        ind_fractal = _find_nearest_fractal(ind_tops, price2.index, window=10)

        if ind_fractal and ind_fractal.index > price1.index:
            prev_ind = _find_previous_fractal(ind_tops, ind_fractal.index, price1.index)

            if prev_ind:
                # Check for divergence pattern
                price_lower_high = price2.value < price1.value
                ind_higher_high = ind_fractal.value > prev_ind.value

                if price_lower_high and ind_higher_high:
                    # Calculate metrics
                    price_change = (price2.value - price1.value) / abs(price1.value)
                    ind_change = (ind_fractal.value - prev_ind.value) / abs(
                        prev_ind.value + 1e-8
                    )
                    time_dist = price2.index - price1.index

                    strength = calculate_divergence_strength(
                        ind_change, price_change, time_dist
                    )

                    confidence = _calculate_confidence(
                        price1.index, price2.index, price_data, indicator_data
                    )

                    signal = DivergenceSignal(
                        divergence_type=DivergenceType.HIDDEN_BEARISH,
                        strength=strength,
                        start_index=price1.index,
                        end_index=price2.index,
                        start_value=prev_ind.value,
                        end_value=ind_fractal.value,
                        price_trend="lower_high",
                        indicator_trend="higher_high",
                        confidence=confidence,
                    )
                    signals.append(signal)

    return signals


def detect_all_divergences(
    price_data: NDArray[np.float64],
    indicator_data: NDArray[np.float64],
    lookback_left: int = 2,
    lookback_right: int = 2,
    min_fractal_distance: int = 5,
) -> DivergenceResult:
    """
    Detect all types of divergences between price and indicator.

    Args:
        price_data: Price values array
        indicator_data: Indicator values array
        lookback_left: Bars to look left for fractals
        lookback_right: Bars to look right for fractals
        min_fractal_distance: Minimum distance between fractals

    Returns:
        DivergenceResult containing all detected divergences
    """
    # Find fractals in both series
    price_fractals = find_fractals(price_data, lookback_left, lookback_right)
    indicator_fractals = find_fractals(indicator_data, lookback_left, lookback_right)

    # Filter fractals by distance
    price_fractals = filter_fractals_by_distance(price_fractals, min_fractal_distance)
    indicator_fractals = filter_fractals_by_distance(
        indicator_fractals, min_fractal_distance
    )

    # Detect all divergence types
    regular_bullish = detect_regular_bullish_divergence(
        price_fractals, indicator_fractals, price_data, indicator_data
    )
    regular_bearish = detect_regular_bearish_divergence(
        price_fractals, indicator_fractals, price_data, indicator_data
    )
    hidden_bullish = detect_hidden_bullish_divergence(
        price_fractals, indicator_fractals, price_data, indicator_data
    )
    hidden_bearish = detect_hidden_bearish_divergence(
        price_fractals, indicator_fractals, price_data, indicator_data
    )

    # Combine all signals
    all_signals = regular_bullish + regular_bearish + hidden_bullish + hidden_bearish

    return DivergenceResult(
        regular_bullish=regular_bullish,
        regular_bearish=regular_bearish,
        hidden_bullish=hidden_bullish,
        hidden_bearish=hidden_bearish,
        all_signals=all_signals,
    )


def filter_divergences_by_quality(
    divergences: list[DivergenceSignal],
    min_strength: float = 0.3,
    min_confidence: float = 0.5,
) -> list[DivergenceSignal]:
    """
    Filter divergences by strength and confidence thresholds.

    Args:
        divergences: List of divergence signals
        min_strength: Minimum strength threshold
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of high-quality divergences
    """
    return [
        d
        for d in divergences
        if d.strength >= min_strength and d.confidence >= min_confidence
    ]


def get_strongest_divergence(
    divergences: list[DivergenceSignal],
) -> DivergenceSignal | None:
    """
    Get the strongest divergence signal.

    Args:
        divergences: List of divergence signals

    Returns:
        Strongest divergence or None if empty
    """
    if not divergences:
        return None

    return max(divergences, key=lambda d: d.strength * d.confidence)


def get_recent_divergences(
    divergences: list[DivergenceSignal], lookback_bars: int = 20
) -> list[DivergenceSignal]:
    """
    Get divergences that ended within the lookback period.

    Args:
        divergences: List of divergence signals
        lookback_bars: Number of bars to look back

    Returns:
        List of recent divergences
    """
    if not divergences:
        return []

    latest_index = max(d.end_index for d in divergences)
    cutoff_index = latest_index - lookback_bars

    return [d for d in divergences if d.end_index >= cutoff_index]


# Helper functions
def _find_nearest_fractal(
    fractals: list[Fractal], target_index: int, window: int = 10
) -> Fractal | None:
    """Find the nearest fractal to target index within window."""
    candidates = [f for f in fractals if abs(f.index - target_index) <= window]

    if not candidates:
        return None

    return min(candidates, key=lambda f: abs(f.index - target_index))


def _find_previous_fractal(
    fractals: list[Fractal], current_index: int, min_index: int
) -> Fractal | None:
    """Find the previous fractal before current index but after min index."""
    candidates = [f for f in fractals if min_index <= f.index < current_index]

    if not candidates:
        return None

    return max(candidates, key=lambda f: f.index)


def _calculate_confidence(
    start_index: int,
    end_index: int,
    price_data: NDArray[np.float64],
    indicator_data: NDArray[np.float64],
) -> float:
    """
    Calculate confidence score based on divergence clarity.

    Args:
        start_index: Start index of divergence
        end_index: End index of divergence
        price_data: Price array
        indicator_data: Indicator array

    Returns:
        Confidence score between 0 and 1
    """
    # Time distance component
    time_distance = end_index - start_index
    time_factor = np.exp(-time_distance / 50)  # Decay over time

    # Trend consistency component
    price_segment = price_data[start_index : end_index + 1]
    ind_segment = indicator_data[start_index : end_index + 1]

    # Calculate trend consistency using linear regression
    if len(price_segment) > 2:
        price_trend = np.polyfit(np.arange(len(price_segment)), price_segment, 1)[0]
        ind_trend = np.polyfit(np.arange(len(ind_segment)), ind_segment, 1)[0]

        # Normalize trends
        price_consistency = abs(price_trend) / (np.std(price_segment) + 1e-8)
        ind_consistency = abs(ind_trend) / (np.std(ind_segment) + 1e-8)

        trend_factor = min(price_consistency + ind_consistency, 2.0) / 2.0
    else:
        trend_factor = 0.5

    # Combine factors
    confidence = time_factor * 0.4 + trend_factor * 0.6

    return max(0.0, min(1.0, confidence))


# Convenience functions for specific indicators
def detect_rsi_divergences(
    price_data: NDArray[np.float64],
    rsi_data: NDArray[np.float64],
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> DivergenceResult:
    """
    Detect divergences between price and RSI with overbought/oversold filters.

    Args:
        price_data: Price values
        rsi_data: RSI values
        overbought: RSI overbought level
        oversold: RSI oversold level

    Returns:
        DivergenceResult with filtered divergences
    """
    result = detect_all_divergences(price_data, rsi_data)

    # Filter regular bearish by overbought condition
    result.regular_bearish[:] = [
        d for d in result.regular_bearish if rsi_data[d.end_index] >= overbought
    ]

    # Filter regular bullish by oversold condition
    result.regular_bullish[:] = [
        d for d in result.regular_bullish if rsi_data[d.end_index] <= oversold
    ]

    # Update all_signals
    result.all_signals[:] = (
        result.regular_bullish
        + result.regular_bearish
        + result.hidden_bullish
        + result.hidden_bearish
    )

    return result


def detect_macd_divergences(
    price_data: NDArray[np.float64],
    macd_histogram: NDArray[np.float64],
) -> DivergenceResult:
    """
    Detect divergences between price and MACD histogram.

    Args:
        price_data: Price values
        macd_histogram: MACD histogram values

    Returns:
        DivergenceResult with MACD divergences
    """
    return detect_all_divergences(price_data, macd_histogram)


def combine_divergence_signals(
    divergence_results: list[DivergenceResult],
    weights: list[float] | None = None,
) -> list[DivergenceSignal]:
    """
    Combine divergence signals from multiple indicators with optional weighting.

    Args:
        divergence_results: List of DivergenceResult from different indicators
        weights: Optional weights for each indicator

    Returns:
        Combined list of weighted divergence signals
    """
    if not divergence_results:
        return []

    if weights is None:
        weights = [1.0] * len(divergence_results)

    combined_signals: list[DivergenceSignal] = []

    for result, weight in zip(divergence_results, weights, strict=False):
        for signal in result.all_signals:
            # Create weighted copy
            weighted_signal = DivergenceSignal(
                divergence_type=signal.divergence_type,
                strength=signal.strength * weight,
                start_index=signal.start_index,
                end_index=signal.end_index,
                start_value=signal.start_value,
                end_value=signal.end_value,
                price_trend=signal.price_trend,
                indicator_trend=signal.indicator_trend,
                confidence=signal.confidence * weight,
            )
            combined_signals.append(weighted_signal)

    # Sort by combined score
    combined_signals.sort(key=lambda s: s.strength * s.confidence, reverse=True)

    return combined_signals
