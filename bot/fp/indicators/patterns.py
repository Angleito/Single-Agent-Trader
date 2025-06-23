"""Pure functional pattern recognition for technical analysis."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class PatternMatch(NamedTuple):
    """Result of a pattern detection."""

    pattern_type: str
    confidence: float  # 0.0 to 1.0
    start_idx: int
    end_idx: int
    metadata: dict[str, float]


class SupportResistance(NamedTuple):
    """Support or resistance level."""

    price: float
    strength: float  # Number of touches/bounces
    confidence: float
    level_type: str  # 'support' or 'resistance'


class TrendLine(NamedTuple):
    """Trend line definition."""

    start_price: float
    end_price: float
    start_idx: int
    end_idx: int
    slope: float
    confidence: float
    line_type: str  # 'support' or 'resistance'


def find_local_extrema(
    prices: NDArray[np.float64], window: int = 5
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Find local minima and maxima indices.

    Args:
        prices: Price series
        window: Window size for local extrema detection

    Returns:
        Tuple of (minima_indices, maxima_indices)
    """
    if len(prices) < window * 2 + 1:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Pad the array to handle edges
    padded = np.pad(prices, window, mode="edge")

    minima = []
    maxima = []

    for i in range(window, len(padded) - window):
        window_slice = padded[i - window : i + window + 1]
        center_idx = window

        if window_slice[center_idx] == np.min(window_slice):
            minima.append(i - window)  # Adjust for padding
        elif window_slice[center_idx] == np.max(window_slice):
            maxima.append(i - window)  # Adjust for padding

    return np.array(minima, dtype=np.int32), np.array(maxima, dtype=np.int32)


def detect_support_resistance(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    min_touches: int = 2,
    price_tolerance: float = 0.02,
) -> list[SupportResistance]:
    """Detect support and resistance levels.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        min_touches: Minimum number of touches to confirm level
        price_tolerance: Price tolerance as fraction (e.g., 0.02 = 2%)

    Returns:
        List of support and resistance levels
    """
    if len(highs) < 20:
        return []

    # Find local extrema
    low_minima, _ = find_local_extrema(lows, window=5)
    _, high_maxima = find_local_extrema(highs, window=5)

    levels = []

    # Process support levels from local minima
    if len(low_minima) > 0:
        support_prices = lows[low_minima]
        unique_supports = []

        for price in support_prices:
            # Check if this price is already in our list (within tolerance)
            is_duplicate = False
            for existing in unique_supports:
                if abs(price - existing["price"]) / existing["price"] < price_tolerance:
                    existing["count"] += 1
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_supports.append({"price": price, "count": 1})

        # Create support levels
        for support in unique_supports:
            if support["count"] >= min_touches:
                confidence = min(
                    1.0, support["count"] / 5.0
                )  # Max confidence at 5 touches
                levels.append(
                    SupportResistance(
                        price=support["price"],
                        strength=float(support["count"]),
                        confidence=confidence,
                        level_type="support",
                    )
                )

    # Process resistance levels from local maxima
    if len(high_maxima) > 0:
        resistance_prices = highs[high_maxima]
        unique_resistances = []

        for price in resistance_prices:
            # Check if this price is already in our list (within tolerance)
            is_duplicate = False
            for existing in unique_resistances:
                if abs(price - existing["price"]) / existing["price"] < price_tolerance:
                    existing["count"] += 1
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_resistances.append({"price": price, "count": 1})

        # Create resistance levels
        for resistance in unique_resistances:
            if resistance["count"] >= min_touches:
                confidence = min(
                    1.0, resistance["count"] / 5.0
                )  # Max confidence at 5 touches
                levels.append(
                    SupportResistance(
                        price=resistance["price"],
                        strength=float(resistance["count"]),
                        confidence=confidence,
                        level_type="resistance",
                    )
                )

    return sorted(levels, key=lambda x: x.price)


def detect_trend_lines(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    min_touches: int = 3,
    max_deviation: float = 0.02,
) -> list[TrendLine]:
    """Detect trend lines from price data.

    Args:
        highs: High prices
        lows: Low prices
        min_touches: Minimum touches to confirm trend line
        max_deviation: Maximum allowed deviation from line

    Returns:
        List of trend lines
    """
    if len(highs) < 20:
        return []

    # Find local extrema
    low_minima, _ = find_local_extrema(lows, window=5)
    _, high_maxima = find_local_extrema(highs, window=5)

    trend_lines = []

    # Detect support trend lines (connecting lows)
    if len(low_minima) >= min_touches:
        for i in range(len(low_minima) - 1):
            for j in range(i + 1, len(low_minima)):
                idx1, idx2 = low_minima[i], low_minima[j]
                price1, price2 = lows[idx1], lows[idx2]

                # Calculate slope
                slope = (price2 - price1) / (idx2 - idx1)

                # Check how many points touch this line
                touches = 0
                for k in range(len(low_minima)):
                    idx = low_minima[k]
                    expected_price = price1 + slope * (idx - idx1)
                    actual_price = lows[idx]

                    if (
                        abs(actual_price - expected_price) / expected_price
                        < max_deviation
                    ):
                        touches += 1

                if touches >= min_touches:
                    confidence = min(1.0, touches / 5.0)
                    trend_lines.append(
                        TrendLine(
                            start_price=price1,
                            end_price=price2,
                            start_idx=idx1,
                            end_idx=idx2,
                            slope=slope,
                            confidence=confidence,
                            line_type="support",
                        )
                    )

    # Detect resistance trend lines (connecting highs)
    if len(high_maxima) >= min_touches:
        for i in range(len(high_maxima) - 1):
            for j in range(i + 1, len(high_maxima)):
                idx1, idx2 = high_maxima[i], high_maxima[j]
                price1, price2 = highs[idx1], highs[idx2]

                # Calculate slope
                slope = (price2 - price1) / (idx2 - idx1)

                # Check how many points touch this line
                touches = 0
                for k in range(len(high_maxima)):
                    idx = high_maxima[k]
                    expected_price = price1 + slope * (idx - idx1)
                    actual_price = highs[idx]

                    if (
                        abs(actual_price - expected_price) / expected_price
                        < max_deviation
                    ):
                        touches += 1

                if touches >= min_touches:
                    confidence = min(1.0, touches / 5.0)
                    trend_lines.append(
                        TrendLine(
                            start_price=price1,
                            end_price=price2,
                            start_idx=idx1,
                            end_idx=idx2,
                            slope=slope,
                            confidence=confidence,
                            line_type="resistance",
                        )
                    )

    return trend_lines


def detect_triangle_pattern(
    highs: NDArray[np.float64], lows: NDArray[np.float64], min_length: int = 20
) -> PatternMatch | None:
    """Detect triangle patterns (ascending, descending, symmetrical).

    Args:
        highs: High prices
        lows: Low prices
        min_length: Minimum pattern length

    Returns:
        Triangle pattern if found
    """
    if len(highs) < min_length:
        return None

    # Get trend lines
    trend_lines = detect_trend_lines(highs, lows, min_touches=3)
    if len(trend_lines) < 2:
        return None

    # Look for converging trend lines
    support_lines = [line for line in trend_lines if line.line_type == "support"]
    resistance_lines = [line for line in trend_lines if line.line_type == "resistance"]

    if not support_lines or not resistance_lines:
        return None

    # Take the most confident lines
    best_support = max(support_lines, key=lambda x: x.confidence)
    best_resistance = max(resistance_lines, key=lambda x: x.confidence)

    # Check if lines are converging
    start_gap = abs(best_resistance.start_price - best_support.start_price)
    end_gap = abs(best_resistance.end_price - best_support.end_price)

    if end_gap < start_gap * 0.7:  # Lines are converging
        # Determine triangle type
        if abs(best_support.slope) < 0.001:  # Horizontal support
            pattern_type = "ascending_triangle"
        elif abs(best_resistance.slope) < 0.001:  # Horizontal resistance
            pattern_type = "descending_triangle"
        else:
            pattern_type = "symmetrical_triangle"

        confidence = (best_support.confidence + best_resistance.confidence) / 2

        return PatternMatch(
            pattern_type=pattern_type,
            confidence=confidence,
            start_idx=min(best_support.start_idx, best_resistance.start_idx),
            end_idx=max(best_support.end_idx, best_resistance.end_idx),
            metadata={
                "support_slope": best_support.slope,
                "resistance_slope": best_resistance.slope,
                "convergence_ratio": end_gap / start_gap,
            },
        )

    return None


def detect_flag_pattern(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    min_pole_length: int = 10,
    max_flag_length: int = 20,
) -> PatternMatch | None:
    """Detect flag patterns (bullish or bearish).

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        min_pole_length: Minimum length for the pole
        max_flag_length: Maximum length for the flag

    Returns:
        Flag pattern if found
    """
    if len(highs) < min_pole_length + max_flag_length:
        return None

    # Look for strong directional move (pole)
    for i in range(min_pole_length, len(closes) - max_flag_length):
        pole_start = i - min_pole_length
        pole_end = i

        # Calculate pole strength
        pole_move = closes[pole_end] - closes[pole_start]
        pole_range = np.max(highs[pole_start:pole_end]) - np.min(
            lows[pole_start:pole_end]
        )

        if abs(pole_move) > pole_range * 0.7:  # Strong directional move
            # Check for consolidation (flag)
            flag_highs = highs[pole_end : pole_end + max_flag_length]
            flag_lows = lows[pole_end : pole_end + max_flag_length]
            flag_closes = closes[pole_end : pole_end + max_flag_length]

            if len(flag_closes) < 5:
                continue

            # Calculate flag characteristics
            flag_range = np.max(flag_highs) - np.min(flag_lows)
            flag_slope = np.polyfit(range(len(flag_closes)), flag_closes, 1)[0]

            # Flag should be relatively narrow compared to pole
            if flag_range < abs(pole_move) * 0.5:
                # Check if flag is counter-trend
                if pole_move > 0 and flag_slope < 0:
                    pattern_type = "bullish_flag"
                elif pole_move < 0 and flag_slope > 0:
                    pattern_type = "bearish_flag"
                else:
                    continue

                confidence = min(1.0, abs(pole_move) / pole_range)

                return PatternMatch(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    start_idx=pole_start,
                    end_idx=pole_end + len(flag_closes) - 1,
                    metadata={
                        "pole_strength": abs(pole_move) / pole_range,
                        "flag_slope": flag_slope,
                        "flag_tightness": flag_range / abs(pole_move),
                    },
                )

    return None


def detect_candlestick_patterns(
    opens: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
) -> list[PatternMatch]:
    """Detect common candlestick patterns.

    Args:
        opens: Open prices
        highs: High prices
        lows: Low prices
        closes: Close prices

    Returns:
        List of detected candlestick patterns
    """
    if len(opens) < 3:
        return []

    patterns = []

    # Helper functions
    def body_size(i: int) -> float:
        return abs(closes[i] - opens[i])

    def upper_shadow(i: int) -> float:
        return highs[i] - max(opens[i], closes[i])

    def lower_shadow(i: int) -> float:
        return min(opens[i], closes[i]) - lows[i]

    def is_bullish(i: int) -> bool:
        return closes[i] > opens[i]

    def is_bearish(i: int) -> bool:
        return closes[i] < opens[i]

    # Scan for patterns
    for i in range(2, len(opens)):
        # Doji
        if body_size(i) < (highs[i] - lows[i]) * 0.1:
            patterns.append(
                PatternMatch(
                    pattern_type="doji",
                    confidence=0.8,
                    start_idx=i,
                    end_idx=i,
                    metadata={"body_ratio": body_size(i) / (highs[i] - lows[i])},
                )
            )

        # Hammer (bullish reversal)
        if (
            lower_shadow(i) > body_size(i) * 2
            and upper_shadow(i) < body_size(i) * 0.5
            and lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
        ):
            patterns.append(
                PatternMatch(
                    pattern_type="hammer",
                    confidence=0.85,
                    start_idx=i,
                    end_idx=i,
                    metadata={"shadow_ratio": lower_shadow(i) / body_size(i)},
                )
            )

        # Shooting Star (bearish reversal)
        if (
            upper_shadow(i) > body_size(i) * 2
            and lower_shadow(i) < body_size(i) * 0.5
            and highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
        ):
            patterns.append(
                PatternMatch(
                    pattern_type="shooting_star",
                    confidence=0.85,
                    start_idx=i,
                    end_idx=i,
                    metadata={"shadow_ratio": upper_shadow(i) / body_size(i)},
                )
            )

        # Engulfing patterns
        if i >= 1:
            # Bullish engulfing
            if (
                is_bearish(i - 1)
                and is_bullish(i)
                and opens[i] < closes[i - 1]
                and closes[i] > opens[i - 1]
            ):
                patterns.append(
                    PatternMatch(
                        pattern_type="bullish_engulfing",
                        confidence=0.9,
                        start_idx=i - 1,
                        end_idx=i,
                        metadata={"engulfing_ratio": body_size(i) / body_size(i - 1)},
                    )
                )

            # Bearish engulfing
            if (
                is_bullish(i - 1)
                and is_bearish(i)
                and opens[i] > closes[i - 1]
                and closes[i] < opens[i - 1]
            ):
                patterns.append(
                    PatternMatch(
                        pattern_type="bearish_engulfing",
                        confidence=0.9,
                        start_idx=i - 1,
                        end_idx=i,
                        metadata={"engulfing_ratio": body_size(i) / body_size(i - 1)},
                    )
                )

        # Morning/Evening Star (3-candle patterns)
        if i >= 2:
            # Morning Star (bullish reversal)
            if (
                is_bearish(i - 2)
                and body_size(i - 1) < body_size(i - 2) * 0.3
                and is_bullish(i)
                and closes[i] > (opens[i - 2] + closes[i - 2]) / 2
            ):
                patterns.append(
                    PatternMatch(
                        pattern_type="morning_star",
                        confidence=0.85,
                        start_idx=i - 2,
                        end_idx=i,
                        metadata={
                            "reversal_strength": (closes[i] - lows[i - 1])
                            / (highs[i - 2] - lows[i - 1])
                        },
                    )
                )

            # Evening Star (bearish reversal)
            if (
                is_bullish(i - 2)
                and body_size(i - 1) < body_size(i - 2) * 0.3
                and is_bearish(i)
                and closes[i] < (opens[i - 2] + closes[i - 2]) / 2
            ):
                patterns.append(
                    PatternMatch(
                        pattern_type="evening_star",
                        confidence=0.85,
                        start_idx=i - 2,
                        end_idx=i,
                        metadata={
                            "reversal_strength": (highs[i - 1] - closes[i])
                            / (highs[i - 1] - lows[i - 2])
                        },
                    )
                )

    return patterns


def analyze_patterns(
    opens: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
) -> dict[str, list[PatternMatch]]:
    """Comprehensive pattern analysis.

    Args:
        opens: Open prices
        highs: High prices
        lows: Low prices
        closes: Close prices

    Returns:
        Dictionary of pattern types to detected patterns
    """
    results = {
        "candlestick": detect_candlestick_patterns(opens, highs, lows, closes),
        "chart_patterns": [],
    }

    # Detect chart patterns
    triangle = detect_triangle_pattern(highs, lows)
    if triangle:
        results["chart_patterns"].append(triangle)

    flag = detect_flag_pattern(highs, lows, closes)
    if flag:
        results["chart_patterns"].append(flag)

    return results
