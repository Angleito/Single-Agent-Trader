"""
Fractal-based divergence detection system for trading indicators.

This module implements the Pine Script fractal and divergence detection logic,
identifying regular and hidden divergences between price and various indicators.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


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


@dataclass
class FractalPoint:
    """Represents a fractal point in the data."""

    index: int
    value: float
    fractal_type: FractalType
    timestamp: pd.Timestamp | None = None


@dataclass
class DivergenceSignal:
    """Represents a detected divergence signal."""

    type: DivergenceType
    strength: float  # 0.0 to 1.0
    start_fractal: FractalPoint
    end_fractal: FractalPoint
    price_trend: str  # "higher_high", "lower_low", etc.
    indicator_trend: str
    confidence: float  # 0.0 to 1.0


class DivergenceDetector:
    """
    Fractal-based divergence detection system.

    Implements Pine Script fractal detection and divergence analysis logic
    for identifying regular and hidden divergences between price and indicators.
    """

    def __init__(
        self,
        lookback_period: int = 5,
        min_fractal_distance: int = 5,
        max_lookback_bars: int = 60,
        use_limits: bool = True,
    ) -> None:
        """
        Initialize the divergence detector.

        Args:
            lookback_period: Period for fractal detection (default: 5)
            min_fractal_distance: Minimum bars between fractals
            max_lookback_bars: Maximum bars to look back for divergences
            use_limits: Whether to use top/bottom limits for fractal validation
        """
        self.lookback_period = lookback_period
        self.min_fractal_distance = min_fractal_distance
        self.max_lookback_bars = max_lookback_bars
        self.use_limits = use_limits

        # Cache for fractal points
        self._fractal_cache: dict[str, list[FractalPoint]] = {}

        # Performance tracking
        self._fractal_detection_count = 0
        self._divergence_detection_count = 0
        self._total_calculation_time = 0.0

        logger.info(
            "Divergence detector initialized",
            extra={
                "indicator": "divergence_detector",
                "parameters": {
                    "lookback_period": lookback_period,
                    "min_fractal_distance": min_fractal_distance,
                    "max_lookback_bars": max_lookback_bars,
                    "use_limits": use_limits,
                },
            },
        )

    def find_fractals(
        self,
        data_series: pd.Series,
        top_limit: float | None = None,
        bot_limit: float | None = None,
    ) -> list[FractalPoint]:
        """
        Identify fractal points in a data series using Pine Script logic.

        Pine Script equivalent:
        f_top_fractal(src) => src[4] < src[2] and src[3] < src[2] and src[2] > src[1] and src[2] > src[0]
        f_bot_fractal(src) => src[4] > src[2] and src[3] > src[2] and src[2] < src[1] and src[2] < src[0]

        Args:
            data_series: Time series data to analyze
            top_limit: Optional upper limit for top fractals
            bot_limit: Optional lower limit for bottom fractals

        Returns:
            List of FractalPoint objects
        """
        if len(data_series) < self.lookback_period:
            return []

        fractals = []
        data_array = data_series.values

        # Start from index 2 (middle of 5-point window) to end-2
        for i in range(2, len(data_array) - 2):
            current_value = data_array[i]

            # Check for top fractal
            # src[4] < src[2] and src[3] < src[2] and src[2] > src[1] and src[2] > src[0]
            is_top_fractal = (
                data_array[i - 2] < current_value  # src[4] < src[2]
                and data_array[i - 1] < current_value  # src[3] < src[2]
                and current_value > data_array[i + 1]  # src[2] > src[1]
                and current_value > data_array[i + 2]  # src[2] > src[0]
            )

            # Check for bottom fractal
            # src[4] > src[2] and src[3] > src[2] and src[2] < src[1] and src[2] < src[0]
            is_bottom_fractal = (
                data_array[i - 2] > current_value  # src[4] > src[2]
                and data_array[i - 1] > current_value  # src[3] > src[2]
                and current_value < data_array[i + 1]  # src[2] < src[1]
                and current_value < data_array[i + 2]  # src[2] < src[0]
            )

            # Apply limits if specified
            if is_top_fractal and self.use_limits and top_limit is not None:
                if current_value < top_limit:
                    is_top_fractal = False

            if is_bottom_fractal and self.use_limits and bot_limit is not None:
                if current_value > bot_limit:
                    is_bottom_fractal = False

            # Create fractal point
            if is_top_fractal:
                fractal = FractalPoint(
                    index=i,
                    value=current_value,
                    fractal_type=FractalType.TOP,
                    timestamp=(
                        data_series.index[i]
                        if hasattr(data_series.index, "__getitem__")
                        else None
                    ),
                )
                fractals.append(fractal)
            elif is_bottom_fractal:
                fractal = FractalPoint(
                    index=i,
                    value=current_value,
                    fractal_type=FractalType.BOTTOM,
                    timestamp=(
                        data_series.index[i]
                        if hasattr(data_series.index, "__getitem__")
                        else None
                    ),
                )
                fractals.append(fractal)

        # Filter fractals by minimum distance
        filtered_fractals = self._filter_fractals_by_distance(fractals)

        return filtered_fractals

    def _filter_fractals_by_distance(
        self, fractals: list[FractalPoint]
    ) -> list[FractalPoint]:
        """Filter fractals to maintain minimum distance between them."""
        if not fractals:
            return []

        filtered = [fractals[0]]

        for fractal in fractals[1:]:
            last_fractal = filtered[-1]

            # Check distance from last fractal of same type
            if fractal.fractal_type == last_fractal.fractal_type:
                if fractal.index - last_fractal.index >= self.min_fractal_distance:
                    # Replace if new fractal is more extreme
                    if (
                        fractal.fractal_type == FractalType.TOP
                        and fractal.value > last_fractal.value
                    ):
                        filtered[-1] = fractal
                    elif (
                        fractal.fractal_type == FractalType.BOTTOM
                        and fractal.value < last_fractal.value
                    ):
                        filtered[-1] = fractal
                    else:
                        filtered.append(fractal)
            else:
                filtered.append(fractal)

        return filtered

    def detect_regular_divergences(
        self,
        indicator_series: pd.Series,
        price_series: pd.Series,
        top_limit: float | None = None,
        bot_limit: float | None = None,
    ) -> list[DivergenceSignal]:
        """
        Detect regular divergences between indicator and price.

        Regular Bullish: Price makes lower lows, indicator makes higher lows
        Regular Bearish: Price makes higher highs, indicator makes lower highs

        Args:
            indicator_series: Indicator values (e.g., RSI, WT)
            price_series: Price values (typically close prices)
            top_limit: Upper limit for top fractals
            bot_limit: Lower limit for bottom fractals

        Returns:
            List of DivergenceSignal objects
        """
        # Find fractals in both series
        indicator_fractals = self.find_fractals(indicator_series, top_limit, bot_limit)
        price_fractals = self.find_fractals(price_series)

        divergences = []

        # Detect bullish regular divergences (bottom fractals)
        bullish_divs = self._detect_bullish_regular_divergences(
            indicator_fractals, price_fractals, indicator_series, price_series
        )
        divergences.extend(bullish_divs)

        # Detect bearish regular divergences (top fractals)
        bearish_divs = self._detect_bearish_regular_divergences(
            indicator_fractals, price_fractals, indicator_series, price_series
        )
        divergences.extend(bearish_divs)

        return divergences

    def _detect_bullish_regular_divergences(
        self,
        indicator_fractals: list[FractalPoint],
        price_fractals: list[FractalPoint],
        indicator_series: pd.Series,
        price_series: pd.Series,
    ) -> list[DivergenceSignal]:
        """Detect bullish regular divergences using bottom fractals."""
        divergences = []

        # Get bottom fractals only
        indicator_bottoms = [
            f for f in indicator_fractals if f.fractal_type == FractalType.BOTTOM
        ]
        price_bottoms = [
            f for f in price_fractals if f.fractal_type == FractalType.BOTTOM
        ]

        if len(indicator_bottoms) < 2 or len(price_bottoms) < 2:
            return divergences

        # Look for divergences between recent bottom fractals
        for i in range(len(indicator_bottoms) - 1):
            for j in range(i + 1, len(indicator_bottoms)):
                ind_fractal1 = indicator_bottoms[i]
                ind_fractal2 = indicator_bottoms[j]

                # Find corresponding price fractals within reasonable time window
                price_fractal1 = self._find_nearest_fractal(
                    price_bottoms, ind_fractal1.index, FractalType.BOTTOM
                )
                price_fractal2 = self._find_nearest_fractal(
                    price_bottoms, ind_fractal2.index, FractalType.BOTTOM
                )

                if price_fractal1 and price_fractal2:
                    # Check for bullish regular divergence
                    # Price: lower low, Indicator: higher low
                    price_lower_low = price_fractal2.value < price_fractal1.value
                    indicator_higher_low = ind_fractal2.value > ind_fractal1.value

                    if price_lower_low and indicator_higher_low:
                        # Calculate divergence strength and confidence
                        strength = self._calculate_divergence_strength(
                            ind_fractal1, ind_fractal2, price_fractal1, price_fractal2
                        )
                        confidence = self._calculate_confidence(
                            ind_fractal1, ind_fractal2, indicator_series, price_series
                        )

                        divergence = DivergenceSignal(
                            type=DivergenceType.REGULAR_BULLISH,
                            strength=strength,
                            start_fractal=ind_fractal1,
                            end_fractal=ind_fractal2,
                            price_trend="lower_low",
                            indicator_trend="higher_low",
                            confidence=confidence,
                        )
                        divergences.append(divergence)

        return divergences

    def _detect_bearish_regular_divergences(
        self,
        indicator_fractals: list[FractalPoint],
        price_fractals: list[FractalPoint],
        indicator_series: pd.Series,
        price_series: pd.Series,
    ) -> list[DivergenceSignal]:
        """Detect bearish regular divergences using top fractals."""
        divergences = []

        # Get top fractals only
        indicator_tops = [
            f for f in indicator_fractals if f.fractal_type == FractalType.TOP
        ]
        price_tops = [f for f in price_fractals if f.fractal_type == FractalType.TOP]

        if len(indicator_tops) < 2 or len(price_tops) < 2:
            return divergences

        # Look for divergences between recent top fractals
        for i in range(len(indicator_tops) - 1):
            for j in range(i + 1, len(indicator_tops)):
                ind_fractal1 = indicator_tops[i]
                ind_fractal2 = indicator_tops[j]

                # Find corresponding price fractals within reasonable time window
                price_fractal1 = self._find_nearest_fractal(
                    price_tops, ind_fractal1.index, FractalType.TOP
                )
                price_fractal2 = self._find_nearest_fractal(
                    price_tops, ind_fractal2.index, FractalType.TOP
                )

                if price_fractal1 and price_fractal2:
                    # Check for bearish regular divergence
                    # Price: higher high, Indicator: lower high
                    price_higher_high = price_fractal2.value > price_fractal1.value
                    indicator_lower_high = ind_fractal2.value < ind_fractal1.value

                    if price_higher_high and indicator_lower_high:
                        # Calculate divergence strength and confidence
                        strength = self._calculate_divergence_strength(
                            ind_fractal1, ind_fractal2, price_fractal1, price_fractal2
                        )
                        confidence = self._calculate_confidence(
                            ind_fractal1, ind_fractal2, indicator_series, price_series
                        )

                        divergence = DivergenceSignal(
                            type=DivergenceType.REGULAR_BEARISH,
                            strength=strength,
                            start_fractal=ind_fractal1,
                            end_fractal=ind_fractal2,
                            price_trend="higher_high",
                            indicator_trend="lower_high",
                            confidence=confidence,
                        )
                        divergences.append(divergence)

        return divergences

    def detect_hidden_divergences(
        self, indicator_series: pd.Series, price_series: pd.Series
    ) -> list[DivergenceSignal]:
        """
        Detect hidden divergences between indicator and price.

        Hidden Bullish: Price makes higher lows, indicator makes lower lows
        Hidden Bearish: Price makes lower highs, indicator makes higher highs

        Args:
            indicator_series: Indicator values
            price_series: Price values

        Returns:
            List of DivergenceSignal objects
        """
        # Find fractals in both series
        indicator_fractals = self.find_fractals(indicator_series)
        price_fractals = self.find_fractals(price_series)

        divergences = []

        # Detect hidden bullish divergences
        hidden_bullish = self._detect_hidden_bullish_divergences(
            indicator_fractals, price_fractals, indicator_series, price_series
        )
        divergences.extend(hidden_bullish)

        # Detect hidden bearish divergences
        hidden_bearish = self._detect_hidden_bearish_divergences(
            indicator_fractals, price_fractals, indicator_series, price_series
        )
        divergences.extend(hidden_bearish)

        return divergences

    def _detect_hidden_bullish_divergences(
        self,
        indicator_fractals: list[FractalPoint],
        price_fractals: list[FractalPoint],
        indicator_series: pd.Series,
        price_series: pd.Series,
    ) -> list[DivergenceSignal]:
        """Detect hidden bullish divergences (price higher lows, indicator lower lows)."""
        divergences = []

        indicator_bottoms = [
            f for f in indicator_fractals if f.fractal_type == FractalType.BOTTOM
        ]
        price_bottoms = [
            f for f in price_fractals if f.fractal_type == FractalType.BOTTOM
        ]

        if len(indicator_bottoms) < 2 or len(price_bottoms) < 2:
            return divergences

        for i in range(len(indicator_bottoms) - 1):
            for j in range(i + 1, len(indicator_bottoms)):
                ind_fractal1 = indicator_bottoms[i]
                ind_fractal2 = indicator_bottoms[j]

                price_fractal1 = self._find_nearest_fractal(
                    price_bottoms, ind_fractal1.index, FractalType.BOTTOM
                )
                price_fractal2 = self._find_nearest_fractal(
                    price_bottoms, ind_fractal2.index, FractalType.BOTTOM
                )

                if price_fractal1 and price_fractal2:
                    # Hidden bullish: price higher low, indicator lower low
                    price_higher_low = price_fractal2.value > price_fractal1.value
                    indicator_lower_low = ind_fractal2.value < ind_fractal1.value

                    if price_higher_low and indicator_lower_low:
                        strength = self._calculate_divergence_strength(
                            ind_fractal1, ind_fractal2, price_fractal1, price_fractal2
                        )
                        confidence = self._calculate_confidence(
                            ind_fractal1, ind_fractal2, indicator_series, price_series
                        )

                        divergence = DivergenceSignal(
                            type=DivergenceType.HIDDEN_BULLISH,
                            strength=strength,
                            start_fractal=ind_fractal1,
                            end_fractal=ind_fractal2,
                            price_trend="higher_low",
                            indicator_trend="lower_low",
                            confidence=confidence,
                        )
                        divergences.append(divergence)

        return divergences

    def _detect_hidden_bearish_divergences(
        self,
        indicator_fractals: list[FractalPoint],
        price_fractals: list[FractalPoint],
        indicator_series: pd.Series,
        price_series: pd.Series,
    ) -> list[DivergenceSignal]:
        """Detect hidden bearish divergences (price lower highs, indicator higher highs)."""
        divergences = []

        indicator_tops = [
            f for f in indicator_fractals if f.fractal_type == FractalType.TOP
        ]
        price_tops = [f for f in price_fractals if f.fractal_type == FractalType.TOP]

        if len(indicator_tops) < 2 or len(price_tops) < 2:
            return divergences

        for i in range(len(indicator_tops) - 1):
            for j in range(i + 1, len(indicator_tops)):
                ind_fractal1 = indicator_tops[i]
                ind_fractal2 = indicator_tops[j]

                price_fractal1 = self._find_nearest_fractal(
                    price_tops, ind_fractal1.index, FractalType.TOP
                )
                price_fractal2 = self._find_nearest_fractal(
                    price_tops, ind_fractal2.index, FractalType.TOP
                )

                if price_fractal1 and price_fractal2:
                    # Hidden bearish: price lower high, indicator higher high
                    price_lower_high = price_fractal2.value < price_fractal1.value
                    indicator_higher_high = ind_fractal2.value > ind_fractal1.value

                    if price_lower_high and indicator_higher_high:
                        strength = self._calculate_divergence_strength(
                            ind_fractal1, ind_fractal2, price_fractal1, price_fractal2
                        )
                        confidence = self._calculate_confidence(
                            ind_fractal1, ind_fractal2, indicator_series, price_series
                        )

                        divergence = DivergenceSignal(
                            type=DivergenceType.HIDDEN_BEARISH,
                            strength=strength,
                            start_fractal=ind_fractal1,
                            end_fractal=ind_fractal2,
                            price_trend="lower_high",
                            indicator_trend="higher_high",
                            confidence=confidence,
                        )
                        divergences.append(divergence)

        return divergences

    def _find_nearest_fractal(
        self,
        fractals: list[FractalPoint],
        target_index: int,
        fractal_type: FractalType,
        max_distance: int = 10,
    ) -> FractalPoint | None:
        """Find the nearest fractal of specified type to the target index."""
        nearest_fractal = None
        min_distance = float("inf")

        for fractal in fractals:
            if fractal.fractal_type == fractal_type:
                distance = abs(fractal.index - target_index)
                if distance <= max_distance and distance < min_distance:
                    min_distance = distance
                    nearest_fractal = fractal

        return nearest_fractal

    def _calculate_divergence_strength(
        self,
        ind_fractal1: FractalPoint,
        ind_fractal2: FractalPoint,
        price_fractal1: FractalPoint,
        price_fractal2: FractalPoint,
    ) -> float:
        """Calculate the strength of a divergence based on the magnitude of differences."""
        # Calculate relative changes
        indicator_change = abs(ind_fractal2.value - ind_fractal1.value) / abs(
            ind_fractal1.value + 1e-8
        )
        price_change = abs(price_fractal2.value - price_fractal1.value) / abs(
            price_fractal1.value + 1e-8
        )

        # Strength is based on the ratio of changes
        if price_change > 0:
            strength = min(indicator_change / price_change, 1.0)
        else:
            strength = indicator_change

        return max(0.0, min(1.0, strength))

    def _calculate_confidence(
        self,
        ind_fractal1: FractalPoint,
        ind_fractal2: FractalPoint,
        indicator_series: pd.Series,
        price_series: pd.Series,
    ) -> float:
        """Calculate confidence score for a divergence based on various factors."""
        # Factor 1: Time distance between fractals (closer = higher confidence)
        time_distance = abs(ind_fractal2.index - ind_fractal1.index)
        time_factor = max(0.1, 1.0 - (time_distance / self.max_lookback_bars))

        # Factor 2: Magnitude of divergence
        magnitude_factor = self._calculate_divergence_strength(
            ind_fractal1,
            ind_fractal2,
            FractalPoint(
                ind_fractal1.index,
                price_series.iloc[ind_fractal1.index],
                FractalType.NONE,
            ),
            FractalPoint(
                ind_fractal2.index,
                price_series.iloc[ind_fractal2.index],
                FractalType.NONE,
            ),
        )

        # Factor 3: Fractal clarity (how well-defined the fractal is)
        clarity_factor = 0.8  # Simplified - could be enhanced with volatility analysis

        # Combine factors
        confidence = time_factor * 0.4 + magnitude_factor * 0.4 + clarity_factor * 0.2

        return max(0.0, min(1.0, confidence))

    def get_divergence_signals(
        self,
        divergences: list[DivergenceSignal],
        min_confidence: float = 0.5,
        min_strength: float = 0.3,
    ) -> list[DivergenceSignal]:
        """
        Filter and return high-quality divergence signals.

        Args:
            divergences: List of detected divergences
            min_confidence: Minimum confidence threshold
            min_strength: Minimum strength threshold

        Returns:
            Filtered list of high-quality divergence signals
        """
        # Filter by confidence and strength thresholds
        quality_signals = [
            div
            for div in divergences
            if div.confidence >= min_confidence and div.strength >= min_strength
        ]

        # Sort by confidence and strength
        quality_signals.sort(
            key=lambda x: (x.confidence + x.strength) / 2, reverse=True
        )

        return quality_signals

    def analyze_multiple_timeframes(
        self,
        data_dict: dict[str, tuple[pd.Series, pd.Series]],
        timeframe_weights: dict[str, float] | None = None,
    ) -> dict[str, list[DivergenceSignal]]:
        """
        Analyze divergences across multiple timeframes.

        Args:
            data_dict: Dictionary mapping timeframe to (indicator_series, price_series)
            timeframe_weights: Optional weights for each timeframe

        Returns:
            Dictionary mapping timeframe to divergence signals
        """
        if timeframe_weights is None:
            timeframe_weights = {tf: 1.0 for tf in data_dict.keys()}

        results = {}

        for timeframe, (indicator_series, price_series) in data_dict.items():
            # Detect regular divergences
            regular_divs = self.detect_regular_divergences(
                indicator_series, price_series
            )

            # Detect hidden divergences
            hidden_divs = self.detect_hidden_divergences(indicator_series, price_series)

            # Combine and weight the signals
            all_divs = regular_divs + hidden_divs
            weight = timeframe_weights.get(timeframe, 1.0)

            # Apply timeframe weight to confidence scores
            for div in all_divs:
                div.confidence *= weight

            results[timeframe] = all_divs

        return results

    def get_latest_signals(
        self, divergences: list[DivergenceSignal], lookback_bars: int = 20
    ) -> list[DivergenceSignal]:
        """
        Get the most recent divergence signals within the lookback period.

        Args:
            divergences: List of all divergence signals
            lookback_bars: Number of bars to look back for recent signals

        Returns:
            List of recent divergence signals
        """
        if not divergences:
            return []

        # Find the latest index across all divergences
        latest_index = max(div.end_fractal.index for div in divergences)
        cutoff_index = latest_index - lookback_bars

        # Filter for recent signals
        recent_signals = [
            div for div in divergences if div.end_fractal.index >= cutoff_index
        ]

        return recent_signals
