"""Pure functional market regime detection using state machines.

This module implements market state detection through pure functional
state machines that identify trending, ranging, and volatility regimes.
"""

from functools import reduce
from typing import Literal, TypedDict, cast

import numpy as np

# Market state types
MarketState = Literal["trending_up", "trending_down", "ranging", "volatile"]
VolumeState = Literal["accumulation", "distribution", "neutral"]


class RegimeMetrics(TypedDict):
    """Metrics for regime detection."""

    trend_strength: float
    volatility_ratio: float
    volume_trend: float
    price_momentum: float
    regime_stability: float


class StateTransition(TypedDict):
    """State transition information."""

    from_state: MarketState
    to_state: MarketState
    probability: float
    confidence: float


def calculate_trend_strength(prices: np.ndarray, lookback: int) -> float:
    """Calculate trend strength using directional movement.

    Args:
        prices: Price array
        lookback: Lookback period

    Returns:
        Trend strength between -1 (strong down) and 1 (strong up)
    """
    if len(prices) < lookback + 1:
        return 0.0

    # Calculate directional movements
    returns = np.diff(prices[-lookback - 1 :])
    up_moves = np.where(returns > 0, returns, 0)
    down_moves = np.where(returns < 0, -returns, 0)

    # Smoothed directional indicators
    alpha = 2 / (lookback + 1)
    up_sum = reduce(lambda acc, x: acc * (1 - alpha) + x * alpha, up_moves, 0.0)
    down_sum = reduce(lambda acc, x: acc * (1 - alpha) + x * alpha, down_moves, 0.0)

    # Directional index
    total = up_sum + down_sum
    if total == 0:
        return 0.0

    return (up_sum - down_sum) / total


def calculate_volatility_ratio(prices: np.ndarray, lookback: int) -> float:
    """Calculate volatility ratio for regime detection.

    Args:
        prices: Price array
        lookback: Lookback period

    Returns:
        Volatility ratio (current vs historical)
    """
    if len(prices) < lookback * 2:
        return 1.0

    # Recent vs historical volatility
    recent_vol = np.std(np.diff(prices[-lookback:]))
    hist_vol = np.std(np.diff(prices[-lookback * 2 : -lookback]))

    if hist_vol == 0:
        return 2.0  # Max ratio when historical vol is zero

    return min(recent_vol / hist_vol, 2.0)


def calculate_volume_trend(volumes: np.ndarray, lookback: int) -> float:
    """Calculate volume trend for accumulation/distribution.

    Args:
        volumes: Volume array
        lookback: Lookback period

    Returns:
        Volume trend between -1 (distribution) and 1 (accumulation)
    """
    if len(volumes) < lookback:
        return 0.0

    # Volume momentum
    recent_avg = np.mean(volumes[-lookback // 2 :])
    older_avg = np.mean(volumes[-lookback : -lookback // 2])

    if older_avg == 0:
        return 0.0

    return np.tanh((recent_avg - older_avg) / older_avg)


def calculate_price_momentum(prices: np.ndarray, lookback: int) -> float:
    """Calculate price momentum for trend detection.

    Args:
        prices: Price array
        lookback: Lookback period

    Returns:
        Price momentum between -1 and 1
    """
    if len(prices) < lookback:
        return 0.0

    # Rate of change
    roc = (prices[-1] - prices[-lookback]) / prices[-lookback]

    # Normalize to [-1, 1]
    return np.tanh(roc * 10)  # Scale factor for sensitivity


def calculate_regime_stability(
    prices: np.ndarray, current_state: MarketState, lookback: int
) -> float:
    """Calculate stability of current regime.

    Args:
        prices: Price array
        current_state: Current market state
        lookback: Lookback period

    Returns:
        Stability score between 0 and 1
    """
    if len(prices) < lookback:
        return 0.5

    # Calculate regime-specific metrics
    returns = np.diff(prices[-lookback:])

    if current_state in ["trending_up", "trending_down"]:
        # For trends, check consistency of direction
        expected_sign = 1 if current_state == "trending_up" else -1
        consistent_returns = np.sum(np.sign(returns) == expected_sign) / len(returns)
        return consistent_returns

    if current_state == "ranging":
        # For ranging, check mean reversion
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 1.0
        # Low absolute mean relative to volatility indicates ranging
        return 1.0 - min(abs(mean_return) / std_return, 1.0)

    # volatile
    # For volatile, check high variance consistency
    rolling_vars = np.array(
        [np.var(returns[i : i + 5]) for i in range(len(returns) - 4)]
    )
    if len(rolling_vars) > 0:
        high_var_ratio = np.sum(rolling_vars > np.median(rolling_vars)) / len(
            rolling_vars
        )
        return high_var_ratio
    return 0.5


def classify_market_state(metrics: RegimeMetrics) -> MarketState:
    """Classify market state based on metrics.

    Args:
        metrics: Calculated regime metrics

    Returns:
        Classified market state
    """
    # High volatility regime
    if metrics["volatility_ratio"] > 1.5:
        return "volatile"

    # Trending regimes
    if abs(metrics["trend_strength"]) > 0.3:
        if metrics["trend_strength"] > 0 and metrics["price_momentum"] > 0.2:
            return "trending_up"
        if metrics["trend_strength"] < 0 and metrics["price_momentum"] < -0.2:
            return "trending_down"

    # Default to ranging
    return "ranging"


def calculate_transition_probabilities(
    current_state: MarketState, metrics: RegimeMetrics, stability: float
) -> dict[MarketState, float]:
    """Calculate state transition probabilities.

    Args:
        current_state: Current market state
        metrics: Current regime metrics
        stability: Current regime stability

    Returns:
        Transition probabilities to each state
    """
    # Base transition matrix (empirical)
    base_transitions = {
        "trending_up": {
            "trending_up": 0.6,
            "trending_down": 0.1,
            "ranging": 0.2,
            "volatile": 0.1,
        },
        "trending_down": {
            "trending_up": 0.1,
            "trending_down": 0.6,
            "ranging": 0.2,
            "volatile": 0.1,
        },
        "ranging": {
            "trending_up": 0.2,
            "trending_down": 0.2,
            "ranging": 0.5,
            "volatile": 0.1,
        },
        "volatile": {
            "trending_up": 0.15,
            "trending_down": 0.15,
            "ranging": 0.3,
            "volatile": 0.4,
        },
    }

    # Get base probabilities
    base_probs = base_transitions[current_state].copy()

    # Adjust based on current metrics
    adjustments = {
        "trending_up": metrics["trend_strength"] * 0.2
        + metrics["price_momentum"] * 0.1,
        "trending_down": -metrics["trend_strength"] * 0.2
        - metrics["price_momentum"] * 0.1,
        "ranging": (1 - abs(metrics["trend_strength"])) * 0.15,
        "volatile": metrics["volatility_ratio"] * 0.1,
    }

    # Apply stability factor
    stability_factor = 0.5 + stability * 0.5

    # Calculate adjusted probabilities
    adjusted_probs = {}
    for state in cast(
        "list[MarketState]", ["trending_up", "trending_down", "ranging", "volatile"]
    ):
        if state == current_state:
            # Increase probability of staying in current state based on stability
            adjusted_probs[state] = (
                base_probs[state] * stability_factor + adjustments[state]
            )
        else:
            # Decrease transition probabilities based on stability
            adjusted_probs[state] = (
                base_probs[state] * (2 - stability_factor) + adjustments[state]
            )

    # Normalize to ensure sum = 1
    total = sum(adjusted_probs.values())
    if total > 0:
        return {state: prob / total for state, prob in adjusted_probs.items()}

    # Fallback to equal probabilities
    return dict.fromkeys(["trending_up", "trending_down", "ranging", "volatile"], 0.25)


def detect_volume_state(
    volumes: np.ndarray, prices: np.ndarray, lookback: int
) -> VolumeState:
    """Detect volume-based market state.

    Args:
        volumes: Volume array
        prices: Price array
        lookback: Lookback period

    Returns:
        Volume state (accumulation/distribution/neutral)
    """
    if len(volumes) < lookback or len(prices) < lookback:
        return "neutral"

    # Price-volume correlation
    price_returns = np.diff(prices[-lookback:])
    volume_changes = np.diff(volumes[-lookback:])

    if len(price_returns) == 0 or len(volume_changes) == 0:
        return "neutral"

    # Calculate correlation
    if np.std(price_returns) == 0 or np.std(volume_changes) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(price_returns, volume_changes)[0, 1]

    # Volume trend
    volume_trend = calculate_volume_trend(volumes, lookback)

    # Classify based on correlation and trend
    if correlation > 0.3 and volume_trend > 0.2:
        return "accumulation"  # Rising volume with rising prices
    if correlation < -0.3 or (volume_trend > 0.2 and np.mean(price_returns) < 0):
        return "distribution"  # Rising volume with falling prices
    return "neutral"


def detect_market_regime(
    prices: np.ndarray, volumes: np.ndarray, lookback: int = 20
) -> tuple[MarketState, dict[MarketState, float]]:
    """Detect current market regime and transition probabilities.

    Args:
        prices: Price array
        volumes: Volume array
        lookback: Lookback period for calculations

    Returns:
        Tuple of (current_state, transition_probabilities)
    """
    # Calculate regime metrics
    metrics: RegimeMetrics = {
        "trend_strength": calculate_trend_strength(prices, lookback),
        "volatility_ratio": calculate_volatility_ratio(prices, lookback),
        "volume_trend": calculate_volume_trend(volumes, lookback),
        "price_momentum": calculate_price_momentum(prices, lookback),
        "regime_stability": 0.5,  # Placeholder, updated after classification
    }

    # Classify current state
    current_state = classify_market_state(metrics)

    # Calculate regime stability
    stability = calculate_regime_stability(prices, current_state, lookback)
    metrics["regime_stability"] = stability

    # Calculate transition probabilities
    transition_probs = calculate_transition_probabilities(
        current_state, metrics, stability
    )

    return current_state, transition_probs


def create_regime_report(
    prices: np.ndarray, volumes: np.ndarray, lookback: int = 20
) -> dict:
    """Create comprehensive regime analysis report.

    Args:
        prices: Price array
        volumes: Volume array
        lookback: Lookback period

    Returns:
        Detailed regime analysis
    """
    # Detect primary regime
    market_state, transition_probs = detect_market_regime(prices, volumes, lookback)

    # Detect volume state
    volume_state = detect_volume_state(volumes, prices, lookback)

    # Calculate detailed metrics
    metrics: RegimeMetrics = {
        "trend_strength": calculate_trend_strength(prices, lookback),
        "volatility_ratio": calculate_volatility_ratio(prices, lookback),
        "volume_trend": calculate_volume_trend(volumes, lookback),
        "price_momentum": calculate_price_momentum(prices, lookback),
        "regime_stability": calculate_regime_stability(prices, market_state, lookback),
    }

    # Find most likely transition
    sorted_transitions = sorted(
        transition_probs.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "current_state": market_state,
        "volume_state": volume_state,
        "metrics": metrics,
        "transition_probabilities": transition_probs,
        "most_likely_transition": sorted_transitions[0][0],
        "confidence": metrics["regime_stability"],
        "recommendation": _generate_recommendation(
            market_state, volume_state, metrics, transition_probs
        ),
    }


def _generate_recommendation(
    market_state: MarketState,
    volume_state: VolumeState,
    metrics: RegimeMetrics,
    transition_probs: dict[MarketState, float],
) -> str:
    """Generate trading recommendation based on regime analysis.

    Args:
        market_state: Current market state
        volume_state: Current volume state
        metrics: Regime metrics
        transition_probs: Transition probabilities

    Returns:
        Trading recommendation string
    """
    recommendations = []

    # Base recommendation on current state
    if market_state == "trending_up":
        if volume_state == "accumulation":
            recommendations.append("Strong bullish regime with accumulation")
        else:
            recommendations.append("Bullish trend but watch volume")

    elif market_state == "trending_down":
        if volume_state == "distribution":
            recommendations.append("Strong bearish regime with distribution")
        else:
            recommendations.append("Bearish trend but volume divergence")

    elif market_state == "ranging":
        recommendations.append("Range-bound market, consider mean reversion")

    else:  # volatile
        recommendations.append("High volatility regime, reduce position size")

    # Add transition warnings
    if transition_probs.get(market_state, 0) < 0.5:
        recommendations.append("Regime change likely, watch for transition")

    # Add stability note
    if metrics["regime_stability"] < 0.3:
        recommendations.append("Low regime stability, expect changes")

    return "; ".join(recommendations)


# Compose multiple regime detections
def detect_multi_timeframe_regime(
    prices: np.ndarray, volumes: np.ndarray, timeframes: list[int]
) -> dict[int, tuple[MarketState, dict[MarketState, float]]]:
    """Detect regimes across multiple timeframes.

    Args:
        prices: Price array
        volumes: Volume array
        timeframes: List of lookback periods

    Returns:
        Regime detection for each timeframe
    """
    return {tf: detect_market_regime(prices, volumes, tf) for tf in timeframes}
