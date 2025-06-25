"""Functional combinators for composing and transforming indicators."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from bot.fp.core.types import (
    Indicator,
    Signal,
    SignalStrength,
    TimeSeries,
)

# Type aliases for clarity
IndicatorCombinator = Callable[[list[Indicator]], Indicator]
IndicatorTransformer = Callable[[Indicator], Indicator]
SignalCombinator = Callable[[list[Signal]], Signal]


def combine_signals(
    combination_fn: Callable[[list[float]], float] = np.mean,
    strength_fn: Callable[[list[float]], SignalStrength] | None = None,
) -> SignalCombinator:
    """
    Create a signal combinator that merges multiple signals.

    Args:
        combination_fn: Function to combine signal values (default: mean)
        strength_fn: Function to determine combined signal strength

    Returns:
        Function that combines multiple signals into one
    """

    def _combine(signals: list[Signal]) -> Signal:
        if not signals:
            return Signal(value=0.0, strength=SignalStrength.NEUTRAL)

        values = [s.value for s in signals]
        combined_value = combination_fn(values)

        # Default strength calculation: average of strengths
        if strength_fn is None:
            strength_values = [s.strength.value for s in signals]
            avg_strength = np.mean(strength_values)

            # Map to discrete strength levels
            if avg_strength >= SignalStrength.STRONG.value:
                strength = SignalStrength.STRONG
            elif avg_strength >= SignalStrength.MEDIUM.value:
                strength = SignalStrength.MEDIUM
            elif avg_strength >= SignalStrength.WEAK.value:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.NEUTRAL
        else:
            strength = strength_fn([s.strength.value for s in signals])

        return Signal(value=combined_value, strength=strength)

    return _combine


def smooth(
    window: int,
    method: str = "sma",
    alpha: float = 0.1,
) -> IndicatorTransformer:
    """
    Apply smoothing to any indicator.

    Args:
        window: Window size for smoothing
        method: Smoothing method ('sma', 'ema', 'ewm')
        alpha: Alpha parameter for exponential smoothing

    Returns:
        Function that smooths an indicator's output
    """

    def _smooth(indicator: Indicator) -> Indicator:
        def smoothed_indicator(data: TimeSeries) -> Signal:
            # Get original signal
            indicator(data)

            # Apply smoothing to the time series
            if method == "sma":
                smoothed_data = TimeSeries(
                    values=data.values.rolling(window=window, min_periods=1).mean(),
                    timestamps=data.timestamps,
                )
            elif method == "ema":
                smoothed_data = TimeSeries(
                    values=data.values.ewm(span=window, adjust=False).mean(),
                    timestamps=data.timestamps,
                )
            elif method == "ewm":
                smoothed_data = TimeSeries(
                    values=data.values.ewm(alpha=alpha, adjust=False).mean(),
                    timestamps=data.timestamps,
                )
            else:
                smoothed_data = data

            # Re-run indicator on smoothed data
            return indicator(smoothed_data)

        return smoothed_indicator

    return _smooth


def normalize(
    method: str = "minmax",
    lookback: int | None = None,
) -> IndicatorTransformer:
    """
    Normalize indicator values.

    Args:
        method: Normalization method ('minmax', 'zscore', 'percentile')
        lookback: Lookback period for normalization (None = use all data)

    Returns:
        Function that normalizes an indicator's output
    """

    def _normalize(indicator: Indicator) -> Indicator:
        def normalized_indicator(data: TimeSeries) -> Signal:
            signal = indicator(data)

            # Get values for normalization
            if lookback is None:
                norm_values = data.values
            else:
                norm_values = data.values.tail(lookback)

            # Apply normalization
            if method == "minmax":
                min_val = norm_values.min()
                max_val = norm_values.max()
                if max_val != min_val:
                    normalized_value = (signal.value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5
            elif method == "zscore":
                mean_val = norm_values.mean()
                std_val = norm_values.std()
                if std_val > 0:
                    normalized_value = (signal.value - mean_val) / std_val
                    # Clip to reasonable range
                    normalized_value = np.clip(normalized_value, -3, 3) / 3
                else:
                    normalized_value = 0.0
            elif method == "percentile":
                normalized_value = (norm_values <= signal.value).mean()
            else:
                normalized_value = signal.value

            # Ensure normalized value is in [0, 1] or [-1, 1] range
            if method in ["minmax", "percentile"]:
                normalized_value = np.clip(normalized_value, 0, 1)
            else:
                normalized_value = np.clip(normalized_value, -1, 1)

            return Signal(value=normalized_value, strength=signal.strength)

        return normalized_indicator

    return _normalize


def lag(periods: int) -> IndicatorTransformer:
    """
    Add time lag to indicators.

    Args:
        periods: Number of periods to lag

    Returns:
        Function that lags an indicator's output
    """

    def _lag(indicator: Indicator) -> Indicator:
        def lagged_indicator(data: TimeSeries) -> Signal:
            if len(data.values) <= periods:
                return Signal(value=0.0, strength=SignalStrength.NEUTRAL)

            # Create lagged data
            lagged_values = data.values.shift(periods)
            lagged_data = TimeSeries(
                values=lagged_values,
                timestamps=data.timestamps,
            )

            # Run indicator on lagged data
            return indicator(lagged_data)

        return lagged_indicator

    return _lag


def combine_indicators(
    indicators: list[Indicator],
    weights: list[float] | None = None,
    combination_fn: Callable[[list[Signal]], Signal] | None = None,
) -> Indicator:
    """
    Combine multiple indicators into a single indicator.

    Args:
        indicators: List of indicators to combine
        weights: Optional weights for each indicator
        combination_fn: Custom combination function

    Returns:
        Combined indicator
    """
    if weights is None:
        weights = [1.0] * len(indicators)
    else:
        assert len(weights) == len(indicators), "Weights must match indicators"

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    def combined_indicator(data: TimeSeries) -> Signal:
        signals = [ind(data) for ind in indicators]

        if combination_fn:
            return combination_fn(signals)
        # Default: weighted average
        weighted_sum = sum(w * s.value for w, s in zip(weights, signals, strict=False))

        # Average strength
        avg_strength = sum(
            w * s.strength.value for w, s in zip(weights, signals, strict=False)
        )

        # Map to discrete strength
        if avg_strength >= SignalStrength.STRONG.value:
            strength = SignalStrength.STRONG
        elif avg_strength >= SignalStrength.MEDIUM.value:
            strength = SignalStrength.MEDIUM
        elif avg_strength >= SignalStrength.WEAK.value:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NEUTRAL

        return Signal(value=weighted_sum, strength=strength)

    return combined_indicator


def threshold(
    upper: float = 0.7,
    lower: float = -0.7,
    neutral_zone: float = 0.1,
) -> IndicatorTransformer:
    """
    Apply thresholds to create discrete signals.

    Args:
        upper: Upper threshold for strong signals
        lower: Lower threshold for strong signals
        neutral_zone: Zone around zero for neutral signals

    Returns:
        Function that applies thresholds to an indicator
    """

    def _threshold(indicator: Indicator) -> Indicator:
        def thresholded_indicator(data: TimeSeries) -> Signal:
            signal = indicator(data)
            value = signal.value

            # Determine strength based on thresholds
            if value >= upper:
                return Signal(value=1.0, strength=SignalStrength.STRONG)
            if value <= lower:
                return Signal(value=-1.0, strength=SignalStrength.STRONG)
            if abs(value) <= neutral_zone:
                return Signal(value=0.0, strength=SignalStrength.NEUTRAL)
            if value > neutral_zone:
                strength = (
                    SignalStrength.MEDIUM
                    if value > (upper + neutral_zone) / 2
                    else SignalStrength.WEAK
                )
                return Signal(value=value / upper, strength=strength)
            strength = (
                SignalStrength.MEDIUM
                if value < (lower - neutral_zone) / 2
                else SignalStrength.WEAK
            )
            return Signal(value=value / abs(lower), strength=strength)

        return thresholded_indicator

    return _threshold


def momentum_transform(lookback: int = 10) -> IndicatorTransformer:
    """
    Transform indicator to measure rate of change.

    Args:
        lookback: Period for momentum calculation

    Returns:
        Function that adds momentum to an indicator
    """

    def _momentum(indicator: Indicator) -> Indicator:
        def momentum_indicator(data: TimeSeries) -> Signal:
            # Need enough data for momentum
            if len(data.values) < lookback + 1:
                return Signal(value=0.0, strength=SignalStrength.NEUTRAL)

            # Calculate momentum over time
            current_signal = indicator(data)

            # Get signal from lookback periods ago
            past_data = TimeSeries(
                values=data.values.iloc[:-lookback],
                timestamps=data.timestamps.iloc[:-lookback],
            )

            if len(past_data.values) > 0:
                past_signal = indicator(past_data)
                momentum = current_signal.value - past_signal.value

                # Determine strength based on momentum magnitude
                abs_momentum = abs(momentum)
                if abs_momentum > 0.5:
                    strength = SignalStrength.STRONG
                elif abs_momentum > 0.25:
                    strength = SignalStrength.MEDIUM
                elif abs_momentum > 0.1:
                    strength = SignalStrength.WEAK
                else:
                    strength = SignalStrength.NEUTRAL

                return Signal(value=momentum, strength=strength)
            return Signal(value=0.0, strength=SignalStrength.NEUTRAL)

        return momentum_indicator

    return _momentum


def divergence_detector(
    price_indicator: Indicator,
    momentum_indicator: Indicator,
    lookback: int = 20,
) -> Indicator:
    """
    Detect divergences between price and momentum indicators.

    Args:
        price_indicator: Indicator based on price
        momentum_indicator: Indicator based on momentum
        lookback: Period to check for divergences

    Returns:
        Indicator that signals divergences
    """

    def divergence_indicator(data: TimeSeries) -> Signal:
        if len(data.values) < lookback:
            return Signal(value=0.0, strength=SignalStrength.NEUTRAL)

        # Get current and past signals
        price_signal = price_indicator(data)
        momentum_signal = momentum_indicator(data)

        # Get signals from lookback period
        past_data = TimeSeries(
            values=data.values.iloc[:-lookback],
            timestamps=data.timestamps.iloc[:-lookback],
        )

        if len(past_data.values) > 0:
            past_price = price_indicator(past_data)
            past_momentum = momentum_indicator(past_data)

            # Calculate changes
            price_change = price_signal.value - past_price.value
            momentum_change = momentum_signal.value - past_momentum.value

            # Detect divergence
            if price_change > 0 and momentum_change < 0:
                # Bearish divergence
                divergence = -1.0
                strength = (
                    SignalStrength.STRONG
                    if abs(momentum_change) > 0.3
                    else SignalStrength.MEDIUM
                )
            elif price_change < 0 and momentum_change > 0:
                # Bullish divergence
                divergence = 1.0
                strength = (
                    SignalStrength.STRONG
                    if abs(momentum_change) > 0.3
                    else SignalStrength.MEDIUM
                )
            else:
                divergence = 0.0
                strength = SignalStrength.NEUTRAL

            return Signal(value=divergence, strength=strength)
        return Signal(value=0.0, strength=SignalStrength.NEUTRAL)

    return divergence_indicator


# Utility function for creating custom combinators
def create_combinator(
    transform_fn: Callable[[pd.Series], pd.Series],
    signal_fn: Callable[[float], Signal] | None = None,
) -> IndicatorTransformer:
    """
    Create a custom indicator transformer.

    Args:
        transform_fn: Function to transform the time series
        signal_fn: Optional function to convert value to signal

    Returns:
        Indicator transformer
    """

    def _transform(indicator: Indicator) -> Indicator:
        def transformed_indicator(data: TimeSeries) -> Signal:
            # Transform the data
            transformed_values = transform_fn(data.values)
            transformed_data = TimeSeries(
                values=transformed_values,
                timestamps=data.timestamps,
            )

            # Run indicator on transformed data
            signal = indicator(transformed_data)

            # Apply custom signal function if provided
            if signal_fn:
                return signal_fn(signal.value)
            return signal

        return transformed_indicator

    return _transform
