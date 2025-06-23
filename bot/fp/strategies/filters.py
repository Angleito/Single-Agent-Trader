"""Signal filtering functions for trading strategies.

This module provides various filters to validate and refine trading signals
based on time, market conditions, and other criteria.
"""

from collections.abc import Callable
from datetime import UTC, datetime, time

import numpy as np
import pandas as pd

from bot.fp.core.types import ValidationResult, failure, success

# Type aliases
Filter = Callable[[dict], ValidationResult[dict]]
TimeRange = tuple[time, time]


# Time-based filters
def trading_hours_filter(
    start_hour: int = 9,
    start_minute: int = 30,
    end_hour: int = 16,
    end_minute: int = 0,
    timezone_str: str = "US/Eastern",  # TODO: Implement timezone handling
) -> Filter:
    """Filter signals based on trading hours.

    Args:
        start_hour: Start hour (0-23)
        start_minute: Start minute (0-59)
        end_hour: End hour (0-23)
        end_minute: End minute (0-59)
        timezone_str: Timezone name

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        timestamp = data.get("timestamp")
        if not timestamp:
            return failure("No timestamp in data")

        # Convert to target timezone
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=UTC)
        else:
            dt = timestamp

        # Check if within trading hours
        current_time = dt.time()
        start_time = time(start_hour, start_minute)
        end_time = time(end_hour, end_minute)

        if start_time <= current_time <= end_time:
            return success(data)
        return failure(f"Outside trading hours: {current_time}")

    return filter_func


def market_session_filter(sessions: list[str]) -> Filter:
    """Filter signals based on market sessions.

    Args:
        sessions: List of allowed sessions (e.g., ["US", "EU", "ASIA"])

    Returns:
        Filter function
    """
    session_times = {
        "ASIA": (time(0, 0), time(9, 0)),  # 00:00 - 09:00 UTC
        "EU": (time(7, 0), time(16, 0)),  # 07:00 - 16:00 UTC
        "US": (time(13, 0), time(22, 0)),  # 13:00 - 22:00 UTC
    }

    def filter_func(data: dict) -> ValidationResult[dict]:
        timestamp = data.get("timestamp")
        if not timestamp:
            return failure("No timestamp in data")

        # Convert to UTC
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=UTC)
        else:
            dt = timestamp.astimezone(UTC)

        current_time = dt.time()

        # Check if in any allowed session
        for session in sessions:
            if session.upper() in session_times:
                start, end = session_times[session.upper()]
                if start <= current_time <= end:
                    data["market_session"] = session.upper()
                    return success(data)

        return failure(f"Not in allowed sessions: {sessions}")

    return filter_func


def news_blackout_filter(
    blackout_minutes_before: int = 30,
    blackout_minutes_after: int = 30,
    news_events: list[dict] | None = None,
) -> Filter:
    """Filter signals around news events.

    Args:
        blackout_minutes_before: Minutes before news to blackout
        blackout_minutes_after: Minutes after news to blackout
        news_events: List of news events with timestamps

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        if not news_events:
            return success(data)

        timestamp = data.get("timestamp")
        if not timestamp:
            return failure("No timestamp in data")

        # Convert to datetime
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=UTC)
        else:
            dt = timestamp

        # Check each news event
        for event in news_events:
            event_time = event.get("timestamp")
            if not event_time:
                continue

            if isinstance(event_time, (int, float)):
                event_dt = datetime.fromtimestamp(event_time, tz=UTC)
            else:
                event_dt = event_time

            # Calculate time difference
            time_diff = (dt - event_dt).total_seconds() / 60

            if -blackout_minutes_before <= time_diff <= blackout_minutes_after:
                return failure(
                    f"Within news blackout period: {event.get('title', 'Unknown event')}"
                )

        return success(data)

    return filter_func


# Market regime filters
def trend_regime_filter(
    required_trend: str, trend_period: int = 20, price_key: str = "close"
) -> Filter:
    """Filter based on market trend regime.

    Args:
        required_trend: Required trend ("up", "down", "sideways")
        trend_period: Period for trend calculation
        price_key: Price field to use

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        prices = data.get(price_key)
        if prices is None:
            return failure(f"No {price_key} prices in data")

        if isinstance(prices, (list, np.ndarray)):
            prices = np.array(prices)
        elif isinstance(prices, pd.Series):
            prices = prices.values
        else:
            return failure(f"Invalid price data type: {type(prices)}")

        if len(prices) < trend_period:
            return failure(
                f"Insufficient data for trend calculation: {len(prices)} < {trend_period}"
            )

        # Calculate trend using linear regression
        x = np.arange(trend_period)
        y = prices[-trend_period:]
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        price_range = np.max(y) - np.min(y)
        if price_range == 0:
            current_trend = "sideways"
        else:
            slope_pct = (slope * trend_period) / price_range

            if slope_pct > 0.1:
                current_trend = "up"
            elif slope_pct < -0.1:
                current_trend = "down"
            else:
                current_trend = "sideways"

        if current_trend == required_trend:
            data["market_trend"] = current_trend
            data["trend_strength"] = abs(slope_pct) if price_range > 0 else 0
            return success(data)
        return failure(f"Wrong trend: {current_trend} != {required_trend}")

    return filter_func


# Volatility filters
def volatility_filter(
    min_volatility: float | None = None,
    max_volatility: float | None = None,
    lookback_period: int = 20,
    price_key: str = "close",
) -> Filter:
    """Filter based on volatility levels.

    Args:
        min_volatility: Minimum required volatility
        max_volatility: Maximum allowed volatility
        lookback_period: Period for volatility calculation
        price_key: Price field to use

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        prices = data.get(price_key)
        if prices is None:
            return failure(f"No {price_key} prices in data")

        if isinstance(prices, (list, np.ndarray)):
            prices = np.array(prices)
        elif isinstance(prices, pd.Series):
            prices = prices.values
        else:
            return failure(f"Invalid price data type: {type(prices)}")

        if len(prices) < lookback_period + 1:
            return failure("Insufficient data for volatility calculation")

        # Calculate returns
        returns = (
            np.diff(prices[-lookback_period - 1 :]) / prices[-lookback_period - 1 : -1]
        )
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # Check volatility bounds
        if min_volatility is not None and volatility < min_volatility:
            return failure(
                f"Volatility too low: {volatility:.2%} < {min_volatility:.2%}"
            )

        if max_volatility is not None and volatility > max_volatility:
            return failure(
                f"Volatility too high: {volatility:.2%} > {max_volatility:.2%}"
            )

        data["volatility"] = volatility
        return success(data)

    return filter_func


def volatility_regime_filter(
    regime: str,
    lookback_period: int = 20,
    low_vol_threshold: float = 0.15,
    high_vol_threshold: float = 0.30,
) -> Filter:
    """Filter based on volatility regime.

    Args:
        regime: Required regime ("low", "medium", "high")
        lookback_period: Period for volatility calculation
        low_vol_threshold: Threshold for low volatility
        high_vol_threshold: Threshold for high volatility

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        # First calculate volatility
        vol_result = volatility_filter(lookback_period=lookback_period)(data)

        if isinstance(vol_result, tuple) and not vol_result[0]:
            return vol_result

        volatility = vol_result[1]["volatility"]

        # Determine regime
        if volatility < low_vol_threshold:
            current_regime = "low"
        elif volatility > high_vol_threshold:
            current_regime = "high"
        else:
            current_regime = "medium"

        if current_regime == regime:
            data["volatility_regime"] = current_regime
            return success(data)
        return failure(f"Wrong volatility regime: {current_regime} != {regime}")

    return filter_func


# Volume filters
def volume_filter(
    min_volume: float | None = None,
    min_volume_ratio: float | None = None,
    lookback_period: int = 20,
    volume_key: str = "volume",
) -> Filter:
    """Filter based on volume levels.

    Args:
        min_volume: Minimum absolute volume
        min_volume_ratio: Minimum volume as ratio of average
        lookback_period: Period for average calculation
        volume_key: Volume field to use

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        volumes = data.get(volume_key)
        if volumes is None:
            return failure(f"No {volume_key} data")

        if isinstance(volumes, (list, np.ndarray)):
            volumes = np.array(volumes)
        elif isinstance(volumes, pd.Series):
            volumes = volumes.values
        else:
            return failure(f"Invalid volume data type: {type(volumes)}")

        if len(volumes) == 0:
            return failure("No volume data available")

        current_volume = volumes[-1]

        # Check absolute volume
        if min_volume is not None and current_volume < min_volume:
            return failure(f"Volume too low: {current_volume} < {min_volume}")

        # Check relative volume
        if min_volume_ratio is not None:
            if len(volumes) < lookback_period:
                return failure("Insufficient data for volume ratio calculation")

            avg_volume = np.mean(volumes[-lookback_period:])
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio < min_volume_ratio:
                    return failure(
                        f"Volume ratio too low: {volume_ratio:.2f} < {min_volume_ratio:.2f}"
                    )
                data["volume_ratio"] = volume_ratio

        data["current_volume"] = current_volume
        return success(data)

    return filter_func


# Correlation filters
def correlation_filter(
    reference_symbol: str,
    min_correlation: float | None = None,
    max_correlation: float | None = None,
    lookback_period: int = 20,
    price_key: str = "close",
) -> Filter:
    """Filter based on correlation with reference symbol.

    Args:
        reference_symbol: Reference symbol for correlation
        min_correlation: Minimum correlation (-1 to 1)
        max_correlation: Maximum correlation (-1 to 1)
        lookback_period: Period for correlation calculation
        price_key: Price field to use

    Returns:
        Filter function
    """

    def filter_func(data: dict) -> ValidationResult[dict]:
        prices = data.get(price_key)
        ref_prices = data.get(f"{reference_symbol}_{price_key}")

        if prices is None:
            return failure(f"No {price_key} prices in data")
        if ref_prices is None:
            return failure(f"No {reference_symbol} prices in data")

        # Convert to arrays
        if isinstance(prices, (list, np.ndarray)):
            prices = np.array(prices)
        elif isinstance(prices, pd.Series):
            prices = prices.values

        if isinstance(ref_prices, (list, np.ndarray)):
            ref_prices = np.array(ref_prices)
        elif isinstance(ref_prices, pd.Series):
            ref_prices = ref_prices.values

        # Check data length
        min_len = min(len(prices), len(ref_prices))
        if min_len < lookback_period + 1:
            return failure("Insufficient data for correlation calculation")

        # Calculate returns
        returns = (
            np.diff(prices[-lookback_period - 1 :]) / prices[-lookback_period - 1 : -1]
        )
        ref_returns = (
            np.diff(ref_prices[-lookback_period - 1 :])
            / ref_prices[-lookback_period - 1 : -1]
        )

        # Calculate correlation
        correlation = np.corrcoef(returns, ref_returns)[0, 1]

        # Check correlation bounds
        if min_correlation is not None and correlation < min_correlation:
            return failure(
                f"Correlation too low: {correlation:.2f} < {min_correlation:.2f}"
            )

        if max_correlation is not None and correlation > max_correlation:
            return failure(
                f"Correlation too high: {correlation:.2f} > {max_correlation:.2f}"
            )

        data[f"correlation_{reference_symbol}"] = correlation
        return success(data)

    return filter_func


# Filter composition
def combine_filters(*filters: Filter, mode: str = "all") -> Filter:
    """Combine multiple filters.

    Args:
        *filters: Variable number of filters
        mode: Combination mode ("all" or "any")

    Returns:
        Combined filter function
    """

    def all_filter(data: dict) -> ValidationResult[dict]:
        current_data = data
        for f in filters:
            result = f(current_data)
            if isinstance(result, tuple) and not result[0]:
                return result
            current_data = result[1]
        return success(current_data)

    def any_filter(data: dict) -> ValidationResult[dict]:
        errors = []
        for f in filters:
            result = f(data)
            if isinstance(result, tuple) and result[0]:
                return result
            errors.append(result[1])
        return failure(f"All filters failed: {errors}")

    if mode == "all":
        return all_filter
    if mode == "any":
        return any_filter
    raise ValueError(f"Invalid mode: {mode}")


def sequential_filter(*filters: Filter) -> Filter:
    """Apply filters sequentially, passing data through each.

    Args:
        *filters: Variable number of filters

    Returns:
        Sequential filter function
    """
    return combine_filters(*filters, mode="all")


def parallel_filter(*filters: Filter) -> Filter:
    """Apply filters in parallel, any can pass.

    Args:
        *filters: Variable number of filters

    Returns:
        Parallel filter function
    """
    return combine_filters(*filters, mode="any")


# Convenience functions
def create_standard_filters(
    trading_hours: TimeRange | None = None,
    min_volume_ratio: float = 1.5,
    max_volatility: float = 0.5,
    required_trend: str | None = None,
) -> Filter:
    """Create a standard set of filters.

    Args:
        trading_hours: Trading hours range
        min_volume_ratio: Minimum volume ratio
        max_volatility: Maximum volatility
        required_trend: Required trend

    Returns:
        Combined filter
    """
    filters = []

    if trading_hours:
        start, end = trading_hours
        filters.append(
            trading_hours_filter(start.hour, start.minute, end.hour, end.minute)
        )

    filters.append(volume_filter(min_volume_ratio=min_volume_ratio))
    filters.append(volatility_filter(max_volatility=max_volatility))

    if required_trend:
        filters.append(trend_regime_filter(required_trend))

    return sequential_filter(*filters)


def create_risk_filters(
    max_volatility: float = 0.4, min_volume: float = 1000000, news_blackout: bool = True
) -> Filter:
    """Create risk-based filters.

    Args:
        max_volatility: Maximum allowed volatility
        min_volume: Minimum volume
        news_blackout: Enable news blackout

    Returns:
        Combined filter
    """
    filters = [
        volatility_filter(max_volatility=max_volatility),
        volume_filter(min_volume=min_volume),
    ]

    if news_blackout:
        filters.append(news_blackout_filter())

    return sequential_filter(*filters)
