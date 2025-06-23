"""Property-based tests for market data consistency invariants."""

from datetime import timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from bot.indicators.vumanchu import VuManChuIndicators


# Custom strategies for market data generation
@composite
def valid_candle(draw: Any) -> dict[str, float]:
    """Generate a valid OHLC candle with invariants preserved."""
    # Generate base prices
    open_price = draw(st.floats(min_value=0.01, max_value=100000))
    close_price = draw(st.floats(min_value=0.01, max_value=100000))

    # Ensure high is >= max(open, close) and low is <= min(open, close)
    price_range = abs(close_price - open_price)
    high_offset = draw(st.floats(min_value=0, max_value=price_range * 2))
    low_offset = draw(st.floats(min_value=0, max_value=price_range * 2))

    high_price = max(open_price, close_price) + high_offset
    low_price = min(open_price, close_price) - low_offset

    # Ensure low_price stays positive
    low_price = max(0.01, low_price)

    # Generate timestamp and volume
    timestamp = draw(st.integers(min_value=1600000000, max_value=2000000000))
    volume = draw(st.floats(min_value=0, max_value=1000000))

    return {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "timestamp": timestamp,
    }


@composite
def candle_series(
    draw: Any, min_size: int = 1, max_size: int = 1000
) -> list[dict[str, float]]:
    """Generate a series of candles with sequential timestamps."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    base_timestamp = draw(st.integers(min_value=1600000000, max_value=1900000000))

    candles = []
    for i in range(size):
        candle = draw(valid_candle())
        # Ensure sequential timestamps
        candle["timestamp"] = base_timestamp + i * 60  # 1 minute intervals
        candles.append(candle)

    return candles


class TestMarketDataInvariants:
    """Test OHLC relationship invariants and data consistency."""

    @given(candle=valid_candle())
    def test_ohlc_relationships(self, candle: dict[str, float]) -> None:
        """Test that OHLC relationships are always valid."""
        # High must be the highest price
        assert candle["high"] >= candle["open"]
        assert candle["high"] >= candle["low"]
        assert candle["high"] >= candle["close"]

        # Low must be the lowest price
        assert candle["low"] <= candle["open"]
        assert candle["low"] <= candle["high"]
        assert candle["low"] <= candle["close"]

        # Volume must be non-negative
        assert candle["volume"] >= 0

        # All prices must be positive
        assert candle["open"] > 0
        assert candle["high"] > 0
        assert candle["low"] > 0
        assert candle["close"] > 0

    @given(candles=st.lists(valid_candle(), min_size=2, max_size=100))
    def test_candle_list_consistency(self, candles: list[dict[str, float]]) -> None:
        """Test consistency across multiple candles."""
        df = pd.DataFrame(candles)

        # No NaN values should exist
        assert not df.isna().any().any()

        # All OHLC relationships should hold for every candle
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["high"] >= df["low"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["low"] <= df["high"]).all()

    @given(
        base_price=st.floats(min_value=1, max_value=100000),
        volatility=st.floats(min_value=0.001, max_value=0.5),
        num_candles=st.integers(min_value=10, max_value=100),
    )
    def test_realistic_price_movement(
        self, base_price: float, volatility: float, num_candles: int
    ) -> None:
        """Test realistic price movements with controlled volatility."""
        candles = []
        current_price = base_price

        for i in range(num_candles):
            # Generate price movement within volatility bounds
            max_move = current_price * volatility
            open_price = current_price
            close_price = current_price + np.random.uniform(-max_move, max_move)
            close_price = max(0.01, close_price)  # Ensure positive

            high_price = max(open_price, close_price) * (
                1 + np.random.uniform(0, volatility / 2)
            )
            low_price = min(open_price, close_price) * (
                1 - np.random.uniform(0, volatility / 2)
            )
            low_price = max(0.01, low_price)  # Ensure positive

            candles.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": np.random.uniform(100, 10000),
                    "timestamp": 1600000000 + i * 60,
                }
            )

            current_price = close_price

        df = pd.DataFrame(candles)

        # Verify all invariants hold
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # Check price continuity
        price_changes = df["close"].pct_change().dropna()
        # Most price changes should be within 2x volatility
        extreme_moves = (price_changes.abs() > 2 * volatility).sum()
        assert extreme_moves < len(price_changes) * 0.05  # Less than 5% extreme moves


class TestDataAggregation:
    """Test data aggregation preserves key properties."""

    @given(candles=candle_series(min_size=10, max_size=100))
    def test_volume_aggregation_preserves_total(
        self, candles: list[dict[str, float]]
    ) -> None:
        """Test that aggregating candles preserves total volume."""
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        # Original total volume
        original_volume = df["volume"].sum()

        # Aggregate to 5-minute candles
        agg_df = (
            df.resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # Aggregated volume should equal original
        if len(agg_df) > 0:
            aggregated_volume = agg_df["volume"].sum()
            assert (
                abs(original_volume - aggregated_volume) < 0.01
            )  # Allow small floating point error

    @given(candles=candle_series(min_size=20, max_size=200))
    def test_aggregation_preserves_price_bounds(
        self, candles: list[dict[str, float]]
    ) -> None:
        """Test that aggregation preserves price boundaries."""
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        # Track global high/low
        global_high = df["high"].max()
        global_low = df["low"].min()

        # Aggregate to 10-minute candles
        agg_df = (
            df.resample("10min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        if len(agg_df) > 0:
            # Aggregated high/low should match global
            assert agg_df["high"].max() <= global_high + 0.01
            assert agg_df["low"].min() >= global_low - 0.01

            # OHLC relationships should still hold
            assert (agg_df["high"] >= agg_df["open"]).all()
            assert (agg_df["high"] >= agg_df["close"]).all()
            assert (agg_df["low"] <= agg_df["open"]).all()
            assert (agg_df["low"] <= agg_df["close"]).all()


class TestMinimumDataRequirements:
    """Test minimum data requirements for indicators."""

    @given(
        num_candles=st.integers(min_value=1, max_value=50),
        base_price=st.floats(
            min_value=10, max_value=10000, allow_nan=False, allow_infinity=False
        ),
    )
    def test_indicator_minimum_candles(
        self, num_candles: int, base_price: float
    ) -> None:
        """Test that indicators handle insufficient data gracefully."""
        # Generate simple candle data
        candles = []
        for i in range(num_candles):
            price = base_price * (1 + np.random.uniform(-0.02, 0.02))
            candles.append(
                {
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price * (1 + np.random.uniform(-0.01, 0.01)),
                    "volume": 1000,
                }
            )

        df = pd.DataFrame(candles)
        df.index = pd.date_range("2024-01-01", periods=num_candles, freq="1h")

        # Initialize indicators
        indicators = VuManChuIndicators()

        # Should not raise exceptions even with minimal data
        try:
            result = indicators.calculate_all(df)
            assert isinstance(result, dict)

            # With insufficient data, signals should be None or empty
            if num_candles < 20:  # Typical minimum for most indicators
                signal = result.get("signal")
                if signal is not None:
                    assert signal == "HOLD" or signal == ""
        except Exception as e:
            # Should handle gracefully, not crash
            assert (
                "insufficient data" in str(e).lower()
                or "not enough data" in str(e).lower()
            )

    @given(candles=candle_series(min_size=100, max_size=500))
    @settings(deadline=timedelta(seconds=10))
    def test_indicator_output_consistency(
        self, candles: list[dict[str, float]]
    ) -> None:
        """Test that indicator outputs are consistent and valid."""
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        indicators = VuManChuIndicators()
        result = indicators.calculate_all(df)

        # Check that result is properly structured
        assert isinstance(result, dict)

        # If we have indicators in the result
        if "indicators" in result:
            ind_df = result["indicators"]

            # No infinite values
            numeric_columns = ind_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                assert not np.isinf(ind_df[col]).any()

            # Signal should be valid if present
            if result.get("signal"):
                assert result["signal"] in ["LONG", "SHORT", "HOLD"]


class TestDataQuality:
    """Test data quality and edge cases."""

    @given(
        prices=st.lists(
            st.floats(
                min_value=0.01,
                max_value=100000,
                allow_nan=False,
                allow_infinity=False,
                width=64,  # Use float64 for better precision
            ),
            min_size=10,
            max_size=100,
        )
    )
    def test_decimal_precision_handling(self, prices: list[float]) -> None:
        """Test handling of decimal precision in price calculations."""
        # Convert to Decimal for precise calculations
        decimal_prices = [Decimal(str(p)) for p in prices]

        # Calculate average using Decimal
        decimal_avg = sum(decimal_prices) / len(decimal_prices)

        # Calculate average using float
        float_avg = sum(prices) / len(prices)

        # The difference should be minimal
        difference = abs(float(decimal_avg) - float_avg)
        relative_error = difference / float_avg if float_avg != 0 else 0

        # For financial data, we want < 0.01% error
        assert relative_error < 0.0001

    @given(
        candles=st.lists(valid_candle(), min_size=2, max_size=1000),
        gap_probability=st.floats(min_value=0, max_value=0.3),
    )
    @settings(
        deadline=timedelta(seconds=10), suppress_health_check=[HealthCheck.too_slow]
    )
    def test_price_gap_detection(
        self, candles: list[dict[str, float]], gap_probability: float
    ) -> None:
        """Test detection of price gaps between candles."""
        if len(candles) < 2:
            return

        # Sort by timestamp
        sorted_candles = sorted(candles, key=lambda x: x["timestamp"])

        gaps = []
        for i in range(1, len(sorted_candles)):
            prev_close = sorted_candles[i - 1]["close"]
            curr_open = sorted_candles[i]["open"]

            gap_pct = abs(curr_open - prev_close) / prev_close if prev_close != 0 else 0

            # Randomly introduce gaps
            if np.random.random() < gap_probability:
                # Create artificial gap
                gap_size = np.random.uniform(0.01, 0.10)  # 1-10% gap
                if np.random.random() < 0.5:
                    sorted_candles[i]["open"] = prev_close * (1 + gap_size)
                else:
                    sorted_candles[i]["open"] = prev_close * (1 - gap_size)

                # Adjust high/low to maintain invariants
                sorted_candles[i]["high"] = max(
                    sorted_candles[i]["high"], sorted_candles[i]["open"]
                )
                sorted_candles[i]["low"] = min(
                    sorted_candles[i]["low"], sorted_candles[i]["open"]
                )

                gaps.append(i)

        # Verify gaps can be detected
        detected_gaps = []
        for i in range(1, len(sorted_candles)):
            prev_close = sorted_candles[i - 1]["close"]
            curr_open = sorted_candles[i]["open"]
            gap_pct = abs(curr_open - prev_close) / prev_close if prev_close != 0 else 0

            if gap_pct > 0.005:  # 0.5% threshold
                detected_gaps.append(i)

        # Should detect most artificial gaps
        if len(gaps) > 0:
            detection_rate = len(set(detected_gaps) & set(gaps)) / len(gaps)
            assert detection_rate > 0.8  # Detect at least 80% of gaps

    @given(
        candles=candle_series(min_size=100, max_size=1000),
        timestamp_error_rate=st.floats(min_value=0, max_value=0.1),
    )
    def test_timestamp_ordering_and_duplicates(
        self, candles: list[dict[str, float]], timestamp_error_rate: float
    ) -> None:
        """Test handling of timestamp issues."""
        # Introduce some timestamp errors
        for i, candle in enumerate(candles):
            if np.random.random() < timestamp_error_rate:
                # Create duplicate or out-of-order timestamp
                if i > 0 and np.random.random() < 0.5:
                    # Duplicate
                    candle["timestamp"] = candles[i - 1]["timestamp"]
                elif i > 1:
                    # Out of order
                    candle["timestamp"] = candles[i - 2]["timestamp"]

        df = pd.DataFrame(candles)

        # Check for duplicates
        duplicates = df["timestamp"].duplicated()
        duplicate_count = duplicates.sum()

        # Clean data: remove duplicates and sort
        df_clean = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Verify cleaning worked
        assert not df_clean["timestamp"].duplicated().any()
        assert df_clean["timestamp"].is_monotonic_increasing

        # Data loss should be proportional to error rate
        if len(df) > 0:
            data_loss_rate = 1 - (len(df_clean) / len(df))
            assert data_loss_rate <= timestamp_error_rate * 2  # Some tolerance

    @given(
        base_price=st.floats(min_value=1, max_value=100000),
        extreme_move_probability=st.floats(min_value=0, max_value=0.1),
        num_candles=st.integers(min_value=10, max_value=100),
    )
    def test_extreme_price_movements(
        self, base_price: float, extreme_move_probability: float, num_candles: int
    ) -> None:
        """Test handling of extreme price movements (circuit breaker scenarios)."""
        candles = []
        current_price = base_price
        circuit_breaker_triggered = []

        for i in range(num_candles):
            if np.random.random() < extreme_move_probability:
                # Extreme move: 10-50% in one candle
                move_size = np.random.uniform(0.10, 0.50)
                if np.random.random() < 0.5:
                    new_price = current_price * (1 + move_size)
                else:
                    new_price = current_price * (1 - move_size)

                circuit_breaker_triggered.append(i)
            else:
                # Normal move: 0-2%
                move_size = np.random.uniform(-0.02, 0.02)
                new_price = current_price * (1 + move_size)

            new_price = max(0.01, new_price)  # Ensure positive

            # Create candle
            open_price = current_price
            close_price = new_price
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            low_price = max(0.01, low_price)

            candles.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": np.random.uniform(100, 10000),
                    "timestamp": 1600000000 + i * 60,
                }
            )

            current_price = new_price

        df = pd.DataFrame(candles)

        # Calculate returns
        returns = df["close"].pct_change().dropna()

        # Detect extreme moves
        extreme_threshold = 0.10  # 10% move
        detected_extremes = (returns.abs() > extreme_threshold).sum()

        # Should detect most circuit breaker events
        if len(circuit_breaker_triggered) > 0:
            # Some extreme moves might be dampened by high/low, so allow some tolerance
            assert detected_extremes >= len(circuit_breaker_triggered) * 0.7

        # All OHLC invariants should still hold
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()


class TestTimeZoneConsiderations:
    """Test timezone handling in market data."""

    @given(
        candles=candle_series(
            min_size=24, max_size=168
        ),  # 1 day to 1 week of hourly data
        timezone_offset=st.integers(min_value=-12, max_value=12),
    )
    def test_timezone_conversions(
        self, candles: list[dict[str, float]], timezone_offset: int
    ) -> None:
        """Test that timezone conversions preserve data integrity."""
        df = pd.DataFrame(candles)

        # Create UTC timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)

        # Original data stats
        original_count = len(df)
        original_first = df.index[0]
        original_last = df.index[-1]

        # Convert to different timezone
        df_tz = df.tz_convert(
            f"Etc/GMT{-timezone_offset:+d}"
        )  # Note: Etc/GMT has reversed signs

        # Data count should remain the same
        assert len(df_tz) == original_count

        # Time differences should be preserved
        time_diff_original = (original_last - original_first).total_seconds()
        time_diff_converted = (df_tz.index[-1] - df_tz.index[0]).total_seconds()
        assert (
            abs(time_diff_original - time_diff_converted) < 1
        )  # Less than 1 second difference

        # OHLCV data should be unchanged
        assert df["open"].equals(df_tz["open"])
        assert df["high"].equals(df_tz["high"])
        assert df["low"].equals(df_tz["low"])
        assert df["close"].equals(df_tz["close"])
        assert df["volume"].equals(df_tz["volume"])

    @given(
        candles=candle_series(min_size=48, max_size=336),  # 2 days to 2 weeks
        market_timezone=st.sampled_from(
            ["US/Eastern", "Europe/London", "Asia/Tokyo", "Australia/Sydney"]
        ),
    )
    def test_market_hours_filtering(
        self, candles: list[dict[str, float]], market_timezone: str
    ) -> None:
        """Test filtering data for market hours across timezones."""
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)

        # Convert to market timezone
        df_market = df.tz_convert(market_timezone)

        # Define market hours (simplified: 9:30 AM - 4:00 PM)
        market_open = 9.5  # 9:30 AM
        market_close = 16  # 4:00 PM

        # Filter for market hours
        df_market_hours = df_market[
            (df_market.index.hour + df_market.index.minute / 60 >= market_open)
            & (df_market.index.hour + df_market.index.minute / 60 < market_close)
        ]

        if len(df_market_hours) > 0:
            # All filtered data should be within market hours
            hours = df_market_hours.index.hour + df_market_hours.index.minute / 60
            assert (hours >= market_open).all()
            assert (hours < market_close).all()

            # OHLC relationships should still hold
            assert (df_market_hours["high"] >= df_market_hours["open"]).all()
            assert (df_market_hours["high"] >= df_market_hours["close"]).all()
            assert (df_market_hours["low"] <= df_market_hours["open"]).all()
            assert (df_market_hours["low"] <= df_market_hours["close"]).all()


# Additional edge case tests
class TestEdgeCases:
    """Test various edge cases in market data."""

    @given(
        price=st.floats(min_value=0.01, max_value=100000),
        num_candles=st.integers(min_value=10, max_value=100),
    )
    def test_flat_market_conditions(self, price: float, num_candles: int) -> None:
        """Test handling of flat market (no price movement)."""
        # Create flat market data
        candles = []
        for i in range(num_candles):
            candles.append(
                {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1000,
                    "timestamp": 1600000000 + i * 60,
                }
            )

        df = pd.DataFrame(candles)

        # Verify all prices are equal
        assert (df["open"] == price).all()
        assert (df["high"] == price).all()
        assert (df["low"] == price).all()
        assert (df["close"] == price).all()

        # OHLC relationships should still hold (all equal)
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # Returns should all be zero
        returns = df["close"].pct_change().dropna()
        assert (returns == 0).all()

    @given(
        base_price=st.floats(min_value=0.01, max_value=1),
        multiplier=st.integers(min_value=1000, max_value=1000000),
    )
    def test_extreme_price_ranges(self, base_price: float, multiplier: int) -> None:
        """Test handling of extreme price ranges (penny stocks to high-value assets)."""
        # Test very low prices (penny stocks)
        low_price_candle = {
            "open": base_price,
            "high": base_price * 1.1,
            "low": base_price * 0.9,
            "close": base_price * 1.05,
            "volume": 1000000,
            "timestamp": 1600000000,
        }

        # Test very high prices
        high_price_candle = {
            "open": base_price * multiplier,
            "high": base_price * multiplier * 1.1,
            "low": base_price * multiplier * 0.9,
            "close": base_price * multiplier * 1.05,
            "volume": 10,
            "timestamp": 1600000060,
        }

        # Both should maintain OHLC relationships
        for candle in [low_price_candle, high_price_candle]:
            assert candle["high"] >= candle["open"]
            assert candle["high"] >= candle["close"]
            assert candle["high"] >= candle["low"]
            assert candle["low"] <= candle["open"]
            assert candle["low"] <= candle["close"]

        # Percentage calculations should work correctly
        low_return = (
            low_price_candle["close"] - low_price_candle["open"]
        ) / low_price_candle["open"]
        high_return = (
            high_price_candle["close"] - high_price_candle["open"]
        ) / high_price_candle["open"]

        # Returns should be similar despite different absolute prices
        assert abs(low_return - high_return) < 0.001

    @given(candles=candle_series(min_size=100, max_size=1000))
    @settings(deadline=timedelta(seconds=10))
    def test_missing_data_handling(self, candles: list[dict[str, float]]) -> None:
        """Test handling of missing data points."""
        df = pd.DataFrame(candles)

        # Randomly remove some data points
        missing_indices = np.random.choice(
            df.index, size=int(len(df) * 0.1), replace=False
        )
        df_with_gaps = df.drop(missing_indices)

        # Convert to time series
        df_with_gaps["timestamp"] = pd.to_datetime(df_with_gaps["timestamp"], unit="s")
        df_with_gaps.set_index("timestamp", inplace=True)

        # Resample to fill gaps
        df_filled = df_with_gaps.resample("1min").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Forward fill missing values
        df_filled = df_filled.ffill()

        # After filling, all OHLC relationships should hold
        df_filled_valid = df_filled.dropna()
        if len(df_filled_valid) > 0:
            assert (df_filled_valid["high"] >= df_filled_valid["open"]).all()
            assert (df_filled_valid["high"] >= df_filled_valid["close"]).all()
            assert (df_filled_valid["low"] <= df_filled_valid["open"]).all()
            assert (df_filled_valid["low"] <= df_filled_valid["close"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
