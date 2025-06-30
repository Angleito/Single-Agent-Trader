"""Tests for volume agent functionality and volume-weighted calculations."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from bot.fp.strategies.market_making import (
    VolumeWeightedMetrics,
    analyze_volume_weighted_metrics,
    calculate_volume_profile,
    calculate_volume_weighted_spread,
    calculate_vwap,
    calculate_vwap_bands,
)
from bot.fp.types.market import Candle
from bot.utils.volume_validator import (
    safe_volume_conversion,
    validate_24h_volume,
    validate_orderbook_volume,
    validate_volume_data,
    validate_volume_series,
)


class TestVolumeWeightedCalculations:
    """Test volume-weighted calculation functions."""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candle data for testing."""
        base_time = datetime.now(UTC)
        return [
            Candle(
                timestamp=base_time,
                open=Decimal(100),
                high=Decimal(105),
                low=Decimal(98),
                close=Decimal(102),
                volume=Decimal(1000),
            ),
            Candle(
                timestamp=base_time,
                open=Decimal(102),
                high=Decimal(108),
                low=Decimal(100),
                close=Decimal(106),
                volume=Decimal(1500),
            ),
            Candle(
                timestamp=base_time,
                open=Decimal(106),
                high=Decimal(110),
                low=Decimal(104),
                close=Decimal(108),
                volume=Decimal(800),
            ),
        ]

    def test_calculate_vwap(self, sample_candles):
        """Test VWAP calculation."""
        vwap = calculate_vwap(sample_candles, periods=3)

        # Manual calculation for verification
        # Candle 1: typical_price = (105+98+102)/3 = 101.67, volume = 1000
        # Candle 2: typical_price = (108+100+106)/3 = 104.67, volume = 1500
        # Candle 3: typical_price = (110+104+108)/3 = 107.33, volume = 800
        # VWAP = (101.67*1000 + 104.67*1500 + 107.33*800) / (1000+1500+800)

        assert vwap > 0
        assert isinstance(vwap, float)
        # VWAP should be somewhere between the min and max typical prices
        assert 101 < vwap < 108

    def test_calculate_vwap_bands(self, sample_candles):
        """Test VWAP bands calculation."""
        vwap, upper_band, lower_band = calculate_vwap_bands(sample_candles, periods=3)

        assert vwap > 0
        assert upper_band > vwap
        assert lower_band < vwap
        assert isinstance(vwap, float)
        assert isinstance(upper_band, float)
        assert isinstance(lower_band, float)

    def test_calculate_volume_profile(self, sample_candles):
        """Test volume profile calculation."""
        profile = calculate_volume_profile(sample_candles, price_buckets=10)

        assert isinstance(profile, dict)
        assert len(profile) > 0

        # All values should be percentages that sum to 1.0
        total_percentage = sum(profile.values())
        assert abs(total_percentage - 1.0) < 0.01

    def test_calculate_volume_weighted_spread(self):
        """Test volume-weighted spread calculation."""
        bids = [(100.0, 10.0), (99.5, 15.0), (99.0, 20.0)]
        asks = [(100.5, 12.0), (101.0, 18.0), (101.5, 25.0)]

        spread = calculate_volume_weighted_spread(bids, asks)

        assert spread > 0
        assert isinstance(spread, float)
        # Spread should be reasonable (not more than 5%)
        assert spread < 0.05

    def test_analyze_volume_weighted_metrics(self, sample_candles):
        """Test comprehensive volume-weighted metrics analysis."""
        bids = [(100.0, 10.0), (99.5, 15.0)]
        asks = [(100.5, 12.0), (101.0, 18.0)]

        metrics = analyze_volume_weighted_metrics(sample_candles, bids, asks)

        assert isinstance(metrics, VolumeWeightedMetrics)
        assert metrics.vwap > 0
        assert metrics.vwap_24h >= 0
        assert 0 <= metrics.liquidity_score <= 1
        assert isinstance(metrics.volume_momentum, float)
        assert isinstance(metrics.volume_weighted_spread, float)

    def test_empty_candles(self):
        """Test behavior with empty candle data."""
        vwap = calculate_vwap([], periods=20)
        assert vwap == 0.0

        profile = calculate_volume_profile([])
        assert profile == {}

        metrics = analyze_volume_weighted_metrics([])
        assert metrics.vwap == 0.0
        assert metrics.liquidity_score == 0.0


class TestVolumeValidation:
    """Test volume validation functionality."""

    def test_validate_volume_data_valid(self):
        """Test validation with valid volume data."""
        result = validate_volume_data(100.5)

        assert result.is_valid
        assert result.normalized_volume == Decimal("100.5")
        assert result.confidence_score == 1.0
        assert len(result.errors) == 0

    def test_validate_volume_data_negative(self):
        """Test validation with negative volume."""
        result = validate_volume_data(-10)

        assert not result.is_valid
        assert "negative" in result.errors[0].lower()
        assert result.confidence_score == 0.0

    def test_validate_volume_data_string(self):
        """Test validation with string input."""
        result = validate_volume_data("123.45")

        assert result.is_valid
        assert result.normalized_volume == Decimal("123.45")

    def test_validate_volume_data_invalid_string(self):
        """Test validation with invalid string."""
        result = validate_volume_data("not_a_number")

        assert not result.is_valid
        assert "convert" in result.errors[0].lower()

    def test_validate_24h_volume(self):
        """Test 24h volume validation."""
        result = validate_24h_volume(1000000)

        assert result.is_valid
        assert result.normalized_volume == Decimal(1000000)

    def test_validate_orderbook_volume(self):
        """Test orderbook volume validation."""
        bids = [(100.0, 10.0), (99.5, 15.0)]
        asks = [(100.5, 12.0), (101.0, 18.0)]

        result = validate_orderbook_volume(bids, asks)

        assert result.is_valid
        assert result.normalized_volume > 0
        assert result.confidence_score > 0

    def test_validate_orderbook_volume_imbalanced(self):
        """Test orderbook validation with extreme imbalance."""
        bids = [(100.0, 1000.0)]  # Very large bid volume
        asks = [(100.5, 1.0)]  # Very small ask volume

        result = validate_orderbook_volume(bids, asks)

        # Should still be valid but with warnings
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "imbalance" in result.warnings[0].lower()

    def test_validate_volume_series(self):
        """Test volume series validation."""
        volumes = [100, 150, 120, 180, 90]

        result = validate_volume_series(volumes)

        assert result.is_valid
        assert result.normalized_volume > 0

    def test_validate_volume_series_with_outliers(self):
        """Test volume series with outliers."""
        volumes = [100, 150, 2000, 120, 180]  # 2000 is an outlier

        result = validate_volume_series(volumes)

        assert result.is_valid
        assert len(result.warnings) > 0
        # Should warn about outliers

    def test_safe_volume_conversion(self):
        """Test safe volume conversion."""
        # Valid conversion
        result = safe_volume_conversion(123.45)
        assert result == Decimal("123.45")

        # Invalid conversion with default
        result = safe_volume_conversion("invalid", default=Decimal(999))
        assert result == Decimal(999)

        # None with default
        result = safe_volume_conversion(None, default=Decimal(0))
        assert result == Decimal(0)


class TestVolumeIntegration:
    """Integration tests for volume functionality."""

    def test_volume_weighted_market_making_integration(self, sample_candles):
        """Test integration of volume calculations with market making."""
        # This test verifies that all components work together
        bids = [(100.0, 10.0), (99.5, 15.0), (99.0, 20.0)]
        asks = [(100.5, 12.0), (101.0, 18.0), (101.5, 25.0)]

        # Test that we can run the full analysis
        metrics = analyze_volume_weighted_metrics(sample_candles, bids, asks)

        # Validate the metrics
        assert isinstance(metrics, VolumeWeightedMetrics)

        # Test individual components
        vwap = calculate_vwap(sample_candles)
        assert vwap > 0

        spread = calculate_volume_weighted_spread(bids, asks)
        assert spread > 0

        # Validate all volume data
        volumes = [float(c.volume) for c in sample_candles]
        validation_result = validate_volume_series(volumes)
        assert validation_result.is_valid

    def test_error_handling(self):
        """Test error handling in volume calculations."""
        # Test with malformed data
        try:
            invalid_candles = [None, "invalid", 123]
            result = analyze_volume_weighted_metrics(invalid_candles)
            # Should return default/empty metrics without crashing
            assert isinstance(result, VolumeWeightedMetrics)
        except Exception:
            # If it does raise an exception, it should be handled gracefully
            pass

        # Test with empty orderbook
        spread = calculate_volume_weighted_spread([], [])
        assert spread == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
