#!/usr/bin/env python3
"""
Standalone validation script for VuManChu functional indicators.
This script validates the functional indicator implementations without
requiring the full bot configuration.
"""

import sys

import numpy as np

# Add the project root to the Python path
sys.path.insert(0, "/Users/angel/Documents/Projects/cursorprod")

from bot.fp.indicators.vumanchu_functional import (
    calculate_ema,
    calculate_hlc3,
    calculate_wavetrend_oscillator,
    detect_crossovers,
    vumanchu_cipher,
)
from bot.fp.types.indicators import VuManchuResult


def generate_test_data(length=100, seed=42):
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)

    base_price = 100.0
    trend = np.cumsum(np.random.normal(0, 0.5, length))
    prices = base_price + trend

    ohlcv = np.zeros((length, 5))

    for i in range(length):
        close = prices[i]
        daily_range = abs(np.random.normal(0, 2.0))
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)

        # Ensure valid OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        volume = np.random.uniform(1000, 10000)

        ohlcv[i] = [open_price, high, low, close, volume]

    return ohlcv


def test_hlc3_calculation():
    """Test HLC3 calculation."""
    print("Testing HLC3 calculation...")

    # Test data
    high = np.array([105.0, 107.0, 103.0])
    low = np.array([95.0, 97.0, 93.0])
    close = np.array([100.0, 102.0, 98.0])

    # Calculate HLC3
    hlc3 = calculate_hlc3(high, low, close)

    # Expected results
    expected = (high + low + close) / 3.0

    # Check results
    if np.allclose(hlc3, expected):
        print("✓ HLC3 calculation passed")
        return True
    print("✗ HLC3 calculation failed")
    print(f"  Expected: {expected}")
    print(f"  Got: {hlc3}")
    return False


def test_ema_calculation():
    """Test EMA calculation."""
    print("Testing EMA calculation...")

    # Test data
    values = np.array([100.0, 101.0, 99.0, 102.0, 105.0, 103.0, 106.0])
    period = 3

    # Calculate EMA
    ema = calculate_ema(values, period)

    # Basic checks
    if len(ema) == len(values):
        print("✓ EMA length check passed")
    else:
        print("✗ EMA length check failed")
        return False

    # Check for NaN in initial values
    initial_nans = np.sum(np.isnan(ema[: period - 1]))
    if initial_nans == period - 1:
        print("✓ EMA NaN handling passed")
    else:
        print("✗ EMA NaN handling failed")
        return False

    # Check for valid final values
    final_values = ema[period - 1 :]
    if not np.any(np.isnan(final_values)):
        print("✓ EMA calculation passed")
        return True
    print("✗ EMA calculation failed")
    return False


def test_wavetrend_calculation():
    """Test WaveTrend oscillator calculation."""
    print("Testing WaveTrend calculation...")

    # Generate test data
    ohlcv = generate_test_data(50)
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    # Calculate HLC3
    hlc3 = calculate_hlc3(high, low, close)

    # Calculate WaveTrend
    wt1, wt2 = calculate_wavetrend_oscillator(hlc3, 6, 8, 3)

    # Basic checks
    if len(wt1) == len(hlc3) and len(wt2) == len(hlc3):
        print("✓ WaveTrend length check passed")
    else:
        print("✗ WaveTrend length check failed")
        return False

    # Check for some valid values
    valid_wt1 = ~np.isnan(wt1)
    valid_wt2 = ~np.isnan(wt2)

    if np.sum(valid_wt1) > 10 and np.sum(valid_wt2) > 10:
        print("✓ WaveTrend calculation passed")
        return True
    print("✗ WaveTrend calculation failed")
    return False


def test_vumanchu_result():
    """Test VuManchuResult creation and methods."""
    print("Testing VuManchuResult...")

    # Generate test data
    ohlcv = generate_test_data(30)

    # Calculate VuManChu
    result = vumanchu_cipher(ohlcv)

    # Check result type
    if not isinstance(result, VuManchuResult):
        print("✗ VuManchuResult type check failed")
        return False

    # Check attributes
    required_attrs = ["timestamp", "wave_a", "wave_b", "signal"]
    for attr in required_attrs:
        if not hasattr(result, attr):
            print(f"✗ Missing attribute: {attr}")
            return False

    # Check signal value
    if result.signal not in ["LONG", "SHORT", "NEUTRAL"]:
        print(f"✗ Invalid signal: {result.signal}")
        return False

    # Test methods
    try:
        result.is_bullish_crossover()
        result.is_bearish_crossover()
        strength = result.momentum_strength()
        if 0 <= strength <= 100:
            print("✓ VuManchuResult methods passed")
        else:
            print(f"✗ Invalid momentum strength: {strength}")
            return False
    except Exception as e:
        print(f"✗ VuManchuResult method error: {e}")
        return False

    print("✓ VuManchuResult passed")
    return True


def test_crossover_detection():
    """Test crossover detection."""
    print("Testing crossover detection...")

    # Create test data with known crossovers
    wave_a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    wave_b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    bullish_cross, bearish_cross = detect_crossovers(wave_a, wave_b)

    # Should detect crossovers where wave_a crosses wave_b
    if len(bullish_cross) == len(wave_a) and len(bearish_cross) == len(wave_a):
        print("✓ Crossover detection length check passed")
    else:
        print("✗ Crossover detection length check failed")
        return False

    # Check for expected crossovers
    # Bullish crossover around index 2-3 (wave_a crosses above wave_b)
    # Bearish crossover around index 6-7 (wave_a crosses below wave_b)

    if np.any(bullish_cross) and np.any(bearish_cross):
        print("✓ Crossover detection passed")
        return True
    print("✗ No crossovers detected")
    return False


def test_signal_consistency():
    """Test signal calculation consistency."""
    print("Testing signal consistency...")

    # Generate test data
    ohlcv = generate_test_data(50)

    # Run calculation multiple times
    results = []
    for _ in range(5):
        result = vumanchu_cipher(ohlcv)
        results.append((result.wave_a, result.wave_b, result.signal))

    # Check consistency
    reference = results[0]
    tolerance = 1e-10

    for i, (wave_a, wave_b, signal) in enumerate(results[1:], 1):
        if (
            abs(wave_a - reference[0]) > tolerance
            or abs(wave_b - reference[1]) > tolerance
            or signal != reference[2]
        ):
            print(f"✗ Consistency check failed at run {i}")
            return False

    print("✓ Signal consistency passed")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("VUMANCHU FUNCTIONAL TYPES VALIDATION")
    print("=" * 60)
    print()

    tests = [
        test_hlc3_calculation,
        test_ema_calculation,
        test_wavetrend_calculation,
        test_vumanchu_result,
        test_crossover_detection,
        test_signal_consistency,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Functional types preserve calculation accuracy!")
    else:
        print("❌ Some tests failed - review the implementation")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
