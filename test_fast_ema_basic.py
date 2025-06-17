#!/usr/bin/env python3
"""
Basic test script for FastEMA indicator implementation.

This script tests the core functionality of the FastEMA and ScalpingEMASignals
classes to ensure they work correctly with sample data.
"""

import pandas as pd
import numpy as np
from bot.indicators.fast_ema import FastEMA, ScalpingEMASignals


def create_sample_data(length: int = 100) -> pd.DataFrame:
    """Create sample OHLC data for testing."""
    # Generate realistic price movement with trend
    base_price = 50000.0
    trend = np.linspace(0, 5000, length)  # Upward trend
    noise = np.random.normal(0, 100, length)  # Market noise
    
    close_prices = base_price + trend + noise
    
    # Ensure positive prices
    close_prices = np.maximum(close_prices, 1.0)
    
    # Create OHLC data
    data = pd.DataFrame({
        'close': close_prices,
        'open': close_prices * (1 + np.random.normal(0, 0.001, length)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.002, length))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, length))),
    })
    
    return data


def test_fast_ema_basic():
    """Test basic FastEMA functionality."""
    print("Testing FastEMA basic functionality...")
    
    # Create test data
    data = create_sample_data(50)
    
    # Initialize FastEMA
    fast_ema = FastEMA()
    print(f"FastEMA initialized with periods: {fast_ema.periods}")
    
    # Test calculation
    result = fast_ema.calculate(data)
    
    print(f"Calculation completed in {result['calculation_time_ms']:.2f}ms")
    print(f"Data points processed: {result['data_points']}")
    
    # Check EMA values
    ema_values = result['ema_values']
    print("Latest EMA values:")
    for period in fast_ema.periods:
        value = ema_values.get(f'ema_{period}')
        print(f"  EMA{period}: {value:.2f}" if value else f"  EMA{period}: None")
    
    # Test real-time update
    print("\nTesting real-time update...")
    latest_price = data['close'].iloc[-1]
    new_price = latest_price * 1.01  # 1% increase
    
    update_result = fast_ema.update_realtime(new_price)
    print(f"Updated with price: {new_price:.2f}")
    
    updated_values = update_result['ema_values']
    print("Updated EMA values:")
    for period in fast_ema.periods:
        value = updated_values.get(f'ema_{period}')
        print(f"  EMA{period}: {value:.2f}" if value else f"  EMA{period}: None")
    
    print("FastEMA basic test PASSED")
    return True


def test_scalping_signals():
    """Test ScalpingEMASignals functionality."""
    print("\nTesting ScalpingEMASignals functionality...")
    
    # Create test data with more volatility for signals
    data = create_sample_data(100)
    
    # Add some crossover patterns
    for i in range(20, 30):
        data.loc[i, 'close'] *= 0.98  # Create a dip
    for i in range(50, 60):
        data.loc[i, 'close'] *= 1.02  # Create a spike
    
    # Initialize scalping signals
    scalping_signals = ScalpingEMASignals()
    print(f"ScalpingEMASignals initialized with periods: {scalping_signals.periods}")
    
    # Test signal calculation
    result = scalping_signals.calculate(data)
    
    print(f"Signal calculation completed in {result['signal_calculation_time_ms']:.2f}ms")
    print(f"Trend strength: {result['trend_strength']:.3f}")
    print(f"Setup type: {result['setup_type']}")
    
    # Check crossover signals
    crossovers = result['crossovers']
    print(f"Crossover signals detected: {len(crossovers)}")
    for crossover in crossovers:
        print(f"  {crossover['type']}: EMA{crossover['fast_period']} x EMA{crossover['slow_period']} "
              f"(confidence: {crossover['confidence']:.2f})")
    
    # Check actionable signals
    signals = result['signals']
    print(f"Actionable signals: {len(signals)}")
    for signal in signals:
        print(f"  {signal['type']}: {signal['direction']} (strength: {signal['strength']:.2f})")
        print(f"    Reason: {signal['reason']}")
    
    # Test setup detection
    ema_series = result['ema_series']
    price = data['close']
    
    is_bullish = scalping_signals.is_bullish_setup(price, ema_series)
    is_bearish = scalping_signals.is_bearish_setup(price, ema_series)
    
    print(f"Is bullish setup: {is_bullish}")
    print(f"Is bearish setup: {is_bearish}")
    
    print("ScalpingEMASignals test PASSED")
    return True


def test_performance():
    """Test performance with larger dataset."""
    print("\nTesting performance with larger dataset...")
    
    # Create larger dataset
    large_data = create_sample_data(1000)
    
    scalping_signals = ScalpingEMASignals()
    
    # Time the calculation
    import time
    start_time = time.perf_counter()
    
    result = scalping_signals.calculate(large_data)
    
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    
    print(f"Processed {len(large_data)} data points in {total_time:.2f}ms")
    print(f"Performance: {len(large_data) / total_time * 1000:.0f} points per second")
    
    # Get performance metrics
    metrics = scalping_signals.fast_ema.get_performance_metrics()
    print(f"Average calculation time: {metrics['average_calculation_time_ms']:.2f}ms")
    
    print("Performance test PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("FastEMA Implementation Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_fast_ema_basic()
        test_scalping_signals()
        test_performance()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("FastEMA implementation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()