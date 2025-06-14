#!/usr/bin/env python3
"""
Quick test script to verify the indicator fixes work correctly.
This script tests the edge cases that were causing warnings.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the bot module to the path
sys.path.insert(0, os.path.abspath('.'))

from bot.indicators.ema_ribbon import EMAribbon
from bot.indicators.schaff_trend_cycle import SchaffTrendCycle


def create_test_data(length: int, price_type: str = "normal") -> pd.DataFrame:
    """Create test data for different market conditions."""
    
    if price_type == "flat":
        # Flat market - minimal price movement
        prices = [100.0] * length
        
    elif price_type == "minimal_variance":
        # Very small price movements
        base_price = 100.0
        prices = [base_price + 0.0001 * np.sin(i * 0.1) for i in range(length)]
        
    elif price_type == "insufficient":
        # Insufficient data
        prices = [100.0 + i * 0.1 for i in range(min(length, 10))]
        
    else:
        # Normal market data
        np.random.seed(42)
        prices = [100.0]
        for i in range(1, length):
            change = np.random.normal(0, 0.01)  # 1% volatility
            prices.append(prices[-1] * (1 + change))
    
    return pd.DataFrame({
        'close': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'open': prices,
        'volume': [1000] * len(prices)
    })


def test_ema_ribbon_fixes():
    """Test EMA ribbon fixes."""
    print("Testing EMA Ribbon fixes...")
    
    ema_ribbon = EMAribbon()
    
    # Test 1: Insufficient data
    print("  Test 1: Insufficient data handling")
    df_small = create_test_data(20, "insufficient")
    result = ema_ribbon.get_ribbon_analysis(df_small)
    print(f"    Small data result shape: {result.shape}")
    
    # Test 2: Flat market
    print("  Test 2: Flat market handling")
    df_flat = create_test_data(100, "flat")
    result = ema_ribbon.get_ribbon_analysis(df_flat)
    print(f"    Flat market result shape: {result.shape}")
    if 'ribbon_strength' in result.columns:
        print(f"    Ribbon strength range: {result['ribbon_strength'].min():.6f} - {result['ribbon_strength'].max():.6f}")
    
    # Test 3: Minimal variance
    print("  Test 3: Minimal variance handling")
    df_minimal = create_test_data(100, "minimal_variance")
    result = ema_ribbon.get_ribbon_analysis(df_minimal)
    print(f"    Minimal variance result shape: {result.shape}")
    
    # Test 4: Normal data
    print("  Test 4: Normal data processing")
    df_normal = create_test_data(100, "normal")
    result = ema_ribbon.get_ribbon_analysis(df_normal)
    print(f"    Normal data result shape: {result.shape}")
    print(f"    Number of EMA columns: {len([col for col in result.columns if col.startswith('ema')])}")


def test_stc_fixes():
    """Test Schaff Trend Cycle fixes."""
    print("\nTesting Schaff Trend Cycle fixes...")
    
    stc = SchaffTrendCycle()
    
    # Test 1: Flat market (zero division scenarios)
    print("  Test 1: Flat market zero division handling")
    df_flat = create_test_data(100, "flat")
    result = stc.calculate(df_flat)
    print(f"    Flat market result shape: {result.shape}")
    if 'stc' in result.columns:
        stc_values = result['stc'].dropna()
        if len(stc_values) > 0:
            print(f"    STC range: {stc_values.min():.6f} - {stc_values.max():.6f}")
        else:
            print("    No valid STC values generated")
    
    # Test 2: Minimal variance
    print("  Test 2: Minimal variance handling")
    df_minimal = create_test_data(100, "minimal_variance")
    result = stc.calculate(df_minimal)
    print(f"    Minimal variance result shape: {result.shape}")
    
    # Test 3: Normal data
    print("  Test 3: Normal data processing")
    df_normal = create_test_data(100, "normal")
    result = stc.calculate(df_normal)
    print(f"    Normal data result shape: {result.shape}")
    if 'stc' in result.columns:
        stc_values = result['stc'].dropna()
        if len(stc_values) > 0:
            print(f"    STC range: {stc_values.min():.2f} - {stc_values.max():.2f}")


def main():
    """Run all tests."""
    print("Testing indicator warning fixes...\n")
    
    try:
        test_ema_ribbon_fixes()
        test_stc_fixes()
        print("\n✅ All tests completed successfully!")
        print("The indicator fixes appear to be working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)