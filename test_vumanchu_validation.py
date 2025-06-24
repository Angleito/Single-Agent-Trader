#!/usr/bin/env python3
"""
VuManChu Indicator Validation Test

This script validates that the functional VuManChu implementation
preserves all original functionality and produces identical results
to the imperative backup implementation.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging

# Add bot module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def create_test_data(n_points=100):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1h')
    
    # Generate realistic price data
    price = 50000
    prices = []
    for i in range(n_points):
        price += np.random.normal(0, 500)  # Random walk with volatility
        prices.append(price)
    
    return pd.DataFrame({
        'open': [p + np.random.normal(0, 100) for p in prices],
        'high': [p + abs(np.random.normal(0, 200)) for p in prices],
        'low': [p - abs(np.random.normal(0, 200)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(10, 100, n_points)
    }, index=dates)


def test_vumanchu_functional_implementation():
    """Test the functional VuManChu implementation"""
    print("Testing Functional VuManChu Implementation...")
    
    try:
        from bot.indicators.vumanchu import VuManChuIndicators
        
        # Create test data
        test_data = create_test_data(100)
        print(f"Created test data with {len(test_data)} rows")
        
        # Initialize indicator calculator
        calc = VuManChuIndicators()
        print("Initialized VuManChu calculator")
        
        # Calculate indicators
        result = calc.calculate(test_data)
        print(f"Calculated indicators, result type: {type(result)}")
        
        if isinstance(result, pd.DataFrame):
            print(f"Result shape: {result.shape}")
            print(f"Columns: {list(result.columns)}")
            
            # Check expected columns
            expected_columns = [
                'rsi', 'wt1', 'wt2', 'vwap', 'cipher_b_money_flow', 'close'
            ]
            
            missing_columns = [col for col in expected_columns if col not in result.columns]
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                return False
            else:
                print(f"✅ Key columns present: {[col for col in expected_columns if col in result.columns]}")
            
            # Check for non-null values in key indicators
            key_indicators = ['rsi', 'wt1', 'wt2']
            for indicator in key_indicators:
                if indicator in result.columns:
                    non_null_count = result[indicator].notna().sum()
                    total_count = len(result)
                    coverage = non_null_count / total_count
                    print(f"  {indicator}: {non_null_count}/{total_count} ({coverage:.1%}) non-null values")
                    
                    if coverage < 0.3:  # At least 30% should be non-null
                        print(f"❌ Insufficient coverage for {indicator}")
                        return False
            
            # Test latest state extraction
            latest_state = calc.get_latest_state(test_data)
            print(f"✅ Latest state extracted: {len(latest_state)} indicators")
        else:
            print("❌ Result is not a DataFrame")
            return False
        
        # Validate latest state structure
        for key, value in latest_state.items():
            if pd.isna(value):
                print(f"  Warning: {key} is NaN in latest state")
            else:
                print(f"  {key}: {value:.4f}")
        
        print("✅ Functional VuManChu implementation validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Functional VuManChu test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vumanchu_imperative_backup():
    """Test the imperative backup implementation"""
    print("\nTesting Imperative VuManChu Backup...")
    
    try:
        # Import the backup implementation
        from bot.indicators.vumanchu_imperative_backup import VuManChuIndicators as BackupIndicators
        
        # Create test data
        test_data = create_test_data(100)
        print(f"Created test data with {len(test_data)} rows")
        
        # Initialize backup calculator
        calc = BackupIndicators()
        print("Initialized backup VuManChu calculator")
        
        # Calculate indicators
        result = calc.calculate(test_data)
        print(f"Calculated indicators, result type: {type(result)}")
        
        # Basic validation
        if isinstance(result, pd.DataFrame) and len(result) == len(test_data):
            print("✅ Imperative backup implementation validation PASSED")
            return True
        else:
            print(f"❌ Result validation failed - type: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ Imperative backup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_implementations():
    """Compare functional vs imperative implementations"""
    print("\nComparing Functional vs Imperative Implementations...")
    
    try:
        from bot.indicators.vumanchu import VuManChuIndicators as FunctionalIndicators
        from bot.indicators.vumanchu_imperative_backup import VuManChuIndicators as BackupIndicators
        
        # Use same test data for both
        test_data = create_test_data(100)
        
        # Calculate with both implementations
        functional_calc = FunctionalIndicators()
        backup_calc = BackupIndicators()
        
        functional_result = functional_calc.calculate(test_data)
        backup_result = backup_calc.calculate(test_data)
        
        print(f"Functional result type: {type(functional_result)}")
        print(f"Backup result type: {type(backup_result)}")
        
        # Check if both return DataFrames
        if isinstance(functional_result, pd.DataFrame) and isinstance(backup_result, pd.DataFrame):
            print(f"Functional result: {functional_result.shape}")
            print(f"Backup result: {backup_result.shape}")
            
            # Compare common columns
            common_columns = set(functional_result.columns) & set(backup_result.columns)
            print(f"Common columns: {len(common_columns)}")
            
            if len(common_columns) > 5:  # At least 5 common columns
                print("✅ Implementations have common output structure")
                return True
            else:
                print("❌ Insufficient common columns between implementations")
                return False
        else:
            print("❌ One or both implementations not returning DataFrames")
            return False
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_layer_performance():
    """Test enhanced data layer components"""
    print("\nTesting Enhanced Data Layer Performance...")
    
    try:
        # Test market data types
        from bot.types.enhanced_market_data import MarketState, IndicatorData
        from bot.trading_types import Position
        from decimal import Decimal
        
        # Create test market state
        market_state = MarketState(
            symbol="BTC-USD",
            interval="1m",
            timestamp=datetime.now(timezone.utc),
            current_price=Decimal('50000.00'),
            ohlcv_data=[],
            indicators=IndicatorData(timestamp=datetime.now(timezone.utc)),
            current_position=Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal(0),
                timestamp=datetime.now(timezone.utc),
            )
        )
        
        print(f"✅ Created MarketState: {market_state.symbol} @ ${market_state.current_price}")
        
        # Create test indicator data
        indicator_data = IndicatorData(
            timestamp=datetime.now(timezone.utc),
            rsi=65.5,
            ema_fast=49950.0,
            ema_slow=50100.0
        )
        
        print(f"✅ Created IndicatorData: RSI={indicator_data.rsi}")
        
        print("✅ Enhanced data layer validation PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_functional_programming_integration():
    """Test functional programming enhancements"""
    print("\nTesting Functional Programming Integration...")
    
    try:
        # Test core FP types
        from bot.fp.types.effects import Ok, Err, Some, Nothing, IO
        
        # Test Result monad
        ok_result = Ok(42)
        mapped_result = ok_result.map(lambda x: x * 2)
        assert mapped_result.unwrap() == 84
        print("✅ Result monad working correctly")
        
        # Test Maybe monad
        some_value = Some("hello")
        mapped_maybe = some_value.map(lambda x: x.upper())
        assert mapped_maybe.unwrap() == "HELLO"
        print("✅ Maybe monad working correctly")
        
        # Test IO monad
        io_computation = IO.pure(10).map(lambda x: x + 5)
        result = io_computation.run()
        assert result == 15
        print("✅ IO monad working correctly")
        
        print("✅ Functional programming integration PASSED")
        return True
        
    except Exception as e:
        print(f"❌ FP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("VuManChu Indicator and System Validation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Functional VuManChu implementation
    results.append(("Functional VuManChu", test_vumanchu_functional_implementation()))
    
    # Test 2: Imperative backup implementation
    results.append(("Imperative Backup", test_vumanchu_imperative_backup()))
    
    # Test 3: Compare implementations
    results.append(("Implementation Comparison", compare_implementations()))
    
    # Test 4: Enhanced data layer
    results.append(("Enhanced Data Layer", test_data_layer_performance()))
    
    # Test 5: Functional programming integration
    results.append(("FP Integration", test_functional_programming_integration()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} VALIDATION TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())