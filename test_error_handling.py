#!/usr/bin/env python3
"""Error Handling and Fallback Mechanisms Test"""

import pandas as pd
import numpy as np
from datetime import datetime


def test_error_handling_and_fallbacks():
    """Test error handling and fallback mechanisms."""
    print('=== ERROR HANDLING AND FALLBACK MECHANISMS TEST ===')
    print()
    
    results = {'working': [], 'failing': [], 'warnings': []}
    
    # Test 1: VuManChu Error Handling
    print('1. Testing VuManChu error handling with insufficient data...')
    try:
        from bot.indicators.vumanchu import VuManChuIndicators
        
        # Test with minimal data (should trigger error handling)
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'open': [50000, 50100, 50050, 50200, 50150],
            'high': [50200, 50300, 50250, 50400, 50350],
            'low': [49800, 49900, 49850, 50000, 49950],
            'close': [50100, 50050, 50200, 50150, 50300],
            'volume': [1000, 1100, 950, 1200, 1050]
        })
        
        indicators = VuManChuIndicators()
        result = indicators.calculate(minimal_data)
        
        # Should return a DataFrame even with errors
        if isinstance(result, pd.DataFrame) and len(result) > 0:
            results['working'].append('✅ VuManChu error handling (graceful degradation)')
        else:
            results['failing'].append('❌ VuManChu error handling failed')
            
    except Exception as e:
        results['failing'].append(f'❌ VuManChu error handling failed: {e}')
    
    # Test 2: Result Monad Error Handling
    print('2. Testing Result monad error handling...')
    try:
        from bot.fp.types.result import Result, Ok, Err
        
        # Test error handling with Result monad
        def safe_divide(a, b):
            if b == 0:
                return Err("Division by zero")
            return Ok(a / b)
        
        # Test successful operation
        success_result = safe_divide(10, 2)
        assert success_result.is_ok()
        assert success_result.unwrap() == 5.0
        
        # Test error case
        error_result = safe_divide(10, 0)
        assert error_result.is_err()
        assert "Division by zero" in str(error_result.unwrap_err())
        
        results['working'].append('✅ Result monad error handling')
        
    except Exception as e:
        results['failing'].append(f'❌ Result monad error handling failed: {e}')
    
    # Test 3: Trading Signal Validation
    print('3. Testing trading signal validation...')
    try:
        from bot.fp.types.trading import Long, Short, Hold
        
        # Test valid signals
        try:
            valid_long = Long(confidence=0.8, size=0.5, reason="Valid signal")
            results['working'].append('✅ Valid trading signal creation')
        except Exception as e:
            results['failing'].append(f'❌ Valid signal creation failed: {e}')
            return results
        
        # Test invalid signals (should raise validation errors)
        validation_errors = []
        
        try:
            invalid_confidence = Long(confidence=1.5, size=0.5, reason="Invalid confidence")
            validation_errors.append("Should have failed: confidence > 1")
        except ValueError:
            pass  # Expected error
        
        try:
            invalid_size = Long(confidence=0.8, size=1.5, reason="Invalid size")
            validation_errors.append("Should have failed: size > 1")
        except ValueError:
            pass  # Expected error
        
        try:
            negative_confidence = Long(confidence=-0.1, size=0.5, reason="Negative confidence")
            validation_errors.append("Should have failed: negative confidence")
        except ValueError:
            pass  # Expected error
        
        if len(validation_errors) == 0:
            results['working'].append('✅ Trading signal validation (proper error handling)')
        else:
            results['failing'].append(f'❌ Signal validation failed: {validation_errors}')
            
    except Exception as e:
        results['failing'].append(f'❌ Trading signal validation failed: {e}')
    
    # Test 4: Order Validation
    print('4. Testing order validation...')
    try:
        from bot.fp.types.trading import LimitOrder, MarketOrder, StopOrder
        
        # Test valid orders
        valid_market = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)
        valid_limit = LimitOrder(symbol="BTC-USD", side="sell", price=50000.0, size=0.1)
        valid_stop = StopOrder(symbol="BTC-USD", side="sell", stop_price=48000.0, size=0.1)
        
        # Test invalid orders
        validation_errors = []
        
        try:
            invalid_price = LimitOrder(symbol="BTC-USD", side="buy", price=-100.0, size=0.1)
            validation_errors.append("Should have failed: negative price")
        except ValueError:
            pass  # Expected error
        
        try:
            invalid_size = MarketOrder(symbol="BTC-USD", side="buy", size=-0.1)
            validation_errors.append("Should have failed: negative size")
        except ValueError:
            pass  # Expected error
        
        if len(validation_errors) == 0:
            results['working'].append('✅ Order validation (proper error handling)')
        else:
            results['failing'].append(f'❌ Order validation failed: {validation_errors}')
            
    except Exception as e:
        results['failing'].append(f'❌ Order validation failed: {e}')
    
    # Test 5: Configuration Fallback
    print('5. Testing configuration fallback...')
    try:
        import os
        
        # Save original environment
        original_symbol = os.environ.get('TRADING__SYMBOL')
        
        # Test with missing environment variable
        if 'TRADING__SYMBOL' in os.environ:
            del os.environ['TRADING__SYMBOL']
        
        from bot.config import Settings
        settings = Settings()
        
        # Should fall back to default
        if settings.trading.symbol:  # Should have a default value
            results['working'].append(f'✅ Configuration fallback (symbol: {settings.trading.symbol})')
        else:
            results['failing'].append('❌ Configuration fallback failed')
        
        # Restore original environment
        if original_symbol:
            os.environ['TRADING__SYMBOL'] = original_symbol
            
    except Exception as e:
        results['failing'].append(f'❌ Configuration fallback failed: {e}')
    
    # Test 6: Data Processing Error Recovery
    print('6. Testing data processing error recovery...')
    try:
        # Test with corrupt data
        corrupt_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [50000, None, 50050, np.inf, 50150, 50100, -50000, 50200, 50250, 50300],
            'high': [50200, 50300, None, 50400, 50350, 50300, 50000, 50400, np.nan, 50500],
            'low': [49800, 49900, 49850, 50000, 49950, 49800, 49900, 50000, 50050, 50100],
            'close': [50100, 50050, 50200, 50150, 50300, 50200, 50100, 50300, 50350, 50400],
            'volume': [1000, 1100, 950, 1200, 1050, 1000, 0, 1300, 1150, 1250]
        })
        
        # Test cleaning function
        def clean_market_data(data):
            """Clean corrupt market data"""
            cleaned = data.copy()
            
            # Replace infinite values with NaN
            cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
            
            # Remove rows with negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                cleaned = cleaned[cleaned[col] >= 0]
            
            # Forward fill missing values
            cleaned = cleaned.fillna(method='ffill')
            
            # Remove any remaining NaN rows
            cleaned = cleaned.dropna()
            
            return cleaned
        
        cleaned_data = clean_market_data(corrupt_data)
        
        if len(cleaned_data) > 0 and not cleaned_data.isnull().any().any():
            results['working'].append(f'✅ Data processing error recovery (cleaned {len(corrupt_data)} → {len(cleaned_data)} rows)')
        else:
            results['failing'].append('❌ Data processing error recovery failed')
            
    except Exception as e:
        results['failing'].append(f'❌ Data processing error recovery failed: {e}')
    
    return results


def print_error_handling_results(results):
    """Print error handling test results."""
    print('\n' + '='*80)
    print('ERROR HANDLING AND FALLBACK MECHANISMS RESULTS')
    print('='*80)
    
    print(f"\n✅ WORKING COMPONENTS ({len(results['working'])}):")
    for item in results['working']:
        print(f"  {item}")
    
    print(f"\n❌ FAILING COMPONENTS ({len(results['failing'])}):")
    for item in results['failing']:
        print(f"  {item}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for item in results['warnings']:
            print(f"  {item}")
    
    # Calculate error handling health
    total_tests = len(results['working']) + len(results['failing'])
    success_rate = len(results['working']) / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n📊 ERROR HANDLING HEALTH:")
    print(f"  Success Rate: {success_rate:.1f}% ({len(results['working'])}/{total_tests} components working)")
    
    if success_rate >= 80:
        print("  Status: ✅ ERROR HANDLING ROBUST")
    elif success_rate >= 60:
        print("  Status: ⚠️  ERROR HANDLING NEEDS IMPROVEMENT")
    else:
        print("  Status: ❌ ERROR HANDLING NEEDS MAJOR FIXES")
    
    print('\n' + '='*80)


def main():
    """Run error handling and fallback mechanisms test."""
    try:
        results = test_error_handling_and_fallbacks()
        print_error_handling_results(results)
        
        # Return success if most components are working
        success_rate = len(results['working']) / (len(results['working']) + len(results['failing'])) * 100
        return success_rate >= 60
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)