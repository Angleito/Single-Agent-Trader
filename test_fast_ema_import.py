#!/usr/bin/env python3
"""
Simple import test for FastEMA to verify the module can be imported correctly.
"""

def test_imports():
    """Test that all imports work correctly."""
    try:
        # Test import from indicators package
        from bot.indicators import FastEMA, ScalpingEMASignals
        print("✓ Successfully imported FastEMA and ScalpingEMASignals from bot.indicators")
        
        # Test direct import
        from bot.indicators.fast_ema import FastEMA as DirectFastEMA, ScalpingEMASignals as DirectScalpingEMASignals
        print("✓ Successfully imported directly from bot.indicators.fast_ema")
        
        # Check that classes can be instantiated (without calling methods)
        fast_ema = FastEMA()
        print(f"✓ FastEMA instance created with periods: {fast_ema.periods}")
        
        scalping_signals = ScalpingEMASignals()
        print(f"✓ ScalpingEMASignals instance created with periods: {scalping_signals.periods}")
        
        # Verify expected attributes exist
        assert hasattr(fast_ema, 'calculate'), "FastEMA missing calculate method"
        assert hasattr(fast_ema, 'update_realtime'), "FastEMA missing update_realtime method"
        assert hasattr(scalping_signals, 'get_crossover_signals'), "ScalpingEMASignals missing get_crossover_signals method"
        assert hasattr(scalping_signals, 'get_trend_strength'), "ScalpingEMASignals missing get_trend_strength method"
        assert hasattr(scalping_signals, 'is_bullish_setup'), "ScalpingEMASignals missing is_bullish_setup method"
        assert hasattr(scalping_signals, 'is_bearish_setup'), "ScalpingEMASignals missing is_bearish_setup method"
        
        print("✓ All expected methods are present")
        
        # Test default periods are correct for scalping
        expected_periods = [3, 5, 8, 13]
        assert fast_ema.periods == expected_periods, f"Expected periods {expected_periods}, got {fast_ema.periods}"
        print(f"✓ Default periods are correct for scalping: {expected_periods}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run import tests."""
    print("=" * 60)
    print("FastEMA Import Test")
    print("=" * 60)
    
    success = test_imports()
    
    if success:
        print("\n" + "=" * 60)
        print("IMPORT TEST PASSED!")
        print("FastEMA module is properly structured and importable.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("IMPORT TEST FAILED!")
        print("Check the module structure and imports.")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    main()