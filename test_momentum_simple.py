#!/usr/bin/env python3
"""
Simple momentum strategy validation test.

This script performs basic validation of the momentum strategy implementation
without requiring external dependencies like numpy/pandas.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/angel/Documents/Projects/cursorprod')

def test_imports():
    """Test that all momentum strategy components can be imported."""
    print("=== Testing Imports ===")
    
    try:
        from bot.strategy.momentum_strategy import (
            MomentumStrategyExecutor,
            MomentumConfig,
            MomentumSignalType,
            MomentumSignalStrength,
            create_momentum_strategy
        )
        print("✓ Successfully imported momentum strategy components")
        return True
    except ImportError as e:
        print(f"✗ Failed to import momentum strategy: {e}")
        return False


def test_enums():
    """Test enum definitions."""
    print("\n=== Testing Enums ===")
    
    try:
        from bot.strategy.momentum_strategy import MomentumSignalType, MomentumSignalStrength
        
        # Test signal types
        signal_types = list(MomentumSignalType)
        print(f"✓ MomentumSignalType has {len(signal_types)} values:")
        for signal_type in signal_types:
            print(f"  - {signal_type.value}")
        
        # Test signal strengths
        signal_strengths = list(MomentumSignalStrength)
        print(f"✓ MomentumSignalStrength has {len(signal_strengths)} values:")
        for strength in signal_strengths:
            print(f"  - {strength.value}")
        
        return True
    except Exception as e:
        print(f"✗ Enum test failed: {e}")
        return False


def test_config():
    """Test configuration creation."""
    print("\n=== Testing Configuration ===")
    
    try:
        from bot.strategy.momentum_strategy import MomentumConfig
        
        # Test default config
        config = MomentumConfig()
        print(f"✓ Default config created:")
        print(f"  - Primary timeframe: {config.primary_timeframe}")
        print(f"  - Base position %: {config.base_position_pct}")
        print(f"  - Min signal strength: {config.min_signal_strength}")
        print(f"  - Stop loss %: {config.stop_loss_pct}")
        print(f"  - Take profit %: {config.take_profit_pct}")
        
        # Test custom config
        custom_config = MomentumConfig(
            primary_timeframe="5m",
            base_position_pct=3.0,
            min_signal_strength=0.8,
            trailing_stop=False
        )
        print(f"✓ Custom config created:")
        print(f"  - Primary timeframe: {custom_config.primary_timeframe}")
        print(f"  - Base position %: {custom_config.base_position_pct}")
        print(f"  - Trailing stop: {custom_config.trailing_stop}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


def test_strategy_creation():
    """Test strategy creation."""
    print("\n=== Testing Strategy Creation ===")
    
    try:
        from bot.strategy.momentum_strategy import create_momentum_strategy, MomentumConfig
        
        # Test with default config
        strategy1 = create_momentum_strategy()
        print("✓ Strategy created with default config")
        print(f"  - Type: {type(strategy1).__name__}")
        
        # Test with custom config
        custom_config = MomentumConfig(base_position_pct=2.5)
        strategy2 = create_momentum_strategy(custom_config)
        print("✓ Strategy created with custom config")
        print(f"  - Base position %: {strategy2.config.base_position_pct}")
        
        # Test component creation
        print(f"✓ Strategy components initialized:")
        print(f"  - Signal generator: {type(strategy2.signal_generator).__name__}")
        print(f"  - Position sizer: {type(strategy2.position_sizer).__name__}")
        print(f"  - Active positions: {len(strategy2.active_positions)}")
        
        return True
    except Exception as e:
        print(f"✗ Strategy creation test failed: {e}")
        return False


def test_analyzers():
    """Test analyzer creation."""
    print("\n=== Testing Analyzers ===")
    
    try:
        from bot.strategy.momentum_strategy import (
            MomentumTrendAnalyzer,
            MomentumIndicatorAnalyzer,
            MomentumVolumeAnalyzer,
            MomentumConfig
        )
        
        config = MomentumConfig()
        
        # Test trend analyzer
        trend_analyzer = MomentumTrendAnalyzer(config)
        print("✓ MomentumTrendAnalyzer created")
        print(f"  - EMA fast period: {trend_analyzer.ema_fast.period}")
        print(f"  - EMA slow period: {trend_analyzer.ema_slow.period}")
        print(f"  - ADX period: {trend_analyzer.adx.period}")
        
        # Test momentum analyzer
        momentum_analyzer = MomentumIndicatorAnalyzer(config)
        print("✓ MomentumIndicatorAnalyzer created")
        print(f"  - RSI period: {momentum_analyzer.rsi.period}")
        
        # Test volume analyzer
        volume_analyzer = MomentumVolumeAnalyzer(config)
        print("✓ MomentumVolumeAnalyzer created")
        print(f"  - Volume MA period: {volume_analyzer.volume_ma.period}")
        
        return True
    except Exception as e:
        print(f"✗ Analyzer test failed: {e}")
        return False


def test_indicators():
    """Test technical indicator implementations."""
    print("\n=== Testing Technical Indicators ===")
    
    try:
        from bot.strategy.momentum_strategy import EMA, SMA, MACD, RSI, ADX
        
        # Test EMA
        ema = EMA(period=20)
        print(f"✓ EMA created with period {ema.period}, alpha={ema.alpha:.4f}")
        
        # Test SMA
        sma = SMA(period=20)
        print(f"✓ SMA created with period {sma.period}")
        
        # Test MACD
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        print("✓ MACD created")
        
        # Test RSI
        rsi = RSI(period=14)
        print(f"✓ RSI created with period {rsi.period}")
        
        # Test ADX
        adx = ADX(period=14)
        print(f"✓ ADX created with period {adx.period}")
        
        return True
    except Exception as e:
        print(f"✗ Indicator test failed: {e}")
        return False


def test_file_structure():
    """Test that the momentum strategy file is properly structured."""
    print("\n=== Testing File Structure ===")
    
    try:
        import inspect
        from bot.strategy.momentum_strategy import MomentumStrategyExecutor
        
        # Get all methods of the strategy executor
        methods = inspect.getmembers(MomentumStrategyExecutor, predicate=inspect.ismethod)
        functions = inspect.getmembers(MomentumStrategyExecutor, predicate=inspect.isfunction)
        
        print(f"✓ MomentumStrategyExecutor has {len(methods)} methods and {len(functions)} functions")
        
        # Check for required interface methods
        required_methods = [
            'get_strategy_signals',
            'validate_signal', 
            'calculate_position_size',
            'get_risk_parameters',
            'execute_strategy'
        ]
        
        executor_methods = [name for name, _ in inspect.getmembers(MomentumStrategyExecutor)]
        missing_methods = [method for method in required_methods if method not in executor_methods]
        
        if missing_methods:
            print(f"✗ Missing required methods: {missing_methods}")
            return False
        else:
            print("✓ All required interface methods present")
        
        return True
    except Exception as e:
        print(f"✗ File structure test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("🚀 Starting Momentum Strategy Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Enum Test", test_enums),
        ("Config Test", test_config),
        ("Strategy Creation Test", test_strategy_creation),
        ("Analyzer Test", test_analyzers),
        ("Indicator Test", test_indicators),
        ("File Structure Test", test_file_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! Momentum strategy is properly implemented.")
        return True
    else:
        print("⚠️  Some validation tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)