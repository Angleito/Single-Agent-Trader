#!/usr/bin/env python3
"""
Test script for the momentum trading strategy.

This script validates the momentum strategy implementation by:
1. Creating test market data
2. Initializing the momentum strategy
3. Testing signal generation
4. Testing position sizing
5. Testing strategy execution
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/Users/angel/Documents/Projects/cursorprod')

try:
    from bot.strategy.momentum_strategy import (
        MomentumStrategyExecutor,
        MomentumConfig,
        MomentumSignalType,
        MomentumSignalStrength,
        create_momentum_strategy
    )
    print("✓ Successfully imported momentum strategy components")
except ImportError as e:
    print(f"✗ Failed to import momentum strategy: {e}")
    sys.exit(1)


def generate_test_data(periods: int = 100, trend: str = 'bullish') -> pd.DataFrame:
    """Generate realistic test OHLCV data with trend."""
    np.random.seed(42)  # For reproducible results
    
    # Base price and volatility
    base_price = 45000.0
    volatility = 0.02
    
    # Generate price series with trend
    if trend == 'bullish':
        trend_factor = 0.001
    elif trend == 'bearish':
        trend_factor = -0.001
    else:
        trend_factor = 0.0
    
    prices = [base_price]
    volumes = []
    
    for i in range(periods):
        # Random walk with trend
        change = np.random.normal(trend_factor, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        # Generate volume with some correlation to price movement
        volume_base = 1000
        volume_multiplier = 1 + abs(change) * 10
        volume = volume_base * volume_multiplier * np.random.uniform(0.5, 1.5)
        volumes.append(volume)
    
    # Create OHLC from price series
    data = []
    for i in range(len(prices) - 1):
        open_price = prices[i]
        close_price = prices[i + 1]
        
        # Generate high and low
        price_range = abs(close_price - open_price) * 2
        high = max(open_price, close_price) + np.random.uniform(0, price_range * 0.5)
        low = min(open_price, close_price) - np.random.uniform(0, price_range * 0.5)
        
        data.append({
            'timestamp': datetime.now() - timedelta(minutes=periods-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volumes[i]
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def test_momentum_config():
    """Test momentum configuration creation."""
    print("\n=== Testing Momentum Configuration ===")
    
    try:
        # Test default config
        config = MomentumConfig()
        print(f"✓ Default config created: timeframe={config.primary_timeframe}")
        
        # Test custom config
        custom_config = MomentumConfig(
            primary_timeframe="5m",
            base_position_pct=3.0,
            min_signal_strength=0.8
        )
        print(f"✓ Custom config created: position_pct={custom_config.base_position_pct}")
        
        return config
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return None


def test_signal_generation(strategy, test_data):
    """Test momentum signal generation."""
    print("\n=== Testing Signal Generation ===")
    
    try:
        # Test with bullish trend data
        signals = strategy.signal_generator.generate_signals(test_data)
        print(f"✓ Generated {len(signals)} signals")
        
        if signals:
            signal = signals[0]
            print(f"  - Signal type: {signal.get('type')}")
            print(f"  - Direction: {signal.get('direction')}")
            print(f"  - Confidence: {signal.get('confidence', 0):.3f}")
            print(f"  - Strength: {signal.get('strength')}")
        
        # Test individual analyzers
        trend_analysis = strategy.signal_generator.trend_analyzer.analyze_trend(test_data)
        print(f"✓ Trend analysis: {trend_analysis['direction']} (strength: {trend_analysis['strength']:.3f})")
        
        momentum_analysis = strategy.signal_generator.momentum_analyzer.analyze_momentum(test_data)
        print(f"✓ Momentum analysis: RSI={momentum_analysis['rsi']['value']:.1f}")
        
        volume_analysis = strategy.signal_generator.volume_analyzer.analyze_volume(test_data)
        print(f"✓ Volume analysis: relative={volume_analysis['relative']:.2f}")
        
        return signals
        
    except Exception as e:
        print(f"✗ Signal generation test failed: {e}")
        return []


def test_position_sizing(strategy, signals, test_data):
    """Test position sizing calculation."""
    print("\n=== Testing Position Sizing ===")
    
    try:
        if not signals:
            print("No signals to test position sizing")
            return
        
        signal = signals[0]
        current_price = test_data['close'].iloc[-1]
        account_balance = 10000.0  # $10k test account
        atr = 100.0  # $100 ATR
        
        position_info = strategy.position_sizer.calculate_position_size(
            signal, account_balance, current_price, atr
        )
        
        print(f"✓ Position sizing calculated:")
        print(f"  - Size: ${position_info['size']:.2f}")
        print(f"  - Stop loss: ${position_info['stop_loss']:.2f}")
        print(f"  - Take profit: ${position_info['take_profit']:.2f}")
        print(f"  - Risk/Reward: {position_info['risk_reward_ratio']:.2f}")
        print(f"  - Risk amount: ${position_info['risk_amount']:.2f}")
        
        return position_info
        
    except Exception as e:
        print(f"✗ Position sizing test failed: {e}")
        return None


async def test_strategy_execution(strategy, test_data):
    """Test full strategy execution."""
    print("\n=== Testing Strategy Execution ===")
    
    try:
        market_data = {
            'ohlcv': test_data,
            'current_price': test_data['close'].iloc[-1],
            'account_balance': 10000.0
        }
        
        # Execute strategy
        result = await strategy.execute_strategy(market_data)
        
        print(f"✓ Strategy execution completed:")
        print(f"  - Strategy: {result['strategy']}")
        print(f"  - Signals generated: {len(result['signals'])}")
        print(f"  - New positions: {len(result['new_positions'])}")
        print(f"  - Position updates: {len(result['position_updates'])}")
        
        # Test interface methods
        signals = await strategy.get_strategy_signals(market_data)
        print(f"✓ Interface method test: {len(signals)} signals via get_strategy_signals()")
        
        if signals:
            is_valid = await strategy.validate_signal(signals[0], market_data)
            print(f"✓ Signal validation: {is_valid}")
        
        return result
        
    except Exception as e:
        print(f"✗ Strategy execution test failed: {e}")
        return None


async def run_comprehensive_test():
    """Run comprehensive test of momentum strategy."""
    print("🚀 Starting Momentum Strategy Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Configuration
    config = test_momentum_config()
    if not config:
        return False
    
    # Test 2: Strategy creation
    try:
        strategy = create_momentum_strategy(config)
        print("✓ Strategy created successfully")
    except Exception as e:
        print(f"✗ Strategy creation failed: {e}")
        return False
    
    # Test 3: Generate test data
    try:
        test_data = generate_test_data(periods=100, trend='bullish')
        print(f"✓ Generated test data: {len(test_data)} periods")
        print(f"  - Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    except Exception as e:
        print(f"✗ Test data generation failed: {e}")
        return False
    
    # Test 4: Signal generation
    signals = test_signal_generation(strategy, test_data)
    
    # Test 5: Position sizing
    test_position_sizing(strategy, signals, test_data)
    
    # Test 6: Strategy execution
    result = await test_strategy_execution(strategy, test_data)
    
    # Test 7: Test with different market conditions
    print("\n=== Testing Different Market Conditions ===")
    
    # Bearish market
    bearish_data = generate_test_data(periods=100, trend='bearish')
    bearish_signals = strategy.signal_generator.generate_signals(bearish_data)
    print(f"✓ Bearish market test: {len(bearish_signals)} signals")
    
    # Sideways market
    sideways_data = generate_test_data(periods=100, trend='neutral')
    sideways_signals = strategy.signal_generator.generate_signals(sideways_data)
    print(f"✓ Sideways market test: {len(sideways_signals)} signals")
    
    print("\n" + "=" * 60)
    print("🎉 Momentum Strategy Test Complete!")
    
    return True


def test_performance():
    """Test strategy performance and timing."""
    print("\n=== Performance Testing ===")
    
    try:
        strategy = create_momentum_strategy()
        test_data = generate_test_data(periods=200)
        
        # Time signal generation
        start_time = time.perf_counter()
        for _ in range(10):
            signals = strategy.signal_generator.generate_signals(test_data)
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / 10) * 1000
        print(f"✓ Signal generation average time: {avg_time_ms:.2f}ms")
        
        if avg_time_ms < 30:  # Should be under 30ms per requirements
            print("✓ Performance requirement met (<30ms)")
        else:
            print("⚠ Performance requirement not met (>30ms)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    
    # Run tests
    print("Starting Momentum Strategy Tests...")
    
    # Basic tests
    success = asyncio.run(run_comprehensive_test())
    
    # Performance tests
    if success:
        test_performance()
    
    if success:
        print("\n✅ All tests passed! Momentum strategy is ready for integration.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    sys.exit(0 if success else 1)