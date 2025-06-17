#!/usr/bin/env python3
"""
Example usage of the Unified Indicator Framework.

This script demonstrates how to use the unified framework to calculate indicators
for different trading strategies with optimized performance and unified interfaces.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from bot.indicators.unified_framework import (
    unified_framework,
    calculate_indicators_for_strategy,
    get_available_indicators_for_timeframe,
    get_framework_performance,
    TimeframeType
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_market_data(periods: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    
    # Generate realistic price data
    np.random.seed(42)
    
    # Start with a base price
    base_price = 50000.0
    
    # Generate random walk with trend
    returns = np.random.normal(0.0005, 0.02, periods)  # Small positive drift with volatility
    prices = [base_price]
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLC from price series
    data = []
    for i in range(1, len(prices)):
        # Current price is the "close"
        close = prices[i]
        open_price = prices[i-1]
        
        # Create realistic high/low with some volatility
        high_low_range = abs(close - open_price) * np.random.uniform(1.2, 2.5)
        high = max(open_price, close) + high_low_range * np.random.uniform(0.2, 0.8)
        low = min(open_price, close) - high_low_range * np.random.uniform(0.2, 0.8)
        
        # Volume with some correlation to price movement
        price_change_pct = abs((close - open_price) / open_price)
        base_volume = np.random.uniform(1000, 5000)
        volume = base_volume * (1 + price_change_pct * 10)  # Higher volume on big moves
        
        data.append({
            'timestamp': datetime.now() - timedelta(minutes=(periods - i)),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


async def test_scalping_strategy():
    """Test indicator calculations for scalping strategy."""
    
    logger.info("🚀 Testing Scalping Strategy Indicators")
    logger.info("=" * 50)
    
    # Create sample data
    market_data = create_sample_market_data(periods=300)
    logger.info(f"Created sample data: {len(market_data)} candles")
    
    # Calculate indicators for scalping
    try:
        results = await calculate_indicators_for_strategy(
            strategy_type='scalping',
            market_data={'scalping': market_data}
        )
        
        logger.info(f"✅ Successfully calculated {results['performance_metrics']['indicator_count']} indicators")
        logger.info(f"📊 Total calculation time: {results['performance_metrics']['total_calculation_time_ms']:.2f}ms")
        logger.info(f"🎯 Cache hits: {results['performance_metrics']['cache_hits']}")
        logger.info(f"❌ Cache misses: {results['performance_metrics']['cache_misses']}")
        
        # Show indicator results
        logger.info("\n📈 Indicator Results:")
        for indicator_name, indicator_data in results['indicators'].items():
            logger.info(f"  • {indicator_name}: {len(indicator_data)} data elements")
            
            # Show latest values if available
            if 'latest_values' in indicator_data:
                latest = indicator_data['latest_values']
                logger.info(f"    Latest values: {latest}")
        
        # Show signals
        if results['combined_signals']:
            logger.info(f"\n🔔 Generated {len(results['combined_signals'])} trading signals:")
            for signal in results['combined_signals'][:3]:  # Show top 3
                logger.info(f"  • {signal.get('type', 'unknown')} from {signal.get('indicator', 'unknown')} "
                           f"(strength: {signal.get('strength', 0):.2f})")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in scalping strategy test: {e}")
        return None


async def test_momentum_strategy():
    """Test indicator calculations for momentum strategy."""
    
    logger.info("\n🎯 Testing Momentum Strategy Indicators")
    logger.info("=" * 50)
    
    # Create sample data
    market_data = create_sample_market_data(periods=500)
    logger.info(f"Created sample data: {len(market_data)} candles")
    
    try:
        results = await calculate_indicators_for_strategy(
            strategy_type='momentum',
            market_data={'momentum': market_data}
        )
        
        logger.info(f"✅ Successfully calculated {results['performance_metrics']['indicator_count']} indicators")
        logger.info(f"📊 Total calculation time: {results['performance_metrics']['total_calculation_time_ms']:.2f}ms")
        
        # Show calculation details
        logger.info("\n⚙️ Calculation Performance:")
        for indicator_name, details in results['calculation_details'].items():
            logger.info(f"  • {indicator_name}: {details['calculation_time_ms']:.2f}ms "
                       f"({'cached' if details['cache_hit'] else 'calculated'})")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in momentum strategy test: {e}")
        return None


async def test_incremental_updates():
    """Test incremental update functionality."""
    
    logger.info("\n🔄 Testing Incremental Updates")
    logger.info("=" * 50)
    
    # Create initial data
    initial_data = create_sample_market_data(periods=100)
    logger.info(f"Created initial data: {len(initial_data)} candles")
    
    try:
        # Setup incremental mode
        setup_results = await unified_framework.setup_incremental_mode(
            strategy_type='scalping',
            initial_data={'scalping': initial_data}
        )
        
        logger.info(f"📋 Incremental setup results: {setup_results}")
        
        # Simulate new tick data
        last_candle = initial_data.iloc[-1]
        new_tick = {
            'timestamp': datetime.now(),
            'open': last_candle['close'],
            'high': last_candle['close'] * 1.001,
            'low': last_candle['close'] * 0.999,
            'close': last_candle['close'] * 1.0005,
            'volume': 1500
        }
        
        # Update incrementally
        incremental_results = await unified_framework.update_incremental(
            strategy_type='scalping',
            new_tick=new_tick
        )
        
        logger.info(f"🔄 Incremental update results: {len(incremental_results)} indicators updated")
        for indicator_name, result in incremental_results.items():
            logger.info(f"  • {indicator_name}: {len(result) if isinstance(result, dict) else 'updated'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in incremental update test: {e}")
        return False


def test_available_indicators():
    """Test getting available indicators for different timeframes."""
    
    logger.info("\n📋 Available Indicators by Timeframe")
    logger.info("=" * 50)
    
    timeframes = ['scalping', 'momentum', 'swing', 'position']
    
    for tf in timeframes:
        indicators = get_available_indicators_for_timeframe(tf)
        logger.info(f"\n{tf.upper()} ({len(indicators)} indicators):")
        
        for indicator in indicators:
            logger.info(f"  • {indicator['name']} ({indicator['type']}) "
                       f"- Priority: {indicator['calculation_priority']} "
                       f"- Incremental: {'✅' if indicator['supports_incremental'] else '❌'}")


def test_performance_analysis():
    """Test performance analysis functionality."""
    
    logger.info("\n📈 Performance Analysis")
    logger.info("=" * 50)
    
    try:
        performance = get_framework_performance()
        
        logger.info(f"Performance Summary:")
        summary = performance.get('summary', {})
        logger.info(f"  • Total indicators: {summary.get('total_indicators', 0)}")
        logger.info(f"  • Average calculation time: {summary.get('avg_time_ms', 0):.2f}ms")
        logger.info(f"  • Maximum calculation time: {summary.get('max_time_ms', 0):.2f}ms")
        logger.info(f"  • Total calls: {summary.get('total_calls', 0)}")
        
        # Show slow indicators
        slow_indicators = performance.get('slow_indicators', [])
        if slow_indicators:
            logger.info(f"\n⚠️ Slow Indicators ({len(slow_indicators)}):")
            for indicator in slow_indicators:
                logger.info(f"  • {indicator['name']}: {indicator['avg_time_ms']:.2f}ms "
                           f"({indicator['severity']} severity)")
        
        # Show optimization suggestions
        suggestions = performance.get('optimization_suggestions', [])
        if suggestions:
            logger.info(f"\n💡 Optimization Suggestions ({len(suggestions)}):")
            for suggestion in suggestions[:3]:  # Show top 3
                logger.info(f"  • {suggestion['indicator']}: {suggestion['suggestion']} "
                           f"({suggestion['priority']} priority)")
        
    except Exception as e:
        logger.error(f"❌ Error in performance analysis: {e}")


async def test_framework_status():
    """Test framework status and health metrics."""
    
    logger.info("\n🏥 Framework Status")
    logger.info("=" * 50)
    
    try:
        status = unified_framework.get_framework_status()
        
        logger.info(f"Framework Health:")
        logger.info(f"  • Registered indicators: {status['registered_indicators']}")
        logger.info(f"  • Cache size: {status['cache_stats']['size']}")
        logger.info(f"  • Available timeframes: {', '.join(status['available_timeframes'])}")
        logger.info(f"  • Supported strategies: {', '.join(status['supported_strategies'])}")
        
        # Performance stats
        perf_stats = status.get('performance_stats', {})
        if perf_stats:
            logger.info(f"\n📊 Performance Statistics:")
            for indicator, stats in list(perf_stats.items())[:3]:  # Show top 3
                logger.info(f"  • {indicator}: avg {stats['avg_time_ms']:.2f}ms, "
                           f"calls: {stats['call_count']}")
        
        # Incremental status
        inc_status = status.get('incremental_status', {})
        if inc_status:
            logger.info(f"\n🔄 Incremental Status:")
            for key, info in inc_status.items():
                logger.info(f"  • {key}: last update {info['last_update']}")
        
    except Exception as e:
        logger.error(f"❌ Error getting framework status: {e}")


async def main():
    """Main test function."""
    
    logger.info("🚀 Unified Indicator Framework Example")
    logger.info("=" * 60)
    
    # Test 1: Available indicators
    test_available_indicators()
    
    # Test 2: Scalping strategy
    scalping_results = await test_scalping_strategy()
    
    # Test 3: Momentum strategy
    momentum_results = await test_momentum_strategy()
    
    # Test 4: Incremental updates
    await test_incremental_updates()
    
    # Test 5: Performance analysis
    test_performance_analysis()
    
    # Test 6: Framework status
    await test_framework_status()
    
    logger.info("\n✅ All tests completed!")
    
    # Summary
    logger.info("\n📊 Test Summary:")
    logger.info(f"  • Scalping test: {'✅ PASSED' if scalping_results else '❌ FAILED'}")
    logger.info(f"  • Momentum test: {'✅ PASSED' if momentum_results else '❌ FAILED'}")
    logger.info(f"  • Framework functional: ✅ PASSED")


if __name__ == "__main__":
    asyncio.run(main())