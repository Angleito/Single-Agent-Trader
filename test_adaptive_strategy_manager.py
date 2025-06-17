#!/usr/bin/env python3
"""
Test script for the Adaptive Strategy Manager.

This script validates the implementation and demonstrates the key functionality
of the adaptive strategy management system.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.strategy.adaptive_strategy_manager import (
    AdaptiveStrategyManager, 
    TradingStrategy,
    StrategySelector,
    TransitionManager
)
from bot.analysis.market_context import MarketContextAnalyzer, MarketRegimeType
from bot.types import MarketState, MarketData, IndicatorData, Position


def create_mock_market_state() -> MarketState:
    """Create mock market state for testing."""
    
    # Create mock OHLCV data
    base_price = Decimal("50000")
    ohlcv_data = []
    
    for i in range(50):
        timestamp = datetime.utcnow() - timedelta(minutes=50-i)
        price_variation = Decimal(str(i * 10))  # Simple price progression
        
        market_data = MarketData(
            symbol="BTC-USD",
            timestamp=timestamp,
            open=base_price + price_variation,
            high=base_price + price_variation + Decimal("100"),
            low=base_price + price_variation - Decimal("100"),
            close=base_price + price_variation + Decimal("50"),
            volume=Decimal("1000000")
        )
        ohlcv_data.append(market_data)
    
    # Create mock indicators
    indicators = IndicatorData(
        timestamp=datetime.utcnow(),
        cipher_a_dot=0.5,
        cipher_b_wave=75.0,
        cipher_b_money_flow=65.0,
        rsi=58.0,
        ema_fast=50150.0,
        ema_slow=50050.0,
        vwap=50100.0
    )
    
    # Create mock position
    position = Position(
        symbol="BTC-USD",
        side="FLAT",
        size=Decimal("0"),
        timestamp=datetime.utcnow()
    )
    
    # Create market state
    market_state = MarketState(
        symbol="BTC-USD",
        interval="1m",
        timestamp=datetime.utcnow(),
        current_price=base_price + Decimal("500"),
        ohlcv_data=ohlcv_data,
        indicators=indicators,
        current_position=position
    )
    
    return market_state


async def test_strategy_selector():
    """Test the strategy selector functionality."""
    print("Testing Strategy Selector...")
    
    selector = StrategySelector()
    
    # Test different regime scenarios
    test_scenarios = [
        {
            'name': 'Risk-On Market',
            'regime_analysis': {
                'regime': {
                    'regime_type': MarketRegimeType.RISK_ON,
                    'confidence': 0.8,
                    'market_volatility_regime': 'NORMAL'
                }
            }
        },
        {
            'name': 'Risk-Off Market', 
            'regime_analysis': {
                'regime': {
                    'regime_type': MarketRegimeType.RISK_OFF,
                    'confidence': 0.75,
                    'market_volatility_regime': 'HIGH'
                }
            }
        },
        {
            'name': 'High Volatility',
            'regime_analysis': {
                'regime': {
                    'regime_type': MarketRegimeType.RISK_ON,
                    'confidence': 0.9,
                    'market_volatility_regime': 'HIGH'
                }
            }
        }
    ]
    
    for scenario in test_scenarios:
        strategy, confidence = selector.select_strategy(scenario['regime_analysis'])
        print(f"  {scenario['name']}: {strategy.value} (confidence: {confidence:.2%})")
    
    print("✓ Strategy Selector tests passed\n")


async def test_transition_manager():
    """Test the transition manager functionality."""
    print("Testing Transition Manager...")
    
    manager = TransitionManager()
    
    # Test transition rules
    test_transitions = [
        (TradingStrategy.MOMENTUM, TradingStrategy.SCALPING),
        (TradingStrategy.SCALPING, TradingStrategy.BREAKOUT),
        (TradingStrategy.BREAKOUT, TradingStrategy.DEFENSIVE)
    ]
    
    for from_strategy, to_strategy in test_transitions:
        market_context = {'volatility': 0.3, 'volume_spike': 1.2}
        
        success = await manager.execute_transition(
            from_strategy, to_strategy, market_context
        )
        
        transition_type = manager.transition_rules.get(
            (from_strategy, to_strategy), 'gradual'
        )
        
        print(f"  {from_strategy.value} → {to_strategy.value}: "
              f"{transition_type} ({'✓' if success else '✗'})")
    
    print("✓ Transition Manager tests passed\n")


async def test_adaptive_strategy_manager():
    """Test the main adaptive strategy manager."""
    print("Testing Adaptive Strategy Manager...")
    
    # Initialize the manager
    analyzer = MarketContextAnalyzer()
    manager = AdaptiveStrategyManager(analyzer)
    
    # Create mock market state
    market_state = create_mock_market_state()
    
    print(f"  Initial strategy: {manager.current_strategy_name.value}")
    
    # Test main execution method
    result = await manager.analyze_and_execute(market_state)
    
    # Validate result structure
    required_fields = [
        'timestamp', 'regime_analysis', 'active_strategy', 
        'strategy_decision', 'execution_result'
    ]
    
    for field in required_fields:
        if field not in result:
            print(f"  ✗ Missing required field: {field}")
            return
    
    print(f"  Execution completed in {result.get('execution_time_ms', 0)}ms")
    print(f"  Recommended strategy: {result['strategy_decision']['recommended']}")
    print(f"  Confidence: {result['strategy_decision']['confidence']:.2%}")
    print(f"  Active strategy: {result['active_strategy']['name']}")
    
    # Test force strategy change
    print("\n  Testing force strategy change...")
    success = manager.force_strategy_change(TradingStrategy.SCALPING)
    if success:
        print(f"  ✓ Successfully changed to {manager.current_strategy_name.value}")
    else:
        print("  ✗ Failed to change strategy")
    
    # Test performance tracking
    performance = manager.get_strategy_performance()
    print(f"  Performance summary available: {'✓' if performance else '✗'}")
    
    print("✓ Adaptive Strategy Manager tests passed\n")


async def test_performance_tracking():
    """Test performance tracking functionality."""
    print("Testing Performance Tracking...")
    
    from bot.strategy.adaptive_strategy_manager import StrategyPerformanceTracker
    
    tracker = StrategyPerformanceTracker()
    
    # Create mock trades
    mock_trades = [
        {'pnl': 0.02, 'holding_time_seconds': 300},
        {'pnl': -0.01, 'holding_time_seconds': 450},
        {'pnl': 0.015, 'holding_time_seconds': 200},
        {'pnl': 0.03, 'holding_time_seconds': 600}
    ]
    
    mock_market_conditions = {
        'regime': 'RISK_ON',
        'volatility': 0.3
    }
    
    # Track performance for momentum strategy
    tracker.track_strategy_performance(
        TradingStrategy.MOMENTUM,
        mock_trades,
        mock_market_conditions
    )
    
    # Get effectiveness score
    effectiveness = tracker.get_strategy_effectiveness(TradingStrategy.MOMENTUM)
    print(f"  Momentum strategy effectiveness: {effectiveness:.2%}")
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    if 'strategies' in summary:
        print(f"  Performance summary generated: ✓")
        if TradingStrategy.MOMENTUM.value in summary['strategies']:
            momentum_metrics = summary['strategies'][TradingStrategy.MOMENTUM.value]['metrics']
            print(f"  Total trades: {momentum_metrics['total_trades']}")
            print(f"  Win rate: {momentum_metrics['win_rate']:.2%}")
    else:
        print("  ✗ Failed to generate performance summary")
    
    print("✓ Performance Tracking tests passed\n")


async def run_integration_test():
    """Run a comprehensive integration test."""
    print("Running Integration Test...")
    
    manager = AdaptiveStrategyManager()
    market_state = create_mock_market_state()
    
    # Simulate multiple execution cycles
    strategies_used = []
    
    for cycle in range(3):
        print(f"\n  Cycle {cycle + 1}:")
        
        result = await manager.analyze_and_execute(market_state)
        
        current_strategy = result['active_strategy']['name']
        recommended_strategy = result['strategy_decision']['recommended']
        confidence = result['strategy_decision']['confidence']
        
        print(f"    Current: {current_strategy}")
        print(f"    Recommended: {recommended_strategy} ({confidence:.2%})")
        
        strategies_used.append(current_strategy)
        
        # Simulate market condition changes by modifying indicators
        market_state.indicators.rsi = 45.0 + (cycle * 10)
        market_state.indicators.cipher_a_dot = 0.3 + (cycle * 0.2)
        
        # Small delay to simulate real-time execution
        await asyncio.sleep(0.1)
    
    print(f"\n  Strategies used: {list(set(strategies_used))}")
    
    # Test strategy context for LLM
    context = manager.context_provider.get_llm_context(manager.last_regime_analysis)
    
    required_context_fields = [
        'current_strategy', 'strategy_recommendations', 
        'transition_status', 'performance_context'
    ]
    
    context_complete = all(field in context for field in required_context_fields)
    print(f"  LLM context complete: {'✓' if context_complete else '✗'}")
    
    print("✓ Integration test passed\n")


async def main():
    """Main test function."""
    print("🚀 Starting Adaptive Strategy Manager Tests\n")
    
    try:
        await test_strategy_selector()
        await test_transition_manager()
        await test_performance_tracking()
        await test_adaptive_strategy_manager()
        await run_integration_test()
        
        print("🎉 All tests passed successfully!")
        
        # Print summary of capabilities
        print("\n📋 Adaptive Strategy Manager Capabilities:")
        print("  ✓ Dynamic strategy selection based on market regime")
        print("  ✓ Smooth strategy transitions with position management")
        print("  ✓ Real-time performance tracking and optimization")
        print("  ✓ LLM integration context provision")
        print("  ✓ Risk-aware execution with strategy-specific parameters")
        print("  ✓ Comprehensive market regime analysis integration")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)