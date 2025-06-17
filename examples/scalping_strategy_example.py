#!/usr/bin/env python3
"""
Example usage of the ScalpingStrategy for high-frequency trading in ranging markets.

This example demonstrates how to:
1. Configure the scalping strategy
2. Process market data for signal generation  
3. Handle different types of scalping signals
4. Integrate with risk management
5. Track performance metrics

The scalping strategy is optimized for:
- Low-volume ranging markets
- 15-second timeframe data
- Quick entries and exits (5 minutes or less)
- Tight risk management (2-8 basis points targets)
"""

import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# This would normally import from the bot module
# from bot.strategy.scalping_strategy import (
#     ScalpingStrategy, 
#     ScalpingConfig, 
#     create_scalping_strategy
# )

def generate_sample_market_data(periods: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    
    # Start with a base price
    base_price = 50000.0
    
    # Generate realistic price movements for ranging market
    np.random.seed(42)  # For reproducible results
    
    # Create ranging price action with noise
    trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 200  # Range-bound movement
    noise = np.random.normal(0, 50, periods)  # Random noise
    
    prices = base_price + trend + noise
    
    # Generate OHLCV data
    data = []
    for i in range(periods):
        open_price = prices[i]
        close_price = prices[i] + np.random.normal(0, 10)
        
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 20))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 20))
        
        volume = max(1000, int(np.random.normal(5000, 1000)))
        
        timestamp = datetime.now() - timedelta(seconds=(periods-i) * 15)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

async def scalping_strategy_example():
    """Demonstrate scalping strategy usage."""
    
    print("Scalping Strategy Example")
    print("=" * 50)
    
    # 1. Create custom configuration for aggressive scalping
    custom_config = {
        'min_profit_target_pct': 0.02,  # 2 basis points minimum
        'max_profit_target_pct': 0.06,  # 6 basis points maximum  
        'stop_loss_pct': 0.015,         # 1.5 basis points stop loss
        'max_holding_time': 180,        # 3 minutes maximum
        'max_daily_trades': 100,        # 100 trades per day max
        'min_signal_strength': 0.65,    # Higher confidence threshold
        'base_position_pct': 0.3,       # Smaller position sizes
        'max_consecutive_losses': 2     # Stricter loss control
    }
    
    print(f"Configuration: {custom_config}")
    
    # 2. Create scalping strategy instance
    strategy = create_scalping_strategy(custom_config)
    print("✓ Scalping strategy created")
    
    # 3. Generate sample market data
    market_df = generate_sample_market_data(100)
    current_price = float(market_df['close'].iloc[-1])
    
    print(f"✓ Generated {len(market_df)} data points")
    print(f"Current price: ${current_price:.2f}")
    
    # 4. Prepare market data for strategy
    market_data = {
        'ohlcv': market_df,
        'current_price': current_price,
        'timestamp': time.time(),
        'account_balance': 10000.0  # $10K account
    }
    
    # 5. Run strategy analysis
    print("\nRunning strategy analysis...")
    result = await strategy.analyze_and_signal(market_data)
    
    # 6. Display results
    print(f"\nStrategy Results:")
    print(f"Action: {result['action']}")
    print(f"Position Size: {result['size_pct']}%")
    print(f"Take Profit: {result['take_profit_pct']:.3f}%")
    print(f"Stop Loss: {result['stop_loss_pct']:.3f}%")
    print(f"Rationale: {result['rationale']}")
    
    # 7. Display signals found
    signals = result.get('signals', [])
    print(f"\nSignals Generated: {len(signals)}")
    
    for i, signal in enumerate(signals[:3]):  # Show top 3 signals
        print(f"\nSignal {i+1}:")
        print(f"  Type: {signal['type']}")
        print(f"  Direction: {signal['direction']}")
        print(f"  Confidence: {signal['confidence']:.2f}")
        print(f"  Timing: {signal['timing']}")
        print(f"  Target: {signal['target_profit_pct']:.3f}%")
        print(f"  Reasons: {', '.join(signal['entry_reasons'][:2])}")
        
        if signal['risk_factors']:
            print(f"  Risk Factors: {', '.join(signal['risk_factors'])}")
    
    # 8. Display market analysis
    analysis = result.get('market_analysis', {})
    print(f"\nMarket Analysis:")
    print(f"  Analysis Time: {analysis.get('analysis_time_ms', 0):.2f}ms")
    print(f"  Strategy State: {analysis.get('strategy_state', 'unknown')}")
    print(f"  Signal Count: {analysis.get('signal_count', 0)}")
    print(f"  Approved Signals: {analysis.get('approved_signal_count', 0)}")
    
    # 9. Display risk assessment
    risk = result.get('risk_assessment', {})
    print(f"\nRisk Assessment:")
    print(f"  Risk Level: {risk.get('risk_level', 'unknown')}")
    print(f"  Daily Trades: {risk.get('daily_trades', 0)}")
    print(f"  Consecutive Losses: {risk.get('consecutive_losses', 0)}")
    print(f"  Cooldown Active: {risk.get('cooldown_active', False)}")
    
    # 10. Simulate trade execution and result
    if result['action'] != 'HOLD':
        print(f"\n--- Simulating Trade Execution ---")
        
        # Simulate a trade result
        simulated_profit = np.random.normal(0.001, 0.002)  # Random outcome
        trade_result = {
            'symbol': 'BTC-USD',
            'side': result['action'],
            'profit_loss': simulated_profit,
            'holding_time_seconds': np.random.randint(30, 180),
            'signal_type': signals[0]['type'] if signals else 'unknown',
            'timestamp': time.time()
        }
        
        # Update strategy with trade result
        strategy.update_trade_result(trade_result)
        
        print(f"Trade Result: {trade_result['profit_loss']:.4f} ({trade_result['profit_loss']*100:.2f}%)")
        print(f"Holding Time: {trade_result['holding_time_seconds']} seconds")
    
    # 11. Get strategy status and performance
    status = strategy.get_strategy_status()
    print(f"\nStrategy Status:")
    print(f"  State: {status['state']}")
    print(f"  Active Positions: {status['active_positions']}")
    
    perf = status.get('performance_metrics', {})
    print(f"\nPerformance Metrics:")
    print(f"  Total Trades: {perf.get('total_trades', 0)}")
    print(f"  Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
    print(f"  Avg Profit/Trade: {perf.get('avg_profit_per_trade', 0)*100:.3f}%")
    print(f"  Signals Generated: {perf.get('signals_generated', 0)}")
    print(f"  Execution Rate: {perf.get('execution_rate', 0)*100:.1f}%")
    print(f"  Scalping Efficiency: {perf.get('scalping_efficiency', 0)*100:.1f}%")

def demonstrate_signal_types():
    """Demonstrate different types of scalping signals."""
    
    print("\n" + "=" * 50)
    print("Scalping Signal Types")
    print("=" * 50)
    
    signal_descriptions = {
        'mean_reversion': """
        Mean Reversion Signals:
        - Triggered when price deviates significantly from VWAP
        - RSI/Williams %R at extreme levels (oversold/overbought)
        - Price near support/resistance with reversal momentum
        - Target: Quick return to mean price level
        """,
        
        'micro_breakout': """
        Micro Breakout Signals:
        - Small range breakouts with volume confirmation
        - Price breaks above resistance or below support
        - Requires 1.5x+ volume and momentum alignment
        - Target: Continuation of breakout move
        """,
        
        'vwap_bounce': """
        VWAP Bounce Signals:
        - Price bounces off VWAP or VWAP bands
        - Volume-weighted price levels act as support/resistance
        - Look for rejection or bounce patterns
        - Target: Move to opposite VWAP band
        """,
        
        'support_resistance': """
        Support/Resistance Signals:
        - Price touches identified support/resistance levels
        - Based on recent swing highs/lows with volume
        - Momentum alignment confirmation required
        - Target: Bounce back from level
        """,
        
        'momentum_spike': """
        Momentum Spike Signals:
        - Sudden price moves with high volume
        - 20+ basis point moves in single period
        - Volume 1.5x+ average with momentum alignment
        - Target: Quick profit from momentum continuation
        """,
        
        'volume_anomaly': """
        Volume Anomaly Signals:
        - Unusual volume patterns (2x+ average)
        - High volume with price movement
        - May indicate institutional activity
        - Target: Capitalize on volume-driven moves
        """
    }
    
    for signal_type, description in signal_descriptions.items():
        print(f"{signal_type.upper()}:")
        print(description)

def performance_optimization_tips():
    """Show performance optimization guidelines."""
    
    print("\n" + "=" * 50)
    print("Performance Optimization Tips")
    print("=" * 50)
    
    tips = [
        "1. Data Quality: Ensure clean, high-frequency OHLCV data",
        "2. Latency: Target <10ms total analysis time for live trading",
        "3. Position Sizing: Start with smaller sizes (0.1-0.5% account)",
        "4. Risk Management: Strict stop losses and daily trade limits",
        "5. Market Conditions: Works best in ranging/low-volatility periods",
        "6. Timeframes: Optimized for 15-second data, max 5-minute holds",
        "7. Volume Requirements: Prefer markets with consistent volume",
        "8. Spread Monitoring: Avoid wide spreads (>2 basis points)",
        "9. News Events: Pause during high-impact news releases",
        "10. Continuous Monitoring: Review performance daily and adjust"
    ]
    
    for tip in tips:
        print(f"  {tip}")
    
    print(f"\nKey Metrics to Monitor:")
    print(f"  - Win Rate: Target >60%")
    print(f"  - Avg Profit/Trade: Target >0.1% (10 basis points)")
    print(f"  - Max Drawdown: Keep under 2%")
    print(f"  - Sharpe Ratio: Target >2.0")
    print(f"  - Trades/Hour: 5-20 depending on market conditions")

if __name__ == '__main__':
    # Run the example
    asyncio.run(scalping_strategy_example())
    
    # Show additional information
    demonstrate_signal_types()
    performance_optimization_tips()
    
    print(f"\n" + "=" * 50)
    print("Example completed successfully!")
    print("Ready to integrate with adaptive strategy manager.")
    print("=" * 50)