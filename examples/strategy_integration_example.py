#!/usr/bin/env python3
"""
Example integration of the Unified Indicator Framework with strategy management.

This example demonstrates how to integrate the unified framework with existing
strategy managers and trading agents for optimized multi-timeframe analysis.
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMarketDataProvider:
    """Mock market data provider for testing."""
    
    def __init__(self):
        self.current_price = 50000.0
        
    def generate_realistic_data(self, periods: int, timeframe: str = "1m") -> pd.DataFrame:
        """Generate realistic OHLCV data."""
        np.random.seed(42)
        
        # Adjust volatility based on timeframe
        volatility = {
            "15s": 0.001,  # Lower volatility for shorter timeframes
            "1m": 0.002,
            "5m": 0.005,
            "15m": 0.008
        }.get(timeframe, 0.002)
        
        # Generate price series with realistic patterns
        returns = np.random.normal(0.0001, volatility, periods)
        prices = [self.current_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create OHLC data
        data = []
        for i in range(1, len(prices)):
            close = prices[i]
            open_price = prices[i-1]
            
            # Realistic high/low with intrabar movement
            range_size = abs(close - open_price) * np.random.uniform(1.5, 3.0)
            high = max(open_price, close) + range_size * np.random.uniform(0.3, 0.7)
            low = min(open_price, close) - range_size * np.random.uniform(0.3, 0.7)
            
            # Volume with correlation to price movement
            price_change_pct = abs((close - open_price) / open_price)
            base_volume = np.random.uniform(500, 2000)
            volume = base_volume * (1 + price_change_pct * 15)
            
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


class EnhancedStrategyManager:
    """Enhanced strategy manager using the unified indicator framework."""
    
    def __init__(self):
        self.market_data_provider = MockMarketDataProvider()
        self.current_strategy = "momentum"
        self.performance_history = []
        
    async def analyze_market_conditions(self, market_data: dict) -> dict:
        """Analyze market conditions using the unified framework."""
        
        try:
            # Import framework functions (would normally be at top level)
            from bot.indicators.unified_framework import (
                calculate_indicators_for_strategy,
                get_framework_performance
            )
            
            logger.info(f"🔍 Analyzing market conditions for {self.current_strategy} strategy")
            
            # Calculate indicators using unified framework
            start_time = datetime.now()
            
            results = await calculate_indicators_for_strategy(
                strategy_type=self.current_strategy,
                market_data=market_data
            )
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract key information
            analysis = {
                'strategy_type': results['strategy_type'],
                'timeframe': results['timeframe'],
                'calculation_time_ms': calculation_time,
                'performance_metrics': results['performance_metrics'],
                'market_signals': self._process_signals(results['combined_signals']),
                'indicator_summary': self._summarize_indicators(results['indicators']),
                'market_regime': self._detect_market_regime(results['indicators']),
                'trading_recommendations': self._generate_recommendations(results)
            }
            
            # Track performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'calculation_time_ms': calculation_time,
                'cache_hit_rate': results['performance_metrics']['cache_hits'] / 
                                 (results['performance_metrics']['cache_hits'] + results['performance_metrics']['cache_misses'])
                                 if (results['performance_metrics']['cache_hits'] + results['performance_metrics']['cache_misses']) > 0 else 0
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in market analysis: {e}")
            return {'error': str(e), 'strategy_type': self.current_strategy}
    
    def _process_signals(self, signals: list) -> dict:
        """Process and categorize trading signals."""
        
        if not signals:
            return {'total_signals': 0, 'strong_signals': 0, 'signal_consensus': 'neutral'}
        
        # Categorize signals by strength
        strong_signals = [s for s in signals if s.get('strength', 0) > 0.7]
        medium_signals = [s for s in signals if 0.4 <= s.get('strength', 0) <= 0.7]
        weak_signals = [s for s in signals if s.get('strength', 0) < 0.4]
        
        # Determine consensus
        buy_signals = len([s for s in signals if 'buy' in s.get('type', '').lower()])
        sell_signals = len([s for s in signals if 'sell' in s.get('type', '').lower()])
        
        if buy_signals > sell_signals * 1.5:
            consensus = 'bullish'
        elif sell_signals > buy_signals * 1.5:
            consensus = 'bearish'
        else:
            consensus = 'neutral'
        
        return {
            'total_signals': len(signals),
            'strong_signals': len(strong_signals),
            'medium_signals': len(medium_signals),
            'weak_signals': len(weak_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_consensus': consensus,
            'top_signals': signals[:3]  # Top 3 signals
        }
    
    def _summarize_indicators(self, indicators: dict) -> dict:
        """Summarize key indicator values."""
        
        summary = {}
        
        # VuManChu Cipher A
        if 'vumanchu_cipher_a' in indicators:
            vumanchu = indicators['vumanchu_cipher_a']
            latest_values = vumanchu.get('latest_values', {})
            
            summary['vumanchu'] = {
                'wt1': latest_values.get('wt1', 0),
                'wt2': latest_values.get('wt2', 0),
                'overbought_level': vumanchu.get('overbought_level', 60),
                'oversold_level': vumanchu.get('oversold_level', -60),
                'signal_state': self._get_vumanchu_state(latest_values)
            }
        
        # Fast EMA
        if 'fast_ema' in indicators:
            ema = indicators['fast_ema']
            
            summary['ema'] = {
                'trend_strength': ema.get('trend_strength', 0),
                'latest_values': ema.get('latest_values', {}),
                'trend_direction': 'bullish' if ema.get('trend_strength', 0) > 0.3 else 
                                 'bearish' if ema.get('trend_strength', 0) < -0.3 else 'neutral'
            }
        
        # Momentum indicators
        if 'scalping_momentum' in indicators:
            momentum = indicators['scalping_momentum']
            latest_values = momentum.get('latest_values', {})
            
            summary['momentum'] = {
                'rsi': latest_values.get('rsi', 50),
                'macd': latest_values.get('macd', 0),
                'williams_r': latest_values.get('williams_r', -50),
                'momentum_state': self._get_momentum_state(latest_values, momentum.get('thresholds', {}))
            }
        
        # Volume indicators
        if 'scalping_volume' in indicators:
            volume = indicators['scalping_volume']
            
            summary['volume'] = {
                'vwap': volume.get('latest_values', {}).get('vwap', 0),
                'volume_ratio': volume.get('latest_values', {}).get('volume_ratio', 1),
                'volume_trend': volume.get('volume_analysis', {}).get('volume_trend', 'neutral'),
                'high_volume': volume.get('latest_values', {}).get('volume_ratio', 1) > 1.5
            }
        
        return summary
    
    def _get_vumanchu_state(self, latest_values: dict) -> str:
        """Determine VuManChu signal state."""
        wt1 = latest_values.get('wt1', 0)
        wt2 = latest_values.get('wt2', 0)
        
        if wt1 > 60 and wt2 > 60:
            return 'overbought'
        elif wt1 < -60 and wt2 < -60:
            return 'oversold'
        elif wt1 > wt2 and wt1 > 0:
            return 'bullish_momentum'
        elif wt1 < wt2 and wt1 < 0:
            return 'bearish_momentum'
        else:
            return 'neutral'
    
    def _get_momentum_state(self, latest_values: dict, thresholds: dict) -> str:
        """Determine overall momentum state."""
        rsi = latest_values.get('rsi', 50)
        macd = latest_values.get('macd', 0)
        williams = latest_values.get('williams_r', -50)
        
        rsi_overbought = thresholds.get('rsi_overbought', 70)
        rsi_oversold = thresholds.get('rsi_oversold', 30)
        
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI analysis
        if rsi > rsi_overbought:
            bearish_signals += 1
        elif rsi < rsi_oversold:
            bullish_signals += 1
        
        # MACD analysis
        if macd > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Williams %R analysis
        if williams > -20:
            bearish_signals += 1
        elif williams < -80:
            bullish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'
    
    def _detect_market_regime(self, indicators: dict) -> dict:
        """Detect current market regime based on indicators."""
        
        # Analyze trend strength
        ema_data = indicators.get('fast_ema', {})
        trend_strength = ema_data.get('trend_strength', 0)
        
        # Analyze volatility (simplified)
        vumanchu_data = indicators.get('vumanchu_cipher_a', {})
        latest_values = vumanchu_data.get('latest_values', {})
        wt1 = abs(latest_values.get('wt1', 0))
        
        # Analyze volume
        volume_data = indicators.get('scalping_volume', {})
        volume_ratio = volume_data.get('latest_values', {}).get('volume_ratio', 1)
        
        # Determine regime
        if abs(trend_strength) > 0.6 and volume_ratio > 1.3:
            regime = 'trending_high_volume'
            confidence = 'high'
        elif abs(trend_strength) > 0.4:
            regime = 'trending_normal_volume'
            confidence = 'medium'
        elif wt1 > 50 and volume_ratio > 1.5:
            regime = 'volatile_high_volume'
            confidence = 'medium'
        elif volume_ratio < 0.7:
            regime = 'low_volume_consolidation'
            confidence = 'medium'
        else:
            regime = 'range_bound'
            confidence = 'low'
        
        return {
            'regime': regime,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'volatility_level': 'high' if wt1 > 40 else 'medium' if wt1 > 20 else 'low',
            'volume_level': 'high' if volume_ratio > 1.5 else 'normal' if volume_ratio > 0.8 else 'low'
        }
    
    def _generate_recommendations(self, results: dict) -> dict:
        """Generate trading recommendations based on analysis."""
        
        signals = results.get('combined_signals', [])
        indicators = results.get('indicators', {})
        
        # Count signal types
        signal_analysis = self._process_signals(signals)
        
        # Get indicator summary
        indicator_summary = self._summarize_indicators(indicators)
        
        # Generate recommendations
        recommendations = []
        confidence_score = 0
        
        # Signal-based recommendations
        if signal_analysis['signal_consensus'] == 'bullish' and signal_analysis['strong_signals'] >= 2:
            recommendations.append("Consider long position - Strong bullish signals detected")
            confidence_score += 30
        elif signal_analysis['signal_consensus'] == 'bearish' and signal_analysis['strong_signals'] >= 2:
            recommendations.append("Consider short position - Strong bearish signals detected")
            confidence_score += 30
        
        # Trend-based recommendations
        ema_data = indicator_summary.get('ema', {})
        if ema_data.get('trend_direction') == 'bullish' and ema_data.get('trend_strength', 0) > 0.5:
            recommendations.append("Strong uptrend confirmed - Favor long positions")
            confidence_score += 25
        elif ema_data.get('trend_direction') == 'bearish' and ema_data.get('trend_strength', 0) < -0.5:
            recommendations.append("Strong downtrend confirmed - Favor short positions")
            confidence_score += 25
        
        # Volume-based recommendations
        volume_data = indicator_summary.get('volume', {})
        if volume_data.get('high_volume') and volume_data.get('volume_trend') != 'neutral':
            recommendations.append(f"High volume {volume_data.get('volume_trend')} - Confirm with price action")
            confidence_score += 15
        
        # Risk management recommendations
        vumanchu_data = indicator_summary.get('vumanchu', {})
        if vumanchu_data.get('signal_state') in ['overbought', 'oversold']:
            recommendations.append(f"Market {vumanchu_data.get('signal_state')} - Consider reversal strategies")
            confidence_score += 10
        
        # Overall recommendation
        if confidence_score >= 60:
            overall = "HIGH CONFIDENCE - Strong trading opportunity"
        elif confidence_score >= 40:
            overall = "MEDIUM CONFIDENCE - Moderate trading opportunity"
        elif confidence_score >= 20:
            overall = "LOW CONFIDENCE - Wait for clearer signals"
        else:
            overall = "NO CLEAR SIGNAL - Maintain neutral position"
        
        return {
            'overall_recommendation': overall,
            'confidence_score': confidence_score,
            'specific_recommendations': recommendations,
            'risk_level': 'high' if confidence_score < 30 else 'medium' if confidence_score < 60 else 'low'
        }
    
    async def switch_strategy(self, new_strategy: str) -> bool:
        """Switch to a different trading strategy."""
        
        valid_strategies = ['scalping', 'momentum', 'swing', 'position']
        
        if new_strategy not in valid_strategies:
            logger.error(f"❌ Invalid strategy: {new_strategy}")
            return False
        
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        
        logger.info(f"🔄 Strategy switched: {old_strategy} → {new_strategy}")
        return True
    
    def get_performance_summary(self) -> dict:
        """Get performance summary of the strategy manager."""
        
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        recent_performance = self.performance_history[-10:]  # Last 10 calculations
        
        avg_calc_time = np.mean([p['calculation_time_ms'] for p in recent_performance])
        avg_cache_hit_rate = np.mean([p['cache_hit_rate'] for p in recent_performance])
        
        return {
            'current_strategy': self.current_strategy,
            'total_analyses': len(self.performance_history),
            'recent_avg_calc_time_ms': avg_calc_time,
            'recent_avg_cache_hit_rate': avg_cache_hit_rate * 100,
            'performance_trend': 'improving' if len(recent_performance) > 5 and 
                               avg_calc_time < np.mean([p['calculation_time_ms'] for p in self.performance_history[-20:-10]])
                               else 'stable'
        }


async def demo_strategy_integration():
    """Demonstrate the enhanced strategy manager in action."""
    
    logger.info("🚀 Enhanced Strategy Manager Demo")
    logger.info("=" * 50)
    
    # Initialize strategy manager
    strategy_manager = EnhancedStrategyManager()
    
    # Test different strategies
    strategies_to_test = ['scalping', 'momentum', 'swing']
    
    for strategy in strategies_to_test:
        logger.info(f"\n📊 Testing {strategy.upper()} Strategy")
        logger.info("-" * 30)
        
        # Switch strategy
        await strategy_manager.switch_strategy(strategy)
        
        # Generate appropriate market data
        periods = {'scalping': 200, 'momentum': 500, 'swing': 1000}[strategy]
        timeframe = {'scalping': '1m', 'momentum': '5m', 'swing': '15m'}[strategy]
        
        market_data = {
            strategy: strategy_manager.market_data_provider.generate_realistic_data(
                periods=periods, timeframe=timeframe
            )
        }
        
        logger.info(f"📈 Generated {len(market_data[strategy])} candles of {timeframe} data")
        
        # Analyze market conditions
        analysis = await strategy_manager.analyze_market_conditions(market_data)
        
        if 'error' in analysis:
            logger.error(f"❌ Analysis failed: {analysis['error']}")
            continue
        
        # Display results
        logger.info(f"⚙️ Calculation time: {analysis['calculation_time_ms']:.2f}ms")
        logger.info(f"🎯 Cache hit rate: {analysis['performance_metrics']['cache_hits']} hits, "
                   f"{analysis['performance_metrics']['cache_misses']} misses")
        
        # Market signals
        signals = analysis['market_signals']
        logger.info(f"🔔 Signals: {signals['total_signals']} total, {signals['strong_signals']} strong")
        logger.info(f"📊 Consensus: {signals['signal_consensus']}")
        
        # Market regime
        regime = analysis['market_regime']
        logger.info(f"🌍 Market regime: {regime['regime']} (confidence: {regime['confidence']})")
        
        # Recommendations
        recommendations = analysis['trading_recommendations']
        logger.info(f"💡 Recommendation: {recommendations['overall_recommendation']}")
        logger.info(f"🎯 Confidence: {recommendations['confidence_score']}/100")
        
        if recommendations['specific_recommendations']:
            logger.info("📋 Specific recommendations:")
            for rec in recommendations['specific_recommendations'][:2]:  # Show top 2
                logger.info(f"  • {rec}")
    
    # Performance summary
    logger.info(f"\n📈 Performance Summary")
    logger.info("-" * 30)
    
    performance = strategy_manager.get_performance_summary()
    logger.info(f"Total analyses: {performance['total_analyses']}")
    logger.info(f"Average calculation time: {performance['recent_avg_calc_time_ms']:.2f}ms")
    logger.info(f"Average cache hit rate: {performance['recent_avg_cache_hit_rate']:.1f}%")
    logger.info(f"Performance trend: {performance['performance_trend']}")


async def demo_real_time_updates():
    """Demonstrate real-time incremental updates."""
    
    logger.info(f"\n🔄 Real-time Updates Demo")
    logger.info("-" * 30)
    
    try:
        from bot.indicators.unified_framework import unified_framework
        
        strategy_manager = EnhancedStrategyManager()
        
        # Generate initial data
        initial_data = {
            'scalping': strategy_manager.market_data_provider.generate_realistic_data(100, '1m')
        }
        
        logger.info(f"📊 Setup incremental mode with {len(initial_data['scalping'])} initial candles")
        
        # Setup incremental mode
        setup_results = await unified_framework.setup_incremental_mode(
            strategy_type='scalping',
            initial_data=initial_data
        )
        
        logger.info(f"🔧 Incremental setup: {setup_results}")
        
        # Simulate real-time ticks
        last_candle = initial_data['scalping'].iloc[-1]
        
        for i in range(5):
            # Create new tick
            price_change = np.random.normal(0, 0.002)
            new_price = last_candle['close'] * (1 + price_change)
            
            new_tick = {
                'timestamp': datetime.now(),
                'open': last_candle['close'],
                'high': new_price * (1 + abs(price_change) * 0.5),
                'low': new_price * (1 - abs(price_change) * 0.5),
                'close': new_price,
                'volume': np.random.uniform(500, 2000)
            }
            
            # Update incrementally
            incremental_results = await unified_framework.update_incremental(
                strategy_type='scalping',
                new_tick=new_tick
            )
            
            logger.info(f"📊 Tick {i+1}: Price {new_price:.2f}, "
                       f"Updated {len(incremental_results)} indicators")
            
            last_candle = pd.Series(new_tick)
    
    except ImportError:
        logger.warning("⚠️ Real-time updates require full framework - skipping demo")


async def main():
    """Main demo function."""
    
    logger.info("🎯 Strategy Integration Demo with Unified Indicator Framework")
    logger.info("=" * 70)
    
    try:
        # Demo 1: Strategy integration
        await demo_strategy_integration()
        
        # Demo 2: Real-time updates (if available)
        await demo_real_time_updates()
        
        logger.info("\n✅ All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())