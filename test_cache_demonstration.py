#!/usr/bin/env python3
"""
Cache Performance Demonstration

This script demonstrates the actual LLM cache performance improvements
with realistic market data scenarios.
"""

import asyncio
import logging
import time
from statistics import mean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_cache_performance():
    """Demonstrate LLM cache performance with realistic scenarios."""
    logger.info("🧠 Demonstrating LLM Cache Performance...")

    try:
        from bot.strategy.llm_cache import LLMResponseCache, MarketStateHasher
        from bot.types import TradeAction

        # Initialize cache
        cache = LLMResponseCache(ttl_seconds=60, max_entries=100)
        hasher = MarketStateHasher()

        # Mock LLM computation function
        async def mock_expensive_llm_call(*args, **kwargs):
            """Simulate expensive LLM API call."""
            await asyncio.sleep(0.3)  # 300ms simulated API call
            return TradeAction(
                action="HOLD",
                size_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                rationale="Mock LLM decision based on market analysis"
            )

        # Create realistic market state scenarios
        market_scenarios = []

        for i in range(20):
            # Create mock market state
            mock_state = type('MockMarketState', (), {
                'current_price': 50000 + (i % 5) * 100,  # Price varies in small ranges
                'indicators': type('MockIndicators', (), {
                    'rsi': 45 + (i % 3) * 5,  # RSI varies slightly
                    'cipher_a_dot': 1.0 + (i % 2) * 0.1,
                    'cipher_b_wave': 0.5 + (i % 4) * 0.01,
                    'cipher_b_money_flow': 55 + (i % 3) * 2,
                    'stablecoin_dominance': 7.5,
                    'market_sentiment': 'NEUTRAL'
                })(),
                'current_position': type('MockPosition', (), {
                    'side': 'FLAT'
                })()
            })()
            market_scenarios.append(mock_state)

        # Test Phase 1: Fresh requests (cache misses)
        logger.info("📊 Phase 1: Testing fresh LLM requests (cache misses)...")
        fresh_times = []

        for i, scenario in enumerate(market_scenarios[:10]):
            start_time = time.time()
            response = await cache.get_or_compute(scenario, mock_expensive_llm_call)
            response_time = (time.time() - start_time) * 1000
            fresh_times.append(response_time)

            logger.info(f"   Request {i+1}: {response_time:.1f}ms (FRESH)")

        # Test Phase 2: Repeated requests (cache hits)
        logger.info("\n📊 Phase 2: Testing repeated requests (cache hits)...")
        cached_times = []

        for i, scenario in enumerate(market_scenarios[:10]):  # Same scenarios
            start_time = time.time()
            response = await cache.get_or_compute(scenario, mock_expensive_llm_call)
            response_time = (time.time() - start_time) * 1000
            cached_times.append(response_time)

            cache_status = "HIT" if response_time < 50 else "MISS"
            logger.info(f"   Request {i+1}: {response_time:.1f}ms ({cache_status})")

        # Test Phase 3: Similar market conditions (should hit cache)
        logger.info("\n📊 Phase 3: Testing similar market conditions...")
        similar_times = []

        for i in range(15):
            # Create similar market state (should hit cache due to bucketing)
            similar_state = type('MockMarketState', (), {
                'current_price': 50000 + i * 10,  # Small price changes
                'indicators': type('MockIndicators', (), {
                    'rsi': 45 + i * 0.5,  # Small RSI changes
                    'cipher_a_dot': 1.0,
                    'cipher_b_wave': 0.5,
                    'cipher_b_money_flow': 55,
                    'stablecoin_dominance': 7.5,
                    'market_sentiment': 'NEUTRAL'
                })(),
                'current_position': type('MockPosition', (), {
                    'side': 'FLAT'
                })()
            })()

            start_time = time.time()
            response = await cache.get_or_compute(similar_state, mock_expensive_llm_call)
            response_time = (time.time() - start_time) * 1000
            similar_times.append(response_time)

            cache_status = "HIT" if response_time < 50 else "MISS"
            if i < 5:  # Log first few
                logger.info(f"   Similar {i+1}: {response_time:.1f}ms ({cache_status})")

        # Calculate performance metrics
        avg_fresh_time = mean(fresh_times)
        avg_cached_time = mean(cached_times)
        avg_similar_time = mean(similar_times)

        # Count cache hits (responses under 50ms are likely cache hits)
        cached_hits = sum(1 for t in cached_times if t < 50)
        similar_hits = sum(1 for t in similar_times if t < 50)

        cache_hit_rate_cached = (cached_hits / len(cached_times)) * 100
        cache_hit_rate_similar = (similar_hits / len(similar_times)) * 100

        # Calculate improvement
        improvement_cached = ((avg_fresh_time - avg_cached_time) / avg_fresh_time) * 100
        improvement_similar = ((avg_fresh_time - avg_similar_time) / avg_fresh_time) * 100

        # Get cache statistics
        cache_stats = cache.get_cache_stats()

        # Display results
        print("\n" + "="*60)
        print("🚀 LLM CACHE PERFORMANCE DEMONSTRATION RESULTS")
        print("="*60)

        print("\n📊 RESPONSE TIME ANALYSIS:")
        print(f"   Fresh Requests (Cache Miss):     {avg_fresh_time:.1f}ms")
        print(f"   Repeated Requests (Cache Hit):   {avg_cached_time:.1f}ms")
        print(f"   Similar Conditions:              {avg_similar_time:.1f}ms")

        print("\n🎯 CACHE EFFECTIVENESS:")
        print(f"   Repeated Requests Hit Rate:      {cache_hit_rate_cached:.1f}%")
        print(f"   Similar Conditions Hit Rate:     {cache_hit_rate_similar:.1f}%")
        print(f"   Overall Cache Hit Rate:          {cache_stats['hit_rate']:.1f}%")

        print("\n⚡ PERFORMANCE IMPROVEMENT:")
        print(f"   Repeated Requests Improvement:   {improvement_cached:.1f}%")
        print(f"   Similar Conditions Improvement:  {improvement_similar:.1f}%")

        print("\n💾 CACHE STATISTICS:")
        print(f"   Cache Size:                      {cache_stats['cache_size']} entries")
        print(f"   Total Requests:                  {cache_stats['total_requests']}")
        print(f"   Total Hits:                      {cache_stats['total_hits']}")
        print(f"   Total Misses:                    {cache_stats['total_misses']}")

        print("\n🎯 TARGET VALIDATION:")
        target_2000ms = avg_cached_time <= 2000
        target_improvement = improvement_cached >= 70

        print(f"   Response Time <2000ms:           {'✅ ACHIEVED' if target_2000ms else '❌ MISSED'}")
        print(f"   Performance Improvement >70%:    {'✅ ACHIEVED' if target_improvement else '❌ MISSED'}")

        overall_success = target_2000ms and (cache_hit_rate_cached > 50 or cache_hit_rate_similar > 30)
        print(f"   Overall Cache Performance:       {'✅ EXCELLENT' if overall_success else '🟡 GOOD' if cache_hit_rate_cached > 30 else '❌ NEEDS_WORK'}")

        print("="*60)

        return {
            'avg_fresh_time_ms': avg_fresh_time,
            'avg_cached_time_ms': avg_cached_time,
            'avg_similar_time_ms': avg_similar_time,
            'cache_hit_rate_repeated': cache_hit_rate_cached,
            'cache_hit_rate_similar': cache_hit_rate_similar,
            'improvement_repeated': improvement_cached,
            'improvement_similar': improvement_similar,
            'cache_stats': cache_stats,
            'target_achieved': overall_success
        }

    except Exception as e:
        logger.error(f"Cache demonstration failed: {e}")
        return {'error': str(e), 'target_achieved': False}


async def main():
    """Run cache performance demonstration."""
    await demonstrate_cache_performance()


if __name__ == "__main__":
    asyncio.run(main())
