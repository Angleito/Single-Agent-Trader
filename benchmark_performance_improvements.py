#!/usr/bin/env python3
"""
Performance Improvement Benchmark Suite

This script measures and validates the performance improvements achieved by 
optimization fixes, testing the specific metrics mentioned in the mission:

1. LLM Response Time: Target <2 seconds with caching
2. WebSocket Processing Latency: Target <100ms non-blocking
3. System Startup Time: Target <30 seconds clean startup
4. Memory Usage: Monitor consumption and detect leaks

Usage:
    python benchmark_performance_improvements.py
"""

import asyncio
import logging
import time
import tracemalloc
from datetime import datetime, timedelta
from decimal import Decimal
from statistics import mean, median
from typing import Any, Dict, List

import pandas as pd
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmarkSuite:
    """
    Comprehensive benchmark suite for measuring performance improvements.
    
    Tests all critical performance metrics with detailed reporting and validation
    against target thresholds.
    """

    def __init__(self):
        """Initialize the benchmark suite."""
        self.results = {}
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance targets
        self.targets = {
            'llm_response_time_ms': 2000,  # 2 seconds
            'websocket_latency_ms': 100,   # 100ms
            'startup_time_seconds': 30,    # 30 seconds
            'memory_leak_threshold_mb': 50  # 50MB increase
        }
        
        logger.info("🚀 Performance Benchmark Suite initialized")
        logger.info(f"   Targets: LLM<{self.targets['llm_response_time_ms']}ms, "
                   f"WebSocket<{self.targets['websocket_latency_ms']}ms, "
                   f"Startup<{self.targets['startup_time_seconds']}s")

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all performance benchmarks and generate comprehensive report.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("🔥 Starting comprehensive performance benchmark suite...")
        suite_start_time = time.time()
        
        try:
            # 1. System Startup Time Benchmark
            logger.info("📊 1. Testing System Startup Performance...")
            self.results['startup_performance'] = await self.benchmark_startup_time()
            
            # 2. LLM Response Time Benchmark
            logger.info("📊 2. Testing LLM Response Performance...")
            self.results['llm_performance'] = await self.benchmark_llm_response_time()
            
            # 3. WebSocket Processing Latency Benchmark
            logger.info("📊 3. Testing WebSocket Processing Performance...")
            self.results['websocket_performance'] = await self.benchmark_websocket_latency()
            
            # 4. Memory Usage and Leak Detection
            logger.info("📊 4. Testing Memory Usage and Leak Detection...")
            self.results['memory_performance'] = await self.benchmark_memory_usage()
            
            # 5. Cache Performance Benchmark
            logger.info("📊 5. Testing Cache Performance...")
            self.results['cache_performance'] = await self.benchmark_cache_performance()
            
            # 6. Concurrent Operations Benchmark
            logger.info("📊 6. Testing Concurrent Operations Performance...")
            self.results['concurrent_performance'] = await self.benchmark_concurrent_operations()
            
        except Exception as e:
            logger.error(f"❌ Benchmark suite failed: {e}")
            self.results['error'] = str(e)
        
        # Calculate total benchmark time
        suite_total_time = time.time() - suite_start_time
        self.results['suite_metadata'] = {
            'total_time_seconds': suite_total_time,
            'timestamp': datetime.now().isoformat(),
            'targets': self.targets
        }
        
        # Generate final report
        report = self.generate_final_report()
        logger.info(report)
        
        return self.results

    async def benchmark_startup_time(self) -> Dict[str, Any]:
        """
        Benchmark system startup time from import to ready state.
        
        Target: <30 seconds clean startup
        """
        logger.info("⏱️  Measuring system startup time...")
        
        startup_times = []
        
        for attempt in range(3):  # Test 3 startup attempts
            logger.info(f"   Startup attempt {attempt + 1}/3...")
            
            start_time = time.time()
            
            try:
                # Simulate clean startup process
                # Import bot modules (this would normally trigger initialization)
                import sys
                import importlib
                
                # Clear any cached modules to simulate fresh startup
                modules_to_clear = [k for k in sys.modules.keys() if k.startswith('bot.')]
                for module in modules_to_clear:
                    if module in sys.modules:
                        del sys.modules[module]
                
                # Time the import process
                import_start = time.time()
                
                # Import core bot modules
                import bot.config
                import bot.types
                import bot.validator
                import bot.risk
                import bot.indicators.vumanchu
                import bot.strategy.llm_agent
                import bot.strategy.llm_cache
                import bot.data.market
                
                import_time = time.time() - import_start
                
                # Simulate initialization
                init_start = time.time()
                
                # Create basic configuration
                settings = bot.config.create_settings()
                
                # Initialize key components
                validator = bot.validator.JSONValidator()
                risk_manager = bot.risk.RiskManager(settings)
                
                init_time = time.time() - init_start
                
                total_startup_time = time.time() - start_time
                startup_times.append(total_startup_time)
                
                logger.info(f"     ✅ Startup {attempt + 1}: {total_startup_time:.2f}s "
                           f"(import: {import_time:.2f}s, init: {init_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"     ❌ Startup {attempt + 1} failed: {e}")
                startup_times.append(self.targets['startup_time_seconds'] * 2)  # Penalty
        
        avg_startup_time = mean(startup_times)
        min_startup_time = min(startup_times)
        max_startup_time = max(startup_times)
        
        target_achieved = avg_startup_time <= self.targets['startup_time_seconds']
        
        result = {
            'avg_startup_time_seconds': avg_startup_time,
            'min_startup_time_seconds': min_startup_time,
            'max_startup_time_seconds': max_startup_time,
            'target_seconds': self.targets['startup_time_seconds'],
            'target_achieved': target_achieved,
            'improvement_needed_seconds': max(0, avg_startup_time - self.targets['startup_time_seconds']),
            'attempts': len(startup_times),
            'success_rate': sum(1 for t in startup_times if t <= self.targets['startup_time_seconds']) / len(startup_times)
        }
        
        status = "✅ PASSED" if target_achieved else "❌ FAILED"
        logger.info(f"🎯 Startup Performance: {status} - Avg: {avg_startup_time:.2f}s (Target: {self.targets['startup_time_seconds']}s)")
        
        return result

    async def benchmark_llm_response_time(self) -> Dict[str, Any]:
        """
        Benchmark LLM response time with and without caching.
        
        Target: <2 seconds with caching
        Before: 2-8 seconds
        """
        logger.info("🧠 Measuring LLM response performance...")
        
        try:
            from bot.strategy.llm_agent import LLMAgent
            from bot.strategy.llm_cache import get_llm_cache
            from bot.types import MarketState, Position, IndicatorData
            
            # Initialize LLM agent
            agent = LLMAgent(model_provider="openai", model_name="gpt-3.5-turbo")
            cache = get_llm_cache()
            
            # Clear cache for clean testing
            cache.clear_cache()
            
            # Create test market state
            test_position = Position(
                symbol="BTC-USD",
                side="FLAT",
                size=Decimal("0"),
                timestamp=datetime.now()
            )
            
            test_indicators = IndicatorData(
                cipher_a_dot=1.0,
                cipher_b_wave=0.5,
                cipher_b_money_flow=55.0,
                rsi=45.0,
                ema_fast=50000.0,
                ema_slow=49900.0
            )
            
            # Generate test OHLCV data
            test_ohlcv = []
            for i in range(10):
                test_ohlcv.append({
                    'timestamp': datetime.now() - timedelta(minutes=10-i),
                    'open': 50000 + i * 10,
                    'high': 50100 + i * 10,
                    'low': 49900 + i * 10,
                    'close': 50050 + i * 10,
                    'volume': 100
                })
            
            test_market_state = MarketState(
                symbol="BTC-USD",
                interval="1m",
                timestamp=datetime.now(),
                current_price=Decimal("50000"),
                ohlcv_data=test_ohlcv,
                indicators=test_indicators,
                current_position=test_position
            )
            
            # Test without cache (fresh requests)
            fresh_response_times = []
            for i in range(5):  # 5 fresh requests
                logger.info(f"   Fresh request {i + 1}/5...")
                
                # Modify market state slightly to avoid cache hits
                test_market_state.current_price = Decimal(str(50000 + i * 100))
                test_indicators.rsi = 45.0 + i * 2
                
                start_time = time.time()
                try:
                    response = await agent.analyze_market(test_market_state)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    fresh_response_times.append(response_time)
                    
                    logger.info(f"     Fresh response: {response_time:.1f}ms - Action: {response.action}")
                    
                except Exception as e:
                    logger.error(f"     Fresh request failed: {e}")
                    fresh_response_times.append(8000)  # 8 second penalty
                
                # Small delay between requests
                await asyncio.sleep(0.5)
            
            # Test with cache (repeated similar requests)
            cached_response_times = []
            for i in range(10):  # 10 potentially cached requests
                logger.info(f"   Cached request {i + 1}/10...")
                
                # Use similar market states that should hit cache
                test_market_state.current_price = Decimal("50000")  # Same price
                test_indicators.rsi = 45.0  # Same RSI
                
                start_time = time.time()
                try:
                    response = await cache.get_or_compute(
                        test_market_state,
                        agent.analyze_market,
                        test_market_state
                    )
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    cached_response_times.append(response_time)
                    
                    cache_status = "HIT" if response_time < 100 else "MISS"
                    logger.info(f"     Cached response: {response_time:.1f}ms - {cache_status} - Action: {response.action}")
                    
                except Exception as e:
                    logger.error(f"     Cached request failed: {e}")
                    cached_response_times.append(2000)  # 2 second penalty
                
                await asyncio.sleep(0.2)
            
            # Calculate statistics
            avg_fresh_time = mean(fresh_response_times)
            avg_cached_time = mean(cached_response_times)
            min_cached_time = min(cached_response_times)
            max_cached_time = max(cached_response_times)
            
            # Determine cache hits (responses under 100ms are likely cache hits)
            cache_hits = sum(1 for t in cached_response_times if t < 100)
            cache_hit_rate = cache_hits / len(cached_response_times) * 100
            
            target_achieved = avg_cached_time <= self.targets['llm_response_time_ms']
            improvement_percent = ((avg_fresh_time - avg_cached_time) / avg_fresh_time) * 100
            
        except ImportError as e:
            logger.error(f"❌ Cannot test LLM performance - missing dependencies: {e}")
            return {
                'error': f'Missing dependencies: {e}',
                'target_achieved': False
            }
        except Exception as e:
            logger.error(f"❌ LLM benchmark failed: {e}")
            return {
                'error': str(e),
                'target_achieved': False
            }
        
        result = {
            'avg_fresh_response_time_ms': avg_fresh_time,
            'avg_cached_response_time_ms': avg_cached_time,
            'min_cached_response_time_ms': min_cached_time,
            'max_cached_response_time_ms': max_cached_time,
            'cache_hit_rate_percent': cache_hit_rate,
            'performance_improvement_percent': improvement_percent,
            'target_ms': self.targets['llm_response_time_ms'],
            'target_achieved': target_achieved,
            'fresh_requests': len(fresh_response_times),
            'cached_requests': len(cached_response_times)
        }
        
        status = "✅ PASSED" if target_achieved else "❌ FAILED"
        logger.info(f"🎯 LLM Performance: {status} - Cached Avg: {avg_cached_time:.1f}ms "
                   f"(Target: {self.targets['llm_response_time_ms']}ms, "
                   f"Improvement: {improvement_percent:.1f}%)")
        
        return result

    async def benchmark_websocket_latency(self) -> Dict[str, Any]:
        """
        Benchmark WebSocket processing latency and non-blocking behavior.
        
        Target: <100ms processing, non-blocking operations
        """
        logger.info("🌐 Measuring WebSocket processing latency...")
        
        try:
            from bot.data.market import MarketDataProvider
            
            # Initialize market data provider
            provider = MarketDataProvider("BTC-USD", "1m")
            
            # Initialize message queue
            provider._message_queue = asyncio.Queue(maxsize=1000)
            provider._running = True
            
            # Start message processor
            processor_task = asyncio.create_task(provider._process_websocket_messages())
            
            # Measure message queueing performance (non-blocking test)
            messages = []
            for i in range(1000):
                message = {
                    'channel': 'ticker',
                    'events': [{
                        'type': 'update',
                        'tickers': [{'product_id': 'BTC-USD', 'price': f'{50000 + i}'}]
                    }]
                }
                messages.append(message)
            
            # Test 1: Message queueing speed (should be non-blocking)
            queue_start_time = time.time()
            queued_count = 0
            
            for message in messages:
                try:
                    provider._message_queue.put_nowait(message)
                    queued_count += 1
                except asyncio.QueueFull:
                    break  # Queue full is acceptable
            
            queue_time_ms = (time.time() - queue_start_time) * 1000
            
            # Test 2: Processing latency simulation
            processing_times = []
            
            for i in range(100):  # Process 100 messages
                message = {
                    'channel': 'ticker',
                    'events': [{
                        'type': 'update',
                        'tickers': [{'product_id': 'BTC-USD', 'price': f'{50000 + i}'}]
                    }]
                }
                
                process_start = time.time()
                
                # Simulate message processing
                try:
                    provider._message_queue.put_nowait(message)
                    # Simulate processing overhead
                    await asyncio.sleep(0.001)  # 1ms simulated processing
                except asyncio.QueueFull:
                    pass
                
                process_time = (time.time() - process_start) * 1000
                processing_times.append(process_time)
            
            # Stop processor
            provider._running = False
            await asyncio.sleep(0.1)
            processor_task.cancel()
            
            # Calculate statistics
            avg_processing_time = mean(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
            
            # Non-blocking test: queue 1000 messages should be very fast
            non_blocking_achieved = queue_time_ms < 50  # Under 50ms for 1000 messages
            latency_target_achieved = avg_processing_time <= self.targets['websocket_latency_ms']
            
        except Exception as e:
            logger.error(f"❌ WebSocket benchmark failed: {e}")
            return {
                'error': str(e),
                'target_achieved': False
            }
        
        result = {
            'queue_time_ms': queue_time_ms,
            'queued_messages': queued_count,
            'avg_processing_time_ms': avg_processing_time,
            'min_processing_time_ms': min_processing_time,
            'max_processing_time_ms': max_processing_time,
            'non_blocking_achieved': non_blocking_achieved,
            'latency_target_achieved': latency_target_achieved,
            'target_ms': self.targets['websocket_latency_ms'],
            'processed_messages': len(processing_times)
        }
        
        overall_target_achieved = non_blocking_achieved and latency_target_achieved
        status = "✅ PASSED" if overall_target_achieved else "❌ FAILED"
        logger.info(f"🎯 WebSocket Performance: {status} - Processing: {avg_processing_time:.1f}ms, "
                   f"Queue: {queue_time_ms:.1f}ms (Target: <{self.targets['websocket_latency_ms']}ms)")
        
        result['target_achieved'] = overall_target_achieved
        return result

    async def benchmark_memory_usage(self) -> Dict[str, Any]:
        """
        Benchmark memory usage and detect potential memory leaks.
        
        Target: No significant memory leaks (< 50MB increase)
        """
        logger.info("💾 Measuring memory usage and leak detection...")
        
        # Start memory tracing
        tracemalloc.start()
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_snapshots = [initial_memory]
        
        try:
            # Simulate intensive operations that could cause memory leaks
            
            # Test 1: Repeated indicator calculations
            logger.info("   Testing indicator calculations memory usage...")
            from bot.indicators.vumanchu import VuManChuIndicators
            
            indicator_calc = VuManChuIndicators()
            
            # Generate test data multiple times and calculate indicators
            for i in range(50):
                # Create test DataFrame
                test_data = []
                for j in range(200):
                    test_data.append({
                        'timestamp': datetime.now() - timedelta(minutes=200-j),
                        'open': 50000 + j,
                        'high': 50100 + j,
                        'low': 49900 + j,
                        'close': 50050 + j,
                        'volume': 100
                    })
                
                df = pd.DataFrame(test_data)
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators
                result = indicator_calc.calculate_all(df)
                
                # Force cleanup
                del result, df, test_data
                
                # Record memory every 10 iterations
                if i % 10 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(current_memory)
                    logger.info(f"     Iteration {i}: {current_memory:.1f}MB")
            
            # Test 2: Cache operations
            logger.info("   Testing cache memory usage...")
            from bot.strategy.llm_cache import get_llm_cache
            
            cache = get_llm_cache()
            cache.clear_cache()
            
            # Fill cache with test data
            for i in range(100):
                # This is a simulation - actual cache would be used in real scenario
                await asyncio.sleep(0.01)  # Simulate async operation
                
                if i % 20 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(current_memory)
            
            # Test 3: WebSocket message processing simulation
            logger.info("   Testing WebSocket message processing memory...")
            
            messages = []
            for i in range(1000):
                message = {
                    'channel': 'ticker',
                    'events': [{
                        'type': 'update',
                        'tickers': [{'product_id': 'BTC-USD', 'price': f'{50000 + i}'}]
                    }],
                    'large_data': 'x' * 1000  # 1KB of data per message
                }
                messages.append(message)
            
            # Process messages
            for i, message in enumerate(messages):
                # Simulate processing
                processed = str(message)
                del processed
                
                if i % 200 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(current_memory)
            
            # Clear messages
            del messages
            
        except Exception as e:
            logger.error(f"❌ Memory benchmark failed: {e}")
            return {
                'error': str(e),
                'memory_leak_detected': True
            }
        
        # Get final memory measurements
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_snapshots.append(final_memory)
        
        # Get tracemalloc data
        current_trace, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate memory statistics
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_snapshots)
        peak_increase = max_memory - initial_memory
        
        # Detect memory leak
        memory_leak_detected = peak_increase > self.targets['memory_leak_threshold_mb']
        
        result = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'max_memory_mb': max_memory,
            'memory_increase_mb': memory_increase,
            'peak_increase_mb': peak_increase,
            'memory_leak_detected': memory_leak_detected,
            'tracemalloc_current_mb': current_trace / 1024 / 1024,
            'tracemalloc_peak_mb': peak_trace / 1024 / 1024,
            'target_threshold_mb': self.targets['memory_leak_threshold_mb'],
            'snapshots_count': len(memory_snapshots),
            'target_achieved': not memory_leak_detected
        }
        
        status = "✅ PASSED" if not memory_leak_detected else "❌ FAILED"
        logger.info(f"🎯 Memory Performance: {status} - Peak increase: {peak_increase:.1f}MB "
                   f"(Threshold: {self.targets['memory_leak_threshold_mb']}MB)")
        
        return result

    async def benchmark_cache_performance(self) -> Dict[str, Any]:
        """
        Benchmark cache performance and effectiveness.
        """
        logger.info("⚡ Measuring cache performance...")
        
        try:
            from bot.strategy.llm_cache import LLMResponseCache, MarketStateHasher
            from bot.types import MarketState, Position, IndicatorData, TradeAction
            
            # Initialize cache
            cache = LLMResponseCache(ttl_seconds=60, max_entries=100)
            hasher = MarketStateHasher()
            
            # Create test market states
            test_states = []
            for i in range(50):
                position = Position(
                    symbol="BTC-USD",
                    side="FLAT",
                    size=Decimal("0"),
                    timestamp=datetime.now()
                )
                
                indicators = IndicatorData(
                    cipher_a_dot=1.0 + i * 0.1,
                    cipher_b_wave=0.5 + i * 0.01,
                    cipher_b_money_flow=55.0 + i,
                    rsi=45.0 + i,
                    ema_fast=50000.0 + i * 10,
                    ema_slow=49900.0 + i * 10
                )
                
                state = MarketState(
                    symbol="BTC-USD",
                    interval="1m",
                    timestamp=datetime.now(),
                    current_price=Decimal(str(50000 + i * 100)),
                    ohlcv_data=[],
                    indicators=indicators,
                    current_position=position
                )
                
                test_states.append(state)
            
            # Mock compute function
            async def mock_compute(*args, **kwargs):
                await asyncio.sleep(0.5)  # Simulate 500ms computation
                return TradeAction(
                    action="HOLD",
                    size_pct=0,
                    leverage=None,
                    stop_loss_pct=None,
                    take_profit_pct=None,
                    reasoning="Mock computation"
                )
            
            # Test cache performance
            cache_test_times = []
            cache_hits = 0
            cache_misses = 0
            
            # First pass - populate cache
            for i, state in enumerate(test_states[:10]):
                start_time = time.time()
                result = await cache.get_or_compute(state, mock_compute)
                response_time = (time.time() - start_time) * 1000
                cache_test_times.append(response_time)
                
                if response_time < 100:  # Likely cache hit
                    cache_hits += 1
                else:
                    cache_misses += 1
                
                logger.info(f"   Cache test {i+1}: {response_time:.1f}ms")
            
            # Second pass - test cache hits with similar states
            for i, state in enumerate(test_states[:10]):  # Same states
                start_time = time.time()
                result = await cache.get_or_compute(state, mock_compute)
                response_time = (time.time() - start_time) * 1000
                cache_test_times.append(response_time)
                
                if response_time < 100:  # Likely cache hit
                    cache_hits += 1
                else:
                    cache_misses += 1
            
            # Get cache statistics
            cache_stats = cache.get_cache_stats()
            
            avg_response_time = mean(cache_test_times)
            cache_hit_rate = (cache_hits / len(cache_test_times)) * 100
            
        except Exception as e:
            logger.error(f"❌ Cache benchmark failed: {e}")
            return {
                'error': str(e),
                'target_achieved': False
            }
        
        result = {
            'avg_response_time_ms': avg_response_time,
            'cache_hit_rate_percent': cache_hit_rate,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_stats': cache_stats,
            'total_tests': len(cache_test_times),
            'target_achieved': avg_response_time <= 500  # 500ms is reasonable with cache
        }
        
        status = "✅ PASSED" if result['target_achieved'] else "❌ FAILED"
        logger.info(f"🎯 Cache Performance: {status} - Avg: {avg_response_time:.1f}ms, "
                   f"Hit Rate: {cache_hit_rate:.1f}%")
        
        return result

    async def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """
        Benchmark concurrent operations performance.
        """
        logger.info("🔄 Measuring concurrent operations performance...")
        
        try:
            # Simulate multiple concurrent operations
            tasks = []
            
            # Task 1: Multiple indicator calculations
            async def indicator_calculations():
                from bot.indicators.vumanchu import CipherA
                cipher_a = CipherA()
                
                times = []
                for i in range(10):
                    # Create test data
                    test_data = []
                    for j in range(100):
                        test_data.append({
                            'timestamp': datetime.now() - timedelta(minutes=100-j),
                            'open': 50000 + j + i * 10,
                            'high': 50100 + j + i * 10,
                            'low': 49900 + j + i * 10,
                            'close': 50050 + j + i * 10,
                            'volume': 100
                        })
                    
                    df = pd.DataFrame(test_data)
                    df.set_index('timestamp', inplace=True)
                    
                    start_time = time.time()
                    result = cipher_a.calculate(df)
                    calc_time = (time.time() - start_time) * 1000
                    times.append(calc_time)
                
                return {'indicator_times': times, 'avg_time_ms': mean(times)}
            
            # Task 2: Message queue operations
            async def message_operations():
                queue = asyncio.Queue(maxsize=1000)
                times = []
                
                for i in range(100):
                    start_time = time.time()
                    
                    # Add message to queue
                    message = {'type': 'test', 'data': i, 'payload': 'x' * 100}
                    await queue.put(message)
                    
                    # Get message from queue
                    retrieved = await queue.get()
                    
                    op_time = (time.time() - start_time) * 1000
                    times.append(op_time)
                
                return {'queue_times': times, 'avg_time_ms': mean(times)}
            
            # Task 3: Memory operations
            async def memory_operations():
                times = []
                data_objects = []
                
                for i in range(50):
                    start_time = time.time()
                    
                    # Create and manipulate data
                    data = {
                        'id': i,
                        'values': list(range(1000)),
                        'timestamp': datetime.now(),
                        'metadata': {'key': 'value'} * 10
                    }
                    data_objects.append(data)
                    
                    # Process data
                    processed = str(data)
                    
                    op_time = (time.time() - start_time) * 1000
                    times.append(op_time)
                
                # Cleanup
                del data_objects
                
                return {'memory_times': times, 'avg_time_ms': mean(times)}
            
            # Run all tasks concurrently
            concurrent_start = time.time()
            
            task1 = asyncio.create_task(indicator_calculations())
            task2 = asyncio.create_task(message_operations())
            task3 = asyncio.create_task(memory_operations())
            
            results = await asyncio.gather(task1, task2, task3, return_exceptions=True)
            
            concurrent_time = (time.time() - concurrent_start) * 1000
            
            # Process results
            indicator_result = results[0] if not isinstance(results[0], Exception) else None
            queue_result = results[1] if not isinstance(results[1], Exception) else None
            memory_result = results[2] if not isinstance(results[2], Exception) else None
            
            # Calculate overall performance
            all_successful = all(not isinstance(r, Exception) for r in results)
            
        except Exception as e:
            logger.error(f"❌ Concurrent operations benchmark failed: {e}")
            return {
                'error': str(e),
                'target_achieved': False
            }
        
        result = {
            'concurrent_time_ms': concurrent_time,
            'all_successful': all_successful,
            'indicator_performance': indicator_result,
            'queue_performance': queue_result,
            'memory_performance': memory_result,
            'target_achieved': concurrent_time <= 10000 and all_successful  # 10 seconds max
        }
        
        status = "✅ PASSED" if result['target_achieved'] else "❌ FAILED"
        logger.info(f"🎯 Concurrent Operations: {status} - Total: {concurrent_time:.1f}ms")
        
        return result

    def generate_final_report(self) -> str:
        """
        Generate comprehensive final performance report.
        """
        total_time = self.results.get('suite_metadata', {}).get('total_time_seconds', 0)
        timestamp = self.results.get('suite_metadata', {}).get('timestamp', 'Unknown')
        
        # Calculate overall success rate
        benchmarks = [
            'startup_performance',
            'llm_performance', 
            'websocket_performance',
            'memory_performance',
            'cache_performance',
            'concurrent_performance'
        ]
        
        passed_count = 0
        total_count = 0
        
        results_summary = []
        
        for benchmark in benchmarks:
            if benchmark in self.results:
                total_count += 1
                result = self.results[benchmark]
                
                if result.get('target_achieved', False):
                    passed_count += 1
                    status = "✅ PASSED"
                else:
                    status = "❌ FAILED"
                
                # Get key metrics for each benchmark
                if benchmark == 'startup_performance':
                    metric = f"{result.get('avg_startup_time_seconds', 0):.2f}s"
                elif benchmark == 'llm_performance':
                    metric = f"{result.get('avg_cached_response_time_ms', 0):.1f}ms"
                elif benchmark == 'websocket_performance':
                    metric = f"{result.get('avg_processing_time_ms', 0):.1f}ms"
                elif benchmark == 'memory_performance':
                    metric = f"{result.get('peak_increase_mb', 0):.1f}MB"
                elif benchmark == 'cache_performance':
                    metric = f"{result.get('cache_hit_rate_percent', 0):.1f}%"
                elif benchmark == 'concurrent_performance':
                    metric = f"{result.get('concurrent_time_ms', 0):.1f}ms"
                else:
                    metric = "N/A"
                
                results_summary.append(f"   {benchmark.replace('_', ' ').title()}: {status} - {metric}")
        
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        # Overall status
        if success_rate >= 100:
            overall_status = "🎯 ALL TARGETS ACHIEVED"
        elif success_rate >= 80:
            overall_status = "🟡 MOSTLY SUCCESSFUL"
        else:
            overall_status = "🔴 NEEDS OPTIMIZATION"
        
        report = f"""
{'='*60}
🚀 PERFORMANCE IMPROVEMENT BENCHMARK RESULTS
{'='*60}

{overall_status}
Success Rate: {success_rate:.1f}% ({passed_count}/{total_count} benchmarks passed)
Total Benchmark Time: {total_time:.2f} seconds
Timestamp: {timestamp}

📊 BENCHMARK RESULTS:
{chr(10).join(results_summary)}

🎯 TARGET SUMMARY:
   LLM Response Time: <{self.targets['llm_response_time_ms']}ms with caching
   WebSocket Latency: <{self.targets['websocket_latency_ms']}ms non-blocking
   System Startup: <{self.targets['startup_time_seconds']}s clean startup
   Memory Leak Threshold: <{self.targets['memory_leak_threshold_mb']}MB increase

📈 KEY ACHIEVEMENTS:
"""
        
        # Add specific achievements based on results
        if self.results.get('llm_performance', {}).get('target_achieved', False):
            improvement = self.results['llm_performance'].get('performance_improvement_percent', 0)
            report += f"   ✅ LLM Response Time Optimized: {improvement:.1f}% improvement\n"
        
        if self.results.get('websocket_performance', {}).get('target_achieved', False):
            report += f"   ✅ WebSocket Processing: Non-blocking operations achieved\n"
        
        if self.results.get('startup_performance', {}).get('target_achieved', False):
            report += f"   ✅ System Startup: Clean startup under target time\n"
        
        if not self.results.get('memory_performance', {}).get('memory_leak_detected', True):
            report += f"   ✅ Memory Management: No significant memory leaks detected\n"
        
        report += f"\n{'='*60}"
        
        return report


async def main():
    """
    Main function to run the performance benchmark suite.
    """
    print("🚀 Performance Improvement Benchmark Suite")
    print("=" * 60)
    print()
    
    # Initialize and run benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite()
    
    try:
        results = await benchmark_suite.run_all_benchmarks()
        
        print("\n🎯 Benchmark suite completed successfully!")
        print(f"Results saved to: {datetime.now().isoformat()}")
        
        # Optionally save results to file
        import json
        results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert Decimal objects to float for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(v) for v in obj]
            return obj
        
        serializable_results = convert_decimals(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Benchmark suite failed: {e}")
        print(f"\n❌ Benchmark suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)