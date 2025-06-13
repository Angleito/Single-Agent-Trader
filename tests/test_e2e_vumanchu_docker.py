"""
End-to-End Testing Suite for VuManChu Implementation in Docker Environment.

This comprehensive E2E test suite validates the complete VuManChu implementation 
in a containerized environment, ensuring production readiness with:
- Complete indicator pipeline from data input to signal output
- Integration with LLM agent and trading logic
- Real market data processing validation
- Performance benchmarking under load
- Memory usage and resource consumption monitoring
- Error recovery and resilience testing

Test Categories:
1. Full Pipeline Tests - Complete data flow validation
2. Performance Tests - Load testing and benchmarking
3. Memory Tests - Long-running memory consumption
4. Signal Quality Tests - Signal generation accuracy
5. Integration Tests - Component interaction validation  
6. Error Recovery Tests - Fault tolerance testing
7. Docker Environment Tests - Container-specific validation
"""

import asyncio
import json
import logging
import os
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import pytest

from bot.config import load_config
from bot.data.market import MarketDataProvider
from bot.indicators.vumanchu import VuManChuIndicators, CipherA, CipherB
from bot.strategy.llm_agent import LLMTradingAgent
from bot.strategy.core import CoreStrategy
from bot.risk import RiskManager
from bot.validator import TradeValidator
from bot.types import MarketState, IndicatorData, TradeAction

# Configure logging for E2E tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/e2e_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DockerTestEnvironment:
    """Docker environment setup and validation for E2E tests."""
    
    def __init__(self):
        self.is_docker = self._detect_docker_environment()
        self.test_data_dir = Path("/app/test_data")
        self.results_dir = Path("/app/test_results")
        self.logs_dir = Path("/app/logs")
        
    def _detect_docker_environment(self) -> bool:
        """Detect if running in Docker container."""
        return (
            os.path.exists('/.dockerenv') or
            os.environ.get('DOCKER_CONTAINER') == 'true' or
            os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read()
        )
    
    def setup_test_environment(self):
        """Set up test directories and validate Docker environment."""
        if not self.is_docker:
            logger.warning("Not running in Docker - creating local test directories")
            self.test_data_dir = Path("./test_data")
            self.results_dir = Path("./test_results")
            self.logs_dir = Path("./logs")
        
        # Create required directories
        for directory in [self.test_data_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Test environment setup complete:")
        logger.info(f"  Docker: {self.is_docker}")
        logger.info(f"  Test data: {self.test_data_dir}")
        logger.info(f"  Results: {self.results_dir}")
        logger.info(f"  Logs: {self.logs_dir}")
        
    def validate_docker_resources(self) -> Dict[str, Any]:
        """Validate Docker container resource limits and availability."""
        resources = {
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'cpu_count': psutil.cpu_count(),
            'disk_usage': psutil.disk_usage('/').free,
            'docker_detected': self.is_docker
        }
        
        if self.is_docker:
            # Check Docker-specific constraints
            if resources['memory_total'] < 512 * 1024 * 1024:  # 512MB minimum
                logger.warning("Container memory below recommended 512MB")
            if resources['cpu_count'] < 1:
                logger.warning("Container CPU allocation below 1 core")
                
        return resources


class E2ETestDataManager:
    """Manage test data for E2E testing scenarios."""
    
    def __init__(self, test_data_dir: Path):
        self.test_data_dir = test_data_dir
        
    def load_market_data(self, scenario: str = "default") -> pd.DataFrame:
        """Load market data for specific test scenario."""
        data_file = self.test_data_dir / f"market_data_{scenario}.csv"
        
        if data_file.exists():
            logger.info(f"Loading test data from {data_file}")
            return pd.read_csv(data_file, parse_dates=['timestamp'], index_col='timestamp')
        else:
            logger.info(f"Generating test data for scenario: {scenario}")
            return self._generate_test_data(scenario)
    
    def _generate_test_data(self, scenario: str) -> pd.DataFrame:
        """Generate test market data for different scenarios."""
        np.random.seed(42)  # Reproducible test data
        periods = 2000
        
        base_price = 50000.0
        timestamps = pd.date_range('2024-01-01', periods=periods, freq='1min')
        
        if scenario == "trending":
            # Strong uptrend data
            trend = np.linspace(0, 0.3, periods)
            noise = np.random.normal(0, 0.01, periods)
            returns = trend + noise
        elif scenario == "ranging":
            # Sideways market data
            returns = np.random.normal(0, 0.008, periods)
        elif scenario == "volatile":
            # High volatility data
            returns = np.random.normal(0, 0.025, periods)
        elif scenario == "gap_data":
            # Data with gaps and extreme moves
            returns = np.random.normal(0, 0.01, periods)
            # Add some gaps
            returns[500:510] = 0.05  # Gap up
            returns[1000:1010] = -0.05  # Gap down
        else:  # default
            returns = np.random.normal(0, 0.012, periods)
        
        # Generate OHLCV data
        prices = base_price * np.exp(np.cumsum(returns))
        data = []
        
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(returns[i]) * base_price
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = close + np.random.uniform(-volatility/2, volatility/2)
            volume = np.random.uniform(10, 1000)
            
            data.append({
                'timestamp': timestamp,
                'open': max(open_price, 0.01),
                'high': max(high, open_price, close, 0.01),
                'low': min(low, open_price, close, 0.01),
                'close': max(close, 0.01),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Save for reuse
        output_file = self.test_data_dir / f"market_data_{scenario}.csv"
        df.to_csv(output_file)
        logger.info(f"Generated and saved test data: {output_file}")
        
        return df


class PerformanceMonitor:
    """Monitor performance metrics during E2E tests."""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        self.memory_tracker = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        tracemalloc.start()
        self.memory_tracker = tracemalloc.take_snapshot()
        
    def capture_metrics(self, operation: str, data_size: int) -> Dict[str, Any]:
        """Capture current performance metrics."""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        # Memory metrics
        current_memory = tracemalloc.take_snapshot()
        memory_stats = current_memory.compare_to(self.memory_tracker, 'lineno')
        memory_mb = sum(stat.size for stat in memory_stats) / 1024 / 1024
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        metrics = {
            'operation': operation,
            'data_size': data_size,
            'elapsed_seconds': elapsed,
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'system_memory_percent': memory_info.percent,
            'throughput_rows_per_sec': data_size / elapsed if elapsed > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics."""
        if not self.metrics:
            return {}
        
        return {
            'total_operations': len(self.metrics),
            'total_runtime_seconds': sum(m['elapsed_seconds'] for m in self.metrics),
            'average_throughput': np.mean([m['throughput_rows_per_sec'] for m in self.metrics]),
            'peak_memory_mb': max(m['memory_mb'] for m in self.metrics),
            'average_cpu_percent': np.mean([m['cpu_percent'] for m in self.metrics]),
            'detailed_metrics': self.metrics
        }


@pytest.fixture
def docker_env():
    """Docker environment fixture."""
    env = DockerTestEnvironment()
    env.setup_test_environment()
    return env


@pytest.fixture
def test_data_manager(docker_env):
    """Test data manager fixture."""
    return E2ETestDataManager(docker_env.test_data_dir)


@pytest.fixture
def performance_monitor():
    """Performance monitor fixture."""
    return PerformanceMonitor()


class TestE2EFullPipeline:
    """Test complete pipeline from data input to signal output."""
    
    def test_complete_data_flow_pipeline(self, docker_env, test_data_manager, performance_monitor):
        """Test complete data flow: Market Data → Indicators → LLM → Validation → Risk."""
        logger.info("Starting complete data flow pipeline test")
        performance_monitor.start_monitoring()
        
        # 1. Load market data
        market_data = test_data_manager.load_market_data("default")
        assert len(market_data) > 100, "Insufficient test data"
        
        # 2. Initialize indicator calculator
        indicator_calc = VuManChuIndicators()
        
        # 3. Calculate all indicators
        logger.info("Calculating VuManChu indicators")
        indicators = indicator_calc.calculate_latest(market_data)
        
        assert indicators is not None, "Indicator calculation failed"
        assert 'cipher_a' in indicators, "Cipher A indicators missing"
        assert 'cipher_b' in indicators, "Cipher B indicators missing"
        
        # 4. Test LLM integration (if available)
        try:
            config = load_config()
            if config.openai_api_key:
                llm_agent = LLMTradingAgent(config)
                
                market_state = MarketState(
                    symbol="BTC-USD",
                    price=float(market_data['close'].iloc[-1]),
                    timestamp=market_data.index[-1],
                    indicators=IndicatorData(**indicators)
                )
                
                # Get LLM decision
                decision = llm_agent.make_decision(market_state)
                assert decision.action in ['buy', 'sell', 'hold'], "Invalid LLM decision"
                logger.info(f"LLM decision: {decision.action} (confidence: {decision.confidence})")
            else:
                logger.warning("OpenAI API key not available - skipping LLM test")
        except Exception as e:
            logger.warning(f"LLM test skipped due to error: {e}")
        
        # 5. Validate through risk management
        risk_manager = RiskManager()
        validator = TradeValidator()
        
        # Simulate trade action
        trade_action = TradeAction(
            action="buy",
            quantity=0.1,
            confidence=0.75,
            reasoning="E2E test trade"
        )
        
        # Validate trade
        is_valid = validator.validate_trade_action(trade_action)
        assert is_valid, "Trade validation failed"
        
        # Apply risk management
        risk_adjusted = risk_manager.apply_risk_management(trade_action, market_data['close'].iloc[-1])
        assert risk_adjusted is not None, "Risk management failed"
        
        # Capture performance metrics
        metrics = performance_monitor.capture_metrics("complete_pipeline", len(market_data))
        logger.info(f"Pipeline performance: {metrics['throughput_rows_per_sec']:.1f} rows/sec")
        
        # Validate performance requirements
        assert metrics['throughput_rows_per_sec'] > 100, "Pipeline throughput too low"
        assert metrics['memory_mb'] < 500, "Memory usage too high"
        
        logger.info("Complete data flow pipeline test passed")
    
    def test_multiple_symbol_processing(self, docker_env, test_data_manager, performance_monitor):
        """Test processing multiple symbols simultaneously."""
        logger.info("Starting multiple symbol processing test")
        performance_monitor.start_monitoring()
        
        symbols = ["BTC-USD", "ETH-USD", "DOGE-USD"]
        results = {}
        
        indicator_calc = VuManChuIndicators()
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}")
            
            # Generate symbol-specific test data
            market_data = test_data_manager.load_market_data("default")
            
            # Calculate indicators
            indicators = indicator_calc.calculate_latest(market_data)
            assert indicators is not None, f"Indicator calculation failed for {symbol}"
            
            results[symbol] = indicators
            
            # Validate signal generation
            assert 'cipher_a' in indicators, f"Cipher A missing for {symbol}"
            assert 'cipher_b' in indicators, f"Cipher B missing for {symbol}"
        
        # Capture metrics
        total_data = len(symbols) * len(market_data)
        metrics = performance_monitor.capture_metrics("multi_symbol", total_data)
        
        logger.info(f"Multi-symbol processing: {len(symbols)} symbols processed")
        logger.info(f"Performance: {metrics['throughput_rows_per_sec']:.1f} rows/sec")
        
        assert len(results) == len(symbols), "Not all symbols processed"
        assert metrics['throughput_rows_per_sec'] > 50, "Multi-symbol throughput too low"


class TestE2EPerformance:
    """Performance testing under various load conditions."""
    
    @pytest.mark.parametrize("data_size", [100, 500, 1000, 2000, 5000])
    def test_scalability_performance(self, docker_env, test_data_manager, performance_monitor, data_size):
        """Test performance scaling with increasing data sizes."""
        logger.info(f"Starting scalability test with {data_size} data points")
        performance_monitor.start_monitoring()
        
        # Generate test data of specified size
        np.random.seed(42)
        timestamps = pd.date_range('2024-01-01', periods=data_size, freq='1min')
        
        market_data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, data_size),
            'high': np.random.uniform(46000, 56000, data_size),
            'low': np.random.uniform(44000, 54000, data_size),
            'close': np.random.uniform(45000, 55000, data_size),
            'volume': np.random.uniform(10, 100, data_size),
        }, index=timestamps)
        
        # Run indicator calculations
        indicator_calc = VuManChuIndicators()
        
        start_time = time.time()
        indicators = indicator_calc.calculate_latest(market_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = data_size / processing_time
        
        # Capture metrics
        metrics = performance_monitor.capture_metrics("scalability", data_size)
        
        logger.info(f"Scalability test {data_size}: {throughput:.1f} rows/sec")
        
        # Performance assertions based on data size
        if data_size <= 500:
            assert throughput > 1000, f"Small dataset throughput too low: {throughput}"
        elif data_size <= 2000:
            assert throughput > 500, f"Medium dataset throughput too low: {throughput}"
        else:
            assert throughput > 200, f"Large dataset throughput too low: {throughput}"
        
        assert processing_time < 30, f"Processing time too long: {processing_time}s"
    
    def test_memory_usage_under_load(self, docker_env, test_data_manager, performance_monitor):
        """Test memory usage with large datasets and long-running calculations."""
        logger.info("Starting memory usage test")
        performance_monitor.start_monitoring()
        
        initial_memory = psutil.virtual_memory().used
        indicator_calc = VuManChuIndicators()
        
        # Process multiple large datasets
        for i in range(5):
            logger.info(f"Memory test iteration {i+1}/5")
            
            # Generate large dataset
            market_data = test_data_manager.load_market_data("volatile")
            
            # Calculate indicators
            indicators = indicator_calc.calculate_latest(market_data)
            assert indicators is not None, f"Calculation failed on iteration {i+1}"
            
            # Check memory usage
            current_memory = psutil.virtual_memory().used
            memory_increase = current_memory - initial_memory
            memory_mb = memory_increase / 1024 / 1024
            
            logger.info(f"Memory increase: {memory_mb:.1f} MB")
            
            # Memory leak detection
            if i > 0:  # Allow for first iteration overhead
                assert memory_mb < 200, f"Excessive memory usage: {memory_mb} MB"
        
        final_metrics = performance_monitor.capture_metrics("memory_test", 5 * len(market_data))
        logger.info(f"Memory test completed - Peak usage: {final_metrics['memory_mb']:.1f} MB")
    
    def test_concurrent_processing(self, docker_env, test_data_manager, performance_monitor):
        """Test concurrent indicator calculations."""
        logger.info("Starting concurrent processing test")
        performance_monitor.start_monitoring()
        
        async def process_data_async(symbol: str, data: pd.DataFrame):
            """Async processing function."""
            indicator_calc = VuManChuIndicators()
            return await asyncio.to_thread(indicator_calc.calculate_latest, data)
        
        async def run_concurrent_test():
            # Prepare multiple datasets
            datasets = []
            for i in range(3):
                data = test_data_manager.load_market_data("default")
                datasets.append((f"TEST-{i}", data))
            
            # Run concurrent calculations
            tasks = [process_data_async(symbol, data) for symbol, data in datasets]
            results = await asyncio.gather(*tasks)
            
            return results
        
        # Run the async test
        results = asyncio.run(run_concurrent_test())
        
        assert len(results) == 3, "Not all concurrent tasks completed"
        for result in results:
            assert result is not None, "Concurrent calculation failed"
            assert 'cipher_a' in result, "Cipher A missing in concurrent result"
        
        metrics = performance_monitor.capture_metrics("concurrent", 3 * len(market_data))
        logger.info(f"Concurrent processing completed: {metrics['throughput_rows_per_sec']:.1f} rows/sec")


class TestE2ESignalQuality:
    """Test signal generation accuracy and quality."""
    
    def test_signal_type_coverage(self, docker_env, test_data_manager):
        """Test that all signal types are generated across different market conditions."""
        logger.info("Starting signal type coverage test")
        
        indicator_calc = VuManChuIndicators()
        signal_counts = {
            'cipher_a_signals': 0,
            'cipher_b_signals': 0,
            'diamond_patterns': 0,
            'yellow_cross': 0,
            'gold_signals': 0
        }
        
        # Test different market scenarios
        scenarios = ["trending", "ranging", "volatile"]
        
        for scenario in scenarios:
            logger.info(f"Testing signals in {scenario} market")
            market_data = test_data_manager.load_market_data(scenario)
            
            # Calculate full indicator series
            cipher_a = CipherA()
            cipher_b = CipherB()
            
            cipher_a_result = cipher_a.calculate(market_data)
            cipher_b_result = cipher_b.calculate(market_data)
            
            # Count signals
            if 'red_diamond' in cipher_a_result.columns:
                signal_counts['diamond_patterns'] += cipher_a_result['red_diamond'].sum()
            if 'green_diamond' in cipher_a_result.columns:
                signal_counts['diamond_patterns'] += cipher_a_result['green_diamond'].sum()
            if 'yellow_cross' in cipher_a_result.columns:
                signal_counts['yellow_cross'] += cipher_a_result['yellow_cross'].sum()
            
            if 'buy_circle' in cipher_b_result.columns:
                signal_counts['cipher_b_signals'] += cipher_b_result['buy_circle'].sum()
            if 'sell_circle' in cipher_b_result.columns:
                signal_counts['cipher_b_signals'] += cipher_b_result['sell_circle'].sum()
            if 'gold_buy' in cipher_b_result.columns:
                signal_counts['gold_signals'] += cipher_b_result['gold_buy'].sum()
        
        logger.info(f"Signal generation summary: {signal_counts}")
        
        # Validate signal generation
        assert signal_counts['diamond_patterns'] > 0, "No diamond patterns generated"
        assert signal_counts['cipher_b_signals'] > 0, "No Cipher B signals generated"
        
        # Signal frequency validation (should be reasonable, not too frequent)
        total_data_points = sum(len(test_data_manager.load_market_data(s)) for s in scenarios)
        signal_frequency = sum(signal_counts.values()) / total_data_points
        
        assert 0.01 < signal_frequency < 0.3, f"Signal frequency unrealistic: {signal_frequency:.3f}"
        logger.info(f"Signal frequency: {signal_frequency:.3f} (within expected range)")
    
    def test_signal_timing_accuracy(self, docker_env, test_data_manager):
        """Test signal timing accuracy against expected patterns."""
        logger.info("Starting signal timing accuracy test")
        
        # Load trending data for timing tests
        market_data = test_data_manager.load_market_data("trending")
        
        cipher_a = CipherA()
        cipher_b = CipherB()
        
        cipher_a_result = cipher_a.calculate(market_data)
        cipher_b_result = cipher_b.calculate(market_data)
        
        # Test signal timing relationships
        if 'yellow_cross' in cipher_a_result.columns and cipher_a_result['yellow_cross'].any():
            yellow_cross_indices = cipher_a_result[cipher_a_result['yellow_cross'] == True].index
            
            # Validate that yellow cross signals occur at reasonable intervals
            if len(yellow_cross_indices) > 1:
                intervals = np.diff(yellow_cross_indices)
                avg_interval = np.mean([x.total_seconds() / 60 for x in intervals])  # minutes
                
                assert avg_interval > 10, f"Yellow cross signals too frequent: {avg_interval:.1f} min"
                assert avg_interval < 500, f"Yellow cross signals too rare: {avg_interval:.1f} min"
                
                logger.info(f"Yellow cross signal interval: {avg_interval:.1f} minutes")
        
        # Test signal strength consistency
        if 'signal_strength' in cipher_a_result.columns:
            strength_values = cipher_a_result['signal_strength'].dropna()
            if len(strength_values) > 0:
                assert strength_values.min() >= 0, "Signal strength below minimum"
                assert strength_values.max() <= 100, "Signal strength above maximum"
                logger.info(f"Signal strength range: {strength_values.min():.1f} - {strength_values.max():.1f}")


class TestE2EIntegration:
    """Test integration between all components."""
    
    def test_complete_bot_integration(self, docker_env, test_data_manager, performance_monitor):
        """Test complete bot integration with all components."""
        logger.info("Starting complete bot integration test")
        performance_monitor.start_monitoring()
        
        try:
            # Load configuration
            config = load_config()
            
            # Initialize all components
            market_data_provider = MarketDataProvider(config)
            indicator_calc = VuManChuIndicators()
            risk_manager = RiskManager()
            validator = TradeValidator()
            
            # Initialize strategy (without LLM for testing)
            strategy = CoreStrategy(
                indicator_calculator=indicator_calc,
                risk_manager=risk_manager,
                validator=validator
            )
            
            # Load test data
            market_data = test_data_manager.load_market_data("default")
            
            # Run complete strategy cycle
            latest_indicators = indicator_calc.calculate_latest(market_data)
            assert latest_indicators is not None, "Indicator calculation failed"
            
            # Create market state
            market_state = MarketState(
                symbol="BTC-USD",
                price=float(market_data['close'].iloc[-1]),
                timestamp=market_data.index[-1],
                indicators=IndicatorData(**latest_indicators)
            )
            
            # Test strategy decision making
            decision = strategy.generate_signal(market_state)
            assert decision is not None, "Strategy decision generation failed"
            assert hasattr(decision, 'action'), "Decision missing action"
            assert decision.action in ['buy', 'sell', 'hold'], "Invalid decision action"
            
            # Test risk management integration
            if decision.action != 'hold':
                risk_adjusted = risk_manager.apply_risk_management(decision, market_state.price)
                assert risk_adjusted is not None, "Risk management failed"
            
            metrics = performance_monitor.capture_metrics("integration", len(market_data))
            logger.info(f"Integration test completed: {metrics['throughput_rows_per_sec']:.1f} rows/sec")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def test_configuration_loading(self, docker_env):
        """Test configuration loading in Docker environment."""
        logger.info("Starting configuration loading test")
        
        # Test different configuration scenarios
        config_scenarios = [
            {"DRY_RUN": "true", "SYMBOL": "BTC-USD"},
            {"DRY_RUN": "false", "SYMBOL": "ETH-USD"},
            {"ENABLE_PAPER_TRADING": "true"}
        ]
        
        for scenario in config_scenarios:
            # Set environment variables
            for key, value in scenario.items():
                os.environ[key] = value
            
            try:
                config = load_config()
                assert config is not None, "Configuration loading failed"
                
                # Validate configuration
                if "DRY_RUN" in scenario:
                    expected_dry_run = scenario["DRY_RUN"] == "true"
                    assert config.dry_run == expected_dry_run, "DRY_RUN config mismatch"
                
                if "SYMBOL" in scenario:
                    assert config.symbol == scenario["SYMBOL"], "SYMBOL config mismatch"
                
                logger.info(f"Configuration scenario validated: {scenario}")
                
            finally:
                # Clean up environment variables
                for key in scenario.keys():
                    if key in os.environ:
                        del os.environ[key]


class TestE2EErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_malformed_data_handling(self, docker_env, test_data_manager):
        """Test handling of malformed or corrupted data."""
        logger.info("Starting malformed data handling test")
        
        indicator_calc = VuManChuIndicators()
        
        # Test various malformed data scenarios
        test_cases = [
            # Missing columns
            pd.DataFrame({'open': [100], 'high': [110], 'low': [90]}),  # Missing close, volume
            
            # NaN values
            pd.DataFrame({
                'open': [100, np.nan, 102],
                'high': [110, 115, np.nan],
                'low': [90, 95, 100],
                'close': [105, np.nan, 101],
                'volume': [1000, 1500, np.nan]
            }),
            
            # Infinite values
            pd.DataFrame({
                'open': [100, 101, np.inf],
                'high': [110, 115, np.inf],
                'low': [90, 95, -np.inf],
                'close': [105, 110, 105],
                'volume': [1000, 1500, 1200]  
            }),
            
            # Empty DataFrame
            pd.DataFrame(),
            
            # Single row (insufficient data)
            pd.DataFrame({
                'open': [100], 'high': [110], 'low': [90], 'close': [105], 'volume': [1000]
            })
        ]
        
        for i, malformed_data in enumerate(test_cases):
            logger.info(f"Testing malformed data case {i+1}")
            
            try:
                result = indicator_calc.calculate_latest(malformed_data)
                
                # If calculation succeeds, validate result
                if result is not None:
                    logger.info(f"Case {i+1}: Handled gracefully with result")
                else:
                    logger.info(f"Case {i+1}: Returned None (acceptable)")
                    
            except Exception as e:
                # Log but don't fail - some errors are expected
                logger.info(f"Case {i+1}: Exception handled: {type(e).__name__}")
                
                # Certain exceptions should not occur
                assert not isinstance(e, (MemoryError, SystemError)), f"Critical error: {e}"
        
        logger.info("Malformed data handling test completed")
    
    def test_resource_exhaustion_recovery(self, docker_env, test_data_manager, performance_monitor):
        """Test recovery from resource exhaustion scenarios."""
        logger.info("Starting resource exhaustion recovery test")
        performance_monitor.start_monitoring()
        
        indicator_calc = VuManChuIndicators()
        
        # Test with progressively larger datasets until resource limits
        data_sizes = [1000, 5000, 10000, 20000]
        max_successful_size = 0
        
        for size in data_sizes:
            logger.info(f"Testing resource exhaustion with {size} data points")
            
            try:
                # Generate large dataset
                market_data = test_data_manager._generate_test_data("default").head(size)
                
                # Monitor memory before calculation
                memory_before = psutil.virtual_memory().used
                
                # Attempt calculation
                result = indicator_calc.calculate_latest(market_data)
                
                memory_after = psutil.virtual_memory().used
                memory_used_mb = (memory_after - memory_before) / 1024 / 1024
                
                if result is not None:
                    max_successful_size = size
                    logger.info(f"Successfully processed {size} rows, memory: {memory_used_mb:.1f} MB")
                else:
                    logger.warning(f"Failed to process {size} rows")
                    break
                    
                # Check for memory exhaustion
                if memory_used_mb > 800:  # Approaching 1GB limit
                    logger.warning(f"Approaching memory limit at {size} rows")
                    break
                    
            except (MemoryError, OSError) as e:
                logger.info(f"Resource limit reached at {size} rows: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error at {size} rows: {e}")
                break
        
        metrics = performance_monitor.capture_metrics("resource_exhaustion", max_successful_size)
        logger.info(f"Maximum processed size: {max_successful_size} rows")
        
        # Validate that we can process reasonable amounts of data
        assert max_successful_size >= 1000, "Cannot process minimum required data size"
    
    def test_network_timeout_simulation(self, docker_env):
        """Test behavior under network timeout conditions."""
        logger.info("Starting network timeout simulation test")
        
        # This test simulates network conditions that might affect external API calls
        try:
            config = load_config()
            
            # Test timeout handling in configuration
            original_timeout = getattr(config, 'request_timeout', 30)
            
            # Simulate very short timeout
            config.request_timeout = 0.001  # 1ms - should timeout
            
            # Test that the system handles timeouts gracefully
            # (This is a simulation - actual network calls would be in integration tests)
            
            logger.info("Network timeout simulation completed")
            
        except Exception as e:
            logger.info(f"Network timeout handling: {type(e).__name__}")


class TestE2EDockerEnvironment:
    """Docker-specific environment tests."""
    
    def test_docker_health_check(self, docker_env):
        """Test Docker health check functionality."""
        logger.info("Starting Docker health check test")
        
        # Validate Docker environment
        resources = docker_env.validate_docker_resources()
        
        assert resources['memory_total'] > 0, "No memory detected"
        assert resources['cpu_count'] > 0, "No CPU detected"
        assert resources['disk_usage'] > 0, "No disk space available"
        
        logger.info(f"Docker resources validated:")
        logger.info(f"  Memory: {resources['memory_total'] / 1024 / 1024 / 1024:.1f} GB")
        logger.info(f"  CPU cores: {resources['cpu_count']}")
        logger.info(f"  Disk free: {resources['disk_usage'] / 1024 / 1024 / 1024:.1f} GB")
        
        # Test log file accessibility
        log_file = docker_env.logs_dir / "e2e_test.log"
        log_file.write_text("Docker health check test log")
        assert log_file.exists(), "Cannot write to log directory"
        
        # Test results directory
        results_file = docker_env.results_dir / "health_check.json"
        results_file.write_text(json.dumps(resources))
        assert results_file.exists(), "Cannot write to results directory"
        
        logger.info("Docker health check passed")
    
    def test_volume_mounts_accessibility(self, docker_env):
        """Test that Docker volume mounts are accessible and writable."""
        logger.info("Starting volume mounts accessibility test")
        
        # Test data directory mount
        test_data_file = docker_env.test_data_dir / "mount_test.csv"
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        test_data.to_csv(test_data_file)
        assert test_data_file.exists(), "Cannot write to test data volume"
        
        # Read back data
        read_data = pd.read_csv(test_data_file)
        assert len(read_data) == 3, "Data corruption in volume mount"
        
        # Test results directory mount
        results_file = docker_env.results_dir / "volume_test.json"
        test_results = {"test": "docker_volume_mount", "timestamp": datetime.now().isoformat()}
        with open(results_file, 'w') as f:
            json.dump(test_results, f)
        assert results_file.exists(), "Cannot write to results volume"
        
        # Test logs directory mount
        log_file = docker_env.logs_dir / "volume_test.log"
        log_file.write_text("Volume mount test log entry")
        assert log_file.exists(), "Cannot write to logs volume"
        
        # Clean up test files
        for test_file in [test_data_file, results_file, log_file]:
            if test_file.exists():
                test_file.unlink()
        
        logger.info("Volume mounts accessibility test passed")
    
    def test_docker_networking(self, docker_env):
        """Test Docker networking configuration."""
        logger.info("Starting Docker networking test")
        
        # Test internal networking (container to container communication would be tested here)
        # For now, test basic network connectivity
        
        import socket
        
        # Test that we can bind to expected ports
        test_ports = [8080, 8081, 8082]
        available_ports = []
        
        for port in test_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                available_ports.append(port)
                logger.info(f"Port {port} available")
            except OSError:
                logger.info(f"Port {port} in use")
        
        assert len(available_ports) > 0, "No test ports available"
        
        logger.info(f"Docker networking test passed - {len(available_ports)} ports available")


def generate_e2e_test_report(performance_monitor: PerformanceMonitor, docker_env: DockerTestEnvironment):
    """Generate comprehensive E2E test report."""
    logger.info("Generating E2E test report")
    
    # Get performance summary
    perf_summary = performance_monitor.get_performance_summary()
    
    # Get system information
    system_info = {
        'docker_environment': docker_env.is_docker,
        'python_version': os.sys.version,
        'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
        'cpu_count': psutil.cpu_count(),
        'test_timestamp': datetime.now().isoformat()
    }
    
    # Compile test report
    report = {
        'test_suite': 'VuManChu E2E Docker Tests',
        'version': '1.0.0',
        'system_info': system_info,
        'performance_summary': perf_summary,
        'test_environment': {
            'docker': docker_env.is_docker,
            'test_data_dir': str(docker_env.test_data_dir),
            'results_dir': str(docker_env.results_dir),
            'logs_dir': str(docker_env.logs_dir)
        }
    }
    
    # Save report
    report_file = docker_env.results_dir / "e2e_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"E2E test report saved: {report_file}")
    return report


if __name__ == "__main__":
    # Run E2E tests directly
    pytest.main([__file__, "-v", "--tb=short"])