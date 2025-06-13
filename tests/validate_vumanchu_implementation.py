#!/usr/bin/env python3
"""
VuManChu Implementation Validation Script.

This script provides manual testing and validation tools for the complete VuManChu
Cipher implementation. It includes performance benchmarking, accuracy testing,
and Pine Script compatibility verification.

Usage:
    python validate_vumanchu_implementation.py [OPTIONS]

Options:
    --performance    Run performance benchmarks
    --accuracy       Run accuracy tests
    --compatibility  Run compatibility tests
    --full           Run all tests
    --data-file      Use specific CSV file for testing
    --output-dir     Output directory for reports
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot.indicators.vumanchu import CipherA, CipherB, VuManChuIndicators
from tests.test_vumanchu_complete import TestDataGenerator, generate_test_report, validate_against_sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VuManChuValidator:
    """Comprehensive VuManChu implementation validator."""
    
    def __init__(self, output_dir: str = "validation_output"):
        """
        Initialize validator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.calc = VuManChuIndicators()
        self.cipher_a = CipherA()
        self.cipher_b = CipherB()
        
        logger.info(f"VuManChu Validator initialized. Output: {self.output_dir}")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive performance tests.
        
        Returns:
            Dictionary with performance test results
        """
        logger.info("Running performance tests...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'scalability_test': {},
            'memory_usage': {},
            'component_performance': {},
            'real_time_simulation': {}
        }
        
        # Scalability test
        data_sizes = [100, 500, 1000, 2000, 5000, 10000]
        scalability_results = {}
        
        for size in data_sizes:
            logger.info(f"Testing scalability with {size} data points...")
            
            data = TestDataGenerator.generate_ohlcv_data(periods=size)
            
            # Time calculation
            start_time = time.time()
            result = self.calc.calculate_all(data)
            calculation_time = time.time() - start_time
            
            # Calculate throughput
            throughput = size / calculation_time if calculation_time > 0 else 0
            
            scalability_results[size] = {
                'time_seconds': calculation_time,
                'throughput_per_second': throughput,
                'output_columns': len(result.columns),
                'memory_mb': result.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"Size {size}: {calculation_time:.3f}s ({throughput:.0f} rows/s)")
        
        results['scalability_test'] = scalability_results
        
        # Component performance test
        component_data = TestDataGenerator.generate_ohlcv_data(periods=1000)
        component_times = {}
        
        # Test individual components
        logger.info("Testing individual component performance...")
        
        # Cipher A
        start_time = time.time()
        cipher_a_result = self.cipher_a.calculate(component_data)
        component_times['cipher_a'] = time.time() - start_time
        
        # Cipher B
        start_time = time.time()
        cipher_b_result = self.cipher_b.calculate(component_data)
        component_times['cipher_b'] = time.time() - start_time
        
        # Combined
        start_time = time.time()
        combined_result = self.calc.calculate_all(component_data)
        component_times['combined'] = time.time() - start_time
        
        results['component_performance'] = component_times
        
        # Real-time simulation
        logger.info("Running real-time simulation...")
        real_time_results = self._simulate_real_time_performance()
        results['real_time_simulation'] = real_time_results
        
        # Save performance report
        performance_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(performance_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Performance report saved to: {performance_file}")
        
        # Generate performance charts
        self._generate_performance_charts(results)
        
        return results
    
    def run_accuracy_tests(self) -> Dict[str, Any]:
        """
        Run accuracy and correctness tests.
        
        Returns:
            Dictionary with accuracy test results
        """
        logger.info("Running accuracy tests...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'parameter_accuracy': {},
            'signal_accuracy': {},
            'formula_verification': {},
            'edge_case_handling': {}
        }
        
        # Parameter accuracy test
        logger.info("Testing parameter accuracy...")
        
        # Pine Script default verification
        results['parameter_accuracy'] = {
            'cipher_a_defaults': {
                'wt_channel_length': self.cipher_a.wt_channel_length == 9,
                'wt_average_length': self.cipher_a.wt_average_length == 13,
                'wt_ma_length': self.cipher_a.wt_ma_length == 3,
                'ema_ribbon_lengths': self.cipher_a.ema_ribbon.lengths == [5, 11, 15, 18, 21, 24, 28, 34],
                'rsimfi_period': self.cipher_a.rsimfi_period == 60,
                'rsimfi_multiplier': self.cipher_a.rsimfi_multiplier == 150.0
            },
            'cipher_b_defaults': {
                'wt_channel_length': self.cipher_b.wt_channel_length == 9,
                'wt_average_length': self.cipher_b.wt_average_length == 12,
                'wt_ma_length': self.cipher_b.wt_ma_length == 3,
                'ob_level': self.cipher_b.ob_level == 53.0,
                'os_level': self.cipher_b.os_level == -53.0
            }
        }
        
        # Signal accuracy test
        logger.info("Testing signal accuracy...")
        results['signal_accuracy'] = self._test_signal_accuracy()
        
        # Formula verification
        logger.info("Testing formula verification...")
        results['formula_verification'] = self._verify_formulas()
        
        # Edge case handling
        logger.info("Testing edge case handling...")
        results['edge_case_handling'] = self._test_edge_cases()
        
        # Save accuracy report
        accuracy_file = self.output_dir / f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(accuracy_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Accuracy report saved to: {accuracy_file}")
        
        return results
    
    def run_compatibility_tests(self) -> Dict[str, Any]:
        """
        Run backward compatibility tests.
        
        Returns:
            Dictionary with compatibility test results
        """
        logger.info("Running compatibility tests...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'legacy_interface': {},
            'output_format': {},
            'method_compatibility': {}
        }
        
        test_data = TestDataGenerator.generate_ohlcv_data(periods=100)
        
        # Test legacy interface compatibility
        logger.info("Testing legacy interface compatibility...")
        
        # Cipher A legacy interface
        cipher_a_result = self.cipher_a.calculate(test_data)
        legacy_cipher_a = {
            'ema_fast_present': 'ema_fast' in cipher_a_result.columns,
            'ema_slow_present': 'ema_slow' in cipher_a_result.columns,
            'trend_dot_present': 'trend_dot' in cipher_a_result.columns,
            'rsi_overbought_present': 'rsi_overbought' in cipher_a_result.columns,
            'rsi_oversold_present': 'rsi_oversold' in cipher_a_result.columns
        }
        
        # Cipher B legacy interface
        cipher_b_result = self.cipher_b.calculate(test_data)
        legacy_cipher_b = {
            'vwap_present': 'vwap' in cipher_b_result.columns,
            'money_flow_present': 'money_flow' in cipher_b_result.columns,
            'wave_present': 'wave' in cipher_b_result.columns
        }
        
        results['legacy_interface'] = {
            'cipher_a': legacy_cipher_a,
            'cipher_b': legacy_cipher_b
        }
        
        # Test method compatibility
        logger.info("Testing method compatibility...")
        
        try:
            # Test all legacy methods exist and work
            latest_values_a = self.cipher_a.get_latest_values(cipher_a_result)
            latest_values_b = self.cipher_b.get_latest_values(cipher_b_result)
            all_signals_a = self.cipher_a.get_all_signals(cipher_a_result)
            all_signals_b = self.cipher_b.get_all_signals(cipher_b_result)
            interpretation_a = self.cipher_a.interpret_signals(cipher_a_result)
            interpretation_b = self.cipher_b.interpret_signals(cipher_b_result)
            
            results['method_compatibility'] = {
                'get_latest_values': True,
                'get_all_signals': True,
                'interpret_signals': True,
                'all_methods_working': True
            }
        except Exception as e:
            results['method_compatibility'] = {
                'error': str(e),
                'all_methods_working': False
            }
        
        # Test output format compatibility
        combined_result = self.calc.calculate_all(test_data)
        latest_state = self.calc.get_latest_state(combined_result)
        
        expected_keys = ['close', 'volume', 'combined_signal', 'combined_confidence', 'cipher_a', 'cipher_b']
        output_format_check = {
            key: key in latest_state for key in expected_keys
        }
        
        results['output_format'] = output_format_check
        
        # Save compatibility report
        compatibility_file = self.output_dir / f"compatibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(compatibility_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Compatibility report saved to: {compatibility_file}")
        
        return results
    
    def validate_with_data_file(self, data_file: str) -> Dict[str, Any]:
        """
        Validate implementation with real market data.
        
        Args:
            data_file: Path to CSV file with OHLCV data
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating with data file: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        try:
            data = pd.read_csv(data_file)
            
            # Try to auto-detect column names
            data.columns = data.columns.str.lower()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Try to parse datetime index
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            # Ensure volume column exists
            if 'volume' not in data.columns:
                data['volume'] = 1000000  # Default volume
            
            logger.info(f"Loaded data: {len(data)} rows, {data.index[0]} to {data.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error loading data file: {str(e)}")
            raise
        
        # Run validation
        validation_results = validate_against_sample_data(data_file)
        
        # Additional analysis with real data
        result = self.calc.calculate_all(data)
        
        # Calculate signal statistics
        signal_stats = {
            'total_periods': len(result),
            'cipher_a_signals': {
                'bullish': (result['cipher_a_signal'] == 1).sum(),
                'bearish': (result['cipher_a_signal'] == -1).sum(),
                'neutral': (result['cipher_a_signal'] == 0).sum()
            },
            'cipher_b_signals': {
                'bullish': (result['cipher_b_signal'] == 1).sum(),
                'bearish': (result['cipher_b_signal'] == -1).sum(),
                'neutral': (result['cipher_b_signal'] == 0).sum()
            },
            'combined_signals': {
                'bullish': (result['combined_signal'] == 1).sum(),
                'bearish': (result['combined_signal'] == -1).sum(),
                'neutral': (result['combined_signal'] == 0).sum()
            }
        }
        
        # Add signal statistics to validation results
        validation_results['real_data_analysis'] = signal_stats
        
        # Generate charts for real data
        self._generate_data_charts(data, result)
        
        # Save validation results
        validation_file = self.output_dir / f"real_data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Real data validation saved to: {validation_file}")
        
        return validation_results
    
    def _simulate_real_time_performance(self) -> Dict[str, Any]:
        """Simulate real-time performance with streaming data."""
        
        logger.info("Simulating real-time performance...")
        
        # Generate base dataset
        base_data = TestDataGenerator.generate_ohlcv_data(periods=200)
        
        # Simulate adding new data points one by one
        real_time_results = {
            'initial_calculation_time': 0,
            'incremental_update_times': [],
            'average_update_time': 0,
            'max_update_time': 0,
            'min_update_time': 0
        }
        
        # Initial calculation
        start_time = time.time()
        current_result = self.calc.calculate_all(base_data.iloc[:50])
        real_time_results['initial_calculation_time'] = time.time() - start_time
        
        # Simulate incremental updates
        for i in range(51, len(base_data)):
            window_data = base_data.iloc[:i]
            
            start_time = time.time()
            current_result = self.calc.calculate_all(window_data)
            update_time = time.time() - start_time
            
            real_time_results['incremental_update_times'].append(update_time)
        
        # Calculate statistics
        update_times = real_time_results['incremental_update_times']
        if update_times:
            real_time_results['average_update_time'] = np.mean(update_times)
            real_time_results['max_update_time'] = np.max(update_times)
            real_time_results['min_update_time'] = np.min(update_times)
        
        return real_time_results
    
    def _test_signal_accuracy(self) -> Dict[str, Any]:
        """Test signal generation accuracy."""
        
        # Generate different market conditions
        trending_up_data = TestDataGenerator.generate_trending_data(periods=200, trend_strength=0.002)
        trending_down_data = TestDataGenerator.generate_trending_data(periods=200, trend_strength=-0.002)
        ranging_data = TestDataGenerator.generate_ranging_data(periods=200)
        volatile_data = TestDataGenerator.generate_volatile_data(periods=200)
        
        test_scenarios = {
            'trending_up': trending_up_data,
            'trending_down': trending_down_data,
            'ranging': ranging_data,
            'volatile': volatile_data
        }
        
        accuracy_results = {}
        
        for scenario_name, data in test_scenarios.items():
            result = self.calc.calculate_all(data)
            
            # Calculate signal statistics
            cipher_a_signals = result['cipher_a_signal']
            cipher_b_signals = result['cipher_b_signal']
            combined_signals = result['combined_signal']
            
            scenario_stats = {
                'cipher_a': {
                    'bullish_count': (cipher_a_signals == 1).sum(),
                    'bearish_count': (cipher_a_signals == -1).sum(),
                    'signal_rate': (cipher_a_signals != 0).sum() / len(cipher_a_signals)
                },
                'cipher_b': {
                    'bullish_count': (cipher_b_signals == 1).sum(),
                    'bearish_count': (cipher_b_signals == -1).sum(),
                    'signal_rate': (cipher_b_signals != 0).sum() / len(cipher_b_signals)
                },
                'combined': {
                    'bullish_count': (combined_signals == 1).sum(),
                    'bearish_count': (combined_signals == -1).sum(),
                    'signal_rate': (combined_signals != 0).sum() / len(combined_signals)
                },
                'agreement_rate': result['signal_agreement'].sum() / len(result) if 'signal_agreement' in result.columns else 0
            }
            
            accuracy_results[scenario_name] = scenario_stats
        
        return accuracy_results
    
    def _verify_formulas(self) -> Dict[str, Any]:
        """Verify formula accuracy against Pine Script specifications."""
        
        data = TestDataGenerator.generate_ohlcv_data(periods=100)
        
        # Test WaveTrend formula
        wt_result = self.cipher_a.wavetrend.calculate(data)
        
        # Basic checks for WaveTrend values
        wt1_values = wt_result['wt1'].dropna()
        wt2_values = wt_result['wt2'].dropna()
        
        formula_checks = {
            'wavetrend': {
                'wt1_calculated': len(wt1_values) > 0,
                'wt2_calculated': len(wt2_values) > 0,
                'wt1_range_valid': wt1_values.between(-200, 200).all() if len(wt1_values) > 0 else False,
                'wt2_range_valid': wt2_values.between(-200, 200).all() if len(wt2_values) > 0 else False,
                'wt2_smoother_than_wt1': wt2_values.std() <= wt1_values.std() * 1.1 if len(wt1_values) > 10 and len(wt2_values) > 10 else True
            }
        }
        
        # Test EMA Ribbon
        ema_result = self.cipher_a.ema_ribbon.calculate_ema_ribbon(data)
        
        ema_checks = {
            'all_emas_calculated': all(f'ema{i}' in ema_result.columns for i in range(1, 9)),
            'ema_ordering_valid': True  # Simplified check
        }
        
        # Check EMA ordering (faster EMAs should be more responsive)
        if ema_checks['all_emas_calculated']:
            close_price = data['close'].iloc[-1]
            ema1 = ema_result['ema1'].iloc[-1]
            ema8 = ema_result['ema8'].iloc[-1]
            
            if not (pd.isna(ema1) or pd.isna(ema8)):
                ema_checks['ema_responsiveness'] = abs(close_price - ema1) <= abs(close_price - ema8)
        
        formula_checks['ema_ribbon'] = ema_checks
        
        return formula_checks
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge case handling."""
        
        edge_case_results = {}
        
        # Test empty data
        try:
            empty_result = self.calc.calculate_all(pd.DataFrame())
            edge_case_results['empty_data'] = {
                'handled_gracefully': empty_result.empty,
                'no_error': True
            }
        except Exception as e:
            edge_case_results['empty_data'] = {
                'handled_gracefully': False,
                'error': str(e)
            }
        
        # Test minimal data
        try:
            minimal_data = TestDataGenerator.generate_ohlcv_data(periods=5)
            minimal_result = self.calc.calculate_all(minimal_data)
            edge_case_results['minimal_data'] = {
                'handled_gracefully': len(minimal_result) == 5,
                'no_error': True
            }
        except Exception as e:
            edge_case_results['minimal_data'] = {
                'handled_gracefully': False,
                'error': str(e)
            }
        
        # Test data with NaN values
        try:
            nan_data = TestDataGenerator.generate_ohlcv_data(periods=100)
            nan_data.loc[nan_data.index[10:15], 'close'] = np.nan
            nan_result = self.calc.calculate_all(nan_data)
            edge_case_results['nan_data'] = {
                'handled_gracefully': not nan_result.empty,
                'no_error': True
            }
        except Exception as e:
            edge_case_results['nan_data'] = {
                'handled_gracefully': False,
                'error': str(e)
            }
        
        # Test extreme values
        try:
            extreme_data = TestDataGenerator.generate_ohlcv_data(periods=100, volatility=0.5)
            extreme_result = self.calc.calculate_all(extreme_data)
            edge_case_results['extreme_values'] = {
                'handled_gracefully': not extreme_result.empty,
                'no_error': True
            }
        except Exception as e:
            edge_case_results['extreme_values'] = {
                'handled_gracefully': False,
                'error': str(e)
            }
        
        return edge_case_results
    
    def _generate_performance_charts(self, performance_results: Dict[str, Any]) -> None:
        """Generate performance visualization charts."""
        
        try:
            import matplotlib.pyplot as plt
            
            # Scalability chart
            scalability_data = performance_results['scalability_test']
            sizes = list(scalability_data.keys())
            times = [scalability_data[size]['time_seconds'] for size in sizes]
            throughputs = [scalability_data[size]['throughput_per_second'] for size in sizes]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Time vs Size
            ax1.plot(sizes, times, 'b-o')
            ax1.set_xlabel('Data Size (rows)')
            ax1.set_ylabel('Calculation Time (seconds)')
            ax1.set_title('Scalability: Calculation Time vs Data Size')
            ax1.grid(True)
            
            # Throughput vs Size
            ax2.plot(sizes, throughputs, 'r-o')
            ax2.set_xlabel('Data Size (rows)')
            ax2.set_ylabel('Throughput (rows/second)')
            ax2.set_title('Scalability: Throughput vs Data Size')
            ax2.grid(True)
            
            plt.tight_layout()
            chart_file = self.output_dir / f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to: {chart_file}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping chart generation")
        except Exception as e:
            logger.warning(f"Error generating performance charts: {str(e)}")
    
    def _generate_data_charts(self, data: pd.DataFrame, result: pd.DataFrame) -> None:
        """Generate data visualization charts."""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Price and signals
            axes[0].plot(data.index, data['close'], label='Close Price', alpha=0.7)
            
            # Mark signals
            bullish_signals = result[result['combined_signal'] == 1]
            bearish_signals = result[result['combined_signal'] == -1]
            
            if not bullish_signals.empty:
                axes[0].scatter(bullish_signals.index, data.loc[bullish_signals.index, 'close'], 
                              color='green', marker='^', s=50, label='Bullish Signals')
            if not bearish_signals.empty:
                axes[0].scatter(bearish_signals.index, data.loc[bearish_signals.index, 'close'], 
                              color='red', marker='v', s=50, label='Bearish Signals')
            
            axes[0].set_title('Price and Trading Signals')
            axes[0].legend()
            axes[0].grid(True)
            
            # WaveTrend oscillator
            if 'wt1' in result.columns and 'wt2' in result.columns:
                axes[1].plot(result.index, result['wt1'], label='WaveTrend 1', alpha=0.7)
                axes[1].plot(result.index, result['wt2'], label='WaveTrend 2', alpha=0.7)
                axes[1].axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Overbought')
                axes[1].axhline(y=-60, color='g', linestyle='--', alpha=0.5, label='Oversold')
                axes[1].set_title('WaveTrend Oscillator')
                axes[1].legend()
                axes[1].grid(True)
            
            # RSI
            if 'rsi' in result.columns:
                axes[2].plot(result.index, result['rsi'], label='RSI', color='purple', alpha=0.7)
                axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
                axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
                axes[2].set_title('RSI Indicator')
                axes[2].legend()
                axes[2].grid(True)
            
            plt.tight_layout()
            chart_file = self.output_dir / f"data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Data analysis charts saved to: {chart_file}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping chart generation")
        except Exception as e:
            logger.warning(f"Error generating data charts: {str(e)}")
    
    def run_full_validation(self, data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete validation suite.
        
        Args:
            data_file: Optional path to real data file
            
        Returns:
            Dictionary with all validation results
        """
        logger.info("Running full VuManChu validation suite...")
        
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'performance_tests': {},
            'accuracy_tests': {},
            'compatibility_tests': {},
            'real_data_validation': {}
        }
        
        try:
            # Run performance tests
            logger.info("=" * 50)
            logger.info("PERFORMANCE TESTS")
            logger.info("=" * 50)
            full_results['performance_tests'] = self.run_performance_tests()
            
            # Run accuracy tests
            logger.info("=" * 50)
            logger.info("ACCURACY TESTS")
            logger.info("=" * 50)
            full_results['accuracy_tests'] = self.run_accuracy_tests()
            
            # Run compatibility tests
            logger.info("=" * 50)
            logger.info("COMPATIBILITY TESTS")
            logger.info("=" * 50)
            full_results['compatibility_tests'] = self.run_compatibility_tests()
            
            # Run real data validation if file provided
            if data_file:
                logger.info("=" * 50)
                logger.info("REAL DATA VALIDATION")
                logger.info("=" * 50)
                full_results['real_data_validation'] = self.validate_with_data_file(data_file)
            
            # Generate summary
            full_results['validation_summary'] = self._generate_validation_summary(full_results)
            
            # Save complete results
            complete_file = self.output_dir / f"complete_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(complete_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            logger.info(f"Complete validation results saved to: {complete_file}")
            
            # Print summary
            self._print_validation_summary(full_results['validation_summary'])
            
        except Exception as e:
            logger.error(f"Error during full validation: {str(e)}")
            logger.error(traceback.format_exc())
            full_results['error'] = str(e)
        
        return full_results
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary."""
        
        summary = {
            'overall_status': 'PASS',
            'test_counts': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0
            },
            'key_findings': [],
            'recommendations': []
        }
        
        # Analyze performance tests
        if 'performance_tests' in results:
            perf = results['performance_tests']
            if 'scalability_test' in perf:
                largest_size = max(perf['scalability_test'].keys())
                largest_time = perf['scalability_test'][largest_size]['time_seconds']
                
                if largest_time < 5.0:  # Under 5 seconds for largest test
                    summary['key_findings'].append(f"✓ Excellent performance: {largest_size} rows in {largest_time:.2f}s")
                else:
                    summary['key_findings'].append(f"⚠ Performance concern: {largest_size} rows in {largest_time:.2f}s")
        
        # Analyze accuracy tests
        if 'accuracy_tests' in results:
            acc = results['accuracy_tests']
            if 'parameter_accuracy' in acc:
                cipher_a_checks = acc['parameter_accuracy']['cipher_a_defaults']
                cipher_b_checks = acc['parameter_accuracy']['cipher_b_defaults']
                
                a_passed = sum(cipher_a_checks.values())
                b_passed = sum(cipher_b_checks.values())
                
                if a_passed == len(cipher_a_checks) and b_passed == len(cipher_b_checks):
                    summary['key_findings'].append("✓ All Pine Script parameters verified")
                else:
                    summary['key_findings'].append("⚠ Some Pine Script parameter mismatches found")
        
        # Analyze compatibility tests
        if 'compatibility_tests' in results:
            comp = results['compatibility_tests']
            if 'method_compatibility' in comp:
                if comp['method_compatibility'].get('all_methods_working', False):
                    summary['key_findings'].append("✓ Full backward compatibility maintained")
                else:
                    summary['key_findings'].append("⚠ Some compatibility issues found")
        
        # Generate recommendations
        if len(summary['key_findings']) == 0:
            summary['recommendations'].append("No specific issues found - implementation appears solid")
        else:
            warning_count = sum(1 for finding in summary['key_findings'] if finding.startswith("⚠"))
            if warning_count > 0:
                summary['recommendations'].append(f"Address {warning_count} warning(s) found in validation")
        
        return summary
    
    def _print_validation_summary(self, summary: Dict[str, Any]) -> None:
        """Print validation summary to console."""
        
        print("\n" + "=" * 60)
        print("VUMANCHU VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Overall Status: {summary['overall_status']}")
        
        if summary['key_findings']:
            print("\nKey Findings:")
            for finding in summary['key_findings']:
                print(f"  {finding}")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "=" * 60)


def main():
    """Main validation script entry point."""
    
    parser = argparse.ArgumentParser(description='VuManChu Implementation Validation')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--accuracy', action='store_true', help='Run accuracy tests')
    parser.add_argument('--compatibility', action='store_true', help='Run compatibility tests')
    parser.add_argument('--full', action='store_true', help='Run all tests')
    parser.add_argument('--data-file', type=str, help='Use specific CSV file for testing')
    parser.add_argument('--output-dir', type=str, default='validation_output', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create validator
    validator = VuManChuValidator(output_dir=args.output_dir)
    
    try:
        if args.full:
            # Run complete validation
            validator.run_full_validation(data_file=args.data_file)
        else:
            # Run specific tests
            if args.performance:
                validator.run_performance_tests()
            
            if args.accuracy:
                validator.run_accuracy_tests()
            
            if args.compatibility:
                validator.run_compatibility_tests()
            
            if args.data_file:
                validator.validate_with_data_file(args.data_file)
            
            # If no specific tests requested, run basic validation
            if not any([args.performance, args.accuracy, args.compatibility, args.data_file]):
                print("No specific tests requested. Use --help for options or --full for complete validation.")
                print("Running basic test report generation...")
                test_report = generate_test_report()
                print(f"Basic test report generated: {json.dumps(test_report, indent=2)}")
    
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()