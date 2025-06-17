"""
Unified Indicator Framework for Seamless Multi-Timeframe Trading.

This module provides a comprehensive indicator management system that efficiently 
supports both momentum (1-5 minute) and scalping (15 second - 1 minute) strategies 
with optimized calculations, unified interfaces, and intelligent caching.

Key Features:
- Multi-timeframe indicator management with optimized configurations
- Async calculation engine with dependency graph optimization
- Incremental update system for real-time performance
- Unified interface for all indicators with adapter pattern
- Intelligent caching and performance optimization
- Thread-safe concurrent operations

Architecture:
- UnifiedIndicatorFramework: Main orchestrator
- IndicatorRegistry: Manages indicator registration and instantiation
- MultiTimeframeCalculator: Async calculation engine with optimization
- IncrementalUpdater: Real-time incremental updates
- TimeframeConfigManager: Timeframe-specific configurations
- Performance monitoring and optimization suggestions
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd

# Lazy imports to avoid dependency issues during framework initialization
# These will be imported when needed by the adapters

logger = logging.getLogger(__name__)


class TimeframeType(Enum):
    """Supported timeframe types with specific characteristics."""
    SCALPING = "scalping"      # 15s-1m: Ultra-fast signals, high frequency
    MOMENTUM = "momentum"      # 1m-5m: Trend following, medium frequency
    SWING = "swing"           # 5m-15m: Position trading, low frequency
    POSITION = "position"     # 15m-1h: Long-term trends, very low frequency


class IndicatorType(Enum):
    """Indicator categories for organization and optimization."""
    TREND = "trend"           # EMAs, MAs, trend analysis
    MOMENTUM = "momentum"     # RSI, MACD, Williams %R
    VOLUME = "volume"         # VWAP, OBV, Volume profile
    VOLATILITY = "volatility" # ATR, Bollinger Bands
    CUSTOM = "custom"         # VuManChu, proprietary indicators


@dataclass
class IndicatorConfig:
    """Configuration for a specific indicator."""
    name: str
    type: IndicatorType
    timeframes: List[TimeframeType]
    parameters: Dict[str, Any] = field(default_factory=dict)
    calculation_priority: int = 5  # 1=highest priority, 10=lowest
    cache_duration: int = 60       # Cache duration in seconds
    dependencies: List[str] = field(default_factory=list)
    supports_incremental: bool = False
    memory_efficient: bool = True


@dataclass
class CalculationResult:
    """Result of indicator calculation with metadata."""
    data: Dict[str, Any]
    calculation_time_ms: float
    cache_hit: bool
    timeframe: TimeframeType
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedIndicatorInterface(ABC):
    """Base interface for all indicators in the unified framework."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeframe = config.get('timeframe', TimeframeType.MOMENTUM)
        self.cache_duration = config.get('cache_duration', 60)
        self.supports_incremental = config.get('supports_incremental', False)
        self._last_calculation = None
        self._incremental_state = None
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Standard calculation method - must be implemented by all indicators."""
        pass
        
    async def calculate_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Async calculation method with optional override."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.calculate, data)
        
    def setup_incremental(self, initial_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Setup for incremental updates - override if supported."""
        if self.supports_incremental:
            self._incremental_state = self._initialize_incremental_state(initial_data)
            return self._incremental_state
        return None
        
    def update_incremental(self, state: Dict[str, Any], new_tick: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Incremental update - override if supported."""
        if not self.supports_incremental:
            return state, {}
        return self._update_incremental_state(state, new_tick)
        
    def get_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trading signals from calculation result."""
        return []
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format and requirements."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns) and len(data) > 0
    
    def _initialize_incremental_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Initialize incremental state - override in subclasses."""
        return {}
    
    def _update_incremental_state(self, state: Dict[str, Any], new_tick: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update incremental state - override in subclasses."""
        return state, {}


class TimeframeConfigManager:
    """Manages timeframe-specific configurations for indicators."""
    
    def __init__(self):
        self.timeframe_configs = {
            TimeframeType.SCALPING: {
                'vumanchu_cipher_a': {
                    'wt_channel_length': 3,
                    'wt_average_length': 5,
                    'wt_ma_length': 2,
                    'overbought_level': 45.0,
                    'oversold_level': -45.0,
                    'cache_duration': 15
                },
                'vumanchu_cipher_b': {
                    'wt_channel_length': 3,
                    'wt_average_length': 5,
                    'overbought_level': 45.0,
                    'oversold_level': -45.0,
                    'cache_duration': 15
                },
                'fast_ema': {
                    'periods': [3, 5, 8],
                    'calculation_method': 'incremental',
                    'cache_duration': 10,
                    'supports_incremental': True
                },
                'fast_rsi': {
                    'period': 7,
                    'overbought': 75,
                    'oversold': 25,
                    'cache_duration': 10
                },
                'fast_macd': {
                    'fast': 5,
                    'slow': 10,
                    'signal': 3,
                    'cache_duration': 10
                },
                'williams_r': {
                    'period': 7,
                    'overbought': -20,
                    'oversold': -80,
                    'cache_duration': 10
                },
                'scalping_vwap': {
                    'period': 20,
                    'bands': [1.0, 2.0],
                    'reset_frequency': 'session',
                    'cache_duration': 15
                },
                'volume_profile': {
                    'bins': 50,
                    'period': 100,
                    'cache_duration': 20
                }
            },
            TimeframeType.MOMENTUM: {
                'vumanchu_cipher_a': {
                    'wt_channel_length': 9,
                    'wt_average_length': 13,
                    'wt_ma_length': 3,
                    'overbought_level': 60.0,
                    'oversold_level': -60.0,
                    'cache_duration': 30
                },
                'vumanchu_cipher_b': {
                    'wt_channel_length': 9,
                    'wt_average_length': 13,
                    'overbought_level': 60.0,
                    'oversold_level': -60.0,
                    'cache_duration': 30
                },
                'ema_ribbon': {
                    'periods': [12, 26, 50],
                    'calculation_method': 'vectorized',
                    'cache_duration': 45
                },
                'macd': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9,
                    'cache_duration': 30
                },
                'rsi': {
                    'period': 14,
                    'overbought': 70,
                    'oversold': 30,
                    'cache_duration': 30
                },
                'williams_r': {
                    'period': 14,
                    'overbought': -20,
                    'oversold': -80,
                    'cache_duration': 30
                },
                'vwap': {
                    'period': 50,
                    'bands': [1.0, 2.0, 3.0],
                    'reset_frequency': 'daily',
                    'cache_duration': 45
                },
                'volume_analysis': {
                    'volume_ma_period': 20,
                    'volume_threshold': 1.5,
                    'cache_duration': 30
                }
            },
            TimeframeType.SWING: {
                'vumanchu_cipher_a': {
                    'wt_channel_length': 10,
                    'wt_average_length': 21,
                    'wt_ma_length': 4,
                    'overbought_level': 53.0,
                    'oversold_level': -53.0,
                    'cache_duration': 120
                },
                'ema_ribbon': {
                    'periods': [21, 50, 100, 200],
                    'calculation_method': 'vectorized',
                    'cache_duration': 180
                },
                'macd': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9,
                    'cache_duration': 120
                },
                'rsi': {
                    'period': 14,
                    'overbought': 70,
                    'oversold': 30,
                    'cache_duration': 120
                }
            }
        }
        
    def get_config(self, indicator_name: str, timeframe: TimeframeType) -> Dict[str, Any]:
        """Get timeframe-optimized configuration for indicator."""
        timeframe_config = self.timeframe_configs.get(timeframe, {})
        indicator_config = timeframe_config.get(indicator_name, {})
        
        # Add default timeframe info
        indicator_config['timeframe'] = timeframe
        
        return indicator_config
    
    def get_all_indicators_for_timeframe(self, timeframe: TimeframeType) -> List[str]:
        """Get list of all indicators available for a timeframe."""
        return list(self.timeframe_configs.get(timeframe, {}).keys())


class IndicatorCache:
    """Thread-safe caching system for indicator results."""
    
    def __init__(self, default_ttl: int = 60):
        self._cache = {}
        self._access_times = {}
        self._ttl_times = {}
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        async with self._lock:
            if key not in self._cache:
                return None
                
            # Check if expired
            if key in self._ttl_times and datetime.now() > self._ttl_times[key]:
                await self._remove(key)
                return None
                
            # Update access time
            self._access_times[key] = datetime.now()
            return self._cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached result with TTL."""
        async with self._lock:
            self._cache[key] = value
            self._access_times[key] = datetime.now()
            
            if ttl is None:
                ttl = self.default_ttl
            self._ttl_times[key] = datetime.now() + timedelta(seconds=ttl)
    
    async def _remove(self, key: str) -> None:
        """Remove key from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._ttl_times.pop(key, None)
    
    async def cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, expiry in self._ttl_times.items()
                if now > expiry
            ]
            
            for key in expired_keys:
                await self._remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'keys': list(self._cache.keys()),
            'oldest_access': min(self._access_times.values()) if self._access_times else None,
            'newest_access': max(self._access_times.values()) if self._access_times else None
        }


class UnifiedIndicatorRegistry:
    """Registry for managing indicator configurations and instances."""
    
    def __init__(self):
        self.indicators = {}
        self.dependency_graph = {}
        self.instances = WeakValueDictionary()  # Auto-cleanup unused instances
        
    def register_indicator(self, config: IndicatorConfig, calculator_class: type) -> None:
        """Register an indicator with its configuration and calculator class."""
        self.indicators[config.name] = {
            'config': config,
            'calculator_class': calculator_class,
            'registered_at': datetime.now()
        }
        
        # Update dependency graph
        self._update_dependency_graph(config)
        
        logger.info(f"Registered indicator: {config.name} for timeframes: {[tf.value for tf in config.timeframes]}")
    
    def get_indicator(self, name: str, timeframe: TimeframeType, config_manager: TimeframeConfigManager) -> UnifiedIndicatorInterface:
        """Get or create indicator instance for specific timeframe."""
        if name not in self.indicators:
            raise ValueError(f"Indicator '{name}' not registered")
        
        instance_key = f"{name}_{timeframe.value}"
        
        # Check if instance already exists
        if instance_key in self.instances:
            return self.instances[instance_key]
        
        # Create new instance
        indicator_info = self.indicators[name]
        calculator_class = indicator_info['calculator_class']
        
        # Get timeframe-specific config
        timeframe_config = config_manager.get_config(name, timeframe)
        
        # Merge with base config
        full_config = {**indicator_info['config'].parameters, **timeframe_config}
        
        # Create instance
        instance = calculator_class(full_config)
        self.instances[instance_key] = instance
        
        return instance
    
    def get_calculation_order(self, indicator_names: List[str]) -> List[List[str]]:
        """Get calculation order based on dependencies (returns groups for parallel execution)."""
        # Build dependency subgraph
        subgraph = {}
        for name in indicator_names:
            if name in self.dependency_graph:
                subgraph[name] = [dep for dep in self.dependency_graph[name] if dep in indicator_names]
            else:
                subgraph[name] = []
        
        # Topological sort with grouping
        groups = []
        remaining = set(indicator_names)
        
        while remaining:
            # Find nodes with no dependencies in remaining set
            ready = []
            for name in remaining:
                if not any(dep in remaining for dep in subgraph[name]):
                    ready.append(name)
            
            if not ready:
                # Circular dependency - break it by taking highest priority
                priorities = {}
                for name in remaining:
                    if name in self.indicators:
                        priorities[name] = self.indicators[name]['config'].calculation_priority
                    else:
                        priorities[name] = 5  # Default priority
                
                ready = [min(remaining, key=lambda x: priorities[x])]
                logger.warning(f"Breaking circular dependency by prioritizing: {ready[0]}")
            
            groups.append(ready)
            remaining -= set(ready)
        
        return groups
    
    def _update_dependency_graph(self, config: IndicatorConfig) -> None:
        """Update dependency graph with new indicator."""
        self.dependency_graph[config.name] = config.dependencies
    
    def get_registered_indicators(self) -> List[str]:
        """Get list of all registered indicator names."""
        return list(self.indicators.keys())
    
    def get_indicator_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered indicator."""
        return self.indicators.get(name)


class MultiTimeframeCalculator:
    """Async calculation engine with optimization and parallel execution."""
    
    def __init__(self, registry: UnifiedIndicatorRegistry, config_manager: TimeframeConfigManager):
        self.registry = registry
        self.config_manager = config_manager
        self.cache = IndicatorCache()
        self.performance_metrics = defaultdict(list)
        self._calculation_semaphore = asyncio.Semaphore(10)  # Limit concurrent calculations
        
    async def calculate_indicators(
        self, 
        data: Dict[TimeframeType, pd.DataFrame],  
        indicator_requests: List[Tuple[str, TimeframeType]]
    ) -> Dict[TimeframeType, Dict[str, CalculationResult]]:
        """Calculate multiple indicators across timeframes with optimization."""
        
        # Group by timeframe
        timeframe_groups = defaultdict(list)
        for indicator_name, timeframe in indicator_requests:
            timeframe_groups[timeframe].append(indicator_name)
        
        results = {}
        
        # Process each timeframe
        for timeframe, indicator_names in timeframe_groups.items():
            if timeframe not in data:
                logger.warning(f"No data provided for timeframe: {timeframe}")
                continue
            
            timeframe_data = data[timeframe]
            timeframe_results = await self._calculate_timeframe_indicators(
                indicator_names, timeframe, timeframe_data
            )
            results[timeframe] = timeframe_results
        
        # Cleanup expired cache entries
        asyncio.create_task(self.cache.cleanup_expired())
        
        return results
    
    async def _calculate_timeframe_indicators(
        self, 
        indicator_names: List[str], 
        timeframe: TimeframeType, 
        data: pd.DataFrame
    ) -> Dict[str, CalculationResult]:
        """Calculate indicators for a specific timeframe with dependency management."""
        
        # Get calculation order (grouped by dependencies)
        calculation_groups = self.registry.get_calculation_order(indicator_names)
        
        results = {}
        
        # Execute groups sequentially, indicators within groups in parallel
        for group in calculation_groups:
            group_tasks = []
            
            for indicator_name in group:
                task = self._calculate_single_indicator(indicator_name, timeframe, data, results)
                group_tasks.append(task)
            
            # Wait for all indicators in this group
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # Process results
            for indicator_name, result in zip(group, group_results):
                if isinstance(result, Exception):
                    logger.error(f"Error calculating {indicator_name}: {result}")
                    # Create empty result for failed calculation
                    results[indicator_name] = CalculationResult(
                        data={},
                        calculation_time_ms=0,
                        cache_hit=False,
                        timeframe=timeframe
                    )
                else:
                    results[indicator_name] = result
        
        return results
    
    async def _calculate_single_indicator(
        self, 
        indicator_name: str, 
        timeframe: TimeframeType, 
        data: pd.DataFrame,
        dependency_results: Dict[str, CalculationResult]
    ) -> CalculationResult:
        """Calculate a single indicator with caching and performance tracking."""
        
        async with self._calculation_semaphore:
            start_time = time.perf_counter()
            
            # Generate cache key
            data_hash = self._get_data_hash(data)
            cache_key = f"{indicator_name}_{timeframe.value}_{data_hash}"
            
            # Check cache
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return CalculationResult(
                    data=cached_result,
                    calculation_time_ms=(time.perf_counter() - start_time) * 1000,
                    cache_hit=True,
                    timeframe=timeframe
                )
            
            try:
                # Get indicator instance
                indicator = self.registry.get_indicator(indicator_name, timeframe, self.config_manager)
                
                # Validate data
                if not indicator.validate_data(data):
                    raise ValueError(f"Invalid data for indicator {indicator_name}")
                
                # Calculate
                if hasattr(indicator, 'calculate_async'):
                    result_data = await indicator.calculate_async(data)
                else:
                    # Run in thread pool for CPU-intensive calculations
                    loop = asyncio.get_event_loop()
                    result_data = await loop.run_in_executor(None, indicator.calculate, data)
                
                # Add dependency results if needed
                if dependency_results:
                    result_data['dependencies'] = {
                        name: result.data for name, result in dependency_results.items()
                        if name in self.registry.dependency_graph.get(indicator_name, [])
                    }
                
                calculation_time = (time.perf_counter() - start_time) * 1000
                
                # Cache result
                cache_duration = self.config_manager.get_config(indicator_name, timeframe).get('cache_duration', 60)
                await self.cache.set(cache_key, result_data, cache_duration)
                
                # Track performance
                self.performance_metrics[indicator_name].append(calculation_time)
                if len(self.performance_metrics[indicator_name]) > 100:
                    self.performance_metrics[indicator_name].pop(0)  # Keep last 100 measurements
                
                return CalculationResult(
                    data=result_data,
                    calculation_time_ms=calculation_time,
                    cache_hit=False,
                    timeframe=timeframe
                )
                
            except Exception as e:
                logger.error(f"Error calculating {indicator_name} for {timeframe}: {e}")
                raise
    
    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for DataFrame to use as cache key."""
        # Use last few rows and basic stats for hash
        if len(data) > 10:
            sample_data = data.tail(10)
        else:
            sample_data = data
        
        # Create hash from basic data characteristics
        hash_input = f"{len(data)}_{sample_data.iloc[-1].to_dict()}_{data.columns.tolist()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all indicators."""
        stats = {}
        for indicator_name, times in self.performance_metrics.items():
            if times:
                stats[indicator_name] = {
                    'avg_time_ms': np.mean(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'std_time_ms': np.std(times),
                    'call_count': len(times)
                }
        return stats


class IncrementalIndicatorUpdater:
    """Manages incremental updates for real-time indicator calculations."""
    
    def __init__(self, registry: UnifiedIndicatorRegistry, config_manager: TimeframeConfigManager):
        self.registry = registry
        self.config_manager = config_manager
        self.indicator_states = {}
        self.update_queues = {}
        self.last_updates = {}
        
    async def setup_incremental_indicator(
        self, 
        indicator_name: str, 
        timeframe: TimeframeType,
        initial_data: pd.DataFrame
    ) -> bool:
        """Setup indicator for incremental updates."""
        key = f"{indicator_name}_{timeframe.value}"
        
        try:
            # Get indicator instance
            indicator = self.registry.get_indicator(indicator_name, timeframe, self.config_manager)
            
            if not indicator.supports_incremental:
                logger.info(f"Indicator {indicator_name} does not support incremental updates")
                return False
            
            # Initialize incremental state
            initial_state = indicator.setup_incremental(initial_data)
            if initial_state is not None:
                self.indicator_states[key] = initial_state
                self.update_queues[key] = asyncio.Queue(maxsize=1000)
                self.last_updates[key] = datetime.now()
                
                logger.info(f"Setup incremental updates for {indicator_name} on {timeframe.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to setup incremental updates for {indicator_name}: {e}")
        
        return False
    
    async def update_indicator_incremental(
        self,
        indicator_name: str,
        timeframe: TimeframeType,
        new_tick: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update indicator with single new tick/candle."""
        key = f"{indicator_name}_{timeframe.value}"
        
        if key not in self.indicator_states:
            logger.warning(f"Incremental state not found for {indicator_name}_{timeframe.value}")
            return None
        
        try:
            indicator = self.registry.get_indicator(indicator_name, timeframe, self.config_manager)
            current_state = self.indicator_states[key]
            
            # Update incrementally
            new_state, result = indicator.update_incremental(current_state, new_tick)
            
            # Update stored state
            self.indicator_states[key] = new_state
            self.last_updates[key] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in incremental update for {indicator_name}: {e}")
            return None
    
    def get_incremental_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all incremental indicators."""
        status = {}
        for key, last_update in self.last_updates.items():
            status[key] = {
                'last_update': last_update,
                'has_state': key in self.indicator_states,
                'queue_size': self.update_queues[key].qsize() if key in self.update_queues else 0
            }
        return status


# Indicator Adapters for Unified Framework

class VuManChuUnifiedAdapter(UnifiedIndicatorInterface):
    """Unified adapter for VuManChu Cipher indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Store config for lazy initialization
        self._vumanchu_config = {
            'wt_channel_length': config.get('wt_channel_length', 9),
            'wt_average_length': config.get('wt_average_length', 13),
            'wt_ma_length': config.get('wt_ma_length', 3),
        }
        
        self.overbought_level = config.get('overbought_level', 60.0)
        self.oversold_level = config.get('oversold_level', -60.0)
        self._vumanchu = None
    
    @property
    def vumanchu(self):
        """Lazy initialization of VuManChu indicators."""
        if self._vumanchu is None:
            try:
                from .vumanchu import VuManChuIndicators
                self._vumanchu = VuManChuIndicators(**self._vumanchu_config)
            except ImportError as e:
                logger.error(f"Failed to import VuManChu indicators: {e}")
                raise
        return self._vumanchu
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate VuManChu indicators with unified output format."""
        try:
            cipher_a_result = self.vumanchu.cipher_a.calculate(data)
            cipher_b_result = self.vumanchu.cipher_b.calculate(data)
            
            # Combine results
            combined_signals = self._combine_cipher_signals(cipher_a_result, cipher_b_result)
            
            return {
                'cipher_a': cipher_a_result,
                'cipher_b': cipher_b_result,
                'combined_signals': combined_signals,
                'overbought_level': self.overbought_level,
                'oversold_level': self.oversold_level,
                'timeframe': self.timeframe.value,
                'latest_values': {
                    'wt1': cipher_a_result.get('wt1', [0])[-1] if cipher_a_result.get('wt1') else 0,
                    'wt2': cipher_a_result.get('wt2', [0])[-1] if cipher_a_result.get('wt2') else 0,
                }
            }
        except Exception as e:
            logger.error(f"Error calculating VuManChu: {e}")
            return {
                'cipher_a': {},
                'cipher_b': {},
                'combined_signals': [],
                'error': str(e),
                'timeframe': self.timeframe.value
            }
    
    def _combine_cipher_signals(self, cipher_a: Dict[str, Any], cipher_b: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine signals from both cipher indicators."""
        signals = []
        
        # Extract signals from cipher A
        if 'signals' in cipher_a:
            for signal in cipher_a['signals']:
                signals.append({
                    'source': 'cipher_a',
                    'type': signal.get('type', 'unknown'),
                    'strength': signal.get('strength', 0.5),
                    'timestamp': signal.get('timestamp', datetime.now()),
                    'details': signal
                })
        
        # Extract signals from cipher B
        if 'signals' in cipher_b:
            for signal in cipher_b['signals']:
                signals.append({
                    'source': 'cipher_b',
                    'type': signal.get('type', 'unknown'),
                    'strength': signal.get('strength', 0.5),
                    'timestamp': signal.get('timestamp', datetime.now()),
                    'details': signal
                })
        
        # Sort by timestamp
        signals.sort(key=lambda x: x['timestamp'])
        
        return signals
    
    def get_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trading signals from calculation result."""
        return result.get('combined_signals', [])


class FastEMAUnifiedAdapter(UnifiedIndicatorInterface):
    """Unified adapter for Fast EMA indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Timeframe-specific periods
        self.periods = config.get('periods', [3, 5, 8] if self.timeframe == TimeframeType.SCALPING else [12, 26, 50])
        self.supports_incremental = config.get('supports_incremental', True)
        
        self._fast_ema = None
        self._ema_signals = None
    
    @property
    def fast_ema(self):
        """Lazy initialization of FastEMA."""
        if self._fast_ema is None:
            try:
                from .fast_ema import FastEMA
                self._fast_ema = FastEMA(self.periods)
            except ImportError as e:
                logger.error(f"Failed to import FastEMA: {e}")
                raise
        return self._fast_ema
    
    @property
    def ema_signals(self):
        """Lazy initialization of ScalpingEMASignals."""
        if self._ema_signals is None:
            try:
                from .fast_ema import ScalpingEMASignals
                self._ema_signals = ScalpingEMASignals()
            except ImportError as e:
                logger.error(f"Failed to import ScalpingEMASignals: {e}")
                raise
        return self._ema_signals
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Fast EMA with unified output format."""
        try:
            ema_values = self.fast_ema.calculate(data)
            signals = self.ema_signals.analyze_signals(ema_values, data)
            
            trend_strength = self._calculate_trend_strength(ema_values)
            
            return {
                'ema_values': ema_values,
                'signals': signals,
                'trend_strength': trend_strength,
                'periods': self.periods,
                'timeframe': self.timeframe.value,
                'latest_values': {
                    f'ema_{period}': values[-1] if len(values) > 0 else 0
                    for period, values in ema_values.items()
                }
            }
        except Exception as e:
            logger.error(f"Error calculating Fast EMA: {e}")
            return {
                'ema_values': {},
                'signals': [],
                'trend_strength': 0,
                'error': str(e),
                'timeframe': self.timeframe.value
            }
    
    def _calculate_trend_strength(self, ema_values: Dict[int, List[float]]) -> float:
        """Calculate trend strength based on EMA alignment."""
        if not ema_values or len(self.periods) < 2:
            return 0.0
        
        latest_values = []
        for period in sorted(self.periods):
            if period in ema_values and len(ema_values[period]) > 0:
                latest_values.append(ema_values[period][-1])
        
        if len(latest_values) < 2:
            return 0.0
        
        # Check alignment
        bullish_alignment = all(latest_values[i] > latest_values[i+1] for i in range(len(latest_values)-1))
        bearish_alignment = all(latest_values[i] < latest_values[i+1] for i in range(len(latest_values)-1))
        
        if bullish_alignment:
            return 1.0
        elif bearish_alignment:
            return -1.0
        else:
            # Partial alignment
            aligned_pairs = sum(1 for i in range(len(latest_values)-1) 
                              if latest_values[i] > latest_values[i+1])
            return (aligned_pairs / (len(latest_values) - 1)) * 2 - 1  # Scale to [-1, 1]
    
    def setup_incremental(self, initial_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Setup incremental state for Fast EMA."""
        if hasattr(self.fast_ema, 'setup_incremental'):
            return self.fast_ema.setup_incremental(initial_data)
        return None
    
    def update_incremental(self, state: Dict[str, Any], new_tick: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update Fast EMA incrementally."""
        if hasattr(self.fast_ema, 'update_incremental'):
            return self.fast_ema.update_incremental(state, new_tick)
        return state, {}
    
    def get_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trading signals from calculation result."""
        return result.get('signals', [])


class ScalpingMomentumUnifiedAdapter(UnifiedIndicatorInterface):
    """Unified adapter for scalping momentum indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Store config for lazy initialization
        if self.timeframe == TimeframeType.SCALPING:
            self._rsi_config = {'period': config.get('rsi_period', 7)}
            self._macd_config = {
                'fast': config.get('macd_fast', 5),
                'slow': config.get('macd_slow', 10),
                'signal': config.get('macd_signal', 3)
            }
            self._williams_config = {'period': config.get('williams_period', 7)}
        else:
            self._rsi_config = {'period': config.get('rsi_period', 14)}
            self._macd_config = {
                'fast': config.get('macd_fast', 12),
                'slow': config.get('macd_slow', 26),
                'signal': config.get('macd_signal', 9)
            }
            self._williams_config = {'period': config.get('williams_period', 14)}
        
        self._fast_rsi = None
        self._fast_macd = None
        self._williams_r = None
        self._momentum_signals = None
        
        # Thresholds
        self.rsi_overbought = config.get('rsi_overbought', 75 if self.timeframe == TimeframeType.SCALPING else 70)
        self.rsi_oversold = config.get('rsi_oversold', 25 if self.timeframe == TimeframeType.SCALPING else 30)
        self.williams_overbought = config.get('williams_overbought', -20)
        self.williams_oversold = config.get('williams_oversold', -80)
    
    @property
    def fast_rsi(self):
        """Lazy initialization of FastRSI."""
        if self._fast_rsi is None:
            try:
                from .scalping_momentum import FastRSI
                self._fast_rsi = FastRSI(**self._rsi_config)
            except ImportError as e:
                logger.error(f"Failed to import FastRSI: {e}")
                raise
        return self._fast_rsi
    
    @property
    def fast_macd(self):
        """Lazy initialization of FastMACD."""
        if self._fast_macd is None:
            try:
                from .scalping_momentum import FastMACD
                self._fast_macd = FastMACD(**self._macd_config)
            except ImportError as e:
                logger.error(f"Failed to import FastMACD: {e}")
                raise
        return self._fast_macd
    
    @property
    def williams_r(self):
        """Lazy initialization of WilliamsPercentR."""
        if self._williams_r is None:
            try:
                from .scalping_momentum import WilliamsPercentR
                self._williams_r = WilliamsPercentR(**self._williams_config)
            except ImportError as e:
                logger.error(f"Failed to import WilliamsPercentR: {e}")
                raise
        return self._williams_r
    
    @property
    def momentum_signals(self):
        """Lazy initialization of ScalpingMomentumSignals."""
        if self._momentum_signals is None:
            try:
                from .scalping_momentum import ScalpingMomentumSignals
                self._momentum_signals = ScalpingMomentumSignals()
            except ImportError as e:
                logger.error(f"Failed to import ScalpingMomentumSignals: {e}")
                raise
        return self._momentum_signals
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators with unified output format."""
        try:
            # Calculate individual indicators
            rsi_result = self.fast_rsi.calculate(data)
            macd_result = self.fast_macd.calculate(data)
            williams_result = self.williams_r.calculate(data)
            
            # Generate combined signals
            combined_signals = self.momentum_signals.analyze_momentum_convergence(
                rsi_result, macd_result, williams_result, data
            )
            
            return {
                'rsi': rsi_result,
                'macd': macd_result,
                'williams_r': williams_result,
                'combined_signals': combined_signals,
                'thresholds': {
                    'rsi_overbought': self.rsi_overbought,
                    'rsi_oversold': self.rsi_oversold,
                    'williams_overbought': self.williams_overbought,
                    'williams_oversold': self.williams_oversold
                },
                'timeframe': self.timeframe.value,
                'latest_values': {
                    'rsi': rsi_result.get('rsi', [0])[-1] if rsi_result.get('rsi') else 0,
                    'macd': macd_result.get('macd', [0])[-1] if macd_result.get('macd') else 0,
                    'macd_signal': macd_result.get('signal', [0])[-1] if macd_result.get('signal') else 0,
                    'williams_r': williams_result.get('williams_r', [0])[-1] if williams_result.get('williams_r') else 0
                }
            }
        except Exception as e:
            logger.error(f"Error calculating scalping momentum: {e}")
            return {
                'rsi': {},
                'macd': {},
                'williams_r': {},
                'combined_signals': [],
                'error': str(e),
                'timeframe': self.timeframe.value
            }
    
    def get_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trading signals from calculation result."""
        return result.get('combined_signals', [])


class ScalpingVolumeUnifiedAdapter(UnifiedIndicatorInterface):
    """Unified adapter for scalping volume indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Store config for lazy initialization
        vwap_period = config.get('vwap_period', 20 if self.timeframe == TimeframeType.SCALPING else 50)
        self._vwap_config = {'period': vwap_period}
        self._volume_profile_config = {
            'bins': config.get('volume_bins', 50),
            'period': config.get('volume_period', 100)
        }
        
        self._scalping_vwap = None
        self._volume_profile = None
        self._volume_signals = None
        
        # Volume analysis parameters
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.vwap_bands = config.get('vwap_bands', [1.0, 2.0])
    
    @property
    def scalping_vwap(self):
        """Lazy initialization of ScalpingVWAP."""
        if self._scalping_vwap is None:
            try:
                from .scalping_volume import ScalpingVWAP
                self._scalping_vwap = ScalpingVWAP(**self._vwap_config)
            except ImportError as e:
                logger.error(f"Failed to import ScalpingVWAP: {e}")
                raise
        return self._scalping_vwap
    
    @property
    def volume_profile(self):
        """Lazy initialization of VolumeProfile."""
        if self._volume_profile is None:
            try:
                from .scalping_volume import VolumeProfile
                self._volume_profile = VolumeProfile(**self._volume_profile_config)
            except ImportError as e:
                logger.error(f"Failed to import VolumeProfile: {e}")
                raise
        return self._volume_profile
    
    @property
    def volume_signals(self):
        """Lazy initialization of ScalpingVolumeSignals."""
        if self._volume_signals is None:
            try:
                from .scalping_volume import ScalpingVolumeSignals
                self._volume_signals = ScalpingVolumeSignals()
            except ImportError as e:
                logger.error(f"Failed to import ScalpingVolumeSignals: {e}")
                raise
        return self._volume_signals
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume indicators with unified output format."""
        try:
            # Calculate individual indicators
            vwap_result = self.scalping_vwap.calculate(data)
            volume_profile_result = self.volume_profile.calculate(data)
            
            # Generate volume signals
            volume_signals = self.volume_signals.analyze_volume_breakout(
                vwap_result, volume_profile_result, data
            )
            
            # Calculate volume analysis
            volume_analysis = self._analyze_volume_patterns(data)
            
            return {
                'vwap': vwap_result,
                'volume_profile': volume_profile_result,
                'volume_analysis': volume_analysis,
                'signals': volume_signals,
                'volume_threshold': self.volume_threshold,
                'vwap_bands': self.vwap_bands,
                'timeframe': self.timeframe.value,
                'latest_values': {
                    'vwap': vwap_result.get('vwap', [0])[-1] if vwap_result.get('vwap') else 0,
                    'volume_ratio': volume_analysis.get('volume_ratio', 0),
                    'high_volume_poc': volume_profile_result.get('poc', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating scalping volume: {e}")
            return {
                'vwap': {},
                'volume_profile': {},
                'volume_analysis': {},
                'signals': [],
                'error': str(e),
                'timeframe': self.timeframe.value
            }
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns and characteristics."""
        if len(data) < 20:
            return {'volume_ratio': 0, 'volume_trend': 'neutral'}
        
        recent_volume = data['volume'].tail(10).mean()
        historical_volume = data['volume'].tail(50).mean()
        
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        
        # Determine volume trend
        if volume_ratio > self.volume_threshold:
            volume_trend = 'increasing'
        elif volume_ratio < (1 / self.volume_threshold):
            volume_trend = 'decreasing'
        else:
            volume_trend = 'neutral'
        
        return {
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'recent_volume': recent_volume,
            'historical_volume': historical_volume,
            'high_volume': volume_ratio > self.volume_threshold
        }
    
    def get_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trading signals from calculation result."""
        return result.get('signals', [])


class PerformanceOptimizer:
    """Analyzes and optimizes indicator calculation performance."""
    
    def __init__(self):
        self.performance_history = defaultdict(deque)
        self.optimization_suggestions = []
        
    def analyze_performance(self, calculator: MultiTimeframeCalculator) -> Dict[str, Any]:
        """Analyze current performance and suggest optimizations."""
        stats = calculator.get_performance_stats()
        
        analysis = {
            'summary': self._generate_performance_summary(stats),
            'slow_indicators': self._identify_slow_indicators(stats),
            'optimization_suggestions': self._generate_optimization_suggestions(stats),
            'cache_effectiveness': self._analyze_cache_effectiveness(calculator.cache)
        }
        
        return analysis
    
    def _generate_performance_summary(self, stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        if not stats:
            return {'total_indicators': 0, 'avg_time_ms': 0}
        
        all_times = []
        for indicator_stats in stats.values():
            all_times.extend([indicator_stats['avg_time_ms']])
        
        return {
            'total_indicators': len(stats),
            'avg_time_ms': np.mean(all_times) if all_times else 0,
            'max_time_ms': np.max(all_times) if all_times else 0,
            'total_calls': sum(s['call_count'] for s in stats.values())
        }
    
    def _identify_slow_indicators(self, stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify indicators that are performing slowly."""
        slow_threshold = 50.0  # 50ms
        
        slow_indicators = []
        for indicator_name, indicator_stats in stats.items():
            if indicator_stats['avg_time_ms'] > slow_threshold:
                slow_indicators.append({
                    'name': indicator_name,
                    'avg_time_ms': indicator_stats['avg_time_ms'],
                    'max_time_ms': indicator_stats['max_time_ms'],
                    'call_count': indicator_stats['call_count'],
                    'severity': 'high' if indicator_stats['avg_time_ms'] > 100 else 'medium'
                })
        
        return sorted(slow_indicators, key=lambda x: x['avg_time_ms'], reverse=True)
    
    def _generate_optimization_suggestions(self, stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate specific optimization suggestions."""
        suggestions = []
        
        for indicator_name, indicator_stats in stats.items():
            avg_time = indicator_stats['avg_time_ms']
            
            if avg_time > 100:
                suggestions.append({
                    'indicator': indicator_name,
                    'issue': 'very_slow_calculation',
                    'suggestion': 'Consider implementing incremental updates or reducing calculation frequency',
                    'priority': 'high',
                    'estimated_improvement': '60-80%'
                })
            elif avg_time > 50:
                suggestions.append({
                    'indicator': indicator_name,
                    'issue': 'slow_calculation',
                    'suggestion': 'Optimize calculation algorithm or increase cache duration',
                    'priority': 'medium',
                    'estimated_improvement': '30-50%'
                })
            
            if indicator_stats['std_time_ms'] > avg_time * 0.5:
                suggestions.append({
                    'indicator': indicator_name,
                    'issue': 'inconsistent_performance',
                    'suggestion': 'Investigate performance variability and optimize data processing',
                    'priority': 'medium',
                    'estimated_improvement': '20-40%'
                })
        
        return suggestions
    
    def _analyze_cache_effectiveness(self, cache: IndicatorCache) -> Dict[str, Any]:
        """Analyze cache hit rates and effectiveness."""
        stats = cache.get_stats()
        
        return {
            'cache_size': stats['size'],
            'cache_keys': len(stats['keys']),
            'suggestion': 'Monitor cache hit rates and adjust TTL based on indicator update frequency'
        }


class UnifiedIndicatorFramework:
    """
    Main orchestrator for the unified indicator framework.
    
    Provides a single entry point for all indicator calculations across timeframes
    with optimization, caching, and performance monitoring.
    """
    
    def __init__(self):
        self.registry = UnifiedIndicatorRegistry()
        self.config_manager = TimeframeConfigManager()
        self.calculator = MultiTimeframeCalculator(self.registry, self.config_manager)
        self.incremental_updater = IncrementalIndicatorUpdater(self.registry, self.config_manager)
        self.performance_optimizer = PerformanceOptimizer()
        
        # Register all available indicators
        self._register_all_indicators()
        
        logger.info("Unified Indicator Framework initialized with all indicators registered")
    
    def _register_all_indicators(self) -> None:
        """Register all available indicators with their configurations."""
        
        # VuManChu Cipher A
        self.registry.register_indicator(
            IndicatorConfig(
                name='vumanchu_cipher_a',
                type=IndicatorType.CUSTOM,
                timeframes=[TimeframeType.SCALPING, TimeframeType.MOMENTUM, TimeframeType.SWING],
                parameters={},
                calculation_priority=2,
                cache_duration=30,
                dependencies=[],
                supports_incremental=False,
                memory_efficient=True
            ),
            VuManChuUnifiedAdapter
        )
        
        # VuManChu Cipher B  
        self.registry.register_indicator(
            IndicatorConfig(
                name='vumanchu_cipher_b',
                type=IndicatorType.CUSTOM,
                timeframes=[TimeframeType.SCALPING, TimeframeType.MOMENTUM, TimeframeType.SWING],
                parameters={},
                calculation_priority=2,
                cache_duration=30,
                dependencies=[],
                supports_incremental=False,
                memory_efficient=True
            ),
            VuManChuUnifiedAdapter  # Same adapter handles both cipher A and B
        )
        
        # Fast EMA
        self.registry.register_indicator(
            IndicatorConfig(
                name='fast_ema',
                type=IndicatorType.TREND,
                timeframes=[TimeframeType.SCALPING, TimeframeType.MOMENTUM],
                parameters={},
                calculation_priority=1,
                cache_duration=15,
                dependencies=[],
                supports_incremental=True,
                memory_efficient=True
            ),
            FastEMAUnifiedAdapter
        )
        
        # Scalping Momentum (RSI, MACD, Williams %R)
        self.registry.register_indicator(
            IndicatorConfig(
                name='scalping_momentum',
                type=IndicatorType.MOMENTUM,
                timeframes=[TimeframeType.SCALPING, TimeframeType.MOMENTUM],
                parameters={},
                calculation_priority=3,
                cache_duration=20,
                dependencies=[],
                supports_incremental=False,
                memory_efficient=True
            ),
            ScalpingMomentumUnifiedAdapter
        )
        
        # Scalping Volume (VWAP, Volume Profile)
        self.registry.register_indicator(
            IndicatorConfig(
                name='scalping_volume',
                type=IndicatorType.VOLUME,
                timeframes=[TimeframeType.SCALPING, TimeframeType.MOMENTUM],
                parameters={},
                calculation_priority=4,
                cache_duration=25,
                dependencies=[],
                supports_incremental=False,
                memory_efficient=True
            ),
            ScalpingVolumeUnifiedAdapter
        )
        
        logger.info(f"Registered {len(self.registry.get_registered_indicators())} indicators")
    
    async def calculate_for_strategy(
        self, 
        strategy_type: str,  
        market_data: Dict[str, pd.DataFrame],
        custom_indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all indicators needed for a specific strategy.
        
        Args:
            strategy_type: 'momentum', 'scalping', 'swing', or 'position'
            market_data: Dictionary with timeframe data {timeframe_str: DataFrame}
            custom_indicators: Optional list of specific indicators to calculate
            
        Returns:
            Dictionary with calculated indicators and metadata
        """
        
        # Map strategy type to timeframe
        strategy_timeframe_map = {
            'scalping': TimeframeType.SCALPING,
            'momentum': TimeframeType.MOMENTUM,
            'swing': TimeframeType.SWING,
            'position': TimeframeType.POSITION
        }
        
        if strategy_type not in strategy_timeframe_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        timeframe = strategy_timeframe_map[strategy_type]
        
        # Determine required indicators
        if custom_indicators:
            required_indicators = custom_indicators
        else:
            required_indicators = self._get_strategy_indicators(strategy_type)
        
        # Convert market data keys to TimeframeType
        converted_market_data = {}
        for tf_str, data in market_data.items():
            if tf_str == strategy_type or tf_str == timeframe.value:
                converted_market_data[timeframe] = data
                break
        
        if not converted_market_data:
            # Try to find compatible data
            for tf_str, data in market_data.items():
                converted_market_data[timeframe] = data
                break
        
        if not converted_market_data:
            raise ValueError(f"No market data provided for {strategy_type} strategy")
        
        # Calculate indicators
        start_time = time.perf_counter()
        
        indicator_requests = [(name, timeframe) for name in required_indicators]
        results = await self.calculator.calculate_indicators(
            converted_market_data, 
            indicator_requests
        )
        
        calculation_time = (time.perf_counter() - start_time) * 1000
        
        # Extract results for the timeframe
        timeframe_results = results.get(timeframe, {})
        
        # Combine all signals
        all_signals = []
        for indicator_name, calc_result in timeframe_results.items():
            indicator_instance = self.registry.get_indicator(indicator_name, timeframe, self.config_manager)
            signals = indicator_instance.get_signals(calc_result.data)
            for signal in signals:
                signal['indicator'] = indicator_name
                all_signals.append(signal)
        
        # Sort signals by timestamp/strength
        all_signals.sort(key=lambda x: (x.get('timestamp', datetime.now()), -x.get('strength', 0)))
        
        return {
            'strategy_type': strategy_type,
            'timeframe': timeframe.value,
            'indicators': {name: result.data for name, result in timeframe_results.items()},
            'combined_signals': all_signals[:10],  # Top 10 signals
            'performance_metrics': {
                'total_calculation_time_ms': calculation_time,
                'indicator_count': len(timeframe_results),
                'cache_hits': sum(1 for result in timeframe_results.values() if result.cache_hit),
                'cache_misses': sum(1 for result in timeframe_results.values() if not result.cache_hit)
            },
            'calculation_details': {
                name: {
                    'calculation_time_ms': result.calculation_time_ms,
                    'cache_hit': result.cache_hit,
                    'data_size': len(result.data) if isinstance(result.data, dict) else 0
                }
                for name, result in timeframe_results.items()
            }
        }
    
    def _get_strategy_indicators(self, strategy_type: str) -> List[str]:
        """Get list of indicators required for a specific strategy."""
        
        strategy_indicators = {
            'scalping': [
                'vumanchu_cipher_a',
                'fast_ema',
                'scalping_momentum',
                'scalping_volume'
            ],
            'momentum': [
                'vumanchu_cipher_a', 
                'vumanchu_cipher_b',
                'fast_ema',
                'scalping_momentum',
                'scalping_volume'
            ],
            'swing': [
                'vumanchu_cipher_a',
                'vumanchu_cipher_b',
                'scalping_momentum',
                'scalping_volume'
            ],
            'position': [
                'vumanchu_cipher_a',
                'vumanchu_cipher_b'
            ]
        }
        
        return strategy_indicators.get(strategy_type, [])
    
    async def setup_incremental_mode(
        self, 
        strategy_type: str, 
        initial_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, bool]:
        """Setup incremental mode for real-time updates."""
        
        timeframe = {
            'scalping': TimeframeType.SCALPING,
            'momentum': TimeframeType.MOMENTUM,
            'swing': TimeframeType.SWING,
            'position': TimeframeType.POSITION
        }.get(strategy_type, TimeframeType.MOMENTUM)
        
        indicators = self._get_strategy_indicators(strategy_type)
        
        # Find compatible data
        data_df = None
        for tf_str, data in initial_data.items():
            if tf_str == strategy_type or tf_str == timeframe.value:
                data_df = data
                break
        
        if data_df is None and initial_data:
            data_df = list(initial_data.values())[0]  # Use first available data
        
        if data_df is None:
            return {}
        
        # Setup incremental indicators
        setup_results = {}
        for indicator_name in indicators:
            success = await self.incremental_updater.setup_incremental_indicator(
                indicator_name, timeframe, data_df
            )
            setup_results[indicator_name] = success
        
        logger.info(f"Setup incremental mode for {strategy_type}: {setup_results}")
        return setup_results
    
    async def update_incremental(
        self, 
        strategy_type: str, 
        new_tick: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update indicators incrementally with new tick data."""
        
        timeframe = {
            'scalping': TimeframeType.SCALPING,
            'momentum': TimeframeType.MOMENTUM,
            'swing': TimeframeType.SWING,
            'position': TimeframeType.POSITION
        }.get(strategy_type, TimeframeType.MOMENTUM)
        
        indicators = self._get_strategy_indicators(strategy_type)
        
        results = {}
        for indicator_name in indicators:
            result = await self.incremental_updater.update_indicator_incremental(
                indicator_name, timeframe, new_tick
            )
            if result is not None:
                results[indicator_name] = result
        
        return results
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis and optimization suggestions."""
        return self.performance_optimizer.analyze_performance(self.calculator)
    
    def get_available_indicators(self, timeframe: Optional[TimeframeType] = None) -> List[Dict[str, Any]]:
        """Get list of available indicators, optionally filtered by timeframe."""
        indicators = []
        
        for name in self.registry.get_registered_indicators():
            info = self.registry.get_indicator_info(name)
            if info:
                config = info['config']
                if timeframe is None or timeframe in config.timeframes:
                    indicators.append({
                        'name': name,
                        'type': config.type.value,
                        'timeframes': [tf.value for tf in config.timeframes],
                        'supports_incremental': config.supports_incremental,
                        'calculation_priority': config.calculation_priority
                    })
        
        return sorted(indicators, key=lambda x: x['calculation_priority'])
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get overall framework status and health metrics."""
        return {
            'registered_indicators': len(self.registry.get_registered_indicators()),
            'cache_stats': self.calculator.cache.get_stats(),
            'performance_stats': self.calculator.get_performance_stats(),
            'incremental_status': self.incremental_updater.get_incremental_status(),
            'available_timeframes': [tf.value for tf in TimeframeType],
            'supported_strategies': ['scalping', 'momentum', 'swing', 'position']
        }


# Global framework instance
unified_framework = UnifiedIndicatorFramework()

# Convenience functions for easy integration
async def calculate_indicators_for_strategy(
    strategy_type: str, 
    market_data: Dict[str, pd.DataFrame],
    custom_indicators: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience function to calculate indicators for a strategy."""
    return await unified_framework.calculate_for_strategy(
        strategy_type, market_data, custom_indicators
    )

def get_available_indicators_for_timeframe(timeframe_str: str) -> List[Dict[str, Any]]:
    """Get available indicators for a specific timeframe."""
    timeframe_map = {
        'scalping': TimeframeType.SCALPING,
        'momentum': TimeframeType.MOMENTUM,
        'swing': TimeframeType.SWING,
        'position': TimeframeType.POSITION
    }
    
    timeframe = timeframe_map.get(timeframe_str)
    return unified_framework.get_available_indicators(timeframe)

def get_framework_performance() -> Dict[str, Any]:
    """Get framework performance analysis."""
    return unified_framework.get_performance_analysis()