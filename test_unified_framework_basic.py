#!/usr/bin/env python3
"""
Basic test for the Unified Indicator Framework without dependencies.

This script tests the framework independently to ensure it imports and works correctly.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path without importing the full bot module
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that the framework imports correctly."""
    try:
        # Direct import to avoid bot module dependencies
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "unified_framework", 
            "bot/indicators/unified_framework.py"
        )
        unified_framework_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unified_framework_module)
        
        # Test enum imports
        TimeframeType = unified_framework_module.TimeframeType
        IndicatorType = unified_framework_module.IndicatorType
        
        logger.info("✅ Enum imports successful")
        
        # Test timeframes
        timeframes = [tf.value for tf in TimeframeType]
        logger.info(f"📊 Available timeframes: {timeframes}")
        
        # Test indicator types
        indicator_types = [it.value for it in IndicatorType]
        logger.info(f"🔧 Available indicator types: {indicator_types}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_config_classes():
    """Test configuration classes."""
    try:
        # Direct import to avoid bot module dependencies
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "unified_framework", 
            "bot/indicators/unified_framework.py"
        )
        unified_framework_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unified_framework_module)
        
        IndicatorConfig = unified_framework_module.IndicatorConfig
        TimeframeConfigManager = unified_framework_module.TimeframeConfigManager
        IndicatorType = unified_framework_module.IndicatorType
        TimeframeType = unified_framework_module.TimeframeType
        
        # Test IndicatorConfig
        config = IndicatorConfig(
            name="test_indicator",
            type=IndicatorType.TREND,
            timeframes=[TimeframeType.SCALPING],
            parameters={"period": 10},
            calculation_priority=1,
            cache_duration=30,
            dependencies=[],
            supports_incremental=True
        )
        logger.info(f"✅ IndicatorConfig created: {config.name}")
        
        # Test TimeframeConfigManager
        config_manager = TimeframeConfigManager()
        scalping_config = config_manager.get_config("vumanchu_cipher_a", TimeframeType.SCALPING)
        logger.info(f"✅ TimeframeConfigManager working, scalping config keys: {list(scalping_config.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config class error: {e}")
        return False

def test_interface_class():
    """Test the unified indicator interface."""
    try:
        from bot.indicators.unified_framework import UnifiedIndicatorInterface
        import pandas as pd
        import numpy as np
        
        # Create a test implementation
        class TestIndicator(UnifiedIndicatorInterface):
            def calculate(self, data):
                return {"test_value": 42, "length": len(data)}
        
        # Test the implementation
        test_config = {"timeframe": TimeframeType.SCALPING, "cache_duration": 30}
        indicator = TestIndicator(test_config)
        
        # Create test data
        test_data = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [101, 102, 101],
            "volume": [1000, 1100, 1200]
        })
        
        # Test validation
        is_valid = indicator.validate_data(test_data)
        logger.info(f"✅ Data validation: {is_valid}")
        
        # Test calculation
        result = indicator.calculate(test_data)
        logger.info(f"✅ Test calculation result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Interface class error: {e}")
        return False

def test_registry():
    """Test the indicator registry."""
    try:
        from bot.indicators.unified_framework import UnifiedIndicatorRegistry, IndicatorConfig
        
        registry = UnifiedIndicatorRegistry()
        
        # Test basic registry functionality
        registered_indicators = registry.get_registered_indicators()
        logger.info(f"✅ Registry created, indicators: {len(registered_indicators)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Registry error: {e}")
        return False

def test_cache():
    """Test the indicator cache."""
    try:
        from bot.indicators.unified_framework import IndicatorCache
        import asyncio
        
        async def test_cache_async():
            cache = IndicatorCache(default_ttl=30)
            
            # Test cache operations
            await cache.set("test_key", {"value": 123}, ttl=60)
            
            result = await cache.get("test_key")
            logger.info(f"✅ Cache test result: {result}")
            
            stats = cache.get_stats()
            logger.info(f"✅ Cache stats: {stats}")
            
            return result is not None
        
        return asyncio.run(test_cache_async())
        
    except Exception as e:
        logger.error(f"❌ Cache error: {e}")
        return False

def main():
    """Run all basic tests."""
    logger.info("🚀 Testing Unified Indicator Framework - Basic Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config Classes", test_config_classes), 
        ("Interface Class", test_interface_class),
        ("Registry", test_registry),
        ("Cache", test_cache)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {status}")
        except Exception as e:
            logger.error(f"   ❌ FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Summary:")
    logger.info("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic tests passed! Framework is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)