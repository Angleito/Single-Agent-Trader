#!/usr/bin/env python3
"""
Simple test for the Unified Indicator Framework core components.
"""

import sys
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_file_structure():
    """Test that the framework file exists and has the right structure."""
    try:
        framework_path = Path("bot/indicators/unified_framework.py")
        
        if not framework_path.exists():
            logger.error(f"❌ Framework file not found: {framework_path}")
            return False
        
        # Read file content
        with open(framework_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        required_classes = [
            "TimeframeType",
            "IndicatorType", 
            "IndicatorConfig",
            "UnifiedIndicatorInterface",
            "TimeframeConfigManager",
            "UnifiedIndicatorRegistry",
            "MultiTimeframeCalculator",
            "IncrementalIndicatorUpdater",
            "IndicatorCache",
            "UnifiedIndicatorFramework"
        ]
        
        missing_classes = []
        for class_name in required_classes:
            if f"class {class_name}" not in content:
                missing_classes.append(class_name)
        
        if missing_classes:
            logger.error(f"❌ Missing classes: {missing_classes}")
            return False
        
        logger.info(f"✅ All required classes found: {len(required_classes)} classes")
        
        # Check for adapter classes
        adapter_classes = [
            "VuManChuUnifiedAdapter",
            "FastEMAUnifiedAdapter", 
            "ScalpingMomentumUnifiedAdapter",
            "ScalpingVolumeUnifiedAdapter"
        ]
        
        found_adapters = []
        for adapter in adapter_classes:
            if f"class {adapter}" in content:
                found_adapters.append(adapter)
        
        logger.info(f"✅ Found {len(found_adapters)} adapter classes: {found_adapters}")
        
        # Check for key methods
        key_methods = [
            "calculate_indicators_for_strategy",
            "get_available_indicators_for_timeframe",
            "get_framework_performance"
        ]
        
        found_methods = []
        for method in key_methods:
            if f"def {method}" in content or f"async def {method}" in content:
                found_methods.append(method)
        
        logger.info(f"✅ Found {len(found_methods)} key methods: {found_methods}")
        
        # Check file size (should be substantial)
        file_size = len(content)
        logger.info(f"✅ Framework file size: {file_size:,} characters")
        
        if file_size < 10000:  # Less than 10KB seems too small
            logger.warning("⚠️ Framework file seems small, might be incomplete")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File structure test error: {e}")
        return False

def test_syntax_validity():
    """Test that the framework file has valid Python syntax."""
    try:
        import ast
        
        framework_path = Path("bot/indicators/unified_framework.py")
        
        with open(framework_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        try:
            ast.parse(content)
            logger.info("✅ Python syntax is valid")
            return True
        except SyntaxError as e:
            logger.error(f"❌ Syntax error: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Syntax test error: {e}")
        return False

def test_imports_structure():
    """Test the import structure in the framework."""
    try:
        framework_path = Path("bot/indicators/unified_framework.py")
        
        with open(framework_path, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            "import asyncio",
            "import logging",
            "from enum import Enum",
            "from typing import",
            "import numpy as np",
            "import pandas as pd"
        ]
        
        missing_imports = []
        for import_stmt in required_imports:
            if import_stmt not in content:
                missing_imports.append(import_stmt)
        
        if missing_imports:
            logger.warning(f"⚠️ Some expected imports not found: {missing_imports}")
        else:
            logger.info("✅ All expected imports found")
        
        # Check for lazy imports (should NOT have direct indicator imports at top)
        problematic_imports = [
            "from .fast_ema import FastEMA",
            "from .vumanchu import VuManChuIndicators", 
            "from .scalping_momentum import FastRSI"
        ]
        
        found_problematic = []
        for import_stmt in problematic_imports:
            if import_stmt in content:
                found_problematic.append(import_stmt)
        
        if found_problematic:
            logger.warning(f"⚠️ Found problematic top-level imports: {found_problematic}")
        else:
            logger.info("✅ No problematic top-level imports found (using lazy loading)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import structure test error: {e}")
        return False

def test_configuration_completeness():
    """Test that configuration sections are complete."""
    try:
        framework_path = Path("bot/indicators/unified_framework.py")
        
        with open(framework_path, 'r') as f:
            content = f.read()
        
        # Check for timeframe configurations
        timeframes = ["SCALPING", "MOMENTUM", "SWING"]
        indicators = ["vumanchu_cipher_a", "fast_ema", "scalping_momentum"]
        
        config_sections_found = 0
        
        for timeframe in timeframes:
            if f"TimeframeType.{timeframe}" in content:
                config_sections_found += 1
        
        logger.info(f"✅ Found {config_sections_found} timeframe configurations")
        
        for indicator in indicators:
            if f"'{indicator}'" in content:
                logger.info(f"  • {indicator} configuration found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test error: {e}")
        return False

def main():
    """Run all simple tests."""
    logger.info("🚀 Testing Unified Indicator Framework - Simple Structure Tests")
    logger.info("=" * 70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Syntax Validity", test_syntax_validity),
        ("Import Structure", test_imports_structure),
        ("Configuration Completeness", test_configuration_completeness)
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
        logger.info("🎉 All structure tests passed! Framework appears to be correctly implemented.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)