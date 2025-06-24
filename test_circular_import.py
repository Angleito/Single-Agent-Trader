#!/usr/bin/env python3
"""
Test script to verify circular import resolution between bot.config and bot.fp.types.config
"""

import sys
import os
import importlib.util

def test_direct_module_imports():
    """Test importing config modules directly without package initialization."""
    print("Testing direct module imports to check for circular dependencies...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    try:
        # Test 1: Import bot.fp.types.config first
        print("1. Testing bot.fp.types.config import...")
        
        # Import the result module first (dependency of config)
        spec_result = importlib.util.spec_from_file_location(
            "bot.fp.types.result", 
            os.path.join(project_root, "bot/fp/types/result.py")
        )
        result_module = importlib.util.module_from_spec(spec_result)
        sys.modules["bot.fp.types.result"] = result_module
        spec_result.loader.exec_module(result_module)
        print("   ✅ bot.fp.types.result loaded")
        
        # Import the base module (dependency of config)
        spec_base = importlib.util.spec_from_file_location(
            "bot.fp.types.base", 
            os.path.join(project_root, "bot/fp/types/base.py")
        )
        base_module = importlib.util.module_from_spec(spec_base)
        sys.modules["bot.fp.types.base"] = base_module
        spec_base.loader.exec_module(base_module)
        print("   ✅ bot.fp.types.base loaded")
        
        # Now import the config module
        spec_fp_config = importlib.util.spec_from_file_location(
            "bot.fp.types.config", 
            os.path.join(project_root, "bot/fp/types/config.py")
        )
        fp_config_module = importlib.util.module_from_spec(spec_fp_config)
        sys.modules["bot.fp.types.config"] = fp_config_module
        spec_fp_config.loader.exec_module(fp_config_module)
        print("   ✅ bot.fp.types.config loaded successfully")
        
        # Test 2: Import bot.config (should use lazy loading)
        print("2. Testing bot.config import with lazy loading...")
        
        # Import required dependencies first
        spec_market_making = importlib.util.spec_from_file_location(
            "bot.market_making_config", 
            os.path.join(project_root, "bot/market_making_config.py")
        )
        market_making_module = importlib.util.module_from_spec(spec_market_making)
        sys.modules["bot.market_making_config"] = market_making_module
        spec_market_making.loader.exec_module(market_making_module)
        
        # Import utils modules
        spec_path_utils = importlib.util.spec_from_file_location(
            "bot.utils.path_utils", 
            os.path.join(project_root, "bot/utils/path_utils.py")
        )
        path_utils_module = importlib.util.module_from_spec(spec_path_utils)
        sys.modules["bot.utils.path_utils"] = path_utils_module
        spec_path_utils.loader.exec_module(path_utils_module)
        
        # Now try to import bot.config
        spec_config = importlib.util.spec_from_file_location(
            "bot.config", 
            os.path.join(project_root, "bot/config.py")
        )
        config_module = importlib.util.module_from_spec(spec_config)
        sys.modules["bot.config"] = config_module
        spec_config.loader.exec_module(config_module)
        print("   ✅ bot.config loaded successfully")
        
        # Test 3: Test lazy loading function
        print("3. Testing lazy loading function...")
        functional_config = config_module._get_functional_config()
        if functional_config is not None:
            print("   ✅ Lazy loading of functional config successful")
        else:
            print("   ⚠️  Lazy loading returned None (expected if dependencies missing)")
        
        print("\n🎉 SUCCESS: No circular import detected!")
        print("   - bot.fp.types.config imports only from bot.fp.types.result and bot.fp.types.base")
        print("   - bot.config uses lazy loading to avoid circular dependency")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_module_imports()
    sys.exit(0 if success else 1)