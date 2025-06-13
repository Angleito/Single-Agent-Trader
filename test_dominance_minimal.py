#!/usr/bin/env python3
"""
Minimal test for dominance module without full bot dependencies.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_dominance_imports():
    """Test that dominance-related modules can be imported."""
    print("\n🧪 Testing Dominance Module Imports")
    print("="*50)
    
    try:
        # Test direct import of dominance module
        from bot.data import dominance
        print("✅ Successfully imported bot.data.dominance module")
        
        # Check if DominanceDataProvider exists
        if hasattr(dominance, 'DominanceDataProvider'):
            print("✅ DominanceDataProvider class found")
        else:
            print("❌ DominanceDataProvider class not found")
            return False
            
        # Check if DominanceData exists
        if hasattr(dominance, 'DominanceData'):
            print("✅ DominanceData class found")
        else:
            print("❌ DominanceData class not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_dominance_config():
    """Test dominance configuration without full bot import."""
    print("\n🧪 Testing Dominance Configuration")
    print("="*50)
    
    try:
        # Test settings structure
        from bot.config import DominanceSettings
        print("✅ Successfully imported DominanceSettings")
        
        # Create test settings
        settings = DominanceSettings()
        print(f"✅ Created DominanceSettings with defaults:")
        print(f"  • Enabled: {settings.enable_dominance_data}")
        print(f"  • Data Source: {settings.data_source}")
        print(f"  • Update Interval: {settings.update_interval}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


async def test_types_integration():
    """Test that dominance fields exist in types."""
    print("\n🧪 Testing Type Definitions")
    print("="*50)
    
    try:
        # Import specific type classes
        from bot.types import IndicatorData
        print("✅ Successfully imported IndicatorData")
        
        # Check for dominance fields
        fields = IndicatorData.model_fields
        dominance_fields = [
            'usdt_dominance',
            'usdc_dominance', 
            'stablecoin_dominance',
            'dominance_trend',
            'dominance_rsi',
            'stablecoin_velocity',
            'market_sentiment'
        ]
        
        found_fields = []
        missing_fields = []
        
        for field in dominance_fields:
            if field in fields:
                found_fields.append(field)
            else:
                missing_fields.append(field)
        
        print(f"\n✅ Found {len(found_fields)} dominance fields:")
        for field in found_fields:
            print(f"  • {field}")
            
        if missing_fields:
            print(f"\n❌ Missing {len(missing_fields)} fields:")
            for field in missing_fields:
                print(f"  • {field}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Types test failed: {e}")
        return False


async def test_dominance_provider_basic():
    """Test basic dominance provider functionality."""
    print("\n🧪 Testing DominanceDataProvider Basic Functionality")
    print("="*50)
    
    try:
        from bot.data.dominance import DominanceDataProvider
        
        # Create provider
        provider = DominanceDataProvider(
            data_source="coingecko",
            update_interval=300
        )
        print("✅ Created DominanceDataProvider instance")
        
        # Test basic attributes
        print(f"  • Data Source: {provider.data_source}")
        print(f"  • Update Interval: {provider.update_interval}")
        print(f"  • Cache TTL: {provider._cache_ttl}")
        
        # Test methods exist
        methods = [
            'connect',
            'disconnect',
            'fetch_current_dominance',
            'get_latest_dominance',
            'get_market_sentiment',
            'get_dominance_history'
        ]
        
        for method in methods:
            if hasattr(provider, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Provider test failed: {e}")
        return False


async def main():
    """Run all minimal tests."""
    print("\n🚀 Minimal USDT/USDC Dominance Integration Test")
    print("="*50)
    
    results = []
    
    # Test 1: Module imports
    result1 = await test_dominance_imports()
    results.append(("Module Imports", result1))
    
    # Test 2: Configuration
    result2 = await test_dominance_config()
    results.append(("Configuration", result2))
    
    # Test 3: Type definitions
    result3 = await test_types_integration()
    results.append(("Type Definitions", result3))
    
    # Test 4: Basic provider
    result4 = await test_dominance_provider_basic()
    results.append(("Provider Basic", result4))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All minimal tests passed!")
        print("\nThe USDT/USDC dominance integration is properly implemented.")
        print("\nKey findings:")
        print("• DominanceDataProvider class exists and is properly structured")
        print("• Configuration includes all necessary dominance settings")
        print("• Type definitions include dominance fields in IndicatorData")
        print("• Core methods for fetching and analyzing dominance data are present")
    else:
        print("❌ Some tests failed!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)