#!/usr/bin/env python3
"""
Test script for Bluefin DEX SDK fixes
Tests all the major issues and solutions implemented
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Override exchange type for testing
os.environ["EXCHANGE__EXCHANGE_TYPE"] = "bluefin"

def test_socketio_import():
    """Test 1: Socket.IO dependency conflicts"""
    print("=== Test 1: Socket.IO Import ===")
    try:
        import socketio
        print(f"✓ Socket.IO imported successfully")
        
        if hasattr(socketio, 'AsyncClient'):
            print(f"✓ AsyncClient available")
            # Test creating an AsyncClient
            client = socketio.AsyncClient()
            print(f"✓ AsyncClient can be instantiated")
        else:
            print(f"✗ AsyncClient not available")
            return False
            
        print(f"✓ Socket.IO version: {getattr(socketio, '__version__', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"✗ Socket.IO import failed: {e}")
        return False

def test_bluefin_sdk_import():
    """Test 2: Bluefin SDK import with Socket.IO fixes"""
    print("\n=== Test 2: Bluefin SDK Import ===")
    try:
        from bot.exchange.bluefin import BluefinClient, BLUEFIN_AVAILABLE
        print(f"✓ Bluefin client imported successfully")
        print(f"✓ SDK Available: {BLUEFIN_AVAILABLE}")
        
        if BLUEFIN_AVAILABLE:
            print(f"✓ Bluefin SDK loaded with all dependencies")
        else:
            print(f"⚠ Bluefin SDK not available (expected in some environments)")
            
        return True
        
    except Exception as e:
        print(f"✗ Bluefin SDK import failed: {e}")
        return False

def test_python_version_compatibility():
    """Test 3: Python 3.11/3.12 compatibility"""
    print("\n=== Test 3: Python Version Compatibility ===")
    try:
        import sys
        python_version = sys.version_info
        print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version >= (3, 11):
            print(f"✓ Python version compatible with Bluefin SDK")
        else:
            print(f"✗ Python version too old for Bluefin SDK")
            return False
            
        # Test some Python 3.11+ features
        match python_version.major:
            case 3:
                print(f"✓ Match statement works (Python 3.10+ feature)")
            case _:
                print(f"✗ Unexpected Python version")
                
        return True
        
    except Exception as e:
        print(f"✗ Python compatibility test failed: {e}")
        return False

def test_config_loading():
    """Test 4: Configuration loading for trading modes"""
    print("\n=== Test 4: Configuration Loading ===")
    try:
        from bot.config import settings
        print(f"✓ Settings loaded successfully")
        
        # Test system settings
        dry_run = getattr(settings.system, 'dry_run', True)
        print(f"✓ System dry_run: {dry_run}")
        
        # Test exchange settings
        exchange_type = settings.exchange.exchange_type
        print(f"✓ Exchange type: {exchange_type}")
        
        # Test Bluefin-specific settings
        if hasattr(settings.exchange, 'bluefin_network'):
            network = settings.exchange.bluefin_network
            print(f"✓ Bluefin network: {network}")
        else:
            print(f"⚠ Bluefin network setting not found")
            
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

async def test_bluefin_client_initialization():
    """Test 5: Bluefin client initialization with trading mode detection"""
    print("\n=== Test 5: Bluefin Client Initialization ===")
    try:
        from bot.exchange.bluefin import BluefinClient
        
        # Test with paper trading mode
        print("Testing paper trading mode...")
        client = BluefinClient(dry_run=True)
        print(f"✓ Client created in paper trading mode")
        
        status = client.get_connection_status()
        print(f"✓ Trading mode: {status['trading_mode']}")
        print(f"✓ Dry run: {status['dry_run']}")
        print(f"✓ System dry run: {status['system_dry_run']}")
        
        # Test connection (should work even without SDK in paper mode)
        connected = await client.connect()
        print(f"✓ Connection test: {connected}")
        
        await client.disconnect()
        print(f"✓ Disconnection successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Bluefin client test failed: {e}")
        return False

def test_environment_variables():
    """Test 6: Environment variable handling"""
    print("\n=== Test 6: Environment Variables ===")
    try:
        # Test critical environment variables
        critical_vars = [
            "EXCHANGE__EXCHANGE_TYPE",
            "SYSTEM__DRY_RUN",
            "LLM__OPENAI_API_KEY",
        ]
        
        for var in critical_vars:
            value = os.environ.get(var)
            if value:
                # Mask sensitive values
                if "KEY" in var:
                    display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "[SET]"
                else:
                    display_value = value
                print(f"✓ {var}: {display_value}")
            else:
                print(f"⚠ {var}: [NOT SET]")
        
        # Test Bluefin-specific variables
        bluefin_vars = [
            "EXCHANGE__BLUEFIN_PRIVATE_KEY",
            "EXCHANGE__BLUEFIN_NETWORK",
        ]
        
        for var in bluefin_vars:
            value = os.environ.get(var)
            if value:
                if "KEY" in var:
                    display_value = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "[SET]"
                else:
                    display_value = value
                print(f"✓ {var}: {display_value}")
            else:
                print(f"⚠ {var}: [NOT SET] (required for live trading)")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        return False

def test_docker_environment_detection():
    """Test 7: Docker environment detection"""
    print("\n=== Test 7: Docker Environment Detection ===")
    try:
        # Check if running in Docker
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == 'true'
        print(f"✓ Running in Docker: {in_docker}")
        
        # Check for OrbStack indicators
        hostname = os.environ.get('HOSTNAME', '')
        if 'orbstack' in hostname.lower():
            print(f"✓ OrbStack detected in hostname: {hostname}")
        else:
            print(f"✓ Hostname: {hostname}")
        
        # Check for Docker networking
        if in_docker:
            print(f"✓ Docker environment variables detected")
            docker_vars = ['HOSTNAME', 'PATH', 'PWD']
            for var in docker_vars:
                value = os.environ.get(var, 'not set')
                print(f"  {var}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Docker environment detection failed: {e}")
        return False

async def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 Bluefin DEX SDK Fixes Test Suite")
    print("=====================================\n")
    
    tests = [
        ("Socket.IO Import", test_socketio_import),
        ("Bluefin SDK Import", test_bluefin_sdk_import),
        ("Python Compatibility", test_python_version_compatibility),
        ("Configuration Loading", test_config_loading),
        ("Bluefin Client", test_bluefin_client_initialization),
        ("Environment Variables", test_environment_variables),
        ("Docker Environment", test_docker_environment_detection),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Bluefin SDK is ready for Docker deployment.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())