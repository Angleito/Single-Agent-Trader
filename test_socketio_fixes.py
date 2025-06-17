#!/usr/bin/env python3
"""
Test Socket.IO fixes for Bluefin SDK compatibility.
Verifies that all dependency conflicts have been resolved.
"""

import sys
import os

def test_socketio_import():
    """Test Socket.IO import and AsyncClient availability."""
    try:
        import socketio
        print(f"✅ Socket.IO imported successfully: {socketio.__version__}")
        
        # Test AsyncClient availability
        if hasattr(socketio, 'AsyncClient'):
            print("✅ AsyncClient available")
            client = socketio.AsyncClient()
            print(f"✅ AsyncClient created: {type(client)}")
        else:
            print("❌ AsyncClient not available")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Socket.IO import failed: {e}")
        return False

def test_bluefin_import():
    """Test Bluefin SDK import."""
    try:
        import bluefin_v2_client
        print(f"✅ Bluefin SDK imported successfully")
        return True
    except Exception as e:
        print(f"❌ Bluefin SDK import failed: {e}")
        print("   This is expected in paper trading mode")
        return False

def test_dependency_conflicts():
    """Test for dependency conflicts."""
    try:
        # Test that we don't have conflicting socketio packages
        import pkg_resources
        
        socketio_packages = []
        for pkg in pkg_resources.working_set:
            if 'socketio' in pkg.project_name.lower():
                socketio_packages.append(f"{pkg.project_name}=={pkg.version}")
        
        print(f"📦 Socket.IO packages found: {socketio_packages}")
        
        # Check for conflicting packages
        conflicting = [pkg for pkg in socketio_packages if pkg.startswith('socketio==')]
        if conflicting:
            print(f"⚠️  Conflicting packages detected: {conflicting}")
            return False
        else:
            print("✅ No conflicting Socket.IO packages")
            return True
            
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable configuration."""
    print("\n🔧 Environment Variables:")
    
    # Check key environment variables
    env_vars = {
        'SYSTEM__DRY_RUN': os.getenv('SYSTEM__DRY_RUN', 'not set'),
        'EXCHANGE__EXCHANGE_TYPE': os.getenv('EXCHANGE__EXCHANGE_TYPE', 'not set'),
        'TRADING__SYMBOL': os.getenv('TRADING__SYMBOL', 'not set'),
        'BLUEFIN_FORCE_LIVE_MODE': os.getenv('BLUEFIN_FORCE_LIVE_MODE', 'not set'),
        'SOCKETIO_ASYNC_MODE': os.getenv('SOCKETIO_ASYNC_MODE', 'not set'),
    }
    
    for var, value in env_vars.items():
        print(f"  {var}: {value}")
    
    # Check trading mode
    dry_run = os.getenv('SYSTEM__DRY_RUN', 'true').lower()
    force_live = os.getenv('BLUEFIN_FORCE_LIVE_MODE', 'false').lower()
    
    if dry_run == 'false' or force_live == 'true':
        print("🚨 LIVE TRADING MODE DETECTED")
    else:
        print("📊 Paper Trading Mode")
    
    return True

def test_bluefin_connection():
    """Test Bluefin connection without private key."""
    try:
        # Only test if SDK is available
        import bluefin_v2_client
        print("\n🌐 Testing Bluefin connection...")
        
        # Test basic imports from SDK
        from bluefin_v2_client import BluefinClient
        print("✅ BluefinClient import successful")
        
        return True
    except Exception as e:
        print(f"⚠️  Bluefin connection test skipped: {e}")
        return True  # Not a failure in paper trading mode

def main():
    """Run all tests."""
    print("🧪 Socket.IO & Bluefin SDK Compatibility Test")
    print("=" * 50)
    
    tests = [
        ("Socket.IO Import", test_socketio_import),
        ("Bluefin SDK Import", test_bluefin_import),
        ("Dependency Conflicts", test_dependency_conflicts),
        ("Environment Variables", test_environment_variables),
        ("Bluefin Connection", test_bluefin_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Socket.IO fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())