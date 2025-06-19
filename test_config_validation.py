#!/usr/bin/env python3
"""
Test script for configuration validation system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_config():
    """Test basic configuration loading."""
    print("🔍 Testing basic configuration loading...")
    
    try:
        # Import with temporary environment override for testing
        import os
        original_key = os.environ.get('EXCHANGE__BLUEFIN_PRIVATE_KEY')
        
        # Set a valid test key for validation
        os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY'] = '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'
        
        from bot.config import create_settings
        
        settings = create_settings()
        print('✅ Configuration loaded successfully')
        print(f'   Exchange: {settings.exchange.exchange_type}')
        print(f'   Environment: {settings.system.environment.value}')
        print(f'   Dry Run: {settings.system.dry_run}')
        
        # Restore original key
        if original_key is not None:
            os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY'] = original_key
        elif 'EXCHANGE__BLUEFIN_PRIVATE_KEY' in os.environ:
            del os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY']
        
        return True
        
    except Exception as e:
        print(f'❌ Configuration loading failed: {e}')
        return False

def test_bluefin_validation():
    """Test Bluefin-specific validation."""
    print("\n🔷 Testing Bluefin configuration validation...")
    
    try:
        import os
        
        # Test different private key formats
        test_keys = [
            ('valid_hex', '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'),
            ('valid_hex_no_prefix', '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'),
            ('invalid_short', '0x123'),
            ('invalid_chars', '0xZZZZ567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'),
            ('mnemonic_12', 'abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about'),
            ('mnemonic_24', 'abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art'),
        ]
        
        results = []
        
        for test_name, test_key in test_keys:
            print(f"   Testing {test_name}...")
            
            # Set test environment
            os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY'] = test_key
            os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
            
            try:
                from bot.config import ExchangeSettings
                
                # Test the validation directly
                exchange_settings = ExchangeSettings(
                    exchange_type='bluefin',
                    bluefin_private_key=test_key,
                    bluefin_network='testnet'
                )
                
                print(f"     ✅ {test_name}: VALID")
                results.append((test_name, True, None))
                
            except Exception as e:
                print(f"     ❌ {test_name}: {str(e)}")
                results.append((test_name, False, str(e)))
        
        # Summary
        valid_count = sum(1 for _, valid, _ in results if valid)
        print(f"\n   Summary: {valid_count}/{len(results)} formats validated successfully")
        
        return True
        
    except Exception as e:
        print(f'❌ Bluefin validation test failed: {e}')
        return False

def test_configuration_features():
    """Test advanced configuration features."""
    print("\n⚙️  Testing advanced configuration features...")
    
    try:
        import os
        
        # Set up test environment
        os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY'] = '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef'
        os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
        os.environ['EXCHANGE__BLUEFIN_NETWORK'] = 'testnet'
        os.environ['SYSTEM__DRY_RUN'] = 'true'
        
        from bot.config import create_settings
        
        settings = create_settings()
        
        # Test configuration hash generation
        config_hash = settings.generate_config_hash()
        print(f"   ✅ Config hash generation: {config_hash[:16]}...")
        
        # Test configuration summary
        summary = settings.get_configuration_summary()
        print(f"   ✅ Configuration summary: {len(summary)} sections")
        
        # Test Bluefin-specific tests
        if settings.exchange.exchange_type == 'bluefin':
            bluefin_results = settings.test_bluefin_configuration()
            print(f"   ✅ Bluefin config test: {bluefin_results['status']}")
            
            # Show test results
            for test in bluefin_results['tests']:
                status_icon = {'pass': '✅', 'fail': '❌', 'warning': '⚠️', 'skip': '⏭️'}[test['status']]
                test_name = test['name'].replace('_', ' ').title()
                print(f"     {status_icon} {test_name}")
        
        # Test configuration backup
        backup = settings.create_backup_configuration()
        print(f"   ✅ Configuration backup: {len(backup['configuration'])} sections")
        
        # Test configuration monitor creation
        monitor = settings.create_configuration_monitor()
        print(f"   ✅ Configuration monitor: {monitor.initial_hash[:16]}...")
        
        return True
        
    except Exception as e:
        print(f'❌ Advanced features test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_validation_scripts():
    """Test validation script availability."""
    print("\n📜 Testing validation scripts...")
    
    scripts = [
        'services/scripts/validate_env.py',
        'scripts/validate_config.py', 
        'scripts/test_bluefin_config.py'
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"   ✅ {script}: Available")
            
            # Check if executable
            if script_path.stat().st_mode & 0o111:
                print(f"     ✅ Executable permissions set")
            else:
                print(f"     ⚠️  No executable permissions")
        else:
            print(f"   ❌ {script}: Not found")
    
    return True

def main():
    """Run all tests."""
    print("🔍 Configuration Validation System Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_config,
        test_bluefin_validation,
        test_configuration_features,
        test_validation_scripts,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        test_display = test_name.replace('test_', '').replace('_', ' ').title()
        print(f"{status} {test_display}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration validation system is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())