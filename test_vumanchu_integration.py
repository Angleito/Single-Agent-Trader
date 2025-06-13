#!/usr/bin/env python3
"""
Test script to verify VuManChu indicators integration without running the full bot.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all VuManChu indicator modules can be imported."""
    try:
        print("🧪 Testing VuManChu indicator imports...")
        
        # Test individual components
        from bot.indicators.wavetrend import WaveTrend
        print("✅ WaveTrend imported successfully")
        
        from bot.indicators.cipher_a_signals import CipherASignals
        print("✅ CipherASignals imported successfully")
        
        from bot.indicators.cipher_b_signals import CipherBSignals
        print("✅ CipherBSignals imported successfully")
        
        from bot.indicators.ema_ribbon import EMAribbon
        print("✅ EMAribbon imported successfully")
        
        from bot.indicators.rsimfi import RSIMFIIndicator
        print("✅ RSIMFIIndicator imported successfully")
        
        from bot.indicators.stochastic_rsi import StochasticRSI
        print("✅ StochasticRSI imported successfully")
        
        from bot.indicators.schaff_trend_cycle import SchaffTrendCycle
        print("✅ SchaffTrendCycle imported successfully")
        
        from bot.indicators.sommi_patterns import SommiPatterns
        print("✅ SommiPatterns imported successfully")
        
        from bot.indicators.divergence_detector import DivergenceDetector
        print("✅ DivergenceDetector imported successfully")
        
        # Test main VuManChu class
        from bot.indicators.vumanchu import VuManChuIndicators, CipherA, CipherB
        print("✅ VuManChuIndicators, CipherA, CipherB imported successfully")
        
        print("🎯 All VuManChu indicator imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_main_bot_integration():
    """Test that the main bot can import the new indicators."""
    try:
        print("\n🤖 Testing main bot integration...")
        
        # Check if main.py uses the correct import
        with open('bot/main.py', 'r') as f:
            content = f.read()
            
        if 'from .indicators.vumanchu import VuManChuIndicators' in content:
            print("✅ Main bot has correct VuManChuIndicators import")
        else:
            print("❌ Main bot missing VuManChuIndicators import")
            return False
            
        if 'self.indicator_calc = VuManChuIndicators()' in content:
            print("✅ Main bot correctly instantiates VuManChuIndicators")
        else:
            print("❌ Main bot not using VuManChuIndicators")
            return False
            
        print("🎯 Main bot integration verified!")
        return True
        
    except FileNotFoundError:
        print("❌ bot/main.py not found")
        return False
    except Exception as e:
        print(f"❌ Error checking main bot: {e}")
        return False

def test_class_structure():
    """Test the class structure without full initialization."""
    try:
        print("\n🏗️  Testing class structure...")
        
        from bot.indicators.vumanchu import VuManChuIndicators
        
        # Check if class has expected methods
        indicator_calc = VuManChuIndicators.__new__(VuManChuIndicators)  # Create without __init__
        
        expected_methods = [
            'calculate_all',
            'get_latest_state', 
            'get_all_signals',
            'get_signal_strength',
            'interpret_signals'
        ]
        
        for method in expected_methods:
            if hasattr(VuManChuIndicators, method):
                print(f"✅ Method '{method}' found")
            else:
                print(f"❌ Method '{method}' missing")
                return False
                
        print("🎯 Class structure verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing class structure: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    try:
        print("\n📁 Testing file structure...")
        
        required_files = [
            'bot/indicators/vumanchu.py',
            'bot/indicators/wavetrend.py',
            'bot/indicators/cipher_a_signals.py',
            'bot/indicators/cipher_b_signals.py',
            'bot/indicators/ema_ribbon.py',
            'bot/indicators/rsimfi.py',
            'bot/indicators/stochastic_rsi.py',
            'bot/indicators/schaff_trend_cycle.py',
            'bot/indicators/sommi_patterns.py',
            'bot/indicators/divergence_detector.py',
            'docker-compose.orbstack.yml',
            'deploy-orbstack.sh',
            'monitor-orbstack.sh'
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - MISSING")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing {len(missing_files)} required files")
            return False
        else:
            print("🎯 All required files present!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing file structure: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting VuManChu integration tests...\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Main Bot Integration", test_main_bot_integration),
        ("Class Structure", test_class_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! VuManChu integration is ready for OrbStack deployment.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)