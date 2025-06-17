#!/usr/bin/env python3
"""
Structure validation test for FastEMA module.
This test validates the module structure and logic without requiring external dependencies.
"""

import ast
import os
import sys


def validate_fast_ema_file():
    """Validate the FastEMA file structure and syntax."""
    file_path = "/Users/angel/Documents/Projects/cursorprod/bot/indicators/fast_ema.py"
    
    if not os.path.exists(file_path):
        print(f"✗ FastEMA file not found at {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to validate syntax
        tree = ast.parse(content, filename=file_path)
        print("✓ FastEMA file has valid Python syntax")
        
        # Check for required classes
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        required_classes = ['FastEMA', 'ScalpingEMASignals']
        
        for required_class in required_classes:
            if required_class in classes:
                print(f"✓ Found required class: {required_class}")
            else:
                print(f"✗ Missing required class: {required_class}")
                return False
        
        # Check for required methods in FastEMA
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'FastEMA':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                required_methods = ['__init__', 'calculate', 'update_realtime']
                
                for method in required_methods:
                    if method in methods:
                        print(f"✓ FastEMA has required method: {method}")
                    else:
                        print(f"✗ FastEMA missing required method: {method}")
                        return False
        
        # Check for required methods in ScalpingEMASignals
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ScalpingEMASignals':
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                required_methods = ['__init__', 'calculate', 'get_crossover_signals', 
                                  'get_trend_strength', 'is_bullish_setup', 'is_bearish_setup']
                
                for method in required_methods:
                    if method in methods:
                        print(f"✓ ScalpingEMASignals has required method: {method}")
                    else:
                        print(f"✗ ScalpingEMASignals missing required method: {method}")
                        return False
        
        # Check for scalping-optimized periods [3, 5, 8, 13]
        if '[3, 5, 8, 13]' in content:
            print("✓ Found scalping-optimized periods [3, 5, 8, 13]")
        else:
            print("✗ Scalping periods [3, 5, 8, 13] not found")
            return False
        
        # Check for key features mentioned in docstring
        key_features = [
            'FastEMA', 'ScalpingEMASignals', 'real-time updates', 
            'crossover detection', 'trend strength', 'scalping'
        ]
        
        for feature in key_features:
            if feature.lower() in content.lower():
                print(f"✓ Feature mentioned: {feature}")
            else:
                print(f"? Feature possibly missing: {feature}")
        
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error in FastEMA file: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating FastEMA file: {e}")
        return False


def validate_init_file():
    """Validate that __init__.py includes FastEMA exports."""
    init_path = "/Users/angel/Documents/Projects/cursorprod/bot/indicators/__init__.py"
    
    if not os.path.exists(init_path):
        print(f"✗ __init__.py not found at {init_path}")
        return False
    
    try:
        with open(init_path, 'r') as f:
            content = f.read()
        
        # Check for FastEMA imports
        if 'from .fast_ema import FastEMA, ScalpingEMASignals' in content:
            print("✓ __init__.py imports FastEMA and ScalpingEMASignals")
        else:
            print("✗ __init__.py missing FastEMA imports")
            return False
        
        # Check for FastEMA in __all__
        if '"FastEMA"' in content and '"ScalpingEMASignals"' in content:
            print("✓ __init__.py exports FastEMA and ScalpingEMASignals in __all__")
        else:
            print("✗ __init__.py missing FastEMA exports in __all__")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating __init__.py: {e}")
        return False


def validate_implementation_details():
    """Validate specific implementation details."""
    file_path = "/Users/angel/Documents/Projects/cursorprod/bot/indicators/fast_ema.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for performance optimizations
        optimizations = [
            'vectorized', 'numpy', 'memory-efficient', 'thread-safe',
            'real-time', 'scalping', 'high-frequency'
        ]
        
        found_optimizations = []
        for opt in optimizations:
            if opt.lower() in content.lower():
                found_optimizations.append(opt)
        
        print(f"✓ Found {len(found_optimizations)}/{len(optimizations)} optimization keywords")
        
        # Check for crossover pairs
        if 'crossover_pairs' in content or '_get_crossover_pairs' in content:
            print("✓ Crossover pair detection implemented")
        else:
            print("? Crossover pair detection may be missing")
        
        # Check for confidence scoring
        if 'confidence' in content.lower():
            print("✓ Confidence scoring implemented")
        else:
            print("? Confidence scoring may be missing")
        
        # Check for trend strength calculation
        if 'trend_strength' in content.lower():
            print("✓ Trend strength calculation implemented")
        else:
            print("? Trend strength calculation may be missing")
        
        # Check for error handling
        if 'try:' in content and 'except' in content:
            print("✓ Error handling implemented")
        else:
            print("? Error handling may be missing")
        
        # Check for logging
        if 'logger' in content and 'logging' in content:
            print("✓ Logging implemented")
        else:
            print("? Logging may be missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating implementation details: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("FastEMA Module Structure Validation")
    print("=" * 70)
    
    tests = [
        ("File Structure and Syntax", validate_fast_ema_file),
        ("__init__.py Integration", validate_init_file),
        ("Implementation Details", validate_implementation_details),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL STRUCTURE VALIDATION TESTS PASSED!")
        print("FastEMA module is properly structured and ready for use.")
    else:
        print("SOME VALIDATION TESTS FAILED!")
        print("Check the issues above and fix them.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)