#!/usr/bin/env python3
"""
Debug the import issue in market.py
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

print("Testing imports step by step...")

# Test 1: Direct import
print("\n1. Testing direct import of coinbase.jwt_generator:")
try:
    from coinbase import jwt_generator
    print(f"✅ Success: {type(jwt_generator)}")
except ImportError as e:
    print(f"❌ Failed: {e}")

# Test 2: What happens when we import market.py
print("\n2. Testing what imports are available when importing market.py:")

# First, let's see what's in the market.py import block
import ast

with open('bot/data/market.py', 'r') as f:
    content = f.read()

# Parse the AST to find the import statements
tree = ast.parse(content)

print("Import statements in market.py:")
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            print(f"  import {alias.name}")
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            print(f"  from {module} import {alias.name}")

# Test 3: Simulate the exact import sequence
print("\n3. Simulating the exact import sequence from market.py:")

try:
    print("  Importing coinbase.rest...")
    from coinbase.rest import RESTClient as _BaseClient
    print(f"  ✅ RESTClient: {_BaseClient}")
    
    print("  Importing coinbase.rest exception...")
    from coinbase.rest import RESTAPIException as _CBApiEx
    print(f"  ✅ Exception: {_CBApiEx}")
    
    print("  Importing coinbase.jwt_generator...")
    from coinbase import jwt_generator
    print(f"  ✅ jwt_generator: {type(jwt_generator)}")
    
    print("  All imports successful - should not use mock")
    
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    print("  Would use mock jwt_generator")

# Test 4: Check if it's a path issue
print("\n4. Checking Python path and coinbase package location:")
try:
    import coinbase
    print(f"  Coinbase package location: {coinbase.__file__}")
    print(f"  Coinbase package dir: {dir(coinbase)}")
except Exception as e:
    print(f"  Error checking coinbase package: {e}")

# Test 5: Check the coinbase.rest specifically
print("\n5. Checking coinbase.rest specifically:")
try:
    import coinbase.rest
    print(f"  coinbase.rest location: {coinbase.rest.__file__}")
    print(f"  coinbase.rest dir: {dir(coinbase.rest)}")
except Exception as e:
    print(f"  Error checking coinbase.rest: {e}")