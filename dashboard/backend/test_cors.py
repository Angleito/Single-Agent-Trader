#!/usr/bin/env python3
"""
Test script to verify CORS configuration
"""

import requests
import json

def test_cors_configuration():
    """Test CORS configuration for the dashboard backend"""
    base_url = "http://localhost:8000"
    frontend_origin = "http://localhost:3000"
    
    print("Testing CORS configuration...")
    print("=" * 50)
    
    # Test 1: Basic API endpoint with Origin header
    print("1. Testing API endpoint with CORS headers...")
    try:
        response = requests.get(
            f"{base_url}/api/status",
            headers={
                "Origin": frontend_origin,
                "Content-Type": "application/json"
            }
        )
        print(f"   Status Code: {response.status_code}")
        print("   CORS Headers:")
        cors_headers = {k: v for k, v in response.headers.items() 
                       if 'access-control' in k.lower() or 'content-type' in k.lower()}
        for header, value in cors_headers.items():
            print(f"     {header}: {value}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test 2: OPTIONS preflight request
    print("2. Testing OPTIONS preflight request...")
    try:
        response = requests.options(
            f"{base_url}/api/status",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        print(f"   Status Code: {response.status_code}")
        print("   Preflight Headers:")
        preflight_headers = {k: v for k, v in response.headers.items() 
                           if 'access-control' in k.lower()}
        for header, value in preflight_headers.items():
            print(f"     {header}: {value}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test 3: Health check endpoint
    print("3. Testing health check endpoint...")
    try:
        response = requests.get(
            f"{base_url}/health",
            headers={"Origin": frontend_origin}
        )
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test 4: Root endpoint
    print("4. Testing root endpoint...")
    try:
        response = requests.get(
            f"{base_url}/",
            headers={"Origin": frontend_origin}
        )
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    print("CORS configuration test completed!")
    print("=" * 50)
    print("\nNOTE: If the backend is not running, you'll see connection errors.")
    print("Start the backend with: python main.py")

if __name__ == "__main__":
    test_cors_configuration()