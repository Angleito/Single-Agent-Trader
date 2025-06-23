#!/usr/bin/env python3
"""Test script to verify Bluefin service connection with fallback URLs."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.exchange.bluefin_client import BluefinServiceClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_connection():
    """Test Bluefin service connection with different scenarios."""

    print("\n" + "=" * 60)
    print("Testing Bluefin Service Connection")
    print("=" * 60 + "\n")

    # Test 1: Default connection
    print("Test 1: Default connection (Docker service name)")
    print("-" * 40)
    try:
        client = BluefinServiceClient()
        connected = await client.connect()
        print(f"✅ Connection successful: {connected}")
        print(f"   Active URL: {client.service_url}")
        print(f"   Service discovery complete: {client.service_discovery_complete}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

    await asyncio.sleep(1)

    # Test 2: Explicit localhost connection
    print("\n\nTest 2: Explicit localhost connection")
    print("-" * 40)
    try:
        client = BluefinServiceClient(service_url="http://localhost:8080")
        connected = await client.connect()
        print(f"✅ Connection successful: {connected}")
        print(f"   Active URL: {client.service_url}")
        print(f"   Service discovery complete: {client.service_discovery_complete}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

    await asyncio.sleep(1)

    # Test 3: Show all URLs being tried
    print("\n\nTest 3: Connection with debug info")
    print("-" * 40)
    try:
        client = BluefinServiceClient()
        print("URLs to try:")
        for i, url in enumerate(client.service_urls):
            print(f"   {i + 1}. {url}")
        print(f"\nIs running in Docker: {client.is_docker}")

        connected = await client.connect()
        print(f"\n✅ Connection result: {connected}")
        if connected:
            print(f"   Successfully connected to: {client.service_url}")
        else:
            print("   All URLs failed")

        # Show circuit breaker states
        print("\nCircuit breaker states:")
        for url, state in client.circuit_states.items():
            status = "OPEN" if state["open"] else "CLOSED"
            failures = state["consecutive_failures"]
            print(f"   {url}: {status} (failures: {failures})")

    except Exception as e:
        print(f"❌ Connection failed with error: {e}")
        import traceback

        traceback.print_exc()

    # Test 4: Make a simple API call
    print("\n\nTest 4: Making an API call (get_account_data)")
    print("-" * 40)
    try:
        if connected:
            # Try to get account data
            account_data = await client.get_account_data()
            print("✅ API call successful")
            print(f"   Account initialized: {account_data.get('initialized', False)}")
        else:
            print("⚠️  Skipping API call - not connected")
    except Exception as e:
        print(f"❌ API call failed: {e}")

    # Cleanup
    if hasattr(client, "disconnect") and callable(client.disconnect):
        await client.disconnect()

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Set a dummy API key if not provided
    if not os.getenv("BLUEFIN_SERVICE_API_KEY"):
        os.environ["BLUEFIN_SERVICE_API_KEY"] = "test-api-key-for-connection-testing"

    asyncio.run(test_connection())
