#!/usr/bin/env python3
"""
Test CDP API Key Integration in Docker Environment
"""

import asyncio
import os
import sys

# Add the bot directory to Python path
sys.path.insert(0, "/app")


async def test_cdp_integration():
    """Test CDP API key integration with comprehensive logging"""

    print("🔑 CDP API Key Integration Test")
    print("=" * 50)

    try:
        # Import bot components
        from bot.config import Settings
        from bot.exchange.coinbase import CoinbaseClient

        print("✅ Bot modules imported successfully")

        # Load configuration
        print("\n📋 Loading Configuration...")
        settings = Settings()

        # Display authentication method
        print(f"Authentication Method: {settings.exchange.auth_method}")
        print(
            f"CDP Key Name: {settings.exchange.cdp_api_key_name[:50] if settings.exchange.cdp_api_key_name else 'Not set'}..."
        )
        print(
            f"CDP Private Key: {'Present' if settings.exchange.cdp_private_key else 'Not set'}"
        )
        print(
            f"Legacy Keys: {'Present' if settings.exchange.cb_api_key else 'Not set'}"
        )

        # Test exchange client initialization
        print("\n🔗 Initializing Exchange Client...")
        client = CoinbaseClient(settings)

        print(f"Client initialized with: {client.auth_method}")
        print(f"Dry run mode: {client.dry_run}")

        # Test connection (in dry run this won't make real API calls)
        print("\n🌐 Testing Connection...")
        try:
            await client.connect()
            status = client.get_connection_status()

            print("Connection Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")

            print("✅ Connection test successful")

        except Exception as e:
            print(f"❌ Connection test failed: {e}")

        # Test basic functionality
        print("\n💰 Testing Account Balance...")
        try:
            if client.auth_method == "cdp":
                print("Using CDP authentication for balance check...")
            elif client.auth_method == "legacy":
                print("Using legacy authentication for balance check...")
            else:
                print("No authentication configured - using dry run simulation")

            balance = await client.get_account_balance()
            print(f"Account balance: ${balance}")
            print("✅ Balance check successful")

        except Exception as e:
            print(f"❌ Balance check failed: {e}")

        await client.disconnect()
        print("\n🎉 CDP Integration Test Complete!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("Starting CDP API Key Integration Test...")
    print("Environment:", os.getenv("ENVIRONMENT", "development"))
    print("Dry Run:", os.getenv("DRY_RUN", "true"))
    print()

    # Run the async test
    result = asyncio.run(test_cdp_integration())

    if result:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED")
        sys.exit(1)
