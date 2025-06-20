#!/usr/bin/env python3
"""Close nano futures position - testing different size formats."""

import asyncio
import os
import uuid

from coinbase.rest import RESTClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv("EXCHANGE__CDP_API_KEY_NAME")
cdp_private_key = os.getenv("EXCHANGE__CDP_PRIVATE_KEY")

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)


async def close_position():
    futures_symbol = "ET-27JUN25-CDE"

    print("CLOSING NANO FUTURES POSITION")
    print("=" * 60)

    try:
        # Check position details
        print("\nChecking position details...")
        positions = client.list_futures_positions()

        for pos in positions.positions:
            if pos.product_id == futures_symbol:
                print("\nPosition Found:")
                print(f"  Product: {pos.product_id}")
                print(f"  Side: {pos.side}")
                print(f"  Contracts: {pos.number_of_contracts}")
                print(f"  Entry Price: ${pos.avg_entry_price}")

                # Try different approaches
                print("\n\nTesting different order sizes:")

                # Test 1: Try with "1" (matching number_of_contracts)
                print("\n1. Trying size='1' (contract count)...")
                try:
                    order_result = client.create_order(
                        client_order_id=str(uuid.uuid4()),
                        product_id=futures_symbol,
                        side="BUY",
                        order_configuration={"market_market_ioc": {"base_size": "1"}},
                    )
                    if order_result.success:
                        print("  ✅ SUCCESS with size='1'!")
                        return
                except Exception as e:
                    print(f"  ❌ Failed: {str(e)[:100]}")

                # Test 2: Try with "0.1" (ETH amount)
                print("\n2. Trying size='0.1' (ETH amount)...")
                try:
                    order_result = client.create_order(
                        client_order_id=str(uuid.uuid4()),
                        product_id=futures_symbol,
                        side="BUY",
                        order_configuration={"market_market_ioc": {"base_size": "0.1"}},
                    )
                    if order_result.success:
                        print("  ✅ SUCCESS with size='0.1'!")
                        return
                except Exception as e:
                    print(f"  ❌ Failed: {str(e)[:100]}")

                # Test 3: Try with reduce_only flag
                print("\n3. Trying with reduce_only=true...")
                try:
                    order_result = client.create_order(
                        client_order_id=str(uuid.uuid4()),
                        product_id=futures_symbol,
                        side="BUY",
                        order_configuration={"market_market_ioc": {"base_size": "1"}},
                        reduce_only=True,
                    )
                    if order_result.success:
                        print("  ✅ SUCCESS with reduce_only!")
                        return
                except Exception as e:
                    print(f"  ❌ Failed: {str(e)[:100]}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(close_position())
