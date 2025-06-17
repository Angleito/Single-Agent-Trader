#!/usr/bin/env python3
"""Close the futures position."""

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
    FUTURES_SYMBOL = "ET-27JUN25-CDE"

    print("CLOSING FUTURES POSITION")
    print("=" * 60)

    try:
        # Check current position
        print("\nCurrent Position:")
        positions = client.list_futures_positions()
        position_to_close = None

        for pos in positions.positions:
            print(f"  {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts")
            if pos.product_id == FUTURES_SYMBOL:
                position_to_close = pos

        if not position_to_close:
            print("\n✓ No position to close")
            return

        # To close a SHORT position, we need to BUY
        contracts = position_to_close.number_of_contracts

        # If it's nano futures (0.1 ETH per contract), calculate the ETH amount
        eth_amount = float(contracts) * 0.1

        print(f"\nClosing {contracts} SHORT contracts ({eth_amount} ETH)...")
        print("Placing BUY order to close position...")

        order_result = client.create_order(
            client_order_id=str(uuid.uuid4()),
            product_id=FUTURES_SYMBOL,
            side="BUY",  # Buy to close short
            order_configuration={"market_market_ioc": {"base_size": str(eth_amount)}},
        )

        if order_result.success:
            print("\n✅ POSITION CLOSED SUCCESSFULLY!")
            resp = order_result.success_response
            if isinstance(resp, dict):
                order_id = resp.get("order_id")
            else:
                order_id = resp.order_id if hasattr(resp, "order_id") else None
            print(f"  Order ID: {order_id}")

            # Check final position
            await asyncio.sleep(3)
            print("\nFinal Position Check:")
            positions = client.list_futures_positions()
            if len(positions.positions) == 0:
                print("  ✓ All positions closed!")
            else:
                for pos in positions.positions:
                    print(
                        f"  {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts"
                    )
        else:
            print("\n❌ Order failed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(close_position())
