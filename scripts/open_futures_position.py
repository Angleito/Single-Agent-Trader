#!/usr/bin/env python3
"""Open a new futures position with correct sizing."""

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


async def open_position():
    futures_symbol = "ET-27JUN25-CDE"

    print("OPENING NEW FUTURES POSITION")
    print("=" * 60)

    try:
        # Check balance
        print("\n1. Checking futures balance...")
        fcm = client.get_futures_balance_summary()
        bs = fcm.balance_summary
        cfm_balance = float(bs.cfm_usd_balance["value"])
        buying_power = float(bs.futures_buying_power["value"])

        print(f"   CFM Balance: ${cfm_balance}")
        print(f"   Futures Buying Power: ${buying_power}")

        # Get current price
        print(f"\n2. Getting {futures_symbol} price...")
        product = client.get_product(futures_symbol)
        current_price = float(product.price)
        print(f"   Current price: ${current_price}")

        # Calculate position
        # 1 contract = 0.1 ETH
        # At ~$2540/ETH, 1 contract = ~$254 notional
        # With 5x leverage, need ~$50 margin per contract
        contracts_to_buy = 2  # 2 contracts = 0.2 ETH
        notional_value = contracts_to_buy * 0.1 * current_price
        estimated_margin = notional_value / 5  # Assuming 5x leverage

        print("\n3. Position calculation:")
        print(f"   Contracts: {contracts_to_buy}")
        print(f"   ETH amount: {contracts_to_buy * 0.1} ETH")
        print(f"   Notional value: ${notional_value:.2f}")
        print(f"   Estimated margin required: ${estimated_margin:.2f}")

        if buying_power < estimated_margin:
            print(
                f"\n❌ Insufficient buying power. Need ${estimated_margin:.2f}, have ${buying_power:.2f}"
            )
            return

        # Place the order
        print(f"\n4. Placing LONG order for {contracts_to_buy} contracts...")

        order_result = client.create_order(
            client_order_id=str(uuid.uuid4()),
            product_id=futures_symbol,
            side="BUY",  # Long position
            order_configuration={
                "market_market_ioc": {
                    "base_size": str(contracts_to_buy)
                }  # Size in contracts
            },
            # NO leverage parameter needed for actual futures contracts
        )

        if order_result.success:
            print("\n✅ FUTURES POSITION OPENED SUCCESSFULLY!")
            resp = order_result.success_response
            if isinstance(resp, dict):
                order_id = resp.get("order_id")
            else:
                order_id = resp.order_id if hasattr(resp, "order_id") else None
            print(f"   Order ID: {order_id}")

            # Check position
            await asyncio.sleep(3)
            print("\n5. New Position:")
            positions = client.list_futures_positions()
            for pos in positions.positions:
                print(
                    f"   {pos.product_id}: {pos.side} {pos.number_of_contracts} contracts"
                )
                print(f"   Entry price: ${pos.avg_entry_price}")
                if hasattr(pos, "unrealized_pnl"):
                    print(f"   Unrealized PnL: ${pos.unrealized_pnl}")

            # Check updated balance
            print("\n6. Updated Balance:")
            fcm = client.get_futures_balance_summary()
            bs = fcm.balance_summary
            print(f"   CFM Balance: ${bs.cfm_usd_balance['value']}")
            print(f"   Initial Margin: ${bs.initial_margin['value']}")
            print(f"   Available Margin: ${bs.available_margin['value']}")

        else:
            print("\n❌ Order failed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(open_position())
