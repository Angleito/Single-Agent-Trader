#!/usr/bin/env python3
"""Wait for sweep to complete and show progress."""

import asyncio
import os
import time

from coinbase.rest import RESTClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv("EXCHANGE__CDP_API_KEY_NAME")
cdp_private_key = os.getenv("EXCHANGE__CDP_PRIVATE_KEY")

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)


async def wait_for_sweep():
    print("WAITING FOR SWEEP TO COMPLETE")
    print("=" * 60)

    start_time = time.time()
    last_cfm_balance = 0

    while True:
        try:
            # Check sweep status
            sweeps = client.list_futures_sweeps()
            pending_sweeps = [s for s in sweeps.sweeps if s.status == "PENDING"]

            # Check balances
            fcm = client.get_futures_balance_summary()
            bs = fcm.balance_summary
            cfm_balance = float(bs.cfm_usd_balance["value"])
            cbi_balance = float(bs.cbi_usd_balance["value"])

            # Clear screen and show status
            print("\033[2J\033[H")  # Clear screen
            print("SWEEP STATUS MONITOR")
            print("=" * 60)
            print(f"Time elapsed: {int(time.time() - start_time)} seconds")
            print("\nBalances:")
            print(f"  CBI (Spot): ${cbi_balance:.2f}")
            print(f"  CFM (Futures): ${cfm_balance:.2f}")
            print(f"  Total: ${cbi_balance + cfm_balance:.2f}")

            if pending_sweeps:
                print(f"\n⏳ {len(pending_sweeps)} pending sweep(s):")
                for sweep in pending_sweeps:
                    print(
                        f"   - ${sweep.requested_amount['value']} scheduled at {sweep.scheduled_time}"
                    )
            else:
                print("\n✓ No pending sweeps")

            # Check if CFM balance increased
            if cfm_balance > last_cfm_balance and cfm_balance > 0:
                print(
                    f"\n🎉 SWEEP COMPLETED! CFM balance increased to ${cfm_balance:.2f}"
                )
                print("\nYou can now trade futures!")
                break

            last_cfm_balance = cfm_balance

            # Check every 30 seconds
            await asyncio.sleep(30)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(wait_for_sweep())
