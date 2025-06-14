#!/usr/bin/env python3
"""Manually sweep funds from CBI to CFM."""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

print("MANUAL FUTURES SWEEP")
print("=" * 40)

try:
    # Check current balances
    fcm = client.get_futures_balance_summary()
    bs = fcm.balance_summary
    
    print(f"\nCurrent Balances:")
    print(f"  CBI (Spot): ${bs.cbi_usd_balance['value']}")
    print(f"  CFM (Futures): ${bs.cfm_usd_balance['value']}")
    print(f"  Total: ${bs.total_usd_balance['value']}")
    
    # Cancel any pending sweeps first
    try:
        client.cancel_pending_futures_sweep()
        print("\n✓ Cancelled pending sweep")
    except:
        print("\n✓ No pending sweep to cancel")
    
    # Ask how much to sweep
    cbi_balance = float(bs.cbi_usd_balance['value'])
    if cbi_balance > 0:
        amount = input(f"\nHow much to sweep to CFM? (max ${cbi_balance}): $")
        amount = float(amount)
        
        if amount > 0 and amount <= cbi_balance:
            # Schedule the sweep
            result = client.schedule_futures_sweep(usd_amount=str(amount))
            print(f"\n✅ SUCCESS! Scheduled ${amount} sweep to CFM")
            print("   Note: Sweeps typically process within 10-30 minutes")
            
            # Show the sweep details
            sweeps = client.list_futures_sweeps()
            if sweeps.sweeps:
                latest = sweeps.sweeps[0]
                print(f"\n   Sweep ID: {latest.id}")
                print(f"   Status: {latest.status}")
                print(f"   Scheduled: {latest.scheduled_time}")
        else:
            print(f"\n❌ Invalid amount. Must be between $0 and ${cbi_balance}")
    else:
        print("\n❌ No funds available in CBI to sweep")
        
except Exception as e:
    print(f"\n❌ Error: {e}")