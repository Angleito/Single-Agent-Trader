#!/usr/bin/env python3
"""Sweep ALL funds from CBI to CFM."""

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

print("SWEEP ALL FUNDS TO FUTURES")
print("=" * 40)

try:
    # Check current balances
    fcm = client.get_futures_balance_summary()
    bs = fcm.balance_summary
    
    cbi_balance = float(bs.cbi_usd_balance['value'])
    cfm_balance = float(bs.cfm_usd_balance['value'])
    
    print(f"\nCurrent Balances:")
    print(f"  CBI (Spot): ${cbi_balance}")
    print(f"  CFM (Futures): ${cfm_balance}")
    print(f"  Total: ${bs.total_usd_balance['value']}")
    
    # Cancel any pending sweeps first
    try:
        client.cancel_pending_futures_sweep()
        print("\n✓ Cancelled pending sweep")
    except:
        print("\n✓ No pending sweep to cancel")
    
    if cbi_balance > 0:
        # Sweep ALL available balance
        amount = cbi_balance
        
        print(f"\nSweeping ALL ${amount} to CFM...")
        
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
            print(f"\nAfter sweep completes:")
            print(f"  CBI (Spot): $0")
            print(f"  CFM (Futures): ~${cfm_balance + amount}")
    else:
        print(f"\n❌ No balance in CBI to sweep (current: ${cbi_balance})")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()