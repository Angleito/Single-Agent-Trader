#!/usr/bin/env python3
"""Sweep funds from CBI to CFM - automatic version."""

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

print("AUTOMATIC FUTURES SWEEP")
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
    
    if cbi_balance > 10:
        # Sweep 80% of CBI balance, keeping some for fees
        amount = round(cbi_balance * 0.8, 2)
        
        print(f"\nSweeping ${amount} to CFM (80% of available)...")
        
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
        print(f"\n❌ Insufficient balance. Need at least $10 in CBI (current: ${cbi_balance})")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()