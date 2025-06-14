#!/usr/bin/env python3
"""Check status of futures sweeps."""

import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from datetime import datetime

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

print("FUTURES SWEEP STATUS")
print("=" * 60)

try:
    # Get current balances first
    fcm = client.get_futures_balance_summary()
    bs = fcm.balance_summary
    
    print(f"\nCurrent Balances:")
    print(f"  CBI (Spot): ${bs.cbi_usd_balance['value']}")
    print(f"  CFM (Futures): ${bs.cfm_usd_balance['value']}")
    print(f"  Total: ${bs.total_usd_balance['value']}")
    print(f"  Futures Buying Power: ${bs.futures_buying_power['value']}")
    
    # List all sweeps
    print(f"\nRecent Sweeps:")
    print("-" * 60)
    
    sweeps = client.list_futures_sweeps()
    if sweeps.sweeps:
        for sweep in sweeps.sweeps[:5]:  # Show last 5 sweeps
            print(f"\nSweep ID: {sweep.id}")
            print(f"  Amount: ${sweep.requested_amount['value']} {sweep.requested_amount['currency']}")
            print(f"  Status: {sweep.status}")
            print(f"  Scheduled: {sweep.scheduled_time}")
            
            # Calculate time ago
            try:
                scheduled_time = datetime.fromisoformat(sweep.scheduled_time.replace('Z', '+00:00'))
                time_ago = datetime.now(scheduled_time.tzinfo) - scheduled_time
                minutes_ago = int(time_ago.total_seconds() / 60)
                
                if minutes_ago < 60:
                    print(f"  Time: {minutes_ago} minutes ago")
                else:
                    hours_ago = minutes_ago / 60
                    print(f"  Time: {hours_ago:.1f} hours ago")
            except:
                pass
                
            # Highlight pending sweeps
            if sweep.status == "PENDING":
                print("  ⏳ PENDING - Should complete within 10-30 minutes")
            elif sweep.status == "PROCESSED":
                print("  ✅ COMPLETED")
            
    else:
        print("  No sweeps found")
        
    # Check for pending sweeps
    pending_sweeps = [s for s in sweeps.sweeps if s.status == "PENDING"]
    if pending_sweeps:
        total_pending = sum(float(s.requested_amount['value']) for s in pending_sweeps)
        print(f"\n⚠️  You have {len(pending_sweeps)} pending sweep(s) totaling ${total_pending:.2f}")
        print("   These should complete within 10-30 minutes")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()