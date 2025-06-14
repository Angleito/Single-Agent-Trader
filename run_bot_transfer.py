#!/usr/bin/env python3
"""
Use the bot's built-in transfer mechanism directly.
"""

import asyncio
import os
from decimal import Decimal
from dotenv import load_dotenv

# Override dry run
os.environ['SYSTEM__DRY_RUN'] = 'false'
os.environ['SYSTEM__ENVIRONMENT'] = 'production'

# Import after setting environment
from bot.exchange.coinbase import CoinbaseClient


async def main():
    """Execute transfer using bot's mechanism."""
    # Load environment variables
    load_dotenv()
    
    print("USING BOT'S TRANSFER MECHANISM")
    print("=" * 60)
    
    # Initialize client
    client = CoinbaseClient()
    
    try:
        # Connect
        print("Connecting to Coinbase...")
        await client.connect()
        
        # Get futures balance summary to see CBI balance
        print("\nChecking balances...")
        
        # Direct API call to get balances
        fcm_response = await client._retry_request(
            client._client.get_fcm_balance_summary
        )
        
        bs = fcm_response.balance_summary
        cbi_balance = Decimal(bs.cbi_usd_balance['value'])
        cfm_balance = Decimal(bs.cfm_usd_balance['value'])
        
        print(f"\nCurrent Balances:")
        print(f"  CBI (Spot):    ${cbi_balance:,.2f}")
        print(f"  CFM (Futures): ${cfm_balance:,.2f}")
        
        if cbi_balance <= 0:
            print("\nNo funds to transfer.")
            return
        
        # Use the bot's transfer method
        print(f"\n💸 Transferring ${cbi_balance:,.2f} using bot's method...")
        
        success = await client.transfer_cash_to_futures(
            amount=cbi_balance,
            reason="MANUAL"
        )
        
        if success:
            print("✅ Transfer successful!")
            
            # Wait and check
            await asyncio.sleep(5)
            
            # Check again
            fcm_new = await client._retry_request(
                client._client.get_fcm_balance_summary
            )
            
            bs_new = fcm_new.balance_summary
            new_cbi = Decimal(bs_new.cbi_usd_balance['value'])
            new_cfm = Decimal(bs_new.cfm_usd_balance['value'])
            
            print(f"\nUpdated Balances:")
            print(f"  CBI (Spot):    ${new_cbi:,.2f}")
            print(f"  CFM (Futures): ${new_cfm:,.2f}")
            
            transferred = cbi_balance - new_cbi
            print(f"\n💰 CASH AMOUNT TRANSFERRED: ${transferred:,.2f}")
        else:
            print("❌ Transfer failed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())