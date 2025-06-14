#!/usr/bin/env python3
"""Check move_portfolio_funds method."""

import os
import inspect
from dotenv import load_dotenv
from coinbase.rest import RESTClient

# Load environment variables
load_dotenv()

# Get CDP credentials
cdp_api_key = os.getenv('EXCHANGE__CDP_API_KEY_NAME')
cdp_private_key = os.getenv('EXCHANGE__CDP_PRIVATE_KEY')

# Initialize client
client = RESTClient(api_key=cdp_api_key, api_secret=cdp_private_key)

# Check move_portfolio_funds signature
print("move_portfolio_funds signature:")
sig = inspect.signature(client.move_portfolio_funds)
print(f"  {sig}")

# Check if we have portfolio IDs
print("\nChecking portfolios...")
try:
    portfolios = client.list_portfolios()
    print(f"Type: {type(portfolios)}")
    if hasattr(portfolios, 'portfolios'):
        for p in portfolios.portfolios:
            print(f"  - {p.name}: {p.uuid}")
except Exception as e:
    print(f"Error listing portfolios: {e}")

# Alternative: Check accounts and their portfolio IDs
print("\nChecking account portfolio IDs...")
accounts = client.get_accounts()
seen_portfolios = set()
for acc in accounts.accounts:
    if hasattr(acc, 'retail_portfolio_id') and acc.retail_portfolio_id:
        if acc.retail_portfolio_id not in seen_portfolios:
            seen_portfolios.add(acc.retail_portfolio_id)
            print(f"  - Portfolio ID: {acc.retail_portfolio_id} (from {acc.currency} account)")