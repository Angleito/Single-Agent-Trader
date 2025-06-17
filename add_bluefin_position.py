#!/usr/bin/env python3
"""
Script to manually add a Bluefin position to the bot's position tracking.
This is useful when the bot can't connect to Bluefin service to query positions.
"""

import json
import os
from datetime import datetime
from decimal import Decimal

# Position details - UPDATE THESE WITH YOUR ACTUAL POSITION
SYMBOL = "SUI-PERP"
SIDE = "LONG"  # or "SHORT"
QUANTITY = Decimal("100")  # Your position size
ENTRY_PRICE = Decimal("2.80")  # Your entry price
TIMESTAMP = datetime.utcnow().isoformat()

# Path to position file
POSITION_FILE = "data/positions/fifo_positions.json"

def add_bluefin_position():
    """Add Bluefin position to the bot's tracking."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
    
    # Load existing positions
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {"positions": {}, "history": []}
    
    # Create position entry
    position_data = {
        "symbol": SYMBOL,
        "side": SIDE,
        "total_realized_pnl": "0",
        "lots": [
            {
                "lot_id": f"manual_bluefin_{SYMBOL}_{TIMESTAMP}",
                "quantity": str(QUANTITY),
                "purchase_price": str(ENTRY_PRICE),
                "purchase_date": TIMESTAMP,
                "remaining_quantity": str(QUANTITY)
            }
        ],
        "sale_history": []
    }
    
    # Add/update position
    data["positions"][SYMBOL] = position_data
    
    # Save updated positions
    with open(POSITION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Added {SIDE} position: {QUANTITY} {SYMBOL} @ ${ENTRY_PRICE}")
    print(f"📄 Position saved to: {POSITION_FILE}")
    
    # Show all positions
    print("\n📊 Current positions:")
    for symbol, pos in data["positions"].items():
        total_qty = sum(Decimal(lot["remaining_quantity"]) for lot in pos["lots"])
        if total_qty > 0:
            print(f"  - {pos['side']} {total_qty} {symbol}")

if __name__ == "__main__":
    print("🔧 Adding Bluefin position to bot tracking...")
    print("\n⚠️  IMPORTANT: Update the position details in this script first!")
    print(f"Current settings:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Side: {SIDE}")
    print(f"  Quantity: {QUANTITY}")
    print(f"  Entry Price: ${ENTRY_PRICE}")
    
    response = input("\nProceed with these values? (y/n): ")
    if response.lower() == 'y':
        add_bluefin_position()
    else:
        print("❌ Cancelled. Edit the script to set your actual position details.")