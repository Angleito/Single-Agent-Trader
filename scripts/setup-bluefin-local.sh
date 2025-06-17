#!/bin/bash
# Setup script for local Bluefin development with UV

set -e

echo "=== Bluefin Local Development Setup with UV ==="
echo

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ UV installed: $(uv --version)"

# Create a dedicated virtual environment for Bluefin
echo
echo "Creating virtual environment for Bluefin development..."
uv venv .venv-bluefin --python 3.11

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv-bluefin/bin/activate

# Install dependencies using UV
echo
echo "Installing dependencies with UV (this is fast!)..."
uv pip install -r pyproject.bluefin.toml

# Verify Bluefin SDK installation
echo
echo "Verifying Bluefin SDK installation..."
python -c "
try:
    import bluefin_v2_client
    from bluefin_v2_client import BluefinClient, Networks
    print('✅ Bluefin SDK installed successfully')
    print(f'   Version: {bluefin_v2_client.__version__ if hasattr(bluefin_v2_client, \"__version__\") else \"Unknown\"}')
    print('   Available classes: BluefinClient, Networks')
except Exception as e:
    print(f'❌ Bluefin SDK import failed: {e}')
    exit(1)
"

# Create a test script
cat > test_bluefin_local.py << 'EOF'
#!/usr/bin/env python3
"""Test Bluefin integration locally with SDK installed"""

import os
import asyncio
from decimal import Decimal

# Set environment
os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
os.environ['SYSTEM__DRY_RUN'] = 'true'

from bot.exchange.bluefin import BluefinClient
from bot.types import TradeAction

async def test_bluefin():
    print("Testing Bluefin with SDK installed...")
    
    # Create client
    client = BluefinClient(dry_run=True)
    
    # Test connection
    connected = await client.connect()
    print(f"Connected: {connected}")
    
    if connected:
        # Get status
        status = client.get_connection_status()
        print(f"Status: {status['trading_mode']}")
        
        # Test symbol conversion
        print(f"ETH-USD -> {client._convert_symbol('ETH-USD')}")
        
        # Test paper trade
        trade_action = TradeAction(
            action='LONG',
            size_pct=10.0,
            leverage=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            confidence=0.85,
            reasoning='Test trade',
            rationale='Testing with SDK'
        )
        
        order = await client.execute_trade_action(
            trade_action,
            'ETH-USD',
            Decimal('3500.00')
        )
        
        if order:
            print(f"✅ Order placed: {order.id}")
        else:
            print("❌ Order failed")
            
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_bluefin())
EOF

chmod +x test_bluefin_local.py

echo
echo "=== Setup Complete! ==="
echo
echo "To use this environment:"
echo "  source .venv-bluefin/bin/activate"
echo "  python test_bluefin_local.py"
echo
echo "To run the bot with Bluefin:"
echo "  python -m bot.main live --dry-run --symbol ETH-PERP"
echo
echo "Environment info:"
echo "  Python: $(python --version)"
echo "  Virtual env: .venv-bluefin"
echo "  Bluefin SDK: Installed ✅"