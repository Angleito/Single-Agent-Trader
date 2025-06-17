#!/usr/bin/env python3
"""
Test Bluefin integration in paper trading mode
This works without the Bluefin SDK installed
"""

import os
import sys

# Set environment for Bluefin paper trading
os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
os.environ['SYSTEM__DRY_RUN'] = 'true'
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['LLM__OPENAI_API_KEY'] = os.environ.get('LLM__OPENAI_API_KEY', 'sk-test-key-for-paper-trading')
os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY'] = os.environ.get('EXCHANGE__BLUEFIN_PRIVATE_KEY', 'suiprivkey1qp3aergkwm2z7fvuw7es4q6w040a9m8wcw84tm4ck3f40k9hfzfmv2v4vs9')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("BLUEFIN PAPER TRADING TEST")
print("=" * 60)
print()

# Test the configuration fix
print("Test 1: Testing configuration validation fix...")
try:
    from bot.config import settings
    print(f"✅ Configuration loaded successfully")
    print(f"   Exchange type: {settings.exchange.exchange_type}")
    print(f"   Dry run mode: {settings.system.dry_run}")
    print(f"   Bluefin network: {settings.exchange.bluefin_network}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# Test exchange creation
print("\nTest 2: Creating Bluefin exchange instance...")
try:
    from bot.exchange.factory import ExchangeFactory
    exchange = ExchangeFactory.create_exchange(exchange_type='bluefin', dry_run=True)
    print("✅ Exchange created successfully")
except Exception as e:
    print(f"❌ Exchange creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test connection
print("\nTest 3: Testing connection...")
try:
    import asyncio
    
    async def test_connection():
        connected = await exchange.connect()
        if connected:
            status = exchange.get_connection_status()
            print("✅ Connected successfully")
            print(f"   Trading mode: {status['trading_mode']}")
            print(f"   Blockchain: {status['blockchain']}")
            print(f"   Network: {status['network']}")
            
            # Test paper trading
            balance = await exchange.get_account_balance()
            print(f"   Mock balance: ${balance}")
            
            await exchange.disconnect()
            return True
        return False
    
    success = asyncio.run(test_connection())
    if not success:
        print("❌ Connection failed")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Connection test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test perpetual trading
print("\nTest 4: Testing perpetual trading functionality...")
try:
    from decimal import Decimal
    from bot.types import TradeAction
    
    async def test_perpetual():
        await exchange.connect()
        
        # Test symbol conversion
        symbol_map = {
            'BTC-USD': 'BTC-PERP',
            'ETH-USD': 'ETH-PERP',
        }
        
        print("   Symbol conversions:")
        for std, perp in symbol_map.items():
            converted = exchange._convert_symbol(std)
            print(f"     {std} -> {converted}")
        
        # Test paper trade
        trade_action = TradeAction(
            action='LONG',
            size_pct=10.0,
            leverage=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            confidence=0.85,
            reasoning='Paper trading test',
            rationale='Testing Bluefin perpetual trading in paper mode'
        )
        
        order = await exchange.execute_trade_action(
            trade_action,
            'ETH-USD',
            Decimal('3500.00')
        )
        
        if order:
            print(f"✅ Paper order placed: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Quantity: {order.quantity}")
        else:
            print("❌ Failed to place paper order")
            
        await exchange.disconnect()
        return order is not None
    
    success = asyncio.run(test_perpetual())
    if not success:
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Perpetual trading test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✅")
print("Bluefin paper trading is working correctly")
print("=" * 60)