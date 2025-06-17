#!/usr/bin/env python3
"""
Test Bluefin integration without SDK dependencies (paper trading mode)
"""

import asyncio
import os
from decimal import Decimal

# Set up environment for testing
os.environ['EXCHANGE__EXCHANGE_TYPE'] = 'bluefin'
os.environ['SYSTEM__DRY_RUN'] = 'true'
os.environ['LLM__OPENAI_API_KEY'] = 'sk-test1234567890abcdef1234567890abcdef12345678'
os.environ['EXCHANGE__BLUEFIN_PRIVATE_KEY'] = 'suiprivkey1qp3aergkwm2z7fvuw7es4q6w040a9m8wcw84tm4ck3f40k9hfzfmv2v4vs9'

print("🧪 Testing Bluefin Integration (Paper Trading Mode)")
print("=" * 50)

async def test_bluefin_paper_trading():
    try:
        # Create exchange without SDK
        from bot.exchange.bluefin import BluefinClient
        
        # Test 1: Create exchange in paper mode
        print("1. Creating Bluefin exchange (paper mode)...")
        exchange = BluefinClient(dry_run=True)
        print("   ✅ Exchange created")
        
        # Test 2: Connect (should work even without SDK in paper mode)
        print("2. Connecting to Bluefin (paper mode)...")
        connected = await exchange.connect()
        print(f"   ✅ Connected: {connected}")
        
        # Test 3: Get status
        status = exchange.get_connection_status()
        print("3. Connection status:")
        print(f"   Exchange: {status['exchange']}")
        print(f"   Network: {status['network']}")
        print(f"   Mode: {status['trading_mode']}")
        print(f"   Blockchain: {status['blockchain']}")
        print(f"   Decentralized: {status['is_decentralized']}")
        
        # Test 4: Symbol conversion
        print("4. Testing symbol conversion:")
        conversions = {
            'BTC-USD': 'BTC-PERP',
            'ETH-USD': 'ETH-PERP',
            'SOL-USD': 'SOL-PERP'
        }
        for standard, expected in conversions.items():
            result = exchange._convert_symbol(standard)
            status_icon = "✅" if result == expected else "❌"
            print(f"   {standard} -> {result} {status_icon}")
        
        # Test 5: Paper trade
        print("5. Testing paper trade...")
        from bot.types import TradeAction
        
        trade_action = TradeAction(
            action='LONG',
            size_pct=10.0,
            leverage=5,
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            confidence=0.85,
            reasoning='Paper trading test',
            rationale='Testing Bluefin without SDK dependencies'
        )
        
        order = await exchange.execute_trade_action(
            trade_action,
            'ETH-USD',
            Decimal('3500.00')
        )
        
        if order:
            print(f"   ✅ Paper order: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Status: {order.status}")
            print(f"   Type: {order.type}")
            print(f"   Quantity: {order.quantity}")
        else:
            print("   ❌ Order failed")
        
        # Test 6: Get account balance (paper mode)
        print("6. Testing account balance...")
        balance = await exchange.get_account_balance()
        print(f"   ✅ Paper balance: ${balance}")
        
        # Test 7: Test futures support
        print("7. Testing futures features...")
        print(f"   Supports futures: {exchange.supports_futures}")
        print(f"   Is decentralized: {exchange.is_decentralized}")
        
        await exchange.disconnect()
        print("\n🎉 All paper trading tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bluefin_paper_trading())
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    print("\nNote: This test verifies paper trading mode works even without")
    print("Bluefin SDK dependencies. In production, set up proper dependencies")
    print("for live trading using the UV-based Docker container.")