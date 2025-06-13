#!/usr/bin/env python3
"""
Test script for the enhanced paper trading system.

This script demonstrates the paper trading functionality and validates
that all components work correctly together.
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

# Add the bot module to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.paper_trading import PaperTradingAccount
from bot.position_manager import PositionManager
from bot.types import TradeAction


async def test_paper_trading():
    """Test the enhanced paper trading system."""
    print("🚀 Testing Enhanced Paper Trading System")
    print("=" * 50)

    # Initialize paper trading account
    paper_account = PaperTradingAccount(starting_balance=Decimal("10000"))
    position_manager = PositionManager(paper_trading_account=paper_account)

    # Test 1: Initial account status
    print("\n📊 Initial Account Status:")
    status = paper_account.get_account_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Test 2: Execute some sample trades
    print("\n📈 Executing Sample Trades:")

    # Long trade
    long_action = TradeAction(
        action="LONG", size_pct=10.0, rationale="Test long position"
    )

    btc_price = Decimal("50000")
    order1 = paper_account.execute_trade_action(long_action, "BTC-USD", btc_price)
    if order1:
        print(
            f"  ✅ Long trade executed: {order1.side} {order1.quantity} @ ${order1.price}"
        )

    # Short trade
    short_action = TradeAction(
        action="SHORT", size_pct=5.0, rationale="Test short position"
    )

    eth_price = Decimal("3000")
    order2 = paper_account.execute_trade_action(short_action, "ETH-USD", eth_price)
    if order2:
        print(
            f"  ✅ Short trade executed: {order2.side} {order2.quantity} @ ${order2.price}"
        )

    # Test 3: Check account status after trades
    print("\n📊 Account Status After Trades:")
    status = paper_account.get_account_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Test 4: Simulate price changes and unrealized P&L
    print("\n💹 Simulating Price Changes:")
    new_btc_price = Decimal("52000")  # BTC up 4%
    new_eth_price = Decimal("2900")  # ETH down 3.33%

    current_prices = {"BTC-USD": new_btc_price, "ETH-USD": new_eth_price}

    status = paper_account.get_account_status(current_prices)
    print(f"  BTC price: ${btc_price} → ${new_btc_price} (+4%)")
    print(f"  ETH price: ${eth_price} → ${new_eth_price} (-3.33%)")
    print(f"  Updated equity: ${status['equity']:,.2f}")
    print(f"  Unrealized P&L: ${status['unrealized_pnl']:,.2f}")

    # Test 5: Close positions
    print("\n🔒 Closing Positions:")

    close_action = TradeAction(
        action="CLOSE", size_pct=0, rationale="Test position closure"
    )

    # Close BTC position
    close_order1 = paper_account.execute_trade_action(
        close_action, "BTC-USD", new_btc_price
    )
    if close_order1:
        print(f"  ✅ BTC position closed @ ${close_order1.price}")

    # Close ETH position
    close_order2 = paper_account.execute_trade_action(
        close_action, "ETH-USD", new_eth_price
    )
    if close_order2:
        print(f"  ✅ ETH position closed @ ${close_order2.price}")

    # Test 6: Final account status
    print("\n📊 Final Account Status:")
    status = paper_account.get_account_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Test 7: Performance analytics
    print("\n📈 Performance Analytics:")

    # Update daily performance
    paper_account.update_daily_performance()

    # Get performance summary
    performance = position_manager.get_paper_trading_performance(days=1)
    if "error" not in performance:
        print(f"  Total Trades: {performance.get('total_trades', 0)}")
        print(f"  Win Rate: {performance.get('overall_win_rate', 0):.1f}%")
        print(f"  Net P&L: ${performance.get('net_pnl', 0):,.2f}")
        print(f"  ROI: {performance.get('roi_percent', 0):.2f}%")
        print(f"  Fees Paid: ${performance.get('total_fees_paid', 0):.2f}")

    # Test 8: Trade history
    print("\n📋 Trade History:")
    trade_history = paper_account.get_trade_history(days=1)
    for i, trade in enumerate(trade_history):
        print(
            f"  Trade {i+1}: {trade['side']} {trade['symbol']} - P&L: ${trade['realized_pnl']:,.2f}"
        )

    # Test 9: Daily report
    print("\n📄 Daily Report:")
    daily_report = position_manager.generate_daily_report()
    print(daily_report)

    # Test 10: Export functionality
    print("\n💾 Testing Export:")
    try:
        trade_export = position_manager.export_trade_history(days=1, format="json")
        print(f"  ✅ JSON export: {len(trade_export)} characters")

        csv_export = position_manager.export_trade_history(days=1, format="csv")
        print(f"  ✅ CSV export: {len(csv_export)} characters")
    except Exception as e:
        print(f"  ❌ Export error: {e}")

    print("\n✅ Paper Trading System Test Complete!")
    print(f"Final account equity: ${status['equity']:,.2f}")
    print(f"Total return: ${status['total_pnl']:,.2f} ({status['roi_percent']:.2f}%)")


if __name__ == "__main__":
    asyncio.run(test_paper_trading())
