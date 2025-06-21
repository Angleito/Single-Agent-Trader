#!/usr/bin/env python3
"""
Example demonstrating the InventoryManager for market making strategies.

This example shows how the InventoryManager integrates with VuManChu signals
to make intelligent rebalancing decisions for market making operations.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot.strategy.inventory_manager import InventoryManager, VuManChuBias
from bot.trading_types import Order, OrderStatus, Position


async def main():
    """Run inventory management example."""
    print("🎯 Inventory Management System Example")
    print("=" * 50)

    # Initialize inventory manager for BTC-USD
    symbol = "BTC-USD"
    inventory_manager = InventoryManager(
        symbol=symbol,
        max_position_pct=10.0,  # 10% of account equity max
        rebalancing_threshold=5.0,  # Rebalance at 5% imbalance
        emergency_threshold=15.0,  # Emergency flatten at 15%
        inventory_timeout_hours=4.0,  # Max 4 hours holding
        data_dir=Path("./data/inventory_example"),
    )

    # Set account equity
    account_equity = Decimal(100000)  # $100k account
    inventory_manager.update_account_equity(account_equity)
    print(f"💰 Account equity: ${account_equity:,}")

    # Simulate market making fills
    print("\n📈 Simulating market making order fills...")

    # Create some sample order fills
    fills = [
        Order(
            id="fill_1",
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            quantity=Decimal("0.5"),
            price=Decimal(50000),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.5"),
        ),
        Order(
            id="fill_2",
            symbol=symbol,
            side="SELL",
            type="LIMIT",
            quantity=Decimal("0.3"),
            price=Decimal(50100),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.3"),
        ),
        Order(
            id="fill_3",
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            quantity=Decimal("0.8"),
            price=Decimal(49950),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(UTC),
            filled_quantity=Decimal("0.8"),
        ),
    ]

    # Current position after fills
    current_position = Position(
        symbol=symbol,
        side="LONG",
        size=Decimal("1.0"),  # Net: 0.5 - 0.3 + 0.8 = 1.0
        entry_price=Decimal(49975),  # Weighted average
        timestamp=datetime.now(UTC),
    )

    # Track position changes
    print("📊 Tracking position changes...")
    metrics = inventory_manager.track_position_changes(fills, current_position)

    print(f"  • Net position: {metrics.net_position} BTC")
    print(f"  • Position value: ${metrics.position_value:,.2f}")
    print(f"  • Imbalance: {metrics.imbalance_percentage:.2f}%")
    print(f"  • Risk score: {metrics.risk_score:.1f}/100")
    print(f"  • Duration: {metrics.inventory_duration_hours:.1f} hours")

    # Test different VuManChu bias scenarios
    print("\n🧠 Testing VuManChu bias integration...")

    market_price = Decimal(50000)

    # Scenario 1: Bullish bias supports long position
    print("\n📊 Scenario 1: Bullish VuManChu bias")
    bullish_bias = VuManChuBias(
        overall_bias="BULLISH",
        cipher_a_signal="GREEN_DIAMOND",
        cipher_b_signal="BUY_CIRCLE",
        wave_trend_direction="UP",
        signal_strength=0.8,
        confidence=0.9,
    )

    imbalance = inventory_manager.calculate_inventory_imbalance()
    action = inventory_manager.suggest_rebalancing_action(
        imbalance, bullish_bias, market_price
    )

    print(f"  • Current imbalance: {imbalance:.2f}%")
    print(f"  • VuManChu bias: {bullish_bias.overall_bias}")
    print(f"  • Recommended action: {action.action_type}")
    print(f"  • Quantity: {action.quantity}")
    print(f"  • Urgency: {action.urgency}")
    print(f"  • Reason: {action.reason}")

    # Scenario 2: Bearish bias conflicts with long position
    print("\n📊 Scenario 2: Bearish VuManChu bias (conflicting)")
    bearish_bias = VuManChuBias(
        overall_bias="BEARISH",
        cipher_a_signal="RED_DIAMOND",
        cipher_b_signal="SELL_CIRCLE",
        wave_trend_direction="DOWN",
        signal_strength=0.7,
        confidence=0.8,
    )

    action = inventory_manager.suggest_rebalancing_action(
        imbalance, bearish_bias, market_price
    )

    print(f"  • VuManChu bias: {bearish_bias.overall_bias}")
    print(f"  • Recommended action: {action.action_type}")
    print(f"  • Quantity: {action.quantity}")
    print(f"  • Urgency: {action.urgency}")
    print(f"  • Reason: {action.reason}")

    # Execute rebalancing trade
    print("\n⚖️ Executing rebalancing trade...")
    success = inventory_manager.execute_rebalancing_trade(action, market_price)
    print(f"  • Execution success: {success}")

    # Show position summary
    print("\n📈 Position Summary:")
    summary = inventory_manager.get_position_summary()

    print(f"  • Symbol: {summary['symbol']}")
    print(f"  • Current position: {summary['current_position']} BTC")
    print(f"  • Position value: ${Decimal(summary['position_value']):,.2f}")
    print(f"  • Imbalance: {summary['imbalance_percentage']:.2f}%")
    print(f"  • Risk score: {summary['risk_score']:.1f}/100")
    print(f"  • Account equity: ${Decimal(summary['account_equity']):,.2f}")

    # Rebalancing statistics
    rebal_stats = summary["rebalancing_stats"]
    print("\n📊 Rebalancing Statistics:")
    print(f"  • Total successful: {rebal_stats['total_success']}")
    print(f"  • Total failed: {rebal_stats['total_failure']}")
    print(f"  • Recent 24h: {rebal_stats['recent_24h_count']}")
    print(f"  • Emergency flattens: {rebal_stats['emergency_flatten_count']}")

    # Test emergency scenario
    print("\n🚨 Testing emergency scenario...")

    # Simulate large position that exceeds emergency threshold
    large_position = Position(
        symbol=symbol,
        side="LONG",
        size=Decimal("5.0"),  # Large position
        entry_price=Decimal(50000),
        timestamp=datetime.now(UTC),
    )

    # Update with large position
    metrics = inventory_manager.track_position_changes([], large_position)
    large_imbalance = inventory_manager.calculate_inventory_imbalance()

    print(f"  • Large position: {large_position.size} BTC")
    print(f"  • Large imbalance: {large_imbalance:.2f}%")

    # Test emergency action
    neutral_bias = VuManChuBias(
        overall_bias="NEUTRAL",
        signal_strength=0.5,
        confidence=0.5,
    )

    emergency_action = inventory_manager.suggest_rebalancing_action(
        large_imbalance, neutral_bias, market_price
    )

    print(f"  • Emergency action: {emergency_action.action_type}")
    print(f"  • Emergency urgency: {emergency_action.urgency}")
    print(f"  • Emergency reason: {emergency_action.reason}")

    print("\n✅ Inventory management example completed!")
    print("\nKey Features Demonstrated:")
    print("  • Real-time inventory tracking and position monitoring")
    print("  • VuManChu signal integration for directional bias")
    print("  • Risk-based rebalancing recommendations")
    print("  • Emergency flatten mechanisms for extreme imbalances")
    print("  • Comprehensive inventory analytics and reporting")
    print("  • State persistence for recovery and continuity")


if __name__ == "__main__":
    asyncio.run(main())
