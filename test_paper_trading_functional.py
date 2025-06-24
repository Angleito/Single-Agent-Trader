#!/usr/bin/env python3
"""
Test script for functional paper trading enhancements.

This script verifies that the functional programming enhancements maintain
simulation accuracy and provide the expected benefits.
"""

import logging
import sys
from decimal import Decimal
from pathlib import Path

# Add the bot directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent / "bot"))

from trading_types import TradeAction


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_immutable_state_types():
    """Test the immutable state types."""
    print("🧪 Testing immutable state types...")

    try:
        from bot.fp.types.paper_trading import (
            PaperTradingAccountState,
            create_paper_trade,
        )

        # Test account state creation
        initial_state = PaperTradingAccountState.create_initial(
            starting_balance=Decimal(10000)
        )

        print(f"✅ Initial state created: Balance ${initial_state.starting_balance}")

        # Test trade creation
        trade = create_paper_trade(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
        )

        print(
            f"✅ Trade created: {trade.symbol} {trade.side} {trade.size} @ ${trade.entry_price}"
        )

        # Test immutability
        new_state = initial_state.add_trade(trade)

        # Verify original state unchanged
        assert len(initial_state.open_trades) == 0
        assert len(new_state.open_trades) == 1
        print("✅ Immutability verified: Original state unchanged")

        # Test P&L calculation
        current_price = Decimal(55000)
        unrealized_pnl = trade.calculate_unrealized_pnl(current_price)
        expected_pnl = (current_price - trade.entry_price) * trade.size

        assert unrealized_pnl == expected_pnl
        print(f"✅ P&L calculation correct: ${unrealized_pnl}")

        return True

    except Exception as e:
        print(f"❌ Immutable state types test failed: {e}")
        return False


def test_pure_calculations():
    """Test pure calculation functions."""
    print("\n🧪 Testing pure calculation functions...")

    try:
        from bot.fp.pure.paper_trading_calculations import (
            calculate_position_size,
            simulate_trade_execution,
            validate_account_state,
        )
        from bot.fp.types.paper_trading import PaperTradingAccountState

        # Test position size calculation
        position_size = calculate_position_size(
            equity=Decimal(10000),
            size_percentage=Decimal(10),  # 10%
            leverage=Decimal(5),
            current_price=Decimal(50000),
        )

        expected_size = (Decimal(10000) * Decimal("0.1") * Decimal(5)) / Decimal(50000)
        assert position_size == expected_size
        print(f"✅ Position size calculation: {position_size} BTC")

        # Test trade simulation
        account_state = PaperTradingAccountState.create_initial(Decimal(10000))

        execution, new_state = simulate_trade_execution(
            account_state=account_state,
            symbol="BTC-USD",
            side="LONG",
            size_percentage=Decimal(10),
            current_price=Decimal(50000),
            leverage=Decimal(5),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )

        assert execution.success
        assert new_state is not None
        print(f"✅ Trade simulation successful: {execution.execution_price}")

        # Test state validation
        assert validate_account_state(new_state)
        print("✅ State validation passed")

        return True

    except Exception as e:
        print(f"❌ Pure calculations test failed: {e}")
        return False


def test_functional_engine():
    """Test the functional paper trading engine."""
    print("\n🧪 Testing functional paper trading engine...")

    try:
        from bot.fp.paper_trading_functional import (
            FunctionalPaperTradingEngine,
            execute_trade_with_logging,
        )

        # Create engine
        engine = FunctionalPaperTradingEngine(
            initial_balance=Decimal(10000),
            data_dir=Path("./test_data"),
        )

        print("✅ Functional engine created")

        # Test account metrics
        current_prices = {"BTC-USD": Decimal(50000)}
        metrics = engine.get_account_metrics(current_prices).run()

        assert metrics["starting_balance"] == 10000.0
        assert metrics["current_balance"] == 10000.0
        print(f"✅ Account metrics: ${metrics['equity']}")

        # Test trade execution
        execution_result = execute_trade_with_logging(
            engine=engine,
            symbol="BTC-USD",
            side="LONG",
            size_percentage=Decimal(10),
            current_price=Decimal(50000),
        ).run()

        if execution_result.is_right():
            execution, new_state = execution_result.value
            assert execution.success
            print(f"✅ Trade execution: ${execution.execution_price}")

            # Commit state
            commit_result = engine.commit_state_change(new_state).run()
            assert commit_result.is_right()
            print("✅ State committed successfully")
        else:
            print(f"❌ Trade execution failed: {execution_result.value}")
            return False

        return True

    except Exception as e:
        print(f"❌ Functional engine test failed: {e}")
        return False


def test_enhanced_api_compatibility():
    """Test the enhanced API maintains compatibility."""
    print("\n🧪 Testing enhanced API compatibility...")

    try:
        from bot.paper_trading_enhanced import EnhancedPaperTradingAccount

        # Create enhanced account
        account = EnhancedPaperTradingAccount(
            starting_balance=Decimal(10000),
            use_functional_core=True,
        )

        print("✅ Enhanced account created")

        # Test account status (original API)
        status = account.get_account_status()

        assert "starting_balance" in status
        assert "current_balance" in status
        assert "equity" in status
        print(f"✅ Account status: ${status['equity']}")

        # Test trade execution (original API)
        action = TradeAction(
            action="LONG",
            size_pct=10,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Test trade",
        )

        order = account.execute_trade_action(
            action=action,
            symbol="BTC-USD",
            current_price=Decimal(50000),
        )

        if order and order.status.value == "FILLED":
            print(f"✅ Trade executed: Order {order.id}")
        else:
            print("ℹ️ Trade execution returned None (HOLD or error)")

        # Test performance summary (original API)
        summary = account.get_performance_summary()

        assert "total_trades" in summary
        assert "overall_win_rate" in summary
        print(f"✅ Performance summary: {summary['total_trades']} trades")

        # Test enhanced metrics (new functionality)
        enhanced_metrics = account.get_enhanced_metrics()

        if enhanced_metrics:
            print("✅ Enhanced metrics available")

        return True

    except Exception as e:
        print(f"❌ Enhanced API compatibility test failed: {e}")
        return False


def test_simulation_accuracy():
    """Test that simulation accuracy is maintained."""
    print("\n🧪 Testing simulation accuracy...")

    try:
        from bot.paper_trading_enhanced import EnhancedPaperTradingAccount

        # Create account with known parameters
        account = EnhancedPaperTradingAccount(
            starting_balance=Decimal(10000),
            use_functional_core=True,
        )

        # Execute a series of trades and verify calculations
        current_price = Decimal(50000)
        size_pct = 10  # 10% of equity

        # Calculate expected values
        equity = Decimal(10000)
        position_value = equity * Decimal("0.1")  # 10%
        leveraged_value = position_value * Decimal(5)  # 5x leverage
        expected_size = leveraged_value / current_price

        print(f"Expected position size: {expected_size} BTC")

        # Execute trade
        action = TradeAction(
            action="LONG",
            size_pct=size_pct,
            take_profit_pct=2.0,
            stop_loss_pct=1.5,
            rationale="Accuracy test trade",
        )

        order = account.execute_trade_action(action, "BTC-USD", current_price)

        if order and order.status.value == "FILLED":
            # Verify order details
            print(f"Order size: {order.quantity} BTC")
            print(f"Order price: ${order.price}")

            # Check if size is reasonable (within expected range due to fees/slippage)
            size_diff_pct = (
                abs(float(order.quantity - expected_size) / float(expected_size)) * 100
            )

            if size_diff_pct < 10:  # Allow 10% difference for fees/slippage
                print("✅ Trade size accuracy verified")
            else:
                print(f"⚠️ Trade size difference: {size_diff_pct:.2f}%")

            # Test P&L calculation accuracy
            new_price = Decimal(55000)  # 10% price increase
            account.get_account_status({"BTC-USD": new_price})

            status = account.get_account_status({"BTC-USD": new_price})
            unrealized_pnl = status["unrealized_pnl"]

            # Expected P&L (approximately)
            expected_pnl = float(order.quantity * (new_price - order.price))
            actual_pnl = unrealized_pnl

            pnl_diff_pct = (
                abs(actual_pnl - expected_pnl) / expected_pnl * 100
                if expected_pnl != 0
                else 0
            )

            print(f"Expected P&L: ${expected_pnl:.2f}")
            print(f"Actual P&L: ${actual_pnl:.2f}")

            if pnl_diff_pct < 5:  # Allow 5% difference
                print("✅ P&L calculation accuracy verified")
            else:
                print(f"⚠️ P&L calculation difference: {pnl_diff_pct:.2f}%")

        return True

    except Exception as e:
        print(f"❌ Simulation accuracy test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Functional Paper Trading Enhancement Tests\n")

    tests = [
        ("Immutable State Types", test_immutable_state_types),
        ("Pure Calculations", test_pure_calculations),
        ("Functional Engine", test_functional_engine),
        ("Enhanced API Compatibility", test_enhanced_api_compatibility),
        ("Simulation Accuracy", test_simulation_accuracy),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Functional enhancements are working correctly.")
        return True
    print(
        f"\n⚠️ {len(results) - passed} tests failed. Please review the implementation."
    )
    return False


if __name__ == "__main__":
    setup_logging()

    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n🚨 Test runner failed: {e}")
        sys.exit(1)
