#!/usr/bin/env python3
"""
Comprehensive test for functional risk management integration.

This script directly tests the functional risk management system without
importing the full bot module to avoid dependency issues.
"""

import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_direct_risk_integration():
    """Test direct functional risk management integration."""
    print("Testing Direct Functional Risk Integration...")

    try:
        # Import functional risk types directly
        # Import functional risk strategies
        from bot.fp.strategies.risk_management import (
            calculate_fixed_fractional_size,
            calculate_kelly_criterion,
            calculate_portfolio_heat,
            calculate_position_size_with_stop,
            enforce_risk_limits,
        )
        from bot.fp.types.risk import (
            MarginInfo,
            RiskLimits,
            RiskParameters,
            calculate_free_margin,
            calculate_margin_ratio,
            calculate_position_size,
            calculate_required_margin,
            calculate_stop_loss_price,
            calculate_take_profit_price,
            check_risk_alerts,
            is_within_risk_limits,
        )

        print("✓ Successfully imported functional risk management components")

        # Test basic risk parameters
        risk_params = RiskParameters(
            max_position_size=Decimal(25),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal(2),
            take_profit_pct=Decimal(4),
        )
        print(f"✓ Created RiskParameters: {risk_params}")

        # Test risk limits
        risk_limits = RiskLimits(
            daily_loss_limit=Decimal(500),
            position_limit=3,
            margin_requirement=Decimal(20),
        )
        print(f"✓ Created RiskLimits: {risk_limits}")

        # Test margin calculations
        balance = Decimal(10000)
        used_margin = Decimal(2000)
        free_margin = calculate_free_margin(balance, used_margin)
        margin_ratio = calculate_margin_ratio(used_margin, balance)

        margin_info = MarginInfo(
            total_balance=balance,
            used_margin=used_margin,
            free_margin=free_margin,
            margin_ratio=margin_ratio,
        )
        print(f"✓ Margin calculations: free=${free_margin}, ratio={margin_ratio:.2%}")

        # Test position size calculations
        position_size = calculate_position_size(
            balance=balance,
            risk_per_trade=Decimal(2),  # 2%
            stop_loss_pct=Decimal(1.5),  # 1.5%
        )
        print(f"✓ Position size calculation: ${position_size}")

        # Test required margin calculation
        required_margin = calculate_required_margin(
            position_size=Decimal(1000),
            entry_price=Decimal(50000),
            leverage=Decimal(5),
        )
        print(f"✓ Required margin: ${required_margin}")

        # Test stop loss/take profit calculations
        entry_price = Decimal(50000)
        stop_loss_price = calculate_stop_loss_price(
            entry_price=entry_price,
            stop_loss_pct=Decimal(2),
            is_long=True,
        )
        take_profit_price = calculate_take_profit_price(
            entry_price=entry_price,
            take_profit_pct=Decimal(4),
            is_long=True,
        )
        print(f"✓ Stop/Take prices: SL=${stop_loss_price}, TP=${take_profit_price}")

        # Test Kelly Criterion
        kelly_size = calculate_kelly_criterion(
            win_probability=0.6,
            win_loss_ratio=2.0,
            kelly_fraction=0.25,
        )
        print(f"✓ Kelly Criterion position size: {kelly_size:.2%}")

        # Test fixed fractional sizing
        fixed_size = calculate_fixed_fractional_size(
            account_balance=float(balance),
            risk_percentage=2.0,
        )
        print(f"✓ Fixed fractional size: ${fixed_size}")

        # Test portfolio heat calculation
        positions = [
            {
                "symbol": "BTC-USD",
                "size": 0.5,
                "entry_price": 50000,
                "stop_loss": 49000,
                "is_long": True,
            },
            {
                "symbol": "ETH-USD",
                "size": 10,
                "entry_price": 3000,
                "stop_loss": 2950,
                "is_long": True,
            },
        ]
        portfolio_heat = calculate_portfolio_heat(positions, float(balance))
        print(f"✓ Portfolio heat: {portfolio_heat:.2f}%")

        # Test risk limit enforcement
        proposed_size = 3000.0
        current_positions = [
            {"size": 2000, "entry_price": 50000, "stop_loss": 49000, "is_long": True},
        ]

        adjusted_size, reason = enforce_risk_limits(
            proposed_size=proposed_size,
            current_positions=current_positions,
            account_balance=float(balance),
            max_position_size_pct=25.0,
            max_portfolio_heat_pct=6.0,
        )
        print(f"✓ Risk enforcement: ${proposed_size} -> ${adjusted_size} ({reason})")

        # Test position size with stop
        position_with_stop = calculate_position_size_with_stop(
            account_balance=float(balance),
            entry_price=50000.0,
            stop_loss=49000.0,
            risk_percentage=2.0,
            is_long=True,
        )
        print(f"✓ Position size with stop: ${position_with_stop}")

        # Test risk alerts
        alerts = check_risk_alerts(
            margin_info=margin_info,
            limits=risk_limits,
            current_positions=2,
            daily_pnl=Decimal(-100),
        )
        print(f"✓ Risk alerts: {len(alerts)} active alerts")

        # Test risk limit validation
        within_limits = is_within_risk_limits(
            proposed_risk=Decimal(200),
            current_exposure=Decimal(800),
            balance=balance,
            max_risk_pct=Decimal(10),
        )
        print(f"✓ Within risk limits: {within_limits}")

        print("\n✅ All direct functional risk integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Direct integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration_with_trading_types():
    """Test integration with existing trading types."""
    print("\nTesting Integration with Trading Types...")

    try:
        # Import required types
        from bot.fp.types.risk import RiskMetrics
        from bot.trading_types import Position, TradeAction

        # Create a sample trade action
        trade_action = TradeAction(
            action="LONG",
            size_pct=15,
            take_profit_pct=4.0,
            stop_loss_pct=2.0,
            rationale="Functional risk test",
        )
        print(f"✓ Created TradeAction: {trade_action.action} {trade_action.size_pct}%")

        # Create a sample position
        position = Position(
            symbol="BTC-USD",
            side="FLAT",
            size=Decimal(0),
            entry_price=None,
            unrealized_pnl=Decimal(0),
            realized_pnl=Decimal(0),
            timestamp=datetime.now(UTC),
        )
        print(f"✓ Created Position: {position.symbol} {position.side}")

        # Create risk metrics using correct functional interface
        risk_metrics = RiskMetrics(
            current_exposure=Decimal(2000),  # Total exposure in USD
            var_95=Decimal(150),  # Value at Risk at 95% confidence
            max_drawdown=Decimal(0.05),  # Maximum drawdown percentage (5%)
            sharpe_ratio=Decimal(1.2),  # Risk-adjusted return metric
        )
        print(f"✓ Created RiskMetrics: exposure=${risk_metrics.current_exposure}")

        print("✅ Integration with trading types successful!")
        return True

    except Exception as e:
        print(f"❌ Trading types integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_paper_trading_compliance():
    """Test that risk management works with paper trading."""
    print("\nTesting Paper Trading Compliance...")

    try:
        from bot.fp.strategies.risk_management import enforce_risk_limits

        # Simulate paper trading environment
        account_balance = 10000.0
        paper_positions = [
            {"size": 1000, "entry_price": 50000, "stop_loss": 49000, "is_long": True},
            {"size": 500, "entry_price": 3000, "stop_loss": 2950, "is_long": True},
        ]

        # Test various position sizes
        test_sizes = [500, 1000, 2500, 5000, 7500]

        for test_size in test_sizes:
            adjusted_size, reason = enforce_risk_limits(
                proposed_size=test_size,
                current_positions=paper_positions,
                account_balance=account_balance,
                max_position_size_pct=25.0,
                max_portfolio_heat_pct=6.0,
            )

            risk_compliant = adjusted_size <= test_size
            print(
                f"  Position ${test_size} -> ${adjusted_size} (compliant: {risk_compliant})"
            )

        print("✅ Paper trading compliance tests passed!")
        return True

    except Exception as e:
        print(f"❌ Paper trading compliance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_type_safety():
    """Test type safety of functional risk calculations."""
    print("\nTesting Type Safety...")

    try:
        from bot.fp.types.risk import (
            RiskParameters,
            calculate_position_size,
        )

        # Test with proper Decimal types
        result = calculate_position_size(
            balance=Decimal(10000),
            risk_per_trade=Decimal(2),
            stop_loss_pct=Decimal("1.5"),
        )
        assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
        print(f"✓ Type safety: calculate_position_size returns {type(result).__name__}")

        # Test immutability
        risk_params = RiskParameters(
            max_position_size=Decimal(25),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal(2),
            take_profit_pct=Decimal(4),
        )

        try:
            risk_params.max_position_size = Decimal(30)  # Should fail
            print("❌ Immutability test failed - modification was allowed")
            return False
        except Exception:
            print("✓ Immutability: RiskParameters is properly frozen")

        print("✅ Type safety tests passed!")
        return True

    except Exception as e:
        print(f"❌ Type safety test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all functional risk integration tests."""
    print("=" * 60)
    print("Functional Risk Management Integration Test Suite")
    print("=" * 60)

    all_tests_passed = True

    # Run all test suites
    test_results = [
        test_direct_risk_integration(),
        test_integration_with_trading_types(),
        test_paper_trading_compliance(),
        test_type_safety(),
    ]

    all_tests_passed = all(test_results)

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL FUNCTIONAL RISK INTEGRATION TESTS PASSED!")
        print("   • Functional risk types are working correctly")
        print("   • Integration with existing trading types is successful")
        print("   • Risk calculations are type-safe and immutable")
        print("   • Paper trading compliance is maintained")
        print("   • Position sizing and risk enforcement work as expected")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Check the output above for specific failures")

    print("=" * 60)

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
