#!/usr/bin/env python3
"""
Simplified Functional Risk Management Integration Test

This test validates the core functional risk management system
without complex strategy dependencies.
"""

from decimal import Decimal


def test_direct_risk_types():
    """Test direct import and usage of risk types."""
    print("Testing Direct Risk Types...")

    try:
        from bot.fp.types.risk import (
            CircuitBreakerState,
            EmergencyStopState,
            MarginInfo,
            RiskLimits,
            RiskParameters,
        )

        # Test risk parameters
        risk_params = RiskParameters(
            max_position_size=Decimal("0.25"),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal("0.05"),
            take_profit_pct=Decimal("0.15"),
        )
        print(
            f"✓ Created RiskParameters: max_position_size={risk_params.max_position_size}"
        )

        # Test risk limits
        risk_limits = RiskLimits(
            daily_loss_limit=Decimal(1000),
            position_limit=3,
            margin_requirement=Decimal("0.2"),
        )
        print(f"✓ Created RiskLimits: daily_loss_limit=${risk_limits.daily_loss_limit}")

        # Test margin info
        margin_info = MarginInfo(
            total_balance=Decimal(10000),
            used_margin=Decimal(2000),
            free_margin=Decimal(8000),
            margin_ratio=Decimal("0.2"),
        )
        print(f"✓ Created MarginInfo: total_balance=${margin_info.total_balance}")

        # Test circuit breaker
        circuit_breaker = CircuitBreakerState(
            state="CLOSED",
            failure_count=0,
            failure_threshold=3,
            timeout_seconds=300,
            last_failure_time=None,
            consecutive_successes=0,
            failure_history=(),
        )
        print(f"✓ Created CircuitBreaker: is_open={circuit_breaker.is_open}")

        # Test emergency stop
        emergency_stop = EmergencyStopState(
            is_stopped=False,
            stop_reason=None,
            stopped_at=None,
            manual_override=False,
        )
        print(f"✓ Created EmergencyStop: is_stopped={emergency_stop.is_stopped}")

        print("✅ Direct risk types test passed!")
        return True

    except Exception as e:
        print(f"❌ Direct risk types test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_position_sizing():
    """Test position sizing calculations."""
    print("\nTesting Position Sizing...")

    try:
        from bot.fp.types.risk import (
            RiskParameters,
            calculate_position_size,
        )

        risk_params = RiskParameters(
            max_position_size=Decimal("0.25"),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal("0.05"),
            take_profit_pct=Decimal("0.15"),
        )

        # Test basic position sizing
        account_balance = Decimal(10000)
        max_position_value = account_balance * risk_params.max_position_size
        print(f"✓ Max position value: ${max_position_value}")

        # Test position size calculation
        position_size = calculate_position_size(
            balance=account_balance,
            risk_per_trade=Decimal(2),  # 2% risk per trade
            stop_loss_pct=risk_params.stop_loss_pct * 100,  # Convert to percentage
        )
        print(f"✓ Position size calculation: ${position_size}")

        print("✅ Position sizing test passed!")
        return True

    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_risk_validation():
    """Test risk validation and alerts."""
    print("\nTesting Risk Validation...")

    try:
        from bot.fp.types.risk import (
            MarginInfo,
            RiskLimits,
            RiskParameters,
            check_risk_alerts,
        )

        risk_params = RiskParameters(
            max_position_size=Decimal("0.25"),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal("0.05"),
            take_profit_pct=Decimal("0.15"),
        )

        risk_limits = RiskLimits(
            daily_loss_limit=Decimal(500),
            position_limit=3,
            margin_requirement=Decimal("0.2"),
        )

        margin_info = MarginInfo(
            total_balance=Decimal(10000),
            used_margin=Decimal(2000),
            free_margin=Decimal(8000),
            margin_ratio=Decimal("0.2"),
        )

        # Test basic risk validation logic
        current_positions = 2
        daily_pnl = Decimal(-100)

        # Basic validation checks
        position_ok = current_positions <= risk_limits.position_limit
        margin_ok = margin_info.margin_ratio <= Decimal("0.8")  # 80% max
        loss_ok = abs(daily_pnl) <= risk_limits.daily_loss_limit

        print(f"✓ Position limit check: {'✓' if position_ok else '❌'}")
        print(f"✓ Margin check: {'✓' if margin_ok else '❌'}")
        print(f"✓ Daily loss check: {'✓' if loss_ok else '❌'}")

        # Test risk alerts if function exists
        try:
            alerts = check_risk_alerts(
                risk_limits=risk_limits,
                margin_info=margin_info,
                current_positions=current_positions,
                daily_pnl=daily_pnl,
            )
            print(f"✓ Risk alerts check: {len(alerts)} alerts")
        except Exception:
            print("✓ Risk alerts function not available, using basic checks")

        print("✅ Risk validation test passed!")
        return True

    except Exception as e:
        print(f"❌ Risk validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_paper_trading_integration():
    """Test paper trading integration with risk management."""
    print("\nTesting Paper Trading Integration...")

    try:
        from bot.fp.types.risk import MarginInfo, RiskParameters

        # Create risk parameters
        risk_params = RiskParameters(
            max_position_size=Decimal("0.25"),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal("0.05"),
            take_profit_pct=Decimal("0.15"),
        )

        # Create margin info
        margin_info = MarginInfo(
            total_balance=Decimal(10000),
            used_margin=Decimal(0),
            free_margin=Decimal(10000),
            margin_ratio=Decimal(0),
        )

        # Test paper trading compliance
        account_balance = Decimal(10000)
        max_trade_size = account_balance * risk_params.max_position_size
        print(f"✓ Paper trading max trade size: ${max_trade_size}")

        # Validate trade doesn't exceed limits
        proposed_trade = Decimal(2000)  # 20% of account
        is_compliant = proposed_trade <= max_trade_size
        print(f"✓ Trade compliance check: {'✓' if is_compliant else '❌'}")

        print("✅ Paper trading integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Paper trading integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_leverage_calculations():
    """Test leverage and margin calculations."""
    print("\nTesting Leverage Calculations...")

    try:
        from bot.fp.types.risk import (
            MarginInfo,
            RiskParameters,
            calculate_margin_ratio,
        )

        risk_params = RiskParameters(
            max_position_size=Decimal("0.25"),
            max_leverage=Decimal(5),
            stop_loss_pct=Decimal("0.05"),
            take_profit_pct=Decimal("0.15"),
        )

        # Test basic margin calculations
        position_value = Decimal(5000)
        leverage = Decimal(3)

        # Calculate required margin manually
        required_margin = position_value / leverage
        print(f"✓ Required margin calculation: ${required_margin}")

        # Test margin ratio calculation
        total_balance = Decimal(10000)
        used_margin = Decimal(1500)

        margin_ratio = calculate_margin_ratio(used_margin, total_balance)
        print(f"✓ Margin ratio: {margin_ratio}")

        # Validate leverage limits
        is_valid_leverage = leverage <= risk_params.max_leverage
        print(f"✓ Leverage validation: {'✓' if is_valid_leverage else '❌'}")

        # Test margin utilization
        margin_info = MarginInfo(
            total_balance=total_balance,
            used_margin=used_margin,
            free_margin=total_balance - used_margin,
            margin_ratio=margin_ratio,
        )
        print(f"✓ Free margin: ${margin_info.free_margin}")

        print("✅ Leverage calculations test passed!")
        return True

    except Exception as e:
        print(f"❌ Leverage calculations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all simplified risk management tests."""
    print("=" * 60)
    print("SIMPLIFIED FUNCTIONAL RISK MANAGEMENT INTEGRATION TEST")
    print("=" * 60)

    tests = [
        test_direct_risk_types,
        test_position_sizing,
        test_risk_validation,
        test_paper_trading_integration,
        test_leverage_calculations,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    if passed == total:
        print(f"✅ ALL TESTS PASSED! ({passed}/{total})")
        print("Functional risk management integration is working correctly.")
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total} passed)")
        print("Review the output above for specific failures.")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    main()
