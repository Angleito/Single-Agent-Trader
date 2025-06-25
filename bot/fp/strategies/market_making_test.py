"""Test file to verify enhanced market making functionality.

This module contains basic tests to ensure all enhanced market making
features work correctly and that existing APIs are preserved.
"""

import sys
from datetime import datetime
from decimal import Decimal

from bot.fp.core import MarketState
from bot.fp.types.market import Candle

from .advanced_market_making import (
    RiskLimits,
    create_market_making_strategy,
    validate_market_making_config,
)

# Import both original and enhanced market making functions
from .market_making import (
    InventoryPolicy,
    InventorySignal,
    InventoryState,
    MarketConditions,
    # Enhanced functions
    SpreadModel,
    SpreadResult,
    analyze_inventory_state,
    calculate_adaptive_spread,
    calculate_garch_spread,
    calculate_inventory_skew,
    calculate_order_book_imbalance,
    # Original preserved functions
    calculate_spread,
    check_inventory_limits,
    generate_quotes,
    market_maker_strategy,
    optimize_inventory_exposure,
)


def test_legacy_api_preserved():
    """Test that all legacy market making APIs are preserved."""

    # Test original spread calculation
    spread = calculate_spread(
        volatility=0.02,
        base_spread=0.001,
        volatility_multiplier=2.0,
        min_spread=0.0005,
        max_spread=0.01,
    )
    assert 0.0005 <= spread <= 0.01
    print(f"✓ Legacy calculate_spread: {spread}")

    # Test inventory skew calculation
    skew = calculate_inventory_skew(
        current_inventory=1000.0, max_inventory=5000.0, skew_factor=0.5
    )
    assert -1.0 <= skew <= 1.0
    print(f"✓ Legacy calculate_inventory_skew: {skew}")

    # Test order book imbalance
    imbalance = calculate_order_book_imbalance(
        bid_volume=1000.0, ask_volume=800.0, imbalance_threshold=0.1
    )
    assert -1.0 <= imbalance <= 1.0
    print(f"✓ Legacy calculate_order_book_imbalance: {imbalance}")

    # Test quote generation
    bid_price, ask_price = generate_quotes(
        mid_price=100.0,
        spread=0.01,
        inventory_skew=0.1,
        order_book_imbalance=0.05,
        competitive_adjustment=0.95,
    )
    assert bid_price < ask_price
    print(f"✓ Legacy generate_quotes: bid={bid_price}, ask={ask_price}")

    # Test inventory limits
    within_limits = check_inventory_limits(
        current_inventory=1000.0, max_inventory=5000.0, proposed_size=500.0, side="buy"
    )
    assert isinstance(within_limits, bool)
    print(f"✓ Legacy check_inventory_limits: {within_limits}")


def test_enhanced_spread_models():
    """Test enhanced spread calculation models."""

    # Test SpreadModel creation
    model = SpreadModel(
        base_spread=0.002,
        min_spread=0.0005,
        max_spread=0.02,
        volatility_factor=2.0,
        liquidity_factor=1.0,
        inventory_factor=0.5,
        skew_factor=0.3,
    )
    print(f"✓ SpreadModel created: base={model.base_spread}")

    # Test MarketConditions
    conditions = MarketConditions(
        volatility=0.015,
        bid_depth=1000.0,
        ask_depth=1200.0,
        spread_ratio=1.1,
        volume_ratio=1.5,
        price_momentum=0.02,
    )
    print(f"✓ MarketConditions created: vol={conditions.volatility}")

    # Test adaptive spread calculation
    spread_result = calculate_adaptive_spread(
        market_conditions=conditions, model=model, inventory_ratio=0.2
    )
    assert isinstance(spread_result, SpreadResult)
    assert spread_result.confidence >= 0.0
    print(
        f"✓ Adaptive spread: {spread_result.adjusted_spread} ({spread_result.model_used})"
    )

    # Test GARCH spread calculation
    volatility_series = [0.01, 0.015, 0.012, 0.018, 0.016] * 15  # 75 data points
    garch_result = calculate_garch_spread(
        volatility_series=volatility_series, model=model, lookback=50
    )
    assert isinstance(garch_result, SpreadResult)
    print(f"✓ GARCH spread: {garch_result.adjusted_spread} ({garch_result.model_used})")


def test_enhanced_inventory_management():
    """Test enhanced inventory management features."""

    # Test InventoryPolicy
    policy = InventoryPolicy(
        max_position_ratio=0.1,
        target_turn_rate=8.0,
        skew_factor=0.5,
        urgency_threshold=0.8,
        timeout_hours=4.0,
    )
    print(f"✓ InventoryPolicy created: max_ratio={policy.max_position_ratio}")

    # Test InventoryState
    state = InventoryState(
        current_position=2000.0,
        max_position=10000.0,
        target_position=0.0,
        last_fill_time=datetime.now(),
        position_value=200000.0,
        unrealized_pnl=1000.0,
        inventory_duration=2.5,
        turn_rate=6.0,
    )
    print(f"✓ InventoryState created: position={state.current_position}")

    # Test inventory analysis
    inventory_signal = analyze_inventory_state(
        state=state, policy=policy, current_time=datetime.now()
    )
    assert isinstance(inventory_signal, InventorySignal)
    assert 0.0 <= inventory_signal.urgency <= 1.0
    print(f"✓ Inventory analysis: urgency={inventory_signal.urgency:.2f}")

    # Test inventory exposure optimization
    conditions = MarketConditions(
        volatility=0.02,
        bid_depth=1000.0,
        ask_depth=1000.0,
        spread_ratio=1.0,
        volume_ratio=1.0,
        price_momentum=0.0,
    )

    bid_size, ask_size = optimize_inventory_exposure(
        state=state, market_conditions=conditions, target_notional=1000.0
    )
    assert bid_size > 0
    assert ask_size > 0
    print(f"✓ Inventory optimization: bid_size={bid_size:.0f}, ask_size={ask_size:.0f}")


def test_enhanced_strategies():
    """Test enhanced market making strategies."""

    # Create test candles
    candles = []
    base_price = 100.0
    for i in range(50):
        price = base_price + (i * 0.1)
        candle = Candle(
            timestamp=datetime.now(),
            open=Decimal(str(price)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price + 0.2)),
            volume=Decimal(1000),
        )
        candles.append(candle)

    # Create market state
    market_state = MarketState(
        candles=candles,
        metadata={"inventory": 1000.0, "bid_volume": 1000.0, "ask_volume": 1200.0},
    )

    # Test legacy strategy still works
    legacy_strategy = market_maker_strategy(
        spread_factor=0.002, inventory_limit=10000.0, skew_factor=0.3
    )
    legacy_signal = legacy_strategy(market_state)
    print(
        f"✓ Legacy strategy signal: {legacy_signal.type if legacy_signal else 'None'}"
    )

    # Test enhanced strategy
    enhanced_strategy = create_market_making_strategy(strategy_type="enhanced")
    enhanced_signal = enhanced_strategy(market_state)
    print(
        f"✓ Enhanced strategy signal: {enhanced_signal.type if enhanced_signal else 'None'}"
    )


def test_configuration_validation():
    """Test configuration validation."""

    # Valid configuration
    spread_model = SpreadModel(base_spread=0.002, min_spread=0.001, max_spread=0.01)

    inventory_policy = InventoryPolicy(max_position_ratio=0.1, target_turn_rate=8.0)

    risk_limits = RiskLimits(
        max_position=10000.0,
        max_notional=100000.0,
        max_concentration=0.2,
        max_drawdown=0.1,
        var_limit=5000.0,
        daily_loss_limit=2000.0,
    )

    errors = validate_market_making_config(
        spread_model=spread_model,
        inventory_policy=inventory_policy,
        risk_limits=risk_limits,
    )
    assert len(errors) == 0
    print(f"✓ Valid configuration: {len(errors)} errors")

    # Invalid configuration
    invalid_model = SpreadModel(
        base_spread=0.002,
        min_spread=0.01,  # Invalid: min > base
        max_spread=0.005,  # Invalid: max < base
    )

    invalid_errors = validate_market_making_config(
        spread_model=invalid_model,
        inventory_policy=inventory_policy,
        risk_limits=risk_limits,
    )
    assert len(invalid_errors) > 0
    print(f"✓ Invalid configuration detected: {len(invalid_errors)} errors")


def run_all_tests():
    """Run all market making enhancement tests."""
    print("🧪 Running Enhanced Market Making Tests\n")

    try:
        test_legacy_api_preserved()
        print()

        test_enhanced_spread_models()
        print()

        test_enhanced_inventory_management()
        print()

        test_enhanced_strategies()
        print()

        test_configuration_validation()
        print()

        print("✅ All Enhanced Market Making Tests Passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
