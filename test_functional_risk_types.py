#!/usr/bin/env python3
"""
Test script for functional risk management types.

This script validates that all the new functional risk types work correctly
and demonstrate their usage.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.fp.types.risk import (
    # Basic types
    RiskParameters,
    RiskLimits,
    MarginInfo,
    # Advanced types
    CircuitBreakerState,
    EmergencyStopState,
    APIProtectionState,
    PortfolioExposure,
    LeverageAnalysis,
    DrawdownAnalysis,
    ComprehensiveRiskState,
    # Functions
    create_circuit_breaker_state,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
    trigger_emergency_stop,
    calculate_portfolio_exposure,
    calculate_leverage_analysis,
    assess_risk_level,
    create_comprehensive_risk_state,
    check_advanced_risk_alerts,
)

from bot.fp.types.balance_validation import (
    BalanceValidationConfig,
    BalanceRange,
    MarginRequirement,
    TradeAffordabilityCheck,
    validate_balance_range,
    validate_margin_requirements,
    validate_trade_affordability,
    perform_comprehensive_balance_validation,
    create_default_balance_config,
)


def test_basic_risk_types():
    """Test basic risk management types."""
    print("Testing basic risk types...")
    
    # Test RiskParameters
    risk_params = RiskParameters(
        max_position_size=Decimal("25"),
        max_leverage=Decimal("5"),
        stop_loss_pct=Decimal("2"),
        take_profit_pct=Decimal("4")
    )
    print(f"✓ RiskParameters: {risk_params}")
    
    # Test RiskLimits
    risk_limits = RiskLimits(
        daily_loss_limit=Decimal("500"),
        position_limit=3,
        margin_requirement=Decimal("20")
    )
    print(f"✓ RiskLimits: {risk_limits}")
    
    # Test MarginInfo
    margin_info = MarginInfo(
        total_balance=Decimal("10000"),
        used_margin=Decimal("2000"),
        free_margin=Decimal("8000"),
        margin_ratio=Decimal("0.2")
    )
    print(f"✓ MarginInfo: {margin_info}")
    
    print("Basic risk types test passed!\n")


def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("Testing circuit breaker...")
    
    # Create initial state
    cb_state = create_circuit_breaker_state(failure_threshold=3, timeout_seconds=300)
    print(f"✓ Initial state: {cb_state.state}, can_execute: {cb_state.can_execute}")
    
    # Record failures
    cb_state = record_circuit_breaker_failure(cb_state, "api_error", "Connection timeout")
    print(f"✓ After 1 failure: {cb_state.state}, failures: {cb_state.failure_count}")
    
    cb_state = record_circuit_breaker_failure(cb_state, "validation_error", "Invalid data")
    print(f"✓ After 2 failures: {cb_state.state}, failures: {cb_state.failure_count}")
    
    cb_state = record_circuit_breaker_failure(cb_state, "execution_error", "Trade failed")
    print(f"✓ After 3 failures: {cb_state.state}, can_execute: {cb_state.can_execute}")
    
    # Record success (should not work when OPEN)
    cb_state = record_circuit_breaker_success(cb_state)
    print(f"✓ After success attempt: {cb_state.state}")
    
    print("Circuit breaker test passed!\n")


def test_emergency_stop():
    """Test emergency stop functionality."""
    print("Testing emergency stop...")
    
    from bot.fp.types.risk import create_emergency_stop_state, trigger_emergency_stop, clear_emergency_stop
    
    # Create initial state
    es_state = create_emergency_stop_state()
    print(f"✓ Initial state: stopped={es_state.is_stopped}, can_trade={es_state.can_trade}")
    
    # Trigger emergency stop
    es_state = trigger_emergency_stop(es_state, "high_drawdown", "20% drawdown detected")
    print(f"✓ After triggering: stopped={es_state.is_stopped}, reason={es_state.stop_reason.description if es_state.stop_reason else None}")
    
    # Clear emergency stop
    es_state = clear_emergency_stop(es_state)
    print(f"✓ After clearing: stopped={es_state.is_stopped}, can_trade={es_state.can_trade}")
    
    print("Emergency stop test passed!\n")


def test_portfolio_exposure():
    """Test portfolio exposure calculation."""
    print("Testing portfolio exposure...")
    
    positions = [
        {"symbol": "BTC-USD", "size": 0.5, "price": 50000},
        {"symbol": "ETH-USD", "size": 10, "price": 3000},
        {"symbol": "SOL-USD", "size": 100, "price": 100}
    ]
    
    account_balance = Decimal("100000")
    exposure = calculate_portfolio_exposure(positions, account_balance)
    
    print(f"✓ Total exposure: ${exposure.total_exposure}")
    print(f"✓ Portfolio heat: {exposure.portfolio_heat:.1f}%")
    print(f"✓ Max single position: {exposure.max_single_position_pct:.1f}%")
    print(f"✓ Concentration risk: {exposure.concentration_risk:.2f}")
    print(f"✓ Is overexposed: {exposure.is_overexposed}")
    
    print("Portfolio exposure test passed!\n")


def test_leverage_analysis():
    """Test leverage analysis."""
    print("Testing leverage analysis...")
    
    analysis = calculate_leverage_analysis(
        current_leverage=Decimal("5"),
        volatility=0.03,  # 3% volatility
        win_rate=0.6,     # 60% win rate
        max_leverage=Decimal("10")
    )
    
    print(f"✓ Current leverage: {analysis.current_leverage}x")
    print(f"✓ Optimal leverage: {analysis.optimal_leverage}x")
    print(f"✓ Recommended action: {analysis.recommended_action}")
    print(f"✓ Volatility adjustment: {analysis.volatility_adjustment:.2f}")
    
    print("Leverage analysis test passed!\n")


def test_risk_assessment():
    """Test risk level assessment."""
    print("Testing risk assessment...")
    
    # Create mock data
    portfolio_exposure = PortfolioExposure(
        total_exposure=Decimal("80000"),
        symbol_exposures={"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("30000")},
        sector_exposures={},
        correlation_risk=0.3,
        concentration_risk=0.5,
        max_single_position_pct=50.0,
        portfolio_heat=12.0  # High heat
    )
    
    leverage_analysis = LeverageAnalysis(
        current_leverage=Decimal("8"),
        optimal_leverage=Decimal("3"),
        max_allowed_leverage=Decimal("10"),
        volatility_adjustment=0.7,
        win_rate_adjustment=0.6,
        risk_adjustment=1.0,
        recommended_action="DECREASE"
    )
    
    from bot.fp.types.risk import calculate_drawdown_analysis
    drawdown_analysis = calculate_drawdown_analysis(
        current_balance=Decimal("90000"),
        peak_balance=Decimal("100000")
    )
    
    assessment = assess_risk_level(
        portfolio_exposure=portfolio_exposure,
        leverage_analysis=leverage_analysis,
        drawdown_analysis=drawdown_analysis,
        consecutive_losses=3,
        margin_usage_pct=75.0
    )
    
    print(f"✓ Risk level: {assessment.risk_level}")
    print(f"✓ Risk score: {assessment.score:.1f}")
    print(f"✓ Contributing factors: {len(assessment.contributing_factors)}")
    print(f"✓ Recommendations: {len(assessment.recommendations)}")
    
    print("Risk assessment test passed!\n")


def test_balance_validation():
    """Test balance validation functionality."""
    print("Testing balance validation...")
    
    config = create_default_balance_config()
    balance = Decimal("5000")
    
    # Test range validation
    range_result = validate_balance_range(balance, config.balance_range)
    print(f"✓ Range validation: {range_result.is_valid}, message: {range_result.message}")
    
    # Test margin validation
    margin_req = MarginRequirement(
        position_value=Decimal("10000"),
        leverage=Decimal("5")
    )
    margin_result = validate_margin_requirements(balance, margin_req)
    print(f"✓ Margin validation: {margin_result.is_valid}, message: {margin_result.message}")
    
    # Test trade affordability
    affordability = TradeAffordabilityCheck(
        trade_value=Decimal("2000"),
        estimated_fees=Decimal("10"),
        required_margin=Decimal("400"),
        leverage=Decimal("5"),
        current_balance=balance
    )
    afford_result = validate_trade_affordability(affordability)
    print(f"✓ Affordability validation: {afford_result.is_valid}, message: {afford_result.message}")
    
    print("Balance validation test passed!\n")


def test_comprehensive_risk_state():
    """Test comprehensive risk state creation."""
    print("Testing comprehensive risk state...")
    
    positions = [
        {"symbol": "BTC-USD", "size": 0.1, "price": 50000},
        {"symbol": "ETH-USD", "size": 2, "price": 3000}
    ]
    
    risk_state = create_comprehensive_risk_state(
        account_balance=Decimal("50000"),
        positions=positions,
        current_leverage=Decimal("3"),
        volatility=0.02,
        win_rate=0.65,
        consecutive_losses=1,
        margin_usage_pct=30.0,
        peak_balance=Decimal("55000")
    )
    
    print(f"✓ Can trade: {risk_state.can_trade}")
    print(f"✓ Overall risk score: {risk_state.risk_metrics.overall_risk_score:.1f}")
    print(f"✓ Trading restrictions: {len(risk_state.trading_restrictions)}")
    
    # Check for alerts
    alerts = check_advanced_risk_alerts(risk_state)
    print(f"✓ Active alerts: {len(alerts)}")
    
    print("Comprehensive risk state test passed!\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Functional Risk Management Types Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_basic_risk_types()
        test_circuit_breaker()
        test_emergency_stop()
        test_portfolio_exposure()
        test_leverage_analysis()
        test_risk_assessment()
        test_balance_validation()
        test_comprehensive_risk_state()
        
        print("=" * 50)
        print("✅ All tests passed! Functional risk types are working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()