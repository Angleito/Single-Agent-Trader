"""
Functional programming risk management tests.

This module provides comprehensive tests for functional risk management using
immutable types, pure functions, and monadic error handling while maintaining
all critical safety validations.
"""

import unittest
from datetime import UTC, datetime, date
from decimal import Decimal
from typing import List, Dict, Any

import pytest
from hypothesis import given, strategies as st

# Functional programming imports
from bot.fp.types.risk import (
    # Core types
    RiskParameters,
    RiskLimits,
    RiskMetrics,
    MarginInfo,
    DailyPnL,
    FailureRecord,
    CircuitBreakerState,
    EmergencyStopState,
    EmergencyStopReason,
    APIProtectionState,
    RiskValidationResult,
    PositionValidationResult,
    RiskLevelAssessment,
    PortfolioExposure,
    LeverageAnalysis,
    DrawdownAnalysis,
    RiskMetricsSnapshot,
    ComprehensiveRiskState,
    AdvancedRiskAlert,
    AdvancedRiskAlertType,
    
    # Pure calculation functions
    calculate_position_size,
    calculate_margin_ratio,
    calculate_free_margin,
    calculate_max_position_value,
    calculate_required_margin,
    calculate_stop_loss_price,
    calculate_take_profit_price,
    check_risk_alerts,
    calculate_position_risk,
    is_within_risk_limits,
    
    # Advanced risk functions
    create_circuit_breaker_state,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
    update_circuit_breaker_state,
    create_emergency_stop_state,
    trigger_emergency_stop,
    clear_emergency_stop,
    create_api_protection_state,
    record_api_failure,
    record_api_success,
    calculate_portfolio_exposure,
    calculate_leverage_analysis,
    calculate_drawdown_analysis,
    assess_risk_level,
    create_comprehensive_risk_state,
    check_advanced_risk_alerts,
)

from bot.fp.strategies.risk_management import (
    calculate_kelly_criterion,
    calculate_fixed_fractional_size,
    calculate_volatility_based_size,
    calculate_atr_stop_loss,
    calculate_percentage_stop_loss,
    calculate_risk_reward_take_profit,
    calculate_trailing_stop,
    calculate_portfolio_heat,
    enforce_risk_limits,
    calculate_position_size_with_stop,
    calculate_correlation_adjustment,
    calculate_optimal_leverage,
    calculate_drawdown_adjusted_size,
)

from bot.fp.types.effects import Result, Ok, Err, Maybe, Some, Nothing
from bot.fp.types.portfolio import Position


class TestFunctionalRiskTypes:
    """Test immutable risk management data types."""
    
    def test_risk_parameters_immutability(self):
        """Test RiskParameters is immutable."""
        params = RiskParameters(
            max_position_size=Decimal("10.0"),
            max_leverage=Decimal("5.0"),
            stop_loss_pct=Decimal("2.0"),
            take_profit_pct=Decimal("4.0")
        )
        
        # Should be frozen
        with pytest.raises(AttributeError):
            params.max_position_size = Decimal("20.0")
            
        assert params.max_position_size == Decimal("10.0")
        assert params.max_leverage == Decimal("5.0")
    
    def test_risk_limits_validation(self):
        """Test RiskLimits validation and immutability."""
        limits = RiskLimits(
            daily_loss_limit=Decimal("500.0"),
            position_limit=5,
            margin_requirement=Decimal("20.0")
        )
        
        assert limits.daily_loss_limit == Decimal("500.0")
        assert limits.position_limit == 5
        assert limits.margin_requirement == Decimal("20.0")
        
        # Should be immutable
        with pytest.raises(AttributeError):
            limits.daily_loss_limit = Decimal("1000.0")
    
    def test_margin_info_calculations(self):
        """Test MarginInfo calculations and properties."""
        margin = MarginInfo(
            total_balance=Decimal("10000.0"),
            used_margin=Decimal("2000.0"),
            free_margin=Decimal("8000.0"),
            margin_ratio=Decimal("0.2")
        )
        
        assert margin.total_balance == Decimal("10000.0")
        assert margin.used_margin == Decimal("2000.0")
        assert margin.free_margin == Decimal("8000.0")
        assert margin.margin_ratio == Decimal("0.2")
    
    def test_daily_pnl_tracking(self):
        """Test DailyPnL tracking and total calculation."""
        today = date.today()
        pnl = DailyPnL(
            date=today,
            realized_pnl=Decimal("150.0"),
            unrealized_pnl=Decimal("-25.0"),
            trades_count=5,
            max_drawdown=Decimal("-50.0")
        )
        
        assert pnl.date == today
        assert pnl.realized_pnl == Decimal("150.0")
        assert pnl.unrealized_pnl == Decimal("-25.0")
        assert pnl.trades_count == 5
        assert pnl.total_pnl == Decimal("125.0")  # 150 - 25
    
    def test_circuit_breaker_state_properties(self):
        """Test CircuitBreakerState properties and state checking."""
        cb_state = CircuitBreakerState(
            state="OPEN",
            failure_count=5,
            failure_threshold=3,
            timeout_seconds=300,
            last_failure_time=datetime.now(UTC),
            consecutive_successes=0,
            failure_history=()
        )
        
        assert cb_state.is_open is True
        assert cb_state.is_closed is False
        assert cb_state.is_half_open is False
        assert cb_state.can_execute is False
    
    def test_emergency_stop_state(self):
        """Test EmergencyStopState functionality."""
        stop_reason = EmergencyStopReason(
            reason_type="market_volatility",
            description="Extreme market volatility detected",
            triggered_at=datetime.now(UTC),
            severity="critical"
        )
        
        emergency_state = EmergencyStopState(
            is_stopped=True,
            stop_reason=stop_reason,
            stopped_at=datetime.now(UTC),
            manual_override=False
        )
        
        assert emergency_state.is_stopped is True
        assert emergency_state.can_trade is False
        assert emergency_state.stop_reason.reason_type == "market_volatility"


class TestPureRiskCalculations:
    """Test pure risk calculation functions."""
    
    def test_calculate_position_size(self):
        """Test position size calculation based on risk parameters."""
        balance = Decimal("10000.0")
        risk_per_trade = Decimal("2.0")  # 2%
        stop_loss_pct = Decimal("1.0")  # 1%
        
        position_size = calculate_position_size(balance, risk_per_trade, stop_loss_pct)
        expected = Decimal("20000.0")  # (10000 * 0.02) / 0.01
        
        assert position_size == expected
    
    def test_calculate_position_size_zero_stop_loss(self):
        """Test position size calculation with zero stop loss."""
        position_size = calculate_position_size(
            Decimal("10000.0"), Decimal("2.0"), Decimal("0.0")
        )
        assert position_size == Decimal("0.0")
    
    def test_calculate_margin_ratio(self):
        """Test margin ratio calculation."""
        used_margin = Decimal("2000.0")
        total_balance = Decimal("10000.0")
        
        ratio = calculate_margin_ratio(used_margin, total_balance)
        assert ratio == Decimal("0.2")  # 20%
    
    def test_calculate_margin_ratio_zero_balance(self):
        """Test margin ratio with zero balance."""
        ratio = calculate_margin_ratio(Decimal("1000.0"), Decimal("0.0"))
        assert ratio == Decimal("0.0")
    
    def test_calculate_free_margin(self):
        """Test free margin calculation."""
        total_balance = Decimal("10000.0")
        used_margin = Decimal("3000.0")
        
        free_margin = calculate_free_margin(total_balance, used_margin)
        assert free_margin == Decimal("7000.0")
    
    def test_calculate_required_margin(self):
        """Test required margin calculation for position."""
        position_size = Decimal("1.0")
        entry_price = Decimal("50000.0")
        leverage = Decimal("5.0")
        
        required = calculate_required_margin(position_size, entry_price, leverage)
        assert required == Decimal("10000.0")  # (1 * 50000) / 5
    
    def test_calculate_required_margin_zero_leverage(self):
        """Test required margin with zero leverage."""
        required = calculate_required_margin(
            Decimal("1.0"), Decimal("50000.0"), Decimal("0.0")
        )
        assert required == Decimal("0.0")
    
    def test_calculate_stop_loss_price_long(self):
        """Test stop loss calculation for long position."""
        entry_price = Decimal("50000.0")
        stop_loss_pct = Decimal("2.0")  # 2%
        
        stop_price = calculate_stop_loss_price(entry_price, stop_loss_pct, True)
        expected = Decimal("49000.0")  # 50000 - (50000 * 0.02)
        
        assert stop_price == expected
    
    def test_calculate_stop_loss_price_short(self):
        """Test stop loss calculation for short position."""
        entry_price = Decimal("50000.0")
        stop_loss_pct = Decimal("2.0")  # 2%
        
        stop_price = calculate_stop_loss_price(entry_price, stop_loss_pct, False)
        expected = Decimal("51000.0")  # 50000 + (50000 * 0.02)
        
        assert stop_price == expected
    
    def test_calculate_take_profit_price_long(self):
        """Test take profit calculation for long position."""
        entry_price = Decimal("50000.0")
        take_profit_pct = Decimal("4.0")  # 4%
        
        tp_price = calculate_take_profit_price(entry_price, take_profit_pct, True)
        expected = Decimal("52000.0")  # 50000 + (50000 * 0.04)
        
        assert tp_price == expected
    
    def test_calculate_take_profit_price_short(self):
        """Test take profit calculation for short position."""
        entry_price = Decimal("50000.0")
        take_profit_pct = Decimal("4.0")  # 4%
        
        tp_price = calculate_take_profit_price(entry_price, take_profit_pct, False)
        expected = Decimal("48000.0")  # 50000 - (50000 * 0.04)
        
        assert tp_price == expected


class TestRiskAlerts:
    """Test risk alert generation and validation."""
    
    def test_check_risk_alerts_position_limit(self):
        """Test position limit alert generation."""
        margin_info = MarginInfo(
            total_balance=Decimal("10000.0"),
            used_margin=Decimal("1000.0"),
            free_margin=Decimal("9000.0"),
            margin_ratio=Decimal("0.1")
        )
        
        limits = RiskLimits(
            daily_loss_limit=Decimal("500.0"),
            position_limit=3,
            margin_requirement=Decimal("50.0")
        )
        
        current_positions = 4  # Exceeds limit
        daily_pnl = Decimal("0.0")
        
        alerts = check_risk_alerts(margin_info, limits, current_positions, daily_pnl)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.current_positions == 4
        assert alert.limit == 3
    
    def test_check_risk_alerts_margin_call(self):
        """Test margin call alert generation."""
        margin_info = MarginInfo(
            total_balance=Decimal("10000.0"),
            used_margin=Decimal("6000.0"),  # 60% usage
            free_margin=Decimal("4000.0"),
            margin_ratio=Decimal("0.6")
        )
        
        limits = RiskLimits(
            daily_loss_limit=Decimal("500.0"),
            position_limit=5,
            margin_requirement=Decimal("50.0")  # 50% threshold
        )
        
        alerts = check_risk_alerts(margin_info, limits, 2, Decimal("0.0"))
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.margin_ratio == Decimal("0.6")
        assert alert.threshold == Decimal("0.5")
    
    def test_check_risk_alerts_daily_loss(self):
        """Test daily loss limit alert generation."""
        margin_info = MarginInfo(
            total_balance=Decimal("10000.0"),
            used_margin=Decimal("1000.0"),
            free_margin=Decimal("9000.0"),
            margin_ratio=Decimal("0.1")
        )
        
        limits = RiskLimits(
            daily_loss_limit=Decimal("500.0"),
            position_limit=5,
            margin_requirement=Decimal("50.0")
        )
        
        daily_pnl = Decimal("-600.0")  # Exceeds loss limit
        
        alerts = check_risk_alerts(margin_info, limits, 2, daily_pnl)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.current_loss == Decimal("600.0")
        assert alert.limit == Decimal("500.0")
    
    def test_check_risk_alerts_no_alerts(self):
        """Test no alerts when all metrics are within limits."""
        margin_info = MarginInfo(
            total_balance=Decimal("10000.0"),
            used_margin=Decimal("2000.0"),  # 20% usage
            free_margin=Decimal("8000.0"),
            margin_ratio=Decimal("0.2")
        )
        
        limits = RiskLimits(
            daily_loss_limit=Decimal("500.0"),
            position_limit=5,
            margin_requirement=Decimal("50.0")  # 50% threshold
        )
        
        alerts = check_risk_alerts(margin_info, limits, 3, Decimal("100.0"))
        
        assert len(alerts) == 0


class TestCircuitBreakerFunctionality:
    """Test circuit breaker state management."""
    
    def test_create_circuit_breaker_state(self):
        """Test creation of initial circuit breaker state."""
        cb = create_circuit_breaker_state(failure_threshold=3, timeout_seconds=60)
        
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        assert cb.failure_threshold == 3
        assert cb.timeout_seconds == 60
        assert cb.last_failure_time is None
        assert cb.consecutive_successes == 0
        assert len(cb.failure_history) == 0
    
    def test_record_circuit_breaker_failure(self):
        """Test recording failures in circuit breaker."""
        cb = create_circuit_breaker_state(failure_threshold=3)
        
        # Record first failure
        cb = record_circuit_breaker_failure(
            cb, "api_error", "Timeout occurred", "medium"
        )
        
        assert cb.failure_count == 1
        assert cb.state == "CLOSED"  # Still closed
        assert len(cb.failure_history) == 1
        assert cb.failure_history[0].failure_type == "api_error"
    
    def test_circuit_breaker_opens_on_threshold(self):
        """Test circuit breaker opens when threshold is reached."""
        cb = create_circuit_breaker_state(failure_threshold=2)
        
        # Record failures to reach threshold
        cb = record_circuit_breaker_failure(cb, "error1", "First error")
        cb = record_circuit_breaker_failure(cb, "error2", "Second error")
        
        assert cb.failure_count == 2
        assert cb.state == "OPEN"
        assert len(cb.failure_history) == 2
    
    def test_record_circuit_breaker_success(self):
        """Test recording successes in circuit breaker."""
        cb = create_circuit_breaker_state()
        cb = record_circuit_breaker_success(cb)
        
        assert cb.consecutive_successes == 1
        assert cb.failure_count == 0
    
    def test_circuit_breaker_half_open_to_closed(self):
        """Test transition from HALF_OPEN to CLOSED on successes."""
        cb = CircuitBreakerState(
            state="HALF_OPEN",
            failure_count=3,
            failure_threshold=3,
            timeout_seconds=60,
            last_failure_time=datetime.now(UTC),
            consecutive_successes=0,
            failure_history=()
        )
        
        # Record enough successes to close
        cb = record_circuit_breaker_success(cb)
        cb = record_circuit_breaker_success(cb)
        cb = record_circuit_breaker_success(cb)
        
        assert cb.state == "CLOSED"
        assert cb.consecutive_successes == 0  # Reset after closing


class TestEmergencyStopFunctionality:
    """Test emergency stop state management."""
    
    def test_create_emergency_stop_state(self):
        """Test creation of initial emergency stop state."""
        stop = create_emergency_stop_state()
        
        assert stop.is_stopped is False
        assert stop.stop_reason is None
        assert stop.stopped_at is None
        assert stop.manual_override is False
        assert stop.can_trade is True
    
    def test_trigger_emergency_stop(self):
        """Test triggering emergency stop."""
        stop = create_emergency_stop_state()
        
        stop = trigger_emergency_stop(
            stop, "market_crash", "Extreme market volatility detected"
        )
        
        assert stop.is_stopped is True
        assert stop.stop_reason is not None
        assert stop.stop_reason.reason_type == "market_crash"
        assert stop.stop_reason.description == "Extreme market volatility detected"
        assert stop.stopped_at is not None
        assert stop.can_trade is False
    
    def test_clear_emergency_stop(self):
        """Test clearing emergency stop."""
        stop = trigger_emergency_stop(
            create_emergency_stop_state(), "test", "Test stop"
        )
        
        assert stop.is_stopped is True
        
        stop = clear_emergency_stop(stop)
        
        assert stop.is_stopped is False
        assert stop.stop_reason is None
        assert stop.stopped_at is None
        assert stop.can_trade is True
    
    def test_emergency_stop_manual_override(self):
        """Test emergency stop with manual override."""
        stop = trigger_emergency_stop(
            create_emergency_stop_state(), "test", "Test stop"
        )
        
        stop = clear_emergency_stop(stop, manual_override=True)
        
        assert stop.is_stopped is False
        assert stop.manual_override is True
        assert stop.can_trade is True


class TestAPIProtectionFunctionality:
    """Test API protection state management."""
    
    def test_create_api_protection_state(self):
        """Test creation of initial API protection state."""
        api = create_api_protection_state(max_retries=3, base_delay=1.0)
        
        assert api.consecutive_failures == 0
        assert api.max_retries == 3
        assert api.base_delay == 1.0
        assert api.is_healthy is True
        assert api.can_retry is True
        assert api.next_retry_delay == 1.0
    
    def test_record_api_failure(self):
        """Test recording API failures."""
        api = create_api_protection_state(max_retries=3)
        
        api = record_api_failure(api)
        
        assert api.consecutive_failures == 1
        assert api.is_healthy is True  # Still healthy
        assert api.next_retry_delay == 2.0  # 1.0 * 2^1
    
    def test_api_protection_unhealthy_on_max_failures(self):
        """Test API protection becomes unhealthy on max failures."""
        api = create_api_protection_state(max_retries=2)
        
        # Record failures to exceed max
        api = record_api_failure(api)
        api = record_api_failure(api)
        
        assert api.consecutive_failures == 2
        assert api.is_healthy is False
        assert api.can_retry is False
    
    def test_record_api_success(self):
        """Test recording API success resets failures."""
        api = create_api_protection_state(max_retries=3)
        api = record_api_failure(api)
        api = record_api_failure(api)
        
        assert api.consecutive_failures == 2
        
        api = record_api_success(api)
        
        assert api.consecutive_failures == 0
        assert api.is_healthy is True


class TestAdvancedRiskCalculations:
    """Test advanced risk calculation functions."""
    
    def test_calculate_portfolio_exposure(self):
        """Test portfolio exposure calculation."""
        positions = [
            {"symbol": "BTC-USD", "size": 0.5, "price": 50000},
            {"symbol": "ETH-USD", "size": 10, "price": 3000},
        ]
        account_balance = Decimal("100000")
        
        exposure = calculate_portfolio_exposure(positions, account_balance)
        
        # Total exposure: (0.5 * 50000) + (10 * 3000) = 25000 + 30000 = 55000
        assert exposure.total_exposure == Decimal("55000")
        assert exposure.symbol_exposures["BTC-USD"] == Decimal("25000")
        assert exposure.symbol_exposures["ETH-USD"] == Decimal("30000")
        assert exposure.portfolio_heat == 55.0  # 55000/100000 * 100
    
    def test_calculate_leverage_analysis(self):
        """Test leverage analysis calculation."""
        current_leverage = Decimal("3.0")
        volatility = 0.15  # 15%
        win_rate = 0.6  # 60%
        max_leverage = Decimal("10.0")
        
        analysis = calculate_leverage_analysis(
            current_leverage, volatility, win_rate, max_leverage
        )
        
        assert analysis.current_leverage == Decimal("3.0")
        assert analysis.max_allowed_leverage == Decimal("10.0")
        assert analysis.volatility_adjustment == 0.85  # 1.0 - 0.15
        assert analysis.win_rate_adjustment == 0.6
        assert analysis.recommended_action in ["INCREASE", "DECREASE", "MAINTAIN"]
    
    def test_calculate_drawdown_analysis(self):
        """Test drawdown analysis calculation."""
        current_balance = Decimal("8000")
        peak_balance = Decimal("10000")
        
        analysis = calculate_drawdown_analysis(current_balance, peak_balance)
        
        assert analysis.current_drawdown_pct == 20.0  # (10000-8000)/10000*100
        assert analysis.peak_balance == Decimal("10000")
        assert analysis.current_balance == Decimal("8000")
        assert analysis.is_in_drawdown is True
        assert analysis.recovery_target == Decimal("10000")
    
    def test_assess_risk_level(self):
        """Test overall risk level assessment."""
        portfolio_exposure = PortfolioExposure(
            total_exposure=Decimal("50000"),
            symbol_exposures={},
            sector_exposures={},
            correlation_risk=0.3,
            concentration_risk=0.2,
            max_single_position_pct=15.0,
            portfolio_heat=12.0  # High heat
        )
        
        leverage_analysis = LeverageAnalysis(
            current_leverage=Decimal("8.0"),  # High leverage
            optimal_leverage=Decimal("4.0"),
            max_allowed_leverage=Decimal("10.0"),
            volatility_adjustment=0.8,
            win_rate_adjustment=0.6,
            risk_adjustment=1.0,
            recommended_action="DECREASE"
        )
        
        drawdown_analysis = DrawdownAnalysis(
            current_drawdown_pct=5.0,  # Moderate drawdown
            max_drawdown_pct=5.0,
            drawdown_duration_hours=2.0,
            peak_balance=Decimal("10000"),
            current_balance=Decimal("9500"),
            is_in_drawdown=True,
            recovery_target=Decimal("10000")
        )
        
        assessment = assess_risk_level(
            portfolio_exposure, leverage_analysis, drawdown_analysis,
            consecutive_losses=2, margin_usage_pct=30.0
        )
        
        assert assessment.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert assessment.score >= 0
        assert len(assessment.contributing_factors) >= 0
        assert len(assessment.recommendations) > 0


class TestStrategicRiskFunctions:
    """Test strategic risk management functions from strategies module."""
    
    def test_calculate_kelly_criterion(self):
        """Test Kelly Criterion position sizing."""
        win_probability = 0.6  # 60% win rate
        win_loss_ratio = 2.0  # 2:1 reward/risk
        kelly_fraction = 0.25  # Use 25% of Kelly
        
        size = calculate_kelly_criterion(win_probability, win_loss_ratio, kelly_fraction)
        
        # Kelly = (0.6 * 2.0 - 0.4) / 2.0 = 0.4
        # With 25% fraction = 0.4 * 0.25 = 0.1
        assert size == 0.1
    
    def test_calculate_kelly_criterion_edge_cases(self):
        """Test Kelly Criterion edge cases."""
        # Invalid win probability
        assert calculate_kelly_criterion(0.0, 2.0) == 0.0
        assert calculate_kelly_criterion(1.0, 2.0) == 0.0
        
        # Invalid win/loss ratio
        assert calculate_kelly_criterion(0.6, 0.0) == 0.0
        assert calculate_kelly_criterion(0.6, -1.0) == 0.0
    
    def test_calculate_fixed_fractional_size(self):
        """Test fixed fractional position sizing."""
        account_balance = 10000.0
        risk_percentage = 2.0  # 2%
        
        size = calculate_fixed_fractional_size(account_balance, risk_percentage)
        
        assert size == 200.0  # 10000 * 0.02
    
    def test_calculate_volatility_based_size(self):
        """Test volatility-based position sizing."""
        account_balance = 10000.0
        volatility = 0.04  # 4%
        target_volatility = 0.02  # 2%
        max_position_pct = 10.0
        
        size = calculate_volatility_based_size(
            account_balance, volatility, target_volatility, max_position_pct
        )
        
        # Base position: 1000 (10%)
        # Volatility adjustment: 0.02/0.04 = 0.5
        # Final size: 1000 * 0.5 = 500
        assert size == 500.0
    
    def test_calculate_atr_stop_loss(self):
        """Test ATR-based stop loss calculation."""
        entry_price = 50000.0
        atr = 1000.0
        atr_multiplier = 2.0
        
        # Long position
        stop_long = calculate_atr_stop_loss(entry_price, atr, atr_multiplier, True)
        assert stop_long == 48000.0  # 50000 - (1000 * 2)
        
        # Short position
        stop_short = calculate_atr_stop_loss(entry_price, atr, atr_multiplier, False)
        assert stop_short == 52000.0  # 50000 + (1000 * 2)
    
    def test_calculate_percentage_stop_loss(self):
        """Test percentage-based stop loss calculation."""
        entry_price = 50000.0
        stop_percentage = 2.0  # 2%
        
        # Long position
        stop_long = calculate_percentage_stop_loss(entry_price, stop_percentage, True)
        assert stop_long == 49000.0  # 50000 - (50000 * 0.02)
        
        # Short position
        stop_short = calculate_percentage_stop_loss(entry_price, stop_percentage, False)
        assert stop_short == 51000.0  # 50000 + (50000 * 0.02)
    
    def test_calculate_risk_reward_take_profit(self):
        """Test risk/reward ratio take profit calculation."""
        entry_price = 50000.0
        stop_loss = 49000.0  # 1000 risk
        risk_reward_ratio = 2.0
        
        # Long position
        tp_long = calculate_risk_reward_take_profit(
            entry_price, stop_loss, risk_reward_ratio, True
        )
        assert tp_long == 52000.0  # 50000 + (1000 * 2)
        
        # Short position
        stop_loss_short = 51000.0
        tp_short = calculate_risk_reward_take_profit(
            entry_price, stop_loss_short, risk_reward_ratio, False
        )
        assert tp_short == 48000.0  # 50000 - (1000 * 2)
    
    def test_calculate_portfolio_heat(self):
        """Test portfolio heat calculation."""
        positions = [
            {
                "size": 1000.0,  # $1000 position
                "entry_price": 50000.0,
                "stop_loss": 49000.0,
                "is_long": True
            },
            {
                "size": 3000.0,  # $3000 position
                "entry_price": 3000.0,
                "stop_loss": 3060.0,
                "is_long": False
            }
        ]
        account_balance = 10000.0
        
        heat = calculate_portfolio_heat(positions, account_balance)
        
        # Position 1: 1000/50000 = 0.02 units, risk = 0.02 * 1000 = 20
        # Position 2: 3000/3000 = 1 unit, risk = 1 * 60 = 60
        # Total risk: 80, heat = 80/10000 * 100 = 0.8%
        assert heat == 0.8
    
    def test_enforce_risk_limits(self):
        """Test risk limit enforcement."""
        proposed_size = 2000.0
        current_positions = []
        account_balance = 10000.0
        max_position_size_pct = 10.0  # 10% max
        
        adjusted_size, reason = enforce_risk_limits(
            proposed_size,
            current_positions,
            account_balance,
            max_position_size_pct
        )
        
        # Should be reduced to 10% of balance
        assert adjusted_size == 1000.0
        assert "max position size" in reason
    
    def test_calculate_position_size_with_stop(self):
        """Test position sizing with stop loss consideration."""
        account_balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0  # 2% stop
        risk_percentage = 2.0  # Risk 2% of account
        
        size = calculate_position_size_with_stop(
            account_balance, entry_price, stop_loss, risk_percentage, True
        )
        
        # Risk amount: 10000 * 0.02 = 200
        # Risk per unit: 50000 - 49000 = 1000
        # Units: 200 / 1000 = 0.2
        # Position size: 0.2 * 50000 = 10000
        assert size == 10000.0
    
    def test_calculate_optimal_leverage(self):
        """Test optimal leverage calculation."""
        volatility = 0.10  # 10%
        win_rate = 0.65  # 65%
        risk_per_trade = 2.0
        max_leverage = 10.0
        
        leverage = calculate_optimal_leverage(
            volatility, win_rate, risk_per_trade, max_leverage
        )
        
        assert 1.0 <= leverage <= max_leverage
        assert isinstance(leverage, float)
    
    def test_calculate_drawdown_adjusted_size(self):
        """Test drawdown-adjusted position sizing."""
        base_size = 1000.0
        current_drawdown_pct = 10.0
        max_drawdown_pct = 20.0
        reduction_factor = 0.5
        
        adjusted_size = calculate_drawdown_adjusted_size(
            base_size, current_drawdown_pct, max_drawdown_pct, reduction_factor
        )
        
        # At 10% drawdown with 20% max: 10/20 = 0.5 ratio
        # Adjustment: 1.0 - (0.5 * (1.0 - 0.5)) = 0.75
        # Final size: 1000 * 0.75 = 750
        assert adjusted_size == 750.0


class TestPropertyBasedRiskTests:
    """Property-based tests for risk calculations using hypothesis."""
    
    @given(
        balance=st.decimals(min_value=Decimal("1000"), max_value=Decimal("1000000")),
        risk_pct=st.decimals(min_value=Decimal("0.1"), max_value=Decimal("10")),
        stop_pct=st.decimals(min_value=Decimal("0.1"), max_value=Decimal("5"))
    )
    def test_position_size_properties(self, balance, risk_pct, stop_pct):
        """Test position size calculation properties."""
        position_size = calculate_position_size(balance, risk_pct, stop_pct)
        
        # Position size should be positive
        assert position_size >= 0
        
        # Position size should scale with balance
        larger_position = calculate_position_size(balance * 2, risk_pct, stop_pct)
        assert larger_position == position_size * 2
    
    @given(
        used_margin=st.decimals(min_value=Decimal("0"), max_value=Decimal("10000")),
        total_balance=st.decimals(min_value=Decimal("10000"), max_value=Decimal("100000"))
    )
    def test_margin_ratio_properties(self, used_margin, total_balance):
        """Test margin ratio calculation properties."""
        # Ensure used margin doesn't exceed total balance
        used_margin = min(used_margin, total_balance)
        
        ratio = calculate_margin_ratio(used_margin, total_balance)
        
        # Ratio should be between 0 and 1
        assert Decimal("0") <= ratio <= Decimal("1")
        
        # Ratio should be proportional
        if used_margin == total_balance:
            assert ratio == Decimal("1")
        if used_margin == Decimal("0"):
            assert ratio == Decimal("0")
    
    @given(
        entry_price=st.decimals(min_value=Decimal("100"), max_value=Decimal("100000")),
        stop_pct=st.decimals(min_value=Decimal("0.1"), max_value=Decimal("10"))
    )
    def test_stop_loss_price_properties(self, entry_price, stop_pct):
        """Test stop loss price calculation properties."""
        # Long position stop should be below entry
        stop_long = calculate_stop_loss_price(entry_price, stop_pct, True)
        assert stop_long < entry_price
        
        # Short position stop should be above entry
        stop_short = calculate_stop_loss_price(entry_price, stop_pct, False)
        assert stop_short > entry_price
        
        # Stop distance should be proportional to percentage
        larger_stop_long = calculate_stop_loss_price(entry_price, stop_pct * 2, True)
        assert (entry_price - larger_stop_long) > (entry_price - stop_long)


class TestRiskIntegrationScenarios:
    """Test complex integration scenarios for risk management."""
    
    def test_complete_risk_assessment_flow(self):
        """Test complete risk assessment flow with all components."""
        # Create sample portfolio positions
        positions = [
            {"symbol": "BTC-USD", "size": 1.0, "price": 50000},
            {"symbol": "ETH-USD", "size": 10.0, "price": 3000},
        ]
        
        account_balance = Decimal("100000")
        current_leverage = Decimal("3.0")
        volatility = 0.12
        win_rate = 0.65
        consecutive_losses = 2
        margin_usage_pct = 25.0
        peak_balance = Decimal("110000")
        
        # Create comprehensive risk state
        risk_state = create_comprehensive_risk_state(
            account_balance=account_balance,
            positions=positions,
            current_leverage=current_leverage,
            volatility=volatility,
            win_rate=win_rate,
            consecutive_losses=consecutive_losses,
            margin_usage_pct=margin_usage_pct,
            peak_balance=peak_balance
        )
        
        # Verify state components
        assert isinstance(risk_state, ComprehensiveRiskState)
        assert risk_state.circuit_breaker.is_closed
        assert not risk_state.emergency_stop.is_stopped
        assert risk_state.api_protection.is_healthy
        
        # Test trading permissions
        assert risk_state.can_trade is True
        assert len(risk_state.trading_restrictions) == 0
        
        # Test risk metrics
        assert risk_state.risk_metrics.account_balance == account_balance
        assert risk_state.risk_metrics.leverage_analysis.current_leverage == current_leverage
        assert risk_state.risk_metrics.consecutive_losses == consecutive_losses
    
    def test_risk_escalation_scenario(self):
        """Test risk escalation with multiple alerts."""
        # Start with normal state
        risk_state = create_comprehensive_risk_state(
            account_balance=Decimal("50000"),  # Reduced balance
            positions=[
                {"symbol": "BTC-USD", "size": 2.0, "price": 50000},  # Large position
            ],
            current_leverage=Decimal("8.0"),  # High leverage
            volatility=0.25,  # High volatility
            win_rate=0.4,  # Poor win rate
            consecutive_losses=6,  # Many losses
            margin_usage_pct=85.0,  # High margin usage
            peak_balance=Decimal("100000")  # Significant drawdown
        )
        
        # Check for advanced alerts
        alerts = check_advanced_risk_alerts(risk_state)
        
        # Should have multiple alerts due to high risk factors
        alert_types = [alert.alert_type for alert in alerts]
        
        # Verify we have critical alerts
        assert len(alerts) > 0
        
        # Check specific alert conditions
        if AdvancedRiskAlertType.LEVERAGE_EXCESSIVE in alert_types:
            leverage_alert = next(
                alert for alert in alerts 
                if alert.alert_type == AdvancedRiskAlertType.LEVERAGE_EXCESSIVE
            )
            assert leverage_alert.severity in ["medium", "high"]
        
        # Verify overall risk assessment
        assert risk_state.risk_metrics.risk_assessment.score > 25  # Should be elevated
    
    def test_circuit_breaker_cascade_scenario(self):
        """Test circuit breaker cascading through states."""
        cb = create_circuit_breaker_state(failure_threshold=2, timeout_seconds=5)
        
        # Record failures to open circuit
        cb = record_circuit_breaker_failure(cb, "api_error", "First failure")
        assert cb.state == "CLOSED"
        
        cb = record_circuit_breaker_failure(cb, "api_error", "Second failure")
        assert cb.state == "OPEN"
        assert not cb.can_execute
        
        # Wait for timeout and update state
        import time
        time.sleep(6)  # Wait longer than timeout
        
        cb = update_circuit_breaker_state(cb)
        assert cb.state == "HALF_OPEN"
        assert cb.can_execute
        
        # Record successes to close
        cb = record_circuit_breaker_success(cb)
        cb = record_circuit_breaker_success(cb)
        cb = record_circuit_breaker_success(cb)
        
        assert cb.state == "CLOSED"
        assert cb.can_execute
    
    def test_position_risk_validation_flow(self):
        """Test complete position risk validation flow."""
        # Test risk calculation for position
        position_size = Decimal("1.0")
        entry_price = Decimal("50000.0")
        stop_loss_price = Decimal("49000.0")
        
        position_risk = calculate_position_risk(
            position_size, entry_price, stop_loss_price
        )
        
        assert position_risk == Decimal("1000.0")
        
        # Test if within risk limits
        current_exposure = Decimal("2000.0")
        balance = Decimal("100000.0")
        max_risk_pct = Decimal("5.0")  # 5%
        
        within_limits = is_within_risk_limits(
            position_risk, current_exposure, balance, max_risk_pct
        )
        
        # Total risk: 1000 + 2000 = 3000
        # Max allowed: 100000 * 0.05 = 5000
        # Should be within limits
        assert within_limits is True
        
        # Test exceeding limits
        large_risk = Decimal("4000.0")
        exceeds_limits = is_within_risk_limits(
            large_risk, current_exposure, balance, max_risk_pct
        )
        
        # Total risk: 4000 + 2000 = 6000 > 5000
        assert exceeds_limits is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])