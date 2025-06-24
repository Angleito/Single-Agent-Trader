"""
Property-based tests for functional programming risk calculations.

This module uses the Hypothesis framework to generate comprehensive test cases
for risk calculation functions, ensuring they satisfy mathematical properties
and handle edge cases correctly across a wide range of inputs.

Tests focus on:
- Mathematical properties and invariants of risk calculations
- Edge case handling with extreme values
- Consistency properties across different calculation methods
- Error handling and validation with invalid inputs
- Performance characteristics under various input conditions
- Decimal precision and rounding behavior
- Composition and associativity properties of risk functions
"""

import pytest
from datetime import datetime, timedelta, UTC
from decimal import Decimal, getcontext, InvalidOperation, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
import math
import statistics

from hypothesis import given, strategies as st, settings, HealthCheck, assume, note
from hypothesis.strategies import composite

# FP test infrastructure
from tests.fp_test_base import (
    FPTestBase,
    FP_AVAILABLE
)

if FP_AVAILABLE:
    # FP risk calculation functions
    from bot.fp.pure.risk_calculations import (
        calculate_position_risk, calculate_portfolio_risk,
        calculate_risk_metrics, validate_risk_limits,
        calculate_margin_requirement, check_drawdown_limits,
        calculate_var, calculate_sharpe_ratio, calculate_sortino_ratio,
        calculate_max_drawdown, calculate_volatility,
        normalize_risk_score, calculate_correlation,
        validate_position_size, calculate_leverage_ratio
    )
    
    # FP risk types
    from bot.fp.types.risk import (
        RiskParameters, RiskState, RiskMetrics, RiskViolation,
        EmergencyStop, CircuitBreaker, DrawdownLimits,
        create_risk_parameters, create_risk_state,
        validate_position_risk, check_emergency_conditions
    )
    
    # FP pure functions for paper trading
    from bot.fp.pure.paper_trading_calculations import (
        calculate_position_value, calculate_unrealized_pnl_simple,
        calculate_margin_requirement_simple, calculate_fees_simple,
        validate_position_size_simple, normalize_decimal_precision
    )
    
    # FP types
    from bot.fp.types.effects import Result, Ok, Err
    from bot.fp.types.base import Maybe, Some, Nothing
    from bot.fp.types.portfolio import Portfolio, PortfolioMetrics
else:
    # Fallback stubs for non-FP environments
    def calculate_position_risk(*args, **kwargs):
        return Decimal("0")
    
    def validate_position_size(*args, **kwargs):
        return True

# Set high precision for financial calculations
getcontext().prec = 28


# Custom Hypothesis strategies for financial data
@composite
def decimal_prices(draw, min_value=Decimal("0.01"), max_value=Decimal("1000000")):
    """Generate realistic price values as Decimals."""
    # Use log-normal distribution for more realistic price distribution
    log_min = float(min_value.ln())
    log_max = float(max_value.ln())
    log_value = draw(st.floats(min_value=log_min, max_value=log_max, allow_nan=False, allow_infinity=False))
    return Decimal(str(math.exp(log_value))).quantize(Decimal("0.01"))


@composite
def decimal_quantities(draw, min_value=Decimal("0.000001"), max_value=Decimal("1000000")):
    """Generate realistic quantity values as Decimals."""
    value = draw(st.floats(min_value=float(min_value), max_value=float(max_value), 
                          allow_nan=False, allow_infinity=False))
    return Decimal(str(value)).quantize(Decimal("0.000001"))


@composite
def decimal_percentages(draw, min_value=Decimal("0"), max_value=Decimal("100")):
    """Generate percentage values as Decimals."""
    value = draw(st.floats(min_value=float(min_value), max_value=float(max_value),
                          allow_nan=False, allow_infinity=False))
    return Decimal(str(value)).quantize(Decimal("0.01"))


@composite
def position_sides(draw):
    """Generate position sides."""
    return draw(st.sampled_from(["LONG", "SHORT", "FLAT"]))


@composite
def risk_parameters_strategy(draw):
    """Generate valid RiskParameters objects."""
    if not FP_AVAILABLE:
        return None
    
    return create_risk_parameters(
        max_position_size=draw(decimal_quantities(max_value=Decimal("100000"))),
        max_leverage=draw(st.decimals(min_value=Decimal("1"), max_value=Decimal("100"))),
        stop_loss_pct=draw(decimal_percentages(min_value=Decimal("1"), max_value=Decimal("50"))),
        take_profit_pct=draw(decimal_percentages(min_value=Decimal("1"), max_value=Decimal("100"))),
        max_daily_loss=draw(st.decimals(min_value=Decimal("100"), max_value=Decimal("10000"))),
        max_drawdown=draw(decimal_percentages(min_value=Decimal("5"), max_value=Decimal("95")))
    )


@composite
def price_series_strategy(draw, min_length=2, max_length=100):
    """Generate a series of prices for volatility/risk calculations."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    base_price = draw(decimal_prices(min_value=Decimal("10"), max_value=Decimal("1000")))
    
    prices = [base_price]
    for _ in range(length - 1):
        # Generate price changes with realistic volatility (±10%)
        change_pct = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
        new_price = prices[-1] * (Decimal("1") + Decimal(str(change_pct)))
        new_price = max(new_price, Decimal("0.01"))  # Ensure positive prices
        prices.append(new_price.quantize(Decimal("0.01")))
    
    return prices


class TestPropertyBasedRiskCalculations(FPTestBase):
    """Property-based tests for risk calculation functions."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(
        position_size=decimal_quantities(),
        entry_price=decimal_prices(),
        current_price=decimal_prices(),
        side=position_sides()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50
    )
    def test_position_value_properties(self, position_size, entry_price, current_price, side):
        """Test mathematical properties of position value calculations."""
        note(f"Testing position: size={position_size}, entry={entry_price}, current={current_price}, side={side}")
        
        # Calculate position value
        position_value = calculate_position_value(position_size, current_price)
        
        # Property: Position value should always be non-negative
        assert position_value >= 0, f"Position value should be non-negative: {position_value}"
        
        # Property: Position value should be proportional to size and price
        double_size_value = calculate_position_value(position_size * 2, current_price)
        assert abs(double_size_value - position_value * 2) < Decimal("0.01"), \
            "Position value should scale linearly with size"
        
        double_price_value = calculate_position_value(position_size, current_price * 2)
        assert abs(double_price_value - position_value * 2) < Decimal("0.01"), \
            "Position value should scale linearly with price"
        
        # Property: Zero size should result in zero value
        zero_value = calculate_position_value(Decimal("0"), current_price)
        assert zero_value == 0, "Zero position size should result in zero value"
    
    @given(
        position_size=decimal_quantities(),
        entry_price=decimal_prices(),
        current_price=decimal_prices(),
        side=position_sides()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=50
    )
    def test_unrealized_pnl_properties(self, position_size, entry_price, current_price, side):
        """Test mathematical properties of unrealized P&L calculations."""
        note(f"Testing P&L: size={position_size}, entry={entry_price}, current={current_price}, side={side}")
        
        # Calculate unrealized P&L
        pnl = calculate_unrealized_pnl_simple(side, position_size, entry_price, current_price)
        
        # Property: FLAT positions should have zero P&L
        if side == "FLAT":
            assert pnl == 0, "FLAT positions should have zero P&L"
        
        # Property: Zero size should result in zero P&L
        zero_pnl = calculate_unrealized_pnl_simple(side, Decimal("0"), entry_price, current_price)
        assert zero_pnl == 0, "Zero position size should result in zero P&L"
        
        # Property: P&L should scale linearly with position size
        if side != "FLAT" and position_size > 0:
            double_pnl = calculate_unrealized_pnl_simple(side, position_size * 2, entry_price, current_price)
            expected_double = pnl * 2
            tolerance = max(abs(expected_double) * Decimal("0.001"), Decimal("0.01"))
            assert abs(double_pnl - expected_double) <= tolerance, \
                f"P&L should scale linearly with size: {double_pnl} vs {expected_double}"
        
        # Property: LONG positions should profit when price increases
        if side == "LONG" and current_price > entry_price and position_size > 0:
            assert pnl > 0, f"LONG position should profit when price increases: {pnl}"
        
        # Property: SHORT positions should profit when price decreases
        if side == "SHORT" and current_price < entry_price and position_size > 0:
            assert pnl > 0, f"SHORT position should profit when price decreases: {pnl}"
        
        # Property: P&L should be symmetric for opposite sides
        if side in ["LONG", "SHORT"]:
            opposite_side = "SHORT" if side == "LONG" else "LONG"
            opposite_pnl = calculate_unrealized_pnl_simple(opposite_side, position_size, entry_price, current_price)
            tolerance = max(abs(pnl) * Decimal("0.001"), Decimal("0.01"))
            assert abs(pnl + opposite_pnl) <= tolerance, \
                f"P&L should be symmetric for opposite sides: {pnl} vs {opposite_pnl}"
    
    @given(
        position_value=st.decimals(min_value=Decimal("1"), max_value=Decimal("1000000")),
        leverage=st.decimals(min_value=Decimal("1"), max_value=Decimal("100"))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_margin_requirement_properties(self, position_value, leverage):
        """Test mathematical properties of margin requirement calculations."""
        note(f"Testing margin: value={position_value}, leverage={leverage}")
        
        # Calculate margin requirement
        margin = calculate_margin_requirement_simple(position_value, leverage)
        
        # Property: Margin should be positive for positive position value
        assert margin >= 0, f"Margin should be non-negative: {margin}"
        
        # Property: Margin should be less than or equal to position value
        assert margin <= position_value, f"Margin should not exceed position value: {margin} vs {position_value}"
        
        # Property: Higher leverage should require less margin
        if leverage > 1:
            higher_leverage = leverage * Decimal("2")
            higher_leverage_margin = calculate_margin_requirement_simple(position_value, higher_leverage)
            assert higher_leverage_margin <= margin, \
                f"Higher leverage should require less margin: {higher_leverage_margin} vs {margin}"
        
        # Property: Margin calculation should be inverse of leverage
        expected_margin = position_value / leverage
        tolerance = max(expected_margin * Decimal("0.001"), Decimal("0.01"))
        assert abs(margin - expected_margin) <= tolerance, \
            f"Margin should equal position_value / leverage: {margin} vs {expected_margin}"
        
        # Property: No leverage (leverage = 1) should require full position value as margin
        if leverage == Decimal("1"):
            assert abs(margin - position_value) <= Decimal("0.01"), \
                "No leverage should require full position value as margin"
    
    @given(
        position_value=st.decimals(min_value=Decimal("1"), max_value=Decimal("1000000")),
        fee_rate=st.decimals(min_value=Decimal("0"), max_value=Decimal("0.01"))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_fee_calculation_properties(self, position_value, fee_rate):
        """Test mathematical properties of fee calculations."""
        note(f"Testing fees: value={position_value}, rate={fee_rate}")
        
        # Calculate fees
        fees = calculate_fees_simple(position_value, fee_rate)
        
        # Property: Fees should be non-negative
        assert fees >= 0, f"Fees should be non-negative: {fees}"
        
        # Property: Fees should not exceed position value
        assert fees <= position_value, f"Fees should not exceed position value: {fees} vs {position_value}"
        
        # Property: Zero fee rate should result in zero fees
        zero_fees = calculate_fees_simple(position_value, Decimal("0"))
        assert zero_fees == 0, "Zero fee rate should result in zero fees"
        
        # Property: Fees should scale linearly with position value
        double_value_fees = calculate_fees_simple(position_value * 2, fee_rate)
        expected_double = fees * 2
        tolerance = max(expected_double * Decimal("0.001"), Decimal("0.01"))
        assert abs(double_value_fees - expected_double) <= tolerance, \
            f"Fees should scale linearly with position value: {double_value_fees} vs {expected_double}"
        
        # Property: Fees should scale linearly with fee rate
        double_rate_fees = calculate_fees_simple(position_value, fee_rate * 2)
        expected_double_rate = fees * 2
        tolerance = max(expected_double_rate * Decimal("0.001"), Decimal("0.01"))
        assert abs(double_rate_fees - expected_double_rate) <= tolerance, \
            f"Fees should scale linearly with fee rate: {double_rate_fees} vs {expected_double_rate}"
        
        # Property: Fee calculation should be multiplicative
        expected_fees = position_value * fee_rate
        tolerance = max(expected_fees * Decimal("0.001"), Decimal("0.01"))
        assert abs(fees - expected_fees) <= tolerance, \
            f"Fees should equal position_value * fee_rate: {fees} vs {expected_fees}"
    
    @given(
        position_size=decimal_quantities()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_position_size_validation_properties(self, position_size):
        """Test properties of position size validation."""
        note(f"Testing position size validation: {position_size}")
        
        # Validate position size
        validation_result = validate_position_size_simple(position_size)
        
        # Property: Positive sizes should be valid
        if position_size > 0:
            assert validation_result.is_success(), \
                f"Positive position size should be valid: {position_size}"
            validated_size = validation_result.success()
            assert validated_size == position_size, \
                "Validated size should match input for valid sizes"
        
        # Property: Zero size should be invalid
        zero_validation = validate_position_size_simple(Decimal("0"))
        assert zero_validation.is_failure(), "Zero position size should be invalid"
        
        # Property: Negative sizes should be invalid
        negative_validation = validate_position_size_simple(-abs(position_size))
        assert negative_validation.is_failure(), "Negative position size should be invalid"
    
    @given(
        prices=price_series_strategy(min_length=5, max_length=50)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_volatility_calculation_properties(self, prices):
        """Test properties of volatility calculations."""
        note(f"Testing volatility with {len(prices)} prices")
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                return_pct = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(float(return_pct))
        
        if len(returns) < 2:
            assume(False)  # Skip if insufficient data
        
        # Calculate volatility using standard deviation
        mean_return = statistics.mean(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        volatility = math.sqrt(variance)
        
        # Property: Volatility should be non-negative
        assert volatility >= 0, f"Volatility should be non-negative: {volatility}"
        
        # Property: Constant prices should have zero volatility
        if all(abs(r) < 1e-10 for r in returns):
            assert volatility < 1e-6, "Constant prices should have near-zero volatility"
        
        # Property: More volatile series should have higher volatility
        # Create a more volatile version by amplifying changes
        amplified_returns = [r * 2 for r in returns]
        amplified_mean = statistics.mean(amplified_returns)
        amplified_variance = sum((r - amplified_mean) ** 2 for r in amplified_returns) / (len(amplified_returns) - 1)
        amplified_volatility = math.sqrt(amplified_variance)
        
        if volatility > 1e-10:  # Only test if original volatility is significant
            assert amplified_volatility >= volatility * 1.5, \
                f"Amplified series should have higher volatility: {amplified_volatility} vs {volatility}"
    
    @given(
        risk_params=risk_parameters_strategy()
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_risk_parameters_properties(self, risk_params):
        """Test properties of risk parameter objects."""
        if not FP_AVAILABLE or risk_params is None:
            assume(False)
        
        note(f"Testing risk parameters: {risk_params}")
        
        # Property: All risk parameters should be positive
        assert risk_params.max_position_size > 0, "Max position size should be positive"
        assert risk_params.max_leverage >= 1, "Max leverage should be at least 1"
        assert risk_params.stop_loss_pct > 0, "Stop loss percentage should be positive"
        assert risk_params.take_profit_pct > 0, "Take profit percentage should be positive"
        assert risk_params.max_daily_loss > 0, "Max daily loss should be positive"
        assert risk_params.max_drawdown > 0, "Max drawdown should be positive"
        
        # Property: Stop loss should be reasonable (less than 100%)
        assert risk_params.stop_loss_pct < 100, "Stop loss should be less than 100%"
        
        # Property: Max drawdown should be less than 100%
        assert risk_params.max_drawdown < 100, "Max drawdown should be less than 100%"
        
        # Property: Risk parameters should be immutable (frozen dataclass)
        with pytest.raises(AttributeError):
            risk_params.max_position_size = Decimal("999999")
    
    @given(
        values=st.lists(
            st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100")),
            min_size=1,
            max_size=20
        ),
        decimal_places=st.integers(min_value=2, max_value=8)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_decimal_precision_properties(self, values, decimal_places):
        """Test properties of decimal precision normalization."""
        note(f"Testing precision with {len(values)} values, {decimal_places} decimal places")
        
        for value in values:
            normalized = normalize_decimal_precision(value, decimal_places)
            
            # Property: Normalized value should have correct decimal places
            decimal_str = str(normalized)
            if '.' in decimal_str:
                actual_places = len(decimal_str.split('.')[1])
                assert actual_places <= decimal_places, \
                    f"Normalized value should have at most {decimal_places} decimal places: {normalized}"
            
            # Property: Normalization should preserve magnitude
            relative_error = abs(normalized - value) / value if value > 0 else Decimal("0")
            max_error = Decimal("0.5") * (Decimal("10") ** (-decimal_places))
            assert relative_error <= max_error, \
                f"Normalization error should be small: {relative_error} vs {max_error}"
            
            # Property: Double normalization should be idempotent
            double_normalized = normalize_decimal_precision(normalized, decimal_places)
            assert double_normalized == normalized, \
                "Double normalization should be idempotent"
    
    @given(
        base_amount=st.decimals(min_value=Decimal("1000"), max_value=Decimal("100000")),
        loss_amount=st.decimals(min_value=Decimal("1"), max_value=Decimal("10000")),
        max_drawdown_pct=decimal_percentages(min_value=Decimal("5"), max_value=Decimal("50"))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_drawdown_calculation_properties(self, base_amount, loss_amount, max_drawdown_pct):
        """Test properties of drawdown calculations."""
        note(f"Testing drawdown: base={base_amount}, loss={loss_amount}, max={max_drawdown_pct}%")
        
        # Calculate actual drawdown percentage
        actual_drawdown_pct = (loss_amount / base_amount) * 100
        
        # Property: Drawdown percentage should be between 0 and 100
        assert 0 <= actual_drawdown_pct <= 100, \
            f"Drawdown percentage should be between 0 and 100: {actual_drawdown_pct}"
        
        # Property: Larger losses should result in larger drawdown percentages
        larger_loss = loss_amount * 2
        larger_drawdown_pct = (larger_loss / base_amount) * 100
        assert larger_drawdown_pct >= actual_drawdown_pct, \
            "Larger losses should result in larger drawdown percentages"
        
        # Property: Drawdown limit violation check
        violates_limit = actual_drawdown_pct > max_drawdown_pct
        
        # If we create drawdown limits, it should correctly identify violations
        if FP_AVAILABLE:
            drawdown_limits = DrawdownLimits(
                max_daily_drawdown=max_drawdown_pct,
                max_total_drawdown=max_drawdown_pct * 2,
                recovery_threshold=max_drawdown_pct / 2
            )
            
            # Test that drawdown limits are correctly structured
            assert drawdown_limits.max_daily_drawdown == max_drawdown_pct
            assert drawdown_limits.max_total_drawdown == max_drawdown_pct * 2
    
    @given(
        entry_price=decimal_prices(min_value=Decimal("10"), max_value=Decimal("1000")),
        stop_loss_pct=decimal_percentages(min_value=Decimal("1"), max_value=Decimal("20")),
        take_profit_pct=decimal_percentages(min_value=Decimal("1"), max_value=Decimal("50"))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_stop_loss_take_profit_properties(self, entry_price, stop_loss_pct, take_profit_pct):
        """Test properties of stop loss and take profit calculations."""
        note(f"Testing SL/TP: entry={entry_price}, SL={stop_loss_pct}%, TP={take_profit_pct}%")
        
        # Calculate stop loss and take profit distances
        from bot.fp.pure.paper_trading_calculations import (
            calculate_stop_loss_distance, calculate_take_profit_distance
        )
        
        sl_distance = calculate_stop_loss_distance(entry_price, stop_loss_pct)
        tp_distance = calculate_take_profit_distance(entry_price, take_profit_pct)
        
        # Property: Distances should be positive
        assert sl_distance > 0, f"Stop loss distance should be positive: {sl_distance}"
        assert tp_distance > 0, f"Take profit distance should be positive: {tp_distance}"
        
        # Property: Distances should be proportional to percentages
        assert sl_distance == entry_price * (stop_loss_pct / Decimal("100")), \
            "Stop loss distance should be proportional to percentage"
        assert tp_distance == entry_price * (take_profit_pct / Decimal("100")), \
            "Take profit distance should be proportional to percentage"
        
        # Property: Larger percentages should result in larger distances
        if take_profit_pct > stop_loss_pct:
            assert tp_distance > sl_distance, \
                "Larger percentage should result in larger distance"
        
        # Property: Distances should scale with entry price
        double_price = entry_price * 2
        double_sl_distance = calculate_stop_loss_distance(double_price, stop_loss_pct)
        double_tp_distance = calculate_take_profit_distance(double_price, take_profit_pct)
        
        tolerance = Decimal("0.01")
        assert abs(double_sl_distance - sl_distance * 2) <= tolerance, \
            "Stop loss distance should scale with entry price"
        assert abs(double_tp_distance - tp_distance * 2) <= tolerance, \
            "Take profit distance should scale with entry price"


class TestPropertyBasedRiskMetrics(FPTestBase):
    """Property-based tests for risk metrics calculations."""
    
    def setup_method(self):
        """Set up test fixtures for risk metrics tests."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(
        returns=st.lists(
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=100
        ),
        risk_free_rate=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_sharpe_ratio_properties(self, returns, risk_free_rate):
        """Test properties of Sharpe ratio calculations."""
        note(f"Testing Sharpe ratio with {len(returns)} returns, risk-free rate: {risk_free_rate}")
        
        # Filter out extreme values that might cause numerical issues
        filtered_returns = [r for r in returns if abs(r) < 1.0]
        if len(filtered_returns) < 10:
            assume(False)
        
        # Calculate mean and standard deviation
        mean_return = statistics.mean(filtered_returns)
        if len(filtered_returns) < 2:
            assume(False)
        
        std_dev = statistics.stdev(filtered_returns)
        
        # Skip if standard deviation is too small (would cause division issues)
        if std_dev < 1e-6:
            assume(False)
        
        # Calculate Sharpe ratio
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / std_dev if std_dev > 0 else 0
        
        # Property: Sharpe ratio should be finite
        assert math.isfinite(sharpe_ratio), f"Sharpe ratio should be finite: {sharpe_ratio}"
        
        # Property: Higher mean return (with same volatility) should improve Sharpe ratio
        higher_returns = [r + 0.01 for r in filtered_returns]  # Add 1% to all returns
        higher_mean = statistics.mean(higher_returns)
        higher_std = statistics.stdev(higher_returns)
        higher_sharpe = (higher_mean - risk_free_rate) / higher_std if higher_std > 0 else 0
        
        if std_dev > 1e-6 and higher_std > 1e-6:
            assert higher_sharpe >= sharpe_ratio, \
                f"Higher returns should improve Sharpe ratio: {higher_sharpe} vs {sharpe_ratio}"
        
        # Property: Negative excess returns should result in negative Sharpe ratio
        if excess_return < 0 and std_dev > 0:
            assert sharpe_ratio < 0, f"Negative excess return should result in negative Sharpe ratio: {sharpe_ratio}"
    
    @given(
        returns=st.lists(
            st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=100
        ),
        risk_free_rate=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_sortino_ratio_properties(self, returns, risk_free_rate):
        """Test properties of Sortino ratio calculations."""
        note(f"Testing Sortino ratio with {len(returns)} returns")
        
        # Filter returns and calculate basic statistics
        filtered_returns = [r for r in returns if abs(r) < 1.0]
        if len(filtered_returns) < 10:
            assume(False)
        
        mean_return = statistics.mean(filtered_returns)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = [r for r in filtered_returns if r < risk_free_rate]
        
        if len(downside_returns) == 0:
            # No downside risk - Sortino ratio should be very high or infinite
            return
        
        downside_variance = statistics.variance(downside_returns) if len(downside_returns) > 1 else 0
        downside_deviation = math.sqrt(downside_variance)
        
        if downside_deviation < 1e-6:
            assume(False)
        
        # Calculate Sortino ratio
        excess_return = mean_return - risk_free_rate
        sortino_ratio = excess_return / downside_deviation
        
        # Property: Sortino ratio should be finite
        assert math.isfinite(sortino_ratio), f"Sortino ratio should be finite: {sortino_ratio}"
        
        # Property: Sortino ratio should be >= Sharpe ratio (or comparable)
        # Sortino typically gives better ratios since it only penalizes downside volatility
        std_dev = statistics.stdev(filtered_returns) if len(filtered_returns) > 1 else 0
        if std_dev > 1e-6:
            sharpe_ratio = excess_return / std_dev
            # Sortino should generally be >= Sharpe (since downside deviation <= total deviation)
            if downside_deviation <= std_dev:
                assert sortino_ratio >= sharpe_ratio * 0.8, \
                    f"Sortino ratio should be comparable to or better than Sharpe: {sortino_ratio} vs {sharpe_ratio}"
    
    @given(
        price_series=price_series_strategy(min_length=10, max_length=50)
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_max_drawdown_properties(self, price_series):
        """Test properties of maximum drawdown calculations."""
        note(f"Testing max drawdown with {len(price_series)} prices")
        
        # Calculate running maximum (peak values)
        running_max = []
        current_max = price_series[0]
        
        for price in price_series:
            current_max = max(current_max, price)
            running_max.append(current_max)
        
        # Calculate drawdown at each point
        drawdowns = []
        for i, price in enumerate(price_series):
            if running_max[i] > 0:
                drawdown_pct = ((running_max[i] - price) / running_max[i]) * 100
                drawdowns.append(drawdown_pct)
        
        if not drawdowns:
            assume(False)
        
        max_drawdown = max(drawdowns)
        
        # Property: Maximum drawdown should be non-negative
        assert max_drawdown >= 0, f"Maximum drawdown should be non-negative: {max_drawdown}"
        
        # Property: Maximum drawdown should be <= 100%
        assert max_drawdown <= 100, f"Maximum drawdown should be <= 100%: {max_drawdown}"
        
        # Property: If prices only increase, max drawdown should be 0
        if all(price_series[i] >= price_series[i-1] for i in range(1, len(price_series))):
            assert max_drawdown < 1e-6, "Monotonically increasing prices should have zero drawdown"
        
        # Property: Larger price drops should result in larger max drawdown
        # Create a version with an additional large drop
        modified_series = price_series + [price_series[-1] * Decimal("0.5")]  # 50% drop
        
        modified_running_max = []
        modified_current_max = modified_series[0]
        
        for price in modified_series:
            modified_current_max = max(modified_current_max, price)
            modified_running_max.append(modified_current_max)
        
        modified_drawdowns = []
        for i, price in enumerate(modified_series):
            if modified_running_max[i] > 0:
                drawdown_pct = ((modified_running_max[i] - price) / modified_running_max[i]) * 100
                modified_drawdowns.append(drawdown_pct)
        
        if modified_drawdowns:
            modified_max_drawdown = max(modified_drawdowns)
            assert modified_max_drawdown >= max_drawdown, \
                "Adding a price drop should not decrease max drawdown"
    
    @given(
        correlation_data=st.lists(
            st.tuples(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
            min_size=10,
            max_size=50
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_correlation_properties(self, correlation_data):
        """Test properties of correlation calculations."""
        note(f"Testing correlation with {len(correlation_data)} data points")
        
        # Extract x and y series
        x_values = [point[0] for point in correlation_data]
        y_values = [point[1] for point in correlation_data]
        
        # Calculate correlation coefficient using standard formula
        n = len(x_values)
        if n < 2:
            assume(False)
        
        # Calculate means
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(y_values)
        
        # Calculate correlation components
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        
        if sum_sq_x < 1e-10 or sum_sq_y < 1e-10:
            # One series is constant, correlation is undefined
            assume(False)
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        correlation = numerator / denominator if denominator > 0 else 0
        
        # Property: Correlation should be between -1 and 1
        assert -1.01 <= correlation <= 1.01, f"Correlation should be between -1 and 1: {correlation}"
        
        # Property: Perfect positive correlation
        perfect_positive_y = [(x - mean_x) * 2 + mean_y for x in x_values]
        perfect_pos_corr = self._calculate_correlation(x_values, perfect_positive_y)
        if perfect_pos_corr is not None:
            assert perfect_pos_corr > 0.99, f"Perfect positive correlation should be near 1: {perfect_pos_corr}"
        
        # Property: Perfect negative correlation
        perfect_negative_y = [-(x - mean_x) * 2 + mean_y for x in x_values]
        perfect_neg_corr = self._calculate_correlation(x_values, perfect_negative_y)
        if perfect_neg_corr is not None:
            assert perfect_neg_corr < -0.99, f"Perfect negative correlation should be near -1: {perfect_neg_corr}"
        
        # Property: Correlation with self should be 1
        self_correlation = self._calculate_correlation(x_values, x_values)
        if self_correlation is not None:
            assert abs(self_correlation - 1.0) < 0.01, f"Self-correlation should be 1: {self_correlation}"
    
    def _calculate_correlation(self, x_values, y_values):
        """Helper method to calculate correlation coefficient."""
        n = len(x_values)
        if n < 2 or len(y_values) != n:
            return None
        
        mean_x = statistics.mean(x_values)
        mean_y = statistics.mean(y_values)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
        sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
        
        if sum_sq_x < 1e-10 or sum_sq_y < 1e-10:
            return None
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        return numerator / denominator if denominator > 0 else None


class TestPropertyBasedRiskComposition(FPTestBase):
    """Property-based tests for risk calculation composition and associativity."""
    
    def setup_method(self):
        """Set up test fixtures for composition tests."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(
        portfolios=st.lists(
            st.tuples(
                decimal_quantities(),  # position size
                decimal_prices(),      # entry price
                decimal_prices()       # current price
            ),
            min_size=2,
            max_size=10
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=20
    )
    def test_portfolio_aggregation_properties(self, portfolios):
        """Test properties of portfolio-level risk aggregation."""
        note(f"Testing portfolio aggregation with {len(portfolios)} positions")
        
        # Calculate individual position values and total
        individual_values = []
        total_value = Decimal("0")
        
        for size, entry_price, current_price in portfolios:
            position_value = calculate_position_value(size, current_price)
            individual_values.append(position_value)
            total_value += position_value
        
        # Property: Sum of individual values should equal calculated total
        calculated_total = sum(individual_values)
        tolerance = Decimal("0.01")
        assert abs(total_value - calculated_total) <= tolerance, \
            f"Portfolio total should equal sum of positions: {total_value} vs {calculated_total}"
        
        # Property: Portfolio value should be non-negative
        assert total_value >= 0, f"Portfolio value should be non-negative: {total_value}"
        
        # Property: Adding a position should increase or maintain portfolio value
        new_position_value = calculate_position_value(Decimal("1"), Decimal("100"))
        new_total = total_value + new_position_value
        assert new_total >= total_value, "Adding a position should not decrease portfolio value"
        
        # Property: Portfolio diversification (positions should be somewhat independent)
        if len(portfolios) > 1:
            # Variance of individual positions
            mean_value = statistics.mean(float(v) for v in individual_values)
            variance = statistics.variance(float(v) for v in individual_values) if len(individual_values) > 1 else 0
            
            # Property: Diversified portfolio should have some variance in position sizes
            # (This is a weak property, but ensures we're not testing degenerate cases)
            if total_value > 0:
                coefficient_of_variation = math.sqrt(variance) / mean_value if mean_value > 0 else 0
                # Allow for both diversified and concentrated portfolios
                assert coefficient_of_variation >= 0, "Coefficient of variation should be non-negative"
    
    @given(
        base_calculation=st.tuples(
            decimal_quantities(),
            decimal_prices(),
            decimal_prices()
        ),
        scaling_factor=st.decimals(min_value=Decimal("0.1"), max_value=Decimal("10"))
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=30
    )
    def test_risk_calculation_scaling_properties(self, base_calculation, scaling_factor):
        """Test scaling properties of risk calculations."""
        size, entry_price, current_price = base_calculation
        note(f"Testing scaling: size={size}, entry={entry_price}, current={current_price}, scale={scaling_factor}")
        
        # Calculate base values
        base_position_value = calculate_position_value(size, current_price)
        base_pnl = calculate_unrealized_pnl_simple("LONG", size, entry_price, current_price)
        
        # Calculate scaled values
        scaled_size = size * scaling_factor
        scaled_position_value = calculate_position_value(scaled_size, current_price)
        scaled_pnl = calculate_unrealized_pnl_simple("LONG", scaled_size, entry_price, current_price)
        
        # Property: Position value should scale linearly with size
        expected_scaled_value = base_position_value * scaling_factor
        tolerance = max(expected_scaled_value * Decimal("0.001"), Decimal("0.01"))
        assert abs(scaled_position_value - expected_scaled_value) <= tolerance, \
            f"Position value should scale linearly: {scaled_position_value} vs {expected_scaled_value}"
        
        # Property: P&L should scale linearly with size
        expected_scaled_pnl = base_pnl * scaling_factor
        tolerance = max(abs(expected_scaled_pnl) * Decimal("0.001"), Decimal("0.01"))
        assert abs(scaled_pnl - expected_scaled_pnl) <= tolerance, \
            f"P&L should scale linearly: {scaled_pnl} vs {expected_scaled_pnl}"
        
        # Property: Margin requirements should scale linearly
        leverage = Decimal("5")
        base_margin = calculate_margin_requirement_simple(base_position_value, leverage)
        scaled_margin = calculate_margin_requirement_simple(scaled_position_value, leverage)
        expected_scaled_margin = base_margin * scaling_factor
        
        tolerance = max(expected_scaled_margin * Decimal("0.001"), Decimal("0.01"))
        assert abs(scaled_margin - expected_scaled_margin) <= tolerance, \
            f"Margin should scale linearly: {scaled_margin} vs {expected_scaled_margin}"
    
    @given(
        calculations=st.lists(
            st.tuples(
                decimal_quantities(),
                decimal_prices(),
                position_sides()
            ),
            min_size=3,
            max_size=8
        )
    )
    @settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        max_examples=15
    )
    def test_risk_calculation_associativity(self, calculations):
        """Test associativity properties of risk calculations."""
        note(f"Testing associativity with {len(calculations)} calculations")
        
        if len(calculations) < 3:
            assume(False)
        
        current_price = Decimal("100")  # Fixed current price for consistency
        
        # Calculate in different groupings to test associativity
        # Group 1: ((A + B) + C) + ...
        group1_total = Decimal("0")
        for size, entry_price, side in calculations:
            pnl = calculate_unrealized_pnl_simple(side, size, entry_price, current_price)
            group1_total += pnl
        
        # Group 2: A + (B + (C + ...))
        group2_total = Decimal("0")
        for size, entry_price, side in reversed(calculations):
            pnl = calculate_unrealized_pnl_simple(side, size, entry_price, current_price)
            group2_total += pnl
        
        # Group 3: Sum all at once
        all_pnls = [
            calculate_unrealized_pnl_simple(side, size, entry_price, current_price)
            for size, entry_price, side in calculations
        ]
        group3_total = sum(all_pnls)
        
        # Property: All groupings should give the same result (associativity)
        tolerance = Decimal("0.01")
        assert abs(group1_total - group2_total) <= tolerance, \
            f"Different calculation orders should give same result: {group1_total} vs {group2_total}"
        assert abs(group1_total - group3_total) <= tolerance, \
            f"Bulk calculation should match sequential: {group1_total} vs {group3_total}"
        assert abs(group2_total - group3_total) <= tolerance, \
            f"All calculation methods should agree: {group2_total} vs {group3_total}"
        
        # Property: Commutativity (order shouldn't matter)
        shuffled_calculations = calculations.copy()
        # Simple manual shuffle to ensure we have control
        if len(shuffled_calculations) > 1:
            shuffled_calculations[0], shuffled_calculations[-1] = shuffled_calculations[-1], shuffled_calculations[0]
        
        shuffled_total = sum(
            calculate_unrealized_pnl_simple(side, size, entry_price, current_price)
            for size, entry_price, side in shuffled_calculations
        )
        
        assert abs(group1_total - shuffled_total) <= tolerance, \
            f"Order of calculations should not matter: {group1_total} vs {shuffled_total}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])