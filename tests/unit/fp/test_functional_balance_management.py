"""
Comprehensive Functional Balance Management Tests

This module tests the functional programming balance management system including:
- Immutable balance validation types and pure validation functions
- Functional paper trading account state with decimal precision
- Pure calculation functions for balance operations
- Balance-related type safety and error handling
- Performance characteristics of FP balance operations
- Compatibility between FP and legacy balance systems

Tests maintain all safety validations while validating functional programming patterns.
"""

import time
from datetime import datetime
from decimal import Decimal, getcontext

import pytest

# FP test infrastructure
from tests.fp_test_base import FP_AVAILABLE, FPTestBase

if FP_AVAILABLE:
    from bot.fp.pure.paper_trading_calculations import (
        calculate_account_metrics,
        calculate_fees_simple,
        calculate_margin_requirement_simple,
        calculate_position_size,
        # Additional pure functions
        calculate_position_value,
        calculate_stop_loss_distance,
        calculate_take_profit_distance,
        calculate_unrealized_pnl,
        calculate_unrealized_pnl_simple,
        normalize_decimal_precision,
        simulate_position_close,
        simulate_trade_execution,
        validate_position_size_simple,
    )
    from bot.fp.types.balance_validation import (
        BalanceRange,
        BalanceValidationType,
        MarginRequirement,
        TradeAffordabilityCheck,
        # Factory functions
        create_default_balance_config,
        create_margin_requirement,
        create_trade_affordability_check,
        perform_comprehensive_balance_validation,
        # Pure validation functions
        validate_balance_range,
        validate_leverage_compliance,
        validate_margin_requirements,
        validate_trade_affordability,
    )
    from bot.fp.types.paper_trading import (
        PaperTradingAccountState,
        apply_slippage,
        calculate_trade_fees,
        # Helper functions
        create_paper_trade,
    )
else:
    # Fallback for non-FP environments - create minimal stubs
    class BalanceValidationType:
        RANGE_CHECK = "range_check"

    def validate_balance_range(*args, **kwargs):
        return None


# Set high precision for financial calculations
getcontext().prec = 28


class TestBalanceValidationTypes(FPTestBase):
    """Test immutable balance validation types."""

    def test_balance_range_creation_and_validation(self):
        """Test BalanceRange creation and validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Valid balance range
        balance_range = BalanceRange(
            minimum=Decimal(100), maximum=Decimal(1000000), currency="USD"
        )

        assert balance_range.minimum == Decimal(100)
        assert balance_range.maximum == Decimal(1000000)
        assert balance_range.currency == "USD"

        # Test contains method
        assert balance_range.contains(Decimal(5000)) is True
        assert balance_range.contains(Decimal(50)) is False
        assert balance_range.contains(Decimal(2000000)) is False

        # Test distance calculations
        assert balance_range.distance_from_range(Decimal(5000)) == Decimal(0)
        assert balance_range.distance_from_range(Decimal(50)) == Decimal(50)  # 100 - 50
        assert balance_range.distance_from_range(Decimal(1500000)) == Decimal(
            500000
        )  # 1500000 - 1000000

        # Test invalid range creation
        with pytest.raises(ValueError, match="Minimum balance cannot be negative"):
            BalanceRange(minimum=Decimal(-100), maximum=Decimal(1000))

        with pytest.raises(
            ValueError, match="Maximum balance .* cannot be less than minimum"
        ):
            BalanceRange(minimum=Decimal(1000), maximum=Decimal(500))

    def test_margin_requirement_calculations(self):
        """Test MarginRequirement calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        margin_req = MarginRequirement(
            position_value=Decimal(10000),
            leverage=Decimal(5),
            maintenance_margin_pct=Decimal("0.05"),
            initial_margin_pct=Decimal("0.10"),
        )

        # Test property calculations
        assert margin_req.required_margin == Decimal(2000)  # 10000 / 5
        assert margin_req.maintenance_margin == Decimal(500)  # 10000 * 0.05
        assert margin_req.initial_margin == Decimal(1000)  # 10000 * 0.10

        # Test validation
        with pytest.raises(ValueError, match="Position value cannot be negative"):
            MarginRequirement(position_value=Decimal(-1000), leverage=Decimal(5))

        with pytest.raises(ValueError, match="Leverage must be positive"):
            MarginRequirement(position_value=Decimal(1000), leverage=Decimal(0))

    def test_trade_affordability_check_calculations(self):
        """Test TradeAffordabilityCheck calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        affordability = TradeAffordabilityCheck(
            trade_value=Decimal(10000),
            estimated_fees=Decimal(10),
            required_margin=Decimal(2000),
            leverage=Decimal(5),
            current_balance=Decimal(5000),
            existing_margin_used=Decimal(1000),
        )

        # Test property calculations
        assert affordability.total_required_capital == Decimal(2010)  # 2000 + 10
        assert affordability.total_margin_after_trade == Decimal(3000)  # 1000 + 2000
        assert affordability.available_balance_after_trade == Decimal(
            2990
        )  # 5000 - 2010
        assert affordability.margin_utilization_pct == 60.0  # 3000 / 5000 * 100

    def test_balance_validation_config_default(self):
        """Test default balance validation configuration."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        config = create_default_balance_config()

        assert config.min_balance == Decimal(100)
        assert config.max_balance == Decimal(10000000)
        assert config.max_margin_utilization_pct == 80.0
        assert config.min_free_balance_pct == 10.0
        assert config.emergency_threshold_pct == 5.0

        # Test balance range property
        balance_range = config.balance_range
        assert balance_range.minimum == config.min_balance
        assert balance_range.maximum == config.max_balance


class TestPureBalanceValidationFunctions(FPTestBase):
    """Test pure balance validation functions."""

    def test_validate_balance_range_success(self):
        """Test successful balance range validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        balance_range = BalanceRange(minimum=Decimal(100), maximum=Decimal(100000))

        result = validate_balance_range(
            balance=Decimal(5000),
            balance_range=balance_range,
            operation_context="test_operation",
        )

        assert result.is_valid is True
        assert result.validation_type == BalanceValidationType.RANGE_CHECK
        assert result.balance == Decimal(5000)
        assert "within valid range" in result.message
        assert result.error is None
        assert len(result.warnings) == 0

    def test_validate_balance_range_failures(self):
        """Test balance range validation failures."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        balance_range = BalanceRange(minimum=Decimal(100), maximum=Decimal(100000))

        # Test below minimum
        result = validate_balance_range(Decimal(50), balance_range)
        assert result.is_valid is False
        assert result.error.severity == "high"
        assert "below minimum" in result.error.message

        # Test zero balance (critical)
        result = validate_balance_range(Decimal(0), balance_range)
        assert result.is_valid is False
        assert result.error.severity == "critical"

        # Test above maximum
        result = validate_balance_range(Decimal(200000), balance_range)
        assert result.is_valid is False
        assert result.error.severity == "medium"
        assert "exceeds maximum" in result.error.message

    def test_validate_margin_requirements_success(self):
        """Test successful margin requirement validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        margin_req = MarginRequirement(
            position_value=Decimal(10000), leverage=Decimal(5)
        )

        result = validate_margin_requirements(
            balance=Decimal(5000),
            margin_requirement=margin_req,
            used_margin=Decimal(1000),
        )

        assert result.is_valid is True
        assert result.validation_type == BalanceValidationType.MARGIN_VALIDATION
        assert "can be met" in result.message

        # Check metadata
        assert "required_margin" in result.metadata
        assert "available_balance" in result.metadata
        assert "margin_utilization_pct" in result.metadata

    def test_validate_margin_requirements_high_utilization_warning(self):
        """Test margin validation with high utilization warning."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        margin_req = MarginRequirement(
            position_value=Decimal(10000),
            leverage=Decimal(2),  # Higher margin requirement
        )

        result = validate_margin_requirements(
            balance=Decimal(6000),
            margin_requirement=margin_req,
            used_margin=Decimal(500),
        )

        assert result.is_valid is True
        assert result.has_warnings is True
        assert any("High margin utilization" in warning for warning in result.warnings)

    def test_validate_margin_requirements_insufficient(self):
        """Test margin validation with insufficient funds."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        margin_req = MarginRequirement(
            position_value=Decimal(10000),
            leverage=Decimal(2),  # Requires 5000 margin
        )

        result = validate_margin_requirements(
            balance=Decimal(3000),
            margin_requirement=margin_req,
            used_margin=Decimal(1000),  # Only 2000 available
        )

        assert result.is_valid is False
        assert result.error.severity == "high"
        assert "Insufficient margin" in result.error.message
        assert "shortage" in result.error.additional_context

    def test_validate_trade_affordability_success(self):
        """Test successful trade affordability validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        affordability = create_trade_affordability_check(
            trade_value=Decimal(10000),
            estimated_fees=Decimal(10),
            leverage=Decimal(5),
            current_balance=Decimal(5000),
            existing_margin_used=Decimal(500),
        )

        result = validate_trade_affordability(affordability)

        assert result.is_valid is True
        assert result.validation_type == BalanceValidationType.TRADE_AFFORDABILITY
        assert "Trade is affordable" in result.message

    def test_validate_trade_affordability_insufficient_funds(self):
        """Test trade affordability with insufficient funds."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        affordability = create_trade_affordability_check(
            trade_value=Decimal(50000),  # Large trade
            estimated_fees=Decimal(50),
            leverage=Decimal(2),  # High margin requirement
            current_balance=Decimal(5000),  # Insufficient balance
            existing_margin_used=Decimal(1000),
        )

        result = validate_trade_affordability(affordability)

        assert result.is_valid is False
        assert result.error.severity == "critical"
        assert "Cannot afford trade" in result.error.message

    def test_validate_leverage_compliance_success(self):
        """Test successful leverage compliance validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        result = validate_leverage_compliance(
            position_value=Decimal(10000),
            leverage=Decimal(5),
            max_leverage=Decimal(10),
            balance=Decimal(5000),
        )

        assert result.is_valid is True
        assert result.validation_type == BalanceValidationType.LEVERAGE_COMPLIANCE
        assert "within limits" in result.message

    def test_validate_leverage_compliance_excess(self):
        """Test leverage compliance with excess leverage."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        result = validate_leverage_compliance(
            position_value=Decimal(10000),
            leverage=Decimal(15),  # Exceeds max
            max_leverage=Decimal(10),
            balance=Decimal(5000),
        )

        assert result.is_valid is False
        assert result.error.severity == "high"  # Small excess
        assert "exceeds maximum" in result.error.message

    def test_comprehensive_balance_validation(self):
        """Test comprehensive balance validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        config = create_default_balance_config()
        margin_req = create_margin_requirement(
            position_value=Decimal(10000), leverage=Decimal(5)
        )
        affordability = create_trade_affordability_check(
            trade_value=Decimal(10000),
            estimated_fees=Decimal(10),
            leverage=Decimal(5),
            current_balance=Decimal(5000),
        )

        validation = perform_comprehensive_balance_validation(
            balance=Decimal(5000),
            config=config,
            margin_requirement=margin_req,
            affordability_check=affordability,
            leverage_check=(Decimal(10000), Decimal(5), Decimal(10)),
            used_margin=Decimal(1000),
        )

        assert validation.is_all_valid is True
        assert len(validation.critical_errors) == 0
        assert validation.range_validation.is_valid is True
        assert validation.margin_validation.is_valid is True
        assert validation.trade_affordability.is_valid is True
        assert validation.leverage_compliance.is_valid is True


class TestPaperTradingAccountState(FPTestBase):
    """Test immutable paper trading account state."""

    def test_account_state_creation_and_validation(self):
        """Test account state creation and validation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Valid account creation
        account = PaperTradingAccountState.create_initial(
            starting_balance=Decimal(10000), session_start_time=datetime.now()
        )

        assert account.starting_balance == Decimal(10000)
        assert account.current_balance == Decimal(10000)
        assert account.equity == Decimal(10000)
        assert account.margin_used == Decimal(0)
        assert len(account.open_trades) == 0
        assert len(account.closed_trades) == 0
        assert account.trade_counter == 0
        assert account.peak_equity == Decimal(10000)
        assert account.max_drawdown == Decimal(0)

        # Test validation errors
        with pytest.raises(ValueError, match="Starting balance must be positive"):
            PaperTradingAccountState.create_initial(Decimal(-1000))

    def test_account_state_immutability(self):
        """Test that account state is immutable."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Test immutability
        with pytest.raises(AttributeError):
            account.current_balance = Decimal(5000)  # Should fail - frozen dataclass

    def test_account_state_pure_calculations(self):
        """Test pure calculation methods on account state."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))
        updated_account = account.update_balance(Decimal(1000))

        # Original unchanged
        assert account.current_balance == Decimal(10000)

        # New state has updated balance
        assert updated_account.current_balance == Decimal(11000)

        # Test calculations
        assert updated_account.get_available_margin() == Decimal(11000)
        assert updated_account.get_total_pnl() == Decimal(1000)
        assert updated_account.get_roi_percent() == Decimal("10.0")
        assert updated_account.get_open_position_count() == 0
        assert updated_account.get_total_trade_count() == 0

    def test_account_state_trade_management(self):
        """Test trade management in account state."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Create and add trade
        trade = create_paper_trade(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
            fees=Decimal(5),
        )

        account_with_trade = account.add_trade(trade)

        # Original unchanged
        assert len(account.open_trades) == 0

        # New state has trade
        assert len(account_with_trade.open_trades) == 1
        assert account_with_trade.trade_counter == 1
        assert account_with_trade.open_trades[0] == trade

        # Test finding trade
        found_trade = account_with_trade.find_open_trade("BTC-USD")
        assert found_trade == trade

        # Test trade not found
        not_found = account_with_trade.find_open_trade("ETH-USD")
        assert not_found is None

    def test_account_state_equity_updates(self):
        """Test equity updates with current prices."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Add profitable trade
        trade = create_paper_trade(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
        )
        account = account.add_trade(trade)

        # Update equity with higher price
        current_prices = {"BTC-USD": Decimal(55000)}
        updated_account = account.update_equity(current_prices)

        # Calculate expected unrealized PnL: 0.1 * (55000 - 50000) = 500
        expected_equity = Decimal(10000) + Decimal(500)
        assert updated_account.equity == expected_equity
        assert updated_account.peak_equity == expected_equity
        assert updated_account.max_drawdown == Decimal(0)

        # Test with loss
        lower_prices = {"BTC-USD": Decimal(45000)}
        loss_account = updated_account.update_equity(lower_prices)

        # Unrealized PnL: 0.1 * (45000 - 50000) = -500
        expected_equity_loss = Decimal(10000) - Decimal(500)
        assert loss_account.equity == expected_equity_loss

        # Peak should remain the same, drawdown should be calculated
        assert loss_account.peak_equity == Decimal(10500)
        expected_drawdown = (
            (Decimal(10500) - expected_equity_loss) / Decimal(10500) * 100
        )
        assert loss_account.max_drawdown == expected_drawdown

    def test_trade_closing_and_balance_updates(self):
        """Test closing trades and balance updates."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Add trade with margin
        trade = create_paper_trade(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
        )
        account = account.add_trade(trade).update_margin(Decimal(1000))

        # Close trade with profit
        closed_account = account.close_trade(
            trade_id=trade.id,
            exit_price=Decimal(55000),
            exit_time=datetime.now(),
            additional_fees=Decimal(5),
        )

        # Check trade moved to closed
        assert len(closed_account.open_trades) == 0
        assert len(closed_account.closed_trades) == 1

        # Check P&L calculation: 0.1 * (55000 - 50000) - 5 = 495
        closed_trade = closed_account.closed_trades[0]
        assert closed_trade.realized_pnl == Decimal(495)
        assert closed_trade.status == "CLOSED"

        # Check balance update
        expected_balance = Decimal(10000) + Decimal(495)
        assert closed_account.current_balance == expected_balance

        # Check margin released
        assert closed_account.margin_used == Decimal(0)  # Should be released


class TestPaperTradingCalculations(FPTestBase):
    """Test pure paper trading calculation functions."""

    def test_calculate_position_size_spot_trading(self):
        """Test position size calculation for spot trading."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Standard position size calculation
        size = calculate_position_size(
            equity=Decimal(10000),
            size_percentage=Decimal(10),  # 10% of equity
            leverage=Decimal(1),  # No leverage for spot
            current_price=Decimal(50000),
            is_futures=False,
        )

        # Expected: 10000 * 0.1 * 1 / 50000 = 0.02
        expected_size = Decimal("0.02")
        assert size == expected_size

    def test_calculate_position_size_futures_trading(self):
        """Test position size calculation for futures trading."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Futures with contract size
        size = calculate_position_size(
            equity=Decimal(10000),
            size_percentage=Decimal(20),  # 20% of equity
            leverage=Decimal(5),  # 5x leverage
            current_price=Decimal(50000),
            is_futures=True,
            contract_size=Decimal("0.1"),  # 0.1 BTC per contract
            fixed_contracts=None,
        )

        # Expected calculation:
        # Position value: 10000 * 0.2 = 2000
        # Leveraged value: 2000 * 5 = 10000
        # Quantity in asset: 10000 / 50000 = 0.2
        # Number of contracts: int(0.2 / 0.1) = 2
        # Final size: 0.1 * 2 = 0.2
        expected_size = Decimal("0.2")
        assert size == expected_size

        # Test with fixed contracts
        fixed_size = calculate_position_size(
            equity=Decimal(10000),
            size_percentage=Decimal(20),
            leverage=Decimal(5),
            current_price=Decimal(50000),
            is_futures=True,
            contract_size=Decimal("0.1"),
            fixed_contracts=3,
        )

        # Expected: 0.1 * 3 = 0.3
        assert fixed_size == Decimal("0.3")

    def test_simulate_trade_execution_success(self):
        """Test successful trade execution simulation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        execution, new_account = simulate_trade_execution(
            account_state=account,
            symbol="BTC-USD",
            side="LONG",
            size_percentage=Decimal(10),
            current_price=Decimal(50000),
            leverage=Decimal(5),
            fee_rate=Decimal("0.001"),  # 0.1%
            slippage_rate=Decimal("0.0005"),  # 0.05%
        )

        assert execution.success is True
        assert execution.trade_state is not None
        assert new_account is not None

        # Check trade details
        trade = execution.trade_state
        assert trade.symbol == "BTC-USD"
        assert trade.side == "LONG"
        assert trade.size > Decimal(0)

        # Check slippage was applied (price should be slightly higher for LONG)
        assert execution.execution_price > Decimal(50000)
        assert execution.slippage_amount > Decimal(0)

        # Check account was updated
        assert len(new_account.open_trades) == 1
        assert new_account.margin_used > Decimal(0)
        assert new_account.current_balance < account.current_balance  # Fees deducted

    def test_simulate_trade_execution_insufficient_funds(self):
        """Test trade execution with insufficient funds."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(100))  # Small balance

        execution, new_account = simulate_trade_execution(
            account_state=account,
            symbol="BTC-USD",
            side="LONG",
            size_percentage=Decimal(100),  # Try to use all equity
            current_price=Decimal(50000),
            leverage=Decimal(1),  # No leverage
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )

        assert execution.success is False
        assert "Insufficient funds" in execution.reason
        assert new_account is None

    def test_simulate_position_close_success(self):
        """Test successful position closing simulation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Create account with open position
        account = PaperTradingAccountState.create_initial(Decimal(10000))
        trade = create_paper_trade(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
        )
        account = account.add_trade(trade).update_margin(Decimal(1000))

        # Close position with profit
        execution, new_account = simulate_position_close(
            account_state=account,
            symbol="BTC-USD",
            exit_price=Decimal(55000),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )

        assert execution.success is True
        assert execution.trade_state.status == "CLOSED"
        assert new_account is not None

        # Check position was closed
        assert len(new_account.open_trades) == 0
        assert len(new_account.closed_trades) == 1

        # Check P&L was realized
        closed_trade = new_account.closed_trades[0]
        assert closed_trade.realized_pnl > Decimal(0)  # Should be profitable

    def test_calculate_unrealized_pnl_multiple_positions(self):
        """Test unrealized P&L calculation for multiple positions."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Add multiple trades
        btc_trade = create_paper_trade(
            "BTC-USD", "LONG", Decimal("0.1"), Decimal(50000)
        )
        eth_trade = create_paper_trade(
            "ETH-USD", "SHORT", Decimal("1.0"), Decimal(3000)
        )

        account = account.add_trade(btc_trade).add_trade(eth_trade)

        # Calculate unrealized P&L
        current_prices = {
            "BTC-USD": Decimal(55000),  # +5000 profit
            "ETH-USD": Decimal(2800),  # +200 profit for short
        }

        unrealized_pnl = calculate_unrealized_pnl(account, current_prices)

        # Expected: BTC: 0.1 * (55000 - 50000) = 500
        #          ETH: 1.0 * (3000 - 2800) = 200
        #          Total: 700
        expected_pnl = Decimal(700)
        assert unrealized_pnl == expected_pnl

    def test_calculate_account_metrics_comprehensive(self):
        """Test comprehensive account metrics calculation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Create account with trading history
        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Add and close a profitable trade
        trade = create_paper_trade("BTC-USD", "LONG", Decimal("0.1"), Decimal(50000))
        account = account.add_trade(trade)
        account = account.close_trade(
            trade_id=trade.id, exit_price=Decimal(55000), exit_time=datetime.now()
        )

        # Add open trade
        open_trade = create_paper_trade(
            "ETH-USD", "LONG", Decimal("1.0"), Decimal(3000)
        )
        account = account.add_trade(open_trade)

        # Calculate metrics
        current_prices = {"ETH-USD": Decimal(3200)}
        metrics = calculate_account_metrics(account, current_prices)

        # Verify metrics structure and values
        assert "starting_balance" in metrics
        assert "current_balance" in metrics
        assert "equity" in metrics
        assert "unrealized_pnl" in metrics
        assert "total_pnl" in metrics
        assert "roi_percent" in metrics
        assert "margin_used" in metrics
        assert "margin_available" in metrics

        # Check that equity includes unrealized P&L
        expected_unrealized = Decimal(200)  # 1.0 * (3200 - 3000)
        assert metrics["unrealized_pnl"] == expected_unrealized

        # Check total P&L includes realized and unrealized
        assert metrics["total_pnl"] > Decimal(0)

    def test_decimal_precision_handling(self):
        """Test decimal precision handling in calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test normalization functions
        high_precision = Decimal("123.123456789012345")

        # Normalize to 2 decimal places (currency)
        normalized_2 = normalize_decimal_precision(high_precision, 2)
        assert str(normalized_2) == "123.12"

        # Normalize to 8 decimal places (crypto)
        normalized_8 = normalize_decimal_precision(high_precision, 8)
        assert str(normalized_8) == "123.12345679"

        # Test position calculations maintain precision
        position_value = calculate_position_value(
            size=Decimal("0.123456789"), price=Decimal("50000.123456789")
        )

        # Should maintain high precision in intermediate calculations
        assert isinstance(position_value, Decimal)
        assert position_value > Decimal(6000)

    def test_additional_pure_calculation_functions(self):
        """Test additional pure calculation functions."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test unrealized P&L simple calculation
        long_pnl = calculate_unrealized_pnl_simple(
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal(50000),
            current_price=Decimal(55000),
        )
        assert long_pnl == Decimal(500)  # 0.1 * (55000 - 50000)

        short_pnl = calculate_unrealized_pnl_simple(
            side="SHORT",
            size=Decimal("1.0"),
            entry_price=Decimal(3000),
            current_price=Decimal(2800),
        )
        assert short_pnl == Decimal(200)  # 1.0 * (3000 - 2800)

        # Test flat position
        flat_pnl = calculate_unrealized_pnl_simple(
            side="FLAT",
            size=Decimal(0),
            entry_price=Decimal(50000),
            current_price=Decimal(55000),
        )
        assert flat_pnl == Decimal(0)

        # Test margin requirement calculation
        margin = calculate_margin_requirement_simple(
            position_value=Decimal(10000), leverage=Decimal(5)
        )
        assert margin == Decimal(2000)  # 10000 / 5

        # Test fees calculation
        fees = calculate_fees_simple(
            position_value=Decimal(10000), fee_rate=Decimal("0.001")
        )
        assert fees == Decimal(10)  # 10000 * 0.001

        # Test position size validation
        valid_result = validate_position_size_simple(Decimal("1.0"))
        assert valid_result.is_ok()
        assert valid_result.unwrap() == Decimal("1.0")

        invalid_result = validate_position_size_simple(Decimal("-1.0"))
        assert invalid_result.is_err()
        assert "negative" in invalid_result.error

        # Test stop loss and take profit calculations
        stop_distance = calculate_stop_loss_distance(
            entry_price=Decimal(50000),
            stop_loss_pct=Decimal(2),  # 2%
        )
        assert stop_distance == Decimal(1000)  # 50000 * 0.02

        profit_distance = calculate_take_profit_distance(
            entry_price=Decimal(50000),
            take_profit_pct=Decimal(5),  # 5%
        )
        assert profit_distance == Decimal(2500)  # 50000 * 0.05


class TestFunctionalBalancePerformance(FPTestBase):
    """Test performance characteristics of functional balance operations."""

    def test_balance_validation_performance(self):
        """Test performance of balance validation operations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        config = create_default_balance_config()

        # Test many balance validations
        start_time = time.time()

        for i in range(1000):
            balance = Decimal(str(1000 + i))
            result = validate_balance_range(balance, config.balance_range)
            assert result.is_valid is True

        validation_time = time.time() - start_time

        # Should complete 1000 validations quickly
        assert validation_time < 2.0

    def test_account_state_operations_performance(self):
        """Test performance of account state operations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        account = PaperTradingAccountState.create_initial(Decimal(10000))

        # Test many state transitions
        start_time = time.time()

        current_account = account
        for i in range(100):
            # Add balance updates
            current_account = current_account.update_balance(Decimal(10))

        transition_time = time.time() - start_time

        # Should complete 100 state transitions quickly
        assert transition_time < 1.0
        assert current_account.current_balance == Decimal(11000)

    def test_calculation_functions_performance(self):
        """Test performance of calculation functions."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test many P&L calculations
        start_time = time.time()

        for i in range(1000):
            pnl = calculate_unrealized_pnl_simple(
                side="LONG",
                size=Decimal("0.1"),
                entry_price=Decimal(50000),
                current_price=Decimal(str(50000 + i)),
            )
            assert isinstance(pnl, Decimal)

        calculation_time = time.time() - start_time

        # Should complete 1000 calculations quickly
        assert calculation_time < 1.0


class TestFunctionalBalanceCompatibility(FPTestBase):
    """Test compatibility between functional and legacy balance systems."""

    def test_decimal_precision_compatibility(self):
        """Test decimal precision compatibility with legacy systems."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test that FP calculations produce same results as legacy expectations
        fp_balance = Decimal("10000.123456789")

        # Normalize for currency (2 decimal places)
        currency_normalized = normalize_decimal_precision(fp_balance, 2)
        assert str(currency_normalized) == "10000.12"

        # Normalize for crypto (8 decimal places)
        crypto_normalized = normalize_decimal_precision(fp_balance, 8)
        assert str(crypto_normalized) == "10000.12345679"

        # Test that results can be converted to float for legacy systems
        legacy_float = float(currency_normalized)
        assert legacy_float == 10000.12

    def test_account_state_to_legacy_format_conversion(self):
        """Test conversion from FP account state to legacy format."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Create FP account state
        account = PaperTradingAccountState.create_initial(Decimal(10000))
        account = account.update_balance(Decimal(500))  # Add some profit

        # Convert to legacy format (dict)
        legacy_format = {
            "starting_balance": float(account.starting_balance),
            "current_balance": float(account.current_balance),
            "equity": float(account.equity),
            "total_pnl": float(account.get_total_pnl()),
            "roi_percent": float(account.get_roi_percent()),
            "margin_used": float(account.margin_used),
            "margin_available": float(account.get_available_margin()),
            "open_positions": account.get_open_position_count(),
            "total_trades": account.get_total_trade_count(),
        }

        # Verify legacy format
        assert legacy_format["starting_balance"] == 10000.0
        assert legacy_format["current_balance"] == 10500.0
        assert legacy_format["total_pnl"] == 500.0
        assert legacy_format["roi_percent"] == 5.0
        assert legacy_format["margin_used"] == 0.0
        assert legacy_format["open_positions"] == 0
        assert legacy_format["total_trades"] == 0

    def test_balance_validation_error_compatibility(self):
        """Test balance validation error compatibility with legacy error handling."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        balance_range = BalanceRange(minimum=Decimal(100), maximum=Decimal(100000))

        # Test validation failure
        result = validate_balance_range(Decimal(50), balance_range)

        # Convert error to legacy format
        if result.error:
            legacy_error = {
                "error_type": result.error.error_type.value,
                "message": result.error.message,
                "severity": result.error.severity,
                "current_balance": float(result.error.current_balance),
                "timestamp": result.error.timestamp.isoformat(),
                "additional_context": result.error.additional_context,
            }

            # Verify legacy error format
            assert legacy_error["error_type"] == "range_check"
            assert "below minimum" in legacy_error["message"]
            assert legacy_error["severity"] == "high"
            assert legacy_error["current_balance"] == 50.0
            assert isinstance(legacy_error["timestamp"], str)

    def test_trading_fees_compatibility(self):
        """Test trading fees compatibility between FP and legacy systems."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Calculate fees using FP functions
        fp_fees = calculate_trade_fees(
            trade_value=Decimal(10000),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            has_exit_fee=True,
        )

        # Convert to legacy format
        legacy_fees = {
            "entry_fee": float(fp_fees.entry_fee),
            "exit_fee": float(fp_fees.exit_fee),
            "slippage_cost": float(fp_fees.slippage_cost),
            "total_fees": float(fp_fees.total_fees),
            "fee_rate": float(fp_fees.fee_rate),
        }

        # Verify calculations match expected legacy behavior
        assert legacy_fees["entry_fee"] == 10.0  # 10000 * 0.001
        assert legacy_fees["exit_fee"] == 10.0  # 10000 * 0.001
        assert legacy_fees["slippage_cost"] == 5.0  # 10000 * 0.0005
        assert legacy_fees["total_fees"] == 25.0  # 10 + 10 + 5


class TestFunctionalBalanceEdgeCases(FPTestBase):
    """Test edge cases in functional balance management."""

    def test_zero_balance_edge_cases(self):
        """Test edge cases with zero balances."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test zero balance validation
        balance_range = BalanceRange(minimum=Decimal(0), maximum=Decimal(100000))
        result = validate_balance_range(Decimal(0), balance_range)
        assert result.is_valid is True

        # Test zero position size calculation
        size = calculate_position_size(
            equity=Decimal(0),
            size_percentage=Decimal(100),
            leverage=Decimal(5),
            current_price=Decimal(50000),
        )
        assert size == Decimal(0)

        # Test zero trade value
        fees = calculate_fees_simple(
            position_value=Decimal(0), fee_rate=Decimal("0.001")
        )
        assert fees == Decimal(0)

    def test_extreme_precision_edge_cases(self):
        """Test edge cases with extreme decimal precision."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test very small amounts
        tiny_balance = Decimal("0.00000001")  # 1 satoshi worth
        tiny_fees = calculate_fees_simple(tiny_balance, Decimal("0.001"))
        assert tiny_fees == Decimal("0.00000000001")

        # Test very large amounts
        large_balance = Decimal("999999999.999999999")
        large_position = calculate_position_value(
            size=Decimal(1000000), price=large_balance
        )
        assert isinstance(large_position, Decimal)
        assert large_position > Decimal(999999999000000)

    def test_leverage_edge_cases(self):
        """Test edge cases with leverage calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test very high leverage
        high_leverage_margin = calculate_margin_requirement_simple(
            position_value=Decimal(10000),
            leverage=Decimal(100),  # 100x leverage
        )
        assert high_leverage_margin == Decimal(100)

        # Test leverage of 1 (no leverage)
        no_leverage_margin = calculate_margin_requirement_simple(
            position_value=Decimal(10000), leverage=Decimal(1)
        )
        assert no_leverage_margin == Decimal(10000)

        # Test zero leverage (should return full position value)
        zero_leverage_margin = calculate_margin_requirement_simple(
            position_value=Decimal(10000), leverage=Decimal(0)
        )
        assert zero_leverage_margin == Decimal(10000)

    def test_slippage_edge_cases(self):
        """Test edge cases with slippage calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")

        # Test zero slippage
        no_slippage_price = apply_slippage(
            price=Decimal(50000), side="LONG", slippage_rate=Decimal(0)
        )
        assert no_slippage_price == Decimal(50000)

        # Test maximum slippage (100%)
        max_slippage_price = apply_slippage(
            price=Decimal(50000),
            side="LONG",
            slippage_rate=Decimal("1.0"),  # 100% slippage
        )
        assert max_slippage_price == Decimal(100000)  # 50000 + 50000

        # Test slippage for SHORT
        short_slippage_price = apply_slippage(
            price=Decimal(50000),
            side="SHORT",
            slippage_rate=Decimal("0.01"),  # 1% slippage
        )
        assert short_slippage_price == Decimal(49500)  # 50000 - 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
