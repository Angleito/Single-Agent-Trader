"""
Functional Programming Futures Trading Tests

Test suite for futures trading with functional programming patterns.
Tests immutable margin types, account management, and futures-specific operations.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.fp.types.trading import (
    FuturesLimitOrder,
    FuturesMarketOrder,
    FuturesStopOrder,
    FuturesAccountBalance,
    MarginInfo,
    MarginHealthStatus,
    AccountType,
    CashTransferRequest,
    Position,
    OrderResult,
    OrderStatus,
    RiskLimits,
    RiskMetrics,
    FuturesMarketData,
    HEALTHY_MARGIN,
    WARNING_MARGIN,
    CRITICAL_MARGIN,
    LIQUIDATION_RISK_MARGIN,
    CFM_ACCOUNT,
    CBI_ACCOUNT,
    create_default_margin_info,
    create_conservative_risk_limits,
    create_aggressive_risk_limits
)
from bot.fp.types.result import Result, Ok, Err
from bot.fp.effects.io import IOEither
from bot.fp.effects.exchange import (
    validate_futures_order,
    calculate_margin_requirement,
    check_margin_health,
    execute_cash_transfer
)


class TestMarginInfo:
    """Test immutable margin information."""

    def test_margin_info_creation_and_validation(self):
        """Test MarginInfo creation and validation."""
        margin_info = MarginInfo(
            total_margin=Decimal("10000"),
            available_margin=Decimal("7000"),
            used_margin=Decimal("3000"),
            maintenance_margin=Decimal("1500"),
            initial_margin=Decimal("2000"),
            health_status=HEALTHY_MARGIN,
            liquidation_threshold=Decimal("500"),
            intraday_margin_requirement=Decimal("2000"),
            overnight_margin_requirement=Decimal("2500")
        )
        
        assert margin_info.total_margin == Decimal("10000")
        assert margin_info.available_margin == Decimal("7000")
        assert margin_info.used_margin == Decimal("3000")
        assert margin_info.health_status.is_healthy()
        assert not margin_info.health_status.needs_attention()

    def test_margin_info_calculations(self):
        """Test margin calculation properties."""
        margin_info = MarginInfo(
            total_margin=Decimal("5000"),
            available_margin=Decimal("2000"),
            used_margin=Decimal("3000"),
            maintenance_margin=Decimal("1000"),
            initial_margin=Decimal("1500"),
            health_status=HEALTHY_MARGIN,
            liquidation_threshold=Decimal("200"),
            intraday_margin_requirement=Decimal("1500"),
            overnight_margin_requirement=Decimal("2000")
        )
        
        assert margin_info.margin_ratio == 0.6  # 3000/5000
        assert margin_info.margin_usage_percentage == 60.0
        assert margin_info.free_margin_percentage == 40.0
        assert margin_info.can_open_position(Decimal("1500"))
        assert not margin_info.can_open_position(Decimal("2500"))

    def test_margin_info_validation_errors(self):
        """Test MarginInfo validation catches errors."""
        # Total margin doesn't equal available + used
        with pytest.raises(ValueError, match="Total margin .* must equal available .* \\+ used"):
            MarginInfo(
                total_margin=Decimal("10000"),
                available_margin=Decimal("7000"),
                used_margin=Decimal("2000"),  # Should be 3000
                maintenance_margin=Decimal("1000"),
                initial_margin=Decimal("1500"),
                health_status=HEALTHY_MARGIN,
                liquidation_threshold=Decimal("500"),
                intraday_margin_requirement=Decimal("1500"),
                overnight_margin_requirement=Decimal("2000")
            )
        
        # Negative values
        with pytest.raises(ValueError, match="Available margin cannot be negative"):
            MarginInfo(
                total_margin=Decimal("5000"),
                available_margin=Decimal("-1000"),  # Invalid
                used_margin=Decimal("6000"),
                maintenance_margin=Decimal("1000"),
                initial_margin=Decimal("1500"),
                health_status=HEALTHY_MARGIN,
                liquidation_threshold=Decimal("500"),
                intraday_margin_requirement=Decimal("1500"),
                overnight_margin_requirement=Decimal("2000")
            )

    def test_margin_health_status(self):
        """Test margin health status functionality."""
        # Healthy margin
        healthy_margin = HEALTHY_MARGIN
        assert healthy_margin.is_healthy()
        assert not healthy_margin.needs_attention()
        assert not healthy_margin.is_critical()
        
        # Warning margin
        warning_margin = WARNING_MARGIN
        assert not warning_margin.is_healthy()
        assert warning_margin.needs_attention()
        assert not warning_margin.is_critical()
        
        # Critical margin
        critical_margin = CRITICAL_MARGIN
        assert not critical_margin.is_healthy()
        assert critical_margin.needs_attention()
        assert critical_margin.is_critical()
        
        # Liquidation risk
        liquidation_margin = LIQUIDATION_RISK_MARGIN
        assert not liquidation_margin.is_healthy()
        assert liquidation_margin.needs_attention()
        assert liquidation_margin.is_critical()


class TestFuturesAccountBalance:
    """Test futures account balance management."""

    def test_futures_account_creation(self):
        """Test FuturesAccountBalance creation."""
        margin_info = create_default_margin_info()
        
        futures_account = FuturesAccountBalance(
            account_type=CFM_ACCOUNT,
            account_id="futures-account-123",
            currency="USD",
            cash_balance=Decimal("50000"),
            futures_balance=Decimal("25000"),
            total_balance=Decimal("75000"),
            margin_info=margin_info,
            max_leverage=20,
            max_position_size=Decimal("1000000")
        )
        
        assert futures_account.account_type.is_futures()
        assert not futures_account.account_type.is_spot()
        assert futures_account.equity == Decimal("75000")
        assert futures_account.buying_power == Decimal("160000")  # available_margin * max_leverage

    def test_cash_transfer_validation(self):
        """Test cash transfer validation."""
        margin_info = create_default_margin_info()
        
        futures_account = FuturesAccountBalance(
            account_type=CFM_ACCOUNT,
            account_id="test-account",
            currency="USD",
            cash_balance=Decimal("10000"),
            futures_balance=Decimal("5000"),
            total_balance=Decimal("15000"),
            margin_info=margin_info,
            auto_cash_transfer_enabled=True,
            min_cash_transfer_amount=Decimal("100"),
            max_cash_transfer_amount=Decimal("5000")
        )
        
        # Valid transfer to futures
        assert futures_account.can_transfer_cash(Decimal("1000"), "to_futures")
        
        # Invalid transfer (too large)
        assert not futures_account.can_transfer_cash(Decimal("15000"), "to_futures")
        
        # Invalid transfer (below minimum)
        assert not futures_account.can_transfer_cash(Decimal("50"), "to_futures")
        
        # Valid transfer to spot
        assert futures_account.can_transfer_cash(Decimal("2000"), "to_spot")

    def test_account_validation(self):
        """Test account validation rules."""
        margin_info = create_default_margin_info()
        
        # Invalid leverage
        with pytest.raises(ValueError, match="Max leverage must be between 1 and 100"):
            FuturesAccountBalance(
                account_type=CFM_ACCOUNT,
                account_id="test-account",
                currency="USD",
                cash_balance=Decimal("10000"),
                futures_balance=Decimal("5000"),
                total_balance=Decimal("15000"),
                margin_info=margin_info,
                max_leverage=150  # Too high
            )
        
        # Empty account ID
        with pytest.raises(ValueError, match="Account ID cannot be empty"):
            FuturesAccountBalance(
                account_type=CFM_ACCOUNT,
                account_id="",  # Empty
                currency="USD",
                cash_balance=Decimal("10000"),
                futures_balance=Decimal("5000"),
                total_balance=Decimal("15000"),
                margin_info=margin_info
            )


class TestCashTransferRequest:
    """Test cash transfer request functionality."""

    def test_cash_transfer_request_creation(self):
        """Test creation of cash transfer request."""
        transfer = CashTransferRequest(
            from_account=CBI_ACCOUNT,
            to_account=CFM_ACCOUNT,
            amount=Decimal("1000"),
            currency="USD",
            reason="MANUAL"
        )
        
        assert transfer.from_account.is_spot()
        assert transfer.to_account.is_futures()
        assert transfer.is_to_futures
        assert not transfer.is_to_spot
        assert not transfer.is_automated

    def test_cash_transfer_validation(self):
        """Test cash transfer validation."""
        # Invalid: same account types
        with pytest.raises(ValueError, match="Cannot transfer between same account types"):
            CashTransferRequest(
                from_account=CFM_ACCOUNT,
                to_account=CFM_ACCOUNT,  # Same as from
                amount=Decimal("1000")
            )
        
        # Invalid: negative amount
        with pytest.raises(ValueError, match="Transfer amount must be positive"):
            CashTransferRequest(
                from_account=CBI_ACCOUNT,
                to_account=CFM_ACCOUNT,
                amount=Decimal("-500")  # Negative
            )

    def test_automated_transfer_detection(self):
        """Test detection of automated transfers."""
        manual_transfer = CashTransferRequest(
            from_account=CBI_ACCOUNT,
            to_account=CFM_ACCOUNT,
            amount=Decimal("1000"),
            reason="MANUAL"
        )
        
        margin_call_transfer = CashTransferRequest(
            from_account=CBI_ACCOUNT,
            to_account=CFM_ACCOUNT,
            amount=Decimal("2000"),
            reason="MARGIN_CALL"
        )
        
        assert not manual_transfer.is_automated
        assert margin_call_transfer.is_automated


class TestFuturesOrderTypes:
    """Test futures-specific order types."""

    def test_futures_limit_order_creation(self):
        """Test FuturesLimitOrder creation and validation."""
        order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.1,
            leverage=10,
            margin_required=Decimal("550.0"),
            reduce_only=False,
            post_only=True,
            time_in_force="GTC"
        )
        
        assert order.symbol == "BTC-PERP"
        assert order.leverage == 10
        assert order.notional_value == Decimal("5500.0")
        assert order.position_value == Decimal("55000.0")  # notional * leverage
        assert not order.reduce_only
        assert order.post_only

    def test_futures_market_order_creation(self):
        """Test FuturesMarketOrder creation."""
        order = FuturesMarketOrder(
            symbol="ETH-PERP",
            side="sell",
            size=1.0,
            leverage=5,
            margin_required=Decimal("600.0"),
            reduce_only=True
        )
        
        assert order.symbol == "ETH-PERP"
        assert order.side == "sell"
        assert order.leverage == 5
        assert order.reduce_only

    def test_futures_stop_order_creation(self):
        """Test FuturesStopOrder creation."""
        order = FuturesStopOrder(
            symbol="SOL-PERP",
            side="buy",
            stop_price=100.0,
            size=5.0,
            leverage=3,
            margin_required=Decimal("166.67"),
            time_in_force="IOC"
        )
        
        assert order.stop_price == 100.0
        assert order.leverage == 3
        assert order.time_in_force == "IOC"

    def test_futures_order_validation(self):
        """Test futures order validation."""
        # Invalid leverage
        with pytest.raises(ValueError, match="Leverage must be between 1 and 100"):
            FuturesLimitOrder(
                symbol="BTC-PERP",
                side="buy",
                price=50000.0,
                size=0.1,
                leverage=200,  # Too high
                margin_required=Decimal("250.0")
            )
        
        # Negative margin requirement
        with pytest.raises(ValueError, match="Margin required cannot be negative"):
            FuturesMarketOrder(
                symbol="ETH-PERP",
                side="sell",
                size=1.0,
                leverage=5,
                margin_required=Decimal("-100")  # Negative
            )


class TestRiskManagement:
    """Test risk management for futures trading."""

    def test_risk_limits_creation(self):
        """Test creation of risk limits."""
        conservative_limits = create_conservative_risk_limits()
        
        assert conservative_limits.max_leverage == 3
        assert conservative_limits.max_drawdown_percentage == 10.0
        assert conservative_limits.stop_loss_percentage == 5.0
        
        aggressive_limits = create_aggressive_risk_limits()
        
        assert aggressive_limits.max_leverage == 10
        assert aggressive_limits.max_drawdown_percentage == 25.0
        assert aggressive_limits.stop_loss_percentage == 10.0

    def test_risk_limits_validation(self):
        """Test risk limits validation."""
        # Invalid max leverage
        with pytest.raises(ValueError, match="Max leverage must be between 1 and 100"):
            RiskLimits(
                max_position_size=Decimal("10000"),
                max_daily_loss=Decimal("500"),
                max_drawdown_percentage=10.0,
                max_leverage=150,  # Too high
                max_open_positions=5,
                max_correlation_exposure=0.5,
                stop_loss_percentage=5.0,
                take_profit_percentage=15.0
            )

    def test_risk_metrics_calculations(self):
        """Test risk metrics calculations."""
        risk_metrics = RiskMetrics(
            account_balance=Decimal("50000"),
            available_margin=Decimal("20000"),
            used_margin=Decimal("10000"),
            daily_pnl=Decimal("-500"),
            total_exposure=Decimal("80000"),
            current_positions=3,
            max_daily_loss_reached=False
        )
        
        assert risk_metrics.margin_utilization == 1/3  # 10000 / 30000
        assert risk_metrics.exposure_ratio == 1.6  # 80000 / 50000
        assert risk_metrics.daily_return_percentage == -1.0  # -500 / 50000 * 100
        
        # Risk score calculation
        score = risk_metrics.risk_score()
        assert 0 <= score <= 100

    def test_risk_limits_compliance(self):
        """Test compliance with risk limits."""
        limits = create_conservative_risk_limits()
        
        compliant_metrics = RiskMetrics(
            account_balance=Decimal("50000"),
            available_margin=Decimal("25000"),
            used_margin=Decimal("5000"),
            daily_pnl=Decimal("-200"),  # Within daily loss limit
            total_exposure=Decimal("8000"),  # Within position size limit
            current_positions=3,  # Within max positions
            max_daily_loss_reached=False
        )
        
        assert compliant_metrics.is_within_risk_limits(limits)
        
        non_compliant_metrics = RiskMetrics(
            account_balance=Decimal("50000"),
            available_margin=Decimal("25000"),
            used_margin=Decimal("5000"),
            daily_pnl=Decimal("-1000"),  # Exceeds daily loss limit
            total_exposure=Decimal("15000"),  # Exceeds position size limit
            current_positions=8,  # Exceeds max positions
            max_daily_loss_reached=True
        )
        
        assert not non_compliant_metrics.is_within_risk_limits(limits)


class TestFuturesMarketData:
    """Test futures-specific market data."""

    def test_futures_market_data_creation(self):
        """Test FuturesMarketData creation with funding info."""
        from bot.fp.types.trading import FunctionalMarketData
        
        base_data = FunctionalMarketData(
            symbol="BTC-PERP",
            timestamp=datetime.now(),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("1000")
        )
        
        futures_data = FuturesMarketData(
            base_data=base_data,
            open_interest=Decimal("5000000"),
            funding_rate=0.0001,
            next_funding_time=datetime.now() + timedelta(hours=8),
            mark_price=Decimal("50505"),
            index_price=Decimal("50500")
        )
        
        assert futures_data.symbol == "BTC-PERP"
        assert futures_data.open_interest == Decimal("5000000")
        assert futures_data.basis == Decimal("5")  # mark - index
        assert abs(futures_data.funding_rate_8h_annualized - 0.1095) < 0.001

    def test_futures_data_validation(self):
        """Test futures data validation."""
        from bot.fp.types.trading import FunctionalMarketData
        
        base_data = FunctionalMarketData(
            symbol="ETH-PERP",
            timestamp=datetime.now(),
            open=Decimal("3000"),
            high=Decimal("3100"),
            low=Decimal("2950"),
            close=Decimal("3050"),
            volume=Decimal("500")
        )
        
        # Invalid open interest
        with pytest.raises(ValueError, match="Open interest cannot be negative"):
            FuturesMarketData(
                base_data=base_data,
                open_interest=Decimal("-1000"),  # Negative
                funding_rate=0.0001
            )
        
        # Unrealistic funding rate
        with pytest.raises(ValueError, match="Funding rate seems unrealistic"):
            FuturesMarketData(
                base_data=base_data,
                open_interest=Decimal("1000000"),
                funding_rate=2.0  # 200% funding rate
            )

    def test_futures_data_updates(self):
        """Test immutable updates to futures data."""
        from bot.fp.types.trading import FunctionalMarketData
        
        base_data = FunctionalMarketData(
            symbol="SOL-PERP",
            timestamp=datetime.now(),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("98"),
            close=Decimal("102"),
            volume=Decimal("10000")
        )
        
        original_futures = FuturesMarketData(
            base_data=base_data,
            open_interest=Decimal("2000000"),
            funding_rate=0.0002,
            next_funding_time=datetime.now() + timedelta(hours=4)
        )
        
        # Update funding information
        new_funding_time = datetime.now() + timedelta(hours=8)
        updated_futures = original_futures.with_updated_funding(0.0003, new_funding_time)
        
        # Original should be unchanged
        assert original_futures.funding_rate == 0.0002
        # Updated should have new values
        assert updated_futures.funding_rate == 0.0003
        assert updated_futures.next_funding_time == new_funding_time


class MockFuturesExchange:
    """Mock futures exchange for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.orders_placed = []
        self.margin_transfers = []
        self.liquidations = []

    async def place_futures_order(self, order) -> IOEither[Exception, OrderResult]:
        """Mock futures order placement."""
        if self.should_fail:
            return IOEither.left(Exception("Futures order failed"))
        
        self.orders_placed.append(order)
        
        result = OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_size=Decimal(str(order.size)),
            average_price=Decimal(str(getattr(order, 'price', 50000))),
            fees=Decimal("2.0"),  # Higher fees for futures
            created_at=datetime.now()
        )
        
        return IOEither.right(result)

    async def get_margin_info(self) -> IOEither[Exception, MarginInfo]:
        """Mock margin info retrieval."""
        if self.should_fail:
            return IOEither.left(Exception("Failed to get margin info"))
        
        margin_info = create_default_margin_info()
        return IOEither.right(margin_info)

    async def transfer_cash(self, transfer_request: CashTransferRequest) -> IOEither[Exception, bool]:
        """Mock cash transfer."""
        if self.should_fail:
            return IOEither.left(Exception("Transfer failed"))
        
        self.margin_transfers.append(transfer_request)
        return IOEither.right(True)

    async def check_liquidation_risk(self, account_id: str) -> IOEither[Exception, bool]:
        """Mock liquidation risk check."""
        if self.should_fail:
            return IOEither.left(Exception("Failed to check liquidation risk"))
        
        # Simulate low risk
        return IOEither.right(False)


class TestFuturesExchangeOperations:
    """Test futures exchange operations."""

    @pytest.mark.asyncio
    async def test_futures_order_placement(self):
        """Test placing futures orders."""
        exchange = MockFuturesExchange()
        
        futures_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.1,
            leverage=10,
            margin_required=Decimal("550.0")
        )
        
        result = await exchange.place_futures_order(futures_order)
        order_result = result.run()
        
        assert order_result.is_right()
        assert len(exchange.orders_placed) == 1
        assert exchange.orders_placed[0].leverage == 10

    @pytest.mark.asyncio
    async def test_margin_info_retrieval(self):
        """Test retrieving margin information."""
        exchange = MockFuturesExchange()
        
        result = await exchange.get_margin_info()
        margin_result = result.run()
        
        assert margin_result.is_right()
        margin_info = margin_result.value
        assert isinstance(margin_info, MarginInfo)
        assert margin_info.health_status.is_healthy()

    @pytest.mark.asyncio
    async def test_cash_transfer_execution(self):
        """Test executing cash transfers."""
        exchange = MockFuturesExchange()
        
        transfer = CashTransferRequest(
            from_account=CBI_ACCOUNT,
            to_account=CFM_ACCOUNT,
            amount=Decimal("5000"),
            reason="MARGIN_CALL"
        )
        
        result = await exchange.transfer_cash(transfer)
        transfer_result = result.run()
        
        assert transfer_result.is_right()
        assert len(exchange.margin_transfers) == 1
        assert exchange.margin_transfers[0].is_automated

    @pytest.mark.asyncio
    async def test_liquidation_risk_monitoring(self):
        """Test liquidation risk monitoring."""
        exchange = MockFuturesExchange()
        
        result = await exchange.check_liquidation_risk("test-account-123")
        risk_result = result.run()
        
        assert risk_result.is_right()
        assert risk_result.value is False  # No liquidation risk

    @pytest.mark.asyncio
    async def test_futures_error_handling(self):
        """Test error handling in futures operations."""
        failing_exchange = MockFuturesExchange(should_fail=True)
        
        futures_order = FuturesMarketOrder(
            symbol="ETH-PERP",
            side="sell",
            size=1.0,
            leverage=5,
            margin_required=Decimal("600.0")
        )
        
        result = await failing_exchange.place_futures_order(futures_order)
        order_result = result.run()
        
        assert order_result.is_left()
        assert "Futures order failed" in str(order_result.value)


class TestLeverageAndMarginCalculations:
    """Test leverage and margin calculations."""

    def test_margin_requirement_calculation(self):
        """Test margin requirement calculations."""
        # 10x leverage on $50,000 position = $5,000 margin
        position_value = Decimal("50000")
        leverage = 10
        
        margin_required = position_value / leverage
        assert margin_required == Decimal("5000")
        
        # Test with futures order
        order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=50000.0,
            size=1.0,
            leverage=10,
            margin_required=margin_required
        )
        
        assert order.notional_value == position_value
        assert order.position_value == position_value * leverage

    def test_position_sizing_with_leverage(self):
        """Test position sizing with different leverage levels."""
        account_balance = Decimal("10000")
        risk_percentage = Decimal("0.02")  # 2% risk
        
        # Calculate position sizes for different leverage
        leverages = [1, 5, 10, 20]
        
        for leverage in leverages:
            # Risk amount stays the same
            risk_amount = account_balance * risk_percentage
            
            # Position size can be larger with higher leverage
            max_position_value = risk_amount * leverage
            
            # But margin requirement is lower
            margin_required = max_position_value / leverage
            
            assert margin_required == risk_amount

    def test_liquidation_price_calculation(self):
        """Test liquidation price calculations."""
        # Long position with 10x leverage
        entry_price = Decimal("50000")
        leverage = 10
        maintenance_margin_rate = Decimal("0.05")  # 5%
        
        # Liquidation occurs when losses equal initial margin minus maintenance
        # For long: liquidation_price = entry_price * (1 - (1/leverage - maintenance_rate))
        liquidation_price = entry_price * (1 - (Decimal("1")/leverage - maintenance_margin_rate))
        
        # With 10x leverage and 5% maintenance, liquidation at ~95% of entry
        expected_liquidation = entry_price * Decimal("0.95")
        
        assert abs(liquidation_price - expected_liquidation) < Decimal("100")

    def test_funding_cost_calculation(self):
        """Test funding cost calculations."""
        position_size = Decimal("1.0")  # 1 BTC
        mark_price = Decimal("50000")
        funding_rate = 0.0001  # 0.01%
        
        funding_cost = position_size * mark_price * Decimal(str(funding_rate))
        
        assert funding_cost == Decimal("5.0")  # $5 funding cost
        
        # Daily funding (3 payments per day)
        daily_funding = funding_cost * 3
        assert daily_funding == Decimal("15.0")

    def test_pnl_calculation_with_leverage(self):
        """Test PnL calculations with leverage."""
        entry_price = Decimal("50000")
        current_price = Decimal("51000")
        position_size = Decimal("0.1")  # 0.1 BTC
        leverage = 10
        
        # PnL is the same regardless of leverage
        # (current_price - entry_price) * position_size
        unrealized_pnl = (current_price - entry_price) * position_size
        
        assert unrealized_pnl == Decimal("100")  # $100 profit
        
        # But ROI is amplified by leverage
        margin_used = (entry_price * position_size) / leverage
        roi_percentage = (unrealized_pnl / margin_used) * 100
        
        # 20% price move with 10x leverage = 200% ROI
        assert roi_percentage == Decimal("200")


class TestMarginCallScenarios:
    """Test margin call scenarios and responses."""

    def test_margin_call_detection(self):
        """Test detection of margin call conditions."""
        # Critical margin situation
        critical_margin = MarginInfo(
            total_margin=Decimal("10000"),
            available_margin=Decimal("500"),  # Very low available
            used_margin=Decimal("9500"),
            maintenance_margin=Decimal("2000"),
            initial_margin=Decimal("3000"),
            health_status=CRITICAL_MARGIN,
            liquidation_threshold=Decimal("1000"),
            intraday_margin_requirement=Decimal("3000"),
            overnight_margin_requirement=Decimal("4000")
        )
        
        # Should trigger margin call
        assert critical_margin.health_status.is_critical()
        assert critical_margin.available_margin < critical_margin.maintenance_margin

    def test_automatic_cash_transfer_for_margin(self):
        """Test automatic cash transfer when margin is low."""
        margin_info = MarginInfo(
            total_margin=Decimal("5000"),
            available_margin=Decimal("200"),  # Below minimum
            used_margin=Decimal("4800"),
            maintenance_margin=Decimal("1000"),
            initial_margin=Decimal("1500"),
            health_status=WARNING_MARGIN,
            liquidation_threshold=Decimal("500"),
            intraday_margin_requirement=Decimal("1500"),
            overnight_margin_requirement=Decimal("2000")
        )
        
        futures_account = FuturesAccountBalance(
            account_type=CFM_ACCOUNT,
            account_id="test-account",
            currency="USD",
            cash_balance=Decimal("15000"),  # Has cash available
            futures_balance=Decimal("5000"),
            total_balance=Decimal("20000"),
            margin_info=margin_info,
            auto_cash_transfer_enabled=True
        )
        
        # Should be able to transfer cash to cover margin
        transfer_amount = Decimal("2000")
        assert futures_account.can_transfer_cash(transfer_amount, "to_futures")
        
        # Create transfer request
        transfer = CashTransferRequest(
            from_account=CBI_ACCOUNT,
            to_account=CFM_ACCOUNT,
            amount=transfer_amount,
            reason="MARGIN_CALL"
        )
        
        assert transfer.is_automated
        assert transfer.reason == "MARGIN_CALL"

    def test_position_reduction_on_margin_call(self):
        """Test position reduction when margin call occurs."""
        # Simulate reducing position to meet margin requirements
        original_position = Position(
            symbol="BTC-PERP",
            side="long",
            size=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("45000"),  # Loss position
            unrealized_pnl=Decimal("-5000"),
            realized_pnl=Decimal("0"),
            entry_time=datetime.now()
        )
        
        # Calculate required reduction (e.g., 50% to meet margin)
        reduction_percentage = Decimal("0.5")
        new_size = original_position.size * (1 - reduction_percentage)
        
        # Create reduce-only order
        reduce_order = FuturesMarketOrder(
            symbol="BTC-PERP",
            side="sell",  # Opposite side to reduce long position
            size=float(original_position.size - new_size),
            leverage=1,  # Reduce-only doesn't use leverage
            margin_required=Decimal("0"),
            reduce_only=True
        )
        
        assert reduce_order.reduce_only
        assert reduce_order.side == "sell"
        assert reduce_order.size == 0.5  # Half position


if __name__ == "__main__":
    pytest.main([__file__, "-v"])