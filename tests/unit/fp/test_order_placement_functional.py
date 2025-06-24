"""
Functional Programming Order Placement Tests

Test suite for order placement with functional programming patterns.
Tests immutable Order types, Result/Either error handling, and order execution safety.
"""

import asyncio
from datetime import datetime
from decimal import Decimal

import pytest

from bot.fp.effects.io import IOEither
from bot.fp.types.trading import (
    HEALTHY_MARGIN,
    FuturesLimitOrder,
    FuturesMarketOrder,
    Hold,
    LimitOrder,
    Long,
    MarginInfo,
    MarketMake,
    MarketOrder,
    OrderResult,
    OrderStatus,
    RiskLimits,
    Short,
    StopOrder,
    convert_trade_signal_to_orders,
    create_conservative_risk_limits,
    create_limit_orders_from_market_make,
    create_market_order_from_signal,
)


class TestImmutableOrderTypes:
    """Test immutable order types and their properties."""

    def test_limit_order_creation_and_properties(self):
        """Test LimitOrder creation and calculated properties."""
        order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.1,
            order_id="test-order-123",
        )

        assert order.symbol == "BTC-USD"
        assert order.side == "buy"
        assert order.price == 50000.0
        assert order.size == 0.1
        assert order.order_id == "test-order-123"
        assert order.value == 5000.0  # price * size

    def test_market_order_auto_id_generation(self):
        """Test automatic order ID generation."""
        order1 = MarketOrder(symbol="ETH-USD", side="sell", size=1.0)
        order2 = MarketOrder(symbol="ETH-USD", side="sell", size=1.0)

        assert order1.order_id != order2.order_id
        assert len(order1.order_id) > 0
        assert len(order2.order_id) > 0

    def test_stop_order_validation(self):
        """Test StopOrder validation."""
        # Valid stop order
        order = StopOrder(symbol="LTC-USD", side="sell", stop_price=200.0, size=5.0)

        assert order.stop_price == 200.0
        assert order.size == 5.0

        # Invalid stop price
        with pytest.raises(ValueError, match="Stop price must be positive"):
            StopOrder(
                symbol="LTC-USD",
                side="sell",
                stop_price=0.0,  # Invalid
                size=5.0,
            )

    def test_futures_order_margin_calculations(self):
        """Test futures order margin calculations."""
        futures_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.1,
            leverage=10,
            margin_required=Decimal("550.0"),
        )

        assert futures_order.notional_value == Decimal("5500.0")
        assert futures_order.position_value == Decimal("55000.0")  # notional * leverage
        assert futures_order.leverage == 10

    def test_futures_order_validation(self):
        """Test futures order validation rules."""
        # Invalid leverage
        with pytest.raises(ValueError, match="Leverage must be between 1 and 100"):
            FuturesLimitOrder(
                symbol="ETH-PERP",
                side="buy",
                price=3000.0,
                size=1.0,
                leverage=150,  # Too high
                margin_required=Decimal("30.0"),
            )

        # Negative margin
        with pytest.raises(ValueError, match="Margin required cannot be negative"):
            FuturesMarketOrder(
                symbol="ETH-PERP",
                side="sell",
                size=1.0,
                leverage=5,
                margin_required=Decimal(-100),  # Negative
            )

    def test_order_immutability(self):
        """Test that orders are immutable after creation."""
        order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.1)

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            order.price = 51000.0  # type: ignore

        with pytest.raises(AttributeError):
            order.size = 0.2  # type: ignore


class TestTradeSignalConversion:
    """Test conversion of trade signals to orders."""

    def test_long_signal_to_market_order(self):
        """Test converting Long signal to market order."""
        signal = Long(confidence=0.8, size=0.5, reason="Strong bullish pattern")

        order = create_market_order_from_signal(signal, "BTC-USD", 1.0)

        assert order is not None
        assert isinstance(order, MarketOrder)
        assert order.symbol == "BTC-USD"
        assert order.side == "buy"
        assert order.size == 0.5  # signal.size * base_size

    def test_short_signal_to_market_order(self):
        """Test converting Short signal to market order."""
        signal = Short(confidence=0.7, size=0.3, reason="Bearish divergence")

        order = create_market_order_from_signal(signal, "ETH-USD", 2.0)

        assert order is not None
        assert isinstance(order, MarketOrder)
        assert order.symbol == "ETH-USD"
        assert order.side == "sell"
        assert order.size == 0.6  # signal.size * base_size

    def test_hold_signal_returns_none(self):
        """Test that Hold signal returns None for order creation."""
        signal = Hold(reason="Market uncertainty")

        order = create_market_order_from_signal(signal, "BTC-USD", 1.0)

        assert order is None

    def test_market_make_signal_to_limit_orders(self):
        """Test converting MarketMake signal to limit orders."""
        signal = MarketMake(
            bid_price=49900.0, ask_price=50100.0, bid_size=0.1, ask_size=0.1
        )

        bid_order, ask_order = create_limit_orders_from_market_make(signal, "BTC-USD")

        # Bid order
        assert isinstance(bid_order, LimitOrder)
        assert bid_order.side == "buy"
        assert bid_order.price == 49900.0
        assert bid_order.size == 0.1

        # Ask order
        assert isinstance(ask_order, LimitOrder)
        assert ask_order.side == "sell"
        assert ask_order.price == 50100.0
        assert ask_order.size == 0.1

    def test_complete_signal_to_orders_conversion(self):
        """Test complete conversion of signals to executable orders."""
        # Long signal with stop loss and take profit
        long_signal = Long(confidence=0.9, size=0.2, reason="Breakout confirmed")

        orders = convert_trade_signal_to_orders(long_signal, "BTC-USD", 50000.0)

        assert len(orders) == 3  # Market order + stop loss + take profit

        market_order = orders[0]
        assert isinstance(market_order, MarketOrder)
        assert market_order.side == "buy"

        # Should include stop loss and take profit orders
        stop_orders = [o for o in orders if isinstance(o, StopOrder)]
        limit_orders = [o for o in orders if isinstance(o, LimitOrder)]

        assert len(stop_orders) == 1  # Stop loss
        assert len(limit_orders) == 1  # Take profit

    def test_signal_validation(self):
        """Test trade signal validation."""
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Long(confidence=1.5, size=0.1, reason="Test")

        # Invalid size
        with pytest.raises(ValueError, match="Size must be between 0 and 1"):
            Short(confidence=0.8, size=0.0, reason="Test")

        # Invalid market make prices
        with pytest.raises(
            ValueError, match="Bid price .* must be less than ask price"
        ):
            MarketMake(
                bid_price=50100.0,
                ask_price=50000.0,  # Bid > Ask (invalid)
                bid_size=0.1,
                ask_size=0.1,
            )


class MockExchangeExecutor:
    """Mock exchange executor for testing order placement."""

    def __init__(self, should_fail: bool = False, failure_type: str = "generic"):
        self.should_fail = should_fail
        self.failure_type = failure_type
        self.orders_placed = []
        self.orders_cancelled = []
        self.order_history = {}

    async def place_order_async(self, order) -> IOEither[Exception, OrderResult]:
        """Mock async order placement."""
        if self.should_fail:
            if self.failure_type == "insufficient_funds":
                return IOEither.left(Exception("Insufficient funds"))
            if self.failure_type == "invalid_symbol":
                return IOEither.left(Exception("Invalid trading symbol"))
            if self.failure_type == "market_closed":
                return IOEither.left(Exception("Market is closed"))
            return IOEither.left(Exception("Order placement failed"))

        # Store order
        self.orders_placed.append(order)
        self.order_history[order.order_id] = order

        # Create result
        result = OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_size=Decimal(str(order.size)),
            average_price=Decimal(str(getattr(order, "price", 50000))),
            fees=Decimal("1.0"),
            created_at=datetime.now(),
        )

        return IOEither.right(result)

    async def cancel_order_async(self, order_id: str) -> IOEither[Exception, bool]:
        """Mock async order cancellation."""
        if self.should_fail:
            return IOEither.left(Exception("Order cancellation failed"))

        self.orders_cancelled.append(order_id)
        return IOEither.right(True)

    def get_placed_orders(self) -> list:
        """Get all placed orders."""
        return self.orders_placed.copy()

    def get_cancelled_orders(self) -> list[str]:
        """Get all cancelled orders."""
        return self.orders_cancelled.copy()


class TestOrderPlacementFlow:
    """Test complete order placement flow."""

    @pytest.mark.asyncio
    async def test_successful_market_order_placement(self):
        """Test successful market order placement."""
        executor = MockExchangeExecutor()

        order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)

        result = await executor.place_order_async(order)
        order_result = result.run()

        assert order_result.is_right()
        assert order_result.value.order_id == order.order_id
        assert order_result.value.status == OrderStatus.FILLED
        assert len(executor.get_placed_orders()) == 1

    @pytest.mark.asyncio
    async def test_successful_limit_order_placement(self):
        """Test successful limit order placement."""
        executor = MockExchangeExecutor()

        order = LimitOrder(symbol="ETH-USD", side="sell", price=3000.0, size=2.0)

        result = await executor.place_order_async(order)
        order_result = result.run()

        assert order_result.is_right()
        assert order_result.value.filled_size == Decimal("2.0")
        assert order_result.value.average_price == Decimal("3000.0")

    @pytest.mark.asyncio
    async def test_futures_order_placement(self):
        """Test futures order placement."""
        executor = MockExchangeExecutor()

        futures_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.1,
            leverage=10,
            margin_required=Decimal("550.0"),
            reduce_only=False,
        )

        result = await executor.place_order_async(futures_order)
        order_result = result.run()

        assert order_result.is_right()
        placed_order = executor.get_placed_orders()[0]
        assert placed_order.leverage == 10
        assert placed_order.margin_required == Decimal("550.0")

    @pytest.mark.asyncio
    async def test_order_placement_error_handling(self):
        """Test error handling in order placement."""
        executor = MockExchangeExecutor(
            should_fail=True, failure_type="insufficient_funds"
        )

        order = MarketOrder(symbol="BTC-USD", side="buy", size=1.0)

        result = await executor.place_order_async(order)
        order_result = result.run()

        assert order_result.is_left()
        assert "Insufficient funds" in str(order_result.value)
        assert len(executor.get_placed_orders()) == 0

    @pytest.mark.asyncio
    async def test_order_cancellation(self):
        """Test order cancellation."""
        executor = MockExchangeExecutor()

        # Place order first
        order = LimitOrder(symbol="BTC-USD", side="buy", price=49000.0, size=0.1)
        await executor.place_order_async(order)

        # Cancel order
        cancel_result = await executor.cancel_order_async(order.order_id)
        cancel_success = cancel_result.run()

        assert cancel_success.is_right()
        assert cancel_success.value is True
        assert order.order_id in executor.get_cancelled_orders()

    @pytest.mark.asyncio
    async def test_concurrent_order_placement(self):
        """Test concurrent order placement."""
        executor = MockExchangeExecutor()

        orders = [
            MarketOrder(symbol="BTC-USD", side="buy", size=0.1),
            LimitOrder(symbol="ETH-USD", side="sell", price=3000.0, size=1.0),
            StopOrder(symbol="LTC-USD", side="buy", stop_price=150.0, size=2.0),
        ]

        # Place orders concurrently
        tasks = [executor.place_order_async(order) for order in orders]
        results = await asyncio.gather(*tasks)

        # All should succeed
        order_results = [result.run() for result in results]
        assert all(result.is_right() for result in order_results)
        assert len(executor.get_placed_orders()) == 3


class TestRiskManagementIntegration:
    """Test integration with risk management systems."""

    def test_order_validation_against_risk_limits(self):
        """Test order validation against risk limits."""
        risk_limits = create_conservative_risk_limits()

        # Valid order within limits
        valid_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=50000.0,
            size=0.1,
            leverage=3,  # Within max leverage of 5
            margin_required=Decimal("1666.67"),  # Within max position size
        )

        # Test validation function would be implemented in actual code
        # For now, we test the risk limits structure
        assert valid_order.leverage <= risk_limits.max_leverage
        assert valid_order.margin_required <= risk_limits.max_position_size

    def test_margin_requirement_validation(self):
        """Test margin requirement validation for futures orders."""
        margin_info = MarginInfo(
            total_margin=Decimal(10000),
            available_margin=Decimal(5000),
            used_margin=Decimal(5000),
            maintenance_margin=Decimal(2000),
            initial_margin=Decimal(3000),
            health_status=HEALTHY_MARGIN,
            liquidation_threshold=Decimal(1000),
            intraday_margin_requirement=Decimal(3000),
            overnight_margin_requirement=Decimal(4000),
        )

        # Order that fits within available margin
        affordable_order = FuturesMarketOrder(
            symbol="ETH-PERP",
            side="buy",
            size=1.0,
            leverage=5,
            margin_required=Decimal(600),  # Less than available 5000
        )

        assert margin_info.can_open_position(affordable_order.margin_required)

        # Order that exceeds available margin
        expensive_order = FuturesMarketOrder(
            symbol="BTC-PERP",
            side="buy",
            size=1.0,
            leverage=10,
            margin_required=Decimal(6000),  # More than available 5000
        )

        assert not margin_info.can_open_position(expensive_order.margin_required)

    def test_position_size_limits(self):
        """Test position size limits enforcement."""
        risk_limits = RiskLimits(
            max_position_size=Decimal(5000),
            max_daily_loss=Decimal(500),
            max_drawdown_percentage=10.0,
            max_leverage=5,
            max_open_positions=3,
            max_correlation_exposure=0.5,
            stop_loss_percentage=5.0,
            take_profit_percentage=15.0,
        )

        # Large order that exceeds position size limit
        large_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=50000.0,
            size=0.2,  # Would require 10000 margin at 1x, 2000 at 5x
            leverage=5,
            margin_required=Decimal(2000),
        )

        # Position value exceeds limits
        position_value = large_order.notional_value
        assert position_value > risk_limits.max_position_size


class TestOrderResultHandling:
    """Test order result handling and status tracking."""

    def test_order_result_creation(self):
        """Test OrderResult creation and validation."""
        result = OrderResult(
            order_id="order-123",
            status=OrderStatus.FILLED,
            filled_size=Decimal("0.5"),
            average_price=Decimal(50000),
            fees=Decimal("5.0"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert result.order_id == "order-123"
        assert result.status == OrderStatus.FILLED
        assert result.filled_size == Decimal("0.5")
        assert result.fees == Decimal("5.0")

    def test_order_result_validation(self):
        """Test OrderResult validation."""
        # Negative filled size
        with pytest.raises(ValueError, match="Filled size cannot be negative"):
            OrderResult(
                order_id="order-123",
                status=OrderStatus.FILLED,
                filled_size=Decimal("-0.1"),  # Invalid
                average_price=Decimal(50000),
                fees=Decimal("1.0"),
                created_at=datetime.now(),
            )

        # Negative fees
        with pytest.raises(ValueError, match="Fees cannot be negative"):
            OrderResult(
                order_id="order-123",
                status=OrderStatus.FILLED,
                filled_size=Decimal("0.1"),
                average_price=Decimal(50000),
                fees=Decimal("-1.0"),  # Invalid
                created_at=datetime.now(),
            )

    def test_order_status_tracking(self):
        """Test order status enumeration."""
        statuses = [
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

        assert len(statuses) == 6
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_partial_fill_handling(self):
        """Test handling of partial fills."""
        # Partial fill result
        partial_result = OrderResult(
            order_id="order-456",
            status=OrderStatus.OPEN,  # Still open for remaining fill
            filled_size=Decimal("0.5"),  # Partially filled
            average_price=Decimal(50000),
            fees=Decimal("2.5"),
            created_at=datetime.now(),
        )

        assert partial_result.status == OrderStatus.OPEN
        assert partial_result.filled_size < Decimal(
            "1.0"
        )  # Assuming original size was 1.0


@pytest.mark.asyncio
class TestOrderExecutionSafety:
    """Test order execution safety and error recovery."""

    async def test_order_execution_with_retries(self):
        """Test order execution with retry logic."""

        # Executor that fails first time, succeeds second time
        class RetryableExecutor:
            def __init__(self):
                self.attempt_count = 0

            async def place_order_async(
                self, order
            ) -> IOEither[Exception, OrderResult]:
                self.attempt_count += 1

                if self.attempt_count == 1:
                    return IOEither.left(Exception("Temporary network error"))

                # Succeed on second attempt
                result = OrderResult(
                    order_id=order.order_id,
                    status=OrderStatus.FILLED,
                    filled_size=Decimal(str(order.size)),
                    average_price=Decimal(50000),
                    fees=Decimal("1.0"),
                    created_at=datetime.now(),
                )
                return IOEither.right(result)

        executor = RetryableExecutor()
        order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)

        # First attempt - should fail
        first_result = await executor.place_order_async(order)
        assert first_result.run().is_left()

        # Second attempt - should succeed
        second_result = await executor.place_order_async(order)
        assert second_result.run().is_right()

    async def test_order_execution_timeout_handling(self):
        """Test handling of order execution timeouts."""

        class TimeoutExecutor:
            async def place_order_async(
                self, order
            ) -> IOEither[Exception, OrderResult]:
                # Simulate timeout
                await asyncio.sleep(0.1)
                return IOEither.left(Exception("Order execution timeout"))

        executor = TimeoutExecutor()
        order = LimitOrder(symbol="ETH-USD", side="buy", price=3000.0, size=1.0)

        start_time = datetime.now()
        result = await executor.place_order_async(order)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        assert duration >= 0.1  # Should have waited
        assert result.run().is_left()
        assert "timeout" in str(result.run().value).lower()

    async def test_order_validation_before_placement(self):
        """Test order validation before placement."""

        def validate_order(order) -> IOEither[Exception, bool]:
            """Validate order before placement."""
            try:
                # Basic validation
                if order.size <= 0:
                    raise ValueError("Order size must be positive")

                if hasattr(order, "price") and order.price <= 0:
                    raise ValueError("Order price must be positive")

                if order.symbol == "":
                    raise ValueError("Order symbol cannot be empty")

                return IOEither.right(True)

            except Exception as e:
                return IOEither.left(e)

        # Valid order
        valid_order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)
        validation_result = validate_order(valid_order)
        assert validation_result.run().is_right()

        # Invalid order (empty symbol)
        invalid_order = MarketOrder(symbol="", side="buy", size=0.1)
        validation_result = validate_order(invalid_order)
        assert validation_result.run().is_left()

    async def test_order_execution_rollback(self):
        """Test order execution rollback on failure."""

        class RollbackExecutor:
            def __init__(self):
                self.placed_orders = []
                self.rolled_back_orders = []

            async def place_order_async(
                self, order
            ) -> IOEither[Exception, OrderResult]:
                # Place order
                self.placed_orders.append(order)

                # Simulate failure after placement
                return IOEither.left(Exception("Post-placement failure"))

            async def rollback_order(self, order_id: str):
                """Rollback order placement."""
                self.rolled_back_orders.append(order_id)

        executor = RollbackExecutor()
        order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.1)

        # Attempt placement
        result = await executor.place_order_async(order)
        assert result.run().is_left()

        # Should have attempted placement
        assert len(executor.placed_orders) == 1

        # Perform rollback
        await executor.rollback_order(order.order_id)
        assert order.order_id in executor.rolled_back_orders


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
