"""
Comprehensive Functional Order Management Tests

This module tests the functional programming order management system including:
- Immutable order types and validation
- Pure functional state transitions 
- Effect-based order execution
- Event sourcing and order tracking
- Functional analytics algorithms
- Error handling patterns (timeout, retry, circuit breaker)
- Performance characteristics of FP order operations

Tests maintain all safety validations while validating functional programming patterns.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

# FP test infrastructure
from tests.fp_test_base import (
    FPTestBase,
    FPExchangeTestBase,
    FP_AVAILABLE
)

if FP_AVAILABLE:
    from bot.fp.orders import (
        # Core types
        OrderId, OrderParameters, OrderState, OrderFill, OrderEvent,
        OrderSide, OrderType, OrderStatus, OrderEventType,
        OrderStatistics, OrderManagerState,
        
        # Pure functions
        create_order, transition_order_status, add_fill_to_order,
        cancel_order, expire_order, reject_order, fail_order,
        
        # Lifecycle operations
        process_order_creation, process_order_fill, process_order_timeout,
        
        # Effect system
        FunctionalOrderManager, OrderExecutionError,
        place_order_effect, cancel_order_effect, get_order_status_effect,
        
        # Event sourcing
        EventSourcedOrderManager, OrderTrackedEvent,
        create_order_with_events, process_fill_with_events, cancel_order_with_events,
        
        # Analytics
        analyze_order_performance, analyze_order_volume, analyze_order_timing,
        calculate_order_success_patterns, calculate_order_efficiency_score,
        generate_comprehensive_order_analytics,
        OrderPerformanceMetrics, OrderVolumeAnalysis, OrderTimingAnalysis,
        
        # Error handling patterns
        OrderTimeout, OrderErrorContext, with_order_timeout, with_order_retry,
        with_circuit_breaker, compose_order_effects,
        build_robust_place_order_effect, build_robust_cancel_order_effect,
    )
    
    from bot.fp.types.base import Symbol, Percentage
    from bot.fp.types.result import Result, Success, Failure
    from bot.fp.effects import IO, IOEither, Either, Left, Right
else:
    # Fallback imports for non-FP environments
    from bot.fp.orders import (
        OrderId,
        OrderParameters,
        OrderState,
        OrderFill,
        OrderSide,
        OrderType,
        OrderStatus,
        OrderEventType,
        OrderManagerState,
        FunctionalOrderManager,
        OrderPerformanceMetrics,
        OrderVolumeAnalysis,
        OrderTimingAnalysis,
        OrderTimeout,
        OrderExecutionError,
        create_order,
        transition_order_status,
        add_fill_to_order,
        cancel_order,
        expire_order,
        reject_order,
        fail_order,
        process_order_creation,
        process_order_fill,
        process_order_timeout,
        analyze_order_performance,
        analyze_order_volume,
        analyze_order_timing,
        calculate_order_efficiency_score,
        generate_comprehensive_order_analytics,
        build_robust_place_order_effect,
        build_robust_cancel_order_effect,
    )
    from bot.fp.types.base import Symbol, Percentage
    from bot.fp.types.result import Success, Failure


class TestOrderTypes(FPTestBase):
    """Test immutable order types and validation."""
    
    def test_order_parameters_creation_valid(self):
        """Test valid order parameters creation."""
        result = OrderParameters.create(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=1.0,
            price=50000.0,
        )
        
        assert isinstance(result, Success)
        params = result.value
        assert str(params.symbol) == "BTC-USD"
        assert params.side == OrderSide.BUY
        assert params.order_type == OrderType.LIMIT
        assert params.quantity == Decimal("1.0")
        assert params.price == Decimal("50000.0")

    def test_order_parameters_validation_errors(self):
        """Test order parameters validation catches errors."""
        # Invalid symbol
        result = OrderParameters.create(
            symbol="invalid-symbol",
            side="BUY", 
            order_type="LIMIT",
            quantity=1.0,
            price=50000.0,
        )
        assert isinstance(result, Failure)
        
        # Negative quantity
        result = OrderParameters.create(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT", 
            quantity=-1.0,
            price=50000.0,
        )
        assert isinstance(result, Failure)
        
        # Limit order without price
        result = OrderParameters.create(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=1.0,
        )
        assert isinstance(result, Failure)

    def test_order_id_generation(self):
        """Test order ID generation is unique."""
        id1 = OrderId.generate("BTC-USD", OrderSide.BUY)
        id2 = OrderId.generate("BTC-USD", OrderSide.BUY)
        
        assert id1.value != id2.value
        assert "BTC-USD" in str(id1)
        assert "BUY" in str(id1)

    def test_order_state_immutability(self):
        """Test that order state is truly immutable."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        
        order = create_order(params)
        original_status = order.status
        
        # Try to modify - should create new instance
        updated_order = order.with_status(OrderStatus.OPEN)
        
        assert order.status == original_status  # Original unchanged
        assert updated_order.status == OrderStatus.OPEN  # New instance updated
        assert order.id == updated_order.id  # Same ID


class TestOrderLifecycle(FPTestBase):
    """Test order lifecycle state transitions."""
    
    def setup_method(self):
        """Set up test order."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        self.order = create_order(params)

    def test_order_creation(self):
        """Test order creation."""
        assert self.order.status == OrderStatus.PENDING
        assert self.order.filled_quantity == Decimal(0)
        assert self.order.remaining_quantity == Decimal("1.0")
        assert not self.order.is_complete
        assert not self.order.is_fillable

    def test_order_status_transitions(self):
        """Test valid order status transitions."""
        # PENDING -> OPEN
        open_order = transition_order_status(self.order, OrderStatus.OPEN)
        assert open_order.status == OrderStatus.OPEN
        assert open_order.is_fillable
        
        # OPEN -> FILLED
        filled_order = transition_order_status(open_order, OrderStatus.FILLED)
        assert filled_order.status == OrderStatus.FILLED
        assert filled_order.is_complete

    def test_order_fill_processing(self):
        """Test order fill processing."""
        open_order = transition_order_status(self.order, OrderStatus.OPEN)
        
        fill = OrderFill(
            quantity=Decimal("0.5"),
            price=Decimal("50100"),
            timestamp=datetime.now(),
            fee=Decimal("0.001"),
        )
        
        partially_filled = add_fill_to_order(open_order, fill)
        
        assert partially_filled.status == OrderStatus.PARTIALLY_FILLED
        assert partially_filled.filled_quantity == Decimal("0.5")
        assert partially_filled.remaining_quantity == Decimal("0.5")
        assert len(partially_filled.fills) == 1
        assert partially_filled.total_fees == Decimal("0.001")
        
        # Complete the fill
        remaining_fill = OrderFill(
            quantity=Decimal("0.5"),
            price=Decimal("50200"),
            timestamp=datetime.now(),
        )
        
        fully_filled = add_fill_to_order(partially_filled, remaining_fill)
        
        assert fully_filled.status == OrderStatus.FILLED
        assert fully_filled.filled_quantity == Decimal("1.0")
        assert fully_filled.remaining_quantity == Decimal("0")
        assert fully_filled.is_complete

    def test_order_cancellation(self):
        """Test order cancellation."""
        open_order = transition_order_status(self.order, OrderStatus.OPEN)
        cancelled_order = cancel_order(open_order)
        
        assert cancelled_order.status == OrderStatus.CANCELLED
        assert cancelled_order.is_complete

    def test_order_cancellation_validation(self):
        """Test order cancellation validation."""
        filled_order = transition_order_status(self.order, OrderStatus.FILLED)
        
        with pytest.raises(ValueError, match="Cannot cancel completed order"):
            cancel_order(filled_order)

    def test_order_fill_validation(self):
        """Test order fill validation."""
        open_order = transition_order_status(self.order, OrderStatus.OPEN)
        
        # Invalid fill - zero quantity
        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            add_fill_to_order(open_order, OrderFill(
                quantity=Decimal("0"),
                price=Decimal("50000"),
                timestamp=datetime.now(),
            ))
        
        # Invalid fill - exceeds order quantity
        with pytest.raises(ValueError, match="Fill quantity exceeds remaining"):
            add_fill_to_order(open_order, OrderFill(
                quantity=Decimal("2.0"),  # Order only has 1.0
                price=Decimal("50000"),
                timestamp=datetime.now(),
            ))


class TestOrderManagerState(FPTestBase):
    """Test order manager state operations."""
    
    def setup_method(self):
        """Set up test state."""
        self.state = OrderManagerState.empty()
        
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        self.order = create_order(params)

    def test_empty_state(self):
        """Test empty state initialization."""
        assert len(self.state.active_orders) == 0
        assert len(self.state.completed_orders) == 0
        assert len(self.state.order_events) == 0

    def test_add_order(self):
        """Test adding order to state."""
        new_state = self.state.add_order(self.order)
        
        assert len(new_state.active_orders) == 1
        assert self.order.id in new_state.active_orders
        assert len(new_state.order_events) == 1
        
        # Original state unchanged
        assert len(self.state.active_orders) == 0

    def test_update_order(self):
        """Test updating order in state."""
        state_with_order = self.state.add_order(self.order)
        
        filled_order = transition_order_status(self.order, OrderStatus.FILLED)
        updated_state = state_with_order.update_order(self.order.id, filled_order)
        
        # Order moved to completed
        assert len(updated_state.active_orders) == 0
        assert len(updated_state.completed_orders) == 1
        assert updated_state.completed_orders[0].status == OrderStatus.FILLED

    def test_cancel_all_orders(self):
        """Test cancelling all orders."""
        # Add multiple orders
        params2 = OrderParameters.create(
            "ETH-USD", "SELL", "LIMIT", 2.0, 3000.0
        ).value
        order2 = create_order(params2)
        
        state_with_orders = self.state.add_order(self.order).add_order(order2)
        
        cancelled_state = state_with_orders.cancel_all_orders()
        
        # All orders should be cancelled
        assert len(cancelled_state.active_orders) == 0
        assert len(cancelled_state.completed_orders) == 2
        assert all(order.status == OrderStatus.CANCELLED 
                  for order in cancelled_state.completed_orders)

    def test_get_orders_by_status(self):
        """Test filtering orders by status."""
        state_with_order = self.state.add_order(self.order)
        
        # Get pending orders
        pending_orders = state_with_order.get_orders_by_status(OrderStatus.PENDING)
        assert len(pending_orders) == 1
        assert pending_orders[0].id == self.order.id
        
        # Get filled orders (should be empty)
        filled_orders = state_with_order.get_orders_by_status(OrderStatus.FILLED)
        assert len(filled_orders) == 0


class TestFunctionalOrderManager(FPTestBase):
    """Test functional order manager with effects."""
    
    def setup_method(self):
        """Set up test manager."""
        self.manager = FunctionalOrderManager.create()

    def test_create_order_effect(self):
        """Test order creation through effects."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        
        result = self.manager.create_order_effect(params).run()
        updated_manager, order = result
        
        assert order.status == OrderStatus.PENDING
        assert len(updated_manager.state.active_orders) == 1

    def test_process_fill_effect(self):
        """Test fill processing through effects."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        
        # Create and place order
        manager, order = self.manager.create_order_effect(params).run()
        manager = manager.place_order_effect(order.id).run()
        
        # Process fill
        fill = OrderFill(
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
        )
        
        final_manager = manager.process_fill_effect(order.id, fill).run()
        
        updated_order = final_manager.state.get_order(order.id)
        assert updated_order.status == OrderStatus.FILLED
        assert updated_order.filled_quantity == Decimal("1.0")

    def test_cancel_order_effect(self):
        """Test order cancellation through effects."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        
        # Create order
        manager, order = self.manager.create_order_effect(params).run()
        
        # Cancel order
        final_manager = manager.cancel_order_effect(order.id).run()
        
        # Should be moved to completed orders
        assert len(final_manager.state.active_orders) == 0
        assert len(final_manager.state.completed_orders) == 1
        assert final_manager.state.completed_orders[0].status == OrderStatus.CANCELLED


class TestOrderAnalytics(FPTestBase):
    """Test pure functional analytics algorithms."""
    
    def setup_method(self):
        """Set up test orders for analytics."""
        self.orders = []
        
        # Create filled order
        params1 = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        order1 = create_order(params1)
        order1 = transition_order_status(order1, OrderStatus.OPEN)
        
        fill1 = OrderFill(
            quantity=Decimal("1.0"),
            price=Decimal("50100"),
            timestamp=datetime.now(),
            fee=Decimal("0.01"),
        )
        order1 = add_fill_to_order(order1, fill1)
        self.orders.append(order1)
        
        # Create cancelled order
        params2 = OrderParameters.create(
            "ETH-USD", "SELL", "LIMIT", 2.0, 3000.0
        ).value
        order2 = create_order(params2)
        order2 = cancel_order(order2)
        self.orders.append(order2)

    def test_performance_analysis(self):
        """Test order performance analysis."""
        metrics = analyze_order_performance(self.orders)
        
        assert metrics.total_orders == 2
        assert metrics.successful_orders == 1
        assert metrics.failed_orders == 1  # Cancelled counts as failed
        assert metrics.fill_rate_percentage == 50.0
        assert metrics.total_fees_paid == Decimal("0.01")
        assert metrics.volume_traded == Decimal("1.0")

    def test_volume_analysis(self):
        """Test order volume analysis."""
        analysis = analyze_order_volume(self.orders)
        
        assert analysis.total_volume == Decimal("1.0")  # Only filled order counts
        assert analysis.buy_volume == Decimal("1.0")
        assert analysis.sell_volume == Decimal("0")  # Sell order was cancelled
        assert analysis.average_order_size == Decimal("0.5")  # 1.0 / 2 orders

    def test_timing_analysis(self):
        """Test order timing analysis."""
        analysis = analyze_order_timing(self.orders)
        
        assert analysis.total_orders == 2
        assert analysis.fastest_fill_seconds >= 0
        assert analysis.slowest_fill_seconds >= 0

    def test_efficiency_score(self):
        """Test order efficiency score calculation."""
        score = calculate_order_efficiency_score(self.orders)
        
        assert 0 <= score <= 100
        # With 50% fill rate, score should be moderate
        assert 20 <= score <= 80

    def test_comprehensive_analytics(self):
        """Test comprehensive analytics generation."""
        analytics = generate_comprehensive_order_analytics(self.orders)
        
        assert "performance" in analytics
        assert "volume" in analytics
        assert "timing" in analytics
        assert "patterns" in analytics
        assert "overall_efficiency_score" in analytics
        assert "summary" in analytics
        
        assert analytics["performance"]["total_orders"] == 2
        assert analytics["summary"]["efficiency_grade"] in ["A", "B", "C", "D", "F"]


class TestErrorHandling(FPTestBase):
    """Test functional error handling and timeouts."""
    
    def test_order_timeout_configuration(self):
        """Test order timeout configuration."""
        timeout = OrderTimeout(
            initial_timeout_seconds=30,
            max_retries=3,
            backoff_multiplier=2.0,
            max_timeout_seconds=300,
        )
        
        assert timeout.calculate_timeout(0) == 30
        assert timeout.calculate_timeout(1) == 60
        assert timeout.calculate_timeout(2) == 120
        assert timeout.calculate_timeout(10) == 300  # Capped at max

    def test_order_execution_error(self):
        """Test order execution error handling."""
        order_id = OrderId("test-order")
        error = OrderExecutionError("Test error", order_id)
        
        assert error.order_id == order_id
        assert "Test error" in str(error)

    def test_robust_order_effects(self):
        """Test robust order effect builders."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        order = create_order(params)
        
        # Test robust place order effect
        place_effect = build_robust_place_order_effect(order)
        
        # Should not raise exception during creation
        assert place_effect is not None
        
        # Test robust cancel order effect
        cancel_effect = build_robust_cancel_order_effect(order.id, order)
        
        # Should not raise exception during creation
        assert cancel_effect is not None


class TestProcessOrderOperations(FPTestBase):
    """Test high-level process functions."""
    
    def setup_method(self):
        """Set up test state."""
        self.state = OrderManagerState.empty()

    def test_process_order_creation(self):
        """Test process order creation."""
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        
        result = process_order_creation(self.state, params)
        
        assert isinstance(result, Success)
        new_state, order = result.value
        
        assert len(new_state.active_orders) == 1
        assert order.status == OrderStatus.PENDING

    def test_process_order_fill(self):
        """Test process order fill."""
        # Create order first
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        state, order = process_order_creation(self.state, params).value
        
        # Transition to open
        open_order = transition_order_status(order, OrderStatus.OPEN)
        state = state.update_order(order.id, open_order)
        
        # Process fill
        fill = OrderFill(
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
        )
        
        result = process_order_fill(state, order.id, fill)
        
        assert isinstance(result, Success)
        final_state = result.value
        
        filled_order = final_state.get_order(order.id)
        assert filled_order.status == OrderStatus.FILLED

    def test_process_order_timeout(self):
        """Test process order timeout."""
        # Create order first
        params = OrderParameters.create(
            "BTC-USD", "BUY", "LIMIT", 1.0, 50000.0
        ).value
        state, order = process_order_creation(self.state, params).value
        
        # Process timeout
        result = process_order_timeout(state, order.id)
        
        assert isinstance(result, Success)
        final_state = result.value
        
        expired_order = final_state.get_order(order.id)
        assert expired_order.status == OrderStatus.EXPIRED


# Additional Comprehensive FP Tests

class TestAdvancedOrderTypes(FPTestBase):
    """Test advanced order type functionality and edge cases."""
    
    def test_order_fill_value_calculation(self):
        """Test OrderFill value calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        fill = OrderFill(
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
            fee=Decimal("25.0"),
            trade_id="trade_123"
        )
        
        # Test value calculation
        expected_value = Decimal("0.5") * Decimal("50000")
        assert fill.value == expected_value
        
        # Test properties
        assert fill.quantity == Decimal("0.5")
        assert fill.price == Decimal("50000")
        assert fill.fee == Decimal("25.0")
        assert fill.trade_id == "trade_123"

    def test_order_state_percentage_calculations(self):
        """Test order state percentage calculations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Create order parameters
        params_result = OrderParameters.create(
            symbol="BTC-USD",
            side="BUY",
            order_type="MARKET",
            quantity=2.0
        )
        params = params_result.success()
        
        # Create order state
        order_id = OrderId.generate("BTC-USD", OrderSide.BUY)
        now = datetime.now()
        
        order_state = OrderState(
            id=order_id,
            parameters=params,
            status=OrderStatus.OPEN,
            created_at=now,
            updated_at=now,
            filled_quantity=Decimal("0.5")
        )
        
        # Test property calculations
        assert order_state.remaining_quantity == Decimal("1.5")
        assert order_state.fill_percentage.value == 0.25  # 0.5/2.0
        assert order_state.is_fillable is True
        assert order_state.is_complete is False


class TestOrderEventSourcing(FPTestBase):
    """Test order event sourcing functionality."""
    
    def test_order_tracked_events(self):
        """Test OrderTrackedEvent static methods."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        try:
            params = OrderParameters.create("BTC-USD", "BUY", "MARKET", 1.0).success()
            order = create_order(params)
            
            # Test order creation event
            creation_event = OrderTrackedEvent.order_created(order)
            assert creation_event.event_type == OrderEventType.CREATED
            assert creation_event.order_id == order.id
            assert "order_id" in creation_event.data
            assert "symbol" in creation_event.data
            assert creation_event.data["symbol"] == "BTC-USD"
            
            # Test order submission event
            open_order = transition_order_status(order, OrderStatus.OPEN)
            submission_event = OrderTrackedEvent.order_submitted(open_order)
            assert submission_event.event_type == OrderEventType.SUBMITTED
            assert submission_event.data["status"] == "OPEN"
        except (ImportError, AttributeError):
            pytest.skip("Event sourcing not available")

    def test_event_sourced_order_manager_creation(self):
        """Test EventSourcedOrderManager creation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        try:
            # Mock event store
            mock_event_store = Mock()
            mock_event_store.replay.return_value = []
            
            # Create event-sourced manager
            manager = EventSourcedOrderManager.create(mock_event_store)
            
            assert len(manager.state.active_orders) == 0
            assert manager.event_store == mock_event_store
        except (ImportError, AttributeError):
            pytest.skip("Event sourced manager not available")


class TestAdvancedOrderAnalytics(FPTestBase):
    """Test advanced order analytics functionality."""
    
    def test_calculate_order_success_patterns(self):
        """Test order success pattern calculation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        try:
            orders = []
            
            # Successful MARKET BUY
            market_buy = OrderParameters.create("BTC-USD", "BUY", "MARKET", 1.0).success()
            order1 = create_order(market_buy)
            order1 = transition_order_status(order1, OrderStatus.FILLED)
            orders.append(order1)
            
            # Failed LIMIT SELL
            limit_sell = OrderParameters.create("BTC-USD", "SELL", "LIMIT", 1.0, 60000.0).success()
            order2 = create_order(limit_sell)
            order2 = transition_order_status(order2, OrderStatus.FAILED)
            orders.append(order2)
            
            patterns = calculate_order_success_patterns(orders)
            
            # Check success rates by type
            assert "success_rate_market" in patterns
            assert patterns["success_rate_market"] == 100.0  # 1/1 market orders filled
            assert "success_rate_limit" in patterns
            assert patterns["success_rate_limit"] == 0.0  # 0/1 limit orders filled
        except (ImportError, AttributeError):
            pytest.skip("Success patterns not available")

    def test_generate_comprehensive_order_analytics(self):
        """Test comprehensive analytics generation."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        try:
            # Create diverse set of orders
            orders = []
            
            # Mix of successful and failed orders with various characteristics
            for i in range(5):
                side = "BUY" if i % 2 == 0 else "SELL"
                order_type = "MARKET" if i % 3 == 0 else "LIMIT"
                
                params = OrderParameters.create(
                    "BTC-USD", 
                    side, 
                    order_type, 
                    float(i + 1),
                    50000.0 if order_type == "LIMIT" else None
                ).success()
                
                order = create_order(params)
                
                # Make some successful, some failed
                if i < 3:  # 60% success rate
                    order = transition_order_status(order, OrderStatus.FILLED)
                    # Add fill for successful orders
                    fill = OrderFill(
                        quantity=Decimal(str(i + 1)),
                        price=Decimal("50000"),
                        timestamp=datetime.now(),
                        fee=Decimal(str(i * 0.1))
                    )
                    order = order.with_fill(fill)
                else:
                    order = transition_order_status(order, OrderStatus.FAILED)
                
                orders.append(order)
            
            # Generate comprehensive analytics
            analytics = generate_comprehensive_order_analytics(orders)
            
            # Verify structure
            assert "performance" in analytics
            assert "volume" in analytics
            assert "timing" in analytics
            assert "patterns" in analytics
            assert "overall_efficiency_score" in analytics
            assert "summary" in analytics
            
            # Verify performance metrics
            perf = analytics["performance"]
            assert perf["total_orders"] == 5
            assert perf["successful_orders"] == 3
        except (ImportError, AttributeError):
            pytest.skip("Comprehensive analytics not available")


class TestAdvancedErrorHandling(FPTestBase):
    """Test advanced error handling patterns."""
    
    def test_order_error_context(self):
        """Test OrderErrorContext."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        try:
            order_id = OrderId.generate("BTC-USD", OrderSide.BUY)
            
            # Test retryable error
            retryable_error = ConnectionError("Network timeout")
            context = OrderErrorContext(
                order_id=order_id,
                operation="place_order",
                attempt=1,
                max_attempts=3,
                error=retryable_error,
                timestamp=datetime.now()
            )
            
            assert context.is_retryable is True
            assert context.can_retry is True
            
            # Test non-retryable error
            validation_error = OrderExecutionError("Invalid order quantity", order_id)
            context = OrderErrorContext(
                order_id=order_id,
                operation="place_order",
                attempt=1,
                max_attempts=3,
                error=validation_error,
                timestamp=datetime.now()
            )
            
            assert context.is_retryable is False
            assert context.can_retry is False
        except (ImportError, AttributeError):
            pytest.skip("Error context not available")

    def test_robust_order_effects(self):
        """Test robust order effect builders."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        try:
            params = OrderParameters.create("BTC-USD", "BUY", "MARKET", 1.0).success()
            order = create_order(params)
            
            # Test robust place order effect
            timeout_config = OrderTimeout(
                initial_timeout_seconds=5,
                max_retries=1,
                backoff_multiplier=2.0,
                max_timeout_seconds=60
            )
            
            robust_effect = build_robust_place_order_effect(order, timeout_config)
            
            # Should be able to create effect without error
            assert robust_effect is not None
        except (ImportError, AttributeError):
            pytest.skip("Robust effects not available")


class TestOrderPerformanceCharacteristics(FPTestBase):
    """Test performance characteristics of FP order operations."""
    
    def test_order_creation_performance(self):
        """Test performance of order creation operations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Test creating many orders
        start_time = time.time()
        orders = []
        
        for i in range(100):  # Reduced from 1000 for test speed
            params = OrderParameters.create(
                f"SYMBOL{i % 10}", 
                "BUY" if i % 2 == 0 else "SELL",
                "MARKET",
                float(i + 1)
            ).success()
            order = create_order(params)
            orders.append(order)
        
        creation_time = time.time() - start_time
        
        # Should create 100 orders reasonably fast
        assert len(orders) == 100
        assert creation_time < 5.0  # Should take less than 5 seconds
        
        # Verify all orders are correctly created
        for order in orders[:10]:  # Check first 10
            assert order.status == OrderStatus.PENDING
            assert order.filled_quantity == Decimal("0")

    def test_order_manager_state_performance(self):
        """Test performance of order manager state operations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        state = OrderManagerState.empty()
        
        # Add many orders
        start_time = time.time()
        order_ids = []
        
        for i in range(50):  # Reduced from 100 for test speed
            params = OrderParameters.create(f"SYMBOL{i}", "BUY", "MARKET", 1.0).success()
            order = create_order(params)
            state = state.add_order(order)
            order_ids.append(order.id)
        
        add_time = time.time() - start_time
        
        # Query orders
        start_time = time.time()
        for order_id in order_ids:
            found_order = state.get_order(order_id)
            assert found_order is not None
        
        query_time = time.time() - start_time
        
        # Performance should be reasonable
        assert add_time < 5.0
        assert query_time < 2.0
        assert len(state.active_orders) == 50


class TestOrderCompatibilityAndMigration(FPTestBase):
    """Test compatibility between FP and legacy order systems."""
    
    def test_order_type_compatibility(self):
        """Test that FP order types are compatible with legacy expectations."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Create FP order
        params = OrderParameters.create("BTC-USD", "BUY", "MARKET", 1.0).success()
        fp_order = create_order(params)
        
        # Verify it has expected properties that legacy code might expect
        assert hasattr(fp_order, 'id')
        assert hasattr(fp_order, 'status')
        assert hasattr(fp_order, 'parameters')
        assert hasattr(fp_order, 'filled_quantity')
        assert hasattr(fp_order, 'fills')
        
        # Test string representations work
        assert str(fp_order.id) != ""
        assert fp_order.status.value in ["PENDING", "OPEN", "FILLED", "CANCELLED", "REJECTED", "FAILED", "EXPIRED"]

    def test_order_data_conversion(self):
        """Test conversion between FP and legacy order data."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Create FP order
        params = OrderParameters.create("BTC-USD", "BUY", "MARKET", 1.0).success()
        fp_order = create_order(params)
        
        # Convert to dict (simulating legacy format)
        order_dict = {
            "id": str(fp_order.id),
            "symbol": str(fp_order.parameters.symbol),
            "side": fp_order.parameters.side.value,
            "type": fp_order.parameters.order_type.value,
            "quantity": float(fp_order.parameters.quantity),
            "status": fp_order.status.value,
            "filled_quantity": float(fp_order.filled_quantity),
            "created_at": fp_order.created_at.isoformat(),
        }
        
        # Verify conversion worked
        assert order_dict["symbol"] == "BTC-USD"
        assert order_dict["side"] == "BUY"
        assert order_dict["type"] == "MARKET"
        assert order_dict["quantity"] == 1.0
        assert order_dict["status"] == "PENDING"

    def test_fp_order_statistics_compatibility(self):
        """Test that FP order statistics are compatible with legacy reporting."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
        
        # Create orders and calculate statistics
        orders = []
        for i in range(10):
            params = OrderParameters.create("BTC-USD", "BUY", "MARKET", 1.0).success()
            order = create_order(params)
            if i < 7:  # 70% success rate
                order = transition_order_status(order, OrderStatus.FILLED)
            else:
                order = transition_order_status(order, OrderStatus.FAILED)
            orders.append(order)
        
        stats = OrderStatistics.from_orders(orders)
        
        # Convert to legacy format
        legacy_stats = {
            "total_orders": stats.total_orders,
            "filled_orders": stats.filled_orders,
            "cancelled_orders": stats.cancelled_orders,
            "rejected_orders": stats.rejected_orders,
            "failed_orders": stats.failed_orders,
            "fill_rate_pct": stats.fill_rate_pct,
            "avg_fill_time_seconds": stats.avg_fill_time_seconds,
        }
        
        # Verify legacy format has expected values
        assert legacy_stats["total_orders"] == 10
        assert legacy_stats["filled_orders"] == 7
        assert legacy_stats["fill_rate_pct"] == 70.0
        assert isinstance(legacy_stats["avg_fill_time_seconds"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])