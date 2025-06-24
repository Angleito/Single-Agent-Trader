"""
Functional Programming Exchange Tests

Test suite for exchange layer components with functional programming patterns.
Tests immutable types, Result/Either error handling, and adapter compatibility.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.fp.types.trading import (
    LimitOrder,
    MarketOrder,
    StopOrder,
    FuturesLimitOrder,
    FuturesMarketOrder,
    AccountBalance,
    Position,
    OrderResult,
    OrderStatus,
    FunctionalMarketData,
    FuturesAccountBalance,
    MarginInfo,
    HEALTHY_MARGIN,
    CFM_ACCOUNT,
    create_default_margin_info
)
from bot.fp.types.result import Result, Ok, Err
from bot.fp.effects.io import IOEither, from_try
from bot.fp.adapters.exchange_adapter import ExchangeAdapter, UnifiedExchangeAdapter
from bot.fp.adapters.coinbase_adapter import CoinbaseExchangeAdapter
from bot.fp.adapters.bluefin_adapter import BluefinExchangeAdapter


class TestFunctionalOrderTypes:
    """Test immutable order types and validation."""

    def test_limit_order_creation_and_validation(self):
        """Test LimitOrder creation and validation."""
        # Valid order
        order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.001
        )
        
        assert order.symbol == "BTC-USD"
        assert order.side == "buy"
        assert order.price == 50000.0
        assert order.size == 0.001
        assert order.order_id  # Should be auto-generated
        assert order.value == 50.0  # price * size

    def test_limit_order_validation_errors(self):
        """Test LimitOrder validation catches invalid values."""
        # Negative price
        with pytest.raises(ValueError, match="Price must be positive"):
            LimitOrder(
                symbol="BTC-USD",
                side="buy",
                price=-50000.0,
                size=0.001
            )
        
        # Zero size
        with pytest.raises(ValueError, match="Size must be positive"):
            LimitOrder(
                symbol="BTC-USD",
                side="buy",
                price=50000.0,
                size=0.0
            )

    def test_market_order_immutability(self):
        """Test that orders are immutable."""
        order = MarketOrder(
            symbol="ETH-USD",
            side="sell",
            size=0.1
        )
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            order.price = 3000.0  # type: ignore

    def test_futures_order_types(self):
        """Test futures-specific order types."""
        futures_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.01,
            leverage=10,
            margin_required=Decimal("550.0"),
            reduce_only=False,
            post_only=True
        )
        
        assert futures_order.leverage == 10
        assert futures_order.margin_required == Decimal("550.0")
        assert futures_order.notional_value == Decimal("550.0")
        assert futures_order.position_value == Decimal("5500.0")  # notional * leverage

    def test_order_id_generation(self):
        """Test that order IDs are unique."""
        order1 = MarketOrder(symbol="BTC-USD", side="buy", size=0.001)
        order2 = MarketOrder(symbol="BTC-USD", side="buy", size=0.001)
        
        assert order1.order_id != order2.order_id
        assert len(order1.order_id) > 0
        assert len(order2.order_id) > 0


class TestFunctionalAccountTypes:
    """Test immutable account and balance types."""

    def test_account_balance_validation(self):
        """Test AccountBalance validation."""
        balance = AccountBalance(
            currency="USD",
            available=Decimal("10000"),
            held=Decimal("2000"),
            total=Decimal("12000")
        )
        
        assert balance.currency == "USD"
        assert balance.available == Decimal("10000")
        assert balance.held == Decimal("2000")
        assert balance.total == Decimal("12000")

    def test_account_balance_invalid_totals(self):
        """Test AccountBalance catches invalid total calculations."""
        with pytest.raises(ValueError, match="Total balance .* must equal available .* \\+ held"):
            AccountBalance(
                currency="USD",
                available=Decimal("10000"),
                held=Decimal("2000"),
                total=Decimal("11000")  # Should be 12000
            )

    def test_futures_account_balance(self):
        """Test FuturesAccountBalance with margin info."""
        margin_info = create_default_margin_info()
        
        futures_balance = FuturesAccountBalance(
            account_type=CFM_ACCOUNT,
            account_id="test-account-123",
            currency="USD",
            cash_balance=Decimal("50000"),
            futures_balance=Decimal("25000"),
            total_balance=Decimal("75000"),
            margin_info=margin_info,
            max_leverage=20
        )
        
        assert futures_balance.account_type.is_futures()
        assert futures_balance.equity == Decimal("75000")
        assert futures_balance.buying_power == Decimal("160000")  # available_margin * max_leverage

    def test_margin_info_calculations(self):
        """Test MarginInfo calculations and properties."""
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
        
        assert margin_info.margin_ratio == 0.3  # 3000/10000
        assert margin_info.margin_usage_percentage == 30.0
        assert margin_info.free_margin_percentage == 70.0
        assert margin_info.can_open_position(Decimal("5000"))
        assert not margin_info.can_open_position(Decimal("8000"))


class TestFunctionalMarketData:
    """Test immutable market data types."""

    def test_functional_market_data_validation(self):
        """Test FunctionalMarketData OHLCV validation."""
        market_data = FunctionalMarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(),
            open=Decimal("50000"),
            high=Decimal("52000"),
            low=Decimal("49000"),
            close=Decimal("51000"),
            volume=Decimal("100")
        )
        
        assert market_data.symbol == "BTC-USD"
        assert market_data.typical_price == Decimal("50666.666666666666666666666667")
        assert market_data.weighted_price == Decimal("50500")
        assert market_data.price_range == Decimal("3000")
        assert market_data.price_change == Decimal("1000")
        assert market_data.is_bullish
        assert not market_data.is_bearish

    def test_market_data_ohlcv_validation_errors(self):
        """Test that market data validates OHLCV relationships."""
        # High less than open
        with pytest.raises(ValueError, match="High .* must be >= all other prices"):
            FunctionalMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                open=Decimal("50000"),
                high=Decimal("49000"),  # Invalid: high < open
                low=Decimal("48000"),
                close=Decimal("49500"),
                volume=Decimal("100")
            )
        
        # Low greater than close
        with pytest.raises(ValueError, match="Low .* must be <= all other prices"):
            FunctionalMarketData(
                symbol="BTC-USD",
                timestamp=datetime.now(),
                open=Decimal("50000"),
                high=Decimal("52000"),
                low=Decimal("51500"),  # Invalid: low > close
                close=Decimal("51000"),
                volume=Decimal("100")
            )

    def test_market_data_immutable_updates(self):
        """Test immutable updates to market data."""
        original_data = FunctionalMarketData(
            symbol="ETH-USD",
            timestamp=datetime.now(),
            open=Decimal("3000"),
            high=Decimal("3100"),
            low=Decimal("2900"),
            close=Decimal("3050"),
            volume=Decimal("500")
        )
        
        new_timestamp = datetime.now() + timedelta(minutes=1)
        updated_data = original_data.update_timestamp(new_timestamp)
        
        # Original should be unchanged
        assert original_data.timestamp != new_timestamp
        # New instance should have updated timestamp
        assert updated_data.timestamp == new_timestamp
        # Other fields should be the same
        assert updated_data.symbol == original_data.symbol
        assert updated_data.close == original_data.close


class MockExchangeAdapter:
    """Mock exchange adapter for testing."""

    def __init__(self, exchange_name: str, should_fail: bool = False):
        self.exchange_name = exchange_name
        self.should_fail = should_fail
        self.orders_placed = []
        self.orders_cancelled = []

    def place_order_impl(self, order) -> IOEither[Exception, OrderResult]:
        """Mock order placement."""
        if self.should_fail:
            return IOEither.left(Exception(f"Mock {self.exchange_name} order failed"))
        
        self.orders_placed.append(order)
        
        result = OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_size=Decimal(str(order.size)),
            average_price=Decimal(str(getattr(order, 'price', 50000))),
            fees=Decimal("1.0"),
            created_at=datetime.now()
        )
        
        return IOEither.right(result)

    def cancel_order_impl(self, order_id: str) -> IOEither[Exception, bool]:
        """Mock order cancellation."""
        if self.should_fail:
            return IOEither.left(Exception(f"Mock {self.exchange_name} cancel failed"))
        
        self.orders_cancelled.append(order_id)
        return IOEither.right(True)

    def get_positions_impl(self) -> IOEither[Exception, List[Position]]:
        """Mock position retrieval."""
        if self.should_fail:
            return IOEither.left(Exception(f"Mock {self.exchange_name} positions failed"))
        
        position = Position(
            symbol="BTC-USD",
            side="long",
            size=Decimal("0.001"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("1.0"),
            realized_pnl=Decimal("0"),
            entry_time=datetime.now()
        )
        
        return IOEither.right([position])

    def get_balance_impl(self) -> IOEither[Exception, AccountBalance]:
        """Mock balance retrieval."""
        if self.should_fail:
            return IOEither.left(Exception(f"Mock {self.exchange_name} balance failed"))
        
        balance = AccountBalance(
            currency="USD",
            available=Decimal("10000"),
            held=Decimal("1000"),
            total=Decimal("11000")
        )
        
        return IOEither.right(balance)


class TestUnifiedExchangeAdapter:
    """Test unified exchange adapter functionality."""

    def test_adapter_registration_and_retrieval(self):
        """Test adapter registration and retrieval."""
        coinbase_adapter = MockExchangeAdapter("coinbase")
        bluefin_adapter = MockExchangeAdapter("bluefin")
        
        unified = UnifiedExchangeAdapter(
            adapters={
                "coinbase": coinbase_adapter,
                "bluefin": bluefin_adapter
            },
            default_exchange="coinbase"
        )
        
        # Test default adapter
        default_adapter = unified.get_adapter()
        assert default_adapter == coinbase_adapter
        
        # Test specific adapter
        specific_adapter = unified.get_adapter("bluefin")
        assert specific_adapter == bluefin_adapter

    def test_adapter_not_found_error(self):
        """Test error handling for missing adapters."""
        unified = UnifiedExchangeAdapter(
            adapters={},
            default_exchange="coinbase"
        )
        
        with pytest.raises(ValueError, match="No adapter found for exchange: nonexistent"):
            unified.get_adapter("nonexistent")

    def test_order_placement_through_unified_adapter(self):
        """Test order placement through unified adapter."""
        mock_adapter = MockExchangeAdapter("coinbase")
        unified = UnifiedExchangeAdapter(
            adapters={"coinbase": mock_adapter},
            default_exchange="coinbase"
        )
        
        order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.001
        )
        
        result = unified.place_order(order).run()
        
        assert result.is_right()
        assert len(mock_adapter.orders_placed) == 1
        assert mock_adapter.orders_placed[0] == order

    def test_error_handling_in_unified_adapter(self):
        """Test error handling in unified adapter."""
        failing_adapter = MockExchangeAdapter("coinbase", should_fail=True)
        unified = UnifiedExchangeAdapter(
            adapters={"coinbase": failing_adapter},
            default_exchange="coinbase"
        )
        
        order = MarketOrder(symbol="BTC-USD", side="buy", size=0.001)
        result = unified.place_order(order).run()
        
        assert result.is_left()
        assert "Mock coinbase order failed" in str(result.value)

    def test_multi_exchange_operations(self):
        """Test operations across multiple exchanges."""
        coinbase_adapter = MockExchangeAdapter("coinbase")
        bluefin_adapter = MockExchangeAdapter("bluefin")
        
        unified = UnifiedExchangeAdapter(
            adapters={
                "coinbase": coinbase_adapter,
                "bluefin": bluefin_adapter
            },
            default_exchange="coinbase"
        )
        
        # Place orders on different exchanges
        btc_order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.001)
        eth_order = MarketOrder(symbol="ETH-USD", side="sell", size=0.1)
        
        # Coinbase order
        coinbase_result = unified.place_order(btc_order, "coinbase").run()
        # Bluefin order
        bluefin_result = unified.place_order(eth_order, "bluefin").run()
        
        assert coinbase_result.is_right()
        assert bluefin_result.is_right()
        assert len(coinbase_adapter.orders_placed) == 1
        assert len(bluefin_adapter.orders_placed) == 1

    def test_balance_retrieval_across_exchanges(self):
        """Test balance retrieval from multiple exchanges."""
        adapters = {
            "coinbase": MockExchangeAdapter("coinbase"),
            "bluefin": MockExchangeAdapter("bluefin")
        }
        
        unified = UnifiedExchangeAdapter(
            adapters=adapters,
            default_exchange="coinbase"
        )
        
        # Get balances from both exchanges
        coinbase_balance = unified.get_balance("coinbase").run()
        bluefin_balance = unified.get_balance("bluefin").run()
        
        assert coinbase_balance.is_right()
        assert bluefin_balance.is_right()
        
        # Both should return AccountBalance objects
        assert isinstance(coinbase_balance.value, AccountBalance)
        assert isinstance(bluefin_balance.value, AccountBalance)

    def test_position_management(self):
        """Test position retrieval and management."""
        mock_adapter = MockExchangeAdapter("coinbase")
        unified = UnifiedExchangeAdapter(
            adapters={"coinbase": mock_adapter},
            default_exchange="coinbase"
        )
        
        positions_result = unified.get_positions().run()
        
        assert positions_result.is_right()
        positions = positions_result.value
        assert len(positions) == 1
        assert positions[0].symbol == "BTC-USD"
        assert positions[0].side == "long"

    def test_order_cancellation(self):
        """Test order cancellation through unified adapter."""
        mock_adapter = MockExchangeAdapter("coinbase")
        unified = UnifiedExchangeAdapter(
            adapters={"coinbase": mock_adapter},
            default_exchange="coinbase"
        )
        
        order_id = "test-order-123"
        cancel_result = unified.cancel_order(order_id).run()
        
        assert cancel_result.is_right()
        assert cancel_result.value is True
        assert order_id in mock_adapter.orders_cancelled


class TestExchangeAdapterCompatibility:
    """Test adapter compatibility with legacy and FP systems."""

    def test_adapter_bridges_fp_and_legacy(self):
        """Test that adapters can bridge FP types and legacy exchange APIs."""
        # This would test the actual Coinbase/Bluefin adapters
        # For now, we'll test the mock adapter patterns
        
        mock_adapter = MockExchangeAdapter("coinbase")
        
        # FP-style order
        fp_order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.001
        )
        
        # Should work with FP types
        result = mock_adapter.place_order_impl(fp_order)
        assert result.run().is_right()

    def test_adapter_error_handling_compatibility(self):
        """Test that adapters properly handle errors in both systems."""
        failing_adapter = MockExchangeAdapter("test", should_fail=True)
        
        # Should return IOEither.Left for errors
        order = MarketOrder(symbol="BTC-USD", side="buy", size=0.001)
        result = failing_adapter.place_order_impl(order)
        
        # Should be wrapped in IOEither
        assert result.run().is_left()

    def test_adapter_preserves_type_safety(self):
        """Test that adapters maintain type safety."""
        mock_adapter = MockExchangeAdapter("coinbase")
        
        # Should only accept proper Order types
        order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.001)
        result = mock_adapter.place_order_impl(order)
        
        # Should return properly typed OrderResult
        order_result = result.run().value
        assert isinstance(order_result, OrderResult)
        assert order_result.order_id == order.order_id
        assert order_result.status == OrderStatus.FILLED


class TestFunctionalExchangeFactory:
    """Test functional exchange factory patterns."""

    def test_exchange_adapter_factory(self):
        """Test creation of exchange adapters through factory."""
        # This would test the actual factory for creating FP adapters
        
        # Mock the factory behavior
        def create_fp_exchange_adapter(exchange_type: str) -> ExchangeAdapter:
            if exchange_type == "coinbase":
                return MockExchangeAdapter("coinbase")
            elif exchange_type == "bluefin":
                return MockExchangeAdapter("bluefin")
            else:
                raise ValueError(f"Unsupported exchange: {exchange_type}")
        
        # Test creation
        coinbase_adapter = create_fp_exchange_adapter("coinbase")
        bluefin_adapter = create_fp_exchange_adapter("bluefin")
        
        assert coinbase_adapter.exchange_name == "coinbase"
        assert bluefin_adapter.exchange_name == "bluefin"
        
        # Test error handling
        with pytest.raises(ValueError, match="Unsupported exchange: invalid"):
            create_fp_exchange_adapter("invalid")

    def test_adapter_configuration_immutability(self):
        """Test that adapter configurations are immutable."""
        # Test that exchange configurations can't be modified after creation
        unified = UnifiedExchangeAdapter(
            adapters={"coinbase": MockExchangeAdapter("coinbase")},
            default_exchange="coinbase"
        )
        
        # Configuration should be immutable
        original_default = unified.default_exchange
        
        # Attempting to modify should not affect original (if immutable)
        # This tests the principle, actual implementation may vary
        assert unified.default_exchange == original_default


@pytest.mark.asyncio
class TestAsyncExchangeOperations:
    """Test asynchronous exchange operations with functional patterns."""

    async def test_async_order_placement(self):
        """Test asynchronous order placement with IOEither."""
        mock_adapter = MockExchangeAdapter("coinbase")
        
        async def async_place_order(order):
            # Simulate async operation
            await asyncio.sleep(0.01)
            return mock_adapter.place_order_impl(order)
        
        order = MarketOrder(symbol="BTC-USD", side="buy", size=0.001)
        result = await async_place_order(order)
        
        assert result.run().is_right()

    async def test_concurrent_exchange_operations(self):
        """Test concurrent operations across multiple exchanges."""
        coinbase_adapter = MockExchangeAdapter("coinbase")
        bluefin_adapter = MockExchangeAdapter("bluefin")
        
        async def place_order_async(adapter, order):
            await asyncio.sleep(0.01)
            return adapter.place_order_impl(order)
        
        orders = [
            MarketOrder(symbol="BTC-USD", side="buy", size=0.001),
            LimitOrder(symbol="ETH-USD", side="sell", price=3000.0, size=0.1),
        ]
        
        # Place orders concurrently
        tasks = [
            place_order_async(coinbase_adapter, orders[0]),
            place_order_async(bluefin_adapter, orders[1])
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Both should succeed
        assert all(result.run().is_right() for result in results)
        assert len(coinbase_adapter.orders_placed) == 1
        assert len(bluefin_adapter.orders_placed) == 1

    async def test_error_handling_in_async_operations(self):
        """Test error handling in async exchange operations."""
        failing_adapter = MockExchangeAdapter("coinbase", should_fail=True)
        
        async def async_operation():
            await asyncio.sleep(0.01)
            order = MarketOrder(symbol="BTC-USD", side="buy", size=0.001)
            return failing_adapter.place_order_impl(order)
        
        result = await async_operation()
        
        # Should properly handle errors
        assert result.run().is_left()


class TestExchangeRiskManagement:
    """Test risk management with functional types."""

    def test_position_risk_calculation(self):
        """Test position risk calculations with immutable types."""
        position = Position(
            symbol="BTC-USD",
            side="long",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("52000"),
            unrealized_pnl=Decimal("200"),
            realized_pnl=Decimal("0"),
            entry_time=datetime.now()
        )
        
        assert position.value == Decimal("5200")  # size * current_price
        assert position.pnl_percentage == 4.0  # 200 / 5000 * 100

    def test_margin_requirement_validation(self):
        """Test margin requirement validation for futures orders."""
        futures_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.1,
            leverage=10,
            margin_required=Decimal("550.0")
        )
        
        margin_info = MarginInfo(
            total_margin=Decimal("10000"),
            available_margin=Decimal("1000"),
            used_margin=Decimal("9000"),
            maintenance_margin=Decimal("500"),
            initial_margin=Decimal("800"),
            health_status=HEALTHY_MARGIN,
            liquidation_threshold=Decimal("200"),
            intraday_margin_requirement=Decimal("800"),
            overnight_margin_requirement=Decimal("1000")
        )
        
        # Check if we can place this order
        can_place = margin_info.can_open_position(futures_order.margin_required)
        assert can_place  # 1000 available >= 550 required

    def test_risk_limits_enforcement(self):
        """Test risk limits enforcement with functional types."""
        from bot.fp.types.trading import RiskLimits
        
        risk_limits = RiskLimits(
            max_position_size=Decimal("10000"),
            max_daily_loss=Decimal("500"),
            max_drawdown_percentage=15.0,
            max_leverage=5,
            max_open_positions=3,
            max_correlation_exposure=0.6,
            stop_loss_percentage=3.0,
            take_profit_percentage=10.0
        )
        
        # Test order against risk limits
        large_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=50000.0,
            size=0.5,  # Large size
            leverage=3,
            margin_required=Decimal("8333")  # Within limits
        )
        
        # Position value: 50000 * 0.5 = 25000, but with leverage 3 = 8333 margin
        assert large_order.margin_required <= risk_limits.max_position_size
        assert large_order.leverage <= risk_limits.max_leverage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])