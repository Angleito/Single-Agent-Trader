"""
Exchange Adapter Compatibility Tests

Test suite for exchange adapter compatibility between functional programming 
and legacy systems. Tests bidirectional conversion, type safety, and error handling.
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.fp.types.trading import (
    LimitOrder,
    MarketOrder,
    FuturesLimitOrder,
    OrderResult,
    OrderStatus,
    Position,
    AccountBalance,
    FuturesAccountBalance,
    FunctionalMarketData,
    convert_pydantic_to_functional_market_data,
    convert_pydantic_to_functional_position,
    convert_functional_to_pydantic_position,
    convert_order_to_functional
)
from bot.fp.types.result import Result, Ok, Err
from bot.fp.effects.io import IOEither
from bot.fp.adapters.exchange_adapter import ExchangeAdapter, UnifiedExchangeAdapter
from bot.fp.adapters.compatibility_layer import (
    LegacyExchangeWrapper,
    FunctionalExchangeWrapper,
    TypeConverter,
    create_compatibility_bridge
)
from bot.fp.adapters.type_converters import (
    convert_fp_order_to_legacy,
    convert_legacy_order_to_fp,
    convert_fp_position_to_legacy,
    convert_legacy_position_to_fp,
    convert_fp_market_data_to_legacy,
    convert_legacy_market_data_to_fp
)


# Mock Legacy Exchange Classes
class MockLegacyOrder:
    """Mock legacy order class."""
    
    def __init__(self, symbol: str, side: str, type: str, quantity: float, 
                 price: Optional[float] = None, id: Optional[str] = None):
        self.symbol = symbol
        self.side = side.upper()
        self.type = type.upper()
        self.quantity = quantity
        self.price = price
        self.id = id or f"legacy-{symbol}-{datetime.now().timestamp()}"


class MockLegacyPosition:
    """Mock legacy position class."""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float,
                 current_price: float, unrealized_pnl: float, realized_pnl: float = 0.0):
        self.symbol = symbol
        self.side = side.lower()
        self.size = size
        self.entry_price = entry_price
        self.current_price = current_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.timestamp = datetime.now()


class MockLegacyMarketData:
    """Mock legacy market data class."""
    
    def __init__(self, symbol: str, timestamp: datetime, open: float, high: float,
                 low: float, close: float, volume: float):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class MockLegacyExchange:
    """Mock legacy exchange for testing compatibility."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.placed_orders = []
        self.cancelled_orders = []
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: Optional[float] = None) -> MockLegacyOrder:
        """Mock legacy order placement."""
        if self.should_fail:
            raise Exception("Legacy exchange order failed")
        
        order = MockLegacyOrder(symbol, side, order_type, quantity, price)
        self.placed_orders.append(order)
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Mock legacy order cancellation."""
        if self.should_fail:
            raise Exception("Legacy exchange cancel failed")
        
        self.cancelled_orders.append(order_id)
        return True
    
    async def get_positions(self) -> List[MockLegacyPosition]:
        """Mock legacy position retrieval."""
        if self.should_fail:
            raise Exception("Legacy exchange positions failed")
        
        return [
            MockLegacyPosition("BTC-USD", "long", 0.1, 50000.0, 51000.0, 100.0),
            MockLegacyPosition("ETH-USD", "short", 1.0, 3000.0, 2950.0, 50.0)
        ]
    
    async def get_balance(self) -> Dict[str, float]:
        """Mock legacy balance retrieval."""
        if self.should_fail:
            raise Exception("Legacy exchange balance failed")
        
        return {
            "currency": "USD",
            "available": 10000.0,
            "held": 2000.0,
            "total": 12000.0
        }


class TestTypeConverters:
    """Test type conversion between FP and legacy systems."""

    def test_fp_order_to_legacy_conversion(self):
        """Test converting FP order to legacy format."""
        fp_order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.1,
            order_id="fp-order-123"
        )
        
        legacy_dict = convert_fp_order_to_legacy(fp_order)
        
        assert legacy_dict["symbol"] == "BTC-USD"
        assert legacy_dict["side"] == "BUY"
        assert legacy_dict["type"] == "LIMIT"
        assert legacy_dict["quantity"] == 0.1
        assert legacy_dict["price"] == 50000.0
        assert legacy_dict["id"] == "fp-order-123"

    def test_legacy_order_to_fp_conversion(self):
        """Test converting legacy order to FP format."""
        legacy_order = MockLegacyOrder(
            symbol="ETH-USD",
            side="SELL",
            type="MARKET",
            quantity=2.0,
            id="legacy-order-456"
        )
        
        fp_order = convert_legacy_order_to_fp(legacy_order)
        
        assert isinstance(fp_order, MarketOrder)
        assert fp_order.symbol == "ETH-USD"
        assert fp_order.side == "sell"
        assert fp_order.size == 2.0
        assert fp_order.order_id == "legacy-order-456"

    def test_fp_position_to_legacy_conversion(self):
        """Test converting FP position to legacy format."""
        fp_position = Position(
            symbol="LTC-USD",
            side="long",
            size=Decimal("5.0"),
            entry_price=Decimal("150.0"),
            current_price=Decimal("155.0"),
            unrealized_pnl=Decimal("25.0"),
            realized_pnl=Decimal("0.0"),
            entry_time=datetime.now()
        )
        
        legacy_dict = convert_fp_position_to_legacy(fp_position)
        
        assert legacy_dict["symbol"] == "LTC-USD"
        assert legacy_dict["side"] == "long"
        assert legacy_dict["size"] == 5.0
        assert legacy_dict["entry_price"] == 150.0
        assert legacy_dict["unrealized_pnl"] == 25.0

    def test_legacy_position_to_fp_conversion(self):
        """Test converting legacy position to FP format."""
        legacy_position = MockLegacyPosition(
            symbol="ADA-USD",
            side="short",
            size=1000.0,
            entry_price=0.5,
            current_price=0.48,
            unrealized_pnl=20.0
        )
        
        fp_position = convert_legacy_position_to_fp(legacy_position)
        
        assert isinstance(fp_position, Position)
        assert fp_position.symbol == "ADA-USD"
        assert fp_position.side == "short"
        assert fp_position.size == Decimal("1000.0")
        assert fp_position.entry_price == Decimal("0.5")
        assert fp_position.unrealized_pnl == Decimal("20.0")

    def test_market_data_conversion_roundtrip(self):
        """Test market data conversion round trip."""
        # FP to Legacy
        fp_data = FunctionalMarketData(
            symbol="DOT-USD",
            timestamp=datetime.now(),
            open=Decimal("25.0"),
            high=Decimal("26.0"),
            low=Decimal("24.5"),
            close=Decimal("25.5"),
            volume=Decimal("1000.0")
        )
        
        legacy_dict = convert_fp_market_data_to_legacy(fp_data)
        
        assert legacy_dict["symbol"] == "DOT-USD"
        assert legacy_dict["open"] == 25.0
        assert legacy_dict["high"] == 26.0
        assert legacy_dict["close"] == 25.5
        
        # Legacy to FP
        legacy_data = MockLegacyMarketData(**legacy_dict)
        fp_data_converted = convert_legacy_market_data_to_fp(legacy_data)
        
        assert fp_data_converted.symbol == fp_data.symbol
        assert fp_data_converted.open == fp_data.open
        assert fp_data_converted.close == fp_data.close

    def test_conversion_error_handling(self):
        """Test error handling in type conversion."""
        # Invalid order type
        invalid_order = MockLegacyOrder(
            symbol="BTC-USD",
            side="BUY",
            type="INVALID_TYPE",
            quantity=0.1
        )
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError, match="Unsupported order type"):
            convert_legacy_order_to_fp(invalid_order)

    def test_futures_order_conversion(self):
        """Test conversion of futures orders."""
        fp_futures_order = FuturesLimitOrder(
            symbol="BTC-PERP",
            side="buy",
            price=55000.0,
            size=0.1,
            leverage=10,
            margin_required=Decimal("550.0"),
            reduce_only=True
        )
        
        legacy_dict = convert_fp_order_to_legacy(fp_futures_order)
        
        assert legacy_dict["symbol"] == "BTC-PERP"
        assert legacy_dict["leverage"] == 10
        assert legacy_dict["margin_required"] == 550.0
        assert legacy_dict["reduce_only"] is True


class TestLegacyExchangeWrapper:
    """Test wrapper that adapts legacy exchange to FP interface."""

    def test_legacy_wrapper_creation(self):
        """Test creation of legacy exchange wrapper."""
        legacy_exchange = MockLegacyExchange()
        wrapper = LegacyExchangeWrapper(legacy_exchange)
        
        assert wrapper.legacy_exchange == legacy_exchange

    @pytest.mark.asyncio
    async def test_legacy_wrapper_order_placement(self):
        """Test order placement through legacy wrapper."""
        legacy_exchange = MockLegacyExchange()
        wrapper = LegacyExchangeWrapper(legacy_exchange)
        
        fp_order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.1
        )
        
        result = await wrapper.place_order_impl(fp_order)
        order_result = result.run()
        
        assert order_result.is_right()
        assert len(legacy_exchange.placed_orders) == 1
        
        placed_order = legacy_exchange.placed_orders[0]
        assert placed_order.symbol == "BTC-USD"
        assert placed_order.side == "BUY"
        assert placed_order.quantity == 0.1

    @pytest.mark.asyncio
    async def test_legacy_wrapper_error_handling(self):
        """Test error handling in legacy wrapper."""
        failing_exchange = MockLegacyExchange(should_fail=True)
        wrapper = LegacyExchangeWrapper(failing_exchange)
        
        fp_order = MarketOrder(symbol="ETH-USD", side="sell", size=1.0)
        
        result = await wrapper.place_order_impl(fp_order)
        order_result = result.run()
        
        assert order_result.is_left()
        assert "Legacy exchange order failed" in str(order_result.value)

    @pytest.mark.asyncio
    async def test_legacy_wrapper_position_retrieval(self):
        """Test position retrieval through legacy wrapper."""
        legacy_exchange = MockLegacyExchange()
        wrapper = LegacyExchangeWrapper(legacy_exchange)
        
        result = await wrapper.get_positions_impl()
        positions_result = result.run()
        
        assert positions_result.is_right()
        positions = positions_result.value
        
        assert len(positions) == 2
        assert all(isinstance(pos, Position) for pos in positions)
        assert positions[0].symbol == "BTC-USD"
        assert positions[1].symbol == "ETH-USD"

    @pytest.mark.asyncio
    async def test_legacy_wrapper_balance_retrieval(self):
        """Test balance retrieval through legacy wrapper."""
        legacy_exchange = MockLegacyExchange()
        wrapper = LegacyExchangeWrapper(legacy_exchange)
        
        result = await wrapper.get_balance_impl()
        balance_result = result.run()
        
        assert balance_result.is_right()
        balance = balance_result.value
        
        assert isinstance(balance, AccountBalance)
        assert balance.currency == "USD"
        assert balance.available == Decimal("10000.0")
        assert balance.total == Decimal("12000.0")


class TestFunctionalExchangeWrapper:
    """Test wrapper that adapts FP exchange to legacy interface."""

    def test_functional_wrapper_creation(self):
        """Test creation of functional exchange wrapper."""
        from bot.fp.adapters.exchange_adapter import MockExchangeAdapter
        fp_adapter = MockExchangeAdapter("test", should_fail=False)
        wrapper = FunctionalExchangeWrapper(fp_adapter)
        
        assert wrapper.fp_adapter == fp_adapter

    @pytest.mark.asyncio
    async def test_functional_wrapper_legacy_interface(self):
        """Test legacy interface methods on functional wrapper."""
        from bot.fp.adapters.exchange_adapter import MockExchangeAdapter
        fp_adapter = MockExchangeAdapter("test", should_fail=False)
        wrapper = FunctionalExchangeWrapper(fp_adapter)
        
        # Place order using legacy interface
        legacy_order = await wrapper.place_order_legacy(
            symbol="BTC-USD",
            side="BUY",
            order_type="MARKET",
            quantity=0.1
        )
        
        assert legacy_order.symbol == "BTC-USD"
        assert legacy_order.side == "BUY"
        assert legacy_order.quantity == 0.1

    @pytest.mark.asyncio
    async def test_functional_wrapper_error_conversion(self):
        """Test error conversion from FP to legacy format."""
        from bot.fp.adapters.exchange_adapter import MockExchangeAdapter
        failing_adapter = MockExchangeAdapter("test", should_fail=True)
        wrapper = FunctionalExchangeWrapper(failing_adapter)
        
        # Should convert FP errors to legacy exceptions
        with pytest.raises(Exception, match="Mock test order failed"):
            await wrapper.place_order_legacy(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.1
            )


class TestCompatibilityBridge:
    """Test the compatibility bridge that handles both systems."""

    def test_compatibility_bridge_creation(self):
        """Test creation of compatibility bridge."""
        bridge = create_compatibility_bridge()
        
        assert bridge is not None
        assert hasattr(bridge, 'register_legacy_exchange')
        assert hasattr(bridge, 'register_fp_exchange')

    def test_bridge_type_detection(self):
        """Test automatic type detection in bridge."""
        bridge = create_compatibility_bridge()
        
        # FP order
        fp_order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.1)
        assert bridge.is_fp_type(fp_order)
        assert not bridge.is_legacy_type(fp_order)
        
        # Legacy order
        legacy_order = MockLegacyOrder("ETH-USD", "SELL", "MARKET", 1.0)
        assert bridge.is_legacy_type(legacy_order)
        assert not bridge.is_fp_type(legacy_order)

    def test_bridge_automatic_conversion(self):
        """Test automatic conversion by bridge."""
        bridge = create_compatibility_bridge()
        
        # Convert FP to legacy
        fp_order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)
        legacy_converted = bridge.convert_to_legacy(fp_order)
        
        assert legacy_converted["symbol"] == "BTC-USD"
        assert legacy_converted["side"] == "BUY"
        assert legacy_converted["type"] == "MARKET"
        
        # Convert legacy to FP
        legacy_order = MockLegacyOrder("ETH-USD", "SELL", "LIMIT", 1.0, 3000.0)
        fp_converted = bridge.convert_to_fp(legacy_order)
        
        assert isinstance(fp_converted, LimitOrder)
        assert fp_converted.symbol == "ETH-USD"
        assert fp_converted.side == "sell"

    @pytest.mark.asyncio
    async def test_bridge_unified_interface(self):
        """Test unified interface that works with both systems."""
        bridge = create_compatibility_bridge()
        
        # Register both types of exchanges
        legacy_exchange = MockLegacyExchange()
        from bot.fp.adapters.exchange_adapter import MockExchangeAdapter
        fp_adapter = MockExchangeAdapter("test")
        
        bridge.register_legacy_exchange("legacy", legacy_exchange)
        bridge.register_fp_exchange("fp", fp_adapter)
        
        # Place order through unified interface
        fp_order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)
        
        # Should work with both exchanges
        legacy_result = await bridge.place_order_unified("legacy", fp_order)
        fp_result = await bridge.place_order_unified("fp", fp_order)
        
        assert legacy_result.is_right()
        assert fp_result.is_right()


class TestBidirectionalCompatibility:
    """Test bidirectional compatibility between systems."""

    @pytest.mark.asyncio
    async def test_fp_to_legacy_to_fp_roundtrip(self):
        """Test round trip from FP to legacy and back to FP."""
        # Start with FP order
        original_order = LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=50000.0,
            size=0.1,
            order_id="original-order-123"
        )
        
        # Convert to legacy
        legacy_dict = convert_fp_order_to_legacy(original_order)
        legacy_order = MockLegacyOrder(**legacy_dict)
        
        # Convert back to FP
        roundtrip_order = convert_legacy_order_to_fp(legacy_order)
        
        # Should be equivalent
        assert roundtrip_order.symbol == original_order.symbol
        assert roundtrip_order.side == original_order.side
        assert roundtrip_order.price == original_order.price
        assert roundtrip_order.size == original_order.size

    @pytest.mark.asyncio
    async def test_legacy_to_fp_to_legacy_roundtrip(self):
        """Test round trip from legacy to FP and back to legacy."""
        # Start with legacy order
        original_order = MockLegacyOrder(
            symbol="ETH-USD",
            side="SELL",
            type="LIMIT",
            quantity=2.0,
            price=3000.0,
            id="legacy-order-456"
        )
        
        # Convert to FP
        fp_order = convert_legacy_order_to_fp(original_order)
        
        # Convert back to legacy
        roundtrip_dict = convert_fp_order_to_legacy(fp_order)
        
        # Should be equivalent
        assert roundtrip_dict["symbol"] == original_order.symbol
        assert roundtrip_dict["side"] == original_order.side
        assert roundtrip_dict["quantity"] == original_order.quantity
        assert roundtrip_dict["price"] == original_order.price

    @pytest.mark.asyncio
    async def test_mixed_system_workflow(self):
        """Test workflow using both FP and legacy systems."""
        # Initialize both systems
        legacy_exchange = MockLegacyExchange()
        legacy_wrapper = LegacyExchangeWrapper(legacy_exchange)
        
        from bot.fp.adapters.exchange_adapter import MockExchangeAdapter
        fp_adapter = MockExchangeAdapter("test")
        
        # Create order in FP format
        fp_order = MarketOrder(symbol="BTC-USD", side="buy", size=0.1)
        
        # Place through legacy system
        legacy_result = await legacy_wrapper.place_order_impl(fp_order)
        assert legacy_result.run().is_right()
        
        # Place through FP system
        fp_result = await fp_adapter.place_order_impl(fp_order)
        assert fp_result.run().is_right()
        
        # Both should have processed the same order
        assert len(legacy_exchange.placed_orders) == 1
        assert len(fp_adapter.orders_placed) == 1

    def test_type_safety_preservation(self):
        """Test that type safety is preserved across conversions."""
        # FP system enforces immutability
        fp_order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.1)
        
        with pytest.raises(AttributeError):
            fp_order.price = 51000.0  # type: ignore
        
        # Conversion should maintain data integrity
        legacy_dict = convert_fp_order_to_legacy(fp_order)
        converted_back = convert_legacy_order_to_fp(MockLegacyOrder(**legacy_dict))
        
        # Converted order should also be immutable
        with pytest.raises(AttributeError):
            converted_back.price = 51000.0  # type: ignore

    def test_error_handling_compatibility(self):
        """Test error handling across system boundaries."""
        # FP errors should convert to legacy exceptions
        fp_error = IOEither.left(ValueError("FP validation error"))
        
        def convert_fp_error_to_legacy(fp_result: IOEither) -> None:
            result = fp_result.run()
            if result.is_left():
                raise Exception(f"Legacy: {result.value}")
        
        with pytest.raises(Exception, match="Legacy: FP validation error"):
            convert_fp_error_to_legacy(fp_error)
        
        # Legacy exceptions should convert to FP errors
        def convert_legacy_error_to_fp(operation) -> IOEither:
            try:
                result = operation()
                return IOEither.right(result)
            except Exception as e:
                return IOEither.left(e)
        
        def failing_operation():
            raise ValueError("Legacy operation failed")
        
        fp_result = convert_legacy_error_to_fp(failing_operation)
        assert fp_result.run().is_left()
        assert "Legacy operation failed" in str(fp_result.run().value)


class TestPerformanceCompatibility:
    """Test performance implications of compatibility layer."""

    @pytest.mark.asyncio
    async def test_conversion_overhead(self):
        """Test overhead of type conversion."""
        import time
        
        # Measure direct FP operation
        start_time = time.time()
        
        for _ in range(1000):
            fp_order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.1)
        
        fp_time = time.time() - start_time
        
        # Measure conversion overhead
        start_time = time.time()
        
        for _ in range(1000):
            fp_order = LimitOrder(symbol="BTC-USD", side="buy", price=50000.0, size=0.1)
            legacy_dict = convert_fp_order_to_legacy(fp_order)
            converted_back = convert_legacy_order_to_fp(MockLegacyOrder(**legacy_dict))
        
        conversion_time = time.time() - start_time
        
        # Conversion overhead should be reasonable
        overhead_ratio = conversion_time / fp_time
        assert overhead_ratio < 5.0  # Less than 5x overhead

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Test concurrent operations across both systems."""
        legacy_exchange = MockLegacyExchange()
        legacy_wrapper = LegacyExchangeWrapper(legacy_exchange)
        
        from bot.fp.adapters.exchange_adapter import MockExchangeAdapter
        fp_adapter = MockExchangeAdapter("test")
        
        # Create multiple orders
        orders = [
            LimitOrder(symbol="BTC-USD", side="buy", price=50000.0 + i*100, size=0.1)
            for i in range(10)
        ]
        
        # Place half through legacy, half through FP
        legacy_tasks = [
            legacy_wrapper.place_order_impl(order) 
            for order in orders[:5]
        ]
        
        fp_tasks = [
            fp_adapter.place_order_impl(order)
            for order in orders[5:]
        ]
        
        # Execute concurrently
        legacy_results = await asyncio.gather(*legacy_tasks)
        fp_results = await asyncio.gather(*fp_tasks)
        
        # All should succeed
        assert all(result.run().is_right() for result in legacy_results)
        assert all(result.run().is_right() for result in fp_results)
        
        assert len(legacy_exchange.placed_orders) == 5
        assert len(fp_adapter.orders_placed) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])