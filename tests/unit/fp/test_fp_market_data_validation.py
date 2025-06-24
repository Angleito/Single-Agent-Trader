"""
Functional Programming Market Data Validation Tests

This module tests the immutable market data types, validation functions,
and data processing pipelines using functional programming patterns.
"""

import pytest
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Any, List
from unittest.mock import MagicMock, patch

from bot.fp.types.market import (
    Candle,
    Trade,
    Ticker,
    OrderBook,
    MarketSnapshot,
    OHLCV,
    WebSocketMessage,
    TickerMessage,
    TradeMessage,
    OrderBookMessage,
    ConnectionState,
    ConnectionStatus,
    DataQuality,
    RealtimeUpdate,
    AggregatedData,
    MarketDataStream,
)
from bot.fp.core.either import Either, Left, Right
from bot.fp.core.option import Option, Some, None_ as NoneOption
from bot.fp.adapters.type_converters import (
    current_market_data_to_fp_candle,
    validate_functional_candle,
    validate_trade_data,
    validate_ticker_data,
    validate_orderbook_data,
    create_connection_state,
    create_data_quality,
    validate_connection_health,
    validate_data_quality,
)
from bot.fp.effects.market_data import (
    validate_market_data_effect,
    process_candle_effect,
    aggregate_trades_effect,
)
from bot.trading_types import MarketData as CurrentMarketData


class TestImmutableMarketDataTypes:
    """Test immutable market data types and their invariants."""
    
    def test_candle_immutability(self):
        """Test that Candle objects are immutable."""
        candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
            symbol="BTC-USD"
        )
        
        # Candle should be frozen
        with pytest.raises(AttributeError):
            candle.open = Decimal("49000")  # type: ignore
        
        # Properties should work correctly
        assert candle.is_bullish
        assert not candle.is_bearish
        assert candle.price_range == Decimal("1500")
        assert candle.body_size == Decimal("500")
        assert candle.upper_shadow == Decimal("500")
        assert candle.lower_shadow == Decimal("0")
    
    def test_candle_validation(self):
        """Test candle validation rules."""
        # Valid candle
        valid_candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
            symbol="BTC-USD"
        )
        assert validate_functional_candle(valid_candle)
        
        # Invalid candle - negative price
        with pytest.raises(ValueError, match="All prices must be positive"):
            Candle(
                timestamp=datetime.now(UTC),
                open=Decimal("-1"),
                high=Decimal("51000"),
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("100.5"),
                symbol="BTC-USD"
            )
        
        # Invalid candle - high < close
        with pytest.raises(ValueError, match="High .* must be >= all other prices"):
            Candle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("49000"),  # Lower than close
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("100.5"),
                symbol="BTC-USD"
            )
        
        # Invalid candle - negative volume
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Candle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("-10"),
                symbol="BTC-USD"
            )
    
    def test_trade_immutability_and_validation(self):
        """Test Trade immutability and validation."""
        trade = Trade(
            id="trade-123",
            timestamp=datetime.now(UTC),
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY",
            symbol="BTC-USD"
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            trade.price = Decimal("51000")  # type: ignore
        
        # Properties should work
        assert trade.value == Decimal("25000")  # price * size
        assert trade.is_buy()
        assert not trade.is_sell()
        
        # Validation should work
        assert validate_trade_data(trade)
        
        # Invalid trade - negative price
        with pytest.raises(ValueError, match="Price must be positive"):
            Trade(
                id="trade-456",
                timestamp=datetime.now(UTC),
                price=Decimal("-100"),
                size=Decimal("0.5"),
                side="BUY"
            )
        
        # Invalid trade - invalid side
        with pytest.raises(ValueError, match="Side must be BUY or SELL"):
            Trade(
                id="trade-789",
                timestamp=datetime.now(UTC),
                price=Decimal("50000"),
                size=Decimal("0.5"),
                side="INVALID"
            )
    
    def test_orderbook_immutability_and_validation(self):
        """Test OrderBook immutability and validation."""
        bids = [
            (Decimal("49990"), Decimal("1.0")),
            (Decimal("49980"), Decimal("2.0")),
            (Decimal("49970"), Decimal("1.5")),
        ]
        asks = [
            (Decimal("50010"), Decimal("1.2")),
            (Decimal("50020"), Decimal("0.8")),
            (Decimal("50030"), Decimal("2.1")),
        ]
        
        orderbook = OrderBook(
            bids=bids,
            asks=asks,
            timestamp=datetime.now(UTC)
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            orderbook.bids = []  # type: ignore
        
        # Properties should work
        assert orderbook.best_bid == (Decimal("49990"), Decimal("1.0"))
        assert orderbook.best_ask == (Decimal("50010"), Decimal("1.2"))
        assert orderbook.mid_price == Decimal("50000")
        assert orderbook.spread == Decimal("20")
        assert orderbook.bid_depth == Decimal("4.5")
        assert orderbook.ask_depth == Decimal("4.1")
        
        # Validation should work
        assert validate_orderbook_data(orderbook)
        
        # Test price impact calculation
        buy_impact = orderbook.price_impact("buy", Decimal("1.0"))
        assert buy_impact == Decimal("50010")  # Fills at best ask
        
        # Invalid orderbook - best bid >= best ask
        with pytest.raises(ValueError, match="Best bid .* must be < best ask"):
            OrderBook(
                bids=[(Decimal("50020"), Decimal("1.0"))],
                asks=[(Decimal("50010"), Decimal("1.0"))],
                timestamp=datetime.now(UTC)
            )
    
    def test_ticker_validation(self):
        """Test Ticker validation."""
        ticker = Ticker(
            symbol="BTC-USD",
            last_price=Decimal("50000"),
            volume_24h=Decimal("1000"),
            change_24h=Decimal("2.5")
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            ticker.last_price = Decimal("51000")  # type: ignore
        
        # Properties should work
        assert ticker.price_24h_ago == Decimal("48780.487804878048780488")  # Approximately
        assert ticker.is_positive_24h
        assert ticker.volatility_24h == Decimal("2.5")
        
        # Validation should work
        assert validate_ticker_data(ticker)
        
        # Invalid ticker
        with pytest.raises(ValueError, match="24h change cannot be less than -100%"):
            Ticker(
                symbol="BTC-USD",
                last_price=Decimal("50000"),
                volume_24h=Decimal("1000"),
                change_24h=Decimal("-150")  # More than -100%
            )


class TestFunctionalDataProcessing:
    """Test functional data processing patterns."""
    
    def test_market_data_effect_validation(self):
        """Test market data validation using effects."""
        # Valid market data
        valid_candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
            symbol="BTC-USD"
        )
        
        # Process using effect
        result = validate_market_data_effect(valid_candle)
        assert isinstance(result, Right)
        assert result.value == valid_candle
        
        # Invalid market data should return Left
        try:
            invalid_candle = Candle(
                timestamp=datetime.now(UTC),
                open=Decimal("50000"),
                high=Decimal("49000"),  # Invalid: high < open
                low=Decimal("49500"),
                close=Decimal("50500"),
                volume=Decimal("100.5"),
                symbol="BTC-USD"
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            # Expected - invalid candle can't be created
            pass
    
    def test_candle_processing_effect(self):
        """Test candle processing using effects."""
        candle = Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5"),
            symbol="BTC-USD"
        )
        
        # Process candle
        result = process_candle_effect(candle)
        assert isinstance(result, Right)
        
        processed_candle = result.value
        assert processed_candle.symbol == "BTC-USD"
        assert processed_candle.is_bullish
    
    def test_trade_aggregation_effect(self):
        """Test trade aggregation using effects."""
        trades = [
            Trade(
                id="1", timestamp=datetime.now(UTC), price=Decimal("50000"),
                size=Decimal("0.1"), side="BUY", symbol="BTC-USD"
            ),
            Trade(
                id="2", timestamp=datetime.now(UTC), price=Decimal("50100"),
                size=Decimal("0.2"), side="SELL", symbol="BTC-USD"
            ),
            Trade(
                id="3", timestamp=datetime.now(UTC), price=Decimal("50050"),
                size=Decimal("0.15"), side="BUY", symbol="BTC-USD"
            ),
        ]
        
        # Aggregate trades
        result = aggregate_trades_effect(trades, timedelta(minutes=1))
        assert isinstance(result, Right)
        
        aggregated = result.value
        assert aggregated.symbol == "BTC-USD"
        assert aggregated.trade_count == 3
        assert aggregated.volume_total == Decimal("0.45")
        assert aggregated.average_trade_size == Decimal("0.15")


class TestWebSocketMessageTypes:
    """Test immutable WebSocket message types."""
    
    def test_websocket_message_validation(self):
        """Test WebSocket message validation."""
        message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"price": "50000", "volume": "100"},
            message_id="msg-123"
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            message.channel = "trades"  # type: ignore
        
        # Invalid message
        with pytest.raises(ValueError, match="Channel cannot be empty"):
            WebSocketMessage(
                channel="",
                timestamp=datetime.now(UTC),
                data={}
            )
    
    def test_ticker_message_validation(self):
        """Test TickerMessage validation."""
        ticker_msg = TickerMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD", "price": "50000"},
            price=Decimal("50000"),
            volume_24h=Decimal("1000")
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            ticker_msg.price = Decimal("51000")  # type: ignore
        
        # Invalid ticker message
        with pytest.raises(ValueError, match="Price must be positive"):
            TickerMessage(
                channel="ticker",
                timestamp=datetime.now(UTC),
                data={},
                price=Decimal("-100")
            )
    
    def test_trade_message_validation(self):
        """Test TradeMessage validation."""
        trade_msg = TradeMessage(
            channel="trades",
            timestamp=datetime.now(UTC),
            data={"trade_id": "123", "symbol": "BTC-USD"},
            trade_id="123",
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY"
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            trade_msg.side = "SELL"  # type: ignore
        
        # Invalid trade message
        with pytest.raises(ValueError, match="Side must be BUY or SELL"):
            TradeMessage(
                channel="trades",
                timestamp=datetime.now(UTC),
                data={},
                side="INVALID"
            )


class TestConnectionStateManagement:
    """Test connection state management with immutable types."""
    
    def test_connection_state_immutability(self):
        """Test ConnectionState immutability."""
        state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            reconnect_attempts=0,
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            state.status = ConnectionStatus.DISCONNECTED  # type: ignore
        
        # Health check should work
        assert state.is_healthy()
    
    def test_connection_state_health_validation(self):
        """Test connection state health validation."""
        # Healthy connection
        healthy_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        assert validate_connection_health(healthy_state)
        
        # Unhealthy connection - stale messages
        stale_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC) - timedelta(minutes=5)
        )
        assert not stale_state.is_healthy()
        assert not validate_connection_health(stale_state)
    
    def test_data_quality_metrics(self):
        """Test DataQuality metrics calculation."""
        quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=95,
            validation_failures=5,
            average_latency_ms=25.5
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            quality.messages_received = 200  # type: ignore
        
        # Metrics should calculate correctly
        assert quality.success_rate == 95.0
        assert quality.error_rate == 5.0
        
        # Validation should work
        assert validate_data_quality(quality, min_success_rate=90.0)
        assert not validate_data_quality(quality, min_success_rate=98.0)


class TestDataLayerAdapters:
    """Test adapter functionality between FP and legacy systems."""
    
    def test_current_market_data_conversion(self):
        """Test conversion from legacy MarketData to FP Candle."""
        legacy_data = CurrentMarketData(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49500"),
            close=Decimal("50500"),
            volume=Decimal("100.5")
        )
        
        # Convert to FP type
        fp_candle = current_market_data_to_fp_candle(legacy_data)
        
        # Verify conversion
        assert fp_candle.symbol == "BTC-USD"
        assert fp_candle.open == Decimal("50000")
        assert fp_candle.high == Decimal("51000")
        assert fp_candle.low == Decimal("49500")
        assert fp_candle.close == Decimal("50500")
        assert fp_candle.volume == Decimal("100.5")
        assert fp_candle.timestamp == legacy_data.timestamp
        
        # Validate converted data
        assert validate_functional_candle(fp_candle)
    
    def test_batch_conversion_preserves_ordering(self):
        """Test that batch conversions preserve time ordering."""
        legacy_data_list = []
        base_time = datetime.now(UTC)
        
        for i in range(10):
            legacy_data_list.append(CurrentMarketData(
                symbol="BTC-USD",
                timestamp=base_time + timedelta(minutes=i),
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{51000 + i}"),
                low=Decimal(f"{49500 + i}"),
                close=Decimal(f"{50500 + i}"),
                volume=Decimal("100.5")
            ))
        
        # Convert all to FP types
        fp_candles = [current_market_data_to_fp_candle(data) for data in legacy_data_list]
        
        # Verify ordering is preserved
        for i in range(1, len(fp_candles)):
            assert fp_candles[i].timestamp > fp_candles[i-1].timestamp
            assert fp_candles[i].open > fp_candles[i-1].open
        
        # All should be valid
        assert all(validate_functional_candle(candle) for candle in fp_candles)


class TestRealTimeDataStreaming:
    """Test real-time data streaming with functional types."""
    
    def test_realtime_update_immutability(self):
        """Test RealtimeUpdate immutability."""
        update = RealtimeUpdate(
            symbol="BTC-USD",
            timestamp=datetime.now(UTC),
            update_type="ticker",
            data={"price": "50000"},
            exchange="coinbase",
            latency_ms=15.5
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            update.symbol = "ETH-USD"  # type: ignore
        
        # Validation should work
        assert update.symbol == "BTC-USD"
        assert update.update_type == "ticker"
    
    def test_market_data_stream_health(self):
        """Test MarketDataStream health monitoring."""
        # Healthy connection state
        healthy_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        
        # Good data quality
        good_quality = DataQuality(
            timestamp=datetime.now(UTC),
            messages_received=100,
            messages_processed=100,
            validation_failures=0
        )
        
        stream = MarketDataStream(
            symbol="BTC-USD",
            exchanges=["coinbase", "binance"],
            connection_states={
                "coinbase": healthy_state,
                "binance": healthy_state
            },
            data_quality=good_quality,
            active=True
        )
        
        # Should be healthy
        assert stream.overall_health
        assert len(stream.get_healthy_exchanges()) == 2
        assert "coinbase" in stream.get_healthy_exchanges()
        assert "binance" in stream.get_healthy_exchanges()
    
    def test_aggregated_data_calculations(self):
        """Test AggregatedData calculations."""
        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(minutes=5)
        
        candles = [
            Candle(
                timestamp=start_time + timedelta(minutes=i),
                open=Decimal(f"{50000 + i * 10}"),
                high=Decimal(f"{50100 + i * 10}"),
                low=Decimal(f"{49900 + i * 10}"),
                close=Decimal(f"{50050 + i * 10}"),
                volume=Decimal("10"),
                symbol="BTC-USD"
            )
            for i in range(5)
        ]
        
        trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=start_time + timedelta(seconds=i * 30),
                price=Decimal(f"{50000 + i * 5}"),
                size=Decimal("1.0"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD"
            )
            for i in range(10)
        ]
        
        aggregated = AggregatedData(
            symbol="BTC-USD",
            start_time=start_time,
            end_time=end_time,
            candles=candles,
            trades=trades,
            volume_total=Decimal("50"),  # 5 candles * 10 volume each
            trade_count=10
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            aggregated.symbol = "ETH-USD"  # type: ignore
        
        # Calculations should work
        assert aggregated.duration_seconds == 300.0  # 5 minutes
        assert aggregated.average_trade_size == Decimal("5")  # 50 / 10
        
        # VWAP calculation
        expected_vwap = sum(t.price * t.size for t in trades) / sum(t.size for t in trades)
        assert aggregated.vwap == expected_vwap
        
        # Price range
        min_price, max_price = aggregated.get_price_range()
        assert min_price == Decimal("49900")  # Lowest low
        assert max_price == Decimal("50140")  # Highest high


class TestErrorHandlingAndValidation:
    """Test error handling using functional patterns."""
    
    def test_either_pattern_for_validation(self):
        """Test Either pattern for validation results."""
        # Test with valid data
        valid_result = Either.right("valid_data")
        assert isinstance(valid_result, Right)
        assert valid_result.value == "valid_data"
        
        # Test with invalid data
        invalid_result = Either.left("validation_error")
        assert isinstance(invalid_result, Left)
        assert invalid_result.value == "validation_error"
        
        # Test mapping over Either
        mapped_valid = valid_result.map(lambda x: x.upper())
        assert isinstance(mapped_valid, Right)
        assert mapped_valid.value == "VALID_DATA"
        
        mapped_invalid = invalid_result.map(lambda x: x.upper())
        assert isinstance(mapped_invalid, Left)
        assert mapped_invalid.value == "validation_error"  # Unchanged
    
    def test_option_pattern_for_nullable_values(self):
        """Test Option pattern for handling nullable values."""
        # Test with Some value
        some_value = Some("data")
        assert some_value.is_some()
        assert not some_value.is_none()
        assert some_value.unwrap() == "data"
        
        # Test with None value
        none_value = NoneOption()
        assert none_value.is_none()
        assert not none_value.is_some()
        
        with pytest.raises(ValueError):
            none_value.unwrap()
        
        # Test mapping over Option
        mapped_some = some_value.map(lambda x: x.upper())
        assert mapped_some.is_some()
        assert mapped_some.unwrap() == "DATA"
        
        mapped_none = none_value.map(lambda x: x.upper())
        assert mapped_none.is_none()


class TestPerformanceWithFunctionalTypes:
    """Test performance characteristics of functional types."""
    
    def test_immutable_type_creation_performance(self):
        """Test performance of creating immutable types."""
        import time
        
        start_time = time.time()
        
        # Create 1000 candles
        candles = []
        for i in range(1000):
            candle = Candle(
                timestamp=datetime.now(UTC),
                open=Decimal(f"{50000 + i}"),
                high=Decimal(f"{51000 + i}"),
                low=Decimal(f"{49500 + i}"),
                close=Decimal(f"{50500 + i}"),
                volume=Decimal("100.5"),
                symbol="BTC-USD"
            )
            candles.append(candle)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Should create 1000 candles quickly (adjust threshold as needed)
        assert creation_time < 1.0  # Less than 1 second
        assert len(candles) == 1000
        
        # All should be valid
        assert all(validate_functional_candle(candle) for candle in candles[:10])  # Sample check
    
    def test_functional_data_processing_performance(self):
        """Test performance of functional data processing operations."""
        import time
        
        # Create test data
        trades = [
            Trade(
                id=f"trade-{i}",
                timestamp=datetime.now(UTC) + timedelta(seconds=i),
                price=Decimal(f"{50000 + (i % 100)}"),
                size=Decimal("0.1"),
                side="BUY" if i % 2 == 0 else "SELL",
                symbol="BTC-USD"
            )
            for i in range(1000)
        ]
        
        start_time = time.time()
        
        # Process trades functionally
        buy_trades = [t for t in trades if t.is_buy()]
        sell_trades = [t for t in trades if t.is_sell()]
        total_volume = sum(t.size for t in trades)
        total_value = sum(t.value for t in trades)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process quickly
        assert processing_time < 0.1  # Less than 100ms
        assert len(buy_trades) == 500
        assert len(sell_trades) == 500
        assert total_volume == Decimal("100")  # 1000 * 0.1
        assert total_value > Decimal("0")


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional Market Data Validation...")
    
    # Test basic immutable types
    test_types = TestImmutableMarketDataTypes()
    test_types.test_candle_immutability()
    test_types.test_candle_validation()
    test_types.test_trade_immutability_and_validation()
    test_types.test_orderbook_immutability_and_validation()
    test_types.test_ticker_validation()
    print("✓ Immutable market data types tests passed")
    
    # Test functional processing
    test_processing = TestFunctionalDataProcessing()
    test_processing.test_market_data_effect_validation()
    test_processing.test_candle_processing_effect()
    test_processing.test_trade_aggregation_effect()
    print("✓ Functional data processing tests passed")
    
    # Test WebSocket messages
    test_websocket = TestWebSocketMessageTypes()
    test_websocket.test_websocket_message_validation()
    test_websocket.test_ticker_message_validation()
    test_websocket.test_trade_message_validation()
    print("✓ WebSocket message types tests passed")
    
    # Test connection state
    test_connection = TestConnectionStateManagement()
    test_connection.test_connection_state_immutability()
    test_connection.test_connection_state_health_validation()
    test_connection.test_data_quality_metrics()
    print("✓ Connection state management tests passed")
    
    print("All functional market data validation tests completed successfully!")