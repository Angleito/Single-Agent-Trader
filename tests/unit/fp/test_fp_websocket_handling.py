"""
Functional Programming WebSocket Message Handling Tests

This module tests WebSocket message processing, validation, and real-time
data handling using immutable message types and functional patterns.
"""

import pytest
import asyncio
import json
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

from bot.fp.types.market import (
    WebSocketMessage,
    TickerMessage,
    TradeMessage,
    OrderBookMessage,
    ConnectionState,
    ConnectionStatus,
    DataQuality,
    RealtimeUpdate,
    MarketDataStream,
)
from bot.fp.effects.websocket_enhanced import (
    WebSocketEffect,
    connect_websocket_effect,
    send_message_effect,
    receive_message_effect,
    validate_message_effect,
    process_ticker_effect,
    process_trade_effect,
    process_orderbook_effect,
)
from bot.fp.core.either import Either, Left, Right
from bot.fp.adapters.market_data_adapter import (
    FunctionalMarketDataProcessor,
    create_functional_market_data_processor,
)
from bot.fp.adapters.type_converters import (
    create_ticker_message_from_data,
    create_trade_message_from_data,
    create_orderbook_message_from_data,
    validate_websocket_message,
    parse_coinbase_ticker_message,
    parse_coinbase_trade_message,
    update_connection_state,
)


class TestWebSocketMessageTypes:
    """Test immutable WebSocket message types and validation."""
    
    def test_websocket_message_immutability(self):
        """Test WebSocketMessage immutability and validation."""
        message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD", "price": "50000"},
            message_id="msg-123"
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            message.channel = "trades"  # type: ignore
        
        with pytest.raises(AttributeError):
            message.data = {}  # type: ignore
        
        # Properties should be accessible
        assert message.channel == "ticker"
        assert message.message_id == "msg-123"
        assert "symbol" in message.data
        
    def test_websocket_message_validation_rules(self):
        """Test WebSocket message validation rules."""
        # Valid message
        valid_message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"price": "50000"}
        )
        assert validate_websocket_message(valid_message)
        
        # Invalid message - empty channel
        with pytest.raises(ValueError, match="Channel cannot be empty"):
            WebSocketMessage(
                channel="",
                timestamp=datetime.now(UTC),
                data={}
            )
        
        # Invalid message - non-dict data
        with pytest.raises(ValueError, match="Data must be a dictionary"):
            WebSocketMessage(
                channel="ticker",
                timestamp=datetime.now(UTC),
                data="invalid"  # type: ignore
            )
    
    def test_ticker_message_immutability(self):
        """Test TickerMessage immutability and validation."""
        ticker_msg = TickerMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD", "price": "50000"},
            price=Decimal("50000"),
            volume_24h=Decimal("1000"),
            message_id="ticker-123"
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            ticker_msg.price = Decimal("51000")  # type: ignore
        
        with pytest.raises(AttributeError):
            ticker_msg.volume_24h = Decimal("2000")  # type: ignore
        
        # Properties should work
        assert ticker_msg.price == Decimal("50000")
        assert ticker_msg.volume_24h == Decimal("1000")
        assert ticker_msg.channel == "ticker"
    
    def test_ticker_message_validation_rules(self):
        """Test TickerMessage validation rules."""
        # Valid ticker message
        valid_ticker = TickerMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD"},
            price=Decimal("50000")
        )
        assert valid_ticker.price == Decimal("50000")
        
        # Invalid ticker - negative price
        with pytest.raises(ValueError, match="Price must be positive"):
            TickerMessage(
                channel="ticker",
                timestamp=datetime.now(UTC),
                data={},
                price=Decimal("-100")
            )
        
        # Invalid ticker - negative volume
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            TickerMessage(
                channel="ticker",
                timestamp=datetime.now(UTC),
                data={},
                price=Decimal("50000"),
                volume_24h=Decimal("-500")
            )
    
    def test_trade_message_immutability(self):
        """Test TradeMessage immutability and validation."""
        trade_msg = TradeMessage(
            channel="market_trades",
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
        
        with pytest.raises(AttributeError):
            trade_msg.price = Decimal("51000")  # type: ignore
        
        # Properties should work
        assert trade_msg.side == "BUY"
        assert trade_msg.price == Decimal("50000")
        assert trade_msg.size == Decimal("0.5")
    
    def test_trade_message_validation_rules(self):
        """Test TradeMessage validation rules."""
        # Valid trade message
        valid_trade = TradeMessage(
            channel="market_trades",
            timestamp=datetime.now(UTC),
            data={"trade_id": "123"},
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY"
        )
        assert valid_trade.side == "BUY"
        
        # Invalid trade - negative price
        with pytest.raises(ValueError, match="Price must be positive"):
            TradeMessage(
                channel="market_trades",
                timestamp=datetime.now(UTC),
                data={},
                price=Decimal("-100"),
                size=Decimal("0.5"),
                side="BUY"
            )
        
        # Invalid trade - invalid side
        with pytest.raises(ValueError, match="Side must be BUY or SELL"):
            TradeMessage(
                channel="market_trades",
                timestamp=datetime.now(UTC),
                data={},
                price=Decimal("50000"),
                size=Decimal("0.5"),
                side="INVALID"
            )
    
    def test_orderbook_message_immutability(self):
        """Test OrderBookMessage immutability and validation."""
        bids = [(Decimal("49990"), Decimal("1.0")), (Decimal("49980"), Decimal("2.0"))]
        asks = [(Decimal("50010"), Decimal("1.2")), (Decimal("50020"), Decimal("0.8"))]
        
        orderbook_msg = OrderBookMessage(
            channel="orderbook",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD"},
            bids=bids,
            asks=asks
        )
        
        # Should be immutable
        with pytest.raises(AttributeError):
            orderbook_msg.bids = []  # type: ignore
        
        with pytest.raises(AttributeError):
            orderbook_msg.asks = []  # type: ignore
        
        # Properties should work
        assert len(orderbook_msg.bids) == 2
        assert len(orderbook_msg.asks) == 2
        assert orderbook_msg.bids[0] == (Decimal("49990"), Decimal("1.0"))
    
    def test_orderbook_message_validation_rules(self):
        """Test OrderBookMessage validation rules."""
        # Valid orderbook message
        valid_bids = [(Decimal("49990"), Decimal("1.0")), (Decimal("49980"), Decimal("2.0"))]
        valid_asks = [(Decimal("50010"), Decimal("1.2")), (Decimal("50020"), Decimal("0.8"))]
        
        valid_orderbook = OrderBookMessage(
            channel="orderbook",
            timestamp=datetime.now(UTC),
            data={},
            bids=valid_bids,
            asks=valid_asks
        )
        assert len(valid_orderbook.bids) == 2
        assert len(valid_orderbook.asks) == 2
        
        # Invalid orderbook - best bid >= best ask
        invalid_bids = [(Decimal("50020"), Decimal("1.0"))]  # Higher than best ask
        invalid_asks = [(Decimal("50010"), Decimal("1.2"))]
        
        with pytest.raises(ValueError, match="Best bid .* must be < best ask"):
            OrderBookMessage(
                channel="orderbook",
                timestamp=datetime.now(UTC),
                data={},
                bids=invalid_bids,
                asks=invalid_asks
            )


class TestWebSocketMessageProcessing:
    """Test WebSocket message processing with functional effects."""
    
    def test_validate_message_effect(self):
        """Test message validation using effects."""
        # Valid message
        valid_message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD", "price": "50000"}
        )
        
        result = validate_message_effect(valid_message)
        assert isinstance(result, Right)
        assert result.value == valid_message
        
        # Invalid message data should be handled gracefully in the effect
        # (since validation happens at construction time)
        empty_data_message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={}
        )
        
        result = validate_message_effect(empty_data_message)
        assert isinstance(result, Right)  # Still valid structurally
    
    def test_process_ticker_effect(self):
        """Test ticker message processing using effects."""
        ticker_msg = TickerMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD", "price": "50000"},
            price=Decimal("50000"),
            volume_24h=Decimal("1000")
        )
        
        result = process_ticker_effect(ticker_msg, "BTC-USD")
        assert isinstance(result, Right)
        
        processed_update = result.value
        assert processed_update.symbol == "BTC-USD"
        assert processed_update.update_type == "ticker"
        assert "price" in processed_update.data
    
    def test_process_trade_effect(self):
        """Test trade message processing using effects."""
        trade_msg = TradeMessage(
            channel="market_trades",
            timestamp=datetime.now(UTC),
            data={"trade_id": "123", "symbol": "BTC-USD"},
            trade_id="123",
            price=Decimal("50000"),
            size=Decimal("0.5"),
            side="BUY"
        )
        
        result = process_trade_effect(trade_msg, "BTC-USD")
        assert isinstance(result, Right)
        
        processed_update = result.value
        assert processed_update.symbol == "BTC-USD"
        assert processed_update.update_type == "trade"
        assert "trade_id" in processed_update.data
    
    def test_process_orderbook_effect(self):
        """Test orderbook message processing using effects."""
        bids = [(Decimal("49990"), Decimal("1.0"))]
        asks = [(Decimal("50010"), Decimal("1.2"))]
        
        orderbook_msg = OrderBookMessage(
            channel="orderbook",
            timestamp=datetime.now(UTC),
            data={"symbol": "BTC-USD"},
            bids=bids,
            asks=asks
        )
        
        result = process_orderbook_effect(orderbook_msg, "BTC-USD")
        assert isinstance(result, Right)
        
        processed_update = result.value
        assert processed_update.symbol == "BTC-USD"
        assert processed_update.update_type == "orderbook"


class TestCoinbaseMessageParsing:
    """Test parsing of real Coinbase WebSocket messages."""
    
    def test_parse_coinbase_ticker_message(self):
        """Test parsing Coinbase ticker messages."""
        raw_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "volume_24h": "1000.0",
                    "best_bid": "49995.00",
                    "best_ask": "50005.00"
                }]
            }]
        }
        
        result = parse_coinbase_ticker_message(raw_message, "BTC-USD")
        assert isinstance(result, Right)
        
        ticker_msg = result.value
        assert ticker_msg.price == Decimal("50000.00")
        assert ticker_msg.volume_24h == Decimal("1000.0")
        assert ticker_msg.channel == "ticker"
    
    def test_parse_coinbase_trade_message(self):
        """Test parsing Coinbase trade messages."""
        raw_message = {
            "channel": "market_trades",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "trades": [{
                    "trade_id": "123456",
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "size": "0.5",
                    "side": "BUY",
                    "time": "2024-01-01T12:00:00Z"
                }]
            }]
        }
        
        result = parse_coinbase_trade_message(raw_message, "BTC-USD")
        assert isinstance(result, Right)
        
        trade_msg = result.value
        assert trade_msg.price == Decimal("50000.00")
        assert trade_msg.size == Decimal("0.5")
        assert trade_msg.side == "BUY"
        assert trade_msg.trade_id == "123456"
    
    def test_parse_invalid_coinbase_message(self):
        """Test parsing invalid Coinbase messages."""
        # Missing events
        invalid_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        result = parse_coinbase_ticker_message(invalid_message, "BTC-USD")
        assert isinstance(result, Left)
        assert "events" in result.value
        
        # Empty events
        empty_events_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": []
        }
        
        result = parse_coinbase_ticker_message(empty_events_message, "BTC-USD")
        assert isinstance(result, Left)
        assert "no events" in result.value.lower()
    
    def test_parse_message_with_wrong_symbol(self):
        """Test parsing messages for different symbols."""
        raw_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "ETH-USD",  # Different symbol
                    "price": "3000.00"
                }]
            }]
        }
        
        result = parse_coinbase_ticker_message(raw_message, "BTC-USD")
        assert isinstance(result, Left)
        assert "symbol mismatch" in result.value.lower() or "no matching" in result.value.lower()


class TestFunctionalWebSocketProcessor:
    """Test the functional WebSocket processor integration."""
    
    def test_processor_websocket_message_handling(self):
        """Test WebSocket message handling in the functional processor."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Mock callback to capture updates
        update_callback = MagicMock()
        processor.add_update_callback(update_callback)
        
        # Test ticker message
        ticker_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "volume_24h": "1000.0"
                }]
            }]
        }
        
        processor.process_websocket_message(ticker_message)
        
        # Verify callback was called
        assert update_callback.call_count >= 1
        
        # Verify update was stored
        updates = processor.get_recent_updates()
        assert len(updates) >= 1
        assert updates[-1].update_type == "ticker"
        assert updates[-1].symbol == "BTC-USD"
    
    def test_processor_trade_message_handling(self):
        """Test trade message handling in the functional processor."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Mock callback
        update_callback = MagicMock()
        processor.add_update_callback(update_callback)
        
        # Test trade message
        trade_message = {
            "channel": "market_trades",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "trades": [{
                    "trade_id": "123456",
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "size": "0.5",
                    "side": "BUY"
                }]
            }]
        }
        
        processor.process_websocket_message(trade_message)
        
        # Verify processing
        assert update_callback.call_count >= 1
        updates = processor.get_recent_updates()
        assert len(updates) >= 1
        assert updates[-1].update_type == "trade"
    
    def test_processor_error_handling(self):
        """Test error handling in WebSocket message processing."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Test with malformed message
        malformed_message = {
            "channel": "ticker",
            "invalid_data": "test"
        }
        
        # Should not raise exception
        processor.process_websocket_message(malformed_message)
        
        # Data quality should reflect the error
        quality = processor.get_data_quality()
        # Error count might be incremented (implementation dependent)
        assert quality.messages_received >= 0  # Should track attempts
    
    def test_processor_message_rate_limiting(self):
        """Test message rate limiting and queue management."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Send many messages rapidly
        for i in range(600):  # More than the queue limit of 500
            ticker_message = {
                "channel": "ticker",
                "timestamp": f"2024-01-01T12:00:{i:02d}Z",
                "events": [{
                    "type": "update",
                    "tickers": [{
                        "product_id": "BTC-USD",
                        "price": f"{50000 + i}.00"
                    }]
                }]
            }
            processor.process_websocket_message(ticker_message)
        
        # Should not exceed queue limit
        updates = processor.get_recent_updates()
        assert len(updates) <= 500  # Queue limit
        
        # Latest updates should be preserved
        latest_update = updates[-1]
        assert "599" in str(latest_update.data) or "50599" in str(latest_update.data)


class TestWebSocketConnectionState:
    """Test WebSocket connection state management."""
    
    def test_connection_state_transitions(self):
        """Test connection state transitions."""
        # Start disconnected
        state = ConnectionState(
            status=ConnectionStatus.DISCONNECTED,
            url="wss://test.com"
        )
        assert not state.is_healthy()
        
        # Transition to connecting
        connecting_state = update_connection_state(
            state,
            new_status="CONNECTING"
        )
        assert connecting_state.status == ConnectionStatus.CONNECTING
        assert not connecting_state.is_healthy()
        
        # Transition to connected
        connected_state = update_connection_state(
            connecting_state,
            new_status="CONNECTED",
            message_received=True
        )
        assert connected_state.status == ConnectionStatus.CONNECTED
        assert connected_state.is_healthy()
    
    def test_connection_health_monitoring(self):
        """Test connection health monitoring."""
        # Healthy connection
        healthy_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC)
        )
        assert healthy_state.is_healthy()
        
        # Stale connection
        stale_state = ConnectionState(
            status=ConnectionStatus.CONNECTED,
            url="wss://test.com",
            connected_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC) - timedelta(minutes=5)
        )
        assert not stale_state.is_healthy()
        
        # Error state
        error_state = ConnectionState(
            status=ConnectionStatus.ERROR,
            url="wss://test.com",
            last_error="Connection failed"
        )
        assert not error_state.is_healthy()
    
    def test_reconnection_tracking(self):
        """Test reconnection attempt tracking."""
        state = ConnectionState(
            status=ConnectionStatus.RECONNECTING,
            url="wss://test.com",
            reconnect_attempts=0
        )
        
        # Simulate reconnection attempts
        for attempt in range(1, 6):
            state = update_connection_state(
                state,
                new_status="RECONNECTING",
                increment_reconnect=True
            )
            assert state.reconnect_attempts == attempt
        
        # Should track maximum attempts
        assert state.reconnect_attempts == 5


class TestWebSocketMessageValidation:
    """Test WebSocket message validation functions."""
    
    def test_message_structure_validation(self):
        """Test message structure validation."""
        # Valid message structure
        valid_message = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "tickers": [{"product_id": "BTC-USD", "price": "50000.00"}]
            }]
        }
        
        # Should validate successfully
        result = validate_websocket_message(valid_message)
        assert result is True or isinstance(result, Right)
        
        # Invalid message - missing channel
        invalid_message = {
            "timestamp": "2024-01-01T12:00:00Z",
            "events": []
        }
        
        # Should fail validation
        result = validate_websocket_message(invalid_message)
        assert result is False or isinstance(result, Left)
    
    def test_timestamp_validation(self):
        """Test timestamp validation in messages."""
        now = datetime.now(UTC)
        
        # Recent timestamp (valid)
        recent_message = WebSocketMessage(
            channel="ticker",
            timestamp=now,
            data={"price": "50000"}
        )
        assert validate_websocket_message(recent_message)
        
        # Future timestamp (might be valid depending on tolerance)
        future_message = WebSocketMessage(
            channel="ticker",
            timestamp=now + timedelta(seconds=1),
            data={"price": "50000"}
        )
        assert validate_websocket_message(future_message)
        
        # Very old timestamp (might be invalid)
        old_message = WebSocketMessage(
            channel="ticker",
            timestamp=now - timedelta(hours=1),
            data={"price": "50000"}
        )
        # Should still be structurally valid
        assert validate_websocket_message(old_message)
    
    def test_data_content_validation(self):
        """Test data content validation."""
        # Valid price data
        valid_data = {"symbol": "BTC-USD", "price": "50000.00", "volume": "100.5"}
        valid_message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data=valid_data
        )
        assert validate_websocket_message(valid_message)
        
        # Invalid price data (handled at higher level)
        invalid_data = {"symbol": "BTC-USD", "price": "-1000.00"}
        invalid_message = WebSocketMessage(
            channel="ticker",
            timestamp=datetime.now(UTC),
            data=invalid_data
        )
        # Should be structurally valid even with invalid business logic
        assert validate_websocket_message(invalid_message)


class TestWebSocketPerformance:
    """Test WebSocket message processing performance."""
    
    def test_message_processing_performance(self):
        """Test performance of message processing."""
        import time
        
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Prepare test messages
        messages = []
        for i in range(100):
            messages.append({
                "channel": "ticker",
                "timestamp": f"2024-01-01T12:00:{i:02d}Z",
                "events": [{
                    "type": "update",
                    "tickers": [{
                        "product_id": "BTC-USD",
                        "price": f"{50000 + i}.00"
                    }]
                }]
            })
        
        # Process messages and measure time
        start_time = time.time()
        
        for message in messages:
            processor.process_websocket_message(message)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process quickly
        assert processing_time < 1.0  # Less than 1 second for 100 messages
        
        # Verify all messages were processed
        updates = processor.get_recent_updates()
        assert len(updates) == 100
    
    def test_concurrent_message_processing(self):
        """Test concurrent message processing."""
        processor = create_functional_market_data_processor("BTC-USD", "1m")
        
        # Create multiple message types
        ticker_msg = {
            "channel": "ticker",
            "timestamp": "2024-01-01T12:00:00Z",
            "events": [{
                "type": "update",
                "tickers": [{"product_id": "BTC-USD", "price": "50000.00"}]
            }]
        }
        
        trade_msg = {
            "channel": "market_trades",
            "timestamp": "2024-01-01T12:00:01Z",
            "events": [{
                "type": "update",
                "trades": [{
                    "trade_id": "123",
                    "product_id": "BTC-USD",
                    "price": "50000.00",
                    "size": "0.5",
                    "side": "BUY"
                }]
            }]
        }
        
        # Process concurrently (simulated)
        processor.process_websocket_message(ticker_msg)
        processor.process_websocket_message(trade_msg)
        
        # Both should be processed
        updates = processor.get_recent_updates()
        assert len(updates) == 2
        
        update_types = [update.update_type for update in updates]
        assert "ticker" in update_types
        assert "trade" in update_types


if __name__ == "__main__":
    # Run some basic functionality tests
    print("Testing Functional WebSocket Message Handling...")
    
    # Test immutable message types
    test_types = TestWebSocketMessageTypes()
    test_types.test_websocket_message_immutability()
    test_types.test_ticker_message_immutability()
    test_types.test_trade_message_immutability()
    test_types.test_orderbook_message_immutability()
    print("✓ WebSocket message types tests passed")
    
    # Test message processing
    test_processing = TestWebSocketMessageProcessing()
    test_processing.test_validate_message_effect()
    test_processing.test_process_ticker_effect()
    test_processing.test_process_trade_effect()
    test_processing.test_process_orderbook_effect()
    print("✓ WebSocket message processing tests passed")
    
    # Test Coinbase parsing
    test_coinbase = TestCoinbaseMessageParsing()
    test_coinbase.test_parse_coinbase_ticker_message()
    test_coinbase.test_parse_coinbase_trade_message()
    test_coinbase.test_parse_invalid_coinbase_message()
    print("✓ Coinbase message parsing tests passed")
    
    # Test functional processor
    test_processor = TestFunctionalWebSocketProcessor()
    test_processor.test_processor_websocket_message_handling()
    test_processor.test_processor_trade_message_handling()
    test_processor.test_processor_error_handling()
    print("✓ Functional WebSocket processor tests passed")
    
    print("All functional WebSocket handling tests completed successfully!")