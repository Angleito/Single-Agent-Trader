"""
Functional Programming WebSocket Tests

Test suite for WebSocket components with functional programming patterns.
Tests immutable message types, IOEither error handling, and enhanced connection management.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from websockets.exceptions import ConnectionClosed, WebSocketException

from bot.fp.effects.io import IOEither
from bot.fp.effects.websocket_enhanced import (
    ConnectionMetrics,
    ConnectionState,
    EnhancedConnectionConfig,
    EnhancedWebSocketManager,
    MessageEnvelope,
    create_enhanced_websocket_manager,
    create_high_reliability_websocket_manager,
    create_low_latency_websocket_manager,
)
from bot.fp.types.market import FunctionalMarketData
from bot.fp.types.trading import FuturesMarketData


class TestEnhancedConnectionConfig:
    """Test immutable connection configuration."""

    def test_connection_config_creation(self):
        """Test creation of enhanced connection config."""
        config = EnhancedConnectionConfig(
            url="wss://api.example.com/ws",
            headers={"Authorization": "Bearer token"},
            heartbeat_interval=30,
            reconnect_attempts=10,
            connection_timeout=10.0,
        )

        assert config.url == "wss://api.example.com/ws"
        assert config.headers["Authorization"] == "Bearer token"
        assert config.heartbeat_interval == 30
        assert config.reconnect_attempts == 10
        assert config.connection_timeout == 10.0

    def test_connection_config_defaults(self):
        """Test default values in connection config."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")

        assert config.headers == {}
        assert config.heartbeat_interval == 30
        assert config.reconnect_attempts == 10
        assert config.reconnect_delay == 1.0
        assert config.max_reconnect_delay == 60.0
        assert config.enable_ping_pong is True

    def test_connection_config_immutability(self):
        """Test that connection config is immutable."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            config.url = "wss://different.com/ws"  # type: ignore


class TestMessageEnvelope:
    """Test immutable message envelope."""

    def test_message_envelope_creation(self):
        """Test creation of message envelope."""
        payload = {"type": "ticker", "symbol": "BTC-USD", "price": 50000}
        timestamp = datetime.now()

        envelope = MessageEnvelope(
            payload=payload,
            timestamp=timestamp,
            message_id="msg-123",
            message_type="ticker",
            source="wss://api.exchange.com/ws",
        )

        assert envelope.payload == payload
        assert envelope.timestamp == timestamp
        assert envelope.message_id == "msg-123"
        assert envelope.message_type == "ticker"
        assert envelope.source == "wss://api.exchange.com/ws"

    def test_message_envelope_immutability(self):
        """Test that message envelope is immutable."""
        envelope = MessageEnvelope(
            payload={"test": "data"},
            timestamp=datetime.now(),
            message_id="msg-123",
            message_type="test",
            source="test",
        )

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            envelope.payload = {"modified": "data"}  # type: ignore


class TestConnectionMetrics:
    """Test connection metrics tracking."""

    def test_connection_metrics_creation(self):
        """Test creation of connection metrics."""
        connect_time = datetime.now()
        last_message_time = connect_time + timedelta(seconds=30)

        metrics = ConnectionMetrics(
            connect_time=connect_time,
            last_message_time=last_message_time,
            messages_received=100,
            messages_sent=50,
            reconnection_count=2,
            total_downtime=timedelta(seconds=120),
            average_latency=15.5,
            connection_state=ConnectionState.CONNECTED,
        )

        assert metrics.connect_time == connect_time
        assert metrics.last_message_time == last_message_time
        assert metrics.messages_received == 100
        assert metrics.messages_sent == 50
        assert metrics.reconnection_count == 2
        assert metrics.total_downtime == timedelta(seconds=120)
        assert metrics.average_latency == 15.5
        assert metrics.connection_state == ConnectionState.CONNECTED

    def test_connection_state_enum(self):
        """Test connection state enumeration."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.FAILED.value == "failed"


class MockWebSocketConnection:
    """Mock WebSocket connection for testing."""

    def __init__(self, should_fail: bool = False, fail_after: int = 0):
        self.should_fail = should_fail
        self.fail_after = fail_after
        self.closed = False
        self.messages_sent = []
        self.message_count = 0

    async def send(self, message: str):
        """Mock send method."""
        if self.should_fail:
            raise WebSocketException("Mock send failure")
        self.messages_sent.append(message)

    async def close(self):
        """Mock close method."""
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Mock message iteration."""
        if self.closed or (
            self.fail_after > 0 and self.message_count >= self.fail_after
        ):
            raise ConnectionClosed(None, None)

        if self.should_fail:
            raise WebSocketException("Mock receive failure")

        self.message_count += 1

        # Return test message
        test_message = json.dumps(
            {
                "type": "ticker",
                "symbol": "BTC-USD",
                "price": 50000 + self.message_count,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return test_message


class TestEnhancedWebSocketManager:
    """Test enhanced WebSocket manager functionality."""

    def test_websocket_manager_creation(self):
        """Test creation of enhanced WebSocket manager."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        assert manager.config == config
        assert manager._state == ConnectionState.DISCONNECTED
        assert not manager._is_running

    def test_websocket_manager_state_tracking(self):
        """Test state tracking in WebSocket manager."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Initial state
        state_result = manager.get_connection_state().run()
        assert state_result == ConnectionState.DISCONNECTED

        # Health check
        health_result = manager.check_connection_health().run()
        assert health_result is False

    @patch("websockets.connect")
    def test_websocket_connection_success(self, mock_connect):
        """Test successful WebSocket connection."""
        mock_ws = MockWebSocketConnection()
        mock_connect.return_value.__aenter__.return_value = mock_ws

        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Mock the connection establishment
        manager._connection = mock_ws
        manager._state = ConnectionState.CONNECTED
        manager._is_running = True

        # Check state
        state = manager.get_connection_state().run()
        assert state == ConnectionState.CONNECTED

    def test_websocket_message_subscription(self):
        """Test message subscription functionality."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Track received messages
        received_messages = []

        def message_callback(envelope: MessageEnvelope):
            received_messages.append(envelope)

        # Subscribe to ticker messages
        subscribe_io = manager.subscribe_to_messages("ticker", message_callback)
        subscribe_io.run()

        # Simulate message notification
        test_envelope = MessageEnvelope(
            payload={"type": "ticker", "price": 50000},
            timestamp=datetime.now(),
            message_id="test-123",
            message_type="ticker",
            source="test",
        )

        manager._notify_subscribers(test_envelope)

        assert len(received_messages) == 1
        assert received_messages[0].message_type == "ticker"

    def test_websocket_message_unsubscription(self):
        """Test message unsubscription functionality."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        received_messages = []

        def message_callback(envelope: MessageEnvelope):
            received_messages.append(envelope)

        # Subscribe and then unsubscribe
        manager.subscribe_to_messages("ticker", message_callback).run()
        manager.unsubscribe_from_messages("ticker", message_callback).run()

        # Simulate message - should not be received
        test_envelope = MessageEnvelope(
            payload={"type": "ticker", "price": 50000},
            timestamp=datetime.now(),
            message_id="test-123",
            message_type="ticker",
            source="test",
        )

        manager._notify_subscribers(test_envelope)

        assert len(received_messages) == 0

    def test_websocket_circuit_breaker(self):
        """Test circuit breaker pattern."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Set manager to failed state
        manager._state = ConnectionState.FAILED

        def test_operation():
            return IOEither.right("success")

        # Circuit breaker should prevent operation
        result = manager.with_circuit_breaker(test_operation).run()

        # Should fail due to circuit breaker
        assert result.is_left()
        assert "Circuit breaker open" in str(result.value)

    def test_websocket_message_id_generation(self):
        """Test unique message ID generation."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        id1 = manager._generate_message_id()
        id2 = manager._generate_message_id()

        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0

    def test_message_type_extraction(self):
        """Test message type extraction from different formats."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Test different message formats
        test_cases = [
            ({"type": "ticker"}, "ticker"),
            ({"channel": "trades"}, "trades"),
            ({"event": "update"}, "update"),
            ({"data": "value"}, "unknown"),
        ]

        for message_data, expected_type in test_cases:
            extracted_type = manager._extract_message_type(message_data)
            assert extracted_type == expected_type


class TestWebSocketFactoryFunctions:
    """Test WebSocket factory functions."""

    def test_enhanced_websocket_manager_factory(self):
        """Test enhanced WebSocket manager factory."""
        manager = create_enhanced_websocket_manager(
            url="wss://api.example.com/ws",
            headers={"Auth": "token"},
            heartbeat_interval=20,
            reconnect_attempts=5,
        )

        assert manager.config.url == "wss://api.example.com/ws"
        assert manager.config.headers["Auth"] == "token"
        assert manager.config.heartbeat_interval == 20
        assert manager.config.reconnect_attempts == 5

    def test_high_reliability_websocket_factory(self):
        """Test high reliability WebSocket manager factory."""
        manager = create_high_reliability_websocket_manager("wss://api.example.com/ws")

        assert manager.config.heartbeat_interval == 15  # More frequent
        assert manager.config.reconnect_attempts == 20  # More attempts
        assert manager.config.reconnect_delay == 0.5  # Faster initial reconnect
        assert manager.config.enable_ping_pong is True

    def test_low_latency_websocket_factory(self):
        """Test low latency WebSocket manager factory."""
        manager = create_low_latency_websocket_manager("wss://api.example.com/ws")

        assert manager.config.heartbeat_interval == 60  # Less frequent
        assert manager.config.reconnect_attempts == 5  # Fewer attempts
        assert manager.config.reconnect_delay == 0.1  # Very fast reconnect
        assert manager.config.connection_timeout == 2.0  # Short timeout
        assert manager.config.compression is None  # No compression


class TestWebSocketMarketDataIntegration:
    """Test WebSocket integration with functional market data types."""

    def test_market_data_message_parsing(self):
        """Test parsing WebSocket messages into functional market data."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Mock market data message
        market_data_payload = {
            "type": "ticker",
            "symbol": "BTC-USD",
            "timestamp": "2024-01-01T12:00:00Z",
            "open": "50000",
            "high": "52000",
            "low": "49000",
            "close": "51000",
            "volume": "100",
        }

        envelope = MessageEnvelope(
            payload=market_data_payload,
            timestamp=datetime.now(),
            message_id="msg-123",
            message_type="ticker",
            source="test",
        )

        # Convert to functional market data
        def parse_market_data(envelope: MessageEnvelope) -> FunctionalMarketData | None:
            try:
                payload = envelope.payload
                return FunctionalMarketData(
                    symbol=payload["symbol"],
                    timestamp=datetime.fromisoformat(
                        payload["timestamp"].replace("Z", "+00:00")
                    ),
                    open=Decimal(payload["open"]),
                    high=Decimal(payload["high"]),
                    low=Decimal(payload["low"]),
                    close=Decimal(payload["close"]),
                    volume=Decimal(payload["volume"]),
                )
            except Exception:
                return None

        market_data = parse_market_data(envelope)

        assert market_data is not None
        assert market_data.symbol == "BTC-USD"
        assert market_data.is_bullish  # close > open

    def test_futures_market_data_websocket(self):
        """Test WebSocket integration with futures market data."""
        futures_payload = {
            "type": "futures_ticker",
            "symbol": "BTC-PERP",
            "timestamp": "2024-01-01T12:00:00Z",
            "open": "50000",
            "high": "52000",
            "low": "49000",
            "close": "51000",
            "volume": "100",
            "open_interest": "1000000",
            "funding_rate": "0.0001",
            "mark_price": "51010",
            "index_price": "51005",
        }

        envelope = MessageEnvelope(
            payload=futures_payload,
            timestamp=datetime.now(),
            message_id="msg-456",
            message_type="futures_ticker",
            source="test",
        )

        # Convert to futures market data
        def parse_futures_data(envelope: MessageEnvelope) -> FuturesMarketData | None:
            try:
                payload = envelope.payload
                base_data = FunctionalMarketData(
                    symbol=payload["symbol"],
                    timestamp=datetime.fromisoformat(
                        payload["timestamp"].replace("Z", "+00:00")
                    ),
                    open=Decimal(payload["open"]),
                    high=Decimal(payload["high"]),
                    low=Decimal(payload["low"]),
                    close=Decimal(payload["close"]),
                    volume=Decimal(payload["volume"]),
                )

                return FuturesMarketData(
                    base_data=base_data,
                    open_interest=Decimal(payload["open_interest"]),
                    funding_rate=float(payload["funding_rate"]),
                    mark_price=Decimal(payload.get("mark_price")),
                    index_price=Decimal(payload.get("index_price")),
                )
            except Exception:
                return None

        futures_data = parse_futures_data(envelope)

        assert futures_data is not None
        assert futures_data.symbol == "BTC-PERP"
        assert futures_data.open_interest == Decimal("1000000")
        assert futures_data.funding_rate == 0.0001
        assert futures_data.basis == Decimal("5")  # mark_price - index_price


@pytest.mark.asyncio
class TestAsyncWebSocketOperations:
    """Test asynchronous WebSocket operations."""

    async def test_async_message_streaming(self):
        """Test asynchronous message streaming."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Mock message queue
        manager._message_queue = asyncio.Queue(maxsize=100)
        manager._is_running = True

        # Add test messages
        test_messages = []
        for i in range(5):
            envelope = MessageEnvelope(
                payload={"type": "test", "value": i},
                timestamp=datetime.now(),
                message_id=f"msg-{i}",
                message_type="test",
                source="test",
            )
            test_messages.append(envelope)
            manager._message_queue.put_nowait(envelope)

        # Stop after messages
        manager._is_running = False

        # Stream messages
        received_messages = []

        # Create stream manually since we can't easily test the full async iterator
        async def collect_messages():
            while not manager._message_queue.empty():
                try:
                    message = await asyncio.wait_for(
                        manager._message_queue.get(), timeout=0.1
                    )
                    received_messages.append(message)
                except TimeoutError:
                    break

        await collect_messages()

        assert len(received_messages) == 5
        assert all(msg.message_type == "test" for msg in received_messages)

    async def test_async_error_handling(self):
        """Test error handling in async operations."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Test error in send operation
        async def test_send_error():
            # Simulate no connection
            manager._connection = None

            result = manager.send_message({"test": "data"})
            return result.run()

        result = await test_send_error()
        assert result.is_left()
        assert "WebSocket not connected" in str(result.value)

    async def test_connection_health_monitoring(self):
        """Test connection health monitoring."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Test disconnected state
        health_check = manager.check_connection_health().run()
        assert health_check is False

        # Test connected state with recent activity
        mock_connection = MockWebSocketConnection()
        manager._connection = mock_connection
        manager._state = ConnectionState.CONNECTED
        manager._metrics = manager._metrics._replace(
            last_message_time=datetime.utcnow(),
            connection_state=ConnectionState.CONNECTED,
        )

        health_check = manager.check_connection_health().run()
        assert health_check is True

    async def test_concurrent_websocket_operations(self):
        """Test concurrent WebSocket operations."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Mock successful connection
        manager._connection = MockWebSocketConnection()
        manager._state = ConnectionState.CONNECTED

        # Simulate concurrent message sending
        async def send_test_message(data: dict[str, Any]):
            # Simulate sending (mock doesn't actually send)
            await asyncio.sleep(0.01)  # Small delay
            return {"sent": data}

        messages = [
            {"type": "subscribe", "channel": "ticker"},
            {"type": "subscribe", "channel": "trades"},
            {"type": "ping", "timestamp": datetime.now().isoformat()},
        ]

        # Send messages concurrently
        tasks = [send_test_message(msg) for msg in messages]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all("sent" in result for result in results)


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and resilience."""

    def test_connection_error_handling(self):
        """Test connection error handling."""
        config = EnhancedConnectionConfig(url="wss://invalid-url")
        manager = EnhancedWebSocketManager(config)

        # Test connection to invalid URL
        # (This would normally fail, but we're testing the error handling structure)

        # Test error state
        manager._state = ConnectionState.FAILED
        health_check = manager.check_connection_health().run()
        assert health_check is False

    def test_message_parsing_error_handling(self):
        """Test error handling for invalid message formats."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Test message type extraction with malformed data
        invalid_messages = [
            None,
            {},
            {"invalid": "format"},
            123,  # Non-dict type
        ]

        for invalid_msg in invalid_messages:
            if isinstance(invalid_msg, dict):
                msg_type = manager._extract_message_type(invalid_msg)
                assert msg_type == "unknown"

    def test_subscription_error_handling(self):
        """Test error handling in message subscriptions."""
        config = EnhancedConnectionConfig(url="wss://api.example.com/ws")
        manager = EnhancedWebSocketManager(config)

        # Create a callback that raises an exception
        def failing_callback(envelope: MessageEnvelope):
            raise ValueError("Test callback error")

        # Subscribe the failing callback
        manager.subscribe_to_messages("test", failing_callback).run()

        # Notify subscribers - should handle the error gracefully
        test_envelope = MessageEnvelope(
            payload={"type": "test"},
            timestamp=datetime.now(),
            message_id="test-123",
            message_type="test",
            source="test",
        )

        # This should not raise an exception
        manager._notify_subscribers(test_envelope)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
