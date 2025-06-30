"""
Docker-specific Orderbook Tests

This module contains tests that specifically validate orderbook functionality
within the Docker test environment, including integration with mock services
and container networking.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from decimal import Decimal

import pytest
import websockets
from httpx import AsyncClient


class TestOrderbookDockerIntegration:
    """Test orderbook functionality in Docker environment."""

    @pytest.mark.asyncio
    async def test_mock_bluefin_orderbook_api(self):
        """Test orderbook retrieval from mock Bluefin service."""
        async with AsyncClient(base_url="http://mock-bluefin:8080") as client:
            # Test health check first
            response = await client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert health_data["service"] == "bluefin"

            # Test orderbook retrieval
            response = await client.get("/orderbook/BTC-USD")
            assert response.status_code == 200

            orderbook_data = response.json()
            assert orderbook_data["symbol"] == "BTC-USD"
            assert "bids" in orderbook_data
            assert "asks" in orderbook_data
            assert "timestamp" in orderbook_data
            assert "sequence" in orderbook_data

            # Validate orderbook structure
            assert len(orderbook_data["bids"]) > 0
            assert len(orderbook_data["asks"]) > 0

            # Validate price ordering
            bid_prices = [Decimal(bid[0]) for bid in orderbook_data["bids"]]
            ask_prices = [Decimal(ask[0]) for ask in orderbook_data["asks"]]

            # Bids should be in descending order
            assert bid_prices == sorted(bid_prices, reverse=True)

            # Asks should be in ascending order
            assert ask_prices == sorted(ask_prices)

            # Best bid should be lower than best ask
            assert bid_prices[0] < ask_prices[0]

    @pytest.mark.asyncio
    async def test_mock_coinbase_orderbook_api(self):
        """Test orderbook retrieval from mock Coinbase service."""
        async with AsyncClient(base_url="http://mock-coinbase:8081") as client:
            # Test health check
            response = await client.get("/health")
            assert response.status_code == 200

            # Test product listing
            response = await client.get("/api/v3/brokerage/products")
            assert response.status_code == 200
            products_data = response.json()
            assert "products" in products_data
            assert len(products_data["products"]) > 0

            # Test orderbook retrieval
            response = await client.get("/api/v3/brokerage/products/BTC-USD/book")
            assert response.status_code == 200

            orderbook_data = response.json()
            assert "pricebook" in orderbook_data

            pricebook = orderbook_data["pricebook"]
            assert pricebook["product_id"] == "BTC-USD"
            assert "bids" in pricebook
            assert "asks" in pricebook
            assert "time" in pricebook

            # Validate Coinbase-specific format
            assert len(pricebook["bids"]) > 0
            assert len(pricebook["asks"]) > 0

            # Each level should be [price, size] string array
            for bid in pricebook["bids"][:5]:
                assert len(bid) == 2
                assert isinstance(bid[0], str)  # price as string
                assert isinstance(bid[1], str)  # size as string
                # Should be valid decimals
                Decimal(bid[0])
                Decimal(bid[1])

    @pytest.mark.asyncio
    async def test_mock_exchange_websocket(self):
        """Test WebSocket orderbook feed from mock exchange."""
        messages_received = []
        connection_established = False

        try:
            async with websockets.connect("ws://mock-exchange:8082/ws") as websocket:
                connection_established = True

                # Send ping to test connectivity
                ping_msg = {"type": "ping", "timestamp": datetime.now(UTC).isoformat()}
                await websocket.send(json.dumps(ping_msg))

                # Collect messages for a short time
                timeout_time = time.time() + 5  # 5 seconds

                while time.time() < timeout_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        messages_received.append(data)

                        # Break if we get enough messages
                        if len(messages_received) >= 5:
                            break

                    except TimeoutError:
                        continue

        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")

        # Validate connection was established
        assert connection_established, "Failed to establish WebSocket connection"

        # Validate we received messages
        assert len(messages_received) > 0, "No messages received from WebSocket"

        # Validate message structure
        for message in messages_received:
            assert "type" in message

            if message["type"] == "pong":
                assert "timestamp" in message
            elif message["type"] in ["orderbook_snapshot", "orderbook_update"]:
                assert "symbol" in message
                assert "timestamp" in message
                assert "sequence" in message

                if message["type"] == "orderbook_snapshot":
                    assert "bids" in message
                    assert "asks" in message
                elif message["type"] == "orderbook_update":
                    assert "updates" in message

    @pytest.mark.asyncio
    async def test_database_connectivity(self):
        """Test that we can connect to the test database."""
        try:
            import psycopg2

            # Test database connection
            conn = psycopg2.connect(
                host="test-postgres",
                port=5432,
                database="orderbook_test",
                user="test_user",
                password="test_password",
            )

            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()

            assert version is not None
            assert "PostgreSQL" in version[0]

            # Test test tables exist
            cursor.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%test%'
            """
            )

            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]

            # Verify key test tables exist
            expected_tables = ["test_runs", "test_results", "performance_benchmarks"]

            for expected_table in expected_tables:
                assert expected_table in table_names, f"Missing table: {expected_table}"

            cursor.close()
            conn.close()

        except ImportError:
            pytest.skip("psycopg2 not available")
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")

    @pytest.mark.asyncio
    async def test_redis_connectivity(self):
        """Test that we can connect to the test Redis instance."""
        try:
            import redis

            # Test Redis connection
            client = redis.Redis(host="test-redis", port=6379, db=0)

            # Test basic operations
            client.set("test_key", "test_value")
            value = client.get("test_key")

            assert value.decode() == "test_value"

            # Cleanup
            client.delete("test_key")
            client.close()

        except ImportError:
            pytest.skip("redis not available")
        except Exception as e:
            pytest.fail(f"Redis connection failed: {e}")

    def test_orderbook_type_creation(self):
        """Test that we can create OrderBook types in the test environment."""
        try:
            from bot.fp.types.market import OrderBook

            # Create test orderbook
            orderbook = OrderBook(
                bids=[
                    (Decimal("50000.00"), Decimal("1.0")),
                    (Decimal("49950.00"), Decimal("2.0")),
                ],
                asks=[
                    (Decimal("50050.00"), Decimal("1.5")),
                    (Decimal("50100.00"), Decimal("2.5")),
                ],
                timestamp=datetime.now(UTC),
            )

            # Test basic properties
            assert orderbook.best_bid == (Decimal("50000.00"), Decimal("1.0"))
            assert orderbook.best_ask == (Decimal("50050.00"), Decimal("1.5"))
            assert orderbook.mid_price == Decimal("50025.00")
            assert orderbook.spread == Decimal("50.00")

        except ImportError as e:
            pytest.fail(f"Cannot import OrderBook type: {e}")

    def test_environment_variables(self):
        """Test that test environment variables are properly set."""
        import os

        # Test mode should be enabled
        assert os.getenv("TEST_MODE") == "true"

        # Mock service URLs should be configured
        assert os.getenv("MOCK_BLUEFIN_URL") == "http://mock-bluefin:8080"
        assert os.getenv("MOCK_COINBASE_URL") == "http://mock-coinbase:8081"
        assert os.getenv("MOCK_EXCHANGE_WS_URL") == "ws://mock-exchange:8082/ws"

        # Test database should be configured
        assert os.getenv("TEST_DB_HOST") == "test-postgres"
        assert os.getenv("TEST_DB_NAME") == "orderbook_test"

    @pytest.mark.performance
    def test_container_performance(self):
        """Test that the container has adequate performance for testing."""
        import psutil

        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        # Should have at least 1GB available
        assert (
            available_gb >= 1.0
        ), f"Insufficient memory: {available_gb:.2f}GB available"

        # Check CPU cores
        cpu_count = psutil.cpu_count()
        assert cpu_count >= 1, f"Insufficient CPU cores: {cpu_count}"

        # Check disk space
        disk = psutil.disk_usage("/app")
        available_gb = disk.free / (1024**3)

        # Should have at least 1GB free disk space
        assert (
            available_gb >= 1.0
        ), f"Insufficient disk space: {available_gb:.2f}GB available"

    @pytest.mark.asyncio
    async def test_concurrent_mock_service_access(self):
        """Test concurrent access to mock services."""

        async def fetch_orderbook(service_url: str, endpoint: str):
            async with AsyncClient(base_url=service_url) as client:
                response = await client.get(endpoint)
                return response.status_code, response.json()

        # Create concurrent requests to different services
        tasks = [
            fetch_orderbook("http://mock-bluefin:8080", "/orderbook/BTC-USD"),
            fetch_orderbook("http://mock-bluefin:8080", "/orderbook/ETH-USD"),
            fetch_orderbook(
                "http://mock-coinbase:8081", "/api/v3/brokerage/products/BTC-USD/book"
            ),
            fetch_orderbook(
                "http://mock-coinbase:8081", "/api/v3/brokerage/products/ETH-USD/book"
            ),
        ]

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all requests succeeded
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent request {i} failed: {result}")

            status_code, data = result
            assert status_code == 200, f"Request {i} returned status {status_code}"
            assert isinstance(data, dict), f"Request {i} returned invalid data type"

    def test_logging_configuration(self):
        """Test that logging is properly configured in the test environment."""
        import logging
        import os

        # Check log level is set correctly
        log_level = os.getenv("LOG_LEVEL", "INFO")
        assert log_level == "DEBUG"

        # Test that we can create and use a logger
        logger = logging.getLogger("test_orderbook_docker")
        logger.info("Test log message")

        # Logger should be configured
        assert logger.level <= logging.DEBUG or logger.parent.level <= logging.DEBUG
