"""
Integration Test Configuration

This module provides configuration and fixtures for integration testing
of the complete orderbook flow across all system components.
"""

import asyncio
import os
from typing import Any

import pytest

# Test environment configuration
TEST_CONFIG = {
    "exchanges": {
        "coinbase": {
            "enabled": True,
            "base_url": "https://api.coinbase.com",
            "ws_url": "wss://ws.coinbase.com",
            "timeout": 30,
        },
        "bluefin": {
            "enabled": True,
            "base_url": "http://localhost:8080",
            "timeout": 30,
        },
    },
    "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "test_timeouts": {"fast": 5, "medium": 30, "slow": 120},
    "performance_thresholds": {
        "orderbook_update_latency_ms": 100,
        "order_placement_latency_ms": 500,
        "websocket_reconnect_time_s": 10,
        "max_memory_usage_mb": 512,
    },
}


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_credentials():
    """Provide mock credentials for testing."""
    return {
        "coinbase": {
            "api_key": "test_coinbase_key",
            "api_secret": "test_coinbase_secret",
            "passphrase": "test_passphrase",
        },
        "bluefin": {
            "private_key": "0x" + "a" * 64,  # Mock private key
            "network": "testnet",
        },
        "openai": {"api_key": "test_openai_key"},
    }


@pytest.fixture
async def integration_test_environment():
    """Set up complete integration test environment."""

    class IntegrationTestEnvironment:
        def __init__(self):
            self.services = {}
            self.clients = {}
            self.test_data = {}
            self.cleanup_tasks = []

        async def setup_mock_services(self):
            """Set up mock services for testing."""
            # This would normally start mock service containers
            self.services["bluefin_service"] = "mock://bluefin-service:8080"
            self.services["market_data_service"] = "mock://market-data:9090"

        async def setup_test_clients(self):
            """Set up test clients."""
            from tests.integration.test_market_making_orderbook_integration import (
                TestMarketMakingOrderbookIntegration,
            )
            from tests.integration.test_orderbook_integration import (
                TestOrderBookIntegration,
            )
            from tests.integration.test_sdk_service_integration import (
                TestSDKServiceIntegration,
            )

            # Initialize test client instances
            self.clients["orderbook"] = TestOrderBookIntegration()
            self.clients["market_making"] = TestMarketMakingOrderbookIntegration()
            self.clients["sdk_service"] = TestSDKServiceIntegration()

        async def generate_test_data(self):
            """Generate test data for integration tests."""
            from datetime import UTC, datetime
            from decimal import Decimal

            self.test_data["sample_orderbook"] = {
                "bids": [(Decimal(50000), Decimal("10.0"))],
                "asks": [(Decimal(50050), Decimal("8.0"))],
                "timestamp": datetime.now(UTC),
            }

            self.test_data["sample_trades"] = [
                {
                    "id": "trade_1",
                    "price": Decimal(50025),
                    "size": Decimal("1.5"),
                    "side": "buy",
                    "timestamp": datetime.now(UTC),
                }
            ]

        async def cleanup(self):
            """Clean up test environment."""
            for task in self.cleanup_tasks:
                try:
                    await task()
                except Exception as e:
                    print(f"Cleanup error: {e}")

            self.services.clear()
            self.clients.clear()
            self.test_data.clear()

    env = IntegrationTestEnvironment()
    await env.setup_mock_services()
    await env.setup_test_clients()
    await env.generate_test_data()

    yield env

    await env.cleanup()


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utilities."""

    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}

        def start_timer(self, operation: str):
            """Start timing an operation."""
            import time

            self.start_times[operation] = time.time()

        def end_timer(self, operation: str) -> float:
            """End timing and return duration."""
            import time

            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                if operation not in self.metrics:
                    self.metrics[operation] = []
                self.metrics[operation].append(duration)
                del self.start_times[operation]
                return duration
            return 0.0

        def get_average_time(self, operation: str) -> float:
            """Get average time for an operation."""
            if self.metrics.get(operation):
                return sum(self.metrics[operation]) / len(self.metrics[operation])
            return 0.0

        def get_max_time(self, operation: str) -> float:
            """Get maximum time for an operation."""
            if self.metrics.get(operation):
                return max(self.metrics[operation])
            return 0.0

        def check_threshold(self, operation: str, threshold: float) -> bool:
            """Check if operation meets performance threshold."""
            avg_time = self.get_average_time(operation)
            return avg_time <= threshold

        def get_report(self) -> dict[str, Any]:
            """Generate performance report."""
            report = {}
            for operation, times in self.metrics.items():
                if times:
                    report[operation] = {
                        "count": len(times),
                        "average": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "total": sum(times),
                    }
            return report

    return PerformanceMonitor()


@pytest.fixture
def test_data_generator():
    """Provide test data generation utilities."""

    class TestDataGenerator:
        def __init__(self):
            self.seed = 42

        def generate_orderbook(self, symbol="BTC-USD", levels=5):
            """Generate mock orderbook data."""
            import random
            from datetime import UTC, datetime
            from decimal import Decimal

            random.seed(self.seed)
            base_price = 50000

            bids = []
            asks = []

            for i in range(levels):
                bid_price = Decimal(str(base_price - (i * 10)))
                ask_price = Decimal(str(base_price + 50 + (i * 10)))

                bid_size = Decimal(str(random.uniform(1, 20)))
                ask_size = Decimal(str(random.uniform(1, 20)))

                bids.append((bid_price, bid_size))
                asks.append((ask_price, ask_size))

            return {
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now(UTC),
            }

        def generate_trade_sequence(self, count=10, symbol="BTC-USD"):
            """Generate sequence of trades."""
            import random
            from datetime import UTC, datetime, timedelta
            from decimal import Decimal

            random.seed(self.seed)
            trades = []
            base_time = datetime.now(UTC)

            for i in range(count):
                price = Decimal(str(50000 + random.uniform(-1000, 1000)))
                size = Decimal(str(random.uniform(0.1, 10)))
                side = random.choice(["buy", "sell"])

                trade = {
                    "id": f"trade_{i + 1}",
                    "symbol": symbol,
                    "price": price,
                    "size": size,
                    "side": side,
                    "timestamp": base_time + timedelta(seconds=i),
                }
                trades.append(trade)

            return trades

        def generate_websocket_messages(self, count=5, message_type="orderbook"):
            """Generate WebSocket messages."""
            from datetime import UTC, datetime

            messages = []

            for i in range(count):
                if message_type == "orderbook":
                    orderbook_data = self.generate_orderbook()
                    message = {
                        "channel": "level2",
                        "type": "snapshot",
                        "product_id": orderbook_data["symbol"],
                        "bids": [
                            [str(bid[0]), str(bid[1])] for bid in orderbook_data["bids"]
                        ],
                        "asks": [
                            [str(ask[0]), str(ask[1])] for ask in orderbook_data["asks"]
                        ],
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                else:
                    message = {
                        "channel": "ticker",
                        "type": "update",
                        "product_id": "BTC-USD",
                        "price": "50000.0",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                messages.append(message)

            return messages

    return TestDataGenerator()


# Pytest markers for test organization
pytest_markers = [
    "integration: marks tests as integration tests",
    "slow: marks tests as slow running",
    "network: marks tests requiring network access",
    "performance: marks tests that measure performance",
    "orderbook: marks tests related to orderbook functionality",
    "market_making: marks tests related to market making",
    "websocket: marks tests using WebSocket connections",
    "sdk: marks tests using SDK clients",
    "service: marks tests using service clients",
]


def pytest_configure(config):
    """Configure pytest markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add specific markers based on test names
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "websocket" in item.name:
            item.add_marker(pytest.mark.websocket)
        if "orderbook" in item.name:
            item.add_marker(pytest.mark.orderbook)
        if "market_making" in item.name:
            item.add_marker(pytest.mark.market_making)
        if "sdk" in item.name:
            item.add_marker(pytest.mark.sdk)
        if "service" in item.name:
            item.add_marker(pytest.mark.service)


# Environment variable defaults for testing
TEST_ENV_DEFAULTS = {
    "TRADING_MODE": "paper",
    "EXCHANGE_TYPE": "coinbase",
    "LOG_LEVEL": "DEBUG",
    "ENABLE_WEBSOCKET": "true",
    "ENABLE_MEMORY": "false",
    "ENABLE_RISK_MANAGEMENT": "true",
    "CACHE_TTL_SECONDS": "5",
    "MAX_CONCURRENT_POSITIONS": "1",
    "DEFAULT_POSITION_SIZE": "0.01",
}


@pytest.fixture(autouse=True)
def set_test_env_vars():
    """Automatically set test environment variables."""
    original_values = {}

    # Set test defaults
    for key, value in TEST_ENV_DEFAULTS.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value
