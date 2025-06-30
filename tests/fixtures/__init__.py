"""
Test Fixtures Package

Comprehensive mock data and test fixtures for orderbook testing and
trading system validation. This package provides:

- Realistic orderbook mock data
- WebSocket message fixtures
- Performance test datasets
- Configuration test scenarios
- Error response fixtures
- High-frequency data streams

Usage:
    from tests.fixtures import create_complete_mock_suite
    mock_suite = create_complete_mock_suite()

    # Access specific fixtures
    orderbook = mock_suite["orderbooks"]["normal"]
    websocket_msgs = mock_suite["websocket_messages"]
    perf_data = mock_suite["performance_datasets"]
"""

from datetime import UTC, datetime
from typing import Any, Dict

from .config_test_fixtures import (
    ConfigTestFixtures,
    ConfigTestScenario,
    ConfigValidationTestSuite,
    create_config_test_suite,
)

# Import all fixture modules
from .orderbook_mock_data import (
    ConfigurationTestFixtures,
    ErrorResponseFixtures,
    OrderBookMockConfig,
    OrderBookMockGenerator,
    PerformanceTestDatasets,
    WebSocketMessageFixtures,
    create_orderbook_mock_suite,
)
from .performance_test_fixtures import (
    LatencyMeasurementFixtures,
    PerformanceDataGenerator,
    PerformanceTestConfig,
    StressTestScenarios,
    SystemResourceMonitor,
    create_performance_test_suite,
)
from .websocket_message_factory import (
    MessageFactoryConfig,
    MessageType,
    MessageValidationFixtures,
    WebSocketMessageFactory,
    create_websocket_message_suite,
)


def create_complete_mock_suite(
    orderbook_config: OrderBookMockConfig = None,
    websocket_config: MessageFactoryConfig = None,
    performance_config: PerformanceTestConfig = None,
) -> dict[str, Any]:
    """
    Create complete mock data suite with all fixture types.

    Args:
        orderbook_config: Configuration for orderbook mock data
        websocket_config: Configuration for WebSocket message factory
        performance_config: Configuration for performance test fixtures

    Returns:
        Dictionary containing all mock data and test fixtures
    """
    # Create individual suites
    orderbook_suite = create_orderbook_mock_suite(orderbook_config)
    websocket_suite = create_websocket_message_suite(websocket_config)
    performance_suite = create_performance_test_suite(performance_config)
    config_suite = create_config_test_suite()

    # Combine into comprehensive suite
    complete_suite = {
        # Core orderbook data
        "orderbooks": {
            "normal": orderbook_suite["normal_orderbook"],
            "illiquid": orderbook_suite["illiquid_orderbook"],
            "volatile": orderbook_suite["volatile_orderbook"],
            "empty": orderbook_suite["empty_orderbook"],
            "single_side_bids": orderbook_suite["single_side_bids"],
            "single_side_asks": orderbook_suite["single_side_asks"],
            "sequence": orderbook_suite["orderbook_sequence"],
        },
        # WebSocket messages
        "websocket_messages": {
            "subscription": websocket_suite["subscription"],
            "subscription_ack": websocket_suite["subscription_ack"],
            "heartbeat": websocket_suite["heartbeat"],
            "snapshots": websocket_suite["snapshots"],
            "l2updates": websocket_suite["l2updates"],
            "trades": websocket_suite["trades"],
            "tickers": websocket_suite["tickers"],
            "stream_sample": websocket_suite["message_stream_sample"],
        },
        # Error scenarios
        "error_scenarios": {
            "http_errors": orderbook_suite["http_errors"],
            "websocket_errors": orderbook_suite["websocket_errors"],
            "malformed_data": orderbook_suite["malformed_data"],
            "crossed_orderbook": orderbook_suite["crossed_orderbook"],
            "malformed_messages": websocket_suite["malformed_messages"],
            "validation_errors": websocket_suite["invalid_samples"],
        },
        # Performance datasets
        "performance_datasets": {
            "large_orderbooks": performance_suite["large_orderbooks"],
            "high_frequency_streams": performance_suite["high_frequency_streams"],
            "burst_load_test": performance_suite["burst_load_test"],
            "memory_growth_test": performance_suite["memory_growth_test"],
            "concurrent_scenarios": performance_suite["concurrent_scenarios"],
            "stress_scenarios": performance_suite["stress_scenarios"],
        },
        # Configuration testing
        "configuration_tests": {
            "scenarios": config_suite["scenarios"],
            "test_suite": config_suite["test_suite"],
            "fixtures": config_suite["fixtures"],
        },
        # Validation and testing utilities
        "test_utilities": {
            "latency_functions": performance_suite["latency_test_functions"],
            "resource_monitor": performance_suite["resource_monitor"],
            "message_validator": websocket_suite["valid_samples"],
            "config_validator": config_suite["test_suite"],
        },
        # Raw suites for advanced usage
        "raw_suites": {
            "orderbook_suite": orderbook_suite,
            "websocket_suite": websocket_suite,
            "performance_suite": performance_suite,
            "config_suite": config_suite,
        },
        # Metadata and summary
        "metadata": {
            "generation_timestamp": datetime.now(UTC),
            "total_orderbooks": len(orderbook_suite["orderbook_sequence"]) + 6,
            "total_websocket_messages": len(websocket_suite["message_stream_sample"]),
            "total_error_scenarios": (
                len(orderbook_suite["http_errors"])
                + len(orderbook_suite["websocket_errors"])
                + len(orderbook_suite["malformed_data"])
                + len(websocket_suite["malformed_messages"])
            ),
            "performance_datasets": len(performance_suite["large_orderbooks"]),
            "config_test_scenarios": config_suite["summary"]["total_scenarios"],
            "capabilities": [
                "orderbook_mocking",
                "websocket_simulation",
                "performance_testing",
                "error_simulation",
                "configuration_validation",
                "stress_testing",
                "latency_measurement",
                "resource_monitoring",
            ],
        },
    }

    return complete_suite


def create_minimal_mock_suite() -> dict[str, Any]:
    """
    Create minimal mock data suite for basic testing.

    Returns:
        Dictionary with essential mock data for quick testing
    """
    orderbook_generator = OrderBookMockGenerator()
    websocket_factory = WebSocketMessageFactory()

    return {
        "basic_orderbook": orderbook_generator.generate_normal_orderbook(),
        "basic_l2update": websocket_factory.generate_l2update_message(),
        "basic_trade": websocket_factory.generate_trade_message(),
        "basic_ticker": websocket_factory.generate_ticker_message(),
        "basic_error": websocket_factory.generate_error_message(),
        "metadata": {
            "type": "minimal_suite",
            "generation_timestamp": datetime.now(UTC),
            "intended_use": "basic_testing_and_development",
        },
    }


def create_testing_mock_suite() -> dict[str, Any]:
    """
    Create focused mock data suite for specific testing scenarios.

    Returns:
        Dictionary with testing-focused mock data
    """
    return {
        # Testing specific orderbook conditions
        "testing_orderbooks": {
            "normal": OrderBookMockGenerator().generate_normal_orderbook(),
            "volatile": OrderBookMockGenerator().generate_volatile_orderbook(),
            "illiquid": OrderBookMockGenerator().generate_illiquid_orderbook(),
            "empty": OrderBookMockGenerator().generate_empty_orderbook(),
        },
        # Testing specific message types
        "testing_messages": {
            "l2update": WebSocketMessageFactory().generate_l2update_message(),
            "trade": WebSocketMessageFactory().generate_trade_message(),
            "ticker": WebSocketMessageFactory().generate_ticker_message(),
            "error": WebSocketMessageFactory().generate_error_message(),
            "malformed": WebSocketMessageFactory().generate_malformed_message(),
        },
        # Edge cases for testing
        "edge_cases": {
            "crossed_orderbook": OrderBookMockGenerator().generate_crossed_orderbook(),
            "single_side_book": OrderBookMockGenerator().generate_single_side_orderbook(),
            "empty_orderbook": OrderBookMockGenerator().generate_empty_orderbook(),
        },
        "metadata": {
            "type": "testing_suite",
            "generation_timestamp": datetime.now(UTC),
            "intended_use": "unit_and_integration_testing",
        },
    }


# Export all main classes and functions
__all__ = [
    # Configuration classes
    "OrderBookMockConfig",
    "MessageFactoryConfig",
    "PerformanceTestConfig",
    "ConfigTestScenario",
    # Generator classes
    "OrderBookMockGenerator",
    "WebSocketMessageFactory",
    "PerformanceDataGenerator",
    "ConfigTestFixtures",
    # Utility classes
    "WebSocketMessageFixtures",
    "ErrorResponseFixtures",
    "PerformanceTestDatasets",
    "ConfigurationTestFixtures",
    "LatencyMeasurementFixtures",
    "SystemResourceMonitor",
    "StressTestScenarios",
    "MessageValidationFixtures",
    "ConfigValidationTestSuite",
    # Factory functions
    "create_orderbook_mock_suite",
    "create_websocket_message_suite",
    "create_performance_test_suite",
    "create_config_test_suite",
    "create_complete_mock_suite",
    "create_minimal_mock_suite",
    "create_testing_mock_suite",
    # Enums
    "MessageType",
]
