#!/usr/bin/env python3
"""
Integration example for the performance monitoring system.

This shows how to integrate the performance monitor into your trading bot code.
"""

import asyncio
from decimal import Decimal

# Import the monitoring decorators and context managers
from performance_monitor import (
    get_monitor,
    monitor_db_query,
    monitor_order_execution,
    monitor_response_time,
)


# Example 1: Monitoring trading decisions
@monitor_response_time("trading_decision")
async def make_trading_decision(market_data):
    """Example trading decision function with monitoring."""
    # Simulate some processing
    await asyncio.sleep(0.1)

    # Your actual trading logic here
    decision = {"action": "LONG", "confidence": 0.85, "size": 0.25}

    return decision


# Example 2: Monitoring order execution
@monitor_order_execution("market_order")
async def execute_order(order_params):
    """Example order execution with latency monitoring."""
    # Simulate order placement
    await asyncio.sleep(0.05)

    # Your actual order execution logic here
    order_result = {
        "order_id": "12345",
        "status": "filled",
        "fill_price": Decimal("45250.50"),
    }

    return order_result


# Example 3: Monitoring database queries
@monitor_db_query("select", "trading_experiences")
async def fetch_similar_experiences(market_conditions):
    """Example database query with performance monitoring."""
    # Simulate database query
    await asyncio.sleep(0.02)

    # Your actual database logic here
    experiences = [
        {"id": "exp1", "similarity": 0.92},
        {"id": "exp2", "similarity": 0.87},
    ]

    return experiences


# Example 4: Using context managers for more control
async def complex_operation():
    """Example using context managers for fine-grained monitoring."""
    monitor = get_monitor()

    # Monitor overall operation
    with monitor.measure_response_time("complex_operation"):
        # Step 1: Analyze market
        with monitor.measure_response_time("market_analysis"):
            await asyncio.sleep(0.05)
            market_data = {"trend": "bullish"}

        # Step 2: Query historical data
        with monitor.measure_query_time("select", "historical_prices"):
            await asyncio.sleep(0.03)
            historical_data = [45000, 45100, 45200]

        # Step 3: Execute trade
        with monitor.measure_order_latency("limit_order"):
            await asyncio.sleep(0.08)
            order = {"status": "pending"}

    return order


# Example 5: Custom metrics recording
async def record_custom_metrics():
    """Example of recording custom metrics."""
    monitor = get_monitor()

    # Record a custom gauge metric
    if hasattr(monitor, "custom_gauge"):
        monitor.custom_gauge.set(42.5)

    # Record a custom counter
    if hasattr(monitor, "custom_counter"):
        monitor.custom_counter.inc()


# Example main function showing full integration
async def main():
    """Example main function with monitoring integration."""
    # Start the performance monitor
    monitor = get_monitor()
    monitor.start_monitoring()

    print("Performance monitoring started!")
    print(f"Metrics available at: http://localhost:{monitor.port}/metrics")
    print(f"Health check at: http://localhost:{monitor.port}/health")

    # Run some example operations
    for i in range(10):
        print(f"\nIteration {i + 1}:")

        # Make trading decision
        decision = await make_trading_decision({"price": 45000})
        print(f"Decision: {decision}")

        # Execute order
        if decision["action"] != "HOLD":
            order = await execute_order({"type": "market", "size": decision["size"]})
            print(f"Order: {order}")

        # Query experiences
        experiences = await fetch_similar_experiences({"trend": "bullish"})
        print(f"Found {len(experiences)} similar experiences")

        # Complex operation
        result = await complex_operation()
        print(f"Complex operation result: {result}")

        # Wait before next iteration
        await asyncio.sleep(2)

    print("\nMonitoring complete! Check http://localhost:9090/metrics for results")


if __name__ == "__main__":
    asyncio.run(main())
