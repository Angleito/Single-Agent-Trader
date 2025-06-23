#!/usr/bin/env python3
"""
Test script for the performance monitoring system.

This script tests all monitoring capabilities and verifies metrics are being collected.
"""

import asyncio
import random

# Add project root to path
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.performance_monitor import (
    get_monitor,
    monitor_db_query,
    monitor_order_execution,
    monitor_response_time,
)


# Test functions with varying performance characteristics
@monitor_response_time("fast_operation")
async def fast_operation():
    """Simulates a fast operation."""
    await asyncio.sleep(random.uniform(0.001, 0.01))
    return {"status": "success", "data": random.random()}


@monitor_response_time("slow_operation")
async def slow_operation():
    """Simulates a slow operation."""
    await asyncio.sleep(random.uniform(0.5, 2.0))
    return {"status": "success", "data": random.random()}


@monitor_order_execution("market_order")
async def simulate_market_order():
    """Simulates a market order."""
    await asyncio.sleep(random.uniform(0.01, 0.1))
    if random.random() > 0.95:  # 5% failure rate
        raise Exception("Order failed")
    return {"order_id": f"MKT-{int(time.time())}", "filled": True}


@monitor_order_execution("limit_order")
async def simulate_limit_order():
    """Simulates a limit order."""
    await asyncio.sleep(random.uniform(0.02, 0.2))
    return {"order_id": f"LMT-{int(time.time())}", "filled": random.random() > 0.3}


@monitor_db_query("select", "experiences")
async def simulate_db_select():
    """Simulates a database SELECT query."""
    await asyncio.sleep(random.uniform(0.001, 0.05))
    return [{"id": i, "data": random.random()} for i in range(random.randint(1, 10))]


@monitor_db_query("insert", "trades")
async def simulate_db_insert():
    """Simulates a database INSERT query."""
    await asyncio.sleep(random.uniform(0.005, 0.02))
    return {"inserted_id": int(time.time())}


@monitor_db_query("update", "positions")
async def simulate_db_update():
    """Simulates a database UPDATE query."""
    await asyncio.sleep(random.uniform(0.003, 0.015))
    return {"updated_rows": random.randint(1, 5)}


async def stress_test_operations(duration_seconds: int = 60):
    """Run stress test with multiple concurrent operations."""
    monitor = get_monitor()
    start_time = time.time()
    operations_count = 0

    print(f"Starting stress test for {duration_seconds} seconds...")

    while time.time() - start_time < duration_seconds:
        # Create multiple concurrent operations
        tasks = []

        # Mix of different operations
        for _ in range(random.randint(1, 5)):
            tasks.append(fast_operation())

        if random.random() > 0.7:
            tasks.append(slow_operation())

        # Order operations
        if random.random() > 0.5:
            tasks.append(simulate_market_order())
        if random.random() > 0.6:
            tasks.append(simulate_limit_order())

        # Database operations
        for _ in range(random.randint(1, 3)):
            tasks.append(simulate_db_select())
        if random.random() > 0.4:
            tasks.append(simulate_db_insert())
        if random.random() > 0.5:
            tasks.append(simulate_db_update())

        # Execute concurrently
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            operations_count += len(tasks)
        except Exception as e:
            print(f"Error in operations: {e}")

        # Small delay between batches
        await asyncio.sleep(0.1)

    print(f"Completed {operations_count} operations in {duration_seconds} seconds")
    print(f"Average: {operations_count / duration_seconds:.1f} operations/second")


async def verify_metrics():
    """Verify that metrics are being collected properly."""
    print("\nVerifying metrics collection...")

    async with aiohttp.ClientSession() as session:
        # Check metrics endpoint
        try:
            async with session.get("http://localhost:9090/metrics") as resp:
                if resp.status == 200:
                    metrics_text = await resp.text()

                    # Check for key metrics
                    expected_metrics = [
                        "trading_bot_response_seconds",
                        "order_execution_latency_milliseconds",
                        "database_query_duration_milliseconds",
                        "memory_usage_bytes",
                        "cpu_usage_percent",
                        "network_io_bytes_per_second",
                        "bot_uptime_seconds",
                        "bot_health_status",
                    ]

                    found_metrics = []
                    missing_metrics = []

                    for metric in expected_metrics:
                        if metric in metrics_text:
                            found_metrics.append(metric)
                        else:
                            missing_metrics.append(metric)

                    print(
                        f"\nFound metrics ({len(found_metrics)}/{len(expected_metrics)}):"
                    )
                    for metric in found_metrics:
                        print(f"  ✓ {metric}")

                    if missing_metrics:
                        print("\nMissing metrics:")
                        for metric in missing_metrics:
                            print(f"  ✗ {metric}")

                    # Show sample of actual metrics
                    print("\nSample metrics output:")
                    lines = metrics_text.split("\n")
                    for line in lines[:20]:  # First 20 lines
                        if line and not line.startswith("#"):
                            print(f"  {line}")

                else:
                    print(f"Metrics endpoint returned status {resp.status}")

        except aiohttp.ClientError as e:
            print(f"Failed to connect to metrics endpoint: {e}")

        # Check health endpoint
        try:
            async with session.get("http://localhost:9090/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print("\nHealth check:")
                    print(f"  Status: {health_data.get('status', 'unknown')}")
                    print(f"  Score: {health_data.get('score', 0):.2f}")
                    print(
                        f"  Uptime: {health_data.get('uptime_seconds', 0):.0f} seconds"
                    )
                    print(f"  Metrics tracked: {health_data.get('metrics', {})}")
                else:
                    print(f"Health endpoint returned status {resp.status}")

        except aiohttp.ClientError as e:
            print(f"Failed to connect to health endpoint: {e}")


async def main():
    """Main test function."""
    print("AI Trading Bot Performance Monitor Test")
    print("=" * 50)

    # Start the monitor
    monitor = get_monitor()
    monitor.start_monitoring()

    print(f"Monitor started on port {monitor.port}")
    print(f"Metrics URL: http://localhost:{monitor.port}/metrics")
    print(f"Health URL: http://localhost:{monitor.port}/health")

    # Wait for monitor to initialize
    await asyncio.sleep(2)

    # Run different test phases
    print("\n1. Testing individual operations...")

    # Test each operation type
    print("  - Fast operations")
    for _ in range(5):
        await fast_operation()

    print("  - Slow operations")
    await slow_operation()

    print("  - Order operations")
    try:
        await simulate_market_order()
        await simulate_limit_order()
    except Exception:
        pass  # Expected some failures

    print("  - Database operations")
    await simulate_db_select()
    await simulate_db_insert()
    await simulate_db_update()

    # Verify initial metrics
    await verify_metrics()

    # Run stress test
    print("\n2. Running stress test...")
    await stress_test_operations(duration_seconds=30)

    # Final verification
    await verify_metrics()

    # Show performance summary
    print("\n3. Performance Summary")
    print("=" * 50)

    # Get latest metrics
    if hasattr(monitor, "_response_times") and monitor._response_times:
        response_times = [t[1] for t in list(monitor._response_times)]
        if response_times:
            print(f"Response times (last {len(response_times)} operations):")
            print(f"  Min: {min(response_times) * 1000:.2f}ms")
            print(f"  Max: {max(response_times) * 1000:.2f}ms")
            print(f"  Avg: {sum(response_times) / len(response_times) * 1000:.2f}ms")

    if hasattr(monitor, "_order_latencies") and monitor._order_latencies:
        latencies = [t[1] for t in list(monitor._order_latencies)]
        if latencies:
            print(f"\nOrder latencies (last {len(latencies)} orders):")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
            print(f"  Avg: {sum(latencies) / len(latencies):.2f}ms")

    if hasattr(monitor, "_query_stats") and monitor._query_stats:
        print("\nDatabase query statistics:")
        for query_type, stats in monitor._query_stats.items():
            if stats["count"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                print(f"  {query_type}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg: {avg_time:.2f}ms")
                print(f"    Min: {stats['min_time']:.2f}ms")
                print(f"    Max: {stats['max_time']:.2f}ms")

    print("\nTest completed successfully!")
    print("Monitor will continue running. Press Ctrl+C to stop.")

    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down...")
        monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
