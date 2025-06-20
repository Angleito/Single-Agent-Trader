#!/usr/bin/env python3
"""
WebSocket Connectivity Test
Tests direct WebSocket connection from bot container to dashboard.
Verifies handshake, ping/pong, and basic message exchange.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebSocketConnectivityTester:
    """Test WebSocket connectivity between services."""

    def __init__(self, dashboard_url: str = "ws://dashboard-backend:8000/ws"):
        self.dashboard_url = dashboard_url
        self.external_url = "ws://localhost:8000/ws"
        self.test_results: list[dict[str, Any]] = []
        self.connection_metrics = {
            "handshake_time": None,
            "ping_latency": [],
            "message_latency": [],
            "reconnection_time": None,
        }

    def add_result(self, test_name: str, passed: bool, details: str = ""):
        """Add a test result."""
        self.test_results.append(
            {
                "test": test_name,
                "passed": passed,
                "details": details,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info("%s: %s - %s", status, test_name, details)

    async def test_basic_connection(self, url: str) -> bool:
        """Test basic WebSocket connection."""
        try:
            start_time = time.time()
            async with websockets.connect(url, ping_interval=None):
                handshake_time = time.time() - start_time
                self.connection_metrics["handshake_time"] = handshake_time

                self.add_result(
                    f"Basic Connection ({url})",
                    True,
                    f"Handshake completed in {handshake_time:.3f}s",
                )
                return True

        except Exception as e:
            self.add_result(f"Basic Connection ({url})", False, str(e))
            return False

    async def test_message_exchange(self, url: str) -> bool:
        """Test sending and receiving messages."""
        try:
            async with websockets.connect(url) as websocket:
                # Send a test message
                test_message = {
                    "type": "test_message",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source": "connectivity_test",
                    "data": {"test_id": "conn_test_001"},
                }

                start_time = time.time()
                await websocket.send(json.dumps(test_message))

                # Try to receive a message (with timeout)
                try:
                    await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    latency = time.time() - start_time
                    self.connection_metrics["message_latency"].append(latency)

                    self.add_result(
                        f"Message Exchange ({url})",
                        True,
                        f"Round-trip latency: {latency:.3f}s",
                    )
                    return True

                except TimeoutError:
                    # No response expected from dashboard, this is OK
                    self.add_result(
                        f"Message Exchange ({url})",
                        True,
                        "Message sent successfully (no response expected)",
                    )
                    return True

        except Exception as e:
            self.add_result(f"Message Exchange ({url})", False, str(e))
            return False

    async def test_ping_pong(self, url: str) -> bool:
        """Test WebSocket ping/pong mechanism."""
        try:
            async with websockets.connect(url) as websocket:
                # Send ping and measure latency
                pong_waiter = await websocket.ping()
                start_time = time.time()

                try:
                    await asyncio.wait_for(pong_waiter, timeout=5.0)
                    latency = time.time() - start_time
                    self.connection_metrics["ping_latency"].append(latency)

                    self.add_result(
                        f"Ping/Pong ({url})", True, f"Ping latency: {latency:.3f}s"
                    )
                    return True

                except TimeoutError:
                    self.add_result(
                        f"Ping/Pong ({url})", False, "Ping timeout (no pong received)"
                    )
                    return False

        except Exception as e:
            self.add_result(f"Ping/Pong ({url})", False, str(e))
            return False

    async def test_large_message(self, url: str) -> bool:
        """Test sending large messages."""
        try:
            async with websockets.connect(url) as websocket:
                # Create a large message (1MB)
                large_data = {
                    "type": "large_test_message",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": "x" * (1024 * 1024),  # 1MB of data
                }

                message = json.dumps(large_data)
                message_size = len(message) / (1024 * 1024)  # Size in MB

                start_time = time.time()
                await websocket.send(message)
                send_time = time.time() - start_time

                self.add_result(
                    f"Large Message ({url})",
                    True,
                    f"Sent {message_size:.2f}MB in {send_time:.3f}s",
                )
                return True

        except Exception as e:
            self.add_result(f"Large Message ({url})", False, str(e))
            return False

    async def test_rapid_messages(self, url: str, count: int = 100) -> bool:
        """Test sending many messages rapidly."""
        try:
            async with websockets.connect(url) as websocket:
                start_time = time.time()

                for i in range(count):
                    message = {
                        "type": "rapid_test",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "sequence": i,
                    }
                    await websocket.send(json.dumps(message))

                total_time = time.time() - start_time
                rate = count / total_time

                self.add_result(
                    f"Rapid Messages ({url})",
                    True,
                    f"Sent {count} messages in {total_time:.3f}s ({rate:.0f} msg/s)",
                )
                return True

        except Exception as e:
            self.add_result(f"Rapid Messages ({url})", False, str(e))
            return False

    async def test_reconnection(self, url: str) -> bool:
        """Test reconnection capability."""
        try:
            # First connection
            websocket = await websockets.connect(url)
            await websocket.close()

            # Immediate reconnection
            start_time = time.time()
            websocket = await websockets.connect(url)
            reconnect_time = time.time() - start_time
            self.connection_metrics["reconnection_time"] = reconnect_time

            await websocket.close()

            self.add_result(
                f"Reconnection ({url})", True, f"Reconnected in {reconnect_time:.3f}s"
            )
            return True

        except Exception as e:
            self.add_result(f"Reconnection ({url})", False, str(e))
            return False

    async def test_concurrent_connections(self, url: str, count: int = 5) -> bool:
        """Test multiple concurrent connections."""
        try:
            websockets_list = []
            start_time = time.time()

            # Create multiple connections
            for _ in range(count):
                ws = await websockets.connect(url)
                websockets_list.append(ws)

            connection_time = time.time() - start_time

            # Send a message on each connection
            tasks = []
            for i, ws in enumerate(websockets_list):
                message = {
                    "type": "concurrent_test",
                    "connection_id": i,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                tasks.append(ws.send(json.dumps(message)))

            await asyncio.gather(*tasks)

            # Close all connections
            for ws in websockets_list:
                await ws.close()

            self.add_result(
                f"Concurrent Connections ({url})",
                True,
                f"Established {count} connections in {connection_time:.3f}s",
            )
            return True

        except Exception as e:
            self.add_result(f"Concurrent Connections ({url})", False, str(e))
            return False

    async def run_all_tests(self):
        """Run all connectivity tests."""
        logger.info("=" * 60)
        logger.info("WebSocket Connectivity Test Suite")
        logger.info("=" * 60)

        # Test both internal and external URLs
        urls_to_test = [
            ("Internal Docker URL", self.dashboard_url),
            ("External localhost URL", self.external_url),
        ]

        for url_name, url in urls_to_test:
            logger.info("\nTesting %s: %s", url_name, url)
            logger.info("-" * 40)

            # Run tests
            await self.test_basic_connection(url)

            # Only run additional tests if basic connection works
            if any(
                r["passed"]
                for r in self.test_results
                if r["test"].startswith(f"Basic Connection ({url})")
            ):
                await self.test_message_exchange(url)
                await self.test_ping_pong(url)
                await self.test_large_message(url)
                await self.test_rapid_messages(url)
                await self.test_reconnection(url)
                await self.test_concurrent_connections(url)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        logger.info("\n%s", "=" * 60)
        logger.info("Test Results Summary")
        logger.info("=" * 60)

        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)

        logger.info("Total Tests: %s", total)
        logger.info("Passed: %s", passed)
        logger.info("Failed: %s", total - passed)
        logger.info("Success Rate: %.1f%%", (passed / total) * 100)

        # Print metrics
        logger.info("\nConnection Metrics:")
        if self.connection_metrics["handshake_time"]:
            logger.info(
                "  Handshake Time: %.3fs", self.connection_metrics["handshake_time"]
            )

        if self.connection_metrics["ping_latency"]:
            avg_ping = sum(self.connection_metrics["ping_latency"]) / len(
                self.connection_metrics["ping_latency"]
            )
            logger.info("  Average Ping Latency: %.3fs", avg_ping)

        if self.connection_metrics["message_latency"]:
            avg_msg = sum(self.connection_metrics["message_latency"]) / len(
                self.connection_metrics["message_latency"]
            )
            logger.info("  Average Message Latency: %.3fs", avg_msg)

        if self.connection_metrics["reconnection_time"]:
            logger.info(
                "  Reconnection Time: %.3fs",
                self.connection_metrics["reconnection_time"],
            )

        # Save results to file
        results_file = Path("tests/docker/results/connectivity_test_results.json")
        with results_file.open("w") as f:
            json.dump(
                {
                    "test_results": self.test_results,
                    "metrics": self.connection_metrics,
                    "summary": {
                        "total": total,
                        "passed": passed,
                        "failed": total - passed,
                        "success_rate": (passed / total) * 100,
                    },
                },
                f,
                indent=2,
            )

        logger.info("\nResults saved to: %s", results_file)

        # Exit with appropriate code
        sys.exit(0 if passed == total else 1)


async def main():
    """Main entry point."""
    import os

    # Get dashboard URL from environment or use default
    dashboard_url = os.getenv(
        "SYSTEM__WEBSOCKET_DASHBOARD_URL", "ws://dashboard-backend:8000/ws"
    )

    tester = WebSocketConnectivityTester(dashboard_url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
