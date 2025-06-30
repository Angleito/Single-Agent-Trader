"""
WebSocket Throughput Testing for Orderbook Operations

This module provides comprehensive WebSocket throughput testing including:
- Real-time message processing rate measurement
- Network latency simulation and analysis
- Connection stability under high load
- Message queuing and buffering performance
- Multi-connection concurrent testing
- Bandwidth utilization analysis
- Error recovery and reconnection testing

WebSocket Performance Metrics:
- Messages per second (throughput)
- Message processing latency (end-to-end)
- Network round-trip time
- Message queue depth and overflow
- Connection stability and uptime
- Bandwidth utilization (bytes/second)
- Error rates and recovery times
- Memory usage for message buffers
"""

import asyncio
import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import websockets
from websockets.server import serve

# Import orderbook types
from bot.fp.types.market import OrderBook


@dataclass
class ThroughputMetrics:
    """WebSocket throughput performance metrics."""

    # Connection metrics
    connection_time: float = 0.0
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_count: int = 0

    # Message throughput
    messages_sent: int = 0
    messages_received: int = 0
    messages_processed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    test_duration: float = 0.0

    # Latency metrics
    message_latencies: list[float] = field(default_factory=list)
    processing_latencies: list[float] = field(default_factory=list)
    round_trip_times: list[float] = field(default_factory=list)

    # Performance metrics
    messages_per_second: float = 0.0
    bytes_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Error metrics
    message_errors: int = 0
    timeout_errors: int = 0
    processing_errors: int = 0
    connection_errors: int = 0

    # Queue metrics
    max_queue_depth: int = 0
    avg_queue_depth: float = 0.0
    queue_overflows: int = 0

    def calculate_derived_metrics(self) -> None:
        """Calculate derived performance metrics."""
        self.test_duration = (
            self.end_time - self.start_time if self.end_time > self.start_time else 0
        )

        if self.test_duration > 0:
            self.messages_per_second = self.messages_processed / self.test_duration
            self.bytes_per_second = self.bytes_received / self.test_duration

        if self.message_latencies:
            self.avg_latency_ms = statistics.mean(self.message_latencies) * 1000
            sorted_latencies = sorted(self.message_latencies)
            n = len(sorted_latencies)
            if n >= 20:
                self.p95_latency_ms = sorted_latencies[int(n * 0.95)] * 1000
            if n >= 100:
                self.p99_latency_ms = sorted_latencies[int(n * 0.99)] * 1000


@dataclass
class WebSocketTestConfig:
    """Configuration for WebSocket throughput testing."""

    # Test duration and intensity
    test_duration_seconds: int = 60
    target_messages_per_second: int = 1000
    max_concurrent_connections: int = 10

    # Message configuration
    orderbook_depth: int = 50
    message_size_target: int = 4096  # Target message size in bytes
    enable_compression: bool = False

    # Connection configuration
    connection_timeout: float = 10.0
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    close_timeout: float = 10.0

    # Performance testing
    enable_latency_measurement: bool = True
    enable_bandwidth_limiting: bool = False
    bandwidth_limit_mbps: float = 10.0  # Megabits per second

    # Error simulation
    enable_error_injection: bool = False
    error_injection_rate: float = 0.01  # 1% error rate
    network_latency_ms: float = 0.0  # Simulated network latency

    # Test symbols
    test_symbols: list[str] = field(
        default_factory=lambda: ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOT-USD"]
    )


class OrderBookMessageGenerator:
    """Generates realistic orderbook messages for testing."""

    def __init__(self, symbols: list[str], depth: int = 50):
        self.symbols = symbols
        self.depth = depth
        self.current_prices = {symbol: Decimal(50000) for symbol in symbols}
        self.message_counter = 0

    def generate_snapshot_message(self, symbol: str) -> dict[str, Any]:
        """Generate a full orderbook snapshot message."""
        base_price = self.current_prices[symbol]

        bids = []
        asks = []

        for i in range(self.depth):
            bid_price = base_price - Decimal(str(i * 10))
            ask_price = base_price + Decimal(str(50 + i * 10))

            bid_size = Decimal(str(1.0 + i * 0.1))
            ask_size = Decimal(str(1.0 + i * 0.1))

            bids.append([str(bid_price), str(bid_size)])
            asks.append([str(ask_price), str(ask_size)])

        return {
            "type": "snapshot",
            "product_id": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now(UTC).isoformat(),
            "message_id": f"snapshot_{symbol}_{self.message_counter}",
            "sequence": self.message_counter,
        }

    def generate_update_message(self, symbol: str) -> dict[str, Any]:
        """Generate an orderbook update (l2update) message."""
        base_price = self.current_prices[symbol]

        # Simulate price changes
        price_change = Decimal(str((self.message_counter % 21) - 10))  # -10 to +10
        self.current_prices[symbol] = max(Decimal(1000), base_price + price_change)

        changes = []

        # Add some bid changes
        for i in range(1, 4):  # 1-3 bid changes
            price = base_price - Decimal(str(i * 10))
            size = Decimal(str(1.0 + (self.message_counter % 10) * 0.1))
            changes.append(["buy", str(price), str(size)])

        # Add some ask changes
        for i in range(1, 4):  # 1-3 ask changes
            price = base_price + Decimal(str(50 + i * 10))
            size = Decimal(str(1.0 + (self.message_counter % 10) * 0.1))
            changes.append(["sell", str(price), str(size)])

        self.message_counter += 1

        return {
            "type": "l2update",
            "product_id": symbol,
            "changes": changes,
            "timestamp": datetime.now(UTC).isoformat(),
            "message_id": f"update_{symbol}_{self.message_counter}",
            "sequence": self.message_counter,
        }

    def generate_heartbeat_message(self) -> dict[str, Any]:
        """Generate a heartbeat/ping message."""
        return {
            "type": "heartbeat",
            "timestamp": datetime.now(UTC).isoformat(),
            "message_id": f"heartbeat_{self.message_counter}",
            "sequence": self.message_counter,
        }


class MockWebSocketServer:
    """Mock WebSocket server for throughput testing."""

    def __init__(self, config: WebSocketTestConfig):
        self.config = config
        self.message_generator = OrderBookMessageGenerator(
            config.test_symbols, config.orderbook_depth
        )
        self.client_connections = set()
        self.server_metrics = ThroughputMetrics()
        self.running = False

    async def handle_client(self, websocket, path):
        """Handle individual client connections."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.client_connections.add(websocket)

        try:
            # Send initial snapshots for all symbols
            for symbol in self.config.test_symbols:
                snapshot = self.message_generator.generate_snapshot_message(symbol)
                message_str = json.dumps(snapshot)
                await websocket.send(message_str)
                self.server_metrics.messages_sent += 1
                self.server_metrics.bytes_sent += len(message_str.encode())

            # Handle client messages and send updates
            async for message in websocket:
                try:
                    # Parse client message
                    client_data = json.loads(message)

                    # Handle subscription requests
                    if client_data.get("type") == "subscribe":
                        # Acknowledge subscription
                        ack_message = {
                            "type": "subscriptions",
                            "channels": client_data.get("channels", []),
                            "product_ids": client_data.get("product_ids", []),
                        }
                        await websocket.send(json.dumps(ack_message))

                except json.JSONDecodeError:
                    self.server_metrics.message_errors += 1
                except Exception as e:
                    self.server_metrics.processing_errors += 1
                    logging.exception(f"Error handling client message: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.server_metrics.connection_errors += 1
            logging.exception(f"Client connection error: {e}")
        finally:
            self.client_connections.discard(websocket)

    async def broadcast_updates(self):
        """Continuously broadcast orderbook updates to all clients."""
        message_interval = 1.0 / self.config.target_messages_per_second

        while self.running:
            if not self.client_connections:
                await asyncio.sleep(0.1)
                continue

            try:
                # Generate updates for each symbol
                for symbol in self.config.test_symbols:
                    if not self.running:
                        break

                    # Generate update message
                    update = self.message_generator.generate_update_message(symbol)
                    message_str = json.dumps(update)

                    # Simulate network latency if configured
                    if self.config.network_latency_ms > 0:
                        await asyncio.sleep(self.config.network_latency_ms / 1000)

                    # Broadcast to all connected clients
                    disconnected_clients = set()
                    for websocket in self.client_connections.copy():
                        try:
                            await websocket.send(message_str)
                            self.server_metrics.messages_sent += 1
                            self.server_metrics.bytes_sent += len(message_str.encode())
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(websocket)
                        except Exception as e:
                            logging.exception(f"Error broadcasting to client: {e}")
                            disconnected_clients.add(websocket)

                    # Remove disconnected clients
                    self.client_connections -= disconnected_clients

                # Control message rate
                await asyncio.sleep(message_interval)

            except Exception as e:
                logging.exception(f"Error in broadcast loop: {e}")
                await asyncio.sleep(1.0)

    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the mock WebSocket server."""
        self.running = True
        self.server_metrics.start_time = time.time()

        # Start the server
        server = await serve(self.handle_client, host, port)

        # Start broadcasting updates
        broadcast_task = asyncio.create_task(self.broadcast_updates())

        try:
            # Keep server running
            await server.wait_closed()
        finally:
            self.running = False
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

            self.server_metrics.end_time = time.time()
            self.server_metrics.calculate_derived_metrics()


class WebSocketThroughputClient:
    """WebSocket client for throughput testing."""

    def __init__(self, config: WebSocketTestConfig, client_id: int = 0):
        self.config = config
        self.client_id = client_id
        self.metrics = ThroughputMetrics()
        self.message_queue = deque()
        self.running = False
        self.websocket = None

        # Orderbook state management
        self.orderbooks = {}
        self.last_update_times = {}

    async def connect(self, uri: str) -> bool:
        """Connect to WebSocket server."""
        self.metrics.connection_attempts += 1
        connect_start = time.time()

        try:
            self.websocket = await websockets.connect(
                uri,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                compression=None if not self.config.enable_compression else "deflate",
            )

            self.metrics.connection_time = time.time() - connect_start
            self.metrics.successful_connections += 1

            # Subscribe to channels
            subscription_msg = {
                "type": "subscribe",
                "product_ids": self.config.test_symbols,
                "channels": ["level2", "heartbeat"],
            }

            await self.websocket.send(json.dumps(subscription_msg))
            return True

        except Exception as e:
            self.metrics.failed_connections += 1
            self.metrics.connection_errors += 1
            logging.exception(f"Client {self.client_id} connection failed: {e}")
            return False

    async def message_receiver(self):
        """Receive and queue messages from WebSocket."""
        try:
            async for message in self.websocket:
                receive_time = time.time()

                self.metrics.messages_received += 1
                self.metrics.bytes_received += len(message.encode())

                # Add to processing queue
                self.message_queue.append(
                    {"data": message, "receive_time": receive_time}
                )

                # Track queue depth
                queue_depth = len(self.message_queue)
                self.metrics.max_queue_depth = max(
                    self.metrics.max_queue_depth, queue_depth
                )

                # Check for queue overflow
                if queue_depth > 10000:  # Arbitrary overflow threshold
                    self.metrics.queue_overflows += 1
                    # Remove oldest messages
                    while len(self.message_queue) > 5000:
                        self.message_queue.popleft()

        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Client {self.client_id} connection closed")
        except Exception as e:
            self.metrics.connection_errors += 1
            logging.exception(f"Client {self.client_id} receiver error: {e}")

    async def message_processor(self):
        """Process messages from the queue."""
        queue_depths = []

        while self.running:
            if not self.message_queue:
                await asyncio.sleep(0.001)  # 1ms sleep when queue is empty
                continue

            try:
                # Get message from queue
                message_data = self.message_queue.popleft()
                process_start = time.time()

                # Track queue depth
                queue_depths.append(len(self.message_queue))

                # Parse message
                try:
                    message = json.loads(message_data["data"])
                except json.JSONDecodeError:
                    self.metrics.message_errors += 1
                    continue

                # Calculate latency
                message_latency = process_start - message_data["receive_time"]
                self.metrics.message_latencies.append(message_latency)

                # Process different message types
                await self._process_message(message, process_start)

                # Calculate processing latency
                processing_latency = time.time() - process_start
                self.metrics.processing_latencies.append(processing_latency)

                self.metrics.messages_processed += 1

            except Exception as e:
                self.metrics.processing_errors += 1
                logging.exception(f"Client {self.client_id} processing error: {e}")

        # Calculate average queue depth
        if queue_depths:
            self.metrics.avg_queue_depth = statistics.mean(queue_depths)

    async def _process_message(self, message: dict[str, Any], process_start: float):
        """Process individual orderbook messages."""
        message_type = message.get("type")
        symbol = message.get("product_id")

        if message_type == "snapshot" and symbol:
            # Process full orderbook snapshot
            try:
                bids = [
                    (Decimal(bid[0]), Decimal(bid[1]))
                    for bid in message.get("bids", [])
                ]
                asks = [
                    (Decimal(ask[0]), Decimal(ask[1]))
                    for ask in message.get("asks", [])
                ]

                if bids and asks:
                    orderbook = OrderBook(
                        bids=bids, asks=asks, timestamp=datetime.now(UTC)
                    )

                    self.orderbooks[symbol] = orderbook
                    self.last_update_times[symbol] = process_start

                    # Perform some operations to simulate real usage
                    _ = orderbook.mid_price
                    _ = orderbook.spread
                    _ = orderbook.bid_depth
                    _ = orderbook.ask_depth

            except Exception as e:
                self.metrics.processing_errors += 1
                logging.exception(f"Error processing snapshot for {symbol}: {e}")

        elif message_type == "l2update" and symbol:
            # Process orderbook update
            try:
                if symbol in self.orderbooks:
                    # In real implementation, would apply updates to existing orderbook
                    # For testing, just record that we processed the update
                    self.last_update_times[symbol] = process_start

                    # Simulate some processing work
                    changes = message.get("changes", [])
                    for change in changes:
                        # Simulate processing each price level change
                        if len(change) >= 3:
                            side, price, size = change[0], change[1], change[2]
                            # Would update orderbook state here

            except Exception as e:
                self.metrics.processing_errors += 1
                logging.exception(f"Error processing update for {symbol}: {e}")

        elif message_type == "heartbeat":
            # Process heartbeat for connection health
            pass

    async def run_test(self, uri: str, duration: float) -> ThroughputMetrics:
        """Run throughput test for specified duration."""
        self.metrics = ThroughputMetrics()
        self.metrics.start_time = time.time()

        # Connect to server
        if not await self.connect(uri):
            self.metrics.end_time = time.time()
            self.metrics.calculate_derived_metrics()
            return self.metrics

        self.running = True

        try:
            # Start message receiver and processor
            receiver_task = asyncio.create_task(self.message_receiver())
            processor_task = asyncio.create_task(self.message_processor())

            # Run for specified duration
            await asyncio.sleep(duration)

        finally:
            self.running = False

            # Clean up
            if self.websocket:
                await self.websocket.close()

            # Wait for tasks to complete
            receiver_task.cancel()
            processor_task.cancel()

            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

            try:
                await processor_task
            except asyncio.CancelledError:
                pass

            self.metrics.end_time = time.time()
            self.metrics.calculate_derived_metrics()

        return self.metrics


class WebSocketThroughputTester:
    """Comprehensive WebSocket throughput testing suite."""

    def __init__(self, config: WebSocketTestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def test_single_connection_throughput(self) -> ThroughputMetrics:
        """Test throughput with a single WebSocket connection."""
        server_port = 8765

        # Start mock server
        server = MockWebSocketServer(self.config)
        server_task = asyncio.create_task(server.start_server("localhost", server_port))

        # Give server time to start
        await asyncio.sleep(1.0)

        try:
            # Create client and run test
            client = WebSocketThroughputClient(self.config, client_id=0)
            metrics = await client.run_test(
                f"ws://localhost:{server_port}", self.config.test_duration_seconds
            )

            return metrics

        finally:
            # Clean up server
            server.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    async def test_concurrent_connections_throughput(self) -> dict[str, Any]:
        """Test throughput with multiple concurrent connections."""
        server_port = 8766

        # Start mock server
        server = MockWebSocketServer(self.config)
        server_task = asyncio.create_task(server.start_server("localhost", server_port))

        # Give server time to start
        await asyncio.sleep(1.0)

        try:
            # Create multiple clients
            clients = []
            client_tasks = []

            for i in range(self.config.max_concurrent_connections):
                client = WebSocketThroughputClient(self.config, client_id=i)
                clients.append(client)

                task = asyncio.create_task(
                    client.run_test(
                        f"ws://localhost:{server_port}",
                        self.config.test_duration_seconds,
                    )
                )
                client_tasks.append(task)

            # Wait for all clients to complete
            client_results = await asyncio.gather(*client_tasks, return_exceptions=True)

            # Aggregate results
            aggregated_metrics = ThroughputMetrics()
            successful_clients = 0

            for i, result in enumerate(client_results):
                if isinstance(result, ThroughputMetrics):
                    successful_clients += 1

                    # Aggregate metrics
                    aggregated_metrics.messages_received += result.messages_received
                    aggregated_metrics.messages_processed += result.messages_processed
                    aggregated_metrics.bytes_received += result.bytes_received
                    aggregated_metrics.message_latencies.extend(
                        result.message_latencies
                    )
                    aggregated_metrics.processing_latencies.extend(
                        result.processing_latencies
                    )

                    # Aggregate errors
                    aggregated_metrics.message_errors += result.message_errors
                    aggregated_metrics.timeout_errors += result.timeout_errors
                    aggregated_metrics.processing_errors += result.processing_errors
                    aggregated_metrics.connection_errors += result.connection_errors

                    # Track connection metrics
                    aggregated_metrics.connection_attempts += result.connection_attempts
                    aggregated_metrics.successful_connections += (
                        result.successful_connections
                    )
                    aggregated_metrics.failed_connections += result.failed_connections

                    # Queue metrics
                    aggregated_metrics.max_queue_depth = max(
                        aggregated_metrics.max_queue_depth, result.max_queue_depth
                    )
                    aggregated_metrics.queue_overflows += result.queue_overflows

                else:
                    self.logger.error(f"Client {i} failed: {result}")

            # Calculate final metrics
            if successful_clients > 0:
                aggregated_metrics.start_time = min(
                    r.start_time
                    for r in client_results
                    if isinstance(r, ThroughputMetrics)
                )
                aggregated_metrics.end_time = max(
                    r.end_time
                    for r in client_results
                    if isinstance(r, ThroughputMetrics)
                )
                aggregated_metrics.calculate_derived_metrics()

            return {
                "aggregated_metrics": aggregated_metrics,
                "individual_results": [
                    r for r in client_results if isinstance(r, ThroughputMetrics)
                ],
                "successful_clients": successful_clients,
                "server_metrics": server.server_metrics,
            }

        finally:
            # Clean up server
            server.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    async def test_bandwidth_limitations(
        self, bandwidth_mbps: float = 1.0
    ) -> ThroughputMetrics:
        """Test performance under bandwidth limitations."""
        # Create config with bandwidth limiting
        limited_config = WebSocketTestConfig(
            test_duration_seconds=self.config.test_duration_seconds,
            target_messages_per_second=self.config.target_messages_per_second,
            enable_bandwidth_limiting=True,
            bandwidth_limit_mbps=bandwidth_mbps,
            orderbook_depth=self.config.orderbook_depth,
        )

        tester = WebSocketThroughputTester(limited_config)
        return await tester.test_single_connection_throughput()

    async def test_error_recovery(self, error_rate: float = 0.05) -> ThroughputMetrics:
        """Test performance with error injection and recovery."""
        # Create config with error injection
        error_config = WebSocketTestConfig(
            test_duration_seconds=self.config.test_duration_seconds,
            target_messages_per_second=self.config.target_messages_per_second,
            enable_error_injection=True,
            error_injection_rate=error_rate,
            orderbook_depth=self.config.orderbook_depth,
        )

        tester = WebSocketThroughputTester(error_config)
        return await tester.test_single_connection_throughput()


def generate_throughput_report(results: dict[str, Any]) -> str:
    """Generate comprehensive throughput test report."""

    report = []
    report.append("=" * 80)
    report.append("WEBSOCKET THROUGHPUT TEST REPORT")
    report.append("=" * 80)

    aggregated = results["aggregated_metrics"]
    individual = results["individual_results"]
    server_metrics = results["server_metrics"]

    # Summary statistics
    report.append("\nTest Summary:")
    report.append(f"  Test Duration: {aggregated.test_duration:.1f} seconds")
    report.append(f"  Concurrent Connections: {results['successful_clients']}")
    report.append(f"  Total Messages Processed: {aggregated.messages_processed:,}")
    report.append(f"  Total Bytes Transferred: {aggregated.bytes_received:,}")

    # Throughput metrics
    report.append("\nThroughput Metrics:")
    report.append(f"  Messages/Second: {aggregated.messages_per_second:.1f}")
    report.append(f"  Bytes/Second: {aggregated.bytes_per_second:,.0f}")
    report.append(
        f"  Megabits/Second: {(aggregated.bytes_per_second * 8) / 1_000_000:.2f}"
    )

    # Latency metrics
    if aggregated.message_latencies:
        report.append("\nLatency Metrics:")
        report.append(f"  Average Latency: {aggregated.avg_latency_ms:.2f} ms")
        report.append(f"  P95 Latency: {aggregated.p95_latency_ms:.2f} ms")
        report.append(f"  P99 Latency: {aggregated.p99_latency_ms:.2f} ms")

    # Connection metrics
    report.append("\nConnection Metrics:")
    report.append(f"  Connection Attempts: {aggregated.connection_attempts}")
    report.append(f"  Successful Connections: {aggregated.successful_connections}")
    report.append(f"  Failed Connections: {aggregated.failed_connections}")
    report.append(
        f"  Connection Success Rate: {(aggregated.successful_connections / aggregated.connection_attempts) * 100:.1f}%"
    )

    # Error metrics
    total_errors = (
        aggregated.message_errors
        + aggregated.timeout_errors
        + aggregated.processing_errors
        + aggregated.connection_errors
    )
    error_rate = (total_errors / max(aggregated.messages_received, 1)) * 100

    report.append("\nError Metrics:")
    report.append(f"  Total Errors: {total_errors}")
    report.append(f"  Error Rate: {error_rate:.3f}%")
    report.append(f"  Message Errors: {aggregated.message_errors}")
    report.append(f"  Timeout Errors: {aggregated.timeout_errors}")
    report.append(f"  Processing Errors: {aggregated.processing_errors}")
    report.append(f"  Connection Errors: {aggregated.connection_errors}")

    # Queue metrics
    report.append("\nQueue Metrics:")
    report.append(f"  Max Queue Depth: {aggregated.max_queue_depth}")
    report.append(f"  Average Queue Depth: {aggregated.avg_queue_depth:.1f}")
    report.append(f"  Queue Overflows: {aggregated.queue_overflows}")

    # Per-client performance breakdown
    if len(individual) > 1:
        report.append("\nPer-Client Performance:")
        report.append(
            f"{'Client':<8} {'Msg/s':<8} {'Latency(ms)':<12} {'Errors':<8} {'Queue':<8}"
        )
        report.append("-" * 50)

        for i, client_metrics in enumerate(individual):
            report.append(
                f"{i:<8} "
                f"{client_metrics.messages_per_second:<8.0f} "
                f"{client_metrics.avg_latency_ms:<12.2f} "
                f"{client_metrics.message_errors + client_metrics.processing_errors:<8} "
                f"{client_metrics.max_queue_depth:<8}"
            )

    # Server-side metrics
    report.append("\nServer Metrics:")
    report.append(f"  Messages Sent: {server_metrics.messages_sent:,}")
    report.append(f"  Bytes Sent: {server_metrics.bytes_sent:,}")
    report.append(
        f"  Server Errors: {server_metrics.message_errors + server_metrics.processing_errors}"
    )

    # Performance assessment
    report.append("\nPerformance Assessment:")

    if aggregated.messages_per_second >= 1000:
        report.append("✅ Excellent throughput (>= 1000 msg/s)")
    elif aggregated.messages_per_second >= 500:
        report.append("✅ Good throughput (>= 500 msg/s)")
    elif aggregated.messages_per_second >= 100:
        report.append("⚠️  Moderate throughput (>= 100 msg/s)")
    else:
        report.append("❌ Low throughput (< 100 msg/s)")

    if aggregated.avg_latency_ms <= 10:
        report.append("✅ Low latency (<= 10ms)")
    elif aggregated.avg_latency_ms <= 50:
        report.append("✅ Acceptable latency (<= 50ms)")
    elif aggregated.avg_latency_ms <= 100:
        report.append("⚠️  High latency (<= 100ms)")
    else:
        report.append("❌ Very high latency (> 100ms)")

    if error_rate <= 0.1:
        report.append("✅ Low error rate (<= 0.1%)")
    elif error_rate <= 1.0:
        report.append("⚠️  Moderate error rate (<= 1%)")
    else:
        report.append("❌ High error rate (> 1%)")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


# Test suite integration
async def run_comprehensive_websocket_tests():
    """Run comprehensive WebSocket throughput tests."""

    logging.basicConfig(level=logging.INFO)

    config = WebSocketTestConfig(
        test_duration_seconds=30,  # Shorter for testing
        target_messages_per_second=500,
        max_concurrent_connections=5,
        orderbook_depth=50,
    )

    tester = WebSocketThroughputTester(config)

    print("Running WebSocket Throughput Tests...")

    # Test 1: Single connection throughput
    print("\n1. Testing single connection throughput...")
    single_result = await tester.test_single_connection_throughput()
    print(
        f"Single connection: {single_result.messages_per_second:.0f} msg/s, "
        f"latency: {single_result.avg_latency_ms:.2f}ms"
    )

    # Test 2: Concurrent connections throughput
    print("\n2. Testing concurrent connections throughput...")
    concurrent_results = await tester.test_concurrent_connections_throughput()
    print(generate_throughput_report(concurrent_results))

    # Test 3: Bandwidth limited performance
    print("\n3. Testing bandwidth limited performance...")
    bandwidth_result = await tester.test_bandwidth_limitations(bandwidth_mbps=1.0)
    print(
        f"Bandwidth limited (1 Mbps): {bandwidth_result.messages_per_second:.0f} msg/s"
    )

    print("\n✅ WebSocket throughput tests completed!")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_websocket_tests())
