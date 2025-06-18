#!/usr/bin/env python3
"""
WebSocket Resilience Test
Tests the improved WebSocket implementation for stability and resilience.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebSocketResilienceTest:
    """Test WebSocket resilience and stability improvements."""

    def __init__(self, target_url: str = "ws://localhost:8000/ws"):
        self.target_url = target_url
        self.test_results = []
        self.metrics = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "connection_failures": 0,
            "ping_successes": 0,
            "ping_failures": 0,
            "messages_sent": 0,
            "messages_failed": 0,
            "reconnection_times": [],
            "ping_latencies": [],
        }

    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log a test result."""
        status = "✓ PASS" if success else "✗ FAIL"
        timestamp = datetime.utcnow().isoformat()
        
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": timestamp,
        }
        
        self.test_results.append(result)
        logger.info(f"{status}: {test_name} - {details}")

    async def test_enhanced_connection_parameters(self) -> bool:
        """Test connection with enhanced parameters."""
        try:
            start_time = time.time()
            
            # Test with enhanced connection parameters
            async with websockets.connect(
                self.target_url,
                timeout=30,
                ping_interval=15,
                ping_timeout=8,
                close_timeout=5,
                max_size=2**20,
                compression=None,
                extra_headers={
                    "User-Agent": "WebSocket-Resilience-Test/1.0",
                    "Accept": "*/*",
                    "Connection": "Upgrade"
                }
            ) as websocket:
                connection_time = time.time() - start_time
                self.metrics["connection_attempts"] += 1
                self.metrics["successful_connections"] += 1
                
                self.log_result(
                    "Enhanced Connection Parameters",
                    True,
                    f"Connected in {connection_time:.3f}s with enhanced parameters"
                )
                return True
                
        except Exception as e:
            self.metrics["connection_attempts"] += 1
            self.metrics["connection_failures"] += 1
            self.log_result("Enhanced Connection Parameters", False, str(e))
            return False

    async def test_ping_pong_resilience(self) -> bool:
        """Test ping/pong mechanism resilience."""
        try:
            async with websockets.connect(
                self.target_url,
                ping_interval=10,  # Aggressive ping for testing
                ping_timeout=5
            ) as websocket:
                
                ping_count = 5
                successful_pings = 0
                
                for i in range(ping_count):
                    try:
                        start_time = time.time()
                        pong_waiter = await websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                        
                        latency = time.time() - start_time
                        self.metrics["ping_latencies"].append(latency)
                        self.metrics["ping_successes"] += 1
                        successful_pings += 1
                        
                        logger.debug(f"Ping {i+1}: {latency:.3f}s")
                        await asyncio.sleep(2)  # Wait between pings
                        
                    except asyncio.TimeoutError:
                        self.metrics["ping_failures"] += 1
                        logger.warning(f"Ping {i+1} timeout")
                    except Exception as e:
                        self.metrics["ping_failures"] += 1
                        logger.warning(f"Ping {i+1} failed: {e}")
                
                success_rate = successful_pings / ping_count
                self.log_result(
                    "Ping/Pong Resilience",
                    success_rate >= 0.8,  # 80% success rate required
                    f"{successful_pings}/{ping_count} pings successful ({success_rate:.1%})"
                )
                return success_rate >= 0.8
                
        except Exception as e:
            self.log_result("Ping/Pong Resilience", False, str(e))
            return False

    async def test_message_queue_stability(self) -> bool:
        """Test message queuing under load."""
        try:
            async with websockets.connect(self.target_url) as websocket:
                
                message_count = 100
                successful_sends = 0
                
                # Send messages rapidly
                for i in range(message_count):
                    try:
                        message = {
                            "type": "test_message",
                            "sequence": i,
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": f"test_data_{i}"
                        }
                        
                        await websocket.send(json.dumps(message))
                        self.metrics["messages_sent"] += 1
                        successful_sends += 1
                        
                        # Small delay to simulate realistic load
                        if i % 10 == 0:
                            await asyncio.sleep(0.1)
                            
                    except Exception as e:
                        self.metrics["messages_failed"] += 1
                        logger.warning(f"Message {i} failed: {e}")
                
                success_rate = successful_sends / message_count
                self.log_result(
                    "Message Queue Stability",
                    success_rate >= 0.95,  # 95% success rate required
                    f"{successful_sends}/{message_count} messages sent ({success_rate:.1%})"
                )
                return success_rate >= 0.95
                
        except Exception as e:
            self.log_result("Message Queue Stability", False, str(e))
            return False

    async def test_reconnection_logic(self) -> bool:
        """Test automatic reconnection logic."""
        try:
            reconnections = 0
            max_reconnections = 3
            
            for attempt in range(max_reconnections):
                try:
                    start_time = time.time()
                    
                    # Connect, close, and reconnect
                    websocket = await websockets.connect(self.target_url)
                    await websocket.close()
                    
                    # Immediate reconnection
                    websocket = await websockets.connect(self.target_url)
                    reconnection_time = time.time() - start_time
                    
                    self.metrics["reconnection_times"].append(reconnection_time)
                    await websocket.close()
                    
                    reconnections += 1
                    logger.debug(f"Reconnection {attempt+1}: {reconnection_time:.3f}s")
                    
                except Exception as e:
                    logger.warning(f"Reconnection {attempt+1} failed: {e}")
            
            success_rate = reconnections / max_reconnections
            avg_reconnection_time = sum(self.metrics["reconnection_times"]) / len(self.metrics["reconnection_times"]) if self.metrics["reconnection_times"] else 0
            
            self.log_result(
                "Reconnection Logic",
                success_rate >= 0.8,  # 80% success rate required
                f"{reconnections}/{max_reconnections} reconnections successful, avg: {avg_reconnection_time:.3f}s"
            )
            return success_rate >= 0.8
            
        except Exception as e:
            self.log_result("Reconnection Logic", False, str(e))
            return False

    async def test_connection_stability_under_load(self) -> bool:
        """Test connection stability under sustained load."""
        try:
            duration = 60  # 1 minute test
            start_time = time.time()
            
            async with websockets.connect(
                self.target_url,
                ping_interval=10,
                ping_timeout=5
            ) as websocket:
                
                messages_sent = 0
                errors = 0
                
                while time.time() - start_time < duration:
                    try:
                        message = {
                            "type": "load_test",
                            "timestamp": datetime.utcnow().isoformat(),
                            "counter": messages_sent
                        }
                        
                        await websocket.send(json.dumps(message))
                        messages_sent += 1
                        
                        # Test ping periodically
                        if messages_sent % 50 == 0:
                            try:
                                pong_waiter = await websocket.ping()
                                await asyncio.wait_for(pong_waiter, timeout=5)
                            except Exception:
                                errors += 1
                        
                        await asyncio.sleep(0.5)  # 2 messages per second
                        
                    except ConnectionClosed:
                        logger.warning("Connection lost during load test")
                        errors += 1
                        break
                    except Exception as e:
                        logger.warning(f"Error during load test: {e}")
                        errors += 1
                
                elapsed = time.time() - start_time
                error_rate = errors / max(messages_sent, 1)
                
                self.log_result(
                    "Connection Stability Under Load",
                    error_rate < 0.05,  # Less than 5% error rate
                    f"{messages_sent} messages in {elapsed:.1f}s, {errors} errors ({error_rate:.1%})"
                )
                return error_rate < 0.05
                
        except Exception as e:
            self.log_result("Connection Stability Under Load", False, str(e))
            return False

    async def run_all_tests(self):
        """Run all resilience tests."""
        logger.info("=" * 60)
        logger.info("WebSocket Resilience Test Suite")
        logger.info("=" * 60)
        
        tests = [
            self.test_enhanced_connection_parameters,
            self.test_ping_pong_resilience,
            self.test_message_queue_stability,
            self.test_reconnection_logic,
            self.test_connection_stability_under_load,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            logger.info(f"\nRunning {test.__name__}...")
            try:
                result = await test()
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
        
        # Print summary
        self.print_summary(passed, total)

    def print_summary(self, passed: int, total: int):
        """Print test results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("WebSocket Resilience Test Summary")
        logger.info("=" * 60)
        
        logger.info(f"Tests Passed: {passed}/{total} ({passed/total:.1%})")
        
        # Print metrics
        logger.info("\nConnection Metrics:")
        logger.info(f"  Connection Attempts: {self.metrics['connection_attempts']}")
        logger.info(f"  Successful Connections: {self.metrics['successful_connections']}")
        logger.info(f"  Connection Failures: {self.metrics['connection_failures']}")
        
        logger.info("\nPing/Pong Metrics:")
        logger.info(f"  Ping Successes: {self.metrics['ping_successes']}")
        logger.info(f"  Ping Failures: {self.metrics['ping_failures']}")
        
        if self.metrics["ping_latencies"]:
            avg_latency = sum(self.metrics["ping_latencies"]) / len(self.metrics["ping_latencies"])
            max_latency = max(self.metrics["ping_latencies"])
            min_latency = min(self.metrics["ping_latencies"])
            logger.info(f"  Ping Latency - Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s")
        
        logger.info("\nMessage Metrics:")
        logger.info(f"  Messages Sent: {self.metrics['messages_sent']}")
        logger.info(f"  Messages Failed: {self.metrics['messages_failed']}")
        
        if self.metrics["reconnection_times"]:
            avg_reconnect = sum(self.metrics["reconnection_times"]) / len(self.metrics["reconnection_times"])
            logger.info(f"  Average Reconnection Time: {avg_reconnect:.3f}s")
        
        # Overall result
        if passed == total:
            logger.info("\n🎉 All tests passed! WebSocket resilience improvements are working.")
        elif passed >= total * 0.8:
            logger.info(f"\n⚠️  Most tests passed ({passed}/{total}). Some issues may remain.")
        else:
            logger.info(f"\n❌ Many tests failed ({passed}/{total}). Significant issues detected.")


async def main():
    """Main entry point."""
    import sys
    
    # Get target URL from command line or use default
    target_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"
    
    logger.info(f"Testing WebSocket resilience against: {target_url}")
    
    tester = WebSocketResilienceTest(target_url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())