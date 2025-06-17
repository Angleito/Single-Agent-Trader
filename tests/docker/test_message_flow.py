#!/usr/bin/env python3
"""
Message Flow Validation Tests
Tests that all message types flow correctly from bot to dashboard.
Validates message schemas and data integrity.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""

    MARKET_DATA = "market_data"
    INDICATOR_UPDATE = "indicator_update"
    AI_DECISION = "ai_decision"
    TRADING_DECISION = "trading_decision"
    TRADE_EXECUTION = "trade_execution"
    POSITION_UPDATE = "position_update"
    SYSTEM_STATUS = "system_status"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    ERROR = "error"
    PING = "ping"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_CLOSED = "connection_closed"
    TRADING_LOOP = "trading_loop"


@dataclass
class MessageValidationResult:
    """Result of message validation."""

    message_type: str
    valid: bool
    errors: list[str]
    warnings: list[str]
    data: dict[str, Any] | None = None


class MessageFlowValidator:
    """Validates message flow from bot to dashboard."""

    def __init__(self, dashboard_url: str = "ws://dashboard-backend:8000/ws"):
        self.dashboard_url = dashboard_url
        self.received_messages: list[dict[str, Any]] = []
        self.validation_results: list[MessageValidationResult] = []
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.listening = False

        # Expected message schemas
        self.message_schemas = {
            MessageType.MARKET_DATA: {
                "required": ["type", "timestamp", "symbol", "price"],
                "optional": ["volume", "bid", "ask", "spread"],
            },
            MessageType.INDICATOR_UPDATE: {
                "required": ["type", "timestamp", "symbol", "indicators"],
                "optional": [],
            },
            MessageType.AI_DECISION: {
                "required": ["type", "timestamp", "action", "reasoning", "confidence"],
                "optional": ["market_context", "risk_assessment"],
            },
            MessageType.TRADING_DECISION: {
                "required": [
                    "type",
                    "timestamp",
                    "action",
                    "reasoning",
                    "confidence",
                    "indicators",
                    "risk_analysis",
                ],
                "optional": ["position_size", "stop_loss", "take_profit", "leverage"],
            },
            MessageType.TRADE_EXECUTION: {
                "required": [
                    "type",
                    "timestamp",
                    "order_id",
                    "symbol",
                    "side",
                    "size",
                    "status",
                ],
                "optional": ["price", "filled_size", "remaining_size", "fees"],
            },
            MessageType.POSITION_UPDATE: {
                "required": [
                    "type",
                    "timestamp",
                    "symbol",
                    "side",
                    "size",
                    "entry_price",
                    "current_price",
                    "pnl",
                    "pnl_percentage",
                ],
                "optional": [
                    "unrealized_pnl",
                    "realized_pnl",
                    "margin",
                    "liquidation_price",
                ],
            },
            MessageType.SYSTEM_STATUS: {
                "required": ["type", "timestamp", "status", "health"],
                "optional": ["errors", "warnings", "metrics"],
            },
            MessageType.TRADING_LOOP: {
                "required": [
                    "type",
                    "timestamp",
                    "symbol",
                    "price",
                    "action",
                    "confidence",
                ],
                "optional": ["reasoning", "indicators"],
            },
        }

    async def connect(self) -> bool:
        """Connect to dashboard WebSocket."""
        try:
            self.websocket = await websockets.connect(self.dashboard_url)
            logger.info(f"Connected to dashboard at {self.dashboard_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from dashboard WebSocket."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def start_listening(self):
        """Start listening for messages."""
        if not self.websocket:
            await self.connect()

        self.listening = True
        try:
            while self.listening:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)

                    # Parse and store message
                    try:
                        data = json.loads(message)
                        data["received_at"] = datetime.utcnow().isoformat()
                        self.received_messages.append(data)
                        logger.debug(f"Received message: {data.get('type', 'unknown')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}")

                except TimeoutError:
                    # No message received, continue
                    continue
                except websockets.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break

        except Exception as e:
            logger.error(f"Error in message listener: {e}")

    def validate_message_schema(
        self, message: dict[str, Any]
    ) -> MessageValidationResult:
        """Validate a message against its expected schema."""
        msg_type = message.get("type")

        if not msg_type:
            return MessageValidationResult(
                message_type="unknown",
                valid=False,
                errors=["Missing 'type' field"],
                warnings=[],
            )

        # Get expected schema
        schema = self.message_schemas.get(msg_type)
        if not schema:
            return MessageValidationResult(
                message_type=msg_type,
                valid=True,  # Unknown types are allowed
                errors=[],
                warnings=[f"Unknown message type: {msg_type}"],
                data=message,
            )

        errors = []
        warnings = []

        # Check required fields
        for field in schema["required"]:
            if field not in message:
                errors.append(f"Missing required field: {field}")

        # Check timestamp format
        if "timestamp" in message:
            try:
                datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                errors.append(f"Invalid timestamp format: {message.get('timestamp')}")

        # Type-specific validations
        if msg_type == MessageType.INDICATOR_UPDATE:
            indicators = message.get("indicators", {})
            if not isinstance(indicators, dict):
                errors.append("'indicators' must be a dictionary")
            elif not indicators:
                warnings.append("'indicators' dictionary is empty")

        elif msg_type == MessageType.AI_DECISION:
            confidence = message.get("confidence")
            if confidence is not None:
                try:
                    conf_value = float(confidence)
                    if not 0 <= conf_value <= 1:
                        warnings.append(
                            f"Confidence {conf_value} outside expected range [0, 1]"
                        )
                except (ValueError, TypeError):
                    errors.append(f"Invalid confidence value: {confidence}")

        elif msg_type == MessageType.POSITION_UPDATE:
            # Validate P&L calculations
            if all(
                k in message for k in ["entry_price", "current_price", "size", "side"]
            ):
                try:
                    entry = float(message["entry_price"])
                    current = float(message["current_price"])
                    size = float(message["size"])
                    side = message["side"]

                    # Calculate expected P&L
                    if side == "long":
                        expected_pnl = (current - entry) * size
                    else:  # short
                        expected_pnl = (entry - current) * size

                    if "pnl" in message:
                        actual_pnl = float(message["pnl"])
                        if abs(actual_pnl - expected_pnl) > 0.01:
                            warnings.append(
                                f"P&L mismatch: expected {expected_pnl:.2f}, "
                                f"got {actual_pnl:.2f}"
                            )
                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid numeric values for P&L calculation: {e}")

        return MessageValidationResult(
            message_type=msg_type,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            data=message,
        )

    async def send_test_stimulus(self, stimulus_type: str) -> bool:
        """Send a test stimulus to trigger message generation."""
        if not self.websocket:
            await self.connect()

        try:
            # Send different stimuli based on type
            if stimulus_type == "market_update":
                message = {
                    "type": "test_stimulus",
                    "stimulus": "market_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"symbol": "BTC-USD", "price": 50000.0, "volume": 100.5},
                }
            elif stimulus_type == "trigger_decision":
                message = {
                    "type": "test_stimulus",
                    "stimulus": "trigger_decision",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                message = {
                    "type": "test_stimulus",
                    "stimulus": stimulus_type,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            await self.websocket.send(json.dumps(message))
            logger.info(f"Sent test stimulus: {stimulus_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to send stimulus: {e}")
            return False

    async def test_message_type(
        self, message_type: MessageType, timeout: float = 10.0
    ) -> bool:
        """Test a specific message type."""
        logger.info(f"\nTesting message type: {message_type}")

        # Clear previous messages
        self.received_messages.clear()

        # Start listening
        listen_task = asyncio.create_task(self.start_listening())

        # Wait a bit for listener to start
        await asyncio.sleep(1)

        # Send appropriate stimulus
        stimulus_map = {
            MessageType.MARKET_DATA: "market_update",
            MessageType.AI_DECISION: "trigger_decision",
            MessageType.INDICATOR_UPDATE: "calculate_indicators",
            MessageType.SYSTEM_STATUS: "check_status",
        }

        stimulus = stimulus_map.get(message_type, "generic")
        await self.send_test_stimulus(stimulus)

        # Wait for messages
        start_time = time.time()
        target_message = None

        while time.time() - start_time < timeout:
            for msg in self.received_messages:
                if msg.get("type") == message_type:
                    target_message = msg
                    break

            if target_message:
                break

            await asyncio.sleep(0.5)

        # Stop listening
        self.listening = False
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except TimeoutError:
            pass

        # Validate message if found
        if target_message:
            result = self.validate_message_schema(target_message)
            self.validation_results.append(result)

            if result.valid:
                logger.info(f"✓ {message_type}: Valid message received")
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(f"  ⚠ {warning}")
            else:
                logger.error(f"✗ {message_type}: Invalid message")
                for error in result.errors:
                    logger.error(f"  - {error}")

            return result.valid
        else:
            logger.warning(f"✗ {message_type}: No message received within {timeout}s")
            self.validation_results.append(
                MessageValidationResult(
                    message_type=message_type,
                    valid=False,
                    errors=[f"No message received within {timeout}s"],
                    warnings=[],
                )
            )
            return False

    async def test_message_ordering(self) -> bool:
        """Test that messages arrive in correct order."""
        logger.info("\nTesting message ordering...")

        self.received_messages.clear()

        # Start listening
        listen_task = asyncio.create_task(self.start_listening())
        await asyncio.sleep(1)

        # Send numbered messages
        sent_sequence = []
        for i in range(10):
            message = {
                "type": "sequence_test",
                "sequence": i,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.websocket.send(json.dumps(message))
            sent_sequence.append(i)
            await asyncio.sleep(0.1)  # Small delay between messages

        # Wait for all messages
        await asyncio.sleep(2)

        # Stop listening
        self.listening = False
        try:
            await asyncio.wait_for(listen_task, timeout=2.0)
        except TimeoutError:
            pass

        # Check ordering
        received_sequence = []
        for msg in self.received_messages:
            if msg.get("type") == "sequence_test":
                received_sequence.append(msg.get("sequence"))

        if received_sequence == sent_sequence:
            logger.info("✓ Message ordering: Messages arrived in correct order")
            return True
        else:
            logger.error("✗ Message ordering: Messages arrived out of order")
            logger.error(f"  Sent: {sent_sequence}")
            logger.error(f"  Received: {received_sequence}")
            return False

    async def test_all_message_types(self):
        """Test all message types."""
        logger.info("=" * 60)
        logger.info("Message Flow Validation Test")
        logger.info("=" * 60)

        # Connect to dashboard
        if not await self.connect():
            logger.error("Failed to connect to dashboard")
            sys.exit(1)

        # Test each message type
        critical_types = [
            MessageType.MARKET_DATA,
            MessageType.INDICATOR_UPDATE,
            MessageType.AI_DECISION,
            MessageType.TRADING_DECISION,
            MessageType.SYSTEM_STATUS,
        ]

        results = {}
        for msg_type in critical_types:
            results[msg_type] = await self.test_message_type(msg_type)
            await asyncio.sleep(1)  # Delay between tests

        # Test message ordering
        ordering_result = await self.test_message_ordering()

        # Disconnect
        await self.disconnect()

        # Print summary
        self.print_summary(results, ordering_result)

    def print_summary(
        self, type_results: dict[MessageType, bool], ordering_result: bool
    ):
        """Print test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)

        # Message type results
        passed = sum(1 for result in type_results.values() if result)
        total = len(type_results)

        logger.info("\nMessage Type Tests:")
        logger.info(f"  Total: {total}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {total - passed}")

        for msg_type, result in type_results.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {msg_type}")

        # Ordering test
        logger.info(
            f"\nMessage Ordering: {'✓ Passed' if ordering_result else '✗ Failed'}"
        )

        # Overall result
        all_passed = passed == total and ordering_result
        logger.info(
            f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}"
        )

        # Save detailed results
        results_file = "tests/docker/results/message_flow_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "validation_results": [
                        {
                            "message_type": r.message_type,
                            "valid": r.valid,
                            "errors": r.errors,
                            "warnings": r.warnings,
                        }
                        for r in self.validation_results
                    ],
                    "type_results": {k.value: v for k, v in type_results.items()},
                    "ordering_result": ordering_result,
                    "summary": {
                        "total_types": total,
                        "passed_types": passed,
                        "failed_types": total - passed,
                        "ordering_passed": ordering_result,
                        "all_passed": all_passed,
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"\nDetailed results saved to: {results_file}")

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)


async def main():
    """Main entry point."""
    import os

    # Get dashboard URL from environment
    dashboard_url = os.getenv(
        "SYSTEM__WEBSOCKET_DASHBOARD_URL", "ws://dashboard-backend:8000/ws"
    )

    validator = MessageFlowValidator(dashboard_url)
    await validator.test_all_message_types()


if __name__ == "__main__":
    asyncio.run(main())
