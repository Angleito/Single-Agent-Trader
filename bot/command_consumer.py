"""
Command Consumer for Dashboard Integration

This module handles receiving and executing commands from the dashboard backend,
enabling bidirectional control of the trading bot. Commands are fetched via
HTTP polling and executed safely with proper validation and error handling.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp

from .config import settings

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Supported command types for bot control."""

    EMERGENCY_STOP = "emergency_stop"
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"
    UPDATE_RISK_LIMITS = "update_risk_limits"
    MANUAL_TRADE = "manual_trade"
    UNKNOWN = "unknown"


@dataclass
class BotCommand:
    """Represents a command received from the dashboard."""

    id: str
    command_type: str
    parameters: dict[str, Any]
    priority: int
    created_at: str
    status: str = "received"
    attempts: int = 0
    max_attempts: int = 3


class CommandConsumer:
    """
    Consumes commands from the dashboard backend and executes them safely.

    Features:
    - HTTP polling for command retrieval
    - Command validation and sanitization
    - Safe execution with error handling
    - Status reporting back to dashboard
    - Emergency stop capability
    - Trading pause/resume functionality
    """

    def __init__(self, dashboard_url: str | None = None, poll_interval: float = 2.0):
        """
        Initialize the command consumer.

        Args:
            dashboard_url: Base URL of the dashboard backend
            poll_interval: How often to check for new commands (seconds)
        """
        self.dashboard_url = (
            dashboard_url
            or settings.system.websocket_dashboard_url.replace(
                "ws://", "http://"
            ).replace("/ws", "")
        )
        self.poll_interval = poll_interval
        self.running = False
        self.session: aiohttp.ClientSession | None = None
        self._polling_task: asyncio.Task | None = None

        # Command execution state
        self.trading_paused = False
        self.emergency_stopped = False
        self.current_risk_limits: dict[str, Any] = {}

        # Execution callbacks - these will be set by the main bot
        self.callbacks = {
            "emergency_stop": None,
            "pause_trading": None,
            "resume_trading": None,
            "update_risk_limits": None,
            "manual_trade": None,
        }

        # Statistics
        self.stats: dict[str, Any] = {
            "commands_processed": 0,
            "commands_succeeded": 0,
            "commands_failed": 0,
            "last_poll_time": None,
            "last_command_time": None,
        }

    async def initialize(self):
        """Initialize HTTP session and connection."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(
                f"Command consumer initialized - Dashboard URL: {self.dashboard_url}"
            )

    async def close(self):
        """Close HTTP session and cleanup."""
        # Stop polling task if running
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error while cancelling polling task: {e}")

        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Command consumer closed")

    def register_callback(self, command_type: str, callback):
        """Register a callback function for a specific command type."""
        if command_type in self.callbacks:
            self.callbacks[command_type] = callback
            logger.info(f"Registered callback for command type: {command_type}")
        else:
            logger.warning(
                f"Unknown command type for callback registration: {command_type}"
            )

    async def start_polling(self):
        """Start the command polling loop."""
        if self.running:
            logger.warning("Command consumer already running")
            return

        self.running = True
        logger.info(f"Starting command polling every {self.poll_interval} seconds")

        while self.running:
            try:
                await self._poll_for_commands()
                self.stats["last_poll_time"] = datetime.now().isoformat()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                logger.info("Command polling cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in command polling loop: {e}")
                await asyncio.sleep(self.poll_interval * 2)  # Back off on errors

    async def stop_polling(self) -> None:
        """Stop the command polling loop."""
        self.running = False
        logger.info("Command polling stopped")

    async def start_polling_task(self) -> None:
        """Start the command polling as a managed background task."""
        if self._polling_task and not self._polling_task.done():
            logger.warning("Command polling task already running")
            return

        await self.initialize()
        self._polling_task = asyncio.create_task(self.start_polling())
        logger.info("Command polling task started")

    async def stop_polling_task(self) -> None:
        """Stop the command polling task and wait for completion."""
        if not self._polling_task:
            logger.debug("No polling task to stop")
            return

        # First, signal the polling loop to stop
        self.running = False

        # Cancel the task if it's still running
        if not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await asyncio.wait_for(self._polling_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Polling task did not stop within timeout")
            except asyncio.CancelledError:
                logger.debug("Polling task cancelled successfully")
            except Exception as e:
                logger.warning(f"Error while stopping polling task: {e}")

        self._polling_task = None
        logger.info("Command polling task stopped")

    async def _poll_for_commands(self):
        """Poll the dashboard for new commands."""
        if not self.session:
            await self.initialize()

        try:
            async with self.session.get(
                f"{self.dashboard_url}/api/bot/commands/queue"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    pending_commands = data.get("pending_commands", [])

                    if pending_commands:
                        logger.debug(f"Found {len(pending_commands)} pending commands")

                        # Process commands by priority (lower number = higher priority)
                        sorted_commands = sorted(
                            pending_commands, key=lambda x: x.get("priority", 5)
                        )

                        for cmd_data in sorted_commands:
                            await self._process_command(cmd_data)

                elif response.status == 404:
                    # Dashboard endpoint not available yet
                    logger.debug("Dashboard command endpoint not available")
                else:
                    logger.warning(f"Failed to fetch commands: HTTP {response.status}")

        except aiohttp.ClientError as e:
            logger.debug(f"Connection error polling for commands: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error polling for commands: {e}")

    async def _process_command(self, cmd_data: dict[str, Any]):
        """Process a single command."""
        try:
            command = BotCommand(
                id=cmd_data["id"],
                command_type=cmd_data["type"],
                parameters=cmd_data.get("parameters", {}),
                priority=cmd_data.get("priority", 5),
                created_at=cmd_data["created_at"],
                status=cmd_data.get("status", "pending"),
                attempts=cmd_data.get("attempts", 0),
            )

            logger.info(
                f"Processing command: {command.command_type} (ID: {command.id})"
            )

            # Validate command
            if not self._validate_command(command):
                await self._report_command_status(
                    command.id, "failed", "Command validation failed"
                )
                return

            # Execute command
            success, message = await self._execute_command(command)

            # Update statistics
            self.stats["commands_processed"] = (
                self.stats.get("commands_processed", 0) or 0
            ) + 1
            if success:
                self.stats["commands_succeeded"] = (
                    self.stats.get("commands_succeeded", 0) or 0
                ) + 1
                self.stats["last_command_time"] = datetime.now().isoformat()
            else:
                self.stats["commands_failed"] = (
                    self.stats.get("commands_failed", 0) or 0
                ) + 1

            # Report status back to dashboard
            status = "completed" if success else "failed"
            await self._report_command_status(command.id, status, message)

            # Remove command from queue if successful
            if success:
                await self._remove_command_from_queue(command.id)

        except Exception as e:
            logger.exception(
                f"Error processing command {cmd_data.get('id', 'unknown')}: {e}"
            )
            await self._report_command_status(
                cmd_data.get("id", "unknown"), "failed", str(e)
            )

    def _validate_command(self, command: BotCommand) -> bool:
        """Validate command before execution."""
        # Check if command type is supported
        try:
            CommandType(command.command_type)
        except ValueError:
            logger.exception(f"Unsupported command type: {command.command_type}")
            return False

        # Check if emergency stopped (only allow resume commands)
        if self.emergency_stopped and command.command_type not in ["resume_trading"]:
            logger.warning(
                f"Bot is emergency stopped, ignoring command: {command.command_type}"
            )
            return False

        # Validate command-specific parameters
        if command.command_type == "update_risk_limits":
            return self._validate_risk_limits(command.parameters)
        elif command.command_type == "manual_trade":
            return self._validate_manual_trade(command.parameters)

        return True

    def _validate_risk_limits(self, parameters: dict[str, Any]) -> bool:
        """Validate risk limit parameters."""
        valid_params = ["max_position_size", "stop_loss_percentage", "max_daily_loss"]

        for param, value in parameters.items():
            if param not in valid_params:
                logger.error(f"Invalid risk limit parameter: {param}")
                return False

            if not isinstance(value, int | float) or value <= 0:
                logger.error(f"Invalid value for {param}: {value}")
                return False

        return True

    def _validate_manual_trade(self, parameters: dict[str, Any]) -> bool:
        """Validate manual trade parameters."""
        required_params = ["action", "symbol", "size_percentage"]

        for param in required_params:
            if param not in parameters:
                logger.error(f"Missing required parameter for manual trade: {param}")
                return False

        # Validate action
        if parameters["action"] not in ["buy", "sell", "close"]:
            logger.error(f"Invalid trade action: {parameters['action']}")
            return False

        # Validate size percentage
        size_pct = parameters.get("size_percentage", 0)
        if not isinstance(size_pct, int | float) or size_pct <= 0 or size_pct > 100:
            logger.error(f"Invalid size percentage: {size_pct}")
            return False

        return True

    async def _execute_command(self, command: BotCommand) -> tuple[bool, str]:
        """Execute a validated command."""
        try:
            command_type = command.command_type
            callback = self.callbacks.get(command_type)

            if not callback:
                return False, f"No callback registered for command type: {command_type}"

            # Execute the command through registered callback
            if command_type == "emergency_stop":
                result = await self._execute_emergency_stop(callback)
            elif command_type == "pause_trading":
                result = await self._execute_pause_trading(callback)
            elif command_type == "resume_trading":
                result = await self._execute_resume_trading(callback)
            elif command_type == "update_risk_limits":
                result = await self._execute_update_risk_limits(
                    callback, command.parameters
                )
            elif command_type == "manual_trade":
                result = await self._execute_manual_trade(callback, command.parameters)
            else:
                return False, f"Unsupported command type: {command_type}"

            return result

        except Exception as e:
            logger.exception(f"Error executing command {command.id}: {e}")
            return False, str(e)

    async def _execute_emergency_stop(self, callback) -> tuple[bool, str]:
        """Execute emergency stop command."""
        try:
            if callable(callback):
                await callback()
            self.emergency_stopped = True
            self.trading_paused = True
            logger.critical("EMERGENCY STOP EXECUTED")
            return True, "Emergency stop executed successfully"
        except Exception as e:
            return False, f"Emergency stop failed: {e}"

    async def _execute_pause_trading(self, callback) -> tuple[bool, str]:
        """Execute pause trading command."""
        try:
            if self.trading_paused:
                return True, "Trading is already paused"

            if callable(callback):
                await callback()
            self.trading_paused = True
            logger.info("Trading paused by dashboard command")
            return True, "Trading paused successfully"
        except Exception as e:
            return False, f"Pause trading failed: {e}"

    async def _execute_resume_trading(self, callback) -> tuple[bool, str]:
        """Execute resume trading command."""
        try:
            if not self.trading_paused and not self.emergency_stopped:
                return True, "Trading is already active"

            if callable(callback):
                await callback()
            self.trading_paused = False
            self.emergency_stopped = False
            logger.info("Trading resumed by dashboard command")
            return True, "Trading resumed successfully"
        except Exception as e:
            return False, f"Resume trading failed: {e}"

    async def _execute_update_risk_limits(
        self, callback, parameters: dict[str, Any]
    ) -> tuple[bool, str]:
        """Execute update risk limits command."""
        try:
            if callable(callback):
                await callback(parameters)

            self.current_risk_limits.update(parameters)
            logger.info(f"Risk limits updated: {parameters}")
            return True, f"Risk limits updated: {parameters}"
        except Exception as e:
            return False, f"Update risk limits failed: {e}"

    async def _execute_manual_trade(
        self, callback, parameters: dict[str, Any]
    ) -> tuple[bool, str]:
        """Execute manual trade command."""
        try:
            if self.emergency_stopped:
                return False, "Cannot execute manual trade: emergency stop active"

            if callable(callback):
                result = await callback(parameters)
                if result:
                    logger.info(f"Manual trade executed: {parameters}")
                    return (
                        True,
                        f"Manual trade executed: {parameters['action']} {parameters['size_percentage']}% {parameters['symbol']}",
                    )
                else:
                    return False, "Manual trade execution returned false"
            else:
                return False, "No manual trade callback registered"
        except Exception as e:
            return False, f"Manual trade failed: {e}"

    async def _report_command_status(self, command_id: str, status: str, message: str):
        """Report command execution status back to dashboard."""
        if not self.session:
            return

        try:
            data = {
                "command_id": command_id,
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "bot_id": "ai-trading-bot",
            }

            async with self.session.post(
                f"{self.dashboard_url}/api/bot/commands/status", json=data
            ) as response:
                if response.status == 200:
                    logger.debug(f"Reported command status: {command_id} -> {status}")
                else:
                    logger.warning(
                        f"Failed to report command status: HTTP {response.status}"
                    )

        except Exception as e:
            logger.debug(f"Error reporting command status: {e}")

    async def _remove_command_from_queue(self, command_id: str):
        """Remove completed command from dashboard queue."""
        if not self.session:
            return

        try:
            async with self.session.delete(
                f"{self.dashboard_url}/api/bot/commands/{command_id}"
            ) as response:
                if response.status == 200:
                    logger.debug(f"Removed completed command from queue: {command_id}")
                else:
                    logger.debug(
                        f"Could not remove command from queue: HTTP {response.status}"
                    )

        except Exception as e:
            logger.debug(f"Error removing command from queue: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current status of the command consumer."""
        return {
            "running": self.running,
            "dashboard_url": self.dashboard_url,
            "poll_interval": self.poll_interval,
            "trading_paused": self.trading_paused,
            "emergency_stopped": self.emergency_stopped,
            "current_risk_limits": self.current_risk_limits,
            "stats": self.stats,
            "callbacks_registered": {
                cmd_type: callback is not None
                for cmd_type, callback in self.callbacks.items()
            },
            "polling_task_running": self._polling_task is not None
            and not self._polling_task.done(),
        }
