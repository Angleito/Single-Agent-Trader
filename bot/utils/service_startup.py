"""
Service startup manager with graceful handling of optional dependencies.

This module provides a robust startup sequence that handles service dependencies,
connection failures, and graceful degradation when optional services are unavailable.
"""

import asyncio
import logging

from bot.config import Settings
from bot.exchange.bluefin_service_client import (
    BluefinServiceClient,
    BluefinServiceUnavailable,
)
from bot.mcp.omnisearch_client import OmniSearchClient
from bot.websocket_publisher import WebSocketPublisher

logger = logging.getLogger(__name__)


class ServiceStatus:
    """Status of a service during startup."""

    def __init__(self, name: str, required: bool = False):
        """Initialize service status."""
        self.name = name
        self.required = required
        self.available = False
        self.error: str | None = None
        self.startup_time: float = 0.0


class ServiceStartupManager:
    """Manages startup sequence for all services with proper dependency handling."""

    def __init__(self, settings: Settings):
        """Initialize startup manager."""
        self.settings = settings
        self.services_status: dict[str, ServiceStatus] = {}
        self.startup_delay = 2.0  # Default delay between service starts

        # Define service dependencies and requirements
        self.services_config = {
            "websocket_publisher": {
                "required": False,
                "enabled": getattr(
                    settings.system, "enable_websocket_publishing", True
                ),
                "startup_delay": 5.0,  # Extra delay for dashboard to be ready
                "max_wait": 30.0,
            },
            "bluefin_service": {
                "required": settings.exchange.exchange_type == "bluefin",
                "enabled": settings.exchange.exchange_type == "bluefin",
                "startup_delay": 3.0,
                "max_wait": 20.0,
            },
            "mcp_memory": {
                "required": False,
                "enabled": getattr(settings.system, "mcp_enabled", False),
                "startup_delay": 2.0,
                "max_wait": 15.0,
            },
            "omnisearch": {
                "required": False,
                "enabled": getattr(settings.omnisearch, "enabled", False),
                "startup_delay": 2.0,
                "max_wait": 15.0,
            },
        }

    async def startup_all_services(
        self,
    ) -> tuple[dict[str, any], dict[str, ServiceStatus]]:
        """
        Start all configured services in the correct order.

        Returns:
            Tuple of (service_instances, service_statuses)
        """
        logger.info("=" * 60)
        logger.info("SERVICE STARTUP SEQUENCE INITIATED")
        logger.info("=" * 60)

        service_instances = {}

        # Start services in order
        for service_name, config in self.services_config.items():
            if not config["enabled"]:
                logger.info("Service %s is disabled in configuration", service_name)
                continue

            status = ServiceStatus(service_name, config["required"])
            self.services_status[service_name] = status

            # Apply startup delay
            if config["startup_delay"] > 0:
                logger.info(
                    "Waiting %.1fs before starting %s...",
                    config["startup_delay"],
                    service_name,
                )
                await asyncio.sleep(config["startup_delay"])

            # Start the service
            instance = await self._start_service(
                service_name, status, config["max_wait"]
            )

            if instance:
                service_instances[service_name] = instance
            elif status.required:
                logger.error(
                    "Required service %s failed to start: %s",
                    service_name,
                    status.error,
                )
                raise RuntimeError(f"Required service {service_name} failed to start")

        # Print startup summary
        self._print_startup_summary()

        return service_instances, self.services_status

    async def _start_service(
        self, service_name: str, status: ServiceStatus, max_wait: float
    ) -> any | None:
        """Start a specific service with timeout and error handling."""
        import time

        start_time = time.time()
        logger.info("Starting %s service...", service_name)

        try:
            if service_name == "websocket_publisher":
                instance = await self._start_websocket_publisher(max_wait)
            elif service_name == "bluefin_service":
                instance = await self._start_bluefin_service(max_wait)
            elif service_name == "mcp_memory":
                instance = await self._start_mcp_memory(max_wait)
            elif service_name == "omnisearch":
                instance = await self._start_omnisearch(max_wait)
            else:
                raise ValueError(f"Unknown service: {service_name}")

            status.available = True
            status.startup_time = time.time() - start_time
            logger.info(
                "✓ %s service started successfully (%.1fs)",
                service_name,
                status.startup_time,
            )
            return instance

        except TimeoutError:
            status.error = f"Timeout after {max_wait}s"
            logger.warning(
                "✗ %s service startup timeout after %.1fs", service_name, max_wait
            )
            return None

        except Exception as e:
            status.error = str(e)
            logger.warning(
                "✗ %s service startup failed: %s", service_name, status.error
            )
            return None

    async def _start_websocket_publisher(
        self, timeout: float
    ) -> WebSocketPublisher | None:
        """Start WebSocket publisher with timeout."""
        try:
            publisher = WebSocketPublisher(self.settings)

            # Initialize with timeout
            async with asyncio.timeout(timeout):
                success = await publisher.initialize()

                if success:
                    return publisher
                logger.warning("WebSocket publisher initialization returned False")
                return None

        except TimeoutError:
            logger.warning("WebSocket publisher initialization timeout")
            raise
        except Exception as e:
            logger.warning("WebSocket publisher initialization error: %s", str(e))
            raise

    async def _start_bluefin_service(
        self, timeout: float
    ) -> BluefinServiceClient | None:
        """Start Bluefin service client with timeout."""
        try:
            from bot.exchange.bluefin_service_client import get_bluefin_service_client

            async with asyncio.timeout(timeout):
                client = await get_bluefin_service_client(self.settings)

                # Verify service is healthy
                if await client.check_health():
                    return client
                raise BluefinServiceUnavailable("Bluefin service health check failed")

        except TimeoutError:
            logger.warning("Bluefin service connection timeout")
            raise
        except Exception as e:
            logger.warning("Bluefin service connection error: %s", str(e))
            raise

    async def _start_mcp_memory(self, timeout: float) -> any | None:
        """Start MCP memory service with timeout."""
        try:
            # MCP memory is handled by the memory-enhanced agent
            # Just check if the service is reachable
            import aiohttp

            mcp_url = getattr(
                self.settings.system, "mcp_server_url", "http://mcp-memory:8765"
            )

            async with asyncio.timeout(timeout):
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{mcp_url}/health") as response:
                        if response.status == 200:
                            logger.info("MCP memory service is healthy")
                            return True
                        raise Exception(
                            f"MCP memory service unhealthy: status={response.status}"
                        )

        except TimeoutError:
            logger.warning("MCP memory service connection timeout")
            raise
        except Exception as e:
            logger.warning("MCP memory service connection error: %s", str(e))
            raise

    async def _start_omnisearch(self, timeout: float) -> OmniSearchClient | None:
        """Start OmniSearch client with timeout."""
        try:
            async with asyncio.timeout(timeout):
                client = OmniSearchClient(self.settings)

                # Test connection
                test_result = await client.search("test connection", limit=1)
                if test_result is not None:
                    return client
                raise Exception("OmniSearch test query failed")

        except TimeoutError:
            logger.warning("OmniSearch connection timeout")
            raise
        except Exception as e:
            logger.warning("OmniSearch connection error: %s", str(e))
            raise

    def _print_startup_summary(self):
        """Print summary of service startup results."""
        logger.info("\n" + "=" * 60)
        logger.info("SERVICE STARTUP SUMMARY")
        logger.info("=" * 60)

        # Count statuses
        total = len(self.services_status)
        available = sum(1 for s in self.services_status.values() if s.available)
        required_failed = sum(
            1 for s in self.services_status.values() if s.required and not s.available
        )

        # Print each service status
        for name, status in self.services_status.items():
            if status.available:
                logger.info("✓ %-20s: Available (%.1fs)", name, status.startup_time)
            else:
                req_marker = " [REQUIRED]" if status.required else ""
                logger.warning(
                    "✗ %-20s: Unavailable%s - %s",
                    name,
                    req_marker,
                    status.error or "Unknown error",
                )

        logger.info("-" * 60)
        logger.info(
            "Services available: %d/%d (%.0f%%)",
            available,
            total,
            (available / total * 100) if total > 0 else 0,
        )

        if required_failed > 0:
            logger.error(
                "ERROR: %d required service(s) failed to start!", required_failed
            )
        else:
            logger.info("All required services started successfully")

        logger.info("=" * 60 + "\n")

    async def shutdown_all_services(self, service_instances: dict[str, any]):
        """Shutdown all services gracefully."""
        logger.info("Shutting down services...")

        for name, instance in service_instances.items():
            try:
                if hasattr(instance, "close"):
                    await instance.close()
                elif hasattr(instance, "disconnect"):
                    await instance.disconnect()
                elif hasattr(instance, "shutdown"):
                    await instance.shutdown()

                logger.info("✓ %s service shut down", name)

            except Exception as e:
                logger.warning("Error shutting down %s: %s", name, str(e))

        # Special handling for singleton services
        try:
            from bot.exchange.bluefin_service_client import close_bluefin_service_client

            await close_bluefin_service_client()
        except Exception:
            pass


async def startup_services_with_retry(
    settings: Settings, max_retries: int = 3
) -> tuple[dict[str, any], dict[str, ServiceStatus]]:
    """
    Start services with retry logic for transient failures.

    Args:
        settings: Application settings
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (service_instances, service_statuses)
    """
    manager = ServiceStartupManager(settings)

    for attempt in range(max_retries):
        try:
            return await manager.startup_all_services()

        except RuntimeError as e:
            if "Required service" in str(e) and attempt < max_retries - 1:
                logger.warning(
                    "Service startup failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    str(e),
                )

                # Exponential backoff
                await asyncio.sleep(5 * (2**attempt))
                continue
            raise

    raise RuntimeError("Failed to start required services after all retries")
