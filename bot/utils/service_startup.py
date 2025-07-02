"""
Service startup manager with graceful handling of optional dependencies.

This module provides a robust startup sequence that handles service dependencies,
connection failures, and graceful degradation when optional services are unavailable.
"""

import asyncio
from typing import Any

from bot.config import Settings
from bot.exchange.bluefin_service_client import (
    BluefinServiceClient,
    BluefinServiceUnavailable,
)
from bot.mcp.omnisearch_client import OmniSearchClient
from bot.utils.logger_factory import get_logger
from bot.utils.typed_config import get_typed

# WebSocket publisher removed

logger = get_logger(__name__)


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
            "bluefin_service": {
                "required": settings.exchange.exchange_type == "bluefin",
                "enabled": settings.exchange.exchange_type == "bluefin",
                "startup_delay": 3.0,
                "max_wait": 20.0,
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
    ) -> tuple[dict[str, Any], dict[str, ServiceStatus]]:
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
            startup_delay = get_typed(config, "startup_delay", 2.0)
            if startup_delay > 0:
                logger.info(
                    "Waiting %.1fs before starting %s...",
                    startup_delay,
                    service_name,
                )
                await asyncio.sleep(startup_delay)

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
    ) -> Any | None:
        """Start a specific service with timeout and error handling."""
        import time

        start_time = time.time()
        logger.info("Starting %s service...", service_name)

        try:
            if service_name == "bluefin_service":
                instance = await self._start_bluefin_service(max_wait)
            elif service_name == "omnisearch":
                instance = await self._start_omnisearch(max_wait)
            else:
                raise ValueError(f"Unknown service: {service_name}")

            status.available = True
            status.startup_time = time.time() - start_time
            logger.info(
                "✓ %s service started successfully (%.1fs)",
                service_name,
                float(status.startup_time),
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

    async def _start_omnisearch(self, timeout: float) -> OmniSearchClient | None:
        """Start OmniSearch client with timeout."""
        try:
            async with asyncio.timeout(timeout):
                client = OmniSearchClient(self.settings)

                # Test connection
                connected = await client.connect()
                if connected:
                    # Test search functionality with a simple query
                    test_result = await client.search("test connection", limit=1)
                    if test_result is not None:
                        logger.info("OmniSearch test query successful")
                        return client
                    logger.warning("OmniSearch test query returned no results")
                    # Return client anyway - it might work for other queries
                    return client
                logger.warning("OmniSearch connection failed")
                # Return unconnected client for graceful degradation
                return client

        except TimeoutError:
            logger.warning("OmniSearch connection timeout")
            # Return None for timeout - service is completely unreachable
            return None
        except Exception as e:
            logger.warning("OmniSearch connection error: %s", str(e))
            # For other errors, try to return a client for graceful degradation
            try:
                return OmniSearchClient(self.settings)
            except Exception:
                return None

    def _print_startup_summary(self):
        """Print summary of service startup results."""
        logger.info("\n%s", "=" * 60)
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
                logger.info(
                    "✓ %-20s: Available (%.1fs)", name, float(status.startup_time)
                )
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
            "Services available: %s/%s (%.0f%%)",
            available,
            total,
            (available / total * 100) if total > 0 else 0,
        )

        if required_failed > 0:
            logger.error(
                "ERROR: %s required service(s) failed to start!", required_failed
            )
        else:
            logger.info("All required services started successfully")

        logger.info("%s\n", "=" * 60)

    async def shutdown_all_services(self, service_instances: dict[str, Any]):
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
) -> tuple[dict[str, Any], dict[str, ServiceStatus]]:
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
                    "Service startup failed (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    str(e),
                )

                # Exponential backoff
                await asyncio.sleep(5 * (2**attempt))
                continue
            raise

    raise RuntimeError("Failed to start required services after all retries")
