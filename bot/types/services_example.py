"""
Example usage of service types.

This file demonstrates how to use the comprehensive service type system.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional

from bot.types import (
    DockerService,
    ServiceEndpoint,
    ServiceHealth,
    ServiceManager,
    ServiceStatus,
    create_endpoint,
    create_health_status,
    is_healthy_service,
    is_valid_endpoint,
)


class BluefinService:
    """Example implementation of DockerService protocol."""

    def __init__(self, name: str, endpoint: ServiceEndpoint, required: bool = True):
        """Initialize Bluefin service."""
        self.name = name
        self.endpoint = endpoint
        self.required = required
        self._is_initialized = False

    def health_check(self) -> ServiceHealth:
        """Perform synchronous health check."""
        # In real implementation, this would check the actual service
        if self._is_initialized:
            return create_health_status(
                status=ServiceStatus.HEALTHY,
                response_time_ms=15.2,
                version="1.0.0",
                connections=42,
            )
        else:
            return create_health_status(
                status=ServiceStatus.UNHEALTHY,
                error="Service not initialized",
            )

    async def async_health_check(self) -> ServiceHealth:
        """Perform asynchronous health check."""
        # Simulate async health check
        await asyncio.sleep(0.1)
        return self.health_check()

    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._is_initialized

    async def initialize(self) -> bool:
        """Initialize service connection."""
        print(f"Initializing {self.name} service...")
        await asyncio.sleep(0.5)  # Simulate initialization
        self._is_initialized = True
        print(f"✓ {self.name} service initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown service cleanly."""
        print(f"Shutting down {self.name} service...")
        self._is_initialized = False
        await asyncio.sleep(0.1)  # Simulate cleanup
        print(f"✓ {self.name} service shut down")


class SimpleServiceManager:
    """Example implementation of ServiceManager protocol."""

    def __init__(self):
        """Initialize service manager."""
        self._services: Dict[str, DockerService] = {}

    def register_service(self, service: DockerService) -> None:
        """Register a service."""
        if not is_valid_endpoint(service.endpoint):
            raise ValueError(f"Invalid endpoint for service {service.name}")
        self._services[service.name] = service
        print(f"Registered service: {service.name}")

    def unregister_service(self, service_name: str) -> None:
        """Unregister a service."""
        if service_name in self._services:
            del self._services[service_name]
            print(f"Unregistered service: {service_name}")

    def get_service(self, service_name: str) -> Optional[DockerService]:
        """Get a registered service."""
        return self._services.get(service_name)

    def get_healthy_services(self) -> list[DockerService]:
        """Get all healthy services."""
        healthy = []
        for service in self._services.values():
            health = service.health_check()
            if is_healthy_service(health):
                healthy.append(service)
        return healthy

    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """Check health of all services."""
        results = {}
        for name, service in self._services.items():
            results[name] = await service.async_health_check()
        return results


async def main():
    """Demonstrate service type usage."""
    print("Service Types Example")
    print("=" * 50)

    # Create service endpoints
    bluefin_endpoint = create_endpoint(
        host="bluefin-service",
        port=8080,
        protocol="http",
        health_endpoint="/health",
        timeout_seconds=10.0,
    )

    websocket_endpoint = create_endpoint(
        host="localhost",
        port=8765,
        protocol="ws",
        health_endpoint=None,  # WebSocket doesn't use HTTP health endpoint
        base_path="/ws",
    )

    # Validate endpoints
    print(f"\nValidating endpoints:")
    print(f"  Bluefin endpoint valid: {is_valid_endpoint(bluefin_endpoint)}")
    print(f"  WebSocket endpoint valid: {is_valid_endpoint(websocket_endpoint)}")

    # Create services
    bluefin_service = BluefinService(
        name="bluefin-sdk",
        endpoint=bluefin_endpoint,
        required=True,
    )

    websocket_service = BluefinService(
        name="websocket-publisher",
        endpoint=websocket_endpoint,
        required=False,
    )

    # Create service manager
    manager = SimpleServiceManager()

    # Register services
    print(f"\nRegistering services:")
    manager.register_service(bluefin_service)
    manager.register_service(websocket_service)

    # Check initial health
    print(f"\nInitial health check:")
    initial_health = await manager.health_check_all()
    for name, health in initial_health.items():
        status = health["status"]
        print(f"  {name}: {status}")

    # Initialize services
    print(f"\nInitializing services:")
    for service_name in ["bluefin-sdk", "websocket-publisher"]:
        service = manager.get_service(service_name)
        if service:
            await service.initialize()

    # Check health after initialization
    print(f"\nHealth check after initialization:")
    final_health = await manager.health_check_all()
    for name, health in final_health.items():
        status = health["status"]
        response_time = health.get("response_time_ms", "N/A")
        print(f"  {name}: {status} (response time: {response_time}ms)")

    # Get healthy services
    healthy_services = manager.get_healthy_services()
    print(f"\nHealthy services: {len(healthy_services)}")
    for service in healthy_services:
        print(f"  - {service.name}")

    # Shutdown services
    print(f"\nShutting down services:")
    for service_name in ["bluefin-sdk", "websocket-publisher"]:
        service = manager.get_service(service_name)
        if service:
            await service.shutdown()

    print(f"\n✅ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())