"""
Integration example showing how service types work with existing service startup.

This demonstrates how the new type system integrates with the existing
ServiceStartupManager and ServiceDiscovery classes.
"""

from typing import Any

from bot.types import (
    ConnectionInfo,
    ConnectionState,
    ServiceEndpoint,
    ServiceHealth,
    ServiceStatus,
    create_endpoint,
    create_health_status,
    is_docker_service,
    is_healthy_service,
    is_valid_endpoint,
)


def convert_service_status_to_health(service_status: Any) -> ServiceHealth:
    """
    Convert existing ServiceStatus to new ServiceHealth type.

    Args:
        service_status: Existing service status object

    Returns:
        ServiceHealth object
    """
    status_map = {
        True: ServiceStatus.HEALTHY,
        False: ServiceStatus.UNHEALTHY,
        None: ServiceStatus.UNKNOWN,
    }

    health_status = status_map.get(
        getattr(service_status, "available", None), ServiceStatus.UNKNOWN
    )

    return create_health_status(
        status=health_status,
        error=getattr(service_status, "error", None),
        response_time_ms=getattr(service_status, "startup_time", 0) * 1000,
        service_name=getattr(service_status, "name", "unknown"),
    )


def convert_service_endpoint_to_typed(service_endpoint: Any) -> ServiceEndpoint:
    """
    Convert existing ServiceEndpoint dataclass to typed ServiceEndpoint.

    Args:
        service_endpoint: Existing service endpoint object

    Returns:
        ServiceEndpoint TypedDict
    """
    # Map protocol string to literal type
    protocol_map = {
        "http": "http",
        "https": "https",
        "ws": "ws",
        "wss": "wss",
        "tcp": "tcp",
        "grpc": "grpc",
    }

    protocol = protocol_map.get(getattr(service_endpoint, "protocol", "http"), "http")

    endpoint = create_endpoint(
        host=getattr(service_endpoint, "host", "localhost"),
        port=getattr(service_endpoint, "port", 8080),
        protocol=protocol,  # type: ignore
        health_endpoint=getattr(service_endpoint, "health_endpoint", None),
    )

    # Add optional fields if present
    if hasattr(service_endpoint, "metadata") and service_endpoint.metadata:
        endpoint["auth"] = service_endpoint.metadata

    return endpoint


class TypedServiceWrapper:
    """
    Wrapper to make existing services conform to DockerService protocol.
    """

    def __init__(self, name: str, service_instance: Any, endpoint: ServiceEndpoint):
        """Initialize wrapper."""
        self.name = name
        self.service_instance = service_instance
        self.endpoint = endpoint
        self.required = True

    def health_check(self) -> ServiceHealth:
        """Perform synchronous health check."""
        # Check for various health check method names
        if hasattr(self.service_instance, "check_health"):
            is_healthy = self.service_instance.check_health()
        elif hasattr(self.service_instance, "is_healthy"):
            is_healthy = self.service_instance.is_healthy()
        else:
            is_healthy = True  # Assume healthy if no health check method

        return create_health_status(
            status=ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
        )

    async def async_health_check(self) -> ServiceHealth:
        """Perform asynchronous health check."""
        if hasattr(self.service_instance, "check_health"):
            if asyncio.iscoroutinefunction(self.service_instance.check_health):
                is_healthy = await self.service_instance.check_health()
            else:
                is_healthy = self.service_instance.check_health()
        else:
            is_healthy = True

        return create_health_status(
            status=ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
        )

    def is_ready(self) -> bool:
        """Check if service is ready."""
        if hasattr(self.service_instance, "is_ready"):
            return self.service_instance.is_ready()
        return True

    async def initialize(self) -> bool:
        """Initialize service."""
        if hasattr(self.service_instance, "initialize"):
            return await self.service_instance.initialize()
        return True

    async def shutdown(self) -> None:
        """Shutdown service."""
        if hasattr(self.service_instance, "close"):
            await self.service_instance.close()
        elif hasattr(self.service_instance, "disconnect"):
            await self.service_instance.disconnect()
        elif hasattr(self.service_instance, "shutdown"):
            await self.service_instance.shutdown()


def validate_service_startup_config(config: dict[str, Any]) -> None:
    """
    Validate service configuration from ServiceStartupManager.

    Args:
        config: Service configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    for service_name, service_config in config.items():
        # Build typed config
        typed_config = {
            "name": service_name,
            "enabled": service_config.get("enabled", False),
            "required": service_config.get("required", False),
            "endpoint": create_endpoint(
                host="localhost",  # Default, would be overridden
                port=8080,  # Default, would be overridden
                protocol="http",
            ),
        }

        # Add optional fields
        if "startup_delay" in service_config:
            typed_config["startup_delay"] = service_config["startup_delay"]
        if "max_wait" in service_config:
            typed_config["max_wait"] = service_config["max_wait"]

        # This will raise if invalid
        from bot.types import validate_service_config

        validate_service_config(typed_config)


# Example of using type guards for runtime checks
def check_service_health_with_guards(service: Any) -> bool:
    """
    Check service health using type guards.

    Args:
        service: Service instance to check

    Returns:
        True if service is healthy
    """
    # First check if it's a valid DockerService
    if not is_docker_service(service):
        print(f"Warning: {service} does not implement DockerService protocol")
        return False

    # Check endpoint validity
    if not is_valid_endpoint(service.endpoint):
        print(f"Warning: Invalid endpoint for service {service.name}")
        return False

    # Check health
    health = service.health_check()
    return is_healthy_service(health)


# Connection tracking example
def create_connection_info(state: ConnectionState) -> ConnectionInfo:
    """
    Create connection info for service tracking.

    Args:
        state: Current connection state

    Returns:
        ConnectionInfo object
    """
    import datetime

    info: ConnectionInfo = {"state": state}

    if state == ConnectionState.CONNECTED:
        info["connected_at"] = datetime.datetime.now()
    elif state == ConnectionState.DISCONNECTED:
        info["disconnected_at"] = datetime.datetime.now()

    return info


import asyncio

if __name__ == "__main__":
    # Example usage
    print("Service Types Integration Example")
    print("=" * 50)

    # Create a typed endpoint
    endpoint = create_endpoint(
        host="bluefin-service", port=8080, protocol="http", health_endpoint="/health"
    )

    print(f"Created endpoint: {endpoint}")
    print(f"Endpoint valid: {is_valid_endpoint(endpoint)}")

    # Create connection info
    conn_info = create_connection_info(ConnectionState.CONNECTED)
    print(f"\nConnection info: {conn_info}")

    print("\n✅ Integration example completed!")
