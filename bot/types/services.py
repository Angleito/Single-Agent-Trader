"""
Service type definitions for Docker and external service management.

This module provides comprehensive type definitions for service management,
including health checks, endpoints, protocols, and runtime validation.

## Design Considerations (Ultrathink Analysis):

### 1. Optional Services Handling
- Services have a `required` field to distinguish between critical and optional services
- ServiceConfig includes `enabled` to allow runtime service toggling
- Health checks can return DEGRADED status for partial functionality
- ServiceManager protocol allows querying only healthy services

### 2. Runtime Validation
- Type guards (is_valid_endpoint, is_healthy_service) for untrusted data validation
- validate_service_config() provides comprehensive configuration validation
- All type guards return TypeGuard[T] for proper type narrowing
- Validation errors include detailed context for debugging

### 3. Async Service Methods
- Protocol includes both sync (health_check) and async (async_health_check) methods
- AsyncHealthCheck type alias for async health check callbacks
- ServiceCallback supports both sync and async callbacks
- Proper async/await typing throughout the protocols

### 4. Error Propagation
- Hierarchical error types (ServiceError base with specific subclasses)
- Errors include service_name and error_code for better tracking
- ServiceHealth includes error field for health check failures
- Connection errors separate from health check errors for clarity

Example usage:
    ```python
    # Create a service endpoint
    endpoint = create_endpoint(
        host="localhost",
        port=8080,
        protocol="http",
        health_endpoint="/health"
    )
    
    # Validate endpoint at runtime
    if is_valid_endpoint(endpoint):
        # Create service with validated endpoint
        service = MyDockerService("my-service", endpoint, required=True)
    
    # Check service health
    health = await service.async_health_check()
    if is_healthy_service(health):
        print("Service is healthy!")
    ```
"""

from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeGuard,
    Union,
    runtime_checkable,
)

from typing_extensions import NotRequired


# Service Status Types
class ServiceStatus(Enum):
    """Service status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"
    DEGRADED = "degraded"


class ServiceHealth(TypedDict):
    """Type-safe service health status."""

    status: Literal["healthy", "unhealthy", "unknown", "starting", "stopping", "degraded"]
    last_check: float  # Unix timestamp
    error: NotRequired[Optional[str]]
    response_time_ms: NotRequired[Optional[float]]
    consecutive_failures: NotRequired[int]
    metadata: NotRequired[Dict[str, Any]]


class ServiceEndpoint(TypedDict):
    """Type-safe service endpoint configuration."""

    host: str
    port: int
    protocol: Literal["http", "https", "ws", "wss", "tcp", "grpc"]
    health_endpoint: Optional[str]
    base_path: NotRequired[str]
    timeout_seconds: NotRequired[float]
    headers: NotRequired[Dict[str, str]]
    auth: NotRequired[Dict[str, Any]]


# Connection State Types
class ConnectionState(Enum):
    """Connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class ConnectionInfo(TypedDict):
    """Connection information."""

    state: ConnectionState
    connected_at: NotRequired[Optional[datetime]]
    disconnected_at: NotRequired[Optional[datetime]]
    reconnect_attempts: NotRequired[int]
    last_error: NotRequired[Optional[str]]
    latency_ms: NotRequired[Optional[float]]


# Service Configuration Types
class ServiceConfig(TypedDict):
    """Service configuration."""

    name: str
    enabled: bool
    required: bool
    endpoint: ServiceEndpoint
    startup_delay: NotRequired[float]
    max_wait: NotRequired[float]
    retry_config: NotRequired["RetryConfig"]
    dependencies: NotRequired[List[str]]
    environment: NotRequired[Dict[str, str]]


class RetryConfig(TypedDict):
    """Retry configuration for service connections."""

    max_attempts: int
    initial_delay: float
    max_delay: float
    exponential_base: NotRequired[float]
    jitter: NotRequired[bool]


# Service Registry Types
class ServiceRegistration(TypedDict):
    """Service registration information."""

    service_id: str
    service_type: str
    endpoint: ServiceEndpoint
    health: ServiceHealth
    registered_at: datetime
    updated_at: datetime
    tags: NotRequired[List[str]]
    metadata: NotRequired[Dict[str, Any]]


class ServiceRegistry(TypedDict):
    """Service registry state."""

    services: Dict[str, ServiceRegistration]
    discovery_enabled: bool
    last_discovery: Optional[datetime]
    discovery_interval_seconds: float


# Protocol Definitions
@runtime_checkable
class DockerService(Protocol):
    """Protocol for Docker service requirements."""

    name: str
    endpoint: ServiceEndpoint
    required: bool

    def health_check(self) -> ServiceHealth:
        """Perform synchronous health check."""
        ...

    async def async_health_check(self) -> ServiceHealth:
        """Perform asynchronous health check."""
        ...

    def is_ready(self) -> bool:
        """Check if service is ready to accept requests."""
        ...

    async def initialize(self) -> bool:
        """Initialize service connection."""
        ...

    async def shutdown(self) -> None:
        """Shutdown service cleanly."""
        ...


@runtime_checkable
class ServiceManager(Protocol):
    """Protocol for service management."""

    def register_service(self, service: DockerService) -> None:
        """Register a service."""
        ...

    def unregister_service(self, service_name: str) -> None:
        """Unregister a service."""
        ...

    def get_service(self, service_name: str) -> Optional[DockerService]:
        """Get a registered service."""
        ...

    def get_healthy_services(self) -> List[DockerService]:
        """Get all healthy services."""
        ...

    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """Check health of all services."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for health checkable services."""

    async def check_health(self) -> bool:
        """Check if service is healthy."""
        ...

    def get_health_status(self) -> ServiceHealth:
        """Get detailed health status."""
        ...


# Error Types
class ServiceError(Exception):
    """Base exception for service-related errors."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        error_code: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message)
        self.service_name = service_name
        self.error_code = error_code
        self.details = kwargs


class ServiceConnectionError(ServiceError):
    """Service connection failed."""

    pass


class ServiceHealthCheckError(ServiceError):
    """Service health check failed."""

    pass


class ServiceTimeoutError(ServiceError):
    """Service operation timed out."""

    pass


class ServiceNotFoundError(ServiceError):
    """Service not found in registry."""

    pass


class ServiceStartupError(ServiceError):
    """Service failed to start."""

    pass


class ServiceDependencyError(ServiceError):
    """Service dependency not met."""

    pass


# Type Guards
def is_valid_endpoint(value: Any) -> TypeGuard[ServiceEndpoint]:
    """
    Check if value is a valid ServiceEndpoint.

    Args:
        value: Value to check

    Returns:
        True if value is a valid ServiceEndpoint
    """
    if not isinstance(value, dict):
        return False

    required_keys = {"host", "port", "protocol", "health_endpoint"}
    if not all(key in value for key in required_keys):
        return False

    # Validate types
    if not isinstance(value["host"], str) or not value["host"]:
        return False

    if not isinstance(value["port"], int) or value["port"] < 1 or value["port"] > 65535:
        return False

    valid_protocols = {"http", "https", "ws", "wss", "tcp", "grpc"}
    if value["protocol"] not in valid_protocols:
        return False

    if value["health_endpoint"] is not None and not isinstance(value["health_endpoint"], str):
        return False

    # Validate optional fields if present
    if "timeout_seconds" in value and (
        not isinstance(value["timeout_seconds"], (int, float)) or value["timeout_seconds"] <= 0
    ):
        return False

    if "headers" in value and not isinstance(value["headers"], dict):
        return False

    return True


def is_healthy_service(health: Any) -> TypeGuard[ServiceHealth]:
    """
    Check if service health indicates a healthy service.

    Args:
        health: Health status to check

    Returns:
        True if service is healthy
    """
    if not isinstance(health, dict):
        return False

    if "status" not in health:
        return False

    return health["status"] == "healthy"


def is_docker_service(obj: Any) -> TypeGuard[DockerService]:
    """
    Check if object implements DockerService protocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements DockerService
    """
    return (
        hasattr(obj, "name")
        and hasattr(obj, "endpoint")
        and hasattr(obj, "required")
        and hasattr(obj, "health_check")
        and hasattr(obj, "is_ready")
        and callable(getattr(obj, "health_check", None))
        and callable(getattr(obj, "is_ready", None))
    )


def validate_service_config(config: Any) -> ServiceConfig:
    """
    Validate and return service configuration.

    Args:
        config: Configuration to validate

    Returns:
        Validated ServiceConfig

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Service config must be a dictionary")

    required_keys = {"name", "enabled", "required", "endpoint"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Validate name
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("Service name must be a non-empty string")

    # Validate booleans
    if not isinstance(config["enabled"], bool):
        raise ValueError("Service enabled must be a boolean")

    if not isinstance(config["required"], bool):
        raise ValueError("Service required must be a boolean")

    # Validate endpoint
    if not is_valid_endpoint(config["endpoint"]):
        raise ValueError("Invalid service endpoint configuration")

    # Validate optional fields
    if "startup_delay" in config:
        if not isinstance(config["startup_delay"], (int, float)) or config["startup_delay"] < 0:
            raise ValueError("Startup delay must be a non-negative number")

    if "max_wait" in config:
        if not isinstance(config["max_wait"], (int, float)) or config["max_wait"] <= 0:
            raise ValueError("Max wait must be a positive number")

    if "dependencies" in config:
        if not isinstance(config["dependencies"], list):
            raise ValueError("Dependencies must be a list")
        if not all(isinstance(dep, str) for dep in config["dependencies"]):
            raise ValueError("All dependencies must be strings")

    return config  # type: ignore


# Utility Functions
def create_health_status(
    status: ServiceStatus,
    error: Optional[str] = None,
    response_time_ms: Optional[float] = None,
    **metadata: Any,
) -> ServiceHealth:
    """
    Create a ServiceHealth object.

    Args:
        status: Service status
        error: Error message if any
        response_time_ms: Response time in milliseconds
        **metadata: Additional metadata

    Returns:
        ServiceHealth object
    """
    health: ServiceHealth = {
        "status": status.value,  # type: ignore
        "last_check": datetime.now().timestamp(),
    }

    if error is not None:
        health["error"] = error

    if response_time_ms is not None:
        health["response_time_ms"] = response_time_ms

    if metadata:
        health["metadata"] = metadata

    return health


def create_endpoint(
    host: str,
    port: int,
    protocol: Literal["http", "https", "ws", "wss", "tcp", "grpc"] = "http",
    health_endpoint: Optional[str] = "/health",
    **kwargs: Any,
) -> ServiceEndpoint:
    """
    Create a ServiceEndpoint object.

    Args:
        host: Service host
        port: Service port
        protocol: Connection protocol
        health_endpoint: Health check endpoint
        **kwargs: Additional endpoint configuration

    Returns:
        ServiceEndpoint object
    """
    endpoint: ServiceEndpoint = {
        "host": host,
        "port": port,
        "protocol": protocol,
        "health_endpoint": health_endpoint,
    }

    # Add optional fields
    for key in ["base_path", "timeout_seconds", "headers", "auth"]:
        if key in kwargs:
            endpoint[key] = kwargs[key]  # type: ignore

    return endpoint


# Async Type Aliases
AsyncHealthCheck = Callable[[], Awaitable[ServiceHealth]]
AsyncServiceValidator = Callable[[DockerService], Awaitable[bool]]
ServiceCallback = Union[Callable[[DockerService], None], Callable[[DockerService], Awaitable[None]]]


# Service Discovery Types
class DiscoveryMethod(Enum):
    """Service discovery methods."""

    STATIC = "static"
    DNS = "dns"
    CONSUL = "consul"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    MANUAL = "manual"


class DiscoveredService(TypedDict):
    """Discovered service information."""

    name: str
    endpoint: ServiceEndpoint
    discovery_method: DiscoveryMethod
    discovered_at: datetime
    tags: NotRequired[List[str]]
    metadata: NotRequired[Dict[str, Any]]