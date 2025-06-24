"""
Base event types for the functional event system.

This module provides the foundational event types and interfaces used
throughout the event sourcing system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EventType(Enum):
    """Standard event types for the trading system."""

    # Market data events
    MARKET_DATA_RECEIVED = "market_data_received"
    PRICE_UPDATE = "price_update"
    VOLUME_UPDATE = "volume_update"

    # Trading events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"

    # Strategy events
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_DECISION = "strategy_decision"
    STRATEGY_ERROR = "strategy_error"

    # Risk events
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    RISK_WARNING = "risk_warning"
    EMERGENCY_STOP = "emergency_stop"

    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"

    # Notification events
    ALERT_TRIGGERED = "alert_triggered"
    NOTIFICATION_SENT = "notification_sent"

    # Audit events
    CONFIG_CHANGED = "config_changed"
    USER_ACTION = "user_action"
    SYSTEM_METRIC = "system_metric"


@dataclass(frozen=True)
class EventMetadata:
    """Metadata associated with events."""

    correlation_id: UUID | None = None
    causation_id: UUID | None = None
    user_id: str | None = None
    session_id: str | None = None
    source_system: str = "trading-bot"
    environment: str = "production"
    tags: dict[str, str] = field(default_factory=dict)

    def with_correlation(self, correlation_id: UUID) -> "EventMetadata":
        """Create new metadata with correlation ID."""
        return EventMetadata(
            correlation_id=correlation_id,
            causation_id=self.causation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            source_system=self.source_system,
            environment=self.environment,
            tags=self.tags,
        )

    def with_causation(self, causation_id: UUID) -> "EventMetadata":
        """Create new metadata with causation ID."""
        return EventMetadata(
            correlation_id=self.correlation_id,
            causation_id=causation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            source_system=self.source_system,
            environment=self.environment,
            tags=self.tags,
        )

    def with_tag(self, key: str, value: str) -> "EventMetadata":
        """Create new metadata with additional tag."""
        new_tags = self.tags.copy()
        new_tags[key] = value
        return EventMetadata(
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            user_id=self.user_id,
            session_id=self.session_id,
            source_system=self.source_system,
            environment=self.environment,
            tags=new_tags,
        )


@dataclass(frozen=True)
class Event(ABC):
    """Base class for all events in the system."""

    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = field(default=EventType.SYSTEM_STARTED)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: str = ""
    version: int = 1
    metadata: EventMetadata = field(default_factory=EventMetadata)
    source: str = "trading-bot"

    @abstractmethod
    def data(self) -> dict[str, Any]:
        """Get the event data as a dictionary."""

    @property
    def correlation_id(self) -> UUID | None:
        """Get correlation ID from metadata."""
        return self.metadata.correlation_id

    @property
    def causation_id(self) -> UUID | None:
        """Get causation ID from metadata."""
        return self.metadata.causation_id

    def with_metadata(self, metadata: EventMetadata) -> "Event":
        """Create new event with updated metadata."""
        # This is a simplified implementation - in practice,
        # you'd want to use copy methods for the specific event type
        return self.__class__(
            event_id=self.event_id,
            event_type=self.event_type,
            timestamp=self.timestamp,
            aggregate_id=self.aggregate_id,
            version=self.version,
            metadata=metadata,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "version": self.version,
            "metadata": {
                "correlation_id": (
                    str(self.metadata.correlation_id)
                    if self.metadata.correlation_id
                    else None
                ),
                "causation_id": (
                    str(self.metadata.causation_id)
                    if self.metadata.causation_id
                    else None
                ),
                "user_id": self.metadata.user_id,
                "session_id": self.metadata.session_id,
                "source_system": self.metadata.source_system,
                "environment": self.metadata.environment,
                "tags": self.metadata.tags,
            },
            "source": self.source,
            "data": self.data(),
        }


@dataclass(frozen=True)
class DomainEvent(Event):
    """Base class for domain-specific events."""

    def __post_init__(self):
        # Ensure domain events have proper aggregate IDs
        if not self.aggregate_id:
            object.__setattr__(self, "aggregate_id", str(self.event_id))


@dataclass(frozen=True)
class SystemEvent(Event):
    """Base class for system-level events."""

    def __post_init__(self):
        # System events use system as aggregate
        object.__setattr__(self, "aggregate_id", "system")


@dataclass(frozen=True)
class IntegrationEvent(Event):
    """Base class for integration/external events."""

    external_id: str | None = None
    external_system: str | None = None

    def __post_init__(self):
        # Integration events use external ID if available
        if self.external_id and not self.aggregate_id:
            object.__setattr__(self, "aggregate_id", self.external_id)


# Event creation utilities
def create_event_with_metadata(
    event_class: type[Event],
    event_type: EventType,
    aggregate_id: str = "",
    metadata: EventMetadata | None = None,
    **kwargs: Any,
) -> Event:
    """Create an event with proper metadata."""
    if metadata is None:
        metadata = EventMetadata()

    return event_class(
        event_type=event_type, aggregate_id=aggregate_id, metadata=metadata, **kwargs
    )


def create_correlated_event(
    event_class: type[Event],
    event_type: EventType,
    correlation_id: UUID,
    aggregate_id: str = "",
    causation_id: UUID | None = None,
    **kwargs: Any,
) -> Event:
    """Create an event with correlation tracking."""
    metadata = EventMetadata(
        correlation_id=correlation_id,
        causation_id=causation_id,
    )

    return create_event_with_metadata(
        event_class=event_class,
        event_type=event_type,
        aggregate_id=aggregate_id,
        metadata=metadata,
        **kwargs,
    )


__all__ = [
    "DomainEvent",
    "Event",
    "EventMetadata",
    "EventType",
    "IntegrationEvent",
    "SystemEvent",
    "create_correlated_event",
    "create_event_with_metadata",
]
