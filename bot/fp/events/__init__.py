"""Event sourcing and storage for trading events."""

from bot.fp.events.base import (
    DomainEvent,
    Event,
    EventMetadata,
    EventType,
    IntegrationEvent,
    SystemEvent,
    create_correlated_event,
    create_event_with_metadata,
)
from bot.fp.events.store import EventStore, FileEventStore, TradingEvent

__all__ = [
    # Base event types
    "Event",
    "DomainEvent",
    "SystemEvent",
    "IntegrationEvent",
    "EventType",
    "EventMetadata",
    "create_event_with_metadata",
    "create_correlated_event",
    # Event storage
    "EventStore",
    "FileEventStore",
    "TradingEvent",
]
