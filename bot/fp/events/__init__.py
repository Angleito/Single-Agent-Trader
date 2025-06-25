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
    "DomainEvent",
    # Base event types
    "Event",
    "EventMetadata",
    # Event storage
    "EventStore",
    "EventType",
    "FileEventStore",
    "IntegrationEvent",
    "SystemEvent",
    "TradingEvent",
    "create_correlated_event",
    "create_event_with_metadata",
]
