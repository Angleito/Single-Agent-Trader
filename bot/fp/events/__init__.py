"""Event sourcing and storage for trading events."""

from bot.fp.events.base import (
    Event,
    DomainEvent,
    SystemEvent,
    IntegrationEvent,
    EventType,
    EventMetadata,
    create_event_with_metadata,
    create_correlated_event,
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
    "TradingEvent"
]
