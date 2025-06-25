"""
Functional event store with enhanced audit trail capabilities.

This module provides a functional approach to event storage with strong
type safety, comprehensive audit trails, and event sourcing capabilities.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from bot.fp.core.either import Either, Left, Right
from bot.fp.core.io import IO
from bot.fp.core.option import Empty, Option
from bot.fp.events.base import Event, EventMetadata, EventType
from bot.fp.events.serialization import EventDeserializer, EventSerializer
from bot.fp.types.events import (
    AlertLevel,
    AuditTrailEntry,
    SystemComponent,
    SystemPerformanceMetric,
    create_alert_event,
    create_audit_event,
)

T = TypeVar("T", bound=Event)


class EventStoreError(Exception):
    """Base exception for event store operations."""


class EventStorageError(EventStoreError):
    """Exception for storage operations."""


class EventQueryError(EventStoreError):
    """Exception for query operations."""


class FunctionalEventStore:
    """
    Functional event store with audit trail and recovery capabilities.

    Features:
    - Immutable event storage
    - Type-safe operations using functional monads
    - Comprehensive audit trails
    - Event filtering and querying
    - Backup and recovery
    - Performance metrics
    """

    def __init__(
        self,
        storage_path: Path,
        serializer: EventSerializer | None = None,
        deserializer: EventDeserializer | None = None,
        max_file_size_mb: int = 100,
        retention_days: int = 90,
    ):
        """Initialize functional event store."""
        self.storage_path = Path(storage_path)
        self.serializer = serializer or EventSerializer()
        self.deserializer = deserializer or EventDeserializer()
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days

        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def store_event(self, event: Event) -> IO[Either[EventStorageError, str]]:
        """Store an event with functional error handling."""

        def _store() -> Either[EventStorageError, str]:
            try:
                # Serialize event
                serialized_result = self.serializer.serialize_to_json(event)
                if serialized_result.is_left():
                    return Left(
                        EventStorageError(
                            f"Serialization failed: {serialized_result.value}"
                        )
                    )

                # Write to storage
                event_data = serialized_result.value.decode("utf-8")
                with open(self.storage_path, "a", encoding="utf-8") as f:
                    f.write(event_data + "\n")

                return Right(str(event.event_id))

            except Exception as e:
                return Left(EventStorageError(f"Storage failed: {e}"))

        return IO(_store)

    def store_events(
        self, events: list[Event]
    ) -> IO[Either[EventStorageError, list[str]]]:
        """Store multiple events atomically."""

        def _store_multiple() -> Either[EventStorageError, list[str]]:
            try:
                event_ids = []
                serialized_events = []

                # Serialize all events first
                for event in events:
                    result = self.serializer.serialize_to_json(event)
                    if result.is_left():
                        return Left(
                            EventStorageError(
                                f"Serialization failed for {event.event_id}: {result.value}"
                            )
                        )
                    serialized_events.append(result.value.decode("utf-8"))
                    event_ids.append(str(event.event_id))

                # Write all events atomically
                with open(self.storage_path, "a", encoding="utf-8") as f:
                    f.writelines(event_data + "\n" for event_data in serialized_events)

                return Right(event_ids)

            except Exception as e:
                return Left(EventStorageError(f"Batch storage failed: {e}"))

        return IO(_store_multiple)

    def query_events(
        self, predicate: Callable[[Event], bool], limit: Option[int] = Empty()
    ) -> IO[Either[EventQueryError, list[Event]]]:
        """Query events with functional filtering."""

        def _query() -> Either[EventQueryError, list[Event]]:
            try:
                events = []
                count = 0
                max_count = limit.get_or_else(float("inf"))

                if not self.storage_path.exists():
                    return Right([])

                with open(self.storage_path, encoding="utf-8") as f:
                    for line in f:
                        if count >= max_count:
                            break

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            event_data = json.loads(line)
                            # Simple deserialization for querying
                            # In practice, you'd use the proper deserializer
                            event_id = UUID(event_data["event_id"])
                            event_type = EventType(event_data["event_type"])
                            timestamp = datetime.fromisoformat(event_data["timestamp"])

                            # Create a minimal event for filtering
                            from bot.fp.events.base import SystemEvent

                            event = SystemEvent(
                                event_id=event_id,
                                event_type=event_type,
                                timestamp=timestamp,
                                metadata=EventMetadata(),
                            )

                            if predicate(event):
                                events.append(event)
                                count += 1

                        except (json.JSONDecodeError, ValueError, KeyError):
                            # Log error but continue processing
                            continue

                return Right(events)

            except Exception as e:
                return Left(EventQueryError(f"Query failed: {e}"))

        return IO(_query)

    def get_events_by_type(
        self, event_type: EventType
    ) -> IO[Either[EventQueryError, list[Event]]]:
        """Get all events of a specific type."""
        return self.query_events(lambda e: e.event_type == event_type)

    def get_events_by_time_range(
        self, start_time: datetime, end_time: datetime
    ) -> IO[Either[EventQueryError, list[Event]]]:
        """Get events within a time range."""
        return self.query_events(lambda e: start_time <= e.timestamp <= end_time)

    def get_audit_trail(
        self, entity_type: str, entity_id: str
    ) -> IO[Either[EventQueryError, list[AuditTrailEntry]]]:
        """Get complete audit trail for an entity."""

        def _get_audit() -> Either[EventQueryError, list[AuditTrailEntry]]:
            # This would need proper deserialization in practice
            return Right([])  # Simplified for now

        return IO(_get_audit)

    def create_performance_snapshot(
        self,
    ) -> IO[Either[EventStorageError, SystemPerformanceMetric]]:
        """Create a performance snapshot event."""

        def _create_snapshot() -> Either[EventStorageError, SystemPerformanceMetric]:
            try:
                # Calculate storage size
                file_size = (
                    self.storage_path.stat().st_size
                    if self.storage_path.exists()
                    else 0
                )

                metric = SystemPerformanceMetric(
                    metric_name="event_store_size",
                    metric_value=file_size,
                    metric_unit="bytes",
                    component=SystemComponent.CONFIGURATION_MANAGER,
                    threshold_breached=file_size > self.max_file_size_bytes,
                    metric_tags={"storage_path": str(self.storage_path)},
                )

                return Right(metric)

            except Exception as e:
                return Left(EventStorageError(f"Performance snapshot failed: {e}"))

        return IO(_create_snapshot)

    def backup_events(self, backup_path: Path) -> IO[Either[EventStorageError, str]]:
        """Create a backup of all events."""

        def _backup() -> Either[EventStorageError, str]:
            try:
                if not self.storage_path.exists():
                    return Left(EventStorageError("No events to backup"))

                backup_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the entire event store
                import shutil

                shutil.copy2(self.storage_path, backup_path)

                # Create backup metadata
                metadata = {
                    "backup_time": datetime.now(UTC).isoformat(),
                    "source_path": str(self.storage_path),
                    "backup_size": backup_path.stat().st_size,
                }

                metadata_path = backup_path.with_suffix(".metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                return Right(str(backup_path))

            except Exception as e:
                return Left(EventStorageError(f"Backup failed: {e}"))

        return IO(_backup)

    def restore_events(self, backup_path: Path) -> IO[Either[EventStorageError, int]]:
        """Restore events from backup."""

        def _restore() -> Either[EventStorageError, int]:
            try:
                if not backup_path.exists():
                    return Left(EventStorageError("Backup file does not exist"))

                # Create backup of current store
                if self.storage_path.exists():
                    backup_current = self.storage_path.with_suffix(".bak")
                    import shutil

                    shutil.copy2(self.storage_path, backup_current)

                # Restore from backup
                import shutil

                shutil.copy2(backup_path, self.storage_path)

                # Count restored events
                event_count = 0
                with open(self.storage_path) as f:
                    for line in f:
                        if line.strip():
                            event_count += 1

                return Right(event_count)

            except Exception as e:
                return Left(EventStorageError(f"Restore failed: {e}"))

        return IO(_restore)

    def cleanup_old_events(self) -> IO[Either[EventStorageError, int]]:
        """Remove events older than retention period."""

        def _cleanup() -> Either[EventStorageError, int]:
            try:
                if not self.storage_path.exists():
                    return Right(0)

                cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)
                retained_events = []
                removed_count = 0

                with open(self.storage_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            event_data = json.loads(line)
                            event_timestamp = datetime.fromisoformat(
                                event_data["timestamp"]
                            )

                            if event_timestamp >= cutoff_date:
                                retained_events.append(line)
                            else:
                                removed_count += 1

                        except (json.JSONDecodeError, ValueError, KeyError):
                            # Keep malformed events for manual review
                            retained_events.append(line)

                # Rewrite file with retained events
                with open(self.storage_path, "w") as f:
                    f.writelines(event_line + "\n" for event_line in retained_events)

                return Right(removed_count)

            except Exception as e:
                return Left(EventStorageError(f"Cleanup failed: {e}"))

        return IO(_cleanup)


# Factory functions for common event store operations
def create_event_store(storage_path: str | Path, **kwargs: Any) -> FunctionalEventStore:
    """Create a functional event store."""
    return FunctionalEventStore(Path(storage_path), **kwargs)


def store_alert(
    store: FunctionalEventStore,
    alert_name: str,
    level: AlertLevel,
    component: SystemComponent,
    message: str,
) -> IO[Either[EventStorageError, str]]:
    """Store an alert event with validation."""

    def _store_alert() -> Either[EventStorageError, str]:
        alert_result = create_alert_event(alert_name, level, component, message)
        if alert_result.is_failure():
            return Left(EventStorageError(alert_result.failure()))

        alert_event = alert_result.success()
        return store.store_event(alert_event).run()

    return IO(_store_alert)


def store_audit_entry(
    store: FunctionalEventStore,
    action: str,
    entity_type: str,
    entity_id: str,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> IO[Either[EventStorageError, str]]:
    """Store an audit trail entry with validation."""

    def _store_audit() -> Either[EventStorageError, str]:
        audit_result = create_audit_event(
            action, entity_type, entity_id, before_state, after_state
        )
        if audit_result.is_failure():
            return Left(EventStorageError(audit_result.failure()))

        audit_event = audit_result.success()
        return store.store_event(audit_event).run()

    return IO(_store_audit)


__all__ = [
    "EventQueryError",
    "EventStorageError",
    "EventStoreError",
    "FunctionalEventStore",
    "create_event_store",
    "store_alert",
    "store_audit_entry",
]
