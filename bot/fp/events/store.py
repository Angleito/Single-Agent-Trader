"""
Append-only event store for trading events.

This module provides an event store implementation using JSON Lines format
for persistent storage of trading events with versioning and compaction support.
"""

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

# Type variable for events
T = TypeVar("T", bound="TradingEvent")


class TradingEvent(BaseModel):
    """Base class for all trading events."""

    event_id: str = Field(description="Unique event identifier")
    event_type: str = Field(description="Type of the event")
    timestamp: datetime = Field(description="When the event occurred")
    version: int = Field(default=1, description="Event schema version")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class EventStore(Protocol):
    """Protocol defining the event store interface."""

    def append(self, event: TradingEvent) -> None:
        """Append an event to the store."""
        ...

    def replay(self) -> list[TradingEvent]:
        """Replay all events from the store."""
        ...

    def query(self, predicate: Callable[[TradingEvent], bool]) -> list[TradingEvent]:
        """Query events matching a predicate."""
        ...

    def get_events_after(self, timestamp: datetime) -> list[TradingEvent]:
        """Get all events after a given timestamp."""
        ...


class FileEventStore:
    """
    File-based event store implementation using JSON Lines format.

    Features:
    - Append-only writes
    - Thread-safe concurrent appends
    - Event versioning support
    - Automatic compaction of old events
    - Efficient replay and querying
    """

    def __init__(
        self, filepath: Path, compaction_days: int = 30, max_file_size_mb: int = 100
    ):
        """
        Initialize the file event store.

        Args:
            filepath: Path to the event store file
            compaction_days: Days after which to compact events
            max_file_size_mb: Maximum file size before rotation
        """
        self.filepath = Path(filepath)
        self.compaction_days = compaction_days
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._lock = threading.Lock()

        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Initialize compacted events file
        self.compacted_filepath = self.filepath.with_suffix(".compacted.jsonl")

    def append(self, event: TradingEvent) -> None:
        """
        Append an event to the store.

        Thread-safe operation that appends events to the JSON Lines file.
        """
        with self._lock:
            # Check if we need to rotate the file
            if self._should_rotate():
                self._rotate_file()

            # Append event as JSON line
            with open(self.filepath, "a", encoding="utf-8") as f:
                json_line = event.json() + "\n"
                f.write(json_line)

    def replay(self) -> list[TradingEvent]:
        """
        Replay all events from the store.

        Returns events from both compacted and current files.
        """
        events = []

        # Read compacted events first
        if self.compacted_filepath.exists():
            events.extend(self._read_events_from_file(self.compacted_filepath))

        # Read current events
        if self.filepath.exists():
            events.extend(self._read_events_from_file(self.filepath))

        # Sort by timestamp to ensure correct order
        events.sort(key=lambda e: e.timestamp)

        return events

    def query(self, predicate: Callable[[TradingEvent], bool]) -> list[TradingEvent]:
        """
        Query events matching a predicate.

        Args:
            predicate: Function that returns True for matching events

        Returns:
            List of events matching the predicate
        """
        all_events = self.replay()
        return [event for event in all_events if predicate(event)]

    def get_events_after(self, timestamp: datetime) -> list[TradingEvent]:
        """
        Get all events after a given timestamp.

        Args:
            timestamp: Cutoff timestamp

        Returns:
            List of events after the timestamp
        """
        return self.query(lambda e: e.timestamp > timestamp)

    def compact(self) -> None:
        """
        Compact old events to reduce file size.

        Moves events older than compaction_days to a separate file
        and optionally aggregates them.
        """
        with self._lock:
            if not self.filepath.exists():
                return

            cutoff_date = datetime.now(UTC) - timedelta(days=self.compaction_days)
            current_events = self._read_events_from_file(self.filepath)

            # Separate old and recent events
            old_events = []
            recent_events = []

            for event in current_events:
                if event.timestamp < cutoff_date:
                    old_events.append(event)
                else:
                    recent_events.append(event)

            if not old_events:
                return

            # Append old events to compacted file
            with open(self.compacted_filepath, "a", encoding="utf-8") as f:
                for event in old_events:
                    f.write(event.json() + "\n")

            # Rewrite current file with only recent events
            with open(self.filepath, "w", encoding="utf-8") as f:
                for event in recent_events:
                    f.write(event.json() + "\n")

    def _read_events_from_file(self, filepath: Path) -> list[TradingEvent]:
        """Read events from a JSON Lines file."""
        events = []

        if not filepath.exists():
            return events

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Handle datetime parsing
                        if "timestamp" in data and isinstance(data["timestamp"], str):
                            data["timestamp"] = datetime.fromisoformat(
                                data["timestamp"]
                            )
                        event = TradingEvent(**data)
                        events.append(event)
                    except (json.JSONDecodeError, ValueError) as e:
                        # Log error but continue reading other events
                        print(f"Error parsing event: {e}")
                        continue

        return events

    def _should_rotate(self) -> bool:
        """Check if the current file should be rotated."""
        if not self.filepath.exists():
            return False

        file_size = self.filepath.stat().st_size
        return file_size > self.max_file_size_bytes

    def _rotate_file(self) -> None:
        """Rotate the current event file."""
        if not self.filepath.exists():
            return

        # Generate rotation filename with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        rotated_file = self.filepath.parent / f"{self.filepath.stem}.{timestamp}.jsonl"

        # Move current file to rotated name
        self.filepath.rename(rotated_file)

        # Compact the rotated file
        self._compact_rotated_file(rotated_file)

    def _compact_rotated_file(self, rotated_file: Path) -> None:
        """Compact a rotated file by moving all events to compacted storage."""
        events = self._read_events_from_file(rotated_file)

        with open(self.compacted_filepath, "a", encoding="utf-8") as f:
            for event in events:
                f.write(event.json() + "\n")

        # Remove the rotated file after compaction
        rotated_file.unlink()


# Example usage and testing
if __name__ == "__main__":
    import uuid
    from datetime import timedelta

    # Create a test event store
    store = FileEventStore(Path("test_events.jsonl"))

    # Create some test events
    events = [
        TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type="ORDER_PLACED",
            timestamp=datetime.now(UTC) - timedelta(days=40),
            data={"symbol": "BTC-USD", "side": "BUY", "size": 0.1},
        ),
        TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type="ORDER_FILLED",
            timestamp=datetime.now(UTC) - timedelta(days=20),
            data={"symbol": "BTC-USD", "price": 50000, "size": 0.1},
        ),
        TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type="POSITION_CLOSED",
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            data={"symbol": "BTC-USD", "pnl": 1000},
        ),
    ]

    # Append events
    for event in events:
        store.append(event)

    # Replay all events
    print("All events:")
    for event in store.replay():
        print(f"  {event.event_type} at {event.timestamp}")

    # Query recent events
    recent_cutoff = datetime.now(UTC) - timedelta(days=7)
    recent_events = store.get_events_after(recent_cutoff)
    print(f"\nEvents after {recent_cutoff}:")
    for event in recent_events:
        print(f"  {event.event_type} at {event.timestamp}")

    # Compact old events
    store.compact()
    print("\nCompaction completed")

    # Clean up test files
    Path("test_events.jsonl").unlink(missing_ok=True)
    Path("test_events.compacted.jsonl").unlink(missing_ok=True)
