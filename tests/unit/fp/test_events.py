"""Comprehensive tests for the event sourcing system.

This test suite covers:
1. Event store operations (append, replay, query)
2. Projections with various event sequences
3. Snapshot/restore functionality
4. Replay capabilities
5. Property-based tests for event ordering
6. Performance tests for large event streams
"""

import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from bot.fp.events.store import FileEventStore, TradingEvent
from bot.fp.types.events import (
    EventMetadata,
    MarketDataReceived,
    OrderFilled,
    OrderPlaced,
    PositionClosed,
    PositionOpened,
    StrategySignal,
    deserialize_event,
)
from bot.fp.types.events import (
    TradingEvent as TypedTradingEvent,
)
from bot.types import OrderSide, OrderType


class TestEventStore:
    """Test basic event store operations."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary event store."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            store = FileEventStore(Path(tmp.name))
            yield store
            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)
            Path(tmp.name).with_suffix(".compacted.jsonl").unlink(missing_ok=True)

    def test_append_and_replay(self, temp_store):
        """Test basic append and replay functionality."""
        # Create test events
        events = [
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="ORDER_PLACED",
                timestamp=datetime.now(UTC),
                data={"symbol": "BTC-USD", "side": "BUY", "size": 0.1},
            ),
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="ORDER_FILLED",
                timestamp=datetime.now(UTC),
                data={"symbol": "BTC-USD", "price": 50000, "size": 0.1},
            ),
        ]

        # Append events
        for event in events:
            temp_store.append(event)

        # Replay and verify
        replayed = temp_store.replay()
        assert len(replayed) == 2
        assert replayed[0].event_type == "ORDER_PLACED"
        assert replayed[1].event_type == "ORDER_FILLED"

    def test_query_with_predicate(self, temp_store):
        """Test querying events with predicates."""
        # Create events with different types
        events = [
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="ORDER_PLACED",
                timestamp=datetime.now(UTC),
                data={"symbol": "BTC-USD", "side": "BUY"},
            ),
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="ORDER_FILLED",
                timestamp=datetime.now(UTC),
                data={"symbol": "ETH-USD", "price": 3000},
            ),
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="ORDER_PLACED",
                timestamp=datetime.now(UTC),
                data={"symbol": "ETH-USD", "side": "SELL"},
            ),
        ]

        for event in events:
            temp_store.append(event)

        # Query only ORDER_PLACED events
        order_events = temp_store.query(lambda e: e.event_type == "ORDER_PLACED")
        assert len(order_events) == 2

        # Query only BTC events
        btc_events = temp_store.query(lambda e: e.data.get("symbol") == "BTC-USD")
        assert len(btc_events) == 1

    def test_get_events_after_timestamp(self, temp_store):
        """Test retrieving events after a specific timestamp."""
        now = datetime.now(UTC)

        # Create events at different times
        events = [
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="OLD_EVENT",
                timestamp=now - timedelta(hours=2),
                data={},
            ),
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="RECENT_EVENT",
                timestamp=now - timedelta(minutes=30),
                data={},
            ),
            TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="NEWEST_EVENT",
                timestamp=now,
                data={},
            ),
        ]

        for event in events:
            temp_store.append(event)

        # Get events from last hour
        cutoff = now - timedelta(hours=1)
        recent_events = temp_store.get_events_after(cutoff)

        assert len(recent_events) == 2
        assert all(e.timestamp > cutoff for e in recent_events)
        assert "OLD_EVENT" not in [e.event_type for e in recent_events]

    def test_concurrent_appends(self, temp_store):
        """Test thread-safe concurrent appends."""
        num_threads = 10
        events_per_thread = 50

        def append_events(thread_id: int):
            for i in range(events_per_thread):
                event = TradingEvent(
                    event_id=f"thread-{thread_id}-event-{i}",
                    event_type="CONCURRENT_TEST",
                    timestamp=datetime.now(UTC),
                    data={"thread_id": thread_id, "sequence": i},
                )
                temp_store.append(event)

        # Run concurrent appends
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(append_events, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        # Verify all events were stored
        all_events = temp_store.replay()
        assert len(all_events) == num_threads * events_per_thread

        # Verify no events were lost
        event_ids = {e.event_id for e in all_events}
        assert len(event_ids) == num_threads * events_per_thread

    def test_compaction(self, temp_store):
        """Test event compaction functionality."""
        now = datetime.now(UTC)

        # Create old and recent events
        old_events = [
            TradingEvent(
                event_id=f"old-{i}",
                event_type="OLD_EVENT",
                timestamp=now - timedelta(days=40),
                data={"index": i},
            )
            for i in range(5)
        ]

        recent_events = [
            TradingEvent(
                event_id=f"recent-{i}",
                event_type="RECENT_EVENT",
                timestamp=now - timedelta(hours=1),
                data={"index": i},
            )
            for i in range(3)
        ]

        # Append all events
        for event in old_events + recent_events:
            temp_store.append(event)

        # Compact
        temp_store.compact()

        # Verify all events are still accessible
        all_events = temp_store.replay()
        assert len(all_events) == 8

        # Check that old events are in compacted file
        assert temp_store.compacted_filepath.exists()

        # Verify main file only has recent events
        main_events = temp_store._read_events_from_file(temp_store.filepath)
        assert len(main_events) == 3
        assert all(e.event_type == "RECENT_EVENT" for e in main_events)

    def test_file_rotation(self, temp_store):
        """Test automatic file rotation on size limit."""
        # Set a very small max file size
        temp_store.max_file_size_bytes = 1024  # 1KB

        # Create many events to trigger rotation
        for i in range(100):
            event = TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="LARGE_EVENT",
                timestamp=datetime.now(UTC),
                data={
                    "large_data": "x" * 100,  # Make events bigger
                    "index": i,
                },
            )
            temp_store.append(event)

        # Verify all events are still accessible
        all_events = temp_store.replay()
        assert len(all_events) == 100

        # Check that compacted file exists (from rotation)
        assert temp_store.compacted_filepath.exists()


class TestTypedEvents:
    """Test typed event classes and serialization."""

    def test_market_data_event_serialization(self):
        """Test MarketDataReceived event serialization/deserialization."""
        event = MarketDataReceived(
            symbol="BTC-USD",
            price=Decimal("50000.50"),
            volume=Decimal("100.5"),
            bid=Decimal("49999.00"),
            ask=Decimal("50001.00"),
            high_24h=Decimal("51000.00"),
            low_24h=Decimal("49000.00"),
            open_24h=Decimal("49500.00"),
        )

        # Serialize
        data = event.to_dict()
        assert data["event_type"] == "MarketDataReceived"
        assert data["symbol"] == "BTC-USD"
        assert data["price"] == "50000.50"

        # Deserialize
        restored = deserialize_event(data)
        assert isinstance(restored, MarketDataReceived)
        assert restored.symbol == "BTC-USD"
        assert restored.price == Decimal("50000.50")

    def test_order_placed_event_serialization(self):
        """Test OrderPlaced event serialization/deserialization."""
        event = OrderPlaced(
            order_id="order-123",
            symbol="ETH-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal("10.5"),
            price=Decimal("3000.00"),
            time_in_force="GTC",
        )

        # Serialize
        data = event.to_dict()
        assert data["order_id"] == "order-123"
        assert data["side"] == "buy"
        assert data["order_type"] == "limit"

        # Deserialize
        restored = deserialize_event(data)
        assert isinstance(restored, OrderPlaced)
        assert restored.side == OrderSide.BUY
        assert restored.order_type == OrderType.LIMIT

    def test_position_lifecycle_events(self):
        """Test position open/close event serialization."""
        position_id = uuid.uuid4()

        # Position opened
        open_event = PositionOpened(
            position_id=position_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            size=Decimal("0.5"),
            entry_price=Decimal(50000),
            leverage=10,
            initial_margin=Decimal(2500),
        )

        # Position closed
        close_event = PositionClosed(
            position_id=position_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_price=Decimal(50000),
            exit_price=Decimal(51000),
            size=Decimal("0.5"),
            realized_pnl=Decimal(500),
            fees=Decimal(10),
            duration_seconds=3600,
        )

        # Test serialization round-trip
        for event in [open_event, close_event]:
            data = event.to_dict()
            restored = deserialize_event(data)
            assert type(restored) == type(event)
            assert restored.position_id == position_id

    def test_event_metadata(self):
        """Test event metadata handling."""
        correlation_id = uuid.uuid4()
        causation_id = uuid.uuid4()

        metadata = EventMetadata(
            version="2.0.0",
            source="test-bot",
            correlation_id=correlation_id,
            causation_id=causation_id,
        )

        event = StrategySignal(
            metadata=metadata,
            signal_type="LONG",
            confidence=0.85,
            symbol="BTC-USD",
            current_price=Decimal(50000),
            indicators={"rsi": 65, "macd": "bullish"},
            reasoning="Strong upward momentum",
        )

        # Serialize and deserialize
        data = event.to_dict()
        restored = deserialize_event(data)

        assert restored.metadata.version == "2.0.0"
        assert restored.metadata.source == "test-bot"
        assert restored.metadata.correlation_id == correlation_id

    def test_unknown_event_type_handling(self):
        """Test handling of unknown event types."""
        data = {
            "event_type": "UnknownEventType",
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "metadata": EventMetadata().to_dict(),
        }

        with pytest.raises(ValueError, match="Unknown event type"):
            deserialize_event(data)


class EventProjection:
    """Example projection for testing."""

    def __init__(self):
        self.positions: dict[str, dict[str, Any]] = {}
        self.total_volume = Decimal(0)
        self.event_count = 0

    def apply(self, event: TypedTradingEvent):
        """Apply event to projection."""
        self.event_count += 1

        if isinstance(event, PositionOpened):
            self.positions[str(event.position_id)] = {
                "symbol": event.symbol,
                "side": event.side,
                "size": event.size,
                "entry_price": event.entry_price,
                "status": "open",
            }
            self.total_volume += event.size * event.entry_price

        elif isinstance(event, PositionClosed):
            if str(event.position_id) in self.positions:
                self.positions[str(event.position_id)]["status"] = "closed"
                self.positions[str(event.position_id)]["exit_price"] = event.exit_price
                self.positions[str(event.position_id)]["pnl"] = event.realized_pnl

        elif isinstance(event, OrderFilled):
            self.total_volume += event.fill_size * event.fill_price


class TestProjections:
    """Test event projections and replay."""

    def test_basic_projection(self):
        """Test applying events to a projection."""
        projection = EventProjection()
        position_id = uuid.uuid4()

        events = [
            PositionOpened(
                position_id=position_id,
                symbol="BTC-USD",
                side=OrderSide.BUY,
                size=Decimal("1.0"),
                entry_price=Decimal(50000),
                leverage=5,
                initial_margin=Decimal(10000),
            ),
            OrderFilled(
                order_id="order-1",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                fill_price=Decimal(50000),
                fill_size=Decimal("1.0"),
                fees=Decimal(50),
            ),
            PositionClosed(
                position_id=position_id,
                symbol="BTC-USD",
                side=OrderSide.BUY,
                entry_price=Decimal(50000),
                exit_price=Decimal(52000),
                size=Decimal("1.0"),
                realized_pnl=Decimal(2000),
                fees=Decimal(100),
                duration_seconds=7200,
            ),
        ]

        # Apply events
        for event in events:
            projection.apply(event)

        # Verify projection state
        assert projection.event_count == 3
        assert projection.total_volume == Decimal(100000)  # 50k + 50k

        position = projection.positions[str(position_id)]
        assert position["status"] == "closed"
        assert position["pnl"] == Decimal(2000)

    def test_projection_replay_from_store(self, temp_store):
        """Test replaying events from store to rebuild projection."""
        # Create and store events
        events = []
        for i in range(5):
            position_id = uuid.uuid4()

            # Open position
            open_event = PositionOpened(
                position_id=position_id,
                symbol=f"TOKEN{i}-USD",
                side=OrderSide.BUY,
                size=Decimal(1),
                entry_price=Decimal(f"{1000 * (i + 1)}"),
                leverage=5,
                initial_margin=Decimal(1000),
            )
            events.append(open_event)

            # Close position
            close_event = PositionClosed(
                position_id=position_id,
                symbol=f"TOKEN{i}-USD",
                side=OrderSide.BUY,
                entry_price=Decimal(f"{1000 * (i + 1)}"),
                exit_price=Decimal(f"{1100 * (i + 1)}"),
                size=Decimal(1),
                realized_pnl=Decimal(f"{100 * (i + 1)}"),
                fees=Decimal(10),
                duration_seconds=3600,
            )
            events.append(close_event)

        # Store events as generic TradingEvent
        for event in events:
            generic_event = TradingEvent(
                event_id=str(event.event_id),
                event_type=event.event_type,
                timestamp=event.timestamp,
                data=json.loads(json.dumps(event.to_dict())),
            )
            temp_store.append(generic_event)

        # Replay and project
        projection = EventProjection()
        replayed = temp_store.replay()

        for generic_event in replayed:
            # Convert back to typed event
            typed_event = deserialize_event(generic_event.data)
            projection.apply(typed_event)

        # Verify projection
        assert projection.event_count == 10
        assert len(projection.positions) == 5
        assert all(p["status"] == "closed" for p in projection.positions.values())


class TestSnapshotRestore:
    """Test snapshot and restore functionality."""

    @pytest.fixture
    def snapshot_store(self):
        """Create a store with snapshot support."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(Path(tmpdir) / "events.jsonl")
            snapshot_path = Path(tmpdir) / "snapshots"
            snapshot_path.mkdir()
            yield store, snapshot_path

    def test_projection_snapshot(self, snapshot_store):
        """Test creating and restoring projection snapshots."""
        store, snapshot_path = snapshot_store

        # Create events
        events = []
        for i in range(10):
            event = OrderPlaced(
                order_id=f"order-{i}",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=Decimal("0.1"),
            )
            events.append(event)

            # Store as generic event
            generic = TradingEvent(
                event_id=str(event.event_id),
                event_type=event.event_type,
                timestamp=event.timestamp,
                data=json.loads(json.dumps(event.to_dict())),
            )
            store.append(generic)

        # Create projection
        projection = EventProjection()
        for event in events:
            projection.apply(event)

        # Snapshot projection state
        snapshot_data = {
            "event_count": projection.event_count,
            "total_volume": str(projection.total_volume),
            "positions": projection.positions,
            "last_event_id": str(events[-1].event_id),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        snapshot_file = snapshot_path / "projection_snapshot.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f)

        # Restore from snapshot
        with open(snapshot_file) as f:
            restored_data = json.load(f)

        restored_projection = EventProjection()
        restored_projection.event_count = restored_data["event_count"]
        restored_projection.total_volume = Decimal(restored_data["total_volume"])
        restored_projection.positions = restored_data["positions"]

        # Verify restoration
        assert restored_projection.event_count == projection.event_count
        assert restored_projection.total_volume == projection.total_volume

    def test_incremental_replay_after_snapshot(self, snapshot_store):
        """Test replaying only new events after snapshot."""
        store, snapshot_path = snapshot_store

        # Phase 1: Initial events
        phase1_events = []
        for i in range(5):
            event = TradingEvent(
                event_id=f"phase1-{i}",
                event_type="PHASE1_EVENT",
                timestamp=datetime.now(UTC),
                data={"index": i},
            )
            phase1_events.append(event)
            store.append(event)

        # Create snapshot after phase 1
        snapshot_timestamp = datetime.now(UTC)

        # Phase 2: New events after snapshot
        time.sleep(0.1)  # Ensure different timestamps
        phase2_events = []
        for i in range(3):
            event = TradingEvent(
                event_id=f"phase2-{i}",
                event_type="PHASE2_EVENT",
                timestamp=datetime.now(UTC),
                data={"index": i},
            )
            phase2_events.append(event)
            store.append(event)

        # Replay only events after snapshot
        new_events = store.get_events_after(snapshot_timestamp)

        assert len(new_events) == 3
        assert all(e.event_type == "PHASE2_EVENT" for e in new_events)


# Property-based tests
class EventStoreStateMachine(RuleBasedStateMachine):
    """State machine for property-based testing of event store."""

    def __init__(self):
        super().__init__()
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.store = FileEventStore(Path(self.temp_file.name))
        self.model_events: list[TradingEvent] = []

    events = Bundle("events")

    @initialize()
    def setup(self):
        """Initialize state machine."""

    @rule(
        target=events,
        event_type=st.sampled_from(["ORDER", "TRADE", "POSITION"]),
        data=st.dictionaries(st.text(), st.integers()),
    )
    def append_event(self, event_type, data):
        """Append a new event."""
        event = TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(UTC),
            data=data,
        )
        self.store.append(event)
        self.model_events.append(event)
        return event

    @rule()
    def replay_and_verify(self):
        """Replay events and verify against model."""
        replayed = self.store.replay()

        # Should have same number of events
        assert len(replayed) == len(self.model_events)

        # Events should be in timestamp order
        for i in range(1, len(replayed)):
            assert replayed[i].timestamp >= replayed[i - 1].timestamp

    @rule(
        timestamp=st.datetimes(
            min_value=datetime(2024, 1, 1, tzinfo=UTC),
            max_value=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    def query_after_timestamp(self, timestamp):
        """Query events after timestamp."""
        result = self.store.get_events_after(timestamp)
        model_result = [e for e in self.model_events if e.timestamp > timestamp]

        assert len(result) == len(model_result)

    @invariant()
    def events_are_immutable(self):
        """Verify stored events haven't changed."""
        if self.model_events:
            replayed = self.store.replay()
            for stored, model in zip(
                replayed,
                sorted(self.model_events, key=lambda e: e.timestamp),
                strict=False,
            ):
                assert stored.event_id == model.event_id
                assert stored.event_type == model.event_type

    def teardown(self):
        """Clean up resources."""
        Path(self.temp_file.name).unlink(missing_ok=True)
        Path(self.temp_file.name).with_suffix(".compacted.jsonl").unlink(
            missing_ok=True
        )


# Performance tests
class TestPerformance:
    """Performance tests for large event streams."""

    @pytest.mark.slow
    def test_large_event_stream_append(self, temp_store):
        """Test performance with large number of events."""
        num_events = 10000

        start_time = time.time()

        for i in range(num_events):
            event = TradingEvent(
                event_id=str(uuid.uuid4()),
                event_type="PERF_TEST",
                timestamp=datetime.now(UTC),
                data={"index": i, "random": str(uuid.uuid4())},
            )
            temp_store.append(event)

        append_time = time.time() - start_time

        # Should be able to append 10k events in reasonable time
        assert append_time < 10.0  # 10 seconds

        # Verify all events
        start_time = time.time()
        all_events = temp_store.replay()
        replay_time = time.time() - start_time

        assert len(all_events) == num_events
        assert replay_time < 2.0  # 2 seconds to replay

    @pytest.mark.slow
    def test_concurrent_read_write_performance(self, temp_store):
        """Test performance under concurrent read/write load."""
        num_writers = 5
        num_readers = 5
        events_per_writer = 1000
        reads_per_reader = 100

        write_times = []
        read_times = []

        def writer_task(writer_id):
            times = []
            for i in range(events_per_writer):
                start = time.time()
                event = TradingEvent(
                    event_id=f"writer-{writer_id}-event-{i}",
                    event_type="CONCURRENT_PERF",
                    timestamp=datetime.now(UTC),
                    data={"writer": writer_id, "seq": i},
                )
                temp_store.append(event)
                times.append(time.time() - start)
            return times

        def reader_task(reader_id):
            times = []
            for i in range(reads_per_reader):
                start = time.time()
                events = temp_store.replay()
                times.append(time.time() - start)
                time.sleep(0.01)  # Small delay between reads
            return times

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=num_writers + num_readers) as executor:
            # Start writers
            writer_futures = [
                executor.submit(writer_task, i) for i in range(num_writers)
            ]

            # Start readers
            reader_futures = [
                executor.submit(reader_task, i) for i in range(num_readers)
            ]

            # Collect results
            for future in writer_futures:
                write_times.extend(future.result())

            for future in reader_futures:
                read_times.extend(future.result())

        # Analyze performance
        avg_write_time = sum(write_times) / len(write_times)
        avg_read_time = sum(read_times) / len(read_times)

        # Performance assertions
        assert avg_write_time < 0.001  # < 1ms per write
        assert avg_read_time < 0.1  # < 100ms per full replay

        # Verify data integrity
        final_events = temp_store.replay()
        assert len(final_events) == num_writers * events_per_writer

    def test_query_performance_with_large_dataset(self, temp_store):
        """Test query performance on large event sets."""
        # Generate events over different time periods
        now = datetime.now(UTC)

        for days_ago in range(30):
            timestamp = now - timedelta(days=days_ago)
            for i in range(100):  # 100 events per day
                event = TradingEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=f"TYPE_{i % 5}",  # 5 different types
                    timestamp=timestamp,
                    data={
                        "symbol": f"SYMBOL_{i % 10}",  # 10 different symbols
                        "value": i,
                    },
                )
                temp_store.append(event)

        # Test query performance
        start_time = time.time()

        # Query by event type
        type_results = temp_store.query(lambda e: e.event_type == "TYPE_0")
        type_query_time = time.time() - start_time

        # Query by time range (last week)
        start_time = time.time()
        week_ago = now - timedelta(days=7)
        time_results = temp_store.get_events_after(week_ago)
        time_query_time = time.time() - start_time

        # Complex query
        start_time = time.time()
        complex_results = temp_store.query(
            lambda e: e.event_type in ["TYPE_0", "TYPE_1"]
            and e.data.get("symbol") == "SYMBOL_5"
            and e.timestamp > week_ago
        )
        complex_query_time = time.time() - start_time

        # Performance assertions
        assert type_query_time < 0.5  # 500ms
        assert time_query_time < 0.5  # 500ms
        assert complex_query_time < 1.0  # 1 second

        # Verify results
        assert len(type_results) == 600  # 30 days * 100 events/day / 5 types
        assert len(time_results) == 700  # 7 days * 100 events/day
        assert len(complex_results) > 0


# Integration tests
class TestEventSourcingIntegration:
    """Integration tests for complete event sourcing scenarios."""

    def test_trading_session_replay(self, temp_store):
        """Test replaying a complete trading session."""
        # Simulate a trading session
        session_events = []

        # Market opens
        market_open = MarketDataReceived(
            symbol="BTC-USD",
            price=Decimal(50000),
            volume=Decimal(1000),
            bid=Decimal(49999),
            ask=Decimal(50001),
            high_24h=Decimal(51000),
            low_24h=Decimal(49000),
            open_24h=Decimal(49500),
        )
        session_events.append(market_open)

        # Strategy signal
        signal = StrategySignal(
            signal_type="LONG",
            confidence=0.75,
            symbol="BTC-USD",
            current_price=Decimal(50000),
            indicators={"rsi": 65, "macd": "bullish"},
            reasoning="Bullish momentum detected",
        )
        session_events.append(signal)

        # Order placed
        order = OrderPlaced(
            order_id="session-order-1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=Decimal("0.5"),
        )
        session_events.append(order)

        # Order filled
        fill = OrderFilled(
            order_id="session-order-1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            fill_price=Decimal(50001),
            fill_size=Decimal("0.5"),
            fees=Decimal(25),
        )
        session_events.append(fill)

        # Position opened
        position_id = uuid.uuid4()
        position = PositionOpened(
            position_id=position_id,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            size=Decimal("0.5"),
            entry_price=Decimal(50001),
            leverage=5,
            initial_margin=Decimal(5000),
        )
        session_events.append(position)

        # Store all events
        for event in session_events:
            generic = TradingEvent(
                event_id=str(event.event_id),
                event_type=event.event_type,
                timestamp=event.timestamp,
                data=json.loads(json.dumps(event.to_dict())),
            )
            temp_store.append(generic)

        # Replay and verify sequence
        replayed = temp_store.replay()
        assert len(replayed) == 5

        # Verify event sequence makes sense
        event_types = [e.event_type for e in replayed]
        expected_sequence = [
            "MarketDataReceived",
            "StrategySignal",
            "OrderPlaced",
            "OrderFilled",
            "PositionOpened",
        ]
        assert event_types == expected_sequence

        # Build projection from replay
        projection = EventProjection()
        for generic_event in replayed:
            typed_event = deserialize_event(generic_event.data)
            projection.apply(typed_event)

        assert projection.event_count == 5
        assert str(position_id) in projection.positions
        assert projection.positions[str(position_id)]["status"] == "open"


# Run property-based tests
@pytest.mark.slow
def test_event_store_properties():
    """Run property-based tests."""
    # Run with explicit settings for CI
    settings = settings(max_examples=50, deadline=5000)
    TestEventStore = EventStoreStateMachine.TestCase
    TestEventStore.settings = settings
    state_machine_test = TestEventStore()
    state_machine_test.runTest()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
