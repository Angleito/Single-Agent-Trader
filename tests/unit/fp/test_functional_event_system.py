"""
Test suite for the functional event system migration.

This module tests the enhanced event types, functional patterns,
and audit trail capabilities.
"""

import tempfile
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from bot.fp.core.either import Left, Right
from bot.fp.core.option import Empty, Some
from bot.fp.events.base import EventMetadata, EventType, SystemEvent
from bot.fp.events.functional_store import (
    FunctionalEventStore,
    create_event_store,
    store_alert,
    store_audit_entry,
)
from bot.fp.events.serialization import EventDeserializer, EventSerializer
from bot.fp.types.events import (
    AlertLevel,
    AlertTriggered,
    AuditTrailEntry,
    ConfigurationChanged,
    ErrorOccurred,
    NotificationChannel,
    NotificationSent,
    SystemComponent,
    create_alert_event,
    create_audit_event,
)


class TestEventBasics:
    """Test basic event functionality."""

    def test_event_metadata_creation(self):
        """Test event metadata creation and correlation."""
        correlation_id = uuid4()
        causation_id = uuid4()

        metadata = EventMetadata(
            correlation_id=correlation_id,
            causation_id=causation_id,
            user_id="test_user",
            session_id="test_session",
            source_system="test_system",
            environment="test",
            tags={"test": "value"},
        )

        assert metadata.correlation_id == correlation_id
        assert metadata.causation_id == causation_id
        assert metadata.user_id == "test_user"
        assert metadata.tags["test"] == "value"

    def test_event_metadata_chaining(self):
        """Test metadata chaining with functional methods."""
        base_metadata = EventMetadata()
        correlation_id = uuid4()

        chained_metadata = (
            base_metadata.with_correlation(correlation_id)
            .with_tag("component", "test")
            .with_tag("version", "1.0")
        )

        assert chained_metadata.correlation_id == correlation_id
        assert chained_metadata.tags["component"] == "test"
        assert chained_metadata.tags["version"] == "1.0"


class TestFunctionalEventTypes:
    """Test the enhanced functional event types."""

    def test_alert_triggered_event(self):
        """Test AlertTriggered event with Option types."""
        threshold = Some(Decimal("100.00"))
        actual = Some(Decimal("150.00"))
        symbol = Some("BTC-USD")

        alert = AlertTriggered(
            alert_id="test_alert",
            alert_name="Price Alert",
            level=AlertLevel.WARNING,
            component=SystemComponent.STRATEGY_ENGINE,
            message="Price exceeded threshold",
            threshold_value=threshold,
            actual_value=actual,
            symbol=symbol,
            triggered_conditions={"price": "150.00", "threshold": "100.00"},
            auto_acknowledge=False,
        )

        assert alert.alert_name == "Price Alert"
        assert alert.level == AlertLevel.WARNING
        assert alert.threshold_value.is_some()
        assert alert.threshold_value.get_or_else(Decimal(0)) == Decimal("100.00")
        assert alert.symbol.is_some()
        assert alert.symbol.get_or_else("") == "BTC-USD"

    def test_alert_event_serialization(self):
        """Test alert event serialization/deserialization."""
        alert = AlertTriggered(
            alert_id="test_alert",
            alert_name="Test Alert",
            level=AlertLevel.INFO,
            component=SystemComponent.RISK_MANAGER,
            message="Test message",
            threshold_value=Some(Decimal("50.0")),
            actual_value=Empty(),
            symbol=Some("ETH-USD"),
        )

        serialized = alert.to_dict()
        deserialized = AlertTriggered.from_dict(serialized)

        assert deserialized.alert_name == alert.alert_name
        assert deserialized.level == alert.level
        assert deserialized.threshold_value.is_some()
        assert deserialized.actual_value.is_empty()
        assert deserialized.symbol.get_or_else("") == "ETH-USD"

    def test_notification_sent_event(self):
        """Test NotificationSent event."""
        notification = NotificationSent(
            notification_id="notif_123",
            channel=NotificationChannel.EMAIL,
            recipient="user@example.com",
            subject="Trading Alert",
            content="Your alert was triggered",
            delivery_status="sent",
            retry_count=0,
            alert_reference=Some("alert_456"),
        )

        assert notification.channel == NotificationChannel.EMAIL
        assert notification.alert_reference.is_some()
        assert notification.alert_reference.get_or_else("") == "alert_456"

    def test_configuration_changed_event(self):
        """Test ConfigurationChanged event."""
        config_change = ConfigurationChanged(
            config_section="trading",
            config_key="max_position_size",
            old_value=0.1,
            new_value=0.2,
            changed_by="admin",
            change_reason="Risk adjustment",
            validation_result="success",
            affected_components=[
                SystemComponent.RISK_MANAGER,
                SystemComponent.ORDER_MANAGER,
            ],
            requires_restart=False,
        )

        assert config_change.config_section == "trading"
        assert config_change.old_value == 0.1
        assert config_change.new_value == 0.2
        assert SystemComponent.RISK_MANAGER in config_change.affected_components

    def test_audit_trail_entry(self):
        """Test AuditTrailEntry event."""
        audit = AuditTrailEntry(
            action="update_position",
            entity_type="position",
            entity_id="pos_123",
            user_id=Some("trader_1"),
            session_id=Some("session_456"),
            ip_address=Some("192.168.1.1"),
            before_state={"size": 1.0, "price": 50000},
            after_state={"size": 1.5, "price": 50000},
            change_summary="Increased position size from 1.0 to 1.5",
            compliance_relevant=True,
        )

        assert audit.action == "update_position"
        assert audit.user_id.is_some()
        assert audit.compliance_relevant is True
        assert "1.0 to 1.5" in audit.change_summary

    def test_error_occurred_event(self):
        """Test ErrorOccurred event with Option types."""
        error = ErrorOccurred(
            error_type="ConnectionError",
            error_message="Failed to connect to exchange",
            error_code=Some("CONN_001"),
            component=SystemComponent.EXCHANGE_CONNECTOR,
            stack_trace=Some("Traceback: ..."),
            user_impact="degraded",
            recovery_action=Some("Retry connection in 30 seconds"),
            error_context={"exchange": "coinbase", "attempt": 3},
        )

        assert error.error_type == "ConnectionError"
        assert error.error_code.is_some()
        assert error.recovery_action.is_some()
        assert error.error_context["exchange"] == "coinbase"


class TestEventCreationFunctions:
    """Test functional event creation utilities."""

    def test_create_alert_event_success(self):
        """Test successful alert event creation."""
        result = create_alert_event(
            alert_name="Test Alert",
            level=AlertLevel.ERROR,
            component=SystemComponent.STRATEGY_ENGINE,
            message="Test error message",
            threshold=Some(Decimal(100)),
            actual=Some(Decimal(200)),
            symbol=Some("BTC-USD"),
        )

        assert result.is_success()
        alert = result.success()
        assert alert.alert_name == "Test Alert"
        assert alert.level == AlertLevel.ERROR
        assert alert.threshold_value.is_some()

    def test_create_alert_event_validation_failure(self):
        """Test alert event creation with validation failure."""
        result = create_alert_event(
            alert_name="",  # Empty name should fail
            level=AlertLevel.INFO,
            component=SystemComponent.STRATEGY_ENGINE,
            message="Test message",
        )

        assert result.is_failure()
        assert "cannot be empty" in result.failure()

    def test_create_audit_event_success(self):
        """Test successful audit event creation."""
        before_state = {"status": "active", "balance": 1000}
        after_state = {"status": "inactive", "balance": 900}

        result = create_audit_event(
            action="deactivate_account",
            entity_type="account",
            entity_id="acc_123",
            before_state=before_state,
            after_state=after_state,
            user_id=Some("admin"),
        )

        assert result.is_success()
        audit = result.success()
        assert audit.action == "deactivate_account"
        assert "active -> inactive" in audit.change_summary
        assert audit.user_id.is_some()

    def test_create_audit_event_validation_failure(self):
        """Test audit event creation with validation failure."""
        result = create_audit_event(
            action="",  # Empty action should fail
            entity_type="test",
            entity_id="test_123",
            before_state={},
            after_state={},
        )

        assert result.is_failure()
        assert "cannot be empty" in result.failure()


class TestEventSerialization:
    """Test enhanced event serialization with functional types."""

    def test_option_serialization(self):
        """Test Option type serialization/deserialization."""
        serializer = EventSerializer()
        deserializer = EventDeserializer()

        # Test Some value
        some_value = Some("test_value")
        serialized = serializer._serialize_value(some_value)
        assert serialized["_option_type"] == "Some"
        assert serialized["_option_value"] == "test_value"

        # Test deserialization
        deserialized = deserializer._deserialize_value(serialized)
        assert deserialized.is_some()
        assert deserialized.get_or_else("") == "test_value"

        # Test Empty value
        empty_value = Empty()
        serialized_empty = serializer._serialize_value(empty_value)
        assert serialized_empty["_option_type"] == "Empty"

        deserialized_empty = deserializer._deserialize_value(serialized_empty)
        assert deserialized_empty.is_empty()

    def test_either_serialization(self):
        """Test Either type serialization/deserialization."""
        serializer = EventSerializer()
        deserializer = EventDeserializer()

        # Test Right value
        right_value = Right("success")
        serialized = serializer._serialize_value(right_value)
        assert serialized["_either_type"] == "Right"
        assert serialized["_either_value"] == "success"

        # Test deserialization
        deserialized = deserializer._deserialize_value(serialized)
        assert deserialized.is_right()
        assert deserialized.fold(lambda l: "", lambda r: r) == "success"

        # Test Left value
        left_value = Left("error")
        serialized_left = serializer._serialize_value(left_value)
        assert serialized_left["_either_type"] == "Left"

        deserialized_left = deserializer._deserialize_value(serialized_left)
        assert deserialized_left.is_left()


class TestFunctionalEventStore:
    """Test the functional event store."""

    def test_event_store_creation(self):
        """Test event store creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            assert isinstance(store, FunctionalEventStore)
            assert store.storage_path == store_path

    def test_store_event(self):
        """Test storing a single event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            event = SystemEvent(
                event_type=EventType.SYSTEM_STARTED,
                metadata=EventMetadata(environment="test"),
            )

            result = store.store_event(event).run()
            assert result.is_right()
            assert store_path.exists()

    def test_store_multiple_events(self):
        """Test storing multiple events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            events = [
                SystemEvent(event_type=EventType.SYSTEM_STARTED),
                SystemEvent(event_type=EventType.SYSTEM_STOPPED),
            ]

            result = store.store_events(events).run()
            assert result.is_right()
            event_ids = result.fold(lambda l: [], lambda r: r)
            assert len(event_ids) == 2

    def test_store_alert_function(self):
        """Test the store_alert utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            result = store_alert(
                store=store,
                alert_name="Test Alert",
                level=AlertLevel.WARNING,
                component=SystemComponent.STRATEGY_ENGINE,
                message="Test message",
            ).run()

            assert result.is_right()
            assert store_path.exists()

    def test_store_audit_entry_function(self):
        """Test the store_audit_entry utility function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            before_state = {"active": True}
            after_state = {"active": False}

            result = store_audit_entry(
                store=store,
                action="deactivate",
                entity_type="user",
                entity_id="user_123",
                before_state=before_state,
                after_state=after_state,
            ).run()

            assert result.is_right()
            assert store_path.exists()

    def test_event_query(self):
        """Test event querying functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            # Store some events
            events = [
                SystemEvent(event_type=EventType.SYSTEM_STARTED),
                SystemEvent(event_type=EventType.SYSTEM_STOPPED),
                SystemEvent(event_type=EventType.SYSTEM_ERROR),
            ]

            store.store_events(events).run()

            # Query for specific event type
            result = store.get_events_by_type(EventType.SYSTEM_STARTED).run()
            assert result.is_right()
            found_events = result.fold(lambda l: [], lambda r: r)
            assert len(found_events) >= 1

    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            backup_path = Path(tmpdir) / "backup.jsonl"
            store = create_event_store(store_path)

            # Store an event
            event = SystemEvent(event_type=EventType.SYSTEM_STARTED)
            store.store_event(event).run()

            # Create backup
            backup_result = store.backup_events(backup_path).run()
            assert backup_result.is_right()
            assert backup_path.exists()

            # Clear original and restore
            store_path.unlink()
            restore_result = store.restore_events(backup_path).run()
            assert restore_result.is_right()
            assert store_path.exists()


class TestEventSystemIntegration:
    """Test complete event system integration."""

    def test_end_to_end_workflow(self):
        """Test complete event workflow from creation to storage to querying."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            # 1. Create and store an alert
            alert_result = store_alert(
                store=store,
                alert_name="Integration Test Alert",
                level=AlertLevel.ERROR,
                component=SystemComponent.RISK_MANAGER,
                message="Risk limit exceeded",
            ).run()
            assert alert_result.is_right()

            # 2. Create and store an audit entry
            audit_result = store_audit_entry(
                store=store,
                action="update_risk_limit",
                entity_type="risk_config",
                entity_id="risk_001",
                before_state={"max_loss": 1000},
                after_state={"max_loss": 800},
            ).run()
            assert audit_result.is_right()

            # 3. Create and store a configuration change event
            config_event = ConfigurationChanged(
                config_section="risk",
                config_key="max_position_size",
                old_value=0.1,
                new_value=0.05,
                changed_by="admin",
                change_reason="Reduce risk exposure",
            )
            store_result = store.store_event(config_event).run()
            assert store_result.is_right()

            # 4. Create performance metric
            performance_result = store.create_performance_snapshot().run()
            assert performance_result.is_right()

            # 5. Verify storage
            assert store_path.exists()
            assert store_path.stat().st_size > 0

            # 6. Test backup
            backup_path = Path(tmpdir) / "test_backup.jsonl"
            backup_result = store.backup_events(backup_path).run()
            assert backup_result.is_right()

    def test_audit_trail_completeness(self):
        """Test that audit trails maintain complete history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "events.jsonl"
            store = create_event_store(store_path)

            # Simulate a series of related changes
            entity_id = "position_123"
            states = [
                {"size": 0.0, "price": 0},
                {"size": 1.0, "price": 50000},
                {"size": 1.5, "price": 50000},
                {"size": 0.0, "price": 51000},
            ]

            # Store audit trail for each state change
            for i in range(1, len(states)):
                store_audit_entry(
                    store=store,
                    action=f"update_position_step_{i}",
                    entity_type="position",
                    entity_id=entity_id,
                    before_state=states[i - 1],
                    after_state=states[i],
                ).run()

            # Verify all changes are recorded
            audit_result = store.get_audit_trail("position", entity_id).run()
            assert audit_result.is_right()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
