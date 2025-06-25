"""Event serialization with schema evolution support.

This module provides JSON and binary serialization for events with:
- Schema versioning and migration
- Backward compatibility
- Compression support
- Schema registry integration
"""

import gzip
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar, cast

try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

    # Create a stub msgpack module for graceful degradation
    class msgpack:
        @staticmethod
        def packb(data, **kwargs):
            """Fallback serialization using JSON."""
            return json.dumps(data).encode("utf-8")

        @staticmethod
        def unpackb(data, **kwargs):
            """Fallback deserialization using JSON."""
            return json.loads(data.decode("utf-8"))


from pydantic import BaseModel

from bot.fp.core.either import Either, Left, Right
from bot.fp.core.option import Empty, Option, Some
from bot.fp.events.base import Event

T = TypeVar("T", bound=Event)


class SerializationError(Exception):
    """Raised when serialization fails."""


class DeserializationError(Exception):
    """Raised when deserialization fails."""


class SchemaVersion:
    """Schema version information."""

    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    @property
    def version_string(self) -> str:
        """Get version as string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version is compatible with another.

        Compatible means same major version and minor >= other.minor
        """
        return self.major == other.major and self.minor >= other.minor

    def __str__(self) -> str:
        return self.version_string

    @classmethod
    def from_string(cls, version: str) -> "SchemaVersion":
        """Create from version string."""
        parts = version.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


class SchemaMigration(ABC):
    """Base class for schema migrations."""

    @property
    @abstractmethod
    def from_version(self) -> SchemaVersion:
        """Version to migrate from."""

    @property
    @abstractmethod
    def to_version(self) -> SchemaVersion:
        """Version to migrate to."""

    @abstractmethod
    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate data from old schema to new schema."""


class SchemaRegistry:
    """Registry for event schemas and migrations."""

    def __init__(self):
        self._schemas: dict[str, dict[str, Any]] = {}
        self._migrations: dict[str, list[SchemaMigration]] = {}
        self._current_versions: dict[str, SchemaVersion] = {}

    def register_schema(
        self,
        event_type: str,
        schema: dict[str, Any],
        version: SchemaVersion,
    ) -> None:
        """Register a schema for an event type."""
        version_key = f"{event_type}:{version.version_string}"
        self._schemas[version_key] = schema
        self._current_versions[event_type] = version

    def register_migration(self, event_type: str, migration: SchemaMigration) -> None:
        """Register a migration for an event type."""
        if event_type not in self._migrations:
            self._migrations[event_type] = []
        self._migrations[event_type].append(migration)

    def get_schema(
        self, event_type: str, version: SchemaVersion | None = None
    ) -> dict[str, Any] | None:
        """Get schema for an event type and version."""
        if version is None:
            version = self._current_versions.get(event_type)
        if version is None:
            return None
        version_key = f"{event_type}:{version.version_string}"
        return self._schemas.get(version_key)

    def get_current_version(self, event_type: str) -> SchemaVersion | None:
        """Get current version for an event type."""
        return self._current_versions.get(event_type)

    def migrate_data(
        self,
        event_type: str,
        data: dict[str, Any],
        from_version: SchemaVersion,
        to_version: SchemaVersion | None = None,
    ) -> dict[str, Any]:
        """Migrate data from one version to another."""
        if to_version is None:
            to_version = self._current_versions.get(event_type)
        if to_version is None:
            raise ValueError(f"No current version for {event_type}")

        if from_version.version_string == to_version.version_string:
            return data

        migrations = self._migrations.get(event_type, [])
        current_version = from_version
        result = data.copy()

        while current_version.version_string != to_version.version_string:
            migration_found = False
            for migration in migrations:
                if (
                    migration.from_version.version_string
                    == current_version.version_string
                ):
                    result = migration.migrate(result)
                    current_version = migration.to_version
                    migration_found = True
                    break

            if not migration_found:
                raise ValueError(
                    f"No migration path from {from_version} to {to_version}"
                )

        return result


class EventSerializer:
    """Serializes events to various formats."""

    def __init__(self, schema_registry: SchemaRegistry | None = None):
        self.schema_registry = schema_registry or SchemaRegistry()

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, str | int | float | bool | type(None)):
            return value
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Option):
            # Serialize Option types
            if value.is_some():
                return {
                    "_option_type": "Some",
                    "_option_value": self._serialize_value(cast("Some", value).value),
                }
            return {"_option_type": "Empty"}
        if isinstance(value, Either):
            # Serialize Either types
            if value.is_right():
                return {
                    "_either_type": "Right",
                    "_either_value": self._serialize_value(cast("Right", value).value),
                }
            return {
                "_either_type": "Left",
                "_either_value": self._serialize_value(cast("Left", value).value),
            }
        if isinstance(value, BaseModel):
            return value.model_dump()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self._serialize_value(v) for v in value]
        return str(value)

    def serialize_to_json(
        self,
        event: Event,
        include_schema_version: bool = True,
        compress: bool = False,
    ) -> Either[SerializationError, bytes]:
        """Serialize event to JSON format."""
        try:
            data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "aggregate_id": event.aggregate_id,
                "version": event.version,
                "metadata": event.metadata,
                "data": self._serialize_value(event.data),
            }

            if include_schema_version:
                current_version = self.schema_registry.get_current_version(
                    event.event_type
                )
                if current_version:
                    data["schema_version"] = current_version.version_string

            json_str = json.dumps(data, separators=(",", ":"))
            json_bytes = json_str.encode("utf-8")

            if compress:
                json_bytes = gzip.compress(json_bytes)

            return Right(json_bytes)

        except Exception as e:
            return Left(SerializationError(f"JSON serialization failed: {e}"))

    def serialize_to_msgpack(
        self, event: Event, compress: bool = False
    ) -> Either[SerializationError, bytes]:
        """Serialize event to MessagePack format."""
        try:
            data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "aggregate_id": event.aggregate_id,
                "version": event.version,
                "metadata": event.metadata,
                "data": self._serialize_value(event.data),
            }

            current_version = self.schema_registry.get_current_version(event.event_type)
            if current_version:
                data["schema_version"] = current_version.version_string

            msgpack_bytes = msgpack.packb(data, use_bin_type=True)

            if compress:
                msgpack_bytes = gzip.compress(msgpack_bytes)

            return Right(msgpack_bytes)

        except Exception as e:
            return Left(SerializationError(f"MessagePack serialization failed: {e}"))


class EventDeserializer:
    """Deserializes events from various formats."""

    def __init__(self, schema_registry: SchemaRegistry | None = None):
        self.schema_registry = schema_registry or SchemaRegistry()

    def _deserialize_value(self, value: Any, type_hint: type | None = None) -> Any:
        """Deserialize a single value."""
        # Handle special functional types
        if isinstance(value, dict):
            # Handle Option types
            if "_option_type" in value:
                if value["_option_type"] == "Some":
                    inner_value = self._deserialize_value(value["_option_value"])
                    return Some(inner_value)
                return Empty()

            # Handle Either types
            if "_either_type" in value:
                inner_value = self._deserialize_value(value["_either_value"])
                if value["_either_type"] == "Right":
                    return Right(inner_value)
                return Left(inner_value)

        if type_hint and hasattr(type_hint, "__origin__"):
            # Handle generic types
            origin = type_hint.__origin__
            if origin in (list, tuple):
                item_type = type_hint.__args__[0] if type_hint.__args__ else None
                return [self._deserialize_value(v, item_type) for v in value]
            if origin is dict:
                key_type, value_type = (
                    type_hint.__args__ if len(type_hint.__args__) == 2 else (None, None)
                )
                return {
                    k: self._deserialize_value(v, value_type) for k, v in value.items()
                }

        if isinstance(value, str):
            # Try to parse as datetime
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
            # Try to parse as Decimal
            try:
                return Decimal(value)
            except (ValueError, TypeError):
                pass

        return value

    def deserialize_from_json(
        self,
        data: bytes,
        event_class: type[T],
        compressed: bool = False,
    ) -> Either[DeserializationError, T]:
        """Deserialize event from JSON format."""
        try:
            if compressed:
                data = gzip.decompress(data)

            json_data = json.loads(data.decode("utf-8"))

            # Handle schema migration if needed
            if "schema_version" in json_data:
                from_version = SchemaVersion.from_string(json_data["schema_version"])
                current_version = self.schema_registry.get_current_version(
                    json_data["event_type"]
                )
                if (
                    current_version
                    and from_version.version_string != current_version.version_string
                ):
                    json_data = self.schema_registry.migrate_data(
                        json_data["event_type"],
                        json_data,
                        from_version,
                        current_version,
                    )

            # Create event instance
            event = event_class(
                event_id=json_data["event_id"],
                event_type=json_data["event_type"],
                timestamp=datetime.fromisoformat(json_data["timestamp"]),
                aggregate_id=json_data["aggregate_id"],
                version=json_data["version"],
                metadata=json_data.get("metadata", {}),
                data=json_data["data"],
            )

            return Right(event)

        except Exception as e:
            return Left(DeserializationError(f"JSON deserialization failed: {e}"))

    def deserialize_from_msgpack(
        self,
        data: bytes,
        event_class: type[T],
        compressed: bool = False,
    ) -> Either[DeserializationError, T]:
        """Deserialize event from MessagePack format."""
        try:
            if compressed:
                data = gzip.decompress(data)

            msgpack_data = msgpack.unpackb(data, raw=False)

            # Handle schema migration if needed
            if "schema_version" in msgpack_data:
                from_version = SchemaVersion.from_string(msgpack_data["schema_version"])
                current_version = self.schema_registry.get_current_version(
                    msgpack_data["event_type"]
                )
                if (
                    current_version
                    and from_version.version_string != current_version.version_string
                ):
                    msgpack_data = self.schema_registry.migrate_data(
                        msgpack_data["event_type"],
                        msgpack_data,
                        from_version,
                        current_version,
                    )

            # Create event instance
            event = event_class(
                event_id=msgpack_data["event_id"],
                event_type=msgpack_data["event_type"],
                timestamp=datetime.fromisoformat(msgpack_data["timestamp"]),
                aggregate_id=msgpack_data["aggregate_id"],
                version=msgpack_data["version"],
                metadata=msgpack_data.get("metadata", {}),
                data=msgpack_data["data"],
            )

            return Right(event)

        except Exception as e:
            return Left(
                DeserializationError(f"MessagePack deserialization failed: {e}")
            )


# Example migration
class OrderEventV1ToV2Migration(SchemaMigration):
    """Migration from OrderEvent v1 to v2."""

    @property
    def from_version(self) -> SchemaVersion:
        return SchemaVersion(1, 0, 0)

    @property
    def to_version(self) -> SchemaVersion:
        return SchemaVersion(2, 0, 0)

    def migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate OrderEvent from v1 to v2.

        Changes:
        - Added 'exchange' field
        - Renamed 'amount' to 'quantity'
        """
        migrated = data.copy()

        # Add exchange field with default
        if "data" in migrated and "exchange" not in migrated["data"]:
            migrated["data"]["exchange"] = "coinbase"

        # Rename amount to quantity
        if "data" in migrated and "amount" in migrated["data"]:
            migrated["data"]["quantity"] = migrated["data"].pop("amount")

        return migrated


# Factory functions
def create_event_serializer(
    with_schema_registry: bool = True,
) -> EventSerializer:
    """Create an event serializer."""
    registry = SchemaRegistry() if with_schema_registry else None
    return EventSerializer(registry)


def create_event_deserializer(
    with_schema_registry: bool = True,
) -> EventDeserializer:
    """Create an event deserializer."""
    registry = SchemaRegistry() if with_schema_registry else None
    return EventDeserializer(registry)
