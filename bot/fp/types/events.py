"""Event sourcing types for the trading system.

This module defines immutable event types that capture all state changes
in the trading system, enabling event sourcing and audit trails.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar
from uuid import UUID, uuid4

from bot.fp.core.option import Empty, Option, Some
from bot.fp.types.result import Failure, Result, Success
from bot.types import OrderType, TradeSide


@dataclass(frozen=True)
class EventMetadata:
    """Metadata common to all events."""

    version: str = "1.0.0"
    source: str = "trading-bot"
    correlation_id: UUID | None = None
    causation_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "version": self.version,
            "source": self.source,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventMetadata":
        """Create metadata from dictionary."""
        return cls(
            version=data["version"],
            source=data["source"],
            correlation_id=(
                UUID(data["correlation_id"]) if data.get("correlation_id") else None
            ),
            causation_id=(
                UUID(data["causation_id"]) if data.get("causation_id") else None
            ),
        )


@dataclass(frozen=True)
class TradingEvent:
    """Base class for all trading events."""

    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: EventMetadata = field(default_factory=EventMetadata)

    # Event type for polymorphic deserialization
    event_type: ClassVar[str] = "TradingEvent"

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradingEvent":
        """Create event from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
        )


@dataclass(frozen=True)
class MarketDataReceived(TradingEvent):
    """Event for market data snapshots."""

    event_type: ClassVar[str] = "MarketDataReceived"

    symbol: str = ""
    price: Decimal = Decimal(0)
    volume: Decimal = Decimal(0)
    bid: Decimal = Decimal(0)
    ask: Decimal = Decimal(0)
    high_24h: Decimal = Decimal(0)
    low_24h: Decimal = Decimal(0)
    open_24h: Decimal = Decimal(0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "symbol": self.symbol,
                "price": str(self.price),
                "volume": str(self.volume),
                "bid": str(self.bid),
                "ask": str(self.ask),
                "high_24h": str(self.high_24h),
                "low_24h": str(self.low_24h),
                "open_24h": str(self.open_24h),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketDataReceived":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            symbol=data["symbol"],
            price=Decimal(data["price"]),
            volume=Decimal(data["volume"]),
            bid=Decimal(data["bid"]),
            ask=Decimal(data["ask"]),
            high_24h=Decimal(data["high_24h"]),
            low_24h=Decimal(data["low_24h"]),
            open_24h=Decimal(data["open_24h"]),
        )


@dataclass(frozen=True)
class OrderPlaced(TradingEvent):
    """Event for order placement."""

    event_type: ClassVar[str] = "OrderPlaced"

    order_id: str = ""
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    order_type: OrderType = OrderType.MARKET
    size: Decimal = Decimal(0)
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "GTC"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side.value,
                "order_type": self.order_type.value,
                "size": str(self.size),
                "price": str(self.price) if self.price else None,
                "stop_price": str(self.stop_price) if self.stop_price else None,
                "time_in_force": self.time_in_force,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderPlaced":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=TradeSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            size=Decimal(data["size"]),
            price=Decimal(data["price"]) if data.get("price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            time_in_force=data["time_in_force"],
        )


@dataclass(frozen=True)
class OrderFilled(TradingEvent):
    """Event for order execution."""

    event_type: ClassVar[str] = "OrderFilled"

    order_id: str = ""
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    fill_price: Decimal = Decimal(0)
    fill_size: Decimal = Decimal(0)
    fees: Decimal = Decimal(0)
    is_partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side.value,
                "fill_price": str(self.fill_price),
                "fill_size": str(self.fill_size),
                "fees": str(self.fees),
                "is_partial": self.is_partial,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderFilled":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=TradeSide(data["side"]),
            fill_price=Decimal(data["fill_price"]),
            fill_size=Decimal(data["fill_size"]),
            fees=Decimal(data["fees"]),
            is_partial=data["is_partial"],
        )


@dataclass(frozen=True)
class OrderCancelled(TradingEvent):
    """Event for order cancellation."""

    event_type: ClassVar[str] = "OrderCancelled"

    order_id: str = ""
    symbol: str = ""
    reason: str = ""
    remaining_size: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "reason": self.reason,
                "remaining_size": (
                    str(self.remaining_size) if self.remaining_size else None
                ),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderCancelled":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            order_id=data["order_id"],
            symbol=data["symbol"],
            reason=data["reason"],
            remaining_size=(
                Decimal(data["remaining_size"]) if data.get("remaining_size") else None
            ),
        )


@dataclass(frozen=True)
class PositionOpened(TradingEvent):
    """Event for position opening."""

    event_type: ClassVar[str] = "PositionOpened"

    position_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    size: Decimal = Decimal(0)
    entry_price: Decimal = Decimal(0)
    leverage: int = 1
    initial_margin: Decimal = Decimal(0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "position_id": str(self.position_id),
                "symbol": self.symbol,
                "side": self.side.value,
                "size": str(self.size),
                "entry_price": str(self.entry_price),
                "leverage": self.leverage,
                "initial_margin": str(self.initial_margin),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionOpened":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            position_id=UUID(data["position_id"]),
            symbol=data["symbol"],
            side=TradeSide(data["side"]),
            size=Decimal(data["size"]),
            entry_price=Decimal(data["entry_price"]),
            leverage=data["leverage"],
            initial_margin=Decimal(data["initial_margin"]),
        )


@dataclass(frozen=True)
class PositionClosed(TradingEvent):
    """Event for position closing."""

    event_type: ClassVar[str] = "PositionClosed"

    position_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    entry_price: Decimal = Decimal(0)
    exit_price: Decimal = Decimal(0)
    size: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    fees: Decimal = Decimal(0)
    duration_seconds: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "position_id": str(self.position_id),
                "symbol": self.symbol,
                "side": self.side.value,
                "entry_price": str(self.entry_price),
                "exit_price": str(self.exit_price),
                "size": str(self.size),
                "realized_pnl": str(self.realized_pnl),
                "fees": str(self.fees),
                "duration_seconds": self.duration_seconds,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionClosed":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            position_id=UUID(data["position_id"]),
            symbol=data["symbol"],
            side=TradeSide(data["side"]),
            entry_price=Decimal(data["entry_price"]),
            exit_price=Decimal(data["exit_price"]),
            size=Decimal(data["size"]),
            realized_pnl=Decimal(data["realized_pnl"]),
            fees=Decimal(data["fees"]),
            duration_seconds=data["duration_seconds"],
        )


@dataclass(frozen=True)
class StrategySignal(TradingEvent):
    """Event for strategy signals."""

    event_type: ClassVar[str] = "StrategySignal"

    signal_type: str = "HOLD"  # "LONG", "SHORT", "CLOSE", "HOLD"
    confidence: float = 0.0  # 0.0 to 1.0
    symbol: str = ""
    current_price: Decimal = Decimal(0)
    indicators: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "signal_type": self.signal_type,
                "confidence": self.confidence,
                "symbol": self.symbol,
                "current_price": str(self.current_price),
                "indicators": self.indicators,
                "reasoning": self.reasoning,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategySignal":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            signal_type=data["signal_type"],
            confidence=data["confidence"],
            symbol=data["symbol"],
            current_price=Decimal(data["current_price"]),
            indicators=data["indicators"],
            reasoning=data.get("reasoning"),
        )


# Enhanced audit trail and notification event types


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    LOG = "log"


class SystemComponent(Enum):
    """System components for audit trails."""

    STRATEGY_ENGINE = "strategy_engine"
    RISK_MANAGER = "risk_manager"
    ORDER_MANAGER = "order_manager"
    POSITION_MANAGER = "position_manager"
    MARKET_DATA_FEED = "market_data_feed"
    EXCHANGE_CONNECTOR = "exchange_connector"
    CONFIGURATION_MANAGER = "configuration_manager"
    NOTIFICATION_SYSTEM = "notification_system"
    WEB_DASHBOARD = "web_dashboard"


@dataclass(frozen=True)
class AlertTriggered(TradingEvent):
    """Event for triggered alerts with functional validation."""

    event_type: ClassVar[str] = "AlertTriggered"

    alert_id: str = ""
    alert_name: str = ""
    level: AlertLevel = AlertLevel.INFO
    component: SystemComponent = SystemComponent.STRATEGY_ENGINE
    message: str = ""
    threshold_value: Option[Decimal] = field(default_factory=Empty)
    actual_value: Option[Decimal] = field(default_factory=Empty)
    symbol: Option[str] = field(default_factory=Empty)
    triggered_conditions: dict[str, Any] = field(default_factory=dict)
    auto_acknowledge: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "alert_id": self.alert_id,
                "alert_name": self.alert_name,
                "level": self.level.value,
                "component": self.component.value,
                "message": self.message,
                "threshold_value": str(self.threshold_value.get_or_else(Decimal(0))),
                "actual_value": str(self.actual_value.get_or_else(Decimal(0))),
                "symbol": self.symbol.get_or_else(""),
                "triggered_conditions": self.triggered_conditions,
                "auto_acknowledge": self.auto_acknowledge,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlertTriggered":
        """Create from dictionary."""
        threshold_str = data.get("threshold_value", "0")
        actual_str = data.get("actual_value", "0")
        symbol_str = data.get("symbol", "")

        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            alert_id=data["alert_id"],
            alert_name=data["alert_name"],
            level=AlertLevel(data["level"]),
            component=SystemComponent(data["component"]),
            message=data["message"],
            threshold_value=(
                Some(Decimal(threshold_str)) if threshold_str != "0" else Empty()
            ),
            actual_value=Some(Decimal(actual_str)) if actual_str != "0" else Empty(),
            symbol=Some(symbol_str) if symbol_str else Empty(),
            triggered_conditions=data.get("triggered_conditions", {}),
            auto_acknowledge=data.get("auto_acknowledge", False),
        )


@dataclass(frozen=True)
class NotificationSent(TradingEvent):
    """Event for sent notifications."""

    event_type: ClassVar[str] = "NotificationSent"

    notification_id: str = ""
    channel: NotificationChannel = NotificationChannel.LOG
    recipient: str = ""
    subject: str = ""
    content: str = ""
    delivery_status: str = "pending"  # "pending", "sent", "failed", "delivered"
    retry_count: int = 0
    alert_reference: Option[str] = field(default_factory=Empty)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "notification_id": self.notification_id,
                "channel": self.channel.value,
                "recipient": self.recipient,
                "subject": self.subject,
                "content": self.content,
                "delivery_status": self.delivery_status,
                "retry_count": self.retry_count,
                "alert_reference": self.alert_reference.get_or_else(""),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NotificationSent":
        """Create from dictionary."""
        alert_ref = data.get("alert_reference", "")

        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            notification_id=data["notification_id"],
            channel=NotificationChannel(data["channel"]),
            recipient=data["recipient"],
            subject=data["subject"],
            content=data["content"],
            delivery_status=data.get("delivery_status", "pending"),
            retry_count=data.get("retry_count", 0),
            alert_reference=Some(alert_ref) if alert_ref else Empty(),
        )


@dataclass(frozen=True)
class ConfigurationChanged(TradingEvent):
    """Event for configuration changes with audit trail."""

    event_type: ClassVar[str] = "ConfigurationChanged"

    config_section: str = ""
    config_key: str = ""
    old_value: Any = None
    new_value: Any = None
    changed_by: str = "system"
    change_reason: str = ""
    validation_result: str = "success"  # "success", "warning", "error"
    affected_components: list[SystemComponent] = field(default_factory=list)
    requires_restart: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "config_section": self.config_section,
                "config_key": self.config_key,
                "old_value": (
                    str(self.old_value) if self.old_value is not None else None
                ),
                "new_value": (
                    str(self.new_value) if self.new_value is not None else None
                ),
                "changed_by": self.changed_by,
                "change_reason": self.change_reason,
                "validation_result": self.validation_result,
                "affected_components": [
                    comp.value for comp in self.affected_components
                ],
                "requires_restart": self.requires_restart,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigurationChanged":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            config_section=data["config_section"],
            config_key=data["config_key"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            changed_by=data.get("changed_by", "system"),
            change_reason=data.get("change_reason", ""),
            validation_result=data.get("validation_result", "success"),
            affected_components=[
                SystemComponent(comp) for comp in data.get("affected_components", [])
            ],
            requires_restart=data.get("requires_restart", False),
        )


@dataclass(frozen=True)
class SystemPerformanceMetric(TradingEvent):
    """Event for system performance metrics."""

    event_type: ClassVar[str] = "SystemPerformanceMetric"

    metric_name: str = ""
    metric_value: Decimal = Decimal(0)
    metric_unit: str = ""
    component: SystemComponent = SystemComponent.STRATEGY_ENGINE
    threshold_breached: bool = False
    metric_tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "metric_name": self.metric_name,
                "metric_value": str(self.metric_value),
                "metric_unit": self.metric_unit,
                "component": self.component.value,
                "threshold_breached": self.threshold_breached,
                "metric_tags": self.metric_tags,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemPerformanceMetric":
        """Create from dictionary."""
        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            metric_name=data["metric_name"],
            metric_value=Decimal(data["metric_value"]),
            metric_unit=data.get("metric_unit", ""),
            component=SystemComponent(data["component"]),
            threshold_breached=data.get("threshold_breached", False),
            metric_tags=data.get("metric_tags", {}),
        )


@dataclass(frozen=True)
class ErrorOccurred(TradingEvent):
    """Event for system errors with context."""

    event_type: ClassVar[str] = "ErrorOccurred"

    error_type: str = ""
    error_message: str = ""
    error_code: Option[str] = field(default_factory=Empty)
    component: SystemComponent = SystemComponent.STRATEGY_ENGINE
    stack_trace: Option[str] = field(default_factory=Empty)
    user_impact: str = "none"  # "none", "degraded", "critical"
    recovery_action: Option[str] = field(default_factory=Empty)
    error_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "error_type": self.error_type,
                "error_message": self.error_message,
                "error_code": self.error_code.get_or_else(""),
                "component": self.component.value,
                "stack_trace": self.stack_trace.get_or_else(""),
                "user_impact": self.user_impact,
                "recovery_action": self.recovery_action.get_or_else(""),
                "error_context": self.error_context,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorOccurred":
        """Create from dictionary."""
        error_code_str = data.get("error_code", "")
        stack_trace_str = data.get("stack_trace", "")
        recovery_str = data.get("recovery_action", "")

        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            error_type=data["error_type"],
            error_message=data["error_message"],
            error_code=Some(error_code_str) if error_code_str else Empty(),
            component=SystemComponent(data["component"]),
            stack_trace=Some(stack_trace_str) if stack_trace_str else Empty(),
            user_impact=data.get("user_impact", "none"),
            recovery_action=Some(recovery_str) if recovery_str else Empty(),
            error_context=data.get("error_context", {}),
        )


@dataclass(frozen=True)
class AuditTrailEntry(TradingEvent):
    """Event for comprehensive audit trails."""

    event_type: ClassVar[str] = "AuditTrailEntry"

    action: str = ""
    entity_type: str = ""
    entity_id: str = ""
    user_id: Option[str] = field(default_factory=Empty)
    session_id: Option[str] = field(default_factory=Empty)
    ip_address: Option[str] = field(default_factory=Empty)
    user_agent: Option[str] = field(default_factory=Empty)
    before_state: dict[str, Any] = field(default_factory=dict)
    after_state: dict[str, Any] = field(default_factory=dict)
    change_summary: str = ""
    compliance_relevant: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "action": self.action,
                "entity_type": self.entity_type,
                "entity_id": self.entity_id,
                "user_id": self.user_id.get_or_else(""),
                "session_id": self.session_id.get_or_else(""),
                "ip_address": self.ip_address.get_or_else(""),
                "user_agent": self.user_agent.get_or_else(""),
                "before_state": self.before_state,
                "after_state": self.after_state,
                "change_summary": self.change_summary,
                "compliance_relevant": self.compliance_relevant,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditTrailEntry":
        """Create from dictionary."""
        user_id_str = data.get("user_id", "")
        session_id_str = data.get("session_id", "")
        ip_str = data.get("ip_address", "")
        user_agent_str = data.get("user_agent", "")

        return cls(
            event_id=UUID(data["event_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=EventMetadata.from_dict(data["metadata"]),
            action=data["action"],
            entity_type=data["entity_type"],
            entity_id=data["entity_id"],
            user_id=Some(user_id_str) if user_id_str else Empty(),
            session_id=Some(session_id_str) if session_id_str else Empty(),
            ip_address=Some(ip_str) if ip_str else Empty(),
            user_agent=Some(user_agent_str) if user_agent_str else Empty(),
            before_state=data.get("before_state", {}),
            after_state=data.get("after_state", {}),
            change_summary=data.get("change_summary", ""),
            compliance_relevant=data.get("compliance_relevant", False),
        )


# Enhanced event type registry for deserialization
EVENT_REGISTRY: dict[str, type[TradingEvent]] = {
    "TradingEvent": TradingEvent,
    "MarketDataReceived": MarketDataReceived,
    "OrderPlaced": OrderPlaced,
    "OrderFilled": OrderFilled,
    "OrderCancelled": OrderCancelled,
    "PositionOpened": PositionOpened,
    "PositionClosed": PositionClosed,
    "StrategySignal": StrategySignal,
    # Enhanced audit and notification events
    "AlertTriggered": AlertTriggered,
    "NotificationSent": NotificationSent,
    "ConfigurationChanged": ConfigurationChanged,
    "SystemPerformanceMetric": SystemPerformanceMetric,
    "ErrorOccurred": ErrorOccurred,
    "AuditTrailEntry": AuditTrailEntry,
}


def deserialize_event(data: dict[str, Any]) -> TradingEvent:
    """Deserialize an event from dictionary based on event_type."""
    event_type = data.get("event_type")
    if not event_type:
        raise ValueError("Missing event_type in event data")

    event_class = EVENT_REGISTRY.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.from_dict(data)


# Functional event creation utilities
def create_alert_event(
    alert_name: str,
    level: AlertLevel,
    component: SystemComponent,
    message: str,
    threshold: Option[Decimal] = Empty(),
    actual: Option[Decimal] = Empty(),
    symbol: Option[str] = Empty(),
) -> Result[AlertTriggered, str]:
    """Create an alert event with validation."""
    if not alert_name.strip():
        return Failure("Alert name cannot be empty")
    if not message.strip():
        return Failure("Alert message cannot be empty")

    try:
        event = AlertTriggered(
            alert_id=str(uuid4()),
            alert_name=alert_name,
            level=level,
            component=component,
            message=message,
            threshold_value=threshold,
            actual_value=actual,
            symbol=symbol,
        )
        return Success(event)
    except Exception as e:
        return Failure(f"Failed to create alert event: {e}")


def create_audit_event(
    action: str,
    entity_type: str,
    entity_id: str,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    user_id: Option[str] = Empty(),
) -> Result[AuditTrailEntry, str]:
    """Create an audit trail event with validation."""
    if not action.strip():
        return Failure("Action cannot be empty")
    if not entity_type.strip():
        return Failure("Entity type cannot be empty")
    if not entity_id.strip():
        return Failure("Entity ID cannot be empty")

    try:
        # Generate change summary
        changes = []
        all_keys = set(before_state.keys()) | set(after_state.keys())
        for key in all_keys:
            old_val = before_state.get(key)
            new_val = after_state.get(key)
            if old_val != new_val:
                changes.append(f"{key}: {old_val} -> {new_val}")

        change_summary = "; ".join(changes) if changes else "No changes detected"

        event = AuditTrailEntry(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            before_state=before_state,
            after_state=after_state,
            change_summary=change_summary,
        )
        return Success(event)
    except Exception as e:
        return Failure(f"Failed to create audit event: {e}")
