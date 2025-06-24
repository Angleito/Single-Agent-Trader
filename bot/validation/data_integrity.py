"""
Functional data integrity validation system.

This module provides pure functional data integrity checks with:
- Immutable validation rules
- Composable integrity constraints
- Pure validation algorithms
- Monadic error handling
- Cross-field validation
- Temporal data consistency checks
- Business rule validation
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import reduce, partial
from dataclasses import dataclass, field
from enum import Enum

from bot.fp.core.functional_validation import (
    FieldError, SchemaError, ValidationChainError, ValidatorError,
    FPResult, FPSuccess, FPFailure, ValidationPipeline
)
from bot.fp.types.base import Money, Percentage, Symbol
from bot.trading_types import TradeAction, Position

# Type variables
T = TypeVar('T')
U = TypeVar('U')


class IntegrityLevel(Enum):
    """Data integrity validation levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class IntegrityViolationType(Enum):
    """Types of integrity violations."""
    INCONSISTENT_STATE = "inconsistent_state"
    TEMPORAL_VIOLATION = "temporal_violation"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    CROSS_FIELD_VIOLATION = "cross_field_violation"
    DATA_CORRUPTION = "data_corruption"
    CONSTRAINT_VIOLATION = "constraint_violation"


@dataclass(frozen=True)
class IntegrityRule:
    """Immutable integrity rule definition."""
    
    name: str
    description: str
    validator: Callable[[Any], bool]
    violation_type: IntegrityViolationType
    severity: str = "error"  # "error", "warning", "info"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, data: Any) -> FPResult[Any, FieldError]:
        """Apply this integrity rule to data."""
        try:
            if self.validator(data):
                return FPSuccess(data)
            else:
                return FPFailure(FieldError(
                    field=self.name,
                    message=self.description,
                    validation_rule="integrity",
                    context={
                        "violation_type": self.violation_type.value,
                        "severity": self.severity,
                        **self.context
                    }
                ))
        except Exception as e:
            return FPFailure(FieldError(
                field=self.name,
                message=f"Integrity rule execution failed: {e}",
                validation_rule="integrity_error",
                context={
                    "violation_type": self.violation_type.value,
                    "severity": "error",
                    "error": str(e)
                }
            ))


@dataclass(frozen=True)
class CrossFieldRule:
    """Immutable cross-field integrity rule."""
    
    name: str
    description: str
    fields: List[str]
    validator: Callable[[Dict[str, Any]], bool]
    violation_type: IntegrityViolationType
    severity: str = "error"
    
    def validate(self, data: Dict[str, Any]) -> FPResult[Dict[str, Any], FieldError]:
        """Apply this cross-field rule to data."""
        try:
            # Check if all required fields are present
            missing_fields = [f for f in self.fields if f not in data]
            if missing_fields:
                return FPFailure(FieldError(
                    field="cross_field",
                    message=f"Missing required fields for {self.name}: {missing_fields}",
                    validation_rule="missing_fields",
                    context={
                        "rule_name": self.name,
                        "missing_fields": missing_fields,
                        "required_fields": self.fields
                    }
                ))
            
            if self.validator(data):
                return FPSuccess(data)
            else:
                return FPFailure(FieldError(
                    field="cross_field",
                    message=self.description,
                    validation_rule="cross_field_integrity",
                    context={
                        "rule_name": self.name,
                        "violation_type": self.violation_type.value,
                        "severity": self.severity,
                        "fields": self.fields
                    }
                ))
        except Exception as e:
            return FPFailure(FieldError(
                field="cross_field",
                message=f"Cross-field rule execution failed: {e}",
                validation_rule="cross_field_error",
                context={
                    "rule_name": self.name,
                    "error": str(e)
                }
            ))


# Trading-Specific Integrity Rules

def create_balance_consistency_rule() -> IntegrityRule:
    """Create rule to validate balance consistency."""
    def validate_balance(data: Dict[str, Any]) -> bool:
        if "total_balance" not in data or "available_balance" not in data:
            return True  # Skip if fields not present
        
        total = Decimal(str(data["total_balance"]))
        available = Decimal(str(data["available_balance"]))
        
        # Available balance should not exceed total balance
        return available <= total
    
    return IntegrityRule(
        name="balance_consistency",
        description="Available balance must not exceed total balance",
        validator=validate_balance,
        violation_type=IntegrityViolationType.INCONSISTENT_STATE
    )


def create_position_size_rule() -> IntegrityRule:
    """Create rule to validate position size consistency."""
    def validate_position_size(data: Dict[str, Any]) -> bool:
        if "position_size" not in data:
            return True
        
        size = float(data["position_size"])
        # Position size should be non-negative
        return size >= 0
    
    return IntegrityRule(
        name="position_size_positive",
        description="Position size must be non-negative",
        validator=validate_position_size,
        violation_type=IntegrityViolationType.BUSINESS_RULE_VIOLATION
    )


def create_price_sanity_rule() -> IntegrityRule:
    """Create rule to validate price sanity."""
    def validate_price(data: Dict[str, Any]) -> bool:
        price_fields = ["price", "entry_price", "current_price", "bid", "ask"]
        prices = []
        
        for field in price_fields:
            if field in data and data[field] is not None:
                try:
                    price = float(data[field])
                    if price <= 0:
                        return False  # Prices must be positive
                    prices.append(price)
                except (ValueError, TypeError):
                    return False  # Invalid price format
        
        # Additional sanity checks
        if "bid" in data and "ask" in data and data["bid"] and data["ask"]:
            bid = float(data["bid"])
            ask = float(data["ask"])
            if bid >= ask:
                return False  # Bid must be less than ask
        
        return True
    
    return IntegrityRule(
        name="price_sanity",
        description="Prices must be positive and bid < ask",
        validator=validate_price,
        violation_type=IntegrityViolationType.DATA_CORRUPTION
    )


def create_timestamp_consistency_rule() -> IntegrityRule:
    """Create rule to validate timestamp consistency."""
    def validate_timestamps(data: Dict[str, Any]) -> bool:
        timestamp_fields = ["timestamp", "created_at", "updated_at", "entry_time", "exit_time"]
        timestamps = []
        
        for field in timestamp_fields:
            if field in data and data[field] is not None:
                try:
                    if isinstance(data[field], datetime):
                        ts = data[field].timestamp()
                    else:
                        ts = float(data[field])
                    
                    # Check if timestamp is reasonable (within last 5 years and not in future)
                    now = datetime.now().timestamp()
                    five_years_ago = now - (5 * 365 * 24 * 60 * 60)
                    one_day_future = now + (24 * 60 * 60)
                    
                    if not (five_years_ago <= ts <= one_day_future):
                        return False
                    
                    timestamps.append((field, ts))
                except (ValueError, TypeError):
                    return False
        
        # Check logical ordering if multiple timestamps present
        if len(timestamps) >= 2:
            for i, (field1, ts1) in enumerate(timestamps):
                for field2, ts2 in timestamps[i+1:]:
                    # created_at should be before updated_at
                    if field1 == "created_at" and field2 == "updated_at" and ts1 > ts2:
                        return False
                    # entry_time should be before exit_time
                    if field1 == "entry_time" and field2 == "exit_time" and ts1 > ts2:
                        return False
        
        return True
    
    return IntegrityRule(
        name="timestamp_consistency",
        description="Timestamps must be reasonable and logically ordered",
        validator=validate_timestamps,
        violation_type=IntegrityViolationType.TEMPORAL_VIOLATION
    )


def create_trade_action_risk_rule() -> CrossFieldRule:
    """Create cross-field rule for trade action risk parameters."""
    def validate_risk_params(data: Dict[str, Any]) -> bool:
        required_fields = ["take_profit_pct", "stop_loss_pct", "size_pct"]
        
        # All fields must be present for this rule
        for field in required_fields:
            if field not in data:
                return True  # Skip if not all fields present
        
        tp_pct = float(data["take_profit_pct"])
        sl_pct = float(data["stop_loss_pct"])
        size_pct = float(data["size_pct"])
        
        # Risk/reward ratio should be reasonable (TP >= 0.5 * SL)
        if tp_pct < 0.5 * sl_pct:
            return False
        
        # Size should be reasonable relative to risk
        if size_pct > 50 and sl_pct > 5:  # Large position with high risk
            return False
        
        return True
    
    return CrossFieldRule(
        name="trade_action_risk",
        description="Trade action risk parameters must be reasonable",
        fields=["take_profit_pct", "stop_loss_pct", "size_pct"],
        validator=validate_risk_params,
        violation_type=IntegrityViolationType.BUSINESS_RULE_VIOLATION
    )


def create_position_balance_rule() -> CrossFieldRule:
    """Create cross-field rule for position-balance consistency."""
    def validate_position_balance(data: Dict[str, Any]) -> bool:
        required_fields = ["position_value", "margin_used", "available_balance"]
        
        for field in required_fields:
            if field not in data:
                return True  # Skip if not all fields present
        
        position_value = Decimal(str(data["position_value"]))
        margin_used = Decimal(str(data["margin_used"]))
        available_balance = Decimal(str(data["available_balance"]))
        
        # Margin used should not exceed position value (for reasonable leverage)
        if margin_used > position_value:
            return False
        
        # Available balance should be sufficient for margin
        if available_balance < margin_used:
            return False
        
        return True
    
    return CrossFieldRule(
        name="position_balance_consistency",
        description="Position value, margin, and balance must be consistent",
        fields=["position_value", "margin_used", "available_balance"],
        validator=validate_position_balance,
        violation_type=IntegrityViolationType.INCONSISTENT_STATE
    )


# Functional Integrity Validator

class FunctionalIntegrityValidator:
    """Pure functional data integrity validator."""
    
    def __init__(self, level: IntegrityLevel = IntegrityLevel.BASIC):
        self.level = level
        self.rules: List[IntegrityRule] = []
        self.cross_field_rules: List[CrossFieldRule] = []
        
        # Load default rules based on level
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """Load default integrity rules based on validation level."""
        # Basic level rules
        basic_rules = [
            create_position_size_rule(),
            create_price_sanity_rule()
        ]
        
        # Strict level rules
        strict_rules = basic_rules + [
            create_balance_consistency_rule(),
            create_timestamp_consistency_rule()
        ]
        
        # Paranoid level rules (all rules)
        paranoid_rules = strict_rules
        
        # Cross-field rules
        cross_field_rules = [
            create_trade_action_risk_rule(),
            create_position_balance_rule()
        ]
        
        if self.level == IntegrityLevel.BASIC:
            self.rules = basic_rules
        elif self.level == IntegrityLevel.STRICT:
            self.rules = strict_rules
            self.cross_field_rules = cross_field_rules
        elif self.level == IntegrityLevel.PARANOID:
            self.rules = paranoid_rules
            self.cross_field_rules = cross_field_rules
    
    def add_rule(self, rule: IntegrityRule) -> "FunctionalIntegrityValidator":
        """Add a custom integrity rule."""
        new_rules = self.rules + [rule]
        validator = FunctionalIntegrityValidator(self.level)
        validator.rules = new_rules
        validator.cross_field_rules = self.cross_field_rules
        return validator
    
    def add_cross_field_rule(self, rule: CrossFieldRule) -> "FunctionalIntegrityValidator":
        """Add a custom cross-field integrity rule."""
        new_cross_rules = self.cross_field_rules + [rule]
        validator = FunctionalIntegrityValidator(self.level)
        validator.rules = self.rules
        validator.cross_field_rules = new_cross_rules
        return validator
    
    def validate_data(self, data: Dict[str, Any]) -> FPResult[Dict[str, Any], SchemaError]:
        """Validate data integrity using all rules."""
        errors = []
        
        # Apply single-field rules
        for rule in self.rules:
            result = rule.validate(data)
            if result.is_failure():
                errors.append(result.failure())
        
        # Apply cross-field rules
        for rule in self.cross_field_rules:
            result = rule.validate(data)
            if result.is_failure():
                errors.append(result.failure())
        
        if errors:
            return FPFailure(SchemaError(
                schema="data_integrity",
                errors=errors,
                severity="error" if any(e.context.get("severity", "error") == "error" for e in errors) else "warning"
            ))
        
        return FPSuccess(data)
    
    def validate_trade_action(self, action: TradeAction) -> FPResult[TradeAction, SchemaError]:
        """Validate trade action data integrity."""
        data = {
            "action": action.action,
            "size_pct": action.size_pct,
            "take_profit_pct": action.take_profit_pct,
            "stop_loss_pct": action.stop_loss_pct,
            "leverage": action.leverage,
            "rationale": action.rationale
        }
        
        result = self.validate_data(data)
        if result.is_success():
            return FPSuccess(action)
        else:
            return FPFailure(result.failure())
    
    def validate_position(self, position: Position) -> FPResult[Position, SchemaError]:
        """Validate position data integrity."""
        data = {
            "symbol": position.symbol,
            "side": position.side,
            "size": position.size,
            "entry_price": position.entry_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl
        }
        
        # Add timestamp if available
        if hasattr(position, 'timestamp'):
            data["timestamp"] = position.timestamp
        
        result = self.validate_data(data)
        if result.is_success():
            return FPSuccess(position)
        else:
            return FPFailure(result.failure())
    
    def validate_batch(self, data_list: List[Dict[str, Any]]) -> FPResult[List[Dict[str, Any]], List[tuple[int, SchemaError]]]:
        """Validate batch of data items."""
        results = []
        errors = []
        
        for i, data in enumerate(data_list):
            result = self.validate_data(data)
            if result.is_success():
                results.append(result.success())
            else:
                errors.append((i, result.failure()))
        
        if errors:
            return FPFailure(errors)
        return FPSuccess(results)


# Temporal Integrity Validators

def validate_temporal_sequence(
    data_points: List[Dict[str, Any]], 
    timestamp_field: str = "timestamp",
    tolerance_seconds: float = 1.0
) -> FPResult[List[Dict[str, Any]], FieldError]:
    """Validate temporal sequence of data points."""
    if len(data_points) < 2:
        return FPSuccess(data_points)
    
    prev_timestamp = None
    
    for i, point in enumerate(data_points):
        if timestamp_field not in point:
            return FPFailure(FieldError(
                field=timestamp_field,
                message=f"Missing timestamp field in data point {i}",
                validation_rule="temporal_sequence"
            ))
        
        try:
            current_timestamp = float(point[timestamp_field])
        except (ValueError, TypeError):
            return FPFailure(FieldError(
                field=timestamp_field,
                message=f"Invalid timestamp in data point {i}",
                value=str(point[timestamp_field]),
                validation_rule="temporal_sequence"
            ))
        
        if prev_timestamp is not None:
            # Check if timestamps are in order
            if current_timestamp < prev_timestamp - tolerance_seconds:
                return FPFailure(FieldError(
                    field=timestamp_field,
                    message=f"Timestamp sequence violation at point {i}",
                    validation_rule="temporal_sequence",
                    context={
                        "current_timestamp": current_timestamp,
                        "previous_timestamp": prev_timestamp,
                        "tolerance_seconds": tolerance_seconds
                    }
                ))
        
        prev_timestamp = current_timestamp
    
    return FPSuccess(data_points)


def validate_temporal_gaps(
    data_points: List[Dict[str, Any]], 
    timestamp_field: str = "timestamp",
    max_gap_seconds: float = 300.0  # 5 minutes
) -> FPResult[List[Dict[str, Any]], FieldError]:
    """Validate there are no excessive gaps in temporal data."""
    if len(data_points) < 2:
        return FPSuccess(data_points)
    
    for i in range(1, len(data_points)):
        try:
            prev_ts = float(data_points[i-1][timestamp_field])
            curr_ts = float(data_points[i][timestamp_field])
            
            gap = curr_ts - prev_ts
            if gap > max_gap_seconds:
                return FPFailure(FieldError(
                    field=timestamp_field,
                    message=f"Excessive temporal gap at point {i}",
                    validation_rule="temporal_gaps",
                    context={
                        "gap_seconds": gap,
                        "max_gap_seconds": max_gap_seconds,
                        "point_index": i
                    }
                ))
        except (ValueError, TypeError, KeyError):
            return FPFailure(FieldError(
                field=timestamp_field,
                message=f"Invalid timestamp data at point {i}",
                validation_rule="temporal_gaps"
            ))
    
    return FPSuccess(data_points)


# Export functional data integrity validators
__all__ = [
    # Core types
    "IntegrityLevel",
    "IntegrityViolationType",
    "IntegrityRule",
    "CrossFieldRule",
    
    # Main validator
    "FunctionalIntegrityValidator",
    
    # Rule creators
    "create_balance_consistency_rule",
    "create_position_size_rule",
    "create_price_sanity_rule",
    "create_timestamp_consistency_rule",
    "create_trade_action_risk_rule",
    "create_position_balance_rule",
    
    # Temporal validators
    "validate_temporal_sequence",
    "validate_temporal_gaps",
]