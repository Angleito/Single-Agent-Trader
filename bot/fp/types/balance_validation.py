"""
Balance validation types for functional programming architecture.

This module defines immutable data structures and pure functions for balance validation,
ensuring trading account integrity and preventing invalid operations.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from bot.fp.types.base import Maybe, Some, Nothing
from bot.fp.types.result import Result, Success, Failure


# Balance Validation Types

class BalanceValidationType(Enum):
    """Types of balance validations."""
    
    RANGE_CHECK = "range_check"
    MARGIN_VALIDATION = "margin_validation"
    TRADE_AFFORDABILITY = "trade_affordability"
    POST_TRADE_BALANCE = "post_trade_balance"
    LEVERAGE_COMPLIANCE = "leverage_compliance"
    COMPREHENSIVE_CHECK = "comprehensive_check"


@dataclass(frozen=True)
class BalanceRange:
    """Valid balance range definition."""
    
    minimum: Decimal
    maximum: Decimal
    currency: str = "USD"
    
    def __post_init__(self):
        """Validate range parameters."""
        if self.minimum < 0:
            raise ValueError(f"Minimum balance cannot be negative: {self.minimum}")
        if self.maximum < self.minimum:
            raise ValueError(f"Maximum balance {self.maximum} cannot be less than minimum {self.minimum}")
    
    def contains(self, balance: Decimal) -> bool:
        """Check if balance is within range."""
        return self.minimum <= balance <= self.maximum
    
    def distance_from_range(self, balance: Decimal) -> Decimal:
        """Calculate distance from valid range."""
        if balance < self.minimum:
            return self.minimum - balance
        elif balance > self.maximum:
            return balance - self.maximum
        return Decimal(0)


@dataclass(frozen=True)
class MarginRequirement:
    """Margin requirement specification."""
    
    position_value: Decimal
    leverage: Decimal
    maintenance_margin_pct: Decimal = Decimal("0.05")  # 5% default
    initial_margin_pct: Decimal = Decimal("0.10")  # 10% default
    
    def __post_init__(self):
        """Validate margin requirements."""
        if self.position_value < 0:
            raise ValueError(f"Position value cannot be negative: {self.position_value}")
        if self.leverage <= 0:
            raise ValueError(f"Leverage must be positive: {self.leverage}")
        if self.maintenance_margin_pct < 0 or self.maintenance_margin_pct > 1:
            raise ValueError(f"Maintenance margin percentage must be between 0 and 1: {self.maintenance_margin_pct}")
        if self.initial_margin_pct < self.maintenance_margin_pct:
            raise ValueError(f"Initial margin {self.initial_margin_pct} cannot be less than maintenance margin {self.maintenance_margin_pct}")
    
    @property
    def required_margin(self) -> Decimal:
        """Calculate required initial margin."""
        return self.position_value / self.leverage
    
    @property
    def maintenance_margin(self) -> Decimal:
        """Calculate maintenance margin requirement."""
        return self.position_value * self.maintenance_margin_pct
    
    @property
    def initial_margin(self) -> Decimal:
        """Calculate initial margin requirement."""
        return self.position_value * self.initial_margin_pct


@dataclass(frozen=True)
class BalanceValidationError:
    """Balance validation error details."""
    
    error_type: BalanceValidationType
    message: str
    current_balance: Decimal
    expected_range: Optional[BalanceRange]
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    additional_context: Dict[str, any] = None
    
    def __post_init__(self):
        if self.additional_context is None:
            object.__setattr__(self, 'additional_context', {})


@dataclass(frozen=True)
class BalanceValidationResult:
    """Result of balance validation."""
    
    is_valid: bool
    validation_type: BalanceValidationType
    balance: Decimal
    message: str
    timestamp: datetime
    error: Optional[BalanceValidationError] = None
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0
    
    @property
    def is_critical_error(self) -> bool:
        """Check if validation has critical error."""
        return self.error is not None and self.error.severity == "critical"


@dataclass(frozen=True)
class TradeAffordabilityCheck:
    """Trade affordability validation parameters."""
    
    trade_value: Decimal
    estimated_fees: Decimal
    required_margin: Decimal
    leverage: Decimal
    current_balance: Decimal
    existing_margin_used: Decimal = Decimal(0)
    
    def __post_init__(self):
        """Validate parameters."""
        if self.trade_value < 0:
            raise ValueError(f"Trade value cannot be negative: {self.trade_value}")
        if self.estimated_fees < 0:
            raise ValueError(f"Estimated fees cannot be negative: {self.estimated_fees}")
        if self.required_margin < 0:
            raise ValueError(f"Required margin cannot be negative: {self.required_margin}")
        if self.leverage <= 0:
            raise ValueError(f"Leverage must be positive: {self.leverage}")
        if self.existing_margin_used < 0:
            raise ValueError(f"Existing margin used cannot be negative: {self.existing_margin_used}")
    
    @property
    def total_required_capital(self) -> Decimal:
        """Calculate total capital required for trade."""
        return self.required_margin + self.estimated_fees
    
    @property
    def total_margin_after_trade(self) -> Decimal:
        """Calculate total margin usage after trade."""
        return self.existing_margin_used + self.required_margin
    
    @property
    def available_balance_after_trade(self) -> Decimal:
        """Calculate available balance after trade."""
        return self.current_balance - self.total_required_capital
    
    @property
    def margin_utilization_pct(self) -> float:
        """Calculate margin utilization percentage."""
        if self.current_balance <= 0:
            return 100.0
        return float(self.total_margin_after_trade / self.current_balance * 100)


@dataclass(frozen=True)
class BalanceValidationConfig:
    """Configuration for balance validation."""
    
    min_balance: Decimal = Decimal("100")  # Minimum account balance
    max_balance: Decimal = Decimal("10000000")  # Maximum account balance
    max_margin_utilization_pct: float = 80.0  # Maximum margin utilization
    min_free_balance_pct: float = 10.0  # Minimum free balance percentage
    emergency_threshold_pct: float = 5.0  # Emergency stop threshold
    
    @property
    def balance_range(self) -> BalanceRange:
        """Get balance range from config."""
        return BalanceRange(
            minimum=self.min_balance,
            maximum=self.max_balance
        )


@dataclass(frozen=True)
class ComprehensiveBalanceValidation:
    """Comprehensive balance validation state."""
    
    current_balance: Decimal
    config: BalanceValidationConfig
    range_validation: BalanceValidationResult
    margin_validation: Optional[BalanceValidationResult]
    trade_affordability: Optional[BalanceValidationResult]
    leverage_compliance: Optional[BalanceValidationResult]
    timestamp: datetime
    
    @property
    def is_all_valid(self) -> bool:
        """Check if all validations passed."""
        validations = [self.range_validation]
        
        if self.margin_validation:
            validations.append(self.margin_validation)
        if self.trade_affordability:
            validations.append(self.trade_affordability)
        if self.leverage_compliance:
            validations.append(self.leverage_compliance)
        
        return all(v.is_valid for v in validations)
    
    @property
    def critical_errors(self) -> List[BalanceValidationError]:
        """Get all critical errors."""
        errors = []
        
        validations = [self.range_validation]
        if self.margin_validation:
            validations.append(self.margin_validation)
        if self.trade_affordability:
            validations.append(self.trade_affordability)
        if self.leverage_compliance:
            validations.append(self.leverage_compliance)
        
        for validation in validations:
            if validation.error and validation.error.severity == "critical":
                errors.append(validation.error)
        
        return errors
    
    @property
    def all_warnings(self) -> List[str]:
        """Get all warnings."""
        warnings = []
        
        validations = [self.range_validation]
        if self.margin_validation:
            validations.append(self.margin_validation)
        if self.trade_affordability:
            validations.append(self.trade_affordability)
        if self.leverage_compliance:
            validations.append(self.leverage_compliance)
        
        for validation in validations:
            warnings.extend(validation.warnings)
        
        return warnings


# Pure Balance Validation Functions


def validate_balance_range(
    balance: Decimal,
    balance_range: BalanceRange,
    operation_context: str = "general"
) -> BalanceValidationResult:
    """Validate balance is within acceptable range."""
    timestamp = datetime.now()
    
    if balance_range.contains(balance):
        return BalanceValidationResult(
            is_valid=True,
            validation_type=BalanceValidationType.RANGE_CHECK,
            balance=balance,
            message=f"Balance {balance} is within valid range [{balance_range.minimum}, {balance_range.maximum}]",
            timestamp=timestamp
        )
    
    # Balance is outside range
    distance = balance_range.distance_from_range(balance)
    
    if balance < balance_range.minimum:
        severity = "critical" if balance <= 0 else "high"
        message = f"Balance {balance} is below minimum {balance_range.minimum} (deficit: {distance})"
    else:
        severity = "medium"  # High balance is less critical
        message = f"Balance {balance} exceeds maximum {balance_range.maximum} (excess: {distance})"
    
    error = BalanceValidationError(
        error_type=BalanceValidationType.RANGE_CHECK,
        message=message,
        current_balance=balance,
        expected_range=balance_range,
        severity=severity,
        timestamp=timestamp,
        additional_context={"operation_context": operation_context}
    )
    
    return BalanceValidationResult(
        is_valid=False,
        validation_type=BalanceValidationType.RANGE_CHECK,
        balance=balance,
        message=message,
        timestamp=timestamp,
        error=error
    )


def validate_margin_requirements(
    balance: Decimal,
    margin_requirement: MarginRequirement,
    used_margin: Decimal = Decimal(0)
) -> BalanceValidationResult:
    """Validate margin requirements can be met."""
    timestamp = datetime.now()
    
    available_balance = balance - used_margin
    required_margin = margin_requirement.required_margin
    
    if available_balance >= required_margin:
        margin_utilization = float((used_margin + required_margin) / balance * 100) if balance > 0 else 100.0
        
        warnings = []
        if margin_utilization > 70.0:
            warnings.append(f"High margin utilization: {margin_utilization:.1f}%")
        
        return BalanceValidationResult(
            is_valid=True,
            validation_type=BalanceValidationType.MARGIN_VALIDATION,
            balance=balance,
            message=f"Margin requirement {required_margin} can be met (available: {available_balance})",
            timestamp=timestamp,
            warnings=tuple(warnings),
            metadata={
                "required_margin": float(required_margin),
                "available_balance": float(available_balance),
                "margin_utilization_pct": margin_utilization
            }
        )
    
    # Insufficient margin
    shortage = required_margin - available_balance
    severity = "critical" if available_balance <= 0 else "high"
    
    error = BalanceValidationError(
        error_type=BalanceValidationType.MARGIN_VALIDATION,
        message=f"Insufficient margin: need {required_margin}, have {available_balance} (shortage: {shortage})",
        current_balance=balance,
        expected_range=None,
        severity=severity,
        timestamp=timestamp,
        additional_context={
            "required_margin": float(required_margin),
            "available_balance": float(available_balance),
            "used_margin": float(used_margin),
            "shortage": float(shortage)
        }
    )
    
    return BalanceValidationResult(
        is_valid=False,
        validation_type=BalanceValidationType.MARGIN_VALIDATION,
        balance=balance,
        message=error.message,
        timestamp=timestamp,
        error=error
    )


def validate_trade_affordability(
    affordability_check: TradeAffordabilityCheck
) -> BalanceValidationResult:
    """Validate that a trade can be afforded."""
    timestamp = datetime.now()
    
    can_afford = affordability_check.current_balance >= affordability_check.total_required_capital
    margin_util = affordability_check.margin_utilization_pct
    
    if can_afford and margin_util <= 80.0:
        warnings = []
        if margin_util > 60.0:
            warnings.append(f"High margin utilization: {margin_util:.1f}%")
        if affordability_check.available_balance_after_trade < affordability_check.current_balance * Decimal("0.1"):
            warnings.append("Low remaining balance after trade")
        
        return BalanceValidationResult(
            is_valid=True,
            validation_type=BalanceValidationType.TRADE_AFFORDABILITY,
            balance=affordability_check.current_balance,
            message=f"Trade is affordable (required: {affordability_check.total_required_capital}, available: {affordability_check.current_balance})",
            timestamp=timestamp,
            warnings=tuple(warnings),
            metadata={
                "trade_value": float(affordability_check.trade_value),
                "estimated_fees": float(affordability_check.estimated_fees),
                "required_margin": float(affordability_check.required_margin),
                "margin_utilization_pct": margin_util,
                "available_after_trade": float(affordability_check.available_balance_after_trade)
            }
        )
    
    # Cannot afford trade
    if not can_afford:
        shortage = affordability_check.total_required_capital - affordability_check.current_balance
        severity = "critical"
        message = f"Cannot afford trade: need {affordability_check.total_required_capital}, have {affordability_check.current_balance} (shortage: {shortage})"
    else:
        severity = "high"
        message = f"Trade would exceed margin limits: {margin_util:.1f}% utilization"
    
    error = BalanceValidationError(
        error_type=BalanceValidationType.TRADE_AFFORDABILITY,
        message=message,
        current_balance=affordability_check.current_balance,
        expected_range=None,
        severity=severity,
        timestamp=timestamp,
        additional_context={
            "trade_value": float(affordability_check.trade_value),
            "total_required": float(affordability_check.total_required_capital),
            "margin_utilization_pct": margin_util
        }
    )
    
    return BalanceValidationResult(
        is_valid=False,
        validation_type=BalanceValidationType.TRADE_AFFORDABILITY,
        balance=affordability_check.current_balance,
        message=message,
        timestamp=timestamp,
        error=error
    )


def validate_leverage_compliance(
    position_value: Decimal,
    leverage: Decimal,
    max_leverage: Decimal,
    balance: Decimal
) -> BalanceValidationResult:
    """Validate leverage compliance."""
    timestamp = datetime.now()
    
    if leverage <= max_leverage:
        warnings = []
        if leverage > max_leverage * Decimal("0.8"):
            warnings.append(f"High leverage: {leverage}x (max: {max_leverage}x)")
        
        # Check if position size is reasonable relative to balance
        margin_required = position_value / leverage
        margin_pct = float(margin_required / balance * 100) if balance > 0 else 100.0
        
        if margin_pct > 50.0:
            warnings.append(f"Large position relative to balance: {margin_pct:.1f}% of balance required")
        
        return BalanceValidationResult(
            is_valid=True,
            validation_type=BalanceValidationType.LEVERAGE_COMPLIANCE,
            balance=balance,
            message=f"Leverage {leverage}x is within limits (max: {max_leverage}x)",
            timestamp=timestamp,
            warnings=tuple(warnings),
            metadata={
                "leverage": float(leverage),
                "max_leverage": float(max_leverage),
                "position_value": float(position_value),
                "margin_required": float(margin_required),
                "margin_pct": margin_pct
            }
        )
    
    # Leverage exceeds limits
    excess = leverage - max_leverage
    severity = "high" if excess <= max_leverage * Decimal("0.2") else "critical"
    
    error = BalanceValidationError(
        error_type=BalanceValidationType.LEVERAGE_COMPLIANCE,
        message=f"Leverage {leverage}x exceeds maximum {max_leverage}x (excess: {excess}x)",
        current_balance=balance,
        expected_range=None,
        severity=severity,
        timestamp=timestamp,
        additional_context={
            "leverage": float(leverage),
            "max_leverage": float(max_leverage),
            "excess": float(excess)
        }
    )
    
    return BalanceValidationResult(
        is_valid=False,
        validation_type=BalanceValidationType.LEVERAGE_COMPLIANCE,
        balance=balance,
        message=error.message,
        timestamp=timestamp,
        error=error
    )


def perform_comprehensive_balance_validation(
    balance: Decimal,
    config: BalanceValidationConfig,
    margin_requirement: Optional[MarginRequirement] = None,
    affordability_check: Optional[TradeAffordabilityCheck] = None,
    leverage_check: Optional[Tuple[Decimal, Decimal, Decimal]] = None,  # (position_value, leverage, max_leverage)
    used_margin: Decimal = Decimal(0)
) -> ComprehensiveBalanceValidation:
    """Perform comprehensive balance validation."""
    timestamp = datetime.now()
    
    # Always validate range
    range_validation = validate_balance_range(balance, config.balance_range)
    
    # Optional margin validation
    margin_validation = None
    if margin_requirement:
        margin_validation = validate_margin_requirements(balance, margin_requirement, used_margin)
    
    # Optional affordability validation
    trade_affordability_validation = None
    if affordability_check:
        trade_affordability_validation = validate_trade_affordability(affordability_check)
    
    # Optional leverage validation
    leverage_validation = None
    if leverage_check:
        position_value, leverage, max_leverage = leverage_check
        leverage_validation = validate_leverage_compliance(position_value, leverage, max_leverage, balance)
    
    return ComprehensiveBalanceValidation(
        current_balance=balance,
        config=config,
        range_validation=range_validation,
        margin_validation=margin_validation,
        trade_affordability=trade_affordability_validation,
        leverage_compliance=leverage_validation,
        timestamp=timestamp
    )


def create_default_balance_config() -> BalanceValidationConfig:
    """Create default balance validation configuration."""
    return BalanceValidationConfig(
        min_balance=Decimal("100"),
        max_balance=Decimal("10000000"),
        max_margin_utilization_pct=80.0,
        min_free_balance_pct=10.0,
        emergency_threshold_pct=5.0
    )


def create_margin_requirement(
    position_value: Decimal,
    leverage: Decimal
) -> MarginRequirement:
    """Create margin requirement from position and leverage."""
    return MarginRequirement(
        position_value=position_value,
        leverage=leverage
    )


def create_trade_affordability_check(
    trade_value: Decimal,
    estimated_fees: Decimal,
    leverage: Decimal,
    current_balance: Decimal,
    existing_margin_used: Decimal = Decimal(0)
) -> TradeAffordabilityCheck:
    """Create trade affordability check."""
    required_margin = trade_value / leverage
    
    return TradeAffordabilityCheck(
        trade_value=trade_value,
        estimated_fees=estimated_fees,
        required_margin=required_margin,
        leverage=leverage,
        current_balance=current_balance,
        existing_margin_used=existing_margin_used
    )