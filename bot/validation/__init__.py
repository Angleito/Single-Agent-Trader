"""
Enhanced validation module with functional programming patterns.

This module provides comprehensive validation systems for:
- Balance integrity and range checks
- Trading action validation with pure functions
- Data integrity checks with composable rules
- Validation pipelines for complex data flows
- Functional decorators for type safety
- Backward compatibility with legacy validation

Features:
- Pure functional validation primitives
- Composable validation chains
- Monadic error handling
- Data integrity verification
- Performance metrics and monitoring
"""

# Legacy validation components (preserved for backward compatibility)
from .balance_validator import BalanceValidationError, BalanceValidator
from .decorators import (
    validate_balance,
    validate_percentage,
    validate_position,
    validate_trade_action,
)

# Enhanced functional validation components
try:
    from .data_integrity import (
        CrossFieldRule,
        FunctionalIntegrityValidator,
        IntegrityLevel,
        IntegrityRule,
        IntegrityViolationType,
        validate_temporal_gaps,
        validate_temporal_sequence,
    )
    from .functional_decorators import (
        functional_balance_validator,
        functional_chain_validator,
        functional_percentage_validator,
        functional_pipeline_validator,
        functional_position_validator,
        functional_trade_action_validator,
        functional_validator,
    )
    from .pipeline import (
        ExecutionMode,
        FunctionalValidationPipeline,
        PipelineMetrics,
        PipelineResult,
        PipelineStage,
        create_market_data_pipeline,
        create_trade_action_pipeline,
    )

    FUNCTIONAL_VALIDATION_AVAILABLE = True

    # Enhanced export list with functional components
    __all__ = [
        # Legacy validation (preserved)
        "BalanceValidationError",
        "BalanceValidator",
        "validate_balance",
        "validate_percentage",
        "validate_position",
        "validate_trade_action",
        # Enhanced functional validation
        "functional_validator",
        "functional_chain_validator",
        "functional_balance_validator",
        "functional_trade_action_validator",
        "functional_position_validator",
        "functional_percentage_validator",
        "functional_pipeline_validator",
        # Data integrity
        "FunctionalIntegrityValidator",
        "IntegrityLevel",
        "IntegrityViolationType",
        "IntegrityRule",
        "CrossFieldRule",
        "validate_temporal_sequence",
        "validate_temporal_gaps",
        # Validation pipelines
        "FunctionalValidationPipeline",
        "PipelineStage",
        "ExecutionMode",
        "PipelineResult",
        "PipelineMetrics",
        "create_trade_action_pipeline",
        "create_market_data_pipeline",
        # Status flag
        "FUNCTIONAL_VALIDATION_AVAILABLE",
    ]

except ImportError:
    FUNCTIONAL_VALIDATION_AVAILABLE = False

    # Fall back to legacy exports only
    __all__ = [
        "FUNCTIONAL_VALIDATION_AVAILABLE",
        "BalanceValidationError",
        "BalanceValidator",
        "validate_balance",
        "validate_percentage",
        "validate_position",
        "validate_trade_action",
    ]
