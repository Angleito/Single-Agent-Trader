"""
Functional learning module for AI trading bot.

This module provides functional programming patterns for learning
and memory management, built on pure functions and immutable data structures.
"""

from .experience import (
    ExperienceState,
    LearningContext,
    PatternState,
    TradeExperience,
    create_experience,
    update_experience_outcome,
    query_similar_experiences,
    analyze_pattern_performance,
)

from .memory_effects import (
    MemoryEffect,
    store_experience,
    retrieve_experience,
    query_experiences,
    update_experience,
    cleanup_old_experiences,
)

from .learning_algorithms import (
    LearningResult,
    PatternAnalysis,
    StrategyInsight,
    analyze_trading_patterns,
    generate_strategy_insights,
    calculate_pattern_confidence,
    optimize_parameters,
)

from .combinators import (
    compose_learning_functions,
    sequence_learning_operations,
    parallel_learning_analysis,
    fold_experiences,
    filter_experiences,
    map_experiences,
    build_analysis_pipeline,
    validate_minimum_experiences,
    when,
    unless,
    with_fallback,
    tap,
    memoize,
)

from .functional_experience_manager import (
    FunctionalExperienceManager,
    FunctionalExperienceManagerState,
    ActiveTradeState,
)

from .functional_self_improvement import (
    FunctionalSelfImprovementEngine,
    FunctionalSelfImprovementState,
    ImprovementRecommendation,
    PerformanceMetrics,
)

__all__ = [
    # Experience management
    "ExperienceState",
    "LearningContext", 
    "PatternState",
    "TradeExperience",
    "create_experience",
    "update_experience_outcome",
    "query_similar_experiences",
    "analyze_pattern_performance",
    
    # Memory effects
    "MemoryEffect",
    "store_experience",
    "retrieve_experience", 
    "query_experiences",
    "update_experience",
    "cleanup_old_experiences",
    
    # Learning algorithms
    "LearningResult",
    "PatternAnalysis",
    "StrategyInsight",
    "analyze_trading_patterns",
    "generate_strategy_insights",
    "calculate_pattern_confidence",
    "optimize_parameters",
    
    # Combinators
    "compose_learning_functions",
    "sequence_learning_operations",
    "parallel_learning_analysis", 
    "fold_experiences",
    "filter_experiences",
    "map_experiences",
    "build_analysis_pipeline",
    "validate_minimum_experiences",
    "when",
    "unless", 
    "with_fallback",
    "tap",
    "memoize",
    
    # Functional managers
    "FunctionalExperienceManager",
    "FunctionalExperienceManagerState", 
    "ActiveTradeState",
    "FunctionalSelfImprovementEngine",
    "FunctionalSelfImprovementState",
    "ImprovementRecommendation",
    "PerformanceMetrics",
]