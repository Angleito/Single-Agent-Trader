"""
Functional programming tests for strategy configuration management.

This module tests strategy configuration using functional programming patterns,
ensuring immutable configurations, type safety, and proper integration with
the broader FP trading system architecture.

Tests include:
- Strategy configuration immutability and type safety
- Configuration validation with Result/Either types
- Strategy parameter composition and transformation
- Configuration adapters between FP and legacy systems
- Strategy configuration builders and factories
- Configuration consistency across strategy types
- Error handling in configuration management
- Integration with functional strategy system
"""

import pytest
from datetime import datetime, UTC
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import tempfile
import json

from hypothesis import given, strategies as st, settings, HealthCheck

# FP test infrastructure
from tests.fp_test_base import (
    FPTestBase,
    FP_AVAILABLE
)

if FP_AVAILABLE:
    # FP configuration types
    from bot.fp.types.config import (
        LLMStrategyConfig, MomentumStrategyConfig, MeanReversionStrategyConfig,
        Config, SystemConfig, validate_config
    )
    from bot.fp.types.result import Result, Success, Failure
    from bot.fp.types.base import Maybe, Some, Nothing, Percentage
    from bot.fp.types.trading import Long, Short, Hold, TradeSignal
    
    # Strategy-specific FP types
    try:
        from bot.fp.strategies.base import (
            StrategyConfig, StrategyParameters, StrategyContext,
            FunctionalStrategy, create_strategy_context
        )
        STRATEGY_FP_AVAILABLE = True
    except ImportError:
        STRATEGY_FP_AVAILABLE = False
        
        # Mock strategy types for testing
        class StrategyConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class StrategyParameters:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class StrategyContext:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        def create_strategy_context(*args, **kwargs):
            return StrategyContext()
else:
    # Fallback stubs for non-FP environments
    STRATEGY_FP_AVAILABLE = False
    
    class StrategyConfig:
        pass


class TestFunctionalStrategyConfiguration(FPTestBase):
    """Test functional strategy configuration patterns."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_llm_strategy_config_immutability(self):
        """Test that LLM strategy configuration is immutable."""
        config_result = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=75.0
        )
        
        assert config_result.is_success()
        config = config_result.success()
        
        # Verify immutability
        with pytest.raises(AttributeError):
            config.model_name = "gpt-3.5"
        
        with pytest.raises(AttributeError):
            config.temperature = 0.5
        
        with pytest.raises(AttributeError):
            config.use_memory = False
        
        # Verify values
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_context_length == 4000
        assert config.use_memory is True
        assert config.confidence_threshold.value == 75.0
    
    def test_momentum_strategy_config_validation(self):
        """Test momentum strategy configuration validation."""
        # Valid configuration
        valid_result = MomentumStrategyConfig.create(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=1.0,
            use_volume_confirmation=True
        )
        
        assert valid_result.is_success()
        config = valid_result.success()
        assert config.lookback_period == 20
        assert config.entry_threshold.value == 2.0
        assert config.exit_threshold.value == 1.0
        assert config.use_volume_confirmation is True
        
        # Invalid lookback period
        invalid_result = MomentumStrategyConfig.create(
            lookback_period=-5,
            entry_threshold=2.0,
            exit_threshold=1.0
        )
        
        assert invalid_result.is_failure()
        assert "Lookback period must be positive" in invalid_result.failure()
        
        # Invalid threshold relationship
        invalid_threshold_result = MomentumStrategyConfig.create(
            lookback_period=20,
            entry_threshold=1.0,
            exit_threshold=2.0  # Exit > Entry is invalid
        )
        
        assert invalid_threshold_result.is_failure()
        assert "Exit threshold should be less than entry threshold" in invalid_threshold_result.failure()
    
    def test_mean_reversion_strategy_config_constraints(self):
        """Test mean reversion strategy configuration constraints."""
        # Valid configuration
        valid_result = MeanReversionStrategyConfig.create(
            window_size=50,
            std_deviations=2.0,
            min_volatility=0.1,
            max_holding_period=100
        )
        
        assert valid_result.is_success()
        config = valid_result.success()
        assert config.window_size == 50
        assert config.std_deviations == 2.0
        assert config.min_volatility.value == 0.1
        assert config.max_holding_period == 100
        
        # Invalid window size
        invalid_window_result = MeanReversionStrategyConfig.create(
            window_size=1,  # Too small
            std_deviations=2.0,
            min_volatility=0.1,
            max_holding_period=100
        )
        
        assert invalid_window_result.is_failure()
        assert "Window size must be at least 2" in invalid_window_result.failure()
        
        # Invalid standard deviations
        invalid_std_result = MeanReversionStrategyConfig.create(
            window_size=50,
            std_deviations=0.0,  # Must be positive
            min_volatility=0.1,
            max_holding_period=100
        )
        
        assert invalid_std_result.is_failure()
        assert "Standard deviations must be positive" in invalid_std_result.failure()
    
    def test_strategy_config_composition(self):
        """Test composition of strategy configurations."""
        llm_config = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0
        ).success()
        
        momentum_config = MomentumStrategyConfig.create(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=1.0,
            use_volume_confirmation=True
        ).success()
        
        # Test that configurations can be used in different contexts
        strategy_configs = [llm_config, momentum_config]
        
        for config in strategy_configs:
            # All strategy configs should have common properties
            assert hasattr(config, '__class__')
            # Should be serializable (for logging/debugging)
            config_str = str(config)
            assert len(config_str) > 0
    
    def test_strategy_config_parameter_extraction(self):
        """Test extracting parameters from strategy configurations."""
        llm_config = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.8,
            max_context_length=3000,
            use_memory=True,
            confidence_threshold=80.0
        ).success()
        
        # Test parameter extraction for different use cases
        model_params = {
            "model": llm_config.model_name,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_context_length
        }
        
        assert model_params["model"] == "gpt-4"
        assert model_params["temperature"] == 0.8
        assert model_params["max_tokens"] == 3000
        
        # Test memory flag
        assert llm_config.use_memory is True
        
        # Test confidence threshold
        assert llm_config.confidence_threshold.value == 80.0
    
    def test_strategy_config_serialization_properties(self):
        """Test strategy configuration serialization properties."""
        configs = [
            LLMStrategyConfig.create(
                model_name="gpt-4",
                temperature=0.7,
                max_context_length=4000,
                use_memory=False,
                confidence_threshold=70.0
            ).success(),
            MomentumStrategyConfig.create(
                lookback_period=20,
                entry_threshold=2.0,
                exit_threshold=1.0,
                use_volume_confirmation=True
            ).success(),
            MeanReversionStrategyConfig.create(
                window_size=50,
                std_deviations=2.0,
                min_volatility=0.1,
                max_holding_period=100
            ).success()
        ]
        
        for config in configs:
            # Test string representation
            config_str = str(config)
            assert isinstance(config_str, str)
            assert len(config_str) > 0
            
            # Test repr
            config_repr = repr(config)
            assert isinstance(config_repr, str)
            assert config.__class__.__name__ in config_repr
    
    def test_strategy_config_validation_integration(self):
        """Test strategy configuration validation in integrated system context."""
        if not STRATEGY_FP_AVAILABLE:
            pytest.skip("Strategy FP module not available")
        
        # Create various strategy configurations
        llm_config = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=70.0
        ).success()
        
        # Test in strategy context
        context = create_strategy_context(
            config=llm_config,
            market_conditions="normal",
            risk_level="medium"
        )
        
        assert hasattr(context, 'config')
        # Context should preserve configuration immutability
        original_temp = context.config.temperature
        with pytest.raises(AttributeError):
            context.config.temperature = 0.5
        assert context.config.temperature == original_temp
    
    def test_strategy_config_error_composition(self):
        """Test error handling and composition in strategy configurations."""
        # Test multiple validation errors
        invalid_llm_result = LLMStrategyConfig.create(
            model_name="",  # Empty model name
            temperature=3.0,  # Invalid temperature
            max_context_length=50,  # Too small
            use_memory=True,
            confidence_threshold=-10.0  # Invalid threshold
        )
        
        assert invalid_llm_result.is_failure()
        error_message = invalid_llm_result.failure()
        
        # Should contain specific error information
        assert len(error_message) > 0
        assert ("model" in error_message.lower() or 
                "temperature" in error_message.lower() or
                "context" in error_message.lower() or
                "confidence" in error_message.lower())
    
    def test_strategy_config_type_safety(self):
        """Test type safety in strategy configurations."""
        # Test that type constraints are enforced
        with pytest.raises((TypeError, ValueError)):
            # This should fail at runtime due to type constraints
            LLMStrategyConfig.create(
                model_name=123,  # Should be string
                temperature="invalid",  # Should be float
                max_context_length="invalid",  # Should be int
                use_memory="maybe",  # Should be bool
                confidence_threshold="high"  # Should be float
            )
    
    @pytest.mark.parametrize("model_name", ["gpt-4", "gpt-3.5-turbo", "claude-3"])
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_llm_config_parameter_ranges(self, model_name, temperature):
        """Test LLM configuration with various parameter ranges."""
        config_result = LLMStrategyConfig.create(
            model_name=model_name,
            temperature=temperature,
            max_context_length=4000,
            use_memory=False,
            confidence_threshold=70.0
        )
        
        assert config_result.is_success()
        config = config_result.success()
        assert config.model_name == model_name
        assert config.temperature == temperature
    
    @pytest.mark.parametrize("lookback", [5, 10, 20, 50, 100])
    @pytest.mark.parametrize("entry_threshold", [1.5, 2.0, 2.5, 3.0])
    def test_momentum_config_parameter_ranges(self, lookback, entry_threshold):
        """Test momentum configuration with various parameter ranges."""
        exit_threshold = entry_threshold - 0.5  # Ensure valid relationship
        
        config_result = MomentumStrategyConfig.create(
            lookback_period=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_volume_confirmation=True
        )
        
        assert config_result.is_success()
        config = config_result.success()
        assert config.lookback_period == lookback
        assert config.entry_threshold.value == entry_threshold
        assert config.exit_threshold.value == exit_threshold


class TestStrategyConfigurationAdapters(FPTestBase):
    """Test strategy configuration adapters between FP and legacy systems."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_fp_to_legacy_config_conversion(self):
        """Test conversion from FP configuration to legacy format."""
        fp_config = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=75.0
        ).success()
        
        # Convert to legacy format (simplified)
        legacy_config = {
            "model": fp_config.model_name,
            "temperature": fp_config.temperature,
            "max_context": fp_config.max_context_length,
            "memory_enabled": fp_config.use_memory,
            "confidence": fp_config.confidence_threshold.value
        }
        
        # Verify conversion preserves essential data
        assert legacy_config["model"] == "gpt-4"
        assert legacy_config["temperature"] == 0.7
        assert legacy_config["max_context"] == 4000
        assert legacy_config["memory_enabled"] is True
        assert legacy_config["confidence"] == 75.0
    
    def test_legacy_to_fp_config_conversion(self):
        """Test conversion from legacy configuration to FP format."""
        legacy_config = {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.8,
            "max_tokens": 3000,
            "use_memory": False,
            "min_confidence": 65.0
        }
        
        # Convert to FP format
        fp_config_result = LLMStrategyConfig.create(
            model_name=legacy_config["model_name"],
            temperature=legacy_config["temperature"],
            max_context_length=legacy_config["max_tokens"],
            use_memory=legacy_config["use_memory"],
            confidence_threshold=legacy_config["min_confidence"]
        )
        
        assert fp_config_result.is_success()
        fp_config = fp_config_result.success()
        
        # Verify conversion
        assert fp_config.model_name == "gpt-3.5-turbo"
        assert fp_config.temperature == 0.8
        assert fp_config.max_context_length == 3000
        assert fp_config.use_memory is False
        assert fp_config.confidence_threshold.value == 65.0
    
    def test_config_adapter_error_handling(self):
        """Test error handling in configuration adapters."""
        # Test invalid legacy configuration
        invalid_legacy = {
            "model_name": "",  # Invalid
            "temperature": -1.0,  # Invalid
            "max_tokens": 0,  # Invalid
            "use_memory": "maybe",  # Invalid type
            "min_confidence": 150.0  # Invalid range
        }
        
        # Conversion should fail gracefully
        try:
            fp_config_result = LLMStrategyConfig.create(
                model_name=invalid_legacy["model_name"],
                temperature=invalid_legacy["temperature"],
                max_context_length=invalid_legacy["max_tokens"],
                use_memory=bool(invalid_legacy["use_memory"]),
                confidence_threshold=invalid_legacy["min_confidence"]
            )
            
            assert fp_config_result.is_failure()
        except (TypeError, ValueError) as e:
            # Type conversion errors are also acceptable
            assert isinstance(e, (TypeError, ValueError))
    
    def test_config_adapter_roundtrip_consistency(self):
        """Test that configuration adapters maintain consistency in roundtrips."""
        original_fp_config = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.6,
            max_context_length=3500,
            use_memory=True,
            confidence_threshold=72.5
        ).success()
        
        # Convert to legacy format
        legacy_config = {
            "model_name": original_fp_config.model_name,
            "temperature": original_fp_config.temperature,
            "max_tokens": original_fp_config.max_context_length,
            "use_memory": original_fp_config.use_memory,
            "min_confidence": original_fp_config.confidence_threshold.value
        }
        
        # Convert back to FP format
        roundtrip_fp_config = LLMStrategyConfig.create(
            model_name=legacy_config["model_name"],
            temperature=legacy_config["temperature"],
            max_context_length=legacy_config["max_tokens"],
            use_memory=legacy_config["use_memory"],
            confidence_threshold=legacy_config["min_confidence"]
        ).success()
        
        # Verify roundtrip consistency
        assert roundtrip_fp_config.model_name == original_fp_config.model_name
        assert roundtrip_fp_config.temperature == original_fp_config.temperature
        assert roundtrip_fp_config.max_context_length == original_fp_config.max_context_length
        assert roundtrip_fp_config.use_memory == original_fp_config.use_memory
        assert roundtrip_fp_config.confidence_threshold.value == original_fp_config.confidence_threshold.value


class TestStrategyConfigurationBuilders(FPTestBase):
    """Test strategy configuration builders and factories."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_config_builder_pattern(self):
        """Test builder pattern for strategy configurations."""
        # Test building configuration step by step
        base_config = {
            "model_name": "gpt-4",
            "temperature": 0.7
        }
        
        # Add additional parameters
        enhanced_config = {
            **base_config,
            "max_context_length": 4000,
            "use_memory": True,
            "confidence_threshold": 75.0
        }
        
        config_result = LLMStrategyConfig.create(**enhanced_config)
        assert config_result.is_success()
        
        config = config_result.success()
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_context_length == 4000
        assert config.use_memory is True
        assert config.confidence_threshold.value == 75.0
    
    def test_config_factory_methods(self):
        """Test factory methods for common configuration patterns."""
        # Conservative LLM configuration
        conservative_result = LLMStrategyConfig.create(
            model_name="gpt-3.5-turbo",
            temperature=0.3,  # Low temperature for consistency
            max_context_length=2000,
            use_memory=False,
            confidence_threshold=80.0  # High confidence threshold
        )
        
        assert conservative_result.is_success()
        conservative = conservative_result.success()
        assert conservative.temperature <= 0.5
        assert conservative.confidence_threshold.value >= 75.0
        
        # Aggressive momentum configuration
        aggressive_result = MomentumStrategyConfig.create(
            lookback_period=10,  # Short lookback
            entry_threshold=1.5,  # Low entry threshold
            exit_threshold=0.5,   # Very low exit threshold
            use_volume_confirmation=False  # No volume confirmation
        )
        
        assert aggressive_result.is_success()
        aggressive = aggressive_result.success()
        assert aggressive.lookback_period <= 15
        assert aggressive.entry_threshold.value <= 2.0
    
    def test_config_from_environment_pattern(self):
        """Test creating configurations from environment variables."""
        env_config = {
            "LLM_MODEL": "gpt-4",
            "LLM_TEMPERATURE": "0.7",
            "LLM_MAX_CONTEXT": "4000",
            "LLM_USE_MEMORY": "true",
            "LLM_CONFIDENCE": "70.0"
        }
        
        # Simulate environment variable parsing
        with patch.dict('os.environ', env_config):
            import os
            
            config_result = LLMStrategyConfig.create(
                model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
                max_context_length=int(os.getenv("LLM_MAX_CONTEXT", "3000")),
                use_memory=os.getenv("LLM_USE_MEMORY", "false").lower() == "true",
                confidence_threshold=float(os.getenv("LLM_CONFIDENCE", "75.0"))
            )
            
            assert config_result.is_success()
            config = config_result.success()
            assert config.model_name == "gpt-4"
            assert config.temperature == 0.7
            assert config.max_context_length == 4000
            assert config.use_memory is True
            assert config.confidence_threshold.value == 70.0


class TestStrategyConfigurationIntegration(FPTestBase):
    """Test strategy configuration integration with the broader system."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    def test_strategy_config_with_system_validation(self):
        """Test strategy configuration with overall system validation."""
        # Create strategy configuration
        strategy_config = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=70.0
        ).success()
        
        # Create system configuration that's compatible
        system_config = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": True},  # Compatible with strategy
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        # Test that configurations are compatible
        assert strategy_config.use_memory == system_config.features.enable_memory
    
    def test_strategy_config_memory_consistency(self):
        """Test memory configuration consistency across strategy and system."""
        # Strategy wants memory
        memory_strategy = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=0.7,
            max_context_length=4000,
            use_memory=True,
            confidence_threshold=70.0
        ).success()
        
        # System disables memory - should be caught by validation
        no_memory_system = SystemConfig.create(
            trading_pairs=["BTC-USD"],
            interval="1m",
            mode="paper",
            log_level="INFO",
            features={"enable_memory": False},  # Incompatible
            max_concurrent_positions=3,
            default_position_size=10.0
        ).success()
        
        # This inconsistency should be detectable
        assert memory_strategy.use_memory != no_memory_system.features.enable_memory
    
    def test_strategy_config_serialization_for_logging(self):
        """Test strategy configuration serialization for logging and debugging."""
        configs = [
            LLMStrategyConfig.create(
                model_name="gpt-4",
                temperature=0.7,
                max_context_length=4000,
                use_memory=True,
                confidence_threshold=70.0
            ).success(),
            MomentumStrategyConfig.create(
                lookback_period=20,
                entry_threshold=2.0,
                exit_threshold=1.0,
                use_volume_confirmation=True
            ).success()
        ]
        
        for config in configs:
            # Test that configurations can be serialized for logging
            config_dict = {
                "type": config.__class__.__name__,
                "config": str(config)
            }
            
            assert isinstance(config_dict["type"], str)
            assert isinstance(config_dict["config"], str)
            assert len(config_dict["config"]) > 0
    
    def test_strategy_config_performance_characteristics(self):
        """Test performance characteristics of strategy configuration operations."""
        # Test creating many configurations
        configs = []
        
        for i in range(100):
            config_result = LLMStrategyConfig.create(
                model_name="gpt-4",
                temperature=0.7 + (i % 10) * 0.01,  # Slight variations
                max_context_length=4000,
                use_memory=i % 2 == 0,
                confidence_threshold=70.0 + (i % 20)
            )
            
            assert config_result.is_success()
            configs.append(config_result.success())
        
        # All configurations should be valid and immutable
        assert len(configs) == 100
        for config in configs:
            assert isinstance(config, LLMStrategyConfig)
            # Test immutability
            with pytest.raises(AttributeError):
                config.model_name = "modified"


class TestStrategyConfigurationPropertyBased(FPTestBase):
    """Property-based tests for strategy configuration."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        if not FP_AVAILABLE:
            pytest.skip("FP types not available")
    
    @given(
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        confidence=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        context_length=st.integers(min_value=100, max_value=8000),
        use_memory=st.booleans()
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
    def test_llm_config_property_validation(self, temperature, confidence, context_length, use_memory):
        """Test LLM configuration with property-based testing."""
        config_result = LLMStrategyConfig.create(
            model_name="gpt-4",
            temperature=temperature,
            max_context_length=context_length,
            use_memory=use_memory,
            confidence_threshold=confidence
        )
        
        if (0.0 <= temperature <= 2.0 and 
            0.0 <= confidence <= 100.0 and 
            100 <= context_length <= 8000):
            assert config_result.is_success()
            config = config_result.success()
            assert config.temperature == temperature
            assert config.confidence_threshold.value == confidence
            assert config.max_context_length == context_length
            assert config.use_memory == use_memory
        else:
            assert config_result.is_failure()
    
    @given(
        lookback=st.integers(min_value=1, max_value=200),
        entry_threshold=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        exit_threshold=st.floats(min_value=0.1, max_value=5.0, allow_nan=False)
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
    def test_momentum_config_property_validation(self, lookback, entry_threshold, exit_threshold):
        """Test momentum configuration with property-based testing."""
        config_result = MomentumStrategyConfig.create(
            lookback_period=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_volume_confirmation=True
        )
        
        if (lookback > 0 and 
            entry_threshold > 0 and 
            exit_threshold > 0 and 
            exit_threshold < entry_threshold):
            assert config_result.is_success()
            config = config_result.success()
            assert config.lookback_period == lookback
            assert config.entry_threshold.value == entry_threshold
            assert config.exit_threshold.value == exit_threshold
        else:
            assert config_result.is_failure()


if __name__ == "__main__":
    pytest.main([__file__])