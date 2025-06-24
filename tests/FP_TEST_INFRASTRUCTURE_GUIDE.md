# Functional Programming Test Infrastructure Guide

This guide provides comprehensive documentation for the functional programming (FP) test infrastructure designed to support the migration from imperative to functional programming patterns in the trading bot codebase.

## Overview

The FP test infrastructure provides:

- **FP-compatible test fixtures** for Result/Maybe/IO monads
- **Migration adapters** to transition imperative tests to FP patterns
- **Specialized mock objects** for FP components
- **Test base classes** with common FP testing patterns
- **Property-based testing** utilities for FP types
- **Dual-mode testing** to run both imperative and FP versions
- **Test data generators** for immutable FP types

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Fixtures](#test-fixtures)
3. [Base Test Classes](#base-test-classes)
4. [Migration Adapters](#migration-adapters)
5. [Mock Objects](#mock-objects)
6. [Test Data Generation](#test-data-generation)
7. [Property-Based Testing](#property-based-testing)
8. [Migration Guidelines](#migration-guidelines)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Quick Start

### Using FP Test Base Classes

```python
from tests.fp_test_base import FPExchangeTestBase

class TestMyExchange(FPExchangeTestBase):
    def test_place_order_fp(self):
        # Setup
        adapter = self.create_mock_exchange_adapter()

        # Test FP method
        result = adapter.place_order(order_data)
        order_result = self.run_io(result)

        # Assert using FP assertions
        self.assert_result_ok(order_result, {"order_id": "test-123"})
```

### Using FP Fixtures

```python
def test_with_fp_fixtures(fp_result_ok, fp_market_snapshot, fp_test_utils):
    # Use FP fixtures directly
    assert fp_result_ok.is_ok()
    assert fp_market_snapshot.symbol == "BTC-USD"

    # Use FP test utilities
    fp_test_utils.assert_result_ok(fp_result_ok, 42)
```

### Migration from Imperative Tests

```python
from tests.fp_migration_adapters import fp_migration

@fp_migration()
class TestMigratedComponent(TestOriginalComponent):
    # Your existing tests will be automatically adapted for FP patterns
    pass
```

## Test Fixtures

The FP test infrastructure provides comprehensive fixtures in `tests/conftest.py`:

### Core FP Type Fixtures

```python
# Result monad fixtures
fp_result_ok          # Ok(42)
fp_result_err         # Err("Test error")

# Maybe monad fixtures
fp_maybe_some         # Some(42)
fp_maybe_nothing      # Nothing()

# IO monad fixtures
fp_io_pure            # IO.pure(42)
```

### Domain-Specific Fixtures

```python
# Market data fixtures
fp_market_snapshot    # MarketSnapshot with test data
fp_position          # Position with test data
fp_portfolio         # Portfolio with test position

# Trading signal fixtures
fp_trade_signal_long  # Long signal
fp_trade_signal_short # Short signal
fp_trade_signal_hold  # Hold signal
fp_market_make_signal # MarketMake signal

# Order fixtures
fp_limit_order        # LimitOrder with test data
fp_market_order       # MarketOrder with test data

# Type system fixtures
fp_base_types         # Money, Percentage, Symbol, TimeInterval
fp_config_types       # TradingConfig, RiskConfig
```

### Mock Object Fixtures

```python
# Mock FP components
fp_mock_exchange_adapter  # Mock exchange adapter with FP methods
fp_mock_strategy         # Mock strategy with FP methods
fp_mock_risk_manager     # Mock risk manager with FP methods
```

### Utility Fixtures

```python
# Test utilities
fp_test_utils           # FP assertion helpers and utilities
fp_property_strategies  # Property-based testing strategies

# Migration support
fp_migration_scenario   # Dual-mode test scenarios
```

## Base Test Classes

### FPTestBase

The core base class for all FP tests:

```python
from tests.fp_test_base import FPTestBase

class TestMyComponent(FPTestBase):
    def test_something(self):
        # FP assertions available
        result = Ok(42)
        self.assert_result_ok(result, 42)

        # Utility methods available
        io_result = self.run_io(IO.pure(100))
        assert io_result == 100
```

**Available Methods:**
- `assert_result_ok(result, expected_value=None)`
- `assert_result_err(result, expected_error=None)`
- `assert_maybe_some(maybe, expected_value=None)`
- `assert_maybe_nothing(maybe)`
- `assert_io_result(io, expected_value=None)`
- `run_io(io)` - Execute IO computation
- `run_io_safe(io, default=None)` - Execute with exception handling

### Specialized Base Classes

#### FPExchangeTestBase

For testing exchange components:

```python
from tests.fp_test_base import FPExchangeTestBase

class TestExchangeAdapter(FPExchangeTestBase):
    def test_get_balance(self):
        adapter = self.create_mock_exchange_adapter(balance=Decimal("5000.00"))
        snapshot = self.create_test_market_snapshot(price=Decimal("60000.00"))

        result = adapter.get_balance()
        self.assert_io_result(result, Ok(Decimal("5000.00")))
```

#### FPStrategyTestBase

For testing strategy components:

```python
from tests.fp_test_base import FPStrategyTestBase

class TestTradingStrategy(FPStrategyTestBase):
    def test_generate_signal(self):
        strategy = self.create_mock_strategy()
        long_signal = self.create_test_long_signal(confidence=0.9)

        self.assert_signal_type(long_signal, Long)
        self.assert_signal_confidence(long_signal, min_confidence=0.8)
```

#### FPRiskTestBase

For testing risk management:

```python
from tests.fp_test_base import FPRiskTestBase

class TestRiskManager(FPRiskTestBase):
    def test_validate_trade(self):
        risk_manager = self.create_mock_risk_manager()
        position = self.create_test_position(side="LONG")

        # Test risk validation
        result = risk_manager.validate_trade(position)
        self.assert_io_result(result, Ok(True))
```

#### FPIndicatorTestBase

For testing indicators:

```python
from tests.fp_test_base import FPIndicatorTestBase

class TestVuManchuIndicator(FPIndicatorTestBase):
    def test_calculate_indicator(self):
        indicator = self.create_mock_indicator("vumanchu")
        ohlcv_data = self.create_test_ohlcv_data(count=50)

        result = indicator.calculate(ohlcv_data)
        indicator_value = self.assert_result_ok(result)
        self.assert_indicator_value_range(indicator_value["rsi"], 0, 100)
```

## Migration Adapters

### Automatic Migration

Use the `@fp_migration` decorator to automatically migrate test classes:

```python
from tests.fp_migration_adapters import fp_migration, MigrationConfig

# Basic migration
@fp_migration()
class TestMigratedClass(TestOriginalClass):
    pass

# Custom migration configuration
config = MigrationConfig(
    enable_fp_mode=True,
    auto_convert_types=True,
    preserve_exceptions=False
)

@fp_migration(config)
class TestCustomMigration(TestOriginalClass):
    pass
```

### Dual-Mode Testing

Run tests in both imperative and FP modes:

```python
from tests.fp_migration_adapters import dual_mode_test

@dual_mode_test()
class TestDualMode(TestOriginalClass):
    # Original test methods will be run in both modes
    # Methods named test_*_imperative and test_*_fp will be created
    # Plus test_*_compare methods that validate consistency
    pass
```

### Manual Migration

For more control over migration:

```python
from tests.fp_migration_adapters import TestClassMigrationAdapter, TypeConversionAdapter

# Create migration adapter
adapter = TestClassMigrationAdapter()
fp_test_class = adapter.migrate_test_class(OriginalTestClass)

# Type conversions
converter = TypeConversionAdapter()
fp_result = converter.to_fp_result(imperative_value)
imperative_value = converter.from_fp_result(fp_result)
```

## Mock Objects

### FP Mock Factory

Create FP-compatible mocks:

```python
from tests.unit.fp.test_fp_infrastructure import FPMockFactory

# Exchange client mock
exchange_client = FPMockFactory.create_mock_exchange_client(supports_fp=True)
assert exchange_client.supports_functional_operations() == True

# FP adapter mock
fp_adapter = FPMockFactory.create_mock_fp_adapter()
balance_io = fp_adapter.get_balance()
assert balance_io.run().unwrap() == Decimal("10000.00")

# Strategy mock
strategy = FPMockFactory.create_mock_fp_strategy()
signal_io = strategy.generate_signal()
signal = signal_io.run().unwrap()
assert isinstance(signal, Hold)
```

### Mock Adaptation

Adapt existing mocks for FP compatibility:

```python
from tests.fp_migration_adapters import MockObjectAdapter

# Adapt existing exchange mock
original_mock = Mock()
fp_compatible_mock = MockObjectAdapter.adapt_exchange_mock(original_mock)

# Now has FP methods
fp_adapter = fp_compatible_mock.get_functional_adapter()
```

## Test Data Generation

### FP Test Data Generator

Generate immutable test data:

```python
from tests.data.fp_test_data_generator import (
    FPTestDataConfig,
    FPMarketDataGenerator,
    FPPortfolioDataGenerator,
    FPTestScenarioGenerator
)

# Configure data generation
config = FPTestDataConfig(
    base_price=50000.0,
    volatility=0.02,
    scenario_type="trending",
    time_periods=1000
)

# Generate market data
market_gen = FPMarketDataGenerator(config)
snapshots = market_gen.generate_market_snapshots(count=500)
ohlcv_data = market_gen.generate_ohlcv_series(count=500, timeframe="1m")

# Generate portfolio data
portfolio_gen = FPPortfolioDataGenerator(config)
portfolios = portfolio_gen.generate_portfolio_timeline(snapshots)
trade_results = portfolio_gen.generate_trade_results(count=100)

# Generate complete scenarios
scenario_gen = FPTestScenarioGenerator(config)
complete_scenario = scenario_gen.generate_complete_scenario()
multi_scenarios = scenario_gen.generate_multi_scenario_suite()
```

### Scenario Types

Available scenario types for testing different market conditions:

- **default**: Normal market with moderate volatility
- **trending**: Strong directional movement (bull/bear)
- **ranging**: Sideways/oscillating price action
- **volatile**: High volatility with clustering and spikes

## Property-Based Testing

### Using Hypothesis with FP Types

```python
from hypothesis import given
from tests.unit.fp.test_fp_infrastructure import FPPropertyStrategies

class TestWithProperties(FPTestBase):
    @given(FPPropertyStrategies.fp_decimal_strategy())
    def test_decimal_operations(self, decimal_value):
        # Test with generated Decimal values
        assert decimal_value >= 0

    @given(FPPropertyStrategies.fp_result_strategy())
    def test_result_operations(self, result):
        # Test with generated Result values
        if result.is_ok():
            value = result.unwrap()
            assert value is not None
        else:
            assert result.is_err()

    @given(FPPropertyStrategies.fp_market_snapshot_strategy())
    def test_market_operations(self, snapshot):
        # Test with generated MarketSnapshot values
        assert snapshot.price > 0
        assert snapshot.bid <= snapshot.ask
```

### Built-in Property Strategies

Available in `fp_property_strategies` fixture:

```python
def test_with_strategies(fp_property_strategies):
    # Use predefined strategies
    decimal_strategy = fp_property_strategies["decimal"]
    timestamp_strategy = fp_property_strategies["timestamp"]
    symbol_strategy = fp_property_strategies["symbol"]
```

## Migration Guidelines

### Step-by-Step Migration Process

1. **Assessment**: Identify tests that need FP migration
2. **Automatic Migration**: Try `@fp_migration()` decorator first
3. **Dual-Mode Testing**: Use `@dual_mode_test()` to validate consistency
4. **Manual Fixes**: Address specific compatibility issues
5. **Validation**: Ensure all tests pass in FP mode
6. **Cleanup**: Remove imperative versions when confident

### Common Migration Patterns

#### Exception Handling to Result Types

**Before (Imperative):**
```python
def test_operation_with_exception(self):
    with pytest.raises(ValueError):
        risky_operation()
```

**After (FP):**
```python
def test_operation_with_result(self):
    result = risky_operation_fp()
    self.assert_result_err(result)
    assert "ValueError" in str(result.error)
```

#### Nullable Values to Maybe Types

**Before (Imperative):**
```python
def test_optional_value(self):
    value = get_optional_value()
    if value is not None:
        assert value > 0
```

**After (FP):**
```python
def test_optional_value_fp(self):
    maybe_value = get_optional_value_fp()
    if maybe_value.is_some():
        self.assert_maybe_some(maybe_value)
        assert maybe_value.unwrap() > 0
```

#### Async Operations to IO Types

**Before (Imperative):**
```python
async def test_async_operation(self):
    result = await async_operation()
    assert result == expected_value
```

**After (FP):**
```python
def test_io_operation(self):
    io_result = async_operation_fp()
    result = self.run_io(io_result)
    self.assert_result_ok(result, expected_value)
```

### Compatibility Considerations

- **Type Conversions**: Use `TypeConversionAdapter` for seamless conversions
- **Mock Adaptations**: Use `MockObjectAdapter` to make existing mocks FP-compatible
- **Gradual Migration**: Migrate one test class at a time
- **Validation**: Run dual-mode tests to ensure consistency

## Best Practices

### Test Organization

1. **Inherit from appropriate base classes** (`FPTestBase`, `FPExchangeTestBase`, etc.)
2. **Use FP fixtures** for consistent test data
3. **Group related tests** in FP-specific modules
4. **Separate FP and imperative tests** during migration period

### Assertion Patterns

```python
# Prefer FP-specific assertions
self.assert_result_ok(result, expected_value)
self.assert_maybe_some(maybe_value, expected_value)
self.assert_io_result(io_computation, expected_value)

# Use context managers for complex testing
with self.fp_test_context(strict_mode=True):
    result = complex_operation()
    self.assert_result_ok(result)
```

### Mock Usage

```python
# Create FP-compatible mocks
mock_adapter = self.create_mock_exchange_adapter()
mock_strategy = self.create_mock_strategy()

# Register mocks for cleanup
self.register_mock("adapter", mock_adapter)

# Use context managers for temporary mocking
with self.mock_fp_dependencies(
    "bot.exchange.adapter": mock_adapter
):
    # Test code here
    pass
```

### Error Handling

```python
# Test both success and failure cases
def test_operation_success(self):
    result = operation()
    self.assert_result_ok(result)

def test_operation_failure(self):
    result = operation_with_invalid_input()
    self.assert_result_err(result, "Expected error message")
```

## Examples

### Complete Test Class Migration

**Before (Imperative):**
```python
class TestExchangeClient:
    def setUp(self):
        self.client = ExchangeClient()
        self.mock_api = Mock()

    async def test_get_balance(self):
        self.mock_api.get_account.return_value = {"balance": 1000.0}
        balance = await self.client.get_balance()
        assert balance == 1000.0

    def test_invalid_symbol(self):
        with pytest.raises(ValueError):
            self.client.validate_symbol("INVALID")
```

**After (FP):**
```python
@fp_migration()
class TestExchangeClientFP(FPExchangeTestBase):
    def setup_test_fixtures(self):
        super().setup_test_fixtures()
        self.adapter = self.create_mock_exchange_adapter()

    def test_get_balance_fp(self):
        # FP version returns IO[Result[Decimal, str]]
        balance_io = self.adapter.get_balance()
        balance_result = self.run_io(balance_io)
        self.assert_result_ok(balance_result, Decimal("10000.00"))

    def test_invalid_symbol_fp(self):
        # FP version returns Result instead of raising
        result = self.client.validate_symbol_fp("INVALID")
        self.assert_result_err(result, "Invalid symbol format")
```

### Property-Based Testing Example

```python
from hypothesis import given
from tests.unit.fp.test_fp_infrastructure import FPPropertyStrategies

class TestMarketDataProperties(FPTestBase):
    @given(FPPropertyStrategies.fp_market_snapshot_strategy())
    def test_market_snapshot_invariants(self, snapshot):
        """Test that market snapshots maintain invariants."""
        # Price relationships
        assert snapshot.bid <= snapshot.price <= snapshot.ask
        assert snapshot.price > 0
        assert snapshot.volume >= 0

        # Spread calculation
        spread = snapshot.ask - snapshot.bid
        assert spread >= 0

    @given(
        FPPropertyStrategies.fp_decimal_strategy(min_value=1000, max_value=100000),
        FPPropertyStrategies.fp_decimal_strategy(min_value=0.01, max_value=1.0)
    )
    def test_position_pnl_calculation(self, entry_price, size):
        """Test P&L calculation properties."""
        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=size,
            entry_price=entry_price,
            current_price=entry_price * Decimal("1.05")  # 5% profit
        )

        # P&L should be positive for profitable long position
        assert position.unrealized_pnl > 0

        # P&L should be proportional to size
        expected_pnl = size * (position.current_price - entry_price)
        assert position.unrealized_pnl == expected_pnl
```

### Dual-Mode Testing Example

```python
@dual_mode_test()
class TestStrategyDualMode(TestOriginalStrategy):
    """
    This creates:
    - test_generate_signal_imperative()
    - test_generate_signal_fp()
    - test_generate_signal_compare()

    The compare method validates that both modes produce equivalent results.
    """

    def test_signal_generation_consistency(self):
        """Custom test to verify consistency between modes."""
        # This test will automatically be migrated to both modes
        signal = self.strategy.generate_signal(market_data)
        assert signal.confidence > 0.5
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure FP types are available with fallbacks
2. **Mock Incompatibility**: Use `MockObjectAdapter` to adapt existing mocks
3. **Type Mismatches**: Use `TypeConversionAdapter` for conversions
4. **Async/IO Confusion**: Use `run_io()` helper for IO computations

### Debug Helpers

```python
# Check FP availability
if not FP_AVAILABLE:
    pytest.skip("FP types not available")

# Debug type conversions
converter = TypeConversionAdapter()
print(f"Original: {value}, FP: {converter.to_fp_result(value)}")

# Migration reporting
with migration_reporting() as registry:
    # Run migrations
    report = registry.get_migration_report()
    print(f"Success rate: {report['success_rate']:.2%}")
```

## Conclusion

The FP test infrastructure provides comprehensive support for migrating from imperative to functional programming patterns while maintaining test coverage and reliability. Use the provided base classes, fixtures, and migration tools to ensure a smooth transition to FP-based testing.

For questions or issues, refer to the source code in:
- `tests/conftest.py` - Test fixtures
- `tests/fp_test_base.py` - Base test classes
- `tests/fp_migration_adapters.py` - Migration utilities
- `tests/unit/fp/test_fp_infrastructure.py` - FP test utilities
- `tests/data/fp_test_data_generator.py` - Test data generation
