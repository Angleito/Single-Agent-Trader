# Property-Based Testing Guide

This guide covers the comprehensive property-based testing framework implemented for the AI trading bot, focusing on orderbook data validation, market data processing, and configuration validation using Hypothesis.

## Overview

Property-based testing validates that your code maintains certain properties across a wide range of inputs, rather than testing specific examples. This approach is particularly valuable for financial systems where data integrity and mathematical correctness are critical.

## Test Modules

### 1. Orderbook Properties (`test_orderbook_properties.py`)

Tests orderbook data structure validation and invariants:

- **Bid/Ask Ordering**: Ensures bids are in descending order, asks in ascending order
- **Spread Calculation**: Validates positive spreads and correct mid-price calculation
- **Depth Calculation**: Tests volume aggregation properties
- **VWAP Calculation**: Validates volume-weighted average price computations
- **Price Impact**: Tests market order impact calculations

**Key Properties Tested:**
```python
# Bid prices must be in strictly descending order
@given(valid_orderbook_strategy())
def test_bid_prices_descending_invariant(self, orderbook: OrderBook):
    for i in range(len(orderbook.bids) - 1):
        assert orderbook.bids[i][0] > orderbook.bids[i + 1][0]

# Spread must always be positive
@given(valid_orderbook_strategy())
def test_positive_spread_invariant(self, orderbook: OrderBook):
    if orderbook.bids and orderbook.asks:
        assert orderbook.spread > 0
```

### 2. Market Data Validation (`test_market_data_validation_properties.py`)

Tests market data structures and calculations:

- **Price/Volume Validation**: Ensures positive prices and non-negative volumes
- **OHLCV Relationships**: Validates high ≥ max(open, close) and low ≤ min(open, close)
- **Spread Calculations**: Tests bid-ask spread properties
- **Technical Indicators**: Validates indicator relationships and bounds
- **Trade Classification**: Tests buy/sell classification logic

**Key Properties Tested:**
```python
# OHLCV relationships must be valid
@given(candle_strategy())
def test_ohlc_relationships_invariant(self, candle: Candle):
    assert candle.high >= candle.open
    assert candle.high >= candle.close
    assert candle.low <= candle.open
    assert candle.low <= candle.close

# VWAP should be within price range
@given(st.lists(trade_strategy(), min_size=1))
def test_vwap_within_range(self, trades: List[Trade]):
    min_price = min(trade.price for trade in trades)
    max_price = max(trade.price for trade in trades)
    vwap = calculate_vwap(trades)
    assert min_price <= vwap <= max_price
```

### 3. Stateful Orderbook Testing (`test_stateful_orderbook.py`)

Tests complex orderbook state transitions and invariants:

- **State Machine Testing**: Models orderbook updates as state transitions
- **Invariant Preservation**: Ensures invariants hold across all state changes
- **Complex Scenarios**: Tests sequences of adds, removes, and updates
- **Concurrent Access Simulation**: Tests performance under simulated load

**Key Features:**
```python
class OrderBookStateMachine(RuleBasedStateMachine):
    @rule()
    def add_bid_level(self):
        # Add new bid while maintaining ordering

    @rule()
    def remove_top_bid(self):
        # Remove best bid and verify invariants

    @invariant()
    def orderbook_always_valid(self):
        # Ensure orderbook remains valid after every operation
        assert self.orderbook.spread > 0
        # Check bid/ask ordering
```

### 4. Configuration Validation (`test_configuration_validation_properties.py`)

Tests the functional programming configuration system:

- **API Key Validation**: Tests key length requirements and masking
- **Private Key Security**: Validates complete masking of sensitive data
- **Parameter Validation**: Tests range checks and type safety
- **Strategy Configuration**: Validates complex configuration objects
- **Security Properties**: Ensures no sensitive data leakage

**Key Properties Tested:**
```python
# API keys should be properly masked
@given(api_key_strategy(valid=True))
def test_api_key_masking_security_property(self, key_string: str):
    result = APIKey.create(key_string)
    api_key = result.success()
    assert key_string not in str(api_key)  # Full key never exposed
    assert key_string[-4:] in str(api_key)  # Last 4 chars visible

# Private keys should be completely hidden
@given(private_key_strategy(valid=True))
def test_private_key_complete_masking(self, key_string: str):
    result = PrivateKey.create(key_string)
    private_key = result.success()
    assert str(private_key) == "PrivateKey(***)"
```

### 5. Performance Properties (`test_performance_properties.py`)

Tests algorithmic complexity and performance characteristics:

- **Time Complexity**: Validates O(1) and O(n) operation performance
- **Memory Usage**: Tests memory scaling properties
- **Large Data Handling**: Validates performance with large datasets
- **Concurrent Access**: Tests performance under load
- **Algorithmic Bounds**: Ensures operations complete within time limits

**Key Properties Tested:**
```python
# Core operations should be O(1)
@given(large_orderbook_strategy())
def test_orderbook_operation_complexity(self, orderbook: OrderBook):
    with PerformanceTimer() as timer:
        for _ in range(1000):
            _ = orderbook.best_bid
            _ = orderbook.spread
    assert timer.duration < 0.1  # Should be very fast

# Memory usage should scale predictably
@given(st.integers(min_value=10, max_value=1000))
def test_memory_scaling(self, num_levels: int):
    orderbook = create_test_orderbook(num_levels)
    memory_per_level = get_object_size(orderbook) / (num_levels * 2)
    assert memory_per_level < 1000  # Reasonable memory usage
```

## Running Property Tests

### Basic Execution

```bash
# Run all property tests
python run_property_tests.py

# Run with verbose output
python run_property_tests.py --verbose

# Quick run with fewer examples
python run_property_tests.py --quick

# Save detailed report
python run_property_tests.py --save-report
```

### Using Poetry

```bash
# Run property tests through poetry
poetry run python run_property_tests.py

# Run specific test module
poetry run pytest tests/property/test_orderbook_properties.py -v

# Run with hypothesis statistics
poetry run pytest tests/property/ --hypothesis-show-statistics
```

### Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Run Property-Based Tests
  run: |
    poetry run python run_property_tests.py --save-report
    # Fail if pass rate < 95%
    if [ $? -gt 1 ]; then exit 1; fi
```

## Test Strategies and Generators

### Custom Hypothesis Strategies

The framework includes custom strategies for generating realistic financial data:

```python
@st.composite
def price_strategy(draw, min_price=0.01, max_price=100000.0):
    """Generate realistic price values as Decimal."""
    price = draw(st.floats(min_value=min_price, max_value=max_price))
    return Decimal(str(round(price, 8)))

@st.composite
def valid_orderbook_strategy(draw):
    """Generate valid orderbook with proper bid/ask ordering."""
    bids = draw(sorted_bids_strategy())
    asks = draw(sorted_asks_strategy(min_price=bids[0][0] + Decimal('0.01')))
    return OrderBook(bids=bids, asks=asks, timestamp=datetime.now(UTC))
```

### Fuzzing Strategies

For edge case testing:

```python
# Test with invalid data
@given(
    st.lists(price_size_tuple_strategy(), max_size=0),  # Empty orderbook
    st.floats(min_value=-1000.0, max_value=0.0),       # Negative prices
)
def test_invalid_data_handling(self, empty_data, negative_price):
    with pytest.raises(ValueError):
        # Should raise appropriate errors
```

## Configuration and Settings

### Hypothesis Configuration

```python
# Configure for thorough testing
hypothesis.settings.register_profile(
    "comprehensive",
    max_examples=500,           # More examples for better coverage
    deadline=10000,             # 10 second timeout
    stateful_step_count=100,    # More steps for stateful tests
    suppress_health_check=[     # Suppress noisy health checks
        hypothesis.HealthCheck.too_slow,
    ]
)
```

### Custom Settings for Different Environments

```python
# Development: Fast feedback
hypothesis.settings.register_profile("dev", max_examples=50, deadline=2000)

# CI: Thorough testing
hypothesis.settings.register_profile("ci", max_examples=200, deadline=15000)

# Nightly: Exhaustive testing
hypothesis.settings.register_profile("nightly", max_examples=1000, deadline=60000)
```

## Interpreting Results

### Test Report Structure

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "total_modules": 5,
  "total_tests": 45,
  "overall_pass_rate": 97.8,
  "execution_time": 45.2,
  "module_results": [
    {
      "module_name": "orderbook_properties",
      "total_tests": 12,
      "passed": 12,
      "failed": 0,
      "coverage_percentage": 100.0,
      "execution_time": 8.5
    }
  ]
}
```

### Understanding Property Test Failures

1. **Falsifying Examples**: Hypothesis provides minimal failing examples
2. **Shrinking**: Automatically reduces complex failing cases to simpler ones
3. **Reproduction**: Failed examples can be reproduced deterministically

Example failure output:
```
Falsifying example: test_positive_spread_invariant(
    orderbook=OrderBook(
        bids=[(Decimal('100.00'), Decimal('10.0'))],
        asks=[(Decimal('99.99'), Decimal('5.0'))],  # Invalid: ask < bid
        timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    )
)
```

## Best Practices

### 1. Property Selection

Choose properties that:
- Are always true for valid inputs
- Are easy to verify computationally
- Cover important business logic
- Test edge cases and boundaries

### 2. Strategy Design

- Generate realistic data distributions
- Include edge cases and boundary conditions
- Use composite strategies for complex objects
- Ensure generated data satisfies preconditions

### 3. Performance Considerations

- Use `@settings(deadline=...)` for time-sensitive tests
- Profile slow generators and optimize them
- Use `assume()` sparingly to avoid filtering too many examples
- Consider memory usage for large generated objects

### 4. Debugging Failed Properties

```python
# Add debugging information
@given(orderbook_strategy())
def test_property(self, orderbook):
    print(f"Testing orderbook with {len(orderbook.bids)} bids, {len(orderbook.asks)} asks")
    # Test logic here
```

### 5. Integration with Regular Tests

Property tests complement but don't replace regular unit tests:

```python
# Unit test: Specific example
def test_orderbook_with_known_values():
    orderbook = OrderBook(bids=[(100, 10)], asks=[(101, 5)], timestamp=now)
    assert orderbook.spread == 1

# Property test: General property
@given(valid_orderbook_strategy())
def test_spread_always_positive(self, orderbook):
    assert orderbook.spread > 0
```

## Extending the Framework

### Adding New Property Tests

1. Create test class in appropriate module
2. Define custom strategies for your data types
3. Implement property test methods with `@given` decorator
4. Add to test suite runner

Example:
```python
class TestNewFeatureProperties:
    @given(your_custom_strategy())
    @settings(max_examples=100)
    def test_your_property(self, generated_data):
        # Test your property here
        assert some_invariant(generated_data)
```

### Custom Generators

```python
@st.composite
def your_domain_strategy(draw):
    # Generate domain-specific test data
    field1 = draw(st.integers(min_value=1, max_value=100))
    field2 = draw(st.text(min_size=5, max_size=20))
    return YourDomainObject(field1=field1, field2=field2)
```

## Troubleshooting

### Common Issues

1. **Too Many Filtered Examples**
   - Reduce use of `assume()`
   - Design better generators
   - Use `filter()` strategy method

2. **Slow Test Execution**
   - Reduce `max_examples`
   - Optimize generators
   - Use `deadline` setting

3. **Non-Deterministic Failures**
   - Set hypothesis database
   - Use fixed seeds for debugging
   - Check for time-dependent logic

### Performance Optimization

```python
# Optimize generators
@st.composite
def optimized_strategy(draw):
    # Pre-generate common values
    common_prices = [Decimal('100.0'), Decimal('101.0'), Decimal('99.0')]
    price = draw(st.sampled_from(common_prices))
    return price

# Use caching for expensive computations
@lru_cache(maxsize=1000)
def expensive_computation(value):
    return complex_calculation(value)
```

## Conclusion

Property-based testing provides robust validation for financial trading systems by:

- Testing thousands of generated examples automatically
- Finding edge cases that manual testing might miss
- Providing strong guarantees about system invariants
- Complementing traditional unit testing approaches

The framework implemented here covers critical areas like orderbook validation, market data processing, configuration security, and performance characteristics, providing confidence in the system's reliability and correctness.
