# Integration Tests for AI Trading Bot

This directory contains comprehensive integration tests for the complete AI trading bot system. These tests validate that all components work together correctly and handle various scenarios including normal operations, error conditions, and edge cases.

## Test Files Overview

### 1. `test_complete_trading_flow.py`
**Purpose**: End-to-end trading flow validation

**Key Tests**:
- Complete trading cycle: data → indicators → LLM → validation → risk → execution
- Position tracking and P&L calculations
- Multiple trading cycles with state consistency
- Error recovery and fallback mechanisms
- Graceful shutdown integration

**Test Scenarios**:
- Long position opening and closing
- Short position trading
- Position tracking through price changes
- Market data connection recovery
- Risk management integration in complete flow
- Multiple consecutive trading cycles

### 2. `test_component_integration.py`
**Purpose**: Integration between major bot components

**Key Tests**:
- MarketDataProvider + IndicatorCalculator integration
- LLMAgent + Validator integration
- PositionManager + OrderManager integration
- RiskManager with real position data
- Exchange client order flow integration

**Test Scenarios**:
- Data flow from market data to indicators
- LLM output validation and error handling
- Position and order state synchronization
- Risk management with various trade scenarios
- Error propagation between components

### 3. `test_startup_validation.py`
**Purpose**: Configuration and startup process validation

**Key Tests**:
- Configuration loading from files
- Environment variable override handling
- Trading profile switching
- Complete startup sequence validation
- Health monitoring initialization

**Test Scenarios**:
- Valid and invalid configuration files
- Environment variable precedence
- Different trading profiles (conservative, moderate, aggressive)
- Missing required environment variables
- Startup failure recovery
- Dry run mode enforcement

### 4. `test_error_handling.py`
**Purpose**: Error handling and failure recovery

**Key Tests**:
- API connection failures
- LLM service outages
- Invalid data handling
- Order execution failures
- Network timeout recovery
- Graceful degradation scenarios

**Test Scenarios**:
- Market data API failures
- Exchange API connectivity issues
- LLM service unavailability
- Corrupted or invalid market data
- Order rejection and partial fills
- Concurrent operation errors
- System shutdown during active operations

### 5. `test_strategy_flow.py` (Existing)
**Purpose**: Strategy decision flow integration

**Key Tests**:
- Complete decision-making flow
- Indicator calculation integration
- Risk management integration
- Error handling in strategy flow

## Running the Tests

### Run All Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/integration/test_complete_trading_flow.py -v
```

### Run with Coverage
```bash
python -m pytest tests/integration/ --cov=bot --cov-report=html
```

### Run Specific Test Method
```bash
python -m pytest tests/integration/test_complete_trading_flow.py::TestCompleteTradingFlow::test_complete_trading_cycle_long_to_close -v
```

## Test Configuration

### Mock Dependencies
All tests use mocked external dependencies to ensure:
- **No real trading**: All exchange operations are mocked
- **No external API calls**: Market data and LLM services are mocked
- **Deterministic results**: Tests use fixed data and predictable responses
- **Fast execution**: No network delays or external service dependencies

### Test Data
Tests use realistic mock data including:
- **Market Data**: OHLCV data with realistic price movements
- **LLM Responses**: Various trading decisions for different market conditions
- **Order Responses**: Successful, failed, and partial order executions
- **Configuration Files**: Valid and invalid configuration scenarios

## Test Coverage Areas

### Core Trading Flow
- ✅ Market data ingestion and processing
- ✅ Technical indicator calculation
- ✅ LLM-based decision making
- ✅ Trade validation and risk management
- ✅ Order execution and position tracking
- ✅ P&L calculation and monitoring

### Component Integration
- ✅ Data flow between components
- ✅ State synchronization
- ✅ Error propagation and handling
- ✅ Configuration sharing
- ✅ Resource management

### Error Handling
- ✅ API connection failures
- ✅ Invalid data scenarios
- ✅ Service outages
- ✅ Network timeouts
- ✅ Order execution failures
- ✅ System recovery mechanisms

### Configuration Management
- ✅ File-based configuration loading
- ✅ Environment variable handling
- ✅ Profile switching
- ✅ Validation and error reporting
- ✅ Startup sequence validation

## Key Testing Principles

### 1. Comprehensive Coverage
Tests cover normal operations, edge cases, and error scenarios to ensure robust system behavior.

### 2. Realistic Scenarios
Mock data and responses are based on real-world trading scenarios and market conditions.

### 3. Isolated Testing
Each test is independent and doesn't rely on external services or previous test state.

### 4. Performance Validation
Tests include scenarios with large datasets and concurrent operations.

### 5. Security Considerations
Tests validate that sensitive information is properly handled and dry-run mode is enforced.

## Mock Strategy

### External Services
- **Market Data APIs**: Mocked with realistic OHLCV data
- **Exchange APIs**: Mocked with various order execution scenarios
- **LLM Services**: Mocked with predefined trading decisions

### Internal Components
- **File System**: Temporary directories for configuration testing
- **Network**: Simulated connection failures and timeouts
- **Time**: Fixed timestamps for deterministic testing

## Assertions and Validations

### State Consistency
Tests verify that:
- Position state matches order execution results
- P&L calculations are accurate
- Component states remain synchronized

### Error Recovery
Tests ensure that:
- System recovers gracefully from failures
- Fallback mechanisms work correctly
- No data corruption occurs during errors

### Configuration Integrity
Tests validate that:
- Configuration loading works correctly
- Environment variables take precedence
- Invalid configurations are rejected

## Continuous Integration

These integration tests are designed to run in CI/CD environments with:
- **No external dependencies**: All services are mocked
- **Fast execution**: Tests complete quickly without network delays
- **Deterministic results**: No flaky tests due to external factors
- **Comprehensive reporting**: Detailed test results and coverage reports

## Adding New Tests

When adding new integration tests:

1. **Follow existing patterns**: Use the same mocking and assertion strategies
2. **Test realistic scenarios**: Base tests on real trading situations
3. **Include error cases**: Test both success and failure paths
4. **Mock external dependencies**: Never make real API calls
5. **Document test purpose**: Add clear docstrings explaining what's being tested

## Dependencies

The integration tests require:
- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `unittest.mock`: Mocking framework
- `pandas`: Data manipulation for test data
- `numpy`: Numerical operations for realistic data generation

All bot components and their dependencies are automatically included through the bot package imports.