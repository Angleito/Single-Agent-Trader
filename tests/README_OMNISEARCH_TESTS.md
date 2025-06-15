# OmniSearch MCP Integration Tests

This directory contains comprehensive tests for the OmniSearch MCP (Model Context Protocol) integration in the AI trading bot. The test suite ensures the reliability, performance, and correctness of the OmniSearch components.

## 📁 Test Structure

```
tests/
├── conftest.py                           # Shared pytest fixtures and configuration
├── requirements-test.txt                 # Additional test dependencies
├── unit/                                 # Unit tests for individual components
│   ├── test_omnisearch_client.py        # OmniSearch client functionality
│   ├── test_financial_sentiment.py      # Financial sentiment service
│   ├── test_market_context.py           # Market context analyzer
│   └── test_web_search_formatter.py     # Web search result formatter
├── integration/                          # Integration tests
│   └── test_omnisearch_integration.py   # End-to-end OmniSearch integration
└── performance/                          # Performance benchmarks
```

## 🧪 Test Components

### Unit Tests

#### OmniSearch Client (`test_omnisearch_client.py`)
- **Connection Management**: Tests for connecting/disconnecting from OmniSearch API
- **Search Methods**: Financial news, crypto sentiment, NASDAQ sentiment, market correlation
- **Caching**: Cache hit/miss scenarios, TTL expiration, cache invalidation
- **Rate Limiting**: Request throttling, rate limit enforcement, backoff strategies
- **Error Handling**: Network failures, API errors, timeout handling
- **Data Models**: Validation of SearchResult, SentimentAnalysis, MarketCorrelation models

#### Financial Sentiment Service (`test_financial_sentiment.py`)
- **News Processing**: Sentiment analysis from news articles
- **Indicator Extraction**: Crypto and NASDAQ market indicators
- **Correlation Calculation**: Cross-asset sentiment correlation
- **LLM Formatting**: Output formatting for AI agent consumption
- **Theme Extraction**: Key market themes identification
- **Signal Generation**: Trading signal extraction from sentiment

#### Market Context Analyzer (`test_market_context.py`)
- **Correlation Analysis**: Crypto-NASDAQ correlation with statistical significance
- **Regime Detection**: Market regime classification (RISK_ON, RISK_OFF, TRANSITION)
- **Risk Sentiment**: Fear & greed index calculation and sentiment classification
- **Momentum Alignment**: Cross-asset momentum analysis
- **Context Summary**: Comprehensive market context formatting

#### Web Search Formatter (`test_web_search_formatter.py`)
- **Content Formatting**: News results formatting for LLM consumption
- **Token Optimization**: Content truncation and optimization for token limits
- **Priority Scoring**: Content relevance, freshness, authority scoring
- **Insight Extraction**: Key insights from search results
- **Error Handling**: Graceful degradation for malformed data

### Integration Tests

#### OmniSearch Integration (`test_omnisearch_integration.py`)
- **End-to-End Workflow**: Complete data flow from OmniSearch to LLM agent
- **Component Integration**: Inter-component communication and data passing
- **Error Resilience**: System behavior under component failures
- **Performance Testing**: Response times and throughput under load
- **Configuration Testing**: Environment variable handling and settings

## 🚀 Running Tests

### Prerequisites

```bash
# Install dependencies
poetry install

# Install additional test dependencies
pip install -r tests/requirements-test.txt
```

### Quick Start

```bash
# Run all OmniSearch tests
python scripts/run_omnisearch_tests.py --all --verbose --coverage

# Run specific test suites
python scripts/run_omnisearch_tests.py --unit           # Unit tests only
python scripts/run_omnisearch_tests.py --integration   # Integration tests only
python scripts/run_omnisearch_tests.py --omnisearch    # OmniSearch-specific tests
```

### Detailed Test Execution

#### Unit Tests
```bash
# Run all unit tests with coverage
pytest tests/unit/ --cov=bot --cov-report=html -v

# Run specific component tests
pytest tests/unit/test_omnisearch_client.py -v
pytest tests/unit/test_financial_sentiment.py -v
pytest tests/unit/test_market_context.py -v
pytest tests/unit/test_web_search_formatter.py -v
```

#### Integration Tests
```bash
# Run integration tests
pytest tests/integration/test_omnisearch_integration.py -v

# Run with external API mocking
pytest tests/integration/test_omnisearch_integration.py -v -m "not external"
```

#### Performance Tests
```bash
# Run performance benchmarks
pytest tests/ -m slow --benchmark-only

# Profile memory usage
pytest tests/ --profile
```

### Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.external` - Tests requiring external services
- `@pytest.mark.omnisearch` - OmniSearch-specific tests

## 📊 Test Coverage

### Coverage Requirements
- **Minimum Coverage**: 85% overall
- **Critical Components**: 95% coverage required
  - OmniSearch client
  - Financial sentiment service
  - Market context analyzer
  - Web search formatter

### Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=bot --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🔧 Test Configuration

### Environment Variables
```bash
# Testing configuration
export OMNISEARCH_API_KEY="test_key_12345"
export OMNISEARCH_SERVER_URL="https://test-api.omnisearch.dev"
export TEST_SKIP_EXTERNAL_APIS="true"
export TEST_USE_MOCK_RESPONSES="true"
```

### pytest.ini Configuration
The tests use the following pytest configuration:
```ini
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
asyncio_mode = "auto"
```

## 🧩 Test Fixtures

### Common Fixtures (conftest.py)
- `sample_financial_news` - Realistic financial news data
- `sample_bullish_sentiment` - Bullish market sentiment data
- `sample_correlation_data` - Market correlation analysis data
- `mock_omnisearch_client` - Mocked OmniSearch client
- `mock_aiohttp_session` - Mocked HTTP session
- `temp_directory` - Temporary directory for testing

### Performance Fixtures
- `performance_timer` - Timing utilities for benchmarks
- `benchmark_timer` - Context manager for performance measurement

## 🐛 Debugging Tests

### Verbose Output
```bash
# Enable verbose output and logging
pytest tests/ -v -s --log-cli-level=DEBUG

# Show local variables in tracebacks
pytest tests/ --tb=long -vv
```

### Test Debugging
```bash
# Run specific test with debugging
pytest tests/unit/test_omnisearch_client.py::TestOmniSearchClient::test_connect_success -v -s

# Debug async tests
pytest tests/ --asyncio-mode=auto -v
```

## 📈 Performance Benchmarks

### Expected Performance Metrics
- **OmniSearch API calls**: < 500ms average response time
- **Sentiment analysis**: < 200ms for 10 news items
- **Correlation calculation**: < 100ms for 200 data points
- **Content formatting**: < 50ms for standard content

### Running Benchmarks
```bash
# Run performance benchmarks
python scripts/run_omnisearch_tests.py --performance

# Custom benchmark thresholds
pytest tests/ --benchmark-min-rounds=5 --benchmark-warmup=true
```

## 🚨 Error Handling Tests

### Network Error Simulation
Tests include comprehensive error handling for:
- Connection timeouts
- HTTP error codes (404, 500, 429)
- JSON parsing errors
- Rate limit exceeded
- API service unavailable

### Fallback Behavior
Tests verify graceful fallback behavior:
- Cached data when API unavailable
- Default neutral sentiment when analysis fails
- Empty results when no data available
- Reduced functionality with partial failures

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
# Example CI configuration
- name: Run OmniSearch Tests
  run: |
    poetry install
    python scripts/run_omnisearch_tests.py --all --coverage
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📝 Writing New Tests

### Test Structure Guidelines
1. **Arrange**: Set up test data and mocks
2. **Act**: Execute the function under test
3. **Assert**: Verify the expected outcomes

### Example Test
```python
@pytest.mark.asyncio
async def test_omnisearch_client_search_news(mock_omnisearch_client):
    """Test financial news search functionality."""
    # Arrange
    expected_news = [
        FinancialNewsResult(
            base_result=SearchResult(
                title="Bitcoin Surges",
                url="https://example.com",
                snippet="Bitcoin price increases",
                source="example.com"
            )
        )
    ]
    mock_omnisearch_client.search_financial_news.return_value = expected_news
    
    # Act
    results = await mock_omnisearch_client.search_financial_news("bitcoin")
    
    # Assert
    assert len(results) == 1
    assert results[0].base_result.title == "Bitcoin Surges"
```

### Mock Usage Guidelines
- Use `AsyncMock` for async methods
- Mock external dependencies (HTTP calls, file I/O)
- Verify mock call counts and arguments
- Reset mocks between tests

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure Python path includes bot module
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Install missing dependencies
poetry install
pip install -r tests/requirements-test.txt
```

#### Async Test Issues
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Use correct async test decorator
@pytest.mark.asyncio
async def test_async_function():
    pass
```

#### Mock Issues
```bash
# Use appropriate mock types
from unittest.mock import AsyncMock, Mock

# For async methods
mock_method = AsyncMock(return_value=expected_result)

# For sync methods  
mock_method = Mock(return_value=expected_result)
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [OmniSearch API Documentation](https://docs.omnisearch.dev/)

## 🤝 Contributing

When adding new tests:
1. Follow existing test patterns and naming conventions
2. Include both positive and negative test cases
3. Add appropriate test markers
4. Update this README if adding new test categories
5. Ensure tests are deterministic and isolated
6. Add performance tests for critical paths

## 📞 Support

For issues with OmniSearch integration tests:
1. Check the test output for specific error messages
2. Verify all dependencies are installed correctly
3. Ensure environment variables are set properly
4. Review the mock configurations for external services
5. Check the GitHub Issues for known problems