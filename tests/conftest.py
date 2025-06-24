"""Pytest configuration and shared fixtures for OmniSearch integration tests."""

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import numpy as np
import pytest

# Import test dependencies
pytest_plugins = ("pytest_asyncio",)


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )
    config.addinivalue_line("markers", "omnisearch: mark test as OmniSearch related")
    config.addinivalue_line(
        "markers", "requires_omnisearch: mark test as requiring OmniSearch API"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Automatically mark tests based on their location."""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add omnisearch marker to omnisearch tests
        if "omnisearch" in str(item.fspath):
            item.add_marker(pytest.mark.omnisearch)


# Event loop policy fixture (let pytest-asyncio handle event_loop automatically)
@pytest.fixture
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Get the event loop policy."""
    return asyncio.get_event_loop_policy()


# Mock data fixtures
@pytest.fixture
def sample_timestamps() -> list[datetime]:
    """Generate sample timestamps for testing."""
    base_time = datetime.now(UTC)
    return [base_time - timedelta(hours=i) for i in range(24)]


@pytest.fixture
def sample_price_data() -> dict[str, Any]:
    """Generate realistic sample price data."""
    rng = np.random.default_rng(42)
    base_price = 50000.0
    prices = [base_price]

    for _i in range(199):
        change = rng.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    return {
        "prices": prices,
        "ohlcv": [
            {
                "open": price * 0.998,
                "high": price * 1.003,
                "low": price * 0.997,
                "close": price,
                "volume": int(rng.uniform(1000000, 5000000)),
            }
            for price in prices[-50:]  # Last 50 candles
        ],
    }


@pytest.fixture
def sample_nasdaq_data() -> dict[str, Any]:
    """Generate realistic NASDAQ sample data."""
    rng = np.random.default_rng(24)
    base_price = 15000.0
    prices = [base_price]

    for _i in range(199):
        change = rng.normal(0.0005, 0.012)  # 0.05% mean, 1.2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    return {
        "prices": prices,
        "candles": [
            {
                "open": price * 0.999,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": int(rng.uniform(500000, 2000000)),
            }
            for price in prices[-50:]
        ],
    }


@pytest.fixture
def sample_financial_news() -> list[dict[str, Any]]:
    """Sample financial news items for testing."""
    return [
        {
            "title": "Bitcoin Breaks Through $60K Resistance on ETF Momentum",
            "content": """
            Bitcoin surged past the critical $60,000 resistance level today, driven by
            unprecedented institutional demand following multiple ETF approvals. The
            breakout was accompanied by a 45% surge in trading volume, with technical
            indicators showing strong bullish momentum. RSI reached 72, indicating
            overbought conditions, while MACD confirmed the bullish crossover.

            Market analysts point to growing institutional adoption as the primary
            driver, with major corporations adding Bitcoin to their treasury reserves.
            Support levels are now established at $55,000 and $57,500, with next
            resistance targets at $65,000 and $68,000.
            """,
            "url": "https://bloomberg.com/bitcoin-etf-surge",
            "source": "bloomberg.com",
            "published_time": datetime.now(UTC),
            "sentiment": "positive",
            "mentioned_symbols": ["BTC", "BTCUSD"],
            "category": "regulation",
            "impact_level": "high",
            "relevance_score": 0.95,
        },
        {
            "title": "Ethereum Network Upgrade Reduces Gas Fees by 40%",
            "content": """
            Ethereum's latest network upgrade has successfully implemented scaling
            improvements that reduce transaction costs by an average of 40%. The
            upgrade has triggered a renaissance in DeFi activity, with Total Value
            Locked (TVL) increasing 25% to $48 billion within 48 hours.

            Smart contract deployments surged 60% as developers return to building
            on Ethereum mainnet. Major institutions are exploring DeFi integration,
            with Goldman Sachs reportedly piloting tokenized asset programs.

            Technical analysis shows ETH forming a bull flag pattern with strong
            support at $2,850. Volume indicators suggest an accumulation phase
            with institutional interest growing rapidly.
            """,
            "url": "https://coindesk.com/ethereum-upgrade-gas-fees",
            "source": "coindesk.com",
            "published_time": datetime.now(UTC) - timedelta(hours=2),
            "sentiment": "positive",
            "mentioned_symbols": ["ETH", "ETHUSD"],
            "category": "technology",
            "impact_level": "medium",
            "relevance_score": 0.88,
        },
        {
            "title": "Fed Chair Powell: Monetary Policy Remains Data Dependent",
            "content": """
            Federal Reserve Chairman Jerome Powell reiterated the central bank's
            data-dependent approach to monetary policy during his congressional
            testimony. Powell emphasized that future rate decisions will be based
            on incoming economic data, particularly inflation and employment metrics.

            Markets interpreted the comments as dovish, with risk assets rallying
            on expectations of a more measured approach to rate hikes. The NASDAQ
            gained 1.8% while Bitcoin surged 3.2% following the remarks.

            Powell also addressed cryptocurrency regulation, stating that innovation
            and proper oversight can coexist, providing further support to digital
            assets.
            """,
            "url": "https://reuters.com/fed-powell-testimony",
            "source": "reuters.com",
            "published_time": datetime.now(UTC) - timedelta(hours=4),
            "sentiment": "neutral",
            "mentioned_symbols": [],
            "category": "monetary_policy",
            "impact_level": "high",
            "relevance_score": 0.82,
        },
    ]


@pytest.fixture
def sample_bullish_sentiment() -> Any:
    """Sample bullish sentiment data."""
    try:
        from bot.services.financial_sentiment import SentimentResult

        return SentimentResult(
            sentiment_score=0.72,
            confidence=0.86,
            key_themes=["Bitcoin", "ETF", "Institutional", "Breakout", "DeFi"],
            bullish_indicators=[
                "ETF approval drives institutional demand",
                "Technical breakout above $60K resistance",
                "Volume surge confirms momentum",
                "DeFi TVL reaches new highs",
                "Institutional treasury adoption",
            ],
            bearish_indicators=[
                "RSI indicates overbought conditions",
                "Profit-taking pressure at resistance",
            ],
            volatility_signals=["High options activity", "VIX volatility expected"],
        )
    except ImportError:
        # Return mock if import fails
        return type(
            "MockSentiment",
            (),
            {
                "sentiment_score": 0.72,
                "confidence": 0.86,
                "key_themes": ["Bitcoin", "ETF", "Institutional"],
                "bullish_indicators": ["ETF approval", "Technical breakout"],
                "bearish_indicators": ["Overbought conditions"],
                "volatility_signals": ["High options activity"],
            },
        )()


@pytest.fixture
def sample_bearish_sentiment() -> Any:
    """Sample bearish sentiment data."""
    try:
        from bot.services.financial_sentiment import SentimentResult

        return SentimentResult(
            sentiment_score=-0.58,
            confidence=0.79,
            key_themes=["Regulation", "Liquidation", "Fear", "Correction"],
            bullish_indicators=["Technical support holding"],
            bearish_indicators=[
                "Regulatory crackdown fears",
                "Massive liquidations triggered",
                "Support levels breaking down",
                "Fear index spiking",
                "Institutional selling pressure",
            ],
            volatility_signals=[
                "VIX surging above 30",
                "Options skew increasing",
                "Funding rates turning negative",
            ],
        )
    except ImportError:
        return type(
            "MockSentiment",
            (),
            {
                "sentiment_score": -0.58,
                "confidence": 0.79,
                "key_themes": ["Regulation", "Fear"],
                "bullish_indicators": ["Support holding"],
                "bearish_indicators": ["Regulatory fears", "Liquidations"],
                "volatility_signals": ["VIX surge"],
            },
        )()


@pytest.fixture
def sample_correlation_data() -> Any:
    """Sample correlation analysis data."""
    try:
        from bot.analysis.market_context import CorrelationAnalysis, CorrelationStrength

        return CorrelationAnalysis(
            correlation_coefficient=0.68,
            correlation_strength=CorrelationStrength.MODERATE,
            direction="POSITIVE",
            p_value=0.003,
            is_significant=True,
            sample_size=200,
            rolling_correlation_24h=0.72,
            rolling_correlation_7d=0.64,
            correlation_stability=0.81,
            regime_dependent_correlation={
                "HIGH_VOLATILITY": 0.78,
                "LOW_VOLATILITY": 0.59,
                "RISK_ON": 0.74,
                "RISK_OFF": 0.52,
            },
            reliability_score=0.85,
        )
    except ImportError:
        return type(
            "MockCorrelation",
            (),
            {
                "correlation_coefficient": 0.68,
                "correlation_strength": type(
                    "MockStrength", (), {"value": "MODERATE"}
                )(),
                "direction": "POSITIVE",
                "p_value": 0.003,
                "is_significant": True,
                "sample_size": 200,
                "reliability_score": 0.85,
            },
        )()


@pytest.fixture
def sample_market_regime() -> Any:
    """Sample market regime data."""
    try:
        from bot.analysis.market_context import MarketRegime, MarketRegimeType

        return MarketRegime(
            regime_type=MarketRegimeType.RISK_ON,
            confidence=0.82,
            key_drivers=[
                "Dovish Fed policy supports risk assets",
                "Institutional crypto adoption accelerating",
                "Technology sector showing strength",
                "Geopolitical tensions easing",
            ],
            fed_policy_stance="DOVISH",
            inflation_environment="STABLE",
            interest_rate_trend="STABLE",
            geopolitical_risk_level="LOW",
            crypto_adoption_momentum="HIGH",
            institutional_sentiment="POSITIVE",
            regulatory_environment="FAVORABLE",
            market_volatility_regime="NORMAL",
            liquidity_conditions="ABUNDANT",
            duration_days=28,
            regime_change_probability=0.15,
        )
    except ImportError:
        return type(
            "MockRegime",
            (),
            {
                "regime_type": type("MockType", (), {"value": "RISK_ON"})(),
                "confidence": 0.82,
                "key_drivers": ["Dovish Fed policy", "Crypto adoption"],
                "fed_policy_stance": "DOVISH",
                "inflation_environment": "STABLE",
            },
        )()


# Mock service fixtures
@pytest.fixture
def mock_omnisearch_client() -> Mock:
    """Mock OmniSearch client for testing."""
    client = Mock()
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.search_financial_news = AsyncMock()
    client.search_crypto_sentiment = AsyncMock()
    client.search_nasdaq_sentiment = AsyncMock()
    client.search_market_correlation = AsyncMock()
    client.health_check = AsyncMock(
        return_value={
            "connected": True,
            "server_url": "https://test.omnisearch.com",
            "cache_enabled": True,
            "cache_size": 0,
            "rate_limit_remaining": 100,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )
    return client


@pytest.fixture
def mock_sentiment_service() -> Mock:
    """Mock financial sentiment service for testing."""
    service = Mock()
    service.analyze_news_sentiment = AsyncMock()
    service.extract_crypto_indicators = Mock()
    service.extract_nasdaq_indicators = Mock()
    service.calculate_correlation_score = Mock(return_value=0.65)
    service.format_sentiment_for_llm = Mock(return_value="Formatted sentiment data")
    return service


@pytest.fixture
def mock_context_analyzer() -> Mock:
    """Mock market context analyzer for testing."""
    analyzer = Mock()
    analyzer.analyze_crypto_nasdaq_correlation = AsyncMock()
    analyzer.detect_market_regime = AsyncMock()
    analyzer.assess_risk_sentiment = AsyncMock()
    analyzer.calculate_momentum_alignment = AsyncMock()
    analyzer.generate_context_summary = Mock(return_value="Context summary")
    return analyzer


@pytest.fixture
def mock_search_formatter() -> Mock:
    """Mock web search formatter for testing."""
    formatter = Mock()
    formatter.format_news_results = AsyncMock(return_value="Formatted news")
    formatter.format_sentiment_data = AsyncMock(return_value="Formatted sentiment")
    formatter.format_correlation_analysis = AsyncMock(
        return_value="Formatted correlation"
    )
    formatter.format_market_context = AsyncMock(return_value="Formatted context")
    formatter.extract_key_insights = AsyncMock(
        return_value=["Key insight 1", "Key insight 2"]
    )
    formatter.truncate_content = Mock(
        side_effect=lambda x, y: x[:y] + "..." if len(x) > y else x
    )
    return formatter


@pytest.fixture
def mock_aiohttp_session() -> AsyncMock:
    """Mock aiohttp session for testing."""
    session = AsyncMock()

    # Default successful response
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"results": []})
    response.text = AsyncMock(return_value="Mock response")

    session.get.return_value.__aenter__.return_value = response
    session.post.return_value.__aenter__.return_value = response
    session.close = AsyncMock()

    return session


# File system fixtures
@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_cache_directory(temp_directory: Path) -> Path:
    """Create a mock cache directory structure."""
    cache_dir = temp_directory / "omnisearch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create some sample cache files
    (cache_dir / "news_123.json").write_text(
        json.dumps(
            [
                {
                    "title": "Cached News Item",
                    "content": "Cached content",
                    "url": "https://cached.com",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ]
        )
    )

    return cache_dir


# API response fixtures
@pytest.fixture
def sample_omnisearch_news_response() -> dict[str, Any]:
    """Sample OmniSearch news API response."""
    return {
        "results": [
            {
                "title": "Bitcoin Surges on ETF Approval News",
                "url": "https://financial-news.com/btc-etf",
                "snippet": "Bitcoin price rallies following ETF approval announcement",
                "source": "financial-news.com",
                "published_date": "2024-01-15T10:00:00Z",
                "relevance_score": 0.95,
                "sentiment": "positive",
                "mentioned_symbols": ["BTC", "BTCUSD"],
                "category": "regulation",
                "impact_level": "high",
            },
            {
                "title": "Ethereum Network Upgrade Successful",
                "url": "https://crypto-updates.com/eth-upgrade",
                "snippet": "Latest Ethereum upgrade reduces gas fees significantly",
                "source": "crypto-updates.com",
                "published_date": "2024-01-15T08:30:00Z",
                "relevance_score": 0.82,
                "sentiment": "positive",
                "mentioned_symbols": ["ETH", "ETHUSD"],
                "category": "technology",
                "impact_level": "medium",
            },
        ],
        "total_results": 2,
        "query_time_ms": 150,
    }


@pytest.fixture
def sample_omnisearch_sentiment_response() -> dict[str, Any]:
    """Sample OmniSearch sentiment API response."""
    return {
        "sentiment": {
            "overall": "bullish",
            "score": 0.68,
            "confidence": 0.84,
            "source_count": 42,
            "news_sentiment": 0.72,
            "social_sentiment": 0.65,
            "technical_sentiment": 0.58,
            "key_drivers": [
                "ETF approval momentum building",
                "Institutional adoption accelerating",
                "Technical breakout patterns emerging",
            ],
            "risk_factors": [
                "Regulatory uncertainty in some regions",
                "Market volatility concerns",
                "Profit-taking pressure at resistance",
            ],
        },
        "analysis_time_ms": 89,
    }


@pytest.fixture
def sample_omnisearch_correlation_response() -> dict[str, Any]:
    """Sample OmniSearch correlation API response."""
    return {
        "correlation": {
            "coefficient": 0.65,
            "beta": 1.42,
            "r_squared": 0.42,
            "p_value": 0.005,
            "sample_size": 200,
            "timeframe": "30d",
            "rolling_correlations": {"24h": 0.68, "7d": 0.62, "30d": 0.65},
        },
        "analysis_time_ms": 245,
    }


# Environment fixtures
@pytest.fixture
def mock_environment_variables() -> Generator[dict[str, str], None, None]:
    """Mock environment variables for testing."""
    env_vars = {
        "OMNISEARCH_API_KEY": "test_api_key_12345",
        "OMNISEARCH_SERVER_URL": "https://test-api.omnisearch.dev",
        "OMNISEARCH_CACHE_TTL": "300",
        "OMNISEARCH_RATE_LIMIT": "100",
        "OMNISEARCH_RATE_WINDOW": "3600",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


# Performance fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""

    class Timer:
        def __init__(self) -> None:
            self.start_time: float | None = None
            self.end_time: float | None = None

        def start(self) -> None:
            self.start_time = time.time()

        def stop(self) -> None:
            self.end_time = time.time()

        @property
        def elapsed(self) -> float | None:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# Async fixtures for testing
@pytest.fixture
async def async_context_manager():
    """Async context manager for testing."""

    class AsyncContextManager:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
            self.exited = True

    return AsyncContextManager()


# Error simulation fixtures
@pytest.fixture
def network_error_responses() -> dict[str, Any]:
    """Mock network error responses for testing error handling."""
    return {
        "connection_error": aiohttp.ClientConnectionError("Connection failed"),
        "timeout_error": TimeoutError("Request timed out"),
        "http_error_404": type(
            "MockResponse", (), {"status": 404, "reason": "Not Found"}
        )(),
        "http_error_500": type(
            "MockResponse", (), {"status": 500, "reason": "Internal Server Error"}
        )(),
        "http_error_429": type(
            "MockResponse", (), {"status": 429, "reason": "Too Many Requests"}
        )(),
        "json_decode_error": json.JSONDecodeError("Invalid JSON", "doc", 0),
    }


# Configuration fixtures
@pytest.fixture
def test_config() -> dict[str, Any]:
    """Test configuration settings."""
    return {
        "omnisearch": {
            "server_url": "https://test-api.omnisearch.dev",
            "api_key": "test_key_12345",
            "cache_ttl": 300,
            "rate_limit_requests": 10,
            "rate_limit_window": 60,
            "timeout": 30,
            "max_retries": 3,
        },
        "sentiment": {
            "confidence_threshold": 0.7,
            "sentiment_threshold": 0.5,
            "max_themes": 10,
            "max_indicators": 5,
        },
        "formatting": {
            "max_tokens_per_section": 400,
            "max_total_tokens": 1500,
            "truncation_length": 200,
        },
        "testing": {
            "skip_external_apis": True,
            "use_mock_responses": True,
            "performance_threshold": 2.0,
        },
    }


# Integration test data fixtures
@pytest.fixture
def comprehensive_test_scenario() -> dict[str, Any]:
    """Comprehensive test scenario with realistic market data."""
    return {
        "market_conditions": {
            "crypto_trend": "BULLISH",
            "nasdaq_trend": "BULLISH",
            "correlation": 0.68,
            "volatility": "MODERATE",
            "sentiment": "POSITIVE",
        },
        "news_events": [
            {
                "type": "REGULATORY",
                "impact": "HIGH",
                "sentiment": "POSITIVE",
                "title": "Major ETF Approval",
            },
            {
                "type": "TECHNICAL",
                "impact": "MEDIUM",
                "sentiment": "POSITIVE",
                "title": "Network Upgrade Success",
            },
        ],
        "expected_outcomes": {
            "overall_sentiment": "BULLISH",
            "confidence_range": (0.7, 0.9),
            "correlation_strength": "MODERATE",
            "regime_type": "RISK_ON",
        },
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def _cleanup_test_artifacts() -> Generator[None, None, None]:
    """Automatically cleanup test artifacts after each test."""
    return

    # Cleanup code here if needed
    # For example: clear caches, reset singletons, etc.


# =============================================================================
# FUNCTIONAL PROGRAMMING TEST FIXTURES
# =============================================================================


@pytest.fixture
def fp_result_ok():
    """Fixture for Ok Result monad."""
    try:
        from bot.fp.types.effects import Ok

        return Ok(42)
    except ImportError:
        return None


@pytest.fixture
def fp_result_err():
    """Fixture for Err Result monad."""
    try:
        from bot.fp.types.effects import Err

        return Err("Test error")
    except ImportError:
        return None


@pytest.fixture
def fp_maybe_some():
    """Fixture for Some Maybe monad."""
    try:
        from bot.fp.types.effects import Some

        return Some(42)
    except ImportError:
        return None


@pytest.fixture
def fp_maybe_nothing():
    """Fixture for Nothing Maybe monad."""
    try:
        from bot.fp.types.effects import Nothing

        return Nothing()
    except ImportError:
        return None


@pytest.fixture
def fp_io_pure():
    """Fixture for pure IO monad."""
    try:
        from bot.fp.types.effects import IO

        return IO.pure(42)
    except ImportError:
        return None


@pytest.fixture
def fp_market_snapshot():
    """Fixture for FP MarketSnapshot."""
    try:
        from datetime import UTC, datetime
        from decimal import Decimal

        from bot.fp.types.market import MarketSnapshot

        return MarketSnapshot(
            timestamp=datetime.now(UTC),
            symbol="BTC-USD",
            price=Decimal("50000.00"),
            volume=Decimal("100.00"),
            bid=Decimal("49950.00"),
            ask=Decimal("50050.00"),
        )
    except ImportError:
        return None


@pytest.fixture
def fp_position():
    """Fixture for FP Position."""
    try:
        from decimal import Decimal

        from bot.fp.types.portfolio import Position

        return Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("50000.00"),
        )
    except ImportError:
        return None


@pytest.fixture
def fp_portfolio():
    """Fixture for FP Portfolio."""
    try:
        from decimal import Decimal

        from bot.fp.types.portfolio import Portfolio, Position

        position = Position(
            symbol="BTC-USD",
            side="LONG",
            size=Decimal("0.1"),
            entry_price=Decimal("45000.00"),
            current_price=Decimal("50000.00"),
        )

        return Portfolio(
            positions=(position,),
            cash_balance=Decimal("10000.00"),
        )
    except ImportError:
        return None


@pytest.fixture
def fp_trade_signal_long():
    """Fixture for FP Long trade signal."""
    try:
        from bot.fp.types.trading import Long

        return Long(confidence=0.8, size=0.25, reason="Strong uptrend detected")
    except ImportError:
        return None


@pytest.fixture
def fp_trade_signal_short():
    """Fixture for FP Short trade signal."""
    try:
        from bot.fp.types.trading import Short

        return Short(confidence=0.7, size=0.3, reason="Bearish divergence detected")
    except ImportError:
        return None


@pytest.fixture
def fp_trade_signal_hold():
    """Fixture for FP Hold trade signal."""
    try:
        from bot.fp.types.trading import Hold

        return Hold(reason="Market uncertainty")
    except ImportError:
        return None


@pytest.fixture
def fp_market_make_signal():
    """Fixture for FP MarketMake trade signal."""
    try:
        from bot.fp.types.trading import MarketMake

        return MarketMake(
            bid_price=49900.0,
            ask_price=50100.0,
            bid_size=0.1,
            ask_size=0.1,
        )
    except ImportError:
        return None


@pytest.fixture
def fp_limit_order():
    """Fixture for FP LimitOrder."""
    try:
        from bot.fp.types.trading import LimitOrder

        return LimitOrder(
            symbol="BTC-USD",
            side="buy",
            price=49000.0,
            size=0.1,
        )
    except ImportError:
        return None


@pytest.fixture
def fp_market_order():
    """Fixture for FP MarketOrder."""
    try:
        from bot.fp.types.trading import MarketOrder

        return MarketOrder(
            symbol="BTC-USD",
            side="buy",
            size=0.1,
        )
    except ImportError:
        return None


@pytest.fixture
def fp_base_types():
    """Fixture providing FP base types utilities."""
    try:
        from decimal import Decimal

        from bot.fp.types.base import Money, Percentage, Symbol, TimeInterval

        return {
            "money": Money(amount=Decimal("1000.50"), currency="USD"),
            "percentage": Percentage(value=Decimal("0.15")),
            "symbol": Symbol(value="BTC-USD"),
            "time_interval": TimeInterval.create("1m").unwrap(),
        }
    except ImportError:
        return {}


@pytest.fixture
def fp_config_types():
    """Fixture providing FP config types."""
    try:
        from decimal import Decimal

        from bot.fp.types.base import Percentage, Symbol, TimeInterval
        from bot.fp.types.config import ExchangeConfig, RiskConfig, TradingConfig

        return {
            "trading_config": TradingConfig(
                symbol=Symbol.create("BTC-USD").unwrap(),
                interval=TimeInterval.create("1m").unwrap(),
                leverage=Decimal(5),
            ),
            "risk_config": RiskConfig(
                max_position_size=Percentage.create(0.25).unwrap(),
                stop_loss_pct=Percentage.create(0.02).unwrap(),
                take_profit_pct=Percentage.create(0.05).unwrap(),
            ),
        }
    except ImportError:
        return {}


# =============================================================================
# FP MOCK OBJECTS AND ADAPTERS
# =============================================================================


@pytest.fixture
def fp_mock_exchange_adapter():
    """Mock FP exchange adapter for testing."""
    try:
        from decimal import Decimal
        from unittest.mock import Mock

        from bot.fp.types.effects import IO, Ok

        adapter = Mock()
        adapter.get_balance = Mock(return_value=IO.pure(Ok(Decimal("10000.00"))))
        adapter.place_order = Mock(return_value=IO.pure(Ok({"order_id": "test-123"})))
        adapter.get_market_data = Mock(return_value=IO.pure(Ok({})))
        adapter.supports_functional = Mock(return_value=True)

        return adapter
    except ImportError:
        return Mock()


@pytest.fixture
def fp_mock_strategy():
    """Mock FP strategy for testing."""
    try:
        from unittest.mock import Mock

        from bot.fp.types.effects import IO, Ok
        from bot.fp.types.trading import Hold

        strategy = Mock()
        strategy.generate_signal = Mock(
            return_value=IO.pure(Ok(Hold(reason="Test hold")))
        )
        strategy.update_market_data = Mock(return_value=IO.pure(Ok(None)))

        return strategy
    except ImportError:
        return Mock()


@pytest.fixture
def fp_mock_risk_manager():
    """Mock FP risk manager for testing."""
    try:
        from unittest.mock import Mock

        from bot.fp.types.effects import IO, Ok

        risk_manager = Mock()
        risk_manager.validate_trade = Mock(return_value=IO.pure(Ok(True)))
        risk_manager.calculate_position_size = Mock(return_value=IO.pure(Ok(0.1)))
        risk_manager.check_limits = Mock(return_value=IO.pure(Ok(True)))

        return risk_manager
    except ImportError:
        return Mock()


# =============================================================================
# FP TEST UTILITIES
# =============================================================================


@pytest.fixture
def fp_test_utils():
    """FP test utilities and helpers."""

    class FPTestUtils:
        """Utilities for testing FP patterns."""

        @staticmethod
        def assert_result_ok(result, expected_value=None):
            """Assert Result is Ok and optionally check value."""
            assert (
                result.is_ok()
            ), f"Expected Ok, got Err: {result.error if hasattr(result, 'error') else result}"
            if expected_value is not None:
                assert result.unwrap() == expected_value

        @staticmethod
        def assert_result_err(result, expected_error=None):
            """Assert Result is Err and optionally check error."""
            assert (
                result.is_err()
            ), f"Expected Err, got Ok: {result.unwrap() if hasattr(result, 'unwrap') else result}"
            if expected_error is not None:
                assert result.error == expected_error

        @staticmethod
        def assert_maybe_some(maybe, expected_value=None):
            """Assert Maybe is Some and optionally check value."""
            assert maybe.is_some(), "Expected Some, got Nothing"
            if expected_value is not None:
                assert maybe.unwrap() == expected_value

        @staticmethod
        def assert_maybe_nothing(maybe):
            """Assert Maybe is Nothing."""
            assert (
                maybe.is_nothing()
            ), f"Expected Nothing, got Some: {maybe.unwrap() if hasattr(maybe, 'unwrap') else maybe}"

        @staticmethod
        def assert_io_result(io, expected_value=None):
            """Assert IO computation result."""
            result = io.run()
            if expected_value is not None:
                assert result == expected_value
            return result

        @staticmethod
        def create_test_market_data():
            """Create test market data in FP format."""
            try:
                from datetime import UTC, datetime
                from decimal import Decimal

                from bot.fp.types.market import OHLCV, MarketSnapshot

                return MarketSnapshot(
                    timestamp=datetime.now(UTC),
                    symbol="BTC-USD",
                    price=Decimal("50000.00"),
                    volume=Decimal("100.00"),
                    bid=Decimal("49950.00"),
                    ask=Decimal("50050.00"),
                )
            except ImportError:
                return None

        @staticmethod
        def create_test_ohlcv_series(length=100):
            """Create test OHLCV series in FP format."""
            try:
                import random
                from datetime import UTC, datetime, timedelta
                from decimal import Decimal

                from bot.fp.types.market import OHLCV

                series = []
                base_time = datetime.now(UTC)
                base_price = 50000.0

                for i in range(length):
                    price_change = random.uniform(-0.02, 0.02)
                    current_price = base_price * (1 + price_change)

                    ohlcv = OHLCV(
                        timestamp=base_time + timedelta(minutes=i),
                        open=Decimal(str(current_price * 0.999)),
                        high=Decimal(str(current_price * 1.001)),
                        low=Decimal(str(current_price * 0.998)),
                        close=Decimal(str(current_price)),
                        volume=Decimal(str(random.uniform(50, 200))),
                    )
                    series.append(ohlcv)
                    base_price = current_price

                return series
            except ImportError:
                return []

    return FPTestUtils()


@pytest.fixture
def fp_property_strategies():
    """Property-based testing strategies for FP types."""
    try:
        from datetime import UTC, datetime, timedelta
        from decimal import Decimal

        from hypothesis import strategies as st

        @st.composite
        def fp_decimal_strategy(draw, min_value=0, max_value=1000000, places=8):
            """Strategy for generating Decimal values."""
            value = draw(
                st.floats(min_value=min_value, max_value=max_value, allow_nan=False)
            )
            return Decimal(str(round(value, places)))

        @st.composite
        def fp_timestamp_strategy(draw):
            """Strategy for generating timestamps."""
            base = datetime.now(UTC)
            delta_days = draw(st.integers(min_value=-365, max_value=1))
            delta_seconds = draw(st.integers(min_value=0, max_value=86400))
            return base + timedelta(days=delta_days, seconds=delta_seconds)

        return {
            "decimal": fp_decimal_strategy,
            "timestamp": fp_timestamp_strategy,
            "symbol": st.sampled_from(["BTC-USD", "ETH-USD", "SOL-USD"]),
            "side": st.sampled_from(["LONG", "SHORT", "FLAT"]),
            "confidence": st.floats(min_value=0.0, max_value=1.0),
            "size": st.floats(min_value=0.01, max_value=1.0),
        }
    except ImportError:
        return {}


# =============================================================================
# ORIGINAL PARAMETRIZED FIXTURES (MAINTAINED FOR BACKWARD COMPATIBILITY)
# =============================================================================


@pytest.fixture(
    params=[
        "bullish_scenario",
        "bearish_scenario",
        "neutral_scenario",
        "high_volatility_scenario",
    ]
)
def market_scenario(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parametrized market scenarios for comprehensive testing."""
    scenarios = {
        "bullish_scenario": {
            "sentiment_score": 0.7,
            "correlation": 0.6,
            "volatility": 0.2,
            "trend": "BULLISH",
        },
        "bearish_scenario": {
            "sentiment_score": -0.6,
            "correlation": 0.8,  # High correlation in risk-off
            "volatility": 0.4,
            "trend": "BEARISH",
        },
        "neutral_scenario": {
            "sentiment_score": 0.1,
            "correlation": 0.3,
            "volatility": 0.15,
            "trend": "NEUTRAL",
        },
        "high_volatility_scenario": {
            "sentiment_score": 0.2,
            "correlation": 0.9,  # Very high correlation in stress
            "volatility": 0.6,
            "trend": "VOLATILE",
        },
    }
    return scenarios[request.param]


@pytest.fixture(params=["fp_scenario", "imperative_scenario"])
def fp_migration_scenario(
    request: pytest.FixtureRequest, fp_market_snapshot, sample_price_data
) -> dict[str, Any]:
    """Parametrized scenarios for testing FP migration compatibility."""
    if request.param == "fp_scenario":
        return {
            "type": "functional",
            "market_data": fp_market_snapshot,
            "is_fp": True,
        }
    return {
        "type": "imperative",
        "market_data": sample_price_data,
        "is_fp": False,
    }


# Custom assertions
def assert_sentiment_in_range(
    sentiment_score: float, expected_min: float, expected_max: float
) -> None:
    """Custom assertion for sentiment score ranges."""
    assert expected_min <= sentiment_score <= expected_max, (
        f"Sentiment score {sentiment_score} not in expected range "
        f"[{expected_min}, {expected_max}]"
    )


def assert_correlation_strength(
    correlation_coef: float, expected_strength: str
) -> None:
    """Custom assertion for correlation strength."""
    abs_corr = abs(correlation_coef)
    strength_ranges = {
        "VERY_WEAK": (0.0, 0.2),
        "WEAK": (0.2, 0.4),
        "MODERATE": (0.4, 0.6),
        "STRONG": (0.6, 0.8),
        "VERY_STRONG": (0.8, 1.0),
    }

    min_val, max_val = strength_ranges[expected_strength]
    assert min_val <= abs_corr <= max_val, (
        f"Correlation {correlation_coef} (abs: {abs_corr}) not in "
        f"{expected_strength} range [{min_val}, {max_val}]"
    )


# Test data validation helpers
def validate_news_item(news_item: dict[str, Any]) -> None:
    """Validate news item structure."""
    required_fields = ["title", "content", "url", "published_time"]
    for field in required_fields:
        assert field in news_item, f"Missing required field: {field}"

    assert len(news_item["title"]) > 0, "Title cannot be empty"
    assert news_item["url"].startswith("http"), "URL must be valid"


def validate_sentiment_result(sentiment_result: Any) -> None:
    """Validate sentiment result structure."""
    assert hasattr(sentiment_result, "sentiment_score")
    assert hasattr(sentiment_result, "confidence")
    assert -1.0 <= sentiment_result.sentiment_score <= 1.0
    assert 0.0 <= sentiment_result.confidence <= 1.0


# Performance testing utilities
@pytest.fixture
def benchmark_timer() -> Any:
    """Benchmarking timer for performance tests."""
    import time

    @contextmanager
    def timer() -> Generator[Any, None, None]:
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start

    return timer
