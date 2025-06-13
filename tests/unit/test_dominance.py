"""Unit tests for stablecoin dominance data provider."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.data.dominance import DominanceData, DominanceDataProvider


@pytest.fixture
def dominance_provider():
    """Create a dominance data provider instance."""
    return DominanceDataProvider(data_source="coingecko", update_interval=300)


@pytest.fixture
def mock_coingecko_response():
    """Mock CoinGecko API response."""
    return {
        "global": {
            "data": {
                "total_market_cap": {"usd": 2000000000000},  # $2T
                "market_cap_change_percentage_24h_usd": 2.5,
            }
        },
        "coins": [
            {
                "id": "tether",
                "market_cap": 100000000000,  # $100B
                "total_volume": 50000000000,  # $50B
                "market_cap_change_percentage_24h": 0.5,
            },
            {
                "id": "usd-coin",
                "market_cap": 40000000000,  # $40B
                "total_volume": 20000000000,  # $20B
                "market_cap_change_percentage_24h": -0.3,
            },
        ],
    }


@pytest.fixture
def sample_dominance_data():
    """Create sample dominance data."""
    return DominanceData(
        timestamp=datetime.now(UTC),
        usdt_market_cap=Decimal("100000000000"),
        usdc_market_cap=Decimal("40000000000"),
        total_stablecoin_cap=Decimal("140000000000"),
        crypto_total_market_cap=Decimal("2000000000000"),
        usdt_dominance=5.0,
        usdc_dominance=2.0,
        stablecoin_dominance=7.0,
        dominance_24h_change=-0.5,
        dominance_7d_change=-1.2,
        stablecoin_velocity=0.5,
        net_flow_24h=Decimal("-1000000000"),
        dominance_sma_20=7.2,
        dominance_rsi=45.0,
    )


class TestDominanceDataProvider:
    """Test cases for DominanceDataProvider."""

    @pytest.mark.asyncio
    async def test_init(self, dominance_provider):
        """Test provider initialization."""
        assert dominance_provider.data_source == "coingecko"
        assert dominance_provider.update_interval == 300
        assert dominance_provider._dominance_cache == []
        assert dominance_provider._session is None

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, dominance_provider):
        """Test connection lifecycle."""
        with patch("aiohttp.ClientSession") as mock_session:
            await dominance_provider.connect()
            assert dominance_provider._session is not None
            assert dominance_provider._running is True

            await dominance_provider.disconnect()
            assert dominance_provider._session is None
            assert dominance_provider._running is False

    @pytest.mark.asyncio
    async def test_fetch_coingecko_dominance(
        self, dominance_provider, mock_coingecko_response
    ):
        """Test fetching dominance data from CoinGecko."""
        # Create mock session
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=[
                mock_coingecko_response["global"],
                mock_coingecko_response["coins"],
            ]
        )

        mock_session.get.return_value.__aenter__.return_value = mock_response
        dominance_provider._session = mock_session

        # Fetch dominance data
        result = await dominance_provider.fetch_current_dominance()

        # Verify result
        assert result is not None
        assert result.usdt_dominance == 5.0  # 100B / 2T * 100
        assert result.usdc_dominance == 2.0  # 40B / 2T * 100
        assert result.stablecoin_dominance == 7.0  # 140B / 2T * 100
        assert result.stablecoin_velocity == 0.5  # 70B / 140B

    @pytest.mark.asyncio
    async def test_get_market_sentiment(
        self, dominance_provider, sample_dominance_data
    ):
        """Test market sentiment analysis."""
        # Add sample data to cache
        dominance_provider._dominance_cache = [sample_dominance_data]

        sentiment = dominance_provider.get_market_sentiment()

        assert sentiment["sentiment"] == "BEARISH"  # 7% dominance = cautious
        assert sentiment["dominance"] == 7.0
        assert sentiment["dominance_change_24h"] == -0.5
        assert "factors" in sentiment
        assert len(sentiment["factors"]) > 0

    def test_sentiment_analysis_high_dominance(self, dominance_provider):
        """Test sentiment analysis with high dominance."""
        # Create high dominance data
        high_dominance = DominanceData(
            timestamp=datetime.now(UTC),
            usdt_market_cap=Decimal("150000000000"),
            usdc_market_cap=Decimal("60000000000"),
            total_stablecoin_cap=Decimal("210000000000"),
            crypto_total_market_cap=Decimal("2000000000000"),
            usdt_dominance=7.5,
            usdc_dominance=3.0,
            stablecoin_dominance=10.5,  # High dominance
            dominance_24h_change=1.2,  # Rising
            dominance_7d_change=3.5,
            stablecoin_velocity=0.8,
            dominance_rsi=75.0,  # Overbought
        )

        dominance_provider._dominance_cache = [high_dominance]
        sentiment = dominance_provider.get_market_sentiment()

        assert sentiment["sentiment"] == "STRONG_BEARISH"
        assert sentiment["score"] < -2  # Strong negative score
        assert "High stablecoin dominance" in str(sentiment["factors"])
        assert "Increasing dominance" in str(sentiment["factors"])

    def test_sentiment_analysis_low_dominance(self, dominance_provider):
        """Test sentiment analysis with low dominance."""
        # Create low dominance data
        low_dominance = DominanceData(
            timestamp=datetime.now(UTC),
            usdt_market_cap=Decimal("50000000000"),
            usdc_market_cap=Decimal("30000000000"),
            total_stablecoin_cap=Decimal("80000000000"),
            crypto_total_market_cap=Decimal("2000000000000"),
            usdt_dominance=2.5,
            usdc_dominance=1.5,
            stablecoin_dominance=4.0,  # Low dominance
            dominance_24h_change=-0.8,  # Falling
            dominance_7d_change=-2.1,
            stablecoin_velocity=0.3,
            dominance_rsi=25.0,  # Oversold
        )

        dominance_provider._dominance_cache = [low_dominance]
        sentiment = dominance_provider.get_market_sentiment()

        assert sentiment["sentiment"] == "BULLISH"
        assert sentiment["score"] > 1  # Positive score
        assert "Low stablecoin dominance" in str(sentiment["factors"])
        assert "Decreasing dominance" in str(sentiment["factors"])

    def test_get_dominance_history(self, dominance_provider):
        """Test retrieving historical dominance data."""
        # Create historical data points
        now = datetime.now(UTC)
        history = []

        for i in range(10):
            data = DominanceData(
                timestamp=now - timedelta(hours=i),
                usdt_market_cap=Decimal("100000000000"),
                usdc_market_cap=Decimal("40000000000"),
                total_stablecoin_cap=Decimal("140000000000"),
                crypto_total_market_cap=Decimal("2000000000000"),
                usdt_dominance=5.0,
                usdc_dominance=2.0,
                stablecoin_dominance=7.0 + (i * 0.1),  # Varying dominance
                dominance_24h_change=0.1 * i,
                dominance_7d_change=0.2 * i,
                stablecoin_velocity=0.5,
            )
            history.append(data)

        dominance_provider._dominance_cache = history

        # Get 5 hours of history
        result = dominance_provider.get_dominance_history(hours=5)

        assert len(result) == 6  # 0-5 hours ago
        assert result[0].timestamp >= now - timedelta(hours=5)

    def test_to_dataframe(self, dominance_provider):
        """Test converting dominance data to pandas DataFrame."""
        # Create sample data
        now = datetime.now(UTC)
        data_points = []

        for i in range(5):
            data = DominanceData(
                timestamp=now - timedelta(hours=i),
                usdt_market_cap=Decimal("100000000000"),
                usdc_market_cap=Decimal("40000000000"),
                total_stablecoin_cap=Decimal("140000000000"),
                crypto_total_market_cap=Decimal("2000000000000"),
                usdt_dominance=5.0,
                usdc_dominance=2.0,
                stablecoin_dominance=7.0,
                dominance_24h_change=-0.5,
                dominance_7d_change=-1.2,
                stablecoin_velocity=0.5,
                dominance_rsi=45.0 + i,
            )
            data_points.append(data)

        dominance_provider._dominance_cache = data_points

        # Convert to DataFrame
        df = dominance_provider.to_dataframe(hours=24)

        assert not df.empty
        assert len(df) == 5
        assert "stablecoin_dominance" in df.columns
        assert "dominance_24h_change" in df.columns
        assert "velocity" in df.columns
        assert "rsi" in df.columns

    @pytest.mark.asyncio
    async def test_update_loop(self, dominance_provider):
        """Test the periodic update loop."""
        # Mock fetch method
        dominance_provider.fetch_current_dominance = AsyncMock()
        dominance_provider._running = True
        dominance_provider.update_interval = 0.1  # Fast updates for testing

        # Start update loop
        update_task = asyncio.create_task(dominance_provider._update_loop())

        # Wait for a few updates
        await asyncio.sleep(0.3)

        # Stop the loop
        dominance_provider._running = False
        update_task.cancel()

        try:
            await update_task
        except asyncio.CancelledError:
            pass

        # Verify updates were called
        assert dominance_provider.fetch_current_dominance.call_count >= 2

    def test_calculate_trend_indicators(self, dominance_provider):
        """Test trend indicator calculations."""
        # Create enough data for indicators
        now = datetime.now(UTC)
        data_points = []

        for i in range(30):
            dominance = 7.0 + (i % 5) * 0.2  # Oscillating dominance
            data = DominanceData(
                timestamp=now - timedelta(hours=29 - i),
                usdt_market_cap=Decimal("100000000000"),
                usdc_market_cap=Decimal("40000000000"),
                total_stablecoin_cap=Decimal("140000000000"),
                crypto_total_market_cap=Decimal("2000000000000"),
                usdt_dominance=5.0,
                usdc_dominance=2.0,
                stablecoin_dominance=dominance,
                dominance_24h_change=0.1,
                dominance_7d_change=0.2,
                stablecoin_velocity=0.5,
            )
            data_points.append(data)

        dominance_provider._dominance_cache = data_points
        dominance_provider._calculate_trend_indicators()

        # Check that indicators were calculated
        latest = dominance_provider._dominance_cache[-1]
        assert latest.dominance_sma_20 is not None
        assert latest.dominance_rsi is not None
        assert 0 <= latest.dominance_rsi <= 100

    @pytest.mark.asyncio
    async def test_coinmarketcap_integration(self, dominance_provider):
        """Test CoinMarketCap data source (mocked)."""
        dominance_provider.data_source = "coinmarketcap"
        dominance_provider.api_key = "test-api-key"

        # Mock response
        mock_response = {
            "status": {"error_code": 0},
            "data": {
                "quote": {
                    "USD": {
                        "total_market_cap": 2000000000000,
                        "stablecoin_market_cap": 140000000000,
                        "stablecoin_volume_24h": 70000000000,
                        "stablecoin_market_cap_change_24h": -0.5,
                    }
                }
            },
        }

        # Create mock session
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        mock_session.get.return_value.__aenter__.return_value = mock_resp
        dominance_provider._session = mock_session

        # Fetch dominance data
        result = await dominance_provider.fetch_current_dominance()

        # Verify result
        assert result is not None
        assert result.stablecoin_dominance == 7.0  # 140B / 2T * 100
        assert result.dominance_24h_change == -0.5

        # Verify API key was used
        call_args = mock_session.get.call_args
        headers = call_args[1]["headers"]
        assert headers["X-CMC_PRO_API_KEY"] == "test-api-key"

    @pytest.mark.asyncio
    async def test_error_handling(self, dominance_provider):
        """Test error handling in data fetching."""
        # Mock failed response
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_session.get.return_value.__aenter__.return_value = mock_response
        dominance_provider._session = mock_session

        # Fetch should return None on error
        result = await dominance_provider.fetch_current_dominance()
        assert result is None

    @pytest.mark.asyncio
    async def test_context_manager(self, dominance_provider):
        """Test async context manager usage."""
        async with dominance_provider as provider:
            assert provider._session is not None
            assert provider._running is True

        assert provider._session is None
        assert provider._running is False
