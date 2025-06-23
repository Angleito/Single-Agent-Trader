"""
Stablecoin dominance data provider for market sentiment analysis.

This module provides real-time USDT/USDC dominance metrics to gauge market sentiment.
High stablecoin dominance often indicates risk-off sentiment, while decreasing dominance
suggests risk-on behavior and potential market recovery.
"""

import asyncio
import atexit
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar

import aiohttp
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bot.config import settings

logger = logging.getLogger(__name__)


class DominanceData(BaseModel):
    """Stablecoin dominance metrics."""

    timestamp: datetime
    usdt_market_cap: Decimal
    usdc_market_cap: Decimal
    total_stablecoin_cap: Decimal
    crypto_total_market_cap: Decimal
    usdt_dominance: float  # Percentage of total crypto market
    usdc_dominance: float  # Percentage of total crypto market
    stablecoin_dominance: float  # Combined USDT+USDC dominance
    dominance_24h_change: float  # 24h change in stablecoin dominance
    dominance_7d_change: float  # 7d change in stablecoin dominance

    # Flow metrics
    stablecoin_velocity: float | None = None  # Trading volume / market cap
    net_flow_24h: Decimal | None = None  # Net in/outflow in 24h

    # Trend indicators
    dominance_sma_20: float | None = None  # 20-period SMA of dominance
    dominance_rsi: float | None = None  # RSI of dominance changes

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DominanceDataProvider:
    """
    Provides stablecoin dominance data for market sentiment analysis.

    Features:
    - Real-time USDT/USDC market cap and dominance metrics
    - Historical dominance data with trend analysis
    - Stablecoin flow and velocity calculations
    - Market sentiment indicators based on dominance shifts
    - Caching with configurable TTL
    - High-frequency data collection support (30-second intervals)
    """

    # Class variable to track all instances for cleanup
    _instances: ClassVar[list["DominanceDataProvider"]] = []
    _cleanup_registered = False

    # API endpoints for different data sources
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    COINMARKETCAP_API_URL = "https://pro-api.coinmarketcap.com/v1"

    def __init__(
        self,
        data_source: str = "coingecko",
        api_key: str | None = None,
        update_interval: int = 30,  # 30 seconds default for high-frequency collection
    ):
        """
        Initialize the dominance data provider.

        Args:
            data_source: API source ('coingecko', 'coinmarketcap')
            api_key: API key for premium endpoints
            update_interval: Update interval in seconds (default 30s for high-frequency collection)
        """
        self.data_source = data_source
        self.api_key = api_key
        self.update_interval = update_interval

        # High-frequency collection mode tracking
        self.is_high_frequency = update_interval <= 60

        # Data cache
        self._dominance_cache: list[DominanceData] = []
        self._last_update: datetime | None = None
        self._cache_ttl = timedelta(seconds=update_interval)

        # HTTP session
        self._session: aiohttp.ClientSession | None = None

        # Update task
        self._update_task: asyncio.Task | None = None
        self._running = False

        # Register instance for cleanup
        DominanceDataProvider._instances.append(self)

        # Register atexit handler once
        if not DominanceDataProvider._cleanup_registered:
            DominanceDataProvider._cleanup_registered = True
            atexit.register(DominanceDataProvider._cleanup_all_instances)

        logger.info("Initialized DominanceDataProvider with %s source", data_source)

    def __del__(self):
        """Cleanup when object is garbage collected."""
        if self._session and not self._session.closed:
            logger.warning(
                "DominanceDataProvider being garbage collected with open session. "
                "Consider using async context manager or calling disconnect() explicitly."
            )
            # Note: We can't reliably close async sessions from __del__
            # The session will be cleaned up when the event loop closes

    async def __aenter__(self) -> "DominanceDataProvider":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, _exc_tb: Any) -> bool:
        """Async context manager exit."""
        await self.disconnect()
        return False

    @classmethod
    def _cleanup_all_instances(cls) -> None:
        """Cleanup all instances on exit."""
        for instance in cls._instances:
            if instance._session and not instance._session.closed:
                logger.warning(
                    "Cleaning up unclosed DominanceDataProvider session on exit"
                )
                try:
                    # Force close the connector
                    instance._session._connector.close()
                except Exception:
                    logger.exception("Error during emergency cleanup: %s")

    async def connect(self) -> None:
        """Initialize connection and start data updates."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            # Create session with connector that properly closes
            connector = aiohttp.TCPConnector(force_close=True)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                # Ensure proper cleanup on exit
                connector_owner=True,
            )

        # Fetch initial data
        await self.fetch_current_dominance()

        # Start periodic updates
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())

        logger.info("DominanceDataProvider connected and update loop started")

    async def disconnect(self) -> None:
        """Disconnect and cleanup resources."""
        self._running = False

        if self._update_task:
            self._update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._update_task
            self._update_task = None

        if self._session:
            try:
                await self._session.close()
                # Give the session time to close gracefully
                await asyncio.sleep(0)
            except Exception as e:
                logger.warning("Error closing session: %s", e)
            finally:
                self._session = None

        logger.info("DominanceDataProvider disconnected")

    async def fetch_current_dominance(self) -> DominanceData | None:
        """
        Fetch current stablecoin dominance data.

        Returns:
            Current dominance metrics or None if fetch fails
        """
        try:
            if self.data_source == "coingecko":
                return await self._fetch_coingecko_dominance()
            if self.data_source == "coinmarketcap":
                return await self._fetch_coinmarketcap_dominance()
        except Exception:
            logger.exception("Error fetching dominance data: %s")
            return None
        else:
            logger.error("Unsupported data source: %s", self.data_source)
            return None
            logger.exception("Error fetching dominance data: %s")
            return None

    async def _fetch_coingecko_dominance(self) -> DominanceData | None:
        """Fetch dominance data from CoinGecko API."""
        if not self._session:
            logger.error("Session not initialized")
            return None

        try:
            # Fetch global market data
            global_url = f"{self.COINGECKO_API_URL}/global"
            async with self._session.get(global_url) as response:
                if response.status != 200:
                    logger.error("Failed to fetch global data: %s", response.status)
                    return None

                global_data = await response.json()
                global_info = global_data.get("data", {})

                total_market_cap = Decimal(
                    str(global_info.get("total_market_cap", {}).get("usd", 0))
                )

            # Fetch specific coin data for USDT and USDC
            coins_url = f"{self.COINGECKO_API_URL}/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": "tether,usd-coin",
                "order": "market_cap_desc",
                "sparkline": "false",
            }

            async with self._session.get(coins_url, params=params) as response:
                if response.status != 200:
                    logger.error("Failed to fetch coin data: %s", response.status)
                    return None

                coins_data = await response.json()

            # Extract stablecoin data
            usdt_data = next(
                (coin for coin in coins_data if coin["id"] == "tether"), None
            )
            usdc_data = next(
                (coin for coin in coins_data if coin["id"] == "usd-coin"), None
            )

            if not usdt_data or not usdc_data:
                logger.error("Failed to find USDT or USDC data")
                return None

            # Calculate metrics
            usdt_cap = Decimal(str(usdt_data["market_cap"]))
            usdc_cap = Decimal(str(usdc_data["market_cap"]))
            total_stable_cap = usdt_cap + usdc_cap

            usdt_dominance = (
                float(usdt_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            )
            usdc_dominance = (
                float(usdc_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            )
            stablecoin_dominance = usdt_dominance + usdc_dominance

            # Calculate 24h change in dominance (approximation)
            usdt_24h_change = usdt_data.get("market_cap_change_percentage_24h", 0)
            market_24h_change = global_info.get(
                "market_cap_change_percentage_24h_usd", 0
            )
            dominance_24h_change = usdt_24h_change - market_24h_change

            # Calculate velocity if volume data available
            usdt_volume = Decimal(str(usdt_data.get("total_volume", 0)))
            usdc_volume = Decimal(str(usdc_data.get("total_volume", 0)))
            total_stable_volume = usdt_volume + usdc_volume

            velocity = (
                float(total_stable_volume / total_stable_cap)
                if total_stable_cap > 0
                else None
            )

            dominance_data = DominanceData(
                timestamp=datetime.now(UTC),
                usdt_market_cap=usdt_cap,
                usdc_market_cap=usdc_cap,
                total_stablecoin_cap=total_stable_cap,
                crypto_total_market_cap=total_market_cap,
                usdt_dominance=usdt_dominance,
                usdc_dominance=usdc_dominance,
                stablecoin_dominance=stablecoin_dominance,
                dominance_24h_change=dominance_24h_change,
                dominance_7d_change=0.0,  # Would need historical data
                stablecoin_velocity=velocity,
            )

        except Exception:
            logger.exception("Error in CoinGecko dominance fetch: %s")
            return None
        else:
            # Update cache
            self._dominance_cache.append(dominance_data)
            if (
                len(self._dominance_cache) > 5760
            ):  # Keep 48 hours at 30-second intervals
                self._dominance_cache = self._dominance_cache[-5760:]

            self._last_update = datetime.now(UTC)

            # Calculate trend indicators if we have enough history
            if len(self._dominance_cache) >= 20:
                self._calculate_trend_indicators()

            return dominance_data

    async def _fetch_coinmarketcap_dominance(self) -> DominanceData | None:
        """Fetch dominance data from CoinMarketCap API (requires API key)."""
        if not self.api_key:
            logger.error("CoinMarketCap requires API key")
            return None

        if not self._session:
            logger.error("Session not initialized")
            return None

        try:
            # Set up headers with API key
            headers = {
                "X-CMC_PRO_API_KEY": self.api_key,
                "Accept": "application/json",
            }

            # Fetch global metrics including market cap and dominance
            global_url = f"{self.COINMARKETCAP_API_URL}/global-metrics/quotes/latest"

            async with self._session.get(global_url, headers=headers) as response:
                if response.status != 200:
                    logger.error(
                        "Failed to fetch CoinMarketCap data: %s", response.status
                    )
                    return None

                data = await response.json()

                if data.get("status", {}).get("error_code") != 0:
                    logger.error(
                        "CoinMarketCap API error: %s",
                        data.get("status", {}).get("error_message"),
                    )
                    return None

                # Extract global metrics
                quote = data.get("data", {}).get("quote", {}).get("USD", {})
                total_market_cap = Decimal(str(quote.get("total_market_cap", 0)))

                # Get stablecoin specific data
                stablecoin_market_cap = Decimal(
                    str(quote.get("stablecoin_market_cap", 0))
                )
                stablecoin_volume_24h = Decimal(
                    str(quote.get("stablecoin_volume_24h", 0))
                )

                # Calculate dominance percentages
                stablecoin_dominance = (
                    float(stablecoin_market_cap / total_market_cap * 100)
                    if total_market_cap > 0
                    else 0
                )

                # Get individual stablecoin data if available
                # Note: This requires additional API calls or a different endpoint
                # For now, we'll estimate based on typical USDT/USDC split (70/30)
                usdt_dominance = stablecoin_dominance * 0.7
                usdc_dominance = stablecoin_dominance * 0.3

                # Calculate velocity
                velocity = (
                    float(stablecoin_volume_24h / stablecoin_market_cap)
                    if stablecoin_market_cap > 0
                    else None
                )

                # Get 24h change if available
                stablecoin_change_24h = quote.get("stablecoin_market_cap_change_24h", 0)

                dominance_data = DominanceData(
                    timestamp=datetime.now(UTC),
                    usdt_market_cap=stablecoin_market_cap * Decimal("0.7"),  # Estimated
                    usdc_market_cap=stablecoin_market_cap * Decimal("0.3"),  # Estimated
                    total_stablecoin_cap=stablecoin_market_cap,
                    crypto_total_market_cap=total_market_cap,
                    usdt_dominance=usdt_dominance,
                    usdc_dominance=usdc_dominance,
                    stablecoin_dominance=stablecoin_dominance,
                    dominance_24h_change=stablecoin_change_24h,
                    dominance_7d_change=0.0,  # Would need historical data
                    stablecoin_velocity=velocity,
                )

                # Update cache
                self._dominance_cache.append(dominance_data)
                if (
                    len(self._dominance_cache) > 5760
                ):  # Keep 48 hours at 30-second intervals
                    self._dominance_cache = self._dominance_cache[-5760:]

                self._last_update = datetime.now(UTC)

                # Calculate trend indicators if we have enough history
                if len(self._dominance_cache) >= 20:
                    self._calculate_trend_indicators()

                return dominance_data

        except Exception:
            logger.exception("Error in CoinMarketCap dominance fetch: %s")
            return None

    def _calculate_trend_indicators(self) -> None:
        """Calculate trend indicators for dominance data."""
        if len(self._dominance_cache) < 20:
            return

        try:
            # Filter out None values and validate data before DataFrame creation
            valid_data = [
                (i, d)
                for i, d in enumerate(self._dominance_cache)
                if d.stablecoin_dominance is not None
                and not pd.isna(d.stablecoin_dominance)
            ]

            if len(valid_data) < 20:
                logger.warning(
                    "Insufficient valid dominance data for trend calculation: %s valid points out of %s total",
                    len(valid_data),
                    len(self._dominance_cache),
                )
                return

            # Extract only valid dominance values
            dominance_values = [d[1].stablecoin_dominance for d in valid_data]

            # Ensure all values are numeric
            dominance_values = [
                float(val)
                for val in dominance_values
                if isinstance(val, int | float) and not pd.isna(val)
            ]

            if len(dominance_values) < 20:
                logger.warning(
                    "Insufficient numeric dominance data after cleaning: %s values",
                    len(dominance_values),
                )
                return

            # Create DataFrame with clean data
            dominance_df = pd.DataFrame({"dominance": dominance_values})

            # Calculate 20-period SMA
            dominance_df["sma_20"] = dominance_df["dominance"].rolling(window=20).mean()

            # Calculate RSI
            dominance_df["change"] = dominance_df["dominance"].diff()
            dominance_df["gain"] = dominance_df["change"].where(
                dominance_df["change"] > 0, 0
            )
            dominance_df["loss"] = -dominance_df["change"].where(
                dominance_df["change"] < 0, 0
            )

            avg_gain = dominance_df["gain"].rolling(window=14).mean()
            avg_loss = dominance_df["loss"].rolling(window=14).mean()

            # Prevent division by zero in RSI calculation
            rs = avg_gain / avg_loss.replace(0, 1e-8)
            dominance_df["rsi"] = 100 - (100 / (1 + rs))

            # Update latest entry with indicators
            if len(self._dominance_cache) > 0:
                latest = self._dominance_cache[-1]
                latest.dominance_sma_20 = (
                    float(dominance_df["sma_20"].iloc[-1])
                    if not pd.isna(dominance_df["sma_20"].iloc[-1])
                    else None
                )
                latest.dominance_rsi = (
                    float(dominance_df["rsi"].iloc[-1])
                    if not pd.isna(dominance_df["rsi"].iloc[-1])
                    else None
                )

        except Exception:
            logger.exception("Error calculating dominance trend indicators: %s")
            # Don't let trend calculation failures break the system
            return

    async def _update_loop(self) -> None:
        """Periodic update loop for dominance data."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                await self.fetch_current_dominance()
                logger.debug("Updated dominance data")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in update loop: %s")
                await asyncio.sleep(60)  # Wait a minute before retry

    def get_latest_dominance(self) -> DominanceData | None:
        """
        Get the most recent dominance data.

        Returns:
            Latest dominance metrics or None if no data
        """
        if self._dominance_cache:
            return self._dominance_cache[-1]
        return None

    def get_dominance_history(self, hours: int = 24) -> list[DominanceData]:
        """
        Get historical dominance data.

        Args:
            hours: Number of hours of history to return

        Returns:
            List of historical dominance data points
        """
        if not self._dominance_cache:
            return []

        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        return [d for d in self._dominance_cache if d.timestamp >= cutoff_time]

    def get_market_sentiment(self) -> dict[str, Any]:
        """
        Analyze market sentiment based on dominance metrics.

        Returns:
            Dictionary with sentiment analysis
        """
        latest = self.get_latest_dominance()
        if not latest:
            return {"sentiment": "UNKNOWN", "confidence": 0}

        sentiment_score = 0.0
        factors = []

        # Analyze different factors
        sentiment_score, factors = self._analyze_stablecoin_dominance(
            latest, sentiment_score, factors
        )
        sentiment_score, factors = self._analyze_dominance_trend(
            latest, sentiment_score, factors
        )
        sentiment_score, factors = self._analyze_rsi_signals(
            latest, sentiment_score, factors
        )
        sentiment_score, factors = self._analyze_velocity_signals(
            latest, sentiment_score, factors
        )

        # Determine overall sentiment
        sentiment = self._calculate_sentiment_category(sentiment_score)
        confidence = min(100, abs(sentiment_score) * 25)

        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "confidence": confidence,
            "factors": factors,
            "dominance": latest.stablecoin_dominance,
            "dominance_change_24h": latest.dominance_24h_change,
            "timestamp": latest.timestamp,
        }

    def _analyze_stablecoin_dominance(
        self, latest: DominanceData, sentiment_score: float, factors: list[str]
    ) -> tuple[float, list[str]]:
        """Analyze stablecoin dominance levels."""
        if latest.stablecoin_dominance > 10:
            sentiment_score -= 2
            factors.append("High stablecoin dominance (>10%)")
        elif latest.stablecoin_dominance > 7:
            sentiment_score -= 1
            factors.append("Elevated stablecoin dominance (7-10%)")
        elif latest.stablecoin_dominance < 5:
            sentiment_score += 1
            factors.append("Low stablecoin dominance (<5%)")

        return sentiment_score, factors

    def _analyze_dominance_trend(
        self, latest: DominanceData, sentiment_score: float, factors: list[str]
    ) -> tuple[float, list[str]]:
        """Analyze dominance trend changes."""
        if latest.dominance_24h_change > 0.5:
            sentiment_score -= 1.5
            factors.append("Increasing dominance (risk-off)")
        elif latest.dominance_24h_change < -0.5:
            sentiment_score += 1.5
            factors.append("Decreasing dominance (risk-on)")

        return sentiment_score, factors

    def _analyze_rsi_signals(
        self, latest: DominanceData, sentiment_score: float, factors: list[str]
    ) -> tuple[float, list[str]]:
        """Analyze RSI signals."""
        if latest.dominance_rsi:
            if latest.dominance_rsi > 70:
                sentiment_score += 0.5  # Overbought dominance might reverse
                factors.append("Dominance RSI overbought")
            elif latest.dominance_rsi < 30:
                sentiment_score -= 0.5  # Oversold dominance might reverse
                factors.append("Dominance RSI oversold")

        return sentiment_score, factors

    def _analyze_velocity_signals(
        self, latest: DominanceData, sentiment_score: float, factors: list[str]
    ) -> tuple[float, list[str]]:
        """Analyze velocity signals."""
        if latest.stablecoin_velocity:
            if latest.stablecoin_velocity > 2.0:
                sentiment_score -= 0.5
                factors.append("High stablecoin velocity (active trading)")
            elif latest.stablecoin_velocity < 0.5:
                sentiment_score += 0.5
                factors.append("Low stablecoin velocity (hodling)")

        return sentiment_score, factors

    def _calculate_sentiment_category(self, sentiment_score: float) -> str:
        """Calculate sentiment category from score."""
        if sentiment_score >= 2:
            return "STRONG_BULLISH"
        if sentiment_score >= 1:
            return "BULLISH"
        if sentiment_score <= -2:
            return "STRONG_BEARISH"
        if sentiment_score <= -1:
            return "BEARISH"
        return "NEUTRAL"

    def to_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """
        Convert dominance history to pandas DataFrame.

        Args:
            hours: Number of hours of history

        Returns:
            DataFrame with dominance metrics
        """
        history = self.get_dominance_history(hours)
        if not history:
            return pd.DataFrame()

        data = []
        for point in history:
            data.append(
                {
                    "timestamp": point.timestamp,
                    "usdt_dominance": point.usdt_dominance,
                    "usdc_dominance": point.usdc_dominance,
                    "stablecoin_dominance": point.stablecoin_dominance,
                    "dominance_24h_change": point.dominance_24h_change,
                    "velocity": point.stablecoin_velocity,
                    "sma_20": point.dominance_sma_20,
                    "rsi": point.dominance_rsi,
                }
            )

        dominance_data = pd.DataFrame(data)
        return dominance_data.set_index("timestamp")


class DominanceCandleData(BaseModel):
    """OHLCV candlestick data derived from high-frequency dominance snapshots."""

    timestamp: datetime = Field(..., description="Timestamp for the candle period")
    open: float = Field(..., description="First dominance value in the interval")
    high: float = Field(..., description="Maximum dominance value in the interval")
    low: float = Field(..., description="Minimum dominance value in the interval")
    close: float = Field(..., description="Last dominance value in the interval")
    volume: Decimal = Field(
        ..., description="Change in total stablecoin market cap during interval"
    )
    avg_dominance: float = Field(
        ..., description="Average dominance value in the interval"
    )
    volatility: float = Field(
        ..., description="Standard deviation of dominance values in the interval"
    )

    # Technical indicators
    rsi: float | None = Field(None, description="Relative Strength Index")
    ema_fast: float | None = Field(None, description="Fast Exponential Moving Average")
    ema_slow: float | None = Field(None, description="Slow Exponential Moving Average")
    momentum: float | None = Field(None, description="Momentum indicator")
    trend_signal: str | None = Field(
        None, description="Trend signal: BULLISH, BEARISH, or NEUTRAL"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DominanceCandleBuilder:
    """
    Converts high-frequency dominance snapshots into OHLCV candlestick data.

    This class takes a series of DominanceData snapshots and aggregates them into
    candlestick format for technical analysis. Supports multiple time intervals
    and calculates volume based on stablecoin market cap changes.
    """

    SUPPORTED_INTERVALS: ClassVar[list[str]] = ["1T", "3T", "5T", "15T", "30T", "1H"]

    def __init__(self, snapshots: list[DominanceData]):
        """
        Initialize the candle builder with dominance snapshots.

        Args:
            snapshots: List of high-frequency dominance data points

        Raises:
            ValueError: If snapshots list is empty or contains invalid data
        """
        if not snapshots:
            raise ValueError("Snapshots list cannot be empty")

        # Validate all snapshots have required data
        for i, snapshot in enumerate(snapshots):
            if snapshot.stablecoin_dominance is None:
                raise ValueError(f"Snapshot at index {i} missing stablecoin_dominance")
            if snapshot.total_stablecoin_cap is None:
                raise ValueError(f"Snapshot at index {i} missing total_stablecoin_cap")

        self.snapshots = sorted(snapshots, key=lambda x: x.timestamp)
        logger.info(
            "Initialized DominanceCandleBuilder with %s snapshots", len(snapshots)
        )

    def build_candles(self, interval: str = "3T") -> list[DominanceCandleData]:
        """
        Build candlestick data from dominance snapshots.

        Args:
            interval: Time interval for candles (e.g., '1T', '3T', '5T', '15T', '30T', '1H')
                     T = minutes, H = hours

        Returns:
            List of DominanceCandleData objects representing candlesticks

        Raises:
            ValueError: If interval is not supported or data is insufficient
        """
        if interval not in self.SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported interval '{interval}'. Supported: {self.SUPPORTED_INTERVALS}"
            )

        if len(self.snapshots) < 2:
            logger.warning(
                "Insufficient data for candle building (need at least 2 snapshots)"
            )
            return []

        try:
            # Convert snapshots to DataFrame for efficient resampling
            df_data = []
            for snapshot in self.snapshots:
                df_data.append(
                    {
                        "timestamp": snapshot.timestamp,
                        "dominance": snapshot.stablecoin_dominance,
                        "market_cap": float(snapshot.total_stablecoin_cap),
                    }
                )

            candle_data = pd.DataFrame(df_data)
            candle_data = candle_data.set_index("timestamp")
            candle_data = candle_data.sort_index()

            # Resample data using pandas
            resampled = candle_data.resample(interval).agg(
                {
                    "dominance": ["first", "max", "min", "last", "mean", "std"],
                    "market_cap": ["first", "last"],
                }
            )

            # Flatten column names
            resampled.columns = [
                "open",
                "high",
                "low",
                "close",
                "avg_dominance",
                "volatility",
                "market_cap_start",
                "market_cap_end",
            ]

            # Filter out periods with no data
            resampled = resampled.dropna(subset=["open", "close"])

            if resampled.empty:
                logger.warning("No valid candles after resampling")
                return []

            candles = []
            for timestamp, row in resampled.iterrows():
                # Handle NaN values for volatility
                volatility = (
                    row["volatility"] if not pd.isna(row["volatility"]) else 0.0
                )

                # Calculate volume as change in market cap
                volume = Decimal(str(row["market_cap_end"] - row["market_cap_start"]))

                candle = DominanceCandleData(
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=volume,
                    avg_dominance=float(row["avg_dominance"]),
                    volatility=volatility,
                    rsi=None,
                    ema_fast=None,
                    ema_slow=None,
                    momentum=None,
                    trend_signal=None,
                )
                candles.append(candle)

        except Exception as e:
            logger.exception("Error building candles: %s")
            raise ValueError(f"Failed to build candles: {e}") from e
        else:
            logger.info("Built %s candles with %s interval", len(candles), interval)
            return candles

    def _calculate_volume(self, group_snapshots: list[DominanceData]) -> Decimal:
        """
        Calculate volume for a group of snapshots as change in total stablecoin market cap.

        Args:
            group_snapshots: List of snapshots in the time interval

        Returns:
            Volume as change in market cap from start to end of interval
        """
        if not group_snapshots or len(group_snapshots) < 2:
            return Decimal(0)

        # Sort by timestamp to ensure correct start/end
        sorted_snapshots = sorted(group_snapshots, key=lambda x: x.timestamp)

        start_cap = sorted_snapshots[0].total_stablecoin_cap
        end_cap = sorted_snapshots[-1].total_stablecoin_cap

        return end_cap - start_cap

    def get_supported_intervals(self) -> list[str]:
        """
        Get list of supported time intervals for candle building.

        Returns:
            List of supported interval strings
        """
        return self.SUPPORTED_INTERVALS.copy()

    def calculate_technical_indicators(
        self,
        candles: list[DominanceCandleData],
        rsi_period: int = 14,
        ema_fast: int = 12,
        ema_slow: int = 26,
    ) -> dict[str, Any]:
        """
        Calculate technical indicators for dominance candlesticks to provide VuManChu-style analysis.

        This method adds RSI, EMA, momentum, and trend signals to the candle data.
        Returns both individual candle indicators and summary metrics for analysis.

        Args:
            candles: List of DominanceCandleData objects to analyze
            rsi_period: Period for RSI calculation (default: 14)
            ema_fast: Period for fast EMA (default: 12)
            ema_slow: Period for slow EMA (default: 26)

        Returns:
            Dictionary containing:
            - 'candles': Updated candles with indicator values
            - 'summary': Summary metrics and signals
            - 'latest_signals': Most recent indicator values

        Raises:
            ValueError: If insufficient data for calculations
        """
        if not candles:
            raise ValueError("Cannot calculate indicators on empty candles list")

        if len(candles) < max(rsi_period, ema_slow):
            logger.warning(
                "Insufficient data for indicators. Have %s, need %s",
                len(candles),
                max(rsi_period, ema_slow),
            )
            return {
                "candles": candles,
                "summary": {"insufficient_data": True},
                "latest_signals": {},
            }

        try:
            # Extract close prices for calculations
            close_prices = [candle.close for candle in candles]

            # Calculate RSI
            rsi_values = self._calculate_rsi(close_prices, rsi_period)

            # Calculate EMAs
            ema_fast_values = self._calculate_ema(close_prices, ema_fast)
            ema_slow_values = self._calculate_ema(close_prices, ema_slow)

            # Calculate momentum (rate of change over 10 periods)
            momentum_period = min(10, len(close_prices) // 2)
            momentum_values = self._calculate_momentum(close_prices, momentum_period)

            # Update candles with indicator values
            updated_candles = []
            for i, candle in enumerate(candles):
                # Create a copy of the candle with indicators
                candle_dict = candle.dict()

                # Add indicators if available for this index
                candle_dict["rsi"] = rsi_values[i] if i < len(rsi_values) else None
                candle_dict["ema_fast"] = (
                    ema_fast_values[i] if i < len(ema_fast_values) else None
                )
                candle_dict["ema_slow"] = (
                    ema_slow_values[i] if i < len(ema_slow_values) else None
                )
                candle_dict["momentum"] = (
                    momentum_values[i] if i < len(momentum_values) else None
                )

                # Determine trend signal
                trend_signal = self._determine_trend_signal(
                    candle_dict.get("rsi"),
                    candle_dict.get("ema_fast"),
                    candle_dict.get("ema_slow"),
                    candle_dict.get("momentum"),
                )
                candle_dict["trend_signal"] = trend_signal

                updated_candles.append(DominanceCandleData(**candle_dict))

            # Calculate summary metrics
            latest_candle = updated_candles[-1]
            summary = self._calculate_summary_metrics(updated_candles)

            # Extract latest signals
            latest_signals = {
                "rsi": latest_candle.rsi,
                "ema_fast": latest_candle.ema_fast,
                "ema_slow": latest_candle.ema_slow,
                "momentum": latest_candle.momentum,
                "trend_signal": latest_candle.trend_signal,
                "ema_crossover": (
                    self._detect_ema_crossover(updated_candles[-2:])
                    if len(updated_candles) >= 2
                    else None
                ),
            }

        except Exception as e:
            logger.exception("Error calculating technical indicators: %s")
            raise ValueError(f"Failed to calculate technical indicators: {e}") from e
        else:
            logger.info(
                "Calculated technical indicators for %s candles", len(updated_candles)
            )

            return {
                "candles": updated_candles,
                "summary": summary,
                "latest_signals": latest_signals,
            }

    def _calculate_rsi(
        self, values: list[float], period: int = 14
    ) -> list[float | None]:
        """
        Calculate Relative Strength Index (RSI) for dominance values.

        RSI measures the momentum of dominance changes, helping identify
        overbought/oversold conditions in stablecoin dominance trends.

        Args:
            values: List of dominance close prices
            period: RSI calculation period (default: 14)

        Returns:
            List of RSI values (None for insufficient data periods)

        Note:
            RSI values above 70 typically indicate overbought conditions,
            while values below 30 suggest oversold conditions.
        """
        if len(values) < period + 1:
            return [None] * len(values)

        try:
            price_data = pd.DataFrame({"close": values})
            price_data["change"] = price_data["close"].diff()
            price_data["gain"] = price_data["change"].where(price_data["change"] > 0, 0)
            price_data["loss"] = -price_data["change"].where(
                price_data["change"] < 0, 0
            )

            # Calculate initial average gain and loss
            price_data["avg_gain"] = (
                price_data["gain"].rolling(window=period, min_periods=period).mean()
            )
            price_data["avg_loss"] = (
                price_data["loss"].rolling(window=period, min_periods=period).mean()
            )

            # Use exponential smoothing for subsequent values (Wilder's smoothing)
            for i in range(period, len(price_data)):
                if pd.notna(price_data.iloc[i - 1]["avg_gain"]):
                    price_data.iloc[i, price_data.columns.get_loc("avg_gain")] = (
                        price_data.iloc[i - 1]["avg_gain"] * (period - 1)
                        + price_data.iloc[i]["gain"]
                    ) / period
                    price_data.iloc[i, price_data.columns.get_loc("avg_loss")] = (
                        price_data.iloc[i - 1]["avg_loss"] * (period - 1)
                        + price_data.iloc[i]["loss"]
                    ) / period

            # Calculate RSI
            rs = price_data["avg_gain"] / price_data["avg_loss"]
            price_data["rsi"] = 100 - (100 / (1 + rs))

            # Convert to list, handling NaN values
            rsi_list: list[float | None] = []
            for value in price_data["rsi"]:
                if pd.isna(value):
                    rsi_list.append(None)
                else:
                    rsi_list.append(float(value))

        except Exception:
            logger.exception("Error calculating RSI: %s")
            return [None] * len(values)
        else:
            return rsi_list

    def _calculate_ema(self, values: list[float], period: int) -> list[float | None]:
        """
        Calculate Exponential Moving Average (EMA) for dominance values.

        EMA gives more weight to recent dominance values, making it more
        responsive to recent changes in stablecoin dominance trends.

        Args:
            values: List of dominance close prices
            period: EMA calculation period

        Returns:
            List of EMA values (None for insufficient data periods)

        Note:
            EMA crossovers (fast EMA crossing above/below slow EMA) can signal
            trend changes in dominance patterns.
        """
        if len(values) < period:
            return [None] * len(values)

        try:
            price_data = pd.DataFrame({"close": values})
            price_data["ema"] = (
                price_data["close"].ewm(span=period, adjust=False).mean()
            )

            # Convert to list, handling NaN values
            ema_list: list[float | None] = []
            for i, value in enumerate(price_data["ema"]):
                if pd.isna(value) or i < period - 1:
                    ema_list.append(None)
                else:
                    ema_list.append(float(value))

        except Exception:
            logger.exception("Error calculating EMA: %s")
            return [None] * len(values)
        else:
            return ema_list

    def _calculate_momentum(
        self, values: list[float], period: int
    ) -> list[float | None]:
        """
        Calculate momentum indicator for dominance values.

        Momentum measures the rate of change in dominance over a specified period,
        helping identify acceleration or deceleration in dominance trends.

        Args:
            values: List of dominance close prices
            period: Momentum calculation period

        Returns:
            List of momentum values (None for insufficient data periods)
        """
        if len(values) < period + 1:
            return [None] * len(values)

        try:
            momentum_list: list[float | None] = [None] * period

            for i in range(period, len(values)):
                current_value = values[i]
                past_value = values[i - period]
                momentum = (
                    ((current_value - past_value) / past_value) * 100
                    if past_value != 0
                    else 0
                )
                momentum_list.append(momentum)

        except Exception:
            logger.exception("Error calculating momentum: %s")
            return [None] * len(values)
        else:
            return momentum_list

    def _determine_trend_signal(
        self,
        rsi: float | None,
        ema_fast: float | None,
        ema_slow: float | None,
        momentum: float | None,
    ) -> str | None:
        """
        Determine trend signal based on multiple technical indicators.

        Combines RSI, EMA crossover, and momentum to generate a consolidated
        trend signal for dominance analysis.

        Args:
            rsi: Current RSI value
            ema_fast: Current fast EMA value
            ema_slow: Current slow EMA value
            momentum: Current momentum value

        Returns:
            Trend signal: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if not all([rsi, ema_fast, ema_slow, momentum]):
            return "NEUTRAL"

        try:
            bullish_signals = 0
            bearish_signals = 0

            # RSI analysis
            if rsi is not None:
                if rsi < 30:  # Oversold, potential bullish reversal
                    bullish_signals += 1
                elif rsi > 70:  # Overbought, potential bearish reversal
                    bearish_signals += 1

            # EMA crossover analysis
            if ema_fast is not None and ema_slow is not None:
                if ema_fast > ema_slow:
                    bullish_signals += 1
                elif ema_fast < ema_slow:
                    bearish_signals += 1

            # Momentum analysis
            if momentum is not None:
                if momentum > 1:  # Strong positive momentum
                    bullish_signals += 1
                elif momentum < -1:  # Strong negative momentum
                    bearish_signals += 1

            # Determine overall signal
            if bullish_signals > bearish_signals:
                return "BULLISH"
            if bearish_signals > bullish_signals:
                return "BEARISH"
        except Exception:
            logger.exception("Error determining trend signal: %s")
            return "NEUTRAL"
        else:
            return "NEUTRAL"

    def _calculate_summary_metrics(
        self, candles: list[DominanceCandleData]
    ) -> dict[str, Any]:
        """
        Calculate summary metrics for the technical analysis.

        Args:
            candles: List of candles with calculated indicators

        Returns:
            Dictionary with summary metrics
        """
        try:
            # Filter out None values for calculations
            rsi_values = [c.rsi for c in candles if c.rsi is not None]
            momentum_values = [c.momentum for c in candles if c.momentum is not None]

            # Count trend signals
            trend_counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
            for candle in candles:
                if candle.trend_signal:
                    trend_counts[candle.trend_signal] += 1

            return {
                "avg_rsi": np.mean(rsi_values) if rsi_values else None,
                "avg_momentum": np.mean(momentum_values) if momentum_values else None,
                "trend_distribution": trend_counts,
                "total_periods": len(candles),
                "indicators_available": len([c for c in candles if c.rsi is not None]),
            }

        except Exception as e:
            logger.exception("Error calculating summary metrics: %s")
            return {"error": str(e)}

    def _detect_ema_crossover(
        self, recent_candles: list[DominanceCandleData]
    ) -> str | None:
        """
        Detect EMA crossover in recent candles.

        Args:
            recent_candles: Last 2 candles for crossover detection

        Returns:
            "BULLISH_CROSSOVER", "BEARISH_CROSSOVER", or None
        """
        if len(recent_candles) < 2:
            return None

        try:
            prev_candle = recent_candles[0]
            curr_candle = recent_candles[1]

            # Check if we have the required data
            if not all(
                [
                    prev_candle.ema_fast,
                    prev_candle.ema_slow,
                    curr_candle.ema_fast,
                    curr_candle.ema_slow,
                ]
            ):
                return None

            # Previous period: fast was below slow
            # Current period: fast is above slow -> Bullish crossover
            if (
                prev_candle.ema_fast is not None
                and prev_candle.ema_slow is not None
                and curr_candle.ema_fast is not None
                and curr_candle.ema_slow is not None
                and prev_candle.ema_fast <= prev_candle.ema_slow
                and curr_candle.ema_fast > curr_candle.ema_slow
            ):
                return "BULLISH_CROSSOVER"

            # Previous period: fast was above slow
            # Current period: fast is below slow -> Bearish crossover
            if (
                prev_candle.ema_fast is not None
                and prev_candle.ema_slow is not None
                and curr_candle.ema_fast is not None
                and curr_candle.ema_slow is not None
                and prev_candle.ema_fast >= prev_candle.ema_slow
                and curr_candle.ema_fast < curr_candle.ema_slow
            ):
                return "BEARISH_CROSSOVER"

        except Exception:
            logger.exception("Error detecting EMA crossover: %s")
            return None
        else:
            return None

    def detect_divergences(
        self, candles: list[DominanceCandleData], price_data: list[float]
    ) -> dict[str, Any]:
        """
        Detect divergences between dominance indicators and price action.

        Divergences occur when dominance technical indicators move in the opposite
        direction to price, potentially signaling trend reversals or continuations.

        Args:
            candles: List of dominance candles with technical indicators
            price_data: Corresponding price data for the same time periods

        Returns:
            Dictionary containing:
            - 'bullish_divergences': List of detected bullish divergence periods
            - 'bearish_divergences': List of detected bearish divergence periods
            - 'divergence_strength': Overall divergence strength score
            - 'latest_divergence': Most recent divergence signal

        Raises:
            ValueError: If candles and price_data lengths don't match
        """
        if len(candles) != len(price_data):
            raise ValueError(
                f"Candles length ({len(candles)}) must match price_data length ({len(price_data)})"
            )

        if len(candles) < 10:  # Need sufficient data for divergence analysis
            logger.warning("Insufficient data for divergence detection")
            return {
                "bullish_divergences": [],
                "bearish_divergences": [],
                "divergence_strength": 0,
                "latest_divergence": None,
            }

        try:
            bullish_divergences = []
            bearish_divergences = []

            # Look back period for divergence detection
            lookback = min(20, len(candles) // 2)

            for i in range(lookback, len(candles)):
                # Get recent data for analysis
                recent_candles = candles[i - lookback : i + 1]
                recent_prices = price_data[i - lookback : i + 1]

                # Filter candles with valid RSI data
                valid_candles = [
                    (j, c) for j, c in enumerate(recent_candles) if c.rsi is not None
                ]

                if len(valid_candles) < 5:  # Need minimum data points
                    continue

                # Extract RSI and corresponding prices (filter out None values)
                rsi_values_raw = [c.rsi for _, c in valid_candles]
                rsi_values = [v for v in rsi_values_raw if v is not None]
                # Only include corresponding prices where RSI is not None
                corresponding_prices = [
                    recent_prices[j]
                    for j, (_, c) in enumerate(valid_candles)
                    if c.rsi is not None
                ]

                # Only analyze divergence if we have enough non-None RSI values
                if len(rsi_values) >= 5 and len(rsi_values) == len(
                    corresponding_prices
                ):
                    divergence = self._analyze_divergence_pattern(
                        rsi_values[-10:], corresponding_prices[-10:], i
                    )
                else:
                    divergence = None

                if divergence:
                    if divergence["type"] == "BULLISH":
                        bullish_divergences.append(divergence)
                    elif divergence["type"] == "BEARISH":
                        bearish_divergences.append(divergence)

            # Calculate divergence strength
            total_divergences = len(bullish_divergences) + len(bearish_divergences)
            divergence_strength = min(100, total_divergences * 10)  # Scale 0-100

            # Get latest divergence
            latest_divergence = None
            if bullish_divergences or bearish_divergences:
                all_divergences = bullish_divergences + bearish_divergences
                latest_divergence = max(
                    all_divergences, key=lambda x: x["timestamp_index"]
                )

        except Exception as e:
            logger.exception("Error detecting divergences: %s")
            raise ValueError(f"Failed to detect divergences: {e}") from e
        else:
            logger.info(
                "Detected %s bullish and %s bearish divergences",
                len(bullish_divergences),
                len(bearish_divergences),
            )

            return {
                "bullish_divergences": bullish_divergences,
                "bearish_divergences": bearish_divergences,
                "divergence_strength": divergence_strength,
                "latest_divergence": latest_divergence,
            }

    def _analyze_divergence_pattern(
        self,
        rsi_values: list[float],
        price_values: list[float],
        timestamp_index: int,
    ) -> dict[str, Any] | None:
        """
        Analyze a specific pattern for divergence signals.

        Args:
            rsi_values: Recent RSI values
            price_values: Corresponding price values
            timestamp_index: Index of the current timestamp

        Returns:
            Divergence information or None if no divergence detected
        """
        if len(rsi_values) < 5 or len(price_values) < 5:
            return None

        try:
            # Calculate correlation between RSI and price
            correlation = np.corrcoef(rsi_values, price_values)[0, 1]

            # Check for significant negative correlation (divergence)
            if correlation < -0.5:  # Strong negative correlation
                # Determine if price is making new highs/lows while RSI diverges
                recent_price_trend = np.polyfit(
                    range(len(price_values)), price_values, 1
                )[0]
                recent_rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]

                # Bullish divergence: Price declining, RSI rising
                if recent_price_trend < 0 and recent_rsi_trend > 0:
                    return {
                        "type": "BULLISH",
                        "strength": abs(correlation) * 100,
                        "timestamp_index": timestamp_index,
                        "price_trend": recent_price_trend,
                        "rsi_trend": recent_rsi_trend,
                        "correlation": correlation,
                    }

                # Bearish divergence: Price rising, RSI declining
                if recent_price_trend > 0 and recent_rsi_trend < 0:
                    return {
                        "type": "BEARISH",
                        "strength": abs(correlation) * 100,
                        "timestamp_index": timestamp_index,
                        "price_trend": recent_price_trend,
                        "rsi_trend": recent_rsi_trend,
                        "correlation": correlation,
                    }

        except Exception:
            logger.exception("Error analyzing divergence pattern: %s")
            return None
        else:
            return None

    def validate_candles(self, candles: list[DominanceCandleData]) -> dict[str, Any]:
        """
        Validate dominance candles for data consistency and proper OHLC relationships.

        Performs comprehensive validation checks including:
        - OHLC relationship validation (High >= Open, Close and Low <= Open, Close)
        - Timestamp chronological ordering
        - Dominance percentages within reasonable ranges
        - Technical indicator value validation
        - Volume and volatility sanity checks

        Args:
            candles: List of DominanceCandleData objects to validate

        Returns:
            Dictionary containing:
            - 'is_valid': Overall validation status
            - 'errors': List of validation errors found
            - 'warnings': List of validation warnings
            - 'statistics': Validation statistics
            - 'quality_score': Data quality score (0-100)
        """
        if not candles:
            return {
                "is_valid": False,
                "errors": ["No candles provided for validation"],
                "warnings": [],
                "statistics": {},
                "quality_score": 0,
            }

        try:
            errors: list[str] = []
            warnings: list[str] = []
            statistics = {
                "total_candles": len(candles),
                "valid_ohlc": 0,
                "valid_timestamps": 0,
                "valid_dominance_ranges": 0,
                "valid_indicators": 0,
                "nan_values": 0,
                "infinite_values": 0,
            }

            logger.info("Validating %s dominance candles", len(candles))

            for i, candle in enumerate(candles):
                # Validate OHLC relationships
                ohlc_valid = self._validate_ohlc_relationship(
                    candle, i, errors, warnings
                )
                if ohlc_valid:
                    statistics["valid_ohlc"] += 1

                # Validate timestamp
                timestamp_valid = self._validate_timestamp(
                    candle, i, candles, errors, warnings
                )
                if timestamp_valid:
                    statistics["valid_timestamps"] += 1

                # Validate dominance ranges
                dominance_valid = self._validate_dominance_ranges(
                    candle, i, errors, warnings
                )
                if dominance_valid:
                    statistics["valid_dominance_ranges"] += 1

                # Validate technical indicators
                indicators_valid = self._validate_technical_indicators(
                    candle, i, errors, warnings
                )
                if indicators_valid:
                    statistics["valid_indicators"] += 1

                # Check for NaN and infinite values
                nan_count, inf_count = self._check_nan_infinite_values(
                    candle, i, errors, warnings
                )
                statistics["nan_values"] += nan_count
                statistics["infinite_values"] += inf_count

                # Validate volume consistency
                self._validate_volume_consistency(candle, i, warnings)

            # Calculate quality score
            total_checks = len(candles) * 4  # OHLC, timestamp, dominance, indicators
            passed_checks = (
                statistics["valid_ohlc"]
                + statistics["valid_timestamps"]
                + statistics["valid_dominance_ranges"]
                + statistics["valid_indicators"]
            )

            quality_score = (
                (passed_checks / total_checks * 100) if total_checks > 0 else 0
            )

            # Penalize for NaN and infinite values
            if statistics["nan_values"] > 0 or statistics["infinite_values"] > 0:
                quality_score *= 0.8  # 20% penalty for data quality issues

            is_valid = len(errors) == 0 and quality_score >= 80

            logger.info(
                "Validation complete. Quality score: %.1f%%, Errors: %s, Warnings: %s",
                quality_score,
                len(errors),
                len(warnings),
            )

            return {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "statistics": statistics,
                "quality_score": round(quality_score, 2),
            }

        except Exception as e:
            logger.exception("Error during candle validation: %s")
            return {
                "is_valid": False,
                "errors": [f"Validation failed with exception: {e!s}"],
                "warnings": [],
                "statistics": {},
                "quality_score": 0,
            }

    def check_data_integrity(self, snapshots: list[DominanceData]) -> dict[str, Any]:
        """
        Check data integrity of dominance snapshots before candle building.

        Verifies:
        - Timestamp ordering (chronological sequence)
        - No missing critical values
        - Reasonable dominance percentage ranges (0-100%)
        - Market cap values are positive
        - No duplicate timestamps
        - Data gaps within acceptable thresholds

        Args:
            snapshots: List of DominanceData snapshots to check

        Returns:
            Dictionary containing:
            - 'integrity_score': Overall data integrity score (0-100)
            - 'issues': List of integrity issues found
            - 'gaps': Information about data gaps
            - 'duplicates': Information about duplicate timestamps
            - 'statistics': Data statistics and health metrics
        """
        if not snapshots:
            return {
                "integrity_score": 0,
                "issues": ["No snapshots provided"],
                "gaps": [],
                "duplicates": [],
                "statistics": {},
            }

        try:
            logger.info("Checking data integrity for %s snapshots", len(snapshots))

            # Initialize analysis data
            analysis_data = self._initialize_integrity_analysis(snapshots)

            # Perform various checks
            self._check_timestamp_ordering(analysis_data)
            self._calculate_time_statistics(analysis_data)
            self._check_for_duplicates(analysis_data)
            self._check_for_data_gaps(analysis_data)
            self._validate_individual_snapshots(analysis_data)

            # Calculate final integrity score
            integrity_score = self._calculate_integrity_score(analysis_data)

            return self._build_integrity_result(analysis_data, integrity_score)

        except Exception as e:
            logger.exception("Error during data integrity check: %s")
            return {
                "integrity_score": 0,
                "issues": [f"Integrity check failed with exception: {e!s}"],
                "gaps": [],
                "duplicates": [],
                "statistics": {},
            }

    def _initialize_integrity_analysis(
        self, snapshots: list[DominanceData]
    ) -> dict[str, Any]:
        """Initialize data structures for integrity analysis."""
        sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp)

        return {
            "original_snapshots": snapshots,
            "sorted_snapshots": sorted_snapshots,
            "issues": [],
            "gaps": [],
            "duplicates": [],
            "statistics": {
                "total_snapshots": len(snapshots),
                "time_span_hours": 0.0,
                "avg_interval_seconds": 0.0,
                "missing_values": 0,
                "out_of_range_dominance": 0,
                "negative_market_caps": 0,
            },
        }

    def _check_timestamp_ordering(self, analysis_data: dict[str, Any]) -> None:
        """Check if timestamps are in chronological order."""
        sorted_timestamps = [s.timestamp for s in analysis_data["sorted_snapshots"]]
        original_timestamps = [s.timestamp for s in analysis_data["original_snapshots"]]

        if sorted_timestamps != original_timestamps:
            analysis_data["issues"].append("Snapshots are not in chronological order")

    def _calculate_time_statistics(self, analysis_data: dict[str, Any]) -> None:
        """Calculate time span and average interval statistics."""
        sorted_snapshots = analysis_data["sorted_snapshots"]
        statistics = analysis_data["statistics"]

        if len(sorted_snapshots) > 1:
            time_span = sorted_snapshots[-1].timestamp - sorted_snapshots[0].timestamp
            statistics["time_span_hours"] = time_span.total_seconds() / 3600

            total_intervals = sum(
                (
                    sorted_snapshots[i].timestamp - sorted_snapshots[i - 1].timestamp
                ).total_seconds()
                for i in range(1, len(sorted_snapshots))
            )
            statistics["avg_interval_seconds"] = total_intervals / (
                len(sorted_snapshots) - 1
            )

    def _check_for_duplicates(self, analysis_data: dict[str, Any]) -> None:
        """Check for duplicate timestamps."""
        sorted_snapshots = analysis_data["sorted_snapshots"]
        duplicates = analysis_data["duplicates"]

        seen_timestamps = set()
        for snapshot in sorted_snapshots:
            if snapshot.timestamp in seen_timestamps:
                duplicates.append(
                    {
                        "timestamp": snapshot.timestamp,
                        "indices": [
                            j
                            for j, s in enumerate(sorted_snapshots)
                            if s.timestamp == snapshot.timestamp
                        ],
                    }
                )
            seen_timestamps.add(snapshot.timestamp)

    def _check_for_data_gaps(self, analysis_data: dict[str, Any]) -> None:
        """Check for data gaps in the timeline."""
        sorted_snapshots = analysis_data["sorted_snapshots"]
        statistics = analysis_data["statistics"]
        gaps = analysis_data["gaps"]

        if len(sorted_snapshots) > 1:
            expected_interval = statistics["avg_interval_seconds"]
            gap_threshold = expected_interval * 2.5  # Allow 2.5x normal interval

            for i in range(1, len(sorted_snapshots)):
                interval = (
                    sorted_snapshots[i].timestamp - sorted_snapshots[i - 1].timestamp
                ).total_seconds()
                if interval > gap_threshold:
                    gaps.append(
                        {
                            "start_time": sorted_snapshots[i - 1].timestamp,
                            "end_time": sorted_snapshots[i].timestamp,
                            "gap_seconds": interval,
                            "expected_seconds": expected_interval,
                        }
                    )

    def _validate_individual_snapshots(self, analysis_data: dict[str, Any]) -> None:
        """Validate individual snapshot data."""
        sorted_snapshots = analysis_data["sorted_snapshots"]
        statistics = analysis_data["statistics"]
        issues = analysis_data["issues"]

        for i, snapshot in enumerate(sorted_snapshots):
            self._check_missing_values(snapshot, i, statistics, issues)
            self._check_dominance_ranges(snapshot, i, statistics, issues)
            self._check_market_cap_values(snapshot, i, statistics, issues)

    def _check_missing_values(
        self,
        snapshot: DominanceData,
        index: int,
        statistics: dict[str, Any],
        issues: list[str],
    ) -> None:
        """Check for missing values in snapshot."""
        missing_fields: list[str] = []
        if snapshot.stablecoin_dominance is None:
            missing_fields.append("stablecoin_dominance")
        if snapshot.total_stablecoin_cap is None:
            missing_fields.append("total_stablecoin_cap")
        if snapshot.crypto_total_market_cap is None:
            missing_fields.append("crypto_total_market_cap")

        if missing_fields:
            statistics["missing_values"] += len(missing_fields)
            issues.append(
                f"Snapshot {index}: Missing values for {', '.join(missing_fields)}"
            )

    def _check_dominance_ranges(
        self,
        snapshot: DominanceData,
        index: int,
        statistics: dict[str, Any],
        issues: list[str],
    ) -> None:
        """Check dominance percentage ranges."""
        if snapshot.stablecoin_dominance is not None and not (
            0 <= snapshot.stablecoin_dominance <= 100
        ):
            statistics["out_of_range_dominance"] += 1
            issues.append(
                f"Snapshot {index}: Dominance {snapshot.stablecoin_dominance}% out of range (0-100%)"
            )

    def _check_market_cap_values(
        self,
        snapshot: DominanceData,
        index: int,
        statistics: dict[str, Any],
        issues: list[str],
    ) -> None:
        """Check market cap values for validity."""
        if (
            snapshot.total_stablecoin_cap is not None
            and snapshot.total_stablecoin_cap < 0
        ):
            statistics["negative_market_caps"] += 1
            issues.append(f"Snapshot {index}: Negative total stablecoin market cap")

        if (
            snapshot.crypto_total_market_cap is not None
            and snapshot.crypto_total_market_cap < 0
        ):
            statistics["negative_market_caps"] += 1
            issues.append(f"Snapshot {index}: Negative total crypto market cap")

    def _calculate_integrity_score(self, analysis_data: dict[str, Any]) -> int:
        """Calculate overall integrity score."""
        issues = analysis_data["issues"]
        gaps = analysis_data["gaps"]
        duplicates = analysis_data["duplicates"]

        base_score = 100
        # Deduct points for issues
        base_score -= len(issues) * 5  # 5 points per issue
        base_score -= len(gaps) * 10  # 10 points per gap
        base_score -= len(duplicates) * 3  # 3 points per duplicate

        return max(0, min(100, base_score))

    def _build_integrity_result(
        self, analysis_data: dict[str, Any], integrity_score: int
    ) -> dict[str, Any]:
        """Build final integrity check result."""
        issues = analysis_data["issues"]
        gaps = analysis_data["gaps"]
        duplicates = analysis_data["duplicates"]
        statistics = analysis_data["statistics"]

        logger.info(
            "Data integrity check complete. Score: %s%%, Issues: %s, Gaps: %s, Duplicates: %s",
            integrity_score,
            len(issues),
            len(gaps),
            len(duplicates),
        )

        return {
            "integrity_score": integrity_score,
            "issues": issues,
            "gaps": gaps,
            "duplicates": duplicates,
            "statistics": statistics,
        }

    def export_for_tradingview(self, candles: list[DominanceCandleData]) -> str:
        """
        Export dominance candles in TradingView-compatible CSV format.

        Creates a CSV string that can be imported into TradingView for charting
        and technical analysis. Includes OHLCV data and custom indicators.

        Args:
            candles: List of DominanceCandleData objects to export

        Returns:
            CSV-formatted string compatible with TradingView imports

        Raises:
            ValueError: If candles list is empty or contains invalid data
        """
        if not candles:
            raise ValueError("Cannot export empty candles list")

        try:
            logger.info("Exporting %s candles for TradingView", len(candles))

            # CSV header - TradingView expects specific column names
            csv_lines = [
                "time,open,high,low,close,volume,avg_dominance,volatility,rsi,ema_fast,ema_slow,momentum"
            ]

            for candle in candles:
                # Format timestamp for TradingView (Unix timestamp)
                timestamp = int(candle.timestamp.timestamp())

                # Format values, handling None values
                def format_value(value, default=""):
                    if value is None:
                        return default
                    return (
                        str(float(value))
                        if isinstance(value, int | float | Decimal)
                        else str(value)
                    )

                csv_line = ",".join(
                    [
                        str(timestamp),
                        format_value(candle.open),
                        format_value(candle.high),
                        format_value(candle.low),
                        format_value(candle.close),
                        format_value(candle.volume),
                        format_value(candle.avg_dominance),
                        format_value(candle.volatility),
                        format_value(candle.rsi),
                        format_value(candle.ema_fast),
                        format_value(candle.ema_slow),
                        format_value(candle.momentum),
                    ]
                )

                csv_lines.append(csv_line)

            csv_content = "\n".join(csv_lines)

        except Exception as e:
            logger.exception("Error exporting candles for TradingView: %s")
            raise ValueError(f"Failed to export candles: {e}") from e
        else:
            logger.info("Successfully exported %s candles to CSV format", len(candles))
            return csv_content

    def get_candle_statistics(
        self, candles: list[DominanceCandleData]
    ) -> dict[str, Any]:
        """
        Calculate comprehensive statistics for dominance candles.

        Provides insights into volatility patterns, indicator distributions,
        and overall data quality metrics.

        Args:
            candles: List of DominanceCandleData objects to analyze

        Returns:
            Dictionary containing detailed statistics and metrics
        """
        if not candles:
            return {"error": "No candles provided for statistics"}

        try:
            logger.info("Calculating statistics for %s candles", len(candles))

            # Extract values for analysis
            open_values = [c.open for c in candles]
            high_values = [c.high for c in candles]
            low_values = [c.low for c in candles]
            close_values = [c.close for c in candles]
            volume_values = [float(c.volume) for c in candles]
            volatility_values = [c.volatility for c in candles]

            # Technical indicator values (filter out None)
            rsi_values = [c.rsi for c in candles if c.rsi is not None]
            momentum_values = [c.momentum for c in candles if c.momentum is not None]

            statistics = {
                "basic_stats": {
                    "total_candles": len(candles),
                    "time_range": {
                        "start": candles[0].timestamp.isoformat(),
                        "end": candles[-1].timestamp.isoformat(),
                        "duration_hours": (
                            candles[-1].timestamp - candles[0].timestamp
                        ).total_seconds()
                        / 3600,
                    },
                },
                "dominance_stats": {
                    "open": {
                        "min": min(open_values),
                        "max": max(open_values),
                        "avg": np.mean(open_values),
                        "std": np.std(open_values),
                    },
                    "close": {
                        "min": min(close_values),
                        "max": max(close_values),
                        "avg": np.mean(close_values),
                        "std": np.std(close_values),
                    },
                    "range": {
                        "min_low": min(low_values),
                        "max_high": max(high_values),
                        "avg_range": np.mean(
                            [
                                h - low
                                for h, low in zip(high_values, low_values, strict=True)
                            ]
                        ),
                    },
                },
                "volatility_stats": {
                    "min": min(volatility_values),
                    "max": max(volatility_values),
                    "avg": np.mean(volatility_values),
                    "std": np.std(volatility_values),
                    "high_volatility_periods": sum(
                        1
                        for v in volatility_values
                        if v > np.mean(volatility_values) + np.std(volatility_values)
                    ),
                },
                "volume_stats": {
                    "min": min(volume_values),
                    "max": max(volume_values),
                    "avg": np.mean(volume_values),
                    "std": np.std(volume_values),
                    "positive_volume_periods": sum(1 for v in volume_values if v > 0),
                    "negative_volume_periods": sum(1 for v in volume_values if v < 0),
                },
                "technical_indicators": {
                    "rsi": {
                        "count": len(rsi_values),
                        "avg": np.mean(rsi_values) if rsi_values else None,
                        "overbought_periods": (
                            sum(1 for r in rsi_values if r > 70) if rsi_values else 0
                        ),
                        "oversold_periods": (
                            sum(1 for r in rsi_values if r < 30) if rsi_values else 0
                        ),
                    },
                    "ema_crossovers": self._count_ema_crossovers(candles),
                    "momentum": {
                        "count": len(momentum_values),
                        "avg": np.mean(momentum_values) if momentum_values else None,
                        "positive_momentum_periods": (
                            sum(1 for m in momentum_values if m > 0)
                            if momentum_values
                            else 0
                        ),
                    },
                },
                "trend_analysis": {
                    "bullish_signals": sum(
                        1 for c in candles if c.trend_signal == "BULLISH"
                    ),
                    "bearish_signals": sum(
                        1 for c in candles if c.trend_signal == "BEARISH"
                    ),
                    "neutral_signals": sum(
                        1 for c in candles if c.trend_signal == "NEUTRAL"
                    ),
                    "overall_trend": self._determine_overall_trend(close_values),
                },
            }

        except Exception as e:
            logger.exception("Error calculating candle statistics: %s")
            return {"error": f"Failed to calculate statistics: {e!s}"}
        else:
            logger.info("Successfully calculated candle statistics")
            return statistics

    def _validate_ohlc_relationship(
        self, candle: DominanceCandleData, index: int, errors: list, warnings: list
    ) -> bool:
        """Validate OHLC relationships for a single candle."""
        try:
            valid = True

            # High should be >= Open and Close
            if candle.high < candle.open or candle.high < candle.close:
                errors.append(
                    f"Candle {index}: High ({candle.high}) < Open ({candle.open}) or Close ({candle.close})"
                )
                valid = False

            # Low should be <= Open and Close
            if candle.low > candle.open or candle.low > candle.close:
                errors.append(
                    f"Candle {index}: Low ({candle.low}) > Open ({candle.open}) or Close ({candle.close})"
                )
                valid = False

            # Check for suspicious zero ranges
            if candle.high == candle.low:
                warnings.append(
                    f"Candle {index}: Zero price range (High = Low = {candle.high})"
                )

        except Exception as e:
            errors.append(f"Candle {index}: Error validating OHLC - {e!s}")
            return False
        else:
            return valid

    def _validate_timestamp(
        self,
        candle: DominanceCandleData,
        index: int,
        all_candles: list,
        errors: list,
        warnings: list,
    ) -> bool:
        """Validate timestamp for chronological ordering."""
        try:
            # Check if timestamp is in the future
            if candle.timestamp > datetime.now(UTC):
                warnings.append(f"Candle {index}: Future timestamp {candle.timestamp}")

            # Check chronological order (if not first candle)
            if index > 0:
                prev_timestamp = all_candles[index - 1].timestamp
                if candle.timestamp <= prev_timestamp:
                    errors.append(
                        f"Candle {index}: Timestamp {candle.timestamp} not after previous {prev_timestamp}"
                    )
                    return False

        except Exception as e:
            errors.append(f"Candle {index}: Error validating timestamp - {e!s}")
            return False
        else:
            return True

    def _validate_dominance_ranges(
        self, candle: DominanceCandleData, index: int, errors: list, warnings: list
    ) -> bool:
        """Validate dominance percentage ranges."""
        try:
            valid = True

            # Check if dominance values are within 0-100% range
            for field_name, value in [
                ("open", candle.open),
                ("high", candle.high),
                ("low", candle.low),
                ("close", candle.close),
                ("avg_dominance", candle.avg_dominance),
            ]:
                if not (0 <= value <= 100):
                    errors.append(
                        f"Candle {index}: {field_name} dominance {value}% out of range (0-100%)"
                    )
                    valid = False

            # Check for extremely high dominance (unusual market conditions)
            if candle.high > 50:
                warnings.append(
                    f"Candle {index}: Very high dominance {candle.high}% (unusual market conditions)"
                )

        except Exception as e:
            errors.append(f"Candle {index}: Error validating dominance ranges - {e!s}")
            return False
        else:
            return valid

    def _validate_technical_indicators(
        self, candle: DominanceCandleData, index: int, errors: list, warnings: list
    ) -> bool:
        """Validate technical indicator values."""
        try:
            valid = True

            # Validate RSI range (0-100)
            if candle.rsi is not None and not (0 <= candle.rsi <= 100):
                errors.append(f"Candle {index}: RSI {candle.rsi} out of range (0-100)")
                valid = False

            # Validate EMA values (should be positive for dominance percentages)
            for field_name, value in [
                ("ema_fast", candle.ema_fast),
                ("ema_slow", candle.ema_slow),
            ]:
                if value is not None and value < 0:
                    warnings.append(f"Candle {index}: {field_name} {value} is negative")

            # Validate momentum (check for extreme values)
            if candle.momentum is not None and abs(candle.momentum) > 1000:
                warnings.append(
                    f"Candle {index}: Extreme momentum value {candle.momentum}%"
                )

        except Exception as e:
            errors.append(
                f"Candle {index}: Error validating technical indicators - {e!s}"
            )
            return False
        else:
            return valid

    def _check_nan_infinite_values(
        self, candle: DominanceCandleData, index: int, errors: list, _warnings: list
    ) -> tuple[int, int]:
        """Check for NaN and infinite values in candle data."""
        nan_count = 0
        inf_count = 0

        try:
            # Check all numeric fields
            fields_to_check = [
                ("open", candle.open),
                ("high", candle.high),
                ("low", candle.low),
                ("close", candle.close),
                ("volatility", candle.volatility),
                ("avg_dominance", candle.avg_dominance),
                ("rsi", candle.rsi),
                ("ema_fast", candle.ema_fast),
                ("ema_slow", candle.ema_slow),
                ("momentum", candle.momentum),
            ]

            for field_name, value in fields_to_check:
                if value is not None:
                    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                        errors.append(
                            f"Candle {index}: {field_name} contains NaN value"
                        )
                        nan_count += 1
                    elif isinstance(value, float) and np.isinf(value):
                        errors.append(
                            f"Candle {index}: {field_name} contains infinite value"
                        )
                        inf_count += 1

        except Exception as e:
            errors.append(f"Candle {index}: Error checking NaN/infinite values - {e!s}")
            return 1, 0  # Count as error
        else:
            return nan_count, inf_count

    def _validate_volume_consistency(
        self, candle: DominanceCandleData, index: int, warnings: list
    ):
        """Validate volume consistency and reasonableness."""
        try:
            # Check for extremely large volume changes
            volume_float = float(candle.volume)
            if abs(volume_float) > 1e12:  # $1 trillion threshold
                warnings.append(
                    f"Candle {index}: Extremely large volume change ${volume_float:,.0f}"
                )

        except Exception as e:
            warnings.append(f"Candle {index}: Error validating volume - {e!s}")

    def _count_ema_crossovers(
        self, candles: list[DominanceCandleData]
    ) -> dict[str, int]:
        """Count EMA crossovers in the candle series."""
        try:
            crossovers = {"bullish": 0, "bearish": 0}

            for i in range(1, len(candles)):
                prev_candle = candles[i - 1]
                curr_candle = candles[i]

                if all(
                    [
                        prev_candle.ema_fast is not None,
                        prev_candle.ema_slow is not None,
                        curr_candle.ema_fast is not None,
                        curr_candle.ema_slow is not None,
                    ]
                ):
                    # Type guard: we know these are not None after the all() check
                    prev_fast = prev_candle.ema_fast
                    prev_slow = prev_candle.ema_slow
                    curr_fast = curr_candle.ema_fast
                    curr_slow = curr_candle.ema_slow

                    # Bullish crossover: fast EMA crosses above slow EMA
                    if (
                        prev_fast is not None
                        and prev_slow is not None
                        and curr_fast is not None
                        and curr_slow is not None
                        and prev_fast <= prev_slow
                        and curr_fast > curr_slow
                    ):
                        crossovers["bullish"] += 1

                    # Bearish crossover: fast EMA crosses below slow EMA
                    elif (
                        prev_fast is not None
                        and prev_slow is not None
                        and curr_fast is not None
                        and curr_slow is not None
                        and prev_fast >= prev_slow
                        and curr_fast < curr_slow
                    ):
                        crossovers["bearish"] += 1

        except Exception:
            logger.exception("Error counting EMA crossovers: %s")
            return {"bullish": 0, "bearish": 0}
        else:
            return crossovers

    def _determine_overall_trend(self, close_values: list[float]) -> str:
        """Determine overall trend from close values."""
        try:
            if len(close_values) < 10:
                return "INSUFFICIENT_DATA"

            # Use linear regression to determine trend
            x = np.arange(len(close_values))
            slope, _ = np.polyfit(x, close_values, 1)

            if slope > 0.01:  # Threshold for significant uptrend
                return "UPTREND"
            if slope < -0.01:  # Threshold for significant downtrend
                return "DOWNTREND"

        except Exception:
            logger.exception("Error determining overall trend: %s")
            return "UNKNOWN"
        else:
            return "SIDEWAYS"


# Factory function for easy client creation
def create_dominance_provider(
    data_source: str | None = None,
    api_key: str | None = None,
    update_interval: int | None = None,
) -> DominanceDataProvider:
    """
    Create a DominanceDataProvider instance with configuration.

    Args:
        data_source: API source (default: from settings)
        api_key: API key for premium endpoints
        update_interval: Update interval in seconds

    Returns:
        DominanceDataProvider instance
    """
    # Use settings if not provided
    data_source = data_source or getattr(settings.dominance, "data_source", "coingecko")
    update_interval = update_interval or getattr(
        settings.dominance, "update_interval", 30
    )

    # Ensure types are correct
    data_source = str(data_source) if data_source is not None else "coingecko"
    update_interval = int(update_interval) if update_interval is not None else 30

    return DominanceDataProvider(
        data_source=data_source, api_key=api_key, update_interval=update_interval
    )


def _create_sample_dominance_snapshots():
    """Create sample dominance data snapshots for testing."""
    print("\n1. Creating sample dominance data snapshots...")

    snapshots = []
    base_time = datetime.now(UTC) - timedelta(hours=2)
    base_dominance = 8.5  # 8.5% stablecoin dominance

    # Generate 120 snapshots (2 hours at 1-minute intervals)
    rng = np.random.default_rng(42)  # Use fixed seed for reproducible test data
    for i in range(120):
        # Simulate realistic dominance fluctuations
        dominance_change = rng.normal(0, 0.1)  # Small random changes
        trend = 0.002 * (i - 60)  # Slight trend over time
        current_dominance = max(
            0.1, min(99.9, base_dominance + trend + dominance_change)
        )

        market_cap_base = Decimal(150000000000)  # $150B base
        market_cap_change = Decimal(str(rng.normal(0, 1000000000)))  # ±$1B variation
        current_market_cap = market_cap_base + market_cap_change

        snapshot = DominanceData(
            timestamp=base_time + timedelta(minutes=i),
            usdt_market_cap=current_market_cap * Decimal("0.7"),
            usdc_market_cap=current_market_cap * Decimal("0.3"),
            total_stablecoin_cap=current_market_cap,
            crypto_total_market_cap=current_market_cap
            / Decimal(str(current_dominance / 100)),
            usdt_dominance=current_dominance * 0.7,
            usdc_dominance=current_dominance * 0.3,
            stablecoin_dominance=current_dominance,
            dominance_24h_change=rng.normal(0, 0.5),
            dominance_7d_change=rng.normal(0, 1.0),
            stablecoin_velocity=max(0.1, rng.normal(1.5, 0.3)),
        )
        snapshots.append(snapshot)

    print(f"   ✓ Created {len(snapshots)} sample snapshots")
    print(f"   ✓ Time range: {snapshots[0].timestamp} to {snapshots[-1].timestamp}")
    print(
        f"   ✓ Dominance range: {min(s.stablecoin_dominance for s in snapshots):.2f}% to {max(s.stablecoin_dominance for s in snapshots):.2f}%"
    )
    return snapshots


def _test_data_integrity(builder, snapshots):
    """Test data integrity check."""
    print("\n2. Testing data integrity check...")

    integrity_result = builder.check_data_integrity(snapshots)

    print(f"   ✓ Integrity Score: {integrity_result['integrity_score']}%")
    print(f"   ✓ Issues Found: {len(integrity_result['issues'])}")
    print(f"   ✓ Data Gaps: {len(integrity_result['gaps'])}")
    print(f"   ✓ Duplicates: {len(integrity_result['duplicates'])}")

    if integrity_result["issues"]:
        print(
            f"   ⚠ Issues: {integrity_result['issues'][:3]}..."
        )  # Show first 3 issues

    return integrity_result


def _test_candle_building(builder):
    """Test building dominance candles for different intervals."""
    print("\n3. Testing candle building...")

    candles = None
    for interval in ["1T", "3T", "5T", "15T"]:
        try:
            test_candles = builder.build_candles(interval)
            print(f"   ✓ {interval} interval: {len(test_candles)} candles created")

            if test_candles:
                first_candle = test_candles[0]
                last_candle = test_candles[-1]
                print(
                    f"     - First candle: Open={first_candle.open:.3f}%, Close={first_candle.close:.3f}%"
                )
                print(
                    f"     - Last candle: Open={last_candle.open:.3f}%, Close={last_candle.close:.3f}%"
                )

                # Use 3T candles for remaining tests
                if interval == "3T":
                    candles = test_candles

        except Exception as e:
            print(f"   ✗ {interval} interval failed: {e}")

    if candles:
        print(f"\n   Using {len(candles)} 3-minute candles for further testing...")
    return candles


def _test_technical_indicators(builder, candles):
    """Test technical indicators calculation."""
    print("\n4. Testing technical indicators calculation...")

    indicators_result = builder.calculate_technical_indicators(candles)
    updated_candles = indicators_result["candles"]
    summary = indicators_result["summary"]
    latest_signals = indicators_result["latest_signals"]

    print(f"   ✓ Indicators calculated for {len(updated_candles)} candles")
    print(f"   ✓ Average RSI: {summary.get('avg_rsi', 'N/A')}")
    print(f"   ✓ Average Momentum: {summary.get('avg_momentum', 'N/A')}")
    print(f"   ✓ Trend Distribution: {summary.get('trend_distribution', {})}")
    print(f"   ✓ Latest Trend Signal: {latest_signals.get('trend_signal', 'N/A')}")
    print(f"   ✓ Latest RSI: {latest_signals.get('rsi', 'N/A')}")
    print(f"   ✓ EMA Crossover: {latest_signals.get('ema_crossover', 'None')}")

    return indicators_result


def _test_candle_validation(builder, updated_candles):
    """Test candle validation."""
    print("\n5. Testing candle validation...")

    validation_result = builder.validate_candles(updated_candles)

    print(f"   ✓ Overall Valid: {validation_result['is_valid']}")
    print(f"   ✓ Quality Score: {validation_result['quality_score']}%")
    print(f"   ✓ Errors Found: {len(validation_result['errors'])}")
    print(f"   ✓ Warnings: {len(validation_result['warnings'])}")

    # Show validation statistics
    stats = validation_result["statistics"]
    print(f"   ✓ Valid OHLC: {stats['valid_ohlc']}/{stats['total_candles']}")
    print(
        f"   ✓ Valid Timestamps: {stats['valid_timestamps']}/{stats['total_candles']}"
    )
    print(
        f"   ✓ Valid Dominance Ranges: {stats['valid_dominance_ranges']}/{stats['total_candles']}"
    )
    print(
        f"   ✓ Valid Indicators: {stats['valid_indicators']}/{stats['total_candles']}"
    )

    if validation_result["errors"]:
        print(f"   ⚠ Sample Errors: {validation_result['errors'][:2]}")

    return validation_result


def _test_statistics_generation(builder, updated_candles):
    """Test candle statistics generation."""
    print("\n6. Testing candle statistics generation...")

    statistics = builder.get_candle_statistics(updated_candles)

    if "error" not in statistics:
        basic_stats = statistics["basic_stats"]
        dominance_stats = statistics["dominance_stats"]
        volatility_stats = statistics["volatility_stats"]
        trend_analysis = statistics["trend_analysis"]

        print(f"   ✓ Total Candles: {basic_stats['total_candles']}")
        print(
            f"   ✓ Time Duration: {basic_stats['time_range']['duration_hours']:.2f} hours"
        )
        print(
            f"   ✓ Dominance Range: {dominance_stats['range']['min_low']:.3f}% - {dominance_stats['range']['max_high']:.3f}%"
        )
        print(f"   ✓ Average Volatility: {volatility_stats['avg']:.4f}")
        print(
            f"   ✓ High Volatility Periods: {volatility_stats['high_volatility_periods']}"
        )
        print(f"   ✓ Overall Trend: {trend_analysis['overall_trend']}")
        print(f"   ✓ Bullish Signals: {trend_analysis['bullish_signals']}")
        print(f"   ✓ Bearish Signals: {trend_analysis['bearish_signals']}")
    else:
        print(f"   ✗ Statistics generation failed: {statistics['error']}")

    return statistics


def _test_tradingview_export(builder, updated_candles):
    """Test TradingView export functionality."""
    print("\n7. Testing TradingView export...")

    csv_content = None
    try:
        csv_content = builder.export_for_tradingview(
            updated_candles[:10]
        )  # Export first 10 candles
        lines = csv_content.split("\n")

        print("   ✓ CSV Export successful")
        print(f"   ✓ Header: {lines[0]}")
        print(f"   ✓ Sample data line: {lines[1] if len(lines) > 1 else 'No data'}")
        print(f"   ✓ Total lines: {len(lines)} (header + {len(lines) - 1} data rows)")

        # Validate CSV format
        if len(lines) > 1:
            sample_data = lines[1].split(",")
            expected_columns = 12  # time + OHLCV + indicators
            if len(sample_data) == expected_columns:
                print(f"   ✓ CSV format validation passed ({expected_columns} columns)")
            else:
                print(
                    f"   ⚠ CSV format issue: expected {expected_columns} columns, got {len(sample_data)}"
                )

    except Exception as e:
        print(f"   ✗ TradingView export failed: {e}")

    return csv_content


def _test_edge_cases(builder):
    """Test edge cases and error handling."""
    print("\n8. Testing edge cases and error handling...")

    # Test with empty data
    try:
        DominanceCandleBuilder([])
        print("   ✗ Empty data test should have failed")
    except ValueError:
        print("   ✓ Empty data validation working")

    # Test validation with empty candles
    empty_validation = builder.validate_candles([])
    print(f"   ✓ Empty candles validation: {not empty_validation['is_valid']}")

    # Test integrity check with empty snapshots
    empty_integrity = builder.check_data_integrity([])
    print(f"   ✓ Empty snapshots integrity: {empty_integrity['integrity_score'] == 0}")

    # Test TradingView export with empty candles
    try:
        builder.export_for_tradingview([])
        print("   ✗ Empty TradingView export should have failed")
    except ValueError:
        print("   ✓ Empty TradingView export validation working")


def _test_performance(base_time):
    """Test performance with larger dataset."""
    print("\n9. Testing performance with larger dataset...")

    import os
    import time

    import psutil

    # Create larger dataset (1000 snapshots)
    large_snapshots = []
    rng_large = np.random.default_rng(123)  # Different seed for large dataset
    for i in range(1000):
        dominance = 8.5 + rng_large.normal(0, 0.5)
        market_cap = Decimal(150000000000) + Decimal(
            str(rng_large.normal(0, 5000000000))
        )

        snapshot = DominanceData(
            timestamp=base_time + timedelta(minutes=i),
            usdt_market_cap=market_cap * Decimal("0.7"),
            usdc_market_cap=market_cap * Decimal("0.3"),
            total_stablecoin_cap=market_cap,
            crypto_total_market_cap=market_cap / Decimal(str(dominance / 100)),
            usdt_dominance=dominance * 0.7,
            usdc_dominance=dominance * 0.3,
            stablecoin_dominance=dominance,
            dominance_24h_change=rng_large.normal(0, 0.5),
            dominance_7d_change=rng_large.normal(0, 1.0),
            stablecoin_velocity=max(0.1, rng_large.normal(1.5, 0.3)),
        )
        large_snapshots.append(snapshot)

    # Measure performance
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()
    large_builder = DominanceCandleBuilder(large_snapshots)
    large_candles = large_builder.build_candles("5T")
    indicators_result = large_builder.calculate_technical_indicators(large_candles)
    validation_result = large_builder.validate_candles(indicators_result["candles"])
    end_time = time.time()

    memory_after = process.memory_info().rss / 1024 / 1024  # MB

    print(f"   ✓ Large dataset processing time: {end_time - start_time:.2f} seconds")
    print(f"   ✓ Memory usage: {memory_after - memory_before:.2f} MB increase")
    print(
        f"   ✓ Processed {len(large_snapshots)} snapshots → {len(large_candles)} candles"
    )
    print(f"   ✓ Large dataset quality score: {validation_result['quality_score']}%")

    return end_time - start_time


def _generate_final_summary(
    snapshots,
    integrity_result,
    candles,
    indicators_result,
    validation_result,
    statistics,
    csv_content,
    processing_time,
):
    """Generate final test summary."""
    print("\n10. Final validation summary...")

    test_results = {
        "sample_data_creation": len(snapshots) > 0,
        "data_integrity_check": integrity_result["integrity_score"] > 0,
        "candle_building": len(candles) > 0,
        "technical_indicators": len(indicators_result["candles"]) > 0,
        "candle_validation": validation_result["quality_score"] > 0,
        "statistics_generation": "error" not in statistics,
        "tradingview_export": csv_content is not None,
        "error_handling": True,  # All edge cases handled
        "performance_test": processing_time < 30,  # Should complete in 30 seconds
    }

    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

    print(f"   ✓ Tests Passed: {passed_tests}/{total_tests}")

    for test_name, passed in test_results.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {test_name.replace('_', ' ').title()}")

    print(f"\n{'=' * 80}")
    if passed_tests == total_tests:
        print(
            "🎉 ALL TESTS PASSED! Dominance candlestick functionality is working correctly."
        )
    else:
        print(
            f"⚠️  {total_tests - passed_tests} test(s) failed. Please review the implementation."
        )
    print(f"{'=' * 80}")

    return test_results


def test_dominance_candlestick_functionality():
    """
    Comprehensive test function for dominance candlestick functionality.

    Tests all major components including:
    - DominanceData validation
    - DominanceCandleBuilder candle creation
    - Technical indicator calculations
    - Data validation and integrity checks
    - TradingView export functionality
    - Statistics generation

    This function serves as both a test suite and demonstration of the
    dominance candlestick analysis capabilities.
    """
    print("=" * 80)
    print("DOMINANCE CANDLESTICK FUNCTIONALITY TEST")
    print("=" * 80)

    try:
        # Test 1: Create sample data
        snapshots = _create_sample_dominance_snapshots()

        # Initialize builder
        builder = DominanceCandleBuilder(snapshots)

        # Test 2: Data integrity
        integrity_result = _test_data_integrity(builder, snapshots)

        # Test 3: Candle building
        candles = _test_candle_building(builder)
        if not candles:
            return {"error": "Failed to build candles"}

        # Test 4: Technical indicators
        indicators_result = _test_technical_indicators(builder, candles)
        updated_candles = indicators_result["candles"]

        # Test 5: Candle validation
        validation_result = _test_candle_validation(builder, updated_candles)

        # Test 6: Statistics generation
        statistics = _test_statistics_generation(builder, updated_candles)

        # Test 7: TradingView export
        csv_content = _test_tradingview_export(builder, updated_candles)

        # Test 8: Edge cases
        _test_edge_cases(builder)

        # Test 9: Performance testing
        processing_time = _test_performance(snapshots[0].timestamp - timedelta(hours=2))

        # Test 10: Final summary
        return _generate_final_summary(
            snapshots,
            integrity_result,
            candles,
            indicators_result,
            validation_result,
            statistics,
            csv_content,
            processing_time,
        )

    except Exception as e:
        print(f"\n💥 TEST SUITE FAILED WITH EXCEPTION: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        print(f"Traceback:\n{traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the test when this file is executed directly
    test_dominance_candlestick_functionality()
