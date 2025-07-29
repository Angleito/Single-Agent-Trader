"""Coinbase-specific market data provider implementation."""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, NoReturn

import aiohttp
import websockets

from bot.config import settings
from bot.data.base_market_provider import AbstractMarketDataProvider
from bot.trading_types import MarketData

logger = logging.getLogger(__name__)


def _extract_secret_value(secret_obj) -> str | None:
    """Safely extract secret value from SecretStr or SecureString objects."""
    if secret_obj is None:
        return None
    if hasattr(secret_obj, 'get_secret_value'):
        # Pydantic SecretStr
        return secret_obj.get_secret_value()
    elif hasattr(secret_obj, 'get_value'):
        # SecureString
        return secret_obj.get_value()
    else:
        # Fallback to string conversion
        return str(secret_obj)


class MarketDataAPIError(Exception):
    """Raised when market data API requests fail."""


# Import Coinbase client handling both legacy and CDP authentication
try:
    from coinbase import jwt_generator
    from coinbase.rest import RESTClient as _BaseClient
    from coinbase.rest.rest_base import HTTPError as _CBApiEx

    CoinbaseAPIError = _CBApiEx

    class CoinbaseAdvancedTrader(_BaseClient):
        """Adapter class exposing legacy method names used in this codebase."""

    COINBASE_AVAILABLE = True

except ImportError:
    # Mock classes for when Coinbase SDK is not installed
    class MockCoinbaseAdvancedTrader:
        def __init__(self, **kwargs):
            pass

    class MockCoinbaseAPIError(Exception):
        pass

    CoinbaseAdvancedTrader = MockCoinbaseAdvancedTrader  # type: ignore[misc,assignment]
    CoinbaseAPIError = MockCoinbaseAPIError  # type: ignore[misc,assignment]

    # Mock jwt_generator module for when SDK is not available
    class MockJwtGenerator:
        @staticmethod
        def build_ws_jwt(_api_key: str, _api_secret: str) -> str | None:
            return None

    jwt_generator = MockJwtGenerator()

    COINBASE_AVAILABLE = False


class WebSocketMessageValidator:
    """
    Validator for WebSocket message schemas to prevent data corruption.

    Validates incoming WebSocket messages for required fields, data types,
    and anomaly detection before processing.
    """

    def __init__(self):
        """Initialize the validator."""
        self.message_count = 0
        self.validation_failures = 0

    def validate_ticker_message(self, message: dict) -> bool:
        """
        Validate ticker message structure.

        Args:
            message: WebSocket ticker message

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required top-level fields
            required_fields = ["channel", "timestamp", "events"]
            if not all(field in message for field in required_fields):
                logger.warning(
                    "Ticker message missing required fields: %s", required_fields
                )
                return False

            # Validate channel
            if message["channel"] != "ticker":
                logger.warning("Invalid ticker channel: %s", message["channel"])
                return False

            # Validate timestamp
            try:
                datetime.fromisoformat(message["timestamp"])
            except (ValueError, AttributeError) as e:
                logger.warning("Invalid ticker timestamp: %s", e)
                return False

            # Validate message timing
            if not self.validate_message_timing(message):
                return False

            # Validate events array
            events = message.get("events", [])
            if not isinstance(events, list) or len(events) == 0:
                logger.warning("Ticker message has no events")
                return False

            # Validate ticker events
            return all(self._validate_ticker_event(event) for event in events)

        except Exception:
            logger.exception("Error validating ticker message")
            return False

    def validate_trade_message(self, message: dict) -> bool:
        """
        Validate trade message structure.

        Args:
            message: WebSocket trade message

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["channel", "events"]
            if not all(field in message for field in required_fields):
                logger.warning(
                    "Trade message missing required fields: %s", required_fields
                )
                return False

            # Validate channel
            if message["channel"] != "market_trades":
                logger.warning("Invalid trade channel: %s", message["channel"])
                return False

            # Validate message timing
            if not self.validate_message_timing(message):
                return False

            # Validate events
            events = message.get("events", [])
            if not isinstance(events, list):
                logger.warning("Trade events is not a list")
                return False

            return all(self._validate_trade_event(event) for event in events)

        except Exception:
            logger.exception("Error validating trade message")
            return False

    def _validate_ticker_event(self, event: dict) -> bool:
        """
        Validate individual ticker event.

        Args:
            event: Ticker event data

        Returns:
            True if valid, False otherwise
        """
        try:
            event_type = event.get("type")
            if event_type not in ["snapshot", "update"]:
                logger.warning("Invalid ticker event type: %s", event_type)
                return False

            tickers = event.get("tickers", [])
            if not isinstance(tickers, list):
                logger.warning("Ticker event has no tickers array")
                return False

            return all(self._validate_ticker_data(ticker) for ticker in tickers)

        except Exception:
            logger.exception("Error validating ticker event")
            return False

    def _validate_ticker_data(self, ticker: dict) -> bool:
        """
        Validate ticker data fields.

        Args:
            ticker: Ticker data

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["product_id", "price"]
            if not all(field in ticker for field in required_fields):
                logger.warning("Ticker missing required fields: %s", required_fields)
                return False

            # Validate price
            try:
                price = Decimal(str(ticker["price"]))
                if price <= 0:
                    logger.warning("Invalid ticker price: %s", price)
                    return False
            except (ValueError, TypeError) as e:
                logger.warning("Invalid ticker price format: %s", e)
                return False

            # Advanced data quality validation
            return self.validate_data_quality(ticker, ticker["product_id"])

        except Exception:
            logger.exception("Error validating ticker data")
            return False

    def _validate_trade_event(self, event: dict) -> bool:
        """
        Validate trade event data.

        Args:
            event: Trade event data

        Returns:
            True if valid, False otherwise
        """
        try:
            event_type = event.get("type")
            if event_type not in ["snapshot", "update"]:
                logger.warning("Invalid trade event type: %s", event_type)
                return False

            trades = event.get("trades", [])
            if not isinstance(trades, list):
                logger.warning("Trade event has no trades array")
                return False

            return all(self._validate_trade_data(trade) for trade in trades)

        except Exception:
            logger.exception("Error validating trade event")
            return False

    def _validate_trade_data(self, trade: dict) -> bool:
        """
        Validate individual trade data.

        Args:
            trade: Trade data

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["product_id", "price", "size", "side"]
            if not all(field in trade for field in required_fields):
                logger.warning("Trade missing required fields: %s", required_fields)
                return False

            # Validate price and size
            try:
                price = Decimal(str(trade["price"]))
                size = Decimal(str(trade["size"]))

                if price <= 0:
                    logger.warning("Invalid trade price: %s", price)
                    return False

                if size <= 0:
                    logger.warning("Invalid trade size: %s", size)
                    return False

            except (ValueError, TypeError) as e:
                logger.warning("Invalid trade data format: %s", e)
                return False

            # Validate side
            if trade["side"] not in ["BUY", "SELL"]:
                logger.warning("Invalid trade side: %s", trade["side"])
                return False

        except Exception:
            logger.exception("Error validating trade data")
            return False
        else:
            return True

    def get_validation_stats(self) -> dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation metrics
        """
        success_rate = (
            ((self.message_count - self.validation_failures) / self.message_count * 100)
            if self.message_count > 0
            else 100.0
        )

        return {
            "total_messages": self.message_count,
            "validation_failures": self.validation_failures,
            "success_rate_pct": success_rate,
        }

    def validate_data_quality(self, price_data: dict, symbol: str) -> bool:
        """
        Advanced data quality validation for price data.

        Args:
            price_data: Price data to validate
            symbol: Trading symbol for context

        Returns:
            True if data quality is acceptable
        """
        try:
            price = Decimal(str(price_data.get("price", 0)))

            # Check for suspicious price values
            if price <= 0:
                logger.warning("Invalid price value: %s", price)
                return False

            # Check for unrealistic prices based on symbol
            if symbol.startswith("BTC"):
                # Bitcoin price should be within reasonable bounds
                if price < Decimal(1000) or price > Decimal(500000):
                    logger.warning("Suspicious BTC price: %s", price)
                    return False
            elif symbol.startswith("ETH") and (
                price < Decimal(10) or price > Decimal(50000)
            ):
                # Ethereum price bounds
                logger.warning("Suspicious ETH price: %s", price)
                return False

            # Check for excessive decimal precision (market manipulation indicator)
            price_str = str(price)
            if "." in price_str:
                decimal_places = len(price_str.split(".")[1])
                if decimal_places > 8:  # More than 8 decimal places is unusual
                    logger.warning(
                        "Excessive precision in price: %s decimals", decimal_places
                    )
                    return False

        except Exception:
            logger.exception("Error in data quality validation")
            return False
        else:
            return True

    def validate_message_timing(self, message: dict) -> bool:
        """
        Validate message timing to detect stale or future data.

        Args:
            message: WebSocket message with timestamp

        Returns:
            True if timing is acceptable
        """
        try:
            timestamp_str = message.get("timestamp")
            if not timestamp_str:
                return True  # No timestamp to validate

            message_time = datetime.fromisoformat(timestamp_str)
            current_time = datetime.now(UTC)

            # Check for messages from the future (more than 5 seconds)
            if message_time > current_time + timedelta(seconds=5):
                logger.warning(
                    "Future timestamp detected: %s vs %s", message_time, current_time
                )
                return False

            # Check for very stale messages (more than 30 seconds old)
            if current_time - message_time > timedelta(seconds=30):
                logger.warning(
                    "Stale message detected: %s vs %s", message_time, current_time
                )
                return False

        except Exception:
            logger.exception("Error validating message timing")
            return False
        else:
            return True


class CoinbaseMarketDataProvider(AbstractMarketDataProvider):
    """
    Coinbase-specific implementation of the market data provider.

    Handles real-time market data ingestion from Coinbase REST API and WebSocket.
    """

    COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
    COINBASE_REST_URL = "https://api.coinbase.com"

    def __init__(self, symbol: str | None = None, interval: str | None = None) -> None:
        """
        Initialize the Coinbase market data provider.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        super().__init__(symbol, interval)

        # For historical data, use spot symbol even if trading futures
        self._data_symbol = self._get_data_symbol(self.symbol)

        # API clients
        self._rest_client: CoinbaseAdvancedTrader | None = None

        # WebSocket message validator
        self._ws_validator = WebSocketMessageValidator()

        if self._data_symbol != self.symbol:
            logger.info(
                "Using %s for historical data (trading: %s)",
                self._data_symbol,
                self.symbol,
            )

    def _get_data_symbol(self, symbol: str) -> str:
        """
        Get the appropriate symbol for fetching historical data.

        For futures contracts, we need to use the underlying spot symbol
        since Coinbase provides historical data for spot pairs, not futures.

        Args:
            symbol: Trading symbol (could be spot or futures)

        Returns:
            Symbol to use for data fetching
        """
        # Import here to avoid circular imports
        from bot.exchange.futures_contract_mapper import FuturesContractMapper

        # Check if this looks like a futures contract (contains month/expiry info)
        if "-" in symbol and len(symbol.split("-")) >= 3:
            # This is likely a futures contract, convert to spot
            spot_symbol = FuturesContractMapper.futures_to_spot_symbol(symbol)
            logger.debug(
                "Converted futures symbol %s to spot symbol %s for data fetching",
                symbol,
                spot_symbol,
            )
            return spot_symbol

        # This is already a spot symbol
        return symbol

    def _build_websocket_jwt(self) -> str | None:
        """
        Build JWT token for WebSocket authentication using SDK's built-in jwt_generator.

        Returns:
            JWT token string or None if credentials not available
        """
        try:
            # Check if CDP credentials are available
            cdp_api_key_obj = getattr(settings.exchange, "cdp_api_key_name", None)
            cdp_private_key_obj = getattr(settings.exchange, "cdp_private_key", None)

            if not cdp_api_key_obj or not cdp_private_key_obj:
                logger.debug(
                    "CDP credentials not available for WebSocket authentication"
                )
                return None

            # Extract actual values from SecretStr objects
            cdp_api_key = (
                _extract_secret_value(cdp_api_key_obj)
                if hasattr(cdp_api_key_obj, "get_secret_value")
                else str(cdp_api_key_obj)
            )
            cdp_private_key = (
                _extract_secret_value(cdp_private_key_obj)
                if hasattr(cdp_private_key_obj, "get_secret_value")
                else str(cdp_private_key_obj)
            )

            # Use SDK's built-in JWT generator for WebSocket authentication
            logger.debug("Generating JWT for WebSocket authentication")

            try:
                jwt_token = jwt_generator.build_ws_jwt(cdp_api_key, cdp_private_key)

                if jwt_token:
                    logger.info("Successfully generated WebSocket JWT token using SDK")
                    return jwt_token
                logger.warning(
                    "SDK jwt_generator returned None - this should not happen with valid credentials"
                )

            except Exception:
                logger.exception("Exception in jwt_generator.build_ws_jwt")
                return None
            else:
                return None

        except Exception as e:
            logger.warning(
                "Failed to generate WebSocket JWT token (outer exception): %s", e
            )
            return None

    async def connect(self, fetch_historical: bool = True) -> None:
        """
        Establish connection to Coinbase data feeds.

        Args:
            fetch_historical: Whether to fetch historical data during connection

        This will connect to both REST API for historical data and WebSocket for real-time data.
        """
        try:
            # Initialize REST client and session
            await self._initialize_clients()

            # Start WebSocket connection for real-time updates (do this early)
            await self._start_websocket()

            # Fetch initial historical data (optional)
            if fetch_historical:
                try:
                    await self.fetch_historical_data()
                except Exception as e:
                    logger.warning(
                        "Failed to fetch historical data, continuing with WebSocket only: %s",
                        e,
                    )

            # Try to get current price to ensure connectivity
            try:
                current_price = await self.fetch_latest_price()
                if current_price:
                    logger.info(
                        "Successfully fetched current price: $%s", current_price
                    )
            except Exception as e:
                logger.warning("Could not fetch current price: %s", e)

            self._is_connected = True
            logger.info("Successfully connected to market data feeds")

        except Exception:
            logger.exception("Failed to connect to market data feeds")
            self._is_connected = False
            raise

    async def _initialize_clients(self) -> None:
        """Initialize REST client and HTTP session."""
        # Initialize HTTP session
        await self._initialize_http_session()

        # Initialize Coinbase REST client if credentials are available
        # Check for legacy credentials
        has_legacy_credentials = (
            settings.exchange.cb_api_key is not None
            and settings.exchange.cb_api_secret is not None
            and settings.exchange.cb_passphrase is not None
        )

        # Check for CDP credentials
        has_cdp_credentials = (
            settings.exchange.cdp_api_key_name is not None
            and settings.exchange.cdp_private_key is not None
        )

        if (
            has_legacy_credentials
            and settings.exchange.cb_api_key
            and settings.exchange.cb_api_secret
            and settings.exchange.cb_passphrase
        ):
            self._rest_client = CoinbaseAdvancedTrader(
                api_key=_extract_secret_value(settings.exchange.cb_api_key),
                api_secret=_extract_secret_value(settings.exchange.cb_api_secret),
                passphrase=_extract_secret_value(settings.exchange.cb_passphrase),
                sandbox=settings.exchange.cb_sandbox,
            )
            logger.info("Initialized REST client with legacy Coinbase credentials")
        elif (
            has_cdp_credentials
            and settings.exchange.cdp_api_key_name
            and settings.exchange.cdp_private_key
        ):
            self._rest_client = CoinbaseAdvancedTrader(
                api_key=_extract_secret_value(settings.exchange.cdp_api_key_name),
                api_secret=_extract_secret_value(settings.exchange.cdp_private_key),
            )
            logger.info("Initialized REST client with CDP Coinbase credentials")
        else:
            logger.warning(
                "No Coinbase credentials provided, using public endpoints only"
            )

    async def fetch_historical_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        granularity: str | None = None,
    ) -> list[MarketData]:
        """
        Fetch historical OHLCV data from Coinbase REST API.

        Args:
            start_time: Start time for data (default: current time - candle_limit * interval)
            end_time: End time for data (default: current time)
            granularity: Candle granularity (default: self.interval)

        Returns:
            List of MarketData objects
        """
        granularity = granularity or self.interval
        end_time = end_time or datetime.now(UTC)

        # Calculate start time based on interval and limit
        interval_seconds = self._interval_to_seconds(granularity)
        # For 8 hours of 1-minute data, we need 480 candles
        # Coinbase API limits us to 300 candles per request
        # We'll need to make 2 requests to get the full 8 hours
        required_candles = 480 if granularity == "1m" else max(self.candle_limit, 200)

        # If we need more than 300 candles, we'll fetch in batches
        if required_candles > 300:
            return await self._fetch_historical_data_batched(
                start_time, end_time, granularity, required_candles
            )

        max_candles = min(required_candles, 300)  # Use 300 to be safe with API limits

        # Calculate the actual number of candles that will be requested
        # This depends on the API granularity, not the requested granularity
        api_granularity = self._format_granularity(granularity)
        api_interval_seconds = self._get_api_interval_seconds(api_granularity)

        # Adjust max_candles based on actual API interval to stay under 350 limit
        if interval_seconds != api_interval_seconds:
            # Calculate how many API candles would be in our time range
            time_range_seconds = interval_seconds * max_candles
            api_candles_needed = int(time_range_seconds / api_interval_seconds)

            if api_candles_needed > 300:
                # Reduce time range to fit within API limits
                safe_time_range = 300 * api_interval_seconds
                max_candles = int(safe_time_range / interval_seconds)
                logger.warning(
                    "Adjusted candle limit from %s to %s due to API granularity mismatch",
                    self.candle_limit,
                    max_candles,
                )

        default_start = end_time - timedelta(seconds=interval_seconds * max_candles)
        start_time = start_time or default_start

        # Log the actual number of candles that will be requested
        time_diff = (end_time - start_time).total_seconds()
        expected_candles = int(time_diff / api_interval_seconds)

        # Final safety check - ensure we never exceed 350 candles
        if expected_candles > 350:
            logger.error(
                "Calculated %s candles which exceeds API limit of 350!",
                expected_candles,
            )
            # Adjust start_time to ensure we stay under limit
            safe_seconds = 300 * api_interval_seconds  # Use 300 to be extra safe
            start_time = end_time - timedelta(seconds=safe_seconds)
            time_diff = safe_seconds
            expected_candles = 300
            logger.warning(
                "Adjusted start_time to %s to stay within API limits", start_time
            )

        logger.info(
            "Fetching historical data for %s from %s to %s",
            self._data_symbol,
            start_time,
            end_time,
        )
        logger.info(
            "Requesting %s candles at %s granularity (requested: %s)",
            expected_candles,
            api_granularity,
            granularity,
        )

        # Warn if we're not getting enough data for indicators
        min_required_for_indicators = 100  # VuManChu needs ~80 + buffer
        if expected_candles < min_required_for_indicators:
            logger.warning(
                "Fetching only %s candles, but indicators need at least %s for reliable calculations. Consider increasing candle_limit.",
                expected_candles,
                min_required_for_indicators,
            )

        def _raise_session_error() -> NoReturn:
            raise RuntimeError("HTTP session not initialized")

        def _raise_api_error(status: int, text: str) -> NoReturn:
            raise MarketDataAPIError(f"API request failed with status {status}: {text}")

        try:
            # Use public API endpoint for historical candles with data symbol
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self._data_symbol}/candles"

            params: dict[str, int | str] = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "granularity": self._format_granularity(granularity),
            }

            if self._session is None:
                _raise_session_error()
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    _raise_api_error(response.status, await response.text())

                data = await response.json()
                candles = data.get("candles", [])

                # Convert to MarketData objects
                historical_data = []
                for candle in candles:
                    try:
                        market_data = self._parse_candle_data(candle)
                        if self._validate_market_data(market_data):
                            historical_data.append(market_data)
                    except Exception as e:
                        logger.warning("Failed to parse candle data: %s", e)
                        continue

                # Sort by timestamp
                historical_data.sort(key=lambda x: x.timestamp)

                # Update cache
                self._ohlcv_cache = historical_data[-self.candle_limit :]
                self._last_update = datetime.now(UTC)
                self._cache_timestamps["ohlcv"] = self._last_update

                logger.info("Loaded %s historical candles", len(self._ohlcv_cache))

                # Validate we have sufficient data for indicators
                self._validate_data_sufficiency(len(self._ohlcv_cache))

                return historical_data

        except Exception:
            logger.exception("Failed to fetch historical data")
            # Fallback to cached data if available
            if self._ohlcv_cache:
                logger.info("Using cached historical data")
                return self._ohlcv_cache
            raise

    async def _fetch_historical_data_batched(
        self,
        start_time: datetime | None,
        end_time: datetime,
        granularity: str,
        required_candles: int,
    ) -> list[MarketData]:
        """
        Fetch historical data in multiple batches to get around API limits.

        Args:
            start_time: Start time for data
            end_time: End time for data
            granularity: Candle granularity
            required_candles: Total number of candles needed

        Returns:
            List of MarketData objects covering the full time range
        """
        interval_seconds = self._interval_to_seconds(granularity)
        batch_size = 300  # Safe batch size for API
        all_data = []

        logger.info(
            "Fetching %s candles in batches of %s", required_candles, batch_size
        )

        # Calculate the actual start time if not provided
        if start_time is None:
            start_time = end_time - timedelta(
                seconds=interval_seconds * required_candles
            )

        # Work backwards from end_time in batches
        current_end = end_time
        batches_needed = (required_candles + batch_size - 1) // batch_size

        for batch_num in range(batches_needed):
            # Calculate start time for this batch
            candles_in_batch = min(
                batch_size, required_candles - (batch_num * batch_size)
            )
            batch_start = current_end - timedelta(
                seconds=interval_seconds * candles_in_batch
            )

            # Ensure we don't go before the requested start time
            batch_start = max(batch_start, start_time)

            logger.info(
                "Fetching batch %s/%s: %s to %s",
                batch_num + 1,
                batches_needed,
                batch_start,
                current_end,
            )

            try:
                # Fetch this batch
                batch_data = await self._fetch_single_batch(
                    batch_start, current_end, granularity
                )
                if batch_data:
                    all_data.extend(batch_data)
                    logger.info(
                        "Batch %s completed: %s candles", batch_num + 1, len(batch_data)
                    )
                else:
                    logger.warning("Batch %s returned no data", batch_num + 1)

                # Add small delay between requests to be respectful to API
                await asyncio.sleep(0.1)

            except Exception:
                logger.exception("Error fetching batch %s", batch_num + 1)
                # Continue with other batches even if one fails

            # Move to next batch
            current_end = batch_start

            # If we've reached the start time, stop
            if current_end <= start_time:
                break

        # Sort all data by timestamp and remove duplicates
        all_data.sort(key=lambda x: x.timestamp)
        unique_data = []
        seen_timestamps = set()

        for data_point in all_data:
            if data_point.timestamp not in seen_timestamps:
                unique_data.append(data_point)
                seen_timestamps.add(data_point.timestamp)

        # Update cache with the combined data
        self._ohlcv_cache = unique_data[-self.candle_limit :]
        self._last_update = datetime.now(UTC)
        self._cache_timestamps["ohlcv"] = self._last_update

        logger.info(
            "Successfully fetched %s candles across %s batches",
            len(unique_data),
            batches_needed,
        )
        self._validate_data_sufficiency(len(unique_data))

        return unique_data

    async def _fetch_single_batch(
        self, start_time: datetime, end_time: datetime, granularity: str
    ) -> list[MarketData]:
        """Fetch a single batch of historical data."""

        def _raise_session_error() -> NoReturn:
            raise RuntimeError("HTTP session not initialized")

        def _raise_api_error(status: int, text: str) -> NoReturn:
            raise MarketDataAPIError(f"API request failed with status {status}: {text}")

        try:
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self._data_symbol}/candles"

            params: dict[str, int | str] = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "granularity": self._format_granularity(granularity),
            }

            if self._session is None:
                _raise_session_error()
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    _raise_api_error(response.status, await response.text())

                data = await response.json()
                candles = data.get("candles", [])

                # Convert to MarketData objects
                batch_data = []
                for candle in candles:
                    try:
                        market_data = self._parse_candle_data(candle)
                        if self._validate_market_data(market_data):
                            batch_data.append(market_data)
                    except Exception as e:
                        logger.warning("Failed to parse candle data: %s", e)
                        continue

                return batch_data

        except Exception:
            logger.exception(
                "Failed to fetch batch from %s to %s", start_time, end_time
            )
            return []

    async def fetch_latest_price(self) -> Decimal | None:
        """
        Fetch the latest price for the symbol from Coinbase REST API.

        Returns:
            Latest price as Decimal or None if unavailable
        """

        def _raise_session_error() -> NoReturn:
            raise RuntimeError("HTTP session not initialized")

        # Check cache first
        if self._is_cache_valid("price"):
            return self._price_cache.get("price")

        try:
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self._data_symbol}"

            if self._session is None:
                _raise_session_error()
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.warning("Failed to fetch latest price: %s", response.status)
                    return self.get_latest_price()  # Fall back to cached data

                data = await response.json()
                price_str = data.get("price")

                if price_str:
                    price = Decimal(price_str)
                    # Update cache
                    self._price_cache["price"] = price
                    self._cache_timestamps["price"] = datetime.now(UTC)
                    return price

        except Exception:
            logger.exception("Error fetching latest price")

        # Fall back to cached OHLCV data
        return self.get_latest_price()

    async def fetch_orderbook(
        self, level: int = 2
    ) -> dict[str, list[list[float]]] | None:
        """
        Fetch order book data from Coinbase REST API.

        Args:
            level: Order book level (1, 2, or 3)

        Returns:
            Order book data or None if unavailable
        """
        # Check cache first
        cache_key = f"orderbook_l{level}"
        if self._is_cache_valid(cache_key):
            return self._orderbook_cache.get(cache_key)

        def _raise_session_error() -> NoReturn:
            raise RuntimeError("HTTP session not initialized")

        try:
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self.symbol}/book"
            params: dict[str, int] = {
                "limit": min(level * 10, 50)
            }  # Reasonable limit based on level

            if self._session is None:
                _raise_session_error()
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning("Failed to fetch orderbook: %s", response.status)
                    return None

                data = await response.json()
                orderbook = {
                    "bids": [
                        (Decimal(bid["price"]), Decimal(bid["size"]))
                        for bid in data.get("bids", [])
                    ],
                    "asks": [
                        (Decimal(ask["price"]), Decimal(ask["size"]))
                        for ask in data.get("asks", [])
                    ],
                    "timestamp": datetime.now(UTC),
                }

                # Update cache
                self._orderbook_cache[cache_key] = orderbook
                self._cache_timestamps[cache_key] = datetime.now(UTC)

                return orderbook

        except Exception:
            logger.exception("Error fetching orderbook")
            return None

    async def _connect_websocket(self) -> None:
        """
        Establish WebSocket connection and handle messages with CDP authentication.
        """
        # Build JWT token for authentication
        jwt_token = self._build_websocket_jwt()

        # Prepare subscription with authentication if available
        # Note: Use the correct WebSocket subscription format per Coinbase documentation
        # Use individual channel subscriptions rather than channels array to avoid auth issues
        subscriptions = [
            {"type": "subscribe", "product_ids": [self.symbol], "channel": "ticker"},
            {
                "type": "subscribe",
                "product_ids": [self.symbol],
                "channel": "market_trades",
            },
            {
                "type": "subscribe",
                "channel": "heartbeats",  # No product_ids needed for heartbeats
            },
        ]

        # Add JWT authentication if available
        if jwt_token:
            for sub in subscriptions:
                if isinstance(sub, dict):
                    sub["jwt"] = jwt_token
            logger.info("Using CDP JWT authentication for WebSocket")
        else:
            logger.info("Using public WebSocket connection (no authentication)")

        logger.debug("WebSocket subscriptions: %s", json.dumps(subscriptions, indent=2))

        async with websockets.connect(
            self.COINBASE_WS_URL, open_timeout=settings.exchange.websocket_timeout
        ) as websocket:
            self._ws_connection = websocket  # type: ignore[assignment]

            # Send all subscriptions
            for i, subscription in enumerate(subscriptions):
                await websocket.send(json.dumps(subscription))
                channel_name = (
                    subscription.get("channel", "unknown")
                    if isinstance(subscription, dict)
                    else "unknown"
                )
                logger.debug(
                    "Sent subscription %s/%s: %s",
                    i + 1,
                    len(subscriptions),
                    channel_name,
                )
                # Small delay between subscriptions to avoid overwhelming the server
                await asyncio.sleep(0.1)

            logger.info(
                "Subscribed to %s WebSocket feeds for %s",
                len(subscriptions),
                self.symbol,
            )

            # Reset reconnection counter on successful connection
            self._reconnect_attempts = 0

            # Handle incoming messages - non-blocking queue approach
            message_count = 0
            async for message in websocket:
                try:
                    message_count += 1
                    parsed_message = json.loads(message)

                    # Log first few messages for debugging
                    if message_count <= 5:
                        logger.debug(
                            "WebSocket message #%s: %s - %s",
                            message_count,
                            parsed_message.get("channel", "unknown"),
                            parsed_message.get("type", "unknown"),
                        )

                    # Add to queue for non-blocking processing
                    try:
                        self._message_queue.put_nowait(parsed_message)
                    except asyncio.QueueFull:
                        logger.warning("Message queue full, dropping oldest message")
                        # Drop oldest message and add new one
                        try:
                            self._message_queue.get_nowait()
                            self._message_queue.put_nowait(parsed_message)
                        except asyncio.QueueEmpty:
                            pass

                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse WebSocket message: %s", e)
                except Exception:
                    logger.exception("Error handling WebSocket message")

    async def _handle_websocket_message(
        self, message: dict[str, str | list | dict]
    ) -> None:
        """
        Handle incoming WebSocket messages with validation.

        Args:
            message: Parsed WebSocket message
        """
        # Increment message counter for validation stats
        self._ws_validator.message_count += 1

        # Handle different message formats from Coinbase Advanced Trading API
        channel = message.get("channel")
        msg_type = message.get("type")  # Note: not all messages have 'type'

        if channel == "subscriptions":
            # Subscription confirmation message
            logger.info(
                "WebSocket subscriptions confirmed: %s", message.get("events", [])
            )
        elif channel == "heartbeats":
            # Heartbeat messages - just log occasionally to confirm connection
            events = message.get("events", [])
            if events:
                counter = events[0].get("heartbeat_counter", "unknown")
                logger.debug("Heartbeat #%s received", counter)
        elif channel == "ticker":
            # Validate ticker message before processing
            if self._ws_validator.validate_ticker_message(message):
                await self._handle_ticker_update(message)
            else:
                self._ws_validator.validation_failures += 1
                logger.warning("Rejected invalid ticker message")
        elif channel == "market_trades":
            # Validate trade message before processing
            if self._ws_validator.validate_trade_message(message):
                await self._handle_trade_update(message)
            else:
                self._ws_validator.validation_failures += 1
                logger.warning("Rejected invalid trade message")
        elif msg_type == "error":
            logger.error("WebSocket error: %s", message.get("message", "Unknown error"))
            # Log the full error message for debugging
            logger.debug(
                "Full WebSocket error message: %s", json.dumps(message, indent=2)
            )
        else:
            # Log any unhandled message types for debugging
            logger.debug(
                "Unhandled WebSocket message - Channel: %s, Type: %s", channel, msg_type
            )
            logger.debug("Message: %s", json.dumps(message, indent=2))

    async def _handle_ticker_update(
        self, message: dict[str, str | list | dict]
    ) -> None:
        """
        Handle ticker price updates.

        Args:
            message: Ticker message from WebSocket (Advanced Trading API format)
        """
        try:
            # Advanced Trading API format: events array with tickers
            events = message.get("events", [])
            timestamp = datetime.fromisoformat(message.get("timestamp", ""))

            for event in events:
                event_type = event.get("type")
                if event_type in ["snapshot", "update"]:
                    tickers = event.get("tickers", [])
                    for ticker in tickers:
                        if ticker.get("product_id") != self.symbol:
                            continue

                        price = Decimal(ticker["price"])

                        # Update price cache
                        self._price_cache["price"] = price
                        self._price_cache["timestamp"] = timestamp
                        self._cache_timestamps["price"] = datetime.now(UTC)

                        # Mark that we've received real WebSocket data
                        if not self._websocket_data_received:
                            self._websocket_data_received = True
                            self._first_websocket_data_time = datetime.now(UTC)
                            logger.info(
                                "First WebSocket market data received for %s at $%s",
                                self.symbol,
                                price,
                            )

                        # Update last update time to keep connection status fresh
                        self._last_update = datetime.now(UTC)

                        logger.debug("Ticker update: %s = $%s", self.symbol, price)

                        # Create market data for current candle update
                        if self._ohlcv_cache:
                            # Update the last candle's close price
                            last_candle = self._ohlcv_cache[-1]
                            # Ensure both timestamps have timezone info for comparison
                            last_timestamp = last_candle.timestamp
                            if last_timestamp.tzinfo is None:
                                last_timestamp = last_timestamp.replace(
                                    tzinfo=timestamp.tzinfo
                                )
                            if (
                                timestamp - last_timestamp
                            ).total_seconds() < self._interval_to_seconds(
                                self.interval
                            ):
                                # Update existing candle
                                updated_candle = MarketData(
                                    symbol=last_candle.symbol,
                                    timestamp=last_candle.timestamp,
                                    open=last_candle.open,
                                    high=max(last_candle.high, price),
                                    low=min(last_candle.low, price),
                                    close=price,
                                    volume=last_candle.volume,  # Volume updated separately via trades
                                )
                                self._ohlcv_cache[-1] = updated_candle

                                # Notify subscribers
                                await self._notify_subscribers(updated_candle)

        except Exception:
            logger.exception("Error handling ticker update: %s")
            logger.debug("Ticker message: %s", json.dumps(message, indent=2))

    async def _handle_trade_update(self, message: dict[str, str | list | dict]) -> None:
        """
        Handle trade/market_trades updates.

        Args:
            message: Market trades message from WebSocket (Advanced Trading API format)
        """
        try:
            # Advanced Trading API format: events array with trades
            events = message.get("events", [])
            timestamp = datetime.fromisoformat(message.get("timestamp", ""))

            for event in events:
                event_type = event.get("type")
                if event_type in ["snapshot", "update"]:
                    trades = event.get("trades", [])
                    for trade in trades:
                        if trade.get("product_id") != self.symbol:
                            continue

                        trade_data = {
                            "price": Decimal(trade["price"]),
                            "size": Decimal(trade["size"]),
                            "side": trade["side"],
                            "timestamp": datetime.fromisoformat(
                                trade.get("time", timestamp.isoformat())
                            ),
                            "trade_id": trade.get("trade_id", ""),
                        }

                        logger.debug(
                            "Trade update: %s %s %s @ $%s",
                            self.symbol,
                            trade_data["side"],
                            trade_data["size"],
                            trade_data["price"],
                        )

                        # Mark that we've received real WebSocket data
                        if not self._websocket_data_received:
                            self._websocket_data_received = True
                            self._first_websocket_data_time = datetime.now(UTC)
                            logger.info(
                                "First WebSocket market data received for %s via trade at $%s",
                                self.symbol,
                                trade_data["price"],
                            )

                        # Update last update time to keep connection status fresh
                        self._last_update = datetime.now(UTC)

                        # Add to tick cache (limited size)
                        self._tick_cache.append(trade_data)
                        if len(self._tick_cache) > 1000:  # Keep last 1000 trades
                            self._tick_cache = self._tick_cache[-1000:]

                        # Update volume in current candle if exists
                        if self._ohlcv_cache:
                            last_candle = self._ohlcv_cache[-1]
                            interval_seconds = self._interval_to_seconds(self.interval)
                            # Ensure both timestamps have timezone info for comparison
                            last_timestamp = last_candle.timestamp
                            if last_timestamp.tzinfo is None:
                                last_timestamp = last_timestamp.replace(
                                    tzinfo=trade_data["timestamp"].tzinfo
                                )
                            if (
                                trade_data["timestamp"] - last_timestamp
                            ).total_seconds() < interval_seconds:
                                # Add trade volume to current candle
                                updated_candle = MarketData(
                                    symbol=last_candle.symbol,
                                    timestamp=last_candle.timestamp,
                                    open=last_candle.open,
                                    high=last_candle.high,
                                    low=last_candle.low,
                                    close=last_candle.close,
                                    volume=last_candle.volume + trade_data["size"],
                                )
                                self._ohlcv_cache[-1] = updated_candle

        except Exception:
            logger.exception("Error handling trade update: %s")
            logger.debug("Trade message: %s", json.dumps(message, indent=2))

    def _format_granularity(self, interval: str) -> str:
        """
        Format interval for Coinbase API granularity parameter.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h')

        Returns:
            Granularity string for API
        """
        # Coinbase uses specific granularity values
        granularity_map = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR",
            "2h": "TWO_HOUR",
            "6h": "SIX_HOUR",
            "1d": "ONE_DAY",
        }

        # For unsupported intervals like 3m, find the next higher supported interval
        if interval not in granularity_map:
            # Parse the interval
            interval_seconds = self._interval_to_seconds(interval)

            # Find the closest supported interval that's greater than or equal
            supported_intervals = [
                ("1m", 60),
                ("5m", 300),
                ("15m", 900),
                ("30m", 1800),
                ("1h", 3600),
                ("2h", 7200),
                ("6h", 21600),
                ("1d", 86400),
            ]

            for supported_interval, seconds in supported_intervals:
                if seconds >= interval_seconds:
                    logger.warning(
                        "Interval %s not supported by Coinbase API, using %s instead",
                        interval,
                        supported_interval,
                    )
                    return granularity_map[supported_interval]

            # Default to ONE_DAY if interval is larger than all supported
            logger.warning(
                "Interval %s not supported by Coinbase API, defaulting to ONE_DAY",
                interval,
            )
            return "ONE_DAY"

        return granularity_map[interval]

    def _get_api_interval_seconds(self, api_granularity: str) -> int:
        """
        Get the interval in seconds for a given API granularity.

        Args:
            api_granularity: API granularity string (e.g., 'ONE_MINUTE', 'FIVE_MINUTE')

        Returns:
            Interval in seconds
        """
        api_intervals = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }

        return api_intervals.get(api_granularity, 60)

    def _parse_candle_data(self, candle_data: dict[str, Any]) -> MarketData:
        """
        Parse candle data from Coinbase API response.

        Args:
            candle_data: Raw candle data from API

        Returns:
            MarketData object
        """
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.fromtimestamp(int(candle_data["start"]), tz=UTC),
            open=Decimal(str(candle_data["open"])),
            high=Decimal(str(candle_data["high"])),
            low=Decimal(str(candle_data["low"])),
            close=Decimal(str(candle_data["close"])),
            volume=Decimal(str(candle_data["volume"])),
        )