"""Market data ingestion and real-time data handling."""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import aiohttp
import pandas as pd
import websockets

# Import Coinbase client handling both legacy and CDP authentication
try:
    from coinbase import jwt_generator
    from coinbase.rest import RESTClient as _BaseClient
    from coinbase.rest.rest_base import HTTPError as _CBApiEx

    CoinbaseAPIException = _CBApiEx

    class CoinbaseAdvancedTrader(_BaseClient):
        """Adapter class exposing legacy method names used in this codebase."""

        pass

    COINBASE_AVAILABLE = True

except ImportError:
    # Mock classes for when Coinbase SDK is not installed
    class CoinbaseAdvancedTrader:
        def __init__(self, **kwargs):
            pass

    class CoinbaseAPIException(Exception):
        pass

    # Mock jwt_generator module for when SDK is not available
    class MockJwtGenerator:
        @staticmethod
        def build_ws_jwt(api_key, api_secret):
            return None

    jwt_generator = MockJwtGenerator()

    COINBASE_AVAILABLE = False

from ..config import settings
from ..types import MarketData

logger = logging.getLogger(__name__)


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
                    f"Ticker message missing required fields: {required_fields}"
                )
                return False

            # Validate channel
            if message["channel"] != "ticker":
                logger.warning(f"Invalid ticker channel: {message['channel']}")
                return False

            # Validate timestamp
            try:
                datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid ticker timestamp: {e}")
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
            for event in events:
                if not self._validate_ticker_event(event):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating ticker message: {e}")
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
                    f"Trade message missing required fields: {required_fields}"
                )
                return False

            # Validate channel
            if message["channel"] != "market_trades":
                logger.warning(f"Invalid trade channel: {message['channel']}")
                return False

            # Validate message timing
            if not self.validate_message_timing(message):
                return False

            # Validate events
            events = message.get("events", [])
            if not isinstance(events, list):
                logger.warning("Trade events is not a list")
                return False

            for event in events:
                if not self._validate_trade_event(event):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade message: {e}")
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
                logger.warning(f"Invalid ticker event type: {event_type}")
                return False

            tickers = event.get("tickers", [])
            if not isinstance(tickers, list):
                logger.warning("Ticker event has no tickers array")
                return False

            for ticker in tickers:
                if not self._validate_ticker_data(ticker):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating ticker event: {e}")
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
                logger.warning(f"Ticker missing required fields: {required_fields}")
                return False

            # Validate price
            try:
                price = Decimal(str(ticker["price"]))
                if price <= 0:
                    logger.warning(f"Invalid ticker price: {price}")
                    return False
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid ticker price format: {e}")
                return False

            # Advanced data quality validation
            if not self.validate_data_quality(ticker, ticker["product_id"]):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating ticker data: {e}")
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
                logger.warning(f"Invalid trade event type: {event_type}")
                return False

            trades = event.get("trades", [])
            if not isinstance(trades, list):
                logger.warning("Trade event has no trades array")
                return False

            for trade in trades:
                if not self._validate_trade_data(trade):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade event: {e}")
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
                logger.warning(f"Trade missing required fields: {required_fields}")
                return False

            # Validate price and size
            try:
                price = Decimal(str(trade["price"]))
                size = Decimal(str(trade["size"]))

                if price <= 0:
                    logger.warning(f"Invalid trade price: {price}")
                    return False

                if size <= 0:
                    logger.warning(f"Invalid trade size: {size}")
                    return False

            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid trade data format: {e}")
                return False

            # Validate side
            if trade["side"] not in ["BUY", "SELL"]:
                logger.warning(f"Invalid trade side: {trade['side']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade data: {e}")
            return False

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
                logger.warning(f"Invalid price value: {price}")
                return False

            # Check for unrealistic prices based on symbol
            if symbol.startswith("BTC"):
                # Bitcoin price should be within reasonable bounds
                if price < Decimal("1000") or price > Decimal("500000"):
                    logger.warning(f"Suspicious BTC price: {price}")
                    return False
            elif symbol.startswith("ETH"):
                # Ethereum price bounds
                if price < Decimal("10") or price > Decimal("50000"):
                    logger.warning(f"Suspicious ETH price: {price}")
                    return False

            # Check for excessive decimal precision (market manipulation indicator)
            price_str = str(price)
            if "." in price_str:
                decimal_places = len(price_str.split(".")[1])
                if decimal_places > 8:  # More than 8 decimal places is unusual
                    logger.warning(
                        f"Excessive precision in price: {decimal_places} decimals"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error in data quality validation: {e}")
            return False

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

            message_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            current_time = datetime.now(UTC)

            # Check for messages from the future (more than 5 seconds)
            if message_time > current_time + timedelta(seconds=5):
                logger.warning(
                    f"Future timestamp detected: {message_time} vs {current_time}"
                )
                return False

            # Check for very stale messages (more than 30 seconds old)
            if current_time - message_time > timedelta(seconds=30):
                logger.warning(
                    f"Stale message detected: {message_time} vs {current_time}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating message timing: {e}")
            return False


class MarketDataProvider:
    """
    Handles real-time market data ingestion from Coinbase REST API and WebSocket.

    Provides OHLCV data, tick data, and maintains a rolling cache for indicators.
    Features:
    - REST API integration for historical and current data
    - WebSocket real-time price updates with auto-reconnect
    - TTL-based data caching
    - Data validation and quality checks
    - Subscriber pattern for real-time updates
    """

    COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
    COINBASE_REST_URL = "https://api.coinbase.com"

    def __init__(self, symbol: str | None = None, interval: str | None = None) -> None:
        """
        Initialize the market data provider.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        self.symbol = symbol or settings.trading.symbol
        self.interval = interval or settings.trading.interval
        self.candle_limit = settings.data.candle_limit

        # For historical data, use spot symbol even if trading futures
        self._data_symbol = self._get_data_symbol(self.symbol)

        # Data cache with TTL
        self._ohlcv_cache: list[MarketData] = []
        self._price_cache: dict[str, Any] = {}
        self._orderbook_cache: dict[str, Any] = {}
        self._tick_cache: list[dict[str, Any]] = []
        self._cache_timestamps: dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=settings.data.data_cache_ttl_seconds)
        self._last_update: datetime | None = None

        # API clients
        self._rest_client: CoinbaseAdvancedTrader | None = None
        self._session: aiohttp.ClientSession | None = None

        # WebSocket connection management
        self._ws_connection = None
        self._ws_task: asyncio.Task | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = settings.exchange.websocket_reconnect_attempts
        self._is_connected = False
        self._connection_lock = asyncio.Lock()

        # Non-blocking message processing
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._processing_task: asyncio.Task | None = None
        self._running = False

        # Subscribers for real-time updates
        self._subscribers: list[Callable] = []

        # Data validation settings
        self._max_price_deviation = 0.1  # 10% max price deviation
        self._min_volume = Decimal("0")

        # WebSocket message validator
        self._ws_validator = WebSocketMessageValidator()

        # Track WebSocket data reception
        self._websocket_data_received = False
        self._first_websocket_data_time: datetime | None = None

        logger.info(
            f"Initialized MarketDataProvider for {self.symbol} at {self.interval}"
        )
        if self._data_symbol != self.symbol:
            logger.info(
                f"Using {self._data_symbol} for historical data (trading: {self.symbol})"
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
        from ..exchange.futures_contract_mapper import FuturesContractMapper

        # Check if this looks like a futures contract (contains month/expiry info)
        if "-" in symbol and len(symbol.split("-")) >= 3:
            # This is likely a futures contract, convert to spot
            spot_symbol = FuturesContractMapper.futures_to_spot_symbol(symbol)
            logger.debug(
                f"Converted futures symbol {symbol} to spot symbol {spot_symbol} for data fetching"
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
                cdp_api_key_obj.get_secret_value()
                if hasattr(cdp_api_key_obj, "get_secret_value")
                else str(cdp_api_key_obj)
            )
            cdp_private_key = (
                cdp_private_key_obj.get_secret_value()
                if hasattr(cdp_private_key_obj, "get_secret_value")
                else str(cdp_private_key_obj)
            )

            # Use SDK's built-in JWT generator for WebSocket authentication
            logger.debug(
                f"Generating JWT with API key: {cdp_api_key[:50]}... (length: {len(cdp_api_key)})"
            )
            logger.debug(
                f"Private key length: {len(cdp_private_key)}, starts with: {cdp_private_key[:20]}..."
            )

            try:
                jwt_token = jwt_generator.build_ws_jwt(cdp_api_key, cdp_private_key)

                if jwt_token:
                    logger.info(
                        f"Successfully generated WebSocket JWT token using SDK (length: {len(jwt_token)})"
                    )
                    logger.debug(f"JWT preview: {jwt_token[:50]}...")
                    return jwt_token
                else:
                    logger.warning(
                        "SDK jwt_generator returned None - this should not happen with valid credentials"
                    )
                    return None

            except Exception as jwt_error:
                logger.error(f"Exception in jwt_generator.build_ws_jwt: {jwt_error}")
                import traceback

                logger.debug(f"JWT generation traceback: {traceback.format_exc()}")
                return None

        except Exception as e:
            logger.warning(
                f"Failed to generate WebSocket JWT token (outer exception): {e}"
            )
            import traceback

            logger.debug(f"Outer exception traceback: {traceback.format_exc()}")
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
                        f"Failed to fetch historical data, continuing with WebSocket only: {e}"
                    )

            # Try to get current price to ensure connectivity
            try:
                current_price = await self.fetch_latest_price()
                if current_price:
                    logger.info(f"Successfully fetched current price: ${current_price}")
            except Exception as e:
                logger.warning(f"Could not fetch current price: {e}")

            self._is_connected = True
            logger.info("Successfully connected to market data feeds")

        except Exception as e:
            logger.error(f"Failed to connect to market data feeds: {e}")
            self._is_connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from all data feeds and cleanup resources."""
        self._is_connected = False
        self._running = False

        # Stop message processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Stop WebSocket connection
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Disconnected from market data feeds")

    async def _initialize_clients(self) -> None:
        """Initialize REST client and HTTP session."""
        # Initialize HTTP session for direct API calls
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.exchange.api_timeout)
        )

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

        if has_legacy_credentials:
            self._rest_client = CoinbaseAdvancedTrader(
                api_key=settings.exchange.cb_api_key.get_secret_value(),
                api_secret=settings.exchange.cb_api_secret.get_secret_value(),
                passphrase=settings.exchange.cb_passphrase.get_secret_value(),
                sandbox=settings.exchange.cb_sandbox,
            )
            logger.info("Initialized REST client with legacy Coinbase credentials")
        elif has_cdp_credentials:
            self._rest_client = CoinbaseAdvancedTrader(
                api_key=settings.exchange.cdp_api_key_name.get_secret_value(),
                api_secret=settings.exchange.cdp_private_key.get_secret_value(),
            )
            logger.info("Initialized REST client with CDP Coinbase credentials")
        else:
            logger.warning(
                "No Coinbase credentials provided, using public endpoints only"
            )

    async def _fetch_historical_data_batched(
        self,
        start_time: datetime,
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

        logger.info(f"Fetching {required_candles} candles in batches of {batch_size}")

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
            if batch_start < start_time:
                batch_start = start_time

            logger.info(
                f"Fetching batch {batch_num + 1}/{batches_needed}: {batch_start} to {current_end}"
            )

            try:
                # Fetch this batch
                batch_data = await self._fetch_single_batch(
                    batch_start, current_end, granularity
                )
                if batch_data:
                    all_data.extend(batch_data)
                    logger.info(
                        f"Batch {batch_num + 1} completed: {len(batch_data)} candles"
                    )
                else:
                    logger.warning(f"Batch {batch_num + 1} returned no data")

                # Add small delay between requests to be respectful to API
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching batch {batch_num + 1}: {e}")
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
            f"Successfully fetched {len(unique_data)} candles across {batches_needed} batches"
        )
        self._validate_data_sufficiency(len(unique_data))

        return unique_data

    async def _fetch_single_batch(
        self, start_time: datetime, end_time: datetime, granularity: str
    ) -> list[MarketData]:
        """Fetch a single batch of historical data."""
        try:
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self._data_symbol}/candles"

            params = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "granularity": self._format_granularity(granularity),
            }

            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(
                        f"API request failed with status {response.status}: {await response.text()}"
                    )

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
                        logger.warning(f"Failed to parse candle data: {e}")
                        continue

                return batch_data

        except Exception as e:
            logger.error(f"Failed to fetch batch from {start_time} to {end_time}: {e}")
            return []

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
                    f"Adjusted candle limit from {self.candle_limit} to {max_candles} due to API granularity mismatch"
                )

        default_start = end_time - timedelta(seconds=interval_seconds * max_candles)
        start_time = start_time or default_start

        # Log the actual number of candles that will be requested
        time_diff = (end_time - start_time).total_seconds()
        expected_candles = int(time_diff / api_interval_seconds)

        # Final safety check - ensure we never exceed 350 candles
        if expected_candles > 350:
            logger.error(
                f"Calculated {expected_candles} candles which exceeds API limit of 350!"
            )
            # Adjust start_time to ensure we stay under limit
            safe_seconds = 300 * api_interval_seconds  # Use 300 to be extra safe
            start_time = end_time - timedelta(seconds=safe_seconds)
            time_diff = safe_seconds
            expected_candles = 300
            logger.warning(
                f"Adjusted start_time to {start_time} to stay within API limits"
            )

        logger.info(
            f"Fetching historical data for {self._data_symbol} from {start_time} to {end_time}"
        )
        logger.info(
            f"Requesting {expected_candles} candles at {api_granularity} granularity (requested: {granularity})"
        )

        # Warn if we're not getting enough data for indicators
        min_required_for_indicators = 100  # VuManChu needs ~80 + buffer
        if expected_candles < min_required_for_indicators:
            logger.warning(
                f"Fetching only {expected_candles} candles, but indicators need at least "
                f"{min_required_for_indicators} for reliable calculations. Consider increasing candle_limit."
            )

        try:
            # Use public API endpoint for historical candles with data symbol
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self._data_symbol}/candles"

            params = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "granularity": self._format_granularity(granularity),
            }

            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(
                        f"API request failed with status {response.status}: {await response.text()}"
                    )

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
                        logger.warning(f"Failed to parse candle data: {e}")
                        continue

                # Sort by timestamp
                historical_data.sort(key=lambda x: x.timestamp)

                # Update cache
                self._ohlcv_cache = historical_data[-self.candle_limit :]
                self._last_update = datetime.now(UTC)
                self._cache_timestamps["ohlcv"] = self._last_update

                logger.info(f"Loaded {len(self._ohlcv_cache)} historical candles")

                # Validate we have sufficient data for indicators
                self._validate_data_sufficiency(len(self._ohlcv_cache))

                return historical_data

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            # Fallback to cached data if available
            if self._ohlcv_cache:
                logger.info("Using cached historical data")
                return self._ohlcv_cache
            raise

    async def fetch_latest_price(self) -> Decimal | None:
        """
        Fetch the latest price for the symbol from Coinbase REST API.

        Returns:
            Latest price as Decimal or None if unavailable
        """
        # Check cache first
        if self._is_cache_valid("price"):
            return self._price_cache.get("price")

        try:
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self._data_symbol}"

            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch latest price: {response.status}")
                    return self.get_latest_price()  # Fall back to cached data

                data = await response.json()
                price_str = data.get("price")

                if price_str:
                    price = Decimal(price_str)
                    # Update cache
                    self._price_cache["price"] = price
                    self._cache_timestamps["price"] = datetime.now(UTC)
                    return price

        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")

        # Fall back to cached OHLCV data
        return self.get_latest_price()

    async def fetch_orderbook(self, level: int = 2) -> dict[str, Any] | None:
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

        try:
            url = f"{self.COINBASE_REST_URL}/api/v3/brokerage/market/products/{self.symbol}/book"
            params = {"limit": min(level * 10, 50)}  # Reasonable limit based on level

            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch orderbook: {response.status}")
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

        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return None

    async def _start_websocket(self) -> None:
        """
        Start WebSocket connection for real-time data with auto-reconnect.

        Subscribes to:
        - Real-time price updates (ticker)
        - Trade stream
        - Level 1 order book updates (if needed)
        """
        self._running = True

        # Start non-blocking message processor
        self._processing_task = asyncio.create_task(self._process_websocket_messages())

        # Start WebSocket handler
        self._ws_task = asyncio.create_task(self._websocket_handler())
        logger.info("WebSocket connection and message processor started")

        # Give WebSocket a moment to establish connection
        await asyncio.sleep(0.5)

    async def _websocket_handler(self) -> None:
        """
        Handle WebSocket connection with automatic reconnection.
        """
        while self._is_connected:
            try:
                await self._connect_websocket()
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    delay = min(
                        2**self._reconnect_attempts, 60
                    )  # Exponential backoff, max 60s
                    logger.info(
                        f"Reconnecting in {delay}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Max reconnection attempts reached, stopping WebSocket"
                    )
                    break

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
                sub["jwt"] = jwt_token
            logger.info("Using CDP JWT authentication for WebSocket")
        else:
            logger.info("Using public WebSocket connection (no authentication)")

        logger.debug(f"WebSocket subscriptions: {json.dumps(subscriptions, indent=2)}")

        async with websockets.connect(
            self.COINBASE_WS_URL, timeout=settings.exchange.websocket_timeout
        ) as websocket:
            self._ws_connection = websocket

            # Send all subscriptions
            for i, subscription in enumerate(subscriptions):
                await websocket.send(json.dumps(subscription))
                logger.debug(
                    f"Sent subscription {i+1}/{len(subscriptions)}: {subscription.get('channel', 'unknown')}"
                )
                # Small delay between subscriptions to avoid overwhelming the server
                await asyncio.sleep(0.1)

            logger.info(
                f"Subscribed to {len(subscriptions)} WebSocket feeds for {self.symbol}"
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
                            f"WebSocket message #{message_count}: {parsed_message.get('channel', 'unknown')} - {parsed_message.get('type', 'unknown')}"
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
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")

    async def _process_websocket_messages(self) -> None:
        """
        Non-blocking WebSocket message processor using asyncio queue.

        Processes messages in background without blocking WebSocket reception.
        """
        while self._running:
            try:
                # Get message without blocking for too long
                message = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)

                # Process message in background task to avoid blocking
                asyncio.create_task(self._handle_websocket_message_async(message))

            except TimeoutError:
                # No message available, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(0.1)

    async def _handle_websocket_message_async(self, message: dict[str, Any]) -> None:
        """
        Handle WebSocket message asynchronously without blocking the queue processor.

        Args:
            message: Parsed WebSocket message
        """
        try:
            await self._handle_websocket_message(message)
        except Exception as e:
            logger.error(f"Error in async message handler: {e}")

    async def _handle_websocket_message(self, message: dict[str, Any]) -> None:
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
                f"WebSocket subscriptions confirmed: {message.get('events', [])}"
            )
        elif channel == "heartbeats":
            # Heartbeat messages - just log occasionally to confirm connection
            events = message.get("events", [])
            if events:
                counter = events[0].get("heartbeat_counter", "unknown")
                logger.debug(f"Heartbeat #{counter} received")
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
            logger.error(f"WebSocket error: {message.get('message', 'Unknown error')}")
            # Log the full error message for debugging
            logger.debug(
                f"Full WebSocket error message: {json.dumps(message, indent=2)}"
            )
        else:
            # Log any unhandled message types for debugging
            logger.debug(
                f"Unhandled WebSocket message - Channel: {channel}, Type: {msg_type}"
            )
            logger.debug(f"Message: {json.dumps(message, indent=2)}")

    async def _handle_ticker_update(self, message: dict[str, Any]) -> None:
        """
        Handle ticker price updates.

        Args:
            message: Ticker message from WebSocket (Advanced Trading API format)
        """
        try:
            # Advanced Trading API format: events array with tickers
            events = message.get("events", [])
            timestamp = datetime.fromisoformat(
                message.get("timestamp", "").replace("Z", "+00:00")
            )

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
                                f"First WebSocket market data received for {self.symbol} at ${price}"
                            )

                        # Update last update time to keep connection status fresh
                        self._last_update = datetime.now(UTC)

                        logger.debug(f"Ticker update: {self.symbol} = ${price}")

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

        except Exception as e:
            logger.error(f"Error handling ticker update: {e}")
            logger.debug(f"Ticker message: {json.dumps(message, indent=2)}")

    async def _handle_trade_update(self, message: dict[str, Any]) -> None:
        """
        Handle trade/market_trades updates.

        Args:
            message: Market trades message from WebSocket (Advanced Trading API format)
        """
        try:
            # Advanced Trading API format: events array with trades
            events = message.get("events", [])
            timestamp = datetime.fromisoformat(
                message.get("timestamp", "").replace("Z", "+00:00")
            )

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
                                trade.get("time", timestamp.isoformat()).replace(
                                    "Z", "+00:00"
                                )
                            ),
                            "trade_id": trade.get("trade_id", ""),
                        }

                        logger.debug(
                            f"Trade update: {self.symbol} {trade_data['side']} {trade_data['size']} @ ${trade_data['price']}"
                        )

                        # Mark that we've received real WebSocket data
                        if not self._websocket_data_received:
                            self._websocket_data_received = True
                            self._first_websocket_data_time = datetime.now(UTC)
                            logger.info(
                                f"First WebSocket market data received for {self.symbol} via trade at ${trade_data['price']}"
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

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
            logger.debug(f"Trade message: {json.dumps(message, indent=2)}")

    def get_latest_ohlcv(self, limit: int | None = None) -> list[MarketData]:
        """
        Get the latest OHLCV data.

        Args:
            limit: Number of candles to return (default: all cached data)

        Returns:
            List of MarketData objects
        """
        if limit is None:
            return self._ohlcv_cache.copy()

        return self._ohlcv_cache[-limit:].copy()

    def get_latest_price(self) -> Decimal | None:
        """
        Get the most recent price from cache.

        Returns:
            Latest price or None if no data available
        """
        # Try price cache first
        if "price" in self._price_cache:
            return self._price_cache["price"]

        # Fall back to latest candle close price
        if self._ohlcv_cache:
            return self._ohlcv_cache[-1].close

        return None

    def to_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        """
        Convert OHLCV data to pandas DataFrame for indicator calculations.

        Args:
            limit: Number of candles to include

        Returns:
            DataFrame with OHLCV columns
        """
        data = self.get_latest_ohlcv(limit)

        if not data:
            return pd.DataFrame()

        df_data = []
        for candle in data:
            df_data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)

        return df

    def subscribe_to_updates(self, callback: Callable[[MarketData], None]) -> None:
        """
        Subscribe to real-time data updates.

        Args:
            callback: Function to call when new data arrives
        """
        self._subscribers.append(callback)
        logger.debug(f"Added subscriber: {callback.__name__}")

    def unsubscribe_from_updates(self, callback: Callable[[MarketData], None]) -> None:
        """
        Unsubscribe from real-time data updates.

        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Removed subscriber: {callback.__name__}")

    async def _notify_subscribers(self, data: MarketData) -> None:
        """
        Notify all subscribers of new data using non-blocking tasks.

        Args:
            data: New market data to broadcast
        """
        # Create tasks for all subscriber callbacks to run concurrently
        tasks = []
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Create task for async callback
                    task = asyncio.create_task(
                        self._safe_callback_async(callback, data)
                    )
                    tasks.append(task)
                else:
                    # Create task for sync callback
                    task = asyncio.create_task(self._safe_callback_sync(callback, data))
                    tasks.append(task)
            except Exception as e:
                logger.error(
                    f"Error creating subscriber task for {callback.__name__}: {e}"
                )

        # Don't wait for all tasks to complete - fire and forget for non-blocking behavior
        # Tasks will run in background without blocking the caller

    async def _safe_callback_async(self, callback: Callable, data: MarketData) -> None:
        """Safely execute async callback."""
        try:
            await callback(data)
        except Exception as e:
            logger.error(f"Error in async subscriber callback {callback.__name__}: {e}")

    async def _safe_callback_sync(self, callback: Callable, data: MarketData) -> None:
        """Safely execute sync callback in thread pool."""
        try:
            # Run sync callback in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, data)
        except Exception as e:
            logger.error(f"Error in sync subscriber callback {callback.__name__}: {e}")

    def is_connected(self) -> bool:
        """
        Check if the data provider is connected and receiving data.

        Returns:
            True if connected and data is fresh
        """
        if not self._is_connected:
            return False

        # Check if we have any data at all (either historical or WebSocket)
        if not self._last_update:
            return False

        # Consider data stale if older than 2 minutes
        staleness_threshold = timedelta(minutes=2)
        return datetime.now(UTC) - self._last_update < staleness_threshold

    def has_websocket_data(self) -> bool:
        """
        Check if WebSocket is receiving real market data.

        Returns:
            True if WebSocket data has been received
        """
        return self._websocket_data_received

    async def wait_for_websocket_data(self, timeout: int = 30) -> bool:
        """
        Wait for WebSocket to start receiving real market data.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if data received, False if timeout
        """
        start_time = datetime.now(UTC)

        while not self._websocket_data_received:
            if (datetime.now(UTC) - start_time).total_seconds() > timeout:
                logger.warning(f"Timeout waiting for WebSocket data after {timeout}s")
                return False

            # Check WebSocket connection status
            if self._ws_connection and hasattr(self._ws_connection, "state"):
                ws_state = self._ws_connection.state
                if ws_state != 1:  # Not OPEN
                    logger.warning(f"WebSocket not in OPEN state: {ws_state}")

            await asyncio.sleep(0.1)

        return True

    def get_data_status(self) -> dict[str, Any]:
        """
        Get comprehensive status information about the data provider.

        Returns:
            Dictionary with status information
        """
        # Check WebSocket connection status more accurately
        ws_connected = False
        if self._ws_connection is not None:
            try:
                # Check if WebSocket is in OPEN state (1)
                ws_connected = (
                    hasattr(self._ws_connection, "state")
                    and self._ws_connection.state == 1
                )
            except Exception:
                ws_connected = False

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "connected": self.is_connected(),
            "websocket_connected": ws_connected,
            "websocket_data_received": self._websocket_data_received,
            "first_websocket_data_time": self._first_websocket_data_time,
            "cached_candles": len(self._ohlcv_cache),
            "cached_ticks": len(self._tick_cache),
            "last_update": self._last_update,
            "latest_price": self.get_latest_price(),
            "subscribers": len(self._subscribers),
            "reconnect_attempts": self._reconnect_attempts,
            "cache_status": {
                key: self._is_cache_valid(key) for key in self._cache_timestamps.keys()
            },
        }

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid based on TTL.

        Args:
            cache_key: Cache key to check

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self._cache_timestamps:
            return False

        cache_age = datetime.now(UTC) - self._cache_timestamps[cache_key]
        return cache_age < self._cache_ttl

    def _interval_to_seconds(self, interval: str) -> int:
        """
        Convert interval string to seconds.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h')

        Returns:
            Interval in seconds
        """
        multipliers = {"m": 60, "h": 3600, "d": 86400}

        if interval[-1] in multipliers:
            return int(interval[:-1]) * multipliers[interval[-1]]

        # Default to 1 minute
        return 60

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
                        f"Interval {interval} not supported by Coinbase API, using {supported_interval} instead"
                    )
                    return granularity_map[supported_interval]

            # Default to ONE_DAY if interval is larger than all supported
            logger.warning(
                f"Interval {interval} not supported by Coinbase API, defaulting to ONE_DAY"
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
            timestamp=datetime.fromtimestamp(int(candle_data["start"])),
            open=Decimal(str(candle_data["open"])),
            high=Decimal(str(candle_data["high"])),
            low=Decimal(str(candle_data["low"])),
            close=Decimal(str(candle_data["close"])),
            volume=Decimal(str(candle_data["volume"])),
        )

    def _validate_market_data(self, data: MarketData) -> bool:
        """
        Validate market data for quality and integrity.

        Args:
            data: MarketData to validate

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Basic validation
            if data.open <= 0 or data.high <= 0 or data.low <= 0 or data.close <= 0:
                logger.warning("Invalid price data: negative or zero prices")
                return False

            if data.volume < self._min_volume:
                logger.warning(f"Invalid volume: {data.volume} < {self._min_volume}")
                return False

            # Price consistency checks
            if data.high < max(data.open, data.close) or data.low > min(
                data.open, data.close
            ):
                logger.warning(
                    f"Inconsistent OHLC data: High={data.high}, Low={data.low}, Open={data.open}, Close={data.close}"
                )
                return False

            # Check for extreme price movements (compared to previous candle)
            if self._ohlcv_cache:
                last_price = self._ohlcv_cache[-1].close
                price_change = abs(data.close - last_price) / last_price
                if price_change > self._max_price_deviation:
                    logger.warning(
                        f"Extreme price movement detected: {price_change:.2%} change"
                    )
                    # Don't reject, but log for monitoring

            return True

        except Exception as e:
            logger.error(f"Error validating market data: {e}")
            return False

    def _validate_price_data(
        self, new_price: Decimal, last_price: Decimal | None
    ) -> bool:
        """
        Validate price data for anomalies.

        Args:
            new_price: New price to validate
            last_price: Previous price for comparison

        Returns:
            True if price is valid, False otherwise
        """
        try:
            if last_price is None or last_price <= 0:
                return True

            # Check for excessive price deviation (>20% change)
            change_pct = abs(new_price - last_price) / last_price
            if change_pct > 0.20:  # 20% max change
                logger.warning(
                    f"Suspicious price change: {change_pct:.2%} from {last_price} to {new_price}"
                )
                return False

            # Check for price spikes (>50% in single update)
            if change_pct > 0.50:
                logger.error(
                    f"Extreme price spike detected: {change_pct:.2%} change - rejecting data"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating price data: {e}")
            return True  # Default to allowing data on validation error

    def _validate_data_sufficiency(self, candle_count: int) -> None:
        """
        Validate that we have sufficient data for reliable indicator calculations.

        Args:
            candle_count: Number of candles available
        """
        # VuManChu indicators need ~80 candles minimum for reliable calculation
        # RSI+MFI period=60, EMA ribbon max=34, plus buffer
        min_required = 100  # Conservative estimate with buffer
        optimal_required = 200  # Optimal for stable calculations

        if candle_count < min_required:
            logger.error(
                f"Insufficient data for reliable indicator calculations! "
                f"Have {candle_count} candles, need minimum {min_required}. "
                f"Indicators may produce unreliable signals or errors."
            )
        elif candle_count < optimal_required:
            logger.warning(
                f"Suboptimal data for indicator calculations. "
                f"Have {candle_count} candles, recommend {optimal_required} for best accuracy. "
                f"Early indicator values may be less reliable."
            )
        else:
            logger.info(
                f"Sufficient data for reliable indicator calculations: {candle_count} candles"
            )

    def clear_cache(self) -> None:
        """
        Clear all cached data.
        """
        self._ohlcv_cache.clear()
        self._price_cache.clear()
        self._orderbook_cache.clear()
        self._tick_cache.clear()
        self._cache_timestamps.clear()
        self._websocket_data_received = False
        self._first_websocket_data_time = None
        logger.info("All cached data cleared")

    def get_tick_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get recent tick/trade data.

        Args:
            limit: Maximum number of ticks to return

        Returns:
            List of tick data dictionaries
        """
        if limit is None:
            return self._tick_cache.copy()

        return self._tick_cache[-limit:].copy()


class MarketDataClient:
    """
    High-level client for market data operations.

    This class provides a simplified interface to the MarketDataProvider
    with additional convenience methods and error handling.
    """

    def __init__(self, symbol: str | None = None, interval: str | None = None) -> None:
        """
        Initialize the market data client.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            interval: Candle interval (e.g., '1m', '5m', '1h')
        """
        self.provider = MarketDataProvider(symbol, interval)
        self._initialized = False

    async def __aenter__(self) -> "MarketDataClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to market data feeds."""
        if not self._initialized:
            await self.provider.connect()
            self._initialized = True
            logger.info("MarketDataClient connected successfully")

    async def disconnect(self) -> None:
        """Disconnect from market data feeds."""
        if self._initialized:
            await self.provider.disconnect()
            self._initialized = False
            logger.info("MarketDataClient disconnected")

    async def get_historical_data(
        self, lookback_hours: int = 24, granularity: str | None = None
    ) -> pd.DataFrame:
        """
        Get historical data as a pandas DataFrame.

        Args:
            lookback_hours: Hours of historical data to fetch
            granularity: Candle granularity (default: self.interval)

        Returns:
            DataFrame with OHLCV data
        """
        if not self._initialized:
            await self.connect()

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=lookback_hours)

        data = await self.provider.fetch_historical_data(
            start_time=start_time, end_time=end_time, granularity=granularity
        )

        return self._to_dataframe(data)

    async def get_current_price(self) -> Decimal | None:
        """
        Get the current market price.

        Returns:
            Current price or None if unavailable
        """
        if not self._initialized:
            await self.connect()

        return await self.provider.fetch_latest_price()

    async def get_orderbook_snapshot(self, level: int = 2) -> dict[str, Any] | None:
        """
        Get a snapshot of the current order book.

        Args:
            level: Order book depth level

        Returns:
            Order book data or None if unavailable
        """
        if not self._initialized:
            await self.connect()

        return await self.provider.fetch_orderbook(level)

    def get_latest_ohlcv_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        """
        Get latest OHLCV data as DataFrame.

        Args:
            limit: Number of candles to include

        Returns:
            DataFrame with OHLCV data
        """
        return self.provider.to_dataframe(limit)

    def subscribe_to_price_updates(
        self, callback: Callable[[MarketData], None]
    ) -> None:
        """
        Subscribe to real-time price updates.

        Args:
            callback: Function to call on price updates
        """
        self.provider.subscribe_to_updates(callback)

    def unsubscribe_from_price_updates(
        self, callback: Callable[[MarketData], None]
    ) -> None:
        """
        Unsubscribe from real-time price updates.

        Args:
            callback: Function to remove from subscribers
        """
        self.provider.unsubscribe_from_updates(callback)

    def get_connection_status(self) -> dict[str, Any]:
        """
        Get detailed connection and data status.

        Returns:
            Status dictionary
        """
        return self.provider.get_data_status()

    def _to_dataframe(self, data: list[MarketData]) -> pd.DataFrame:
        """
        Convert MarketData list to DataFrame.

        Args:
            data: List of MarketData objects

        Returns:
            DataFrame with OHLCV columns
        """
        if not data:
            return pd.DataFrame()

        df_data = []
        for candle in data:
            df_data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": float(candle.open),
                    "high": float(candle.high),
                    "low": float(candle.low),
                    "close": float(candle.close),
                    "volume": float(candle.volume),
                }
            )

        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df


# Factory function for easy client creation
def create_market_data_client(
    symbol: str | None = None, interval: str | None = None
) -> MarketDataClient:
    """
    Factory function to create a MarketDataClient instance.

    Args:
        symbol: Trading symbol (default: from settings)
        interval: Candle interval (default: from settings)

    Returns:
        MarketDataClient instance
    """
    return MarketDataClient(symbol, interval)
