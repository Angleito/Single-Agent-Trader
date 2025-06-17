"""Bluefin WebSocket client for real-time market data streaming."""

import asyncio
import json
import logging
from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..types import MarketData

logger = logging.getLogger(__name__)


class BluefinWebSocketClient:
    """
    WebSocket client for Bluefin real-time market data.
    
    Connects to Bluefin's notification WebSocket service to receive:
    - Real-time tick/trade data
    - Order book updates
    - Market status updates
    
    Builds OHLCV candles from incoming tick data and maintains
    a rolling buffer for indicator calculations.
    """
    
    # Bluefin WebSocket endpoints
    NOTIFICATION_WS_URL = "wss://notifications.api.sui-prod.bluefin.io"
    DAPI_WS_URL = "wss://dapi.api.sui-prod.bluefin.io"
    
    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        candle_limit: int = 500,
        on_candle_update: Optional[Callable[[MarketData], None]] = None
    ):
        """
        Initialize the Bluefin WebSocket client.
        
        Args:
            symbol: Trading symbol (e.g., 'SUI-PERP')
            interval: Candle interval for aggregation
            candle_limit: Maximum number of candles to maintain in buffer
            on_candle_update: Callback function for candle updates
        """
        self.symbol = symbol
        self.interval = interval
        self.candle_limit = candle_limit
        self.on_candle_update = on_candle_update
        
        # WebSocket connection state
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # seconds
        
        # Candle building state
        self._candle_buffer = deque(maxlen=candle_limit)
        self._current_candle: Optional[MarketData] = None
        self._tick_buffer = deque(maxlen=1000)  # Store recent ticks
        
        # Subscription tracking
        self._subscribed_channels = set()
        self._subscription_id = 1
        
        # Tasks
        self._connection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._candle_aggregation_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self._last_message_time: Optional[datetime] = None
        self._message_count = 0
        self._error_count = 0
        
        logger.info(f"Initialized BluefinWebSocketClient for {symbol} with {interval} candles")
    
    async def connect(self) -> None:
        """Establish WebSocket connection and start data streaming."""
        if self._connected:
            logger.warning("Already connected to Bluefin WebSocket")
            return
        
        logger.info(f"Connecting to Bluefin WebSocket at {self.NOTIFICATION_WS_URL}")
        
        # Start connection task
        self._connection_task = asyncio.create_task(self._connection_handler())
        
        # Start candle aggregation task
        self._candle_aggregation_task = asyncio.create_task(self._candle_aggregator())
        
        # Wait for initial connection
        await self._wait_for_connection(timeout=30)
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources."""
        logger.info("Disconnecting from Bluefin WebSocket")
        
        self._connected = False
        
        # Cancel tasks
        tasks = [
            self._connection_task,
            self._heartbeat_task,
            self._candle_aggregation_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket connection
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        logger.info("Disconnected from Bluefin WebSocket")
    
    async def _connection_handler(self) -> None:
        """Handle WebSocket connection with automatic reconnection."""
        while True:
            try:
                await self._connect_and_subscribe()
                self._reconnect_attempts = 0
                
                # Handle incoming messages
                await self._message_handler()
                
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self._connected = False
                self._ws = None
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._connected = False
                self._ws = None
                self._error_count += 1
            
            # Check if we should reconnect
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                logger.error("Max reconnection attempts reached, stopping")
                break
            
            self._reconnect_attempts += 1
            delay = min(self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)), 60)
            logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
            
            await asyncio.sleep(delay)
    
    async def _connect_and_subscribe(self) -> None:
        """Establish WebSocket connection and subscribe to channels."""
        # Connect to WebSocket
        self._ws = await websockets.connect(
            self.NOTIFICATION_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        )
        
        self._connected = True
        logger.info("WebSocket connection established")
        
        # Start heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_handler())
        
        # Subscribe to market data channels
        await self._subscribe_to_market_data()
    
    async def _subscribe_to_market_data(self) -> None:
        """Subscribe to relevant market data channels."""
        # Try different subscription formats for Bluefin
        
        # Format 1: Direct channel subscription
        sub1 = {
            "type": "subscribe",
            "channel": f"trade:{self.symbol}"
        }
        await self._send_message(sub1)
        
        # Format 2: Standard subscribe with params
        sub2 = {
            "id": self._get_next_subscription_id(),
            "method": "subscribe",
            "params": {
                "channel": "trades",
                "symbol": self.symbol
            }
        }
        await self._send_message(sub2)
        
        # Format 3: Bluefin-specific format (if different)
        sub3 = {
            "action": "subscribe",
            "channel": "market",
            "symbol": self.symbol,
            "type": "trades"
        }
        await self._send_message(sub3)
        
        # Also try ticker subscription
        ticker_sub = {
            "type": "subscribe",
            "channel": f"ticker:{self.symbol}"
        }
        await self._send_message(ticker_sub)
        
        logger.info(f"Sent multiple subscription formats for {self.symbol}")
    
    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        async for message in self._ws:
            try:
                data = json.loads(message)
                self._last_message_time = datetime.now(UTC)
                self._message_count += 1
                
                # Log message count for debugging (without sensitive data)
                if self._message_count <= 20:
                    msg_type = data.get('type', 'unknown') if isinstance(data, dict) else 'unknown'
                    logger.debug(f"WebSocket message #{self._message_count} type: {msg_type}")
                
                await self._process_message(data)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse WebSocket message: {e}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                self._error_count += 1
    
    async def _process_message(self, data: dict[str, Any]) -> None:
        """
        Process incoming WebSocket message.
        
        Args:
            data: Parsed message data
        """
        # Handle subscription confirmations
        if "id" in data and "result" in data:
            logger.debug(f"Subscription {data['id']} confirmed: {data['result']}")
            return
        
        # Handle error messages
        if "error" in data:
            logger.error(f"WebSocket error: {data['error']}")
            return
        
        # Handle Bluefin-specific event names
        event_name = data.get("eventName")
        
        if event_name == "TickerUpdate":
            await self._handle_bluefin_ticker_update(data)
        elif event_name == "MarketDataUpdate":
            await self._handle_bluefin_market_update(data)
        elif event_name == "RecentTrades":
            await self._handle_bluefin_trades(data)
        else:
            # Handle standard channels
            channel = data.get("channel", data.get("ch"))
            
            if channel == "trade" or "trade" in str(channel):
                await self._handle_trade_update(data)
            elif channel == "ticker" or "ticker" in str(channel):
                await self._handle_ticker_update(data)
            elif channel == "orderbook" or "orderbook" in str(channel):
                await self._handle_orderbook_update(data)
            else:
                # Log unhandled message types periodically
                if self._message_count % 100 == 0:
                    logger.debug(f"Unhandled event/channel: {event_name or channel}")
    
    async def _handle_trade_update(self, data: dict[str, Any]) -> None:
        """
        Handle trade/tick updates.
        
        Args:
            data: Trade message data
        """
        try:
            # Extract trade data (adjust based on actual Bluefin format)
            trades = data.get("data", data.get("trades", []))
            if isinstance(trades, dict):
                trades = [trades]
            
            for trade in trades:
                # Parse trade fields
                price = Decimal(str(trade.get("price", 0)))
                size = Decimal(str(trade.get("size", trade.get("amount", 0))))
                side = trade.get("side", "")
                timestamp = self._parse_timestamp(trade.get("timestamp", trade.get("ts")))
                
                if price > 0 and size > 0:
                    trade_data = {
                        "price": price,
                        "size": size,
                        "side": side,
                        "timestamp": timestamp,
                        "trade_id": trade.get("id", "")
                    }
                    
                    # Add to tick buffer
                    self._tick_buffer.append(trade_data)
                    
                    # Update current candle
                    await self._update_candle_with_trade(trade_data)
                    
                    logger.debug(f"Trade: {self.symbol} {side} {size} @ {price}")
        
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    async def _handle_ticker_update(self, data: dict[str, Any]) -> None:
        """
        Handle ticker price updates.
        
        Args:
            data: Ticker message data
        """
        try:
            # Extract ticker data
            ticker = data.get("data", data)
            if isinstance(ticker, list) and ticker:
                ticker = ticker[0]
            
            last_price = Decimal(str(ticker.get("last", ticker.get("lastPrice", 0))))
            timestamp = self._parse_timestamp(ticker.get("timestamp", ticker.get("ts")))
            
            if last_price > 0:
                # Update current candle close price
                if self._current_candle:
                    self._current_candle = MarketData(
                        symbol=self._current_candle.symbol,
                        timestamp=self._current_candle.timestamp,
                        open=self._current_candle.open,
                        high=max(self._current_candle.high, last_price),
                        low=min(self._current_candle.low, last_price),
                        close=last_price,
                        volume=self._current_candle.volume
                    )
                
                logger.debug(f"Ticker update: {self.symbol} = {last_price}")
        
        except Exception as e:
            logger.error(f"Error handling ticker update: {e}")
    
    async def _handle_orderbook_update(self, data: dict[str, Any]) -> None:
        """
        Handle orderbook updates.
        
        Args:
            data: Orderbook message data
        """
        # For now, just log that we received orderbook data
        # Full orderbook handling can be implemented if needed
        logger.debug(f"Received orderbook update for {self.symbol}")
    
    async def _handle_bluefin_ticker_update(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin-specific ticker updates.
        
        Args:
            data: Ticker update message
        """
        try:
            tickers = data.get("data", [])
            
            for ticker in tickers:
                if ticker.get("symbol") == self.symbol:
                    # Extract price data (values are in 18 decimal format)
                    price_str = ticker.get("price", "0")
                    last_price_str = ticker.get("lastPrice", "0")
                    
                    # Convert from 18 decimal format to regular decimal
                    price = Decimal(price_str) / Decimal(10**18)
                    last_price = Decimal(last_price_str) / Decimal(10**18)
                    
                    # Use the most recent price
                    current_price = price if price > 0 else last_price
                    
                    if current_price > 0:
                        # Create a tick from ticker data
                        trade_data = {
                            "price": current_price,
                            "size": Decimal("0.1"),  # Placeholder size
                            "side": "buy" if ticker.get("priceDirection", 0) > 0 else "sell",
                            "timestamp": datetime.now(UTC),
                            "trade_id": f"ticker_{self._message_count}"
                        }
                        
                        # Add to tick buffer
                        self._tick_buffer.append(trade_data)
                        
                        # Update current candle
                        await self._update_candle_with_trade(trade_data)
                        
                        logger.debug(f"Ticker update: {self.symbol} = ${current_price}")
                    
                    break
        
        except Exception as e:
            logger.error(f"Error handling Bluefin ticker update: {e}")
    
    async def _handle_bluefin_market_update(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin market data updates.
        
        Args:
            data: Market update message
        """
        # Similar to ticker but with more detailed market data
        await self._handle_bluefin_ticker_update(data)
    
    async def _handle_bluefin_trades(self, data: dict[str, Any]) -> None:
        """
        Handle Bluefin recent trades.
        
        Args:
            data: Trades message
        """
        try:
            trades = data.get("data", [])
            
            for trade in trades:
                if trade.get("symbol") == self.symbol:
                    # Extract trade data
                    price_str = trade.get("price", "0")
                    size_str = trade.get("quantity", trade.get("size", "0"))
                    
                    # Convert from 18 decimal format
                    price = Decimal(price_str) / Decimal(10**18)
                    size = Decimal(size_str) / Decimal(10**18)
                    
                    if price > 0 and size > 0:
                        trade_data = {
                            "price": price,
                            "size": size,
                            "side": trade.get("side", "").lower(),
                            "timestamp": self._parse_timestamp(trade.get("timestamp")),
                            "trade_id": trade.get("id", "")
                        }
                        
                        # Add to tick buffer
                        self._tick_buffer.append(trade_data)
                        
                        # Update current candle
                        await self._update_candle_with_trade(trade_data)
                        
                        logger.debug(f"Trade: {self.symbol} {trade_data['side']} {size} @ ${price}")
        
        except Exception as e:
            logger.error(f"Error handling Bluefin trades: {e}")
    
    async def _update_candle_with_trade(self, trade: dict[str, Any]) -> None:
        """
        Update current candle with trade data.
        
        Args:
            trade: Trade data dictionary
        """
        price = trade["price"]
        size = trade["size"]
        timestamp = trade["timestamp"]
        
        # Check if we need to create a new candle
        if self._current_candle is None or self._should_create_new_candle(timestamp):
            await self._create_new_candle(timestamp, price)
        
        # Update current candle
        self._current_candle = MarketData(
            symbol=self._current_candle.symbol,
            timestamp=self._current_candle.timestamp,
            open=self._current_candle.open,
            high=max(self._current_candle.high, price),
            low=min(self._current_candle.low, price),
            close=price,
            volume=self._current_candle.volume + size
        )
    
    async def _create_new_candle(self, timestamp: datetime, price: Decimal) -> None:
        """
        Create a new candle and add the previous one to buffer.
        
        Args:
            timestamp: Candle timestamp
            price: Opening price
        """
        # Add previous candle to buffer
        if self._current_candle:
            self._candle_buffer.append(self._current_candle)
            
            # Notify callback
            if self.on_candle_update:
                try:
                    if asyncio.iscoroutinefunction(self.on_candle_update):
                        await self.on_candle_update(self._current_candle)
                    else:
                        self.on_candle_update(self._current_candle)
                except Exception as e:
                    logger.error(f"Error in candle update callback: {e}")
        
        # Create new candle
        candle_timestamp = self._get_candle_timestamp(timestamp)
        self._current_candle = MarketData(
            symbol=self.symbol,
            timestamp=candle_timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=Decimal("0")
        )
        
        logger.debug(f"Created new {self.interval} candle at {candle_timestamp}")
    
    async def _candle_aggregator(self) -> None:
        """
        Background task to ensure candles are created at regular intervals.
        """
        interval_seconds = self._interval_to_seconds(self.interval)
        
        while self._connected:
            try:
                await asyncio.sleep(interval_seconds)
                
                # Check if we need to force a new candle
                if self._current_candle:
                    now = datetime.now(UTC)
                    candle_age = (now - self._current_candle.timestamp).total_seconds()
                    
                    if candle_age >= interval_seconds:
                        # Use last close price for new candle
                        last_price = self._current_candle.close
                        await self._create_new_candle(now, last_price)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in candle aggregator: {e}")
    
    async def _heartbeat_handler(self) -> None:
        """Send periodic heartbeat/ping messages to keep connection alive."""
        while self._connected and self._ws:
            try:
                # Send ping every 30 seconds
                await asyncio.sleep(30)
                
                ping_message = {
                    "id": self._get_next_subscription_id(),
                    "method": "ping"
                }
                
                await self._send_message(ping_message)
                logger.debug("Sent heartbeat ping")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
    
    async def _send_message(self, message: dict[str, Any]) -> None:
        """
        Send message to WebSocket.
        
        Args:
            message: Message to send
        """
        if not self._ws or not self._connected:
            raise RuntimeError("WebSocket not connected")
        
        await self._ws.send(json.dumps(message))
        logger.debug(f"Sent message: {message}")
    
    async def _wait_for_connection(self, timeout: int = 30) -> bool:
        """
        Wait for WebSocket connection to be established.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if connected, False if timeout
        """
        start_time = datetime.now(UTC)
        
        while not self._connected:
            if (datetime.now(UTC) - start_time).total_seconds() > timeout:
                logger.warning(f"Timeout waiting for WebSocket connection after {timeout}s")
                return False
            
            await asyncio.sleep(0.1)
        
        return True
    
    def get_candles(self, limit: Optional[int] = None) -> list[MarketData]:
        """
        Get historical candles from buffer.
        
        Args:
            limit: Maximum number of candles to return
            
        Returns:
            List of MarketData objects
        """
        candles = list(self._candle_buffer)
        
        # Include current candle if available
        if self._current_candle:
            candles.append(self._current_candle)
        
        if limit:
            return candles[-limit:]
        
        return candles
    
    def get_latest_price(self) -> Optional[Decimal]:
        """
        Get the latest price.
        
        Returns:
            Latest price or None if not available
        """
        if self._current_candle:
            return self._current_candle.close
        
        if self._candle_buffer:
            return self._candle_buffer[-1].close
        
        return None
    
    def get_ticks(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Get recent tick data.
        
        Args:
            limit: Maximum number of ticks to return
            
        Returns:
            List of tick data dictionaries
        """
        ticks = list(self._tick_buffer)
        
        if limit:
            return ticks[-limit:]
        
        return ticks
    
    def get_status(self) -> dict[str, Any]:
        """
        Get WebSocket client status.
        
        Returns:
            Status dictionary
        """
        return {
            "connected": self._connected,
            "symbol": self.symbol,
            "interval": self.interval,
            "candles_buffered": len(self._candle_buffer),
            "current_candle": self._current_candle is not None,
            "ticks_buffered": len(self._tick_buffer),
            "subscribed_channels": list(self._subscribed_channels),
            "reconnect_attempts": self._reconnect_attempts,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "last_message_time": self._last_message_time,
            "latest_price": self.get_latest_price()
        }
    
    def _get_next_subscription_id(self) -> int:
        """Get next subscription ID."""
        sub_id = self._subscription_id
        self._subscription_id += 1
        return sub_id
    
    def _should_create_new_candle(self, timestamp: datetime) -> bool:
        """
        Check if a new candle should be created based on timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if new candle needed
        """
        if not self._current_candle:
            return True
        
        interval_seconds = self._interval_to_seconds(self.interval)
        candle_age = (timestamp - self._current_candle.timestamp).total_seconds()
        
        return candle_age >= interval_seconds
    
    def _get_candle_timestamp(self, timestamp: datetime) -> datetime:
        """
        Get normalized candle timestamp based on interval.
        
        Args:
            timestamp: Raw timestamp
            
        Returns:
            Normalized candle timestamp
        """
        interval_seconds = self._interval_to_seconds(self.interval)
        
        # Round down to nearest interval
        epoch = timestamp.timestamp()
        rounded = (epoch // interval_seconds) * interval_seconds
        
        return datetime.fromtimestamp(rounded, UTC)
    
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
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """
        Parse timestamp from various formats.
        
        Args:
            ts: Timestamp in various formats
            
        Returns:
            Parsed datetime object
        """
        if isinstance(ts, datetime):
            return ts
        
        if isinstance(ts, (int, float)):
            # Assume milliseconds if large number
            if ts > 1e10:
                return datetime.fromtimestamp(ts / 1000, UTC)
            else:
                return datetime.fromtimestamp(ts, UTC)
        
        if isinstance(ts, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                pass
        
        # Default to current time
        return datetime.now(UTC)


# Integration with BluefinMarketDataProvider
async def integrate_websocket_with_provider(
    provider: Any,
    symbol: str,
    interval: str
) -> BluefinWebSocketClient:
    """
    Create and integrate WebSocket client with BluefinMarketDataProvider.
    
    Args:
        provider: BluefinMarketDataProvider instance
        symbol: Trading symbol
        interval: Candle interval
        
    Returns:
        Connected BluefinWebSocketClient instance
    """
    # Create WebSocket client with callback to update provider
    async def on_candle_update(candle: MarketData):
        # Update provider's cache
        provider._ohlcv_cache.append(candle)
        if len(provider._ohlcv_cache) > provider.candle_limit:
            provider._ohlcv_cache = provider._ohlcv_cache[-provider.candle_limit:]
        
        provider._last_update = datetime.now(UTC)
        provider._cache_timestamps["ohlcv"] = provider._last_update
        
        # Update price cache
        provider._price_cache["price"] = candle.close
        provider._cache_timestamps["price"] = datetime.now(UTC)
        
        # Notify provider's subscribers
        await provider._notify_subscribers(candle)
    
    # Create and connect WebSocket client
    ws_client = BluefinWebSocketClient(
        symbol=symbol,
        interval=interval,
        candle_limit=provider.candle_limit,
        on_candle_update=on_candle_update
    )
    
    await ws_client.connect()
    
    # Update provider's initial cache with WebSocket data
    candles = ws_client.get_candles()
    if candles:
        provider._ohlcv_cache = candles[-provider.candle_limit:]
        provider._last_update = datetime.now(UTC)
        
        # Update price from latest candle
        latest_price = ws_client.get_latest_price()
        if latest_price:
            provider._price_cache["price"] = latest_price
            provider._cache_timestamps["price"] = datetime.now(UTC)
    
    return ws_client