"""
Bluefin Client - HTTP client to communicate with Bluefin microservice
This replaces direct SDK usage with HTTP calls to the isolated Bluefin service
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
import aiohttp
import json
import websockets
from collections.abc import Coroutine

from ..config import settings

logger = logging.getLogger(__name__)

class BluefinServiceClient:
    """Client for communicating with Bluefin microservice"""
    
    def __init__(self, config=None):
        self.config = config or settings
        self.base_url = getattr(self.config.exchange, 'bluefin_service_url', "http://bluefin-service:8080")
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False
        
        # WebSocket components
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.ws_task: Optional[asyncio.Task] = None
        self.ws_connected = False
        self.subscribers: List[Callable] = []
        self.subscribed_symbols: List[str] = []
        
        # Reconnection settings
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 60.0  # Max 60 seconds
        self.current_reconnect_attempts = 0
        
        # Data caching for real-time updates
        self.latest_prices: Dict[str, float] = {}
        self.latest_trades: Dict[str, Dict] = {}
        self.price_update_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to Bluefin service"""
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")
    
    async def connect(self) -> bool:
        """Connect to Bluefin service"""
        try:
            health = await self._request("GET", "/health")
            self.connected = health.get("status") == "healthy"
            
            if self.connected:
                logger.info(f"Connected to Bluefin service at {self.base_url}")
                logger.info(f"Bluefin SDK available: {health.get('bluefin_available', False)}")
                logger.info(f"Bluefin DEX connected: {health.get('connected', False)}")
            else:
                logger.warning("Bluefin service is not healthy")
                
            return self.connected
            
        except Exception as e:
            logger.error(f"Failed to connect to Bluefin service: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from service"""
        # Disconnect WebSocket first
        await self.disconnect_websocket()
        
        if self.session and not self.session.closed:
            await self.session.close()
        self.connected = False
        logger.info("Disconnected from Bluefin service")
    
    async def get_account_data(self) -> Dict:
        """Get account information"""
        try:
            return await self._request("GET", "/account")
        except Exception as e:
            logger.error(f"Failed to get account data: {e}")
            return {"balance": "10000", "address": "mock_address"}
    
    async def get_user_positions(self) -> List[Dict]:
        """Get user positions"""
        try:
            positions = await self._request("GET", "/positions")
            return positions
        except Exception as e:
            logger.error(f"Failed to get user positions: {e}")
            return []
    
    async def place_order(self, order_data: Dict) -> Dict:
        """Place an order"""
        try:
            return await self._request("POST", "/orders", json=order_data)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_candlestick_data(self, params: Dict) -> List[List]:
        """Get historical candlestick data"""
        symbol = params.get("symbol", "SUI-PERP")
        interval = params.get("interval", "5m")
        limit = params.get("limit", 500)
        
        try:
            candles = await self._request(
                "GET", 
                f"/market_data/{symbol}",
                params={"interval": interval, "limit": limit}
            )
            
            # Convert to format expected by the bot
            result = []
            for candle in candles:
                result.append([
                    candle["timestamp"],  # Keep timestamp as-is (seconds)
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"]
                ])
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get candlestick data: {e}")
            # Return empty list if service is unavailable
            return []
    
    async def get_market_ticker(self, symbol: str) -> Dict:
        """Get market ticker data"""
        try:
            # For now, get latest price from candlestick data
            candles = await self.get_candlestick_data({
                "symbol": symbol,
                "interval": "1m",
                "limit": 1
            })
            
            if candles:
                latest = candles[0]
                return {
                    "symbol": symbol,
                    "price": latest[4],  # Close price
                    "timestamp": latest[0]
                }
            
            # Fallback to mock price if no candles available
            base_prices = {
                "ETH-PERP": 2500.0,
                "BTC-PERP": 45000.0,
                "SUI-PERP": 3.50,
                "SOL-PERP": 150.0,
            }
            mock_price = base_prices.get(symbol, 2500.0)
            
            return {
                "symbol": symbol, 
                "price": mock_price, 
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Failed to get market ticker for {symbol}: {e}")
            return {"symbol": symbol, "price": 0.0, "timestamp": int(datetime.now().timestamp() * 1000)}
    
    async def connect_websocket(self, symbols: List[str]) -> bool:
        """Connect to WebSocket for real-time data"""
        if self.ws_connected:
            logger.info("WebSocket already connected")
            return True
        
        self.subscribed_symbols = symbols
        
        try:
            # Build WebSocket URL for multiple symbols
            if len(symbols) == 1:
                ws_url = f"{self.ws_url}/ws/{symbols[0]}"
            else:
                symbol_str = ",".join(symbols)
                ws_url = f"{self.ws_url}/ws/multi/{symbol_str}"
            
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(ws_url)
            self.ws_connected = True
            self.current_reconnect_attempts = 0
            
            # Start WebSocket message handler
            self.ws_task = asyncio.create_task(self._websocket_handler())
            
            logger.info(f"Connected to WebSocket for symbols: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            self.ws_connected = False
            return False
    
    async def disconnect_websocket(self):
        """Disconnect from WebSocket"""
        self.ws_connected = False
        
        if self.ws_task and not self.ws_task.done():
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("Disconnected from WebSocket")
    
    async def _websocket_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.ws_connected = False
            await self._handle_websocket_disconnect()
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            self.ws_connected = False
            await self._handle_websocket_disconnect()
    
    async def _handle_websocket_message(self, data: Dict):
        """Process incoming WebSocket messages"""
        message_type = data.get("type")
        symbol = data.get("symbol")
        timestamp = data.get("timestamp", time.time())
        message_data = data.get("data", {})
        
        if message_type == "price_update":
            # Handle price updates
            price = message_data.get("price")
            volume = message_data.get("volume", 0)
            
            if symbol and price is not None:
                self.latest_prices[symbol] = price
                
                # Notify price update subscribers
                price_update = {
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "timestamp": timestamp
                }
                
                for callback in self.price_update_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(price_update)
                        else:
                            callback(price_update)
                    except Exception as e:
                        logger.error(f"Error in price update callback: {e}")
        
        elif message_type == "trade":
            # Handle trade data
            trade_data = {
                "symbol": symbol,
                "price": message_data.get("price"),
                "size": message_data.get("size"),
                "side": message_data.get("side"),
                "trade_id": message_data.get("trade_id"),
                "timestamp": timestamp
            }
            
            if symbol:
                self.latest_trades[symbol] = trade_data
                
                # Notify trade subscribers
                for callback in self.trade_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(trade_data)
                        else:
                            callback(trade_data)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")
        
        elif message_type == "heartbeat":
            # Handle heartbeat messages
            logger.debug(f"Received heartbeat: {message_data}")
        
        elif message_type == "error":
            # Handle error messages
            error_msg = data.get("message", "Unknown WebSocket error")
            logger.error(f"WebSocket error: {error_msg}")
        
        elif message_type == "pong":
            # Handle pong responses
            logger.debug("Received pong from server")
        
        else:
            logger.debug(f"Unhandled WebSocket message type: {message_type}")
    
    async def _handle_websocket_disconnect(self):
        """Handle WebSocket disconnection and attempt reconnection"""
        if self.current_reconnect_attempts < self.max_reconnect_attempts:
            self.current_reconnect_attempts += 1
            
            # Calculate exponential backoff delay
            delay = min(
                self.reconnect_delay * (2 ** (self.current_reconnect_attempts - 1)),
                self.max_reconnect_delay
            )
            
            logger.info(
                f"Attempting WebSocket reconnection {self.current_reconnect_attempts}/"
                f"{self.max_reconnect_attempts} in {delay}s"
            )
            
            await asyncio.sleep(delay)
            
            # Attempt reconnection
            success = await self.connect_websocket(self.subscribed_symbols)
            if not success:
                await self._handle_websocket_disconnect()  # Try again
        else:
            logger.error("Max WebSocket reconnection attempts reached, giving up")
    
    async def send_ping(self):
        """Send ping to WebSocket server"""
        if self.websocket and self.ws_connected:
            try:
                ping_message = {
                    "type": "ping",
                    "timestamp": time.time()
                }
                await self.websocket.send(json.dumps(ping_message))
            except Exception as e:
                logger.error(f"Failed to send ping: {e}")
    
    def subscribe_to_price_updates(self, callback: Callable):
        """Subscribe to real-time price updates"""
        if callback not in self.price_update_callbacks:
            self.price_update_callbacks.append(callback)
            logger.debug(f"Added price update callback: {callback.__name__}")
    
    def unsubscribe_from_price_updates(self, callback: Callable):
        """Unsubscribe from price updates"""
        if callback in self.price_update_callbacks:
            self.price_update_callbacks.remove(callback)
            logger.debug(f"Removed price update callback: {callback.__name__}")
    
    def subscribe_to_trades(self, callback: Callable):
        """Subscribe to real-time trade updates"""
        if callback not in self.trade_callbacks:
            self.trade_callbacks.append(callback)
            logger.debug(f"Added trade callback: {callback.__name__}")
    
    def unsubscribe_from_trades(self, callback: Callable):
        """Unsubscribe from trade updates"""
        if callback in self.trade_callbacks:
            self.trade_callbacks.remove(callback)
            logger.debug(f"Removed trade callback: {callback.__name__}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest cached price for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_latest_trade(self, symbol: str) -> Optional[Dict]:
        """Get latest cached trade for symbol"""
        return self.latest_trades.get(symbol)
    
    def is_websocket_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.ws_connected and self.websocket is not None
    
    def get_websocket_status(self) -> Dict:
        """Get WebSocket connection status"""
        return {
            "connected": self.ws_connected,
            "subscribed_symbols": self.subscribed_symbols,
            "reconnect_attempts": self.current_reconnect_attempts,
            "price_subscribers": len(self.price_update_callbacks),
            "trade_subscribers": len(self.trade_callbacks),
            "latest_prices": self.latest_prices.copy(),
            "websocket_url": self.ws_url
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # Create a new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    loop.create_task(self.session.close())
                else:
                    # If loop is not running, run cleanup
                    loop.run_until_complete(self.session.close())
            except RuntimeError:
                # If no event loop, create one for cleanup
                asyncio.run(self.session.close())