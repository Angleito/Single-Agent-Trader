#!/usr/bin/env python3
"""
Bluefin SDK Service - Isolated microservice for Bluefin DEX operations
Provides REST API interface for the main trading bot to interact with Bluefin
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import random
import math
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

import uvicorn
from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Bluefin SDK
try:
    from bluefin_v2_client import BluefinClient, Networks
    BLUEFIN_AVAILABLE = True
    logger.info("Bluefin SDK imported successfully")
except ImportError as e:
    BLUEFIN_AVAILABLE = False
    logger.warning(f"Bluefin SDK not available: {e}")
    BluefinClient = None
    Networks = None

# Pydantic models
class OrderRequest(BaseModel):
    symbol: str
    side: str  # "BUY" or "SELL"
    size: float
    price: Optional[float] = None
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    leverage: Optional[int] = None

class OrderResponse(BaseModel):
    success: bool
    order_id: Optional[str] = None
    message: str
    data: Optional[Dict] = None

class PositionResponse(BaseModel):
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    pnl: float
    margin: float

class MarketDataRequest(BaseModel):
    symbol: str
    interval: str = "5m"
    limit: int = 500

class CandleData(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

class HealthResponse(BaseModel):
    status: str
    bluefin_available: bool
    connected: bool
    timestamp: str

class WebSocketMessage(BaseModel):
    type: str  # "price_update", "trade", "heartbeat", "error"
    symbol: str
    timestamp: float
    data: Dict = {}

class PriceUpdateData(BaseModel):
    price: float
    volume: float = 0.0
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None

class TradeData(BaseModel):
    price: float
    size: float
    side: str  # "buy" or "sell"
    trade_id: Optional[str] = None

class WebSocketConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_symbols: Dict[WebSocket, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, symbols: List[str]):
        """Connect a WebSocket client and subscribe to symbols"""
        await websocket.accept()
        
        # Track connection and its subscribed symbols
        self.connection_symbols[websocket] = set(symbols)
        
        # Add connection to each symbol's subscriber list
        for symbol in symbols:
            if symbol not in self.active_connections:
                self.active_connections[symbol] = set()
            self.active_connections[symbol].add(websocket)
        
        logger.info(f"WebSocket connected for symbols: {symbols}")
        
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.connection_symbols:
            symbols = self.connection_symbols[websocket]
            
            # Remove from all symbol subscriber lists
            for symbol in symbols:
                if symbol in self.active_connections:
                    self.active_connections[symbol].discard(websocket)
                    if not self.active_connections[symbol]:
                        del self.active_connections[symbol]
            
            # Remove connection tracking
            del self.connection_symbols[websocket]
            
        logger.info("WebSocket disconnected")
    
    async def broadcast_to_symbol(self, symbol: str, message: Dict):
        """Broadcast message to all connections subscribed to a symbol"""
        if symbol in self.active_connections:
            disconnected = set()
            
            for connection in self.active_connections[symbol].copy():
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send message to WebSocket: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected connections
            for connection in disconnected:
                self.disconnect(connection)
    
    def get_subscriber_count(self, symbol: str) -> int:
        """Get number of subscribers for a symbol"""
        return len(self.active_connections.get(symbol, set()))
    
    def get_total_connections(self) -> int:
        """Get total number of active connections"""
        return len(self.connection_symbols)

class RealtimeDataGenerator:
    """Generates realistic real-time market data for testing"""
    
    def __init__(self):
        self.base_prices = {
            "SUI-PERP": 3.50,
            "ETH-PERP": 2500.0,
            "BTC-PERP": 45000.0,
            "SOL-PERP": 150.0,
        }
        self.current_prices = self.base_prices.copy()
        self.last_update = {}
        self.volatility = 0.001  # 0.1% volatility per update
        
    def generate_price_update(self, symbol: str) -> Dict:
        """Generate realistic price movement"""
        current_price = self.current_prices.get(symbol, self.base_prices.get(symbol, 100.0))
        
        # Generate price movement with some momentum
        change_pct = random.uniform(-self.volatility, self.volatility)
        
        # Add some momentum - prices tend to continue in the same direction
        if symbol in self.last_update:
            last_change = self.last_update[symbol].get('change_pct', 0)
            momentum = last_change * 0.3  # 30% momentum effect
            change_pct += momentum
        
        new_price = current_price * (1 + change_pct)
        
        # Ensure price doesn't go negative or too extreme
        base_price = self.base_prices.get(symbol, 100.0)
        new_price = max(new_price, base_price * 0.5)
        new_price = min(new_price, base_price * 2.0)
        
        self.current_prices[symbol] = new_price
        
        # Generate volume (higher volume during price movements)
        base_volume = 1000.0
        volume_multiplier = 1 + abs(change_pct) * 50  # Higher volume on bigger moves
        volume = base_volume * volume_multiplier * random.uniform(0.5, 2.0)
        
        update_data = {
            'price': round(new_price, 6),
            'volume': round(volume, 2),
            'change_pct': change_pct,
            'timestamp': time.time()
        }
        
        self.last_update[symbol] = update_data
        return update_data
    
    def generate_trade_data(self, symbol: str) -> Dict:
        """Generate realistic trade data"""
        current_price = self.current_prices.get(symbol, self.base_prices.get(symbol, 100.0))
        
        # Trade price varies slightly from current price
        price_variation = random.uniform(-0.0005, 0.0005)  # 0.05% variation
        trade_price = current_price * (1 + price_variation)
        
        # Trade size
        trade_size = random.uniform(0.1, 10.0)
        
        # Trade side (buy/sell) - slightly biased towards the current price trend
        side_bias = 0.5
        if symbol in self.last_update:
            last_change = self.last_update[symbol].get('change_pct', 0)
            if last_change > 0:
                side_bias = 0.6  # More buys when price is going up
            elif last_change < 0:
                side_bias = 0.4  # More sells when price is going down
        
        side = "buy" if random.random() < side_bias else "sell"
        
        return {
            'price': round(trade_price, 6),
            'size': round(trade_size, 4),
            'side': side,
            'trade_id': f"trade_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            'timestamp': time.time()
        }

class BluefinService:
    """Bluefin DEX service wrapper"""
    
    def __init__(self):
        self.client: Optional[BluefinClient] = None
        self.connected = False
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.private_key = os.getenv("BLUEFIN_PRIVATE_KEY")
        self.network = os.getenv("BLUEFIN_NETWORK", "mainnet")
        
        # WebSocket streaming components
        self.ws_manager = WebSocketConnectionManager()
        self.data_generator = RealtimeDataGenerator()
        self.streaming_task: Optional[asyncio.Task] = None
        self.streaming_active = False
        self.stream_interval = float(os.getenv("STREAM_INTERVAL_SECONDS", "1.0"))  # Default 1 second
        self.trade_frequency = float(os.getenv("TRADE_FREQUENCY", "5.0"))  # Trade every 5 seconds on average
        
    async def initialize(self):
        """Initialize Bluefin client"""
        if not BLUEFIN_AVAILABLE:
            logger.warning("Bluefin SDK not available - running in simulation mode")
            return
            
        if not self.private_key:
            logger.warning("No private key provided - running in simulation mode")
            return
            
        try:
            # Determine network
            network = Networks.TESTNET if self.network == "testnet" else Networks.MAINNET
            
            # Initialize client
            self.client = BluefinClient(
                True,  # Are you deploying on mainnet?
                network,
                self.private_key
            )
            
            # Test connection
            await self.client.init(True)
            self.connected = True
            
            logger.info(f"Connected to Bluefin {self.network} network")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bluefin client: {e}")
            self.connected = False
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected or self.dry_run:
            return {
                "equity": 10000.0,
                "margin_used": 0.0,
                "available_margin": 10000.0,
                "unrealized_pnl": 0.0
            }
            
        try:
            account = await self.client.get_account_data()
            return {
                "equity": float(account.get("accountValue", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "available_margin": float(account.get("availableMargin", 0)),
                "unrealized_pnl": float(account.get("unrealizedPnl", 0))
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_positions(self) -> List[PositionResponse]:
        """Get open positions"""
        if not self.connected or self.dry_run:
            return []  # No positions in simulation mode
            
        try:
            positions = await self.client.get_user_positions()
            result = []
            
            for pos in positions:
                if float(pos.get("size", 0)) != 0:  # Only return non-zero positions
                    result.append(PositionResponse(
                        symbol=pos.get("market", ""),
                        side="LONG" if float(pos.get("size", 0)) > 0 else "SHORT",
                        size=abs(float(pos.get("size", 0))),
                        entry_price=float(pos.get("entryPrice", 0)),
                        mark_price=float(pos.get("markPrice", 0)),
                        pnl=float(pos.get("unrealizedPnl", 0)),
                        margin=float(pos.get("margin", 0))
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place an order"""
        if self.dry_run:
            # Simulate order placement
            order_id = f"PAPER_{int(datetime.now().timestamp())}"
            logger.info(f"PAPER TRADE: {order.side} {order.size} {order.symbol} at {order.price or 'MARKET'}")
            
            return OrderResponse(
                success=True,
                order_id=order_id,
                message="Paper trade executed successfully",
                data={
                    "symbol": order.symbol,
                    "side": order.side,
                    "size": order.size,
                    "price": order.price,
                    "type": order.order_type
                }
            )
        
        if not self.connected:
            return OrderResponse(
                success=False,
                message="Not connected to Bluefin DEX"
            )
        
        try:
            # Convert order to Bluefin format
            bluefin_order = {
                "symbol": order.symbol,
                "side": order.side.upper(),
                "orderType": order.order_type.upper(),
                "quantity": str(order.size),
                "timeInForce": "IOC"  # Immediate or Cancel
            }
            
            if order.price:
                bluefin_order["price"] = str(order.price)
            
            if order.leverage:
                bluefin_order["leverage"] = str(order.leverage)
            
            # Place order
            response = await self.client.post_order(bluefin_order)
            
            return OrderResponse(
                success=True,
                order_id=response.get("id"),
                message="Order placed successfully",
                data=response
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResponse(
                success=False,
                message=f"Failed to place order: {str(e)}"
            )
    
    async def get_market_data(self, symbol: str, interval: str = "5m", limit: int = 500) -> List[CandleData]:
        """Get historical market data"""
        if self.dry_run or not self.connected:
            # Generate realistic mock data
            import time
            import random
            import math
            
            now = int(time.time())
            
            # Set realistic base prices by symbol
            base_prices = {
                "SUI-PERP": 2.5,
                "ETH-PERP": 2500.0,
                "BTC-PERP": 45000.0,
                "SOL-PERP": 150.0,
            }
            base_price = base_prices.get(symbol, 2.5)
            
            candles = []
            
            # Convert interval to seconds
            interval_seconds = {
                "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "4h": 14400, "1d": 86400
            }.get(interval, 300)
            
            for i in range(limit):
                timestamp = now - (limit - i) * interval_seconds
                
                # More realistic price movement (smaller volatility)
                volatility = 0.002  # 0.2% volatility per candle
                price_change = random.uniform(-volatility, volatility)
                
                # Generate OHLC with more realistic behavior
                open_price = base_price
                close_price = open_price * (1 + price_change)
                
                # High and low with smaller ranges
                spread = abs(close_price - open_price)
                high_price = max(open_price, close_price) + spread * random.uniform(0, 0.5)
                low_price = min(open_price, close_price) - spread * random.uniform(0, 0.5)
                
                # Ensure realistic OHLC relationships
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Volume with some variation
                volume = random.uniform(500, 5000)
                
                candles.append(CandleData(
                    timestamp=timestamp,
                    open=round(open_price, 6),
                    high=round(high_price, 6),
                    low=round(low_price, 6),
                    close=round(close_price, 6),
                    volume=round(volume, 2)
                ))
                
                # Update base price for next candle (smooth progression)
                base_price = close_price
            
            return candles
        
        try:
            # Get real market data from Bluefin
            response = await self.client.get_candlestick_data({
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            })
            
            candles = []
            for candle in response:
                candles.append(CandleData(
                    timestamp=int(candle[0]),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                ))
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def start_streaming(self):
        """Start the real-time data streaming task"""
        if not self.streaming_active:
            self.streaming_active = True
            self.streaming_task = asyncio.create_task(self._streaming_loop())
            logger.info(f"Started real-time data streaming (interval: {self.stream_interval}s)")
    
    async def stop_streaming(self):
        """Stop the real-time data streaming task"""
        self.streaming_active = False
        if self.streaming_task and not self.streaming_task.done():
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped real-time data streaming")
    
    async def _streaming_loop(self):
        """Main streaming loop that generates and broadcasts real-time data"""
        last_trade_time = {}
        
        try:
            while self.streaming_active:
                current_time = time.time()
                
                # Get all symbols that have active subscribers
                active_symbols = list(self.ws_manager.active_connections.keys())
                
                for symbol in active_symbols:
                    subscriber_count = self.ws_manager.get_subscriber_count(symbol)
                    if subscriber_count > 0:
                        # Generate and broadcast price update
                        price_data = self.data_generator.generate_price_update(symbol)
                        
                        price_message = {
                            "type": "price_update",
                            "symbol": symbol,
                            "timestamp": price_data['timestamp'],
                            "data": {
                                "price": price_data['price'],
                                "volume": price_data['volume']
                            }
                        }
                        
                        await self.ws_manager.broadcast_to_symbol(symbol, price_message)
                        
                        # Generate trade data less frequently
                        last_trade = last_trade_time.get(symbol, 0)
                        if current_time - last_trade >= self.trade_frequency * random.uniform(0.5, 1.5):
                            trade_data = self.data_generator.generate_trade_data(symbol)
                            
                            trade_message = {
                                "type": "trade",
                                "symbol": symbol,
                                "timestamp": trade_data['timestamp'],
                                "data": {
                                    "price": trade_data['price'],
                                    "size": trade_data['size'],
                                    "side": trade_data['side'],
                                    "trade_id": trade_data['trade_id']
                                }
                            }
                            
                            await self.ws_manager.broadcast_to_symbol(symbol, trade_message)
                            last_trade_time[symbol] = current_time
                
                # Send heartbeat every 30 seconds
                if int(current_time) % 30 == 0:
                    heartbeat_message = {
                        "type": "heartbeat",
                        "symbol": "system",
                        "timestamp": current_time,
                        "data": {
                            "connections": self.ws_manager.get_total_connections(),
                            "active_symbols": len(active_symbols)
                        }
                    }
                    
                    # Broadcast heartbeat to all connections
                    for symbol in active_symbols:
                        await self.ws_manager.broadcast_to_symbol(symbol, heartbeat_message)
                
                # Wait for next update
                await asyncio.sleep(self.stream_interval)
                
        except asyncio.CancelledError:
            logger.info("Streaming loop cancelled")
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            self.streaming_active = False
    
    async def handle_websocket_connection(self, websocket: WebSocket, symbols: List[str]):
        """Handle a new WebSocket connection"""
        await self.ws_manager.connect(websocket, symbols)
        
        # Start streaming if this is the first connection
        if not self.streaming_active and self.ws_manager.get_total_connections() > 0:
            await self.start_streaming()
        
        try:
            # Keep connection alive and handle client messages
            while True:
                try:
                    # Wait for client messages (ping, subscription changes, etc.)
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle client commands
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": time.time()
                        })
                    elif message.get("type") == "subscribe":
                        # Handle additional symbol subscriptions
                        new_symbols = message.get("symbols", [])
                        if new_symbols:
                            # Update subscription
                            current_symbols = list(self.ws_manager.connection_symbols.get(websocket, set()))
                            combined_symbols = current_symbols + new_symbols
                            
                            # Disconnect and reconnect with new symbols
                            self.ws_manager.disconnect(websocket)
                            await self.ws_manager.connect(websocket, combined_symbols)
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"Error handling WebSocket message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    })
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            # Clean up connection
            self.ws_manager.disconnect(websocket)
            
            # Stop streaming if no more connections
            if self.ws_manager.get_total_connections() == 0:
                await self.stop_streaming()
    
    def get_streaming_status(self) -> Dict:
        """Get current streaming status"""
        return {
            "streaming_active": self.streaming_active,
            "total_connections": self.ws_manager.get_total_connections(),
            "active_symbols": list(self.ws_manager.active_connections.keys()),
            "stream_interval": self.stream_interval,
            "trade_frequency": self.trade_frequency
        }

# Global service instance
bluefin_service = BluefinService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Bluefin service...")
    await bluefin_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Bluefin service...")
    await bluefin_service.stop_streaming()

# Create FastAPI app
app = FastAPI(
    title="Bluefin DEX Service",
    description="Microservice for Bluefin DEX operations",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        bluefin_available=BLUEFIN_AVAILABLE,
        connected=bluefin_service.connected,
        timestamp=datetime.now().isoformat()
    )

@app.get("/account")
async def get_account():
    """Get account information"""
    return await bluefin_service.get_account_info()

@app.get("/positions", response_model=List[PositionResponse])
async def get_positions():
    """Get open positions"""
    return await bluefin_service.get_positions()

@app.post("/orders", response_model=OrderResponse)
async def place_order(order: OrderRequest):
    """Place an order"""
    return await bluefin_service.place_order(order)

@app.get("/market_data/{symbol}", response_model=List[CandleData])
async def get_market_data(
    symbol: str,
    interval: str = "5m",
    limit: int = 500
):
    """Get historical market data"""
    return await bluefin_service.get_market_data(symbol, interval, limit)

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data streams"""
    symbols = [symbol]  # Single symbol subscription
    await bluefin_service.handle_websocket_connection(websocket, symbols)

@app.websocket("/ws/multi/{symbols}")
async def websocket_multi_endpoint(websocket: WebSocket, symbols: str):
    """WebSocket endpoint for multiple symbol subscriptions"""
    # Parse comma-separated symbols
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="No symbols provided")
        return
    
    await bluefin_service.handle_websocket_connection(websocket, symbol_list)

@app.get("/streaming/status")
async def get_streaming_status():
    """Get current WebSocket streaming status"""
    return bluefin_service.get_streaming_status()

@app.post("/streaming/start")
async def start_streaming():
    """Manually start streaming (for testing)"""
    await bluefin_service.start_streaming()
    return {"message": "Streaming started", "status": bluefin_service.get_streaming_status()}

@app.post("/streaming/stop")
async def stop_streaming():
    """Manually stop streaming (for testing)"""
    await bluefin_service.stop_streaming()
    return {"message": "Streaming stopped", "status": bluefin_service.get_streaming_status()}

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting Bluefin service on {host}:{port}")
    
    # Run the service
    uvicorn.run(
        "bluefin_service:app",
        host=host,
        port=port,
        log_level=log_level.lower(),  # Convert to lowercase
        reload=False
    )