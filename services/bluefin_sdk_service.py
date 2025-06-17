#!/usr/bin/env python3
"""
Bluefin SDK Service - REST API wrapper for the Bluefin SDK.

This service runs in an isolated Docker container with the Bluefin SDK installed,
providing a REST API for the main bot to interact with Bluefin DEX.
"""

import asyncio
import logging
import os
import sys
import time
import secrets
from decimal import Decimal
from typing import Any, Dict, List, Optional
from collections import defaultdict

from aiohttp import web
from bluefin_v2_client import BluefinClient, MARKET_SYMBOLS, Networks

# Add parent directory to path to import secure logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bot.utils.secure_logging import create_secure_logger
    logger = create_secure_logger(__name__, level=logging.INFO)
except ImportError:
    # Fallback to standard logging if secure logging not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


class BluefinSDKService:
    """
    REST API service wrapping the Bluefin SDK.
    """
    
    def __init__(self):
        self.client: Optional[BluefinClient] = None
        self.private_key = os.getenv("BLUEFIN_PRIVATE_KEY")
        self.network = os.getenv("BLUEFIN_NETWORK", "mainnet")
        self.initialized = False
    
    def _get_market_symbol(self, symbol: str):
        """
        Convert symbol string to MARKET_SYMBOLS attribute.
        
        Args:
            symbol: Symbol string like "BTC-PERP", "ETH-PERP", "SUI-PERP"
            
        Returns:
            The corresponding MARKET_SYMBOLS attribute
        """
        # Convert symbol format: "BTC-PERP" -> "BTC_PERP"
        symbol_attr = symbol.replace("-", "_")
        
        # Get the attribute from MARKET_SYMBOLS
        if hasattr(MARKET_SYMBOLS, symbol_attr):
            return getattr(MARKET_SYMBOLS, symbol_attr)
        else:
            # If exact match not found, try without PERP suffix
            base_symbol = symbol_attr.replace("_PERP", "")
            if hasattr(MARKET_SYMBOLS, base_symbol):
                return getattr(MARKET_SYMBOLS, base_symbol)
            else:
                raise ValueError(f"Unknown market symbol: {symbol}")
        
    async def initialize(self):
        """Initialize the Bluefin SDK client."""
        try:
            if not self.private_key:
                logger.error("BLUEFIN_PRIVATE_KEY not set")
                return False
                
            # Determine network
            network = Networks["SUI_PROD"] if self.network == "mainnet" else Networks["SUI_STAGING"]
            
            # Initialize Bluefin client
            logger.info(f"Initializing Bluefin client on {self.network}")
            self.client = BluefinClient(
                True,  # On-chain transactions
                network,
                self.private_key
            )
            
            # Initialize the client
            await self.client.init(True)
            
            
            self.initialized = True
            logger.info("Bluefin SDK initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Bluefin SDK: {e}")
            return False
    
    async def get_account_data(self) -> Dict[str, Any]:
        """Get account data including balances."""
        if not self.initialized:
            return {"error": "SDK not initialized"}
            
        try:
            # Get account balances
            balance = await self.client.get_usdc_balance()
            
            # Get margin info
            margin_info = await self.client.get_margin_bank_balance()
            
            return {
                "balance": str(balance),
                "margin": margin_info,
                "address": self.client.get_public_address()
            }
            
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return {"error": str(e)}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        if not self.initialized:
            return []
            
        try:
            # Get user position (returns a single position object)
            position = await self.client.get_user_position({})
            
            # Check if we got data
            if not position:
                return []
            
            # If it's a list of positions
            if isinstance(position, list):
                formatted_positions = []
                for pos in position:
                    if pos:  # Skip None/empty entries
                        # Log position symbol only for debugging
                        logger.debug(f"Processing position for symbol: {pos.get('symbol', 'Unknown')}")
                        
                        # Check if there's a direct side field
                        if "side" in pos:
                            side = pos["side"]
                            # Map Bluefin side to our convention
                            if side == "SELL":
                                side = "SHORT"
                            elif side == "BUY":
                                side = "LONG"
                        elif "positionSide" in pos:
                            side = pos["positionSide"]
                        else:
                            # For Bluefin, negative quantity = SHORT, positive = LONG
                            quantity = float(pos.get("quantity", 0))
                            side = "LONG" if quantity > 0 else "SHORT"
                        
                        formatted_positions.append({
                            "symbol": pos.get("symbol", ""),
                            "quantity": pos.get("quantity", "0"),
                            "avgPrice": pos.get("avgEntryPrice", pos.get("avgPrice", "0")),
                            "unrealizedPnl": pos.get("unrealizedPnl", "0"),
                            "realizedPnl": pos.get("realizedPnl", "0"),
                            "leverage": pos.get("leverage", 1),
                            "margin": pos.get("margin", "0"),
                            "side": side
                        })
                return formatted_positions
            # If it's a single position dict
            elif isinstance(position, dict):
                # Log position symbol only for debugging
                logger.debug(f"Processing position for symbol: {position.get('symbol', 'Unknown')}")
                
                # Check if there's a direct side field
                if "side" in position:
                    side = position["side"]
                    # Map Bluefin side to our convention
                    if side == "SELL":
                        side = "SHORT"
                    elif side == "BUY":
                        side = "LONG"
                elif "positionSide" in position:
                    side = position["positionSide"]
                else:
                    # For Bluefin, negative quantity = SHORT, positive = LONG
                    quantity = float(position.get("quantity", 0))
                    side = "LONG" if quantity > 0 else "SHORT"
                
                return [{
                    "symbol": position.get("symbol", ""),
                    "quantity": position.get("quantity", "0"),
                    "avgPrice": position.get("avgEntryPrice", position.get("avgPrice", "0")),
                    "unrealizedPnl": position.get("unrealizedPnl", "0"),
                    "realizedPnl": position.get("realizedPnl", "0"),
                    "leverage": position.get("leverage", 1),
                    "margin": position.get("margin", "0"),
                    "side": side
                }]
            else:
                logger.warning(f"Unexpected position type: {type(position)}")
                return []
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order."""
        if not self.initialized:
            return {"status": "error", "message": "SDK not initialized"}
            
        try:
            symbol = order_data["symbol"]
            side = order_data["side"]
            quantity = float(order_data["quantity"])
            order_type = order_data.get("orderType", "MARKET")
            
            if order_type == "MARKET":
                # Place market order
                response = await self.client.place_market_order(
                    symbol=self._get_market_symbol(symbol),
                    side=side,
                    quantity=quantity,
                    reduce_only=order_data.get("reduceOnly", False)
                )
            else:
                # Place limit order
                price = float(order_data["price"])
                response = await self.client.place_limit_order(
                    symbol=self._get_market_symbol(symbol),
                    side=side,
                    price=price,
                    quantity=quantity,
                    reduce_only=order_data.get("reduceOnly", False)
                )
                
            return {
                "status": "success",
                "order": response
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.initialized:
            return False
            
        try:
            await self.client.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self.initialized:
            return False
            
        try:
            await self.client.adjust_leverage(
                symbol=self._get_market_symbol(symbol),
                leverage=leverage
            )
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False


# Global service instance
service = BluefinSDKService()

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("BLUEFIN_SERVICE_RATE_LIMIT", "100"))  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds
rate_limit_storage = defaultdict(lambda: {'count': 0, 'window_start': time.time()})


# Authentication middleware
@web.middleware
async def auth_middleware(request, handler):
    """Validate API key authentication."""
    # Skip auth for health check endpoint
    if request.path == '/health':
        return await handler(request)
    
    # Get API key from environment
    expected_api_key = os.getenv("BLUEFIN_SERVICE_API_KEY")
    if not expected_api_key:
        logger.error("BLUEFIN_SERVICE_API_KEY not configured")
        return web.json_response(
            {"error": "Service misconfigured"},
            status=500
        )
    
    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning(f"Missing or invalid auth header from {request.remote}")
        return web.json_response(
            {"error": "Unauthorized"},
            status=401
        )
    
    # Extract and validate token
    provided_api_key = auth_header[7:]  # Remove "Bearer " prefix
    if not secrets.compare_digest(provided_api_key, expected_api_key):
        logger.warning(f"Invalid API key attempt from {request.remote}")
        return web.json_response(
            {"error": "Unauthorized"},
            status=401
        )
    
    # API key is valid, proceed to handler
    return await handler(request)


# Rate limiting middleware
@web.middleware
async def rate_limit_middleware(request, handler):
    """Apply rate limiting per client IP."""
    # Skip rate limiting for health check
    if request.path == '/health':
        return await handler(request)
    
    # Get client IP
    client_ip = request.headers.get('X-Forwarded-For', request.remote)
    if client_ip:
        client_ip = client_ip.split(',')[0].strip()
    
    # Check rate limit
    current_time = time.time()
    client_data = rate_limit_storage[client_ip]
    
    # Reset window if expired
    if current_time - client_data['window_start'] > RATE_LIMIT_WINDOW:
        client_data['count'] = 0
        client_data['window_start'] = current_time
    
    # Check if limit exceeded
    if client_data['count'] >= RATE_LIMIT_REQUESTS:
        remaining_time = int(RATE_LIMIT_WINDOW - (current_time - client_data['window_start']))
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return web.json_response(
            {
                "error": "Rate limit exceeded",
                "retry_after": remaining_time
            },
            status=429,
            headers={"Retry-After": str(remaining_time)}
        )
    
    # Increment counter
    client_data['count'] += 1
    
    # Add rate limit headers to response
    response = await handler(request)
    response.headers['X-RateLimit-Limit'] = str(RATE_LIMIT_REQUESTS)
    response.headers['X-RateLimit-Remaining'] = str(RATE_LIMIT_REQUESTS - client_data['count'])
    response.headers['X-RateLimit-Reset'] = str(int(client_data['window_start'] + RATE_LIMIT_WINDOW))
    
    return response


# REST API Routes
async def health_check(request):
    """Health check endpoint."""
    return web.json_response({
        "status": "healthy" if service.initialized else "unhealthy",
        "initialized": service.initialized,
        "network": service.network
    })


async def get_account(request):
    """Get account data."""
    try:
        data = await service.get_account_data()
        return web.json_response(data)
    except Exception as e:
        logger.error(f"Error in get_account: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_positions(request):
    """Get positions."""
    try:
        positions = await service.get_positions()
        return web.json_response({"positions": positions})
    except Exception as e:
        logger.error(f"Error in get_positions: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def place_order(request):
    """Place an order."""
    try:
        order_data = await request.json()
        result = await service.place_order(order_data)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Error in place_order: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def cancel_order(request):
    """Cancel an order."""
    try:
        order_id = request.match_info['order_id']
        success = await service.cancel_order(order_id)
        return web.json_response({"success": success})
    except Exception as e:
        logger.error(f"Error in cancel_order: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_ticker(request):
    """Get market ticker."""
    try:
        symbol = request.query.get('symbol', 'SUI-PERP')
        
        # Get orderbook for best bid/ask
        orderbook = await service.client.get_orderbook(service._get_market_symbol(symbol))
        
        best_bid = orderbook['bids'][0]['price'] if orderbook['bids'] else 0
        best_ask = orderbook['asks'][0]['price'] if orderbook['asks'] else 0
        
        return web.json_response({
            "symbol": symbol,
            "price": str((float(best_bid) + float(best_ask)) / 2),
            "bestBid": str(best_bid),
            "bestAsk": str(best_ask)
        })
    except Exception as e:
        logger.error(f"Error in get_ticker: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def set_leverage(request):
    """Set leverage."""
    try:
        data = await request.json()
        success = await service.set_leverage(data['symbol'], data['leverage'])
        return web.json_response({"success": success})
    except Exception as e:
        logger.error(f"Error in set_leverage: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def startup(app):
    """Initialize service on startup."""
    await service.initialize()


def create_app():
    """Create the aiohttp application."""
    # Create app with middleware
    app = web.Application(
        middlewares=[rate_limit_middleware, auth_middleware]
    )
    
    # Add routes
    app.router.add_get('/health', health_check)
    app.router.add_get('/account', get_account)
    app.router.add_get('/positions', get_positions)
    app.router.add_post('/orders', place_order)
    app.router.add_delete('/orders/{order_id}', cancel_order)
    app.router.add_get('/market/ticker', get_ticker)
    app.router.add_post('/leverage', set_leverage)
    
    # Add startup handler
    app.on_startup.append(startup)
    
    return app


if __name__ == '__main__':
    # Get host and port from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8080))
    
    # Check if API key is configured
    if not os.getenv("BLUEFIN_SERVICE_API_KEY"):
        logger.warning("BLUEFIN_SERVICE_API_KEY not set - generating a random key")
        api_key = secrets.token_urlsafe(32)
        os.environ["BLUEFIN_SERVICE_API_KEY"] = api_key
        logger.info(f"Generated API key: {api_key}")
        logger.info("Please set BLUEFIN_SERVICE_API_KEY in your environment for production use")
    
    # Create and run app
    app = create_app()
    logger.info(f"Starting Bluefin SDK service on {host}:{port}")
    logger.info(f"Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds")
    web.run_app(app, host=host, port=port)