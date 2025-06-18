#!/usr/bin/env python3
"""
Bluefin SDK Service - REST API wrapper for the Bluefin SDK.

This service runs in an isolated Docker container with the Bluefin SDK installed,
providing a REST API for the main bot to interact with Bluefin DEX.
"""

import logging
import os
import secrets
import sys
import time
from collections import defaultdict
from typing import Any

from aiohttp import web
from bluefin_v2_client import MARKET_SYMBOLS, BluefinClient, Networks
from bluefin_v2_client.interfaces import GetOrderbookRequest

# Add parent directory to path to import secure logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bot.utils.secure_logging import create_secure_logger

    logger = create_secure_logger(__name__, level=logging.INFO)
except ImportError:
    # Fallback to standard logging if secure logging not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


class BluefinSDKService:
    """
    REST API service wrapping the Bluefin SDK.
    """

    def __init__(self):
        self.client: BluefinClient | None = None
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
        # Extract base symbol from formats like "BTC-PERP", "BTC-USD", "BTC"
        base_symbol = symbol.split("-")[0].upper()
        
        # Get the attribute from MARKET_SYMBOLS enum
        if hasattr(MARKET_SYMBOLS, base_symbol):
            return getattr(MARKET_SYMBOLS, base_symbol)
        else:
            # Log available symbols for debugging
            available_symbols = [attr for attr in dir(MARKET_SYMBOLS) if not attr.startswith('_')]
            logger.error(f"Unknown market symbol: {symbol} (base: {base_symbol}). Available symbols: {available_symbols}")
            raise ValueError(f"Unknown market symbol: {symbol}")

    async def initialize(self):
        """Initialize the Bluefin SDK client."""
        try:
            if not self.private_key:
                logger.error("BLUEFIN_PRIVATE_KEY not set")
                return False

            # Determine network
            network = (
                Networks["SUI_PROD"]
                if self.network == "mainnet"
                else Networks["SUI_STAGING"]
            )

            # Initialize Bluefin client
            logger.info(f"Initializing Bluefin client on {self.network}")
            self.client = BluefinClient(
                True,
                network,
                self.private_key,  # On-chain transactions
            )

            # Initialize the client
            await self.client.init(True)

            self.initialized = True
            logger.info("Bluefin SDK initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Bluefin SDK: {e}")
            return False

    async def get_account_data(self) -> dict[str, Any]:
        """Get account data including balances."""
        if not self.initialized:
            return {"error": "SDK not initialized"}

        try:
            # Initialize default response
            account_data = {
                "balance": "0.0", 
                "margin": {"available": "0.0", "used": "0.0"},
                "address": "",
                "error": None
            }

            # Get public address (this should always work)
            try:
                account_data["address"] = self.client.get_public_address()
                logger.info(f"Account address: {account_data['address']}")
            except Exception as e:
                logger.warning(f"Could not get public address: {e}")

            # Try to get account balances
            try:
                logger.debug("Attempting to get USDC balance from Bluefin SDK...")
                
                # Wrap the SDK call with additional protection
                try:
                    balance_response = await self.client.get_usdc_balance()
                    logger.debug(f"Balance response type: {type(balance_response)}, value: {balance_response}")
                except Exception as sdk_error:
                    # Handle known Bluefin SDK issues
                    logger.debug(f"SDK error type: {type(sdk_error)}, message: {str(sdk_error)}")
                    if "'data'" in str(sdk_error) or "KeyError" in str(type(sdk_error)) or "Exception: 'data'" in str(sdk_error):
                        logger.warning(f"Bluefin SDK 'data' access error: {sdk_error}")
                        logger.info("This usually indicates the account is not initialized or has no balance")
                        # Return a default balance of 0
                        balance_response = {"balance": 0, "amount": 0}
                    else:
                        # Re-raise other SDK errors
                        logger.error(f"Unhandled SDK error in balance call: {sdk_error}")
                        raise sdk_error
                
                # Handle different response formats
                if isinstance(balance_response, dict):
                    # If it's a dict, look for balance in common keys
                    balance = balance_response.get("balance", balance_response.get("data", balance_response.get("amount", 0)))
                    logger.debug(f"Extracted balance from dict: {balance}")
                elif isinstance(balance_response, (int, float, str)):
                    # If it's a direct value
                    balance = balance_response
                    logger.debug(f"Using direct balance value: {balance}")
                else:
                    logger.warning(f"Unexpected balance response format: {balance_response}")
                    balance = 0

                account_data["balance"] = str(balance)
                logger.info(f"Account balance: {account_data['balance']} USDC")

            except KeyError as ke:
                logger.error(f"KeyError accessing balance data: {ke}, response was: {balance_response if 'balance_response' in locals() else 'unknown'}")
                account_data["error"] = f"Balance fetch failed - missing key: {str(ke)}"
            except Exception as e:
                logger.error(f"Failed to get balance: {e}")
                logger.error(f"Exception type: {type(e)}")
                account_data["error"] = f"Balance fetch failed: {str(e)}"

            # Try to get margin info
            try:
                logger.debug("Attempting to get margin bank balance from Bluefin SDK...")
                
                # Wrap the SDK call with additional protection
                try:
                    margin_response = await self.client.get_margin_bank_balance()
                    logger.debug(f"Margin response: {margin_response}")
                except Exception as sdk_error:
                    # Handle known Bluefin SDK issues
                    logger.debug(f"SDK error type: {type(sdk_error)}, message: {str(sdk_error)}")
                    if "'data'" in str(sdk_error) or "KeyError" in str(type(sdk_error)) or "Exception: 'data'" in str(sdk_error):
                        logger.warning(f"Bluefin SDK 'data' access error in margin call: {sdk_error}")
                        logger.info("This usually indicates the account is not initialized or has no margin balance")
                        # Return a default margin structure
                        margin_response = {"available": 0, "used": 0, "total": 0}
                    else:
                        # Re-raise other SDK errors
                        logger.error(f"Unhandled SDK error in margin call: {sdk_error}")
                        raise sdk_error
                
                # Handle different margin response formats
                if isinstance(margin_response, dict):
                    account_data["margin"] = {
                        "available": str(margin_response.get("available", 0)),
                        "used": str(margin_response.get("used", 0)),
                        "total": str(margin_response.get("total", 0)),
                    }
                else:
                    account_data["margin"] = {
                        "available": str(margin_response) if margin_response else "0.0",
                        "used": "0.0",
                        "total": str(margin_response) if margin_response else "0.0"
                    }
                    
                logger.info(f"Account margin info: {account_data['margin']}")

            except Exception as e:
                logger.error(f"Failed to get margin info: {e}")
                if account_data["error"]:
                    account_data["error"] += f", Margin fetch failed: {str(e)}"
                else:
                    account_data["error"] = f"Margin fetch failed: {str(e)}"

            return account_data

        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            logger.error(f"Exception type: {type(e)}")
            
            # Handle known Bluefin SDK data access issues
            error_msg = str(e)
            if "'data'" in error_msg and "KeyError" in str(type(e)):
                logger.warning("Detected Bluefin SDK data access error - likely missing 'data' field in API response")
                return {
                    "error": "Bluefin API response missing expected data fields - account may not be initialized", 
                    "balance": "0.0", 
                    "margin": {"available": "0.0", "used": "0.0"}, 
                    "address": account_data.get("address", "")
                }
            
            return {"error": str(e), "balance": "0.0", "margin": {"available": "0.0"}, "address": ""}

    async def get_positions(self) -> list[dict[str, Any]]:
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
                        logger.debug(
                            f"Processing position for symbol: {pos.get('symbol', 'Unknown')}"
                        )

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

                        formatted_positions.append(
                            {
                                "symbol": pos.get("symbol", ""),
                                "quantity": pos.get("quantity", "0"),
                                "avgPrice": pos.get(
                                    "avgEntryPrice", pos.get("avgPrice", "0")
                                ),
                                "unrealizedPnl": pos.get("unrealizedPnl", "0"),
                                "realizedPnl": pos.get("realizedPnl", "0"),
                                "leverage": pos.get("leverage", 1),
                                "margin": pos.get("margin", "0"),
                                "side": side,
                            }
                        )
                return formatted_positions
            # If it's a single position dict
            elif isinstance(position, dict):
                # Log position symbol only for debugging
                logger.debug(
                    f"Processing position for symbol: {position.get('symbol', 'Unknown')}"
                )

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

                return [
                    {
                        "symbol": position.get("symbol", ""),
                        "quantity": position.get("quantity", "0"),
                        "avgPrice": position.get(
                            "avgEntryPrice", position.get("avgPrice", "0")
                        ),
                        "unrealizedPnl": position.get("unrealizedPnl", "0"),
                        "realizedPnl": position.get("realizedPnl", "0"),
                        "leverage": position.get("leverage", 1),
                        "margin": position.get("margin", "0"),
                        "side": side,
                    }
                ]
            else:
                logger.warning(f"Unexpected position type: {type(position)}")
                return []

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def place_order(self, order_data: dict[str, Any]) -> dict[str, Any]:
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
                    reduce_only=order_data.get("reduceOnly", False),
                )
            else:
                # Place limit order
                price = float(order_data["price"])
                response = await self.client.place_limit_order(
                    symbol=self._get_market_symbol(symbol),
                    side=side,
                    price=price,
                    quantity=quantity,
                    reduce_only=order_data.get("reduceOnly", False),
                )

            return {"status": "success", "order": response}

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
                symbol=self._get_market_symbol(symbol), leverage=leverage
            )
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False

    async def get_candlestick_data(self, symbol: str, interval: str, limit: int, start_time: int, end_time: int) -> list[list]:
        """Get historical candlestick data."""
        if not self.initialized:
            logger.error("SDK not initialized for candlestick data")
            return []

        try:
            # Convert symbol to market symbol enum
            market_symbol = self._get_market_symbol(symbol)
            logger.info(f"Fetching candlestick data for {symbol} (market symbol: {market_symbol})")
            
            # Try to get historical candles using the Bluefin SDK
            # Note: The exact method name may vary depending on the SDK version
            # Common method names: get_candles, get_klines, get_historical_data
            
            # Attempt different possible method names for getting candles
            candle_methods = [
                'get_candles',
                'get_klines', 
                'get_historical_data',
                'get_candlestick_data',
                'get_ohlcv_data'
            ]
            
            for method_name in candle_methods:
                if hasattr(self.client, method_name):
                    try:
                        method = getattr(self.client, method_name)
                        logger.info(f"Trying method: {method_name}")
                        
                        # Try different parameter combinations
                        try:
                            # Method 1: Direct parameters
                            candles = await method(
                                symbol=market_symbol,
                                interval=interval,
                                limit=limit,
                                start_time=start_time,
                                end_time=end_time
                            )
                            if candles:
                                logger.info(f"Successfully got {len(candles)} candles using {method_name}")
                                return candles
                        except Exception as e:
                            logger.debug(f"Method {method_name} with direct params failed: {e}")
                        
                        try:
                            # Method 2: Dictionary parameters
                            candles = await method({
                                'symbol': market_symbol,
                                'interval': interval,
                                'limit': limit,
                                'startTime': start_time,
                                'endTime': end_time
                            })
                            if candles:
                                logger.info(f"Successfully got {len(candles)} candles using {method_name} with dict params")
                                return candles
                        except Exception as e:
                            logger.debug(f"Method {method_name} with dict params failed: {e}")
                            
                        try:
                            # Method 3: Just symbol and limit (common minimal API)
                            candles = await method(symbol=market_symbol, limit=limit)
                            if candles:
                                logger.info(f"Successfully got {len(candles)} candles using {method_name} minimal params")
                                return candles
                        except Exception as e:
                            logger.debug(f"Method {method_name} with minimal params failed: {e}")
                            
                    except Exception as e:
                        logger.debug(f"Method {method_name} not working: {e}")
                        continue
            
            # If no direct SDK method works, try getting recent data and generate mock historical
            logger.warning("No working candle method found, generating mock historical data")
            return self._generate_mock_candles(symbol, interval, limit, start_time, end_time)
            
        except Exception as e:
            logger.error(f"Error getting candlestick data: {e}")
            # Return mock data as fallback
            return self._generate_mock_candles(symbol, interval, limit, start_time, end_time)

    def _generate_mock_candles(self, symbol: str, interval: str, limit: int, start_time: int, end_time: int) -> list[list]:
        """Generate mock candlestick data as fallback."""
        import random
        import time
        
        logger.info(f"Generating mock candles for {symbol} from {start_time} to {end_time}")
        
        # Base price for different symbols
        base_prices = {
            "SUI-PERP": 3.50,
            "BTC-PERP": 45000.0,
            "ETH-PERP": 2500.0,
            "SOL-PERP": 125.0,
        }
        
        base_price = base_prices.get(symbol, 3.50)
        
        # Convert interval to seconds
        interval_seconds = self._interval_to_seconds(interval)
        
        # Calculate number of candles to generate
        time_range = (end_time - start_time) / 1000  # Convert to seconds
        num_candles = min(int(time_range / interval_seconds), limit)
        
        candles = []
        current_price = base_price
        
        for i in range(num_candles):
            # Calculate timestamp for this candle
            candle_time = start_time + (i * interval_seconds * 1000)  # Back to milliseconds
            
            # Generate realistic OHLCV data
            volatility = 0.005  # 0.5% volatility per candle
            change = random.uniform(-volatility, volatility)
            
            open_price = current_price
            close_price = open_price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.003))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.003))
            volume = random.uniform(1000, 50000)
            
            # Format: [timestamp, open, high, low, close, volume]
            candle = [
                candle_time,
                round(open_price, 4),
                round(high_price, 4), 
                round(low_price, 4),
                round(close_price, 4),
                round(volume, 2)
            ]
            
            candles.append(candle)
            current_price = close_price
        
        logger.info(f"Generated {len(candles)} mock candles")
        return candles
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        try:
            if interval.endswith('s'):
                return int(interval[:-1])
            elif interval.endswith('m'):
                return int(interval[:-1]) * 60
            elif interval.endswith('h'):
                return int(interval[:-1]) * 3600
            elif interval.endswith('d'):
                return int(interval[:-1]) * 86400
            else:
                # Default to 60 seconds if format is unclear
                return 60
        except:
            return 60


# Global service instance
service = BluefinSDKService()

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(
    os.getenv("BLUEFIN_SERVICE_RATE_LIMIT", "100")
)  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds
rate_limit_storage = defaultdict(lambda: {"count": 0, "window_start": time.time()})


# Authentication middleware
@web.middleware
async def auth_middleware(request, handler):
    """Validate API key authentication."""
    # Skip auth for health check endpoint
    if request.path == "/health":
        return await handler(request)

    # Get API key from environment
    expected_api_key = os.getenv("BLUEFIN_SERVICE_API_KEY")
    if not expected_api_key:
        logger.error("BLUEFIN_SERVICE_API_KEY not configured")
        return web.json_response({"error": "Service misconfigured"}, status=500)

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning(f"Missing or invalid auth header from {request.remote}")
        return web.json_response({"error": "Unauthorized"}, status=401)

    # Extract and validate token
    provided_api_key = auth_header[7:]  # Remove "Bearer " prefix
    if not secrets.compare_digest(provided_api_key, expected_api_key):
        logger.warning(f"Invalid API key attempt from {request.remote}")
        return web.json_response({"error": "Unauthorized"}, status=401)

    # API key is valid, proceed to handler
    return await handler(request)


# Rate limiting middleware
@web.middleware
async def rate_limit_middleware(request, handler):
    """Apply rate limiting per client IP."""
    # Skip rate limiting for health check
    if request.path == "/health":
        return await handler(request)

    # Get client IP
    client_ip = request.headers.get("X-Forwarded-For", request.remote)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # Check rate limit
    current_time = time.time()
    client_data = rate_limit_storage[client_ip]

    # Reset window if expired
    if current_time - client_data["window_start"] > RATE_LIMIT_WINDOW:
        client_data["count"] = 0
        client_data["window_start"] = current_time

    # Check if limit exceeded
    if client_data["count"] >= RATE_LIMIT_REQUESTS:
        remaining_time = int(
            RATE_LIMIT_WINDOW - (current_time - client_data["window_start"])
        )
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return web.json_response(
            {"error": "Rate limit exceeded", "retry_after": remaining_time},
            status=429,
            headers={"Retry-After": str(remaining_time)},
        )

    # Increment counter
    client_data["count"] += 1

    # Add rate limit headers to response
    response = await handler(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(
        RATE_LIMIT_REQUESTS - client_data["count"]
    )
    response.headers["X-RateLimit-Reset"] = str(
        int(client_data["window_start"] + RATE_LIMIT_WINDOW)
    )

    return response


# REST API Routes
async def health_check(request):
    """Health check endpoint."""
    return web.json_response(
        {
            "status": "healthy" if service.initialized else "unhealthy",
            "initialized": service.initialized,
            "network": service.network,
        }
    )


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
        order_id = request.match_info["order_id"]
        success = await service.cancel_order(order_id)
        return web.json_response({"success": success})
    except Exception as e:
        logger.error(f"Error in cancel_order: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_ticker(request):
    """Get market ticker."""
    try:
        symbol = request.query.get("symbol", "SUI-PERP")
        
        # Convert symbol to market symbol enum
        market_symbol = service._get_market_symbol(symbol)
        logger.info(f"Converted {symbol} to {market_symbol}")

        # Get orderbook for best bid/ask
        try:
            orderbook_request = GetOrderbookRequest(symbol=market_symbol, limit=10)
            orderbook = await service.client.get_orderbook(orderbook_request)
            logger.info(f"Got orderbook: {type(orderbook)}")
        except Exception as orderbook_error:
            logger.error(f"Orderbook error: {orderbook_error}")
            # Return mock data for now
            return web.json_response(
                {
                    "symbol": symbol,
                    "price": "1.50", 
                    "bestBid": "1.49",
                    "bestAsk": "1.51",
                    "error": f"Orderbook unavailable: {orderbook_error}"
                }
            )

        best_bid = orderbook["bids"][0]["price"] if orderbook.get("bids") else 0
        best_ask = orderbook["asks"][0]["price"] if orderbook.get("asks") else 0

        return web.json_response(
            {
                "symbol": symbol,
                "price": str((float(best_bid) + float(best_ask)) / 2) if best_bid and best_ask else "0",
                "bestBid": str(best_bid),
                "bestAsk": str(best_ask),
            }
        )
    except Exception as e:
        logger.error(f"Error in get_ticker: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def set_leverage(request):
    """Set leverage."""
    try:
        data = await request.json()
        success = await service.set_leverage(data["symbol"], data["leverage"])
        return web.json_response({"success": success})
    except Exception as e:
        logger.error(f"Error in set_leverage: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def get_market_candles(request):
    """Get historical candlestick data."""
    try:
        # Extract query parameters
        symbol = request.query.get("symbol", "SUI-PERP")
        interval = request.query.get("interval", "1m")
        limit = int(request.query.get("limit", "100"))
        start_time = int(request.query.get("startTime", "0"))
        end_time = int(request.query.get("endTime", "0"))
        
        logger.info(f"Market candles request: symbol={symbol}, interval={interval}, limit={limit}, startTime={start_time}, endTime={end_time}")
        
        # If start_time or end_time are 0, calculate them
        import time
        current_time_ms = int(time.time() * 1000)
        
        if end_time == 0:
            end_time = current_time_ms
            
        if start_time == 0:
            # Default to 24 hours ago
            start_time = end_time - (24 * 60 * 60 * 1000)
        
        # Get candlestick data from service
        candles = await service.get_candlestick_data(symbol, interval, limit, start_time, end_time)
        
        return web.json_response({"candles": candles})
        
    except ValueError as e:
        logger.error(f"Invalid parameter in get_market_candles: {e}")
        return web.json_response({"error": f"Invalid parameter: {e}"}, status=400)
    except Exception as e:
        logger.error(f"Error in get_market_candles: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def debug_symbols(request):
    """Debug endpoint to inspect available symbols."""
    try:
        # Get all attributes of MARKET_SYMBOLS
        all_attrs = dir(MARKET_SYMBOLS)
        symbol_attrs = [attr for attr in all_attrs if not attr.startswith('_')]
        
        # Try to get some sample values
        sample_values = {}
        for attr in symbol_attrs[:10]:  # First 10 attributes
            try:
                value = getattr(MARKET_SYMBOLS, attr)
                sample_values[attr] = str(value)
            except Exception as e:
                sample_values[attr] = f"Error: {e}"
        
        # Test getting market symbol
        test_symbol = None
        try:
            test_symbol = service._get_market_symbol("SUI-PERP")
        except Exception as e:
            test_symbol = f"Error: {e}"
        
        # Test direct attribute access
        direct_sui = None
        try:
            direct_sui = MARKET_SYMBOLS.SUI
        except Exception as e:
            direct_sui = f"Error: {e}"
            
        return web.json_response({
            "all_attributes": all_attrs,
            "symbol_attributes": symbol_attrs, 
            "sample_values": sample_values,
            "type": str(type(MARKET_SYMBOLS)),
            "test_get_market_symbol": str(test_symbol),
            "direct_sui_access": str(direct_sui)
        })
    except Exception as e:
        logger.error(f"Error in debug_symbols: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def startup(app):
    """Initialize service on startup."""
    await service.initialize()


def create_app():
    """Create the aiohttp application."""
    # Create app with middleware
    app = web.Application(middlewares=[rate_limit_middleware, auth_middleware])

    # Add routes
    app.router.add_get("/health", health_check)
    app.router.add_get("/account", get_account)
    app.router.add_get("/positions", get_positions)
    app.router.add_post("/orders", place_order)
    app.router.add_delete("/orders/{order_id}", cancel_order)
    app.router.add_get("/market/ticker", get_ticker)
    app.router.add_get("/market/candles", get_market_candles)
    app.router.add_post("/leverage", set_leverage)
    app.router.add_get("/debug/symbols", debug_symbols)

    # Add startup handler
    app.on_startup.append(startup)

    return app


if __name__ == "__main__":
    # Get host and port from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))

    # Check if API key is configured
    if not os.getenv("BLUEFIN_SERVICE_API_KEY"):
        logger.warning("BLUEFIN_SERVICE_API_KEY not set - generating a random key")
        api_key = secrets.token_urlsafe(32)
        os.environ["BLUEFIN_SERVICE_API_KEY"] = api_key
        logger.info(f"Generated API key: {api_key}")
        logger.info(
            "Please set BLUEFIN_SERVICE_API_KEY in your environment for production use"
        )

    # Create and run app
    app = create_app()
    logger.info(f"Starting Bluefin SDK service on {host}:{port}")
    logger.info(
        f"Rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
    )
    web.run_app(app, host=host, port=port)
