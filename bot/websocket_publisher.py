"""
WebSocket Publisher for real-time trading data streaming to dashboard.

This module provides a fault-tolerant WebSocket client that publishes trading events
to the dashboard backend in real-time, enabling live monitoring and visualization.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from decimal import Decimal

import aiohttp
from pydantic import BaseModel

from .config import Settings


class WebSocketPublisher:
    """
    Fault-tolerant WebSocket publisher for real-time trading data streaming.
    
    Publishes trading events to dashboard backend using the same message schemas
    expected by the frontend WebSocket client.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Use configuration settings
        self.enabled = settings.system.enable_websocket_publishing
        self.dashboard_url = settings.system.websocket_dashboard_url
        self.connection_healthy = False
        self.last_ping_time = 0
        self.ping_interval = settings.system.websocket_publish_interval
        
        # Message queue for handling connection failures
        self.message_queue = []
        self.max_queue_size = settings.system.websocket_queue_size
        
        # Connection retry settings from configuration
        self.max_retries = settings.system.websocket_max_retries
        self.retry_delay = settings.system.websocket_retry_delay
        self.retry_count = 0
        self.connection_timeout = settings.system.websocket_timeout
        
    async def initialize(self) -> bool:
        """Initialize WebSocket connection if enabled."""
        if not self.enabled:
            self.logger.info("WebSocket publishing disabled via configuration")
            return False
            
        try:
            self.session = aiohttp.ClientSession()
            await self._connect()
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize WebSocket publisher: {e}")
            await self._cleanup()
            return False
    
    async def _connect(self) -> bool:
        """Establish WebSocket connection to dashboard."""
        try:
            self.logger.info(f"Connecting to dashboard WebSocket: {self.dashboard_url}")
            
            self.ws = await self.session.ws_connect(
                self.dashboard_url,
                timeout=aiohttp.ClientTimeout(total=self.connection_timeout),
                heartbeat=30
            )
            
            self.connection_healthy = True
            self.retry_count = 0
            self.logger.info("WebSocket connection established successfully")
            
            # Send initial connection message
            await self._send_message({
                "type": "connection_established", 
                "data": {
                    "source": "trading_bot",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Trading bot WebSocket publisher connected"
                }
            })
            
            # Process any queued messages
            await self._process_queued_messages()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to dashboard WebSocket: {e}")
            self.connection_healthy = False
            return False
    
    async def _send_message(self, message: Dict[str, Any]) -> bool:
        """Send message via WebSocket with error handling."""
        if not self.enabled or not self.ws:
            return False
            
        try:
            # Ensure message has required fields
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            
            # Convert Decimal to float for JSON serialization
            message = self._serialize_message(message)
            
            await self.ws.send_str(json.dumps(message))
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to send WebSocket message: {e}")
            self.connection_healthy = False
            
            # Queue message for retry if connection is restored
            if len(self.message_queue) < self.max_queue_size:
                self.message_queue.append(message)
            
            # Attempt to reconnect
            asyncio.create_task(self._reconnect())
            return False
    
    def _serialize_message(self, message: Any) -> Any:
        """Convert Decimal and other non-JSON types to serializable format."""
        if isinstance(message, dict):
            return {k: self._serialize_message(v) for k, v in message.items()}
        elif isinstance(message, list):
            return [self._serialize_message(item) for item in message]
        elif isinstance(message, Decimal):
            return float(message)
        elif isinstance(message, BaseModel):
            return message.model_dump()
        else:
            return message
    
    async def _process_queued_messages(self):
        """Process any queued messages after reconnection."""
        while self.message_queue and self.connection_healthy:
            message = self.message_queue.pop(0)
            await self._send_message(message)
    
    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self.retry_count >= self.max_retries:
            self.logger.error("Max WebSocket reconnection attempts reached")
            return
            
        self.retry_count += 1
        delay = self.retry_delay * (2 ** (self.retry_count - 1))
        
        self.logger.info(f"Attempting WebSocket reconnection {self.retry_count}/{self.max_retries} in {delay}s")
        await asyncio.sleep(delay)
        
        try:
            await self._connect()
        except Exception as e:
            self.logger.warning(f"Reconnection attempt {self.retry_count} failed: {e}")
    
    async def publish_market_data(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """Publish market data update."""
        message = {
            "type": "market_data",
            "data": {
                "symbol": symbol,
                "price": price,
                "timestamp": (timestamp or datetime.now()).isoformat()
            }
        }
        await self._send_message(message)
    
    async def publish_trading_loop(self, price: float, action: str, confidence: float, symbol: str = "BTC-USD"):
        """Publish trading loop update matching TradingLoopMessage schema."""
        message = {
            "type": "trading_loop",
            "data": {
                "price": price,
                "action": action,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol
            }
        }
        await self._send_message(message)
    
    async def publish_ai_decision(self, action: str, reasoning: str, confidence: float):
        """Publish AI decision matching AIDecisionMessage schema."""
        message = {
            "type": "ai_decision", 
            "data": {
                "action": action,
                "reasoning": reasoning,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self._send_message(message)
    
    async def publish_trading_decision(self, request_id: str, action: str, confidence: float, 
                                     reasoning: str, price: float, quantity: Optional[float] = None,
                                     leverage: Optional[int] = None, indicators: Optional[Dict] = None,
                                     risk_analysis: Optional[Dict] = None):
        """Publish detailed trading decision matching TradingDecisionMessage schema."""
        message = {
            "type": "trading_decision",
            "data": {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "action": action.upper() if action.upper() in ["BUY", "SELL", "HOLD"] else "HOLD",
                "confidence": confidence,
                "reasoning": reasoning,
                "price": price,
                "quantity": quantity,
                "leverage": leverage,
                "indicators": indicators or {},
                "risk_analysis": risk_analysis or {}
            }
        }
        await self._send_message(message)
    
    async def publish_system_status(self, status: str, health: bool, errors: Optional[list] = None):
        """Publish system status matching SystemStatusMessage schema."""
        message = {
            "type": "system_status",
            "data": {
                "status": status,
                "health": health,
                "errors": errors or [],
                "timestamp": datetime.now().isoformat()
            }
        }
        await self._send_message(message)
    
    async def publish_llm_request(self, request_id: str, model: str, prompt_tokens: Optional[int] = None,
                                max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                                context: Optional[Dict] = None):
        """Publish LLM request matching LLMRequestMessage schema."""
        message = {
            "type": "llm_request",
            "data": {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "context": context or {}
            }
        }
        await self._send_message(message)
    
    async def publish_llm_response(self, request_id: str, model: str, response_time_ms: int,
                                 completion_tokens: Optional[int] = None, total_tokens: Optional[int] = None,
                                 cost_estimate_usd: Optional[float] = None, success: bool = True,
                                 error: Optional[str] = None, raw_response: Optional[str] = None):
        """Publish LLM response matching LLMResponseMessage schema."""
        message = {
            "type": "llm_response",
            "data": {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "response_time_ms": response_time_ms,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_estimate_usd": cost_estimate_usd,
                "success": success,
                "error": error,
                "raw_response": raw_response
            }
        }
        await self._send_message(message)
    
    async def publish_position_update(self, symbol: str, side: str, size: float, entry_price: float,
                                    current_price: float, unrealized_pnl: float):
        """Publish position update."""
        message = {
            "type": "position_update",
            "data": {
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self._send_message(message)
    
    async def publish_indicator_data(self, symbol: str, indicators: Dict[str, float]):
        """Publish technical indicator data."""
        message = {
            "type": "indicator_update",
            "data": {
                "symbol": symbol,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self._send_message(message)
    
    async def publish_trade_execution(self, order_id: str, symbol: str, side: str, quantity: float,
                                    price: float, status: str, trade_action: Optional[Dict] = None):
        """Publish trade execution event."""
        message = {
            "type": "trade_execution",
            "data": {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "status": status,
                "trade_action": trade_action,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self._send_message(message)
    
    async def publish_error(self, error_message: str, level: str = "ERROR"):
        """Publish error message matching ErrorMessage schema."""
        message = {
            "type": "error",
            "data": {
                "message": error_message,
                "level": level,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self._send_message(message)
    
    async def ping(self):
        """Send ping to keep connection alive."""
        if not self.enabled or not self.connection_healthy:
            return
            
        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_ping_time >= self.ping_interval:
            message = {
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            }
            await self._send_message(message)
            self.last_ping_time = current_time
    
    async def _cleanup(self):
        """Clean up WebSocket connection and session."""
        try:
            if self.ws:
                await self.ws.close()
            if self.session:
                await self.session.close()
        except Exception as e:
            self.logger.debug(f"Error during WebSocket cleanup: {e}")
        finally:
            self.ws = None
            self.session = None
            self.connection_healthy = False
    
    async def close(self):
        """Close WebSocket connection gracefully."""
        if self.enabled:
            await self._send_message({
                "type": "connection_closed",
                "data": {
                    "source": "trading_bot",
                    "message": "Trading bot WebSocket publisher disconnecting"
                }
            })
            await self._cleanup()