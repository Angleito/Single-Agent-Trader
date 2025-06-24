"""
Functional Market Data Adapter

This module provides an adapter that bridges the existing WebSocket market data
system with functional programming types, enabling real-time data processing
while maintaining compatibility with the current infrastructure.
"""

import asyncio
import logging
from datetime import datetime, UTC
from decimal import Decimal
from typing import Any, Callable, Protocol

from bot.data.market import MarketDataProvider
from bot.trading_types import MarketData as CurrentMarketData

from .type_converters import (
    create_connection_state,
    create_data_quality,
    create_market_data_stream,
    create_orderbook_message_from_data,
    create_realtime_update,
    create_ticker_message_from_data,
    create_trade_message_from_data,
    current_market_data_to_fp_candle,
    update_connection_state,
    update_data_quality,
    validate_connection_health,
    validate_data_quality,
)
from ..types.market import (
    ConnectionState,
    ConnectionStatus,
    DataQuality,
    FPCandle,
    MarketDataStream,
    OrderBookMessage,
    RealtimeUpdate,
    StreamProcessor,
    TickerMessage,
    TradeMessage,
)

logger = logging.getLogger(__name__)


class FunctionalMarketDataProcessor:
    """
    Functional processor for real-time market data.
    
    This class processes market data using functional types while maintaining
    compatibility with the existing WebSocket infrastructure.
    """
    
    def __init__(self, symbol: str, interval: str = "1m"):
        """
        Initialize the functional market data processor.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            interval: Data interval (e.g., '1m', '5m')
        """
        self.symbol = symbol
        self.interval = interval
        
        # Functional state
        self._connection_state = create_connection_state(
            url=f"wss://advanced-trade-ws.coinbase.com",
            status="DISCONNECTED"
        )
        self._data_quality = create_data_quality()
        self._stream = create_market_data_stream(symbol, ["coinbase"])
        
        # Processed data storage
        self._recent_candles: list[FPCandle] = []
        self._recent_updates: list[RealtimeUpdate] = []
        
        # Callbacks for functional data
        self._candle_callbacks: list[Callable[[FPCandle], None]] = []
        self._update_callbacks: list[Callable[[RealtimeUpdate], None]] = []
        self._stream_callbacks: list[Callable[[MarketDataStream], None]] = []
        
        # Integration with existing system
        self._market_data_provider: MarketDataProvider | None = None
        
        logger.info(f"Initialized FunctionalMarketDataProcessor for {symbol}")
    
    async def start(self, market_data_provider: MarketDataProvider) -> None:
        """
        Start the functional market data processor.
        
        Args:
            market_data_provider: Existing market data provider to integrate with
        """
        self._market_data_provider = market_data_provider
        
        # Subscribe to existing provider's updates
        market_data_provider.subscribe_to_updates(self._on_market_data_update)
        
        # Update connection state
        self._connection_state = update_connection_state(
            self._connection_state,
            new_status="CONNECTING"
        )
        
        # Wait for connection
        if await market_data_provider.wait_for_websocket_data(timeout=30):
            self._connection_state = update_connection_state(
                self._connection_state,
                new_status="CONNECTED",
                message_received=True
            )
            logger.info("Functional market data processor connected")
        else:
            self._connection_state = update_connection_state(
                self._connection_state,
                new_status="ERROR",
                error="Connection timeout"
            )
            logger.error("Failed to connect functional market data processor")
    
    async def stop(self) -> None:
        """Stop the functional market data processor."""
        if self._market_data_provider:
            self._market_data_provider.unsubscribe_from_updates(self._on_market_data_update)
        
        self._connection_state = update_connection_state(
            self._connection_state,
            new_status="DISCONNECTED"
        )
        
        logger.info("Functional market data processor stopped")
    
    def _on_market_data_update(self, market_data: CurrentMarketData) -> None:
        """
        Handle market data updates from the existing provider.
        
        Args:
            market_data: Market data from the existing system
        """
        try:
            # Convert to functional type
            fp_candle = current_market_data_to_fp_candle(market_data)
            
            # Update functional state
            self._recent_candles.append(fp_candle)
            if len(self._recent_candles) > 1000:  # Keep last 1000 candles
                self._recent_candles = self._recent_candles[-1000:]
            
            # Create real-time update
            update = create_realtime_update(
                symbol=self.symbol,
                update_type="candle",
                data={
                    "timestamp": market_data.timestamp.isoformat(),
                    "open": str(market_data.open),
                    "high": str(market_data.high),
                    "low": str(market_data.low),
                    "close": str(market_data.close),
                    "volume": str(market_data.volume),
                },
                exchange="coinbase"
            )
            
            self._recent_updates.append(update)
            if len(self._recent_updates) > 500:  # Keep last 500 updates
                self._recent_updates = self._recent_updates[-500:]
            
            # Update data quality
            self._data_quality = update_data_quality(
                self._data_quality,
                new_message=True,
                processed=True
            )
            
            # Update connection state
            self._connection_state = update_connection_state(
                self._connection_state,
                message_received=True
            )
            
            # Update stream state
            self._stream = MarketDataStream(
                symbol=self._stream.symbol,
                exchanges=self._stream.exchanges,
                connection_states={
                    "coinbase": self._connection_state
                },
                data_quality=self._data_quality,
                active=self._stream.active,
            )
            
            # Notify callbacks
            self._notify_candle_callbacks(fp_candle)
            self._notify_update_callbacks(update)
            self._notify_stream_callbacks(self._stream)
            
        except Exception as e:
            logger.error(f"Error processing market data update: {e}")
            
            # Update data quality with failure
            self._data_quality = update_data_quality(
                self._data_quality,
                new_message=True,
                processed=False,
                validation_failed=True
            )
    
    def process_websocket_message(self, message: dict[str, Any]) -> None:
        """
        Process raw WebSocket message using functional types.
        
        Args:
            message: Raw WebSocket message
        """
        try:
            channel = message.get("channel", "")
            
            if channel == "ticker":
                self._process_ticker_message(message)
            elif channel == "market_trades":
                self._process_trade_message(message)
            elif channel == "orderbook":
                self._process_orderbook_message(message)
            else:
                logger.debug(f"Unhandled WebSocket channel: {channel}")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            
            # Update data quality with failure
            self._data_quality = update_data_quality(
                self._data_quality,
                new_message=True,
                processed=False,
                validation_failed=True
            )
    
    def _process_ticker_message(self, message: dict[str, Any]) -> None:
        """Process ticker message using functional types."""
        try:
            ticker_msg = create_ticker_message_from_data(message, self.symbol)
            
            # Create real-time update
            update = create_realtime_update(
                symbol=self.symbol,
                update_type="ticker",
                data=message,
                exchange="coinbase"
            )
            
            self._recent_updates.append(update)
            self._notify_update_callbacks(update)
            
            logger.debug(f"Processed ticker: {ticker_msg.price}")
            
        except Exception as e:
            logger.error(f"Error processing ticker message: {e}")
    
    def _process_trade_message(self, message: dict[str, Any]) -> None:
        """Process trade message using functional types."""
        try:
            trade_msg = create_trade_message_from_data(message, self.symbol)
            
            # Create real-time update
            update = create_realtime_update(
                symbol=self.symbol,
                update_type="trade",
                data=message,
                exchange="coinbase"
            )
            
            self._recent_updates.append(update)
            self._notify_update_callbacks(update)
            
            logger.debug(f"Processed trade: {trade_msg.price} x {trade_msg.size}")
            
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
    
    def _process_orderbook_message(self, message: dict[str, Any]) -> None:
        """Process order book message using functional types."""
        try:
            orderbook_msg = create_orderbook_message_from_data(message, self.symbol)
            
            # Create real-time update
            update = create_realtime_update(
                symbol=self.symbol,
                update_type="orderbook",
                data=message,
                exchange="coinbase"
            )
            
            self._recent_updates.append(update)
            self._notify_update_callbacks(update)
            
            logger.debug(f"Processed orderbook: {len(orderbook_msg.bids)} bids, {len(orderbook_msg.asks)} asks")
            
        except Exception as e:
            logger.error(f"Error processing orderbook message: {e}")
    
    # Callback management
    
    def add_candle_callback(self, callback: Callable[[FPCandle], None]) -> None:
        """Add a callback for candle updates."""
        self._candle_callbacks.append(callback)
    
    def add_update_callback(self, callback: Callable[[RealtimeUpdate], None]) -> None:
        """Add a callback for real-time updates."""
        self._update_callbacks.append(callback)
    
    def add_stream_callback(self, callback: Callable[[MarketDataStream], None]) -> None:
        """Add a callback for stream status updates."""
        self._stream_callbacks.append(callback)
    
    def remove_candle_callback(self, callback: Callable[[FPCandle], None]) -> None:
        """Remove a candle callback."""
        if callback in self._candle_callbacks:
            self._candle_callbacks.remove(callback)
    
    def remove_update_callback(self, callback: Callable[[RealtimeUpdate], None]) -> None:
        """Remove an update callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    def remove_stream_callback(self, callback: Callable[[MarketDataStream], None]) -> None:
        """Remove a stream callback."""
        if callback in self._stream_callbacks:
            self._stream_callbacks.remove(callback)
    
    def _notify_candle_callbacks(self, candle: FPCandle) -> None:
        """Notify all candle callbacks."""
        for callback in self._candle_callbacks:
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Error in candle callback: {e}")
    
    def _notify_update_callbacks(self, update: RealtimeUpdate) -> None:
        """Notify all update callbacks."""
        for callback in self._update_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    def _notify_stream_callbacks(self, stream: MarketDataStream) -> None:
        """Notify all stream callbacks."""
        for callback in self._stream_callbacks:
            try:
                callback(stream)
            except Exception as e:
                logger.error(f"Error in stream callback: {e}")
    
    # State access
    
    def get_recent_candles(self, limit: int | None = None) -> list[FPCandle]:
        """Get recent candles."""
        if limit is None:
            return self._recent_candles.copy()
        return self._recent_candles[-limit:].copy()
    
    def get_recent_updates(self, limit: int | None = None) -> list[RealtimeUpdate]:
        """Get recent updates."""
        if limit is None:
            return self._recent_updates.copy()
        return self._recent_updates[-limit:].copy()
    
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state
    
    def get_data_quality(self) -> DataQuality:
        """Get current data quality metrics."""
        return self._data_quality
    
    def get_stream_status(self) -> MarketDataStream:
        """Get current stream status."""
        return self._stream
    
    def is_healthy(self) -> bool:
        """Check if the processor is healthy."""
        return (
            validate_connection_health(self._connection_state) and
            validate_data_quality(self._data_quality) and
            self._stream.overall_health
        )


class FunctionalStreamProcessorImpl(StreamProcessor):
    """
    Implementation of StreamProcessor protocol for functional market data processing.
    """
    
    def __init__(self, processor: FunctionalMarketDataProcessor):
        """
        Initialize with a functional market data processor.
        
        Args:
            processor: The functional market data processor
        """
        self.processor = processor
    
    def process_ticker(self, message: TickerMessage) -> None:
        """Process ticker message."""
        # Convert to real-time update and notify callbacks
        update = create_realtime_update(
            symbol=self.processor.symbol,
            update_type="ticker",
            data=message.data,
            exchange="coinbase"
        )
        self.processor._notify_update_callbacks(update)
    
    def process_trade(self, message: TradeMessage) -> None:
        """Process trade message."""
        # Convert to real-time update and notify callbacks
        update = create_realtime_update(
            symbol=self.processor.symbol,
            update_type="trade",
            data=message.data,
            exchange="coinbase"
        )
        self.processor._notify_update_callbacks(update)
    
    def process_orderbook(self, message: OrderBookMessage) -> None:
        """Process order book message."""
        # Convert to real-time update and notify callbacks
        update = create_realtime_update(
            symbol=self.processor.symbol,
            update_type="orderbook",
            data=message.data,
            exchange="coinbase"
        )
        self.processor._notify_update_callbacks(update)


# Factory functions

def create_functional_market_data_processor(symbol: str, interval: str = "1m") -> FunctionalMarketDataProcessor:
    """
    Create a functional market data processor.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        
    Returns:
        Configured functional market data processor
    """
    return FunctionalMarketDataProcessor(symbol, interval)


def create_stream_processor(processor: FunctionalMarketDataProcessor) -> StreamProcessor:
    """
    Create a stream processor from a functional market data processor.
    
    Args:
        processor: The functional market data processor
        
    Returns:
        Stream processor implementation
    """
    return FunctionalStreamProcessorImpl(processor)


# Integration helpers

async def integrate_with_existing_provider(
    processor: FunctionalMarketDataProcessor,
    symbol: str,
    interval: str = "1m"
) -> MarketDataProvider:
    """
    Create and integrate with an existing MarketDataProvider.
    
    Args:
        processor: Functional market data processor
        symbol: Trading symbol
        interval: Data interval
        
    Returns:
        Connected MarketDataProvider
    """
    # Create existing provider
    provider = MarketDataProvider(symbol, interval)
    
    # Connect and start functional processor
    await provider.connect()
    await processor.start(provider)
    
    logger.info(f"Integrated functional processor with existing provider for {symbol}")
    
    return provider


def create_integrated_market_data_system(symbol: str, interval: str = "1m") -> tuple[FunctionalMarketDataProcessor, MarketDataProvider]:
    """
    Create an integrated market data system with both functional and existing components.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        
    Returns:
        Tuple of (functional_processor, market_data_provider)
    """
    functional_processor = create_functional_market_data_processor(symbol, interval)
    provider = MarketDataProvider(symbol, interval)
    
    return functional_processor, provider