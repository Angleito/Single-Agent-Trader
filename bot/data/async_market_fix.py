"""
Fix for async coroutine bug in market data loading.

The issue is that Bluefin market providers have async get_latest_ohlcv methods
while Coinbase market providers have sync methods, causing coroutine errors
when the code tries to use len() on an unawaited coroutine.
"""

import asyncio
import logging
from typing import List, Optional, Union
from ..types import MarketData

logger = logging.getLogger(__name__)


class UnifiedMarketDataInterface:
    """
    Unified interface to handle both sync and async market data providers.
    
    This wrapper ensures consistent behavior regardless of whether the underlying
    provider uses sync or async methods for get_latest_ohlcv.
    """
    
    def __init__(self, provider):
        """Initialize with either sync or async market data provider."""
        self.provider = provider
        self._is_async_provider = self._detect_async_provider(provider)
        logger.info(f"Initialized unified interface (async: {self._is_async_provider})")
    
    def _detect_async_provider(self, provider) -> bool:
        """Detect if the provider has async get_latest_ohlcv method."""
        import inspect
        
        if hasattr(provider, 'get_latest_ohlcv'):
            method = getattr(provider, 'get_latest_ohlcv')
            return inspect.iscoroutinefunction(method)
        return False
    
    async def get_latest_ohlcv_async(self, limit: Optional[int] = None) -> List[MarketData]:
        """Get latest OHLCV data - handles both sync and async providers."""
        try:
            if self._is_async_provider:
                # Provider has async method - await it
                return await self.provider.get_latest_ohlcv(limit=limit)
            else:
                # Provider has sync method - call directly
                return self.provider.get_latest_ohlcv(limit=limit)
        except Exception as e:
            logger.error(f"Failed to get OHLCV data: {e}")
            return []
    
    def get_latest_ohlcv_sync(self, limit: Optional[int] = None) -> List[MarketData]:
        """
        Synchronous wrapper that handles async providers.
        
        This method should be used when called from synchronous contexts
        that need to handle both sync and async providers.
        """
        try:
            if self._is_async_provider:
                # Provider has async method - need to run in event loop
                try:
                    # Try to get the current event loop
                    loop = asyncio.get_running_loop()
                    # We're in an async context - this should not be called from sync context
                    logger.error("Sync method called from async context - use get_latest_ohlcv_async instead")
                    return []
                except RuntimeError:
                    # No event loop running - create one and run the async method
                    return asyncio.run(self.provider.get_latest_ohlcv(limit=limit))
            else:
                # Provider has sync method - call directly
                return self.provider.get_latest_ohlcv(limit=limit)
        except Exception as e:
            logger.error(f"Failed to get OHLCV data synchronously: {e}")
            return []
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying provider."""
        return getattr(self.provider, name)


def fix_async_market_data_calls():
    """
    Apply fixes to main.py to handle async market data calls properly.
    
    This function provides the strategy for fixing the coroutine issues:
    1. Identify calls to get_latest_ohlcv that need to be awaited
    2. Update the calling code to handle both sync and async providers
    3. Ensure proper error handling for both cases
    """
    fixes_applied = []
    
    # Fix 1: Update main.py _wait_for_initial_data method
    fix1 = """
    # Replace this line in main.py around line 594:
    # data = self.market_data.get_latest_ohlcv(limit=500)
    
    # With this async-safe version:
    if hasattr(self.market_data, 'get_latest_ohlcv'):
        import inspect
        if inspect.iscoroutinefunction(self.market_data.get_latest_ohlcv):
            data = await self.market_data.get_latest_ohlcv(limit=500)
        else:
            data = self.market_data.get_latest_ohlcv(limit=500)
    else:
        data = []
    """
    fixes_applied.append(("_wait_for_initial_data method", fix1))
    
    # Fix 2: Update main.py _should_trade method  
    fix2 = """
    # Replace this line in main.py around line 294:
    # latest_data = self.market_data.get_latest_ohlcv(limit=1)
    
    # With this async-safe version:
    if hasattr(self.market_data, 'get_latest_ohlcv'):
        import inspect
        if inspect.iscoroutinefunction(self.market_data.get_latest_ohlcv):
            latest_data = await self.market_data.get_latest_ohlcv(limit=1)
        else:
            latest_data = self.market_data.get_latest_ohlcv(limit=1)
    else:
        latest_data = []
    """
    fixes_applied.append(("_should_trade method", fix2))
    
    # Fix 3: Update main.py run method
    fix3 = """
    # Replace this line in main.py around line 776:
    # latest_data = self.market_data.get_latest_ohlcv(limit=200)
    
    # With this async-safe version:
    if hasattr(self.market_data, 'get_latest_ohlcv'):
        import inspect
        if inspect.iscoroutinefunction(self.market_data.get_latest_ohlcv):
            latest_data = await self.market_data.get_latest_ohlcv(limit=200)
        else:
            latest_data = self.market_data.get_latest_ohlcv(limit=200)
    else:
        latest_data = []
    """
    fixes_applied.append(("run method", fix3))
    
    return fixes_applied


def create_market_data_provider_wrapper(provider):
    """
    Create a unified wrapper for market data providers.
    
    This ensures consistent sync/async behavior regardless of the underlying provider.
    """
    return UnifiedMarketDataInterface(provider)


# Async-safe utility functions for market data access
async def safe_get_market_data(provider, limit: Optional[int] = None) -> List[MarketData]:
    """Safely get market data from any provider (sync or async)."""
    import inspect
    
    if hasattr(provider, 'get_latest_ohlcv'):
        method = getattr(provider, 'get_latest_ohlcv')
        if inspect.iscoroutinefunction(method):
            return await method(limit=limit)
        else:
            return method(limit=limit)
    return []


def safe_get_market_data_sync(provider, limit: Optional[int] = None) -> List[MarketData]:
    """Safely get market data from any provider in sync context."""
    import inspect
    
    if hasattr(provider, 'get_latest_ohlcv'):
        method = getattr(provider, 'get_latest_ohlcv')
        if inspect.iscoroutinefunction(method):
            # Async provider called from sync context - run with asyncio.run
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - this should not happen
                logger.error("Sync function called from async context - use async version instead")
                return []
            except RuntimeError:
                # No event loop - create one and run the async method
                try:
                    return asyncio.run(method(limit=limit))
                except Exception as e:
                    logger.error(f"Failed to run async method in new event loop: {e}")
                    return []
        else:
            return method(limit=limit)
    return []