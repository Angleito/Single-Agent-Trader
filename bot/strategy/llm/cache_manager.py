"""LLM response cache manager."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        ...

    async def clear(self) -> None:
        """Clear all cache entries."""
        ...


class SimpleCacheAdapter:
    """Simple adapter for basic dict-like caches to async interface."""

    def __init__(self, cache: dict[str, Any] | None = None) -> None:
        """Initialize with optional cache dict."""
        self._cache: dict[str, Any] = cache if cache is not None else {}

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        return self._cache.get(key)

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = value

    async def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return cache size."""
        return len(self._cache)


class CacheManager:
    """Manages LLM response caching with statistics tracking."""

    def __init__(self, cache: CacheProtocol | None = None) -> None:
        """
        Initialize the cache manager.

        Args:
            cache: Optional cache instance. If not provided, uses SimpleCacheAdapter.
        """
        self._cache: CacheProtocol = cache if cache is not None else SimpleCacheAdapter()
        self._hits = 0
        self._misses = 0

    async def get_or_compute(
        self, key: str, compute_fn: Callable[..., Awaitable[T]], **kwargs: Any
    ) -> T:
        """
        Get value from cache or compute it if not cached.

        Args:
            key: Cache key to lookup
            compute_fn: Function to compute value if cache miss
            **kwargs: Additional arguments passed to compute_fn

        Returns:
            Cached or computed value
        """
        try:
            # Try to get from cache
            cached_value = await self._cache.get(key)
            if cached_value is not None:
                self._hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return cached_value

            # Cache miss - compute value
            self._misses += 1
            logger.debug(f"Cache miss for key: {key}")

            # Call compute function with kwargs if provided
            if kwargs:
                value = await compute_fn(**kwargs)
            else:
                value = await compute_fn()

            # Store in cache
            await self._cache.set(key, value)

            return value

        except Exception as e:
            logger.error(f"Error in cache get_or_compute: {e}")
            # Fallback to compute function on cache error
            if kwargs:
                return await compute_fn(**kwargs)
            return await compute_fn()

    def get_cache_stats(self) -> dict[str, int | float]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hit rate and cache size
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        # Get cache size if available
        cache_size = 0
        try:
            if hasattr(self._cache, "size"):
                cache_size = self._cache.size()
            elif hasattr(self._cache, "__len__"):
                cache_size = len(self._cache)
        except Exception as e:
            logger.debug(f"Could not get cache size: {e}")

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cache_size": cache_size,
            "total_requests": total_requests,
        }

    async def clear_cache(self) -> None:
        """Clear the cache and reset statistics."""
        try:
            if hasattr(self._cache, "clear"):
                await self._cache.clear()
            else:
                # Fallback for caches without clear method
                logger.warning("Cache does not support clear operation")

            # Reset statistics
            self._hits = 0
            self._misses = 0

            logger.info("Cache cleared and statistics reset")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
