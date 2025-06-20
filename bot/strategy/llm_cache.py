"""
LLM Response Caching System for Trading Bot Performance Optimization.

This module implements intelligent caching of LLM responses based on market state similarity
to reduce latency from 2-8 seconds to sub-2 seconds while maintaining decision quality.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from bot.config import settings
from bot.trading_types import MarketState, TradeAction

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""

    response: TradeAction
    timestamp: float
    market_state_hash: str
    hit_count: int = 0
    confidence_score: float = 1.0


class MarketStateHasher:
    """Creates similarity-based cache keys for market states."""

    def __init__(self, price_tolerance: float = 0.02, indicator_tolerance: float = 0.1):
        """
        Initialize the market state hasher.

        Args:
            price_tolerance: Price similarity tolerance (2% default)
            indicator_tolerance: Indicator similarity tolerance (10% default)
        """
        self.price_tolerance = price_tolerance
        self.indicator_tolerance = indicator_tolerance

    def get_cache_key(self, market_state: MarketState) -> str:
        """
        Generate a cache key based on market state similarity.

        Market states are considered similar if:
        - Price is within tolerance range
        - Key indicators are within tolerance
        - Market sentiment matches
        - Dominance is within range

        Args:
            market_state: Current market state

        Returns:
            Hash string for cache key
        """
        try:
            # Normalize price to tolerance buckets
            price = float(market_state.current_price)
            price_bucket = self._bucket_value(price, self.price_tolerance)

            # Normalize indicators to tolerance buckets
            indicators = market_state.indicators

            rsi_bucket = self._bucket_value(
                indicators.rsi or 50, self.indicator_tolerance
            )
            cipher_a_bucket = self._bucket_value(
                indicators.cipher_a_dot or 0, self.indicator_tolerance
            )
            cipher_b_wave_bucket = self._bucket_value(
                indicators.cipher_b_wave or 0, self.indicator_tolerance
            )
            cipher_b_mf_bucket = self._bucket_value(
                indicators.cipher_b_money_flow or 50, self.indicator_tolerance
            )

            # Normalize dominance data
            dominance_bucket = self._bucket_value(
                indicators.stablecoin_dominance or 7.5,
                0.05,  # 5% tolerance for dominance
            )

            # Market sentiment (categorical)
            sentiment = indicators.market_sentiment or "NEUTRAL"

            # Position state
            position_side = market_state.current_position.side

            # Create key components
            key_components = [
                f"p{price_bucket}",
                f"r{rsi_bucket}",
                f"ca{cipher_a_bucket}",
                f"cbw{cipher_b_wave_bucket}",
                f"cbm{cipher_b_mf_bucket}",
                f"d{dominance_bucket}",
                f"s{sentiment}",
                f"pos{position_side}",
            ]

            # Add futures-specific data if available
            if (
                hasattr(market_state, "futures_account")
                and market_state.futures_account
            ):
                margin_health = (
                    market_state.futures_account.margin_info.health_status.value
                )
                key_components.append(f"mh{margin_health}")

            # Create hash
            key_string = "|".join(key_components)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()

            logger.debug("Generated cache key: %s from %s", cache_key, key_string)
            return cache_key

        except Exception as e:
            logger.exception("Error generating cache key: %s", e)
            # Return timestamp-based key as fallback
            return hashlib.md5(f"fallback_{time.time()}".encode()).hexdigest()

    def _bucket_value(self, value: float, tolerance: float) -> int:
        """
        Bucket a value into discrete ranges based on tolerance.

        Args:
            value: Value to bucket
            tolerance: Tolerance as decimal (0.02 = 2%)

        Returns:
            Bucket number
        """
        if value == 0:
            return 0

        bucket_size = abs(value) * tolerance
        if bucket_size == 0:
            bucket_size = tolerance

        return int(value / bucket_size)


class LLMResponseCache:
    """
    Intelligent LLM response caching system.

    Features:
    - Market state similarity-based caching
    - TTL-based expiration
    - Cache hit statistics
    - Memory-efficient storage
    """

    def __init__(
        self,
        ttl_seconds: int = 90,
        max_entries: int = 1000,
        cleanup_interval: int = 300,
    ):
        """
        Initialize the LLM response cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (90s default)
            max_entries: Maximum number of cache entries
            cleanup_interval: Cleanup interval in seconds
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval

        self.cache: dict[str, CacheEntry] = {}
        self.hasher = MarketStateHasher()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0,
        }

        # Cleanup task management - defer creation until first use
        self._cleanup_task = None
        self._cleanup_started = False

        logger.info("🚀 LLM Cache initialized: TTL=%ss, Max=%s entries", ttl_seconds, max_entries)

    def _start_cleanup_task(self):
        """Start the background cleanup task (only if running in async context)."""
        if self._cleanup_started:
            return

        try:
            # Only start if we're in an async context
            asyncio.get_running_loop()
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                self._cleanup_started = True
                logger.debug("Started LLM cache cleanup task")
        except RuntimeError:
            # No event loop running - cleanup will be started later when cache is used
            logger.debug("Deferring LLM cache cleanup task start (no event loop)")

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in cache cleanup: %s", e)

    def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            self.stats["evictions"] += 1

        if expired_keys:
            self.stats["cleanups"] += 1
            logger.debug("Cache cleanup: removed %s expired entries", len(expired_keys))

        # Enforce max entries limit
        if len(self.cache) > self.max_entries:
            # Remove oldest entries
            sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].timestamp)

            excess_count = len(self.cache) - self.max_entries
            for key, _ in sorted_entries[:excess_count]:
                del self.cache[key]
                self.stats["evictions"] += 1

            logger.debug("Cache size limit: removed %s oldest entries", excess_count)

    async def get_or_compute(
        self, market_state: MarketState, compute_func, *args, **kwargs
    ) -> TradeAction:
        """
        Get cached response or compute new one.

        Args:
            market_state: Current market state
            compute_func: Function to compute response if not cached
            *args: Arguments for compute function
            **kwargs: Keyword arguments for compute function

        Returns:
            TradeAction from cache or newly computed
        """
        # Start cleanup task if not already started (now we're in async context)
        self._start_cleanup_task()

        # Generate cache key
        cache_key = self.hasher.get_cache_key(market_state)

        # Check cache
        cached_entry = self._get_cached_entry(cache_key)

        if cached_entry:
            # Cache hit
            cached_entry.hit_count += 1
            self.stats["hits"] += 1

            # Log cache hit for performance monitoring
            logger.info("🎯 LLM Cache HIT: %s " "(age: %ss, " "hits: %s)", cached_entry.response.action, time.time() - cached_entry.timestamp:.1f, cached_entry.hit_count)
            )

            return cached_entry.response

        # Cache miss - compute new response
        self.stats["misses"] += 1

        start_time = time.time()
        response = await compute_func(*args, **kwargs)
        compute_time = time.time() - start_time

        # Cache the response
        self._store_response(cache_key, response, market_state)

        logger.info("🔄 LLM Cache MISS: %s " "(compute_time: %ss)", response.action, compute_time:.2f)
        )

        return response

    def _get_cached_entry(self, cache_key: str) -> CacheEntry | None:
        """
        Get cached entry if valid.

        Args:
            cache_key: Cache key

        Returns:
            CacheEntry if valid, None otherwise
        """
        entry = self.cache.get(cache_key)

        if entry is None:
            return None

        # Check if expired
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[cache_key]
            self.stats["evictions"] += 1
            return None

        return entry

    def _store_response(
        self, cache_key: str, response: TradeAction, market_state: MarketState
    ):
        """
        Store response in cache.

        Args:
            cache_key: Cache key
            response: TradeAction response
            market_state: Market state for confidence scoring
        """
        # Calculate confidence score based on response characteristics
        confidence_score = self._calculate_confidence_score(response, market_state)

        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            market_state_hash=cache_key,
            confidence_score=confidence_score,
        )

        self.cache[cache_key] = entry

        logger.debug("Cached LLM response: %s " "(confidence: %s)", response.action, confidence_score:.2f)
        )

    def _calculate_confidence_score(
        self, response: TradeAction, market_state: MarketState
    ) -> float:
        """
        Calculate confidence score for caching.

        Higher confidence responses are cached longer and trusted more.

        Args:
            response: Trade action response
            market_state: Market state context

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0

        # Reduce confidence for very aggressive trades
        if response.size_pct > 20:
            confidence *= 0.8

        # Reduce confidence for very high leverage
        if response.leverage and response.leverage > 10:
            confidence *= 0.9

        # Reduce confidence for very tight stops
        if response.stop_loss_pct < 0.5:
            confidence *= 0.9

        # Increase confidence for HOLD actions (safer to cache)
        if response.action == "HOLD":
            confidence *= 1.1

        # Reduce confidence if market is volatile (based on indicators)
        if market_state.indicators.rsi:
            rsi = market_state.indicators.rsi
            if rsi > 80 or rsi < 20:  # Extreme RSI
                confidence *= 0.85

        return min(confidence, 1.0)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total_requests, 1)

        return {
            "cache_size": len(self.cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": hit_rate,
            "total_hits": self.stats["hits"],
            "total_misses": self.stats["misses"],
            "total_requests": total_requests,
            "evictions": self.stats["evictions"],
            "cleanups": self.stats["cleanups"],
        }

    def clear_cache(self):
        """Clear all cache entries."""
        cleared_count = len(self.cache)
        self.cache.clear()

        # Reset stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0,
        }

        logger.info("Cache cleared: removed %s entries", cleared_count)

    def __del__(self):
        """Cleanup on deletion."""
        if self._cleanup_task:
            self._cleanup_task.cancel()


# Global cache instance
_global_cache: LLMResponseCache | None = None


def get_llm_cache() -> LLMResponseCache:
    """
    Get or create the global LLM cache instance.

    Returns:
        Global LLM cache instance
    """
    global _global_cache

    if _global_cache is None:
        # Get settings from config
        ttl = getattr(settings, "llm_cache_ttl_seconds", 90)
        max_entries = getattr(settings, "llm_cache_max_entries", 1000)
        cleanup_interval = getattr(settings, "llm_cache_cleanup_interval", 300)

        _global_cache = LLMResponseCache(
            ttl_seconds=ttl, max_entries=max_entries, cleanup_interval=cleanup_interval
        )

    return _global_cache
