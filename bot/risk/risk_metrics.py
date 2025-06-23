"""
Cached risk metric calculations.

This module provides optimized risk metric calculations with caching
for performance-critical operations.
"""

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """
    Calculator for risk metrics with caching support.

    Provides cached calculations for frequently-used risk metrics
    to improve performance in high-frequency trading scenarios.
    """

    def __init__(self, cache_size: int = 128):
        """
        Initialize the risk metrics calculator.

        Args:
            cache_size: Maximum number of cached entries for LRU cache
        """
        self.cache_size = cache_size
        self._cache_timestamp = datetime.now(UTC)
        self._ttl_seconds = 60  # Cache TTL for time-sensitive metrics

        logger.info(
            "RiskMetricsCalculator initialized with cache_size=%d, TTL=%ds",
            cache_size,
            self._ttl_seconds,
        )

    def _is_cache_expired(self) -> bool:
        """Check if the cache TTL has expired for time-sensitive metrics."""
        return (
            datetime.now(UTC) - self._cache_timestamp
        ).total_seconds() > self._ttl_seconds

    def _refresh_cache_if_needed(self) -> None:
        """Refresh cache timestamp and clear caches if TTL expired."""
        if self._is_cache_expired():
            # Clear time-sensitive caches
            self.calculate_rapid_loss_percentage.cache_clear()
            self.calculate_margin_usage.cache_clear()
            self.calculate_daily_loss_percentage.cache_clear()
            self._cache_timestamp = datetime.now(UTC)
            logger.debug("Cache refreshed due to TTL expiration")

    @lru_cache(maxsize=32)
    def calculate_rapid_loss_percentage(
        self,
        risk_metrics_history: tuple[tuple[str, Any], ...],
        account_balance: Decimal,
        timeframe_minutes: int = 5,
    ) -> float:
        """
        Calculate rapid loss percentage over specified timeframe.

        Args:
            risk_metrics_history: Tuple of (timestamp, metrics) tuples for immutability
            account_balance: Current account balance
            timeframe_minutes: Timeframe in minutes to calculate loss over

        Returns:
            Loss percentage as float (0.0 to 1.0)
        """
        try:
            if not risk_metrics_history:
                return 0.0

            # Convert tuple back to list for processing
            metrics_list = [
                {"timestamp": datetime.fromisoformat(t), **m}
                for t, m in risk_metrics_history
            ]

            cutoff_time = datetime.now(UTC) - timedelta(minutes=timeframe_minutes)
            recent_metrics = [
                m
                for m in metrics_list[-10:]  # Last 10 samples
                if m["timestamp"] >= cutoff_time
            ]

            if len(recent_metrics) < 2:
                return 0.0

            # Calculate loss over the period
            initial_balance = recent_metrics[0].get(
                "account_balance", float(account_balance)
            )
            current_balance = float(account_balance)

            if initial_balance <= 0:
                return 0.0

            loss_percentage = max(
                0, (initial_balance - current_balance) / initial_balance
            )

            if loss_percentage > 0:
                logger.debug(
                    "Rapid loss calculated: %.2f%% over %d minutes",
                    loss_percentage * 100,
                    timeframe_minutes,
                )

            return loss_percentage

        except Exception:
            logger.exception("Error calculating rapid loss percentage")
            return 0.0

    @lru_cache(maxsize=16)
    def calculate_margin_usage(
        self, used_margin_pct: float, total_margin: float = 100.0
    ) -> float:
        """
        Calculate current margin usage percentage.

        Args:
            used_margin_pct: Used margin as percentage
            total_margin: Total available margin (default 100%)

        Returns:
            Margin usage as float (0.0 to 1.0)
        """
        try:
            if total_margin <= 0:
                return 0.0

            available_margin = total_margin - used_margin_pct
            if available_margin >= total_margin:
                return 0.0

            margin_usage = (total_margin - available_margin) / total_margin

            if margin_usage > 0.7:  # Log warning for high margin usage
                logger.warning("High margin usage detected: %.1f%%", margin_usage * 100)

            return min(1.0, max(0.0, margin_usage))

        except Exception:
            logger.exception("Error calculating margin usage")
            return 0.0

    @lru_cache(maxsize=32)
    def calculate_daily_loss_percentage(
        self,
        daily_realized_pnl: Decimal,
        daily_unrealized_pnl: Decimal,
        account_balance: Decimal,
    ) -> float:
        """
        Calculate daily loss as percentage of account balance.

        Args:
            daily_realized_pnl: Realized P&L for the day
            daily_unrealized_pnl: Unrealized P&L for the day
            account_balance: Current account balance

        Returns:
            Daily loss percentage as float (0.0 to 1.0)
        """
        try:
            if account_balance <= 0:
                return 0.0

            total_pnl = daily_realized_pnl + daily_unrealized_pnl

            # Only return positive values for losses
            daily_loss_pct = max(0, float(-total_pnl / account_balance))

            if daily_loss_pct > 0.02:  # Log if daily loss exceeds 2%
                logger.warning(
                    "Significant daily loss: %.2f%% (Realized: %s, Unrealized: %s)",
                    daily_loss_pct * 100,
                    daily_realized_pnl,
                    daily_unrealized_pnl,
                )

            return daily_loss_pct

        except Exception:
            logger.exception("Error calculating daily loss percentage")
            return 0.0

    @lru_cache(maxsize=64)
    def calculate_expected_pnl(
        self,
        position_side: str,
        position_size: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
    ) -> Decimal:
        """
        Calculate expected unrealized P&L for position validation.

        Args:
            position_side: Position side (LONG/SHORT/FLAT)
            position_size: Size of the position
            entry_price: Entry price of the position
            current_price: Current market price

        Returns:
            Expected P&L as Decimal
        """
        try:
            if position_side == "FLAT" or entry_price is None or entry_price <= 0:
                return Decimal(0)

            price_diff = current_price - entry_price

            if position_side == "LONG":
                expected_pnl = position_size * price_diff
            elif position_side == "SHORT":
                expected_pnl = position_size * (-price_diff)
            else:
                logger.error("Invalid position side: %s", position_side)
                return Decimal(0)

            logger.debug(
                "Expected P&L for %s position: %s (entry: %s, current: %s)",
                position_side,
                expected_pnl,
                entry_price,
                current_price,
            )

            return expected_pnl

        except Exception:
            logger.exception("Error calculating expected P&L")
            return Decimal(0)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": self.cache_size,
            "ttl_seconds": self._ttl_seconds,
            "cache_age_seconds": (
                datetime.now(UTC) - self._cache_timestamp
            ).total_seconds(),
            "rapid_loss_cache": {
                "hits": self.calculate_rapid_loss_percentage.cache_info().hits,
                "misses": self.calculate_rapid_loss_percentage.cache_info().misses,
                "size": self.calculate_rapid_loss_percentage.cache_info().currsize,
            },
            "margin_usage_cache": {
                "hits": self.calculate_margin_usage.cache_info().hits,
                "misses": self.calculate_margin_usage.cache_info().misses,
                "size": self.calculate_margin_usage.cache_info().currsize,
            },
            "daily_loss_cache": {
                "hits": self.calculate_daily_loss_percentage.cache_info().hits,
                "misses": self.calculate_daily_loss_percentage.cache_info().misses,
                "size": self.calculate_daily_loss_percentage.cache_info().currsize,
            },
            "expected_pnl_cache": {
                "hits": self.calculate_expected_pnl.cache_info().hits,
                "misses": self.calculate_expected_pnl.cache_info().misses,
                "size": self.calculate_expected_pnl.cache_info().currsize,
            },
        }

    def clear_all_caches(self) -> None:
        """Clear all LRU caches."""
        self.calculate_rapid_loss_percentage.cache_clear()
        self.calculate_margin_usage.cache_clear()
        self.calculate_daily_loss_percentage.cache_clear()
        self.calculate_expected_pnl.cache_clear()
        self._cache_timestamp = datetime.now(UTC)
        logger.info("All caches cleared")


# Global instance for easy access
risk_metrics_calculator = RiskMetricsCalculator()
