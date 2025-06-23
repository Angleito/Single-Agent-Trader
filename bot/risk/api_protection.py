"""
API failure protection with exponential backoff.

This module provides resilient API call execution with automatic retries
and exponential backoff to handle temporary API issues.
"""

import asyncio
import logging
from datetime import UTC, datetime

from bot.types.base_types import APIHealthStatus

logger = logging.getLogger(__name__)


class APIFailureProtection:
    """
    API failure protection with exponential backoff.

    Provides resilient API call execution with automatic retries
    and exponential backoff to handle temporary API issues.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize API failure protection.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_success_time: datetime | None = None

    async def execute_with_protection(self, func, *args, **kwargs):
        """
        Execute function with failure protection and exponential backoff.

        Args:
            func: Function to execute (can be async or sync)
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Execute function (handle both sync and async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - reset failure counters
                self.consecutive_failures = 0
                self.last_success_time = datetime.now(UTC)
                return result
            except Exception as e:
                last_exception = e
                self.consecutive_failures += 1
                self.total_failures += 1

                if attempt < self.max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = self.base_delay * (2**attempt)
                    logger.warning(
                        "API call failed (attempt %s/%s): %s\nRetrying in %.1fs...",
                        attempt + 1,
                        self.max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "API call failed after %s attempts", self.max_retries
                    )

        # All attempts failed
        raise last_exception or Exception("All retry attempts exhausted")

    def get_health_status(self) -> APIHealthStatus:
        """Get API health status information."""
        return APIHealthStatus(
            consecutive_failures=self.consecutive_failures,
            total_failures=self.total_failures,
            last_success=(
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            health_score=max(0, 100 - min(100, self.consecutive_failures * 20)),
        )
