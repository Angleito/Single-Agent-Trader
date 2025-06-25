"""Rate limiting middleware for the trading bot dashboard API."""

import logging
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter with per-IP tracking."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        block_duration: int = 300,  # 5 minutes
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Allowed requests per minute
            burst_size: Maximum burst requests allowed
            block_duration: How long to block IPs that exceed limits (seconds)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.block_duration = block_duration

        # Token buckets per IP
        self.buckets: dict[
            str, tuple[float, float, float]
        ] = {}  # ip -> (tokens, last_update, blocked_until)

        # Request history for pattern detection
        self.request_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Whitelist for internal services
        self.whitelist = {
            "127.0.0.1",
            "localhost",
            "ai-trading-bot",  # Docker service name
            "dashboard-backend",
            "dashboard-frontend",
        }

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For header (from proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return "unknown"

    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        return ip in self.whitelist or ip.startswith(("172.", "10."))

    def _refill_bucket(self, ip: str, current_time: float) -> tuple[float, float]:
        """Refill token bucket based on time elapsed."""
        if ip not in self.buckets:
            # New IP, start with full bucket
            return float(self.burst_size), current_time

        tokens, last_update, blocked_until = self.buckets[ip]

        # Check if still blocked
        if blocked_until > current_time:
            return tokens, last_update

        # Calculate tokens to add
        time_elapsed = current_time - last_update
        tokens_to_add = (time_elapsed / 60.0) * self.requests_per_minute

        # Cap at burst size
        new_tokens = min(tokens + tokens_to_add, float(self.burst_size))

        return new_tokens, current_time

    def _detect_suspicious_pattern(self, ip: str) -> bool:
        """Detect suspicious request patterns."""
        history = self.request_history[ip]

        if len(history) < 10:
            return False

        # Check for rapid-fire requests (more than 10 requests in 1 second)
        recent_requests = [t for t in history if time.time() - t < 1.0]
        if len(recent_requests) > 10:
            logger.warning(
                "Suspicious pattern detected for %s: rapid-fire requests", ip
            )
            return True

        # Check for consistent high-frequency requests
        one_minute_ago = time.time() - 60
        minute_requests = [t for t in history if t > one_minute_ago]
        if len(minute_requests) > self.requests_per_minute * 2:
            logger.warning(
                "Suspicious pattern detected for %s: sustained high frequency", ip
            )
            return True

        return False

    async def check_rate_limit(self, request: Request) -> JSONResponse | None:
        """
        Check if request should be rate limited.

        Returns:
            None if request is allowed, JSONResponse with 429 status if limited
        """
        ip = self._get_client_ip(request)

        # Skip rate limiting for whitelisted IPs
        if self._is_whitelisted(ip):
            return None

        current_time = time.time()

        # Record request
        self.request_history[ip].append(current_time)

        # Get current bucket state
        tokens, last_update = self._refill_bucket(ip, current_time)

        # Check if blocked
        if ip in self.buckets:
            _, _, blocked_until = self.buckets[ip]
            if blocked_until > current_time:
                retry_after = int(blocked_until - current_time)
                logger.warning(
                    "Blocked IP %s attempted request. Blocked for %ss", ip, retry_after
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too many requests. IP temporarily blocked.",
                        "retry_after": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(blocked_until)),
                    },
                )

        # Check for suspicious patterns
        if self._detect_suspicious_pattern(ip):
            # Block the IP
            blocked_until = current_time + self.block_duration
            self.buckets[ip] = (0, current_time, blocked_until)
            logger.warning(
                "Blocking IP %s for %ss due to suspicious pattern",
                ip,
                self.block_duration,
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Suspicious request pattern detected. IP temporarily blocked.",
                    "retry_after": self.block_duration,
                },
                headers={
                    "Retry-After": str(self.block_duration),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(blocked_until)),
                },
            )

        # Check if we have tokens
        if tokens >= 1:
            # Consume a token
            self.buckets[ip] = (tokens - 1, current_time, 0)

            # Add rate limit headers
            request.state.rate_limit_headers = {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": str(int(tokens - 1)),
                "X-RateLimit-Reset": str(int(current_time + 60)),
            }

            return None
        # No tokens available
        retry_after = 60 - int(current_time - last_update)
        logger.info("Rate limit exceeded for IP %s", ip)

        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded. Please try again later.",
                "retry_after": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(current_time + retry_after)),
            },
        )


# Global rate limiter instances for different endpoints
api_limiter = RateLimiter(requests_per_minute=60, burst_size=10)
websocket_limiter = RateLimiter(requests_per_minute=20, burst_size=5)
health_limiter = RateLimiter(requests_per_minute=10, burst_size=2)


async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting."""
    path = request.url.path

    # Choose appropriate limiter based on path
    if path.startswith("/ws"):
        limiter = websocket_limiter
    elif path == "/health":
        limiter = health_limiter
    else:
        limiter = api_limiter

    # Check rate limit
    rate_limit_response = await limiter.check_rate_limit(request)
    if rate_limit_response:
        return rate_limit_response

    # Process request
    response = await call_next(request)

    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit_headers"):
        for header, value in request.state.rate_limit_headers.items():
            response.headers[header] = value

    return response
