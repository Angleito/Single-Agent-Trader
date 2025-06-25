"""Security validation functions for critical operations."""

import re
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


def validate_hex_key(key: str) -> bool:
    """Validate hex format private keys (32 or 64 hex chars)."""
    if not key or not isinstance(key, str):
        return False
    clean_key = key.strip().lower()
    clean_key = clean_key.removeprefix("0x")

    # Must be exactly 64 hex characters
    if not re.match(r"^[0-9a-f]{64}$", clean_key):
        return False

    # Reject invalid private keys (all zeros, all F's)
    return clean_key not in ("0" * 64, "f" * 64)


def validate_bip39(mnemonic: str) -> bool:
    """Basic BIP39 mnemonic validation (12/24 words)."""
    if not mnemonic or not isinstance(mnemonic, str):
        return False
    words = mnemonic.strip().lower().split()
    return len(words) in (12, 24) and all(
        word.isalpha() and 3 <= len(word) <= 8 for word in words
    )


_rate_limits: dict[str, tuple[float, int]] = {}


def rate_limit(
    calls: int = 10, period: int = 60
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for rate limiting function calls."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = func.__name__
            now = time.time()
            if key in _rate_limits:
                last_reset, count = _rate_limits[key]
                if now - last_reset < period:
                    if count >= calls:
                        raise RuntimeError(f"Rate limit exceeded: {calls}/{period}s")
                    _rate_limits[key] = (last_reset, count + 1)
                else:
                    _rate_limits[key] = (now, 1)
            else:
                _rate_limits[key] = (now, 1)
            return func(*args, **kwargs)

        return wrapper

    return decorator
