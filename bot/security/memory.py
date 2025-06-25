"""Secure memory handling for sensitive data."""

import ctypes
import sys


def zero_memory(data: bytes) -> None:
    """Securely zero out memory containing sensitive data.

    Args:
        data: Bytes to be zeroed out in memory
    """
    if not data:
        return

    # Get memory address and size
    address = id(data)
    size = sys.getsizeof(data) - sys.getsizeof(0)

    # Use ctypes to directly zero the memory
    ctypes.memset(address, 0, size)


class SecureString:
    """Secure string wrapper that zeros memory on deletion."""

    def __init__(self, data: str):
        """Initialize with sensitive string data.

        Args:
            data: Sensitive string to protect
        """
        self._data: str | None = data
        self._bytes: bytes | None = data.encode("utf-8")

    def get(self) -> str:
        """Get the string value (use sparingly)."""
        if self._data is None:
            raise ValueError("SecureString already cleared")
        return self._data

    def __del__(self) -> None:
        """Zero out memory when object is deleted."""
        if self._bytes is not None:
            zero_memory(self._bytes)
        self._data = None
        self._bytes = None


def secure_string(data: str) -> SecureString:
    """Create a secure string wrapper for sensitive data.

    Args:
        data: Sensitive string to wrap

    Returns:
        SecureString instance with auto-cleanup
    """
    return SecureString(data)
