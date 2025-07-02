"""Secure memory handling for sensitive data like API keys and private keys."""

import ctypes
import os
import platform


class SecureString:  # noqa: PLW1641
    """A string class that securely handles sensitive data in memory.

    This class attempts to:
    1. Prevent string interning
    2. Zero out memory when the object is deleted
    3. Provide controlled access to the underlying value
    4. Mask the value in string representations
    """

    def __init__(self, value: str | bytes):
        """Initialize SecureString with a sensitive value.

        Args:
            value: The sensitive string or bytes to protect
        """
        if isinstance(value, str):
            # Convert to bytes to have better control over memory
            self._data = bytearray(value.encode("utf-8"))
        elif isinstance(value, bytes):
            self._data = bytearray(value)
        else:
            raise TypeError("SecureString requires str or bytes")

        # Store length for later use
        self._length = len(self._data)

        # Try to lock memory pages (requires appropriate permissions)
        self._lock_memory()

    def _lock_memory(self):
        """Attempt to lock memory pages to prevent swapping to disk."""
        if platform.system() == "Linux":
            try:
                # Try to use mlock to prevent memory from being swapped
                libc = ctypes.CDLL("libc.so.6")
                libc.mlock(
                    ctypes.c_void_p(id(self._data)), ctypes.c_size_t(self._length)
                )
            except Exception:  # noqa: S110
                # If mlock fails (usually due to permissions), continue without it
                pass

    def _unlock_memory(self):
        """Unlock memory pages."""
        if platform.system() == "Linux":
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.munlock(
                    ctypes.c_void_p(id(self._data)), ctypes.c_size_t(self._length)
                )
            except Exception:  # noqa: S110
                pass

    def _zero_memory(self):
        """Overwrite the memory with zeros."""
        if hasattr(self, "_data") and self._data:
            # Multiple overwrites with different patterns for security
            for pattern in [0x00, 0xFF, 0xAA, 0x55, 0x00]:
                for i in range(len(self._data)):
                    self._data[i] = pattern

            # Final zero out
            self._data[:] = bytearray(len(self._data))

            # Try to use platform-specific secure zero
            if platform.system() in ["Linux", "Darwin"]:
                try:
                    # Use explicit_bzero if available (more secure)
                    if platform.system() == "Linux":
                        libc = ctypes.CDLL("libc.so.6")
                        libc.explicit_bzero(
                            ctypes.c_void_p(id(self._data)),
                            ctypes.c_size_t(len(self._data)),
                        )
                    elif platform.system() == "Darwin":
                        libc = ctypes.CDLL("libc.dylib")
                        libc.memset_s(
                            ctypes.c_void_p(id(self._data)),
                            ctypes.c_size_t(len(self._data)),
                            0,
                            ctypes.c_size_t(len(self._data)),
                        )
                except Exception:  # noqa: S110
                    pass

    def get_value(self) -> str:
        """Get the decrypted value. Use with caution!

        Returns:
            The sensitive string value
        """
        if not hasattr(self, "_data") or not self._data:
            return ""
        return self._data.decode("utf-8", errors="replace")

    def get_bytes(self) -> bytes:
        """Get the raw bytes. Use with caution!

        Returns:
            The sensitive bytes value
        """
        if not hasattr(self, "_data") or not self._data:
            return b""
        return bytes(self._data)

    def __str__(self) -> str:
        """Return a masked representation."""
        if not hasattr(self, "_data") or not self._data:
            return "SecureString(empty)"

        # Show only first and last 2 characters for debugging
        if self._length > 8:
            value = self.get_value()
            return f"SecureString({value[:2]}...{value[-2:]})"
        return "SecureString(***)"

    def __repr__(self) -> str:
        """Return a masked representation."""
        return f"<SecureString length={self._length}>"

    def __eq__(self, other) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        if not isinstance(other, (SecureString, str, bytes)):
            return False

        if isinstance(other, SecureString):
            other_bytes = other.get_bytes()
        elif isinstance(other, str):
            other_bytes = other.encode("utf-8")
        else:
            other_bytes = other

        my_bytes = self.get_bytes()

        # Constant-time comparison
        if len(my_bytes) != len(other_bytes):
            return False

        result = 0
        for a, b in zip(my_bytes, other_bytes, strict=False):
            result |= a ^ b

        return result == 0

    def __del__(self):
        """Securely clean up the sensitive data."""
        try:
            self._unlock_memory()
            self._zero_memory()
            # Delete the reference
            if hasattr(self, "_data"):
                del self._data
        except Exception:  # noqa: S110
            # Ensure cleanup doesn't raise exceptions during gc
            pass

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on context exit."""
        self._zero_memory()
        return False


class SecureEnvironment:
    """Utilities for securely handling environment variables."""

    @staticmethod
    def get(key: str, default: str | None = None) -> SecureString | None:
        """Get an environment variable as a SecureString.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            SecureString containing the value, or None
        """
        value = os.environ.get(key, default)
        if value is None:
            return None
        return SecureString(value)

    @staticmethod
    def set(key: str, value: str | SecureString) -> None:
        """Set an environment variable securely.

        Args:
            key: Environment variable name
            value: Value to set (will be converted to SecureString)
        """
        if isinstance(value, SecureString):
            os.environ[key] = value.get_value()
        else:
            os.environ[key] = str(value)

    @staticmethod
    def clear_sensitive() -> None:
        """Clear sensitive environment variables from memory."""
        sensitive_patterns = [
            "API_KEY",
            "PRIVATE_KEY",
            "SECRET",
            "PASSWORD",
            "TOKEN",
            "CREDENTIAL",
            "AUTH",
        ]

        for key in list(os.environ.keys()):
            for pattern in sensitive_patterns:
                if pattern in key.upper():
                    # Overwrite before deletion
                    os.environ[key] = "X" * len(os.environ[key])
                    del os.environ[key]
                    break


def secure_compare(
    a: str | bytes | SecureString, b: str | bytes | SecureString
) -> bool:
    """Constant-time comparison for sensitive values.

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if values are equal, False otherwise
    """
    if isinstance(a, SecureString):
        a_bytes = a.get_bytes()
    elif isinstance(a, str):
        a_bytes = a.encode("utf-8")
    else:
        a_bytes = a

    if isinstance(b, SecureString):
        b_bytes = b.get_bytes()
    elif isinstance(b, str):
        b_bytes = b.encode("utf-8")
    else:
        b_bytes = b

    # Length must match
    if len(a_bytes) != len(b_bytes):
        return False

    # Constant-time comparison
    result = 0
    for x, y in zip(a_bytes, b_bytes, strict=False):
        result |= x ^ y

    return result == 0
