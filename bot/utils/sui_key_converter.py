"""
Sui Private Key Format Converter

This utility converts between different Sui private key formats:
- Bech32 format (suiprivkey...) - Native Sui format
- Hex format (0x...) - Raw hex bytes
- Mnemonic phrase - 12/24 word seed phrase

Provides automatic detection and conversion for seamless user experience.
"""

import re
from typing import Any

from bot.security.memory import SecureString, secure_string
from bot.security.validation import validate_hex_key as validate_hex_secure


def detect_key_format(private_key: Any) -> str:
    """
    Detect the format of a Sui private key.

    Args:
        private_key: The private key string

    Returns:
        Format type: 'bech32', 'hex', 'mnemonic', or 'unknown'
    """
    if not private_key or not isinstance(private_key, str):
        return "unknown"

    key = private_key.strip()

    # Bech32 format (suiprivkey...)
    if key.startswith("suiprivkey"):
        return "bech32"

    # Hex format (0x... or raw hex)
    if (key.startswith("0x") and len(key) == 66) or re.match(
        r"^[0-9a-fA-F]{64}$", key
    ):  # 0x + 64 hex chars
        return "hex"

    # Mnemonic phrase (12 or 24 words)
    words = key.split()
    if len(words) in [12, 24]:
        # Check if all words are alphabetic and reasonable length
        if all(word.isalpha() and 2 <= len(word) <= 15 for word in words):
            return "mnemonic"

    return "unknown"


def bech32_to_hex(bech32_key: str) -> str | None:
    """
    Convert Sui bech32 private key to hex format.

    Note: This is a simplified placeholder. For full bech32 support,
    use the Sui CLI or appropriate library.

    Args:
        bech32_key: Private key in bech32 format (suiprivkey...)

    Returns:
        None (conversion not supported in simplified implementation)
    """
    # Simplified implementation - recommend using Sui CLI for conversion
    return None


def mnemonic_to_hex(mnemonic: str) -> str | None:
    """
    Convert mnemonic phrase to hex private key.

    Note: This is a simplified placeholder. For full mnemonic support,
    use the Sui CLI or appropriate cryptographic library.

    Args:
        mnemonic: 12 or 24 word mnemonic phrase

    Returns:
        None (conversion not supported in simplified implementation)
    """
    # Simplified implementation - recommend using Sui CLI for conversion
    return None


def convert_to_hex(private_key: str) -> tuple[SecureString | None, str]:
    """
    Convert any supported private key format to hex.

    Args:
        private_key: Private key in any supported format

    Returns:
        Tuple of (converted_hex_key, error_message)
        If successful: (hex_key, "")
        If failed: (None, error_message)
    """
    if not private_key or not isinstance(private_key, str):
        return None, "Private key is empty or invalid"

    key = private_key.strip()
    format_type = detect_key_format(key)

    if format_type == "hex":
        # Already in hex format, just ensure proper prefix
        if key.startswith("0x"):
            # Additional validation for hex format
            if validate_hex_secure(key):
                return secure_string(key), ""
        normalized_key = f"0x{key}"
        if validate_hex_secure(normalized_key):
            return secure_string(normalized_key), ""
        return None, "Invalid hex format after validation"

    if format_type == "bech32":
        # Convert from bech32 to hex
        hex_key = bech32_to_hex(key)
        if hex_key:
            if validate_hex_secure(hex_key):
                return secure_string(hex_key), ""
        return None, (
            "Failed to convert bech32 key to hex format. "
            "Please convert your Sui wallet private key to hex format manually. "
            "You can do this in your Sui wallet by exporting the private key in hex format, "
            "or use the Sui CLI: 'sui keytool export <address> --key-scheme ed25519'"
        )

    if format_type == "mnemonic":
        # Convert from mnemonic to hex
        hex_key = mnemonic_to_hex(key)
        if hex_key:
            if validate_hex_secure(hex_key):
                return secure_string(hex_key), ""
        return None, (
            "Failed to convert mnemonic phrase to private key. "
            "Please ensure your mnemonic is valid (12 or 24 words) and try again, "
            "or use the Sui CLI: 'sui keytool import \"<mnemonic>\" ed25519'"
        )

    return None, (
        "Unknown private key format. Expected formats:\n"
        "- Hex: 0x1234567890abcdef... (64 characters)\n"
        "- Bech32: suiprivkey... (Sui native format)\n"
        "- Mnemonic: 12 or 24 word phrase"
    )


def validate_hex_private_key(hex_key: str) -> tuple[bool, str]:
    """
    Validate a hex format private key.

    Args:
        hex_key: Private key in hex format

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not hex_key or not isinstance(hex_key, str):
        return False, "Private key is empty"

    key = hex_key.strip()

    # Check for proper hex format
    if key.startswith("0x"):
        if len(key) != 66:  # 0x + 64 hex chars
            return (
                False,
                f"Hex private key must be 66 characters (0x + 64 hex), got {len(key)}",
            )
        hex_part = key[2:]
    else:
        if len(key) != 64:  # 64 hex chars
            return False, f"Hex private key must be 64 characters, got {len(key)}"
        hex_part = key

    # Validate hex characters
    if not re.match(r"^[0-9a-fA-F]+$", hex_part):
        return False, "Private key contains invalid characters (must be hexadecimal)"

    # Check for obvious invalid keys
    if hex_part == "0" * 64:
        return False, "Private key cannot be all zeros"

    if hex_part.upper() == "F" * 64:
        return False, "Private key cannot be all F's"

    return True, ""


def auto_convert_private_key(private_key: str) -> tuple[str | None, str, str]:
    """
    Automatically detect and convert private key to hex format.

    Args:
        private_key: Private key in any supported format

    Returns:
        Tuple of (converted_key, format_detected, message)
    """
    if not private_key:
        return None, "empty", "Private key is required"

    format_type = detect_key_format(private_key)

    if format_type == "hex":
        # Validate and normalize hex format
        is_valid, error = validate_hex_private_key(private_key)
        if is_valid:
            # Ensure proper 0x prefix
            normalized = (
                private_key if private_key.startswith("0x") else f"0x{private_key}"
            )
            return normalized, format_type, "✅ Hex format private key validated"
        return None, format_type, f"❌ Invalid hex private key: {error}"

    if format_type == "bech32":
        # Attempt automatic conversion from bech32
        hex_key = bech32_to_hex(private_key)
        if hex_key:
            return (
                hex_key,
                format_type,
                "✅ Bech32 private key automatically converted to hex",
            )
        return (
            None,
            format_type,
            (
                "🔧 Bech32 format detected but conversion failed. Please convert manually:\n"
                "1. Open your Sui wallet → Settings → Export Private Key\n"
                "2. Choose 'Raw Private Key' or 'Hex Format'\n"
                "3. Copy the hex string (should start with 0x)\n"
                "4. Update your .env file with the hex format key"
            ),
        )

    if format_type == "mnemonic":
        # Attempt automatic conversion from mnemonic
        hex_key = mnemonic_to_hex(private_key)
        if hex_key:
            return (
                hex_key,
                format_type,
                "✅ Mnemonic phrase automatically converted to private key",
            )
        return (
            None,
            format_type,
            (
                "🔧 Mnemonic phrase detected but conversion failed. Please ensure it's valid:\n"
                "1. Check that you have 12 or 24 words\n"
                "2. Ensure words are lowercase and spelled correctly\n"
                '3. Alternatively, use Sui CLI: sui keytool import "<mnemonic>" ed25519\n'
                "4. Then export as hex: sui keytool export <address> --key-scheme ed25519"
            ),
        )

    return (
        None,
        format_type,
        (
            "❌ Unknown private key format. Supported formats:\n"
            "• Hex: 0x1234...abcd (64 hex characters with 0x prefix)\n"
            "• Bech32: suiprivkey... (Sui wallet export format)\n"
            "• Mnemonic: 12 or 24 word seed phrase"
        ),
    )


def get_conversion_instructions(format_type: str) -> str:
    """Get conversion instructions for each format type."""
    base_instruction = (
        "Use Sui CLI for conversion:\n"
        "1. Install Sui CLI: https://docs.sui.io/build/install\n"
        "2. For bech32: sui keytool import 'suiprivkey...' ed25519\n"
        "3. For mnemonic: sui keytool import '<words>' ed25519\n"
        "4. Export as hex: sui keytool export <address> --key-scheme ed25519"
    )
    return base_instruction


# Export main functions
__all__ = [
    "auto_convert_private_key",
    "convert_to_hex",
    "detect_key_format",
    "get_conversion_instructions",
    "validate_hex_private_key",
]
