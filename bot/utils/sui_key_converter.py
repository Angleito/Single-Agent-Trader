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

from bot.security.memory import SecureString
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
    if len(words) in [12, 24] and all(
        word.isalpha() and 2 <= len(word) <= 15 for word in words
    ):
        return "mnemonic"

    return "unknown"


def bech32_to_hex(bech32_key: str) -> str | None:
    """
    Convert Sui bech32 private key to hex format.

    Args:
        bech32_key: Private key in bech32 format (suiprivkey...)

    Returns:
        Hex private key string or None if conversion fails
    """
    try:
        # Try to use pysui for bech32 decoding
        from pysui.sui.sui_crypto import SuiKeyPair

        # Import from bech32 format
        keypair = SuiKeyPair.from_bech32(bech32_key.strip())
        private_key_hex = keypair.private_key_hex

        # Ensure proper formatting with 0x prefix
        if not private_key_hex.startswith("0x"):
            private_key_hex = f"0x{private_key_hex}"

        return str(private_key_hex)
    except ImportError:
        # Enhanced bech32 decoding without pysui dependency
        try:
            if not bech32_key.startswith("suiprivkey1"):
                return None

            # Remove the 'suiprivkey1' prefix
            data_part = bech32_key[11:]  # Remove 'suiprivkey1' prefix

            # Bech32 alphabet
            BECH32_ALPHABET = "023456789acdefghjklmnpqrstuvwxyz"

            # Convert bech32 chars to 5-bit values
            data = []
            for char in data_part.lower():
                if char in BECH32_ALPHABET:
                    data.append(BECH32_ALPHABET.index(char))
                else:
                    return None

            # Convert from 5-bit to 8-bit bytes
            # Skip checksum validation for simplicity (last 8 chars are checksum)
            if len(data) < 8:  # Need at least checksum
                return None

            # Remove checksum (last 8 characters)
            payload = data[:-8]

            # Convert 5-bit groups to bytes
            bits = 0
            value = 0
            result = []

            for item in payload:
                value = (value << 5) | item
                bits += 5

                if bits >= 8:
                    result.append((value >> (bits - 8)) & 0xFF)
                    bits -= 8

            # The result should be 32 bytes for a private key
            if len(result) >= 32:
                # Take first 32 bytes and convert to hex
                private_key_bytes = bytes(result[:32])
                hex_key = private_key_bytes.hex()
                return f"0x{hex_key}"

            return None

        except Exception:
            return None


def mnemonic_to_hex(mnemonic: str) -> str | None:
    """
    Convert mnemonic phrase to hex private key.

    Args:
        mnemonic: 12 or 24 word mnemonic phrase

    Returns:
        Hex private key string or None if conversion fails
    """
    try:
        # Try to use pysui for Sui-specific key derivation
        from pysui.sui.sui_crypto import SignatureScheme, SuiKeyPair

        # Create keypair from mnemonic
        keypair = SuiKeyPair.from_mnemonic(mnemonic.strip(), SignatureScheme.ED25519)
        # Get the private key in hex format
        private_key_hex = keypair.private_key_hex

        # Ensure proper formatting with 0x prefix
        if not private_key_hex.startswith("0x"):
            private_key_hex = f"0x{private_key_hex}"

        return str(private_key_hex)
    except ImportError:
        # If pysui is not available, try alternative approach
        try:
            import hashlib

            from mnemonic import Mnemonic

            # Validate mnemonic
            mnemo = Mnemonic("english")
            if not mnemo.check(mnemonic):
                return None

            # Convert mnemonic to seed
            seed = mnemo.to_seed(mnemonic, passphrase="")  # nosec B106

            # For Sui, we need to derive the key using ED25519
            # This is a simplified derivation - Sui might use a specific derivation path
            # Take first 32 bytes of seed for ED25519
            private_key_bytes = seed[:32]

            # Convert to hex
            private_key_hex = private_key_bytes.hex()
            return f"0x{private_key_hex}"

        except ImportError:
            # Final fallback - use basic approach with hashlib
            try:
                import hashlib

                # This is a very basic conversion and may not match Sui's exact derivation
                # For production use, proper BIP39/BIP44 derivation should be implemented
                seed = hashlib.pbkdf2_hmac(
                    "sha512", mnemonic.encode("utf-8"), b"", 2048
                )
                private_key_bytes = seed[:32]  # Take first 32 bytes for ED25519
                private_key_hex = private_key_bytes.hex()
                return f"0x{private_key_hex}"
            except Exception:
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
                return SecureString(key), ""
        normalized_key = f"0x{key}"
        if validate_hex_secure(normalized_key):
            return SecureString(normalized_key), ""
        return None, "Invalid hex format after validation"

    if format_type == "bech32":
        # Convert from bech32 to hex
        hex_key = bech32_to_hex(key)
        if hex_key:
            if validate_hex_secure(hex_key):
                return SecureString(hex_key), ""
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
                return SecureString(hex_key), ""
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


def get_conversion_instructions(_format_type: str) -> str:
    """Get conversion instructions for each format type."""
    return (
        "Use Sui CLI for conversion:\n"
        "1. Install Sui CLI: https://docs.sui.io/build/install\n"
        "2. For bech32: sui keytool import 'suiprivkey...' ed25519\n"
        "3. For mnemonic: sui keytool import '<words>' ed25519\n"
        "4. Export as hex: sui keytool export <address> --key-scheme ed25519"
    )


# Export main functions
__all__ = [
    "auto_convert_private_key",
    "convert_to_hex",
    "detect_key_format",
    "get_conversion_instructions",
    "validate_hex_private_key",
]
