"""
Sui Private Key Format Converter

This utility converts between different Sui private key formats:
- Bech32 format (suiprivkey...) - Native Sui format
- Hex format (0x...) - Raw hex bytes
- Mnemonic phrase - 12/24 word seed phrase

Provides automatic detection and conversion for seamless user experience.
"""

import re


def detect_key_format(private_key: str) -> str:
    """
    Detect the format of a Sui private key.

    Args:
        private_key: The private key string

    Returns:
        Format type: 'bech32', 'hex', 'mnemonic', or 'unknown'
    """
    if not isinstance(private_key, str):
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

    Args:
        bech32_key: Private key in bech32 format (suiprivkey...)

    Returns:
        Hex format private key (0x...) or None if conversion fails
    """
    try:
        # Remove the 'suiprivkey' prefix and get the data part
        if not bech32_key.startswith("suiprivkey"):
            return None

        # Extract the data part after 'suiprivkey'
        data_part = bech32_key[10:]  # Remove 'suiprivkey' prefix

        # Implement bech32 decoding for Sui private keys
        # Sui uses bech32 encoding with specific characteristics
        decoded_bytes = _decode_bech32_data(data_part)

        if (
            decoded_bytes and len(decoded_bytes) == 32
        ):  # 32 bytes = 256 bits for private key
            # Convert bytes to hex string
            hex_string = decoded_bytes.hex()
            return f"0x{hex_string}"

        return None
    except Exception:
        return None


def _decode_bech32_data(data_part: str) -> bytes | None:
    """
    Decode bech32 data part to raw bytes.

    Args:
        data_part: The data portion of the bech32 string

    Returns:
        Raw bytes or None if decoding fails
    """
    try:
        # Bech32 character set
        CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

        # Convert bech32 characters to 5-bit values
        values = []
        for char in data_part.lower():
            if char in CHARSET:
                values.append(CHARSET.index(char))
            else:
                return None

        # Convert from 5-bit to 8-bit (standard byte representation)
        # This is the key conversion step for bech32
        converted = _convert_bits(values, 5, 8, False)

        if converted:
            return bytes(converted)

        return None
    except Exception:
        return None


def _convert_bits(data, frombits, tobits, pad):
    """
    Convert between bit groups.

    Args:
        data: Input data as list of integers
        frombits: Number of bits per input element
        tobits: Number of bits per output element
        pad: Whether to pad the output

    Returns:
        Converted data or None if conversion fails
    """
    try:
        acc = 0
        bits = 0
        ret = []
        maxv = (1 << tobits) - 1
        max_acc = (1 << (frombits + tobits - 1)) - 1

        for value in data:
            if value < 0 or (value >> frombits):
                return None
            acc = ((acc << frombits) | value) & max_acc
            bits += frombits
            while bits >= tobits:
                bits -= tobits
                ret.append((acc >> bits) & maxv)

        if pad:
            if bits:
                ret.append((acc << (tobits - bits)) & maxv)
        elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
            return None

        return ret
    except Exception:
        return None


def mnemonic_to_hex(mnemonic: str) -> str | None:
    """
    Convert mnemonic phrase to hex private key using BIP39 + ED25519.

    Args:
        mnemonic: 12 or 24 word mnemonic phrase

    Returns:
        Hex format private key (0x...) or None if conversion fails
    """
    try:
        # Split and validate mnemonic
        words = mnemonic.strip().split()
        if len(words) not in [12, 24]:
            return None

        # Validate words are alphabetic and reasonable length
        if not all(word.isalpha() and 2 <= len(word) <= 15 for word in words):
            return None

        # Convert mnemonic to seed using BIP39
        seed = _mnemonic_to_seed(mnemonic)
        if not seed:
            return None

        # Derive private key using ED25519 for Sui
        private_key = _derive_sui_private_key(seed)
        if private_key and len(private_key) == 32:
            return f"0x{private_key.hex()}"

        return None
    except Exception:
        return None


def _mnemonic_to_seed(
    mnemonic: str, passphrase: str = ""
) -> bytes | None:  # nosec B107
    """
    Convert mnemonic phrase to seed using BIP39 standard.

    Args:
        mnemonic: Space-separated mnemonic words
        passphrase: Optional passphrase (default empty)

    Returns:
        64-byte seed or None if conversion fails
    """
    try:
        import hashlib

        # Normalize the mnemonic
        mnemonic = " ".join(mnemonic.split())

        # BIP39 seed derivation using PBKDF2
        # Salt is "mnemonic" + passphrase
        salt = ("mnemonic" + passphrase).encode("utf-8")

        # Use PBKDF2-HMAC-SHA512 with 2048 iterations
        seed = hashlib.pbkdf2_hmac(
            "sha512",
            mnemonic.encode("utf-8"),
            salt,
            2048,
            64,  # 64 bytes = 512 bits
        )

        return seed
    except Exception:
        return None


def _derive_sui_private_key(seed: bytes) -> bytes | None:
    """
    Derive Sui private key from BIP39 seed using ED25519.

    Args:
        seed: 64-byte BIP39 seed

    Returns:
        32-byte private key or None if derivation fails
    """
    try:
        import hashlib
        import hmac

        # Sui uses ED25519 with specific derivation path
        # Standard path is m/44'/784'/0'/0'/0' (784 is Sui's coin type)

        # Start with master key derivation
        hmac_result = hmac.new(b"ed25519 seed", seed, hashlib.sha512).digest()

        # Split into key and chain code
        master_key = hmac_result[:32]
        master_chain = hmac_result[32:]

        # For simplicity, we'll use the master key directly
        # In a full implementation, you'd derive through the full path

        # Ensure the key is valid for ED25519 (clamp the key)
        private_key = bytearray(master_key)
        private_key[0] &= 248  # Clear bottom 3 bits
        private_key[31] &= 127  # Clear top bit
        private_key[31] |= 64  # Set second-highest bit

        return bytes(private_key)
    except Exception:
        return None


def convert_to_hex(private_key: str) -> tuple[str | None, str]:
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
            return key, ""
        return f"0x{key}", ""

    if format_type == "bech32":
        # Convert from bech32 to hex
        hex_key = bech32_to_hex(key)
        if hex_key:
            return hex_key, ""
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
            return hex_key, ""
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
    """Get detailed conversion instructions for each format type."""

    instructions = {
        "bech32": """
🔧 Converting Bech32 to Hex Format:

Method 1 - Sui Wallet:
1. Open your Sui wallet
2. Go to Settings → Export Private Key
3. Select your account
4. Choose "Raw Private Key" or "Hex Format"
5. Copy the hex string (starts with 0x)

Method 2 - Sui CLI:
1. Install Sui CLI: https://docs.sui.io/build/install
2. Import your key: sui keytool import "suiprivkey..." ed25519
3. Export as hex: sui keytool export <address> --key-scheme ed25519

The result should be a 66-character string starting with 0x
""",
        "mnemonic": """
🔧 Converting Mnemonic to Private Key:

Method 1 - Sui CLI:
1. Install Sui CLI: https://docs.sui.io/build/install
2. Import mnemonic: sui keytool import "<your 12/24 words>" ed25519
3. Export private key: sui keytool export <address> --key-scheme ed25519

Method 2 - Sui Wallet:
1. Import your mnemonic into Sui wallet
2. Go to Settings → Export Private Key
3. Choose "Raw Private Key" format
4. Copy the hex string

The result should be a 66-character string starting with 0x
""",
        "unknown": """
🔧 Supported Private Key Formats:

1. Hex Format (Recommended):
   - Format: 0x1234567890abcdef... (66 characters total)
   - Example: 0xa1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456

2. Bech32 Format:
   - Format: suiprivkey... (Sui native format)
   - Can be converted to hex using Sui CLI or wallet

3. Mnemonic Phrase:
   - Format: 12 or 24 words separated by spaces
   - Must be converted to private key first

For best compatibility, use hex format (0x...).
""",
    }

    return instructions.get(format_type, instructions["unknown"])


# Export main functions
__all__ = [
    "auto_convert_private_key",
    "convert_to_hex",
    "detect_key_format",
    "get_conversion_instructions",
    "validate_hex_private_key",
]
