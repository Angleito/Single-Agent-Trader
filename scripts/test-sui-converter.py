#!/usr/bin/env python3
"""Test Sui private key converter functionality."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.utils.sui_key_converter import (
    auto_convert_private_key,
    detect_key_format,
)


def test_converter():
    """Test the Sui key converter with various formats."""
    print("🧪 Sui Private Key Converter Test")
    print("=" * 50)
    print()

    # Test cases
    test_keys = [
        # Hex format (already valid)
        ("0x1234567890abcdef" * 4, "hex"),
        # Mnemonic format (12 words)
        (
            "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12",
            "mnemonic",
        ),
        # Bech32 format
        ("suiprivkey1234567890abcdef", "bech32"),
    ]

    for key, expected_format in test_keys:
        print(f"📝 Testing {expected_format} format...")
        print(f"   Input: {key[:50]}..." if len(key) > 50 else f"   Input: {key}")

        # Detect format
        detected = detect_key_format(key)
        print(f"   Detected: {detected}")

        # Try conversion
        result, format_type, message = auto_convert_private_key(key)

        if result:
            print("   ✅ Conversion successful!")
            print(f"   Output: {result[:20]}...{result[-10:]}")
        else:
            print(f"   ❌ Conversion failed: {message}")

        print()

    # Test with actual environment variable
    print("📋 Testing Environment Variable...")

    # Check legacy format
    legacy_key = os.environ.get("EXCHANGE__BLUEFIN_PRIVATE_KEY")
    if legacy_key:
        print("   Found EXCHANGE__BLUEFIN_PRIVATE_KEY")
        detected = detect_key_format(legacy_key)
        print(f"   Format: {detected}")

        if detected == "mnemonic":
            print("   🔄 Attempting automatic conversion...")
            result, format_type, message = auto_convert_private_key(legacy_key)
            if result:
                print("   ✅ Successfully converted to hex!")
                print("   You can update your .env with:")
                print(f"   BLUEFIN_PRIVATE_KEY={result}")
            else:
                print(f"   ❌ Conversion failed: {message}")

    # Check FP format
    fp_key = os.environ.get("BLUEFIN_PRIVATE_KEY")
    if fp_key:
        print("   Found BLUEFIN_PRIVATE_KEY")
        detected = detect_key_format(fp_key)
        print(f"   Format: {detected}")

        if detected == "mnemonic":
            print("   🔄 Attempting automatic conversion...")
            result, format_type, message = auto_convert_private_key(fp_key)
            if result:
                print("   ✅ Successfully converted to hex!")
            else:
                print(f"   ❌ Conversion failed: {message}")

    if not legacy_key and not fp_key:
        print("   ⚠️  No Bluefin private key found in environment")

    print()
    print("✅ Test completed!")


if __name__ == "__main__":
    test_converter()
