#!/usr/bin/env python3
"""
Test CDP Integration

This script tests the CDP API key integration with the Coinbase client.
It validates that both legacy and CDP authentication methods work correctly.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from bot
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import SecretStr

from bot.config import ExchangeSettings
from bot.exchange.coinbase import CoinbaseClient


def test_legacy_credentials():
    """Test legacy API key validation."""
    print("Testing legacy credentials...")

    # Test with legacy credentials
    exchange_settings = ExchangeSettings(
        cb_api_key=SecretStr("test_api_key"),
        cb_api_secret=SecretStr("test_api_secret"),
        cb_passphrase=SecretStr("test_passphrase"),
    )

    # Create client with legacy credentials
    client = CoinbaseClient(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
    )

    assert client.auth_method == "legacy"
    assert client.api_key == "test_api_key"
    assert client.api_secret == "test_api_secret"
    assert client.passphrase == "test_passphrase"
    assert client.cdp_api_key_name is None
    assert client.cdp_private_key is None

    print("✓ Legacy credentials test passed")


def test_cdp_credentials():
    """Test CDP API key validation."""
    print("Testing CDP credentials...")

    # Test with CDP credentials
    test_private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIFakePrivateKeyForTestingPurposesOnlyNotRealKey123456789ABCDEF
GHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=
-----END EC PRIVATE KEY-----"""

    exchange_settings = ExchangeSettings(
        cdp_api_key_name=SecretStr("organizations/test-org/apiKeys/test-key-id"),
        cdp_private_key=SecretStr(test_private_key),
    )

    # Create client with CDP credentials
    client = CoinbaseClient(
        cdp_api_key_name="organizations/test-org/apiKeys/test-key-id",
        cdp_private_key=test_private_key,
    )

    assert client.auth_method == "cdp"
    assert client.api_key is None
    assert client.api_secret is None
    assert client.passphrase is None
    assert client.cdp_api_key_name == "organizations/test-org/apiKeys/test-key-id"
    assert client.cdp_private_key == test_private_key

    print("✓ CDP credentials test passed")


def test_no_credentials():
    """Test client with no credentials (dry run mode)."""
    print("Testing no credentials (dry run mode)...")

    # Create client with no credentials
    client = CoinbaseClient()

    assert client.auth_method == "none"
    assert client.api_key is None
    assert client.api_secret is None
    assert client.passphrase is None
    assert client.cdp_api_key_name is None
    assert client.cdp_private_key is None

    print("✓ No credentials test passed")


def test_mixed_credentials_error():
    """Test that providing both legacy and CDP credentials raises an error."""
    print("Testing mixed credentials validation...")

    test_private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIFakePrivateKeyForTestingPurposesOnlyNotRealKey123456789ABCDEF
GHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=
-----END EC PRIVATE KEY-----"""

    try:
        # This should raise a validation error
        exchange_settings = ExchangeSettings(
            cb_api_key=SecretStr("test_api_key"),
            cb_api_secret=SecretStr("test_api_secret"),
            cb_passphrase=SecretStr("test_passphrase"),
            cdp_api_key_name=SecretStr("organizations/test-org/apiKeys/test-key-id"),
            cdp_private_key=SecretStr(test_private_key),
        )
        assert False, "Should have raised validation error for mixed credentials"
    except ValueError as e:
        assert "Cannot use both legacy and CDP credentials" in str(e)
        print("✓ Mixed credentials validation test passed")


def test_invalid_cdp_private_key():
    """Test validation of CDP private key format."""
    print("Testing invalid CDP private key validation...")

    try:
        # This should raise a validation error
        exchange_settings = ExchangeSettings(
            cdp_api_key_name=SecretStr("organizations/test-org/apiKeys/test-key-id"),
            cdp_private_key=SecretStr("invalid_private_key_format"),
        )
        assert False, "Should have raised validation error for invalid private key"
    except ValueError as e:
        assert "CDP private key must be in PEM format" in str(e)
        print("✓ Invalid CDP private key validation test passed")


def test_connection_status():
    """Test connection status reporting for different auth methods."""
    print("Testing connection status reporting...")

    # Test legacy client status
    legacy_client = CoinbaseClient(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
    )

    status = legacy_client.get_connection_status()
    assert status["auth_method"] == "legacy"
    assert status["has_credentials"] == True

    # Test CDP client status
    test_private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIFakePrivateKeyForTestingPurposesOnlyNotRealKey123456789ABCDEF
GHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=
-----END EC PRIVATE KEY-----"""

    cdp_client = CoinbaseClient(
        cdp_api_key_name="organizations/test-org/apiKeys/test-key-id",
        cdp_private_key=test_private_key,
    )

    status = cdp_client.get_connection_status()
    assert status["auth_method"] == "cdp"
    assert status["has_credentials"] == True

    # Test no credentials status
    no_creds_client = CoinbaseClient()
    status = no_creds_client.get_connection_status()
    assert status["auth_method"] == "none"
    assert status["has_credentials"] == False

    print("✓ Connection status reporting test passed")


def main():
    """Run all CDP integration tests."""
    print("Running CDP Integration Tests")
    print("=" * 50)

    try:
        test_legacy_credentials()
        test_cdp_credentials()
        test_no_credentials()
        test_mixed_credentials_error()
        test_invalid_cdp_private_key()
        test_connection_status()

        print("\n" + "=" * 50)
        print("🎉 All CDP integration tests passed!")
        print("\nCDP API key support has been successfully integrated.")
        print("\nNext steps:")
        print(
            "1. Use scripts/extract_cdp_keys.py to extract keys from your CDP JSON file"
        )
        print("2. Add the extracted keys to your .env file")
        print("3. Remove or comment out legacy credentials if switching to CDP")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
