#!/usr/bin/env python3
"""
Simple CDP API Key Configuration Test (No Dependencies)
"""

import os


def test_cdp_configuration():
    """Test CDP configuration directly from environment"""

    print("🔑 CDP API Key Configuration Test")
    print("=" * 50)

    # Read environment variables
    cdp_key_name = os.getenv("CDP_API_KEY_NAME")
    cdp_private_key = os.getenv("CDP_PRIVATE_KEY")

    # Legacy variables
    legacy_key = os.getenv("COINBASE_API_KEY")
    legacy_secret = os.getenv("COINBASE_API_SECRET")
    legacy_passphrase = os.getenv("COINBASE_PASSPHRASE")

    print("Environment Variables:")
    print(f"  CDP_API_KEY_NAME: {'Present' if cdp_key_name else 'Not set'}")
    print(f"  CDP_PRIVATE_KEY: {'Present' if cdp_private_key else 'Not set'}")
    print(f"  Legacy COINBASE_API_KEY: {'Present' if legacy_key else 'Not set'}")

    if cdp_key_name and cdp_private_key:
        print("\n✅ CDP AUTHENTICATION DETECTED!")
        print("\nCDP Key Analysis:")

        # Parse the key name
        if "organizations/" in cdp_key_name and "apiKeys/" in cdp_key_name:
            parts = cdp_key_name.split("/")
            org_id = parts[1] if len(parts) > 1 else "Unknown"
            key_id = parts[3] if len(parts) > 3 else "Unknown"

            print(f"  Organization ID: {org_id}")
            print(f"  API Key ID: {key_id}")
        else:
            print(f"  Raw Key Name: {cdp_key_name}")

        # Analyze private key
        if cdp_private_key.startswith('"-----BEGIN EC PRIVATE KEY-----'):
            print("  Private Key Format: EC Private Key (PEM)")
            print(f"  Private Key Length: {len(cdp_private_key)} characters")
            print("  Private Key Status: ✅ Valid PEM format")
        else:
            print("  Private Key Format: Unknown")
            print(f"  Private Key Preview: {cdp_private_key[:50]}...")

        print("\n🔐 Authentication Method: CDP (Modern)")
        print("📋 How CDP Works:")
        print("  1. Uses Elliptic Curve cryptographic keys")
        print("  2. Creates JWT tokens for each API request")
        print("  3. No shared secrets - private key stays local")
        print("  4. Time-limited tokens for enhanced security")

    elif legacy_key and legacy_secret and legacy_passphrase:
        print("\n⚠️  LEGACY AUTHENTICATION DETECTED")
        print("📋 Recommendation: Migrate to CDP keys for better security")

    else:
        print("\n❌ NO AUTHENTICATION CONFIGURED")
        print("📋 You need either:")
        print("  - CDP credentials: CDP_API_KEY_NAME + CDP_PRIVATE_KEY")
        print(
            "  - Legacy credentials: COINBASE_API_KEY + COINBASE_API_SECRET + COINBASE_PASSPHRASE"
        )

    # Test dry run configuration
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    print(
        f"\n💰 Trading Mode: {'DRY RUN (Paper Trading)' if dry_run else 'LIVE TRADING'}"
    )

    if not dry_run:
        print("⚠️  WARNING: Live trading is enabled!")
    else:
        print("✅ Safe mode: All trades will be simulated")

    return cdp_key_name and cdp_private_key


if __name__ == "__main__":
    result = test_cdp_configuration()

    print(f"\n{'='*50}")
    if result:
        print("🎉 CDP Configuration: READY FOR TRADING")
    else:
        print("⚠️  Configuration: NEEDS API KEYS")
    print("=" * 50)
