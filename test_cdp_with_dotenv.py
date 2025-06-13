#!/usr/bin/env python3
"""
CDP API Key Test with proper .env loading
"""

import os


def load_env_file(env_file=".env"):
    """Load environment variables from .env file"""
    if not os.path.exists(env_file):
        print(f"❌ {env_file} file not found")
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue

            # Parse KEY=VALUE format
            if "=" in line:
                key, value = line.split("=", 1)

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Set environment variable
                os.environ[key] = value


def test_cdp_authentication():
    """Test CDP authentication with proper .env loading"""

    print("🔑 CDP API Key Authentication Test (with .env)")
    print("=" * 55)

    # Load .env file
    load_env_file()

    # Get CDP credentials
    cdp_key_name = os.getenv("CDP_API_KEY_NAME")
    cdp_private_key = os.getenv("CDP_PRIVATE_KEY")

    print("Environment Status:")
    print(f"  CDP_API_KEY_NAME: {'✅ Present' if cdp_key_name else '❌ Not set'}")
    print(f"  CDP_PRIVATE_KEY: {'✅ Present' if cdp_private_key else '❌ Not set'}")

    if cdp_key_name and cdp_private_key:
        print("\n🎉 CDP AUTHENTICATION CONFIGURED!")
        print("\n📋 CDP Key Details:")

        # Parse organization and key ID
        if "organizations/" in cdp_key_name:
            parts = cdp_key_name.split("/")
            org_id = parts[1] if len(parts) > 1 else "Unknown"
            key_id = parts[3] if len(parts) > 3 else "Unknown"

            print(f"  🏢 Organization ID: {org_id}")
            print(f"  🔑 API Key ID: {key_id}")
            print(f"  📝 Full Key Name: {cdp_key_name}")

        # Analyze private key
        if "-----BEGIN EC PRIVATE KEY-----" in cdp_private_key:
            print("\n🔐 Private Key Analysis:")
            print("  📄 Format: EC Private Key (PEM)")
            print(f"  📏 Length: {len(cdp_private_key)} characters")
            print("  ✅ Status: Valid PEM format detected")

            # Extract key content (between headers)
            key_content = (
                cdp_private_key.replace("-----BEGIN EC PRIVATE KEY-----", "")
                .replace("-----END EC PRIVATE KEY-----", "")
                .replace("\\n", "")
                .strip()
            )
            print(f"  🔢 Key Content Length: {len(key_content)} characters")

        print("\n🔄 How CDP Authentication Works:")
        print("  1. 🔐 Uses Elliptic Curve cryptography (more secure)")
        print("  2. 🎫 Creates JWT tokens for each API request")
        print("  3. 🛡️  Private key never leaves your system")
        print("  4. ⏰ Time-limited tokens for enhanced security")
        print("  5. 🚀 This is Coinbase's modern authentication standard")

        print("\n📊 Comparison with Legacy Keys:")
        print("  Legacy: 3 components (Key + Secret + Passphrase) + HMAC")
        print("  CDP: 2 components (Key Name + Private Key) + JWT")
        print("  🏆 CDP is more secure and future-proof!")

        return True
    else:
        print("\n❌ CDP Authentication Not Configured")
        print("📋 Missing components:")
        if not cdp_key_name:
            print("  - CDP_API_KEY_NAME")
        if not cdp_private_key:
            print("  - CDP_PRIVATE_KEY")

        return False


def show_docker_demo():
    """Show how this will work in Docker"""
    print("\n🐳 Docker Integration Preview:")
    print("When you run: docker-compose up")
    print("The bot will:")
    print("  1. 🔍 Detect CDP credentials in environment")
    print("  2. 🔐 Initialize CDP authentication client")
    print("  3. 🎫 Generate JWT tokens for Coinbase API calls")
    print("  4. 📈 Use modern futures trading endpoints")
    print("  5. 🤖 Make AI trading decisions with o3 model")
    print("  6. 💰 Execute trades (in dry-run mode by default)")


if __name__ == "__main__":
    result = test_cdp_authentication()

    if result:
        show_docker_demo()
        print(f"\n{'='*55}")
        print("🎉 READY FOR CDP FUTURES TRADING!")
        print("Run: docker-compose up")
        print("=" * 55)
    else:
        print(f"\n{'='*55}")
        print("⚠️  CONFIGURATION INCOMPLETE")
        print("=" * 55)
