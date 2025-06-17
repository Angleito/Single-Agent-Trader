#!/usr/bin/env python3
"""Generate a secure API key for the Bluefin service."""

import secrets
import sys


def generate_api_key(length: int = 32) -> str:
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(length)


def main():
    """Generate and display API key with instructions."""
    api_key = generate_api_key()
    
    print("=" * 60)
    print("Bluefin Service API Key Generated")
    print("=" * 60)
    print(f"\nAPI Key: {api_key}")
    print("\nTo use this API key:")
    print("\n1. Add to your .env file:")
    print(f"   BLUEFIN_SERVICE_API_KEY={api_key}")
    print("\n2. Or set as environment variable:")
    print(f"   export BLUEFIN_SERVICE_API_KEY={api_key}")
    print("\n3. For Docker Compose, add to docker-compose.yml:")
    print("   environment:")
    print(f"     - BLUEFIN_SERVICE_API_KEY={api_key}")
    print("\n4. The client will automatically use this key from the environment")
    print("\nSecurity Notes:")
    print("- Keep this key secret and never commit it to version control")
    print("- Rotate the key regularly for production use")
    print("- Use different keys for different environments")
    print("=" * 60)


if __name__ == "__main__":
    main()