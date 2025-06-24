"""
Bluefin WebSocket Authentication System.

This module provides JWT token generation and authentication functionality
for Bluefin's private WebSocket channels using ED25519 signatures.
"""

import base64
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from .bluefin_order_signature import BluefinOrderSigner

logger = logging.getLogger(__name__)


class BluefinWebSocketAuthError(Exception):
    """Exception raised when WebSocket authentication fails."""


class BluefinJWTGenerator:
    """
    Generates JWT tokens for Bluefin WebSocket authentication using ED25519 signatures.

    Bluefin requires JWT tokens with specific claims and ED25519 signatures
    for accessing private WebSocket channels (userUpdates room).
    """

    def __init__(self, private_key_hex: str):
        """
        Initialize JWT generator with ED25519 private key.

        Args:
            private_key_hex: Hexadecimal private key string (64 characters)
        """
        self.private_key_hex = private_key_hex
        self.order_signer = BluefinOrderSigner(private_key_hex)
        self._private_key = self.order_signer._private_key
        self.public_key_hex = self.order_signer.get_public_key_hex()

        logger.info(
            "Initialized Bluefin JWT generator with public key: %s",
            self.public_key_hex[:16] + "...",
        )

    def generate_auth_token(
        self,
        user_id: str | None = None,
        expires_in_seconds: int = 3600,
        custom_claims: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a JWT token for WebSocket authentication.

        Args:
            user_id: User identifier (defaults to public key)
            expires_in_seconds: Token expiration time in seconds (default 1 hour)
            custom_claims: Additional claims to include in the token

        Returns:
            JWT token string for WebSocket authentication
        """
        try:
            # Use public key as user_id if not provided
            if user_id is None:
                user_id = self.public_key_hex

            # Current time
            now = int(time.time())

            # JWT Header
            header = {
                "alg": "EdDSA",  # Algorithm for ED25519
                "typ": "JWT",
                "kid": self.public_key_hex[:16],  # Key ID based on public key
            }

            # JWT Payload with Bluefin-specific claims
            payload = {
                "iss": "bluefin-client",  # Issuer
                "sub": user_id,  # Subject (user identifier)
                "aud": "bluefin-websocket",  # Audience
                "iat": now,  # Issued at
                "exp": now + expires_in_seconds,  # Expiration
                "nbf": now - 60,  # Not before (60 seconds ago for clock skew)
                "jti": f"{user_id}_{now}_{int(time.time() * 1000000) % 1000000}",  # JWT ID
                # Bluefin-specific claims
                "user_id": user_id,
                "public_key": self.public_key_hex,
                "timestamp": now * 1000,  # Timestamp in milliseconds
                "nonce": self.order_signer.nonce_manager.generate_nonce(),
                "scope": "websocket:userUpdates",  # Permission scope
            }

            # Add custom claims if provided
            if custom_claims:
                payload.update(custom_claims)

            # Encode header and payload
            header_b64 = self._base64url_encode(
                json.dumps(header, separators=(",", ":"))
            )
            payload_b64 = self._base64url_encode(
                json.dumps(payload, separators=(",", ":"))
            )

            # Create signing input
            signing_input = f"{header_b64}.{payload_b64}"

            # Sign with ED25519
            signature_bytes = self._private_key.sign(signing_input.encode("utf-8"))
            signature_b64 = self._base64url_encode(signature_bytes)

            # Construct final JWT
            jwt_token = f"{signing_input}.{signature_b64}"

            logger.debug(
                "Generated JWT token for user %s, expires at %s",
                user_id[:16] + "..." if len(user_id) > 16 else user_id,
                datetime.fromtimestamp(payload["exp"], UTC).isoformat(),
            )

            return jwt_token

        except Exception as e:
            raise BluefinWebSocketAuthError(f"Failed to generate JWT token: {e}") from e

    def generate_user_updates_token(self, expires_in_seconds: int = 3600) -> str:
        """
        Generate a token specifically for userUpdates WebSocket channel.

        Args:
            expires_in_seconds: Token expiration time in seconds

        Returns:
            JWT token for userUpdates subscription
        """
        return self.generate_auth_token(
            custom_claims={
                "channels": ["userUpdates"],
                "permissions": [
                    "account:read",
                    "positions:read",
                    "orders:read",
                    "trades:read",
                ],
            },
            expires_in_seconds=expires_in_seconds,
        )

    def generate_read_only_token(self, expires_in_seconds: int = 7200) -> str:
        """
        Generate a read-only token for account data access.

        Args:
            expires_in_seconds: Token expiration time in seconds (default 2 hours)

        Returns:
            JWT token for read-only access
        """
        return self.generate_auth_token(
            custom_claims={"permissions": ["read_only"], "access_level": "read"},
            expires_in_seconds=expires_in_seconds,
        )

    def verify_token(self, jwt_token: str) -> dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            jwt_token: JWT token to verify

        Returns:
            Decoded payload if valid

        Raises:
            BluefinWebSocketAuthError: If token is invalid
        """
        try:
            # Split token into parts
            parts = jwt_token.split(".")
            if len(parts) != 3:
                raise BluefinWebSocketAuthError("Invalid JWT format")

            header_b64, payload_b64, signature_b64 = parts

            # Decode and verify signature
            signing_input = f"{header_b64}.{payload_b64}"
            signature_bytes = self._base64url_decode(signature_b64)

            # Verify signature
            self._private_key.public_key().verify(
                signature_bytes, signing_input.encode("utf-8")
            )

            # Decode payload
            payload_json = self._base64url_decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)

            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise BluefinWebSocketAuthError("Token has expired")

            # Check not-before
            if payload.get("nbf", 0) > time.time():
                raise BluefinWebSocketAuthError("Token not yet valid")

            logger.debug("Successfully verified JWT token")
            return payload

        except BluefinWebSocketAuthError:
            raise
        except Exception as e:
            raise BluefinWebSocketAuthError(f"Token verification failed: {e}") from e

    def is_token_expired(self, jwt_token: str, buffer_seconds: int = 300) -> bool:
        """
        Check if a token is expired or will expire soon.

        Args:
            jwt_token: JWT token to check
            buffer_seconds: Consider token expired if it expires within this time

        Returns:
            True if token is expired or will expire soon
        """
        try:
            payload = self.verify_token(jwt_token)
            exp_time = payload.get("exp", 0)
            return exp_time < time.time() + buffer_seconds
        except BluefinWebSocketAuthError:
            return True  # Invalid tokens are considered expired

    def _base64url_encode(self, data: bytes | str) -> str:
        """Base64URL encode data."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Base64 encode and make URL-safe
        encoded = base64.urlsafe_b64encode(data).decode("utf-8")
        # Remove padding
        return encoded.rstrip("=")

    def _base64url_decode(self, data: str) -> bytes:
        """Base64URL decode data."""
        # Add padding if necessary
        missing_padding = len(data) % 4
        if missing_padding:
            data += "=" * (4 - missing_padding)

        return base64.urlsafe_b64decode(data)


class BluefinWebSocketAuthenticator:
    """
    Manages authentication state and token refresh for Bluefin WebSocket connections.

    Handles automatic token generation, refresh, and provides authentication
    tokens for WebSocket subscriptions.
    """

    def __init__(self, private_key_hex: str, auto_refresh: bool = True):
        """
        Initialize authenticator with private key.

        Args:
            private_key_hex: ED25519 private key in hex format
            auto_refresh: Automatically refresh tokens before expiration
        """
        self.jwt_generator = BluefinJWTGenerator(private_key_hex)
        self.auto_refresh = auto_refresh

        # Token storage
        self._current_token: str | None = None
        self._token_expires_at: datetime | None = None
        self._refresh_threshold_seconds = 300  # Refresh 5 minutes before expiration

        logger.info("Bluefin WebSocket authenticator initialized")

    def get_auth_token(self, force_refresh: bool = False) -> str:
        """
        Get current authentication token, refreshing if necessary.

        Args:
            force_refresh: Force generate a new token

        Returns:
            Valid JWT authentication token
        """
        # Check if we need to generate or refresh token
        if force_refresh or self._current_token is None or self._should_refresh_token():
            self._generate_new_token()

        return self._current_token

    def get_user_updates_subscription_message(self) -> dict[str, Any]:
        """
        Get WebSocket subscription message for userUpdates channel.

        Returns:
            Subscription message with authentication token
        """
        auth_token = self.get_auth_token()

        return {"e": "userUpdates", "t": auth_token}

    def get_subscription_with_auth(self, channel: str, **params) -> dict[str, Any]:
        """
        Get authenticated subscription message for any channel.

        Args:
            channel: Channel name to subscribe to
            **params: Additional subscription parameters

        Returns:
            Subscription message with authentication
        """
        auth_token = self.get_auth_token()

        subscription = {"e": channel, "t": auth_token, **params}

        return subscription

    def refresh_token(self) -> str:
        """
        Force refresh the authentication token.

        Returns:
            New JWT token
        """
        return self.get_auth_token(force_refresh=True)

    def is_authenticated(self) -> bool:
        """
        Check if we have a valid authentication token.

        Returns:
            True if authenticated with valid token
        """
        if self._current_token is None:
            return False

        try:
            self.jwt_generator.verify_token(self._current_token)
            return True
        except BluefinWebSocketAuthError:
            return False

    def get_public_key(self) -> str:
        """Get the public key for this authenticator."""
        return self.jwt_generator.public_key_hex

    def _generate_new_token(self) -> None:
        """Generate a new authentication token."""
        try:
            # Generate token valid for 1 hour
            self._current_token = self.jwt_generator.generate_user_updates_token(
                expires_in_seconds=3600
            )

            # Set expiration time
            self._token_expires_at = datetime.now(UTC) + timedelta(seconds=3600)

            logger.info(
                "Generated new WebSocket auth token, expires at %s",
                self._token_expires_at.isoformat(),
            )

        except Exception as e:
            logger.error("Failed to generate new auth token: %s", e)
            raise BluefinWebSocketAuthError(f"Token generation failed: {e}") from e

    def _should_refresh_token(self) -> bool:
        """Check if token should be refreshed."""
        if not self.auto_refresh or self._token_expires_at is None:
            return False

        # Refresh if token expires within threshold
        time_until_expiration = (
            self._token_expires_at - datetime.now(UTC)
        ).total_seconds()
        return time_until_expiration <= self._refresh_threshold_seconds

    def get_status(self) -> dict[str, Any]:
        """
        Get authentication status information.

        Returns:
            Status dictionary with auth information
        """
        return {
            "authenticated": self.is_authenticated(),
            "has_token": self._current_token is not None,
            "token_expires_at": (
                self._token_expires_at.isoformat() if self._token_expires_at else None
            ),
            "token_expires_in_seconds": (
                (self._token_expires_at - datetime.now(UTC)).total_seconds()
                if self._token_expires_at
                else None
            ),
            "auto_refresh": self.auto_refresh,
            "refresh_threshold_seconds": self._refresh_threshold_seconds,
            "public_key": self.jwt_generator.public_key_hex,
        }


# Convenience functions for easy integration


def create_websocket_authenticator(
    private_key_hex: str,
) -> BluefinWebSocketAuthenticator:
    """
    Create a WebSocket authenticator instance.

    Args:
        private_key_hex: ED25519 private key in hex format

    Returns:
        Configured authenticator instance
    """
    return BluefinWebSocketAuthenticator(private_key_hex)


def generate_auth_token(private_key_hex: str, expires_in_seconds: int = 3600) -> str:
    """
    Generate an authentication token for WebSocket access.

    Args:
        private_key_hex: ED25519 private key in hex format
        expires_in_seconds: Token expiration time

    Returns:
        JWT authentication token
    """
    generator = BluefinJWTGenerator(private_key_hex)
    return generator.generate_user_updates_token(expires_in_seconds)


def create_user_updates_subscription(private_key_hex: str) -> dict[str, Any]:
    """
    Create a userUpdates subscription message with authentication.

    Args:
        private_key_hex: ED25519 private key in hex format

    Returns:
        Subscription message for userUpdates channel
    """
    authenticator = BluefinWebSocketAuthenticator(private_key_hex)
    return authenticator.get_user_updates_subscription_message()
