"""Docker secrets loader for secure credential management."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from .memory import SecureString

logger = logging.getLogger(__name__)


class SecretsLoader:
    """Load secrets from Docker secrets or environment variables."""

    DOCKER_SECRETS_PATH = Path("/run/secrets")

    @classmethod
    def is_docker_secrets_enabled(cls) -> bool:
        """Check if Docker secrets are enabled."""
        return (
            os.environ.get("SECRETS_ENABLED", "false").lower() == "true"
            and cls.DOCKER_SECRETS_PATH.exists()
        )

    @classmethod
    @lru_cache(maxsize=32)
    def load_secret(
        cls, secret_name: str, env_fallback: str | None = None
    ) -> SecureString | None:
        """Load a secret from Docker secrets or environment variable.

        Args:
            secret_name: Name of the secret file (without path)
            env_fallback: Environment variable name to fall back to

        Returns:
            SecureString containing the secret, or None if not found
        """
        # First, check if we should use Docker secrets
        if cls.is_docker_secrets_enabled():
            secret_path = cls.DOCKER_SECRETS_PATH / secret_name
            if secret_path.exists() and secret_path.is_file():
                try:
                    with open(secret_path) as f:
                        value = f.read().strip()
                    if value:
                        logger.debug(f"Loaded secret from Docker: {secret_name}")
                        return SecureString(value)
                except Exception as e:
                    logger.error(f"Failed to read Docker secret {secret_name}: {e}")

        # Fall back to environment variable
        if env_fallback:
            # Check for _FILE suffix first (Docker secrets convention)
            file_env = f"{env_fallback}_FILE"
            if file_env in os.environ:
                file_path = Path(os.environ[file_env])
                if file_path.exists() and file_path.is_file():
                    try:
                        with open(file_path) as f:
                            value = f.read().strip()
                        if value:
                            logger.debug(f"Loaded secret from file: {file_path}")
                            return SecureString(value)
                    except Exception as e:
                        logger.error(f"Failed to read secret file {file_path}: {e}")

            # Finally, check direct environment variable
            if env_fallback in os.environ:
                value = os.environ[env_fallback]
                if value:
                    logger.debug(f"Loaded secret from environment: {env_fallback}")
                    return SecureString(value)

        return None

    @classmethod
    def load_all_secrets(cls) -> dict[str, SecureString]:
        """Load all known secrets.

        Returns:
            Dictionary mapping secret names to SecureString objects
        """
        # Define secret mappings (Docker secret name -> env variable name)
        secret_mappings = {
            "openai_api_key": "LLM__OPENAI_API_KEY",
            "coinbase_api_key": "EXCHANGE__CDP_API_KEY_NAME",
            "coinbase_private_key": "EXCHANGE__CDP_PRIVATE_KEY",
            "bluefin_private_key": "EXCHANGE__BLUEFIN_PRIVATE_KEY",
            "database_password": "DATABASE_PASSWORD",
            "jwt_secret": "JWT_SECRET",
        }

        secrets = {}
        for secret_name, env_name in secret_mappings.items():
            secret = cls.load_secret(secret_name, env_name)
            if secret:
                secrets[secret_name] = secret

        return secrets

    @classmethod
    def get_openai_api_key(cls) -> SecureString | None:
        """Get OpenAI API key from secrets or environment."""
        return cls.load_secret("openai_api_key", "LLM__OPENAI_API_KEY")

    @classmethod
    def get_coinbase_api_key(cls) -> SecureString | None:
        """Get Coinbase API key from secrets or environment."""
        return cls.load_secret("coinbase_api_key", "EXCHANGE__CDP_API_KEY_NAME")

    @classmethod
    def get_coinbase_private_key(cls) -> SecureString | None:
        """Get Coinbase private key from secrets or environment."""
        return cls.load_secret("coinbase_private_key", "EXCHANGE__CDP_PRIVATE_KEY")

    @classmethod
    def get_bluefin_private_key(cls) -> SecureString | None:
        """Get Bluefin private key from secrets or environment."""
        return cls.load_secret("bluefin_private_key", "EXCHANGE__BLUEFIN_PRIVATE_KEY")

    @classmethod
    def clear_cache(cls):
        """Clear the secrets cache."""
        cls.load_secret.cache_clear()


class SecretsMixin:
    """Mixin class to add secrets loading capability to other classes."""

    _secrets_loader = SecretsLoader

    def load_secret(
        self, secret_name: str, env_fallback: str | None = None
    ) -> SecureString | None:
        """Load a secret using the SecretsLoader."""
        return self._secrets_loader.load_secret(secret_name, env_fallback)

    def get_secret_value(
        self, secret_name: str, env_fallback: str | None = None
    ) -> str | None:
        """Load a secret and return its value (use with caution)."""
        secret = self.load_secret(secret_name, env_fallback)
        return secret.get_value() if secret else None
