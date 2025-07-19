"""Configuration management utilities for backup and restore operations."""

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Environment, Settings, TradingProfile, create_settings
from .utils.path_utils import get_config_directory

logger = logging.getLogger(__name__)


class ConfigManager:
    """Enhanced configuration management utilities."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize configuration manager."""
        if config_dir:
            self.config_dir = config_dir
        else:
            try:
                self.config_dir = get_config_directory()
            except OSError:
                # Fallback to a secure temporary directory
                temp_config_dir = Path(tempfile.mkdtemp(prefix="config_"))
                logger.warning("Using temporary config directory: %s", temp_config_dir)
                self.config_dir = temp_config_dir

        self.config_dir.mkdir(exist_ok=True)
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def save_profile_config(self, profile: TradingProfile, settings: Settings) -> Path:
        """Save configuration for a specific trading profile."""
        filename = f"{profile.value}_config.json"
        file_path = self.config_dir / filename

        settings.save_to_file(file_path)
        logger.info("Saved %s configuration to %s", profile.value, file_path)

        return file_path

    def load_profile_config(self, profile: TradingProfile) -> Settings | None:
        """Load configuration for a specific trading profile."""
        filename = f"{profile.value}_config.json"
        file_path = self.config_dir / filename

        if not file_path.exists():
            logger.warning("Configuration file not found: %s", file_path)
            return None

        try:
            settings = Settings.load_from_file(file_path)
        except Exception:
            logger.exception("Failed to load configuration from %s", file_path)
            return None
        else:
            logger.info("Loaded %s configuration from %s", profile.value, file_path)
            return settings

    def create_environment_configs(self) -> dict[Environment, Path]:
        """Create configuration files for different environments."""
        configs = {}

        # Development configuration
        dev_overrides = {
            "system": {"environment": Environment.DEVELOPMENT, "dry_run": True, "log_level": "DEBUG"},
            "exchange": {"cb_sandbox": True},
            "trading": {"leverage": 2},
            "risk": {"max_daily_loss_pct": 2.0},
        }
        dev_settings = create_settings(overrides=dev_overrides)
        dev_path = self.config_dir / "development.json"
        dev_settings.save_to_file(dev_path)
        configs[Environment.DEVELOPMENT] = dev_path

        # Staging configuration
        staging_overrides = {
            "system": {"environment": Environment.STAGING, "dry_run": True, "log_level": "INFO"},
            "exchange": {"cb_sandbox": True},
            "trading": {"leverage": 5},
            "risk": {"max_daily_loss_pct": 3.0},
        }
        staging_settings = create_settings(overrides=staging_overrides)
        staging_path = self.config_dir / "staging.json"
        staging_settings.save_to_file(staging_path)
        configs[Environment.STAGING] = staging_path

        # Production configuration template
        prod_overrides = {
            "system": {"environment": Environment.PRODUCTION, "dry_run": False, "log_level": "INFO"},
            "exchange": {"cb_sandbox": False},
            "trading": {"leverage": 3},
            "risk": {"max_daily_loss_pct": 5.0},
        }
        prod_settings = create_settings(overrides=prod_overrides)
        prod_path = self.config_dir / "production.json"
        prod_settings.save_to_file(prod_path)
        configs[Environment.PRODUCTION] = prod_path

        logger.info("Created environment-specific configuration files")
        return configs

    def create_trading_profile_configs(self) -> dict[TradingProfile, Path]:
        """Create configuration files for different trading profiles."""
        configs = {}
        base_settings = create_settings()

        for profile in [
            TradingProfile.CONSERVATIVE,
            TradingProfile.BALANCED,
            TradingProfile.AGGRESSIVE,
        ]:
            profile_settings = base_settings.apply_profile(profile)
            file_path = self.save_profile_config(profile, profile_settings)
            configs[profile] = file_path

        logger.info("Created trading profile configuration files")
        return configs

    def create_config_backup(
        self, settings: Settings, backup_name: str | None = None
    ) -> Path:
        """Create a backup of the current configuration."""
        if not backup_name:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}.json"

        backup_path = self.backup_dir / backup_name
        settings.save_to_file(backup_path)
        logger.info("Configuration backup created: %s", backup_path)
        return backup_path

    def restore_config_backup(self, backup_name: str) -> Settings | None:
        """Restore configuration from a backup."""
        backup_path = self.backup_dir / backup_name
        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return None

        try:
            settings = Settings.load_from_file(backup_path)
        except Exception:
            logger.exception("Failed to restore backup %s", backup_path)
            return None
        else:
            logger.info("Configuration restored from backup: %s", backup_path)
            return settings

    def list_config_backups(self) -> list[dict[str, Any]]:
        """List all available configuration backups."""
        backups = []
        for backup_file in self.backup_dir.glob("*.json"):
            try:
                stat = backup_file.stat()
                backups.append(
                    {
                        "name": backup_file.name,
                        "path": str(backup_file),
                        "created": datetime.fromtimestamp(
                            stat.st_ctime, UTC
                        ).isoformat(),
                        "size": stat.st_size,
                    }
                )
            except Exception as e:
                logger.warning("Error reading backup file %s: %s", backup_file, e)

        return sorted(backups, key=lambda x: x["created"], reverse=True)

    def export_configuration(
        self, settings: Settings, export_format: str = "json"
    ) -> Path:
        """Export configuration in various formats."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        if export_format.lower() == "json":
            export_path = self.config_dir / f"export_{timestamp}.json"
            settings.save_to_file(export_path)

        elif export_format.lower() == "env":
            export_path = self.config_dir / f"export_{timestamp}.env"
            env_content = self._settings_to_env_format(settings)
            with export_path.open("w") as f:
                f.write(env_content)

        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        logger.info("Configuration exported to: %s", export_path)
        return export_path

    def import_configuration(
        self, import_path: Path, import_format: str = "json"
    ) -> Settings | None:
        """Import configuration from various formats."""
        if not import_path.exists():
            logger.error("Import file not found: %s", import_path)
            return None

        def _validate_import_format(fmt: str) -> None:
            """Validate import format."""
            if fmt.lower() != "json":
                raise ValueError(f"Unsupported import format: {fmt}")

        try:
            _validate_import_format(import_format)
            settings = Settings.load_from_file(import_path)
        except Exception:
            logger.exception("Failed to import configuration from %s", import_path)
            return None
        else:
            logger.info("Configuration imported from: %s", import_path)
            return settings

    def switch_profile(
        self, settings: Settings, new_profile: TradingProfile, save_current: bool = True
    ) -> Settings:
        """Switch trading profile with optional backup of current settings."""
        if save_current:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            backup_name = f"profile_{settings.profile.value}_backup_{timestamp}.json"
            self.create_config_backup(settings, backup_name)

        new_settings = settings.apply_profile(new_profile)
        logger.info(
            "Switched from %s to %s profile", settings.profile.value, new_profile.value
        )
        return new_settings

    def _settings_to_env_format(self, settings: Settings) -> str:
        """Convert settings to environment variable format."""
        lines = ["# Exported Trading Bot Configuration", ""]

        # This would be a comprehensive mapping of all settings to env vars
        # For brevity, showing key mappings
        env_mappings = {
            f"TRADING__SYMBOL={settings.trading.symbol}",
            f"TRADING__INTERVAL={settings.trading.interval}",
            f"TRADING__LEVERAGE={settings.trading.leverage}",
            f"LLM__PROVIDER={settings.llm.provider}",
            f"LLM__MODEL_NAME={settings.llm.model_name}",
            f"SYSTEM__DRY_RUN={str(settings.system.dry_run).lower()}",
            f"SYSTEM__ENVIRONMENT={settings.system.environment.value}",
            f"SYSTEM__LOG_LEVEL={settings.system.log_level}",
            f"EXCHANGE__EXCHANGE_TYPE={settings.exchange.exchange_type}",
            f"RISK__MAX_DAILY_LOSS_PCT={settings.risk.max_daily_loss_pct}",
            f"RISK__MAX_CONCURRENT_TRADES={settings.risk.max_concurrent_trades}",
            f"PROFILE={settings.profile.value}",
        }

        lines.extend(sorted(env_mappings))
        return "\n".join(lines)

    def generate_env_template(self) -> Path:
        """Generate a .env template file with all available options."""
        # This method is kept for backward compatibility
        # The actual template is now in .env.example
        template_path = Path(".env.template")
        logger.info(
            "Use .env.example as the template file. Legacy template: %s", template_path
        )
        return template_path


__all__ = [
    "ConfigManager",
]