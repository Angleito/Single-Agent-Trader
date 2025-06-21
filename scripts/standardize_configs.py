#!/usr/bin/env python3
"""
Configuration Standardization Script

This script standardizes all configuration files to use consistent field names,
adds missing required sections, and ensures proper validation.
"""

import json
import logging
from pathlib import Path
from typing import Any

import click

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ConfigurationStandardizer:
    """Standardizes configuration files for consistency."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.config_dir = base_path / "config"
        self.changes_made = []

        # Define standardization rules
        self.field_standardization = {
            # LLM section standardization
            ("llm", "timeout"): ("llm", "request_timeout"),
            ("llm", "log_file"): ("llm", "completion_log_file"),
            # System section standardization
            ("system", "log_file"): ("system", "log_file_path"),
            # Exchange section standardization
            ("exchange", "timeout"): ("exchange", "api_timeout"),
        }

        # Required sections for all configs
        self.required_sections = {
            "trading": {},
            "llm": {},
            "exchange": {},
            "risk": {},
            "data": {},
            "system": {},
            "paper_trading": {},
            "mcp": {"enabled": False},
            "omnisearch": {"enabled": False},
            "monitoring": {"enabled": True},
        }

        # Standard field values
        self.standard_defaults = {
            "llm": {
                "request_timeout": 30,
                "max_retries": 3,
                "retry_delay": 1.0,
                "completion_log_level": "INFO",
            },
            "system": {
                "log_file_path": "logs/bot.log",
                "max_log_size_mb": 100,
                "log_retention_days": 30,
            },
            "exchange": {
                "api_timeout": 10,
                "rate_limit_requests": 10,
                "websocket_reconnect_attempts": 5,
            },
            "data": {
                "candle_limit": 250,
                "real_time_updates": True,
                "indicator_warmup": 100,
            },
            "mcp": {
                "enabled": False,
                "server_url": "http://localhost:8765",
                "memory_retention_days": 90,
                "track_trade_lifecycle": True,
            },
            "omnisearch": {"enabled": False, "server_url": "http://localhost:8766"},
            "monitoring": {
                "enabled": True,
                "health_check_interval": 300,
                "enable_performance_monitoring": True,
            },
        }

    def standardize_all_configs(self, dry_run: bool = False) -> dict[str, Any]:
        """Standardize all configuration files."""
        results = {"files_processed": [], "changes_made": [], "errors": []}

        # Process all JSON config files
        config_files = list(self.config_dir.glob("*.json"))
        config_files.extend(self.config_dir.glob("profiles/*.json"))

        for config_file in config_files:
            try:
                logger.info("Processing %s...", config_file.name)

                # Load config
                with open(config_file) as f:
                    config_data = json.load(f)

                # Apply standardizations
                original_data = json.dumps(config_data, sort_keys=True)
                config_data = self._standardize_config(config_data, config_file.name)
                modified_data = json.dumps(config_data, sort_keys=True)

                # Check if changes were made
                if original_data != modified_data:
                    if not dry_run:
                        # Write back to file with proper formatting
                        with open(config_file, "w") as f:
                            json.dump(config_data, f, indent=2, sort_keys=False)

                        logger.info("✅ Updated %s", config_file.name)
                    else:
                        logger.info("🔍 Would update %s", config_file.name)

                    results["changes_made"].extend(self.changes_made)
                    self.changes_made = []  # Reset for next file
                else:
                    logger.info("📋 No changes needed for %s", config_file.name)

                results["files_processed"].append(
                    str(config_file.relative_to(self.base_path))
                )

            except Exception as e:
                error_msg = f"Error processing {config_file.name}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        return results

    def _standardize_config(
        self, config: dict[str, Any], filename: str
    ) -> dict[str, Any]:
        """Apply standardization rules to a single config."""
        config = config.copy()  # Don't modify original

        # 1. Standardize field names
        config = self._standardize_field_names(config)

        # 2. Add missing required sections
        config = self._add_missing_sections(config, filename)

        # 3. Apply standard defaults where missing
        config = self._apply_standard_defaults(config)

        # 4. Validate and fix risk parameters
        config = self._validate_risk_parameters(config, filename)

        # 5. Ensure proper environment-specific settings
        config = self._apply_environment_specific_rules(config, filename)

        return config

    def _standardize_field_names(self, config: dict[str, Any]) -> dict[str, Any]:
        """Standardize field names according to rules."""
        for (section, old_field), (
            new_section,
            new_field,
        ) in self.field_standardization.items():
            if section in config and old_field in config[section]:
                if new_section not in config:
                    config[new_section] = {}

                # Move field to new location
                config[new_section][new_field] = config[section][old_field]
                del config[section][old_field]

                self.changes_made.append(
                    f"Renamed {section}.{old_field} to {new_section}.{new_field}"
                )

        return config

    def _add_missing_sections(
        self, config: dict[str, Any], filename: str
    ) -> dict[str, Any]:
        """Add missing required sections."""
        for section, defaults in self.required_sections.items():
            if section not in config:
                config[section] = defaults.copy()
                self.changes_made.append(f"Added missing section: {section}")

        return config

    def _apply_standard_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply standard default values where missing."""
        for section, defaults in self.standard_defaults.items():
            if section in config:
                for field, default_value in defaults.items():
                    if field not in config[section]:
                        config[section][field] = default_value
                        self.changes_made.append(
                            f"Added default {section}.{field} = {default_value}"
                        )

        return config

    def _validate_risk_parameters(
        self, config: dict[str, Any], filename: str
    ) -> dict[str, Any]:
        """Validate and fix dangerous risk parameters."""
        if "risk" not in config:
            return config

        risk_section = config["risk"]

        # Check for dangerous combinations
        leverage = config.get("trading", {}).get("leverage", 1)
        daily_loss = risk_section.get("max_daily_loss_pct", 0)

        # Apply safety limits based on config type
        if "production" in filename.lower() or "live" in filename.lower():
            # Production configs should be conservative
            if leverage > 5:
                risk_section["max_leverage_override"] = 5
                self.changes_made.append(
                    f"Limited production leverage to 5x (was {leverage}x)"
                )

            if daily_loss > 3:
                risk_section["max_daily_loss_pct"] = 3.0
                self.changes_made.append(
                    f"Limited production daily loss to 3% (was {daily_loss}%)"
                )

        elif "conservative" in filename.lower():
            # Conservative configs should be extra safe
            if leverage > 3:
                config.setdefault("trading", {})["leverage"] = 3
                self.changes_made.append(
                    f"Limited conservative leverage to 3x (was {leverage}x)"
                )

        # Ensure stop loss < take profit
        stop_loss = risk_section.get("default_stop_loss_pct", 0)
        take_profit = risk_section.get("default_take_profit_pct", 0)

        if stop_loss >= take_profit and stop_loss > 0:
            risk_section["default_take_profit_pct"] = stop_loss * 2
            self.changes_made.append("Fixed take profit to be 2x stop loss")

        return config

    def _apply_environment_specific_rules(
        self, config: dict[str, Any], filename: str
    ) -> dict[str, Any]:
        """Apply environment-specific configuration rules."""
        system_section = config.setdefault("system", {})

        # Set environment based on filename
        if "production" in filename.lower():
            system_section["environment"] = "production"
            system_section["dry_run"] = True  # Safety first
            system_section["log_level"] = "INFO"
        elif "development" in filename.lower():
            system_section["environment"] = "development"
            system_section["dry_run"] = True
            system_section["log_level"] = "DEBUG"
        elif "docker" in filename.lower():
            system_section["environment"] = "production"
            system_section["container_mode"] = True
            # Docker-specific paths
            if "data" in config:
                config["data"]["data_storage_path"] = "/app/data"
            system_section["log_file_path"] = "/app/logs/bot_docker.log"

        # Exchange-specific settings
        if "bluefin" in filename.lower():
            exchange_section = config.setdefault("exchange", {})
            exchange_section["exchange_type"] = "bluefin"

            # Ensure trading section is compatible
            trading_section = config.setdefault("trading", {})
            if "enable_futures" not in trading_section:
                trading_section["enable_futures"] = True
                self.changes_made.append("Enabled futures for Bluefin config")

        return config

    def create_master_schema(self) -> dict[str, Any]:
        """Create a master JSON schema for all configurations."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "AI Trading Bot Configuration",
            "type": "object",
            "properties": {
                "trading": {
                    "type": "object",
                    "required": ["symbol", "interval", "leverage"],
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "pattern": "^[A-Z]+-[A-Z]+$|^[A-Z]+-PERP$",
                        },
                        "interval": {
                            "type": "string",
                            "enum": ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        },
                        "leverage": {"type": "number", "minimum": 1, "maximum": 20},
                        "max_size_pct": {"type": "number", "minimum": 1, "maximum": 50},
                        "enable_futures": {"type": "boolean"},
                        "order_timeout_seconds": {
                            "type": "number",
                            "minimum": 10,
                            "maximum": 300,
                        },
                    },
                },
                "llm": {
                    "type": "object",
                    "required": ["provider", "model_name"],
                    "properties": {
                        "provider": {
                            "type": "string",
                            "enum": ["openai", "anthropic", "ollama"],
                        },
                        "model_name": {"type": "string"},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_tokens": {
                            "type": "number",
                            "minimum": 100,
                            "maximum": 50000,
                        },
                        "request_timeout": {
                            "type": "number",
                            "minimum": 10,
                            "maximum": 300,
                        },
                    },
                },
                "risk": {
                    "type": "object",
                    "required": ["max_daily_loss_pct", "default_stop_loss_pct"],
                    "properties": {
                        "max_daily_loss_pct": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 20,
                        },
                        "max_weekly_loss_pct": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "max_monthly_loss_pct": {
                            "type": "number",
                            "minimum": 5,
                            "maximum": 100,
                        },
                        "default_stop_loss_pct": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 10,
                        },
                        "default_take_profit_pct": {
                            "type": "number",
                            "minimum": 0.2,
                            "maximum": 20,
                        },
                    },
                },
            },
            "required": ["trading", "llm", "risk", "system"],
        }

        return schema


@click.command()
@click.option("--config-path", default=".", help="Path to configuration directory")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be changed without making changes"
)
@click.option("--create-schema", is_flag=True, help="Create master JSON schema file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(config_path: str, dry_run: bool, create_schema: bool, verbose: bool):
    """Standardize configuration files for consistency."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_path = Path(config_path).resolve()
    standardizer = ConfigurationStandardizer(base_path)

    if create_schema:
        click.echo("📋 Creating master JSON schema...")
        schema = standardizer.create_master_schema()
        schema_path = base_path / "config" / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        click.echo(f"✅ Schema created at: {schema_path}")
        return

    click.echo("🔧 Starting configuration standardization...")
    if dry_run:
        click.echo("🔍 DRY RUN MODE - No files will be modified")

    results = standardizer.standardize_all_configs(dry_run=dry_run)

    # Report results
    click.echo("\n📊 STANDARDIZATION RESULTS:")
    click.echo(f"Files processed: {len(results['files_processed'])}")
    click.echo(f"Changes made: {len(results['changes_made'])}")
    click.echo(f"Errors: {len(results['errors'])}")

    if results["changes_made"]:
        click.echo("\n🔧 CHANGES MADE:")
        for change in results["changes_made"]:
            click.echo(f"  • {change}")

    if results["errors"]:
        click.echo("\n❌ ERRORS:")
        for error in results["errors"]:
            click.echo(f"  • {error}")

    if not dry_run and results["changes_made"]:
        click.echo("\n✅ Configuration standardization complete!")
    elif dry_run and results["changes_made"]:
        click.echo("\n🔍 Run without --dry-run to apply these changes")
    else:
        click.echo("\n✅ All configurations are already standardized!")


if __name__ == "__main__":
    main()
